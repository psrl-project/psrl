// Package dockerbackend runs sandboxes as containers on one node's daemon.
//
// It speaks the Engine API over the node's Unix socket rather than through a
// client library, for two reasons. The surface it needs is small -- create,
// start, inspect, remove, pause, commit -- and a vendored client would pull a
// dependency tree larger than the whole service for those six calls.
//
// The agent endpoint it reports is the daemon itself: a container has no process
// listening for commands, so this backend is the one case where the service's
// own node agent proxies command traffic. Every provider backend reports an
// in-sandbox agent instead and is not proxied.
package dockerbackend

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// The label every sandbox carries, so a sweep can find what this service owns
// without a local record. A container is the source of truth; a database that
// disagreed with the daemon would be worse than no database.
const (
	labelSandbox = "psrl.sandbox"
	labelOwner   = "psrl.owner"
	labelClass   = "psrl.resource_class"
	labelNode    = "psrl.node"
	// These carry what a restart cannot otherwise recover: how the sandbox is
	// executed, how long it may live, and which create produced it.
	labelExecMode    = "psrl.exec_mode"
	labelLifetime    = "psrl.lifetime_s"
	labelIdempotency = "psrl.idempotency_key"
	// Disk is charged by the node but not enforced by the Engine, so unlike
	// memory and CPU it cannot be read back from the container's limits.
	labelDisk = "psrl.disk_mb"
)

// DefaultPullTimeout is the image pull budget a node uses when its configuration
// states none. It is exported because a control plane that bounds a create over
// the node plane has to allow for the same pull before it gives up.
const DefaultPullTimeout = 10 * time.Minute

// Config is one node's Docker settings.
type Config struct {
	// Socket is the daemon's Unix socket.
	Socket string
	// APIVersion is pinned, because a daemon older than the service must fail
	// loudly at startup rather than on an endpoint that silently did not exist.
	APIVersion string
	NodeID     string
	OwnerID    string
	// Runtime selects a stronger isolation boundary when the node provides one.
	Runtime string
	// CgroupParent is the cgroup subtree every sandbox is created in, relative to the
	// daemon's own cgroup root. It is what makes the measured memory domain describe
	// sandboxes rather than everything the daemon's host happens to be doing, so an
	// empty value means the sampler has nothing narrower than the root to read.
	//
	// The cgroupfs driver places a container at <parent>/<id>; the systemd driver
	// treats the value as a slice name instead, which is why preflight refuses a
	// daemon that would interpret it differently.
	CgroupParent string
	// NetworkMode is the Docker network mode for new containers.
	// "host" removes per-container veth/netns creation and iptables rule churn,
	// which are the primary kernel-lock bottlenecks under concurrent creates.
	// Leave empty for the daemon default (bridge).
	NetworkMode string
	// MaxCreateConcurrency caps the number of concurrent /containers/create +
	// /start pairs in-flight to dockerd. It is the `max_create_concurrency`
	// deployment switch, and zero means unbounded.
	//
	// A daemon serializes parts of container setup in the kernel, so an unbounded
	// burst turns latency into timeouts rather than throughput. Size it from the
	// daemon rather than from the machine: a many-core node reports a GOMAXPROCS
	// far above what the create path can schedule at once, so leaving this unset on
	// a large node is what produced the create and exec timeouts it exists to stop.
	MaxCreateConcurrency int
	// RequestTimeout bounds one Engine call. A create that pulls an image can
	// legitimately exceed it, so the pull is a separate call with its own budget.
	RequestTimeout time.Duration
	PullTimeout    time.Duration
	// WarmPool, when enabled, keeps pre-created containers ready so a create
	// that matches the pool's spec claims one instantly rather than paying the
	// container-build cost. Zero-value Size disables the pool.
	WarmPool WarmPoolConfig
	// MountAllowlist is the set of host path prefixes a sandbox may be given. Empty
	// means none, and a spec that asks for a mount is then refused rather than
	// created without it.
	//
	// A host mount is the one request in the contract that is about the machine rather
	// than the sandbox: it hands a sandbox a view of a path the node owns, and a node
	// that honoured any path would let any caller read or overwrite anything that user
	// can reach -- including the daemon's own socket. So the node states which trees it
	// is willing to share, and everything else is refused at create.
	MountAllowlist []string
}

// Backend is the Docker runtime for one node.
type Backend struct {
	cfg  Config
	http *http.Client
	mode backend.SchedulingMode

	// createSem limits concurrent /containers/create + /start pairs in-flight to
	// dockerd. At high concurrency dockerd's internal worker pool saturates and all
	// requests queue on the same kernel locks (cgroup, netns, iptables), inflating
	// p95 far beyond p50. A bounded semaphore keeps the in-flight count at a level
	// dockerd can schedule without queueing, which flattens the latency distribution.
	// nil means unlimited (MaxCreateConcurrency == 0).
	createSem chan struct{}

	// shells holds the persistent shell sessions, one per sandbox. A sandbox
	// created with ExecMode "persistent" keeps a login shell open between calls;
	// one with ExecMode "one_shot" (or empty) creates a fresh process per call.
	shells *shellSessions

	// pool holds pre-created containers that are claimed on Create when the spec
	// matches. nil when the pool is disabled or not configured.
	pool *warmPool

	mu    sync.Mutex
	known map[string]string // sandbox id -> container id
}

// New returns a Docker backend bound to one node's daemon.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if cfg.Socket == "" {
		cfg.Socket = "/var/run/docker.sock"
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = "v1.40"
	}
	if !strings.HasPrefix(cfg.APIVersion, "v") {
		cfg.APIVersion = "v" + cfg.APIVersion
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 30 * time.Second
	}
	if cfg.PullTimeout <= 0 {
		cfg.PullTimeout = DefaultPullTimeout
	}
	if !mode.Valid() {
		return nil, fmt.Errorf("docker backend scheduling mode %q is not direct or provider", mode)
	}
	if cfg.MaxCreateConcurrency < 0 {
		return nil, fmt.Errorf(
			"docker backend max create concurrency %d is negative, and zero is what means unbounded",
			cfg.MaxCreateConcurrency)
	}
	var createSem chan struct{}
	if cfg.MaxCreateConcurrency > 0 {
		createSem = make(chan struct{}, cfg.MaxCreateConcurrency)
	}
	socket := cfg.Socket
	b := &Backend{
		cfg:       cfg,
		mode:      mode,
		createSem: createSem,
		http: &http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", socket)
				},
				MaxIdleConns:        64,
				MaxIdleConnsPerHost: 64,
			},
		},
		known:  map[string]string{},
		shells: newShellSessions(),
	}
	// Built after the backend exists, because the pool drives this backend's own
	// create and remove: a pool with its own transport would be a second way to
	// reach the daemon and could disagree with this one about concurrency.
	pool, err := newWarmPool(cfg.WarmPool,
		b.createPoolEntry,
		b.remove,
		nil, // installed by SetPoolBudget once the admission gate exists
	)
	if err != nil {
		return nil, fmt.Errorf("docker warm pool: %w", err)
	}
	b.pool = pool
	return b, nil
}

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "docker" }

// Mode says who places sandboxes for this backend.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what this node can do, with each conditional feature
// earned rather than assumed.
//
// A capability is declared only where the node can honour it: a stronger
// isolation boundary needs a configured runtime, and claiming one the node lacks
// would admit a spec and then run it with less isolation than it asked for.
func (b *Backend) Capabilities() backend.Capabilities {
	// host_mount is earned: a node with no allowlist shares nothing, and claiming the
	// feature there would let a caller ask for a mount it can only be refused.
	features := []string{"freeze", "filesystem_snapshot", "restore", "credential_injection"}
	if len(b.cfg.MountAllowlist) > 0 {
		features = append(features, "host_mount")
	}
	if b.cfg.Runtime != "" {
		features = append(features, "isolation_runtime")
	}
	sort.Strings(features)
	// No resume level: a commit captures the writable layer and not memory, so a
	// restore elsewhere needs a published snapshot, which this backend does not do
	// on its own. Claiming full state would let a conformance run read a workspace
	// restore as proof a harness survived a move.
	return backend.Capabilities{Features: features, PauseModes: []string{"freeze"}}
}

// Nodes reports this backend's single node. A Docker daemon is per machine, so
// one backend instance is one node.
func (b *Backend) Nodes(context.Context) ([]string, error) {
	return []string{b.cfg.NodeID}, nil
}

// Headroom is reported by the node's own admission rather than guessed here, so
// this returns nothing and lets the node agent fill it in. Returning a daemon
// statistic instead would be a second, disagreeing account of the same envelope.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// RegisterBinding is a no-op: a container carries its node in a label, so there
// is no routing table to keep in step.
func (b *Backend) RegisterBinding(context.Context, string, string) error { return nil }

// Create starts one container and waits until it can run a command.
//
// When the warm pool holds a matching entry the container is claimed rather
// than built, which removes the per-container namespace and cgroup overhead
// from the create latency entirely. A spec that does not match the pool spec
// (different image, larger footprint, GPU request) falls through to a cold
// create so the pool never silently degrades a caller's resource constraints.
func (b *Backend) Create(ctx context.Context, nodeID string, spec backend.Spec, callback string) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return backend.Created{}, fmt.Errorf("docker runs an image, so source kind %q cannot be served", spec.Source.Kind)
	}
	// Try the warm pool first. A cache hit skips image pull, container creation,
	// and the start call — the three steps whose latency scales with concurrency.
	// A claim that cannot be verified falls through rather than failing the create:
	// the caller is already admitted, and a cold create is what it would have had.
	if entry, ok := b.pool.claim(ctx, spec); ok {
		if created, adopted := b.adoptWarm(ctx, entry, spec, nodeID); adopted {
			return created, nil
		}
	}
	// Mounts are resolved before anything is created: a mount this node will not make is
	// a refusal, and refusing after the image pull would waste the pull and leave the
	// caller with a slower failure and the same answer.
	binds, err := b.binds(spec)
	if err != nil {
		return backend.Created{}, err
	}
	if err := b.ensureImage(ctx, spec.Source.Reference); err != nil {
		return backend.Created{}, err
	}
	release, err := b.acquireCreateSlot(ctx)
	if err != nil {
		return backend.Created{}, err
	}
	defer release()

	body := b.createBody(spec, nodeID, binds)
	created, err := b.createNamed(ctx, sandboxNamePrefix, body)
	if err != nil {
		return backend.Created{}, fmt.Errorf("docker create: %w", err)
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+created.ID+"/start", nil, nil); err != nil {
		// The container exists and holds resources, so it is removed here rather
		// than left for a sweep: the caller never saw a handle it could release.
		_ = b.remove(ctx, created.ID)
		return backend.Created{}, fmt.Errorf("docker start: %w", err)
	}
	b.mu.Lock()
	b.known[created.ID] = created.ID
	b.mu.Unlock()
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: created.ID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		// A container has no in-sandbox agent, so the node agent proxies commands
		// for it. Every provider backend reports a real address here instead.
		Agent: backend.AgentEndpoint{Address: ""},
	}, nil
}

// The prefixes every container this service creates carries.
//
// A warm-pool entry has to be distinguishable from a claimed sandbox, and the marker
// has to be changeable when the entry is adopted: the label set is written at create
// and immutable afterwards, so a label that said "inventory" would still say it about a
// live sandbox. That is the state that made a restarting node refuse to serve, and the
// state that would make it destroy a live sandbox if only the refusal were removed. A
// name is the one attribute the daemon lets us change after creation, so inventory is
// marked here and cleared by a rename at claim.
const (
	warmNamePrefix    = "psrl-warm-"
	sandboxNamePrefix = "psrl-sbx-"
)

// createdContainer is the reply to a create.
type createdContainer struct {
	ID       string   `json:"Id"`
	Warnings []string `json:"Warnings"`
}

// newContainerName returns a prefixed name with a random suffix.
func newContainerName(prefix string) string {
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		// crypto/rand does not fail on Linux in practice. A timestamp keeps the
		// function total, and a duplicate is retried by the caller as a conflict.
		return prefix + strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return prefix + hex.EncodeToString(suffix[:])
}

// createNamed creates a container under a name this service chose.
//
// The name is not cosmetic: it is how pool inventory is told from a claimed sandbox,
// and how an operator finds the container behind a sandbox id. A conflict means the
// random suffix collided, which is cheap to retry and must not surface as a create
// failure.
func (b *Backend) createNamed(ctx context.Context, prefix string, body map[string]any) (createdContainer, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		var created createdContainer
		err := b.call(ctx, http.MethodPost,
			"/containers/create?name="+url.QueryEscape(newContainerName(prefix)), body, &created)
		if err == nil {
			return created, nil
		}
		lastErr = err
		var apiErr *apiError
		if !asAPIError(err, &apiErr) || apiErr.status != http.StatusConflict {
			return createdContainer{}, err
		}
	}
	return createdContainer{}, lastErr
}

// acquireCreateSlot takes one of the backend's create slots and returns its release.
//
// Create and the pool's refill share it: the gate exists to bound the daemon's queue,
// and a refill that bypassed it would stampede that queue at the moment the pool is
// supposed to be keeping the node quiet.
func (b *Backend) acquireCreateSlot(ctx context.Context) (func(), error) {
	if b.createSem == nil {
		return func() {}, nil
	}
	select {
	case b.createSem <- struct{}{}:
		return func() { <-b.createSem }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (b *Backend) createBody(spec backend.Spec, nodeID string, binds []string) map[string]any {
	env := make([]string, 0, len(spec.Env))
	for key, value := range spec.Env {
		env = append(env, key+"="+value)
	}
	sort.Strings(env)

	labels := map[string]string{
		labelSandbox: "true",
		labelOwner:   b.cfg.OwnerID,
		labelClass:   spec.ResourceClass,
		labelNode:    nodeID,
	}
	if spec.ExecMode != "" {
		labels[labelExecMode] = spec.ExecMode
	}
	if spec.Lifetime > 0 {
		labels[labelLifetime] = strconv.FormatInt(int64(spec.Lifetime/time.Second), 10)
	}
	if spec.Resources.DiskMB > 0 {
		labels[labelDisk] = strconv.FormatInt(spec.Resources.DiskMB, 10)
	}
	if spec.IdempotencyKey != "" {
		labels[labelIdempotency] = spec.IdempotencyKey
	}
	for key, value := range spec.Metadata {
		labels[key] = value
	}

	hostConfig := map[string]any{
		// A sandbox is destroyed by this service, not by the daemon, so its exit
		// must leave a record a post mortem can read.
		"AutoRemove": false,
	}
	if b.cfg.NetworkMode != "" {
		// "host" removes per-container veth/netns creation and iptables churn,
		// eliminating the main kernel-lock bottleneck under concurrent creates.
		hostConfig["NetworkMode"] = b.cfg.NetworkMode
	}
	if spec.Resources.MemoryMB > 0 {
		hostConfig["Memory"] = spec.Resources.MemoryMB * 1024 * 1024
	}
	if spec.Resources.CPUCount > 0 {
		// Quota against the default 100ms period, which is how the Engine expresses
		// a fractional core.
		hostConfig["CpuPeriod"] = 100000
		hostConfig["CpuQuota"] = int64(spec.Resources.CPUCount * 100000)
	}
	if b.cfg.Runtime != "" {
		hostConfig["Runtime"] = b.cfg.Runtime
	}
	if b.cfg.CgroupParent != "" {
		hostConfig["CgroupParent"] = b.cfg.CgroupParent
	}
	if len(binds) > 0 {
		hostConfig["Binds"] = binds
	}
	if len(spec.AssignedGPUs) > 0 {
		hostConfig["DeviceRequests"] = []map[string]any{{
			"Driver":       "nvidia",
			"DeviceIDs":    deviceIDs(spec.AssignedGPUs),
			"Capabilities": [][]string{{"gpu"}},
		}}
		// Restrict the container to exactly its granted devices. Without this the
		// NVIDIA runtime exposes all devices and two concurrent sandboxes can
		// interfere with each other's workloads even though admission partitioned
		// the indices correctly.
		env = append(env, "CUDA_VISIBLE_DEVICES="+cudaVisibleDevices(spec.AssignedGPUs))
		sort.Strings(env)
	}

	body := map[string]any{
		"Image":      spec.Source.Reference,
		"Env":        env,
		"Labels":     labels,
		"HostConfig": hostConfig,
		// A sandbox waits for commands rather than running a workload of its own, so
		// it needs an entry process that does not exit.
		"Cmd": []string{"sleep", "infinity"},
	}
	if spec.Workdir != "" {
		body["WorkingDir"] = spec.Workdir
	}
	return body
}

// binds turns a spec's mounts into the daemon's bind strings, refusing what this node
// will not share.
//
// Two things make this more than a string join. A path is resolved before it is compared,
// because a symlink inside an allowed tree is a way out of it: a caller who can create a
// link under /shared can otherwise mount /etc. And a path that does not exist is refused
// rather than passed through, because the daemon creates a missing bind source as a
// directory on the host -- a caller would get an empty mount and no error, which reads as
// success for a request that was never honoured.
func (b *Backend) binds(spec backend.Spec) ([]string, error) {
	if len(spec.Mounts) == 0 {
		return nil, nil
	}
	if len(b.cfg.MountAllowlist) == 0 {
		return nil, fmt.Errorf(
			"docker: this sandbox asks for %d host mount(s) but this node allows none; "+
				"set backends[].mount_allowlist to the host trees this node may share", len(spec.Mounts))
	}
	var allowed []string
	for _, root := range b.cfg.MountAllowlist {
		resolved, err := filepath.EvalSymlinks(root)
		if err != nil {
			return nil, fmt.Errorf("docker: mount allowlist entry %q is not a path on this node: %w", root, err)
		}
		allowed = append(allowed, resolved)
	}
	binds := make([]string, 0, len(spec.Mounts))
	for _, mount := range spec.Mounts {
		if mount.Target == "" {
			return nil, fmt.Errorf("docker: mount source %q names no path inside the sandbox", mount.Source)
		}
		source, err := filepath.EvalSymlinks(mount.Source)
		if err != nil {
			return nil, fmt.Errorf("docker: mount source %q is not a path on this node: %w", mount.Source, err)
		}
		permitted := false
		for _, root := range allowed {
			if withinPath(root, source) {
				permitted = true
				break
			}
		}
		if !permitted {
			return nil, fmt.Errorf(
				"docker: mount source %q is outside this node's mount allowlist (%s)",
				source, strings.Join(b.cfg.MountAllowlist, ", "))
		}
		mode := "rw"
		if mount.ReadOnly {
			mode = "ro"
		}
		binds = append(binds, fmt.Sprintf("%s:%s:%s", source, mount.Target, mode))
	}
	return binds, nil
}

// withinPath reports whether path is root itself or lives under it.
//
// The separator is part of the comparison: a prefix check alone would let
// /shared/psrl-secrets through an allowlist of /shared/psrl.
func withinPath(root, path string) bool {
	if path == root {
		return true
	}
	return strings.HasPrefix(path, root+string(filepath.Separator))
}

func deviceIDs(indices []int32) []string {
	out := make([]string, len(indices))
	for i, index := range indices {
		out[i] = fmt.Sprint(index)
	}
	return out
}

// cudaVisibleDevices renders the CUDA_VISIBLE_DEVICES value for the granted
// indices. This restricts what the NVIDIA runtime exposes inside the container,
// so a sandbox that was admitted for device 2 cannot see device 0 or 1.
func cudaVisibleDevices(indices []int32) string {
	parts := make([]string, len(indices))
	for i, index := range indices {
		parts[i] = fmt.Sprint(index)
	}
	return strings.Join(parts, ",")
}

// ensureImage pulls an image the node does not have.
//
// The pull has its own budget, because its duration is the registry and the
// image rather than a round trip, and holding a create's deadline against it
// would report a slow registry as a node fault.
func (b *Backend) ensureImage(ctx context.Context, reference string) error {
	if reference == "" {
		return fmt.Errorf("docker needs an image reference")
	}
	inspectCtx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if err := b.call(inspectCtx, http.MethodGet, "/images/"+reference+"/json", nil, nil); err == nil {
		return nil
	}
	pullCtx, cancelPull := context.WithTimeout(ctx, b.cfg.PullTimeout)
	defer cancelPull()
	return b.call(pullCtx, http.MethodPost, "/images/create?fromImage="+reference, nil, nil)
}

// Release destroys one sandbox and confirms it is gone.
//
// A removal that is merely requested is not a removal: until the daemon confirms
// it, the container still holds the memory its reservation covers, so returning
// the reservation here would over-commit the node.
func (b *Backend) Release(ctx context.Context, handle backend.Handle) error {
	// Close any persistent shell first, so its process exits with the container
	// rather than being orphaned against a container that is already gone.
	b.shells.drop(handle.SandboxID)
	if err := b.remove(ctx, handle.SandboxID); err != nil {
		return err
	}
	b.mu.Lock()
	delete(b.known, handle.SandboxID)
	b.mu.Unlock()
	return nil
}

func (b *Backend) remove(ctx context.Context, containerID string) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	err := b.call(ctx, http.MethodDelete, "/containers/"+containerID+"?force=true&v=true", nil, nil)
	if err == nil || isNotFound(err) {
		// Already gone is the outcome the caller wanted.
		return nil
	}
	return fmt.Errorf("docker remove %s: %w", containerID, err)
}

// Status reports a sandbox's portable state.
func (b *Backend) Status(ctx context.Context, handle backend.Handle) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var inspected struct {
		State struct {
			Status   string `json:"Status"`
			Running  bool   `json:"Running"`
			Paused   bool   `json:"Paused"`
			OOMKill  bool   `json:"OOMKilled"`
			ExitCode int    `json:"ExitCode"`
		} `json:"State"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/"+handle.SandboxID+"/json", nil, &inspected); err != nil {
		if isNotFound(err) {
			return "terminated", nil
		}
		return "", err
	}
	switch {
	case inspected.State.Paused:
		return "paused", nil
	case inspected.State.Running:
		// Running wins over OOMKilled: a container that was restarted after an
		// OOM reports the historical flag while its process is healthy, and
		// reading that as "oom" would reap a live sandbox.
		return "running", nil
	case inspected.State.OOMKill:
		return "oom", nil
	case inspected.State.Status == "exited" || inspected.State.Status == "dead":
		return "exited", nil
	default:
		// "created" and "restarting" are transitional; a caller that read them
		// as gone would destroy a sandbox the daemon is bringing back.
		return "unknown", nil
	}
}

// ExitDetail reports the daemon's exit code and finish time for a sandbox that
// stopped. A container the daemon no longer has answers with Running=false and
// an unknown code rather than an error: "removed" is a stronger statement than
// "stopped", and the node's own attribution names it.
func (b *Backend) ExitDetail(ctx context.Context, handle backend.Handle) (backend.ExitDetail, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var inspected struct {
		RestartCount int `json:"RestartCount"`
		State        struct {
			Status     string `json:"Status"`
			Running    bool   `json:"Running"`
			OOMKilled  bool   `json:"OOMKilled"`
			ExitCode   int    `json:"ExitCode"`
			Error      string `json:"Error"`
			StartedAt  string `json:"StartedAt"`
			FinishedAt string `json:"FinishedAt"`
		} `json:"State"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/"+handle.SandboxID+"/json", nil, &inspected); err != nil {
		if isNotFound(err) {
			// The container is removed, not merely stopped. The caller's
			// status vocabulary has a distinct word for that, so this reports
			// the sentinel rather than an empty detail that would read as a
			// plain exit.
			return backend.ExitDetail{}, backend.ErrRuntimeGone
		}
		return backend.ExitDetail{}, err
	}
	detail := backend.ExitDetail{
		Running:       inspected.State.Running,
		OOMKilled:     inspected.State.OOMKilled && !inspected.State.Running,
		ExitCode:      inspected.State.ExitCode,
		ExitCodeKnown: !inspected.State.Running,
		Status:        inspected.State.Status,
		Error:         inspected.State.Error,
		RestartCount:  inspected.RestartCount,
	}
	if started, err := time.Parse(time.RFC3339Nano, inspected.State.StartedAt); err == nil {
		detail.StartedAt = started
	}
	if finished, err := time.Parse(time.RFC3339Nano, inspected.State.FinishedAt); err == nil {
		detail.FinishedAt = finished
	}
	return detail, nil
}

// Pause freezes a container's process tree, keeping it resident on this node.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf("docker keeps the container resident, so it offers a freeze rather than %q", mode)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	return b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/pause", nil, nil)
}

// Resume unfreezes a paused container.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	return b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/unpause", nil, nil)
}

// Snapshot commits the container's filesystem.
//
// Filesystem only: Docker has no supported memory checkpoint, so claiming a full
// state capture would promise a resume this backend cannot perform.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf("docker commits the writable layer, so it captures a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.PullTimeout)
	defer cancel()
	tag := fmt.Sprintf("psrl-snapshot:%s-%d", handle.SandboxID[:min(12, len(handle.SandboxID))], time.Now().UnixNano())
	path := fmt.Sprintf("/commit?container=%s&repo=%s&tag=%s",
		handle.SandboxID, "psrl-snapshot", strings.SplitN(tag, ":", 2)[1])
	var committed struct {
		ID string `json:"Id"`
	}
	if err := b.call(ctx, http.MethodPost, path, nil, &committed); err != nil {
		return "", fmt.Errorf("docker commit: %w", err)
	}
	return committed.ID, nil
}

// ReadFile fetches one file from a sandbox via the Engine's archive endpoint.
//
// The shell-based approach (base64 | base64 -d) puts the entire file content
// on the command line, which fails with ENOMEM / "argument list too long" for
// files larger than ~96 KiB. The archive endpoint streams a tar containing the
// file without any shell involvement, so it works for arbitrarily large files.
func (b *Backend) ReadFile(ctx context.Context, handle backend.Handle, path string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	raw, err := b.callRaw(ctx, http.MethodGet,
		"/containers/"+handle.SandboxID+"/archive?path="+urlEncodePath(path), nil)
	if err != nil {
		// The Engine answers 404 both for "this path is not in the container"
		// and for "this container does not exist". They are different facts to
		// the caller: one leaves the sandbox usable, the other does not.
		switch {
		case isContainerGone(err):
			return nil, fmt.Errorf("docker archive get %q: %w (docker: %v)", path, backend.ErrRuntimeGone, err)
		case isArchiveFileMissing(err):
			return nil, fmt.Errorf("docker archive get %q: %w (docker: %v)", path, backend.ErrPathNotFound, err)
		default:
			return nil, fmt.Errorf("docker archive get %q: %w", path, err)
		}
	}
	return extractTarSingle(raw)
}

// WriteFile uploads one file into a sandbox via the Engine's archive endpoint.
//
// Using PUT /containers/{id}/archive avoids the shell-argument-list limit that
// breaks the old printf-base64 approach for files larger than ~96 KiB (e.g.
// a large model patch). The data travels as a tar stream in the request body.
func (b *Backend) WriteFile(ctx context.Context, handle backend.Handle, path string, data []byte) error {
	tarBytes, err := singleFileTar(path, data)
	if err != nil {
		return fmt.Errorf("docker archive put %q: build tar: %w", path, err)
	}
	dir := parentDir(path)
	// Ensure the directory exists first so the archive upload lands cleanly.
	//
	// The command runs on the caller's context, not on the Engine budget that bounds
	// the upload below. How long a command takes is the command's own business, and
	// its bounds are the caller's own command timeout and the sandbox's lifetime.
	// Charging it to the request timeout makes a daemon too busy to start a process
	// report a stalled mkdir as a dead write, which is a fault in the wrong place.
	mkdirCmd := "mkdir -p " + shellescape(dir)
	result, execErr := b.execOneShot(ctx, handle, mkdirCmd, "", nil)
	if execErr != nil {
		return fmt.Errorf("docker archive put %q: mkdir: %w", path, execErr)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("docker archive put %q: mkdir exited %d: %s", path, result.ExitCode, result.Stderr)
	}
	putCtx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if err := b.putArchive(putCtx, handle.SandboxID, dir, tarBytes); err != nil {
		return fmt.Errorf("docker archive put %q: %w", path, err)
	}
	return nil
}

// putArchive calls PUT /containers/{id}/archive with a tar body.
func (b *Backend) putArchive(ctx context.Context, containerID, dir string, tarData []byte) error {
	url := "http://docker/" + b.cfg.APIVersion + "/containers/" + containerID + "/archive?path=" + urlEncodePath(dir)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(tarData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-tar")
	resp, err := b.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	_, _ = raw.ReadFrom(resp.Body)
	if resp.StatusCode >= 400 {
		return &apiError{status: resp.StatusCode, body: strings.TrimSpace(raw.String())}
	}
	return nil
}

// singleFileTar creates an in-memory tar archive containing one file.
// The entry name is the base name; the caller controls where it is written via
// the archive upload path parameter.
func singleFileTar(filePath string, data []byte) ([]byte, error) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	name := path.Base(filePath)
	if err := tw.WriteHeader(&tar.Header{
		Typeflag: tar.TypeReg,
		Name:     name,
		Size:     int64(len(data)),
		Mode:     0o644,
	}); err != nil {
		return nil, err
	}
	if _, err := tw.Write(data); err != nil {
		return nil, err
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// extractTarSingle reads the single file from a tar that the archive endpoint
// returns for a non-directory path.
func extractTarSingle(tarData []byte) ([]byte, error) {
	tr := tar.NewReader(bytes.NewReader(tarData))
	_, err := tr.Next()
	if err != nil {
		return nil, fmt.Errorf("archive entry: %w", err)
	}
	return io.ReadAll(tr)
}

// urlEncodePath percent-encodes a filesystem path for use in a query string.
// Only the characters that would break a URL need encoding; / is left intact
// because Docker parses it as a path component, not as a delimiter.
func urlEncodePath(p string) string {
	// net/url.PathEscape encodes / as %2F, which Docker does not accept in the
	// path query parameter. Replace it back after encoding.
	return strings.ReplaceAll(pathEscape(p), "%2F", "/")
}

// pathEscape is net/url.PathEscape without the special treatment of /.
func pathEscape(s string) string {
	var buf strings.Builder
	const safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:@!$&'()*+,;=/"
	for _, b := range []byte(s) {
		if strings.ContainsRune(safe, rune(b)) {
			buf.WriteByte(b)
		} else {
			fmt.Fprintf(&buf, "%%%02X", b)
		}
	}
	return buf.String()
}

// parentDir returns the directory component of a path, or "/" when there is
// none.
func parentDir(p string) string {
	d := path.Dir(p)
	if d == "" || d == "." {
		return "/"
	}
	return d
}

// shellescape wraps a string in single quotes, escaping any embedded single
// quotes. This is sufficient for mkdir paths that may contain spaces.
func shellescape(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}

// DeleteSnapshot removes a committed image.
func (b *Backend) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	err := b.call(ctx, http.MethodDelete, "/images/"+snapshotID+"?force=true", nil, nil)
	if err != nil && !isNotFound(err) {
		return err
	}
	return nil
}

// Exec runs one command in a sandbox under the strategy the spec asked for.
//
// This is here because a container has no in-sandbox agent to talk to. It is
// reached through the node agent rather than the control plane, so command
// traffic still stays node-local.
//
// A persistent shell keeps working directory, exported variables, and an
// activated environment alive between calls, which is what a multi-turn harness
// needs. One shot starts a process per command, which is cheaper and is right
// for a grader step that runs exactly once. The strategy is a property of the
// workload rather than of the backend, so it travels on the spec and the node
// passes it here.
func (b *Backend) Exec(
	ctx context.Context, handle backend.Handle, execMode, command, workdir string, env map[string]string,
	opts backend.ExecOptions,
) (backend.ExecResult, error) {
	if execMode == "persistent" {
		return b.execPersistent(ctx, handle, command, workdir, env, opts.SilenceTimeout)
	}
	// A one-shot command is a single request whose reply is the whole result, so the
	// caller's deadline is the only budget it needs: there is no stream to watch for
	// silence.
	return b.execOneShot(ctx, handle, command, workdir, env)
}

// execOneShot starts a new process per command, holding no state between calls.
func (b *Backend) execOneShot(ctx context.Context, handle backend.Handle, command string, workdir string, env map[string]string) (backend.ExecResult, error) {
	envList := make([]string, 0, len(env))
	for key, value := range env {
		envList = append(envList, key+"="+value)
	}
	sort.Strings(envList)
	create := map[string]any{
		"AttachStdout": true,
		"AttachStderr": true,
		"Cmd":          []string{"/bin/sh", "-c", command},
	}
	if workdir != "" {
		create["WorkingDir"] = workdir
	}
	if len(envList) > 0 {
		create["Env"] = envList
	}
	var created struct {
		ID string `json:"Id"`
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/exec", create, &created); err != nil {
		return backend.ExecResult{}, commandError("docker exec create", err)
	}
	output, err := b.callRaw(ctx, http.MethodPost, "/exec/"+created.ID+"/start", map[string]any{"Detach": false, "Tty": false})
	if err != nil {
		return backend.ExecResult{}, commandError("docker exec start", err)
	}
	var inspected struct {
		ExitCode int  `json:"ExitCode"`
		Running  bool `json:"Running"`
	}
	if err := b.call(ctx, http.MethodGet, "/exec/"+created.ID+"/json", nil, &inspected); err != nil {
		return backend.ExecResult{}, commandError("docker exec inspect", err)
	}
	stdout, stderr, truncated := demultiplexStreams(output)
	if stoppedExecOutput(stdout, stderr) {
		// runc/OCI reports a container whose namespaces are gone on the exec's
		// output stream, not as an Engine error: the API call succeeds, the exec
		// exits 127, and the text says why. Returning that as a command result
		// makes a dead sandbox look like a collector that exited 127. It is a
		// runtime fact, so it takes the same ErrRuntimeGone path as a 409.
		detail := strings.TrimSpace(strings.TrimSpace(stdout) + " " + strings.TrimSpace(stderr))
		return backend.ExecResult{}, fmt.Errorf("docker exec: %w: %s", backend.ErrRuntimeGone, detail)
	}
	return backend.ExecResult{
		ExitCode:  inspected.ExitCode,
		Stdout:    stdout,
		Stderr:    stderr,
		Truncated: truncated,
	}, nil
}

// stoppedExecOutput reports whether the exec's own output is runc saying the
// container could not be entered.
//
// The Engine returns HTTP 200 for an exec that starts and immediately fails in
// the runtime; the OCI error text is the only signal. These phrases are runc's
// stable wording for a container whose init process or namespaces are gone.
func stoppedExecOutput(stdout, stderr string) bool {
	text := stdout + "\n" + stderr
	return strings.Contains(text, "OCI runtime exec failed") ||
		strings.Contains(text, "cannot exec in a stopped container") ||
		(strings.Contains(text, "RWLayer of container") && strings.Contains(text, "unexpectedly nil"))
}

// commandError classifies a failed Engine call.
//
// A container that is not running or no longer exists is a fact about the
// runtime, not a transport fault. The caller must replace the sandbox and the
// node must stop charging it, so it is reported as ErrRuntimeGone rather than
// as an indistinguishable "the daemon did not answer".
func commandError(op string, err error) error {
	if isContainerGone(err) {
		return fmt.Errorf("%s: %w (docker: %v)", op, backend.ErrRuntimeGone, err)
	}
	return fmt.Errorf("%s: %w", op, err)
}

// demultiplexStreams separates the Engine's stdout and stderr frames.
//
// An attached exec without a TTY returns one framed stream carrying both
// streams: an eight-byte header per chunk whose first byte names the stream
// (1 stdout, 2 stderr) and whose last four bytes are the payload length. The
// order within each stream is preserved; interleaving between them is not
// recoverable from two separate strings, which is why the result keeps them
// apart instead of concatenating them.
func demultiplexStreams(raw []byte) (stdout, stderr string, truncated bool) {
	var out, errOut bytes.Buffer
	sawFrame := false
	for len(raw) >= 8 {
		sawFrame = true
		stream := raw[0]
		size := int(binary.BigEndian.Uint32(raw[4:8]))
		raw = raw[8:]
		if size > len(raw) {
			// A short frame means the stream ended mid-write. Keep what arrived
			// and tell the caller its output is incomplete rather than silently
			// returning a truncated answer as if it were whole.
			size = len(raw)
			truncated = true
		}
		if stream == 2 {
			errOut.Write(raw[:size])
		} else {
			out.Write(raw[:size])
		}
		raw = raw[size:]
	}
	if !sawFrame {
		// A daemon that returned unframed bytes is older than the framing; keep
		// them rather than dropping a command's output.
		out.Write(raw)
	}
	return out.String(), errOut.String(), truncated
}

// Reconcile returns every sandbox this service owns on this node, with the facts
// a restarted node needs to charge for it.
//
// The daemon is the source of truth: a service that trusted its own list would
// leak every container it forgot across a restart. Unclaimed pool inventory comes back
// marked as such, by name, so the node reclaims it without ever mistaking it for a
// sandbox -- which is what a label could not express, because a label is immutable and
// an entry keeps it after it has been claimed.
func (b *Backend) Reconcile(ctx context.Context) ([]backend.Reconciled, error) {
	listed, err := b.listOwned(ctx, labelSandbox+"=true", labelOwner+"="+b.cfg.OwnerID)
	if err != nil {
		return nil, err
	}
	found, err := b.inspectAll(ctx, listed)
	if err != nil {
		return nil, err
	}
	// A sandbox outside the configured subtree is invisible to the sampler: its memory
	// counts against the ancestor but not against any ceiling this node enforces. That
	// is a configuration change with live sandboxes behind it, so it is refused rather
	// than adopted.
	if b.cfg.CgroupParent != "" {
		for _, rec := range found {
			if rec.Running && !rec.PoolInventory && rec.CgroupParent != b.cfg.CgroupParent {
				return nil, fmt.Errorf(
					"container %s is running outside the configured cgroup subtree %q (it is in %q): "+
						"the sampler does not measure it, so its memory is invisible to this node's "+
						"ceilings; release it before restarting on the new configuration",
					rec.Handle.SandboxID, b.cfg.CgroupParent, rec.CgroupParent)
			}
		}
	}
	return found, nil
}

// FindByIdempotencyKey returns the sandbox a create with this key already made.
//
// A create whose response was lost leaves a container the caller never learned
// the id of, so a retry has to find it rather than make a second one.
func (b *Backend) FindByIdempotencyKey(
	ctx context.Context, key string,
) (backend.Reconciled, bool, error) {
	if key == "" {
		return backend.Reconciled{}, false, nil
	}
	listed, err := b.listOwned(ctx,
		labelSandbox+"=true", labelOwner+"="+b.cfg.OwnerID, labelIdempotency+"="+key)
	if err != nil {
		return backend.Reconciled{}, false, err
	}
	if len(listed) == 0 {
		return backend.Reconciled{}, false, nil
	}
	found, err := b.inspectOne(ctx, listed[0])
	if err != nil {
		return backend.Reconciled{}, false, err
	}
	return found, true, nil
}

// ForeignContainers counts running containers on this daemon that this service does not
// own.
//
// They cannot be inside the sandbox subtree -- nothing but this service sets the parent
// -- so they add to the ancestor's measured memory without appearing in any ceiling the
// node enforces. That is worth a warning: the operator is sharing a daemon, and the
// capacity it costs is invisible.
func (b *Backend) ForeignContainers(ctx context.Context) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var listed []struct {
		Labels map[string]string `json:"Labels"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/json", nil, &listed); err != nil {
		return 0, err
	}
	foreign := 0
	for _, container := range listed {
		if container.Labels[labelSandbox] != "true" {
			foreign++
		}
	}
	return foreign, nil
}

// listOwned lists the container ids this service owns that match every label.
func (b *Backend) listOwned(ctx context.Context, labels ...string) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	filter, err := json.Marshal(map[string][]string{"label": labels})
	if err != nil {
		return nil, err
	}
	var listed []struct {
		ID string `json:"Id"`
	}
	path := "/containers/json?all=true&filters=" + urlEncode(string(filter))
	if err := b.call(ctx, http.MethodGet, path, nil, &listed); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(listed))
	for _, container := range listed {
		out = append(out, container.ID)
	}
	return out, nil
}

// inspectAll reads every id back, bounded so a node holding many containers does
// not serialize the work into a slow startup.
func (b *Backend) inspectAll(ctx context.Context, ids []string) ([]backend.Reconciled, error) {
	const inspectors = 8
	out := make([]backend.Reconciled, len(ids))
	errs := make([]error, len(ids))
	sem := make(chan struct{}, inspectors)
	var wg sync.WaitGroup
	for i, id := range ids {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			out[i], errs[i] = b.inspectOne(ctx, id)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("inspect %s: %w", ids[i], err)
		}
	}
	return out, nil
}

// inspectOne rebuilds what one container holds from the daemon's own view of it.
func (b *Backend) inspectOne(ctx context.Context, id string) (backend.Reconciled, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var inspected struct {
		ID      string `json:"Id"`
		Name    string `json:"Name"`
		Created string `json:"Created"`
		State   struct {
			Running    bool   `json:"Running"`
			OOMKilled  bool   `json:"OOMKilled"`
			ExitCode   int    `json:"ExitCode"`
			Status     string `json:"Status"`
			FinishedAt string `json:"FinishedAt"`
		} `json:"State"`
		Config struct {
			Labels map[string]string `json:"Labels"`
			Env    []string          `json:"Env"`
		} `json:"Config"`
		HostConfig struct {
			Memory       int64  `json:"Memory"`
			CpuQuota     int64  `json:"CpuQuota"`
			CpuPeriod    int64  `json:"CpuPeriod"`
			CgroupParent string `json:"CgroupParent"`
		} `json:"HostConfig"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/"+id+"/json", nil, &inspected); err != nil {
		return backend.Reconciled{}, err
	}
	labels := inspected.Config.Labels
	rec := backend.Reconciled{
		Name:          inspected.Name,
		PoolInventory: isPoolInventoryName(inspected.Name),
		CgroupParent:  inspected.HostConfig.CgroupParent,
		Handle:        backend.Handle{Backend: b.Name(), SandboxID: id, NodeID: labels[labelNode]},
		Owner:         labels[backend.OwnerMetadataKey],
		Class:         labels[labelClass],
		ExecMode:      labels[labelExecMode],
		Running:       inspected.State.Running,
		// The daemon's attribution for a container that is no longer running.
		// A node that adopts by restart must be able to say why a stopped
		// container is being reclaimed, not just that it is.
		OOMKilled: inspected.State.OOMKilled && !inspected.State.Running,
		ExitCode:  inspected.State.ExitCode,
		Resources: backend.Resources{
			MemoryMB: inspected.HostConfig.Memory / (1024 * 1024),
		},
	}
	// CPU is charged in millis and written as a quota against a fixed period, so
	// reading it back is the same division the create made.
	if inspected.HostConfig.CpuPeriod > 0 {
		rec.Resources.CPUCount = float64(inspected.HostConfig.CpuQuota) /
			float64(inspected.HostConfig.CpuPeriod)
	}
	if seconds, err := strconv.ParseInt(labels[labelLifetime], 10, 64); err == nil {
		rec.Lifetime = time.Duration(seconds) * time.Second
	}
	if diskMB, err := strconv.ParseInt(labels[labelDisk], 10, 64); err == nil {
		rec.Resources.DiskMB = diskMB
	}
	if created, err := time.Parse(time.RFC3339Nano, inspected.Created); err == nil {
		rec.CreatedAt = created
	}
	if finished, err := time.Parse(time.RFC3339Nano, inspected.State.FinishedAt); err == nil {
		rec.FinishedAt = finished
	}
	for _, entry := range inspected.Config.Env {
		if value, found := strings.CutPrefix(entry, "CUDA_VISIBLE_DEVICES="); found {
			rec.GPUIndices = parseDeviceList(value)
		}
	}
	return rec, nil
}

// parseDeviceList reads the comma-separated device indices a create wrote.
func parseDeviceList(value string) []int32 {
	if value == "" {
		return nil
	}
	out := make([]int32, 0, 1)
	for _, part := range strings.Split(value, ",") {
		if index, err := strconv.ParseInt(strings.TrimSpace(part), 10, 32); err == nil {
			out = append(out, int32(index))
		}
	}
	return out
}

// Start begins the warm pool's refill and sweep loops.
//
// It is separate from New because the pool creates containers, and construction
// has to be able to happen inside a preflight that is not allowed to change the
// node's state. A backend whose pool is disabled does nothing here.
func (b *Backend) Start(ctx context.Context) {
	b.pool.Start(ctx)
}

// Stop ends the pool's loops and destroys every entry it still holds, so a
// shutdown leaves no unowned container behind for the next start to reclaim.
func (b *Backend) Stop(ctx context.Context) {
	b.pool.Stop(ctx)
}

// WarmPool reports what the pool holds, for the metric hook. A backend with no
// pool reports a zero value rather than nothing, so the family is present either
// way and a reader does not have to special-case its absence.
func (b *Backend) WarmPool() WarmPoolReport {
	return b.pool.Snapshot()
}

// Preflight refuses a node whose daemon this service cannot drive.
//
// Every check here is something that would otherwise fail inside an episode,
// where it costs a sample and reads as a flaky rollout rather than as the
// configuration error it is. The pinned API version is compared against what
// the daemon will actually serve, and a configured runtime has to exist: both
// are declared as capabilities, and a capability the node cannot honour is
// worse than one it never claimed.
func (b *Backend) Preflight(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var version struct {
		APIVersion    string `json:"ApiVersion"`
		MinAPIVersion string `json:"MinAPIVersion"`
		Version       string `json:"Version"`
	}
	// Unversioned, because the pinned version is what is being checked: asking
	// through the prefix makes a too-new pin look like a dead daemon.
	if err := b.callUnversioned(ctx, http.MethodGet, "/version", &version); err != nil {
		return fmt.Errorf("docker daemon at %s is not answering: %w", b.cfg.Socket, err)
	}
	if version.APIVersion == "" {
		return fmt.Errorf("docker daemon at %s reported no API version", b.cfg.Socket)
	}
	// The service pins an API version, so a daemon that will not serve it has to
	// fail here. Discovering it on the first create means finding out from an
	// endpoint that silently did not exist.
	if version.MinAPIVersion != "" && olderAPI(b.cfg.APIVersion, version.MinAPIVersion) {
		return fmt.Errorf(
			"docker daemon at %s serves API %s and newer, but this service is pinned to %s; "+
				"set backends[].api_version to %s or newer",
			b.cfg.Socket, version.MinAPIVersion, b.cfg.APIVersion, version.MinAPIVersion)
	}
	if olderAPI(version.APIVersion, b.cfg.APIVersion) {
		return fmt.Errorf(
			"docker daemon at %s serves API up to %s, but this service is pinned to %s; "+
				"upgrade the daemon or set backends[].api_version to %s",
			b.cfg.Socket, version.APIVersion, b.cfg.APIVersion, version.APIVersion)
	}
	if err := b.preflightCgroupDriver(ctx); err != nil {
		return err
	}
	return b.preflightRuntime(ctx)
}

// preflightCgroupDriver refuses a daemon that would not put a container in the
// configured subtree.
//
// Only the cgroupfs driver treats CgroupParent as a path. The systemd driver reads it as
// a slice name, so the same configuration would place containers somewhere else and the
// sampler would measure a domain they are not in -- a failure that otherwise shows up as
// a ceiling that never moves.
func (b *Backend) preflightCgroupDriver(ctx context.Context) error {
	if b.cfg.CgroupParent == "" {
		return nil
	}
	var info struct {
		CgroupDriver string `json:"CgroupDriver"`
	}
	if err := b.call(ctx, http.MethodGet, "/info", nil, &info); err != nil {
		return fmt.Errorf("docker daemon at %s did not report its cgroup driver: %w", b.cfg.Socket, err)
	}
	if info.CgroupDriver != "" && info.CgroupDriver != "cgroupfs" {
		return fmt.Errorf(
			"this backend sets cgroup_parent %q but the daemon at %s uses the %q cgroup driver, "+
				"which reads that value as a slice name rather than a path; containers would not land in "+
				"the sandbox subtree the sampler measures",
			b.cfg.CgroupParent, b.cfg.Socket, info.CgroupDriver)
	}
	return nil
}

// preflightRuntime refuses a configured runtime the daemon does not have.
//
// This backend declares isolation_runtime whenever a runtime is configured, and
// placement routes a spec that requires it here. If the runtime is absent, the
// daemon rejects the create -- so without this check the service advertises an
// isolation boundary it cannot provide and the refusal arrives per episode.
func (b *Backend) preflightRuntime(ctx context.Context) error {
	if b.cfg.Runtime == "" {
		return nil
	}
	var info struct {
		Runtimes map[string]struct {
			Path string `json:"path"`
		} `json:"Runtimes"`
	}
	if err := b.call(ctx, http.MethodGet, "/info", nil, &info); err != nil {
		return fmt.Errorf("docker daemon at %s did not report its runtimes: %w", b.cfg.Socket, err)
	}
	if _, has := info.Runtimes[b.cfg.Runtime]; !has {
		available := make([]string, 0, len(info.Runtimes))
		for name := range info.Runtimes {
			available = append(available, name)
		}
		sort.Strings(available)
		return fmt.Errorf(
			"docker runtime %q is not installed on this node (available: %v), but this backend "+
				"declares isolation_runtime with it, so a spec requiring stronger isolation would be "+
				"admitted and then run without it",
			b.cfg.Runtime, available)
	}
	return nil
}

// olderAPI reports whether a Docker API version precedes another.
//
// Compared field by field as integers rather than as strings, because the
// versions pass 1.9: "v1.10" sorts before "v1.9" lexically and after it
// numerically, and the wrong answer here refuses a daemon that would have worked.
func olderAPI(version, floor string) bool {
	left := apiParts(version)
	right := apiParts(floor)
	if left == nil || right == nil {
		// Not comparable. The only safe answer is "not older", so an unrecognised
		// version string never refuses a daemon that would have worked.
		return false
	}
	for i := 0; i < len(left) && i < len(right); i++ {
		if left[i] != right[i] {
			return left[i] < right[i]
		}
	}
	return len(left) < len(right)
}

func apiParts(version string) []int {
	fields := strings.Split(strings.TrimPrefix(version, "v"), ".")
	out := make([]int, 0, len(fields))
	for _, field := range fields {
		if field == "" {
			// An empty field is not a zero: "" would otherwise parse as version 0 and
			// compare older than everything.
			return nil
		}
		value := 0
		for i := 0; i < len(field); i++ {
			if field[i] < '0' || field[i] > '9' {
				// A version with a non-numeric field is not comparable, so the caller
				// gets the only safe answer: not older, and therefore not refused.
				return nil
			}
			value = value*10 + int(field[i]-'0')
		}
		out = append(out, value)
	}
	return out
}

func (b *Backend) call(ctx context.Context, method, path string, body any, out any) error {
	raw, err := b.callRaw(ctx, method, path, body)
	if err != nil {
		return err
	}
	if out == nil || len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, out)
}

// callUnversioned reaches an endpoint without the pinned API prefix.
//
// Only /version uses this, and it has to: a daemon rejects a URL carrying an
// API version it does not serve, so asking it through the prefix returns a 400
// that reads as "the daemon is not answering". The point of preflight is to say
// which setting is wrong, and that answer is in the reply this fetches.
func (b *Backend) callUnversioned(ctx context.Context, method, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, "http://docker"+path, bytes.NewReader(nil))
	if err != nil {
		return err
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	if _, err := raw.ReadFrom(resp.Body); err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("docker API returned %d: %s", resp.StatusCode, strings.TrimSpace(raw.String()))
	}
	if out == nil || raw.Len() == 0 {
		return nil
	}
	return json.Unmarshal(raw.Bytes(), out)
}

func (b *Backend) callRaw(ctx context.Context, method, path string, body any) ([]byte, error) {
	var payload *bytes.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		payload = bytes.NewReader(encoded)
	} else {
		payload = bytes.NewReader(nil)
	}
	url := "http://docker/" + b.cfg.APIVersion + path
	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	if _, err := raw.ReadFrom(resp.Body); err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, &apiError{status: resp.StatusCode, body: strings.TrimSpace(raw.String())}
	}
	return raw.Bytes(), nil
}

type apiError struct {
	status int
	body   string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("docker API returned %d: %s", e.status, e.body)
}

func isNotFound(err error) bool {
	var apiErr *apiError
	if ok := asAPIError(err, &apiErr); ok {
		return apiErr.status == http.StatusNotFound
	}
	return false
}

// apiErrorDetail returns the Engine status and body behind a wrapped error.
func apiErrorDetail(err error) (status int, body string, ok bool) {
	var apiErr *apiError
	if !asAPIError(err, &apiErr) {
		return 0, "", false
	}
	return apiErr.status, apiErr.body, true
}

// isContainerGone reports whether the Engine said the container is stopped or
// no longer exists.
//
// Both are 409 or 404 bodies rather than distinct status codes, so the message
// is the only signal. The two phrases are Docker's own stable wording:
// "Container <id> is not running" for a stopped container, "cannot exec in a
// stopped container" from the exec start, and "No such container" for a removed
// one. A removed container can also answer 500 with a nil writable layer, which
// is the daemon's storage backend reporting the same fact. A paused container
// also answers 409, and is deliberately not included: it can be resumed, so it
// is not gone.
func isContainerGone(err error) bool {
	status, body, ok := apiErrorDetail(err)
	if !ok {
		return false
	}
	lowered := strings.ToLower(body)
	switch status {
	case http.StatusConflict:
		return strings.Contains(lowered, "is not running") ||
			strings.Contains(lowered, "cannot exec in a stopped container")
	case http.StatusNotFound:
		return strings.Contains(lowered, "no such container")
	case http.StatusInternalServerError:
		return strings.Contains(lowered, "rwlayer of container") &&
			strings.Contains(lowered, "unexpectedly nil")
	}
	return false
}

// isArchiveFileMissing reports whether an archive read failed because the path
// is absent from a container that does exist.
func isArchiveFileMissing(err error) bool {
	status, body, ok := apiErrorDetail(err)
	if !ok || status != http.StatusNotFound {
		return false
	}
	return strings.Contains(strings.ToLower(body), "could not find the file")
}

func asAPIError(err error, target **apiError) bool {
	for err != nil {
		if typed, ok := err.(*apiError); ok {
			*target = typed
			return true
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return false
}

func urlEncode(value string) string {
	var out strings.Builder
	for _, r := range value {
		switch {
		case r == '{' || r == '}' || r == '[' || r == ']' || r == '"' || r == ',' ||
			r == ':' || r == '=' || r == ' ' || r == '\\' || r == '%':
			fmt.Fprintf(&out, "%%%02X", r)
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
