// Package backend is the contract a sandbox runtime implements.
//
// It is deliberately narrow. Everything a backend must provide is here;
// everything optional is a separate interface it may also implement, so a
// backend that cannot pause writes no pause method rather than one that returns
// an error.
//
// One method deserves its own explanation. AgentEndpoint is what makes the
// service affordable: an episode issues one create and dozens of commands, so
// command traffic goes straight from the caller to the agent inside the sandbox.
// A backend that could not report that address would force every command through
// the control plane, which adds a hop and a serialization per command without
// making a decision.
package backend

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrSessionLost marks a failure that is a fact about the caller's session
// rather than a fault of the runtime: the session a command was sent to is gone,
// so state the caller built in it is gone too.
//
// It is a sentinel rather than a status code so this package stays free of the
// wire's vocabulary. The service maps it to the classification a caller acts on,
// and the distinction is what lets a caller tell "the node did not answer" from
// "my session is gone and an infrastructure retry will not bring it back".
var ErrSessionLost = errors.New("sandbox session lost")

// ErrRuntimeGone marks a failure where the sandbox's runtime object is gone:
// the container stopped or no longer exists. It is deliberately distinct from
// ErrSessionLost, whose meaning is narrower: the state a caller built inside a
// command session is gone (a persistent shell exited), while the container may
// still be alive and able to serve a fresh command.
//
// The two need different answers. A runtime-gone sandbox must be charged back
// and replaced; a lost shell only costs the caller its working directory and
// exported variables, and destroying the sandbox would throw away a healthy
// runtime.
var ErrRuntimeGone = errors.New("sandbox runtime gone")

// ErrPathNotFound marks a file request that failed because the path does not
// exist inside a sandbox that is otherwise healthy. It exists because a missing
// file and a missing sandbox are both Docker 404s, and collapsing them made a
// harness report an absent diagnostics file as a lost session.
var ErrPathNotFound = errors.New("path not found in sandbox")

// ExecResult is one completed command: its output streams kept separate, its
// exit code, and whether the backend stopped retaining output at its budget.
type ExecResult struct {
	ExitCode  int
	Stdout    string
	Stderr    string
	Truncated bool
}

// ExecOptions are the budget facts a command carries with it.
//
// They exist because a command's deadline belongs to the command, not to the
// connection that carries it. A caller that wants ten minutes of `pytest` wants
// the sandbox to stop it at ten minutes and say so; a transport that gives up at
// the same instant instead leaves the process running, holds the sandbox's shell
// for the next caller, and reports a fault that says nothing about the command.
type ExecOptions struct {
	// Timeout is the command's whole budget. Zero means no deadline.
	Timeout time.Duration
	// SilenceTimeout is the longest the command may produce no output before it is
	// declared stuck. Zero takes the backend's default.
	SilenceTimeout time.Duration
}

// Executor is implemented by a backend that can run a command inside a
// sandbox. It is optional and separate from Backend because a backend whose
// sandboxes run their own in-agent command server never reaches this path.
//
// execMode is "persistent" or "one_shot"; an empty value takes the backend's
// default. The result carries stdout and stderr separately: a caller diagnosing
// a failed command needs to know which stream said what, and a backend that
// merged them cannot answer that.
type Executor interface {
	Exec(
		ctx context.Context, handle Handle, execMode, command, workdir string, env map[string]string, opts ExecOptions,
	) (ExecResult, error)
}

// SchedulingMode says who chooses the node a sandbox lands on.
//
// It is a deployment property rather than a backend capability: the same backend
// can run both ways in one fleet, which is what makes a scheduler ablation
// possible with every other variable held fixed.
type SchedulingMode string

const (
	// SchedulingDirect means this service both places the sandbox and drives the
	// node runtime itself: its own fleet view and node admission choose the node,
	// and the call goes straight to that node's runtime. The backend's own control
	// plane is not started and is not in the path, so nothing between this service
	// and the runtime can add a hop or serialize a burst of creates.
	//
	// Used where the backend's own scheduler has nothing to add, and where its
	// control plane would only be a relay.
	SchedulingDirect SchedulingMode = "direct"
	// SchedulingProvider means the backend's control plane chooses the node and
	// this service does cross-backend quota only. Used where the backend has a real
	// scheduler whose decisions would be lost by overriding it.
	SchedulingProvider SchedulingMode = "provider"
)

// Valid reports whether a mode is one of the two shapes.
func (m SchedulingMode) Valid() bool {
	return m == SchedulingDirect || m == SchedulingProvider
}

// Source is where a sandbox's filesystem comes from.
type Source struct {
	// Kind is "image" or "template".
	Kind      string
	Reference string
}

// Resources is a portable resource request. A zero field states no requirement,
// which is not the same as requesting nothing.
type Resources struct {
	CPUCount float64
	MemoryMB int64
	DiskMB   int64
	GPUCount int32
}

// Spec is the portable creation request.
// Mount is one host directory a sandbox needs to see.
//
// A mount is a request about the machine, not about the sandbox: it hands a sandbox a
// view of a path the node owns. That is why a backend is allowed to refuse one -- see the
// allowlist in the docker backend -- and why it travels as part of the spec's identity
// rather than as a backend option: two environments that see different host paths are not
// the same environment, and an idempotent create that silently dropped the mount would
// hand back a sandbox that cannot do its work.
type Mount struct {
	// Source is the path on the node, which must exist and must be allowed.
	Source string
	// Target is where the sandbox sees it.
	Target string
	// ReadOnly keeps the sandbox from writing to a path the node shares with others.
	ReadOnly bool
}

type Spec struct {
	Source    Source
	Resources Resources
	// Mounts are the host paths this sandbox must see. Empty is the common case, and a
	// backend that cannot mount refuses a spec that asks rather than degrading silently.
	Mounts []Mount

	ResourceClass  string
	WorkflowID     string
	IdempotencyKey string

	Env      map[string]string
	Metadata map[string]string
	Workdir  string
	// ExecMode is "persistent" or "one_shot". Empty takes the backend's default.
	ExecMode string

	RequiredFeatures []string
	RequiredResume   string
	Backend          string

	// RequiredNodeLabel pins this sandbox to nodes carrying the label, and
	// ForbiddenNodeLabels keep it off nodes carrying any of them. They are
	// separate constraints because neither implies the other: a dedicated
	// request names where it must go, and a request that must not share a
	// trainer's machine names where it must not, which holds even when no node
	// is labelled for it.
	RequiredNodeLabel   string
	ForbiddenNodeLabels []string

	// Lifetime is how long the sandbox may exist, counted from creation. Zero
	// takes the node's default. It is stated by the caller because the caller
	// knows what the sandbox is for, and it is the node's backstop for a caller
	// that dies or forgets to release.
	Lifetime time.Duration

	// AssignedGPUs are the device indices node admission granted. Written by the
	// service and never by a caller, which is why it is not part of a spec's
	// identity.
	AssignedGPUs []int32

	// BackendOptions is tuning only one backend understands, keyed by backend
	// name. It never takes part in routing and is not part of a spec's identity,
	// so a provider's own knobs stay reachable without making the portable spec
	// unportable.
	BackendOptions map[string]map[string]string
}

// OwnerMetadataKey is where a caller states the owner its sandboxes belong to.
//
// It is metadata rather than a field because the value is an opaque caller
// identity that the service only compares for equality. It lives here because
// the runtime copies spec metadata into its own labels, so this key is also how
// a restarted node recovers the owner of a container it no longer remembers.
const OwnerMetadataKey = "owner_id"

// Options returns this backend's own options from a spec, and whether the spec
// named any.
func (s Spec) Options(backend string) (map[string]string, bool) {
	options, named := s.BackendOptions[backend]
	return options, named
}

// Handle identifies one live sandbox.
type Handle struct {
	Backend   string
	SandboxID string
	NodeID    string
}

// AgentEndpoint is where a caller reaches the agent inside a sandbox.
type AgentEndpoint struct {
	// Address is the agent's base URL, with a scheme.
	Address string
	// Headers the agent requires. Carried rather than reconstructed, because a
	// resumed sandbox can move and can require different ones.
	Headers map[string]string
	// CallbackHostAlias and CallbackPort are how this sandbox reaches the caller's
	// own server, when the hosting node rewrites loopback URLs at all.
	CallbackHostAlias string
	CallbackPort      int32
}

// Capabilities is what a backend actually grants, which is not always what a
// deployment declared.
type Capabilities struct {
	Features []string
	// ResumeLevel is reported only with the resume_anywhere feature, so the two
	// cannot contradict each other.
	ResumeLevel string
	// PauseModes are the pause semantics on offer: a freeze keeps the sandbox
	// resident, a hibernation releases its compute.
	PauseModes []string
}

// Supports reports whether a feature is implemented with its declared semantics.
func (c Capabilities) Supports(feature string) bool {
	for _, have := range c.Features {
		if have == feature {
			return true
		}
	}
	return false
}

// Missing returns the required features this backend does not have.
func (c Capabilities) Missing(required []string) []string {
	var missing []string
	for _, feature := range required {
		if !c.Supports(feature) {
			missing = append(missing, feature)
		}
	}
	return missing
}

// Created is what a backend returns for a new sandbox.
type Created struct {
	Handle       Handle
	Capabilities Capabilities
	Agent        AgentEndpoint
	// WarmStart reports that this sandbox came from a pool rather than a cold
	// create, which is the only way a caller learns the claim was warm.
	WarmStart bool
}

// Backend is the required contract. Every backend implements all of it.
type Backend interface {
	Name() string
	Mode() SchedulingMode
	Capabilities() Capabilities

	// Create provisions one sandbox. A node id is empty when the backend's own
	// control plane chooses, and names the node when this service does.
	Create(ctx context.Context, nodeID string, spec Spec, callback string) (Created, error)
	Release(ctx context.Context, handle Handle) error
	Status(ctx context.Context, handle Handle) (string, error)
}

// NodeScheduled is implemented by a backend this service can place against. A
// backend without it runs in provider mode only.
type NodeScheduled interface {
	// Nodes returns the backend's node ids, for the service's own placement.
	Nodes(ctx context.Context) ([]string, error)
	// Headroom returns what one node could grant each class right now.
	Headroom(ctx context.Context, nodeID string) (map[string]Resources, error)
	// RegisterBinding tells the backend where a sandbox landed, so its own routing
	// keeps working while this service does the placing. A backend with no such
	// concept implements it as a no-op.
	RegisterBinding(ctx context.Context, nodeID string, sandboxID string) error
}

// FileHandler is implemented by a backend that supports efficient direct file
// transfer to and from a sandbox.
//
// A backend that implements this avoids the shell-based printf/base64 approach
// that fails with ENOMEM for files larger than ~96 KiB (Linux ARG_MAX). The
// docker backend implements this via the Engine API archive endpoint.
//
// When a backend does not implement FileHandler, the node falls back to the
// shell-based approach.
type FileHandler interface {
	ReadFile(ctx context.Context, handle Handle, path string) ([]byte, error)
	WriteFile(ctx context.Context, handle Handle, path string, data []byte) error
}

// Stateful is implemented by a backend that can pause, resume, or snapshot.
type Stateful interface {
	Pause(ctx context.Context, handle Handle, mode string) error
	Resume(ctx context.Context, handle Handle) error
	Snapshot(ctx context.Context, handle Handle, kind string) (string, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error
}

// Preflighter is implemented by a backend that can refuse a deployment it
// cannot drive, before the deployment serves rather than during it.
//
// A runtime too old, a missing device, an isolation runtime that was configured
// but never installed: each is a configuration error, and each is invisible
// until a create fails. Checked at startup it names the wrong setting; checked
// on the first rollout it costs a sample and reads as a flaky run.
//
// Installing a runtime is deliberately not here. That is a deployment's own job
// -- a package manager, an image, a daemonset -- and a service that shelled out
// to do it would own a second, worse provisioning tool. What the service owns is
// the refusal.
type Preflighter interface {
	Preflight(ctx context.Context) error
}

// Reconciled is one runtime object a backend found already present on this node,
// carrying the facts a restarted node needs to charge for it exactly as the
// create did.
type Reconciled struct {
	Handle     Handle
	Owner      string
	Class      string
	ExecMode   string
	Resources  Resources
	GPUIndices []int32
	Lifetime   time.Duration
	CreatedAt  time.Time
	Running    bool
	// OOMKilled and ExitCode are the daemon's attribution for a container that
	// stopped. They are what lets a restarted node report *why* an adoption was
	// refused instead of a bare "not running".
	OOMKilled bool
	ExitCode  int
	// FinishedAt is when the runtime observed the container stop. It is the
	// clock a post mortem compares against a node's cgroup and kernel events.
	FinishedAt time.Time
	// Name is the runtime's name for the object, when it has one. It is what
	// distinguishes unclaimed warm-pool inventory from a sandbox: a label written at
	// create cannot be removed when an entry is claimed, but a name can be changed.
	Name string
	// PoolInventory reports that this object is an unclaimed warm-pool entry. A
	// restarted node destroys these rather than charging them: nothing has been
	// promised them, and they are rebuilt by the pool's own refill.
	PoolInventory bool
	// CgroupParent is where the object actually runs, so a restarted node can tell a
	// sandbox it may adopt from one the sampler cannot see.
	CgroupParent string
}

// ExitDetail is the runtime's attribution for a sandbox that stopped.
//
// It is more than Status's one-word answer: a caller that is about to release a
// sandbox and blame a resource or an exit path needs the daemon's exit code and
// the moment it happened, not a reconstruction from the reason.
type ExitDetail struct {
	Running   bool
	OOMKilled bool
	ExitCode  int
	// ExitCodeKnown distinguishes a clean exit 0 from a runtime that had no
	// code to report (a removed container, or one that never started).
	ExitCodeKnown bool
	// Status and Error are the daemon's own state and error text. They are what
	// separates "the entrypoint exited 0" from "the runtime killed it" in a post
	// mortem; the exit code alone does not name which process or why.
	Status string
	Error  string
	// StartedAt precedes FinishedAt, so a loss report carries the lifetime the
	// sandbox actually had, not just the moment it stopped.
	StartedAt    time.Time
	FinishedAt   time.Time
	RestartCount int
}

// Evidence renders the daemon's account of the stop in one line. It is what
// distinguishes an entrypoint that exited cleanly from a runtime kill when the
// service's reason alone says only "exited".
func (d ExitDetail) Evidence() string {
	return fmt.Sprintf(
		"status=%s running=%t oom_killed=%t exit_code=%d exit_code_known=%t error=%q "+
			"started_at=%s finished_at=%s restart_count=%d",
		d.Status,
		d.Running,
		d.OOMKilled,
		d.ExitCode,
		d.ExitCodeKnown,
		d.Error,
		evidenceTime(d.StartedAt),
		evidenceTime(d.FinishedAt),
		d.RestartCount,
	)
}

func evidenceTime(t time.Time) string {
	if t.IsZero() {
		return "<unknown>"
	}
	return t.UTC().Format(time.RFC3339Nano)
}

// ExitDetailer is a backend that can report why a sandbox stopped, not merely
// that it did. It is optional: a backend that cannot inspect keeps the
// status-only attribution.
type ExitDetailer interface {
	ExitDetail(ctx context.Context, handle Handle) (ExitDetail, error)
}

// PoolBudget is what a backend that prefetches containers has to ask before it
// builds one.
//
// A warm pool is not a class: its entries exist before any episode does, so charging
// them to a class would let prefetching starve the work it exists to accelerate. The
// pool is instead held against the node's declared envelope, subtracted from every
// class's headroom, and released the moment an entry is claimed or destroyed.
type PoolBudget interface {
	// AcquirePool reserves one entry's footprint, reporting whether it fits.
	AcquirePool(Resources) bool
	// ReleasePool returns a footprint reserved by AcquirePool.
	ReleasePool(Resources)
	// MemoryPressure reports the measured memory fraction and whether it is usable.
	// An unusable reading must stop refill rather than let it run blind.
	MemoryPressure() (float64, bool)
}

// PoolBudgeted is implemented by a backend whose pool needs a budget.
type PoolBudgeted interface {
	SetPoolBudget(PoolBudget)
}

// Reconciler is implemented by a backend that can read back what it already
// holds on this node.
//
// The runtime is the source of truth, so a restarted node rebuilds its
// accounting from the daemon rather than from a second record that can disagree
// with it. Without this a restart leaks every container the node had adopted,
// because the ledger came back empty and nothing re-adopts them.
type Reconciler interface {
	// Reconcile returns every object this node holds, running or not.
	Reconcile(ctx context.Context) ([]Reconciled, error)
	// FindByIdempotencyKey returns the object a create with this key already
	// made, so a create whose response was lost is adopted rather than repeated.
	FindByIdempotencyKey(ctx context.Context, key string) (Reconciled, bool, error)
}

// Registry resolves a spec to the backend that will serve it.
type Registry struct {
	backends map[string]Backend
	order    []string
	fallback string
}

// NewRegistry returns a registry over the configured backends. The default is
// tried first, so a deployment with one backend never pays for a search.
func NewRegistry(backends []Backend, defaultBackend string) (*Registry, error) {
	if len(backends) == 0 {
		return nil, fmt.Errorf("a sandbox service needs at least one backend")
	}
	byName := make(map[string]Backend, len(backends))
	order := make([]string, 0, len(backends))
	for _, b := range backends {
		if _, duplicate := byName[b.Name()]; duplicate {
			return nil, fmt.Errorf("sandbox backend %q is configured twice; a backend is keyed by its name", b.Name())
		}
		if !b.Mode().Valid() {
			return nil, fmt.Errorf("sandbox backend %q declares scheduling mode %q, want psrl or provider", b.Name(), b.Mode())
		}
		byName[b.Name()] = b
		order = append(order, b.Name())
	}
	if defaultBackend == "" {
		defaultBackend = order[0]
	}
	if _, known := byName[defaultBackend]; !known {
		return nil, fmt.Errorf("default sandbox backend %q is not configured", defaultBackend)
	}
	// The default first, then the rest in declaration order, so selection is
	// deterministic and a one-backend deployment short-circuits.
	ordered := append([]string{defaultBackend}, filter(order, defaultBackend)...)
	return &Registry{backends: byName, order: ordered, fallback: defaultBackend}, nil
}

// Get returns one backend by name.
func (r *Registry) Get(name string) (Backend, error) {
	if name == "" {
		return r.backends[r.fallback], nil
	}
	b, known := r.backends[name]
	if !known {
		return nil, fmt.Errorf("sandbox backend %q is not configured (configured: %v)", name, r.order)
	}
	return b, nil
}

// All returns every configured backend, in selection order.
func (r *Registry) All() []Backend {
	out := make([]Backend, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.backends[name])
	}
	return out
}

// Select resolves a spec to a backend by capability, honouring a pin.
//
// Admission refuses rather than degrades: a caller that needs a live process to
// survive a move must never be handed a workspace-only resume and told it is the
// same thing.
func (r *Registry) Select(spec Spec) (Backend, error) {
	if spec.Backend != "" {
		return r.Get(spec.Backend)
	}
	reasons := make(map[string]string, len(r.order))
	for _, name := range r.order {
		b := r.backends[name]
		capabilities := b.Capabilities()
		if missing := capabilities.Missing(spec.RequiredFeatures); len(missing) > 0 {
			reasons[name] = fmt.Sprintf("missing features %v", missing)
			continue
		}
		if spec.RequiredResume != "" && !resumeSatisfies(capabilities.ResumeLevel, spec.RequiredResume) {
			reasons[name] = fmt.Sprintf("resumes at %q, which does not satisfy %q",
				capabilities.ResumeLevel, spec.RequiredResume)
			continue
		}
		return b, nil
	}
	return nil, fmt.Errorf(
		"no configured sandbox backend can serve this spec (%v). Declare the requirement on a backend "+
			"that supports it, or pin one", reasons)
}

var resumeRank = map[string]int{"": 0, "filesystem": 1, "full_state": 2}

func resumeSatisfies(have, need string) bool { return resumeRank[have] >= resumeRank[need] }

func filter(names []string, drop string) []string {
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name != drop {
			out = append(out, name)
		}
	}
	return out
}

// RetryAfter is how long a caller should wait before asking again, when a
// backend reports its own backpressure rather than a failure.
type RetryAfter struct {
	After time.Duration
}

func (r RetryAfter) Error() string {
	return fmt.Sprintf("the backend is applying backpressure; retry after %s", r.After)
}
