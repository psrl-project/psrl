package server

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
)

// Node is one machine's agent: admission, lifecycle, reclamation, and the
// report the fleet view is built from.
//
// It is the final authority on admission. A placement decision arrives here as
// a proposal, and this is where it meets the node's live pressure.
//
// It hosts every runtime installed on this machine rather than one, because the
// runtimes share the machine: a container daemon and a microVM node draw on the
// same memory, so one admission ledger has to cover both or the node admits
// twice what it has.
type Node struct {
	v1.UnimplementedSandboxNodeServer

	nodeID    string
	admission *node.Admission
	lifecycle *node.Lifecycle
	backends  map[string]backend.Backend
	// order is the declaration order, so a report lists runtimes the same way
	// twice and a default is the first declared.
	order []string

	// pressure reads what the machine is actually doing. Without it the node
	// would only know its own accounting, which is a record of reservations
	// rather than of load.
	pressure func() node.Pressure

	// elastic prices a sandbox from what its class actually costs and asks idle sandboxes
	// for their reclaimable memory. nil when the deployment has not enabled it, which
	// leaves admission on declared maxima.
	elastic *elastic
	// elasticDomain and elasticParent are where the controller reads per-sandbox usage and
	// lowers a limit. Without a domain it cannot do either, so it does nothing.
	elasticDomain cgroup.Domain
	elasticParent string
	// elasticBudget is the subtree's memory limit, the number every fraction is taken of.
	elasticBudget int64
}

// NodeConfig is what a node agent needs to run.
type NodeConfig struct {
	NodeID    string
	Admission *node.Admission
	Lifecycle *node.Lifecycle
	// Backends are every runtime this machine hosts, in declaration order.
	Backends []backend.Backend
	Pressure func() node.Pressure
	// Elastic turns on usage-based admission and the pressure-driven release. Disabled is
	// the previous behaviour: charge each sandbox its declared maximum.
	Elastic ElasticConfig
	// Domain and SandboxParent are the cgroup subtree the sandboxes live in. Without them
	// the controller has nothing to measure and stays inert.
	Domain        cgroup.Domain
	SandboxParent string
	// SandboxBudgetMB is that subtree's memory limit.
	SandboxBudgetMB int64
}

// NewNode returns the node-level server.
func NewNode(cfg NodeConfig) (*Node, error) {
	if cfg.NodeID == "" || cfg.Admission == nil || cfg.Lifecycle == nil || len(cfg.Backends) == 0 {
		return nil, fmt.Errorf("a node agent needs an id, admission, a lifecycle, and at least one backend")
	}
	backends := make(map[string]backend.Backend, len(cfg.Backends))
	order := make([]string, 0, len(cfg.Backends))
	for _, hosted := range cfg.Backends {
		if _, duplicate := backends[hosted.Name()]; duplicate {
			return nil, fmt.Errorf("node %q hosts backend %q twice; a runtime is keyed by its name",
				cfg.NodeID, hosted.Name())
		}
		backends[hosted.Name()] = hosted
		order = append(order, hosted.Name())
	}
	if cfg.Pressure == nil {
		cfg.Pressure = func() node.Pressure { return node.Pressure{} }
	}
	agent := &Node{
		nodeID:        cfg.NodeID,
		admission:     cfg.Admission,
		lifecycle:     cfg.Lifecycle,
		backends:      backends,
		order:         order,
		pressure:      cfg.Pressure,
		elasticDomain: cfg.Domain,
		elasticParent: cfg.SandboxParent,
		elasticBudget: cfg.SandboxBudgetMB,
	}
	if cfg.Elastic.Enabled {
		agent.elastic = &elastic{
			cfg:                   ElasticConfigWithDefaults(cfg.Elastic),
			memoryFootprints:      map[string]*node.Footprint{},
			cpuFootprints:         map[string]*node.Footprint{},
			maxMemoryMBBySandbox:  map[string]int64{},
			maxCPUMillisBySandbox: map[string]int64{},
			startMBBySandbox:      map[string]int64{},
			lowSince:              map[string]time.Time{},
			commandBusy:           map[string]bool{},
			forcedBurst:           map[string]int64{},
			failCountBySandbox:    map[string]int64{},
			oomCountBySandbox:     map[string]int64{},
			exhausted:             map[string]bool{},
			reapFailed:            map[string]bool{},
			foreignSince:          map[string]time.Time{},
			owned:                 map[string]struct{}{},
			classBySandbox:        map[string]string{},
			throttled:             map[string]bool{},
			cpuUsageNanos:         map[string]int64{},
			cpuRateMillis:         map[string]int64{},
			cpuThrottledNanos:     map[string]int64{},
			cpuThrottle:           map[string]float64{},
			throttleRun:           map[string]int{},
			previousPrice:         map[string]node.Resources{},
			classCount:            map[string]int{},
		}
		// Seed one estimate per declared class, so the node's first report already
		// publishes a price. Without it the batch admitted before the first report -- which
		// is every batch after a restart -- is charged the request's maximum, the
		// reservation this exists to remove.
		for class := range cfg.Admission.Snapshot().ClassHeadroom {
			agent.memoryFootprintFor(class)
			agent.cpuFootprintFor(class)
		}
	}
	return agent, nil
}

// Admit accepts or refuses one request against this node's live state.
//
// A refusal is returned as a reason rather than an error, because it is not a
// failure: the control plane reads it and asks placement for a different node.
func (n *Node) Admit(ctx context.Context, req *v1.AdmitRequest) (*v1.AdmitResponse, error) {
	spec := specFromProto(req.GetSpec())
	n.admission.SetPressure(n.pressure())
	owner := ownerOf(spec)
	// Charge what a sandbox of this class actually costs rather than its declared maximum.
	// The maximum stays as the kernel's hard bound for this sandbox; the ledger's job is to
	// know what the node has promised, and nothing has promised the maximum.
	//
	// CPU is charged the same way, and it is the dimension that decides how large a burst
	// the fleet admits: every sandbox declares the cores one tool call may want, and nearly
	// all of them are idle between calls.
	// Charge the class's measured price plus the growth it is committed to, because the
	// question admission answers is not "does this fit now" but "will it still fit when
	// it has grown, and will the ones before it".
	charge := n.elasticCharge(spec)
	if growth := n.admission.GrowthPerSandbox(spec.ResourceClass); growth.MemoryMB > 0 || growth.CPUMillis > 0 {
		if charge.MemoryMB > 0 {
			charge.MemoryMB += growth.MemoryMB
		}
		if charge.CPUMillis > 0 {
			charge.CPUMillis += growth.CPUMillis
		}
	}
	grant, refusal, ok := n.admission.Reserve(spec.ResourceClass, owner, charge)
	if !ok {
		// The refusal is what makes this class's guarantee real. Without it the node has
		// no record of a class waiting, so nothing is held back from the classes that are
		// running, and the first one to arrive can take the whole envelope -- which is how
		// grading sandboxes ended up un-admittable behind a full rollout wave.
		n.admission.Enqueue(spec.ResourceClass)
		n.admission.ExpireDemand()
		return &v1.AdmitResponse{Admitted: false, Refusal: string(refusal)}, nil
	}
	// One waiter of this class is served, so its claim on the guarantee is released. The
	// demand that is still waiting keeps the share held.
	n.admission.Dequeue(spec.ResourceClass)
	// The record is created from the grant, so a grant that never becomes a
	// container is still swept rather than held until the next restart.
	n.lifecycle.Reserve(grant, owner, spec.ExecMode, lifetimeOf(spec))
	return &v1.AdmitResponse{Admitted: true, LeaseId: grant.LeaseID, GpuIndices: grant.GPUIndices}, nil
}

// CreateOn provisions a sandbox against an admission this node already granted.
//
// The lease must exist: creating without one would put a sandbox on the node
// that its own accounting never charged. The runtime is named by the caller
// rather than chosen here, because routing already matched the spec's
// requirements against one and a second choice could disagree with it.
func (n *Node) CreateOn(ctx context.Context, req *v1.CreateOnRequest) (*v1.CreateResponse, error) {
	leaseID := req.GetLeaseId()
	grant, claimed := n.lifecycle.Claim(leaseID)
	if !claimed {
		return nil, status.Errorf(codes.FailedPrecondition,
			"lease %q was not granted by this node or has already been used, "+
				"so a sandbox created against it would not be charged", leaseID)
	}
	// The record is held in flight for as long as this runs, so the sweep's clock cannot
	// reap a create that is still pulling its image.
	release := n.lifecycle.BeginCreate(leaseID)
	defer release()
	hosted, hosts := n.hosted(req.GetBackend())
	if !hosts {
		// No sandbox exists, so the reserved budget must not stay held.
		n.lifecycle.FailCreate(leaseID)
		return nil, status.Errorf(codes.FailedPrecondition,
			"node %q does not host backend %q (hosts: %v)", n.nodeID, req.GetBackend(), n.order)
	}
	spec := specFromProto(req.GetSpec())
	spec.AssignedGPUs = grant.GPUIndices
	spec.Lifetime = lifetimeOf(spec)
	// The container carries the owner as a label, so a restarted node recovers it
	// from the daemon alone.
	owner := ownerOf(spec)
	if spec.Metadata == nil {
		spec.Metadata = map[string]string{}
	}
	spec.Metadata[backend.OwnerMetadataKey] = owner

	// The container is born under its guarantee, not under its declared maximum. Doing it
	// here rather than by lowering the limit afterwards means there is no window in which
	// the sandbox is bounded by a number the node has not agreed to back.
	//
	// The lowered bound goes into a copy. The spec itself keeps what the caller declared,
	// because that is the ceiling growth may reach: remembering the ceiling from the copy
	// would record the guarantee as the maximum, and a sandbox with nowhere to grow is
	// exactly the one this exists to help.
	createSpec := n.applyStartLimit(spec)

	if handled, found, err := n.adoptByIdempotencyKey(ctx, hosted, leaseID, spec.IdempotencyKey); found {
		return handled, err
	}

	created, err := hosted.Create(ctx, n.nodeID, createSpec, req.GetCallbackTarget())
	if err != nil {
		// The sandbox does not exist, so its admission must not stay charged.
		n.lifecycle.FailCreate(leaseID)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	// The ceiling is the declared maximum even though the container was created under the
	// guarantee and the ledger charges the class's estimate, so the controller needs all
	// three numbers -- for CPU too, because a throttled quota is restored to what the
	// sandbox declared.
	n.elasticRemember(created.Handle.SandboxID, spec)
	if !n.lifecycle.AdoptCreated(leaseID, created.Handle, hosted) {
		// The record was reclaimed while the create was in flight, so this
		// container has no owner to charge it to. Destroy it on a budget of its
		// own rather than leak it or release capacity it still holds.
		return nil, destroyReclaimed(hosted, created.Handle)
	}
	return createdToProto(created), nil
}

// adoptByIdempotencyKey resolves a create whose key names a container that
// already exists. The second result is whether this retry is resolved by the
// answer, which is true both when a container was found and when asking failed:
// either way the caller must not go on to create one.
//
// A create whose response was lost leaves a container the caller never got a
// handle for, and its key is on the container. So the recovery is to ask the
// runtime rather than to build a second container, which would double the
// memory a single logical sandbox holds.
//
// The reservation is the thing that has to be unwound. When the node still owns
// the container, the record from the first attempt is the authority and this
// retry's grant is returned, so one container stays charged once. When it does
// not, the container outlived a restart and this retry is its first record.
func (n *Node) adoptByIdempotencyKey(
	ctx context.Context, hosted backend.Backend, leaseID, key string,
) (*v1.CreateResponse, bool, error) {
	if key == "" {
		return nil, false, nil
	}
	finder, canFind := hosted.(backend.Reconciler)
	if !canFind {
		// A runtime that cannot look an object up by its key cannot be asked, and
		// a missed retry is a smaller fault than a create that never happens.
		return nil, false, nil
	}
	found, exists, err := finder.FindByIdempotencyKey(ctx, key)
	if err != nil {
		n.lifecycle.FailCreate(leaseID)
		return nil, true, status.Error(codes.Unavailable, err.Error())
	}
	if !exists {
		return nil, false, nil
	}
	if owned, hostedBy, held := n.lifecycle.Lookup(found.Handle.SandboxID); held {
		n.lifecycle.FailCreate(leaseID)
		return createdToProto(backend.Created{
			Handle: owned, Capabilities: hostedBy.Capabilities(),
		}), true, nil
	}
	if !n.lifecycle.AdoptCreated(leaseID, found.Handle, hosted) {
		return nil, true, destroyReclaimed(hosted, found.Handle)
	}
	return createdToProto(backend.Created{
		Handle: found.Handle, Capabilities: hosted.Capabilities(),
	}), true, nil
}

// destroyReclaimed destroys a container whose record was reclaimed while the
// create was in flight, so a create that lost its race leaks nothing and
// returns no capacity the container still holds.
func destroyReclaimed(hosted backend.Backend, handle backend.Handle) error {
	cleanupCtx, cancel := context.WithTimeout(context.Background(), createCleanupTimeout)
	defer cancel()
	if err := hosted.Release(cleanupCtx, handle); err != nil {
		return status.Errorf(codes.Internal,
			"sandbox %s was reclaimed while its create was in flight and could not be destroyed: %v",
			handle.SandboxID, err)
	}
	return status.Errorf(codes.Aborted,
		"sandbox %s was reclaimed while its create was in flight", handle.SandboxID)
}

// createCleanupTimeout bounds destroying a sandbox whose create lost its record.
const createCleanupTimeout = 30 * time.Second

// lifetimeOf is how long a sandbox may exist, defaulting a spec that states none
// so a caller cannot create an unbounded one.
func lifetimeOf(spec backend.Spec) time.Duration {
	if spec.Lifetime > 0 {
		return spec.Lifetime
	}
	return node.DefaultLifetime
}

// hosted returns one runtime by name. An empty name takes the first declared,
// which is what a single-runtime node reports.
func (n *Node) hosted(name string) (backend.Backend, bool) {
	if name == "" {
		return n.backends[n.order[0]], true
	}
	hosted, hosts := n.backends[name]
	return hosted, hosts
}

// ReleaseOn destroys one sandbox this node holds.
func (n *Node) ReleaseOn(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	if err := n.lifecycle.Release(ctx, handle.GetSandboxId(), node.ExitReleased); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

// LedgerAudit reports whether the node's charge can still be accounted for by its
// envelope.
//
// Granted above Envelope is not a full node. Every admission is checked against the
// headroom that is left, so the charge can only pass the envelope if something changed
// it without going through an admission -- which means the ledger is charging capacity
// to leases it no longer holds, and every request from then on is refused against
// capacity nobody is using. It is reported rather than corrected: guessing at which
// lease is wrong would trade a visible fault for an invisible one.
func (n *Node) LedgerAudit() (granted node.Resources, envelope node.Resources, accounted bool) {
	report := n.admission.Snapshot()
	return report.Granted, report.Envelope, report.Granted.FitsIn(report.Envelope)
}

// Report is this node's view of itself, which the monitor collects.
//
// Headroom is reported per class because that is the granularity admission
// works at: a scheduler given the envelope remainder sends work this node will
// refuse, and the mismatch shows up only as sandboxes that never start.
func (n *Node) Report(context.Context, *v1.Empty) (*v1.NodeView, error) {
	view, _ := n.view()
	return view, nil
}

// throttledPct is the share of the last interval the sandboxes spent stopped for
// exceeding their CPU quota, for the wire.
//
// Read from the controller rather than recomputed, so the number the fleet publishes is
// the same number admission used to decide the envelope.
func (n *Node) throttledPct() float64 {
	// Guarded before the lock: a node with the controller off has no controller to lock.
	if n.elastic == nil {
		return 0
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	return n.elastic.lastThrottledPct
}

func (n *Node) view() (*v1.NodeView, node.LifecycleReport) {
	// One sample per report, used for both admission and the wire fields. Calling it
	// twice made the reported CPU figure a rate over the microseconds between two
	// RPC-internal calls, which is noise, and it disagreed with what admission stored.
	pressure := n.pressure()
	n.admission.SetPressure(pressure)
	report := n.admission.Snapshot()
	life := n.lifecycle.Snapshot()
	out := &v1.NodeView{
		NodeId:        n.nodeID,
		ClassHeadroom: map[string]*v1.Headroom{},
		Envelope:      headroomToProto(report.EffectiveEnvelope),
		// What the node promised, what it measured, and whether the kernel is refusing
		// work. Published together because the utilisation above cannot answer the
		// question this controller is judged on: is it refusing because the machine is
		// full, or because it is charging more than the sandboxes use? Identical from
		// the measured side, opposite in what to do about them.
		Granted:       headroomToProto(report.Granted),
		Observed:      headroomToProto(report.Observed),
		ThrottledPct:  n.throttledPct(),
		LiveSandboxes: int32(life.Resident),
		CpuUsedPct:    pressure.CPUUsedPct,
		MemUsedPct:    pressure.MemUsedPct,
		GpuFree:       int32(report.FreeGPUs),
		Draining:      report.Draining,
	}
	for class, headroom := range n.admission.ClassHeadroom() {
		out.ClassHeadroom[class] = headroomToProto(headroom)
	}
	// Per runtime, because a node hosting a container daemon and a microVM
	// satisfies a full-state resume through one of them and not the other.
	for _, name := range n.order {
		capabilities := n.backends[name].Capabilities()
		hosted := &v1.BackendView{
			Name:        name,
			ResumeLevel: resumeLevelValue(capabilities.ResumeLevel),
		}
		for _, feature := range capabilities.Features {
			hosted.Features = append(hosted.Features, featureValue(feature))
			if feature == "host_mount" {
				hosted.HostMounts = true
			}
		}
		out.Backends = append(out.Backends, hosted)
	}
	return out, life
}

// View returns this node's state as placement reads it, for an in-process
// deployment where the monitor and the node share a process.
func (n *Node) View() placement.NodeView {
	proto, life := n.view()
	view := placement.NodeView{
		NodeID:                proto.GetNodeId(),
		SeenAt:                time.Now(),
		ClassHeadroom:         map[string]placement.Headroom{},
		LiveSandboxes:         int(proto.GetLiveSandboxes()),
		CPUUsedPct:            proto.GetCpuUsedPct(),
		MemUsedPct:            proto.GetMemUsedPct(),
		GPUFree:               proto.GetGpuFree(),
		Draining:              proto.GetDraining(),
		ImageDigests:          map[string]struct{}{},
		ImageReferences:       map[string]struct{}{},
		Labels:                map[string]struct{}{},
		ClassReserveMB:        n.classMemoryReservesMB(),
		ClassReserveCPUMillis: n.classCPUReservesMillis(),
	}
	// The inventory travels with the sequence it was taken under, so the
	// control plane can tell a report generated before a grant existed from one
	// generated after it.
	view.ReportSeq = life.ReportSeq
	view.InventoryReported = life.InventoryReported
	view.InventoryTruncated = life.InventoryTruncated
	view.LiveSandboxIDs = life.LiveSandboxIDs
	view.LiveOwners = life.LiveOwners
	view.Diagnostics = map[string]float64{
		"lifecycle/resident":       float64(life.Resident),
		"lifecycle/busy":           float64(life.Busy),
		"lifecycle/destroying":     float64(life.Destroying),
		"lifecycle/cleanup_failed": float64(life.CleanupFailed),
		"lifecycle/failures":       float64(life.Failures),
	}
	for reason, count := range life.Exits {
		view.Diagnostics["lifecycle/exit/"+string(reason)] = float64(count)
	}
	checks, gone, probesFailed, probeErrors := n.lifecycle.ProbeCounters()
	view.Diagnostics["probe/checks"] = float64(checks)
	view.Diagnostics["probe/gone"] = float64(gone)
	view.Diagnostics["probe/failures"] = float64(probesFailed)
	view.Diagnostics["probe/errors"] = float64(probeErrors)
	if envelope := proto.GetEnvelope(); envelope != nil {
		view.Envelope = placement.Headroom{
			MemoryMB: envelope.GetMemoryMb(), CPUMillis: envelope.GetCpuMillis(),
			GPUCount: envelope.GetGpuCount(), DiskMB: envelope.GetDiskMb(),
		}
	}
	// The ledger crosses into placement for the same reason it crosses onto the wire: a
	// scheduler, an operator and a test all need to see what the node promised beside
	// what it measured. Placement itself does not consume these -- it ranks on class
	// headroom -- but the fleet report is built from this view, and dropping them here
	// is what made the controller unverifiable from outside.
	if granted := proto.GetGranted(); granted != nil {
		view.Granted = placement.Headroom{
			MemoryMB: granted.GetMemoryMb(), CPUMillis: granted.GetCpuMillis(),
			GPUCount: granted.GetGpuCount(), DiskMB: granted.GetDiskMb(),
		}
	}
	if observed := proto.GetObserved(); observed != nil {
		view.Observed = placement.Headroom{
			MemoryMB: observed.GetMemoryMb(), CPUMillis: observed.GetCpuMillis(),
			GPUCount: observed.GetGpuCount(), DiskMB: observed.GetDiskMb(),
		}
	}
	view.ThrottledPct = proto.GetThrottledPct()
	for class, headroom := range proto.GetClassHeadroom() {
		view.ClassHeadroom[class] = placement.Headroom{
			MemoryMB: headroom.GetMemoryMb(), CPUMillis: headroom.GetCpuMillis(),
			GPUCount: headroom.GetGpuCount(), DiskMB: headroom.GetDiskMb(),
		}
	}
	for _, name := range n.order {
		capabilities := n.backends[name].Capabilities()
		hosted := placement.BackendCapability{
			Name:        name,
			Features:    make(map[string]struct{}, len(capabilities.Features)),
			ResumeLevel: capabilities.ResumeLevel,
		}
		for _, feature := range capabilities.Features {
			hosted.Features[feature] = struct{}{}
			if feature == "host_mount" {
				hosted.HostMounts = true
			}
		}
		view.Backends = append(view.Backends, hosted)
	}
	return view
}

// Sweep runs one reclamation pass on demand, for an operator or a test. The
// node also sweeps on its own cadence.
func (n *Node) Sweep(ctx context.Context, _ *v1.Empty) (*v1.SweepReport, error) {
	report := n.lifecycle.Sweep(ctx)
	return &v1.SweepReport{
		ReapedLifetime:  report.ReapedLifetime,
		ReclaimedOrphan: report.ReclaimedOrphan,
		SkippedBusy:     int32(report.SkippedBusy),
		Failures:        int32(report.Failures),
	}, nil
}

// sessionGoneError is a command against a sandbox this node no longer holds.
//
// It carries the lifecycle's attribution when one is still known, and maps to
// gRPC NotFound so an SDK classifies it as a lost session rather than a
// transport fault. `reason` is empty when the node never held the sandbox.
//
// `cause` is the runtime's own account of a session that ended under a sandbox
// the node still holds -- a persistent shell that died, for instance. It is kept
// apart from the attribution because nothing exited: the sandbox is alive and the
// caller needs to know that, not to be told it was never here.
type sessionGoneError struct {
	sandboxID  string
	nodeID     string
	reason     string
	cause      string
	exitCode   int
	codeKnown  bool
	finished   time.Time
	stateError string
}

// sessionGone builds the failure from the lifecycle's attribution. Keeping the
// construction in one place is what makes the exit code and finish time travel
// with every path that discovers a loss, rather than only the one that happens
// to remember them.
func sessionGone(sandboxID, nodeID string, attribution node.ExitAttribution) *sessionGoneError {
	return &sessionGoneError{
		sandboxID:  sandboxID,
		nodeID:     nodeID,
		reason:     string(attribution.Reason),
		exitCode:   attribution.ExitCode,
		codeKnown:  attribution.ExitCodeKnown,
		finished:   attribution.FinishedAt,
		stateError: attribution.StateError,
	}
}

// sessionGoneWithCause builds the failure for a sandbox the node still holds
// whose command session ended, naming what happened to the session. The caller
// must replace the session, not the sandbox, and "not held" would tell it the
// opposite.
func sessionGoneWithCause(sandboxID, nodeID string, cause error) *sessionGoneError {
	gone := sessionGone(sandboxID, nodeID, node.ExitAttribution{})
	gone.cause = cause.Error()
	return gone
}

func (e *sessionGoneError) Error() string {
	message := fmt.Sprintf("sandbox %q is not held by node %q", e.sandboxID, e.nodeID)
	if e.cause != "" {
		message = fmt.Sprintf(
			"sandbox %q is held by node %q, but its command session was lost: %s",
			e.sandboxID, e.nodeID, e.cause,
		)
	}
	var details []string
	if e.reason != "" {
		details = append(details, "exit_reason="+e.reason)
	}
	if e.codeKnown {
		details = append(details, fmt.Sprintf("exit_code=%d", e.exitCode))
	} else if e.reason != "" {
		details = append(details, "exit_code=unknown")
	}
	if !e.finished.IsZero() {
		details = append(details, "finished_at="+e.finished.UTC().Format(time.RFC3339Nano))
	}
	if e.stateError != "" {
		details = append(details, fmt.Sprintf("state_error=%q", e.stateError))
	}
	if len(details) > 0 {
		message += " (" + strings.Join(details, ", ") + ")"
	}
	return message
}

// ExitReason exposes the attribution to the wire layer without that layer
// having to type-switch on this package's concrete error.
func (e *sessionGoneError) ExitReason() string { return e.reason }

// ExitCode reports the runtime's exit code, when one was observed.
func (e *sessionGoneError) ExitCode() (int, bool) { return e.exitCode, e.codeKnown }

// FinishedAt reports when the runtime observed the stop, when it did.
func (e *sessionGoneError) FinishedAt() time.Time { return e.finished }

// StateError exposes the runtime evidence or the inspect failure behind the
// attribution.
func (e *sessionGoneError) StateError() string { return e.stateError }

// WireCode names the caller-facing classification. It is checked before the
// gRPC status so a lost session stays a lost session across every hop.
func (e *sessionGoneError) WireCode() string { return "session_lost" }

func (e *sessionGoneError) GRPCStatus() *status.Status {
	return status.New(codes.NotFound, e.Error())
}

// pathNotFoundError is a file operation against a path that is not present in a
// sandbox this node still holds.
//
// It exists because a missing file is not a missing sandbox. Both are Docker
// 404s, and the old code mapped both to a lost session, which made a failed
// harness report say the sandbox had died when the real problem was a file the
// agent removed.
type pathNotFoundError struct {
	path string
	err  error
}

func (e *pathNotFoundError) Error() string {
	return fmt.Sprintf("path %q is not present in the sandbox: %v", e.path, e.err)
}

func (e *pathNotFoundError) Unwrap() error { return e.err }

// WireCode names the caller-facing classification, checked before the status.
func (e *pathNotFoundError) WireCode() string { return "path_not_found" }

func (e *pathNotFoundError) GRPCStatus() *status.Status {
	return status.New(codes.NotFound, e.Error())
}

// Exec runs one command in a sandbox this node holds.
//
// It routes to the exec strategy the sandbox was created with. A persistent
// shell keeps shell state (working directory, environment, activated venv)
// alive between calls, which is what a multi-turn harness needs. One-shot
// starts a fresh process per call, which is cheaper and correct for a grader.
// The strategy is chosen at sandbox creation time and stored on the lifecycle
// resident so this path cannot diverge from what the spec asked for.
func (n *Node) Exec(
	ctx context.Context, handle backend.Handle, command, workdir string, env map[string]string, opts backend.ExecOptions,
) (backend.ExecResult, error) {
	_, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		// The sandbox is gone. Report why it left when the lifecycle still
		// remembers, so the caller can tell a reaped sandbox from one that was
		// never here -- and gets a lost-session rather than a transport fault.
		attribution, _ := n.lifecycle.ExitAttributionOf(handle.SandboxID)
		return backend.ExecResult{}, sessionGone(handle.SandboxID, n.nodeID, attribution)
	}
	// Mark the sandbox busy for the whole command, so the reclamation sweep
	// cannot read a long command as idle and pause a running test suite.
	done := n.lifecycle.Begin(handle.SandboxID)
	defer done()
	// Data-plane traffic is the strongest liveness evidence the node has: it
	// observes the command itself. Refreshing the owner here is what keeps a
	// sandbox alive when the control plane's heartbeat is delayed by another
	// node, without adding a second protocol.
	n.renewOwnerOf(handle.SandboxID)

	executor, canRun := held.(backend.Executor)
	if !canRun {
		// A backend whose sandboxes run their own in-sandbox agent is never
		// reached here; the SDK contacts the agent directly.
		return backend.ExecResult{}, fmt.Errorf("backend %q runs commands through its own in-sandbox agent", held.Name())
	}
	execMode := n.lifecycle.ExecMode(handle.SandboxID)
	// A command start is the earliest evidence of a burst. Reserve the burst limit
	// before the command runs; the usage sampler is already too late by the time a
	// `git diff` or a compiler reaches the bound.
	n.PrepareForCommand(ctx, handle.SandboxID)
	defer n.FinishCommand(handle.SandboxID)
	result, err := executor.Exec(ctx, handle, execMode, command, workdir, env, opts)
	if err == nil {
		return result, nil
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		// The command ran out of its budget and the backend stopped it. The shell
		// session is closed, so the sandbox survives and the next command gets a
		// fresh one: this is a fact about the command, not a transport fault, and a
		// caller that reads it as one abandons work the sandbox could still grade.
		return result, status.Errorf(
			codes.DeadlineExceeded, "command exceeded its budget and was stopped: %v", err,
		)
	}
	return result, n.classifyRuntimeFailure(ctx, held, handle, err)
}

// classifyRuntimeFailure turns a backend command error into the attribution the
// caller acts on, and updates the node's ledger when the runtime is gone.
func (n *Node) classifyRuntimeFailure(
	ctx context.Context, held backend.Backend, handle backend.Handle, err error,
) error {
	switch {
	case errors.Is(err, backend.ErrRuntimeGone):
		attribution := n.runtimeGoneAttribution(ctx, held, handle)
		n.lifecycle.MarkLost(handle.SandboxID, attribution)
		return sessionGone(handle.SandboxID, n.nodeID, attribution)
	case errors.Is(err, backend.ErrSessionLost):
		// The command's shell is gone, but the container may still be healthy. The
		// caller replaces the sandbox; the node does not release a runtime it has no
		// evidence is dead, so there is no exit code to report. The runtime's own
		// account travels with the loss, because a caller told only "not held"
		// cannot tell a session that ended from a sandbox that was never there.
		return sessionGoneWithCause(handle.SandboxID, n.nodeID, err)
	case errors.Is(err, backend.ErrPathNotFound):
		return &pathNotFoundError{path: "", err: err}
	default:
		return err
	}
}

// runtimeGoneAttribution asks the runtime why it is gone, with a short budget.
// A failed or slow probe answers "exited" with no code: the caller still needs
// the sandbox replaced, and an unknown cause is not a reason to leave the charge
// in place -- but neither is it a reason to invent an exit code.
func (n *Node) runtimeGoneAttribution(
	ctx context.Context, held backend.Backend, handle backend.Handle,
) node.ExitAttribution {
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	var inspectErr error
	if detailer, ok := held.(backend.ExitDetailer); ok {
		if detail, err := detailer.ExitDetail(probeCtx, handle); err == nil {
			attribution := node.ExitAttribution{
				Reason:        node.ExitExited,
				ExitCode:      detail.ExitCode,
				ExitCodeKnown: detail.ExitCodeKnown,
				FinishedAt:    detail.FinishedAt,
				StateError:    exitEvidence(detail),
			}
			if detail.OOMKilled {
				attribution.Reason = node.ExitOOM
				if !attribution.ExitCodeKnown || attribution.ExitCode == 0 {
					attribution.ExitCode = 137
					attribution.ExitCodeKnown = true
				}
			}
			if attribution.FinishedAt.IsZero() {
				attribution.FinishedAt = time.Now()
			}
			return attribution
		} else {
			inspectErr = err
		}
	} else {
		inspectErr = fmt.Errorf("backend %q does not implement ExitDetailer", held.Name())
	}
	status, statusErr := held.Status(probeCtx, handle)
	if statusErr != nil {
		return node.ExitAttribution{
			Reason:     node.ExitExited,
			FinishedAt: time.Now(),
			StateError: fmt.Sprintf("exit_detail_error=%q status_error=%q", inspectErr, statusErr),
		}
	}
	reason := node.ExitExited
	switch status {
	case "oom":
		reason = node.ExitOOM
	case "terminated":
		reason = node.ExitRuntimeGone
	}
	attribution := node.ExitAttribution{
		Reason:     reason,
		FinishedAt: time.Now(),
		StateError: fmt.Sprintf("exit_detail_error=%q status=%q", inspectErr, status),
	}
	if reason == node.ExitOOM {
		attribution.ExitCode, attribution.ExitCodeKnown = 137, true
	}
	return attribution
}

// exitEvidence is the daemon's own account of a stopped sandbox, in one line.
// It is what lets a post mortem tell an entrypoint that exited cleanly from a
// runtime kill when the service's reason alone says only "exited".
func exitEvidence(detail backend.ExitDetail) string {
	return detail.Evidence()
}

// renewOwnerOf refreshes the owner's liveness on this node. An owner this node
// never admitted is ignored by the admission gate, so a stale sandbox id is
// harmless.
func (n *Node) renewOwnerOf(sandboxID string) {
	if owner, ok := n.lifecycle.OwnerOf(sandboxID); ok {
		n.admission.Renew(owner)
	}
}

// ReadBytes reads one file from a sandbox this node holds, preserving the
// distinction between a missing file and a missing sandbox.
func (n *Node) ReadBytes(ctx context.Context, handle backend.Handle, path string) ([]byte, error) {
	_, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		attribution, _ := n.lifecycle.ExitAttributionOf(handle.SandboxID)
		return nil, sessionGone(handle.SandboxID, n.nodeID, attribution)
	}
	n.renewOwnerOf(handle.SandboxID)
	if files, ok := held.(backend.FileHandler); ok {
		data, err := files.ReadFile(ctx, handle, path)
		if err == nil {
			return data, nil
		}
		switch {
		case errors.Is(err, backend.ErrPathNotFound):
			return nil, &pathNotFoundError{path: path, err: err}
		case errors.Is(err, backend.ErrRuntimeGone):
			return nil, n.classifyRuntimeFailure(ctx, held, handle, err)
		case errors.Is(err, backend.ErrSessionLost):
			return nil, n.classifyRuntimeFailure(ctx, held, handle, err)
		default:
			return nil, err
		}
	}
	// Shell fallback for a backend with no file API. A non-zero exit is the
	// only signal a shell can give, and for a read it means the path did not
	// exist or could not be read; both are path-level facts.
	executor, ok := held.(backend.Executor)
	if !ok {
		return nil, fmt.Errorf("backend %q has neither a file API nor a command executor", held.Name())
	}
	done := n.lifecycle.Begin(handle.SandboxID)
	defer done()
	result, err := executor.Exec(ctx, handle, n.lifecycle.ExecMode(handle.SandboxID),
		fmt.Sprintf("base64 %q", path), "", nil, backend.ExecOptions{})
	if err != nil {
		return nil, n.classifyRuntimeFailure(ctx, held, handle, err)
	}
	if result.ExitCode != 0 {
		return nil, &pathNotFoundError{path: path, err: fmt.Errorf("command exited %d: %s", result.ExitCode, result.Stderr)}
	}
	decoded, err := base64.StdEncoding.DecodeString(compactBase64(result.Stdout))
	if err != nil {
		return nil, fmt.Errorf("reading %q: the shell fallback produced invalid base64: %w", path, err)
	}
	return decoded, nil
}

// WriteBytes writes one file into a sandbox this node holds.
func (n *Node) WriteBytes(ctx context.Context, handle backend.Handle, path string, data []byte) error {
	_, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		attribution, _ := n.lifecycle.ExitAttributionOf(handle.SandboxID)
		return sessionGone(handle.SandboxID, n.nodeID, attribution)
	}
	n.renewOwnerOf(handle.SandboxID)
	if files, ok := held.(backend.FileHandler); ok {
		if err := files.WriteFile(ctx, handle, path, data); err != nil {
			return n.classifyRuntimeFailure(ctx, held, handle, err)
		}
		return nil
	}
	executor, ok := held.(backend.Executor)
	if !ok {
		return fmt.Errorf("backend %q has neither a file API nor a command executor", held.Name())
	}
	done := n.lifecycle.Begin(handle.SandboxID)
	defer done()
	command := fmt.Sprintf("mkdir -p \"$(dirname %q)\" && printf %%s %q | base64 -d > %q",
		path, base64.StdEncoding.EncodeToString(data), path)
	result, err := executor.Exec(
		ctx, handle, n.lifecycle.ExecMode(handle.SandboxID), command, "", nil, backend.ExecOptions{},
	)
	if err != nil {
		return n.classifyRuntimeFailure(ctx, held, handle, err)
	}
	if result.ExitCode != 0 {
		return status.Errorf(codes.FailedPrecondition, "writing %q exited %d: %s", path, result.ExitCode, result.Stderr)
	}
	return nil
}

// Touch records activity on a sandbox this node holds.
//
// The SDK calls it for a sandbox whose commands go straight to an in-container
// agent, which never reaches Exec; without it the reclamation sweep would judge
// that sandbox idle from creation and pause it mid-episode.
func (n *Node) Touch(handle backend.Handle) {
	n.lifecycle.Touch(handle.SandboxID)
	n.renewOwnerOf(handle.SandboxID)
}

// RenewOwner records that an owner is still using this node's sandboxes.
//
// The reclamation sweep reads owner liveness from the admission gate, so a
// heartbeat is what keeps a live caller's sandboxes from being reclaimed as
// orphans. An owner this node never admitted is ignored: a renewal for another
// node's caller must not resurrect anything here.
func (n *Node) RenewOwner(owner string) {
	n.admission.Renew(owner)
}

// StatusOn reports a sandbox's state from this node.
func (n *Node) StatusOn(ctx context.Context, handle backend.Handle) (string, error) {
	stored, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		return "terminated", nil
	}
	return held.Status(ctx, stored)
}

// ownerOf names the caller whose liveness decides when a sandbox is an orphan.
//
// The caller states it in the spec's metadata, because that is the identity the
// reclaimer is keyed on and nothing else on the spec carries it: `workflow_id`
// reserves a scheduling phase, which a caller may share across sandboxes or not
// set at all, and reading ownership out of it left every sandbox owned by
// "anonymous", an owner that never renews, so a silence budget would reclaim
// live work. `workflow_id` stays a fallback for a caller that has no separate
// owner and means its workflow to be one.
func ownerOf(spec backend.Spec) string {
	if owner := spec.Metadata[backend.OwnerMetadataKey]; owner != "" {
		return owner
	}
	if spec.WorkflowID != "" {
		return spec.WorkflowID
	}
	return "anonymous"
}
