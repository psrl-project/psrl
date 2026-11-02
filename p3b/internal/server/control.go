package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// NodeClient reaches one node agent. It is an interface so the control plane can
// be driven against an in-process node in a test and over gRPC in a cluster,
// with the same code path either way.
type NodeClient interface {
	Admit(ctx context.Context, nodeID string, spec backend.Spec) (leaseID string, gpus []int32, refusal string, err error)
	// CreateOn names the runtime to provision on, because one node can host
	// several and routing already matched the spec against one of them.
	CreateOn(
		ctx context.Context, nodeID, leaseID, backendName string, spec backend.Spec, callback string,
	) (backend.Created, error)
	ReleaseOn(ctx context.Context, handle backend.Handle) error
	StatusOn(ctx context.Context, handle backend.Handle) (string, error)
	// RenewOwnerOn tells one node that an owner is still running, so the node
	// does not reclaim that owner's sandboxes as orphans.
	RenewOwnerOn(ctx context.Context, nodeID, owner string) error

	// ExecOn, ReadBytesOn, and WriteBytesOn route command and file traffic to
	// the node that holds the sandbox. In a single-process deployment they call
	// the node directly; in a fleet they reach the right machine over the
	// node-plane protocol so the control plane never proxies data traffic.
	ExecOn(
		ctx context.Context, handle backend.Handle, command, cwd string, env map[string]string, opts backend.ExecOptions,
	) (backend.ExecResult, error)
	ReadBytesOn(ctx context.Context, handle backend.Handle, path string) (string, error)
	WriteBytesOn(ctx context.Context, handle backend.Handle, path, data string) error
}

// Control is the cluster-level service.
//
// It routes, admits against the cross-backend quota, places, and then asks the
// chosen node to accept. The node keeps the last word, so a refusal here is a
// reason to ask placement again rather than a failed episode.
type Control struct {
	v1.UnimplementedSandboxControlServer

	registry  *backend.Registry
	ledger    *quota.Ledger
	placement *placement.Service
	monitor   *monitor.Monitor
	nodes     NodeClient

	// placementAttempts bounds how many nodes one create will try before giving
	// up. A node refusing is normal under load; an unbounded retry would hide a
	// fleet that is genuinely full behind a slow create.
	placementAttempts int
	// admitPollInterval is how often a queued request re-tries its class. Short
	// against a sandbox lifetime, so a freed share is picked up promptly, and long
	// enough that a queue of waiters is not a spin on the ledger's lock.
	admitPollInterval time.Duration
	// reservationRenewInterval is how often a create renews its placement
	// reservation while it provisions. A cold image pull can dwarf the reservation
	// TTL, and a reservation that lapses mid-provision lets placement send another
	// sandbox to a node this one is still about to occupy.
	reservationRenewInterval time.Duration

	mu      sync.Mutex
	sandbox map[string]sandboxRecord
	// lastSeq is the newest inventory report applied per node. A report is only
	// evidence about grants that existed before it was generated, so a grant is
	// armed two sequences past the last applied one.
	lastSeq map[string]uint64
	// provisions is what a create with an idempotency key is doing, so a caller
	// that has no response yet can tell "still working" from "lost the request".
	provisions map[string]*provisionRecord
	// pricing is the reserve this control last charged a class, so a class whose
	// price moves is stated once instead of once per request.
	pricing map[string]int64
	log     *slog.Logger

	// Owner renewal is a liveness signal, and its own health is reported beside
	// the fleet's: a heartbeat that never reaches a node is invisible from the
	// node side, which is exactly the failure this accounting exists to expose.
	renewAttempts  atomic.Int64
	renewFailures  atomic.Int64
	renewNodes     atomic.Int64
	renewLastNanos atomic.Int64
}

// provisionRecord is what a create with an idempotency key is doing.
//
// As coarse as the service can honestly be: it advances when the control plane
// reaches a new phase -- registering, then done -- rather than while a node pulls
// an image, because that is one blocking call from here. It is enough to tell a
// caller its request is being worked on rather than quietly dropped.
type provisionRecord struct {
	state     string
	startedAt time.Time
	updatedAt time.Time
}

// provisionRetention is how long a finished create stays queryable, so a caller
// that polled during the create can still read the outcome just after.
const provisionRetention = 10 * time.Minute

type sandboxRecord struct {
	handle        backend.Handle
	backend       backend.Backend
	grantID       string
	reservationID string
	class         string
	// owner is the caller identity the node keys liveness on. Renewal targets
	// the nodes that hold this owner's sandboxes, not every node the fleet has
	// ever heard from.
	owner     string
	createdAt time.Time
	// Reconciliation state: a grant is released only after two inventory
	// reports that were generated after it was recorded both fail to list it.
	misses        int
	armedAfterSeq uint64
	// charged is what this sandbox cost the fleet when it was admitted, and declared is
	// the maximum its caller asked for. Both are kept so the grant can be repriced: the
	// class's price is a measurement that moves, and a fleet still charging the
	// admission-time number promises capacity no node will honour.
	charged  quota.Amount
	declared quota.Amount
}

// ControlConfig is what the control plane needs to run.
type ControlConfig struct {
	Registry          *backend.Registry
	Ledger            *quota.Ledger
	Placement         *placement.Service
	Monitor           *monitor.Monitor
	Nodes             NodeClient
	PlacementAttempts int
	// AdmitPollInterval overrides how often a request queued on a full class
	// re-tries. Zero takes the default.
	AdmitPollInterval time.Duration
	// ReservationRenewInterval overrides how often a create renews its placement
	// reservation while provisioning. Zero takes the default.
	ReservationRenewInterval time.Duration
	// Logger states pricing changes to an operator. Zero takes slog's default.
	Logger *slog.Logger
}

// NewControl returns the cluster-level server.
func NewControl(cfg ControlConfig) (*Control, error) {
	if cfg.Registry == nil || cfg.Ledger == nil || cfg.Placement == nil || cfg.Monitor == nil {
		return nil, fmt.Errorf("the control plane needs a registry, a ledger, placement, and a monitor")
	}
	if cfg.PlacementAttempts <= 0 {
		cfg.PlacementAttempts = 3
	}
	if cfg.AdmitPollInterval <= 0 {
		cfg.AdmitPollInterval = 250 * time.Millisecond
	}
	if cfg.ReservationRenewInterval <= 0 {
		cfg.ReservationRenewInterval = 20 * time.Second
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Control{
		registry:                 cfg.Registry,
		ledger:                   cfg.Ledger,
		placement:                cfg.Placement,
		monitor:                  cfg.Monitor,
		nodes:                    cfg.Nodes,
		placementAttempts:        cfg.PlacementAttempts,
		admitPollInterval:        cfg.AdmitPollInterval,
		reservationRenewInterval: cfg.ReservationRenewInterval,
		sandbox:                  map[string]sandboxRecord{},
		provisions:               map[string]*provisionRecord{},
		pricing:                  map[string]int64{},
		lastSeq:                  map[string]uint64{},
		log:                      cfg.Logger,
	}, nil
}

// Create provisions one sandbox: quota, then routing, then placement, then the
// node's own admission.
//
// newGrantID returns the identity the quota ledger indexes one admission by.
//
// The ledger refuses a duplicate id, so the id has to be unique across creates. A
// timestamp alone is not: a grading wave or a fleet-wide rollout issues creates
// concurrently, and two can land in the same clock tick. The random suffix removes
// that collision without asking callers to retry.
func newGrantID() string {
	var suffix [6]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		// crypto/rand does not fail on Linux; keeping the timestamp alone is still
		// better than failing a create over its identifier.
		return fmt.Sprintf("grant-%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("grant-%d-%s", time.Now().UnixNano(), hex.EncodeToString(suffix[:]))
}

// A create is observable by its spec's idempotency key, when it has one: a
// caller that has no response yet can ask what state its create is in, which is
// what tells "still provisioning" apart from "the request never reached a
// working service".
func (c *Control) Create(ctx context.Context, req *v1.CreateRequest) (*v1.CreateResponse, error) {
	ticket := specFromProto(req.GetSpec()).IdempotencyKey
	if ticket != "" {
		c.beginProvision(ticket)
	}
	created, err := c.create(ctx, req)
	if ticket != "" {
		c.finishProvision(ticket, err)
	}
	return created, err
}

// create is the body of Create, split out so the provision record brackets every
// return path rather than only the success.
//
// Every failure path returns what it took. A create that gave up still holding
// a grant or a reservation would starve the fleet one slot at a time.
func (c *Control) create(ctx context.Context, req *v1.CreateRequest) (*v1.CreateResponse, error) {
	spec := specFromProto(req.GetSpec())
	selected, err := c.registry.Select(spec)
	if err != nil {
		// No backend can serve this spec. Waiting will not add a capability.
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}

	// The ledger and the node must key on the same owner, because that is the
	// identity the caller heartbeats. ownerOf reads it the way the node does.
	owner := ownerOf(spec)
	grantID := newGrantID()
	charged := c.charged(spec)
	amount := quotaAmount(charged)
	if err := c.admit(ctx, grantID, spec.ResourceClass, owner, amount); err != nil {
		return nil, err
	}

	created, err := c.place(ctx, selected, spec, charged, req.GetCallbackTarget(), owner)
	if err != nil {
		c.ledger.Release(grantID)
		return nil, err
	}
	c.remember(created.handle.SandboxID, sandboxRecord{
		handle: created.handle, backend: selected,
		grantID: grantID, reservationID: created.reservationID, class: spec.ResourceClass,
		owner: owner, createdAt: time.Now(),
		// What it costs now, and what its caller declared: the first is what the fleet
		// charges until the class's price moves, and the second is the ceiling a reprice
		// may never exceed.
		charged: amount, declared: quotaAmount(spec),
	})
	return createdToProto(created.created), nil
}

// charged is the spec admission should charge for.
//
// A node publishes what one sandbox of a class actually costs it, measured from its own
// sandboxes. Charging that instead of the request's declared maximum is what lets the
// fleet and placement admit a burst the node has room for: the maximum is a protection
// bound for one sandbox, and summing it across a batch reserves capacity nobody is using.
//
// Memory and CPU are repriced; disk and devices are not. Disk is stated rather than
// measured because it is consumed rather than held, and a device is a whole object: a
// sandbox either has a GPU or it does not. A node that has not enabled the pricing
// publishes nothing, which leaves the request as it was.
func (c *Control) charged(spec backend.Spec) backend.Spec {
	if c.monitor == nil {
		return spec
	}
	charged := spec
	if reserve, published := c.monitor.ClassReserveMB(spec.ResourceClass); published &&
		reserve > 0 && (spec.Resources.MemoryMB <= 0 || reserve < spec.Resources.MemoryMB) {
		charged.Resources.MemoryMB = reserve
		c.notePricing(spec.ResourceClass, "memory_mb", spec.Resources.MemoryMB, reserve)
	}
	requestedCPU := int64(spec.Resources.CPUCount * 1000)
	if reserve, published := c.monitor.ClassReserveCPUMillis(spec.ResourceClass); published &&
		reserve > 0 && (requestedCPU <= 0 || reserve < requestedCPU) {
		charged.Resources.CPUCount = float64(reserve) / 1000
		c.notePricing(spec.ResourceClass, "cpu_millis", requestedCPU, reserve)
	}
	return charged
}

// notePricing states what a class costs the fleet, once per value.
//
// Elastic pricing is the difference between admitting a batch and queueing it, and it
// happens between two processes that each look correct alone: the node publishes a
// reserve, the control charges it. When one of them stops, nothing fails -- the fleet
// simply goes back to charging every request's maximum -- so the charge is stated
// where an operator can see it rather than inferred from throughput later.
func (c *Control) notePricing(class, dimension string, requested, reserve int64) {
	key := class + "/" + dimension
	c.mu.Lock()
	if c.pricing[key] == reserve {
		c.mu.Unlock()
		return
	}
	c.pricing[key] = reserve
	c.mu.Unlock()
	c.log.Info("pricing a sandbox class from the nodes' published reserve",
		"class", class, "dimension", dimension, "requested", requested, "charged", reserve)
}

// admit charges one request against its class, waiting for room when the class
// is momentarily full.
//
// A class at its share is a queue, not a verdict: the share clears as sandboxes
// are released, and the request is for capacity that will exist rather than for a
// capability that never will. So the wait is unbounded -- there is no deadline
// here, because a deadline can only guess at when the fleet frees a slot. The one
// thing a wait cannot fix is refused definitively up front: a request larger than
// the class's whole share can never be admitted, and queueing on it is a livelock
// with a configuration cause.
//
// The demand is enqueued for the whole wait, which is what reserves the class's
// guarantee against borrowers while it waits. It is withdrawn on every exit path,
// so a caller that gave up does not leave the class reserving room for a request
// that no longer exists.
func (c *Control) admit(ctx context.Context, grantID, class, owner string, amount quota.Amount) error {
	if !c.ledger.CanEverAdmit(class, amount) {
		return status.Errorf(codes.OutOfRange,
			"the %q class can never hold a request this size, so no release will ever make room", class)
	}
	admitted, err := c.ledger.Acquire(grantID, class, owner, amount)
	if err != nil {
		return status.Error(codes.Internal, err.Error())
	}
	if admitted {
		return nil
	}

	c.ledger.Enqueue(class, amount)
	defer c.ledger.Cancel(class, amount)

	// Polled rather than signalled: a release happens on another caller's
	// goroutine and the ledger publishes no readiness channel, so the choice is
	// between a poll here and a condition variable threaded through every release
	// path. The interval is short against a sandbox lifetime and the waiters are
	// few, so the poll costs a lock acquisition per class per interval.
	ticker := time.NewTicker(c.admitPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			// The caller's own deadline fired. Its error is the honest one: nothing
			// about the fleet is known to be wrong.
			return status.FromContextError(ctx.Err()).Err()
		case <-ticker.C:
			admitted, err := c.ledger.Acquire(grantID, class, owner, amount)
			if err != nil {
				return status.Error(codes.Internal, err.Error())
			}
			if admitted {
				return nil
			}
		}
	}
}

// createOnRenewing provisions while keeping the placement reservation alive.
//
// A cold image pull can take longer than the reservation TTL, and a reservation
// that lapses mid-provision lets placement send another sandbox to a node this
// one is still about to occupy. Renewing closes that gap, and it does so without
// lengthening the TTL, so a genuinely dead create is still reclaimed promptly
// rather than held for the length of a pull it never started.
func (c *Control) createOnRenewing(
	ctx context.Context,
	reservationID string,
	provision func() (backend.Created, error),
) (backend.Created, error) {
	done := make(chan struct{})
	defer close(done)
	go func() {
		ticker := time.NewTicker(c.reservationRenewInterval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				c.placement.Renew(reservationID)
			}
		}
	}()
	return provision()
}

type placed struct {
	created       backend.Created
	handle        backend.Handle
	reservationID string
}

// place chooses a node and asks it to accept, trying another node when one
// refuses.
//
// A node refusal is information rather than an error: it knows its own pressure
// and placement only knows what the node last reported, so the right response is
// to ask placement again with that node now charged.
func (c *Control) place(
	ctx context.Context, selected backend.Backend, spec, charged backend.Spec, callback, owner string,
) (placed, error) {
	if selected.Mode() == backend.SchedulingProvider {
		// The backend's own control plane chooses the node, so this service does
		// quota only and passes an empty node id.
		created, err := selected.Create(ctx, "", spec, callback)
		if err != nil {
			return placed{}, status.Error(codes.Unavailable, err.Error())
		}
		return placed{created: created, handle: created.Handle}, nil
	}

	var lastRefusal string
	var lastTransportErr error
	transportFailures := 0
	// refused carries the nodes that already said no into the next attempt. Placement
	// ranks on a reported headroom that can be seconds old, while a refusal is that
	// node's own ledger speaking about now; re-asking the same node the same question is
	// what turns a single full machine into a fleet-wide exhaustion error.
	refused := make([]string, 0, c.placementAttempts)
	// Do not try more times than there are nodes to try. The attempt after the last node
	// has refused has an empty candidate set, and the error it produces no longer says
	// whether the fleet was full or unable to serve the request -- which is the one thing
	// the caller needs in order to know whether to wait or give up.
	attempts := c.placementAttempts
	if live := c.placement.LiveCount(); live > 0 && live < attempts {
		attempts = live
	}
	for attempt := 0; attempt < attempts; attempt++ {
		// Charge what the node says a sandbox costs, not what the caller asked for as a
		// maximum: placement's promises are cleared by the node's own report, so a promise
		// of the maximum would refuse a batch the node is willing to hold.
		request := placementRequest(charged, selected.Name(), owner)
		request.ExcludeNodes = refused
		decision, err := c.placement.Choose(request)
		if err != nil {
			if errors.Is(err, placement.ErrExhausted) {
				return placed{}, status.Error(codes.ResourceExhausted, err.Error())
			}
			return placed{}, status.Error(codes.FailedPrecondition, err.Error())
		}
		leaseID, gpus, refusal, err := c.nodes.Admit(ctx, decision.NodeID, spec)
		if err != nil {
			// A node that did not answer is not a full fleet. Excluding it and
			// trying the next candidate keeps one unreachable machine from
			// failing every create; only when every attempt was a transport
			// failure is the fleet itself unreachable.
			c.placement.Cancel(decision.ReservationID)
			lastTransportErr = err
			transportFailures++
			refused = append(refused, decision.NodeID)
			if c.log != nil {
				c.log.Warn("a node did not answer admission, trying the next candidate",
					"node", decision.NodeID, "error", err)
			}
			continue
		}
		if refusal != "" {
			// Try elsewhere. The reservation is withdrawn so the node is not held for
			// a sandbox it just declined.
			c.placement.Cancel(decision.ReservationID)
			lastRefusal = refusal
			refused = append(refused, decision.NodeID)
			if c.log != nil {
				c.log.Warn("a node refused this sandbox",
					"node", decision.NodeID, "class", charged.ResourceClass, "reason", refusal,
					"asked_memory_mb", charged.Resources.MemoryMB, "asked_cpu_millis", chargedResourcesMillis(charged))
			}
			continue
		}
		withDevices := spec
		withDevices.AssignedGPUs = gpus
		// The runtime placement matched on, not the one routing picked: on a node
		// hosting several they agree, and where they could not, placement's is the
		// one whose capabilities were checked against this node.
		created, err := c.createOnRenewing(ctx, decision.ReservationID, func() (backend.Created, error) {
			return c.nodes.CreateOn(ctx, decision.NodeID, leaseID, decision.Backend, withDevices, callback)
		})
		if err != nil {
			c.placement.Cancel(decision.ReservationID)
			return placed{}, status.Errorf(codes.Unavailable, "node %s could not create: %v", decision.NodeID, err)
		}
		return placed{created: created, handle: created.Handle, reservationID: decision.ReservationID}, nil
	}
	if transportFailures > 0 && transportFailures == attempts && lastTransportErr != nil {
		return placed{}, status.Errorf(codes.Unavailable,
			"no node answered admission after %d attempt(s): %v", attempts, lastTransportErr)
	}
	if c.log != nil {
		// A refusal is the one thing the caller cannot see through: it arrives as a
		// class of error that means "wait", and whether waiting helps depends on why
		// the nodes said no. Logged here because the fleet is the only place that
		// knows every node answered.
		c.log.Warn("no node would take this sandbox",
			"class", charged.ResourceClass, "nodes_asked", len(refused), "last_reason", lastRefusal,
			"asked_memory_mb", charged.Resources.MemoryMB, "asked_cpu_millis", chargedResourcesMillis(charged))
	}
	return placed{}, status.Errorf(codes.ResourceExhausted,
		"every node placement tried refused this request (last reason: %q); this clears as sandboxes are released",
		lastRefusal)
}

// chargedResourcesMillis is the CPU the fleet is asking a node for, in the units the
// node's ledger reports.
func chargedResourcesMillis(spec backend.Spec) int64 {
	return int64(spec.Resources.CPUCount * 1000)
}

// groupCleanupTimeout bounds releasing the members of a group that failed part
// way, so a cleanup cannot outlive the request that triggered it.
const groupCleanupTimeout = 30 * time.Second

// CreateGroup provisions one group.
//
// Members are admitted independently, because a group is a completion unit
// rather than a scheduling unit: the algorithm needs its trajectories to finish
// together, not to start together. No member holds capacity while a sibling
// queues, which is what keeps two concurrent groups from starving each other.
func (c *Control) CreateGroup(ctx context.Context, req *v1.CreateGroupRequest) (*v1.CreateGroupResponse, error) {
	specs := req.GetSpecs()
	if len(specs) == 0 {
		return nil, status.Error(codes.InvalidArgument, "a sandbox group needs at least one member spec")
	}
	out := &v1.CreateGroupResponse{}
	created := make([]*v1.SandboxHandle, 0, len(specs))
	for _, spec := range specs {
		member, err := c.Create(ctx, &v1.CreateRequest{Spec: spec, CallbackTarget: req.GetCallbackTarget()})
		if err != nil {
			// A failed group leaves nothing charged. The cleanup cannot run on the
			// caller's context: the usual reason a member fails is that context's
			// own deadline, and releasing under a cancelled context would leave
			// every member already made charged, with no handle a caller could
			// retry from.
			cleanupCtx, cancel := context.WithTimeout(context.Background(), groupCleanupTimeout)
			for _, handle := range created {
				_, _ = c.Release(cleanupCtx, handle)
			}
			cancel()
			return nil, err
		}
		created = append(created, member.GetHandle())
		out.Members = append(out.Members, member)
	}
	return out, nil
}

// Release destroys one sandbox and returns everything it held.
func (c *Control) Release(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	record, known := c.forget(handle.GetSandboxId())
	if !known {
		return c.releaseUnrecorded(ctx, handle)
	}
	var err error
	if record.backend.Mode() == backend.SchedulingProvider {
		err = record.backend.Release(ctx, record.handle)
	} else {
		err = c.nodes.ReleaseOn(ctx, record.handle)
	}
	if err != nil {
		// The sandbox may still exist, so put it back: a caller must be able to
		// retry, and the quota must stay charged until destruction is confirmed.
		c.remember(handle.GetSandboxId(), record)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	if record.reservationID != "" {
		c.placement.Release(record.reservationID)
	}
	c.ledger.Release(record.grantID)
	return &v1.Empty{}, nil
}

// releaseUnrecorded destroys a sandbox this control plane holds no record of,
// which is what a restarted control plane has for every sandbox the fleet is
// still running. Returning success without asking would leak the container, so
// the handle's node and backend are used to reach the runtime that owns it.
//
// There is no grant to return here. The record and the ledger were both in
// memory and came back empty, so nothing is charged for this sandbox.
func (c *Control) releaseUnrecorded(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	if nodeID := handle.GetNodeId(); nodeID != "" {
		err := c.nodes.ReleaseOn(ctx, backend.Handle{
			Backend: handle.GetBackend(), SandboxID: handle.GetSandboxId(), NodeID: nodeID,
		})
		if err != nil {
			// The sandbox may still exist, so the caller has to be able to retry.
			return nil, status.Error(codes.Unavailable, err.Error())
		}
		return &v1.Empty{}, nil
	}
	selected, err := c.registry.Get(handle.GetBackend())
	if err != nil || selected.Mode() != backend.SchedulingProvider {
		// No runtime to ask, so there is nothing left to do about it.
		return &v1.Empty{}, nil
	}
	if err := selected.Release(ctx, backend.Handle{
		Backend: handle.GetBackend(), SandboxID: handle.GetSandboxId(),
	}); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

// Status reports one sandbox's portable state.
func (c *Control) Status(ctx context.Context, handle *v1.SandboxHandle) (*v1.StatusResponse, error) {
	record, known := c.lookup(handle.GetSandboxId())
	if !known {
		return &v1.StatusResponse{Status: v1.SandboxStatus_TERMINATED}, nil
	}
	var (
		state string
		err   error
	)
	if record.backend.Mode() == backend.SchedulingProvider {
		state, err = record.backend.Status(ctx, record.handle)
	} else {
		state, err = c.nodes.StatusOn(ctx, record.handle)
	}
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.StatusResponse{Status: statusValue(state)}, nil
}

// Pause releases a sandbox's compute while keeping its state.
func (c *Control) Pause(ctx context.Context, req *v1.PauseRequest) (*v1.Empty, error) {
	record, stateful, err := c.stateful(req.GetHandle())
	if err != nil {
		return nil, err
	}
	if err := stateful.Pause(ctx, record.handle, pauseModeName(req.GetMode())); err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	return &v1.Empty{}, nil
}

// Resume restarts a paused sandbox.
func (c *Control) Resume(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	record, stateful, err := c.stateful(handle)
	if err != nil {
		return nil, err
	}
	if err := stateful.Resume(ctx, record.handle); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

// Snapshot captures a sandbox's state.
func (c *Control) Snapshot(ctx context.Context, req *v1.SnapshotRequest) (*v1.SnapshotRef, error) {
	record, stateful, err := c.stateful(req.GetHandle())
	if err != nil {
		return nil, err
	}
	kind := snapshotKindName(req.GetKind())
	id, err := stateful.Snapshot(ctx, record.handle, kind)
	if err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	return &v1.SnapshotRef{Backend: record.handle.Backend, SnapshotId: id, Kind: req.GetKind()}, nil
}

// DeleteSnapshot removes a captured state.
func (c *Control) DeleteSnapshot(ctx context.Context, ref *v1.SnapshotRef) (*v1.Empty, error) {
	selected, err := c.registry.Get(ref.GetBackend())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	stateful, ok := selected.(backend.Stateful)
	if !ok {
		return nil, status.Errorf(codes.FailedPrecondition, "backend %q keeps no snapshots", selected.Name())
	}
	if err := stateful.DeleteSnapshot(ctx, ref.GetSnapshotId()); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

func (c *Control) stateful(handle *v1.SandboxHandle) (sandboxRecord, backend.Stateful, error) {
	record, known := c.lookup(handle.GetSandboxId())
	if !known {
		return sandboxRecord{}, nil, status.Errorf(codes.NotFound, "sandbox %q is not owned by this service", handle.GetSandboxId())
	}
	stateful, ok := record.backend.(backend.Stateful)
	if !ok {
		return sandboxRecord{}, nil, status.Errorf(codes.FailedPrecondition,
			"backend %q has no state operations, so this cannot be served rather than being served weakly",
			record.backend.Name())
	}
	return record, stateful, nil
}

// Fleet reports what the monitor sees, for the caller's metric hook.
func (c *Control) Fleet(context.Context, *v1.Empty) (*v1.FleetReport, error) {
	report := c.placement.Snapshot()
	out := &v1.FleetReport{
		DrainedNodes:      int32(report.DrainedNodes),
		ReservationsOpen:  int32(report.ReservationsOpen),
		CapacityExhausted: report.CapacityExhausted,
		NoCandidate:       report.NoCandidate,
		LocalityHitRatio:  report.LocalityHitRatio,
	}
	for _, view := range c.monitor.Fleet() {
		reported := &v1.NodeView{
			NodeId:        view.NodeID,
			LiveSandboxes: int32(view.LiveSandboxes),
			CpuUsedPct:    view.CPUUsedPct,
			MemUsedPct:    view.MemUsedPct,
			GpuFree:       view.GPUFree,
			Draining:      view.Draining,
			Granted:       placementHeadroomToProto(view.Granted),
			Observed:      placementHeadroomToProto(view.Observed),
			ThrottledPct:  view.ThrottledPct,
			Envelope:      placementHeadroomToProto(view.Envelope),
			ClassHeadroom: map[string]*v1.Headroom{},
		}
		for class, headroom := range view.ClassHeadroom {
			reported.ClassHeadroom[class] = placementHeadroomToProto(headroom)
		}
		for _, hosted := range view.Backends {
			reported.Backends = append(reported.Backends, &v1.BackendView{Name: hosted.Name})
		}
		out.Nodes = append(out.Nodes, reported)
	}
	return out, nil
}

// Quota reports the cross-backend ledger, for the caller's metric hook.
func (c *Control) Quota(context.Context, *v1.Empty) (*v1.QuotaReport, error) {
	out := &v1.QuotaReport{Classes: map[string]*v1.ClassQuota{}}
	for name, report := range c.ledger.Report() {
		out.Classes[name] = &v1.ClassQuota{
			GuaranteedShare: report.Guaranteed,
			MaxShare:        report.Max,
			Granted:         quotaHeadroomToProto(report.Granted),
			Headroom:        quotaHeadroomToProto(report.Headroom),
			Queued:          int32(report.Queued),
		}
	}
	return out, nil
}

func (c *Control) remember(id string, record sandboxRecord) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// A report generated before this grant was recorded cannot be evidence that
	// its sandbox is absent. Two sequences of grace make sure the first eligible
	// report was generated after the create returned.
	record.armedAfterSeq = c.lastSeq[record.handle.NodeID] + 2
	c.sandbox[id] = record
}

// ReconcileGrants releases grants whose sandbox has vanished from the node that
// was holding it.
//
// The quest the ledger answered at create time is "what did the fleet promise",
// and only the node can answer whether that promise still has a runtime behind
// it. A node that crashed, a container that was removed out of band, or a
// release whose response was lost all leave a grant charged for nothing; this
// is where the ledger converges back to the node's truth.
//
// The guards are the whole safety argument:
//   - an older node that sends no inventory is skipped, so a rolling upgrade
//     cannot read "no inventory" as "holds nothing";
//   - a truncated inventory is skipped, because a partial list is not evidence
//     of absence;
//   - a report generated before the grant was recorded is skipped, so a create
//     in flight is never mistaken for a vanished sandbox;
//   - two consecutive eligible misses are required, so one lost report is not a
//     release;
//   - a sequence regression means the node restarted and its report belongs to
//     a new epoch, so everything is re-armed rather than compared.
func (c *Control) ReconcileGrants(view placement.NodeView) {
	if view.NodeID == "" || !view.InventoryReported || view.InventoryTruncated {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	last := c.lastSeq[view.NodeID]
	if view.ReportSeq < last {
		c.lastSeq[view.NodeID] = view.ReportSeq
		for id, record := range c.sandbox {
			if record.handle.NodeID != view.NodeID {
				continue
			}
			record.misses = 0
			record.armedAfterSeq = view.ReportSeq + 2
			c.sandbox[id] = record
		}
		return
	}
	if view.ReportSeq == last {
		return
	}
	c.lastSeq[view.NodeID] = view.ReportSeq

	live := make(map[string]struct{}, len(view.LiveSandboxIDs))
	for _, sandboxID := range view.LiveSandboxIDs {
		live[sandboxID] = struct{}{}
	}
	for id, record := range c.sandbox {
		if record.handle.NodeID != view.NodeID {
			continue
		}
		if view.ReportSeq < record.armedAfterSeq {
			continue
		}
		if _, present := live[record.handle.SandboxID]; present {
			record.misses = 0
			c.sandbox[id] = record
			continue
		}
		record.misses++
		if record.misses < 2 {
			c.sandbox[id] = record
			continue
		}
		c.ledger.Release(record.grantID)
		delete(c.sandbox, id)
		if c.log != nil {
			c.log.Warn("released a grant whose sandbox the node no longer holds",
				"node", view.NodeID, "sandbox_id", record.handle.SandboxID,
				"grant", record.grantID, "reports_missed", record.misses)
		}
	}
}

// RepriceGrants brings the fleet's ledger back in line with what the nodes measure.
// It is the exported face of repriceGrants, so the loop that drives it lives with the
// rest of the process's background work.
//
// The nodes reprice their own reservations as they sample; the fleet charges each grant
// what its class cost at admission and otherwise forgets it. Left alone that gap is the
// whole failure: the fleet keeps admitting against a price that has moved on, every one
// of those admissions is refused by the node that owns the pages, and the caller sees
// capacity exhaustion until the under-charged sandboxes happen to finish.
//
// It is cheap on purpose -- a read of the published reserve per class and a walk of the
// live grants -- so it can run between reports rather than on the admission path.
func (c *Control) RepriceGrants() {
	if c.monitor == nil {
		return
	}
	c.mu.Lock()
	type pending struct {
		grantID string
		amount  quota.Amount
		class   string
	}
	updates := make([]pending, 0, len(c.sandbox))
	for _, record := range c.sandbox {
		if record.grantID == "" {
			continue
		}
		want := c.chargedAmount(record.declared, record.class)
		if want == record.charged {
			continue
		}
		updates = append(updates, pending{grantID: record.grantID, amount: want, class: record.class})
	}
	c.mu.Unlock()

	for _, update := range updates {
		c.ledger.Reprice(update.grantID, update.amount)

		c.mu.Lock()
		for id, record := range c.sandbox {
			if record.grantID != update.grantID {
				continue
			}
			record.charged = update.amount
			c.sandbox[id] = record
			break
		}
		c.mu.Unlock()
	}
}

// chargedAmount is what a class's published price makes this request cost the fleet,
// never more than the caller declared.
func (c *Control) chargedAmount(declared quota.Amount, class string) quota.Amount {
	want := declared
	if reserve, published := c.monitor.ClassReserveMB(class); published && reserve > 0 && reserve < want.MemoryMB {
		want.MemoryMB = reserve
	}
	if reserve, published := c.monitor.ClassReserveCPUMillis(class); published && reserve > 0 && reserve < want.CPUMillis {
		want.CPUMillis = reserve
	}
	return want
}

func (c *Control) lookup(id string) (sandboxRecord, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.sandbox[id]
	return record, known
}

func (c *Control) forget(id string) (sandboxRecord, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.sandbox[id]
	delete(c.sandbox, id)
	return record, known
}

// beginProvision records a create as in flight under its ticket, and ages out the
// finished records a caller is no longer asking about.
func (c *Control) beginProvision(ticket string) {
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	for id, rec := range c.provisions {
		if now.Sub(rec.updatedAt) >= provisionRetention {
			delete(c.provisions, id)
		}
	}
	c.provisions[ticket] = &provisionRecord{state: "provisioning", startedAt: now, updatedAt: now}
}

// finishProvision records a create's outcome under its ticket.
func (c *Control) finishProvision(ticket string, err error) {
	state := "created"
	if err != nil {
		state = "failed"
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if rec, known := c.provisions[ticket]; known {
		rec.state = state
		rec.updatedAt = time.Now()
	}
}

// ProvisionState is what one create's ticket reports.
type ProvisionState struct {
	State         string
	Elapsed       time.Duration
	SinceProgress time.Duration
}

// ProvisionState reports how a create with this ticket is doing, and whether it
// is known. An unknown ticket means no create with that key has been seen here,
// which is not the same as a failure.
func (c *Control) ProvisionState(ticket string) (ProvisionState, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	rec, known := c.provisions[ticket]
	if !known {
		return ProvisionState{}, false
	}
	now := time.Now()
	return ProvisionState{
		State:         rec.state,
		Elapsed:       now.Sub(rec.startedAt),
		SinceProgress: now.Sub(rec.updatedAt),
	}, true
}

// The two budgets a renewal has to fit inside the caller's own deadline. The
// client waits ten seconds; a batch that used a node's full RPC budget would
// never answer inside that, so the whole fan-out is bounded well below it.
const (
	renewNodeTimeout = 3 * time.Second
	renewBatchBudget = 5 * time.Second
)

// RenewOwner keeps one caller's grants alive and tells every node that actually
// holds its sandboxes that the caller is still there.
//
// The fan-out targets the nodes named by this owner's records, not every node in
// the fleet. A node that once reported and then went dark is still in the
// monitor's historical view, and renewing it would make every heartbeat pay a
// dead node's timeout -- the failure this path used to have. A control plane
// that has just restarted has no records, so it falls back to the live fleet.
//
// The fan-out is concurrent and bounded: one stalled node delays its own
// goroutine, never the nodes that would answer.
func (c *Control) RenewOwner(ctx context.Context, owner string) {
	c.ledger.Renew(owner)
	if c.nodes == nil {
		return
	}
	nodes := c.renewalNodes(owner)
	c.renewAttempts.Add(1)
	c.renewNodes.Add(int64(len(nodes)))
	if len(nodes) == 0 {
		return
	}

	started := time.Now()
	batchCtx, cancel := context.WithTimeout(ctx, renewBatchBudget)
	defer cancel()

	var (
		wg     sync.WaitGroup
		mu     sync.Mutex
		failed []string
	)
	for _, nodeID := range nodes {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			nodeCtx, nodeCancel := context.WithTimeout(batchCtx, renewNodeTimeout)
			defer nodeCancel()
			if err := c.nodes.RenewOwnerOn(nodeCtx, id, owner); err != nil {
				mu.Lock()
				failed = append(failed, id)
				mu.Unlock()
			}
		}(nodeID)
	}
	wg.Wait()

	c.renewLastNanos.Store(int64(time.Since(started)))
	if len(failed) > 0 {
		c.renewFailures.Add(int64(len(failed)))
		if c.log != nil {
			// One line per batch, not one per node per beat: the failure is a
			// property of the fleet, and a log that repeats it every thirty
			// seconds is a log an operator stops reading.
			c.log.Warn("owner renewal did not reach every node holding this owner's sandboxes",
				"owner", owner, "failed_nodes", failed, "targeted", len(nodes))
		}
	}
}

// renewalNodes returns the nodes that hold this owner's sandboxes, or the live
// fleet when neither the control's records nor the nodes themselves name one.
//
// The node's own LiveOwners is the authority: a control restart, a lost release
// response, or a reconciliation can leave a sandbox out of `c.sandbox` while
// the node still holds it, and targeting only the recorded nodes would let that
// node's owner lease expire and reclaim every sandbox it holds for this owner.
func (c *Control) renewalNodes(owner string) []string {
	seen := map[string]struct{}{}
	c.mu.Lock()
	for _, record := range c.sandbox {
		if record.owner == owner && record.handle.NodeID != "" {
			seen[record.handle.NodeID] = struct{}{}
		}
	}
	c.mu.Unlock()
	for _, view := range c.monitor.LiveFleet() {
		for _, heldOwner := range view.LiveOwners {
			if heldOwner == owner {
				seen[view.NodeID] = struct{}{}
				break
			}
		}
	}
	if len(seen) == 0 {
		for _, view := range c.monitor.LiveFleet() {
			seen[view.NodeID] = struct{}{}
		}
	}
	nodes := make([]string, 0, len(seen))
	for id := range seen {
		nodes = append(nodes, id)
	}
	sort.Strings(nodes)
	return nodes
}

// RenewStats reports the renewal path's own health, for the diagnostics surface.
func (c *Control) RenewStats() (attempts, failures, targeted int64, last time.Duration) {
	return c.renewAttempts.Load(), c.renewFailures.Load(), c.renewNodes.Load(),
		time.Duration(c.renewLastNanos.Load())
}

// Diagnostics reports the control plane's own counters and every node's, in the
// shape the SDK's diagnostics call returns.
//
// These are the facts that used to exist only as log lines: how many renewals
// failed, how long the last fan-out took, and how many nodes are stale. A
// service that reports only what it schedules is a service an operator has to
// guess about exactly when it is misbehaving.
func (c *Control) Diagnostics() map[string]any {
	attempts, failures, targeted, last := c.RenewStats()
	fleet := c.monitor.Fleet()
	live := c.monitor.LiveFleet()
	c.mu.Lock()
	records := len(c.sandbox)
	c.mu.Unlock()
	nodes := make(map[string]any, len(fleet))
	for _, view := range fleet {
		nodes[view.NodeID] = map[string]any{
			"live_sandboxes":    view.LiveSandboxes,
			"last_report_age_s": time.Since(view.SeenAt).Seconds(),
			"draining":          view.Draining,
			"diagnostics":       view.Diagnostics,
		}
	}
	return map[string]any{
		"renew": map[string]any{
			"attempts":         attempts,
			"failures":         failures,
			"nodes_targeted":   targeted,
			"last_duration_ms": float64(last.Microseconds()) / 1000.0,
		},
		"fleet": map[string]any{
			"live_nodes":  len(live),
			"stale_nodes": len(fleet) - len(live),
			"records":     records,
		},
		"nodes": nodes,
	}
}
