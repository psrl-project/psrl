//go:build integration

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/dockerbackend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/timing"
)

// The whole service, wired as it is deployed: a control plane over a monitor and
// a placement engine, one node agent with its own admission, and a real Docker
// daemon underneath. A test against fakes at every layer would only prove the
// layers agree with each other.

type stack struct {
	control *Control
	node    *Node
	monitor *monitor.Monitor
	ledger  *quota.Ledger
	place   *placement.Service
	backend *dockerbackend.Backend
}

// dockerSocket returns the daemon to test against, skipping where none is.
func dockerSocket(t *testing.T) string {
	t.Helper()
	socket := os.Getenv("DOCKER_SOCKET")
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	if _, err := os.Stat(socket); err != nil {
		t.Skipf("no docker socket at %s", socket)
	}
	return socket
}

func newStack(t *testing.T, fleetMemoryMB int64) *stack {
	t.Helper()
	socket := dockerSocket(t)

	// The node TTL has to outlast the reservation TTL or placement refuses the
	// configuration. It is not what makes these tests fast: reclamation here is
	// driven per sandbox by the lifetime its create states, and by the sweep.
	spans, err := timing.New(timing.DefaultNodeTTL, 5*time.Second)
	if err != nil {
		t.Fatalf("timing: %v", err)
	}

	docker, err := dockerbackend.New(dockerbackend.Config{
		Socket:         socket,
		APIVersion:     "v1.40",
		NodeID:         "node-a",
		OwnerID:        fmt.Sprintf("sandboxd-int-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second,
		PullTimeout:    3 * time.Minute,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("docker backend: %v", err)
	}
	if err := docker.Preflight(context.Background()); err != nil {
		t.Skipf("daemon not usable: %v", err)
	}

	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: fleetMemoryMB, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: timing.OwnerLease,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle("node-a", gate, timing.SweepInterval)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	agent, err := NewNode(NodeConfig{
		NodeID: "node-a", Admission: gate, Lifecycle: life, Backends: []backend.Backend{docker},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}

	mon := monitor.New(spans.NodeTTL)
	mon.Report(agent.View())

	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: fleetMemoryMB, CPUMillis: 100000, Sandboxes: 1000},
		Classes:  map[string]quota.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: timing.OwnerLease,
	})
	if err != nil {
		t.Fatalf("ledger: %v", err)
	}
	place, err := placement.New(placement.Config{
		NodeTTL:        spans.NodeTTL,
		ReservationTTL: timing.ReservationTTL,
		SweepInterval:  timing.SweepInterval,
	}, mon)
	if err != nil {
		t.Fatalf("placement: %v", err)
	}
	registry, err := backend.NewRegistry([]backend.Backend{docker}, "docker")
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	control, err := NewControl(ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: mon,
		Nodes: NewLocalNodeClient(agent),
	})
	if err != nil {
		t.Fatalf("control: %v", err)
	}
	return &stack{control: control, node: agent, monitor: mon, ledger: ledger, place: place, backend: docker}
}

func (s *stack) refresh() { s.monitor.Report(s.node.View()) }

func createSpec(memoryMB int64) *v1.SandboxSpec {
	memory := memoryMB
	cpu := 0.25
	return &v1.SandboxSpec{
		Source:        &v1.Source{Kind: v1.Source_IMAGE, Reference: "alpine:latest"},
		Resources:     &v1.Resources{MemoryMb: &memory, CpuCount: &cpu},
		ResourceClass: "rollout",
	}
}

func TestOneSandboxGoesEndToEndThroughEveryLayer(t *testing.T) {
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(ctx, created.GetHandle())

	if created.GetHandle().GetNodeId() != "node-a" {
		t.Fatalf("sandbox landed on %q, want node-a", created.GetHandle().GetNodeId())
	}
	status, err := s.control.Status(ctx, created.GetHandle())
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status.GetStatus() != v1.SandboxStatus_RUNNING {
		t.Fatalf("status is %v, want RUNNING", status.GetStatus())
	}

	// The command path goes to the node, never through the control plane.
	result, err := s.node.Exec(ctx,
		backend.Handle{SandboxID: created.GetHandle().GetSandboxId()}, "echo wired", "", nil, backend.ExecOptions{})
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("exec: code=%d err=%v", result.ExitCode, err)
	}
	if !strings.Contains(result.Stdout, "wired") {
		t.Fatalf("output %q", result.Stdout)
	}
}

func TestReleasingReturnsBothTheFleetQuotaAndTheNodeEnvelope(t *testing.T) {
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	before := s.ledger.Headroom("rollout").MemoryMB

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	during := s.ledger.Headroom("rollout").MemoryMB
	if during != before-64 {
		t.Fatalf("quota went %d -> %d, want -64MB charged", before, during)
	}

	if _, err := s.control.Release(ctx, created.GetHandle()); err != nil {
		t.Fatalf("release: %v", err)
	}

	if after := s.ledger.Headroom("rollout").MemoryMB; after != before {
		t.Fatalf("quota went %d -> %d, want the charge returned", before, after)
	}
	if left := s.node.lifecycle.Snapshot().Resident; left != 0 {
		t.Fatalf("%d sandboxes still held by the node", left)
	}
}

func TestAFleetWithoutRoomRefusesRatherThanOversubscribing(t *testing.T) {
	// A 128MB fleet takes two 64MB sandboxes and must refuse the third, at the
	// quota rather than at the node: the fleet ledger is the outer bound.
	s := newStack(t, 128)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	var handles []*v1.SandboxHandle
	defer func() {
		for _, handle := range handles {
			s.control.Release(context.Background(), handle)
		}
	}()
	for i := 0; i < 2; i++ {
		created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
		if err != nil {
			t.Fatalf("create %d: %v", i, err)
		}
		handles = append(handles, created.GetHandle())
	}

	// A full class queues rather than refusing, by design: the share clears as
	// sandboxes are released, and the server does not guess when. So an immediate
	// refusal is not what the caller gets. What must hold is that no release ever
	// makes room for a third 64MB sandbox on a fleet of two, so a caller that
	// gives up fails and the ledger is unchanged.
	giveUp, cancelGiveUp := context.WithTimeout(context.Background(), heldOnAFullFleet)
	defer cancelGiveUp()
	if _, err := s.control.Create(giveUp, &v1.CreateRequest{Spec: createSpec(64)}); err == nil {
		t.Fatal("a third sandbox must not be admitted to a full fleet")
	}
	if granted := s.ledger.Report()["rollout"].Granted.MemoryMB; granted != 128 {
		t.Fatalf("the ledger accounts %dMB, want the two admitted 64MB sandboxes", granted)
	}
}

// heldOnAFullFleet bounds a create this test expects to fail. It is a bound on
// the test rather than a behaviour under test: the fleet must never oversubscribe
// whether the caller waits a second or a minute.
const heldOnAFullFleet = 3 * time.Second

func TestASpecNoBackendCanServeIsRefusedAtAdmission(t *testing.T) {
	// Refused rather than degraded: a workspace restore must never be handed to a
	// caller that asked for a live process to survive.
	s := newStack(t, 4096)
	spec := createSpec(64)
	spec.RequiredFeatures = []v1.Feature{v1.Feature_NATIVE_FORK}

	_, err := s.control.Create(context.Background(), &v1.CreateRequest{Spec: spec})

	if err == nil {
		t.Fatal("docker has no native fork, so this spec must be refused")
	}
	if !strings.Contains(err.Error(), "native_fork") {
		t.Fatalf("the refusal must name what was missing, got %v", err)
	}
}

func TestANodeUnderPressureSendsTheRequestElsewhere(t *testing.T) {
	// The node keeps the last word: placement compares a headroom reported
	// seconds ago, and the machine knows its pressure now.
	s := newStack(t, 4096)
	s.node.pressure = func() node.Pressure {
		return node.Pressure{MemUsedPct: 0.99, SampledAt: time.Now(), Valid: true}
	}
	s.node.admission = mustPressureGate(t)

	_, err := s.control.Create(context.Background(), &v1.CreateRequest{Spec: createSpec(64)})

	if err == nil {
		t.Fatal("a node under pressure must refuse, and no other node exists")
	}
}

func mustPressureGate(t *testing.T) *node.Admission {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope:        node.Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:         map[string]node.ClassShare{"rollout": {Guaranteed: 1.0}},
		LocalMemCeiling: 0.9,
		LeaseTTL:        time.Minute,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	return gate
}

func TestASandboxPastItsLifetimeIsReclaimedByTheNode(t *testing.T) {
	// Reclamation is the node's, so this runs without the control plane doing
	// anything: a sandbox that outlives the lifetime its create stated must not
	// stay resident, because its caller may be gone.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	request := createSpec(64)
	request.LifetimeS = 1
	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: request})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(context.Background(), created.GetHandle())

	// Past the lifetime the create stated.
	time.Sleep(2 * time.Second)
	reaped, err := s.node.Sweep(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(reaped.GetReapedLifetime()) != 1 {
		t.Fatalf("the sandbox past its lifetime was not reclaimed: %+v", reaped)
	}
}

func TestANodeAdoptsTheSandboxesItFindsAfterARestart(t *testing.T) {
	// The daemon outlives the process, so a restarted node does not start on an
	// empty machine. Adoption is what stops it handing out memory a live sandbox
	// is already holding, and what lets it reclaim work whose caller is gone.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(context.Background(), created.GetHandle())
	sandboxID := created.GetHandle().GetSandboxId()

	// A fresh node over the same daemon is what a restart leaves: an empty
	// ledger and the containers the previous process made.
	restarted, restartedGate, restartedLife := nodeOverTheSameDaemon(t, s)
	report, err := restartedLife.Reconcile(ctx, []backend.Backend{s.backend})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	if report.Adopted != 1 || report.Reclaimed != 0 {
		t.Fatalf("reconcile %+v, want the one sandbox the daemon still holds", report)
	}
	if granted := restartedGate.Snapshot().Granted.MemoryMB; granted != 64 {
		t.Fatalf("the adopted sandbox was charged %dMB, want its own 64MB", granted)
	}
	if live := restarted.View().LiveSandboxes; live != 1 {
		t.Fatalf("the restarted node reports %d live sandboxes, want 1", live)
	}
	// Owning it is what lets the restarted node destroy it, which is the path a
	// caller's release takes after a restart.
	if err := restartedLife.Release(ctx, sandboxID, node.ExitReleased); err != nil {
		t.Fatalf("release: %v", err)
	}
	status, err := s.backend.Status(ctx, backend.Handle{Backend: "docker", SandboxID: sandboxID})
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status != "terminated" {
		t.Fatalf("the container survived its release: status %q", status)
	}
}

// nodeOverTheSameDaemon returns a node, gate and lifecycle that share the
// daemon with s but know nothing of what s created, which is the state a
// process restart leaves behind.
func nodeOverTheSameDaemon(t *testing.T, s *stack) (*Node, *node.Admission, *node.Lifecycle) {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: timing.OwnerLease,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle("node-a", gate, timing.SweepInterval)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	life.SetOwnerLiveness(gate.OwnerAlive)
	agent, err := NewNode(NodeConfig{
		NodeID: "node-a", Admission: gate, Lifecycle: life, Backends: []backend.Backend{s.backend},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent, gate, life
}

// burstGiveUp bounds the concurrent burst below. Eight of the twenty-four fit
// at once and the rest queue, so the bound is what makes the test finish rather
// than a statement about admission.
const burstGiveUp = 20 * time.Second

func TestABurstOfCreatesStaysInsideTheFleet(t *testing.T) {
	// Creates arrive in bursts, so the invariant that matters is that concurrency
	// never admits past the fleet, whatever the interleaving.
	s := newStack(t, 512)
	// A full class queues, so the creates that cannot fit wait for the caller's
	// deadline rather than failing at once. The bound is on the test: the
	// invariant below has to hold whatever the wait.
	ctx, cancel := context.WithTimeout(context.Background(), burstGiveUp)
	defer cancel()

	var mu sync.Mutex
	var handles []*v1.SandboxHandle
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
			if err != nil {
				return
			}
			mu.Lock()
			handles = append(handles, created.GetHandle())
			mu.Unlock()
		}()
	}
	wg.Wait()
	defer func() {
		for _, handle := range handles {
			s.control.Release(context.Background(), handle)
		}
	}()

	// 512MB fleet, 64MB each: exactly eight fit.
	if len(handles) != 8 {
		t.Fatalf("admitted %d sandboxes onto a 512MB fleet of 64MB each, want 8", len(handles))
	}
	if granted := s.ledger.Report()["rollout"].Granted.MemoryMB; granted != 512 {
		t.Fatalf("the ledger accounts %dMB, want 512MB", granted)
	}
}

func TestTheFleetAndQuotaReportsAreServable(t *testing.T) {
	// The planes report and never log, and the trainer's metric hook is the only
	// reader, so these have to answer even on an empty fleet.
	s := newStack(t, 4096)
	ctx := context.Background()

	fleet, err := s.control.Fleet(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("fleet: %v", err)
	}
	if len(fleet.GetNodes()) != 1 {
		t.Fatalf("fleet reports %d nodes, want 1", len(fleet.GetNodes()))
	}

	quotaReport, err := s.control.Quota(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("quota: %v", err)
	}
	if _, present := quotaReport.GetClasses()["rollout"]; !present {
		t.Fatalf("quota report is missing the rollout class: %v", quotaReport.GetClasses())
	}
}

func TestAGroupIsAdmittedMemberByMember(t *testing.T) {
	// A group is a completion unit rather than a scheduling unit, so no member
	// holds capacity while a sibling queues.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	group, err := s.control.CreateGroup(ctx, &v1.CreateGroupRequest{
		Specs: []*v1.SandboxSpec{createSpec(64), createSpec(64), createSpec(64)},
	})
	if err != nil {
		t.Fatalf("create group: %v", err)
	}
	defer func() {
		for _, member := range group.GetMembers() {
			s.control.Release(context.Background(), member.GetHandle())
		}
	}()

	if len(group.GetMembers()) != 3 {
		t.Fatalf("group has %d members, want 3", len(group.GetMembers()))
	}
}

func TestAFailedGroupLeavesNothingCharged(t *testing.T) {
	// 128MB takes two members; the third cannot fit and the first two must be
	// returned. The third queues rather than failing at once, so the bound is on
	// the test.
	s := newStack(t, 128)
	ctx, cancel := context.WithTimeout(context.Background(), heldOnAFullFleet)
	defer cancel()
	before := s.ledger.Headroom("rollout").MemoryMB

	_, err := s.control.CreateGroup(ctx, &v1.CreateGroupRequest{
		Specs: []*v1.SandboxSpec{createSpec(64), createSpec(64), createSpec(64)},
	})

	if err == nil {
		t.Fatal("a group larger than the fleet must fail")
	}
	if after := s.ledger.Headroom("rollout").MemoryMB; after != before {
		t.Fatalf("quota went %d -> %d; a failed group must leave nothing charged", before, after)
	}
}

func TestExecOnAReleasedSandboxIsReportedAsALostSession(t *testing.T) {
	// A caller has to be able to tell "my session is gone" from "the node did not
	// answer": the first means its own state is gone and no retry brings it back,
	// and reading it as the second corrupts whatever the caller does next.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	handle := backend.Handle{
		Backend:   created.GetHandle().GetBackend(),
		SandboxID: created.GetHandle().GetSandboxId(),
		NodeID:    created.GetHandle().GetNodeId(),
	}
	if _, err := s.control.Release(ctx, created.GetHandle()); err != nil {
		t.Fatalf("release: %v", err)
	}

	// Through the node-plane handler, which is one of the two places a failure
	// could be flattened into a transport fault on its way to the caller.
	payload, err := json.Marshal(map[string]any{
		"backend": handle.Backend, "sandbox_id": handle.SandboxID, "node_id": handle.NodeID,
		"command": "echo hi",
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	listener := NewNodeListener(s.node, nil)
	_, err = listener.exec(ctx, payload)
	if err == nil {
		t.Fatal("a command against a released sandbox must fail rather than run")
	}
	if reported := classify(err); reported.Code != "session_lost" {
		t.Fatalf("the wire classified it %q, want session_lost (error: %v)", reported.Code, err)
	} else if reported.ExitReason != string(node.ExitReleased) {
		t.Fatalf("the attribution is %q, want %q so a post mortem reads why it went",
			reported.ExitReason, node.ExitReleased)
	}
}

func TestTheReportSamplesPressureOnce(t *testing.T) {
	// A report is not a measurement. Calling the sampler twice per report made the
	// CPU figure a rate over the microseconds between two calls inside one RPC, which
	// is noise, and it disagreed with the sample admission had just stored.
	s := newStack(t, 4096)
	calls := 0
	s.node.pressure = func() node.Pressure {
		calls++
		return node.Pressure{MemUsedPct: 0.1, SampledAt: time.Now(), Valid: true}
	}

	s.node.View()

	if calls != 1 {
		t.Errorf("a report sampled pressure %d times, want 1", calls)
	}
}
