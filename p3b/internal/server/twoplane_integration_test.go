package server_test

// Two-plane integration test: verifies the design claim that a caller on the
// control plane can create, exec, and release a sandbox that lives on a remote
// node, with the same code path as a single-process deployment.
//
// Design step 2.4 exit criterion: harness_agent_loop and the MiniSWE runner are
// unchanged while running against a remote node. The proxy to that exit criterion
// at the Go layer is this test: one goroutine runs a node-plane listener (the
// shape a standalone node process would have), another runs the control plane
// talking to it via FleetNodeClient, and the test verifies that create/exec/release
// round-trips over the JSON node-plane protocol without the control plane and
// the node knowing they share a machine.
//
// What this covers:
//   - RemoteNodeClient admit → leaseID crosses the wire correctly
//   - RemoteNodeClient createOn → backend.Created including Handle crosses
//   - RemoteNodeClient statusOn → string state crosses
//   - RemoteNodeClient execOn → exit code and stdout cross
//   - RemoteNodeClient releaseOn → sandbox is gone
//   - FleetNodeClient routes by nodeID
//   - NodeListener dispatches all five methods without error
//
// What this does not cover: a real container runtime. The backend is a fakeExecBackend
// that keeps sandboxes in memory and returns a canned exec output. The test is
// about the wire protocol and the routing, not about Docker.

import (
	"context"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/server"
	"psrl.dev/sandboxd/internal/timing"
)

// fakeExecBackend is a minimal backend that keeps sandboxes in memory.
type fakeExecBackend struct {
	name    string
	created map[string]bool
}

func newFakeExecBackend(name string) *fakeExecBackend {
	return &fakeExecBackend{name: name, created: map[string]bool{}}
}

func (f *fakeExecBackend) Name() string { return f.name }
func (f *fakeExecBackend) Mode() backend.SchedulingMode {
	return backend.SchedulingDirect
}
func (f *fakeExecBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{}
}
func (f *fakeExecBackend) Nodes(_ context.Context) ([]string, error) {
	return []string{"local"}, nil
}
func (f *fakeExecBackend) Create(_ context.Context, nodeID string, spec backend.Spec, _ string) (backend.Created, error) {
	id := "sb-" + spec.WorkflowID
	f.created[id] = true
	return backend.Created{
		Handle: backend.Handle{Backend: f.name, SandboxID: id, NodeID: nodeID},
	}, nil
}
func (f *fakeExecBackend) Release(_ context.Context, handle backend.Handle) error {
	delete(f.created, handle.SandboxID)
	return nil
}
func (f *fakeExecBackend) Status(_ context.Context, handle backend.Handle) (string, error) {
	if f.created[handle.SandboxID] {
		return "running", nil
	}
	return "terminated", nil
}
func (f *fakeExecBackend) Exec(
	_ context.Context, _ backend.Handle, _, command, _ string, _ map[string]string, _ backend.ExecOptions,
) (backend.ExecResult, error) {
	return backend.ExecResult{Stdout: "hello-from-" + f.name}, nil
}

// buildNodeForTest builds a real Node backed by the fake backend, returning it
// and the admission/lifecycle it uses so the test can control them.
func buildNodeForTest(t *testing.T, nodeID string, b backend.Backend) *server.Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 4000},
		Classes:  map[string]node.ClassShare{"default": {Guaranteed: 0.5, Max: 1.0}},
		LeaseTTL: 30 * time.Second,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle(nodeID, gate, timing.SweepInterval)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	n, err := server.NewNode(server.NodeConfig{
		NodeID: nodeID, Admission: gate, Lifecycle: life, Backends: []backend.Backend{b},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return n
}

// startNodeListener starts a NodeListener on a random TCP port and returns the address.
func startNodeListener(t *testing.T, n *server.Node) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	nl := server.NewNodeListener(n, nil)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		l.Close()
	})
	go func() { _ = nl.Serve(ctx, l) }()
	return l.Addr().String()
}

// TestTwoPlaneCreateExecRelease is the step 2.4 integration test.
//
// A control plane talks to a node plane over the node-plane JSON protocol.
// The test verifies that admit → create → exec → status → release all complete
// and that the exec output and sandbox state are correct at each step.
func TestTwoPlaneCreateExecRelease(t *testing.T) {
	b := newFakeExecBackend("fake")
	n := buildNodeForTest(t, "node-1", b)
	addr := startNodeListener(t, n)

	remote := server.NewRemoteNodeClient(addr, 10*time.Second, 10*time.Second)
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-1": remote,
	})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-twoplane-test",
		Resources:     backend.Resources{MemoryMB: 256, CPUCount: 0.5},
	}

	// Step 1: admission
	leaseID, gpus, refusal, err := fleet.Admit(ctx, "node-1", spec)
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if refusal != "" {
		t.Fatalf("unexpected refusal: %q", refusal)
	}
	if leaseID == "" {
		t.Fatal("admit returned empty lease_id")
	}
	_ = gpus

	// Step 2: create
	spec.AssignedGPUs = gpus
	created, err := fleet.CreateOn(ctx, "node-1", leaseID, "fake", spec, "")
	if err != nil {
		t.Fatalf("create_on: %v", err)
	}
	if created.Handle.SandboxID == "" {
		t.Fatal("create_on returned empty sandbox_id")
	}
	if created.Handle.NodeID != "node-1" {
		t.Errorf("node_id: got %q, want node-1", created.Handle.NodeID)
	}

	// Step 3: status should be running
	state, err := fleet.StatusOn(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status_on: %v", err)
	}
	if state != "running" {
		t.Errorf("status before release: got %q, want running", state)
	}

	// Step 4: exec — the node proxies for a backend whose sandboxes have no
	// in-sandbox agent, which is exactly what the docker backend does.
	result, err := fleet.ExecOn(ctx, created.Handle, "echo hello", "", nil, backend.ExecOptions{})
	if err != nil {
		t.Fatalf("exec_on: %v", err)
	}
	if result.ExitCode != 0 {
		t.Errorf("exec exit code: got %d, want 0", result.ExitCode)
	}
	if !strings.Contains(result.Stdout, "hello-from-fake") {
		t.Errorf("exec stdout: got %q, want it to contain hello-from-fake", result.Stdout)
	}

	// Step 5: release
	if err := fleet.ReleaseOn(ctx, created.Handle); err != nil {
		t.Fatalf("release_on: %v", err)
	}

	// Step 6: status should now be terminated
	state, err = fleet.StatusOn(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status_on after release: %v", err)
	}
	if state != "terminated" {
		t.Errorf("status after release: got %q, want terminated", state)
	}
}

// TestTwoPlaneMultiNodeRouting verifies that FleetNodeClient routes to the
// correct node when there are two nodes configured.
func TestTwoPlaneMultiNodeRouting(t *testing.T) {
	b1 := newFakeExecBackend("fake1")
	b2 := newFakeExecBackend("fake2")
	n1 := buildNodeForTest(t, "node-a", b1)
	n2 := buildNodeForTest(t, "node-b", b2)
	addr1 := startNodeListener(t, n1)
	addr2 := startNodeListener(t, n2)

	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-a": server.NewRemoteNodeClient(addr1, 5*time.Second, 5*time.Second),
		"node-b": server.NewRemoteNodeClient(addr2, 5*time.Second, 5*time.Second),
	})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-routing-test",
		Resources:     backend.Resources{MemoryMB: 256},
	}

	// Place on node-b explicitly.
	leaseID, _, refusal, err := fleet.Admit(ctx, "node-b", spec)
	if err != nil || refusal != "" {
		t.Fatalf("admit node-b: err=%v refusal=%q", err, refusal)
	}
	created, err := fleet.CreateOn(ctx, "node-b", leaseID, "fake2", spec, "")
	if err != nil {
		t.Fatalf("create_on node-b: %v", err)
	}
	if created.Handle.NodeID != "node-b" {
		t.Errorf("handle node_id: got %q, want node-b", created.Handle.NodeID)
	}

	// Exec must reach node-b's backend, not node-a's.
	result, err := fleet.ExecOn(ctx, created.Handle, "echo hello", "", nil, backend.ExecOptions{})
	if err != nil {
		t.Fatalf("exec_on: %v", err)
	}
	if !strings.Contains(result.Stdout, "hello-from-fake2") {
		t.Errorf("exec reached wrong node: got stdout %q, want hello-from-fake2", result.Stdout)
	}

	// node-a must not have seen the sandbox.
	if len(b1.created) != 0 {
		t.Errorf("node-a saw %d creates but should have seen none", len(b1.created))
	}
}

// TestTwoPlaneExecOnUnknownNodeReturnsError verifies that ExecOn named to a
// node not in the fleet returns an error rather than panicking or timing out.
func TestTwoPlaneExecOnUnknownNodeReturnsError(t *testing.T) {
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-a": server.NewRemoteNodeClient("127.0.0.1:1", 1*time.Second, 1*time.Second),
	})
	handle := backend.Handle{Backend: "fake", SandboxID: "sb-1", NodeID: "node-missing"}
	_, err := fleet.ExecOn(context.Background(), handle, "echo hi", "", nil, backend.ExecOptions{})
	if err == nil {
		t.Fatal("expected error for sandbox on an unknown node, got nil")
	}
}

// TestCapabilitiesSurviveTheLocalNodePath pins a defect that only a
// single-process deployment could have.
//
// The local node client rebuilt capabilities from the proto with the resume
// level alone, dropping the feature list and the pause modes. The remote client
// carried all three, so a fleet deployment behaved correctly while a
// single-machine one reported a backend that declared nothing it could do --
// and a capability check at the SDK ("can this sandbox take a full-state
// snapshot?") answered no for a backend whose whole purpose is answering yes.
//
// Asserted on the local path specifically, because that is the one that was
// wrong and the one a developer runs by default.
func TestCapabilitiesSurviveTheLocalNodePath(t *testing.T) {
	declared := backend.Capabilities{
		Features:    []string{"hibernate", "full_state_snapshot", "resume_anywhere", "native_fork"},
		ResumeLevel: "full_state",
		PauseModes:  []string{"hibernate"},
	}
	b := &capabilityBackend{name: "fake", declared: declared}
	n := buildNodeForTest(t, "node-caps", b)

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-caps",
		Resources:     backend.Resources{MemoryMB: 128},
	}
	local := server.NewLocalNodeClient(n)
	leaseID, _, refusal, err := local.Admit(ctx, "node-caps", spec)
	if err != nil || refusal != "" {
		t.Fatalf("admit: err=%v refusal=%q", err, refusal)
	}
	created, err := local.CreateOn(ctx, "node-caps", leaseID, "fake", spec, "")
	if err != nil {
		t.Fatalf("create_on: %v", err)
	}

	if created.Capabilities.ResumeLevel != "full_state" {
		t.Errorf("resume level: got %q, want full_state", created.Capabilities.ResumeLevel)
	}
	if len(created.Capabilities.Features) != len(declared.Features) {
		t.Fatalf("the create returned %d features, want %d: %v",
			len(created.Capabilities.Features), len(declared.Features), created.Capabilities.Features)
	}
	for _, want := range declared.Features {
		found := false
		for _, got := range created.Capabilities.Features {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("feature %q was dropped crossing the node boundary", want)
		}
	}
	if len(created.Capabilities.PauseModes) != 1 || created.Capabilities.PauseModes[0] != "hibernate" {
		t.Errorf("pause modes: got %v, want [hibernate]", created.Capabilities.PauseModes)
	}
}

// capabilityBackend is a fake that declares a full capability set, so the test
// can tell which parts survive the round trip.
type capabilityBackend struct {
	name     string
	declared backend.Capabilities
}

func (c *capabilityBackend) Name() string                       { return c.name }
func (c *capabilityBackend) Mode() backend.SchedulingMode       { return backend.SchedulingDirect }
func (c *capabilityBackend) Capabilities() backend.Capabilities { return c.declared }
func (c *capabilityBackend) Nodes(context.Context) ([]string, error) {
	return []string{"node-caps"}, nil
}
func (c *capabilityBackend) Create(
	_ context.Context, nodeID string, spec backend.Spec, _ string,
) (backend.Created, error) {
	return backend.Created{
		Handle:       backend.Handle{Backend: c.name, SandboxID: "sb-caps", NodeID: nodeID},
		Capabilities: c.declared,
	}, nil
}
func (c *capabilityBackend) Release(context.Context, backend.Handle) error { return nil }
func (c *capabilityBackend) Status(context.Context, backend.Handle) (string, error) {
	return "running", nil
}

// TestRenewOwnerCrossesTheNodePlane pins the control-to-node heartbeat path.
//
// A node reclaims an owner's sandboxes once it has been silent past the lease
// TTL, so the control plane has to relay the heartbeat over the node plane. A
// renewal that stopped at the control plane would leave every sandbox on every
// node to be reclaimed as an orphan while its run is still alive.
func TestRenewOwnerCrossesTheNodePlane(t *testing.T) {
	b := newFakeExecBackend("fake")
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 4000},
		Classes:  map[string]node.ClassShare{"default": {Guaranteed: 0.5, Max: 1.0}},
		LeaseTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	// The gate is read from the listener's own goroutines, so its clock is driven
	// through an atomic rather than a plain closure variable.
	var clockNs atomic.Int64
	clockNs.Store(time.Now().UnixNano())
	gate.SetClock(func() time.Time { return time.Unix(0, clockNs.Load()) })
	life, err := node.NewLifecycle("node-r", gate, timing.SweepInterval)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	n, err := server.NewNode(server.NodeConfig{
		NodeID: "node-r", Admission: gate, Lifecycle: life, Backends: []backend.Backend{b},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	addr := startNodeListener(t, n)
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-r": server.NewRemoteNodeClient(addr, 10*time.Second, 10*time.Second),
	})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "owner-r",
		Resources:     backend.Resources{MemoryMB: 128},
	}
	if _, _, refusal, err := fleet.Admit(ctx, "node-r", spec); err != nil || refusal != "" {
		t.Fatalf("admit: err=%v refusal=%q", err, refusal)
	}

	clockNs.Add(int64(90 * time.Second))
	if gate.OwnerAlive("owner-r") {
		t.Fatal("an owner silent past the lease TTL must read as gone")
	}

	if err := fleet.RenewOwnerOn(ctx, "node-r", "owner-r"); err != nil {
		t.Fatalf("renew_owner: %v", err)
	}
	if !gate.OwnerAlive("owner-r") {
		t.Fatal("renew_owner did not reach the node's admission")
	}
}

// deadlineBackend runs a command until its context ends, which is what a shell
// executing a long command looks like to the node.
type deadlineBackend struct{ *fakeExecBackend }

func (d *deadlineBackend) Exec(
	ctx context.Context, _ backend.Handle, _, _, _ string, _ map[string]string, _ backend.ExecOptions,
) (backend.ExecResult, error) {
	<-ctx.Done()
	return backend.ExecResult{ExitCode: 1}, ctx.Err()
}

// TestExecStopsAtTheCommandDeadlineAndKeepsTheSandbox pins the contract that made
// the ledger's collector timeouts possible to fix: the budget a caller sends with
// a command is enforced where the command runs, the failure crosses the node
// boundary as a command timeout rather than a transport fault, and the sandbox
// survives it.
func TestExecStopsAtTheCommandDeadlineAndKeepsTheSandbox(t *testing.T) {
	b := &deadlineBackend{fakeExecBackend: newFakeExecBackend("slow")}
	n := buildNodeForTest(t, "node-1", b)
	addr := startNodeListener(t, n)
	remote := server.NewRemoteNodeClient(addr, 10*time.Second, 10*time.Second)
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{"node-1": remote})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-deadline",
		Resources:     backend.Resources{MemoryMB: 256, CPUCount: 0.5},
	}
	leaseID, gpus, refusal, err := fleet.Admit(ctx, "node-1", spec)
	if err != nil || refusal != "" {
		t.Fatalf("admit: err=%v refusal=%q", err, refusal)
	}
	spec.AssignedGPUs = gpus
	created, err := fleet.CreateOn(ctx, "node-1", leaseID, "slow", spec, "")
	if err != nil {
		t.Fatalf("create_on: %v", err)
	}

	started := time.Now()
	_, err = fleet.ExecOn(ctx, created.Handle, "pytest", "", nil, backend.ExecOptions{Timeout: 100 * time.Millisecond})
	if err == nil {
		t.Fatal("a command past its deadline reported success")
	}
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("deadline failure crossed as %v (%v), want DeadlineExceeded", status.Code(err), err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("exec took %s for a 100ms deadline", elapsed)
	}
	// A command deadline is about the command: the sandbox and its filesystem stay.
	state, err := fleet.StatusOn(ctx, created.Handle)
	if err != nil || state != "running" {
		t.Fatalf("status after a command deadline: state=%q err=%v, want running", state, err)
	}
}
