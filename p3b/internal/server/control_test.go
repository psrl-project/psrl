package server

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// providerStub is a backend that places on its own cluster, so place() needs no
// monitor report and the test can focus on the ledger.
type providerStub struct{}

func (providerStub) Name() string                       { return "stub" }
func (providerStub) Mode() backend.SchedulingMode       { return backend.SchedulingProvider }
func (providerStub) Capabilities() backend.Capabilities { return backend.Capabilities{} }

// stubSandboxIDs gives every created sandbox its own id. A stub that hands out one id to
// every caller is not a stub of anything: the lifecycle indexes by sandbox id, so the
// second sandbox silently becomes the first one's record and a release leaves the rest
// charged. A test of more than one sandbox needs them to be distinguishable.
var stubSandboxIDs atomic.Uint64

func (providerStub) Create(context.Context, string, backend.Spec, string) (backend.Created, error) {
	id := stubSandboxIDs.Add(1)
	return backend.Created{
		Handle: backend.Handle{Backend: "stub", SandboxID: fmt.Sprintf("sb-%d", id)},
	}, nil
}
func (providerStub) Release(context.Context, backend.Handle) error          { return nil }
func (providerStub) Status(context.Context, backend.Handle) (string, error) { return "running", nil }

// stubNodes records what the control plane asked a node to do, so a test can
// assert on the calls that never reach a real node.
type stubNodes struct {
	released   []backend.Handle
	renewed    []string
	releaseErr error
}

func (s *stubNodes) Admit(context.Context, string, backend.Spec) (string, []int32, string, error) {
	return "lease-1", nil, "", nil
}
func (s *stubNodes) CreateOn(
	context.Context, string, string, string, backend.Spec, string,
) (backend.Created, error) {
	return backend.Created{Handle: backend.Handle{Backend: "stub", SandboxID: "sb-1"}}, nil
}
func (s *stubNodes) ReleaseOn(_ context.Context, handle backend.Handle) error {
	s.released = append(s.released, handle)
	return s.releaseErr
}
func (s *stubNodes) StatusOn(context.Context, backend.Handle) (string, error) { return "running", nil }
func (s *stubNodes) RenewOwnerOn(_ context.Context, _, owner string) error {
	s.renewed = append(s.renewed, owner)
	return nil
}
func (s *stubNodes) ExecOn(
	context.Context, backend.Handle, string, string, map[string]string, backend.ExecOptions,
) (backend.ExecResult, error) {
	return backend.ExecResult{}, nil
}
func (s *stubNodes) ReadBytesOn(context.Context, backend.Handle, string) (string, error) {
	return "", nil
}
func (s *stubNodes) WriteBytesOn(context.Context, backend.Handle, string, string) error { return nil }

// newTestControl returns a control plane over a stub backend and a ledger whose
// clock the caller drives.
func newTestControl(t *testing.T, nodes NodeClient, clock func() time.Time) (*Control, *quota.Ledger) {
	t.Helper()
	fleet := monitor.New(2 * time.Minute)
	place, err := placement.New(placement.Config{
		NodeTTL: 2 * time.Minute, ReservationTTL: time.Minute, SweepInterval: 15 * time.Second,
	}, fleet)
	if err != nil {
		t.Fatalf("placement.New: %v", err)
	}
	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: 4096, CPUMillis: 8000, Sandboxes: 1 << 20},
		Classes:  map[string]quota.ClassShare{"default": {Guaranteed: 1.0}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		t.Fatalf("quota.New: %v", err)
	}
	ledger.SetClock(clock)
	registry, err := backend.NewRegistry([]backend.Backend{providerStub{}}, "")
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	control, err := NewControl(ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet, Nodes: nodes,
	})
	if err != nil {
		t.Fatalf("NewControl: %v", err)
	}
	return control, ledger
}

// The node reclaims a sandbox by owner silence, so the ledger has to charge the
// same owner identity the caller heartbeats. Keying it on anything else means
// the heartbeat renews nothing and the grant is reclaimed mid-run.
func TestControlRenewsTheGrantTheNodeSees(t *testing.T) {
	var clock time.Time
	control, ledger := newTestControl(t, &stubNodes{}, func() time.Time { return clock })

	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: &v1.SandboxSpec{
		ResourceClass: "default",
		Source:        &v1.Source{Kind: v1.Source_IMAGE, Reference: "alpine:latest"},
		Resources:     &v1.Resources{CpuCount: floatPtr(1), MemoryMb: intPtr(512)},
		Metadata:      map[string]string{"owner_id": "w1-worker"},
	}}); err != nil {
		t.Fatalf("create: %v", err)
	}

	// Well past the lease, so only a heartbeat that matches the charged owner can
	// keep the grant.
	clock = clock.Add(2 * time.Hour)
	control.RenewOwner(context.Background(), "w1-worker")
	clock = clock.Add(30 * time.Minute)

	if got := ledger.Report()["default"].Granted.MemoryMB; got != 512 {
		t.Errorf("a grant renewed by its owner must stay charged, got %dMB", got)
	}
}

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int64) *int64       { return &v }

// A restarted control plane holds no record of a sandbox the fleet is still
// running. Release returning success without asking would leak the container,
// so the handle's node has to be used to reach the runtime that owns it.
func TestReleaseWithUnknownRecordCallsTheNode(t *testing.T) {
	nodes := &stubNodes{}
	control, _ := newTestControl(t, nodes, time.Now)

	_, err := control.Release(context.Background(), &v1.SandboxHandle{
		Backend: "stub", SandboxId: "sb-1", NodeId: "node-a",
	})
	if err != nil {
		t.Fatalf("release: %v", err)
	}
	if len(nodes.released) != 1 {
		t.Fatalf("release must ask the node, got %d calls", len(nodes.released))
	}
	if got := nodes.released[0]; got.NodeID != "node-a" || got.SandboxID != "sb-1" {
		t.Errorf("release reached the wrong handle: %+v", got)
	}
}

// A node that answered but could not destroy is not success. The sandbox may
// still exist, so the caller has to be able to retry.
func TestReleaseWithUnknownRecordSurfacesANodeFailure(t *testing.T) {
	nodes := &stubNodes{releaseErr: errors.New("daemon refused")}
	control, _ := newTestControl(t, nodes, time.Now)

	_, err := control.Release(context.Background(), &v1.SandboxHandle{
		Backend: "stub", SandboxId: "sb-1", NodeId: "node-a",
	})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("want Unavailable so the caller retries, got %v", err)
	}
}

// TestGrantIDsAreUnique pins the property the quota ledger relies on: concurrent
// creates must not share a grant id, or every create after the first one in a tick
// fails with "quota grant already exists" instead of running.
func TestGrantIDsAreUnique(t *testing.T) {
	const n = 4096
	ids := make(chan string, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			ids <- newGrantID()
		}()
	}
	wg.Wait()
	close(ids)
	seen := make(map[string]struct{}, n)
	for id := range ids {
		if _, duplicate := seen[id]; duplicate {
			t.Fatalf("duplicate grant id %q among %d concurrent creates", id, n)
		}
		seen[id] = struct{}{}
	}
}
