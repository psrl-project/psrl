package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net"
	"sort"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
)

// Ensure unused imports are referenced (node is used for the nil check below).
var _ = (*node.Admission)(nil)

// -- NodeListener round-trip tests ---------------------------------------------

// fakeNodeServer is a minimal Node whose Admit always admits and CreateOn
// always succeeds. It is enough to prove the wire protocol without a real runtime.
type fakeNodeServer struct {
	nodeID      string
	admitCalls  int
	createCalls int
	execCalls   int
}

func (f *fakeNodeServer) admit(_ context.Context, _ *jsonSpec) (map[string]any, error) {
	f.admitCalls++
	return map[string]any{
		"admitted":    true,
		"lease_id":    "lease-1",
		"gpu_indices": []int32{},
		"refusal":     "",
	}, nil
}

// startFakeNodeListener starts a NodeListener backed by a real Node (built with
// fake backends that never hit the kernel). Returns the listener address and a
// cleanup function.
//
// Because NewNode requires a real Admission and Lifecycle and those are not
// cheap to construct with fakes, we test the wire protocol directly instead —
// we start a minimal JSON listener that mimics the methods and assert the
// RemoteNodeClient parses replies correctly.
func startFakeJSONNodeServer(t *testing.T) (address string) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go serveOneFakeNodeConn(conn)
		}
	}()
	t.Cleanup(func() { l.Close() })
	return l.Addr().String()
}

func serveOneFakeNodeConn(conn net.Conn) {
	defer conn.Close()
	for {
		var header [4]byte
		if err := readFull(conn, header[:]); err != nil {
			return
		}
		size := binary.BigEndian.Uint32(header[:])
		if size == 0 || size > 64<<20 {
			return
		}
		body := make([]byte, size)
		if err := readFull(conn, body); err != nil {
			return
		}
		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			writeWireReply(conn, reply{Error: &wireError{Code: "invalid", Message: err.Error()}})
			continue
		}
		result := fakeDispatch(req)
		writeWireReply(conn, reply{Result: result})
	}
}

func readFull(conn net.Conn, buf []byte) error {
	for got := 0; got < len(buf); {
		n, err := conn.Read(buf[got:])
		got += n
		if err != nil {
			return err
		}
	}
	return nil
}

func writeWireReply(conn net.Conn, out reply) {
	encoded, _ := json.Marshal(out)
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(encoded)))
	_, _ = conn.Write(append(header[:], encoded...))
}

func fakeDispatch(req request) any {
	switch req.Method {
	case "admit":
		return map[string]any{
			"admitted": true, "lease_id": "lease-1", "gpu_indices": []int32{}, "refusal": "",
		}
	case "create_on":
		return map[string]any{
			"handle":       map[string]any{"backend": "docker", "sandbox_id": "sb-1", "node_id": "n1"},
			"capabilities": map[string]any{"features": []string{}, "resume_level": "", "pause_modes": []string{}},
			"agent":        map[string]any{"address": "", "headers": map[string]string{}, "callback_host_alias": "", "callback_port": int32(0)},
			"warm_start":   false,
		}
	case "release_on":
		return map[string]any{}
	case "status_on":
		return map[string]any{"status": "running"}
	case "exec":
		return map[string]any{"exit_code": 0, "stdout": "hello", "stderr": "", "truncated": false}
	case "read_bytes":
		return map[string]any{"data": "aGVsbG8="}
	case "write_bytes":
		return map[string]any{}
	case "report":
		return nodeViewToWire(placement.NodeView{
			NodeID:          "n1",
			SeenAt:          time.Now(),
			LiveSandboxes:   2,
			Envelope:        placement.Headroom{MemoryMB: 4096, CPUMillis: 4000},
			ClassHeadroom:   map[string]placement.Headroom{"default": {MemoryMB: 2048}},
			ImageDigests:    map[string]struct{}{},
			ImageReferences: map[string]struct{}{},
			Labels:          map[string]struct{}{},
		})
	default:
		return map[string]any{"error": "unknown"}
	}
}

// TestRemoteNodeClientAdmit proves the admit round-trip: request serialised,
// reply parsed, leaseID and admitted flag returned correctly.
func TestRemoteNodeClientAdmit(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)

	leaseID, gpus, refusal, err := client.admit(context.Background(), backend.Spec{ResourceClass: "default"})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if refusal != "" {
		t.Fatalf("expected no refusal, got %q", refusal)
	}
	if leaseID != "lease-1" {
		t.Fatalf("lease_id: got %q, want lease-1", leaseID)
	}
	_ = gpus
}

func TestRemoteNodeClientCreateOn(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)

	created, err := client.createOn(context.Background(), "lease-1", "docker", backend.Spec{}, "")
	if err != nil {
		t.Fatalf("create_on: %v", err)
	}
	if created.Handle.SandboxID != "sb-1" {
		t.Errorf("sandbox_id: got %q, want sb-1", created.Handle.SandboxID)
	}
}

func TestRemoteNodeClientReleaseOn(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "n1"}
	if err := client.releaseOn(context.Background(), handle); err != nil {
		t.Fatalf("release_on: %v", err)
	}
}

func TestRemoteNodeClientStatusOn(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "n1"}
	state, err := client.statusOn(context.Background(), handle)
	if err != nil {
		t.Fatalf("status_on: %v", err)
	}
	if state != "running" {
		t.Errorf("status: got %q, want running", state)
	}
}

func TestRemoteNodeClientExecOn(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "n1"}
	result, err := client.execOn(context.Background(), handle, "echo hello", "", nil, backend.ExecOptions{})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if result.ExitCode != 0 || result.Stdout != "hello" {
		t.Errorf("exec: got code=%d output=%q, want 0/hello", result.ExitCode, result.Stdout)
	}
}

func TestRemoteNodeClientReadBytesOn(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "n1"}
	data, err := client.readBytesOn(context.Background(), handle, "/tmp/f")
	if err != nil {
		t.Fatalf("read_bytes: %v", err)
	}
	if data != "aGVsbG8=" {
		t.Errorf("data: got %q, want aGVsbG8=", data)
	}
}

func TestRemoteNodeClientReport(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	client := NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second)
	wire, err := client.report(context.Background())
	if err != nil {
		t.Fatalf("report: %v", err)
	}
	if wire.NodeID != "n1" {
		t.Errorf("node_id: got %q, want n1", wire.NodeID)
	}
	if wire.LiveSandboxes != 2 {
		t.Errorf("live_sandboxes: got %d, want 2", wire.LiveSandboxes)
	}
	view := nodeViewFromWire(wire)
	if view.Envelope.MemoryMB != 4096 {
		t.Errorf("envelope memory_mb: got %d, want 4096", view.Envelope.MemoryMB)
	}
}

// -- FleetNodeClient routing tests --------------------------------------------

func TestFleetNodeClientRoutesToCorrectNode(t *testing.T) {
	addr1 := startFakeJSONNodeServer(t)
	addr2 := startFakeJSONNodeServer(t)
	fleet := NewFleetNodeClient(map[string]*RemoteNodeClient{
		"n1": NewRemoteNodeClient(addr1, 5*time.Second, 5*time.Second),
		"n2": NewRemoteNodeClient(addr2, 5*time.Second, 5*time.Second),
	})

	// n2's sandbox — must reach n2
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "n2"}
	state, err := fleet.StatusOn(context.Background(), handle)
	if err != nil {
		t.Fatalf("status_on: %v", err)
	}
	if state != "running" {
		t.Errorf("state: got %q, want running", state)
	}
}

func TestFleetNodeClientRefusesUnknownNode(t *testing.T) {
	fleet := NewFleetNodeClient(map[string]*RemoteNodeClient{
		"n1": NewRemoteNodeClient("127.0.0.1:1", 1*time.Second, 1*time.Second),
	})
	handle := backend.Handle{Backend: "docker", SandboxID: "sb-1", NodeID: "nowhere"}
	_, err := fleet.StatusOn(context.Background(), handle)
	if err == nil {
		t.Fatal("expected error for unconfigured node, got nil")
	}
}

func TestFleetNodeClientNodesAreSorted(t *testing.T) {
	fleet := NewFleetNodeClient(map[string]*RemoteNodeClient{
		"n3": NewRemoteNodeClient("127.0.0.1:1", 1*time.Second, 1*time.Second),
		"n1": NewRemoteNodeClient("127.0.0.1:2", 1*time.Second, 1*time.Second),
		"n2": NewRemoteNodeClient("127.0.0.1:3", 1*time.Second, 1*time.Second),
	})
	ids := fleet.order
	if !sort.StringsAreSorted(ids) {
		t.Errorf("order is not sorted: %v", ids)
	}
}

// -- PollFleet wires views into the monitor -----------------------------------

func TestPollFleetPublishesNodeViews(t *testing.T) {
	addr := startFakeJSONNodeServer(t)
	fleet := NewFleetNodeClient(map[string]*RemoteNodeClient{
		"n1": NewRemoteNodeClient(addr, 5*time.Second, 5*time.Second),
	})

	m := monitor.New(60 * time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go fleet.PollFleet(ctx, m, 20*time.Millisecond, nil)

	deadline := time.Now().Add(2 * time.Second)
	var views []placement.NodeView
	for time.Now().Before(deadline) {
		views = m.Fleet()
		if len(views) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(views) == 0 {
		t.Fatal("PollFleet: monitor has no views after 2 seconds")
	}
	if views[0].NodeID != "n1" {
		t.Errorf("node_id: got %q, want n1", views[0].NodeID)
	}
}

// -- LocalNodeClient satisfies the extended interface -------------------------

func TestLocalNodeClientSatisfiesNodeClient(t *testing.T) {
	// Compile-time check that LocalNodeClient implements the full NodeClient
	// interface including the new Exec/ReadBytes/WriteBytes methods. If it does
	// not, this test will not compile.
	var _ NodeClient = (*LocalNodeClient)(nil)
}

// -- nodeViewToWire / nodeViewFromWire roundtrip ------------------------------

func TestNodeViewWireRoundtrip(t *testing.T) {
	original := placement.NodeView{
		NodeID:        "n1",
		SeenAt:        time.Now().Truncate(time.Second),
		LiveSandboxes: 3,
		CPUUsedPct:    0.42,
		Draining:      false,
		Envelope:      placement.Headroom{MemoryMB: 8192, CPUMillis: 8000, GPUCount: 2},
		ClassHeadroom: map[string]placement.Headroom{
			"default": {MemoryMB: 4096, CPUMillis: 2000},
			"gpu":     {MemoryMB: 2048, GPUCount: 1},
		},
		Backends: []placement.BackendCapability{{
			Name:     "docker",
			Features: map[string]struct{}{"host_mount": {}},
		}},
		ImageDigests:    map[string]struct{}{"sha256:abc": {}},
		ImageReferences: map[string]struct{}{"alpine:latest": {}},
		Labels:          map[string]struct{}{},
	}

	wire := nodeViewToWire(original)
	restored := nodeViewFromWire(wire)

	if restored.NodeID != original.NodeID {
		t.Errorf("node_id: got %q, want %q", restored.NodeID, original.NodeID)
	}
	if restored.LiveSandboxes != original.LiveSandboxes {
		t.Errorf("live_sandboxes: got %d, want %d", restored.LiveSandboxes, original.LiveSandboxes)
	}
	if restored.Envelope.MemoryMB != original.Envelope.MemoryMB {
		t.Errorf("envelope.memory_mb: got %d, want %d", restored.Envelope.MemoryMB, original.Envelope.MemoryMB)
	}
	if len(restored.ClassHeadroom) != len(original.ClassHeadroom) {
		t.Errorf("class_headroom: got %d entries, want %d", len(restored.ClassHeadroom), len(original.ClassHeadroom))
	}
	if _, ok := restored.ImageDigests["sha256:abc"]; !ok {
		t.Error("image_digests: sha256:abc missing after roundtrip")
	}
	if _, ok := restored.ImageReferences["alpine:latest"]; !ok {
		t.Error("image_references: alpine:latest missing after roundtrip")
	}
	if len(restored.Backends) != 1 || restored.Backends[0].Name != "docker" {
		t.Errorf("backends: got %v, want [{docker}]", restored.Backends)
	}
	if _, ok := restored.Backends[0].Features["host_mount"]; !ok {
		t.Error("backend features: host_mount missing after roundtrip")
	}
	// SeenAt is filled by nodeViewFromWire at time of receipt, not preserved.
}

// Stub types so the LocalNodeClient nil-check compiles without needing real
// node.Admission / node.Lifecycle plumbing.
var _ = (*node.Admission)(nil)

// A node that accepts the connection and then never answers must not hold the
// control plane. Dialing is not the only step that can stall, so the whole
// exchange is bounded rather than only the connect.
func TestANodePlaneCallTimesOutAgainstASilentPeer(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	release := make(chan struct{})
	defer close(release)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		<-release
	}()

	client := NewRemoteNodeClient(ln.Addr().String(), 200*time.Millisecond, 200*time.Millisecond)
	done := make(chan error, 1)
	go func() {
		_, err := client.report(context.Background())
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("a silent peer must not read as a successful report")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the call never returned, so a node-plane exchange is not bounded")
	}
}
