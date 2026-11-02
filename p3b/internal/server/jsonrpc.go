package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/timing"
)

// JSONListener serves the SDK's wire format: a four-byte length then a JSON
// frame.
//
// JSON rather than gRPC for the Python side, deliberately. The SDK's job is to
// serialize and forward, the control path is about two calls per episode, and a
// generated stub would make the SDK's install depend on a protobuf runtime for
// no measurable gain. The message shapes are still the ones the proto defines,
// so the Go-to-Go path and a future generated SDK speak the same contract.
type JSONListener struct {
	control *Control
	node    *Node
	log     *slog.Logger

	// dataPlane resolves a node id to the address the SDK can reach that node's
	// own plane on, so a sandbox with no in-sandbox agent still gets a data-plane
	// address and its commands stay off the control plane. Nil in a combined
	// deployment, where the SDK already shares a socket with the node.
	dataPlane func(nodeID string) string

	mu       sync.Mutex
	listener net.Listener
	closed   bool
	conns    sync.WaitGroup
}

// NewJSONListener returns the SDK-facing server. The node may be nil on a
// control-only process; command calls then report that they have nowhere to go
// rather than failing obscurely.
func NewJSONListener(control *Control, node *Node, log *slog.Logger) *JSONListener {
	if log == nil {
		log = slog.Default()
	}
	return &JSONListener{control: control, node: node, log: log}
}

// WithDataPlane supplies the node-id-to-address lookup a fleet deployment needs,
// so a create reply can tell the SDK which node to send commands to.
func (j *JSONListener) WithDataPlane(resolve func(nodeID string) string) *JSONListener {
	j.dataPlane = resolve
	return j
}

// Serve accepts connections until the listener is closed.
func (j *JSONListener) Serve(ctx context.Context, listener net.Listener) error {
	j.mu.Lock()
	j.listener = listener
	j.mu.Unlock()

	go func() {
		<-ctx.Done()
		_ = j.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			j.mu.Lock()
			closed := j.closed
			j.mu.Unlock()
			if closed || errors.Is(err, net.ErrClosed) {
				j.conns.Wait()
				return nil
			}
			return err
		}
		j.conns.Add(1)
		go func() {
			defer j.conns.Done()
			j.handle(ctx, conn)
		}()
	}
}

// Close stops accepting and waits for live connections to drain.
func (j *JSONListener) Close() error {
	j.mu.Lock()
	if j.closed {
		j.mu.Unlock()
		return nil
	}
	j.closed = true
	listener := j.listener
	j.mu.Unlock()
	if listener != nil {
		return listener.Close()
	}
	return nil
}

type request struct {
	Method  string          `json:"method"`
	Payload json.RawMessage `json:"payload"`
}

type reply struct {
	Result any        `json:"result,omitempty"`
	Error  *wireError `json:"error,omitempty"`
}

type wireError struct {
	// Code is the service's classification, not a guess from the message: the
	// caller's response differs per type, and an infrastructure fault read as a
	// task result would corrupt a reward.
	Code    string `json:"code"`
	Message string `json:"message"`
	// ExitReason attributes a `session_lost` sandbox (released/reaped_idle/
	// reaped_lifetime/reclaimed_orphan/oom/exited/runtime_gone) so a post mortem
	// does not have to guess. Empty when the failure carries no attribution.
	ExitReason string `json:"exit_reason,omitempty"`
	// ExitCode is the runtime's exit code, when it observed one. It is a pointer
	// so "the process exited 0" and "the runtime never reported a code" are not
	// the same JSON value.
	ExitCode *int `json:"exit_code,omitempty"`
	// FinishedAt is when the runtime observed the stop, RFC 3339 in UTC. It is
	// the clock an operator correlates with the node's cgroup and kernel events.
	FinishedAt string `json:"finished_at,omitempty"`
	// StateError is the runtime's own account of the stop, or the inspect error
	// that made the exit code unknown. A loss with no code is diagnosable only
	// if this travels with it.
	StateError string `json:"state_error,omitempty"`
	// Preserved says whether the sandbox survived the failure. A command that ran out
	// of its budget keeps its runtime and only loses the shell session, so a caller can
	// continue with a fresh shell; a lost session has nothing left to continue in.
	Preserved bool `json:"preserved,omitempty"`
}

func (j *JSONListener) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	for {
		var header [4]byte
		if _, err := io.ReadFull(conn, header[:]); err != nil {
			// A client that closed its connection is the normal end of a session.
			return
		}
		size := binary.BigEndian.Uint32(header[:])
		if size == 0 || size > 64<<20 {
			j.log.Warn("sandbox SDK frame out of range", "bytes", size)
			return
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(conn, body); err != nil {
			return
		}
		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			j.write(conn, reply{Error: &wireError{Code: "invalid", Message: err.Error()}})
			continue
		}
		result, err := j.dispatch(ctx, req)
		if err != nil {
			j.write(conn, reply{Error: classify(err)})
			continue
		}
		j.write(conn, reply{Result: result})
	}
}

func (j *JSONListener) write(conn net.Conn, out reply) {
	encoded, err := json.Marshal(out)
	if err != nil {
		j.log.Error("sandbox SDK reply could not be encoded", "error", err)
		return
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(encoded)))
	if _, err := conn.Write(append(header[:], encoded...)); err != nil {
		j.log.Debug("sandbox SDK reply could not be written", "error", err)
	}
}

// classify maps a failure onto the attribution the caller acts on.
func classify(err error) *wireError {
	var code string
	if wireCode, ok := wireCodeOf(err); ok {
		// An error's own classification wins over the status code it happens to
		// carry, because the status vocabulary is too coarse to express the
		// difference between a missing file and a missing sandbox.
		code = wireCode
	} else {
		code = "internal"
		switch status.Code(err) {
		case codes.FailedPrecondition:
			code = "capability"
		case codes.ResourceExhausted:
			code = "capacity_timeout"
		case codes.OutOfRange:
			// The request can never fit its class, so retrying cannot help. Distinct
			// from capacity_timeout, which clears as sandboxes are released.
			code = "cannot_admit"
		case codes.NotFound:
			code = "session_lost"
		case codes.Unavailable:
			code = "transport"
		case codes.DeadlineExceeded:
			code = "command_timeout"
		}
	}
	out := &wireError{Code: code, Message: err.Error(), ExitReason: exitReasonOf(err)}
	// A deadline stops the command, not the sandbox: the backend closes the command's
	// shell session and keeps the runtime, so the caller continues with a fresh shell.
	out.Preserved = code == "command_timeout"
	if exitCode, ok := exitCodeOf(err); ok {
		out.ExitCode = &exitCode
	}
	if finished := finishedAtOf(err); !finished.IsZero() {
		out.FinishedAt = finished.UTC().Format(time.RFC3339Nano)
	}
	out.StateError = stateErrorOf(err)
	return out
}

// exitReasonOf extracts an attribution from an error that carries one. The
// concrete type lives in this package (nodeserver.go); the interface keeps the
// wire layer from depending on it and a wrapped/unrelated error answers "".
// preserveSessionLost keeps a lost session distinguishable from a transport
// fault, and reads every other failure as the transport fault it read before.
//
// The two need different answers from a caller. A node that did not answer is an
// infrastructure fault and retrying it is sensible. A session that is gone means
// the caller's own working directory, exported variables, or whole sandbox went
// with it, so an infrastructure retry cannot bring it back and the caller has to
// decide what to do. Only the second carries an attribution, so this passes that
// one through untouched and covers everything else.
// preserveSessionLost keeps a typed classification intact across the wire.
//
// A lost session and a missing path are different answers: the first replaces
// the sandbox, the second leaves it usable. Anything that carries its own wire
// code is passed through untouched; a bare ErrSessionLost is promoted to the
// lost-session status, and everything else is the transport fault it reads as.
func preserveSessionLost(err error) error {
	if _, ok := wireCodeOf(err); ok {
		return err
	}
	switch {
	case status.Code(err) == codes.NotFound:
		// Already carries the code and the attribution the wire layer reports.
		return err
	case errors.Is(err, backend.ErrSessionLost):
		return status.Error(codes.NotFound, err.Error())
	case status.Code(err) != codes.Unknown:
		// A failure that already carries a classification keeps it. The wire layer maps
		// a gRPC code onto the caller's vocabulary; rewriting it to Unavailable here
		// would turn a command that hit its deadline into a transport fault.
		return err
	default:
		return status.Error(codes.Unavailable, err.Error())
	}
}

// wireCodeOf reports the caller-facing classification an error carries, if any.
// It is checked before the gRPC status so an error's own semantics survive
// every hop, including hops that only preserve a status code.
func wireCodeOf(err error) (string, bool) {
	var coded interface{ WireCode() string }
	if errors.As(err, &coded) {
		if code := coded.WireCode(); code != "" {
			return code, true
		}
	}
	return "", false
}

func exitReasonOf(err error) string {
	var attributed interface{ ExitReason() string }
	if errors.As(err, &attributed) {
		return attributed.ExitReason()
	}
	return ""
}

// exitCodeOf extracts the runtime's exit code when the error carries one. A
// separate accessor from ExitReason keeps "unknown" distinct from "exited 0".
func exitCodeOf(err error) (int, bool) {
	var attributed interface{ ExitCode() (int, bool) }
	if errors.As(err, &attributed) {
		return attributed.ExitCode()
	}
	return 0, false
}

// finishedAtOf extracts when the runtime observed the stop, if it did.
func finishedAtOf(err error) time.Time {
	var attributed interface{ FinishedAt() time.Time }
	if errors.As(err, &attributed) {
		return attributed.FinishedAt()
	}
	return time.Time{}
}

// stateErrorOf extracts the runtime evidence or inspect failure behind a loss.
func stateErrorOf(err error) string {
	var attributed interface{ StateError() string }
	if errors.As(err, &attributed) {
		return attributed.StateError()
	}
	return ""
}

func (j *JSONListener) dispatch(ctx context.Context, req request) (any, error) {
	switch req.Method {
	case "create":
		return j.create(ctx, req.Payload)
	case "create_group":
		return j.createGroup(ctx, req.Payload)
	case "release":
		return j.release(ctx, req.Payload)
	case "status":
		return j.status(ctx, req.Payload)
	case "provision_status":
		return j.provisionStatus(req.Payload)
	case "pause":
		return j.pause(ctx, req.Payload)
	case "resume":
		return j.resume(ctx, req.Payload)
	case "snapshot":
		return j.snapshot(ctx, req.Payload)
	case "delete_snapshot":
		return j.deleteSnapshot(ctx, req.Payload)
	case "exec":
		return j.exec(ctx, req.Payload)
	case "read_bytes":
		return j.readBytes(ctx, req.Payload)
	case "write_bytes":
		return j.writeBytes(ctx, req.Payload)
	case "touch":
		return j.touch(ctx, req.Payload)
	case "renew_owner":
		return j.renewOwner(ctx, req.Payload)
	case "fleet":
		return j.fleet(ctx)
	case "quota":
		return j.quota(ctx)
	case "diagnostics":
		return j.diagnostics(ctx)
	case "contract":
		return j.contract(ctx)
	default:
		return nil, status.Errorf(codes.Unimplemented, "the sandbox service has no method %q", req.Method)
	}
}

type specEnvelope struct {
	Spec           jsonSpec `json:"spec"`
	CallbackTarget string   `json:"callback_target"`
}

type jsonSpec struct {
	Source struct {
		Kind      string `json:"kind"`
		Reference string `json:"reference"`
	} `json:"source"`
	Resources struct {
		CPUCount *float64 `json:"cpu_count"`
		MemoryMB *int64   `json:"memory_mb"`
		DiskMB   *int64   `json:"disk_mb"`
		GPUCount *int32   `json:"gpu_count"`
	} `json:"resources"`
	ResourceClass       string                       `json:"resource_class"`
	WorkflowID          string                       `json:"workflow_id"`
	IdempotencyKey      string                       `json:"idempotency_key"`
	Env                 map[string]string            `json:"env"`
	Metadata            map[string]string            `json:"metadata"`
	Workdir             string                       `json:"workdir"`
	ExecMode            string                       `json:"exec_mode"`
	RequiredFeatures    []string                     `json:"required_features"`
	RequiredResumeLevel string                       `json:"required_resume_level"`
	Backend             string                       `json:"backend"`
	RequiredNodeLabel   string                       `json:"required_node_label"`
	ForbiddenNodeLabels []string                     `json:"forbidden_node_labels"`
	LifetimeS           int64                        `json:"lifetime_s"`
	BackendOptions      map[string]map[string]string `json:"backend_options"`
	Mounts              []jsonMount                  `json:"mounts"`
}

// jsonMount is one host path a sandbox must see, on the node-plane wire.
type jsonMount struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"read_only"`
}

func (s jsonSpec) toProto() *v1.SandboxSpec {
	out := &v1.SandboxSpec{
		Source:              &v1.Source{Kind: sourceKindValue(s.Source.Kind), Reference: s.Source.Reference},
		ResourceClass:       s.ResourceClass,
		WorkflowId:          s.WorkflowID,
		IdempotencyKey:      s.IdempotencyKey,
		Env:                 s.Env,
		Metadata:            s.Metadata,
		Workdir:             s.Workdir,
		ExecMode:            execModeValue(s.ExecMode),
		RequiredResumeLevel: resumeLevelValue(s.RequiredResumeLevel),
		Backend:             s.Backend,
		RequiredNodeLabel:   s.RequiredNodeLabel,
		ForbiddenNodeLabels: s.ForbiddenNodeLabels,
		LifetimeS:           s.LifetimeS,
		Resources: &v1.Resources{
			CpuCount: s.Resources.CPUCount,
			MemoryMb: s.Resources.MemoryMB,
			DiskMb:   s.Resources.DiskMB,
			GpuCount: s.Resources.GPUCount,
		},
	}
	for _, feature := range s.RequiredFeatures {
		out.RequiredFeatures = append(out.RequiredFeatures, featureValue(feature))
	}
	for _, mount := range s.Mounts {
		out.Mounts = append(out.Mounts, &v1.Mount{
			Source: mount.Source, Target: mount.Target, ReadOnly: mount.ReadOnly,
		})
	}
	if len(s.BackendOptions) > 0 {
		out.BackendOptions = map[string]*v1.BackendOptions{}
		for name, values := range s.BackendOptions {
			out.BackendOptions[name] = &v1.BackendOptions{Values: values}
		}
	}
	return out
}

func (j *JSONListener) create(ctx context.Context, raw json.RawMessage) (any, error) {
	var env specEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	created, err := j.control.Create(ctx, &v1.CreateRequest{Spec: env.Spec.toProto(), CallbackTarget: env.CallbackTarget})
	if err != nil {
		return nil, err
	}
	return j.createdJSON(created), nil
}

func (j *JSONListener) createGroup(ctx context.Context, raw json.RawMessage) (any, error) {
	var env struct {
		Specs          []jsonSpec `json:"specs"`
		CallbackTarget string     `json:"callback_target"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	req := &v1.CreateGroupRequest{CallbackTarget: env.CallbackTarget}
	for _, spec := range env.Specs {
		req.Specs = append(req.Specs, spec.toProto())
	}
	group, err := j.control.CreateGroup(ctx, req)
	if err != nil {
		return nil, err
	}
	members := make([]any, 0, len(group.GetMembers()))
	for _, member := range group.GetMembers() {
		members = append(members, j.createdJSON(member))
	}
	return map[string]any{"members": members}, nil
}

// createdJSON renders a create reply, resolving the data-plane address this
// deployment's topology implies.
func (j *JSONListener) createdJSON(created *v1.CreateResponse) map[string]any {
	out := createdJSON(created)
	// A sandbox with no agent of its own is driven by the node that owns it, so
	// the node's address is what the SDK needs to keep its commands off the
	// control plane. Resolved here rather than carried up from the backend,
	// because the address belongs to this deployment's topology and the backend
	// only knows that it has no agent to report.
	if j.dataPlane != nil {
		if agent, ok := out["agent"].(map[string]any); ok {
			address := j.dataPlane(created.GetHandle().GetNodeId())
			// The node plane is resolved for every sandbox, not only the
			// agentless ones: an agent-backed sandbox uses it to report activity
			// its commands never carry to the node, and an agentless one also
			// uses it as its exec driver.
			agent["node_plane"] = address
			if created.GetAgent().GetAddress() == "" {
				agent["data_plane"] = address
			}
		}
	}
	return out
}

// createdJSON renders a create reply with no topology applied. The node plane
// uses this for its own replies, where the caller is the control plane rather
// than the SDK and no data-plane address is wanted.
func createdJSON(created *v1.CreateResponse) map[string]any {
	features := make([]string, 0, len(created.GetCapabilities().GetFeatures()))
	for _, feature := range created.GetCapabilities().GetFeatures() {
		features = append(features, featureName(feature))
	}
	modes := make([]string, 0, len(created.GetCapabilities().GetPauseModes()))
	for _, mode := range created.GetCapabilities().GetPauseModes() {
		modes = append(modes, pauseModeName(mode))
	}
	return map[string]any{
		"handle": map[string]any{
			"backend":    created.GetHandle().GetBackend(),
			"sandbox_id": created.GetHandle().GetSandboxId(),
			"node_id":    created.GetHandle().GetNodeId(),
		},
		"capabilities": map[string]any{
			"features":     features,
			"resume_level": resumeLevelName(created.GetCapabilities().GetResumeLevel()),
			"pause_modes":  modes,
		},
		"agent": map[string]any{
			"address":             created.GetAgent().GetAddress(),
			"data_plane":          "",
			"node_plane":          "",
			"headers":             created.GetAgent().GetHeaders(),
			"callback_host_alias": created.GetAgent().GetCallbackHostAlias(),
			"callback_port":       created.GetAgent().GetCallbackPort(),
		},
		"warm_start": created.GetWarmStart(),
	}
}

type handlePayload struct {
	Backend   string `json:"backend"`
	SandboxID string `json:"sandbox_id"`
	NodeID    string `json:"node_id"`
}

func (h handlePayload) proto() *v1.SandboxHandle {
	return &v1.SandboxHandle{Backend: h.Backend, SandboxId: h.SandboxID, NodeId: h.NodeID}
}

func (h handlePayload) handle() backend.Handle {
	return backend.Handle{Backend: h.Backend, SandboxID: h.SandboxID, NodeID: h.NodeID}
}

func (j *JSONListener) release(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Release(ctx, payload.proto()); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) status(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	reply, err := j.control.Status(ctx, payload.proto())
	if err != nil {
		return nil, err
	}
	return map[string]any{"status": statusName(reply.GetStatus())}, nil
}

// provisionStatus reports a create in flight under its ticket, so a caller with no
// response yet can tell "still provisioning" from "the request was lost". An
// unknown ticket is reported as such rather than as a failure, because a create
// that has not been seen here is not one that failed.
func (j *JSONListener) provisionStatus(raw json.RawMessage) (any, error) {
	var payload struct {
		Ticket string `json:"ticket"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	state, known := j.control.ProvisionState(payload.Ticket)
	if !known {
		return map[string]any{"state": "unknown"}, nil
	}
	return map[string]any{
		"state":            state.State,
		"elapsed_s":        state.Elapsed.Seconds(),
		"since_progress_s": state.SinceProgress.Seconds(),
	}, nil
}

func (j *JSONListener) pause(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Pause(ctx, &v1.PauseRequest{
		Handle: payload.proto(), Mode: pauseModeValue(payload.Mode),
	}); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) resume(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Resume(ctx, payload.proto()); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) snapshot(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	ref, err := j.control.Snapshot(ctx, &v1.SnapshotRequest{
		Handle: payload.proto(), Kind: snapshotKindValue(payload.Kind),
	})
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"backend":      ref.GetBackend(),
		"snapshot_id":  ref.GetSnapshotId(),
		"kind":         snapshotKindName(ref.GetKind()),
		"resume_level": resumeLevelName(ref.GetResumeLevel()),
	}, nil
}

func (j *JSONListener) deleteSnapshot(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		Backend    string `json:"backend"`
		SnapshotID string `json:"snapshot_id"`
		Kind       string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.DeleteSnapshot(ctx, &v1.SnapshotRef{
		Backend: payload.Backend, SnapshotId: payload.SnapshotID, Kind: snapshotKindValue(payload.Kind),
	}); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

// exec runs one command. In a fleet deployment the control plane forwards to
// the node that holds the sandbox via NodeClient.ExecOn, so a control-only
// process can still serve exec without being co-located with the sandbox.
// In a single-process deployment ExecOn calls the local node directly.
func (j *JSONListener) exec(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented,
			"no node client is configured; this process cannot route commands")
	}
	var payload struct {
		handlePayload
		Command         string            `json:"command"`
		Cwd             string            `json:"cwd"`
		Env             map[string]string `json:"env"`
		TimeoutS        *float64          `json:"timeout_s"`
		SilenceTimeoutS *float64          `json:"silence_timeout_s"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	opts := backend.ExecOptions{
		Timeout:        durationOf(payload.TimeoutS),
		SilenceTimeout: durationOf(payload.SilenceTimeoutS),
	}
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}
	result, err := j.control.nodes.ExecOn(ctx, payload.handle(), payload.Command, payload.Cwd, payload.Env, opts)
	if err != nil {
		return nil, preserveSessionLost(err)
	}
	return map[string]any{
		"exit_code": result.ExitCode,
		"stdout":    result.Stdout,
		"stderr":    result.Stderr,
		"truncated": result.Truncated,
	}, nil
}

// touch refreshes a held sandbox's activity.
//
// A node-plane call, like exec: the SDK reports activity for a sandbox whose
// commands bypass the node, and the node that owns the sandbox is the one with
// the lifecycle to update. A control-only process holds no sandboxes, so it says
// so rather than accepting the call silently.
func (j *JSONListener) touch(_ context.Context, raw json.RawMessage) (any, error) {
	if j.node == nil {
		return nil, status.Error(codes.Unimplemented,
			"no node is configured; this process holds no sandboxes")
	}
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	j.node.Touch(payload.handle())
	return map[string]any{"ok": true}, nil
}

// renewOwner keeps one caller's ownership alive across the fleet.
//
// The SDK calls it on a timer: the control plane refreshes the caller's quota and
// forwards the renewal to every live node, so a node reclaims that caller's
// sandboxes only once it has really stopped renewing. A failed renewal is retried
// by the next heartbeat, so a node that is briefly unreachable is not an error.
func (j *JSONListener) renewOwner(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control == nil {
		return nil, status.Error(codes.Unimplemented,
			"no control plane is configured; this process cannot renew an owner")
	}
	var payload struct {
		Owner string `json:"owner"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	j.control.RenewOwner(ctx, payload.Owner)
	return map[string]any{"ok": true}, nil
}

func (j *JSONListener) readBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented, "no node client is configured; this process cannot route file reads")
	}
	var payload struct {
		handlePayload
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	data, err := j.control.nodes.ReadBytesOn(ctx, payload.handle(), payload.Path)
	if err != nil {
		return nil, err
	}
	return map[string]any{"data": data}, nil
}

func (j *JSONListener) writeBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented, "no node client is configured; this process cannot route file writes")
	}
	var payload struct {
		handlePayload
		Path string `json:"path"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if err := j.control.nodes.WriteBytesOn(ctx, payload.handle(), payload.Path, payload.Data); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

// compactBase64 strips the line breaks `base64` emits, which a decoder on the
// other side would otherwise have to tolerate.
func compactBase64(raw string) string {
	out := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		if raw[i] != '\n' && raw[i] != '\r' {
			out = append(out, raw[i])
		}
	}
	// Validate here rather than letting the caller fail on a decode: a command
	// that printed something other than base64 is a node fault, not a bad file.
	if _, err := base64.StdEncoding.DecodeString(string(out)); err != nil {
		return ""
	}
	return string(out)
}

// headroomMap and classHeadroomMap render headroom for the wire. They take the proto type
// so a field added to the report is carried without this file naming it again.

func headroomMap(h *v1.Headroom) map[string]any {
	if h == nil {
		return nil
	}
	return map[string]any{
		"memory_mb":  h.GetMemoryMb(),
		"cpu_millis": h.GetCpuMillis(),
		"gpu_count":  h.GetGpuCount(),
		"disk_mb":    h.GetDiskMb(),
	}
}

func classHeadroomMap(m map[string]*v1.Headroom) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for class, h := range m {
		out[class] = headroomMap(h)
	}
	return out
}

func (j *JSONListener) fleet(ctx context.Context) (any, error) {
	report, err := j.control.Fleet(ctx, &v1.Empty{})
	if err != nil {
		return nil, err
	}
	nodes := make([]any, 0, len(report.GetNodes()))
	for _, node := range report.GetNodes() {
		hosted := make([]string, 0, len(node.GetBackends()))
		for _, b := range node.GetBackends() {
			hosted = append(hosted, b.GetName())
		}
		// The ledger travels with the measurement, not instead of it. This handler used to
		// hand-build a map naming only the utilisation, which silently threw away the
		// envelope, each class's headroom, and everything added to the report since -- so
		// no caller could tell a node that was refusing because it was full from one that
		// was refusing because it was charging too much. Built from the report rather than
		// enumerated here, so it cannot fall behind the report again.
		nodes = append(nodes, map[string]any{
			"node_id":        node.GetNodeId(),
			"backends":       hosted,
			"live_sandboxes": node.GetLiveSandboxes(),
			"cpu_used_pct":   node.GetCpuUsedPct(),
			"mem_used_pct":   node.GetMemUsedPct(),
			"gpu_free":       node.GetGpuFree(),
			"draining":       node.GetDraining(),
			"envelope":       headroomMap(node.GetEnvelope()),
			"class_headroom": classHeadroomMap(node.GetClassHeadroom()),
			"granted":        headroomMap(node.GetGranted()),
			"observed":       headroomMap(node.GetObserved()),
			"throttled_pct":  node.GetThrottledPct(),
		})
	}
	return map[string]any{
		"nodes":              nodes,
		"drained_nodes":      report.GetDrainedNodes(),
		"reservations_open":  report.GetReservationsOpen(),
		"capacity_exhausted": report.GetCapacityExhausted(),
		"no_candidate":       report.GetNoCandidate(),
		"locality_hit_ratio": report.GetLocalityHitRatio(),
	}, nil
}

func (j *JSONListener) quota(ctx context.Context) (any, error) {
	report, err := j.control.Quota(ctx, &v1.Empty{})
	if err != nil {
		return nil, err
	}
	classes := map[string]any{}
	for name, quota := range report.GetClasses() {
		classes[name] = map[string]any{
			"guaranteed_share":   quota.GetGuaranteedShare(),
			"max_share":          quota.GetMaxShare(),
			"granted_memory_mb":  quota.GetGranted().GetMemoryMb(),
			"headroom_memory_mb": quota.GetHeadroom().GetMemoryMb(),
			"queued":             quota.GetQueued(),
		}
	}
	return map[string]any{"classes": classes}, nil
}

// diagnostics returns the control plane's own counters and every node's.
//
// It exists because the facts an operator needs when the service misbehaves --
// renewal failures, fan-out latency, stale nodes, runtime-probe outcomes -- were
// previously only log lines. A plane that reports what it schedules but not
// whether its own liveness machinery is working is a plane that has to be
// guessed about exactly when guessing is most expensive.
func (j *JSONListener) diagnostics(_ context.Context) (any, error) {
	if j.control == nil {
		return nil, status.Error(codes.Unimplemented,
			"no control plane is configured; this process has no fleet diagnostics")
	}
	return j.control.Diagnostics(), nil
}

// contract publishes the mechanism's own timing spans.
//
// The SDK used to hardcode a heartbeat interval and a timeout that had to agree
// with these values; when they drifted, the client reported "the node reclaims
// after three beats" while the node used a different clock. A client derives its
// cadence from this instead of restating it.
func (j *JSONListener) contract(_ context.Context) (any, error) {
	return map[string]any{
		"owner_heartbeat_interval_s": timing.OwnerHeartbeatInterval.Seconds(),
		"owner_silence_ttl_s":        timing.OwnerSilenceTTL.Seconds(),
		"owner_lease_s":              timing.OwnerLease.Seconds(),
		"sweep_interval_s":           timing.SweepInterval.Seconds(),
		"reservation_ttl_s":          timing.ReservationTTL.Seconds(),
		"runtime_probe_interval_s":   node.DefaultRuntimeProbeInterval.Seconds(),
	}, nil
}

// Enum values the SDK sends as names, mapped back explicitly so an unknown one
// is "unspecified" rather than a silently wrong value.

func sourceKindValue(name string) v1.Source_Kind {
	switch name {
	case "image":
		return v1.Source_IMAGE
	case "template":
		return v1.Source_TEMPLATE
	default:
		return v1.Source_KIND_UNSPECIFIED
	}
}

func execModeValue(name string) v1.ExecMode {
	switch name {
	case "persistent":
		return v1.ExecMode_PERSISTENT
	case "one_shot":
		return v1.ExecMode_ONE_SHOT
	default:
		return v1.ExecMode_EXEC_MODE_UNSPECIFIED
	}
}

func snapshotKindValue(name string) v1.SnapshotKind {
	switch name {
	case "filesystem":
		return v1.SnapshotKind_SNAPSHOT_FILESYSTEM
	case "full_state":
		return v1.SnapshotKind_SNAPSHOT_FULL_STATE
	default:
		return v1.SnapshotKind_SNAPSHOT_KIND_UNSPECIFIED
	}
}

func statusName(s v1.SandboxStatus) string {
	switch s {
	case v1.SandboxStatus_RUNNING:
		return "running"
	case v1.SandboxStatus_PAUSED:
		return "paused"
	case v1.SandboxStatus_EXITED:
		return "exited"
	case v1.SandboxStatus_TERMINATED:
		return "terminated"
	default:
		return "unknown"
	}
}

// RefreshFleet republishes this process's node into the monitor, for a
// single-process deployment where the two share memory.
func (j *JSONListener) RefreshFleet(report func(placement.NodeView)) {
	if j.node == nil {
		return
	}
	report(j.node.View())
}
