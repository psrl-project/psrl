package server

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
)

// typedBackend is a fake whose exec and file operations fail with a chosen
// sentinel, so a test can pin how the node turns a runtime fact into a caller
// classification.
type typedBackend struct {
	*fakeDirect
	execErr   error
	readErr   error
	status    string
	detail    *backend.ExitDetail
	detailErr error
}

func (b *typedBackend) Exec(
	context.Context, backend.Handle, string, string, string, map[string]string, backend.ExecOptions,
) (backend.ExecResult, error) {
	return backend.ExecResult{}, b.execErr
}

func (b *typedBackend) ReadFile(context.Context, backend.Handle, string) ([]byte, error) {
	return nil, b.readErr
}

func (b *typedBackend) WriteFile(context.Context, backend.Handle, string, []byte) error {
	return nil
}

func (b *typedBackend) Status(context.Context, backend.Handle) (string, error) {
	if b.status != "" {
		return b.status, nil
	}
	return "running", nil
}

func (b *typedBackend) ExitDetail(context.Context, backend.Handle) (backend.ExitDetail, error) {
	if b.detailErr != nil {
		return backend.ExitDetail{}, b.detailErr
	}
	if b.detail != nil {
		return *b.detail, nil
	}
	return backend.ExitDetail{}, errors.New("no exit detail configured")
}

func createOneSandbox(
	t *testing.T, hosted backend.Backend,
) (*Node, *node.Lifecycle, backend.Handle) {
	t.Helper()
	agent, _, life, spec := newNodeOverFake(t, hosted)
	created, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: reserveOne(t, agent, spec), Backend: "fake", Spec: spec,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	handle := backend.Handle{
		Backend:   created.GetHandle().GetBackend(),
		SandboxID: created.GetHandle().GetSandboxId(),
		NodeID:    "node-a",
	}
	return agent, life, handle
}

func TestALossCarriesTheRuntimeEvidenceForItsExitCode(t *testing.T) {
	finished := time.Date(2026, 10, 23, 11, 5, 23, 0, time.UTC)
	started := finished.Add(-42 * time.Minute)
	detail := backend.ExitDetail{
		Status:        "exited",
		ExitCode:      0,
		ExitCodeKnown: true,
		StartedAt:     started,
		FinishedAt:    finished,
	}
	hosted := &typedBackend{
		fakeDirect: &fakeDirect{}, execErr: backend.ErrRuntimeGone, detail: &detail,
	}
	agent, _, handle := createOneSandbox(t, hosted)

	_, err := agent.Exec(context.Background(), handle, "true", "", nil, backend.ExecOptions{})
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a stopped container must be a lost session, got %v", err)
	}
	if !gone.codeKnown || gone.exitCode != 0 {
		t.Fatalf("exit code %d (known=%v), want the daemon's 0", gone.exitCode, gone.codeKnown)
	}
	for _, want := range []string{"status=exited", "exit_code=0", "started_at=", "finished_at="} {
		if !strings.Contains(gone.stateError, want) {
			t.Fatalf("state_error %q does not contain %q", gone.stateError, want)
		}
	}
	wire := classify(err)
	if wire.StateError != gone.stateError {
		t.Fatalf("wire state_error %q, want %q", wire.StateError, gone.stateError)
	}
	if wire.ExitCode == nil || *wire.ExitCode != 0 {
		t.Fatalf("wire exit code %v, want 0", wire.ExitCode)
	}
}

func TestALostCommandSessionKeepsTheSandboxAndNamesItsCause(t *testing.T) {
	// A shell that dies mid-command is a fact about the command session, not about
	// the sandbox: the node still holds a healthy container. Telling the caller it
	// was never here sends it to replace a sandbox that is fine, and hides what
	// actually ended the session.
	cause := fmt.Errorf(
		"the persistent shell exited while this command was running: %w", backend.ErrSessionLost)
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, execErr: cause}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.Exec(context.Background(), handle, "true", "", nil, backend.ExecOptions{})
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a dead shell must be a lost session, got %v", err)
	}
	if gone.cause == "" {
		t.Fatal("the loss dropped the runtime's account of what ended the session")
	}
	if !strings.Contains(err.Error(), "its command session was lost") {
		t.Fatalf("message %q does not say the session was lost", err.Error())
	}
	if strings.Contains(err.Error(), "is not held by node") {
		t.Fatalf("message %q claims the sandbox is not held, but the node still holds it", err.Error())
	}
	if !life.Running(handle.SandboxID) {
		t.Fatal("the node released a sandbox whose runtime it has no evidence is gone")
	}
	wire := classify(err)
	if wire.Code != "session_lost" || wire.ExitReason != "" {
		t.Fatalf("wire %q reason %q, want session_lost with no exit attribution", wire.Code, wire.ExitReason)
	}
}

func TestALossWithoutInspectEvidenceNamesTheInspectFailure(t *testing.T) {
	hosted := &typedBackend{
		fakeDirect: &fakeDirect{},
		execErr:    backend.ErrRuntimeGone,
		status:     "exited",
		detailErr:  errors.New("container disappeared during inspect"),
	}
	agent, _, handle := createOneSandbox(t, hosted)

	_, err := agent.Exec(context.Background(), handle, "true", "", nil, backend.ExecOptions{})
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("want a lost session, got %v", err)
	}
	if !strings.Contains(gone.stateError, "container disappeared during inspect") {
		t.Fatalf("state_error %q does not name the inspect failure", gone.stateError)
	}
	if gone.codeKnown {
		t.Fatal("an uninspectable container must not claim a known exit code")
	}
}

func TestExecOnAStoppedContainerIsSessionLostAndReleasesTheCharge(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, execErr: backend.ErrRuntimeGone, status: "oom"}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.Exec(context.Background(), handle, "true", "", nil, backend.ExecOptions{})
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a stopped container must be reported as a lost session, got %v", err)
	}
	if gone.reason != "oom" {
		t.Fatalf("exit reason %q, want oom from the runtime's own status", gone.reason)
	}
	if !gone.codeKnown || gone.exitCode != 137 {
		t.Fatalf("exit code %d (known=%v), want 137 from the OOM attribution", gone.exitCode, gone.codeKnown)
	}
	if gone.finished.IsZero() {
		t.Fatal("a lost session must carry when the runtime observed the stop")
	}
	wire := classify(err)
	if wire.ExitCode == nil || *wire.ExitCode != 137 {
		t.Fatalf("wire exit code %v, want 137", wire.ExitCode)
	}
	if wire.FinishedAt == "" {
		t.Fatal("wire attribution must carry finished_at")
	}
	life.WaitCleanup()
	if len(hosted.release) != 1 {
		t.Fatalf("the backend saw %d releases, want 1", len(hosted.release))
	}
}

func TestReadOfAMissingPathIsNotALostSession(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, readErr: backend.ErrPathNotFound}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.ReadBytes(context.Background(), handle, "/root/.psrl-harness/missing")
	var missing *pathNotFoundError
	if !errors.As(err, &missing) {
		t.Fatalf("a missing file must be its own error, got %v", err)
	}
	life.WaitCleanup()
	if len(hosted.release) != 0 {
		t.Fatal("a missing file must not destroy a healthy sandbox")
	}
	if code := classify(err).Code; code != "path_not_found" {
		t.Fatalf("wire code %q, want path_not_found", code)
	}
}

func TestReadFromAGoneContainerIsALostSession(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, readErr: fmt.Errorf("archive: %w", backend.ErrRuntimeGone)}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.ReadBytes(context.Background(), handle, "/tmp/x")
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a gone container must be a lost session, got %v", err)
	}
	life.WaitCleanup()
	if len(hosted.release) != 1 {
		t.Fatalf("the backend saw %d releases, want 1", len(hosted.release))
	}
}

func TestClassifyPreservesPathNotFoundAcrossTheNodePlane(t *testing.T) {
	original := &wireError{Code: "path_not_found", Message: "reading /x: missing"}
	acrossTheHop := wireErrorToStatus(original)

	if code := classify(acrossTheHop).Code; code != "path_not_found" {
		t.Fatalf("wire code after the node hop %q, want path_not_found", code)
	}
}

func TestClassifyPreservesExitAttributionAcrossTheNodePlane(t *testing.T) {
	code := 137
	finished := time.Date(2026, 10, 22, 2, 13, 7, 0, time.UTC)
	original := &wireError{
		Code:       "session_lost",
		Message:    `sandbox "sb-1" is not held by node "node-a"`,
		ExitReason: "oom",
		ExitCode:   &code,
		FinishedAt: finished.Format(time.RFC3339Nano),
	}
	acrossTheHop := wireErrorToStatus(original)
	out := classify(acrossTheHop)

	if out.ExitReason != "oom" {
		t.Fatalf("exit reason after the node hop %q, want oom", out.ExitReason)
	}
	if out.ExitCode == nil || *out.ExitCode != 137 {
		t.Fatalf("exit code after the node hop %v, want 137", out.ExitCode)
	}
	if out.FinishedAt != finished.Format(time.RFC3339Nano) {
		t.Fatalf("finished_at after the node hop %q, want %q", out.FinishedAt, finished.Format(time.RFC3339Nano))
	}
}

func TestPreserveSessionLostStillMapsTheBareSentinel(t *testing.T) {
	mapped := preserveSessionLost(fmt.Errorf("shell exited: %w", backend.ErrSessionLost))
	if status.Code(mapped) != codes.NotFound {
		t.Fatalf("a bare ErrSessionLost must map to NotFound, got %v", mapped)
	}
}
