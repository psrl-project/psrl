package dockerbackend

import (
	"context"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// frame builds one Docker multiplexing frame for a test.
func frame(stream byte, payload string) []byte {
	out := make([]byte, 8+len(payload))
	out[0] = stream
	binary.BigEndian.PutUint32(out[4:8], uint32(len(payload)))
	copy(out[8:], payload)
	return out
}

func TestDemultiplexStreamsKeepsStdoutAndStderrApart(t *testing.T) {
	raw := append(frame(1, "out-1\n"), frame(2, "err-1\n")...)
	raw = append(raw, frame(1, "out-2\n")...)

	stdout, stderr, truncated := demultiplexStreams(raw)

	if stdout != "out-1\nout-2\n" {
		t.Fatalf("stdout %q, want the two stdout frames in order", stdout)
	}
	if stderr != "err-1\n" {
		t.Fatalf("stderr %q, want the stderr frame", stderr)
	}
	if truncated {
		t.Fatal("a fully framed stream must not be reported truncated")
	}
}

func TestDemultiplexStreamsMarksAShortFrameTruncated(t *testing.T) {
	raw := frame(1, "complete")
	short := frame(1, "partial")
	raw = append(raw, short[:len(short)-3]...)

	stdout, _, truncated := demultiplexStreams(raw)

	if stdout != "completepart" {
		t.Fatalf("stdout %q, want the complete frame plus the bytes that did arrive", stdout)
	}
	if !truncated {
		t.Fatal("a short trailing frame must be reported as truncated output")
	}
}

func TestExecMapsAStoppedContainerToRuntimeGone(t *testing.T) {
	engine := &fakeEngine{
		execCreateStatus: 409,
		execCreateBody:   `{"message":"Container c-1 is not running"}`,
	}
	b := newFakeBackend(t, engine, time.Second, 0)
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}

	_, err := b.Exec(context.Background(), handle, "", "true", "", nil, backend.ExecOptions{})
	if !errors.Is(err, backend.ErrRuntimeGone) {
		t.Fatalf("a stopped container must be ErrRuntimeGone, got %v", err)
	}
	if errors.Is(err, backend.ErrSessionLost) {
		t.Fatal("a stopped container is not a lost shell session")
	}
}

func TestExecMapsAnOciExecFailureToRuntimeGone(t *testing.T) {
	// The Engine answers 200 and the exec exits 127, but the output is runc
	// saying it could not enter the container's namespaces. That is a dead
	// runtime, not a command that returned 127.
	engine := &fakeEngine{
		execStartOutput: frame(1, "OCI runtime exec failed: exec failed: unable to start container process: "+
			"error executing setns process: exit status 1; nsexec-1[1]: failed to open "+
			"/proc/42/ns/ipc: No such file or directory\n"),
		execInspectBody: `{"ExitCode":127,"Running":false}`,
	}
	b := newFakeBackend(t, engine, time.Second, 0)
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}

	_, err := b.Exec(context.Background(), handle, "", "true", "", nil, backend.ExecOptions{})
	if !errors.Is(err, backend.ErrRuntimeGone) {
		t.Fatalf("an OCI exec failure must be ErrRuntimeGone, got %v", err)
	}
}

func TestExecKeepsAnOrdinaryExit127AsACommandResult(t *testing.T) {
	engine := &fakeEngine{
		execStartOutput: frame(1, "wc: not found\n"),
		execInspectBody: `{"ExitCode":127,"Running":false}`,
	}
	b := newFakeBackend(t, engine, time.Second, 0)
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}

	result, err := b.Exec(context.Background(), handle, "", "wc -c", "", nil, backend.ExecOptions{})
	if err != nil {
		t.Fatalf("a missing tool is a command result, got %v", err)
	}
	if result.ExitCode != 127 || result.Stdout != "wc: not found\n" {
		t.Fatalf("result %+v, want the command's own 127 and output", result)
	}
}

func TestAContainerWithNoWritableLayerIsGone(t *testing.T) {
	err := &apiError{
		status: 500,
		body:   `{"message":"RWLayer of container c-1 is unexpectedly nil"}`,
	}
	if !isContainerGone(err) {
		t.Fatal("a nil writable layer is a removed container, not a transport fault")
	}
}

func TestExecDoesNotMapAPausedContainerToRuntimeGone(t *testing.T) {
	engine := &fakeEngine{
		execCreateStatus: 409,
		execCreateBody:   `{"message":"Container c-1 is paused, unpause the container before exec"}`,
	}
	b := newFakeBackend(t, engine, time.Second, 0)
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}

	_, err := b.Exec(context.Background(), handle, "", "true", "", nil, backend.ExecOptions{})
	if err == nil || errors.Is(err, backend.ErrRuntimeGone) {
		t.Fatalf("a paused container can be resumed, so it must not be reported gone: %v", err)
	}
}

func TestReadFileDistinguishesAMissingPathFromAMissingContainer(t *testing.T) {
	missingPath := &fakeEngine{
		archiveStatus: 404,
		archiveBody:   `{"message":"Could not find the file /tmp/x in container c-1"}`,
	}
	b := newFakeBackend(t, missingPath, time.Second, 0)
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}
	if _, err := b.ReadFile(context.Background(), handle, "/tmp/x"); !errors.Is(err, backend.ErrPathNotFound) {
		t.Fatalf("a missing file must be ErrPathNotFound, got %v", err)
	}

	missingContainer := &fakeEngine{
		archiveStatus: 404,
		archiveBody:   `{"message":"No such container: c-1"}`,
	}
	b = newFakeBackend(t, missingContainer, time.Second, 0)
	if _, err := b.ReadFile(context.Background(), handle, "/tmp/x"); !errors.Is(err, backend.ErrRuntimeGone) {
		t.Fatalf("a missing container must be ErrRuntimeGone, got %v", err)
	}
}

func TestStatusReportsOomOnlyWhenTheContainerIsNotRunning(t *testing.T) {
	handle := backend.Handle{Backend: "docker", SandboxID: "c-1"}

	dead := &fakeEngine{inspectBody: `{"State":{"Status":"exited","Running":false,"OOMKilled":true}}`}
	if state, err := newFakeBackend(t, dead, time.Second, 0).Status(context.Background(), handle); err != nil || state != "oom" {
		t.Fatalf("status %q err %v, want oom", state, err)
	}

	restarted := &fakeEngine{inspectBody: `{"State":{"Status":"running","Running":true,"OOMKilled":true}}`}
	if state, err := newFakeBackend(t, restarted, time.Second, 0).Status(context.Background(), handle); err != nil || state != "running" {
		t.Fatalf("status %q err %v, want running: a restarted container's historical OOM flag is not its state", state, err)
	}
}
