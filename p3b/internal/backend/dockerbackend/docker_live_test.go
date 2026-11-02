//go:build integration

package dockerbackend

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// These run against a real daemon, because the Engine API is the contract under
// test: a fake would only prove this package agrees with my model of Docker.
// They are skipped where no socket is present, so the suite still runs anywhere.

const testImage = "alpine:latest"

func liveBackend(t *testing.T) *Backend {
	t.Helper()
	socket := os.Getenv("DOCKER_SOCKET")
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	if _, err := os.Stat(socket); err != nil {
		t.Skipf("no docker socket at %s", socket)
	}
	b, err := New(Config{
		Socket:     socket,
		APIVersion: "v1.40",
		NodeID:     "test-node",
		// Scoped to this test run, so a sweep here can never see a production
		// sandbox and nothing this test makes is mistaken for one.
		OwnerID:        fmt.Sprintf("sandboxd-test-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second,
		PullTimeout:    3 * time.Minute,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	if err := b.Preflight(context.Background()); err != nil {
		t.Skipf("daemon not usable: %v", err)
	}
	return b
}

func spec() backend.Spec {
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: testImage},
		Resources:     backend.Resources{MemoryMB: 64, CPUCount: 0.5},
		ResourceClass: "rollout",
	}
}

func createOne(t *testing.T, b *Backend) backend.Created {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), time.Minute)
		defer releaseCancel()
		if err := b.Release(releaseCtx, created.Handle); err != nil {
			t.Errorf("cleanup release: %v", err)
		}
	})
	return created
}

func TestPreflightRejectsAMissingDaemon(t *testing.T) {
	b, err := New(Config{Socket: "/nonexistent/docker.sock", RequestTimeout: 2 * time.Second}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a node whose daemon cannot be reached")
	}
}

func TestACreatedSandboxIsRunningAndRunsACommand(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()

	status, err := b.Status(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status != "running" {
		t.Fatalf("status is %q, want running", status)
	}

	result, err := b.Exec(ctx, created.Handle, "", "echo hello-sandbox", "", nil)
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if result.ExitCode != 0 {
		t.Fatalf("exit code %d, want 0 (output %q)", result.ExitCode, result.Stdout)
	}
	if !strings.Contains(result.Stdout, "hello-sandbox") {
		t.Fatalf("output %q does not carry the command's stdout", result.Stdout)
	}
}

func TestExecReportsANonZeroExitRatherThanAnError(t *testing.T) {
	// A failing command is a result, not a transport fault: reading it as an error
	// would make a task failure indistinguishable from a broken node.
	b := liveBackend(t)
	created := createOne(t, b)

	result, err := b.Exec(context.Background(), created.Handle, "", "exit 42", "", nil)

	if err != nil {
		t.Fatalf("a non-zero exit must not be an error: %v", err)
	}
	if result.ExitCode != 42 {
		t.Fatalf("exit code %d, want 42", result.ExitCode)
	}
}

func TestExecCarriesTheWorkingDirectoryAndEnvironment(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)

	result, err := b.Exec(context.Background(), created.Handle, "", "pwd && echo $SANDBOX_MARK", "/tmp",
		map[string]string{"SANDBOX_MARK": "carried"})

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if !strings.Contains(result.Stdout, "/tmp") || !strings.Contains(result.Stdout, "carried") {
		t.Fatalf("output %q is missing the working directory or the environment", result.Stdout)
	}
}

func TestOutputIsDemultiplexedRatherThanFramed(t *testing.T) {
	// An attached exec returns a framed stream; passing the frames through would
	// put binary headers into a command's output.
	b := liveBackend(t)
	created := createOne(t, b)

	result, err := b.Exec(context.Background(), created.Handle, "", "printf 'clean'", "", nil)

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if result.Stdout != "clean" {
		t.Fatalf("output is %q, want exactly \"clean\"", result.Stdout)
	}
}

func TestAFreezeKeepsTheSandboxResidentAndResumable(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()

	if err := b.Pause(ctx, created.Handle, "freeze"); err != nil {
		t.Fatalf("pause: %v", err)
	}
	status, err := b.Status(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status while paused: %v", err)
	}
	if status != "paused" {
		t.Fatalf("status is %q, want paused", status)
	}

	if err := b.Resume(ctx, created.Handle); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if result, err := b.Exec(ctx, created.Handle, "", "true", "", nil); err != nil || result.ExitCode != 0 {
		t.Fatalf("a resumed sandbox must run commands: code=%d err=%v", result.ExitCode, err)
	}
}

func TestAHibernationIsRefusedBecauseDockerKeepsTheContainerResident(t *testing.T) {
	// Claiming a hibernation would promise that compute was released, which a
	// freeze does not do.
	b := liveBackend(t)
	created := createOne(t, b)

	if err := b.Pause(context.Background(), created.Handle, "hibernate"); err == nil {
		t.Fatal("docker offers a freeze, so a hibernation must be refused")
	}
}

func TestASnapshotCommitsTheFilesystemAndCanBeDeleted(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()
	if result, err := b.Exec(ctx, created.Handle, "", "echo marked > /tmp/mark", "", nil); err != nil || result.ExitCode != 0 {
		t.Fatalf("writing the marker: code=%d err=%v", result.ExitCode, err)
	}

	snapshotID, err := b.Snapshot(ctx, created.Handle, "filesystem")
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if snapshotID == "" {
		t.Fatal("a snapshot must return an id a restore can name")
	}

	if err := b.DeleteSnapshot(ctx, snapshotID); err != nil {
		t.Fatalf("delete snapshot: %v", err)
	}
}

func TestAFullStateSnapshotIsRefused(t *testing.T) {
	// Docker commits the writable layer and has no supported memory checkpoint, so
	// a full-state claim would let a caller read a workspace restore as proof that
	// a live process survived.
	b := liveBackend(t)
	created := createOne(t, b)

	if _, err := b.Snapshot(context.Background(), created.Handle, "full_state"); err == nil {
		t.Fatal("docker cannot capture memory, so full_state must be refused")
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	// A retried release is normal: the first may have timed out after the daemon
	// already removed the container.
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("first release: %v", err)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("a repeated release must be a no-op, got %v", err)
	}
}

func TestStatusOfAReleasedSandboxIsTerminatedRatherThanAnError(t *testing.T) {
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}

	status, err := b.Status(ctx, created.Handle)

	if err != nil {
		t.Fatalf("a gone sandbox is a state, not an error: %v", err)
	}
	if status != "terminated" {
		t.Fatalf("status is %q, want terminated", status)
	}
}

func TestAReconcileRebuildsWhatThisServiceHoldsFromTheDaemon(t *testing.T) {
	// The daemon is the source of truth: a service that trusted its own list would
	// leak every container it forgot across a restart.
	b := liveBackend(t)
	created := createOne(t, b)

	found, err := b.Reconcile(context.Background())
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	var matched bool
	for _, rec := range found {
		if rec.Handle.SandboxID != created.Handle.SandboxID {
			continue
		}
		matched = true
		if rec.Handle.NodeID != "test-node" {
			t.Errorf("reconciled handle carries node %q, want test-node", rec.Handle.NodeID)
		}
		if !rec.Running {
			t.Error("a live sandbox must reconcile as running")
		}
		if rec.Class != "rollout" {
			t.Errorf("class: got %q, want rollout", rec.Class)
		}
		// The charge has to round-trip, or a restart rebuilds a different envelope.
		if rec.Resources.MemoryMB != 64 || rec.Resources.CPUCount != 0.5 {
			t.Errorf("resources: got %+v, want 64MB and 0.5 cores", rec.Resources)
		}
	}
	if !matched {
		t.Fatalf("reconcile returned %d records, none of them the live sandbox", len(found))
	}
}

func TestAReconcileRecoversTheFactsOnlyALabelCarries(t *testing.T) {
	// How a sandbox is executed, how long it may live, how much disk it was
	// charged and who owns it cannot be read off the container's limits, so the
	// labels are the only place a restart can recover them from.
	b := liveBackend(t)
	request := spec()
	request.ExecMode = "persistent"
	request.Lifetime = 90 * time.Minute
	request.Resources.DiskMB = 512
	request.Metadata = map[string]string{backend.OwnerMetadataKey: "worker-7"}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", request, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), time.Minute)
		defer releaseCancel()
		_ = b.Release(releaseCtx, created.Handle)
	})

	found, err := b.Reconcile(context.Background())
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	for _, rec := range found {
		if rec.Handle.SandboxID != created.Handle.SandboxID {
			continue
		}
		if rec.Owner != "worker-7" {
			t.Errorf("owner: got %q, want worker-7", rec.Owner)
		}
		if rec.ExecMode != "persistent" {
			t.Errorf("exec mode: got %q, want persistent", rec.ExecMode)
		}
		if rec.Lifetime != 90*time.Minute {
			t.Errorf("lifetime: got %s, want 90m", rec.Lifetime)
		}
		if rec.Resources.DiskMB != 512 {
			t.Errorf("disk: got %dMB, want 512MB", rec.Resources.DiskMB)
		}
		if rec.CreatedAt.IsZero() {
			t.Error("created at: a reconciled sandbox needs its start time to age out on")
		}
		return
	}
	t.Fatalf("reconcile returned %d records, none of them the live sandbox", len(found))
}

func TestACreateIsFoundByItsIdempotencyKey(t *testing.T) {
	// A create whose response was lost leaves a container the caller never learned
	// the id of, so a retry has to find it rather than make a second one.
	b := liveBackend(t)
	request := spec()
	request.IdempotencyKey = "ticket-lost-response"
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", request, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), time.Minute)
		defer releaseCancel()
		_ = b.Release(releaseCtx, created.Handle)
	})

	found, ok, err := b.FindByIdempotencyKey(context.Background(), "ticket-lost-response")
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if !ok || found.Handle.SandboxID != created.Handle.SandboxID {
		t.Fatalf("the lost create must be found by its key, got %+v ok=%v", found, ok)
	}
	if _, ok, err := b.FindByIdempotencyKey(context.Background(), "no-such-ticket"); err != nil || ok {
		t.Errorf("an unknown key must miss, got ok=%v err=%v", ok, err)
	}
}

func TestTheMemoryLimitIsAppliedToTheContainer(t *testing.T) {
	// An unenforced limit is worse than none: the node's accounting would believe
	// a bound the kernel is not keeping.
	b := liveBackend(t)
	created := createOne(t, b)

	result, err := b.Exec(context.Background(), created.Handle, "",
		"cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes", "", nil)

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	// 64MB was requested; the cgroup must report that rather than the host total.
	if !strings.Contains(result.Stdout, "67108864") {
		t.Fatalf("cgroup reports %q, want the 64MB limit (67108864)", strings.TrimSpace(result.Stdout))
	}
}

func TestConcurrentCreatesAndReleasesAllSucceed(t *testing.T) {
	// Creates arrive in bursts, so the backend has to hold up under them rather
	// than only in sequence.
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Warm the image once, so the burst measures creates and not one pull.
	if err := b.ensureImage(ctx, testImage); err != nil {
		t.Fatalf("pull: %v", err)
	}

	const n = 12
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			created, err := b.Create(ctx, "test-node", spec(), "")
			if err != nil {
				errs <- fmt.Errorf("create: %w", err)
				return
			}
			if result, err := b.Exec(ctx, created.Handle, "", "true", "", nil); err != nil || result.ExitCode != 0 {
				errs <- fmt.Errorf("exec: code=%d err=%w", result.ExitCode, err)
			}
			if err := b.Release(ctx, created.Handle); err != nil {
				errs <- fmt.Errorf("release: %w", err)
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent lifecycle: %v", err)
	}

	// Nothing of this run's may be left behind.
	left, err := b.Reconcile(ctx)
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if len(left) != 0 {
		t.Fatalf("%d sandboxes survived the burst", len(left))
	}
}

// -- the persistent shell's contract -------------------------------------------
//
// A caller builds state in the session shell across turns: a working directory,
// an activated venv, exported variables. That state is only worth anything if the
// session outlives every command sent to it, so these pin the two ways a shell
// used to end early and take that state with it.

// persistentOne creates a sandbox whose commands share one shell, which is the
// mode a multi-turn harness uses.
func persistentOne(t *testing.T, b *Backend) backend.Created {
	t.Helper()
	request := spec()
	request.ExecMode = "persistent"
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", request, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), time.Minute)
		defer releaseCancel()
		if err := b.Release(releaseCtx, created.Handle); err != nil {
			t.Errorf("cleanup release: %v", err)
		}
	})
	return created
}

func execIn(t *testing.T, b *Backend, handle backend.Handle, command string) (backend.ExecResult, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	return b.Exec(ctx, handle, "persistent", command, "", nil, backend.ExecOptions{})
}

// setUpSessionState builds the state whose loss these tests are about.
func setUpSessionState(t *testing.T, b *Backend, handle backend.Handle) {
	t.Helper()
	if result, err := execIn(t, b, handle, "cd /tmp && export PSRL_MARK=kept"); err != nil || result.ExitCode != 0 {
		t.Fatalf("setting session state failed: code=%d err=%v out=%q", result.ExitCode, err, result.Stdout)
	}
}

func assertSessionSurvived(t *testing.T, b *Backend, handle backend.Handle) {
	t.Helper()
	result, err := execIn(t, b, handle, "pwd")
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("pwd after the command failed: code=%d err=%v", result.ExitCode, err)
	}
	if strings.TrimSpace(result.Stdout) != "/tmp" {
		t.Errorf("the working directory is %q, so the session was replaced", strings.TrimSpace(result.Stdout))
	}
	result, err = execIn(t, b, handle, "echo $PSRL_MARK")
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("reading the exported variable failed: code=%d err=%v", result.ExitCode, err)
	}
	if strings.TrimSpace(result.Stdout) != "kept" {
		t.Errorf("the exported variable is %q, so the session was replaced", strings.TrimSpace(result.Stdout))
	}
}

func TestAnExitCommandReportsItsOwnStatusAndTheShellSurvives(t *testing.T) {
	// The status of a command is data the caller reads, and a session that a
	// command can end is not a session: `exit 3` has to report 3 and leave the
	// shell it ran in standing.
	b := liveBackend(t)
	created := persistentOne(t, b)
	setUpSessionState(t, b, created.Handle)

	result, err := execIn(t, b, created.Handle, "exit 3")
	if err != nil {
		t.Fatalf("exit 3 was reported as a failure rather than as a status: %v", err)
	}
	if result.ExitCode != 3 {
		t.Fatalf("exit 3 reported %d, want 3", result.ExitCode)
	}
	assertSessionSurvived(t, b, created.Handle)
}

func TestACallerErrexitDoesNotReplaceTheSession(t *testing.T) {
	// A caller that runs `set -e` gets errexit in this shell, because the command
	// runs in it. Without the group in the wrapper, the next failing command ends
	// the shell before the status is written, and the caller's state is silently
	// gone -- the failure it asked about looks like an ordinary non-zero exit.
	b := liveBackend(t)
	created := persistentOne(t, b)
	setUpSessionState(t, b, created.Handle)

	if result, err := execIn(t, b, created.Handle, "set -e"); err != nil || result.ExitCode != 0 {
		t.Fatalf("set -e failed: code=%d err=%v", result.ExitCode, err)
	}
	result, err := execIn(t, b, created.Handle, "false")
	if err != nil {
		t.Fatalf("a failing command under errexit reported an error: %v", err)
	}
	if result.ExitCode != 1 {
		t.Fatalf("false under errexit reported %d, want 1", result.ExitCode)
	}
	assertSessionSurvived(t, b, created.Handle)
}

func TestADeadShellIsReportedAndItsCommandIsNotRepeated(t *testing.T) {
	// A shell can still be killed outright, and then the caller's state is gone
	// whether or not anyone likes it. What must not happen is silence: the command
	// is not re-run on a fresh shell, because nobody said it was safe to repeat,
	// and the caller is told its session is lost rather than that the node failed.
	b := liveBackend(t)
	created := persistentOne(t, b)

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if _, err := b.Exec(ctx, created.Handle, "persistent",
		"echo ran >> /tmp/psrl-attempts && kill -9 $$", "", nil); err == nil {
		t.Fatal("a command that killed its shell must be reported as a failure")
	} else if !errors.Is(err, backend.ErrSessionLost) {
		t.Fatalf("a dead shell must be reported as a lost session, got: %v", err)
	}

	// One line means one execution. A retry on a fresh shell would append twice.
	data, err := b.ReadFile(ctx, created.Handle, "/tmp/psrl-attempts")
	if err != nil {
		t.Fatalf("reading the attempt log: %v", err)
	}
	if attempts := strings.Count(string(data), "ran"); attempts != 1 {
		t.Fatalf("the command ran %d times, want exactly one: a lost session must not be retried", attempts)
	}

	// The dead session is replaced rather than left poisoned, so the next command
	// works -- in a new shell, which is what "the state is gone" means.
	if result, err := execIn(t, b, created.Handle, "echo after"); err != nil || result.ExitCode != 0 ||
		!strings.Contains(result.Stdout, "after") {
		t.Fatalf("the session was not replaced: code=%d err=%v out=%q", result.ExitCode, err, result.Stdout)
	}
}
