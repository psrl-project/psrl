# Findings: the sandbox cgroup domain and the warm pool

Results for the two changes in `internal/cgroup`, `internal/pressure` and the docker
backend's warm pool. Each number names the configuration it was measured on; the scripts
that produce them are listed with it. Negative results and the limits of each claim are
at the end, because a claim without its scope is how a benchmark becomes folklore.

## Configuration measured

Two-node fleet, cgroup v1, Docker with the `cgroupfs` driver, both hosts 2319618 MB and
384 cores. Nodes declare 1100000 MB (the one that also runs the trainer) and 1400000 MB
(the sandbox-only one), `overcommit` 1.5, `local_mem_ceiling` 0.9, `utilization_target`
0.75, `cgroup_parent: psrl-sandboxes`, `sample_interval_s: 5`.

## 1. The measured domain is the sandbox subtree, not the pod

`internal/cgroup/live_test.go`, plus a fleet report before and after the change.

| Quantity | Before | After |
| --- | --- | --- |
| node reported `mem_used_pct` (the node that shares its cgroup with the trainer) | **48.8%** | **0.00%** |
| what that fraction describes | the pod: 1100 GiB of 2252 GiB, of which containers were 109 GiB | the subtree: 0 of 1650000 MiB |
| subtree limit resolved on that node | — | 1650000 MiB (requested 1100000 x 1.5) |
| subtree limit resolved on the sandbox-only node | — | 2041953 MiB, from a request of 2100000: capped by the ancestor, whose 2252800 MiB limit held 98206 MiB of unreclaimable memory plus a 5% safety share |

Exclusivity, measured by charging memory inside and outside the subtree:

| Charge | Subtree movement |
| --- | --- |
| 1 GiB container created with the daemon's default parent | 0 MiB (below the 512 MiB threshold the test sets) |
| 1 GiB container created with `--cgroup-parent psrl-sandboxes` | 1030 MiB |
| 4 GiB container created with `--cgroup-parent psrl-sandboxes` | 4108 MiB |

`Setup` also had to learn two facts the hard way, both caught by the live tests:

- `memory.memsw.limit_in_bytes` must be **raised before** `memory.limit_in_bytes` and
  lowered after it. Writing memsw down first returns `EINVAL`, because a fresh cgroup
  is unlimited in both and the kernel requires `memory <= memsw` at every step.
- Hierarchical accounting (`memory.use_hierarchy`) is verified rather than assumed. With
  it off a subtree reports only its own processes, so every ceiling derived from it is
  unreachable.

## 2. A warm claim costs milliseconds; a cold create costs hundreds

`tests/sandbox/bench_warm_pool.py`, against a node with `warm_pool` of 8 entries built
from `alpine:latest` (512 MB, 1 CPU, class `rollout`, one-shot, 7200 s lifetime).

Paired, alternating order, eight of each, one at a time:

| Kind | p50 | min | max |
| --- | --- | --- | --- |
| claimed from the pool | **0.004 s** | 0.004 s | 0.006 s |
| cold create + start | 0.274 s | 0.253 s | 0.520 s |

Sixteen concurrent, pool full (so eight of the sixteen claim) versus sixteen concurrent
that the pool cannot serve:

| Batch | p50 | p95 | min |
| --- | --- | --- | --- |
| 8 warm + 8 cold | 0.471 s | 0.894 s | 0.006 s |
| 16 cold | 0.831 s | 1.337 s | 0.588 s |

The lifetime rule end to end, against a template that states 7200 s: a request for
3600 s was served from the pool in 0.005 s and took an entry with it, and a request for
9000 s cold-created in 0.269 s because the entry could not outlive it.

The batch improves by less than a single claim does, and that is the predicted shape: the
pool covers `size/batch` of a burst, so a depth of 8 halves a batch of 16. The earlier
create-contention work measured cold creates at 3.3 s with 8 in flight and 10.7 s with 64,
which is where a claim's 4 ms matters most -- this image is small enough that a cold
create on an idle daemon is only ~0.3 s, so these numbers understate the pool's value at
the workload's real concurrency rather than overstating it.

The measurements also found three defects that no unit test covered:

1. The gate's pressure sample is only refreshed by an admit or a report, so a node's very
   first pool refill saw an *unusable* sample and did nothing for one refill interval.
2. `Config.WorkingDir` in an inspect reports the *resolved* directory (the image's `/`),
   not the empty request, so verification rejected every entry whose caller stated no
   workdir -- and destroyed it. Claims looked like 1.1 s creates because they were
   destroys plus cold creates. This is why the "warm" batch was slower than the cold one
   in the first measurement.
3. A pool that cannot build its entries was silent: it simply never served a claim. It
   now warns with the create error.
4. A template that states no lifetime matched nothing. The node resolves an unspecified
   request lifetime to `node.DefaultLifetime` before the backend sees the spec, so the
   request carried 4 h while the template carried 0. Matching treats the two as unequal
   by design, because a zero-lifetime entry would replace a stated default with no
   backstop. The template now defaults to the same lifetime the node applies, so the
   longest-lifetime rule works without every deployment restating it.

## 3. Elastic admission: the node prices a sandbox, and the fleet charges that price

Implemented: a class's per-sandbox charge is estimated from the trimmed maximum of what its
sandboxes are resident (fast grow, slow shrink, clamped to the largest declared maximum); the
node's ledger is bounded by `max(reservations, measured usage)`; a sandbox holding reclaimable
memory while the subtree is at `shrink_above` has its kernel limit lowered to `resident +
margin` (never while a command is in flight) and restored at `restore_below`; and the node
**publishes** the resulting per-class price on its node-plane report, which the control charges
for both the fleet quota and placement's promises.

Measured on the deployment, same binary, same 34 requests of a 128 GiB spec:

| Admission | Admitted | Refused with |
| --- | --- | --- |
| declared maximum (elastic off) | **27 of 34** | "every capable sandbox node is full" |
| published reserve (elastic on) | **34 of 34**, in 2.7 s | — |

27 x 128 GiB is 3.46 TB against the nodes' 1.65 TB + 2.04 TB overcommitted envelopes, which is
where the static number comes from. With the reserve published the fleet and placement charge
512 MiB per sandbox instead, and the batch fits.

The release mechanism, measured against the kernel: a sandbox holding 3.08 GiB (3.07 GiB of it
file cache) kept its process and dropped to 263 MiB when its limit was lowered to 265 MiB -- 2.8
GiB reclaimed -- answered `docker exec` afterwards, and restored to 4 GiB cleanly.

Two things had to change before the pricing was worth anything, and the second is the one the
design does not name. The control had to charge the published price for both the fleet quota and
placement's short-term promises: charging the request's maximum in either place refused the
batch the node was willing to hold. The first measurement of the change -- node ledger repriced,
control unchanged -- admitted exactly as many sandboxes as before, and raising the fleet total
did not move it either, which is how placement's accounting was identified.

### 3.1 CPU is the dimension a burst is bounded by, and pricing it moves the fleet 4x

The memory reserve above changed nothing end to end on the mini_swe step (§3.2). The reason
is the other dimension: every sandbox declares `cpu_count=2` because one tool call may want
two cores, and between calls it uses a fraction of one.

Measured with a burst of creates of one 30 GiB / 2-core spec, same binary both times:

| Charge | Admitted | Time |
| --- | --- | --- |
| declared cores (elastic off) | **180 of 200** | 60.1 s (the rest timed out queued) |
| published CPU reserve (elastic on) | **700 of 700** | 23.3 s |

180 is exactly `fleet.cpu_millis / 2000`: the fleet sells 360 cores and hands out two per
sandbox. At the measured reserve of 250 millicores a sandbox costs an eighth of a core, and
the ledger stops being the thing that decides how large a burst runs.

Both nodes publish the CPU reserve on their report (`class_reserve_cpu_millis`), and the
control states what it charges:

```
level=INFO msg="pricing a sandbox class from the nodes' published reserve"
    class=rollout dimension=cpu_millis requested=2000 charged=250
```

### 3.2 One step of the real rollout, elastic off against elastic on

Same recipe, same tasks, one training step, both nodes restarted between arms:

| Metric | off | on |
| --- | --- | --- |
| `timing_s/step` | 1185.9 s | 1059.7 s |
| `timing_s/wait_for_gen` | 339.8 s | 263.2 s |
| `sandbox/admission/waits_total` | 279 | 279 |
| `admission/wait_s_mean_worst_worker` | 57.1 s | 59.4 s |
| `sandbox/usage/peak_sessions_total` | 168 | 168 |
| OOM kills (both nodes) | 0 | 0 |

The fleet admits 700 sandboxes where it admitted 180, and the step's concurrency does not
move. **The fourth ledger is on the PSRL side.** `SandboxCapacityCoordinator`
(`psrl/sandbox/capacity.py`) keeps its own mirror of the node envelope and charges each
request its declared vector against it, so it admits `1,802,240 MiB / 30,720 MiB = 58`
sandboxes per node -- which is the 58 `peak_active_sessions` a worker reports and, across
two nodes plus graders, exactly the 168 every arm reports. It never sees the published
reserve.

`wait_for_gen` moved by 77 s, and the same metric has ranged over 100 s between runs of the
same arm, so that is not a result. What is a result is that the number every arm agrees on
is 168, and it is a PSRL-side number.

### 3.3 Routing the rollout through the service, end to end

The step's sandboxes were being created by the **in-process docker backend**, not by the
service: the harness loop names its backend from `sandbox_config.backend`, which fell
through to the manager's `default_backend` of `docker`. That put a fourth ledger in the
path -- `SandboxCapacityCoordinator` in `psrl/sandbox/capacity.py`, which mirrors the node
envelope locally and charges each sandbox its declared 30 GiB, so it admits
`1,802,240 MiB / 30,720 MiB = 58` per node. That 58 is the `peak_active_sessions` a worker
reports, and two nodes plus graders is the 168 every arm agreed on.

Routing the rollout to the service needed host mounts, which the service advertised without
implementing: the docker backend listed `host_mount` and had no mount plumbing at all. It
now binds them, behind an allowlist (`backends[].mount_allowlist`), and advertises the
feature only when the allowlist is non-empty.

Same recipe, one training step, one binary, both nodes restarted between arms:

| Admission | `waits_total` | `wait_s_mean_worst_worker` | `peak_sessions` | OOM |
| --- | --- | --- | --- | --- |
| docker backend, elastic off | 279 | 57.1 s | 168 | 0 |
| docker backend, elastic on (service not in path) | 279 | 59.4 s | 168 | 0 |
| rollout on the service, elastic on | **139** | **9.0 s** | **110** | 0 |

Six times less queueing per sandbox. `timing_s/step` did not move (1186 / 1060 / 1151 s
across the three), because the step's critical path is twenty turns of model latency rather
than sandbox admission: the queue was costing latency per sandbox, not throughput.

That arm was still half on the old path: **node 97 carries its own, stale checkout of
`/home/psrl`** (HEAD from 10-07, with its own local modifications), so the worker placed on
97 ran the old PSRL -- no `HOST_MOUNT` on its p3b client, no `backend: p3b` in its harness
config -- and created its sandboxes in the in-process docker backend. Syncing the changed
files to 97 and configuring p3b alone gives the full result:

| Arm | `waits_total` | `wait_s_mean_worst_worker` | sandboxes live at peak |
| --- | --- | --- | --- |
| docker backend in path | 279 | 57.1 s | ~7-15 per node |
| rollout on the service, one stale node | 139 | 9.0 s | ~15 fleet-wide |
| rollout on the service, both nodes current | **0** | **0 s** | **128 per node (256 fleet-wide)** |

`timing_s/step` still did not move (1181 s). The concurrency that was available all along
is now used -- 256 sandboxes where the ledger used to admit 58 per node -- and the step is
still bounded by twenty turns of model latency, not by admission.

The CPU reserve is doing what it was built to do: the grader class, measured while its
containers were running, now prices at 1386 and 890 millicores on the two nodes (up from the
250 floor), while rollout sits at 250-266 -- its sandboxes are idle between turns. The
grader costs what it uses and the rollout no longer pays for cores it never touches.

### 3.4 The gaps that only appear once density is real

The 256-sandbox step exposed three defects that the 58-per-node ceiling had been hiding.
All three are now fixed; all three were found by reading what the machine said rather than
what the ledger assumed.

**a. A class guarantee at the node was never exercised.** `Node.Admit` returned a refusal
without calling `Admission.Enqueue`, so `queued[class]` stayed zero and
`reservedForOthersLocked` never held a share back. The first wave took the whole envelope
and the second wave -- grading -- could not get in. It surfaced in a production log as:

```
Grader sandbox was never admitted for task '...mypy-5617'; keeping the trajectory as ungraded.
```

which is §1.3's failure mode exactly: an infrastructure fault becoming training signal,
because an ungraded trajectory is either dropped or scored as if it had been measured.
Admission now records refused demand (`Enqueue` on refusal, `Dequeue` when a class is
served) with a TTL (`DemandTTL`, default 2 min) so a caller that went away does not hold a
share idle. `TestARefusedRequestHoldsItsClassGuarantee` asserts it, and
`TestWithoutRefusedDemandTheFirstWaveTakesEverything` is the counterfactual that keeps the
assertion about the mechanism rather than about the envelope.

**b. CPU demand was invisible past the quota.** `cpuacct` usage is censored from above: a
sandbox that wants more than it was given only ever reaches its quota, so its measured rate
saturates and stops describing demand. The throttling counters are the only place the kernel
says a sandbox asked and was refused. Live, on this host:

| Reading | Value |
| --- | --- |
| one sandbox's `cpu.stat` | `nr_periods 2544`, `nr_throttled 2092` (**82%** of periods), `throttled_time` 21.4 s |
| live test, 2 threads in a 0.5-core quota | throttled 5.86 s against usage 2.11 s -- usage capped at the quota, throttling kept climbing |
| 127 sandboxes during a step | throttled fraction **0.019**, none above 80% |

So the mechanism is in place and correctly dormant at this load: a throttled sandbox's
demand is at least its quota, and that lower bound replaces the truncated measurement
(`ThrottledAbove`, default 10% of the interval). It also counts as busy, so a sandbox being
throttled is never shrunk on the grounds that no command is in flight.

**c. The metrics plane shared a concurrency budget with the data plane.**
`sandbox_metrics_snapshot` is a synchronous actor method that had to take a slot from
`max_concurrency_per_worker`; at 256 concurrent sandboxes every slot was held by trajectory
work and the per-step read timed out:

```
Could not read sandbox worker snapshots for metrics.   (GetTimeoutError)
```

The read now runs in its own Ray concurrency group. Verified against the live cluster: with
the single rollout slot held busy for 8 s, the snapshot returns in 0.00 s.

One step, before and after (both step 1, same recipe):

| | before | after |
| --- | --- | --- |
| `Grader sandbox was never admitted` | 1 | **0** |
| `Could not read sandbox worker snapshots` | 1 | **0** |
| `timing_s/step` | 1387 s | 1149 s |
| `timing_s/wait_for_gen` | 877 s | 349 s |

The two errors occurred once each in the before run (~0.4% of its samples), so zero in one
run is consistent rather than conclusive; what is conclusive is the mechanism, which the
unit tests pin. The timings are single samples of a metric that varied by more than 300 s
between identical arms earlier in this document.

### 3.5 Why 256 was not a result: pricing the future, not the present

The 256-sandbox step in §3.3 was measured with a clipped workload (`max_turns=20`,
`trajectory_timeout=300`). Run at the production budget (`max_turns=80`,
`trajectory_timeout=1800`) the same configuration failed:

```
Grader sandbox was never admitted for task '...'                    x37
Sandbox capacity was never granted for request [17920]              x32
notify_group_failed -> 16 requests of one parent retried             x2
```

The mechanism is not a bug in the measurements, it is what the measurements said:

| | at the floor | after measuring |
|---|---|---|
| rollout price | 512 MiB / 250 millicores | **15,723 MiB / 2,981 millicores** (peaked 33,276 / 3,265) |
| fleet admits (CPU, 360,000 millis) | 1440 | **120** |
| physical demand of 256 sandboxes | — | 768 cores against 360; 3.8 TB against 3.3 TB of subtree limit |

So 256 was **over-admission made possible by a price that had not seen the workload yet**, and the failures were the system discovering it. Three things were missing, all of which the design document had already named:

1. **Admission charged the present, not the future.** A sandbox's cost is what it will have used by the time anyone notices, and noticing is a sample away. Fixed: each class's observed growth rate is reserved over a horizon (`GrowthHorizonS`, default 60 s), both for the sandboxes already running (`pending_growth`) and for the one being admitted (`candidate_headroom`) -- §6.2's formula and VenusRL's `G_active + G_candidate`.
2. **The price could jump without a bound, and the fleet never caught up.** The nodes reprice their reservations as they measure; the fleet charged each grant what its class cost at admission and forgot it, so it kept granting capacity no node would honour. Fixed twice: one sample may now raise a class's estimate by at most `MaxGrowthFactor` (default 2x, still converging within a few samples), and the fleet reprices its outstanding grants on the report cadence (`Ledger.Reprice`, driven between reports rather than on the admission path).
3. **Nothing was kept for a wave that had not arrived.** Fixed: `AdmissionReserve` (default 10%) is a share of the envelope no running sandbox may spend, which is WeEnv's admission reservation.

Measured at the production budget after the three fixes (one step, run to completion):

| | the failing run | after the fixes |
| --- | --- | --- |
| `Grader sandbox was never admitted` | 37 | **0** |
| `Sandbox capacity was never granted` (rollout) | 32 | **0** |
| `Could not read sandbox worker snapshots` | 1 | **0** |
| tracebacks | 1 | **0** |
| concurrent sandboxes (peak, sampled) | 256, over-subscribed | **127** (64 + 63; the recipe asks for 8 prompts x 16 responses) |
| throttled fraction while loaded | — | 0.168, none above 80% |
| `timing_s/wait_for_gen` | 877 s | 6219 s |
| `timing_s/step` | 1387 s | 8249 s |
| `termination/finished/fraction` | 0.649 (step 1 of that run) | 0.473 |
| `max_turns_exceeded/fraction` | 0.352 | 0.527 |
| OOM kills | 0 | **3** of ~270 sandboxes (1.1%) |

Two of those numbers need their caveats stated rather than smoothed over:

- **The step got six times slower, and the two runs are not comparable.** The failing run
  was faster because it was dropping work: 32 rollout requests never got a sandbox and 37
  graders never got one, and a dropped trajectory does not take time. A controlled A/B at
  the production budget -- elastic off against elastic on with the fixes, same seed, same
  data -- is what would say whether the fixes cost throughput or only made the cost
  visible. It is not done here.
- **Three sandboxes were OOM-killed, and the controller did not cause it.** The log line
  at each event reads `shrunk=0 restored=0 throttled=0 released=0 oom_kills=1`: nothing
  was lowered, the counter is the kernel's. Those sandboxes exceeded the 30 GB the recipe
  declares, which is also the kernel hard limit they were created with. Either the
  declaration is too small for the hardest ~1% of tasks, or the harness needs to be told
  to use less. It is a workload-sizing signal the previous runs could not surface, because
  their episodes were too short to reach a large footprint.

What is solid: no request is refused for capacity any more, no group is lost, and the
concurrency the system reports is the concurrency the machine can hold -- 127 at ~3 cores
and ~15 GB each against 360 cores and 3.3 TB of subtree limit. The honest number for this
workload is the number the workload asks for, admitted without refusal, not the largest
number a price that has not measured anything will let through.

## 4. Limits of these results

- **The pool serves one image, and this workload uses one image per task.** The mini_swe
  rollout resolves a per-problem base image (`psrl/swebench-harness:<digest>` of that
  problem's image), so a pool built from a single image can only serve the tasks whose
  image it names. That is the pool's design -- a pool across images divides a fixed
  budget by the number of images and stops being a pool for any of them -- and it means
  the mechanism is right but inapplicable to this rollout until the workload shares an
  image (or the pool becomes per-image with a budget per key).
- Lifetime is matched as `request <= template`, so a template states the longest lifetime
  the deployment runs. For the codex recipe that number is the episode budget plus the
  grading budget plus the lifetime slack, i.e. `7200 + 0 + 3600 = 10800` seconds.
- **Coverage is `size/batch`.** A burst larger than the pool is mostly cold, and the
  budget is what bounds the depth (`memory_mb` of the declared envelope, not a class).
- **The domain does not increase capacity by itself.** It makes the measurement describe
  sandboxes, which is what the elastic controller (§6) needs; admission is still
  reservation-based, so concurrency is still bounded by the declared envelope.
- **Only the docker backend on this host was measured**, and only cgroup v1 with the
  cgroupfs driver. The v2 path is covered by unit tests, not by a deployment.
- **cgroup v1 has no `memory.high`.** `memory.soft_limit_in_bytes` is a hint the kernel consults
  only when the whole host is under pressure, so the release mechanism here is a hard-limit
  write: it reclaims clean pages and throttles the sandbox by making it wait for reclaim, rather
  than by applying the soft throttle `memory.high` gives on v2. The controller's policy is
  written against the limit for that reason; a v2 host should write `memory.high` and keep the
  limit as the hard bound.
- **The release path is verified at test scale, not at workload scale.** Filling a 1.65 TB
  subtree needs a workload this deployment does not run, so the shrink is proven by the
  integration test above rather than by a training run.
- **The published reserve is charged by p3b, not yet by the caller.** Three ledgers inside
  p3b charge the reserve (fleet quota, placement, node admission); the trainer's own
  `SandboxCapacityCoordinator` still charges the declared vector, and on this workload it
  is the one that decides concurrency (§3.2). Until it charges the reserve too, an elastic
  arm is measured as a fleet that admits more sandboxes than its caller asks for.
- **The growth model assumes sandboxes grow independently, and a GRPO group's do not.** The
  reservation follows a class's observed resident memory, so a batch that grows *after* it was
  admitted is only bounded by each sandbox's own hard limit and by the node's measured ceiling
  for new work. §6.2 asks for the correlated burst to be calibrated against real failure rates;
  that calibration and an A/B over full training steps are the gate before this is trusted for a
  long run, and they are not done here. The switch is one field per node (`node.elastic.enabled`).
- The RL-level metrics the design asks for -- successful rollouts/s, complete groups/s,
  resident byte-seconds during model waits -- are not measured here. They need an A/B of
  full training steps, not these microbenchmarks.

## 5. Fleet validation: liveness, reconciliation, and error taxonomy

A control plane and three node planes (384 cores and 2.2 TB each, Docker 26,
`alpine:latest`, `max_create_concurrency=64` per node) were driven by
`tests/sandbox/fault_injection.py`, `tests/sandbox/fleet_faults.py`, and
`verify_cluster.py`. The numbers below are what a reader can act on; each is
reproduced by the command named beside it.

| Case | Measurement | Reading |
|---|---|---|
| stdout/stderr | exit 7 with `stdout="out\n"`, `stderr="err\n"` | the Engine's stream byte is honored; the old `stderr=""` was a producer-side loss |
| missing path | `SandboxPathNotFound`, later exec succeeds | a missing file no longer destroys a healthy sandbox |
| container killed | `session_lost(exit_reason=exited)` in 0.001–0.002 s, charge released | the exec path attributes the runtime's 409 instead of calling it transport |
| owner stops renewing | reclaimed at 100.1 s | 90 s silence + 15 s sweep, inside the [90, 105] s bound |
| container removed out of band | node charge released at 30.0 s, control quota at 60.1 s | one probe interval, then two inventory reports |
| node process killed | renewal batch 0.001 s (max 0.006 s), stale after 125 s, 0/9 new placements on it | the heartbeat no longer pays a dead node's timeout |
| node blackholed (accepts, never replies) | renewal batch 3.002 s, one node failure recorded, data plane `SandboxTransportError`, recovery after unblock | the per-node 3 s budget bounds a stall instead of the old 60 s serial fan-out |
| node restarted | the sandbox's container was adopted and the same handle executed again | containers outlive the node process; startup reconciliation owns them |

Held load: 128 concurrent episode workers for one hour against the three-node
fleet completed 36,945 episodes. Six failed (0.016%), all during the final
drain (a cancelled release leaves its grant to the inventory; the control's
ledger released those grants in two reports, which is the convergence path
working rather than a leak). At the end the fleet reported zero live sandboxes,
zero containers, and file-descriptor counts flat between 39 and 72 across the
hour.

Load, measured as create + four short execs + a file read + release per episode:

| Fleet | Concurrency | Episodes | Failures | p50 | p99 | Throughput |
|---|---|---|---|---|---|---|
| 1 node | 64 | 256 | 0 | 6.22 s | 10.94 s | 7.21/s |
| 3 nodes | 64 | 256 | 0 | 1.58 s | 4.65 s | 21.91/s |
| 3 nodes | 256 | 1024 | 0 | 8.16 s | 18.88 s | 14.67/s |

Three nodes at equal per-node concurrency scale almost exactly linearly (21.9/s
against 7.2/s). Pushing the fleet to 256 concurrent episodes *lowers* throughput
(14.7/s) while raising p99: each node's create gate is 64, so the extra
concurrency queues behind the gate instead of doing work. Size the caller's
in-flight episodes to the fleet's create gate, not to the number of episodes a
step needs.

## Reproducing

```
# unit tests, no privileges
go test ./internal/cgroup/ ./internal/pressure/ ./internal/backend/dockerbackend/

# the domain, against a real daemon and cgroup hierarchy
go test -tags integration ./internal/cgroup/ ./internal/pressure/

# a node with a warm_pool block configured, then
python tests/sandbox/bench_warm_pool.py <control-endpoint> --batches 16 --pool-size 8

# the single-node failure matrix and an episode-shaped burst
python tests/sandbox/fault_injection.py --endpoint <control-endpoint> \
    --node-ssh root@<node-host> --image alpine:latest --concurrency 64

# a fleet: node outage with restart, and out-of-band container removal
python tests/sandbox/fleet_faults.py --endpoint <control-endpoint> \
    --nodes 'node-a=root@<host-a>,node-b=root@<host-b>' --target node-b \
    --image alpine:latest --restart-cmd '<binary> -config <node-config> > /tmp/node.log 2>&1 & sleep 3'

# held load; one JSON sample per cadence, and a final leak check
python tests/sandbox/soak.py --endpoint <control-endpoint> \
    --nodes 'node-a=root@<host-a>,node-b=root@<host-b>' \
    --image alpine:latest --seconds 900 --concurrency 128
```

## Member refill, bounded collection, and command-start bursts

Measured on a three-node CPU fleet with cgroup v1 and Docker: each node declared
200000 MB and 64000 millicores, `overcommit` 1.0, elastic on with a 512 MB class
reserve, `command_burst_mb: 2048`, and a 200000 MB sandbox subtree.

| Scenario | Result |
| --- | --- |
| create+exec+release, 8 / 16 / 32 / 64 concurrent | 8/8, 16/16, 32/32, 64/64; at 64: 8.59/s, p50 1832 ms, p95 3555 ms |
| 120 s soak, 64 concurrent, one command per sandbox | 828 completed, 1 transient failure (0.12%), 0 renewal failures, 0 stale nodes; fleet records returned to 0 |
| command-start burst | a memory-less sandbox's live cgroup limit was 2048 MB while a command ran, from a 512 MB class-reserve start |
| collector caps (local git repository) | an untracked file above the per-file cap is skipped and named in the diagnostics; a patch above the total cap exits 3 with `patch_capped=1` and leaves the sandbox running |

The 64-level create latency is dominated by Docker container start; the fleet
spread (4/4/4 over three nodes) and the zero-leak final check are the properties
this run is testing, not single-node create speed.

Reproduce:

```bash
# member refill manager behavior and the readmit primitive
pytest tests/agent_loop/test_manager_refill_breaker.py \
    tests/agent_loop/test_manager_validation_recovery.py \
    tests/agent_loop/test_member_refill_status.py

# collector size bounds, including a real shell run against a temporary repo
pytest tests/agent_loop/test_harness.py -k collector

# fleet create+exec+release under increasing concurrency
PSRL_BENCH_ENDPOINT=<control-endpoint> python tests/sandbox/bench_p3b_concurrency.py

# held load with live inventory, renewal and leak checks
python tests/sandbox/soak.py --endpoint <control-endpoint> \
    --nodes 'node-a=root@<host-a>,node-b=root@<host-b>,node-c=root@<host-c>' \
    --image alpine:latest --seconds 120 --concurrency 64
```

## Command budgets travel with the command

A command's deadline used to live only in the caller's wait. A caller that gave up
left the process running in the sandbox's persistent shell, so the next command
queued behind it until its own budget expired. In the NeoHorse rollout run every one
of the 260 exec timeouts was the artifact collector, and admission never waited once.

The node plane now parses `timeout_s` and `silence_timeout_s`, enforces the deadline
where the command runs, and answers `command_timeout` with `preserved=true`; the SDK
waits a fixed grace past the deadline so that typed reply wins the race; and a
cancel closes the command's shell so the next command starts on a fresh one.

| Scenario | Result |
| --- | --- |
| `sleep 30` under a 2 s command budget | typed `command_timeout` at the budget, sandbox preserved |
| a command issued right after it | succeeds immediately on a fresh shell |
| concurrent creates | unique grant ids, no "quota grant already exists" |

### A wedged container takes the node out of service

Sandbox containers left by crashed runs had their `fuse-overlayfs` processes stuck in
`D` state (`inode_wait_for_write`), so the daemon never answered
`GET /containers/<id>/json`. Startup reconciliation inspects every container the node
owns before it serves, so the restart timed out inspecting one of them and the node
refused to start. `docker rm -f` on those containers times out as well, while a new
container starts normally on the same daemon.

The refusal is deliberate: a node that cannot account for a container must not admit
against memory it cannot see. Recovering such a node is a host operation. Serving in a
degraded mode, with an un-inspectable container charged from its cgroup instead of
from the daemon, is the open design question this raised.

Reproduce the deadline contract:

```bash
go test ./internal/server -run 'TestExecStopsAtTheCommandDeadline|TestGrantIDsAreUnique' -v
```

## A session's shell is not a place to redefine `exit`

The persistent shell installed an `exit` override so a command ending in `exit N`
would report N and leave the shell standing. dash (`/bin/sh` on Debian and Ubuntu)
and busybox ash (`/bin/sh` on Alpine) refuse to define a function named after a
special builtin, and a non-interactive shell answers that syntax error by exiting.
Installing the override therefore killed the session before its first command, and
the node reported the loss as if the sandbox had never been there. Neither shell
family this deployment runs accepts it; only a shell where `/bin/sh` is non-POSIX
bash does, and no image here is built that way.

The override is removed. `exit` now means what it means in a shell: the command
session ends, the container and its filesystem stay, the caller gets a lost
session, and the next command starts on a fresh shell. The node names the runtime's
cause in that message instead of claiming the sandbox is not held, which is what it
was doing while the container was still running.

The live tests reach this through the daemon, on both shell families:

```bash
go test -tags integration ./internal/backend/dockerbackend -run 'Persistent|Exit|Errexit' -v
```

## A typed command timeout is a TimeoutError

`SandboxCommandTimeout` derives from both `SandboxError` and `TimeoutError`, so the
SDK's transport caught it in its `asyncio.TimeoutError` branch and rewrote it as
"the service did not answer within Ns". The node had answered in time with a typed
command timeout; the caller abandoned work the sandbox had preserved. The typed
error is now caught first, and the transport clause is left for a reply that never
came:

```bash
pytest p3b/tests/sandbox/test_client_failures.py -q
```

## A node's container network is part of the fleet contract

Replacement sandbox nodes were Kubernetes workers whose dockerd runs with
`--bridge=none --iptables=false`. Containers created there start and run, but get
only a loopback interface: the harness inside the sandbox could not reach the
rollout gateway, so every Codex trajectory failed with "codex produced no usable
TITO trajectory" while every p3b call reported success. The harness's own stderr
tail, carried in that error, named the real fault: "Reconnecting... waiting for
network (Connection failed: error sending request)".

A dedicated bridge network plus the NAT rule Docker would otherwise install gives
sandboxes the same outbound path the old fleet had:

```bash
docker network create --driver=bridge --subnet=172.31.247.0/24 --gateway=172.31.247.1 p3b
iptables -t nat -A POSTROUTING -s 172.31.247.0/24 -j MASQUERADE
# node backend config: "network_mode": "p3b"
```

This is host state rather than repository state: it does not survive a reboot and
has to be reapplied on any node whose daemon does not manage its own networking.
A fleet swap should therefore check the sandbox's view of the network (one
container plus a connection to the gateway), not only that the node registers.

