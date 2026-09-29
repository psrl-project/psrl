# SWE grading

This package turns a prepared SWE task and a test log into a reward scorecard.
The training host packages the driver once per process. The driver runs inside
an independent grader sandbox with Python 3.9 or newer and the standard library.
The task's own interpreter still runs its original evaluation script.

## Layout

| Module | Responsibility |
| --- | --- |
| `schema.py` | Validated `GradingPlan` and `GradingResult`; JSON boundary |
| `freeze.py` | Dataset preparation, SWE-smith script freezing, parser selection |
| `parsers.py` | Cached parser registry and strict explicit-name lookup |
| `scoring.py` | PSRL log validation and test-result aggregation |
| `driver.py` | Zipapp entry point with explicit input/log/output arguments |
| `payload.py` | Deterministic zip packaging and sandbox command construction |
| `runtime.py` | Sandbox file transfer, execution, timeout and scorecard validation |
| `editable.py` | Conservative optional rewriting of editable installs |
| `editable_probe.py` | Standalone probe executed by the task's Python |
| `vendor_parsers.py` | Offline parser regeneration tool |
| `_vendor/` | Bundled upstream parser implementations and minimal support code |

## Why keep `_vendor`?

SWE-bench and SWE-smith parsers encode repository-specific test-log semantics.
Replacing them with a generic pytest regex changes rewards. Installing both
upstream packages into every historical task image introduces dependency and
Python-version conflicts, and adds setup latency to each rollout.

Bundling their small standard-library parser subset keeps runtime installation
unnecessary and fixes the parser version used by an experiment. `_vendor` is a
private upstream boundary, not a directory for PSRL orchestration. Local scoring
and registry logic live outside it. Language modules are retained because the
existing prepared-task contract can name any supported upstream parser; reducing
that set would be a separate compatibility change.

`_vendor/PROVENANCE.md` records the upstream versions and the sha256 of every
file. The generated files are **committed artifacts**, not a build step: they
are the offline payload that `payload.py` packages into each grading sandbox, so
a plain checkout must contain them. Regeneration is only for an intentional
upstream version bump — never a per-training step:

```bash
pip install "swebench==4.1.0" "swesmith[generate]==0.0.9"
python -m examples.mini_swe.grading.vendor_parsers
```

The tool rejects flattened SWE-smith methods that still depend on `self`, `cls`,
or `super`. Review upstream upgrades and their dependencies before publishing a
new bundle; the shared source hash preserves existing SWE-smith parser names. Do
not hand-edit generated language parsers or `swesmith_parsers.py`. The root
`.gitattributes` marks the generated files so reviews collapse their diffs.

`parsers.py` raises an actionable error when the bundle is missing, and
`payload.grader_zip_bytes` additionally verifies every file against
`PROVENANCE.md` before building the payload, so a pruned or hand-edited checkout
fails fast instead of silently changing rewards.

## Scoring contract

Preparation writes `eval_script`, an explicit `log_parser`, `FAIL_TO_PASS`, and
`PASS_TO_PASS`. Verified uses its repository parser; SWE-Gym pins pytest;
SWE-smith freezes its profile parser. Runtime retains the existing pytest default
for old Gym rows without `log_parser`, but unknown explicit names are errors.

Test lists accept arrays or JSON arrays of nonempty strings. Duplicate IDs are
normalized, overlapping F2P/P2P sets and completely empty expectations are
rejected. `eval_type` must be `pass_and_fail` (default) or explicitly `fail_only`.

- `pass_and_fail`: PASSED/XFAIL succeed; missing, skipped, error, and unknown
  statuses fail. Expected tests never disappear from the denominator.
- `fail_only`: retains the upstream explicit-FAILED rule. It is never selected
  by a failed enum comparison or an unknown configuration value.
- Empty/unparseable parser output is an infrastructure failure in either mode.
- A nonzero test-process status alone does not decide the reward. Driver failure,
  malformed scorecards, and contradictory counts cannot produce a success.
- Runtime failures preserve the expected test counts. The configured timeout
  bounds the combined test/driver command; artifact upload is separate.

The caller owns disposable sandbox lifecycle and cleanup after a timeout.
Artifact paths are declared once in `payload.py` and passed explicitly to the
packaged driver. Each command clears previous scorecard/log artifacts. A session
must not run concurrent grading calls, since those artifact paths are shared.
The driver uses isolated Python mode to avoid importing from task `PYTHONPATH`.
This is not a security boundary against code with write access to the grader's
files; independent sandboxing and patch policies remain caller responsibilities.

## Editable-install optimization

The optimization is always on; there is no configuration toggle. The frozen eval
scripts re-run `pip install -e .` before the tests, but a task image already has
the checkout installed editable, so that command only rebuilds the same wheel
(~20s on a small repo, minutes on large ones). `grade_fresh_container` therefore
skips it unless the patch changes what the install would produce: a patch that
touches packaging metadata or dependency pins, or that adds a Python package
directory, keeps the original command. Evaluation grades through the same entry
point, so it shares the behavior.

The rewrite recognizes a standalone `python[3[.minor]] -m pip install -e .`
command with a conservative flag set. Chained commands, extras, build settings,
and dependency installation are left intact. The probe requires a PEP 660
editable installation pointing exactly at the current checkout. Legacy egg-info
or a nested checkout cannot prove that skipping installation is safe, so they
fall back to the original command. Older Python without `importlib.metadata`
also falls back. This deliberately favors scoring equivalence over hit rate.

## Verification

The focused tests exercise the actual zipapp in a separate isolated Python
process, scoring edge cases, malformed inputs, parser selection, install guards,
timeouts, scorecard validation, and deterministic packaging. They do not need
Ray, torch, Docker, or the upstream grading packages:

```bash
python -m pytest --confcutdir=tests/agent_loop tests/agent_loop/test_grading_driver.py
```

`--confcutdir` excludes the unrelated root fixture which preloads training
modules. Live task-image grading and upstream-version parity should additionally
be validated in the Linux training environment before a new benchmark run.
