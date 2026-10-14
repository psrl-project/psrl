# README Style Guide

Rules for every `README.md` and user-facing `.md` in this repository.
The [prose punctuation bans](coding-style.md#banned-characters-in-all-prose-) apply
here in full: no `;`, no `—`/`–`, no clause dashes, no double spaces.

`scripts/audit_prose_style.py` enforces the code-side rules and runs in pre-commit.
It does not read `.md` files, so this guide is on you. Run
`pre-commit run --all-files` before claiming a docs change is done, because a README
edit usually comes with comment edits that the linter does check.

## The one rule

> **A README tells a reader what to do and what will bite them.
> It is not the lab notebook that proves how we learned it.**

Every sentence must survive this test: *does a first-time user act differently
because of it?* If not, delete it. The evidence lives in the commit message, the
PR, or a dedicated findings file, never in the recipe.

---

## §1. Measurements ⚠️

**Do not paste forensic data into a README.** Run-specific numbers age badly, make
claims the next reader cannot reproduce, and bury the instruction.

| Keep | Drop |
|------|------|
| A number the reader must **act on**: a default, a cap, a required flag value | A number that only **justifies** a decision already made |
| Orders of magnitude that set expectations (`8 to 15 minutes per episode`) | Precision nobody can reuse (`0 of 5816 in 25 minutes`) |
| Dataset sizes the reader will compare against their own output | Counts from one historical run (`56 of 144 trials`, `145 tasks at the time`) |
| A threshold that triggers an action (`load >> core count, do not launch`) | The incident that discovered the threshold (`load average of 3616 with 1769 runnable threads`) |

```markdown
<!-- BAD: forensics. Three numbers, none actionable, and it dates the file. -->
Dangling images accumulate and wedge the daemon. `docker image prune -f` was
measured removing 0 of 5816 in 25 minutes on a degraded daemon, while batched
`docker rmi -f` cleared them in under 3.

<!-- GOOD: same warning, tells the reader what to run. -->
Dangling images accumulate over days and eventually wedge the daemon. Prefer
batched `docker rmi -f`, because `docker image prune -f` crawls once the daemon
is already degraded.
```

```markdown
<!-- BAD: a full A/B report inside a config table's footnote. -->
Without it, the cheapest way to shed the penalty is shorter turns, which spends
the turn cap faster: that loop was measured collapsing a run from 0.573 at step
11 to 0.078 at step 16.

<!-- GOOD: the mechanism, which is what generalizes. -->
Without it, a truncated episode is graded as a policy failure, and the cheapest
way to shed that penalty is to emit shorter turns, which spends the turn cap
faster still.
```

**Benchmark tables and published results are the exception.** A `Results` section
exists to report numbers, and a `FINDINGS.md` exists to record an experiment.
Those numbers must name what produced them (model, dataset, step count) and
should be reproducible from a committed script.

## §2. Length

| Document | Target | Hard ceiling |
|---|---|---|
| Hub README (entry point) | ~150 lines | 250 |
| Topic README (`prepare/`, `eval/`) | ~200 lines | 300 |

Past the ceiling, split by **task** rather than by chapter: a reader preparing
data should not scroll through training flags. Link from the hub, do not inline.

A hub README opens with what the thing is, then results or a quickstart. Move
reference detail (every flag, every stage, every failure mode) to the topic
README that owns it.

## §3. What earns a place

In rough priority:

1. **What this is** and what problem it solves, in a short paragraph
2. **Quickstart**: the shortest command sequence that produces a real result
3. **A file or directory table**, so a reader can navigate without `ls`
4. **The commands**, with the flags that matter and their defaults
5. **Gotchas**: failure modes with a *symptom* and a *fix*

Cut on sight:

- **History.** `Previously we...`, `this was reverted...`, `the old version...`.
  Nobody reads a README to learn what it used to say
- **Design-space tourism.** `We could have used X, but...`, unless the reader
  would actively try X and break something
- **Rationale longer than the instruction it defends.** One clause of "because",
  not a paragraph
- **Tutorials** on Docker, FSDP, or GRPO. Assume the reader knows the stack

## §4. Gotchas

The most abused section. Each entry is **symptom, then cause, then fix**, in one
to three lines. It is not a place to relocate the forensics §1 banned.

```markdown
<!-- GOOD -->
- **`/tmp` is node-local.** Put datasets on the shared filesystem before a
  cross-node run, or the remote node reads a stale copy.
```

If an entry has no fix, it is not a gotcha. It is a bug, so file it.

## §5. Script headers

`audit_prose_style.py` caps a comment block at 2 lines (3 for a `NOTE`), and a
`# --- Section ---` separator counts toward that. So a script header is two lines,
matching the rest of the repository:

```bash
#!/usr/bin/env bash
# Pull Docker images referenced by a PSRL dataset.
# Usage: `prefetch_images.sh SOURCE [options]`
set -euo pipefail

usage() { sed -n '2,3p' "$0"; }
```

Do not write a multi-line `Usage:` block. It trips the linter, and the flag table
belongs in the README anyway. Keep the `sed` range in `usage()` in step with the
header, or `--help` silently prints source code.

## §6. Paths and reproducibility

Everything here is published, so nothing may leak an environment:

- **No absolute paths.** Use `${PSRL_WORKSPACE}`, repo-relative paths from the
  repository root, or an obvious placeholder
- **No real hostnames or cluster IPs.** Use `192.168.1.x` in examples
- **No internal proxies, registries, or apt mirrors** as defaults. Make them an
  explicit opt-in flag and say so
- **Define placeholders in a table** near the top, once
- **Every command must run as written.** Verify before committing, including
  that referenced files exist
- **Every committed figure must be regenerable** by a committed script, and the
  README says which one

## §7. Voice

Direct and declarative. Prefer the imperative for instructions.

- State the constraint, not its biography
- Bold the thing that will bite them, not whole sentences
- One idea per bullet
- Tables for anything with more than three parallel items
