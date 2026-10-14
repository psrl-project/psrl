# Coding Style Guide

All new and modified code in PSRL must follow these conventions.
Sections are ordered by risk: rules that cause **silent runtime bugs** come first,
cosmetic formatting rules come last.

## Quick Decision Table

When writing a string, ask: **"Is this string a name the program looks up, or a message a human reads?"**

| Writing a… | Capitalize? | Period? | Backticks for identifiers? | `!r` for values? |
|------------|-------------|---------|---------------------------|-------------------|
| **Identifier string** (hasattr, getattr, ==, dict key, Ray method name, enum value) | No change | **NEVER** | N/A | N/A |
| **Assertion message** | Uppercase | Yes `.` | No (plain text) | Yes |
| **Log message** | Uppercase | Yes `.` / `...` / `!` | No (plain text) | Yes |
| **Comment — sentence** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Comment — fragment** | lowercase OK | No | Yes `` `id` `` | N/A |
| **Annotation marker** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Docstring prose** | Uppercase | Yes `.` | Yes `` `id` `` | N/A |
| **Docstring arg desc** | Uppercase | `.` optional | Yes for *other* identifiers | N/A |

### Banned characters in ALL prose ⚠️

In every comment, docstring, log message, and assertion message, these characters are
**forbidden anywhere**, not only at the end. Break the sentence in two instead.

| Banned | Instead write | Why it is easy to get wrong |
|--------|---------------|------------------------------|
| `;` semicolon | `.` and a new sentence, or a conjunction | The most frequent violation. `A; B` reads naturally but is banned |
| `—` em dash, `–` en dash | `.` and a new sentence, or restructure | Editors and models emit these automatically |
| `-` dash joining clauses | plain prose (`# worker-side` → `# worker side`) | Hyphenated compounds inside words are fine |
| `. ` double space | one space | Invisible in review |

```python
# BAD  — semicolon, em dash, double space
# Recovery succeeded; the slot is not wasted — keep the data.  It trains fine.

# GOOD — separate sentences, single spaces
# Recovery succeeded, so the slot is not wasted. Keep the data because it trains fine.
```

Before finishing an edit, grep your own diff: `git diff -U0 | grep -nE '^\+.*[;—–]'`
and confirm every hit is real code (a `for` loop, a shell command) rather than prose.

### Comment length ⚠️

**One line is the default. Two is the ceiling for ordinary code.**
A comment states the intuition behind the design, never the history of the file
and never a paraphrase of the line below it. Full rules in [§5.1](#51-what-a-comment-must-earn).

---

## §1. Identifier Strings vs. Human-Readable Messages ⚠️

> **This is the single most dangerous mistake in this codebase.
> Adding a period to an identifier string causes silent runtime failures —
> no exception, no warning, just broken behavior.**

Strings fall into exactly two categories:

- **Identifier strings** — looked up or compared by the program:
  `hasattr(obj, "attr")`, `getattr`, `setattr`, `delattr`,
  `mode == "nixl_cpu"`, `unit in ("GB", "MB")`,
  Ray calls (`execute_all_async("method_name")`),
  dict keys, config keys, enum values.
  → **Never** add a period. **Never** change casing. Copy the name exactly.

- **Human-readable messages** — shown in logs, assertion failures, comments, docstrings.
  → Capitalize and punctuate per §2.

**Decision test**: if the string is passed to `==`, `in`, `hasattr`, `getattr`,
`execute_all_async`, `execute_rank_zero_async`, or used as a dict/config key,
it is an identifier. Everything else is a message.

```python
# GOOD — identifier strings are exact; message strings have period
assert hasattr(self, "device_mesh"), "device_mesh is not initialized."
assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    f"Expected nixl_cpu or nixl_gpu, got {self.psrl_config.ps_mode!r}."
)
self.ps_wg.execute_all_async("get_nixl_train_storage_client_name")
if config.routing_strategy.method == "throughput_balance":
    ...

# BAD — period leaked into identifier (silently broken!)
assert hasattr(self, "device_mesh."), "device_mesh is not initialized."
self.ps_wg.execute_all_async("get_nixl_train_storage_client_name.")
if config.routing_strategy.method == "throughput_balance.":
    ...
```

---

## §2. Capitalization, Punctuation & Identifier References

Three rules govern all human-readable text:

### 2.1 Sentence vs. Fragment

> **Sentence** (subject + verb) → **Uppercase** first letter + **period** `.`
> **Fragment** (no verb, short label) → **lowercase** OK + **no period**
> **When in doubt, treat it as a sentence.**

| Context | First letter | Ending | Example |
|---------|-------------|--------|---------|
| Comment — sentence | Uppercase | `.` | `# Set the rollout number based on batch size.` |
| Comment — fragment | lowercase OK | — | `# worker-side attributes` |
| Annotation marker | Uppercase | `.` | `# TODO(lhy): Add timeout handling.` |
| Docstring summary | Uppercase | `.` | `"""Wait for the NIXL push to complete."""` |
| Docstring arg desc | Uppercase | `.` optional | `timeout (float): Max wait time in seconds` |
| Assertion message | Uppercase | `.` | `"nixl_storage_client is not initialized."` |
| Log — normal | Uppercase | `.` | `"Initialized PS successfully."` |
| Log — ongoing | Uppercase | `...` | `"Getting the current PS model version..."` |
| Log — milestone | Uppercase | `!` | `"All configuration checks passed!"` |

### 2.2 After Colons

Text after `MARKER(author):` and `[prefix]:` always starts **uppercase**:

```python
# TODO(lhy): Add timeout handling for fault tolerance.     # GOOD
# TODO(lhy): add timeout handling for fault tolerance.     # BAD

psrl_logger.warning(f"[{log_prefix}]: Failed to log {name}: {e}.")   # GOOD
psrl_logger.warning(f"[{log_prefix}]: failed to log {name}: {e}.")   # BAD
```

### 2.3 Forbidden Endings

Never end a comment or message with `;` or `,`.

### 2.4 Backticks for Identifier References

The rule depends on whether the text is **read by a developer** (source code)
or **seen at runtime** (terminal output):

- **Comments & docstrings** (developer reads source) → wrap identifiers in backticks: `` `identifier` ``
- **Log messages & assertion messages** (shown at runtime) → plain text, no backticks

```python
# GOOD — backticks in comments
# The `rollout_coordinator` dispatches requests to `GenWorker` instances.

# BAD — identifier blends into prose
# The rollout_coordinator dispatches requests to GenWorker instances.

# GOOD — plain text in logs and assertions
psrl_logger.debug("Calling pull_model_state_dict_nixl for model update.")
assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."
```

In docstrings, use backticks in prose paragraphs. In `Args` / `Returns` / `Raises`,
the parameter name itself needs no backticks, but references to *other* identifiers do:

```python
def push_model(self, version: int) -> None:
    """
    Push the model state dict to `PSStorageWorker` via NIXL.

    Args:
        version (int): The target model version to push.
            Passed to `PSManager.bump_version` after the push completes.
    """
```

### 2.5 `!r` for Ambiguous Dynamic Values

In log messages and assertion messages, use `!r` for any dynamic value whose type
might be ambiguous (strings, None, enums). This makes debugging output unambiguous:

```python
psrl_logger.error(f"Unexpected status {status!r} for request {request_id}.")
assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    f"Expected nixl_cpu or nixl_gpu, got {self.psrl_config.ps_mode!r}."
)
```

---

## §3. Line Length & Line Breaking

**Hard limit: 119 characters** (enforced by Ruff).
**Never** use `\` for line continuation — always use parentheses.

### 3.1 Function Signatures

One parameter per line, 4-space indent, trailing comma, closing `)` on its own line.

```python
# GOOD
def push_model_state_dict_nixl(
    self,
    key: str,
    shards_to_transfer: list[int],
    target_client_name: str,
    next_ps_model_version: int,
) -> None:
    ...

# BAD
def push_model_state_dict_nixl(self, key: str, shards_to_transfer: list[int], target_client_name: str, next_ps_model_version: int) -> None:
```

### 3.2 Function Calls

Same pattern. Grouping on fewer lines is acceptable when only slightly over the limit.

```python
# GOOD — one per line
result = some_function(
    first_arg,
    second_arg,
    third_arg=value,
)

# GOOD — grouped, still under limit per line
result = some_function(
    first_arg, second_arg, third_arg=value,
)
```

### 3.3 Strings & f-strings

Use **implicit string concatenation** inside parentheses.

```python
# GOOD
psrl_logger.info(
    f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} "
    f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards."
)

# BAD — backslash continuation
psrl_logger.info(
    f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} " \
    f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards."
)
```

### 3.4 Conditionals

Break **before** each boolean operator.

```python
# GOOD
if (
    self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu")
    and self.nixl_storage_client is not None
    and version > self.current_version
):
    ...

# BAD — breaks after operator
if (self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu") and
    self.nixl_storage_client is not None and
    version > self.current_version):
    ...
```

### 3.5 Collections & Comprehensions

```python
# Collection — one element per line, trailing comma
supported_modes = [
    "nixl_cpu",
    "nixl_gpu",
    "cpu",
    "cpu_ref",
]

# Comprehension — one-liner if it fits
names = [w.name for w in workers if w.is_active]

# Comprehension — multi-line if long
names = [
    worker.name
    for worker in workers
    if worker.is_active and worker.role == PSRL_Role.Actor
]
```

---

## §4. Docstrings

Use **Google-style**. Opening `"""` on its **own line**.

```python
def wait_for_nixl_push_completion(self, timeout: float | None = None) -> bool:
    """
    Wait for the NIXL push wait thread to complete.

    Args:
        timeout (float | None): Maximum wait time in seconds.
            If None, wait indefinitely.

    Returns:
        bool: True if completed successfully, False if timed out.

    Raises:
        RuntimeError: If the push thread encountered an unrecoverable error.
    """
```

- Summary: one sentence, uppercase, period.
- Sections: `Args:`, `Returns:`, `Raises:`, `Usage:`.
- Arg type in parentheses with `|` union: `timeout (float | None):`. Never write `optional`.
- Long arg descriptions: continuation at 8-space indent.
- Reference other identifiers with backticks (§2.4).
- Add `Usage:` with code block when call sequencing is non-obvious.

**Length.** The summary line plus the sections is the whole docstring. Add a prose
paragraph only to state an invariant, a threading or ownership rule, or a unit that the
signature cannot express, and keep it to two or three lines. The comment anti-patterns
in §5.1 (history, restating the signature, comparing alternatives, teaching the
framework) are banned here too. One line of arg description per arg is normal.

---

## §5. Comments & Annotation Markers

### 5.1 What a Comment Must Earn ⚠️

> **A comment earns its line by carrying information the code cannot.
> One line is the default. Two is the ceiling for ordinary code.
> Three is the absolute maximum, and only for a `NOTE` explaining a real design trap.**

Write the **intuition**: the constraint, the invariant, the unit, the failure mode that
made this code look the way it does. Then stop. If you cannot say it in one line, the
explanation belongs in the function docstring or the code needs a better name, not a
longer comment.

**Never write these.** They are the four ways comments get bloated:

| Anti-pattern | Why it is worthless |
|--------------|---------------------|
| **History** (`# Previously we called X`, `# The old version did Y`, `# Changed from Z because...`) | Nobody reads code to learn what it used to be. `git log` and `git blame` own this. Delete the old approach, do not narrate it |
| **Restating the code** (`# Increment the counter` above `counter += 1`) | Costs a line, adds zero information, and rots the moment the code changes |
| **Comparison to alternatives** (`# We could have used a heap here, but...`) | Design-space tourism. Keep it only if the alternative is one a reader would actively try and break something doing so, and then it is one line |
| **Tutorials** (explaining what FSDP or a semaphore is) | The reader knows the language and the framework. Explain *this* code, not the concept |

```python
# BAD — 6 lines of history and self-narration for a 1-line fact
# Previously, we computed the staleness inside the trainer loop, which meant the
# value was recomputed on every microbatch. That turned out to be wasteful because
# the PS version only changes once per step. We could have cached it on the worker,
# but that would require plumbing the version through the worker constructor.
# So now we compute it once here and pass it down. This is much cleaner than
# the old approach and avoids the redundant recomputation.
staleness = self.ps_version - batch_version

# GOOD — one line, states the invariant a reader cannot see
# `ps_version` only advances once per step, so this is safe to hoist out of the microbatch loop.
staleness = self.ps_version - batch_version
```

```python
# BAD — restates the code
# Loop over the workers and call push on each one.
for worker in self.workers:
    worker.push()

# GOOD — no comment needed at all
for worker in self.workers:
    worker.push()
```

**A missing comment is cheaper than a bloated one.** When nothing non-obvious is
happening, write nothing.

### 5.2 Block Comments

Above the code, indented to match. Follow §2.1 (sentence vs. fragment).
Reference identifiers with backticks (§2.4).

```python
# Set the validation rollout number based on the global batch size.
config.actor_rollout_ref.rollout.val_rollout_n = ...

# worker-side attributes
self.worker_rank = worker_rank
```

- Blank line above when starting a new logical section.
- **Punctuation bans (`;`, `—`, `–`, clause dashes, double spaces) apply here in full.**
  See [Banned characters in ALL prose](#banned-characters-in-all-prose-) at the top.
  They are banned *anywhere* in the text, not only at the end.

### 5.3 Inline Comments

Two spaces before `#`, one space after. Short fragment preferred.

```python
self.ray_worker_group_cls = ray_worker_group_cls  # used only on train side
x = compute_staleness()  # This resets every epoch.
```

- Fragment → no period. Sentence → period.
- Do **not** vertically align across lines.

### 5.4 Section Separators

Sparingly, only in files >150 lines:

```python
# --- Initialization ---

# --- Main Training Loop ---
```

### 5.5 Annotation Markers

Format: `# MARKER(author): Uppercase message ending with period.`

| Marker | Purpose |
|--------|---------|
| `NOTE(xx)` | Design decisions, non-obvious choices |
| `TODO(xx)` | Future improvements, missing features |
| `FIXME(xx)` | Known bugs that must be fixed |
| `HACK(xx)` | Temporary workarounds (must explain why) |

- Always include author initials: `NOTE(lhy)`, `TODO(claude)`.
- Colon immediately after `)`, one space, then the message.
- **Uppercase** first letter, ends with **period**.
- **Three lines maximum**, including the marker line. Repeat `#` on each line,
  period on the **last line only**. A marker is not a place to escape §5.1.

```python
# GOOD — 2 lines, states the trap and its consequence
# NOTE(claude): Merge on the PS side rather than per key/shard, because one remote
# call per shard floods the Ray actor and it dies.

# BAD — no period, lowercase start
# NOTE(claude): we use a dict to store the PS handle and merge on the PS side
```

---

## §6. Assertions

Every `assert` **must** include a message. Uppercase first letter, period at the end.
Plain text for identifiers (§2.4). Include actual value with `!r` (§2.5).

```python
# Single-line
assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."

# Multi-line — parenthesized, 4-space indent
assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    "push_model_state_dict_nixl should only be used in nixl_cpu or nixl_gpu mode, "
    f"got: {self.psrl_config.ps_mode!r}."
)
```

---

## §7. Logging

### 7.1 Setup

```python
psrl_logger = logging.getLogger(__file__)
```

- Always `psrl_logger`, never `print()`.
- Prefix with `[component_name]` at key entry points.

### 7.2 Message Format

- Follow §2 for capitalization and punctuation.
- **f-strings only** — never `%`-style or `.format()`.
- `!r` for ambiguous dynamic values (§2.5).
- Plain text for identifiers, no backticks (§2.4).

```python
psrl_logger.debug("Getting the current PS model version...")
psrl_logger.info("[validate_config] All configuration checks passed!")
psrl_logger.info(f"PS mode set to {self.psrl_config.ps_mode!r}.")
psrl_logger.warning(f"[{log_prefix}]: Failed to log {name}: {e}.")
```

### 7.3 Multi-line

Implicit string concatenation, 4-space indent:

```python
psrl_logger.info(
    f"Worker {self.worker_rank} completed NIXL push for version {version} "
    f"with {num_shards} shards to {target_name} in {elapsed:.2f}s."
)
```

### 7.4 Log Levels

| Level | Use for | Example |
|-------|---------|---------|
| `debug` | Internal state, fine-grained tracing | `"Getting the current PS model version..."` |
| `info` | Lifecycle events, milestones | `"Initialized parameter server successfully."` |
| `warning` | Recoverable issues, degraded state | `"Staleness buffer is full, dropping oldest entry."` |
| `error` | Failures that need attention | `"Failed to push model to PSStorageWorker: {e}."` |

---

## §8. Imports & Type Annotations

### 8.1 Imports

**isort** order (enforced by Ruff): stdlib → third-party → local. Blank line between groups.
Use explicit imports; avoid `from module import *`.

```python
import logging
import time

import ray
from omegaconf import DictConfig

from psrl.utils.logger import (
    EventType,
    deprecated,
    get_ps_logger,
    get_worker_info,
    log_dual_events,
    log_single_event,
)
```

### 8.2 Type Annotations

- `X | None`, not `Optional[X]`.
- `list[int]`, `dict[str, Any]`, `tuple[int, ...]` (lowercase builtins).
- Complex types → `TypeAlias`:

```python
ShardMapping: TypeAlias = dict[str, list[int]]
```

---

## §9. Language

**English only.** No Chinese in comments, docstrings, or log messages.
