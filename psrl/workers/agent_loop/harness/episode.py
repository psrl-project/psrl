"""What the harness CLI itself reported about the episode it ran.

The framework must not reconstruct an episode's stop reason or its turn count from
its own bookkeeping. Two of its counters are known to disagree with the CLI's:

- A harness that rewrites its own context (Claude Code's auto-compact) replays
  messages the session never generated, and the gateway cuts a trajectory there.
  The CLI's turn counter keeps running across that rewrite, so a per-trajectory
  count covers only the segment after the last rewrite while the CLI counts the run.
- A harness that runs a subagent makes model calls the CLI does not count against
  the parent, while every call reaches the same endpoint.

So the CLI's own machine-readable output is the authority, parsed here per
``trajectory_format``. Everything downstream reads that verdict rather than a
reconstruction of it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from psrl.utils.common.http_utils import SESSION_TURN_BUDGET_ERROR_CODE

# Trajectory output formats. These key both this dispatch and `HarnessConfig`'s
# kind default, so they live here rather than in two places.
CLAUDE_CODE_STREAM_JSON = "claude_code_stream_json"
CODEX_JSONL = "codex_jsonl"

# The stop reason a harness reports when its own turn cap stopped it. A harness
# maps its own wording onto this, so a caller never parses harness-specific text.
STOP_REASON_MAX_TURNS = "max_turns"


@dataclass(frozen=True)
class HarnessEpisodeSummary:
    """What a harness CLI said about the episode it just ran.

    ``stop_reason`` is the CLI's own reason in the vocabulary this module defines,
    or its raw terminal reason when it has no equivalent here. ``num_turns`` is the
    number of turns the CLI counted for the episode: subagent turns excluded, and
    every internal run of one invocation summed, because a single invocation can
    report several results.
    """

    stop_reason: str | None = None
    num_turns: int | None = None
    # Subagents still live when the main sample closed, when the CLI reports
    # enough lifecycle state to know. None means the CLI did not report it.
    background_subagents: int | None = None


def episode_summary(trajectory_format: str, log_bytes: bytes) -> HarnessEpisodeSummary | None:
    """Parse a harness's own log for what the CLI reported about the episode.

    Returns ``None`` when the format carries no such report, so a caller can tell
    "the harness reported nothing" from "the harness reported zero turns".
    """
    parser = _EPISODE_PARSERS.get(trajectory_format)
    if parser is None:
        return None
    return parser(log_bytes)


def _parse_claude_code_episode(log_bytes: bytes) -> HarnessEpisodeSummary | None:
    """Read the Claude Code stream-json results.

    ``-p --output-format stream-json`` closes every internal run with a ``result``
    event carrying ``num_turns`` and a ``terminal_reason``. One invocation can run
    more than once (a continuation re-enters the query loop in the same session and
    emits its own ``init``/``result`` pair), and each run counts from its own start,
    so the episode's count is their sum. That sum matched the main-agent model calls
    observed on the wire, with a subagent's calls excluded.
    """
    capped = False
    saw_turns = False
    turns = 0
    last_reason: str | None = None
    found = False
    background_subagents: int | None = None

    for event in _iter_json_events(log_bytes):
        if event.get("type") != "result":
            continue
        found = True
        reported = event.get("num_turns")
        run_turns = reported if isinstance(reported, int) and reported >= 0 else None
        stats = event.get("subagent_stats")
        if isinstance(stats, dict):
            started = stats.get("started_in_background")
            completed = stats.get("completed")
            failed = stats.get("failed")
            killed = stats.get("killed")
            killed_total = (
                sum(int(value) for value in killed.values() if isinstance(value, int))
                if isinstance(killed, dict)
                else 0
            )
            if isinstance(started, int):
                background_subagents = max(
                    0,
                    started
                    - (int(completed) if isinstance(completed, int) else 0)
                    - (int(failed) if isinstance(failed, int) else 0)
                    - killed_total,
                )
        if _claude_code_hit_the_turn_cap(event):
            capped = True
            # A capped run reports the turn it refused to start, not the turns it
            # completed, so its count is one more than the work it did.
            if run_turns is not None:
                run_turns = max(0, run_turns - 1)
        else:
            last_reason = event.get("terminal_reason") or event.get("subtype") or last_reason
        if run_turns is not None:
            saw_turns = True
            turns += run_turns

    if not found:
        return None
    return HarnessEpisodeSummary(
        stop_reason=STOP_REASON_MAX_TURNS if capped else last_reason,
        num_turns=turns if saw_turns else None,
        background_subagents=background_subagents,
    )


def _claude_code_hit_the_turn_cap(event: dict) -> bool:
    """Whether one Claude Code result event reports its ``--max-turns`` cap."""
    return event.get("terminal_reason") == STOP_REASON_MAX_TURNS or event.get("subtype") == "error_max_turns"


def _parse_codex_episode(log_bytes: bytes) -> HarnessEpisodeSummary | None:
    """Read a Codex run's own report of how it stopped and which agents are live.

    Codex reports no turn count of its own: its ``turn`` is one submission, not one model
    call, so a single ``exec`` reports one turn however many times it called the model.
    The one stop reason it can be asked about is a budget this framework enforced, because
    it arrives as an error whose body the CLI echoes verbatim. Any other reason is the
    CLI's own wording, which is not this module's vocabulary, so it is left unreported.
    """
    stop_reason = None
    agents: dict[str, str] = {}
    saw_agents = False
    for event in _iter_json_events(log_bytes):
        if any(SESSION_TURN_BUDGET_ERROR_CODE in message for message in _codex_error_messages(event)):
            stop_reason = STOP_REASON_MAX_TURNS
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "collab_tool_call":
            continue
        states = item.get("agents_states")
        if not isinstance(states, dict):
            continue
        for agent_id, state in states.items():
            if not isinstance(state, dict):
                continue
            status = state.get("status")
            if isinstance(status, str):
                agents[str(agent_id)] = status
                saw_agents = True
    if stop_reason is None and not saw_agents:
        return None
    terminal = {"completed", "failed", "killed", "cancelled", "error", "aborted"}
    background_subagents = (
        sum(1 for status in agents.values() if status not in terminal) if saw_agents else None
    )
    return HarnessEpisodeSummary(
        stop_reason=stop_reason,
        background_subagents=background_subagents,
    )


def _codex_error_messages(event: dict) -> Iterator[str]:
    """Yield the error text of one Codex event, if it carries any."""
    if event.get("type") == "error" and isinstance(event.get("message"), str):
        yield event["message"]
        return
    if event.get("type") == "turn.failed":
        error = event.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            yield error["message"]


def _iter_json_events(log_bytes: bytes) -> Iterator[dict]:
    """Yield the JSON objects in one JSONL stream, skipping anything else.

    A harness log interleaves JSON events with human-readable output, and a partial
    line is possible when the CLI was stopped mid-write. Neither is an error: the
    caller is reading a report, so anything unparsable is simply not part of it.
    """
    for raw_line in log_bytes.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"{"):
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(event, dict):
            yield event


_EPISODE_PARSERS: dict[str, Callable[[bytes], HarnessEpisodeSummary | None]] = {
    CLAUDE_CODE_STREAM_JSON: _parse_claude_code_episode,
    CODEX_JSONL: _parse_codex_episode,
}
