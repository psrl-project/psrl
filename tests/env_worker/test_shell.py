import pytest
from psrl.workers.env_worker.shell import (
    PROCESS_DONE_MARKER_END,
    PROCESS_DONE_MARKER_START,
    parse_sentinel,
    truncate_observation,
    wrap_command,
)


@pytest.mark.cpu_test
def test_wrap_command_appends_sentinel_echo_and_trailing_newline():
    wrapped = wrap_command("ls -la")

    assert wrapped.startswith("ls -la\n"), "Command must be newline-terminated before the sentinel."
    assert wrapped.endswith("\n"), "Wrapped command must end with a newline so the shell executes it."
    assert PROCESS_DONE_MARKER_START in wrapped
    assert "$EXITSTATUS" in wrapped, "Sentinel must carry the shell exit status."


@pytest.mark.cpu_test
def test_wrap_command_does_not_double_newline():
    wrapped = wrap_command("ls -la\n")

    assert "ls -la\n\n" not in wrapped, "Already-terminated commands must not gain a blank line."


@pytest.mark.cpu_test
def test_parse_sentinel_returns_none_when_incomplete():
    assert parse_sentinel("partial output with no marker") is None


@pytest.mark.cpu_test
def test_parse_sentinel_extracts_body_and_exit_code():
    buffer = f"hello world\n{PROCESS_DONE_MARKER_START}0{PROCESS_DONE_MARKER_END}\n"
    parsed = parse_sentinel(buffer)

    assert parsed is not None, "A complete buffer must parse."
    body, exit_code = parsed
    assert exit_code == 0
    assert PROCESS_DONE_MARKER_START not in body, "Sentinel must be stripped from the observation."
    assert body.strip() == "hello world"


@pytest.mark.cpu_test
def test_parse_sentinel_handles_nonzero_exit_code():
    buffer = f"boom\n{PROCESS_DONE_MARKER_START}127{PROCESS_DONE_MARKER_END}\n"
    parsed = parse_sentinel(buffer)

    assert parsed is not None
    _, exit_code = parsed
    assert exit_code == 127


@pytest.mark.cpu_test
def test_parse_sentinel_returns_none_exit_code_for_unexpanded_variable():
    # A badly failing command can leave the variable unexpanded, which MLGym also
    # special-cases. It must not raise.
    buffer = f"garbage\n{PROCESS_DONE_MARKER_START}$EXITSTATUS{PROCESS_DONE_MARKER_END}\n"
    parsed = parse_sentinel(buffer)

    assert parsed is not None
    body, exit_code = parsed
    assert exit_code is None, "Unparseable exit status must degrade to None, not raise."
    assert "garbage" in body


@pytest.mark.cpu_test
def test_truncate_observation_keeps_short_text_unchanged():
    assert truncate_observation("short", max_chars=100) == "short"


@pytest.mark.cpu_test
def test_truncate_observation_keeps_head_and_tail_with_marker():
    text = "".join(f"line{i}\n" for i in range(1000))
    result = truncate_observation(text, max_chars=200)

    assert len(result) <= 400, f"Truncated output length={len(result)}. Expected at most 400 characters."
    assert "line0" in result, "Head of the output must be preserved."
    assert "line999" in result, "Tail of the output must be preserved."
    assert "omitted" in result, "An elision marker must tell the agent output was cut."


@pytest.mark.cpu_test
def test_truncate_observation_disabled_when_max_chars_non_positive():
    text = "x" * 5000
    assert truncate_observation(text, max_chars=0) == text
