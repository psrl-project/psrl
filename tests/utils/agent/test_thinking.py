"""Trajectory-retention and template-compatibility rules for `thinking_template`."""

from pathlib import Path

import pytest
from psrl.utils.agent.thinking import (
    DISABLE_THINKING,
    LONGEST_TRAJ,
    MULTI_THINKING,
    MULTI_TRAJ,
    harness_extra_body,
    requires_accumulating_template,
    select_trajectories,
    template_prefills_assistant_think,
    validate_thinking_template,
)

pytestmark = pytest.mark.cpu_test


def _traj(trajectory_id: int, num_turns: int, response_len: int) -> dict:
    """Build one TITO trajectory shaped like `build_training_data` returns it."""
    return {
        "trajectory_id": trajectory_id,
        "num_turns": num_turns,
        "response_ids": list(range(response_len)),
    }


class TestSelectTrajectories:
    """LONGEST_TRAJ must rank by turn count first and response tokens second."""

    def test_turn_count_beats_token_count(self):
        # The deep chain carries more of the episode despite being terser.
        deep = _traj(0, num_turns=5, response_len=100)
        verbose = _traj(1, num_turns=1, response_len=9000)
        assert select_trajectories(LONGEST_TRAJ, [verbose, deep]) == [deep]

    def test_token_count_breaks_turn_ties(self):
        # Every fork is one turn deep in the observed failure mode, so the
        # tiebreak decides. It must still pick the longest response.
        short = _traj(0, num_turns=1, response_len=10)
        long = _traj(1, num_turns=1, response_len=400)
        assert select_trajectories(LONGEST_TRAJ, [short, long]) == [long]

    def test_full_tie_keeps_earliest(self):
        first = _traj(0, num_turns=2, response_len=50)
        second = _traj(1, num_turns=2, response_len=50)
        assert select_trajectories(LONGEST_TRAJ, [first, second]) == [first]

    def test_missing_num_turns_does_not_raise(self):
        # A trajectory dict without `num_turns` must sort as depth zero.
        bare = {"trajectory_id": 0, "response_ids": [1, 2, 3]}
        real = _traj(1, num_turns=3, response_len=1)
        assert select_trajectories(LONGEST_TRAJ, [bare, real]) == [real]

    @pytest.mark.parametrize("mode", [MULTI_TRAJ, MULTI_THINKING, DISABLE_THINKING])
    def test_other_modes_pass_through(self, mode):
        data = [_traj(0, 1, 10), _traj(1, 5, 20)]
        assert select_trajectories(mode, data) == data

    def test_single_trajectory_passes_through(self):
        data = [_traj(0, 1, 10)]
        assert select_trajectories(LONGEST_TRAJ, data) == data

    def test_empty_passes_through(self):
        assert select_trajectories(LONGEST_TRAJ, []) == []


class TestTemplateCompatibility:
    """Only `multi_thinking` wants an accumulating template. Rejection keys on the
    template's own text, not on the mode, because an accumulating template that emits
    a bare content stays balanced with thinking off. SkyRL ships exactly that pairing
    in `examples/train/thunder_agent/scripts/r2egym_32b/run_trainer.sh`.
    """

    # The Qwen3.5 shape. It opens a `<think>` the template never closes itself.
    PREFILL_SRC = "{{- '<|im_start|>assistant\\n<think>\\n' + content + '<|im_end|>\\n' }}"
    # The Qwen3 shape, which SkyRL also uses. Bare content, nothing to leave unclosed.
    BARE_SRC = "{{- '<|im_start|>' + message.role + '\\n' + content }}"

    def test_multi_thinking_requires_accumulating(self):
        assert requires_accumulating_template(MULTI_THINKING)

    @pytest.mark.parametrize("mode", [MULTI_TRAJ, LONGEST_TRAJ, DISABLE_THINKING])
    def test_other_modes_do_not_require_accumulating(self, mode):
        assert not requires_accumulating_template(mode)

    def test_prefilling_template_is_detected(self):
        assert template_prefills_assistant_think(self.PREFILL_SRC)

    def test_bare_content_template_is_allowed(self):
        assert not template_prefills_assistant_think(self.BARE_SRC)

    def test_real_qwen3_template_does_not_prefill(self):
        src = Path("psrl/utils/tito/templates/qwen3_thinking_2507_and_next_fixed.jinja").read_text()
        assert not template_prefills_assistant_think(src)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="thinking_template must be one of"):
            validate_thinking_template("no_thinking")


class TestHarnessExtraBody:
    """The gateway overrides must match what each mode needs on the wire."""

    def test_disable_thinking_turns_the_toggle_off(self):
        assert harness_extra_body(DISABLE_THINKING) == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_multi_thinking_keeps_reasoning_inline(self):
        assert harness_extra_body(MULTI_THINKING) == {"separate_reasoning": False}

    @pytest.mark.parametrize("mode", [MULTI_TRAJ, LONGEST_TRAJ])
    def test_trajectory_modes_send_nothing(self, mode):
        assert harness_extra_body(mode) == {}
