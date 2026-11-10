import inspect

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register


@register("dapo")
class DAPORewardLoop(RewardLoopBase):
    """Reward loop for DAPO."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        **reward_kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score_async
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = reward_kwargs.get("max_resp_len", None)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    async def run_single(self, data: TensorDict) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = tu.get(data_item, "data_source")
        ground_truth = tu.get(data_item, "reward_model")["ground_truth"]
        num_turns = tu.get(data_item, "num_turns", None)
        rollout_reward_scores = tu.get(data_item, "reward_scores", {})
        extra_info = self.merge_extra_info(
            data_item,
            num_turns=num_turns,
            rollout_reward_scores=rollout_reward_scores,
        )

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["score"] = score
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
