import inspect

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register


@register("gdpo")
class GDPORewardLoop(RewardLoopBase):
    """Reward loop for GDPO."""

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

    async def run_single(self, data: TensorDict) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = tu.get(data_item, "data_source")
        ground_truth = tu.get(data_item, "reward_model")["ground_truth"]
        extra_info = self.merge_extra_info(
            data_item,
            experiment_name=self.config.trainer.experiment_name,
        )

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
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
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
