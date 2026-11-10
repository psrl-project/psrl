import logging
import os

from psrl.workers.reward.gen_reward_function.base import GenRewardFunctionBase
from psrl.workers.reward.gen_reward_function.registry import gen_reward_func

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@gen_reward_func("skywork")
class SkyworkGenRewardFunction(GenRewardFunctionBase):
    def __init__(self):
        super().__init__(using_sys_prompt=False)

    def prompt_constructor(self, prompt_str: str, response_str: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": prompt_str,
            },
            {
                "role": "assistant",
                "content": response_str,
            },
        ]

    def compute_score(
        self,
        data_source: str,
        solution_str: str,
        rm_output: str,
        rm_output_value: float | None = None,
        ground_truth: str = "",
        extra_info: dict | None = None,
        **kwargs,
    ) -> float:
        # For Skywork reward models, use logits if available, otherwise fall back to string output
        psrl_logger.info(f"SkyworkGenRewardFunction.compute_score called with rm_output_value={rm_output_value}")
        if rm_output_value is not None:
            score = float(rm_output_value)
            psrl_logger.info(f"SkyworkGenRewardFunction returning score={score}")
            return score
        else:
            psrl_logger.warning("No rm_output_value provided for Skywork reward model, returning 0.0")
            return 0.0
