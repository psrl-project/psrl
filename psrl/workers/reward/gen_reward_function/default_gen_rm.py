import logging

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

from psrl.workers.reward.gen_reward_function.base import GenRewardFunctionBase
from psrl.workers.reward.gen_reward_function.registry import gen_reward_func

psrl_logger = logging.getLogger(__file__)
DEFAULT_GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


@gen_reward_func("default")
class DefaultGenRewardFunction(GenRewardFunctionBase):
    def __init__(self):
        super().__init__(using_sys_prompt=True)
        self.prompt_template = DEFAULT_GENRM_PROMPT_TEMPLATE

    def prompt_constructor(self, prompt_str: str, response_str: str) -> list[dict]:
        # Use template to construct prompt (default: GENRM_PROMPT_TEMPLATE format)
        rm_prompt = DEFAULT_GENRM_PROMPT_TEMPLATE.format(problem=prompt_str, solution=response_str)
        return [
            {
                "role": "user",
                "content": rm_prompt,
            }
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
        # Default gen_rm uses string output, ignore rm_output_value
        reward_score = 0.0
        try:
            boxed_result = last_boxed_only_string(rm_output)
            if boxed_result is not None:
                result = remove_boxed(boxed_result)
                reward_score = float(result == "True")
        except Exception as e:
            psrl_logger.warning(f"Error computing reward score from RM output: {e}")
        return reward_score
