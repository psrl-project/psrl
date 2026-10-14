from abc import ABC, abstractmethod


class GenRewardFunctionBase(ABC):
    def __init__(self, using_sys_prompt: bool = True):
        self.using_sys_prompt = using_sys_prompt

    @abstractmethod
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
        """
        Scoring function for reward function.

        Args:
            data_source: Dataset source identifier.
            solution_str: Agent's solution string.
            rm_output: Reward model's generated output (decoded string).
            rm_output_value: Reward model's value output (for models like Skywork that output value directly).
            ground_truth: Ground truth answer.
            extra_info: Additional information.
            **kwargs: Additional keyword arguments.

        Returns:
            float: Reward score.
        """
        raise NotImplementedError

    @abstractmethod
    def prompt_constructor(self, **kwargs) -> list[dict]:
        """
        Prompt constructor for reward function.

        Args:
            **kwargs: Keyword arguments. Self-defined.

        Returns:
            list[dict]: Prompt.
                - "role": "system" | "user" | "assistant"
                - "content": str
        """
        raise NotImplementedError
