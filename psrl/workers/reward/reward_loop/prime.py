import asyncio
import inspect
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import psutil
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register


async def single_compute_score(
    evaluation_func,
    completion,
    reference,
    task,
    task_extra_info,
    executor,
    timeout=300.0,
):
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info),
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None


@register("prime")
class PrimeRewardLoop(RewardLoopBase):
    """Reward loop used by https://github.com/PRIME-RL/PRIME."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        is_validate=False,
        **reward_kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score_async
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        self.num_examine = reward_kwargs.get("num_examine", 1)
        self.reward_fn_key = reward_kwargs.get("reward_fn_key", "data_source")
        self.num_processes = reward_kwargs.get("num_processes", 64)
        self.timeout = reward_kwargs.get("timeout", 300.0)

        self.already_print_data_sources = {}

    async def run_single(self, data: TensorDict) -> dict:
        """Process a single data item and return reward score.

        Args:
            data: TensorDict containing a single data item

        Returns:
            dict: Dictionary containing 'reward_score' and 'reward_extra_info'
        """
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
            try:
                result = await asyncio.wait_for(
                    self.compute_score(data_source, response_str, ground_truth, extra_info),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                print(f"[Timeout] Task timeout: {response_str[:80]}")
                result = None
            except Exception as e:
                print(f"[Error] PRIME scoring failed: {e}, completion: {response_str[:80]}")
                result = None
        else:
            with ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    result = await single_compute_score(
                        self.compute_score,
                        response_str,
                        ground_truth,
                        data_source,
                        extra_info,
                        executor,
                        timeout=self.timeout,
                    )
                except Exception as e:
                    print(f"[Error] PRIME scoring failed: {e}, completion: {response_str[:80]}")
                    result = None
                finally:
                    for pid, proc in executor._processes.items():
                        try:
                            p = psutil.Process(pid)
                            p.terminate()
                            try:
                                p.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                p.kill()
                        except Exception:
                            pass

        reward_extra_info = {}

        score: float
        if result is None or isinstance(result, Exception):
            score = 0.0
        elif isinstance(result, int | float | bool):
            score = float(result)
        elif isinstance(result, dict):
            score = result.get("score", float(result[0]) if result else 0.0)
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = float(result[0]) if result else 0.0

        reward_extra_info["acc"] = score
        reward_extra_info["score"] = score
        reward = score

        if data_source not in self.already_print_data_sources:
            self.already_print_data_sources[data_source] = 0

        if self.already_print_data_sources[data_source] < self.num_examine:
            self.already_print_data_sources[data_source] += 1
            print(f"[PRIME Example] Data source: {data_source}, Response: {response_str[:200]}, Score: {score}")

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
