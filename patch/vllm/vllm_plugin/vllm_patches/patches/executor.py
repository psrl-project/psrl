import logging
import os
import time

from vllm.v1.executor.abstract import Executor

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@min_vllm_version("0.22.0")
class TMSExecutorPatch(vLLMPatch[Executor]):
    """
    Add graph sleep support in TMS.

    Compatible with vLLM 0.22.0+
    """

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            psrl_logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        if os.environ.get("PSRL_VLLM_PATCHES", "") == "TMS:GRAPH":
            self.sleeping_tags.add("graph")
        self.is_sleeping = True
        psrl_logger.info("It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep)
