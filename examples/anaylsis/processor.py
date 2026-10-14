import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

Processor = Callable[[dict[str, Any]], tuple[bool, dict[str, float], Any]]


def build_processor(processor_name: str) -> Processor:
    if processor_name == "intertoken_indexed":
        return make_intertoken_indexed_processor()
    elif processor_name == "intertoken_indexed_by_kv_cache_usage":
        return make_intertoken_indexed_by_kv_cache_usage_processor()
    elif processor_name == "prompt_time_indexed":
        return make_prompt_time_indexed_processor()
    elif processor_name == "generation_time_indexed":
        return make_generation_time_indexed_processor()
    elif processor_name == "instance_request_num_indexed_by_time":
        return make_instance_request_num_indexed_by_time_processor()
    elif processor_name == "instance_throughput_indexed_by_time":
        return make_instance_throughput_indexed_by_time_processor()
    elif processor_name == "instance_kv_cache_indexed_by_time":
        return make_instance_kv_cache_indexed_by_time_processor()
    elif processor_name == "instance_request_num_indexed_by_time_smg":
        return make_instance_request_num_indexed_by_time_smg_processor()
    elif processor_name == "instance_throughput_indexed_by_time_smg":
        return make_instance_throughput_indexed_by_time_smg_processor()
    elif processor_name == "instance_kv_cache_indexed_by_time_smg":
        return make_instance_kv_cache_indexed_by_time_smg_processor()
    else:
        raise ValueError(f"Invalid processor name: {processor_name}")


def make_intertoken_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)

    - Keep rows where:
        iteration_stats.num_finished_requests == 0
        AND iteration_stats.time_to_first_tokens_avg == 0.0
    - values: {"inter_token_latencies_avg": float_value}
    - x_value: 1-based index of the kept row (1,2,3,...)
    """
    idx = 0  # enclosed counter

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        nonlocal idx
        it = obj.get("iteration_stats")
        if not isinstance(it, dict):
            return False, {}, None

        # Check num_finished_requests == 0
        try:
            num_finished = it.get("num_finished_requests", None)
            if num_finished is None or int(num_finished) != 0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Check time_to_first_tokens_avg == 0.0 (exact match)
        try:
            ttf = it.get("time_to_first_tokens_avg", None)
            if ttf is None or float(ttf) != 0.0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Extract inter_token_latencies_avg as float
        try:
            inter = it.get("inter_token_latencies_avg", None)
            if inter is None:
                return False, {}, None
            val = float(inter)
            if not math.isfinite(val):
                return False, {}, None
        except Exception:
            return False, {}, None

        # Passed all checks -> increment index and return point
        idx += 1
        return True, {"inter_token_latencies_avg": val}, idx

    return processor


def make_intertoken_indexed_by_kv_cache_usage_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)

    - Keep rows where:
        iteration_stats.num_finished_requests == 0
        AND iteration_stats.time_to_first_tokens_avg == 0.0
    - values: {"inter_token_latencies_avg": float_value}
    - x_value: kv_cache_usage as float
    """
    inter_token_latencies_avg_threshold = 0.1
    kv_cache_usage_threshold = 0.95
    stop_plot = False
    print_num = 10
    is_print_init = 0
    is_print_final = 0

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        nonlocal \
            inter_token_latencies_avg_threshold, \
            kv_cache_usage_threshold, \
            stop_plot, \
            print_num, \
            is_print_init, \
            is_print_final
        if stop_plot:
            return False, {}, None

        iteration_stats_it = obj.get("iteration_stats")
        scheduler_stats_it = obj.get("scheduler_stats")
        if not isinstance(iteration_stats_it, dict) or not isinstance(scheduler_stats_it, dict):
            return False, {}, None

        # Check num_finished_requests == 0
        try:
            num_finished = iteration_stats_it.get("num_finished_requests", None)
            if num_finished is None or int(num_finished) != 0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Check time_to_first_tokens_avg == 0.0 (exact match)
        try:
            ttf = iteration_stats_it.get("time_to_first_tokens_avg", None)
            if ttf is None or float(ttf) != 0.0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Extract kv_cache_usage as float
        try:
            kv_cache_usage = scheduler_stats_it.get("kv_cache_usage", None)
            if kv_cache_usage is None:
                return False, {}, None
            kv_cache_usage_val = float(kv_cache_usage)
            if not math.isfinite(kv_cache_usage_val):
                return False, {}, None
            if kv_cache_usage_val >= kv_cache_usage_threshold:
                stop_plot = True
                return False, {}, None
        except Exception:
            return False, {}, None

        # Extract inter_token_latencies_avg as float
        try:
            inter = iteration_stats_it.get("inter_token_latencies_avg", None)
            if inter is None:
                return False, {}, None
            val = float(inter)
            if not math.isfinite(val):
                return False, {}, None
            if val == 0:
                return False, {}, None
            if val > inter_token_latencies_avg_threshold:
                return False, {}, None
        except Exception:
            return False, {}, None

        if is_print_init < print_num:
            is_print_init += 1
            print(f"init: kv_cache_usage: {kv_cache_usage_val}, inter_token_latencies_avg: {val}")

        if is_print_final < print_num and kv_cache_usage_val >= 0.1:
            is_print_final += 1
            print(f"10%: kv_cache_usage: {kv_cache_usage_val}, inter_token_latencies_avg: {val}")

        # Passed all checks
        return True, {"inter_token_latencies_avg": val}, kv_cache_usage_val

    return processor


def make_prompt_time_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """
    total_prompt_tokens = 0
    total_prompt_time = 0

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        nonlocal total_prompt_tokens, total_prompt_time
        it = obj.get("iteration_stats")
        if not isinstance(it, dict):
            return False, {}, None

        # Extract num_prompt_tokens
        num_prompt_tokens = obj["iteration_stats"]["num_prompt_tokens"]
        num_prompt_tokens = float(num_prompt_tokens)

        if num_prompt_tokens == 0:
            return False, {}, None

        # Extract prompt_throughput
        prompt_throughput = obj["throughput_stats"]["prompt_throughput"]
        prompt_throughput = float(prompt_throughput)
        assert prompt_throughput > 0

        # Passed all checks -> increment index and return point
        total_prompt_tokens += num_prompt_tokens
        total_prompt_time += num_prompt_tokens / prompt_throughput
        print(f"total_prompt_tokens: {total_prompt_tokens}, total_prompt_time: {total_prompt_time}")
        return (
            True,
            {"prompt_time": num_prompt_tokens / prompt_throughput},
            num_prompt_tokens,
        )

    return processor


def make_generation_time_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """
    idx = 0

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        nonlocal idx
        if obj["type"] != "generation_end":
            return False, {}, None

        # Extract generation_time as float
        try:
            generation_time = obj.get("generation_time")
            if generation_time is None:
                return False, {}, None
            val = float(generation_time)
            if not math.isfinite(val):
                return False, {}, None
        except Exception:
            return False, {}, None

        # Passed all checks -> increment index and return point
        idx += 1
        return True, {"generation_time": val}, idx

    return processor


def make_instance_request_num_indexed_by_time_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        if "total_elapsed_time" not in obj:
            return False, {}, None

        total_elapsed_time = float(obj["total_elapsed_time"])

        it = obj.get("scheduler_stats")
        if not isinstance(it, dict):
            return False, {}, None

        # Extract request num
        running_request_num = obj["scheduler_stats"]["num_running_reqs"]
        waiting_request_num = obj["scheduler_stats"]["num_waiting_reqs"]
        request_num = running_request_num + waiting_request_num

        # Passed all checks -> increment index and return point
        return (
            True,
            {"request_num": request_num, "running_request_num": running_request_num},
            total_elapsed_time,
        )

    return processor


def make_instance_throughput_indexed_by_time_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        if "total_elapsed_time" not in obj:
            return False, {}, None

        total_elapsed_time = float(obj["total_elapsed_time"])

        generation_throughput = obj.get("generation_throughput")
        if generation_throughput is None:
            return False, {}, None
        generation_throughput = float(generation_throughput)

        return (
            True,
            {"generation_throughput": generation_throughput},
            total_elapsed_time,
        )

    return processor


def make_instance_kv_cache_indexed_by_time_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)

    - x_value: total_elapsed_time
    - values: {"kv_cache_usage": float}
    """

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        if "total_elapsed_time" not in obj:
            return False, {}, None

        total_elapsed_time = float(obj["total_elapsed_time"])

        scheduler_stats = obj.get("scheduler_stats")
        if not isinstance(scheduler_stats, dict):
            return False, {}, None

        kv_cache_usage = scheduler_stats.get("kv_cache_usage")
        if kv_cache_usage is None:
            return False, {}, None
        try:
            kv_cache_usage_val = float(kv_cache_usage)
        except Exception:
            return False, {}, None
        if not math.isfinite(kv_cache_usage_val):
            return False, {}, None

        return (
            True,
            {"kv_cache_usage": kv_cache_usage_val},
            total_elapsed_time,
        )

    return processor


# --- SMG flat schema processors ---


def _smg_elapsed_seconds(obj: dict[str, Any], state: dict[str, Any]):
    """
    Return relative seconds for the x-axis, or None if unavailable.

    Prefer `total_elapsed_time`. Otherwise, derive elapsed
    seconds from the absolute ``ts`` timestamp using this file's first row as t0.
    """
    if "total_elapsed_time" in obj:
        try:
            return float(obj["total_elapsed_time"])
        except Exception:
            return None

    ts = obj.get("ts")
    if not isinstance(ts, str):
        return None
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return None
    if state["t0"] is None:
        state["t0"] = t
    return (t - state["t0"]).total_seconds()


def make_instance_request_num_indexed_by_time_smg_processor() -> Processor:
    """SMG flat-schema variant of instance_request_num_indexed_by_time."""
    state = {"t0": None}

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        x = _smg_elapsed_seconds(obj, state)
        if x is None:
            return False, {}, None

        if "num_running_reqs" not in obj or "num_waiting_reqs" not in obj:
            return False, {}, None
        running_request_num = obj["num_running_reqs"]
        waiting_request_num = obj["num_waiting_reqs"]
        request_num = running_request_num + waiting_request_num

        return (
            True,
            {"request_num": request_num, "running_request_num": running_request_num},
            x,
        )

    return processor


def make_instance_throughput_indexed_by_time_smg_processor() -> Processor:
    """SMG flat-schema variant of instance_throughput_indexed_by_time."""
    state = {"t0": None}

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        x = _smg_elapsed_seconds(obj, state)
        if x is None:
            return False, {}, None

        generation_throughput = obj.get("generation_throughput")
        if generation_throughput is None:
            return False, {}, None
        try:
            generation_throughput = float(generation_throughput)
        except Exception:
            return False, {}, None

        return (
            True,
            {"generation_throughput": generation_throughput},
            x,
        )

    return processor


def make_instance_kv_cache_indexed_by_time_smg_processor() -> Processor:
    """SMG flat-schema variant of instance_kv_cache_indexed_by_time."""
    state = {"t0": None}

    def processor(obj: dict[str, Any]) -> tuple[bool, dict[str, float], int]:
        x = _smg_elapsed_seconds(obj, state)
        if x is None:
            return False, {}, None

        kv_cache_usage = obj.get("kv_cache_usage")
        if kv_cache_usage is None:
            return False, {}, None
        try:
            kv_cache_usage_val = float(kv_cache_usage)
        except Exception:
            return False, {}, None
        if not math.isfinite(kv_cache_usage_val):
            return False, {}, None

        return (
            True,
            {"kv_cache_usage": kv_cache_usage_val},
            x,
        )

    return processor
