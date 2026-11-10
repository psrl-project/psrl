import asyncio
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_test


SCRIPT = Path(__file__).parents[2] / "scripts" / "test_vllm_reprefill_vs_fsdp.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("test_vllm_reprefill_vs_fsdp_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mismatch_statistics_and_ratio_masking():
    result = MODULE.mismatch_statistics(
        trainer=[0.0, 0.0, 0.0],
        rollout=[0.0, 1.0, -1.0],
        outlier_thresholds=[0.5],
        ratio_low=0.5,
        ratio_high=2.0,
    )

    assert result["signed_mean"] == pytest.approx(0.0)
    assert result["mean_absolute_error"] == pytest.approx(2.0 / 3.0)
    assert result["absolute_outliers"]["0.5"]["count"] == 2
    assert result["ratio"]["hypothetical_masked_count"] == 2


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (91, "far"),
        (92, "before_-8_-1"),
        (100, "boundary_0"),
        (108, "after_1_8"),
        (109, "after_9_32"),
        (150, "far"),
    ],
)
def test_boundary_categories(position, expected):
    result = MODULE.token_boundary_metadata(position, [100, 200])
    assert result["boundary_category"] == expected


def test_boundary_distance_metadata():
    result = MODULE.token_boundary_metadata(108, [100, 200])

    assert result == {
        "is_segment_boundary": False,
        "distance_to_previous_boundary": 8,
        "distance_to_next_boundary": 92,
        "nearest_boundary_signed_distance": 8,
        "boundary_category": "after_1_8",
    }


def test_longest_common_prefix():
    assert MODULE._longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert MODULE._longest_common_prefix([1, 2], [1, 2, 3]) == 2


def test_segmented_reprefill_uses_latest_historical_logprobs(monkeypatch):
    response_ids = [20, 21, 22, 23, 24]
    decode_logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    call_index = 0

    async def fake_collect_request(
        _engine,
        *,
        prompt_token_ids,
        sampling_params,
        request_id,
        abort_after,
        timeout_s,
    ):
        nonlocal call_index
        del sampling_params, request_id, timeout_s
        consumed = len(prompt_token_ids) - 2
        count = abort_after or (len(response_ids) - consumed)
        records = [
            {"token_id": token_id, "logprob": logprob}
            for token_id, logprob in zip(
                response_ids[consumed : consumed + count],
                decode_logprobs[consumed : consumed + count],
                strict=True,
            )
        ]
        prompt_records = [{"token_id": token_id, "logprob": None} for token_id in prompt_token_ids[:2]]
        prompt_records.extend(
            {
                "token_id": token_id,
                "logprob": -10.0 * call_index - position,
            }
            for position, token_id in enumerate(prompt_token_ids[2:])
        )
        call_index += 1
        return records, prompt_records, abort_after is not None

    monkeypatch.setattr(MODULE, "_collect_request", fake_collect_request)
    monkeypatch.setattr(MODULE, "_sampling_params", lambda _args, max_tokens: max_tokens)
    args = SimpleNamespace(response_length=5, timeout_s=1.0)

    case = asyncio.run(
        MODULE._generate_segmented(
            object(),
            args,
            prompt_id=0,
            prompt=None,
            prompt_token_ids=[10, 11],
            segment_length=2,
        )
    )

    assert case.response_token_ids == response_ids
    assert case.logprob_concat == decode_logprobs
    assert case.logprob_replace == [-20.0, -21.0, -22.0, -23.0, -5.0]
    assert case.replacement_source == ["reprefill_2"] * 4 + ["decode"]
    assert case.segment_id == [0, 0, 1, 1, 2]
    assert case.boundaries == [2, 4]
    assert case.reprefill_history[0]["reprefill_index"] == 1
    assert case.reprefill_history[1]["reprefill_index"] == 2
    assert case.final_prefill["enabled"] is False


def test_segmented_final_prefix_score_is_separate_from_online_replacement(monkeypatch):
    response_ids = [20, 21, 22, 23, 24]
    decode_logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    call_index = 0

    async def fake_collect_request(
        _engine,
        *,
        prompt_token_ids,
        sampling_params,
        request_id,
        abort_after,
        timeout_s,
    ):
        nonlocal call_index
        del sampling_params, request_id, timeout_s
        consumed = len(prompt_token_ids) - 2
        count = abort_after or (len(response_ids) - consumed)
        records = [
            {"token_id": token_id, "logprob": logprob}
            for token_id, logprob in zip(
                response_ids[consumed : consumed + count],
                decode_logprobs[consumed : consumed + count],
                strict=True,
            )
        ]
        prompt_records = [{"token_id": token_id, "logprob": None} for token_id in prompt_token_ids[:2]]
        prompt_records.extend(
            {
                "token_id": token_id,
                "logprob": -10.0 * call_index - position,
            }
            for position, token_id in enumerate(prompt_token_ids[2:])
        )
        call_index += 1
        return records, prompt_records, abort_after is not None

    monkeypatch.setattr(MODULE, "_collect_request", fake_collect_request)
    monkeypatch.setattr(MODULE, "_sampling_params", lambda _args, max_tokens: max_tokens)
    args = SimpleNamespace(response_length=5, timeout_s=1.0, final_prefix_score=True)

    case = asyncio.run(
        MODULE._generate_segmented(
            object(),
            args,
            prompt_id=0,
            prompt=None,
            prompt_token_ids=[10, 11],
            segment_length=2,
        )
    )

    assert case.logprob_replace == [-20.0, -21.0, -22.0, -23.0, -5.0]
    assert case.logprob_replace_final == [-30.0, -31.0, -32.0, -33.0, -34.0]
    assert case.replacement_source == ["reprefill_2"] * 4 + ["decode"]
    assert case.final_replacement_source == ["final_prefill"] * 5
    assert case.final_prefill == {
        "enabled": True,
        "prefix_length": 7,
        "available_count": 5,
        "available_fraction": 1.0,
    }


def test_prompt_logprob_alignment_rejects_token_mismatch_and_marks_missing():
    values, available = MODULE._response_prompt_records(
        [{"token_id": 10, "logprob": None}, {"token_id": 20, "logprob": -1.5}],
        prompt_length=1,
        response_ids=[20],
    )
    assert values == [-1.5]
    assert available == [True]

    values, available = MODULE._response_prompt_records([], prompt_length=1, response_ids=[20])
    assert values == [None]
    assert available == [False]

    with pytest.raises(RuntimeError, match="token mismatch"):
        MODULE._response_prompt_records(
            [{"token_id": 10, "logprob": None}, {"token_id": 21, "logprob": -1.5}],
            prompt_length=1,
            response_ids=[20],
        )


def test_mode5_geometric_rs_and_offpolicy_metrics_match_rollout_corr_signs():
    trainer = [-1.0, -1.0]
    rollout = [-1.1, -0.9]
    offpolicy = MODULE._offpolicy_statistics(trainer, rollout)

    assert offpolicy["kl"] == pytest.approx(0.0)
    assert offpolicy["log_ppl_diff"] == pytest.approx(0.0)
    assert offpolicy["ppl_ratio"] == pytest.approx(1.0)
    assert offpolicy["chi2_token"] > 0.0
    assert MODULE._offpolicy_statistics([-1.0], [-1.1])["log_ppl_diff"] == pytest.approx(-0.1)

    rejected = MODULE._geometric_rs_statistics(
        trainer=[-1.0, -1.0],
        rollout=[-1.01, -1.0],
        upper_threshold=1.005,
    )
    assert rejected["sequence_rejected"] is True
    assert rejected["token_masked_fraction"] == 1.0
    batch_rs = MODULE._geometric_rs_batch_statistics(
        [([-1.0, -1.0], [-1.01, -1.0])],
        upper_threshold=1.005,
    )
    assert batch_rs["rollout_rs_masked_fraction"] == 1.0
    assert batch_rs["rollout_rs_seq_masked_fraction"] == 1.0

    kept = MODULE._geometric_rs_statistics(
        trainer=[-1.0, -1.0],
        rollout=[-1.001, -1.0],
        upper_threshold=1.005,
    )
    assert kept["sequence_rejected"] is False
    assert kept["token_masked_fraction"] == 0.0


def test_psrl_batch_tensors_use_left_padding_masks_and_explicit_positions():
    cases = [
        SimpleNamespace(
            case_id="short",
            prompt_token_ids=[10, 11],
            response_token_ids=[20, 21],
        ),
        SimpleNamespace(
            case_id="long",
            prompt_token_ids=[12, 13, 14],
            response_token_ids=[22, 23],
        ),
    ]

    tensors = MODULE._build_psrl_batch_tensors(
        cases,
        pad_token_id=0,
        prompt_length=4,
    )

    assert tensors["input_ids"].tolist() == [
        [0, 0, 10, 11, 20, 21],
        [0, 12, 13, 14, 22, 23],
    ]
    assert tensors["responses"].tolist() == [[20, 21], [22, 23]]
    assert tensors["attention_mask"].tolist() == [
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
    ]
    assert tensors["position_ids"].tolist() == [
        [0, 0, 0, 1, 2, 3],
        [0, 0, 1, 2, 3, 4],
    ]


def test_plots_cover_required_outputs_and_allow_empty_boundary_groups(tmp_path):
    metrics = MODULE.mismatch_statistics(
        trainer=[-0.1, -0.2],
        rollout=[-0.11, -0.18],
        outlier_thresholds=[0.05],
        ratio_low=0.5,
        ratio_high=2.0,
    )
    token_rows = [
        {
            "case_id": "segmented-p0-n2",
            "token_position": position,
            "delta_concat": delta,
            "delta_replace": delta / 2,
            "abs_delta_concat": abs(delta),
            "abs_delta_replace": abs(delta / 2),
            "is_segment_boundary": position == 1,
            "boundary_category": category,
        }
        for position, delta, category in (
            (0, 0.01, "far"),
            (1, -0.02, "boundary_0"),
        )
    ]
    aggregate_summary = {
        "2": {
            "concat": metrics,
            "replace": metrics,
        }
    }

    paths = MODULE._plot_results(tmp_path, token_rows, aggregate_summary)

    assert {Path(path).name for path in paths} == {
        "per_token_delta.png",
        "absolute_delta_hist_ecdf.png",
        "concat_replace_percentiles.png",
        "segment_length_comparison.png",
        "boundary_distance.png",
        "hypothetical_rs_masked_fraction.png",
    }
    assert all(Path(path).is_file() for path in paths)


def test_automated_conclusions_distinguish_concat_from_replacement():
    def metrics(rollout):
        return MODULE.mismatch_statistics(
            trainer=[0.0, 0.0],
            rollout=rollout,
            outlier_thresholds=[0.5],
            ratio_low=0.5,
            ratio_high=2.0,
        )

    aggregate_summary = {
        "2": {
            "concat": metrics([1.0, 1.0]),
            "replace": metrics([0.2, 0.2]),
        }
    }
    token_rows = [
        {
            "segment_length": 2,
            "boundary_category": category,
            "abs_delta_concat": 1.0,
            "abs_delta_replace": 0.2,
        }
        for category in ("boundary_0", "far")
    ]

    result = MODULE._build_automated_conclusions(aggregate_summary, token_rows)["2"]

    assert result["replacement_improves_core_metrics"] is True
    assert result["canonicalization_candidate_supported"] is True
    assert "continuous" not in result["hypothetical_masked_fraction"]


def test_argument_defaults_target_qwen_and_plan_matrix():
    args = MODULE._build_parser().parse_args([])

    assert args.model == "models/Qwen2.5-7B"
    assert args.response_length == 1024
    assert args.segment_lengths == [32, 64, 128, 256]
    assert args.temperature == 1.0
    assert args.top_k == -1
    assert args.include_continuous_baseline is False
    assert args.kv_cache_dtype == "auto"
    assert args.enable_prefix_caching is False
    assert args.enable_chunked_prefill is False
    assert args.enforce_eager is True
    assert args.trainer_model is None
    assert args.reprefill_count is None
    assert args.sampling_temperature is None
    assert args.vllm_max_num_seqs == 1
    assert args.vllm_logprobs == 1
    assert args.vllm_prompt_logprobs == 1
    assert args.final_prefix_score is True
    assert args.match_mode5 is False
    assert args.mode5_rs_threshold == pytest.approx(1.005)
    assert args.trainer_model_dtype == "float32"
    assert args.trainer_dtype == "bfloat16"
    assert args.trainer_attention_implementation == "flash_attention_2"
    assert args.trainer_prompt_length == 512
    assert args.trainer_log_prob_micro_batch_size == 1
    assert args.trainer_log_prob_use_dynamic_bsz is False
    assert args.trainer_log_prob_max_token_len == 16384
    assert args.trainer_use_remove_padding is False
    assert args.trainer_use_fused_kernels is False
    assert args.trainer_use_torch_compile is True
    assert args.trainer_entropy_from_logits_with_chunking is False
    assert args.trainer_enable_gradient_checkpointing is True
    assert args.trainer_reshard_after_forward is True
    assert len(MODULE.DEFAULT_PROMPTS) == 4


def test_prepare_args_supports_exact_reprefill_count_and_mode5_preset():
    args = MODULE._build_parser().parse_args(["--response-length", "1024", "--reprefill-count", "8", "--match-mode5"])
    MODULE._prepare_args(args)

    assert args.segment_lengths == [math.ceil(1024 / 9)]
    assert args.vllm_max_num_seqs == 512
    assert args.max_num_batched_tokens == 32768
    assert args.gpu_memory_utilization == pytest.approx(0.7)
    assert args.max_model_len == 32768
    assert args.enable_prefix_caching is True
    assert args.enable_chunked_prefill is True
    assert args.enforce_eager is False
    assert args.trainer_prompt_length == 1024
    assert args.trainer_log_prob_use_dynamic_bsz is True
    assert args.trainer_use_remove_padding is True
    assert args.trainer_use_torch_compile is True
