"""
Compare native checkpoint loading with the production RM sleep and wake flow.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import psrl.workers.gen.vllm_extension as extension_module
import ray
import torch
from omegaconf import OmegaConf
from psrl.utils.common.tms_env import prepend_python_library_path
from psrl.workers.gen.vllm_async_server import GenInterface, PSRL_vLLMHttpServer
from psrl.workers.gen.vllm_extension import vLLMWorkerExtension
from psrl.workers.reward.gen_reward_function.default_gen_rm import DefaultGenRewardFunction
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import AutoTokenizer
from verl.workers.rollout.replica import RolloutMode
from vllm import LLM, SamplingParams

psrl_logger = logging.getLogger(__name__)


class ProbeWorkerExtension(vLLMWorkerExtension):
    """
    Inspect runtime weights and optionally inject a missing postprocessing step.
    """

    def probe_weight_state(self):
        model = self.model_runner.get_model()
        digest = hashlib.sha256()
        for name, param in model.named_parameters():
            flat = param.detach().reshape(-1)
            indices = torch.arange(32, device=flat.device) * (flat.numel() - 1) // 31
            values = flat[indices].float().cpu().numpy()
            digest.update(name.encode())
            digest.update(values.tobytes())
        backends = sorted(
            {
                str(module.quant_method.unquantized_backend)
                for module in model.modules()
                if hasattr(getattr(module, "quant_method", None), "unquantized_backend")
            }
        )
        return {"sampled_weight_sha256": digest.hexdigest(), "moe_backends": backends}

    def load_weights_from_cpu_cache(self, blocking=True):
        if os.environ.get("PSRL_RM_PROBE_OMIT_POSTPROCESSING") != "1":
            return super().load_weights_from_cpu_cache(blocking=blocking)
        original = extension_module.process_weights_after_loading
        extension_module.process_weights_after_loading = lambda *args: None
        try:
            return super().load_weights_from_cpu_cache(blocking=blocking)
        finally:
            extension_module.process_weights_after_loading = original


class ProbeServer(PSRL_vLLMHttpServer):
    """
    Use the production server with worker diagnostics enabled.
    """

    def _get_worker_extension_cls(self):
        return "examples.elastic_analysis.rm_sleep_wake_probe.ProbeWorkerExtension"


def _prompts(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    reward_function = DefaultGenRewardFunction()
    return [
        {
            "expected_score": expected,
            "token_ids": tokenizer.apply_chat_template(
                reward_function.prompt_constructor("What is 1 + 1?", solution),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            ),
        }
        for solution, expected in (("1 + 1 = 2. The answer is 2.", 1.0), ("1 + 1 = 3. The answer is 3.", 0.0))
    ]


def _output_summary(output, expected_score):
    completion = output.outputs[0]
    score = DefaultGenRewardFunction().compute_score("probe", "", completion.text)
    return {
        "text": completion.text,
        "token_ids": list(completion.token_ids),
        "output_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "score": score,
        "expected_score": expected_score,
    }


class ProbeActor:
    """
    Own one four GPU inference engine and release it after the experiment.
    """

    async def run(self, config_path, output_dir, mode, max_tokens, cycles):
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        rm_config = config.reward.managers.gen_qwen3_8b
        model_path = rm_config.model.path
        prompts = _prompts(model_path)
        sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, seed=0)
        report = {"mode": mode, "model": Path(model_path).name, "stages": {}}
        report_path = Path(output_dir) / f"{mode}.json"
        server = None
        native = None
        engine = None
        start = time.monotonic()
        try:
            if mode == "reference":
                kwargs = OmegaConf.to_container(rm_config.rollout.engine_kwargs.vllm, resolve=True)
                native = LLM(
                    model=model_path,
                    load_format="auto",
                    dtype=rm_config.rollout.dtype,
                    tensor_parallel_size=rm_config.rollout.tensor_model_parallel_size,
                    enable_expert_parallel=rm_config.rollout.expert_parallel_size > 1,
                    enforce_eager=True,
                    gpu_memory_utilization=rm_config.rollout.gpu_memory_utilization,
                    max_model_len=rm_config.rollout.max_model_len,
                    max_num_batched_tokens=rm_config.rollout.max_num_batched_tokens,
                    max_num_seqs=rm_config.rollout.max_num_seqs,
                    enable_chunked_prefill=rm_config.rollout.enable_chunked_prefill,
                    enable_prefix_caching=False,
                    worker_extension_cls="examples.elastic_analysis.rm_sleep_wake_probe.ProbeWorkerExtension",
                    **kwargs,
                )
                outputs = native.generate(
                    [{"prompt_token_ids": prompt["token_ids"]} for prompt in prompts],
                    sampling,
                    use_tqdm=False,
                )
                report["stages"]["baseline"] = {
                    "weights": native.collective_rpc("probe_weight_state"),
                    "outputs": [
                        _output_summary(output, prompt["expected_score"])
                        for output, prompt in zip(outputs, prompts, strict=True)
                    ],
                }
            else:
                config.psrl.logging_path = output_dir
                config.psrl.status_collection.enable = False
                config.psrl.lmcache.enable = False
                server = ProbeServer(
                    psrl_config=config.psrl,
                    config=rm_config.rollout,
                    model_config=rm_config.model,
                    rollout_mode=RolloutMode.STANDALONE,
                    workers=[],
                    replica_rank=0,
                    node_rank=0,
                    gpus_per_node=4,
                    nnodes=1,
                    cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                    gen_interface=GenInterface(role="reward_probe", rollout_replica_idx=0),
                )
                await server.launch_server()
                engine = server.engine
                # Elastic deployment initializes dummy GPU weights before the first level 2 sleep.
                for cycle in range(cycles + 1):
                    await server.sleep(level=2)
                    await server.wake_up()
                    outputs = []
                    for index, prompt in enumerate(prompts):
                        final = None
                        async for output in engine.generate(
                            {"prompt_token_ids": prompt["token_ids"]},
                            sampling,
                            f"probe-{mode}-{cycle}-{index}",
                        ):
                            final = output
                        assert final is not None, "Generation must return an output."
                        outputs.append(_output_summary(final, prompt["expected_score"]))
                    stage = f"wake_{cycle}"
                    report["stages"][stage] = {
                        "weights": await server.collective_rpc("probe_weight_state"),
                        "outputs": outputs,
                    }
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    psrl_logger.info(f"Completed {mode!r} stage {stage!r}.")
            report["elapsed_s"] = time.monotonic() - start
        except Exception as exc:
            report["error"] = repr(exc)
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if server is not None and hasattr(server, "_grpc_server"):
                await server._grpc_server.stop(grace=0)
            if engine is not None:
                engine.shutdown()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("reference", "fixed", "missing_postprocessing"), required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ray.init(address="auto", namespace="rm_sleep_wake_probe", log_to_driver=True)
    env = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_CUMEM_ENABLE": "0",
        "VLLM_SKIP_P2P_CHECK": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "PSRL_LOGGING_LEVEL": "INFO",
        "PSRL_RM_PROBE_OMIT_POSTPROCESSING": "1" if args.mode == "missing_postprocessing" else "0",
    }
    if args.mode != "reference":
        import torch_memory_saver

        hook = Path(torch_memory_saver.__file__).parent.parent / "torch_memory_saver_hook_mode_preload.abi3.so"
        env.update(
            {
                "LD_PRELOAD": str(hook),
                "LD_LIBRARY_PATH": prepend_python_library_path(),
                "PSRL_VLLM_PATCHES": "TMS",
                "TMS_INIT_ENABLE": "0",
                "TMS_INIT_ENABLE_CPU_BACKUP": "0",
            }
        )
    else:
        env.update({"PSRL_VLLM_PATCHES": "", "TMS_INIT_ENABLE": "0", "TMS_INIT_ENABLE_CPU_BACKUP": "0"})
    actor = None
    try:
        actor = (
            ray.remote(ProbeActor)
            .options(
                num_gpus=4,
                num_cpus=4,
                runtime_env={"env_vars": env},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote()
        )
        report = ray.get(
            actor.run.remote(
                str(Path(args.config).resolve()), str(output_dir), args.mode, args.max_tokens, args.cycles
            ),
            timeout=1800,
        )
        psrl_logger.info(
            f"Probe completed in {report['elapsed_s']:.1f}s. Report: {output_dir / (args.mode + '.json')}."
        )
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
