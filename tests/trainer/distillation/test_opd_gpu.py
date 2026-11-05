"""Opt-in SMG and vLLM teacher scoring followed by a CUDA student update."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import aiohttp
import pytest
import torch
from omegaconf import OmegaConf
from psrl.trainer.distillation.data import align_teacher_targets
from psrl.workers.config import DistillationConfig, DistillationLossConfig
from psrl.workers.gen.smg_adapter import build_worker_registration_payload
from psrl.workers.reward.reward_loop.opd import OPDRewardLoop
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.utils.padding import no_padding_2_padding

import grpc

pytestmark = pytest.mark.gpu


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def register_teacher(url, processes, teacher_port, model_path, tp_size):
    deadline = time.monotonic() + 300
    worker_id = None
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{teacher_port}")
    stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
    async with channel, aiohttp.ClientSession() as client:
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("SMG or vLLM exited during startup. Check the smoke test logs.")
            try:
                teacher_health = await stub.HealthCheck(vllm_engine_pb2.HealthCheckRequest(), timeout=1.0)
                async with client.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=1)) as response:
                    ready = response.status == 200 and teacher_health.healthy
            except (aiohttp.ClientError, grpc.aio.AioRpcError, TimeoutError):
                await asyncio.sleep(1)
                continue
            if ready and worker_id is None:
                payload = build_worker_registration_payload(
                    url=f"grpc://127.0.0.1:{teacher_port}",
                    model_id=model_path,
                    role="reward_model_teacher",
                    fuse_rollout_with_validate=False,
                    max_model_len=128,
                    dp_size=1,
                    tp_size=tp_size,
                    pp_size=1,
                )
                async with client.post(f"{url}/workers", json=payload) as response:
                    response.raise_for_status()
                    worker_id = (await response.json())["worker_id"]
            if worker_id is not None:
                async with client.get(f"{url}/workers/{worker_id}") as response:
                    response.raise_for_status()
                    if (await response.json())["is_healthy"]:
                        return
            await asyncio.sleep(1)
    raise TimeoutError("SMG teacher did not become healthy within 300 seconds.")


@pytest.fixture(scope="module")
def native_teacher(tmp_path_factory, gpu_model_path, gpu_tensor_parallel_size):
    tmp_path = tmp_path_factory.mktemp("opd_native_teacher")
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpu_tensor_parallel_size:
        pytest.fail("GPU smoke requires CUDA devices for the requested teacher tensor parallel size.")
    teacher_port, gateway_port, metrics_port = free_port(), free_port(), free_port()
    url = f"http://127.0.0.1:{gateway_port}"
    commands = [
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.grpc_server",
            "--model",
            gpu_model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(teacher_port),
            "--tensor-parallel-size",
            str(gpu_tensor_parallel_size),
            "--gpu-memory-utilization",
            "0.3",
            "--max-model-len",
            "128",
            "--max-logprobs",
            "8",
            "--enforce-eager",
        ],
        [
            sys.executable,
            "-m",
            "smg.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(gateway_port),
            "--prometheus-port",
            str(metrics_port),
            "--connection-mode",
            "grpc",
            "--model-path",
            gpu_model_path,
            "--policy",
            "round_robin",
        ],
    ]
    processes, logs = [], []
    try:
        for index, command in enumerate(commands):
            log = (tmp_path / f"service-{index}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        asyncio.run(register_teacher(url, processes, teacher_port, gpu_model_path, gpu_tensor_parallel_size))
        yield url
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        for log in logs:
            log.close()


@pytest.mark.parametrize("mode", ["k1", "forward_kl_topk", "forward_kl_topk_tail"])
async def test_smg_vllm_student_update(native_teacher, gpu_model_path, mode):
    topk = mode != "k1"
    url = native_teacher
    tokenizer = AutoTokenizer.from_pretrained(gpu_model_path)
    student = AutoModelForCausalLM.from_pretrained(gpu_model_path, dtype=torch.float32).cuda()
    student.eval()
    prompt = tokenizer.encode("The sum of 2 and 3 is", add_special_tokens=True)
    with torch.no_grad():
        prompt_tensor = torch.tensor([prompt], device="cuda")
        generated = student.generate(
            prompt_tensor,
            attention_mask=torch.ones_like(prompt_tensor),
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=2,
            do_sample=True,
        )
    response = generated[0, len(prompt) :].tolist()
    cfg = OmegaConf.create(
        {
            "distillation": {
                "distillation_loss": {
                    "loss_mode": mode,
                    "topk": 8,
                }
            }
        }
    )
    runtime = RewardModelRuntimeInfo(
        url,
        tokenizer,
        OmegaConf.create(
            {
                "reward_model_name": gpu_model_path,
                "rollout": {"runner": "generate", "max_model_len": 128},
            }
        ),
    )
    loop = OPDRewardLoop(cfg, tokenizer, runtime)
    scored = await loop.run_single(
        TensorDict(
            {
                "prompts": torch.tensor([prompt]),
                "responses": torch.tensor([response]),
                "attention_mask": torch.ones(1, len(prompt) + len(response), dtype=torch.int64),
            },
            batch_size=[1],
        )
    )
    row = {
        "prompts": torch.tensor(prompt),
        "responses": torch.tensor(response),
        "response_mask": torch.ones(len(response)),
        "attention_mask": torch.ones(len(prompt) + len(response), dtype=torch.int64),
        "teacher_weight": torch.tensor(1.0),
        **{key: value for key, value in scored.items() if key.startswith("teacher_")},
    }
    data = align_teacher_targets(list_of_dict_to_tensordict([row]), 8 if topk else 1, topk).cuda()
    input_ids = torch.tensor([prompt + response], device="cuda")
    logits = student(input_ids).logits
    labels = input_ids.roll(-1, dims=1)
    sampled_logprobs = logits.detach().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze()
    # A small trainable bias provides a nonzero signal when both checkpoints are identical.
    bias = torch.nn.Parameter(torch.zeros(logits.shape[-1], device="cuda"))
    selected = scored["teacher_ids"].unique().cuda() if topk else torch.tensor(response, device="cuda")
    with torch.no_grad():
        bias[selected] = -0.1
    logits = logits + bias
    labels = input_ids.roll(-1, dims=1)
    output = {"log_probs": logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze()}
    data["old_log_probs"] = torch.nested.as_nested_tensor(
        [no_padding_2_padding(sampled_logprobs, data)[0]],
        layout=torch.jagged,
    )
    actor = SimpleNamespace(strategy="fsdp2", loss_agg_mode="token-mean", global_batch_info={})
    distill = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            loss_mode=mode,
            topk=8,
            use_policy_gradient=not topk,
            use_task_rewards=False,
            log_prob_min_clamp=None,
        ),
    )
    if topk:
        output.update(
            {
                key: value.squeeze(0)
                for key, value in distillation_ppo_loss(
                    actor,
                    distill,
                    data=data,
                    student_logits=logits,
                ).items()
            }
        )
    optimizer = torch.optim.SGD([*student.parameters(), bias], lr=1e-3)
    loss, _ = distillation_ppo_loss(actor, distill, model_output=output, data=data)
    assert torch.isfinite(loss), "GPU distillation loss must be finite."
    loss.backward()
    assert bias.grad is not None and torch.isfinite(bias.grad).all(), "GPU gradients must be finite."
    head = student.get_output_embeddings().weight
    assert head.grad is not None and torch.isfinite(head.grad).all(), "Student model gradients must be finite."
    head_before = head[selected[0]].detach().clone()
    before = bias.detach().clone()
    optimizer.step()
    assert not torch.equal(head_before, head[selected[0]]), "The student model weights must change."
    assert not torch.equal(before, bias), "A real teacher score must produce a student update."
