# Chunked Prefill Micro-Benchmark 使用手册

> **用途**：验证 RL rollout 场景下 chunked prefill 的性价比，找到 GEMM 饱和点 M\*，量化混合 batch 代价，测 activation 预留对 KV 容量的挤占。
> **代码位置**：`psrl/bench/chunked_prefill/`
> **Shell 脚本**：`examples/bench/rollout/run_chunked_prefill_*.sh`

---

## 快速上手

```bash
# 激活环境
source ${PSRL_WORKSPACE}/env/psrl.sh

# 最小冒烟测试（E1，3 个配置点，约 2 分钟）
python -m psrl.bench.chunked_prefill.main_micro \
    model.path=${PSRL_WORKSPACE}/models/SWE-agent-LM-7B \
    micro.experiment=e1 \
    micro.e1.m_values=[512,2048,8192] \
    micro.warmup=2 micro.iters=5 \
    rollout.max_num_batched_tokens=8192

# 完整 sweep（TP=1 和 TP=4，全部实验）
bash examples/bench/rollout/run_chunked_prefill_exp.sh \
    ${PSRL_WORKSPACE}/models/SWE-agent-LM-7B
```

结果写到 `./results/micro_TP1_N65536_<timestamp>.jsonl`，用 `plot.py` 出图。

---

## 四个实验说明

### `e1` — 找 GEMM 饱和点 M\*

**问题**：step 内 token 总数 M 越大，GEMM 效率（MFU）越高，但收益是饱和的。M\* 是"再加 token 也不再提升 MFU"的拐点。**M\* 是整个研究方向的判决条件**：如果 M\* 很小（比如 4096），把 `max_num_batched_tokens` 从 8192 抬到 65536 只是亏显存，H2/H3 就没价值了。

**怎么跑**：在一个大 budget 的 engine 里，构造"纯 prefill"的合成 step，扫 M（step 内 query token 总数）。每个 M 默认跑两种拆法：
- `single`：1 个请求，q\_len = M（验证单请求 prefill 效率）
- `multi`：多个小请求，q\_len 之和 = M（验证 GEMM 只看 M，不看每个请求的 q\_len）

如果 `single` 和 `multi` 的 MFU 接近，说明 GEMM 效率确实由 M 决定，测量可信。

**输出**：`throughput(M)` 和 `MFU(M)` 曲线，结果落 `e1_results.csv`。

**关键参数**：

```bash
# 扫哪些 M 值（token 数）
micro.e1.m_values=[512,1024,2048,4096,8192,16384,32768,65536]

# 是否同时测单请求和多请求两种拆法（默认 true）
micro.e1.test_decompositions=true

# engine 要能容纳最大的 M，所以 budget 要 >= max(m_values)
rollout.max_num_batched_tokens=65536
```

**注意**：用 `e1` 时，engine 需要用较大的 `max_num_batched_tokens` 启动（至少等于 `m_values` 的最大值），否则大 M 的配置点会被跳过（`skipped=true`）。

---

### `e2` — 混合 batch 的额外代价

**问题**：同一个 step 里既有 decode（q=1）又有 prefill（q=C），比"纯 decode"和"纯 prefill 分开跑"更慢多少？额外代价来自 varlen attention kernel 内部 query length 分布不均匀（不是两个 kernel 互相抢占）。

**怎么跑**：对每个 `(D, C, L)` 三元组（D 个 decode、prefill chunk 大小 C、context 长度 L），分三次 probe：
1. `mixed`：D 个 decode 请求 + 1 个 prefill 请求
2. `decode_only`：D 个 decode 请求（没有 prefill）
3. `prefill_only`：1 个 prefill 请求（没有 decode）

然后算 `overhead = t_mixed / (t_decode_only + t_prefill_only)`。overhead > 1 说明混在一起比分开跑更慢（"打架"），overhead ≈ 1 说明没有显著干扰。

**输出**：overhead 热力图（x 轴 = prefill chunk C，y 轴 = decode 数 D），每个 context 长度 L 一张子图。结果落 `e2_results.csv`。

**关键参数**：

```bash
# 扫多少个 decode 请求（包含 0，表示纯 prefill 基线）
micro.e2.decode_counts=[0,4,8,16,32,64]

# prefill chunk 大小（单个 prefill 请求的 q_len）
micro.e2.prefill_chunks=[256,512,1024,2048,4096]

# decode 请求的 context 长度（已有多少 KV cache）
micro.e2.context_lens=[512,2048,8192]
```

**注意**：配置点数量 = `|decode_counts| × |prefill_chunks| × |context_lens| × 3`（三种 variant），默认配置约 270 次 probe，耗时较长。做快速验证时建议缩减：

```bash
micro.e2.decode_counts=[0,8,32]
micro.e2.prefill_chunks=[512,2048]
micro.e2.context_lens=[2048]
```

---

### `e3a` — 每步 activation 内存增量

**问题**：一个 forward pass 跑完，activation 峰值比 forward 前多了多少？这是**实际运行时的瞬时 activation**，不是启动时预留的静态值。

**怎么跑**：复用 E1 的 M sweep，在每次 `probe_step` 里加 `torch.cuda.reset_peak_memory_stats()` → forward → `max_memory_allocated() - allocated_before`，取增量。

**输出**：`activation(M)` 曲线（GiB），结果嵌在主 JSONL 里，画图时出现在 E3 图的左轴。

**关键参数**：
```bash
micro.e3a.m_values=[512,1024,2048,4096,8192,16384,32768,65536]
```

**注意**：e3a 测的是"每步的 activation 增量"，不是启动时预留的值。要测"不同 N 下预留多少 activation"需要用 `e3b`（见下）。

---

### `e3b` — activation 预留 vs KV 容量（独立进程）

**问题**：vLLM 在启动时，会用 `max_num_batched_tokens` 跑一次 profile run，然后把测到的 peak activation 永久预留。这个预留值随 N 增大而增大，直接压缩 KV 容量。**M\* 的代价在这里**。

**为什么必须独立进程**：`peak_activation` 在启动时测一次就固定了，同一个 engine 实例里无法观测不同 N 的预留值。必须每个 N 起一个新进程，读完 breakdown 就退出。

**怎么跑**：`memory_probe.py` 是专门为此设计的单点探针。用 shell 脚本循环调：

```bash
# 自动循环所有默认 N 值
bash examples/bench/rollout/run_chunked_prefill_memory_sweep.sh \
    1 ${PSRL_WORKSPACE}/models/SWE-agent-LM-7B 0.90

# 也可以手动单点
python -m psrl.bench.chunked_prefill.memory_probe \
    --model ${PSRL_WORKSPACE}/models/SWE-agent-LM-7B \
    --max-batched-tokens 8192 \
    --tp 1 --gpu-util 0.90 \
    --output results/e3b/e3b_N8192.json
```

**输出**：每个 N 一个 JSON 文件，包含：
- `peak_activation_bytes`：启动时预留的 activation
- `available_kv_bytes`：剩余给 KV 的字节数
- `kv_token_capacity`：最多能存多少 KV token

画图时 `e3_activation_kv.png` 左轴画 activation 预留（GiB），右轴画 KV token 容量（k tokens），可以直观看到"N 抬高一倍，KV 损失多少"。

**关键参数**（命令行参数，不是 Hydra）：
```
--model          HuggingFace 模型路径（必填）
--max-batched-tokens  要探测的 N 值（必填）
--tp             tensor parallel size（默认 1）
--gpu-util       gpu_memory_utilization（默认 0.90）
--max-model-len  最大序列长度（默认 32768）
--output         输出 JSON 路径（省略则打印到 stdout）
```

---

## 完整参数速查

所有默认值在 `config/chunked_prefill_micro.yaml`，可通过 Hydra CLI 覆盖。

| 参数 | 默认 | 说明 |
|---|---|---|
| `micro.experiment` | `all` | 要跑哪些实验：`e1` / `e2` / `e3a` / `all` |
| `micro.warmup` | `3` | 每个配置点的热身次数（不计入统计） |
| `micro.iters` | `10` | 每个配置点的正式测量次数（报 median/p10/p90） |
| `micro.peak_tflops` | `148.0` | GPU bf16 峰值算力（H20=148，H100=989，A100=312），用于算 MFU |
| `micro.output_dir` | `./results` | JSONL 结果目录 |
| `rollout.max_num_batched_tokens` | `65536` | engine 的 token budget，必须 >= E1/E3a 的最大 M |
| `rollout.tensor_parallel_size` | `1` | TP 大小 |
| `rollout.gpu_memory_utilization` | `0.90` | 显存利用率上限 |
| `rollout.max_model_len` | `32768` | 最大序列长度 |

---

## 画图

```bash
# 读 results/ 目录出三张图（需要先跑 E1/E2/E3a）
python -m psrl.bench.chunked_prefill.plot \
    --results-dir ./results \
    --out ./plots

# 同时包含 E3b 内存探针结果
python -m psrl.bench.chunked_prefill.plot \
    --results-dir ./results \
    --e3b-dir ./results/e3b \
    --out ./plots

# 只画 E1
python -m psrl.bench.chunked_prefill.plot \
    --results-dir ./results \
    --experiments e1 \
    --out ./plots
```

**输出文件**：
- `e1_mfu.png` — throughput(M) 和 MFU(M) 曲线（含 M\* 参考线）
- `e2_overhead.png` — 混合 batch overhead 热力图
- `e3_activation_kv.png` — activation 预留 + KV 容量双轴图
- `e1_results.csv` / `e2_results.csv` / `e3b_results.csv` — 原始数据

---

## 典型实验序列

### 第一次运行（验证 harness 可用 + 快速判断 M\*）

```bash
# 步骤 1：冒烟 E1（5 分钟内出结果）
python -m psrl.bench.chunked_prefill.main_micro \
    model.path=${PSRL_WORKSPACE}/models/SWE-agent-LM-7B \
    micro.experiment=e1 \
    micro.e1.m_values=[512,1024,2048,4096,8192] \
    micro.warmup=2 micro.iters=5 \
    rollout.max_num_batched_tokens=8192

# 步骤 2：确认结果合理（两列 MFU 接近 → 测量可信）
python -m psrl.bench.chunked_prefill.plot --results-dir ./results --experiments e1 --out ./plots
```

### 完整基线（TP=1 和 TP=4 全部实验）

```bash
bash examples/bench/rollout/run_chunked_prefill_exp.sh \
    ${PSRL_WORKSPACE}/models/SWE-agent-LM-7B 65536 0.90
```

运行完后：
```bash
python -m psrl.bench.chunked_prefill.plot \
    --results-dir examples/bench/rollout/exp/chunked_prefill \
    --e3b-dir examples/bench/rollout/exp/chunked_prefill/e3b \
    --out examples/bench/rollout/exp/chunked_prefill/plots
```

---

## 结果文件格式

每个 JSONL 行是一个 JSON 对象，包含：

```jsonc
{
  "run_id": "a1b2c3d4",         // 8 位运行 ID，同一次运行所有行相同
  "timestamp": "2026-08-02T...",
  "gpu_name": "NVIDIA H20",
  "vllm_version": "0.22.1.dev0+...",
  "tensor_parallel_size": 1,
  "max_num_batched_tokens": 65536,
  "experiment": "e1",            // e1 / e2 / e3a / setup
  // 实验特定字段（e1 有 m、decomposition；e2 有 decode_count 等）
  "m": 4096,
  "decomposition": "single",
  // 测量结果
  "latency_ms_median": 12.3,
  "latency_ms_p10": 11.8,
  "latency_ms_p90": 13.1,
  "throughput_tok_per_s": 332764,
  "mfu": 0.43,
  "activation_bytes_median": 1234567890,
  "skipped": false              // true 时有 skip_reason 字段说明原因
}
```

第一行 `"experiment": "setup"` 是显存账快照，包含 `memory_breakdown_per_rank`，不参与画图。

---

## 常见问题

**Q：配置点被跳过，日志里出现 `skipped=true`**

检查 `skip_reason`：
- `total_q_tokens ... exceeds max_num_batched_tokens`：`rollout.max_num_batched_tokens` 设得比要测的 M 小，调大 budget 或减小 `m_values` 的最大值。
- `num_reqs ... exceeds max_num_seqs`：E2 的 decode 数太大，调大 `rollout.max_num_seqs` 或减小 `decode_counts`。
- `Requested N blocks but only M usable`：KV pool 装不下请求的 context，调大 `gpu_memory_utilization` 或减小 `kv_lens`。

**Q：MFU 全是 null**

需要 `VLLM_DEBUG_MFU_METRICS=1`，或者确认 `micro.peak_tflops > 0`。JSONL 里有 `flops` 字段，若为 0 说明 `ModelMetrics` 初始化失败（通常是模型架构不支持），MFU 无法计算，但 latency 和 throughput 仍然有效。

**Q：连续 probe 结果不稳定（每次差很多）**

- 确认 `micro.warmup >= 3`，前几次有 cudagraph capture 或内存搬运。
- 检查是否有其他进程在同卡抢占，用 `nvidia-smi` 观察。
- 极端情况（M 很大，接近 KV pool 上限）可能触发抢占，导致延迟突增。

**Q：E3b 每个 N 都要启动一次 engine，能不能在一个进程里做完？**

不行。`peak_activation` 在 engine 启动时的 profile run 里测一次就固定了，整个 engine 生命周期内不变。要观测"N=8192 时预留多少 vs N=65536 时预留多少"，必须是两个独立的进程，这是 E3b 的根本设计约束。

**Q：TP=4 时 `probe_step` 的结果怎么理解？**

`collective_rpc` 返回 per-rank 的 list，`main_micro.py` 取 **max latency**（step 的墙钟时间由最慢的 rank 决定），activation 和 FLOPs 是 rank 0 的值（TP 下每个 rank 持有一部分权重，rank 0 的值代表单卡开销，整体 activation 是 TP 倍）。JSONL 的 `per_rank` 字段保存了所有 rank 的原始数据。
