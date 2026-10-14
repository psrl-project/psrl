# 单 Step Latency 建模

> 本文给出 vLLM V1 engine 在一个 forward step 中的延迟解析模型。
> 输入是 batch 的精确几何描述，输出是 latency 的 roofline 估计及各项的量级分析。

---

## 1. 符号定义

### 1.1 Batch 描述

一个 step 内有两类请求：

**Decode 请求**（共 $n$ 个）：
- 每个请求 $i$ 本步只算 **1 个新 token**，已有 $d_i$ 个 token 的 KV cache
- query length = 1，KV length = $d_i + 1 \approx d_i$

**Prefill 请求**（共 $m$ 个）：
- 每个请求 $j$ 有前缀 $p_j$ 个 token 已被 prefix cache 命中（不需要算）
- 本步实际需要 compute 的新 token 数为 $c_j$（即 chunk 大小）
- KV length = $p_j + c_j$

定义几个汇总量：

$$
M_d = n \quad \text{（decode 总 query tokens，每请求 1 个）}
$$

$$
M_p = \sum_{j=1}^{m} c_j \quad \text{（prefill 总 query tokens）}
$$

$$
M = M_d + M_p \quad \text{（step 内总 query tokens，即 token budget 消耗量）}
$$

$$
\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i \quad \text{（decode 平均 context 长度）}
$$

$$
\bar{p} = \frac{1}{m}\sum_{j=1}^m p_j \quad \text{（prefill 平均已命中长度）}
$$

$$
\bar{c} = \frac{M_p}{m} \quad \text{（prefill 平均 chunk 大小）}
$$

### 1.2 模型参数

| 符号 | 含义 | Qwen2.5-7B 示例值 |
|---|---|---|
| $L$ | 层数 | 28 |
| $H$ | hidden size | 3584 |
| $h_q$ | query head 数 | 28 |
| $h_{kv}$ | KV head 数（GQA） | 4 |
| $d_h$ | head dim = $H / h_q$ | 128 |
| $H_{ff}$ | FFN intermediate size | 18944 |
| $\text{TP}$ | tensor parallel size | 1 / 4 |
| $V$ | vocab size | 152064 |

在 TP 下，各分片后的实际矩阵宽度：
$$
h_q^{tp} = h_q / \text{TP}, \quad h_{kv}^{tp} = h_{kv} / \text{TP}
$$
$$
H_{ff}^{tp} = H_{ff} / \text{TP}
$$

### 1.3 硬件参数

| 符号 | 含义 | H20 示例值 |
|---|---|---|
| $\text{BW}$ | HBM 带宽（GB/s） | 4000 |
| $F$ | 峰值算力（bf16 TFLOPS） | 148 |
| $I^* = F / \text{BW}$ | **roofline ridge point**（FLOP/Byte） | 37 |

**Roofline**：某个 kernel 的实际算术强度 $I = \text{FLOPs} / \text{Bytes}$

$$
T_{\text{kernel}} = \max\!\left(\frac{\text{FLOPs}}{F},\ \frac{\text{Bytes}}{\text{BW}}\right)
$$

当 $I < I^*$ 时 memory-bound（访存是瓶颈），当 $I > I^*$ 时 compute-bound。

---

## 2. 矩阵乘（Linear 层）分析

每层 Transformer 有这些 Linear 投影：

| 操作 | 权重矩阵 shape（TP 后每卡） | FLOPs | 读权重 Bytes |
|---|---|---|---|
| QKV 投影 | $H \times (h_q^{tp} + 2h_{kv}^{tp}) \cdot d_h$ | $2M \cdot H \cdot (h_q^{tp} + 2h_{kv}^{tp}) d_h$ | $H \cdot (h_q^{tp} + 2h_{kv}^{tp}) d_h \cdot 2$ |
| O 投影 | $h_q^{tp} d_h \times H$ | $2M \cdot h_q^{tp} d_h \cdot H$ | $h_q^{tp} d_h \cdot H \cdot 2$ |
| FFN gate/up | $H \times H_{ff}^{tp}$（×2） | $2 \cdot 2M \cdot H \cdot H_{ff}^{tp}$ | $2 \cdot H \cdot H_{ff}^{tp} \cdot 2$ |
| FFN down | $H_{ff}^{tp} \times H$ | $2M \cdot H_{ff}^{tp} \cdot H$ | $H_{ff}^{tp} \cdot H \cdot 2$ |

单层合计（factor 2 for bf16）：
$$
\text{FLOP}_{\text{linear}} = 2M \cdot H \cdot \left[(h_q^{tp} + 2h_{kv}^{tp})d_h + h_q^{tp}d_h + 3H_{ff}^{tp}\right]
$$

由于 $h_q^{tp} d_h = H / \text{TP}$（Q 投影等效为 $H \to H/\text{TP}$）：

$$
\text{FLOP}_{\text{linear}} \approx 2M \cdot \frac{1}{\text{TP}} \cdot H \cdot \left[H + 2h_{kv}d_h + H + 3H_{ff}\right]
$$

通常 $H_{ff} \approx \frac{8}{3}H$ 或 $5.3H$（SwiGLU），$2H + 3H_{ff} \approx 18H$，所以粗估：

$$
\text{FLOP}_{\text{linear}} \approx \frac{36 M H^2}{\text{TP}} \quad (\text{单层})
$$

**读权重字节数**（bf16，全精度权重常驻 HBM）：
$$
\text{Bytes}_{\text{weight}} \approx \frac{14 H^2}{\text{TP}} \times 2 = \frac{28 H^2}{\text{TP}} \quad \text{Bytes/层}
$$

> 实际系数取决于 $H_{ff}/H$ 的比值，这里用近似估计；精确数字从模型 config 算。

**Linear 层的算术强度**：

$$
I_{\text{linear}} = \frac{\text{FLOP}}{\text{Bytes}} = \frac{36 M H^2 / \text{TP}}{28 H^2 / \text{TP}} \approx \frac{9M}{7} \approx 1.3 M
$$

**关键结论**：

- $M = 1$（纯 decode，1 个请求）：$I \approx 1.3$，远小于 $I^* = 37$，**深度 memory-bound**
- $M = 28$（$n = 28$ 个 decode）：$I \approx 37$，恰好在 ridge point
- $M > 37$：compute-bound

所以对 **decode**（$M = M_d = n$），ridge point 约在 $n \approx I^* \approx 37$ 个请求处。

$$
\boxed{T_{\text{linear}}^{\text{(per layer)}} = \max\!\left(\frac{36 M H^2 / \text{TP}}{F},\ \frac{28 H^2 / \text{TP}}{\text{BW}}\right)}
$$

---

## 3. Attention 分析

Attention 是 vLLM V1 里最复杂的部分：所有请求被拼成一条 flat token 序列，用一次 **varlen FlashAttention** 跑完（不是分 decode/prefill 两次调用）。

### 3.1 FLOPs

Attention 的 FLOPs = $4 \times q\_len \times kv\_len \times h_{kv}^{tp} \times d_h$（QK + softmax + SV，factor 2 for multiply-add）。

各请求分别计算：

**Decode 请求 $i$（$q = 1$，$kv = d_i$）**：
$$
\text{FLOP}_{d,i} = 4 \cdot 1 \cdot d_i \cdot h_{kv}^{tp} d_h
$$

**Prefill 请求 $j$（$q = c_j$，$kv = p_j + c_j$）**：
$$
\text{FLOP}_{p,j} = 4 \cdot c_j \cdot (p_j + c_j) \cdot h_{kv}^{tp} d_h
$$

Step 总 attention FLOPs：

$$
\text{FLOP}_{\text{attn}} = 4 h_{kv}^{tp} d_h \left[\sum_{i=1}^n d_i + \sum_{j=1}^m c_j(p_j + c_j)\right]
$$

$$
= 4 h_{kv}^{tp} d_h \left[n\bar{d} + \sum_{j=1}^m c_j(p_j + c_j)\right]
$$

### 3.2 访存字节数

Attention 的访存瓶颈在于**从 HBM 读 KV cache**：

- 每个 KV token：$2 \times h_{kv}^{tp} \times d_h \times 2$ 字节（K + V，bf16）

**Decode 请求 $i$**：读 $d_i$ 个 KV 位置：
$$
\text{Bytes}_{d,i} = 4 h_{kv}^{tp} d_h \cdot d_i
$$

**Prefill 请求 $j$**：FlashAttention 可以做 online softmax，但仍需读 $p_j + c_j$ 个 KV 位置（已命中的 prefix KV 也需要从 HBM 读进来参与注意力计算）：
$$
\text{Bytes}_{p,j} = 4 h_{kv}^{tp} d_h \cdot (p_j + c_j)
$$

> **注意**：prefix cache 命中只省去了 KV 写入（不需要重新 compute），但 attention 计算时仍需从 HBM **读出**这些 KV block 参与 softmax。这是 prefill 请求里 $p_j$ 不为零时仍然有显存读开销的原因。

Step 总 attention 访存：

$$
\text{Bytes}_{\text{attn}} = 4 h_{kv}^{tp} d_h \left[n\bar{d} + \sum_{j=1}^m (p_j + c_j)\right]
$$

$$
= 4 h_{kv}^{tp} d_h \left[n\bar{d} + m(\bar{p} + \bar{c})\right]
$$

### 3.3 Attention 的算术强度

$$
I_{\text{attn}} = \frac{\text{FLOP}_{\text{attn}}}{\text{Bytes}_{\text{attn}}} = \frac{n\bar{d} + \sum_j c_j(p_j+c_j)}{n\bar{d} + m(\bar{p}+\bar{c})}
$$

**两个极限情况**：

**纯 decode**（$m=0$）：
$$
I_{\text{attn}}^{\text{decode}} = \frac{\sum_i d_i}{\sum_i d_i} = 1 \quad \text{FLOP/Byte}
$$
远小于 $I^*=37$，**始终 memory-bound**。这是 decode 慢的根本原因：每读 1 Byte KV 只做 1 FLOP。

**纯 prefill**（$n=0$，$p_j=0$，即无前缀缓存）：
$$
I_{\text{attn}}^{\text{prefill}} = \frac{\sum_j c_j^2}{\sum_j c_j} = \bar{c} \quad \text{（各请求 chunk 大小一致时）}
$$

所以 prefill attention 的强度 ≈ **平均 chunk 大小**（token 数）。

- $\bar{c} = 256$：memory-bound（$I < I^* = 37$）——这就是"小 chunk 吃不满"
- $\bar{c} = 2048$：compute-bound（$I = 2048 \gg 37$）——大 prefill 是 compute-bound

**混合情况**：decode 把分子拉低（decode 的 $q=1$ 使 $c_j(p_j+c_j) \to d_i$），而分母几乎不变，所以混合 batch 中 decode 请求会**拉低整体算术强度**，使原本 compute-bound 的 prefill kernel 退化。

$$
\boxed{T_{\text{attn}}^{\text{(per layer)}} = \max\!\left(\frac{\text{FLOP}_{\text{attn}}}{F},\ \frac{\text{Bytes}_{\text{attn}}}{\text{BW}}\right)}
$$

---

## 4. 全 Step Latency 模型

总延迟约为 $L$ 层的累加（各层独立，忽略层间 pipeline overlap 和 all-reduce latency）：

$$
T_{\text{step}} = L \cdot \left(T_{\text{linear}} + T_{\text{attn}}\right) + T_{\text{overhead}}
$$

其中 $T_{\text{overhead}}$ 包括：sampling、logit 计算、调度、NCCL all-reduce（TP > 1 时）、Python 侧 overhead，通常是几毫秒的常数项。

### 4.1 展开公式

$$
T_{\text{step}} = L \cdot \max\!\left(\frac{36MH^2/\text{TP}}{F},\ \frac{28H^2/\text{TP}}{\text{BW}}\right)
+ L \cdot \max\!\left(\frac{\text{FLOP}_{\text{attn}}}{F},\ \frac{\text{Bytes}_{\text{attn}}}{\text{BW}}\right)
+ T_{\text{overhead}}
$$

定义单层 **权重读时间**（memory-bound 下限）：

$$
t_w = \frac{28 H^2 / \text{TP}}{\text{BW}}
$$

对 H20（BW=4000 GB/s，H=3584，TP=1）：$t_w \approx \frac{28 \times 3584^2 \times 2}{4 \times 10^{12}} \approx 0.18 \text{ ms/层}$

$L=28$ 层合计：$28 \times 0.18 \approx 5 \text{ ms}$，这是**纯 decode（1个请求）时 linear 层的耗时下限**。

### 4.2 两个典型工作点

**工作点 A：纯 decode，$n$ 个请求，无 prefill**

$$
T_A = L \cdot \max\!\left(\frac{1.3n \cdot t_w}{I^*},\ t_w\right) + L \cdot \frac{4 h_{kv} d_h \cdot n\bar{d} / \text{TP}}{\text{BW}} + T_{\text{ov}}
$$

简化（$n < I^*$ 时 linear 部分 memory-bound）：

$$
T_A \approx L \cdot t_w + L \cdot \frac{4 h_{kv} d_h \cdot n\bar{d}/\text{TP}}{\text{BW}} + T_{\text{ov}}
$$

注意 attention 项随 $n\bar{d}$ 线性增长——**decode 的 context 越长，attention 越贵**。

**工作点 B：纯 prefill，$m$ 个请求，$p_j=0$（无前缀命中）**

$$
T_B = L \cdot \max\!\left(\frac{M_p}{I^*} \cdot \frac{28 H^2/\text{TP}}{\text{FLOPs} \cdot \text{something}},\ t_w\right) + \ldots
$$

实际上更直接：

$$
T_B = L \cdot \frac{36 M_p H^2 / \text{TP}}{F} \quad (\text{当 } M_p \gg I^* \text{ 时 compute-bound})
$$

$$
T_B = L \cdot \frac{28 H^2 / \text{TP}}{\text{BW}} \quad (\text{当 } M_p \ll I^* \text{ 时 memory-bound，退化成 decode 的下限})
$$

---

## 5. 关键推论

### 5.1 Mixed batch 的干扰机制（H1 的量化）

混合 batch 中，attention FLOPs 约为：

$$
\text{FLOP}_{\text{attn}}^{\text{mixed}} \approx 4h_{kv}d_h\left[n\bar{d} + M_p(\bar{p}+\bar{c})\right]
$$

而分开跑（decode 先，prefill 后）的 attention FLOPs 之和完全相同。所以：

> **混合 batch 的 FLOPs 与分开跑完全相同，理论上没有额外计算代价。**

额外代价来自：
1. **kernel 调度开销**：varlen kernel 需要处理不均匀的 q\_len 分布，tile 效率下降
2. **decode 拉低了整体算术强度**：原本 compute-bound 的 prefill 被 decode 的 $I=1$ 稀释，若 mixed batch 的等效 $I < I^*$，则退化为 memory-bound

等效算术强度：

$$
I_{\text{mixed}} = \frac{n\bar{d} + \sum_j c_j(p_j+c_j)}{n\bar{d} + m(\bar{p}+\bar{c})}
$$

当 $n\bar{d} \gg \sum_j c_j(p_j+c_j)$ 时，$I_{\text{mixed}} \to 1$（退化成纯 decode）。

**H1 成立的条件**：$n\bar{d}$ 大到足以把 $I_{\text{mixed}}$ 拉到 $I^*$ 以下，使原本 compute-bound 的 prefill attention 变成 memory-bound。这在 `decode 数多 + context 长 + prefill chunk 小` 时最显著。

### 5.2 M\* 的位置（H2 的量化）

Linear 层 ridge point（$M$ 使 linear 从 memory-bound 转为 compute-bound）：

$$
M^* = I^* = \frac{F}{\text{BW}} \approx \frac{148 \times 10^{12}}{4 \times 10^{12}} = 37 \text{ tokens}
$$

Attention 层 ridge point（$\bar{c}$ 使 prefill attention 从 memory-bound 转为 compute-bound）：

$$
\bar{c}^* = I^* = 37 \text{ tokens}
$$

所以 **37 个 token 是 H20 上的硬件 ridge point**。实际测出来的 MFU 饱和点 $M^*$ 会高一些，因为：

- 小 $M$ 时有 kernel launch overhead 等常数项，MFU 提升快
- 大 $M$ 时受 cudagraph 大小、显存带宽饱和等次级约束，MFU 增长变缓

**重要的推论 B 验证**：对 GEMM 来说，$8$ 个请求 × $512$ token 和 $1$ 个请求 × $4096$ token 的 $M$ 相同，因此：

$$
T_{\text{linear}}(\text{8×512}) = T_{\text{linear}}(\text{1×4096})
$$

Attention 则**不等**：前者每个请求的 $c_j = 512$，后者 $c_j = 4096$，$I_{\text{attn}}$ 不同（但 FLOPs 相同，差异在 kernel 效率）。

### 5.3 activation 代价与 H3 的量化（H3 的量化）

每步 forward 的 **peak activation** 约为：

- 每层 attention 的中间结果（QKV 投影输出）：$\sim M \times H \times 2$ Bytes
- FFN 中间激活：$\sim M \times H_{ff}^{tp} \times 2$ Bytes
- Residual stream：$\sim M \times H \times 2$ Bytes

粗估单步峰值：

$$
A_{\text{peak}} \approx 4 M H \times 2 \text{ Bytes} = 8 M H \text{ Bytes} \quad \text{（每卡，TP 后）}
$$

对 H20（H=3584，TP=1）：

$$
A_{\text{peak}}(M) \approx 8M \times 3584 \times 2 / 10^9 \approx 5.7 \times 10^{-5} M \text{ GiB}
$$

| M（token 数） | 估计 activation（TP=1） |
|---|---|
| 8 192 | 0.47 GiB |
| 16 384 | 0.94 GiB |
| 65 536 | 3.74 GiB |

**这个 activation 是在 `max_num_batched_tokens` 下测一次然后永久预留的**（见 `gpu_worker.py:448-507`）。

KV token 容量随 $M$ 的代价：

$$
\Delta\text{KV\_tokens} = \frac{A(M_2) - A(M_1)}{\text{bytes per KV token}}
$$

每 KV token 字节数（Qwen2.5-7B，所有层，GQA）：

$$
\text{bytes/token} = 2 \times h_{kv} \times d_h \times 2 \times L = 2 \times 4 \times 128 \times 2 \times 28 = 57344 \approx 56 \text{ KiB}
$$

从 $M_1=8192$ 升到 $M_2=65536$，activation 增量约 $3.74 - 0.47 = 3.27$ GiB，折合 KV token 损失：

$$
\Delta\text{KV\_tokens} = \frac{3.27 \times 10^9}{56 \times 1024} \approx 57000 \text{ tokens} \approx 57\text{k tokens}
$$

对比 colocated RL（util=0.5）约 524k token 的 KV 容量，这是约 **11% 的损失**，接近 H3b 判决阈值（10%）。**实测将确认这个数字，以及 TP=4 时每卡 activation 减小但 KV 容量也减小的对消效应。**

---

## 6. 完整公式汇总

给定：
- Decode 请求：$n$ 个，context 长度 $d_1, \ldots, d_n$
- Prefill 请求：$m$ 个，已命中 $p_1, \ldots, p_m$，本步 compute $c_1, \ldots, c_m$
- 模型参数：$L, H, h_q, h_{kv}, d_h, H_{ff}$
- 硬件参数：$F$ (TFLOPS), $\text{BW}$ (GB/s), $\text{TP}$

定义 $M = n + \sum c_j$，$I^* = F / \text{BW}$。

$$
\text{FLOP}_{\text{linear}} = \frac{2M}{\text{TP}} \cdot \left[H(h_q + 2h_{kv})d_h + h_q d_h H + 3H \cdot H_{ff}\right]
$$

$$
\approx \frac{M}{\text{TP}} \cdot (4H^2 + 6H \cdot H_{ff})
$$

$$
\text{Bytes}_{\text{weight}} = \frac{2}{\text{TP}} \cdot \left[H(h_q + 2h_{kv})d_h + h_q d_h H + 3H \cdot H_{ff}\right]
$$

$$
\text{FLOP}_{\text{attn}} = \frac{4 h_{kv} d_h}{\text{TP}} \left[\sum_i d_i + \sum_j c_j(p_j + c_j)\right]
$$

$$
\text{Bytes}_{\text{attn}} = \frac{4 h_{kv} d_h}{\text{TP}} \left[\sum_i d_i + \sum_j (p_j + c_j)\right]
$$

$$
\boxed{
T_{\text{step}} = L\!\left[\max\!\left(\frac{\text{FLOP}_{\text{linear}}}{F},\ \frac{\text{Bytes}_{\text{weight}}}{\text{BW}}\right)
+ \max\!\left(\frac{\text{FLOP}_{\text{attn}}}{F},\ \frac{\text{Bytes}_{\text{attn}}}{\text{BW}}\right)\right] + T_{\text{ov}}
}
$$

---

## 7. 数值示例（Qwen2.5-7B on H20，TP=1）

参数：$L=28, H=3584, h_q=28, h_{kv}=4, d_h=128, H_{ff}=18944$
硬件：$F=148\text{T}, \text{BW}=4\text{T Byte/s}, I^*=37$

权重读时间（单层）：
$$
t_w = \frac{\text{Bytes}_{\text{weight}}}{\text{BW}} \approx \frac{2 \times (3584 \times 28 \times 128 + 3584 \times 28 \times 128 + 3 \times 3584 \times 18944)}{4 \times 10^{12}} \approx 0.178 \text{ ms}
$$

| 场景 | $M$ | linear 类型 | $T_{\text{linear}}$（×28层） | attention 类型 | $T_{\text{attn}}$（×28层） | $T_{\text{step}}$ 估计 |
|---|---|---|---|---|---|---|
| 1 decode，$\bar{d}=2048$ | 1 | mem-bound | $28 \times 0.178 \approx 5.0$ ms | mem-bound | $\approx 0.07$ ms | **5.1 ms** |
| 32 decode，$\bar{d}=2048$ | 32 | mem-bound | $32 \times 0.178 \approx 5.7$ ms | mem-bound | $\approx 2.2$ ms | **8 ms** |
| 64 decode，$\bar{d}=2048$ | 64 | 接近 ridge | $\approx 6.7$ ms | mem-bound | $\approx 4.4$ ms | **11 ms** |
| 1 prefill，$c=4096$，$p=0$ | 4096 | compute-bound | $\approx 8.2$ ms | compute-bound | $\approx 2.8$ ms | **11 ms** |
| 1 prefill，$c=512$，$p=0$ | 512 | compute-bound | $\approx 1.0$ ms | mem-bound | $\approx 0.35$ ms | **1.4 ms** |
| 32 decode + 1 prefill $c=2048$ | 2080 | compute-bound | $\approx 4.2$ ms | mixed $I≈2.1$ | mem-bound $\approx 3.1$ ms | **7.3 ms** |

> 这些是 roofline 理论值，实际会有 kernel launch、cudagraph、NCCL 等常数开销（$T_{\text{ov}} \approx 1$–$3$ ms）。E1/E2 实测数据将校准这个模型。

---

## 8. 模型的局限与待校准项

| 假设 | 实际情况 | 影响 |
|---|---|---|
| 各层顺序执行，无 pipeline overlap | cudagraph 把层内 kernel 串行启动，基本成立 | 误差小 |
| FlashAttention 完全 IO-bound 时等于读 KV 时间 | FA2 有 shared memory recompute，实际效率高于估计 | 可能低估 attention 速度 |
| 权重在 HBM，每步都需读 | 若 L1/L2 cache 能 hold 住权重（小模型 + 大 M），bandwidth 需求降低 | 大 M 时 linear 可能比 roofline 快 |
| NCCL all-reduce 被忽略 | TP>1 时每层有 2 次 all-reduce，带宽 $\approx \text{NVLink}$ 900 GB/s | TP=4 时加约 0.3 ms/层 |
| activation 估计用线性模型 | 实际 peak 受 FlashAttention workspace、cuDNN buffer 等影响 | E3b 实测为准 |

**E1 实测的核心价值**：用真实 MFU(M) 曲线校准上表中"linear 类型"的转折点，确认 $M^*$ 的实际位置（预期比理论 ridge point 37 高，因为 kernel launch overhead 使小 $M$ 的 MFU 本来就低）。
