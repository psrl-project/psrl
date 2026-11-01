# PSRL FP8 权重传输与 PS 架构设计

> 适用范围：`rollout FP8 + training FP8` 与 `rollout FP8 + training BF16` 两种组合。
> 本文只写 FP8 相对 `weight_transport.md` 的**增量**。共享架构（PS 索引化、trainer 直供、
> pull 语义、去冗余规划、chunk 执行层、colocate 原地交接）不在此重复，引用其章节号。
>
> 状态：**BF16 / FP32 train → blockwise FP8 rollout 已落地**（`psrl/utils/nixl/block_fp8.py`，
> 落地项 F2、F3、F6、F7，F4 只含 tile 切分的拒绝），含 vLLM 细化的 MoE scale 网格（§3 例外 1）。
> **FP8 train（TE hybrid autocast，bf16 参数）→ blockwise FP8 rollout 已落地**，走发布副本
> 重量化，不是零拷贝（§10 表中 tensorwise 一行）。TE 常驻 FP8 参数（`fp8_param`）在训练引擎
> 路径上不可用：模型不在 `fp8_model_init` 内构建，开启会破坏首次更新，配置校验拒绝（见 §10）。
> 实测见 `weight_transport_FINDINGS.md` §5–§6、§12。
> 零拷贝的 FP8-native、MXFP8 与 colocate 指针交接未做。

---

## 0. TL;DR

五条结论，每条都是从「PSRL 是 pull 架构」推出来的，不是从别的框架抄的：

1. **量化结构不声明，向消费者读取。** rollout 已经以 block FP8（`psrl.nixl.weight_quantization.scheme=fp8`）实例化，vLLM 的
   quant method 已经建好了 `weight`(fp8) + `weight_scale{,_inv}` 并写了 shape。
   哪些层量化、scale 叫什么、grid 多大，**全部可以从已注册的 rollout 参数读出来**。
   miles 与 SkyRL 都维护手写白名单（`ModelFp8Spec`）是因为它们是 push 架构，生产者
   必须先知道消费者要什么；PSRL 是 pull，消费者自己说。**FP8 不引入任何 per-model 白名单。**

2. **scale 是一等 plan 公民，不是附属元数据。** vLLM 自己的 `validate_fp8_block_shape{,_moe}`
   强制每个 TP 分片的 `output_partition_size % block_n == 0`、`input_size_per_partition %
   block_k == 0`，所以 scale 的 block grid **按分片可组合**——全局 grid 恰等于各 rank grid 的拼接。
   例外是 FP8 MoE：分片切不开 tile 时 vLLM 0.30 把 grid 细化并无损上采样 scale（§10 末行）。
   这正是 `NIXLSharding` 需要的条件，于是 scale 走与 weight 完全相同的 `(key, shard)` 抽象，
   复用同一套去冗余规划、peer 扩散与 chunk 流水。不需要第二条通路。

3. **唯一新增的硬不变量：scale 与其 weight 必须落在同一个 commit group。** 这比 bf16 下的
   atomic group 更严厉，因为**半到达不会报错**：拿到新 weight 配旧 scale 的张量数值合法、
   推理不崩、只是输出错。bf16 下 chunk 边界错了会被 coordinate judge 抓住，FP8 下不会。
   所以这条必须是运行时断言（§3.4.6 的形式），不是文档要求。

4. **两种组合差一个算子，不差一条架构。** FP8-native train 直供字节（零转换零缓冲）；
   BF16 train 在发布点做一次 cast，产物是每 rank 一份常驻的 FP8 发布副本（§5）。链路、plan、pull 语义、
   peer 扩散全部不变——**链路不感知精度**（§3.8 原则）。

5. **FP8-native 是唯一能让 colocate 切换零拷贝的形态。** trainer 持有的就是 rollout 要的字节，
   切换退化为指针交接；BF16 train 下 colocate 必须付一次 `M/2` 的 cast。这是本文给出的
   选型建议：**追求 colocate 切换效率就用 FP8-native 训练**，而不是去优化搬运。

---

## 1. 现状：为什么今天不能直接打开 FP8

三条，都在代码里：

- **`_DTYPE_MAP` 缺 `float8_e8m0fnu`**（`psrl/utils/nixl/nixl_spec.py:12`）。
  vLLM 的 blockwise scale 在 power-of-2 模式下 dtype 是 `torch.float8_e8m0fnu`
  （`quantization/fp8.py` 的 `create_fp8_scale_parameter(..., scale_dtype=torch.float8_e8m0fnu
  if self.is_scale_e8m0 else None)`）。该 dtype 不在表里，`NIXLShardMetaInfo.deserialize`
  会在握手期直接失败。表里已有 `float8_e4m3fn` / `float8_e5m2` / `uint8`，
  所以 MXFP8（E8M0 存在 uint8 里）不受影响，**只有 blockwise + power-of-2 这一格受影响**。
- **converter 的 layout 变换无 scale 意识**（`weight_layout_transforms.py`）。
  qkv / merged_column / fused_moe 的切分规则是按 weight 的 `output_dim` 写的，
  scale 的对应维度是 `ceil(dim / block)`，不能复用同一个除法。见 §3。
- **`ray_trainer.py` 按后端硬分流 train dtype**，假定 FSDP2 恒为 fp32。
  这条已记在 `weight_transport.md` 的落地项 9d-a，FP8 下同样要先修。

注意**不包含**「量化语义缺失」这一条。旧方案把 `transfer_train_to_gen` 的朴素 cast 列为硬伤，
但约束 (b) 下 PS 不再持有字节、该函数本身要删，所以它不是 FP8 的问题。

---

## 2. 两种精度组合：同一条路径，差一个算子

```
                      ┌─ FP8-native train (TE fp8_param) ─┐
trainer 参数张量 ─────┤                                    ├──► wire payload ──► rollout 已注册参数
                      └─ BF16 train ── cast(chunk) ───────┘      (fp8 + scale)        (直写，RDMA)
```

| | FP8 train | BF16 train |
|---|---|---|
| 量化算子 | 无 | `per_block_cast_to_fp8` / `MXFP8Quantizer`，发布时逐张量 |
| scale 处理 | **仅编码转换**（TE 的 scale ↔ vLLM 的 `weight_scale_inv`） | 随 cast 产出 |
| 额外显存 | **0**（直读参数） | 每 rank 参数字节的一半（常驻发布副本），见 §5 |
| 数值误差 | 不新增（源头就是 fp8） | 一次 bf16→fp8 |
| colocate 切换 | 指针交接 | 一次 `M/2` cast |

**FP8 train 下绝不「反量化回 bf16 再量化」**：那会白付两次精度损失加一次全量 amax，
而 TE 的 scale 与 vLLM 的 scale 只差一个编码约定。

**scale 编码的硬件分叉（来自 SkyRL 的实测结论，PSRL 必须继承）**：
Hopper 上常驻 `fp8_param` 只在 FP32 scale 下可用；Blackwell 要求 power-of-2 scale，
而 power-of-2 与常驻 fp8 参数**不能同时成立**。后果是：

- Hopper + FP8-native + blockwise(FP32 scale) → **可用**，是 Hopper 上的最优形态。
- Blackwell + FP8-native + blockwise(power-of-2) → **不可用**，必须改走 MXFP8。
- Blackwell + FP8-native + MXFP8 → **可用**，是 B300 级集群上的目标形态。

所以「rollout FP8 + training FP8」在 Hopper 上落到 blockwise、在 Blackwell 上落到 MXFP8，
这不是两套实现——两者都是「trainer 已持有 fp8 字节，只做 scale 编码转换」，
差别封装在 §4 的 wire resolver 里。

### 2.1 wire 开销：MXFP8 的 scale 不可忽略，blockwise 的可以

| wire | scale 粒度 | scale dtype | scale/weight 字节比 | 相对 bf16 总字节 |
|---|---|---|---|---|
| blockwise | 128×128 tile | fp32（4B / 16384 元素） | 1/4096 ≈ 0.024% | **0.5002×** |
| blockwise (po2) | 128×128 tile | `float8_e8m0fnu`（1B） | 1/16384 | 0.5001× |
| MXFP8 | 沿 K 方向 1×32 | E8M0 存 uint8（1B / 32 元素） | **1/32 = 3.125%** | **0.5156×** |

结论：带宽上 MXFP8 比 blockwise 贵 3%，换来 32 元素粒度的量化精度。这个 3% 在两种链路上
都是字节时间的 3%，同节点链路（实测聚合 793 GB/s）上的绝对值约为跨节点（386 GB/s）的一半。
**于是 wire 选择与 §8 的选源策略耦合**：peer 扩散比例高的部署下 MXFP8 的 3% 几乎免费。

### 2.2 真正的开销不在字节，在 key 数

每个被量化的线性层从 1 个 key 变成 2 个 key。按 §3.6.5 的 cost model，
控制面是 `N_keys × 12.5μs`。MoE 分组后实测 579 key 的模型，若七成 key 被量化，
key 数涨到约 980，控制面从 7.2ms 涨到 12.3ms——**比字节减半的收益小一个量级，但不是零**。

解法不是压缩 key，而是**让 scale 不单独占提交位**：scale 必须与 weight 同 commit group
（§3），而同 commit group 内的 descriptor 本来就批量提交。所以把 scale 紧跟其 weight 排进
同一个 chunk，`N_keys` 涨一倍但**提交轮数不变**，`12.5μs/key` 里只有元数据编解码那部分真的翻倍。
这是 §3.5.5「元数据平面先去按 key 展开」已经做掉的工作的直接受益。

---

## 3. scale 作为一等 plan 公民：可组合性是前提，vLLM 自己保证了它

要让 scale 复用 weight 的整条通路，必须成立一件事：**scale 张量的全局形状，
等于各 rank 本地 scale 形状按同一个 shard_mesh 拼起来的结果**。

这件事不是显然的。vLLM 的 scale shape 是向上取整的：

```
w13_scale: [E, w13_num_shards × ceil(I_per_partition / block_n), ceil(H / block_k)]
w2_scale:  [E, ceil(H / block_n),                                ceil(I_per_partition / block_k)]
```

`ceil` 一般不可分配：`ceil(a/b) + ceil(c/b) ≠ ceil((a+c)/b)`。若 scale grid 不可组合，
scale 就无法用 `NIXLSharding` 表达，只能另走一条「每 rank 独立、不参与去冗余」的通路——
那会让 FP8 退化成两套传输架构。

**但 vLLM 自己把 `ceil` 消掉了。** `validate_fp8_block_shape` 对列并行/融合矩阵强制
`output_partition_size % block_n == 0`，对行并行强制 `input_size_per_partition % block_k == 0`；
`validate_fp8_block_shape_moe` 对 `intermediate_size_per_partition` 强制同样两条。
整除时 `ceil` 退化为除法，而除法可分配。于是：

> **在 vLLM 接受的所有 FP8 配置下，scale 的 block grid 按 TP/EP 分片可组合。**
> `shard_mesh` 与其 weight 相同，每维的全局长度除以 `block` 即为 scale 的全局长度。

这是本设计的技术支点。它的含义是：scale 的 `NIXLSharding` **从 weight 的 sharding 机械导出**，
不需要为 scale 写第二套 layout 规则：

```
scale_mesh   = weight_mesh                       # 分片结构完全相同
scale_global = tuple(d // blk if blk else d      # 逐维除以 block（None 表示该维不分块）
                     for d, blk in zip(weight_global, block_shape))
scale_indices = weight_indices                   # 同一个 rank 拿同一块
```

**两处例外，必须显式处理而不是假装不存在：**

1. **vLLM 会细化 MoE 的 scale 网格。** TP 切分使每 rank 的 expert 中间维不是 128 的整数倍时
   （Qwen3-30B-A3B 在 TP4 下每 rank 192 行），vLLM 0.30 把 expert 的 scale 网格细化为
   `gcd(128, 每 rank 中间维, hidden)`（这里是 64 × 64），量化仍按 checkpoint 的 128 × 128 tile，
   loader 用 `repeat_interleave` 把每个 128 tile 的 scale 复制到它覆盖的 2 × 2 个细 tile，再按 rank 切。
   FP8 字节不变。miles 与 verl 都把全量权重按 128 tile 量化后交给引擎的 `load_weights`，
   所以它们自动拿到这一步。PSRL 直写参数，不经 loader，于是**在发布侧做同一件事**：
   rollout 的 converter 在每个 scale key 的 sharding 上声明它的 scale tile（权重形状 / scale 形状），
   meta server 把它带给推导出的 trainer scale sharding，trainer 仍按 128 tile 量化，
   把 scale 按 `128 / scale tile` 复制后发布。结果与 vLLM 从 128 tile checkpoint 加载的逐位相同。
   trainer 自己的 shard 必须含整 128 tile，否则注册时拒绝。
2. **per-tensor scale 不可重分片。** per-tensor 下 scale 是整张矩阵一个标量
   （MoE 下是 `[E, w13_num_shards]`），trainer 的分片与 rollout 的分片若不同，
   两边的 amax 范围不同 → scale 不同 → 字节不可换。vLLM 的 loader 靠
   `requantize_with_max_scale` 在加载时做全局 requant 来解决，而直写路径没有这一步。
   **处理方式：per-tensor FP8 只在 train 与 rollout 的该维分片一致时允许，否则配置校验拒绝，
   提示改用 blockwise 或 MXFP8。** blockwise/MX 不受此限，因为它们的 scale 是局部的——
   这正是 blockwise 成为 RL 场景默认 wire 的根本原因，不只是精度。

---

## 4. 量化结构从消费者推导，不从生产者声明

这是本文与 miles / SkyRL 最大的架构分歧，也是 PSRL pull 架构唯一的、别人拿不到的红利。

**它们的做法（push 架构的必然结果）：** 生产者先量化再发，所以生产者必须提前知道
「哪些层要量化、scale 叫什么、grid 多大」。miles 用显式白名单枚举
（`self_attention.linear_{proj,qkv}`、`mlp.linear_fc1/fc2`、MoE experts/shared_experts、MLA…），
SkyRL 用 `ModelFp8Spec` registry。两者都是 **per-model 手工资产**，接新模型就要加一条。

**PSRL 的做法：** rollout 侧已经以 block FP8（`psrl.nixl.weight_quantization.scheme=fp8`）实例化，
vLLM 的 quant method 已经在 `create_weights` 里建好了全部参数：
`create_fp8_weight_parameter` 建 fp8 weight，`register_parameter("weight_scale_inv")`
或 `w13_{weight_scale_name}` 建 scale，并把 `quant_method`
（`FusedMoeWeightScaleSupported.BLOCK/TENSOR`）写进 weight attrs。
而 `convert_vllm_inplace` 已经在遍历这些真实参数做注册。

于是**整个量化结构是读出来的**：

| 需要知道的事 | 从哪读 | 不需要什么 |
|---|---|---|
| 哪些 key 是 fp8 | 已注册参数的 `dtype is float8_e4m3fn` | 白名单 |
| scale 的 key 名 | 同一模块下 `weight_scale` / `weight_scale_inv` / `w13_*` / `w2_*` 的存在 | 命名约定表 |
| scale 的 dtype 与编码 | 该参数的 `dtype`（fp32 / `float8_e8m0fnu` / uint8） | 硬件判断 |
| block grid | `layer.weight_block_size` + scale 实际 shape | 推算 |
| wire 变体 | `quant_method` attr + scale dtype | 配置项 |
| 哪些 key 保持 BF16 | 不是 fp8 的那些（embedding/norm/router/bias/lm_head 自然落在这里） | 排除列表 |

**这不是探测（probing），是读声明。** WeightBridge 探测 `load_weights` 的行为来反推对应关系，
那种做法脆弱（依赖副作用可观测）。这里读的是 vLLM 自己在推理时依赖的同一批参数与 attr：
**如果它们是错的，vLLM 加载任何 checkpoint 都会失败。** 元数据来源在上游自己的
正确性关键路径上，这是它可信的原因，也是本文愿意把它当架构基础的原因。

trainer 侧需要做的只有两件事，都与模型无关：

- **BF16 train**：对「rollout 要求是 fp8 的那些 canonical key」做 cast，block_shape 由
  rollout 的 scale shape 反解。要量化的集合完全由 pull plan 给出。
- **FP8 train**：TE 已持有 fp8 + scale，做 scale 编码转换。TE 侧的布局由
  `megatron.bridge.models.conversion.fp8_export.detect_fp8_export_layout` 从张量的
  `get_metadata()` 读出（`format_name` / `block_shape` / `scale_dtype` / `compact_scale_shape`
  / `with_gemm_swizzled_scales`），**同样是读而不是声明**。两侧都读，中间做形状与编码的一致性断言。

一处必须警觉的不对称：`detect_fp8_export_layout` 的 `validate()` 要求 MXFP8 的 scale 是
**compact 且未 swizzle** 的。TE 在某些 recipe 下持有 GEMM-swizzled scale，那种布局不能直供，
必须先 unswizzle。这是 FP8-native 路径上唯一一处真实的格式转换工作，字节量是 scale 的
3%，可以放在 chunk 流水里。

---

## 5. cast 产物的生命周期：pull 语义下由 plan 决定，不由运行时猜

这是「rollout FP8 + training BF16」组合唯一的真实架构难点，值得单独一节。

§3.3 的「零额外缓冲」依赖一件事：rollout 读的就是 trainer 的参数张量本身。
BF16 train + FP8 rollout 打破了它——cast 的产物是一块**新内存**，trainer 的参数里没有它。
于是 trainer 必须持有这块 buffer，持有到什么时候？

push 架构下这个问题不存在：发送方知道自己发完了。**pull 架构下 trainer 不知道读者何时读完**，
而提前释放会让读者读到已回收的内存——RDMA 写进来的地址仍然有效，所以**不报错，只是错**。

**解法：cast 产物活一个版本，在发布点刷新。** 流式复用的 buffer 只有在所有读者跟随
trainer 的 cast 节拍读取时才可回收，而 pull 语义下版本 v 的读者可以在 v+1 发布前任意时刻到达。
挂在 plan 上的回收（「`planned_readers(c)` 全部读完才回收 chunk `c`」）在正确性上成立，
但它让 trainer 的下一次 cast 等最慢的读者。peer 扩散还会延长持有期：follower 从 designate 读，
designate 读完上游的时刻就晚于 trainer 的 cast。

所以 cast 产物是**常驻的发布副本**：trainer 为本 rank 每个需要转换的 shard 留一份 rollout dtype
的连续拷贝，在推送锁内重量化，读者在锁外随时读。没有回收协议，也没有在飞读者计数。
发布时逐张量量化，瞬时显存是一个 key。

**代价按 rank 计。** block FP8 副本是本 rank 参数字节的一半，另加 1/16384 的 scale。
它占的是 trainer 显存，与 rollout 侧的 KV cache 无关。peer 扩散的 follower 读的是 designate
落地后的引擎参数，不读 trainer 的副本，所以不延长副本的持有期。

**与 §3.7.2.3 的衔接**：布局转换产物被那一节认定为「第三类字节」。cast 产物是同一类，
生命周期规则相同——这里把那条规则具体化到 FP8，并补上 peer 扩散带来的额外持有期。

**FP8-native train 完全没有这一节的问题**：没有 cast 产物，读的是参数本身，
回收由参数自己的生命周期管。这是 §0 第 5 条结论的另一半理由。

---

## 6. colocate 切换：FP8-native 是唯一零拷贝形态

PSRL 的需求明确指出当前基于 PS-CPU 的 colocate 切换比 verl 类框架慢。
`weight_transport.md` §1.3 已定位原因（经由 host 搬全量模型），§3.7 给出 colocate
原地交接方案。FP8 在这件事上的增量是**它改变了「要搬多少」的量级，而不只是搬法**。

eval 用的 colocate 切换，本质是一次资源再分配：trainer 的参数与 optimizer 必须让位给
KV cache，rollout 必须拿到当前版本权重。三种形态的账：

| 形态 | rollout 要的字节 | 从哪来 | 切换期额外 HBM |
|---|---|---|---|
| BF16 train + BF16 rollout | `M` | trainer 参数（同精度直读） | 0（但 trainer 参数必须整块留在 HBM） |
| BF16 train + FP8 rollout | `M/2` | 一次全量 cast | 每 rank 一份 FP8 副本（合计 `M/2`） |
| **FP8 train + FP8 rollout** | `M/2` | **trainer 参数就是它** | **0** |

第三行是定性而非定量的差别：rollout 的参数张量可以**指向 trainer 已持有的那块 fp8 内存**，
切换退化为「停 trainer、交接指针、起 rollout」。没有搬运，所以没有搬运带宽、
没有 cast kernel、没有第三类字节。§3.7.2.2 已算出「FP8-native 是唯一能压到单份的路径」，
本节把它的推论说完：**它也是唯一能让切换时间不随模型规模增长的路径。**

第二行是 Hopper 上 BF16 训练时的现实选择。它比今天好在两点：cast 产物是
`M/2` 而非 `M`，且全程在 HBM 内、不经 host。按实测同节点带宽 365 GB/s 与
cast kernel 的显存带宽上界，这是一次「`M/2` 字节的本地操作」，而今天是
「`M` 字节的两跳 host 往返」。

**切换方向的不对称，必须说清。** eval 结束切回 trainer 时，trainer 要的是 bf16 参数，
而 HBM 里只有 fp8——**fp8 不能还原成 bf16**。所以 trainer 的 bf16 参数必须在切出时
保存到某处（§3.3.1「从 CPU reload，绝不能读 checkpoint」）。这对
「rollout FP8 + training BF16」是真实成本，而 FP8-native 形态下再次免疫：
trainer 要的也是 fp8，交接是对称的。

**这给出一条明确的选型建议**：若 colocate 切换效率是首要目标，
正确的动作是把训练改成 FP8-native（Hopper 走 blockwise+FP32 scale，Blackwell 走 MXFP8），
而不是去优化 bf16 权重的搬运路径。前者让成本归零，后者只能让成本减半。

---

## 7. CPU 不留副本；不可避免时必须分片而非复制

需求里两条合起来是一个强约束：B300 级集群或 1TB 级模型下 CPU 放不下权重，
且占用 host 内存会挤掉 LMCache 的 KV cache 空间。

**稳态传输链路上 CPU 副本为零**，这是约束 (b) 的既有结论（§3.1 PS 不再持有字节），
FP8 不改变它，只是让任何残留副本的字节数减半。

**唯一真实需要 host 的地方是 colocate 切出时的 trainer 状态寄存**（§6 末段）。
对它的要求是「分布式而非集中」，这一点在本架构里是**自然成立的，不需要额外设计**：

- 每个 trainer rank 只寄存**自己那一份分片**到**本节点**的 host 内存。
  没有 gather、没有副本、没有跨节点流量。总 host 占用 `= Σ 本地分片 = 模型大小`，
  按节点数均摊，而不是每节点一份全量。
- 这与 PivotRL 的 Global Parameter Storage 形成对照：GPS 是
  `num_models × M` 的集群 CPU 池，用来支持「角色缩容到 0 再恢复」。
  本架构不要那个能力（§3.10.4 已把它列为档位 B 选项），所以不付那个代价。
- **与 LMCache 的空间冲突由此变成可核算的**：占用只在切换窗口内存在、
  只占本地分片、FP8 形态下再减半。寄存区应当向 LMCache 显式预留（配置项），
  而不是两者各自 best-effort 地争 host 内存——后者的失败模式是 OOM 或
  LMCache 静默降低命中率，都很难归因。

1TB 级模型的量级检查：1TB bf16 参数 + 典型 optimizer 状态，按 128 节点均摊，
每节点寄存量在百 GB 量级——**这只在分片成立时可行，集中式 PS 在这个规模上不存在可行解**。
FP8-native 下参数部分减半，但 optimizer 的 fp32 master 与 moment 不减，
所以这一节的结论是「分片是必需的，FP8 是减免的」，不能反过来指望 FP8 让集中式重新可行。

---

## 8. 传输效率：FP8 把瓶颈从带宽推向链路选择与控制面

字节减半不是均匀的收益。按 §3.6.5 的 cost model：

```
T_rdma = M_v / BW_NIC + ceil(M_v / chunk) × 2μs
T_fmt  = N_keys × 12.5μs + 2 × M_v / 4800
```

FP8 把 `M_v` 减半、把 `N_keys` 近乎翻倍。于是：

- **`M_v/BW` 项减半**，这是跨节点路径（实测节点聚合 386 GB/s）上的主项，收益最大。
- **`N_keys × 12.5μs` 项翻倍**，控制面占比上升。在同节点路径
  （实测聚合 793 GB/s，约为跨节点的 2×）上，字节项更小，所以**FP8 下同节点传输
  可能变成控制面主导**。这改变了 chunk 大小的最优取值：§3.5.6 已给出「轮数与 chunk
  大小的规则是反的」，FP8 下还要再向「少而大的 chunk」偏移一档，因为每 chunk 的
  key 数变多而字节数变少。
- **peer 扩散的价值不随精度改变**。扩散省的是 trainer 出口（`R × M_v` 降到每节点一份），
  `M_v` 减半时它与 peer 读的字节同比减半，而成本（每 key 一个 descriptor）不变。

**cast 不在拉取路径上。** BF16 train 路径下 cast 发生在发布点（推送锁内），拉取读的是已经量化好的副本，
所以 cast 的代价算在 trainer 的发布时间里，不算在 rollout 的 sync 时间里。

**不要做的优化**：把 scale 单独成一波传（「scale 很小，先发完再发 weight」）。
它违反 §3 的同组不变量，且 scale 在 blockwise 下只占 0.024% 字节——
省不出任何带宽，只换来一个静默错误的入口。

---

## 9. 自动模型接入：从「排除」改为「分两层」

`weight_transport.md` §8 按约束 (a) 排除了自动接入。本节在 FP8 的语境下**重开这个决定**，
理由是 FP8 本来会让手工成本翻倍（每个模型多一套量化白名单与 scale 命名），
而 §4 已经证明那一半成本可以归零。既然量化结构能读，就该问：**layout 结构能不能也读？**

**先把手工资产量化清楚。** 今天 vLLM 侧是 16 个手写 layout builder
（`vllm_patches/weight_layouts.py`），覆盖约 26 个注册类。读其内容，
绝大多数 builder 是同一段样板：

```
builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
builder.fused_moe("w13_weight", "w2_weight", ..., num_experts=<config 里某个字段名>)
builder.exclude_substr("rotary_emb.inv_freq")
```

真正的差异只有三类：(1) `num_experts` 在 config 里叫什么
（`num_experts` / `n_routed_experts` / 模型属性）；(2) MLA 的
`fused_qkv_a_proj` 需要 `q_lora_rank` / `kv_lora_rank` / `qk_rope_head_dim` 算分段宽度；
(3) 多模态模型的塔划分与 delegate。

**而前两类 vLLM 已经机器可读地声明了：**

| 手写的东西 | vLLM 的声明 | 覆盖面 |
|---|---|---|
| `builder.qkv(...)` / `builder.merged(...)` | 类属性 `packed_modules_mapping` | **143 个模型类** |
| `builder.fused_moe(..., num_experts=...)` | 实例方法 `get_expert_mapping()`（→ `fused_moe_make_expert_params_mapping`，返回 `(param_name, weight_name, expert_id, shard_id)`） | 16 个 MoE 类 |
| 分片维度与 TP 偏移 | 参数对象的 `output_dim` / `input_dim` / `packed_dim` / `tp_rank` | 全部 vLLM 参数 |
| 量化结构与 scale | 已注册参数本身（§4） | 全部 |

Megatron 侧**早已是自动的**：`megatron_converter.py` 用
`AutoBridge.from_hf_config` 产出 megatron↔HF 映射，PSRL 只消费它。
所以「自动接入」从来不是一个两侧对称的问题——**它是一个单侧问题，
缺的只有 vLLM 侧那一层**。这个判断是重开 §8 的直接依据：工作量比 §8 假定的小得多。

### 9.1 架构：derive-then-override，不是 probe

```
vLLM 模型实例
   ├─ packed_modules_mapping  ──┐
   ├─ get_expert_mapping()    ──┼──► LayoutDeriver ──► WeightLayoutPlan（自动）
   ├─ 已注册参数 + attrs      ──┤                           │
   └─ quant method 建的 scale ──┘                           │
                                                            ▼
                              手写 override（仅残差）──► 合并后的 plan
```

两条设计原则：

1. **自动推导产出的是 plan 数据，不是代码路径。** 这正是 §8 保留的第二点接口余地
   （「plan 是数据而非代码路径」）。planner 与执行层完全不需要改——
   它们消费 `(key, shard) layout`，不关心这个 layout 是手写的还是推导的。
   **这意味着自动接入可以增量上线，一个模型一个模型地切，随时回退。**
2. **override 是残差机制，不是兜底机制。** 推导不出来就**报错**，不是静默回退到
   「假设是 dense」。静默回退会生成一个结构上自洽但语义错误的 plan，
   而 FP8 下这种错误不会被 coordinate judge 抓住（§3）。

### 9.2 为什么敢信它：迁移路径自带验证

自动推导的风险是「看起来对」。但这里有一个**零信任代价的迁移路径**：
26 个已支持的模型都有手写 plan 作为 oracle。

```
对每个已支持模型：
    auto = LayoutDeriver(model).build()
    hand = <现有 builder>(model)
    断言 key 集合、每 key 的 shape、shard_mesh、shard_indices 覆盖完全相等
    相等 → 删掉手写 builder
    不等 → 差异就是待补的推导规则（或一条合法的 override）
```

这个 diff 不需要集群、不需要跑训练、不需要信任任何推理——它是两个数据结构的相等性。
**先让自动推导复现全部 26 个已知正确的 plan，再让它去接第 27 个模型。**
`weight_transport.md` §7.0 的四维参数化矩阵已经有做这件事的测试骨架。

### 9.3 FP8 下自动接入的额外收益：scale 映射是导出的

Megatron-Bridge 的 `quant_mapping.convert_to_amax_map` 给了一个值得直接采纳的模式：
**量化相关的映射不手写，从 weight 映射机械导出**
（`layer.weight` → `layer.weight_quantizer._amax`，MoE 下导出为
`MoeAmaxFanoutMapping`，QKV/gate-up 的 dict 映射导出为 `AmaxFanoutMapping`）。

PSRL 对 scale 做同一件事，且比 amax 更简单——§3 已证明 scale 的 sharding
从 weight 的 sharding 机械导出，§4 已证明 scale 的 key 名与 dtype 从 rollout 读出。
所以 **FP8 在自动接入这条线上不增加任何 per-model 工作量**：
推导出 weight 的 layout 就等于推导出了 scale 的 layout。

这是把「自动接入」和「FP8」放在同一份文档里的原因：两件事不是并列的两个需求，
后者是前者的免费推论，而反过来若坚持手写，FP8 会让手写成本翻倍。

### 9.4 诚实的边界

- **多模态塔划分推导不出来。** `get_expert_mapping` / `packed_modules_mapping`
  不描述「哪些模块属于视觉塔、视觉塔的 head_size 是多少」。VL 模型继续手写 override，
  这是 26 个里的 7 个左右。
- **MLA 的 `fused_qkv_a_proj` 分段宽度**来自 config 的 lora rank，
  `packed_modules_mapping` 只给成员名不给宽度。可以从**参数的实际 shape** 反解
  （`q_a_proj` 的 output 宽度就是 `q_lora_rank`），但这条反解只在两个成员时唯一，
  三个以上就欠定。先 override，不勉强。
- **这不会消除模型接入工作，只会把它从「写一套 layout」降到「写残差」。**
  不要宣称自动接入等于零工作量——`exclude_substr("rotary_emb.inv_freq")`
  这类条目依然要人判断。
- **小于一个 tile 的融合参数按参数整体量化。** rollout 的量化单位是它自己消费的张量，
  不是 checkpoint 视图。Qwen3.5 GDN 的 `in_proj_ba` 只有 `2 × linear_num_value_heads` 行，
  checkpoint 里的 `in_proj_b`/`in_proj_a` 各切掉半个 tile；读取端把两个视图的源字节按
  目标行序组装，用 vLLM 的 `per_block_cast_to_fp8` 语义（补齐到 128 行再裁回）整体量化一次，
  scale 以参数名登记。这套「`quant_unit`」只改变读取端的量化粒度，wire 仍是 HF 视图。

---

## 10. 配置矩阵与拒绝规则

`weight_transport.md` §3.8.0 的矩阵按 (后端, train dtype, rollout dtype) 列举。
本节只补 FP8 行上的硬件与 wire 维度，以及**必须拒绝的格子**。

| 后端 | train | rollout wire | 硬件 | 状态 |
|---|---|---|---|---|
| Megatron | bf16 | blockwise | Hopper+ | 支持。**Hopper 上 BF16 训练的推荐配置** |
| Megatron | bf16 | MXFP8 | SM100+ | 支持，wire 贵 3%，量化精度更好 |
| Megatron | fp8 (TE `fp8_param`, blockwise recipe) | blockwise + FP32 scale | **Hopper，CUDA ≥ 12.9** | 设计上零转换零缓冲。**未落地**：本集群 CUDA 12.8，TE 的 `Float8BlockScaling` 不可用 |
| Megatron | fp8 (TE `fp8_param`, tensorwise / delayed) | blockwise | Hopper+ | **训练引擎路径拒绝**：模型不在 `fp8_model_init` 内构建，flag 不会产生 QuantizedTensor，却会改坏 FP8 反向的 weight-grad 记账（第 2 步 entropy/grad 爆掉）。FP8 hybrid autocast（bf16 参数）支持并已端到端验证 |
| Megatron | fp8 (TE) | blockwise + power-of-2 | — | **拒绝**：po2 与常驻 fp8 参数互斥（硬件约束，非本设计限制） |
| Megatron | fp8 (TE) | MXFP8 | **SM100+** | 支持。**B300 级集群的目标形态**，colocate 零拷贝 |
| FSDP2 | fp32 / bf16 | blockwise / MXFP8 | 按 wire | 支持（cast 路径，冲突恒为零） |
| FSDP2 | fp8-native | — | — | **拒绝**：FSDP2 无 TE `fp8_param` 等价物（上游缺失） |
| 任意 | 任意 | **per-tensor**，且该维 train/rollout 分片不一致 | — | **拒绝**：scale 不可重分片（§3 例外 2） |
| 任意 | 任意 | blockwise + `weight_scale_refine`（MoE） | — | 支持：trainer 按 rollout 声明的细网格复制 scale（§3 例外 1） |
| 任意 | 任意 | 任意 | — | **拒绝**：validate 与 rollout 的 `quantization` 不一致。两者消费同一份发布字节，validate 引擎由 `train_actor_rollout_ref.rollout` 构建；只给 `gen_actor_rollout_ref.rollout` 开 FP8 时，validate 实例以 bf16 注册，第一次拉取新版本即触发 dtype 断言 |

拒绝必须发生在**握手期的配置校验**，而不是传输期。理由与 §3 同一条：
FP8 的错误不会在数值上暴露。校验要检查的恰好是 §3 与 §4 推导所依赖的前提：
scale grid 可组合、scale dtype 在 `_DTYPE_MAP` 里、每个 fp8 weight 都有配对 scale
且两者同组、wire 变体与硬件相容、validate 与 rollout 的量化声明一致。

---

## 10b. 三条未落地精度路径：什么是硬件/CUDA 限制，什么还要写代码

本集群实测（H20，compute capability 9.0，CUDA 12.8，TE 2.7.0）：

| 路径 | TE / vLLM 的门禁 | 本集群 | 还需要的代码 |
|---|---|---|---|
| FP8 delayed / tensorwise 训练 | `check_fp8_support` → True | ✅ 可用，**已落地** | 无 |
| **零拷贝 FP8-native**（blockwise 常驻参数） | `check_fp8_block_scaling_support` → False：`requires Hopper and CUDA >= 12.9` | ❌ CUDA 12.8 | **仍需 F8**：TE 的 block scale 布局与 vLLM 的 `weight_scale_inv` 不同，要做编码转换才能把 TE 的字节直供出去。升级 CUDA 只解开 recipe，不解开布局 |
| **MXFP8** | `check_mxfp8_support` → False：`compute capability 10.0 or higher` | ❌ Hopper 不支持 | 需要 F8 的 MXFP8 分支（含 scale 的 unswizzle）。`torch.float8_e8m0fnu` 已存在，缺的是硬件与布局转换 |
| **FP4** | vLLM 的 FP4 kernel 要 SM100+ | ❌ 同上 | `torch.float4_e2m1fn_x2` 已存在；§3.8.1 的两处（scale dtype、grid 可组合性）仍要显式处理 |

**DeepGEMM 后端。** vLLM 0.30 对 block-scaled FP8 线性层默认选择 DeepGEMM，veRL 的
`set_default_vllm_quantization_backends` 会把 `linear_backend` 设为 `deep_gemm`。本仓库固定的
DeepGEMM 版本在 JIT 编译时强制要求 nvcc >= 12.9；GPU 镜像基于 CUDA 13.0.3，满足该要求并在
vLLM 之后安装同一 pin。开发机若只有更老的 toolkit，需要把 `CUDA_HOME` 指向 12.9+ 的 toolkit，
或显式设置 `gen_actor_rollout_ref.rollout.engine_kwargs.vllm.linear_backend=cutlass`。包缺失时
direct-supply FP8 会在启动阶段直接报错并给出这两条出路，不会静默回退到别的 kernel。

**结论：三条都卡在硬件或 CUDA 版本上，但没有一条只靠升级就完成。** 升级 CUDA 到 12.9 让 Hopper 上的
blockwise FP8 训练跑起来，可 PSRL 仍要把 TE 的 scale 布局翻译成 rollout 声明的那一种，否则只能像现在这样
反量化再重量化（实测每 rank 发布刷新 513 ms，现已被融合 kernel 降到 56 ms 量级）。
MXFP8 与 FP4 在拿到 Blackwell 之前无法验证，按设计惯例**不写无法验证的代码**，留作规划。

## 11. 落地项

顺序即依赖顺序。标 `*` 的是两种精度组合共用。

| # | 项 | 说明 |
|---|---|---|
| F1\* | `_DTYPE_MAP` 补 `float8_e8m0fnu` | `nixl_spec.py:12`。一行，但不补则 blockwise+po2 在握手期失败 |
| F2\* | `ScaleSpec` 推导器 | 从 rollout 已注册参数读出 `(weight_key → scale_key, dtype, block_shape, quant_method)`，§4 的表 |
| F3\* | scale 的 `NIXLSharding` 导出 | 从 weight sharding 机械导出（§3 的三行公式），含两处例外的断言 |
| F4\* | 配置校验与拒绝 | §10 的四条拒绝 + §3/§4 的前提检查，握手期执行 |
| F5\* | commit group 原子性断言 | 每个 fp8 weight 与其 scale 同组；**运行时断言**（§3.4.6 形式），不是文档要求 |
| F6 | BF16→FP8 cast 算子 | 复用 vLLM `per_block_cast_to_fp8` / TE `MXFP8Quantizer`，block_shape 由 rollout scale shape 反解 |
| F7 | cast 产物生命周期 | 常驻发布副本，推送锁内刷新（§5） |
| F8 | TE scale 编码转换 | 用 `fp8_export.detect_fp8_export_layout` 读 TE 布局；含 MXFP8 unswizzle（§4 末） |
| F9 | colocate 指针交接（FP8-native） | §6 第三行；bf16 train 的切回路径仍走 §3.3.1 的 CPU reload |
| F10\* | host 寄存区分片化 + 向 LMCache 显式预留 | §7 |
| F11 | chunk 大小重标定 | FP8 下 `N_keys` 翻倍、`M_v` 减半，§3.6.5 的 cost model 重新取点（§8） |
| F12 | `LayoutDeriver` + 26 模型 diff | §9.2 的相等性迁移，纯 CPU 测试，不需集群 |
| F13 | scale 映射从 weight 映射导出 | §9.3，F12 落地后几乎免费 |

F1–F5 是 FP8 的正确性地基，必须先于 F6/F8 任一条传输实现。
F12/F13 与 F1–F11 正交，可并行推进。

---

## 12. 验证

分层沿用 §7.0 的结构。FP8 特有的四条：

1. **纯 CPU：scale sharding 导出的可组合性。** 对矩阵里每个合法 (block_shape, TP, EP)，
   断言「各 rank 导出的 scale shard 拼起来等于全局 scale shape」，
   并断言两处例外被拒绝。不需要 GPU。
2. **纯 CPU：原子性反例。** 构造一个 scale 与 weight 落在不同 commit group 的 plan，
   断言 F5 的断言触发。**这条必须存在**，因为它是唯一能在没有数值 oracle 的情况下
   守住 §3 不变量的手段。
3. **单卡数值：量化一致性 oracle。** 同一份 bf16 权重走
   (a) PSRL 的 cast + 直写、(b) 写成 HF fp8 checkpoint 再由 vLLM 正常
   `load_weights` + `process_weights_after_loading` 加载，断言两条路径的
   最终参数**按位相等**。这是唯一能证明「绕过 loader 是安全的」的测试。
   SkyRL 的 torch 版 MX 实现可作为 MXFP8 的第二个 oracle。
4. **多节点：两种组合各跑一次端到端。** 断言 rollout 输出与 BF16 基线的
   logprob 偏差在量化噪声内，且 `N_keys`/字节/轮数符合 §8 的预测。
   **运行日志必须含「所有节点 import 本工作树」的那一行**——
   否则结果不可用（这是本仓库已踩过的坑）。

F12 的 26 模型 diff 本身就是自动接入的验证，不另设。

---

## 13. 风险与边界

- **FP8 的错误是静默的。** 这是本文反复强调的那一条：scale 配错、
  grid 不匹配、半组到达，全都产出数值合法的张量。所以本设计把
  §3/§4 的每个推导前提都变成握手期校验或运行时断言（F4/F5），
  并要求一个按位相等的 oracle（§12.3）。**不要用「跑起来了、loss 没炸」当验证。**
- **本文不覆盖 FP4。** §3.8.1 已指出 FP4 需要额外处理 swizzle，
  且 `weight_scale_refine` 类的 grid 问题在 FP4 下更普遍。
- **本文不覆盖 LoRA。** §3.11 已显式拒绝；SkyRL 的
  「serialized blockwise 拒绝 adapter-only Megatron LoRA」是同一条边界。
- **自动接入不等于零工作量**（§9.4）。多模态塔与 MLA 分段宽度继续手写 override。
- **FP8-native 训练的收敛性不是本文的论证范围。** SkyRL 报告端到端
  步时降低至多 23%、收敛与 BF16 基线可比，但也明确要求长程实验中监控数值稳定性。
  本文只负责「字节正确地到达」，算法侧的精度结论需要独立实验。
- **`fp8_param` 的 HBM 收益（参数部分 39%~42%）不改变 optimizer 状态的占用。**
  §7 的 1TB 量级检查依赖分片，不依赖 FP8。
