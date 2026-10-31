# PSRL 权重传输与 PS 架构设计

> 状态：**数据面已落地**。trainer 直供、同节点去冗余、发布副本、cast 与 block FP8
> 均已接入生产拉取路径。落地状态与待办见 `weight_transport_implementation.md` §9，
> 实测数据见 `weight_transport_FINDINGS.md`。FP8 精度组合的增量架构见 `fp8_weight_transport.md`。
> 约束（用户给定，本文据此收敛）：
> **(a) 不做自动模型接入**，沿用现有显式 layout plan（该决定已重开，见 §8）。
> **(b) 不引入额外的 GPU 权重存储，也不引入额外的 CPU 权重存储**，一律从 trainer 或 rollout
> GPU 上**已经存在**的权重取数。
> **(c) TMS 仅在必不可少或能极大简化实现时使用。**
> **(d) 不依赖 SkyRL / miles 作为第三方库。** 它们在本文中只作为**设计调研对象**，
> 不产生任何 import。所需原语全部来自 PSRL 已有依赖：vLLM（`fp8_utils`、`deep_gemm`、
> `CuMemAllocator`）、Transformer Engine（MXFP8 quantizer、fp8 参数）、NIXL、Megatron-Bridge。
> 调研对象：SkyRL v0.4（`third_party/SkyRL`）、miles（`third_party/miles`）、verl（`third_party/verl`）、
> WeightBridge（arXiv 2609.25442）、PivotRL（`third_party/pivotrl.pdf`）、
> NIXL（`third_party/nixl`）、vLLM（`third_party/vllm`）。
>
> **本文已按 [weight_transport_review](weight_transport_review) 的审核结论修订。**
> 审核推翻了初稿的两处事实判断（Megatron 需要 gather、rollout 初始化会读 checkpoint）
> 和一处时序论证（quiesce 窗口），修订已并入 §3.2 / §3.3 / §3.4 / §3.5 / §3.6。

---

## 0. TL;DR

约束 (b) 把设计空间收得非常窄，而这恰好让方案变锐利：**既然不许有任何一份额外副本，
那么权重的唯一副本就只能是 trainer GPU 上那份。PS 于此退化为一个不持有字节的索引服务。**

**约束 (b) 的可操作表述**（逐节核实后收敛，因为字面的「零额外存储」在三处不成立）：

> **不允许任何「随模型大小增长」的额外权重存储；允许固定大小的、与模型无关的传输缓冲。**

按这条标准，本设计的账是：

| 项 | 量级 | 是否符合 |
|---|---|---|
| CPU 侧权重副本（今天的 `N×M`） | `N×M` → **0** | 符合，这是设计的主要收益 |
| 发布副本（cast 路径与被细切的 shard） | **每 rank 自己 shard 的 wire 字节**，不随 rollout 数增长 | **违反字面，有意为之**：pull 语义下版本 v 的读者可在 v+1 发布前任意时刻到达，cast 产物必须活一个版本（§3.4.2）。同精度且连续的 shard 为 0，被细切的部分由 `publish_copy_budget_mb` 封顶 |
| 布局转换产物（`ConcatenatedQKVSync` 的 cat） | 约 `0.15×M`，**随模型增长** | **违反字面，但今天就存在、非本设计引入**（§3.7.2.3） |
| colocate transfer 期的第二份参数 | `0.5×M`~`M`，**仅 transfer 期间** | **违反字面但不可消除**（源与目标必须同时在场）。eval 期 train 侧 offload 到 CPU，GPU 上只剩一份（§3.7.3） |
| colocate eval 期 train 权重的 CPU 落点 | `M`（每 rank 自己那份 shard） | **违反字面，但这是三家的最佳实践**：腾 GPU 给 KV cache。今天是 `N×M`（N 个 PS 副本），本设计降到 `M`（§3.7.3） |
| **ref 的 CPU 备份**（§3.10.2） | `M` pinned，**随模型增长** | **违反字面，但换掉了「ref 独立进程 + 独立读盘 + 跨角色精度判定」三件事**，净复杂度降低 |
| delta 传输的 pinned 快照 | `M` | 违反，所以只作显式 opt-in（§5.1） |

第三、四项是**既有成本而非新增成本**，把它们写明比假装约束被完美满足更有用。
验证项 1 据此分开记录「与模型无关的缓冲」和「随模型增长的常驻量」两个数字。

这不是把 PS 做小，而是**把 PS 从数据面彻底移出**：

```
今天      train GPU ──写满 N 份──▶ N×(CPU 全量副本) ──读──▶ rollout GPU
本设计    train GPU ◀──────────── rollout GPU 按需 READ ────────┘
          PSManager 只存 (version, key, shard) -> holder，零字节
```

| 维度 | 今天 | 本设计 |
|---|---|---|
| train 出口流量 | `N×M` | `M`（fp8 下 `M/2`） |
| 额外 GPU 权重存储 | 无 | 无（约束 b） |
| 额外 CPU 权重存储 | `N×M` | **0** |
| colocate 切换 | 全量 host→GPU 回灌 + NIXL 重握手 | 不搬权重，不重注册，不走 host |
| 传输提交 | 每 (key, shard) 一次 `initialize_xfer` | 握手期 prep 一次，每轮按 index 提交 |
| pull 阻塞 | push 期间全体 pull 阻塞 | 无锁，trainer 侧版本门闸 |
| FP8 | 不支持 | train 侧出 wire 格式，链路不感知精度 |
| 后端覆盖 | Megatron 与 FSDP2 各走 converter，PS 吸收两者差异 | 同样两个 converter，但**差异收敛为 §3.9 的四条**，数据面与 index 不感知后端 |

代价必须说在前面：**去掉 CPU 副本就去掉了吸收版本偏斜的缓冲区**，所以 trainer 的
optimizer step 与 rollout 的读取会直接冲突。两个决定性事实：一个训练步内权重被改写
`num_minibatches × ppo_epochs` 次（`ray_trainer.py:3061`），但 Megatron
`DistributedOptimizer`（默认开启）下 optimizer 写的是 **fp32 master shard**，
只有 param 回写那一瞬触碰 rollout 正在读的张量。**所以冲突窗口远比直觉窄**，
§3.4 把它统一成**一条规则**：rollout 永远读「发布缓冲」。cast 路径下该缓冲就是 cast 产物
（零额外显存，因为 cast 本来就要有落点），冲突归零。同精度路径下直读 trainer 参数，
用一个 per-chunk 的在飞读者计数器加写屏障放行，不需要租约表。

PS 的新定位：**不持有字节，但它是权重的版本与位置权威**，维护当前版本号、holder 表、
在飞读者计数、routing plan，以及合并进来的 staleness 账本。全是元数据，KB 级。
§3.2 同时给出控制面不成为新瓶颈的四条措施。

**holder 是对称的**：trainer 与 rollout 都可以是权威源，取决于谁当前持有最新版本
且处于唤醒状态。这不是为了优雅，而是正确性要求，因为
`is_rollout_mode_in_actor=True` 时 trainer 以 `meta` 模式注册、没有真实显存（§3.3.1）。

**两种部署模式的正确性今天都由暂停保证，差别在「谁决定何时暂停」**：colocate 由 trainer
整批切（全局 barrier），disaggregate 由每个 instance 自行择时——
但**自行择时不等于一定错开**，版本发布是同时事件，所以扩散仍必须由 index 强制
（§3.5.2 的 W），这正是 §3.7.2.4 要说清的一点。
所有精度组合在两种模式下都合法（§3.7.1 对比了 verl / SkyRL / miles
三家，它们都是时序互斥这个模型）。
**§3.4.2.1 的写屏障是把 disaggregate 的 pause 窗口收窄的余地，不是正确性前提**——
详见 §3.7.2.4，实现顺序是先保留暂停、后收窄。

**训练后端决定了 train dtype 的可取值，而 train dtype 决定走哪条路径**：Megatron 的
model params 是 bf16，FSDP2 的 `model_dtype` **默认 fp32 但常被改为 bf16**
（verl 8 个生产 example 在用）。fp32 下 FSDP2 永远产生 cast 产物、**与 optimizer 的冲突恒为零**
（但 wire buffer slot 的复用仍需在飞读者计数守住，§3.4.2 末段）；
bf16 下它落在同精度路径，且**版本级等待从优化升为正确性必需**（§3.8.0a）。
`ray_trainer.py:1962` 今天按后端硬分流而不读 `model_dtype`，这是落地项 9d-a 要修的。
完整矩阵见 §3.8.0（十一格十格支持），后端差异见 §3.9，三维展开见 §3.9.0。

**两处范围之外，但必须显式拒绝而非留白**：LoRA 的 adapter 传输（§3.11，
`merge=true` 支持、`merge=false` 拒绝，否则静默同步基座模型）；
训练中途改并行度（§3.10.4c，弹性只沿 DP 维）。

**本文主体只描述 actor 一个模型。** PPO 的 critic / ref / reward model
今天都不走 NIXL（各自读盘），直供后也不受影响。§3.10 给出多角色扩展分析，三条结论：
**holder 表必须加 role 维度**（正确性前提，否则 actor 的 v3 与 critic 的 c3 互相覆盖，
已并入落地项 8）；**ref 应实现为 actor 的 CPU 备份 tag 而非 holder**
（照 miles 的 `TensorBackuper`，顺带删掉一条跨角色精度匹配约束）；
**角色切换用按操作的二维懒加载**（照 SkyRL 的 `_ensure_on_gpu`，
区分 `need_model` 与 `need_optimizer`）。

**关于跨 stage 的资源弹性，必须先讲清一个取舍**：约束 (b) 与「把某个 role 缩容到 0
再恢复」**直接互斥**——role 归零后权重在集群里一份不剩，只能读盘。
PivotRL 的 Global Parameter Storage 正是为此保留一份 CPU 全模型副本，
实测换来 1.5–2× 端到端吞吐。所以本设计给出**两档形态**（§3.10.4a）：
档位 A（约束 b 下）支持 role 内实例增减但要求每个 role 常驻至少一个副本；
档位 B 放开约束 (b)、加回 `STORE_CPU` holder，支持完整资源迁移。
**两档共用同一套 holder 抽象与数据面**，差别只是一种 holder 枚举。

六个支柱：**PS 索引化 → trainer 直供 → 冲突隔离 → 全局去冗余规划 → 执行层批处理 → colocate 原地交接**，
FP8 作为贯穿全链路的 wire 格式。自动接入按约束 (a) 移出本设计，§8 只留接口预留说明。

---

## 1. 现状：三个量化事实

三条都从源码核实，是设计的出发点。

### 1.1 push 是 N 倍冗余的，而且冗余的是 CPU 副本

PS worker 按「rollout / actor / validate 涉及的不同节点」各起一个
（`ray_trainer.py:1980` 起的 `ps_node_ids` 收集），每个 worker 用
`NIXLSharding.default()` 注册**整张**张量（`hf_converter.py` 注释原文：
`PS holds full (non-TP-sharded) weights`）。

comm plan 对每个 (PS target, key) 把所需 shard 分给持有它的 train source
（`comm_plan.py::_make_comm_plan_generic`）。由于每个 PS worker 都需要全部 shard，
train rank 必须把自己那份 shard 写给**全部** `N` 个 PS worker
（`base_train_worker.py:156` 的 `for key ... for target_agent, target_client in zip(...)`）。

> **train 侧总出口 = `N × M`，CPU 总占用 = `N × M`。**

这正是约束 (b) 要消灭的东西。注意它不是「CPU 放不下一份」，是放不下 N 份。

pull 方向是干净的，而且**已经是 pull 模式**：comm plan 的 link priority 让每个 rollout rank
从本节点 PS worker 读自己的 TP shard。rollout 侧发起 READ 的骨架已存在
（`vllm_extension.py::nixl_pull_model_core`），这是 PSRL 相对 miles/verl 的结构性优势，必须保住。

### 1.2 热路径是逐张量提交的

`client_read` / `client_write` 对同一 `(operation, target, key)` 的**全部连续 shard 只提交
一个 NIXL 请求**（`merge_contiguous_xfer`，默认开启，`_post_contiguous_group`）；只有非连续
shard 才逐 shard 经临时缓冲提交。旧的 cached-xfer 队列与 `merge_and_finish_cached_xfer`
已移除，提交开销因此是 O(#keys) 而不是 O(#shards)。

WeightBridge 在 30B 上实测：逐张量的格式转换 + 控制面开销单项 590ms，超过 `3×T_min`。
这部分开销与模型大小无关、只与张量数量有关，MoE 下尤其致命。

**但要分清这 590ms 的构成**：§3.6.3 的实测显示**提交 API 本身不是瓶颈**
（16384 个 descriptor 的 `initialize_xfer` 只要 0.42ms）。
真正贵的是**逐张量的格式转换与 kernel launch**，
所以治它的是 §3.5.5 的 MoE 分组（14 万 key → 数百），
不是换一个提交 API。

而 NIXL 已有的正确原语 PSRL 没有用：`prep_xfer_dlist`（握手期准备一次）+
`make_prepped_xfer(local_indices, remote_indices)`（按下标组合、每轮提交），
以及 Nx5 strided descriptor（`(addr, len, dev_id, stride, count)`，原生表达非连续 shard）。

### 1.3 colocate 切换在经由 host 搬全量模型

`switch_to_trainer_mode` / `switch_to_rollout_mode`（`ray_trainer.py:2453` 起）每次切换做：
NIXL dereg → 释放显存 → 对侧分配显存 → **重新握手**（`nixl_send_local_info_to` +
`nixl_wait_for_update_infos` + 全局 broadcast client infos）→ **从 CPU PS 回灌全量模型**。

三项成本：全量 host→GPU 的 PCIe 拷贝、`O(#tensors)` 的 descriptor 重建与元数据广播、显存 pause/resume。
verl / SkyRL 的 colocate 快，是因为它们**根本不把权重挪出显存**。PSRL 慢的原因不在 NIXL，
在于把「同一块物理显存在 train 与 rollout 间交接」实现成了「经由 host 的一次完整往返」。

**切换期 `UPDATE_INFO` 的两条规则**（实测见 FINDINGS §13）：

1. **只发给读者。** 更新后的 client info 只有它的读者会消费。目的地由 comm plan 反查
   （含 peer-diffusion follower，见 `descriptor_readers_by_source`），没有读者时既不发广播
   也不等待 ack。发出的目的地要覆盖读者所在实例的**所有 rank**，因为 ack 是实例级的。
2. **不发派生数据。** descriptor 是由 `read_addrs` + stride + shard meta 派生的：direct supply
   的 pull 引擎在读者侧本地构造，info 只携带几何与地址。PS 的 pairwise 传输
   （`client_read`/`client_write`）仍按远端 descriptor 寻址，所以只在 PS 数据面上序列化它们。

这两条把切换期载荷从 `O(展开行数)` 降到 `O(shard 数)`，把 fan-out 从全 fleet 降到读者数。
3. **fan-out 由 train rank 的 agent 并行发出。** 读者数随 fleet 增长（万卡下 O(10^3–10^4)），
   单个 agent 串行发送会成为切换瓶颈。meta server 只生成发送计划（更新源序列化一次，
   每个目的地只带它读取的源子集与 conn-info），各 train rank 用自己的 NIXL agent 发送
   交错切片；读者仍只收到一个通知，ack 语义不变。发送覆盖数与计划不一致时 trainer 直接报错。

---

## 2. 外部方案调研：各自解决了哪一块

| 系统 | 核心机制 | 本设计采纳 | 不采纳的原因 |
|---|---|---|---|
| **WeightBridge** | 全局三阶段路由（receive / external-exchange / internal-exchange）。round-based 批处理。探测 `load_weights` 自动推导对应关系 | 三阶段去冗余规划、`T_min` 下界、round 批处理 | trainer 发起、固定 worker 集合，不含 pull 与弹性。自动接入按约束 (a) 排除 |
| **miles P2P** | train rank RDMA 直写 rollout 显存。CPU 侧建 sglang 模型副本以复用其 `weight_loader`。atomic group。bucket 512MB | atomic group 不变量、bucket 流水 | **CPU replica 违反约束 (b)**。单源瓶颈：rollout DP 一涨 AGST 近线性退化 |
| **verl** | `CheckpointEngineRegistry` 可插拔后端。NIXL 后端是 prev/next 中继链，双 buffer 边收边转发。`delta_sharded` 只传变化的 `(position, value)` | 双 buffer 流水思想。delta 作为跨 DC 正交选项 | 中继链是固定环拓扑，不感知 rollout 实际需求，也不是 pull。**其 2×bucket 常驻 buffer 要计入约束 (b) 预算** |
| **SkyRL v0.4** | `sharded_rdt`：rollout **pull**，consumer 在 init 期用 `FakeRDTTensor` **烘焙静态 pull plan**。producer sidecar 以 CUDA IPC 共享 trainer 显存。serve slot 跨 replica 共享 | **pull + 预烘焙 plan**、**sidecar 以 IPC 共享而非复制 trainer 显存**、slot 共享、FP8 wire | sidecar 的 ring buffer 仍是额外显存，本设计改为直读参数（§3.3） |
| **SkyRL（多角色放置）** | `colocate_all` / `colocate_policy_ref` 三档配置（默认全 colocate）。`_ensure_on_gpu(model, need_model, need_optimizer)` 按操作懒加载，`_gpu_state` 分别追踪 model 与 optimizer 是否在 GPU | **二维懒加载 + `colocation_group`**（§3.10.4e） | 无不采纳项，这是三家里多角色切换最完整的设计 |
| **miles（ref 形态）** | **没有独立 ref 进程**。ref 是 `TensorBackuper` 的一个 tag（CPU pinned 备份），用时 `restore("ref")` 换入。`_TensorBackuperMainCast` 让 actor 靠重放 cast + all-gather 重建而不备份。支持 `ref_update_interval` 周期性刷新 ref。**无 critic** | **ref 作为 actor 的备份 tag 而非 holder**（§3.10.2） | critic 缺失，PSRL 需要它；`MainCast` 的重建优化依赖 Megatron optimizer 内部结构，暂不采纳 |
| **PivotRL** | **Global Parameter Storage**：集群 CPU 内存池存全部模型（policy + GenRM）的参数，`Read`/`Write` 两个原语 + all-to-all 重分区。角色上下文按 stateless/stateful 分类处理。planner 用 `B = H·ΔT − φ` 决策 | **角色上下文分类**（§3.10.4b）、**shift 粒度限制在 DP 维**（§3.10.4c）、**benefit−penalty 决策框架**（§3.10.4d） | **GPS 本身违反约束 (b)**，它是 `num_models × M` 的 CPU 副本。但这正是「role 可缩容到 0 再恢复」的前提，所以本设计把它列为**档位 B 的架构选项**而非排除项（§3.10.4a） |

**与本设计最近的是 SkyRL 的 `sharded_rdt`**：它同样是 rollout 从 trainer 侧拉取、同样预烘焙 plan、
同样用 NIXL。它的 producer sidecar 以 **CUDA IPC 共享** trainer rank 的显存而非复制，
这正是约束 (b) 想要的形态。本设计在两点上更进一步：

1. SkyRL 仍有 K×chunk 的 publish ring 与 receive ring（文档明言「RDT receive buffers live
   **outside** `gpu_memory_utilization`」）。本设计让 rollout **直读 trainer 的参数张量本身**，
   bf16 路径下零额外 buffer（§3.3）。
2. SkyRL 的 `sharded_rdt` 与 FP8 serialized wire **互斥**，且不支持 colocate。本设计两者都要。

另外两条事实值得单列：

- **FP8 wire 有两种，都已在 SkyRL 打通。** blockwise（Hopper）：128×128 tile，FP32 或 power-of-2
  scale，`.weight_scale_inv`，vLLM `quant_method=fp8`。MXFP8（SM100+）：沿 K 方向 1×32 group，
  E8M0 偏置指数存 uint8，`.weight_scale`，走 compressed-tensors。TE 的 C++ kernel 跑在显存带宽
  （B200 约 4TB/s），torch 组合实现约 100GB/s 且作为 bitwise 一致性 oracle。
- **`fp8_param` 证明 FP8-native 训练可行**：TE 持有常驻 fp8 参数 + scale，参数 HBM 省 39%~42%。
  这同时意味着 FP8 train 形态下 trainer 本来就持有 fp8 权重，**直供给 rollout 不需要任何转换缓冲**。

---

## 3. 架构：PS 索引化 + trainer 直供

### 3.1 核心抽象：PS 不再持有字节

```
控制面  WeightIndex（PSManager 内，零字节）
          (role, version, key, shard) -> [holder, ...]，holder 带链路成本与租约
          role 维度见 §3.10：actor 与 critic 是两条独立版本流，ref/RM 冻结在 v0
          holder 种类：TRAIN_GPU | ROLLOUT_GPU | SELF（colocate 同卡），前两者地位对称
          职责：版本可见性、租约计数、holder 注册/失效、下发预编译 routing plan

数据面  纯 NIXL，端点就是两侧已有的权重张量
          train 侧：converter 产出的 unified_state_dict（已注册，今天就在注册）
          rollout 侧：convert_vllm_inplace 注册的 vLLM 运行时参数张量
          没有中间存储，没有 PS worker 进程
```

**holder 语义是对称的：trainer 不是唯一的权威源。**
初稿把「trainer 持有权威副本」当成不变量，这是错的。PSRL 的三种部署形态都会
让源发生转移：colocate 切换（trainer 睡下、rollout 持有）、
`is_rollout_mode_in_actor=True` 启动（trainer 以 `meta` 模式注册，没有真实显存，
见 §3.3.1）、弹性伸缩（instance 集合变化）。正确的表述是：

> **权重的权威副本在「当前持有最新版本且处于唤醒状态的那组 GPU」上，
> 它可能是 trainer，也可能是 rollout。index 的职责就是回答这个问题。**

所以 `TRAIN_GPU` 与 `ROLLOUT_GPU` 在 holder 表里**地位相同**，
区别只在链路成本与「是否为本版本的首个源」。§3.5.4 的 `from_trainer` 位
应改名 `is_origin`，定义为**「本版本中不从任何 peer 获得权重的节点」**，
涵盖三种来源：从 trainer 读、从 checkpoint 读（bootstrap 的 rollout 代表）、
自己就是 trainer。只有 `is_origin` 为真的 holder 才可被指派为 peer 源，
这样禁止链式的规则在 bootstrap 路径上也成立。
这个修正不改变数据面，但它让 bootstrap、colocate 切换、弹性伸缩三条路径
共用同一套 holder 语义，而不是各写一套。

holder 种类从四种缩到三种，`STORE_CPU` / `STORE_GPU` 按约束 (b) 删除。
**注意 `STORE_CPU` 是「删除」而非「不可能」**：它正是 PivotRL 的 GPS 形态，
放开约束 (b) 时加回来即可支持 role 缩容到 0 再恢复（§3.10.4a）。
holder 抽象为它预留了位置，**数据面与选源逻辑都不需要改**，
它只是「一类链路成本较高、但永不失效的 holder」。
`PSStorageWorker` 这个角色随之**整体消失**，`PSWorkerGroup` / `PSStoragePlan` 一并移除。

**CPU 侧权重归零的完整账**：今天持有 CPU 权重的只有 `PSStorageWorker`，三个入口全部删除：

| 入口 | 今天的作用 | 本设计 |
|---|---|---|
| `preload_checkpoint_to_cpu`（`ps_storage_worker.py:429`） | bootstrap 时读 checkpoint 进 CPU | 删除，改为 trainer 或 rollout 代表直接读盘（§3.3.1） |
| 接收 train 侧 `push_model` | 常驻 `N×M` 的 CPU 副本 | 删除，rollout 直读 trainer GPU |
| `initial_pull_from_ps` 灌给两侧 | 从 CPU 分发 | 删除，复用 `broadcast.py` 树形扩散，只换源头角色 |

所以**稳态下 CPU 侧零权重**。唯一的例外是 delta 传输的 pinned 快照（`M` 量级、
随模型增长），所以它只作显式 opt-in（§5.1）。
验证项 1 的判据正是「host 侧无权重常驻」。
但 **`broadcast_init` 的树形算法要保留**：今天唯一读 checkpoint 的就是 PS worker
（`preload_checkpoint_to_cpu`），rollout 用 `load_format="dummy"`
（`vllm_async_server.py:110`）、trainer 用 `init_model("empty")`（`ray_trainer.py:2116`），
所以删掉 PS worker 后必须新建 bootstrap 路径（详见 §3.3 末段），
做法是让 trainer rank 0 接替 PS rank 0 的读盘与扩散角色，复用 `broadcast.py`。

一次 pull 统一表述为：**rollout 向 index 声明「我要 version v 的这些 (key, shard)」，
index 返回按链路成本排序的 holder，执行层按预编译 plan 读取。** 三种场景是同一机制的三种解析结果：

- colocate → `SELF`，同一块显存，退化为零拷贝（§3.6）
- 分离部署 → `TRAIN_GPU`，跨节点 RDMA 直读 trainer 参数
- 大 fan-out → 追加 `PEER_ROLLOUT_GPU`，已拿到 v 的 rollout 分担出口（§3.5）

### 3.2 支柱一：PS 的新定位，它维护什么

**PS 不再是存储，它是权重的「版本与位置权威」。** 保留 `PSManager` 这个角色与进程，
删除的是 `PSStorageWorker`（数据面）。它维护的全部状态如下，**全是元数据，总量 KB 级**：

| 状态 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| **当前权重版本号** | `{role: v}`（**每角色一个**，非标量，见 §3.10.3），以及每个版本的发布时间戳 | trainer 发布时 | rollout、coordinator、router |
| **holder 位图** | 每 instance 一个 `Bitmap(num_chunks)` + 节点索引（**不按 key 展开**，见 §3.5.3c，64 instance 约 16KB） | trainer 注册、rollout 完成时搭车上报 | pull 选源 |
| **可读状态位** | 每个 trainer rank 的「权重注册有效且非中间态」。**§3.4.5 后它退化为 holder 身份的同义词**，不是独立状态 | trainer 注册/注销 holder 时 | pull 准入 |
| **在飞读者计数** | per-chunk 的整数计数器（**不是「版本×chunk 租约表」**，§3.4.2.1）。保护**参数内存**不被 optimizer 覆盖，仅同精度路径（8c）需要 | rollout 开始/结束读时 ±1 | trainer 回写前检查 |
| **routing plan** | 握手期编译的每 instance 三阶段 plan | planner 一次 | rollout 首次拉取后本地缓存 |
| **staleness 账本** | Reserve/Occupy/Consume 缓冲（**合并自现有 staleness controller**） | 现有逻辑 | 现有逻辑 |

**版本号的语义必须保持连续**，不能因为冲突而跳号：`maybe_delete_buffer(version_tag - 1)`
（`ps_manager.py:1203`）和 staleness 缓冲回收都按 `v-1` 索引，跳号会让缓冲泄漏。
所以 §3.4.3 的「超时强制推进」推进的是**写权限**，不是版本号。

它与现有 staleness controller 的 Reserve/Occupy/Consume 本质同源，
**应当合并进同一个版本权威**而不是两套状态互相同步（今天
`set_ps_model_version` 要向 coordinator 单向推送，就是两套状态的症状）。

`ps_mode` 枚举（`cpu` / `cpu_ref` / `nixl_cpu` / `nixl_gpu`）删除：四个值里 `nixl_gpu`
至今是 `NotImplementedError`，其余三个都在描述「副本放哪」，而本设计没有副本。

#### 3.2.1 控制面是本设计最可能的单点瓶颈，这是解法

索引化之后，PS 不再搬字节，但**所有 rollout 的所有拉取都要经过它做准入与解析**。
WeightBridge 明确指出其端到端时间的增长主要来自 pipeline bubble 与
**控制面消息数的平方增长**。今天 `PSManager` 是单个 Ray actor，而且
`@_state_locked` 把它的状态操作全序列化了。数十 instance × 数千 chunk
的租约消息会直接把它打满。

四条措施，按「消除 > 降频 > 加速」排序：

**(a) 消除：plan 不进热路径。** plan 在握手期编译、稳态下不变，rollout **本地缓存**，
拉取时**完全不查询 PS**。只有 holder 集合变化（instance 增减、trainer 重启）
才由 PS 主动失效通知。这一条把稳态下最大的消息源直接删掉。

**(b) 降频：租约按 (instance, version) 聚合，不按 chunk。**
一个 instance 一次预约覆盖它整轮要读的全部 chunk，释放同理。
消息量从 `O(instances × chunks)` 降到 `O(instances)`。
chunk 粒度只用于 **trainer 侧的本地放行判断**：trainer 自己知道哪些 chunk 已被读完
（由 rollout 完成时的一次聚合通知携带 bitmap），不需要每 chunk 一次往返。

**(c) 降频：版本发布用推送而非轮询。** 今天 rollout coordinator 有 `status_loop`
周期性拉状态。版本发布是低频事件（每步一次），应当由 PS 主动推给 coordinator
（现有 `set_ps_model_version` 已是此形态，保持），rollout 侧不轮询 PS。

**(d) 加速：热路径走 gRPC，与 Ray actor 状态分离。**
`psrl/grpc/ps_manager_service.py` 已有先例（SMG router 就是这么读 PS 状态的）。
准入与租约这两个高频读走 gRPC + 内存快照，**不进 `@_state_locked` 的临界区**。
写路径（版本发布、holder 变更）仍走 Ray actor 保持强一致。
这是读写分离，不是双写。

**残余风险**：PS 仍是单进程单点。本设计不引入 PS 高可用，但因为它只持有 KB 级元数据，
重建成本极低（trainer 与 rollout 重新注册 holder 即可），这与「PS 持有 N×M 字节」
时代的故障语义完全不同。应确保 PS 重启后能从两侧重建状态，而不是要求全局重启。

### 3.3 支柱二：trainer 直供，零额外缓冲

**rollout 直接 READ trainer 的参数张量。** 两侧的 NIXL 端点今天就已经存在：

- train 侧 `nixl_convert_params` 产出 `unified_state_dict`，是 FSDP2/Megatron 参数的 HF 视图，
  已经 `register_local_tensors`。
- rollout 侧 `convert_vllm_inplace` 注册的是 vLLM 运行时参数张量本身，`nixl_pull_model_core`
  的 `client_read` 已经是 RDMA 直写进去（绕过 `load_weights`）。

也就是说**今天的链路已经是「写进最终张量」，缺的只是把中间那跳 CPU 副本去掉**。
改动是把 pull 的 target 从 `ps_client_pull_name` 换成 train client，
并把 `NIXLClientType` 里的 `PS_FOR_PUSH` / `PS_FOR_PULL` 折叠掉。

四种 train 形态的缓冲账（后端相关，见 §3.9；**此表为 disaggregate**，
colocate 不需要 wire buffer，见 §3.7.2.2）：

| 后端 | train 形态 | rollout 读到的是 | 额外显存（每 train rank） |
|---|---|---|---|
| Megatron | FP8 train（TE `fp8_param`）+ FP8 rollout | TE 常驻 fp8 参数 + scale（转码后） | **0**，直读 |
| Megatron / FSDP2 | 同精度，shard 不被读方细切 | trainer 参数本身 | **0**，直读 |
| Megatron / FSDP2 | 同精度，shard 被读方细切 | 发布副本（连续拷贝） | 被细切 shard 的字节，`publish_copy_budget_mb` 封顶 |
| Megatron / FSDP2 | BF16 / fp32 train + BF16 / FP8 rollout | 发布副本（cast 产物） | 本 rank shard 的 wire 字节 |

另有一项与 dtype 无关的常驻开销：**布局转换产物**约 `0.15×M`，
只对走 `ConcatenatedQKVMapping` 的模型存在（§3.7.2.3）。

发布副本是**每 rank 自己 shard 的一份**，与 rollout 数无关，也不随 trainer 规模放大：
`P` 个 train rank 的副本合计恰好是一份 wire 格式的模型。副本必须活一个版本而不能流式复用，
理由见 §3.4.2：pull 语义下读者可以在下次发布前的任意时刻到达。
副本**在 disaggregate 下必须是 NIXL 注册过的**，因此常驻、按 dtype 合成一块注册，
注册区域数不随 key 数增长。**colocate 下不注册**，走 CUDA IPC（§3.7.2.1、§3.7.3）。

**「零额外 buffer」有一条前提必须说清**：直读参数要求**参数地址永久稳定**，
因而**禁止 trainer 做 CPU offload**。所以前两行的零 buffer 只适用于
「disaggregate + 同精度 + trainer 常驻」。
一旦 trainer 需要在 eval 时 offload 腾显存（§3.7.3），
就必须改读发布副本：**零 buffer 与可 offload 互斥**（§3.7.3.0a）。

> **为什么这比 SkyRL 的 sidecar 更省**：SkyRL 的 producer 要把 gather 结果 pack 进
> CUDA IPC ring slot，consumer 再 pull 进 receive ring，两侧各一份 K×chunk。
> 本设计在同精度直读路径下根本不 pack，参数张量自己就是传输源，descriptor 直接指向它。
> cast 路径下每 rank 一份 wire 格式的副本，rollout 侧零缓冲（直写引擎参数）。

**Megatron 不需要 gather，shard 代数已经处理。** `megatron_converter.py` 全文没有任何
`all_gather`：converter 为每个参数**声明** `NIXLSharding`（`get_sharding_for_param`），
把本地 TP/PP/EP 分片如实登记，重分片由 NIXL 的 shard 代数在传输时完成
（`find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh`）。PSRL 一直是 shard-to-shard
直传，所以直供下 rollout 按 shard 代数直接读 trainer 的本地分片，同样不需要 gather。

**FSDP2 同样不 gather**，但原因不同：`FSDPConverter` 用 `param.to_local()`
取本地 DTensor 分片并声明其 placement（`fsdp_converter.py:299`），
`fully_shard` 的 all-gather 只发生在 forward 期间、产出的是临时副本，
converter 看到的始终是持久的本地分片。两个后端在这一点上是对称的。
**代价在 §3.9.2.1**：FSDP2 声明的 shard 数是 `mesh.size()`，
纯 FSDP2 下等于 world_size，比 Megatron 的 `tp_size` 大一两个数量级，
这是独立于 MoE 的元数据放大源。

**PP 归属已经是启动期错误而非静默错误。** converter 只遍历本 rank 的 `named_parameters()`，
天然只登记自己持有的参数。comm plan 对「target 需要但无 source 提供」的 key 直接 assert
（`No available key {key} in source client groups`），所以 PP stage 覆盖不全在启动期暴露。

**但 `ParamSyncPlan` 的语义必须改造，这是一处真实工作量。** 现有 sync action 是
「push 前搬正、push 后还原」的成对钩子：`ZeroCenteredGammaSync.before_push` 做 `add_(1)`、
`after_push` 做 `sub_(1)`（`param_sync.py:65`）。直供模式下**没有「push 完成」这个时刻**，
rollout 可能在任意时间读取，所以 `after_push` 不能再被调用。这些 action 必须改成
**常驻正确**（`state_dict` 里那份始终是对外正确的值）。而 `ZeroCenteredGammaSync` 常驻 `+1`
会让 trainer 自己的 forward 读到错的 gamma，所以该参数需要一份独立副本。
`ZeroCenteredGammaSync` 涉及的参数量极小（layernorm gamma），
但 **`ConcatenatedQKVSync` 不是「不增不减」，它的语义变化更实质**：
今天它只需在 `before_push` 那一刻内容正确（可以 lazy），
直供下必须**常驻正确**，于是它进入版本发布的关键路径，
且作为「第三类字节」需要独立的冲突论证。详见 §3.7.2.3。
`DTypeCastSync`（`megatron_converter.py:177`）的 `.float()` 副本同理。

**这项改造是 Megatron 专有的**：FSDP2 路径构造的是空的 `ParamSyncPlan()`
（`engine_train_worker.py` 的 fsdp 分支），没有成对钩子，因此没有改造负担。
这是 FSDP2 侧工作量明显小于 Megatron 的一处具体原因。

**bootstrap 必须新建，不是「本来就有」。** 今天 rollout 用 `load_format="dummy"`
（`vllm_async_server.py:110`），trainer 用 `init_model("empty")`（`ray_trainer.py:2116`），
**唯一读 checkpoint 的是 PS worker**，再经 `broadcast_init` 树形扩散与 `initial_pull_from_ps`
灌给两侧。删掉 PS worker 后改为：**trainer rank 0 读 checkpoint，按 shard 代数分发**，
保留 `broadcast.py` 的树形算法、只换源头角色。不要让两侧各自读共享文件系统，
那会把 `broadcast_init` 当初要解决的「N 个 reader 打爆文件系统」搬回来。

#### 3.3.1 bootstrap 与弹性：三条路径共用 holder 语义

本节是 §3.1 holder 对称性的落地。三个场景，一套机制。

**(a) `is_rollout_mode_in_actor=True` 启动：rollout 自己做源。**

启动时全部 GPU 跑 rollout，trainer 以 `meta` 模式注册
（`ray_trainer.py:2183` 的 `actor_protocol_mode = "meta"`），
`meta_only=True` **跳过真实张量注册**（`client.py:503` 起），
配合 `init_model("empty")` 与 `nixl_sleep("meta")`，
此时 trainer **既无权重数据，也无可被 RDMA 读的注册内存**。

所以 bootstrap 绝不能假定 trainer 是源。做法：

```
rollout 每节点一个代表从 checkpoint 读自己需要的 shard
  -> 注册为 holder，版本号 = v0
其余 rollout 走 §3.5.2 的同版本 peer 扩散拿到 v0
  -> 复用 broadcast.py 的树形拓扑，源头从 PS rank 0 换成 rollout 代表
trainer 首次被唤醒时（switch_to_trainer_mode，且 trainer 从未持有过权重）
  -> 从 checkpoint 加载训练起点，不从 rollout pull
```

**为什么首次切回 trainer 要读盘而不是从 rollout pull**：cast 路径下 rollout 持有的是
fp8/fp4 权重，**不能作为 bf16 训练的起点**（反量化有损，且 optimizer master params
需要 fp32 精度）。今天这条链路能用 `pull_model`，是因为 PS 持有一份训练精度的副本,
直供后该副本不存在。首次切回是一次性事件，读盘可接受,
且这保证了训练起点的精度正确。

排除的两个候选：**每个 rollout 都读盘**会把 `broadcast_init` 当初要解决的
「N 个 reader 打爆文件系统」搬回来，1TB 下不可接受。
**临时唤醒 trainer 分发**与「全部 GPU 跑 rollout」直接冲突，那一刻显存可能不够。

##### 稳态 eval 之后切回 trainer：从 CPU reload，绝不能读 checkpoint

**上面那条只适用于「trainer 从未持有过权重」的首次唤醒。**
稳态下每次 eval 结束都会调同一个 `switch_to_trainer_mode`，
而那时 trainer 的 bf16 权重**在 CPU 上**（eval 开始时已 offload，§3.7.3）：

| | eval 期间位置 |
|---|---|
| trainer bf16 权重 | **CPU**（让位给 rollout 的 KV cache） |
| trainer optimizer state | **DRAM**（§3.7.3.1） |
| rollout fp8 权重 | GPU，eval 推理在用 |

所以稳态切回是 reload 而非读盘：

```
eval 结束 -> rollout pause_generation 完成
         -> rollout release_kv_cache_memory()        （只丢 KV cache）
         -> trainer reload 权重 (CPU -> GPU) + 重注册 NIXL
         -> trainer reload optimizer state (DRAM -> GPU)
         -> 恢复训练
```

**「从 CPU reload」与「从 checkpoint 读盘」是两件完全不同的事**：
前者拿回的是**最新训练版本**，后者会把训练进度**静默回退到上次 save 的版本**——
不报错，只让 reward 曲线莫名倒退，必须避免。

**`fp8 → bf16` 这个方向永远不会发生**：cast 是单向的（发布时 bf16 → fp8），
切回来不需要逆转它，因为 bf16 源在 CPU 上完好。

**判据不能用 `is_rollout_mode_in_actor`。** 该 flag 初始化为
`colocate_validate_and_train and val_before_train`（`ray_trainer.py:381`），
之后被两个 switch 函数来回翻转，**所以它区分不了「首次」与「稳态」**。
需要一个独立的一次性标记：**trainer 本进程是否曾持有过真实权重**。
首次唤醒时为假 → 读盘；之后恒为真 → 从 CPU reload。

**与今天实现的对比**：

| | 今天 | 本设计 |
|---|---|---|
| 切去 rollout | `nixl_sleep("full")` resize 到 0，**权重直接销毁** | offload 到 CPU，权重保留 |
| 切回 train | `nixl_wake_up()` + **`pull_model()` 从 CPU PS 回灌**（经 NIXL，走 comm plan） | **直接 H2D reload**，不经 PS、不走 NIXL |
| eval 期间 CPU | `N×M`（N 个 PS worker 各一份） | `M`（每 trainer rank 自己那份 shard） |
| eval 期间 GPU | 仅 rollout | 仅 rollout（**与今天相同**） |

**收益不在显存而在路径**：两者 eval 期间的 GPU 占用相同，
但今天要经由 `N` 份 PS 副本 + NIXL comm plan 回灌，
本设计是每个 rank 自己的 CPU↔GPU 直接搬运。
CPU 占用从 `N×M` 降到 `M`，且回灌不再需要控制面参与。

**(b) 非 colocate 启动：trainer 代表读盘，同样走扩散。**
与 (a) 结构相同，只是源头角色换成 trainer rank 0。
**两条路径共用一套代码**，差别只是「谁被选为读盘代表」,
由 index 根据「哪一侧当前是唤醒的」决定。

**(c) 弹性伸缩：三个必须处理的点。**

- **扩容进来的新 instance 冷启动**：从 index 查当前版本 → 作为排队者走 §3.5.2 解析
  （此时必然有 peer，因为其他 instance 已持有）→ 读完注册为 holder。
  逻辑上完全复用已有机制，无需新路径。
- **缩容时必须主动撤销 holder 资格**，不能依赖 §3.5.7 的被动摘除（靠 RDMA 失败）。
  被缩容的 instance 可能正作为 peer 服务他人，被动摘除会让读者先失败一次。
  协议与 §3.4.5 的 trainer sleep **完全对称**：
  先撤销 holder 资格、等在飞读者归零、再 sleep。
- **NIXL 层需要支持增量注册，这是弹性的前置阻塞项**，详见 §3.3.1d。
  **但阻塞点不在 NIXL——NIXL 的增量能力是齐备的，缺的是 PSRL 的握手协议与
  finest mesh 的全局耦合。**

##### 3.3.1d 「NIXL 增量注册」这个说法是错的：阻塞在 PSRL，不在 NIXL

前几轮把 9c 记成「NIXL 能力缺失」。**查过 `nixl.h` 与 Python 绑定后，
这个归因是错的——NIXL 侧该有的全有，而且还多给了一个我们没用的能力。**
真正的阻塞有三层，只有第一层是协议问题，后两层是架构耦合。

**先澄清 NIXL 侧的真实状态**（`nixl.h:489-616`，Python 绑定全部存在）：

| 能力 | C++ | Python | PSRL 用了吗 |
|---|---|---|---|
| 加载某个 peer 的元数据 | `loadRemoteMD` | `add_remote_agent` | **用了**（握手期） |
| 撤销某个 peer | `invalidateRemoteMD` | `remove_remote_agent` | **用了**（重注册、缩容吊销） |
| 撤销自己（向某个 peer） | `invalidateLocalMD` | `invalidate_local_metadata` | 只在 server `shutdown` 用 |
| **只发布部分 descriptor** | `getLocalPartialMD` / `sendLocalPartialMD` | `get_partial_agent_metadata` / `send_partial_agent_metadata` | **没用,而这是关键** |
| 查询某 peer 是否已就绪 | `checkRemoteMD` | `check_remote_metadata` | 用了 |

**实测（H20 + NIXL 1.5.0，单 agent 本地测量）**：

```
全量 MD blob 随注册区线性增长，约 406 B/region：
      1 region  ->       883 B
   1024 regions ->   416,221 B
   4096 regions -> 1,663,453 B        getLocalMD 本身耗时 1.0 ms

partial MD（4096 区中只发布 1 个）->  883 B，0.012 ms     压缩 1884x
conn-info-only（descs 为空）      ->  401 B
```

**`get_partial_agent_metadata` 是第三层问题的现成解**,见下。

---

**第一层：握手协议是一次性的,这一层是真的,但改动很小。**

`server.py:56` 与 `:90` 的 `_is_all_client_shardings_recved` / `_is_all_client_infos_recved`
一旦置真就直接 `return`,而 `ps_manager.py:907` 的 `nixl_protocol()` 是一条
**八步直线**：等全部 sharding → 建 unified → 广播 → 等全部 info → 建 plan → 广播 → 等 temp → 广播。

它的形状是「屏障同步」:每一步都要凑齐 `expected_agents` 才能往下走。
**新 instance 没有入口,不是因为缺 API,而是因为协议里没有「中途加入」这个状态。**
改成事件驱动的 join/leave 循环是直接的工作量,但**它只是必要条件,不是充分条件**——
下面两层才是架构问题。

---

**第二层(真正的阻塞)：finest shard mesh 是全局 lcm,新成员会改变所有人的分片。**

`make_unified_sharding`（`server.py:149`）对每个 key 取所有 client 的
`shard_mesh` 的**最小公倍数**（`nixl_spec.py:180` 的 `find_finest_shard_mesh`）,
然后让每个 client `refactor_based_on_finer_shard_mesh` 到这个全局粒度。

```
现状: train TP=8, rollout TP=4   -> finest = lcm(8,4) = 8
新加入一个 TP=3 的 rollout       -> finest = lcm(8,4,3) = 24
                                    ^^^^^^^^^^^^^^^^^^^^^
                     所有既有 client 的 descriptor 全部作废并需重算
```

**这才是「弹性伸缩」的硬阻塞:它不是注册问题,是分片代数问题。**
而且后果是全局的——一个新成员可以让 descriptor 数翻三倍,
进而让 §3.5.5 的元数据扇出和 §3.9.2.1 的 FSDP2 shard 放大一起恶化。

**处理方式:整除情形是快路径,非整除情形告警放行,但两者必须先分开判定。**

> **扩容进来的 instance,若其 `shard_mesh` 整除当前 finest mesh,
> lcm 不变、既有 descriptor 全部保持有效,这是零成本的快路径。
> 不整除时不拒绝,而是告警说明本次 join 会触发全局重分片及其代价。**

```
new_finest = lcm(current_finest, new_mesh)
if new_finest == current_finest:
    # 快路径:既有 client 完全不受影响,只需为新成员解析 plan
else:
    WARNING: 该并行度会把 finest mesh 从 {current} 抬到 {new_finest}
             （×{ratio}）,触发全体 {n} 个既有 client 重算 descriptor,
             预计新增 {m} 条 descriptor 与一次全局 plan 重编译。
             沿 DP 维扩缩（或选用整除当前 mesh 的并行度）可避免这一步。
```

**整除条件与 §3.10.4c 的「弹性只沿 DP 维」是同一件事的两种表述**——
沿 DP 维扩缩不改 `shard_mesh`（train 侧 sharding 本来就不含 DP 维,§3.5.7）,
所以常规弹性天然走快路径,告警只在真正改并行度时出现。

**代价比初稿估计的小,因为注册不受影响。**
`_record_region_registration`（`client.py:219`）以
**`untyped_storage()` 为单位**去重注册,一个 allocation 只注册一次,
与它被切成几个 shard 无关（`client.py:277` 的注释明确说明这是唯一安全的批处理）。
所以重分片**不触发 `register_memory`**,只重算 descriptor 与 plan:

| 受影响 | 是否需要 |
|---|---|
| `register_memory` / 显存 pin | **否**（按 allocation 去重,与 shard 数无关） |
| `get_local_sharded_tensors` 重切 + descriptor 重算 | 是,纯 CPU |
| comm plan 重编译 + 重广播 | 是 |
| **既有传输的正确性** | **不受影响**,前提是重分片在版本边界进行（见下） |

**但有三件事必须做对,否则「告警放行」会变成静默损坏:**

**(1) 非整除 join 必须在版本边界执行,不能在传输中途。**
重分片改变的是「同一块内存被切成几份」,在飞读者持有的是旧切分下的 descriptor。
这与 §3.7.3.0a 的陈旧 descriptor 完全同构,**所以必须复用同一套保护**:
撤销 holder → 等在飞读者归零 → **全体成员**按新 finest mesh 重切并重注册 →
`UPDATE_INFO` 广播 → 重编 plan。「同一地址的不同切分」对远端同样是陈旧描述,
所以每个既有成员都必须收到新切法并重新加载对端 info;按 §3.7.3.0a 的更正,
这由驱逐 + 重广播保证,不靠 `registration_epoch`（单边读无处校验 epoch）。
**现状**:`admit_joiners` 的 RESHARD 分支只把新切法发出去,既有成员的
`wait_for_server_sharding` 已缓存不再接收,所以该分支目前不可用,应在协议实现前拒绝 RESHARD。

**(2) 整除性是运行时约束,不只是性能问题——它可能让 join 直接失败。**
`get_local_sharded_tensors`（`nixl_spec.py:166`）要求
`tensor.shape[dim] % shard_count == 0`,**否则抛 `ValueError`**。
finest mesh 抬高后,这个检查是对**所有既有 key** 重新做的:

```
hidden=4096, 现状 finest=lcm(8,4)=8
  + 一个 TP=3 的 rollout -> finest=24 -> 4096 % 24 != 0 -> ValueError
  + 一个 TP=16 的 rollout -> finest=16 -> 4096 % 16 == 0 -> 正常
```

**所以非整除 join 有两种结局,必须分开**:
`new_finest` 能整除所有张量的相应维度 → 告警放行；
不能 → **这一个必须拒绝**,因为放行的结果不是「慢」而是
**既有 client 在重注册时集体抛异常,把一次扩容变成全链路故障**。

告警文案要区分这两者,不能都说「开销较大」。判据:

```
for key, shape in all_tensor_shapes:
    for dim, count in new_finest.items():
        if shape[dim] % count != 0:
            reject(f"TP={new_tp} 会把 {key} 的 dim{dim}({shape[dim]}) "
                   f"切成 {count} 份,不整除。这不是开销问题,是无法表达。")
```

**(3) 告警必须带可操作的替代项,否则它只是噪声。**
按 §3.10.4c,正确的替代是「沿 DP 维扩缩」或「选一个整除当前 finest 的并行度」,
告警里应当直接算出**最接近用户请求值的合法并行度**并给出来。

**这样处理的理由**:非整除重分片在数学上是正确的(更细的 mesh 仍能表达更粗的切分),
拒绝它等于把一个「慢但可行」的操作说成不可行。
但「可行」的前提是上面三条都落实——
**(2) 尤其关键,因为它是一个被整除性伪装成性能问题的正确性约束。**

---

**第二层附录:GPU 实测发现的两条握手时序硬约束。**

实现 9c-b/9c-c 时必须遵守,否则远端 prep 会失败或静默少搬数据
（`tests/nixl/conftest_agents.py` 已把两条编码进 fixture 并注明原因）：

**(i) 注册必须早于元数据交换。** MD blob 记录「哪个 backend 服务哪种内存类型」,
在任何 VRAM 注册之前交换的 blob **不含 VRAM backend**,
之后远端 `prep_xfer_dlist` 报 `no available backends for mem type 'VRAM_SEG'`。
**而本地 prep 仍然成功**,所以故障看起来像 descriptor 问题而不是握手顺序问题。

**(ii) 握手之后新注册的内存,远端看不见,必须重新发布 MD。** 实测:

```
注册 keepalive -> 交换 MD -> 再注册新 buffer -> 远端 prep
  -> failed to prepare the descriptors for any of the specified or potential backends
重新 add_remote_agent(fresh metadata) -> 远端 prep: OK
```

**这正是 9c-c 的机制来源**:增量 join 不能「注册完就用」,必须重新发布元数据。
`getLocalPartialMD` 的价值因此从「可选优化」变成「让这次重发布只花 ~400 字节
而不是全量 blob」——没有它,每次扩容都要全量重发一遍。

**(iii) 等待完成必须有 progress thread。** `enable_prog_thread=False` 下
传输提交成功但**永远停在 `PROC`**：没有线程推进 UCX worker。
benchmark 可以关掉它,只因为它只测提交开销、从不等完成。
**凡是要等 DONE 的代码路径都必须开着它。**

---

**第三层:全量 MD 广播让 join 的代价随集群规模平方增长。**

今天 `notify_all_client_infos_and_comm_plan`（`server.py:260`）
给每个 agent 发一份「它需要的全部 client_info」。
直供之后对端集合从「本节点一个 PS worker」变成「所有 trainer rank + 所有 peer」,
**每次 join 都要让 `O(agents)` 个既有成员各加载一次新成员的全量 MD**。

代入实测的 406 B/region：一个注册了 4096 个 region 的 trainer rank
其全量 MD 是 1.6MB,1024 个成员各加载一次就是 1.7GB 的扇出——**每次扩容一个实例**。

**解法:join 用 partial MD 分两段。**

```
第一段(所有人都要):conn-info-only blob,401 B
  新成员向 index 发布「我存在,我的连接信息是这个」
  index 转发给所有既有成员 -> 1024 × 401B = 410KB,可接受

第二段(只有真正的对端要):按 plan 解析出的那部分 descriptor
  新 rollout 的 plan 说它要从 trainer rank 7/19/33 读
  -> 只有这三个 rank 需要加载它的 descriptor MD
  -> 且只需 get_partial_agent_metadata(那几个 region) 而非全量
```

**这把 join 的代价从 `O(agents × full_MD)` 降到 `O(agents × 401B) + O(fanout × partial_MD)`。**
实测的 1884x 压缩正是为这个场景准备的。

**与 §3.5.5(c)「元数据走 RDMA 按需拉取」的关系**:两者方向一致但 partial MD 更简单——
不需要自己实现指针表与按需 RDMA 读,NIXL 已经把「只发布一部分」做好了。
**§3.5.5(c) 应当改为「优先用 `sendLocalPartialMD`,自研指针表降级为备选」**,
因为前者是上游维护的能力,后者是我们要自己维护的一层。

---

**地址正确性:缩容路径必须吊销 descriptor,不能只靠 holder 撤销。**

§3.4.5 的「撤销 holder 资格 → 等在飞读者归零 → sleep」保证了**不会有新的读发起**,
但**不保证远端忘掉了 descriptor**。远端此时仍持有可用的 MD,
若因 bug 或重试路径发起读,读到的是已释放的显存——
**这正是 §3.7.3.0a 的 `registration_epoch` 要防的那一类静默损坏,
只不过触发源从「地址变更」换成了「成员退出」。**

所以缩容协议是四步而非三步:

```
撤销 holder 资格（index 层,阻止新读者）
  -> 等在飞读者归零（§3.4.5,现有）
  -> 吊销 descriptor 并收齐确认（让所有持有者忘掉我的 descriptor）  ← 新增
  -> sleep / 释放显存
```

**第三步缺失的后果与 epoch 缺失完全同构**,所以验证项 5(c) 的「陈旧 descriptor
必须被拒绝」应当扩一个 case:成员退出后远端持有的 descriptor 也必须失效。

**落地时第三步的形态与这里的初稿不同,两点实测修正**（详见
`psrl/utils/nixl/leave_protocol.py` 与实现文档 E4）:

1. **不调 `invalidateLocalMD`,走 PSRL 自己的 notif 通道。**
   读 NIXL 源码后,`invalidateLocalMD(ip, port)` 的全部语义就是
   「向该 peer 发一条 `NIXLCOMM:INVL`,让它执行一次 `invalidateRemoteMD(我)`」——
   真正干活的永远是持有者本地那一次驱逐。它**不是广播**（广播形式只在
   etcd/tcpstore 中心存储下成立,而 PSRL 的 MD 是 out-of-band 分发的）、
   **需要对端监听端口**（`NIXLClientInfo` 不发布）、
   且**fire-and-forget 无法确认**。而第四步释放显存的安全性恰恰取决于这个确认,
   所以真正的缺口是确认而不是调用。
2. **第三步本身不需要 epoch 同构的额外机制。** 实测(H20)驱逐之后,
   用退出前的 prepped dlist 新建请求、以及 repost 退出前已建好的句柄,
   **都被 `NIXL_ERR_NOT_FOUND` 拒绝**,不会读到已释放显存——
   NIXL 对 remote section 持弱引用并在 `makeXferReq` / `postXferReq` /
   `getXferStatus` 三处校验 generation。所以这里只需保证「驱逐确实执行过」,
   而这正是 ack 的作用。

---

**小结:9c 的性质被改写了。**

| 层 | 初稿的判断 | 核实后 |
|---|---|---|
| NIXL 能力 | **缺失,硬阻塞** | **齐备**,且 partial MD 是我们没发现的解 |
| 握手协议 | — | 一次性屏障,需改事件驱动。**工作量,非难题** |
| **finest mesh 全局耦合** | 未识别 | 整除走快路径,非整除**告警放行**；但**整除性同时是一条运行时约束**,不可表达的那一类必须拒绝 |
| MD 扇出规模 | 未识别 | partial MD 两段式,实测 1884x 压缩 |
| 缩容的地址正确性 | 未识别 | 需吊销 descriptor 并收齐确认（实测:驱逐后陈旧 descriptor 已被 NIXL 拒绝,故不需要额外 epoch） |
| 重分片的地址正确性 | 未识别 | 必须在版本边界进行并推进 `epoch`,与陈旧 descriptor 同构 |

**所以 9c 不应再写成「等 NIXL 能力」,而是「改 PSRL 握手协议 + 把重分片做成一等路径」。**
它从「前置阻塞」降级为「有明确路径的工程项」。
**但「非整除 join 会不会把既有 client 切到抛异常」这条判定必须先做**——
它不是性能问题,放行的结果是整条链路故障而不是变慢。

### 3.4 支柱三：读写冲突，用「发布点」而非「互斥」解决

这是约束 (b) 引入的**唯一新增耦合**，也是本设计最需要讲清楚的一点。

今天 `PSManager` 是单写多读锁：push 独占，期间所有 pull 阻塞
（`_try_acquire_shared_pull_lock` 在 `_exclusive_push_locked` 时直接返回 False）。
这是一个伪装成锁的**全局同步点**，与「rollout 自行拉取」直接冲突。

#### 3.4.1 两个决定性的时序事实

**事实一：一个训练步内权重被改写多次，只有最后一次该被发布。**

`_run_scheduled_update`（`ray_trainer.py:3061`）按 `ppo_mini_batch_size` 与 `ppo_epochs`
构建 schedule 并**循环多次** `update_fn`，每次都是完整的 fwd/bwd/optimizer step。
`push_model` 只在 `finalize_step and is_final_step` 置位（`:3067`）。
所以权重被改写 `num_minibatches × ppo_epochs` 次（典型 4~16），
而 mini-batch 中间态**不是任何合法策略版本**，不该被 rollout 读到。

**事实二：Megatron `DistributedOptimizer` 下 optimizer 写的不是模型参数本身。**
（**这一条是 Megatron 专有的**，FSDP2 没有 master params，见 §3.9.3a。
FSDP2 的 `model_dtype=fp32`（默认）下不需要它，因为 cast 产物天然隔离；
但 **`model_dtype=bf16` + bf16 rollout 下 FSDP2 的整个 optimizer step 都在写参数**，
必须靠 §3.4.2 约束 2 的版本级等待，见 §3.8.0a。）

`use_distributed_optimizer: True` 是默认（`engine/megatron.yaml:32`）。此模式下存在
**两份**东西：fp32 master params（optimizer 的 shard）与 bf16 model params。
optimizer step 先更新自己的 fp32 master shard，再通过
`_copy_main_params_to_model_params` + param all-gather 落回 bf16 model params
（PSRL 自己就依赖这个结构，见 `megatron_optimizer.py::sync_master_params_from_model`
调用的 `_copy_model_params_to_main_params`）。

**这个事实把冲突窗口从「整个 optimizer step」缩小到「param 回写那一瞬」。**
optimizer 的数值计算（Adam 的 moment 更新、梯度裁剪、offload 往返）全都发生在
master shard 上，**完全不触碰 rollout 正在读的 bf16 参数**。

#### 3.4.2 回答「optimizer step 过长会不会拖长 rollout 读取延迟」

会，但不是以直觉的方式，而且**有办法把它降到接近零**。先把时间轴摆清楚：

```
一个训练步：
  [mb 1] fwd bwd opt_step   ← 权重被改写，中间态，不可读
  [mb 2] fwd bwd opt_step
   ...
  [mb k] fwd bwd opt_step   ← 最后一次，这之后才是合法版本 v
  ├── 发布点：v 可读 ────────────────────────────┐
  [下一步数据准备 + mb 1 的 fwd/bwd]              │ rollout 在这段窗口内读
  [下一步 mb 1 的 opt_step]  ← 窗口关闭，v 被破坏 ┘
```

**三种延迟来源，必须分开对待：**

| 来源 | 影响 | 是否可消除 |
|---|---|---|
| ① rollout 必须等到最后一个 mini-batch 完成才能开始读 | 发布点延迟，与 optimizer step **次数**成正比 | 不可消除（算法语义决定），但可用细粒度发布缓解 |
| ② rollout 读的过程中被 optimizer 的 param 回写打断 | 需要等待或重读 | **可消除**（见下） |
| ③ trainer 被在飞读者挡住，无法进入下一步 | 训练吞吐损失 | **cast 路径下不存在**（FSDP2 恒定如此），同精度路径下由写屏障收窄 |

**①是本质的，②和③才是设计问题。** 而 optimizer step「过长」主要影响的是①，
它的性质是「rollout 要多等一会儿才有新版本」，而不是「rollout 读得更慢」。
这是 staleness 增加，不是带宽损失。

**消除②和③：让被读的字节不是 optimizer 要写的字节。**

**统一解法：读者读的是「发布缓冲」，而发布缓冲是否是独立内存，取决于源头字节能否直接被读。**

**适用范围先划清：下面这张表只对 disaggregate 成立。**
colocate 下 rollout 已暂停，不存在「读的字节是否被改写」这个问题，所有组合走同一条路径。

| 情形 | 发布缓冲是什么 | 额外显存 | 冲突 |
|---|---|---|---|
| **cast 路径**（fp32→bf16、bf16→block FP8 等） | **发布副本**：本 rank 每个 shard 在 rollout dtype 下的连续拷贝，发布点刷新 | 本 rank shard 的 wire 字节（block FP8 是参数的一半，另加 1/16384 的 scale） | 无。optimizer 写参数，读者读副本 |
| **同精度且 shard 连续** | trainer 参数本身（直读） | **0** | 有，由 E1 版本级等待放行（§3.4.2.1） |
| **同精度但被读方切得更细** | 发布副本（连续拷贝） | 被切 shard 的字节，由 `publish_copy_budget_mb` 封顶，超出部分退回 span 读 | 无 |
| **布局转换产物**（§3.7.2.3） | `torch.cat` 的 QKV、`.float()` 的 fp32 副本 | 常驻，约 `0.15×M`（仅受影响的 key） | 无，但**必须在发布点前刷新** |

**为什么 cast 产物必须常驻一个版本，而不是 `2×chunk` 流式复用。** 流式 slot 只有在所有读者
跟随 trainer 的 cast 节拍读取时才可复用，这要求读者在一个有界窗口内同步拉取。
而 pull 语义下 instance 在发布后的任意时刻拉取（§3.5.1：greedy sync 天然错峰），
**版本 v 的读者可以一直到 v+1 发布前才出现**，所以 cast 产物的寿命就是一个版本。
流式方案只能二选一：要么让 trainer 的 cast 等最慢的读者（训练被拖住），
要么丢弃 slot 让迟到读者重拉（读不到 v）。常驻副本两者都不需要：
**副本在推送锁内刷新，读者在锁外随时读，不需要 ack bitmap，也不需要在飞读者计数守 slot**。

代价按 rank 计算，而不是按模型：每个 rank 只为自己的 shard 留副本。
例如 1TB bf16 MoE 分布在 `TP8 × EP64` 的 512 卡上，每卡 shard 约 2GB，block FP8 副本约 1GB。
它与 rollout 侧 KV cache 争的不是同一块显存（副本在 trainer 上），
真正受约束的是 trainer 的激活显存，可以按需用 `publish_copy_budget_mb` 封顶。

**副本及其派生张量必须按设备合并分配。** 副本在 colocate 切换的 sleep 阶段被释放
（把显存让给验证实例的 KV cache），唤醒时重新分配并注册；NIXL 的注册开销按 region
数计，而不是按 shard 数。因此 block FP8 的 scale 不能每权重一个 tensor：Qwen3-8B
每 rank 会从个位数 region 涨到 257 个，唤醒多花约 9 秒。所有 scale 合并进一块按设备
分配的 float32 池，各 shard 作为连续 view，注册回到 6 个 region、0.1 秒级。

**注册粒度取 allocator segment，不取单个 tensor 分配。** 对侧是 vLLM 引擎时，每个权重
（FP8 再加 scale）都是独立分配，一个实例数百个 region，注册因此主导切换。caching
allocator 的 segment 是整段映射的 cudaMalloc 分配，一个 NIXL region 可以覆盖其中的全部
张量：注册前用 `memory_snapshot()` 建立 allocation→segment 映射，同一 segment 只登记一次。
`read_bases` / IPC handle 仍按 allocation 保留，数据路径与显存占用不变。expandable
segments 的 VA 按需映射，必须关闭该模式（vLLM server 知道自己的 sleep 设置并显式传入）；
snapshot 无法定位的分配（CuMem pool、注册后才创建）逐分配回退。


**副本还顺带消除了源端 strided。** 同精度下若读方切得比 trainer 细（gen TP 大于 train TP 的
row-parallel 权重、EP 一侧与 TP 一侧的 expert），trainer 的 shard 对读方是逐行 strided 的，
descriptor 数随行数或 expert 数线性增长。发布副本是连续的，读方任意切法都只需一个 descriptor。
同一机制同时服务 cast 与重分片，**不是两套缓冲**。

**最后一列是本节的实际分流依据。** FSDP2 的参数是 fp32 而 rollout 永远不是 fp32，
所以 **FSDP2 的默认配置（`model_dtype=fp32`）只落在 cast 这一格**。
但 `model_dtype=bf16` 是常见配置（verl 8 个生产 example 在用），
此时 FSDP2 **也会走同精度路径，且比 Megatron 更依赖版本级等待**（§3.8.0a）。
这意味着下面的放行机制与写屏障是 **Megatron 专有的代码路径**，
不要把它做成两个后端共用的中间层。

**同精度路径不引入任何缓冲**，因为引入一份 fp8→fp8 的拷贝纯属浪费：
它消耗带宽与显存，只为了规避一个可以用放行机制解决的问题。
所以这条路径保留放行逻辑，但**把它收窄到最小**（见 §3.4.2.1）。

**两条配置期约束，属于正确性而非优化：**

1. **FP4 的量化源必须是 fp32 参数，不是 fp8 model params。**
   `fp8 → fp4` 是二次量化，TE 的 fp8 参数已损失精度，再降到 fp4 会放大误差。
   两个后端的 fp32 来源不同：
   - **Megatron**：由 §3.4.1 的事实二，master params 在 optimizer 手里且是 fp32。
     从它量化是单次量化，但这要求量化路径能访问 optimizer 状态
     （与写屏障读的是同一块结构，可行）。
   - **FSDP2**：model params 本身就是 fp32，**直接量化即可，无需触碰 optimizer**（§3.9.3b）。
2. **同精度路径下「chunk 偏移与写屏障互相加剧」的系统性解法。**

   问题根源：§3.5.2.3 的偏移让任意时刻都有某个 chunk 在被读，
   而「在回写 chunk i 之前等 chunk i 的读者清零」这个 per-chunk 写屏障
   会在每个 chunk 上都等一次，最坏情况把整个 optimizer step 变成逐 chunk 的串行等待。

   系统性解法是**把等待点从「per-chunk 回写前」提到「version 发布前」**，做一次性等待：

   ```
   optimizer step 完成，准备发布 v+1 之前：
     等待所有对版本 v 的在飞读者计数归零（一次性，不分 chunk）
     -> 此时所有 chunk 均无读者，trainer 参数内存可自由写
     -> 发布 v+1，holder 更新为 trainer(v+1)
   trainer 执行下一步 backward / optimizer step 期间：
     rollout 开始拉取 v+1，与 trainer 的 backward 完全并行
     （读 v+1 的参数，而 optimizer step 写的是 fp32 master shard，两者不冲突）
   ```

   这把 `O(chunks)` 个等待点合并成一个，与偏移的有无完全解耦。
   因此**偏移在 cast 路径和同精度路径均可启用**，不需要按路径关闭。
   等待时间是「读者完成最慢的那个 chunk」，这与偏移是否启用无关
   （偏移改变的是哪个读者在哪个 chunk 上，不改变最慢读者的完成时间）。

   这与 §3.4.2.1 的描述是一致的——只是把写屏障从「per-chunk 回写前」
   统一提到了「version 发布前」，粒度从 chunk 级变成了 version 级。

   **这条约束只在 Megatron + disaggregate + 同精度这一格里需要。**

**colocate 下同精度是完全合法的，见 §3.7.1 的时序互斥分析。**

这样做的收益是消掉了「三种形态」这个坏味道：
**配置维度从「dtype 组合」变成一个布尔量（源头是否产生新字节）**，
测试矩阵从 3 降到 2，而其中 cast 路径的分支几乎没有逻辑。
叠加后端维度后，真实的路径数依然是 2 而不是 4：FSDP2 只落在 cast 这一格（§3.9.0）。

#### 3.4.2.1 同精度路径的放行：窄到只剩一个写屏障

**适用范围：Megatron + disaggregate + 同精度。** 这是四格里唯一用得上写屏障的一格
（§3.9.0）。colocate 用时序互斥；FSDP2 的 fp32 默认配置走 cast 路径，
bf16 配置则同样落在这一格且**必需**（§3.8.0a）。

**且它是优化而非正确性前提**：disaggregate 今天也在 `pause_for_sync` 之后才拉取
（§3.7.2.4），所以直供可以先在现有暂停语义下落地。写屏障买到的是
「把 pause 窗口从整个 transfer 收窄到 commit group 切换瞬间」，
收益需用 §7 验证项 4 的等待时长分布衡量。

> **定位更正（v2，§3.4.2.2）**：publish 前的 drain 无法保证读取原子性。
> 一个在 optimizer 写期间开始、并在 publish 之前读完的读者，既不在 drain 的计数里，
> 也不会看到 INVALID，却读到了两个版本的混合字节。直供 + 同精度路径改用
> **write epoch 提交校验 + peer 副本修复**；本节描述的发布期屏障保留给
> cast 发布副本刷新与 PS 数据面。

同精度直读下，冲突窗口由两个已核实的事实决定，比直觉窄得多：

1. **Megatron `DistributedOptimizer`（默认开启）的 optimizer step 写的是 fp32 master
   shard，不是 bf16 model params。** 回写由 `_copy_main_params_to_model_params`
   加 param all-gather 完成（PSRL 自己依赖这个结构，见
   `megatron_optimizer.py::sync_master_params_from_model`）。
2. **optimizer offload 也只搬 master shard。** `offload_megatron_copy_params`
   （`megatron_utils.py:728`）只处理 `shard_fp32_from_float16_groups`,
   **完全不碰 model params**，所以 offload 与 RDMA 读天然无关。

于是放行只需要一个写屏障，不需要租约子系统：

```
trainer 在执行 param 回写之前：
  对即将回写的那段 chunk，检查「是否有在飞读者」
    无  -> 直接写（绝大多数情况，因为读者从发布点就开始读了）
    有  -> 等待，上限 conflict_wait_budget_ms
           超限则置该 chunk 为 INVALID 并继续写
             在飞读者收到 INVALID，放弃本轮，重拉更新版本
```

「在飞读者」是一个 **per-chunk 的计数器**，不是版本化租约表：
读者开始读时 +1、结束时 -1，由 index 维护但 trainer 可本地缓存。
**这比原方案的「版本 × chunk 租约表」简单一个量级**，
因为同精度路径下不需要回答「谁预约了哪个版本」,只需要回答「这段现在有人在读吗」。

#### 3.4.2.2 v2：乐观提交校验 + peer 副本修复

**适用范围：直供（rollout 读 train rank 的活参数）+ 同精度。** cast / FP8 rollout
读独立的发布副本，PS 数据面的读者读 PS 内存，两者都由发布锁保护，不需要本节机制。

##### 为什么「publish 前 drain」不够

冲突窗口不是 publish 事件，而是 **optimizer 写参数**的那一段。一个读者可能在写开始后
开始、在 publish 之前读完：

- 它结束时计数已归零，barrier 看不到它；
- publish 尚未发生，它读到的旧版本戳没有被标记 INVALID；
- 它提交的字节混合了 `v` 与 `v+1`，而任何版本戳校验都不会报错。

因此正确性不能依赖「写者等到没人读」，而应依赖 **读者能否证明自己读的是单一版本**。

##### 协议

状态由 coordinator（`PSManager`）维护：

```
version, write_epoch, state ∈ {OPEN, CLOSING, WRITING}
holders[v] = 已提交 v 的实例集合
inflight   = {instance: (admission_epoch, acked, preempt_epoch)}
```

1. **准入**：读者开始 pull 时以 `(instance, version)` 请求共享准入，记录
   `start_epoch`。`state != OPEN`（写者已宣布）时拒绝：读者继续用当前版本生成，
   发布后重新拉取 `v+1`。
2. **写者宣布**：在第一次 optimizer 参数写之前，rank 0 调用 `begin_write()`：
   `write_epoch += 1`、`state = CLOSING`，向所有在飞读者发一条控制面通知
   `PULL_PREEMPT{epoch, version, holders}`。`holders` 只包含该读者**同节点**的
   已提交实例。
3. **读者 ack**：读者在传输等待循环里轮询通知（0.5 ms 粒度）。一旦看到更新的 epoch：
   立即回 `PULL_PREEMPT_ACK{epoch}`——这是**控制面往返，不是数据 drain**；
   把已完成的 key 记为安全，把在飞/未开始的 key 记为 suspect，取消在飞请求。
4. **写者写**：写者最多等 `preempt_ack_timeout_ms`（默认 100 ms），无论 ack 是否齐全
   都开始写。**写者从不等待数据传输完成**：单边 READ 不会破坏写者内存，未 ack 的读者
   会在提交点被拒绝。
5. **读者修复**：suspect key 从通知里列出的同节点 holder 整 key 重读（走既有 peer
   路径，NVLink）。全部 key 都有 holder → 落地同一版本并提交；任一 key 无 holder →
   放弃本轮，保持暂停直到 `v+1` 发布后整轮重拉。
6. **提交**：`finish_shared_pull` 校验以下之一：
   - 读者实际读取的 epoch == 当前 `write_epoch`（写发生在读取之后）；
   - 读者在本 epoch 内 ack，且所有 suspect key 来自 certified holder。
   否则拒绝，实例不得服务其本地字节。
   记录到 staleness 账本的版本号取 **coordinator 的已发布版本**，不取 reader 缓存的
   per-shard 版本戳：rollout 侧戳只在重注册时刷新，publish 不会更新它，天生滞后于
   coordinator，用它做提交判据会误拒每一次稳态 pull。

##### 正确性论证

- **提交 accepted ⇒ 读到的字节全属一个版本。**
  - epoch 未变：读者在提交前完成所有传输；写者的 epoch 推进在 coordinator 的总序中
    晚于这次提交，而实际写入又在推进之后，所以全部读发生在写之前。
  - 修复路径：ack 前的完成读由写者等待 ack 保证在写之前；其余 key 从 holder 重读，
    holder 在写窗口内被 pin 住，字节是不可变的同版本副本。
- **写者写入期间存在并发读者 ⇒ 该读者无法提交旧版本。** epoch 已推进，且它无法同时
  满足 ack 与 holder 认证。
- **单边读不改变写者数据**：所以写者不需要等待任何传输完成。

##### 与 peer 扩散的关系

peer 机制同时是**修复基质**：已提交 `v` 的实例天然持有一份不可变副本，suspect key
可以整 key 从同节点 holder 读回，走 NVLink 而不是让 trainer 重新出口。没有 holder 的
key 只能等 `v+1`，这是可用性代价（一个 sync 周期），不是正确性代价。
holder 在 `WRITING` 期间被 pin，不能开始自己的下一次 pull，否则修复读到的字节会变。

##### 拓扑判据

是否需要本机制，判据是「rollout 是否读 trainer 的活参数」：

| 配置 | 需要 | 原因 |
|---|---|---|
| 直供 + 同精度 + rollout 跨节点 | 是 | 读活 bf16 参数 |
| 直供 + cast/FP8 | 否 | 读发布副本，刷新在发布锁内 |
| PS 数据面 | 否 | 读 PS 内存，发布锁保护 |
| `colocate_validate_and_train=True` 但 rollout 在别的节点 | **是** | 只有 validation 共卡，rollout 仍是异步读者 |

最后一行是 E1 原判据的错误来源：`colocate_validate_and_train` 描述的是 validation 的
放置，不是 rollout 是否暂停。v2 的判据在 `validate_config` 里解析为
`psrl.nixl.await_write_gate`。

##### 规模与延迟

- **稳态零成本**：没有在飞读者时 `begin_write` 不枚举 holder、不发通知，O(1)。
- **有读者时**：每读者一条通知 + 一条 ack；payload 只带同节点 holder
  （与集群规模无关）；holder 分组 O(holders + readers)。
- **写者延迟**：一次控制面 ack 往返，上限 `preempt_ack_timeout_ms`；与 pull 大小、
  慢读者、模型规模无关。
- **生成不暂停**：只有权重刷新路径被短暂关闭；长尾 rollout 继续用当前版本生成。

#### 3.4.3 取舍原则：等待的上限由算法目标决定，不由 infra 方便决定

**本节只保留原则，机制细节（`conflict_wait_budget_ms`、INVALID 状态码、
重拉协议）移入落地项 8c 的说明。** 理由是机制服务于一个窄条件
（同精度路径的等待），而原则独立于是否实现写屏障——
**它约束的是「遇到传输压力时能不能调大 staleness」，这个诱惑在任何实现下都存在。**

> **trainer 等待 rollout 读完，是用确定的吞吐换确定的 on-policy 程度。
> rollout 读旧版本，是用不确定的算法质量换吞吐。前者可测量可预算，后者不可。
> 所以默认取向是前者，但必须有界。**

三条推论：

1. **等待必须有硬上限。** 无界等待会让 trainer 被 rollout 拖死，
   这是用 infra 的确定性换了一个不确定的风险。上限的具体取值与超时行为见 8c。
2. **绝不为了避免等待而放宽 staleness。** 这是最容易犯的错：把 `staleness` 调大
   让 rollout 读旧版本，表面上消除了等待，实际是把 infra 问题转嫁成算法质量损失，
   且表现为 reward 曲线变差、极难归因。
3. **优先用细粒度发布换 on-policy，而不是用 staleness 换吞吐。** 见 §3.4.4。

**一个必须守住的边界**：超时后被打断的 rollout 应当**重拉更新的版本**，
而不是「继续用旧的」。因为重拉的是更新的权重，**这在算法上是改善而非退化**。
任何让它退回旧版本的实现都违反了上面那条原则。

#### 3.4.4 从算法角度：细粒度发布让 rollout 拿到更新的权重

延迟来源①（rollout 要等最后一个 mini-batch）是可以用**更细的发布粒度**缓解的，
而且这顺带降低 off-policy：

PSRL 已有 `fine_grain_overlap` 这个 StepStrategy（`psrl.fine_grain_overlap`），
它把训练步切成 chunk 以与 rollout 重叠。在 cast 路径下，可以在**每个 mini-batch 后**
都量化并发布一个中间版本，让 rollout 拿到更新的策略。

**但这必须由算法决定，infra 只提供开关。** mini-batch 中间态是否是合法采样策略，
取决于算法：GRPO/GSPO 的 importance sampling 修正项只需要知道「采样时的 logprob」，
所以中间态在数学上定义良好，但会增加策略方差。
建议做成 `publish_granularity: step | minibatch`，默认 `step`，由算法侧实验决定。

另外一条**免费的 on-policy 收益**：今天的链路要 `push(N×M)` 全部完成才登记版本，
直供下 trainer 登记后 rollout 立刻可读，省掉的正是 `N×M / B_train`。
N=32 时这不是小数。**少一跳就是少一份 staleness**，这是约束 (b) 带来的意外红利。

#### 3.4.5 内存注册有效性：留在 trainer 本地，不交给 PS

初稿要求 index 维护一个「trainer 权重可读」状态位，让 PS 感知 trainer 的内存生命周期。
**这是抽象泄漏**：PS 本该只管版本与位置，不该追踪别人何时释放内存。
重新评估后，这件事可以完全留在 trainer 本地，依据是三条已核实的事实：

1. **optimizer offload 不碰 model params。** `offload_megatron_copy_params`
   （`megatron_utils.py:728`）只搬 `shard_fp32_from_float16_groups`。
   所以 offload 根本不是风险来源，初稿把它列进去是多余的。
2. **真正会让 rkey 失效的只有 sleep / wake 路径**：`_sleep_megatron_model` 的
   `param_data.untyped_storage().resize_(0)` 与 `_sleep_fsdp_model` 的同类操作，
   之后必须 `nixl_register_after_wake_up` 重注册。
3. **这条路径今天只在 colocate 切换时触发**（`ray_trainer.py:2573` 的
   `nixl_wake_up`），而它本来就是一个**有明确起止的受控区间**，不是随时可能发生的事件。

因此正确的做法是**把窗口做成不可观测**，而不是把窗口状态发布出去：

```
trainer 侧（本地，无需 PS 参与）：
  进入 sleep 前：撤销自己作为 holder 的资格（一次 index 调用，与版本发布同一通道）
                 等待在飞读者归零（计数器已有，见 §3.4.2.1）
  resize / 重注册完成后：重新注册为 holder
```

**于是「内存注册有效」不再是一条需要持续维护的不变量，而是 holder 注册/注销的
一个自然结果**，不是 holder 就不会被指派，不会被指派就不会被读。
PS 只需要知道「谁是 holder」这一件事，而这是它本来就要维护的。

两点必须说清楚的边界：

- **这依赖「撤销 holder 资格」与「实际 resize」之间的严格顺序**。若顺序颠倒，
  窗口依然存在。所以撤销必须是**同步**调用并等待在飞读者归零，不能 fire-and-forget。
  代价是 colocate 切换多一次 round trip，而切换本来就是重量级操作，可忽略。
- **非 colocate 的稳态下这条路径根本不触发**，所以稳态零开销。

剩下的唯一不变量是**版本不被覆盖**，由 §3.4.2.1 的在飞读者计数器 + 写屏障保证。
从两条不变量减到一条，且剩下那条的实现是一个计数器而非一张表。

#### 3.4.6 不变量必须是运行时断言，不是文档要求

本文正确性大量依赖若干「必须遵守」的约定。文档里的要求无法阻止实现漂移，
所以下列四条必须落为**运行时断言**，在违反时立即失败而非静默错误：

| 不变量 | 断言点 | 违反后果（若无断言） |
|---|---|---|
| **atomic group 完整性** | 标记版本可见前，校验 group 内全部 key 已到达 | 引擎 loader 静默 no-op，权重半更新 |
| **版本相等性** | index 指派 peer 时，校验 `version(peer) == v` | 混合策略，算法侧不可见 |
| **peer 非链式** | index 指派时，校验 peer 的 `is_origin` 位为真（§3.1） | 长链延迟累积，上游覆盖导致下游读到脏数据 |
| **holder 即注册有效** | pull 准入时校验 holder 身份，sleep 前校验在飞读者归零 | RDMA 读到已释放物理页，静默数据损坏 |

前三条是纯元数据校验，开销是常数级，**应当常开**。
第四条的「在飞读者归零」在 colocate 切换路径上，频率低，也应常开。

另有两条属于配置期校验，不进热路径：
FP4 下 `is_weights_pre_processed` 必须为假（§3.8.1）、
block 对齐必须通过 `validate_fp8_block_shape`（§3.8）。

### 3.5 支柱四：全局去冗余规划，pull 语义下

`_make_comm_plan_generic` 是对每个 (target, key) 独立贪心（link priority 优先、
再按累计 volume 平衡）。结构性局限：逐 key 决策看不到全局负载，无法表达
「先去重、再在接收侧互换」这种两跳方案。而没有这个能力，trainer 直供就会撞上
**miles P2P 的单源瓶颈**：所有 rollout 都从同一批 trainer rank 读，rollout DP 一涨就线性退化
（WeightBridge 实测 16 replica 下差 34×）。

新 planner 产出 WeightBridge 的三阶段计划，建立在 PSRL 已有的 shard 代数上：

1. **receive**：两侧逻辑去重。trainer 侧多副本（DP replica）各领一份不相交切片，
   使 trainer 合计只发一份。rollout 侧多副本也各自只领一份切片。many-to-many 塌缩成 one-to-one。
2. **external-exchange**：rollout 节点间 RDMA 互换，使每节点恰好拿到它需要的全集一份。
3. **internal-exchange**：节点内走 NVLink 补齐各 worker 的副本。

目标是流量最优，对齐下界

```
T_min = max( M / B_train ,  max_v ( M_v / B_v ) )
```

`M` 为模型大小，`B_train` 为 trainer 聚合出口带宽，`M_v` / `B_v` 为 rollout 节点 `v` 的
去重后需求量与入口带宽。

**pull 语义的适配**（WeightBridge 没有覆盖的部分）：它的 plan 由 trainer 发起、
覆盖固定 worker 集合，而 PSRL 要的是每个 instance 自行拉取。做法是把 plan
**按 instance 切分、握手期编译一次**：

- planner 在握手期为每个 rollout instance 编译「它的三阶段 plan」，存在 index 里。
- instance 拉取时拿到自己那份独立执行，**不等全局 barrier**。
- external-exchange 阶段需要 peer，通过 index 查询「谁已持有 v」动态解析，
  而不是预设一个固定的交换伙伴。先到的 instance 从 trainer 读，后到的可以从 peer 读，
  **这同时就是 trainer 出口的天然减压**（SkyRL 已验证 NIXL 读单边非破坏、可被多方共享）。

这一条是 trainer 直供能否成立的关键：它把「所有 rollout 都压 trainer」变成
「trainer 只发一份，其余在 rollout 侧扩散」。

#### 3.5.1 greedy sync 与 peer 扩散的真实关系

先更正一个说法。初稿把 greedy 描述为「sync all instances ASAP，所有 instance 同时压 trainer」，
查 `greedy.py::_greedy_sync_and_migrate_loop` 后发现**它本身已带有自然的分批效果**：

- 循环每 `check_interval_in_ms`（默认 100ms）扫一轮，收集「版本落后于 PS」的 instance。
- `sync_with_ps` 一进入就把这批 instance 标记
  `instance_to_latest_stale_model_version = 当前版本`（`sync_and_migrate_mixin.py:148`）。
- 下一轮扫描时，**正在同步中的 instance 命中 `<=` 判断被跳过**（`greedy.py:48`）。
- 且 `partial_rollout.enable=False` 时还要求 `check_no_activate_tasks`，
  即只有手上没活的 instance 才会被纳入本轮。

所以真实行为是：**第一轮纳入「此刻恰好空闲且落后」的那些，后续轮次纳入陆续空闲的**。
instance 之间因负载与请求完成时间不同而天然错峰。**greedy 不等于同时发起。**

那么真正的风险是什么？不是「必然同时」，而是**「可能同时，而设计不能依赖它不同时」**：

- 版本发布是一个**同时到达**的事件（PS 推给 coordinator），所以第一轮扫描时
  所有空闲 instance 会在**同一轮**里被纳入同一个 `sync_with_ps` 调用。
- `sync_with_ps` 内部对这批是 `asyncio.gather` 并发 pull，没有串行化。
- 极端但现实的情况：训练步较长、rollout 已全部跑完在等新权重，此时发布版本
  → 全部空闲且落后 → 一轮全部纳入 → 并发压 trainer。
  **这恰好是 colocate eval 之后、以及 staleness=0 强同步配置下的常态。**

#### 3.5.2 解法：扩散由 index 强制，且 peer 必须是同版本

正确的分层是：**sync 策略决定「谁该更新」（算法语义），index 决定「从哪读」（传输语义）。
后者不能依赖前者的时序巧合。** 扩散由 index 在解析源时强制形成，无论上游是否同时发起。

**先确立一条硬约束（版本相等性）：**

> **只有当 peer 持有的版本与 trainer 当前发布版本严格相等时，它才可以作为读取源。**
> 不允许从「持有 v-1 的 peer」读，哪怕它在拓扑上更近。

理由有三层，从强到弱：

1. **算法正确性**：从 `v-1` 的 peer 读等价于主动引入一个单位的 staleness，
   而这个 staleness 是**传输层偷偷加的、算法侧不知情的**。它会污染
   importance sampling 的前提（算法以为采样策略是 v）。这与 §3.4.3 的取向一致：
   infra 绝不以降低 on-policy 为代价换吞吐。
2. **版本混杂风险**：若一个 instance 的部分 chunk 读自 v 的 trainer、部分读自 v-1 的 peer，
   得到的是**一个不存在的混合策略**，比单纯的 staleness 更糟，且无法被任何修正项补偿。
3. **实现简单性**：版本相等使 holder 位图可以**按版本整体失效**（见下），
   不需要维护「每 chunk 的版本号」这种细粒度状态。

落实到数据结构上，这条约束让实现**更简单而非更复杂**：

```
holder_bitmap 只为「当前发布版本 v」维护一份，版本推进即整体清空
  trainer 发布 v   -> 清空上一版位图，仅 trainer 为 holder
  instance 读完 v  -> 置位它的 bitmap，成为合格 peer
  trainer 发布 v+1 -> 整体清空，所有 instance 退回非 holder
```

副作用是**每个新版本的第一波必须从 trainer 读**，扩散不能跨版本累积。
这是正确性的必要代价，而 §3.5.2 的 W 机制正是为了让这第一波不至于压垮 trainer。

**选源规则（统一架构，同节点先落地）：**

```
index 解析每个 (instance, chunk) 的源，候选集 = {trainer} ∪ {version(peer) == v 的 holder}：
  1. 同节点 peer（NVLINK / PCIE）     -> 第一优先，不占 NIC
  2. trainer（IB）                    -> 在飞量 < W 时
  3. 跨节点 peer（IB）                -> 仅当 trainer 在飞量 >= W   [阶段二启用]
  4. 以上皆不可用                      -> 排队，等前面完成后重新解析
                                          （此时同节点或跨节点已出现合格 peer）
```

> **peer 读的粒度是读方的本地张量，不是全局 finest-mesh cell。**
> `make_unified_sharding` 把每个 key 切到 train ∪ gen 的 LCM mesh，这是**跨后端**重分片所需。
> 同版本、同 gen TP rank 的两个 rollout 持有逐字节相同的本地张量，二者之间无需重分片：
> 读方把两侧的连续段配对并合并，一个 key 只需一个 descriptor。
> 若沿用 finest cell，row-parallel key 的每个 cell 是 `hidden_size` 行的 strided 视图，
> 一个 Qwen3-8B gen client 的 peer 读需要约 59 万个 descriptor，提交开销压过传输本身。
>
> **去冗余按 key 分组，designate 按字节均衡轮转。** planner 为每个 `(节点, key, shard 集合)`
> 给出一组 wanter，排序时让上游字节在组员间均摊，于是节点入口分散到每个 instance 的 NIC 上，
> 而不是集中在一个代表的 NIC。每次 sync 只在**参与这次 sync 的 instance** 之间扩散，
> 首个参与者读上游，其余读它。上游读分成若干波，follower 读第 `w` 波时 designate 正在读第 `w+1` 波。

**阶段一只开 1、2、4，阶段二再开 3。** 架构上两者是同一条解析链路上的不同优先级,
区别仅在候选集过滤条件（`same_node(peer, requester)`）与排序权重，
**不是两套机制**。所以阶段二不需要改接口，只需放开一个开关并把节点身份
从 IP 换成 Ray node id（见 §3.5.3a）。

需要补的一处现有行为：`sync_with_ps` 目前在 pull 之前就对整批 instance
`pause_for_sync`（`sync_and_migrate_mixin.py:155`）。若扩散把一批 pull 拉长成几轮，
**先暂停的 instance 会空等更久**。所以暂停必须下移到「该 instance 真正要开始读」的时刻，
而不是整批统一暂停。这是一处实际改动，初稿没提。

#### 3.5.2.1 W 由什么决定

上一轮我把 W 说成「`W × 单读带宽 ≈ trainer 出口带宽`」，这个说法**不准确**，
因为它把 W 当成了限流阈值。想清楚之后，W 的本质是另一回事：

> **W 不是「trainer 能承受多少并发」，而是「在 trainer 只发一份数据的前提下，
> 第一波应该由几个 instance 来分摊接收」。它是去冗余的分组大小，不是限流阀门。**

这个区别很重要。NIXL 的读是**单边的**：rollout 发起 READ，trainer 侧是被动 RDMA target，
不消耗 trainer 的 CPU（`enable_prog_thread` 只影响发起方的进度推进）。
所以 trainer 并不会因为「并发读者多」而忙不过来，真正的约束是**它那张 NIC 的总带宽**。

于是 W 的取值由三个量决定，取其最小：

| 约束 | 表达式 | 含义 |
|---|---|---|
| ① NIC 不超卖 | `W ≤ B_egress / B_per_reader_useful` | 超过这个数，每个读者都变慢，总时间不变但 p99 变差 |
| ② 扩散收益存在 | `W < R` | `W ≥ R` 意味着全体直读源，扩散退化为无 |
| ③ 轮数可接受 | `W ≥ R^(1/k_max)` | 限制扩散轮数不超过 `k_max`，每轮都有同步与控制面开销 |
| ④ **QP 预算** | `W × ranks_per_reader ≤ QP_budget` | 单边读仍需 NIC 维护 QP 状态。万卡下 W=8 × 2000 rank = 1.6 万 QP，**可能超出 NIC 的 QP 缓存导致性能悬崖** |

**不要去测 QP_budget。** 不同 NIC 的 QP 缓存容量差异大，且难以静态确定。
正确做法是**让它被 W 的反馈信号自动兜住**：QP 超限表现为单读带宽的悬崖式下降，
而 W 的实测反馈（本节下文）本来就在监测单读带宽。
所以约束 ④ 不是一个要计算的公式，而是**「W 上调时必须验证带宽未劣化」这条规则的理由**。

**①是硬上限，③是软下限，实际取值在两者之间。**

更实用的是把 W 表述为「**trainer 出口的饱和点**」而非一个配置数字：

```
W_effective = max(1, floor(B_trainer_egress_available / B_per_reader_target))
  B_trainer_egress_available：trainer 节点 NIC 可用带宽
    注意要扣掉 trainer 自身的 collective（DP all-reduce / param all-gather）占用,
    这些与权重读取抢同一张 NIC
  B_per_reader_target：单个读者的目标带宽，由「可接受的单次拉取时长」反推
    = chunk_bytes / target_chunk_latency
```

三条实践建议：

1. **W 应当由 cost model 给出初始值，再用实测反馈收敛。** 不是静态配置，也不是「随便取一个然后调」。

   **初始值公式**（基于 Roofline + LogGP）：
   ```
   W_init = floor(B_NIC_node / B_per_reader_target)
   
   B_NIC_node      = NIC_count × NIC_line_Gbps × eff / 8      [GB/s]
   B_per_reader_target = chunk_bytes / T_rdma_target           [GB/s]
   T_rdma_target   = T_rdma_full / R   (每 instance 分到的时间预算)
   ```
   对 H20 8×200GbE 节点：B_NIC_node = 170 GB/s。
   若 T_rdma_target = 100ms，chunk = 512MB：B_per_reader = 5 GB/s → W_init = 34。
   但这超出 §3.5.1 的物理约束（W < R），取 `W = min(W_init, nodes_per_instance)` 即可。

   **实测收敛**：租约释放时捎带本轮完成时间（§3.2b 已规划），
   index 计算 `t_p99`；若 `t_p99 > T_rdma_target`，说明 NIC 超卖，下调 W。
   这把实测反馈限制在一个已知量（完成时间）上，而不是「实测所有参数再决定 W」。
2. **初始值取 `W = 节点内 instance 数`**，即「每个 rollout 节点派一个代表去 trainer 读」。
   这个默认值有清晰的物理含义：跨节点流量恰好是每节点一份（WeightBridge 的第二原则），
   节点内其余 instance 走 NVLink 从代表那里扩散。**这同时自动满足了阶段一只用同节点 peer 的设定。**

**不应该用来决定 W 的东西**：rollout instance 总数（那是 R，不是 W）、
GPU 数量、NIXL 线程数。这些与 trainer 的出口带宽没有直接关系。

#### 3.5.2.2 版本发布挂在 param all-gather 回调上（阶段上报已移出本期）

§3.5.2.1 把 W 定为「trainer 出口的饱和点」，并提了用实测完成时间反馈调整。
但纯反馈是**滞后**的：它要等一个版本的读取劣化之后才下调。
而 trainer 的 NIC 余量其实是**可预知**的，因为它由训练阶段决定：

| 训练阶段 | trainer NIC 占用 | W 应当 |
|---|---|---|
| forward | 低（仅 TP all-reduce，多在 NVLink 内） | 放大 |
| backward | **高**（grad reduce-scatter 跨节点） | 趋近下限 |
| optimizer 数值计算 | 低（master shard 本地计算，offload 走 PCIe 不走 NIC） | 放大 |
| param 回写 + all-gather | **高** | 趋近下限，且此时写屏障生效（§3.4.2.1） |

**但这张表不该落成「持续上报训练阶段」那套机制。** 初稿同时写了四件事
（阶段枚举上报 + DP rank 0 限制 + TTL + stage_factor 查找表），
又紧接着说「阶段分档偏粗，更准确的信号是 NCCL 在飞状态」——
**即它自己否定了刚提出的机制**。这是过度设计，收敛成两条：

**(a) 只做一件事：把版本发布挂在 param all-gather 完成的回调上。**

这一条**取代**上面整张表的大部分价值，因为它直接把第一波读取放进 NIC 的空档：

```
optimizer step 完成（含 param all-gather）
  -> on_train_step_end 回调触发
  -> 此刻：权重刚就位（staleness 最小）且 NIC 刚空闲（带宽最大）
  -> 发布版本 v
```

**已查证可行**：Megatron-Bridge 的 `on_train_step_end` 在 `VALID_EVENTS` 中
（`callbacks.py`），于 `train.py:518` 的 optimizer step 后触发，
`CallbackContext` 携带 `optimizer` 与 `model`。注册方式是向 `pretrain()` 传入
`CallbackManager`，PSRL 的 Megatron worker 已持有该调用链。
成本是**一个回调**，不是一套上报协议。

**(b) W 只保留实测反馈，不做阶段前馈。** 理由是：

- 有了 (a) 之后，第一波本来就落在 NIC 空闲窗口，**前馈要解决的主要问题已经消失**。
- `overlap_grad_reduce` 开启时 grad reduce 与 backward 交织，
  **不存在「干净的 bwd 阶段」**，所以阶段枚举本身就不准。
- 真正准确的信号（NCCL 在飞状态）需要 hook Megatron 的 `grad_sync` / `param_sync`，
  这比阶段上报更侵入，而收益只在「版本发布之后、下一步 backward 期间
  仍有 instance 在读」这个次要场景。
- 实测反馈（租约释放时搭车上报耗时，§3.5.2.1）是**零额外消息**的，
  而阶段上报是 `O(minibatches)` 条消息加 TTL 加 DP rank 0 的额外约束。

**所以阶段标记整体移出本期**，记入 §5.1 待优化清单。
若实测发现「发布后的持续读取与 backward 的 NCCL 争带宽」是真实瓶颈，
再按 NCCL 在飞状态（而非训练阶段枚举）实现一次。

这样 W 的机制只剩一条：**初始值 = 节点内 instance 数，按实测完成时间反馈调整**，
约束是 §3.5.2.1 的四条上限。没有阶段表、没有 TTL、没有上报协议。

#### 3.5.2.3 chunk 起始偏移错开，让扩散在第一波内部发生

§3.5.2 的 W 机制让第一波只有 `W` 个 instance 直读 trainer，其余排队。
但若这 `W` 个 instance 都**从 chunk 0 开始按相同顺序推进**，它们会同时持有 chunk 0、
同时持有 chunk 1，排队者要等到某个 instance **整轮读完**才能找到 peer。

改进是零成本的：**让 instance `i` 从 chunk `(i × num_chunks / R) mod num_chunks` 开始，
环形推进**。于是：

- 第一波的 `W` 个 instance 在任意时刻持有的是**不同的 chunk 集合**。
- 排队者一旦被放行，立刻就能为它需要的大部分 chunk 找到合格 peer，
  **不必等任何 instance 整轮完成**，扩散在第一波内部就开始了。
- 对 trainer 出口也更友好：并发读者命中的是不同地址区间，减少同一块内存的争用。

三点约束：

1. **偏移必须与 commit group 边界对齐**（§3.6.1），不能把一个 group 切在环形起点，
   否则该 group 的可见性判定要跨环回绕，变复杂。
2. **偏移是确定性的**（由 instance 序号导出），不是随机，这样 index 可以在不通信的
   情况下预测谁会先持有哪些 chunk，用于选源。
3. 这项优化**只在 `R > W` 时有意义**，`R <= W` 时全体直读 trainer，无需扩散。

### 3.5.3 拓扑感知选源：可行性、开销与可扩展性

**结论：可行，但必须限定在「节点级、离散分级、不做实时带宽估计」这个范围内。** 逐项评估。

**(a) 拓扑数据已经有了，但精度有限。**

`network_topology.py` 已实现 `GLOBAL_TOPOLOGY`，按 `(node_ip, gpu_id)` 判定链路类型并给出
`get_link_priority()`（`LOCAL > NVLINK > IB > PCIE > IB_PCIE > ETH`），comm plan 今天就在用它。
所以**不需要新建拓扑子系统**，但必须认清两个局限：

- **带宽是硬编码常量**（`_init_default_topology` 注释写明是「8*H20 机器的默认配置」）。
  它能可靠回答「同节点还是跨节点」，**不能**可靠回答「两个跨节点 peer 哪个更快」。
- **节点身份依赖 `LOCAL_NODE_IP` 或 `gethostbyname`**，而 `LOCAL_NODE_IP` 在仓库里
  只有定义、没有任何地方设置。多网卡 / 容器网络下 `gethostbyname` 可能返回与 RDMA 实际
  走的网卡无关的地址，使同节点被误判为跨节点（后果是选源变差，不是出错）。
  **阶段一就应把节点身份换成 Ray node id**（`get_node_id()` 已有，比 IP 可靠），
  这也是阶段二开启跨节点 peer 的前置条件。

**(b) PS 侧开销：holder 位图，不按 key 展开。**

朴素的 `(v, key, shard) -> [holder]` 在 1T MoE + 数十 instance 下是千万级条目，不可接受。
正确结构是按 instance 聚合的位图，且因 §3.5.2 的版本相等性约束，**只需为当前版本维护一份**：

```
holder_bitmap : dict[instance_id, Bitmap(num_chunks)]   # 仅当前版本 v
node_index    : dict[node_id, set[instance_id]]          # 同节点 peer 查询
in_flight     : int                                      # trainer 当前在飞读者数，用于 W
```

量化：1TB 模型 / 512MB bucket ≈ 2000 bucket，位图 250 字节／instance，
64 instance 共 **16KB**。加节点索引与在飞计数，PS 的选源状态总量在 **几十 KB 量级**，
与模型大小几乎无关（只随 `M / chunk_size` 线性）。

查询代价：「同节点是否有合格 peer 持有 chunk i」= 对该节点 instance 集合做位图 OR，
`O(同节点 instance 数)` 的位运算，微秒级，**不遍历 key**。
版本推进时整体清空，`O(R)`。

**(c) 消息开销：不新增往返。**

holder 更新**搭在已有的租约释放调用上**携带 bitmap（§3.2 的按 instance 聚合）。
稳态消息量 `O(instances)` 每版本，与 chunk 数、key 数无关，仍在 §3.2 承诺的量级内。

**(d) 可扩展性评估。**

| 规模维度 | 影响 | 失效点 |
|---|---|---|
| instance 数 R | 位图 `O(R)`、查询 `O(R_同节点)`、消息 `O(R)` | R 到数百时瓶颈是 PS 单进程消息处理，与拓扑选源无关（§3.2 的控制面问题） |
| 模型大小 M | 位图随 `M/chunk_size` 线性 | 10TB 模型下 2.5KB／instance，仍可忽略 |
| chunk 粒度 | 变细则位图变大、放行更精细 | chunk < 64MB 时位图开销与收益不再划算 |
| 节点数 | 节点索引 `O(nodes)` | 无实际限制 |

**真正的可扩展性风险不是数据结构，是两个语义问题：**

1. **peer 供给会抢 rollout 自己的推理资源。** 一个 instance 为他人提供读取时，
   NIXL progress thread 占 CPU、RDMA 占它自己的 NIC，而推理是延迟敏感的。
   必须有上限：每个 peer 同时最多服务 `P` 个读者（建议 1~2），超出则不再被指派。
   **colocate 模式下直接禁用 peer 供给**（它与 eval 争同一张卡）。
2. **「已持有」不等于「可安全被读」——但 peer 这一侧其实没有这个问题。**
   初稿担心「peer 的 wire buffer 下一轮会被覆盖」，**核实后发现 peer 根本不从
   wire buffer 供给**：rollout 侧是 NIXL 直写 `convert_vllm_inplace` 注册的
   运行时张量（§8 的渐进可用分析已确认这条路径），**没有接收侧的中转 buffer**。
   所以 peer 作为源时供的是它**已落地的参数**，而那块内存在下一个版本到来前不会变。
   **这是 pull 语义的一个意外红利**：发送侧(trainer wire buffer)需要 ack 守护，
   而 peer 侧天然安全，因为它供的不是流式缓冲而是稳定的落地参数。
   仍然成立的约束只有 §3.5.2 的版本相等性（保证内容是 v），
   以及新版本到来时 peer 要先退出 holder 集合再覆盖自己的参数——
   后者由 holder 位图「版本推进即整体清空」自动保证，不需要额外机制。

**(e) 落地建议：两阶段，架构一致。**

- **阶段一**：候选集过滤为「同节点 + 版本等于 v」，开启 trainer 并发上限 W
  （默认取每节点一个代表，见 §3.5.2.1），节点身份换成 Ray node id。
  这已拿到两个最确定的收益：同节点读走 NVLink 不占 NIC、第一波扇出被 W 收窄。
- **阶段二**：放开跨节点 peer 作为 trainer 饱和后的兜底。
  **架构上只是去掉候选集的 `same_node` 过滤并加一档排序权重，不是新机制。**
  收益随 R 增大才显现，应在实测到 trainer 出口饱和后再开。

**不建议做的**：实时带宽探测、基于 RTT 的动态选源、全局最优流量规划。
带宽是常量，这类优化会建立在不可靠输入上，且让 PS 承担它不该有的实时性要求。

#### 3.5.3.1 rail 拓扑感知

万卡集群普遍采用多 NIC / 多轨道（rail）组网：每节点有多张 HCA，
每张接入不同的网络平面，同 rail 的跨节点流量不共享交换机，带宽更可预测。

`NetworkTopology` 目前只有 `(node_ip, gpu_id)` 两个维度，没有 rail 概念。
UCX 有 multi-rail 支持（`UCX_IB_DEVICES`），所以数据面本身可能已受益，
但选源决策看不到它。

落地方案（1-2 天工作量）：

1. `NetworkTopology.register_client` 额外接收 `rail_id`（整数），
   由 worker 启动时通过环境变量 `PSRL_RAIL_ID` 或 `ibstat` 查询上报。
2. `get_link_priority` 新增一档 `IB_SAME_RAIL`，优先级介于 `NVLINK` 与 `IB` 之间：
   同节点 > 同 rail 跨节点 > 跨 rail 跨节点。
3. 三阶段 planner 的 source 侧分配优先选**同 rail 上有副本的 trainer rank**，
   让 W 个并发读者里同 rail 流量天然占优。

收益在万卡下显现，小规模部署下无影响。

#### 3.5.4 禁止链式 peer

peer 必须是**直接从 trainer 读到的那一批**。holder 位图额外记一位
「来源是 trainer」，只有该位为真的 instance 才可被指派为源。

理由：若允许「A 读 B，而 B 自己读的是 C」，在 W 较小时会形成长链，
延迟逐跳累积，且一旦链中某个 instance 开始下一轮覆盖自己的 buffer，
下游全部失效。限制为单跳后，最坏延迟是 `trainer→peer` 加 `peer→self` 两段，
与 WeightBridge 的 external-exchange 完全一致，不会退化。

### 3.5.5 元数据平面必须先去「按 key 展开」

**这是大 MoE 与万卡场景的前置阻塞项，优先级高于本文其余所有性能优化。**

PSRL 的 canonical 命名空间是 HF 展开形式，MoE 的每个 expert 的每个投影都是独立 key。
Kimi-K2（1T，384 experts，61 层）量级：

| 形式 | 张量数 |
|---|---|
| vLLM fused（`w13_weight` / `w2_weight` per layer） | **122** |
| HF 展开（gate / up / down per expert） | **70,272** |
| HF 展开 + fp8 scale 成对 | **140,544** |

而握手期 `send_local_info`（`client.py:861`）把整个 `client_info`（含全部
`tensor_infos`）pickle 后经 `agent.send_notif` 发出，server 再按 agent 扇出
（`server.py:257`）。按每 key 约 400 字节估，**单个 client_info 就是 56MB**，
1024 client 规模下是数十 GB 的 pickle 流量与等量反序列化 CPU。
`comm_plan` 同样按 key 展开（`comm_plan.py:22`），量级相同。

**注意直供会加剧而非缓解这一点**：rollout 的对端从「本节点一个 PS worker」
变成「所有 trainer rank」，peer 读取又让 rollout 之间互为对端，
`_get_relevant_client_names_for_agent` 的过滤效果变差。

三条措施：

**(a) canonical 层面做 MoE 分组（根因治理，必做）。**
把 `experts.{i}.gate_proj.weight` 合并为 `experts.gate_proj.weight` 一个 key，
expert 维度表达为 `NIXLSharding` 的一个分片维度（`shard_mesh` 是
`OrderedDict[dim, count]`，本来就支持多维）。Kimi-K2 的 key 数从 14 万降到数百。

> **分组只治元数据，不治 descriptor。** descriptor 数由「源端每个 cell 是否连续」决定。
> 一侧按整 expert 持有（EP）、另一侧切 expert 内部的 projection（TP）时，finest mesh 是
> `{0: E, 1|2: tp}`，源端每个 expert 的每个 cell 都是 strided，descriptor 数随 expert 数线性增长。
> **源端发布副本**消除它：train rank 为「被读方切得比自己细」的 shard 保留一份连续副本，
> 在发布点（推送锁内）刷新，读方任意切法都只需一个 descriptor。副本显存由
> `publish_copy_budget_mb` 封顶，超出预算的 shard 改为读方 span 读整段再本地 scatter。
> cast 路径与 block FP8 复用同一机制：发布副本本身就是 cast 产物（§3.4.2、§3.8）。

这更贴近引擎真实布局：vLLM 的 fused loader 本来就要 `w13_weight` 这个 3D 张量，
今天 PSRL 是主动把它拆开（`vllm_converter.py:400` 起的
`FUSED_MOE_W13_SPLIT`），拆开反而是额外工作。
风险点是 `ETP != 1` 时 expert 内还有 TP 切分，shard mesh 成为
`{0: EP, 1: ETP}` 两维，需验证 `find_finest_shard_mesh` 的 lcm 逻辑在两维上正确。

两处连带影响，都由 §3.6.1 的 chunk / commit group 分离自然吸收：

- **commit group 粒度变粗**。vLLM 的 fused loader 要求整个 `w13_weight` 就位，
  所以 group 从「一个 expert 的三个投影」变成「整个 fused 张量」。
  因为 chunk 与 commit group 已解耦，一个 group 跨多个 chunk 是常态而非特例。
- **单 key 体积变得极大**。Kimi-K2 单层 `w13_weight` 在 fp8 下约 11GB，
  所以 chunk 必须能切开单个 key，这正是 §3.6.1 给 chunk 的定义。

**(b) 描述符用 Nx5 strided 表达。** expert 维度分片通常是规则跨步的，
一条 strided desc 替代 `count` 条普通 desc。
**已实测**：descriptor **列表**压缩 16384:1，prep 开销降到与 count 无关（0.002ms）。
**但提交索引不压缩**——strided 行在 prepped handle 内部展开为 `count` 个可索引条目，
提交时仍要给 `count` 个索引（§3.6.3 的更正）。所以收益在元数据与 prep，不在提交。
**前提是 NIXL ≥ 1.5.0**（§3.6.3）。这与 §3.6 的 strided 改造是同一件事，
此处确认它还有元数据压缩收益。

**(c) 万卡规模：优先用 NIXL 的 partial MD，自研指针表降为备选。**
初稿写的是「每个 client 把 `client_info` 注册为可 RDMA 读的内存，
server 只分发指针表，需要时按需读」。**核实后发现 NIXL 已经提供了这个能力**：
`sendLocalPartialMD` / `getLocalPartialMD`（`nixl.h:507/569`，
Python 侧 `send_partial_agent_metadata` / `get_partial_agent_metadata`）
允许只发布 descriptor 的一个子集，`descs` 为空时退化为「仅连接信息」。

实测（H20 + NIXL 1.5.0）：全量 MD 约 **406 B/region** 线性增长，
4096 区时 1.6MB；只发布其中 1 个区是 **883 B**，仅连接信息是 **401 B**，
**压缩 1884 倍**。详见 §3.3.1d。

所以这一项的正确形态是「按 plan 的对端集合分发 partial MD」而非自研指针表——
**前者是上游维护的能力，后者是我们要自己维护的一层**。
自研指针表只在实测 partial MD 仍不够时作为备选。

**另一处零风险改动**：`make_xfer_tag`（`client.py:42`）用 `pickle.dumps`
为每个传输生成唯一标识。**已落地**：改为 UTF-8 NUL 分隔编码 + 单 shard 的 f-string 快路径。
**实测 140k descriptor 下 55ms → 20ms**（不是初稿暗示的「秒级」——
pickle 单次约 0.4μs 而非 15μs，初稿把量级说大了约 20 倍）。
收益真实但有限，**这个规模下的主导开销是 NIXL 自己的 descriptor 处理，不是 tag 构造**。
回归保护见 `tests/nixl/test_xfer_tag.py`（含 `(1,2)` vs `(12,)` 的单射性测试）。

### 3.5.6 轮数与 chunk 大小：去冗余之后，规则是反的

**本节修正一处算错的规则。** 初稿写「chunk 大小应随模型规模上调，
1TB 用 4GB chunk 得到 250 轮」——**那是按全量 `M` 算的，
而去冗余之后每个节点只收 `M_v = M / num_nodes`**。重算：

```
1TB MoE，fp8 wire，32 节点
  全量 M        = 500GB
  每节点入口 M_v = 500GB / 32 ≈ 15.6GB      ← 这才是轮数的分母
  chunk = 4GB   -> 15.6 / 4 ≈ 3.9 轮         ← 不是 250 轮
  chunk = 512MB -> 15.6 / 0.5 ≈ 31 轮
```

**3.9 轮会让流水失效**，这与「chunk 调大」的本意完全相反。

#### 决定 chunk 大小的是流水级数，不是模型规模

真正的约束来自 §3.6 的三类重叠。设 `T_rdma` 为单节点 RDMA 总时长、
`T_fmt` 为 rollout 侧落地格式化总时长，流水深度 `D`（本设计 `D=2`）：

```
轮数 N = M_v / chunk
深度 D 的流水能隐藏的格式化量 ≈ T_fmt × (N-1)/N
所以要让格式化被充分隐藏，需要 N 足够大
```

**1TB 场景的实测量级**（miles 的 Kimi-K2 数据）：

| 项 | 时长 |
|---|---|
| 每节点 RDMA（15.6GB / 50GB·s⁻¹） | **312ms** |
| 落地格式化 `post_load_weights` | **884ms** |

**`T_fmt / T_rdma ≈ 2.8`——格式化比传输还长**。这导出两条结论：

1. **1TB 下的瓶颈是 rollout 侧的 GPU 计算，不是网络。**
   即使 NIC 打满、重叠完美，下界也是 `max(312, 884) = 884ms` 而非 312ms。
   **打满带宽解决不了这个问题。**
2. **chunk 应当调小而非调大**：512MB 给出 31 轮，
   让格式化有足够级数被 RDMA 掩盖；4GB 只有 3.9 轮，
   最多隐藏 1 轮（约 226ms），剩下 658ms 完全暴露。

#### 修正后的 chunk 选取规则

> **chunk 的上界由「轮数必须足够支撑流水」决定，下界由「轮级控制面开销可忽略」决定。
> 模型规模只通过 `M_v` 间接影响它，且方向与初稿相反。**

```
N_min = ceil(max(T_fmt / T_rdma, 1) × D)     # 流水需要的最少轮数
chunk_max = M_v / N_min
chunk_min = 由完成检测粒度定，不是由控制面开销定（§3.6.4）：
            轮级决策已本地化（见下），单轮控制面开销 ~O(10μs) 可忽略；
            真正的下界是 check_xfer_state 的 1ms 轮询退避，
            它让每轮多付 ~0.5ms，256MB chunk（~5ms）下已占 10%
```

1TB 场景代入：`N_min = ceil(2.8 × 2) = 6`，`chunk_max = 15.6GB / 6 ≈ 2.6GB`。
**取 512MB（31 轮）留足余量**，而不是 4GB。

**下界比初稿以为的高一个量级。** 初稿写「控制面开销 ~O(10μs)，所以 chunk 可以很小」，
但那只算了控制面，没算**完成检测**——后者是 1ms 轮询（§3.6.4），
把实际下界抬到约 256MB。`chunk_max < chunk_min` 时调 chunk 已无解，
必须转向 §3.6.0 推论 3 的「削减格式化本身」或先做落地项 5b 的 notif 改造。

**注意 `T_fmt / T_rdma` 是模型相关的**：dense 模型的格式化远轻于 MoE
（MoE 的 `post_load_weights` 要重排 expert 权重），
所以这个比值由 **cost model 预先给出，不需要首次同步时实测**：

```
T_fmt(N_keys, M_v)  =  N_keys × 12.5μs  +  2·M_v / 4800 GB/s     [ms]
T_rdma(M_v, chunk)  =  M_v / BW_NIC_node  +  ⌈M_v/chunk⌉ × 2μs   [ms]
  BW_NIC_node = NIC_count × NIC_line_Gbps × 0.85 / 8              [GB/s]
  (H20 8×200GbE 节点: 170 GB/s)

T_fmt / T_rdma 的决定因素：
  - 未做 MoE 分组时(K2 70k keys): 885ms / 92ms ≈ 9.6  → fmt 主导
  - 完成 MoE 分组后(K2 183 keys):  8.8ms / 92ms ≈ 0.1 → rdma 主导
```

**结论：完成 §3.5.5（MoE 分组）之后，T_fmt/T_rdma < 1，fmt 不再是主导项。**
自适应 chunk 的规则不变（§3.5.6），但**先决条件是 §3.5.5 已完成**。
Dense 模型（~100 keys）中 T_fmt 约 12ms，rdma 仍主导，不需要 MoE 特殊处理。

#### 轮级决策必须本地化（这一条不变）

无论轮数是 31 还是 2000，每轮都不能有跨进程往返：

- **源选择在拉取开始时一次性解析完整轮次**，而非每轮查一次。
- plan 本地缓存（§3.2a）、完成通知批量上报（§3.2b），二者已规划。

**初稿担心的「2000 轮 × RTT」在去冗余之后不再是主要矛盾**
（真实轮数是数十量级），但本地化依然要做——
它让 chunk 可以自由调小而不必担心控制面开销，
**这恰恰是上面「chunk 调小」结论能成立的前提**。

### 3.5.7 trainer 单 rank 故障：DP 副本是天然的降级目标

万卡下 trainer rank 数达数千，单 rank 故障概率不可忽略，而直供去掉了 CPU 副本，
看似没有降级目标。**但实际上有，而且是免费的。**

**关键事实：train 侧的 sharding 声明不含 DP 维度。**
`megatron_converter.py::get_sharding_for_param` 只按 TP / ETP 维度声明分片
（`shard_indices = [(self.mpu.tp_rank,)]` 或 etp_rank），
**DP 维度完全不出现在 `NIXLSharding` 里**。这意味着同一个 `(key, shard)`
在 DP 组内有 `dp_size` 个**内容完全相同**的副本，分布在不同 rank 上。

这正是 §3.5 的三阶段规划第一步「source 侧多副本各领一份不相交切片」所利用的东西：
平时它让 `dp_size` 个 rank 分摊出口（去冗余）。**故障时它就是冗余来源。**

**但这个事实是后端相关的，不是普遍成立的。** 纯 FSDP2（`fsdp_size >= world_size`）
下 mesh 是一维的，副本数为 1，下面的 ① 不存在（§3.9.1 的对比表）。
所以**副本数必须是 index 从 `NIXLSharding` 推导出的运行时量**，
不能是「DP 副本一定存在」这个假设。`replica_count == 1` 时 ① 直接跳到 ②。

#### 三层响应，按故障范围递进

```
① 单 rank 故障（Megatron / HSDP 下最常见；纯 FSDP2 下不可用，见上）
   index 把该 rank 从 holder 表摘除
   -> 它负责的 chunk 由同 DP 组的其他 rank 接管（内容相同，无需重算）
   -> 受影响的 pull 重新解析源，继续读
   对 rollout 不可见，对训练无影响（Megatron 自身的容错另论）

② 整个 DP 组故障（同一 chunk 的所有副本都没了）
   该 chunk 无合格 holder
   -> index 返回 CHUNK_UNAVAILABLE（明确错误，不挂起）
   -> rollout 放弃本轮，保持当前版本继续生成，等下一个版本
   这是「等下一个版本」而非「卡死」，也不是「读旧版本」:
   它用的是自己已有的旧权重，staleness 对算法可见，与计划外的混合策略不同

③ trainer 全体重启
   index 的 holder 表整体清空，版本号保留（§3.2 的版本连续性要求）
   -> 所有 pull 立即返回 CHUNK_UNAVAILABLE
   -> rollout 维持现状生成，由 coordinator 决定是否降级或暂停
   -> trainer 恢复后重新注册 holder 并发布新版本，链路自然恢复
```

#### 为什么 peer 不是主要降级手段

§3.5 的 peer 机制看似也提供冗余，但它**只覆盖「已经被读到的那个版本」**，
且 §3.5.2 的版本相等性要求在新版本发布时整体清空 holder 位图。
所以 peer 对「新版本的第一波」完全无帮助，而故障恰好可能发生在那时。
**DP 副本才是结构性的冗余，peer 只是带宽优化。** 这两者不要混为一谈。

#### 必须做对的三件事

1. **holder 摘除要能被动触发，不能只靠主动上报。** 故障 rank 不会自己说「我挂了」。
   依据是 NIXL 传输失败：rollout 的 `client_read` 报错时，**连同失败的 holder 身份
   一起上报 index**，index 据此摘除并让该 pull 重新解析。
   这把故障检测搭在已有的错误路径上，不需要心跳。
2. **`CHUNK_UNAVAILABLE` 必须是显式返回值，不能是超时。**
   初稿只说「index 需返回明确错误而非挂起」，这里落实为一个具体状态码，
   且 rollout 侧要有对应处理：放弃本轮、保持版本、不重试到死。
3. **不允许「部分 chunk 成功」就标记版本可见。**
   这是 §3.4.6 的 commit group 完整性断言的直接应用：
   少了任何一个 chunk，该 group 就不可见，整个版本对该 instance 不可见。
   否则会得到一个半更新的模型，比旧版本更糟。

#### 与现有机制的衔接

PSRL 已有 `_quarantine_failed_replicas`（`sync_and_migrate_mixin.py:74`）
处理 rollout 侧同步失败：`fail_sync` + 网关暂停 + 恢复路由循环。
本节的 ② ③ 应当复用这条路径，而不是新建一套，区别只在于触发源从
「rollout 自己失败」变成「index 报告源不可用」。

##### 但「保持当前版本继续生成」与现有的两处代码直接冲突，必须一并改

**这是第六轮核实衔接点时发现的**：②③ 说「rollout 放弃本轮、保持当前版本继续生成」，
而现有链路做不到这件事——它只会让该 replica 变成不可用。两处：

**(a) `pull_model_for_sync` 对版本不达标是 `assert`，不是返回值。**

```python
# vllm_async_server.py:1036
assert curr_model_version >= ps_version, (
    f"Pulled model version is stale: current={curr_model_version}, required={ps_version}.")
```

`CHUNK_UNAVAILABLE` 下 `curr_model_version` 必然小于 `ps_version`,
**这个 assert 会抛出**,经 `_finish_gateway_sync` 的 `except` 走到
`fail_sync` + `/workers/pause`（`:66-72`）——**replica 被摘掉，而不是「继续生成」**。
所以 ② 要成立，这个 assert 必须改成「返回实际版本并标记本轮未达标」。

**(b) 版本号在 pull 之前就已经发布出去了。**

```python
pull_futures = [...pull_model_for_sync.remote(self.ps_model_version)...]
updates = build_weight_version_updates(instance_ids, self.ps_model_version)
await self._publish_weight_version_updates(updates)      # ← 先声明成功
...
_finish_gateway_sync: actual_versions = await gather(*pull_futures)
                      if actual_version != advertised_version: 再发一次更正
```

**这是一个乐观发布 + 事后更正的结构。** 正常路径下没问题（SMG 很快收到更正），
但 `CHUNK_UNAVAILABLE` 下存在一个窗口：
**SMG 认为该 instance 已是 `v+1`，并按 `v+1` 路由请求，而它实际还是 `v`。**
窗口长度等于一整轮 pull 的时长——在 1TB 场景下是秒级。

这不是本设计引入的，但**直供把它放大了**:今天 pull 失败是小概率的 PS 故障,
直供下 `CHUNK_UNAVAILABLE` 是设计内的正常降级路径（②③），会被常态触发。

**所以 ②③ 的落地必须包含一条：版本发布改为 pull 成功后发布，不再乐观预发。**
代价是 SMG 感知新版本晚一个 pull 时长；收益是**消除「声明版本 ≠ 实际版本」这个窗口**,
而该窗口下的请求会被按错误的策略版本记账，**污染 staleness 账本与 importance sampling**。

**与 §3.4.4 的「免费 on-policy 收益」不冲突**:那一条说的是
「trainer 登记后 rollout 立刻可读」（发布点提前），
这里说的是「rollout 自己的版本号在它真正拿到权重后才对外可见」（声明点推后）。
两者方向相反但作用于不同主体,叠加之后仍是净收益。

**新增落地项 8d**,与 8b（DP 副本接管）同批——两者都是 `CHUNK_UNAVAILABLE`
这条降级路径的组成部分,分开做会让 8b 落地后立刻撞上 (a) 的 assert。

### 3.6 支柱五：执行层，plan once post many

三处改动，彼此独立可分别落地：

- **预备描述符**：握手期对每个 client 调一次 `prep_xfer_dlist`（本地 + 远端两侧），
  之后每轮用 `make_prepped_xfer(local_indices, remote_indices)` 按下标组合，
  替换每 (key, shard) 一次 `initialize_xfer`。这与 §3.5 的「plan 握手期编译」是同一件事的两面：
  plan 里的下标就是 prepped dlist 的下标。
- **strided 描述符**（**需 NIXL ≥ 1.5.0**，§3.6.3）：用 Nx5
  `(addr, len, dev_id, stride, count)` 原生表达非连续 shard，
  删掉 `max_pinned_temp_memory_slots` 整套中转 + 槽位等待逻辑
  （`client_read` 里 `_pinned_slot_running_read_xfer` 的串行等待是真实的流水阻塞点）。

  **一处事实更正**：初稿称这些槽位是「额外的 CPU 权重存储」，
  读代码后发现**不是**——`client.py:541` 分配在 `device=self.device`，
  即 GPU 上（变量名 `pinned_memory` 是误导）。所以它服务的是约束 (b) 的
  **GPU** 那一半，不是 CPU 那一半。
  **收益因此要重新表述**：省下的是 `slots × 最大非连续 shard` 的 GPU 显存
  与槽位串行等待，而不是 CPU 内存。槽位等待才是主要收益。
- **round-based 双 buffer**：chunk 默认 512MB，深度 2 流水。需要被重叠的是**三类**工作，
  不只是量化：
  - trainer 侧的 cast / scale 转码（cast 路径）
  - RDMA 本身
  - **rollout 侧的落地格式化**（`process_weights_after_loading` 系）

  按 atomic group 流水，让第 i 组的格式化与第 i+1 组的 RDMA 重叠。
  WeightBridge 实测 pack/assemble kernel 占端到端不足 3% 且被完全隐藏，前提正是这种流水。

#### 3.6.0 大 MoE 下的瓶颈随 §3.5.5 完成后发生转移

**这一条决定了 §3.6 的优先级顺序，两阶段结论不同：**

三类工作量级，用 Roofline + LogGP 模型推导（以 H20 8×200GbE 节点为例，数据可参数化）：

| 项 | 1TB MoE / 32 节点（**未做 MoE 分组**） | 1TB MoE / 32 节点（**已做 MoE 分组**，§3.5.5） |
|---|---|---|
| **每节点 RDMA**（15.6GB / NIC_BW） | **92ms**（8×200GbE×85%=170GB/s）| **92ms**（不变） |
| **trainer 侧 cast**（BF16→FP8，Roofline） | **9.8ms**（HBM 4800GB/s，内存带宽限制） | **9.8ms**（不变） |
| **rollout 侧落地格式化**（`post_load_weights`） | **885ms**（K2 70,272 key × 12.5μs + 6.5ms BW） | **8.8ms**（183 key × 12.5μs + 6.5ms BW） |

**T_fmt 模型**（基于 kernel launch 分析，miles Kimi-K2 实测值 884ms 拟合验证）：

```
T_fmt(N_keys, M_v)  =  N_keys × T_py_overhead  +  2·M_v / BW_HBM
  T_py_overhead ≈ 12.5μs/key（Python dispatch + dtype check + reshape）
  BW term: 2 × 15.6GB / 4800GB/s ≈ 6.5ms（可忽略）

拟合依据（注意 N_keys 取 §3.5.5 的架构推导值，不是估计值）:
  Kimi-K2 HF 展开 = 384 experts × 61 层 × 3 投影 = 70,272 key（§3.5.5 的表）
  70272 × 12.5μs + 6.5ms ≈ 885ms  ≈ miles 实测 884ms  ✓

**一处已修正的拟合错误**：本节初稿按 `14000 key × 60μs` 拟合，
而 §3.5.5 自己算出的是 **70,272**（fp8 配 scale 则 140,544）。
14000 这个数没有推导来源，是 140k 的十分之一——转写时掉了一个数量级。
按正确 key 数重新拟合得 **12.5μs/key**，这个值对「Python 调度 + dtype 检查 + reshape」
也更合理（60μs/key 偏高一个数量级）。
**结论方向不变**（见下），但公式的绝对预测值此前偏大约 5 倍。
```

BW 项（6.5ms）相对 launch 项可忽略**仅在 key 数较大时成立**：
200 key × 12.5μs = 2.5ms 已小于 BW 项的 6.5ms，
所以**分组后 T_fmt 由带宽项主导，分组前由 launch 项主导**——
这是两个不同的 regime，不能用一句「launch-dominated」概括。
format 优化（MoE 分组、减少 key 数）有效的根本原因是分组前处在 launch regime。

**实测校准（本机 `/jizhicfs/johnnyslin/models`）**：

| 模型 | HF key 数 | expert key 占比 | 分组后 | 压缩比 |
|---|---|---|---|---|
| Qwen2.5-0.5B（dense） | 290 | 0% | 290 | 1× |
| Moonlight-16B-A3B | 5,344 | 94.9% | — | — |
| **Qwen3-30B-A3B** | **18,867** | **97.7%** | **579** | **32.6×** |

**所以「14 万 → 数百」的说法要收紧为「压缩比 30~400×，随 expert 数而变」**：
Qwen3-30B-A3B（128 experts）实测 32.6×，Kimi-K2（384 experts）理论 70272/183 ≈ 384×。
回归测试见 `tests/nixl/test_cost_model.py`。

**两阶段结论**（按修正后的 `T_py=12.5μs` 与实测 key 数重算）：

| 阶段 | 主导项 | 正确目标 |
|---|---|---|
| **§3.5.5 完成前** | K2 70k key：fmt **885ms** >> rdma 92ms（9.6×）；Qwen3-30B 18.9k key：fmt **242ms** >> rdma 92ms（2.6×） | **优先做 §3.5.5（MoE 分组），这才是瓶颈** |
| **§3.5.5 完成后** | K2 183 key：rdma 92ms >> fmt **8.8ms**（10×）；Qwen3-30B 579 key：rdma 92ms >> fmt **13.7ms**（6.7×） | **优先做 §3.6（overlap）打满 NIC 带宽** |

> **「格式化是主导项」只在 §3.5.5 完成之前成立。完成之后，RDMA 是主导项，
> 「打满带宽、充分 overlap」才是正确目标。**
>
> 两者必须按序：先做 §3.5.5，再做 §3.6。
> 先做 §3.6 而没有 §3.5.5，overlap 做完端到端仍被格式化主导，浪费实现工作量。

**一处边界条件**：节点数极多时这个结论会翻转回来。
`T_rdma ∝ M_v` 随节点数线性下降，而分组后的 `T_fmt` 只有 BW 项随 `M_v` 下降、
launch 项（`N_keys × 12.5μs`）是常数。128 节点 + K2 分组后：
`T_rdma = 23ms` vs `T_fmt = 8.9ms`，仍是 rdma 主导；
但若 key 数更多或节点数再翻倍，交叉点会出现。
**判据写成「比较两者」而不是「假定 rdma 主导」**，验证项 2 已据此表述。

**cast 资源分析：**
BF16→FP8 Roofline 为 1596 GB/s（HBM 4800GB/s / 3.008 字节/元素），
15.6GB 只需 9.8ms，RDMA 用 92ms——**cast 始终不是瓶颈**。
cast 期间 NIC 在做 RDMA，两者天然并行，不需要额外调度。

**验证项 2 的判据据此修正**：不能只测「trainer 出口 == M」，
必须**分别计量 RDMA 时长与格式化时长**，并报告两者的重叠率。
若 §3.5.5 已完成，判据是「端到端时间 ≈ T_rdma + T_fmt_per_chunk」（最后一轮无重叠），
验证 overlap 是否真正生效。

#### 3.6.3 实测：prepped descriptor 的收益成立，但有两个前提

**落地项 0 的 micro-benchmark 已完成**
（`tests/nixl/bench_nixl_prepped_xfer.py`，H20 + 从 `third_party/nixl` 源码构建的
**NIXL 1.5.0** + UCX 1.20，原始数据存 `bench_nixl_prepped_xfer.result.json`）。

| descriptors | prep (ms) | `make_prepped_xfer` (ms) | `initialize_xfer` (ms) | 加速 |
|---|---|---|---|---|
| 1024 | 0.007 | 0.004 | 0.024 | **5.4×** |
| 4096 | 0.019 | 0.006 | 0.096 | **16.8×** |
| 16384 | 0.085 | 0.014 | 0.424 | **30.1×** |

**§1.2 的判断成立且收益随规模放大**：prepped 路径每次只提交索引，
而 `initialize_xfer` 每次都要重新序列化整个 descriptor list。
但实测暴露了两个前提，不满足就拿不到这个收益。

##### 前提一：索引必须是 int32 ndarray，不能是 Python list

| 索引形式 | 4096 descriptors 下的提交耗时 |
|---|---|
| `list(range(n))` | **0.088 ms** |
| `np.arange(n, dtype=np.int32)` | **0.006 ms** |

**差 15 倍。** `make_prepped_xfer` 的签名是
`local_indices: Union[list[int], np.ndarray]`，两者都接受，
但 pybind11 对 list 是**逐元素转换**——**这恰好把 prepped 路径要消除的
per-descriptor 开销又加了回来**，而且不会报错。

这不是理论推演：**本 benchmark 的早期版本就是传 list，
因而测出「prepped 比 `initialize_xfer` 慢 2~6 倍」的错误结论。**
实现时必须在接口上就固定为 ndarray，不要依赖调用方自觉。

##### 前提二：Nx5 strided 需要 NIXL ≥ 1.5.0

NIXL 在 **v1.5.0** 之前，`prepXferDlist` 只有两个重载（都接受
`nixlXferDList`，而后者是 Nx3-only）。1.5.0 增加了第三个重载
`(agent_name, mem_type, numpy.ndarray, backend)`，走 `to_stride_dlist`
（`nixl_bindings.cpp:148`）。Dockerfile 与 `scripts/install_nixl.sh` 均 pin
v1.5.0（UCX v1.23.x）。实测：

| | prep | submit |
|---|---|---|
| 16384 条 plain descriptor | 0.085 ms | 0.014 ms |
| **1 条 strided descriptor**（等价表达） | **0.002 ms** | **0.001 ms** |

**descriptor 列表压缩 16384:1，prep 开销与 count 无关。**
这对 §3.5.5 的元数据扇出与 §3.9.2.1 的 FSDP2 shard 数放大是直接的解。

##### 一处实测更正：strided 压缩的是 descriptor 列表，不是提交索引

初稿据 benchmark 的 `strided_submit_ms=0.001` 推断「提交开销也与 count 无关」。
**GPU 实测证伪**（`tests/nixl/test_descriptor_correctness_gpu.py`）：
一条 strided 行在 prepped handle 内部**展开为 `count` 个可索引条目**，
提交时必须给 `count` 个索引。只给 1 个索引时：

```
indices=[0]        -> state=DONE, 8 个 slice 里只有 1 个被搬运
indices=[0..7]     -> state=DONE, 8/8 正确
```

**失败形态是静默的**：状态返回 `DONE`，只是大部分字节没动。
benchmark 测到的 0.001ms 是「提交 1 个索引」的耗时，不是「提交等价于 16384 条
descriptor 的传输」的耗时——这两件事被混为一谈了。

**所以 strided 的真实收益是**：元数据体积（descriptor 列表 16384→1）
与 prep 时间（与 count 无关），**不含提交时间**。
这不改变做 A2 的理由（元数据扇出是 §3.5.5 的硬阻塞），但要修正收益预期。

**落地要求**：NIXL 固定为 v1.5.0（Dockerfile 与 install 脚本一致），否则 Nx5
strided 不可用。**注意 `nixlXferDList` 在任何版本下都只接受 Nx3**——
strided 只能经由 `prep_xfer_dlist` 使用，这条不对称容易被误判成「功能缺失」。

##### 一处澄清：WeightBridge 那 590ms 不是提交 API 的开销

16384 个 descriptor 的 `initialize_xfer` 也只要 0.42ms，
相对 §3.6.0 算出的 312ms RDMA 与 884ms 格式化都是噪声。
所以 §1.2 引用的 WeightBridge「590ms 控制面开销」**是 per-tensor 的格式转换**，
不是提交 API——治它的是 §3.5.5 的 MoE 分组（14 万 key → 数百），
而 prepped descriptor 治的是另一件事。

##### prepped handle 可被任意索引模式复用（已验证）

| 索引模式 | 结果 |
|---|---|
| identity / rotated / subset / reversed | **全部通过** |

这支撑 §3.5.2.3 的 chunk 偏移错开与 §3.6.1 的 commit group 子集提交：
**握手期每个 client 准备一次，之后每个 instance 用自己的偏移顺序组合，
不必重建 descriptor**。

#### 3.6.4 完成检测是 1ms 轮询，这给「chunk 调小」划了一条硬下界

**§3.5.6 推导出「chunk 应当调小以提供足够流水级数」，但没有算下界在哪。**
下界来自现有的完成检测实现，而它是轮询而非事件驱动：

```python
# client.py:1572 起（逐 shard 等待）与 :1520 起（merged 等待），两处同构
while True:
    state = self.agent.check_xfer_state(handle)
    if state == "DONE":
        break
    time.sleep(0.001)   # 1ms backoff to avoid CPU starvation at large scale
```

**每一轮传输因此多付 0~1ms 的检测延迟，期望值 0.5ms。** 代入 §3.6.0 的量级：

| chunk | 每轮 RDMA | 轮数（15.6GB/节点） | 轮询开销占比 |
|---|---|---|---|
| 4GB | 80ms | 3.9 | 0.6% |
| 512MB | 10ms | 31 | **5%** |
| 128MB | 2.5ms | 125 | **20%** |
| 32MB | 0.6ms | 500 | **80%，完全不可用** |

**所以 chunk 不能小于约 256MB**，否则轮询开销反超流水收益。
这与 §3.5.6 的「`N_min = ceil(max(T_fmt/T_rdma, 1) × D)`」一起
构成 chunk 的上下界：**下界由轮询粒度定，上界由流水级数定**。
两者若交叉（即 `M_v / N_min < 256MB`），说明这个规模下靠调 chunk
已经压不住格式化，必须转向 §3.6.0 推论 3 的「削减格式化本身」。

**提交与等待在接口层已经分开。** `client_read` / `client_write` 提交一个
`(target, key)` 的合并请求后立即返回 `shards_to_transfer`，调用方按 key `wait`；
handle 在 `wait` 中释放，prepared dlist 可复用并在源重注册时失效。不存在一个
「提交后立刻进入轮询」的全局串行队列，因此 §3.6 的流水重叠可以落在调用方的
wave 结构上（pull engine 的上游分波与 peer 读重叠）。

#### 3.6.5 通用 cost model：将「应实测」替换为公式

本节把分散在 §3.5.2.1、§3.5.6、§3.6.0、§3.6.2(d) 的参数化问题收拢为一个
可解析计算的模型，适用于任意 GPU 节点配置。

**理论基础**：Roofline（Williams SC'08）用于识别内存/计算瓶颈；
LogGP（Alexandrov PPoPP'95）的 `T = L + (n-1)g + NG` 用于 RDMA 时延；
kernel launch 开销分析（Jangda MLSys'22）用于 T_fmt。

```
── 输入：硬件配置 ─────────────────────────────────────────────────────
BW_HBM      : GPU HBM 带宽 [GB/s]  (H20 = 4800, H100 = 3350, A100 80G = 2000)
NIC_count   : 节点 NIC 端口数       (自动探测: ibstat 中 Active 端口数)
NIC_Gbps    : 单端口线速 [Gbps]     (200G RoCE 常见配置)
NIC_eff     : RDMA 效率             (大包 0.85，推荐值来自 Kalia OSDI'16 Fig.5)
BW_NIC      = 实测节点聚合 READ 带宽 [GB/s]
  按端口数推导会错：端口可能 bond 成更少的链路，单流只占一条。
  探测方法：每张 GPU 起一个并发读者，取总字节 / 联合时间窗
  （`tests/nixl/scripts/run_pair_bw.sh`）。同节点 NVLink 带宽用同一脚本测。

── T_fmt 模型（拟合自 miles Kimi-K2 884ms 实测值）─────────────────────
T_fmt(N_keys, M_v)  =  N_keys × 12.5μs  +  2·M_v / BW_HBM
  拟合依据：K2 HF 展开 = 384 experts × 61 层 × 3 投影 = 70,272 key（§3.5.5）
            70272 × 12.5μs + 6.5ms ≈ 885ms ≈ 实测 884ms ✓
  **两个 regime**：key 数大时 launch 项主导；分组后（数百 key）BW 项主导
  MoE 分组后 (§3.5.5)：K2 → 183 key，Qwen3-30B 实测 18867 → 579 key

── T_rdma 模型（LogGP，大块消息 L/g 项可忽略）─────────────────────────
T_rdma(M_v, chunk)  =  M_v / BW_NIC  +  ⌈M_v/chunk⌉ × 2μs
  注：2μs 是 RoCE per-chunk RTT（Kalia 实测，IB 约 0.5μs）

── T_cast 模型（Roofline，内存带宽受限）────────────────────────────────
T_cast(M_v, dtype)  =  M_v × bytes_per_src / BW_HBM
  BF16→FP8: bytes_per_src = 2 + 1 + 1/128 ≈ 3.008B  → BW_eff ≈ BW_HBM/3
  FP32→FP8: bytes_per_src = 4 + 1 + 1/128 ≈ 5.008B  → BW_eff ≈ BW_HBM/5
  H20 BF16→FP8: BW_eff ≈ 1596 GB/s → T_cast(15.6GB) ≈ 9.8ms (远小于 T_rdma)

── W 上限（取以下两者较小）──────────────────────────────────────────────
① W ≤ BW_NIC / (chunk / T_rdma_budget)   (NIC 不超卖)
② W < R                                   (R = rollout instance 数)

初始值: W_init = min(NIC_count, nodes_per_instance)
  物理含义: 每个 rollout 节点派一个代表去读 trainer，节点内其余走 NVLink 扩散
  H20 8×200G 节点、每节点 8 GPU: W_init = 8

── chunk 大小──────────────────────────────────────────────────────────
前提: §3.5.5 完成（MoE 已分组）
  N_min = ceil(max(T_fmt_grouped / T_rdma, 1) × D)  [D=2 pipeline depth]
  chunk_max = M_v / N_min

H20 / 32 节点 / 1T MoE（§3.5.5 完成后）：
  T_fmt = 8.8ms（K2 183 key）, T_rdma = 92ms → ratio = 0.10 < 1 → rdma 主导
  N_min = ceil(max(0.20, 1) × 2) = 2 → chunk_max ≈ 7.8GB
  建议: 512MB（给出 30 轮，充足余量；chunk 下界见 §3.6.4）

── num_threads ─────────────────────────────────────────────────────────
num_threads = min(NIC_count, cpu_cores // 8)
  依据: UCX 每线程独立驱动一个 NIC CQ；超过 NIC 数后竞争 CQ 退化
  H20 8×200G 节点: min(8, 192//8) = min(8, 24) = 8
```

**参数化表**（覆盖常见配置）：

| 节点 | BW_NIC | T_rdma(15.6GB) | T_fmt_grouped | W_init | num_threads |
|---|---|---|---|---|---|
| H20 8×200G RoCE | 170 GB/s | 92ms | 8.8ms | 8 | 8 |
| H100 8×400G IB (NDR) | 340 GB/s | 46ms | 8.8ms | 8 | 8 |
| A100 8×200G IB (HDR) | 170 GB/s | 92ms | 9.3ms | 8 | 8 |
| H800 8×400G IB (NDR) | 340 GB/s | 46ms | 8.8ms | 8 | 8 |

注：T_fmt_grouped ≈ 不变（launch 时间与 GPU 型号弱相关，差异 < 2ms）。

**此模型替代文档中所有「应实测」标注**，除了以下两个量需要首轮真实测量后更新：
1. `T_py_overhead`（12.5μs/key）在不同 vLLM 版本 / Python 版本下可能变化，
   首轮实测后按实际值修正 `T_fmt` 公式；
2. `T_rdma_p99`（W 的反馈调整输入），可用 §3.2b 的完成通知捎带，
   无需额外测量基础设施。

初稿让一个 `bucket` 同时承担六个角色：plan 分桶、提交/等待、版本可见性、放行粒度、
holder 位图的位、wire buffer 复用单位。好处是省事，坏处是**任何一处要调粒度就牵动其余五处**
（§3.5.2.1 「W 过大会让 bucket 周转变慢」就是这个耦合的症状）。

本质上这里有两个独立的概念，应当分开命名：

| 概念 | 服务于 | 边界由什么决定 | 可否切开一个 key |
|---|---|---|---|
| **chunk**（传输单位） | 流水与带宽 | 固定字节数（随模型规模自适应，§3.5.6） | **可以**，大 MoE 的 11GB 单 key 必须切开 |
| **commit group**（可见性单位） | 正确性 | 引擎 loader 的原子要求（atomic group） | **不可以**，这是它的定义 |

两者的关系是 **多对多**：一个 chunk 可以含多个小 commit group，
一个大 commit group（如 MoE 的 fused `w13_weight`）可以跨多个 chunk。
于是：

- **传输、流水、W、wire buffer 复用**全部以 chunk 为单位，可以自由调大调小。
- **版本可见性与运行时格式化**以 commit group 为单位，一个 group 的全部 chunk 到齐才提交。
- **放行（在飞读者计数）以 chunk 为单位**，因为它关心的是「这段内存现在有人读吗」,
  与原子性无关。
- **holder 位图以 chunk 为单位**，因为它回答「这段我有没有」。

这一拆解消掉了初稿里最别扭的一条约束：不再需要「让一个 fused 张量跨多个 bucket
但共享一个 group 标记」这种打补丁的说法，跨 chunk 共享 commit group 本来就是
两个概念分离后的自然形态。

**atomic group 是执行层的硬不变量**（miles `atomic_groups.py` 的教训：引擎 loader 会把多个
HF 权重 cache-and-concat 成一个引擎参数，半份到达是**静默 no-op 或断言失败**）。
group 是四件事的共同单位：plan 分桶、提交与等待、版本可见性、§3.4 的租约粒度。
fp8 的 `weight` 与其 `scale` 天然同组。

#### 3.6.2 底层通信栈的落地要点

本设计不引入 NCCL 通信，数据面是纯 RDMA / NVLink。几条必须落实的细节：

**(a) GPUDirect 可用性必须探测，但只作告警不作选源硬约束。**

**「退化行为」指的是什么**：rollout 直读 trainer 显存时，
若 NIC 与 GPU 在同一 PCIe root complex 下且两侧内存已注册，
RDMA 可以**直接读写 GPU 显存**（GPUDirect RDMA，零 CPU 参与）。
否则驱动会退化为 **host bounce**——先把数据 DMA 到一块 CPU 中转缓冲，
再从那里发出去。**这就是「退化行为」**：同一行 `client_read` 代码，
底层可能走两条带宽差一个量级的路径，而上层完全无感。

它与两件事相关：

| 相关项 | 关系 |
|---|---|
| **硬件拓扑** | NIC 与 GPU 是否挂在同一 PCIe switch 下。跨 NUMA 或经根桥转发时 GPUDirect 可能不可用 |
| **驱动与注册** | 需要 `nvidia-peermem` / `dmabuf` 支持，且内存必须是注册过的 GPU 内存。本设计 colocate 不注册（§3.7.3.0），所以这条只影响 disaggregate |

`network_topology.py:42` 的 `LinkType` 只有
`ETH / IB_PCIE / PCIE / IB / NVLINK / LOCAL` 六档，
**没有区分「IB + GPUDirect」与「IB + host bounce」**。

**处理方式：启动期告警，不做选源硬约束。** 三条理由：

1. GPUDirect 不可用通常是**集群配置问题**（驱动缺失、拓扑不当），
   应当让运维看到并修复，而不是让传输层静默走慢一个量级的路径。
2. `network_topology.py` 的带宽是**硬编码常量**，
   给 host bounce 估一个权重值只是猜测，这与 §3.5.3
   「不做实时带宽探测、不做连续优化」的取向冲突。
3. 若**所有**候选源都不支持 GPUDirect，降级也无源可选——
   此时报警比静默慢跑更有用。

落地：握手期对每个注册区用 `query_memory` 探测一次，
**不可用时打印一条包含 NIC/GPU 拓扑信息的 WARNING**，
并在 §7 验证项 2 的带宽对比里记录这一项，让它在实测中可见。
**不新增 `LinkType` 档位，不改选源逻辑。**

**(b) PCIe 流量归零是未被计入的收益。**
今天 train GPU → CPU PS → rollout GPU 跨 PCIe 两次（D2H + H2D），各 `N×M`。
直供后数据面 PCIe 流量**归零**（GPU 到 GPU，走 NVLink 或 NIC）。
1TB × 32 节点下这是数十 TB 的 PCIe 流量被消除。

注意 optimizer offload 仍走 PCIe（`offload_megatron_copy_params`），
与 NIC 上的权重读取**不争同一资源**，所以 optimizer 期间的读取不受它影响。
但若 NIC 是 PCIe-attached 且与 GPU 共享 switch，两者**会**争 PCIe 带宽,
这取决于硬件拓扑，应作为可配置的保守假设。

**(c) cast kernel 不需要低优先级 stream，这条曾被误列为优化。**
初稿写「colocate 下 cast kernel 应走低优先级 stream，避免与 rollout 的 decode kernel
争显存带宽」。**这个前提不成立**：

- **colocate 下**，transfer 发生在 `pause_generation` 之后（§3.7.2），
  **没有 decode kernel 在跑**，没有争用对象。
- **disaggregate 下**，train 与 rollout 在不同 GPU，**不存在 SM 或显存带宽的共享**。

所以两种模式下它都不产生收益，**已从方案移除**。
唯一可能需要它的是将来「colocate 下 transfer 与 decode 部分重叠」这项优化
（即不完全暂停），那时它是前置条件而非独立优化项。
记在 §5.1 待优化清单里，不在本期做。

**(d) `num_threads` 由 NIC 数量决定，不需要实测。**

UCX 的多线程模式让每条工作线程独立驱动一个 NIC 端口的完成队列（CQ），
从而让多个 RDMA 传输真正并行推进。最优线程数的理论上界是 NIC 端口数；
超过之后每个线程争抢同一个 CQ，反而退化。

```
num_threads_optimal = min(NIC_port_count, cpu_cores // 8)

H20 8×200GbE 节点:
  NIC_port_count = 8,  cpu_cores = 192
  → num_threads = min(8, 24) = 8
  CPU 开销: 8 线程 / 192 核 = 4.2%，可忽略

通用公式:
  NIC 数量 = len([p for p in ibstat.ports if p.state == "Active"])
  可在握手期自动探测，写入 nixl_agent_config.num_threads
```

**把它写成可探测的公式而不是「应实测」**，因为这个量只取决于硬件配置，
在同型号节点上不变，而实测需要在两个 agent 都能连通时才能做（本节点当前状态阻止了它）。

**(e) FP4 的 swizzle 有一次全量 scale 拷贝。**
`swizzle_blockscale` 的 `permute(...).contiguous()` 是真实的重排开销。
scale 体积约为权重的 1/16，绝对量不大，但应从 FP4 的收益账里扣掉。

### 3.7 支柱六：colocate 原地交接，时序互斥消除读写冲突

#### 3.7.1 三个框架的 colocate 模型对比

查 verl / SkyRL / miles 的源码确认了 colocate 的正确模型：

| 框架 | colocate 做法 | 传输缓冲 | 关键点 |
|---|---|---|---|
| **verl** | `sleep(level)` 释放权重显存 → `get_per_tensor_param()` **生成器**逐张量产出（DTensor 走 `.full_tensor()`）→ pack 进 bucket → `update_weights_from_ipc` → `resume(tags=["kv_cache"])` | **512MB 可复用 uint8 bucket**，`reduce_tensor` 导出一次 handle（`bucketed_weight_transfer.py:183`） | 训练完成后才开始写，不存在并发 |
| **SkyRL / vLLM** | `skyrl_ipc` 继承 vLLM 的 `IPCTrainerWeightTransferEngine`。两种模式：packed 用复用 buffer，unpacked 每 param 一个 handle | **1GB 可复用 buffer**（`packed_tensor.py:15`）或 per-param | IPC 句柄传递发生在 rollout 暂停后 |
| **miles** | P2P RDMA 直写。训练阶段结束后才触发 transfer | `FlattenedTensorBucket` 按 dtype 分组打平 | rollout pause → transfer → rollout resume |

**三者的共同点有两条，都被本设计采纳**：

1. **正确性来自时序互斥**，而非内存屏障或 cast 产物。
   trainer 持有控制权期间 rollout 是暂停的，rollout 运行期间 trainer 不写权重。
   **这与精度配置和训练后端都无关**：bf16+bf16、fp32+bf16、fp8+bf16、fp8+fp8
   在 colocate 下全部合法，不需要 cast 产物作为隔离手段。
2. **显存形态是「两份常驻参数 + 一块固定大小的可复用传输缓冲」**，
   且对所有张量无条件拷贝、不做零拷贝优化。见 §3.7.2.1 与 §3.7.2.2 的逐项核实。

#### 3.7.2 PSRL 的 colocate 方案（兼容所有精度组合与两个后端）

PSRL 的 `switch_to_train_mode()` / `switch_to_rollout_mode()`
（`ray_trainer.py:2525-2620`）已经实现了这条互斥，`pause_for_sync` 就在这条路径上。
本设计需要做的是：

```
train phase 结束 -> rollout 暂停（pause_generation 完成）-> 进入 transfer phase：
  1. index 发布新版本，holder = trainer（此刻 rollout 不在读写权重）
  2. 刷新布局转换产物（重新 cat QKV 等，§3.7.2.3）
  3. 逐 bucket 搬运（所有 dtype 组合走同一条路径）：
       源头准备（cast 或 scale 转码，仅 dtype 不同时）
         -> pack 进可复用 IPC buffer（§3.7.2.1）
         -> rollout 侧 rebuild 并写入自己的参数显存
         -> per-bucket barrier（buffer 要复用，必须等对侧读完）
  4. rollout instance 标记为 holder(v)，resume_after_sync
```

**与 disaggregate 的关键差别在第 3 步**：colocate 下**不需要写屏障**，
因为 rollout 已暂停，读写在时间上不可能重叠；
且**所有 dtype 组合走同一条搬运路径**，不像 disaggregate 那样要分
「同精度直读」与「cast 路径」两套（§3.4.2）。
**同一条 dtype 配置在两种部署模式下走不同机制**，这是设计意图而非缺口。

#### 3.7.2.1 colocate 的搬运方式：学三个框架的「一块可复用 IPC buffer」

**CUDA IPC 是跨进程的，不要求同进程。** `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` 的设计目的就是让不同进程映射同一块显存，
PSRL 的 colocate 下 train worker 与 rollout server 是不同的 Ray actor，
正是 IPC 的目标形态。

**但「零拷贝 IPC」不是正确的目标——三个框架都不这么做，理由成立。**
核实三家源码后，它们的 colocate 路径**全部是无条件拷贝**：

| 框架 | 缓冲形态 | 拷贝点 |
|---|---|---|
| **verl** | **512MB 可复用 bucket**（`bucketed_weight_transfer.py:183`）：`torch.empty(bucket_size, dtype=uint8)` + `reduce_tensor` 导出一次 handle，全程复用 | `:156` 的 `buffer[off:off+n].view(dtype).view(shape).copy_(weight)` |
| **vLLM / SkyRL** packed 模式 | **1GB 可复用 buffer**（`packed_tensor.py:15` 的 `DEFAULT_PACKED_BUFFER_SIZE_BYTES`） | `:353` 的 `.contiguous().view(torch.uint8).view(-1)` 再 `copy_` |
| **vLLM / SkyRL** unpacked 模式 | per-param handle，无统一 buffer | `ipc_engine.py:399` 的 `tensor.detach().contiguous()` |
| **miles** | `FlattenedTensorBucket` 按 dtype 分组打平 | flatten 本身就是拷贝 |

vLLM 对这块 buffer 的意图写得最明确（`packed_tensor.py:324`）：

> Allocates a single GPU buffer of `buffer_size_bytes` and registers it for IPC
> once via `reduce_tensor`. Each chunk's packed data is copied into this buffer
> before yielding, so **only one IPC-shared allocation exists for the lifetime
> of the transfer**.

而 unpacked 模式的 docstring 直接承认会拷贝（`ipc_engine.py:380`）：

> non-contiguous inputs allocate fresh storage in `.contiguous()`; the caller
> must keep these alive until the post-send barrier

**为什么无条件拷贝是对的，三条理由**：

1. **布局转换本来就产生新张量**，所以「是否可零拷贝」对大多数 key 无意义。
   verl 的 `get_per_tensor_param` 对 DTensor 调 `.full_tensor()`
   （`transformer_impl.py:1032`），这本身就是 all-gather 加新分配。
2. **逐 key 判定「可否零拷贝」的复杂度不划算**：要同时满足 dtype 相同、
   布局相同、连续性相同、rollout 侧参数地址不变。判错一次就是静默数据损坏。
3. **省下的是同 GPU 内的 D2D 拷贝**，跑在 HBM 带宽（3TB/s 量级），
   1GB 约 0.3ms。**用正确性风险换 0.3ms 不值得。**

**所以 PSRL 的 colocate 方案是**：一块固定大小的可复用 IPC buffer，
所有权重（含布局转换产物）无条件 pack 进去，逐 bucket 送。
**不做 per-key 的 IPC 判定**——初稿曾提出这一条，是过度设计，已移除。

> `nixl_direct` 改变了这个前提，但**没有恢复被否决的那条**。直供下没有 wire buffer，
> 源就是 trainer 的活分配，读者映射后仍然**无条件拷贝**到自己的参数里；逐 key 判定的
> 只是「映射区间是否覆盖本次读」（`locate` 做边界检查、`offset` 做元素对齐），判错
> 只会退回 NIXL，不会静默损坏。省下的是「把字节先打包进中转 buffer」的那次额外拷贝，
> 不是去零拷贝（§3.7.4）。

**一个值得直接采纳的实现细节**：verl 的 bucket 用 `torch.uint8` 作统一类型，
各 dtype 的张量通过 `.view(dtype).view(shape)` 映射进去，
并用 `alignment = weight.element_size()` 做对齐（`:131`）。
**这让一块 buffer 服务所有 dtype，不需要 per-dtype buffer**，
PSRL 的 wire buffer 应当照此实现。

#### 3.7.2.2 colocate 的显存账：与三个框架同量级，且 FP8-native 是唯一能压到单份的路径

**三个框架的 colocate 都是两份常驻参数，没有一个做到单份。**
它们优化的对象不是「两份参数」而是「传输期的第三份」：

| 框架 | 常驻参数 | 传输缓冲 | 合计（`M` 为 bf16 全量） |
|---|---|---|---|
| verl | 2 份 | 512MB 固定 bucket | `~2×M + 0.5GB` |
| vLLM / SkyRL packed | 2 份 | 1GB 固定 buffer | `~2×M + 1GB` |
| vLLM / SkyRL unpacked | 2 份 | per-param，无上界 | `~2×M + O(最大张量)` |
| **本设计** | 2 份 | `2×chunk`（默认 2×512MB） | `~2×M + 1GB` |

**本设计与 verl / vLLM 是同一量级、同一思路**，不存在数量级差距。
`2×chunk` 对应 vLLM 的 packed buffer，深度 2 是为了流水重叠
（vLLM 的 packed 模式是深度 1，靠 `_post_send_sync` 的 per-chunk barrier 串行）。

**两份参数是结构性的，不是实现不好**：train 需要训练精度（fp32/bf16），
rollout 需要推理精度加推理专用布局（fused QKV、swizzle 过的 scale），
两者不可能共享同一块内存。

**唯一的例外是 Megatron TE `fp8_param` + fp8 rollout**：此时两侧 dtype 相同，
trainer 持有的就是 rollout 要的字节，账面降到 `~0.5×M`（§3.7.3 的表）。
**这个组合三个框架都没覆盖**——SkyRL 的 `sharded_rdt` 与 FP8 wire 明确互斥，
verl 与 miles 没有 FP8-native train 路径。
**这是本设计相对三家的唯一显存优势，也是 colocate 的推荐配置。**

另外 **colocate 下 cast 不额外占显存**：cast 产物的落点就是 rollout
已注册的参数显存本身，不需要独立的 wire buffer：

```
disaggregate: trainer 参数 -> [wire buffer 2×chunk] -> RDMA -> rollout 参数
colocate:     trainer 参数 -> pack 进可复用 buffer -> 写入 rollout 参数
```

cast kernel 的 amax 中间量约为 chunk 的 1/16384，可忽略。

#### 3.7.2.3 布局转换产物是第三类字节，§3.4.2 的两条路径分类覆盖不到它

**这是通读 converter 后发现的一处真实缺口。**

§3.4.2 把发布缓冲分成两类：「trainer 参数本身」（同精度直读）与「cast 产物」。
但 converter 里存在**第三类**——为了对齐 vLLM 的 canonical 布局而物化的张量，
它既不是参数本身，也不是 dtype 转换的产物。

**布局对齐实际是三层机制，其中只有一层会搬数据：**

| 层 | 做什么 | 搬数据 |
|---|---|---|
| canonical 命名空间 | 两侧统一到 HF key。vLLM 的 fused `qkv_proj` 被 `slice_qkv_proj` **narrow 成三个视图**（`model_mappings.py:370`） | **否**，`narrow` 是视图 |
| `NIXLSharding` 声明 | Megatron 声明 `{dim: tp_size}`，FSDP2 声明 `{dim: mesh.size()}`，vLLM 声明 `{dim: tp_size}`（`vllm_converter.py:511`） | 否 |
| shard 代数对账 | `server.py:169` 对每个 key 取所有 client 的 **lcm** 作为 finest mesh，各方 `refactor_based_on_finer_shard_mesh` 展开索引 | 否，只改索引 |

**所以 train TP=8 ↔ rollout TP=4 这类并行度不匹配是 shard 代数自动处理的**，
不需要 per-model 代码，这部分设计成立。
`slice_qkv_proj` / `slice_gate_up_proj` / `split_dim0_fused_fsdp_param`
全部基于 `narrow`，产出的是 `make_slice_parameter` 包装的视图。

**但有两处真的物化了内存**（全仓库只有这两处，已逐行核实）：

| 位置 | 操作 | 为什么必须物化 |
|---|---|---|
| `megatron_converter.py:286` `ConcatenatedQKVSync` | `torch.cat((q, k, v), dim=0)` | 走 `ConcatenatedQKVMapping` 的模型要求 HF 形式 Q/K/V 内存连续，而 Megatron 的 interleaved 布局无法用视图表达 |
| `megatron_converter.py:177` `DTypeCastSync` | `tensor.float()` | `A_log` 这类参数架构上必须 fp32 暴露，而训练用 bf16 |

**初稿对这两者的判断是「今天就已物化，本设计不增不减」，这对第一项是错的。**

今天 `ConcatenatedQKVSync` 只需要在 `before_push` 的那一刻内容正确，
所以它可以 lazy——push 前 cat 一次，push 完就不管了。
**直供下 rollout 可能在任意时刻读它，所以它必须常驻正确**，这带来两个后果：

1. **它进入了版本发布的关键路径**：每次 optimizer step 之后、发布 v 之前
   必须重新 cat 一遍，成为发布延迟的一部分。频率没变（仍是每步一次），
   但时机从「push 时」变成「发布点之前」，**而发布点是 §3.4.2 时间轴上的关键位置**。
2. **它的读写冲突需要独立论证**。optimizer 改写 Megatron 的 `linear_qkv` 之后，
   这份 cat 产物就过期了，而 rollout 可能正在读它。它不能套用任何一条现成结论：
   - 不是「cast 产物」——它不随 dtype 转换产生，同精度路径下也存在；
   - 不是「trainer 参数本身」——写屏障保护的是参数内存，而它是另一块。

**正确的处理：把布局转换产物归入发布缓冲，与 cast 产物同等对待。**

```
发布 v 之前（在等待 v-1 读者清零之后、发布之前）：
  1. 刷新全部布局转换产物（重新 cat QKV、重新 cast fp32 副本）
  2. 若走 cast 路径，同时产出 cast 产物
  3. 发布 v
```

这样做之后三类字节的语义统一了：**rollout 读的永远是「发布点那一刻冻结的字节」**，
无论它来自参数本身、cast 产物，还是布局转换产物。
前者靠写屏障保证不被改写，后两者天然与参数内存分离。

**代价要记清**：布局转换产物是**常驻的额外显存**，大小等于受影响参数的总量。
QKV 在典型模型里约占参数的 1/8~1/6，所以走 `ConcatenatedQKVMapping` 的模型
会有约 `0.15×M` 的常驻开销。**这违反约束 (b) 的字面表述**，
但它今天就存在（不是本设计引入的），且是布局正确性的必要成本，
应当在约束 (b) 下显式记为例外，而不是假装它不存在。

**判定哪些模型受影响**：走 `QKVMapping`（`_convert_qkv_parameter`，纯视图）的不受影响，
走 `ConcatenatedQKVMapping`（`_convert_concatenated_qkv_parameter`，cat）的受影响。
这是 per-model 的属性，应当在配置校验期打印出来，让显存预算可见。

**三个框架都不为这类字节做特殊处理，这一点值得对齐。**
它们的 colocate 路径对所有张量无条件 `.contiguous()` + `copy_`
（§3.7.2.1 的三处证据），所以布局转换产物在它们眼里与普通参数无差别——
反正都要拷一次。**PSRL 应当照此简化**：

- **搬运层面不做区分**：布局转换产物与普通参数一样 pack 进可复用 buffer。
  初稿曾提出「为它排除 IPC 零拷贝路径」，但零拷贝路径本身已被移除（§3.7.2.1），
  所以这个例外自动消失。
- **只有「发布点前刷新」这一条是 PSRL 特有的**，因为三个框架都是
  push 模式（trainer 主动推、推完即止），不存在「rollout 在任意时刻拉取」
  这个窗口。**这是 pull 语义的固有代价，不是实现缺陷**，
  也是本设计唯一需要为这类字节额外做的事。

#### 3.7.2.4 两种模式的暂停语义：差别在「谁决定何时暂停」，不在「是否暂停」

一处必须说清的事实，否则会误解写屏障的定位：**今天的 disaggregate 路径也是暂停期间拉取的。**
`sync_and_migrate_mixin.py:155` 在 pull 之前无条件调 `pause_for_sync`：

```python
await asyncio.gather(*[self.server_handles[r].pause_for_sync.remote() for r in replica_ids])
...
pull_futures = [self.server_handles[r].pull_model_for_sync.remote(self.ps_model_version) ...]
```

`pause_for_sync`（`vllm_async_server.py:740`）关 admission gate 并
`pause_generation(clear_cache=False)`。所以两种模式的真实差别是：

| | 谁决定何时暂停 | 暂停是否同步发生 |
|---|---|---|
| **colocate** | trainer 主导，`switch_to_*_mode` 整批切 | 是，全局 barrier |
| **disaggregate** | 每个 instance 由 `_check_should_sync` 自行择时（waiting queue、未消费 reservation、throughput / KV 阈值） | **不保证错开**，见下 |

**但「各 instance 自行择时」不等于「实际一定错开」，这一点必须说准**，
否则会和 §3.5.1 的结论冲突：

- `command_loop.py:134` 对一批 `instance_ids` **循环下发** `sync_with_ps`，
  每个 instance 一个 Ray future，再 `asyncio.gather`。所以**调度粒度是 per-instance 的**。
- **但版本发布是一个同时到达的事件**，第一轮扫描会把「此刻恰好空闲且落后」的
  instance 全部纳入同一批（§3.5.1 已论证）。训练步较长、rollout 已跑完在等权重时，
  这批就是全部 instance。

所以准确的表述是：**错开是常见情形而非保证**。
`_check_should_sync` 的择时让 instance 因负载差异天然错峰，
但设计**不能依赖它**——这正是 §3.5.2 的 W 由 index 强制扩散的理由
（「sync 策略决定谁该更新，index 决定从哪读，后者不能依赖前者的时序巧合」）。

一处与之相关的现有行为需要改（§3.5.2 末段已记）：
`sync_and_migrate_mixin.py:155` 在 pull 之前就对整批 `pause_for_sync`。
若扩散把一批 pull 拉成几轮，**先暂停的 instance 会空等更久**，
所以暂停必须下移到「该 instance 真正开始读」的时刻。
**这条改动的价值正是把「不保证错开」变成「传输层强制错开」。**

**所以写屏障的定位是「把 pause 窗口收窄到不存在」的余地，不是「让 rollout 边读边服务」。**
同精度路径下有了写屏障，`pause_for_sync` 才**可能**从「覆盖整个 transfer」
降级为「只覆盖 commit group 的切换瞬间」。这是一项优化而非正确性需求——
正确性在两种模式下今天都由暂停保证。
实现顺序上应当**先保留暂停、后收窄**：先让直供在现有暂停语义下正确，
再用写屏障逐步缩短窗口，并用 §7 验证项 4 的 p50/p99 衡量收益是否值得。

#### 3.7.3 colocate 切换：train 权重必须 offload 到 CPU，让位给 KV cache

**本小节修正初稿的一处错误取舍。** 初稿的目标是「让权重根本不动」——
保住 NIXL 注册、避免 dereg/rereg。**这个目标本身诱人，但代价算错了。**

**三个框架的一致做法是：transfer 完成后立刻把 train 权重 offload 到 CPU，
然后才给 rollout 分配 KV cache。** verl 的 `update_weights`
（`engine_workers.py:811-826`）把顺序和理由都写明了：

```python
# 3. offload model to cpu
if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
get_torch_device().synchronize()
get_torch_device().empty_cache()

# 4. resume kv_cache
if self.config.rollout.free_cache_engine:
    # ... Wait for their offload and device cleanup before a leader wakes the
    # rollout workers; local device synchronization alone cannot prevent OOM on peers.
    await asyncio.to_thread(torch.distributed.barrier)
    await self.rollout.resume(tags=["kv_cache"])
```

**offload 在 KV cache 分配之前，中间还有一道 barrier，注释直接说是为了防 OOM**——
所以「腾显存给 KV cache」不是副作用，是刻意设计的顺序。另外两条佐证：

- verl 的 example 里 `param_offload=True` 出现 **96 次**、`False` 36 次。
  72B colocate 配置（`run_qwen2_5_72b_fsdp.sh`）actor 的
  `param_offload` 与 `optimizer_offload` **全开**。
- SkyRL 更彻底：`worker_dispatch.py:161` 的 `_ensure_on_gpu` 实现了完整互斥——
  唤醒一个模型前**先 offload 同 colocation group 内其他所有模型**，
  且 `_gpu_state` 分别追踪 `model_on_gpu` 与 `optimizer_on_gpu`。

**算一笔账就知道初稿错在哪**（72B，8×H100 80GB，fp8 rollout）：

| | 初稿（权重不动） | 三家做法（offload） |
|---|---|---|
| train bf16 权重 | **18GB/卡 常驻** | 0（在 CPU） |
| rollout fp8 权重 | 9GB/卡 | 9GB/卡 |
| **剩给 KV cache** | ~50GB | **~68GB** |

**KV cache 少 26%，直接等比例压低 rollout 并发**，而换来的只是
18GB/卡 的 CPU 内存（CPU 侧通常 1~2TB）。这个交换明显亏。
更根本的是：**初稿用「NIXL 注册不失效」这个实现便利换掉了 rollout 吞吐**，
而 §3.4.3 自己立的原则是「代价的上限由算法/业务目标决定，不由 infra 方便决定」。

##### 与 pull 语义的冲突，以及它的解

三家没有这个顾虑是因为它们是 **push 模式**——trainer 推完就不管，之后 offload 无所谓。
本设计是 pull 模式，看似会让注册的指针悬空（§3.9.3c）。
**但核实三家后发现这个担心本身是多余的：colocate 下压根不应该做 NIXL 注册。**

##### colocate 不注册 RDMA，所以「指针悬空」不存在

**miles 把它写成第一优先级分支**（`protocol.py:73-82`）：

```python
def get_weight_transfer_protocol(args):
    mode = getattr(args, "update_weight_transfer_mode", "broadcast")
    ...
    if args.colocate:                      # ← 在所有 mode 判断之前短路
        from ...protocols.cuda_ipc import UpdateWeightFromTensor
        return UpdateWeightFromTensor(args)
    if mode in ("broadcast", "broadcast_packed"): ...
    if mode == "p2p": ...                  # ← 只有非 colocate 才可能走到
```

**`cuda_ipc.py` 全文没有任何 `register` / `rdma` / `transfer_engine`。**

**三家的注册条件归纳**（逐文件核实）：

| 场景 | 注册什么 | 位置与大小 | 频率 | 出处 |
|---|---|---|---|---|
| **colocate** | **不注册** | — | — | miles `protocol.py:79`；verl 走 `mode="naive"`（`engine_workers.py:759`），不建 NIXL agent |
| disaggregate，CPU 中转 | **一份完整的 CPU 模型副本**，per-tensor 注册 | **CPU**，`M`（rollout dtype 的全量） | 一次，进程生命周期内不注销 | miles `_create_cpu_replica`（`p2p.py:205`）+ `register_cpu_memory`（`p2p_transfer_utils.py:189`） |
| disaggregate，GPU 直传 | **固定大小 bucket**（非参数） | **GPU**（`device` 默认 `cuda`），`2×bucket_size`，**默认 2×2048MB = 4GB** | 一次，`finalize()` 时注销（`:356`） | verl `nixl_checkpoint_engine.py:274`，`rollout.py:135` 的 `update_weights_bucket_megabytes=2048` |
| **本设计** | **固定大小 wire buffer**（非参数） | **GPU**，`2×chunk`，**默认 2×512MB = 1GB** | 一次 | §3.3 |

**miles 的 CPU buffer 不是传输缓冲，是整个模型**，这一点值得说清：
`_create_cpu_replica` 用 sglang 的 `get_model(..., DeviceConfig(device="cpu"))`
建一个完整 CPU 模型（`load_format="dummy"`，只建结构不读盘），
然后 `param.data = param.data.pin_memory()` 把**全部参数** pin 住并逐个注册。
只有 `first_engine_rank` 真正分配，其余 engine rank 复用同一份（`p2p.py:252`），
所以每个 trainer 进程 1 份而非 N 份。这正是 §2 调研表里
「**CPU replica 违反约束 (b)**」的确切出处。

verl 的 docstring 把 GPU 侧开销写得很明确（`nixl_checkpoint_engine.py:246`）：

> bucket_size ... Note that we use two buffer to send and recv weights at same time,
> **so the device memory overhead is 2 * bucket_size**.

**三家的中转形态是三种取舍**：

| 框架 | 中转介质 | 量级 | 代价与收益 |
|---|---|---|---|
| miles p2p | CPU 全量副本 | **`M`** | 内存大，但**地址永久稳定**、注册一次、**完全不受 GPU offload 影响** |
| verl NIXL | GPU 固定 bucket | **4GB** | 内存小，每轮 pack/unpack |
| 本设计 | GPU 固定 chunk | **1GB** | 同 verl，chunk 更小、流水更细（§3.6 深度 2） |

##### miles 的 CPU 副本是「中转站」而非「快照」，且训练侧不注册

三条容易误解的事实，逐一核实：

**(a) CPU 副本每步都被覆写**。`send_bucket`（`p2p.py:107`）每轮调
`target.model_replica.load_weights(ready_hf_tensors)` 写入新权重。
它存在的理由是两个，都不是「保存一份」：
**复用 sglang 的 `load_weights` 做 HF → 引擎 fused 格式转换**（QKV 合并、MoE fused），
以及**提供一块地址永不变的注册内存**。

**(b) 训练侧的 GPU 权重完全不注册**。注册只发生在两端：

| 端 | 注册什么 | 出处 |
|---|---|---|
| 训练侧 | **CPU pinned 模型副本** | `register_cpu_memory`（`p2p_transfer_utils.py:189`） |
| rollout 侧 | **GPU 参数本身** | sglang 提供，`query_remote_weight_infos`（`:213`）查到地址 |

**所以 trainer 可以随意 offload / resize / 改并行度，都不影响 RDMA**——
这正是 §3.7.3.0a 那个地址正确性问题在 miles 下不存在的原因。

**(c) 传输是 CPU → 远端 GPU 的单边批量写**：

```
训练 GPU 权重 → [all-gather + HF 转换，在 GPU 上]
             → load_weights() 写入 CPU pinned 副本        （D2H，走 PCIe）
             → batch_transfer_sync_write(session, source_ptrs, target_ptrs, lens)
                                                          （RDMA 单边写，CPU → 远端 GPU）
```

`_do_p2p_write_one_session`（`p2p.py:311`）**批量提交指针数组**而非逐张量，
并用线程池并发（`P2PTransferManager`，默认 4 workers，
注释原文「RDMA ops won't be affected by the python GIL」）。
最后一个 engine rank 的写是 fire-and-forget，其余要 `f.result()` 等——
因为共享 buffer 会被下一轮覆写。

##### miles vs verl 的效率对比：取决于瓶颈在哪

| 维度 | miles | verl |
|---|---|---|
| 数据路径 | GPU → **CPU** → 远端 GPU | GPU → **GPU bucket** → 远端 GPU |
| **PCIe 流量** | **`M` 的 D2H**（必经） | **0** |
| 跳数 | 2 | 1 |
| pack 开销 | `load_weights()`，per-tensor 格式转换 + 拷贝 | `copy_` 进 uint8 buffer，纯内存拷贝 |
| **unpack 开销** | rollout 侧收到**终态格式**，近零 | rollout 侧要切分 + `load_weights` |
| GPU 显存 | **0** | 4GB |
| CPU 内存 | **`M`** | 0 |
| 注册次数 | `O(#tensors)`，一次性 | **2**，一次性 |

**纯传输延迟 verl 赢**：miles 多付一次 `M / B_pcie` 的 D2H，
72B fp8 下约 72GB / 63GB·s⁻¹ ≈ **1.1s**，且在关键路径上
（要先写完 CPU 才能发起 RDMA）。

**但两点缩小差距**：① miles 的 D2H 与 RDMA 逐 bucket 流水，
理想情况下只剩首个 bucket 的延迟暴露；
② **pack/unpack 的总账 miles 更优**——verl 把 `load_weights` 的成本留给 rollout 侧，
而 miles 在 CPU 上提前做完。miles 自己实测 Kimi-K2 的 `post_load_weights`
要 **884ms**，这笔钱 verl 要在 rollout 侧付。

**结论是条件性的**：

| 瓶颈 | 更优 |
|---|---|
| trainer 出口带宽（大 fan-out） | 相同，都是 `M` 出口 |
| **rollout 侧落地（大 MoE）** | **miles**，格式转换挪到了 CPU |
| **GPU 显存紧张** | **miles**，零 GPU 占用 |
| CPU 内存紧张 / 追求最低延迟 | **verl** |

##### miles 的实测数据：小模型净亏，大 MoE 才赢

miles 文档（`docs/advanced/p2p-weight-transfer.md`）有完整 benchmark
（H100-80GB，1GB bucket，steady-state 步 3–12 平均，对照组是 NCCL broadcast）：

| 模型 | 规模 | NCCL | **miles RDMA** | Delta |
|---|---|---|---|---|
| GLM-4.7-9B-Flash | 30B(3B) | 2,508 ms | 4,229 ms | **+68.6%** |
| GLM-5_4layer | 4 层 | 732 ms | 1,261 ms | **+72.2%** |
| Qwen3-30B-A3B | 30B(3B) | 2,670 ms | 2,160 ms | −19.1% |
| GLM-4.5-Air | 106B(12B) | 5,001 ms | 2,637 ms | −47.3% |
| Qwen3-235B-A22B | 235B(22B) | 10,754 ms | 3,162 ms | −70.6% |
| GLM-5 | 744B(40B) | 58,302 ms | 8,480 ms | −85.5% |
| **Kimi-K2** | **1T(64B)** | **53,279 ms** | **7,227 ms** | **−86.4%** |

**前两行是关键：小模型上 miles 比 NCCL broadcast 还慢 68~72%。**
原因正是那次 D2H——小模型的传输本身只要 2.5s，多付一次 PCIe 往返就是净亏。
**只有传输时间远超 D2H 时，去冗余的收益才压过中转成本。交叉点在 30B 左右。**

注意 miles 除 CPU replica 外**还有一个 512MB 的 bucket**
（`--update-weight-buffer-size`，`arguments.py:944`），传输是分 bucket 流的。
所以它的开销是 **`M`（CPU replica）+ 512MB（bucket）**，不是只有前者。

##### pack 频率与带宽打满：chunk 变小的代价比直觉小得多

重读 verl 的 `send_weights`（`:397-426`）后，一处初稿的误判要更正：

```python
async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
    if offset + tensor_meta.chunk_size > self.bucket_size:
        torch.cuda.synchronize()
        if readable_op is not None:
            await readable_op.wait_for_complete()    # ← 等上一个 bucket
        readable_op = ReadableOperation(...)          # ← 发起这个
        send_buf, recv_buf = recv_buf, send_buf       # ← 双 buffer 交换
    send_buf[offset:offset+size].copy_(chunk, non_blocking=True)
```

**每个字节都 `copy_` 一次，与 bucket 大小无关——pack 的总字节数恒为 `M`。**
所以「chunk 小 → pack 频率高 → 开销大」是错的。随 chunk 变小而增加的只有：

| 项 | 量级 |
|---|---|
| `torch.cuda.synchronize()` 次数 | 每次 10~50μs，72B 模型 1GB chunk 下约 72 次 ≈ 几 ms，**可忽略** |
| RDMA 提交次数 | 同上 |
| 流水气泡占比 | 见下 |

**verl 的双 buffer 是严格交替的**：`wait_for_complete()` 在发起新传输之前，
所以同一时刻只有一个传输在飞，理论上限是 `T = Σ max(pack_i, rdma_i)`
而非 `T = max(Σpack, Σrdma)`。
但 **pack 是 D2D 拷贝（HBM ~3TB/s），RDMA 是 ~50GB/s**，`pack/rdma ≈ 1/60`，
**所以气泡占比约 1.6%，双 buffer 已经足够打满带宽**。

**结论：chunk 取 1GB 没有带宽劣势**，pack 时间本就被 RDMA 完全掩盖，
chunk 小只是让掩盖的粒度更细，反而**更有利于与 rollout 侧落地格式化重叠**（§3.6）。
真正的取舍是**首包延迟（小 chunk 优）vs 显存占用（小 chunk 优）vs 同步次数（大 chunk 优）**，
而第三项可忽略，所以小 chunk 在两个有效维度上都不劣。

##### 选型结论：是 trade-off，不存在绝对更优

| 维度 | CPU 全模型（miles） | GPU wire buffer（verl / 本设计） |
|---|---|---|
| GPU 显存 | **0** | 1~4GB |
| CPU 内存 | **`M`** | 0 |
| PCIe | `M` 的 D2H，**在关键路径上** | 0 |
| **小模型（<30B）** | **劣**（实测 +68~72%） | 优 |
| **大 MoE（>100B）** | **优**（实测 −47~86%） | 需自己做去冗余才能追平 |
| 格式转换位置 | CPU（借 sglang loader） | rollout GPU |
| 地址稳定性 | 天然稳定，trainer 可自由 offload | 稳定（buffer 不参与 offload） |
| **并行度不匹配** | **自动处理**（CPU replica 按 target 布局构建） | **需自己实现 shard 代数** |

**最后一行是 miles 的隐藏价值**：那份 CPU replica 是按 target 的并行度配置构建的
（`_create_cpu_replica(parallelism_config, ...)`），`load_weights` 自动完成
train 布局 → rollout 布局 的重分片。**这省掉了整个 shard 代数子系统**——
而那正是 PSRL 的 `nixl_spec.py` + `comm_plan.py` 在做的事，
也是 §3.9.2.1 那个 finest mesh 膨胀问题的来源。

> **没有绝对更优。CPU 全模型用 `M` 的 CPU 内存加一次 PCIe D2H，
> 买到「地址稳定 + 自动重分片 + 零 GPU 占用」；
> GPU wire buffer 用 1~4GB 显存买到「零 PCIe + 最低延迟」，但必须自己实现重分片。
> 选型判据首先是模型规模（<30B 选 GPU buffer，实测 miles 在该区间净亏），
> 其次是「GPU 显存是否紧张」与「是否已有 shard 代数基础设施」。**

##### 一处方法论更正，以及拓扑比 buffer 位置更重要

**上面那张 benchmark 的对照组是 NCCL broadcast，不是 verl。**
它只证明「miles P2P vs NCCL broadcast」，不能直接推出 miles vs verl。
要比较后两者，必须看 **verl NIXL 的拓扑**——而那才是主导因素。

`build_topology`（`nixl_checkpoint_engine.py:288`）揭示 verl NIXL 是**串行中继链**：

```python
rollout_kwargs = {
    "rank": list(range(1, rollout_world_size + 1)),
    "prev_agent_metadata": [metadata[0]] + metadata[-rollout_world_size:-1],
    "next_agent_metadata": metadata[-rollout_world_size + 1:] + [None],
}
```

即 `trainer_rank0 → rollout_0 → rollout_1 → ... → rollout_{N-1}`。
`_receive_weight_chunks`（`:452`）每跳「收下 → 转发给下一跳」。
注意 `send_weights` 开头的 `assert self.rank <= 0`——**只有 trainer rank 0 发送**。

| | verl NIXL 中继链 | miles P2P |
|---|---|---|
| trainer 出口 | **仅 rank 0** | **所有 trainer rank 并发** |
| 每 rollout 收到 | **全量 `M`** | **`M / ep`**（EP 分片） |
| 总时间 | `≈ (N + num_buckets) × t_bucket`，**随 rollout 数 N 线性** | **与 N 无关** |

**所以 >100B MoE 下 miles 显著优于 verl，这是结构性的，不需要 benchmark 佐证。**
miles 文档的公式说得很直接：「utilizes `M // src_pp` times more source bandwidth」
+「each target rank receives `sgl_ep` times less data」——**两项乘性叠加**。
这也解释了 miles 的 delta 为何随规模单调改善（−19% → −47% → −70% → −86%）：
**去冗余收益随 EP 规模线性增长**。

##### 1TB 场景：形态优于位置，但前提是做了去冗余

回答「1TB 大 MoE 下 4GB GPU wire buffer vs miles CPU 全模型哪个好」，
必须区分「verl 的实现」与「GPU wire buffer 这个形态」：

**(a) 照搬 verl 的链式 + 4GB buffer**：1TB / 32 节点下链长 32 跳，
每跳 `1TB / 4GB = 250` 轮，即使每轮打满 50GB/s 单跳也要约 20s，
**链式累积到分钟级**——远差于 miles 实测的 7.2s。**这个组合在 1TB 下不可用。**

**(b) GPU wire buffer + 本设计的三阶段去冗余**（§3.5）：

| 项 | miles（CPU 全模型） | 本设计（GPU buffer + 去冗余） |
|---|---|---|
| trainer 出口 | `M`（并发） | `M`（并发） |
| 每 rollout 入口 | `M / ep` | `M / ep` |
| **PCIe D2H** | **`M`，在关键路径** | **0** |
| GPU 显存 | 0 | 8GB（`2×4GB`） |

**此时本设计更优，优势正是省掉的那次 D2H。**
1TB fp8 约 500GB / 63GB·s⁻¹ ≈ **8s**；miles 实测 Kimi-K2 总共 7.2s，
说明它的 D2H 已被 bucket 流水掩盖了大部分，
但**首个 bucket 的延迟与 PCIe 争用仍在**。

> **结论：1TB 下「GPU wire buffer」形态优于「CPU 全模型」，
> 但前提是自己实现了去冗余规划。verl 的链式实现在该规模不可用——
> 它的劣势来自拓扑而非 buffer 位置。**

**这恰好是本设计的定位**：取 verl 的 buffer 形态（零 PCIe）
+ miles / WeightBridge 的去冗余（并发出口 + EP 分片），
而 PSRL 已有的 shard 代数（`nixl_spec.py`）让后者不必从零做。
**三者的关系不是「选一个」，而是「各取一半」。**

**对 PSRL 的结论**：shard 代数（`nixl_spec.py`）**已存在且成熟**，
所以 miles 方案的最大卖点对 PSRL 是重复建设；
加上目标场景是大 MoE 且 GPU 显存紧张——**GPU wire buffer 是对的选择**。
但 chunk 必须可配置：小模型调小、1TB 调到 4GB（§3.5.6 已有这条）。
而 §3.6 的「落地格式化与 RDMA 重叠」针对的正是 verl 留给 rollout 侧的那 884ms，
**做对了就能同时拿到 verl 的低延迟与 miles 的低落地开销**，这是本设计相对两者的主要增量。

**对 §3.7.3.0a 的地址正确性问题，三者抗性相同且都没问题**——
因为它们注册的都**不是参数**，所以参数怎么 offload 都不影响注册有效性。
miles 的 CPU 副本还多一层好处：它连 GPU 显存压力都不占。
**真正不稳定的只有「直读参数」那条路径**（§3.3 的零 buffer 方案），
这再次印证：**零 buffer 与可 offload 互斥**。

**两条共性**：没有任何一家注册**模型参数本身**，
也没有任何一家在 **stage 切换时** register/deregister。

**这对本设计是实质性简化。** 初稿在 colocate 下的 NIXL 使用整个是多余的，
且注册成本不低：`client.py:410` 对每个 key 的每个 shard 建 `nixlRegDList`
并调 `register_memory`，**这是 per-tensor 的**，大 MoE 下就是 §3.5.5 的 14 万次，
colocate 下完全白付。初稿设计的「撤销 holder → 等读者归零 → dereg → offload」
四步协议，**前三步都可以删掉**。

于是 colocate 切换简化为：

```
switch_to_rollout:
  transfer 完成（CUDA IPC + 本地 copy，§3.7.2.1，全程无 NIXL）
    -> offload 权重到 CPU            （无需 dereg，因为本来就没注册）
    -> offload optimizer state 到 DRAM（§3.7.3.1）
    -> rollout resume(tags=["kv_cache"])   ← 拿到完整显存

switch_to_trainer:
  rollout release_kv_cache_memory()
    -> trainer reload 权重 (CPU -> GPU)    （无需重注册）
    -> trainer reload optimizer state
    -> 恢复训练
```

**代价**：72B 下每卡 18GB 的 D2H + H2D，PCIe 63GB/s 下各约 0.3s。
eval 是分钟级，可忽略（PivotRL 实测 shift up ≤8.5s / down ≤6s 同量级）。

##### 3.7.3.0 注册的通用判据：由「读者集合」推导，不由部署模式配置

**第一性原理**：RDMA 注册的唯一目的是让远端 NIC 绕过 CPU 直接访问内存。
所以必要条件只有一条——**存在一个「不共享本进程地址空间、且不共享本 GPU」的读者**。

按这条过一遍四种情况：

| 场景 | 读者在哪 | 访问方式 | 需要注册 |
|---|---|---|---|
| colocate，同 GPU 跨进程 | 同 GPU | **CUDA IPC**（共享物理页） | **否** |
| colocate，同节点跨 GPU | 同节点 | `cudaMemcpyPeer` / NVLink | **否** |
| disaggregate，跨节点 | 远端节点 | RDMA | **是** |
| **disaggregate 内 trainer GPU 的 eval 切换** | 远端 rollout 仍要来读 | RDMA | **是，且 offload 前后必须 dereg/reg** |

**第四行是初稿「colocate 不注册 / disaggregate 注册」这个二分漏掉的格子**：
disaggregate 下 trainer 所在 GPU 也会在 eval 时做 colocate 式切换，
此时对它做 CPU offload，就必须 dereg + 重新 reg，否则远端 RDMA 读会失败。

所以判据不能是部署模式开关，而应当由 index 从 plan 推导——**plan 在握手期已编译，
读者集合是已知的**（§3.5）：

> **需要注册 `(key, shard)` ⟺ 它的 plan 里存在 `node(reader) ≠ node(self)` 的读者。**

这条判据的三个好处：**无需配置项**；**自然覆盖混合部署**（同一 trainer 可同时服务
本地 rollout 走 IPC、远端 rollout 走 RDMA，判定对每个 `(key, shard)` 独立成立）；
**自然覆盖第四行**。它与 §3.1 的 holder 抽象是同一套东西——
**注册是「跨节点 holder」身份的物理实现**。

##### 3.7.3.0a 地址正确性：初稿不保证，这是一处真实缺陷

**只要发生真 offload（而非 TMS pause），重新注册后必须指向新地址。
核实 `client.py` 后发现现有重注册路径做不到这一点。**

`client.py:400-460` 的 re-register 分支复用**缓存的旧地址**：

```python
for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
    slice_addr, slice_len, device_id, mem_type = slice_info   # ← 旧 VA，不是重读 data_ptr()
    contig_by_memtype[mem_type].append((key, shard_idx, slice_addr, ...))
```

**它今天安全，只因为被 `enable_tms_for_temp_buffers` 门控**（`:401`）——
TMS 的 pause/resume 保证**虚拟地址不变**，只换物理页。

**但真 offload（`model.to("cpu")` 再 reload）必然换虚拟地址**，此时这段代码会：
用旧 VA 注册到一块已不属于这些张量的地址 → `get_xfer_descs` 生成错误 descriptor →
**远端 RDMA 读到垃圾数据或段错误**。
而 `nixl_wake_up`（`engine_train_worker.py:243`）传的是缓存的 `self.unified_state_dict`，
其 `Parameter.data` 在 offload 后已被换掉，但快速路径根本不会去读新的 `data_ptr()`。

**三条不变量，按重要性排序：**

**(1) 注册对象应是固定 wire buffer，不是参数本身——这同时解决成本与正确性。**

这是 verl 做法（`nixl_checkpoint_engine.py:281` 注册两块固定 `bucket_size` buffer）
的真正价值，初稿只看到了成本优势：

| | 注册参数（初稿） | 注册固定 wire buffer |
|---|---|---|
| 注册次数 | `O(#shards)`，大 MoE 下 14 万 | **2** |
| offload 后地址 | **变了，必须全量重建** | **不变**，buffer 不参与 offload |
| 远端 descriptor | **必须重新广播** | **永不失效** |

**wire buffer 不参与 offload，所以地址天然稳定。**
参数可以自由 offload/reload，因为远端从不直读参数，它读的是 wire buffer。

**这让 §3.3 的「rollout 直读 trainer 参数张量本身、零额外 buffer」需要加一条前提**：
它要求**参数地址永久稳定**，因而**禁止 trainer offload**。
所以它只适用于「disaggregate + 同精度 + trainer 常驻」这一格；
一旦 trainer 需要在 eval 时 offload，就必须退回 wire buffer 方案。
**零 buffer 与可 offload 是互斥的,这一点初稿没说清。**

**(2) 若确实直读参数，重注册必须从张量重新取址。**

`client.py:401` 的快速路径必须加前置断言：**仅当 TMS 保证 VA 不变时可用**。
真 offload 路径必须走完整重建，重新遍历 `state_dict` 取 `data_ptr()`。
代价是大 MoE 下的 14 万次注册——**这个代价本身就是选 (1) 的理由**。

**(3) 地址变更必须有显式 epoch，不能靠隐式广播。**

`_broadcast_updated_client_infos_from_ps_manager` 是全局操作，
**没有任何机制阻止远端在广播完成前用旧 descriptor 发起读**。需要：

- 每次重注册递增 `registration_epoch`
- descriptor 携带 epoch，远端发起传输时带上
- 本地校验 epoch 不匹配即拒绝，返回明确错误让读者重新解析 plan

这与 §3.4.5 的「撤销 holder → 等在飞读者归零 → 变更 → 重新注册」是同一套协议，
但初稿只覆盖了 colocate sleep 路径，**没覆盖「disaggregate 下 trainer 的 eval 切换」**。
epoch 是那条协议在「地址会变」情形下的补充：
holder 协议保证**没有在飞读者**，epoch 保证**没有陈旧 descriptor**。

###### epoch 与 version 必须两段独立校验，不能合并成单一代次

两者都是单调递增的整数，很容易被实现成一个「generation」字段。
**但它们语义正交，合并会同时产生漏判和误判**：

| | `version` 变化 | `registration_epoch` 变化 |
|---|---|---|
| 含义 | **内容**变了（optimizer step 之后） | **地址**变了（offload/reload 之后） |
| 触发 | 每个训练步 | 每次真 offload，频率低得多 |
| 内容是否变 | 是 | **否**（reload 回来的是同一份权重） |
| 地址是否变 | 否 | 是 |

**四种组合的正确处理**：

| 读者持有的 (version, epoch) | 实际状态 | 处理 |
|---|---|---|
| 都匹配 | 正常 | **允许** |
| version 旧、epoch 匹配 | 常规 staleness | **拒绝并重新解析**（§3.5.2 的版本相等性） |
| **version 匹配、epoch 旧** | **内容对、地址错** | **必须拒绝**——这是会读到垃圾数据的那一格 |
| 都旧 | 两者都过期 | 拒绝 |

**第三行是合并成单一代次会漏掉的情形**：
若只比较一个递增的 generation，「version 匹配而 epoch 旧」在数值上可能表现为
generation 相等（因为 version 没变），于是**校验通过、RDMA 读到错误地址**。

所以落地要求是明确的：

```
准入校验（两段，顺序无关，都必须通过）：
  1. version(request) == version(holder)   -> 否则 CHUNK_STALE，重新解析
  2. epoch(request)   == epoch(holder)     -> 否则 DESC_STALE，重新拉取 descriptor
```

**两个错误码必须区分**，因为恢复动作不同：
`CHUNK_STALE` 要重新走选源（可能换 holder），
`DESC_STALE` 只需重新取同一个 holder 的 descriptor，不必换源。

###### 更正：上面的「准入校验」没有执行点，epoch 由失效 + 重广播取代

上面三段假定「远端发起传输时带上 epoch，本地校验」。**NIXL 读是单边 RDMA**：
源进程不参与读，没有任何代码能看到请求携带的 epoch，也就无处拒绝。
读失败时 `CHUNK_STALE` 与 `DESC_STALE` 同样表现为 `ERR` / `NIXL_ERR_NOT_FOUND`，读者也分不开。

两个保证改由已有机制给出，不需要 epoch：

| 保证 | 机制 |
|---|---|
| 没有在飞读者 | 地址变更前排空（§3.4.5 holder 协议、E1 的在飞计数） |
| 没有陈旧 descriptor 被使用 | 变更方重注册后经 `UPDATE_INFO` 广播；接收方 `remove_remote_agent` 旧对端并丢弃 pull engine。NIXL 对 remote section 是弱引用，驱逐后用旧 dlist 或旧 handle 发起读返回 `NIXL_ERR_NOT_FOUND`（E4 实测），不会读到错误地址 |
| 版本相等 | 每分片的版本戳（`shard_version.py`），读完即校验，撕裂读与陈旧读都会报错 |

「version 匹配、epoch 旧」那一格因此被收敛为：要么 descriptor 已被驱逐（读失败，重新取 info），
要么还没收到 `UPDATE_INFO`（变更方在排空前不得改地址，所以旧地址仍然有效）。
**正确性前提从「读时校验」移到「变更方必须先排空、再改地址、再广播」这个顺序上**。

**关于约束 (c)：这条路径不需要 TMS。** 核实了 pinned 的 vLLM
（`third_party/vllm`）后发现它已经有正好合适的原语：

- `CuMemAllocator.discard(tags)`（`cumem.py:300`）丢弃指定 tag 的映射**而不做 CPU 备份**。
- `EngineCore.release_kv_cache_memory()`（`core.py:981`）就是 `discard(("kv_cache",))`，
  且前置断言「必须先 pause 完成」「executor 内存必须 resident」，正是切换时的状态。
- KV cache 本就单独打了 `kv_cache` tag（`gpu_worker.py:842`），权重打 `weights` tag
  （`gpu_worker.py:569`），粒度天然分离。
- 反过来，`SleepModeBackend.suspend(level)`（`sleep_mode_backend.py:133`）在 level 1 会把
  `weights` **offload 到 host**，level 2 丢弃后需从 model source 重载。这两种都违反我们的目标。
  所以**不能用 `sleep()`，要用 `discard(("kv_cache",))`**。

因此 colocate 切换的 rollout 侧 = `pause` + `release_kv_cache_memory()` + 后续 `wake_up(["kv_cache"])`，
全部是 vLLM 原生 API。

**train 侧要释放的是 optimizer state 与 activation，而这里有一处设计缺口。**
初稿说「用现有的 storage resize helper 即可」，但核实后发现
**`_sleep_fsdp_model` / `_sleep_megatron_model` 都不碰 optimizer state**：

```python
# _sleep_megatron_model（engine_train_worker.py:376）只处理 DDP buffer
buf.param_data.untyped_storage().resize_(0)   # 参数
buf.grad_data.untyped_storage().resize_(0)    # 梯度
# ← 没有 optimizer state
```

`_sleep_fsdp_model` 同样只遍历 `named_parameters()` 与 `named_buffers()`。
今天这条路径能工作，是因为走的是 `nixl_sleep(level=2)` 的全量丢弃 + 事后回灌；
**改成「权重区不动」之后，optimizer state 的释放必须显式处理**，
否则 colocate 下 train 与 rollout 的显存会同时占着。

**正确手段是 verl 已有的 `optimizer_offload`**（见 §3.7.3.1），
不是新写一个 helper，也不是 TMS。

**TMS 的保留范围**（符合约束 c）：仅在「`optimizer_offload` 的 D2H 搬运
在切换路径上太慢」时作为可选后端。权重区和 KV cache 都不走 TMS。
`tms.range` 不需要新增 weights/kv_cache 粒度，
上一版设计提的那条需求因为改用 `discard` 而消失了。

#### 3.7.3.1 verl 的两个 offload 开关：一个必须禁用，一个正是切换需要的

verl 有两个独立的 offload 开关，**它们在本设计中的地位截然相反**，
而且两者都不是「colocate 切换」粒度的——它们挂在**每个 `train_mode()` 上下文**上
（`base.py:330-336` 的 `__enter__` 搬上 GPU、`__exit__` 搬回 CPU），
即**每个训练步两次**：

```python
# BaseEngineCtx._context_switch（base.py:314）
self.engine.to(device=device,
               model=self.engine.is_param_offload_enabled,        # param_offload
               optimizer=self.engine.is_optimizer_offload_enabled, # optimizer_offload
               grad=self.engine.is_param_offload_enabled)
```

| 开关 | 搬什么 | 本设计 | 理由 |
|---|---|---|---|
| **`param_offload`**（自动开关，挂每个 `train_mode()`） | 模型参数 + 梯度 | **必须禁用** | ① `offload_fsdp2_model_to_cpu` 就是 `model.to("cpu")`，重新分配每个 DTensor 的 local storage，**注册过的 GPU 指针悬空**；② 频率是**每训练步两次**，而 NIXL 重注册昂贵（§3.3），每步两次不可接受。**注意这不等于「权重不能 offload」**——colocate 的 stage 边界 offload 是必须的（§3.7.3），只是频率是每 eval 一次而非每步两次，且走手动 `to()` 而非这个自动开关 |
| **`optimizer_offload`** | optimizer state | **保留，且正是上面缺口的解** | ① 它搬的是 optimizer 状态，**完全不碰注册过的参数**，与直供无冲突；② colocate 切换要腾的显存恰好就是 optimizer state（权重区按设计不动）；③ 它是现成的、经过验证的路径 |

**所以 colocate 切换的 train 侧应当是**：

```
switch_to_rollout:  engine.to(device="cpu", model=False, optimizer=True, grad=False)
                    （即只搬 optimizer state，参数与注册都不动）
switch_to_trainer:  engine.to(device=cuda, model=False, optimizer=True, grad=False)
```

注意第二个参数 `model=False` 是关键——**它把「释放 optimizer」与「搬动参数」解耦**，
而 verl 的 `to()` 本来就支持这个粒度（`transformer_impl.py:844` 的签名
`to(device, model=True, optimizer=True, grad=True)`，注释原文
「executes irrespective of offload config. It serves as manual control」）。
PSRL 需要的正是这条手动控制路径，而不是 `param_offload` 的自动开关。

**配置校验据此分两条**：拒绝 `param_offload=True` 与 `offload_policy=True`（§3.9.3c），
但**不要**连带拒绝 `optimizer_offload`——它是切换路径的依赖项。

**再强调一次区别，因为这两件事极易混淆**：

| | 频率 | 走哪条路 | 本设计 |
|---|---|---|---|
| `param_offload` 自动开关 | **每训练步两次**（`train_mode()` 进出） | `BaseEngineCtx._context_switch` | **禁用** |
| colocate stage 边界的权重 offload | **每次 eval 一次** | 手动 `engine.to("cpu", model=True, ...)` + 撤销 holder + dereg | **必须做**（§3.7.3） |

同一个 `offload_fsdp2_model_to_cpu` 底层函数，**频率差两个数量级**，
所以一个不可接受、一个是最佳实践。

colocate 的显存账要**分 transfer 期与 eval 期两段看**（`M` 为 bf16 全量权重）。
transfer 期两份权重必然同时在 GPU 上（源与目标），eval 期 train 侧已 offload：

| 后端 | transfer 期峰值（train + rollout） | **eval 期**（train 已 offload） |
|---|---|---|
| Megatron bf16 + fp8 rollout | `M + 0.5×M = 1.5×M` | **`0.5×M`** |
| Megatron fp8（TE `fp8_param`）+ fp8 rollout | `0.5×M`（共享同一份） | **`0.5×M`** |
| **FSDP2 fp32 + fp8 rollout** | **`2×M` + `0.5×M` = `2.5×M`** | **`0.5×M`** |
| **FSDP2 fp32 + bf16 rollout** | **`2×M` + `M` = `3×M`** | **`M`** |

**两列的用途不同**：transfer 期峰值决定「会不会 OOM」，
eval 期占用决定「KV cache 有多大、rollout 并发多高」。
初稿只算了第一列并把它当成常驻，于是得出「两份常驻」的错误结论（§3.7.3 已修正）。

transfer 期的峰值是 FSDP2 在 colocate 下的真实代价：
**fp32 参数本身就是 Megatron bf16 的两倍**，`3×M` 那一行在大模型上很容易 OOM。
所以 **colocate 应当优先选 Megatron 后端**，FSDP2 + colocate 技术上可行但峰值不划算。

FP8-native train（第二行）是唯一两列都是 `0.5×M` 的配置——
两侧共享同一份 fp8 权重，**transfer 退化为无**，也就没有峰值问题。
这是 FP8 与 colocate 在本设计里真正合流的地方，也是 colocate 的推荐配置。


#### 3.7.4 同节点跨 GPU：按拓扑移除注册（设计与实测）

**问题。** colocate 的 validate 实例是 pull 的目的地：它每次 wake 都要把 439 个分配注册
给 NIXL（151 个 allocator segment，5.06 s），占 7.3–9.8 s validate 切换的七成。注册存在的
理由只有一个——让**网卡**能寻址目的显存。validate 实例在 colocate 下的所有源（trainer rank
与同节点 peer）都在本节点内，没有任何一个字节需要经过 NIC。

**第一性原理：传输原语应当与拓扑匹配。** 同设备 → CUDA IPC 映射 + 设备拷贝（已实现，
`_ipc_cells`）；同节点跨设备 → IPC 映射 + **P2P 拷贝**（`cudaMemcpyPeerAsync` / NVLink，
本设计新增）；跨节点 → NIXL/RDMA（保留）。MR 只对第三种边有意义，因此注册策略应从
「默认注册」改为「**只对必须走 NIXL 的边注册**」。判定分两侧，都是逐 shard 的：

```
读者需要 MR  ⇔  它的某个源在跨节点
            或  它要读的某个 shard 的源分配没有 IPC handle

源需要 MR    ⇔  它有跨节点读者
            或  某个读者要读的 shard 在它这里没有 IPC handle
```

逐 shard 而不是逐 client 是必要的：vLLM 的 allocator 拒绝导出**全部**权重
（实测 435/435 个分配 `cudaIpcGetMemHandle` 返回 `cudaErrorInvalidValue`），
逐 client 的「可导出」位会让同节点所有读者白白注册。源侧同理：一个没人读的
validate 实例即使自己导不出任何分配，也不需要注册——它的读者根本不存在。

**机制。** 源端已有的 `export_region`（`reduce_tensor`，整个 allocation 一个 handle，
353 B/个）与读者端的 `RegionMapper` 可以直接复用，不需要新的导出格式：

1. `_spec` 判断源与本机 `node_ip` 相同且源已导出 handle → 生成 peer-cell：
   `RegionMapper` 按需导入源 region，读者用 `local.copy_(view, non_blocking=True)` 在
   copy stream 上完成设备拷贝。源 region 的 strided 视图用与 `_ipc_cells` 完全相同的
   `as_strided` 构造，字节语义不变。
2. **设备编号翻译。** exporter 与 importer 对同一物理 GPU 的序号通常不同（train rank
   只看见 1 张卡时它是 0，validate 进程看见整节点时它是 `nvidia-smi` 序号）。handle
   里记录的是 exporter 的序号，必须用 `device_uuid` 解析成本地序号后重写
   `rebuild_cuda_tensor` 的 `storage_device` 参数；否则导入的张量会被记在错误的设备上。
3. **设备不可见不构成障碍。** 每个 vLLM 实例的 `CUDA_VISIBLE_DEVICES` 只有它自己的
   GPU，但 `cudaIpcOpenMemHandle` 仍能映射同节点其他 GPU 的 handle（lazy peer access），
   实测逐位正确且 390 GB/s。因此无需改部署的可见性，也不需要 IPC 白名单。
4. **按需导入**：`adopt` 只登记 handle 目录，第一次读到某个 allocation 时才打开它。
   实测打开 219 µs/handle（400 个 warm），且首次打开有约 70–80 ms 的进程级 CUDA
   context 初始化（一次性，不是每 handle）。按需导入让切换成本与**实际读取的分配数**
   成正比，而不是与发布数成正比。
5. **handle epoch**：源在 wake 时重新注册，旧 handle 指向已释放的显存。`UPDATE_INFO`
   到达时对源调用 `RegionMapper.forget`，下一次拉取重新导入。这条对**现有同设备 IPC
   路径同样必要**——`forget` 此前从未被调用。
6. **完成的可见性**：一次 peer copy 也在读 writer 正在写的页，和 NIXL read 一样会被
   抢占撕裂。因此 copy 批次用一个 CUDA event 标记完成，抢占判据只在 event 信号后才把
   这些 key 归为 completed；否则会把半拷贝的 key 认证为安全。
7. **失败回退**：`export_region` 是 best-effort。读者在计划阶段就检查源在本 shard 上
   是否可导出；不可导出则该读者注册，同时源侧检测到「有读者要读缺失 shard」后也注册，
   两条 NIXL 边同时存在，回退路径始终可用。若运行时导入意外失败，读者调用
   `ensure_rdma_registered` 就地补注册（慢路径，仍正确）。

**PoC 实测（H20，同节点两 GPU，NVLink P2P）：**

| 量 | 实测 |
|---|---|
| `reduce_tensor` 导出 | **14 µs/handle**（400 个 1 MiB 分配，5.7 ms） |
| `cudaIpcOpenMemHandle` 导入（warm） | **219 µs/handle**（同 400 个 handle，87.7 ms） |
| 跨 GPU 拷贝 1 GiB（目标已预热） | 2.7–2.8 ms，**390–393 GB/s** |
| 源 GPU 对读者**不可见**（CVD 单卡） | 打开 12 ms、逐位正确、**390 GB/s** |
| 首次写入全新目标 | 该次 1.1 GB/s（页表建立），之后 390 GB/s |
| vLLM/TMS 进程的分配导出 | **435/435 失败**（`cudaErrorInvalidValue`） |

最后一行改变了 peer diffusion 的结论：**vLLM 进程（TMS/CuMem 分配）只能做 peer copy
的目的地，不能做源**。因此扩散只能折叠**跨节点**上游读；同节点上游已是设备拷贝，
再经过一个 vLLM 中间实例既多一跳，又因中间实例不可导出而迫使 follower 注册。

**成本模型与预期收益。** 一次 validate wake 的读者读取整个模型，约 439 个分配：

| 项 | 现状（NIXL MR + pull） | peer copy |
|---|---|---|
| 注册 / 导入 | 5.06 s（151 segment） | **0**（读者跳过注册） |
| 数据搬运 | 0.8–1.3 s（~12 GB/s 端到端） | 设备拷贝，390 GB/s |
| 源端导出 | — | 6 个 trainer 分配，<0.1 ms |
| **validate 切换** | **7.3–9.8 s** | **2.57 s**（实测） |

打开成本是唯一不可忽略的新项，且是 **O(本 rank 实际读取的分配数)**、与集群规模无关、
各 rank 并行；TB 级 MoE 上每个 rank 读到的分配数由其 shard 份额决定，不会随 fleet
增长。vLLM 侧的分配无法 IPC 导出时，用 CUDA VMM 的
`cuMemGetHandleForAddressRange`（按 VA 区间导出，一个 fd 覆盖多个映射）是第二阶段
让 vLLM 也能作为 source 的路径；它要求两端都用 VMM 分配与映射。

**与 peer diffusion 的关系。**

- **扩散只应折叠跨节点读。** 设计初衷是「一次昂贵 upstream + 同节点 peer 读」；当
  upstream 已经在同节点时，读者直接拷贝源就是设备速度，再加一个 designate 中转只会
  多一跳，并把一个 vLLM 实例放进数据路径——而 vLLM 的分配无法 IPC 导出（见上表），
  follower 反而必须注册。因此 `_make_peer_groups` 现在只对「至少一个源在别的节点」的
  (reader, key) 成组。disaggregate rollout 的 trainer→gen 边跨节点，扩散保留；
  colocate 与 validate（源是自己节点的 trainer）不再扩散。
- **vLLM 实例只能做目的地。** IPC 导出在 trainer（标准 caching allocator）上成功，
  在 vLLM/TMS 进程上 435/435 失败。所以 validate/gen 读者的 peer 源必须是 trainer；
  「reader 服务 reader」在 vLLM 侧要等 VMM 导出（`cuMemGetHandleForAddressRange`）。
- **不改变扩散语义。** 版本提交、共享锁、v2 门闩与抢占仍由 pull 引擎/协调器执行，
  与 NIXL 路径完全相同；改变的是「谁读谁」的成组条件与搬运原语。
- **冲突与边界。** NIXL 仍是控制面与跨节点数据面的唯一通道；peer copy 只在同节点、
  可导出、且可映射的 cell 上启用。新增的不变量是**源重新注册必须使映射失效**
  （`UPDATE_INFO` → `forget`），否则会读到已释放显存；跨节点绝不能走 IPC
  （handle 不跨节点；`node_ip` 不同即拒绝）。

**对整体架构的影响。**

1. **plan 描述意图，执行层选择原语**：`(reader, key, source)` 的传输由节点拓扑与源的
   `ipc_regions` 推导；注册策略成为 plan 的函数，而不是部署模式的函数。同一套原语覆盖
   colocate、disaggregate、单机与多机。
2. **MR 从「每个目的地」变成「必须走 NIXL 的边」**：colocate 验证与单机部署不再有
   注册开销；多机 rollout 的注册保持一次性。规模上，注册成本从
   O(模型分配数 × 每次切换) 降为 O(跨节点客户端 × 一次性)。
3. **数据面路径与拓扑一致**：同设备与同节点跨 GPU 走 HBM/NVLink 速率，不再让同节点
   流量绕道 RDMA 协议栈。

**实施与验证（已完成）。** `_peer_cells`（现为 `_ipc_cells` 的跨设备分支）+ 按需
`adopt` + `UPDATE_INFO` 时 `forget` + 复制批次的 event 完成语义；注册策略改为两侧逐
shard 判定；vLLM 源不可导出时的源侧回退；扩散只对跨节点上游成组。验证：
（a）单节点双 GPU（含源 GPU 不可见）逐位 PoC；（b）坐标判据单节点 train 4 + gen 4，
两轮 **399 key × 4 client 逐位通过、0 失败**，32.76 GB 全部走 `ipc`（0.041 s，
213–243 GB/s/client）；同一判据 2 节点（跨节点）也通过且 `ipc = 0`；（c）2 节点 10 步
fp8gen e2e（GSM8K）validate 切换 **2.57 / 2.11 / 1.90 s**（原 7.3–9.8 s）、trainer 切换
**1.68 / 1.63 / 1.56 s**（原 1.66–1.88 s）、0 次 deferred registration、初始/最终验证
0.1986 / 0.6884（历史 0.1948–0.1986 / 0.6520–0.6672）。


#### 3.7.5 rollout 与 training 共卡部署

`psrl.colocate_rollout_and_train=True` 让 rollout、train、validate 共享一个节点的 8 张卡。
资源池按 **train 0.6 / rollout 0.3 / validate 0.09** 的 Ray 份额分配；份额只是调度账目，
仍然每 worker 独占一张卡的可见性。取值保证每张卡每种角色最多一个 worker：任意两个同角色
bundle 都放不下（`2×0.6>1`、`0.6+2×0.3>1`、`0.6+0.3+2×0.09>1`），份额之和 0.99 留出
浮点余量。

三条与传输无关但决定该部署能否成立的约束，均已实测：

1. **初始化顺序**。rollout 服务器必须在 trainer 加载权重之前初始化，否则 vLLM 按已被
   trainer 占用的空闲显存规划 KV，部分卡直接启动失败。共卡模式下先初始化 validate
   （随后睡眠）、再 rollout、最后 actor；不共卡时仍保持 rollout 与 trainer 并行初始化。
2. **显存配额**。trainer 与 rollout 常驻（约 65 GiB/卡），validate 训练时睡眠、切换时
   唤醒；两个 vLLM 角色同时唤醒会超出 96 GiB 卡。因此 rollout/validate 的
   `gpu_memory_utilization` 取 0.3，`max_model_len` 取 prompt+response（10240），而不是
   模型原生的 40960。
3. **权重传输不变**。共卡读者的源仍在同节点，走 §3.7.4 的 IPC：16.38 GB/实例
   **0.078–0.083 s**，全部跳过 NIXL 注册，0 次回退。validate/trainer 切换分别
   **2.74/1.53 s** 与 **1.22/1.82 s**，与不共卡部署相同——共卡不增加传输开销。

实测（Qwen3-8B/GSM8K，10 步，8 卡）：初始/最终验证 **0.1766 / 0.6429**；同配置 bf16
不共卡基线 0.1706 / 0.6998，逐步 reward 均值 0.381 vs 0.370。精度与传输一致，最终验证
差异来自单种子 RL 的批次组成；训练 step 85–91 s vs 24–36 s，是共卡分摊 GPU 的预期代价
（8 卡 vs 16 卡）。完整数据见 `weight_transport_FINDINGS.md` §13.5。

#### 3.7.6 架构 / 并行 / 精度覆盖（实测）

传输层用坐标判据逐位验收（train 按 `(key, version, 全局坐标)` 填哈希，rollout 按自己的
sharding 复核，两轮换版本），不跑完整 RL。覆盖与结论：

| 维度 | 已验证的取值 |
|---|---|
| 架构 | Qwen3 dense（GQA）、Qwen3.5 dense（GDN 混合注意力）、Qwen3-MoE、Qwen3.5-MoE（256 experts）、DeepSeek-V3/MLA（Moonlight） |
| 训练引擎 | Megatron、FSDP2 |
| 并行 | TP 对齐（TP4→TP4、TP2→TP2）、TP 不匹配（TP4→TP2，staging 1 GiB）、EP4/EP8→TP2/TP4 重分片；共卡与 disaggregate |
| 精度 | bf16→bf16、bf16→block FP8（位与 scale 逐位） |
| 规模 | 486.78 GB/次同步（Qwen3-235B-A22B，8 个 gen GPU，稳态 3.29 s） |

复杂度在实测范围内保持设计值：payload O(#shards)（单 client 0.07–0.4 MB）、
注册 O(跨节点客户端 × allocator segment)（共卡为 0；235B 427 region/12.5 s 一次）、
导入 O(读者实际触达的 allocation)（0.22 ms/handle）、wire 时间随字节线性
（148 GB/s 聚合、8 读者）。EP→TP 重分片的发布副本是唯一随模型/并行组合增长的新项
（235B 23.5 GiB/rank），由 `publish_copy_budget_mb` 封顶，需按显存预算配置。

两个架构边界（均已写入 FINDINGS §14）：GDN 的 `linear_attn.in_proj_ba` 在 block FP8 下
必须保持 bf16（64 行 tile 不能被 TP shard 切开），因此 **Qwen3.5 的 FP8-train + FP8-rollout
组合当前不支持**；DeepSeek-V3/MLA 的 `W_UK_T`/`W_UV` 是 `kv_b_proj` 的运行时吸收结果，
不参与传输；每次 pull 落地后读者在 `weights_already_processed()` 下调用 vLLM 上游
`process_weights_after_loading`（MLA 吸收、MoE kernel、量化 stamp 一并重建，
实测 4.5–8.2 ms），不支持的量化方法回退到仅 MLA 的 refresh。

### 3.8 FP8：贯穿全链路的 wire 格式

> 本节给出 wire 形态与精度矩阵。**两种目标组合（rollout FP8 + train FP8、
> rollout FP8 + train BF16）的完整架构在 `fp8_weight_transport.md`**：
> scale 作为一等 plan 公民的可组合性证明、量化结构从 rollout 读取而非手写白名单、
> cast 产物在 pull 语义下的生命周期、colocate 零拷贝形态、以及落地/验证清单。

**依赖边界先说清楚：本节所有原语都在 PSRL 已有依赖里，不 import SkyRL / miles。**

| 需要的能力 | 来源 | 符号 |
|---|---|---|
| blockwise 量化（128×128 tile） | vLLM | `vllm/utils/deep_gemm.py::per_block_cast_to_fp8` |
| per-tensor / per-group 量化 | vLLM | `_custom_ops.py::scaled_fp8_quant`、`fp8_utils.py::per_token_group_quant_fp8` |
| MXFP8 量化（1×32，E8M0） | Transformer Engine | `transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Quantizer` |
| fp8 参数 layout 与命名 | vLLM | `fp8_utils.py::create_fp8_weight_parameter` |
| 运行时格式化（UE8M0 / TMA） | vLLM | `fp8_utils.py::{process_fp8_weight_block_strategy, requant_weight_ue8m0_inplace}` |
| 分块合法性校验 | vLLM | `fp8_utils.py::{validate_fp8_block_shape, validate_fp8_block_shape_moe}` |
| FP8-native 训练参数 | Transformer Engine | `fp8_param` 路径下的常驻 fp8 参数 + scale |

SkyRL 的 `quantize.py` / `vllm_format.py` 只作为**参考读物**，说明两种 wire 的字段命名与
`quant_method` 取值该怎么填，其算法本身就是上表这些 vLLM/TE 原语的组合。PSRL 自己实现这层
组合逻辑（约数百行），放在 `psrl/utils/fp8/` 下。

**一条原则：量化发生在源头，链路不感知精度。** 两种训练形态产出**同一种** wire payload：

| train 形态 | 产出路径 | 代价 |
|---|---|---|
| BF16 train | train GPU 上 blockwise / MX cast（复用 vLLM `fp8_utils` 或 TE kernel） | 一次 amax + cast，与 chunk 流水重叠，瞬态 `O(chunk)` |
| FP8 train（TE `fp8_param`） | TE 已持有 fp8 参数 + scale，只做 **scale 格式转码**（TE blockwise ↔ `weight_scale_inv`，MXFP8 E8M0 ↔ compressed-tensors uint8） | 无重量化，不引入额外数值误差，省掉 amax，**额外显存 0** |

FP8 train 下**不要「反量化回 bf16 再量化」**：那会白付两次精度损失和一次全量 amax，
而 TE 的 scale 与 vLLM 的 scale 只差一个编码约定。注意 FP8-native 形态在约束 (b) 下
是**最优形态**，因为 trainer 持有的就是 rollout 要的字节，直供零转换零缓冲。

链路收益：传输字节减半。且因为没有 PS 存储，今天 `PSStoragePlan` 为 train/gen dtype
不等准备的第二块缓冲和 `transfer_train_to_gen` 拷贝**一并消失**（这两者本来就是约束 (b)
要删的东西）。

**运行时格式化**（转置 / TMA 对齐 / UE8M0 打包）留在 rollout 落地后做，复用 vLLM 的
`process_fp8_weight_{tensor,block}_strategy` 与 `requant_weight_ue8m0_inplace`，尽量 in-place。
train 与 inference 并行度匹配时源头可直接产出终态布局，此时 `is_weights_pre_processed()`
为真，这一步退化为 no-op。

##### 3.8.0 精度配置矩阵：哪些组合被支持

矩阵必须按 **(后端, train dtype, rollout dtype)** 三元组而不是 (train dtype, rollout dtype)
二元组列举，因为**后端决定了 train dtype 的可取值**：

| 后端 | train 参数 dtype | rollout | 路径 | 状态 |
|---|---|---|---|---|
| Megatron | bf16 | bf16 | 同精度直读 | 支持。disaggregate 用版本级等待（优化），colocate 用时序互斥 |
| Megatron | bf16 | fp8 | cast | **推荐默认**，冲突归零且带宽减半 |
| Megatron | bf16 | fp4 | cast | 支持，需 §3.8.1 的 swizzle 处理 |
| Megatron | fp8（TE `fp8_param`） | fp8 | 同精度直读 | 支持，只做 scale 转码，零额外字节 |
| Megatron | fp8（TE） | fp4 | cast | 支持，但**量化源必须是 fp32 master params**（§3.4.2 约束 1） |
| **FSDP2** | **fp32**（`engine/fsdp.yaml:33` 默认） | bf16 | cast | 支持 |
| **FSDP2** | **fp32** | fp8 | cast | 支持，fp32→fp8 单次量化，精度优于 Megatron 的 bf16→fp8 |
| **FSDP2** | **fp32** | fp4 | cast | 支持，**唯一能从 fp32 单次量化到 fp4 的后端**，无需触碰 optimizer |
| **FSDP2** | **bf16**（8 个 verl 生产 example 在用） | bf16 | 同精度直读 | **支持**，见 §3.8.0a。**disaggregate 下版本级等待是正确性必需** |
| **FSDP2** | **bf16** | fp8 | cast | **推荐**：省显存且冲突归零，是 FSDP2 大模型下的最佳配置 |
| **FSDP2** | **bf16** | fp4 | cast | 支持。注意量化源是 bf16 而非 fp32，精度略低于 fp32 行 |
| ~~FSDP2~~ | ~~fp8-native~~ | — | — | **不支持**：FSDP2 侧无 TE `fp8_param` 等价物（`transformer_impl.py` 全文无 fp8），TE 的 `fp8_autocast` 只影响 GEMM 计算精度，参数仍以 `model_dtype` 存储。将来接入 torchao `Float8Linear` 再扩 |

**十一格里十格支持，唯一不支持的是 FSDP2 的 fp8-native 训练**，
而那是上游能力缺失而非本设计的限制。

**FSDP2 的 train dtype 由 `model_dtype` 决定，默认 fp32 但常被改为 bf16。**
verl 的 FSDP2 引擎以 `model_dtype`（默认 `fp32`）构建模块并
`module.to(torch_dtype)`（`transformer_impl.py:333`）；
`MixedPrecision(param_dtype=bf16)` 只影响 forward all-gather 的临时副本，
**不改变持久参数的 dtype**。所以 `model.state_dict()` 返回的 DTensor
与 `model_dtype` 一致，`fsdp_converter.py` 注册给 NIXL 的就是这个 dtype。

`ray_trainer.py:1962` 今天按后端硬分流，**这本身是一处需要修正的简化**：

```python
train_model_dtype = torch.bfloat16 if ...strategy == "megatron" else torch.float32
```

它假定 FSDP2 恒为 fp32，而实际应当读 `engine_config.model_dtype`。
**这是落地项 9d-a 要修的一条**，否则 `model_dtype=bf16` 下
PSRL 会按 fp32 准备缓冲，与真实字节不符。

**两条分支的后果**：

| `model_dtype` | 与 rollout dtype 的关系 | 路径 | 冲突 |
|---|---|---|---|
| `fp32`（默认） | 永不相等（rollout 不会是 fp32） | 恒定 cast | **恒为零** |
| `bf16`（8 个生产 example 在用） | **可能相等** | 同精度直读或 cast | 同精度下需版本级等待（§3.8.0a） |

代价要一并记下：fp32 下直读 FSDP2 参数的出口是 `2×M_bf16` 字节，
所以 cast kernel 在该配置下不是「可选优化」而是必须项，
`2×chunk` 的 wire buffer 对 FSDP2 也是必须项（Megatron 的 bf16+bf16 组合可以省掉它）。

##### 3.8.0a 「FSDP2 + bf16 + bf16」是常见配置，必须支持

初稿把这个组合列为「配置校验拒绝」，**这是错的**。
核实 verl 的 example 后发现 `model_dtype=bf16` 出现在 **8 个生产脚本**里
（`run_qwen3_8b_fsdp.sh:82`、`run_qwen3_next_80b_a3b_fsdp.sh:84`、
`run_gpt_oss_20b_fsdp.sh:98` 等），**是主流配置而非边缘情形**。
大模型下 fp32 参数占两倍显存，改 bf16 是标准做法。

**这个组合的真实约束比初稿写的窄。** FSDP2 确实没有 master params
（`build_optimizer(module.parameters())`，`optimizer.py:298`），
所以 optimizer step 原地改写参数内存。但这不意味着冲突窗口是整个 step——
**§3.4.2 约束 2 的「版本级一次性等待」本来就覆盖这种情况**：

```
optimizer step 完成，准备发布 v+1 之前：
  等待所有对版本 v 的在飞读者计数归零（一次性，不分 chunk）
  -> 发布 v+1
trainer 执行下一步 forward/backward 期间：
  rollout 拉取 v+1
```

**关键在于等待点在「发布前」而非「每次写之前」。** 对 Megatron 这是优化
（因为它的冲突窗口本来就窄）；对 FSDP2 bf16 这是**必需**，
因为它的整个 optimizer step 都在写参数内存。两者用同一套机制，只是必要性不同：

| 后端 + 精度 | 冲突窗口 | 版本级等待的性质 |
|---|---|---|
| Megatron bf16+bf16 | param 回写那一瞬（master params 隔离） | 优化（8c，可不做） |
| **FSDP2 bf16+bf16** | **整个 optimizer step** | **必需**，不能省 |
| FSDP2 fp32+任意 | 无（cast 产物隔离） | 不适用 |

**代价要说清**：FSDP2 bf16 下 trainer 在发布前必须等读者清零，
而它的下一步 forward 就会开始写参数——所以**等待窗口比 Megatron 更难隐藏**。
若实测这个等待显著，两条退路：

1. **改用 `model_dtype=fp32`**（走 cast 路径，冲突归零，代价是 2× 参数显存）
2. **改用 fp8 rollout**（同样走 cast 路径，且省显存）

**配置校验不拒绝，但在 FSDP2 + 同精度 + disaggregate 下打印一条提示**，
说明「此配置依赖版本级写屏障，若等待时长显著请考虑 fp8 rollout」。

**colocate 下无此问题**：时序互斥让 rollout 在 transfer 期间暂停，
不存在「读者在飞」，所以 FSDP2 bf16 + colocate 是干净的（§3.7.1）。

### 3.8.1 扩展到 FP4：架构可行，但两处必须显式处理

vLLM 已有 FP4 支持（`quantization/modelopt.py` NVFP4、`quantization/mxfp4.py` MXFP4、
`utils/nvfp4_utils.py`），所以「源头量化、链路不感知精度」的原则可直接延伸。
但 FP4 有两个 FP8 没有的结构性差异：

**差异一：两级 scale。** NVFP4 是 `weight` + `weight_scale`（per-block，fp8_e4m3）
+ `weight_scale_2`（per-tensor，fp32），见 `modelopt.py:854` 的
`uses_weight_scale_2_pattern()` 与 MoE 的 `w13_weight_scale_2` / `w2_weight_scale_2`。
**atomic group 要从「weight + scale」扩展为「weight + scale + scale_2」**，
这是参数化而非新机制。另有「w1 与 w3 的 `scale_2` 必须相等」的约束
（`modelopt.py:988` 断言），属量化侧正确性，由源头保证。

**差异二：scale 要 swizzle，且 swizzle 与分片不交换。**
`nvfp4_utils.py:13` 的 `swizzle_blockscale` 把 block scale 填充到
`(M→128 对齐, K→4 对齐)` 后做
`reshape(B, M/128, 4, 32, K/4, 4).permute(0,1,4,3,2,5)`。
swizzle 跨 128 行分组，而 TP 切分也在行维度，**两者会互相破坏**。

后果：下文「并行度匹配时源头直接产出终态布局、`is_weights_pre_processed()` 为真、
落地格式化退化为 no-op」这条优化**在 FP4 下不成立**。FP4 必须
**传未 swizzle 的 canonical scale，在 rollout 侧落地后再 swizzle**，
这正属于 §3.6「落地格式化与 RDMA 重叠」的那类工作。
架构上有位置，但**必须显式禁用 FP4 的 no-op 优化**，否则得到静默错误的 scale。

收益：FP4 相对 FP8 把传输字节再减半，1TB 模型降到 `M/4`，wire buffer 同样减半。

**分块对齐是断言而非风险**：在 block 对齐的 canonical 层面量化，raw block scale 形如
`(⌈N/128⌉, ⌈K/128⌉)`，对任意 block 对齐的分区都是干净子块。每个量化 key 过
`validate_fp8_block_shape` / `_moe`。不整除的 模型×TP 组合 vLLM 引擎本身也会拒绝。
miles 记录的 Kimi-K2 需把 `[128,128]` 重量化为 `[64,64]` 才能跑 `tp=32`，就是这条约束的
真实表现，应在配置校验期报出，而不是传输中途。

### 3.9 训练后端与 vLLM 格式对齐：Megatron 与 FSDP2 并列支持

PSRL 有两个训练后端，各自持有不同的权重布局，在直供路径下都需要生产
vLLM 能直接读取的 wire 格式。

**FSDP1 不在支持范围内，这是既有决定而非本设计的新增约束。**
`utils/config.py:303` 在 NIXL 启用时直接拒绝它：

```python
assert config.train_actor_rollout_ref.actor.strategy != "fsdp", (
    "FSDP1 is not supported for NIXL because it uses flat_param"
)
```

理由是 FSDP1 的 `flat_param` 把多个参数打平进一块连续存储，
导致 state dict 无法保证 in-place 表示（`fsdp_converter.py:253` 的 deprecation warning
原文：「cannot guarantee the in-place representation of the state dict」），
而直供的整个前提是**注册的就是参数本身**。
所以本节只讨论 **FSDP2**，`fsdp_converter.py` 的 `fsdp` 分支应随本设计一并删除
（按「不保留兼容路径」的原则）。

#### 3.9.0 配置空间的真实维数

本设计要同时覆盖的配置空间是 **部署模式 × 训练后端 × 精度组合**，
但这三者不是独立的三维：**后端约束了精度（§3.8.0），精度决定了走哪条冲突路径
（§3.4.2），部署模式决定了用哪种冲突机制（§3.7.2）。** 展开后只有四格：

| | **Megatron** | **FSDP2** |
|---|---|---|
| **disaggregate**（各 instance 错开暂停） | bf16+bf16 走同精度直读，版本级等待是**优化**（master params 隔离）；→ fp8/fp4 走 cast，无冲突 | `model_dtype=fp32`（默认）恒走 cast、无冲突；**`model_dtype=bf16` + bf16 rollout 走同精度，版本级等待是必需**（§3.8.0a） |
| **colocate**（trainer 整批切） | 所有精度组合均合法，正确性来自 `switch_to_*_mode` 的时序互斥（§3.7.1） | 同左。cast 产物直写 rollout 已注册显存 |

两种模式的正确性今天都由暂停保证，差别只在「谁决定何时暂停」（§3.7.2.4）。

##### 完整的支持矩阵：后端 × 放置 × 精度

把三个维度展开，逐格给出支持状态与所依赖的机制
（`M` 为 bf16 全量；「版本级等待」指 §3.4.2 约束 2）：

| 后端 | 放置 | train → rollout | 支持 | 正确性机制 | 备注 |
|---|---|---|---|---|---|
| Megatron | disagg | bf16 → bf16 | 是 | 暂停 +（可选）版本级等待 | 等待是优化（master params 隔离） |
| Megatron | disagg | bf16 → fp8/fp4 | 是 | cast 产物隔离 | **推荐默认** |
| Megatron | disagg | fp8 → fp8 | 是 | scale 转码，零额外字节 | 直供最优形态 |
| Megatron | disagg | fp8 → fp4 | 是 | cast，源须为 fp32 master | §3.4.2 约束 1 |
| Megatron | colocate | 全部五种 | 是 | **时序互斥** | fp8→fp8 下 `~0.5×M`，最省 |
| **FSDP2** | disagg | fp32 → bf16/fp8/fp4 | 是 | 发布副本（cast 产物）隔离 | 与 optimizer 冲突恒为零，副本在推送锁内刷新（§3.4.2） |
| **FSDP2** | disagg | **bf16 → bf16** | 是 | **版本级等待（必需）** | 唯一「等待非可选」的一格（§3.8.0a） |
| **FSDP2** | disagg | bf16 → fp8/fp4 | 是 | cast 产物隔离 | **FSDP2 大模型推荐** |
| **FSDP2** | colocate | 全部六种 | 是 | **时序互斥** | 峰值见 §3.7.3，fp32 下 `2.5~3×M` 偏高 |
| FSDP2 | 任意 | fp8-native → * | **否** | — | 上游缺失，非本设计限制 |

**所以回答「FSDP2 是否像 Megatron 一样得到支持」：是，且覆盖面相同。**
两者的差别不在「支持与否」而在三处**特性差异**，都已有对应机制：

| 差异 | Megatron | FSDP2 | 本设计的处理 |
|---|---|---|---|
| 参数 dtype 来源 | 固定 bf16（或 TE fp8） | **`model_dtype` 配置决定** | 9d-a：真实读取而非按后端硬分流 |
| optimizer master params | 有（`DistributedOptimizer`） | **无** | bf16 同精度下版本级等待升为必需（§3.8.0a） |
| DP 副本 | 恒有 | **纯 FSDP2 下无**（HSDP 有） | 9d-b：副本数运行时推导（§3.9.1） |
| fp8-native | TE `fp8_param` 支持 | **不支持** | 矩阵里标注，不做兼容层 |

**三处差异都不影响「能不能用」，只影响「用哪条机制」**——
这正是 holder 抽象与「发布缓冲」统一规则的价值：
数据面与 index 完全不感知后端（§3.9.1）。

**这张表读出的三条结论**：

1. **写屏障（§3.4.2.1）只在一格里用得上**：Megatron + disaggregate + 同精度，
   且在那一格里它也只是优化（§3.7.2.4）。
   **注意与「版本级等待」区分**：后者在 FSDP2 + bf16 同精度下是必需的（§3.8.0a），
   而 per-chunk 写屏障始终是优化。实现时不要把两者混为一个中间层。
2. **FSDP2 是两种部署模式下冲突语义都更简单的后端**，代价是恒定的 cast 开销与
   每 rank 一份 wire 格式的发布副本。这个权衡不是选择，是 fp32 参数的必然结果。
   它的难点不在冲突，而在 §3.9.2.1 的 shard 粒度。
3. **colocate 的正确性与精度完全解耦**，所以 colocate 下的后端差异
   只剩「谁来产出 cast 产物」，而这本来就是 converter 的职责。

#### 3.9.1 两个后端的分片结构

| 后端 | 分片维度 | 参数 dtype | PSRL 转换器 |
|---|---|---|---|
| **Megatron** | TP（行/列切分）+ EP / ETP（expert 切分）+ PP（层切分） | bf16 model params + fp32 master params（`DistributedOptimizer`） | `megatron_converter.py::MegatronConverter` |
| **FSDP2** | **per-parameter `DTensor`**，按 placement 切分（`Shard(dim=0)` 或 `Shard(dim=1)`）。HSDP 增加第一维 replicate | **fp32**，无 master 副本（§3.8.0） | `fsdp_converter.py::FSDPConverter` |

两个转换器都输出 `(state_dict, sharding_dict)` → `NIXLSharding`，
接入同一套 NIXL 传输和 index 基础设施。**数据面与 index 不感知后端**，
后端差异全部收敛在 converter 和下面四条上。

**「per-parameter DTensor」是 FSDP2 相对 FSDP1 的关键差异，也是它能支持直供的原因。**
FSDP1 的 `flat_param` 把一个 FSDP2 unit 内的多个参数打平进一块连续存储，
`named_parameters()` 返回的是 view，state dict 无法保证 in-place，
所以 `config.py:303` 拒绝它。FSDP2 的每个参数是独立的 `DTensor`，
`param.to_local()` 拿到的就是该参数持久的本地分片，
**可以直接注册给 NIXL 并被 RDMA 读**，与 Megatron 的参数地位相同。

**DP 冗余在两个后端下不等价，这推翻了 §3.5.7 的一条前提。**

| 配置 | mesh | 同一 `(key, shard)` 的副本数 | §3.5.7 的 ① 是否可用 |
|---|---|---|---|
| Megatron | TP×PP×EP×DP，`NIXLSharding` 只声明前三者 | `dp_size` | 可用 |
| **HSDP**（`fsdp_size < world_size`） | `(ddp, fsdp)` 二维，`fsdp_converter.py:325` 只用 `mesh_dim=1` | `world_size / fsdp_size` | 可用 |
| **纯 FSDP2**（`fsdp_size >= world_size`） | **一维 `(fsdp,)`**（`fsdp/utils.py:47`） | **1** | **不可用** |

`create_device_mesh` 在 `fsdp_size >= world_size` 时建一维 mesh，所有 rank 全切不复制。
此时每个 `(key, shard)` **全局只有一份**，单 rank 故障**直接落到 §3.5.7 的 ②**
（`CHUNK_UNAVAILABLE`、rollout 保持当前版本），没有 ① 这一层可以兜。

所以 §3.5.7 把 ① 称为「最常见情况」只在 Megatron 与 HSDP 下成立。
index 必须**从 `NIXLSharding` 的实际副本数推导降级能力**，
而不是假定 DP 副本一定存在：`replica_count == 1` 时 holder 摘除即意味着
该 chunk 立刻进入 `CHUNK_UNAVAILABLE`，这个判定必须是运行时查表的结果，
不能是写死的分支。纯 FSDP2 部署应当在启动期就把「无 DP 冗余」记入 index，
让 §3.5.7 的三层响应退化为两层，而不是在故障发生时才发现 ① 走不通。

**注意这与 §3.9.2.1 指向同一个配置旋钮**：`fsdp_size` 调小同时
恢复 DP 冗余并压回 shard 数放大。两个问题一个解法，这不是巧合——
它们都源于「`fsdp_size` 等于 world_size 时 mesh 退化成一维」。

#### 3.9.2 vLLM 要求的权重格式

vLLM 的 weight loader 期望 HF 格式的 canonical key（如 `model.layers.0.self_attn.q_proj.weight`），
以及 vLLM 自己的 fused 格式（如 `model.layers.0.self_attn.qkv_proj.weight`，三矩阵合并）。

**两个后端的格式转换路径**：

| 后端 | HF canonical → vLLM fused | 关键转换点 |
|---|---|---|
| Megatron | QKV 合并、MoE expert 维度声明 | `megatron_converter.py::slice_qkv_proj_megatron`、`get_sharding_for_param` |
| FSDP2 | QKV 3D reshape（`maybe_reshape_qkv_to_3d`）、MoE expert 拆分（`split_fused_moe_fsdp_param`） | `fsdp_converter.py::FSDPConverter.convert_state_and_sharding_dict` |

注意 FSDP2 目前的 MoE 处理是**把 fused MoE 参数拆成 per-expert key**
（`split_fused_moe_fsdp_param`），这与 §3.5.5 提出的「MoE 分组（保持 fused）」
方向相反。MoE 分组改造需要同步修改 `fsdp_converter.py` 的 split 逻辑，
使其也保持 fused 形态并改为声明多维 shard。**这是 §3.5.5 的一部分，不是额外工作。**

顺带会修掉一个今天就存在的限制：`split_fused_moe_fsdp_param` 断言
`len(shard_mesh) == 1 and len(shard_indices) == 1`，
而 HSDP 的 sharding 本身也是一维（replicate 维被排除），所以这条今天能过，
但**改成多维 expert shard 后这个断言必须一并放宽**，否则 HSDP + MoE 会直接抛错。

**FSDP2 的 MoE 分片有两种形态，MoE 分组必须覆盖两者。**
`split_fused_moe_fsdp_param` 今天按 `shard_dim` 分两条路：
`shard_dim == 0` 是切 expert 维（每 rank 持有连续的 expert 子集，
产出的 per-expert key 带 `NIXLSharding.default()`，即未分片），
`shard_dim == 1` 是切 projection 维（每 rank 持有**全部** expert 的一个 projection 切片）。
分组改造后前者天然映射为 expert 维的 shard 声明，
**后者才是真正需要两维 shard mesh `{0: 1, 1: fsdp_size}` 的情形**，
而这正是 §0.4a 所说「EP 与 TP 不对齐时需要确认 `find_finest_shard_mesh`
的 lcm 逻辑在两维上正确」的 FSDP2 版本。两条路都要测。

#### 3.9.2.1 FSDP2 的 shard 粒度是独立于 MoE 的第二个元数据膨胀源

§3.5.5 把元数据扇出归因于「MoE 按 expert 展开 key」，并提出 MoE 分组作为根治。
**但 FSDP2 有一个 MoE 分组治不了的放大源，必须单列。它不是 key 数，是每个 key 的 shard 数。**

FSDP2 的声明本身是干净的——`get_sharding_for_param`（`fsdp_converter.py:299`）
对 per-parameter DTensor 直接给出 `{shard_dim: mesh.size()}`，没有 FSDP1 那种
flat_param 的边界错配。**问题在于 `mesh.size()` 这个数本身有多大。**

`create_device_mesh`（`fsdp/utils.py:47`）在 `fsdp_size >= world_size` 时建一维 mesh，
于是 `mesh.size() == world_size`。对比 Megatron：

| 配置 | train 侧 `shard_mesh[0]` | rollout tp=8 | `find_finest_shard_mesh` 取 lcm |
|---|---|---|---|
| Megatron TP=8 | `tp_size` = **8** | 8 | **8** |
| HSDP，`fsdp_size`=8 | **8** | 8 | **8** |
| **纯 FSDP2，512 卡** | **512** | 8 | **512** |

`refactor_based_on_finer_shard_mesh`（`nixl_spec.py:196`）会把每个 client 的
`shard_indices` 展开到 finest mesh：rollout 侧原本 1 条索引的 key
变成 `512/8 = 64` 条。所以

> **纯 FSDP2 下每个 key 的 `desc_bytes_list` 与 comm plan 的 `shard_indices`
> 被放大 `world_size / tp_size` 倍，这个因子乘在 §3.5.5 的 key 数之上，
> 且与 MoE 完全无关**（attention 的 QKV、MLP 同样受影响）。

Megatron 不存在这个问题：TP / ETP 切分沿 canonical 维度且
`tp_size` / `etp_size` 是小整数（`megatron_converter.py:495`）。
**这是 FSDP2 在万卡场景下的真实可扩展性短板**，三条应对：

- **优先用 HSDP 而非纯 FSDP2**。把 `fsdp_size` 设到节点内（典型 8），
  `shard_mesh[0]` 就回到 8，放大因子归一。这同时带回 DP 副本（§3.9.1），
  一举解决两个问题。**这应当是大规模 FSDP2 部署的默认建议。**
- **把全局 finest mesh 各维大小作为握手期可观测量**，超阈值告警。
  今天这个数字完全不可见，坏配置只表现为握手慢。
- **shard 数的放大是 strided descriptor 的用例**（§3.6）：
  FSDP2 的 64 条索引在内存上是规则跨步的，一条 Nx5 strided desc 即可表达。
  所以 §3.5.5 的 strided 改造对 FSDP2 的收益比对 Megatron 更大。
  **已实测可用**（NIXL 1.5.0，descriptor 列表 16384:1 压缩；提交索引不压缩，§3.6.3），
  与 HSDP 是互补的两条缓解路径，不是二选一。

**另有一条今天未校验的 divisibility 约束。**
`get_local_sharded_tensors`（`nixl_spec.py:166`）要求
`tensor.shape[dim] % shard_mesh_at_dim == 0`，而 FSDP2 的 DTensor 本身
**允许 uneven sharding**（不整除时 pad 最后一个 shard）。
所以 `fsdp_size` 较大时，任何 dim0 不被 `mesh.size()` 整除的参数
会在传输期抛 `ValueError`，而不是在配置校验期。
`get_sharding_for_param` 应当在声明时就断言整除性并给出可操作的错误信息
（指向调小 `fsdp_size`），这与 §3.8.1 的 block 对齐约束是同一类处理。

#### 3.9.3 FSDP2 的 fp32 参数：既是约束也是优势

§3.8.0 已经确立 FSDP2 的 model params 是 fp32。这里补四条落地后果。

**(a) FSDP2 没有 master params，§3.4.1 的事实二不适用。**
Megatron `DistributedOptimizer` 下 optimizer 写的是 fp32 master shard，
所以冲突窗口窄到「param 回写那一瞬」。FSDP2 用
`build_optimizer(module.parameters(), ...)`（`optimizer.py:298`）
**直接作用在 sharded 参数上，没有第二份副本**，
所以 optimizer step 期间 rollout 读的那块内存就是 optimizer 正在改写的内存。

`model_dtype=fp32`（默认）下这不构成问题，**因为 cast 产物天然隔离**（rollout dtype 不会是 fp32），
发布缓冲是 cast 产物，与参数内存无关。
但它意味着 §3.4.2.1 的同精度放行逻辑在 FSDP2 下**不可达且不可用**：
若有人强行配置 FSDP2 + 同精度，写屏障必须覆盖整个 optimizer step，
这就是 §3.8.0 拒绝那个组合的原因。

**(b) fp32 → fp4 是 FSDP2 独有的优势。**
§3.4.2 约束 1 要求「FP4 的量化源必须是 fp32 参数，不是 fp8 model params」,
在 Megatron 下这需要 cast 路径能访问 optimizer 内部状态，是一处真实的耦合。
**FSDP2 下这个问题不存在**：model params 本身就是 fp32，
直接从它量化到 fp4 就是单次量化，不需要触碰 optimizer。

**(c) `param_offload` 必须在直供下禁用。**
§3.4.5 的注册有效性不变量是围绕 Megatron DDP buffer resize 设计的。
FSDP2 的等价触发源是 `param_offload=True` 时的
`offload_fsdp2_model_to_cpu`（`fsdp_utils.py:196`），它的实现就一行：

```python
model.to("cpu", non_blocking=True)
```

这会**重新分配每个 DTensor 的 local storage 到 host**，
注册过的 GPU 指针全部悬空。比 FSDP1 的 `flat_param_to` 更彻底——
FSDP1 至少保持 `_local_shard` 的身份，FSDP2 这里是整个 module 的设备迁移。

关键区别：这发生在**正常训练步之间**（`transformer_impl.py:893/904/912` 的
前后钩子），不是 colocate sleep 路径，所以现有的 `_sleep_fsdp_model` /
`nixl_sleep` 钩子盖不住它。两个选项：

- **禁用 `param_offload`**（建议）：直供模式下 trainer 的参数必须常驻且注册有效，
  这与 `param_offload` 的语义直接冲突。配置校验拒绝，理由明确。
- 或让可读状态位跟随 offload 翻转，并在每次 reload 后重新注册。
  重注册昂贵（§3.3 已说明），每步两次不可接受。

所以选第一条：**`param_offload` 与直供互斥**，和 `ps_mode` 一样作为被删除的配置面。
`optimizer_offload` 不受影响（它搬的是 optimizer 状态，不是注册过的参数）。

**同理要拒绝 `CPUOffloadPolicy`。** `transformer_impl.py:461` 在
`offload_policy or forward_only` 时给 `fully_shard` 传
`CPUOffloadPolicy(pin_memory=True)`，这让 FSDP2 **把参数常驻在 host**，
GPU 上只在 forward 期间有临时副本。直供下这等于没有可读的 GPU 源，
与 `is_rollout_mode_in_actor=True` 时 trainer 的 `meta` 注册是同一类问题（§3.3.1a），
但 `meta` 那条有 holder 对称化兜底，`CPUOffloadPolicy` 没有。
注意 verl 在设置它时会把 `_is_offload_param` 强制置 `False`（`:460`），
所以**不能只校验 `param_offload` 这一个 flag**，必须同时校验 `offload_policy`。

**(d) FSDP2 的 FP8 训练不在本设计范围内。**
Megatron 的 TE `fp8_param=True` 提供常驻 fp8 参数，这是 §3.8 中「FP8-native train」
形态的来源。FSDP2 侧没有等价物：TE 的 `fp8_autocast` 只影响 GEMM 的计算精度,
**参数仍以 fp32 存储**。所以 §3.8.0 矩阵里 FSDP2 行的 train dtype 只有 fp32 一个值,
这不是遗漏。若将来接入 torchao 的 `Float8Linear` 常驻 fp8 权重，
再按 Megatron 的 fp8 行扩展矩阵。

### 3.10 多角色：critic / ref / reward model，以及跨 stage 的弹性

**本节确认初稿的一处范围遗漏**：前面所有章节只描述了 **actor 一个模型**的权重传输。
PPO 还涉及 critic、ref、reward model，它们今天与本设计完全无关，
直供后也享受不到收益。本节给出扩展分析，并指出必须补的三处。

#### 3.10.1 今天的真实状态：三个角色都不走 NIXL

按 worker 类核实（`main_ppo.py:241-405`）：

| 角色 | worker 类 | 有 NIXL | 权重来源 | 是否随训练更新 |
|---|---|---|---|---|
| **Actor** | `PSRL_EngineTrainWorker` | **有**（继承 `PSRL_BaseTrainWorker`） | 本设计的全部机制 | **是**，每步 |
| **Critic** | `PSRL_CriticTrainWorker`（= verl `TrainingWorker`） | **无** | 自己 `init_model()` 读 checkpoint | **是**，每步（独立 optimizer） |
| **Ref** | verl `ActorRolloutRefWorker`（原生，非 PSRL 子类） | **无** | 自己 `init_model()` 读 checkpoint | **否**，冻结。§3.10.2 建议改为 actor 的 CPU 备份 tag |
| **Reward Model** | `PSRL_ServerAdapter`（**与 Rollout 同一个类**） | **无** | `replica.init_model(wg)` 自己加载 | **否**，冻结。基座与 actor 不同，必须保留独立路径 |

关键证据：critic / ref / RM 的构造都不经过 `PSRL_BaseTrainWorker.__init__`，
所以 `nixl_storage_client` 为 `None`。RM 的 loader 显式标注了这件事
（`reward_model/manager.py:79`）：

```python
ps_manager_handle=None,  # reward model: no PS sync
```

所以这三个角色今天靠**各自读盘一次**工作，与权重传输链路无关，
也**不受直供影响**。这解释了为什么初稿漏掉它们而没有暴露问题——
但它们也不是免费的。

#### 3.10.2 冻结角色（ref / RM）：ref 不该进 holder 表，它是 actor 的一个备份 tag

**ref 的权重在整个训练过程中永不更新。** `_compute_ref_log_prob`
（`ray_trainer.py:3257`）只做前向、`compute_loss=False`，从不接受 optimizer step。
RM 同理（它是评估器，不是被训练对象）。

初稿的方案是「ref 冻结在 v0，可与 actor 共享 v0 的 holder」。
**核实 miles 后发现有更简单的形态：ref 根本不需要是一个 holder。**

##### miles 的做法：ref 是 actor 权重的一份 CPU 备份，不是独立模型

miles **没有独立的 ref 进程**。它的 ref 是 `TensorBackuper` 里的一个 tag
（`tensor_backper.py:53`）：

```python
def backup(self, tag: str) -> None:            # tag = "ref"
    for name, param in self._source_getter():
        backup_dict[name] = torch.empty_like(param, device="cpu", pin_memory=True)
        backup_dict[name].copy_(param.detach(), non_blocking=True)
```

算 ref logprob 时 `restore("ref")` 换进 GPU，算完 `restore("actor")` 换回来。
`_TensorBackuperMainCast` 更进一步——**actor 那份根本不备份，
靠重放 optimizer 的 cast + all-gather 重建**，docstring 明说是 bit-identical：

> Rebuilds the actor weights instead of keeping a pinned CPU copy. Restore replays
> the step end's cast + all-gather, so it is bit-identical. **Non-actor tags
> (ref/teacher) have no master weights to rebuild from, so they keep full pinned copies.**

##### PSRL 采纳这条，因为它删掉一条约束而不是增加一条

**ref 作为「actor 的一个 CPU 备份 tag」而非 holder**，收益有三：

1. **彻底不进 holder 表**，于是不需要「跨角色共享 v0」这套机制，
   也**不需要初稿那条精度匹配判定**（原文：「ref 与 actor 只在同精度时才能共享，
   若 rollout 用 fp8 而 ref 需要 bf16 则不是同一份字节」）——
   备份是从 **trainer 的 bf16 参数**拷的，与 rollout 的 fp8 无关，**精度问题自动消失**。
2. **PSRL 已有同构先例**：`ray_trainer.py:303` 的 `ref_in_actor` 在 LoRA 场景下
   就是「ref = 不加 adapter 的 actor」。把它推广到非 LoRA 场景即可，
   不是新机制而是现有机制的扩展。
3. **顺带支持 `ref_update_interval`**（miles `actor.py:795`）：
   周期性把 ref 重新 backup 成当前 actor，在长训练中避免 KL 约束过度收紧。
   这是算法侧的真实需求，而 holder 方案要支持它反而更麻烦
   （要给「冻结」角色重新发版本，与 v0 永不过期的设定矛盾）。

**代价是 `M` 的 CPU pinned 内存**（按约束 (b) 的可操作表述，这是随模型增长的，
属 §0 表格里的例外项，需显式记账）。但它换掉的是「ref 独立进程 + 独立读盘
+ 跨角色精度判定」三件事，**净复杂度是降低的**。

##### RM 不同：它是独立模型，必须保留独立路径

RM 的基座与 actor **不同**（通常是专门训练的 reward model，架构可能都不一样），
所以它**不能**走 backup tag 这条路。它的形态是：

- **独立的 vLLM server**（PSRL 今天就是 `PSRL_ServerAdapter`，与 Rollout 同类）
- **自己读盘一次，之后永不更新**
- **可以进 holder 表**（作为 `(role=rm, v0)` 的永久 holder），
  也可以完全不进——因为没有任何人需要从它拉取权重。**建议不进**，
  它只是一个消费者而非生产者。

所以准确的分类不是「冻结 vs 滚动」，而是**三类**：

| 角色 | 形态 | 进 holder 表 |
|---|---|---|
| actor | 滚动版本流 | **是**，核心 |
| critic | 独立滚动版本流 | **是**（§3.10.3） |
| **ref** | **actor 的 CPU 备份 tag** | **否** |
| **RM** | **独立模型，自己读盘** | **否** |

**holder 表只服务「需要被别人拉取」的角色**，这把 §3.10.3 的 role 维度
从四值缩到两值（actor / critic），实现负担进一步下降。

#### 3.10.3 Critic 是第二个独立版本流，holder 表必须加 role 维度

critic 每步都在更新且有自己的 optimizer，所以它是**与 actor 互不相关的第二个版本流**：

```
actor:   v0 → v1 → v2 → ...
critic:  c0 → c1 → c2 → ...
```

两条流的版本号**恰好同步递增**（同一个训练步里各自更新一次），但内容独立。
这对现有设计是三处冲击，**必须改，否则多角色下 holder 表会互相覆盖**：

1. **holder 表加 role 维度**。§3.1 现在写的是 `(version, key, shard) -> holder`，
   应为 `(role, version, key, shard) -> holder`。这是最小改动，
   但不做的话 actor 的 v3 与 critic 的 c3 会冲突。
2. **「当前版本号」不是标量而是每角色一个**。§3.2 的状态表现在只有一个 `v`，
   需要改为 `{role: v}`，且 §3.4.2 的版本连续性与 `maybe_delete_buffer(v-1)` 的
   索引语义要按角色各自成立。
3. **colocate 下的显存账要按角色数展开**。§3.7.3 的表只算了 actor 一侧，
   actor + critic 共存时训练显存翻倍（critic 通常是同尺寸模型，
   且它**没有 rollout 侧**，所以是纯增量）。

**一个缓解因素**：critic 不参与 rollout，所以它**不需要 peer 扩散、
不需要 W 限流、不需要 §3.5 的三阶段规划**——它只有「被自己的训练步读」这一个用途。
它的传输需求远小于 actor，可以先用最朴素的一条路径落地。

#### 3.10.4 跨 stage 资源弹性：约束 (b) 与它直接冲突，这是一个真实取舍

**本小节修正初稿的一处错误判断。** 初稿写「弹性 = 时间片调度，不是 GPU 数量调度」，
依据是四个角色的 pool 声明（`main_ppo.py:245/255/265`）：

```python
self.mapping[PSRL_Role.Actor]      = ["train_pool"]
self.mapping[PSRL_Role.Critic]     = ["train_pool"]
self.mapping[PSRL_Role.RefPolicy]  = ["train_pool"]
```

**但这是 PSRL 今天的配置，不是架构约束。** 把当前实现当成约束，
就排除了「GPU 数量在 stage 间迁移」这条真正有收益的路径。
PivotRL（`third_party/pivotrl.pdf`）在 128 GPU 上实测该路径
相对 colocated 提升 **1.81–2.05×**、相对 disaggregated **1.50–1.95×**，
所以它不是理论可能性而是主要收益来源。

##### (a) 为什么约束 (b) 与完整的资源迁移互斥

PivotRL 的核心基础设施 **Global Parameter Storage (GPS)**
**就是一份 CPU 侧的全模型副本**——正是约束 (b) 禁止的东西。
它给出的理由恰好命中本设计的死角（§5.1 原文转述）：

> 最直接的做法是从 peer GPU 直传参数。但这**预设目标参数已经在集群某处**，
> 而一旦某个 role 被完全 shift down、任何 GPU 上都不再有副本，这个假设就不成立。
> 另一个选择是从磁盘重载，可用性有保证但 I/O 开销过高。

**这正是 §3.5.7 三层降级的盲区**：那里的「所有 holder 都没了 → `CHUNK_UNAVAILABLE`
→ rollout 保持旧版本」在**故障**场景下合理，
但在**主动把某个 role 缩容到 0** 时是致命的——
GenRM 被完全 shift down 后权重在集群里一份不剩，想恢复只能读盘。

所以诚实的表述是**两档形态，而不是「本设计也能做弹性」**：

| | **档位 A：约束 (b) 下（本设计当前）** | **档位 B：放开约束 (b)（PivotRL 形态）** |
|---|---|---|
| CPU 副本 | **0** | `num_models × M`（GPS） |
| role 内弹性（实例增减） | **支持**（§3.3.1c） | 支持 |
| **role 缩容到 0 再恢复** | **不支持**，权重会消失 | **支持**，这是 GPS 存在的理由 |
| GenRM 场景的要求 | RM 必须**常驻至少一个副本** | 无要求 |
| 收益 | 省 CPU 内存 | 1.5–2× 端到端吞吐 |

**档位 B 不需要推翻本设计。** `STORE_CPU` 这类 holder 在 §3.1 的初版枚举里就有，
是我按约束 (b) 删掉的（「holder 种类从四种缩到三种」）。
**把它加回来是一行枚举的事，数据面完全不用改**——
GPS 在 holder 抽象里就是「一类链路成本较高、但永不失效的 holder」。
这反过来验证了 holder 抽象的正确性：它预留了这个位置。

**但「永不失效」这个描述掩盖了一个真实缺口：CPU holder 的内容会陈旧。**

holder 表只记录「谁持有 (role, version, key, shard)」，
而 §3.5.3 的选源按**链路成本**排序——CPU holder 在同节点时成本很低，
会被优先选中。于是：

```
trainer 发布 v+1
  -> GPU holder 更新为 v+1
  -> CPU holder（GPS）仍停留在 v          ← 没人更新它
  -> rollout 请求 v+1，选源时 CPU holder 因成本低排在前面
  -> 若只按「链路成本」选，会选中一个不持有 v+1 的 holder
```

**这不是实现细节，是档位 B 的正确性缺口。** 三点修复，按侵入性排序：

**(1) 版本相等性判定已经覆盖它，前提是 CPU holder 也走同一张表。**
§3.5.2 的硬约束是「只有 `version(peer) == v` 的 holder 才可作源」，
而 holder 位图在版本推进时**整体清空**。
**只要 `STORE_CPU` 不被豁免这条规则**，陈旧的 CPU holder 在 v+1 发布那一刻
就自动退出候选集，不会被选中。**所以缺口的根源不是机制缺失，
而是「永不失效」这个表述诱导实现者给 CPU holder 开后门。**

修正表述：CPU holder **不是「永不失效」，而是「物理介质不会消失」**——
它的版本有效性与 GPU holder 完全同等，一样随版本推进失效。

**(2) GPS 的更新必须是版本发布流程的一部分，不是后台任务。**
对应 PivotRL 的 `Write` 操作（§5.1 的图 5b：「After Each RL Step,
training GPUs update GPS with the latest policy model parameters」）。
在本设计里它是发布序列的一步：

```
optimizer step 完成
  -> （FSDP2 bf16 同精度时）等待 v 的读者清零
  -> 刷新布局转换产物（§3.7.2.3）
  -> 【档位 B】Write 到 GPS，完成后 CPU holder 注册为 v+1
  -> 发布 v+1
```

**Write 必须在发布之前完成**，否则 v+1 发布后会有一段时间
CPU holder 既不是 v 也不是 v+1。PivotRL 提到可以
「overlap Write with training in the subsequent RL step」来隐藏开销——
但那要求 Write 的目标是 `v+1` 而发布的是 `v`，**即 GPS 永远落后一个版本**。
这与版本相等性冲突，**本设计不采纳这条优化**，宁可让 Write 在关键路径上。

**(3) 若 Write 的开销不可接受，正确的降级是「GPS 只服务 bootstrap 与 shift-up」。**
即它不参与稳态的选源，只在「某个 role 从 0 恢复」时被读取。
此时它持有的是**最近一次完整写入的版本**，shift-up 的 role 拉到后
再走一次常规同步追到最新。**这把一致性问题转化为一个 staleness 问题**，
而后者有现成的处理（§3.5.2 的版本相等性 + 重新解析）。

**建议采纳 (1) + (3)**：不给 CPU holder 开版本豁免，
且把它的角色限定为「bootstrap 与 shift-up 的源」而非稳态选源候选。
这样 Write 的频率从「每步」降到「每次 role 归零前」，开销可忽略。

##### (b) PivotRL 的角色上下文分类：本设计完全没覆盖的维度

本文前面所有章节只讨论**权重**，而 shift 要搬的远不止权重。
PivotRL §5.2 的分类应当直接采纳：

| 类别 | 内容 | 动作 | 本设计是否已覆盖 |
|---|---|---|---|
| Stateless-临时 | KV cache、activation、临时 buffer | **discard**，重算 | 是（§3.7.3 的 `discard(("kv_cache",))`） |
| Stateless-参数 | 模型权重 | **discard**，从 GPS `Read` | **仅档位 B**；档位 A 下权重不是 stateless |
| Stateful-可卸载 | optimizer state、梯度、**CUDA Graph** | **offload 到 DRAM**，按需 reload | 部分（§3.7.3.1 的 `optimizer_offload`），**CUDA Graph 未覆盖** |
| Stateful-须保留 | CUDA context、**NCCL 通信组**、描述符 | **留在 GPU**（实测 <3.37 GB/GPU） | **未覆盖** |

两条关键推论：

1. **「模型参数是 stateless 的」只在有 GPS 时成立**，这是档位 A 与 B 的分水岭。
   档位 A 下权重是 stateful 的（唯一副本），所以不能 discard。
2. **「NCCL 通信组必须保留在 GPU」是秒级切换的前提**，重建通信组很贵。
   **这一条 PSRL 天然满足，核实后确认不是缺口**：

   ```
   grep -rn "destroy_process_group|destroy_model_parallel" --include=*.py psrl/
   → 零命中
   ```

   `init_process_group` 在 worker 构造时调一次
   （`engine_train_worker.py:78` 的 `initialize_global_process_group_ray`），
   之后**从不销毁**。`switch_to_*_mode` 只做 NIXL dereg 与显存 resize，
   不碰通信组。所以 PivotRL 归类为「须保留在 GPU」的那部分，
   在 PSRL 里本来就保留着。

   **但这是一个需要防回归的隐式不变量**：将来若有人为省显存去销毁通信组，
   切换成本会从亚秒级涨到秒级甚至更高，且不会有任何报错。
   **应当在角色切换路径上加一条断言**：
   `assert torch.distributed.is_initialized()`，
   并在注释里写明「通信组必须跨 stage 存活，重建代价远超它占的显存」。

##### (c) shift 粒度：PivotRL 的工程限制与本设计的 §3.9.2.1 相互印证

PivotRL 限制 shift 粒度（§5.2 末段）：

- **rollout / reward 按 replica 粒度**，即沿 DP 维扩缩，**不改模型切分**，
  所以只需重调度 sequence，不碰 shard mesh。
- **training 整体搬**，避免重分 optimizer state 与重建通信组。

**这与 §3.9.2.1 的结论一致**：改并行度会改 `shard_mesh`，
而 finest mesh 是全局 lcm，一处变动牵动所有 client 的 descriptor。
所以「沿 DP 维扩缩」是唯一不触碰传输层的弹性维度——
**本设计的 §3.3.1c（弹性伸缩）应当显式写明这条限制**，
而不是泛泛地说「instance 集合变化」。

##### (d) shift 决策：收益必须覆盖切换成本

PivotRL 的 planner 把 shift 决策形式化为带 penalty 的优化（§7.3）：

```
B = H · (T_candidate − T_current) − φ
```

`H` 是收益预期持续的时间窗（默认 60s），`φ` 是切换代价，
候选集用 DP 在 shift pair 上搜最小 penalty。
实测 shift up ≤ 8.5s、shift down ≤ 6s，sequence migration ≤ 1.18s。

**这个框架比本设计的 W 自适应反馈（§3.5.2.1）严谨一个量级**：
W 只有「带宽劣化就下调」这个单向反馈，没有「这次调整值不值」的判据。
若将来做档位 B，**planner 应当照搬这个 benefit − penalty 结构**，
而不是沿用 W 的朴素反馈。

##### (e) 切换粒度：学 SkyRL 的按操作懒加载，不做 stage 边界统一切

初稿写的是「每个 stage 唤醒需要的角色、睡下其余」，即**全局 stage 划分**。
**核实 SkyRL 后发现按操作懒加载更优**（`worker_dispatch.py:161`）：

```python
def _ensure_on_gpu(self, model, need_optimizer=True, need_model=True):
    group = self._get_colocation_group(model)
    for other in group:                       # 先把同组其他模型踢出去
        if other != model and (state.model_on_gpu or state.optimizer_on_gpu):
            self._offload_inactive_model(other)
    backload_model     = need_model     and not state.model_on_gpu       # 只补缺失的
    backload_optimizer = need_optimizer and not state.optimizer_on_gpu
```

**两个关键设计要点，初稿都没有：**

**(1) `model_on_gpu` 与 `optimizer_on_gpu` 是二维独立状态。**
SkyRL 按操作声明需求，不同操作需求不同：

| 操作 | `need_model` | `need_optimizer` | 出处 |
|---|---|---|---|
| forward / logprob | 是 | **否** | `:269`、`:313` |
| train step | 是 | 是 | `:383`、`:425` |
| optimizer step only | **否** | 是 | `:469` |

注释原文：「model weights may remain resident while the optimizer is offloaded」。
**ref logprob 只需 model 不需 optimizer**——这正是省搬运的关键，
而初稿的「唤醒/睡下」是一维开关，做不到这个粒度。

**(2) 懒加载而非预先切换**：不在 stage 边界统一搬，而是每次调用前检查、只补缺的。
好处是**连续两个同角色操作之间零搬运**（例如 critic 的 forward 后紧接 train step，
model 已在 GPU 上，只需补 optimizer）。

**PSRL 采纳这条，形态是**：

```
_ensure_on_gpu(role, need_model, need_optimizer):
    for other in colocation_group(role):
        if other != role and other 占着 GPU:
            撤销 other 的 holder 资格（若它是 holder）   ← §3.4.5 协议
            offload other 的 model / optimizer
    只补 role 缺失的部分（model 和 / 或 optimizer）
```

与 §3.4.5 的衔接很自然：**holder 撤销只对「进 holder 表的角色」需要**
（按 §3.10.2 只有 actor 与 critic），ref / RM 的换入换出不涉及 holder 协议。

**三档 colocation 配置也应当照搬 SkyRL**（`config.py:666`，它的默认值就是全 colocate）：

| 档 | 含义 | 对应 SkyRL |
|---|---|---|
| 全 colocate | 所有角色共卡，互斥换入 | `colocate_all=True`（默认） |
| 部分 colocate | actor + ref 共卡，critic 独立 | `colocate_policy_ref=True` |
| 全独立 | 各占各的 pool | 两者皆 False |

**`colocation_group` 这个概念是必须的**，因为互斥范围由它定义而非全局。
初稿假定「所有角色共享 train_pool」就是隐含地只支持第一档（§3.10.4 开头已更正这个错误）。

##### (f) 档位 A 下仍然可做的事

即使保持约束 (b)，下面三条仍然成立且值得做：

- **RM 的实例数弹性已经可用**：`elastic_rm_mode`（`ray_trainer.py:1488`）
  用 `SubRayResourcePool` 在共享 pool 上划弹性子池，RM 按需增减副本。
  **前提是至少留一个副本**（否则权重消失），这正是档位 A 的边界。
- **按操作懒加载的角色换入换出**（上文 (e)），复用 §3.4.5 的 holder 注销协议。
  频率远低于训练步，重量级成本可接受。
- **多角色的显存交接比单角色的读写冲突简单**：交接点是操作边界
  （粗粒度、低频、天然同步），不是 §3.4 那种训练步内的窄窗口。
  **所以多角色场景下 §3.4.2.1 的写屏障更不必要。**

#### 3.10.5 本节新增的待办

| 项 | 性质 | 档位 | 说明 |
|---|---|---|---|
| **holder 表加 role 维度** | **正确性** | A+B | 不做则 actor 与 critic 的版本互相覆盖（§3.10.3）。按 §3.10.2 只需两值（actor / critic） |
| **ref 实现为 actor 的 CPU 备份 tag，不进 holder 表** | **简化** | A+B | 照 miles 的 `TensorBackuper`，顺带删掉「跨角色精度匹配判定」这条约束，并支持 `ref_update_interval`（§3.10.2） |
| **RM 不进 holder 表** | 简化 | A+B | 它是消费者不是生产者，没人需要从它拉权重（§3.10.2） |
| **`_ensure_on_gpu(role, need_model, need_optimizer)` 二维懒加载** | 性能 | A+B | 照 SkyRL `worker_dispatch.py:161`。ref logprob 只需 model 不需 optimizer，一维开关做不到（§3.10.4e） |
| **三档 colocation 配置 + `colocation_group`** | 架构 | A+B | 互斥范围由 group 定义而非全局；初稿隐含只支持「全 colocate」一档（§3.10.4e） |
| **弹性只沿 DP 维，显式写入 §3.3.1c** | 正确性 | A+B | 改并行度会动 finest mesh（§3.9.2.1），PivotRL 同样限制在 replica 粒度（§3.10.4c） |
| **NCCL 通信组的防回归断言** | 防回归 | A+B | **不是缺口**——PSRL 从不销毁进程组（grep 零命中），天然满足 PivotRL 的「须保留在 GPU」。但需加断言防将来有人为省显存而销毁（§3.10.4b） |
| **CUDA Graph 纳入可卸载上下文** | 性能 | A+B | 与 `optimizer_offload` 同类，§3.7.3.1 只覆盖了后者（§3.10.4b） |
| **colocate 显存账按角色展开** | 容量规划 | A+B | actor + critic 共存时的真实上限（§3.10.3）。ref 按 §3.10.2 只占 CPU，不进 GPU 账 |
| **ref 备份的 CPU 内存计入约束 (b) 例外** | 记账 | A+B | `M` 的 pinned 内存，随模型增长，属 §0 表格的显式例外项（§3.10.2） |
| **`switch_to_*_mode` 收敛为 `_ensure_on_gpu`** | 弹性 | A | 不是「推广到五角色」而是**换成按操作懒加载**，现有的 sleep/wake 成为它的实现细节（§3.10.4e） |
| **`STORE_CPU` holder（GPS 形态）** | **架构选项** | **B** | 放开约束 (b) 换 role 归零恢复能力与 1.5–2× 吞吐。数据面不改，只加一种 holder（§3.10.4a） |
| **benefit − penalty 的 shift planner** | 弹性 | B | `B = H·ΔT − φ`，比 W 的单向反馈严谨（§3.10.4d） |

**前两处正确性项应当并入落地项 8**（版本可见性），而不是新开一项——
它们是同一张 holder 表的字段扩展，分开实现会导致返工。

---

### 3.11 LoRA：不在本设计覆盖范围内，但必须显式拒绝而不是静默出错

**这是第六轮审核新发现的一格。** 前面所有章节都假定「训练的参数集合
== 要同步给 rollout 的参数集合」，而 LoRA 让这个等式不成立。

**PSRL 今天的 LoRA 状态是「半接入」**，这比完全不支持更危险：

| 位置 | 状态 |
|---|---|
| 配置项 | **有**（`model.py:60` 的 `lora_rank` / `lora_alpha` / `target_modules`） |
| 配置校验 | **有**（`config.py:234-243` 校验 vLLM 的 `max_lora_rank`） |
| rollout 侧引擎 | **有**（`vllm_async_server.py:428` 开 `enable_lora`） |
| ref 复用 | **有**（`ray_trainer.py:303` 的 `ref_in_actor`，LoRA 下 ref = 不加 adapter 的 actor） |
| **权重同步路径** | **无**。`convert_fsdp_inplace` / `convert_megatron_inplace` 都不认 LoRA |

**失败形态是静默的。** `fsdp_converter` 走 `model.state_dict()`，
`vllm_converter` 走 `module.named_parameters(recurse=False)`。
LoRA 下这两者给出的是：

- **train 侧**：PEFT 包装后的 `base_model.model.*.base_layer.weight`（冻结的基座）
  加上 `lora_A` / `lora_B`。按 canonical 命名对齐时
  **基座能对上、adapter 对不上**（vLLM 侧的 adapter 不在 `named_parameters` 里，
  见下），于是 adapter 被静默丢弃。
- **rollout 侧**：vLLM 的 adapter 存在 `lora_a_stacked` / `lora_b_stacked`
  （`vllm/lora/layers/base_linear.py:128`），它们是**普通 tensor 而非
  `nn.Parameter`，也没有 `register_buffer`**，所以 `named_parameters`
  和 `named_buffers` 都看不见它们。NIXL 根本没有可注册的落点。

**结果**：rollout 拿到一个权重正确但**不含任何 adapter** 的基座模型，
而训练侧以为同步成功。表现为 reward 完全不涨，且没有任何报错。

**verl 的做法说明这不是加个分支能解决的**（`transformer_impl.py:995-1085`）：
它为 LoRA 准备了**两条完全独立的路径**——
`merge_lora=True` 时在 `merged_lora_context` 里物化合并后的全量权重
（注意注释：「Materializing after exit silently sends base weights without adapters」，
正是上面那个静默失败），`merge_lora=False` 时走 `collect_lora_params`
+ vLLM 的 `add_lora`，而后者**要求一次性拿到完整的 adapter dict**
（`bucketed_weight_transfer.py:277` 的注释），
与本设计的 chunk 流式传输**语义上直接冲突**。

**所以本设计的处理是：**

1. **`merge_lora=True` 可以无改动支持**——合并后参数集合与全量训练一致，
   canonical 命名空间与 shard 代数都适用。这一格应当测试覆盖。
2. **`merge_lora=False`（真正的 adapter 传输）不在本期范围内**，
   原因是它需要一条绕过 chunk 流水的 whole-adapter 路径，
   与 §3.6 的执行层模型不兼容。
3. **配置校验必须显式拒绝 2，并给出 1 作为替代**，
   这是本节唯一的落地项（并入 9d-a，与其余配置期防御同批）：

```
if lora_rank > 0 and not lora.merge:
    raise ValueError(
        "LoRA adapter sync is not supported by the NIXL weight transport. "
        "Set model.lora.merge=true to sync merged weights, or disable LoRA."
    )
```

**为什么必须拒绝而不是留白**：LoRA 的配置项今天已经存在且校验通过，
用户完全可以开着它跑起来，然后得到一个不涨的 reward 曲线。
**「不支持」与「支持但静默错」的区别就在这个 assert 上。**

## 4. 端到端数据流

```
train step v 完成
  参数 dtype 由后端决定（§3.9）：Megatron bf16 或 TE fp8 / FSDP2 fp32
  │
  ├─ 向 WeightIndex 登记：v 可读，holder = 本 rank（附 PP/EP 归属与副本数）
  │     副本数由 NIXLSharding 推导，纯 FSDP2 下为 1（决定 §3.5.7 的降级层数）
  │
  ├─ colocate？──是──▶ rollout pause_generation 完成后进入 transfer（§3.7.2）
  │                      cast 产物或直接 memcpy 写入 rollout 已注册参数显存
  │                      切换 = rollout release_kv_cache_memory / wake_up(["kv_cache"])
  │                             train 释放/恢复 optimizer state
  │                      全程无 NIXL（CUDA IPC + 本地 copy，§3.7.3），不经 host
  │                      正确性来自时序互斥，与 dtype 组合和后端均无关
  │
  └─ 否（disaggregate）──▶ rollout instance 自行发起 pull（无全局 barrier）
             │              各 instance 自行择时 pause_for_sync（错开，非全局 barrier）
             │
             │  向 index 声明 (v, chunks) 并取得租约
             │  index 返回预编译 plan + holder，按链路成本排序：
             │     同节点 peer rollout（已持有 v）> 跨节点 peer > trainer rank
             │
             ├─[①源头准备] 两条路径，由「源头是否产生新字节」分流（§3.4.2）
             │   cast 路径（FSDP2 全部 / Megatron 的 fp8-fp4 rollout）
             │     -> cast 或 scale 转码写入 wire buffer，瞬态 O(chunk)，深度 2 流水
             │     -> 发布缓冲与参数内存分离，冲突归零，无需写屏障
             │   同精度路径（仅 Megatron bf16+bf16 / fp8+fp8）
             │     -> 直读参数，写屏障在 version 发布前一次性等待读者清零
             │   两个后端均无需 gather：Megatron 靠 shard 代数，
             │   FSDP2 靠 to_local() 的持久本地分片（§3.3）
             │
             ├─[②三阶段执行] receive -> external-exchange -> internal-exchange
             │   prepped descriptor 按 index 组合，round-based 双 buffer
             │   atomic group 为提交与等待单位
             │
             ├─[③落地格式化] vLLM fp8_utils 运行时格式化（并行度匹配时 no-op，FP4 下必做）
             │   group 完整到达才标记版本可见
             │
             └─ 释放租约
                 │
                 ▼
            Megatron 同精度路径：trainer 在发布 v+1 前等待 v 的读者清零（通常已清零）
            超时则撤销租约并要求重拉，trainer 不被无限阻塞
            FSDP2 bf16 同精度：同上，但此处是正确性必需而非优化（§3.8.0a）
            cast 路径：无等待。发布副本在推送锁内刷新，读者在锁外读（§3.4.2）
```

---

## 5. 优化项清单

| # | 优化 | 收益 | 独立落地 |
|---|---|---|---|
| **0** | **NIXL 新 API 的 micro-benchmark + 升级到 1.5.0** | **已完成**（`tests/nixl/bench_nixl_prepped_xfer.py`）。确认收益成立（5~30×）但有两个前提，见 §3.6.3。Dockerfile 与 install 脚本均 pin **NIXL v1.5.2 + UCX v1.23.x**，否则 Nx5 strided 不可用 | 已做 |
| 1 | `prep_xfer_dlist` / `make_prepped_xfer` 替代 per-key `initialize_xfer` | **实测 5~30× 加速且随 descriptor 数放大**（16384 条下 0.014ms vs 0.424ms）。**索引必须用 int32 ndarray**，传 Python list 会慢 15× 并抵消全部收益（§3.6.3） | 是 |
| 2 | colocate 走 CUDA IPC + stage 边界 offload（`discard(("kv_cache",))`） | 切换成本从「全量 PCIe 往返 + 重握手」降到「权重 D2H/H2D + KV cache 重分配」，**且 colocate 路径完全不碰 NIXL 注册**（§3.7.3），省掉大 MoE 下 14 万次 `register_memory` | 是 |
| 3 | Nx5 strided 描述符 | 删掉非连续 shard 的中转缓冲（**GPU 上，不是 CPU**——见 §3.6 的更正）与槽位串行等待。**槽位串行等待是主要收益**，显存次之 | 是 |
| 4 | 源头 FP8 wire | 传输字节减半，消掉 train/gen 双缓冲，且量化产物天然就是发布缓冲 | 是 |
| 5 | 上游分波与 peer 读重叠 | follower 读第 `w` 波时 designate 读第 `w+1` 波，peer 阶段藏到上游后面。cast 不在拉取路径上，发生在发布点（§3.4.2） | 是 |
| **5b** | **按 key 合并连续 shard 的传输** | `client_read`/`client_write` 对同一 (op, target, key) 的全部连续 shard 只提交一个 NIXL 请求（`merge_contiguous_xfer`），替代旧的 cached-xfer 队列；handle 在完成后立即释放，prepared dlist 可复用并在重注册时失效（§3.7.7）。提交开销从 O(#shards) 降到 O(#keys) | 是 |
| 6 | **canonical MoE 分组**（key 数 14万 → 数百，§3.5.5） | 元数据从 56MB/client 降到几百 KB，**大 MoE 与 1TB 的前置阻塞项** | 是 |
| 6b | **控制面下热路径**（plan 本地缓存 + 租约按 instance 聚合 + gRPC） | 消除 index 单点，**这是 7/8/9 的前置** | 是 |
| 6c | `make_xfer_tag` 去 pickle、**chunk 按 `T_fmt/T_rdma` 自适应**（§3.5.6，注意方向与初稿相反）、轮级决策本地化 | 纯 CPU 与 RTT 开销；**本地化是「chunk 可调小」的前提**，而调小是让格式化被流水掩盖的手段 | 是 |
| 6d | chunk / commit group 概念分离（§3.6.1） | 解开传输粒度与原子性的耦合，是 6 的前置 | 是 |
| 7b | **版本发布挂 `on_train_step_end` 回调**（§3.5.2.2） | 发布点落在「权重刚就位且 NIC 刚空闲」的时刻，同时减 staleness 与增带宽。一个回调，非上报协议 | 是 |
| 7c | chunk 起始偏移错开 | 零成本，让扩散在第一波内部发生 | 否，依赖 7 |
| 8b | trainer 单 rank 故障：DP 副本接管 + `CHUNK_UNAVAILABLE` | 万卡下的可用性，复用已有 quarantine 路径 | 否，依赖 8 |
| **8d** | **让 `CHUNK_UNAVAILABLE` 真能「保持当前版本继续生成」**：`pull_model_for_sync` 的版本 assert 改返回值（`vllm_async_server.py:1036`）+ 版本发布从乐观预发改为 pull 成功后发布（`sync_and_migrate_mixin.py:167`） | **8b 的前置**：不改则降级路径会撞 assert → `fail_sync` → replica 被摘掉，与「继续生成」相反。乐观预发还留下「SMG 认为是 v+1、实际是 v」的秒级窗口，**直供下这条从异常路径变成常态路径**（§3.5.7 末段） | 否，**与 8b 同批** |
| 7 | 三阶段去冗余规划 + 同版本同节点 peer 优先 + 自适应 W | 消除直供带来的单源瓶颈 | 否，需新 planner |
| 8 | 版本可见性 + 三条运行时断言（§3.4.6）+ **holder 表的 role 维度**（§3.10.3） | 无副本下保护 trainer 权重。**注意「租约」已收敛为 per-chunk 计数器，不是版本×chunk 租约表**。role 维度是 PPO 多角色的正确性前提，与 holder 表同批实现 | 否，需与 staleness 合并 |
| 8c | **版本级等待 + 可选的 per-chunk 写屏障**（§3.4.2.1） | **版本级等待对 FSDP2 bf16+bf16 是必需**（整个 optimizer step 在写参数，§3.8.0a）；对 Megatron 同精度只是把 pause 窗口收窄的优化 | 否，依赖 8。**启用 FSDP2 同精度前必须完成** |
| 9 | trainer 直供，删除 PS worker，新建 bootstrap | 额外 CPU 存储 `N×M → 0`，train 出口 `N×M → M` | 否，需 6+7+8 |
| 9b | **holder 语义对称化 + bootstrap 走 rollout 代表读盘** | `is_rollout_mode_in_actor=True` 的**正确性前置**，无此项该场景不成立 | 否，需 index |
| **9c-a** | **join 的三级判定：整除→快路径；非整除但可表达→告警放行；不可表达→拒绝**（§3.3.1d 第二层） | 第三级是**正确性**:`new_finest` 不整除某张量的相应维度时,`get_local_sharded_tensors` 会让**所有既有 client** 在重注册时抛 `ValueError`——一次扩容变成全链路故障。前两级是性能告警。**必须最先定,它决定弹性的语义边界** | **是**，与 1-5 同批 |
| 9c-a2 | **非整除 join 的重分片走版本边界,全体重注册 + `UPDATE_INFO` 重广播** | 「同一地址的不同切分」对远端同样是陈旧描述,与 §3.7.3.0a 同构。复用撤 holder → 等读者归零 → 全体重切重注册 → 驱逐旧对端并重广播;单边 RDMA 无处校验 epoch,见 §3.7.3.0a 更正 | 否，依赖 8 |
| 9c-b | **握手协议从一次性屏障改为事件驱动 join / leave** | `ps_manager.py:907` 的八步直线没有「中途加入」状态（§3.3.1d 第一层）。工作量项,非难题 | 否，依赖 6b |
| 9c-c | **join / leave 走 partial MD 两段式 + 缩容吊销 descriptor（带确认）** | 实测 **1884x** 压缩（conn-info 401B vs 全量 1.6MB @4096 区）。缩容不吊销会留下可用的陈旧 descriptor,与缺 epoch 同构的静默损坏（§3.3.1d 第三层与末段） | 否，依赖 9c-b |
| **9d-a** | **配置期防御五项**：`model_dtype` 真实读取（不再按后端硬分流，§3.8.0）、offload 双 flag 校验拒绝、uneven sharding 断言、finest mesh 可观测、**LoRA `merge=false` 拒绝**（§3.11） | **纯防御性**，不做则以难归因的方式失败（传输期报错而非校验期、按错 dtype 准备缓冲、**LoRA 下静默同步基座模型**） | **是**，与 1-5 同批 |
| 9d-b | FSDP2 的副本数运行时推导 | 纯 FSDP2 无 DP 副本，§3.5.7 的 ① 不可用 | 否，**与 8b 同批实现** |
| 9e | **删除 `fsdp_converter.py` 的 FSDP1 分支** | FSDP1 已被 `config.py:303` 拒绝，保留它只是死代码 | 是 |
| 10 | delta 传输（跨 DC / 慢链路） | 每步仅 1%~3% 字节变化 | 是，正交 |

落地顺序建议：**0 已完成**（micro-benchmark，已消除 NIXL API 的不确定性），
然后 **1-4 与 9d-a**（独立、立刻见效，且 3 已在削减 CPU 存储），
接着 **5b → 5**（这个顺序不能反：5 的双 buffer 流水依赖 5b 把 post/wait 拆开，
先做 5 会得到一个「看起来有流水、实测没有重叠」的实现，§3.6.4），
然后 **6 / 6b / 6c 必须先做**，6 是大 MoE 与 1TB 场景的硬阻塞（否则连握手都过不去），
6b 避免直供成功后瓶颈转移到 index，
最后 **7 → 8 → 9**（这三者必须一起才能安全地删掉 PS worker）。
**9b 与 9 同批**（holder 对称化是 `is_rollout_mode_in_actor=True` 的正确性前提，
不是优化）。**9c-a 与 1-5 同批**（三级判定本身很小，但它定义了弹性的语义边界，
而其中「不可表达必须拒绝」那一级是正确性，越晚定越贵）；
**9c-a2 / 9c-b / 9c-c 在启用弹性伸缩前必须完成**。
**9d 已拆为两半**：`9d-a` 的四条是配置期防御，不依赖 index，**与 1-5 同批做**；
`9d-b` 的副本数推导与 8b 耦合（8b 的 DP 接管在纯 FSDP2 下不可用），
两者一起实现而不是先写死 Megatron 假设再回头改。
**8c 要分两半看**：其中的**版本级等待**（发布 v+1 前等 v 的读者清零）
在 **FSDP2 + `model_dtype=bf16` + bf16 rollout** 下是**正确性必需**（§3.8.0a），
启用该配置前必须完成；而 **per-chunk 写屏障**（把等待粒度从 version 降到 chunk）
始终是优化，只在「实测 pause 窗口确实拖累 rollout 吞吐」时才值得，
先用验证项 4 测出窗口时长再决定。
**Megatron 同精度下两者都是优化**，因为 master params 已经把冲突窗口收窄了。
10 是独立增量，可随时插入（文档无第 11 项，此前笔误）。
完整的第二轮审核见 [weight_transport_review2](weight_transport_review2)。

**8c 的机制细节**（从 §3.4.3 移来，那里只保留取舍原则）：

```
版本级等待（FSDP2 bf16 同精度下必需，Megatron 同精度下是优化）：
  发布 v+1 之前，等待对 v 的在飞读者计数归零
    上限 conflict_wait_budget_ms，建议默认为单步时长的 5%
    超限则发布 v+1 并把 v 标记为 INVALID
      在飞读者收到 INVALID -> 放弃本轮，重拉 v+1（更新的权重，算法上是改善）

per-chunk 写屏障（始终是优化，把等待粒度从 version 降到 chunk）：
  仅 Megatron 同精度路径可用——它的 master params 让冲突窗口窄到
  「param 回写那一瞬」，所以 per-chunk 判定有意义
  FSDP2 bf16 下整个 optimizer step 都在写，降粒度没有收益
```

「在飞读者」是一个 **per-chunk 的整数计数器**，不是版本化租约表：
读者开始读 +1、结束 -1，由 index 维护但 trainer 可本地缓存。
**这比「版本 × chunk 租约表」简单一个量级**——
同精度路径只需回答「这段现在有人读吗」，不需回答「谁预约了哪个版本」。

第 10 项：verl 的 `delta_sharded` 与本设计正交（它优化「发什么」，本设计优化「发给谁、怎么发」），
在跨 DC 或慢链路时才是主要收益项，不应作默认。它压缩的是**传输**而非 reload。
注意它需要一份 pinned 快照作 diff base，**在约束 (b) 下这本身就是一份额外 CPU 权重存储**，
所以只能是显式 opt-in 的例外。

---

## 5.1 待优化清单（已评估，暂不纳入）

| 项 | 为何暂不做 | 重新评估的触发条件 |
|---|---|---|
| **跳过未变化权重的量化与传输**（冻结层、未激活 expert、lr=0） | 判定成本低但需要与 optimizer 的 grad 状态耦合，且收益依赖具体训练配方 | 实测到量化或传输占端到端比例显著，且确认有大量冻结层 |
| **渐进可用（部分 chunk 到达即开始生成）** | **vLLM 层面不支持，见下** | vLLM 提供「按层增量切换」的引擎 API |
| delta 传输 | 需要一份 pinned 快照作 diff base，违反约束 (b) | 跨 DC 或链路显著慢于 GPU 时 |
| 跨节点 peer | 阶段二，收益随 R 增大才显现 | 实测到 trainer 出口饱和 |
| 元数据 RDMA 按需拉取 | MoE 分组后扇出可能已够小 | 实测握手期元数据总量仍超预算 |
| **W 跟随训练阶段的前馈调整** | 有了 7b 的回调发布后，前馈要解决的主要问题已消失；且 `overlap_grad_reduce` 下不存在干净的 bwd 阶段，阶段枚举本身不准。准确信号需 hook `grad_sync` / `param_sync`，比阶段上报更侵入 | 实测「发布后的持续读取与 backward 的 NCCL 争带宽」是真实瓶颈 |
| **rail 感知选源** | `NetworkTopology` 无 rail 概念，万卡多轨道下有真实带宽差。UCX 本身有 multi-rail 支持，所以数据面可能已受益，但选源决策看不到 | 万卡部署且实测到同 rail 与跨 rail 带宽差显著 |
| **colocate 下 transfer 与 decode 部分重叠** | 需要放弃「完全暂停」这个简单正确的模型，换来的是 transfer 期间的部分吞吐。cast kernel 的低优先级 stream（§3.6.2c）是它的前置条件而非独立优化 | 实测 colocate 的 pause 窗口占 eval 时长比例显著 |

**渐进可用为何不可行**（已查证 vLLM 源码）：

vLLM 的增量加载路径 `initialize_layerwise_reload`
（`model_executor/model_loader/reload/layerwise.py:82`）会
**先把层参数恢复到 meta device**（`restore_layer_on_meta`），
等该层全部权重到齐后才 materialize 回真实设备。
也就是说**在 reload 期间模型根本不处于可推理状态**，
「部分 chunk 到达就开始生成」在引擎层面无处落脚。

而 PSRL 走的是另一条路径：NIXL 直写 `convert_vllm_inplace` 注册的运行时张量，
**完全不经过 layerwise reload**。这条路径下参数始终在真实设备上，
但也因此没有「哪些层已更新」的引擎级概念,
生成期间若权重被部分改写，得到的是半新半旧的混合策略,
这正是 §3.4.1 明确要排除的东西。

所以渐进可用需要的是「引擎能按层原子切换新旧权重」这个能力，
vLLM 当前不提供。**这一项从方案中移除，而非保留为可选优化。**
PSRL 现有的 `pause_generation` + 直写 + `resume_after_sync` 才是正确的语义:
传输期间暂停，传完整体切换。

---

## 6. 风险与边界

- **读写冲突已收窄为一条规则。** cast 路径（train/rollout dtype 不同）读的是发布副本，
  与 optimizer 无冲突，副本只在推送锁内刷新，所以也不与在飞读者冲突（§3.4.2）。
  代价是副本常驻一个版本：W 个独立 instance 的进度差没有上界，
  流式复用的 slot 要么让 trainer 等最慢的读者，要么让迟到读者读不到 v。
  **这条不能省：省掉的后果是 rollout 读到半新半旧的 chunk，且只在读者进度差异大时偶发。**
  同精度路径直读参数，用 per-chunk 在飞读者计数器加写屏障放行，需埋点统计等待时长 p50/p99。
  两条路径的选择由「源头是否产生新字节」决定，不由 dtype 枚举决定。
  **写屏障的适用范围是 Megatron + disaggregate + 同精度这一格**（§3.9.0）：
  colocate 靠时序互斥；FSDP2 的 fp32 默认配置走 cast，bf16 配置则必需版本级等待（§3.8.0a）。
- **index 控制面是最可能的实际单点瓶颈。** 见 §3.2 的三条约束（plan 本地缓存、
  租约按 instance 聚合、热路径走 gRPC）。必须在直供之前完成，否则瓶颈只是转移。
- **扇出不能依赖 sync 策略的时序巧合。** greedy 本身有自然分批（§3.5.1），但版本发布是
  同时事件，全体空闲时仍会一轮并发。扩散必须由 index 用 trainer 并发上限 W 强制形成，
  且 `pause_for_sync` 要从「整批统一暂停」下移到「该 instance 真正开始读」。
- **peer 供给会抢 rollout 自己的推理资源。** 每 peer 限 `P` 个并发读者（建议 1~2），
  colocate 模式下直接禁用 peer 供给。
  **但 peer 不需要 slot ack**：它供的是已落地的运行时参数而非流式 wire buffer
  （rollout 侧无接收缓冲，NIXL 直写注册张量），版本相等性已足够（§3.5.3 的 2）。
- **只能从同版本 peer 读（§3.5.2）。** 从 `v-1` 的 peer 读等于传输层偷偷加了一个单位
  staleness，算法侧不知情。部分 chunk 读 v、部分读 v-1 更糟，那是一个不存在的混合策略。
  代价是每个新版本的第一波必须从 trainer 读，扩散不跨版本累积，这正是 W 存在的理由。
- **拓扑数据精度有限，不要在它之上做连续优化。** `network_topology.py` 的带宽是硬编码常量，
  且 `LOCAL_NODE_IP` 全仓库只有定义没有设置处。只做「同节点 / 跨节点」这种离散判定，
  并建议把节点身份从 IP 换成 Ray node id（`get_node_id()` 已有）。
- **colocate 下应禁用 peer 供给。** colocate 时 rollout 与 train 共享 GPU，
  一个 rollout 为 peer 提供读取会占用它自己的 NIC 带宽与 NIXL progress thread 的 CPU，
  直接拖慢 eval 本身。
- **失去了 PS 作为版本快照的调试能力。** 今天可以对比「PS 里的 v」与「rollout 里的 v」
  来区分传输错与训练错（`_debug_log_ps_info` 就为此存在）。直供下这个中间观测点消失，
  需补一个可选的 checksum 校验模式（miles 的 `--check-weight-update-equal` 与
  `local_weight_checksum.py` 是现成范式）。
- **`ParamSyncPlan` 的 push 成对语义必须改造。** 见 §3.3。不改造会让 rollout 读到
  被 `after_push` 还原过的值。
- **布局转换产物是第三类发布缓冲，约 `0.15×M` 常驻，且违反约束 (b) 的字面表述。**
  `ConcatenatedQKVSync` 的 `torch.cat`（`megatron_converter.py:286`）与
  `DTypeCastSync` 的 `.float()`（`:177`）物化了真实内存。
  它今天就存在、不是本设计引入的，但**直供让它从 lazy 变成必须常驻正确**，
  并进入版本发布的关键路径（§3.7.2.3）。
  只影响走 `ConcatenatedQKVMapping` 的模型，应在配置校验期打印以使显存预算可见。
  其余布局转换（`slice_qkv_proj` / `slice_gate_up_proj` /
  `split_dim0_fused_fsdp_param`）全部基于 `narrow`，是零拷贝视图。
- **注册的判据是「plan 里有无跨节点读者」，不是部署模式开关**（§3.7.3.0）。
  这条自然覆盖「disaggregate 下 trainer GPU 做 eval 切换」——
  那一格既不是纯 colocate 也不是稳态 disaggregate，初稿的二分漏掉了它。
- **真 offload 后重注册必须重新取址，初稿不保证。** `client.py:401` 的快速路径
  复用缓存 VA，仅在 TMS（保证 VA 不变）下安全；真 offload 会让它注册到错误地址，
  远端 RDMA 读到垃圾（§3.7.3.0a）。**根治办法是注册 wire buffer 而非参数**，
  buffer 不参与 offload，地址天然稳定。
- **地址变更需要 `registration_epoch`。** 隐式广播无法阻止远端用陈旧 descriptor 发起读。
  holder 协议保证「没有在飞读者」，epoch 保证「没有陈旧 descriptor」，两者缺一不可（§3.7.3.0a）。
- **colocate 不追求零拷贝 IPC，与 verl / vLLM / miles 对齐。** 三家的 colocate
  路径全部无条件 `.contiguous()` + `copy_` 进一块可复用 IPC buffer
  （verl 512MB、vLLM 1GB），理由是逐 key 判定「可否零拷贝」的正确性风险
  不值得省下同 GPU 内 0.3ms 的 D2D 拷贝（§3.7.2.1）。
  初稿提出的 per-key IPC 判定是过度设计，已移除。
- **colocate 下两份参数只在 transfer 期同时在 GPU，eval 期 train 侧必须 offload 到 CPU。**
  这是 verl / SkyRL / miles 一致的最佳实践，目的是把显存腾给 KV cache（§3.7.3）。
  verl 的 `update_weights` 把 offload 排在 `resume(tags=["kv_cache"])` 之前并加了
  barrier 防 OOM（`engine_workers.py:811-826`）。
  初稿主张「权重常驻、注册不失效」，用 26% 的 KV cache 换实现便利，是错误取舍。
  **唯一两期都只需 `0.5×M` 的是 Megatron TE `fp8_param` + fp8 rollout**，
  它让 transfer 退化为无，这个组合三家都没覆盖（§3.7.2.2）。
- **FP8 的算法层影响不是 infra 能消化的。** rollout(fp8) 与 train(bf16) 的 logprob 有系统性偏差，
  需 off-policy / importance sampling 修正。infra 的边界是保证 rollout logprob 与 fp8 权重版本
  严格对齐。SkyRL 用 rollout-vs-trainer logprob gap 作为「权重确实落对了」的判据，
  PSRL 应同样把它作为回归指标而不是只看 reward。
- **trainer 成为数据面关键路径，DP 副本是结构性冗余但不普遍存在。** Megatron 与 HSDP 下
  同一 chunk 有多份相同副本，单 rank 故障可由同组接管（§3.5.7 的 ①）。
  **纯 FSDP2 下副本数为 1，这一层不存在**（§3.9.1），故障直接进 ②。
  副本数必须由 index 从 `NIXLSharding` 运行时推导，不能假定。
  整组故障或全体重启时返回 `CHUNK_UNAVAILABLE`，rollout 保持当前版本等下一版，
  而不是读旧版本或卡死。注意 peer 只是带宽优化，不是降级手段。
- **只支持 FSDP2，FSDP1 已被 `config.py:303` 拒绝**（`flat_param` 无法保证 in-place
  表示，而直供的前提是注册的就是参数本身）。`fsdp_converter.py` 的 `fsdp` 分支
  应随本设计删除。
- **FSDP2 后端有四处与 Megatron 不对称的地方，全部在 §3.9**：参数是 fp32 而非 bf16
  （**默认**如此，所以默认恒走 cast、恒无冲突）、无 master params
  （所以 `model_dtype=bf16` + bf16 rollout 下版本级等待从优化升为必需，§3.8.0a）、
  纯 FSDP2 无 DP 副本、`shard_mesh[0] = world_size` 放大每个 key 的 shard 数。
  **前三条是正确性，第四条是可扩展性。后两条共用一个解法：把 `fsdp_size` 调到节点内。**
- **`param_offload` 与 `offload_policy` 都与直供互斥。** FSDP2 的
  `offload_fsdp2_model_to_cpu`（`fsdp_utils.py:196`）就是 `model.to("cpu")`，
  在正常训练步之间重新分配每个 DTensor 的 local storage，让注册过的 GPU 指针悬空，
  而现有 sleep 钩子盖不住它。`CPUOffloadPolicy` 更彻底，让参数常驻 host。
  **两个 flag 必须都校验**：verl 在设置 `offload_policy` 时会把
  `_is_offload_param` 强制置 False（`transformer_impl.py:460`），
  只查 `param_offload` 会漏掉它（§3.9.3c）。
  **但 `optimizer_offload` 必须保留**，它是 colocate 切换释放 optimizer state 的
  唯一现成手段（§3.7.3.1）——现有的 `_sleep_*_model` helper 只碰参数和梯度，
  不碰 optimizer state，这是初稿的一处缺口。
- **FSDP2 的 uneven sharding 会在传输期而非校验期报错**（§3.9.2.1 末段）：
  DTensor 允许 pad，而 `get_local_sharded_tensors`（`nixl_spec.py:166`）要求整除。
  `get_sharding_for_param` 应在声明时断言并指向调小 `fsdp_size`。
- **弹性伸缩的阻塞点不在 NIXL,在 finest mesh 的全局耦合**（§3.3.1d）。
  NIXL 的增量能力齐备（`loadRemoteMD` / `invalidateRemoteMD` / `invalidateLocalMD` /
  **`getLocalPartialMD`**,Python 绑定全部存在）。真正的问题是
  `make_unified_sharding` 取全局 lcm,**一个并行度不兼容的新成员会抬高 finest mesh,
  让所有既有 client 重算 descriptor**。整除时 lcm 不变,是零成本快路径
  （沿 DP 维扩缩天然满足,§3.10.4c）；不整除时**告警放行而非拒绝**,
  因为更细的 mesh 仍能正确表达更粗的切分——代价是 CPU 重算与 plan 重编译,
  **不含重注册**（`_record_region_registration` 按 allocation 去重,与 shard 数无关）。
  **但整除性同时是一条运行时约束**:`new_finest` 不整除某张量的相应维度时
  （如 hidden=4096 遇到 finest=24）,`get_local_sharded_tensors` 会让
  **所有既有 client** 抛 `ValueError`——**这一类必须拒绝**,
  放行的结果是全链路故障而非变慢（9c-a 的三级判定）。
  非整除重分片还必须在版本边界进行并推进 `registration_epoch`（9c-a2）。
  `server.py:56/90` 的一次性屏障是第二层（9c-b,工作量项）。
- **缩容必须吊销 descriptor 并收齐确认,只撤 holder 不够**（§3.3.1d 末段）。
  撤销 holder 阻止的是「新读者发起」,但远端仍持有可用的 descriptor,
  retry 路径下会读到已释放显存。
  落地修正两点（实测,详见 §3.3.1d 末段与实现文档 E4）:
  吊销走 PSRL 的 notif 通道而非 `invalidateLocalMD`（后者无法广播、无法确认）;
  且**驱逐之后 NIXL 自己就会拒绝陈旧 descriptor**,
  所以这一项并不像初稿判断的那样需要与 `registration_epoch` 同构的额外机制,
  真正要保证的只是「每个持有者都确实执行过驱逐」。
- **QP 数量是万卡下的隐性上限**（§3.5.2.1 约束 ④），单边读仍需 NIC 维护 QP 状态。
- **GPUDirect 不可用会退化为 host bounce**，带宽可能差一个量级，必须探测（§3.6.2a）。
- **容错未在本设计完整覆盖。** holder 抽象天然支持（失效即摘除、pull 重新解析），
  但租约回收时机、plan 重编译、instance 动态增减需单独设计。WeightBridge 同样列为 future work。
- **block 对齐约束会拒绝某些 模型×TP 组合。** 这是 vLLM 引擎自身的约束，PSRL 只是提前报出。
  不要在传输层绕过它。
- **Megatron PP/EP 下 holder 归属是启动期错误，风险可控。** converter 只遍历本 rank 的
  `named_parameters()`，天然只登记自己持有的参数，comm plan 对「target 需要但无 source 提供」
  的 key 直接 assert（`No available key {key} in source client groups`）。
  所以 PP stage 覆盖不全在启动期暴露，而非静默读到垃圾。仍需在握手期校验覆盖完整。
- **LoRA 训练不在本设计覆盖范围内，且必须在配置校验期显式拒绝**（§3.11）。
  不拒绝的话 `convert_fsdp_inplace` 会把 base weight 当成全部权重注册，
  rollout 读到的是**不含 adapter 的基座模型**——静默错误，表现为 reward 不涨。
- **完成检测是 1ms 轮询，这是 chunk 调小的硬下界**（§3.6.4）。
  `client.py:1604` 与 `:1539` 的 `time.sleep(0.001)` 让每轮多付最多 1ms，
  512MB chunk（~10ms）下是 10% 开销，再调小就不划算。

---

## 7. 验证

### 7.0 测试分层与改造方案

**现状**：`tests/nixl/test_nixl_e2e.py`（891 行）已有 meta server + train client
（`fsdp` / `fsdp_hybrid` / `megatron`）+ gen client（TP/PP/EP/DP 可配）的骨架，
配置（`config/nixl_e2e.yaml`）已参数化并行度与 `train_dtype` / `gen_dtype`。
**骨架正是直供需要的两端**，所以改造而非重写是对的。

#### 三处必须先做的结构性改动

**(a) 正确性判据从「全 1」升级为坐标函数——这是第一优先级。**

现有 `assert_state_dict_all_ones`（`:95`）把权重设成全 1 再验证全 1。
**这个判据对本设计要验证的东西完全无效**：

| 错误类型 | 全 1 判据能否检出 |
|---|---|
| 数据没到 | 能 |
| **布局错误**（转置、reshape 错） | **不能**——全 1 转置后还是全 1 |
| **分片错位**（读到别的 rank 的 shard） | **不能**——全 1 的任意切片都是全 1 |
| **精度转换错误** | **不能**——全 1 cast 到 fp8 还是 1.0 |
| **MoE expert 映射错误** | **不能** |

替换为**每个元素等于其全局坐标的确定性函数**：

```python
def fill_by_global_coord(key: str, tensor: torch.Tensor, global_offset: tuple[int, ...]):
    # 值 = f(key 的稳定哈希, 该元素在全局张量中的多维坐标)
    # 要求：f 对任意坐标单射，且在目标 dtype 下可精确表示
    ...

def assert_matches_global_coord(key, tensor, global_offset, dtype_tol):
    # 逐元素比对，失败时报出第一个不匹配的坐标——直接定位到是哪一维错位
```

**两个设计约束**：值域必须落在目标 dtype 可精确表示的范围内
（fp8 e4m3 只有 ~3.5 位尾数，所以不能用大整数做哈希值），
且 `f` 必须对坐标单射，否则错位换位检不出来。
实现上建议用**小整数的位拼接**（如 `layer_idx * 2^k + row * 2^j + col`）
并在 fp8 下只用低位，而不是浮点哈希。

> **「值可精确表示」会让判据在 bf16 下失明。** bf16 只能精确表示 256 个整数，
> 取 `flat_index mod 2^8` 时任何偏移 256 倍数的错位都通过，而 4096 宽权重的每一行都是 256 的倍数。
> **传输只搬字节、不解释数值，所以判据不需要「可表示的数」**：把 `(key, version, 全局多维坐标)`
> 的哈希写进元素的**原始位**，错位元素的漏检概率是 `2^-位宽`。
> 全局坐标从**各方自己声明的** `NIXLSharding` 推出，与传输无关
> （`tests/nixl/coord_judge.py::fill_by_layout / check_by_layout`）。
> cast 路径下哈希取自目标 dtype，train 侧经精确的上转存储，下转后恰好还原。
> block FP8 下 train 侧每个 tile 存 `fp8 值 × 2^-k`，并让一个元素取 FP8 最大值，
> 于是 block 量化精确还原 FP8 位与 scale `2^-k`，两者都能逐位判定（`fill_fp8_exact / check_fp8_exact`）。

**(b) 删除 `create_ps_worker_group`（`:649`），改为 trainer 直供。**
`GenClientActor` 的 pull target 从 `ps_for_pull_name` 换成 train client，
这与 §3.3 的改动一一对应。**PS worker 本来就要删，测试不应保留它。**

**(c) 新增 colocate 模式。** 现有测试只有 disaggregate（train 与 gen 在不同节点）。
colocate 需要：同一组 GPU 上先跑 train client、再跑 gen client，
中间走 §3.7.3 的 offload + CUDA IPC 路径。**这是改造里工作量最大的一项**，
但它是 §3.7 全部结论的唯一验证手段。

#### 四维参数化矩阵

| 维度 | 取值 | 现状 |
|---|---|---|
| 后端 | `megatron` / `fsdp2` | 已有（删掉 `fsdp1`、`fsdp_hybrid` 保留为 HSDP） |
| 精度 | `bf16+bf16` / `bf16+fp8` / `fp32+bf16` / `fp32+fp8` | 配置项已有，**cast 路径待实现** |
| 放置 | `disaggregate` / `colocate` | **需新增 colocate** |
| 并行度 | train TP/PP/EP × gen TP/EP（含不匹配组合） | 已有，只需扩参数化 |

**不做全笛卡尔积**（2×4×2×N 会爆炸）。按风险选组合：

```
必跑（每次 CI）：
  megatron + bf16+fp8  + disaggregate + TP匹配      ← 推荐默认配置
  fsdp2    + fp32+bf16 + disaggregate + TP匹配      ← FSDP2 默认配置
  megatron + bf16+bf16 + colocate     + TP匹配      ← 时序互斥路径

每日跑：
  上述 × {TP 不匹配、EP 不匹配}                      ← shard 代数
  fsdp2 + bf16+bf16 + disaggregate                  ← 版本级等待必需的那一格（§3.8.0a）
  megatron + fp8+fp8 + colocate                     ← FP8-native，显存最优路径

发版前：
  全部 8 个精度组合 × 2 放置                         ← §7 验证项 14
  MoE 模型（Qwen3-30B-A3B 级）                       ← expert 分片与 MoE 分组
```

#### 分层与覆盖率

| 层 | 载体 | 覆盖的验证项 | 条数 |
|---|---|---|---|
| **L1 单元** | `test_nixl_sharding.py` / `test_comm_planner.py` 扩展 | shard 代数、finest mesh、FSDP2 uneven sharding 断言、MoE 分组后的多维 shard | — |
| **L2 e2e 单节点** | `test_nixl_e2e.py` 改造 | 7、8、11、14、**15**、16、17、17b、19、22、23 | **11** |
| **L3 e2e 多节点** | 新建 | 5、**15b**、18、20、21、24 | **6** |
| **L4 规模实测** | 人工，无法自动化 | 1–4、6、9、10、12、13、25 | **10** |

**L4 的 11 条是本方案的真实风险**——它们恰恰支撑着「万卡 / 1TB」的声明，
而只能在真实集群上验证。**这一点必须在落地计划里显式承认**，
不能把「设计上可扩展」与「已验证可扩展」混为一谈。

**L2 是性价比最高的一步**：骨架已存在，而「判据升级 + 四维参数化」
一次性覆盖最容易出错的部分（格式转换、精度矩阵、后端差异、shard 代数）。

### 7.1 验证项清单

1. **零额外存储**：实测稳态下无 PS worker 进程、host 侧无权重常驻，且 cast 显存
   上界为 `2×chunk` 而非 `O(M)`。**判据是「换一个大一倍的模型，这个数字不变」**——
   若它随模型增长，说明 chunk 流式复用没生效，退化成了冗余全量副本。
   同时单独记录布局转换产物的常驻量（§3.7.2.3），它**确实**随模型增长，
   两者不要混在一个指标里。
2. **流量最优性与重叠率**：实测 trainer 出口字节 == `M`、每 rollout 节点入口 == `M_v`，
   并与 `T_min` 比值（WeightBridge 在 16~64 节点报 `1.27×`~`2.12×` ideal，可作参照）。
   **但不能只测带宽**——必须**分别计量 `T_rdma` 与 `T_fmt` 并报告重叠率**（§3.6.0）。
   大 MoE 下 `T_fmt` 约为 `T_rdma` 的 2.8 倍，**端到端下界是 `max` 而非 `T_rdma`**。
   **判据**：端到端 ≈ `max(T_rdma, T_fmt)` 说明流水生效；
   ≈ `T_rdma + T_fmt` 说明完全没重叠，此时「带宽打满」是一个误导性指标。
   同时记录实测的 `T_fmt / T_rdma` 比值，它是 §3.5.6 自适应 chunk 的输入。
3. **单源瓶颈已消除**：固定 trainer、逐步增加 rollout replica 数，AGST 应为亚线性
   （miles P2P 在此处近线性退化，是对照基线）。**同时记录控制面消息数**，
   确认它不随 replica 数平方增长（§3.2）。
4. **pause 窗口与冲突等待分布**：先测**每个 instance 的 `pause_for_sync` 到
   `resume_after_sync` 时长**（这是 8c 是否值得做的唯一依据，§3.7.2.4），
   再测同精度路径下写屏障等待的 p50/p99 及其随 staleness 与 replica 数的变化。
   cast 路径下「等 optimizer」应恒为零，这是「发布副本确实隔离了 optimizer 写」的判据。
   另需注入一个不响应的 designate，确认 follower 在 `peer_timeout_s` 后改读上游，
   而不是让同组卡住，这是慢读者不拖垮全局的判据（CPU 档已由 `test_pull_engine.py` 覆盖）。
   **同时记录各 instance 暂停时刻的分布，并特意构造「全部 instance 同时空闲」的场景**
   （长训练步 + staleness=0，§3.5.1 说明这是常态而非极端）。
   此时自然错开必然失效，**判据是 W 把并发读者数压在上限内**，
   而不是「错开确实发生」——后者是运气，前者是设计。
5. **注册有效性不变量**（三条，都测静默损坏，必须有）：
   (a) 在 trainer DDP buffer resize 后、重注册前注入一次 pull，
   必须被 index 拒绝，而不是读到已释放页。
   (b) **地址变更后的正确性**（§3.7.3.0a）：让 trainer 做一次真 CPU offload + reload
   （非 TMS pause，确保 VA 改变），然后让远端 rollout 拉取，
   **逐字节校验内容正确**。若走 `client.py:401` 的缓存地址快速路径，这条必然失败——
   这正是它要暴露的缺陷。
   (c) **陈旧 descriptor 必须被拒绝**：远端持有重注册前的 descriptor 发起读，
   必须因 `registration_epoch` 不匹配而被明确拒绝，不能静默读到错误地址。
5b. **注册次数与判据**：断言 `register_memory` 的调用次数等于
   「plan 中有跨节点读者的 `(key, shard)` 数」（§3.7.3.0），
   colocate 纯本地场景下应为 0，混合部署下只覆盖远端那部分。
   **若等于总 shard 数，说明判据没生效，退化成了无条件注册。**
6. **colocate 切换**：**首要判据是 eval 期间的 KV cache 可用量**，
   它必须与「不做直供、纯 vLLM sleep/wake」的基线相当（§3.7.3）。
   若明显更低，说明权重没有真正让位，整个 colocate 路径的收益为负。
   其次测切换耗时（权重 D2H/H2D + 重注册 + optimizer reload，预期亚秒级，
   对比 eval 的分钟级应可忽略），以及 transfer 期的显存峰值
   （§3.7.3 的第一列，FSDP2 + bf16 rollout 的 `3×M` 最容易 OOM）。
   **并断言 colocate 路径上 `register_memory` 调用次数为 0**（§3.7.3）——
   colocate 走 CUDA IPC，不应触及 NIXL 注册，这是三家的一致做法。
7. **atomic group 不变量**：故意只送达 group 的一半，必须报错而不是静默 no-op。
8. **等价性**：小模型上逐张量比对「本设计产出」与「vLLM 从原生 fp8 checkpoint 加载 +
   `process_weights_after_loading`」，在 UE8M0 / requant 容差内一致。
9. **故障注入**：杀掉一个 trainer rank，pull 必须由同 DP 组接管且 rollout 无感知。
   杀掉整个 DP 组，必须返回 `CHUNK_UNAVAILABLE` 且 rollout 保持当前版本继续生成，
   不卡死、不读到半更新模型。
   **判据要具体到两点（8d，§3.5.7 末段），否则这条会假通过**：
   (a) 该 replica 在降级后**仍在 SMG 的可用集合里**且继续服务请求——
   不是 `fail_sync` + `/workers/pause` 之后的「不可用」；
   (b) 降级全程 SMG 记录的该 instance 版本**始终等于它实际持有的版本**，
   不存在「声明 v+1 而实际 v」的中间窗口。
   **(b) 必须在请求粒度上验**：降级窗口内到达的请求，其记账版本要与
   生成时实际使用的权重版本一致，否则 staleness 账本被污染且无人发现。
10. **扩散有效性**：固定 R 与 W，记录每个 instance 的源分布（trainer / 同节点 peer），
    开启 chunk 偏移错开前后对比第一波的 peer 命中率。
11. **四条运行时断言**（§3.4.6）必须各有一个故意违反的测试，确认是报错而非静默通过。
12. **`is_rollout_mode_in_actor=True` 全流程**：启动时全 GPU 跑 rollout 能拿到 v0、
    **首次**切回 trainer 从 checkpoint 拿到训练起点、再切回 rollout 仍正确。
    **这是 §3.3.1a 的回归测试，缺它则该场景的正确性无保障。**
    **另需单测稳态路径**：连续两轮 eval，第二轮切回 trainer 后
    权重版本必须等于 eval 前的版本（**不是上次 save 的版本**）。
    这条测的是「稳态切回不读盘」，读盘会静默回退训练进度且不报错。
13. **弹性伸缩**（§3.3.1d 的四条，前两条是正确性）：
    (a) **扩容 join 的三级判定**（9c-a）：
    整除的新 instance 加入后，既有 client 的 descriptor 与 finest mesh
    **逐字段不变**（快路径判据）；
    非整除但可表达的（如 hidden=4096 下 TP=16）**告警后成功加入**，
    且重分片后逐张量校验内容正确；
    **不可表达的（如 hidden=4096 下 TP=3，finest 抬到 24）必须在准入期拒绝**，
    并断言**拒绝后既有 client 仍处于可服务状态**——
    这一条是重点，放行它的后果是所有既有 client 在重注册时集体抛
    `ValueError`，把扩容变成全链路故障。
    (b) **缩容的地址正确性**：缩容的 instance 先撤 holder、等读者归零、
    吊销 descriptor 并收齐确认、再 sleep。**让远端用它退出前的
    descriptor 发起读，必须失败而非读到垃圾**——与验证项 5(c) 同构。
    再加一条：**任一持有者未确认吊销时，必须拒绝释放显存**，
    因为「已发出吊销」与「吊销已生效」之间的差别正是这一步存在的理由。
    (c) 扩容的新 instance 能冷启动拿到当前版本，期间无读者失败。
    (d) **join 的扇出代价**：记录每次 join 时每个既有成员加载的 MD 字节数，
    应为 conn-info 量级（~400B）而非全量（~406B × region 数），
    这是 partial MD 两段式是否生效的判据。
14. **精度矩阵**：§3.8.0 的**十个**支持组合各跑一次端到端
    （Megatron 五格 + FSDP2 五格，后者含 `model_dtype=bf16` 两格）。
    **没有需要被拒绝的精度组合**——初稿曾拒绝 FSDP2 + bf16 + bf16，
    §3.8.0a 已更正为支持。
    **其中 `FSDP2 + bf16 + bf16 + disaggregate` 是必跑项**：
    它是唯一「版本级等待属正确性而非优化」的一格，
    关掉版本级等待后必须能观测到权重被破坏（读到半新半旧），
    **这是该机制存在价值的判据**。
15. **LoRA 的两条路径**（§3.11）：(a) `model.lora.merge=true` 端到端跑通，
    rollout 侧权重逐张量等于「基座 + adapter 合并」的参考值；
    (b) `lora_rank > 0` 且 `merge=false` 时**配置校验必须拒绝**。
    **(b) 是重点**——不拒绝的话会静默同步基座模型，reward 不涨且无报错。
15b. **chunk 下界与流水有效性**（§3.6.4）：扫 chunk ∈ {4GB, 512MB, 256MB, 128MB}，
    端到端时间应先降后升，**升起来的那一点就是轮询开销反超流水收益的位置**。
    判据是 256MB 处仍在下降段；若不是，说明 `get_new_notifs` 改造不可省。
    同时断言 post 与 wait 已拆开——用两个 chunk 的提交时间戳验证
    第二个 post **早于**第一个 wait 返回，这是双 buffer 成立的充要条件。
16. **后端对称性**：同一模型分别用 Megatron 与 FSDP2 训一步，
    rollout 侧落地后的权重应逐张量一致（在各自 train dtype 的 cast 容差内）。
    这条测的是 §3.9.2 两条格式转换路径的等价性，缺它则后端切换是未验证的。
17. **FSDP2 专项五条**：(a) 纯 FSDP2 下杀一个 rank，必须返回 `CHUNK_UNAVAILABLE`
    而**不是**尝试 DP 接管；HSDP 下同一测试必须走 DP 接管（§3.9.1）；
    (b) `param_offload=True` **与** `offload_policy=True` 各自都被配置校验拒绝
    （§3.9.3c，两个 flag 分别测，因为 verl 会把前者置 False）；
    (c) 握手期打印的全局 finest mesh 各维大小落在阈值内（§3.9.2.1）；
    (d) 构造一个 dim0 不被 `fsdp_size` 整除的参数，必须在配置校验期报错
    而非传输期（§3.9.2.1 末段）；
    (e) MoE 的 `shard_dim=0`（切 expert）与 `shard_dim=1`（切 projection）
    两条路径分别端到端跑通（§3.9.2）；
    (f) **`model_dtype` 真实读取**：分别用 `fp32` 与 `bf16` 启动，
    断言注册给 NIXL 的字节 dtype 与配置一致——
    `ray_trainer.py:1962` 今天按后端硬分流，**`bf16` 下会按 fp32 准备缓冲**（9d-a）。
17b. **NCCL 通信组防回归**（§3.10.4b）：角色切换前后断言
    `torch.distributed.is_initialized()` 恒为真。
    **这不是在测当前实现**（PSRL 从不销毁进程组），
    而是防止将来有人为省显存去销毁它——那会让切换成本涨一个量级且不报错。
18. **布局转换产物的时效性**（§3.7.2.3）：对走 `ConcatenatedQKVMapping` 的模型，
    在 optimizer step 之后、发布 v 之前**故意跳过** QKV 的重新 cat，
    rollout 读到的必须是可检测的陈旧值而非静默通过。
    这条测的是「第三类字节确实在发布点被刷新」，缺它则布局陈旧是静默错误。
19. **并行度不匹配的 shard 代数**：train TP=8 ↔ rollout TP=4、
    train TP=4 ↔ rollout TP=8 两个方向各跑一次，逐张量比对。
    这条测 `find_finest_shard_mesh` 的 lcm 对账（`server.py:169`），
    是「布局对齐不需要 per-model 代码」这个论断的判据。
20. **colocate 的可复用 buffer 上界**（§3.7.2.1）：确认整个 transfer 期间
    只存在**一块** IPC-shared 分配，且其大小等于配置的 buffer size、
    不随模型或张量数增长。同时验证 uint8 buffer 对多 dtype 的
    `.view(dtype).view(shape)` 映射与对齐处理正确
    （含 fp32 的 `A_log`、fp8 的 weight+scale 混合在同一 bucket 的情形）。
21. **多角色切换**（§3.10）：(a) `_ensure_on_gpu` 的二维状态生效——
    连续两个同角色操作之间搬运次数为 0，且 ref logprob 不触发 optimizer 加载；
    (b) 三档 colocation 各跑一次，确认互斥范围等于 `colocation_group` 而非全局；
    (c) **ref 备份的正确性**：`restore("ref")` 后算出的 logprob
    必须与「独立 ref 进程」基线逐元素一致；
    (d) `ref_update_interval` 生效后，ref 版本等于当时的 actor 版本。
22. **role 维度隔离**（§3.10.3）：actor 与 critic 同时发布版本，
    各自的 holder 表项不得互相覆盖。故意用相同版本号（v3 / c3）跑一次，
    确认两者解析到不同的 holder。
23. **地址变更后不读到错误地址**（§3.7.3.0a 更正）：让 trainer 做一次 offload+reload
    但不推进版本，读者在收到 `UPDATE_INFO` 前用旧 descriptor 发起读，
    **必须失败（`NIXL_ERR_NOT_FOUND`）而非读到 reload 后地址上的字节**；
    收到后重新加载对端 info，下一次 pull 坐标判据全过。
24. **档位 B 的 GPS 一致性**（§3.10.4a）：CPU holder 在版本推进后
    必须自动退出候选集，**不得因链路成本低被选中**。
    故意让 GPS 停留在 v 而发布 v+1，确认选源不会命中它。
25. **端到端**：rollout logprob 与 bf16 参考的偏差落在预期区间，长跑中不漂移。

---

## 8. 自动模型接入：本期沿用手写，但该决定已在 FP8 文档中重开

按约束 (a)，**本期**沿用现有显式 layout plan（`weight_layouts.py` + `model_mappings.py` +
`weight_layout_transforms.py`），不引入探测式对应关系发现。

**但这个排除的理由已不成立，重开的论证见 `fp8_weight_transport.md` §9。**
关键事实是本节初稿没有核实的两条：Megatron 侧**早已是自动的**（`AutoBridge`），
而 vLLM 侧要的结构已由上游机器可读地声明（`packed_modules_mapping` 覆盖 143 个模型类、
`get_expert_mapping()` 覆盖 16 个 MoE 类）。所以这是一个**单侧**问题，
且是「读声明」而非 WeightBridge 式的「探测 `load_weights` 副作用」——
后者才是本节真正想排除的东西。FP8 另给了一个推力：手写方案下量化会让 per-model
成本翻倍，而推导方案下 scale 的 layout 是 weight layout 的免费推论。

下面两点接口余地因此从「将来不返工」变成了**增量上线的依据**：
plan 是数据，所以自动推导可以一个模型一个模型地切，随时回退。

- **canonical 命名空间不变。** PSRL 的 unified HF state dict 正是 WeightBridge 所说的
  「作为桥梁的 checkpoint」，自动化只会替换**映射的来源**，不改变 key/shard 抽象本身。
- **plan 是数据而非代码路径。** §3.5 的三阶段 plan 在握手期编译并存入 index，
  编译输入是「两侧的 (key, shard) layout」。无论这个 layout 来自手写 builder 还是将来的
  自动推导，planner 与执行层都不需要改。
