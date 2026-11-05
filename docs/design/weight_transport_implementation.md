# 权重传输落地实现方案

> [weight_transport](weight_transport) 的实现清单。每一项给出**改哪个文件、做什么、
> 完成判据、验证方式**，以便实现期逐项打勾并确认覆盖完整。
>
> **本文不重复设计论证**，只做「设计 → 代码」的映射。设计理由一律回指主文档章节。

---

## 0. 怎么用这份文档

每个任务有四个字段，缺一不可：

| 字段 | 含义 |
|---|---|
| **落点** | 要改的 `file:line`（行号为写作时的锚点，实现时以符号名为准） |
| **做什么** | 具体改动，不是目标描述 |
| **判据** | 怎么算完成。**必须是可执行的断言或可观测的量，不是「实现了 X」** |
| **验证** | 哪一层测试覆盖（L1 单元 / L2 单节点 e2e / L3 多节点 / L4 规模实测） |

**阶段之间有硬依赖**，见 §1 的依赖图。**同一阶段内的任务可并行**。

**「不可分割块」的含义**：阶段 D（7/8/9）一旦开始就必须完成，
因为它删除 PS 数据面，中途停止会让链路不可用。
阶段 A/B/C 全部是可独立回滚的。

---

## 1. 阶段划分与依赖图

```
阶段 A  独立项，可立即做，各自可回滚
  A1  1    prepped descriptor（int32 ndarray 索引）
  A2  3    Nx5 strided descriptor
  A3  5b   post/wait 拆分 + get_new_notifs 完成检测     ← 阶段 B 的前置
  A4  6c   make_xfer_tag 去 pickle
  A5  9d-a 配置期防御五项
  A6  9e   删除 fsdp_converter 的 FSDP1 分支
  A7  9c-a join 三级判定
  A8  2    colocate CUDA IPC + stage offload
  A9  4    源头 FP8 wire

阶段 B  执行层，依赖 A3
  B1  6d   chunk / commit group 概念分离               ← 6 的前置
  B2  5    三类重叠 + slot ack bitmap
  B3  6c'  chunk 按 cost model 自适应 + 轮级决策本地化

阶段 C  元数据与控制面，依赖 B1
  C1  6    canonical MoE 分组                           ← 大 MoE 硬阻塞
  C2  6b   控制面下热路径（plan 本地缓存 + gRPC）        ← 阶段 D 的前置
  C3  7b   版本发布挂 on_train_step_end

阶段 D  不可分割块：trainer 直供（依赖 C1 + C2）
  D1  7    三阶段去冗余规划 + peer 优先 + 自适应 W
  D2  8    版本可见性 + 三条运行时断言 + holder role 维度
  D3  9    trainer 直供 + 删除 PS worker + bootstrap
  D4  9b   holder 对称化 + bootstrap 走 rollout 代表读盘
  D5  7c   chunk 起始偏移错开
  D6  8d   CHUNK_UNAVAILABLE 真能继续生成               ← 与 8b 同批
  D7  8b   DP 副本接管 + CHUNK_UNAVAILABLE
  D8  9d-b FSDP2 副本数运行时推导                        ← 与 8b 同批

阶段 E  正确性补强，依赖 D
  E1  8c   版本级等待（FSDP2 bf16 同精度必需）+ per-chunk 写屏障（可选）
  E2  9c-a2 非整除 join 重分片走版本边界 + 推 epoch
  E3  9c-b 握手改事件驱动 join / leave
  E4  9c-c partial MD 两段式 + 缩容吊销 descriptor（带确认）

阶段 F  正交可选
  F1  10   delta 传输（跨 DC），显式 opt-in
```

**关键顺序约束**（违反会导致返工或假完成）：

1. **A3 → B2**：不拆 post/wait，双 buffer 流水在实现层无处落脚（§3.6.4）
2. **C1 → B2 的收益**：MoE 分组前 `T_fmt` 主导，overlap 做了也看不到收益（§3.6.0）
3. **B1 → C1**：MoE 分组让单 key 变成 11GB，必须先有「chunk 可切开 key」
4. **C2 → D**：直供成功后瓶颈会转移到 index
5. **D6 与 D7 同批**：不改 D6 则 D7 落地即撞 assert（§3.5.7 末段）
6. **D8 与 D7 同批**：纯 FSDP2 无 DP 副本，先写死 Megatron 假设会返工
7. **E1 在启用 FSDP2 + `model_dtype=bf16` + bf16 rollout 之前必须完成**（正确性，§3.8.0a）
8. **A7 越早越好**：它定义弹性的语义边界，影响调度层设计

---

## 2. 阶段 A：独立项

### A1 — prepped descriptor（设计项 1）

**落点**：`psrl/utils/nixl/client.py`（`client_read` / `client_write` / `_cached_xfer_descs` 路径）

**做什么**：
1. 握手期对每个 `(local_client, remote_client)` 对调一次 `prep_xfer_dlist`，
   缓存 `(local_handle, remote_handle)`，与 plan 的下标对齐
2. 热路径用 `make_prepped_xfer(op, local_handle, local_idx, remote_handle, remote_idx)`
   替换 per-(key, shard) 的 `initialize_xfer`
3. **索引参数的类型必须在接口上固定为 `np.ndarray[int32]`**，不接受 list

**判据**：
- `initialize_xfer` 在热路径（版本同步期间）调用次数为 **0**，
  只在握手期出现
- 索引构造处有 `assert idx.dtype == np.int32`，
  且有一个单测故意传 list 并断言被拒绝
- 同等 descriptor 数下提交耗时相对改造前下降 **≥ 5×**（1024 条）

**为什么判据是「0 次」而非「变少」**：§3.6.3 实测传 list 会慢 15× 并抵消全部收益，
而 list 与 ndarray 都能跑通、不报错。只有「热路径零 `initialize_xfer` + dtype 断言」
这两条同时成立才能保证收益真的拿到了。

**验证**：L1（新增 `tests/nixl/test_prepped_xfer.py`：dtype 断言、索引复用、热路径计数）
+ 已有的 `bench_nixl_prepped_xfer.py` 作为性能回归基线

---

### A2 — Nx5 strided descriptor（设计项 3）

**落点**：`psrl/utils/nixl/client.py`（`max_pinned_temp_memory_slots` 整套逻辑），
`psrl/utils/nixl/nixl_spec.py`（`get_local_sharded_tensors` 的非连续分片路径）

**做什么**：
1. 对规则跨步的非连续 shard，用 Nx5 `(addr, len, dev_id, stride, count)`
   经 `prep_xfer_dlist` 表达
2. **删除** `max_pinned_temp_memory_slots` / `_pinned_slot_running_read_xfer` /
   `_temp_desc_bytes_mapping` / `_pinned_memory` 整套 pinned 中转
3. 保留一条**非规则跨步**的回退路径（strided 表达不了的情形），
   但它不应再用 pinned 中转，而是直接多条 Nx3 descriptor

**判据**：
- `_pinned_memory` 这个属性在代码中不存在（grep 零命中）
- `wait_for_server_temp_mappings` / `notify_all_client_temp_mappings` /
  `_client_temp_mappings` 全部删除，`nixl_protocol()` 从八步降为六步
- MoE 模型握手期 descriptor 数相对改造前下降 **≥ 100×**（expert 维度规则跨步）
- CPU 侧 pinned 内存占用为 **0**（约束 (b)）

**验证**：L1（`test_nixl_sharding.py` 扩展：strided 表达的分片等价性）+ L2

**注意**：`nixlXferDList` 在任何 NIXL 版本下都只接受 Nx3，
strided 只能经 `prep_xfer_dlist`（§3.6.3）。实现时不要误判为功能缺失。

---

### A3 — post/wait 拆分（设计项 5b）★ 阶段 B 的前置

**落点**：`psrl/utils/nixl/client.py:1519-1544`（`merge_and_finish_cached_xfer`），
`client.py:1572`（`wait` 的轮询循环）

**做什么**：
1. 把 `merge_and_finish_cached_xfer` 拆成两个方法：
   - `post_cached_xfer() -> list[handle]`：只提交，立即返回 handle
   - `wait_xfer(handles, timeout)`：等待这批 handle
2. 完成检测从 `time.sleep(0.001)` 轮询改为 `get_new_notifs`：
   发起方在 `make_prepped_xfer` 时带 notif 标签，完成时收到通知
3. 保留轮询作为 notif 不可用时的回退，但**默认走 notif**

**判据**：
- 存在一个测试：连续 post 两个 chunk，断言**第二个 post 的返回时刻
  早于第一个 wait 的返回时刻**。这是双 buffer 成立的充要条件
- 单轮完成检测延迟从「最多 1ms」降到 RTT 量级（< 50μs）
- chunk 下界从 256MB 降到 RTT 决定（§3.6.4 的表）

**为什么这是前置**：现有实现提交后立刻进轮询循环、不返回控制权，
`post(i) → wait(i) → post(i+1)` 严格串行，**「深度 2 双 buffer」无处落脚**。
先做 B2 会得到一个「看起来有流水、实测没有重叠」的实现。

**验证**：L1（post/wait 交错顺序断言）+ L2（端到端重叠率）

---

### A4 — `make_xfer_tag` 去 pickle（设计项 6c 的一部分）✅ 已落地

**落点**：`psrl/utils/nixl/client.py:42`

**做什么**：`pickle.dumps` 换成 UTF-8 NUL 分隔编码，单 shard 走 f-string 快路径

**判据**：140 万 key 规模下 tag 构造 < 40ms

**实际结果**：**55ms → 20ms（140k descriptor）**。
注意**主文档原来的估计是错的**：它暗示 pickle 开销是「秒级」，
实测 pickle 单次约 0.4μs，140k 次共 55ms。已回改主文档。
**收益真实但有限**——这个规模下主导开销是 NIXL 自己的 descriptor 处理。

两处实现细节值得记：
1. `target_client` 在 `wait()` 里是 `str | None`（`client.py:1553`），
   所以 `None` 可达，必须用哨兵值与空字符串区分，否则两个不同传输会碰撞
2. `(1,2)` 与 `(12,)` 必须不碰撞——无分隔符的朴素拼接会碰撞

**验证**：`tests/nixl/test_xfer_tag.py`（9 个测试全过，含单射性与 MoE 规模性能）

---

### A5 — 配置期防御五项（设计项 9d-a）

**落点**：`psrl/utils/config.py`、`psrl/trainer/ppo/ray_trainer.py:1962`、
`psrl/utils/converter/fsdp_converter.py`（`get_sharding_for_param`）

**做什么**：

| # | 改动 | 落点 |
|---|---|---|
| a | `train_model_dtype` 真实读 `engine_config.model_dtype`，不再按后端硬分流 | `ray_trainer.py:1962` |
| b | `param_offload` **与** `offload_policy` 两个 flag 分别校验并拒绝 | `config.py` |
| c | FSDP2 uneven sharding 在**声明期**断言并指向调小 `fsdp_size` | `fsdp_converter.py` 的 `get_sharding_for_param` |
| d | 握手期打印全局 finest mesh 各维大小 | `psrl/utils/nixl/server.py` 的 `make_unified_sharding` |
| e | **LoRA `merge=false` 拒绝**（§3.11） | `config.py` |

**判据**：五条各有一个故意违反的测试，断言**在配置校验期报错**而非传输期。
特别是 (a)：用 `model_dtype=bf16` 启动，断言注册给 NIXL 的字节 dtype 是 bf16 而非 fp32；
(e)：`lora_rank>0 and not lora.merge` 必须 raise，错误信息包含 `merge=true` 这条替代路径。

**为什么全是「配置期」**：这五条不做的话失败形态分别是
「传输期报错而非校验期」「按错 dtype 准备缓冲」「LoRA 下静默同步基座模型」，
**都是难归因的失败**，而 (e) 甚至完全静默（reward 不涨、无报错）。

**验证**：L1（五个 `pytest.raises`）

---

### A6 — 删除 FSDP1 分支（设计项 9e）

**落点**：`psrl/utils/converter/fsdp_converter.py`

**做什么**：删除 `fsdp_strategy == "fsdp"` 的分支与 `flat_param` 相关代码路径

**判据**：`grep -n "flat_param\|StateDictType.SHARDED_STATE_DICT" fsdp_converter.py` 零命中；
`convert_fsdp_inplace` 的 `fsdp_strategy` 参数若保留则只接受 `fsdp2` / `fsdp_hybrid`

**验证**：L1（已有 FSDP2 测试仍通过）

---

### A7 — join 三级判定（设计项 9c-a）★ 越早越好

**落点**：`psrl/utils/nixl/server.py`（`make_unified_sharding` 前插入准入判定）

**做什么**：

```python
def classify_join(current_finest, new_mesh, all_tensor_shapes):
    new_finest = {d: lcm(current_finest.get(d, 1), new_mesh.get(d, 1)) ...}
    if new_finest == current_finest:
        return "FAST_PATH"            # 整除，既有 descriptor 全部有效
    for key, shape in all_tensor_shapes.items():
        for dim, count in new_finest.items():
            if dim < len(shape) and shape[dim] % count != 0:
                return "REJECT"       # 不可表达，放行会让所有既有 client 抛 ValueError
    return "WARN_RESHARD"             # 可表达但要全局重分片
```

**判据**：三级各有一个测试
- `FAST_PATH`：hidden=4096、current=lcm(8,4)=8、新成员 TP=16 → finest=16，
  断言既有 client 的 `shard_mesh` 与 `shard_indices` **逐字段不变**
- `WARN_RESHARD`：hidden=1536、新成员 TP=3 → finest=24，1536%24==0 →
  断言告警已打印**且加入成功**，重分片后逐张量内容正确
- `REJECT`：hidden=4096、新成员 TP=3 → finest=24，4096%24!=0 →
  断言准入期被拒绝，**且既有 client 仍处于可服务状态**

**为什么第三级是正确性**：`get_local_sharded_tensors`（`nixl_spec.py:166`）
要求 `shape[dim] % count == 0`，不整除会让**所有既有 client** 在重分片时抛
`ValueError`——一次扩容变成全链路故障，而不是「变慢」。

**验证**：L1（三级判定的纯函数测试，不需要真实 NIXL）

---

### A8 — colocate CUDA IPC + stage offload（设计项 2）

**落点**：`psrl/workers/train/engine_train_worker.py`（`switch_to_*_mode`），
`psrl/workers/gen/vllm_async_server.py`（KV cache 释放与恢复）

**做什么**：
1. colocate 路径走 CUDA IPC（`reduce_tensor` / `rebuild_ipc`）pack 进**一块可复用
   uint8 buffer**，**完全不碰 NIXL 注册**
2. 切换顺序严格按 verl 的 `engine_workers.py:811-826`：
   `pause_generation` → `release_kv_cache_memory()` → transfer →
   **train 权重 offload 到 CPU** → `wake_up(["kv_cache"])`
3. 稳态切回 trainer 从 **CPU reload** 而非读 checkpoint（§3.3.1a 末段），
   判据是独立的一次性标记「trainer 本进程是否曾持有过真实权重」，
   **不能用 `is_rollout_mode_in_actor`**（它被两个 switch 函数来回翻转）

**判据**：
- colocate 路径上 `register_memory` 调用次数 **为 0**
- eval 期间 KV cache 可用量与「纯 vLLM sleep/wake」基线相当（差异 < 5%）
- 整个 transfer 期只存在**一块** IPC buffer，大小等于配置值、不随模型或张量数增长
- 连续两轮 eval，第二轮切回 trainer 后权重版本 **等于 eval 前的版本**
  （不是上次 save 的版本）——这条测「稳态切回不读盘」

**验证**：L2（新增 colocate 模式）+ L1（一次性标记的状态机）

---

### A9 — 源头 FP8 wire（设计项 4）

**落点**：新建 `psrl/utils/fp8/`，调用 vLLM 与 TE 的现成原语

**做什么**：按 §3.8 的依赖表组合 vLLM / TE 原语，不自己实现量化算法：
- blockwise：`vllm/utils/deep_gemm.py::per_block_cast_to_fp8`
- per-tensor/group：`_custom_ops.py::scaled_fp8_quant`、`fp8_utils.py::per_token_group_quant_fp8`
- 落地格式化：`fp8_utils.py::{process_fp8_weight_block_strategy, requant_weight_ue8m0_inplace}`
- 合法性校验：`fp8_utils.py::{validate_fp8_block_shape, validate_fp8_block_shape_moe}`

**判据**：
- 小模型上逐张量比对「本设计产出」与
  「vLLM 从原生 fp8 checkpoint 加载 + `process_weights_after_loading`」，
  在 UE8M0 / requant 容差内一致（§7 验证项 8）
- 传输字节相对 bf16 **减半**
- `weight` 与其 `scale` 在同一个 commit group（原子性）

**验证**：L1 量化器逐位；L2 精度矩阵 fp8 行；生产 e2e 覆盖两条 FP8 训练路径
（`qwen3_8b_megatron_direct_fp8gen.sh`：bf16 训练 + FP8 rollout；
`qwen3_8b_megatron_direct_fp8gen.sh TRAIN_FP8=true`：FP8 hybrid 训练 + FP8 rollout），均 10 步 0 错误
且最终验证与 bf16 训练臂一致（FINDINGS §12）。

---

## 3. 阶段 B：执行层

### B1 — chunk / commit group 分离（设计项 6d）★ C1 的前置

**落点**：`psrl/utils/nixl/comm_plan.py`、`client.py`

**做什么**：把现在一个 `bucket` 承担的六个角色拆成两个概念（§3.6.1）：

| 概念 | 单位 | 服务于 | 可否切开一个 key |
|---|---|---|---|
| `chunk` | 固定字节数（自适应） | 流水、带宽、W、wire buffer 复用、holder 位图 | **可以** |
| `commit_group` | 引擎 loader 的原子要求 | 版本可见性、落地格式化 | **不可以** |

两者多对多：一个 chunk 含多个小 group，一个大 group（MoE fused `w13_weight`）跨多个 chunk。

**判据**：
- 存在一个测试：一个 11GB 的单 key 被切成多个 chunk 传输，
  且该 key 的 commit group 在**全部 chunk 到齐后**才标记可见
- 故意只送达 group 的一半，必须**报错**而不是静默 no-op（§7 验证项 7）
- 调整 chunk 大小不影响 commit group 边界（两个概念真的解耦了）

**验证**：L1（`test_comm_planner.py` 扩展）+ L2

---

### B2 — 三类重叠 + slot ack bitmap（设计项 5）

**落点**：`psrl/utils/nixl/client.py`（流水调度），`psrl/workers/ps/ps_manager.py`（ack 上报）

**做什么**：

1. **三类重叠**：cast / RDMA / 落地格式化按 commit group 流水，
   第 i 组的格式化与第 i+1 组的 RDMA 重叠
2. **slot ack bitmap**（**正确性，不是优化**，§3.4.2 末段）：

```
每个 wire buffer slot 持有:
    (chunk_id, version, expected_readers: Bitmap, acked: Bitmap)

读者完成一个 chunk -> 在它本来就要发的完成通知里捎带 chunk_id
trainer 收到        -> acked.set(reader_id)
acked == expected   -> slot 可复用
holder 摘除 reader  -> expected.clear(reader_id)   ← 崩溃自动收敛
超时未填满          -> 对未 ack 的读者返回 CHUNK_STALE 让其重拉
```

**判据**：
- wire buffer 常驻量 **== `2 × chunk`**，且**不随 W 增长**
- 存在一个测试：人为延迟一个读者的 ack，断言
  `conflict_wait_budget_ms` 到期后**对它返回 `CHUNK_STALE`，而不是让整条流水停住**
- 端到端时间 ≈ `max(T_rdma, T_fmt) + T_fmt/N_rounds`（流水生效）
  而非 `T_rdma + T_fmt`（完全没重叠）
- **expected 集合由 plan 派生**，不是运行期计数器：
  断言 ack 通道不引入新的控制面往返（复用 §3.2b 的完成通知）

**ack bitmap 已被常驻发布副本取代**（§9 B2）。循环复用的 slot 必须等最慢的读者归还，
而 pull 语义下 W 个独立 instance 的进度差没有上界。副本活一个版本、在推送锁内刷新，
读者在锁外随时读，不需要归还信号。

**验证**：L2（重叠率、slot 等待分布）+ L3（慢读者注入）

---

### B3 — chunk 自适应 + 轮级决策本地化（设计项 6c 剩余部分）

**落点**：`psrl/utils/nixl/comm_plan.py`（chunk 大小计算），`client.py`（轮级决策）

**做什么**：
1. chunk 大小按 §3.6.5 的 cost model 计算，**不写死常量**：
   ```
   N_min     = ceil(max(T_fmt / T_rdma, 1) × D)        D = 2
   chunk_max = M_v / N_min
   chunk_min = 由完成检测粒度定（A3 之后是 RTT 级，A3 之前是 256MB）
   取 chunk = clamp(512MB, chunk_min, chunk_max)
   ```
2. **轮级决策本地化**：源选择在拉取开始时一次性解析完整轮次，不每轮查 index

**判据**：
- 换一个大一倍的模型，chunk 数变化而 chunk **字节数**按公式变化（不是固定值）
- 每轮传输**零跨进程往返**（trace 中无 index RPC）
- cost model 的三个输入（`T_fmt`、`T_rdma`、`BW_NIC`）可从配置或探测得到，
  不需要人工填

**验证**：L1（cost model 纯函数）+ L2（轮级 RPC 计数为 0）

---

## 4. 阶段 C：元数据与控制面

### C1 — canonical MoE 分组（设计项 6）★ 大 MoE 硬阻塞

**落点**：`psrl/utils/converter/model_mappings.py`、`vllm_converter.py`（`FUSED_MOE_W13_SPLIT` 等）

**做什么**：
1. `experts.{i}.gate_proj.weight` 合并为 `experts.gate_proj.weight` **一个 key**，
   expert 维度表达为 `NIXLSharding` 的一个分片维度
2. 删除主动拆 fused 张量的逻辑（`vllm_converter.py:400` 起）——
   vLLM 的 fused loader 本来就要 3D 的 `w13_weight`，拆开反而是额外工作
3. `ETP != 1` 时 shard mesh 变成 `{0: EP, 1: ETP}` 两维，
   **必须验证 `find_finest_shard_mesh` 的 lcm 在两维上正确**

**判据**：
- Kimi-K2 量级模型的 key 数从 **14 万降到数百**
- 单 client `client_info` 的 pickle 体积从 **56MB 降到数百 KB**
- `T_fmt` 从 846ms 降到 **~18.5ms**（§3.6.5 cost model 的预测，实测校准）
- MoE 的 `shard_dim=0`（切 expert）与 `shard_dim=1`（切 projection）
  两条路径分别端到端跑通

**为什么是硬阻塞**：不做这一项，1TB 场景连握手都过不去
（数十 GB 的 pickle 流量 + 等量反序列化 CPU）。
而且 §3.6.0 的 cost model 表明：**分组前 `T_fmt` 主导，overlap 做了也看不到收益**。

**验证**：L1（两维 lcm）+ L2（MoE 模型端到端）+ L4（1TB 规模）

---

### C2 — 控制面下热路径（设计项 6b）★ 阶段 D 的前置

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/grpc/ps_manager_service.py`

**做什么**（按「消除 > 降频 > 加速」排序，§3.2.1）：

| 措施 | 做什么 |
|---|---|
| a 消除 | plan 握手期编译、rollout **本地缓存**，稳态拉取**完全不查 PS**；只有 holder 集合变化才由 PS 主动失效通知 |
| b 降频 | 租约按 `(instance, version)` 聚合，不按 chunk。消息量 `O(instances × chunks)` → `O(instances)` |
| c 降频 | 版本发布用推送而非轮询（现有 `set_ps_model_version` 已是此形态，保持） |
| d 加速 | 准入与租约两个高频读走 gRPC + 内存快照，**不进 `@_state_locked` 临界区**；写路径仍走 Ray actor 保持强一致 |

**判据**：
- 稳态同步期间，rollout → PS 的 RPC 次数 **为 0**（plan 已本地缓存）
- 控制面消息数**不随 replica 数平方增长**（§7 验证项 3）
- `@_state_locked` 只保护写路径

**验证**：L2（RPC 计数）+ L4（64 instance 规模下的控制面消息数）

---

### C3 — 版本发布挂 `on_train_step_end`（设计项 7b）

**落点**：`psrl/workers/train/engine_train_worker.py`

**做什么**：把版本发布挂在 Megatron-Bridge 的 `on_train_step_end` 回调上
（已查证可用：`callbacks.py` 的 `VALID_EVENTS`、`train.py:518`）

**判据**：发布时刻落在「权重刚就位且 NIC 刚空闲」——
trace 中发布事件紧跟 param all-gather 完成，且不与 backward 的 NCCL 重叠

**注意**：这是**一个回调**，不是阶段上报协议（后者已移出本期，§3.5.2.2）

**验证**：L2（发布时刻的 trace）

---

## 5. 阶段 D：trainer 直供（不可分割块）

> **这一块一旦开始必须完成**，因为它删除 PS 数据面。
> 开始之前，阶段 A/B/C 必须已稳定运行——这样出问题可以二分定位到 D 内部。

### D1 — 三阶段去冗余规划（设计项 7）

**落点**：`psrl/utils/nixl/comm_plan.py`（新 planner）

**做什么**：
1. 三阶段 plan：receive / external-exchange / internal-exchange（吸收自 WeightBridge）
2. 选源优先级（§3.5.2）：
   ```
   候选集 = {trainer} ∪ {version(peer) == v 的 holder}
     1. 同节点 peer（NVLINK/PCIE）    ← 第一优先，不占 NIC
     2. trainer（IB）                 ← 在飞量 < W 时
     3. 跨节点 peer（IB）             ← 仅当 trainer 在飞量 >= W   [阶段二]
     4. 皆不可用                      ← 排队，等前面完成后重新解析
   ```
3. **硬约束：只能从同版本 peer 读**。holder 位图只为当前版本 v 维护一份，
   版本推进即整体清空
4. W 按 §3.6.5 的 cost model 给初始值，再用实测反馈收敛：
   `W_init = min(NIC_count, nodes_per_instance)`
5. **禁止链式 peer**（单跳，§3.5.4）

**判据**：
- trainer 出口字节 **== `M`**（不是 `N×M`）
- 每 rollout 节点入口 **== `M_v = M/num_nodes`**
- 与 `T_min` 的比值可报告（WeightBridge 在 16~64 节点报 `1.27×`~`2.12×`，可作参照）
- 固定 trainer、逐步增加 rollout replica，AGST 应为**亚线性**
- 断言**不存在 A 读 B 而 B 读 C** 的链式结构

**验证**：L2（流量计量）+ L3（peer 扩散）+ L4（亚线性扩展）

---

### D2 — 版本可见性 + 运行时断言 + holder role 维度（设计项 8）

**落点**：`psrl/workers/ps/ps_manager.py`（holder 表）

**做什么**：
1. holder 表加 **role 维度**：`(role, version, key, shard) -> holder`
   —— **正确性前提**，否则 actor 的 v3 与 critic 的 c3 互相覆盖（§3.10.3）
2. **四条运行时断言**（§3.4.6），必须是断言不是文档要求：

| 不变量 | 检查点 | 违反后果 |
|---|---|---|
| holder 即注册有效 | pull 准入时校验 holder 身份 | RDMA 读到已释放页，静默损坏 |
| sleep 前在飞读者归零 | colocate 切换路径 | 同上 |
| commit group 完整性 | 标记版本可见前 | 半更新模型 |
| NCCL 通信组存活 | 角色切换前后 `assert torch.distributed.is_initialized()` | 切换成本涨一个量级且不报错（防回归，§3.10.4b） |

3. 在飞读者计数：**per-chunk 整数计数器**，不是「版本×chunk 租约表」

**判据**：四条断言各有一个故意违反的测试，确认**报错而非静默通过**；
role 维度：actor 与 critic 同时发布版本（v3 / c3），断言解析到不同 holder

**验证**：L1（断言）+ L2（role 隔离）

---

### D3 — trainer 直供 + 删除 PS worker（设计项 9）

**落点**：`psrl/workers/ps/ps_storage_worker.py`（**删除**）、
`ps_worker_group.py`（**删除** `create_ps_worker_group`）、
`ps_manager.py`（只留索引功能）、
`psrl/workers/train/engine_train_worker.py`（删除 `ray_push_model` 等 push 路径）

**做什么**：
1. 删除 `PSStorageWorker` 与 `create_ps_worker_group`
2. 删除 push 路径（`ray_push_model` / `push_model_state_dict_cpu` / `cpu_ref` 等）
3. rollout 的 pull target 从 `ps_for_pull` 换成 train client
4. 删除 `ps_mode` 枚举（四个值里 `nixl_gpu` 至今 `NotImplementedError`，
   其余三个都在描述「副本放哪」，而本设计没有副本）
5. ~~`ParamSyncPlan` 的 push 成对语义改造~~ —— **已完成**。
   `ResidentZeroCenteredGammaSync`（`param_sync.py`）把暴露值搬进**独立张量**：
   `source_param` 对 trainer 保持 zero-centered，`state_dict[key]` 常驻 `+1`，
   `after_push` 刻意为空。converter 通过 `resident_exposure` 选择，
   该开关由 `not has_ps_data_plane(ps_mode)` 驱动。

   **为什么必须独立张量而非常驻 `+1` 原张量**：后者会让 trainer 自己的 forward
   读到错的 gamma。代价是每个 norm 层一份 gamma，相对权重本身可忽略，
   且注册地址因此在整个 run 内保持有效。

   **为什么这是静默错误而非显式失败**：`after_push` 还原后，rollout 的 pull
   **照常成功**，只是返回 `gamma-1`。`tests/nixl/test_resident_exposure.py`
   第一个用例就把旧 action 的这个性质钉死，再证明新 action 不具备它。

   `DTypeCastSync` 与 `ConcatenatedQKVSync` 本来就写入独立张量、`after_push` 为空，
   无需改造。FSDP2 路径构造空 `ParamSyncPlan()`，同样无负担。

**判据**：
- 稳态下**无 PS worker 进程**，host 侧**无权重常驻**
- CPU 侧权重副本 `N×M → **0**`
- cast 显存是每 rank 自己 shard 的 wire 字节（常驻发布副本），合计一份 wire 格式的模型。
  **判据是「加 rollout 数，这个数字不变」**
- 布局转换产物的常驻量**单独记录**（它确实随模型增长，约 `0.15×M`），
  两者不要混在一个指标里

**验证**：L2（无 PS 进程、显存上界）+ L4（1TB 规模）

---

### D4 — holder 对称化 + bootstrap（设计项 9b）

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/workers/ps/broadcast.py`

**做什么**（§3.3.1a）：
1. holder 类型对称化：`TRAIN_GPU` / `ROLLOUT_GPU` / `SELF`，
   **trainer 不再被假定为唯一权威源**
2. `is_rollout_mode_in_actor=True` 启动：
   ```
   rollout 每节点一个代表从 checkpoint 读自己需要的 shard → 注册为 holder，版本 = v0
   其余 rollout 走同版本 peer 扩散拿到 v0（复用 broadcast.py 的树形拓扑，
                                          源头从 PS rank 0 换成 rollout 代表）
   trainer 首次被唤醒 → 从 checkpoint 加载训练起点，不从 rollout pull
   ```
3. **首次 vs 稳态的判据是独立的一次性标记**，不能用 `is_rollout_mode_in_actor`

**判据**：
- 启动时全 GPU 跑 rollout 能拿到 v0
- **首次**切回 trainer 从 checkpoint 拿到训练起点
- **稳态**切回从 CPU reload，权重版本等于 eval 前的版本（**不是上次 save 的版本**）
  —— 读盘会静默回退训练进度且不报错

**为什么首次要读盘**：cast 路径下 rollout 持有的是 fp8/fp4 权重，
**不能作为 bf16 训练的起点**（反量化有损，且 optimizer master params 需要 fp32）。

**验证**：L2（三段式全流程 + 稳态单测）

---

### D5 — chunk 起始偏移错开（设计项 7c）

**落点**：`psrl/utils/nixl/comm_plan.py`

**做什么**：instance `i` 从 `chunk (i × num_chunks / R) mod num_chunks` 开始，环形推进

**判据**：
- 偏移**与 commit group 边界对齐**（不能把一个 group 切在环形起点）
- 偏移是**确定性的**（由 instance 序号导出），不是随机——
  这样 index 可以在不通信的情况下预测谁先持有哪些 chunk
- 开启前后对比第一波的 **peer 命中率**提升
- 只在 `R > W` 时启用

**验证**：L3（源分布统计）

---

### D6 — `CHUNK_UNAVAILABLE` 真能继续生成（设计项 8d）★ 与 D7 同批

**落点**：`psrl/workers/gen/vllm_async_server.py:1036`、
`psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py:167`

**做什么**：
1. `pull_model_for_sync` 的版本 assert 改为**返回实际版本 + 标记本轮未达标**：
   ```python
   # 现状（会抛，经 _finish_gateway_sync 的 except 走到 fail_sync + /workers/pause）
   assert curr_model_version >= ps_version, "Pulled model version is stale: ..."
   # 改为：返回 (actual_version, reached_target: bool)
   ```
2. 版本发布从**乐观预发**改为 **pull 成功后发布**：
   现状是先 `_publish_weight_version_updates(ps_model_version)` 再 pull，
   失败时靠事后更正——留下「SMG 认为是 v+1、实际是 v」的秒级窗口

**判据**（§7 验证项 9）：
- (a) 该 replica 在降级后**仍在 SMG 的可用集合里**且继续服务请求——
  不是 `fail_sync` + `/workers/pause` 之后的「不可用」
- (b) 降级全程 SMG 记录的该 instance 版本**始终等于它实际持有的版本**，
  不存在「声明 v+1 而实际 v」的中间窗口。
  **必须在请求粒度上验**：降级窗口内到达的请求，其记账版本要与
  生成时实际使用的权重版本一致，否则 staleness 账本被污染且无人发现

**为什么必须与 D7 同批**：不改这一项，D7 的 DP 接管落地当天就撞 assert，
降级路径表现为「replica 被摘掉」，与「保持当前版本继续生成」正好相反。

**验证**：L3（故障注入 + 请求粒度的版本记账）

---

### D7 — DP 副本接管（设计项 8b）

**落点**：`psrl/workers/ps/ps_manager.py`、
`psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py:74`

**做什么**（三层响应，§3.5.7）：
```
① 单 rank 故障（Megatron/HSDP；纯 FSDP2 下不可用）
   index 摘除该 rank → 同 DP 组其他 rank 接管（内容相同）→ 受影响 pull 重新解析源
② 整个 DP 组故障
   返回 CHUNK_UNAVAILABLE（显式返回值，不是超时）→ rollout 保持当前版本继续生成
③ trainer 全体重启
   holder 表整体清空，版本号保留 → 所有 pull 立即 CHUNK_UNAVAILABLE
```
**holder 摘除要能被动触发**：rollout 的 `client_read` 报错时，
连同失败的 holder 身份一起上报 index——把故障检测搭在已有错误路径上，不需要心跳。

**判据**：
- 杀掉一个 trainer rank，pull 由同 DP 组接管且 rollout **无感知**
- 杀掉整个 DP 组，返回 `CHUNK_UNAVAILABLE` 且 rollout 保持当前版本继续生成，
  **不卡死、不读到半更新模型**
- 复用 `_quarantine_failed_replicas` 路径，不新建一套

**验证**：L3（故障注入）

---

### D8 — FSDP2 副本数运行时推导（设计项 9d-b）★ 与 D7 同批

**落点**：`psrl/workers/ps/ps_manager.py`（副本数推导）

**做什么**：副本数必须由 index 从 `NIXLSharding` **运行时推导**，
不能假定「DP 副本一定存在」。`replica_count == 1` 时 D7 的 ① 直接跳到 ②。

**判据**：纯 FSDP2 下杀一个 rank，必须返回 `CHUNK_UNAVAILABLE`
而**不是**尝试 DP 接管；HSDP 下同一测试必须走 DP 接管

**为什么与 D7 同批**：先写死 Megatron 假设再回头改会返工。

**验证**：L3（两种后端的同一故障场景）

---

## 6. 阶段 E：正确性补强

### E1 — 版本级等待 + per-chunk 写屏障（设计项 8c）

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/workers/train/engine_train_worker.py`

**做什么**：

```
版本级等待（FSDP2 bf16 同精度下【必需】，Megatron 同精度下是优化）：
  发布 v+1 之前，等待对 v 的在飞读者计数归零（一次性，不分 chunk）
    上限 conflict_wait_budget_ms，建议默认为单步时长的 5%
    超限则发布 v+1 并把 v 标记为 INVALID
      在飞读者收到 INVALID → 放弃本轮，重拉 v+1（更新的权重，算法上是改善）

per-chunk 写屏障（始终是优化）：
  仅 Megatron 同精度路径可用——master params 让冲突窗口窄到「param 回写那一瞬」
  FSDP2 bf16 下整个 optimizer step 都在写，降粒度没有收益
```

**判据**：
- **`FSDP2 + bf16 + bf16 + disaggregate` 是必跑项**：
  关掉版本级等待后**必须能观测到权重被破坏**（读到半新半旧）——
  这是该机制存在价值的判据
- 超时后被打断的 rollout **重拉更新的版本**，而不是「继续用旧的」
- cast 路径的「等 optimizer」等待恒为零

**启用 FSDP2 + `model_dtype=bf16` + bf16 rollout 之前必须完成这一项。**

**验证**：L2（精度矩阵的该格 + 关闭机制后的破坏观测）

---

### E2 — 非整除 join 的重分片走版本边界（设计项 9c-a2）

**落点**：`psrl/utils/nixl/server.py`、`ps_manager.py`

**做什么**：A7 判定为 `WARN_RESHARD` 时，重分片必须在版本边界进行：
```
撤销 holder → 等在飞读者归零 → 全体按新 finest mesh 重切重注册 → UPDATE_INFO 广播 → 重编 plan
```
「同一地址的不同切分」对远端同样是陈旧描述，所以每个既有成员都必须重新加载对端 info。
单边 RDMA 下源端无处校验 epoch，失效由驱逐 + 重广播保证（主文档 §3.7.3.0a 更正）。

**判据**：reshard 前后所有成员坐标判据全过；reshard 期间没有读者读到混合切法；
收到 `UPDATE_INFO` 前用旧 descriptor 发起的读失败而非读到错误字节

**验证**：L3（两段校验）

---

### E3 — 握手改事件驱动 join / leave（设计项 9c-b）

**落点**：`psrl/utils/nixl/server.py:56,90`、`psrl/workers/ps/ps_manager.py:907`

**做什么**：`nixl_protocol()` 的八步直线改为事件驱动循环。
现状 `_is_all_client_shardings_recved` / `_is_all_client_infos_recved`
一旦置真就直接 return，**新 instance 没有入口**。

**判据**：运行中的集群能接受新 instance 加入（走 A7 的三级判定），
且既有 instance 不中断服务

**验证**：L3（运行中扩容）

---

### E4 — partial MD 两段式 + 缩容吊销 descriptor（设计项 9c-c）

**落点**：`psrl/utils/nixl/client.py`（握手与退出路径）

**做什么**：
1. **join 分两段**：
   ```
   第一段（所有人）：conn-info-only blob，401 B
   第二段（只有真正的对端）：按 plan 解析出的那部分 descriptor
                            用 get_partial_agent_metadata 而非全量
   ```
2. **缩容四步**：
   ```
   撤销 holder 资格 → 等在飞读者归零 → 吊销 descriptor（带确认） → sleep/释放显存
   ```
   第三步不走 NIXL 的 `invalidateLocalMD` socket，而是走 PSRL 自己的 notif 通道，
   原因见 `leave_protocol` 模块头：`invalidateLocalMD` 的实现就是
   「让对端执行一次 `remove_remote_agent(我)`」，它**无法广播、需要对端监听端口、
   且 fire-and-forget 无法确认**。而第四步（释放显存）的安全性恰恰取决于
   「所有持有者都已忘记」这个确认。

**判据**：
- 每次 join 时每个既有成员加载的 MD 字节数为 **conn-info 量级（~400B）**
  而非全量（~406B × region 数）
- 让远端用成员退出前的 descriptor 发起读，**必须失败而非读到垃圾**
- 任一持有者未确认吊销时，**必须拒绝释放显存**（而不是超时后照常释放）

**实测依据**：全量 MD 约 406 B/region 线性增长（4096 区 = 1.6MB）；
partial（4096 区中发布 1 个）= 883 B；conn-info-only = 401 B，**压缩 1884×**。

**验证**：L3（join 扇出字节数 + 退出后的陈旧 descriptor）

---

## 7. 阶段 F：正交可选

### F1 — delta 传输（设计项 10）

**落点**：新增，与主路径正交

**做什么**：只传变化的 `(position, value)`，参考 verl 的 `delta_sharded`

**判据**：跨 DC 或慢链路场景下传输字节降到 1%~3%

**注意（此处需更正，更正后 F1 反而更可行）**：原文称 diff base 是「额外 CPU 权重存储」，
因此只能作约束 (b) 的例外。核实 verl 实现后发现**不是这个形态**：
每个 rank 只 pin **自己那一份分片**并在本地做 diff
（`delta_checkpoint_engine.py` 原话 "no rank ever holds a full-model snapshot"），
所以代价是 `M / ranks`、**随集群增大而摊薄**——
正是需求里允许的「CPU 部分也分布式存储」，而非它禁止的集中式副本。

**决策已可计算**：`psrl/utils/nixl/delta_plan.py`，`test_delta_plan.py` 13 passed。

**两处实测/算术更正**：
1. **index 项决定盈亏平衡点**。每个变化元素要带位置，bf16 + 4 字节索引下
   一个变化元素是重发的 3 倍，**约 1/3 变化率就不划算**，不是直觉的 100%。
2. **量化 wire 让盈亏平衡点更早到**（而非更晚）：value 减半而 index 不变，
   所以 fp8 下变化元素相对更贵。
3. **「链路快所以不划算」在本集群不成立**：我原本预期实测 103 GB/s 会拒绝，
   算下来 1GB 分片 diff 2.7ms、省 9.8ms；同节点 NVLink 343 GB/s 仍省 2.9ms。
   真正的分界在**链路快过 diff 带宽**之后。
   **所以让 F1 保持 opt-in 的是「变化率」而非链路** —— 训练早期几乎每个元素都在动，
   而这一项无法预估，只能按运行观测。

**验证**：L1（决策函数，已完成）+ L4（跨 DC，待）

---

## 8. 测试改造（§7.0）

### 8.1 三处必须先做的结构性改动

**(a) 正确性判据从「全 1」升级为坐标函数 ★ 第一优先级**

现有 `assert_state_dict_all_ones`（`test_nixl_e2e.py:95`）**对本设计要验证的东西完全无效**：

| 错误类型 | 全 1 判据能否检出 |
|---|---|
| 数据没到 | 能 |
| **布局错误**（转置、reshape 错） | **不能**——全 1 转置后还是全 1 |
| **分片错位**（读到别的 rank 的 shard） | **不能**——全 1 的任意切片都是全 1 |
| **精度转换错误** | **不能**——全 1 cast 到 fp8 还是 1.0 |
| **MoE expert 映射错误** | **不能** |

替换为**每个元素等于其全局坐标的确定性函数**：
```python
def fill_by_global_coord(key, tensor, global_offset): ...
def assert_matches_global_coord(key, tensor, global_offset, dtype_tol): ...
    # 失败时报出第一个不匹配的坐标 —— 直接定位到是哪一维错位
```
**两个约束**：值域必须落在目标 dtype 可精确表示的范围内
（fp8 e4m3 只有 ~3.5 位尾数，**不能用大整数做哈希值**），
且 `f` 必须对坐标**单射**，否则错位换位检不出来。
实现上用**小整数的位拼接**（`layer_idx * 2^k + row * 2^j + col`），不是浮点哈希。

**(b) 删除 `create_ps_worker_group`（`:649`）**，`GenClientActor` 的 pull target
从 `ps_for_pull_name` 换成 train client。

**(c) 新增 colocate 模式**（现有测试只有 disaggregate）。
**这是改造里工作量最大的一项**，但它是 §3.7 全部结论的唯一验证手段。

### 8.2 四维参数化矩阵

| 维度 | 取值 | 现状 |
|---|---|---|
| 后端 | `megatron` / `fsdp2` | 已有（删 `fsdp1`，`fsdp_hybrid` 保留为 HSDP） |
| 精度 | `bf16+bf16` / `bf16+fp8` / `fp32+bf16` / `fp32+fp8` | 配置项已有，**cast 路径待实现** |
| 放置 | `disaggregate` / `colocate` | **需新增 colocate** |
| 并行度 | train TP/PP/EP × gen TP/EP（含不匹配） | 已有，只需扩参数化 |

**不做全笛卡尔积**，按风险选组合：
```
必跑（每次 CI）：
  megatron + bf16+fp8  + disaggregate + TP匹配      ← 推荐默认配置
  fsdp2    + fp32+bf16 + disaggregate + TP匹配      ← FSDP2 默认配置
  megatron + bf16+bf16 + colocate     + TP匹配      ← 时序互斥路径
每日跑：
  上述 × {TP 不匹配、EP 不匹配}                      ← shard 代数
  fsdp2 + bf16+bf16 + disaggregate                  ← 版本级等待必需的那一格
  megatron + fp8+fp8 + colocate                     ← FP8-native，显存最优
发版前：
  全部精度组合 × 2 放置 + MoE 模型（Qwen3-30B-A3B 级）
```

### 8.3 分层与覆盖

| 层 | 载体 | 覆盖验证项 | 条数 |
|---|---|---|---|
| **L1 单元** | `test_nixl_sharding.py` / `test_comm_planner.py` 扩展 + 新增 | shard 代数、finest mesh、FSDP2 uneven、MoE 两维 shard、A7 三级判定、cost model | — |
| **L2 e2e 单节点** | `test_nixl_e2e.py` 改造 | 7、8、11、14、15、16、17、17b、19、22、23 | **11** |
| **L3 e2e 多节点** | 新建 | 5、13、15b、18、20、21、24 | **7** |
| **L4 规模实测** | 人工，无法自动化 | 1–4、6、9、10、12、25 | **9** |

**L4 的 9 条是本方案的真实风险**——它们恰恰支撑着「万卡 / 1TB」的声明，
而只能在真实集群上验证。**这一点必须在落地计划里显式承认**，
不能把「设计上可扩展」与「已验证可扩展」混为一谈。

---

## 9. 完成度检查表

> **当前状态总览**
>
> 三列分开：**模块**（有实现 + 单测）、**生产接线**（被 `psrl/` 的生产路径 import 并调用，
> 以 `grep` 生产代码的 import 为准）、**验证层级**（L1 单元 / L2 单节点 e2e / L3 多节点 / L4 规模）。
> 实测数据见 `weight_transport_FINDINGS.md`。
>
> | 项 | 模块 | 生产接线 | 验证层级与结论 |
> |---|---|---|---|
> | A1 prepped descriptor | ✅ | ✅ `pull_engine.py` 每源每次 pull 一个 prepped handle | L2 ✅ |
> | A2 strided descriptor | ✅ | ✅ `read_geometry.py` 两侧连续段配对合并 | L2 ✅ |
> | A3 post/wait 拆分 | ✅ | ✅ | L1 ✅ |
> | A4 xfer tag | ✅ | ✅ | L1 ✅ |
> | A5 配置防御 | ✅ | ✅ `config.py` | L1 ✅ |
> | A6 删 FSDP1 | ✅ | ✅ | L1 ✅ |
> | A7 join 三级判定 | ✅ | ✅ `server.py` | L1 ✅ |
> | A8 colocate IPC | ✅ `device_ipc.py` | ✅ 同设备的源改为设备拷贝 | L1 4 条跨进程；L2 colocate 三格（Megatron bf16 / FP8、FSDP2）逐位通过，16.4 / 9.4 / 2.1 GB 走拷贝 |
> | **A9 FP8 wire** | ✅ `block_fp8.py` | ✅ trainer 发布 block FP8 副本与 scale | L1 量化器与 vLLM 逐位一致，L2 17 格 FP8 位与 scale 逐位判定（含 Megatron FP8 train 4 格），单 GPU 真实权重贪心 token 100% 一致；生产 e2e `qwen3_8b_megatron_direct_fp8gen.sh` 10 步 0 错误，与 bf16 臂指标一致（FINDINGS §12） |
> | B1 chunk/commit group | — | — | **已删除**：拉取引擎按源一次提交，不再需要 chunk 计划 |
> | B2 slot ack | ✅ | ❌ 零 import | **不再需要**：cast 产物是常驻发布副本，在推送锁内刷新（主文档 §3.4.2） |
> | B3 chunk 自适应 | — | — | **已删除**，同 B1 |
> | C1 MoE 分组 | ✅ | ✅ | L2 ✅ |
> | C2 控制面 | ✅ 计量 | ⚠️ 仅 meter | L2 ✅ 稳态零 RPC（`test_the_pull_path_makes_no_control_plane_calls`） |
> | C3 发布边界 | ✅ | ✅ | L2 ✅ |
> | **D1 去冗余** | ✅ | ✅ `pull_engine.py`，默认开 | L3 ✅ 4 节点 16 rollout：出口 131 → 32.8 GB，稳态 0.178 s |
> | D1 选源 `source_resolution` | ✅ | ❌ 零 import | L1，选源由 planner 的 designate 次序承担 |
> | **D1 版本感知选源** | ✅ `ps_manager._nixl_version_holders` | ✅ `pull_engine.pull(version_holders=...)` + `vllm_async_server.nixl_pull_model` | L1 16 单元测试（`tests/unit/test_version_aware_source_selection.py`）；PSManager 维护 `(version, frozenset[trainer_names])`，每步一次 O(1) RPC，pull_engine 在搬字节前按集合过滤 upstream，`version_holders=None` 时旧行为保持不变 |
> | **D1 拓扑感知 designate** | ✅ `comm_plan._make_peer_groups` | ✅ planner 建立 peer group 时使用 | L1 15 单元测试（`tests/nixl/test_topology_aware_designate.py`）；sort key 从 `(upstream_load, name)` 改为 `(-trainer_link_priority, upstream_load, name)`，和 trainer 有 NVLink 的 instance 优先成为 designate，O(G log G) 每组（G 通常 4-8），总复杂度不变 |
> | **D1 _DONE barrier 删除** | ✅ | ✅ `pull_engine.py` | L1 60 tests pass；`_DONE` 信号不解锁任何操作（follower 在收到 `_DONE` 前已完成自己所有 peer 读，buffer 生命周期由外部调度控制）；`peer_timeout_s` 保留用于 `_READY` 等待 |
> | D2 `holder_index` | ✅ | ❌ 零 import | L1；多模型隔离改由客户端名承担（§11.1） |
> | D3 `nixl_direct` | ✅ | ✅ | L2 ✅ 31 格，L3 ✅，**PS 未删** |
> | **R1 reader-side FP8（Megatron）** | ✅ `client.py` + `pull_engine.py` | ✅ `psrl.nixl.reader_quantization=true` | L1 ✅ 6 条 reader-quant（1→2 / 2→2 / 4→2 / 2→4，两版本，exact-shard peer）+ 1 条 MoE 形状（3 维分组专家张量，同样覆盖 4 组 TP 组合，逐位+逐 scale 校验通过）+ 全量 CPU suite 1516 passed / 38 skipped（含本轮新增用例）；GPU 量化器与 vLLM 等价 46 条通过；**L2 两节点 8B 实测**（Megatron TP4×2DP → vLLM TP1×4，跨节点 UCX 设备列表一致后多次复现）：握手 4.63s，首次拉取每副本 16.38GB（= 活体 BF16 模型，非压缩副本），trainer 每卡 13.5-13.7GiB（无额外 publish copy 可见增量）；**与 publisher 方案的定量显存对比**（同拓扑跑 `qwen3_8b_megatron_direct_fp8gen.sh`，不加 `reader_quantization`）：publisher 在首次 sync 前 trainer 每卡 13.3-13.6GiB（与 reader-quant 的 BF16+optimizer 基线一致），sync 完成后稳定涨到 15.18-15.42GiB——**+1.8GiB/卡**，就是 reader-quant 不分配的那份 FP8 publish copy；拉取量也相应不同：publisher 每副本上行 9.44GB（已量化的 FP8 字节），reader-quant 是 16.38GB（未压缩的活体 BF16）——publisher 用更小的上行流量换一份常驻显存分配，reader-quant 反过来，上行流量更大但零常驻分配。这次复现同样先撞上跨机架 UCX 不可达（`29.162.247.148` 和 `29.162.224.39` 不在同一机架，`ping` 实测除 `bond1` 外的所有 bond 双向 100% 丢包，详见 `reader_quantization_impl.md` 的 "Network profile"），用 `UCX_NET_DEVICES=bond1,mlx5_bond_1:1` 收紧设备列表后才测出来，这个收紧会牺牲带宽，所以这里的拉取耗时数字不能和上面无收紧时测的数字直接比；**后续一次干净复现跑完整训练 step**：2 个 PPO step 完成（`actor/loss`、`critic/score`、`perf/mfu/actor` 0.59/0.78、response 长度上探到 8192 tokens），且 step 后的 resync 拉取在 4 个副本上全部观测到 peer diffusion 生效（每副本 upstream 3.48-4.71GB + peer 7.08GB，3 follower），与 CPU 侧 `test_reader_quantization_peer_diffusion_shares_training_reads_across_separate_instances` 的分组形状一致——**peer diffusion 问题在 transport 设计层面视为已解决**：规划算法本身 CPU 隔离测试和 GPU 端到端均验证正确。同一份代码/配置也会在另一些干净复现中卡在生成请求上（gateway `/routing_loop/status` 显示 `queue_len` 等于全部已派发请求、`routing:false`、`running_tasks:0`，vLLM 引擎输入队列全程为空）——**这是间歇性的，不是确定性的**：相同 build/config/topology 有时跑完、有时卡死；已排除 per-worker admission 门、`naive` vs `psrl` selector、migration 队列饿死路径三种可能（`mig.enable` 默认 `False` 且 PSRL 从不调用会注册 target 的 HTTP 接口）；另有一次复现观测到 worker 注册请求本身直接报 `ServerDisconnectedError`——与「请求被接受但从未派发」同属一类故障，指向 gateway 自身 worker 注册/首次派发路径上的窄竞争窗口，而非稳态逻辑错误。Root cause 定位需要 Rust 侧调试工具，本环境不具备，另案跟踪，不影响本 transport 设计——GPU 实测已验证到完整训练 step。**MoE GPU 验证门（30B-A3B EP/TP 组合）**：模型权重已确认可用（`/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-30B-A3B`），脚本已就位（`examples/grpo_trainer/qwen3_30b_moe_megatron_direct_reader_fp8.sh`），但尚未跑通：单机 8 GPU（train TP2·EP2 + gen TP4·EP4）在 trainer 自身的 Megatron 模块构建阶段（`_ParamAndGradBuffer`，PS/rollout pull 尚未开始）即 OOM，申请 54GiB 梯度缓冲区失败；在不加 `reader_quantization` 的原始 baseline 脚本上同拓扑复现同一 OOM，确认与本设计无关，是 Megatron-Bridge 在此模型/并行度组合下梯度缓冲区大小计算的已有问题。尝试 `+train_actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True`（参考 `third_party/verl` 235B Ascend 示例）在当前 Megatron-Bridge 版本上报 `GPTModelProvider has no attribute 'use_distributed_optimizer'`；尝试跨两节点 TP4·EP2 训练（摊薄单卡梯度缓冲区）又遇到与本设计无关的 `ray.init()` 挂起（两节点 `ray status` 显示集群健康、资源齐备，但 driver 卡在 `connect()` 超过 15 分钟不返回，与此前 `check_two_node_split.sh` 遇到的同类两节点挂起同源）。两个障碍均在 transport 设计之外（Megatron-Bridge 依赖问题 + 环境级 Ray 两节点连接问题），该验证门暂未跑通；MoE correctness 的最强证据仍是 CPU 侧 tiling 复现。**Phase 2（canonical chunk，跨 layout 共享）已实现并 CPU 验证通过**：核心发现是"逻辑 chunk"就是统一 sharding 后的单个最细粒度 shard（`_unify_role` 早已把每个 client 的 sharding 跑过 `find_finest_shard_mesh`/`refactor_based_on_finer_shard_mesh`，无需新地址方案），"有界缓存"就是 designate 自己常驻的 FP8 shard（无需新缓存结构）。改动：`comm_plan._make_peer_groups` 按单个 shard 分组（而非整个 shard 集合），为避免负载均衡把"同一 wanter 集合"的多个 shard 意外拆给两个不同 designate（白白多一次读、零收益），先按"同 key + 同 wanter 集合"重新合并成一个原子选举单元，再分组间独立选举；`pull_engine._assign` 按 shard 选举，`peer` 的形状从 `{designate:[key,...]}` 改成和 `upstream` 对称的 `{key:{source:shards}}`，让一个 key 的不同 shard 能分别来自不同 designate；`registration_policy.py` 的 `sources_by_reader`/`descriptor_readers_by_source` 同步改成逐 shard 处理。新增 CPU 单测（`test_different_target_layouts_share_an_individual_shard`）和集成测试（`test_reader_quantization_shares_across_different_gen_layouts_on_one_node`：train_tp=4，三组同节点 layout TP4/TP2/TP2，逐位+逐 scale 校验通过，每组都观测到 `peer_keys>0`，总上行字节数 < 3 份独立拷贝，验证了跨 layout 共享确实生效）；`tests/unit`（97 passed/1 skipped）与 CPU 标记的 `tests/nixl`（815 passed/7 skipped/641 deselected）全量回归通过。`_pick_holder`/`_repair_after_preempt`（抢占后故障恢复）保持原样，明确不在本阶段范围内。GPU 上的真实混合 layout 部署尚未验证 |
> | D4 单向标记 | ✅ | ✅ `ray_trainer.py` | L2 ✅ |
> | D4 `holder_roles` / bootstrap | ✅ | ❌ 零 import | L1 |
> | D5 `chunk_offset` | ✅ | ❌ 零 import | L1 |
> | D6 降级继续生成 | ✅ | ✅ | L2 ✅ |
> | D7/D8 DP 副本接管 | ✅ `pull_engine._replica_of` / `_reroute` | ✅ `nixl_pull_model_core` 将 `failed_sources` 提升到 `unavailable`；`chunk_availability.py` 是零 import 的重复实现 | L1 五条故障注入：源失效改读 DP 副本，整组失效抛 `ChunkUnavailable` 且不动权重，mid-pull 失效下次 pull 直接跳过 |
> | E1 版本级等待 | ✅ | ✅ `ps_manager` + `vllm_async_server` + `version_barrier` | L2 ✅，读者侧重拉已接线；**直供 + 同精度路径由 E5 的提交校验取代**（publish 前 drain 无法覆盖「写期间开始、publish 前读完」的读者） |
> | **E5 写入原子性（v2）** | ✅ `write_state.py` | ✅ `PSManager.begin_write` / `finish_shared_pull`，pull engine 抢占与 holder 修复，trainer optimizer 写门闩 | L1 37 条（真值表、抢占/修复/无 holder、扇出复杂度）；两节点生产 e2e 10/10 步，门闩 ≤ 0.09 ms，0 拒绝/错误（FINDINGS §10） |
> | E2 运行中 reshard | ❌ `staleness.py` 只有纯函数 | ❌ 零调用 | 原设计不可行（单边 RDMA，源端无法比对 epoch），见 §9 阶段 E 的 E2 |
> | E3 握手解复用 | ✅ | ✅ `server.py` / `client.py` | L2 ✅ |
> | E3 `admit_joiners` | ✅ | ❌ 生产无调用方（`nixl_admit_joiners` 不存在） | L3 真实 agent 准入（测试直接调用）；RESHARD 分支对既有成员无效，见 E2 |
> | E4 partial MD 两段式 join | ✅ `partial_md.py` `conn_info_only` + `NIXLClientInfo.conn_info_meta` | ✅ `client.py:_ensure_client_info_fetched` 读者加载全量 MD，非读者仅加载 ~401 B conn-info | L1 ✅；NIXL 缺绑定的记录有误——`invalidate_local_metadata` 一直存在，探针用了错误名称 |
> | E4 缩容吊销 descriptor | ✅ `leave_protocol.py` | ✅ `client.py` `_revoke_member` + ack | L1 ✅；L2 `test_revoke_descriptors_gpu.py` 5 passed：吊销后 `NIXL_ERR_NOT_FOUND`，未确认时拒绝释放显存 |
> | F1 delta | ✅ 决策函数 | ❌ | L1 |
> | T1 判据 | ✅ layout 判据 + FP8 逐位判据 | — | 坐标哈希写入原始位，对整行 / 整块错位可见 |
> | T3 colocate | ✅ | — | L2 ✅ |
> | T4 矩阵 | ✅ | — | L2 34 格通过（`weight_transport_FINDINGS.md` §5），L4 235B |

实现期逐项打勾。**判据栏必须可执行**，不接受「已实现」这种描述。

### 阶段 A
- [x] **A1 已落地并验证**：`psrl/utils/nixl/prepped_xfer.py` 的 `PreppedXferCache`
      接入 `_initialize_merged_xfer`，热路径 `initialize_xfer` 调用数 **== 0**（真实 agent 实测）；
      逐字节正确（含 rotated / subset / 重注册）；**2048 descriptor 下提交快 >2×**，
      且优势随 descriptor 数增长。`test_prepped_xfer_cache.py` 29 passed +
      `test_prepped_hotpath_gpu.py` 7 passed。

      **实现中发现三处会让收益归零的陷阱**（都已修并有回归测试）：
      1. `prep_xfer_dlist` 的 `mem_type` 要字符串，merged 路径给的是 `nixl_mem_t` 枚举 →
         裸 `KeyError`，被 fallback 吞掉只留一条 warning
      2. **每次提交重建索引数组是 O(n)**（0.085μs/desc），比 `initialize_xfer` 还慢 5 倍 →
         必须缓存索引数组
      3. **`tuple(wanted)` 做缓存 key 也是 O(n)**（0.027μs/desc）→ 需要 identity 快路径
      最终 4096 descriptor 下 cache 开销从 0.110ms 降到 0.006ms（18×）
- [x] **A2 已落地并验证**：`psrl/utils/nixl/strided_desc.py` 的 `describe_shard`
      判定哪些非连续 shard 可免中转拷贝。strided 与逐条 Nx3 **逐字节一致**。
      `test_strided_desc.py` 35 passed。

      **一处事实更正**：初稿称中转槽是「额外 CPU 权重存储」，
      读 `client.py:541` 后发现分配在 `device=self.device` 即 **GPU** 上
      （变量名 `pinned_memory` 误导）。主要收益是**消除槽位串行等待**，显存次之。

      **三处测试前提被实现推翻**（实现是对的，测试错了）：3-D 张量切中间维
      **是** strided 可表达的（尾部维连续，逐字节验证通过）；
      `torch.flip` 产生拷贝而非负 stride 视图；
      PyTorch 的 `as_strided` 根本拒绝负 stride，该分支只能直接测谓词
- [x] **A3** 「第二个 post 早于第一个 wait 返回」已有断言 — `test_post_wait_split.py` 15 passed
- [x] **A4** 140k tag 构造 20ms（预算 40ms）；单射性含 `(1,2)`vs`(12,)`、None vs ""
      — `tests/nixl/test_xfer_tag.py` 9 passed
- [x] **A5** 五条配置期防御；`model_dtype=bf16` 被真实读取
      — `tests/nixl/test_config_defenses.py` 18 passed / 4 skipped（见下）
- [x] **A6** `flat_param` / FSDP1 分支已删；e2e harness 迁到 `fully_shard`
- [x] **A7** 三级判定；**56 组穷举与真实 runtime 逐一对账**
      — `tests/nixl/test_join_admission.py` 76 passed，
      `tests/nixl/test_meta_server_sharding.py` 10 passed
- [x] **A8 CUDA IPC 机制已验证**（`test_colocate_ipc_gpu.py` 10 passed）：
      跨进程别名双向可见、一块 uint8 bucket 承载 bf16/fp32/fp8/fp16、
      fp8 weight+scale 共存、buffer 不随模型增长、**`register_memory` 调用数 == 0**。
      **KV cache 对比与稳态切回需 vLLM engine，属 L2**
- [x] **A9 block FP8 已接入生产并验证**：`psrl/utils/nixl/block_fp8.py`。
      rollout 声明 `weight_quantization.scheme: fp8` 时 `validate_config` 解析出 `publish_quantization` 计划。
      vLLM converter 在每个 FP8 权重的规范视图旁给出 `<key>_scale_inv`，取融合 scale 的对应视图，
      于是 q/k/v、gate/up 与分组 expert 各自带 scale key，切法与权重一致。meta server 从 trainer
      权重的切法推出 scale 的切法，trainer 分配 scale 张量、把这些权重发布为 FP8 副本，
      发布时逐张量重量化，瞬时显存只有一个 key。
      `quantize_blockwise` 与 vLLM 的 `per_block_cast_to_fp8` 逐位一致（`test_block_fp8.py`）。
      L2 8 格 FP8 位与 scale 逐位判定通过。真实 Qwen3-8B 权重下贪心 token 与 bf16 rollout 100% 一致。
      比一个 tile 更细的切法在注册时拒绝并报出 key 名（细化网格见 §12 W9）。

**单机验证总计：510 passed, 23 skipped**（含全部 GPU 测试）。
FSDP2 converter 在 2 卡与 4 卡各验证 27 个张量（torchrun，独立运行）。

**A5 的 4 个 skip 不是未完成**：`fsdp_converter` 导入 `verl.utils.model` →
`transformers.AutoModel`，在本环境因 numpy 2 冲突而失败，**改动前后同样失败**
（已用 `git stash` 对照确认）。该断言是纯 shape 检查，已用 AST 抽取函数体单独验证通过，
测试用 `importorskip` 跳过而非失败，以免掩盖真实回归。

### 阶段 B
- **B1 / B3 已删除。** 拉取引擎为每个源规划一次读、每次 pull 提交一个 prepped handle，
  chunk 计划与 chunk 自适应没有消费者。
- **B2 不再需要。** `slot_ack.py` 保护「`2×chunk` 循环复用的 wire buffer 不被下一轮 cast 覆盖」。
  pull 语义下版本 v 的读者可以在 v+1 发布前任意时刻到达，cast 产物必须活一个版本，
  所以 cast 产物是常驻的发布副本，在推送锁内刷新，没有可复用的 slot（主文档 §3.4.2）。
  模块与单测保留，零生产 import。

### 阶段 C
- [x] **C1 已落地并验证**：`psrl/utils/nixl/moe_grouping.py` + `BaseConverter.maybe_group_experts`。
      **真实 checkpoint 端到端实测**（经 HF converter）：Qwen3-30B-A3B 18867 → 579 key（32.6×），
      144 个 expert group，每个 key 128 shard、`shard_mesh {0: 128}`；
      `client_info` pickle **3.475MB → 0.324MB（10.7×）**；dense 模型零改动（Qwen3-1.7B 311 key，无分组）。
      **关键正确性**：expert id 必须**数值排序而非字典序**——
      字典序会把 expert 10 排在 2 前面,静默置换所有 expert。
      5 节点 e2e 通过（16 个 gen client 全部 399/399 坐标匹配）。
      `test_moe_grouping.py` 42 + `test_shard_meta_interning.py` 12 + `test_moe_grouping_gpu.py` 9 passed

      **三处设计文档未写明、由实测确定的事实**：
      1. **两侧 expert 存储结构不同**：vLLM 的 `w13_weight` 是一个 fused 3-D 张量
         （expert 轴是真实张量维度），而 Megatron/TE 每个 expert 是独立的 `torch.empty`
         （TE `grouped_linear.py` 按 GEMM 注册 `weight{i}`），彼此不共享 storage。
         所以一个 grouped key 在一侧是 1 个张量、在另一侧是 N 个——`GroupedShards` 即为此而设。
         强行 stack 成单张量会在每次同步复制整个 MoE，**比分组省下的元数据还大**。
      2. **10.7× 中只有 2.5× 来自分组本身**，其余来自 **shard metadata 的 interning**：
         128 个 expert 的 metadata 对象「相等但非同一」，而 pickle 按 identity 记忆化
         （实测 7473 B vs 488 B）。不做 interning 会把 key 数的收益大半吐回去。
      3. **§3.5.5(b) 的 strided 16384:1 压缩只适用于 vLLM 侧**。
         实测真实尺寸（768×2048）下 Megatron 相邻 expert 分配**不是均匀跨步**，
         单条 strided descriptor 覆盖不了，所以训练侧的 per-expert descriptor 不可约。
         分组后随 expert 数增长的是 **descriptor 而非 metadata**，
         判据应是「远低于线性」而不是「常数」。
- [ ] **C2 两条判据均已验证，握手期 plan 缓存待接线**：
      1. **稳态 rollout→PS RPC == 0** —— 已验证：plan 握手期一次编译并本地缓存，
         拉取路径 `.remote(` / `ray.get` 源码零命中（与 B3 同一组断言）。
         5 节点日志 `Making communication plan` 出现 1 次。
         `ControlPlaneMeter` 提供窗口计数，undercount 比 overcount 危险
         （会让判据假装满足），故用锁保护。
      2. **控制面消息非平方增长** —— 已验证：
         `test_partial_md.py::TestControlPlaneScaling`，
         rollout 数 4→32 时单个 train client 的 descriptor 接收者**不超线性增长**。
         机制是 plan 按分片指派源，而非按人头广播 ——
         每个 rollout 只读 train client 的一个子集（测试同时钉住这条机制）。

      **仍待做**：措施 (a) 的「holder 集合变化时由 PS 主动失效通知」与
      (d) 的 gRPC 内存快照读路径。当前稳态已不查 PS，
      所以这两项影响的是**弹性期**而非稳态，与 E3 同批更合适。
- [x] **C3 已落地并验证**：`psrl/workers/train/step_boundary.py` 的 `StepBoundaryRecorder`，
      NIXL 与 Ray 两条发布路径都记录。`test_step_boundary.py` 13 passed。

      **一处设计文档的事实更正**：本文原方案要求把发布挂到 Megatron-Bridge 的
      `on_train_step_end`。该回调确实存在且时机正确（`train.py` 中在 `wrapped_train_step`
      之后触发），但**在 PSRL 中永远不会触发**——它属于 Bridge 自己的 `pretrain`/`train`
      循环，而 PSRL 只用 Bridge 做**模型转换**（`megatron_converter` 的 `AutoBridge`），
      RL 的 optimizer step 走 verl 的 `train_mini_batch`；全仓库没有任何 `pretrain()` 调用。
      挂上去会**正常导入、正常运行、永不触发**，失败形态是发布 trace 恒为空。
      因此边界定义在 PSRL 实际运行的循环上。设计想要的时序其实**已经结构性成立**
      （`update_actor` 在 optimizer step 返回后立刻调 `push_model`，NIXL 路径又在所有
      shard wait 完成后才发布），缺的只是**可观测性**——而那正是判据要求验证的东西。

### 阶段 D（不可分割）

> **当前状态**：D1 与 D3 的直供路径在生产中运行，扩散默认开。D2/D5/D7/D8 只有逻辑层与单测。
> PS 代码（`PSStorageWorker` 等）尚未删除。
>
> **多节点验证必须先确认 actor 导入的是本工作树。** Ray actor 可能从另一个 checkout 导入同名包，
> driver 跑本树、actor 跑那棵树，测试测的是不存在的代码且会通过。harness 启动时逐节点断言
> `[PASS] all N nodes import this tree`，没有这一行的多节点结果不算数。

- [x] **D1 已接入生产并验证**：`psrl/utils/nixl/pull_engine.py`，由 `nixl_pull_model_core` 驱动。
      同节点需要同一 key 同一组 shard 的 rollout 组成一组，designate 次序按上游字节在组员间均摊。
      每次 sync 由 coordinator 开一个 epoch 并给出参与者，组内首个参与者读上游，其余整张量读它。
      判据「trainer 出口 == 每 gen 节点每 gen TP rank 一份」已达成：4 节点 16 rollout 下
      131 GB → 32.8 GB，稳态 sync 0.178 s（无扩散 0.67–0.84 s）。数据见 `weight_transport_FINDINGS.md` §4。

      **实现中钉住的四条规则**（都有 CPU 回归测试，`test_pull_engine.py`）：
      1. **peer 读必须等 designate 的字节落地。** 提前的 peer 读会「成功」：注册有效、传输格式正确，
         读到的是未填充的缓冲区。designate 每落地一波就向 follower 发一条 NIXL 通知，方向只有
         designate → follower。全局屏障正确但会让整组等最慢的一个，共享波次计数器在各 client
         波次表不同时全局乱序，两者都不可用。
      2. **就绪信号按 sync epoch 区分，不按权重版本。** 两次 sync 可以拉同一版本，
         复用版本号会让 follower 接受上一次 sync 的就绪信号，读到半旧的权重。
      3. **peer 读按 gen 本地整张量。** `postXferReq` 的成本按 descriptor 计（约 2 µs 一个），
         沿用全局 finest cell 时 descriptor 数随行数增长，扩散因此净亏。
      4. **designate 不响应时 follower 在 `peer_timeout_s` 后改读上游**，不让同组卡住。

- **D1 peer 读的版本保证：现状是时序，不是断言。** 设计 §3.5.2 要求 `version(peer) == v`，
      实现里没有对应检查。实测（CPU 假集群，`peer_diffusion=True`）：

      | 观测 | 值 |
      |---|---|
      | rollout 报告的拉取版本（第 1、2 次） | 均为 `[-1]`，包括第 2 次已读到 v8 字节时 |
      | rollout 缓存的另一个 rollout（designate）的版本戳 | `[-1]` |

      `publish_version` 只对 trainer 调用，rollout 的分片戳从不更新，所以 peer 读上
      `vllm_extension.py` 的 `assert len(pulled_version) <= 1` 恒通过，对 peer 读什么也没校验。
      读到的字节是对的，靠的是：designate 先完成该波上游读并 `cuda.synchronize`，之后才发
      `(epoch, wave, sender)` 就绪信号；follower 读完发 `_DONE`，designate 返回前 `_await(_DONE)`。
      正常路径下 designate 不会在 follower 读完前进入下一轮。

      **已知缺口**：`_DONE` 屏障有 `peer_timeout_s`（默认 120 s）的超时，超时只打 warning 就放行，
      之后下一轮可能在慢 follower 仍在读时改写缓冲区。另外若 designate 自己读到了旧版本
      （真实栈上读者缓存的 trainer 戳只在握手或 colocate 切换时刷新），follower 会跟着拿到旧版本，
      版本集合仍是单元素，不报错。后一条依赖真实栈的缓存陈旧，假集群测不出，**未验证**。

- [x] **D1 选源模块已落地**：`psrl/utils/nixl/source_resolution.py`。
      trainer 出口 == 节点数（4 节点 × 8 reader 实测为 4 次而非 32 次）；
      出口随「每节点 reader 数」增长保持不变；无链式结构（逐轮断言源不是二手 holder）；
      扩散轮数对数级（16 reader ≤ 6 轮）。`test_source_resolution.py` 22 passed

      **一个实现期发现的真实 bug**：W 个 trainer 名额若按 reader 列表顺序发放，
      4 个名额会全落在 node0，trainer 服务 16 次而非 4 次 —— 正是本项要消除的 `N×M`。
      名额必须**按节点**发放：拿到一份的节点能用 NVLink 扩散，没拿到的节点下一轮还得回 trainer。
      **判据写成「数 trainer 出口次数」而非「断言发生了扩散」，才看得见这个差别。**
- [x] **D2 已落地（模块）**：`psrl/workers/ps/holder_index.py`。
      `(role, version, chunk)` 三元键；actor 的 v3 与 critic 的 v3 互不覆盖；
      pull 准入四类拒绝（非 holder / 版本不符 / 二手 holder / 角色未发布）；
      在飞读者为 per-chunk 整数，释放前断言归零且**报错点名是哪些 chunk**。
      `test_holder_index.py` 21 passed
- [ ] **D3 部分落地：直供链路已端到端打通，删除未做**。
      5 节点实测（`nixl_e2e_5node_direct.yaml`，与 PS 版仅差一行配置）：
      **exit=0，16 个 gen client 全部 399/399，运行中零 `PSStorageWorker` 进程**，
      gen 从 train client 直接拉取。PS 路径同时复测通过（exit=0，16/16，pull 1.14s）。

      **跑通过程中暴露的两个真实阻塞点**（planner 改动本身覆盖不到）：
      1. **meta server 不给 rollout 任何 descriptor**。
         `_get_relevant_client_names_for_agent` 只会把 `PS_FOR_PULL` 发给 `PULL_SIDE`，
         无 PS 时该集合为空，首次读直接报
         `target client not found in client infos`——看起来像握手 bug，
         实际是过滤器没跟上。现改为**按注册情况判断**，与 planner 的分派方式一致。
         测试同时校验 PS 路径的扇出**没有变宽**，否则「永远加上 train client」
         也能让直供用例通过。
      2. **`client_read` 在查 plan 之前就解引用对端的 tensor info**。
         PP 下每个 train rank 只持有本 stage 的 key，向 rank 0 要 rank 12 的 key
         会抛裸 `KeyError`——读起来像权重丢失，实则是正常的 PP 分片。
         对 PS 从不触发（每个 PS worker 持有全部 key），所以这个顺序一直是
         **碰巧安全**而非设计安全。`client_write` 同步加了同样的保护。

      **性能现状要如实说**：直供 pull 2.59s vs PS 路径 1.14s。
      这是预期之内且尚未处理的——当前每个 rollout 都去读**每一个** train rank，
      正是 D1 的选源要消除的 `N×M`。接线 D1 是下一步。

      **删除的前置已补上：生产侧新增 `ps_mode=nixl_direct`**
      （`ray_trainer.py` 的分发分支 + `config.py` 的校验，
      `test_direct_supply_mode.py` 8 passed）。

      **这是一处此前会导致删错东西的缺口**：`direct_supply` 只是**测试配置键**，
      生产的 `ray_trainer.py` 按 `ps_mode` 分发，而该枚举**没有「无 PS」的取值**。
      先删 `PSStorageWorker` 会让生产失去全部传输路径，
      **而 e2e harness 仍然全绿**——它根本不读 `ps_mode`。
      新分支同时拒绝 `lmcache.enable_p2p`（它要经 PS 路由），
      在配置期报错而非传输期。

      **第二处前置也已补上：`ps_mode` 的判定集中化**
      （`psrl/utils/ps_mode.py`，`test_ps_mode.py` 14 passed）。

      此前 `ps_mode == "nixl_cpu" or ps_mode == "nixl_gpu"` 这个短语
      **在 6 个文件里重复了 12 次**，于是加一个模式变成了搜索题：
      `nixl_direct` 被配置校验和 worker 构建接受，却仍被另外 10 处守卫拒绝 ——
      **模式存在但不可用，而且没有任何地方会这么报告**，
      每处守卫都以自己的理由失败，离引入模式的那个决定很远。

      集中化同时纠正了一处被混淆的语义：这些守卫问的其实是**两个不同的问题** ——
      `uses_nixl`（传输是否走 NIXL，直供为真）与
      `has_ps_data_plane`（PS 是否持有权重，直供为假）。
      **多数站点意思是前者、却写成了后者**。现在按真实意图分别替换：
      push 用 `has_ps_data_plane`（没有 PS 就无处可推），
      pull 与握手用 `uses_nixl`（源是 PS 还是 train rank 都一样）。
      保留的唯一一处 `nixl_cpu` 单独判定是**放置相关**（PS worker 落在 host 还是 device），
      已有测试记录它为何幸存，以免看起来像漏网。

      **生产脚本已就位**：`examples/grpo_trainer/qwen3_8b_megatron_direct.sh`，
      与基线脚本只差 `ps_mode=nixl_direct` 与运行长度（2 步，覆盖第二个权重版本）。

      **但生产路径实跑被一处环境问题阻断，且该问题与直供无关**：
      `TaskRunner` actor 里 `megatron.bridge → modelopt → tilelang` 先加载了
      `libcudart_stub.so`，之后 vLLM 懒加载 `flashinfer.comm` 时
      解析 `cudaDeviceReset` 失败。**基线代码同样阻断**（已 `git stash` 对照确认），
      所以这不是直供引入的。

      已做的两件事：
      1. `main_ppo.py` 与 `ray_trainer.py` 顶部预加载 `flashinfer.comm`
         （与 e2e harness 同款修法）。**在进程内复现时有效**，
         但 actor 里 tilelang 的加载发生得更早，预加载本身就已失败。
      2. 把该预加载的失败**从静默 `pass` 改为告警**。
         这是定位的关键：失败在 `TaskRunner` 里早已发生，
         而真正的报错出现在很远的 `verl.workers.rollout` 内部，
         读起来像 vLLM 的问题。现在两者能被连起来。

      **另一处必须记录的环境事实**：本树 vendored 的 verl（`8718ca30`）与集群环境
      默认解析到的（`ls/psrl` 的 `74cebc50`）**不是同一个版本**，
      且两边 schema 互不兼容（前者无 `vanilla_mbridge`，后者无 `configure_verl_logging`）。
      用 `PYTHONPATH=<repo>:<repo>/third_party/verl` 跑测试：
      **1203 passed / 15 failed / 21 errors → 1309 passed / 5 failed / 3 errors**。
      此前报告的「既有失败」大部分是这个路径问题，不是真实缺陷。

      **根因不是直供，而是本树处于一次未完成的 verl 迁移中。** 这一点花了若干次
      集群实跑才定位清楚，值得完整记录：

      集群上有两个 verl checkout，差别**不是版本号**：

      | | `<repo>/third_party/verl` | 环境默认 (`ls/psrl/third_party/verl`) |
      |---|---|---|
      | upstream commit | `8718ca30`（较新） | `74cebc50`（较旧） |
      | PSRL 补丁 | 无 | **705 行新增 / 20 个文件 / 未提交** |

      不带 `PYTHONPATH` 时 `import verl` 解析到**打了补丁的旧版**；较新的那个
      **未被 git 跟踪、也不是 submodule**。

      **必须用较新的那个**，这不是偏好：PSRL HEAD 需要
      `verl.trainer.ppo.metric_utils.RolloutMoELoadBalanceMetricsAccumulator`，
      只有新版有。我试过用打补丁的旧版以"免费"获得那 705 行，结果在 import 阶段就失败。

      **于是每一个"verl 缺少的符号"其实都是一处需要前移到 PSRL 的补丁。**
      不要再去改 vendored checkout —— 把能力搬进 PSRL，它才能活过下一次 verl bump。
      已完成的前移（每一项都是一次实跑暴露出来的）：

      | 补丁 | 前移到 | 原来的失败形态 |
      |---|---|---|
      | `resource_num_per_bundle` | `psrl/trainer/ppo/resource_pool.py` | `init_workers()` 里 `TypeError` |
      | `configure_verl_logging` | `psrl/utils/logger/verl_logging.py` | import 期 `ImportError` |
      | `max_concurrency` | `PSRL_vLLMReplica` 的 property | 启 rollout server 时 `AttributeError` |
      | `use_per_rank_checkpoint` | 从 verl patch 移除 | struct 模式下写未声明键；nvrx 落地后成为死代码 |
      | `load_weight` | 让 upstream 照常加载 | `McoreEngineConfig` 收到未知 kwarg |

      其中**分数 bundle 必须在两处同时生效**：placement group 的 bundle，
      *以及* actor 的 `num_gpus`。只做前者会得到
      `resource request {'GPU': 1.0} cannot fit into any bundles`。
      而 0.9/0.1 的**不相等**本身就是机制 —— 两个 0.5 会让 Ray 把第二个
      train worker 塞进同一张卡。

      **`LD_PRELOAD` 已试过且不成立，原因值得记下来**：
      预加载真实 `libcudart.so.12` 在**进程内**确实能让
      `import tilelang` 之后的 `flashinfer.comm` 正常工作（已验证），
      且 `LD_PRELOAD` 已通过 `_HOST_RUNTIME_ENV_KEYS` 送达 actor
      （日志里可见它出现在 runtime env 中）。
      但**对已启动的 Ray worker 进程无效** —— `LD_PRELOAD` 由动态链接器在
      `execve` 时读取，Ray 把它作为环境变量注入到一个早已运行的进程，为时已晚。
      要生效需要在 **Ray worker 启动前**设置，即在 `ray start` 的环境里，
      属于集群部署配置而非本仓库可改的范围。

      **生产实跑已跑通。** `examples/grpo_trainer/qwen3_8b_megatron_direct.sh`
      以 `ps_mode=nixl_direct` 在 5 节点 40 卡上完成启动并进入训练循环，
      日志依次出现 `Binding direct-supply sources`、`Initial pull`、
      `Training Progress`、`step:1`、`actor/pg_loss` ——
      即**无 PS worker 的直供路径完成了一次完整的权重供给与一个训练步**。

      跑通过程中暴露并修掉的 `nixl_direct` 真实缺陷（每一个都只在实跑中出现，
      单测与 e2e harness 都测不到，因为 harness 根本不读 `ps_mode`）：

      | 缺陷 | 症状 | 修法 |
      |---|---|---|
      | `self.ps_wg` 在 `uses_nixl` 块里被读 | `AttributeError`，发生在 Ray 起好、server 注册完之后 | 按 `has_ps_data_plane` 分流 |
      | agent 数把不存在的 PS 算进去 | 不报错，meta server 等一批永不注册的 agent 直到超时 | `ps_world_size = 0` |
      | gen 侧 pull 源无人作答 | `get_ps_nixl_agent_names` assert | 新增 `bind_direct_supply_sources`，用 train rank 回答 |
      | train rank 自己也去 pull | 同上 assert | 直供下跳过；它们**就是**源 |
      | `push_model` 落到 `else` 分支 | `NotImplementedError` | 新增 `direct_supply_publish`：不搬字节，但仍发布版本 |

      其中**第 2 和第 5 条是静默失败**：前者表现为超时而非错误，
      后者若只是跳过 push 而不发版本，rollout 会**继续成功地 pull 到旧权重**。

      **稳态实跑结果**（`/tmp/prod_direct26.log`，5 节点 40 卡，2 个训练步）：

      | 量 | 值 |
      |---|---|
      | rollout 从 train rank 拉取 | **1.38 / 1.77 / 1.78 s** |
      | 版本发布（`direct_supply_publish`，不搬字节） | **0.40 s** |
      | 成功的 pull 次数 | 5 |
      | 错误 | 0 |

      **但这次实跑不能用来验证权重的数值正确性**，必须说清楚：
      `actor/pg_loss`、`critic/score/mean` 全为 0。原因是 2 步的 base model
      在数学题上答不对，reward 恒为 0，GRPO 的 advantage 因此恒为 0。
      **一个只能是 0 的 loss 不构成"权重正确"的证据。**
      数值正确性仍然由 `tests/nixl/` 的坐标判据（`coord_judge.py`）保证 ——
      那里才有逐元素的 oracle，而端到端训练只证明了**链路能跑通且性能符合预期**。

      途中还踩到一个值得记的配置陷阱：最初把 `max_response_length` 从基线的 8192
      改成 1024 以求快，导致 `response_length/clip_ratio = 1.0`（每条回答都被截断），
      于是 reward 恒 0、loss 恒 0，而**日志完全正常、两步都"成功"**。
      改回 8192 后 clip_ratio 降到 0.0625 / 0.5。
      缩小 e2e 配置时应该动 `train_batch_size` / `rollout_n` / `total_training_steps`，
      不要动 `max_response_length`。

      **设计里缺的一条：直供与 Megatron 分布式 checkpoint 互斥。**
      这不是实现疏漏，`weight_transport.md` 里没有讨论过它 ——
      是生产实跑在第 2 步暴露出来的：

      - 直供要求 train rank 上的 **NIXL progress thread 常开**，
        因为 rollout 在它计算的同时 RDMA 读它，而那条路径要等 `DONE`；
      - 原先的判断是：Megatron 的 `dist_checkpointing` 的 `all_gather_object`
        会破坏该线程读取的 UCX endpoint 地址，所以 `checkpoint.use_dcp_save`
        在 DCP 保存时把线程关掉。

      症状是：传输**提交成功**然后永远停在 `PROC`，
      最终报在**读方**身上（`Timeout waiting for transfer ... to NIXLTrainClient_N`），
      而真正没能服务的那个 rank **什么都不打**。
      而且**第 1 步能过、第 2 步才挂** —— 首次 pull 发生在 trainer 空闲时。

      **这个归因没有被复现，且症状有一个更简单的解释。** 进度线程关闭时，
      传输同样是「提交成功、停在 `PROC`」（`conftest_agents.py` 记录了同一现象），
      而 `use_dcp_save=True` 恰恰就是关线程的开关。症状与「线程被关掉」吻合，
      不需要「DCP 破坏了 UCX」这个假设。这是推断，不是已证实的因果。

      实测见 `weight_transport_FINDINGS.md` §9：进度线程常开时，同步 DCP 保存与 NIXL 读
      并发，没有出现传输错误或字节不一致；`addr_version` 断言没有复现。
      **能确认存在的冲突只有一个：fork 式异步保存。**

      **当前解法（已落地，commit c1ffe51）**：nvrx 异步 DCP——`nvidia-resiliency-ext`
      0.7.0 的 `AsyncCallsQueue(persistent=True)` 以 `spawn`（非 fork）启动 persistent
      worker，worker 进程不继承父进程的 UCX context 和已注册显存，因此与 NIXL 进度线程
      完全兼容。落地改动：

      - `patch/verl/dc709af.patch`：`_dispatch_finalize` 从
        `megatron.core.strategies.base.async_calls`（Megatron 0.19 不存在）改为 nvrx
        `AsyncCallsQueue`；`save_dist_checkpointing` 用 `async_strategy="nvrx"` 而非
        `async_sharded_save=True`（后者触发 fork）
      - `patch/verl/dc709af.patch`：删除 `use_per_rank_checkpoint` 相关 hunk。没有任何
        活跃配置再把它置为 true，逐 rank 路径已由 `PSRL_PerRankMegatronCheckpointManager`
        子类独立提供，不再需要改 vendored checkout
      - `psrl/utils/config.py`：移除 `nixl_direct + use_dcp_save=True` 的 assert，
        nvrx spawn 路径安全
      - `psrl/workers/train/engine_train_worker.py`：`_maybe_use_per_rank_checkpointing`
        改为 no-op；进度线程始终开启（不再被 `use_dcp_save` 关掉）
      - `scripts/install_basic.sh`、`docker/Dockerfile`：step 9 安装 nvrx 0.7.0 from source
        （PyPI 只有 0.0.0a0 占位包）

      nvrx + NIXL 并发 30 s 实测（`repro_dcp_nvrx.py`）：113–119 GB/s，0 错误，PASS。
      nvrx 独立（无 NIXL）同样 PASS。fork 路径实测 FAIL（`NIXL_ERR_REMOTE_DISCONNECT`），
      是 OS 层面约束，版本升级无法修复。

      **`PSRL_PerRankMegatronCheckpointManager` 的状态**：已保留但不再激活。
      `_maybe_use_per_rank_checkpointing` 是 no-op，该 manager 作为后备保留直到真实栈
      的 nvrx async save 端到端验证完成（需 Ray 集群跑到 `save_freq` 边界，确认
      `.distcp` 目录完整、NIXL 0 错误、能 resume）。它是自包含子类，不依赖 verl patch
      中的逐 rank 钩子。

      **旧的临时方案（已废弃，但代码保留）**：`psrl/utils/checkpoint/megatron_manager.py`
      中的 `PSRL_PerRankMegatronCheckpointManager` 用逐 rank `torch.save` 取代
      `dist_checkpointing`，绕开了 UCX 兼容问题，但**不可 reshard**（只能用相同
      world size 与并行布局恢复）。nvrx 方案落地后，该 manager 不再被激活。

      **§3.4「版本相等性」已落地**，但走了一段弯路，值得记下来。

      我先做了**内容 checksum**，这是对设计的误读 —— §3.4 要的是
      「**版本相等性**」断言，并明确说元数据类校验「开销是常数级，**应当常开**」。
      实测证明 checksum 不可能常开：

      | | 8B 模型一次 pull | 随模型增大 |
      |---|---|---|
      | 内容 hash | **61,000 ms** | 线性，且随 dtype/量化变化 |
      | 版本戳（publish） | **0.057 ms** | **不变** |
      | 版本戳（observe） | **0.076 ms** | **不变** |

      pull 本身只要 1.4 s —— hash 比它要检查的东西贵 **40 倍**，
      于是只能采样、只能默认关，**故障恰好落在没采样到的 key 上**。
      `weight_checksum.py` 与其 22 个测试已删除。

      版本戳覆盖 **stale / torn / misdirected** 读；**不覆盖**正确版本字节的位翻转
      —— 这是有意取舍：RDMA 由 NIC 校验，而 stale 读是**逻辑错误且完全无症状**
      （pull 成功、版本号前进、引擎服务上一步权重）。

      **校验分两层，因为元数据新鲜度不免费**：撕裂读检查每次 pull 都跑
      （只比较分片之间，不需要新元数据）；而「announced vs observed」
      只在**首次 pull** 跑 —— 读者缓存的 `NIXLClientInfo` 仅在握手和 colocate 切换时刷新，
      做成逐步检查需要每步一次元数据广播，比它要抓的故障更贵。

      **验证方法本身也是一个教训。** 定位「版本戳写在哪」花了 **3 次多节点实跑、约 90 分钟**，
      而三次失败全都是**调用顺序**问题（stamp 在 `nixl_protocol` 之后 → 读者缓存 `-1`；
      在握手内问 PSManager → 与它自己的 `wait_for_client_infos` 死锁）。
      这些用源码检查（毫秒）和单节点元数据往返（1.9 s）都能定下来。

      **今后按「能达到验证目的的最轻档位」选择**：
      CPU-only < 单 GPU < 单节点多 GPU < 多节点。
      签名兼容、调用顺序、配置合法性、代价模型都属于 CPU 档；
      多节点只留给**跨主机 RDMA 与真实放置**。

      **版本断言已在多节点实跑验证通过**（run 38，`/tmp/prod_direct38.log`）：
      6 个 client 全部连上、握手 **1 秒**完成、
      `Binding direct-supply` → `Initial pull` → `step:1`，
      **0 错误、0 条版本违例**（既无 `torn read` 也无 `stale pull`）。
      rollout pull **1.25 / 1.56 / 1.69 s**，版本发布 **0.00 s**。

      这意味着：首次 pull 的 announced-vs-observed 检查、
      以及每次 pull 的撕裂读检查，在真实跨主机 RDMA 下都**不误报**。

      **但在此之前 run 33–37 连续失败，根因不是超时值，值得记下来。**
      它们都报 `Timeout waiting for agents`，于是看起来像"启动慢"：

      | | 等待开始 | 握手完成 |
      |---|---|---|
      | run 28（成功） | 14:45:58 | **14:45:59**（~1 s） |
      | run 30（成功） | 16:11:56 | **16:11:56**（~0 s） |
      | run 36 | 20:19:27 | 20:49:28 **超时** |

      成功的 run 里握手只要 ~1 秒 —— 因为客户端早已连上。
      run 36 等满了配置的 1800 s，**而只有 4 个 client 连上（成功时 6 个）**，
      且 `Capturing CUDA graphs (FULL): 100%` 只出现 5 次（成功时 6 次）：
      **有一个 vLLM replica 自始至终没有完成启动**。

      **真正的根因是 Ray actor 调用串行化造成的死锁。**
      `PSManager` 是 actor，一旦 `nixl_protocol` 跑起来它就卡在
      `wait_for_client_shardings`，**此后任何对该 handle 的调用都排在它后面** ——
      包括 train rank 启动握手前必须读的 `get_ps_model_version`。
      于是服务端等客户端、客户端等服务端。

      **区分"死锁"与"启动慢"的判据**（两者都报同一句话）：
      成功的 run 握手只要 **~1 秒**，所以 27 分钟的等待从来不是"超时值太紧"；
      再看 `Capturing CUDA graphs (FULL): 100%` 已满 6 个而
      `connect_to_server` 仍为 0 —— 那就是死锁。

      把 `handshake_timeout_s` 做成可配置仍是对的（客户端 1200 s / 服务端 600 s
      的不一致是真实缺陷：服务端先放弃，客户端侧什么都没记下来），
      但它**不是** run 33–37 失败的原因，文档不应暗示那是修复。

      顺带还暴露一个我自己引入的缺陷：那次把 11 个 wait 改成
      `timeout: float | None = None` 时，有 7 处函数体仍在比较 `timeout`，
      而生产调用全都省略该参数 —— 首次循环即
      `TypeError: '>' not supported between 'float' and 'NoneType'`。
      它表现为客户端未连上、**服务端**报 `Timeout waiting for agents`，
      与上面两种现象完全同形。现已由按**模式**（而非按实例）扫描的测试守住。

      **仍待做**：删除 `PSStorageWorker` / `create_ps_worker_group` / push 路径 /
      `ps_mode` 的 PS 取值。

      **删除前还剩一个真实障碍，必须说清楚**：75 个生产脚本里有
      **75 个仍在用 `ps_mode=nixl_cpu`**。删掉 PS 等于要求它们全部切到直供，
      而直供强制 `use_dcp_save=False`，于是它们的 checkpoint 格式
      从可 reshard 的 `torch_dist` 变成**不可 reshard 的逐 rank 文件**。
      这是一次对现有产出物格式的单向改变，**不是纯粹的代码删除**，
      应当由项目主人决定迁移节奏，而不是随手跟着删代码一起做掉。
      判据（CPU 权重 == 0、cast 显存不随 rollout 数增长）需删除后实测。
      `test_direct_supply_mode.py` 留了一个**成功即失败**的闸门测试：
      `PSStorageWorker` 文件一旦消失它就报错，提示连同这组闸门一起退役。
- [x] **D4 已落地并集群验证**：
      `psrl/workers/ps/holder_roles.py`，`test_holder_roles.py` 17 passed。

      三件事已就位：
      1. **holder 对称化**：`HolderKind` = `SELF` / `ROLLOUT_GPU` / `TRAIN_GPU`，
         按「字节在哪」而非「进程是什么角色」标识。
         这是 rollout-only 启动能成立的前提 —— trainer 尚未存在时，`v0` 必须有源。
      2. **bootstrap 路由**：`BootstrapPolicy.source_for`。
         trainer 在 **cast 路径下恒读 checkpoint**，即使 peer 持有该版本 ——
         rollout 手里是 fp8/fp4，反量化有损且 fp32 master params 从未上过线，
         「peer 有权重」不等于「trainer 能续训」。同精度 wire 下该限制自动放宽。
      3. **首次 vs 稳态用单向标记**，不从 `is_rollout_mode_in_actor` 推导 ——
         后者每次 colocate 切换都翻转，分不出「第一次到这」和「又回到这」。
         稳态切回误读盘会**静默回退训练进度**（权重合法、只是旧的），
         已用变异测试确认该断言真能抓到：去掉 parked 判定后该测试失败。

      每节点一个 checkpoint reader（`choose_checkpoint_readers`），
      读者数**不随每节点 rollout 数增长**，与传输路径的去冗余同构。

      **已接线（T3 完成后）**：`ray_trainer.py` 的 `switch_to_trainer_mode`
      现在带一条断言——**稳态切回时若仍要读 checkpoint 就报错**，
      并在权重拉回之后才置标记（中途异常仍算「首次」）。
      `test_colocate_switchback.py` 8 passed，含变异测试：
      把标记改回由 `is_rollout_mode_in_actor` 推导后，3 个测试失败。

      **连续两轮已验证（colocate 配置，`rounds: 2`）**：
      exit=0，第二轮拉取 0.112s，**8 个 gen client 全部 399/399 匹配「新版本」的坐标**。

      关键在于第二轮必须是**真的测试**：train 侧用 `advance_version(base)` 平移坐标基址
      重新填充，否则重填会产生**完全相同的字节**，于是「什么都没收到」与
      「收到了新版本」在判据下无法区分 —— 这正是稳态切回两种失败
      （误读盘、服务缓存版本）的共同形态：权重合法，只是错的。
      已用单测钉住：旧版本权重在新判据下**必须失败**（`test_coord_judge.py` 34 passed）。
- [x] **D5 已落地**：`psrl/utils/nixl/chunk_offset.py`。
      偏移**向下**吸附到 commit group 边界（逐 instance 验证每个 group 在读序中连续），
      向下而非就近是为了保持对 instance 序号单调，index 据此无通信预测 holder；
      **收益实测**：4 instance 各读 2 个 chunk 后，错开覆盖 8 个 chunk、不错开只覆盖 2 个。
      `R <= W` 时不启用。`test_chunk_offset.py` 19 passed
- [x] **D6 已落地**：`pull_model_for_sync` 的版本 assert 已删除 ——
      它会经 `_finish_gateway_sync` 的 except 走到 `fail_sync` + `/workers/pause`，
      **把「落后一个版本」变成「摘除该 replica」**，与判据正好相反。
      发布时机从「pull 前乐观预发」改为「pull 落地后按实际版本发布」，
      消除了「SMG 记 v+1 而实例持 v」的窗口。
      连带改动：原先「仅在实际版本与目标不符时才发布」依赖预发存在，
      去掉预发后达标的 replica 将永不发布，故改为无条件发布。
      `test_degraded_sync.py` 8 passed + 5 节点 e2e exit=0
- [x] **D7 已落地（模块）**：`psrl/workers/ps/chunk_availability.py`。
      单 rank 故障由同 DP 组接管且 rollout 无感知；整组故障返回显式
      `CHUNK_UNAVAILABLE`（非超时）；trainer 全体重启清空 holder 但保留版本号。
      故障检测搭在已有错误路径上（`client_read` 失败时上报 holder 身份），不加心跳。
- [x] **D8 已落地（模块）**：副本数由 `NIXLSharding` 运行时推导。
      **决定性的一对测试**：同一个单 rank 故障，HSDP（`{0:8}`/16 rank → 2 副本）走 DP 接管，
      纯 FSDP2（`{0:16}`/16 rank → 1 副本）返回 `CHUNK_UNAVAILABLE`。
      mesh 声明的分片数超过 world size 时**报错而非返回 0 副本**（后者会让所有 chunk 静默不可用）。
      `test_chunk_availability.py` 21 passed

### 阶段 E
- [x] **E1 已接线并达成 L2 判据（29 passed，含真值表与撕裂复现）**：
      `psrl/workers/ps/version_barrier.py`。
      发布前按预算等待 v 的在飞读者归零；**超限即发布并把 v 标记 INVALID**，
      读者改拉 v+1（更新的权重，算法上是改善）——
      正确性靠**失效通知**而非无限等待，因为 trainer 的下一步就是在产 v+2。
      `requires_version_barrier` 把「谁需要」写成配置的函数：
      FSDP2 同精度 **必需**（无 master params，整个 optimizer step 都在写），
      Megatron 同精度是优化，**cast 路径与 colocate 恒为不需要**（结构上无共享内存）。
      **前置已就绪**：`nixl_e2e_5node_fsdp2.yaml` 已端到端跑通
      （FSDP2 + bf16+bf16 + disaggregate，8/8 client × 399/399 keys，拉取 2.91s），
      这正是 `requires_version_barrier` 返回 True 的那一格。

      **已接线**，三处：

      1. **读者上报**：`PSManager.begin_read` / `end_read` / `abort_reads`，
         按 `(version, rollout_instance_id)` 计数。按实例而非单纯总数，
         这样中途死掉的读者不会把计数永久卡在非零 —— `abort_reads` 可整体清掉一个实例。
      2. **发布前等待**：`push_model_state_dict_nixl` 在建 `ModelStore` 之前
         drain 上一版本；超限则按设计标记 INVALID 并告警。
      3. **决策落点**：`validate_config` 解析 `await_readers_before_publish`，
         因为只有它同时看得到 backend 和两个 dtype —— `PSManager` 只收到 `config.psrl`。

      两个实现上的坑，都记在注释里：
      - `count_readers` **故意不加锁**：barrier 在紧循环里轮询它，
        而读者结束读时**要拿**这把锁 —— 加锁会让"正在结束的读者"
        排在"等它结束的 barrier"后面，互相死等。
      - `await_readers_before_publish` 默认值是 `auto` 而不是 `False`：
        L2 判据要显式**关掉**它来观测破坏，而 `False` 必须能与"由我决定"区分开。

      **多节点回归已过**（run 43）：接线后 0 错误、0 版本违例、6 次 pull 全完成，
      rollout pull **0.46–0.48 s**（接线前 0.66–0.90 s，PS 1.14 s）。
      **迁移后复测（run 44）**：仓库与 `verl` 换装到新路径、`verl` 升到较新 commit 后重跑同一配置,
      0 错误、0 版本违例、6 次 pull 全完成,且**两个 step 都跑完**。
      奖励全为 0 不是截断—— `response_length/clip_ratio` 为 0.0625 / 0.4375,
      生成是真实的,两步跑难题拿不到奖励是预期的。

      **单次 run 的拉取耗时不能当作「这个数」**。四次同配置 run 的均值：

      | run | E3 | mean | min–max |
      |---|---|---|---|
      | 44 | 前 | 0.497 s | 0.44–0.59 |
      | 47 | 前 | 0.789 s | 0.66–1.05 |
      | 45 | 后 | 0.817 s | 0.65–1.01 |
      | 46 | 后 | 0.732 s | 0.64–0.84 |

      **两次 E3 之前的 run 彼此就差 1.6×**,所以 run 间方差压过了改动的影响,
      E3 对拉取没有可测量的代价。这条记在这里是因为中间一度把 45/46 的变慢
      归因于 E3,而做 A/B（把 E3 的源码 revert 掉重跑 = run 47）才看出
      **44 才是那个偏快的离群点**。引用单次 run 的区间会反复制造这种假结论。

      该 run 的 `await_readers_before_publish` 解析为 **False** 且**这是对的** ——
      日志里 `colocate_validate_and_train=True`，colocate 下 rollout 全程暂停，
      结构上不存在在飞读者。也就是说真值表在生产配置上确实生效，而非接线失效。

      **L2 判据已达成**（`tests/parameter_server/test_publish_corruption.py`，7 passed）：
      不拦截时，读者在一次分块拉取中拿到同时来自 `v` 和 `v+1` 的字节 —— 既不是旧策略
      也不是新策略，没有任何一步的权重长这样；拦截打开后同样的交错下拿到的是完整的 `v`。

      判据用**内容**而非版本戳：戳是发布时写的，被改写一半的分片照样报告自己的版本，
      `check_pull_versions` 看到的是一次干净的拉取。这正是 `shard_version` 文档里
      写明的盲区。

      交错是**编排**出来的，不是靠调度抢出来的 —— 否则写入落在读窗口内极少发生,
      测试会在没有复现竞争的情况下通过。坚持这一点抓出两处:第一版读者在写者醒来前
      就读完了 8 个分块,拦截报 `NO_READERS`,根本没有可拦的撕裂;重复发布那一组也
      需要读取真正占用时间(传输本就如此),否则每一轮都是 `NO_READERS`。两处都是靠
      断言 outcome 抓到的,只断言「没有撕裂」会一路绿灯。

      **这不能证明的事**：FSDP2 的 optimizer step 与真实 NIXL 拉取在真机上确实重叠。
      那是关于集群时序的论断,也正因如此拦截的必要性由配置决定(`requires_version_barrier`)
      而不是靠实测。这里证明的是:**一旦**重叠,结果就是撕裂,而拦截消除它。

      **读者侧恢复已接线**。此前 INVALID 标记只写不读,撕裂则直接抛异常,两者都
      让一次超预算的发布停在「无法修复」上。现在:

      - `VersionBarrier.invalidated_versions()` 暴露失效集合,有界保留最近 64 个版本。
      - `PSManager.get_nixl_version_state()` 一次返回 `(版本, holder 集合, 失效集合)`,
        替代原来的 `get_nixl_version_holders`。读者的两件事共用一次调用。
      - `nixl_pull_model_core` 返回 `PullVerdict` 而不是 bool:区分「一个字节没写」
        (可以继续用旧版本,设计 D6) 与「混了两个版本」(只能整轮重拉)。
      - `_pull_until_consistent` 在 shared pull lock 内重拉,上限 `max_pull_attempts`
        (默认 3)。重拉在锁内,所以与训练发布天然串行:push 持独占锁期间重拉等待,
        发布完成后读到新版本。超限抛 `WeightRefreshFailed`,协调器隔离该副本。

      重拉**不参与 peer diffusion**。epoch 是 peer 分块通知的键且不可复用,协调同步
      的 epoch 所有副本共用,重拉若沿用会重放陈旧通知,若另造则各副本互不相同、
      每个 follower 都要先等满 `peer_timeout_s`。重拉改走纯上游读取,代价只是放弃
      一个本就罕见的路径上的扩散收益。

      **`conflict_wait_budget_ms` 此前读错了层级**。`ps_manager.py` 用
      `getattr(config.psrl, ...)` 读,而该键在 `psrl.nixl` 下,于是永远落到默认 50 ms,
      yaml 改动不生效。现已改为 `config.psrl.nixl.get(...)`。
- [ ] **E2 未落地，按原设计也无法落地**。`psrl/utils/nixl/staleness.py` 只是一组纯函数
      （`classify` / `recovery_for` / `RegistrationEpoch`），生产零调用，`test_staleness.py`
      测的是这些函数本身，不是任何读路径。

      **原设计的前提不成立。** 「源端比对请求携带的 epoch，陈旧则拒绝」要求源能看到读请求，
      而 NIXL 读是单边 RDMA：源进程不参与，没有任何地方可以比对 epoch。
      `DESC_STALE` / `CHUNK_STALE` 在读失败时也分不开 —— 两者都表现为 `ERR` / `NOT_FOUND`。

      **「重取 descriptor」这条恢复路径已经存在，只是不叫这个名字**：客户端收到
      `UPDATE_INFO` 时（`client.py` 的 `wait_for_update_infos`）重载对端 info、
      `remove_remote_agent` 并 `_drop_pull_engine`。正确的 reshard 不是让读者事后发现陈旧，
      而是在版本边界把所有成员一起推到新切法：
      排空读者 → 全体按新 finest mesh 重注册 → `UPDATE_INFO` 广播 → 新 plan。
      地址失效靠 `remove_remote_agent` 的弱引用（E4 实测 `NIXL_ERR_NOT_FOUND`）兜底，
      不需要 PSRL 自己的 epoch。

      **现状下 RESHARD 准入是坏的**：既有成员的 `wait_for_server_sharding` 因
      `_unified_sharding_dict_fetched` 已为真而直接返回，收不到新切法；服务端
      `notify_all_client_shardings` 发的是未带 kind 的裸 pickle。结果是 joiner 按细网格加入、
      既有成员仍按粗网格注册，plan 按细网格生成。目前不触发，因为 `admit_joiners`
      在生产中无调用方（见 E3）。

      **决策**：先让 `admit_joiners` 只接受 FAST_PATH、拒绝 RESHARD，并删除 `staleness.py`；
      运行中 reshard 作为独立功能按上面的版本边界协议实现，判据是「reshard 前后所有成员
      坐标判据全过、期间无读者读到混合切法」。
- [ ] **E3 成员状态机已落地并单测通过（14 passed），接线待握手循环改造**：
      `psrl/utils/nixl/membership.py`。
      准入复用 A7 的 `classify_join`（不新造一套「是否允许」），
      本模块定的是**何时生效、以及集群因此欠什么**（replan / epoch bump）。
      **离开必须两阶段**：先停止路由（`begin_leave`），等在飞归零后才移除
      （`complete_leave` 在仍有读者时**拒绝**）——
      合并两步会在活跃读上释放内存，而那次读会**成功**。
      **握手循环已改造**（`psrl/utils/nixl/handshake_notif.py`，
      `test_handshake_notif.py` 21 passed）。改造中发现阻塞不在循环结构，而在**线格式**：
      三种握手负载都是 `{client_name: value}` 的 pickle，**结构上无法区分**,
      `isinstance(obj, dict)` 对三者都成立。服务端靠「自己处在哪个阶段」分辨它们 ——
      **时间位置就是类型标签**。只要握手是 lock-step 这成立；而后加入的客户端
      按定义就是乱序的。

      两种失败都**静默**：被错误的 `wait_for_*` 消费掉的通知会进**错误的注册表**
      （一个 `{name: value}` 和另一个长得一样）；没人在等的通知则直接丢失,
      因为 `get_new_notifs` 是**破坏性**的,每条只交给一个调用方。第四个读者
      `wait_for_update_infos` 更糟 —— 它对解析不了的东西**抛异常**而非跳过。

      所以信封带上 kind,**一个解复用 drain** 供四个注册表。单一 drain 不是风格选择:
      三个读者各自轮询同一条破坏性流,先跑的那个就吞掉别人的信。

      per-kind 的「握手已完成」布尔量**删除**了 —— 布尔量只能表达「第一次握手完成」,
      而这正是让后续加入隐形的那个答案;改成会继续增长的 roster。
      `UPDATE_INFO` 的 roster **每轮清空**,因为它问的是「这一轮谁重新注册了」。

      **多节点验证**（run 45）：sharding 1.5s / client infos 0.8s / temp mappings 0.1s,
      0 超时、0 错误、0 版本违例,两个 step 跑完。

      **入口已补上**（`NIXLMetaServer.admit_joiners`，`test_admit_joiners.py` 18 passed）。
      光让通知送达是不够的：unified sharding 与 comm plan 各自**只构建一次**且带
      「already made」断言,所以后到的客户端会躺在 `client_sharding_dicts` 里、
      **不出现在任何 plan 中** —— 不报错,也没人读它。

      `admit_joiners` 做 drain → 问 A7（复用既有 `classify_incoming_client`）→
      记入 `Membership` → replan。**准入不在这里重新推导**:
      规则有第二份拷贝,正是握手路径与弹性路径开始各说各话的方式。

      两个构建函数改成显式 `rebuild` 参数而不是删掉守卫 —— 初始握手里构建两次
      是真 bug（说明步骤乱序）,join 之后重建才是目的,光凭「already made」
      分不出这两件事。

      **写拒绝路径的测试时发现 REJECT 当时根本不可达**：`classify_join` 是拿
      **张量 shape** 去验整除的,而一个 shape 都没传进去,`blocking_dims` 恒为空,
      于是每次 join 都返回 RESHARD —— 不可整除的 mesh 会被放进来,
      然后**所有既有客户端**在重新注册时抛错。全局 shape 没有任何地方存,
      但可还原:分片 extent × mesh,这就是 `registered_tensor_shapes`。

      **L3 判据已达成**（`tests/nixl/test_elastic_join_agents.py`，5 passed）：
      三个**真实 NIXL agent**、DRAM 注册 —— incumbent 完成握手后 joiner 才连上,
      被准入、被重规划,且 incumbent 仍在 plan 里。

      **这不需要多节点,也不需要 GPU**:要证的是握手协议,不是传输。
      通知能否送达、是否被认成 sharding、是否驱动重规划,都与内存在不在 GPU、
      对端在不在另一台主机无关。因此它**不能**证明的是:加入后的 instance
      能被跨主机 RDMA 读到 —— 那是 D1/D4,由生产 run 覆盖。

      harness 本身有两处必须做对,一开始两处都错了:客户端**必须开 progress thread**,
      否则 `send_notif` 只在该客户端自己轮询时才 flush,服务端会超时并报
      「0 个 agent」—— 读起来像连通性故障;以及 joiner 必须**先进入等待**再让服务端
      重新发布,因为两半互相依赖:它拿不到 sharding 就发不出 client info,
      而服务端拿不到 info 就结束不了准入。
- [x] **E4 两半都已落地并实测**。
      `psrl/utils/nixl/partial_md.py`，`test_partial_md.py` 10 passed，
      实测 `bench_partial_md.py`（H20）：

      | 注册区数 | 全量 MD | conn-info only | 压缩 |
      |---|---|---|---|
      | 1 | 2,139 B | 1,640 B | 1.3× |
      | 256 | 110,008 B | 1,644 B | 66.9× |
      | 4096 | **1,734,329 B** | **1,645 B** | **1054×** |

      **conn-info 恒定（1640~1645 B），增长全部来自 descriptor。**
      设计原文引的 406 B/region、883 B、401 B 三个绝对值与本环境不符
      （实测 423 B/region、606 B、1645 B），**但结构性结论成立且更强**。

      万卡量级：单次 join 从 **17.34 GB 降到 16.45 MB**。
      第二段只发给 plan 指明的读者（实测 606 B/读者），
      且**读者数不随集群规模增长** —— 这是测试钉住的性质。

      **第二半（吊销 descriptor）已落地**：`psrl/utils/nixl/leave_protocol.py`，
      `NIXLMetaServer.release_member`，`test_leave_protocol.py` 13 passed +
      `test_release_member.py` 18 passed + `test_revoke_descriptors_gpu.py` 5 passed。

      **先纠正一个连续两轮的错误归因。** 此前写的「本 NIXL 构建缺
      `invalidateLocalMD` 绑定」是错的，错在**探测名写错**：真实名字是
      `invalidate_local_metadata`，`hasattr(nixl_agent, 'invalidate_local_md')`
      当然永远为 False。更直接的反证是 PSRL 自己一直在调它
      （`server.py` 的 `shutdown`），而 `weight_transport.md` §9c 开头的
      早期审计（「NIXL 侧该有的全有」）本来就是对的。
      那个「等 NIXL 补上绑定就失败」的哨兵测试因此永远不会触发，
      是个假哨兵，已删除；替换为直接断言真实绑定存在的测试。

      **读源码后，真正的缺口不是绑定而是确认。**
      `nixl_p2p_metadata_backend.cpp`：

      ```
      invalidateLocal(ip, port) -> sendToPeer(ip, port, "NIXLCOMM:INVL" + myName)
      对端 listener 收到 INVL  -> ctx_.invalidateRemoteMD(myName)
      ```

      即 `invalidateLocalMD` 只是「远程触发对端执行一次 `remove_remote_agent(我)`」，
      干活的始终是持有者本地那一次驱逐。由此三点：
      (a) **不是广播** —— `ip_addr=""` 的广播形式只在中心 MD 存储（etcd/tcpstore）
      下成立，而 PSRL 的 MD 走 server 的 notif 通道 out-of-band 分发，没有中心存储；
      (b) **需要对端监听端口**，而 `NIXLClientInfo` 只发布 `node_ip`；
      (c) **无法确认** —— `worker_.submit` 之后没有完成通知，失败只打日志，
      而第四步释放显存的安全性正好取决于这个确认。

      所以吊销改走 PSRL 已有的 notif 通道：
      `MEMBER_LEAVE`（server → 各 peer agent）/ `LEAVE_ACK`（peer → server），
      由 `Membership.complete_leave` 在确认未齐时**拒绝**完成缩容。
      这条通道本来就在做同一件事 —— `wait_for_update_infos` 对重注册的 peer
      执行的就是 `remove_remote_agent`。

      **实测（H20，`test_revoke_descriptors_gpu.py`）：驱逐之后陈旧 descriptor
      会被拒绝，而不是读到已释放显存。**

      | 用法 | 结果 |
      |---|---|
      | 用退出前的 prepped dlist 新建请求 | `makeXferReq` 拒绝，`NIXL_ERR_NOT_FOUND` |
      | 退出前已建好的 req 句柄 repost | `postXferReq` 拒绝，`NIXL_ERR_NOT_FOUND` |

      机制是 NIXL 对 remote section 的弱引用 + generation 校验
      （`nixl_agent.cpp` 在 `makeXferReq` / `postXferReq` / `getXferStatus`
      三处检查 `remoteSection.expired()`）。
      **所以这条路径不需要 PSRL 自己再加一层 epoch** —— §3.4.5 担心的
      「retry 路径读到已释放显存且成功」在本构建上不成立，
      前提只是「驱逐确实在每个持有者上执行过」，而这正是 ack 保证的。

      **确认的粒度是 agent 而非 client**：`remove_remote_agent` 一次驱逐覆盖
      该 agent 上的所有 client，按 client 收 ack 会等一个第二次调用无法产生的确认。

      **另一条实测修正了接口形状。** `remove_remote_agent` 在本构建上
      **从不报错**：对已驱逐过的 peer、甚至对一个从未加载过的名字，
      都正常返回（probe 验证）。所以「这次驱逐是否真的忘掉了什么」
      在 NIXL 这一层**无法回答**，`revoke_descriptors` 因此不返回布尔值——
      返回一个没有依据的值会把它写进 ack。
      ack 里的 `held` 改为取自 PSRL 自己的连接缓存
      （`_target_client_connected`），那是确实知道的。
      顺带：这也意味着重复吊销天然幂等，丢 ack 后的重试是安全的。

      **退化行为**：任一持有者未确认 → `release_member` 抛 `TimeoutError` 并点名，
      成员停留在 `draining`（不再被路由新读，但显存保留）。
      这是安全的那一侧退化；不安全的那一侧是释放了别人还能寻址的显存。

      **读方的入口是 `absorb_membership_notifs()`**：其余 drain 都在某个 waiter 里，
      而 leave 恰恰在客户端**干活时**到达，不在任何等待里。
      真机测试（`test_elastic_join_agents.py::TestAMemberLeavesOverRealNixl`）
      就是让 incumbent 在后台跑这个 pump 的同时让 server 发起缩容。

      **重发节奏单独一个常数（`LEAVE_RESEND_INTERVAL_S = 1.0`）。**
      最初写成「每轮轮询都重发」，真机跑出来一次被放弃的 1 秒尝试给对端留下了
      **100 多条排队的通告**。消息本身幂等，重发只为补丢失，
      按轮询频率重发就是控制面风暴 —— 万卡量级正是 C2 要避免的那类。

      **「先排空再吊销」这个次序是承重的，不是说法上的。** 初版把 `in_flight`
      的检查交给 `complete_leave`（即在整轮吊销**之后**），review 时发现这会
      在读还在飞的时候就吊销 —— NIXL 的 `getXferStatus` 查的是同一个失效引用，
      于是在飞的读会失败，**恰好破坏排空这一步要保护的那些传输**。
      现在 `release_member` 在 `begin_leave` 之后、发出任何通告之前就拒绝，
      并有一个测试断言「此时一条通告都没发出去」。

      **同 agent 的多 client 必须一起失效。** `remove_remote_agent` 是 agent 级的，
      而 `NIXLMultiStorageClients` 让 actor / critic 共用一个 agent。
      只清自己那份缓存会让兄弟 client 仍以为「还连着」，
      于是跳过 `_ensure_client_info_fetched` 直接对一个已被忘掉的远端发请求。
      `_agent_siblings` 让吊销作用于该 agent 上的每个 client。

      **仍待做**：把两段式接到 `client_info` 的真实扇出路径上
      （当前 `meta=self.agent.get_agent_metadata()` 仍是全量），需与 E3 的事件驱动 join 同批。

### E5 — 写入原子性：乐观提交校验 + peer 副本修复（v2）

**落点**：
`psrl/workers/ps/write_state.py`（纯状态机）、
`psrl/workers/ps/ps_manager.py`（准入 / `begin_write` / 提交校验 / 通知扇出）、
`psrl/utils/nixl/handshake_notif.py`（`PULL_PREEMPT` / `PULL_PREEMPT_ACK`）、
`psrl/utils/nixl/server.py`（`drain_pull_acks`）、
`psrl/utils/nixl/client.py`（通知收发）、
`psrl/utils/nixl/pull_engine.py`（抢占轮询 / suspect 分类 / holder 修复 / pool 预分配）、
`psrl/workers/gen/vllm_extension.py` 与 `vllm_async_server.py`（接线与重拉）、
`psrl/workers/train/engine_train_worker.py`（optimizer 写门闩）、
`psrl/utils/config.py` + `nixl.yaml`（`await_write_gate` 判据）。

**做什么**：

1. **写 epoch**：`begin_write` 在第一次 optimizer 参数写**之前**推进 epoch；
   状态机 `OPEN → CLOSING → WRITING → OPEN(v+1)`。
2. **读者准入**：pull 开始时记录 `start_epoch`；`state != OPEN` 时拒绝，读者保持当前
   一致版本，发布后重拉。
3. **抢占**：向在飞读者发 `PULL_PREEMPT{epoch, version, holders}`，payload 只带该读者
   同节点的 holder。写者最多等 `preempt_ack_timeout_ms`，**不等待数据传输完成**。
4. **读方响应**：在传输等待循环中以 0.5 ms 粒度轮询通知；看到新 epoch 立即回
   `PULL_PREEMPT_ACK`，把未完成 key 记为 suspect，取消在飞请求。
5. **修复**：suspect key 从 certified holder 整 key 重读（NVLink）；全部可修复则落地
   原版本，否则放弃本轮并等 `v+1`。holder 在 `WRITING` 期间被 pin。
6. **提交校验**：`finish_shared_pull` 接受两种证明——读取 epoch == 当前 epoch，或
   本 epoch 内 ack + 全部 suspect key 由 holder 修复。其余一律拒绝，实例不得服务
   混合字节。
7. **池几何**：pull 开始在 `_prepare_pool` 里预扫描计划内所有源并一次性分配 staging
   pool；中途只在 take-over 出现更大 item 时才可能扩容，而扩容已保证不释放在飞句柄。
8. **判据**：`validate_config` 解析 `await_write_gate`：直供 + 同精度为 True；
   cast/FP8、PS 数据面为 False；显式布尔覆盖保留。启用时关闭 E1 的发布期屏障，
   避免两套机制互相失效。
9. **重试不能持锁等待**：被抢占/版本已关闭的读者先退出共享准入（释放 reader count），
   再等下一版本并重新准入。持锁等待会与仍取 exclusive 侧的 publish 形成死锁，
   也会让旧版本的 reader count 无谓地拖住发布。写者在门闩路径下也不再取
   exclusive push lock——匹配精度下 publish 只翻转版本，不移动字节。

**判据**：

- 提交点的真值表覆盖所有 `(pull 起点, ack, repaired)` 组合：只有可证明单版本的拉取
  能提交（`tests/unit/test_write_state.py`）。
- 被抢占的 pull 能从 holder 修复并落地同一版本；无 holder 时抛
  `PullPreempted(repaired=False)` 且不分发任何字节（`tests/nixl/test_pull_preempt.py`）。
- 稳态写者 O(1)：无在飞读者时不枚举 holder、不发通知（`test_write_state.py`；
  `bench_write_gate.py` 实测 10k 实例、0 读者 fanout **0.6 µs**）。
- 扇出 O(readers)、payload 与集群规模无关：10k 实例、1000 读者时 lookups ==
  holders + readers、通知数 == readers、单条 payload < 4 KB（`test_write_gate_fanout.py`）。
- 旧行为回归：`test_pull_engine.py` 的句柄所有权测试（已修复的在飞释放）与
  `MemcpyAgent` 的释放语义仍通过。

**验证**：
- `tests/unit/test_write_state.py`（真值表与 O(1) 稳态）、`test_write_gate_config.py`
  （拓扑/精度判据）、`test_write_gate_fanout.py`（O(readers) 扇出）、
  `tests/nixl/test_pull_preempt.py`（真实 engine 的抢占/修复/拒绝）共 **37 passed**。
- 全量 CPU 回归：`tests/unit + tests/nixl -m cpu_test` **797 passed**（另有 2 条既有的
  source-text 断言与 1 条仅在合并运行时出现的 planner 测试偶发失败，均与 v2 无关）。
- 两节点生产 e2e：10/10 步、step 5 与最终验证完成；写门闩 10 次，
  等待 **0.03–0.09 ms**；26 次 pull（中位 1.05 s），0 抢占、0 提交拒绝、0 错误。
  见 `weight_transport_FINDINGS.md` §10。

**遗留取舍**：
- 没有 holder 的 suspect key 需要等 `v+1`（可用性代价一个 sync 周期），不是正确性代价。
- take-over 仍可能让 staging pool 在 pull 中途扩容；此时由 `_ensure_pool` 的在飞句柄
  保活逻辑兜底（owner 规则已由 `test_a_pool_resize_keeps_a_running_reads_handle` 钉住）。
- **提交判据不使用 engine 报告的 per-shard 版本戳**：rollout 侧戳在 publish 后不刷新
  （D1 已记录），用它比较已发布版本会误拒稳态 pull。epoch/ack/修复认证是权威判据，
  engine 的 `versions` 只用于同一次 pull 内的撕裂检测。

### 测试改造
- [x] **T1 已落地并验证**：`tests/nixl/coord_judge.py`，`test_coord_judge.py` 30 passed，
      且已接入 e2e harness —— 5 节点日志的
      `Gen client N: 399/399 keys match their coordinates` 即该判据在集群上运行的证据。
      两条性质都在：值按全局坐标单射编码（换位可见），
      且 `assert_dtype_capacity` 预先检查目标 dtype 能否精确表示
      （fp8_e4m3 仅 3 位尾数，整数上界 16），避免静默舍入冒充成功。
- [ ] T2 删除 `create_ps_worker_group`
- [x] **T3 colocate 模式已落地并集群验证**：`tests/nixl/config/nixl_e2e_colocate.yaml`。
      **8 train + 8 gen 共卡，exit=0、8 个 gen client 全部 399/399、拉取 0.039s。**

      | 放置 | 拉取耗时 |
      |---|---|
      | disaggregate（跨节点） | 0.589 s |
      | **colocate（同卡）** | **0.039 s** |

      **15.1× —— 这是 §3.7「同卡传输不碰 NIC」的首次端到端实测。**

      **三处由实测暴露的真实约束**（都不是 harness bug，是 colocate 本身的性质）：
      1. **`num_gpus=0.5` 不足以表达共卡**。Ray 据此把**两个 gen actor** 塞到同一张卡
         （8 个 actor 落在 4 张卡），vLLM 的 TP=2 于是试图在一张卡上初始化两个 NCCL rank，
         报 `NCCL error: invalid usage`。配对必须是 **train 配 gen**，
         所以要用 placement group 显式表达，而不能交给调度器。
      2. **`STRICT_SPREAD` 是错的策略**：它要求每个 bundle 独占一个**节点**，
         8 个 bundle 在 5 节点集群上永远 ready 不了，表现为 placement group 超时
         而非任何与 colocate 有关的错误。改用 `PACK`。
      3. **vLLM 的默认 `gpu_memory_utilization`（>0.9）在共卡下不可用**。
         它按**总显存**比例预留并在启动时检查空闲量，而 train rank 已占约 25 GiB，
         于是报 `Free memory on device cuda:0 (69.89/95.22 GiB) is less than desired`。
         **这正是设计里「colocate 必须先 offload 再让 rollout 分配 KV cache」的账
         以报错形式出现**。harness 无切换路径，故两侧各占一半（`gpu_memory_utilization: 0.5`）。

      **顺带验证了 descriptor 规划的 §2.3**：TP 匹配（train TP=2 == gen TP=2）时
      输入维无需 LCM 重分片，**最差分片仅 4 行**（对比不匹配时的 4096 行），
      原地提交成本 0.9ms。
- [ ] **T4 矩阵与 CI 分档已显式化并自检（16 passed），FSDP2 格已补齐，余一格待 cast 路径**：
      `tests/nixl/test_config_matrix.py`。
      把「哪些组合被覆盖」从「读遍所有 YAML 才知道」变成一张可断言的表：
      每个 cell 要么指向一个真实配置并与之对账（后端/精度/放置必须一致），
      要么**写明为什么还跑不了**——空槽没有理由会被测试判为疏漏而非记录。

      | 档 | 组合 | 状态 |
      |---|---|---|
      | 每次 CI | megatron + bf16+bf16 + **colocate** | ✅ `nixl_e2e_colocate.yaml` |
      | 每次 CI | megatron + bf16+**fp8** + disaggregate | ⏳ cast 路径未接入 harness（A9 仅单测级） |
      | 每次 CI | **fsdp2** + fp32+bf16 + disaggregate | ⏳ 无 FSDP2 e2e 配置 |
      | 每日 | megatron + bf16+bf16 + disaggregate（TP 不匹配） | ✅ `nixl_e2e_5node_direct.yaml` |
      | 每日 | fsdp2 + bf16+bf16 + disaggregate | ✅ `nixl_e2e_5node_fsdp2.yaml`，**8/8 client × 399/399 keys，拉取 2.91s** |
      | 每日 | megatron + fp8+fp8 + colocate | ⏳ 需 TE `fp8_param` + cast 路径 |

      另有两条结构性断言：**每次 CI 档不得超过 4 格**（每格都是一次多节点运行，
      增长直接吃迭代速度）；**PS 版与直供版配置只能差 `direct_supply` 一个键**，
      否则两者之间的性能差异无法归因。

---

## 11. 与设计文档的对照：哪些已落地，哪些没有

「生产接线」= 被 `psrl/` 下非测试文件 import **且**在活跃同步路径上
（rollout 侧 `vllm_extension.nixl_pull_model_core` 与 `vllm_async_server.pull_model_for_sync`，
trainer 侧 `client.py` 的注册与发布）。只在测试里被调用的模块算「仅模块」。

### 11.1 数据面：已落地

`weight_transport.md` §5 优化项清单的 0–6d 与 7b、9、9b、9d-a、9e 在生产路径上运行：

| 设计项 | 落点 | 验证 |
|---|---|---|
| 1 prepped descriptor | `pull_engine.py` 每源每次 pull 一个 handle | L2 |
| 2 colocate 不碰 NIXL 注册 | ❌ **未落地**，见 §11.2 | — |
| 3 strided / 连续段配对 | `read_geometry.py` | L2 |
| 4 源头 FP8 wire | `block_fp8.py`，trainer 发布 FP8 副本与 scale | L2 17 格，L4 235B |
| 5 上游分波与 peer 读重叠 | `pull_engine.py` 的 `_waves_of` | L3 |
| 5b post / wait 拆分 | `pull_engine.py` 的 `_start` / `_finish` | L1 |
| 6 MoE 分组 | `moe_grouping.py` | L2，235B 下 1131 key |
| 6b 控制面下热路径 | plan 握手期编译，稳态零 RPC | L2（`test_the_pull_path_makes_no_control_plane_calls`） |
| 6c / 6d chunk 相关 | **已删除**：引擎按源一次提交，无 chunk 概念 | — |
| 7 三阶段去冗余 + 同节点 peer | `comm_plan.py` + `pull_engine.py`，默认开 | L3 |
| 7b 发布点挂步边界 | `step_boundary.py` | L2 |
| 8c 版本级等待 | `ps_manager.py` 的 `VersionBarrier`，`conflict_wait_budget_ms` | L2 |
| 9 trainer 直供 / 不建 PS | `ray_trainer.py:2005` 的 `ps_mode=nixl_direct` 分支 | L2 / L3 |
| 9d-a 配置期防御 | `config.py` 的 dtype 读取、offload 双 flag、LoRA `merge=false` 拒绝 | L1 |
| 9e 删 FSDP1 | 已删 | L1 |
| D6 降级继续生成 | `vllm_async_server.py:1109` 返回真实版本而非 assert | L2 |

### 11.2 未落地的设计项

**(a) 有模块但零生产 import，且当前架构已不需要**

| 项 | 为什么不需要了 |
|---|---|
| B2 `slot_ack` | cast 产物改为常驻发布副本，在推送锁内刷新，没有可复用的 slot（主文档 §3.4.2） |
| D5 `chunk_offset` | chunk 概念已删除 |
| D1 `source_resolution` | 选源由 planner 的 designate 次序承担 |

**(b) 有模块但零生产 import，仍然需要**

| 项 | 缺它会怎样 | 依赖 |
|---|---|---|
| D2 `holder_index` / D4 `holder_roles` | 没有「谁持有哪个版本的哪一段」的索引。多模型隔离改由客户端名承担（§11.1），它只剩「档位 B 的 CPU holder」这一个用途 | — |
| F1 `delta_plan` | 跨 DC / 慢链路的增量传输，设计上就是 opt-in | — |

**(c) 完全没有实现**

| 项 | 现状 |
|---|---|
| **stage 边界 offload** | ✅ 已落地：`stage_offload.py` `TrainerStage` + `StorageOffloader`，`weight_backup.py` `WeightBackup`，接入 `engine_train_worker.py` + `ray_trainer.py`；`colocate_switch.py` 编码直供正确切换顺序；`registration_policy.py` 按 plan 决定是否 RDMA 注册。实测 H20 PCIe 53–57 GB/s（D2H/H2D），18 GB 模型 offload < 0.35s。`test_stage_offload_gpu.py` 14 passed，`test_weight_backup_gpu.py` 9 passed，`test_colocate_switch.py` 14 passed，`test_registration_policy.py` 17 passed，`test_colocate_switchback.py` 8 passed。 |
| **3.10 多角色的生产接线** | 数据面已按模型隔离并有 CPU 测试（§11.1），但生产只有 actor 一个模型走 NIXL，critic / ref / RM 仍各自读 checkpoint |
| **3.11 LoRA 合并路径** | 只做了「`merge=false` 拒绝」这一半，`merge=true` 的端到端没跑过 |
| **3.8.1 FP4 / MXFP8** | 只有 blockwise FP8。MXFP8 需要 SM100+ |
| **零拷贝 FP8-native** | TE 的 blockwise recipe 需要 CUDA ≥ 12.9，本集群 12.8。当前 FP8 训练走「反量化 + 重量化进发布副本」 |

### 11.3 验证项对照（主文档 §7.1 的 25 条）

已覆盖：1（零额外存储）、2（流量与重叠率，含 235B）、5（注册有效性三条）、
7（atomic group，FP8 的 weight 与 scale 同组）、8（等价性，量化器与 vLLM 逐位一致）、
**9 与 17a（故障注入：源失效改读 DP 副本，整组失效不动权重）**、
**13 的第三条（缩容：吊销后陈旧 descriptor 被拒，未确认时拒绝释放）**、
10（扩散有效性，含 235B 的结构性上界）、**13 的前两条（后加入的 instance 被准入并拉到正确字节）**、
14（精度矩阵，34 格）、16（后端对称性）、18（布局转换产物时效性）、
19（并行度不匹配的 shard 代数）、**22（role 隔离：两个模型同名 key 互不串读）**。

未覆盖：3（单源瓶颈随 replica 数的亚线性）、4（pause 窗口与冲突等待分布）、
6（colocate 切换的 KV cache 可用量）、11（四条运行时断言各一个反例）、
12（`is_rollout_mode_in_actor=True` 全流程）、13 的最后一条（非整除 join）、
15（LoRA 两条路径，按要求不做）、20（colocate buffer 上界）、21（多角色切换的生产编排）、
23（epoch 与 version 两段校验）、24（档位 B 的 GPS 一致性）、25（长跑 logprob 不漂移）。

## 12. 剩余工作

数据面（直供、去冗余、发布副本、cast、block FP8）已接入生产并通过 §9 所列验证，
实测见 `weight_transport_FINDINGS.md`。以下按「解锁的能力 ÷ 改动面」排序，每项给判据与最轻验证档位。

| # | 项 | 做什么 | 判据 | 档位 |
|---|---|---|---|---|
| W5 | ~~cost model 对齐实测~~ | ✅ `BW_NIC_PAIR_GBS=49`（单对实测）、`BW_NIC_NODE_GBS=386`（8 对聚合实测）、`BW_INTRA_NODE_GBS=399`（同节点 CUDA IPC 实测）写入 `test_cost_model.py`，旧的按理论公式计算的 `BW_NIC_GBS=170` 全量替换为实测值，并加了 per-agent 与 node-aggregate 的明确注释 | `test_cost_model.py` 20 passed | 已完成 |
| W6 | ~~状态表与 import 图对齐~~ | ✅ 同 W5 一并落地——`TestHardwareConstants.test_nic_bandwidth_constants_match_measurements` 断言三个实测常数，`TestCostModelShape` 里的 per-agent 比较改用 `BW_NIC_PAIR_GBS`，节点聚合容量计算用 `BW_NIC_NODE_GBS` | 同上 | 已完成 |
| W7 | D7/D8 接入生产 | E3 准入已接生产并有真机测试；E4 两半（join 扇出 + 缩容吊销）均已接生产，`test_revoke_descriptors_gpu.py` 5 passed；其余（D7/D8 生产故障注入）为纯模块 | 故障注入 | 多节点 |
| W8 | 删 PS（D3/T2） | 需项目主人定迁移节奏 | — | — |
| W9 | ~~block FP8 的细化 scale 网格~~ | ✅ rollout 在 scale sharding 上声明 scale tile，trainer 按 128 tile 量化后复制 scale（与 vLLM loader 的 `repeat_interleave` 逐位相同） | MoE Megatron TP2·EP2 → vLLM TP4（EP1）FP8 格通过 | 已完成 |
| W11 | ~~MoE 下 vLLM DP > 1 的 L2 覆盖~~ | ✅ 根因是 NCCL 2.28.9 让 vLLM 禁用 pynccl，改用 2.29.7，harness 的 EP 规则改为 EP = DP × TP | EP4 跨 DP 的三格通过 | 已完成 |
| ~~W10~~ | ~~规模验证~~ | ✅ Qwen3-235B-A22B（470 GB bf16，94 层，128 experts）4 节点通过，见 `weight_transport_FINDINGS.md` §4b。**1TB 级仍未测**，需要更多节点 | — | 已完成（235B 档） |
| ~~W12~~ | ~~大 MoE 的拉取并发度（根因定性完成）~~ | ✅ 根因已通过系统性实测完全锁定。235B 稳态每节点 107 GB/s（28%）的完整根因链：① EP16 拓扑让每个 gen GPU 只与 1 个 train GPU 通信，链路上限 49 GB/s；② UCX NIC 亲和性不均衡——8 对并发中 GPU 0/3/4/5 稳定达到 49 GB/s，GPU 1/2/6/7 只有约 21 GB/s，8 对聚合 278 GB/s（71% 效率），而非理论 8×49=392 GB/s；③ EP 串行化——每个 gen GPU 按 key 顺序读 36 个专家，链路并非持续满载。实测排除四条候选：波数（1/4/8 无差别）、planner 选源（一一匹配慢 1.6 倍）、提交开销（0.58 ms vs 158 ms）、段大小（4 KB→2 GB IB 与 NVLink 全部平坦，206 KB MoE 专家段无惩罚）。**改善路径是改变拓扑**（expert coalesce / 减小 vLLM EP / train 侧 EP 分组连续发布），与协议层无关。见 FINDINGS §1b、§1c | 根因已定性；改善路径待实测（2 节点） | 根因完成 |
| W15 | ~~FP8 发布点的重量化成本~~ | ✅ 融合 Triton kernel（`fp8_kernels.quantize_into`）单遍读 bf16、写 FP8 + scale，3169 GB/s，235B 下每 rank 从 513 ms 降到约 35 ms（14.8×）。已达目标（≤ 稳态 sync 的 20%；稳态 ~1.5 s，20% = 300 ms，35 ms = 2.3%）。`test_fp8_kernels_gpu.py` 22 passed，含 roofline 断言。选择性重量化（只重量化变化过的 key）可在此基础上进一步优化，但已超出当前目标，按需实施 | 发布刷新 ≤ 稳态 sync 的 20% | 已完成 |
| W13 | ~~colocate 走 CUDA IPC（设计项 2 / A8）~~ | ✅ `_should_register_rdma()` 已接入 `register_local_tensors` 的两处 `_register_memory` 调用点：当 comm plan 证明所有读者都在同节点时跳过 RDMA 注册；否则（无 plan 或存在跨节点读者）回落到注册。同时 `registration_policy.py` 已由 `_is_reader_of()` 接入 E4 两段式 join 路径。`test_registration_policy.py` 增加 3 个 wiring 测试（共 20 passed） | colocate 切换路径上 `register_memory` 调用数 == 0（comm plan 为全同节点时）；测试 20 passed | 已完成 |
| W14 | ~~多角色与 role 隔离（设计 §3.10）~~ | ✅ `WeightBackup` ref 路径落地：`engine_train_worker.py` 增加 `capture_ref_backup()` / `compute_ref_log_prob()` / `mark_modified()` 三个接入点；`ray_trainer.py` 在 LoRA path（`ref_in_actor=True`）的 `init_model` 后调用 `capture_ref_backup()` 并将 `_compute_ref_log_prob` 的 LoRA 分支改为 `actor_wg.compute_ref_log_prob(batch)`。代价：init 一次 D2H，每 ref pass 一次 H2D+H2D（~46 GB/s/rank），替代一整份 GPU 模型副本。`test_weight_backup_gpu.py` 9 passed，含 logits 等值校验与 PCIe 带宽断言。非 LoRA 的 `ref_policy_wg` 路径不变，向后兼容 | 两个角色各自推进版本而互不影响；LoRA ref 路径用 host snapshot 而非独立 GPU | 已完成 |
| ~~W16~~ | ~~DCP async save 与 NIXL 共存（nvrx 集成）~~ | ✅ nvrx 0.7.0 `AsyncCallsQueue(persistent=True)` spawn 路径与 NIXL 进度线程共存；verl patch 修 `_dispatch_finalize` + `save_dist_checkpointing`；config 移除阻断断言；进度线程始终开启；`scripts/install_basic.sh` 和 `docker/Dockerfile` 均加入安装步骤。verl patch 中的 `use_per_rank_checkpoint` 钩子已移除。实测 30s：113–119 GB/s，0 错误，PASS。真实栈端到端验证（Ray 集群到 `save_freq` 边界）待完成后可删 `PSRL_PerRankMegatronCheckpointManager` | `repro_dcp_nvrx.py` PASS；真实栈 e2e 待验 | 合成层已完成，真实栈待验 |
| ~~W17~~ | ~~rollout 版本感知选源~~ | ✅ PSManager 维护 `(version, frozenset[trainer_names])`（`_nixl_version_holders`），`push_model_state_dict_nixl` 每步刷新，新增 `get_nixl_version_state()` 原子 RPC（后扩展为同时返回失效集合）；`vllm_async_server.nixl_pull_model` 拉取持有者集合并透传；`pull_engine.pull(version_holders=...)` 在任何字节搬动前按集合过滤 upstream，`version_holders=None` 时旧行为保持。每步一次 O(1) 新 RPC，选源时 O(S) 集合查找（S=每 key 的 source 数，通常 1-2），与集群规模无关 | `tests/unit/test_version_aware_source_selection.py` 16 passed | 已完成 |
| ~~W18~~ | ~~peer diffusion 拓扑感知 designate 选择~~ | ✅ `_make_peer_groups` sort key 从 `(upstream_load, name)` 改为 `(-trainer_link_priority, upstream_load, name)`：和 trainer 有 NVLink 的 instance 优先成为 designate，IB 带宽只用于必须的跨节点流量，NVLink 在节点内自由使用；负载均衡作为同优先级内的第二排序键保留；规划期 O(G log G) 每组，总复杂度不变，无运行时开销 | `tests/nixl/test_topology_aware_designate.py` 15 passed；scale test：1024 instance × 200 key 的 `_make_peer_groups` < 1.15s（< 3s 限制） | 已完成 |
| ~~W19~~ | ~~删除 `_DONE` barrier~~ | ✅ 从第一性原理确认 `_DONE` 不解锁任何操作（follower 在 `_await(_DONE)` 前已完成自己所有 peer 读，buffer 生命周期由外部调度控制），删除 `_DONE` 常量、designate 侧 `publish(_DONE)` 和 follower 侧 `_await(_DONE)` 及超时 warning；`peer_timeout_s` 保留用于 `_READY` 等待 | 60 tests pass（`test_pull_engine.py` + `test_version_aware_source_selection.py`） | 已完成 |
| W16-真实栈 | nvrx async save 真实栈 e2e 验证 | Ray 集群跑到 `save_freq` 边界：确认 `.distcp` 目录完整、NIXL 0 错误、能 resume；通过后删 `PSRL_PerRankMegatronCheckpointManager` | resume 后训练继续、权重与保存前一致 | 多节点 |

**顺序约束**：W5、W6 独立且便宜，W9 独立。W10 依赖足够的节点，W7 的故障注入最好在 W10 的规模上跑。
**复现命令**见 `weight_transport_FINDINGS.md` §8。

---

## 13. 与主设计文档的对应关系

| 本文阶段 | 设计项 | 主文档章节 |
|---|---|---|
| A1 | 1 | §3.6、§3.6.3 |
| A2 | 3 | §3.6、§3.5.5(b) |
| A3 | 5b | §3.6.4 |
| A4 | 6c（部分） | §3.5.5 末段 |
| A5 | 9d-a | §3.8.0、§3.9.3c、§3.9.2.1、§3.11 |
| A6 | 9e | §3.9.1 |
| A7 | 9c-a | §3.3.1d 第二层 |
| A8 | 2 | §3.7.2.1、§3.7.3、§3.3.1a |
| A9 | 4 | §3.8 |
| B1 | 6d | §3.6.1 |
| B2 | 5 | §3.6、§3.6.0、§3.4.2 末段 |
| B3 | 6c（剩余） | §3.5.6、§3.6.5 |
| C1 | 6 | §3.5.5 |
| C2 | 6b | §3.2.1 |
| C3 | 7b | §3.5.2.2 |
| D1 | 7 | §3.5、§3.5.2、§3.5.3、§3.5.4 |
| D2 | 8 | §3.4.6、§3.10.3 |
| D3 | 9 | §3.1、§3.2、§3.3 |
| D4 | 9b | §3.3.1a |
| D5 | 7c | §3.5.2.3 |
| D6 | 8d | §3.5.7 末段 |
| D7 | 8b | §3.5.7 |
| D8 | 9d-b | §3.9.1 |
| E1 | 8c | §3.4.2.1、§3.8.0a |
| E2 | 9c-a2 | §3.3.1d、§3.7.3.0a |
| E3 | 9c-b | §3.3.1d 第一层 |
| E4 | 9c-c | §3.3.1d 第三层与末段 |
| F1 | 10 | §5 第 10 项 |
| 测试改造 | — | §7.0 |

**覆盖性核对**：设计项 0–10（含 5b/6b/6c/6d/7b/7c/8b/8c/8d/9b/9c-a/9c-a2/9c-b/9c-c/9d-a/9d-b/9e）
共 **26 项**，本文覆盖 **26 项**（0 已完成，其余 25 项分布在 A–F）。
§5.1 的「待优化清单」按设计决定**不纳入**，不在本文范围内。
