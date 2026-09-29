# PSRL Codebase Map

> **Purpose**: Fast-lookup reference for the entire PSRL project. Read this file first to orient yourself in any module.
> **Last updated**: 2026-07-26 | **Total**: 213 Python files in `psrl/`, ~100 test files in `tests/`, config-as-code via veRL re-export
>
> **Major shifts since 2026-05**: rollout now runs through the vendored **SMG** gateway (Rust router) with **SessionRouter + TITO** session capture; sample data moves over **TransferQueue** instead of Ray-serialized batches; KV cache offload/reuse via **LMCache**; **RolloutCoordinator** (mixin-composed) replaces the old `rollout_coordinator.py`; a unified **engine train worker** supersedes the separate FSDP/Megatron workers; a standalone **reward-model service** and **gen reward functions** were added; **resource elasticity** (`elastic_rm`) and a **gRPC PSManager** service landed. See §15 for the terminology glossary.
>
> **Latest changes (2026-07-25/26, commit `11b6d65`)**: the monolithic `psrl.yaml` was **split into 16 per-topic group files** plus the 6 existing `rollout_coordination/` ones (§4) — note `elastic_rm` moved under `psrl.deployment.elastic_rm`; **fine-grained rollout/train overlap** landed as a `StepStrategy` (`psrl.fine_grain_overlap`); pre-SMG dead code (gen worker, FSDP/Megatron train workers, Python router/route-strategy/request-queue, mini-SWE loop v0) moved to top-level `deprecated/`; the real test suite is `tests/` (pytest `testpaths`), not `unit_tests/`.

---

## 1. What Is PSRL

PSRL (Post-training Streaming RL) is a distributed LLM post-training framework. Core innovations:

- **Asynchronous training**: rollout generation and PPO training run concurrently, connected via a parameter server
- **Staleness control**: multi-buffer system tracks model versions; training consumes rollouts within an acceptable staleness window
- **Streaming rollout via SMG**: a Rust routing gateway dispatches live inference requests to vLLM replicas with PSRL-aware worker selection, partial-rollout loopback, and weight-version gating
- **GPU-direct communication (NIXL)**: model weights transfer GPU→GPU bypassing CPU/Ray serialization
- **TransferQueue data plane**: rollout/reward/train components exchange lightweight metadata references instead of serializing whole batches
- **Resource elasticity**: GPUs are transparently re-shared across train/gen/eval/reward via TMS memory pause/resume
- **Pluggable parallelism**: FSDP and Megatron-LM for training (behind one engine worker), vLLM for inference

---

## 2. System Architecture (Data Flow)

```
                         ┌──────────────────┐
                         │   main_ppo.py     │  ← Hydra entry point
                         │   TaskRunner      │  ← Ray driver; also inits TransferQueue
                         └────────┬─────────┘
                                  │
         ┌────────────────┬───────┼────────────┬──────────────────┐
         ▼                ▼       ▼            ▼                  ▼
  ┌────────────┐  ┌───────────┐ ┌────────┐ ┌──────────────┐ ┌──────────────┐
  │DataProcessor│  │ PSManager │ │Reward  │ │RolloutGateway│ │AgentLoopMgr  │
  │ (Ray actor) │  │(+ gRPC svc)│ │Manager │ │(SMG + Session│ │(multi-turn / │
  └─────┬──────┘  └─────┬─────┘ │/Worker │ │  Router)     │ │ session)     │
        │               │       │/RM svc │ └──────┬───────┘ └──────┬───────┘
        │        ┌──────┼───────┴───┬────┘        │                │
        ▼        ▼      ▼           ▼             ▼                ▼
   ┌────────┐ ┌────────┐  ┌──────────────┐  ┌───────────────────────────┐
   │GenWorker│ │GenWorker│ │ EngineTrain  │  │ RolloutCoordinator         │
   │(vLLM)  │ │(vLLM)  │  │ Worker(FSDP/ │  │ (stats/pause/weight-version│
   └────────┘ └────────┘  │  Megatron)   │  │  loops driving SMG)        │
        │          │      └──────┬───────┘  └───────────────────────────┘
        └──────────┴──── NIXL ───┘   (GPU-direct model push/pull)
        └─────── TransferQueue ───────── sample fields (prompt/response/reward/logp) ──────┘
```

**Loop**: EngineTrainWorker pushes model → PSManager bumps version → RolloutCoordinator gates SMG on the new version → GenWorkers pull weights via NIXL → SMG dispatches requests → GenWorkers generate → RewardManager/Worker scores → fields land in TransferQueue + staleness buffer fills → DataProcessor batches → Trainer calls update → repeat

---

## 3. Directory Tree (Annotated)

```
psrl/
├── __init__.py
├── grpc/                         # ★ NEW: gRPC service surface for PSManager
│   └── ps_manager_service.py     # Serves admission/reserve/status over gRPC (used by SMG router)
│
├── trainer/                      # ★ Orchestration layer
│   ├── main_ppo.py               # Hydra entry: TransferQueue init → TaskRunner → trainer.fit()
│   ├── constants_ppo.py          # Env vars, Ray runtime config, resource names
│   ├── ppo/
│   │   ├── ray_trainer.py        # PSRL_RayPPOTrainer — main training loop
│   │   ├── strategies/           # ★ Pluggable training step orchestration
│   │   │   ├── base.py           # StepStrategy(ABC), STAGE_META, build_step_strategy
│   │   │   ├── full_batch.py     # FullBatchStepStrategy — default full-batch step
│   │   │   └── fine_grain_overlap.py  # FineGrainOverlapStrategy — chunk-pipelined step
│   │   └── utils.py              # Helpers + PSRL_Role enum (moved here)
│   └── config/                   # ★ Configuration system (see §4)
│       ├── __init__.py           # Re-exports ALL veRL trainer configs (AlgoConfig etc. come from veRL)
│       ├── cost_model/analyze.py # Performance cost model analysis
│       ├── throughput_model/     # Per-model throughput fit JSONs (Qwen2.5-7B, Qwen3-8B)
│       ├── hydra_plugins/psrl_searchpath.py  # Hydra search-path plugin (adds veRL + PSRL config dirs)
│       ├── psrl/                 # ★ RESTRUCTURED: psrl.yaml split into per-topic group files
│       │   ├── psrl.yaml         # Thin root: `defaults:` list + a few scalars (staleness, ps_mode, ...)
│       │   ├── deployment.yaml   # Node/GPU layout per role + heterogeneous_rollout + elastic_rm.*
│       │   ├── rollout_gateway.yaml   # SMG gateway process knobs (concurrency, RUST_LOG, post actors)
│       │   ├── server_rollout.yaml    # HTTP rollout service entrypoint for agent loops
│       │   ├── status_collection.yaml # Engine→coordinator→router status sync + stats_recorder
│       │   ├── fine_grain_overlap.yaml# ★ NEW: granularity / multiplier / overlap_scope
│       │   ├── lmcache.yaml, tms.yaml, nixl.yaml       # KV offload / memory pause-resume / RDMA
│       │   ├── agentic_rl.yaml, checkpoint.yaml, broadcast_init.yaml
│       │   ├── group_post_process.yaml, buffer_post_process.yaml
│       │   ├── log_prob.yaml, memory_logger.yaml, profile.yaml
│       │   └── rollout_coordination/*.yaml  # routing / sync_and_mig / partial / redundant /
│       │                                     #   proactive_filter / session strategies
│       ├── critic/, data/, reward/, rollout/ # component group configs
│       └── ppo_trainer.yaml / ppo_megatron_trainer.yaml  # top-level templates
│
├── workers/                      # ★ Distributed workers (Ray actors)
│   ├── gen/                      # --- Generation / Rollout ---
│   │   ├── vllm_async_server.py  # PSRL_vLLMHttpServer / PSRL_vLLMReplica: gRPC vLLM engine server for SMG
│   │   ├── vllm_rollout.py       # vLLM rollout wrapper
│   │   ├── vllm_extension.py     # vLLM engine extensions
│   │   ├── rollout_gateway.py    # RolloutGateway Ray actor: launches SMG Router + SessionRouter subprocs
│   │   ├── session_router.py     # SessionRouter (uvicorn): session-scoped OpenAI API, hang/continue state
│   │   ├── smg_adapter.py        # SMG glue: RouterArgs build, endpoint paths, weight-version payloads
│   │   ├── rollout_scheduler.py  # RolloutScheduler(AsyncScheduler): custom vLLM v1 scheduling
│   │   ├── stats_collector.py    # EngineStats: per-replica rollout stats
│   │   ├── stats_recorder.py     # StatsRecorder: periodic per-replica stats JSONL dump
│   │   ├── zmq_queue.py          # ZMQPush/PullQueue: cross-process stats/command transport
│   │   ├── utils.py              # RolloutInstanceId + shared gen constants
│   │   └── rollout_coordination/ # ★ NEW: RolloutCoordinator (mixin composition)
│   │       ├── coordinator.py    # RolloutCoordinator: composes all mixins, public API (~838 lines)
│   │       ├── base.py           # CoordinatorBase: HTTP gateway helpers + routing-loop control
│   │       ├── command_loop.py   # CommandHandlerMixin: consumes command queue
│   │       ├── status_loop.py    # StatusMixin: polls SMG stats / routing-loop status
│   │       ├── sync_and_migrate/ # weight-sync + request-migration strategies
│   │       │   ├── sync_and_migrate_mixin.py  # shared sync/migrate helpers
│   │       │   ├── greedy.py     # GreedySyncMixin (sync all instances ASAP)
│   │       │   └── status_based.py # StatusBasedSyncMixin (sync gated on instance status)
│   │       └── session/          # session hang/continue scheduling
│   │           ├── base.py       # SessionScheduler ABC + InstanceCapacity/SessionInfo dataclasses
│   │           └── thunder_agent.py # ThunderAgentScheduler: KV-capacity hang/continue (ThunderAgent port)
│   │
│   ├── train/                    # --- Training ---
│   │   ├── engine_train_worker.py # ★ NEW unified worker (fsdp_/megatron_ workers moved to deprecated/)
│   │   └── base_train_worker.py  # PSRL_BaseTrainWorker: NIXL push, version mgmt
│   │
│   ├── ps/                       # --- Parameter Server ---
│   │   ├── ps_manager.py         # PSManager (~1373 lines): version store + staleness inventory
│   │   ├── ps_storage_worker.py  # PSStorageWorker: holds model shards on GPU
│   │   ├── ps_worker_group.py    # Group of PS storage workers
│   │   ├── staleness_controller.py # StalenessInventory + StalenessBuffer (~1365 lines)
│   │   ├── request_status_tracker.py # PSRL_RequestStatus lifecycle state machine
│   │   └── broadcast.py          # Model broadcast utilities
│   │
│   ├── reward/                   # --- Reward Computation (expanded) ---
│   │   ├── reward_manager.py     # RewardManager (~1138 lines): orchestrates reward pipeline over TransferQueue
│   │   ├── reward_worker.py      # ★ NEW: Ray reward worker (TransferQueue producer/consumer)
│   │   ├── reward_protocol.py    # RewardModelRuntimeInfo dataclass (gateway URL + tokenizer)
│   │   ├── reward_loop/          # Rule/score reward algorithms
│   │   │   ├── base.py           # RewardManagerBase
│   │   │   ├── naive.py, dapo.py, prime.py, gdpo.py  # algorithm variants
│   │   │   ├── gen.py            # GenRewardManager (routes to SMG reward-model gateway)
│   │   │   └── registry.py       # RewardLoop factory registry
│   │   ├── gen_reward_function/  # ★ NEW: pluggable generative-RM scoring fns
│   │   │   ├── base.py, registry.py  # GenRewardFunctionBase + @gen_reward_func decorator
│   │   │   ├── default_gen_rm.py, skywork_rm.py  # concrete reward functions
│   │   └── reward_model/         # ★ NEW: reward-model inference service
│   │       ├── manager.py        # RewardModelManager (owns replicas + coordinator)
│   │       ├── coordinator.py    # RewardModelCoordinator (weight sync / routing)
│   │       ├── replica.py        # RewardModelReplica (vLLM-backed RM instance)
│   │       └── gateway.py        # RM gateway wiring
│   │
│   ├── agent_loop/               # --- Multi-Turn / Session Agent Loop ---
│   │   ├── manager.py            # PSRL_AgentLoopManager (~1708): dispatch, chunk emission for
│   │   │                         #   fine-grain overlap, distributed HTTP POST actor pool
│   │   ├── worker.py             # AgentLoopWorker: single worker process
│   │   ├── gateway_client.py     # Client for SMG rollout gateway
│   │   ├── sticky_session.py     # Session affinity for multi-turn
│   │   ├── prometheus_utils.py   # Monitoring metrics
│   │   ├── harness/              # Harness protocol/config/adapters + immutable task contract
│   │   ├── agent_data/           # Data structures for agent interactions
│   │   │   ├── base.py, conversation_agent_data.py, tool_agent_data.py
│   │   │   └── mini_swe_agent_data.py  # MiniSWEAgentData (patch + grading state)
│   │   └── loops/                # Agent loop implementations
│   │       ├── base_agent_loop.py, generate_agent_loop.py, batch_generate_agent_loop.py
│   │       ├── multi_turn_agent_loop.py, multi_turn_completion_agent_loop.py
│   │       ├── session_agent_loop.py    # ★ NEW: SMG SessionRouter + TITO-backed loop
│   │       ├── harness_agent_loop.py    # Generic task/sandbox/harness/TITO lifecycle (template method)
│   │       ├── mini_swe_harness_agent_loop.py # Mini-SWE task hooks: env, patch, clean grader
│   │       ├── mini_swe_agent_loop_v1.py# ★ CURRENT SWE-bench loop (pluggable sandbox + async rollout)
│   │       └── utils.py
│   │
│   └── config/                   # Worker-level config dataclasses (trimmed)
│       ├── model.py              # ModelConfig
│       ├── reward_model.py       # RewardModelConfig
│       └── rollout.py            # RolloutConfig
│                                 # (actor/critic/engine/optimizer now sourced from veRL)
│
├── tools/                        # ★ Tool-use infrastructure
│   ├── base.py                   # BaseTool abstract class
│   ├── function_tool.py          # Function-call tool wrapper
│   ├── sandbox_fusion_tool.py    # SandboxFusion code execution tool
│   ├── utils.py                  # Tool utilities
│   ├── mcp_clients/manager.py    # MCP client management
│   └── tool_parser/              # Parse tool calls from LLM output
│       ├── base.py               # ToolParser ABC + registry
│       ├── hermes_tool_parser.py, xml_fc_tool_parser.py
│       └── gemma4_/gpt_oss_/qwen3_coder_tool_parser.py  # model-specific parsers
│
├── environments/                 # ★ Training environments
│   ├── base.py, tool_env.py
│   └── mini_swe_env.py           # MiniSWEEnvironment (SWE-bench task parsing + config merging)
│
├── utils/                        # ★ Shared utilities
│   ├── config.py                 # validate_config() (cross-group config checks) +
│   │                             #   resolve_fine_grain_chunk_size()
│   ├── transferqueue_utils.py    # ★ NEW: TransferQueue field read/write helpers
│   │
│   ├── kv_cache/                 # ★ NEW: LMCache integration
│   │   ├── manager.py            # KVCacheManager (offload / prefix retrieval / cross-instance xfer)
│   │   ├── config.py             # LMCacheConfig
│   │   └── types.py              # KVCacheBackend (CPU/DISK/REMOTE)
│   │
│   ├── elastic_rm/               # ★ NEW: resource elasticity
│   │   ├── elastic_executor.py   # ElasticExecutor: pause/resume workloads via TMS
│   │   ├── scaling_policy.py     # ScalingPolicy + InstanceSignal
│   │   ├── cluster_topology.py   # ClusterTopology / GPUSlot / InstanceStatus
│   │   └── diagnostics.py        # Backlog diagnostics logging
│   │
│   ├── concurrency/              # Process-local concurrency utilities
│   │   └── token_bucket.py       # rate limiting
│   │
│   ├── checkpoint/               # ★ NEW: checkpoint helpers
│   │   └── megatron_saver.py     # Per-rank Megatron save (bypasses DCP to dodge UCX corruption)
│   │
│   ├── tito/                     # ★ NEW: TITO session → training arrays
│   │   ├── training_data.py      # Build prompt/response/mask/logprob arrays from TITO sessions
│   │   └── templates/            # TITO templates
│   │
│   ├── nixl/                     # GPU-direct communication (client/server/comm_plan/... 9 files)
│   ├── converter/                # Model format conversion (see §8)
│   │   ├── {fsdp,megatron,hf,vllm}_converter.py, base_converter.py, model_mappings.py
│   │   ├── megatron_optimizer.py # ★ NEW optimizer-state conversion
│   │   ├── weight_layout_plan.py, weight_layout_transforms.py  # ★ NEW vllm-hf layout plugin
│   │   ├── utils/parallel_states.py
│   │   └── modeling/             # per-format model loading
│   ├── logger/                   # Logging subsystem (data/env/memory/ps/ray + deprecated)
│   ├── dataset/                  # data_processor.py (DataProcessor), rl_dataset.py, utils.py
│   ├── post_processor/           # base.py + buffer_post_process/ + group_post_process/
│   ├── profiling/                # collector / event_converter / records
│   ├── common/                   # async_utils, chat_template, http_utils (+ distributed POST
│   │                             #   actor pool), http_io_thread, memory_utils, nixl_names,
│   │                             #   patch_utils, serialization, worker_naming, dynamic_import
│   ├── ray/                      # lazy_primitives, lock_context
│   ├── reward_score/             # sandbox_fusion.py + default async scorer
│   ├── rollout/                  # loop_timer, overflow, rollout_trace, trajectory_writer, vision_utils
│   ├── server/command.py         # Command / CommandType / CommandExtension (coordinator commands)
│   └── visualization/            # event_type, log_stats, log_visualizer (+ static/templates)
│
└── bench/                        # ★ Benchmarking
    ├── basic/cluster_scanner.py
    └── rollout/                  # main_rollout.py, stats_collector.py, vllm_extension.py, config/
```

### Non-`psrl/` Directories

```
third_party/
├── vllm/            # Vendored vLLM (PSRL-patched) — inference engine
├── verl/            # Vendored veRL — RLHF base framework + config source of truth
├── smg/             # ★ NEW: SMG Rust rollout gateway / model gateway (default router)
├── TransferQueue/   # ★ NEW: asynchronous sample data plane
├── LMCache/         # ★ NEW: KV cache offload / prefix reuse / cross-instance transfer
├── Megatron-Bridge/ # ★ NEW: Megatron model bridge
└── nixl/            # Vendored NIXL — GPU-direct RDMA communication
                     # (torch_memory_saver / TMS now installed as a dependency, not vendored here)

patch/               # Git patches applied at install time (see apply_patch.sh)
├── vllm/  verl/  nixl/  lm_cache/  transfer_queue/  megatron_bridge/  transformer_engine/

hydra_plugins/       # Repo-root Hydra plugin package (auto-discovered by Hydra on sys.path);
└── psrl_searchpath.py   # mirrors psrl/trainer/config/hydra_plugins/psrl_searchpath.py

scripts/
├── install_basic.sh        # Core deps + vLLM + veRL (patches applied)
├── install_nixl.sh         # NIXL + UCX
├── install_lmcache.sh      # ★ NEW: LMCache
├── install_megatron.sh     # Megatron-LM + Megatron-Bridge
├── reinstall_smg.sh        # ★ NEW: rebuild/reinstall SMG router
├── convert_hf_to_mcore.py  # HF→Megatron conversion
├── convert_perrank_to_dcp.{py,sh}  # ★ NEW: per-rank ckpt → DCP
└── depunct_docs.py

docs/                # ★ Sphinx/MyST docs — read design/ for authoritative subsystem specs
└── design/  architecture.md, router_tito.md, transfer_queue.md, kv_cache.md,
             parameter_server.md, staleness_control.md, flexible_rollout.md,
             resource_elasticity.md, index.md

tests/               # ★ THE test suite (pyproject `testpaths = ["tests"]`) — ~100 files, see §10

unit_tests/          # Legacy remnant, NOT on testpaths; only kv_cache/ + workers/ps/ survive
                     # (agent_loop suite moved to deprecated/unit_tests/ with the router code)

deprecated/          # ★ Top-level graveyard: pre-SMG code, not imported by anything active
├── workers/gen/gen_worker.py               # PSRL_GenWorker (superseded by PSRL_vLLMReplica)
├── workers/train/fsdp_train_worker.py      # PSRL_FSDPTrainWorker (superseded by EngineTrainWorker)
├── workers/train/megatron_train_worker.py  # PSRL_MegatronTrainWorker (superseded by EngineTrainWorker)
├── workers/agent_loop/router.py            # RolloutRouter (superseded by RolloutGateway/SMG)
├── workers/agent_loop/route_strategy.py    # Python routing strategies (now in the SMG Rust router)
├── workers/agent_loop/request_queue.py     # Python request queue (now in SMG)
├── workers/agent_loop/loops/mini_swe_agent_loop.py     # MiniSWEAgentLoop v0 (superseded by V1)
└── unit_tests/workers/agent_loop/{conftest.py,test_kv_cache_aware_strategy.py}

examples/
├── mini_swe/        # ★ SWE-bench RL recipe: fsdp_/megatron_ launch scripts (7B–32B, swe_gym +
│                    #   swe_smith), config.py, reward.py, swebench_grader.py, runner.py,
│                    #   config/, data/, eval/, prepare/, utils/, plus checked-in run logs
├── dapo_trainer/    # DAPO recipes (fsdp + megatron, 3B–70B)
├── tx/              # Qwen3 / Qwen3.5 launch scripts (8B/32B, 4B/35B-A3B)
├── anaylsis/        # [sic] Rollout analysis plots: plot_request_route_timeline.py,
│                    #   plot_trajectory_record.py, build_cost_model.py, regression.py, scripts/
├── bench/rollout/, ray/, retool/, visualization/
└── deprecated/      # Old grpo/ppo/precision recipes (kept for reference)
```

---

## 4. Configuration System

**Hydra-based**, compositional YAML. Since the veRL merge, **config dataclasses are re-exported from veRL** (`psrl/trainer/config/__init__.py` does `from verl.trainer.config import *`). PSRL-specific config lives under `psrl/trainer/config/psrl/`. A Hydra search-path plugin stitches the veRL and PSRL config dirs together; it exists twice — `psrl/trainer/config/hydra_plugins/psrl_searchpath.py` and a repo-root `hydra_plugins/psrl_searchpath.py` (Hydra auto-discovers the top-level `hydra_plugins` package on `sys.path`). Keep both in sync.

### PSRL-specific config root: `psrl/trainer/config/psrl/psrl.yaml`

**★ Restructured (commit `11b6d65`)**: `psrl.yaml` used to be a ~530-line monolith. It is now a thin root holding only a `defaults:` list plus a handful of scalars; every topic lives in its own sibling YAML, merged at `psrl.<group>`. When looking for a knob, open the group file rather than grepping `psrl.yaml`.

Scalars still in `psrl.yaml`: `ps_manager_ip`, `reward_service_ip`, `staleness`, `staleness_buffer_entries`, `rollout_n`, `ps_mode` (`cpu_ref`/`nixl_cpu`), `logging_path`, `retry_bound`, `retry_ratio`, `colocate_validate_and_train`, `fuse_rollout_with_validate`.

| Group file → key | Controls |
|-----------|----------|
| `deployment.yaml` → `psrl.deployment` | Node/GPU counts per role (`train_nnodes`, `n_rollout_instances`, `rollout_ngpus_per_node_per_instance`, `total_nnodes`), `heterogeneous_rollout.*`, and **`elastic_rm.*`** (★ moved here from `psrl.elastic_rm`) |
| `rollout_gateway.yaml` → `psrl.rollout_gateway` | SMG gateway process: `server_max_concurrency`, `rust_log_filter`, health-check waits, `use_distributed_post` + `post_actor_num_per_node` (Ray HTTP POST actor pool) |
| `server_rollout.yaml` → `psrl.server_rollout` | HTTP rollout service entrypoint used by agent loops (`gateway.router_ip/port`, `server_concurrency`) |
| `status_collection.yaml` → `psrl.status_collection` | Engine→coordinator→router status sync intervals, file-dump level, `stats_recorder.*` |
| `fine_grain_overlap.yaml` → `psrl.fine_grain_overlap` | ★ NEW: `granularity` (`none`/`mini_batch`/`micro_batch`), `multiplier`, `overlap_scope` (`recompute`/`pre_step`) |
| `lmcache.yaml` → `psrl.lmcache` | LMCache KV offload (`enable`, `backend`, offload budget, prefix reuse) |
| `tms.yaml` → `psrl.tms` | torch_memory_saver `range` (`null`/`train`/`all`), `enable_cuda_graph`, `enable_nixl` |
| `nixl.yaml` → `psrl.nixl` | NIXL server IP/port, pinned temp-memory slots (used by `ps_mode: nixl_cpu`) |
| `agentic_rl.yaml` → `psrl.agentic_rl` | `refill_failure_threshold` (consecutive group-failure breaker, raised from the train waiters), per-trajectory text dump via `TrajectoryWriter` |
| `checkpoint.yaml` → `psrl.checkpoint` | `use_dcp_save` (DCP vs PSRL's UCX-safe per-rank `torch.save`) |
| `broadcast_init.yaml` → `psrl.broadcast_init` | Rank-0 PS worker reads ckpt and broadcasts via NIXL (`binary_tree`) |
| `group_post_process.yaml` / `buffer_post_process.yaml` | Registered post-processors (`dynamic_sampling_filter`, `no_filter`) for group/buffer stages |
| `log_prob.yaml`, `memory_logger.yaml`, `profile.yaml` | Rollout-engine logprobs; periodic memory logging; analysis toggles (`disable_attn`, `fix_weight`) |
| `rollout_coordination/routing_strategy.yaml` | SMG worker-selection `method` (`request_num_balance` default; also `random`, `round_robin`, `throughput_optimal[_with_budget]`, `cache_aware`, `cache_aware_v1`) |
| `rollout_coordination/sync_and_mig_strategy.yaml` | weight-sync mode (`greedy` vs `status_based`) + request migration rules |
| `rollout_coordination/partial_rollout.yaml` | partial-rollout loopback on weight-sync abort |
| `rollout_coordination/redundant_rollout.yaml` | redundant/over-sampling rollout |
| `rollout_coordination/proactive_filter_strategy.yaml` | proactive filtering of samples (`retry`/`truncate`) |
| `rollout_coordination/session_strategy.yaml` | session hang/continue (ThunderAgent) params |

There is also a top-level `transfer_queue.*` group, force-enabled in `main_ppo.py`. Cross-group consistency is checked by `psrl/utils/config.py::validate_config()`.

### Launch Command

```bash
python -m psrl.trainer.main_ppo \
    ++trainer.total_epochs=10 \
    ++psrl.deployment.train_nnodes=2 \
    +train_actor_rollout_ref.model.path=/path/to/model
```

Templates: `ppo_trainer.yaml` (FSDP) / `ppo_megatron_trainer.yaml` (Megatron).

---

## 5. Key Classes Quick Reference

### Trainer Layer

| Class | File | Role |
|-------|------|------|
| `PSRL_RayPPOTrainer` | `trainer/ppo/ray_trainer.py` | Main training loop orchestrator |
| `StepStrategy` | `trainer/ppo/strategies/base.py` | ABC for training step; `FullBatchStepStrategy` (default) and `FineGrainOverlapStrategy` (chunk-pipelined) are the concrete implementations |
| `TaskRunner` | `trainer/main_ppo.py` | Ray driver: builds worker groups, inits TransferQueue, elastic_rm/reward pools |
| `DataProcessor` | `utils/dataset/data_processor.py` | Ray actor: dataset loading, batching, TransferQueue producer |

### Worker Layer

| Class | File | Role |
|-------|------|------|
| `PSRL_vLLMHttpServer` / `PSRL_vLLMReplica` | `workers/gen/vllm_async_server.py` | gRPC vLLM engine server registered with SMG |
| `RolloutGateway` | `workers/gen/rollout_gateway.py` | Ray actor launching SMG Router + SessionRouter subprocesses |
| `RolloutCoordinator` | `workers/gen/rollout_coordination/coordinator.py` | Drives SMG: stats, pause/resume, weight-version, sync/migrate (mixin-composed) |
| `SessionRouter` | `workers/gen/session_router.py` | Session-scoped OpenAI API, hang/continue state (separate uvicorn proc) |
| `EngineTrainWorker` | `workers/train/engine_train_worker.py` | Unified FSDP/Megatron PPO update worker |
| `PSRL_BaseTrainWorker` | `workers/train/base_train_worker.py` | Shared train logic: NIXL push, version mgmt |
| `PSManager` | `workers/ps/ps_manager.py` | Model version store + staleness inventory (also served over gRPC) |
| `PSStorageWorker` | `workers/ps/ps_storage_worker.py` | Holds model shards on GPU |
| `RewardManager` | `workers/reward/reward_manager.py` | Reward scoring pipeline over TransferQueue |
| `RewardModelManager` | `workers/reward/reward_model/manager.py` | Owns RM replicas + coordinator (RM inference service) |
| `GenRewardManager` | `workers/reward/reward_loop/gen.py` | Routes reward requests to SMG reward-model gateway |
| `KVCacheManager` | `utils/kv_cache/manager.py` | LMCache offload / prefix retrieval / cross-instance transfer |
| `ElasticExecutor` | `utils/elastic_rm/elastic_executor.py` | Pause/resume workloads for GPU re-sharing |
| `PSRL_AgentLoopManager` | `workers/agent_loop/manager.py` | Multi-turn / session tool-use orchestration; also yields training chunks (`wait_for_training_chunk`) for fine-grain overlap and owns the distributed POST actor pool |
| `HarnessAgentLoop` | `workers/agent_loop/loops/harness_agent_loop.py` | Generic template-method loop owning task sandbox, harness process, session-scoped TITO capture, abort, and cleanup |
| `HarnessTaskContext` | `workers/agent_loop/harness/task.py` | Immutable task description: prompt, sandbox specs/backend, snapshot/metrics policy, and opaque task state |

### mini-SWE Agent Layer

| Class | File | Role |
|-------|------|------|
| `MiniSWEAgentLoopV1` | `workers/agent_loop/loops/mini_swe_agent_loop_v1.py` | CURRENT SWE-bench loop: sync sandbox facade + async PSRL rollout bridge |
| `MiniSWEHarnessAgentLoop` | `workers/agent_loop/loops/mini_swe_harness_agent_loop.py` | Harness-based SWE loop; supplies environment, prompt, patch collection, clean grading, and task cleanup hooks |
| `MiniSWEAgentData` | `workers/agent_loop/agent_data/mini_swe_agent_data.py` | Trajectory building, patch extraction, grading state |
| `ConversationAgentData` | `workers/agent_loop/agent_data/conversation_agent_data.py` | Chat-template base (OpenAI format, token counting) |
| `MiniSWEEnvironment` | `environments/mini_swe_env.py` | SWE task parsing, per-problem config override merging |
| `ClaudeCodeHarness` / `CodexHarness` | `workers/agent_loop/harness/` | CLI-specific installation, launch, callback environment, output bounding, and abort behavior |

### Sandbox Layer

| Class | File | Role |
|-------|------|------|
| `SandboxManager` / `SandboxLease` | `sandbox/manager.py` | Worker registry, idempotent create, capability/state-policy gates and lifecycle ownership |
| `SyncSandboxManager` / `SyncSandboxSession` | `sandbox/sync.py` | Thread facade over the owning worker event loop; no extra backend client/loop |
| `DockerBackend` | `sandbox/backends/docker.py` | Persistent Engine API data plane, typed security/disk policy, container limits and metrics |
| `DockerLifecycle` | `sandbox/backends/docker_lifecycle.py` | Worker owner lease, graceful cleanup and restartable node-level crash collector |
| `SandboxCapacityCoordinator` | `sandbox/capacity.py` | One Ray actor per node; weighted CPU+memory admission using actual sandbox requests |
| `AgentEnvBackend` | `sandbox/backends/e2b.py` | AgentEnv image/template factory plus native full-state fork/snapshot driver |
| `CubeSandboxBackend` | `sandbox/backends/e2b.py` | Cube template factory plus snapshot/restore branch driver |

### Staleness System

| Class | File | Role |
|-------|------|------|
| `StalenessInventory` | `workers/ps/staleness_controller.py` | Collection of versioned buffers |
| `StalenessBuffer` | `workers/ps/staleness_controller.py` | Fixed-size buffer per model version |
| `PSRL_RequestStatus` | `workers/ps/request_status_tracker.py` | Request lifecycle state machine |

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `EntryCategory` | EMPTY → RESERVED → OCCUPIED | Entry lifecycle in staleness buffer |
| `BufferStatus` | READY, READY_WITH_CAPACITY, STUCK, PENDING | Buffer readiness for training |
| `PSRL_Role` | Actor, Rollout, Critic, RewardModel, RefPolicy, DummyPolicy, Validate | Worker role enum (in `trainer/ppo/utils.py`) |
| `KVCacheBackend` | CPU, DISK, REMOTE | LMCache offload backend |

---

## 6. Core Flows

### 6.1 Startup

```
main_ppo.py::main()
  → config.transfer_queue.enable = True; Hydra resolves config
  → run_ppo(config, TaskRunner)
    → ray.init(runtime_env=PPO_RAY_RUNTIME_ENV)
    → tq.init(config.transfer_queue)          # TransferQueue data plane
    → TaskRunner (Ray actor on head node)
      → build worker groups (Gen, EngineTrain, Critic, Reward, RewardModel, ...)
      → if elastic_rm.enable: set up shared GPU pools
      → PSManager + gRPC service, DataProcessor, RolloutGateway (SMG + SessionRouter)
      → PSRL_RayPPOTrainer(config) → trainer.fit()
```

### 6.2 Training Loop (`fit()`)

```
for step in range(total_steps):
  ① batch ← DataProcessor / TransferQueue (OCCUPIED staleness entries)
  ② scores ← RewardManager / RewardModel service (rule-based / RM / sandbox)
  ③ Apply KL penalty → token_level_rewards
  ④ advantages ← compute_advantage (GAE/GRPO/REINFORCE++)
  ⑤ critic update; ⑥ actor (PPO) update  [both via EngineTrainWorker]
  ⑦ Periodic: validate, checkpoint, log
```

Steps ①–⑥ are executed by the `StepStrategy` selected via `build_step_strategy()`:
`FullBatchStepStrategy` waits for the whole batch (default), while `FineGrainOverlapStrategy`
consumes mini/micro-batch chunks from `PSRL_AgentLoopManager.wait_for_training_chunk()` so the
per-sample stages (`old_log_prob`, `ref_log_prob`, `values`, `reward` — see `STAGE_META`) start
before rollout finishes. `overlap_scope: pre_step` additionally runs advantage+update per chunk.

### 6.3 Asynchronous Rollout + Staleness

```
EngineTrainWorker completes update
  → nixl_push_model() [async background, GPU-direct] → PSManager.push(version=N)
  → PSManager creates StalenessBuffer(version=N)
  → RolloutCoordinator gates SMG: pause dispatch → publish weight-version update → resume

GenWorker: pulls weights via NIXL → generates. Entries EMPTY→RESERVED→OCCUPIED;
buffer PENDING→READY. DataProcessor consumes READY buffers → EMPTY; old buffers freed.
```

### 6.4 Rollout via SMG (see docs/design/router_tito.md)

```
AgentLoopWorker ──HTTP generate/chat──► SMG RolloutGateway
Session AgentLoop ──session OpenAI API──► SessionRouter ──► SMG
SMG ◄──gRPC admission + Reserve──► PSManager
SMG ──gRPC──► PSRL vLLM replicas (PSRL_vLLMReplica)
RolloutCoordinator ──stats / pause-resume / weight-version──► SMG

PSRL worker-selection ("psrl" strategy): filter by servable version_tag →
honor partial/sticky hint → prompt-group affinity → PSManager admission/reserve →
apply SMG policy → optionally trigger LMCache KV transfer on migration.

Partial rollout: on weight-sync abort, SMG drains the gRPC stream, preserves
token IDs + logprobs (+ routed-expert meta), loops the request back, continues
from the accumulated prefix.
```

### 6.5 Session / TITO Loop (multi-turn)

```
SessionRouter tracks per-session state (status: generate|env, hang_state: running|hung),
pins each session to one (base_worker_id, dp_rank) instance.
ThunderAgentScheduler decides hang (evict over-capacity instance) / continue (readmit
when the pinned instance has KV room), using per-instance KV capacity snapshots.
On completion, TITO GET /tito/sessions returns accumulated_token_ids + per-turn records;
utils/tito/training_data.py converts them to prompt/response/mask/logprob arrays.
```

### 6.6 mini-SWE Agent Flow (SWE-bench RL)

```
MiniSWEAgentLoopV1.run(request)
  → MiniSWEEnvironment.reset(task): parse extra_info, apply_data_overrides → runtime_config
  → SandboxManager.sync() → SyncSandboxManager
  → worker thread: DefaultAgent.run(task) through MiniSWEAgentAdapter
      (observe → model query → parse action → generic SyncSandboxSession.exec)
  → capable microVM only: clean baseline snapshot → matching verifier restore
  → post-rollout grading (smith: checkout+patch+revert-tests+harness / gym: eval_script)
  → finalize: compute_score(data_source) → DataProto with reward
```

Harness-based comparison path:

```
HarnessAgentLoop.run(request)                       # generic lifecycle owner
  → MiniSWEHarnessAgentLoop.prepare_harness_task  # environment + task sandbox spec
  → create TITO session + acquire task sandbox
  → ClaudeCodeHarness/CodexHarness.prepare + run  # CLI calls session-scoped URL
  → TITO prefix tree emits one or more trajectories
  → collect_harness_artifact                     # patch before sandbox release
  → release agent sandbox + delete/drain session
  → MiniSWEHarnessAgentLoop.finalize_harness_task # independent clean grader
  → reward + TokenOutput(s) + generic cleanup
```

---

## 7. Reward System

| Loop | File | Algorithm |
|------|------|-----------|
| `NaiveRewardLoop` | `reward_loop/naive.py` | Standard single reward per response |
| `DAPORewardLoop` | `reward_loop/dapo.py` | DAPO dynamic reward |
| `PRIMERewardLoop` | `reward_loop/prime.py` | PRIME process reward model |
| `GDPORewardLoop` | `reward_loop/gdpo.py` | GDPO variant |
| `GenRewardManager` | `reward_loop/gen.py` | Generative/pooling RM via SMG reward gateway |

- **Registry**: `reward_loop/registry.py` (rule/score) + `gen_reward_function/registry.py` (`@gen_reward_func` generative RMs: `default_gen_rm`, `skywork_rm`).
- **Reward-model service**: `reward/reward_model/` (manager → coordinator → replicas → gateway) serves RM inference; `RewardWorker` produces/consumes reward fields over TransferQueue.
- **Sources**: rule-based, LLM/RM-as-judge (via SMG model gateway: `/v1/completions`, `/v1/classify`, `/v1/embeddings`), SandboxFusion.

---

## 8. Model Format Converter

```
HuggingFace (HF)  ←→  FSDP  ←→  vLLM
                   ←→  Megatron-LM   (weights + optimizer state)
```

| Converter | File |
|-----------|------|
| `FSDPConverter` | `utils/converter/fsdp_converter.py` |
| `MegatronConverter` | `utils/converter/megatron_converter.py` |
| `MegatronOptimizer` conv. | `utils/converter/megatron_optimizer.py` |
| `VLLMConverter` | `utils/converter/vllm_converter.py` |
| `HFConverter` | `utils/converter/hf_converter.py` |
| Key mappings | `utils/converter/model_mappings.py` |
| vLLM↔HF layout plugin | `utils/converter/weight_layout_{plan,transforms}.py` |

Supports newer arches (DeepSeek-V2, Qwen3.5). Per-format loading in `converter/modeling/`.

---

## 9. Third-Party Dependencies

| Dependency | Location | Purpose | Patches |
|-----------|----------|---------|---------|
| **vLLM** | `third_party/vllm/` | Inference engine | `patch/vllm/` (GPU mem, attention, scheduler, TMS) |
| **veRL** | `third_party/verl/` | RLHF base + config source | `patch/verl/` |
| **SMG** | `third_party/smg/` | Rust rollout + model gateway (default router) | built via `scripts/reinstall_smg.sh` |
| **TransferQueue** | `third_party/TransferQueue/` | Async sample data plane | `patch/transfer_queue/` |
| **LMCache** | `third_party/LMCache/` | KV cache offload / prefix reuse | `patch/lm_cache/` |
| **Megatron-Bridge** | `third_party/Megatron-Bridge/` | Megatron model bridge | `patch/megatron_bridge/` |
| **NIXL** | `third_party/nixl/` | GPU-direct RDMA comm | `patch/nixl/` |
| **torch_memory_saver (TMS)** | installed dep | GPU memory pause/resume (elasticity) | — |
| **transformer_engine** | installed dep | Megatron FP8/attn kernels | `patch/transformer_engine/` |

**Patch system**: git patches under `patch/<dep>/` applied at install time (consolidated via `patch/apply_patch.sh`), plus runtime monkey-patches for TMS integration.

---

## 10. Testing Map

**`tests/` is the real suite** — `pyproject.toml` sets `testpaths = ["tests"]`. (`unit_tests/` is a legacy leftover holding only `kv_cache/test_lmcache_config.py` and `workers/ps/test_ps_manager_broadcast.py`; it is not collected by default.)

```
tests/
├── trainer/          → step strategies, fine-grain overlap config, chunk manager,
│                       main_ppo imports, PSRL_Role
├── config/           → reward config + YAML merge semantics
├── converter/        → converter compat, model registry, vLLM weight-layout plugin, packing specs
├── state_dict/       → FSDP1/FSDP2/vLLM converter round-trips (+ scripts/ launchers)
├── nixl/             → comm planner, sharding, send/recv, e2e (+ config/nixl_e2e.yaml, scripts/)
├── parameter_server/ → PSManager, request status tracker
├── staleness/        → staleness controller
├── workers/reward/   → RM manager / coordinator (incl. elastic) / gateway / gen reward manager
├── workers/train/    → EngineTrainWorker init, load-weight flag
├── gen_dplb/         → SMG adapter cache-aware routing, stats recorder
├── tito/, e2e/tito/  → SessionRouter + TITO training-data build (+ test_tito_e2e.sh)
├── elastic_rm/       → elastic executor + scaling policy instance IDs
├── ray_utils/        → lazy primitives, lock context, GPU sharing, version waiter, ...
├── fsdp/, megatron/, torch_dist/, checkpoint/  → GPU/dist smoke tests (mostly .sh-driven)
├── dataset/, environments/, tools/, unit/      → DataProcessor, env registry, LMCache snapshot
└── test_thunder_agent_scheduler.py             → session hang/continue scheduler
```

CPU-only tests are marked `pytest.mark.cpu_test`. Run with pytest under the psrl conda env.

---

## 11. Key Conventions

| Convention | Rule |
|-----------|------|
| **Logging** | `psrl_logger = logging.getLogger(__file__)` (or `__name__`); level from `PSRL_LOGGING_LEVEL` — never `print()` |
| **Comments** | `# NOTE(author):`, `# TODO(author):`, `# FIXME(author):`, `# HACK(author):` |
| **Docstrings** | Google-style, opening `"""` on its own line |
| **Assertions** | Always include descriptive message with period |
| **Linting** | Ruff, line length 119 |
| **Language** | English only in all code artifacts |

---

## 12. File Size Landmarks (by complexity)

| File | Lines | Why it matters |
|------|-------|---------------|
| `trainer/ppo/ray_trainer.py` | ~3337 | THE main training loop — start here for any training question |
| `workers/agent_loop/manager.py` | ~1708 | Request dispatch, chunk emission for fine-grain overlap (`_emit_pending_chunks`, `wait_for_training_chunk`), distributed POST actor pool |
| `workers/ps/ps_manager.py` | ~1373 | Central coordination point for all workers (also gRPC-served) |
| `workers/ps/staleness_controller.py` | ~1365 | THE staleness system — core async innovation |
| `workers/reward/reward_manager.py` | ~1138 | Reward pipeline over TransferQueue |
| `workers/gen/rollout_coordination/coordinator.py` | ~838 | SMG-driving coordinator (mixin composition) |
| `workers/train/engine_train_worker.py` | ~592 | Unified FSDP/Megatron update worker |
| `workers/gen/session_router.py` | ~571 | Session/TITO hang-continue routing |
| `trainer/ppo/strategies/` | ~510 | Step orchestration: `base.py` (StepStrategy ABC + `STAGE_META`), `full_batch.py` (default), `fine_grain_overlap.py` (chunk-pipelined) |
| `utils/config.py` | ~394 | `validate_config()` cross-group checks + fine-grain chunk sizing |

---

## 13. Import Dependency Graph (Simplified)

```
main_ppo.py (TaskRunner, TransferQueue init)
  → ray_trainer.py (PSRL_RayPPOTrainer)
    → workers/gen/vllm_async_server.py (PSRL_vLLMReplica) → utils/nixl/client.py, vllm_rollout.py
    → workers/gen/rollout_gateway.py (RolloutGateway) → smg_adapter.py → third_party/smg
    → workers/gen/rollout_coordination/coordinator.py (RolloutCoordinator)
        → session/thunder_agent.py, sync_and_migrate/*, smg_adapter.py, elastic_rm/diagnostics
    → workers/train/engine_train_worker.py → base_train_worker.py → nixl/client.py, converter/*
    → workers/ps/ps_manager.py → staleness_controller.py, request_status_tracker.py
        → grpc/ps_manager_service.py (gRPC surface)
    → workers/reward/reward_manager.py → reward_loop/*, reward_model/manager.py
    → workers/agent_loop/manager.py → loops/* (incl. session_agent_loop, mini_swe_*)
        → tools/*, environments/*
    → utils/dataset/data_processor.py, utils/transferqueue_utils.py
    → utils/kv_cache/manager.py, utils/elastic_rm/elastic_executor.py
```

---

## 14. Quick "Where Do I Look?" Index

| I want to... | Go to |
|--------------|-------|
| Understand the training loop | `trainer/ppo/ray_trainer.py::fit()` |
| Understand / extend the step pipeline | `trainer/ppo/strategies/` — `base.py` for the ABC, `full_batch.py` for the default path, `fine_grain_overlap.py` for chunk-overlap |
| Change PPO hyperparameters | veRL configs (re-exported) + `psrl/trainer/config/psrl/*.yaml` (per-topic group files, see §4) |
| Find a PSRL config knob | Open the matching group file under `psrl/trainer/config/psrl/` — `psrl.yaml` is now only a `defaults:` list |
| Enable rollout/train overlap | `psrl/trainer/config/psrl/fine_grain_overlap.yaml` + `trainer/ppo/strategies/fine_grain_overlap.py` |
| Change node/GPU layout | `psrl/trainer/config/psrl/deployment.yaml` (`psrl.deployment.*`) |
| Understand rollout routing / SMG | `docs/design/router_tito.md`, `workers/gen/smg_adapter.py`, `rollout_gateway.py` |
| Change weight-sync / request migration | `workers/gen/rollout_coordination/sync_and_migrate/` + `sync_and_mig_strategy.yaml` |
| Change session hang/continue | `rollout_coordination/session/thunder_agent.py` + `session_strategy.yaml`, `session_router.py` |
| Debug model version issues | `workers/ps/staleness_controller.py`, `workers/ps/ps_manager.py` |
| Debug model transfer failures | `utils/nixl/client.py`, `workers/train/base_train_worker.py` |
| Work with sample data plane | `utils/transferqueue_utils.py`, `docs/design/transfer_queue.md` |
| Tune KV cache / LMCache | `utils/kv_cache/`, `psrl.lmcache.*` (`lmcache.yaml`), `docs/design/kv_cache.md` |
| Configure resource elasticity | `utils/elastic_rm/`, **`psrl.deployment.elastic_rm.*`** (`deployment.yaml`), `docs/design/resource_elasticity.md` |
| Add/modify training backend | `workers/train/engine_train_worker.py` (unified FSDP/Megatron) |
| Add a rule/score reward | `workers/reward/reward_loop/` + `registry.py` |
| Add a generative reward model | `workers/reward/gen_reward_function/` (`@gen_reward_func`) + `reward_model/` service |
| Modify vLLM inference / scheduling | `workers/gen/vllm_async_server.py`, `rollout_scheduler.py` |
| Add a tool / tool parser | `tools/`, `tools/tool_parser/` (model-specific parsers) |
| Change data loading | `utils/dataset/data_processor.py`, `rl_dataset.py` |
| Add model arch / weight layout | `utils/converter/` (converter + `weight_layout_*`, `modeling/`) |
| Save/restore Megatron ckpt | `utils/checkpoint/megatron_saver.py`, `scripts/convert_perrank_to_dcp.py` |
| Build TITO training arrays | `utils/tito/training_data.py` |
| Train on SWE-bench | `examples/mini_swe/` (launch scripts, `runner.py`, `config.py`) |
| Debug SWE agent rollouts | `workers/agent_loop/loops/mini_swe_agent_loop_v1.py` |
| Change SWE grading | `examples/mini_swe/swebench_grader.py` |
| Run benchmarks | `psrl/bench/`, `examples/bench/` |
| Run / add tests | `tests/` (pytest `testpaths`); `unit_tests/` is a legacy remnant |
| Find pre-SMG legacy code | top-level `deprecated/` (gen worker, FSDP/Megatron train workers, Python router, mini-SWE loop v0) |
| Plot rollout routing / trajectories | `examples/anaylsis/plot_request_route_timeline.py`, `plot_trajectory_record.py` |
| Install from scratch | `scripts/install_basic.sh` → `install_nixl.sh` → `install_lmcache.sh` → `install_megatron.sh` → `reinstall_smg.sh` |
| Read authoritative subsystem specs | `docs/design/` |

---

## 15. Terminology Glossary

| Term | Meaning |
|------|---------|
| **SMG** | Vendored Rust rollout/model **gateway** (`third_party/smg`), PSRL's default router; dispatches live inference requests to vLLM replicas with a PSRL-aware worker-selection strategy |
| **RolloutCoordinator** | Ray-side controller that keeps SMG state current (worker stats, weight versions, routing-loop pause/resume, sync & migrate) — composed from mixins |
| **SessionRouter** | Separate uvicorn process exposing a session-scoped OpenAI API; tracks per-session hang/continue state, pins sessions to one instance |
| **TITO** | "Token-In-Token-Out" session capture in SMG; `GET /tito/sessions` returns accumulated token IDs + per-turn records used to build training arrays |
| **ThunderAgent** | Capacity-based session hang/continue scheduler (ported); "program→session, reasoning→generate, acting→env, pause/resume→hang/continue" |
| **TransferQueue** | Async sample **data plane**; components exchange `KVBatchMeta` references instead of serializing whole TensorDict batches (distinct from SMG's request queue) |
| **LMCache** | KV cache offload / prefix reuse / cross-instance transfer backend |
| **NIXL** | GPU-direct RDMA transport for GPU→GPU weight movement |
| **TMS** | torch_memory_saver — GPU memory pause/resume primitive underpinning resource elasticity |
| **elastic_rm** | Resource elasticity subsystem re-sharing GPUs across train/gen/eval/reward |
| **partial rollout** | On weight-sync abort, SMG preserves the generated prefix and continues the request after re-selection instead of restarting |
| **RolloutInstanceId** | `(base_worker_id, dp_rank)` — identifies one data-parallel vLLM instance for routing/affinity |
| **fine-grain overlap** | Starting training stages on mini/micro-batch chunks before the full rollout batch is collected; implemented as `FineGrainOverlapStrategy` and configured by `psrl.fine_grain_overlap` |
| **StepStrategy** | Pluggable per-step orchestration seam in `trainer/ppo/strategies/`; `STAGE_META` tags each phase `per_sample` / `batch_coupled` / `optimizer_step` |
```
