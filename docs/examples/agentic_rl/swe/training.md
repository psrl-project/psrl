# SWE Training

Once the [data and Docker images](prepare) are in place, kick off training with the
smallest reference recipe, FSDP2 on Qwen2.5-7B against the toy dataset:

```bash
bash examples/mini_swe/fsdp_qwen_7b_dapo.sh        # default staleness=2
bash examples/mini_swe/fsdp_qwen_7b_dapo.sh 3      # override staleness
```

The other launch scripts (`fsdp_qwen_7b_swe_smith.sh`, `fsdp_qwen_7b_swe_gym.sh`,
`megatron_qwen_*_swe_*.sh`) follow the same structure, switching paths only changes
the dataset, the agent-config YAML, and the cluster layout, not the agentic-RL knobs
documented below.

---

## SWE-specific knobs

On top of the [common agentic-RL knobs](../index) (`multi_turn.enable`,
`multi_turn.max_turns`, `agent.env.name`, `agent.data.name`, `data.return_raw_chat`),
the launch script pins the agent loop to mini-SWE-agent's Docker-sandboxed bash
environment.

| Knob | Value in `fsdp_qwen_7b_dapo.sh` | Role |
|---|---|---|
| `*.rollout.agent.env.name` | `mini_swe_env` | Selects `MiniSWEEnvironment` (Docker bash sandbox) |
| `*.rollout.agent.data.name` | `mini_swe_agent_data` | Selects `MiniSWEAgentData` (bridges `DefaultAgent` ↔ PSRL via thread-safe queues) |
| `*.rollout.agent.agent_loop_config_path` | `examples/mini_swe/config/simple_agent_config.yaml` | mini-SWE-agent YAML (system prompt, command parser, max steps). Swap to `swebench_agent_config.yaml` for SWE-smith / SWE-Gym. |
| `*.rollout.agent.num_workers` | `${NNODES}` | Parallel agent worker threads per rollout instance, each one runs its own Docker container, so set this to the number of nodes (or available cores) |
| `custom_reward_function.path` / `.name` | `examples/mini_swe/reward.py` / `compute_score` | Patch-grading reward (toy) or F2P/P2P test-execution reward (SWE-smith / SWE-Gym) |

The `agent_loop_config_path` is the field that decides *which* SWE flavour the agent
runs in (toy vs. SWE-smith vs. SWE-Gym), picking the wrong YAML is the most common
silent misconfiguration. The launch scripts wire this up automatically.

---

## Reward function

`compute_score` (in `examples/mini_swe/reward.py`) returns different `score` / `acc`
pairs depending on the dataset path.

### SWE-smith / SWE-Gym / Verified

| Condition | `score` | `acc` |
|---|---|---|
| All F2P pass, no P2P regressions | `+1.0` | `1.0` |
| Patch modified test/config files (policy violation) | `0.0` | `0.0` |
| Not resolved (tests failed) | `-1.0` | `0.0` |
| No patch / 0 turns (aborted) | `0.0` | `0.0` |

Grading runs in a **fresh** Docker container started from the same per-problem image,
so it is independent of any side effects from the agent's exploration. The eval
script and log parser are frozen into the prepared row (`swe_problem.eval_script`
plus a vendored parser registry), and the test log is parsed *inside* the grading
sandbox — no `swebench`/`swesmith` package is imported on the training host, so
rewards are reproducible across worker hosts. Splits prepared before this change
must be regenerated; see the [preparation guide](prepare).

### Toy path

Graduated scoring from `-0.1` (premature exit) to `+1.0` (exact patch match), with
partial credit for file-level and line-level overlap.

---

```{seealso}
Full launch-script parameter reference, agent-YAML schema, evaluation guide, and
troubleshooting:
[`examples/mini_swe/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/mini_swe/README.md).
```
