# PSRL Elastic Simulator

This standalone C++17 benchmark consumes the schema-v1 JSONL produced by
`scripts/extract_elastic_simulation_inputs.py`. It evaluates the baseline and
all selected candidate plans without starting Ray, vLLM, or a training job.

Build a release binary:

```bash
cmake -S tools/elastic_simulator -B build/elastic_simulator \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/elastic_simulator -j
```

RapidJSON headers (`rapidjson/document.h`) and pthreads must be available to
CMake. No dependency is downloaded during configuration.

The same binary can serve production candidate evaluation through one
persistent subprocess. With the binary at the default build path, enable it in
the mode5 launcher with:

```bash
PSRL_CANDIDATE_EVALUATION_BACKEND=cpp \
  bash examples/deployment_modes_qwen7b_rm32b/mode5_elastic_rl.sh
```

Set `PSRL_CANDIDATE_EVALUATION_CPP_BINARY` when the release binary is installed
elsewhere. The mode5 launchers default to `cpp`; set
`PSRL_CANDIDATE_EVALUATION_BACKEND=python` to select the Python implementation.
A configured C++ backend fails at policy startup if its binary is missing or
not executable, and evaluator process/protocol failures propagate instead of
being converted into a no-action scaling decision.

Schema-v1 priority fields represent numeric infinities with the standard-JSON
objects `{"__psrl_float__":"inf"}` and `{"__psrl_float__":"-inf"}`. This
preserves the live router's reserve-capability ordering without emitting the
non-standard JSON tokens `Infinity` or `-Infinity`.

Run serial and parallel benchmarks separately:

```bash
build/elastic_simulator/elastic_simulator \
  --input /tmp/elastic_simulation_inputs.jsonl \
  --threads 1 --warmup 3 --repeat 20 \
  --output /tmp/elastic_simulation_cpp_t1.jsonl

build/elastic_simulator/elastic_simulator \
  --input /tmp/elastic_simulation_inputs.jsonl \
  --threads 64 --warmup 3 --repeat 20 \
  --output /tmp/elastic_simulation_cpp_t64.jsonl
```

Use `--cycle`, `--candidate`, or `--role` for a focused benchmark.
`--emit-migrations` includes individual migration rows; otherwise only the
migration count is emitted.

`timing_ns.json_parse` is measured once per input record. Context preparation
and candidate evaluation are repeated after warm-up. The `*_worker_sum`
fields add time measured inside every candidate-role task and are not wall
time. `rebalance_wall_union` and `router_wall_union` directly merge the
corresponding task intervals on the monotonic clock. Because those intervals
can overlap across worker threads, use `rebalance_router_wall_union` for their
combined wall occupancy and `rebalance_router_wall_overlap` to inspect the
overlap. `candidate_evaluation_wall` is the direct wall time for the complete
parallel candidate batch, including task setup and work outside the two inner
simulation stages.
