#include "elastic_simulator/json_io.h"
#include "elastic_simulator/simulator.h"

#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace psrl::elastic_simulator {
namespace {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

struct Options {
  std::string input_path;
  std::optional<std::string> output_path;
  std::size_t threads = 1;
  std::size_t warmup = 3;
  std::size_t repeat = 20;
  std::optional<std::int64_t> cycle;
  std::optional<int> candidate;
  std::optional<std::string> role;
  bool emit_migrations = false;
};

[[noreturn]] void usage_error(const std::string &message) {
  throw std::invalid_argument(
      message +
      "\nusage: elastic_simulator --input PATH [--output PATH] [--threads N]"
      " [--warmup N] [--repeat N] [--cycle ID] [--candidate INDEX]"
      " [--role Rollout|RewardModel] [--emit-migrations]");
}

std::int64_t parse_integer_argument(const std::string &value,
                                    const std::string &flag) {
  std::size_t consumed = 0;
  std::int64_t parsed = 0;
  try {
    parsed = std::stoll(value, &consumed);
  } catch (const std::exception &) {
    usage_error(flag + " expects an integer, got: " + value);
  }
  if (consumed != value.size()) {
    usage_error(flag + " expects an integer, got: " + value);
  }
  return parsed;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "usage: elastic_simulator --input PATH [options]\n\n"
          << "  --output PATH          Write benchmark JSONL instead of "
             "stdout\n"
          << "  --threads N            Persistent evaluation workers (default: "
             "1)\n"
          << "  --warmup N             Unreported warm-up iterations (default: "
             "3)\n"
          << "  --repeat N             Measured iterations (default: 20)\n"
          << "  --cycle ID             Evaluate only one cycle\n"
          << "  --candidate INDEX      Evaluate baseline and one candidate\n"
          << "  --role ROLE            Evaluate only Rollout or RewardModel\n"
          << "  --emit-migrations      Include request-level migration rows\n";
      std::exit(0);
    }
    if (argument == "--emit-migrations") {
      options.emit_migrations = true;
      continue;
    }
    if (index + 1 >= argc) {
      usage_error("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--input") {
      options.input_path = value;
    } else if (argument == "--output") {
      options.output_path = value;
    } else if (argument == "--threads") {
      const std::int64_t parsed = parse_integer_argument(value, argument);
      if (parsed <= 0) {
        usage_error("--threads must be positive");
      }
      options.threads = static_cast<std::size_t>(parsed);
    } else if (argument == "--warmup") {
      const std::int64_t parsed = parse_integer_argument(value, argument);
      if (parsed < 0) {
        usage_error("--warmup must be non-negative");
      }
      options.warmup = static_cast<std::size_t>(parsed);
    } else if (argument == "--repeat") {
      const std::int64_t parsed = parse_integer_argument(value, argument);
      if (parsed <= 0) {
        usage_error("--repeat must be positive");
      }
      options.repeat = static_cast<std::size_t>(parsed);
    } else if (argument == "--cycle") {
      options.cycle = parse_integer_argument(value, argument);
    } else if (argument == "--candidate") {
      const std::int64_t parsed = parse_integer_argument(value, argument);
      if (parsed < std::numeric_limits<int>::min() ||
          parsed > std::numeric_limits<int>::max()) {
        usage_error("--candidate exceeds int range");
      }
      options.candidate = static_cast<int>(parsed);
    } else if (argument == "--role") {
      if (value != "Rollout" && value != "RewardModel") {
        usage_error("--role must be Rollout or RewardModel");
      }
      options.role = value;
    } else {
      usage_error("unknown argument: " + argument);
    }
  }
  if (options.input_path.empty()) {
    usage_error("--input is required");
  }
  return options;
}

class ThreadPool {
public:
  explicit ThreadPool(std::size_t thread_count) {
    if (thread_count <= 1) {
      return;
    }
    workers_.reserve(thread_count);
    for (std::size_t index = 0; index < thread_count; ++index) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (auto &worker : workers_) {
      worker.join();
    }
  }

  template <typename Function>
  auto submit(Function &&function) -> std::future<decltype(function())> {
    using Result = decltype(function());
    auto task = std::make_shared<std::packaged_task<Result()>>(
        std::forward<Function>(function));
    std::future<Result> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace_back([task] { (*task)(); });
    }
    condition_.notify_one();
    return future;
  }

  bool active() const { return !workers_.empty(); }

private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_ = false;
};

struct TaskKey {
  std::string role;
  RoleCandidatePlan plan;

  bool operator<(const TaskKey &other) const {
    return std::tie(role, plan) < std::tie(other.role, other.plan);
  }
};

struct TaskSpec {
  std::string role;
  RoleCandidatePlan plan;
  const RoleEvaluationContext *context = nullptr;
};

struct CandidateTaskIds {
  int candidate_index = 0;
  std::map<std::string, std::size_t> role_task_ids;
};

struct TaskBatch {
  std::vector<TaskSpec> tasks;
  std::map<std::string, std::size_t> baseline_task_ids;
  std::vector<CandidateTaskIds> candidate_task_ids;
  std::size_t logical_task_count = 0;
};

std::vector<std::string> selected_roles(const Options &options) {
  if (options.role.has_value()) {
    return {*options.role};
  }
  return {"Rollout", "RewardModel"};
}

TaskBatch
build_task_batch(const CycleInput &cycle,
                 const std::map<std::string, RoleEvaluationContext> &contexts,
                 const Options &options) {
  TaskBatch batch;
  std::map<TaskKey, std::size_t> task_ids;
  auto add_once = [&](const std::string &role, const RoleCandidatePlan &plan) {
    const TaskKey key{role, plan};
    const auto found = task_ids.find(key);
    if (found != task_ids.end()) {
      return found->second;
    }
    const std::size_t task_id = batch.tasks.size();
    task_ids.emplace(key, task_id);
    batch.tasks.push_back({role, plan, &contexts.at(role)});
    return task_id;
  };

  const std::vector<std::string> roles = selected_roles(options);
  for (const auto &role : roles) {
    batch.baseline_task_ids.emplace(role, add_once(role, RoleCandidatePlan{}));
    ++batch.logical_task_count;
  }
  for (const auto &candidate : cycle.candidates) {
    if (options.candidate.has_value() &&
        candidate.index != *options.candidate) {
      continue;
    }
    CandidateTaskIds task_ids_for_candidate;
    task_ids_for_candidate.candidate_index = candidate.index;
    for (const auto &role : roles) {
      const auto plan = candidate.role_plans.find(role);
      const RoleCandidatePlan empty;
      const RoleCandidatePlan &selected_plan =
          plan == candidate.role_plans.end() ? empty : plan->second;
      task_ids_for_candidate.role_task_ids.emplace(
          role, add_once(role, selected_plan));
      ++batch.logical_task_count;
    }
    batch.candidate_task_ids.push_back(std::move(task_ids_for_candidate));
  }
  return batch;
}

std::vector<RoleEvaluationResult> execute_batch(const TaskBatch &batch,
                                                ThreadPool &thread_pool) {
  std::vector<RoleEvaluationResult> results(batch.tasks.size());
  if (!thread_pool.active()) {
    for (std::size_t index = 0; index < batch.tasks.size(); ++index) {
      const auto &task = batch.tasks[index];
      results[index] = evaluate_role_candidate(*task.context, task.plan);
    }
    return results;
  }

  std::vector<std::future<RoleEvaluationResult>> futures;
  futures.reserve(batch.tasks.size());
  for (const auto &task : batch.tasks) {
    futures.push_back(thread_pool.submit(
        [task] { return evaluate_role_candidate(*task.context, task.plan); }));
  }
  for (std::size_t index = 0; index < futures.size(); ++index) {
    results[index] = futures[index].get();
  }
  return results;
}

struct BenchmarkSample {
  Nanoseconds context_preparation{0};
  Nanoseconds evaluation_wall{0};
  Nanoseconds rebalance_wall_union{0};
  Nanoseconds router_wall_union{0};
  Nanoseconds rebalance_router_wall_union{0};
  Nanoseconds rebalance_router_wall_overlap{0};
  Nanoseconds rebalance_worker{0};
  Nanoseconds router_worker{0};
  Nanoseconds other_worker{0};
  Nanoseconds total_wall{0};
};

Nanoseconds interval_union_wall(
    std::vector<std::pair<Clock::time_point, Clock::time_point>> intervals) {
  intervals.erase(
      std::remove_if(intervals.begin(), intervals.end(), [](const auto &item) {
        return item.second <= item.first;
      }),
      intervals.end());
  if (intervals.empty()) {
    return Nanoseconds{0};
  }
  std::sort(intervals.begin(), intervals.end(), [](const auto &lhs,
                                                   const auto &rhs) {
    return lhs.first < rhs.first ||
           (lhs.first == rhs.first && lhs.second < rhs.second);
  });
  Clock::time_point current_start = intervals.front().first;
  Clock::time_point current_end = intervals.front().second;
  Nanoseconds total{0};
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    const auto &[start, end] = intervals[index];
    if (start <= current_end) {
      current_end = std::max(current_end, end);
      continue;
    }
    total += std::chrono::duration_cast<Nanoseconds>(current_end - current_start);
    current_start = start;
    current_end = end;
  }
  total += std::chrono::duration_cast<Nanoseconds>(current_end - current_start);
  return total;
}

struct IterationResult {
  BenchmarkSample timing;
  TaskBatch batch;
  std::vector<RoleEvaluationResult> results;
};

IterationResult run_iteration(const CycleInput &cycle, const Options &options,
                              ThreadPool &thread_pool) {
  const auto preparation_started = Clock::now();
  std::map<std::string, RoleEvaluationContext> contexts;
  for (const auto &role : selected_roles(options)) {
    contexts.emplace(role,
                     prepare_role_evaluation_context(cycle.roles.at(role)));
  }
  const auto preparation_finished = Clock::now();

  const auto evaluation_started = Clock::now();
  TaskBatch batch = build_task_batch(cycle, contexts, options);
  std::vector<RoleEvaluationResult> results = execute_batch(batch, thread_pool);
  const auto evaluation_finished = Clock::now();

  BenchmarkSample sample;
  sample.context_preparation = std::chrono::duration_cast<Nanoseconds>(
      preparation_finished - preparation_started);
  sample.evaluation_wall = std::chrono::duration_cast<Nanoseconds>(
      evaluation_finished - evaluation_started);
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      rebalance_intervals;
  std::vector<std::pair<Clock::time_point, Clock::time_point>> router_intervals;
  rebalance_intervals.reserve(results.size());
  router_intervals.reserve(results.size());
  for (const auto &result : results) {
    sample.rebalance_worker += result.rebalance_time;
    sample.router_worker += result.router_time;
    sample.other_worker += result.other_time;
    if (result.rebalance_executed) {
      rebalance_intervals.emplace_back(result.rebalance_started_at,
                                       result.rebalance_finished_at);
    }
    router_intervals.emplace_back(result.router_started_at,
                                  result.router_finished_at);
  }
  std::vector<std::pair<Clock::time_point, Clock::time_point>>
      simulation_intervals = rebalance_intervals;
  simulation_intervals.insert(simulation_intervals.end(),
                              router_intervals.begin(),
                              router_intervals.end());
  sample.rebalance_wall_union = interval_union_wall(rebalance_intervals);
  sample.router_wall_union = interval_union_wall(router_intervals);
  sample.rebalance_router_wall_union =
      interval_union_wall(std::move(simulation_intervals));
  sample.rebalance_router_wall_overlap =
      sample.rebalance_wall_union + sample.router_wall_union -
      sample.rebalance_router_wall_union;
  sample.total_wall = sample.context_preparation + sample.evaluation_wall;
  return {sample, std::move(batch), std::move(results)};
}

struct TimingSummary {
  std::int64_t minimum = 0;
  double mean = 0.0;
  std::int64_t p50 = 0;
  std::int64_t p95 = 0;
};

TimingSummary summarize(std::vector<std::int64_t> values) {
  if (values.empty()) {
    return {};
  }
  std::sort(values.begin(), values.end());
  long double total = 0.0;
  for (const std::int64_t value : values) {
    total += static_cast<long double>(value);
  }
  const auto percentile = [&](double fraction) {
    const std::size_t rank = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(fraction * values.size())));
    return values[std::min(rank - 1, values.size() - 1)];
  };
  TimingSummary summary;
  summary.minimum = values.front();
  summary.mean = static_cast<double>(total / values.size());
  summary.p50 = percentile(0.50);
  summary.p95 = percentile(0.95);
  return summary;
}

struct BenchmarkResult {
  std::int64_t parse_time_ns = 0;
  std::vector<BenchmarkSample> samples;
  TaskBatch functional_batch;
  std::vector<RoleEvaluationResult> functional_results;
  std::uint64_t checksum = 0;
};

BenchmarkResult benchmark_cycle(const CycleInput &cycle, const Options &options,
                                ThreadPool &thread_pool,
                                std::int64_t parse_time_ns) {
  for (std::size_t index = 0; index < options.warmup; ++index) {
    static_cast<void>(run_iteration(cycle, options, thread_pool));
  }

  BenchmarkResult benchmark;
  benchmark.parse_time_ns = parse_time_ns;
  benchmark.samples.reserve(options.repeat);
  for (std::size_t index = 0; index < options.repeat; ++index) {
    IterationResult iteration = run_iteration(cycle, options, thread_pool);
    const std::uint64_t checksum = functional_checksum(iteration.results);
    if (index == 0) {
      benchmark.functional_batch = std::move(iteration.batch);
      benchmark.functional_results = std::move(iteration.results);
      benchmark.checksum = checksum;
    } else if (checksum != benchmark.checksum) {
      throw std::runtime_error("cycle " + std::to_string(cycle.cycle_id) +
                               " produced nondeterministic functional results");
    }
    benchmark.samples.push_back(iteration.timing);
  }
  return benchmark;
}

template <typename Writer>
void write_string(Writer &writer, const std::string &value) {
  writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

template <typename Writer>
void write_plan(Writer &writer, const RoleCandidatePlan &plan) {
  writer.StartObject();
  writer.Key("wake_instance_ids");
  writer.StartArray();
  for (const int instance_id : plan.wake_instance_ids) {
    writer.Int(instance_id);
  }
  writer.EndArray();
  writer.Key("sleep_instance_ids");
  writer.StartArray();
  for (const int instance_id : plan.sleep_instance_ids) {
    writer.Int(instance_id);
  }
  writer.EndArray();
  writer.Key("primary_scale_up");
  writer.Bool(plan.primary_scale_up);
  writer.EndObject();
}

template <typename Writer>
void write_role_result(Writer &writer, const RoleEvaluationResult &result,
                       bool emit_migrations) {
  writer.StartObject();
  writer.Key("throughput");
  writer.Double(result.throughput);
  writer.Key("routed_count");
  writer.Int64(result.routed_count);
  writer.Key("unrouted_count");
  writer.Int64(result.unrouted_count);
  writer.Key("rebalance_move_count");
  writer.Uint64(result.rebalance_moves.size());
  writer.Key("instance_throughputs");
  writer.StartArray();
  for (const auto &[instance_id, throughput] : result.instance_throughputs) {
    writer.StartObject();
    writer.Key("instance_id");
    writer.Int(instance_id);
    writer.Key("throughput");
    writer.Double(throughput);
    writer.EndObject();
  }
  writer.EndArray();
  if (emit_migrations) {
    writer.Key("rebalance_moves");
    writer.StartArray();
    for (const auto &move : result.rebalance_moves) {
      writer.StartObject();
      writer.Key("request_id");
      write_string(writer, move.request_id);
      writer.Key("source_instance_id");
      writer.Int(move.source_instance_id);
      writer.Key("destination_instance_id");
      writer.Int(move.destination_instance_id);
      writer.EndObject();
    }
    writer.EndArray();
  }
  writer.EndObject();
}

template <typename Writer, typename Getter>
void write_summary(Writer &writer, const std::vector<BenchmarkSample> &samples,
                   Getter getter) {
  std::vector<std::int64_t> values;
  values.reserve(samples.size());
  for (const auto &sample : samples) {
    values.push_back(getter(sample).count());
  }
  const TimingSummary summary = summarize(std::move(values));
  writer.StartObject();
  writer.Key("min");
  writer.Int64(summary.minimum);
  writer.Key("mean");
  writer.Double(summary.mean);
  writer.Key("p50");
  writer.Int64(summary.p50);
  writer.Key("p95");
  writer.Int64(summary.p95);
  writer.EndObject();
}

template <typename Writer>
void write_benchmark(Writer &writer, const CycleInput &cycle,
                     const Options &options, const BenchmarkResult &benchmark) {
  writer.StartObject();
  writer.Key("schema_version");
  writer.Int(1);
  writer.Key("record_type");
  writer.String("elastic_simulation_benchmark");
  writer.Key("cycle_id");
  writer.Int64(cycle.cycle_id);
  writer.Key("config");
  writer.StartObject();
  writer.Key("threads");
  writer.Uint64(options.threads);
  writer.Key("warmup");
  writer.Uint64(options.warmup);
  writer.Key("repeat");
  writer.Uint64(options.repeat);
  writer.Key("role_filter");
  if (options.role.has_value()) {
    write_string(writer, *options.role);
  } else {
    writer.Null();
  }
  writer.Key("candidate_filter");
  if (options.candidate.has_value()) {
    writer.Int(*options.candidate);
  } else {
    writer.Null();
  }
  writer.Key("emit_migrations");
  writer.Bool(options.emit_migrations);
  writer.EndObject();

  writer.Key("counts");
  writer.StartObject();
  writer.Key("candidates");
  writer.Uint64(benchmark.functional_batch.candidate_task_ids.size());
  writer.Key("logical_tasks");
  writer.Uint64(benchmark.functional_batch.logical_task_count);
  writer.Key("unique_tasks");
  writer.Uint64(benchmark.functional_batch.tasks.size());
  writer.Key("deduplicated_tasks");
  writer.Uint64(benchmark.functional_batch.logical_task_count -
                benchmark.functional_batch.tasks.size());
  writer.EndObject();

  writer.Key("timing_ns");
  writer.StartObject();
  writer.Key("json_parse");
  writer.Int64(benchmark.parse_time_ns);
  writer.Key("context_preparation");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.context_preparation; });
  writer.Key("candidate_evaluation_wall");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.evaluation_wall; });
  writer.Key("rebalance_wall_union");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.rebalance_wall_union; });
  writer.Key("router_wall_union");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.router_wall_union; });
  writer.Key("rebalance_router_wall_union");
  write_summary(writer, benchmark.samples, [](const auto &sample) {
    return sample.rebalance_router_wall_union;
  });
  writer.Key("rebalance_router_wall_overlap");
  write_summary(writer, benchmark.samples, [](const auto &sample) {
    return sample.rebalance_router_wall_overlap;
  });
  writer.Key("rebalance_worker_sum");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.rebalance_worker; });
  writer.Key("router_worker_sum");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.router_worker; });
  writer.Key("other_worker_sum");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.other_worker; });
  writer.Key("total_wall");
  write_summary(writer, benchmark.samples,
                [](const auto &sample) { return sample.total_wall; });
  writer.EndObject();

  writer.Key("reference_timing_seconds");
  if (cycle.reference_timing_seconds.empty() &&
      !cycle.reference_timing_source_timestamp.has_value()) {
    writer.Null();
  } else {
    writer.StartObject();
    for (const auto &[name, value] : cycle.reference_timing_seconds) {
      write_string(writer, name);
      writer.Double(value);
    }
    if (cycle.reference_timing_source_timestamp.has_value()) {
      writer.Key("source_timestamp");
      write_string(writer, *cycle.reference_timing_source_timestamp);
    }
    writer.EndObject();
  }

  std::ostringstream checksum;
  checksum << std::hex << std::setfill('0') << std::setw(16)
           << benchmark.checksum;
  writer.Key("functional_checksum");
  write_string(writer, checksum.str());
  writer.Key("functional_results");
  writer.StartObject();
  writer.Key("baseline");
  writer.StartObject();
  for (const auto &[role, task_id] :
       benchmark.functional_batch.baseline_task_ids) {
    write_string(writer, role);
    write_role_result(writer, benchmark.functional_results.at(task_id),
                      options.emit_migrations);
  }
  writer.EndObject();
  writer.Key("candidates");
  writer.StartArray();
  for (const auto &candidate : benchmark.functional_batch.candidate_task_ids) {
    writer.StartObject();
    writer.Key("index");
    writer.Int(candidate.candidate_index);
    writer.Key("roles");
    writer.StartObject();
    for (const auto &[role, task_id] : candidate.role_task_ids) {
      write_string(writer, role);
      writer.StartObject();
      writer.Key("plan");
      write_plan(writer, benchmark.functional_batch.tasks.at(task_id).plan);
      writer.Key("result");
      write_role_result(writer, benchmark.functional_results.at(task_id),
                        options.emit_migrations);
      writer.EndObject();
    }
    writer.EndObject();
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();
  writer.EndObject();
}

} // namespace
} // namespace psrl::elastic_simulator

int main(int argc, char **argv) {
  using namespace psrl::elastic_simulator;
  try {
    const Options options = parse_options(argc, argv);
    std::ifstream input_file;
    std::istream *input = &std::cin;
    if (options.input_path != "-") {
      input_file.open(options.input_path);
      if (!input_file) {
        throw std::runtime_error("failed to open input: " + options.input_path);
      }
      input = &input_file;
    }

    std::ofstream output_file;
    std::ostream *output = &std::cout;
    if (options.output_path.has_value() && *options.output_path != "-") {
      output_file.open(*options.output_path);
      if (!output_file) {
        throw std::runtime_error("failed to open output: " +
                                 *options.output_path);
      }
      output = &output_file;
    }
    rapidjson::OStreamWrapper output_wrapper(*output);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(output_wrapper);
    ThreadPool thread_pool(options.threads);

    std::string line;
    std::size_t line_number = 0;
    std::size_t emitted = 0;
    bool candidate_matched = !options.candidate.has_value();
    while (std::getline(*input, line)) {
      ++line_number;
      if (line.empty()) {
        continue;
      }
      const auto parse_started = Clock::now();
      CycleInput cycle = parse_cycle_json(line, line_number);
      const auto parse_finished = Clock::now();
      if (options.cycle.has_value() && cycle.cycle_id != *options.cycle) {
        continue;
      }
      if (options.candidate.has_value()) {
        candidate_matched =
            std::any_of(cycle.candidates.begin(), cycle.candidates.end(),
                        [&](const auto &candidate) {
                          return candidate.index == *options.candidate;
                        });
        if (!candidate_matched) {
          continue;
        }
      }
      const std::int64_t parse_time_ns =
          std::chrono::duration_cast<Nanoseconds>(parse_finished -
                                                  parse_started)
              .count();
      const BenchmarkResult benchmark =
          benchmark_cycle(cycle, options, thread_pool, parse_time_ns);
      write_benchmark(writer, cycle, options, benchmark);
      *output << '\n';
      output->flush();
      ++emitted;
    }
    if (emitted == 0) {
      if (!candidate_matched) {
        throw std::runtime_error("no input record matched --candidate");
      }
      throw std::runtime_error("no input record matched the selected filters");
    }
    std::cerr << "benchmarked " << emitted << " cycle record(s)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
