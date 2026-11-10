#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace psrl::elastic_simulator {

struct PriorityValue {
  enum class Kind { kNull, kNumber, kString, kArray };

  Kind kind = Kind::kNull;
  double number = 0.0;
  std::string string;
  std::vector<PriorityValue> array;

  static PriorityValue null();
  static PriorityValue number_value(double value);
  static PriorityValue string_value(std::string value);
  static PriorityValue array_value(std::vector<PriorityValue> value);
};

int compare_priority(const PriorityValue &lhs, const PriorityValue &rhs);

struct RequestSnapshot {
  std::string request_id;
  std::int64_t seq_len = 0;
  std::optional<int> source_instance_id;
  bool is_waiting = false;
  bool movable = true;
  std::int64_t route_order = 0;
  std::vector<PriorityValue> routing_priority;
  std::optional<std::vector<int>> eligible_instance_ids;
  std::optional<std::vector<int>> fallback_instance_ids;
  std::vector<std::pair<int, PriorityValue>> candidate_priorities;
};

struct InstanceSnapshot {
  int instance_id = 0;
  bool is_awake = false;
  std::int64_t model_version = 0;
  std::vector<RequestSnapshot> requests;
  std::int64_t route_request_count = 0;
  std::int64_t running_count = 0;
  std::int64_t waiting_count = 0;
  std::int64_t token_count = 0;
  std::int64_t max_model_len = std::numeric_limits<std::int64_t>::max();
  std::array<double, 4> throughput_params{0.0, 1.0, 1.0, 0.0};
  std::optional<std::array<double, 5>> route_cost_params;
};

enum class RoutingStrategy { kThroughputOptimal, kItl };

struct RoleSnapshot {
  std::string role;
  RoutingStrategy strategy = RoutingStrategy::kThroughputOptimal;
  std::vector<InstanceSnapshot> instances;
  std::vector<RequestSnapshot> pending_requests;
  std::string queue_scope = "running";
  std::optional<std::int64_t> max_concurrent_requests;
  double delta_throughput_threshold = 0.0;
  std::optional<double> itl_max_itl;
};

struct RoleCandidatePlan {
  std::set<int> wake_instance_ids;
  std::set<int> sleep_instance_ids;
  bool primary_scale_up = false;

  bool operator==(const RoleCandidatePlan &other) const;
  bool operator<(const RoleCandidatePlan &other) const;
};

struct CandidateInput {
  int index = 0;
  std::map<std::string, RoleCandidatePlan> role_plans;
};

struct CycleInput {
  std::int64_t cycle_id = 0;
  std::map<std::string, RoleSnapshot> roles;
  std::vector<CandidateInput> candidates;
  std::map<std::string, double> reference_timing_seconds;
  std::optional<std::string> reference_timing_source_timestamp;
};

struct RequestMigration {
  std::string request_id;
  int source_instance_id = 0;
  int destination_instance_id = 0;

  bool operator==(const RequestMigration &other) const;
};

struct RoleEvaluationResult {
  double throughput = 0.0;
  std::int64_t routed_count = 0;
  std::int64_t unrouted_count = 0;
  std::vector<RequestMigration> rebalance_moves;
  std::vector<std::pair<int, double>> instance_throughputs;
  std::chrono::nanoseconds rebalance_time{0};
  std::chrono::nanoseconds router_time{0};
  std::chrono::nanoseconds other_time{0};
  bool rebalance_executed = false;
  std::chrono::steady_clock::time_point rebalance_started_at{};
  std::chrono::steady_clock::time_point rebalance_finished_at{};
  std::chrono::steady_clock::time_point router_started_at{};
  std::chrono::steady_clock::time_point router_finished_at{};
};

struct PreparedRequest {
  RequestSnapshot snapshot;
  std::vector<PriorityValue> route_priority_key;
  bool request_id_is_numeric = false;
  bool request_id_is_negative = false;
  std::string request_id_digits;
  std::vector<std::vector<int>> eligible_groups;
  std::optional<std::vector<std::vector<int>>> fallback_groups;
};

struct PreparedInstance {
  InstanceSnapshot snapshot;
  std::int64_t visible_request_count = 0;
  std::int64_t visible_token_count = 0;
  std::vector<PreparedRequest> movable_requests;
  std::vector<PreparedRequest> reroute_requests;
};

struct RoleEvaluationContext {
  std::string role;
  RoutingStrategy strategy = RoutingStrategy::kThroughputOptimal;
  std::string queue_scope;
  std::optional<std::int64_t> max_concurrent_requests;
  double delta_throughput_threshold = 0.0;
  std::optional<double> itl_max_itl;
  std::vector<PreparedInstance> instances;
  std::vector<int> all_instance_ids;
  std::unordered_map<int, std::size_t> index_by_id;
  std::set<int> before_awake;
  std::vector<PreparedRequest> pending_requests;
};

RoleEvaluationContext
prepare_role_evaluation_context(const RoleSnapshot &snapshot);

RoleEvaluationResult
evaluate_role_candidate(const RoleEvaluationContext &context,
                        const RoleCandidatePlan &plan);

std::uint64_t
functional_checksum(const std::vector<RoleEvaluationResult> &results);

} // namespace psrl::elastic_simulator
