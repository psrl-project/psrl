#include "elastic_simulator/simulator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace psrl::elastic_simulator {
namespace {

using Clock = std::chrono::steady_clock;

struct PriorityLess {
  bool operator()(const PriorityValue &lhs, const PriorityValue &rhs) const {
    return compare_priority(lhs, rhs) < 0;
  }
};

struct NumericRequestId {
  bool numeric = false;
  bool negative = false;
  std::string digits;
};

NumericRequestId parse_numeric_request_id(const std::string &request_id) {
  std::size_t begin = 0;
  std::size_t end = request_id.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(request_id[begin]))) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(request_id[end - 1]))) {
    --end;
  }
  bool negative = false;
  if (begin < end && (request_id[begin] == '+' || request_id[begin] == '-')) {
    negative = request_id[begin] == '-';
    ++begin;
  }
  if (begin == end) {
    return {};
  }
  for (std::size_t index = begin; index < end; ++index) {
    if (!std::isdigit(static_cast<unsigned char>(request_id[index]))) {
      return {};
    }
  }
  while (begin < end && request_id[begin] == '0') {
    ++begin;
  }
  std::string digits =
      begin == end ? "0" : request_id.substr(begin, end - begin);
  if (digits == "0") {
    negative = false;
  }
  NumericRequestId result;
  result.numeric = true;
  result.negative = negative;
  result.digits = std::move(digits);
  return result;
}

int compare_numeric_request_id(bool lhs_negative, const std::string &lhs_digits,
                               bool rhs_negative,
                               const std::string &rhs_digits) {
  if (lhs_negative != rhs_negative) {
    return lhs_negative ? -1 : 1;
  }
  if (lhs_digits.size() != rhs_digits.size()) {
    const int result = lhs_digits.size() < rhs_digits.size() ? -1 : 1;
    return lhs_negative ? -result : result;
  }
  if (lhs_digits != rhs_digits) {
    const int result = lhs_digits < rhs_digits ? -1 : 1;
    return lhs_negative ? -result : result;
  }
  return 0;
}

int compare_request_id(const PreparedRequest &lhs, const PreparedRequest &rhs) {
  if (lhs.request_id_is_numeric != rhs.request_id_is_numeric) {
    return lhs.request_id_is_numeric ? -1 : 1;
  }
  if (lhs.request_id_is_numeric) {
    const int numeric_comparison = compare_numeric_request_id(
        lhs.request_id_is_negative, lhs.request_id_digits,
        rhs.request_id_is_negative, rhs.request_id_digits);
    if (numeric_comparison != 0) {
      return numeric_comparison;
    }
  }
  if (lhs.snapshot.request_id == rhs.snapshot.request_id) {
    return 0;
  }
  return lhs.snapshot.request_id < rhs.snapshot.request_id ? -1 : 1;
}

int compare_priority_vector(const std::vector<PriorityValue> &lhs,
                            const std::vector<PriorityValue> &rhs) {
  const std::size_t common = std::min(lhs.size(), rhs.size());
  for (std::size_t index = 0; index < common; ++index) {
    const int comparison = compare_priority(lhs[index], rhs[index]);
    if (comparison != 0) {
      return comparison;
    }
  }
  if (lhs.size() == rhs.size()) {
    return 0;
  }
  return lhs.size() < rhs.size() ? -1 : 1;
}

int compare_route_key(const PreparedRequest &lhs, const PreparedRequest &rhs) {
  const int priority_comparison =
      compare_priority_vector(lhs.route_priority_key, rhs.route_priority_key);
  if (priority_comparison != 0) {
    return priority_comparison;
  }
  if (lhs.snapshot.route_order != rhs.snapshot.route_order) {
    return lhs.snapshot.route_order < rhs.snapshot.route_order ? -1 : 1;
  }
  return compare_request_id(lhs, rhs);
}

std::vector<std::vector<int>>
group_candidates(const std::vector<int> &candidates,
                 const std::vector<std::pair<int, PriorityValue>> &priorities) {
  std::unordered_map<int, PriorityValue> priority_by_id;
  for (const auto &[instance_id, priority] : priorities) {
    priority_by_id[instance_id] = priority;
  }
  std::map<PriorityValue, std::vector<int>, PriorityLess> groups;
  for (const int instance_id : candidates) {
    const auto found = priority_by_id.find(instance_id);
    const PriorityValue priority = found == priority_by_id.end()
                                       ? PriorityValue::number_value(0.0)
                                       : found->second;
    groups[priority].push_back(instance_id);
  }
  std::vector<std::vector<int>> result;
  result.reserve(groups.size());
  for (auto &[unused, group] : groups) {
    static_cast<void>(unused);
    result.push_back(std::move(group));
  }
  return result;
}

PreparedRequest prepare_request(const RequestSnapshot &request,
                                const std::vector<int> &all_instance_ids,
                                bool prepare_rollout_groups) {
  PreparedRequest prepared;
  prepared.snapshot = request;
  prepared.route_priority_key = request.routing_priority;
  if (prepared.route_priority_key.empty()) {
    prepared.route_priority_key.push_back(
        PriorityValue::number_value(static_cast<double>(request.route_order)));
  }
  const NumericRequestId request_id_key =
      parse_numeric_request_id(request.request_id);
  prepared.request_id_is_numeric = request_id_key.numeric;
  prepared.request_id_is_negative = request_id_key.negative;
  prepared.request_id_digits = request_id_key.digits;
  if (!prepare_rollout_groups) {
    return prepared;
  }
  const auto &eligible = request.eligible_instance_ids.has_value()
                             ? *request.eligible_instance_ids
                             : all_instance_ids;
  prepared.eligible_groups =
      group_candidates(eligible, request.candidate_priorities);
  if (request.fallback_instance_ids.has_value()) {
    prepared.fallback_groups = group_candidates(*request.fallback_instance_ids,
                                                request.candidate_priorities);
  }
  return prepared;
}

double compute_itl(const std::array<double, 4> &params, std::int64_t tokens,
                   std::int64_t requests) {
  const double nonnegative_tokens =
      static_cast<double>(std::max<std::int64_t>(0, tokens));
  const double nonnegative_requests =
      static_cast<double>(std::max<std::int64_t>(0, requests));
  return std::max(params[0] * nonnegative_tokens +
                      std::max(params[1], params[2] * nonnegative_requests) +
                      params[3],
                  1e-9);
}

struct InstanceState {
  const PreparedInstance *prepared = nullptr;
  std::int64_t throughput_request_count = 0;
  std::int64_t throughput_token_count = 0;
  std::int64_t route_request_count = 0;
  std::int64_t running_count = 0;
  std::int64_t waiting_count = 0;
  std::int64_t token_count = 0;
  bool active = false;
};

double throughput_for_load(const InstanceState &state, std::int64_t requests,
                           std::int64_t tokens) {
  if (requests <= 0) {
    return 0.0;
  }
  return static_cast<double>(requests) /
         compute_itl(state.prepared->snapshot.throughput_params, tokens,
                     requests);
}

double instance_throughput(const InstanceState &state) {
  return throughput_for_load(state, state.throughput_request_count,
                             state.throughput_token_count);
}

double route_latency(const InstanceState &state, std::int64_t running,
                     std::int64_t tokens) {
  const auto &route_cost = state.prepared->snapshot.route_cost_params;
  if (!route_cost.has_value()) {
    return compute_itl(state.prepared->snapshot.throughput_params, tokens,
                       running);
  }
  const auto &params = *route_cost;
  return std::max(
      params[3] + params[4] * static_cast<double>(tokens) +
          std::max(params[0],
                   params[1] + params[2] * static_cast<double>(running)),
      1e-9);
}

InstanceState &state_for(const RoleEvaluationContext &context,
                         std::vector<InstanceState> &states, int instance_id) {
  const auto found = context.index_by_id.find(instance_id);
  if (found == context.index_by_id.end() || !states[found->second].active) {
    throw std::runtime_error("active instance state is missing: " +
                             std::to_string(instance_id));
  }
  return states[found->second];
}

const InstanceState &state_for(const RoleEvaluationContext &context,
                               const std::vector<InstanceState> &states,
                               int instance_id) {
  const auto found = context.index_by_id.find(instance_id);
  if (found == context.index_by_id.end() || !states[found->second].active) {
    throw std::runtime_error("active instance state is missing: " +
                             std::to_string(instance_id));
  }
  return states[found->second];
}

bool rollout_can_run_directly(const InstanceState &state,
                              const RequestSnapshot &request) {
  if (state.waiting_count > 0) {
    return false;
  }
  const std::int64_t available =
      state.prepared->snapshot.max_model_len - state.token_count;
  return request.seq_len <= available;
}

std::vector<std::vector<int>>
filter_active_groups(const std::vector<std::vector<int>> &groups,
                     const std::set<int> &active_ids) {
  std::vector<std::vector<int>> filtered_groups;
  for (const auto &group : groups) {
    std::vector<int> filtered;
    for (const int instance_id : group) {
      if (active_ids.count(instance_id) != 0) {
        filtered.push_back(instance_id);
      }
    }
    if (!filtered.empty()) {
      filtered_groups.push_back(std::move(filtered));
    }
  }
  return filtered_groups;
}

std::optional<int> rollout_select(const RoleEvaluationContext &context,
                                  const std::vector<InstanceState> &states,
                                  const std::set<int> &active_ids,
                                  const PreparedRequest &request) {
  std::vector<std::vector<int>> groups =
      filter_active_groups(request.eligible_groups, active_ids);
  if (groups.empty() && request.fallback_groups.has_value()) {
    groups = filter_active_groups(*request.fallback_groups, active_ids);
  }
  if (groups.empty()) {
    return std::nullopt;
  }

  for (const auto &group : groups) {
    const InstanceState &threshold_state =
        state_for(context, states, group.front());
    const double threshold =
        context.delta_throughput_threshold /
        route_latency(threshold_state, 1, request.snapshot.seq_len);
    std::optional<int> best_candidate;
    double best_delta = -std::numeric_limits<double>::infinity();
    for (const int instance_id : group) {
      const InstanceState &state = state_for(context, states, instance_id);
      if (!rollout_can_run_directly(state, request.snapshot)) {
        continue;
      }
      const double current =
          static_cast<double>(state.running_count) /
          route_latency(state, state.running_count, state.token_count);
      const std::int64_t next_running = state.running_count + 1;
      const std::int64_t next_tokens =
          state.token_count + request.snapshot.seq_len;
      const double next = static_cast<double>(next_running) /
                          route_latency(state, next_running, next_tokens);
      const double delta = next - current;
      if (delta > best_delta) {
        best_delta = delta;
        best_candidate = instance_id;
      }
    }
    if (best_candidate.has_value() && best_delta >= threshold) {
      const InstanceState &best_state =
          state_for(context, states, *best_candidate);
      if (!context.max_concurrent_requests.has_value() ||
          best_state.route_request_count < *context.max_concurrent_requests) {
        return best_candidate;
      }
    }
  }
  return std::nullopt;
}

bool rm_can_admit(const RoleEvaluationContext &context,
                  const InstanceState &state) {
  const std::int64_t load = state.route_request_count;
  return !context.max_concurrent_requests.has_value() ||
         load < *context.max_concurrent_requests;
}

struct RmHeapEntry {
  double estimated_itl = 0.0;
  std::int64_t load = 0;
  int instance_id = 0;
  std::uint64_t version = 0;
};

struct RmHeapGreater {
  bool operator()(const RmHeapEntry &lhs, const RmHeapEntry &rhs) const {
    return std::tie(lhs.estimated_itl, lhs.load, lhs.instance_id, lhs.version) >
           std::tie(rhs.estimated_itl, rhs.load, rhs.instance_id, rhs.version);
  }
};

class RmSelectorHeap {
public:
  RmSelectorHeap(const RoleEvaluationContext &context,
                 std::vector<InstanceState> &states,
                 const std::set<int> &active_ids)
      : context_(context), states_(states), versions_(states.size(), 0) {
    for (const int instance_id : active_ids) {
      push(instance_id);
    }
  }

  std::optional<int> select(const RequestSnapshot &request) {
    if (!request.eligible_instance_ids.has_value()) {
      return peek_global();
    }
    std::optional<std::tuple<double, std::int64_t, int>> best;
    for (const int instance_id : *request.eligible_instance_ids) {
      const auto found = context_.index_by_id.find(instance_id);
      if (found == context_.index_by_id.end() ||
          !states_[found->second].active) {
        continue;
      }
      const InstanceState &state = states_[found->second];
      if (!rm_can_admit(context_, state)) {
        continue;
      }
      const std::int64_t load = state.route_request_count;
      const double estimated_itl =
          compute_itl(state.prepared->snapshot.throughput_params, 0, load + 1);
      if (context_.itl_max_itl.has_value() &&
          estimated_itl > *context_.itl_max_itl) {
        continue;
      }
      const auto key = std::make_tuple(estimated_itl, load, instance_id);
      if (!best.has_value() || key < *best) {
        best = key;
      }
    }
    return best.has_value() ? std::optional<int>(std::get<2>(*best))
                            : std::nullopt;
  }

  void instance_changed(int instance_id) {
    const std::size_t index = context_.index_by_id.at(instance_id);
    ++versions_[index];
    push(instance_id);
  }

private:
  void push(int instance_id) {
    const std::size_t index = context_.index_by_id.at(instance_id);
    const InstanceState &state = states_[index];
    if (!rm_can_admit(context_, state)) {
      return;
    }
    const std::int64_t load = state.route_request_count;
    const double estimated_itl =
        compute_itl(state.prepared->snapshot.throughput_params, 0, load + 1);
    if (context_.itl_max_itl.has_value() &&
        estimated_itl > *context_.itl_max_itl) {
      return;
    }
    heap_.push({estimated_itl, load, instance_id, versions_[index]});
  }

  std::optional<int> peek_global() {
    while (!heap_.empty()) {
      const RmHeapEntry &entry = heap_.top();
      const std::size_t index = context_.index_by_id.at(entry.instance_id);
      if (entry.version != versions_[index]) {
        heap_.pop();
        continue;
      }
      return entry.instance_id;
    }
    return std::nullopt;
  }

  const RoleEvaluationContext &context_;
  std::vector<InstanceState> &states_;
  std::vector<std::uint64_t> versions_;
  std::priority_queue<RmHeapEntry, std::vector<RmHeapEntry>, RmHeapGreater>
      heap_;
};

bool dispatch_request(const RoleEvaluationContext &context,
                      std::vector<InstanceState> &states,
                      const std::set<int> &active_ids,
                      const PreparedRequest &request,
                      RmSelectorHeap *rm_selector) {
  std::optional<int> destination;
  if (context.strategy == RoutingStrategy::kThroughputOptimal) {
    destination = rollout_select(context, states, active_ids, request);
  } else {
    if (rm_selector == nullptr) {
      throw std::runtime_error("RM selector heap was not initialized");
    }
    destination = rm_selector->select(request.snapshot);
  }
  if (!destination.has_value()) {
    return false;
  }

  InstanceState &state = state_for(context, states, *destination);
  ++state.throughput_request_count;
  state.throughput_token_count += request.snapshot.seq_len;
  ++state.route_request_count;
  if (context.strategy == RoutingStrategy::kThroughputOptimal) {
    ++state.running_count;
    state.token_count += request.snapshot.seq_len;
  } else {
    rm_selector->instance_changed(*destination);
  }
  return true;
}

struct LoadHeapEntry {
  double load = 0.0;
  int instance_id = 0;
  std::uint64_t version = 0;
};

struct LoadHeapGreater {
  bool operator()(const LoadHeapEntry &lhs, const LoadHeapEntry &rhs) const {
    return std::tie(lhs.load, lhs.instance_id, lhs.version) >
           std::tie(rhs.load, rhs.instance_id, rhs.version);
  }
};

using LoadHeap = std::priority_queue<LoadHeapEntry, std::vector<LoadHeapEntry>,
                                     LoadHeapGreater>;

std::optional<int>
peek_valid_donor(const RoleEvaluationContext &context, LoadHeap &donor_heap,
                 const std::vector<std::uint64_t> &versions,
                 std::vector<std::size_t> &cursors,
                 const std::unordered_set<std::string> &migrated_request_ids) {
  while (!donor_heap.empty()) {
    const LoadHeapEntry &entry = donor_heap.top();
    const std::size_t index = context.index_by_id.at(entry.instance_id);
    const auto &requests = context.instances[index].movable_requests;
    std::size_t cursor = cursors[index];
    while (cursor < requests.size() &&
           migrated_request_ids.count(requests[cursor].snapshot.request_id) !=
               0) {
      ++cursor;
    }
    cursors[index] = cursor;
    if (entry.version != versions[index] || cursor >= requests.size()) {
      donor_heap.pop();
      continue;
    }
    return entry.instance_id;
  }
  return std::nullopt;
}

std::optional<int> peek_valid_destination_excluding(
    const RoleEvaluationContext &context, LoadHeap &destination_heap,
    const std::vector<std::uint64_t> &versions, int excluded_instance_id) {
  std::vector<LoadHeapEntry> held;
  std::optional<int> destination;
  while (!destination_heap.empty()) {
    const LoadHeapEntry entry = destination_heap.top();
    const std::size_t index = context.index_by_id.at(entry.instance_id);
    if (entry.version != versions[index]) {
      destination_heap.pop();
      continue;
    }
    if (entry.instance_id == excluded_instance_id) {
      held.push_back(entry);
      destination_heap.pop();
      continue;
    }
    destination = entry.instance_id;
    break;
  }
  for (const auto &entry : held) {
    destination_heap.push(entry);
  }
  return destination;
}

std::vector<RequestMigration> rebalance(const RoleEvaluationContext &context,
                                        std::vector<InstanceState> &states,
                                        const std::set<int> &donor_ids,
                                        std::size_t active_count) {
  if (active_count <= 1) {
    return {};
  }

  std::vector<std::size_t> cursors(context.instances.size(), 0);
  std::vector<std::uint64_t> versions(context.instances.size(), 0);
  LoadHeap donor_heap;
  LoadHeap destination_heap;
  for (const int donor_id : donor_ids) {
    const std::size_t index = context.index_by_id.at(donor_id);
    if (!context.instances[index].movable_requests.empty()) {
      donor_heap.push({-instance_throughput(states[index]), donor_id, 0});
    }
  }
  for (const int instance_id : context.all_instance_ids) {
    const std::size_t index = context.index_by_id.at(instance_id);
    if (states[index].active) {
      destination_heap.push(
          {instance_throughput(states[index]), instance_id, 0});
    }
  }

  std::vector<RequestMigration> moves;
  std::unordered_set<std::string> migrated_request_ids;
  while (!donor_heap.empty()) {
    const auto donor_id = peek_valid_donor(context, donor_heap, versions,
                                           cursors, migrated_request_ids);
    if (!donor_id.has_value()) {
      break;
    }
    const auto destination_id = peek_valid_destination_excluding(
        context, destination_heap, versions, *donor_id);
    if (!destination_id.has_value()) {
      break;
    }

    const std::size_t donor_index = context.index_by_id.at(*donor_id);
    const std::size_t destination_index =
        context.index_by_id.at(*destination_id);
    const PreparedRequest &request =
        context.instances[donor_index].movable_requests[cursors[donor_index]];
    const RequestSnapshot &snapshot = request.snapshot;
    InstanceState &donor = states[donor_index];
    InstanceState &destination = states[destination_index];
    const double current_pair =
        instance_throughput(donor) + instance_throughput(destination);
    const double next_pair =
        throughput_for_load(donor, donor.throughput_request_count - 1,
                            donor.throughput_token_count - snapshot.seq_len) +
        throughput_for_load(
            destination, destination.throughput_request_count + 1,
            destination.throughput_token_count + snapshot.seq_len);
    if (next_pair <= current_pair) {
      break;
    }

    ++cursors[donor_index];
    migrated_request_ids.insert(snapshot.request_id);
    --donor.throughput_request_count;
    donor.throughput_token_count -= snapshot.seq_len;
    donor.route_request_count =
        std::max<std::int64_t>(0, donor.route_request_count - 1);
    donor.running_count = std::max<std::int64_t>(
        0, donor.running_count - (snapshot.is_waiting ? 0 : 1));
    donor.waiting_count = std::max<std::int64_t>(
        0, donor.waiting_count - (snapshot.is_waiting ? 1 : 0));
    donor.token_count =
        std::max<std::int64_t>(0, donor.token_count - snapshot.seq_len);

    ++destination.throughput_request_count;
    destination.throughput_token_count += snapshot.seq_len;
    ++destination.route_request_count;
    ++destination.running_count;
    destination.token_count += snapshot.seq_len;
    moves.push_back({snapshot.request_id, *donor_id, *destination_id});

    const std::set<int> changed_ids = {*donor_id, *destination_id};
    for (const int instance_id : changed_ids) {
      const std::size_t index = context.index_by_id.at(instance_id);
      const std::uint64_t version = ++versions[index];
      const double throughput = instance_throughput(states[index]);
      destination_heap.push({throughput, instance_id, version});
      if (donor_ids.count(instance_id) == 0) {
        continue;
      }
      const auto &requests = context.instances[index].movable_requests;
      std::size_t cursor = cursors[index];
      while (cursor < requests.size() &&
             migrated_request_ids.count(requests[cursor].snapshot.request_id) !=
                 0) {
        ++cursor;
      }
      cursors[index] = cursor;
      if (cursor < requests.size()) {
        donor_heap.push({-throughput, instance_id, version});
      }
    }
  }
  return moves;
}

struct RouteStreamEntry {
  const std::vector<PreparedRequest> *stream = nullptr;
  std::size_t stream_order = 0;
  std::size_t request_index = 0;
};

struct RouteStreamGreater {
  bool operator()(const RouteStreamEntry &lhs,
                  const RouteStreamEntry &rhs) const {
    const int comparison = compare_route_key((*lhs.stream)[lhs.request_index],
                                             (*rhs.stream)[rhs.request_index]);
    if (comparison != 0) {
      return comparison > 0;
    }
    return lhs.stream_order > rhs.stream_order;
  }
};

std::uint64_t fnv_append(std::uint64_t hash, const void *data,
                         std::size_t size) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= kPrime;
  }
  return hash;
}

template <typename T>
std::uint64_t fnv_value(std::uint64_t hash, const T &value) {
  return fnv_append(hash, &value, sizeof(value));
}

} // namespace

PriorityValue PriorityValue::null() { return {}; }

PriorityValue PriorityValue::number_value(double value) {
  PriorityValue result;
  result.kind = Kind::kNumber;
  result.number = value;
  return result;
}

PriorityValue PriorityValue::string_value(std::string value) {
  PriorityValue result;
  result.kind = Kind::kString;
  result.string = std::move(value);
  return result;
}

PriorityValue PriorityValue::array_value(std::vector<PriorityValue> value) {
  PriorityValue result;
  result.kind = Kind::kArray;
  result.array = std::move(value);
  return result;
}

int compare_priority(const PriorityValue &lhs, const PriorityValue &rhs) {
  if (lhs.kind != rhs.kind) {
    return lhs.kind < rhs.kind ? -1 : 1;
  }
  switch (lhs.kind) {
  case PriorityValue::Kind::kNull:
    return 0;
  case PriorityValue::Kind::kNumber:
    if (lhs.number == rhs.number) {
      return 0;
    }
    return lhs.number < rhs.number ? -1 : 1;
  case PriorityValue::Kind::kString:
    if (lhs.string == rhs.string) {
      return 0;
    }
    return lhs.string < rhs.string ? -1 : 1;
  case PriorityValue::Kind::kArray:
    return compare_priority_vector(lhs.array, rhs.array);
  }
  return 0;
}

bool RoleCandidatePlan::operator==(const RoleCandidatePlan &other) const {
  return wake_instance_ids == other.wake_instance_ids &&
         sleep_instance_ids == other.sleep_instance_ids &&
         primary_scale_up == other.primary_scale_up;
}

bool RoleCandidatePlan::operator<(const RoleCandidatePlan &other) const {
  return std::tie(wake_instance_ids, sleep_instance_ids, primary_scale_up) <
         std::tie(other.wake_instance_ids, other.sleep_instance_ids,
                  other.primary_scale_up);
}

bool RequestMigration::operator==(const RequestMigration &other) const {
  return request_id == other.request_id &&
         source_instance_id == other.source_instance_id &&
         destination_instance_id == other.destination_instance_id;
}

RoleEvaluationContext
prepare_role_evaluation_context(const RoleSnapshot &snapshot) {
  RoleEvaluationContext context;
  context.role = snapshot.role;
  context.strategy = snapshot.strategy;
  context.queue_scope = snapshot.queue_scope;
  context.max_concurrent_requests = snapshot.max_concurrent_requests;
  context.delta_throughput_threshold = snapshot.delta_throughput_threshold;
  context.itl_max_itl = snapshot.itl_max_itl;

  std::vector<const InstanceSnapshot *> instances;
  instances.reserve(snapshot.instances.size());
  for (const auto &instance : snapshot.instances) {
    instances.push_back(&instance);
  }
  std::sort(instances.begin(), instances.end(),
            [](const auto *lhs, const auto *rhs) {
              return lhs->instance_id < rhs->instance_id;
            });
  for (const auto *instance : instances) {
    if (!context.all_instance_ids.empty() &&
        context.all_instance_ids.back() == instance->instance_id) {
      throw std::invalid_argument("duplicate instance id in " + snapshot.role +
                                  ": " + std::to_string(instance->instance_id));
    }
    context.all_instance_ids.push_back(instance->instance_id);
  }

  const bool prepare_rollout_groups =
      snapshot.strategy == RoutingStrategy::kThroughputOptimal;
  context.instances.reserve(instances.size());
  for (const auto *instance : instances) {
    PreparedInstance prepared_instance;
    prepared_instance.snapshot.instance_id = instance->instance_id;
    prepared_instance.snapshot.is_awake = instance->is_awake;
    prepared_instance.snapshot.model_version = instance->model_version;
    prepared_instance.snapshot.route_request_count =
        instance->route_request_count;
    prepared_instance.snapshot.running_count = instance->running_count;
    prepared_instance.snapshot.waiting_count = instance->waiting_count;
    prepared_instance.snapshot.token_count = instance->token_count;
    prepared_instance.snapshot.max_model_len = instance->max_model_len;
    prepared_instance.snapshot.throughput_params = instance->throughput_params;
    prepared_instance.snapshot.route_cost_params = instance->route_cost_params;
    std::vector<PreparedRequest> prepared_requests;
    prepared_requests.reserve(instance->requests.size());
    for (const auto &request : instance->requests) {
      prepared_requests.push_back(prepare_request(
          request, context.all_instance_ids, prepare_rollout_groups));
    }
    for (const auto &request : prepared_requests) {
      if (snapshot.queue_scope != "running" || !request.snapshot.is_waiting) {
        ++prepared_instance.visible_request_count;
        prepared_instance.visible_token_count +=
            std::max<std::int64_t>(0, request.snapshot.seq_len);
        if (request.snapshot.movable) {
          prepared_instance.movable_requests.push_back(request);
        }
      }
      if (request.snapshot.movable) {
        prepared_instance.reroute_requests.push_back(request);
      }
    }
    std::sort(prepared_instance.movable_requests.begin(),
              prepared_instance.movable_requests.end(),
              [](const PreparedRequest &lhs, const PreparedRequest &rhs) {
                if (lhs.snapshot.seq_len != rhs.snapshot.seq_len) {
                  return lhs.snapshot.seq_len < rhs.snapshot.seq_len;
                }
                return compare_request_id(lhs, rhs) < 0;
              });
    std::sort(prepared_instance.reroute_requests.begin(),
              prepared_instance.reroute_requests.end(),
              [](const PreparedRequest &lhs, const PreparedRequest &rhs) {
                return compare_route_key(lhs, rhs) < 0;
              });
    const std::size_t index = context.instances.size();
    context.index_by_id.emplace(instance->instance_id, index);
    if (instance->is_awake) {
      context.before_awake.insert(instance->instance_id);
    }
    context.instances.push_back(std::move(prepared_instance));
  }

  context.pending_requests.reserve(snapshot.pending_requests.size());
  for (const auto &request : snapshot.pending_requests) {
    context.pending_requests.push_back(prepare_request(
        request, context.all_instance_ids, prepare_rollout_groups));
  }
  std::sort(context.pending_requests.begin(), context.pending_requests.end(),
            [](const PreparedRequest &lhs, const PreparedRequest &rhs) {
              return compare_route_key(lhs, rhs) < 0;
            });
  return context;
}

RoleEvaluationResult
evaluate_role_candidate(const RoleEvaluationContext &context,
                        const RoleCandidatePlan &plan) {
  const auto evaluation_started = Clock::now();
  std::set<int> active_ids = context.before_awake;
  active_ids.insert(plan.wake_instance_ids.begin(),
                    plan.wake_instance_ids.end());
  for (const int instance_id : plan.sleep_instance_ids) {
    active_ids.erase(instance_id);
  }
  for (const int instance_id : active_ids) {
    if (context.index_by_id.count(instance_id) == 0) {
      throw std::invalid_argument("candidate active instance is absent from " +
                                  context.role + ": " +
                                  std::to_string(instance_id));
    }
  }

  std::vector<InstanceState> states(context.instances.size());
  for (const int instance_id : active_ids) {
    const std::size_t index = context.index_by_id.at(instance_id);
    const PreparedInstance &prepared = context.instances[index];
    const bool was_awake = context.before_awake.count(instance_id) != 0;
    InstanceState state;
    state.prepared = &prepared;
    state.throughput_request_count = prepared.visible_request_count;
    state.throughput_token_count = prepared.visible_token_count;
    state.route_request_count =
        was_awake ? prepared.snapshot.route_request_count : 0;
    state.running_count = was_awake ? prepared.snapshot.running_count : 0;
    state.waiting_count = was_awake ? prepared.snapshot.waiting_count : 0;
    state.token_count = was_awake ? prepared.snapshot.token_count : 0;
    state.active = true;
    states[index] = state;
  }

  const auto rebalance_started = Clock::now();
  std::vector<RequestMigration> moves;
  if (plan.primary_scale_up) {
    std::set<int> donor_ids;
    std::set_intersection(context.before_awake.begin(),
                          context.before_awake.end(), active_ids.begin(),
                          active_ids.end(),
                          std::inserter(donor_ids, donor_ids.end()));
    moves = rebalance(context, states, donor_ids, active_ids.size());
  }
  const auto rebalance_finished = Clock::now();

  const auto router_started = Clock::now();
  std::vector<const std::vector<PreparedRequest> *> streams;
  std::int64_t total_to_route = 0;
  for (const int instance_id : plan.sleep_instance_ids) {
    const auto found = context.index_by_id.find(instance_id);
    if (found == context.index_by_id.end()) {
      throw std::invalid_argument("sleep instance is absent from " +
                                  context.role + ": " +
                                  std::to_string(instance_id));
    }
    const auto &requests = context.instances[found->second].reroute_requests;
    if (!requests.empty()) {
      streams.push_back(&requests);
      total_to_route += static_cast<std::int64_t>(requests.size());
    }
  }
  if (!context.pending_requests.empty()) {
    streams.push_back(&context.pending_requests);
    total_to_route +=
        static_cast<std::int64_t>(context.pending_requests.size());
  }

  RoleEvaluationResult result;
  result.rebalance_moves = std::move(moves);
  result.rebalance_executed = plan.primary_scale_up;
  result.rebalance_started_at = rebalance_started;
  result.rebalance_finished_at = rebalance_finished;
  if (result.rebalance_executed) {
    result.rebalance_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            rebalance_finished - rebalance_started);
  }
  if (!active_ids.empty()) {
    std::optional<RmSelectorHeap> rm_selector;
    if (context.strategy == RoutingStrategy::kItl) {
      rm_selector.emplace(context, states, active_ids);
    }
    std::priority_queue<RouteStreamEntry, std::vector<RouteStreamEntry>,
                        RouteStreamGreater>
        route_heap;
    for (std::size_t stream_order = 0; stream_order < streams.size();
         ++stream_order) {
      route_heap.push({streams[stream_order], stream_order, 0});
    }
    while (!route_heap.empty()) {
      RouteStreamEntry entry = route_heap.top();
      route_heap.pop();
      const PreparedRequest &request = (*entry.stream)[entry.request_index];
      if (dispatch_request(context, states, active_ids, request,
                           rm_selector.has_value() ? &*rm_selector : nullptr)) {
        ++result.routed_count;
      }
      ++entry.request_index;
      if (entry.request_index < entry.stream->size()) {
        route_heap.push(entry);
      }
    }
    for (const int instance_id : active_ids) {
      const double throughput =
          instance_throughput(state_for(context, states, instance_id));
      result.instance_throughputs.emplace_back(instance_id, throughput);
      result.throughput += throughput;
    }
  }
  result.unrouted_count = total_to_route - result.routed_count;
  const auto router_finished = Clock::now();
  result.router_started_at = router_started;
  result.router_finished_at = router_finished;
  result.router_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      router_finished - router_started);
  const auto evaluation_finished = Clock::now();
  const auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      evaluation_finished - evaluation_started);
  result.other_time = total_time - result.rebalance_time - result.router_time;
  if (result.other_time.count() < 0) {
    result.other_time = std::chrono::nanoseconds(0);
  }
  return result;
}

std::uint64_t
functional_checksum(const std::vector<RoleEvaluationResult> &results) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto &result : results) {
    std::uint64_t throughput_bits = 0;
    static_assert(sizeof(throughput_bits) == sizeof(result.throughput));
    std::memcpy(&throughput_bits, &result.throughput, sizeof(throughput_bits));
    hash = fnv_value(hash, throughput_bits);
    hash = fnv_value(hash, result.routed_count);
    hash = fnv_value(hash, result.unrouted_count);
    for (const auto &move : result.rebalance_moves) {
      hash = fnv_append(hash, move.request_id.data(), move.request_id.size());
      hash = fnv_value(hash, move.source_instance_id);
      hash = fnv_value(hash, move.destination_instance_id);
    }
    for (const auto &[instance_id, throughput] : result.instance_throughputs) {
      hash = fnv_value(hash, instance_id);
      std::memcpy(&throughput_bits, &throughput, sizeof(throughput_bits));
      hash = fnv_value(hash, throughput_bits);
    }
  }
  return hash;
}

} // namespace psrl::elastic_simulator
