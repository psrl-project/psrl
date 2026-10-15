#include "elastic_simulator/simulator.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using psrl::elastic_simulator::evaluate_role_candidate;
using psrl::elastic_simulator::InstanceSnapshot;
using psrl::elastic_simulator::prepare_role_evaluation_context;
using psrl::elastic_simulator::RequestSnapshot;
using psrl::elastic_simulator::RoleCandidatePlan;
using psrl::elastic_simulator::RoleSnapshot;
using psrl::elastic_simulator::RoutingStrategy;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

InstanceSnapshot make_instance(
    int instance_id,
    const std::vector<std::tuple<std::string, std::int64_t, bool>> &requests,
    bool awake = true) {
  InstanceSnapshot instance;
  instance.instance_id = instance_id;
  instance.is_awake = awake;
  instance.max_model_len = 10'000;
  instance.throughput_params = {0.1, 1.0, 0.0, 0.0};
  instance.route_cost_params = std::array<double, 5>{0.0, 0.0, 0.0, 1.0, 0.01};
  for (const auto &[request_id, seq_len, is_waiting] : requests) {
    RequestSnapshot request;
    request.request_id = request_id;
    request.seq_len = seq_len;
    request.source_instance_id = instance_id;
    request.is_waiting = is_waiting;
    instance.requests.push_back(std::move(request));
    ++instance.route_request_count;
    instance.running_count += is_waiting ? 0 : 1;
    instance.waiting_count += is_waiting ? 1 : 0;
    instance.token_count += seq_len;
  }
  return instance;
}

void test_rebalance_moves_shortest_requests_from_highest_throughput() {
  RoleSnapshot snapshot;
  snapshot.role = "Rollout";
  snapshot.strategy = RoutingStrategy::kThroughputOptimal;
  snapshot.max_concurrent_requests = 32;
  snapshot.instances = {
      make_instance(0, {{"10", 10, false}, {"2", 2, false}, {"1", 1, false}}),
      make_instance(1, {}, false),
  };
  RoleCandidatePlan plan;
  plan.wake_instance_ids = {1};
  plan.primary_scale_up = true;

  const auto result =
      evaluate_role_candidate(prepare_role_evaluation_context(snapshot), plan);
  check(result.rebalance_moves.size() == 2, "expected two rebalance moves");
  check(result.rebalance_moves[0].request_id == "1",
        "first move must use shortest request");
  check(result.rebalance_moves[1].request_id == "2",
        "second move must use next request");
  for (const auto &move : result.rebalance_moves) {
    check(move.source_instance_id == 0, "unexpected rebalance donor");
    check(move.destination_instance_id == 1,
          "unexpected rebalance destination");
  }
}

void test_rebalance_stops_on_first_non_increasing_move() {
  RoleSnapshot snapshot;
  snapshot.role = "RewardModel";
  snapshot.strategy = RoutingStrategy::kItl;
  InstanceSnapshot donor =
      make_instance(0, {{"0", 1, false}, {"1", 100, false}});
  InstanceSnapshot destination = make_instance(1, {}, false);
  donor.throughput_params = {1.0, 0.0, 0.0, 0.0};
  destination.throughput_params = {1000.0, 0.0, 0.0, 0.0};
  snapshot.instances = {donor, destination};
  RoleCandidatePlan plan;
  plan.wake_instance_ids = {1};
  plan.primary_scale_up = true;

  const auto result =
      evaluate_role_candidate(prepare_role_evaluation_context(snapshot), plan);
  check(result.rebalance_moves.empty(),
        "rebalance must stop when the first proposed move has no strict gain");
}

void test_rebalance_ties_use_smallest_ids() {
  RoleSnapshot snapshot;
  snapshot.role = "RewardModel";
  snapshot.strategy = RoutingStrategy::kItl;
  snapshot.instances = {
      make_instance(0, {{"9", 5, false}, {"3", 5, false}}),
      make_instance(1, {{"8", 5, false}, {"4", 5, false}}),
      make_instance(2, {}, false),
      make_instance(3, {}, false),
  };
  RoleCandidatePlan plan;
  plan.wake_instance_ids = {2, 3};
  plan.primary_scale_up = true;

  const auto result =
      evaluate_role_candidate(prepare_role_evaluation_context(snapshot), plan);
  check(!result.rebalance_moves.empty(), "expected at least one tie-case move");
  const auto &first = result.rebalance_moves.front();
  check(first.source_instance_id == 0, "donor tie did not use smallest id");
  check(first.destination_instance_id == 2,
        "destination tie did not use smallest id");
  check(first.request_id == "3", "request tie did not use numeric id order");
}

void test_rm_heap_preserves_load_and_instance_ties() {
  RoleSnapshot snapshot;
  snapshot.role = "RewardModel";
  snapshot.strategy = RoutingStrategy::kItl;
  InstanceSnapshot first = make_instance(0, {});
  InstanceSnapshot second = make_instance(1, {});
  first.throughput_params = {0.0, 1.0, 0.0, 0.0};
  second.throughput_params = {0.0, 1.0, 0.0, 0.0};
  snapshot.instances = {first, second};
  snapshot.max_concurrent_requests = 8;
  for (int index = 0; index < 5; ++index) {
    RequestSnapshot request;
    request.request_id = std::to_string(index);
    request.seq_len = 1;
    request.route_order = index;
    snapshot.pending_requests.push_back(std::move(request));
  }

  const auto result = evaluate_role_candidate(
      prepare_role_evaluation_context(snapshot), RoleCandidatePlan{});
  check(result.routed_count == 5, "RM should route all pending requests");
  check(result.instance_throughputs.size() == 2, "expected two RM throughputs");
  check(result.instance_throughputs[0].first == 0, "instance ordering changed");
  check(std::abs(result.instance_throughputs[0].second - 3.0) < 1e-12,
        "RM tie broke incorrectly");
  check(std::abs(result.instance_throughputs[1].second - 2.0) < 1e-12,
        "RM load update changed");
}

void test_scale_down_reroutes_waiting_request() {
  RoleSnapshot snapshot;
  snapshot.role = "RewardModel";
  snapshot.strategy = RoutingStrategy::kItl;
  snapshot.queue_scope = "running";
  snapshot.instances = {
      make_instance(0, {{"waiting", 3, true}}),
      make_instance(1, {}),
  };
  snapshot.max_concurrent_requests = 4;
  RoleCandidatePlan plan;
  plan.sleep_instance_ids = {0};

  const auto result =
      evaluate_role_candidate(prepare_role_evaluation_context(snapshot), plan);
  check(result.routed_count == 1,
        "waiting request on sleeping instance was not rerouted");
  check(result.unrouted_count == 0, "waiting request unexpectedly unrouted");
}

void test_empty_active_role_is_valid() {
  RoleSnapshot snapshot;
  snapshot.role = "RewardModel";
  snapshot.strategy = RoutingStrategy::kItl;
  snapshot.instances = {
      make_instance(0, {}, false),
      make_instance(1, {}, false),
  };
  RequestSnapshot first;
  first.request_id = "p0";
  first.seq_len = 1;
  RequestSnapshot second;
  second.request_id = "p1";
  second.seq_len = 2;
  second.route_order = 1;
  snapshot.pending_requests = {first, second};

  const auto result = evaluate_role_candidate(
      prepare_role_evaluation_context(snapshot), RoleCandidatePlan{});
  check(result.throughput == 0.0, "empty role throughput must be zero");
  check(result.routed_count == 0, "empty role routed requests");
  check(result.unrouted_count == 2, "empty role lost pending requests");
}

} // namespace

int main() {
  try {
    test_rebalance_moves_shortest_requests_from_highest_throughput();
    test_rebalance_stops_on_first_non_increasing_move();
    test_rebalance_ties_use_smallest_ids();
    test_rm_heap_preserves_load_and_instance_ties();
    test_scale_down_reroutes_waiting_request();
    test_empty_active_role_is_valid();
    std::cout << "elastic simulator core tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
