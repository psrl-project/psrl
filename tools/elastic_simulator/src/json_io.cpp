#include "elastic_simulator/json_io.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace psrl::elastic_simulator {
namespace {

using rapidjson::Value;

[[noreturn]] void fail(std::size_t line_number, const std::string &path,
                       const std::string &message) {
  throw std::runtime_error("input line " + std::to_string(line_number) + " " +
                           path + ": " + message);
}

const Value *member(const Value &object, const char *name) {
  if (!object.IsObject()) {
    return nullptr;
  }
  const auto found = object.FindMember(name);
  return found == object.MemberEnd() ? nullptr : &found->value;
}

const Value &required_member(const Value &object, const char *name,
                             std::size_t line_number, const std::string &path) {
  const Value *value = member(object, name);
  if (value == nullptr) {
    fail(line_number, path + "." + name, "required field is missing");
  }
  return *value;
}

std::int64_t integer_value(const Value &value, std::size_t line_number,
                           const std::string &path) {
  if (value.IsInt64()) {
    return value.GetInt64();
  }
  if (value.IsUint64()) {
    if (value.GetUint64() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      fail(line_number, path, "integer exceeds int64 range");
    }
    return static_cast<std::int64_t>(value.GetUint64());
  }
  if (value.IsNumber()) {
    const double number = value.GetDouble();
    if (!std::isfinite(number) ||
        number <
            static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        number >
            static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      fail(line_number, path, "number cannot be represented as int64");
    }
    return static_cast<std::int64_t>(number);
  }
  fail(line_number, path, "expected an integer");
}

std::int64_t integer_member_or(const Value &object, const char *name,
                               std::int64_t fallback, std::size_t line_number,
                               const std::string &path) {
  const Value *value = member(object, name);
  if (value == nullptr || value->IsNull()) {
    return fallback;
  }
  return integer_value(*value, line_number, path + "." + name);
}

double number_value(const Value &value, std::size_t line_number,
                    const std::string &path) {
  if (!value.IsNumber()) {
    fail(line_number, path, "expected a finite number");
  }
  const double result = value.GetDouble();
  if (!std::isfinite(result)) {
    fail(line_number, path, "expected a finite number");
  }
  return result;
}

double number_member_or(const Value &object, const char *name, double fallback,
                        std::size_t line_number, const std::string &path) {
  const Value *value = member(object, name);
  if (value == nullptr || value->IsNull()) {
    return fallback;
  }
  return number_value(*value, line_number, path + "." + name);
}

bool bool_member_or(const Value &object, const char *name, bool fallback,
                    std::size_t line_number, const std::string &path) {
  const Value *value = member(object, name);
  if (value == nullptr || value->IsNull()) {
    return fallback;
  }
  if (!value->IsBool()) {
    fail(line_number, path + "." + name, "expected a boolean");
  }
  return value->GetBool();
}

std::string string_value(const Value &value, std::size_t line_number,
                         const std::string &path) {
  if (value.IsString()) {
    return std::string(value.GetString(), value.GetStringLength());
  }
  if (value.IsInt64()) {
    return std::to_string(value.GetInt64());
  }
  if (value.IsUint64()) {
    return std::to_string(value.GetUint64());
  }
  if (value.IsDouble()) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value.GetDouble();
    return stream.str();
  }
  if (value.IsBool()) {
    return value.GetBool() ? "True" : "False";
  }
  if (value.IsNull()) {
    return "None";
  }
  fail(line_number, path, "expected a scalar value");
}

std::string string_member_or(const Value &object, const char *name,
                             std::string fallback, std::size_t line_number,
                             const std::string &path) {
  const Value *value = member(object, name);
  if (value == nullptr) {
    return fallback;
  }
  return string_value(*value, line_number, path + "." + name);
}

PriorityValue parse_priority(const Value &value, std::size_t line_number,
                             const std::string &path) {
  if (value.IsNull()) {
    return PriorityValue::null();
  }
  if (value.IsBool()) {
    return PriorityValue::number_value(value.GetBool() ? 1.0 : 0.0);
  }
  if (value.IsNumber()) {
    return PriorityValue::number_value(number_value(value, line_number, path));
  }
  if (value.IsString()) {
    return PriorityValue::string_value(
        std::string(value.GetString(), value.GetStringLength()));
  }
  if (value.IsArray()) {
    std::vector<PriorityValue> items;
    items.reserve(value.Size());
    for (rapidjson::SizeType index = 0; index < value.Size(); ++index) {
      items.push_back(parse_priority(value[index], line_number,
                                     path + "[" + std::to_string(index) + "]"));
    }
    return PriorityValue::array_value(std::move(items));
  }
  if (value.IsObject()) {
    const Value *tag = member(value, "__psrl_float__");
    if (tag == nullptr || !tag->IsString() || value.MemberCount() != 1) {
      fail(line_number, path, "invalid tagged priority value");
    }
    const std::string special(tag->GetString(), tag->GetStringLength());
    if (special == "inf") {
      return PriorityValue::number_value(
          std::numeric_limits<double>::infinity());
    }
    if (special == "-inf") {
      return PriorityValue::number_value(
          -std::numeric_limits<double>::infinity());
    }
    fail(line_number, path, "unsupported tagged priority value: " + special);
  }
  fail(line_number, path,
       "priority values must be null, scalar, array, or a tagged infinity");
}

std::vector<int> parse_instance_ids(const Value &value, std::size_t line_number,
                                    const std::string &path) {
  if (!value.IsArray()) {
    fail(line_number, path, "expected an array of instance ids");
  }
  std::vector<int> result;
  result.reserve(value.Size());
  for (rapidjson::SizeType index = 0; index < value.Size(); ++index) {
    const std::int64_t parsed = integer_value(
        value[index], line_number, path + "[" + std::to_string(index) + "]");
    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      fail(line_number, path, "instance id exceeds int range");
    }
    result.push_back(static_cast<int>(parsed));
  }
  return result;
}

RequestSnapshot parse_request(const Value &raw, std::size_t line_number,
                              const std::string &path) {
  if (!raw.IsObject()) {
    fail(line_number, path, "expected a request object");
  }
  RequestSnapshot request;
  const Value *request_id = member(raw, "request_id");
  if (request_id == nullptr) {
    request_id = member(raw, "uid");
  }
  request.request_id =
      request_id == nullptr
          ? ""
          : string_value(*request_id, line_number, path + ".request_id");
  const Value *seq_len = member(raw, "seq_len");
  if (seq_len == nullptr) {
    seq_len = member(raw, "token_count");
  }
  request.seq_len =
      seq_len == nullptr || seq_len->IsNull()
          ? 0
          : std::max<std::int64_t>(
                0, integer_value(*seq_len, line_number, path + ".seq_len"));
  if (const Value *source = member(raw, "source_instance_id");
      source != nullptr && !source->IsNull()) {
    const std::int64_t parsed =
        integer_value(*source, line_number, path + ".source_instance_id");
    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      fail(line_number, path + ".source_instance_id",
           "instance id exceeds int range");
    }
    request.source_instance_id = static_cast<int>(parsed);
  }
  request.is_waiting =
      bool_member_or(raw, "is_waiting", false, line_number, path);
  request.movable = bool_member_or(raw, "movable", true, line_number, path);
  request.route_order =
      integer_member_or(raw, "route_order", 0, line_number, path);

  if (const Value *priority = member(raw, "routing_priority");
      priority != nullptr && !priority->IsNull()) {
    if (priority->IsArray()) {
      for (rapidjson::SizeType index = 0; index < priority->Size(); ++index) {
        request.routing_priority.push_back(parse_priority(
            (*priority)[index], line_number,
            path + ".routing_priority[" + std::to_string(index) + "]"));
      }
    } else {
      request.routing_priority.push_back(
          parse_priority(*priority, line_number, path + ".routing_priority"));
    }
  }
  if (const Value *eligible = member(raw, "eligible_instance_ids");
      eligible != nullptr && !eligible->IsNull()) {
    request.eligible_instance_ids = parse_instance_ids(
        *eligible, line_number, path + ".eligible_instance_ids");
  }
  if (const Value *fallback = member(raw, "fallback_instance_ids");
      fallback != nullptr && !fallback->IsNull()) {
    request.fallback_instance_ids = parse_instance_ids(
        *fallback, line_number, path + ".fallback_instance_ids");
  }
  if (const Value *priorities = member(raw, "candidate_priorities");
      priorities != nullptr && !priorities->IsNull()) {
    if (!priorities->IsArray()) {
      fail(line_number, path + ".candidate_priorities", "expected an array");
    }
    for (rapidjson::SizeType index = 0; index < priorities->Size(); ++index) {
      const Value &item = (*priorities)[index];
      if (!item.IsArray() || item.Size() < 2) {
        continue;
      }
      const std::int64_t instance_id = integer_value(
          item[0], line_number, path + ".candidate_priorities[].instance_id");
      if (instance_id < std::numeric_limits<int>::min() ||
          instance_id > std::numeric_limits<int>::max()) {
        fail(line_number, path + ".candidate_priorities",
             "instance id exceeds int range");
      }
      request.candidate_priorities.emplace_back(
          static_cast<int>(instance_id),
          parse_priority(item[1], line_number,
                         path + ".candidate_priorities[].priority"));
    }
  }
  return request;
}

template <std::size_t Size>
std::array<double, Size> parse_number_array(const Value &value,
                                            std::size_t line_number,
                                            const std::string &path) {
  if (!value.IsArray() || value.Size() != Size) {
    fail(line_number, path,
         "expected an array of length " + std::to_string(Size));
  }
  std::array<double, Size> result{};
  for (rapidjson::SizeType index = 0; index < value.Size(); ++index) {
    result[index] = number_value(value[index], line_number,
                                 path + "[" + std::to_string(index) + "]");
  }
  return result;
}

InstanceSnapshot parse_instance(const Value &raw, std::size_t line_number,
                                const std::string &path) {
  if (!raw.IsObject()) {
    fail(line_number, path, "expected an instance object");
  }
  InstanceSnapshot instance;
  const std::int64_t instance_id =
      integer_value(required_member(raw, "instance_id", line_number, path),
                    line_number, path + ".instance_id");
  if (instance_id < std::numeric_limits<int>::min() ||
      instance_id > std::numeric_limits<int>::max()) {
    fail(line_number, path + ".instance_id", "instance id exceeds int range");
  }
  instance.instance_id = static_cast<int>(instance_id);
  const Value *is_awake = member(raw, "is_awake");
  if (is_awake == nullptr) {
    is_awake = member(raw, "available");
  }
  if (is_awake != nullptr && !is_awake->IsNull()) {
    if (!is_awake->IsBool()) {
      fail(line_number, path + ".is_awake", "expected a boolean");
    }
    instance.is_awake = is_awake->GetBool();
  }
  const Value *model_version = member(raw, "candidate_model_version");
  if (model_version == nullptr) {
    model_version = member(raw, "model_version");
  }
  if (model_version != nullptr && !model_version->IsNull()) {
    instance.model_version =
        integer_value(*model_version, line_number, path + ".model_version");
  }
  if (const Value *requests = member(raw, "requests");
      requests != nullptr && !requests->IsNull()) {
    if (!requests->IsArray()) {
      fail(line_number, path + ".requests", "expected an array");
    }
    instance.requests.reserve(requests->Size());
    for (rapidjson::SizeType index = 0; index < requests->Size(); ++index) {
      instance.requests.push_back(
          parse_request((*requests)[index], line_number,
                        path + ".requests[" + std::to_string(index) + "]"));
    }
  }
  const Value *route_request_count = member(raw, "route_request_count");
  if (route_request_count == nullptr) {
    route_request_count = member(raw, "request_count");
  }
  instance.route_request_count =
      route_request_count == nullptr || route_request_count->IsNull()
          ? 0
          : std::max<std::int64_t>(
                0, integer_value(*route_request_count, line_number,
                                 path + ".route_request_count"));
  instance.running_count = std::max<std::int64_t>(
      0, integer_member_or(raw, "running_count", 0, line_number, path));
  instance.waiting_count = std::max<std::int64_t>(
      0, integer_member_or(raw, "waiting_count", 0, line_number, path));
  instance.token_count = std::max<std::int64_t>(
      0, integer_member_or(raw, "token_count", 0, line_number, path));
  instance.max_model_len = std::max<std::int64_t>(
      0, integer_member_or(raw, "max_model_len",
                           std::numeric_limits<std::int64_t>::max(),
                           line_number, path));
  if (const Value *params = member(raw, "throughput_params");
      params != nullptr && !params->IsNull()) {
    instance.throughput_params = parse_number_array<4>(
        *params, line_number, path + ".throughput_params");
  }
  if (const Value *route_cost = member(raw, "route_cost_params");
      route_cost != nullptr && !route_cost->IsNull()) {
    instance.route_cost_params = parse_number_array<5>(
        *route_cost, line_number, path + ".route_cost_params");
  }
  return instance;
}

RoleSnapshot parse_role(const Value &raw, const std::string &role_name,
                        std::size_t line_number, const std::string &path) {
  if (!raw.IsObject()) {
    fail(line_number, path, "expected a role object");
  }
  RoleSnapshot role;
  role.role = string_member_or(raw, "role", role_name, line_number, path);
  std::string strategy =
      string_member_or(raw, "strategy", "", line_number, path);
  std::transform(strategy.begin(), strategy.end(), strategy.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (strategy == "throughput_optimal") {
    role.strategy = RoutingStrategy::kThroughputOptimal;
  } else if (strategy == "itl") {
    role.strategy = RoutingStrategy::kItl;
  } else {
    fail(line_number, path + ".strategy", "unsupported strategy: " + strategy);
  }
  const Value &instances = required_member(raw, "instances", line_number, path);
  if (!instances.IsArray()) {
    fail(line_number, path + ".instances", "expected an array");
  }
  role.instances.reserve(instances.Size());
  for (rapidjson::SizeType index = 0; index < instances.Size(); ++index) {
    role.instances.push_back(
        parse_instance(instances[index], line_number,
                       path + ".instances[" + std::to_string(index) + "]"));
  }
  if (const Value *pending = member(raw, "pending_requests");
      pending != nullptr && !pending->IsNull()) {
    if (!pending->IsArray()) {
      fail(line_number, path + ".pending_requests", "expected an array");
    }
    role.pending_requests.reserve(pending->Size());
    for (rapidjson::SizeType index = 0; index < pending->Size(); ++index) {
      role.pending_requests.push_back(parse_request(
          (*pending)[index], line_number,
          path + ".pending_requests[" + std::to_string(index) + "]"));
    }
  }
  role.queue_scope =
      string_member_or(raw, "queue_scope", "running", line_number, path);
  if (const Value *maximum = member(raw, "max_concurrent_requests");
      maximum != nullptr && !maximum->IsNull()) {
    role.max_concurrent_requests =
        integer_value(*maximum, line_number, path + ".max_concurrent_requests");
  }
  role.delta_throughput_threshold = number_member_or(
      raw, "delta_throughput_threshold", 0.0, line_number, path);
  if (const Value *max_itl = member(raw, "itl_max_itl");
      max_itl != nullptr && !max_itl->IsNull()) {
    role.itl_max_itl =
        number_value(*max_itl, line_number, path + ".itl_max_itl");
  }
  return role;
}

RoleCandidatePlan parse_plan(const Value &raw, std::size_t line_number,
                             const std::string &path) {
  if (!raw.IsObject()) {
    fail(line_number, path, "expected a role plan object");
  }
  RoleCandidatePlan plan;
  if (const Value *wake = member(raw, "wake_instance_ids"); wake != nullptr) {
    const auto ids =
        parse_instance_ids(*wake, line_number, path + ".wake_instance_ids");
    plan.wake_instance_ids.insert(ids.begin(), ids.end());
  }
  if (const Value *sleep = member(raw, "sleep_instance_ids");
      sleep != nullptr) {
    const auto ids =
        parse_instance_ids(*sleep, line_number, path + ".sleep_instance_ids");
    plan.sleep_instance_ids.insert(ids.begin(), ids.end());
  }
  plan.primary_scale_up =
      bool_member_or(raw, "primary_scale_up", false, line_number, path);
  return plan;
}

} // namespace

CycleInput parse_cycle_json(const std::string &line, std::size_t line_number) {
  rapidjson::Document document;
  document.Parse(line.data(), line.size());
  if (document.HasParseError()) {
    fail(line_number, "json",
         std::string(rapidjson::GetParseError_En(document.GetParseError())) +
             " at byte " + std::to_string(document.GetErrorOffset()));
  }
  if (!document.IsObject()) {
    fail(line_number, "$", "expected a JSON object");
  }
  const std::int64_t schema_version = integer_value(
      required_member(document, "schema_version", line_number, "$"),
      line_number, "$.schema_version");
  if (schema_version != 1) {
    fail(line_number, "$.schema_version",
         "unsupported schema version " + std::to_string(schema_version));
  }
  const std::string record_type =
      string_value(required_member(document, "record_type", line_number, "$"),
                   line_number, "$.record_type");
  if (record_type != "elastic_simulation_input") {
    fail(line_number, "$.record_type",
         "unexpected record type: " + record_type);
  }

  CycleInput cycle;
  cycle.cycle_id =
      integer_value(required_member(document, "cycle_id", line_number, "$"),
                    line_number, "$.cycle_id");
  const Value *roles = member(document, "roles");
  if (roles == nullptr) {
    fail(line_number, "$.roles", "required field is missing");
  }
  if (!roles->IsObject()) {
    fail(line_number, "$.roles", "expected an object");
  }
  for (const char *role_name : {"Rollout", "RewardModel"}) {
    const Value *raw_role = member(*roles, role_name);
    if (raw_role == nullptr) {
      fail(line_number, "$.roles." + std::string(role_name),
           "required field is missing");
    }
    cycle.roles.emplace(role_name,
                        parse_role(*raw_role, role_name, line_number,
                                   "$.roles." + std::string(role_name)));
  }

  const Value *candidates = member(document, "candidates");
  if (candidates == nullptr) {
    fail(line_number, "$.candidates", "required field is missing");
  }
  if (!candidates->IsArray()) {
    fail(line_number, "$.candidates", "expected an array");
  }
  std::set<int> candidate_indices;
  cycle.candidates.reserve(candidates->Size());
  for (rapidjson::SizeType index = 0; index < candidates->Size(); ++index) {
    const Value &raw = (*candidates)[index];
    const std::string path = "$.candidates[" + std::to_string(index) + "]";
    if (!raw.IsObject()) {
      fail(line_number, path, "expected a candidate object");
    }
    CandidateInput candidate;
    const std::int64_t candidate_index =
        integer_value(required_member(raw, "index", line_number, path),
                      line_number, path + ".index");
    if (candidate_index < std::numeric_limits<int>::min() ||
        candidate_index > std::numeric_limits<int>::max()) {
      fail(line_number, path + ".index", "candidate index exceeds int range");
    }
    candidate.index = static_cast<int>(candidate_index);
    if (!candidate_indices.insert(candidate.index).second) {
      fail(line_number, path + ".index", "duplicate candidate index");
    }
    const Value &plans = required_member(raw, "role_plans", line_number, path);
    if (!plans.IsObject()) {
      fail(line_number, path + ".role_plans", "expected an object");
    }
    for (const char *role_name : {"Rollout", "RewardModel"}) {
      if (const Value *plan = member(plans, role_name); plan != nullptr) {
        candidate.role_plans.emplace(
            role_name,
            parse_plan(*plan, line_number,
                       path + ".role_plans." + std::string(role_name)));
      } else {
        candidate.role_plans.emplace(role_name, RoleCandidatePlan{});
      }
    }
    cycle.candidates.push_back(std::move(candidate));
  }
  std::sort(cycle.candidates.begin(), cycle.candidates.end(),
            [](const CandidateInput &lhs, const CandidateInput &rhs) {
              return lhs.index < rhs.index;
            });

  if (const Value *timing = member(document, "reference_timing");
      timing != nullptr && !timing->IsNull()) {
    if (!timing->IsObject()) {
      fail(line_number, "$.reference_timing", "expected an object or null");
    }
    for (auto field = timing->MemberBegin(); field != timing->MemberEnd();
         ++field) {
      const std::string name(field->name.GetString(),
                             field->name.GetStringLength());
      if (field->value.IsNumber()) {
        cycle.reference_timing_seconds.emplace(
            name, number_value(field->value, line_number,
                               "$.reference_timing." + name));
      } else if (name == "source_timestamp" && field->value.IsString()) {
        cycle.reference_timing_source_timestamp = std::string(
            field->value.GetString(), field->value.GetStringLength());
      }
    }
  }
  return cycle;
}

} // namespace psrl::elastic_simulator
