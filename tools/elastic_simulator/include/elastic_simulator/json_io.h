#pragma once

#include <cstddef>
#include <string>

#include "elastic_simulator/simulator.h"

namespace psrl::elastic_simulator {

CycleInput parse_cycle_json(const std::string &line, std::size_t line_number);

} // namespace psrl::elastic_simulator
