cmake_minimum_required(VERSION 3.16)

project(psrl_elastic_simulator LANGUAGES CXX)

include(CTest)
find_package(Threads REQUIRED)
find_path(RAPIDJSON_INCLUDE_DIR rapidjson/document.h)

if(NOT RAPIDJSON_INCLUDE_DIR)
  message(FATAL_ERROR "RapidJSON headers were not found (expected rapidjson/document.h)")
endif()

add_library(elastic_simulator_core
  src/simulator.cpp
)
target_include_directories(elastic_simulator_core PUBLIC include)
target_compile_features(elastic_simulator_core PUBLIC cxx_std_17)
target_compile_options(elastic_simulator_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(elastic_simulator
  src/json_io.cpp
  src/main.cpp
)
target_include_directories(elastic_simulator PRIVATE ${RAPIDJSON_INCLUDE_DIR})
target_link_libraries(elastic_simulator PRIVATE elastic_simulator_core Threads::Threads)
target_compile_features(elastic_simulator PRIVATE cxx_std_17)
target_compile_options(elastic_simulator PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(elastic_simulator_tests tests/simulator_test.cpp)
  target_link_libraries(elastic_simulator_tests PRIVATE elastic_simulator_core)
  target_compile_features(elastic_simulator_tests PRIVATE cxx_std_17)
  target_compile_options(elastic_simulator_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME elastic_simulator_tests COMMAND elastic_simulator_tests)
endif()
