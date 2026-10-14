# The balance policy runner (DESIGN.md §3.2): r1policy (fixed-size Eigen MLP), the weights converted from the
# published ONNX at build time, the r1rt_policy_eval CLI and the golden tests. Included from rt/CMakeLists.txt.
set(R1RT_REPO_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Eigen.cmake)

add_library(r1policy STATIC ${CMAKE_CURRENT_LIST_DIR}/src/policy_mlp.cpp)
target_include_directories(r1policy PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(r1policy PUBLIC r1rt_eigen)
# Eigen asserts if anything inside it tries to allocate while Eigen::internal::set_is_malloc_allowed(false) - the
# no-allocation test of run() uses that (a flag check in Eigen's allocator; no cost in the products themselves).
target_compile_definitions(r1policy PUBLIC EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(r1policy PRIVATE ${R1RT_WARNINGS})
# Optimised in Debug too (keeps -g and the sanitizers): unoptimised Eigen products are ~18x slower (M4, UBSan:
# 1.8 ms per inference at -O0, ~0.1 ms at -O2; Release ~0.04 ms), which would make the Debug CI golden runs crawl.
target_compile_options(r1policy PRIVATE $<$<CONFIG:Debug>:-O2>)

# Validation CLI: weights + an R1G1 observation file -> actions (+ max error against the file's expected actions).
add_executable(r1rt_policy_eval ${CMAKE_CURRENT_LIST_DIR}/src/policy_eval.cpp)
target_link_libraries(r1rt_policy_eval PRIVATE r1policy)
target_compile_options(r1rt_policy_eval PRIVATE ${R1RT_WARNINGS})

# The weights file is generated from the published ONNX by tools/policy_convert.py (needs the `onnx` package: the
# project venv's dev group; scripts/ci.sh runs CMake under `uv run`, so FindPython picks the venv). Without it the
# golden test is skipped with a message (a C++-only build still compiles everything).
set(R1RT_POLICY_ONNX ${R1RT_REPO_DIR}/policies/r1mc2_manip2_lean_20261006T0421Z/policy.onnx)
set(R1RT_POLICY_WEIGHTS ${CMAKE_BINARY_DIR}/policy/policy_weights.r1w)
set(R1RT_POLICY_GOLDEN ${R1RT_REPO_DIR}/tests/assets/policy_golden/golden_1000.bin)
find_package(Python3 COMPONENTS Interpreter)
set(R1RT_HAVE_CONVERTER OFF)
if(Python3_Interpreter_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import onnx" RESULT_VARIABLE _onnx_rc OUTPUT_QUIET ERROR_QUIET)
  if(_onnx_rc EQUAL 0)
    set(R1RT_HAVE_CONVERTER ON)
  endif()
endif()
if(R1RT_HAVE_CONVERTER)
  add_custom_command(
    OUTPUT ${R1RT_POLICY_WEIGHTS}
    COMMAND ${Python3_EXECUTABLE} ${R1RT_REPO_DIR}/tools/policy_convert.py ${R1RT_POLICY_ONNX} ${R1RT_POLICY_WEIGHTS}
    DEPENDS ${R1RT_POLICY_ONNX} ${R1RT_REPO_DIR}/tools/policy_convert.py
    COMMENT "Converting the balance policy ONNX -> ${R1RT_POLICY_WEIGHTS}"
    VERBATIM)
  add_custom_target(r1rt_policy_weights ALL DEPENDS ${R1RT_POLICY_WEIGHTS})
  message(STATUS "r1rt policy: weights converted at build time with ${Python3_EXECUTABLE}")
else()
  message(WARNING "r1rt policy: no Python with the `onnx` package found - the policy weights are not converted and "
                  "the golden test will be SKIPPED. Run CMake under `uv run` (scripts/ci.sh does).")
endif()

if(R1RT_BUILD_TESTS)
  add_executable(r1rt_policy_tests
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_policy_mlp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_policy_golden.cpp)
  target_link_libraries(r1rt_policy_tests PRIVATE r1policy GTest::gtest_main)
  target_compile_options(r1rt_policy_tests PRIVATE ${R1RT_WARNINGS})
  target_compile_definitions(r1rt_policy_tests PRIVATE
    R1RT_POLICY_WEIGHTS="${R1RT_POLICY_WEIGHTS}"
    R1RT_POLICY_GOLDEN="${R1RT_POLICY_GOLDEN}")
  if(R1RT_HAVE_CONVERTER)
    add_dependencies(r1rt_policy_tests r1rt_policy_weights)
  endif()
  gtest_discover_tests(r1rt_policy_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 300)
endif()
