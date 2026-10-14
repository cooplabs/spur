# The real-time core (session 3; DESIGN §3.1-§3.2, §4.2-§4.3): r1rt::Core + r1rt::PolicyRunner behind the step C ABI
# (r1core_api, defined in CMakeLists.txt from src/step_api.cpp), the hg LowCmd_ image + CRC helper (r1core_hg), the
# C++ tests and the policy-runner golden check. Included at the end of rt/CMakeLists.txt.
target_sources(r1core_api PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/core.cpp ${CMAKE_CURRENT_LIST_DIR}/src/policy_runner.cpp)
target_link_libraries(r1core_api PUBLIC r1policy r1exec r1arm)  # the upper-body executor (executor.cmake, included first)
# The runner mirrors the Python reference statement by statement: no fused multiply-add contraction (NumPy does none),
# so its arithmetic matches the reference to the network's rounding (golden <= 1e-12).
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/policy_runner.cpp ${CMAKE_CURRENT_LIST_DIR}/src/core.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

add_library(r1core_hg STATIC ${CMAKE_CURRENT_LIST_DIR}/src/core_hg.cpp)
target_link_libraries(r1core_hg PUBLIC r1io)
target_compile_options(r1core_hg PRIVATE ${R1RT_WARNINGS})

set(R1RT_POLICY_RUNNER_GOLDEN ${CMAKE_CURRENT_LIST_DIR}/../tests/assets/policy_runner/sequence_1600.r1pr)
if(R1RT_BUILD_TESTS)
  add_executable(r1rt_core_tests
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core_takeover.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core_threads.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core_mailbox.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core_onstand.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_core_stale.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_policy_runner.cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(r1rt_core_tests PRIVATE r1core_api r1core_hg GTest::gtest_main Threads::Threads)
  target_compile_options(r1rt_core_tests PRIVATE ${R1RT_WARNINGS})
  target_compile_definitions(r1rt_core_tests PRIVATE
    R1RT_POLICY_WEIGHTS="${R1RT_POLICY_WEIGHTS}"
    R1RT_POLICY_RUNNER_GOLDEN="${R1RT_POLICY_RUNNER_GOLDEN}")
  if(R1RT_HAVE_CONVERTER)
    add_dependencies(r1rt_core_tests r1rt_policy_weights)
  endif()
  gtest_discover_tests(r1rt_core_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 300)
endif()
