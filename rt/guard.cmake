# r1guard (session 4; DESIGN.md §2.1, §4.3 rung 2; docs/architecture/GUARD.md): r1rt::Guard - the core's hot standby
# on the same shared segment (own lowstate, heartbeat check, warm shadow policy, token CAS, by-mode takeover, never
# damp) - its unit tests, and its Python binding compiled into r1rt_py (rt/bindings/r1rt_py_guard.cpp). Included at the
# end of rt/CMakeLists.txt (after bindings.cmake and core.cmake: r1core_api and r1rt_py exist here).
add_library(r1guard STATIC ${CMAKE_CURRENT_LIST_DIR}/src/guard.cpp ${CMAKE_CURRENT_LIST_DIR}/src/guard_forensics.cpp)
target_link_libraries(r1guard PUBLIC r1core_api)  # PolicyRunner (compiled into r1core_api) + r1policy + r1shm
target_compile_options(r1guard PRIVATE ${R1RT_WARNINGS})

if(TARGET r1rt_py)
  target_sources(r1rt_py PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py_guard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/guard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/guard_forensics.cpp)
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py_guard.cpp
    PROPERTIES COMPILE_OPTIONS "${R1RT_WARNINGS};-Wno-shadow")
endif()

if(R1RT_BUILD_TESTS)
  add_executable(r1rt_guard_tests ${CMAKE_CURRENT_LIST_DIR}/tests/test_guard.cpp)
  target_link_libraries(r1rt_guard_tests PRIVATE r1guard GTest::gtest_main)
  target_compile_options(r1rt_guard_tests PRIVATE ${R1RT_WARNINGS})
  target_compile_definitions(r1rt_guard_tests PRIVATE
    R1RT_POLICY_WEIGHTS="${R1RT_POLICY_WEIGHTS}"
    R1RT_POLICY_RUNNER_GOLDEN="${R1RT_POLICY_RUNNER_GOLDEN}")
  if(R1RT_HAVE_CONVERTER)
    add_dependencies(r1rt_guard_tests r1rt_policy_weights)
  endif()
  gtest_discover_tests(r1rt_guard_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 300)
endif()

# session 9 (U19 / D25): the guard's stale rung (docs/sim/STATE_AGE.md)
if(R1RT_BUILD_TESTS)
  add_executable(r1rt_guard_stale_tests ${CMAKE_CURRENT_LIST_DIR}/tests/test_guard_stale.cpp)
  target_link_libraries(r1rt_guard_stale_tests PRIVATE r1guard GTest::gtest_main)
  target_compile_options(r1rt_guard_stale_tests PRIVATE ${R1RT_WARNINGS})
  target_compile_definitions(r1rt_guard_stale_tests PRIVATE
    R1RT_POLICY_WEIGHTS="${R1RT_POLICY_WEIGHTS}"
    R1RT_POLICY_RUNNER_GOLDEN="${R1RT_POLICY_RUNNER_GOLDEN}")
  if(R1RT_HAVE_CONVERTER)
    add_dependencies(r1rt_guard_stale_tests r1rt_policy_weights)
  endif()
  gtest_discover_tests(r1rt_guard_stale_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 300)
endif()
