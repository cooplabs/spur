# The upper-body executor (DESIGN.md §3.3; docs/architecture/EXECUTOR.md): r1exec (HOLD / TRAJECTORY / STREAM /
# FREEDRIVE per arm and head, deadman / TTL / guards / limits) and its unit tests. Included from rt/CMakeLists.txt.
add_library(r1exec STATIC ${CMAKE_CURRENT_LIST_DIR}/src/executor.cpp)
target_include_directories(r1exec PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(r1exec PUBLIC r1shm)
target_compile_options(r1exec PRIVATE ${R1RT_WARNINGS})

if(R1RT_BUILD_TESTS)
  add_executable(r1rt_executor_tests ${CMAKE_CURRENT_LIST_DIR}/tests/test_executor.cpp)
  target_link_libraries(r1rt_executor_tests PRIVATE r1exec GTest::gtest_main)
  target_compile_options(r1rt_executor_tests PRIVATE ${R1RT_WARNINGS})
  gtest_discover_tests(r1rt_executor_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
endif()
