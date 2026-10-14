# r1rt and r1guard as PROCESSES (session 5; DESIGN.md §2.1, §3.1, §4.3; docs/architecture/PROCESSES.md): the OS side
# (clock, absolute sleeps, the real-time request and its honest report), the config file, the DDS layer (rt/lowstate in,
# rt/lowcmd out, the built-in's rt/lowcmd from foreign writers) and the two main()s. Included from io.cmake's
# R1RT_WITH_CYCLONEDDS block (the processes need DDS); the executables land in <build>/apps/{r1rt,r1guard}.
find_package(Threads REQUIRED)

add_library(r1proc STATIC ${CMAKE_CURRENT_LIST_DIR}/src/rt_thread.cpp ${CMAKE_CURRENT_LIST_DIR}/src/proc_config.cpp)
target_link_libraries(r1proc PUBLIC r1core_api)
target_compile_options(r1proc PRIVATE ${R1RT_WARNINGS})

add_library(r1dds_io STATIC ${CMAKE_CURRENT_LIST_DIR}/src/dds_io.cpp)
target_link_libraries(r1dds_io PUBLIC r1hg_dds r1core_hg r1io r1core_api CycloneDDS::ddsc Threads::Threads)
# The generated DDS headers are not ours: -Wconversion / -Wshadow would fail on them; our own warnings otherwise.
target_compile_options(r1dds_io PRIVATE ${R1RT_WARNINGS} -Wno-shadow -Wno-conversion)

add_executable(r1rt_proc ${CMAKE_CURRENT_LIST_DIR}/apps/r1rt_main.cpp)
target_link_libraries(r1rt_proc PRIVATE r1proc r1dds_io r1core_api r1core_hg Threads::Threads)
target_compile_options(r1rt_proc PRIVATE ${R1RT_WARNINGS})
set_target_properties(r1rt_proc PROPERTIES OUTPUT_NAME r1rt RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

add_executable(r1guard_proc ${CMAKE_CURRENT_LIST_DIR}/apps/r1guard_main.cpp)
target_link_libraries(r1guard_proc PRIVATE r1guard r1proc r1dds_io Threads::Threads)
target_compile_options(r1guard_proc PRIVATE ${R1RT_WARNINGS})
set_target_properties(r1guard_proc PROPERTIES OUTPUT_NAME r1guard RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

if(R1RT_BUILD_TESTS)
  add_executable(r1rt_proc_tests
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_proc_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_proc_timing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_proc_dds.cpp)
  target_link_libraries(r1rt_proc_tests PRIVATE r1proc r1dds_io GTest::gtest_main Threads::Threads)
  target_compile_options(r1rt_proc_tests PRIVATE ${R1RT_WARNINGS} -Wno-shadow -Wno-conversion)
  gtest_discover_tests(r1rt_proc_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
endif()
