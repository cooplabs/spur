# The built-in arm model (DESIGN.md §3.4; docs/architecture/ARM_MODEL.md): r1arm = the evaluator (src/arm_model.cpp)
# + the table generated from the simulator's MJCF (src/arm_model_gen.cpp, tools/gen_arm_model.py), its unit tests,
# and its functions in the Python module r1rt_py (for the comparison with MuJoCo's mj_rne / mj_kinematics).
# Included at the end of rt/CMakeLists.txt (after bindings.cmake, so the r1rt_py target exists).
add_library(r1arm STATIC ${CMAKE_CURRENT_LIST_DIR}/src/arm_model.cpp ${CMAKE_CURRENT_LIST_DIR}/src/arm_model_gen.cpp)
target_include_directories(r1arm PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_options(r1arm PRIVATE ${R1RT_WARNINGS})

if(R1RT_BUILD_TESTS)
  add_executable(r1rt_arm_model_tests ${CMAKE_CURRENT_LIST_DIR}/tests/test_arm_model.cpp)
  target_link_libraries(r1rt_arm_model_tests PRIVATE r1arm GTest::gtest_main)
  target_compile_options(r1rt_arm_model_tests PRIVATE ${R1RT_WARNINGS})
  gtest_discover_tests(r1rt_arm_model_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
endif()

if(TARGET r1rt_py)
  target_sources(r1rt_py PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py_arm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/arm_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/arm_model_gen.cpp)
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py_arm.cpp
    PROPERTIES COMPILE_OPTIONS "${R1RT_WARNINGS};-Wno-shadow")
endif()
