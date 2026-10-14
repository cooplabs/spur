# r1shm on POSIX shared memory + the hg message types and Unitree's CRC32 (session 2; DESIGN §10.2,
# docs/architecture/RT_CORE_INTERFACES.md). Included at the end of rt/CMakeLists.txt.
add_library(r1io STATIC src/shm.cpp src/crc32.cpp)
target_link_libraries(r1io PUBLIC r1core_api)  # shm.cpp initialises the segment with r1rt_segment_init
target_compile_options(r1io PRIVATE ${R1RT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(r1io PUBLIC rt)  # shm_open on older glibc
endif()

if(R1RT_BUILD_TESTS)
  # Second process for the two-process shared-memory tests (spawned with posix_spawn).
  add_executable(r1rt_shm_helper tests/test_shm_helper.cpp)
  target_link_libraries(r1rt_shm_helper PRIVATE r1io)
  target_compile_options(r1rt_shm_helper PRIVATE ${R1RT_WARNINGS})

  add_executable(r1rt_io_tests tests/test_shm.cpp tests/test_crc.cpp tests/test_hg_msgs.cpp)
  target_link_libraries(r1rt_io_tests PRIVATE r1io GTest::gtest_main)
  target_compile_options(r1rt_io_tests PRIVATE ${R1RT_WARNINGS})
  target_compile_definitions(r1rt_io_tests PRIVATE
    R1RT_SHM_HELPER="$<TARGET_FILE:r1rt_shm_helper>"
    R1RT_CRC_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/../tests/assets/crc_golden/crc_golden.txt")
  add_dependencies(r1rt_io_tests r1rt_shm_helper)
  gtest_discover_tests(r1rt_io_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
endif()

# Optional: the hg DDS types from OUR CycloneDDS + cyclonedds-cxx 0.10.2 build (rt/idl/build_cyclonedds.sh prints the
# prefix to pass as CMAKE_PREFIX_PATH). OFF by default: the default build and scripts/ci.sh need no DDS.
option(R1RT_WITH_CYCLONEDDS "Generate + build the hg DDS types (needs CycloneDDS-CXX 0.10.2)" OFF)
if(R1RT_WITH_CYCLONEDDS)
  find_package(CycloneDDS-CXX REQUIRED)  # brings CycloneDDS (idlc, ddsc) too
  find_package(CycloneDDS REQUIRED)
  if(NOT CycloneDDS_VERSION VERSION_EQUAL 0.10.2)
    message(WARNING "CycloneDDS ${CycloneDDS_VERSION}: Unitree's hg types were generated with 0.10.2 "
                    "(HgDds.GeneratedTypesEqualUnitrees will tell whether this version generates the same types)")
  endif()
  set(_hg_rel unitree/idl/hg)
  set(_hg_gen ${CMAKE_CURRENT_BINARY_DIR}/hg_idl_gen)
  set(_hg_src ${CMAKE_CURRENT_SOURCE_DIR}/idl/src)
  file(MAKE_DIRECTORY ${_hg_gen}/${_hg_rel})
  file(GLOB _hg_idls ${_hg_src}/${_hg_rel}/*.idl)
  set(_hg_cpp)
  foreach(_idl ${_hg_idls})
    get_filename_component(_n ${_idl} NAME_WE)
    # idlc writes into its working directory; generating inside unitree/idl/hg keeps the generated
    # `#include "unitree/idl/hg/X.hpp"` lines resolvable from ${_hg_gen}. Extensibility final = Unitree's.
    add_custom_command(
      OUTPUT ${_hg_gen}/${_hg_rel}/${_n}.hpp ${_hg_gen}/${_hg_rel}/${_n}.cpp
      COMMAND CycloneDDS::idlc -x final -Wno-implicit-extensibility -l $<TARGET_FILE:CycloneDDS-CXX::idlcxx>
              -I ${_hg_src} ${_idl}
      WORKING_DIRECTORY ${_hg_gen}/${_hg_rel}
      DEPENDS ${_hg_idls} CycloneDDS-CXX::idlcxx
      VERBATIM)
    list(APPEND _hg_cpp ${_hg_gen}/${_hg_rel}/${_n}.cpp)
  endforeach()
  add_library(r1hg_dds STATIC ${_hg_cpp})
  target_include_directories(r1hg_dds PUBLIC ${_hg_gen})
  target_link_libraries(r1hg_dds PUBLIC CycloneDDS-CXX::ddscxx)
  if(R1RT_BUILD_TESTS)
    add_executable(r1rt_dds_tests tests/test_hg_dds.cpp)
    target_link_libraries(r1rt_dds_tests PRIVATE r1hg_dds r1io GTest::gtest_main)
    target_compile_definitions(r1rt_dds_tests PRIVATE
      R1RT_HG_GEN_DIR="${_hg_gen}/${_hg_rel}"
      R1RT_UNITREE_HG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/idl/unitree_hg")
    gtest_discover_tests(r1rt_dds_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
  endif()
  include(${CMAKE_CURRENT_LIST_DIR}/proc.cmake)  # r1rt / r1guard as processes (session 5)
endif()
