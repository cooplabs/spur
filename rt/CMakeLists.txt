# r1rt: the C++ real-time core of r1mc3 (DESIGN §2-§4). Session 1: interfaces (shared-memory layout, seqlock,
# SPSC ring, step C ABI) + unit tests. Build from the repo root:
#   uv run cmake -S rt -B build/rt -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON
#   uv run cmake --build build/rt && uv run ctest --test-dir build/rt --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(r1rt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(R1RT_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(R1RT_TSAN "Build with ThreadSanitizer (concurrency tests; not combinable with R1RT_SANITIZE)" OFF)
option(R1RT_BUILD_TESTS "Build the GoogleTest unit tests" ON)

if(R1RT_SANITIZE AND R1RT_TSAN)
  message(FATAL_ERROR "R1RT_SANITIZE (ASan/UBSan) and R1RT_TSAN cannot be combined")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SanitizerProbe.cmake)
set(R1RT_ACTIVE_SANITIZERS "none")
if(R1RT_SANITIZE)
  r1rt_probe_sanitizer("-fsanitize=address" R1RT_ASAN_RUNS)
  if(R1RT_ASAN_RUNS)
    set(_san "address,undefined")
  else()
    message(WARNING "R1RT_SANITIZE: AddressSanitizer binaries do not run on this host (they hang at start-up on "
                    "macOS 26.5 + Apple clang 17): building with UndefinedBehaviorSanitizer ONLY. ASan runs in the "
                    "Linux CI job.")
    set(_san "undefined")
  endif()
  add_compile_options(-fsanitize=${_san} -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=${_san})
  set(R1RT_ACTIVE_SANITIZERS "${_san}")
endif()
if(R1RT_TSAN)
  r1rt_probe_sanitizer("-fsanitize=thread" R1RT_TSAN_RUNS)
  if(NOT R1RT_TSAN_RUNS)
    message(FATAL_ERROR "R1RT_TSAN: ThreadSanitizer binaries do not run on this host (they crash at start-up on "
                        "macOS 26.5 + Apple clang 17). Run the TSan build on Linux.")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
  # GCC >= 11 warns (-Wtsan) that ThreadSanitizer does not instrument std::atomic_thread_fence - the seqlock's fences
  # (rt/include/r1rt/seqlock.hpp). Under our -Werror that stopped every TSan build with GCC 13 (Ubuntu 24.04, found by
  # the public CI, 14 Oct 2026); the warning stays visible, not fatal. TSan still checks every atomic and plain access.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Wtsan R1RT_CXX_HAS_WTSAN)
  if(R1RT_CXX_HAS_WTSAN)
    add_compile_options(-Wno-error=tsan)
  endif()
  set(R1RT_ACTIVE_SANITIZERS "thread")
endif()
message(STATUS "r1rt sanitizers: ${R1RT_ACTIVE_SANITIZERS}")

# Our targets only (not googletest): strict warnings as errors.
set(R1RT_WARNINGS -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wno-sign-conversion)

# Header-only shared-memory primitives and the layout.
add_library(r1shm INTERFACE)
target_include_directories(r1shm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(r1shm INTERFACE cxx_std_17)

# The C ABI of the core (session 1: segment functions + a stub core that never publishes).
add_library(r1core_api STATIC src/step_api.cpp)
target_link_libraries(r1core_api PUBLIC r1shm)
target_compile_options(r1core_api PRIVATE ${R1RT_WARNINGS})

if(R1RT_BUILD_TESTS)
  include(FetchContent)
  # Pinned release tarball + hash (checked by CMake). googletest is BSD-3-Clause.
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(r1rt_tests
    tests/test_seqlock.cpp
    tests/test_spsc_ring.cpp
    tests/test_shm_layout.cpp
    tests/test_step_api.cpp
    tests/test_resume_clock.cpp
    tests/c_abi_check.c)
  target_link_libraries(r1rt_tests PRIVATE r1core_api GTest::gtest_main)
  target_compile_options(r1rt_tests PRIVATE ${R1RT_WARNINGS})
  gtest_discover_tests(r1rt_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/policy.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/io.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/bindings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/executor.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/core.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/arm_model.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/guard.cmake)

# The policy-timing check (session 8) for the Orin (rt/apps/r1rt_policy_bench.cpp, docs/hardware/T7_BENCH.md):
# the PolicyRunner called as r1rt's policy thread calls it, on a recorded observation file; no DDS, built in every tree,
# <build>/apps/r1rt_policy_bench next to r1rt / r1guard (the release's build/rt on the backpack).
find_package(Threads REQUIRED)
add_executable(r1rt_policy_bench apps/r1rt_policy_bench.cpp src/rt_thread.cpp)
target_link_libraries(r1rt_policy_bench PRIVATE r1core_api Threads::Threads)
target_compile_options(r1rt_policy_bench PRIVATE ${R1RT_WARNINGS})
set_target_properties(r1rt_policy_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)
