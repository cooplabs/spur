# Python binding of the C++ policy runner: module `r1rt_py` (nanobind) exposing r1rt::PolicyMlp, so the simulator can
# run the C++ network in lockstep (sim/r1sim/cpp_policy.py; RT_CORE_INTERFACES.md "Python binding"). Simulation and
# validation only - the robot's RT path never loads Python. Included from rt/CMakeLists.txt.
#
# Needs a Python with nanobind (the project venv's dev group: scripts/ci.sh runs CMake under `uv run`); without it
# the module is skipped with a message and everything else builds as before.
#
# Sanitizers: the module keeps the build's UndefinedBehaviorSanitizer (s2, 10 Oct: the UBSan-instrumented module
# loads into the venv's uninstrumented interpreter and runs - same results, ~5x slower per call, ~0.2 ms), but never
# AddressSanitizer / ThreadSanitizer: their runtimes must come first in the process (an instrumented extension
# imported into a plain `python` aborts unless the runtime is preloaded), which would break the Python tests in the
# Linux ASan job. R1RT_BINDINGS_SANITIZE=OFF strips the sanitizers from the module completely.
option(R1RT_PYTHON_BINDINGS "Build the r1rt_py Python module (needs nanobind in the Python CMake finds)" ON)
# Default ON only on macOS: on Linux (glibc) libubsan uses initial-exec TLS, and dlopen()ing a UBSan module into a
# running Python fails once the static TLS block is used up ("libubsan.so.1: cannot allocate memory in static TLS
# block" - 55 Python tests on Ubuntu 20.04 arm64 in session 6's Linux build check, each passing alone).
if(APPLE)
  set(_r1rt_bind_san_default ON)
else()
  set(_r1rt_bind_san_default OFF)
endif()
option(R1RT_BINDINGS_SANITIZE "Keep UBSan (never ASan/TSan) on the Python module; OFF = no sanitizers"
       ${_r1rt_bind_san_default})

if(R1RT_PYTHON_BINDINGS)
  set(_r1rt_nb_ok OFF)
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
      OUTPUT_VARIABLE _r1rt_nb_dir OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _r1rt_nb_rc ERROR_QUIET)
    if(_r1rt_nb_rc EQUAL 0)
      set(_r1rt_nb_ok ON)
    endif()
  endif()
  if(_r1rt_nb_ok)
    list(APPEND CMAKE_PREFIX_PATH ${_r1rt_nb_dir})
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(r1rt_py NB_STATIC
      ${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/policy_mlp.cpp
      # session 3: the core (step ABI + Core + PolicyRunner + the upper-body executor) and the hg image helper,
      # compiled into the module
      ${CMAKE_CURRENT_LIST_DIR}/src/step_api.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/core.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/policy_runner.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/executor.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/core_hg.cpp
      ${CMAKE_CURRENT_LIST_DIR}/src/crc32.cpp
      ${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py_shm.cpp ${CMAKE_CURRENT_LIST_DIR}/src/shm.cpp)  # s4 logger: r1log's segment reader
    target_include_directories(r1rt_py PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
    target_link_libraries(r1rt_py PRIVATE r1rt_eigen)
    # GCC (Linux, the Orin): nanobind's own headers trip -Wconversion (nb_cast.h: int -> uint8_t) - treat them as
    # system headers so the module's own code keeps every warning (session 6, the Linux build check; CMake >= 3.25).
    if(TARGET nanobind-static)
      set_property(TARGET nanobind-static PROPERTY SYSTEM ON)
    endif()
    # Optimised in every configuration (as r1policy: unoptimised Eigen products are ~18x slower).
    target_compile_options(r1rt_py PRIVATE -O2)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/bindings/r1rt_py.cpp
      PROPERTIES COMPILE_OPTIONS "${R1RT_WARNINGS};-Wno-shadow")
    # nanobind-static is compiled with the module's flags, so both get the same filtering.
    foreach(_t r1rt_py nanobind-static)
      if(TARGET ${_t})
        foreach(_prop COMPILE_OPTIONS LINK_OPTIONS)
          get_target_property(_opts ${_t} ${_prop})
          if(_opts)
            set(_kept "")
            foreach(_o IN LISTS _opts)
              if(_o MATCHES "^-fsanitize=(.*)$")
                if(R1RT_BINDINGS_SANITIZE)
                  string(REPLACE "," ";" _kinds "${CMAKE_MATCH_1}")
                  list(REMOVE_ITEM _kinds address thread)
                  if(_kinds)
                    list(JOIN _kinds "," _kinds)
                    list(APPEND _kept "-fsanitize=${_kinds}")
                  endif()
                endif()
              elseif(_o MATCHES "^-fno-sanitize")
                if(R1RT_BINDINGS_SANITIZE)
                  list(APPEND _kept "${_o}")
                endif()
              else()
                list(APPEND _kept "${_o}")
              endif()
            endforeach()
            set_target_properties(${_t} PROPERTIES ${_prop} "${_kept}")
          endif()
        endforeach()
      endif()
    endforeach()
    message(STATUS "r1rt bindings: r1rt_py with nanobind ${nanobind_VERSION} for ${Python_EXECUTABLE}")
  else()
    message(STATUS "r1rt bindings: nanobind not found in the Python CMake sees - r1rt_py SKIPPED "
                   "(run CMake under `uv run`, as scripts/ci.sh does)")
  endif()
endif()
