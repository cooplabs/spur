# r1rt_probe_sanitizer(<flags> <out_var>): compile AND run a trivial program with <flags> (10 s timeout).
# Why: on macOS 26.5 with Apple clang 17 (Command Line Tools 26.3) every AddressSanitizer binary hangs at start-up and
# every ThreadSanitizer binary crashes (found 9 Oct 2026, session 1; a toolchain problem, not ours). A sanitizer that
# cannot run is reported loudly and left out instead of hanging the test run.
function(r1rt_probe_sanitizer flags out_var)
  set(dir "${CMAKE_BINARY_DIR}/sanitizer_probe")
  string(MAKE_C_IDENTIFIER "${flags}" tag)
  file(MAKE_DIRECTORY "${dir}")
  file(WRITE "${dir}/probe.cpp" "int main() { return 0; }\n")
  separate_arguments(flag_list UNIX_COMMAND "${flags}")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" ${flag_list} "${dir}/probe.cpp" -o "${dir}/probe_${tag}"
    RESULT_VARIABLE build_rc OUTPUT_QUIET ERROR_QUIET)
  set(ok FALSE)
  if(build_rc EQUAL 0)
    execute_process(COMMAND "${dir}/probe_${tag}" RESULT_VARIABLE run_rc TIMEOUT 10 OUTPUT_QUIET ERROR_QUIET)
    if(run_rc EQUAL 0)
      set(ok TRUE)
    endif()
  endif()
  set(${out_var} ${ok} PARENT_SCOPE)
endfunction()
