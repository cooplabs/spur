# rt/ - the C++ real-time core (r1rt, r1guard, r1shm)

C++17, CMake, GoogleTest. Session 1 holds the **interfaces only**: the shared-memory segment (`shm_layout.h`), the
seqlock and SPSC ring (`seqlock.hpp`, `spsc_ring.hpp`), the C payload types (`types.h`) and the step C ABI
(`step_api.h`, a stub core that never publishes). Design note: `docs/architecture/SHARED_MEMORY.md`.

Build and test from the repository root (cmake and ninja come from the uv venv):

    uv run cmake -S rt -B build/rt -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON
    uv run cmake --build build/rt
    uv run ctest --test-dir build/rt --output-on-failure

Options: `R1RT_SANITIZE` (ASan + UBSan; UBSan only where ASan binaries cannot run, e.g. macOS 26.5 + Apple clang 17),
`R1RT_TSAN` (ThreadSanitizer build of the same tests; Linux), `R1RT_BUILD_TESTS` (default ON).
