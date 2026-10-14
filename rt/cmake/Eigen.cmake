# Eigen 3.4.0 (MPL-2.0; EIGEN_MPL2_ONLY keeps the few LGPL parts out), headers only, for the policy runner
# (DESIGN.md §3.2, §10.2). Pinned release tarball + SHA256, checked by CMake; the hash was computed from the downloaded
# file on 10 Oct 2026 and equals the widely published one for eigen-3.4.0.tar.gz.
# SOURCE_SUBDIR points at a folder without a CMakeLists.txt, so FetchContent only downloads and unpacks: Eigen's own
# build (tests, BLAS, docs) is never configured.
include(FetchContent)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR r1rt_headers_only)
FetchContent_MakeAvailable(eigen)

add_library(r1rt_eigen INTERFACE)
# SYSTEM: Eigen's headers are not held to our -Werror warning set.
target_include_directories(r1rt_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(r1rt_eigen INTERFACE EIGEN_MPL2_ONLY)
