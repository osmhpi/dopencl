#!/usr/bin/env bash
set -euo pipefail

# Build
mkdir -p build_tests
cd build_tests
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Wno-ignored-attributes -DBUILD_UNIT_TESTS=ON -DUSE_OPTIMIZED_SERIAL_IMPLEMENTATION=ON ..
make -j$(nproc)

TEMPDIR="testdir" && rm -Rf "$TEMPDIR" && mkdir -p "$TEMPDIR"
trap 'pkill dcld; pkill -9 dcld;' EXIT

g++ ../standalone_tests/test_explicit_copy_cl.cpp -lOpenCL -Wall -Wextra -DCL_HPP_TARGET_OPENCL_VERSION=120 -o"$TEMPDIR/test_explicit_copy_cl"
g++ ../standalone_tests/test_implicit_copy_cl.cpp -lOpenCL -Wall -Wextra -DCL_HPP_TARGET_OPENCL_VERSION=120 -o"$TEMPDIR/test_implicit_copy_cl"

cd "$TEMPDIR"

(mkdir -p n1 && cd n1 && DCL_LOG_LEVEL=VERBOSE ../../daemon/dcld 127.0.0.1:30000) &
(mkdir -p n2 && cd n2 && DCL_LOG_LEVEL=VERBOSE ../../daemon/dcld 127.0.0.1:30001) &
sleep 1 # TODO: Find a better way to wait for dcld startup

printf "127.0.0.1:30000\n127.0.0.1:30001" > dcl.nodes
find ../test/ -maxdepth 1 -type f -executable -exec env LD_PRELOAD=../icdpp/libdOpenCL.so {} \;

echo "**test_explicit_copy_cl**"
env LD_PRELOAD=../icdpp/libdOpenCL.so ./test_explicit_copy_cl

echo "**test_implicit_copy_cl**"
env LD_PRELOAD=../icdpp/libdOpenCL.so ./test_implicit_copy_cl
