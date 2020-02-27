#!/usr/bin/env bash
set -euo pipefail

# Build
mkdir -p build_tests
cd build_tests
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Wno-ignored-attributes -DBUILD_UNIT_TESTS=ON -DUSE_OPTIMIZED_SERIAL_IMPLEMENTATION=ON ..
make -j"$(nproc)"

TEMPDIR="testdir" && rm -Rf "$TEMPDIR" && mkdir -p "$TEMPDIR"
trap 'pkill dcld; pkill -9 dcld;' EXIT

standalone_tests=(test_explicit_copy_cl test_implicit_copy_cl test_createbuffer_ptr test_createbuffer_ptr_multi test_createbuffer_ptr_race)
for test_name in "${standalone_tests[@]}"; do
    g++ ../standalone_tests/"$test_name".cpp -lOpenCL -Wall -Wextra -DCL_HPP_TARGET_OPENCL_VERSION=120 -o"$TEMPDIR"/"$test_name" &
done
wait

cd "$TEMPDIR"

export DCL_LOG_LEVEL=VERBOSE
(mkdir -p n1 && cd n1 && ../../daemon/dcld 127.0.0.1:30000) &
(mkdir -p n2 && cd n2 && ../../daemon/dcld 127.0.0.1:30001) &
sleep 1 # TODO: Find a better way to wait for dcld startup

printf "127.0.0.1:30000\n127.0.0.1:30001\n" > dcl.nodes
find ../test/ -maxdepth 1 -type f -executable -exec env LD_PRELOAD=../icdpp/libdOpenCL.so {} \;
for test_name in "${standalone_tests[@]}"; do
    echo "**$test_name**" && env LD_PRELOAD=../icdpp/libdOpenCL.so "./$test_name"
done
