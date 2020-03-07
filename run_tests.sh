#!/usr/bin/env bash
set -euo pipefail

# Build
mkdir -p build_tests
cd build_tests
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Wno-ignored-attributes -DBUILD_UNIT_TESTS=ON \
    -DENABLE_IO_LINK_COMPRESSION=ON -DUSE_OPTIMIZED_SERIAL_IMPLEMENTATION=ON -DUSE_CL_IO_LINK_COMPRESSION_INPLACE=ON ..
make -j"$(nproc)"

TEMPDIR="testdir" && rm -Rf "$TEMPDIR" && mkdir -p "$TEMPDIR"
cd "$TEMPDIR"

#../test/DataTransfer --run_test=DataTransfer_Benchmark && exit 0

# Start OpenCL daemons
export DCL_LOG_LEVEL=VERBOSE
trap 'pkill dcld; pkill -9 dcld;' EXIT
(mkdir -p n1 && cd n1 && ../../daemon/dcld 127.0.0.1:30000) &
(mkdir -p n2 && cd n2 && ../../daemon/dcld 127.0.0.1:30001) &
sleep 1 # TODO: Find a better way to wait for dcld startup
printf "127.0.0.1:30000\n127.0.0.1:30001\n" > dcl.nodes

find ../test/ -maxdepth 1 -type f -executable -exec echo "**" {} "**" \; -exec env LD_PRELOAD=../icdpp/libdOpenCL.so {} -l unit_scope \;

standalone_tests=(test_explicit_copy_cl test_implicit_copy_cl test_createbuffer_ptr test_createbuffer_ptr_multi test_createbuffer_ptr_race bitonic mapreduce_addition)
for test_name in "${standalone_tests[@]}"; do
    echo "**$test_name**" && env LD_PRELOAD=../icdpp/libdOpenCL.so "../standalone_test/$test_name"
done
