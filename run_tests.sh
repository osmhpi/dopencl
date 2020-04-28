#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[0;36m**Building dOpenCL...**\033[0m"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_UNIT_TESTS=ON ..
make -j"$(nproc)"

TEMPDIR="testdir" && rm -Rf "$TEMPDIR" && mkdir -p "$TEMPDIR"
cd "$TEMPDIR"

echo -e "\033[0;36m**Starting dOpenCL daemons...**\033[0m"
trap 'timeout 3 killall -vws2 dcld || killall -vws9 dcld;' EXIT
(mkdir -p n1 && cd n1 && ../../daemon/dcld 127.0.0.1:30000) &
(mkdir -p n2 && cd n2 && ../../daemon/dcld 127.0.0.1:30001) &
printf "127.0.0.1:30000\n127.0.0.1:30001\n" > dcl.nodes

# Wait for the daemons to start up
echo -e "\033[0;36m**Waiting for dOpenCL to be up...**\033[0m"
if command -v nc >/dev/null; then
    timeout 15 sh -c 'while ! nc -z 127.0.0.1 30000; do sleep 0.1; done
                      while ! nc -z 127.0.0.1 30001; do sleep 0.1; done'
else
    echo "WARNING: netcat not installed, can not probe for dcld startup. Execution may fail!"
    sleep 1 # And pray for the best
fi

echo -e "\033[0;36m**Running unit tests...**\033[0m"
find ../test/ -maxdepth 1 -type f -executable -exec echo "**" {} "**" \; -exec env LD_PRELOAD=../icdpp/libdOpenCL.so {} -l unit_scope \;

echo -e "\033[0;36m**Running standalone tests...**\033[0m"
standalone_tests=(test_explicit_copy_cl test_implicit_copy_cl test_createbuffer_ptr test_createbuffer_ptr_multi test_createbuffer_ptr_race bitonic mapreduce_addition)
for test_name in "${standalone_tests[@]}"; do
    echo "**$test_name**" && env LD_PRELOAD=../icdpp/libdOpenCL.so "../standalone_test/$test_name"
done
