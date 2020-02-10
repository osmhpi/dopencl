#!/usr/bin/env bash
set -euo pipefail

TEMPDIR="testdir" && rm -Rf "$TEMPDIR" && mkdir -p "$TEMPDIR"
trap 'pkill dcld; pkill -9 dcld;' EXIT
gcc ../../Utils/test_implicit_cl.c -lOpenCL -Wall -Wextra -DLEGACY_OPENCL_1_2 -o"$TEMPDIR/test_implicit_cl"

cd "$TEMPDIR"

(mkdir -p n1 && cd n1 && DCL_LOG_LEVEL=VERBOSE dcld 127.0.0.1:30000) &
(mkdir -p n2 && cd n2 && DCL_LOG_LEVEL=VERBOSE dcld 127.0.0.1:30001) &
sleep 0.25

printf "127.0.0.1:30000\n127.0.0.1:30001" > dcl.nodes
env LD_PRELOAD=/opt/dopencl/lib/libdOpenCL.so ./test_implicit_cl
