#!/bin/bash
# Occupancy/tile sweep for the sm70 tensor-op GEMM. Builds standalone (it needs no ninfer headers)
# and prints kernel attributes plus achieved TFLOP/s. Usage: occ_probe.sh [tokens]
set -u
T=${1:-3412}
cd /root
timeout 1800 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 bash -lc \
  "nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] -Xptxas -v -I/w/ninfer-v100/third_party/cutlass/include -o /w/tile_occ /w/tile_occ.cu -lcudart" > /root/occ_build.log 2>&1
echo "BUILD RC: $?"
grep -E "error" /root/occ_build.log | head -20
[ -x /root/tile_occ ] || { echo "no binary"; exit 1; }
timeout 900 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 /w/tile_occ "$T"
