#!/bin/bash
cd /root
timeout 1800 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 bash -lc \
  'nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] -I/w/ninfer-v100/third_party/cutlass/include -o /w/cutlass_tiles /w/cutlass_tiles.cu -lcudart' > /root/ct_build.log 2>&1
echo "BUILD RC: $?"
grep -E "error" /root/ct_build.log | head -20
[ -x /root/cutlass_tiles ] || { echo "no binary"; exit 1; }
timeout 900 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 /w/cutlass_tiles 3412
