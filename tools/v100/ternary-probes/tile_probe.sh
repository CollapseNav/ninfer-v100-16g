#!/bin/bash
cd /root
timeout 900 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 bash -lc \
  'nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] -o /w/tile_probe /w/tile_probe.cu -lcudart' > /root/tp_build.log 2>&1
echo "BUILD RC: $?"
grep -E "error" /root/tp_build.log | head -20
[ -x /root/tile_probe ] || { echo "no binary"; exit 1; }
timeout 900 docker run --rm --gpus all -v /root:/w -w /w ninfer-v100-buildenv:cu128 /w/tile_probe 40
