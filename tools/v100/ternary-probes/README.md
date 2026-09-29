# Ternary port measurement probes

Standalone CUDA microbenchmarks used to justify the numbers in
[`docs/ternary-port.md`](../../../docs/ternary-port.md).

They exist because **this host has no working profiler**: `ncu` is present in the build image but the
driver returns `ERR_NVGPUCTRPERM` (it needs `NVreg_RestrictProfilingToAdminUsers=0` plus a module
reload), and even then ncu sees nothing inside CUDA graphs. Every claim in that document about *why*
a kernel is slow was therefore obtained by isolating its memory access pattern, or by deleting one
load class at a time on the real model, rather than by reading counters.

Build and run one inside the build container:

```bash
docker run --rm --gpus all -v $PWD:/w -w /w ninfer-v100-buildenv:cu128 bash -lc \
  'nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] \
     -I/w/third_party/cutlass/include -o /w/probe /w/tools/v100/ternary-probes/<probe>.cu \
     -lcudart && /w/probe'
```

The `*.sh` wrappers do exactly that plus the `docker run` for the result, but they assume the probe
source is at `/w/<name>.cu`; copy the `.cu` next to the script first, or adjust the path.

| probe | question it answers | result |
|---|---|---|
| `cutlass_probe.cu` | How fast is the CUTLASS Sm70 fp16 **tensor-op** GEMM that `src/ops/linear/fp8/fp8_cutlass_sm70.cu` already uses, on this model's real shapes with plain fp16 operands? | qkv/gate_up 99.7, o/down 93.8, lm_head 99.3 TFLOP/s (**75-80%** of the 125 TFLOP/s peak). At M=1 the same GEMM measures **0.3 TFLOP/s** — which is why the CUTLASS prefill arm is gated on token count |
| `cutlass_tiles.cu` | Which CUTLASS tile shape wins? | `128x128x32` (94.0). `256x128x32` loses (84.5) because the extra M-tile height costs more latency-hiding than the halved B re-reads buy; K=64 loses harder (61.1) |
| `gemv_probe.cu` | Is the T=1 GEMV's decode limited by its **one-code-byte-per-lane** load pattern? | No. The row pattern reaches **869.6 GB/s** against an **882.6 GB/s** flat `uint4` ceiling — 98.5%. Widening to 2/4/8/16 bytes per lane buys 3%. This **refuted** the "wide transactions" plan |
| `tile_probe.cu` | Does the MTP verify kernel (`ternary_pq2_gemv_tile_kernel`) actually amortise the weights across its token tile? | Only partly. Against the shipped T=1 kernel on the same weights it costs **1.91x at M=2** and **2.56x at M=4**, where weight-residency would be ~1.05x. It also showed `kT=4` running with only two live tokens wastes **20%** (1.283 ms vs 1.024 ms) — which is now fixed by dispatching `kT` from the token count |
| `dq_probe.cu` | Why was the PQ2 -> fp16 dequant kernel slow? | Its store pattern. One 16-byte store per thread at byte-stride 32 measured **367 GB/s**; putting one contiguous 16-byte store in each thread measured **1112 GB/s** (the `cudaMemset` reference on the same buffer is 892). This is the one probe whose finding became a shipped change |

Two probes that belong with these but are env-gated arms of the real kernels rather than standalone
files: `NINFER_TERNARY_MMA_PROBE` (fused tensor-core prefill: `nodecode` +20.6%, `noaload` +13.9%)
and `NINFER_TERNARY_GEMV_PROBE` (`noact` / `nocode` / `codealu` / `noscale` on the T=1 GEMV, which
located the decode step's cost split). Both are numerically wrong by design and are never defaults.

**A probe that is optimised away measures nothing.** A read-only variant of the dequant needs its
store kept alive behind a condition the compiler cannot fold, or the loads it is supposed to be
measuring disappear with it.
