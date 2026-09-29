# Ternary (PQ2_0_G128) port on Volta / sm_70

The `Ambolio/ninfer-4090-windows` ternary lineage is Ada-only by construction: four of its five
GEMM schedules are bf16 or int8 **tensor-core** kernels (`m16n8k32.s8`, bf16 mma), and Volta has
neither. This port compiles the tensor-core schedules out, keeps the SIMT ones, and adds two things
the author's tree does not have: an fp16 `mma.m8n8k4` tensor-core prefill GEMM, and a dequantise-once
+ CUTLASS Sm70 fp16 GEMM arm.

Artifact: `bonsai2_27b_swift_pq2.ninfer` (8.31 GB, identity `qwen3.8-27b/groupwise-int`),
322 `PQ2_0_G128` tensors and zero `PTQ1_0_G128`.

## Current numbers (Tesla V100-SXM2-16GB, driver 580.178.04, `--kv-dtype int8`)

| | value |
|---|---|
| prefill, 3412-token prompt | **1190 t/s** (3 runs: 1190 / 1190 / 1190) |
| — as a fraction of the card | 59.5 TFLOP/s = **47.6%** of the 125 TFLOP/s fp16 peak |
| decode | **41.6 t/s** (unchanged by every prefill change) |
| causal scoring (`ninfer-perplexity`) | 1091 tok/s, was 106.3 |
| startup | 8.7 s (weights 6.70 GiB) |

TFLOP/s convention used throughout: 5.0e10 FLOP per token (2 x 25.1e9), so `t/s * 5.0e10`.

Reference points on the same card and prompt: llama.cpp IQ2_S (MMQ) 661.8 / 33.7.
`docs/v100.md`'s own V100 sweep reports 1,083.9 prefill for the `groupwise-int` MTP artifact — but
that artifact is 17.5-23.7 GB and does not fit a 16 GiB card (that sweep used a V100-32GB), so only
its efficiency (40% of peak) transfers. This port is now above it.

## Correctness of the fast routes, measured numerically

Comparing greedy token ids is a **discrete** test: two routes can differ in the third decimal of
every logit and still emit the same ids, so it can only show that no argmax flipped. It is reported
below as a smoke test and nothing more. The real statement comes from a continuous measure —
perplexity, which aggregates NLL over the whole scored set.

Full-precision NLL on 26,449 scored tokens of ordinary English prose (this tree's own `docs/*.md`,
`--context 4096 --stride 2048`, int8 KV):

| arm | route | mean_nll | PPL | scoring rate |
|---|---|---|---:|---:|
| A | `NINFER_TERNARY_PREFILL=block`, CUTLASS off — PQ2 -> FP32, FP32 FMA | 1.6502392743834533 | 5.208226 | 58.1 tok/s |
| B | `NINFER_TERNARY_CUTLASS=0` — fused fp16 `mma.m8n8k4` tensor-core GEMM | 1.6504842170034242 | 5.209502 | 410.7 |
| C | `NINFER_TERNARY_CUTLASS=1` (product default) — dequantise once + CUTLASS Sm70 fp16 GEMM | 1.6504842170034242 | 5.209502 | 634.0 |

* **B and C are bit-identical**, Δ = `+0.000e+00`. The two fp16 routes agree exactly, per-domain as
  well as in aggregate.
* **Both sit Δnll = +2.449426e-04 above A**, i.e. **+0.0245% PPL**. That is the real cost of fp16
  operands, and it is exactly what the token-id comparison was hiding: no argmax flips, so the ids
  match while the underlying values do not.
* For scale, `q4_volta_mma_gemm.cuh` documents 0.223% L2 relative error for fp16 operands with fp32
  accumulate, and 0.17% for its CUTLASS Sm70 paths. This port measures 0.0245% on PPL.

The scoring rates are also the proof that each arm ran the route it claims: the FP32 SIMT route costs
58.1 tok/s of scoring against 634.0 for the default, and a mis-set environment variable is invisible
in the NLL but obvious in the rate. **Check the rate before believing a route comparison.**

Reproduce with `docs/ternary-port` in mind: any arm must have a *fresh* `--output` directory, or the
app exits and the report read afterwards is a stale one from an earlier arm.

## What runs where

| T | route | into the arena | prefill (3412-token prompt) |
|---|---|---|---|
| 1 | warp-per-row GEMV `ternary_pq2_gemv_kernel` | none | — |
| 2..4 | token-tile GEMV `ternary_pq2_gemv_tile_kernel<4>` (weights read once for all tokens) | none | — |
| 5..31 | row-blocked GEMV `ternary_pq2_gemv_tile_block_kernel<kR,kT>` | none | — |
| >= 32 | fp16 tensor-core GEMM `ternary_volta_mma_gemm` | none | 717 (this is the arm all other rows below beat) |
| >= 256 | **dequantise + CUTLASS Sm70 fp16 tensor-op GEMM** `ternary_cutlass_sm70` | weight chunk (<= 64 MiB) + fp16 activation | **1190** |
| any | correctness-first reference | none | — |

Both fast prefill arms are selected automatically; `NINFER_TERNARY_CUTLASS=0` removes the second and
`NINFER_TERNARY_PREFILL=block` forces the reference.

## Parameters

### Effective

| variable | effect |
|---|---|
| `NINFER_TERNARY_CUTLASS` | **default on** (unset or any value except `0`). `0` disables the dequantise+CUTLASS arm, leaving the fused fp16 MMA route. Costs 1190 -> 717 on prefill, changes nothing on decode |
| `NINFER_TERNARY_CUTLASS_MIN_T` | token threshold for the arm, default **256**. Below it CUTLASS is a GEMV and measures 0.3 TFLOP/s, so this is what keeps decode and the MTP verify off the route. Lowering it to 1 is a correctness-preserving but very slow configuration |
| `NINFER_TERNARY_CUTLASS_TILE` | `128x128` (default) or `128x256`. `128x256` halves the A-panel re-reads in principle but measured 862.8 against 877.8, so it stays unselected |
| `NINFER_TERNARY_CUTLASS_PROBE` | `dequant` / `dequant2` / `dqwrite` / `dqread` / `gemm`. Timing probes, **numerically wrong by design**. `dequant2` runs the dequant twice; the delta between `dequant` and `dequant2` is one whole dequant pass. See the "how the chunk was found" section |
| `NINFER_TERNARY_PREFILL` | `mma` (default) = fused fp16 tensor-core GEMM; `simt` = PQ2 SIMT GEMM; `block` = row-blocked SIMT GEMV, the FP32 A/B oracle; `ref` = reference kernel; `wide` **throws** (see below) |
| `NINFER_TERNARY_MMA` | fused-arm shape: `auto` (default) or `c4t1` `c4t2` `c4t4` `c4t8` `c8t2` `c8t4`. An unknown name throws |
| `NINFER_TERNARY_MMA_COMMIT` | `early` (default) / `trailing` / `split`. Where the weight decode and its shared store sit relative to the MMA loop. Orthogonal to `_MMA` |
| `NINFER_TERNARY_MMA_PROBE` | `nodecode` / `noaload`. Timing probes, **numerically wrong by design** |
| `NINFER_TERNARY_ROWS` / `_TILE` | row/token block of the blocked GEMV, defaults 4 / 8. Only read on the `block` route |
| `NINFER_TERNARY_SIMT` | shape of the `simt` route: `r8c4` (default) / `r8c8` / `r16c4` / `r8c8p2` / `r16c8` / `r32c4` |
| `NINFER_TERNARY_GEMV_WARPS` | warps per block of the T=1 GEMV: 4/8/16/32, default 8. Measured 41.2 / 40.4 / 38.3 / 35.8 — flat to inverse |
| `NINFER_TERNARY_GEMV_UNROLL` | group-walk unroll, default **8**. 1->29.9, 2->34.1, 4->40.4, **8->41.6**, 12->33.9, 16->30.9, 20->25.7 t/s. Register counts: 2->36, 4->48, **8->72**, 12->95, 16->116, 20->134 |
| `NINFER_TERNARY_GEMV_ROWS` | output rows per warp, 1 (default) / 2 / 4. **2 and 4 lose** (24.0 and 16.2 t/s against 41.6) because register pressure binds before the shared activation pays |
| `NINFER_TERNARY_GEMV_PROBE` | `noact` / `nocode` / `codealu` / `noscale`. Deletes one load class at a time, **numerically wrong by design**. This is how the decode step's cost was located |
| `NINFER_TERNARY_HADAMARD=0` | skip the folded-basis rotation. **Numerically meaningless** — diagnostic only |
| `NINFER_TERNARY_GDN_PERM` | default off. `=1` restores llama.cpp's `.ssm_out.` permutation; leave it off, the runtime already emits grouped V |

### Wrong values, and what they cost

| setting | side effect |
|---|---|
| `--prefill-chunk 1024` (the old default) | prefill falls 1190 -> 996 and causal scoring 1091 -> 106 tok/s. See the section below; this is the single most expensive wrong value in the tree |
| `NINFER_TERNARY_CUTLASS_MIN_T=1` | decode drops from 41.6 t/s to a walk: the CUTLASS GEMM at M=1 measures 0.3 TFLOP/s. Correct output, unusable rate |
| `NINFER_TERNARY_PREFILL=wide` | **throws `std::invalid_argument`**. Deliberate: it names the author's bf16 weight-resident kernel, which does not exist here, and silently aliasing it to a SIMT route would make an A/B against the author's numbers look like a regression when it is really a missing route |
| `NINFER_TERNARY_MMA=bogus` | throws `std::invalid_argument`, and because it is thrown from a `static` initialiser the process dies with rc=139 (`terminate called after throwing...`). Documented behaviour, verified |
| `NINFER_TERNARY_HADAMARD=0` | forward pass is fast and the output is garbage. Never a default |
| `NINFER_TERNARY_CUTLASS_PROBE` / `_MMA_PROBE` | same: fast, wrong. Never defaults |

### Ineffective on sm_70 — and why

| variable | why it does nothing |
|---|---|
| `NINFER_TERNARY_TSUB` | superseded by `NINFER_TERNARY_MMA`; unknown variables are ignored |
| `NINFER_TERNARY_S8=0` | the int8 rung is `m16n8k32.s8` and is inside `#ifndef NINFER_VOLTA_BUILD`; the scratch is not even allocated, so the arena is not charged |
| `NINFER_TERNARY_VERIFY`, `_SMALL_T_ROWS` | only select between the author's small-T bf16 mma kernel and the tiled GEMV; the former does not exist here and the latter is already the only route |
| `NINFER_TERNARY_TOKEN_GRID` | only the wide-tile and s8 kernels read it, both compiled out |

### Compile-time

`NINFER_VOLTA_BUILD` is defined in the top-level `CMakeLists.txt:20` when configuring for sm_70. It
gates the tensor-core includes themselves, not just call sites — including `ternary_rowsplit_mma*.cuh`
at all pulls in mma PTX and bf16 arithmetic helpers that do not exist for sm_70.

## Prefill: how 717 -> 1190 was reached

### Stage 1 — the fused fp16 tensor-core GEMM (111.7 -> 703.5)

`ternary_volta_mma_gemm.{cuh,cu}` is `q4_volta_mma_gemm.cuh` with the code fetch and the decode
swapped; the shared-memory layout is identical to Q4's. PQ2 packs four 2-bit codes per byte, so a
32-weight kStep is 8 code bytes per row and one lane's share is a single aligned 16-bit load. The
decode reuses "0x6400|n is exactly 1024+n as fp16": one `__hsub2` against 1025.0 yields `code-1`
exactly and one `__hmul2` by the group scale finishes it, so `(code-1)*scale` is an exact fp16 result
and the route reproduces the FP32 reference.

Shape sweep (`kWarps` warps x `kTSub` 32-token A sub-tiles), 3412-token prompt:

| variant | warps | tsub | shared/CTA | CTAs/SM | threads/SM | FLOP/byte | measured |
|---|---:|---:|---:|---:|---:|---:|---:|
| c4t1 | 4 | 1 | 10240 | 8 | 1024 | 4.0 | 463.3 |
| c4t2 | 4 | 2 | 15360 | 6 | 768 | 5.33 | 580.4 |
| c4t4 | 4 | 4 | 25600 | 3 | 384 | 6.4 | 628.7 |
| c4t8 | 4 | 8 | 46080 | 2 | 256 | 7.11 | 578.7 |
| c8t2 | 8 | 2 | 20480 | 4 | 1024 | 5.33 | 674.6 |
| c8t4 | 8 | 4 | 30720 | 3 | 768 | 6.4 | **714.1** |
| c8t8 | 8 | 8 | 51200 | — | — | 7.11 | **not built**: over the 48 KiB static shared limit, and dominated by c4t8 anyway |

`FLOP/byte = 8*kTSub/(kTSub+1)` is independent of `kWarps`; `kTSub` is the only ratio lever and it
saturates, which is why c8t4 wins. Two claims that were in this document and are now **wrong**:

* *"c4t4 beats c8t4"* — that was an artefact of a dispatcher bug. `NINFER_TERNARY_MMA` was only
  honoured on the `trailing` commit path; the `early`/`split` arms hardcoded `c8t4`, so
  `_MMA=c4t4 _COMMIT=early` silently ran c8t4 and the two "different" numbers were one
  configuration measured twice. Shape and commit are now composed (`launch_selected<kProbe,kCommit>`)
  and the sweep above is the corrected one.
* *"auto re-decides per prefill chunk, and that is where it beats every fixed shape"* — with the
  CUTLASS arm on, the fused arm is only reached below 256 tokens, where `auto` resolves to c8t4.

`NINFER_TERNARY_MMA_COMMIT`: trailing 709.6, **early 734.2**, split 725.1, all token-identical. The
commit hoist recovers only 3.5% of the 20.6% the `nodecode` probe says the decode costs, so the
decode's price is the work itself plus its shared store, not latency exposure.

### Stage 2 — dequantise once, then CUTLASS (703.5 -> 996)

The fused kernel's ceiling is shared-memory traffic, not the tensor core: per k-slice the A-fragment
load and the decode's shared store bracket the MMA, so ~28-34% of peak is all the shape can reach.
CUTLASS's own pipeline has no such constraint. Measured standalone on this card with plain fp16
operands and this model's exact shapes:

| shape (M=3412) | TFLOP/s | % of 125 |
|---|---:|---:|
| qkv/gate_up 17408x5120 | 99.7 | 79.7% |
| o/down 5120x17408 | 93.8 | 75.1% |
| lm_head 248320x5120 | 99.3 | 79.4% |
| **M=1, 17408x5120** | **0.3** | 0.2% |
| M=16, 17408x5120 | 4.8 | 3.8% |

`ternary_cutlass_sm70.{h,cu}` dequantises a bounded chunk of the PQ2 weight to fp16 once and then
runs the same `cutlass::gemm::device::Gemm<..., OpClassTensorOp, Sm70, 128x128x32, ...>` that
`fp8_cutlass_sm70.cu` already used. This is **prefill-only**: the M=1 rows above are why `admits()`
gates on the token count.

Traffic is the trade: PQ2 is 0.266 B/weight and fp16 is 2 B/weight, and the GEMM re-reads the B panel
once per M-tile. That is 8x more weight traffic; it wins because the fused kernel is compute/shared
bound at 29% while this is at 85% of both the compute and the bandwidth ceiling simultaneously.

Coverage. Reaching every projection needed the arena threaded into the `_basis` entry points
(attention q/gate/k/v and GDN qk/value/z consume one shared activation each). Those projections are
19.3% of the ternary FLOPs and were still on the fused arm:

| family | params | before | after |
|---|---:|---|---|
| `mlp/gate_up` (linear_swiglu) | 11.409 B | CUTLASS | CUTLASS |
| `mlp/down` (linear_add) | 5.704 B | CUTLASS | CUTLASS |
| `gdn/value_z` | 3.020 B | fused | **CUTLASS** |
| `gdn/output` | 1.510 B | CUTLASS | CUTLASS |
| `output_head` | 1.271 B | CUTLASS | CUTLASS |
| `gdn/query_key` | 1.007 B | fused | **CUTLASS** |
| `attention/query_key` | 0.587 B | fused | **CUTLASS** |
| `attention/gate_value` | 0.587 B | fused | **CUTLASS** |
| `attention/output` | 0.503 B | CUTLASS | CUTLASS |

Two details that are load-bearing:

* The shared-activation projections must **not** convert the activation in place — the attention
  parent feeds four weights from one rotation buffer, so an in-place bf16->fp16 would corrupt the
  three that follow. The launcher takes its own fp16 copy from the arena.
* `ternary_cutlass_sm70_workspace_bytes(k, cols)` deliberately does not take `n`. The attention and
  GDN planners are handed only `(input_rows, max_tokens)` and do not know the output row counts at
  all, and an under-declared reservation is what makes the arena reject the graph at runtime. The
  launcher allocates a strict subset, so the approximation is always in the safe direction.

Also required: the CUTLASS C operand advances by a whole chunk, so the chunk length has to keep
`c + n0` on a 128-bit boundary. An unrounded 6553-row chunk made `can_implement()` fail outright.

### Stage 3 — the prefill chunk, which was the expensive one (996 -> 1190)

The dequant looked 9x slower in situ than standalone. It is not — the forward pass was being entered
several times per prompt.

How it was found, in order, because the first two steps each pointed somewhere wrong:

1. `dequant2` probe (run the dequant twice, everything else held fixed). One pass, isolated
   additively:

   | T | 556 | 1060 | 2068 | 3412 |
   |---|---:|---:|---:|---:|
   | `T(dq2)-T(dq1)` | 0.152 | 0.149 | 0.308 | 0.579 |

   One pass is 0.152 s for 58 GB = **382 GB/s**, which matches the standalone 413 GB/s. So the
   kernel was never the problem, and the cost scales with the prompt length even though the dequant
   work does not.
2. Two counters, because a host counter cannot see a replayed CUDA graph: entries into
   `run_chunks` (host) and blocks executed (device). The host counter alone answered it —
   **400 / 400 / 800 / 1600** at those four lengths, where 400 is exactly one forward pass
   (64 layers x ~6 routed linears). `prefill_chunk` splits the prompt and every chunk re-runs the
   entire forward, so every chunk re-materialises all 25.6e9 weights.
3. `prefill_chunk` defaults to **1024** and is a CLI flag (`--prefill-chunk`):

   | `--prefill-chunk` | forward entries | prefill |
   |---|---:|---:|
   | 1024 (old default) | 1600 | 996 |
   | 2048 | 800 | 1100 |
   | **4096 (new default)** | **400** | **1190** |
   | 8192 | 400 | 1190 |

**This is not specific to the CUTLASS arm.** The fused arm re-reads the PQ2 planes once per pass too:
717 -> 791 from the same flag. Defaults changed in `apps/cli/options.h` (the CLI's real default —
`serve_options.h` only serves `ninfer-serve`, and `engine.cpp:30` only `CausalScoring`) and in
`src/serve/serve_options.h`. Cost: workspace peak 244.8 -> 787.8 MiB, still 7.24 GiB free at
max-context 8192. Explicit `--prefill-chunk` still overrides.

### The budget now (2.87 s for a 3412-token prompt)

| segment | time | note |
|---|---:|---|
| CUTLASS GEMM | 2.08 s | 84 TFLOP/s of the 99 measured solo; simultaneously 663 GB/s = 74% of HBM |
| dequant | 0.145 s | one pass, 382 GB/s. Was 0.579 s |
| everything else | 0.64 s | rotations, norms, attention/GDN, embeddings, the <256-token tail |

The GEMM is near both ceilings at once, so it has little left; the reachable remainder is roughly
15-20%, not the 30-40% this document previously projected. A run with `NINFER_TERNARY_CUTLASS=0` on
the new default isolates the arm: 791 -> 1190 is the route, 717 -> 791 is the chunk.

### A cheaper alternative that is not worth it

Cutting the B re-reads by raising the M tile from 128 to 256 halves the dominant traffic term
(27 -> 14 re-reads). Standing alone, the 256x128 tile measures 84.5 TFLOP/s against 128x128's 94.0,
which gives back exactly what the traffic saves. `NINFER_TERNARY_CUTLASS_TILE=128x256` is selectable
and likewise loses (862.8 against 877.8).

### The bigger structural idea, and why it is not next

An sm70 tensor-core GEMM that decodes PQ2 in the B-load path would drop B traffic from 1.38 TB to
~0.18 TB, since only 0.266 B/weight would move instead of 2. But that is exactly what the fused
kernel is, and its ceiling is the shared-memory staging, not the tensor core. Getting past ~33%
inside that structure means feeding B to the MMA from registers, which is a rewrite. Not while
1.38 TB is still being fed at 74% of HBM.

## Traps hit while porting (each cost a build+run cycle)

1. **`DeviceArena::alloc` throws `std::bad_alloc` when the arena is undersized.** A folded ternary
   weight always allocates its rotation buffer, and the GDN/attention/SwiGLU/residual compose routes
   additionally allocate a `[rows, T]` projection scratch *even at widths where the fused Q4/Q5
   schedule needs neither*. Every `*_workspace_capacity_bytes` that can serve a ternary parent must
   reserve both, or the failure surfaces as a bare `error: std::bad_alloc` with no phase detail. The
   original tree's `gdn_input_proj_conv_snapshot_workspace_capacity_bytes` had an early `return 0`
   when `largest_materialized_width == 0` that must be removed. The CUTLASS arm adds two more
   consumers — the weight chunk and the fp16 activation — and they belong in the same four planners.

2. **The LM head uses the workspace-free `linear()` overload** in five places (`program_impl.h`,
   `text_prefill_impl.h`, `dflash_impl.h`, and four in `text_context_impl.h`). Those must switch to
   `ops::linear(..., ops::LinearPolicy::A16Only, arena, stream)`. The symptom names the weight:
   `ternary linear: the folded basis needs a rotation workspace [N=248320, K=5120, T=1, qtype=10]`.

3. **The `causal_score` plan needs `ternary_rotation_workspace_bytes`.** The output-head linear runs
   inside that arena; without the scratch the *first scored window* dies with `bad_alloc`.

4. **A capture-time launch that was correct can be invalidated by a chunk-width change.** The
   `CausalScoring` purpose forced `prefill_chunk = 1024` while its default window is 4096, so every
   scored window was four forward passes. Measured 106.3 -> 1091.4 tok/s from changing only that.
   Note the scored NLL moves with the chunk width (1.18127 at 1024 against 1.17942 at 4096 on the
   same tree) because the chunk width changes which kernel handles a given token count; the two arms
   are not bit-identical. The CUTLASS arm *is* exactly neutral against the fused arm at a fixed
   chunk (1.17942 against 1.17942).

5. **`cp.async` is polyfilled, silently.** `src/ops/common/memory.cuh` makes it a synchronous
   vectorized load+store below `__CUDA_ARCH__ 800`, with `cp_commit`/`cp_wait` as no-ops. Kernels
   written with a cp.async pipeline therefore compile and are correct on sm_70 — they just lose the
   overlap. Do not read "it uses cp.async" as "it cannot run on Volta".

6. **bf16 is fine in SIMT, only bf16 *mma* is missing.** This tree's `q4_rowsplit_gemm_simt.cuh`
   already loads `__nv_bfloat16` and converts with `bf16x2_bits_to_float2`; the author's GEMV uses
   `__bfloat1622float2` / `__float2bfloat16_rn` only, so it needed no arithmetic changes.

7. **The same Bonsai weights exist in three incompatible dialects.** `PQ2_0_G128`/`PTQ1_0_G128` =
   this tree; `t2_g128_fp16` = `iamwavecut/ninfer-all`; NVFP4/groupwise-int = `Neroued/ninfer`. The
   binding layer reads the *declared* format, which is what lets one `groupwise-int` weights profile
   serve both the INT-scheduled and the ternary artifact.

8. **`--print-token-ids` writes to stderr**, on a line `tokens      generated ids     <ids...>`.
   stdout carries only the container's CUDA banner, so grepping numbers out of stdout compares the
   banner and proves nothing. Extract with `sed -n 's/^tokens *generated ids *//p'`.

9. **A timing probe that is optimised away measures nothing.** A read-only variant of the dequant
   needs the store kept alive behind a condition the compiler cannot fold.

10. **`static_assert` on shared memory is not the launch limit.** c8t8's 51200 B compiles under
    `__launch_bounds__` accounting but fails at nvlink with `uses too much shared data (0xc800 bytes,
    0xc000 max)`; the message names the mangled kernel, not the template arguments.

## Decode

T=1 warp-per-row GEMV, **41.6 t/s** (3 runs identical), against llama.cpp's 33.7 on the same prompt.
Unaffected by every prefill change; verified again at chunk 4096.

**What it is not limited by**, measured rather than argued:

| knob | result | conclusion |
|---|---|---|
| inner-loop instruction count | a probe deleting 4 of ~30 instructions gives 40.3 / 40.6 / 39.6 against 40.4 | not ALU-bound |
| warps per block | 4 -> 41.2, 8 -> 40.4, 16 -> 38.3, 32 -> 35.8 | not resident-warp bound |
| group-loop unroll | 1 -> 29.9, 2 -> 34.1, 4 -> 40.4, **8 -> 41.6**, 12 -> 33.9, 16 -> 30.9, 20 -> 25.7 | **memory-level-parallelism bound** |

The unroll curve is decisive: throughput moves 39% across it while deleting instructions moves it 0%.
The old default carried no pragma and landed on 40.4 — exactly ptxas's own 4x — so pinning 8 is a free
3% and stops the default depending on what ptxas happens to choose.

**Where the headroom is.** 5.0e10 FLOP per token over 7.19 GB of weights is 6.95 FLOP/byte, against
the card's ridge point of 139 FLOP/byte. Decode is 20x below the ridge and can only be bandwidth
bound. The per-token budget: FP32 CUDA cores 3.2 ms, fp16 tensor cores 0.40 ms, HBM 8.0 ms, measured
24.0 ms. The tensor core is 20x faster than the memory floor, so using it would buy nothing — and it
cannot be used at T=1 anyway, because `m8n8k4`'s A operand is 32 tokens tall and one token leaves 31
rows idle. llama.cpp splits the same way (`mmvq` for decode).

**What the 24.0 ms is actually made of.** Five probes on the real T=1 kernel, each deleting one load
class (`NINFER_TERNARY_GEMV_PROBE`, all numerically wrong by design), `--max-new 200`:

| probe | deletes | decode | share of the step |
|---|---|---:|---:|
| baseline | -- | 41.6 | -- |
| `nocode` | code byte load **and** its decode | 63.6 | **35%** |
| `codealu` | the code load only (byte computed from indices) | 50.1 | 15% |
| `noact` | the two activation loads + their bf16->f32 converts | 45.5 | 9% |
| `noscale` | the group-scale load | 45.3 | 8% |

`nocode` splits as **39% load, 61% decode**: the decode arithmetic alone is ~13% of the step and the
code load ~8%. That the load is only ~15% is the decisive number, because:

**The code read is already at the memory limit.** The code path costs 8.3 ms of the 24.0, and moving
7.15 GB at the bandwidth this access pattern actually reaches is 8.2 ms. There is nothing left to win
there.

**Two hypotheses this document used to carry are now refuted by measurement.**

*Widening the transactions* — the previous "main lever". A standalone read-pattern benchmark on this
model's real PQ2 geometry (`tools/v100/ternary-probes/gemv_probe.cu`, 713 MB, warp-per-row as in the kernel) shows the existing
one-byte-per-lane pattern is **not** the problem:

| pattern | GB/s |
|---|---:|
| flat uint4 stream (ceiling) | 882.6 |
| **row pattern, 1 byte per lane (today)** | **869.6** |
| row pattern, 2 / 4 / 8 / 16 bytes per lane | 894 |

1 byte per lane already reaches 98.5% of the flat-stream ceiling, and widening buys 3%. The loads are
perfectly coalesced; "sixteen narrow loads" was a wrong diagnosis.

*Several output rows per warp* — the activation window is shared by every row, so holding `kRows`
rows per warp cuts instructions per weight and multiplies the independent code streams per warp. It
loses badly, and the reason is registers:

| rows per warp | registers | decode |
|---:|---:|---:|
| 1 (default) | 72 | **41.6 t/s** |
| 2 | 74 | 24.0 |
| 4 | 85 | 16.2 |

The default already sits at 72 registers, i.e. 28 of the 64 warps per SM. Giving each warp more work
costs more resident warps than it saves in instructions.

**Register pressure is the wall, and `kUnroll` is the only lever on it.** Measured register counts for
the shipped kernel: unroll 2 -> 36, 4 -> 48, **8 -> 72**, 12 -> 95, 16 -> 116, 20 -> 134. Unroll 8 is
the throughput peak (41.6) and everything above it trades warps for loads it cannot afford.

**What is left.** The only removable pieces are the decode arithmetic (13%), the activation handling
(9%) and the scale load (8%) — about 30% between them, and no one of them is large. The one concrete
design that attacks all three at once is to run the group dot in fp16: build `(code-1)` as a half2
through the same `0x6400|n == fp16(1024+n)` identity the prefill kernel uses, keep the activation as
half2, accumulate the group's four terms with `__hfma2`, and convert to fp32 once per group. That
takes the inner loop from ~28 instructions to ~18. **It changes precision** — each group of 128
weights would be summed in fp16 before joining the fp32 accumulator — so it is a real change, not a
free win, and it has to be qualified against the `block` FP32 oracle rather than assumed. It was not
taken here.

**Refuted, do not retry:** dp4a / fp16x2 as an instruction-count play (the ALU is not the whole
story: the decode is 13% and the load 15%, so removing instructions cannot pay much), warps-per-block
(flat to inverse), multiple rows per warp (above), wide transactions (above), and QPN for the verify —
`q4_launch.h`'s own measured table has QPN losing at T=4 (n=34816: 284.7us incumbent against 298.0us
QPN; n=4096: 51.1 against 66.6), and T=4 is exactly our verify width.

**MTP: a 2x win on context-reproduction output, a loss on prose.**

The verify pass runs `ternary_pq2_gemv_tile_kernel`, which loads the code byte and the scale once per
group and reuses them for every token in the tile. Three defects were found and fixed:

1. `launch_pq2_gemv_tile` **always instantiated `kT = 4`** whatever the token count, so a 2-token
   verify ran a 4-token tile with two of them dead. Isolated (`tools/v100/ternary-probes/tile_probe.cu`): 1.283 ms at two live
   tokens against 1.024 ms for a real `kT = 2`, i.e. a fifth of the pass thrown away. `kT` now follows
   the token count.
2. A token's four activations are four contiguous bf16, loaded as two 4-byte reads; one 8-byte read
   replaces them.
3. The per-token activation pointer was recomputed inside the group walk; it is now hoisted.

Then the context-lookup fast path, which is what `docs/v100.md`'s 100+ decode numbers actually ride on.
`lookup_draft()` takes the last 16 tokens and searches the ledger for an earlier occurrence of that
exact suffix; if it finds one it proposes the tokens that followed it, up to
`kMtpLookupMaximumDrafts`, and the target verifies all of them in one wide pass. It therefore fires
only when the output reproduces earlier text. Measured on this artifact, 256 generated tokens:

| prompt | no spec | MTP k3 | acceptance | tok/round |
|---|---:|---:|---:|---:|
| ordinary prose | **41.7** | 30.7 | 1.90 | 1.90 |
| the prompt's own list, repeated verbatim | 41.7 | **80.5** | 100% | 13.00 |

with `--no-thinking`. The K sweep on the verbatim prompt:

| arm | decode | accepted length | round ms |
|---|---:|---:|---:|
| no spec | 41.7 | -- | 23.98 |
| K=1 | 74.6 | 10.40 | 139.41 |
| K=3 | 80.5 | 13.00 | 161.49 |
| K=5 | 78.2 | 13.87 | 177.37 |
| K=7 | **81.2** | 14.86 | 183.00 |

Two operational details that matter and are easy to get wrong:

* **`--thinking` breaks it.** The first attempt without `--no-thinking` measured 56.0 t/s with lookup
  accepting only ~20% of positions: the model spent the start of its output reasoning *about* the
  request instead of emitting the copy, so the 16-token suffix did not yet match anything. The
  acceptance histogram is the diagnostic -- `mtp accepted by pos` has 15 entries when lookup is
  active and only `draft_window + 1` when it is not.
* It is **workload-specific and must stay opt-in**. On ordinary prose MTP is a 26% loss (30.7 against
  41.7); the doc's own wording is "context-reproduction results, not general-generation headlines".

**Why 200 t/s is not reachable here by this route.** The proposal is capped at
`kMtpLookupMaximumDrafts` = 15 lookup tokens, so a round cannot exceed ~16 committed tokens. At the
measured round cost that caps the route near 100 t/s: 16 tokens would cost roughly
`24 + 15 * 11.1 = 190 ms`, and 11.1 ms/token is the marginal verify cost read off the K=1 row above.
`docs/v100.md` reports 201.0 t/s at 12.91 tokens per round, i.e. a 64 ms round and a marginal cost of
~3.3 ms/token -- **3.4x cheaper per verified token than this port**. That gap is the non-amortisable
part of a verify token (activation reads and conversions, four FMAs per four weights, plus the
attention/GDN/norm work that scales with M), and it is the same wall the T=1 kernel hits: the group's
~28 instructions against ~11 per additional token, so no amount of weight residency fixes it.

**The remaining constraint on ordinary generation is acceptance, and it is a property of the
artifact.** This draft head accepts 45.5% on prose (2.55 tokens per round at K=3); the `Qwen3.8-27B`
rows in `docs/v100.md` accept 97.1%. Their own round-latency table (35B-A3B, real greedy acceptance,
same card class) shows what a weight-resident verify looks like: **K=1 -> K=3 grows the round from
23.80 to 30.20 ms, +27% for two more draft tokens**, and that is what turns a 79.84 t/s step rate into
125.85. Their step rate at K=1 is 42.0 rounds/s; this port's step rate is 41.6 tokens/s. **The kernels
are equally fast; the difference is accepted length.**

## The SIMT GEMM (`NINFER_TERNARY_PREFILL=simt`)

`ternary_rowsplit_gemm_simt.cuh` is this tree's `q4_rowsplit_gemm_simt.cuh` with the group geometry
swapped: Q4 packs eight 4-bit codes per 32-byte group, PQ2 packs sixteen 2-bit codes into the same 32
bytes, so the staging geometry is bit-identical and only the per-phase K offset and the per-lane
element count change (`phase*256 + lane*8` -> `phase*512 + lane*16`). It needs `decode_sixteen`, which
was added to `PQ2SimtDecodeAtom`.

With `--greedy`, all eight routes (`ref`, `block`, `r8c4`, `r8c8`, `r16c4`, `r8c8p2`, `r16c8`, `r32c4`)
produce byte-identical token id sequences. That is the discrete smoke test only — see the measured
perplexity comparison above for what these routes actually differ by.

| schedule | prefill | MTP decode |
|---|---:|---:|
| block (reference) | 111.8 | 30.5 |
| r8c4 | 94.7 | 35.1 |
| r8c8 | 111.2 | 31.0 |
| r16c4 | 98.0 | **36.4** |
| r8c8p2 | 111.8 | — |
| r16c8 | **115.4** | 30.1 |
| r32c4 | 87.5 | 33.1 |

It is not the default: it ties the blocked GEMV on prefill and only wins on the verify band (+19%),
which still leaves MTP below plain decode.

`kGroupsPerStage` is pinned to 8 rather than Q4's 16 because it must divide the row's group count for
a CTA to take the unpredicated path; this model's widths give 40 / 48 / 80 / 136 groups per row and 8
is their common divisor. With 16, k=5120 and k=17408 fall back to the predicated walk.

## Container version: this tree reads v2 only, and the ModelScope artifacts are v3

`src/artifact/reader.cpp` puts the version byte inside the magic and compares all eight bytes:

```cpp
constexpr std::array<std::byte, 8> kMagic = {'N','I','N','F','E','R', 0, 2};
...
if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
    throw ArtifactError("artifact magic is not NInfer v2");
}
```

`sanbanfu/Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer` (9,520,051,456 B, sha256
`6d8b62b589cfc57736b515f4dc13e7c5c06c09a726ceee3628ac20a86d8aa883`) and its sibling
`Swift-Bonsai-2-ninfer-v3-aux.ninfer` (8,307,236,592 B) are both **version 3** and are rejected there.

| offset | v2 | v3 |
|---|---|---|
| 0..6 | `NINFER\0` | `NINFER\0` |
| 7 | version `2` | version `3` |
| 8 | u64 json_len | u64 json_len |
| 16 | **JSON starts here** | 16 more bytes (a hash) |
| 32 | — | **JSON starts here** |

The schema differs too: v3 is `{"components":{"text":{"config":...}}}` against v2's
`{"identity":{...},"objects":[...]}`. Supporting v3 is bounded work (version dispatch, the extra 16
bytes, the payload offset, the components schema) but not a one-liner.

Per that repo's README the *official* v3 artifact's ternary core is corrupted (PPL stuck near 16.3)
and `-spliced` rebuilds 322 ternary objects from a `Ternary-Bonsai-2-27B-PQ2_0.gguf`, measuring PPL
5.628. The README lists the format as `t2_g128_fp16 (PQ2_0, 128 groups of 34 bytes)` — the same 32
code bytes + 2 scale bytes as this tree's `PQ2_0_G128`, but the dialect name decides the code
dictionary and they differ: T2 is two's-complement signed 2-bit (00->0, 01->+1, 11->-1), PQ2 is
`code - 1` over {0,1,2}. Same container geometry, different dictionary.

## Perplexity

`ninfer-perplexity` is not in the default build target. Corpus:
`eval/corpora/perplexity-1m/manifest.json` (3.9 MB, 261,167 scored tokens in `--quick`).

```bash
cmake --build build-sm70 --parallel 12 --target ninfer-perplexity
./build-sm70/apps/ninfer-perplexity models/x.ninfer \
    --corpus eval/corpora/perplexity-1m/manifest.json --quick
```

Defaults are `--context 4096 --stride 2048`; `--quick` selects the corpus subset. A full `--quick`
pass was ~1h45m at 33-86 tok/s of scoring throughput; after the `prefill_chunk` fix the scoring rate
is ~1090 tok/s on a short text and the pass is bounded by the stride-2048/window-4096 overlap
recomputing half of every prefill, not by the weights.

Measured on `bonsai2_27b_swift_pq2.ninfer`, **partial (20 of 124 windows), PPL ~7.5**, still trending
down when interrupted. Reference points: the repaired `-spliced` v3 artifact is 5.628 and the
corrupted official v3 artifact is ~16.3. So this v2 artifact sits between them — functional, but
measurably behind the repaired one. Treat 7.5 as an upper bound until the pass completes; it has not
been re-run since the prefill work and the chunk width shifts the score slightly, so it should be.

## What was deliberately not ported

* The four author tensor-core schedules (`ternary_rowsplit_mma`, `_mma_small_t`, `_mma_wide_t`,
  `_mma_s8`), ~135 KB. Replaced by the two that exist here.
* bonsai's removal of `#ifdef NINFER_VOLTA_BUILD` around `kNvfp4TextPolicy`/`kFp8TextPolicy` (Volta
  forces `A16Only`), the `w8_gdn_input_workspace_bytes` return, the q4 fused-tensor-core workspace
  budget in `linear.cpp`, and the q4/q5 dispatch `workspace` parameter. Those are Volta work the Ada
  tree does not have; reverting them would break Q4/Q5.
* `text_context_impl.h`'s `kMtpLookupMaximumWidth` -> `kMaximumMtpDraftTokens + 1` bound change and
  the `mtp_lookup_replay_records` plan block. Unrelated to ternary.

## Profile access

`ncu` is in the build image at `/usr/local/cuda/bin/ncu` (not on `PATH`). It cannot profile here: the
driver returns `ERR_NVGPUCTRPERM`, which needs `NVreg_RestrictProfilingToAdminUsers=0` in a host
modprobe file plus a module reload or reboot — not done, the host GPU is in use. Also, ncu sees
nothing inside CUDA graphs; pass `--no-cuda-graph`.

Because of that, every number above comes from A/B runs and timing probes on the real model rather
than from a profiler, which is why each probe is documented with what it deletes.

## Known cosmetic issue

`launch_pq2_gemv` in `ternary_rowsplit_gemm.cu` is declared but never referenced (the T=1 path goes
through `launch_pq2_gemv_tile`). Harmless; one nvcc warning. Dead in the author's tree too.
