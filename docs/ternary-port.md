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
| `NINFER_TERNARY_GEMV_UNROLL` | group-walk unroll, default **8**. 1->29.9, 2->34.1, 4->40.4, **8->41.6**, 12->33.9, 16->30.9, 20->25.7 t/s uncapped; register counts 2->36, 4->48, **8->72**, 12->95, 16->116, 20->134. Inside the shipped 64-register cap: 4->39.7, **8->43.6**, 12->41.1, 16->39.6, 20->42.4 |
| `NINFER_TERNARY_GEMV_MINBLOCKS` | `__launch_bounds__`'s min-blocks, default **4** (measured best: 41.6 -> 43.6 t/s). 1/2/3/5/6/8 exist to re-run the sweep; registers 72/72/72/62/48/40/32 and decode 41.6/41.6/41.5/**43.6**/42.5/38.5/33.5 |
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

The GEMM is at 93% of what the same tile reaches with the card to itself (`tile_occ.cu`, below), so
the reachable prefill remainder is inside the 0.64 s of "everything else", not inside the GEMM — the
15-20% this document projected for the GEMM is not there. A run with `NINFER_TERNARY_CUTLASS=0` on
the new default isolates the arm: 791 -> 1190 is the route, 717 -> 791 is the chunk.

### A cheaper alternative that is not worth it

Cutting the B re-reads by raising the M tile from 128 to 256 halves the dominant traffic term
(27 -> 14 re-reads). Standing alone, the 256x128 tile measures 84.5 TFLOP/s against 128x128's 94.0,
which gives back exactly what the traffic saves. `NINFER_TERNARY_CUTLASS_TILE=128x256` is selectable
and likewise loses (862.8 against 877.8). See the tile table below for the one-CTA occupancy that
explains the loss.

### Software-pipeline depth is not a knob on Volta

Nothing in CUTLASS's sm70 tensor-op path accepts a stage count above two, so there is no
`NINFER_TERNARY_CUTLASS_STAGES`. `kernel::DefaultGemm<..., arch::Sm70, ..., Stages, ...>` is
specialised for `Stages == 2` only (`cutlass/gemm/kernel/default_gemm.h:685`), and the threadblock
kernel it selects is `MmaPipelined`, which asserts `kStages == 2`
(`cutlass/gemm/threadblock/mma_pipelined.h:137`). Asking `device::Gemm` for 3 or 4 stages leaves
`kernel::DefaultGemm` an incomplete type and the compile dies inside `cutlass/gemm/device/gemm.h`:
55 errors, all downstream of that one missing specialisation, and **no static_assert names the real
cause** — the first symptom is "incomplete type ... is not allowed" on a 4000-character template
dump. The multistage route is closed too: every multistage `DefaultMma` specialisation is keyed on
`Sm80`/`Sm75`, and the sm70 shared-memory iterators carry no stage dimension to index. Deeper
pipelining on Volta is a hand-written kernel, not a parameter.

### Tile and occupancy sweep: the GEMM is already at its own ceiling

With stages out, the only remaining lever on loads-in-flight is resident CTAs per SM, which the
threadblock tile fixes through the per-thread accumulator count. Measured by
`tools/v100/ternary-probes/tile_occ.cu`, which reads registers and occupancy out of the real
`cutlass::Kernel<GemmKernel>` via `cudaFuncGetAttributes` /
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than estimating them, then times the GEMM at
M=3412 on this model's two dominant shapes:

| tile | warp | thr | regs | CTAs/SM | gate_up 34816x5120 | down 5120x17408 |
|---|---:|---:|---:|---:|---:|---:|
| **128x128x32** | **64x64** | **128** | **232** | **2** | **86.1 TFLOP/s** | **95.4 TFLOP/s** |
| 128x128x32 | 32x64 | 256 | 138 | 1 | 78.2 | 79.1 |
| 128x64x32 | 64x32 | 128 | 148 | 3 | 76.6 | 33.5 |
| 64x128x32 | 32x64 | 128 | 152 | 3 | 71.2 | 56.6 |
| 64x128x32 | 64x64 | 64 | 250 | 4 | 67.5 | 78.1 |
| 128x128x64 | 64x64 | 128 | — | — | 61.2 | 59.2 |
| 64x64x32 | 32x32 | 128 | 100 | 4 | 26.4 | 25.8 |
| 256x128x32 | 64x64 | 256 | 226 | 1 | (84.5 standalone) | (—) |

Rows smaller than the shipped tile are not neutral, they are strictly worse twice over: a smaller
warp tile costs more shared-memory traffic per MMA than the extra resident CTAs give back, and a
smaller M tile doubles the B re-reads. The 64x128 rows make that second effect visible without
benefit — 1127 GB/s of apparent B traffic on gate_up, *above* HBM peak because it is L2-served, for
71 TFLOP/s against 86. So the shipped 128x128x32/64x64 at 2 CTAs/SM and 232 registers is the
measured optimum of this family, and its in-situ 84 TFLOP/s is 93% of the 86.1/95.4 it reaches with
nothing else on the card. (The older "84 of 99" comparison was against a different tile's standalone
number.)

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

### In-situ kernel budget, and what else is in the step (nvprof)

`nvprof` works on this host even though `ncu` needs `NVreg_RestrictProfilingToAdminUsers=0`. A full
`--print-gpu-trace` of a decode-only window (slice after the last CUTLASS prefill kernel) accounts for
**22.5 ms of GPU kernel time per token at a 96% busy rate** with only 0.84 ms/token of inter-kernel
gap, so the step is not host- or launch-bound. Sorted by summed duration, at a 612-token context
(37.4 t/s, i.e. 26.7 ms/token):

| kernel | calls/token | µs/call | ms/token | share |
|---|---:|---:|---:|---:|
| `ternary_pq2_gemv_w_kernel` (T=1 GEMV) | 301 | 45.8 | 13.8 | **50%** |
| `ternary_pq2_gemv_tile_kernel<2>` (T=2) | 100 | 70.4 | 7.0 | **26%** |
| `ternary_rotation` (folded-basis transform) | 258 | 13.4 | 3.45 | **13%** |
| `causal_attention_small_t_*` | 16 | 44 | 0.71 | 3% |
| `rmsnorm_cta` / `rmsnorm_warp` | 209 | 4-5 | 1.0 | 4% |
| `gdn recurrent/gating/conv` | 96 | 4-10 | 0.83 | 3% |
| `residual_add`, `silu_and_mul` | 320 | 3-5 | 0.69 | 3% |

The two numbers that were not known before: the **T=2 tile path is a quarter of decode**, and the
**rotation is 13%**. The T=2 path is not duplication — the call multiplicities say exactly 16 of the
model's 64 layers process two tokens per step and 48 process one (48x6.25 + 16x6.25 = 400 calls, and
`lm_head` is the 401st), so no weight is read twice. Routing those layers to the tile kernel, which
reads each weight once for both tokens, is the right call.

**The rotation's 13% is real, and it is not bandwidth.** `NINFER_TERNARY_HADAMARD=0` (numerically
wrong by construction, it skips the transform) measures **43.7 -> 47.7 t/s**, i.e. 2.1 ms of the
22.9 ms step, which matches the trace. Each call is one warp per (1024-block, token) pair — 5 warps
for k=5120 at T=1 — with 32 strided 2-byte loads and 32 four-byte sign loads per lane, so it is pure
latency: `rotation.cu` records the author measuring 3.40 µs for the same launch standalone and 4.74 µs
when the warps are packed 8 to a block. In situ it costs 8.1 µs, which is why it is worth 9% of the
step while moving 4 KB. Forcing `NINFER_TERNARY_ROTATE_WPB=8` reproduces the author's loss in situ
(43.3 against 43.7) and `=2` is neutral (43.6), so packing is not the fix. Cutting it means fewer calls
(258 serve 401 linears today) or a rewrite that puts more warps and 16-byte loads behind each D1024.

**Two hypotheses this document used to carry are now refuted by measurement.**

*Widening the transactions* — the previous "main lever", and the refutation that this document used to
carry was itself wrong. `gemv_probe.cu` measures pure-load **bandwidth** (869.6 GB/s for the row
pattern against an 882.6 GB/s flat `uint4` ceiling), and a bandwidth measurement cannot refute a
**latency** argument: fewer load instructions can pay even when the bytes are identical. The real
reason the idea is dead is arithmetic. PQ2 packs a group into 32 contiguous code bytes, one per lane,
so the warp's code read is already **one** `LDG` per group per warp — there is no instruction count
left to remove — and the bytes cannot be reduced either, because 32 of the group's 34 bytes are codes.
A 16-bit-per-lane load would have each lane fetch its neighbour's byte, which the decode does not want;
making lane *l* fetch groups *4g..4g+3* at the same offset needs those bytes to be contiguous, and they
are 32 bytes apart. Only a re-packed code plane would change that, i.e. a different artifact layout.

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

**`kMinBlocks` buys some of that back — this one shipped.** `__launch_bounds__`'s second argument caps
registers at `65536/(threads*kMinBlocks)`; at 256 threads that is 256/128/85/64/51/42/32 registers for
kMinBlocks = 1/2/3/4/5/6/8, so only 4 and up constrain the 72-register kernel. ptxas lands inside the
cap with **no spill** (STACK and LOCAL both 0 at every value):

| kMinBlocks | 1 | 2 | 3 | **4** | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| register cap | 256 | 128 | 85 | **64** | 51 | 42 | 32 |
| registers used | 72 | 72 | 72 | **62** | 48 | 40 | 32 |
| warps/SM | 28 | 28 | 28 | **32** | 42 | 50 | 64 |
| decode t/s | 41.6 | 41.6 | 41.5 | **43.6** | 42.5 | 38.5 | 33.5 |

**4 is the default: 41.6 -> 43.6 t/s (+4.8%)**, identical across repetitions, with the greedy token
ids unchanged at every value. Below 48 registers it falls off — the unroll-8 arm runs out of room for
its eight in-flight loads — so the memory-level-parallelism hypothesis is confirmed but with a small
payoff: +14% occupancy for +4.8% throughput.

Re-sweeping `kUnroll` *inside* the 64-register cap does not unlock a higher unroll, which was the
obvious follow-up: 4 -> 39.7, **8 -> 43.6**, 12 -> 41.1, 16 -> 39.6, 20 -> 42.4. Unroll 8 is still the
peak.

**Keeping the weights resident.** The loader opens the artifact with `O_DIRECT`
(`src/artifact/reader.cpp:182`), deliberately, so the 7.16 GiB is read from disk on **every process
start** and can never occupy the page cache — measured 6856 MiB of disk reads and 7.3 s per run at
~940 MiB/s. `tmpfs` accepts `O_DIRECT` (reads are served from RAM), so a copy on `/dev/shm` needs no
code change at all:

| artifact at | load | rate | disk reads |
|---|---:|---:|---:|
| `/opt/models/ninfer` (nvme) | 7.3 s | 940 MiB/s | 6856 MiB |
| `/dev/shm` (tmpfs, 7.8 GB of 62 GB RAM) | **1.5 s** | **4.3 GiB/s** | **0 MiB** |

Throughput is untouched (prefill 1.20k, decode 43.6 from either copy) and the tmpfs copy reproduces
the disk copy's token ids exactly. It is volatile and costs 7.8 GB of RAM, so it is a benchmarking
convenience, not a deployment change — and the structural fix for the repeat cost is to load once per
process (`ninfer_bench`, or `ninfer-serve`), which is what `docs/v100.md`'s own sweep does.

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
   verify ran a 4-token tile with two of them dead. Isolated (`tile_probe.cu`): 1.283 ms at two live
   tokens against 1.024 ms for a real `kT = 2`, i.e. a fifth of the pass thrown away. `kT` now follows
   the token count.
2. A token's four activations are four contiguous bf16, loaded as two 4-byte reads; one 8-byte read
   replaces them.
3. The per-token activation pointer was recomputed inside the group walk; it is now hoisted.

**Why an MTP round costs what it costs, and why prose cannot win.** A profile of one decode round
(`nvprof --print-gpu-trace`, slice between two consecutive once-per-token sample kernels, 88-token
context) gives the verify pass's true price. Per round, 401 routed linears each:

| round | verify kernel | calls | total | per call | vs a T=1 step |
|---|---|---:|---:|---:|---:|
| no spec (T=1) | `ternary_pq2_gemv_w_kernel` | 401 | 18.06 ms | 45 µs | 1.00x |
| K=3 (T=4) | `ternary_pq2_gemv_tile_kernel<4>` | 401 | 50.59 ms | 126 µs | **2.80x** |
| K=7 (T=8) | `ternary_pq2_gemv_tile_block_kernel<4,8>` | 401 | 93.43 ms | 233 µs | **5.17x** |

Both differences give the same **marginal cost of one extra verified token: 0.60 of a whole T=1
step**, and that single number decides the whole feature. A round costs `1 + 0.60*(T-1)` and yields
`1 + sum(position acceptances)`, so a verified slot only pays for itself when its acceptance exceeds
0.60. Measured acceptance by draft position on the same prose prompt (counts over the rounds):

| K | decode t/s | rounds | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.1 | 125 | 58.4% | -- | -- | -- | -- | -- | -- |
| 2 | 40.8 | 105 | 60.0% | 28.6% | -- | -- | -- | -- | -- |
| 3 | 31.7 | 103 | 48.5% | 30.1% | 14.6% | -- | -- | -- | -- |
| 4 | 20.8 | 92 | 55.4% | 38.0% | 13.0% | 9.8% | -- | -- | -- |
| 5 | 18.4 | 102 | 47.1% | 27.5% | 13.7% | 5.9% | 1.0% | -- | -- |
| 7 | 18.8 | 96 | 54.2% | 30.2% | 15.6% | 6.3% | 1.0% | **0** | **0** |
| no spec | 43.8 | -- | -- | -- | -- | -- | -- | -- | -- |

The model reproduces the measurements to a few percent: K=3 costs 2.80 for 1.93 tokens = 0.69x
(measured 31.7/43.8 = 0.72), K=7 costs 5.20 for 2.07 = 0.40x (measured 0.43). **No position and no
window wins**: positions 6 and 7 are never accepted, position 5 once in 102 rounds, and even
position 1 at 48-60% is at or below the 0.60 break-even. K=1 is the best arm and it is a 1.6% loss.
Upstream's V100 and 5090 numbers are not mysterious by comparison: on the 5090 the same arithmetic
with a 0.10 marginal gives a break-even acceptance of 0.15, which is why their story fixtures still
win 1.63x at 37.9% acceptance.

**The missing unroll -- this one shipped, and it flips prose MTP from a loss into a win.** The
small-tile GEMV's group walk carried **no `#pragma unroll` at all**. The T=1 GEMV's own curve
(1 -> 29.9, 2 -> 34.1, 4 -> 40.4, 8 -> 41.6, 12 -> 33.9 t/s) is this family's only control on
memory-level parallelism, and it was simply absent here while the per-token work was layered on top.
Measured at MTP K=3 (T=4), prose:

| tile unroll | 1 | 2 | **4** | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| decode t/s | 31.7 | 36.6 | **42.0** | 38.8 | 36.8 |

and at K=1 (T=2), where depth 8 is the better arm: unroll 4 -> 50.5, **8 -> 51.8**. The default is
therefore per-`kT` (8 at two tokens, 4 at three and four) with `NINFER_TERNARY_GEMV_TILE_UNROLL`
overriding it.

The row-blocked kernel, which serves every verify pass wider than four tokens, has the same shape of
loop and behaves the **opposite** way: its token loop is already fully unrolled at kT=8, so deepening
the group walk only inflates an already-large body. K=7 on the repeated-sentence prompt:

| block unroll | **1** | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| decode t/s | **63.1** | 46.9 | 34.9 | 34.7 | 18.8 |

Depth 1 is what it shipped with, so the kernel keeps no pragma and `NINFER_TERNARY_BLOCK_UNROLL`
exists only to keep the negative result reproducible.

**The row-blocked kernel should not serve the verify band at all -- this also shipped.** It is the
kernel the dispatcher hands every `T >= 5` pass to, and it loses the entire band to the small-tile
GEMV once `kT = 5..8` instantiations exist:

| arm | K=4 | K=5 | K=6 | K=7 (prose) | K=7 (repeated sentence) |
|---|---:|---:|---:|---:|---:|
| row-blocked `4x8` | 20.8 | 18.4 | 19.0 | 18.8 | 63.1 |
| **small-tile, wide** | **38.9** | **29.3** | **28.1** | **25.1** | **83.7** |

+87% at K=4 and +33% on the repetition workload the context-lookup path rides on. The tile kernel is
now the default for the whole `T = 2..8` band; `NINFER_TERNARY_TILE_WIDE=0` restores the row-blocked
kernel as the A/B arm. This also erases the anomaly the cost model showed: K=4 and K=5 now sit on the
`1 + 0.60*(T-1)` line instead of far below it, so the model holds across every window measured.

The final prose sweep, defaults as shipped:

| K | no spec | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode t/s | 43.7 | **51.8** | **48.0** | 41.9 | 38.9 | 29.3 | 25.1 |
| acceptance length | -- | 1.58 | 1.89 | 1.93 | 2.16 | 1.95 | 2.07 |

and on the repeated-sentence prompt (the context-lookup workload), no spec 43.5 -> K=3 44.8,
K=5 69.5, **K=7 83.7**.

### The full 15-token lookup window

The context-lookup path is real and now measured end to end. Its gate is in `program_impl.h` and has
three conditions, only the first of which is documented elsewhere: an earlier **exact** occurrence of
the last 16 tokens must exist, `max_lookup_extent > draft_window` (so any K below 15), and -- the one
that explains why the same prompt fires at some windows and not others -- **the MTP head's first K
drafts must agree exactly with the lookup's**, via `std::equal(found->begin(), found->begin()+K,
mtp_drafts.begin())`. When it fires, `verify_k` becomes `kMtpLookupMaximumDrafts` = 15, so the verify
pass is **T = 16**, and the `mtp accepted by pos` histogram grows from `draft_window + 1` entries to
15 -- that histogram is the diagnostic.

On a prompt that asks the model to repeat a sentence ten times (no spec 43.8 t/s):

| K | decode t/s | acceptance length | lookup active |
|---:|---:|---:|:--:|
| 1 | 100.8 | 7.37 | yes |
| 2 | 103.6 | 9.05 | yes |
| 3 | 104.6 | 9.95 | yes |
| 7 | **104.7** | 12.44 | yes |

**2.3-2.4x over no spec.** On the 16x-repeated-sentence prompt it fires at K=2 (82.6), K=5 (81.7) and
K=7 (83.9) but not at K=1 or K=3, which is condition three doing the selecting.

The T=16 pass used to go to the row-blocked kernel. Extending the small-tile GEMV to `kT = 16` -- the
`kT <= 8` static assert was the only thing stopping it -- and routing it there is worth:

| lookup prompt | tile `kT=16` | row-blocked | delta |
|---|---:|---:|---:|
| K=7 | **105.0** | 74.1 | +42% |
| K=1 | **100.8** | 76.1 | +32% |

The 80.5 t/s this document previously quoted for the lookup path was the row-blocked kernel's number.

### Bit-exactness: an in-tree claim that does not hold here

`src/product/speculative_options.h:41` records that the sm_70 "width-6+ drift" was caused by the
Volta small_t verify kernels being dropped in the DFlash2 merge, and that with them restored "**every
draft window through 7 is bit-exact against `--spec none` again**". Measured on this artifact that is
not true, and the pattern is the opposite of the one described: **K=7 is exact and K=1 and K=3 are
not.** With one prompt, one KV setting (int8, capacity 4096, 53 prompt tokens), greedy and 40
generated tokens, K=1 and K=3 fork at position 12 into a deterministic continuation that no-spec and
K=7 do not take. Reproducible across repetitions, and five causes are ruled out: the unroll depth
(1/4/8 give byte-identical ids), the wide-tile routing, `--lm-head-draft` (`k1_plain` == `k1_draft`),
run-to-run noise, and a settings confound (prompt tokens, KV capacity, token count and finish reason
all match). `--greedy` is honoured in every arm (`sampling greedy (temperature 0)`).

What is established is that a near-tie exists: with BF16 KV the same two continuations appear but the
*assignment flips*, so no-spec takes the branch K=1 took under int8. That is consistent with rounding
deciding a near-tie, but it does not establish which arm is right, and it does not reconcile the
in-tree claim. **Either that comment is stale or the small-window verify path has a real defect.**
Until it is settled, the speculative path should not be described as bit-exact with `--spec none`.

**The greedy-id divergence is a numerical tie, not a logic error.** Under `--greedy
--print-token-ids` the speculative paths do not always reproduce the no-spec sequence: on this prompt
the ids agree for twelve tokens and then fork deterministically into one of exactly two
continuations. Three pieces of evidence say that is rounding rather than a bug in acceptance or
rollback. The unroll is not the cause -- depths 1, 4 and 8 produce byte-identical ids, and the
no-spec arm is stable across repetitions. The fork depends on the **KV dtype**, and the *assignment
flips*: with INT8 KV, no spec takes `95761 99128` and K=1 takes `96304 98267`; with BF16 KV it is the
other way round, with the same two branches. So both branches are reachable in every configuration,
the two arms differ only in which kernel's rounding decides a near-tie, and the verify pass is
authoritative for what it emits (which is the documented contract). It is not bit-exactness with the
T=1 path, and it should not be described as such.

The K sweep after the fix, same prose prompt as the acceptance table above:

| K | no spec | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode t/s | 43.7 | **51.8** | **48.1** | 41.9 | 20.8 | 18.4 | 18.8 |
| acceptance length | -- | 1.58 | 1.89 | 1.93 | 2.16 | 1.95 | 2.07 |

**K=1 is +18.5% and K=2 is +10.1% over no spec on ordinary prose**, which is the first time this
port beats its own baseline with speculation on text that is not reproducing the prompt. Note what
the fix did *not* change: the acceptance, the cost model, or the shape of the curve. It removed dead
latency from the verify kernel and the two smallest windows fell out on the winning side of the 0.60
break-even. K=4 and K=5 are still far below the model's 27.8 and 21.4 because `T >= 5` switches to
the row-blocked kernel; extending the small-tile GEMV past four tokens is the next thing to measure
there.

**Correctness caveat.** See the bit-exactness section above: the speculative and non-speculative paths
can commit different tokens at a near-tie, deterministically per draft window, and the in-tree claim
that all windows through 7 are bit-exact does not reproduce. The unroll and the wide-tile changes are
both numerically neutral (identical greedy ids across unroll depths 1/4/8), so neither is the cause.
Until the claim is reconciled, no speculative window should be described as bit-exact with
`--spec none`.


### Where the T=1 decode could get to (the ceiling question)

The T=1 step is **issue-bound, not bandwidth-bound**, and the probe arithmetic says so directly.
Baseline 22.9 ms/token; `NINFER_TERNARY_GEMV_PROBE` arms on the real kernel:

| arm | removes | decode | ms/token | delta |
|---|---|---:|---:|---:|
| baseline | -- | 41.6 | 24.0 | -- |
| `nocode` | the code load **and** its 12-instruction decode | 63.6 | 15.7 | -8.3 |
| `codealu` | the code load only, value from ALU | 50.1 | 20.0 | -4.0 |
| `noact` | the two activation loads and their converts | 45.5 | 22.0 | -2.0 |
| `noscale` | the group-scale load | 45.3 | 22.1 | -1.9 |

The code stream is 6.4 GB of the 7.19 GB moved per token, and deleting its *load* buys only 4.0 ms --
which would be 1.6 TB/s if it were a bandwidth effect, i.e. impossible on a 900 GB/s card. So the
load's cost is its instruction, and the decode arithmetic behind it is another 4.3 ms. That is what
"314 GB/s, a third of peak" means: the memory system is idle most of the step.

The same kernel shape proves the machine can go much faster when there is something to amortise over.
The `kT = 16` verify kernel processes sixteen tokens per weight pass at **7.4 ms per verified token**
(measured: 104.7 t/s at an acceptance length of 12.44), against 22.9 ms for the T=1 kernel on the same
weights. And `gemv_probe.cu`'s row-pattern read ceiling (869.6 GB/s) puts the pure weight stream at
7.19/0.8696 = 8.3 ms/token = **120 t/s absolute floor**, while the non-GEMV half of the step
(rotation 4.0 ms, norms, attention, GDN, launch) is a measured 7.7 ms that nothing in the GEMV can
touch.

Reachable estimates, all anchored on measurements rather than on the peak:

| target | ms/token | t/s |
|---|---:|---:|
| today | 22.9 | 43.7 |
| GEMV at the `nocode` bound (code stream gone -- not implementable, it is the weights) | 15.7 | 63.6 |
| GEMV at its measured weight-read floor + today's 7.7 ms of other work | 16.0 | 62 |
| the same, with the rotation's 4.0 ms fused away | 12.0 | 83 |
| absolute weight-stream floor (no other kernel costs at all) | 8.3 | 120 |
| demonstrated with amortisation (16-token verify) | 7.4 | 105-135 |

**The honest ceiling for a non-speculative single-stream decode on this card is ~60-65 t/s**, and
~83 t/s only if the folded-basis rotation is folded into its producer. Both are capped by the fact
that a T=1 step has exactly one token to hide its latency behind; the 105 t/s the lookup path reaches
is not a decode-kernel improvement at all, it is sixteen tokens sharing one weight pass.

**A measurement that is not available on this host.** `nvprof --print-gpu-trace` works, but
`nvprof --metrics` does not: any metric collection fails with "No events/metrics were profiled" /
CUDA profiling error (exit 12). So there is no DRAM/L2 counter read for the decode step; the
bandwidth figures in this document come from byte counts and wall time, not from counters.

**Three fixes were attempted and all are refuted.**

*Occupancy.* The tile kernel was the obvious suspect: the T=1 kernel is pinned at 62 registers and
32 warps/SM, while the tile kernel carried no `__launch_bounds__` second argument at all. It does not
need one -- `tile_kernel<4>` compiles to **32 registers**, the minimum of the family, and therefore
already runs at maximum occupancy. There is no register budget to reclaim.

*Routing the verify band to the row-blocked kernel.* The tile kernel re-reads the activation once per
output row, and the row-blocked kernel exists precisely to amortise that over `kR` rows (loads per
FMA 0.56 at kR=1/kT=4 against 0.19 at kR=4/kT=4). The author measured it losing on Ada (MTP decode
43.0 -> 32.8 t/s) because there the tile kernel was issue-bound at 95% occupancy; on Volta that
argument is gone, so it was re-measured in situ at K=3:

| (kR,kT) | 1,2 | 1,4 | 1,8 | 2,2 | 2,4 | 2,8 | 4,2 | 4,4 | 4,8 | 8,2 | 8,4 | tile<4> |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode t/s | 19.6 | 23.5 | 14.5 | 22.3 | 30.3 | 17.1 | 23.1 | 26.7 | 19.0 | 16.6 | 17.9 | **31.7** |

The Ada result reproduces on Volta: nothing in that family reaches the tile kernel. The switch is kept
as `NINFER_TERNARY_T2BLOCK` (off by default) so the measurement can be repeated.

*Shape sweeping the wide verify band.* K=7's T=8 pass goes to `gemv_tile_block_kernel<kR,kT>`; swept
with `NINFER_TERNARY_ROWS` x `NINFER_TERNARY_TILE` on the repeated-sentence prompt (no spec 43.5,
default kR=4/kT=8 = 63.1): r4t4 **63.8**, r2t4 58.2, r2t8 57.0, r8t4 50.5, r1t8 48.5, r1t4 44.8,
r2t2 44.3, r4t2 43.9, r8t2 39.7, r1t2 37.3, r8t8 22.5. The shipped default is within 1.1% of the best
of twelve.

**The strategic conclusion.** MTP's marginal token is expensive here for one reason only: the T=1
decode step is not bandwidth-bound. It moves 7.19 GB of weights in 22.9 ms = **314 GB/s, a third of
the card's 900 GB/s**, so a second or third token through the same weights costs real issue/latency
time rather than riding along free. Fix the base decode and MTP follows; tune MTP first and it cannot
pay. Two corollaries that are easy to get wrong: the K=4 and K=5 arms are worse than the model
predicts (20.8 and 18.4 against 27.8 and 21.4) because `T >= 5` switches kernels, so the small-tile
GEMV's `kT` should be extended past 4 before that band is judged; and the draft stack itself is cheap
-- 15 `w8` and 3 `q4` kernels, 4.2 ms of a 63.6 ms round, 6.6%.

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

## External comparison: llama.cpp on the same card

`llama.cpp:cuda12.8-sm70-mtp-server` (the sm_70 build already on this host) running
`Swift-Qwen3.8-27B-GGUF/IQ2_S` from the local Ollama blob store -- the same Qwen3.8-27B family, a
different quantization -- with `-ngl 99 -c 8192`, greedy sampling, `cache_prompt=false`, one
request per point. Its MTP is `--spec-type draft-mtp`, whose window is a server-start argument, so
every row is a fresh server.

| metric | this port (ternary PQ2, 6.70 GiB) | llama.cpp (IQ2_S, 9.90 GB) |
|---|---:|---:|
| prefill, ~3.4k-token prompt | **1,200 t/s** | 693 t/s |
| decode, no spec, short prompt | **43.7 t/s** | 35.2 t/s |
| decode, no spec, after a 3.4k prefill | 39.7 t/s | 32.4 t/s |
| decode + speculation, ordinary prose | 51.8 t/s (K=1) | **55.4 t/s** (n-max 3) |
| decode + speculation, repetitive text | **105-108 t/s** (context lookup) | 85.9 t/s (n-max 5) |
| effective weight bandwidth per token | 314 GB/s | 348 GB/s |

Three things this settles.

**The prefill route is the port's real win.** 1.7x llama.cpp at the same prompt length and the same
card, which is what the dequantise-once-plus-CUTLASS arm was built for.

**The T=1 GEMV is not badly underperforming.** llama.cpp, a much more mature implementation, lands
at 348 GB/s of weight stream against this port's 314 GB/s -- 11% better per byte, not 2-3x. That
also revises the ceiling estimate above downwards: at llama.cpp's efficiency this model's 7.19 GB
per token would decode at ~48 t/s, so the GEMV-side headroom is about 10%, not 40%. The earlier
"120 t/s is the floor" figure is the DRAM peak, which neither implementation is remotely near.

**The verify band is where llama.cpp is genuinely better, and it is measured.** Its MTP on ordinary
prose is 1.57x (35.2 -> 55.4) where this port manages 1.19x (43.7 -> 51.8). Backing that out of its
own timings: draft_n=128 with draft_n_accepted=83 over 128 committed tokens is 45 rounds, so a round
costs 51.3 ms against a 28.4 ms single step = 1.81x for a T=4 verify, i.e. a **marginal cost of 0.27
per extra verified token against this port's 0.60**. That is the same quantity the cost model above
is built on, measured on a second implementation on the same hardware, and it says 0.27 is
achievable. How the weight is shared across the token dimension inside the kernel is the difference,
and it is the most concrete lead left for this port's verify path.

Operational note, measured: after the model loads, `nvidia-smi` reports **16,063 of 16,384 MiB in
use** -- the 9.90 GB model plus the 0.93 GB mmproj plus the KV/graph allocations leave almost nothing.
At `--spec-draft-n-max 7` a 42-token request still completes (47.3 t/s, 93/219 drafts accepted,
42.5%), but the 3,360-token request drops the connection and the server then refuses further ones.
n-max 1, 3 and 5 all complete the same long-prompt request. So the widest window is not usable at
this context on 16 GB, which is a memory limit rather than a throughput one.

### The verify loop's real cost, and the one change that would pay

SASS of the two kernels, counted by opcode (`cuobjdump -sass` on the ternary object):

| kernel | instructions | per (token, group) | what they are |
|---|---:|---:|---|
| `gemv_w_kernel<8,unroll 8>` | 352 for 8 groups | ~31 (one token) | 4.5 I2F + 4.5 RMT + 4 LOP3 + 3.5 SHF + 4.5 LDG + 4.5 FFMA + 6 IADD3 |
| `gemv_tile_kernel<4,unroll 4>` | 624 for 4 groups, 4 tokens | ~26 | 5 FFMA + 5 PRMT + 1.25 I2F + 1.25 LDG.64 + 2 FADD + ~11 moves/addressing |
| `gemv_tile_kernel<16,unroll 4>` | 2304 for 4 groups, 16 tokens | ~26 | 320 FFMA / 320 PRMT / 297 MOV / 228 IMAD.MOV |

Only five of the tile kernel's ~26 instructions per (token, group) are the multiply-accumulate.
Five are `PRMT` -- the bf16 -> fp32 conversion of that token's four activations -- and ~11 are
address arithmetic and register moves driven by the kT pointers and accumulators. That is why the
marginal cost is 0.60 of a T=1 step rather than the ~0.25 the FMA count alone would suggest.

`NINFER_TERNARY_TILE_SHARE_ACT=1` prices the activation side by making every token in the tile reuse
the first token's values (numerically wrong by construction, never a default):

| arm | K=1 (kT=2) | K=3 (kT=4) | K=7 (kT=8) | repeat prompt (kT=16) |
|---|---:|---:|---:|---:|
| real | 51.8 | 42.0 | 25.1 | 107.8 |
| all-but-one token's loads+converts removed | **69.3** | **51.5** | **32.4** | 32.3 |

**+23% to +34% at kT = 2, 4 and 8**, which is the size of the prize for removing the per-token
`PRMT` conversion. (The kT=16 arm is not interpretable: sharing one activation across sixteen live
accumulators regressed it 3x, which looks like spilling rather than a measurement.)

A real version of that prize is to have the rotation emit **fp32** activations for the decode band,
so no GEMV has to convert: the rotation already holds fp32 in registers and currently rounds to bf16
only for the GEMV to convert straight back. The catch is dtype selection by token count -- prefill
(the fused MMA and CUTLASS arms) reads bf16, so `folded_activation` would have to pick fp32 for
T <= 16 and bf16 above it, the workspace reservation would double for that band, and both GEMV
kernels would need fp32-input variants. It is a multi-file change and it is the largest single item
left on the decode side.
