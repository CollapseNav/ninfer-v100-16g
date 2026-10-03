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
| prefill, 3412-token prompt | **1190 t/s** (3 runs: 1190 / 1190 / 1190); re-measured on the current build at **1200-1210** (3662 tokens) and **1230** (4042 tokens, 4096 chunk) |
| — as a fraction of the card | 59.5 TFLOP/s = **47.6%** of the 125 TFLOP/s fp16 peak |
| decode | **51.1 t/s** with `NINFER_TERNARY_ROTATE_SPLIT=16`, **48.7** with no environment variables, 44.6 before `327f4b10` + `1df45be1`. The `41.6` used throughout the Decode section is the dedicated warp-per-row kernel, which is no longer the default route. **54.2-54.3** after the warp-staged decode code plane (`e0dd854d`, `e974ec60`; rounds 4-5 of `docs/decode-round-2026-10-01.md`), which also takes the MTP K = 1 verify from 61.9 to 63.9 t/s. **55.2-55.3** after the fused residual epilogue (round 9 of the same document), which deletes the decode step's last 132 `residual_add` launches; on `real_code` 52.3 -> 53.2 and on `prose4k` 50.6 -> 51.5, all with byte-identical output |
| context-lookup K=7 (repeated text, verify at T = 8, NOT 16 -- see below) | **254.2** t/s `lookup10`, **265.2** `lookup16` with the QPN wide-verify route (default); **123.7 / 126.7** with `NINFER_TERNARY_QPN=0`. The `+8.2%` over the previous 235.1/249.8 is round 12's `kTiles = 1` NACC fix, which turned out to govern the lookup verify too: `T = K + 1 = 8` gives `tiles = ceil(8/8) = 1`, so the lookup verify has always been on the `kTiles = 1` arm and the unswept `NACC = 4` was holding it back. Only the T >= 6 verify band changes -- the decode and MTP bands are md5-identical |
| causal scoring (`ninfer-perplexity`) | 1091 tok/s, was 106.3 |
| startup | 8.7 s (weights 6.70 GiB); 10.1-10.2 s re-measured as `engine ready` at `--max-context 8192` on this build |

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
| `NINFER_TERNARY_CUTLASS_MIN_T` | token threshold for the arm, default **256**. Below it CUTLASS is a GEMV and measures 0.3 TFLOP/s, so this is what keeps decode and the MTP verify off the route. Re-measured on the current build: **128 and 512 switch the arm cleanly** (see "What tensor cores are worth here"), but **1 no longer reaches a walk -- it fails startup**; see Wrong values below |
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
| `NINFER_TERNARY_FUSED_RESIDUAL` | **default on** (unset or any value except `0`). `0` restores the composed route (ternary GEMM into a scratch, then `ops::residual_add`) for the T = 1..5 decode/verify band. Bit-identical either way — it is the round-9 A/B arm and the rollback switch, not a numerical knob |
| `NINFER_TERNARY_QPN_MIN_T` | lower edge of the QPN verify band, default **6**. It exists to re-run the crossover on a later build, not to tune. Round 12 moved the crossover from T ~ 4.3 to T ~ 3.5 (after the `kTiles = 1` NACC fix: `=4` costs the K = 3 round **-8.9%**, `=3` ties at K = 2, `=2` costs the K = 1 round **+18.7%**), **and the edge stays at 6 anyway**: the T = 4 window it would move costs **+0.097% PPL** against SIMT -- four times the fp16 prefill margin this tree accepted (+0.0245%) -- and it only makes MTP K = 3/4 less bad, since K = 1 (15.54 ms/token) stays the best setting and its round does not move. `=4` is a one-line escape hatch for a workload whose acceptance curve makes K >= 3 worth running |
| `NINFER_TERNARY_QPN_MINBLOCKS` | `__launch_bounds__`'s second argument for the QPN kernel, default off (the built-in value is 4 at kTiles = 1, i.e. 50% occupancy). **Measured and rejected**: `6` costs the T = 1 step -16.6% and `8` costs it **-76%**. The kernel is not occupancy-starved despite sitting at 50% -- it wants its registers, like the rest of this family. This closes the QPN small-T item: NACC's +21.8% was all it had. Kept as the record |
| `NINFER_TERNARY_QPN_NACC` | accumulators per 16-k unit. Default **1 at kTiles = 1**, 2 at kTiles = 2, 1 above. Round 12 found the kTiles = 1 default was 4 and had never been swept: **1 and 2 both give +21.8%** on the T = 1 step and **+19.7%** on the MTP K = 1 verify over 4, and they also move the lookup verify, which turns out to be at `tiles = 1` (T = K+1 = 8): **lookup10 235.1 -> 254.2, lookup16 249.8 -> 265.2**. Continuous check at forward width 7: **+0.00055% PPL** against NACC = 4, and the override now reaches EVERY kTiles (it used to be read only at kTiles = 2, i.e. the knob was silently inert on T <= 8 and on T >= 17) |
| `NINFER_TERNARY_TILE_STAGE` | default on. `0` disables the warp-staged T = 1..2 decode GEMV (costs 3.9%) and also takes the fused residual epilogue off that band, since the fused arm is the staged kernel |
| `NINFER_TERNARY_STAGE_DEPTH` | staged kernel prefetch distance, default 1. `2` measured -2.7%, and also falls back to the composed route |
| `NINFER_TERNARY_STAGE_ACT` | **measured and rejected**: `1` stages the block's activation span in shared memory once instead of reading it per warp. Bit-identical (md5 IDENTICAL, text identical on 12 pairs) but **-10.0% at T = 1** and -9.6% at the MTP K = 1 verify, with the T >= 3 arms untouched. It proves the eight duplicate reads were already L1 hits, i.e. the activation side is L1/latency-bound rather than L2 or DRAM traffic. Kept as the record; default off |
| `NINFER_TERNARY_PROBE_SKIP_RESIDUAL` | do not launch `ops::residual_add` at all. **Numerically wrong by design**; this is how the fused epilogue's ceiling was priced (+1.9% on the T = 1 step) |

### Wrong values, and what they cost

| setting | side effect |
|---|---|
| `--prefill-chunk 1024` (the old default) | prefill falls 1190 -> 996 and causal scoring 1091 -> 106 tok/s. See the section below; this is the single most expensive wrong value in the tree |
| `NINFER_TERNARY_CUTLASS_MIN_T=1` | on the current build this **does not start at all**: `error: startup failed | preparing CUDA graphs | std::bad_alloc` -- admitting CUTLASS at every token count makes the per-shape workspace explode across the graph set. On the older build it reached a walk instead: decode collapsed because the CUTLASS GEMM at M=1 measures 0.3 TFLOP/s. Same verdict, different failure mode; re-measured 2026-10-01 |
| `NINFER_TERNARY_PREFILL=wide` | **throws `std::invalid_argument`**. Deliberate: it names the author's bf16 weight-resident kernel, which does not exist here, and silently aliasing it to a SIMT route would make an A/B against the author's numbers look like a regression when it is really a missing route |
| `NINFER_TERNARY_MMA=bogus` | throws `std::invalid_argument`, and because it is thrown from a `static` initialiser the process dies with rc=139 (`terminate called after throwing...`). Documented behaviour, verified |
| `NINFER_TERNARY_HADAMARD=0` | forward pass is fast and the output is garbage. Never a default |
| `NINFER_TERNARY_FP16_ACT` left at its default with a PTQ1_0 or mixed artifact | **startup fails**: `error: ternary gemv: a non-bf16 activation reached the bf16 reference route`. The rotation chose the fp16 container from the token count alone, and every kernel that serves a non-PQ2 tensor is bf16-only. Fixed by gating the container on `weight.qtype == QType::PQ2_0_G128`; PQ2_0 is unaffected because that condition was already true for it. `NINFER_TERNARY_FP16_ACT=0` was the only workaround |
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
> The 2026-10-01 rounds -- the PQ2 tensor-core (QPN) wide-verify route and the T = 1 decode
> step priced component by component, with the five decode levers that were tried and
> refuted -- are consolidated in `docs/decode-round-2026-10-01.md`. The decode material below
> predates it and is kept because it records this port's earlier budget measurements.


T=1 warp-per-row GEMV, **41.6 t/s** (3 runs identical), against llama.cpp's 33.7 on the same prompt.
Unaffected by every prefill change; verified again at chunk 4096.

**Superseded route.** Since the half2 group-dot landed, T=1 no longer runs this kernel: it goes
through `ternary_pq2_gemv_tile_kernel` at `kT=1`. Every T=1 number below (and the knob table's T=1
rows) belongs to the dedicated warp-per-row kernel and still describes it correctly -- it is
simply not the default any more. Re-measured on the current build through the tile route: **44.6
t/s**, then **46.6** once the rotation ran four warps per D1024 (`327f4b10`), then **51.1** once
the launcher's hardcoded depth 4 was fixed to follow `NINFER_TERNARY_GEMV_TILE_UNROLL`
(`1df45be1`); **48.7** with no environment variables at all. The "T=1 unroll is flat" reading was
an artefact of that hardcode -- swept after the fix: 1 -> 32.8, 4 -> 46.5, **8 -> 51.1**, 16 ->
35.2 t/s, so depth 8 is the new default for `tokens == 1`.

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

**Re-measured: rotation is 8.6% of the step, not 13%, and nvprof is why the two disagree.** Three
wall-clock arms, two repetitions each, identical prompt (300 generated tokens, int8 KV): full
rotation **44.6 t/s** = 22.42 ms, `NINFER_TERNARY_ROTATE_PROBE=copy` (same launch, same
read/write, no transform) **47.0** = 21.28 ms, `NINFER_TERNARY_HADAMARD=0` (no kernel captured)
**47.4** = 21.10 ms. So the transform is **1.15 ms** and the separate launch plus round trip is
**~0.75 ms** -- **1.9 ms together, 8.6% of a 21.5 ms step**, against the 3.45 ms in the nvprof
budget table and the 4.0 ms the ceiling table below was priced with. The same trace reports ~15 us
per rotation call where the wall clock says ~7.5 us: **nvprof inflates these tiny kernels by about
2x**, so read the budget table as shares, not absolutes. The clean way to separate the arms is
`NINFER_TERNARY_FP16_ACT=0`, which pins every arm to the same bf16 GEMV path -- those three
measure 43.5 / 45.8 / 47.5 and price the fp16 half2 GEMV at 0.55 ms of the step as well.

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

### The fused residual epilogue (round 9): 54.2 -> 55.3 t/s, bit-identical

The decode step's largest remaining tail item was the 132 `residual_add` launches it issued, one per
folded linear. `ops::linear_add` **looked** fused and was not: its ternary branch composed the shared
`ternary_dispatch` into a scratch tensor and then called `ops::residual_add`, because no ternary
kernel had a residual epilogue. The GEMV family now has one (`gemv_store<kAddResidual>` in
`ternary_rowsplit_gemv.cuh`), and the decode/verify band uses it, so a T = 1 step issues **zero**
`residual_add` launches; the only ones left in a decode trace are the prefill's.

It is **bit-identical**, and by construction: `residual_add` computes
`__float2bfloat16_rn(bf16(y) + bf16(x))` on a projection the GEMV has already rounded to bf16, so the
composed route rounds twice. The fused epilogue reproduces exactly those two roundings. Adding the
fp32 accumulator instead would be more accurate and would move greedy ids — that is the version
round 8 priced, and the reason it estimated a re-recorded baseline plus a perplexity run. Doing the
double rounding instead is what made the change cheap: the 96-token greedy md5 against
`/root/wt/base.out` is IDENTICAL with the arm on and off, and the output text is byte-identical on
all eight fixtures (`real_task`, `real_code`, `prose4k`, `lookup10`, `lookup16`).

| arm, same batch, interleaved | T = 1 | MTP K = 1 | MTP K = 2 | MTP K = 3 | `real_code` | `prose4k` | lookup10 | lookup16 | prefill `real_code` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| composed (`NINFER_TERNARY_FUSED_RESIDUAL=0`) | 54.2 | 63.9 | 54.5 | 49.3 | 52.3 | 50.6 | 237.3 | 250.0 | 1.16k |
| **fused (default)** | **55.3** | **64.4** | **54.9** | **49.6** | **53.2** | **51.5** | 237.7 | 249.8 | 1.16k |
| delta | **+2.0%** | +0.8% | +0.6% | +0.5% | **+1.7%** | **+1.7%** | +0.1% | -0.1% | 0 |

The lookup and prefill columns are the control: the fused band is T = 1..5, so the QPN verify band
(6..32) and prefill are not touched and sit inside their own noise. Acceptance length is identical on
every MTP arm (1.65 / 1.85 / 2.03 / 12.11 / 12.70 tok/round), which is the discrete confirmation of
the bit-identity. `NINFER_TERNARY_FUSED_RESIDUAL=0` is both the A/B arm and the rollback.

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
| GEMV at its measured weight-read floor + 5.6 ms of other work (rotation re-priced at 1.9, see below) | 13.9 | 72 |
| the same, with the rotation's launch+round-trip (0.75 ms of the measured 1.9) folded into its producer | 13.2 | 76 |
| absolute weight-stream floor (no other kernel costs at all) | 8.3 | 120 |
| demonstrated with amortisation (16-token verify) | 7.4 | 105-135 |

**Re-priced on the current build (2026-10-01).** The rows above were computed against the
41.6/22.9 baseline of the dedicated T=1 kernel and the rotation was priced from nvprof at 4.0 ms;
the wall-clock A/B puts it at 1.9 ms, so the rotation-dependent rows now read 13.9 ms / 72 t/s for
the weight-read floor and **13.2 ms / 76 t/s** once only the separate launch is folded away. On
the shipped route (tile kernel, depth 8, four-warp rotation) the step is 21.5 ms = **51.1 t/s**,
and the probes in the section below put the code stream (load + decode) at 9.1 ms of it.

**The honest ceiling for a non-speculative single-stream decode on this card is ~60-65 t/s**, and
~76 t/s only if the folded-basis rotation's launch is folded into its producer. Both are capped by the fact
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

### What the v3 port actually costs, measured against `fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer`

A local v3 reference tree exists at `/root/ninfer-all` (it rejects v2 outright and ships
`tools/upgrade_ninfer_v2_to_v3.py`). Measured against it and against a downloaded v3 artifact
(`swift15_iq2xs_mtp.ninfer`, 9,420,962,560 B):

**Framing** (`src/artifact/framing.h` upstream): entry magic `NINFER\0\x03`, `kHeaderBytes = 32`
(8 magic + u64 json_len + **16-byte artifact id**), JSON at 32, payload at
`align_up(32 + json_len, 4096)`. Continuation files use `NINPRT\0\x03`. The artifact here is
single-file (`files: [{"path": null, "payload_bytes": 9420610304}]`), so the multi-file path is not
exercised.

**The schema differs in one way that reaches past the reader.** In v2 the object `name` IS the
parameter path (`text/layers/0/gdn/query_key`), so `binder.cpp` matches on names. In v3 objects carry
opaque ids (`weight/000000`) and the semantic names live in a separate `bindings` map, which
additionally allows **one parameter to be assembled from several objects by byte range**:

```json
"text/layers/0/gdn/query": {"parts": [{"object": "weight/000000", "range": [0, 10485760]}]}
```

This artifact has 1422 bindings against 1126 v2-style objects (752 names in common), so the port is
three pieces, not one: (1) framing + `id`-instead-of-`name` + snake_case format/layout/encoding
spellings + the extra top-level keys, (2) a bindings layer with `parts`/`range` assembly, which
`materializer.cpp` must then honour, (3) the model config moving from `identity` to
`components.<name>.config`.

**And it is necessary but not sufficient for the artifact that prompted it.** The IQ2_XS artifact's
tensor formats, counted from its own JSON:

| format | tensors | format | tensors |
|---|---:|---|---:|
| `bf16` | 582 | `gguf_iq2_xxs` | 73 |
| `fp32` | 96 | `gguf_iq2_s` | 58 |
| `q4_g64_fp16` | 54 | `gguf_q2_k` | 58 |
| `q5_g64_fp16` | 54 | `gguf_iq2_xs` | 49 |
| `gguf_iq3_s` | 46 | `gguf_iq3_xxs` | 36 |
| `gguf_iq1_s` | 32 | `gguf_iq1_m` | 28 |
| `gguf_iq4_xs` | 12 | `gguf_q6_k` | 5 |
| `gguf_q4_k` | 4 | `q8_g32_fp16` | 2 |
| `q6_g64_fp16` | 1 | `int32` | 2 |

**401 of 1192 tensors are `gguf_blocks_v1`**, and this tree has no GGUF support at all (the only
matches for "gguf" in `src/` and `include/` are two comments about llama.cpp). The upstream tree that
does have those kernels excludes Volta by construction: its CMake requires
`CMAKE_CUDA_ARCHITECTURES` to match `^(80|86|89|120a)$` and `find src -iname '*volta*'` returns zero
files. So running this artifact on the V100 would additionally need eleven GGUF dequant kernel
families ported to sm_70 — IQ2_XS alone is a 512-entry codebook with 8-bit indices, and IQ1_S/IQ1_M/
IQ2_S/IQ2_XXS/IQ3_S/IQ3_XXS/Q2_K/Q4_K/Q6_K/IQ4_XS each have their own block structure. That is a
project of the same order as the whole ternary port, which covered one format.

**What the v3 port does buy**: the v3 ternary artifacts (`t2_g128_fp16`, the sanbanfu spliced one) become
reachable, needing only a T2 dictionary addition on top of the v3 reader — this tree's PQ2 kernels have
the identical container geometry and differ only in the 2-bit code dictionary.

### Where the v3 implementation actually lives: two sibling V100 trees, not the public upstream

`Neroued/ninfer` (the root upstream) is **sm_120a only** — its `CMakeLists.txt` rejects anything but
`120a` and it has no Volta sources at all — so it cannot be the reference for a v3 port to this card.
The v3 work for sm_70 exists in two *sibling* V100 trees, both of which this tree should be read
against:

| tree | arch | reads | formats | notes |
|---|---|---|---|---|
| `/root/duo` (local checkout; the `duo` remote was `fetch /root/duo`) | `70\|86\|89` | v2 + v3 | nvfp4, fp8, **`GGML_K`** (`weights_id == "gguf-q4-k-m"`) | v3 via `Qwen38Nvfp4V3Adapter`, a v3->v2 projection |
| `ww485000/ninfer-windows-v100`, branch `v100-sm70` | `sm_70` only | **v2 + v3** | the nine v2 formats only (nvfp4 line) | README: "Official `.ninfer` v2 and v3 containers are supported; arbitrary GGUF/Safetensors files are not", and it reports a **verified v3 run on a V100** (Qwen3.8-27B NVFP4 v3, pp2048 1,135.88 tok/s, pp2048+tg256 228.14) |

Both are the same lineage as this tree — the artifact-layer file list is *identical*
(`binder`, `materializer`, `reader`, `storage_layouts`, `typed_binding`), not the refactored 23-file
layer of the sm_120a tree — so their v3 code is portable. Against the shared merge base `b37d0dd3`,
this tree has changed the artifact layer by only **+31 lines across 6 files** (adding
`PTQ1_0_G128`/`PQ2_0_G128` and their layouts), while `duo` added ~370 lines of v3 adapter inside
`src/artifact/reader.cpp` (`class Qwen38Nvfp4V3Adapter`, ~lines 281-633, plus `V3CompatibilityDirectory`
and the version dispatch in `Impl`). `ww485000`'s `reader.cpp` is 43,026 bytes against this tree's
16,483, and is the larger and better-qualified of the two references.

**Neither documents v3.** Both trees' `docs/maintainer/artifact-container.md` is still titled "NInfer
Artifact Container Version 2" and specifies only the v2 framing; v3 is a code-only compatibility
adapter in both, not a published contract.

**And neither has the IQ family.** `duo` has exactly one GGUF format, `GGML_K`, for
`weights_id == "gguf-q4-k-m"`; a grep for `iq1|iq2|iq3|iq4` across `duo/src` and `duo/include`
returns **zero**. `ww485000` declares only the nine v2 formats. So the eleven `gguf_*` formats in the
fyb423 IQ2_XS artifact have no kernel in any V100 tree.

**Consequence for this card.** Reading v3 is worth doing (it aligns this tree with both siblings and
is a bounded port), but no currently available v3 artifact is runnable here: the v3 NVFP4 artifact is
22.09 GiB against 16 GB of VRAM (`ww485000`'s own README says "V100 16GB is not a qualified
Qwen3.8-27B NVFP4 target"), the v3 ternary repository returns `record not found` on ModelScope, and
the IQ2_XS artifact needs the eleven missing IQ kernels. The v3 port is infrastructure, not a new
runnable model.

### Running the IQ2_XS artifact: what was built, and the one structural gap that is left

Four commits so far, each verified:

| commit | what | verification |
|---|---|---|
| `da02a7d7` | reads v3 containers (32-byte header, `id`->`name`, snake_case vocabulary, the logical alias layer, GGUF formats in the container) | the artifact now parses completely; v2 bit-identical (`REGRESSION_OK_md5_identical`) |
| `ebdbf57f` | vendors llama.cpp's ggml-cuda (`third_party/ggml-quants`) and compiles the GGUF block kernels for **sm_70** | 0 errors; `libninfer_ggml_quants.a` 81 MB with `iq2_xs` symbols; v2 bit-identical |
| `adcc04af` | routes GGUF qtypes in `linear`, `linear_swiglu`, `linear_add` to the vendored kernels | 0 errors; v2 bit-identical |

**The vendoring is the reason this is tractable at all.** The upstream GGUF path is not hand-written:
it vendors llama.cpp's ggml-cuda behind a thin bridge, and ggml-cuda already carries sm_70 paths
(`GGML_CUDA_CC_VOLTA 700`, `#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA` in `mmq.cuh`). So Volta needed
no new kernel work -- 18,845 vendored lines instead of eleven kernel families written from scratch.
Its block table is ggml's own and was validated against the artifact before any C++ was written: all
401 gguf tensors satisfy `rows * (K / block_elements) * block_bytes` exactly.

**What the artifact actually contains** (from its own `bindings`), which is what the remaining binder
work has to match:

| role | format |
|---|---|
| `text/token_embedding` | `gguf_iq1_m` (248320, 5120) |
| `text/output_head` | `gguf_iq4_xs` (248320, 5120) |
| `proposal/head` | `gguf_iq4_xs` (131072, 5120) |
| MLP gate / up / down | `gguf_iq1_s` / `gguf_iq1_m` / `gguf_iq2_xxs` |
| attention query / key / gate / value | `gguf_iq2_xxs` / `gguf_iq2_xxs` / `gguf_iq2_xxs` / `gguf_iq2_s` |
| GDN query / key / value / z / output | `gguf_iq3_s` / `gguf_iq3_s` / `gguf_iq3_s` / `gguf_iq3_s` / `gguf_iq4_xs` |
| MTP layer (all projections) | `gguf_q6_k` |
| every norm, GDN convolution, GDN a/b projection | `bf16` |

So `text/layers/N/*` is 960 bindings = 64 layers x 15 roles, the same layer structure as the v2
contract, and the proposal head exists -- it is named `proposal/head` rather than `text/draft_head`,
which the v3 logical layer's `fork_rename` already maps.

**THE GAP. The fork's contract is FUSED; this artifact stores every projection SEPARATELY.** The fork
binds one `attention/query_key_gate_value` weight and its op is
`attn_input_proj(x, query_key_gate_value_weight, ...)`; the artifact has four separate objects, and
they do not even share a format (`iq2_xxs` query/key/gate against `iq2_s` value), so they cannot be
concatenated into one fused tensor either. The same holds for `query_key_value_z`, `a_b_projection`
and `gate_up`.

The upstream tree has already solved exactly this, and the mechanism is small: it adds a
`GgufProjectionWeights` overload carrying a list of parts (`{Weight, output, row}`) and projects them
all in ONE call, because `gguf_project` already takes a `std::span<const GgufProduct>`:

```cpp
struct GgufProjectionWeights { struct Part { Weight weight; int output; int row; }; std::vector<Part> parts; };
// ... build one GgufProduct per part, then:
detail::gguf_project(x, products, workspace, stream);
```

`gguf_linear.h` in this tree already declares that span-taking entry point, so the remaining work is
plumbing rather than new kernels:

1. `WeightsProfile::Qwen38GgufMixed` in `qwen3_6_27b/export/.../package.h`, accepted by
   `Package::resolve_weights` for `qwen3.8-27b` + `gguf-mixed`;
2. a `bind_gguf_text_layers` that binds the artifact's real inventory -- per-projection objects,
   `endpoint_format` resolved from the DECLARED format per role rather than one profile-wide format
   (the embedding is `iq1_m` while the head is `iq4_xs`), and `Weight::input_columns` populated for
   the GDN output projection whose stored columns are permuted;
3. `GgufProjectionWeights` overloads on `attn_input_proj` / `gdn_input_proj` / `linear_swiglu`, and
   the `BindingPlan`/program plumbing to call them when the profile is GGUF;
4. cases for the new profile in `variant.cpp`'s policy switches (28 sites).

That is a multi-file, multi-hour change and it is the next unit of work; it is not started here
rather than half-landed, because a partially plumbed binding cannot load and would leave the tree
worse than the four verified commits above.

### The GGUF load path, and the one leaf it still needs

Landed since (each verified, v2 bit-identical throughout):

| commit | what |
|---|---|
| `bd362e0e` | the GGUF projection parts: `GgufProjectionPart`/`GgufProjectionWeights`, the `attn_input_proj` parts overload, the payload variants, the GGUF MLP triple |
| `abbb7f88` | the v3 row-range reader (`ObjectSlice`), the GGUF binder, the GGUF materializer, the GDN parts overload, the variant/profile plumbing, the embedding route, the chat-template fallback |

`ninfer /models/swift15_iq2xs_mtp.ninfer` now reports

```
loading weights | 7.83 GiB
weights ready | 7.83 GiB | 9.3s | 863.6 MiB/s
error: startup failed | preparing CUDA graphs | 2.18 ms
error: gdn_input_projection_snapshot: the GGUF parts projection has no conv snapshot yet
```

so the container, the binder, the materializer and the frontend are all through, and the remaining
item is one leaf.

**What the artifact actually is, which the binder had to match.** Its v3 `parts` are ROW RANGES of
one stored object, and the fusion varies per layer:

| role | stored as |
|---|---|
| attention query, key | two row ranges of one 7168-row object on 2 of the 16 full layers, separate objects on the other 14 |
| attention gate, value | always separate objects, and their formats differ (`iq2_xxs` against `iq2_s`) |
| GDN q, k, v | one 10240-row object, with z its own object on 37 of the 48 GDN layers |
| MLP gate, up | one 34816-row object on some layers (so the fused parent exists), separate objects with different formats on the rest |
| MTP layer | fuses cleanly: its four attention projections are ranges of one 14336-row object and its gate/up pair of one 34816-row object |
| vision | `q4_g64_fp16`/`q5_g64_fp16`/bf16 -- already supported |

Hence the parts route for attention and GDN, a per-layer conditional for MLP, and `bind_gguf_object`
to bind a range's object at its own declared name and shape.

**What is left.** `gdn_input_projection_snapshot` and `..._record` have no GGUF form. The composition
is known -- project the parts into one `[16384, T]` plane and hand it to
`gdn_projected_conv_snapshot_launch`, which is exactly how the NVFP4 batched path composes it -- but
the conv snapshot's row and state semantics have to be exact, and a wrong version corrupts the GDN
state *silently* rather than failing. So both leaves refuse with a named error rather than
approximate, and that is the next unit of work. After it, the decode path's GGUF GEMV still has to be
verified end to end against the CLI's own numbers.

### It runs now -- and the numbers are still wrong

Both of those landed. `ninfer /models/swift15_iq2xs_mtp.ninfer` generates: **prefill 152.8 tok/s,
decode 50.5 tok/s** on the V100, with 7.83 GiB resident. The conv leaves compose as the NVFP4 batched
path does (project q|k|v into one `[10240, W, B]` plane and z into its own output, then hand the plane
to `gdn_projected_conv_*_launch`).

**The GDN output permutation is what stopped it being NaN.** That matrix's stored columns are a
permutation of its input's, and the artifact ships the INT32 [6144] gather ONCE as `auxiliary/000000`
for all 48 GDN layers:

| | result |
|---|---|
| without the permutation | `error: causal scoring returned a non-finite logprob` |
| with it | `mean NLL 10.255982, PPL 28452.225828` over 8675 scored tokens |

so it is both required and applied in the right direction. The reader now keeps objects a `uses`
auxiliary names reachable (the remap dropped `auxiliary/000000` before any binder could see it).

**But the path is still numerically wrong**, and the failure mode is worth recording: it does not
crash and does not obviously garble -- it emits fluent-looking fragments
(`We ... includeres incluiderederederedered...`) at a PPL that is merely too high. Only the number
catches it. Same text, 8675 scored tokens, `--stride 4`:

| artifact | context 8 (forward width 7) | context 16 (width 15) |
|---|---:|---:|
| this tree's ternary artifact (working) | **4.836** | **4.015** |
| the IQ2_XS artifact | 10.256 | 10.480 |

**Verified negatives, so the next session does not repeat them:**

* block geometry: every one of the fifteen block types' `static_assert(sizeof(block_*))` in the
  vendored `ggml-common.h` equals the geometry table the bridge and the reader use (IQ1_M is 56, not
  58 -- it carries no `d` field);
* row slices: the artifact's own declared byte counts match `rows * (K / block_elements) * block_bytes`
  for all 401 gguf tensors, including every object a slice is taken from;
* the permutation's direction: the table differs from the identity in 95.8% of entries, so the two
  readings are genuinely different, and rewriting the artifact's auxiliary with its INVERSE returns
  to non-finite -- the shipped direction is the correct one;
* kernel path: PPL is wrong by a similar factor at width 7 (vector kernel only) and width 15 (integer
  matrix kernel), so this is not a vector-versus-matrix selection problem;
* the vendored `dispatch_dequantize` covers all fifteen types including IQ1_M and IQ4_XS.

**What that leaves**: the dequantized VALUES themselves. Every structural check above passes, so the
next step is a reference implementation -- dequantize one GGUF row in Python the way llama.cpp does,
read back what the bridge produces for the same bytes, and compare. That needs a small device
readback harness, which is why it is a fresh unit of work rather than a fix.

### The chain is verified; the numbers still are not

That harness was built, and it says the GGUF path is correct at every link:

| what | how | result |
|---|---|---|
| dequant | Python transcription of llama.cpp's `dequantize_iq2_xs` (grid/sign/mask tables parsed straight out of the vendored `ggml-common.h`) against `dequantize_rows` on the same 1480 bytes | worst absolute difference **1.14e-4** on values ~0.04 -- **0.3%, one bf16 step** |
| the whole product | one IQ2_XS tensor (10240 x 5120) through `quantize_vector_activation` + `vector_product`, all 10240 rows, against reference dequant + dot | 8479 rows have \|expected\| > 0.01; worst absolute difference **7.86e-4** on values ~0.1; worst relative difference among meaningful rows **5.6%**, i.e. one bf16 output step |
| every format | dequantize a row of all fifteen types present in iq3xxs | **zero non-finite**, ranges 0.04-0.16 |
| every slice | all 523 parameters' row slices against the shapes in the artifact's own `methods` record | **0 mismatches, both artifacts** |
| no converter transform | the `methods` section | every GGUF tensor is `import_encoded` |
| the architecture | `TextConfig` against `components.text.config` | field for field identical |
| `query` vs `gate` bytes | the `methods` list both as `blk.N.attn_q.weight`, which would mean duplication | sha256 differ -- the labels name the source tensor, not the slice |
| the one FUSED parent this port uses | `mlp/gate_up` row ranges in both artifacts | `gate` is the first half in both, matching the swiglu's assumption |

So the path is not where the error is. Two things narrow it further:

**iq3xxs generates but does not score.** Its CLI run emits 24 tokens at 46.3 tok/s -- a finite forward
pass -- while `ninfer-perplexity` reports a non-finite logprob. That combination means the hidden
state does not go NaN but does grow until the logits overflow: argmax still picks a finite maximum,
the log-softmax does not.

**Both artifacts are wrong, and they share everything the ternary artifact does not exercise.** The
converter's record proves the binding; the harness proves the arithmetic. What is left is code that
runs for both GGUF artifacts and for neither ternary one -- and the newest such code is the GDN conv
snapshot/record composition added when the model first ran, which is what every prefill goes through
(the first generated token comes from it). That is the next thing to check, and it is checkable: the
composition can be compared against the split path's own `gdn_input_proj_conv_snapshot` output for
the same inputs, which needs a fixture rather than a reference implementation.

### The gate is the TEXT, not the PPL

That comparison was the wrong instrument. The tree's ternary artifact and the Swift-1.5 artifact are
**different models**: their BF16/FP32 roles differ in 442 of 537 cases, down to bf16 rounding in
`a_log` and `dt_bias`. So 4.836 against 10.256 compares two models, not two paths, and it cannot say
which is wrong. (The v2 ternary artifact is `qwen3.8-27b`/`groupwise-int`; the v3 one is
`swift-1.5-qwen3.8-27b-gsq-rco-iq2xs`.)

The gate that does work is the generated text, same harness, same prompt:

| artifact | output |
|---|---|
| ternary (working) | `用户要求用 C++17 写一个线程安全的环形缓冲区，支持单生产者单消费者（SPSC），给出完整实现、关键设计说明和` |
| IQ2_XS | `The user is writing/popularizing/populante/populante/poptantepptantepptantepptant` |

and on a one-line prompt the GGUF artifact is telling: it opens coherently --
`The user wants to know about ...` -- and then collapses into `...myselfakashifujibuzujireireireire`.
Coherent first, then degenerate, means the error ACCUMULATES rather than being present in the first
forward pass.

**Exonerated since**: the KV cache (int8 and bf16 produce byte-identical text, so the cache is not
what accumulates), the dequantizer and the whole product chain (measured against a reference), the
binding (all 523 slices match the converter's own record in both artifacts), the block geometry, the
`gate_up` row order, and the converter (no transform). What accumulates is the GDN's recurrent state,
and the newest code on that path is this port's conv snapshot/record composition.

### The port is fundamentally right; something accumulates

The conv composition survived its own audit, and the artifact shows the port's core is correct:

| check | result |
|---|---|
| `conv_record` shape | the working ternary path composes with the same `compose_record` and the same `gdn_input_proj(x, weight, record_flat, z_flat)`, so `conv_record` is `kChannels = 2048+2048+6144 = 10240` rows -- what this port reshapes to |
| workspace sufficiency | the shape-keyed capacity query reserves the `[channels, projected_width]` leaf the compose route asks for, and `DeviceArena::alloc` throws rather than silently overrunning -- the runs do not throw, so the arena is not the problem |
| `Tensor::view`/`reshape` | requires contiguity and preserves the `data` pointer including a slice's offset, so the rank-3-to-rank-2 reshapes in the conv leaves cannot silently mis-address |
| basic capability | on `Complete this: 2 + 2 =` the GGUF artifact opens `User sent instruction requesting completion likely ...` and on a one-line question `The user wants to know about ...` -- both coherent |

That last row matters most: **the model reasons about the prompt correctly and then collapses**, so the
weights, the binding and the kernels are fundamentally right and what is wrong is state that
accumulates over decode steps. `--prefill-chunk` cannot help pin it down (it must be a multiple of
128, so the prefill cannot be forced onto the record path), and the KV dtype does not change the text
at all.

### Prefill and decode take different paths -- and only decode is wrong

The runtime settles this. A wide prefill never touches the conv composition at all:

```cpp
    } else {                                            // T > 1
        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        ops::causal_conv1d_silu_split(qkv, *w.conv1d, conv_state_in, conv_state_out, qc, kc, vc, s);
    }
```

so the prompt goes through `gdn_input_projection` plus a separate conv, while T == 1 goes through
`gdn_input_projection_snapshot`. **The prefill's output is coherent, so the projection op is right in
situ**; the collapse is on the decode side.

And this port's snapshot leaf turns out to be line-for-line the composition the WORKING ternary path
uses -- `dispatch_single_parent_snapshot`'s final fallback is

```cpp
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states, ...);
```

which is exactly what this port does with the parts overload in place of the fused one.

**Three more negatives from this round:**

* the vector product WITH the artifact's `input_columns` permutation is correct -- measured against a
  reference IQ4_XS dequantization with the same gather, relative error p50 0.32%, p90 1.1%, p99 2.4%,
  max 5.3%, and the largest absolute difference (6.8e-3) sits on the largest product (|expected| =
  1.155, i.e. 0.59%);
* greedy decoding is deterministic -- three runs of the same prompt give byte-identical output, so
  nothing is reading unwritten memory;
* `Tensor::view`/`reshape` and the workspace are both safe (above).

**The signature is a repetition loop.** Coherent for about eight tokens, then
`... likely meaning complete identity requesting requesting requesting ... Strec Strec Strec` -- the
state collapses rather than drifting. That is the shape of a broken recurrent state or positional
handling, not of wrong weights.










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

## The PTQ1_0 tier: 18% fewer bytes, and 38% slower decode

The same release ships a second container packed from the same trit set:
`bonsai2_27b_swift_ptq1.ninfer`, 7,047,407,628 B, `PTQ1_0_G128`, sha256 `cc9e8907...`, from the
ModelScope repo `fyb423/Swift-Bonsai-2-27B-NInfer`. It is worth measuring because it is strictly
smaller -- 28 bytes per 128-weight group against PQ2_0's 34 -- and the whole question is whether that
17.6% converts into speed. It does not, and the reason is a measurement rather than an argument.

Container level, read off the two manifests (both v2, both `model_id qwen3.8-27b`):

| | PQ2_0_G128 | PTQ1_0_G128 |
|---|---:|---:|
| ternary tensors | 322 | 322 |
| ternary bytes | 7,137,280,000 | 5,877,760,000 |
| ratio | 1 | exactly 28/34 = 0.8235 |
| all other tensors | BF16 582 / Q4G64 55 / Q5G64 54 / W8G32 7 / FP32 97 / I32 2 | byte-identical |
| CLI weight footprint | 6.70 GiB | 5.52 GiB |

PTQ1_0 cannot reuse PQ2_0's decode: four two-bit codes live in one byte and come out through two
magic numbers and two `HADD2`, about 7 ALU ops per four weights, whereas base-3 packing spreads a
group's 128 weights over five exponent phases. The port that was built and measured decoded through a
per-block shared LUT holding the final fp32 weight, laid out `[byte][exponent]` so that the five lanes
sharing a byte land on five distinct banks. Per four weights that is one 32-bit code load, four byte
extracts, four LUT-index IMADs, four `LDS.32` and four `FFMA` -- about 16 decode ops against PQ2_0's
7, in exchange for 18% fewer bytes.

Measured in situ: same prompt, greedy, `--no-thinking`, 128 new tokens, `--max-context 16384`.

| arm | decode | prefill | registers | CTAs/SM |
|---|---:|---:|---:|---:|
| PQ2_0 tile kernel, kT=1, fp16 half2 (shipped) | **44.6** | 90.1 | 32 | 8 |
| PTQ1_0 reference SIMT kernel | 8.1 | 23.2 | -- | -- |
| PTQ1_0 tile kernel, bf16, unpinned | **27.8** | 28.5 | 39 | 6 |
| PTQ1_0 tile kernel, `kMinBlocks=6` | 27.4 | 28.5 | <=42 | 6 |
| PTQ1_0 tile kernel, `kMinBlocks=8` | 27.2 | -- | <=32 | 8 |

Greedy token ids are **identical** between the PQ2_0 and the PTQ1_0 arm (`16 220 17 220 ... 19 21` on
both), so the decode is correct; the port is a 3.4x improvement over PTQ1_0's own reference kernel.

The `kMinBlocks` sweep is the decisive measurement, not a knob to tune. ptxas gives the PTQ1_0 kernel
39 registers against PQ2_0's 32, i.e. 48 resident warps per SM against 64, a quarter of the card's
latency hiding thrown away. Capping registers at 32 restores the full 64 warps and measures **slower**
(27.8 -> 27.2). Occupancy is therefore not what binds: the kernel is limited by instruction issue, so
an 18% byte saving bought with more than twice the decode instructions cannot be turned into
throughput. This is a property of the encoding, not of the implementation -- 1.75-bit dense base-3 has
no bit-parallel form.

The tier is therefore not usable on this card. Its only benefit is 1.18 GiB of weights, about +37k
tokens of context at 33.0 KiB/token (131,072 -> ~168k), and it costs 0.62x the decode and 0.32x the
prefill. The whole fast-path port -- the tile kernel, the `gemv_admits` admission, and the
`pq2_admits` isolation of the routes that still decode two-bit slots -- was written, verified and then
reverted; it is kept as `ptq1_port_wip.patch` outside the tree rather than committed.

Two findings outlive it:

* `gemv_admits` admitting both formats while the SIMT GEMM, the row-blocked GEMV and the int8 rung
  stayed PQ2-only is a silent-corruption shape, not a crash: those kernels would read a 24-byte code
  plane as a 32-byte one and return plausible garbage rather than failing. A second ternary format
  needs a `pq2_admits`-style gate at every such call site before its own decode is written.
* The fp16 activation container is a PQ2_0 property, not a global default. See the row in "Wrong
  values, and what they cost".

## Four warps per D1024: the rotation rewrite (`327f4b10`)

The rotation was the last named non-GEMV item in the step, and its cost splits in two: the transform
itself and the fact that it is a separate kernel launched 258 times per token. The second half is a
launch floor (~2.9 us per call) that no kernel-side work can touch; the first was 1.15 ms of the
21.5 ms step. The rewrite attacks the transform by changing who owns the data:

    i (within the token) = 256*unit + 8*lane + j,  unit 0..3, lane 0..31, j 0..7

so a lane owns **eight contiguous elements** instead of 32 elements at stride 32. The whole x read
becomes one `uint4`, the signs two `float4` (3 loads instead of 64), each warp runs a quarter of the
butterfly over its own 256-element unit, and the four units are joined through 4 KB of shared memory
and one `__syncthreads()`.

Bit-identical is the design constraint, not a hope: a Sylvester H1024 is ten pairwise stages, one per
index bit, and stages on disjoint bit sets commute. The one-warp kernel executes them in global bit
order 0..9 (shuffle strides 1,2,4,8,16 then register spans 1,2,4,8,16); the split kernel executes the
same ten stages in the same order, with the same `low+high` / `low-high` convention and the same
`__fadd_rn` / `__fsub_rn`, on the same index pairs. Every intermediate value matches, so the output
does too.

`NINFER_TERNARY_ROTATE_SPLIT=<maxT>` is the knob: 0 or unset is off, **16 covers the decode and
MTP-verify band**. The threshold is not cosmetic -- with the split kernel on for every T, a
3662-token prefill falls 1.21k -> 1.18k tok/s, because at large T the one-warp kernel's grid already
hides the serial load chain and only the `__syncthreads()` remains. Measured on the shipped route:

| | before | after |
|---|---:|---:|
| decode, 300 tokens, int8 KV | 44.6 | **46.6 (+4.5%)** |
| prefill, 3662 tokens | 1.21k | 1.21k (unchanged with the threshold) |
| 96-token greedy output | md5 `85f5cf2c...` | **identical** |
| perplexity, docs/cli.md | mean_nll 1.671277 / PPL 5.318954 | **identical** |

Registers 45 against the one-warp kernel's 42, no spills, no change to the workspace accounting.
The probe caveats that come with this rewrite -- HADAMARD=0 no longer isolates rotation cost, and the copy probe still
launches the one-warp shape -- are in "Low-cost sweeps" below.

## The decode step now: probes, and what it is made of (`9cb0ca1f`, `1df45be1`)

`NINFER_TERNARY_GEMV_PROBE` reached only the dedicated T=1 kernel, which the fp16 tile path never
calls -- all four arms measured the same 44.6 t/s and said nothing about the kernel that is 67.5% of
the decode step. The arms are now a compile-time template parameter of
`ternary_pq2_gemv_tile_kernel` (default off, so the shipped instantiation is unchanged), selected for
the decode shape only (`kT == 1`) at the default depth. They mean what the T=1 arms meant.

Base 46.6 t/s = 21.46 ms/token, one arm at a time, 300 generated tokens:

| arm | removes | t/s | ms/token | saved | share of step |
|---|---|---:|---:|---:|---:|
| base | -- | 46.6 | 21.46 | -- | -- |
| `nocode` | the code load **and** its decode | 80.7 | 12.39 | 9.07 ms | 42% |
| `codealu` | the code load only | 57.2 | 17.48 | 3.98 ms | 19% |
| `noscale` | the group scale load | 49.9 | 20.04 | 1.42 ms | 7% |
| `noact` | the activation load and its converts | 48.6 | 20.58 | 0.88 ms | 4% |

Arms are upper bounds and do not add: `nocode` also lets the dot fold against constant weights, so
the code **decode** is *at most* the 5.09 ms gap to `codealu` (24% of the step) and the code **load**
is 3.98 ms. The code stream -- load plus decode -- is 9.1 ms of 21.46, the largest named item.
The activation side is only 0.88 ms: it is L2-served and was never the story.

The first `noact` implementation was wrong and is recorded here so it is not repeated: setting
`dot = 0.0F` let `fmaf(scale, 0, acc)` fold, which took the scale load, the code load and the decode
with it and measured 120 t/s as "delete the whole group walk". Constant activation *values* with the
math kept live is the arm the T=1 kernel carries, and that measures 48.6.

SASS of the shipping instantiation (`kT=1, kUnroll=4, fp mode 1`) puts the group loop at **~26.5
instructions per group**: 3 `LDG`, **~12.3 of code decode** (LOP3/SHF/IMAD.SHL plus two `HADD2
-1025`), ~7 of dot, ~3.5 of address arithmetic -- which is exactly where the `nocode`/`codealu` delta
lands. 42 registers, no spills.

**A refuted fix: the weight table.** `NINFER_TERNARY_TILE_WTABLE=1` reads the four `(code-1)` fp16
values of a byte straight out of a 2 KB device table (`table[byte]`, one `LDG.64`) instead of
rebuilding them from the magic-number construction. It works exactly as designed -- 16 instructions
per group against 26.5, 45 registers with no spill, output bit-identical -- and it is **28% slower
(33.5 t/s) at every depth**. The dependent table load costs more than the arithmetic it replaces, and
unroll does not hide it. Kept behind the knob so the negative result stays reproducible; never a
default.

Composition of the step on this build (nvprof shares, inflated as described above): tile GEMV
**67.5%**, rotation **5.3%**, norms/residual/attention/GDN/sampling the remaining 27%.

## MTP K=1 against K=3, at three context sizes

Same prompt files, same seed (`--greedy`), `--max-new 400`, `--max-context 32768`,
`--prefill-chunk 4096`, int8 KV, `NINFER_TERNARY_ROTATE_SPLIT=16`. Two repetitions per cell; they
agree to 0.1 t/s, so the tables show the mean. Arms: no spec, `--spec mtp --draft-tokens 1
--lm-head-draft` (what `ninfer-serve` runs), `--draft-tokens 3` (what the upstream example runs).

Decode, tok/s:

| prompt | no spec | K=1 | K=1 gain | K=3 | K=3 gain |
|---|---:|---:|---:|---:|---:|
| 1,002 tok | 51.05 | **65.5** | **+28.3%** | 58.9 | +15.4% |
| 4,042 tok | 49.55 | **62.95** | **+27.0%** | 61.0 | +23.1% |
| 15,632 tok | 43.70 | **52.05** | **+19.1%** | 44.6 | +2.1% |

Prefill, tok/s (spec costs 1.5-2.5%; the 1,002-token row is low because the prompt does not fill a
4096 chunk):

| prompt | no spec | K=1 | K=3 |
|---|---:|---:|---:|
| 1,002 | 1.035k | 1.015k | 1.01k |
| 4,042 | **1.22k** | 1.20k | 1.20k |
| 15,632 | 1.13k | 1.11k | 1.11k |

Speculative statistics from the CLI summary:

| prompt | K=1 rate | K=1 tok/round | K=3 rate | K=3 tok/round |
|---|---:|---:|---:|---:|
| 1,002 | 70.1% | 1.91 | 47.9% | **2.59** |
| 4,042 | 75.2% | 1.75 | 53.4% | **2.60** |
| 15,632 | 59.0% | 1.59 | 34.9% | **2.05** |

**K=3 does take more tokens per round -- and still loses.** The net rate is
`acceptance_length / r`, where `r` is a verify round against a plain T=1 step; solving the six cells:

| prompt | arm | measured gain | tok/round | implied r |
|---|---|---:|---:|---:|
| 1,002 | K=1 | +28.3% | 1.91 | 1.49 |
| 1,002 | K=3 | +15.4% | 2.59 | 2.24 |
| 4,042 | K=1 | +27.0% | 1.75 | 1.38 |
| 4,042 | K=3 | +23.1% | 2.60 | 2.11 |
| 15,632 | K=1 | +19.1% | 1.59 | 1.34 |
| 15,632 | K=3 | +2.1% | 2.05 | 2.01 |

r(T=2) sits at 1.34-1.49 and r(T=4) at 2.01-2.24: K=3 buys 26-49% more tokens per round and pays
~50% more per round. Per extra verified token that is ~0.40 (T=2) and ~0.37 (T=4) of a T=1 step --
against the 0.60 this document recorded on the older build, so the fp16 group-dot, the four-warp
rotation and depth 8 all made extra verified tokens cheaper, just not cheap enough.

Context costs both routes: the no-spec baseline falls 51.05 -> 49.55 -> 43.70 from 1k to 16k
(-14.4%), the verify round grows faster than the T=1 step, and acceptance decays (K=1 70.1 -> 59.0%,
K=3 47.9 -> 34.9%), so the MTP premium shrinks +28.3% -> +19.1% over the same range.

So **K=1 is the right window on this card** and `ninfer-serve` already runs it. Other frameworks
defaulting to 3 are not contradicting this: their verify is a tensor-core batched GEMM where
r(T=4)/r(T=2) is close to 1, so free draft tokens are nearly free. Here every extra token walks the
activation + FMA chain of a ternary tile GEMV, and r grows.

One caveat on the counters: at 1,002 tokens both arms show a 15-wide tail in `accepted by pos`
(`147,5,5,...` against `108,63,33,0,0,...` at the longer prompts) and `drafted > rounds x window`, so
an extra lookup-path verify contributed ~43 accepts at short context only. Both arms carry it, so the
comparison stands; the counters should not be used to back out per-round timings. Logs and the
matrix script are in `/root/ninfer_ab/` on the host.

## What tensor cores are worth here, measured

The claim this document leans on -- tensor cores carry prefill and have nothing to do with decode --
was re-measured end to end on this model and build. One 4,042-token prompt, `--max-new 4`, only the
route switches:

| route | tensor core | prefill, tok/s |
|---|---|---:|
| dequantise + CUTLASS Sm70 fp16 tensor-op (default) | yes | **1,230** |
| fused fp16 `mma.m8n8k4` (`NINFER_TERNARY_CUTLASS=0`) | yes | 798.5 |
| row-blocked SIMT (`CUTLASS=0` + `PREFILL=block`) | no | 112.5 |
| SIMT GEMM (`CUTLASS=0` + `PREFILL=simt`) | no | 96.7 |
| correctness-first reference (`CUTLASS=0` + `PREFILL=ref`) | no | **33.6** |

**Tensor cores are worth 10.9x the best non-tensor-core route and 36.6x the reference** on prefill.

Method trap, recorded because it cost a control arm: `NINFER_TERNARY_PREFILL=block` alone does *not*
remove tensor cores -- CUTLASS admission is a separate switch, so a `block` arm with CUTLASS still on
measured 1,230, identical to the default. Both switches are required for a true no-TC arm.

The two admission gates were also swept instead of trusted (`--prefill-chunk` sets the M each forward
is measured at):

| M | fused MMA | CUTLASS | winner | default gate picks |
|---|---:|---:|---|---|
| 128 | **440.6** | 295.5 | MMA by 49% | `MMA >= 32` -- correct |
| 512 | 686.0 | **777.7** | CUTLASS by 13% | `CUTLASS >= 256` -- correct |
| 4096 | 798.5 | **1,230** | CUTLASS by 54% | correct |

CUTLASS keeps climbing with M (295 -> 778 -> 1,230) while the fused MMA saturates near 800; the
crossover sits between 128 and 512, so the shipped thresholds (32 / 256) select the winner at every
measured point. And `NINFER_TERNARY_CUTLASS_MIN_T=1` no longer reaches the walk this document records
-- it fails startup with `std::bad_alloc` while preparing CUDA graphs (see Wrong values).

Why decode and the verify band stay off tensor cores, restated with current numbers: Volta's
`m8n8k4` A operand is 32 tokens tall (M=1 measures 0.3 TFLOP/s, i.e. 0.24% of peak), the verify
widths actually used are T=2..4 (draft windows of 1 and 3, MTP capped at 5), and decode runs 20x
below the ridge (6.95 against 139 FLOP/byte) so it is bound by bytes and issue slots rather than by
FLOPs. The one angle never measured here is the fused-MMA gate at T=16 (the lookup window width):
it would need a dispatch change rather than an environment variable, and it only affects the lookup
path -- left open deliberately.

## Low-cost sweeps: K=2, verify depth, and the switches (2026-10-01)

Thirty-five runs, every one of them an environment switch or a CLI flag -- no rebuild. Logs live in
`/root/ninfer_ab/low` and `/root/ninfer_ab/low2` on the host, driven by `run_lowcost.sh` and
`run_low2.sh`.

### What these numbers are worth: batch discipline

The non-MTP arms reproduce across batches to 0.1% (49.55 against 49.6 tok/s of medium decode, 1.13k
against 1.13k of long prefill). The MTP arms do not: the same K=1 configuration measured **65.7 and
62.9 tok/s in two different batches** (4%), while two runs inside one batch agreed to 0.05%
(65.7 / 65.7). Every MTP number below is therefore same-batch, sequential, against a control measured
in that same batch; a cross-batch MTP comparison is not usable.

### The missing arm: K=2

| prompt | K=1 | K=2 | K=3 |
|---|---:|---:|---:|
| 1,002 tok | **65.7** | 63.2 | 58.9 |
| 4,042 tok | 63.2 | **64.7** | 61.0 |
| 15,632 tok | **52.2** | 49.3 | 44.6 |

Acceptance length for K=2 is 2.31 / 2.28 / 1.88 tok/round, between K=1 and K=3 where the model puts
it: solving `gain = tok_per_round / r` gives r(T=3) = 1.66-1.86, again between r(T=2) = 1.34-1.49 and
r(T=4) = 2.01-2.24.

K=2 takes the 4k context only (+2.4% over K=1) and loses at 1k (-4.0%) and 16k (-5.9%). **K=1 stays
the shipped window**: a win of that size does not pay for a per-context setting, and the long-context
loss is the larger error.

### Verify-band depth, re-measured

`NINFER_TERNARY_GEMV_TILE_UNROLL` reaches kt = 2, 3 and 4 (only the old `case 1` was hardcoded), so
this sweep is pure environment:

| kt (window) | sweep, tok/s | best | shipped default |
|---|---|---|---|
| 2 (K=1) | 1 -> 44.2, 2 -> 55.3, 4 -> 64.9, **8 -> 65.7**, 16 -> 49.9 | **8** | 8 -- unchanged |
| 4 (K=3) | **4 -> 58.9**, 8 -> 54.1, 16 -> 36.3 | **4** | 4 -- unchanged |

Both shipped defaults hold on this build. An early sweep appeared to put 4 above 8 for kt=2 (64.8
against 62.9); the confirmation batch measured 8 at 65.7 twice, so that reading was the batch effect
rather than the kernel.

### The switches, one at a time, same batch

| switch | control | switched | net |
|---|---:|---:|---|
| `--lm-head-draft` | 65.7 (acc 1.91) | 62.6 (acc 1.83) | **+4.7%** -- keep it; `ninfer-serve` already passes it |
| `NINFER_TERNARY_ROTATE_SPLIT=16` under MTP | 65.7 | 63.5 (off) | **+3.4%**, against +4.5% without spec |
| `--no-cuda-graph` | 49.6 | 49.4 | no measurable effect (one run each, ~0.5% resolution) |
| `--kv-dtype bf16` | 49.6 / 1.22k | 49.3 / 1.22k | no measurable effect on decode or prefill -- int8 KV buys context, not speed |

### `--prefill-chunk` is not monotonic

On the 15,632-token prompt, chunk 4096 prefills at **1.13k tok/s** and chunk 8192 at **783.6 and
788.3** -- a 31% *loss*, reproduced in both batches, with decode unchanged (43.7 either way). Bigger
chunks look better on paper (fewer passes, taller M); on this build they are not. The mechanism was
not investigated -- the candidates are the rotation's warp-packing switch at very large T and the
CUTLASS dequant chunk sizing -- so 4096 stays the setting and this row exists to stop anyone
"optimising" it upwards.

### Two probes that are no longer clean

`NINFER_TERNARY_HADAMARD=0` measured *faster* before the rotation rewrite (47.4 against 44.6) and
**slower** after it (45.6 against 49.5, reproduced twice). It also demotes the activation to the bf16
GEMV path and changes which buffer the GEMV reads, so once the transform itself got cheap those side
effects dominated: it no longer isolates rotation cost. `ROTATE_PROBE=copy` (49.8 against 49.5) puts
the transform at only ~0.3 ms over a copy-only kernel, but the copy probe still launches the
**one-warp** shape while the default path now runs four warps per D1024 -- so a split-shaped copy arm
is the missing measurement for a clean post-split decomposition.

**Same day: the arm exists now.** `ternary_rotate_bf16_split_copy_kernel` prices the split rotation's transform at
**0.52 ms** per token (50.6 against 49.3 tok/s) -- see "Round 2" below.

## Round 2: knob sweeps, the CUTLASS split, and probe builds (2026-10-01)

Third through fifth batches on this build (`1df45be1` + `NINFER_TERNARY_ROTATE_SPLIT=16`), same
discipline as before: same prompt files, `--greedy`, `--no-thinking` where a workload needs it, two
runs per arm where a claim is within 5% of its control, and `ninfer-serve` stopped only while the
GPU is claimed. Logs in `/root/ninfer_ab/{opt1,opt2,tier2}/`, drivers `run_opt1.sh`,
`run_opt2.sh`, `run_tier2.sh`.

### Every remaining knob, measured

| knob | arms | result |
|---|---|---|
| `NINFER_TERNARY_GEMV_TILE_UNROLL` at **kt=16** (the lookup verify width) | 1 -> 56.2, 2 -> 87.7, **4 -> 123.4**, 8 -> 87.9, 16 -> 77.8 | **4 is a sharp peak** and the shipped default is right; one step either way costs 30-54%. Never swept before -- only kt 1..4 had data |
| `--draft-tokens` | the CLI rejects anything outside **[1,7]** | K=7 is the hard ceiling. The 15 in `kMtpLookupMaximumDrafts` is the lookup's *verify* width, not the draft window |
| `NINFER_TERNARY_ROTATE_WPB` at 15.6k prefill | auto vs `=1`: 1.15k / 1.14k | the automatic packing choice is confirmed; no gain |
| `--prefill-chunk` at 15.6k tokens | 4096 -> **1.13k**, 6144 -> 1.09k, **7168 -> 858**, 8192 -> 784, 16384 -> 695 | **the cliff is at 7168** (-21% in one step), then monotone decay; 4096 stays the setting and "do not optimise it upwards" now has a boundary |
| `NINFER_TERNARY_CUTLASS_TILE=128x256` | 1.21k against 1.22k | no gain, keep 128x128 (reproduces the tile sweep) |
| `NINFER_TERNARY_FP16_ACT=2` | decode **42.5** against 49.2 | the half2 dot is worth **+15.8%** today -- the doc's -18% on the older build, same magnitude, and it is already the default |
| `NINFER_TERNARY_TILE_SHARE_ACT` under spec decode | acceptance 12.11 -> **1.97**, decode 34.8 | **unusable on the speculative path**: it garbles the output, so the acceptance collapse -- not the activation side -- is what the number measures |

### What the CUTLASS arm is made of (first decomposition on this build)

`NINFER_TERNARY_CUTLASS_PROBE` modes skip work rather than time it, so the five arms solve
`base = O + D + G` per token on the 4,042-token prompt (O = everything outside the arm: rotation,
norms, attention/GDN, embeddings; D = dequantise pass; G = the CUTLASS GEMM):

| arm | prefill | per token | reads as |
|---|---:|---:|---|
| base | 1.22k | 0.820 ms | O + D + G |
| `gemm` (dequant skipped) | 1.39k | 0.719 ms | O + G -> **G ~0.57 ms, 70% of prefill** |
| `dequant` (GEMM skipped) | 4.07k | 0.246 ms | O + D |
| `dequant2` (dequant twice) | 3.50k | 0.286 ms | O + 2D, consistent with the row above within read noise |
| `dqread` / `dqwrite` | 4.53k / 4.08k | 0.221 / 0.245 ms | the panel read is ~0.03 ms; the write is free within noise |

G at M=4042 is 4042 x 5.0e10 / 2.32 s = **87 TFLOP/s in situ** -- against the doc's standalone 84
and this family's measured 86-95 ceiling, so **the dominant term of prefill is already at ~93% of
what this kernel can do**. With O at ~18% and D at ~9-12%, prefill has no large lever left; the
"tile sweep says the GEMM is at its own ceiling" conclusion now has a full decomposition behind it.

### The lookup path, reproduced: the "80+" number

The context-lookup workload (a prompt that asks for N repetitions, `--no-thinking` mandatory -- see
the pitfalls list), control 51.6 tok/s with no spec:

| arm | doc (older build) | **now** | delta |
|---|---:|---:|---:|
| K=1, repeat x10 | 100.8 | **103.6** | +2.8% |
| K=3, repeat x10 | 104.6 | **119.0** | +13.8% |
| K=7, repeat x10 | 104.7 | **123.9** | +18.3% |
| K=2, repeat x16 | 82.6 | **118.2** | +43% (prompt shape differs from the doc's; magnitude only) |
| K=5, repeat x16 | 81.7 | **124.1** | +52% (same caveat) |
| control, no spec | 43.8 | 51.6 | +18% |

Every spec arm shows a 15-entry `mtp accepted by pos`, i.e. lookup was live in all of them, and the
2.0-2.4x-over-no-spec ratio reproduces the doc's 2.3-2.4x. The two older claims this revises: the
**80.5** quoted for this path was already corrected once (row-blocked kernel), and the doc's
**"ordinary prose MTP is a 26% loss (30.7 against 41.7)"** is now reversed -- on plain prose this
build measures +19% to +28% (65.7 / 63.2 / 52.2 against 51.05 / 49.55 / 43.70), because the base
step got 14.6% cheaper while the verify band got the fp16 dot. The lookup path is still 1.9x above
plain prose; the gap is the workload, not the window.

### Probe builds: wall-clock prices for the decode microkernels

Three wrappers gained an env-gated "do not launch this kernel" arm
(`NINFER_TERNARY_PROBE_SKIP_{RMSNORM,RESIDUAL,SILU}=1`; sizes and downstream addresses unchanged),
and the rotation gained a split-shaped copy arm. Control 49.3 tok/s = 20.28 ms/token:

| arm | decode | delta | price |
|---|---:|---:|---:|
| skip `rmsnorm` (210 calls/pass) | 51.2 | +3.9% | **0.77 ms**, 3.7 us/call |
| skip `residual_add` (128 calls) | 50.4 | +2.2% | 0.45 ms |
| skip `silu_mul` (61 calls) | 50.2 | +1.8% | 0.37 ms |
| all three | 52.9 | +7.3% | 1.47 ms -- **within 8% of the sum: the arms are additive** |

**The `residual_add` row has since been collected** (round 9 of
`docs/decode-round-2026-10-01.md`): the re-priced probe read +1.9% on this build's faster step
(54.2/54.3 -> 55.3/55.3) and the fused epilogue delivered the whole of it -- 54.2 -> 55.3 t/s, md5
unchanged. The other two rows are still open, and the `rmsnorm` row is the largest single item left
on the tail.
| split-shaped copy probe | 50.6 | +2.6% | -> **split rotation's transform = 0.52 ms** |

Two readings that matter:

* **These kernels are launch-dominated.** 3.7 us per rmsnorm call against a ~2.6 us launch floor
  (measured on the rotation) plus one row of work at T=1 says most of the cost is *starting* the
  kernel, not computing it. So the fusion prize is bigger than the rotation alone: rotation launches
  plus norm/residual/silu launches are ~1.0-1.5 ms of launch floor, **+5-7% of the step**, against
  the +3.4% that only counted rotation. Merging microkernels is the largest identified lever that
  does not need an artifact change.
* **nvprof reliability is per-kernel, not a uniform factor.** On this batch `residual_add` reads
  0.445 ms in the trace against 0.45 ms on the wall clock (exact), `rmsnorm` 1.10 against 0.77
  (+42%), and the rotation is high as well. Read the budget table as shares with per-kernel error
  bars, not as one systematic 2x.

Verification and caveats: the probe patch leaves the default path **byte-identical** (96-token
greedy md5 compared against the recorded baseline, `REGRESSION_OK`). The control run for this batch
stopped at 79 tokens on a natural end-of-sequence while the probe arms -- whose output is garbage
and therefore rarely ends -- ran to 400; rates stay comparable and the probe deltas are therefore
conservative (longer context is slower). The skip arms also show +6-9% on **prefill**
(1.22k -> 1.32k / 1.30k) which the element counts cannot explain by three orders of magnitude;
that is logged as an open question below rather than claimed as a lever.

**Open question (2026-10-01): why do the skip arms speed up prefill?** The kernels in question move
83 MB (rmsnorm) and 422 MB (silu) per 4,042-token prompt, worth ~0.1 ms against a 3.31 s prefill,
yet skipping them saves 0.21-0.25 s. What the follow-up established:

* **Reproducible and specific.** Control 1.22k / 1.22k, skip_silu 1.32k / 1.31k, skip_rmsnorm 1.31k
  / 1.29k, and -- the negative control -- skip_residual 1.22k, run *after* the fast arms in the
  same batch. So it is not run order, not thermal, and not "any skip helps".
* **It lives in the forward pass.** `--max-new 4` still prefills at 1.31k, so generation length is
  irrelevant; `--no-cuda-graph` leaves both arms where they were (1.21k control, 1.32k skip), so it
  is not graph capture.
* **It is execution time, not scheduling.** Two nvprof traces (control against skip_rmsnorm,
  `--max-new 1`) sum to 3,338 ms against 3,108 ms of kernel time -- a **-230 ms** delta that matches
  the wall clock. The saving is not gaps: `ternary_cutlass_sm70` alone accounts for **-175 ms**
  (2,268 -> 2,113, plus its two smaller instances), the GDN recurrent kernel -18.5 ms, the rotation
  -4.8 ms, the tile GEMM -7.3 ms. Every kernel that *consumes the normed stream* got faster; nothing
  else moved.

So the mechanism is **data-content dependent**: whatever the control's real activations contain makes
these kernels slower than the unwritten arena buffer the skip arms leave behind. Candidate
explanations, in order: fp16 subnormals in the rotated activation panel (the classic tensor-core
penalty, though the magnitude needs checking against the actual value distribution), a zero/garbage
fast path in one of the kernels, or a value-dependent scheduling effect. The decisive experiment is
an activation histogram probe (count subnormals in the rotation output during one prefill pass);
until it runs, the candidate optimisation this opens -- **~7% of prefill, 0.2 s of a 3.31 s prompt**
-- is logged but not claimed.

* **The histogram probe ran** (`NINFER_TERNARY_ROTATE_HIST=1`: six magnitude bins, ~53% grid
  coverage, two runs agreeing to seven digits). Control, i.e. the real rotated activation: exact
  zeros ~0%, would-be fp16 subnormals **0.228%**, <2^-6 36.7%, <1 47.1%, >=1 12.6% -- so 96% of
  the buffer is comfortably normal and the subnormal hypothesis that motivated the probe is
  **weakened**: 0.23% is a thin thing to hang a 7% penalty on. skip_rmsnorm, i.e. the unwritten
  arena buffer: **79.7% exact zeros + 20.3% below 2^-14 and nothing else**. The two arms therefore
  differ by ~40 percentage points of magnitude mass, which -- by elimination, since removing one
  small kernel cannot cost 0.2 s -- makes the buffer CONTENT the cause, while "why does content
  cost time" (every arithmetic in these consumers looks value-independent) stays open. Next
  discriminator: an arm that runs the real norm and then zeroes its output; fast would confirm
  content, slow would point back at kernel count. The discriminator ran
  (`NINFER_TERNARY_PROBE_RMSNORM_ZERO`: the real norm executes, then its output is zeroed, so cost
  is >= control while the content downstream reads is clean). Same batch, two runs each: control
  1.22k / 1.22k, zeroed **1.29k / 1.28k** of prefill (+5.3%) -- doing *more* work with cleaner
  content is faster, so **content is confirmed as the cause and kernel count is ruled out**. Decode
  moves the other way (48.5 against 49.2, -1.4%), so the sensitivity is per kernel: the CUTLASS
  arm dislikes real activations at M=4042 while the T=1 tile GEMV mildly prefers them. What is
  still open is *which feature* of the content matters -- zeros are fast, yet control is only 0.23%
  subnormal, so the subnormal penalty does not survive -- and therefore whether the ~5% is even
  reclaimable: real activations are the input, so it becomes a lever only if the feature that makes
  them slow turns out to be removable. The next probe would substitute constants inside the CUTLASS
  arm (zeros against ones against real) to separate "zeros are special" from "real values are slow".

## Fusion, measured: bit-identical and -2.6% (2026-10-01)

The microkernel prices in "Round 2" said the launch floors -- rotation, rmsnorm, residual, silu --
were worth +5-7% if the kernels merged. The adjacency census narrowed that to what actually
adjacates: rmsnorm's output feeds the fold rotation at two sites per layer, 128 calls per decode
step, and the add sits inside another op so it cannot be folded without deferring it. The first
implementation landed exactly as designed:

* `split_transform_unit()` factored out of the split kernel and shared by both callers (md5
  identical afterwards);
* a fused `rmsnorm_rotate_kernel` whose phase 1 is the shipped
  `rmsnorm_cta_bf16x2_kernel<Offset, 256, 10, true, 5120>` body and whose phase 2 runs that same
  `split_transform_unit()` on those exact bf16 values;
* wired through `linear_swiglu` (`in_norm` + `NINFER_TERNARY_NORMROT`) and the shared graph's
  `mlp_tail`, with the 35b MoE target taking a plain-norm path so the shared graph stays correct
  for both models.

Verification first, timing second:

| gate | result |
|---|---|
| md5, fusion on | **identical** to the recorded baseline (prefill exercises the relocated plain norm, decode the fused path) |
| md5, fusion off | identical |
| prefill, same batch | 1.22k either way -- prefill never fuses (t <= 16 only) |
| decode, same batch, two runs each | off 49.2 / 49.1, **on 47.9 / 47.9 -> -2.6%** |

The fusion is bit-identical and LOSES, and the reason is what the estimate under-weighted: the
split rotation is **five CTAs of four warps on five SMs**, while inside one 256-thread CTA the same
five D1024 units have to run in three rounds of two with shared staging and a barrier per round.
The saved launch floor is ~2.6 us per call; the serialization costs ~4.1 us per call at 128 calls
per step. Merging microkernels pays only when the merged shape keeps the parallelism, and a
row-norm CTA (one block per token row) cannot give a unit-parallel transform that.

The knob ships **off** (`NINFER_TERNARY_NORMROT` defaults to disabled; `=1` reproduces the losing
arm). The follow-up candidate was built and measured too: a 640-thread CTA (20 warps = five four-warp
groups, all five D1024 units in ONE round) with phase 1 pinned to the original 256 threads --
indexing, pair count and `block_reduce_sum<256>` fixed, so idle threads only write unread slots of
a 20-entry `warp_sums`, pass the same barriers, and skip the stores. md5 with
`NINFER_TERNARY_NORMROT=2` is **identical** as well: all three arms (default, `=1`, `=2`) match
the baseline byte for byte, so the shape change is provably value-neutral. It still measures
**48.2 / 48.2 against 49.2 / 49.2 (-2.0%)**.

Two shapes, both bit-exact, both losing: the conclusion is not that the implementation is wrong but
that this pattern cannot pay here. What the fused kernel saves -- one launch floor, ~2.6 us -- it
gives back and more putting five independent CTAs, each on its own SM, into one block, plus the
shared-memory staging and the extra barriers. **Fusion is closed for norm+rotate on this card.**
The knob ships off, both arms stay in-tree for reproduction, and the launch-floor arithmetic above
is superseded by these two measurements.

## Decode/MTP round decomposition: the verify band is issue-bound (2026-10-01)

Measured while chasing the decode/MTP gap to upstream, with prefill taken off the table.

Baseline matrix on non-repetitive prose (prose4k/prose16k, built from this tree's own docs so no
repeated 16-token suffix exists and the lookup path cannot fire) plus the context-reproduction
fixture:

| arm | prose4k | prose16k | lookup (repeat) |
|---|---:|---:|---:|
| no spec | 48.2 | 39.2 | -- |
| K=1 | 64.1 (acc 1.84) | 55.1 (1.83) | -- |
| K=2 | **65.4** (2.38) | -- | -- |
| K=3 | 61.8 (2.71) | **58.6** (2.83) | -- |
| K=4 | 60.6 (3.09) | -- | -- |
| K=7 | 46.9 (3.59) | -- | **122.2** (12.11) |

Two things this replaces. The earlier "K=1 always" reading came from the water-cycle fixtures,
whose repeated suffix lets lookup fire intermittently (`accepted by pos` showing stray counts in
slots 4..15) and inflates acceptance; on prose the best window moves with context (K=2 at 4k, K=3
at 16k). And fitting `round_ms = base_ms * (1 + r*K)` gives **r = 0.37 at 4k / 0.30 at 16k** -- a
verified token costs ~a third of a T=1 step, against the 0.60 this document used to quote. The
cost-model break-even acceptance is r, which is exactly why K=4's position 4 (35%) stops paying and
K >= 5 loses.

Against upstream's 35B-A3B row (round 23.80 -> 30.20 ms for two more drafts, r = 0.135) our
marginal cost is still ~2.7x higher, and that -- not the draft head -- is the dominant ordinary-prose
lever now: at upstream's r our own K=3 acceptance (2.71) would measure ~93 t/s against the 61.8
measured. The artifact is not the blocker either: the draft stack is W8 with a Q4 draft head, i.e.
*more* precise than the 2-bit target it has to predict.

nvprof family rollups, same tiny prompt, three captures (T=1 step, K=1 round, K=7 lookup round whose
verify runs at T=16):

| family | T=1 | T=2 | T=16 verify |
|---|---:|---:|---:|
| ternary-gemv | 675 ms | 767 (72.5%) | 713 (71.6%) |
| ternary cutlass/mma (prefill only) | 122 | 147 | 168 |
| rotation | 63 | 32 | 19 |
| norm | 33 | 23 | 13 |
| attention | 20 | 16 | 12 |
| gdn | 22 | 15 | 9 |

The wide verify costs **~71 ms per round for 16 tokens = 9.2e11 FLOP -> 13.0 TFLOP/s**: arithmetic
issue-bound at ~42% of this part's fp16x2 peak, and 3.4x above the 20.7 ms that pure weight traffic
(7.19 GB at the achieved 347 GB/s) would need.

**Two candidates tested and refuted.**

* *Padded tensor-core verify.* The fused MMA path measures **17.2 TFLOP/s at M=32** (331 us for
  17408x5120, ternary dequant fused) -- better per FLOP than the SIMT tile, but padding T=16 to the
  m8n8k4 A-operand's 32 rows doubles the work and leaves 8.6 TFLOP/s of useful throughput: worse
  than the SIMT path it would replace. Do not build it.
* *Wider code reads.* Re-confirmed dead twice over: the row-pattern probe already sits at 98.5% of a
  flat uint4 stream, and the CUTLASS fp16 probe is flat in M from 8 to 128 at ~310-350 GB/s, so this
  band's problem is issue, not traffic.

**What is left, and where it actually sits.** Two different bottlenecks, easy to conflate:

* *The wide verify* is per-token work, not weight traffic: at T=16 the code byte and scale are read
  once and reused sixteen times (177 us/call against 42 us at T=1, i.e. the second through sixteenth
  tokens cost ~0.2 of a T=1 pass each in the GEMV -- the reuse is working). What remains is the
  per-token body: one 8-byte activation load and two half2 FMAs per (token, group), which is why the
  band sits at 13 TFLOP/s of a 31-TFLOP/s fp16x2 peak. The activation load is already minimal (four
  contiguous bf16) and the FMA count is the mathematics, so this band has no large lever left; the
  padded-MMA route above was the one candidate and it loses.
* *The T=1 step* is where the code-stream instructions are not amortised at all: 42 us per linear
  call for 17.9 MB of codes is 426 GB/s of the card's ~900, and the older `nocode` / `codealu`
  probes put the code load at ~15% and its decode at ~13% of the step. Re-packing the code plane so
  one lane's bytes for four consecutive groups are contiguous -- a same-size 32x4 transpose inside
  each 128-byte block, still perfectly coalesced, and every model width here has a group count
  divisible by four -- collapses four one-byte loads into one 32-bit load. That is the next thing to
  prototype at kernel level (repack into a scratch buffer from the loader, no artifact change yet),
  and it lifts decode and MTP together because every verify round contains the same code walk. That prototype ran
  (`code_layout_probe.cu`, 17408x5120, 40 groups, 200 reps, all three patterns bit-identical:
  shipped 0.101 ms, transposed with four byte loads 0.109 ms (-7.0%), transposed with one 32-bit
  load per lane per four groups **0.094 ms (+7.5%)**). So the candidate is real but modest -- +7.5%
  on the code walk, which `nocode`/`codealu` bound at ~15-18% of a T=1 step -- and the cost is not
  the kernel but the layout's blast radius: the code plane is read by the tile GEMV, the row-blocked
  GEMV, the SIMT prefill, the fused MMA and the CUTLASS dequant, so either every reader learns the
  transposed form (including the prefill paths this port has frozen) or the loader repacks in place
  and prefill carries an inverse repack. Parked as a measured candidate. **Second pass -- the candidate is withdrawn.** That
  first probe reached only 205 GB/s while the shipped kernel runs at ~350-425 GB/s, i.e. it was
  load-latency-bound and therefore over-credited any load-count reduction. Re-running it with the
  group-walk unroll and a launch-bounds register cap (same arithmetic; outputs still bit-identical)
  collapses the advantage:

  | unroll | shipped | transposed | gain |
  |---:|---:|---:|---:|
  | 1 | 0.105 ms (198 GB/s) | 0.075 ms (278 GB/s) | +40.1% |
  | 2 | 0.089 ms (234 GB/s) | 0.072 ms (288 GB/s) | +22.9% |
  | 4 | 0.073 ms (286 GB/s) | 0.071 ms (291 GB/s) | **+1.9%** |

  The transpose buys back load-latency stalls only, and the shipped kernel already hides them
  (unroll 4-8 with `kMinBlocks` 4). At the shipped operating point it is worth at most ~2%, not the
  +7.5% of the first probe, so a layout change that would have to touch all five code-plane readers
  -- frozen prefill paths included -- is not worth doing. Withdrawn.

## Session delta: this work against the pre-session snapshot (2026-10-01)

The numbers above are scattered by topic; this is the like-for-like roll-up for whoever picks this up
next. "Remote snapshot" is the state this document was in before the session
(`docs/ternary-port-remote.md` locally), i.e. the tree at `6609089e` with the pre-session defaults.

| metric | remote snapshot | now | delta |
|---|---:|---:|---|
| decode, old default route (warp-per-row) | 41.6 t/s | 44.6 t/s was that route's successor | -- |
| decode, **same route** (tile walk) | 44.6 t/s | **51.1** t/s (`ROTATE_SPLIT=16`), 48.7 with no env | **+14.6%** (+22.8% against 41.6) |
| prefill | 1,190 t/s @3412 | **1.21-1.23k** @3662/4042 | +2-3% |
| prose MTP K=1 | 51.8 (no spec 43.7, +18.5%) | **64.1** (no spec 48.2, **+33%**) | +23.7% on the arm |
| prose MTP K=2 / K=3 | 48.0 / 41.9 -- both BELOW no spec | **65.4 / 61.8** -- both ABOVE no spec | loss -> win |
| context-lookup K=7 | 83.7 (repeated sentence), 81.2 (verbatim list) | **122.2-124.1** | **+46-53%** |
| marginal verified token `r` | 0.60 of a step | **0.37 (4k) / 0.30 (16k)** | -38 to -50% |
| wide (T=16) verify throughput | not measured | 13.0 TFLOP/s, issue-bound | new characterization |
| context-lookup K=7, QPN wide verify (default on) | 83.7 / 81.2 | **234.8 / 246.6** | **+90-98%**, only the T = 16 verify pass changes |
| realistic serving (thinking, K=1) | not measured | **59.3-61.7 t/s decode**, acceptance 58.5-73.9%, prefill 1.13-1.15k, TTFT 0.32-9.9 s | new |

Attribution, from the commits and the measured unroll/rotation curves: 41.6 -> 44.6 is the fp16
half2 dot (`6609089e`); 44.6 -> 46.6 is four warps per D1024 in the fold rotation (`327f4b10`); 46.5
-> 51.1 is defaulting the T=1 tile walk to depth 8 (`1df45be1`). `0bcc2681` adds wall-clock probes
only. Two caveats: the old 41.6 was a *different kernel route* (the doc says so explicitly), so the
same-route delta is +14.6%; and the MTP rows mix fixtures -- the snapshot's prose was the
repeated-sentence water-cycle text where lookup fires intermittently, while the current rows use
non-repetitive prose built from this tree's own documentation.

**Directions this session measured and refuted** (do not re-open without new evidence): code-plane
transpose of the code stream (<=2% at the shipped unroll, +7.5% only in a latency-bound probe);
padding T=16 to the tensor core's 32-row A operand (17.2 TFLOP/s at M=32, but 2x the work leaves 8.6
effective against the SIMT path's 13.0); wider code reads (the walk is already 98.5% of a flat
uint4 stream and the fp16 CUTLASS probe is flat from M=8 to 128); norm+rotation fusion (bit-identical
in both shapes, -2.6% and -2.0%); widening the draft window on short and medium real tasks (K=1 is
optimal there because position 2 accepts only 46-55%); row-blocked verify routes, T2BLOCK, the
device weight table, and occupancy/register tuning (the tile kernel already runs at 32 registers);
KV dtype and CUDA graphs (about zero for decode). The QPN resumption adds two more: a load-time code-plane permutation for the QPN kernel (the row-major plane is already contiguous per lane, worth ~4% of a 35% deficit, and adapting the three other readers to it is what parked the route), and routing the decode/MTP bands (T <= 5) through the tensor core (loses 47% at T = 2 and 21% at T = 3; the flat-cost win only starts at T = 6).

**Directions still open**, ordered by expected value: (1) serving concurrency -- `--max-concurrency`
is unset today, so every decode step streams the full 7.19 GB for one sequence, and the measured
`r` says a second sequence's token costs 0.2-0.37 of a step; this is the only remaining lever with a
large multiplier (aggregate throughput, not single-request latency). (2) the draft window at
contexts above 10k, never measured: acceptance there is already 73.9% at K=1 and the prose16k
fixture showed K=3 ahead of K=1 by 6.4%. (3) split-K for the T=1 group walk (+3-8%, uncertain, the
kernel is at full occupancy). (4) cheapening the draft stack, which is W8 with a Q4 head against a
2-bit target and costs 6.6% of a round. (6) inside the QPN band itself, the T = 6..16 rounds are latency-bound at 47-52 ms against an ~8 ms traffic floor: a software-pipelined weight fetch (prefetch the next group's two uint4 while decoding the current one), more CTAs for the small-n layers (kColsPerCta is pinned at 32 by the mma N), and NACC above 1 at kTiles = 2 were not tried. It only moves the lookup-style wide verify, not `r`. (5) model-side: recalibrating the draft head against the
quantized target is the only path to a large acceptance gain, and it is not an engine change.

## PQ2 tensor-core (QPN) route: resumed, bounded, and kept for the wide verify (2026-10-01)

**Verdict: kept, but only for `6 <= T <= 32`.** The route was ported to make a *verify token* cheap;
it does not do that, and shipping it ungated would have cost 47% on every real MTP round. What it does
have is a **cost curve that is nearly flat in T** (39.2 ms per round at T = 2, 46.8 ms at T = 16)
against the SIMT tile kernel's 26.6 -> 98.1 ms over the same span. The two lines cross at
**T ~ 4.3**, so the whole MTP band a real task uses (K = 1 and K = 2, i.e. T = 2 and T = 3) loses,
while the wide verify wins big: the T = 16 context-lookup round goes 98.1 -> 46.8 ms, **123.7 ->
234.8 t/s** on `lookup10` and **126.7 -> 246.6 t/s** on `lookup16` (same batch, identical acceptance
12.11 / 12.70, byte-identical output text; 300-token runs 123.2 -> 235.0). The gate buys that and
nothing else: the 96-token greedy md5 against `/root/wt/base.out` is **IDENTICAL** both with the
route enabled by default and with `NINFER_TERNARY_QPN=0`.

### What the parked version was actually doing wrong

Three faults, and the one the parking note blamed was the smallest of them.

1. **The code read was narrow, and vectorising it was worth almost nothing.** The reader took the
   plane four bytes at a time (`u * 4` into a 32-byte group). A lane here owns an output *row*, so a
   warp's 32 lanes read 32 rows `k/4` bytes apart and every narrow load fans out to 32 sectors -- the
   mechanism the parking note charged the whole 37% to. The artifact's row-major plane already puts a
   lane's whole group in 32 contiguous bytes, so two `uint4` loads replace all eight and each lane
   consumes exactly one sector. Measured against the SIMT arm in the same batch:
   **28.9/45.9 = 0.63 before, 32.6/49.8 = 0.66 after** -- worth ~4%, not 37%. (Consistent with the
   older "wider code reads" result: the row pattern is already 98.5% of a flat `uint4` stream.)
2. **SPLITK = 4 was half of upstream's, and it is a parallelism knob rather than a K-split one.** One
   CTA covers 32 output rows, so the grid is `n/32` -- 160 CTAs at n = 5120 -- and at SPLITK = 4 that
   is 8 warps per SM, nowhere near enough to hide a weight stream's latency. The NVFP4 sibling's own
   launcher calls SPLITK = 8 "the production floor across every kTiles bucket"; this kernel now
   matches it (every width in the model has a group count divisible by 8: 40, 48, 80, 136). NACC
   stays 1 above one 8-token tile: four accumulators per 16-k unit is 32 registers per live tile, and
   above one tile the tiles are already independent.
3. **Just over half the ternary weights could not reach the route at all.** A temporary shape probe
   (`n/k/t` printed once per distinct tuple, at both entry points) shows the linears split into two
   disjoint sets: `n = 5120/34816/5120/248320` at `k = 6144/5120/17408/5120` through
   `ternary_dispatch`, and `n = 4096/6144/1024` at `k = 5120` through
   `ternary_dispatch_basis_strided` -- the attention input projections, the GDN input projection and
   the SwiGLU pair, which had no QPN gate in it at all. The gate is now on both entries, because the
   route is a property of the format and the shape and not of the caller that got there. This is what
   turned "+61% on the lookup, half the work covered" into "+90%".

**The prepack is not needed, and that dissolves the structural conflict that parked the route.** The
load-time permutation existed to make a warp's fetch contiguous; the row-major plane is already
contiguous *per lane*, so two `uint4` loads buy the same coalescing for free, and the three other
readers (fused MMA at T = 32..255, CUTLASS dequant at T >= 256, block GEMV at T = 5..31) never have
to learn a permuted index -- which the parking note listed as a ~2-300 line change across three
kernels, with prefill carrying it. `ternary_prepack_qpn` and the dispatcher's prepacked guard stay in
the tree, uncalled, as the record. The T = 5..31 block GEMV band is served by the QPN kernel instead
of by learning the permuted index, exactly as that note's table proposed as the cheaper branch.

### The measured band

Same-batch A/B, `real_task`, 128 generated tokens, `NINFER_TERNARY_QPN=0` against `=1`. The MTP arms
run one verify pass of width T = K+1 per round; the lookup arms are the context-lookup replay. Round
time is `acceptance length / decode speed`, which is the fair comparison once the two arms'
acceptance has diverged.

| verify width T | 2 | 3 | 4 | 5 | 6 | 8 | 16 (lookup) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIMT tile round, ms | 26.6 | 33.9 | 41.2 | 48.6 | 58.0 | 74.1 | 98.1 |
| QPN round, ms | **39.2** | **41.1** | **43.0** | **44.9** | 47.5 | 52.0 | **46.8** |
| QPN / SIMT | 1.47 | 1.21 | 1.04 | 0.92 | 0.82 | 0.70 | 0.48 |

The gate is 6 rather than 5 for margin: T = 5 is an 8% win and this workload's MTP batch drift is
+/-4%, while T = 6 is 18%. At T = 1 -- outside the gate in the shipped build -- the parked
configuration measured 30.7 ms against the block GEMV's 19.5 ms; the current configuration was never
run at T = 1 and does not need to be. The gate and this table both live in
`ternary_volta_qpn_supported`.

### Numerics of the band

Two checks, because greedy ids alone are a discrete test.

* **Output text.** Byte-identical between arms on both lookup fixtures (110 and 128 tokens), on
  `real_code` at K = 4 and K = 7 (4.4k-token prompt, identical acceptance 3.23 / 3.50), and on
  `real_task` at K = 1, K = 2, K = 4 and no-spec. One arm does flip: `real_task` K = 7 (T = 8),
  acceptance 2.74 -> 2.59 at 128 tokens and 2.18 -> 3.04 at 300 -- a last-bit difference that a
  marginal draft then resolves either way, not a systematic one. The decode and MTP bands are
  untouched, which is exactly what the md5 identity above shows.
* **Perplexity in the band.** `ninfer-perplexity --text` scores every window at `t = context`, so
  `--context 32` (stride 16, 542 windows) *is* the band, and the scoring rate is the proof each arm
  ran the route it claims: **49.0 tok/s against 147.3 (3.0x)**. Over the same 8,675 scored tokens:
  mean_nll 3.621945 (SIMT) against 3.622088 (QPN), i.e. **+0.0143% PPL** -- the same order as, and
  smaller than, the +0.0245% the fp16 prefill routes measured against the FP32 SIMT route.
  `--context 64` (outside the band; rate 203.5 against 201.6) moves +0.0018%, which bounds any
  leakage into it. Recorded because it wastes a cycle: `--context 16` aborts with "causal scoring
  returned a non-finite logprob" on the *shipping* route (QPN off), so a 16-token window is too short
  for this scorer independently of this work.

### Where the MTP money still is

The route does not lower `r`, which was the entire reason to port it; the honest statement is that
the MTP band is exactly where it was -- `r` = 0.37 at 4k / 0.30 at 16k against upstream's 0.135. What
this work does change is who owns that band's ceiling: the QPN sweep is a flat 47.5 ms at T = 6
against a 7.19 GB/~900 GB/s traffic floor of ~8 ms, so it is latency- and parallelism-bound rather
than bandwidth-bound either, and no configuration tried here (SPLITK 4 against 8, NACC 1 against 4,
one entry covered against two) beats the SIMT tile below T = 5. The "441-637 GB/s" the port was
chasing does not reproduce on this card and artifact as a bandwidth figure in the per-token bands;
what does reproduce is a flat cost curve, and that is worth the 2.1x it is worth on the lookup
verify. For `r` itself the lever is the one already recorded above: the per-verify-token body of the
SIMT tile kernel -- the `TILE_SHARE_ACT` probe prices its activation side at +23-34%, and the fp32
activation container is the multi-file change that collects it -- not the tensor cores.

## What was deliberately not ported

* The four author tensor-core schedules (`ternary_rowsplit_mma`, `_mma_small_t`, `_mma_wide_t`,
  `_mma_s8`), ~135 KB. Replaced by the two that exist here.
* bonsai's removal of `#ifdef NINFER_VOLTA_BUILD` around `kNvfp4TextPolicy`/`kFp8TextPolicy` (Volta
  forces `A16Only`), the `w8_gdn_input_workspace_bytes` return, the q4 fused-tensor-core workspace
  budget in `linear.cpp`, and the q4/q5 dispatch `workspace` parameter. Those are Volta work the Ada
  tree does not have; reverting them would break Q4/Q5.
* `text_context_impl.h`'s `kMtpLookupMaximumWidth` -> `kMaximumMtpDraftTokens + 1` bound change and
  the `mtp_lookup_replay_records` plan block. Unrelated to ternary.
* The `PTQ1_0_G128` fast paths. Written and measured (see "The PTQ1_0 tier" above), reverted: the
  format is issue-bound and 1.75-bit dense base-3 decodes for more instructions than it saves in
  bytes. `ternary_row_view()` and `launch_by_qtype<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom>` already
  keep PTQ1_0 loadable and numerically correct through the reference kernel.

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

### The fp32-activation experiment: bit-identical, and much slower

The probe above prices the activation side of the tile loop at +23-34%. The obvious way to collect
that is to stop converting: have the rotation emit the folded activation as **fp32** for the decode
band, so no GEMV widens bf16. The trick that makes it free of any numerical change is to store
`fp32(bf16(v))` -- round to bf16 exactly as today, then widen, which is precisely what the bf16
buffer held plus the exact widen the GEMV performed on load.

That property held exactly. With `NINFER_TERNARY_FP32_ACT` on and off, greedy ids are **identical in
all six comparisons** (two prompts x no-spec / K=1 / K=7, 40 tokens each) -- a useful confirmation
that the bf16 round trip in the pipeline is lossless, independent of the outcome below.

The throughput was not close:

| arm | bf16 activation | fp32 activation | delta |
|---|---:|---:|---:|
| no spec, prose | 43.8 | 41.8 | -5% |
| K=1 | 51.8 | 29.8 | -42% |
| K=3 | 42.0 | 19.5 | -54% |
| K=7 | 25.1 | 8.6 | -66% |
| K=7, repeat prompt (kT=16) | 107.7 | 8.5 | **-92%** |
| prefill 3412 (stays bf16 by construction) | 1.21k | 1.20k | unchanged |

**Why it collapsed -- and it is not activation traffic.** `cuobjdump -res-usage` on the same build,
for `<kT, kUnroll>` with the fp32 flag off and on. Nothing spills anywhere (STACK and LOCAL both 0):

| kernel | bf16 registers | fp32 registers | CTAs/SM (256 threads) |
|---|---:|---:|---|
| kT=2, u8 | 79 | 117 | 3 -> 2 |
| kT=4, u4 | **64** | **108** | **4 -> 2** |
| kT=4, u8 | 103 | 180 | 2 -> 1 |
| kT=8, u4 | 111 | 162 | 2 -> 1 |
| kT=16, u4 | 128 | **206** | **2 -> 1** |
| kT=16, u8 | 170 | 254 | 1 -> 1 |

An fp32 activation costs four registers per token instead of two, and the unroll keeps several tokens'
worth live at once, so the resident warp count halves -- and at kT=16 the shipped shape is the one
that loses a CTA. That is the measured cause of the collapse, and it matches its shape: the loss grows
with kT because the register footprint grows with kT.

**It also corrects the reading of the share-activation probe** (a first draft of this section blamed
activation traffic, which the numbers do not support). That probe removed kT-1 loads *and* their
converts, and it *reduced* the per-token footprint from kT tokens' worth to one, so its +23-34% is
consistent with instruction count and register pressure. This experiment kept the load count, removed
the converts, and doubled the footprint -- and lost, badly. **The verify kernel is bound by
instructions and registers, not by activation bandwidth**, and any further attempt has to reduce both
at once.

The one lever this document already carries that does exactly that is the fp16 group-dot, which was
shelved as "not taken": packing the four activations into two `half2` registers instead of four fp32
registers halves the per-token footprint, and `__hfma2` performs two multiply-accumulates per
instruction, so the dot's instruction count falls with it. It is a precision change -- each group of
128 weights would be summed in fp16 before joining the fp32 accumulator -- so it must be qualified
against the FP32 `block` oracle rather than assumed, and the bf16 -> fp16 conversion it needs may eat
part of the saving.

On the second question in this section, shared-memory staging is still the untried idea, but this
result removes its motivation: it was proposed to cut activation *traffic*, and traffic is not what
binds. It would add a staging loop and a barrier per K-chunk on top of an instruction-bound loop.

The code was reverted rather than parked behind a switch: it changes the rotation's output pointer
type and the workspace reservation, and it is a loss on every arm.

### The fp16 group-dot: the verify band's first real win since the unroll

The diagnosis above says the verify loop is bound by instructions *and* registers, so the lever has to
cut both. The fp16 group-dot does: the activation is kept as two `half2` registers instead of four
fp32 ones, and `__hfma2` performs two multiply-accumulates per instruction. The weights use the same
magic number the prefill kernel uses -- the bit pattern `0x6400|n` is exactly 1024+n as fp16, so one
`__hsub2` against 1025.0 returns four exact `(code-1)` values in two instructions, replacing the four
`I2F` plus masks the integer path needs.

It needs the activation in an fp16 container, which costs an env knob rather than a plumbing change
because fp16 is still two bytes: the workspace reservation is untouched, and only the decode band is
affected. `NINFER_TERNARY_FP16_ACT` selects the mode (`=1` the half2 dot, `=2` an fp16 container with
the original fp32 chain, unset or `0` today's bf16 path). Measured on prose and on the repeated-
sentence prompt, 200/300 generated tokens:

| arm | bf16 (mode 0) | **fp16 + half2 dot (mode 1)** | fp16 + fp32 dot (mode 2) |
|---|---:|---:|---:|
| K=1 (kT=2) | 51.8 | **56.2 (+8.5%)** | 42.5 (-18%) |
| K=3 (kT=4) | 42.1 | **44.4 (+5.5%)** | 32.2 (-24%) |
| K=7 (kT=8) | 25.2 | **26.6 (+5.6%)** | 20.2 (-20%) |
| K=7, repeat prompt (kT=16) | 108.0 | **121.8 (+12.8%)** | 84.2 (-22%) |

Mode 2 is the control that tells the two effects apart: it moves the activation into the same fp16
container but keeps the fp32 dot chain, and it *loses* 18-24%. So the container itself is a loss --
fp16 has a narrower exponent range than bf16 (5 bits against 8), so small rotated values fall into
fp16's subnormal range -- and the entire win is the half2 dot removing instructions. That is
consistent with the diagnosis and is the second independent confirmation of it.

Registers move only where there was room to move: kT=8 goes 111 -> 96, while kT=4 (64) and kT=16
(128) are unchanged, so occupancy is not what improved. The gain is instruction count.

**Numerics, and why this is opt-in.** The half2 dot rounds the group's four-term sum once to fp16
(products are exact, since a weight is -1, 0 or +1), so the logits shift in the last bits. Greedy ids
against the bf16 arm: K=7 on prose is identical over 40 tokens, K=1 differs in 19 of 40 positions
from position 21, K=3 in 29 of 40 from position 11 -- the same single-position fork this document
already records for the speculative-versus-non-speculative paths, i.e. a near-tie flipping and
everything after it changing with it. The MTP acceptance length drifts slightly down (1.58 -> 1.56 at
K=1, 1.93 -> 1.87 at K=3, 2.07 -> 1.98 at K=7) and is unchanged on the repeat prompt (13.11).

There is **no continuous measure that covers T = 2..16**: the perplexity harness runs the prefill
route, so the only quality evidence available for this band is the smoke test above. **The fp16 dot is
the default anyway**, and the trade is smaller than it sounds. This is a 2-bit-weight model: the
weights carry four levels, the activations arrive as bf16 (8 mantissa bits, 2^-8 = 4e-3 of relative
quantization noise), and the KV cache is int8. One rounding of a four-term sum whose inputs are
already only 8 bits deep costs 2^-11 = 5e-4 -- an order of magnitude *below* the noise the model
already carries. Alongside that, the logit change is of the same kind and magnitude this port already
accepts between its speculative and non-speculative paths, and the acceptance drift sits at the
rounding level rather than being systematic. `NINFER_TERNARY_FP16_ACT=0` restores the bf16 path, and `=2` selects the fp32-dot control
that the table above shows losing.

### What the model's own precision is, which is the real justification

The activation container is worth stating against what the artifact already carries. The ternary
linear tensors are `PQ2_0_G128`: **2 bits per weight** in a 32-byte group of 128, plus one fp16 scale
per group, i.e. 2 + 16/128 = **2.125 bits = 0.266 bytes per weight**. Four code levels exist but the
quantization is effectively ternary -- the code histogram measures approximately 33 / 32 / 33 / 1
percent, so the fourth level is essentially unused.

The precision ladder is therefore: 2-bit weights (four levels, three used) < int8 KV cache < bf16
activations (8 mantissa bits, 2^-8) < the fp16 group sum (11 mantissa bits, 2^-11). **The rounding this
section adds is the finest step in the pipeline** -- eight times finer than the bf16 activation
quantization that feeds it, and the products going into it are exact because a weight is -1, 0 or +1.
That is the argument that decided the default, and the T=1 measurement below agrees with it.

### The T = 1 path

The same half2 dot reaches the headline decode. It routes a single token through the tile kernel at
`kT = 1` instead of the dedicated T = 1 kernel (which is bf16-only and pinned at `kMinBlocks = 4`):

| arm | decode | registers | CTAs/SM | greedy ids |
|---|---:|---:|---:|---|
| dedicated T = 1 kernel, bf16 (`FP16_TERNARY_FP16_ACT=0`) | 43.7 | 62 | 4 | reference |
| **tile kernel, kT = 1, fp16 half2 (default)** | **44.7** | **42** | **6** | **identical** |

+2.3%, and at T = 1 the greedy ids are bit-identical, unlike the verify band. The unroll does not
matter at `kT = 1` (1/2/4/8/16 all measure 44.7-44.8) and the occupancy gain from 32 to 48 resident
warps buys nothing either, which says the single-token step is not instruction- or occupancy-bound:
it is waiting on the code stream, as the `codealu` probe (removing that one load alone is worth 17%)
already showed.

Prefill is untouched by construction and measured so: 3412 tokens at 1.20k tok/s with the flag off and
1.21k with it on.
