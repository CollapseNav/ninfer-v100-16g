# Session report — PQ2 wide-verify (QPN) route and the T = 1 decode step

Card: Tesla V100-SXM2-16GB, driver 580.178.04. Artifact `bonsai2_27b_swift_pq2.ninfer`
(`qwen3.8-27b/groupwise-int`), `--kv-dtype int8`, `NINFER_TERNARY_ROTATE_SPLIT=16` on every timing
run. All A/B pairs are same-batch (MTP drifts +/-4% across batches). Commands are at the end.

This is the consolidated record for the two rounds of 2026-10-01. Round 1 is also written up in
`docs/ternary-port.md` (section "PQ2 tensor-core (QPN) route: resumed, bounded, and kept for the
wide verify"); round 2 is new here and its negative results are the useful part.

---

## Round 1 — PQ2 tensor-core (QPN) route: **kept, gated to 6 <= T <= 32**

**Verdict.** The route was ported to make a *verify token* cheap and it does not do that: the
tensor-core weight sweep costs 39.2 ms at T = 2 and 46.8 ms at T = 16, i.e. it is nearly **flat in
T**, while the SIMT tile kernel's round grows 26.6 -> 98.1 ms over the same range. The two lines
cross at **T ~ 4.3**, so everything an MTP round uses (K = 1, K = 2 -> T = 2, 3) loses, and the wide
verify wins big. It is therefore gated to `6 <= T <= 32`; the decode and MTP bands are untouched,
which the md5 regression proves byte for byte.

| verify width T | 2 | 3 | 4 | 5 | 6 | 8 | 16 (lookup) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIMT tile round, ms | 26.6 | 33.9 | 41.2 | 48.6 | 58.0 | 74.1 | 98.1 |
| QPN round, ms | **39.2** | **41.1** | **43.0** | **44.9** | 47.5 | 52.0 | **46.8** |
| ratio | 1.47 | 1.21 | 1.04 | 0.92 | 0.82 | 0.70 | 0.48 |

End-to-end, `NINFER_TERNARY_QPN=0` against `=1`, 128 generated tokens:

| arm | QPN off | QPN on | note |
|---|---:|---:|---|
| no-spec decode (T = 1) | 51.4 | 51.4 | identical speed and text (gate excludes) |
| real_task K = 1 (T = 2) | 62.0 | 61.9 | gate excludes |
| real_task K = 2 (T = 3) | 54.7 | 54.6 | gate excludes |
| real_task K = 7 (T = 8) | 37.0 | **49.8** | round 74.1 -> 52.0 ms |
| real_code 4.4k K = 7 (T = 8) | 44.9 | **63.3** | acceptance 3.50 equal, text identical |
| **lookup10 K = 7 (T = 16)** | 123.7 | **234.8** | acceptance 12.11 equal, text identical |
| **lookup16 K = 7 (T = 16)** | 126.7 | **246.6** | acceptance 12.70 equal |
| prefill, 4.4k prompt | 1.14k | 1.14k | unchanged |

Why the ported route was losing, in the order the fixes mattered:

1. **Just over half the ternary weights could not reach the route at all.** A shape probe showed the
   linears split into two disjoint sets: `n = 5120/34816/5120/248320` at `k = 6144/5120/17408/5120`
   through `ternary_dispatch`, and `n = 4096/6144/1024` at `k = 5120` through
   `ternary_dispatch_basis_strided` (attention input projections, GDN input projection, SwiGLU pair),
   which had **no QPN gate**. Covering both took the lookup win from +61% to +90%.
2. **SPLITK was 4 where upstream's own NVFP4 launcher calls 8 the production floor.** One CTA covers
   32 output rows, so the grid is `n/32` (160 CTAs at n = 5120) and SPLITK = 4 left 8 warps per SM.
3. **The reader took the plane four bytes at a time.** Vectorising it to two `uint4` per lane was
   worth only ~4% of the 35% deficit the parking note blamed it for; the row-major plane is already
   contiguous per lane.
4. **The load-time prepack is not needed at all**, which is what dissolved the "structural conflict"
   that had parked the route: the permutation existed to make a warp's fetch contiguous, and the
   row-major plane already is contiguous per lane, so making it work would have cost an index rewrite
   in the fused MMA (T = 32..255), the CUTLASS dequant pass (T >= 256) and the block GEMV
   (T = 5..31) to buy nothing. `ternary_prepack_qpn` and the dispatcher's guard stay in the tree,
   uncalled, as the record.

Numerics. The 96-token greedy md5 against `/root/wt/base.out` is **IDENTICAL** with the route enabled
by default and with `NINFER_TERNARY_QPN=0`. For the band itself, two checks: output text is
byte-identical on both lookup fixtures and on `real_code` K = 4/K = 7, and
`ninfer-perplexity --text --context 32` (542 windows; the scoring rate, 49.0 against 147.3 tok/s, is
the proof the two arms ran different routes) measures **+0.0143% PPL** for the band, against the
+0.0245% the fp16 prefill routes cost against FP32 SIMT. One arm does flip: `real_task` K = 7 (T = 8),
acceptance 2.74 -> 2.59 at 128 tokens — a last-bit accumulation-order difference that a marginal
draft then resolves either way.

This work does **not** lower `r`: the MTP band is exactly where it was (`r` = 0.37 at 4k / 0.30 at
16k against upstream's 0.135), and the MTP windows stay on the SIMT route.

---

## Round 2 — the T = 1 decode step (51.5 t/s baseline), priced component by component

### What the step is made of (nvprof, no-spec decode of real_task, 64 tokens)

`nvprof --print-gpu-trace`, decode-only window. nvprof inflates the tiny kernels by ~2x (recorded in
`ternary-port.md`), so read the small rows as upper bounds; the per-token column is at nvprof's 46.3
t/s, which scales to ~0.89x at the real 51.5 t/s.

| kernel family | µs/call | calls/token | ms/token | share |
|---|---:|---:|---:|---:|
| `ternary_pq2_gemv_tile_kernel` (T = 1 GEMV) | 39.4 | 413 | **16.3** | **75%** |
| `ternary_rotation` | 5.7 | 270 | 1.54 | 7% |
| `rmsnorm_cta` / `rmsnorm_warp` | 5-6 | 78 | 1.15 | 5% |
| elementwise (`silu_and_mul`, `residual_add`, ...) | 3-6 | 234 | 0.99 | 4.6% |
| `causal_attention_small_t_*` | 20-33 | 12 | 0.71 | 3.3% |
| `gdn` (recurrent / gating / conv) | 4-10 | 96 | 0.55 | 2.6% |

**The T = 1 decode is one kernel plus a tail of two hundred tiny latency-bound ones.** That tail is
where the structural headroom is, and it is also where the least has been tried.

### The T = 1 GEMV, measured at the instruction level

* Achieved weight bandwidth: **444 GB/s** summed over the model (7.19 GB in 16.3 ms), 390-500 GB/s
  per shape, against a 882 GB/s flat `uint4` ceiling measured by this tree's own probe.
  Per shape, from the trace's launch geometry: n = 34816/k = 5120 -> 476 GB/s; n = 248320 (LM head)
  -> 498; n = 6144 -> 389; n = 4096 -> 350; **n = 1024 -> 184** (grid 128 = 1.6 CTAs/SM, i.e. the
  small-output layers are occupancy-starved and split-K is the only fix there).
* SASS opcode histogram of the shipped instantiation (`kT=1, unroll=8, fp16 container`, 321
  instructions for 8 groups, loop body ~29/group):
  `LOP3 76 (23.7%) · IMAD 46 · HADD2 45 · LDG 27 · IADD3 25 · SHF 16 · FADD 14 · HMUL2/HFMA2/FFMA
  27 · SHFL 6` -- i.e. **the 2-bit -> fp16 decode is ~15 instructions per 32-byte group (52% of the
  body)** and address arithmetic another ~7.
* That puts the kernel at **82% of the pure issue ceiling**: 29 instructions per 34 weight bytes, and
  4.9e11 issue slots/s, allow 575 GB/s where 444 is measured.

### Component prices, with the probes used correctly

`NINFER_TERNARY_GEMV_PROBE` takes **text** (`noact|nocode|noscale|codealu`), not a number — passing
numbers silently runs the baseline, which cost one batch. The tile probe instantiates at depth 4, so
the reference arm is `TILE_UNROLL=4` (46.9 t/s), not the shipped depth 8.

| arm | decode t/s | vs u4 | what it removes |
|---|---:|---:|---|
| `TILE_UNROLL=4` (probe reference) | 46.9 | -- | -- |
| `noact` | 49.0 | **+4.5%** | activation load + convert (values folded to a constant) |
| `codealu` | 57.8 | **+23%** | the code LDG, decode kept alive from ALU |
| `nocode` | 80.3 | **+71%** | code LDG + decode + the dependent dot |
| `noscale` | 50.2 | **+7%** | the scale load |
| `ROTATE_PROBE=copy` (shipped depth) | 52.7 | +2.3% | the Hadamard transform only, same launch and traffic |
| `FP16_ACT=0` (bf16 container) | 45.6 | **-11.5%** | the fp16 half2 dot (so the container is worth +11.5%) |
| `HADAMARD=0` | 47.7 | -7.4% | *not* a valid rotation probe: it also drops the fp16 container |

So the code load (23%) and the decode (~30%) dominate, the activation is nearly free, and the scale
is worth 7%.

### Five levers tried, five negatives

Every one is default-off in the tree now, with its measurement in the comment.

| lever | implementation | result | reading |
|---|---|---|---|
| decode via a table | 256-entry `uint2` table in **shared** memory, one `LDS.64` replacing ~15 ALU; SASS body 29 -> 22 instructions (`LOP3` 76 -> 12) | **39.8 t/s, -23%** | cutting 25% of the instructions makes it *slower*: the loop is latency-bound on the code-load chain, so a dependent load costs more than the arithmetic it removes |
| global table (pre-existing arm) | 2 KB `__device__` table | 33.6 t/s, -28% | reproduces the earlier refutation |
| register cap / occupancy | `__launch_bounds__` min-blocks 4/6/8 on the tile kernel (60 regs / 32 warps of 64 today) | 51.6 / 41.3 / 35.8 t/s | the same lever that gave the *dedicated* T = 1 GEMV +4.8% is flat at 4 (cap above the 60 in use) and loses above; the kernel is not warp-limited |
| wider lanes | new `ternary_pq2_gemv_wide1_kernel`: 2 code bytes + 16 activation bytes (one `LDG.16` + one `LDG.128`) per lane per iteration, covering two groups, ~25% fewer instructions per byte and 2x the weight bytes per load instruction | 42.7 / 46.1 / **50.9** / 48.3 t/s for unroll 1/2/4/8 | best case -1.3%; halving the loop iterations also halves the code loads in flight, which is what the latency-bound loop needs |
| unroll depth | `TILE_UNROLL` 4 / 8 / 16 | 46.9 / **51.5** / 35.4 | 8 is still the peak (117 registers at 16) |
| upstream comparison | the Ada ternary sources this port came from (`ternary/` in the workspace) | -- | upstream's T = 1 kernel is **behind** ours: no unroll pragma, one bf16 `__bfloat1622float2` pair per group, no vectorised activation load, 422 GB/s. There is nothing to import here |

### Knobs that are silently inert on the T = 1 path

`1df45be1` moved T = 1 off the dedicated `ternary_pq2_gemv_w_kernel` and onto the tile kernel at
`kT = 1`, so **`NINFER_TERNARY_GEMV_ROWS`, `GEMV_MINBLOCKS`, `GEMV_WARPS` and `GEMV_NOBIAS` do
nothing at T = 1** (they are template parameters of the w_kernel). Measured: `GEMV_ROWS=2` and
`GEMV_MINBLOCKS=8` are bit-for-bit identical in speed to the default. `GEMV_PROBE` and
`TILE_UNROLL`/`TILE_WIDE`/`FP16_ACT`/`ROTATE_*` *are* live. Worth knowing before pricing anything
with the wrong knob -- one batch was spent on exactly that.

### Where the remaining decode headroom is

1. **The T = 1 GEMV is at a local optimum for this design.** Five orthogonal levers (instruction
   count, occupancy, unroll, bytes-per-load, container precision) all land at or below the shipped
   configuration, and the loop runs at 82% of its own issue ceiling. A further win needs a different
   schedule, not a knob: the only shape that has not been tried is a **block-cooperative code-plane
   stage** -- the whole block loads a group's code for its 8 rows with coalesced 128-byte reads into
   shared memory in a layout where each lane's bytes are contiguous, then decodes from shared memory.
   That is the QPN prepack idea applied to the SIMT GEMV, and it is the only route that removes the
   code LDG from the per-warp chain instead of lengthening it. Expected size: the code load is worth
   23%, so a perfect version of it is bounded by ~+20% on the GEMV, i.e. ~+15% on the step.
2. **The other 25% is 240 tiny latency-bound kernels per token.** Each is already minimal in bytes
   (a norm over 10-34 KB should be ~1 µs, and measures 5-6 µs), so the lever is not their arithmetic
   but their launch geometry and their serialisation: more blocks for the small ones, and merging or
   co-scheduling independent ones (the four attention input projections and the SwiGLU pair already
   share an activation). Unmeasured; this is the largest untouched block of the step.
3. **Occupancy-starved small-output layers.** n = 1024 runs at 184 GB/s on a 128-CTA grid (1.6
   CTAs/SM). Split-K with a second-pass reduction would fix that shape, but it is ~1% of the step
   (0.24 ms/token), so it is not where the money is.
4. **Something not on this axis at all**: serving concurrency (`--max-concurrency` is unset, so every
   decode step streams the full 7.19 GB for one sequence). It does not raise single-request decode
   speed, but it is the only remaining lever with a large multiplier on aggregate throughput, and it
   is unchanged by this work.

---

## Reproduce

```bash
# Round 1, the band A/B (interleaved arms, same batch)
bash /root/validate_qpn.sh

# Round 2, the step budget and the launch geometry
bash /root/trace_t1.sh
python3 /root/nvp_geom.py /root/nvp_t1/t1.txt        # kernel x grid x block x mean µs

# Round 2, component prices (TEXT-valued probe env)
bash /root/probe_t1b.sh

# Round 2, the five levers
bash /root/sweep_dec.sh      # shared table + register caps
bash /root/sweep_wide1.sh    # wide-lane kernel
bash /root/verify_default.sh # md5 + the shipped 51.5 t/s

# SASS instruction counts
python3 /root/sass_ops.py <obj.o> tile_kernelILi1ELi8ELb0ELi1ELi0ELi0ELi1E
python3 /root/sass_dump.py <obj.o> tile_kernelILi1ELi8ELb0ELi1ELi0ELi0ELi1E 40
```

Raw data on the host: `/root/ninfer_ab/qpn3/{v,f,x,head}` (round 1), `/root/ninfer_ab/dec{1,2,3,4,5}`
and `/root/nvp_t1/t1.txt` (round 2), `/root/ppl_qpn/` (the perplexity band check).

---

## Round 3 — the tail: ~1300 kernel launches per token, and it is ~all fixed cost

Per-token launch counts in the decode window (85448 kernels in the capture, ~1300/token):

| family | launches/token | kernels |
|---|---:|---|
| ternary GEMV (T = 1) | 413 | `ternary_pq2_gemv_tile_kernel` |
| ternary rotation | 261 | `ternary_rotation` (one warp per D1024 block) |
| elementwise | 234 | `residual_add_bf16x8` 130, `silu_and_mul` 65, `sigmoid_gate_mul` 16, `rope_fixed` 16 |
| norm | 180 | `rmsnorm_cta_bf16x2` 131, `rmsnorm_warp_bf16x2` 49 |
| GDN | ~195 | `recurrent_batch_update` 49, `gdn_gating_proj` 49, `gdn_projected_conv` 49, `recurrent_bf16_direct` 1.5 |
| attention | 12 | `causal_attention_small_t_tc_volta_partial_i8` + reduce |

**The proof that the tail is launch cost and not data cost**: the same capture holds one prefill
(85 tokens of data) and 64 decode steps (1 token), so every kernel appears at both sizes.

| kernel | decode | decode µs | prefill | prefill µs | data ratio |
|---|---|---:|---|---:|---:|
| `rmsnorm_cta_bf16x2` | grid(1), 256 thr | 6.04 | grid(83), 256 thr | 5.27 | **83x** |
| `residual_add_bf16x8` | grid(3), 256 thr | 3.42 | grid(208), 256 thr | 6.44 | 70x |
| `residual_add_bf16x8` | -- | -- | grid(5), 256 thr | 2.78 | floor |
| `rmsnorm_warp_bf16x2` | grid(6), 128 thr | 3.71 | grid(498), 128 thr | 4.84 | **83x** |
| `rmsnorm_warp_bf16x2` | grid(1), 128 thr | 2.24 | grid(83), 128 thr | 2.39 | floor |
| `ternary_rotation` | grid(5), 128 thr | 4.89 | grid(63), 256 thr | 12.37 | ~13x |
| `rope_fixed` | grid(1), 896 thr | 3.95 | grid(83), 256 thr | 4.79 | 83x |

83x the data costs 0.87x the time, and 498 CTAs cost the same as 83: **these kernels' marginal
data cost is ~zero and their whole duration is the launch ramp plus one memory round trip plus the
block reduction.** The floor measured at the smallest geometries is 1.9-2.8 µs (nvprof; the
documented nvprof inflation for this class is ~2x, so ~1-1.4 µs real). Tuning what a tail kernel
computes can therefore not buy anything: at `grid(1)` it is already doing a single round trip.

What this implies for decode speed:

* **The only lever on the tail is the number of launches.** ~890 of the 1300 launches per token are
  tail kernels at a ~1-1.4 µs floor, i.e. ~0.9-1.2 ms/token that cannot be removed without removing
  the launches, on top of ~1.4 ms of above-floor cost that a good fusion could also reach. The
  addressable prize is therefore **~1-1.5 ms of the 19.4 ms step, 5-8%**, and it is collected by
  epilogue fusion: `residual_add` (130/token) and `silu_and_mul` (65/token) are elementwise
  consumers of a tensor the preceding GEMV just produced, so each can in principle be folded into
  that producer's epilogue. Cost: a real refactor across the model impl, every linear op growing an
  optional epilogue.
* **A naive fusion loses.** This tree has tried exactly one (norm+rotation, bit-identical) and it
  measured -2.0/-2.6%: the fused kernel serialised two phases behind a barrier and paid more than
  it saved. A fusion only pays if it deletes a launch without adding a synchronisation point.
* **Overlapping the tail with the GEMV is not available.** The decode is a strict chain
  (norm -> rotate -> projections -> attention -> MLP -> residual), and the only siblings are the
  attention input projections, which this engine already fuses into one op. The GEMV also runs at
  82% of its own issue ceiling, so there are few issue slots to lend.

So round 3 closes item 1 with a negative: the tail is not a kernel-tuning target. It is a launch
count target, worth 5-8%, and it needs an engine refactor rather than a kernel rewrite — which puts
it behind the other open item in value-per-risk, the block-cooperative code-plane stage for the
GEMV itself (worth up to ~15% of the step, one kernel, no engine change).

---

## Round 4 — staging the T = 1 code plane per warp: **+3.9% decode, bit-identical**

The shipped T = 1 GEMV reads one code byte per lane per group, so eight groups cost the warp eight
32-byte loads (one sector each), each a separate DRAM latency sitting on the decode's dependency
chain, plus eight 16-bit scale loads. This round stages eight groups at a time instead:

* one 64-bit load per lane covers the block's whole 256-byte code span (eight sectors in a single
  instruction) into a **private per-warp shared buffer**;
* the block's eight scales come from one broadcast 128-bit load instead of eight 16-bit ones;
* the **next** block's span is issued before the current block's inner loop, so its DRAM latency
  overlaps eight groups of decode and dot (software pipeline).

Only `__syncwarp()` is involved: the eight warps of a block keep private buffers, so no block barrier
is added -- which matters, because the one fusion this tree has tried (norm+rotation, bit-identical)
lost 2.0-2.6% to exactly that.

| arm | real_task, 96 tokens | 4.4k real_code | lookup10 K = 7 |
|---|---:|---:|---:|
| shipped (no staging) | 51.5 / 51.6 / 51.5 | 49.7 | 123.6 |
| staged load, no pipeline | 52.6 / 52.7 | -- | -- |
| **staged + software pipeline** | **53.5 / 53.5** | **51.7** | 123.4 |

**+3.9%** on the decode step, and the output is byte-identical on all three workloads; the 96-token
greedy md5 against `/root/wt/base.out` is IDENTICAL with staging on by default (three independent
runs) and with `NINFER_TERNARY_TILE_STAGE=0`. That is expected, because the staging changes only how
the bytes arrive: same bytes, same lane->k mapping, same accumulation order.

Why this worked where the round-2 levers did not: it *removes* load instructions and replaces eight
small ones with one big one, while the shared-memory decode table *added* a dependent load to the
same chain. Same shared memory, opposite direction.

The bit-identity gate earned its keep. The first pipeline draft baked a fixed offset into the
prefetch (it re-read block 1 every iteration), which corrupted the output *and* made those loads
L1-resident -- it measured +7.2% on wrong data. The text comparison caught it; the speed number
alone would have shipped a broken kernel. A second false alarm was this document's own harness: a
`docker run` called with an empty argument failed with "invalid reference format" and left a 0-byte
output, which read as an md5 mismatch until the file size was checked.

Still untried in this direction: staging 16 groups per block (half the `__syncwarp()` pairs, 4 KB of
shared memory per block) and double-buffering to drop the second `__syncwarp()` of each iteration.
Both are small edits to this kernel, and the +2.2% -> +3.9% step from adding the pipeline says the
remaining latency is still worth something.

### Final figures (five runs, same batch ordering)

| arm | no-spec decode, real_task 128 tokens |
|---|---:|
| shipped default (staged) | 53.4 / 53.4 / 53.5 |
| `NINFER_TERNARY_TILE_STAGE=0` | 51.5 / 51.5 |

+3.7-3.9%, reproducible across three batches, with byte-identical output. The MTP K = 1 arm is
unchanged by construction rather than by measurement: the staged kernel is selected only for
`tokens == 1`, and a K = 1 verify round runs every target-model linear at T = 2, so it stays on the
tile kernel at kT = 2 (61.9-62.0 t/s, as recorded in round 1).

---

## Round 5 — the staged decode extended to the T = 2 verify, and bounded there

The staged kernel is now templated on kT and serves **T = 1..2**. Measured over the whole band the
tile kernel still owns (QPN took T = 6..32, so that band is T = 1..5), same batch, two repetitions
each, `NINFER_TERNARY_TILE_STAGE=0` against unset:

| arm | T | no staging | staged | delta |
|---|---:|---:|---:|---:|
| no-spec decode, real_task | 1 | 51.4 / 51.5 | **54.2 / 54.2** | **+5.3%** |
| MTP K = 1 (the verify) | 2 | 61.9 / 61.9 | **64.0 / 63.8** | **+3.3%** |
| MTP K = 2 | 3 | 54.6 / 54.6 | 52.9 / 52.9 | -3.1% |
| MTP K = 3 | 4 | 49.2 / 49.2 | 48.1 / 48.0 | -2.4% |

Both repetitions agree to 0.1 t/s, and the no-staging arms reproduce the historical baselines
(51.4-51.6 / 61.9-62.0), which is the harness cross-check. **The band stops at T = 2**: staging
removes the code load from the critical path, and at kT = 1 that load is the entire weight stream,
while above kT = 2 the per-token activation work (which staging does not touch) dominates and the
extra shared-memory hop plus the registers cost more than the code load saves.

Final verification of the shipped configuration (staging at T = 1 and 2 only, same batch):

| arm | no staging | staged |
|---|---:|---:|
| no-spec decode, real_task 128 tokens | 51.5 | **54.3 (+5.4%)** |
| MTP K = 1 | 61.9 | **63.9 (+3.2%)** |
| MTP K = 1, 4.4k real_code prompt | 66.4 | **68.5 (+3.2%)** |
| MTP K = 2 (not staged) | 54.6 | 54.6 |

Acceptance length is identical in every pair (1.65 / 1.85 / 1.94), and the output text is
byte-identical in all four comparisons. The 96-token greedy md5 against `/root/wt/base.out` is
IDENTICAL with staging on by default and with `NINFER_TERNARY_TILE_STAGE=0`.

Against the numbers this session started from, the decode step is **51.1 -> 54.3 t/s** and the MTP
K = 1 arm -- the window a real task actually uses -- is **61.9 -> 63.9 t/s**, both bit-identical to
the path that produced the recorded baseline.

---

## Round 6 — the QPN default was dead, plus two more knob verdicts

**A bug that this session's own protocol had been hiding.** `ternary_qpn_enabled()` shipped as
`env != nullptr && env != "0"`, i.e. the route ran **only when a value was passed**, while its own
header comment, the task that commissioned the route, and every document in this tree say the
opposite ("`NINFER_TERNARY_QPN=0` disables"). Every A/B in rounds 1-5 passed the variable explicitly
(0 or 1), so those comparisons were valid and the +90% was real -- but the *default* path never took
the route, and the shipped lookup number stayed at the SIMT 123.7 t/s. It was found by re-running the
lookup arm with no `NINFER_TERNARY_QPN` at all, while chasing an unrelated knob.

| lookup10 K = 7 | t/s |
|---|---:|
| `NINFER_TERNARY_QPN=1` | 235.1 |
| `NINFER_TERNARY_QPN=0` | 123.7 |
| unset, **before** the fix | **123.8** |
| unset, **after** the fix | **235.5** |

Fixed to `env == nullptr || env != "0"`. The 96-token greedy md5 against `/root/wt/base.out` is
still IDENTICAL with the default now on, because the gate excludes T < 6 and the whole baseline path
(150-token prefill, T = 1 decode, T = 2 tail) is below it. **This is the largest single number the
session moved: the context-lookup decode is 123.7 -> 235.5 t/s by default**, not only under an
explicit switch. Lesson worth keeping: a route whose A/B always passes the switch explicitly is not
evidence that the switch's default works.

**NACC at kTiles = 2** -- the T = 9..16 lookup verify, where the mma count doubles with the tile and
the accumulator chain is four deep at NACC = 1:

| arm | lookup10 K = 7 |
|---|---:|
| NACC = 1 (shipped) | 235.5 |
| **NACC = 2** | **237.5 (+0.85%)** |
| NACC = 4 | 227.6 (-3.4%) |

Output text identical at all three, but NACC reassociates the fp32 accumulation, so adopting it would
require re-running the band's perplexity check; at +0.85% it stays a measured, default-off knob
(`NINFER_TERNARY_QPN_NACC`). Note the first attempt at this A/B was void -- it ran before the default
fix, so all three arms were the SIMT route at 123.7 t/s.

**Prefetch depth 2 for the staged decode: refuted.** Same batch, two repetitions: no-spec 54.2/54.3
-> 52.8/52.7 and MTP K = 1 64.0/64.0 -> 62.4/62.4, i.e. **-2.7%**, with identical output. One block
of prefetch already covers the DRAM latency; the second block only costs registers. The default stays
1 (`NINFER_TERNARY_STAGE_DEPTH`).

---

## Round 7 — NACC = 2 at kTiles = 2, enabled after the perplexity check

NACC round-robins the four mma of one 16-k unit into independent accumulators, breaking the RAW chain
on the accumulator. At kTiles = 2 -- T = 9..16, the context-lookup verify, where the mma count doubles
with the tile -- two accumulators win and four lose. Same batch:

| arm | lookup10 K = 7 | lookup16 K = 7 |
|---|---:|---:|
| NACC = 1 (previous shipped) | 234.9 | 247.4 |
| **NACC = 2 (now shipped)** | **237.6 (+1.1%)** | **249.8 (+1.0%)** |
| NACC = 4 | 227.4 (-3.2%) | -- |
| no-spec decode step | 54.2 both | -- |

Acceptance length identical (12.11 / 12.70) and the output text identical between NACC 1 and 2 on both
fixtures.

Because NACC reassociates the fp32 accumulation, the continuous check is what decides it.
`ninfer-perplexity --text --context 16 --stride 8` is the only window plan that lands every forward on
kTiles = 2; with both arms on the QPN route over 8,675 scored tokens:

| arm | mean_nll | PPL |
|---|---:|---:|
| NACC = 1 | 4.165766 | 64.442012 |
| NACC = 2 | 4.166126 | 64.465258 |

**+0.036% PPL** -- 2.5x the QPN band's own deviation from SIMT (+0.0143%) and 1.5x the fp16 operand
margin this tree accepted for its prefill routes (+0.0245%) -- bought for about +1% on one path. That
is a trade, not a free win, so it is recorded as one: `NINFER_TERNARY_QPN_NACC=1` restores the tighter
numbers. The 96-token greedy md5 against `/root/wt/base.out` is IDENTICAL with the new default and
with `=1`, since the knob is above the gate.

---

## Round 8 — the tail's epilogue fusion: reconnaissance, real ceiling, and why it was not attempted

Item ② was the last open direction: fold the tail's elementwise consumers into their producers'
epilogues, which the launch-count arithmetic priced at 5-8% of the step. Reconnaissance on the
ternary path changes that estimate and shows the work is not a wrapper change.

* The two biggest elementwise consumers are `residual_add` (130 launches/token, 3.42 us each) and
  `silu_and_mul` (65/token, 5.36 us).
* `ops::linear_add` **looks** like the fused producer+add, and it is not: on the ternary path
  `src/ops/wrapper/linear_add.cpp` composes `ternary_dispatch` into a scratch tensor and then calls
  `residual_add`, with the comment "There is no fused ternary residual kernel, so compose it from the
  shared ternary GEMM and the existing elementwise add". The model impl's one adjacent call site
  (`variant.cpp`: `ops::linear(down_proj)` then `ops::residual_add`) therefore cannot be redirected
  to it for any gain.
* So the fusion has to be an epilogue inside the kernels. Concretely:
  1. a second ternary entry point taking the residual tensor (`ternary_dispatch_add`), with the same
     shape gate and the same staged / tile / QPN / CUTLASS cascade;
  2. one bf16 load and one add per output element in the epilogue of the kernels that serve the
     decode and verify band (the staged kernel at T = 1..2, the tile kernel at T = 3..5) -- free
     against the ~39 us of weight streaming each call already pays;
  3. `linear_add`'s ternary branch rewritten to call it.
* **Real ceiling: +0.8-1%, not 5-8%.** Only the ~130 residual adds sit next to a producer; the
  tail's other ~760 launches are rotations, norms and GDN stages with no producer to fold into. At
  the measured fixed cost of 1-1.5 us per removed launch that is 0.15-0.2 ms of a 19.4 ms step.
* **Numerics: not bit-identical.** Today the linear's bf16 output is rounded once and then added to
  the residual; a fused epilogue would add in fp32 before the single bf16 rounding. That is strictly
  more accurate, but it moves greedy ids, so it needs the md5/PPL discipline and either a re-recorded
  baseline or a gate.

Not attempted. Four files plus two validation runs for a measured ceiling an order of magnitude below
the two changes that did move this session (the staged code plane, +6% decode; the QPN default fix,
+90% on the lookup path). Recorded with its cost so the next session can decide on numbers rather
than on the 5-8% estimate that the launch-count arithmetic suggested.

---

## Round 9 — the fused residual epilogue: **+1.8-2.0% decode, bit-identical**

Round 8 left item ② unbuilt on a ceiling of +0.8-1%. That ceiling was an estimate from launch
arithmetic; this round re-priced it before writing any kernel, and the estimate was wrong by a factor
of two.

### The price, measured first

`ops::residual_add` already carried a probe (`NINFER_TERNARY_PROBE_SKIP_RESIDUAL=1`, "do not launch
this kernel at all"), so the ceiling was one same-batch A/B away. Interleaved arms, two repetitions,
`real_task`, no spec:

| arm | run 1 | run 2 |
|---|---:|---:|
| control | 54.2 | 54.3 |
| `NINFER_TERNARY_PROBE_SKIP_RESIDUAL=1` | **55.3** | **55.3** |

**+1.9%**, against round 8's +0.8-1% and at the top of round 3's "5-8% for the whole tail" band. The
MTP arm of the same probe is unusable as a price: skipping the add corrupts the verify logits, so the
acceptance length collapses from 1.65 to 1.00 tok/round and the arm measures a different amount of
work (39.3 t/s). Recorded so nobody re-runs it as a speed arm.

### What was built

`gemv_store<kAddResidual>` in `ternary_rowsplit_gemv.cuh`, used by the two kernels that serve the
shipped decode/verify band: the warp-staged kernel at T = 1..2 (depth 1) and the small-tile kernel at
T = 3..5. `ops::linear_add`'s ternary branch now tries `ternary_dispatch_add` first and composes only
when it declines. Everything outside the band -- prefill, the QPN 6..32 verify, the MTP head's own
`mtp_post_mixer` (one launch per draft forward, not worth a workspace change) -- is untouched.

**The arithmetic is the one round 8 did not consider, and it is why this was cheap.** Round 8 assumed
the epilogue would add the fp32 accumulator to the residual: strictly more accurate, but it moves
greedy ids, so it needed a re-recorded md5 baseline plus a perplexity run -- which is the cost that
made a +1% change a bad trade. Instead the epilogue rounds the projection to bf16 *first* and adds in
fp32 with a second rounding, reproducing the composed route's double rounding exactly. The change is
therefore **bit-identical**, and the whole numerical-qualification budget disappears: the 96-token
greedy md5 against `/root/wt/base.out` is IDENTICAL with the arm on and off, and the generated text is
byte-identical on all eight fixtures.

### Proof that it is actually running

A route that is never taken is trivially bit-identical, and that is exactly what the first build
measured: **0.0% on every arm with a passing md5 gate.** The bug was the gate itself --
`ternary_pq2_gemv_add_admits` checked for the fp16 activation container, and `ternary_dispatch_add`
called it on the *raw* bf16 activation, so it always declined and the composed route ran. Recorded
because "identical output, identical speed" is an easy state to mistake for a negative result; the
way out is to check that the kernel actually ran:

| evidence | composed | fused |
|---|---:|---:|
| `ternary_pq2_gemv_stage_kernel<..., bool=1>` calls in a 64-step decode trace | 0 | 16,896 |
| `residual_add_bf16x8` launches, whole trace | 8,576 | 128 |
| -- of those, at the decode geometry grid(3) | 8,320 | **0** |
| -- of those, prefill geometry grid(208) | 128 | 128 |

The 128 that remain are the prefill's; a decode step now issues **zero** `residual_add` launches.

### A/B, same batch, interleaved arms

Two repetitions of the full band, `NINFER_TERNARY_FUSED_RESIDUAL=0` as the control:

| load | composed | fused | delta |
|---|---:|---:|---:|
| T = 1 decode, `real_task` | 54.2 / 54.2 | **55.2 / 55.3** | **+1.9%** |
| T = 1 decode, `real_code` | 52.3 / 52.3 | **53.2 / 53.2** | **+1.7%** |
| T = 1 decode, `prose4k` | 50.6 / 50.7 | **51.5 / 51.5** | **+1.7%** |
| MTP K = 1 | 63.9 / 63.9 | **64.4 / 64.4** | +0.8% |
| MTP K = 2 | 54.5 / 54.6 | **54.9 / 54.8** | +0.6% |
| MTP K = 3 | 49.3 / 49.3 | **49.6 / 49.5** | +0.5% |
| lookup10 K = 7 (T = 16) | 237.3 / 237.6 | 237.7 / 237.8 | +0.1% |
| lookup16 K = 7 | 250.0 / 250.0 | 249.8 / 249.7 | -0.1% |
| prefill, `real_code` | 1.16k | 1.16k | 0 |

Acceptance length identical on every MTP arm (1.65 / 1.85 / 2.03 / 12.11 / 12.70 tok/round) and the
generated text byte-identical on all sixteen run pairs -- the discrete confirmation of bit-identity.
The last three rows are the control: the fused band is T = 1..5, so the QPN band and prefill should
not move, and they do not.

**MTP gains only a third of what T = 1 does, and that is reported rather than explained away.** The
verify forward at T = 2 deletes the same 132 launches the T = 1 step does, and the round should
therefore gain about 0.33 ms of its 25.8 ms; it gains 0.16 ms. The MTP head's *own* `residual_add`
cannot account for the difference -- it is one launch per draft forward, not 132. Whatever absorbs the
rest, the same-batch fused/composed comparison is the number that matters and it is consistent across
three K values and two repetitions.

### What is left

The tail-fusion direction is now half collected. `residual_add` (the 130-132 launches/token) is gone
from the decode step; the other two elementwise consumers the probes priced are not:

1. **`rmsnorm`, +3.9% / 0.77 ms by the same probe** (210 calls/pass) -- the largest single item left
   on the tail. Not an epilogue problem: the norm reduces over the whole hidden vector while each
   GEMV warp produces one element of it, so folding it into the producer needs a grid-wide reduction.
   The one fusion this tree tried in that family (norm+rotation, bit-identical) lost 2.0-2.6% to the
   barrier it added.
2. **`silu_and_mul`, +1.8% / 0.37 ms** (61-65 calls/pass). Its producer is `linear_swiglu`'s gate_up
   GEMM, whose two halves are different output rows of one tensor, so the epilogue of the warp that
   computes `gate[r]` cannot see `up[r]` without a cross-warp handshake. A kernel whose warp owns
   rows `r` and `r + intermediate` would make it local, at the price of changing the GEMV's row ->
   warp mapping.
3. **Tail structural concurrency** is now worth strictly less than when round 3 priced it: the
   ~0.45 ms it was meant to overlap is gone. The chain is still strictly serial and the only siblings
   are still the attention input projections, already fused into one op.
4. Unchanged from round 8: small-n split-K (~1%), and the QPN in-band leftovers (kTiles 3/4 NACC,
   small-n grids, weight-fetch pipelining -- lookup only).

### Reproduce

```bash
# the ceiling, before any code change
bash /root/probe_skipres.sh          # -> /root/ninfer_ab/skipres

# bit-identity + the whole band, fused on vs NINFER_TERNARY_FUSED_RESIDUAL=0
bash /root/fused_final.sh            # -> /root/ninfer_ab/fused9f

# the launch-count and kernel-instantiation proof
bash /root/fused_check.sh            # -> /root/nvp_t1/fused_{on,off}.txt
```

---

## Round 10 — where a speculative round goes, and the QPN band edge re-measured (negative)

The question this round started from was the round, not the token: at the shipped defaults an MTP
K = 1 round is 25.6 ms and yields 1.65 tokens, so cutting the round is worth about as much as cutting
the step. Nothing in this tree had ever priced one.

### The round budget, measured

`nvprof --print-gpu-trace` on an MTP K = 1 decode and on an MTP K = 3 decode, differenced per kernel
family. The traces are all decode: the capture's pre-decode rows are the 8.7 s weight upload, and the
first decode launch is the only thing between them.

| family | K = 1 ms/round | K = 3 ms/round | launches/rd at K = 1 |
|---|---:|---:|---:|
| **ternary GEMV** (staged + tile) | **22.1** | **35.7** | 434 |
| ternary_rotation | 1.70 | 1.93 | 295 |
| rmsnorm (cta + warp) | 1.28 | 1.44 | 246 |
| lm_head class (Q4 + W8 rowsplit) | 1.64 | 3.53 | 6.6 |
| ternary_volta_qpn_gemm | 1.25 | 1.71 | 11.4 |
| causal attention | 0.80 | 1.02 | 37 |
| GDN family | ~1.0 | ~1.2 | 163 |
| silu / rope / sigmoid / rest | ~1.2 | ~1.4 | 130 |
| total | ~30.4 | ~48.0 | ~1400 |

**A trap worth recording.** `ternary_volta_mma_gemm` reads 5.50 ms/round at K = 1, which looks like a
seventh of the round. It is not in the round at all: its *absolute* total is 193 ms in one trace and
181 ms in the other, i.e. constant in the round count -- it is the 85-token prefill running the fp16
mma route. Normalising a capture "per round" silently converts a fixed cost into apparent per-round
work, and the only way to see it is to compare absolute totals across two captures with different
round counts. A per-round table must be read with that filter applied.

The same caveat bounds the totals: ~30.4 ms of summed decode kernels for a 25.6 ms round, because
nvprof charges something per launch and a round issues ~1400 of them. Use the table to attribute
*changes*, not to audit the wall clock.

### Where the K-slope actually comes from

Fitting the GEMV family on the two points, T = 2 (22.1) and T = 4 (35.7):

    GEMV(T) ~= 8.5 + 6.8*T   ms

so one pass over all the weights is **8.5 ms** and each additional verified token costs **6.8 ms**,
i.e. **0.80 of a weight pass per token**. That is the structural cost of the tile GEMV reading the
whole k-wide activation once per output warp: the activation side carries `(n/8) * k * 2` bytes per
token against `n * k / 4` bytes of 2-bit weights, which is the same number. Round 2's "+a third of a
T = 1 step per verified token" and this fit agree.

But the GEMV is not the whole slope. Differencing the two traces, K = 1 -> K = 3 costs +21.2 ms for
two extra draft tokens:

| source | ms (two K steps) | share |
|---|---:|---:|
| ternary GEMV (main-model verify) | +13.6 | **64%** |
| lm_head class (Q4 +1.24, W8 +0.65/+1.23) | +3.1 | 15% |
| the MTP head's own projections and the rest | +4.5 | 21% |

The lm_head result is the surprise: `q4_rowsplit_gemv` runs **1.1 times per round at K = 1 and 3.5
times at K = 3** -- it is invoked per *draft token*, not per round, so a third of the round's K-slope
is the draft head looping over tokens calling a forward-shaped kernel each time.

### L1a: the QPN band edge, re-measured — still 6

The QPN route tiles the activation through shared memory and has no per-warp re-read, so lowering its
band edge is the obvious way to flatten the 6.8 ms/token. The crossover was already recorded in
`ternary_volta_qpn_gemm.cu` (QPN loses at T = 2 by 47%, T = 3 by 21%, crosses at T ~ 4.3), but it
predates the staged code plane, the fused residual epilogue and NACC = 2 -- so it was re-run rather
than trusted, by making the edge an env knob (`NINFER_TERNARY_QPN_MIN_T`, default 6, no behaviour
change). Same batch, interleaved, two repetitions, round = acceptance / decode_speed:

| arm | K = 1 (T = 2) | K = 2 (T = 3) | K = 3 (T = 4) |
|---|---:|---:|---:|
| min_t = 6 (shipped) | **25.62 ms** | **33.67 ms** | **40.97 ms** |
| min_t = 4 | 25.62 | 33.73 | 43.49 (+6.2%) |
| min_t = 2 | 36.18 (**+41%**) | 39.92 (+18.6%) | 43.13 (+5.3%) |

**Negative, and the old table holds.** The knob stays at 6. Two details worth keeping:

* **The acceptance drifts, and that is the trap.** Opening the band raises it -- 1.85 -> 1.98 at
  K = 2, 2.03 -> 2.17 at K = 3 -- because the QPN verify reassociates the accumulation, not because
  the draft got better. At K = 3 the `min_t = 4` arm measures **49.9 against 49.6 t/s**, which reads
  as a +0.6% win in the headline metric while its *round* is 6.2% slower. The tree's own rule
  (compare `acceptance / decode_speed`) is what catches it; comparing t/s across arms with different
  numerics would have shipped a latency regression as a throughput gain.
* **At T = 1 and T = 2 the QPN arm and the SIMT arm are not comparable shapes.** The QPN schedule's A
  tile is eight tokens tall whatever T is (`kRowsPerTile = 8`), so T = 1 and T = 2 run the *same*
  QPN kernel -- and that kernel costs about twice what the staged SIMT GEMV costs for the same work
  (which is why the T = 1 row of the old table reads 19.4/19.5: with the edge at 2, neither arm
  enters QPN there at all). QPN only wins once the SIMT activation re-read dominates, at T >= 5.
  That also means **QPN is not the answer to the activation re-read**: it avoids the re-read by
  paying about 2x on the weight side. A SIMT kernel that stages the activation across the warps of a
  block -- same fast weight path, the re-read removed -- is the shape that would actually collect it,
  and that is where this round's remaining work is.
* `min_t = 2` changes the 96-token non-MTP greedy output (984 against 982 bytes; `min_t = 4` is
  identical). Some non-MTP launch therefore runs in 2 <= T <= 32. Not chased, because the arm loses on
  round time regardless; recorded because it is a launch nobody has accounted for.

### Reproduce

```bash
# the round budget at two draft counts (nvprof; ~90 s each)
bash /root/trace_mtp.sh              # K = 1 -> /root/nvp_mtp/k1.txt
bash /root/trace_mtp3.sh             # K = 3 -> /root/nvp_mtp/k3.txt
python3 /root/nvp_roundslope.py /root/nvp_mtp/k1.txt /root/nvp_mtp/k3.txt 35.165 22.695

# the band edge
bash /root/qpn_min_t.sh              # -> /root/ninfer_ab/qpnmin
python3 /root/qpn_round.py /root/ninfer_ab/qpnmin
```

---

## Round 11 — the CTA-staged activation: **bit-identical, and −10%**. The activation side is L1-served

Round 10 left one structural question: the tile GEMV spends 37% of a T = 1 call on the activation
side, costing 0.5-0.6 of a weight pass per verified token, and QPN is not the answer because it pays
about 2x on the weight side at small T. The one mechanism for that cost that had never been built is
**cross-warp reuse inside a block**.

### What was built

The shipped staged kernel gives each warp a **private** code buffer, deliberately, so that no block
barrier is ever needed -- but it reads the activation straight from global once per output warp, and
all eight warps of a block read exactly the same bytes, because the k walk does not depend on the
warp. `ternary_pq2_gemv_stage_act_kernel` (`NINFER_TERNARY_STAGE_ACT=1`, a separate kernel so the
shipped instantiation stays byte-identical) stages one eight-group activation span per token in
shared memory once:

| | per block, per token |
|---|---|
| shipped | 8 warps x 8 x LDG.64 -- the same 2 KB, eight times over |
| staged | 1 LDG.64 + 1 STS.64 **per thread**, then 8 LDS.64 per warp, plus one `__syncthreads()` |

It is bit-identical by construction (same bytes, same lane -> k mapping, same accumulation order;
only where the byte is read from changes), so the md5 gate is the whole correctness argument. Every
warp must reach both barriers, so the shipped kernel's early return for `warp >= rows` became a
predicate.

### Result: −10%, uniformly

Same batch, interleaved arms, two repetitions:

| load | shipped | `STAGE_ACT=1` | delta |
|---|---:|---:|---:|
| T = 1, `real_task` | 55.2 / 55.2 | 49.7 / 49.6 | **−10.0%** |
| T = 1, `real_code` | 53.3 / 53.3 | 48.1 / 48.1 | −9.8% |
| T = 1, `prose4k` | 51.5 / 51.5 | 46.7 / 46.7 | −9.3% |
| MTP K = 1 (T = 2) | 64.4 / 64.4 | 58.2 / 58.2 | −9.6% |
| MTP K = 2 (T = 3) | 54.9 / 54.8 | 54.9 / 54.8 | 0.0% |
| lookup10 (T = 16) | 237.4 / 237.7 | 237.8 / 237.7 | 0.0% |

Verification, against the round-9 lesson that a route which is never taken is trivially
bit-identical: the 96-token greedy md5 against `/root/wt/base.out` is IDENTICAL with the arm on and
off, the generated text is byte-identical on all twelve run pairs, and the tracing run shows
**26,466 launches of the new kernel and zero of the shipped one**. The two untouched arms (T = 3 and
T = 16, both outside the band) read identically to the digit, which pins the gate.

### Why the negative is the informative part

The loss says the eight duplicate reads **were already being served**. The warps of a block walk the
same cache lines within a few hundred cycles of each other, so they were L1 hits; staging traded an
L1 hit for an STS, a barrier and eight LDS whose throughput is no better than L1's. So the
activation side of this kernel is **L1-served and latency/issue-bound, not L2 or DRAM traffic** --
which is exactly the regime in which staging cannot help.

That closes the third and last barrier-free mechanism for the per-token activation cost:

| mechanism | reduces bytes? | cost | measured |
|---|---|---|---|
| wide lane (`TILE_WIDE1`) | no | lane remap, reassociation | −1.3% best |
| row block (kR rows/warp) | yes, 2.5x at kR=4 | registers -> occupancy | loses (Ada 43.0 -> 32.8 t/s; `T2BLOCK` off) |
| CTA staging (`STAGE_ACT`) | yes, 8x on the load side | +STS +LDS +barrier | **−10.0%** |

The 0.80-weight-passes-per-token K-slope is therefore a property of this SIMT shape, not an
oversight, and the reachable levers on round latency are now:

1. **make the tensor-core route cheap at small T** -- QPN already avoids the re-read, so its 2x
   weight-side penalty in the T = 1..2 regime is the only thing between the MTP band and a flat
   verify;
2. **the MTP head's per-draft-token projections** -- 15% of the K-slope, and `q4_rowsplit_gemv` was
   measured running 1.1 times per round at K = 1 and 3.5 times at K = 3, i.e. per draft token rather
   than per round;
3. acceptance, which divides the same round latency over more tokens (K = 1 is 15.5 ms/token against
   4.2 ms/token on the lookup path).

### Reproduce

```bash
bash /root/stage_act.sh              # -> /root/ninfer_ab/stageact  (+ /root/nvp_t1/stageact.txt)
```

---

## Round 12 — the QPN `kTiles = 1` accumulator default had never been swept: **lookup +8%, free**

Round 10 left the QPN route as the only lever that reaches both the MTP and the lookup bands, and
round 10's own negative said why: lowering the band edge to T = 2 cost +41% on the K = 1 round. This
round went after the route's own cost instead, and found a default that had been wrong since the
route was ported.

### First, a correction to how the bands are described

**The context-lookup verify is T = 8, not T = 16.** `K = 7` draft tokens verify as `T = K + 1 = 8`.
With `kRowsPerTile = 8` that is `tiles = ceil(8/8) = 1`, i.e. the `kTiles = 1` arm -- not the
`kTiles = 2` arm the notes in this tree assumed. Everything below follows from that.

### The unswept default

`launch_shape` computes `kNacc = kNaccOverride != 0 ? kNaccOverride : (kTiles == 1 ? 4 : 1)`, and
`NINFER_TERNARY_QPN_NACC` was read **only** in the `tiles == 2` branch -- so the knob was silently
inert on `kTiles = 1`, and its value of 4 there had never been measured. (Same class of dead default
as round 6's `NINFER_TERNARY_QPN` gate.) The override now reaches every `kTiles` (refused above
`kTiles = 2`, where four accumulators per tile cannot fit), with every built-in default unchanged
except this one. Measured with `NINFER_TERNARY_QPN_MIN_T=1`, which puts the whole T = 1 step on QPN
and isolates the weight path -- the A tile is eight tokens tall with seven dead, so there is no
activation slope to confuse the reading:

| NACC at kTiles = 1 | T = 1 whole step | MTP K = 1 (QPN verify only) |
|---|---:|---:|
| 4 (the old default) | 31.2 / 31.2 t/s | 45.6 / 45.6 |
| **1 (now shipped)** | **38.0 / 38.0 (+21.8%)** | **54.6 / 54.6 (+19.7%)** |
| 2 | 38.0 / 38.0 | 54.6 / 54.6 |

SIMT control in the same batch: 55.2/55.3 at T = 1, 64.4 at MTP K = 1. One and two tie, which says
the four-deep mma chain was not what the four accumulators were buying -- the registers were.

### It also moved the lookup path

The sweep's `lookup10` arm read **235.0/235.2 t/s** with the old default and **254.2/254.5** with the
new one, in two adjacent batches whose T = 1 SIMT control was identical (55.2/55.3 both) -- so
+8.2% causal, not drift. That was the surprise that led to the T = 8 correction above. Shipped-default
smoke, same batch, two repetitions:

| load | before | after | delta |
|---|---:|---:|---:|
| T = 1 decode | 55.2 / 55.3 | 55.2 / 55.3 | 0 (control) |
| MTP K = 1 | 64.4 | 64.4 / 64.3 | 0 (control) |
| MTP K = 3 | 49.3 | 49.3 / 49.2 | 0 (control) |
| `real_code` decode | 53.3 | 53.3 / 53.2 | 0 (control) |
| `prose4k` decode | 51.5 | 51.5 | 0 (control) |
| **lookup10 K = 7** | 237.4 / 237.6 | **253.9 / 254.5** | **+7.4%** |
| **lookup16 K = 7** | 249.8 / 249.8 | **265.0 / 265.4** | **+6.1%** |
| prefill | 1.16k / 1.18k | 1.16k / 1.18k | 0 (control) |
| perplexity score rate, `--context 8` | 98.5 tok/s | **122.4** | **+24.3%** |

The 96-token greedy md5 against `/root/wt/base.out` is IDENTICAL (T = 1 never enters the QPN band),
and the acceptance lengths are unmoved (1.65 / 12.11 / 12.70).

**Continuous check**, `--context 8 --stride 4` so every forward is an 8-token `kTiles = 1` call,
NACC 4 against 1 over 8,675 scored tokens: mean_nll 4.835767 against 4.835772, **+0.00055% PPL** --
one forty-fourth of the fp16 prefill operand margin this tree accepted (+0.0245%).

### The band edge: re-measured, moved, and then refused

With the `kTiles = 1` arm fixed, the crossover moves and the edge deserves a re-look. Same batch,
interleaved, two repetitions, round = acceptance / decode_speed:

| arm | K = 1 (T = 2) | K = 2 (T = 3) | K = 3 (T = 4) |
|---|---:|---:|---:|
| min_t = 6 (shipped) | **25.62** | **33.73** | 41.01 |
| min_t = 4 | 25.62 | 33.73 | **37.37 (-8.9%)** |
| min_t = 3 | 25.62 | 33.93 | 37.37 |
| min_t = 2 | 30.40 (+18.7%) | 34.06 | 37.74 |

So the crossover is now T ~ 3.5. **The edge stays at 6**, and not for the round-time reason:

* The T = 4 window costs **+0.097% PPL** against SIMT -- four times the accepted fp16 prefill margin
  and 2.7x the NACC = 2 trade. (Forward width 7, the other `kTiles = 1` window, costs only
  **+0.0043%**; the two differ by 22x in absolute NLL and the reason is not established.)
* What it buys is only that K = 3 and K = 4 stop being so much worse than K = 1. They stay worse.
  Per-token latency `round / acceptance` after the NACC fix: K = 1 **15.54 ms**, K = 2 18.27,
  K = 3 17.06, K = 4 17.13. **K = 1 is still the best setting and its round does not move at all with
  the edge**, so the change cannot improve the best configuration -- it pays a real numerical cost to
  make a losing one less losing. `NINFER_TERNARY_QPN_MIN_T=4` is left as the escape hatch and
  recorded as a trade.

**Method note that cost this round two runs:** `ninfer-perplexity` scores at a forward width of
`context - 1`, not `context`. Traced: `--context 4` runs `ternary_pq2_gemv_tile_kernel<int=3,...>`
and no QPN kernel at all, which is why the first edge check (`--context 4 --stride 2`) came back
bit-identical to six decimals and measured nothing. `--context 5 --stride 4` is the T = 4 window.

### What this round changes about where the remaining headroom is

Round 10's budget said a K = 1 round is 73% verify GEMV, and rounds 10-11 closed the activation
re-read from the SIMT side. This round adds the other half of that picture: the GEMV's weight side
runs at 789 GB/s = 88% of the card's peak, so it is done, and the QPN route -- the only structure that
avoids the re-read -- had a 21.8% default bug in it. With that fixed, QPN's T = 1 step is 26.3 ms
against SIMT's 18.1 ms, i.e. still 1.45x, so T = 2 stays SIMT and the K = 1 round is unchanged.

That puts the round-latency budget at roughly: **~0% left in the weight stream, ~0% in the SIMT
activation side (three mechanisms measured and refused), ~2.5% in a rmsnorm fusion, ~1.3% in a
silu_and_mul fusion, 2-4% in the MTP head's per-draft-token projections, and everything else in one
uncertain item -- bringing the QPN small-T weight path to parity, worth ~18% on the K = 1 round if it
lands.** Against that, the acceptance spread this tree has already measured is 1.65 (real_task, K = 1)
against 12.11 (repeated text), and the lookup path's 3.93 ms/token against K = 1's 15.54 comes
entirely from acceptance -- its round is 1.86x *more* expensive. So the round is close to done and
the draft is not, which is where the next round should go.

### Reproduce

```bash
bash /root/qpn_weight_path.sh        # QPN's weight path alone, T = 1 -> /root/ninfer_ab/qpnt1
bash /root/qpn_nacc.sh               # the kTiles = 1 NACC sweep -> /root/ninfer_ab/qpnnacc
bash /root/qpn_cross.sh              # the band edge with the fix in -> /root/ninfer_ab/qpncross
bash /root/r12_ship_smoke.sh         # the shipped default -> /root/ninfer_ab/r12ship
bash /root/ppl_qpn_acc.sh            # A: NACC 4 against 1, forward width 7
bash /root/ppl_width_probe.sh        # the forward-width trace + QPN against SIMT at width 7
bash /root/ppl_edge_t4.sh            # D: the T = 4 window (forward width 4)
```

---

## Round 13 — the round-latency budget closes, and the draft window has a dominated basin

Round 12 left one item with more than 10% in it: bring the QPN route's small-T weight path to parity,
which would let T = 2 take the tensor-core route and cut the K = 1 round by ~18%. It also left the
observation that the acceptance spread this tree has already measured (1.65 against 12.11) is an order
of magnitude larger than anything left on the round. This round closed the first and went looking in
the second.

### The QPN small-T weight path: closed, negative

| min-blocks | T = 1 whole step (QPN everywhere) | MTP K = 3 (T = 4 on QPN) |
|---|---:|---:|
| **4 (the built-in value)** | **38.0 / 38.0 t/s** | **58.6 / 58.6** |
| 6 | 31.7 / 31.7 (-16.6%) | 51.1 / 51.1 (-12.8%) |
| 8 | 9.2 / 9.2 (**-76%**) | 18.1 / 18.1 (-69%) |

(`lookup10`, which does not touch this arm, read 256.8/256.8 -- the control.) So the kernel is **not**
occupancy-starved despite sitting at 50%: it wants its registers, like the rest of this family, and
`minBlocks = 8` caps it at 32 and destroys it. SPLITK was not swept alongside it because it has to
divide the group count and every width in this model has groups in {40, 48, 80, 136}, whose only
common divisors are 1, 2, 4 and 8 -- 8 is shipped and 4 is the only alternative, which the constant's
own comment already rejects on warp count.

With NACC's +21.8% (round 12) as the only thing the route had, **lever 1 is closed**: QPN's T = 1 step
stays at 26.3 ms against SIMT's 18.1, T = 2 stays SIMT, and the K = 1 round stays at 25.6 ms.

### The draft window, swept past 4 for the first time

The tree's recorded sweep is K = 1..4, and it concluded window 1 is the right choice for text that is
not reproducing the prompt. The validation ceiling is K = 7 (`--spec mtp requires --draft-tokens in
[1,7]`). Same batch, two repetitions, `real_task`, `--max-new 128`, L = 1000/t/s:

| K | t/s | acceptance | round ms | **L ms/token** | acceptance rate |
|---:|---:|---:|---:|---:|---:|
| **1** | 64.45 | 1.65 | 25.60 | **15.52** | 64.9% |
| 2 | 54.90 | 1.85 | 33.70 | 18.22 | 43.0% |
| 3 | 49.50 | 2.03 | 41.01 | 20.20 | 34.8% |
| 4 | 45.65 | 2.21 | 48.41 | **21.91 -- the worst point** | 30.3% |
| **5** | 64.00 | 2.49 | 38.91 | **15.62** | 30.3% |
| 6 | 61.80 | 2.54 | 41.10 | 16.18 | 26.3% |
| 7 | 59.70 | 2.59 | 43.38 | 16.75 | 23.4% |

**The 29% step between K = 4 and K = 5 is the QPN band edge, seen from the other side.** K = 4 verifies
at T = 5, which is below the edge and therefore runs the SIMT tile; K = 5 verifies at T = 6, takes QPN,
and its round is 38.9 ms against 48.4 ms *while accepting more*: doing more work is 20% cheaper. That
makes **K = 2, 3 and 4 a strictly dominated basin**, and it also confirms round 12's refusal of the
edge change -- moving the edge to 4 would take K = 4 from 21.91 to roughly 18.6 ms/token, still worse
than K = 5's 15.62, so it cannot move the optimum.

So the curve is a plateau of two: K = 1 at 15.52 and K = 5..7 at 15.6-16.8 ms/token, with a valley
between them. There is no win here, only a trap to avoid, and `--draft-tokens` is now a documented
choice rather than a swept one.

**The same sweep on the code workload, where the draft head is much better and the optimum moves.**
Same batch, two repetitions, `real_code`, `--max-new 128`:

| arm | t/s | acceptance | round ms | **L ms/token** | against no spec |
|---|---:|---:|---:|---:|---:|
| no spec | 53.3 | 1.00 | 18.76 | 18.76 | -- |
| K = 1 | 71.60 | 1.94 | 27.09 | 13.97 | -25.5% |
| K = 2 | 72.80 | 2.57 | 35.30 | 13.74 | -26.8% |
| K = 3 | 70.15 | 3.00 | 42.77 | 14.26 | -24.0% |
| K = 4 | 64.25 | 3.23 | 50.27 | 15.56 | -17.1% |
| **K = 5** | **77.10** | 3.23 | 41.89 | **12.97 <- best** | **-30.9%** |
| K = 7 | 75.00 | 3.50 | 46.67 | 13.33 | -28.9% |

Code is far friendlier to the draft head than ordinary prose -- acceptance 1.94 against 1.65 at K = 1,
3.00 against 2.03 at K = 3, 3.23 against 2.49 at K = 5 -- so speculation pays much more there (+45%
against no spec at its best, against +24% on `real_task`). Two differences from the prose curve:

* the dominated basin is narrower but still there: K = 2 (13.74) actually edges K = 1 (13.97), and the
  only unambiguous trap is **K = 4 (15.56)**;
* **K = 5 is a real optimum on code**, 7% better per token than K = 1, which it was not on `real_task`.

The K = 4 -> K = 5 step reproduces exactly and for the same reason: round 50.27 -> 41.89 ms
(**-16.7%**) at *identical* acceptance 3.23, because K = 5 verifies at T = 6 and takes the QPN route
while K = 4's T = 5 is below the edge and runs the SIMT tile. So `--draft-tokens` is a workload
parameter -- high-acceptance workloads want the large window -- and K = 4 is wrong everywhere.

### The K = 5 reversal, and why the record said the opposite

The tree's recorded advice is that window 1 is the best choice, and that advice was correct on the
build it was taken on. Re-running the two NACC values side by side on `real_code` -- where there is no
repetition, so the lookup gate cannot interfere -- shows the whole difference:

| `real_code` | K | NACC = 4 (pre-round-12) | NACC = 1 (shipped) | change |
|---|---:|---:|---:|---:|
| verify T = 2 (SIMT) | 1 | 71.4 | 71.6 | 0 |
| verify T = 5 (SIMT, below the edge) | 4 | 64.2 | 64.4 | **0 -- the control** |
| verify T = 6 (QPN, `tiles = 1`) | **5** | **63.9** | **77.1** | **+20.7%** |

So **before round 12, K = 5 was 63.9 against K = 1's 71.4, i.e. 11.7% worse per token** -- a negative
result, and the reason the window default was set to 1. Round 12's `kTiles = 1` NACC fix is what
flipped it: K = 5's verify is the arm that bug lived on. K = 1 and K = 4 are bit-identical across the
knob, which is what attributes the change to it rather than to batch drift.

The same run corrects two other statements in this tree:

* **`--spec mtp` on prose is not a wash.** Window 1 on `prose4k` is **62.3 t/s against 51.5 with no
  spec at all (+21%)**. An earlier note in this round compared window 1 with the *no-spec* arm and
  read it as a tie; that was the wrong pair.
* **The lookup fast path does not want a small window.** `apps/cli/options.cpp` says "a small window
  is what lets the fast path engage", because the lookup's third gate wants the MTP head's first K
  drafts to agree with it exactly. Measured on `lookup10`: K = 1 **171.8 t/s** (acceptance 5.74),
  K = 3 188.9 (8.46), K = 5 245.3 (10.90), **K = 7 256.7 (12.11)**. Window 1 is 49% *behind* window 7
  on the workload the gate exists for.

### Window 1 is a prose-tuned default

With the shipped binary, every workload's own optimum:

| workload | K = 1 | best wide window | verdict |
|---|---:|---:|---|
| `prose4k` | **62.3** (L 16.05) | K = 5: 58.6 (17.06) | **K = 1** |
| `real_task` | **64.45** (L 15.52) | K = 5: 64.0 (15.63) | tie |
| `real_code` | 71.6 (13.97) | **K = 5: 77.1 (12.97)** | **K = 5** |
| `lookup10` (repeated text) | 171.8 (L 5.82) | **K = 7: 256.7 (3.90)** | **K = 7** |

So the CLI default of 1 is right for prose and wrong by 7% on code and by 49% on repeated text. Since
acceptance is observable at run time, an acceptance-driven adaptive window is the obvious follow-up --
not a fixed constant. No default was changed here; the reading is recorded.

**A self-inflicted bug found by this comparison, and worth recording as a method note.** The first
attempt at the NACC A/B came back *bit-identical* on both arms (76.9 against 77.1, 64.2 against 64.3).
That is the tell: `NINFER_TERNARY_QPN_NACC=4` no longer reached the kTiles = 1 arm, because round 12
had changed that arm's inline default from 4 to 1 and the dispatch's last branch was still spelled
`launch_shape<1, half>` -- an abbreviation that now resolves to NACC = 1. The dispatch names all three
values explicitly. **Two arms that agree to the digit are not a result; they are a routing bug.**

---

## Round 14 — the context-length axis, which nothing in this document had measured

This round did not start from a hypothesis. It started from the resident server's own request log,
where a real long agent turn reads:

```
req#2 done | prompt 21,006 | output 34,766 | TTFT 21.0s | total 12m 27.7s |
            prefill 1.08k tok/s | decode 47.9 tok/s | mtp accepted 21,818/65,878 (33.1%)
```

**47.9 t/s.** Every decode number in this document -- 51.5, 54.2, 55.3, 64.4, 77.1 -- was taken at
45 to 20,000 tokens of context with a 128-token generation. A long agent turn is neither. So the two
are not in disagreement; the tree had simply never measured the axis the traffic actually lives on.

### Decode rate against context, with and without speculation

`--max-new 128` throughout, so the number is a rate and not an average over a growing context.
Fixtures are built from **distinct** source files: a fixture assembled by repeating one document would
trigger `lookup_draft`'s 16-token verbatim suffix match and hand the workload the context-copy fast
path, which is not what a conversation gets. Two repetitions each; the spread is under 1%.

| prompt tokens | **no spec** (T = 1) | K = 1 | K = 5 | K = 5 acceptance |
|---:|---:|---:|---:|---:|
| 45 | -- | 66.4 | **76.1** | 3.00 |
| 4,381 | -- | 66.8 | 69.2 | 2.82 |
| 5,120 | 51.7 | 68.5 | **80.1** | 3.34 |
| 11,276 | -- | 61.5 | 60.0 | 2.82 |
| 15,592 | -- | 61.5 | 70.2 | 3.34 |
| 19,930 | -- | 52.9 | 59.9 | 3.10 |
| 39,009 | **36.2** | 47.4 | 47.6 | 2.86 |
| 81,397 | **28.6** | 40.8 | 43.1 | 3.17 |

Three readings:

* **The loss is in the base step, not in speculation.** No-spec falls 51.7 -> 36.2 -> 28.6 t/s, i.e.
  the T = 1 step goes 19.3 -> 27.6 -> 35.0 ms per token, **+43% at 39k and +81% at 81k**. Attention
  compute and KV bytes do not obviously pay for that (+8.3 ms at 39k against an estimated ~1-3 ms of
  KV traffic), so which kernel grows is **not yet known** -- that is the open question this round
  hands to the next one. Nothing in rounds 1-13 touches it: every number there is a short-context
  measurement, and the GEMV family that dominates the budget does not read the context at all.
* **The draft window's advantage collapses at long context.** At 39k, K = 1 and K = 5 are equal
  (47.4 against 47.6) despite K = 5's acceptance being 68% higher (2.86 against 1.70): the verify's
  own cost grows with context faster than the extra accepted tokens pay for it. At 81k K = 5 is still
  ahead (43.1 against 40.8). So **K = 5 is never worse, but at 21k-56k -- the range the harness
  runs in -- it is worth approximately nothing**, and the deployment's value is concentrated at
  short context and on repetition.
* **Acceptance tracks the content, not the length.** Same acceptance 3.34 gives 80.1 t/s at 5k and
  70.2 at 15.6k; acceptance 2.82 gives 69.2 at 4.4k and 60.0 at 11.3k. So the two effects are
  separable and the context term is the smaller one until about 20k.

### A caveat this round found in its own round-13 table

Round 13's per-workload window comparison did **not** pass `--no-thinking`; the fixture sweep in this
round does. The window comparison is sensitive to that:

| workload | thinking on (round 13) K = 1 -> K = 5 | thinking off (round 14) K = 1 -> K = 5 |
|---|---|---|
| `prose4k` | 62.3 -> 58.6 (K = 1 wins 6%) | 68.5 -> **80.1** (K = 5 wins 17%) |
| `real_task` | 64.45 -> 64.0 (tie) | 66.4 -> **76.1** (K = 5 wins 15%) |
| `real_code` | 71.6 -> **77.1** (K = 5 wins 7%) | 66.8 -> 69.2 (K = 5 wins 3.6%) |
| `prose16k` | -- | 52.9 -> 59.9 (K = 5 wins 13%) |

So K = 5 wins in both modes on `real_task` and `real_code`, and the prose exception is
**thinking-mode-specific** -- with thinking off, K = 5 wins there too, by 17%. The serving deployment
runs thinking on by default, so the thinking-on column is the one that governs it, and the prose
regression it was flagged for is real but narrow.

### Reproduce

```bash
python3 /root/build_ctx.py /root/ninfer_ab/ctxlong   # distinct-source long fixtures
bash /root/ctx_sweep.sh        # short-to-20k, K = 1 against K = 5, thinking off
bash /root/ctx_long.sh         # 39k and 81k
bash /root/ctx_nospec.sh       # the no-spec baseline that localises the loss
```

---

## Round 15 — what the MTP path actually does, and why K = 5 wins on code

Round 14 left two things open: an unexplained 3% (K = 5 beating K = 4 at the same band edge and the
same acceptance) and the question of whether the edge's holes at T = 4 and T = 5 were worth closing.
This round read the MTP code against the upstream trees and closed both questions.

### The MTP code, read properly

* **The "early stop" is a budget clamp, not a quality test.** `src/ops/kernel/mtp_round.cuh`:
  `next_extents = clamp(min(remaining_budget - 1, max_context - frontier - 1), 0, proposal_k)`. With
  budget and context to spare -- every measurement here -- that is `proposal_k`, so `extent = K` and
  **all K draft positions are verified**. An earlier guess in this round that the head trims its own
  tail was wrong.
* **`verify_k = proposal_k = draft_window`.** The round's verify width and the drafter's proposal count
  are the same number, so the window is doing two jobs at once.
* **Upstream does not decouple them either, except for ngram drafts.** `/root/ninfer-all`'s copy of the
  same kernel is annotated "K=1..31 verification (up to 63 at B=1) and P=1..15 next proposals", and its
  copy of the draft logic *does* accept a wider verify than proposal -- but the guard is explicit:
  `(!state.ngram && proposal_k != k)` throws. A neural round verifies at the drafter's own width. So
  the V100 port's coupling matches upstream's neural semantics and there is no structural difference to
  import.
* **The acceptance rate is not a quality metric.** It is `accepted / (K * rounds)`, so it falls
  mechanically as K grows: at K = 1 it averages only position 0 (77.2% on code), at K = 5 it averages in
  position 4's 18% and position 5's 3% and reads 44.4%. The number that decides anything is the
  acceptance *length*.

### The wall is per-position survival, and it is at position 4

`accepted by pos` divided by rounds is the joint probability that every draft up to that position
survived:

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `real_code`, K = 5 | 72% | 62% | 33% | 18% | 18% | 3% | -- |
| `real_task`, K = 7 | 63% | 39% | 22% | 14% | 10% | 6% | 4% |
| ctx32k, K = 5 | 66% | 48% | 34% | 23% | 16% | **0%** | -- |

On `real_code` **K = 4 and K = 5 accept exactly the same number of drafts -- 87 over 39 rounds --
so the fifth position contributes nothing**, and the two arms' acceptance lengths are identical
(3.23). That is the whole explanation for the user-facing oddity that a wider window does not raise
the acceptance rate: it cannot, because the drafter is done by position 4.

### Where K = 5's win actually comes from

Because K = 4 and K = 5 accept the same, the only thing separating them is **which token width the
verify lands on**:

| `real_code`, same batch, two reps | t/s | round ms |
|---|---:|---:|
| edge 6, K = 4 (T = 5 -> SIMT) | 64.20 | 50.31 |
| edge 5, K = 4 (T = 5 -> QPN) | 74.70 | 43.24 |
| edge 5, K = 5 (T = 6 -> QPN) | 77.10 | 41.89 |

So the window in this model is **a route selector, not a quality knob**: T = 5 and T = 6 are both
`tiles = 1` on the same QPN kernel, and the 16% between the first two rows is purely the band edge.

### The edge's holes are worth nothing, so the edge stays at 6

Round 12 refused an edge change on the T = 4 window alone. Same batch, both windows, 8,675 scored
tokens each, `--text /root/ppl_qpn/text.txt` (perplexity scores at a forward width of `context - 1`):

| window | SIMT arm | QPN arm | delta PPL | score rate |
|---|---:|---:|---:|---:|
| T = 4 (`--context 5 --stride 4`) | 319.4361 | 319.7467 | **+0.097%** | 108.6 -> 119.8 |
| T = 5 (`--context 6 --stride 5`) | 275.9156 | 276.0385 | **+0.0446%** | 115.5 -> 140.1 |

Closing them does lift the configurations that sit in them -- K = 4 by 16.4%, K = 3 by 8.9% -- but
**K = 5 remains the optimum either way**, because its acceptance length (3.23) is at least as high as
K = 4's (3.23, equal) and higher than K = 3's (3.00). The edge at 6 is therefore not costing anything
in achievable performance: every window it disables (T = 3, 4, 5) is a *dominated* configuration on
every workload measured. Paying +0.0446% (1.8x the fp16 prefill margin this tree accepted) and
+0.097% for "make a losing setting less losing" is the same trade round 12 already refused, and it is
still not worth it. The edge stays at 6 and the item is closed with both windows measured.

Where each workload's optimum actually sits, and whether the edge is involved:

| workload | best window | its verify T | on QPN? |
|---|---|---|---|
| `prose4k` | K = 1 | T = 2 | no -- deliberately, QPN at T = 2 measured +18.7% slower |
| `real_task` | K = 1 / K = 5 | T = 2 / T = 6 | T = 6 only |
| `real_code` | **K = 5** | T = 6 | yes |
| `lookup10` | **K = 7** | T = 8 | yes |

### The 3% anomaly, attributed -- and the harness fix that made it possible

K = 5 beats K = 4 **by 3% even at the same edge, with identical acceptance (3.23 / 4.13) and identical
round counts**, while doing one more draft step and verifying one more token (87.8 against 85.5 t/s at
`--max-new 64`, 77.10 against 74.70 at 128 -- reproducible). Attention cannot explain it: K = 5's
dominant `causal_attention_small_t_tc` call is *slower* (153 us against 92 us).

Three cut rules were tried before this could be measured, and two were wrong, each for a reason worth
recording: "first decode GEMV" fails because the staged GEMV serves the prefill's tail chunk, and
"first `mtp_prepare_next_round`" fails because that kernel runs during the staged MTP prefill too. The
right rule is **the last CUTLASS launch** -- CUTLASS is the prefill's GEMM and refuses T < 256, so
nothing in a decode round can use it. `nvp_phases.py` proves it instead of asserting it: bucketing the
capture by time gives 0.6-3.4 s of CUTLASS-only rows, a transition at 3.8-4.2 s, then ~1 s of
QPN/rotation/tile rows with no CUTLASS. The capture is 4.88 s for a 0.75 s decode -- 84% of it is
prefill, which is why every per-round column before this was meaningless.

With the cut fixed, `real_code`, `QPN_MIN_T=5`, 15 rounds per arm:

| kernel | K = 4 (T = 5) ms/rd | K = 5 (T = 6) ms/rd | delta | calls/rd A / B |
|---|---:|---:|---:|---:|
| `ternary_volta_qpn_gemm` | 22.10 | 29.26 | +7.15 | 299.7 / 427.7 |
| **`ternary_pq2_gemv_tile_kernel`** | **11.68** | **0.00** | **-11.68** | 128.0 / 0.0 |
| `causal_attention_small_t_tc` | 2.30 | 3.44 | +1.14 | 23.6 / 24.7 |
| lm_head class (`w8_rowsplit` + `q4_rowsplit`) | 4.71 | 6.04 | +1.33 | 21.0 / 27.4 |
| everything else | ~9.10 | ~9.87 | +0.77 | -- |
| **TOTAL** | **49.89** | **48.61** | **-1.27** | -- |

-1.27 ms/round against a wall-clock -1.35 ms. **The anomaly is 128 GEMV calls per round that take the
SIMT tile kernel (11.68 ms) in the K = 4 arm and the tensor-core route (+7.15 ms) in the K = 5 arm** --
the QPN count rises by exactly the 128 the tile count loses. So the draft window does not only set the
verify's width; it also moves a second, K-dependent pass in the draft path between routes. That is why
the window's cost/benefit has never matched a simple "wider verify, more accepted tokens" model.

It also confirms the edge decision from the other side: edge 4 and edge 5 give K = 4 the same 74.70
t/s, so lowering the edge cannot move those 128 calls, which bounds their width at <= 3. The T = 5
window's +0.0446% PPL therefore buys only a configuration's verify that K = 5 dominates anyway.

**Method lesson, recorded because it cost two rounds: a per-round table is not evidence until the
phase boundary is proven, and the right proof is a marker that cannot appear in the phase being
excluded -- not a better guess at a cut.**

**Next, and it is the larger item:** round 14's finding that the base T = 1 step goes 19.3 -> 27.6 ->
35.0 ms/token from 5k to 39k to 81k of context (+43% at 39k) is *still* unattributed, and it was
blocked by exactly this harness problem. One no-spec trace at 39k context, cut the same way, settles
which kernel pays it. That is worth ~43% against this round's 3%.

---

## Round 16 — the long-context cost is one kernel, and it is 99% of the +43%

Round 15 fixed the capture cut (last CUTLASS launch) and the first thing it unlocks is the item round 14
could not attribute. Two no-spec captures, `--max-new 32` so every decode step is one token and one
round, same binary, same batch, 5,120 against 39,009 tokens of prompt:

| kernel | 5,120 ctx ms/rd | 39,009 ctx ms/rd | delta | calls/rd |
|---|---:|---:|---:|---:|
| ternary GEMV (staged + tile) | 15.46 | 15.47 | **+0.01** | 401 |
| **`causal_attention_small_t_tc`** | **1.54** | **9.98** | **+8.45** | 16.0 |
| `causal_attention_small_t_reduce` | 0.16 | 0.26 | +0.11 | 16.0 |
| rotation, rmsnorm, GDN, silu, everything else | ~4.0 | ~4.0 | ~0.00 | -- |
| **total** | **21.13** | **29.70** | **+8.57** | -- |

**The entire long-context cost is one kernel.** +8.45 of the +8.57 ms/token is
`causal_attention_small_t_tc_volta_partial_i8`, the 16 calls per round that correspond to the model's
16 full-attention layers. The ternary GEMV -- 401 calls and 15.5 ms of the round, and the family that
has absorbed every round of optimisation in this document -- is **completely context independent**
(+0.01 ms).

Per call, the attention goes from 96 us at 5k to 624 us at 39k: the context grows 7.6x and the call
grows 6.5x, i.e. linear. That is ~16 ns per context token per layer, or about 24 cycles, which for a
GQA dot of `n_kv x head_dim` is on the order of 43 MAC/cycle -- **a loop/compute-bound number, not a
KV-bandwidth-bound one**, so there is room in it that a bandwidth argument would have written off.

Size of the prize: at 39k context this single kernel is 9.98 ms of a 29.70 ms step (**34%**), and at
81k the step is 35.0 ms. Halving it takes decode from 33.5 to roughly 40 t/s (**+20%**) at 39k, and
more at the 21k-56k range the resident server actually serves. **Every previous round of this document
measured at 45-20,000 tokens of context and optimised the 15.5 ms GEMV; the kernel that decides the
serving rate is the 10 ms one nobody has touched.**

Reproduce:
```bash
bash /root/trace_ctx_cost.sh
python3 /root/nvp_phases.py /root/nvp_ctx/long.txt 400
```

---

## Round 17 — the long-context attention kernel: profiled, and the root cause is the tile's row count

Round 16 named the kernel. This round profiled it with Nsight Compute (`ncu` 2025.1.1 needs
`--privileged`; `nvprof` does not, which is why the tree had nvprof numbers but no occupancy data).

### What ncu says

| metric | value |
|---|---|
| Theoretical Occupancy | **12.50%** |
| Block Limit Shared Mem | **2 blocks** |
| Block Limit Registers | **2 blocks** |
| No Eligible (scheduler had nothing to issue) | **93.3%** |
| Eligible Warps Per Scheduler | 0.07 |
| Compute (SM) / DRAM / L2 / L1-TEX Throughput | 0.44% / 0.22% / 0.42% / 6.1% |

It is starved for warps: two blocks of four warps is 8 warps of the SM's 64, and 93% of cycles have no
eligible warp. Every pipe is under 1% of its peak.

(The achieved-occupancy figure in that report is 6.25%, but it comes from a 16-CTA launch -- grid
`(4,4,1)`, an early decode step -- so it reflects an underfilled grid, not the 39k steady state. The
*theoretical* 12.5% and the block limits are properties of the kernel and hold everywhere.)

Roofline on the same call for scale: at 39,009 tokens of context the kernel reads
`2 * kv_heads(4) * head_dim(256) * 39009` = **79.9 MB** of int8 KV in **623 us** = **128 GB/s, 14% of
the card**, and does 24 heads x 39,009 x 256 = **0.48 GFLOP** at 0.77 TFLOP/s, 0.6% of peak. Both ends
of the roofline are two orders of magnitude away, which is the signature of a warp-starved kernel and
not of a machine limit.

### The blocker, and it is not a tuning constant

The shared-memory budget per CTA is **36.7 KB**:

| buffer | bytes | note |
|---|---:|---|
| `q_s` | **16,896** | `Br=32` rows x `SmemStride=264` halves x 2 |
| `q_tail_s` | 2,112 | read only by the `CompactTail` (5-warp) instantiation |
| `k_s` / `v_s` | 8,448 each | `Bc=16` keys |
| `p_tail_s` + `physical_pages_s` | 384 | |

96 KB of SM shared memory divided by 36.7 KB is 2.6, i.e. **2 blocks**, and registers independently
limit it to 2 as well. Three blocks needs <= 32 KB, so it needs 4.7 KB cut.

**The obvious cut is not available.** `q_s` is 32 rows wide while the T = 1 decode shape has
`row_count = tokens * GroupSize = 1 * 6 = 6` rows live, so it looks like 13.7 KB of waste -- and the
first attempt here sized it by the live row count. It was wrong, and the source says why:
`volta_load_qp` reads the row `volta_qp_get_i()` chooses, and the mma it feeds is documented as
`QK^T: D[32x8] += Q[32x8] @ K[8x8]^T` -- **the A operand consumes the whole 32-row tile in one call**.
Rows 6..31 are zero-filled and participate. So `q_s` cannot shrink without changing the tile topology,
and the remaining trims (`q_tail_s` 2.1 KB, the two small arrays) leave 34.5 KB, still 2 blocks.

**That single fact is the root cause of the long-context cost.** The tile is 32 rows because that is
one Volta `m8n8k4` quadpair group; a 6-row problem pays for 32, and the topology additionally has all
`DimSplit = 4` warps redundantly recompute the QK^T and online-softmax step (the file's own design
note #1 says so). Together that is ~21x the useful QK^T issue work on the hot decode shape -- invisible
at 5k tokens, where the whole call is 96 us, and the dominant term at 39k.

### The three ways forward, with their costs

1. **int8 `k_s`/`v_s` plus dequantize in the fragment loader.** 36.7 -> 28.3 KB, i.e. **3 blocks,
   +50% warps**. But it moves the conversion into the mma feed: `volta_load_k`/`volta_load_v` run 64
   times per warp per key tile, and each would grow from one `LDS.128` to a load plus ~10 conversion
   instructions -- roughly +640 instructions per warp per key tile against a current ~384. The
   arithmetic says this is a wash at best, and it should be measured before being believed either way.
2. **Double `WarpsPerCta` to 8** (two key-range halves). Doubles the warps per CTA at the same shared
   memory, i.e. 25% occupancy -- the only route that raises occupancy without touching the buffers.
   It doubles the already-redundant QK^T, and `k_s`/`v_s` would have to be double-buffered for two
   key ranges, which costs the 16.9 KB this is trying to save. Not obviously available.
3. **A topology for the small shapes**: no 32-row Q tile (6 live rows), no 4x redundant QK^T, K/V
   staged once per CTA rather than per row-pass. This is the change that actually addresses the root
   cause, and it is a kernel rewrite rather than a constant.

Nothing was changed. The first attempt -- sizing `q_s` by the live row count -- was written, found
unsafe against `volta_qp_get_i`'s 32-row A operand, and reverted; the tree is byte-identical.
Recommendation: (1) is a bounded experiment worth one round even though the instruction arithmetic
pessimistic, because the occupancy estimate is the measured one and the instruction estimate is not;
(3) is where the 20%+ lives and should be scoped as a project.








**A run-length artifact, recorded because it looked like a 24% win.** A probe at `--max-new 64` read
K = 7 at 90.9 t/s and acceptance 3.94 against K = 1's 70.2 and 1.82, i.e. 11.0 against 14.5 ms/token.
The round cost is identical at both lengths (43.3 against 43.4 ms) -- what changes is the acceptance,
because the first 64 tokens of this trace are easier to draft than the whole 128. Any acceptance or
L comparison has to fix `--max-new`.

### The acceptance dead ends

* **`--spec dflash2` is unreachable, and not for a kernel reason.** The CMakeLists does compile Volta
  implementations for five of its pieces (dynamic conv, attn-input, linear_topk, **candidate_selector**,
  rmsnorm_rope), and `dflash2_sm70_stub.cu` throws for the rest -- but the run does not get that far:
  `error: DFlash2 was selected but the artifact has no DFlash2 weight bundle`. The codebook drafter
  needs weights `bonsai2_27b_swift_pq2.ninfer` does not carry, so it is an artifact-repackaging job,
  not a port. `--spec dflash` reports `selected masked draft backend is not supported by this target`.
* **The lookup path is not a general drafter.** `lookup_draft` looks for the nearest prior occurrence
  of the last **16 tokens** and copies what followed; the caller only takes it when
  `agrees_with_mtp && max_lookup_extent > draft_window`, i.e. when the context literally repeats and
  the MTP head already agrees. On ordinary text `lookup_draft` returns `nullopt` and the MTP head
  drafts alone.
* **The proposal head is already the better one.** `ProposalHead::Full` is the default and
  `--lm-head-draft` selects `Optimized`; every number in this tree was taken with `Optimized`, and it
  wins on both metrics -- acceptance 1.65 against 1.57 at K = 1, 2.49 against 2.31 at K = 5, 2.59
  against 2.35 at K = 7, with a round 0.1-0.7 ms cheaper as well.

Acceptance on ordinary text is therefore bounded by the MTP draft head, which lives in the artifact.

### Where both directions now stand

| direction | remaining | what it needs |
|---|---:|---|
| round latency, SIMT activation side | 0 | three mechanisms measured and refused |
| round latency, weight stream | 0 | 789 GB/s = 88% of the card |
| round latency, QPN small-T | 0 | NACC +21.8% is all it had; min-blocks negative |
| round latency, rmsnorm fusion | ~2.5% | non-bit-identical, so PPL |
| round latency, `silu_and_mul` fusion | ~1.3% | row -> warp remap in the SwiGLU producer |
| round latency, MTP head per-draft-token projections | 2-4% | merge the draft tokens' lm_head call |
| acceptance | 0 in-tree | the MTP head is artifact-bound; dflash2 needs a weight bundle |

Both directions are close to exhausted inside this tree. The remaining round-latency work is a set of
small, non-bit-identical fusions worth ~4-8% in total, and the remaining acceptance work is outside
the tree.

### Reproduce

```bash
bash /root/qpn_minblocks.sh          # -> /root/ninfer_ab/qpnminb
bash /root/dflash2_probe.sh          # -> /root/ninfer_ab/dflash2
bash /root/k_sweep.sh                # -> /root/ninfer_ab/ksweep
bash /root/head_ab.sh                # -> /root/ninfer_ab/head
python3 /root/k_extract.py /root/ninfer_ab/ksweep
```






