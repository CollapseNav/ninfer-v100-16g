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



