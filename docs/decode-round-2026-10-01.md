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
