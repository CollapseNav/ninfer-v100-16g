# V100-16GB port: tuning log

> Development narrative moved out of the top-level README. It records what was measured,
> what was refuted, and what is still open across the ternary port and the gguf-mixed tuning
> rounds. For user-facing usage see [../README.md](../README.md).

## V100-16GB ternary port (branch `v100-16g`)

This branch adds the **PQ2_0_G128** (2-bit ternary) path used by community requantized artifacts --
the reference one here is `bonsai2_27b_swift_pq2.ninfer` (artifact identity `qwen3.8-27b` /
`groupwise-int`, 7.45 GiB of weights), sized so the model plus KV fits a single **V100-SXM2-16GB**.

The upstream tables below are NVFP4 and groupwise-int results on 32GB cards and do **not** apply to
this artifact: Volta has no NVFP4 route (that kernel is inside `#ifndef NINFER_VOLTA_BUILD`), the
weights are 2-bit ternary rather than 4-bit, and this decode is issue/latency-bound rather than
HBM-bound. Measured on the 16GB SXM2, greedy, INT8 KV, one resident model:

| scenario | prefill | decode | speculation |
|---|---:|---:|---|
| natural prose, no speculation | 1.21-1.23k tok/s @4k | **51.1 tok/s** with `NINFER_TERNARY_ROTATE_SPLIT=16`, 48.7 with no environment variables | -- |
| ordinary engineering task, thinking on | 1.13-1.15k tok/s | 49.8-51.4 tok/s | **K=1: 59.3-61.7 tok/s**, draft acceptance 58.5-73.9% |
| context reproduction (repetitive text) | -- | -- | K=7: **122-124 tok/s**, acceptance 97.1% |

Against the pre-port snapshot recorded in `docs/ternary-port.md`: decode **44.6 -> 51.1 tok/s
(+14.6%)** on the same route, prose MTP K=1 **51.8 -> 64.1**, context lookup **83.7 -> 122-124**, and
the marginal cost of one verified token **0.60 -> 0.30-0.37** of a decode step. What was measured,
what was refuted, and what is still open lives in `docs/ternary-port.md`; the remote build/run
recipe is in `docs/e5-ssh-workflow.md`.

```bash
cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-sm70 --parallel 8
./build-sm70/apps/ninfer-serve models/bonsai2_27b_swift_pq2.ninfer \
    --host 0.0.0.0 --port 8080 --cors --max-context 131072 --kv-capacity auto \
    --kv-dtype int8 --prefill-chunk 4096 --spec mtp --draft-tokens 1 --lm-head-draft
```

The binaries link `libavformat.so.60`, which the host here does not provide, so they are run through
the `ninfer-v100-buildenv:cu128` image as `docs/e5-ssh-workflow.md` describes.

## V100-16GB gguf-mixed tuning round (2026-10-05/06)

A tuning round over the *other* 16GB artifact — `swift15_iq2xs_mtp.ninfer` (identity
`qwen3.8-27b` / `gguf-mixed`, 8.15 GiB, 11 GGUF formats mixed across 401 tensors) — on the
V100-SXM2-16GB with `--kv-dtype int8` group-64, MTP K=2, greedy, `--max-new 128`, CUDA graphs, one
resident model. The three arms are prefixes of a single text (`wikitext/00` + `pg19/00`), so they
differ only in depth; two repetitions each.

| context | prompt | prefill | decode | MTP acceptance |
|---|---:|---:|---:|---:|
| short | 4,084 | 1,075 tok/s | **92.3 tok/s** | 2.82 tok/round |
| medium | 16,351 | 1,055 tok/s | **83.2 tok/s** | 2.76 |
| long | 130,771 | 557.7 tok/s | **52.6 tok/s** | 2.80 |

The baseline row is at `--draft-tokens 2`, which is the value `ternary-port.md` ships. Sweeping the
window afterwards — same fixture, same settings, only `--draft-tokens` varying — puts the peak
somewhere that sweep never looked:

| `--draft-tokens` | 4K decode | 16K decode | 128K decode |
|---:|---:|---:|---:|
| 1 | -- | 70.2 | -- |
| **2 (shipped)** | 86.8 | 83.3 | 53.3 |
| **3** | 85.3 | **93.5** | **59.9** |
| 4 | -- | 86.8 | -- |
| 5 | **89.8** | 81.7 | 48.6 |

**`--draft-tokens 3` is confirmed at 128K and not at 16K.** The sweep above is one fixture, and MTP
acceptance is text-sensitive — the `--prefill-chunk` A/B below moved it 2.82 -> 2.65 purely by
reordering the reduction — so a second, disjoint text (`wikitext/01 + pg19/01`, same prefix-slicing,
same depths) and a third real-world fixture were run against it:

| fixture | depth | K=2 | K=3 | Δ |
|---|---|---:|---:|---:|
| text A (`wikitext/00 + pg19/00`) | 128K | 53.3 | 59.9 | **+12.4%** |
| text B (`wikitext/01 + pg19/01`) | 128K | 50.0 | 55.3 | **+10.6%** |
| text A | 16K | 83.3 | 93.5 | +12.2% |
| text B | 16K | 85.3 | 88.1 | +3.3% |
| `prose16k.json` (19,930 tok) | 16K | 68.3 | 67.8 | −0.7% |
| text A | 4K | 86.8 | 85.3 | −1.7% |

**At 128K it holds on both texts (+10.6 / +12.4%); at 16K it does not** — three fixtures span
−0.7% to +12.2% with a median of +3.3%, which is text luck rather than a speedup. So
`--draft-tokens 3` is justified **for a serve that must reach 128K** — its worst measured depth is
−1.7% at 4K — and it is not a general upgrade.

Window 5 wins only at 4K and collapses at 128K (48.6 tok/s on an 87.0 ms round), so wider is not
uniformly better. The window axis is also not monotone — 16K reads 70.2 / 83.3 / **93.5** / 86.8 /
81.7 — which is why `ternary-port.md`'s "monotone and steep", drawn from the {2, 4, 6} set at 31.2k,
does not generalise: **window 3 was never in that table**. Nor does it conflict with
`decode-round-2026-10-01.md` Round 13, which sweeps K = 1..7 on `real_task` and finds K = 2/3/4 a
dominated basin — that sweep runs the **PQ2/QPN** route on `bonsai2_27b_swift_pq2`, where the QPN band
edge at T = 5 makes K = 4 verify through SIMT while K = 5 verifies through QPN and comes out 20%
cheaper while accepting more. The gguf route has no such band: two routes, two curves.

**Kept (3).**

1. `--prefill-chunk 8192` whenever `max-context <= 16K` is worth **+9.1% prefill @4K** and
   **+8.4% @16K** with decode unchanged — it cannot be used at 128K, see the wall below.
2. `--draft-tokens 3` for a 128K-capable serve: **+10.6..12.4% at 128K on two disjoint texts**,
   −1.7% at 4K, no confirmed gain at 16K (recheck table above). Deployed on `ninfer-serve`.
3. `small_t_i8_volta.cuh` stages the KV gather 16 codes at a time (one `int4` load replacing two
   `int2`): bit-identical (md5-equal at both depths, K = 1/3/5 including the 5-warp CompactTail path)
   and worth **128K decode 52.6 -> 53.3 tok/s (+1.33%)**. `ncu` attributes the stall this removes to
   `long_scoreboard`, 103.5% -> 69.3% of issue-active.

**Refuted on measurement (10).** The documented 7-9 ms per-round stall does not reproduce — five
configurations measure 95.9-97.4% GPU-busy and the only hole above 1 ms sits on the
prefill→decode boundary. `--lm-head-draft` collapses acceptance to **1.00** on this artifact
(decode 92.4 -> 33.5), so the "every number used Optimized" note in `decode-round` does not apply
here. `--kv-dtype fp8` is 65-90% slower; `NINFER_CAUSAL_WAVE_ALIGN`'s default already wins against
0/2/3; `NINFER_GGUF_F16_PREFILL=0` fails at startup; lowering `NINFER_GGUF_DEQUANT_SCRATCH_MIB`
costs more than the chunk it buys. The 128K prefill's flash-attention is at its roofline — ncu reads
**DRAM 81.5% (735 of 902 GB/s)** on the kernel that is 53.8% of that prefill. And
shared-staging the vec kernel's weight blocks (the universal form of the per-format repack) is
bit-identical but **-17..-29%**: it relocates `decode`'s loads instead of removing them — 192
global loads become 66 global **plus** 192 shared.

**Two walls.** The runtime reservation follows `--max-context`, not the prompt: at 131072 tokens of
int8 KV, `--prefill-chunk 4096` is 108 MB over budget and `--lm-head-draft` 59 MB over, so at 128K
the chunk is pinned to 3072 and the optimized proposal head cannot be loaded. `--max-context` itself
is speed-neutral (mc 8192 and mc 131072 measure identically at 4K). And the KV attention kernel's
**18.4% occupancy with 55.9% of cycles holding no eligible warp** is not improvable: `q_s` must keep
its 32 rows because `volta_qp_get_i()` returns `threadIdx.x & 31` (the QK fragment addresses rows
0..31 whatever the row capacity is), `k_s`/`v_s` are pinned to `StageKeys = 8` by the mma N
dimension, and 25,600 B > 98,304/4. Shrinking `Br` past that faults with an invalid `__shared__`
read that `compute-sanitizer` pins to `small_t_i8_volta.cuh:472`. The "+37.96%" ncu attaches to that
kernel is an occupancy heuristic, not a budget — the same class of estimate cashed at only
**+3.5..3.9%** on the previous 2→3 CTA step (`51632bfb`).

**Copying vs explaining.** The 219 tok/s at the top of this file is a *context-lookup* number: a
round cannot commit more than ~16 tokens, and 219 tok/s at the documented 64 ms round needs 12.91 of
them. The same mechanism measured here on `lookup10`/`lookup16` ("repeat this sentence ..."):

| prompt | K | decode | acceptance |
|---|---:|---:|---:|
| natural prose, 4K | 2 | 92.3 | 2.82 tok/round |
| `lookup10` | 2 / 5 / **7** | 149.4 / 175.1 / **171.5** | 7.27 / 10.90 / **12.11** |
| `lookup16` | **7** | **178.4** | **12.70** |
| `lookup10`, thinking on | 2 | 113.0 | 3.85 |

At K=7 the acceptance is **12.11 / 12.70 — identical to the values recorded in
`docs/decode-round-2026-10-01.md`** for a different artifact on a different card, so the mechanism
and the tokens-per-round agree exactly; what differs is round cost (70.6 ms here against 51.6 ms
there), i.e. the weight route and the harness (`--max-new 128` full-run average against
`ninfer_bench`'s ten measured rounds after two warmups). On this artifact alone, copying against
explaining is **92 -> 178 tok/s (+93%)**, and turning thinking back on drops it to 113.

### Three more routes, all measured, one kept

Everything below was tried against the same fixtures and the same md5 gate. Only A ships.

**A — checkpoint pool (shipped as `--max-shared-prefixes 16`).** Twelve *distinct* ~6k-token
contexts round-robin, two passes, one variable at a time:

| arm | round-2 hits | round-2 TTFT |
|---|---:|---|
| shipped default (`--max-shared-prefixes` **4**) | **4 / 12** | 0.14 s hit, **4.9 s miss** |
| `--max-shared-prefixes 16` | **8 / 12** (reproduced twice) | **0.11-0.14 s** on the hits |
| `+ --host-state-slots 16` | 1 / 12 (reproduced twice) — **harmful** | |
| `+ --host-kv-mib 16384` | 8 / 12 — no change | |

The first gate is `shared`, which defaults to `max(max-concurrency, 4)` = **4**: exactly four
prefixes survived the first pass, matching the log's `| shared 4`. Raising it doubles the hit rate
and takes a retained prompt from **4.9 s to 0.11 s (~43x)**. The second gate (8 -> 12) was not
located: `--device-state-slots 12` cannot start on this card (reservation needs 8.29 GB + 1.07 GB
headroom against 7.81 GB free after weights), and both host knobs are ruled out above. The serve
runs `--max-shared-prefixes 16`.

**B — GGUF weight repack (reverted).** A block-local regroup of six formats (IQ2_XXS, IQ2_XS,
IQ2_S, Q2_K, IQ3_XXS, IQ3_S) plus a decoder-only change to IQ1_M, applied on the CPU during load —
legal because no metadata crosses a block boundary, so the transform round-trips byte-exactly over
all 401 real tensors. `cuobjdump -sass` proves the instruction count really fell:

| format | emitted loads / `decode()` | | control IQ1_S (not repacked) |
|---|---:|---|---|
| Q2_K | 11 -> **4** (−64%) | | 4 -> 4, byte-identical |
| IQ1_M | 8 -> **3** (−63%) | | |
| IQ3_S | 9 -> 5 (−44%) | | |
| IQ2_S / IQ3_XXS | 7 -> 4 (−43%) | | |
| IQ2_XS / IQ2_XXS | 6 -> 4 / 5 -> 3 | | |

**33-64% fewer load instructions over 68.7% of the weight payload, md5 byte-identical, weights
8.1506 -> 8.2723 GiB (+1.45%, 14.8 MB of a measured 141.5 MB budget left) — and decode moved
+0.1% / 0.0%.** So this kernel is not instruction-bound: `ncu` puts 34-47% of its cycles in
`long_scoreboard` waiting on the *last* the dot product depends on, and independent loads only
shrink issue slots. A single wide load needs 8-byte-aligned records, which cost 387.6 MB = **2.7x
the budget**, so no subset of the ten is affordable either. Reverted; md5 re-verified.

**C — the fourth CTA (reverted, −5% to −66%).** The 1,024 B that stood between 25,600 and
98,304/4 was two buffers the file had called untouchable: the row pad `SmemPad 8 -> 0`
(32+8+8 rows x 16 B = **768 B**) and `physical_pages_s`, a pure prefetch cache of `block_table`
(**256 B**). **768 + 256 = 1,024 exactly**, with the three arrays then summing to
16,384+4,096+4,096 = 24,576. An in-row XOR at 8-element groups replaces the pad's bank-conflict
property. The build confirmed the structure: `SHARED 25,600 -> 24,576` and `REG 168 -> 128`, i.e.
both limits now admit four CTAs.

It still lost, badly:

| depth | baseline (K=3) | fourth CTA | Δ |
|---|---:|---:|---:|
| 4K | 85.3 | 80.9 | **−5.2%** |
| 16K | 93.5 | 75.6 | **−19.1%** |
| 128K | 59.9 | 20.2 | **−66.3%** |

The register cut (`__launch_bounds__(…, 4)`) moved the compiler from **STACK 16 to STACK 400 bytes**
— it bought the 40 registers back with rematerialisation, exactly the failure
`docs/ternary-port.md` records at 15-19% on the vec kernel and at 3.9% on the reverted int8-V
attempt. Dropping `physical_pages_s` also left a dependent `block_table` read in the hot loop,
which is what the 128K column punishes. md5 initially **failed** — `swizzled(k_dst) + 8` is not
`swizzled(k_dst + 8)`, because XOR does not preserve successor, so `hi` landed in the wrong group
and every row's K/V was corrupted (the model emitted token 0 repeatedly); fixing both halves
restored byte-identity, after which the timing was still decisively negative. Reverted.

> **Build hygiene found on the way:** `ninja -t deps <object>` reported `#deps 0` for
> `small_t.cu.o`, i.e. no header was registered, so editing its `.cuh` produced
> `ninja: no work to do`. Forcing one rebuild re-registered them. If a header edit seems ignored,
> check with `ninja -t deps` and `touch` the `.cu`.


---

## Appendix: original deployment profile (gguf-mixed tuning round)


Rebuild after any source change — all three apps, or the server keeps linking a stale file:

```bash
docker run --rm -v /root/ninfer-v100:/src -w /src ninfer-v100-buildenv:cu128 \
  bash -lc "cd /src && ninja -C build-sm70 -j\$(nproc) \
            apps/ninfer apps/ninfer-serve apps/ninfer-perplexity"
```

Then start it. Every flag below is a decision the sections above paid for:

```bash
docker run -d --name ninfer-serve --restart unless-stopped \
  --gpus all --shm-size 64m -p 8080:8080 \
  -e NINFER_TERNARY_ROTATE_SPLIT=16 \
  -e NINFER_MTP_LOOKUP_MATCH=6 \
  -v /root/ninfer-v100:/src \
  -v /root/ms:/models:ro \
  ninfer-v100-buildenv:cu128 \
  /src/build-sm70/apps/ninfer-serve /models/swift15_iq2xs_mtp.ninfer \
    --host 0.0.0.0 --port 8080 --cors \
    --max-context 131072 --kv-capacity auto \
    --kv-dtype int8 --prefill-chunk 3072 \
    --spec mtp --draft-tokens 3 --max-shared-prefixes 16
```

| flag | why (all measured on this artifact) |
|---|---|
| `--draft-tokens 3` | **+11.6% @16K, +12.2% @128K** against the shipped 2, and the peak on *both* workloads — prose 93.5 and agent-shaped 81.6 (K2 80.6 / K4 79.2 / K5 67.9). Window 5 also collapses at 128K: 48.6 on an 87.0 ms round |
| `--max-shared-prefixes 16` | the default resolves to `max(max-concurrency, 4)` = **4**; on a 12-context round-robin that caps pass-2 hits at **4/12**. Raising it gives **8/12** and takes a retained prompt from **4.9 s to 0.11 s (~43×)** |
| `--prefill-chunk 3072` | memory-capped at `--max-context 131072`: chunk 4096 needs **+108 MB** that the card does not have, and 1024 costs **−14%** on cold prefill (4.07 s vs 3.50 s for the same 3.7k prompt) |
| `--kv-dtype int8` | `fp8` is **−65..90%** on both phases — Volta has no fp8 path |
| `--max-context 131072` | speed-neutral by itself (mc 8192 ≡ mc 131072 at 4K); sized to the workload, and it is what pins the chunk to 3072 |
| *(absent)* `--lm-head-draft` | collapses acceptance to **1.00** — decode 92.4 → 33.5 (**−64%**) |
| *(absent)* `--no-thinking` | worth **+17.4%** decode, rejected on output quality |
| `NINFER_TERNARY_ROTATE_SPLIT=16` | carried over from the ternary artifact's sweep (51.1 vs 48.7 unset); **not re-measured on gguf-mixed** |
| `NINFER_MTP_LOOKUP_MATCH=6` | carried over for copy-heavy traffic (documented +20%); **inert on this traffic — 0 of 14 real requests hit lookup** |
| `--host-state-slots 8` / `--host-kv-mib 8192` (defaults) | raising `--host-state-slots` to 16 **halved** pass-2 hits (8/12 → 1/12, reproduced twice) |

Prefix reuse itself is already fast where it lands: a 100% hit returns first token in
**7 / 23 / 43 / 52 ms** for 4K / 16K / 32K / 60K retained prefixes — so the tuning above is the
whole of what the deployment side currently offers.

