# NInfer — `v100-16g` fork

> Up to 219 decode tok/s from Qwen 3.8 27B on a single V100. With software NVFP4 on Volta.

**Upstream:** [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) — NInfer, a from-scratch
C++/CUDA inference engine optimized for selected Qwen checkpoints on NVIDIA Tesla V100.
**This fork:** [CollapseNav/ninfer-v100-16g](https://github.com/CollapseNav/ninfer-v100-16g), branch
`v100-16g` — 157 commits on top of upstream, aimed at running larger models and longer contexts on a
single **V100-SXM2-16GB**.

Text, image, and video input through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs. The
runtime is intentionally narrow: one GPU, one resident model, 1–8 active requests.

---

## Upstream vs this fork

Read this first: it decides which numbers apply to you and which code you are looking at.

| Area | Upstream | `v100-16g` |
|---|---|---|
| Engine core — scheduler, chunked prefill, CUDA-graph decode, FIFO ingress | ✅ | tuned for `sm_70` |
| KV storage formats (BF16 / INT8 / FP8), offline perplexity CLI | ✅ | — |
| **Device/Host KV tiering, context cache, pressure planner** | ✅ | **extended — see [KVMem](#kvmem-device-pool-smaller-than-the-context-this-fork)** |
| Volta attention and prefill kernels | ✅ | **+ KVMem mask plumbing** |
| MTP, DFlash, DFlash2 speculative decoding | ✅ | — |
| NVFP4 weights (not available on Volta) | ✅ | — |
| 32GB-class artifacts (Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B) | ✅ | — |
| **PQ2_0_G128 ternary weights** | — | ✅ |
| **GGUF mixed-format linear layers** — IQ1_S/M, IQ2_S/XS/XXS, IQ3_S/XXS, IQ4_NL/XS, Q2_K–Q6_K, Q4_0, Q8_0 | — | ✅ |
| **KVMem: a device KV pool smaller than the logical context** | — | ✅ |
| **16GB-class artifacts, their context ceilings and throughput** | — | ✅ |

Two things follow from that table:

- **KVMem is an extension, not a replacement.** Upstream already tiers KV between the Device and a
  pinned Host arena (see [Resource-aware long-context reuse](#resource-aware-long-context-reuse-upstreamkvmem-extends-it)).
  KVMem adds the ability to size the *device* pool below the logical context, plus relevance-aware
  retention and cross-owner reclaim on top of that tiering.
- **Upstream performance tables do not transfer to this fork's artifacts.** The NVFP4 and
  groupwise-int figures below are 32GB-card results; Volta has no NVFP4 route, the 16GB artifacts use
  2-bit ternary or mixed GGUF weights, and this decode is issue/latency-bound rather than HBM-bound.
  Use the [16GB tables](#throughput-this-fork).

---

## Build

```bash
cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-sm70 --parallel 8
```

The binaries link `libavformat.so.60`, which some hosts do not provide. Either install FFmpeg or build
and run through the container images (see [Docker](#docker)):

```bash
docker run --rm -v /root/ninfer-v100:/src -w /src ninfer-v100-buildenv:cu128 \
  ninja -C build-sm70 apps/ninfer apps/ninfer-serve apps/ninfer-perplexity
```

Rebuild **all** the apps after a source change — otherwise the server keeps linking a stale binary.

---

## Run on a 16GB V100 *(this fork)*

A complete 16GB-V100 profile with KVMem enabled. Every flag is a decision the measurements paid for;
the justification cites the artifact the number came from.

```bash
docker run -d --name ninfer-serve --restart unless-stopped \
  --gpus all --shm-size 64m -p 8080:8080 \
  -e NINFER_KV_WINDOW=16384 \
  -v /root/ninfer-v100:/src \
  -v /root/ms:/models:ro \
  ninfer-v100-buildenv:cu128 \
  /src/build-sm70/apps/ninfer-serve /models/swift15_iq2xs_mtp.ninfer \
    --host 0.0.0.0 --port 8080 --cors \
    --max-context 262144 --kv-capacity 24576 \
    --kv-dtype int8 --host-kv-mib 12288 \
    --prefill-chunk 512 \
    --max-concurrency 1 \
    --spec mtp --draft-tokens 3
```

| flag | what it does | why this value |
|---|---|---|
| `NINFER_KV_WINDOW` | **enables KVMem** and sets the resident window in tokens | Unset, the engine behaves exactly as upstream (the dense path is bit-identical). Set, the pool may be smaller than the context. `16384` is the validated window |
| `--max-context` | logical context limit per sequence | `262144` is the native cap of the Qwen3.8-27B variant; larger values are rejected at startup |
| `--kv-capacity` | device KV pool, in tokens | With KVMem the pool only has to hold the resident window: the minimum is `window + prefill-chunk + 8 pages`. `24576` (384 pages) leaves headroom |
| `--host-kv-mib` | Host-tier KV arena (the demotion target) | 262,144 tokens needs ~8.9 GiB of Host KV; `12288` fits it. The upstream default is 8192 |
| `--kv-dtype int8` | KV storage format | `fp8` is −65..90% on both phases (Volta has no fp8 path). **KVMem requires `int8`** — it is the only storage whose Volta attention kernels honour the retrieval mask; other values are refused at startup |
| `--prefill-chunk` | tokens per prefill step | Larger is faster (+26% going 768 → 3072 on swift15) but costs workspace VRAM; `512` keeps a 262K context startable |
| `--max-concurrency N` | simultaneous requests | `1`..`8`. Every lane keeps its own resident window, so the pool must hold `N × (window + prefill-chunk + sink) + 8` pages — the startup error prints the minimum. `2` measured: two cold 40K prompts, both answered correctly |
| `--draft-tokens 3` | MTP speculative window | +11.6% @16K and +12.2% @128K against the shipped 2, and the peak on both prose and agent-shaped traffic |
| *(absent)* `--lm-head-draft` | LM-head draft head | Adding it collapses acceptance to 1.00 — decode 92.4 → 33.5 (−64%) |
| *(absent)* `--no-thinking` | forces non-thinking mode | Worth +17.4% decode, but rejected on output quality |

Startup prints the resolved layout — check it before trusting a profile:

```
capacity | KV 24,576 tokens, int8, explicit | pages 384/4,096 | runtime 2.67 GiB | free 213.5 MiB
```

`384/4,096` is the whole point: **384 device pages serving a 4,096-page logical context (10.7×)**.

---

## KVMem: device pool smaller than the context *(this fork)*

A 16GB V100 cannot hold the KV for a 262,144-token context. KVMem keeps only a resident window on the
device and demotes the rest to the pinned Host arena, then attends over the resident set. When the pool
fills, retention chooses *which* pages leave by relevance to the current query rather than by age, so
the pages a request actually needs stay resident. Device pages still held by an earlier cached turn are
reclaimed when a new request needs room.

Enable it with one environment variable, and size the pool to the **window**, not to the context:

```bash
-e NINFER_KV_WINDOW=16384
--max-context 262144 --kv-capacity 24576 --host-kv-mib 12288 --kv-dtype int8 --max-concurrency 1
```

### Measured cost

Same prompt, same chunk, 128K context, `swift15_iq2xs_mtp.ninfer`:

| | dense | KVMem |
|---|---:|---:|
| device pool | 2,048 pages = 131,072 tokens | **384 pages = 24,576 tokens** |
| runtime VRAM | 5.79 GiB | **2.23 GiB** |
| free VRAM | 1.29 GiB | **4.85 GiB** |
| prefill | 525.4 tok/s | 520.3 tok/s (−1.0%) |
| decode | 59.6 tok/s | 59.5 tok/s (−0.2%) |
| answer (needle at the start of a 124K prompt) | correct | correct |

**About 1% slower for 3.56 GiB back**, and the context ceiling moves from ~164K to 262K.

### Limits

- **Multi-lane is budget-partitioned.** With `--max-concurrency N` each lane holds at most
  `pool / N` Device pages (window + chunk + sink at minimum); a lane never grows past that and
  spills its own pages, so lanes cannot starve each other. Retention scoring and the attention mask
  are per lane. **Prefill is still serialized** (the engine runs one chunked prefill at a time):
  a second long prompt waits for the first to finish prefill, then both decode together.
- **INT8 KV only.** Other storages are refused at startup.
- **Retention, not recall.** A relevant page is *kept resident*, so a single-turn long prompt whose
  question comes last works. In a *multi-turn* conversation, a page that only a later turn makes
  relevant cannot be brought back.
- **Prefill slows with depth**, as one would expect: ~680 tok/s at 32K, ~460 tok/s at 128K.
- **No quality measurement yet.** Every number above is throughput. No perplexity has been run, so how
  small the resident window can go before output quality degrades is an open question.

### Multi-lane measured *(this fork)*

`coder390`, `NINFER_KV_WINDOW=16384`, `--max-concurrency 2`, pool 544 pages (`--kv-capacity 34816`),
two cold 40K-token prompts with different needles, sent concurrently:

| | lane 1 | lane 2 |
|---|---:|---:|
| prefill | 655 tok/s | 668 tok/s |
| decode | 77 tok/s | 77 tok/s |
| answer | own needle ✅ | own needle ✅ |

Prefill is serialized (lane 2's TTFT includes the wait: ~61 s queue), decode overlaps. Five cold
pairs — ten concurrent requests with different needles — all answered with their own needle, and a
cached repeat restored its checkpoint in 1 s. Single-lane profiles are unchanged
(`ring_lane_budget_pages` returns the whole pool when N = 1): the 77K needle still answers correctly
at 574 tok/s prefill / 62 tok/s decode. CUDA graphs stay enabled; the per-lane mask matrix is
allocated at construction so a captured decode graph bakes a stable pointer, not the nullptr it
would have captured from a lazily allocated mask (that nullptr is what mixed two prompts' K/V).

### Medium context under concurrency: raise the prefill chunk *(this fork)*

`coder390`, four concurrent requests of **15,442 prompt tokens** each, `--max-concurrency 4`, MTP
draft 3, 256 output tokens:

| profile | wall | prefill | TTFT | answers |
|---|---:|---:|---:|---:|
| `--prefill-chunk 512`, `NINFER_KV_WINDOW=4096`, pool 312 pages | 86.9 s | 720 tok/s | 21.6 s | 4/4 |
| **`--prefill-chunk 1024`, `NINFER_KV_WINDOW=1024`, pool 152 pages** | **71.0 s** | **890 tok/s** | **17.5 s** | **4/4** |

−18% wall, +24% prefill, −19% TTFT. The prefill GEMM is the int8/dp4a MMQ path, so doubling the
chunk doubles its row count and the dp4a pipeline utilises better — the same effect that makes a
wider decode batch cheaper per token.

The chunk is bounded by VRAM: `prefill_chunk 1024` needs the smaller window to fit at four lanes
(3.45 GB of runtime reservation vs 3.29 GB available); at two lanes the larger window fits. Chunk
2048 does not fit at four lanes with this 12.3 GiB artifact.

Two limits this profile does not fix: **prefill is still serialized** (each request's TTFT is its
own prefill time, and four requests' wall is four prefills back to back), and a lane that is already
decoding drops from ~70 tok/s to **3–4 tok/s** while another lane prefills.

Prefill throughput tracks the chunk's row count, so spend the artifact's VRAM headroom on it:
`swift15` (8.15 GiB) fits `--prefill-chunk 2048` at four lanes and measures **1003 tok/s / 62.9 s
wall / 15.7 s TTFT** against 870 / 72.3 s / 18.0 s for chunk 1024 (+15%). `coder390` (12.3 GiB)
reaches chunk 2048 too once `--device-state-slots 2` gives back the checkpoint state images the
default (one per lane) reserves: **1015 tok/s / 62.1 s wall / 15 s TTFT, 4/4 answers correct**,
against 890 / 71.0 s / 17.5 s at chunk 1024.

`--device-state-slots 0` is not usable: requests fail outright. One extra slot per lane is enough
to keep every lane concurrent, and the rest of the default reservation is checkpoint capacity this
profile does not need.

### Context-cache reuse does not survive the ring *(this fork)*

The KVMem ring gives a sequence more logical context than the device pool holds, but a checkpoint it
publishes cannot be restored: the ring keeps retention, not recall. The context cache therefore
stops paying off under the ring.

Measured on `swift15`, four concurrent 15,424-token prompts that share a 15,412-token prefix, with
that prefix first prefilled once by a probe request:

| | cache hit | prefill per request | TTFT | wall |
|---|---:|---:|---:|---:|
| ring (`NINFER_KV_WINDOW=1024`, pool 13,824) | **0** | 15.4K tokens | 14.7–15.2 s | **60.8 s** |
| dense (`--kv-capacity 24576`) | **15,405** | **19 tokens** | **0.1–0.2 s** | **3.7 s** |

Dense mode costs nothing when the prompts do *not* share a prefix: four distinct 15,424-token
prompts take 63.0 s dense against 62.9 s ring at the same 1,035 tok/s prefill. So for medium context
the dense configuration is strictly better, and the ring is worth its lost cache only when a
sequence's logical context exceeds the device pool (the 77K single-request profile).

Simultaneous arrivals do not coalesce in either mode: four shared-prefix requests sent together
report a 0-token cache and each prefills the whole prefix, because the first has to publish before
the rest can reuse. Turning that around needs prefill coalescing (one prefill of the common prefix,
then per-request suffixes), which this fork does not implement.

The decode rate during another lane's prefill is arithmetic, not a scheduling gap: the engine
already alternates decode and prefill units (log: `prefill 733 tok/s | decode 1.2 tok/s | running 2
(prefill 1, decode-ready 1)`), but one 1024-token chunk costs ~1.1 s, so the decoding lane gets one
step per chunk. Smaller chunks buy responsiveness at the throughput listed above.

---

## Models on a 16GB V100 *(this fork)*

Three Qwen3.8-27B requantizations, all verified to run on this card. They share one geometry —
64 layers = 16 full-attention (paged KV) + 48 GDN (recurrent state), `kv_heads=4`, `head_dim=256`,
64-token pages — so 33,792 bytes of paged KV per token at INT8.

| model | weights | file | artifact |
|---|---:|---|---|
| `swift15_iq2xs_mtp.ninfer` | 8.15 GiB | 9.42 GB | `gguf-mixed` / IQ2_XS |
| `bonsai2_27b_swift_pq2.ninfer` | 7.12 GiB | 8.31 GB | `groupwise-int` / PQ2_0_G128 |
| `Qwen3.8-27B-Coder390-…-Q2LynnStyle-MTP.ninfer` | 12.3 GiB | 13.27 GB | `gguf-mixed` / Q2-LynnStyle |

### Context ceiling

| model | dense (KV all resident) | with KVMem |
|---|---|---|
| swift15 | 163,840 works / 180,224 fails | **262,144** |
| bonsai2 | 180,224 works / 196,608 fails | **262,144** |
| coder390 | 57,344 works / 61,440 fails | **262,144** |

262,144 is the variant's native context (`qwen3_6_27b/impl/config.h`, `kNativeContext`). Dense mode is
capped by VRAM and varies with weight size; **coder390 gains 4.6× from KVMem.**

### Throughput *(this fork)*

KVMem profile (`NINFER_KV_WINDOW=16384`, pool 24,576, `--prefill-chunk 512`, 256 output tokens, greedy):

| model | depth | prompt | prefill | TTFT | decode | MTP acceptance |
|---|---|---:|---:|---:|---:|---:|
| swift15 | short | 1,049 | 681.9 tok/s | 1.5 s | 110.9 tok/s | — |
| | medium | 32,049 | 679.8 tok/s | 47.1 s | 89.2 tok/s | 89.4% |
| | long | 128,049 | 461.3 tok/s | 277.6 s | 55.0 tok/s | — |
| bonsai2 | short | 1,049 | 727.6 tok/s | 1.4 s | 92.0 tok/s | — |
| | medium | 32,049 | 690.7 tok/s | 46.4 s | 64.3 tok/s | 74.1% |
| | long | 128,049 | 467.0 tok/s | 274.2 s | 55.9 tok/s | — |
| coder390 | short | 1,049 | 688.1 tok/s | 1.5 s | 108.2 tok/s | — |
| | medium | 32,049 | 684.8 tok/s | 46.8 s | 93.1 tok/s | 100.0% |
| | long | 128,049 | 463.8 tok/s | 276.1 s | 62.9 tok/s | — |

Prefill is flat to 32K and falls to ~460 tok/s at 128K; decode falls with depth. All three curves have
the same shape (same architecture and KV geometry). Speculation payoff is model-dependent — coder390
accepts 100%, swift15 89.4%, bonsai2 74.1%, and decode tracks it.

### Where the artifacts come from

| model | source |
|---|---|
| swift15 | [fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer](https://modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer) |
| bonsai2 | [fyb423/Swift-Bonsai-2-27B-NInfer](https://modelscope.cn/models/fyb423/Swift-Bonsai-2-27B-NInfer) |
| coder390 | [Merkyor/Qwen3.8-27B-Coder390-…-MTP-DFlash2](https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2) |

Sibling quantizations are published alongside each of them (for coder390: Q2/Q3/Q4/Q6_K/Q8_0
LynnStyle and W4A4 NVFP4 builds).

---

## Quick start

```bash
# 1. Build (see Build above), then package a slim runtime image
docker build -f Dockerfile.run -t ninfer-v100:cu128-kvmem build-sm70/apps

# 2. Start the server (see Run on a 16GB V100 above), then check it
curl -s http://127.0.0.1:8080/v1/models

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift15","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
```

Useful log lines while a request runs:

```
[ring] spill: mapped=464 keep_from=208 candidates=129 taking=6       # demotion to the Host tier
[ring] grow: own pages exhausted (…), reclaimed N from other owners  # room reclaimed from a cached turn
req#1 done | prompt 128,049 | output 256 | TTFT 277.6s | prefill 461.3 tok/s | decode 55.0 tok/s | mtp accepted …
```

---

## Models *(upstream)*

Upstream registered artifacts, for 32GB-class cards:

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Artifacts contain the exact model weights, tokenizer, chat template, and required media frontend resources.

## Performance *(upstream)*

Qwen3.8-27B NVFP4 reaches **218.98 decode tok/s** at K=1, with 99.2% MTP draft acceptance.
That result is on a V100-PCIe-32GB, not SXM. The equivalent SXM2 card is roughly 7% faster; decode is
predominantly HBM-bound, so PCIe bandwidth and host performance have little effect.

The upstream NVFP4 and groupwise-int tables do **not** transfer to this fork's 16GB artifacts, for the
reasons in [Upstream vs this fork](#upstream-vs-this-fork). Use the 16GB tables instead.

## Resource-aware long-context reuse *(upstream; KVMem extends it)*

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

KVMem builds on this: it relaxes the "device pool ≥ logical context" rule and adds relevance-aware
retention plus cross-owner reclaim. See
[KVMem](#kvmem-device-pool-smaller-than-the-context-this-fork).

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Evaluation *(upstream)*

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens for
its measured memory envelope. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

These are upstream results on upstream artifacts. **No capability evaluation has been run on the
16GB-class artifacts in this fork**, and KVMem has no quality measurement at all.

## Startup notes *(upstream)*

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution,
and DFlash2 for Qwen3.8-27B.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile *(upstream)*:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

This fork also ships `Dockerfile.run` (a slim runtime image built from `build-sm70/apps`),
`Dockerfile.sm70`, and `Dockerfile.env`.

## Capabilities and limits

All registered model IDs support *(upstream)*:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to seven;
- DFlash2 masked-block speculative decoding for Qwen3.8-27B (from the upstream integration),
  draft windows from one to fifteen;
- BF16, INT8, and FP8 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

Added by this fork:

- **PQ2_0_G128 ternary weights** and **GGUF mixed-format linear layers**:
  IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q3_K, Q4_0, Q4_K,
  Q5_K, Q6_K, Q8_0 (MMQ and/or vec kernels per format);
- **KVMem**, which with `NINFER_KV_WINDOW` set changes two upstream rules: the device KV pool may be
  smaller than `--max-context`, and `--max-concurrency` must be 1.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one Tesla V100 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [V100 qualification and performance](docs/v100.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [V100-16GB port tuning log](docs/v100-16g-tuning.md) *(this fork)*
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.

The 16GB-class artifacts referenced by this fork are distributed by their own authors — see
[Where the artifacts come from](#where-the-artifacts-come-from).
