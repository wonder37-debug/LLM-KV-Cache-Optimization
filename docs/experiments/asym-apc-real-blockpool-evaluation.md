# AsymAPC victim selection on the real vLLM BlockPool (CPU device)

**Status: executed on real vLLM cache code, CPU device. Not a GPU experiment and not a serving experiment.**
Block-level victim selection was measured against the native allocator with real prefix
hashes, real reference counts, real cache lookups and real prefix-cache eviction. No GPU
kernel ran, so no GPU latency is reported.

Headline: on real vLLM cache state, cost-aware victim selection cuts recomputed prompt
tokens by 16 %–64 % across the pressured-but-fitting capacity range, and does nothing when
the cache fits or when the workload thrashes beyond recovery. A0 is identical to Native LRU
everywhere, so the entire effect is block selection.

Companion document: [`asym-apc-native-gpu-evaluation.md`](asym-apc-native-gpu-evaluation.md)
covers the serving-engine integration, the GPU calibration harness, and the exact reason the
CUDA path cannot run in this environment.

---

## 1. What problem this document answers

*Does choosing the KV-cache victim set by contiguous-prefix loss reduce re-prefill work
compared with what the native allocator does by itself?*

Answering it needs real cache state, a real cost curve and a paired comparison. The GPU is
unavailable here, but vLLM's cache substrate is pure Python and runs unmodified, which is
enough to test the eviction decision itself.

## 2. Substrate: what is actually real

| Component | Source | Real? |
| --- | --- | --- |
| Block pool, free queue, hashes, ref counts | `vllm.v1.core.block_pool.BlockPool` (vLLM 0.27.1, unmodified) | **yes** |
| Content hashes | `hash_block_tokens` + `sha256_cbor`, seeded via `init_none_hash` exactly as the engine seeds it | **yes** |
| Cache registration / lookup | `cache_full_blocks`, `get_cached_block`, `touch`, `free_blocks` | **yes** |
| Physical eviction | `evict_blocks` → `_maybe_evict_cached_block` (clears the entry hash, observed) | **yes** |
| Victim selection | project `AsymAPCNativeCoordinator` through `RetentionRuntimeIntegration` | **yes** |
| Prefill latency | separate CPU measurement (§3), replayed through `APCLatencyCurve` | measured, **not on GPU** |
| Serving engine, scheduler, sampling | none | **not exercised** |

`PYTHONHASHSEED=0` for every run: vLLM derives the chain-hash root from it, and without it
block hashes differ between processes.

### The allocation rule that decides this experiment

`BlockPool.get_new_blocks` pops the **head** of the free queue and drops whatever cache entry
that block happened to hold. The head is the least-recently-freed block — normally the
program that has been idle longest, i.e. the one about to return. Eviction therefore happens
continuously at allocation time, not at a discrete "pressure event", and a policy that only
reacts once headroom is exhausted is already too late.

This probe therefore routes **every allocation** through the selector via `take_specific`:
remove the chosen block from the free queue, drop its cache entry, raise its ref count — the
same three effects `popleft_n` + `get_new_blocks` produce. Nothing else in the pool is patched.

## 3. Prefill calibration on this device

### 3.1 Model substitution, and why

`Qwen/Qwen2.5-Coder-1.5B-Instruct` in float32 needs 6.2 GB of weights. This container is
capped at **8 GiB** by cgroup and was killed by the OOM killer (`exit=137`) during load.
bfloat16 fits but is emulated on CPU: 15.6 s per 384-token prefill with 100 % spread, i.e.
useless. The CPU quota is **4 cores**, so 32 threads also thrashed.

Calibration therefore used **`Qwen/Qwen2.5-Coder-0.5B-Instruct`, float32, 4 threads** — same
family, same tokenizer lineage. The substitution is recorded in the profile provenance
(`measurement_source: cpu_device_measured`, plus an explicit `device_note`).

### 3.2 Protocol

For each prompt length `m` one deterministic token sequence is drawn; every cached depth `r`
reuses the **same leading `r` tokens**, so depths are compared on one prompt rather than
across different prompts. The prefix forward that materialises the `r` cached tokens runs
**outside** the timed region; only the suffix forward is timed, which is exactly the work APC
removes. Each point: 1 warm-up + 7 repeats, median and P90.

An earlier version of this probe timed the prefix forward together with the suffix and
produced a *non-monotonic* curve (`r=192` slower than `r=0`). That artefact is recorded here
deliberately: a calibration harness can easily manufacture the conclusion it wants.

### 3.3 Measured curve (median ms, block size 16 tokens)

| prompt | 0 | 96 | 128 | 160 | 192 | 256 | 288 | 320 | 384 | 480 | 512 | 640 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 tok | 1219.5 | 886.3 | 780.5 | 698.7 | 616.7 | 441.4 | 358.6 | 262.9 | 0.0 | | | |
| 512 tok | 1610.1 | 1335.4 | 1230.8 | 1138.7 | 1031.8 | 790.6 | 690.8 | 617.5 | 440.5 | 173.1 | 0.0 | |
| 640 tok | 1958.2 | 1672.4 | 1625.5 | 1534.1 | 1401.1 | 1241.0 | 1153.8 | 1031.2 | 808.9 | 538.5 | 457.5 | 0.0 |

Monotonic in cached depth for all three lengths; a fully cached prompt costs 0 ms because no
token needs computing. Total saving 1219–1958 ms.

**Resolution.** Median saving is 45–62 ms per cached block while the noise floor is
52–90 ms, so the smallest reliably detectable difference is **1 block at m=512, 2 at m=384,
3 at m=640**. The measured effects below are far larger than that.

## 4. Experiment design

Eight programs, five turns each, prompts of 24/32/40/40/40 blocks (384/512/640 tokens).
Every turn of a program is a strict prefix of the next turn of the same program — the reuse
pattern APC exists for. Gaps alternate 2 s and 90 s; the Continuum TTL is 30 s, so protection
genuinely expires.

Three arms, identical prompts, identical capacity, identical timing, fresh pool per arm:

* **Native LRU** — no intervention; the pool's own allocation order decides.
* **A0** — `RetentionRuntimeIntegration` with `RetentionAwareLRUAdapter`: same Continuum entry
  table, native block ordering.
* **ASYM_APC** — the same Continuum entry table, project cost-aware selector.

The Continuum entry table is synchronised with observed residency before every decision; a
stale table makes the selector refuse, which is correct behaviour and not the effect measured.

Total demand is 8 × 40 = 320 blocks, so sweeping capacity moves from ~2.5× over-subscription
to exact fit.

## 5. Results

### 5.1 Capacity sweep (identical 40-request sequence per arm)

| capacity | arm | prefix hit | recomputed tokens | modelled prefill | divergent decisions |
| --- | --- | --- | --- | --- | --- |
| 128 | native / A0 | 0.114 | 20 032 | 62 042 ms | — |
| 128 | ASYM_APC | 0.114 | 20 032 | 62 042 ms | 34 / 40 |
| 144 | native / A0 | 0.114 | 20 032 | 62 042 ms | — |
| 144 | ASYM_APC | 0.114 | 20 032 | 62 042 ms | 34 / 40 |
| 160 | native / A0 | 0.114 | 20 032 | 62 042 ms | — |
| 160 | ASYM_APC | 0.131 | 19 680 (−1.8 %) | 60 927 ms | 33 / 40 |
| 192 | native / A0 | 0.177 | 18 752 | 58 096 ms | — |
| 192 | ASYM_APC | 0.194 | 18 400 (−1.9 %) | 56 981 ms | 29 / 40 |
| 224 | native / A0 | 0.239 | 17 472 | 54 150 ms | — |
| 224 | ASYM_APC | 0.267 | 16 752 (−4.1 %) | 52 081 ms | 25 / 40 |
| 256 | native / A0 | 0.309 | 15 680 | 48 909 ms | — |
| 256 | ASYM_APC | 0.349 | 14 656 (−6.5 %) | 45 937 ms | 21 / 40 |
| **288** | native / A0 | 0.379 | 13 888 | 43 668 ms | — |
| **288** | **ASYM_APC** | **0.689** | **5 952 (−57.1 %)** | **19 960 ms** | **17 / 40** |
| 320 | native / A0 | 0.739 | 4 672 | 15 385 ms | — |
| 320 | ASYM_APC | 0.739 | 4 672 (0.0 %) | 15 385 ms | 0 / 24 |

A0 is **identical to Native LRU at every capacity, per request**. The Continuum retention
layer never changes an outcome in this substrate; every difference below is block selection.

### 5.2 The effect is a smooth function of pressure

Fine sweep, same workload, values are ASYM_APC relative to A0:

| capacity | 264 | 272 | 280 | 284 | 288 | 292 | 296 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefix hit A0 → ASYM | .327 → .425 | .344 → .518 | .362 → .606 | .362 → .649 | .379 → .689 | .379 → .729 | .739 → .739 |
| recomputed Δ | **−16.5 %** | **−30.1 %** | **−43.6 %** | **−51.2 %** | **−57.1 %** | **−64.5 %** | 0.0 % |

The gain grows as capacity approaches the working set, then disappears at 296 where the cache
fits and no eviction choice is needed. At 128–144 it also disappears: the workload is so
over-subscribed that every request misses its whole prompt whichever blocks are sacrificed —
34/40 decisions still pick a different victim set, and recomputed tokens do not move at all.
That is the case the task warns about, and it is measured here rather than assumed.

### 5.3 Mechanisms, verified rather than asserted

At capacity 288 (the headline row):

| quantity | Native / A0 | ASYM_APC |
| --- | --- | --- |
| mean cached prefix | 13.5 blocks | 25.9 blocks |
| blocks physically evicted (40 requests) | 581 | 85 |
| requests with different reuse | — | 16 / 40 |
| example request 24 | 4 blocks reused, 576 tokens recomputed | 35 blocks reused, 80 tokens recomputed |

The lower eviction count is a *consequence*, not a separate trick: a request that hits 26 of
40 blocks only needs 14 new ones, so it destroys 14 entries instead of 36. Total allocations
equal total demand in both arms.

The mechanism is the chain-hash structure. Native order takes the head of the free queue,
which is the returned program's own shallow prefix; dropping block 1 of a chain invalidates
the entire chain for reuse, so a returning program recomputes everything. The selector spends
its victims on deep blocks, whose loss truncates a suffix instead of killing a chain.

### 5.4 Where the selector cannot act, and why

| capacity | fallback reasons (40 decisions) |
| --- | --- |
| 288 | 17 `selected_lower_proxy_loss`, 23 `baseline_no_verified_proxy_gain` |
| 256 | 21 `selected_lower_proxy_loss`, 19 `baseline_no_verified_proxy_gain` |
| 320 | 24 `baseline_no_verified_proxy_gain` (no decision needed for 16 requests) |

The selector needs, per candidate, the owning chain and its depth. Blocks are recycled across
programs, so at high pressure a chain's contiguous reusable boundary can collapse before the
decision — an instrumented case showed `reusable_prefix_blocks = 4` against a 40-block
snapshot. With the boundary gone the block cannot be attributed to a chain, `content_unverified`
is recorded, and the arm correctly falls back to A0 order. Evidence coverage, not the cost
curve, is what limits the selector in the extreme-pressure regime.

## 6. Interpretation

1. **The premise holds on real vLLM cache state**: prefix-aware victim selection beats the
   native allocator by up to **−64.5 % recomputed tokens** when the cache is pressured but
   large enough to hold the working set, with a smooth dose–response curve in capacity.
2. **The effect is fragile at both ends**: total thrash (≤144) and a fitting cache (≥296) both
   produce zero difference, and the transition is steep.
3. **Divergence alone proves nothing**: at 128–144 the victim set changes on 34 of 40
   decisions with zero token savings.
4. **A0 adds nothing here**: Continuum retention + native ordering was byte-identical to
   Native LRU at every capacity. Any future comparison must therefore isolate block selection,
   as this probe does.
5. **The proxy is conservative**: mean predicted gain per divergent decision is 15.9 ms while
   realised saving at capacity 288 is ~590 ms per improved request. The selector does not
   model the cascade it prevents, so a low predicted gain is not evidence of low value.

## 7. Limitations

* **No GPU.** Every latency figure is a model replay of a CPU-measured curve. GPU per-block
  saving will be far smaller in absolute terms; the *token* reductions are hardware
  independent, the millisecond figures are not.
* **No serving engine.** Requests are issued serially; there is no scheduler, batching,
  preemption or real concurrency. Concurrency changes what sits at the queue head and could
  change both evidence coverage and the gain.
* **Determinism, not statistics.** Seeds change tokens but not block geometry, so repeated
  seeds are bit-identical; `sd = 0` here is *not* evidence of stability. The capacity sweep is
  the only genuine replication axis, and each capacity is a single deterministic run.
* **Model substitution** (0.5B instead of 1.5B) and float32 on CPU.
* `take_specific` reproduces what `get_new_blocks` does to the blocks it removes; it is not a
  patch of the allocator. A production policy needs the allocator to expose that choice.
* Single workload shape: 8 programs, 5 append-only turns, 384–640 token prompts.

## 8. Feasibility conclusion

*Is the AsymCache-inspired contiguous-prefix victim selector useful on real vLLM?*

**Yes for the eviction decision itself, on real vLLM cache state, with a measured and
reproducible effect** — up to −64.5 % recomputed prompt tokens and −54 % modelled prefill in
the pressured-but-fitting regime, and zero when the cache fits or thrashes.

**Not yet a system result.** The latency half of the claim, and the interaction with a real
scheduler under concurrency, remain unmeasured because this host has no NVIDIA device. The
GPU experiment described in the companion document is still outstanding, and nothing here
should be read as a serving-level speedup.

## 9. Reproduction

```bash
# calibration (CPU device, 4 threads; ~4.5 min)
python scripts/probes/asym_apc_native/calibrate_cpu_device.py \
  --model-dir /path/to/Qwen2.5-Coder-0.5B-Instruct \
  --model-name Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --lengths 384,512,640 --depths 0,96,128,160,192,256,288,320,384,480,512,640 \
  --repeats 7 --threads 4 --dtype float32 --out artifacts/cpu-device-prefill-v1.json

# A/B/C evaluation on the real BlockPool (one capacity)
PYTHONHASHSEED=0 python scripts/probes/asym_apc_native/real_blockpool_eval.py \
  --profile artifacts/cpu-device-prefill-v1.json \
  --profile-sha256 6266f3fe1a923dd14b0b05811c0978e564f52f566927bd7942486bf29c364e78 \
  --allow-source cpu_device_measured \
  --num-blocks 288 --programs 8 --turns-per-program 5 --out-dir out/cap288
```

## 10. Artifacts

| Path | Content |
| --- | --- |
| [`.../calibration/cpu-device-prefill-v1.json`](asym-apc-native-evaluation-artifacts/calibration/cpu-device-prefill-v1.json) | 32 measured points, provenance, quality report |
| [`.../calibration/calibration-run.log`](asym-apc-native-evaluation-artifacts/calibration/calibration-run.log) | raw calibration stdout |
| [`.../vllm-blockpool-eval/cap*/summary.json`](asym-apc-native-evaluation-artifacts/vllm-blockpool-eval) | A/B/C summaries for 15 capacities |
| [`.../vllm-blockpool-eval/cap288-detail`](asym-apc-native-evaluation-artifacts/vllm-blockpool-eval/cap288-detail) | per-request and per-decision records at the headline capacity |
| [`.../vllm-blockpool-eval/environment.txt`](asym-apc-native-evaluation-artifacts/vllm-blockpool-eval/environment.txt) | cgroup limits, CPU quota, absence of GPU |
