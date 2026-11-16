# Experiment design: AsymAPC block eviction on the real vLLM BlockPool

This document is the pre-registered design of the evaluation whose raw data lives under
[`raw/`](raw/) and whose numbers are reported in
[`../asym-apc-real-blockpool-evaluation.md`](../asym-apc-real-blockpool-evaluation.md).

It exists so the dataset can be judged against a stated plan instead of against a story told
after the fact. Sections marked **deviation** record where execution departed from the plan.

---

## 1. Question

Does choosing the KV-cache victim set by contiguous-prefix loss change how much prompt work
must be redone, compared with the order vLLM's own free queue evicts in?

The question is deliberately narrower than "is AsymCache faster?". It asks about one
decision — which cached blocks to drop — with everything else held fixed.

| ID | Hypothesis | Falsified when |
| --- | --- | --- |
| H1 | Under memory pressure with a recoverable working set, prefix-aware victim selection reuses more of each prompt than native FIFO order | ASYM_APC recomputed tokens ≥ Native LRU across the pressured range |
| H2 | The effect is a function of capacity: absent when the cache fits, absent under total thrash, maximal just below the working set | flat curve, or a gain at the extremes |
| H3 | The effect comes from block choice, not from the retention machinery that supplies it | A0 (retention machinery + native ordering) differs from Native LRU |
| H4 | Victim-set divergence is not itself evidence of benefit | divergence counts correlate with token savings everywhere |

## 2. Factors and arms

One factor: **victim-selection policy**, three levels, run on identical inputs.

| Arm | Label in data | Victim order | Retention evidence | Role |
| --- | --- | --- | --- | --- |
| A | `native_lru` | free-queue head, untouched vLLM behaviour | none | **control** |
| B | `a0` | identical to A | Continuum retention state maintained and priced | **placebo / ablation**: isolates the effect of block choice from the effect of having the machinery |
| C | `asym_apc` | by minimum contiguous-prefix loss | same as B | **treatment** |

All three arms run in the same process, on the same BlockPool configuration, over the same
request order, and are priced with the same calibration profile (its SHA-256 is recorded in
every summary). A0 is not a weaker version of C: it is the honest way to show that whatever C
achieves is attributable to selection and not to the bookkeeping C depends on.

**Fixed across arms**

| Quantity | Value |
| --- | --- |
| Block size | 16 tokens (vLLM default) |
| Hash function | `sha256_cbor` via `hash_block_tokens`, `init_none_hash` seeded |
| `PYTHONHASHSEED` | `0` (vLLM derives the chain-hash root from it) |
| Calibration profile | `calibration/cpu-device-prefill-v1.json`, SHA-256 `6266f3fe…c364e78` |
| Capacity | identical for all three arms of a capacity point |
| Seed | `20260101` unless stated |

## 3. Workload

A synthetic multi-program session generator, chosen because it creates the exact structure the
policy reasons about: long shared prefixes, append-only growth, and returns after idle gaps.

| Parameter | Value | Why |
| --- | --- | --- |
| Programs | 8 | enough chains to make the free queue non-trivial |
| Turns per program | 5, append-only | prefix reuse is real, not accidental |
| Shared prefix | 4 blocks | one block chain shared by all programs |
| Program prefix | 12 blocks | per-program stable head |
| Prompt length per turn | 24, 32, 40, 40, 40 blocks (384–640 tokens) | grows into the pressure zone |
| Inter-turn gap | 2 s (short), 90 s (long) | crosses the 30 s retention TTL |
| TTL | 30 s | recency weighting has something to weight |
| Requests | 40 | 8 × 5, issued serially in a fixed order |

Total prompt volume is 22 528 tokens in every arm at every capacity — verified in
`results/arm-metrics-by-capacity.csv`, where `total_prompt_tokens` is constant. Identical input
is a checked property of the dataset, not an assumption.

**Deviation.** The generator's `target_blocks` schedule is a CLI default (`24,32,40,40,40`)
and is not persisted into `summary.json`; the totals confirm it was used, but a reader cannot
recover it from the record alone. Fixed in the harness after this run set, not backfilled.

## 4. Capacity grid

Capacity is the independent variable that moves the system between regimes. Blocks are the
unit; the working set is ≈296 blocks.

| Regime | Capacities | Expected outcome under H2 |
| --- | --- | --- |
| Total thrash | 128, 144 | no effect |
| Pressured, recoverable | 160, 192, 224, 240, 256, 264, 272, 280, 284, 288, 292 | effect grows with capacity |
| Fits | 296, 304, 312, 320 | no effect, because nothing is evicted |

Two sweeps, both with the final code: `sweep2` (128–320, step 32) and `fine` (264–312, the
boundary region at 8-block resolution). `replication` repeats 240 and 288 at three seeds.

**Deviation.** The original plan called for three seeds across the whole grid. The seeds turned
out to be inert — they change token content but not block geometry, so all three produce
bit-identical block traces. The seed dimension was therefore reduced to a determinism check and
the capacity dimension was extended instead. Treating three identical runs as three
replications would have manufactured an `sd = 0` that means nothing.

## 5. Instruments

Nothing about reuse is inferred from policy intent; every reuse number is read back from the
pool after the fact.

| Instrument | What it reads | Failure mode it rules out |
| --- | --- | --- |
| `get_cached_block` walk over the prompt | how many leading blocks the pool still holds | "the policy intended to keep it" |
| Free-queue / ref-count snapshot before and after each decision | whether the chosen victim IDs actually left the pool | policy that decides but does not act |
| `evict_blocks` side effect (`entry.hash` cleared) | physical eviction count | double counting a logical drop as a physical one |
| Per-request record | prompt tokens, cached blocks, recomputed tokens, modelled ms | aggregates hiding request-level regressions |
| Per-decision record | candidate counts, baseline vs selected victim sets, proxy ms, fallback reason | silent fallbacks presented as successes |
| Wall clock per arm | probe runtime | model time presented as measured time |

Recomputed tokens are computed as `prompt_tokens − 16 × cached_prefix_blocks`, both terms read
from the pool. Latency is *modelled*: `APCLatencyCurve` interpolates the CPU-measured prefill
curve (median and P90 per depth, no extrapolation beyond the measured range). Every record
carries `scope.latency_is_model_based = true` and `scope.gpu_executed = false`.

When a decision is issued, the probe reconciles the requested victim IDs against observed
physical evictions and classifies the outcome as `EXACT`, `PARTIAL`, `DISJOINT` or
`NO_PHYSICAL_EVICTION_OBSERVED`. A policy that cannot show its victims reached the pool is
not credited.

## 6. Decision rules of the treatment arm

`ASYM_APC` may only diverge from A0 order when all of the following hold:

1. the candidate set is legal (no pinned, no in-use, no already-free block);
2. each candidate can be attributed to an owning chain with a known depth;
3. a contiguous reusable prefix boundary exists for that chain at decision time;
4. the modelled proxy loss of the selected set is strictly lower than A0's.

Otherwise it records a fallback reason — `content_unverified`, `recency_missing`,
`no_verified_alternative`, `baseline_no_verified_proxy_gain`,
`retention_registry_residency_mismatch` — and reproduces A0 order.

This is why `decision_reasons` is a reported metric: a run where most decisions fall back is a
run about evidence coverage, not about the cost curve.

## 7. Analysis plan

* Primary endpoint: total recomputed prompt tokens per arm per capacity.
* Secondary: mean/median recomputed tokens, prefix hit rate, modelled prefill ms, physical
  evictions, divergence count, selector overhead.
* Comparisons are paired by construction (same requests, same order); the per-request records
  are kept so the pairing can be re-derived, not just trusted.
* A capacity point is reported only with all three arms present.
* No significance test is applied: each capacity is a single deterministic run. The claim is a
  measured difference in a controlled replay, not an estimate of a population effect.

## 8. Threats to validity

| Threat | Status |
| --- | --- |
| Latency is not measured on GPU | **open** — every ms figure is a CPU-curve replay; token counts are hardware independent, ms are not |
| No serving engine, scheduler, batching, concurrency | **open** — requests are serial; real concurrency changes the queue head |
| Calibration model substitution (0.5B, fp32, CPU) | **open** — recorded in the profile provenance |
| Seed replicates are not independent | **documented** — seeds are inert; capacity is the replication axis |
| Single workload shape | **open** — 8 programs, 5 append-only turns, one gap schedule |
| `take_specific` is a probe-side reimplementation of the removal step | **documented** — it reproduces the three effects of `get_new_blocks`; production needs allocator support |
| Selection may be measured under evidence collapse | **measured** — see fallback reasons at 288 vs 256 |

## 9. Run inventory

Every run ever executed for this probe is committed, including the ones that were later
invalidated by a harness fix. `raw/MANIFEST.json` carries, per run, the config, the file list
and a SHA-256 per file.

| Group | Runs | Meaning |
| --- | --- | --- |
| authoritative | `sweep2`, `fine`, `replication` | final code; numbers quoted in the report |
| superseded | `smoke*`, `pc-*`, `sweep-*`, `sweepb-*`, `syncd-*`, `keyfix-*`, `alloc-*`, `real-256-s1` | each predates a fix to key convention, retention sync, allocation routing or evidence coverage; all decisions fell back, so AP C ≡ Native LRU |
| superseded | `grid` | seed grid captured **before** the evidence-coverage fix; `cap160` agrees with the final code, `cap256`/`cap320` do not. Replaced by `replication` |
| superseded | `sweep-real` | full sweep captured before the same fix; AP C ≡ Native LRU at every capacity |

The superseded runs are kept because they are the evidence that the harness was wrong and
then fixed. A dataset that only contains the runs that worked cannot support a claim about
how the numbers were obtained.

## 10. Reproduction

```bash
export PYTHONHASHSEED=0
PROFILE=docs/experiments/asym-apc-native-evaluation-artifacts/calibration/cpu-device-prefill-v1.json
SHA=6266f3fe1a923dd14b0b05811c0978e564f52f566927bd7942486bf29c364e78

python scripts/probes/asym_apc_native/real_blockpool_eval.py \
  --profile "$PROFILE" --profile-sha256 "$SHA" --allow-source cpu_device_measured \
  --num-blocks 288 --programs 8 --turns-per-program 5 --seed 20260101 \
  --out-dir out/cap288
```

One invocation writes `summary.json`, `requests-<arm>.jsonl` and `decisions-<arm>.jsonl` for
all three arms. The 23 arm-triples in [`results/paired-deltas.csv`](results/paired-deltas.csv)
are 23 such invocations (8 in `sweep2`, 9 in `fine`, 6 in `replication`); the CSV tables are
regenerated from the run directories, not hand-maintained.

## 11. What this design deliberately does not claim

* No GPU measured anything. `scope.gpu_executed` is `false` in every summary.
* No serving-level throughput or latency claim; the modelled ms column is a pricing device.
* No claim that the gain survives concurrency, preemption or real scheduling.
* No statistical claim; single deterministic runs per capacity.
