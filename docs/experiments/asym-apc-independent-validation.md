# Independent validation of AsymAPC victim selection (real vLLM BlockPool, CPU)

**Status: executed. Real vLLM 0.27.1 `BlockPool` on a CPU device. No GPU, no serving engine,
no measured latency.**

**Branch base:** `experiment/asym-apc-native-gpu-arm` at `a3f14e2` (the corrected replay plus
its validity fixes). This report adds independent workloads, a four-arm ablation, a held-out
seed set, real decision-overhead measurement and a workload-generality verdict.

**What changed relative to the corrected replay:** the previous work established *that* the
selector can change physical cache state on one workload. This work asks *how much of that
survives* when the workload is varied, when the algorithm is ablated, and when the CPU cost of
deciding is charged against the modelled saving. Two of those three answers are negative.

---

## 1. Source commit and execution environment

| Item | Value |
| --- | --- |
| Repository | `github.com/wonder37-debug/LLM-KV-Cache-Optimization` |
| Base commit | `a3f14e2` (`experiment/asym-apc-native-gpu-arm`) |
| vLLM | 0.27.1 (`vllm.v1.core.block_pool.BlockPool`, unmodified) |
| PyTorch | 2.10.0+cu128, CUDA **not** available (`torch.cuda.is_available() == False`) |
| Python | 3.11.1 |
| Host | Linux container, 4 CPU cores, 8 GiB cgroup limit, no `/dev/nvidia*`, no `nvidia-smi` |
| Hash function | `sha256_cbor` via `hash_block_tokens`, `init_none_hash` seeded |
| `PYTHONHASHSEED` | `0` for every run |
| Calibration profile | `asym-apc-native-evaluation-artifacts/calibration/cpu-device-prefill-v1.json`, SHA-256 `6266f3fe…c364e78` |

**Deviation, stated up front.** The task asked for the pinned `vllm-0.27.1+cpu` wheel. Fetching
that wheel (and the matching CPU PyTorch) into a fresh environment was attempted and abandoned:
11 minutes produced 22 MiB of a multi-hundred-MB download. The measurements below therefore run
against the same **vLLM 0.27.1** release in an existing environment whose PyTorch build carries
CUDA libraries while reporting no device. This matters less than it sounds — `BlockPool` is
pure Python, no kernel is launched, and the reproduction in §4 matches the CI replay (which used
the CPU wheel) on every one of 12 capacity points. It is still a deviation from the requested
environment and is not hidden.

Repository tests, before any experiment:

```
python -m pytest -q tests --disable-warnings --tb=line \
  --deselect=...test_materialize_campaign_freezes_54_runs_and_cross_arm_time \
  --deselect=...test_validation_rejects_tampered_logical_sequence
→ 2043 passed, 19 skipped, 2 failed, 2 deselected
```

Both failures are artifacts of my working environment, not of the branch: one required the
package to be installed (`pip install -e .`, done — it now passes), the other asserts a clean
git tree (`test_phase2e_runner.py`, resolved by the commit that carries this report). No test
related to the selector failed.

## 2. Design and controlled variables

Four arms, one factor each. Everything that is not the factor is held identical.

| Arm | Retention Entry lifecycle | Cost model | Recency weighting |
| --- | --- | --- | --- |
| `a0` | maintained, identical to treatment | — (native order) | — |
| `depth_only` | maintained | contiguous-prefix cascade | **uniform** (no time) |
| `recency_only` | maintained | independent block marginals | two-timescale decay |
| `asym_apc` | maintained | contiguous-prefix cascade | two-timescale decay |

`depth_only` is the ablation that matters: same evidence, same candidate legality, same tier
plan, same search budget — only the per-chain weight is removed. `recency_only` is the mirror
image: the weights stay, the position-sensitive cascade is replaced by independent marginals.

Controlled across arms inside every cell: capacity, request order, prompt bytes, inter-turn
gaps, TTL (30 s), calibration profile and SHA-256, hash seed, search limits
(`max_exact_subsets=12`, `max_greedy_operations=200000`, `min_proxy_gain_ms=0`), and the
Continuum Entry Release plan. A run is only comparable to another run with the same
`geometry_signature` and the same profile SHA; both are recorded per cell.

**Ablation fairness fix.** The uniform-weight arm would previously have inherited the recency
arm's fallback rate, because the coordinator required a pre-decision access timestamp for every
candidate chain. It now requires time only when the scoring function consumes time. Without that
change the ablation would have measured evidence coverage, not scoring.

## 3. Workloads and seed behaviour

Eight deterministic families. Each is a function of `(family, seed)`; the seed drives arrival
order, gaps, lengths and revisits as well as token values.

| Family | Shape | Working set (blocks) |
| --- | --- | --- |
| `legacy_round_robin` | 8 programs × 5 append-only turns, round-robin | 292 |
| `agent_random_arrival` | same shape, arrival order permuted per seed | 292 |
| `program_turn_scaling` | 10 programs, 1–6 turns each, lengths 24–72 | 500 |
| `tool_gap_mix` | gaps sampled from {1, 5, 20, 45, 90, 300} s | 292 |
| `shared_prefix_ratio` | 16-block shared head instead of 4 | 208 |
| `mixed_lengths` | per-turn length drawn from 16–80 blocks | 516 |
| `zipf_revisit` | 10 programs, Zipf-skewed turn counts (6…1) | 332 |
| `unpredictable_revisit` | 35 % of turns re-read an earlier depth | 292 |

Capacity is expressed as a **fraction of the family's working set** (0.85 and 0.95) instead of
an absolute block count. A single absolute capacity puts different shapes in different regimes —
the first attempt at 256/288 blocks produced zero-pressure cells for five of eight families,
which looks like "no effect" but only means "no eviction".

**Seed independence is a property, not a claim.** `geometry_signature` hashes
(program, prompt length, gap, revisit) per arrival. For the order-sensitive families, three
seeds produce three distinct signatures (tested: `tests/test_asym_apc_workload_families.py`).
For `legacy_round_robin` the signature is seed-invariant — which is exactly why the original
single-workload seed axis could not be a replication, and why that family is used only as the
bridge to the earlier replay.

**Exploratory vs held-out.** Parameter choices (capacity fractions, decay constants) were made
on seed `20260101`. Every claim in §5–§6 is evaluated on seeds `20260301`, `20260302`,
`20260303`, which were never used for tuning.

## 4. Reproduction of the corrected replay

Before adding anything, the existing 12-capacity result was re-executed with the new driver.
Recomputed prompt tokens, identical arms, identical capacity:

| Capacity | A0 | ASYM_APC | Δ naïf | Corrected CI replay | Match |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 20 032 | 20 032 | 0.0 % | 20 032 / 20 032 | yes |
| 160 | 20 032 | 19 680 | −1.8 % | 20 032 / 19 680 | yes |
| 224 | 17 472 | 16 752 | −4.1 % | 17 472 / 16 752 | yes |
| 240 | 16 576 | 15 552 | −6.2 % | 16 576 / 15 552 | yes |
| 256 | 15 680 | 14 144 | −9.8 % | 15 680 / 14 144 | yes |
| 264 | 15 232 | 12 720 | −16.5 % | 15 232 / 12 720 | yes |
| 272 | 14 784 | 10 336 | −30.1 % | 14 784 / 10 336 | yes |
| 280 | 14 336 | 8 080 | −43.6 % | 14 336 / 8 080 | yes |
| 288 | 13 888 | 5 952 | −57.1 % | 13 888 / 5 952 | yes |
| 292 | 13 888 | 4 928 | −64.5 % | 13 888 / 4 928 | yes |
| 296 | 4 672 | 4 672 | 0.0 % | 4 672 / 4 672 | yes |
| 320 | 4 672 | 4 672 | 0.0 % | 4 672 / 4 672 | yes |

Retention parity at capacity 288, both controlled arms: **39/40** pressure events with
Retention Entries, **23/40** with protected Entries. `native_lru` records 0 because it never
enters the controlled path. This is the defect the earlier fix addressed, and it stays fixed.

The three zero rows are not evidence of failure: at 128 the workload thrashes (every request
misses its whole prompt regardless of victims), at 296/320 the cache fits and nothing is
evicted.

## 5. Ablation: which component produces the gain

Reference workflow, seed `20260101`, four arms, identical inputs:

| Capacity | A0 | depth_only | recency_only | combined | depth vs A0 | combined vs A0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 17 472 | 16 752 | 17 248 | 16 752 | −4.1 % | −4.1 % |
| 240 | 16 576 | 12 480 | 16 352 | 15 552 | **−24.7 %** | −6.2 % |
| 256 | 15 680 | 10 816 | 15 456 | 14 144 | **−31.0 %** | −9.8 % |
| 264 | 15 232 | 10 848 | 15 008 | 12 720 | **−28.8 %** | −16.5 % |
| 272 | 14 784 | 10 128 | 14 560 | 10 336 | −31.5 % | −30.1 % |
| 280 | 14 336 | 8 080 | 14 112 | 8 080 | −43.6 % | −43.6 % |
| 288 | 13 888 | 5 952 | 13 856 | 5 952 | −57.1 % | −57.1 % |
| 292 | 13 888 | 4 928 | 13 856 | 4 928 | −64.5 % | −64.5 % |

Reading:

* **The position-sensitive cost model is the whole mechanism.** Removing the recency term does
  not weaken the result; it improves it at 240–264 and ties everywhere else.
* **The recency term alone is close to useless** (−1.4 % to −1.7 %), and at 288–292 it changes
  nothing at all while still iterating over the same candidates.
* **Both arms see the same evidence.** `depth_only` and `asym_apc` record identical divergence
  counts (23/21/19/17 at 240/256/272/288) and `evidence_missing_total = 0`. The difference
  between them is the weight, not coverage.

### 5.1 Why the recency weight hurts here

With the incumbent constants (`α=60 s, β=12 s, knee=60 s`) a chain idle for 90 s weighs
`min(e^-1.5, e^-2.5) ≈ 0.08`. But the reference workload's long inter-turn gap *is* 90 s, and a
chain that has been idle for one long gap is exactly the chain that returns next. The decay
therefore discounts the most likely return. Fitting the constants to the gap distribution
recovers most of the gap but never beats dropping the term:

| Capacity | A0 | depth_only | combined 60/12/60 | combined 240/60/120 | combined 600/180/300 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 240 | 16 576 | 12 480 | 15 552 | 14 272 | 14 128 |
| 256 | 15 680 | 10 816 | 14 144 | 12 608 | 12 224 |
| 272 | 14 784 | 10 128 | 10 336 | 10 240 | 10 240 |
| 288 | 13 888 | 5 952 | 5 952 | 5 952 | 5 952 |

Slower decay halves the damage (240: −6.2 % → −14.8 % vs A0) and still loses to no decay at all
(−24.7 %). This is a negative result about the recency term as a *ranking* signal under
periodic revisits, not about recency in general.

## 6. Capacity and workload generality (held-out seeds)

Held-out seeds `20260301–03`, two capacity fractions per family, four arms, 48 cells.
Mean relative change in recomputed tokens against the controlled A0:

| Family | depth_only | combined | recency_only | Cells better/equal/worse (combined) |
| --- | ---: | ---: | ---: | --- |
| `legacy_round_robin` | **−33.2 %** | −23.1 % | −1.5 % | 6 / 0 / 0 |
| `tool_gap_mix` | **−38.6 %** | −22.5 % | −1.5 % | 6 / 0 / 0 |
| `shared_prefix_ratio` | **−24.4 %** | −20.4 % | −0.3 % | 6 / 0 / 0 |
| `agent_random_arrival` | −4.7 % | −3.5 % | +1.5 % | 3 / 3 / 0 |
| `unpredictable_revisit` | −3.8 % | −3.2 % | −0.8 % | 2 / 4 / 0 |
| `zipf_revisit` | −2.7 % | −1.6 % | −0.4 % | 2 / 4 / 0 |
| `program_turn_scaling` | −0.9 % | −0.8 % | −0.4 % | 1 / 5 / 0 |
| `mixed_lengths` | −0.3 % | −0.3 % | 0.0 % | 1 / 5 / 0 |

The ordering is stable across seeds and matches the exploratory run: the combined arm is worse
than or equal to the depth-only arm in 47 of 48 cells. The single exception is
`agent_random_arrival` at 277 blocks, seed `20260301`, where the combined arm is 128 tokens
better — small enough to be a tie-break effect, and not reproduced at the same capacity with the
other two seeds. `recency_only` is worse than A0 in one cell (agent arrival, +1.5 %).

Re-running the tuned decay (`600/180/300`) on the held-out seeds confirms the same ordering:
`legacy_round_robin` −28.8 % vs depth-only −33.2 %; `tool_gap_mix` −22.8 % vs −38.6 %.

**Why five families show almost nothing** is in §8: not a scoring failure but missing evidence.
Only the three families with complete, in-coverage evidence produce large effects.

## 7. Selector overhead and scalability

Measured with `time.perf_counter_ns()` around the `select_victims` call only, on an otherwise
idle host, two identical passes (token counts bit-identical across passes; p50 differs 13 %).

| Capacity | Decisions | p50 | p90 | p95 | max | Total CPU | Modelled prefill saved | Saving / overhead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 40 | 7.02 ms | 22.7 ms | 23.0 ms | 25.1 ms | 461 ms | 0 ms | 0× |
| 192 | 40 | 7.28 ms | 44.0 ms | 44.3 ms | 45.7 ms | 696 ms | 1 115 ms | 1.6× |
| 256 | 40 | 9.63 ms | 66.9 ms | 67.1 ms | 67.6 ms | 841 ms | 4 416 ms | 5.3× |
| 288 | 40 | 5.48 ms | 11.3 ms | 11.5 ms | 11.6 ms | 241 ms | 23 708 ms | 98.6× |

Scaling, pooled over 1 450 real decisions (held-out campaign):

| Dimension | Spearman ρ vs overhead |
| --- | ---: |
| Legal candidate count | 0.33 |
| Required victim count | 0.33 |
| Observed chain count | 0.015 |

* Overhead grows with the **product** of candidates and victims, which is the greedy search's
  arithmetic: `take × |members|` loss evaluations, each rebuilding a prefix-length map over all
  involved chains and scanning the curve linearly. At 256 blocks that is ≈24 × 120 evaluations,
  i.e. ~2.5 µs per evaluation.
* **Exact enumeration never runs.** `C(n, k) ≤ 12` is unreachable at realistic candidate counts
  (120 candidates, 24 victims → 10^25 subsets), so every decision is greedy
  (`exact_search_decisions = 0` in every cell). Exact-vs-greedy is therefore not a trade-off
  this implementation can make; it is only available for tiny victim counts.
* **Chain count is irrelevant** (ρ ≈ 0.015): per-chain work is O(1) after the curve lookup, so
  the cost is driven by candidate enumeration, not by the number of sessions.
* **The overhead is not free.** At 128 blocks it is pure cost (461 ms, zero saving). At 192 the
  modelled saving is only 1.6× the overhead. On a GPU, prefill is 10–100× faster than this CPU
  curve, so the 256-block margin (5.3×) becomes roughly 0.05–0.5× — i.e. **the selector as
  currently implemented would likely cost more than it saves in a real serving loop.** That is
  an engineering defect, not a property of the policy: the same selection can be computed
  incrementally, but this implementation does not.

## 8. Fallback and evidence coverage

Held-out, `asym_apc` arm. "Divergence" is the share of decisions whose victim set differs from
A0's order.

| Family | Decisions | Divergence | Dominant reasons | Missing evidence (blocks) |
| --- | ---: | ---: | --- | --- |
| `legacy_round_robin` | 240 | 50.0 % | 120 `selected_lower_proxy_loss`, 120 `baseline_no_verified_proxy_gain` | 0 |
| `tool_gap_mix` | 240 | 50.0 % | same split | 0 |
| `shared_prefix_ratio` | 240 | 46.2 % | 111 selected, 126 baseline, 3 `no_choice` | 0 |
| `zipf_revisit` | 156 | 16.7 % | 26 selected, 130 baseline | 0 |
| `agent_random_arrival` | 100 | 12.0 % | 35 `content_unverified`, 53 baseline, 7 partial, 5 selected | `content_unverified` 280 |
| `unpredictable_revisit` | 103 | 4.9 % | 28 `content_unverified` | `content_unverified` 480, `profile_missing` 332 |
| `program_turn_scaling` | 282 | 3.2 % | 240 baseline, 25 `content_unverified`, 7 `profile_missing` | `profile_missing` 5 776 |
| `mixed_lengths` | 89 | 1.1 % | 33 `content_unverified`, 55 baseline | `profile_missing` 1 628, `content_unverified` 661 |

Three distinct constraints, all visible in the data:

1. **Calibration coverage.** The profile covers 24/32/40-block prompts only. Any resident chain
   longer than 40 blocks cannot be priced (`profile_missing`), so workloads with 48–80-block
   prompts fall back almost completely — `program_turn_scaling` alone loses 5 776 candidate
   blocks to this. This is a deployment cost of the policy: it can only act on prompt lengths
   that were measured.
2. **Evidence collapse.** With shuffled arrival or non-append-only re-reads, a chain's
   contiguous reusable boundary has already been cut by other programs' allocations, so the
   candidate cannot be attributed to a chain (`content_unverified`). Divergence falls to 1–12 %
   and the gain with it.
3. **No pressure.** At ≥0.95 of the working set some families never evict (`no_choice`), which
   yields an honest 0 % rather than a manufactured win.

Every fallback reproduces A0's exact victim order. There is no case where missing evidence was
priced as zero and still produced a "win" — the earlier review's concern is enforced by
`baseline_unverified` and `insufficient_verified_candidates`.

## 9. CPU BlockPool versus GPU serving

| Question | Answer in this work |
| --- | --- |
| Did a CUDA kernel run? | No. `torch.cuda.is_available() == False`, no `/dev/nvidia*`, no `nvidia-smi`. |
| Is any number a GPU measurement? | No. `gpu_executed: false` in every cell; all milliseconds are a replay of a CPU-measured curve. |
| Was the gated Native runner exercised? | No. No scheduler, batching, preemption, TTFT or E2E latency. |
| Is the token accounting hardware-independent? | Yes — it is pool state read back through `get_cached_block`, and would not change on a GPU for the same geometry. |
| Would the same decisions be made on a GPU? | Unknown. The decisions depend on the cost curve; a GPU curve has a different shape (cheaper per token, different fixed costs), so rankings could change. |

Validation gates, checked rather than asserted:

| Gate | Result |
| --- | --- |
| Equal input and Retention semantics across arms | **Pass** — same turn list, same Entry lifecycle, parity counts in §4 |
| Legal victims and correct physical transition | **Pass** — `take_specific` mirrors `popleft_n` + `get_new_blocks`; evictions reconciled per decision |
| Different victim *sets*, not just order | **Pass** — `victim_set_differs` computed as set inequality, 50 % divergence in the strong families |
| Measurable change in future reuse | **Pass** for 3 families, **fail** for 5 (documented, §8) |
| No future information | **Pass** — `future_observation` rejects post-decision timestamps; tested |
| No zero-cost assignment to unknown content | **Pass** — unverified candidates force `baseline_unverified` |
| Complete accounting of invalid evidence | **Pass** — `evidence_missing_reason_counts` per decision |
| No silent substitution of CPU/synthetic latency | **Pass** — scope flags in every record; this report repeats it |
| Reproducible across independent workload groups | **Partial** — deterministic and seed-stable, but the effect size is workload-dependent by a factor of 100 |

## 10. Feasibility judgement

*On the real vLLM cache substrate, a contiguous-prefix cost model is a real, reproducible
mechanism: it cuts recomputed prompt tokens by 23–39 % on prefix-friendly agent traces with
complete evidence, and by ≤5 % when evidence is incomplete.*

*The two-timescale recency term does not contribute. Position-sensitive cost alone matches or
beats the combination in all 48 held-out cells.*

*It is not ready to deploy. The policy is gated by calibration coverage (prompt lengths must be
measured), by evidence completeness (append-only chains), and by a decision cost of 5–10 ms that
the modelled saving does not always repay, and would repay far less on a GPU.*

### Limitations

* No GPU, no serving engine, no TTFT: no performance claim exists.
* One synthetic generator family tree; real agent traces may break the append-only assumption
  even more than `unpredictable_revisit` does.
* Capacity is a fraction of a computed working set, not a measured model footprint.
* Each cell is a single deterministic run per (family, capacity, seed); spread comes from seeds
  and capacities, not from repetition of the same cell.
* Calibration uses `Qwen2.5-Coder-0.5B` in float32 on CPU; the substitution is recorded in the
  profile provenance.

### Reproduction

```bash
export PYTHONHASHSEED=0
PROFILE=docs/experiments/asym-apc-native-evaluation-artifacts/calibration/cpu-device-prefill-v1.json
SHA=6266f3fe1a923dd14b0b05811c0978e564f52f566927bd7942486bf29c364e78

# reference reproduction (12 capacities, 3 arms)
python scripts/probes/asym_apc_native/run_campaign.py \
  --profile "$PROFILE" --profile-sha256 "$SHA" --allow-source cpu_device_measured \
  --families legacy_round_robin \
  --capacities 128,160,224,240,256,264,272,280,288,292,296,320 \
  --seeds 20260101 --arms native_lru,a0,asym_apc \
  --out-dir out/repro

# four-arm ablation
python scripts/probes/asym_apc_native/run_campaign.py \
  --profile "$PROFILE" --profile-sha256 "$SHA" --allow-source cpu_device_measured \
  --families legacy_round_robin \
  --capacities 224,240,256,264,272,280,288,292 --seeds 20260101 \
  --arms a0,depth_only,recency_only,asym_apc --out-dir out/ablation

# held-out generality (capacity as a fraction of each family's working set)
python scripts/probes/asym_apc_native/run_campaign.py \
  --profile "$PROFILE" --profile-sha256 "$SHA" --allow-source cpu_device_measured \
  --families legacy_round_robin,agent_random_arrival,program_turn_scaling,tool_gap_mix,shared_prefix_ratio,mixed_lengths,zipf_revisit,unpredictable_revisit \
  --capacity-fractions 0.85,0.95 --seeds 20260301,20260302,20260303 \
  --arms a0,depth_only,recency_only,asym_apc --out-dir out/heldout

# analysis tables
python scripts/probes/asym_apc_native/analyze_campaign.py \
  --campaign out/heldout --out-dir out/analysis
```

## 11. Negative and inconclusive results

1. **The recency term is not additive** and is harmful at 240–264 blocks with the incumbent
   constants (−6.2 % combined vs −24.7 % depth-only). Tuning the constants narrows but never
   closes the gap.
2. **`recency_only` is worse than A0 in one held-out cell** (+1.5 %, agent arrival, 0.85 of
   working set): a weighting without position sensitivity can reorder victims badly.
3. **Five of eight workload families show ≤5 % improvement**, and in three of them the selector
   barely diverges at all (1–12 % of decisions).
4. **Widening the calibration profile failed and was abandoned.** A profile covering 16–80
   blocks at 128-token depth resolution was measured (`calibration/cpu-device-prefill-wide-v1.json`),
   but every length was flagged `resolvable: false` — the measured noise floor (127–286 ms)
   exceeded the per-block saving (48–72 ms), because the run overlapped another campaign. It is
   committed as evidence of the attempt and **is not used for any number in this report**.
   Extending coverage requires a quiet host and 32-token depth steps, which is hours of CPU
   prefill, not minutes.
5. **The overhead result is inconclusive at 192 blocks** (1.6× margin) and negative at 128
   (zero saving, 461 ms cost).

## 12. Direct answers

**1. Is the combined AsymAPC policy better than a correctly controlled A0?**
Yes, on this substrate: −23.1 % and −22.5 % mean recomputed tokens on the two prefix-friendly
families, −20.4 % on shared-prefix-heavy traces, and 0 % where evidence is missing. It never
increases recomputation in any of the 48 held-out cells except via the `recency_only` arm, which
is a different arm. The effect is causal at the mechanism level: A0 and all treatment arms
maintain identical Retention Entry lifecycles (39/40 events with entries, 23/40 protected), and
the victims are reconciled against observed physical eviction.

**2. Does position-sensitive cost awareness alone explain the gains?**
Yes, with one tie-break exception. `depth_only` — contiguous-prefix cascade with uniform
weights — matches or beats the combined arm in 8/8 ablation capacities and 47/48 held-out cells
(the exception being one agent-arrival cell, 128 tokens in the combined arm's favour), reaching
−33.2 % / −38.6 % where the combined arm reaches −23.1 % / −22.5 %. Deleting the recency term
costs nothing measurable and gains 10 points at moderate capacity.

**3. Does the two-timescale recency term add measurable value?**
No. With the incumbent constants it *reduces* the benefit (−6.2 % vs −24.7 % at 240 blocks;
−9.8 % vs −31.0 % at 256). With constants fitted to the workload's 90 s gap distribution it
recovers to −14.8 % / −22.0 % and still loses to no decay at all, and on held-out seeds it
reaches −28.8 % against depth-only's −33.2 %. Its only measurable effect is dilution.

**4. In what regimes does the policy lose or make no difference?**
Four, all measured: (a) total thrash — at 128 blocks the victim set changes on 34/40 decisions
and saves exactly zero; (b) a fitting cache — at ≥296 blocks nothing is evicted and the policy
is a no-op with real CPU cost; (c) incomplete evidence — shuffled arrival or re-read patterns
push `content_unverified` to 280–480 blocks and divergence to 1–12 %; (d) out-of-profile prompt
lengths — `profile_missing` removes 5 776 candidate blocks in `program_turn_scaling`, leaving a
−0.9 % result that reflects unmeasured lengths, not a bad decision rule.

**5. Does the measured CPU decision overhead offset the savings?**
At the extremes, yes; in the middle, only barely. Per-run CPU cost is 241–841 ms for 40
decisions (p50 5.5–9.6 ms, p95 up to 67 ms). The modelled prefill saving is 0× at 128 blocks,
1.6× at 192, 5.3× at 256 and 98.6× at 288. Because those milliseconds come from a CPU prefill
curve and GPU prefill is 10–100× faster, the 5.3× margin at 256 blocks would plausibly invert on
a GPU. The overhead is removable — the greedy loop repeats an O(candidates × chains) loss
evaluation it could update incrementally — but it has not been removed here, so the honest
answer is: **the current implementation is not overhead-safe for serving.**

**6. What evidence supports or prevents a Native GPU performance claim?**
Nothing supports it. No CUDA device was present (`torch.cuda.is_available() == False`, no
`/dev/nvidia*`, no `nvidia-smi`), no serving engine ran, no TTFT or E2E latency was measured,
and every cell carries `gpu_executed: false` and `latency_is_model_based: true`. What would be
needed: the gated Native runner on real hardware with request-time cached-token evidence,
verified model revision, genuine physical-eviction reconciliation, GPU-measured prefill
calibration at the actual block lengths, and a paired campaign under concurrency. The token
reductions here are a necessary condition for a latency win, not evidence of one — and given
§7, the decision cost could erase it.
