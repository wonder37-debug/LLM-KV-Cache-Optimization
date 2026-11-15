# AsymCache-Inspired Native APC Block Selection — Experimental Fork Prototype

**Status:** standalone **experimental proposal and CPU demonstration** on the personal fork; not an upstream-authorized Phase 3/4/5 arm and not a verified performance improvement. Do **not** change frozen A0, A1, A2, A3, E0 or Protocol v4 artifacts to accommodate this work. Based on fork `main` at `ced1d1ce1d9135c1d013d34acd9850648a2e1d42` (2026-11-15).

## Motivation and distinction from AsymCache

AsymCache combines a recency-dependent reuse-value proxy with position-aware GPU recomputation cost and supports *non-contiguous* KV reuse through Multi-Segment Attention. This prototype **does not** implement MSA, source-level AsymCache, or its attention kernels. It retains native vLLM 0.27.1 **contiguous-prefix** APC semantics. An evicted depth `d` truncates the reusable prefix to `d-1` even if deeper blocks remain resident. Therefore, AsymCache's MSA-specific per-block `Delta T` is **not** reused.

## Exact contract

- Entry Release/TTL/soft protection and eligibility tiers are entirely owned by the existing Continuum coordinator (`RetentionAwareSelectionCoordinator`).
- The new `AsymAPCNativeCoordinator` is accepted as `block_selection_coordinator` by `RetentionRuntimeIntegration`; it has exactly the required `select_victims(candidates, preparation, retention_entries, required_blocks, timestamp)` shape.
- The coordinator **only reorders within the same legal tier** (TIER_1 before TIER_2). It cannot add a candidate, delete a protected block, change refcounts, mutate the vLLM free queue or bypass native bookkeeping.
- `A2NativeSnapshotRegistry.build_request()` supplies provable current content hash, per-chain depths, and effective contiguous reusable prefix. The registry's `snapshot(chain)` supplies original chain length. It is critical not to confuse a free-queue `lru_rank` with chain depth.
- `observe_chain_access(chain_id, timestamp)` must be called from an **actual observed request/prefix event before** the eviction decision. Historical future request outcomes are forbidden as policy features. **The existing formal runner has not been modified to wire this new event observation**; actual Native execution requires an additional *explicitly authorized* opt-in runner path and profiles.
- `APCLatencyCurve` holds monotone measured *or clearly labelled synthetic* prefill milliseconds for a fixed total chain length `m` versus the *cached contiguous prefix depth* `r`. It requires the full `[0,m]` range and refuses extrapolation. Actual GPU profile data for this full input surface has **not** been collected.

For a physical-victim set `S`, current reusable prefix `r_c` and original chain length `m_c`:

```
r_after(c,S) = min(r_c, min_{b in S cap c} (depth_c(b) - 1))
L(S) = sum_c w_c(t) * [ F(m_c, r_after(c,S)) - F(m_c, r_c) ]
w_c(t) = min(exp(-age_c / alpha), exp(-(age_c-knee) / beta))
```

`F` is piecewise-linear interpolation **within an externally supplied profile curve**, not a claim about a particular GPU. `w` is a ranking proxy based on elapsed time from a real observation; it is not a measured P(reuse). Shared physical victims reduce each affected chain's prefix, counted once per chain. Hashless blocks cost zero. The optimizer chooses an exact low-cost within-tier subset when combinations are bounded, otherwise a bounded greedy approximation. It declines to override A0 when no strictly lower *estimated* proxy cost is found.

Unknown content, hash mismatch, missing depth, missing timestamps, missing profile, nonzero refcount, tier mismatch or search budget exhaustion **fail closed to A0's exact `virtual_eligible_order`**. A single candidate or a fully consumed legal set has no policy leverage.

## CPU test & synthetic benchmark

With full fork checkout, from repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_asym_apc_native.py tests/test_asym_apc_repo_integration.py
python scripts/probes/asym_apc_native/simulate.py --groups 160 --seed 20261115
```

The current assistant sandbox had **no outbound DNS access to github.com**, so direct `git clone` failed. The baseline architecture and file contents were read through the authenticated GitHub connector. For local sandbox testing, only the small original-shaped runtime data types were mirrored; the test against **actual** A2 registry + Continuum runtime is included as `tests/test_asym_apc_repo_integration.py` but could not be run in that selective mirror. The locally executed selective-mirror tests gave **17 passed, 1 skipped**. The GitHub Actions CPU workflow then ran against the **full fork checkout (Python 3.11)** and reported **18 passed in 0.27 seconds**, including `tests/test_asym_apc_repo_integration.py`, in [workflow run #43697595312](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/actions/runs/43697595312). This verifies the existing A2 registry and Continuum integration seam in a real checkout; it does **not** prove GPU performance or an opt-in Native runner is wired.

Seeded CPU example (`groups=160`, two synthetic 8-block chains, 16 tokens/block, invented monotone quadratic cost curve, true future returns sampled with a time-locality correlation):

| Outcome | A0 tier/LRU | New proposal |
| --- | ---: | ---: |
| Victim-set differences | — | 160 / 160 |
| Realized lost/recomputed tokens (simulation only) | 7,552 | 1,920 |
| Synthetic prefill milliseconds, sum over groups | 448.40 | 60.45 |
| Paired independent synthetic groups: win/tie/loss | — | 59 / 93 / 8 |

These results are intentionally **mechanism-feasibility** measurements for a favorable constructed geometry: two competing chains, native free-queue order depth-ordered per chain, costs exactly representable by the synthetic curve, and nontrivial future revisits. They are **not** results from Phase 3, W1-C, BFCL, a real model, real GPU profiling, or a naturally sampled Agent workload. The 160/160 divergence is itself evidence of this workload's intentionally discriminating construction, **not** a predicted production intervention rate.

## What remains before any Native claim

1. Wire decision-time chain recency from real vLLM prefix/Agent lifecycle events to `observe_chain_access`, with matching clock domain and real hash/process identity.
2. Collect real calibration curves across meaningful total-prefix lengths and cached-depths; check monotonic measurement intervals, fit quality, and overhead. No curve ⇒ A0 fallback.
3. Validate true fork imports, retention+registry integration smoke, content identity, shared chains, and physical victim audit in a full checkout. Add a separately gated runner entry; **never** alter frozen experiments silently.
4. Replay actual archived content-verified decisions to count whether the new policy **really differs** from A0, including fallback rates and missingness. Historical Phase 3 lacks the needed chain-access timestamps and prefix-conditioned profile; do not invent those inputs.
5. Only after new study authorization, measure prefill/TTFT/E−C (with correct denominators and group-level paired units) and policy overhead in Native GPU execution.

**Conclusion:** The coordinator and contiguous-prefix cost objective can be implemented against the project's existing Block-selection seam and make legal different choices in CPU simulation. Neither algorithm effectiveness on real Agent workloads nor Native end-to-end compatibility is proven by these artifacts.
