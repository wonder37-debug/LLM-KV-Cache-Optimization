# AsymAPC fork main — consolidated merge and CPU feasibility verification

**Scope:** personal fork `wonder37-debug/LLM-KV-Cache-Optimization` only. **Evidence status:** CPU behavior and code integration verified; Native GPU performance **NOT EVALUATED**.

## 1. Merge provenance

- [PR #1](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/pull/1): experimental AsymCache-inspired *contiguous-prefix* APC selection, CPU simulator, unit and integration tests. Merged into fork `main` at `94ec68e7a2d88a5a9038b65472077dc236efbc9c`.
- [PR #2](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/pull/2): generation-qualified Native prefix access timestamps, per-candidate content validation and safe fallback, existing runner observation-only hook. Merged into fork `main` at `025416de488e545d81d2509c68fdcc142e17283f`.
- Frozen research baseline, upstream repository, original A0/A1/A2/A3/E0 arm semantics and historical archives are untouched.

## 2. Implemented decision boundary

The opt-in `AsymAPCNativeCoordinator` accepts a precomputed Continuum `EligibilityPreparation` and only reorders legal free-queue Block identities within the existing eligibility tiers. It does not unpin entries or mutate Native hash tables, refcounts, allocations, or actual free queue.

It uses the **observed** time of the latest verified Native prefix-snapshot generation, a two-timescale recency ranking proxy, and a calibrated/synthetic `APCLatencyCurve(total_blocks=m, cached_depth=r)`. The **contiguous-APC** proxy is

`L(S) = sum_c w_c(t) * [F(m_c, r_c after S) - F(m_c, r_c before S)]`.

A previously unobserved or otherwise unverified unrelated candidate is not silently assigned zero cost; it can be excluded without invalidating fully verified alternatives. If the baseline victim itself is unverified, or a refcount / physical-ID / Native hash / clock / depth / performance-curve contract is violated, preserve the exact A0 selection. The selector does not claim the original AsymCache MSA semantics.

## 3. Tests on merged main

The concentrated CI workflow is [`.github/workflows/asym-apc-consolidated.yml`](../../.github/workflows/asym-apc-consolidated.yml).

### Direct and adjacent CPU tests

- **28 new algorithm plus real-fork integration tests passed**.
- **177 A2, Continuum, Native observation and pressure regression tests passed**.
- Code compiled with `python -m compileall`; no test modified the frozen dataset.
- The `RetentionRuntimeIntegration` + `A2NativeSnapshotRegistry` integration test selects an alternative Block and verifies that the supplied physical-removal callback sees that Block. This is a **CPU mocked free-queue test**, not a real vLLM running model.

### Five seeded synthetic paired replays

Each seed generates 320 independently sampled synthetic groups, total **1,600 groups**, with two content chains, a fixed pressure target, synthetic monotone prefill profile and hidden future arrivals sampled only after the decision. Baseline and proposal use the same future outcomes. No GPU, actual vLLM model inference, measured TTFT or real Agent trace.

| Seed | Decision divergences | Better / Tie / Worse groups | A0 recompute tokens | Proposed recompute tokens |
| --- | ---: | ---: | ---: | ---: |
| 20261115 | 320 | 117 / 186 / 17 | 14,976 | 3,712 |
| 20261116 | 320 | 106 / 196 / 18 | 13,568 | 3,408 |
| 2027 | 320 | 106 / 206 / 8 | 13,568 | 3,280 |
| 42 | 320 | 110 / 199 / 11 | 14,080 | 3,424 |
| 7 | 320 | 133 / 173 / 14 | 17,024 | 4,432 |
| **Total** | **1,600** | **572 / 960 / 68** | **73,216** | **18,256** |

Synthetic recomputation decrease: **75.07%**. The workload deliberately offers many choices and depth-ordered queues; this result is **not** a real-world performance estimate. Even here 68 groups are worse.

### Full repository test diagnostic and unavailable provenance

The raw all-tests command reported **1,979 passed, 19 skipped, 2 failed** in [the full-suite diagnostic run](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/actions/runs/43710105852). Both failures occur in `tests/test_continuum_bounded_campaign.py`, because the fork checkout is missing `artifacts/continuum-w1c-pilot-runs-v1/S100/psf__requests-2931/acquisition-sidecar.json`:

- `test_materialize_campaign_freezes_54_runs_and_cross_arm_time`
- `test_validation_rejects_tampered_logical_sequence`

The next CI run executes the entire **available** repository suite using explicit `--deselect` for exactly those two tests, and keeps their unavailable status in the workflow and report. This does not turn them into passes or synthesize provenance.

## 4. Verification and readiness conclusion

**PASS, bounded CPU feasibility:** the algorithm exists in `main`, computes a different lower-proxy-loss legal victim set with complete/partial verified evidence, honors Continuum tiers and Native candidate checks, and passes direct/adjacent regressions.

**NOT YET ESTABLISHED, Native serving effectiveness:** the existing `phase2g_block_policy` runner only accepts `A0` and `A2`. The AsymAPC Coordinator has not been wired to a new explicitly gated Native runner arm. There is no GPU-calibrated cost curve, true model-serving run, measured time overhead, actual pressure replay of archived Phase 3 candidate sets, or real-TTFT improvement. Phase 3 old A2 `content_unverified` fallback count cannot be retroactively recalculated.

**Interpretation:** deployability as a read-only CPU selection policy is supported, while end-to-end Native feasibility and production usefulness remain an open experiment.
