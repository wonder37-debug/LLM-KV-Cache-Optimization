# AsymAPC trustworthy evidence bridge — experimental follow-up

## Root cause / evidence boundary

Phase 3: A2 returned `content_unverified` in 74/127 decisions. The original A2 contract globally fails closed whenever **any** legal cached hashed Block lacks verified chain membership/weight; not necessarily when the A0-selected victims were themselves unverified. Native vLLM owns the actual block hash/refcount/free queue; the project side owns a subset of observed program-prefix chains. A cached block from another/unobserved/expired or recycled prefix may be legal to evict while not provable as part of a tracked chain. The aggregate 74/127 cannot identify which mechanism dominated. `content_unverified` and no candidate-level intervention are separate problems.

## Implemented (experimental branch only)

- Keep `RetentionRuntimeIntegration` unchanged. Its mandatory eligibility tier, valid block ids, ref_cnt and `remove_selected`/Native ownership stay intact.
- In `A2NativeSnapshotRegistry.observe`, accept optional `observed_at` and `clock_domain` **only from a real, native prefix-snapshot event**. Bind to `(native_hash_process_id, exact ordered content_hashes)`; invalidate on changed generation without a new observation; reject backdating/NaN, never synthesize timestamps.
- The existing Native runner now passes its already-recorded `snapshot_timestamp` from `SystemMonotonicClock()` at `VLLM_PREFIX_SNAPSHOT` creation. This is an observational source change, not a modification of the frozen A2 policy.
- `AsymAPCNativeCoordinator` consumes this access observation when the clocks and content generation match, eliminating the earlier requirement that every caller separately inject `observe_chain_access`.
- Replace whole-queue cost-evidence fallback with **per-candidate** verification: unrelated unverified hashed blocks *outside* A0's selected victims are excluded from the proposal, but the verified subset can still be scored and selected. Every proposed victim must have verified content identity/chain depth/access/profile, or be provably hashless. A baseline victim that cannot be proven stays A0; no false zero-loss assumptions, no tier crossing. Per-reason missing counts appear in `AsymAPCAudit.missing_reasons`. `selected_partial_coverage` clearly labels optimized decisions when unrelated evidence was missing.
- Exact hash matching, process/generation matching, no future access, actual Native free-queue rank, and refcount verification remain compulsory.

## What cannot be guaranteed

No vLLM API can prove an unobserved future reuse, nor reconstruct a full logical prefix chain from an isolated opaque `BlockHashWithGroupId` alone. A cache free queue can contain blocks whose parent/request chain was never recorded. Without new GPU measurements, the prefilling curve is synthetic or absent; no credible measured cost can be invented. Even with all inputs valid, a policy may tie native LRU, or be forced to evict the whole eligible set. The branch therefore **reduces unnecessary global evidence fallback**, not the existence of all fallback routes. Historical Phase 3 archives do not contain all time/profiling inputs required to recompute the new strategy.

## Tests

**Full-checkout GitHub Actions result:** [Run #43700964837](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/actions/runs/43700964837) — **PASS** on Python 3.11:
- 28 new selector / real A2 registry + Continuum integration tests passed;
- 177 existing A2 native/evidence/contract, Continuum pressure/integration, and vLLM observation-runner regression tests passed;
- 205 tests passed total, compileall passed, fixed-seed synthetic CPU replay passed.
- No GPU, no original Phase 3 archive re-execution, no measured latency-profile input. This is not a claim that the historical 74 fallbacks were retrospectively repaired.

Run on fork checkout:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_asym_apc_native.py tests/test_asym_apc_repo_integration.py \
  tests/test_costaware_a2_native.py tests/test_costaware_a2_evidence.py \
  tests/test_continuum_retention_integration.py tests/test_continuum_pressure.py
python scripts/probes/asym_apc_native/simulate.py --groups 160 --seed 20261115
```

Failure classification assertions:
- Unknown content in one later candidate cannot poison complete baseline + alternate evidence.
- Missing recency/profile in an unrelated chain cannot poison verified choices.
- Unknown A0 baseline victim is *not* assigned zero cost, and remains selected.
- Native exact-content/clock-domain mismatch and future-dated events are rejected.
- Reused physical slot/content generation invalidates old access timestamp.
- Legacy A2 and core Continuum remain untouched in functional semantics.

## Sources / related design

- vLLM v1 Prefix Caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
- AsymCache (MSA-specific costs are **not** transferred): https://arxiv.org/abs/2606.02964
- Existing fork contracts: `src/kvopt/costaware/a2_native.py`, `src/kvopt/runtime/vllm/bridge.py`, `src/kvopt/runtime/vllm/retention_integration.py`

## Next actual system work

1. Separate opt-in runner using `AsymAPCNativeCoordinator`, preserving existing `phase2g_block_policy` A0/A2 freeze; do not silently rewrite historical arms.
2. GPU-measure and validate prefix-depth Prefill curves for exact hardware/model/cache configuration; configure them explicitly.
3. Native pressure capture with before/after hash matching and controlled replay; count eligible and baseline victim proof coverage, partial-evidence selections, truly distinct victim sets, and observed cached-prefix/prefill effects.
4. Distinguish `evidence_missing`, `no_choice`, `model_tie`, `search_budget_exceeded`, and `content_identity_mismatch` in the results. Report CPU policy overhead and paired real-group outcomes.

**No production/GPU performance claim in this patch.**
