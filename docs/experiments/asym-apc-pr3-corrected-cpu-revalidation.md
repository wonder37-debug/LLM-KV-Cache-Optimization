# PR #3 follow-up: corrected real vLLM CPU BlockPool revalidation

**Status:** Executed, CPU-only; live vLLM 0.27.1 `BlockPool`, **no model serving or GPU inference**.

**Baseline:** PR #3 original commit `d91d238b630d4a0dd217a5efc71b7eafd0053fec` (backed up as `backup/asym-apc-pr3-before-validity-fix`).

**Proof:** [GitHub Actions pinned-CPU-wheel A/B/C replay #43794567374](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/actions/runs/43794567374) and [raw JSONL, summaries, and logs](https://github.com/wonder37-debug/LLM-KV-Cache-Optimization/actions/runs/43794567374/artifacts/14193542224). The live CPU replay used the official vLLM v0.27.1 CPU wheel on Ubuntu 24.04 and torch 2.13.0+cpu. Both jobs returned success. The self-contained repository test suite reported **2022 passed, 19 skipped, 2 explicitly deselected**. The two deselected tests require the absent historical W1-C `acquisition-sidecar.json`.

## 1. Why the original PR #3 raw results are historical, not a valid final causal comparison

The original `real_blockpool_eval.py` constructed an A2 observation registry and called `RetentionManager.upsert` only for `asym_apc`. The `a0` arm had **no populated Retention Entry state**. The original report's assertion that A0 and AsymAPC differed only by Block selector was consequently false. Retain all original artifacts as prior evidence, but do **not** use their percentages as independently validated algorithm gains.

The corrected probe instantiates `A2NativeSnapshotRegistry` in both controlled arms, observes Native chains in both, synchronizes actual residency and TTL-protection state in both, and feeds the identical `RetentionRuntimeIntegration` contract to both. A new CPU test checks that a second pressure event sees the same nonzero, protected Entry in both arms. The code also audits retention and protected-entry counts per pressure decision in actual CPU probe summaries.

The corrected probe uses `time.perf_counter` for decision CPU time instead of synthetic `ManualClock`, and configures the intended `AsymDecay(alpha_seconds=60, beta_seconds=12, knee_seconds=60)` rather than an accidentally single-exponential setting. These are substantive changes, so the new and old measurements must not be pooled as identical-policy replications.

## 2. Corrected controlled CPU BlockPool experiment

All cells share **8 sessions × 5 turns** (40 serial requests), 16 tokens/block, an append-only shared-prefix workload with 2/90-second inter-turn intervals, capacity held equal across arms, `PYTHONHASHSEED=0`, and a hash-pinned **CPU** cost curve. The three arms are `native_lru`, `a0`, and `asym_apc`. Native `get_cached_block` readback, not a future-demand oracle, determines tokens that would need Prefill.

| Capacity, blocks | A0 tokens needing Prefill | AsymAPC tokens needing Prefill | Change vs A0 |
|---:|---:|---:|---:|
| 128 | 20,032 | 20,032 | 0.0% |
| 160 | 20,032 | 19,680 | -1.8% |
| 224 | 17,472 | 16,752 | -4.1% |
| 240 | 16,576 | 15,552 | -6.2% |
| 256 | 15,680 | 14,144 | -9.8% |
| 264 | 15,232 | 12,720 | -16.5% |
| 272 | 14,784 | 10,336 | -30.1% |
| 280 | 14,336 | 8,080 | -43.6% |
| 288 | 13,888 | 5,952 | -57.1% |
| 292 | 13,888 | 4,928 | -64.5% |
| 296 | 4,672 | 4,672 | 0.0% |
| 320 | 4,672 | 4,672 | 0.0% |

The bounded interpretation is **yes, the fixed selector can change the physical cache outcome and reduce the number of prompt tokens that would require recomputation on this particular real vLLM CPU BlockPool workload**. It is not evidence of GPU TTFT improvement, statistically independent repeated trials, or improvement across arbitrary Agent workloads. All GPU latency fields are still unmeasured; CPU-calibrated predicted milliseconds must be labelled model-based. Each capacity is a different experimental regime, not an independent seed replication.

## 3. Independent correctness and runtime repairs

- **Generator version:** the materializer now selects the version actually accepted by the Phase 2C backend (`PILOT_PREFIX_GENERATOR_VERSION`).
- **Calibration units:** Native profile coverage checks convert token counts to 16-token Block counts.
- **Gated pressure-accounting option:** the new experimental campaign is explicitly authorized for `phase2g_unique_protected_blocks`; older frozen campaigns remain unchanged.
- **GPU report metrics:** use `VLLM_NATIVE_REQUEST_OBSERVATION.native_cached_prefix_tokens` and `native_prompt_tokens` (request-time facts) instead of the post-prefill `VLLM_PREFIX_SNAPSHOT`. Missing/duplicate Native records invalidate the corresponding aggregate, rather than being replaced with apparent full hits.
- **GPU calibrator:** after seeding exactly the requested prefix, each warm-up/measurement request uses a distinct uncached suffix (except the all-cached endpoint), preventing warm-ups from filling the next repetition's entire prompt cache.
- **Regression checks:** unit tests cover parity of A0/AsymAPC Retention Entries, actual Native request-time metrics, missing evidence, supported generator, and distinct calibration suffixes.

## 4. How to reproduce without a GPU

The GitHub Actions job uses the official pinned CPU wheel:

```bash
uv pip install --system --torch-backend cpu \
  "https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"
uv pip install --system -e .

export PYTHONHASHSEED=0
PROFILE=docs/experiments/asym-apc-native-evaluation-artifacts/calibration/cpu-device-prefill-v1.json
SHA=6266f3fe1a923dd14b0b05811c0978e564f52f566927bd7942486bf29c364e78
python scripts/probes/asym_apc_native/real_blockpool_eval.py \
  --profile "$PROFILE" --profile-sha256 "$SHA" \
  --allow-source cpu_device_measured \
  --num-blocks 288 --programs 8 --turns-per-program 5 \
  --seed 20260101 --out-dir out/corrected-cap288
```

No CUDA device, real `LLM.generate`, scheduler, batching, preemption or TTFT was exercised. The CPU wheel executes only the actual Native cache substrate. GPU code is implemented but **not executed**; additional activation risks include actual KV-block geometry coverage, perf profile quality and physical-reconciliation timing.

## 5. Remaining evaluation priorities

1. Run diverse, workload-independent traces with varied arrival orders, shared-prefix rates, tool gaps, and chain lengths; freeze a held-out set before tuning.
2. Run depth-only vs recency-only vs combined ablations and report selector CPU overhead alongside token savings. This fixed replay alone does not isolate the benefit of each term.
3. Exercise the fully gated Native Runner using a real GPU, with request-time cached-token evidence, exact physical eviction reconciliation, proper Prefill calibration and TTFT/E2E measurements.
4. Never promote the original uncorrected CSVs as final proof; use versioned fresh artifacts from the CI replay or a later held-out campaign.

**Research conclusion:** credible, bounded *cache-substrate feasibility* evidence after parity repair; **no validated serving/GPU performance claim**.
