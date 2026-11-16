# Independent validation artifacts

Raw evidence bundle for [`../asym-apc-independent-validation.md`](../asym-apc-independent-validation.md).

**Execution state: real vLLM 0.27.1 `BlockPool`, CPU device, no GPU, no serving engine.**
Every cell records `scope.gpu_executed: false`, `scope.latency_is_model_based: true`,
`scope.real_vllm_block_pool: true`, `scope.recomputed_tokens_are_measured: true`.

## Layout

| Path | Content |
| --- | --- |
| [`campaigns/repro/`](campaigns/repro) | 12 capacities × 3 arms, reference reproduction of the corrected CI replay |
| [`campaigns/ablation/`](campaigns/ablation) | 8 capacities × 4 arms (`a0`, `depth_only`, `recency_only`, `asym_apc`) |
| [`campaigns/decay-current/`](campaigns/decay-current) [`-medium`](campaigns/decay-medium) [`-slow`](campaigns/decay-slow) | decay sweeps at `60/12/60`, `240/60/120`, `600/180/300` s |
| [`campaigns/exploratory2/`](campaigns/exploratory2) | 8 families × 2 capacity fractions × 4 arms, exploratory seed `20260101` |
| [`campaigns/heldout/`](campaigns/heldout) | 8 families × 2 fractions × 3 held-out seeds × 4 arms (48 cells) |
| [`campaigns/heldout-slow/`](campaigns/heldout-slow) | tuned decay re-checked on held-out seeds |
| [`campaigns/overhead-p1/`](campaigns/overhead-p1) [`-p2`](campaigns/overhead-p2) | quiet-host overhead ladder, two passes, full re-sum search |
| [`campaigns/overhead-capacity-full/`](campaigns/overhead-capacity-full) [`-delta/`](campaigns/overhead-capacity-delta) | same 12-capacity sweep, full re-sum versus delta-priced search, run back to back |
| [`campaigns/overhead-families-full/`](campaigns/overhead-families-full) [`-delta/`](campaigns/overhead-families-delta) | 8 families × 2 fractions, same two paths |
| [`campaigns/overhead-cap192-full/`](campaigns/overhead-cap192-full) [`-delta/`](campaigns/overhead-cap192-delta) | the 192-block regime that was previously inconclusive |
| [`analysis/`](analysis) | `cells.csv`, `family-arm-summary.csv`, `decisions-overhead.csv`, `overhead-strata.csv`, `overhead-scaling.csv`, `overhead-correlations.csv` per campaign |
| [`analysis/overhead/`](analysis/overhead) | before/after decision-cost tables, and the decision-identity reports that show the two paths produced the same victims |
| [`calibration/`](calibration) | the active profile, and the rejected wide-profile attempt (see report §11.4) |
| [`MANIFEST.json`](MANIFEST.json) | per-campaign matrix, calibration SHA-256, and a SHA-256 for every file |

## One cell

| File | Content |
| --- | --- |
| `summary.json` | engine identity, family description, `geometry_signature`, capacity fraction, working set, decay config, per-arm aggregates, paired deltas against A0, scope flags |
| `requests-<arm>.jsonl` | per request: prompt blocks, cached prefix read back from the pool, recomputed tokens, evictions during the request, free blocks after, modelled ms |
| `decisions-<arm>.jsonl` | per decision: candidate counts, baseline vs selected victim IDs, proxy ms, predicted gain, divergence flag, fallback reason, evidence-missing counts, `selector_overhead_ns`, `observed_chains` |
| `SHA256SUMS.json` | hashes of the three files above |

`a0` and `native_lru` produce empty `decisions-*.jsonl` files because they do not select; that
is a recorded absence, not a missing artifact.

## Decision identity

`analysis/overhead/identity-*.json` compares two campaigns row by row: every request line and
every decision line (candidate counts, baseline and selected victim IDs, reason, proxy ms,
missing-evidence counts) must match, with the measured `selector_overhead_ns` and the
`loss_evaluation` label excluded. It also lists the overhead series behind the tables.

Reproduce with:

```bash
python scripts/probes/asym_apc_native/compare_campaigns.py \
  --baseline docs/experiments/asym-apc-independent-validation-artifacts/campaigns/overhead-capacity-full \
  --candidate docs/experiments/asym-apc-independent-validation-artifacts/campaigns/overhead-capacity-delta \
  --out /tmp/identity.json
```

## Why cells are grouped by campaign

Each campaign is a self-contained comparison: all arms inside a cell share the same profile
SHA-256, capacity, turn list and decay configuration. Cross-campaign comparison is only valid
where the profile and decay are identical, which is why the report never pools them and why the
manifest records both per campaign.

## Verify

```bash
cd docs/experiments/asym-apc-independent-validation-artifacts
python3 - <<'PY'
import hashlib, json, pathlib
manifest = json.loads(pathlib.Path("MANIFEST.json").read_text())
files = [f for c in manifest["campaigns"] for f in c["files"]]
files += manifest["analysis_files"] + manifest["calibration_files"]
bad = [f["path"] for f in files
       if hashlib.sha256(pathlib.Path(f["path"]).read_bytes()).hexdigest() != f["sha256"]]
print("files:", len(files), "mismatches:", len(bad))
PY
```

## Reading rules

* Recomputed tokens are measured against live pool state; milliseconds are a replay of a
  CPU-measured curve. Do not quote them as GPU latency.
* A cell with zero divergence is not a failure of the harness: check `reason` and
  `evidence_missing_reason_counts` before drawing a conclusion.
* `profile_missing` means the resident chain length was never calibrated — the policy is blind
  there by construction.
* `depth_only` and `asym_apc` share the same evidence and candidate sets; only the weight
  differs. Any difference between them is the recency term's contribution.
* Every campaign produced before the delta-priced search exists is still the full re-sum
  result; campaigns whose name ends in `-full` or `-delta` state which path produced them in
  `selector.loss_evaluation` inside each cell's `summary.json`.
