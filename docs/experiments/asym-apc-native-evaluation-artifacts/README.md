# AsymAPC native evaluation artifacts

> **Historical pre-fix dataset.** The original A0 and AsymAPC arms did not have
> equivalent populated Retention Entry state. Keep this manifest/archive for
> reproducibility; use the [corrected CPU replay report](../asym-apc-pr3-corrected-cpu-revalidation.md)
> and the linked fresh CI artifacts for causal comparison.
>
Complete evidence bundle for the AsymCache-inspired contiguous-prefix victim selector
evaluated against the real vLLM `BlockPool`.
Design: [`experiment-design.md`](experiment-design.md). Report:
[`../asym-apc-real-blockpool-evaluation.md`](../asym-apc-real-blockpool-evaluation.md).

**Execution state: real vLLM 0.27.1 cache code, CPU device, no GPU.** No CUDA device existed
on the host, so no GPU measurement, no serving engine and no measured latency is included.
Every `summary.json` states this in `scope`: `gpu_executed: false`,
`latency_is_model_based: true`, `real_vllm_block_pool: true`,
`recomputed_tokens_are_measured: true`.

## Layout

| Path | Files | Content |
| --- | --- | --- |
| [`experiment-design.md`](experiment-design.md) | 1 | hypotheses, arms, workload, capacity grid, instruments, decision rules, analysis plan, threats, deviations |
| [`results/arm-metrics-by-capacity.csv`](results/arm-metrics-by-capacity.csv) | 69 rows | every arm × every capacity of the three authoritative run sets |
| [`results/paired-deltas.csv`](results/paired-deltas.csv) | 23 rows | per-capacity pairing: Native LRU vs A0 vs ASYM_APC, absolute and relative deltas |
| [`raw/MANIFEST.json`](raw/MANIFEST.json) | 31 runs | per-run config, classification, file list with SHA-256 for all 462 raw files |
| [`raw/blockpool/authoritative/`](raw/blockpool/authoritative) | 161 | `sweep2` (8 capacities), `fine` (9 capacities), `replication` (2 capacities × 3 seeds) |
| [`raw/blockpool/superseded/`](raw/blockpool/superseded) | 301 | 28 earlier runs, each annotated with the harness defect that invalidated it |
| [`raw/calibration/`](raw/calibration) | 8 | measured prefill curve, probe iterations, calibration stdout, feasibility script |
| [`calibration/`](calibration) | 2 | the profile consumed by every arm, and its run log |
| [`vllm-blockpool-eval/`](vllm-blockpool-eval) | 27 | curated per-capacity summaries; `cap288-detail/` holds the full trace at the headline capacity |
| [`vllm-blockpool-eval/environment.txt`](vllm-blockpool-eval/environment.txt) | 1 | cgroup limits, CPU quota, absence of GPU |
| [`test-run-summary.txt`](test-run-summary.txt) | 1 | full-repository test outcome |

## What one run directory contains

Six files per capacity point, three per arm:

| File | Content |
| --- | --- |
| `summary.json` | engine versions, workload spec, calibration profile path + SHA-256, scope flags, per-arm aggregates, paired deltas |
| `requests-<arm>.jsonl` | one record per request: prompt blocks, cached prefix blocks read back from the pool, recomputed tokens, prefix hit rate, blocks evicted during the request, free blocks after, modelled prefill ms, status |
| `decisions-<arm>.jsonl` | one record per allocation decision: required blocks, legal/ranked candidate counts, baseline and selected victim block IDs, proxy ms for both, predicted gain, divergence flag, fallback reason and reason category, selector overhead, claim boundary |

`native_lru` has no decision records by construction: the control does not decide anything.
`a0` records decisions taken in native order, which is what makes the placebo check possible.

## Regenerating the tables

The CSVs are derived from the run directories, never hand-edited. The aggregation walks
`raw/blockpool/authoritative/<run_set>/<capacity>*/summary.json` and, for each of the three
arms, copies `total_recomputed_tokens`, `mean_recomputed_tokens`, `mean_prefix_hit_rate`,
`physically_evicted_blocks`, `decisions`, `pressure_events`, `victim_set_divergence_count`,
`modelled_prefill_ms_total`, `modelled_prefill_ms_mean` and `wall_seconds`, then attaches
`paired.recomputed_token_relative_change` and `paired.hit_rate_delta` from the same file.
A capacity point contributes one row per arm, so 23 capacity points give 69 rows.

Column names are English and units are in the header (`*_tokens`, `*_ms`, `*_blocks`). Empty
cells mean the quantity is undefined for that pairing, never zero — a policy that could not
be compared must not read as "no change".

## Integrity

`raw/MANIFEST.json` is the authority for what was measured:

* `runs[].config` — capacity, programs, turns, seed, generation timestamp, scope flags
* `runs[].group` — `authoritative` or `superseded`
* `runs[].note` — why a superseded run is superseded
* `runs[].files[].sha256` — content hash per file, so a checked-out tree can be verified file
  by file without trusting this README

Verify the checkout:

```bash
cd raw && python3 - <<'PY'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("MANIFEST.json").read_text())
bad = []
for run in m["runs"]:
    for f in run["files"]:
        p = pathlib.Path(f["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            bad.append(f["path"])
print("files:", sum(len(r["files"]) for r in m["runs"]), "mismatches:", len(bad))
PY
```

## Reading the numbers without over-claiming

* `total_recomputed_tokens` is measured against pool state. `modelled_prefill_ms_*` is a replay
  of a CPU-measured curve and must not be quoted as GPU latency.
* The gain only exists in the pressured-but-fitting regime: 0 % at ≤144 blocks (thrash) and
  0 % at ≥296 blocks (fits). A single capacity point is not the result.
* A0 is byte-identical to Native LRU at every capacity, so the effect is block selection.
* Divergence counts are not savings: at 128–144 the victim set changes on 34/40 decisions with
  zero token change.
* Three seeds produce bit-identical traces; the seed axis is a determinism check, not
  replication with variance.
