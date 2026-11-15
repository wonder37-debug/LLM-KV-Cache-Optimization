# AsymAPC-inspired contiguous-prefix Block eviction: Native integration and evaluation report

**Scope:** personal fork `wonder37-debug/LLM-KV-Cache-Optimization`, branch
`experiment/asym-apc-native-gpu-arm`.
**Status of this document:** implementation complete; Native GPU execution **not performed**
in the environment where this work was done. See §2 for the exact blocker, §4 for what was
actually verified, and §10 for the resulting claim boundary.

Evidence labels used throughout:

| Label | Meaning |
| --- | --- |
| `IMPLEMENTED_UNEXECUTED` | Code exists and is reachable, but has never run against the pinned Native runtime. |
| `CPU_TESTED` | Executed on CPU with real repository components and/or test doubles. |
| `NATIVE_GPU_TESTED` | Executed against real vLLM inference on a GPU. |
| `OBSERVED_IMPROVEMENT` | A measured improvement, with raw artifacts. |
| `INCONCLUSIVE` | Evidence exists but cannot support the claim. |

---

## 1. Objective and implementation

### 1.1 Objective

Determine whether the AsymCache-inspired contiguous-prefix Block victim selector gives a
measurable, reproducible benefit over Continuum Entry Release + Native LRU (`A0`), using real
vLLM execution instead of the earlier synthetic simulator.

The policy minimizes the weighted loss of reusable contiguous prefixes,

```
L(S) = Σ_c w_c(t) · [ F(m_c, r'_c(S)) − F(m_c, r_c) ]
```

over the legal victim set `S`, inside unaffected Continuum TTL, Entry Release, eligibility-tier
and Native BlockPool ownership semantics. `F(m, r)` is measured Prefill latency for `m` blocks
with `r` reusable cached prefix blocks.

### 1.2 What existed before this branch

`main` at `3bb8549` contained the selector (`src/kvopt/costaware/asym_apc_native.py`), the A2
Native evidence registry, and CPU tests. The formal Native runner
(`scripts/spikes/phase2_minimal_observability_metal.py`) accepted only
`phase2g_block_policy ∈ {A0, A2}`, so the policy could not be selected on a real serving path.
`docs/experiments/asym-apc-main-consolidated-validation.md` recorded exactly this gap, and also
recorded that the ~75 % recomputed-token reduction came from a simulator whose depth-ordered
free queue may favour the selector.

### 1.3 What this branch adds

| Area | File | Status |
| --- | --- | --- |
| Gated `ASYM_APC` Native arm, opt-in parsing, profile binding | `src/kvopt/costaware/asym_apc_runner.py` | `IMPLEMENTED_UNEXECUTED` / `CPU_TESTED` |
| Runner dispatch, decision audit emission, physical reconciliation hooks | `scripts/spikes/phase2_minimal_observability_metal.py` | `IMPLEMENTED_UNEXECUTED` / `CPU_TESTED` |
| GPU Prefill calibration protocol and validation | `src/kvopt/costaware/asym_apc_calibration.py` | `IMPLEMENTED_UNEXECUTED` / aggregation `CPU_TESTED` |
| Calibration CLI (fresh process per measurement point) | `scripts/probes/asym_apc_native/calibrate.py` | `IMPLEMENTED_UNEXECUTED` (dry-run `CPU_TESTED`) |
| A/B/C campaign design (Native LRU / A0 / ASYM_APC) | `configs/asym_apc/asym-apc-native-evaluation-v1.json`, `src/kvopt/workload/asym_apc_materialize.py` | `IMPLEMENTED_UNEXECUTED` / materialization `CPU_TESTED` |
| Campaign execution | `src/kvopt/workload/asym_apc_runner.py` | `IMPLEMENTED_UNEXECUTED` / failure accounting `CPU_TESTED` |
| Paired analysis and Markdown report | `src/kvopt/profiling/asym_apc_report.py` | `IMPLEMENTED_UNEXECUTED` / arithmetic `CPU_TESTED` |
| Tests | `tests/test_asym_apc_native_arm.py` (22), `tests/test_asym_apc_campaign.py` (7) | `CPU_TESTED` |

### 1.4 Design decisions worth recording

1. **Opt-in, fail-closed.** `ASYM_APC` requires
   `backend_options.phase2g_block_policy = "ASYM_APC"`, an authorized campaign kind
   (`asym_apc_native_evaluation`), Native content evidence, and
   `backend_options.asym_apc.enabled = true` with a SHA-256-bound calibration profile. Any
   missing field aborts before a model is loaded. `A0` remains the default and is unchanged.
2. **A calibration profile cannot be a simulator artifact.** The loader rejects
   `measurement_source != "native_gpu"`, requires provenance fields (model name and revision,
   hardware, block size, vLLM version, runtime mode, measurement timestamp), and refuses
   resident chain lengths that were never measured.
3. **The arm is constructed after evidence, not before.** The coordinator is built only once the
   A2 snapshot registry exists, so the policy can never run against an empty evidence store and
   silently appear to be `A0`.
4. **Proposed victims are not evidence.** Reconciliation compares the audit's proposed Block IDs
   with the Block IDs observed by the Native hashed-eviction callback inside the same execution
   context, and reports unmatched, unexpected and deferred cases separately.
5. **Two-phase reconciliation.** Native eviction callbacks can arrive after the allocation call
   returns. A decision with no observed eviction yet is left pending and settled again at
   execution-context close instead of being reported as a negative.
6. **Nothing frozen was touched.** `A0/A1/A2/A3/E0` semantics, the Phase 3 Protocol v4, the
   sealed cohort, and historical artifacts are unchanged. The only gating change is that the new
   campaign kind may use `policy = "native-lru"` (true Native baseline) and `N0` release, both
   previously restricted to earlier campaigns.

### 1.5 Exact commands

```bash
# repository and branch
git clone https://github.com/wonder37-debug/LLM-KV-Cache-Optimization.git
git checkout -b experiment/asym-apc-native-gpu-arm

# quality gates actually executed here
python -m pytest tests -q
python -m pytest tests/test_asym_apc_native_arm.py tests/test_asym_apc_campaign.py -q
python -m pytest tests/test_asym_apc_native.py tests/test_asym_apc_repo_integration.py \
                tests/test_costaware_a2_native.py tests/test_continuum_vllm_retention_hook.py -q
python -m compileall -q src scripts/spikes/phase2_minimal_observability_metal.py
python -m ruff check src/kvopt/costaware/asym_apc_runner.py \
                     src/kvopt/costaware/asym_apc_calibration.py \
                     src/kvopt/profiling/asym_apc_report.py
python scripts/probes/asym_apc_native/calibrate.py --dry-run --grid 4,8,12,16,20

# what a GPU host must run next (not executed here)
python scripts/probes/asym_apc_native/calibrate.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct --revision <pinned-sha> \
  --grid 4,8,12,16,20 --warmups 3 --repeats 7 \
  --output results/asym_apc/calibration/prefill-calibration-profile.json

python -m kvopt.workload.asym_apc_materialize \
  --design configs/asym_apc/asym-apc-native-evaluation-v1.json \
  --output-dir results/asym_apc/campaign-screening \
  --profile results/asym_apc/calibration/prefill-calibration-profile.json \
  --subset screening

python -m kvopt.workload.asym_apc_runner \
  --manifest results/asym_apc/campaign-screening/manifest.json \
  --config-root results/asym_apc/campaign-screening \
  --output-root results/asym_apc/runs-screening

python -m kvopt.profiling.asym_apc_report \
  --run-dir results/asym_apc/runs-screening/<run-id> \
  --output results/asym_apc/analysis/summary.json
```

## 2. Runtime environment

### 2.1 Intended environment

The task and repository target an RTX 4070 under Windows 11 + WSL2 Ubuntu, vLLM 0.27.1 with GPU
Automatic Prefix Caching, block size 16, and `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

### 2.2 Environment actually available

The work was performed in an isolated Linux container with **no NVIDIA device of any kind**:

```
$ nvidia-smi
(eval):2: command not found: nvidia-smi

$ ls /dev/nvidia*
ls: cannot access '/dev/nvidia*': No such file or directory

$ ls /proc/driver/nvidia
ls: cannot access '/proc/driver/nvidia': No such file or directory

$ python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
2.10.0+cu128 False 0

$ python3 -c 'import vllm'
ModuleNotFoundError: No module named 'vllm'

$ uname -a
Linux 0eceec0ed1cb 6.6.117-45.19.1.tl4.x86_64 ... x86_64 GNU/Linux
$ grep -c ^processor /proc/cpuinfo   # 32
$ free -g | head -2                  # 123 GiB total, 93 GiB available
$ lspci | grep -i vga                # no output
```

**Blocker classification:** absent GPU hardware and absent NVIDIA driver/kernel interface inside
the execution sandbox. This is not a code, dependency or configuration failure that can be fixed
locally: no CUDA context can be created, so `LLM(...)` with a CUDA device cannot be constructed,
and vLLM is not installed. No approved accessible GPU environment was provided for this task.

**What was deliberately not done:** no CPU mock was substituted for a GPU experiment, no
synthetic numbers were written into a profile labelled `native_gpu`, and no measurement was
inferred from the earlier simulator.

## 3. Experiment methodology

### 3.1 Arms

| Arm | Entry Release | Physical Block victim | Purpose |
| --- | --- | --- | --- |
| `NATIVE_LRU` | none (Native cache management) | Native LRU | System baseline |
| `A0` | Frozen Continuum Entry Release | Native LRU | Adapted baseline |
| `ASYM_APC` | same `A0` Entry Release | AsymCache-inspired selector | Candidate |

Primary comparison is `ASYM_APC − A0`; `A0 − NATIVE_LRU` isolates the Continuum contribution.
Only the Block-selection decision varies between `A0` and `ASYM_APC`; the Entry Release path is
byte-identical by construction.

### 3.2 Controls held identical

Per paired unit (case × pressure regime × repeat) the materializer emits three configs that share
trace file and hash, `program_prefix_tokens`, `prefix_segments_by_request`,
`pressure_prompt_tokens`, `initial_shortage_blocks`, `block_override`, `seed`, model and
tokenizer revision, generation settings, batching and GPU memory settings. Only
`policy`, `phase2g_release_policy`, `phase2g_block_policy` and the `asym_apc` block differ.
`test_materialize_screening_campaign` asserts this equality field by field.

### 3.3 Cache isolation between runs

Every run is a separate process (`_run_phase2d_stage` → fresh backend → fresh `LLM`), so no
resident prefix, hash table entry, refcount or free-queue order can leak across paired runs. The
calibrator applies the same rule per measurement point, one child process per
(prompt length, cached depth).

### 3.4 Workload conditions

Six materialized cases, split explicitly into representative and stress classes:

| Case | Class | Intent |
| --- | --- | --- |
| `w1-long-prefix-reuse` | representative | two sessions replay 768–1024-token stable prefixes |
| `w2-multi-session-competition` | representative | four sessions compete; fewer long prefixes fit than exist |
| `w3-short-tool-gap` | representative | 1.5 s tool gap, reuse highly likely |
| `w4-long-tool-gap` | representative | 45 s tool gap, reuse genuinely uncertain |
| `w5-shared-prefix` | stress | three sessions share a 640-token preamble; evicting a shared block damages several chains |
| `w6-mixed-context-lengths` | stress | one 1536-token session against two 256-token sessions |

Pressure regimes are `low` / `medium` / `high` (`initial_shortage_blocks` 2 / 6 / 12) with an
identical pressure prompt per case, so cache pressure varies without changing the workload.

Sizes: `screening` = 2 cases × 2 regimes × 3 arms × 2 repeats = **24 runs**;
`full` = 6 × 3 × 3 × 3 = **162 runs**. Both are `IMPLEMENTED_UNEXECUTED`.

### 3.5 Native LRU ordering is measured, not assumed

The task explicitly warns that Native LRU already reacts to prefix depth. The implementation
therefore records, per decision: legal Block IDs in Native free-queue order (`native_rank`
order), each candidate's content hash and chain memberships with depths, the A0 baseline victim
tuple, the selected tuple, and whether the *set* differs. Victim *ordering* is never reported as
a performance signal, and `victim_set_divergence_rate` is reported separately from reuse,
recomputed tokens and latency in the analysis output.

`A0`'s baseline is read from `preparation.virtual_eligible_order`, i.e. the real Continuum
eligibility plan over the real Native free queue — not a reconstructed depth-ordered list. This
is the specific defect the earlier simulator could not exclude.

### 3.6 Statistical structure

The paired unit is `(case, pressure regime, repeat)`. The analysis reports, per metric: paired
unit count, median/mean delta, standard deviation across pairs, min/max, and the count of
better/tie/worse units for recomputed tokens. With 2–3 repeats per cell the honest interpretation
is "sign consistency plus spread", not a significance claim; §8 states this limitation.

## 4. Correctness validation

### 4.1 Test inventory and results

| Suite | Result |
| --- | --- |
| `tests/test_asym_apc_native_arm.py` (new) | **22 passed** |
| `tests/test_asym_apc_campaign.py` (new) | **7 passed** |
| Combined AsymAPC + A2 + Continuum retention subset (includes the new tests) | **97 passed** |
| Whole repository | **2007 passed, 19 skipped, 3 failed** |

The three whole-repository failures are pre-existing environment gaps, not regressions:

* `tests/test_continuum_bounded_campaign.py` (2 tests) — the fork checkout lacks
  `artifacts/continuum-w1c-pilot-runs-v1/S100/psf__requests-2931/acquisition-sidecar.json`; this
  is the same unavailable-provenance failure already recorded in
  `docs/experiments/asym-apc-main-consolidated-validation.md` §3.
* `tests/test_continuum_logging.py` — the test spawns a subprocess that imports `kvopt` without
  the repository `PYTHONPATH`; the package is not pip-installed in this container.

Baseline on `main` before this branch was 1975 passed / 19 skipped / 6 failed in the same
container; the extra passes are the 29 new tests plus three tests that require a clean git
worktree, which this branch satisfies once committed.

### 4.2 What the new tests actually prove (`CPU_TESTED`)

* **Opt-in gating** — `ASYM_APC` is refused outside its campaign, refused without Native content
  evidence, and refused without an explicit `asym_apc.enabled = true` object; `A0` remains the
  default.
* **Profile binding** — a profile with a mismatched SHA-256, a `simulator` measurement source, or
  missing provenance is rejected; resident chain lengths outside the measured grid are rejected.
* **Decision audit** — the instrumented boundary records the exact reason, its category, measured
  selector overhead, proxy gain and victim-set divergence, and exposes the pending decision
  exactly once.
* **Evidence discipline** — with no observed access timestamp the arm falls back to the exact A0
  victim set with reason `recency_missing` and category `EVIDENCE_INCOMPLETE`; it never invents
  recency or promotes an unverified candidate.
* **Physical reconciliation** — exact match, partial match, disjoint sets, unexpected evictions,
  deferred settlement (no eviction observed at decision return, settled at context close), and
  rejection of a proposal that disagrees with the recorded audit.
* **BlockPool evidence** — availability, `PARTIAL` state and explicit
  `native_free_block_queue_length_unavailable` reasons; no fabricated counts.
* **Next-request reuse expectation** — expected reusable prefix depth per chain after a victim
  set, comparison against the observed depth, and `MATCH` / `MISMATCH` / `UNCHECKED` /
  `NO_EXPECTATION` classification.
* **Campaign structure** — 24 identical-input paired runs, correct per-arm wiring
  (`native-lru` + `N0` for `NATIVE_LRU`, `A0` release for both Continuum arms), profile SHA-256
  propagation into each `ASYM_APC` config, and rejection of a tampered design.
* **Failure accounting** — a run that raises is recorded as a failure with its reason and is
  never counted as a completed observation.
* **Report arithmetic** — recomputed tokens, hit rate, divergence, physical exact-match rate, and
  the `better`/`tie`/`worse` paired sign counts, plus explicit `None` with a reason when prompt
  token counts are unavailable.
* **Calibration gates** — a depth whose observed APC hit disagrees with the requested depth is
  excluded and a depth with too few valid samples fails the calibration; a non-monotone
  measured curve fails instead of being clamped; interpolation error is measured; a
  non-discriminative profile fails the preflight.

### 4.3 What the tests do **not** prove

* That the selector receives real vLLM free-queue candidates on a GPU (`IMPLEMENTED_UNEXECUTED`).
* That a selected Block ID reaches `BlockPool._maybe_evict_cached_block` on a GPU
  (`IMPLEMENTED_UNEXECUTED`).
* That a follow-up request observes the predicted cached-prefix depth on real hardware
  (`IMPLEMENTED_UNEXECUTED`).
* Any latency, throughput or recomputed-token effect (`INCONCLUSIVE`, no data).

The reconciliation, pool capture and reuse checks are the mechanisms that would turn a
callback-level demonstration into a cache-state demonstration; they are wired and unit-tested,
but they have not observed a real eviction.

## 5. Prefill calibration and accuracy

Status: `IMPLEMENTED_UNEXECUTED`. No profile exists for this repository, and none was fabricated.

### 5.1 Protocol

For each prompt length `m ∈ {4, 8, 12, 16, 20}` blocks (64–320 tokens at block size 16) and each
cached depth `r ∈ [0, m]` (65 points in total):

1. a fresh child process starts a fresh `LLM` with `enable_prefix_caching=True`,
   `enforce_eager=True`, `num_gpu_blocks_override` fixed, `prefix_caching_hash_algo="sha256"`;
2. one request with the first `r` blocks inserts that prefix;
3. `--warmups 3` warm-up and `--repeats 7` measured generations of the full prompt run with
   `max_tokens=1`, timed as `torch.cuda.synchronize()` → `perf_counter` → `generate` →
   `torch.cuda.synchronize()`;
4. every measured request's recorded cached-token count must equal `r × block_size`; otherwise
   the sample is marked invalid with `apc_hit_depth_mismatch:<observed>!=<expected>` or
   `apc_hit_evidence_unavailable:<source>`.

### 5.2 Aggregation and validation

* median, P90, min, max and relative spread per depth;
* `noise_floor = 3 × max(P90 − median)` across depths of a curve;
* monotonicity required within the noise floor; a violation beyond it raises
  `CalibrationError` and the violating `(depth, earlier, later)` triple is reported — no clamping;
* `discriminative = min(adjacent median gap) > noise_floor`; if false the curve is recorded as
  `INSUFFICIENT_RESOLUTION` and `preflight_profile` refuses the campaign by default, because a
  curve that cannot separate candidates cannot support a policy comparison;
* interpolation error is measured by re-sampling the measured grid at stride 2 and comparing the
  linearly interpolated value with the fully measured value at every skipped depth;
* no extrapolation: `APCLatencyCurve.at()` rejects any depth outside `[0, total_blocks]`, and the
  loader refuses chain lengths that were never calibrated.

### 5.3 Expected accuracy properties

Not measured. The open questions this protocol is designed to answer are: (i) whether Prefill
latency differences between adjacent depths exceed measurement noise on an RTX 4070 at these
sizes, and (ii) whether linear interpolation between integer block depths is adequate. If (i)
fails, the honest outcome is "the cost model cannot discriminate" — which would itself be a
relevant negative result about the policy's deployability, and the preflight is built to surface
it rather than to hide it.

## 6. Decision-level observations

Status: no Native decision was recorded (`IMPLEMENTED_UNEXECUTED`). What the instrumentation will
report per decision, and why each field exists:

| Field | Purpose |
| --- | --- |
| `required_blocks`, `legal_candidate_count` | whether a choice existed at all |
| `baseline_victim_block_ids` / `selected_victim_block_ids` | exact A0 order versus proposed order |
| `victim_set_differs` | set-level divergence, separate from ordering |
| `reason`, `reason_category` | exact fallback reason plus a fixed category (`NO_PRESSURE`, `NO_LEGAL_CHOICE`, `CONTRACT_VIOLATION`, `EVIDENCE_INCOMPLETE`, `EVIDENCE_INVALID`, `CALIBRATION_INCOMPLETE`, `SEARCH_BUDGET`, `NO_PROXY_GAIN`, `SELECTED`, `UNCLASSIFIED`) |
| `evidence_missing_reason_counts` | per-candidate evidence gaps, e.g. `content_unverified`, `profile_missing`, `recency_missing` |
| `selector_overhead_seconds` | wall-clock selector cost measured per decision |
| `predicted_incremental_prefill_gain_ms` | proxy loss difference under the measured curve |
| `ranked_candidate_count`, `exact_search` | exhaustive versus greedy search |
| `ASYM_APC_PHYSICAL_OUTCOME` | `EXACT_PHYSICAL_MATCH` / `PARTIAL_PHYSICAL_MATCH` / `PROPOSAL_AND_PHYSICAL_DISJOINT` / `NO_PHYSICAL_EVICTION_OBSERVED` / `EVICTION_WITHOUT_SELECTION` / `NO_PRESSURE`, with agreement ratio and BlockPool free-block delta |
| `ASYM_APC_NEXT_REQUEST_REUSE_CHECK` | expected versus observed reusable prefix depth on the next request |

`predicted_incremental_prefill_gain_ms` is a proxy-loss difference under a cost curve; it is
explicitly **not** an expected recomputed-token count, and the event payload says so.

The investigation rule from the task is encoded in the reporting: if the two arms produce
identical physical selections, the first places to look are `legal_candidate_count`, evidence
coverage and tier restrictions; if they produce different selections with no reuse or latency
change, the cost model and workload assumptions are the suspect.

## 7. Quantitative comparison against Native LRU and A0

Status: `INCONCLUSIVE` — no run exists. The comparison table this report would contain is
produced by `kvopt.profiling.asym_apc_report`, which emits, per arm:

`mean_reusable_prefix_tokens`, `prefix_hit_rate`, `recomputed_prefill_tokens`,
`request_span_seconds_p50`, `decision_count`, `victim_set_divergence_rate`,
`physical_exact_match_rate` — plus paired deltas for `ASYM_APC − A0` and `A0 − NATIVE_LRU`.

Two deliberate constraints in that reporter:

* `prefix_hit_rate` and `recomputed_prefill_tokens` are computed only when prompt token counts are
  derivable from the frozen input (`prefix_segments_by_request`); otherwise they are `None` with
  `prompt_token_count_unavailable:<source>`. A hit rate computed against a guessed denominator
  would be worse than no number.
* `request_span_seconds_*` is labelled a recorded first-to-last event span per request in one
  clock domain, not a vendor-reported TTFT. The environment records no vendor TTFT field, and
  labelling a span as TTFT would overstate the evidence.

The earlier synthetic result (~75 % fewer recomputed tokens, 1 600 synthetic groups) is **not**
carried into this section as an estimate. Its free queue was depth-ordered, which the task
correctly identifies as potentially biased in favour of a depth-sensitive selector.

## 8. Statistical variability

Status: not estimable without runs. The design fixes what will be computed:

* repeated runs per cell (2 in `screening`, 3 in `full`) with distinct seeds recorded per repeat;
* per-metric paired delta: median, mean, sample standard deviation, min, max, and the number of
  paired units contributing;
* better/tie/worse counts on recomputed tokens, so a mean improvement dominated by one case is
  visible;
* decision-level overhead reported separately from end-to-end latency, because a selector can
  change physical victims without changing user-visible latency.

With 8 paired units in `screening` and 54 in `full`, no p-value is planned: the claim will be
limited to sign consistency and spread. Any statement stronger than that would need many more
independent workload groups, which is why Phase 3 used a Workload Group as the statistical unit.

## 9. Failure and fallback analysis

### 9.1 Enumerated fallback paths (implemented)

Non-selected outcomes are classified by exact reason, and each one is a *safe* outcome: the exact
A0 victim tuple is returned, not a partial or invented set.

| Category | Representative reasons | Behaviour |
| --- | --- | --- |
| `NO_PRESSURE` | `no_pressure` | nothing to evict |
| `NO_LEGAL_CHOICE` | `no_choice` | every legal block is needed |
| `CONTRACT_VIOLATION` | `duplicate_eligibility`, `nonzero_refcount`, `tier_order_mismatch`, `request_contract_mismatch`, `selection_contract_failed` | refuse to re-order |
| `EVIDENCE_INCOMPLETE` | `content_unverified`, `depth_collision`, `recency_missing`, `chain_length_mismatch`, `no_verified_alternative`, `insufficient_verified_candidates`, `baseline_unverified` | unknown candidates are never promoted |
| `EVIDENCE_INVALID` | `invalid_native_evidence:<Exception>` | registry contract failure |
| `CALIBRATION_INCOMPLETE` | `profile_missing` | no curve for that resident length |
| `SEARCH_BUDGET` | `search_budget_exceeded` | bounded search, A0 preserved |
| `NO_PROXY_GAIN` | `baseline_no_verified_proxy_gain` | proposed set is not better under the curve |
| `UNCLASSIFIED` | any new reason | counted, never silently mapped |

### 9.2 Environment-level failures encountered

| Failure | Cause | Resolution |
| --- | --- | --- |
| `nvidia-smi` absent, no `/dev/nvidia*` | sandbox has no GPU | recorded as the experiment blocker (§2.2) |
| `import vllm` fails | vLLM not installed and not installable without CUDA target | recorded; CPU tests avoid importing vLLM |
| `git clone` from github.com stalled | egress path to github.com slow/blocked | repository obtained through a read-only mirror; commit history preserved |
| 3 pre-existing test failures | missing W1-C artifacts, subprocess `PYTHONPATH` | unchanged from `main`; documented, not "fixed" by weakening tests |
| A test double lacked the new evidence forwarder | strict call broke an unrelated existing test | forwarder made optional (`getattr` + `callable`), because it is evidence-only and changes no decision |

## 10. Limitations and final feasibility conclusion

### 10.1 Honest status summary

| Claim | Status |
| --- | --- |
| `ASYM_APC` is selectable on the real Native runner path, gated and default-off | `IMPLEMENTED_UNEXECUTED` for GPU; gating `CPU_TESTED` |
| Decision-level audit, fallback classification and overhead measurement | `IMPLEMENTED_UNEXECUTED` for GPU; `CPU_TESTED` |
| Proposed victims are reconciled against physically evicted Block IDs | `IMPLEMENTED_UNEXECUTED` for GPU; `CPU_TESTED` |
| Next request's reusable prefix depth is verified against expectation | `IMPLEMENTED_UNEXECUTED` for GPU; `CPU_TESTED` |
| GPU Prefill calibration protocol exists and refuses unverifiable curves | `IMPLEMENTED_UNEXECUTED`; aggregation/validation `CPU_TESTED` |
| Three-arm paired campaign is fully specified and materializable | `IMPLEMENTED_UNEXECUTED`; materialization `CPU_TESTED` |
| AsymAPC beats A0 on recomputed tokens / latency / throughput | `INCONCLUSIVE` — no evidence |
| AsymAPC is worse than A0 | `INCONCLUSIVE` — no evidence |
| AsymAPC equals A0 | `INCONCLUSIVE` — no evidence |
| The earlier 75 % synthetic reduction transfers to real serving | rejected as evidence; the simulator's free queue may favour the selector |

### 10.2 Limitations

1. **No Native GPU measurement exists.** Every performance section is empty by necessity, not by
   choice. Nothing in this report should be read as a performance result.
2. **The calibration profile does not exist.** Until it does, the arm cannot start — by design.
   A reader who sees `ASYM_APC` refused at startup is seeing the intended fail-closed behaviour.
3. **Workloads are hand-built from token segments**, not replayed from archived Agent traces. The
   representative/stress split is declared in the design so a later trace-based run can be added
   without relabelling these cases as representative.
4. **`screening` is underpowered** for anything beyond sign consistency.
5. **`NATIVE_LRU` uses the Continuum runtime with `N0` release and `RetentionMode.NATIVE`**; it is
   the project's Native baseline path, not an untouched third-party vLLM build. This matches how
   Phase 3 defined its baselines and must be stated in any comparison.
6. **Latency evidence is a recorded event span**, not vendor TTFT; a future run may extend the
   observation adapter to record a vendor per-request Prefill timestamp, which the capability
   report currently lists as `UNAVAILABLE`.

### 10.3 Final feasibility conclusion

The engineering question this branch could answer — *can the experimental selector be placed on
the real Native execution path with verifiable physical-effect and audit evidence, without
touching any frozen baseline?* — is answered **yes**, at the implementation and CPU-verification
level, with the gate, the calibration contract, the physical reconciliation and the paired
campaign all in place and tested.

The research question — *does it actually help?* — remains **open and unmeasured**. The honest
current position is:

> The policy is deployable as an opt-in Native arm whose decisions are auditable and whose
> physical effect is checkable. Whether it improves cached-prefix reuse, recomputed Prefill tokens,
> TTFT or throughput over Continuum + Native LRU on real hardware is unknown, and the earlier
> synthetic advantage must not be cited as evidence either way.

The next action is mechanical: run §1.5 on a CUDA host with an RTX 4070-class device, then read
the calibration `resolution` block before interpreting any paired result. A negative or
non-discriminative outcome is a legitimate and useful finding for this line of work.
