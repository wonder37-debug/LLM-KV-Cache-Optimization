"""Materialize the experimental A/B/C Native AsymAPC evaluation campaign.

Three arms share identical inputs, cache capacity, tool-call timing, arrivals and
runtime settings:

``NATIVE_LRU``
    Standard vLLM cache management, no Continuum intervention.
``A0``
    Continuum Entry Release with Native LRU physical block selection.
``ASYM_APC``
    The same Continuum Entry Release with the AsymCache-inspired selector.

Every run gets its own config (and therefore its own process and empty cache).
The materializer refuses to emit a run without a hash-bound GPU calibration
profile, because an uncalibrated cost curve would silently degrade the policy
into A0 while still being labelled ``ASYM_APC``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DESIGN_SCHEMA = "asym_apc.native_evaluation_design.v1"
MATERIALIZATION_SCHEMA = "asym_apc.native_evaluation_materialization.v1"
CAMPAIGN_KIND = "asym_apc_native_evaluation"
# Reuse the generator that the existing Phase 2C Native backend accepts.
from kvopt.workload.phase2c_native_pilot import PILOT_PREFIX_GENERATOR_VERSION

PREFIX_GENERATOR_VERSION = PILOT_PREFIX_GENERATOR_VERSION
ARMS = ("NATIVE_LRU", "A0", "ASYM_APC")


class MaterializationError(ValueError):
    """Raised when the design or its bindings cannot produce a valid campaign."""


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterializationError(f"{field_name} must be an object")
    return value


def _sequence(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise MaterializationError(f"{field_name} must be a non-empty list")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MaterializationError(f"{field_name} must be a positive integer")
    return value


def _non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterializationError(f"{field_name} must be a number")
    result = float(value)
    if result < 0:
        raise MaterializationError(f"{field_name} must be non-negative")
    return result


def load_design(path: Path) -> dict[str, Any]:
    """Load and structurally validate the experiment design."""
    document = json.loads(path.read_text(encoding="utf-8"))
    design = _object(document, "design")
    if design.get("schema_version") != DESIGN_SCHEMA:
        raise MaterializationError("unsupported AsymAPC evaluation design schema")
    if design.get("campaign_kind") != CAMPAIGN_KIND:
        raise MaterializationError(f"design campaign_kind must be {CAMPAIGN_KIND}")
    if design.get("execution_authorized") is not True:
        raise MaterializationError("design must be explicitly authorized")
    if tuple(design.get("arms", ())) != ARMS:
        raise MaterializationError(f"design arms must be exactly {list(ARMS)}")
    if design.get("primary_comparison") != ["A0", "ASYM_APC"]:
        raise MaterializationError("primary comparison must be A0 versus ASYM_APC")
    if design.get("system_baseline") != "NATIVE_LRU":
        raise MaterializationError("system baseline must be NATIVE_LRU")
    repeats = _positive_int(design.get("runtime_repeats"), "runtime_repeats")
    seeds = _sequence(design.get("seeds"), "seeds")
    if len(seeds) != repeats or any(not isinstance(seed, int) for seed in seeds):
        raise MaterializationError("seeds must provide one integer seed per repeat")
    regimes = _sequence(design.get("pressure_regimes"), "pressure_regimes")
    if [regime.get("id") for regime in regimes] != ["low", "medium", "high"]:
        raise MaterializationError(
            "pressure regimes must be exactly low, medium and high"
        )
    cases = _sequence(design.get("cases"), "cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise MaterializationError("case IDs must be unique")
    subsets = _object(design.get("execution_subsets"), "execution_subsets")
    for subset_name in ("screening", "full"):
        subset = _object(subsets.get(subset_name), f"execution_subsets.{subset_name}")
        unknown = set(subset.get("case_ids", ())) - set(case_ids)
        if unknown:
            raise MaterializationError(
                f"subset {subset_name} references unknown cases: {sorted(unknown)}"
            )
    return design


def _case_requests(
    case: Mapping[str, Any], *, block_size: int
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, list[list[object]]], int]:
    case_id = str(case["case_id"])
    programs = _object(case.get("programs"), f"{case_id}.programs")
    requests: list[dict[str, object]] = []
    prefixes: dict[str, int] = {}
    segments_by_request: dict[str, list[list[object]]] = {}
    unique_tokens: dict[str, int] = {}
    for program_id, turns_value in programs.items():
        turns = _sequence(turns_value, f"{case_id}.programs[{program_id}]")
        first_turn_tokens = 0
        for turn_index, turn_value in enumerate(turns, start=1):
            turn = _object(turn_value, f"{case_id}.{program_id}.turn{turn_index}")
            segments = _sequence(
                turn.get("segments"), f"{case_id}.{program_id}.turn{turn_index}.segments"
            )
            normalized: list[list[object]] = []
            turn_tokens = 0
            for segment_value in segments:
                segment = _sequence(
                    segment_value, f"{case_id}.{program_id}.turn{turn_index}.segment"
                )
                if len(segment) != 2:
                    raise MaterializationError("segments must be [content_key, tokens]")
                key = str(segment[0])
                tokens = _positive_int(segment[1], "segment tokens")
                if tokens % block_size:
                    raise MaterializationError(
                        f"segment {key!r} token count must align to the block size"
                    )
                previous = unique_tokens.setdefault(key, tokens)
                if previous != tokens:
                    raise MaterializationError(
                        f"shared content key {key!r} declares inconsistent lengths"
                    )
                normalized.append([key, tokens])
                turn_tokens += tokens
            if turn_index == 1:
                first_turn_tokens = turn_tokens
            terminal = turn.get("terminal") is True
            tool_gap = turn.get("tool_gap_after_seconds")
            request_id = f"{program_id}:{turn_index}"
            requests.append(
                {
                    "program_id": program_id,
                    "request_id": request_id,
                    "turn_index": turn_index,
                    "planned_arrival_offset_seconds": _non_negative_float(
                        turn.get("arrival_offset_seconds"),
                        f"{case_id}.{program_id}.arrival_offset_seconds",
                    ),
                    "prompt": f"AsymAPC evaluation {case_id} {request_id}",
                    "prefix_prompt": "identity-is-defined-by-explicit-token-segments",
                    "is_terminal": terminal,
                    "next_tool_type": None if terminal else f"asym-apc-{case_id}",
                    "tool_gap_seconds": (
                        None
                        if terminal
                        else _non_negative_float(tool_gap, "tool_gap_after_seconds")
                    ),
                }
            )
            segments_by_request[request_id] = normalized
        prefixes[program_id] = first_turn_tokens
    requests.sort(key=lambda row: (row["planned_arrival_offset_seconds"], row["request_id"]))
    unique_protected_blocks = sum(unique_tokens.values()) // block_size
    return requests, prefixes, segments_by_request, unique_protected_blocks


def _arm_config(
    *,
    design: Mapping[str, Any],
    arm: str,
    seed: int,
    repeat: int,
    case_id: str,
    case_kind: str,
    regime_id: str,
    block_override: int,
    shortage: int,
    pressure_tokens: int,
    prefixes: Mapping[str, int],
    segments_by_request: Mapping[str, list[list[object]]],
    unique_protected_blocks: int,
    trace_name: str,
    asym_profile: Mapping[str, object],
) -> dict[str, object]:
    runtime = _object(design["runtime"], "runtime")
    model = _object(design["model"], "model")
    block_size = int(design["block_size_tokens"])
    runtime_mode = str(runtime.get("runtime_mode", "vllm-cuda-observability"))
    if runtime_mode != "vllm-cuda-observability":
        raise MaterializationError(
            "the Native GPU evaluation must run on the CUDA observation adapter"
        )
    options: dict[str, object] = {
        "execute_planned_timing": False,
        "executor_mode": "submit_collect",
        "gpu_memory_utilization": runtime["gpu_memory_utilization"],
        "enforce_eager": bool(runtime.get("enforce_eager", True)),
        "max_num_batched_tokens": int(runtime.get("max_num_batched_tokens", 8192)),
        "phase2c_native_pilot_mode": True,
        "phase2c_prefix_generator_version": PREFIX_GENERATOR_VERSION,
        "phase2g_release_policy": "N0" if arm == "NATIVE_LRU" else "A0",
        "phase2g_block_policy": "ASYM_APC" if arm == "ASYM_APC" else "A0",
        "phase2g_unique_protected_blocks": unique_protected_blocks,
        "prefix_segments_by_request": dict(segments_by_request),
        "program_prefix_tokens": dict(prefixes),
        "pressure_prompt_tokens": pressure_tokens,
        "prefill_profile_path": (
            "docs/experiments/phase1b-continuum/artifacts/"
            "prefill-reload-profile-formal-20260920T045859Z-6183d106.json"
        ),
        "capture_terminal_prefix_snapshot": True,
    }
    if arm == "ASYM_APC":
        options["asym_apc"] = {
            "enabled": True,
            **{key: value for key, value in asym_profile.items() if key != "enabled"},
        }
    return {
        "schema_version": "phase2.config.v1",
        "runtime_mode": runtime_mode,
        "backend_factory": runtime["backend_factory"],
        "backend_revision": "asym-apc-native-evaluation-v1",
        "campaign_kind": CAMPAIGN_KIND,
        "policy": "native-lru" if arm == "NATIVE_LRU" else "continuum-a0",
        "seed": seed,
        "observe": True,
        "model": {"name": model["name"], "revision": model["revision"]},
        "tokenizer": {"name": model["name"], "revision": model["revision"]},
        "generation": {
            "max_new_tokens": int(runtime["max_new_tokens"]),
            "temperature": float(runtime["temperature"]),
        },
        "batching": {"max_concurrent_requests": 1},
        "cache": {"block_size": block_size, "block_override": block_override},
        "pressure": {
            "initial_shortage_blocks": shortage,
            "required_blocks": pressure_tokens // block_size,
            "safety_ceiling": 1,
        },
        "backend_options": options,
        "trace": trace_name,
        "asym_apc_stage": {
            "stage": "ASYM_APC_NATIVE_EVALUATION",
            "case_id": case_id,
            "case_kind": case_kind,
            "pressure_regime": regime_id,
            "arm": arm,
            "repeat": repeat,
            "formal": False,
            "primary_comparison": arm in {"A0", "ASYM_APC"},
        },
    }


def materialize_campaign(
    *,
    design_path: Path,
    output_dir: Path,
    profile_path: Path,
    subset: str = "screening",
) -> Path:
    """Write traces, per-run configs and a run manifest for one execution subset."""
    design_path = design_path.resolve()
    design = load_design(design_path)
    if subset not in {"screening", "full"}:
        raise MaterializationError("subset must be screening or full")
    if not profile_path.is_file():
        raise MaterializationError(
            "a GPU-measured calibration profile is required before the ASYM_APC arm "
            f"can be materialized: {profile_path} does not exist"
        )
    profile_sha = _sha256(profile_path)
    declared = _object(design.get("asym_apc"), "asym_apc")
    asym_profile = {
        key: declared[key]
        for key in (
            "alpha_seconds",
            "beta_seconds",
            "knee_seconds",
            "tau_seconds",
            "max_exact_subsets",
            "max_greedy_operations",
            "min_proxy_gain_ms",
        )
        if key in declared
    }
    asym_profile["profile_path"] = str(profile_path)
    asym_profile["profile_sha256"] = profile_sha

    block_size = int(design["block_size_tokens"])
    subset_spec = _object(design["execution_subsets"][subset], f"subset {subset}")
    case_ids = list(subset_spec["case_ids"])
    regime_by_id = {str(regime["id"]): regime for regime in design["pressure_regimes"]}
    regime_ids = list(subset_spec["pressure_regime_ids"])
    repeats = _positive_int(subset_spec["runtime_repeats"], "subset runtime_repeats")
    seeds = list(design["seeds"])[:repeats]

    cases_by_id = {str(case["case_id"]): case for case in design["cases"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    runs: list[dict[str, object]] = []
    for case_id in case_ids:
        case = cases_by_id[case_id]
        requests, prefixes, segments_by_request, unique_protected_blocks = _case_requests(
            case, block_size=block_size
        )
        pressure_tokens = _positive_int(
            case.get("pressure_prompt_tokens"), f"{case_id}.pressure_prompt_tokens"
        )
        if pressure_tokens % block_size:
            raise MaterializationError("pressure prompt tokens must align to the block size")
        pressure_blocks = pressure_tokens // block_size
        trace = {
            "schema_version": "phase2.trace.v1",
            "trace_id": f"asym-apc-{case_id}",
            "requests": requests,
            "pressure_stages": [
                {
                    "stage_id": f"asym-apc-{case_id}-pressure",
                    "planned_arrival_offset_seconds": 1.0,
                    "prompt": f"AsymAPC evaluation {case_id} Native pressure",
                    "max_requests": 1,
                    "stop_on_forced_release": True,
                }
            ],
        }
        trace_name = f"{case_id}.trace.json"
        trace_path = output_dir / trace_name
        _write_json(trace_path, trace)

        for regime_id in regime_ids:
            regime = regime_by_id[regime_id]
            shortage = _positive_int(
                regime.get("initial_shortage_blocks"), f"regime {regime_id} shortage"
            )
            block_override = unique_protected_blocks + pressure_blocks + 1 - shortage
            if block_override < 1:
                raise MaterializationError(
                    f"case {case_id} regime {regime_id} yields non-positive cache capacity"
                )
            if pressure_blocks > block_override:
                raise MaterializationError(
                    f"case {case_id} regime {regime_id} pressure exceeds cache capacity"
                )
            for repeat_index in range(repeats):
                seed = int(seeds[repeat_index])
                for arm in ARMS:
                    run_id = f"{case_id}-{regime_id}-{arm.lower()}-repeat-{repeat_index + 1}"
                    config = _arm_config(
                        design=design,
                        arm=arm,
                        seed=seed,
                        repeat=repeat_index + 1,
                        case_id=case_id,
                        case_kind=str(case.get("kind") or "unspecified"),
                        regime_id=regime_id,
                        block_override=block_override,
                        shortage=shortage,
                        pressure_tokens=pressure_tokens,
                        prefixes=prefixes,
                        segments_by_request=segments_by_request,
                        unique_protected_blocks=unique_protected_blocks,
                        trace_name=trace_name,
                        asym_profile=asym_profile,
                    )
                    config_path = output_dir / f"{run_id}.config.json"
                    _write_json(config_path, config)
                    runs.append(
                        {
                            "run_id": run_id,
                            "case_id": case_id,
                            "case_kind": case.get("kind"),
                            "pressure_regime": regime_id,
                            "arm": arm,
                            "repeat": repeat_index + 1,
                            "seed": seed,
                            "block_override": block_override,
                            "initial_shortage_blocks": shortage,
                            "config": config_path.name,
                            "config_sha256": _sha256(config_path),
                            "trace": trace_name,
                            "trace_sha256": _sha256(trace_path),
                            "pair_key": f"{case_id}-{regime_id}-repeat-{repeat_index + 1}",
                        }
                    )

    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "scope": design["scope"],
        "formal": False,
        "campaign_kind": CAMPAIGN_KIND,
        "subset": subset,
        "design_path": str(design_path),
        "design_sha256": _sha256(design_path),
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha,
        "arms": list(ARMS),
        "case_count": len(case_ids),
        "pressure_regimes": regime_ids,
        "runtime_repeats": repeats,
        "planned_run_count": len(runs),
        "paired_units": len({run["pair_key"] for run in runs}),
        "runs": runs,
        "execution_started": False,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--subset", choices=("screening", "full"), default="screening")
    arguments = parser.parse_args(argv)
    manifest_path = materialize_campaign(
        design_path=arguments.design,
        output_dir=arguments.output_dir,
        profile_path=arguments.profile,
        subset=arguments.subset,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"materialized {manifest['planned_run_count']} runs over "
        f"{manifest['paired_units']} paired units -> {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
