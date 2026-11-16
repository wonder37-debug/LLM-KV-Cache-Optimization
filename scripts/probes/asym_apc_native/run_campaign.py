"""Execute a matrix of real-BlockPool cells and write checksummed artifacts.

One cell = (workload family, capacity, seed). Every cell runs the requested arms
against a fresh ``BlockPool`` in the same process, so cells cannot share cache
state, and every arm inside a cell receives the identical turn list.

The driver exists so that the reported numbers are produced by one reproducible
command instead of by hand-assembled loops, and so that every artifact directory
carries its own SHA-256 manifest.

Usage
-----
    python scripts/probes/asym_apc_native/run_campaign.py \
      --profile docs/.../cpu-device-prefill-v1.json \
      --profile-sha256 6266f3fe... \
      --allow-source cpu_device_measured \
      --families legacy_round_robin,agent_random_arrival \
      --capacities 240,288 --seeds 20260101 \
      --out-dir artifacts/campaign
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[2]


def _load_probe():
    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "real_blockpool_eval", PROBE_DIR / "real_blockpool_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["real_blockpool_eval"] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--allow-source", action="append", default=None)
    parser.add_argument("--families", required=True)
    parser.add_argument("--capacities", default="")
    parser.add_argument(
        "--capacity-fractions",
        default="",
        help="per-family capacity as a fraction of the family working set, e.g. 0.85,0.95",
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--arms", default="native_lru,a0,asym_apc")
    parser.add_argument("--ttl", type=float, default=30.0)
    parser.add_argument("--regime", default="")
    parser.add_argument("--min-proxy-gain-ms", type=float, default=0.0)
    parser.add_argument("--decay-alpha", type=float, default=60.0)
    parser.add_argument("--decay-beta", type=float, default=12.0)
    parser.add_argument("--decay-knee", type=float, default=60.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    probe = _load_probe()
    workload_families = sys.modules["workload_families"]

    profile_path = Path(args.profile).resolve()
    actual_sha = sha256_file(profile_path)
    if actual_sha != args.profile_sha256:
        raise SystemExit(f"profile SHA-256 mismatch: {args.profile_sha256} != {actual_sha}")
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    measurement_source = str(document.get("measurement_source"))
    allowed = tuple(args.allow_source or [measurement_source])

    families = [name.strip() for name in args.families.split(",") if name.strip()]
    capacities = [int(v) for v in args.capacities.split(",") if v.strip()]
    fractions = [float(v) for v in args.capacity_fractions.split(",") if v.strip()]
    if not capacities and not fractions:
        raise SystemExit("give --capacities and/or --capacity-fractions")
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, object]] = []
    for family_name in families:
        if family_name not in workload_families.FAMILIES:
            raise SystemExit(f"unknown family: {family_name}")
        spec = workload_families.FAMILIES[family_name]
        for seed in seeds:
            turns = workload_families.build_turns(spec, seed)
            geometry = workload_families.geometry_signature(turns)
            working_set = workload_families.working_set_blocks(turns, spec.shared_prefix_blocks)
            planned: list[tuple[int, float | None]] = [(c, None) for c in capacities]
            for fraction in fractions:
                planned.append((max(32, round(fraction * working_set)), fraction))
            for capacity, fraction in planned:
                suffix = f"-f{fraction:g}" if fraction is not None else ""
                cell_dir = out_root / f"{family_name}-cap{capacity}-s{seed}{suffix}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                summaries: dict[str, dict[str, object]] = {}
                started_cell = time.perf_counter()
                for arm in arms:
                    started = time.perf_counter()
                    result = probe.run_arm(
                        arm,
                        turns,
                        num_blocks=capacity,
                        ttl_seconds=args.ttl,
                        profile_path=profile_path,
                        profile_sha=args.profile_sha256,
                        allowed_sources=allowed,
                        proxy_gain_ms=args.min_proxy_gain_ms,
                        decay_alpha=args.decay_alpha,
                        decay_beta=args.decay_beta,
                        decay_knee=args.decay_knee,
                    )
                    elapsed = time.perf_counter() - started
                    summary = probe.summarise(result, measurement_source=measurement_source)
                    summary["wall_seconds"] = elapsed
                    summary["arm_spec"] = {
                        "retains": probe.ARM_SPECS[arm].retains,
                        "cost_model": probe.ARM_SPECS[arm].cost_model,
                        "weighting": probe.ARM_SPECS[arm].weighting,
                        "note": probe.ARM_SPECS[arm].note,
                    }
                    summaries[arm] = summary
                    with (cell_dir / f"requests-{arm}.jsonl").open("w", encoding="utf-8") as handle:
                        for row in result.requests:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                    with (cell_dir / f"decisions-{arm}.jsonl").open("w", encoding="utf-8") as handle:
                        for row in result.decisions:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")

                baseline = summaries.get("a0")
                comparisons: dict[str, object] = {}
                if baseline is not None:
                    for arm, summary in summaries.items():
                        if arm == "a0":
                            continue
                        delta = (
                            summary["total_recomputed_tokens"]
                            - baseline["total_recomputed_tokens"]
                        )
                        comparisons[arm] = {
                            "recomputed_token_delta": delta,
                            "recomputed_token_relative_change": (
                                delta / baseline["total_recomputed_tokens"]
                                if baseline["total_recomputed_tokens"]
                                else None
                            ),
                            "hit_rate_delta": (
                                summary["mean_prefix_hit_rate"]
                                - baseline["mean_prefix_hit_rate"]
                            ),
                        }
                        summary["vs_a0_recomputed_delta"] = delta

                payload = {
                    "schema": "kvopt.asym_apc.real_blockpool_campaign_cell.v1",
                    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "engine": probe._engine_identity(),
                    "workload": {
                        "family": workload_families.describe(spec),
                        "family_name": family_name,
                        "geometry_signature": geometry,
                        "regime": args.regime,
                        "seed": seed,
                        "num_blocks": capacity,
                        "working_set_blocks": working_set,
                        "capacity_fraction": fraction,
                        "block_tokens": probe.BLOCK_TOKENS,
                        "ttl_seconds": args.ttl,
                        "total_requests": len(turns),
                        "programs": spec.programs,
                        "turns_per_program": list(spec.turns_per_program),
                        "shared_prefix_blocks": spec.shared_prefix_blocks,
                        "program_prefix_blocks": spec.program_prefix_blocks,
                    },
                    "arms": {
                        "executed": list(arms),
                        "specs": {
                            name: {
                                "retains": probe.ARM_SPECS[name].retains,
                                "cost_model": probe.ARM_SPECS[name].cost_model,
                                "weighting": probe.ARM_SPECS[name].weighting,
                                "note": probe.ARM_SPECS[name].note,
                            }
                            for name in arms
                        },
                    },
                    "decay": {
                        "alpha_seconds": args.decay_alpha,
                        "beta_seconds": args.decay_beta,
                        "knee_seconds": args.decay_knee,
                    },
                    "label": args.label,
                    "calibration": {
                        "profile_path": str(profile_path),
                        "profile_sha256": args.profile_sha256,
                        "measurement_source": measurement_source,
                        "accepted_sources": list(allowed),
                    },
                    "summaries": summaries,
                    "paired": {"baseline_arm": "a0", "comparisons": comparisons},
                    "scope": {
                        "real_vllm_block_pool": True,
                        "gpu_executed": False,
                        "measured_prefill_latency": False,
                        "latency_is_model_based": True,
                        "recomputed_tokens_are_measured": True,
                    },
                    "cell_wall_seconds": time.perf_counter() - started_cell,
                }
                summary_path = cell_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                checksums = {
                    path.name: sha256_file(path)
                    for path in sorted(cell_dir.iterdir())
                    if path.is_file()
                }
                (cell_dir / "SHA256SUMS.json").write_text(
                    json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                cells.append(
                    {
                        "cell": cell_dir.name,
                        "family": family_name,
                        "capacity": capacity,
                        "capacity_fraction": fraction,
                        "working_set_blocks": working_set,
                        "seed": seed,
                        "geometry_signature": geometry,
                        "a0_tokens": baseline["total_recomputed_tokens"] if baseline else None,
                        "comparisons": comparisons,
                    }
                )
                print(
                    f"{cell_dir.name:42s} "
                    + " ".join(
                        f"{arm}={summaries[arm]['total_recomputed_tokens']}"
                        for arm in arms
                    ),
                    flush=True,
                )

    manifest = {
        "schema": "kvopt.asym_apc.real_blockpool_campaign.v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": probe._engine_identity(),
        "calibration": {
            "profile_path": str(profile_path),
            "profile_sha256": args.profile_sha256,
            "measurement_source": measurement_source,
        },
        "matrix": {
            "families": families,
            "capacities": capacities,
            "seeds": seeds,
            "arms": list(arms),
            "ttl_seconds": args.ttl,
            "decay": {
                "alpha_seconds": args.decay_alpha,
                "beta_seconds": args.decay_beta,
                "knee_seconds": args.decay_knee,
            },
            "label": args.label,
        },
        "cells": cells,
    }
    (out_root / "campaign-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_root/'campaign-manifest.json'} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
