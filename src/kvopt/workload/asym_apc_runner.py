"""Execute the materialized AsymAPC Native evaluation campaign run by run.

Each run is a separate process-level execution with an empty Native cache, so a
paired comparison never inherits a previous run's resident prefixes. A failed
run is recorded and does not abort the remainder of the campaign, but a failed
run is never counted as a completed observation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .phase2d_e2_runner import _run_phase2d_stage

EXPECTED_STAGE = "ASYM_APC_NATIVE_EVALUATION"
STAGE_FIELD = "asym_apc_stage"
CAMPAIGN_SUMMARY_SCHEMA = "asym_apc.native_evaluation_campaign.v1"


def _load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("manifest must contain an object")
    if document.get("schema_version") != "asym_apc.native_evaluation_materialization.v1":
        raise ValueError("unsupported materialization manifest")
    runs = document.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest contains no runs")
    return document


def run_single(config_path: Path, *, output_root: Path, run_id: str) -> Path:
    """Execute one materialized run through the shared Native stage runner."""
    return _run_phase2d_stage(
        config_path,
        output_root=output_root,
        run_id=run_id,
        expected_stage=EXPECTED_STAGE,
        stage_field=STAGE_FIELD,
        run_kind="EXPERIMENTAL_ASYM_APC_NATIVE_EVALUATION",
        schema_namespace="asym_apc",
    )


def run_campaign(
    *,
    manifest_path: Path,
    config_root: Path,
    output_root: Path,
    run_ids: Sequence[str] | None = None,
    stop_on_failure: bool = False,
) -> dict[str, object]:
    """Execute every selected run and return the campaign-level outcome."""
    manifest = _load_manifest(manifest_path)
    selected = {run["run_id"] for run in manifest["runs"]}
    if run_ids is not None:
        unknown = set(run_ids) - selected
        if unknown:
            raise ValueError(f"unknown run IDs requested: {sorted(unknown)}")
        selected = set(run_ids)
    output_root.mkdir(parents=True, exist_ok=False)
    completed: list[str] = []
    failed: list[dict[str, object]] = []
    for run in manifest["runs"]:
        run_id = str(run["run_id"])
        if run_id not in selected:
            continue
        config_path = config_root / str(run["config"])
        try:
            output = run_single(config_path, output_root=output_root, run_id=run_id)
            run_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - preserve arbitrary runtime failure
            failed.append({"run_id": run_id, "reason": f"{type(error).__name__}: {error}"})
            if stop_on_failure:
                break
            continue
        if run_manifest.get("execution_status") != "success":
            failed.append(
                {
                    "run_id": run_id,
                    "reason": str(run_manifest.get("failure_reason")),
                }
            )
            if stop_on_failure:
                break
            continue
        completed.append(run_id)

    summary = {
        "schema_version": CAMPAIGN_SUMMARY_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha": manifest.get("design_sha256"),
        "profile_sha256": manifest.get("profile_sha256"),
        "subset": manifest.get("subset"),
        "planned_run_count": len(selected),
        "completed_run_count": len(completed),
        "failed_run_count": len(failed),
        "completed_run_ids": completed,
        "failures": failed,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "interpretation": (
            "completed runs are Native GPU executions; performance interpretation "
            "requires the paired analysis report and is not implied by completion"
        ),
    }
    (output_root / "campaign-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run_campaign(
        manifest_path=arguments.manifest,
        config_root=arguments.config_root,
        output_root=arguments.output_root,
        run_ids=arguments.run_id,
        stop_on_failure=arguments.stop_on_failure,
    )
    print(
        f"campaign: {summary['completed_run_count']}/{summary['planned_run_count']} "
        f"runs completed, {summary['failed_run_count']} failed"
    )
    return 0 if summary["failed_run_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
