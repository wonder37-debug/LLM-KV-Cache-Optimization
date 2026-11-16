"""Turn campaign artifacts into analysis tables.

Every table is derived from the per-cell ``summary.json`` and per-decision
JSONL records written by ``run_campaign.py``; nothing is hand-typed into the
report. Ratios are emitted as empty cells, never as zero, when the denominator
is undefined.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ARM_ORDER = ("native_lru", "a0", "depth_only", "recency_only", "asym_apc")


def _load_cells(root: Path) -> list[dict]:
    cells = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["_dir"] = summary_path.parent
        payload["_campaign"] = root.name
        cells.append(payload)
    return cells


def _decision_rows(cell: dict) -> list[dict]:
    rows = []
    for path in sorted(cell["_dir"].glob("decisions-*.jsonl")):
        arm = path.name[len("decisions-") : -len(".jsonl")]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "cell": cell["_dir"].name,
                    "family": cell["workload"]["family_name"],
                    "capacity": cell["workload"]["num_blocks"],
                    "seed": cell["workload"]["seed"],
                    "arm": arm,
                    "observed_chains": (record.get("observed_chains") or 0),
                    "legal_candidate_count": record.get("legal_candidate_count") or 0,
                    "ranked_candidate_count": record.get("ranked_candidate_count") or 0,
                    "required_blocks": record.get("required_blocks") or 0,
                    "exact_search": bool(record.get("exact_search")),
                    "reason": record.get("reason"),
                    "victim_set_differs": bool(record.get("victim_set_differs")),
                    "overhead_ns": record.get("selector_overhead_ns") or 0,
                    "greedy_ops": (record.get("legal_candidate_count") or 0)
                    * (record.get("required_blocks") or 0),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, help="campaign output dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    for root in args.campaign:
        cells.extend(_load_cells(Path(root)))

    # ---- per-cell, per-arm metrics ----------------------------------------
    columns = [
        "campaign",
        "campaign_cell",
        "family",
        "capacity",
        "seed",
        "geometry_signature",
        "arm",
        "retains",
        "cost_model",
        "weighting",
        "total_prompt_tokens",
        "total_recomputed_tokens",
        "mean_recomputed_tokens",
        "mean_prefix_hit_rate",
        "physically_evicted_blocks",
        "pressure_events",
        "events_with_retention_entries",
        "events_with_protected_entries",
        "decisions",
        "victim_set_divergence_count",
        "exact_search_decisions",
        "greedy_search_decisions",
        "evidence_missing_total",
        "selector_overhead_ns_p50",
        "selector_overhead_ns_p90",
        "selector_overhead_ns_p95",
        "selector_overhead_ns_max",
        "selector_overhead_ms_total",
        "recomputed_delta_vs_a0",
        "recomputed_rel_change_vs_a0",
        "hit_rate_delta_vs_a0",
        "modelled_prefill_ms_total",
        "wall_seconds",
    ]
    rows = []
    for cell in cells:
        family = cell["workload"]["family_name"]
        capacity = cell["workload"]["num_blocks"]
        seed = cell["workload"]["seed"]
        comparisons = cell.get("paired", {}).get("comparisons", {})
        for arm, summary in cell["summaries"].items():
            spec = summary.get("arm_spec", {})
            comparison = comparisons.get(arm, {})
            rows.append(
                {
                    "campaign": cell["_campaign"],
                    "campaign_cell": cell["_dir"].name,
                    "family": family,
                    "capacity": capacity,
                    "seed": seed,
                    "geometry_signature": cell["workload"]["geometry_signature"],
                    "arm": arm,
                    "retains": spec.get("retains"),
                    "cost_model": spec.get("cost_model"),
                    "weighting": spec.get("weighting"),
                    "total_prompt_tokens": summary["total_prompt_tokens"],
                    "total_recomputed_tokens": summary["total_recomputed_tokens"],
                    "mean_recomputed_tokens": round(summary["mean_recomputed_tokens"], 4),
                    "mean_prefix_hit_rate": round(summary["mean_prefix_hit_rate"], 6),
                    "physically_evicted_blocks": summary["physically_evicted_blocks"],
                    "pressure_events": summary["pressure_events"],
                    "events_with_retention_entries": summary[
                        "pressure_events_with_retention_entries"
                    ],
                    "events_with_protected_entries": summary[
                        "pressure_events_with_protected_entries"
                    ],
                    "decisions": summary["decisions"],
                    "victim_set_divergence_count": summary["victim_set_divergence_count"],
                    "exact_search_decisions": summary["exact_search_decisions"],
                    "greedy_search_decisions": summary["greedy_search_decisions"],
                    "evidence_missing_total": summary["evidence_missing_total"],
                    "selector_overhead_ns_p50": summary["selector_overhead_ns_p50"],
                    "selector_overhead_ns_p90": summary["selector_overhead_ns_p90"],
                    "selector_overhead_ns_p95": summary["selector_overhead_ns_p95"],
                    "selector_overhead_ns_max": summary["selector_overhead_ns_max"],
                    "selector_overhead_ms_total": summary["selector_overhead_ms_total"],
                    "recomputed_delta_vs_a0": comparison.get("recomputed_token_delta"),
                    "recomputed_rel_change_vs_a0": comparison.get(
                        "recomputed_token_relative_change"
                    ),
                    "hit_rate_delta_vs_a0": comparison.get("hit_rate_delta"),
                    "modelled_prefill_ms_total": summary["modelled_prefill_ms_total"],
                    "wall_seconds": round(summary["wall_seconds"], 4),
                }
            )
    rows.sort(key=lambda r: (r["campaign"], r["family"], r["capacity"], r["seed"],
                             ARM_ORDER.index(r["arm"])
                             if r["arm"] in ARM_ORDER else 99))
    with (out / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # ---- family × arm aggregate ------------------------------------------
    families = sorted({row["family"] for row in rows})
    arms = [arm for arm in ARM_ORDER if any(row["arm"] == arm for row in rows)]
    aggregate = []
    for family in families:
        for arm in arms:
            group = [row for row in rows if row["family"] == family and row["arm"] == arm]
            if not group:
                continue
            deltas = [
                row["recomputed_delta_vs_a0"]
                for row in group
                if isinstance(row["recomputed_delta_vs_a0"], (int, float))
            ]
            relatives = [
                row["recomputed_rel_change_vs_a0"]
                for row in group
                if isinstance(row["recomputed_rel_change_vs_a0"], (int, float))
            ]
            aggregate.append(
                {
                    "family": family,
                    "arm": arm,
                    "cells": len(group),
                    "mean_recomputed_tokens": round(
                        statistics.fmean(row["total_recomputed_tokens"] for row in group), 2
                    ),
                    "mean_delta_vs_a0": round(statistics.fmean(deltas), 2) if deltas else "",
                    "mean_rel_change_vs_a0": (
                        round(statistics.fmean(relatives), 6) if relatives else ""
                    ),
                    "cells_better": sum(1 for value in deltas if value < 0),
                    "cells_equal": sum(1 for value in deltas if value == 0),
                    "cells_worse": sum(1 for value in deltas if value > 0),
                    "mean_overhead_ns_p50": round(
                        statistics.fmean(
                            row["selector_overhead_ns_p50"]
                            for row in group
                            if isinstance(row["selector_overhead_ns_p50"], (int, float))
                        ),
                        1,
                    )
                    if any(isinstance(row["selector_overhead_ns_p50"], (int, float)) for row in group)
                    else "",
                }
            )
    with (out / "family-arm-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    # ---- per-decision overhead records -----------------------------------
    decision_rows = []
    for cell in cells:
        decision_rows.extend(_decision_rows(cell))
    with (out / "decisions-overhead.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0].keys()))
        writer.writeheader()
        writer.writerows(decision_rows)

    # ---- overhead strata --------------------------------------------------
    strata = []
    for arm in arms:
        arm_rows = [row for row in decision_rows if row["arm"] == arm]
        if not arm_rows:
            continue
        for label, predicate in (
            ("exact", lambda r: r["exact_search"]),
            ("greedy", lambda r: not r["exact_search"]),
        ):
            group = [row["overhead_ns"] for row in arm_rows if predicate(row)]
            if not group:
                continue
            ordered = sorted(group)
            strata.append(
                {
                    "arm": arm,
                    "stratum": label,
                    "decisions": len(ordered),
                    "p50_ns": ordered[len(ordered) // 2],
                    "p90_ns": ordered[int(len(ordered) * 0.9)],
                    "max_ns": ordered[-1],
                    "mean_ns": round(statistics.fmean(ordered), 1),
                }
            )
    with (out / "overhead-strata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strata[0].keys()))
        writer.writeheader()
        writer.writerows(strata)

    # ---- overhead scaling -------------------------------------------------
    def _rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2.0 + 1.0
            for position in range(index, end + 1):
                ranks[order[position]] = average
            index = end + 1
        return ranks

    def _spearman(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
            return None
        rx, ry = _rank(xs), _rank(ys)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (
            sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
        )
        return num / den if den else None

    scaling_rows = []
    correlation_rows = []
    for arm in arms:
        arm_rows = [row for row in decision_rows if row["arm"] == arm]
        if len(arm_rows) < 3:
            continue
        candidates = [float(row["legal_candidate_count"]) for row in arm_rows]
        required = [float(row["required_blocks"]) for row in arm_rows]
        chains = [float(row["observed_chains"]) for row in arm_rows]
        overhead = [float(row["overhead_ns"]) for row in arm_rows]
        correlation_rows.append(
            {
                "arm": arm,
                "decisions": len(arm_rows),
                "spearman_candidates": _spearman(candidates, overhead),
                "spearman_required": _spearman(required, overhead),
                "spearman_chains": _spearman(chains, overhead),
                "median_ns": statistics.median(overhead),
            }
        )
        for label, buckets in (
            ("legal_candidate_count", [(0, 32), (33, 64), (65, 96), (97, 128), (129, 192), (193, 100000)]),
            ("required_blocks", [(0, 8), (9, 16), (17, 24), (25, 32), (33, 1000)]),
            ("observed_chains", [(0, 4), (5, 8), (9, 12), (13, 100)]),
        ):
            for low, high in buckets:
                group = [
                    row["overhead_ns"]
                    for row in arm_rows
                    if low <= row[label] <= high
                ]
                if not group:
                    continue
                ordered = sorted(group)
                scaling_rows.append(
                    {
                        "arm": arm,
                        "dimension": label,
                        "bucket_low": low,
                        "bucket_high": high if high < 100000 else "",
                        "decisions": len(ordered),
                        "p50_ns": ordered[len(ordered) // 2],
                        "p90_ns": ordered[int(len(ordered) * 0.9)],
                        "max_ns": ordered[-1],
                    }
                )

    if scaling_rows:
        with (out / "overhead-scaling.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scaling_rows)
    if correlation_rows:
        with (out / "overhead-correlations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(correlation_rows)

    print(f"cells.csv                {len(rows)} rows")
    print(f"family-arm-summary.csv   {len(aggregate)} rows")
    print(f"decisions-overhead.csv   {len(decision_rows)} rows")
    print(f"overhead-strata.csv      {len(strata)} rows")
    print(f"overhead-scaling.csv     {len(scaling_rows)} rows")
    print(f"overhead-correlations.csv {len(correlation_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
