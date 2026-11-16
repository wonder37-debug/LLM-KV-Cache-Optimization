"""Build the before/after decision-cost table from two campaigns of one matrix.

Both directories must contain the same cells and arms; only the arithmetic path
used to price candidates is supposed to differ. The script refuses to produce a
table when the token outcomes differ, because a cost comparison between runs that
made different decisions would be meaningless.

Outputs a CSV and a markdown table. The "saving" column is the modelled prefill
milliseconds the asym_apc arm avoided relative to a0 in the *same* cell, and
"cover" is that saving divided by the measured decision cost of the arm.

Usage
-----
    python scripts/probes/asym_apc_native/summarize_overhead_ab.py \
      --full /tmp/ab/repro-full --incremental /tmp/ab/repro-inc \
      --out-csv artifacts/analysis/overhead-ab-capacity.csv \
      --out-md artifacts/analysis/overhead-ab-capacity.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _load(root: Path) -> dict[str, dict[str, object]]:
    cells: dict[str, dict[str, object]] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        cells[summary_path.parent.name] = json.loads(summary_path.read_text(encoding="utf-8"))
    return cells


def _row(cell: str, full: dict[str, object], incremental: dict[str, object]) -> dict[str, object]:
    full_summaries = full["summaries"]  # type: ignore[index]
    inc_summaries = incremental["summaries"]  # type: ignore[index]
    full_arm = full_summaries["asym_apc"]
    inc_arm = inc_summaries["asym_apc"]
    baseline = full_summaries["a0"]
    saving_ms = (
        float(baseline["modelled_prefill_ms_total"]) - float(full_arm["modelled_prefill_ms_total"])
    )
    full_total = float(full_arm["selector_overhead_ms_total"] or 0.0)
    inc_total = float(inc_arm["selector_overhead_ms_total"] or 0.0)
    return {
        "cell": cell,
        "family": full["workload"]["family_name"],  # type: ignore[index]
        "capacity": full["workload"]["num_blocks"],  # type: ignore[index]
        "decisions": int(full_arm["decisions"]),
        "tokens_full": int(full_arm["total_recomputed_tokens"]),
        "tokens_incremental": int(inc_arm["total_recomputed_tokens"]),
        "full_p50_ms": (full_arm["selector_overhead_ns_p50"] or 0) / 1e6,
        "full_p90_ms": (full_arm["selector_overhead_ns_p90"] or 0) / 1e6,
        "full_p95_ms": (full_arm["selector_overhead_ns_p95"] or 0) / 1e6,
        "full_max_ms": (full_arm["selector_overhead_ns_max"] or 0) / 1e6,
        "full_total_ms": full_total,
        "incremental_p50_ms": (inc_arm["selector_overhead_ns_p50"] or 0) / 1e6,
        "incremental_p90_ms": (inc_arm["selector_overhead_ns_p90"] or 0) / 1e6,
        "incremental_p95_ms": (inc_arm["selector_overhead_ns_p95"] or 0) / 1e6,
        "incremental_max_ms": (inc_arm["selector_overhead_ns_max"] or 0) / 1e6,
        "incremental_total_ms": inc_total,
        "speedup_total": (full_total / inc_total) if inc_total else None,
        "modelled_saving_ms": saving_ms,
        "cover_full": (saving_ms / full_total) if full_total else None,
        "cover_incremental": (saving_ms / inc_total) if inc_total else None,
    }


def _markdown(rows: list[dict[str, object]], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        (
            "| Cell | Decisions | p50 full | p50 delta | p90 full | p90 delta | p95 full "
            "| p95 delta | max full | max delta | Total full | Total delta | Speed-up "
            "| Saving / delta-cost |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {cell} | {decisions} | {fp50:.2f} | {ip50:.2f} | {fp90:.1f} | {ip90:.1f} "
            "| {fp95:.1f} | {ip95:.1f} | {fmax:.1f} | {imax:.1f} | {ftot:.0f} | {itot:.0f} "
            "| {sp:.1f}x | {cov:.1f}x |".format(
                cell=row["cell"],
                decisions=row["decisions"],
                fp50=row["full_p50_ms"], ip50=row["incremental_p50_ms"],
                fp90=row["full_p90_ms"], ip90=row["incremental_p90_ms"],
                fp95=row["full_p95_ms"], ip95=row["incremental_p95_ms"],
                fmax=row["full_max_ms"], imax=row["incremental_max_ms"],
                ftot=row["full_total_ms"], itot=row["incremental_total_ms"],
                sp=row["speedup_total"] or 0.0,
                cov=row["cover_incremental"] or 0.0,
            )
        )
    p50_speed = statistics.median(
        (row["full_p50_ms"] / row["incremental_p50_ms"])
        for row in rows if row["incremental_p50_ms"]
    )
    total_speed = statistics.median(row["speedup_total"] for row in rows if row["speedup_total"])
    lines += [
        "",
        (
            f"Median per-decision speed-up (p50): **{p50_speed:.2f}x**; "
            f"median per-cell total speed-up: **{total_speed:.2f}x**."
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True)
    parser.add_argument("--incremental", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--title", default="Decision cost, full re-sum versus delta")
    args = parser.parse_args(argv)

    full_cells = _load(Path(args.full))
    inc_cells = _load(Path(args.incremental))
    if set(full_cells) != set(inc_cells):
        raise SystemExit("the two campaigns must cover the same cells")
    rows = [_row(cell, full_cells[cell], inc_cells[cell]) for cell in sorted(full_cells)]
    differing = [row for row in rows if row["tokens_full"] != row["tokens_incremental"]]
    if differing:
        raise SystemExit(
            "refusing to compare cost across different decisions: "
            + ", ".join(str(row["cell"]) for row in differing[:5])
        )

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows, args.title), encoding="utf-8")
    print(f"wrote {csv_path} and {md_path} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
