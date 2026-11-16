"""Compare two campaign directories that were supposed to make the same decisions.

Used to check that a change to how the selector *prices* a candidate leaves the
*decisions* untouched. The comparison is deliberately strict: not just the total
recomputed tokens, but every request row and every per-decision victim set,
reason and proxy loss must match. Only the measured CPU overhead is allowed to
differ, and it is reported separately rather than compared.

Usage
-----
    python scripts/probes/asym_apc_native/compare_campaigns.py \
      --baseline docs/.../campaigns/heldout \
      --candidate /tmp/replay-heldout \
      --out docs/.../analysis/heldout-decision-identity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUEST_FIELDS_IGNORED: tuple[str, ...] = ()
DECISION_FIELDS_IGNORED = (
    # Measured wall-clock cost is the one thing a pricing change is allowed to
    # move; everything that describes the decision must stay identical.
    "selector_overhead_ns",
    "selector_overhead_seconds",
    # The label of the arithmetic path is provenance, not a decision: comparing
    # two paths must not fail simply because each names itself.
    "loss_evaluation",
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _comparable(row: dict[str, object], ignored: tuple[str, ...]) -> dict[str, object]:
    return {k: v for k, v in row.items() if k not in ignored}


def _digest(rows: list[dict[str, object]], ignored: tuple[str, ...],
            shared: frozenset[str] | None = None) -> str:
    payload = []
    for row in rows:
        comparable = _comparable(row, ignored)
        if shared is not None:
            comparable = {k: v for k, v in comparable.items() if k in shared}
        payload.append(comparable)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compare(baseline: Path, candidate: Path) -> dict[str, object]:
    cells = sorted(p.name for p in baseline.iterdir() if p.is_dir())
    candidate_cells = sorted(p.name for p in candidate.iterdir() if p.is_dir())
    report: dict[str, object] = {
        "schema": "kvopt.asym_apc.decision_identity.v1",
        "baseline_dir": str(baseline),
        "candidate_dir": str(candidate),
        "cells_in_baseline": len(cells),
        "cells_in_candidate": len(candidate_cells),
        "missing_cells": sorted(set(cells) - set(candidate_cells)),
        "extra_cells": sorted(set(candidate_cells) - set(cells)),
        "cells": [],
    }
    mismatches: list[dict[str, object]] = []
    overhead: dict[str, dict[str, list[float]]] = {}
    decisions_compared = 0
    requests_compared = 0
    for cell in cells:
        if cell not in candidate_cells:
            continue
        base_dir, cand_dir = baseline / cell, candidate / cell
        arms = sorted(
            p.stem.removeprefix("decisions-")
            for p in base_dir.glob("decisions-*.jsonl")
        )
        cell_entry: dict[str, object] = {
            "cell": cell,
            "arms": {},
            "summary_tokens_identical": False,
        }
        base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
        cand_summary = json.loads((cand_dir / "summary.json").read_text(encoding="utf-8"))
        base_tokens = {arm: s["total_recomputed_tokens"]
                       for arm, s in base_summary["summaries"].items()}
        cand_tokens = {arm: s["total_recomputed_tokens"]
                       for arm, s in cand_summary["summaries"].items()}
        cell_entry["summary_tokens_identical"] = base_tokens == cand_tokens
        cell_entry["baseline_tokens"] = base_tokens
        cell_entry["candidate_tokens"] = cand_tokens
        if base_tokens != cand_tokens:
            mismatches.append({"cell": cell, "kind": "summary_tokens",
                               "baseline": base_tokens, "candidate": cand_tokens})
        for arm in arms:
            base_decisions = _load_jsonl(base_dir / f"decisions-{arm}.jsonl")
            cand_decisions = _load_jsonl(cand_dir / f"decisions-{arm}.jsonl")
            base_requests = _load_jsonl(base_dir / f"requests-{arm}.jsonl")
            cand_requests = _load_jsonl(cand_dir / f"requests-{arm}.jsonl")
            decisions_compared += len(base_decisions)
            requests_compared += len(base_requests)
            base_keys = {k for row in base_decisions for k in row} - set(DECISION_FIELDS_IGNORED)
            cand_keys = {k for row in cand_decisions for k in row} - set(DECISION_FIELDS_IGNORED)
            shared = frozenset(base_keys & cand_keys)
            identical = _digest(base_decisions, DECISION_FIELDS_IGNORED, shared) == _digest(
                cand_decisions, DECISION_FIELDS_IGNORED, shared
            )
            request_keys = {k for row in base_requests for k in row} - set(REQUEST_FIELDS_IGNORED)
            requests_identical = _digest(
                base_requests, REQUEST_FIELDS_IGNORED, frozenset(request_keys)
            ) == _digest(cand_requests, REQUEST_FIELDS_IGNORED, frozenset(request_keys))
            cell_entry["arms"][arm] = {
                "decisions": len(base_decisions),
                "decisions_identical": identical,
                "requests": len(base_requests),
                "requests_identical": requests_identical,
                # Fields that exist on one side only are reported, not scored:
                # added provenance must not be confused with a changed decision.
                "decision_fields_only_in_baseline": sorted(base_keys - cand_keys),
                "decision_fields_only_in_candidate": sorted(cand_keys - base_keys),
            }
            if not identical or not requests_identical:
                mismatches.append({"cell": cell, "arm": arm, "kind": "row_content"})
            base_ns = [d.get("selector_overhead_ns") for d in base_decisions]
            cand_ns = [d.get("selector_overhead_ns") for d in cand_decisions]
            for label, values in (("baseline", base_ns), ("candidate", cand_ns)):
                bucket = overhead.setdefault(label, {}).setdefault(cell, [])
                bucket.extend(float(v) for v in values if isinstance(v, (int, float)))
        report["cells"].append(cell_entry)  # type: ignore[union-attr]
    report["requests_compared"] = requests_compared
    report["decisions_compared"] = decisions_compared
    report["mismatches"] = mismatches
    report["overhead"] = overhead
    report["identical"] = not mismatches and not report["missing_cells"]
    return report


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(percent / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def overhead_table(report: dict[str, object]) -> list[dict[str, object]]:
    """Per-cell before/after decision cost, in milliseconds."""
    rows: list[dict[str, object]] = []
    for cell_entry in report["cells"]:  # type: ignore[union-attr]
        cell = str(cell_entry["cell"])
        base = report["overhead"]["baseline"].get(cell, [])  # type: ignore[index]
        cand = report["overhead"]["candidate"].get(cell, [])  # type: ignore[index]
        rows.append(
            {
                "cell": cell,
                "decisions": len(base),
                "baseline_p50_ms": _percentile(base, 50) / 1e6 if base else None,
                "candidate_p50_ms": _percentile(cand, 50) / 1e6 if cand else None,
                "baseline_p90_ms": _percentile(base, 90) / 1e6 if base else None,
                "candidate_p90_ms": _percentile(cand, 90) / 1e6 if cand else None,
                "baseline_total_ms": sum(base) / 1e6 if base else None,
                "candidate_total_ms": sum(cand) / 1e6 if cand else None,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    report = compare(Path(args.baseline), Path(args.candidate))
    # The overhead series lives outside the identity claim, so keep it in a
    # separate block instead of mixing measured cost into a correctness result.
    payload = {k: v for k, v in report.items() if k != "overhead"}
    payload["overhead_by_cell_ms"] = overhead_table(report)
    payload["overhead_raw_ns"] = report["overhead"]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    print(
        f"cells={payload['cells_in_baseline']} requests={payload['requests_compared']} "
        f"decisions={payload['decisions_compared']} identical={payload['identical']} "
        f"mismatches={len(payload['mismatches'])}"
    )
    return 0 if payload["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
