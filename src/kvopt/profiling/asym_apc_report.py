"""Analyze paired Native runs of the AsymAPC evaluation campaign.

The analysis is deliberately conservative:

* metrics are derived only from recorded events and frozen inputs;
* an unavailable metric is reported with its reason, never as zero;
* a different victim set is reported separately from any changed reuse or
  latency, because a different victim set does not imply a benefit;
* paired deltas are computed per ``pair_key`` (case, pressure regime, repeat),
  and variability across repeats is reported next to the point estimate.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRIMARY_ARMS = ("A0", "ASYM_APC")
ALL_ARMS = ("NATIVE_LRU", "A0", "ASYM_APC")
METRIC_KEYS = (
    "mean_reusable_prefix_tokens",
    "prefix_hit_rate",
    "recomputed_prefill_tokens",
    "request_span_seconds_p50",
    "decision_count",
    "victim_set_divergence_rate",
    "physical_exact_match_rate",
)


class ReportError(RuntimeError):
    """Raised when recorded evidence cannot support the requested analysis."""


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _median(values: Sequence[float]) -> float | None:
    return _percentile(values, 50.0)


@dataclass(slots=True)
class RunEvidence:
    """One materialized run joined to its recorded events."""

    run_id: str
    run_dir: Path
    arm: str
    case_id: str
    case_kind: str
    pressure_regime: str
    repeat: int
    pair_key: str
    execution_status: str
    events: tuple[Mapping[str, Any], ...]
    prompt_tokens_by_request: Mapping[str, int] = field(default_factory=dict)
    prompt_token_source: str = "unavailable"

    def of_type(self, event_type: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            event for event in self.events if event.get("event_type") == event_type
        )


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ReportError(f"{path} contains a non-object event")
        rows.append(value)
    return tuple(rows)


def _prompt_tokens(config: Mapping[str, Any]) -> tuple[dict[str, int], str]:
    options = config.get("backend_options")
    if not isinstance(options, Mapping):
        return {}, "config.backend_options_unavailable"
    segments = options.get("prefix_segments_by_request")
    if not isinstance(segments, Mapping):
        return {}, "config.prefix_segments_by_request_unavailable"
    totals: dict[str, int] = {}
    for request_id, rows in segments.items():
        if not isinstance(rows, Sequence):
            return {}, "config.prefix_segments_malformed"
        total = 0
        for row in rows:
            if not isinstance(row, Sequence) or len(row) != 2:
                return {}, "config.prefix_segments_malformed"
            total += int(row[1])
        totals[str(request_id)] = total
    return totals, "config.prefix_segments_by_request"


def load_run(run_dir: Path) -> RunEvidence:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise ReportError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = run_dir / "config.json"
    config: Mapping[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    stage = config.get("asym_apc_stage")
    if not isinstance(stage, Mapping):
        raise ReportError(f"{run_dir} is not an AsymAPC evaluation run")
    prompt_tokens, prompt_source = _prompt_tokens(config)
    return RunEvidence(
        run_id=str(manifest.get("run_id", run_dir.name)),
        run_dir=run_dir,
        arm=str(stage.get("arm")),
        case_id=str(stage.get("case_id")),
        case_kind=str(stage.get("case_kind") or ""),
        pressure_regime=str(stage.get("pressure_regime")),
        repeat=int(stage.get("repeat", 0)),
        pair_key=(
            f"{stage.get('case_id')}-{stage.get('pressure_regime')}-"
            f"repeat-{stage.get('repeat')}"
        ),
        execution_status=str(manifest.get("execution_status")),
        events=_read_jsonl(run_dir / "events.jsonl"),
        prompt_tokens_by_request=prompt_tokens,
        prompt_token_source=prompt_source,
    )


def _request_id(event: Mapping[str, Any]) -> str | None:
    """Read a request ID from either the JSONL (string) or object form."""
    value = event.get("request_id")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        inner = value.get("value")
        return None if not isinstance(inner, str) else inner
    return None


def _request_spans(run: RunEvidence) -> list[float]:
    """Observed wall-clock span per completed request, in one clock domain.

    This is the recorded evidence span between the first and last event that
    carries a request ID, restricted to a single clock domain. It is a request
    span, not a vendor-reported TTFT; the report labels it accordingly.
    """
    first_seen: dict[str, tuple[float, str]] = {}
    last_seen: dict[str, tuple[float, str]] = {}
    for event in run.events:
        value = _request_id(event)
        timestamp = event.get("timestamp")
        domain = event.get("clock_domain")
        if value is None or not isinstance(timestamp, (int, float)):
            continue
        if not isinstance(domain, str):
            continue
        if value not in first_seen:
            first_seen[value] = (float(timestamp), domain)
        last_seen[value] = (float(timestamp), domain)
    completed_ids = {
        request_id
        for request_id in (_request_id(event) for event in run.of_type("VLLM_REQUEST_COMPLETED"))
        if request_id is not None
    }
    spans: list[float] = []
    for request_id in completed_ids:
        start = first_seen.get(request_id)
        end = last_seen.get(request_id)
        if start is None or end is None or start[1] != end[1]:
            continue
        span = end[0] - start[0]
        if span >= 0:
            spans.append(span)
    return spans


def run_metrics(run: RunEvidence) -> dict[str, object]:
    """Derive decision-level and runtime metrics for one run."""
    snapshots = run.of_type("VLLM_PREFIX_SNAPSHOT")
    reusable: list[int] = []
    recomputed_total: int | None = 0 if run.prompt_tokens_by_request else None
    recomputed_reason: str | None = (
        None
        if run.prompt_tokens_by_request
        else f"prompt_token_count_unavailable:{run.prompt_token_source}"
    )
    for event in snapshots:
        payload = event.get("payload") or {}
        reused = payload.get("reusable_token_count")
        if not isinstance(reused, int):
            continue
        reusable.append(int(reused))
        request_id = _request_id(event)
        if recomputed_total is not None:
            total = run.prompt_tokens_by_request.get(str(request_id))
            if total is None:
                recomputed_total = None
                recomputed_reason = "prompt_token_count_missing_for_request"
            else:
                recomputed_total += max(0, total - reused)

    decisions = run.of_type("ASYM_APC_BLOCK_DECISION")
    reasons = Counter(
        str((event.get("payload") or {}).get("reason") or "unknown") for event in decisions
    )
    categories = Counter(
        str((event.get("payload") or {}).get("reason_category") or "UNCLASSIFIED")
        for event in decisions
    )
    divergent = sum(
        1
        for event in decisions
        if (event.get("payload") or {}).get("victim_set_differs") is True
    )
    overheads = [
        float((event.get("payload") or {}).get("selector_overhead_seconds"))
        for event in decisions
        if isinstance((event.get("payload") or {}).get("selector_overhead_seconds"), (int, float))
    ]
    proxy_gains = [
        float((event.get("payload") or {}).get("predicted_incremental_prefill_gain_ms"))
        for event in decisions
        if isinstance(
            (event.get("payload") or {}).get("predicted_incremental_prefill_gain_ms"),
            (int, float),
        )
    ]

    physical = run.of_type("ASYM_APC_PHYSICAL_OUTCOME")
    physical_status = Counter(
        str((event.get("payload") or {}).get("status") or "unknown") for event in physical
    )
    agreements = [
        float((event.get("payload") or {}).get("agreement_ratio"))
        for event in physical
        if isinstance((event.get("payload") or {}).get("agreement_ratio"), (int, float))
    ]
    exact = sum(
        1
        for event in physical
        if (event.get("payload") or {}).get("status") == "EXACT_PHYSICAL_MATCH"
    )

    reuse_checks = run.of_type("ASYM_APC_NEXT_REQUEST_REUSE_CHECK")
    reuse_match = sum(
        1
        for event in reuse_checks
        if (event.get("payload") or {}).get("status") == "MATCH"
    )
    reuse_mismatch = sum(
        1
        for event in reuse_checks
        if (event.get("payload") or {}).get("status") == "MISMATCH"
    )

    service_seconds = _request_spans(run)
    completions = run.of_type("VLLM_REQUEST_COMPLETED")
    total_reused = sum(reusable)
    total_prompt = (
        sum(run.prompt_tokens_by_request.values()) if run.prompt_tokens_by_request else None
    )
    return {
        "run_id": run.run_id,
        "arm": run.arm,
        "case_id": run.case_id,
        "pressure_regime": run.pressure_regime,
        "repeat": run.repeat,
        "pair_key": run.pair_key,
        "execution_status": run.execution_status,
        "request_count": len(completions),
        "prefix_snapshot_count": len(snapshots),
        "mean_reusable_prefix_tokens": (
            float(sum(reusable) / len(reusable)) if reusable else None
        ),
        "prefix_hit_rate": (
            float(total_reused / total_prompt)
            if total_prompt
            else None
        ),
        "prefix_hit_rate_reason": (
            None if total_prompt else f"prompt_token_count_unavailable:{run.prompt_token_source}"
        ),
        "recomputed_prefill_tokens": recomputed_total,
        "recomputed_prefill_tokens_reason": recomputed_reason,
        "decision_count": len(decisions),
        "decision_reason_counts": dict(sorted(reasons.items())),
        "decision_reason_category_counts": dict(sorted(categories.items())),
        "victim_set_divergence_count": divergent,
        "victim_set_divergence_rate": (divergent / len(decisions)) if decisions else None,
        "selector_overhead_seconds_median": _median(overheads),
        "selector_overhead_seconds_p90": _percentile(overheads, 90.0),
        "predicted_proxy_gain_ms_median": _median(proxy_gains),
        "physical_outcome_counts": dict(sorted(physical_status.items())),
        "physical_exact_match_count": exact,
        "physical_exact_match_rate": (exact / len(physical)) if physical else None,
        "physical_agreement_median": _median(agreements),
        "next_request_reuse_match_count": reuse_match,
        "next_request_reuse_mismatch_count": reuse_mismatch,
        "request_span_seconds_p50": _median(service_seconds),
        "request_span_seconds_p90": _percentile(service_seconds, 90.0),
        "request_span_samples": len(service_seconds),
        "request_span_note": (
            "first-to-last recorded event per completed request in one clock "
            "domain; not a vendor-reported TTFT"
        ),
    }


def _paired_delta(
    left: Mapping[str, object], right: Mapping[str, object], metric: str
) -> float | None:
    left_value = left.get(metric)
    right_value = right.get(metric)
    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
        return float(right_value) - float(left_value)
    return None


def paired_analysis(
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare arms within each paired unit and summarize variability."""
    by_pair: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in metrics:
        if row.get("execution_status") != "success":
            continue
        by_pair.setdefault(str(row["pair_key"]), {})[str(row["arm"])] = row

    comparisons: dict[str, dict[str, object]] = {}
    for label, (left_arm, right_arm) in {
        "asym_apc_minus_a0": ("A0", "ASYM_APC"),
        "a0_minus_native_lru": ("NATIVE_LRU", "A0"),
    }.items():
        deltas: dict[str, list[float]] = {metric: [] for metric in METRIC_KEYS}
        signs: Counter[str] = Counter()
        pair_count = 0
        for arms in by_pair.values():
            left = arms.get(left_arm)
            right = arms.get(right_arm)
            if left is None or right is None:
                continue
            pair_count += 1
            recompute_delta = _paired_delta(left, right, "recomputed_prefill_tokens")
            if recompute_delta is not None:
                signs[
                    "worse"
                    if recompute_delta > 0
                    else "better"
                    if recompute_delta < 0
                    else "tie"
                ] += 1
            for metric in METRIC_KEYS:
                delta = _paired_delta(left, right, metric)
                if delta is not None:
                    deltas[metric].append(delta)
        comparisons[label] = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "paired_unit_count": pair_count,
            "recomputed_token_outcome_counts": dict(sorted(signs.items())),
            "deltas": {
                metric: {
                    "sample_count": len(values),
                    "median": _median(values),
                    "mean": (sum(values) / len(values)) if values else None,
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                    "stdev": _stdev(values),
                }
                for metric, values in deltas.items()
            },
        }
    return {"comparisons": comparisons}


def _stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def summarize_runs(run_dirs: Iterable[Path]) -> dict[str, object]:
    evidences = [load_run(path) for path in run_dirs]
    if not evidences:
        raise ReportError("no run directories were provided")
    metrics = [run_metrics(run) for run in evidences]
    by_arm: dict[str, list[Mapping[str, object]]] = {}
    for row in metrics:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    arm_summary: dict[str, object] = {}
    for arm, rows in sorted(by_arm.items()):
        completed = [row for row in rows if row.get("execution_status") == "success"]
        arm_summary[arm] = {
            "run_count": len(rows),
            "completed_run_count": len(completed),
            "statistics": {
                metric: {
                    "sample_count": len(values),
                    "median": _median(values),
                    "p90": _percentile(values, 90.0),
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                }
                for metric in METRIC_KEYS
                for values in [
                    [
                        float(row[metric])
                        for row in completed
                        if isinstance(row.get(metric), (int, float))
                    ]
                ]
            },
        }
    return {
        "run_count": len(metrics),
        "completed_run_count": sum(
            1 for row in metrics if row.get("execution_status") == "success"
        ),
        "arms": arm_summary,
        "paired": paired_analysis(metrics),
        "runs": metrics,
        "claim_boundary": (
            "paired Native GPU measurements; a victim-set difference is not a "
            "performance claim, and unavailable metrics carry explicit reasons"
        ),
    }


def render_markdown(summary: Mapping[str, object], *, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(
        f"Runs analyzed: **{summary['run_count']}** "
        f"(completed {summary['completed_run_count']})."
    )
    lines.append("")
    lines.append("## Per-arm metrics")
    lines.append("")
    header = "| Metric | " + " | ".join(
        str(arm) for arm in summary["arms"]  # type: ignore[union-attr]
    ) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in summary["arms"]) + " |")  # type: ignore[union-attr]
    for metric in METRIC_KEYS:
        cells = []
        for arm in summary["arms"]:  # type: ignore[union-attr]
            stats = summary["arms"][arm]["statistics"][metric]  # type: ignore[index]
            median = stats["median"]
            cells.append("n/a" if median is None else f"{median:.4f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Paired comparisons")
    for label, comparison in summary["paired"]["comparisons"].items():  # type: ignore[union-attr,index]
        lines.append("")
        lines.append(
            f"### {label} ({comparison['right_arm']} − {comparison['left_arm']})"
        )
        lines.append("")
        lines.append(f"Paired units: **{comparison['paired_unit_count']}**.")
        outcome = comparison["recomputed_token_outcome_counts"]
        if outcome:
            lines.append(
                "Recomputed-token outcomes (right vs left): "
                + ", ".join(f"{key}={value}" for key, value in outcome.items())
                + "."
            )
        lines.append("")
        lines.append("| Metric | n | median delta | mean delta | stdev | min | max |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for metric, stats in comparison["deltas"].items():
            lines.append(
                f"| {metric} | {stats['sample_count']} | "
                f"{_fmt(stats['median'])} | {_fmt(stats['mean'])} | "
                f"{_fmt(stats['stdev'])} | {_fmt(stats['minimum'])} | "
                f"{_fmt(stats['maximum'])} |"
            )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(str(summary["claim_boundary"]))
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize AsymAPC Native runs")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="AsymAPC Native evaluation summary")
    arguments = parser.parse_args(argv)
    summary = summarize_runs(arguments.run_dir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path = arguments.output.with_suffix(".md")
    markdown_path.write_text(
        render_markdown(summary, title=arguments.title), encoding="utf-8"
    )
    print(f"summary: {arguments.output}\nmarkdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
