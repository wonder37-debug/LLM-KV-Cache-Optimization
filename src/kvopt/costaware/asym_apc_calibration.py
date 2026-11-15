"""Native GPU prefill-cost calibration for the contiguous-APC cost model.

``APCLatencyCurve`` needs measured Prefill latency ``F(m, r)`` for a prompt of
``m`` blocks with ``r`` reusable cached prefix blocks. This module owns the
measurement protocol, aggregation, validation and profile serialization; the
caller owns the actual GPU request execution.

The design refuses to fabricate values:

* a measurement is only accepted when the *observed* cached-prefix tokens equal
  the requested cached depth and the observed computed tokens equal the
  remaining prompt tokens;
* a depth with too few valid repetitions fails the calibration instead of
  borrowing a neighbouring value;
* a non-monotone profile fails instead of being clamped;
* interpolation error is measured, not assumed;
* prompt lengths outside the measured grid cannot be served by the resulting
  profile (``APCLatencyCurve`` itself refuses to extrapolate).

This module is CPU-testable: the measurement function is injected.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
import platform
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import kvopt
from kvopt.costaware.asym_apc_native import APCLatencyCurve
from kvopt.costaware.asym_apc_runner import PROFILE_SCHEMA

CALIBRATION_SCHEMA = "kvopt.asym_apc.prefill_calibration_evidence.v1"
MEASUREMENT_SOURCE = "native_gpu"


class CalibrationError(RuntimeError):
    """Raised when measurements cannot support an honest cost curve."""


@dataclass(frozen=True, slots=True)
class PrefillMeasurement:
    """One timed Prefill observation with its verified APC depth."""

    total_blocks: int
    requested_cached_blocks: int
    elapsed_seconds: float
    observed_cached_tokens: int | None
    observed_computed_tokens: int | None
    repetition: int
    warmup: bool = False
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.total_blocks < 1 or not 0 <= self.requested_cached_blocks <= self.total_blocks:
            raise ValueError("measurement depths are inconsistent")
        if not self.warmup:
            if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
                raise ValueError("measured prefill elapsed time must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    """Frozen measurement grid; depths are block counts, not tokens."""

    prompt_lengths_blocks: tuple[int, ...]
    block_size: int = 16
    warmup_repetitions: int = 3
    measured_repetitions: int = 7
    minimum_valid_repetitions: int = 5
    noise_multiplier: float = 3.0
    interpolation_stride: int = 2

    def __post_init__(self) -> None:
        if not self.prompt_lengths_blocks:
            raise ValueError("protocol requires at least one prompt length")
        if any(length < 2 for length in self.prompt_lengths_blocks):
            raise ValueError("prompt lengths must contain at least two blocks")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.warmup_repetitions < 0:
            raise ValueError("warmup repetitions cannot be negative")
        if self.measured_repetitions < 1:
            raise ValueError("measured repetitions must be positive")
        if self.minimum_valid_repetitions < 1:
            raise ValueError("minimum valid repetitions must be positive")
        if self.minimum_valid_repetitions > self.measured_repetitions:
            raise ValueError("minimum valid repetitions exceeds measured repetitions")
        if self.noise_multiplier <= 0:
            raise ValueError("noise multiplier must be positive")
        if self.interpolation_stride < 1:
            raise ValueError("interpolation stride must be positive")

    def depths_for(self, total_blocks: int) -> tuple[int, ...]:
        return tuple(range(total_blocks + 1))


@dataclass(frozen=True, slots=True)
class DepthSummary:
    """Aggregated F(total, depth) with variability and noise floor."""

    total_blocks: int
    cached_blocks: int
    valid_samples: int
    excluded_samples: int
    median_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float
    relative_spread: float

    def payload(self) -> dict[str, object]:
        return {
            "total_blocks": self.total_blocks,
            "cached_blocks": self.cached_blocks,
            "valid_samples": self.valid_samples,
            "excluded_samples": self.excluded_samples,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "relative_spread": self.relative_spread,
        }


@dataclass(frozen=True, slots=True)
class CurveSummary:
    """One validated curve plus the evidence needed to judge its resolution."""

    total_blocks: int
    curve: APCLatencyCurve
    depths: tuple[DepthSummary, ...]
    monotone: bool
    noise_floor_ms: float
    minimum_adjacent_gap_ms: float | None
    discriminative: bool
    interpolation_max_absolute_ms: float | None
    interpolation_max_relative: float | None
    excluded_sample_reasons: Mapping[str, int] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        return {
            "total_blocks": self.total_blocks,
            "samples": [
                {"cached_blocks": depth, "prefill_ms": self.curve.at(depth)}
                for depth in sorted({sample[0] for sample in self.curve.samples})
            ],
            "depth_summaries": [summary.payload() for summary in self.depths],
            "monotone": self.monotone,
            "noise_floor_ms": self.noise_floor_ms,
            "minimum_adjacent_gap_ms": self.minimum_adjacent_gap_ms,
            "discriminative": self.discriminative,
            "interpolation_max_absolute_ms": self.interpolation_max_absolute_ms,
            "interpolation_max_relative": self.interpolation_max_relative,
            "excluded_sample_reasons": dict(sorted(self.excluded_sample_reasons.items())),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
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


def _medians_are_monotone(values: Sequence[float], tolerance_ms: float) -> bool:
    return all(
        later <= earlier + tolerance_ms
        for earlier, later in itertools.pairwise(values)
    )


def summarize_depth(
    measurements: Sequence[PrefillMeasurement],
    *,
    total_blocks: int,
    cached_blocks: int,
) -> DepthSummary:
    """Aggregate one depth; invalid observations are excluded, then reported."""
    accepted: list[float] = []
    excluded: list[str] = []
    for measurement in measurements:
        if measurement.total_blocks != total_blocks:
            raise CalibrationError("mixed prompt lengths in one depth summary")
        if measurement.requested_cached_blocks != cached_blocks:
            raise CalibrationError("mixed cached depths in one depth summary")
        if measurement.warmup:
            continue
        if measurement.invalid_reason is not None:
            excluded.append(measurement.invalid_reason)
            continue
        accepted.append(measurement.elapsed_seconds * 1000.0)
    if not accepted:
        raise CalibrationError(
            f"no valid measurement for total_blocks={total_blocks} "
            f"cached_blocks={cached_blocks}; excluded reasons: {sorted(set(excluded))}"
        )
    median_ms = statistics.median(accepted)
    p90_ms = _percentile(accepted, 90.0)
    return DepthSummary(
        total_blocks=total_blocks,
        cached_blocks=cached_blocks,
        valid_samples=len(accepted),
        excluded_samples=len(excluded),
        median_ms=float(median_ms),
        p90_ms=float(p90_ms),
        minimum_ms=float(min(accepted)),
        maximum_ms=float(max(accepted)),
        relative_spread=(
            float((max(accepted) - min(accepted)) / median_ms) if median_ms > 0 else 0.0
        ),
    )


def build_curve(
    depth_summaries: Sequence[DepthSummary],
    *,
    protocol: CalibrationProtocol,
) -> CurveSummary:
    """Validate monotonicity and resolution, then construct ``APCLatencyCurve``."""
    if not depth_summaries:
        raise CalibrationError("curve requires depth summaries")
    total_blocks = depth_summaries[0].total_blocks
    if [summary.cached_blocks for summary in depth_summaries] != list(
        range(total_blocks + 1)
    ):
        raise CalibrationError(
            f"curve for total_blocks={total_blocks} must cover cached depths 0..{total_blocks}"
        )
    for summary in depth_summaries:
        if summary.valid_samples < protocol.minimum_valid_repetitions:
            raise CalibrationError(
                f"depth {summary.cached_blocks}/{total_blocks} has only "
                f"{summary.valid_samples} valid repetitions "
                f"(minimum {protocol.minimum_valid_repetitions})"
            )
    medians = [summary.median_ms for summary in depth_summaries]
    noise_floor = max(
        (summary.p90_ms - summary.median_ms) for summary in depth_summaries
    ) * protocol.noise_multiplier
    monotone = _medians_are_monotone(medians, noise_floor)
    if not monotone:
        violations = [
            (index, medians[index], medians[index + 1])
            for index in range(len(medians) - 1)
            if medians[index + 1] > medians[index] + noise_floor
        ]
        raise CalibrationError(
            "measured prefill cost is not non-increasing in cached depth beyond the "
            f"noise floor ({noise_floor:.4f} ms); violating (depth, earlier, later): "
            f"{violations}"
        )
    gaps = [earlier - later for earlier, later in itertools.pairwise(medians)]
    minimum_gap = min(gaps) if gaps else None
    discriminative = bool(gaps) and (minimum_gap or 0.0) > noise_floor
    curve = APCLatencyCurve(
        total_blocks,
        tuple((summary.cached_blocks, summary.median_ms) for summary in depth_summaries),
    )
    interpolation_abs, interpolation_rel = _interpolation_error(
        depth_summaries, stride=protocol.interpolation_stride
    )
    excluded: dict[str, int] = {}
    for summary in depth_summaries:
        if summary.excluded_samples:
            excluded[f"total_blocks={total_blocks}"] = excluded.get(
                f"total_blocks={total_blocks}", 0
            ) + summary.excluded_samples
    return CurveSummary(
        total_blocks=total_blocks,
        curve=curve,
        depths=tuple(depth_summaries),
        monotone=monotone,
        noise_floor_ms=float(noise_floor),
        minimum_adjacent_gap_ms=None if minimum_gap is None else float(minimum_gap),
        discriminative=discriminative,
        interpolation_max_absolute_ms=interpolation_abs,
        interpolation_max_relative=interpolation_rel,
        excluded_sample_reasons=excluded,
    )


def _interpolation_error(
    depth_summaries: Sequence[DepthSummary], *, stride: int
) -> tuple[float | None, float | None]:
    """Measure linear-interpolation error over a coarser sub-grid.

    The cost model interpolates linearly between calibrated depths. Comparing a
    stride-reduced grid against the fully measured depths quantifies that
    assumption instead of asserting it.
    """
    if stride < 2 or len(depth_summaries) < 2 * stride + 1:
        return None, None
    by_depth = {summary.cached_blocks: summary.median_ms for summary in depth_summaries}
    coarse = sorted(by_depth)[::stride]
    if coarse[-1] != max(by_depth):
        coarse.append(max(by_depth))
    max_abs = 0.0
    max_rel = 0.0
    for depth in by_depth:
        if depth in coarse:
            continue
        lower = max(value for value in coarse if value < depth)
        upper = min(value for value in coarse if value > depth)
        weight = (depth - lower) / (upper - lower)
        interpolated = by_depth[lower] + (by_depth[upper] - by_depth[lower]) * weight
        actual = by_depth[depth]
        max_abs = max(max_abs, abs(interpolated - actual))
        if actual > 0:
            max_rel = max(max_rel, abs(interpolated - actual) / actual)
    return float(max_abs), float(max_rel)


def run_calibration(
    measure: Callable[[int, int, int], PrefillMeasurement],
    protocol: CalibrationProtocol,
) -> tuple[tuple[CurveSummary, ...], tuple[PrefillMeasurement, ...]]:
    """Execute the frozen measurement grid and validate every curve."""
    raw: list[PrefillMeasurement] = []
    curves: list[CurveSummary] = []
    for total_blocks in protocol.prompt_lengths_blocks:
        summaries: list[DepthSummary] = []
        for depth in protocol.depths_for(total_blocks):
            observations: list[PrefillMeasurement] = []
            for repetition in range(protocol.warmup_repetitions):
                observations.append(measure(total_blocks, depth, repetition))
            for repetition in range(protocol.measured_repetitions):
                observations.append(measure(total_blocks, depth, repetition))
            raw.extend(observations)
            summaries.append(
                summarize_depth(
                    observations, total_blocks=total_blocks, cached_blocks=depth
                )
            )
        curves.append(build_curve(summaries, protocol=protocol))
    return tuple(curves), tuple(raw)


def environment_provenance(
    *,
    model_name: str,
    model_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    runtime_mode: str,
    hardware: str,
    vllm_version: str,
    torch_version: str,
    cuda_version: str | None,
    driver_version: str | None,
    block_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    enable_chunked_prefill: bool,
    cache_block_override: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Freeze the measurement environment; every field is externally supplied."""
    for name, value in (
        ("model_name", model_name),
        ("model_revision", model_revision),
        ("runtime_mode", runtime_mode),
        ("hardware", hardware),
        ("vllm_version", vllm_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance field {name} must be nonempty text")
    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "runtime_mode": runtime_mode,
        "hardware": hardware,
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "block_size": block_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": enforce_eager,
        "enable_chunked_prefill": enable_chunked_prefill,
        "cache_block_override": cache_block_override,
        "measured_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kvopt_version": getattr(kvopt, "__version__", "0.1.0"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        **(dict(extra) if extra else {}),
    }


def profile_document(
    *,
    curves: Sequence[CurveSummary],
    protocol: CalibrationProtocol,
    provenance: Mapping[str, Any],
    measurement_note: str,
) -> dict[str, object]:
    """Serialize the artifact consumed by the Native runner arm."""
    if not curves:
        raise CalibrationError("profile requires at least one calibrated curve")
    if provenance.get("block_size") != protocol.block_size:
        raise CalibrationError(
            "protocol block_size must equal the measured runtime block size "
            f"({protocol.block_size} vs {provenance.get('block_size')})"
        )
    return {
        "schema_version": PROFILE_SCHEMA,
        "measurement_source": MEASUREMENT_SOURCE,
        "measurement_note": measurement_note,
        "protocol": {
            "prompt_lengths_blocks": list(protocol.prompt_lengths_blocks),
            "block_size": protocol.block_size,
            "warmup_repetitions": protocol.warmup_repetitions,
            "measured_repetitions": protocol.measured_repetitions,
            "minimum_valid_repetitions": protocol.minimum_valid_repetitions,
            "noise_multiplier": protocol.noise_multiplier,
            "interpolation_stride": protocol.interpolation_stride,
            "timing_method": (
                "torch.cuda.synchronize before/after one prefill execution, "
                "wall clock; warmup excluded from aggregation"
            ),
        },
        "provenance": dict(provenance),
        "curves": [curve.payload() for curve in curves],
        "resolution": {
            "all_curves_discriminative": all(curve.discriminative for curve in curves),
            "curve_status": {
                str(curve.total_blocks): (
                    "DISCRIMINATIVE"
                    if curve.discriminative
                    else "INSUFFICIENT_RESOLUTION"
                )
                for curve in curves
            },
        },
        "calibration_evidence_schema": CALIBRATION_SCHEMA,
    }


def write_profile(path: Path, document: Mapping[str, Any]) -> str:
    """Write the profile atomically and return its SHA-256."""
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preflight_profile(
    profile_path: Path, *, require_discriminative: bool = True
) -> dict[str, object]:
    """Check a written profile before an expensive Native campaign."""
    from kvopt.costaware.asym_apc_runner import load_calibration_profile

    profile = load_calibration_profile(profile_path)
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    resolution = document.get("resolution", {})
    discriminative = resolution.get("all_curves_discriminative")
    if require_discriminative and discriminative is not True:
        raise CalibrationError(
            "calibration profile is not discriminative at the measured noise floor; "
            "the proxy loss cannot separate candidates, so the campaign must not be "
            "interpreted as a policy comparison"
        )
    return {
        "profile_path": str(profile_path),
        "profile_sha256": profile.sha256,
        "covered_lengths": list(profile.covered_lengths),
        "block_size": profile.block_size,
        "discriminative": discriminative,
        "provenance": dict(profile.provenance),
    }
