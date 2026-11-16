"""Gated Native runner arm for the AsymCache-inspired contiguous-prefix selector.

This module is the deployment boundary between the experimental
``AsymAPCNativeCoordinator`` and a real vLLM Native execution path. It provides:

* strict, fail-closed parsing of the opt-in policy configuration;
* a validated loader for the GPU-measured prefill calibration profile;
* an instrumented coordinator wrapper that records decision-level audit facts
  (including selector overhead) without changing selection semantics;
* before/after Native ``BlockPool`` state capture with explicit unavailability
  reasons instead of invented values;
* reconciliation of *proposed* victim Block IDs against the Block IDs that the
  Native runtime physically evicted;
* a check that the next request observes the cached-prefix depth implied by the
  evicted set, so a Python callback returning IDs is never treated as proof of
  cache-state change.

Nothing here is enabled implicitly. ``ASYM_APC`` must be selected explicitly and
must carry a hash-bound calibration profile; otherwise the arm refuses to start.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kvopt.costaware.asym_apc_native import (
    APCLatencyCurve,
    AsymAPCAudit,
    AsymAPCNativeCoordinator,
    AsymDecay,
)

POLICY_NAME = "ASYM_APC"
PROFILE_SCHEMA = "kvopt.asym_apc.prefill_calibration_profile.v1"

#: Exact-reason classification. Unknown reasons are counted as UNCLASSIFIED
#: rather than silently mapped onto an existing category.
_REASON_CATEGORIES: dict[str, str] = {
    "no_pressure": "NO_PRESSURE",
    "no_choice": "NO_LEGAL_CHOICE",
    "selected_lower_proxy_loss": "SELECTED",
    "selected_partial_coverage": "SELECTED",
    "duplicate_eligibility": "CONTRACT_VIOLATION",
    "candidate_snapshot_mismatch": "CONTRACT_VIOLATION",
    "nonzero_refcount": "CONTRACT_VIOLATION",
    "tier_contract_mismatch": "CONTRACT_VIOLATION",
    "tier_order_mismatch": "CONTRACT_VIOLATION",
    "request_contract_mismatch": "CONTRACT_VIOLATION",
    "selection_contract_failed": "CONTRACT_VIOLATION",
    "no_verified_alternative": "EVIDENCE_INCOMPLETE",
    "insufficient_verified_candidates": "EVIDENCE_INCOMPLETE",
    "content_contract_mismatch": "EVIDENCE_INCOMPLETE",
    "content_unverified": "EVIDENCE_INCOMPLETE",
    "content_identity_mismatch": "EVIDENCE_INCOMPLETE",
    "hashless_has_memberships": "EVIDENCE_INCOMPLETE",
    "depth_unverified": "EVIDENCE_INCOMPLETE",
    "depth_collision": "EVIDENCE_INCOMPLETE",
    "chain_length_mismatch": "EVIDENCE_INCOMPLETE",
    "profile_missing": "CALIBRATION_INCOMPLETE",
    "recency_missing": "EVIDENCE_INCOMPLETE",
    "future_observation": "EVIDENCE_INCOMPLETE",
    "baseline_unverified": "EVIDENCE_INCOMPLETE",
    "invalid_native_evidence": "EVIDENCE_INVALID",
    "search_budget_exceeded": "SEARCH_BUDGET",
    "baseline_no_verified_proxy_gain": "NO_PROXY_GAIN",
}


def classify_fallback_reason(reason: str) -> str:
    """Map one exact coordinator reason onto a reporting category."""
    if not isinstance(reason, str) or not reason:
        raise ValueError("fallback reason must be nonempty text")
    base = reason.split(":", 1)[0]
    return _REASON_CATEGORIES.get(base, "UNCLASSIFIED")


class AsymAPCConfigurationError(ValueError):
    """Raised when the opt-in arm is configured in an unverifiable way."""


class CalibrationProfileError(ValueError):
    """Raised when a calibration profile cannot support the requested range."""


# ---------------------------------------------------------------------------
# Calibration profile loading
# ---------------------------------------------------------------------------


DEFAULT_ALLOWED_MEASUREMENT_SOURCES: tuple[str, ...] = ("native_gpu",)


@dataclass(frozen=True, slots=True)
class AsymAPCProfile:
    """Hash-bound prefill cost curves plus the device they were measured on."""

    path: Path
    sha256: str
    curves_by_length: Mapping[int, APCLatencyCurve]
    provenance: Mapping[str, Any]
    block_size: int
    measurement_source: str = "native_gpu"

    @property
    def covered_lengths(self) -> tuple[int, ...]:
        return tuple(sorted(self.curves_by_length))

    def require(self, lengths: Sequence[int]) -> None:
        """Fail closed when a resident chain length was never calibrated."""
        missing = sorted({length for length in lengths if length not in self.curves_by_length})
        if missing:
            raise CalibrationProfileError(
                "calibration profile does not cover resident prefix lengths "
                f"{missing}; covered lengths are {list(self.covered_lengths)}"
            )


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationProfileError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise CalibrationProfileError(f"{field_name} must be finite and nonnegative")
    return number


def load_calibration_profile(
    path: Path,
    *,
    expected_sha256: str | None = None,
    allowed_sources: Sequence[str] = DEFAULT_ALLOWED_MEASUREMENT_SOURCES,
) -> AsymAPCProfile:
    """Load and validate a calibration profile written by a calibrator.

    ``allowed_sources`` defaults to GPU measurements only, so a profile produced
    on another device can never be picked up silently. A caller that intends to
    run the selector on a different device must pass that source explicitly, and
    the source is then carried into every audit record instead of being relabelled
    as a GPU measurement.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.is_file():
        raise CalibrationProfileError(f"calibration profile not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise CalibrationProfileError(
            f"calibration profile SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise CalibrationProfileError("calibration profile must contain an object")
    if document.get("schema_version") != PROFILE_SCHEMA:
        raise CalibrationProfileError("unsupported calibration profile schema")
    measurement_source = document.get("measurement_source")
    if measurement_source not in allowed_sources:
        raise CalibrationProfileError(
            "calibration profile measurement source "
            f"{measurement_source!r} is not accepted here; allowed sources are "
            f"{list(allowed_sources)}"
        )
    curves: dict[int, APCLatencyCurve] = {}
    for entry in document.get("curves", []):
        if not isinstance(entry, Mapping):
            raise CalibrationProfileError("curve entries must be objects")
        total_blocks = entry.get("total_blocks")
        if isinstance(total_blocks, bool) or not isinstance(total_blocks, int):
            raise CalibrationProfileError("curve total_blocks must be an int")
        samples: list[tuple[int, float]] = []
        for sample in entry.get("samples", []):
            if not isinstance(sample, Mapping):
                raise CalibrationProfileError("curve samples must be objects")
            depth = sample.get("cached_blocks")
            if isinstance(depth, bool) or not isinstance(depth, int):
                raise CalibrationProfileError("sample cached_blocks must be an int")
            samples.append((depth, _require_finite(sample.get("prefill_ms"), "prefill_ms")))
        curves[total_blocks] = APCLatencyCurve(total_blocks, tuple(samples))
    if not curves:
        raise CalibrationProfileError("calibration profile contains no curves")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CalibrationProfileError("calibration profile requires measurement provenance")
    for required in (
        "model_name",
        "model_revision",
        "hardware",
        "block_size",
        "vllm_version",
        "runtime_mode",
        "measured_at_utc",
    ):
        if not provenance.get(required):
            raise CalibrationProfileError(
                f"calibration provenance is missing required field {required!r}"
            )
    block_size = provenance["block_size"]
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise CalibrationProfileError("provenance block_size must be a positive int")
    return AsymAPCProfile(
        path=path,
        sha256=digest,
        curves_by_length=curves,
        provenance=dict(provenance),
        block_size=block_size,
        measurement_source=str(measurement_source),
    )


# ---------------------------------------------------------------------------
# Opt-in policy configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsymAPCPolicyConfig:
    """Explicit opt-in configuration; every field must be supplied by config."""

    profile_path: Path
    profile_sha256: str
    alpha_seconds: float = 60.0
    beta_seconds: float = 12.0
    knee_seconds: float = 60.0
    tau_seconds: float = 1.0
    max_exact_subsets: int = 3000
    max_greedy_operations: int = 50000
    min_proxy_gain_ms: float = 1e-9
    clock_domain: str = "system_monotonic"

    def decay(self) -> AsymDecay:
        return AsymDecay(
            alpha_seconds=self.alpha_seconds,
            beta_seconds=self.beta_seconds,
            knee_seconds=self.knee_seconds,
        )

    def describe(self) -> dict[str, object]:
        return {
            "policy": POLICY_NAME,
            "profile_path": str(self.profile_path),
            "profile_sha256": self.profile_sha256,
            "alpha_seconds": self.alpha_seconds,
            "beta_seconds": self.beta_seconds,
            "knee_seconds": self.knee_seconds,
            "tau_seconds": self.tau_seconds,
            "max_exact_subsets": self.max_exact_subsets,
            "max_greedy_operations": self.max_greedy_operations,
            "min_proxy_gain_ms": self.min_proxy_gain_ms,
            "clock_domain": self.clock_domain,
        }


def parse_policy_config(
    raw: object, *, repo_root: Path, clock_domain: str
) -> AsymAPCPolicyConfig:
    """Parse ``backend_options.asym_apc``; refuse partially specified opt-ins."""
    if not isinstance(raw, Mapping):
        raise AsymAPCConfigurationError("backend_options.asym_apc must be an object")
    if raw.get("enabled") is not True:
        raise AsymAPCConfigurationError(
            "ASYM_APC requires backend_options.asym_apc.enabled = true"
        )
    profile_value = raw.get("profile_path")
    sha_value = raw.get("profile_sha256")
    if not isinstance(profile_value, str) or not profile_value:
        raise AsymAPCConfigurationError("asym_apc.profile_path must be nonempty text")
    if not isinstance(sha_value, str) or len(sha_value) != 64:
        raise AsymAPCConfigurationError(
            "asym_apc.profile_sha256 must bind the measured calibration profile"
        )
    profile_path = Path(profile_value)
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path

    def positive(name: str, default: float | None) -> float:
        value = raw.get(name, default)
        if value is None:
            raise AsymAPCConfigurationError(f"asym_apc.{name} is required")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AsymAPCConfigurationError(f"asym_apc.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise AsymAPCConfigurationError(f"asym_apc.{name} must be positive")
        return number

    def positive_int(name: str, default: int | None) -> int:
        value = raw.get(name, default)
        if value is None:
            raise AsymAPCConfigurationError(f"asym_apc.{name} is required")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AsymAPCConfigurationError(f"asym_apc.{name} must be a positive int")
        return value

    minimum_gain = raw.get("min_proxy_gain_ms", 1e-9)
    if isinstance(minimum_gain, bool) or not isinstance(minimum_gain, (int, float)):
        raise AsymAPCConfigurationError("asym_apc.min_proxy_gain_ms must be numeric")
    if not math.isfinite(float(minimum_gain)) or float(minimum_gain) < 0:
        raise AsymAPCConfigurationError("asym_apc.min_proxy_gain_ms must be nonnegative")

    return AsymAPCPolicyConfig(
        profile_path=profile_path,
        profile_sha256=sha_value,
        alpha_seconds=positive("alpha_seconds", 60.0),
        beta_seconds=positive("beta_seconds", 12.0),
        knee_seconds=positive("knee_seconds", 60.0),
        tau_seconds=positive("tau_seconds", 1.0),
        max_exact_subsets=positive_int("max_exact_subsets", 3000),
        max_greedy_operations=positive_int("max_greedy_operations", 50000),
        min_proxy_gain_ms=float(minimum_gain),
        clock_domain=clock_domain,
    )


# ---------------------------------------------------------------------------
# Decision-level audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsymAPCDecision:
    """One instrumented selection attempt, including measured CPU overhead."""

    decision_id: str
    timestamp: float
    clock_domain: str
    required_blocks: int
    legal_candidate_count: int
    audit: AsymAPCAudit
    selector_overhead_seconds: float
    execution_context: Mapping[str, object] = field(default_factory=dict)
    # Monotonic high-resolution measurement of the same call. Kept separate
    # from the timestamp clock so that a coarse or synthetic clock can never
    # masquerade as a decision-cost measurement.
    selector_overhead_ns: int = 0

    @property
    def reason_category(self) -> str:
        return classify_fallback_reason(self.audit.reason)

    @property
    def proxy_gain_ms(self) -> float | None:
        if self.audit.baseline_proxy_ms is None or self.audit.selected_proxy_ms is None:
            return None
        return self.audit.baseline_proxy_ms - self.audit.selected_proxy_ms

    def payload(self) -> dict[str, object]:
        return {
            "policy": POLICY_NAME,
            "decision_id": self.decision_id,
            "decision_timestamp": self.timestamp,
            "decision_clock_domain": self.clock_domain,
            "required_blocks": self.required_blocks,
            "legal_candidate_count": self.legal_candidate_count,
            "baseline_victim_block_ids": tuple(self.audit.baseline),
            "selected_victim_block_ids": tuple(self.audit.selected),
            "victim_set_differs": self.audit.victim_set_differs,
            "reason": self.audit.reason,
            "reason_category": self.reason_category,
            "baseline_proxy_ms": self.audit.baseline_proxy_ms,
            "selected_proxy_ms": self.audit.selected_proxy_ms,
            "predicted_incremental_prefill_gain_ms": self.proxy_gain_ms,
            "ranked_candidate_count": self.audit.ranked_candidate_count,
            "exact_search": self.audit.exact_search,
            "evidence_missing_reason_counts": dict(self.audit.missing_reasons),
            "selector_overhead_seconds": self.selector_overhead_seconds,
            "selector_overhead_ns": self.selector_overhead_ns,
            "claim_boundary": (
                "contiguous-APC proxy loss difference under a calibrated cost curve; "
                "not an expected recomputed-token count or measured latency"
            ),
            **dict(self.execution_context),
        }


class InstrumentedAsymAPCCoordinator:
    """A0-compatible ``select_victims`` boundary that records audit facts.

    The wrapper adds no decision logic. It measures the wall-clock cost of the
    selector, captures the immutable audit record, and exposes both to the
    runner for structured event emission.
    """

    def __init__(
        self,
        coordinator: AsymAPCNativeCoordinator,
        *,
        clock: Callable[[], float] = time.perf_counter,
        decision_prefix: str = "asym-apc",
        context_provider: Callable[[], Mapping[str, object]] | None = None,
        time_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not callable(getattr(coordinator, "select_victims", None)):
            raise TypeError("coordinator must provide select_victims")
        self._coordinator = coordinator
        self._clock = clock
        self._time_ns = time_ns
        self._prefix = decision_prefix
        self._context_provider = context_provider
        self._sequence = 0
        self.decisions: list[AsymAPCDecision] = []
        self.last_decision: AsymAPCDecision | None = None
        self.pending_decision: AsymAPCDecision | None = None

    @property
    def coordinator(self) -> AsymAPCNativeCoordinator:
        return self._coordinator

    def select_victims(
        self,
        candidates: Sequence[Any],
        preparation: Any,
        retention_entries: Sequence[Any],
        *,
        required_blocks: int,
        timestamp: float,
    ) -> tuple[int, ...]:
        self._sequence += 1
        decision_id = f"{self._prefix}-{self._sequence}"
        legal_count = len(preparation.virtual_eligible_order)
        started = self._clock()
        started_ns = self._time_ns()
        selected = self._coordinator.select_victims(
            candidates,
            preparation,
            retention_entries,
            required_blocks=required_blocks,
            timestamp=timestamp,
        )
        elapsed_ns = self._time_ns() - started_ns
        elapsed = self._clock() - started
        audit = self._coordinator.last_audit
        if audit is None:
            raise RuntimeError("coordinator returned without an audit record")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("selector overhead measurement is invalid")
        if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns < 0:
            raise RuntimeError("selector nanosecond overhead measurement is invalid")
        decision = AsymAPCDecision(
            decision_id=decision_id,
            timestamp=float(timestamp),
            clock_domain=self._coordinator.clock_domain,
            required_blocks=int(required_blocks),
            legal_candidate_count=legal_count,
            audit=audit,
            selector_overhead_seconds=float(elapsed),
            selector_overhead_ns=int(elapsed_ns),
            execution_context=(
                dict(self._context_provider()) if self._context_provider is not None else {}
            ),
        )
        self.decisions.append(decision)
        self.last_decision = decision
        self.pending_decision = decision
        return selected

    def pop_pending_decision(self) -> AsymAPCDecision | None:
        """Consume the latest decision exactly once (physical reconciliation)."""
        decision, self.pending_decision = self.pending_decision, None
        return decision


# ---------------------------------------------------------------------------
# Native BlockPool state capture
# ---------------------------------------------------------------------------


def capture_native_pool_state(pool: Any) -> dict[str, object]:
    """Capture observable Native ``BlockPool`` facts without inventing values.

    Every field is either an observed value or ``None`` with an explicit
    ``unavailable_reasons`` entry. This function never mutates the pool.
    """
    reasons: list[str] = []
    if pool is None:
        return {
            "availability": "UNAVAILABLE",
            "unavailable_reasons": ("native_block_pool_not_captured",),
        }
    blocks = getattr(pool, "blocks", None)
    if blocks is None:
        return {
            "availability": "UNAVAILABLE",
            "unavailable_reasons": ("native_block_pool_without_blocks",),
        }
    total_blocks = len(blocks)
    null_blocks = 0
    hashed_free = 0
    refcnt_nonzero = 0
    for block in blocks:
        is_null = getattr(block, "is_null", None)
        if is_null:
            null_blocks += 1
        ref_cnt = getattr(block, "ref_cnt", None)
        if isinstance(ref_cnt, int) and ref_cnt > 0:
            refcnt_nonzero += 1
        block_hash = getattr(block, "block_hash", None)
        if block_hash is not None and ref_cnt == 0:
            hashed_free += 1
    free_queue = getattr(pool, "free_block_queue", None)
    free_count: int | None = None
    if free_queue is None:
        reasons.append("native_free_block_queue_not_exposed")
    else:
        try:
            free_count = len(free_queue)
        except TypeError:
            candidate = getattr(free_queue, "num_free_blocks", None)
            if isinstance(candidate, int):
                free_count = candidate
            else:
                reasons.append("native_free_block_queue_length_unavailable")
    return {
        "availability": "AVAILABLE" if not reasons else "PARTIAL",
        "unavailable_reasons": tuple(reasons),
        "total_blocks": total_blocks,
        "null_block_count": null_blocks,
        "hashed_free_block_count": hashed_free,
        "allocated_block_count": refcnt_nonzero,
        "free_block_count": free_count,
    }


# ---------------------------------------------------------------------------
# Physical eviction reconciliation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PendingReconciliation:
    pool_before: Mapping[str, object]
    opened_at: float
    decision: AsymAPCDecision | None = None
    proposed: tuple[int, ...] = ()
    evicted: list[tuple[int, str | None]] = field(default_factory=list)


class PhysicalEvictionReconciler:
    """Reconcile proposed victims with physically evicted Native Block IDs.

    A proposed victim that never reaches the Native hash eviction path, or a
    physical eviction the selector did not propose, is reported explicitly. The
    reconciler never assumes that returning IDs from Python changed the cache.

    Native eviction callbacks can arrive slightly after the allocation call that
    triggered them, so the record is settled twice: once when the decision
    returns and again when the execution context closes. A decision with no
    observed eviction yet stays pending instead of being reported as a negative.
    """

    def __init__(self) -> None:
        self._pending: _PendingReconciliation | None = None
        self._sequence = 0
        self.reconciled: list[dict[str, object]] = []

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def open_context(self, *, pool_state: Mapping[str, object], timestamp: float) -> None:
        """Start buffering evictions for one Native execution context."""
        if self._pending is not None:
            raise RuntimeError("previous physical reconciliation was not finalized")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("reconciliation timestamp must be finite and nonnegative")
        self._pending = _PendingReconciliation(
            pool_before=dict(pool_state), opened_at=float(timestamp)
        )

    def attach_decision(self, decision: AsymAPCDecision, proposed: Sequence[int]) -> None:
        if self._pending is None:
            raise RuntimeError("reconciliation context is not open")
        if self._pending.decision is not None:
            raise RuntimeError("a decision was already attached to this context")
        expected = tuple(decision.audit.selected)
        proposed_tuple = tuple(int(value) for value in proposed)
        if proposed_tuple != expected:
            raise RuntimeError(
                "proposed victims do not match the recorded audit selection"
            )
        self._pending.decision = decision
        self._pending.proposed = proposed_tuple

    def observe_eviction(self, block_id: int, content_hash: str | None) -> None:
        """Record one Native hashed-eviction observation, if a context is open."""
        if self._pending is None:
            return
        self._pending.evicted.append((int(block_id), content_hash))

    def settle(
        self,
        *,
        pool_state_after: Mapping[str, object],
        phase: str,
    ) -> dict[str, object] | None:
        """Produce the reconciliation record for the current phase.

        ``phase`` is ``decision_return`` right after the selector ran, or
        ``context_close`` when the execution context ends. Only the latter
        finalizes a decision with no observed physical eviction.
        """
        if phase not in {"decision_return", "context_close"}:
            raise ValueError("phase must be decision_return or context_close")
        pending = self._pending
        if pending is None:
            return None
        if pending.decision is None and phase == "decision_return":
            return None
        if (
            phase == "decision_return"
            and not pending.evicted
            and pending.decision is not None
        ):
            # Eviction callbacks may still arrive before the context closes.
            return None
        self._pending = None
        self._sequence += 1
        decision = pending.decision
        proposed_set = set(pending.proposed)
        evicted_ids = tuple(block_id for block_id, _hash in pending.evicted)
        evicted_set = set(evicted_ids)
        matched = tuple(sorted(proposed_set & evicted_set))
        missing = tuple(sorted(proposed_set - evicted_set))
        unexpected = tuple(sorted(evicted_set - proposed_set))
        if not proposed_set and not evicted_set:
            status = "NO_PRESSURE"
        elif not proposed_set:
            status = "EVICTION_WITHOUT_SELECTION"
        elif not evicted_set:
            status = "NO_PHYSICAL_EVICTION_OBSERVED"
        elif not missing and not unexpected:
            status = "EXACT_PHYSICAL_MATCH"
        elif matched:
            status = "PARTIAL_PHYSICAL_MATCH"
        else:
            status = "PROPOSAL_AND_PHYSICAL_DISJOINT"
        before_free = pending.pool_before.get("free_block_count")
        after_free = pool_state_after.get("free_block_count")
        free_delta = (
            after_free - before_free
            if isinstance(before_free, int) and isinstance(after_free, int)
            else None
        )
        record = {
            "policy": POLICY_NAME,
            "reconciliation_id": f"asym-apc-physical-{self._sequence}",
            "decision_id": None if decision is None else decision.decision_id,
            "settled_phase": phase,
            "status": status,
            "proposed_victim_block_ids": pending.proposed,
            "physically_evicted_block_ids": evicted_ids,
            "matched_block_ids": matched,
            "unmatched_proposed_block_ids": missing,
            "unexpected_evicted_block_ids": unexpected,
            "agreement_ratio": (
                len(matched) / len(proposed_set) if proposed_set else None
            ),
            "pool_state_before": dict(pending.pool_before),
            "pool_state_after": dict(pool_state_after),
            "pool_free_block_delta": free_delta,
            "evidence_note": (
                "proposed victims come from the selector audit; "
                "physically_evicted_block_ids come from Native hashed cache-entry "
                "eviction callbacks inside the same execution context"
            ),
        }
        self.reconciled.append(record)
        return record


# ---------------------------------------------------------------------------
# Next-request cached-prefix expectation
# ---------------------------------------------------------------------------


def expected_reusable_prefix_by_chain(
    registry: Any, evicted_block_ids: Sequence[int]
) -> dict[str, int]:
    """Expected leading reusable prefix per chain after a victim set is evicted.

    Only chains the registry can prove are reported. A chain whose identity is
    unavailable is omitted rather than assigned an assumed depth.
    """
    evicted = {int(value) for value in evicted_block_ids}
    snapshots = getattr(registry, "_chains", None)
    if not isinstance(snapshots, Mapping):
        return {}
    expectations: dict[str, int] = {}
    for chain_id, snapshot in snapshots.items():
        block_ids = tuple(snapshot.block_ids)
        depth = 0
        for block_id in block_ids:
            if block_id in evicted:
                break
            depth += 1
        expectations[str(chain_id)] = depth
    return expectations


def check_observed_reuse(
    expectations: Mapping[str, int],
    observed: Mapping[str, int],
) -> dict[str, object]:
    """Compare expected and observed reusable prefix depth per chain."""
    mismatched: list[dict[str, object]] = []
    unchecked: list[str] = []
    compared = 0
    for chain_id, expected in expectations.items():
        actual = observed.get(chain_id)
        if actual is None:
            unchecked.append(chain_id)
            continue
        compared += 1
        if actual != expected:
            mismatched.append(
                {"chain_id": chain_id, "expected_blocks": expected, "observed_blocks": actual}
            )
    return {
        "policy": POLICY_NAME,
        "chain_count": len(expectations),
        "compared_chain_count": compared,
        "unchecked_chain_ids": tuple(sorted(unchecked)),
        "mismatches": tuple(mismatched),
        "status": (
            "NO_EXPECTATION"
            if not expectations
            else "MISMATCH"
            if mismatched
            else "MATCH"
            if compared
            else "UNCHECKED"
        ),
    }


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


def build_asym_apc_arm(
    *,
    registry: Any,
    config: AsymAPCPolicyConfig,
    pool_probe: Callable[[], Any] | None = None,
    context_provider: Callable[[], Mapping[str, object]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    allowed_sources: Sequence[str] = DEFAULT_ALLOWED_MEASUREMENT_SOURCES,
    cost_model: str = "contiguous_prefix",
    weighting: str = "two_timescale",
) -> tuple[InstrumentedAsymAPCCoordinator, AsymAPCProfile, PhysicalEvictionReconciler]:
    """Build the gated arm, refusing to start without a measured profile."""
    profile = load_calibration_profile(
        config.profile_path,
        expected_sha256=config.profile_sha256,
        allowed_sources=allowed_sources,
    )
    coordinator = AsymAPCNativeCoordinator(
        registry,
        profile.curves_by_length,
        decay=config.decay(),
        tau_seconds=config.tau_seconds,
        max_exact_subsets=config.max_exact_subsets,
        max_greedy_operations=config.max_greedy_operations,
        min_proxy_gain_ms=config.min_proxy_gain_ms,
        clock_domain=config.clock_domain,
        cost_model=cost_model,
        weighting=weighting,
    )
    instrumented = InstrumentedAsymAPCCoordinator(
        coordinator,
        clock=clock,
        context_provider=context_provider,
    )
    del pool_probe  # pool state is captured at decision boundaries by the runner
    return instrumented, profile, PhysicalEvictionReconciler()


def fallback_summary(decisions: Sequence[AsymAPCDecision]) -> dict[str, object]:
    """Aggregate decision-level fallback reasons and divergence statistics."""
    reasons = Counter(decision.audit.reason for decision in decisions)
    categories = Counter(decision.reason_category for decision in decisions)
    missing = Counter()
    for decision in decisions:
        for reason, count in decision.audit.missing_reasons:
            missing[reason] += count
    divergent = sum(1 for decision in decisions if decision.audit.victim_set_differs)
    overheads = [decision.selector_overhead_seconds for decision in decisions]
    return {
        "decision_count": len(decisions),
        "victim_set_divergence_count": divergent,
        "victim_set_divergence_rate": (divergent / len(decisions)) if decisions else None,
        "reason_counts": dict(sorted(reasons.items())),
        "reason_category_counts": dict(sorted(categories.items())),
        "evidence_missing_reason_counts": dict(sorted(missing.items())),
        "selector_overhead_seconds_median": _median(overheads),
        "selector_overhead_seconds_p90": _percentile(overheads, 90.0),
        "selector_overhead_seconds_max": max(overheads) if overheads else None,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


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
