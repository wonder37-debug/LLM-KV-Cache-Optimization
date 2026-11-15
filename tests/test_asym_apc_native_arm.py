"""CPU tests for the gated ASYM_APC Native runner arm.

These tests exercise the deployment boundary (opt-in configuration, calibration
profile binding, decision auditing, BlockPool evidence and physical-eviction
reconciliation) without a GPU. They do not, and cannot, establish Native GPU
performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("kvopt.costaware.a2_native")

from kvopt.continuum import FakeClock, RetentionMode
from kvopt.continuum.retention import RetentionManager
from kvopt.costaware import asym_apc_calibration as calibration
from kvopt.costaware.a2_native import A2NativeChainSnapshot, A2NativeSnapshotRegistry
from kvopt.costaware.asym_apc_native import APCLatencyCurve, AsymAPCNativeCoordinator
from kvopt.costaware.asym_apc_runner import (
    AsymAPCConfigurationError,
    AsymAPCProfile,
    CalibrationProfileError,
    InstrumentedAsymAPCCoordinator,
    PhysicalEvictionReconciler,
    build_asym_apc_arm,
    capture_native_pool_state,
    check_observed_reuse,
    classify_fallback_reason,
    expected_reusable_prefix_by_chain,
    fallback_summary,
    load_calibration_profile,
    parse_policy_config,
)
from kvopt.profiling.experiment_events import NullExperimentEventSink
from kvopt.runtime.vllm.continuum_observation import encode_native_hash
from kvopt.runtime.vllm.retention_integration import RetentionRuntimeIntegration

REPO_ROOT = Path(__file__).resolve().parents[1]


class RecordingSink(NullExperimentEventSink):
    """Collect events for assertions."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)

    def event_types(self) -> list[str]:
        return [getattr(event, "event_type", "") for event in self.events]

    def first(self, event_type: str) -> object:
        for event in self.events:
            if getattr(event, "event_type", "") == event_type:
                return event
        raise AssertionError(f"event {event_type} was not emitted")


def _measurement(
    total_blocks: int,
    cached_blocks: int,
    repetition: int,
    *,
    block_size: int = 16,
    invalid_reason: str | None = None,
) -> calibration.PrefillMeasurement:
    # A convex, strictly decreasing cost surface in milliseconds.
    remaining = total_blocks - cached_blocks
    elapsed_ms = 3.0 + 4.0 * remaining + 0.05 * remaining * remaining
    return calibration.PrefillMeasurement(
        total_blocks=total_blocks,
        requested_cached_blocks=cached_blocks,
        elapsed_seconds=elapsed_ms / 1000.0,
        observed_cached_tokens=cached_blocks * block_size,
        observed_computed_tokens=remaining * block_size,
        repetition=repetition,
        invalid_reason=invalid_reason,
    )


def _synthetic_profile_for_tests(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (4, 8),
    block_size: int = 16,
    discriminative: bool = True,
) -> Path:
    """Build a schema-valid profile from synthetic measurements (test fixture only)."""
    protocol = calibration.CalibrationProtocol(
        prompt_lengths_blocks=lengths,
        block_size=block_size,
        warmup_repetitions=1,
        measured_repetitions=3,
        minimum_valid_repetitions=2,
    )

    def measure(total_blocks: int, cached_blocks: int, repetition: int):
        return _measurement(total_blocks, cached_blocks, repetition, block_size=block_size)

    curves, _ = calibration.run_calibration(measure, protocol)
    if not discriminative:
        curves = tuple(
            calibration.CurveSummary(
                total_blocks=curve.total_blocks,
                curve=curve.curve,
                depths=curve.depths,
                monotone=curve.monotone,
                noise_floor_ms=curve.noise_floor_ms,
                minimum_adjacent_gap_ms=curve.minimum_adjacent_gap_ms,
                discriminative=False,
                interpolation_max_absolute_ms=curve.interpolation_max_absolute_ms,
                interpolation_max_relative=curve.interpolation_max_relative,
            )
            for curve in curves
        )
    provenance = calibration.environment_provenance(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        model_revision="test-revision",
        tokenizer_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tokenizer_revision="test-revision",
        runtime_mode="vllm-cuda-observability",
        hardware="test-fixture",
        vllm_version="0.27.1",
        torch_version="2.10.0",
        cuda_version="12.8",
        driver_version=None,
        block_size=block_size,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        enable_chunked_prefill=False,
        cache_block_override=64,
    )
    document = calibration.profile_document(
        curves=curves,
        protocol=protocol,
        provenance=provenance,
        measurement_note="synthetic fixture; not a GPU measurement",
    )
    path = tmp_path / "profile.json"
    calibration.write_profile(path, document)
    return path


def _policy_options(profile_path: Path, sha256: str) -> dict[str, object]:
    return {
        "enabled": True,
        "profile_path": str(profile_path),
        "profile_sha256": sha256,
        "alpha_seconds": 60.0,
        "beta_seconds": 12.0,
        "knee_seconds": 60.0,
        "tau_seconds": 1.0,
    }


def _registry_with_chains(
    chains: dict[str, tuple[SimpleNamespace, ...]], *, observed_at: float | None = 50.0
) -> A2NativeSnapshotRegistry:
    registry = A2NativeSnapshotRegistry()
    for chain_id, blocks in chains.items():
        snapshot = A2NativeChainSnapshot(
            chain_id=chain_id,
            block_ids=tuple(block.block_id for block in blocks),
            content_hashes=tuple(encode_native_hash(block.block_hash) for block in blocks),
            native_hash_process_id="test-process",
        )
        if observed_at is None:
            registry.observe(snapshot)
        else:
            registry.observe(
                snapshot, observed_at=observed_at, clock_domain="system_monotonic"
            )
    return registry


def _blocks(count: int, start: int = 1) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            block_id=index,
            ref_cnt=0,
            block_hash=f"content-{index}".encode(),
        )
        for index in range(start, start + count)
    )


# ---------------------------------------------------------------------------
# Opt-in configuration gates
# ---------------------------------------------------------------------------


def test_policy_requires_explicit_enable(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    options = _policy_options(profile, profile_sha)
    options["enabled"] = False
    with pytest.raises(AsymAPCConfigurationError, match="enabled"):
        parse_policy_config(options, repo_root=REPO_ROOT, clock_domain="system_monotonic")
    with pytest.raises(AsymAPCConfigurationError, match="object"):
        parse_policy_config(None, repo_root=REPO_ROOT, clock_domain="system_monotonic")


def test_policy_requires_hash_bound_profile(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    options = _policy_options(profile, "0" * 64)
    with pytest.raises(CalibrationProfileError, match="SHA-256 mismatch"):
        config = parse_policy_config(
            options, repo_root=REPO_ROOT, clock_domain="system_monotonic"
        )
        build_asym_apc_arm(registry=A2NativeSnapshotRegistry(), config=config)


def test_profile_must_be_a_native_measurement(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    document = json.loads(profile.read_text(encoding="utf-8"))
    document["measurement_source"] = "simulator"
    profile.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CalibrationProfileError, match="Native GPU measurement"):
        load_calibration_profile(profile)


def test_profile_requires_complete_provenance(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    document = json.loads(profile.read_text(encoding="utf-8"))
    del document["provenance"]["hardware"]
    profile.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CalibrationProfileError, match="hardware"):
        load_calibration_profile(profile)


def test_profile_must_cover_resident_chain_lengths(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path, lengths=(4,))
    loaded = load_calibration_profile(profile)
    loaded.require((4,))
    with pytest.raises(CalibrationProfileError, match=r"\[8\]"):
        loaded.require((4, 8))


def test_runner_rejects_asym_apc_outside_its_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.spikes.phase2_minimal_observability_metal import (
        MinimalMetalObservabilityBackend,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    config = {
        "runtime_mode": "vllm-cuda-observability",
        "campaign_kind": "phase3_final_evaluation",
        "seed": 7,
        "policy": "continuum-baseline",
        "model": {"name": "m", "revision": "r"},
        "tokenizer": {"name": "m", "revision": "r"},
        "cache": {"block_override": 64},
        "pressure": {"initial_shortage_blocks": 1},
        "generation": {"max_new_tokens": 1, "temperature": 0.0},
        "backend_options": {"phase2g_block_policy": "ASYM_APC"},
    }
    with pytest.raises(ValueError, match="asym_apc_native_evaluation"):
        MinimalMetalObservabilityBackend(config, NullExperimentEventSink())


def test_runner_requires_content_evidence_before_policy_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Native content evidence the arm is refused before it is built."""
    from scripts.spikes.phase2_minimal_observability_metal import (
        MinimalMetalObservabilityBackend,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    config = {
        "runtime_mode": "vllm-cuda-observability",
        "campaign_kind": "asym_apc_native_evaluation",
        "seed": 7,
        "policy": "continuum-baseline",
        "model": {"name": "m", "revision": "r"},
        "tokenizer": {"name": "m", "revision": "r"},
        "cache": {"block_override": 64},
        "pressure": {"initial_shortage_blocks": 1},
        "generation": {"max_new_tokens": 1, "temperature": 0.0},
        "backend_options": {
            "phase2g_block_policy": "ASYM_APC",
            "asym_apc": {"enabled": True},
        },
    }
    with pytest.raises(ValueError, match="content evidence"):
        MinimalMetalObservabilityBackend(config, NullExperimentEventSink())


def test_default_block_policy_remains_a0(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    options = _policy_options(profile, profile_sha)
    config = parse_policy_config(
        options, repo_root=REPO_ROOT, clock_domain="system_monotonic"
    )
    assert config.profile_sha256 == profile_sha
    assert config.clock_domain == "system_monotonic"


# ---------------------------------------------------------------------------
# Fallback classification and audit
# ---------------------------------------------------------------------------


def test_fallback_reason_classification_is_exact_and_total() -> None:
    assert classify_fallback_reason("no_pressure") == "NO_PRESSURE"
    assert classify_fallback_reason("no_choice") == "NO_LEGAL_CHOICE"
    assert classify_fallback_reason("selected_lower_proxy_loss") == "SELECTED"
    assert classify_fallback_reason("profile_missing") == "CALIBRATION_INCOMPLETE"
    assert (
        classify_fallback_reason("invalid_native_evidence:ValueError")
        == "EVIDENCE_INVALID"
    )
    assert classify_fallback_reason("baseline_no_verified_proxy_gain") == "NO_PROXY_GAIN"
    assert classify_fallback_reason("something_new") == "UNCLASSIFIED"


def test_instrumented_arm_records_overhead_and_divergence(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    arm, profile_object, reconciler = build_asym_apc_arm(
        registry=registry, config=config
    )
    assert isinstance(profile_object, AsymAPCProfile)
    assert isinstance(reconciler, PhysicalEvictionReconciler)
    arm.coordinator.observe_chain_access("A", 50.0)
    arm.coordinator.observe_chain_access("B", 60.0)
    runtime = RetentionRuntimeIntegration(
        RetentionManager(FakeClock(100.0)), block_selection_coordinator=arm
    )
    removed: list[int] = []
    result = runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED,
        blocks=blocks,
        required_blocks=1,
        timestamp=100.0,
        remove_selected=lambda ids: removed.extend(ids),
    )
    assert result.selected_block_ids == (4,)
    assert removed == [4]
    decision = arm.last_decision
    assert decision is not None
    assert decision.audit.victim_set_differs
    assert decision.reason_category == "SELECTED"
    assert decision.selector_overhead_seconds >= 0.0
    assert decision.payload()["predicted_incremental_prefill_gain_ms"] is not None
    assert arm.pop_pending_decision() is decision
    assert arm.pop_pending_decision() is None

    summary = fallback_summary(arm.decisions)
    assert summary["decision_count"] == 1
    assert summary["victim_set_divergence_count"] == 1
    assert summary["reason_category_counts"] == {"SELECTED": 1}


def test_arm_never_promotes_unverified_candidates(tmp_path: Path) -> None:
    """A candidate without proved content is not a legal victim."""
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]}, observed_at=None)
    arm, _, _ = build_asym_apc_arm(registry=registry, config=config)
    runtime = RetentionRuntimeIntegration(
        RetentionManager(FakeClock(100.0)), block_selection_coordinator=arm
    )
    # No observed access evidence for either chain: the coordinator must fall back
    # to the exact A0 victim set instead of inventing recency.
    result = runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED,
        blocks=blocks,
        required_blocks=1,
        timestamp=100.0,
        remove_selected=lambda ids: None,
    )
    assert result.selected_block_ids == (1,)
    assert arm.last_decision is not None
    assert arm.last_decision.audit.victim_set_differs is False
    assert arm.last_decision.audit.reason == "recency_missing"
    assert arm.last_decision.reason_category == "EVIDENCE_INCOMPLETE"


# ---------------------------------------------------------------------------
# Native BlockPool evidence
# ---------------------------------------------------------------------------


def test_pool_state_capture_reports_unavailability() -> None:
    assert capture_native_pool_state(None)["availability"] == "UNAVAILABLE"
    no_blocks = SimpleNamespace()
    assert capture_native_pool_state(no_blocks)["availability"] == "UNAVAILABLE"

    blocks = [
        SimpleNamespace(is_null=True, ref_cnt=0, block_hash=None),
        SimpleNamespace(is_null=False, ref_cnt=0, block_hash=b"aaa"),
        SimpleNamespace(is_null=False, ref_cnt=2, block_hash=b"bbb"),
    ]
    pool = SimpleNamespace(blocks=blocks, free_block_queue=[1, 2])
    state = capture_native_pool_state(pool)
    assert state["availability"] == "AVAILABLE"
    assert state["total_blocks"] == 3
    assert state["null_block_count"] == 1
    assert state["hashed_free_block_count"] == 1
    assert state["allocated_block_count"] == 1
    assert state["free_block_count"] == 2

    opaque_queue = SimpleNamespace()
    partial = capture_native_pool_state(
        SimpleNamespace(blocks=blocks, free_block_queue=opaque_queue)
    )
    assert partial["availability"] == "PARTIAL"
    assert partial["free_block_count"] is None
    assert "native_free_block_queue_length_unavailable" in partial["unavailable_reasons"]


def test_reconciliation_statuses(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    arm, _, reconciler = build_asym_apc_arm(registry=registry, config=config)
    arm.coordinator.observe_chain_access("A", 50.0)
    arm.coordinator.observe_chain_access("B", 60.0)
    runtime = RetentionRuntimeIntegration(
        RetentionManager(FakeClock(100.0)), block_selection_coordinator=arm
    )

    def run(evicted: tuple[int, ...], *, phase: str = "decision_return") -> dict:
        before = {"free_block_count": 4}
        reconciler.open_context(pool_state=before, timestamp=100.0)
        result = runtime.apply_pressure(
            mode=RetentionMode.CONTROLLED,
            blocks=blocks,
            required_blocks=1,
            timestamp=100.0,
            remove_selected=lambda ids: None,
        )
        decision = arm.pop_pending_decision()
        reconciler.attach_decision(decision, decision.audit.selected)
        for block_id in evicted:
            reconciler.observe_eviction(block_id, None)
        record = reconciler.settle(
            pool_state_after={"free_block_count": 4 + len(evicted)}, phase=phase
        )
        assert result.selected_block_ids == decision.audit.selected
        return record

    exact = run((4,))
    assert exact is not None
    assert exact["status"] == "EXACT_PHYSICAL_MATCH"
    assert exact["agreement_ratio"] == 1.0
    assert exact["pool_free_block_delta"] == 1

    partial = run((4, 5))
    assert partial["status"] == "PARTIAL_PHYSICAL_MATCH"
    assert partial["unexpected_evicted_block_ids"] == (5,)

    disjoint = run((7,))
    assert disjoint["status"] == "PROPOSAL_AND_PHYSICAL_DISJOINT"
    assert disjoint["agreement_ratio"] == 0.0

    deferred = run((), phase="decision_return")
    assert deferred is None, "unobserved eviction must stay pending, not report failure"
    assert reconciler.has_pending
    final = reconciler.settle(pool_state_after={"free_block_count": 4}, phase="context_close")
    assert final["status"] == "NO_PHYSICAL_EVICTION_OBSERVED"


def test_attach_decision_rejects_mismatched_proposal(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    arm, _, reconciler = build_asym_apc_arm(registry=registry, config=config)
    arm.coordinator.observe_chain_access("A", 50.0)
    arm.coordinator.observe_chain_access("B", 60.0)
    runtime = RetentionRuntimeIntegration(
        RetentionManager(FakeClock(100.0)), block_selection_coordinator=arm
    )
    reconciler.open_context(pool_state={}, timestamp=100.0)
    runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED,
        blocks=blocks,
        required_blocks=1,
        timestamp=100.0,
        remove_selected=lambda ids: None,
    )
    decision = arm.pop_pending_decision()
    with pytest.raises(RuntimeError, match="audit selection"):
        reconciler.attach_decision(decision, (99,))


# ---------------------------------------------------------------------------
# Next-request cached-prefix expectation
# ---------------------------------------------------------------------------


def test_expected_reuse_and_observed_check() -> None:
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    expectations = expected_reusable_prefix_by_chain(registry, (2,))
    assert expectations == {"A": 1, "B": 4}
    result = check_observed_reuse(expectations, {"A": 1})
    assert result["status"] == "MATCH"
    assert result["compared_chain_count"] == 1
    assert result["unchecked_chain_ids"] == ("B",)
    mismatch = check_observed_reuse(expectations, {"A": 3})
    assert mismatch["status"] == "MISMATCH"
    assert check_observed_reuse({}, {})["status"] == "NO_EXPECTATION"


# ---------------------------------------------------------------------------
# Calibration quality gates
# ---------------------------------------------------------------------------


def test_calibration_rejects_unverified_apc_hits() -> None:
    protocol = calibration.CalibrationProtocol(
        prompt_lengths_blocks=(4,),
        warmup_repetitions=0,
        measured_repetitions=3,
        minimum_valid_repetitions=2,
    )

    def measure(total_blocks: int, cached_blocks: int, repetition: int):
        return _measurement(
            total_blocks,
            cached_blocks,
            repetition,
            invalid_reason=(
                "apc_hit_depth_mismatch" if cached_blocks == 2 else None
            ),
        )

    with pytest.raises(calibration.CalibrationError, match="no valid measurement"):
        calibration.run_calibration(measure, protocol)


def test_calibration_rejects_non_monotone_curve() -> None:
    protocol = calibration.CalibrationProtocol(
        prompt_lengths_blocks=(3,),
        warmup_repetitions=0,
        measured_repetitions=3,
        minimum_valid_repetitions=2,
    )
    counter = {"n": 0}

    def measure(total_blocks: int, cached_blocks: int, repetition: int):
        counter["n"] += 1
        if cached_blocks == 1:
            elapsed_ms = 40.0  # slower than depth 0: physically impossible
        else:
            elapsed_ms = 30.0 - 5.0 * cached_blocks
        return calibration.PrefillMeasurement(
            total_blocks=total_blocks,
            requested_cached_blocks=cached_blocks,
            elapsed_seconds=elapsed_ms / 1000.0,
            observed_cached_tokens=cached_blocks * 16,
            observed_computed_tokens=(total_blocks - cached_blocks) * 16,
            repetition=repetition,
        )

    with pytest.raises(calibration.CalibrationError, match="not non-increasing"):
        calibration.run_calibration(measure, protocol)


def test_calibration_reports_interpolation_error(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path, lengths=(8,))
    document = json.loads(profile.read_text(encoding="utf-8"))
    curve = document["curves"][0]
    assert curve["interpolation_max_absolute_ms"] is not None
    assert curve["interpolation_max_relative"] is not None
    assert curve["discriminative"] is True
    assert document["resolution"]["all_curves_discriminative"] is True


def test_preflight_rejects_non_discriminative_profile(tmp_path: Path) -> None:
    profile = _synthetic_profile_for_tests(tmp_path, lengths=(4,), discriminative=False)
    with pytest.raises(calibration.CalibrationError, match="not discriminative"):
        calibration.preflight_profile(profile)
    report = calibration.preflight_profile(profile, require_discriminative=False)
    assert report["discriminative"] is False


# ---------------------------------------------------------------------------
# Integration with the observed retention boundary
# ---------------------------------------------------------------------------


def test_observed_integration_emits_decision_and_physical_outcome(tmp_path: Path) -> None:
    from scripts.spikes.phase2_minimal_observability_metal import (
        _CountingForcedReleaseObserver,
        _IndexedEventSink,
        _ObservedRetentionIntegration,
    )

    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    arm, _, reconciler = build_asym_apc_arm(registry=registry, config=config)
    arm.coordinator.observe_chain_access("A", 50.0)
    arm.coordinator.observe_chain_access("B", 60.0)

    sink = RecordingSink()
    event_index = _IndexedEventSink(sink)
    clock = FakeClock(100.0)
    forced = _CountingForcedReleaseObserver(event_index, event_index)
    integration = _ObservedRetentionIntegration(
        SimpleNamespace(retention=RetentionManager(FakeClock(100.0))),
        clock,
        forced,
        phase2g_block_policy="ASYM_APC",
        phase2g_a2_registry=registry,
        asym_apc_arm=arm,
        asym_apc_reconciler=reconciler,
        asym_apc_pool_provider=lambda: None,
        event_index=event_index,
        sink=event_index,
        clock_domain="system_monotonic",
    )
    integration.begin_pressure_context(
        boundary_id="boundary", stage_id="stage", pressure_request_id="pressure-1"
    )
    removed: list[int] = []
    integration.apply_pressure(
        mode=RetentionMode.CONTROLLED,
        blocks=blocks,
        required_blocks=1,
        timestamp=100.0,
        remove_selected=lambda ids: removed.extend(ids),
    )
    integration.observe_native_hash_eviction(4, "hash-4")
    integration.observe_native_selection_returned(((4, None),))
    integration.end_pressure_context()

    types = sink.event_types()
    assert "ASYM_APC_BLOCK_DECISION" in types
    assert "ASYM_APC_PHYSICAL_OUTCOME" in types
    decision_event = sink.first("ASYM_APC_BLOCK_DECISION")
    payload = decision_event.payload
    assert payload["policy"] == "ASYM_APC"
    assert payload["selected_victim_block_ids"] == (4,)
    assert payload["pressure_request_id"] == "pressure-1"
    outcome = sink.first("ASYM_APC_PHYSICAL_OUTCOME")
    assert outcome.payload["status"] == "EXACT_PHYSICAL_MATCH"
    assert outcome.payload["pool_state_before"]["availability"] == "UNAVAILABLE"
    assert removed == [4]

    integration.asym_apc_reuse_check("A", 3)
    reuse = sink.first("ASYM_APC_NEXT_REQUEST_REUSE_CHECK")
    assert reuse.payload["expected_reusable_prefix_blocks"] == 3
    assert reuse.payload["status"] == "MATCH"
    summary = integration.asym_apc_fallback_summary()
    assert summary["decision_count"] == 1


def test_observed_integration_rejects_arm_for_other_policy() -> None:
    from scripts.spikes.phase2_minimal_observability_metal import (
        _CountingForcedReleaseObserver,
        _IndexedEventSink,
        _ObservedRetentionIntegration,
    )

    sink = RecordingSink()
    event_index = _IndexedEventSink(sink)
    forced = _CountingForcedReleaseObserver(event_index, event_index)
    coordinator = InstrumentedAsymAPCCoordinator(
        AsymAPCNativeCoordinator(
            A2NativeSnapshotRegistry(), {4: APCLatencyCurve(4, ((0, 20.0), (4, 0.0)))}
        )
    )
    with pytest.raises(ValueError, match="must not be constructed"):
        _ObservedRetentionIntegration(
            SimpleNamespace(retention=RetentionManager(FakeClock(100.0))),
            FakeClock(100.0),
            forced,
            phase2g_block_policy="A0",
            asym_apc_arm=coordinator,
            asym_apc_reconciler=PhysicalEvictionReconciler(),
            event_index=event_index,
            sink=event_index,
        )
    with pytest.raises(ValueError, match="requires the instrumented arm"):
        _ObservedRetentionIntegration(
            SimpleNamespace(retention=RetentionManager(FakeClock(100.0))),
            FakeClock(100.0),
            forced,
            phase2g_block_policy="ASYM_APC",
            event_index=event_index,
            sink=event_index,
        )


def test_two_decisions_in_one_context_settle_the_first(tmp_path: Path) -> None:
    """A second allocation attempt must not collide with a pending reconciliation."""
    from scripts.spikes.phase2_minimal_observability_metal import (
        _CountingForcedReleaseObserver,
        _IndexedEventSink,
        _ObservedRetentionIntegration,
    )

    profile = _synthetic_profile_for_tests(tmp_path)
    profile_sha = load_calibration_profile(profile).sha256
    config = parse_policy_config(
        _policy_options(profile, profile_sha),
        repo_root=REPO_ROOT,
        clock_domain="system_monotonic",
    )
    blocks = _blocks(8)
    registry = _registry_with_chains({"A": blocks[:4], "B": blocks[4:]})
    arm, _, reconciler = build_asym_apc_arm(registry=registry, config=config)
    arm.coordinator.observe_chain_access("A", 50.0)
    arm.coordinator.observe_chain_access("B", 60.0)

    sink = RecordingSink()
    event_index = _IndexedEventSink(sink)
    forced = _CountingForcedReleaseObserver(event_index, event_index)
    integration = _ObservedRetentionIntegration(
        SimpleNamespace(retention=RetentionManager(FakeClock(100.0))),
        FakeClock(100.0),
        forced,
        phase2g_block_policy="ASYM_APC",
        phase2g_a2_registry=registry,
        asym_apc_arm=arm,
        asym_apc_reconciler=reconciler,
        asym_apc_pool_provider=lambda: None,
        event_index=event_index,
        sink=event_index,
        clock_domain="system_monotonic",
    )
    integration.begin_pressure_context(
        boundary_id="boundary", stage_id="stage", pressure_request_id="pressure-1"
    )
    for _ in range(2):
        integration.apply_pressure(
            mode=RetentionMode.CONTROLLED,
            blocks=blocks,
            required_blocks=1,
            timestamp=100.0,
            remove_selected=lambda ids: None,
        )
        integration.observe_native_selection_returned(((4, None),))
    integration.end_pressure_context()

    types = sink.event_types()
    assert types.count("ASYM_APC_BLOCK_DECISION") == 2
    assert "ASYM_APC_PHYSICAL_OUTCOME" in types
    statuses = {
        event.payload["status"] for event in sink.events if event.event_type == "ASYM_APC_PHYSICAL_OUTCOME"
    }
    assert "NO_PHYSICAL_EVICTION_OBSERVED" in statuses
