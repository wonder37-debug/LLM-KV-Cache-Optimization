"""CPU tests for the AsymAPC campaign materializer, runner contract and report.

No GPU is required. These tests prove the campaign definition, arm wiring, paired
structure and reporting arithmetic; they do not execute Native inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("kvopt.costaware.a2_native")

from kvopt.costaware import asym_apc_calibration as calibration
from kvopt.profiling import asym_apc_report as report
from kvopt.workload import asym_apc_materialize as materialize
from kvopt.workload import asym_apc_runner as campaign_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN = REPO_ROOT / "configs/asym_apc/asym-apc-native-evaluation-v1.json"


def _profile(tmp_path: Path) -> Path:
    protocol = calibration.CalibrationProtocol(
        prompt_lengths_blocks=(4, 8),
        block_size=16,
        warmup_repetitions=1,
        measured_repetitions=3,
        minimum_valid_repetitions=2,
    )

    def measure(total_blocks: int, cached_blocks: int, repetition: int):
        remaining = total_blocks - cached_blocks
        return calibration.PrefillMeasurement(
            total_blocks=total_blocks,
            requested_cached_blocks=cached_blocks,
            elapsed_seconds=(2.0 + 3.0 * remaining) / 1000.0,
            observed_cached_tokens=cached_blocks * 16,
            observed_computed_tokens=remaining * 16,
            repetition=repetition,
        )

    curves, _ = calibration.run_calibration(measure, protocol)
    provenance = calibration.environment_provenance(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        model_revision="fixture-revision",
        tokenizer_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tokenizer_revision="fixture-revision",
        runtime_mode="vllm-cuda-observability",
        hardware="fixture",
        vllm_version="0.27.1",
        torch_version="2.10.0",
        cuda_version="12.8",
        driver_version=None,
        block_size=16,
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        enable_chunked_prefill=False,
        cache_block_override=512,
    )
    document = calibration.profile_document(
        curves=curves,
        protocol=protocol,
        provenance=provenance,
        measurement_note="test fixture",
    )
    path = tmp_path / "profile.json"
    calibration.write_profile(path, document)
    return path


def test_materialize_requires_measured_profile(tmp_path: Path) -> None:
    with pytest.raises(materialize.MaterializationError, match="calibration profile"):
        materialize.materialize_campaign(
            design_path=DESIGN,
            output_dir=tmp_path / "out",
            profile_path=tmp_path / "missing.json",
            subset="screening",
        )


def test_materialize_screening_campaign(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    manifest_path = materialize.materialize_campaign(
        design_path=DESIGN,
        output_dir=tmp_path / "out",
        profile_path=profile,
        subset="screening",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 2 cases x 2 regimes x 3 arms x 2 repeats
    assert manifest["planned_run_count"] == 24
    assert manifest["paired_units"] == 8
    assert manifest["arms"] == ["NATIVE_LRU", "A0", "ASYM_APC"]
    assert manifest["profile_sha256"] == calibration.preflight_profile(profile)[
        "profile_sha256"
    ]
    assert manifest["execution_started"] is False

    configs = {
        (run["arm"], run["case_id"], run["pressure_regime"], run["repeat"]): json.loads(
            (tmp_path / "out" / run["config"]).read_text(encoding="utf-8")
        )
        for run in manifest["runs"]
    }
    for run in manifest["runs"]:
        assert run["config_sha256"] == materialize._sha256(tmp_path / "out" / run["config"])

    native = configs[("NATIVE_LRU", "w1-long-prefix-reuse", "high", 1)]
    a0 = configs[("A0", "w1-long-prefix-reuse", "high", 1)]
    asym = configs[("ASYM_APC", "w1-long-prefix-reuse", "high", 1)]

    assert native["policy"] == "native-lru"
    assert native["backend_options"]["phase2g_release_policy"] == "N0"
    assert native["backend_options"]["phase2g_block_policy"] == "A0"
    assert "asym_apc" not in native["backend_options"]

    assert a0["policy"] == "continuum-a0"
    assert a0["backend_options"]["phase2g_release_policy"] == "A0"
    assert a0["backend_options"]["phase2g_block_policy"] == "A0"
    assert "asym_apc" not in a0["backend_options"]
    assert a0["campaign_kind"] == "asym_apc_native_evaluation"

    assert asym["backend_options"]["phase2g_release_policy"] == "A0"
    assert asym["backend_options"]["phase2g_block_policy"] == "ASYM_APC"
    assert asym["backend_options"]["asym_apc"]["enabled"] is True
    assert asym["backend_options"]["asym_apc"]["profile_sha256"] == manifest["profile_sha256"]

    # Identical inputs, capacity and timing across the three arms of one pair.
    for field in ("trace", "cache", "pressure", "generation", "seed", "model"):
        assert native[field] == a0[field] == asym[field]
    for option in ("program_prefix_tokens", "prefix_segments_by_request", "pressure_prompt_tokens"):
        assert (
            native["backend_options"][option]
            == a0["backend_options"][option]
            == asym["backend_options"][option]
        )
    from kvopt.workload.phase2c_native_pilot import PILOT_PREFIX_GENERATOR_VERSION

    assert asym["backend_options"]["phase2c_prefix_generator_version"] == (
        PILOT_PREFIX_GENERATOR_VERSION
    )
    assert a0["backend_options"]["phase2c_prefix_generator_version"] == (
        PILOT_PREFIX_GENERATOR_VERSION
    )


def test_materialize_rejects_tampered_design(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    design["arms"] = ["A0", "ASYM_APC"]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(materialize.MaterializationError, match="arms"):
        materialize.materialize_campaign(
            design_path=tampered,
            output_dir=tmp_path / "out",
            profile_path=profile,
            subset="screening",
        )


def _event(event_type: str, timestamp: float, payload: dict, request_id: str | None = None):
    record = {
        "schema_version": "phase2.event.v1",
        "run_id": "fixture",
        "event_index": 0,
        "event_type": event_type,
        "timestamp": timestamp,
        "clock_domain": "system_monotonic",
        "source": "fixture",
        "payload": payload,
    }
    if request_id is not None:
        record["request_id"] = request_id
    return record


def _write_run(
    root: Path,
    *,
    run_id: str,
    arm: str,
    case_id: str,
    regime: str,
    repeat: int,
    reusable: tuple[int, int],
    segments: dict[str, list[list[object]]] | None,
    divergent: bool,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {
        "schema_version": "phase2.config.v1",
        "asym_apc_stage": {
            "stage": "ASYM_APC_NATIVE_EVALUATION",
            "case_id": case_id,
            "case_kind": "representative",
            "pressure_regime": regime,
            "arm": arm,
            "repeat": repeat,
        },
        "backend_options": {},
    }
    if segments is not None:
        config["backend_options"]["prefix_segments_by_request"] = segments
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "execution_status": "success",
                "stage": "ASYM_APC_NATIVE_EVALUATION",
            }
        ),
        encoding="utf-8",
    )
    events = [
        _event("REQUEST_ARRIVED", 1.0, {"kind": "turn"}, "p:1"),
        _event(
            "VLLM_NATIVE_REQUEST_OBSERVATION",
            1.15,
            {"native_prompt_tokens": 96, "native_cached_prefix_tokens": reusable[0]},
            "p:1",
        ) if segments is not None else None,
        _event(
            "VLLM_PREFIX_SNAPSHOT",
            1.2,
            {"reusable_token_count": reusable[0]},
            "p:1",
        ),
        _event("VLLM_REQUEST_COMPLETED", 1.5, {"output_token_count": 1}, "p:1"),
        _event("REQUEST_ARRIVED", 2.0, {"kind": "turn"}, "p:2"),
        _event(
            "VLLM_NATIVE_REQUEST_OBSERVATION",
            2.15,
            {"native_prompt_tokens": 96, "native_cached_prefix_tokens": reusable[1]},
            "p:2",
        ) if segments is not None else None,
        _event(
            "VLLM_PREFIX_SNAPSHOT",
            2.2,
            {"reusable_token_count": reusable[1]},
            "p:2",
        ),
        _event("VLLM_REQUEST_COMPLETED", 2.7, {"output_token_count": 1}, "p:2"),
    ]
    # Keep missing request-time evidence unavailable, never replace with a
    # post-prefill snapshot merely because the latter is present.
    events = [event for event in events if event is not None]
    if arm == "ASYM_APC":
        events.append(
            _event(
                "ASYM_APC_BLOCK_DECISION",
                2.5,
                {
                    "policy": "ASYM_APC",
                    "reason": "selected_lower_proxy_loss",
                    "reason_category": "SELECTED",
                    "victim_set_differs": divergent,
                    "selector_overhead_seconds": 0.002,
                    "predicted_incremental_prefill_gain_ms": 3.5,
                },
                "pressure:1",
            )
        )
        events.append(
            _event(
                "ASYM_APC_PHYSICAL_OUTCOME",
                2.6,
                {"status": "EXACT_PHYSICAL_MATCH", "agreement_ratio": 1.0},
                "pressure:1",
            )
        )
        events.append(
            _event(
                "ASYM_APC_NEXT_REQUEST_REUSE_CHECK",
                3.0,
                {"status": "MATCH", "expected_reusable_prefix_blocks": 3},
                "p:2",
            )
        )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_report_computes_paired_metrics(tmp_path: Path) -> None:
    segments = {"p:1": [["sys", 64], ["t1", 32]], "p:2": [["sys", 64], ["t2", 32]]}
    run_dirs = [
        _write_run(
            tmp_path,
            run_id="case-a0",
            arm="A0",
            case_id="case",
            regime="medium",
            repeat=1,
            reusable=(0, 64),
            segments=segments,
            divergent=False,
        ),
        _write_run(
            tmp_path,
            run_id="case-asym",
            arm="ASYM_APC",
            case_id="case",
            regime="medium",
            repeat=1,
            reusable=(0, 96),
            segments=segments,
            divergent=True,
        ),
    ]
    summary = report.summarize_runs(run_dirs)
    assert summary["run_count"] == 2
    assert summary["completed_run_count"] == 2

    a0_metrics = next(row for row in summary["runs"] if row["arm"] == "A0")
    asym_metrics = next(row for row in summary["runs"] if row["arm"] == "ASYM_APC")
    assert a0_metrics["recomputed_prefill_tokens"] == 96 + 32
    assert asym_metrics["recomputed_prefill_tokens"] == 96
    assert asym_metrics["prefix_hit_rate"] == pytest.approx(96 / 192)
    assert asym_metrics["decision_count"] == 1
    assert asym_metrics["victim_set_divergence_rate"] == 1.0
    assert asym_metrics["physical_exact_match_rate"] == 1.0
    assert asym_metrics["next_request_reuse_match_count"] == 1
    assert a0_metrics["decision_count"] == 0
    assert a0_metrics["physical_exact_match_rate"] is None

    paired = summary["paired"]["comparisons"]["asym_apc_minus_a0"]
    assert paired["paired_unit_count"] == 1
    assert paired["recomputed_token_outcome_counts"] == {"better": 1}
    assert paired["deltas"]["recomputed_prefill_tokens"]["median"] == -32.0

    markdown = report.render_markdown(summary, title="Fixture report")
    assert "asym_apc_minus_a0" in markdown
    assert "recomputed_prefill_tokens" in markdown


def test_report_marks_unavailable_prompt_token_counts(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="case-a0",
        arm="A0",
        case_id="case",
        regime="low",
        repeat=1,
        reusable=(0, 64),
        segments=None,
        divergent=False,
    )
    evidence = report.load_run(run_dir)
    metrics = report.run_metrics(evidence)
    assert metrics["recomputed_prefill_tokens"] is None
    assert metrics["recomputed_prefill_tokens_reason"].startswith(
        "native_request_time_cache_evidence_incomplete"
    )
    assert metrics["prefix_hit_rate"] is None
    assert metrics["mean_reusable_prefix_tokens"] is None



def test_report_ignores_post_prefill_snapshot_hits(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="cold-then-hit",
        arm="A0",
        case_id="case",
        regime="high",
        repeat=1,
        reusable=(0, 64),
        segments={"p:1": [["head", 64], ["body", 32]], "p:2": [["head", 64], ["body2", 32]]},
        divergent=False,
    )
    path = run_dir / "events.jsonl"
    events = [json.loads(row) for row in path.read_text().splitlines()]
    for event in events:
        if event["event_type"] == "VLLM_PREFIX_SNAPSHOT":
            # The full prompt is resident AFTER prefill, not a request-time hit.
            event["payload"]["reusable_token_count"] = 96
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    metrics = report.run_metrics(report.load_run(run_dir))
    assert metrics["recomputed_prefill_tokens"] == 128
    assert metrics["prefix_hit_rate"] == pytest.approx(64 / 192)


def test_report_rejects_missing_native_request_observation(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="missing-native",
        arm="A0",
        case_id="case",
        regime="high",
        repeat=1,
        reusable=(0, 64),
        segments={"p:1": [["head", 64], ["body", 32]], "p:2": [["head", 64], ["body2", 32]]},
        divergent=False,
    )
    path = run_dir / "events.jsonl"
    events = [json.loads(row) for row in path.read_text().splitlines()]
    events = [
        e for e in events
        if not (e["event_type"] == "VLLM_NATIVE_REQUEST_OBSERVATION"
                and e["request_id"] == "p:2")
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    metrics = report.run_metrics(report.load_run(run_dir))
    assert metrics["recomputed_prefill_tokens"] is None
    assert "request_observation_count:p:2:0" in metrics["recomputed_prefill_tokens_reason"]
    assert metrics["prefix_hit_rate"] is None

def test_campaign_runner_rejects_unknown_run_ids(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "asym_apc.native_evaluation_materialization.v1",
        "subset": "screening",
        "runs": [{"run_id": "only-run", "config": "only-run.config.json"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown run IDs"):
        campaign_runner.run_campaign(
            manifest_path=manifest_path,
            config_root=tmp_path,
            output_root=tmp_path / "runs",
            run_ids=["other-run"],
        )


def test_campaign_runner_records_failures_without_counting_them_as_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "schema_version": "asym_apc.native_evaluation_materialization.v1",
        "subset": "screening",
        "runs": [
            {"run_id": "run-a", "config": "run-a.config.json"},
            {"run_id": "run-b", "config": "run-b.config.json"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_run_single(config_path: Path, *, output_root: Path, run_id: str) -> Path:
        del config_path
        if run_id == "run-b":
            raise RuntimeError("CUDA out of memory")
        output = output_root / run_id
        output.mkdir(parents=True)
        (output / "run.json").write_text(
            json.dumps({"execution_status": "success"}), encoding="utf-8"
        )
        return output

    monkeypatch.setattr(campaign_runner, "run_single", fake_run_single)
    summary = campaign_runner.run_campaign(
        manifest_path=manifest_path,
        config_root=tmp_path,
        output_root=tmp_path / "runs",
    )
    assert summary["completed_run_count"] == 1
    assert summary["failed_run_count"] == 1
    assert summary["failures"][0]["run_id"] == "run-b"
    stored = json.loads((tmp_path / "runs" / "campaign-summary.json").read_text())
    assert stored["completed_run_ids"] == ["run-a"]
