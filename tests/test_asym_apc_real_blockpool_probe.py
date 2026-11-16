"""CPU tests for the real-BlockPool probe's workload and curve-quality logic.

The probe itself needs vLLM, but its workload construction and its calibration
quality report are pure Python. Both decide whether a later measurement is
interpretable, so they are pinned here: an append-only workload is what makes
prefix reuse real, and a curve-quality report that hides a non-monotonic curve
would turn a measurement artefact into a claimed gain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES = REPO_ROOT / "scripts" / "probes" / "asym_apc_native"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROBES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("real_blockpool_eval")
calibrate = _load("calibrate_cpu_device")
gpu_calibrate = _load("calibrate")


def _workload(**overrides) -> list:
    spec = probe.WorkloadSpec(
        programs=overrides.get("programs", 3),
        turns_per_program=overrides.get("turns", 4),
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        short_gap_seconds=2.0,
        long_gap_seconds=90.0,
        seed=overrides.get("seed", 11),
        target_blocks=overrides.get("target_blocks", (24, 32, 40, 40)),
    )
    return spec.build()


def test_turns_of_one_program_are_prompt_prefixes_of_each_other() -> None:
    turns = _workload()
    by_program: dict[str, list] = {}
    for turn in turns:
        by_program.setdefault(turn.program, []).append(turn)
    for program_turns in by_program.values():
        program_turns.sort(key=lambda item: item.sequence_index)
        for earlier, later in zip(program_turns, program_turns[1:]):
            assert later.token_ids[: len(earlier.token_ids)] == earlier.token_ids
            assert later.prompt_blocks >= earlier.prompt_blocks


def test_every_prompt_is_block_aligned() -> None:
    for turn in _workload():
        assert turn.prompt_tokens % probe.BLOCK_TOKENS == 0
        assert turn.prompt_blocks == turn.prompt_tokens // probe.BLOCK_TOKENS


def test_workload_reaches_every_requested_prompt_length() -> None:
    targets = (24, 32, 40, 40)
    lengths = {turn.prompt_blocks for turn in _workload(target_blocks=targets)}
    assert lengths == set(targets)


def test_seed_changes_tokens_but_not_geometry() -> None:
    first = _workload(seed=1)
    second = _workload(seed=2)
    assert [turn.prompt_blocks for turn in first] == [turn.prompt_blocks for turn in second]
    assert [turn.token_ids for turn in first] != [turn.token_ids for turn in second]


def _point(length: int, cached: int, ms: float, spread: float = 0.01):
    return calibrate.Point(
        total_blocks=length // probe.BLOCK_TOKENS,
        cached_blocks=cached // probe.BLOCK_TOKENS,
        prompt_tokens=length,
        cached_tokens=cached,
        median_ms=ms,
        p90_ms=ms * (1 + spread),
        minimum_ms=ms * (1 - spread),
        maximum_ms=ms * (1 + spread),
        repeats=7,
    )


def test_quality_accepts_a_monotonic_curve() -> None:
    report = calibrate._curve_quality([_point(384, 0, 1200.0), _point(384, 384, 0.0)])
    assert report["all_monotonic"] is True
    assert report["per_length"][0]["violations"] == []


def test_quality_reports_a_non_monotonic_curve_instead_of_smoothing_it() -> None:
    report = calibrate._curve_quality(
        [_point(384, 0, 1200.0), _point(384, 192, 1400.0), _point(384, 384, 0.0)]
    )
    assert report["all_monotonic"] is False
    violation = report["per_length"][0]["violations"][0]
    assert violation["from_cached_blocks"] == 0
    assert violation["to_cached_blocks"] == 12


def test_quality_reports_saving_per_cached_block() -> None:
    report = calibrate._curve_quality(
        [_point(512, 0, 1600.0), _point(512, 256, 800.0), _point(512, 512, 0.0)]
    )
    entry = report["per_length"][0]
    assert entry["total_saving_ms"] == pytest.approx(1600.0)
    assert entry["ms_saved_per_cached_block_min"] == pytest.approx(50.0)


def test_fully_cached_point_costs_zero() -> None:
    payload = _point(640, 640, 0.0).payload()
    assert payload["computed_tokens"] == 0
    assert payload["prefill_ms"] == 0.0


def test_gpu_calibration_repetitions_preserve_only_the_requested_prefix() -> None:
    base = list(range(400, 464))
    x = gpu_calibrate.tokens_for_repetition(
        base, cached_blocks=2, block_size=16, repetition=0, vocabulary_size=1000
    )
    y = gpu_calibrate.tokens_for_repetition(
        base, cached_blocks=2, block_size=16, repetition=1, vocabulary_size=1000
    )
    assert x[:32] == y[:32] == base[:32]
    assert x[32] != y[32] != base[32]
    assert x[33:] == y[33:] == base[33:]


def test_gpu_calibration_cold_samples_never_hit_the_same_first_block() -> None:
    base = list(range(400, 464))
    samples = [
        gpu_calibrate.tokens_for_repetition(
            base, cached_blocks=0, block_size=16, repetition=rep, vocabulary_size=1000
        )
        for rep in range(10)
    ]
    assert len({sample[0] for sample in samples}) == 10


def test_gpu_calibration_fully_cached_sample_is_unchanged() -> None:
    base = list(range(400, 464))
    assert gpu_calibrate.tokens_for_repetition(
        base, cached_blocks=4, block_size=16, repetition=7, vocabulary_size=1000
    ) == base


def test_a0_and_asym_apc_both_register_the_same_retention_entry(monkeypatch, tmp_path) -> None:
    """A0 must not silently run with an empty RetentionManager."""
    from types import SimpleNamespace
    from kvopt.continuum.retention import RetentionManager
    from kvopt.costaware import asym_apc_runner

    class FakeSubstrate:
        def __init__(self, num_blocks):
            self.blocks = {
                i: SimpleNamespace(block_id=i, ref_cnt=0, block_hash=None)
                for i in range(1, num_blocks + 1)
            }
            self.queue = list(self.blocks)

        def request(self, _request_id, _token_ids):
            return SimpleNamespace(block_hashes=[b"requested"])

        def lookup_chain(self, _hashes):
            return []

        def touch(self, _ids):
            return None

        def num_free(self):
            return len(self.queue)

        def free_queue_ids(self):
            return list(self.queue)

        def free_queue_blocks(self):
            return [self.blocks[i] for i in self.queue]

        def hash_of(self, block_id):
            return self.blocks[block_id].block_hash

        def take_specific(self, ids):
            for bid in ids:
                self.queue.remove(bid)
                self.blocks[bid].ref_cnt += 1
            return list(ids)

        def allocate(self, count):
            return self.take_specific(self.queue[:count])

        def cache(self, _request, ids, _cached):
            for bid in ids:
                self.blocks[bid].block_hash = f"block:{bid}".encode()

        def release(self, ids):
            for bid in ids:
                self.blocks[bid].ref_cnt = 0
                self.queue.append(bid)

    class FakeAsymCoordinator:
        def __init__(self):
            self.coordinator = self

        def select_victims(self, _candidates, preparation, _entries, *, required_blocks, timestamp):
            return tuple(b.block_id for b in preparation.virtual_eligible_order[:required_blocks])

        def pop_pending_decision(self):
            return None

        def observe_chain_access(self, chain_id, timestamp):
            return None

    profile = SimpleNamespace(curves_by_length={})
    monkeypatch.setattr(probe, "VLLMSubstrate", FakeSubstrate)
    monkeypatch.setattr(asym_apc_runner, "load_calibration_profile", lambda *args, **kwargs: profile)
    monkeypatch.setattr(asym_apc_runner, "parse_policy_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        asym_apc_runner, "build_asym_apc_arm",
        lambda **kwargs: (FakeAsymCoordinator(), profile, None),
    )
    original_upsert = RetentionManager.upsert
    observed = []

    def spy_upsert(self, snapshot):
        observed.append(snapshot)
        return original_upsert(self, snapshot)

    monkeypatch.setattr(RetentionManager, "upsert", spy_upsert)

    turns = [probe.Turn(
        sequence_index=0, program="p", prefix_key="p:0",
        token_ids=tuple(range(16)), arrival_gap_seconds=1.0,
    )]
    captured = {}
    for arm in ("a0", "asym_apc"):
        observed.clear()
        result = probe.run_arm(
            arm, turns, num_blocks=8, ttl_seconds=30.0,
            profile_path=tmp_path / "mock.json", profile_sha="test",
            allowed_sources=("native_gpu",), proxy_gain_ms=0.0,
        )
        assert result.requests[0]["status"] == "OK"
        assert observed, f"{arm} failed to register an observed Retention Entry"
        captured[arm] = tuple(
            (entry.protected, tuple(b.block_id for b in entry.block_ids),
             entry.deadline_timestamp)
            for entry in observed
        )
    assert captured["a0"] == captured["asym_apc"]
