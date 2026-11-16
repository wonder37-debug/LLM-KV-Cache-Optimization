"""Workload-family contract tests.

Two properties matter for the evaluation:

1. The reference family must reproduce the original workload's geometry, so a
   new run can be compared with the corrected CI replay.
2. For families that claim to vary cache competition, a different seed must
   actually change the *structure* (order, lengths, gaps, revisits), not just
   the token values. ``geometry_signature`` makes that checkable.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES = REPO_ROOT / "scripts" / "probes" / "asym_apc_native"


def _load(name: str):
    if str(PROBES) not in sys.path:
        sys.path.insert(0, str(PROBES))
    spec = importlib.util.spec_from_file_location(name, PROBES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


families = _load("workload_families")


def test_reference_family_reproduces_the_original_geometry():
    """The regression family must be structurally identical to the old probe."""
    probe = _load("real_blockpool_eval")
    legacy = probe.WorkloadSpec(
        programs=8,
        turns_per_program=5,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        short_gap_seconds=2.0,
        long_gap_seconds=90.0,
        seed=20260101,
        target_blocks=families.LEGACY_LENGTHS,
    ).build()
    family = families.build_turns(families.FAMILIES["legacy_round_robin"], 20260101)

    legacy_shape = [
        (turn.sequence_index, turn.program, turn.prompt_blocks, round(turn.arrival_gap_seconds, 3))
        for turn in legacy
    ]
    family_shape = [
        (turn.sequence_index, turn.program, turn.prompt_blocks, round(turn.arrival_gap_seconds, 3))
        for turn in family
    ]
    assert family_shape == legacy_shape
    assert families.geometry_signature(family) == families.geometry_signature(legacy)


@pytest.mark.parametrize(
    "name",
    ["agent_random_arrival", "tool_gap_mix", "zipf_revisit", "unpredictable_revisit"],
)
def test_order_sensitive_families_change_geometry_across_seeds(name):
    spec = families.FAMILIES[name]
    signatures = {families.geometry_signature(families.build_turns(spec, seed)) for seed in (1, 2, 3)}
    assert len(signatures) == 3, f"{name} produced identical geometry for different seeds"


def test_reference_family_geometry_is_seed_invariant_which_is_the_old_limitation():
    """Documents why the original seed axis was not a replication."""
    spec = families.FAMILIES["legacy_round_robin"]
    signatures = {families.geometry_signature(families.build_turns(spec, seed)) for seed in (1, 2, 3)}
    assert len(signatures) == 1


def test_build_is_deterministic_for_a_fixed_seed():
    spec = families.FAMILIES["unpredictable_revisit"]
    first = families.build_turns(spec, 7)
    second = families.build_turns(spec, 7)
    assert [t.token_ids for t in first] == [t.token_ids for t in second]
    assert [t.arrival_gap_seconds for t in first] == [t.arrival_gap_seconds for t in second]


def test_zipf_arrival_skews_revisit_frequency():
    spec = families.FAMILIES["zipf_revisit"]
    counts = Counter(turn.program for turn in families.build_turns(spec, 11))
    assert len(counts) == spec.programs
    assert max(counts.values()) > min(counts.values())


def test_unpredictable_revisit_re_reads_earlier_depths_but_stays_prefix_shaped():
    spec = families.FAMILIES["unpredictable_revisit"]
    turns = families.build_turns(spec, 5)
    assert any(turn.revisit for turn in turns)

    by_program: dict[str, list] = {}
    for turn in turns:
        by_program.setdefault(turn.program, []).append(turn)
    for program_turns in by_program.values():
        longest = max(program_turns, key=lambda t: t.prompt_tokens)
        for turn in program_turns:
            # Every request must be an exact prefix of the program's stream,
            # otherwise "cache hit" would stop meaning "prefix reused".
            assert longest.token_ids[: turn.prompt_tokens] == turn.token_ids, program_turns[0].program


@pytest.mark.parametrize("name", families.family_names())
def test_every_prompt_is_block_aligned(name):
    turns = families.build_turns(families.FAMILIES[name], 3)
    assert turns
    for turn in turns:
        assert turn.prompt_tokens % families.BLOCK_TOKENS == 0
        assert turn.prompt_blocks >= families.FAMILIES[name].program_prefix_blocks


def test_tool_gap_mix_uses_more_than_two_gaps():
    turns = families.build_turns(families.FAMILIES["tool_gap_mix"], 3)
    assert len({turn.arrival_gap_seconds for turn in turns}) > 2
