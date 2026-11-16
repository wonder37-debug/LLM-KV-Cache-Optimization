"""Ablation contract tests: cost model and recency weighting must be separable.

The point of these tests is that the ablation is not cosmetic. Each variant is
checked against a scenario where the two scoring functions must disagree, and a
variant that does not consume time must not be blocked on time evidence it never
uses -- otherwise its fallback rate would differ for an unrelated reason.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kvopt.costaware.asym_apc_native import (
    APCLatencyCurve,
    AsymAPCNativeCoordinator,
    AsymDecay,
)
from kvopt.runtime.vllm.types import EvictionCandidate


@dataclass(frozen=True)
class Membership:
    chain: str
    depth: int
    content_hash: str


@dataclass(frozen=True)
class BlockDecisionInput:
    block_id: int
    native_rank: int
    has_block_hash: bool
    memberships: tuple[Membership, ...]


class DummyRegistry:
    def __init__(self, req):
        self.req = req

    def build_request(self, candidates, context, retention_entries, *, tau_seconds):
        return SimpleNamespace(
            required_blocks=context.required_blocks,
            timestamp=context.timestamp,
            blocks=tuple(
                b for b in self.req.blocks if b.block_id in {c.block_id for c in candidates}
            ),
            chain_length=self.req.chain_length,
        )


class DummyPreparation:
    def __init__(self, eligible: list[int]):
        self.virtual_eligible_order = tuple(SimpleNamespace(block_id=b) for b in eligible)
        self.original_free_queue = tuple(
            SimpleNamespace(
                block_id=SimpleNamespace(block_id=b), eligibility_tier=0
            )
            for b in eligible
        )


# A deliberately kinked (but monotone) curve. The contiguous model prices the
# *truncation point* F(d-1), the independent model prices the *marginal step*
# F(d-1)-F(d). Here the first step is almost free (30 -> 29) while the second is
# huge (29 -> 5), so the two models must rank different depths as cheapest:
#   truncation point: d4 (3 ms)   marginal step: d1 (1 ms)
CURVE = APCLatencyCurve(4, ((0, 30.0), (1, 29.0), (2, 5.0), (3, 3.0), (4, 0.0)))


def _fixture(*, cost_model: str, weighting: str, decide_depth: bool = True):
    """Two chains of four blocks each; ids 1-4 are chain A, 5-8 are chain B."""
    candidates = [EvictionCandidate(i, 0, True, i, f"h{i}") for i in range(1, 9)]
    blocks = tuple(
        BlockDecisionInput(
            i,
            i,
            True,
            (Membership("A" if i <= 4 else "B", ((i - 1) % 4) + 1, f"h{i}"),),
        )
        for i in range(1, 9)
    )
    # Eligible order puts the two shallow blocks first, so A0's baseline is the
    # expensive end and a correct selector has to move away from it.
    eligible = [1, 5, 2, 6, 3, 7, 4, 8]
    req = SimpleNamespace(blocks=blocks, chain_length={"A": 4, "B": 4})
    coordinator = AsymAPCNativeCoordinator(
        DummyRegistry(req),
        {4: CURVE},
        cost_model=cost_model,
        weighting=weighting,
    )
    if decide_depth:
        coordinator.observe_chain_access("A", 50.0)
        coordinator.observe_chain_access("B", 60.0)
    return candidates, DummyPreparation(eligible), coordinator


def _decide(coordinator, candidates, preparation, need=1, timestamp=100.0):
    return coordinator.select_victims(
        candidates, preparation, (), required_blocks=need, timestamp=timestamp
    )


def test_rejects_unknown_cost_model_and_weighting():
    with pytest.raises(ValueError, match="cost_model must be one of"):
        AsymAPCNativeCoordinator(DummyRegistry(SimpleNamespace(blocks=(), chain_length={})), {},
                                 cost_model="magic")
    with pytest.raises(ValueError, match="weighting must be one of"):
        AsymAPCNativeCoordinator(DummyRegistry(SimpleNamespace(blocks=(), chain_length={})), {},
                                 weighting="magic")


def test_cost_models_disagree_on_which_depth_is_cheapest():
    """Position sensitivity must actually change the ranking, not just numbers."""
    contiguous_candidates, contiguous_prep, contiguous = _fixture(
        cost_model="contiguous_prefix", weighting="uniform"
    )
    independent_candidates, independent_prep, independent = _fixture(
        cost_model="independent_block", weighting="uniform"
    )

    contiguous_pick = _decide(contiguous, contiguous_candidates, contiguous_prep)
    independent_pick = _decide(independent, independent_candidates, independent_prep)

    # Contiguous: truncating at depth 4 keeps 3 blocks cached, so the deep block
    # is the cheap victim. Independent: the marginal step at depth 4 is the
    # largest single step, so a shallow block is cheaper per victim.
    assert contiguous_pick == (4,)
    assert independent_pick == (1,)
    assert contiguous.last_audit is not None and independent.last_audit is not None
    assert contiguous.last_audit.selected_proxy_ms != independent.last_audit.selected_proxy_ms


def test_uniform_weighting_ignores_recency_and_two_timescale_does_not():
    """A stale shallow block must be cheap for the recency term, not for depth."""
    # Chain A was observed recently, chain B long ago (age 240s).
    candidates, preparation, two_timescale = _fixture(
        cost_model="contiguous_prefix", weighting="two_timescale"
    )
    _, _, uniform = _fixture(cost_model="contiguous_prefix", weighting="uniform")
    # Chain A is touched at t=200; chain B was last seen at t=60 (140 s idle).
    for coordinator in (two_timescale, uniform):
        coordinator.observe_chain_access("A", 200.0)

    stale_pick = _decide(two_timescale, candidates, preparation, timestamp=200.0)
    uniform_pick = _decide(uniform, candidates, preparation, timestamp=200.0)

    # Depth alone cannot separate the two chains: both offer an equally deep
    # block (3 ms of truncation). Only the recency term prefers the chain that
    # has been idle for 140 s (block 8) over the one touched just now (block 4).
    assert stale_pick == (8,)
    assert uniform_pick == (4,)


def test_uniform_weighting_does_not_require_time_evidence():
    """The depth-only arm must not inherit recency's fallback rate."""
    candidates, preparation, coordinator = _fixture(
        cost_model="contiguous_prefix", weighting="uniform", decide_depth=False
    )
    selected = _decide(coordinator, candidates, preparation)
    assert selected
    assert coordinator.last_audit is not None
    assert coordinator.last_audit.reason != "recency_missing"

    candidates, preparation, blocked = _fixture(
        cost_model="contiguous_prefix", weighting="two_timescale", decide_depth=False
    )
    blocked_selected = _decide(blocked, candidates, preparation)
    assert blocked.last_audit is not None
    assert blocked.last_audit.reason == "recency_missing"
    # The fallback must be the exact A0 order, never a guess.
    assert blocked_selected == tuple(range(1, 1 + len(blocked_selected)))


def test_selection_matches_brute_force_minimum_for_both_cost_models():
    """The reported metric must be the minimum of that model's own loss."""
    import itertools

    for cost_model in ("contiguous_prefix", "independent_block"):
        candidates, preparation, coordinator = _fixture(
            cost_model=cost_model, weighting="two_timescale"
        )
        selected = _decide(coordinator, candidates, preparation, need=2)
        audit = coordinator.last_audit
        assert audit is not None and audit.selected_proxy_ms is not None

        depths = {1: ("A", 1), 2: ("A", 2), 3: ("A", 3), 4: ("A", 4),
                  5: ("B", 1), 6: ("B", 2), 7: ("B", 3), 8: ("B", 4)}

        def model_loss(ids):
            weight = {"A": AsymDecay().weight(50.0), "B": AsymDecay().weight(40.0)}
            if cost_model == "independent_block":
                return sum(
                    weight[depths[i][0]]
                    * (CURVE.at(depths[i][1] - 1) - CURVE.at(depths[i][1]))
                    for i in ids
                )
            truncation = {"A": 4, "B": 4}
            for i in ids:
                chain, depth = depths[i]
                truncation[chain] = min(truncation[chain], depth - 1)
            return sum(weight[c] * (CURVE.at(truncation[c]) - CURVE.at(4)) for c in truncation)

        best = min(
            model_loss(choice) for choice in itertools.combinations(depths, 2)
        )
        assert audit.selected_proxy_ms == pytest.approx(best, rel=1e-9), cost_model
        assert set(selected).issubset(depths)
        assert len(selected) == 2
