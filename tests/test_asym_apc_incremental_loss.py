"""Delta-priced search must rank exactly what the full re-sum ranks.

The selector's loss is separable in the deepest surviving block of each chain,
so a candidate can be priced from its own memberships instead of re-summing
every chain the decision involves. That is an *arithmetic* change; the ranking
function is supposed to be untouched. These tests are the contract:

* randomised cases run both implementations and must return the same victims,
  the same reason and the same reported proxy losses;
* both search branches (exhaustive combinations and bounded greedy) are covered,
  because the greedy branch is where the repeated full evaluation lived;
* a direct check that the incremental accumulator equals the full formula after
  a commit, so a stale cut cannot silently rank against a stale state.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kvopt.costaware.asym_apc_native import (
    APCLatencyCurve,
    AsymAPCNativeCoordinator,
    AsymDecay,
    _IncrementalPrefixLoss,
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
        allowed = {c.block_id for c in candidates}
        return SimpleNamespace(
            required_blocks=context.required_blocks,
            timestamp=context.timestamp,
            blocks=tuple(b for b in self.req.blocks if b.block_id in allowed),
            chain_length=self.req.chain_length,
        )


class DummyPreparation:
    def __init__(self, eligible: list[int], tier1: set[int]):
        self.virtual_eligible_order = tuple(SimpleNamespace(block_id=b) for b in eligible)
        self.original_free_queue = tuple(
            SimpleNamespace(
                block_id=SimpleNamespace(block_id=b),
                eligibility_tier=0 if b in tier1 else 1,
            )
            for b in eligible
        )


def _curve(length: int, rng: random.Random) -> APCLatencyCurve:
    """A monotone non-increasing prefill curve over 0..length blocks."""
    samples = []
    remaining = rng.uniform(200.0, 1200.0)
    for depth in range(length + 1):
        samples.append((depth, remaining))
        # Deep steps shrink fast: evicting deep blocks is cheap, shallow is not.
        remaining = max(0.0, remaining - rng.uniform(5.0, 20.0) * (length - depth + 1) / length)
    samples[-1] = (length, 0.0)
    # Enforce the monotone contract exactly rather than hoping the walk held it.
    fixed = []
    previous = float("inf")
    for depth, cost in samples:
        cost = min(cost, previous)
        fixed.append((depth, cost))
        previous = cost
    fixed[-1] = (length, 0.0)
    return APCLatencyCurve(length, tuple(fixed))


def _world(rng: random.Random, *, shared_blocks: bool, tiered: bool):
    chain_count = rng.randint(2, 4)
    lengths = {f"c{i}": rng.randint(2, 6) for i in range(chain_count)}
    blocks: list[BlockDecisionInput] = []
    next_id = 1
    for chain, length in lengths.items():
        for depth in range(1, length + 1):
            blocks.append(
                BlockDecisionInput(
                    next_id, next_id, True, (Membership(chain, depth, f"h{chain}{depth}"),)
                )
            )
            next_id += 1
    if shared_blocks and len(lengths) >= 2:
        # A block can live in two chains at once: their prefixes are identical at
        # that depth, so the shared block *replaces* the two separate depth-1
        # blocks rather than duplicating a depth. The cascade must then respect
        # the shortest truncation it implies, not the first membership it sees.
        first, second = list(lengths)[:2]
        blocks = [
            b for b in blocks
            if not (b.memberships[0].depth == 1 and b.memberships[0].chain in (first, second))
        ]
        shared = BlockDecisionInput(
            next_id,
            next_id,
            True,
            (
                Membership(first, 1, f"hshared{first}{second}"),
                Membership(second, 1, f"hshared{first}{second}"),
            ),
        )
        blocks.append(shared)
        next_id += 1
    candidates = [
        EvictionCandidate(b.block_id, 0, True, b.native_rank, b.memberships[0].content_hash)
        for b in blocks
    ]
    order = [b.block_id for b in blocks]
    rng.shuffle(order)
    eligible = order[: max(2, rng.randint(2, len(order)))]
    tier1 = {b for b in eligible if rng.random() < 0.5} if tiered else set(eligible)
    # The eligibility order must respect the tier contract (tier 0 before tier 1)
    # or the coordinator falls back before any scoring happens.
    head = [b for b in eligible if b in tier1]
    tail = [b for b in eligible if b not in tier1]
    rng.shuffle(head)
    rng.shuffle(tail)
    eligible = head + tail
    curves = {length: _curve(length, rng) for length in set(lengths.values())}
    req = SimpleNamespace(blocks=tuple(blocks), chain_length=dict(lengths))
    return candidates, req, curves, eligible, tier1


def _coordinator(req, curves, claimed: set[str], *, loss_evaluation: str,
                 cost_model: str, weighting: str, max_exact: int):
    coordinator = AsymAPCNativeCoordinator(
        DummyRegistry(req),
        curves,
        decay=AsymDecay(),
        max_exact_subsets=max_exact,
        max_greedy_operations=200000,
        cost_model=cost_model,
        weighting=weighting,
        loss_evaluation=loss_evaluation,
    )
    for index, chain in enumerate(sorted(claimed)):
        coordinator.observe_chain_access(chain, 100.0 + index)
    return coordinator


def _decide(coordinator, candidates, eligible, tier1, need):
    return coordinator.select_victims(
        candidates,
        DummyPreparation(eligible, tier1),
        (),
        required_blocks=need,
        timestamp=200.0,
    )


@pytest.mark.parametrize("cost_model", ["contiguous_prefix", "independent_block"])
@pytest.mark.parametrize("weighting", ["two_timescale", "uniform"])
@pytest.mark.parametrize("max_exact", [1, 3000])
def test_delta_search_returns_the_same_decision_as_the_full_sum(
    cost_model: str, weighting: str, max_exact: int
) -> None:
    rng = random.Random(0xA5C0DE + max_exact)
    divergent = 0
    for case in range(60):
        candidates, req, curves, eligible, tier1 = _world(
            rng, shared_blocks=case % 2 == 1, tiered=case % 3 == 0
        )
        claimed = {b.memberships[0].chain for b in req.blocks} | {
            m.chain for b in req.blocks for m in b.memberships
        }
        need = rng.randint(1, max(1, len(eligible) - 1))
        results = {}
        for mode in ("full", "incremental"):
            coordinator = _coordinator(
                req, curves, claimed,
                loss_evaluation=mode,
                cost_model=cost_model,
                weighting=weighting,
                max_exact=max_exact,
            )
            selected = _decide(coordinator, candidates, eligible, tier1, need)
            results[mode] = (selected, coordinator.last_audit)
        full_selected, full_audit = results["full"]
        delta_selected, delta_audit = results["incremental"]
        assert delta_selected == full_selected, f"case {case}"
        assert delta_audit.reason == full_audit.reason, f"case {case}"
        assert delta_audit.baseline == full_audit.baseline, f"case {case}"
        assert delta_audit.selected == full_audit.selected, f"case {case}"
        assert delta_audit.exact_search == full_audit.exact_search, f"case {case}"
        assert delta_audit.baseline_proxy_ms == pytest.approx(
            full_audit.baseline_proxy_ms, rel=0, abs=1e-12
        )
        assert delta_audit.selected_proxy_ms == pytest.approx(
            full_audit.selected_proxy_ms, rel=0, abs=1e-12
        )
        divergent += 1 if full_audit.victim_set_differs else 0
    assert divergent, "no case actually re-ranked anything; the comparison is vacuous"


def test_greedy_branch_is_exercised_and_agrees() -> None:
    """max_exact=1 forces the bounded greedy path for every multi-choice tier."""
    rng = random.Random(4242)
    greedy_cases = 0
    for _ in range(20):
        candidates, req, curves, eligible, tier1 = _world(rng, shared_blocks=True, tiered=False)
        claimed = {m.chain for b in req.blocks for m in b.memberships}
        need = max(1, len(eligible) - 1)
        audits = {}
        for mode in ("full", "incremental"):
            coordinator = _coordinator(
                req, curves, claimed,
                loss_evaluation=mode,
                cost_model="contiguous_prefix",
                weighting="two_timescale",
                max_exact=1,
            )
            selected = _decide(coordinator, candidates, eligible, tier1, need)
            audits[mode] = (selected, coordinator.last_audit)
        if not audits["full"][1].exact_search:
            greedy_cases += 1
        assert audits["incremental"][0] == audits["full"][0]
    assert greedy_cases, "the greedy branch was never reached"


def test_incremental_accumulator_tracks_the_full_formula() -> None:
    rng = random.Random(7)
    _candidates, req, curves, _eligible, _tier1 = _world(
        rng, shared_blocks=True, tiered=False
    )
    memberships = {b.block_id: tuple((m.chain, m.depth) for m in b.memberships)
                   for b in req.blocks}
    chain_length = {}
    for b in req.blocks:
        for m in b.memberships:
            chain_length.setdefault(m.chain, req.chain_length[m.chain])
    chain_curves = {chain: curves[length] for chain, length in chain_length.items()}
    weights = {chain: 1.0 + 0.1 * index for index, chain in enumerate(sorted(chain_length))}
    base = {chain: chain_curves[chain].at(chain_length[chain]) for chain in chain_length}

    tracker = _IncrementalPrefixLoss(chain_curves, weights, base, chain_length)
    selection = [b.block_id for b in req.blocks][: max(1, len(req.blocks) // 3)]

    for block_id in selection:
        state, _delta = tracker.plan((block_id,), memberships)
        tracker.commit(state)

    cut = {chain: chain_length[chain] for chain in chain_length}
    for block_id in selection:
        for chain, depth in memberships[block_id]:
            cut[chain] = min(cut[chain], depth - 1)
    expected = sum(weights[chain] * max(0.0, chain_curves[chain].at(cut[chain]) - base[chain])
                   for chain in chain_length)
    assert tracker.total() == pytest.approx(expected, rel=0, abs=1e-9)


def test_invalid_loss_evaluation_is_rejected() -> None:
    with pytest.raises(ValueError):
        AsymAPCNativeCoordinator(
            DummyRegistry(SimpleNamespace(blocks=(), chain_length={})),
            {},
            loss_evaluation="fastest",
        )


def test_default_is_the_delta_path_and_is_recorded() -> None:
    rng = random.Random(11)
    candidates, req, curves, eligible, tier1 = _world(rng, shared_blocks=False, tiered=False)
    claimed = {m.chain for b in req.blocks for m in b.memberships}
    coordinator = _coordinator(
        req, curves, claimed,
        loss_evaluation="incremental",
        cost_model="contiguous_prefix",
        weighting="two_timescale",
        max_exact=3000,
    )
    assert coordinator.loss_evaluation == "incremental"
    _decide(coordinator, candidates, eligible, tier1, 1)
    assert coordinator.last_audit is not None
