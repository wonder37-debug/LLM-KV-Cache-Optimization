"""CPU-only implementation tests against the fork's A2-shaped data contract."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kvopt.costaware.asym_apc_native import (
    APCLatencyCurve, AsymAPCNativeCoordinator, AsymDecay,
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
        self.called = 0

    def build_request(self, candidates, context, retention_entries, *, tau_seconds):
        assert all(c.ref_cnt == 0 for c in candidates)
        assert tau_seconds > 0
        self.called += 1
        return SimpleNamespace(
            required_blocks=context.required_blocks,
            timestamp=context.timestamp,
            blocks=tuple(b for b in self.req.blocks if b.block_id in {c.block_id for c in candidates}),
            chain_length=self.req.chain_length,
        )


class DummyPreparation:
    def __init__(self, eligible: list[int], tier1=None):
        tier1 = set(eligible) if tier1 is None else set(tier1)
        self.virtual_eligible_order = tuple(SimpleNamespace(block_id=b) for b in eligible)
        self.original_free_queue = tuple(SimpleNamespace(
            block_id=SimpleNamespace(block_id=b),
            eligibility_tier=0 if b in tier1 else 1,
        ) for b in eligible)


def build_fixture(*, curve=True, reuse=True, tier1=None, mutate=None):
    candidates = [EvictionCandidate(i, 0, True, i, f'h{i}') for i in range(1, 9)]
    blocks = tuple(BlockDecisionInput(i, i, True, (Membership(
        'A' if i <= 4 else 'B', ((i - 1) % 4) + 1, f'h{i}'),)) for i in range(1, 9))
    if mutate is not None:
        candidates, blocks = mutate(candidates, blocks)
    req = SimpleNamespace(blocks=blocks, chain_length={'A': 4, 'B': 4})
    curves = {4: APCLatencyCurve(4, ((0, 20.0), (1, 14.0), (2, 9.0),
                                    (3, 4.0), (4, 0.0)))} if curve else {}
    coordinator = AsymAPCNativeCoordinator(DummyRegistry(req), curves)
    if reuse:
        coordinator.observe_chain_access('A', 50)
        coordinator.observe_chain_access('B', 60)
    preparation = DummyPreparation(list(range(1, 9)), tier1)
    return candidates, preparation, coordinator


def decide(coordinator, candidates, preparation, need=1, timestamp=100):
    return coordinator.select_victims(
        candidates, preparation, (), required_blocks=need, timestamp=timestamp
    )


def test_contiguous_apc_prefers_cheap_deep_eviction():
    candidates, preparation, coordinator = build_fixture()
    result = decide(coordinator, candidates, preparation)
    assert result == (4,)  # A's deepest block, not A's shallowest LRU block
    audit = coordinator.last_audit
    assert audit.baseline == (1,)
    assert audit.victim_set_differs
    assert audit.reason == 'selected_lower_proxy_loss'
    assert audit.selected_proxy_ms < audit.baseline_proxy_ms
    assert audit.exact_search


def test_strict_tier_priority_even_if_tier2_is_cheaper():
    candidates, _, coordinator = build_fixture()
    prep = DummyPreparation([1, 8], tier1=[1])
    assert decide(coordinator, candidates, prep) == (1,)
    assert not coordinator.last_audit.victim_set_differs


def test_protected_block_is_never_selected():
    candidates, _, coordinator = build_fixture()
    prep = DummyPreparation([1, 2, 3, 4], tier1=[1, 2, 3, 4])
    assert decide(coordinator, candidates, prep) == (4,)
    assert 8 not in coordinator.last_audit.selected


def test_missing_access_timestamp_falls_back_to_a0():
    candidates, preparation, coordinator = build_fixture(reuse=False)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'recency_missing'


def test_missing_profile_falls_back_to_a0():
    candidates, preparation, coordinator = build_fixture(curve=False)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'profile_missing'


def test_nonzero_refcount_falls_back():
    def mutate(candidates, blocks):
        candidates[0] = EvictionCandidate(1, 1, True, 1, 'h1')
        return candidates, blocks
    candidates, preparation, coordinator = build_fixture(mutate=mutate)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'nonzero_refcount'


def test_missing_identity_and_mismatched_content_fall_back():
    def mutate(candidates, blocks):
        blocks = list(blocks)
        blocks[0] = BlockDecisionInput(1, 1, True, (Membership('A', 1, 'wrong'),))
        return candidates, tuple(blocks)
    candidates, preparation, coordinator = build_fixture(mutate=mutate)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'content_identity_mismatch'


def test_future_observations_are_not_permitted():
    candidates, preparation, coordinator = build_fixture()
    coordinator.observe_chain_access('B', 150)
    # A future-dated *unselected* chain is excluded without suppressing
    # the verified A-chain decision. It is never consumed as a feature.
    assert decide(coordinator, candidates, preparation) == (4,)
    assert coordinator.last_audit.reason == 'selected_partial_coverage'
    assert dict(coordinator.last_audit.missing_reasons)['future_observation'] == 4
    with pytest.raises(ValueError):
        coordinator.observe_chain_access('B', 149)


def test_all_blocks_needed_no_choice():
    candidates, preparation, coordinator = build_fixture()
    assert decide(coordinator, candidates, preparation, need=8) == tuple(range(1, 9))
    assert coordinator.last_audit.reason == 'no_choice'


def test_greedy_path_and_search_budget_limit():
    candidates, preparation, coordinator = build_fixture()
    coordinator._max_exact = 1
    assert decide(coordinator, candidates, preparation, need=2) == (4, 8)
    assert not coordinator.last_audit.exact_search
    coordinator._max_ops = 1
    assert decide(coordinator, candidates, preparation, need=2) == (1, 2)
    assert coordinator.last_audit.reason == 'search_budget_exceeded'


def test_cost_curve_and_decay_contracts():
    curve = APCLatencyCurve(4, ((0, 20.0), (2, 9.0), (4, 0.0)))
    assert curve.at(1) == 14.5
    assert curve.at(3) == 4.5
    with pytest.raises(ValueError):
        APCLatencyCurve(2, ((0, 3.0), (1, 4.0), (2, 0.0)))
    with pytest.raises(ValueError):
        APCLatencyCurve(3, ((0, 9.0), (2, 1.0)))
    model = AsymDecay(alpha_seconds=60, beta_seconds=12, knee_seconds=60)
    assert model.weight(0) == 1
    assert model.weight(110) < model.weight(20)


def test_hashless_block_is_zero_cost_and_never_assumed_hashed():
    def mutate(candidates, blocks):
        candidates[0] = EvictionCandidate(1, 0, False, 1, None)
        blocks = list(blocks)
        blocks[0] = BlockDecisionInput(1, 1, False, ())
        return candidates, tuple(blocks)
    candidates, preparation, coordinator = build_fixture(mutate=mutate)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'baseline_no_verified_proxy_gain'


def test_shared_block_penalty_counts_all_chains_once():
    candidates = [EvictionCandidate(i, 0, True, i, f'h{i}') for i in (1, 2, 3)]
    blocks = (
        BlockDecisionInput(1, 1, True, (Membership('A', 1, 'h1'),
                                         Membership('B', 1, 'h1'))),
        BlockDecisionInput(2, 2, True, (Membership('A', 4, 'h2'),)),
        BlockDecisionInput(3, 3, True, (Membership('B', 4, 'h3'),)),
    )
    registry = DummyRegistry(SimpleNamespace(blocks=blocks, chain_length={'A': 4, 'B': 4}))
    curve = APCLatencyCurve(4, ((0, 20.0), (4, 0.0)))
    coordinator = AsymAPCNativeCoordinator(registry, {4: curve})
    coordinator.observe_chain_access('A', 50)
    coordinator.observe_chain_access('B', 50)
    prep = DummyPreparation([1, 2, 3])
    selected = decide(coordinator, candidates, prep)
    assert selected in ((2,), (3,))
    assert coordinator.last_audit.selected_proxy_ms < coordinator.last_audit.baseline_proxy_ms


def test_no_beneficial_divergence_defaults_to_lru():
    candidates, preparation, coordinator = build_fixture()
    # Place the lowest-loss deep block first in native order.
    order = [4, 1, 2, 3, 5, 6, 7, 8]
    re_ranked = [EvictionCandidate(i, 0, True, rank, f'h{i}')
                 for rank, i in enumerate(order)]
    prep = DummyPreparation(order)
    # A2-like registry factory validates native ranks, so regenerate inputs.
    old = coordinator._registry.req.blocks
    new = tuple(BlockDecisionInput(b.block_id, order.index(b.block_id), True,
                                   b.memberships) for b in old)
    coordinator._registry.req = SimpleNamespace(blocks=new, chain_length={'A': 4, 'B': 4})
    assert decide(coordinator, re_ranked, prep) == (4,)
    assert coordinator.last_audit.reason == 'baseline_no_verified_proxy_gain'


def test_broken_prefix_uses_original_length_curve_for_incremental_cost():
    # Current reusable depth is 2, while the actual original chain has length 4.
    # We must price the change F(total=4, cached=0/1)-F(total=4, cached=2).
    candidate = [EvictionCandidate(i, 0, True, i, f'h{i}') for i in (1, 2)]
    blocks = tuple(BlockDecisionInput(i, i, True, (Membership('A', i, f'h{i}'),))
                   for i in (1, 2))
    registry = DummyRegistry(SimpleNamespace(blocks=blocks, chain_length={'A': 2}))
    registry.snapshot = lambda c: SimpleNamespace(block_ids=(1, 2, 3, 4))
    curve = APCLatencyCurve(4, ((0, 20.0), (1, 14.0), (2, 9.0),
                                (3, 4.0), (4, 0.0)))
    coordinator = AsymAPCNativeCoordinator(registry, {4: curve})
    coordinator.observe_chain_access('A', 50)
    assert decide(coordinator, candidate, DummyPreparation([1, 2])) == (2,)
    weight = coordinator._decay.weight(50)
    assert coordinator.last_audit.baseline_proxy_ms == pytest.approx(11.0 * weight)
    assert coordinator.last_audit.selected_proxy_ms == pytest.approx(5.0 * weight)


def test_no_content_hash_reverts_to_true_a0_order():
    def mutate(candidates, blocks):
        candidates[0] = EvictionCandidate(1, 0, True, 1, None)
        return candidates, blocks
    candidates, preparation, coordinator = build_fixture(mutate=mutate)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'content_unverified'


def test_rejected_tier_order_cannot_use_fake_eligibility():
    candidates, _, coordinator = build_fixture()
    prep = DummyPreparation([8, 1], tier1=[1])
    assert decide(coordinator, candidates, prep) == (8,)
    assert coordinator.last_audit.reason == 'tier_order_mismatch'


# Missing evidence in an *unselected* candidate no longer poisons an entire
# decision. The unknown candidate is never promoted to a new victim.
def test_unverified_competitor_outside_baseline_does_not_force_global_fallback():
    candidates, preparation, coordinator = build_fixture()
    candidates[-1] = EvictionCandidate(8, 0, True, 8, None)
    assert decide(coordinator, candidates, preparation) == (4,)
    assert coordinator.last_audit.reason == 'selected_partial_coverage'
    assert dict(coordinator.last_audit.missing_reasons)['content_unverified'] == 1
    assert 8 not in coordinator.last_audit.selected


def test_missing_competitor_recency_does_not_force_global_fallback():
    candidates, preparation, coordinator = build_fixture()
    coordinator._last_access.pop('B')
    assert decide(coordinator, candidates, preparation) == (4,)
    assert dict(coordinator.last_audit.missing_reasons)['recency_missing'] == 4


def test_missing_competitor_profile_does_not_force_global_fallback():
    candidates, preparation, coordinator = build_fixture()
    coordinator._registry.req.chain_length['B'] = 5
    assert decide(coordinator, candidates, preparation) == (4,)
    assert coordinator.last_audit.reason == 'selected_partial_coverage'
    assert dict(coordinator.last_audit.missing_reasons)['profile_missing'] == 4


def test_unknown_baseline_victim_is_never_replaced_or_priced_as_free():
    candidates, preparation, coordinator = build_fixture()
    candidates[0] = EvictionCandidate(1, 0, True, 1, None)
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'content_unverified'
    assert coordinator.last_audit.baseline_proxy_ms is None


def test_registry_monotonic_access_observation_replaces_manual_recency():
    candidates, preparation, coordinator = build_fixture(reuse=False)
    snapshots = {
        chain: SimpleNamespace(native_hash_process_id='proc',
                               content_hashes=('a', 'b'), block_ids=(0, 1, 2, 3))
        for chain in ('A', 'B')
    }
    coordinator._registry.snapshot = lambda chain: snapshots[chain]
    coordinator._registry.last_access_observation = lambda chain: SimpleNamespace(
        timestamp=50 if chain == 'A' else 60,
        clock_domain='system_monotonic',
        native_hash_process_id='proc',
        content_hashes=('a', 'b'),
    )
    assert decide(coordinator, candidates, preparation) == (4,)
    assert coordinator.last_audit.reason == 'selected_lower_proxy_loss'
    assert coordinator.last_audit.missing_reasons == ()


def test_registry_observation_wrong_clock_cannot_be_trusted():
    candidates, preparation, coordinator = build_fixture(reuse=False)
    coordinator._registry.snapshot = lambda chain: SimpleNamespace(
        native_hash_process_id='proc', content_hashes=('a',), block_ids=(1, 2, 3, 4))
    coordinator._registry.last_access_observation = lambda chain: SimpleNamespace(
        timestamp=60, clock_domain='replay_fake', native_hash_process_id='proc',
        content_hashes=('a',))
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'recency_missing'


def test_registry_observation_old_content_generation_is_rejected():
    candidates, preparation, coordinator = build_fixture(reuse=False)
    coordinator._registry.snapshot = lambda chain: SimpleNamespace(
        native_hash_process_id='new', content_hashes=('a',), block_ids=(1, 2, 3, 4))
    coordinator._registry.last_access_observation = lambda chain: SimpleNamespace(
        timestamp=60, clock_domain='system_monotonic', native_hash_process_id='old',
        content_hashes=('a',))
    assert decide(coordinator, candidates, preparation) == (1,)
    assert coordinator.last_audit.reason == 'recency_missing'
