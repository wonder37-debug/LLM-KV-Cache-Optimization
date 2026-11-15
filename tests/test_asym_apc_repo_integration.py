"""True in-repository coordinator/Continuum/A2 registry integration smoke (CPU).

This test imports the existing repository implementation, not a mocked bridge.
The standalone local sandbox lacks the full project checkout and skips it;
run it on the fork branch to exercise the actual integration end-to-end.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("kvopt.costaware.a2_native")

from kvopt.continuum import FakeClock, RetentionMode
from kvopt.continuum.retention import RetentionManager
from kvopt.costaware.a2_native import A2NativeChainSnapshot, A2NativeSnapshotRegistry
from kvopt.costaware.asym_apc_native import APCLatencyCurve, AsymAPCNativeCoordinator
from kvopt.runtime.vllm.continuum_observation import encode_native_hash
from kvopt.runtime.vllm.retention_integration import RetentionRuntimeIntegration


def test_existing_continuum_pressure_and_a2_registry_accept_new_policy():
    blocks = tuple(SimpleNamespace(block_id=i, ref_cnt=0,
                                  block_hash=f'content-{i}'.encode())
                   for i in range(1, 9))
    registry = A2NativeSnapshotRegistry()
    for chain, subset in [('A', blocks[:4]), ('B', blocks[4:])]:
        registry.observe(A2NativeChainSnapshot(
            chain_id=chain,
            block_ids=tuple(b.block_id for b in subset),
            content_hashes=tuple(encode_native_hash(b.block_hash) for b in subset),
            native_hash_process_id='simulation-process',
        ))
    policy = AsymAPCNativeCoordinator(
        registry,
        {4: APCLatencyCurve(4, ((0,20.0),(1,14.0),(2,9.0),
                                (3,4.0),(4,0.0)))},
    )
    policy.observe_chain_access('A', 50)
    policy.observe_chain_access('B', 60)
    manager = RetentionManager(FakeClock(100.0))
    runtime = RetentionRuntimeIntegration(
        manager, block_selection_coordinator=policy,
    )
    physical_removed = []
    result = runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED,
        blocks=blocks,
        required_blocks=1,
        timestamp=100.0,
        remove_selected=lambda ids: physical_removed.extend(ids),
    )
    assert result.selected_block_ids == (4,)
    assert physical_removed == [4]
    assert policy.last_audit.baseline == (1,)
    assert policy.last_audit.victim_set_differs
    assert result.selection_plan is None  # A0-only SelectionPlan is intentionally not forged


def test_registry_observed_access_is_bound_to_content_generation():
    registry = A2NativeSnapshotRegistry()
    original = A2NativeChainSnapshot(
        chain_id='agent:prefix', block_ids=(1, 2),
        content_hashes=('hash-1', 'hash-2'), native_hash_process_id='proc',
    )
    registry.observe(original, observed_at=50.0, clock_domain='system_monotonic')
    assert registry.last_access_observation('agent:prefix').timestamp == 50.0
    with pytest.raises(ValueError, match='nondecreasing'):
        registry.observe(original, observed_at=49.0)
    with pytest.raises(ValueError, match='finite'):
        registry.observe(original, observed_at=float('nan'))

    # Same physical slots may be recycled for different Native hashes.
    registry.observe(A2NativeChainSnapshot(
        chain_id='agent:prefix', block_ids=(1, 2),
        content_hashes=('new-hash', 'new-hash-2'), native_hash_process_id='proc',
    ))
    assert registry.last_access_observation('agent:prefix') is None


def test_real_registry_prefix_timestamp_is_consumed_without_manual_clock_hook():
    blocks = tuple(SimpleNamespace(block_id=i, ref_cnt=0,
                                   block_hash=f'live-hash-{i}'.encode())
                   for i in range(1, 9))
    registry = A2NativeSnapshotRegistry()
    for chain, subset, access in [('A', blocks[:4], 50), ('B', blocks[4:], 60)]:
        registry.observe(A2NativeChainSnapshot(
            chain_id=chain, block_ids=tuple(b.block_id for b in subset),
            content_hashes=tuple(encode_native_hash(b.block_hash) for b in subset),
            native_hash_process_id='simulation-process',
        ), observed_at=access, clock_domain='system_monotonic')
    policy = AsymAPCNativeCoordinator(
        registry,
        {4: APCLatencyCurve(4, ((0, 20.0), (1, 14.0), (2, 9.0),
                                (3, 4.0), (4, 0.0)))},
    )
    manager = RetentionManager(FakeClock(100.0))
    runtime = RetentionRuntimeIntegration(manager, block_selection_coordinator=policy)
    result = runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED, blocks=blocks, required_blocks=1,
        timestamp=100.0, remove_selected=lambda ids: None,
    )
    assert result.selected_block_ids == (4,)
    assert policy.last_audit.reason == 'selected_lower_proxy_loss'
    assert policy.last_audit.missing_reasons == ()


def test_verified_live_chain_survives_unrelated_unobserved_hashed_queue_block():
    blocks = tuple(SimpleNamespace(block_id=i, ref_cnt=0,
                                   block_hash=f'live-hash-{i}'.encode())
                   for i in range(1, 9))
    registry = A2NativeSnapshotRegistry()
    registry.observe(A2NativeChainSnapshot(
        chain_id='A', block_ids=(1, 2, 3, 4),
        content_hashes=tuple(encode_native_hash(b.block_hash) for b in blocks[:4]),
        native_hash_process_id='simulation-process',
    ), observed_at=50.0)
    registry.observe(A2NativeChainSnapshot(
        chain_id='B', block_ids=(5, 6, 7),
        content_hashes=tuple(encode_native_hash(b.block_hash) for b in blocks[4:7]),
        native_hash_process_id='simulation-process',
    ), observed_at=60.0)
    policy = AsymAPCNativeCoordinator(
        registry,
        {4: APCLatencyCurve(4, ((0, 20.0), (4, 0.0))),
         3: APCLatencyCurve(3, ((0, 15.0), (3, 0.0)))},
    )
    runtime = RetentionRuntimeIntegration(
        RetentionManager(FakeClock(100.0)), block_selection_coordinator=policy)
    selection = runtime.apply_pressure(
        mode=RetentionMode.CONTROLLED, blocks=blocks, required_blocks=1,
        timestamp=100.0, remove_selected=lambda ids: None)
    assert selection.selected_block_ids == (4,)
    assert policy.last_audit.reason == 'selected_partial_coverage'
    assert dict(policy.last_audit.missing_reasons)['content_unverified'] == 1
    assert 8 not in selection.selected_block_ids
