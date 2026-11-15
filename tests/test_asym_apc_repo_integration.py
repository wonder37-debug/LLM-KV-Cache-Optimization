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
