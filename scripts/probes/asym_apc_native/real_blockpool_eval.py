"""A/B/C evaluation of APC victim selection on the REAL vLLM BlockPool.

Why this exists
---------------
The project's native runner needs a CUDA device. This host has none, so the
Gated ``ASYM_APC`` arm cannot be exercised through the serving engine here.
What *is* available is vLLM's own cache substrate: the ``BlockPool`` class from
the pinned vLLM release, imported and executed unmodified. Prefix hashes,
reference counts, the free queue, cache lookups and physical prefix-cache
eviction are all real vLLM code, not a simulator.

So this probe answers a narrower question than the full serving experiment, but
it answers it with real state changes:

    Given identical prompts, identical cache capacity, identical Continuum
    retention state and identical arrival timing, does choosing the victim set
    by contiguous-prefix loss change how many prompt tokens must be recomputed,
    compared with the order the native free queue actually evicts in?

What is real here
-----------------
* ``vllm.v1.core.block_pool.BlockPool`` from the pinned vLLM release.
* Block hashes computed by ``hash_block_tokens`` over real token ID sequences,
  seeded exactly as the engine seeds them.
* ``cache_full_blocks`` / ``get_cached_block`` / ``touch`` / ``free_blocks`` /
  ``evict_blocks``: the same entry points the scheduler uses.
* Prefix hits are read back from the pool, so "recomputed tokens" is a
  consequence of what the pool still holds, not an assumption.
* The ``ASYM_APC`` decision path is the project's own coordinator, including its
  evidence, legality and fallback rules.

What is NOT real here
---------------------
* No GPU kernel runs, so no measured prefill *latency* is produced by this
  probe. Latency is reported as a model estimate from a separately measured
  calibration curve, and the calibration's measurement source is carried into
  every record.
* No serving engine, scheduler or sampling loop is involved. Requests are
  issued sequentially by this harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BLOCK_TOKENS = 16
HASH_FN_NAME = "sha256_cbor"
# Chain identity convention used by the retention table and the A2 registry:
# one append-only prefix per program, so the key stays stable as the chain grows.
PREFIX_KEY = "prefix"


# ---------------------------------------------------------------------------
# Real vLLM substrate
# ---------------------------------------------------------------------------


class VLLMSubstrate:
    """Thin wrapper over the real vLLM BlockPool used by this probe."""

    def __init__(self, num_blocks: int) -> None:
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_utils import init_none_hash

        self._hash_fn = get_hash_fn_by_name(HASH_FN_NAME)
        # The engine calls this at startup; it seeds the chain-hash root.
        init_none_hash(self._hash_fn)
        self.pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=BLOCK_TOKENS,
        )

    # -- request construction -------------------------------------------------
    def request(self, request_id: str, token_ids: list[int]):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.kv_cache_utils import hash_block_tokens
        from vllm.v1.request import Request

        hash_fn = self._hash_fn

        def chain_hasher(req: Any) -> list[Any]:
            hashes: list[Any] = []
            parent = None
            ids = req.prompt_token_ids or []
            for start in range(0, len(ids) - BLOCK_TOKENS + 1, BLOCK_TOKENS):
                parent = hash_block_tokens(hash_fn, parent, ids[start : start + BLOCK_TOKENS])
                hashes.append(parent)
            return hashes

        return Request(
            request_id,
            list(token_ids),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=chain_hasher,
        )

    # -- observation ----------------------------------------------------------
    def free_queue_ids(self) -> list[int]:
        """Free blocks in the exact order the native allocator would take them."""
        queue = self.pool.free_block_queue
        tail = queue.fake_free_list_tail
        node = queue.fake_free_list_head.next_free_block
        out: list[int] = []
        while node is not None and node is not tail:
            out.append(node.block_id)
            node = node.next_free_block
        return out

    def free_queue_blocks(self) -> list[Any]:
        return [self.pool.blocks[block_id] for block_id in self.free_queue_ids()]

    def hash_of(self, block_id: int) -> bytes | None:
        return self.pool.blocks[block_id].block_hash

    def num_free(self) -> int:
        return self.pool.get_num_free_blocks()

    def lookup_chain(self, hashes: list[Any]) -> list[int]:
        """Walk the prefix chain exactly as the scheduler does."""
        hits: list[int] = []
        for block_hash in hashes:
            got = self.pool.get_cached_block(block_hash, [0])
            if got is None:
                break
            hits.append(got[0].block_id)
        return hits

    # -- mutation -------------------------------------------------------------
    def touch(self, block_ids: list[int]) -> None:
        if block_ids:
            self.pool.touch([self.pool.blocks[block_id] for block_id in block_ids])

    def allocate(self, count: int) -> list[int]:
        return [block.block_id for block in self.pool.get_new_blocks(count)]

    def cache(self, request: Any, block_ids: list[int], num_cached: int) -> None:
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id  # noqa: F401

        self.pool.cache_full_blocks(
            request,
            [self.pool.blocks[block_id] for block_id in block_ids],
            num_cached,
            len(block_ids),
            BLOCK_TOKENS,
            0,
        )

    def release(self, block_ids: list[int]) -> None:
        self.pool.free_blocks([self.pool.blocks[block_id] for block_id in block_ids])

    def evict(self, block_ids: list[int]) -> list[int]:
        """Physically evict cached entries; return the IDs that actually held a hash."""
        present = [block_id for block_id in block_ids if self.hash_of(block_id) is not None]
        if present:
            self.pool.evict_blocks(set(present))
        return present

    def take_specific(self, block_ids: list[int]) -> list[int]:
        """Hand chosen victim blocks to allocation instead of the queue head.

        The native allocator pops the free-queue head and drops whatever cache
        entry that block happened to hold. To let a Block-level policy influence
        the victim set, the selected blocks are removed from the same free queue
        and then allocated, mirroring ``popleft_n`` + ``get_new_blocks``:
        remove from queue, drop the cache entry, raise the reference count.
        """
        taken: list[int] = []
        for block_id in block_ids:
            block = self.pool.blocks[block_id]
            if block.is_null or block.ref_cnt != 0:
                continue
            self.pool.free_block_queue.remove(block)
            self.pool.evict_blocks({block_id})
            block.ref_cnt += 1
            taken.append(block_id)
        return taken

    def resident_snapshot(self) -> dict[str, object]:
        hashed = [b for b in self.pool.blocks if b.block_hash is not None]
        return {
            "num_free_blocks": self.num_free(),
            "cached_block_count": len(hashed),
            "cached_block_ids": sorted(b.block_id for b in hashed),
            "free_queue_order": self.free_queue_ids(),
        }


def require_vllm() -> None:
    try:
        import vllm  # noqa: F401
    except Exception as error:  # pragma: no cover - environment guard
        raise SystemExit(
            f"real vLLM BlockPool is unavailable in this interpreter: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    sequence_index: int
    program: str
    prefix_key: str
    token_ids: tuple[int, ...]
    arrival_gap_seconds: float

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def prompt_blocks(self) -> int:
        return len(self.token_ids) // BLOCK_TOKENS


@dataclass(slots=True)
class WorkloadSpec:
    programs: int
    turns_per_program: int
    shared_prefix_blocks: int
    program_prefix_blocks: int
    short_gap_seconds: float
    long_gap_seconds: float
    seed: int
    target_blocks: tuple[int, ...] = (24, 32, 48, 64, 64)

    def build(self) -> list[Turn]:
        """Append-only agent turns: each turn extends the previous prompt.

        That is the reuse pattern APC exists for. Turn ``s+1`` of a program has
        turn ``s`` as an exact prefix, so a healthy cache should serve almost all
        of it and only the appended blocks should be recomputed.
        """
        rng = random.Random(self.seed)

        def tokens(count: int) -> list[int]:
            return [rng.randrange(1000, 150000) for _ in range(count * BLOCK_TOKENS)]

        shared = tokens(self.shared_prefix_blocks)
        program_context = {
            f"program-{index}": tokens(self.program_prefix_blocks)
            for index in range(self.programs)
        }
        # One long append-only stream per program: later turns take a longer
        # prefix of the same stream, which is what makes prefix reuse real.
        history = {
            program: tokens(max(self.target_blocks))
            for program in program_context
        }
        turns: list[Turn] = []
        sequence = 0
        for step in range(self.turns_per_program):
            target = self.target_blocks[min(step, len(self.target_blocks) - 1)]
            for index in range(self.programs):
                program = f"program-{index}"
                base = shared + program_context[program]
                base_blocks = len(base) // BLOCK_TOKENS
                history_blocks = max(0, target - base_blocks)
                body = history[program][: history_blocks * BLOCK_TOKENS]
                turns.append(
                    Turn(
                        sequence_index=sequence,
                        program=program,
                        prefix_key=f"{program}:step{step}",
                        token_ids=tuple(base + body),
                        arrival_gap_seconds=(
                            self.short_gap_seconds if step % 2 == 0 else self.long_gap_seconds
                        ),
                    )
                )
                sequence += 1
        return turns


# ---------------------------------------------------------------------------
# Continuum retention state (identical across the A0 and ASYM_APC arms)
# ---------------------------------------------------------------------------


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def now(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = value

    def advance(self, seconds: float) -> None:
        self._value += seconds


def retention_entry(
    program: str,
    prefix: str,
    block_ids: tuple[int, ...],
    *,
    decision_timestamp: float,
    ttl_seconds: float,
    protected: bool = True,
):
    from kvopt.continuum.snapshots import (
        RetentionEntrySnapshot,
        TTLDecision,
        TTLHistoryMode,
        TTLInput,
    )
    from kvopt.continuum.types import (
        BlockIdentity,
        InputProvenance,
        InputSource,
        PrefixIdentity,
        ProgramIdentity,
        RequestIdentity,
    )

    program_id = ProgramIdentity(program)
    prefix_id = PrefixIdentity(prefix)
    observed = InputProvenance(InputSource.OBSERVED)
    approximated = InputProvenance(InputSource.APPROXIMATED, "synthetic workload parameter")
    ttl_input = TTLInput(
        program_id=program_id,
        request_id=RequestIdentity(f"{program}:{prefix}:{decision_timestamp:.3f}"),
        prefix_id=prefix_id,
        decision_timestamp=decision_timestamp,
        next_tool_type=None,
        global_server_gap_samples_seconds=(),
        tool_server_gap_samples_seconds=(),
        queue_delay_t_seconds=0.0,
        eta=1.0,
        prefill_reload_seconds=1.0,
        default_ttl_seconds=ttl_seconds,
        server_gap_history_provenance=observed,
        queue_delay_provenance=observed,
        eta_provenance=approximated,
        prefill_reload_provenance=approximated,
        default_ttl_provenance=approximated,
    )
    decision = TTLDecision(
        ttl_input=ttl_input,
        ttl_seconds=ttl_seconds,
        history_mode=TTLHistoryMode.COLD_START,
        reason="probe cold start",
    )
    return RetentionEntrySnapshot(
        ttl_decision=decision,
        protected=protected,
        waiting_followup=False,
        block_ids=tuple(BlockIdentity(block_id) for block_id in block_ids),
    )


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    requests: list[dict[str, object]] = field(default_factory=list)
    decisions: list[dict[str, object]] = field(default_factory=list)
    pressure_events: int = 0
    physically_evicted: int = 0
    pressure_retention_entries: list[int] = field(default_factory=list)
    pressure_protected_entries: list[int] = field(default_factory=list)
    fallbacks: int = 0
    unavailable_reason: str | None = None


def run_arm(
    arm: str,
    turns: list[Turn],
    *,
    num_blocks: int,
    ttl_seconds: float,
    profile_path: Path,
    profile_sha: str,
    allowed_sources: tuple[str, ...],
    proxy_gain_ms: float,
) -> ArmResult:
    from kvopt.continuum.retention import RetentionManager
    from kvopt.costaware.a2_native import A2NativeChainSnapshot, A2NativeSnapshotRegistry
    from kvopt.runtime.vllm.continuum_observation import encode_native_hash
    from kvopt.runtime.vllm.retention_integration import (
        RetentionMode,
        RetentionRuntimeIntegration,
    )

    substrate = VLLMSubstrate(num_blocks)
    clock = ManualClock(0.0)
    manager = RetentionManager(clock)
    result = ArmResult(arm=arm)

    from kvopt.costaware.asym_apc_runner import (
        build_asym_apc_arm,
        load_calibration_profile,
        parse_policy_config,
    )

    # Every arm is priced with the same measured curve, so the latency columns
    # are comparable; only the victim selection differs.
    profile = load_calibration_profile(
        profile_path, expected_sha256=profile_sha, allowed_sources=allowed_sources
    )
    # Both controlled arms must observe and maintain the same Retention Entry
    # lifecycle. Otherwise A0 is merely Native LRU with an empty entry table.
    registry: A2NativeSnapshotRegistry | None = (
        A2NativeSnapshotRegistry() if arm in {"a0", "asym_apc"} else None
    )
    arm_coordinator = None
    if arm == "asym_apc":
        config = parse_policy_config(
            {
                "enabled": True,
                "profile_path": str(profile_path),
                "profile_sha256": profile_sha,
                "alpha_seconds": 60.0,
                "beta_seconds": 12.0,
                "knee_seconds": 60.0,
                "tau_seconds": 30.0,
                "max_exact_subsets": 12,
                "max_greedy_operations": 200000,
                "min_proxy_gain_ms": proxy_gain_ms,
            },
            repo_root=REPO_ROOT,
            clock_domain="probe_monotonic",
        )
        assert registry is not None
        arm_coordinator, profile, _reconciler = build_asym_apc_arm(
            registry=registry,
            config=config,
            # Decision CPU overhead must be measured with the real clock,
            # not the synthetic workload clock that is constant during selection.
            clock=time.perf_counter,
            allowed_sources=allowed_sources,
        )
    deadlines: dict[str, float] = {}

    def sync_entries(timestamp: float) -> None:
        """Keep the Continuum entry table identical to observed residency.

        The selector refuses to act on an entry whose recorded block IDs are not
        the blocks that actually hold that content, which is the correct rule.
        A probe that observes residency must therefore keep the entry table in
        the same generation instead of leaving last turn's IDs behind.
        """
        if registry is None:
            return
        for chain_id in sorted(registry._chains):
            state = registry.chain_state(chain_id)
            if state is None or not state.resident_depths:
                continue
            ids = tuple(
                state.snapshot.block_ids[depth - 1] for depth in state.resident_depths
            )
            program, _, prefix = chain_id.partition(":")
            manager.upsert(
                retention_entry(
                    program,
                    prefix,
                    ids,
                    decision_timestamp=timestamp,
                    ttl_seconds=ttl_seconds,
                    protected=timestamp < deadlines.get(chain_id, 0.0),
                )
            )

    integration: RetentionRuntimeIntegration | None = None
    if arm in {"a0", "asym_apc"}:
        integration = RetentionRuntimeIntegration(
            manager,
            block_selection_coordinator=arm_coordinator,
        )

    def pressure(required: int, timestamp: float) -> list[int]:
        """Ask the arm for ``required`` victims and hand them to allocation."""
        result.pressure_events += 1
        sync_entries(timestamp)
        # Record actual Continuum state in each controlled arm, not only the
        # victim ids. This makes retention-state parity auditable after replay.
        entry_snapshots = manager.planning_snapshots()
        result.pressure_retention_entries.append(len(entry_snapshots))
        result.pressure_protected_entries.append(
            sum(bool(entry.protected) for entry in entry_snapshots)
        )
        blocks = substrate.free_queue_blocks()
        chosen: list[int] = []
        integration.apply_pressure(
            mode=RetentionMode.CONTROLLED,
            blocks=blocks,
            required_blocks=required,
            timestamp=timestamp,
            remove_selected=lambda ids: chosen.extend(ids),
        )
        taken = substrate.take_specific(chosen)
        if arm_coordinator is not None:
            decision = arm_coordinator.pop_pending_decision()
            if decision is not None:
                result.decisions.append(decision.payload())
        return taken

    for turn in turns:
        clock.advance(turn.arrival_gap_seconds)
        manager.expire_due()
        request = substrate.request(f"req-{turn.sequence_index}", list(turn.token_ids))
        hashes = list(request.block_hashes)

        hit_ids = substrate.lookup_chain(hashes)
        substrate.touch(hit_ids)
        hit_blocks = len(hit_ids)
        needed = turn.prompt_blocks - hit_blocks

        free_ids = substrate.free_queue_ids()
        hashed_free = {block_id for block_id in free_ids if substrate.hash_of(block_id) is not None}
        hashless_free = len(free_ids) - len(hashed_free)
        del hashless_free  # kept only for the audit record below

        if needed > substrate.num_free():
            result.requests.append(
                {
                    "sequence_index": turn.sequence_index,
                    "status": "STALLED_INSUFFICIENT_FREE_BLOCKS",
                    "needed": needed,
                    "free": substrate.num_free(),
                }
            )
            substrate.release(hit_ids)
            continue

        if arm == "native_lru":
            new_ids = substrate.allocate(needed)
        else:
            # Every allocation is routed through the selector.
            #
            # The native allocator always pops the free-queue head, so it
            # reclaims cache entries continuously rather than at one discrete
            # "pressure event". Deciding only when a block is already out of
            # hashless headroom is therefore too late: by then the shallow
            # blocks of the chain that is about to return have already been
            # recycled. A Block-level policy has to own the whole allocation.
            required_victims = needed
            new_ids = []
            if required_victims:
                new_ids = pressure(required_victims, clock.now())
            rest = needed - len(new_ids)
            if rest > 0:
                new_ids.extend(substrate.allocate(rest))

        evicted_ids = [block_id for block_id in new_ids if block_id in hashed_free]
        result.physically_evicted += len(evicted_ids)
        substrate.cache(request, hit_ids + new_ids, hit_blocks)

        if registry is not None:
            for block_id in evicted_ids:
                registry.observe_eviction(block_id)
            chain_blocks = tuple(hit_ids + new_ids)
            registry.observe(
                A2NativeChainSnapshot(
                    chain_id=f"{turn.program}:{PREFIX_KEY}",
                    block_ids=chain_blocks,
                    content_hashes=tuple(
                        encode_native_hash(substrate.hash_of(block_id))
                        for block_id in chain_blocks
                    ),
                    native_hash_process_id="probe-block-pool",
                ),
                observed_at=clock.now(),
                clock_domain="probe_monotonic",
            )
            if arm_coordinator is not None:
                arm_coordinator.coordinator.observe_chain_access(
                    f"{turn.program}:{PREFIX_KEY}", clock.now()
                )

        recomputed_tokens = (turn.prompt_blocks - hit_blocks) * BLOCK_TOKENS
        prefill_ms = None
        if profile is not None and turn.prompt_blocks in profile.curves_by_length:
            prefill_ms = profile.curves_by_length[turn.prompt_blocks].at(hit_blocks)

        result.requests.append(
            {
                "sequence_index": turn.sequence_index,
                "status": "OK",
                "program": turn.program,
                "prompt_tokens": turn.prompt_tokens,
                "prompt_blocks": turn.prompt_blocks,
                "cached_prefix_blocks": hit_blocks,
                "recomputed_tokens": recomputed_tokens,
                "prefix_hit_rate": hit_blocks / turn.prompt_blocks,
                "evicted_blocks_this_request": len(evicted_ids),
                "free_blocks_after": substrate.num_free(),
                "modelled_prefill_ms": prefill_ms,
                "timestamp": clock.now(),
            }
        )

        substrate.release(hit_ids + new_ids)
        deadlines[f"{turn.program}:{PREFIX_KEY}"] = clock.now() + ttl_seconds
        if registry is not None:
            sync_entries(clock.now())

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def summarise(result: ArmResult, *, measurement_source: str) -> dict[str, object]:
    completed = [row for row in result.requests if row.get("status") == "OK"]
    recomputed = [row["recomputed_tokens"] for row in completed]
    hit_rates = [row["prefix_hit_rate"] for row in completed]
    modelled = [
        row["modelled_prefill_ms"]
        for row in completed
        if isinstance(row.get("modelled_prefill_ms"), (int, float))
    ]
    prompt_tokens = [row["prompt_tokens"] for row in completed]
    divergent = [
        decision for decision in result.decisions if decision.get("victim_set_differs")
    ]
    reasons: dict[str, int] = {}
    for decision in result.decisions:
        reason = str(decision.get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "arm": result.arm,
        "requests_completed": len(completed),
        "requests_stalled": len(result.requests) - len(completed),
        "pressure_events": result.pressure_events,
        "retention_entries_at_pressure_max": max(result.pressure_retention_entries, default=0),
        "protected_entries_at_pressure_max": max(result.pressure_protected_entries, default=0),
        "pressure_events_with_retention_entries": sum(
            n > 0 for n in result.pressure_retention_entries
        ),
        "pressure_events_with_protected_entries": sum(
            n > 0 for n in result.pressure_protected_entries
        ),
        "physically_evicted_blocks": result.physically_evicted,
        "total_prompt_tokens": sum(int(value) for value in prompt_tokens),
        "total_recomputed_tokens": sum(int(value) for value in recomputed),
        "mean_recomputed_tokens": statistics.fmean(recomputed) if recomputed else None,
        "median_recomputed_tokens": statistics.median(recomputed) if recomputed else None,
        "mean_prefix_hit_rate": statistics.fmean(hit_rates) if hit_rates else None,
        "modelled_prefill_ms_total": sum(float(value) for value in modelled) if modelled else None,
        "modelled_prefill_ms_mean": statistics.fmean(modelled) if modelled else None,
        "decisions": len(result.decisions),
        "decision_reasons": reasons,
        "victim_set_divergence_count": len(divergent),
        "victim_set_divergence_rate": (
            len(divergent) / len(result.decisions) if result.decisions else None
        ),
        "cost_model_source": measurement_source if modelled else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="calibration profile JSON")
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument(
        "--allow-source",
        action="append",
        default=None,
        help="accepted calibration measurement_source (repeatable)",
    )
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--programs", type=int, default=6)
    parser.add_argument("--turns-per-program", type=int, default=6)
    parser.add_argument("--shared-prefix-blocks", type=int, default=4)
    parser.add_argument("--program-prefix-blocks", type=int, default=12)
    parser.add_argument("--short-gap", type=float, default=2.0)
    parser.add_argument("--long-gap", type=float, default=90.0)
    parser.add_argument("--ttl", type=float, default=30.0)
    parser.add_argument("--min-proxy-gain-ms", type=float, default=0.0)
    parser.add_argument(
        "--target-blocks",
        default="24,32,40,40,40",
        help="prompt length in blocks for successive turns of one program",
    )
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    require_vllm()
    if os.getenv("PYTHONHASHSEED") is None:
        print(
            "warning: PYTHONHASHSEED is unset, so vLLM block hashes will differ "
            "between processes; set PYTHONHASHSEED=0 for a reproducible run",
            file=sys.stderr,
        )

    profile_path = Path(args.profile).resolve()
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    measurement_source = str(document.get("measurement_source"))
    allowed = tuple(args.allow_source or [measurement_source])
    sha = args.profile_sha256
    actual_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if sha != actual_sha:
        raise SystemExit(f"profile SHA-256 mismatch: argument {sha} != file {actual_sha}")

    workload = WorkloadSpec(
        programs=args.programs,
        turns_per_program=args.turns_per_program,
        shared_prefix_blocks=args.shared_prefix_blocks,
        program_prefix_blocks=args.program_prefix_blocks,
        short_gap_seconds=args.short_gap,
        long_gap_seconds=args.long_gap,
        seed=args.seed,
        target_blocks=tuple(int(v) for v in args.target_blocks.split(",")),
    )
    turns = workload.build()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, object]] = {}
    for arm in ("native_lru", "a0", "asym_apc"):
        started = time.perf_counter()
        result = run_arm(
            arm,
            turns,
            num_blocks=args.num_blocks,
            ttl_seconds=args.ttl,
            profile_path=profile_path,
            profile_sha=sha,
            allowed_sources=allowed,
            proxy_gain_ms=args.min_proxy_gain_ms,
        )
        elapsed = time.perf_counter() - started
        summary = summarise(result, measurement_source=measurement_source)
        summary["wall_seconds"] = elapsed
        summaries[arm] = summary

        with (out_dir / f"requests-{arm}.jsonl").open("w", encoding="utf-8") as handle:
            for row in result.requests:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with (out_dir / f"decisions-{arm}.jsonl").open("w", encoding="utf-8") as handle:
            for row in result.decisions:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    paired = {
        "arm_a": "a0",
        "arm_b": "asym_apc",
        "recomputed_token_delta": (
            summaries["asym_apc"]["total_recomputed_tokens"]
            - summaries["a0"]["total_recomputed_tokens"]
        ),
        "recomputed_token_relative_change": (
            (
                summaries["asym_apc"]["total_recomputed_tokens"]
                - summaries["a0"]["total_recomputed_tokens"]
            )
            / summaries["a0"]["total_recomputed_tokens"]
            if summaries["a0"]["total_recomputed_tokens"]
            else None
        ),
        "hit_rate_delta": (
            summaries["asym_apc"]["mean_prefix_hit_rate"]
            - summaries["a0"]["mean_prefix_hit_rate"]
        ),
        "native_vs_a0_recomputed_delta": (
            summaries["a0"]["total_recomputed_tokens"]
            - summaries["native_lru"]["total_recomputed_tokens"]
        ),
    }

    payload = {
        "schema": "kvopt.asym_apc.real_blockpool_evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": _engine_identity(),
        "workload": {
            "programs": args.programs,
            "turns_per_program": args.turns_per_program,
            "shared_prefix_blocks": args.shared_prefix_blocks,
            "program_prefix_blocks": args.program_prefix_blocks,
            "short_gap_seconds": args.short_gap,
            "long_gap_seconds": args.long_gap,
            "ttl_seconds": args.ttl,
            "num_blocks": args.num_blocks,
            "block_tokens": BLOCK_TOKENS,
            "seed": args.seed,
            "total_requests": len(turns),
        },
        "calibration": {
            "profile_path": str(profile_path),
            "profile_sha256": sha,
            "measurement_source": measurement_source,
            "accepted_sources": list(allowed),
        },
        "summaries": summaries,
        "paired": paired,
        "scope": {
            "real_vllm_block_pool": True,
            "gpu_executed": False,
            "measured_prefill_latency": False,
            "latency_is_model_based": True,
            "recomputed_tokens_are_measured": True,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(paired, indent=2, sort_keys=True))
    return 0


def _engine_identity() -> dict[str, object]:
    identity: dict[str, object] = {"python": sys.version.split()[0]}
    try:
        import vllm

        identity["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        identity["vllm_version"] = "unavailable"
    try:
        import torch

        identity["torch_version"] = torch.__version__
        identity["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        identity["torch_version"] = "unavailable"
    identity["block_hash_function"] = HASH_FN_NAME
    identity["pythonhashseed"] = os.getenv("PYTHONHASHSEED")
    return identity


if __name__ == "__main__":
    raise SystemExit(main())
