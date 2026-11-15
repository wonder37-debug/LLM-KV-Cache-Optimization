"""Experimental AsymCache-inspired victim selection for *contiguous* vLLM APC.

This is an opt-in, offline-first coordinator compatible with
RetentionRuntimeIntegration(block_selection_coordinator=...). It deliberately
preserves Continuum Entry Release, eligibility tiers and native BlockPool ownership.
It is NOT the original AsymCache (no multi-segment attention).

No future request outcome enters the decision. A missing/invalid decision-time
chain observation or latency profile returns the exact A0 eligible victim order.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    val = float(value)
    if not math.isfinite(val) or val < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return val


@dataclass(frozen=True, slots=True)
class APCLatencyCurve:
    """Measured/simulated prefill milliseconds for (total blocks, cached depth).

    Samples must span cached depth 0..total_blocks. Interpolation is allowed
    INSIDE that range; no extrapolation to a different prompt length.
    """

    total_blocks: int
    samples: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.total_blocks, int) or isinstance(self.total_blocks, bool) or self.total_blocks < 1:
            raise ValueError("total_blocks must be positive")
        if len(self.samples) < 2:
            raise ValueError("curve requires at least two samples")
        previous_depth = -1
        previous_cost = math.inf
        for depth, cost in self.samples:
            if not isinstance(depth, int) or isinstance(depth, bool):
                raise ValueError("sample depth must be an int")
            if depth <= previous_depth or not (0 <= depth <= self.total_blocks):
                raise ValueError("samples must have strictly increasing valid depths")
            cost = _finite_nonnegative(cost, "prefill milliseconds")
            if cost > previous_cost + 1e-9:
                raise ValueError("prefill cost must be non-increasing in cached depth")
            previous_depth, previous_cost = depth, cost
        if self.samples[0][0] != 0 or self.samples[-1][0] != self.total_blocks:
            raise ValueError("samples must cover 0 and total_blocks")

    def at(self, cached_depth: int) -> float:
        if not isinstance(cached_depth, int) or not 0 <= cached_depth <= self.total_blocks:
            raise ValueError("cached_depth outside calibrated curve")
        for i, (depth, cost) in enumerate(self.samples):
            if depth == cached_depth:
                return float(cost)
            if depth > cached_depth:
                left_depth, left_cost = self.samples[i - 1]
                return float(left_cost) + (float(cost) - float(left_cost)) * (
                    (cached_depth - left_depth) / (depth - left_depth)
                )
        raise AssertionError("curve must end at total_blocks")


@dataclass(frozen=True, slots=True)
class AsymDecay:
    """Two-timescale, AsymCache-inspired *ranking proxy*, not calibrated P(reuse)."""

    alpha_seconds: float = 60.0
    beta_seconds: float = 12.0
    knee_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("alpha_seconds", "beta_seconds"):
            if _finite_nonnegative(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        _finite_nonnegative(self.knee_seconds, "knee_seconds")

    def weight(self, age_seconds: float) -> float:
        age = _finite_nonnegative(age_seconds, "age_seconds")
        first = math.exp(-age / self.alpha_seconds)
        if age <= self.knee_seconds:
            return first  # second exponential would be >= 1 here
        return min(first, math.exp(-(age - self.knee_seconds) / self.beta_seconds))


@dataclass(frozen=True, slots=True)
class AsymAPCAudit:
    timestamp: float
    baseline: tuple[int, ...]
    selected: tuple[int, ...]
    reason: str
    baseline_proxy_ms: float | None = None
    selected_proxy_ms: float | None = None
    ranked_candidate_count: int = 0
    exact_search: bool = False

    @property
    def victim_set_differs(self) -> bool:
        return set(self.selected) != set(self.baseline)


class AsymAPCNativeCoordinator:
    """A0-compatible coordinator; inject into RetentionRuntimeIntegration.

    registry: existing A2NativeSnapshotRegistry (build_request method).
    curves_by_length: costs calibrated for each *exact* resident prefix length.
    observe_chain_access(): call only with timestamps observed before pressure.

    Protecting a retained Entry is handled by Continuum's preparation, NOT here.
    Strict TIER_1 then TIER_2 ordering is retained; cost reorders blocks only
    within tiers. The exact A0 order is a conservative baseline/fallback.
    """

    def __init__(
        self,
        registry: Any,
        curves_by_length: Mapping[int, APCLatencyCurve],
        *,
        decay: AsymDecay | None = None,
        tau_seconds: float = 1.0,
        max_exact_subsets: int = 3000,
        max_greedy_operations: int = 50000,
        min_proxy_gain_ms: float = 1e-9,
    ) -> None:
        if not callable(getattr(registry, "build_request", None)):
            raise TypeError("registry needs A2NativeSnapshotRegistry.build_request")
        if _finite_nonnegative(tau_seconds, "tau_seconds") <= 0:
            raise ValueError("tau_seconds must be positive")
        if max_exact_subsets < 1 or max_greedy_operations < 1:
            raise ValueError("search limits must be positive")
        self._registry = registry
        self._curves = dict(curves_by_length)
        for length, curve in self._curves.items():
            if not isinstance(curve, APCLatencyCurve) or length != curve.total_blocks:
                raise ValueError("curve key must match APCLatencyCurve.total_blocks")
        self._decay = decay or AsymDecay()
        self._tau_seconds = tau_seconds
        self._max_exact = max_exact_subsets
        self._max_ops = max_greedy_operations
        self._min_gain = _finite_nonnegative(min_proxy_gain_ms, "min_proxy_gain_ms")
        self._last_access: dict[str, float] = {}
        self.last_audit: AsymAPCAudit | None = None

    def observe_chain_access(self, chain_id: str, timestamp: float) -> None:
        """Ingest a real request/prefix observation; not a future/TTL deadline."""
        if not chain_id:
            raise ValueError("chain_id required")
        timestamp = _finite_nonnegative(timestamp, "timestamp")
        previous = self._last_access.get(chain_id)
        if previous is not None and timestamp < previous:
            raise ValueError("cannot backdate observed chain access")
        self._last_access[chain_id] = timestamp

    def select_victims(
        self,
        candidates: Sequence[Any],
        preparation: Any,
        retention_entries: Sequence[Any],
        *,
        required_blocks: int,
        timestamp: float,
    ) -> tuple[int, ...]:
        """Compute the minimum estimated *lost contiguous-prefix prefill*.

        This makes no mutations to native blocks, their hashes or their queue.
        The caller performs the existing native commit/removal after validation.
        """
        timestamp = _finite_nonnegative(timestamp, "timestamp")
        if isinstance(required_blocks, bool) or required_blocks < 0:
            raise ValueError("required_blocks must be nonnegative")
        eligible_order = tuple(b.block_id for b in preparation.virtual_eligible_order)
        need = min(required_blocks, len(eligible_order))
        baseline = eligible_order[:need]

        def finish(reason: str, selected: tuple[int, ...] | None = None,
                   base_loss: float | None = None, proposed_loss: float | None = None,
                   ranked: int = 0, exact: bool = False) -> tuple[int, ...]:
            result = baseline if selected is None else selected
            self.last_audit = AsymAPCAudit(timestamp, baseline, result, reason,
                                           base_loss, proposed_loss, ranked, exact)
            return result

        if not need:
            return finish("no_pressure")
        if len(eligible_order) != len(set(eligible_order)):
            return finish("duplicate_eligibility")
        by_id = {c.block_id: c for c in candidates}
        if len(by_id) != len(candidates) or not set(eligible_order).issubset(by_id):
            return finish("candidate_snapshot_mismatch")
        if any(by_id[b].ref_cnt != 0 for b in eligible_order):
            return finish("nonzero_refcount")
        if need == len(eligible_order):
            return finish("no_choice")
        # Validate the tier labels from the *actual* frozen Continuum plan.
        tier_by_id: dict[int, int] = {}
        for item in preparation.original_free_queue:
            if item.eligibility_tier is None:
                continue
            tier_by_id[item.block_id.block_id] = int(item.eligibility_tier)
        if set(eligible_order) != set(tier_by_id):
            return finish("tier_contract_mismatch")
        if any(tier_by_id[a] > tier_by_id[b]
               for a, b in zip(eligible_order, eligible_order[1:])):
            return finish("tier_order_mismatch")

        # Registry proves content identity and actual depth, reusing A2's source.
        try:
            from kvopt.runtime.vllm.types import EvictionContext
            context = EvictionContext(need, len(candidates), len(candidates), timestamp)
            legal = tuple(by_id[bid] for bid in eligible_order)
            req = self._registry.build_request(
                legal, context, retention_entries, tau_seconds=self._tau_seconds
            )
            blocks = {b.block_id: b for b in req.blocks}
            if (len(blocks) != len(req.blocks) or set(blocks) != set(eligible_order)
                or req.required_blocks != need or req.timestamp != timestamp):
                return finish("request_contract_mismatch")
            for bid in eligible_order:
                b, original = blocks[bid], by_id[bid]
                if (b.native_rank != original.lru_rank
                    or b.has_block_hash != original.has_block_hash):
                    return finish("content_contract_mismatch")
                if not b.has_block_hash:
                    if b.memberships:
                        return finish("hashless_has_memberships")
                    continue
                if not b.memberships or not original.content_hash:
                    return finish("content_unverified")
                hashes = {m.content_hash for m in b.memberships}
                if hashes != {original.content_hash}:
                    return finish("content_identity_mismatch")
                for m in b.memberships:
                    length = req.chain_length.get(m.chain)
                    if (not isinstance(length, int) or isinstance(length, bool)
                        or length < 1 or m.depth < 1 or m.depth > length):
                        return finish("depth_unverified")
                    if m.chain not in self._last_access:
                        return finish("recency_missing")
                    if self._last_access[m.chain] > timestamp:
                        return finish("future_observation")
            # Refuse ambiguous duplicate depths; a physical slot may belong to
            # multiple chains but two *different* blocks may not claim a depth.
            positions: dict[tuple[str, int], int] = {}
            for b in blocks.values():
                for m in b.memberships:
                    pos = (m.chain, m.depth)
                    if pos in positions and positions[pos] != b.block_id:
                        return finish("depth_collision")
                    positions[pos] = b.block_id
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            return finish(f"invalid_native_evidence:{type(exc).__name__}")

        # Use *all* current chains, including shared physical blocks.
        memberships = {bid: tuple((m.chain, m.depth) for m in blocks[bid].memberships)
                       for bid in eligible_order}
        involved_chains = {chain for entries in memberships.values()
                           for chain, _ in entries}
        lengths = {c: req.chain_length[c] for c in involved_chains}
        curve_by_chain = {}
        for chain, reusable_depth in lengths.items():
            # In A2, chain_length is the CURRENT contiguous reusable prefix,
            # not necessarily the original target prompt length. A broken
            # prefix still has a longer original chain; estimate incremental
            # prefill using the original full-length calibrated curve.
            snapshot_getter = getattr(self._registry, "snapshot", None)
            snapshot = snapshot_getter(chain) if callable(snapshot_getter) else None
            original_blocks = (len(snapshot.block_ids) if snapshot is not None
                               else reusable_depth)
            if original_blocks < reusable_depth:
                return finish("chain_length_mismatch")
            curve = self._curves.get(original_blocks)
            if curve is None:
                return finish("profile_missing")
            curve_by_chain[chain] = curve
        weights = {c: self._decay.weight(timestamp - self._last_access[c])
                   for c in curve_by_chain}
        base_cost = {c: curve_by_chain[c].at(lengths[c]) for c in curve_by_chain}

        def loss(ids: Sequence[int]) -> float:
            depths = dict(lengths)
            for bid in ids:
                for chain, depth in memberships[bid]:
                    depths[chain] = min(depths[chain], depth - 1)
            return sum(weights[c] * max(0.0, curve_by_chain[c].at(d) - base_cost[c])
                       for c, d in depths.items())

        selected: list[int] = []
        exact_only = True
        for tier in (0, 1):
            members = [bid for bid in eligible_order if tier_by_id[bid] == tier]
            take = min(len(members), need - len(selected))
            if not take:
                continue
            if take == len(members):
                selected.extend(members)
                continue
            if math.comb(len(members), take) <= self._max_exact:
                combinations = itertools.combinations(members, take)
                best = min(combinations,
                           key=lambda choice: (loss((*selected, *choice)),
                                               tuple(by_id[v].lru_rank for v in choice)))
                selected.extend(best)
            else:
                exact_only = False
                if len(members) * take > self._max_ops:
                    return finish("search_budget_exceeded")
                remaining = set(members)
                for _ in range(take):
                    chosen = min(remaining,
                                 key=lambda b: (loss((*selected, b)), by_id[b].lru_rank))
                    selected.append(chosen)
                    remaining.remove(chosen)
        proposal = tuple(selected)
        if len(proposal) != need or len(set(proposal)) != need:
            return finish("selection_contract_failed")
        baseline_loss = loss(baseline)
        proposal_loss = loss(proposal)
        if (set(proposal) == set(baseline)
            or proposal_loss + self._min_gain >= baseline_loss):
            return finish("baseline_no_verified_proxy_gain", None,
                          baseline_loss, proposal_loss, len(eligible_order), exact_only)
        return finish("selected_lower_proxy_loss", proposal,
                      baseline_loss, proposal_loss, len(eligible_order), exact_only)