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
    missing_reasons: tuple[tuple[str, int], ...] = ()

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
        clock_domain: str = "system_monotonic",
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
        if not isinstance(clock_domain, str) or not clock_domain:
            raise ValueError("clock_domain must be nonempty")
        self._clock_domain = clock_domain
        self._last_access: dict[str, float] = {}
        self.last_audit: AsymAPCAudit | None = None

    @property
    def clock_domain(self) -> str:
        """Clock domain every injected timestamp must belong to."""
        return self._clock_domain

    def observe_chain_access(self, chain_id: str, timestamp: float) -> None:
        """Ingest a real request/prefix observation; not a future/TTL deadline."""
        if not chain_id:
            raise ValueError("chain_id required")
        timestamp = _finite_nonnegative(timestamp, "timestamp")
        previous = self._last_access.get(chain_id)
        if previous is not None and timestamp < previous:
            raise ValueError("cannot backdate observed chain access")
        self._last_access[chain_id] = timestamp

    def _observed_access_time(self, chain_id: str) -> float | None:
        """Use observed Native prefix snapshots, never TTL or post-decision truth.

        The registry may expose a separately recorded monotonic timestamp.
        Reject other clocks and snapshots whose content generation has changed.
        """
        explicit = self._last_access.get(chain_id)
        getter = getattr(self._registry, "last_access_observation", None)
        if callable(getter):
            fact = getter(chain_id)
            if fact is not None and getattr(fact, "clock_domain", None) == self._clock_domain:
                chain_snapshot = getattr(self._registry, "snapshot", lambda _c: None)(chain_id)
                if (chain_snapshot is not None
                    and getattr(fact, "native_hash_process_id", None)
                        == getattr(chain_snapshot, "native_hash_process_id", None)
                    and getattr(fact, "content_hashes", None)
                        == getattr(chain_snapshot, "content_hashes", None)):
                    when = getattr(fact, "timestamp", None)
                    if isinstance(when, (float, int)) and math.isfinite(when) and when >= 0:
                        return max(float(when), explicit) if explicit is not None else float(when)
        return explicit

    def select_victims(
        self,
        candidates: Sequence[Any],
        preparation: Any,
        retention_entries: Sequence[Any],
        *,
        required_blocks: int,
        timestamp: float,
    ) -> tuple[int, ...]:
        """Select from proved candidates, preserving unknown candidates' A0 status.

        A missing hash/chain/profile outside the A0 victim set must not veto an
        otherwise complete decision. Unknown candidates are NEVER promoted to
        victims, and an unverified A0 victim is never displaced or assigned a
        fictitious zero cost. Only a verified subset may be re-ordered, within
        the real Continuum eligibility tiers. Audit reports partial coverage.
        """
        timestamp = _finite_nonnegative(timestamp, "timestamp")
        if isinstance(required_blocks, bool) or not isinstance(required_blocks, int) or required_blocks < 0:
            raise ValueError("required_blocks must be a non-negative integer")
        eligible_order = tuple(b.block_id for b in preparation.virtual_eligible_order)
        need = min(required_blocks, len(eligible_order))
        baseline = eligible_order[:need]

        def finish(reason: str, selected: tuple[int, ...] | None = None,
                   base_loss: float | None = None, proposed_loss: float | None = None,
                   ranked: int = 0, exact: bool = False,
                   missing: Mapping[str, int] | None = None) -> tuple[int, ...]:
            result = baseline if selected is None else selected
            self.last_audit = AsymAPCAudit(
                timestamp, baseline, result, reason, base_loss, proposed_loss,
                ranked, exact, tuple(sorted((missing or {}).items())),
            )
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
        tier_by_id: dict[int, int] = {}
        for item in preparation.original_free_queue:
            if item.eligibility_tier is not None:
                tier_by_id[item.block_id.block_id] = int(item.eligibility_tier)
        if set(eligible_order) != set(tier_by_id):
            return finish("tier_contract_mismatch")
        if any(tier_by_id[a] > tier_by_id[b]
               for a, b in zip(eligible_order, eligible_order[1:])):
            return finish("tier_order_mismatch")

        # A2's registry performs read-only content-hash, chain-depth and residency
        # reconciliation. Physical IDs or Native LRU ranks are never depths.
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
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            return finish(f"invalid_native_evidence:{type(exc).__name__}")

        from collections import Counter
        missing: Counter[str] = Counter()
        invalid: dict[int, str] = {}
        memberships: dict[int, tuple[tuple[str, int], ...]] = {}
        lengths: dict[str, int] = {}
        curves: dict[str, APCLatencyCurve] = {}
        access: dict[str, float] = {}
        seen_depths: dict[tuple[str, int], int] = {}

        # Each candidate is validated independently, including all its shared
        # memberships. A failure elsewhere in the free queue does not taint it.
        for bid in eligible_order:
            b, original = blocks[bid], by_id[bid]
            reason: str | None = None
            candidate_members: list[tuple[str, int]] = []
            if b.native_rank != original.lru_rank or b.has_block_hash != original.has_block_hash:
                reason = "content_contract_mismatch"
            elif not b.has_block_hash:
                if b.memberships:
                    reason = "hashless_has_memberships"
            elif not b.memberships or not original.content_hash:
                reason = "content_unverified"
            elif {m.content_hash for m in b.memberships} != {original.content_hash}:
                reason = "content_identity_mismatch"
            else:
                for m in b.memberships:
                    length = req.chain_length.get(m.chain)
                    if (not isinstance(length, int) or isinstance(length, bool)
                        or length < 1 or not isinstance(m.depth, int)
                        or m.depth < 1 or m.depth > length):
                        reason = "depth_unverified"
                        break
                    prev = seen_depths.get((m.chain, m.depth))
                    if prev is not None and prev != bid:
                        # A depth collision invalidates both competing records.
                        invalid[prev] = "depth_collision"
                        reason = "depth_collision"
                        break
                    curve = curves.get(m.chain)
                    if curve is None:
                        snapshot_getter = getattr(self._registry, "snapshot", None)
                        snap = snapshot_getter(m.chain) if callable(snapshot_getter) else None
                        original_len = len(snap.block_ids) if snap is not None else length
                        if original_len < length:
                            reason = "chain_length_mismatch"
                            break
                        curve = self._curves.get(original_len)
                        if curve is None:
                            reason = "profile_missing"
                            break
                        curves[m.chain] = curve
                        lengths[m.chain] = length
                    elif lengths[m.chain] != length:
                        reason = "chain_length_mismatch"
                        break
                    when = access.get(m.chain)
                    if when is None:
                        when = self._observed_access_time(m.chain)
                        if when is None:
                            reason = "recency_missing"
                            break
                        if when > timestamp:
                            reason = "future_observation"
                            break
                        access[m.chain] = when
                    candidate_members.append((m.chain, m.depth))
            if reason is not None:
                invalid[bid] = reason
                missing[reason] += 1
                continue
            memberships[bid] = tuple(candidate_members)
            for chain, depth in candidate_members:
                seen_depths[(chain, depth)] = bid

        # Avoid treating incomplete chain relationships as a complete marginal
        # loss estimate. A collision may have been discovered later in the pass.
        for bid in tuple(invalid):
            memberships.pop(bid, None)
        # Ensure every chain considered has a validated pre-decision timestamp.
        verified = set(memberships)
        if any(bid not in verified for bid in baseline):
            first = next(bid for bid in baseline if bid not in verified)
            # The baseline itself contains unknown content. No unsupported
            # counterfactual can be claimed better, so keep the exact A0 set.
            return finish(invalid.get(first, "baseline_unverified"),
                          ranked=len(verified), missing=missing)
        if len(verified) == need:
            return finish("no_verified_alternative", ranked=len(verified), missing=missing)

        # Unknown blocks not selected by baseline remain unselected under this
        # proposal. All victim sets are therefore content-verified, even if
        # unrelated blocks elsewhere in the queue have no logical-chain proof.
        involved = {chain for bid in verified for chain, _ in memberships[bid]}
        base_cost = {c: curves[c].at(lengths[c]) for c in involved}
        weights = {c: self._decay.weight(timestamp - access[c]) for c in involved}

        def loss(ids: Sequence[int]) -> float:
            prefixes = {c: lengths[c] for c in involved}
            for block_id in ids:
                for chain, depth in memberships[block_id]:
                    prefixes[chain] = min(prefixes[chain], depth - 1)
            return sum(weights[c] * max(0.0, curves[c].at(prefixes[c]) - base_cost[c])
                       for c in involved)

        selected: list[int] = []
        exact_only = True
        for tier in (0, 1):
            # The number required at each tier is fixed by A0's eligibility
            # plan; unknown blocks never consume or receive a new victim slot.
            take = sum(tier_by_id[bid] == tier for bid in baseline)
            if not take:
                continue
            members = [bid for bid in eligible_order
                       if tier_by_id[bid] == tier and bid in verified]
            if len(members) < take:
                return finish("insufficient_verified_candidates", ranked=len(verified),
                              missing=missing)
            if take == len(members):
                selected.extend(members)
                continue
            if math.comb(len(members), take) <= self._max_exact:
                options = itertools.combinations(members, take)
                chosen = min(options,
                             key=lambda choice: (loss((*selected, *choice)),
                                                 tuple(by_id[v].lru_rank for v in choice)))
                selected.extend(chosen)
            else:
                exact_only = False
                if len(members) * take > self._max_ops:
                    return finish("search_budget_exceeded", ranked=len(verified), missing=missing)
                remaining = set(members)
                for _ in range(take):
                    pick = min(remaining,
                               key=lambda bid: (loss((*selected, bid)), by_id[bid].lru_rank))
                    selected.append(pick)
                    remaining.remove(pick)
        proposed = tuple(selected)
        if len(proposed) != need or len(set(proposed)) != need:
            return finish("selection_contract_failed", ranked=len(verified), missing=missing)
        baseline_loss, proposed_loss = loss(baseline), loss(proposed)
        if (set(proposed) == set(baseline)
            or proposed_loss + self._min_gain >= baseline_loss):
            return finish("baseline_no_verified_proxy_gain", base_loss=baseline_loss,
                          proposed_loss=proposed_loss, ranked=len(verified), exact=exact_only,
                          missing=missing)
        mode = "selected_partial_coverage" if missing else "selected_lower_proxy_loss"
        return finish(mode, proposed, baseline_loss, proposed_loss,
                      len(verified), exact_only, missing)
