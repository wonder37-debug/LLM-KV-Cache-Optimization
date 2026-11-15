"""Seeded synthetic CPU replay for AsymAPCNativeCoordinator.

This is NOT GPU/vLLM execution and is NOT evidence of Phase 3 gains.
It uses A2-shaped immutable content/depth/eligibility inputs; future return
is sampled AFTER selection solely for evaluation, never used as a feature.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from kvopt.costaware.asym_apc_native import (
    APCLatencyCurve, AsymAPCNativeCoordinator, AsymDecay
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


class ReplayRegistry:
    def __init__(self, inputs, length):
        self.inputs = inputs
        self.length = length

    def build_request(self, candidates, context, entries, *, tau_seconds):
        allowed = {c.block_id for c in candidates}
        return SimpleNamespace(
            required_blocks=context.required_blocks,
            timestamp=context.timestamp,
            blocks=tuple(b for b in self.inputs if b.block_id in allowed),
            chain_length={'A': self.length, 'B': self.length},
        )


class ReplayPreparation:
    def __init__(self, ordered_eligible):
        self.virtual_eligible_order = tuple(SimpleNamespace(block_id=i)
                                            for i in ordered_eligible)
        self.original_free_queue = tuple(SimpleNamespace(
            block_id=SimpleNamespace(block_id=i), eligibility_tier=0,
        ) for i in ordered_eligible)


def after_depth(victims, chain, length):
    depths = [((bid-1) % length)+1 for bid in victims
              if (bid <= length if chain == 'A' else bid > length)]
    return min(depths)-1 if depths else length


def run_simulation(groups: int = 160, seed: int = 20261115, length: int = 8):
    rng = random.Random(seed)
    decision_count = group_wins = group_ties = group_losses = 0
    divergences = 0
    a0_lost_tokens = ours_lost_tokens = 0
    a0_true_ms = ours_true_ms = 0.0
    predicted_gain_ms = 0.0
    # A *synthetic* monotonic performance curve, not GPU profiling evidence.
    def prefill_ms(cached):
        miss = length - cached
        return 0.35 * miss + 0.075 * miss * miss

    curve = APCLatencyCurve(length, tuple((i, prefill_ms(i))
                                           for i in range(length+1)))
    for _ in range(groups):
        # Two independent agent sessions; native free queue is depth-ordered
        # *within* each session. Session age determines their relative recency.
        age_a, age_b = rng.uniform(8, 115), rng.uniform(8, 115)
        ordered = (list(range(1, length+1)) + list(range(length+1,2*length+1))
                   if age_a >= age_b else
                   list(range(length+1,2*length+1)) + list(range(1,length+1)))
        ranks = {bid: i for i, bid in enumerate(ordered)}
        candidates = [EvictionCandidate(i, 0, True, ranks[i], f'{i:04x}')
                      for i in range(1, 2*length+1)]
        inputs = tuple(BlockDecisionInput(i, ranks[i], True,
                         (Membership('A' if i<=length else 'B',
                         ((i-1) % length)+1, f'{i:04x}'),))
                       for i in range(1,2*length+1))
        coordinator = AsymAPCNativeCoordinator(
            ReplayRegistry(inputs, length), {length:curve},
            decay=AsymDecay(alpha_seconds=60,beta_seconds=12,knee_seconds=60),
        )
        coordinator.observe_chain_access('A', 150-age_a)
        coordinator.observe_chain_access('B', 150-age_b)
        need = rng.choice([1,2,3])
        chosen = coordinator.select_victims(candidates, ReplayPreparation(ordered), (),
                                            required_blocks=need, timestamp=150)
        audit = coordinator.last_audit
        assert len(chosen) == need and all(i in ordered for i in chosen)
        assert audit.baseline_proxy_ms >= audit.selected_proxy_ms - 1e-9
        divergences += audit.victim_set_differs
        decision_count += 1
        predicted_gain_ms += audit.baseline_proxy_ms - audit.selected_proxy_ms
        # Hidden future outcome for each session: time-locality correlated, noisy.
        # Common random numbers used for both policies (paired counterfactual).
        revisit = {
            'A': rng.random() < 0.12 + 0.79 * pow(2.718281828, -age_a / 56),
            'B': rng.random() < 0.12 + 0.79 * pow(2.718281828, -age_b / 56),
        }
        a0_group_ms = ours_group_ms = 0.0
        for chain in ('A','B'):
            if revisit[chain]:
                before = after_depth(audit.baseline, chain, length)
                after = after_depth(chosen, chain, length)
                a0_lost_tokens += (length-before)*16
                ours_lost_tokens += (length-after)*16
                a0_group_ms += curve.at(before)-curve.at(length)
                ours_group_ms += curve.at(after)-curve.at(length)
        a0_true_ms += a0_group_ms
        ours_true_ms += ours_group_ms
        if ours_group_ms + 1e-8 < a0_group_ms:
            group_wins += 1
        elif ours_group_ms > a0_group_ms + 1e-8:
            group_losses += 1
        else:
            group_ties += 1
    return {
        'status': 'SYNTHETIC_CPU_ONLY',
        'no_native_gpu_or_real_trace_claim': True,
        'seed': seed, 'independent_synthetic_groups': groups,
        'blocks_per_chain': length, 'block_size_tokens': 16,
        'decision_count': decision_count,
        'victim_set_divergence_count': divergences,
        'group_win_tie_loss': [group_wins, group_ties, group_losses],
        'a0_realized_recompute_tokens': a0_lost_tokens,
        'ours_realized_recompute_tokens': ours_lost_tokens,
        'a0_realized_prefill_proxy_ms': round(a0_true_ms, 6),
        'ours_realized_prefill_proxy_ms': round(ours_true_ms, 6),
        'selected_predicted_gain_proxy_ms': round(predicted_gain_ms, 6),
        'assumptions': [
            'Two independent sessions and depth-ordered free queue per session',
            'Profile is an invented monotone quadratic, not measured GPU timing',
            'Hidden future returns are stochastic and correlated with elapsed time',
            'No scheduler, preemption, real refcount churn, queueing or model inference',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, default=160)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = run_simulation(groups=args.groups, seed=args.seed)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(output+'\n',encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
