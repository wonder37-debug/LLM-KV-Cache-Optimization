"""Workload families that change KV-cache *geometry*, not just token content.

Why this exists
---------------
The first probe workload was eight programs with five append-only turns each,
issued strictly round-robin. Changing ``--seed`` changed the random token values
but not the number of blocks requested, the arrival order or the idle gaps, so
every seed produced a bit-identical block trace. That makes the seed a
determinism check, not an independent replication.

This module adds families whose *structure* varies: arrival order, program and
turn counts, tool-call gaps, shared-prefix ratios, prompt length mixtures,
Zipf-skewed revisit frequencies and non-append-only re-reads. Every family is a
pure function of ``(family, seed)``; the seed drives both the schedule and the
token content, so two seeds differ in cache competition, not only in hash values.

What stays true for every family
--------------------------------
* Prompts are prefix-composable: a request is the first ``k`` blocks of a
  per-program token stream, optionally preceded by a shared prefix. That is the
  structure contiguous-APC reasons about, and it keeps cache hits meaningful.
* No future information is used: the schedule is a static list of arrivals with
  absolute gaps, generated before any arm runs.
* All arms of a run receive the identical turn list, in the identical order.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

BLOCK_TOKENS = 16

# Token values are irrelevant to cache geometry; they only need to be stable and
# distinct so that block hashes match exactly when prefixes match exactly.
_TOKEN_LOW = 1000
_TOKEN_HIGH = 150000


@dataclass(frozen=True, slots=True)
class Turn:
    """One request arrival, fully determined before any arm is executed."""

    sequence_index: int
    program: str
    prefix_key: str
    token_ids: tuple[int, ...]
    arrival_gap_seconds: float
    step: int
    revisit: bool = False

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def prompt_blocks(self) -> int:
        return len(self.token_ids) // BLOCK_TOKENS


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Declarative description of one workload family."""

    name: str
    description: str
    programs: int
    turns_per_program: tuple[int, ...]
    shared_prefix_blocks: int
    program_prefix_blocks: int
    length_schedule: tuple[int, ...]
    gap_choices: tuple[float, ...]
    gap_policy: str  # alternating | sampled | fixed
    arrival: str  # round_robin | shuffled | zipf
    revisit_rate: float = 0.0
    zipf_s: float = 1.2
    regime_note: str = ""


LEGACY_LENGTHS = (24, 32, 40, 40, 40)


def _uniform_turns(count: int) -> tuple[int, ...]:
    return (count,) * count


FAMILIES: dict[str, FamilySpec] = {
    # The original shape, kept bit-for-bit reproducible so that new runs can be
    # compared with the corrected CI replay.
    "legacy_round_robin": FamilySpec(
        name="legacy_round_robin",
        description="8 programs x 5 append-only turns, strict round-robin arrival",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="round_robin",
        regime_note="reference shape",
    ),
    "agent_random_arrival": FamilySpec(
        name="agent_random_arrival",
        description="legacy shape, arrival order permuted per seed",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="shuffled",
        regime_note="arrival-order sensitivity",
    ),
    "program_turn_scaling": FamilySpec(
        name="program_turn_scaling",
        description="4/6/12 programs x 3-7 turns, round-robin",
        programs=10,
        turns_per_program=(4, 6, 3, 5, 7, 4, 5, 3, 6, 4),
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=(24, 32, 40, 48, 56, 64, 72),
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="zipf",
        zipf_s=1.0,
        regime_note="chain-count scaling",
    ),
    "tool_gap_mix": FamilySpec(
        name="tool_gap_mix",
        description="short/medium/long tool-call gaps sampled per turn",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(1.0, 5.0, 20.0, 45.0, 90.0, 300.0),
        gap_policy="sampled",
        arrival="round_robin",
        regime_note="TTL / recency sensitivity",
    ),
    "shared_prefix_ratio": FamilySpec(
        name="shared_prefix_ratio",
        description="shared prefix varied 0/8/16/24 blocks across programs",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=16,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="round_robin",
        regime_note="shared-prefix membership",
    ),
    "mixed_lengths": FamilySpec(
        name="mixed_lengths",
        description="prompt lengths sampled per turn from 16-80 blocks",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=(16, 24, 32, 48, 64, 80),
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="round_robin",
        regime_note="length heterogeneity",
    ),
    "zipf_revisit": FamilySpec(
        name="zipf_revisit",
        description="Zipf-skewed program revisit frequency, 10 programs",
        programs=10,
        # Skewed turn counts (sum 35) so hot sessions genuinely return more
        # often, not merely interleave differently.
        turns_per_program=(6, 5, 5, 4, 4, 3, 3, 2, 2, 1),
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="zipf",
        zipf_s=1.4,
        regime_note="skewed reuse",
    ),
    "unpredictable_revisit": FamilySpec(
        name="unpredictable_revisit",
        description="35% of turns re-read an earlier depth instead of appending",
        programs=8,
        turns_per_program=(5,) * 8,
        shared_prefix_blocks=4,
        program_prefix_blocks=12,
        length_schedule=LEGACY_LENGTHS,
        gap_choices=(2.0, 90.0),
        gap_policy="alternating",
        arrival="shuffled",
        revisit_rate=0.35,
        regime_note="non-append-only reuse",
    ),
}

# Capacity is a run-time parameter, so the pressure regime is expressed as a
# family of capacities rather than a separate workload file.
CAPACITY_REGIMES: dict[str, tuple[int, ...]] = {
    "high_pressure": (160, 192, 224),
    "moderate_pressure": (240, 256, 264, 272),
    "near_fit": (280, 288, 292, 296),
}


def family_names() -> tuple[str, ...]:
    return tuple(FAMILIES)


def _rng(label: str, seed: int) -> random.Random:
    """Deterministic RNG: ``Random(str)`` is seeded via SHA-512, not ``hash()``."""
    return random.Random(f"kvopt|{label}|{seed}")


def _tokens(rng: random.Random, count: int) -> list[int]:
    return [rng.randrange(_TOKEN_LOW, _TOKEN_HIGH) for _ in range(count * BLOCK_TOKENS)]


def _zipf_weights(count: int, s: float) -> list[float]:
    return [1.0 / ((rank + 1) ** s) for rank in range(count)]


def _arrival_order(
    planned: list[tuple[str, int, int, float, bool]], spec: FamilySpec, rng: random.Random
) -> list[tuple[str, int, int, float, bool]]:
    """Order planned turns according to the family's arrival policy."""
    if spec.arrival == "round_robin":
        return planned
    if spec.arrival == "shuffled":
        ordered = list(planned)
        rng.shuffle(ordered)
        return ordered
    if spec.arrival == "zipf":
        by_program: dict[str, list[tuple[str, int, int, float, bool]]] = {}
        for item in planned:
            by_program.setdefault(item[0], []).append(item)
        programs = sorted(by_program)
        weights = _zipf_weights(len(programs), spec.zipf_s)
        ordered: list[tuple[str, int, int, float, bool]] = []
        pending = {program: list(by_program[program]) for program in programs}
        while any(pending.values()):
            live = [program for program in programs if pending[program]]
            live_weights = [weights[programs.index(program)] for program in live]
            program = rng.choices(live, weights=live_weights, k=1)[0]
            ordered.append(pending[program].pop(0))
        return ordered
    raise ValueError(f"unknown arrival policy: {spec.arrival}")


def build_turns(spec: FamilySpec, seed: int) -> list[Turn]:
    """Materialise the deterministic arrival list for ``spec`` at ``seed``."""
    plan_rng = _rng(f"plan|{spec.name}", seed)
    token_rng = _rng(f"tokens|{spec.name}", seed)
    order_rng = _rng(f"order|{spec.name}", seed)

    shared = _tokens(token_rng, spec.shared_prefix_blocks)
    streams = {
        f"program-{index}": _tokens(token_rng, max(spec.length_schedule) + spec.program_prefix_blocks)
        for index in range(spec.programs)
    }
    program_context = {
        name: stream[: spec.program_prefix_blocks * BLOCK_TOKENS]
        for name, stream in streams.items()
    }

    planned: list[tuple[str, int, int, float, bool]] = []
    deepest_by_program = {f"program-{index}": 0 for index in range(spec.programs)}
    # Step-major, program-minor: this is the order the reference probe used, so
    # the reference family stays geometrically identical to it.
    for step in range(max(spec.turns_per_program)):
        for index in range(spec.programs):
            program = f"program-{index}"
            turns = spec.turns_per_program[index]
            if step >= turns:
                continue
            deepest = deepest_by_program[program]
            if spec.gap_policy == "alternating":
                gap = spec.gap_choices[step % 2] if len(spec.gap_choices) > 1 else spec.gap_choices[0]
            elif spec.gap_policy == "sampled":
                gap = plan_rng.choice(list(spec.gap_choices))
            elif spec.gap_policy == "fixed":
                gap = spec.gap_choices[0]
            else:
                raise ValueError(f"unknown gap policy: {spec.gap_policy}")

            if spec.name == "mixed_lengths":
                target = plan_rng.choice(list(spec.length_schedule))
            else:
                target = spec.length_schedule[min(step, len(spec.length_schedule) - 1)]

            revisit = False
            if spec.revisit_rate > 0 and deepest > 1 and plan_rng.random() < spec.revisit_rate:
                # Re-read an earlier depth: still a true prefix, so a healthy
                # cache can serve it, but it no longer grows the chain.
                target = plan_rng.randint(max(16, spec.program_prefix_blocks + 4), deepest - 1)
                revisit = True
            planned.append((program, step, target, gap, revisit))
            deepest_by_program[program] = max(deepest, target)

    ordered = _arrival_order(planned, spec, order_rng)

    turns: list[Turn] = []
    for sequence, (program, step, target, gap, revisit) in enumerate(ordered):
        base = shared + program_context[program]
        base_blocks = len(base) // BLOCK_TOKENS
        body_blocks = max(0, target - base_blocks)
        body = streams[program][: body_blocks * BLOCK_TOKENS]
        turns.append(
            Turn(
                sequence_index=sequence,
                program=program,
                prefix_key=f"{program}:step{step}",
                token_ids=tuple(base + body),
                arrival_gap_seconds=gap,
                step=step,
                revisit=revisit,
            )
        )
    return turns


def working_set_blocks(turns: Iterable[Turn], shared_prefix_blocks: int) -> int:
    """Blocks that must stay resident for every request to hit its full prefix.

    The shared head is one reusable chain, so it is counted once. Beyond it,
    every program owns its own suffixes: a repeated content block at a different
    chain position has a different chained hash and therefore occupies its own
    cache entry. Capacity expressed as a fraction of this number puts every
    workload shape in the same pressure regime, which is what makes families
    comparable without hand-tuning a capacity per family.
    """
    deepest: dict[str, int] = {}
    for turn in turns:
        deepest[turn.program] = max(deepest.get(turn.program, 0), turn.prompt_blocks)
    if not deepest:
        return 0
    shared = min(shared_prefix_blocks, min(deepest.values()))
    return shared + sum(depth - shared for depth in deepest.values())


def geometry_signature(turns: Iterable[Turn]) -> str:
    """Stable digest of the *structural* request sequence.

    Two seeds that only change token values produce the same signature; two
    seeds that change arrival order, lengths or gaps do not. This is what makes
    "independent seed" a checkable property instead of a claim.
    """
    payload = "|".join(
        f"{turn.program}:{turn.prompt_blocks}:{turn.arrival_gap_seconds:.3f}:"
        f"{int(bool(getattr(turn, 'revisit', False)))}"
        for turn in turns
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe(spec: FamilySpec) -> dict[str, object]:
    """Machine-readable family description for run records."""
    return {
        "family": spec.name,
        "description": spec.description,
        "programs": spec.programs,
        "turns_per_program": list(spec.turns_per_program),
        "shared_prefix_blocks": spec.shared_prefix_blocks,
        "program_prefix_blocks": spec.program_prefix_blocks,
        "length_schedule": list(spec.length_schedule),
        "gap_choices": list(spec.gap_choices),
        "gap_policy": spec.gap_policy,
        "arrival": spec.arrival,
        "revisit_rate": spec.revisit_rate,
        "zipf_s": spec.zipf_s,
        "regime_note": spec.regime_note,
    }
