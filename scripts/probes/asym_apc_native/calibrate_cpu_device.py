"""Measure the APC prefill cost curve F(m, r) on the CPU device with a real model.

This probe answers one question only: how long does prefill actually take for a
prompt of ``m`` tokens when ``r`` of those tokens are already present in a
reusable prefix KV cache?

It is deliberately *not* a vLLM engine run. The vLLM engine cannot execute on
this host (no NVIDIA device), so the numbers produced here are labelled
``cpu_device_measured`` and must never be presented as GPU measurements. What
they do provide is a real measurement of the shape and resolution of F(m, r)
using the project's model and real transformer forward passes, which is enough
to decide whether a cost-aware selector can distinguish candidate victim sets
at all.

Measurement protocol
--------------------
* For a fixed prompt length ``m``, one deterministic token sequence is drawn.
  Every cached depth ``r`` reuses the *same* leading ``r`` tokens, so the
  comparison across depths is a genuine prefix-cache comparison and not a
  comparison of different prompts.
* The prefix forward pass that materialises the ``r`` cached tokens is executed
  and then thrown away; only the suffix forward pass is timed. That is exactly
  the quantity APC changes: the work that remains after a cache hit.
* Each point is warmed up once and then repeated; median, P90, min, max and the
  relative spread are recorded.
* Timing uses ``time.perf_counter`` around a materialised output tensor on CPU.
  There is no CUDA stream to synchronise, and no ``torch.cuda.synchronize`` is
  claimed.
* Monotonicity in ``r`` and a noise floor are reported explicitly. A curve whose
  adjacent steps are smaller than the measurement noise is reported as
  unresolved instead of being smoothed into a nicer-looking shape.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import statistics
from itertools import pairwise
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BLOCK_SIZE_TOKENS = 16
SCHEMA_VERSION = "kvopt.asym_apc.prefill_calibration_profile.v1"
MEASUREMENT_SOURCE = "cpu_device_measured"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(slots=True)
class Point:
    total_blocks: int
    cached_blocks: int
    prompt_tokens: int
    cached_tokens: int
    median_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float
    repeats: int

    @property
    def spread(self) -> float:
        return 0.0 if self.median_ms <= 0 else (self.p90_ms - self.minimum_ms) / self.median_ms

    def payload(self) -> dict[str, object]:
        return {
            "total_blocks": self.total_blocks,
            "cached_blocks": self.cached_blocks,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "computed_tokens": self.prompt_tokens - self.cached_tokens,
            "prefill_ms": self.median_ms,
            "prefill_ms_p90": self.p90_ms,
            "prefill_ms_min": self.minimum_ms,
            "prefill_ms_max": self.maximum_ms,
            "relative_spread": self.spread,
            "repeats": self.repeats,
        }


def _token_ids(length: int, seed: int) -> list[int]:
    """Deterministic pseudo-token sequence drawn from the model's vocabulary."""
    import random

    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def _resolve_revision(model_dir: Path) -> str:
    """Read the exact revision recorded by the downloader, never guess it."""
    snapshots = model_dir / ".cache" / "huggingface"
    for path in sorted(snapshots.rglob("*.metadata")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("commit_hash:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def _hardware_description() -> dict[str, object]:
    description: dict[str, object] = {
        "device": "cpu",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    description["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover - non-Linux hosts
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    description["memory_total_kib"] = int(line.split()[1])
                    break
    except OSError:  # pragma: no cover - non-Linux hosts
        pass
    return description


def _vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # pragma: no cover - vLLM is optional for this probe
        return "not-installed"


def measure(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()

    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    depths_tokens = [
        int(value) for value in args.depths.split(",") if value.strip()
    ]
    points: list[Point] = []

    for length in lengths:
        if length % BLOCK_SIZE_TOKENS:
            raise SystemExit("prompt lengths must be multiples of the block size")
        token_ids = _token_ids(length, seed=args.seed + length)
        total_blocks = length // BLOCK_SIZE_TOKENS
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        for cached_tokens in depths_tokens:
            if cached_tokens > length:
                continue
            if cached_tokens % BLOCK_SIZE_TOKENS:
                raise SystemExit("cached depths must be block aligned")
            prefix_ids = input_ids[:, :cached_tokens]
            suffix_ids = input_ids[:, cached_tokens:]

            cached_state = None
            if cached_tokens:
                # Materialise the reusable prefix ONCE, outside the timed region:
                # the cost of building it was already paid when it was cached.
                with torch.inference_mode():
                    prefix = model(input_ids=prefix_ids, use_cache=True)
                    cached_state = prefix.past_key_values

            def forward_suffix(past=None, suffix_ids=suffix_ids) -> None:
                with torch.inference_mode():
                    if past is None:
                        out = model(
                            input_ids=suffix_ids, use_cache=False, logits_to_keep=1
                        )
                    else:
                        out = model(
                            input_ids=suffix_ids,
                            past_key_values=copy.deepcopy(past),
                            use_cache=False,
                            logits_to_keep=1,
                        )
                    # Force the computation to materialise before stopping the clock.
                    float(out.logits[0, -1, 0].item())

            samples: list[float] = []
            if suffix_ids.numel() == 0:
                # Fully cached prompt: no token needs computing, so prefill work
                # is exactly zero by definition and there is nothing to time.
                samples = [0.0] * args.repeats
            else:
                for _ in range(args.warmup):
                    forward_suffix(cached_state)
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    forward_suffix(cached_state)
                    samples.append((time.perf_counter() - started) * 1000.0)

            points.append(
                Point(
                    total_blocks=total_blocks,
                    cached_blocks=cached_tokens // BLOCK_SIZE_TOKENS,
                    prompt_tokens=length,
                    cached_tokens=cached_tokens,
                    median_ms=statistics.median(samples),
                    p90_ms=_percentile(samples, 90.0),
                    minimum_ms=min(samples),
                    maximum_ms=max(samples),
                    repeats=len(samples),
                )
            )
            print(
                f"[calibrate] m={length:5d} r={cached_tokens:5d} "
                f"median={points[-1].median_ms:8.2f}ms "
                f"p90={points[-1].p90_ms:8.2f}ms spread={points[-1].spread:.3f}",
                flush=True,
            )

    quality = _curve_quality(points)
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_source": MEASUREMENT_SOURCE,
        "curves": _curves(points),
        "points": [point.payload() for point in points],
        "quality": quality,
        "provenance": {
            "model_name": args.model_name,
            "model_revision": _resolve_revision(model_dir),
            "hardware": _hardware_description(),
            "block_size": BLOCK_SIZE_TOKENS,
            "vllm_version": _vllm_version(),
            "runtime_mode": "transformers-prefill-probe",
            "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch_version": torch.__version__,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "torch_threads": args.threads,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "device_note": (
                "MEASURED ON CPU. No NVIDIA device exists on this host, so this "
                "profile must not be read as a GPU prefill measurement."
            ),
        },
    }


def _curves(points: list[Point]) -> list[dict[str, object]]:
    by_length: dict[int, list[Point]] = {}
    for point in points:
        by_length.setdefault(point.total_blocks, []).append(point)
    curves: list[dict[str, object]] = []
    for total_blocks in sorted(by_length):
        ordered = sorted(by_length[total_blocks], key=lambda point: point.cached_blocks)
        curves.append(
            {
                "total_blocks": total_blocks,
                "samples": [
                    {
                        "cached_blocks": point.cached_blocks,
                        "prefill_ms": point.median_ms,
                    }
                    for point in ordered
                ],
            }
        )
    return curves


def _curve_quality(points: list[Point]) -> dict[str, object]:
    """Report monotonicity and whether neighbouring depths are distinguishable.

    A cost-aware selector can only be better than a blind one if the measured
    curve actually resolves the difference between candidate victim sets. This
    block states, per prompt length, the smallest measured decrease per block of
    cached depth and compares it with the measurement spread.
    """
    by_length: dict[int, list[Point]] = {}
    for point in points:
        by_length.setdefault(point.total_blocks, []).append(point)
    report: dict[str, object] = {"per_length": [], "all_monotonic": True}
    for total_blocks in sorted(by_length):
        ordered = sorted(by_length[total_blocks], key=lambda point: point.cached_blocks)
        violations = []
        steps = []
        for left, right in pairwise(ordered):
            delta = right.median_ms - left.median_ms
            block_step = max(1, right.cached_blocks - left.cached_blocks)
            steps.append(-delta / block_step)
            if delta > 0:
                violations.append(
                    {
                        "from_cached_blocks": left.cached_blocks,
                        "to_cached_blocks": right.cached_blocks,
                        "increase_ms": delta,
                        "noise_floor_ms": max(left.spread, right.spread) * left.median_ms,
                    }
                )
        noise_floor_ms = max(
            (point.spread * point.median_ms for point in ordered), default=0.0
        )
        min_step = min(steps) if steps else 0.0
        max_step = max(steps) if steps else 0.0
        report["per_length"].append(
            {
                "total_blocks": total_blocks,
                "prompt_tokens": ordered[0].prompt_tokens,
                "monotonic": not violations,
                "violations": violations,
                "noise_floor_ms": noise_floor_ms,
                "ms_saved_per_cached_block_min": min_step,
                "ms_saved_per_cached_block_max": max_step,
                "resolvable": bool(min_step > noise_floor_ms) if steps else False,
                "total_saving_ms": ordered[0].median_ms - ordered[-1].median_ms,
            }
        )
        if violations:
            report["all_monotonic"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--lengths", default="256,512,1024")
    parser.add_argument("--depths", default="0,64,128,256")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="weights dtype; the container has an 8 GiB memory limit, so fp32 OOMs",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    document = measure(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality = document["quality"]
    print(
        json.dumps(
            {
                "written": str(destination),
                "measurement_source": document["measurement_source"],
                "all_monotonic": quality["all_monotonic"],
                "resolvable": {
                    entry["total_blocks"]: entry["resolvable"]
                    for entry in quality["per_length"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
