"""Measure the Native GPU Prefill cost curve for the AsymAPC arm.

Protocol summary
----------------
For every calibrated prompt length ``m`` (number of 16-token blocks) and every
cached prefix depth ``r`` in ``0..m``:

1. a child process starts a fresh vLLM instance (empty Native cache, clean CUDA
   context) so a previous point cannot leave resident prefixes behind;
2. the first ``r`` blocks of the prompt are inserted by one warm-up request;
3. ``--warmups`` warm-up and ``--repeats`` measured generations of the full
   ``m``-block prompt run with ``max_tokens=1``, each timed with
   ``torch.cuda.synchronize()`` around the call;
4. the observed cached-token count of every measured request is compared with
   the requested depth; a mismatch marks the sample invalid with its reason.

The parent aggregates the samples with :mod:`kvopt.costaware.asym_apc_calibration`
and writes a hash-bound profile that the Native runner arm requires.

Example (RTX 4070, WSL2)::

    python scripts/probes/asym_apc_native/calibrate.py \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --revision <pinned-sha> \
        --grid 4,8,12,16,20 \
        --warmups 3 --repeats 7 \
        --output results/asym_apc/calibration/prefill-calibration-profile.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kvopt.costaware.asym_apc_calibration import (
    CalibrationError,
    CalibrationProtocol,
    PrefillMeasurement,
    environment_provenance,
    profile_document,
    run_calibration,
    write_profile,
)

_CACHED_TOKEN_ATTRIBUTES = (
    ("metrics", "num_cached_tokens"),
    ("metrics", "cached_tokens"),
    ("num_cached_tokens",),
    ("cached_tokens",),
)


def _read_cached_tokens(output: object) -> tuple[int | None, str]:
    """Probe the recorded cached-token count without inventing a value."""
    for path in _CACHED_TOKEN_ATTRIBUTES:
        target: object | None = output
        for attribute in path:
            target = getattr(target, attribute, None)
            if target is None:
                break
        if isinstance(target, int) and not isinstance(target, bool):
            return target, ".".join(path)
    return None, "cached_token_count_unavailable"


def _prompt_token_ids(
    *,
    total_blocks: int,
    block_size: int,
    vocabulary_size: int,
    salt: int,
) -> list[int]:
    """Deterministic, content-distinct token IDs that never wrap into a repeat."""
    count = total_blocks * block_size
    base = (salt * 7919) % max(1, vocabulary_size - (count * 2 + 64))
    offset = base + 32
    return [offset + index for index in range(count)]


def measure_point(
    *,
    model: str,
    revision: str,
    total_blocks: int,
    cached_blocks: int,
    block_size: int,
    warmups: int,
    repeats: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    cache_block_override: int,
    salt: int,
) -> dict[str, object]:
    """Measure one (prompt length, cached depth) point in a fresh process."""
    import torch
    from vllm import LLM, SamplingParams

    tokenizer_kwargs = {"revision": revision}
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer=model,
        tokenizer_revision=tokenizer_kwargs["revision"],
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        enforce_eager=True,
        enable_chunked_prefill=False,
        num_gpu_blocks_override=cache_block_override,
        prefix_caching_hash_algo="sha256",
        disable_log_stats=True,
        seed=salt,
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    hardware = (
        f"{torch.cuda.get_device_name(torch.cuda.current_device())} "
        f"({properties.total_memory // (1024 ** 2)} MiB, capability "
        f"{'.'.join(str(part) for part in torch.cuda.get_device_capability())})"
    )
    tokenizer = llm.get_tokenizer()
    vocabulary_size = len(tokenizer)
    tokens = _prompt_token_ids(
        total_blocks=total_blocks,
        block_size=block_size,
        vocabulary_size=vocabulary_size,
        salt=salt,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, seed=salt)
    prefix_tokens = tokens[: cached_blocks * block_size]
    if prefix_tokens:
        llm.generate([{"prompt_token_ids": prefix_tokens}], sampling, use_tqdm=False)

    observations: list[dict[str, object]] = []

    def run_once(index: int, *, warmup: bool) -> None:
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        cached, source = _read_cached_tokens(outputs[0])
        expected = cached_blocks * block_size
        if cached is None:
            invalid_reason = f"apc_hit_evidence_unavailable:{source}"
        elif cached != expected:
            invalid_reason = f"apc_hit_depth_mismatch:{cached}!={expected}"
        else:
            invalid_reason = None
        observations.append(
            {
                "repetition": index,
                "warmup": warmup,
                "elapsed_seconds": elapsed,
                "observed_cached_tokens": cached,
                "cached_token_source": source,
                "observed_computed_tokens": None,
                "invalid_reason": invalid_reason,
            }
        )

    for index in range(warmups):
        run_once(index, warmup=True)
    for index in range(repeats):
        run_once(index, warmup=False)

    provenance = environment_provenance(
        model_name=model,
        model_revision=revision,
        tokenizer_name=model,
        tokenizer_revision=revision,
        runtime_mode="vllm-cuda-observability",
        hardware=hardware,
        vllm_version=_distribution_version("vllm"),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        driver_version=_driver_version(),
        block_size=block_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        enforce_eager=True,
        enable_chunked_prefill=False,
        cache_block_override=cache_block_override,
        extra={
            "total_blocks": total_blocks,
            "cached_blocks": cached_blocks,
            "salt": salt,
        },
    )
    return {
        "point": {
            "total_blocks": total_blocks,
            "cached_blocks": cached_blocks,
            "block_size": block_size,
        },
        "observations": observations,
        "provenance": provenance,
    }


def _distribution_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception:  # noqa: BLE001 - record unavailability instead of guessing
        return "unavailable"


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = output.stdout.strip().splitlines()
    return value[0].strip() if value else None


def _measurement_from_observation(
    observation: Mapping[str, object], *, total_blocks: int, cached_blocks: int
) -> PrefillMeasurement:
    return PrefillMeasurement(
        total_blocks=total_blocks,
        requested_cached_blocks=cached_blocks,
        elapsed_seconds=float(observation["elapsed_seconds"]),
        observed_cached_tokens=observation["observed_cached_tokens"],  # type: ignore[arg-type]
        observed_computed_tokens=observation["observed_computed_tokens"],  # type: ignore[arg-type]
        repetition=int(observation["repetition"]),
        warmup=bool(observation["warmup"]),
        invalid_reason=observation["invalid_reason"],  # type: ignore[arg-type]
    )


def _run_point_subprocess(arguments: argparse.Namespace, total: int, cached: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--measure-point",
        str(total),
        str(cached),
        "--model",
        arguments.model,
        "--revision",
        arguments.revision,
        "--block-size",
        str(arguments.block_size),
        "--warmups",
        str(arguments.warmups),
        "--repeats",
        str(arguments.repeats),
        "--gpu-memory-utilization",
        str(arguments.gpu_memory_utilization),
        "--max-model-len",
        str(arguments.max_model_len),
        "--cache-block-override",
        str(arguments.cache_block_override),
        "--salt",
        str(arguments.salt),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise CalibrationError(
            f"measurement process failed for total={total} cached={cached}: "
            f"{result.stderr.strip()[-2000:]}"
        )
    payload = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not payload:
        raise CalibrationError(
            f"measurement process produced no JSON for total={total} cached={cached}"
        )
    return json.loads(payload[-1])


def _drive(arguments: argparse.Namespace) -> int:
    lengths = tuple(int(value) for value in arguments.grid.split(",") if value.strip())
    protocol = CalibrationProtocol(
        prompt_lengths_blocks=lengths,
        block_size=arguments.block_size,
        warmup_repetitions=arguments.warmups,
        measured_repetitions=arguments.repeats,
        minimum_valid_repetitions=arguments.minimum_valid,
        interpolation_stride=arguments.interpolation_stride,
    )
    if arguments.dry_run:
        plan = [
            {
                "total_blocks": total,
                "cached_blocks": list(protocol.depths_for(total)),
                "prompt_tokens": total * protocol.block_size,
            }
            for total in protocol.prompt_lengths_blocks
        ]
        print(json.dumps({"dry_run": True, "protocol": plan}, indent=2))
        return 0
    if shutil.which("nvidia-smi") is None:
        raise CalibrationError(
            "nvidia-smi is unavailable: the calibration requires a real CUDA GPU"
        )

    raw: dict[tuple[int, int], list[PrefillMeasurement]] = {}
    provenance: dict[str, object] | None = None
    for total in protocol.prompt_lengths_blocks:
        for cached in protocol.depths_for(total):
            result = _run_point_subprocess(arguments, total, cached)
            provenance = dict(result["provenance"])
            raw[(total, cached)] = [
                _measurement_from_observation(
                    observation, total_blocks=total, cached_blocks=cached
                )
                for observation in result["observations"]
            ]
            print(
                f"measured total={total} cached={cached} "
                f"({len(raw[(total, cached)])} observations)",
                file=sys.stderr,
            )

    def measure(total: int, cached: int, repetition: int) -> PrefillMeasurement:
        for observation in raw[(total, cached)]:
            if observation.repetition == repetition and not observation.warmup:
                return observation
        raise CalibrationError(f"missing observation {total}/{cached}/{repetition}")

    curves, _ = run_calibration(measure, protocol)
    if provenance is None:
        raise CalibrationError("no measurement provenance was recorded")
    document = profile_document(
        curves=curves,
        protocol=protocol,
        provenance=provenance,
        measurement_note=(
            "fresh process per (prompt length, cached depth) point; APC hit depth "
            "verified from the recorded cached-token count"
        ),
    )
    output = Path(arguments.output)
    digest = write_profile(output, document)
    print(f"profile: {output}\nsha256: {digest}")
    if not document["resolution"]["all_curves_discriminative"]:
        print(
            "WARNING: at least one curve is not discriminative at the measured noise "
            "floor; the proxy loss cannot separate candidates on that length",
            file=sys.stderr,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--revision", required=False, default="")
    parser.add_argument("--grid", default="4,8,12,16,20")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-valid", type=int, default=5)
    parser.add_argument("--interpolation-stride", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--cache-block-override", type=int, default=512)
    parser.add_argument("--salt", type=int, default=20261115)
    parser.add_argument("--output", default="results/asym_apc/calibration/profile.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-point", nargs=2, type=int, metavar=("TOTAL", "CACHED"))
    arguments = parser.parse_args(argv)

    if arguments.measure_point is not None:
        total, cached = arguments.measure_point
        result = measure_point(
            model=arguments.model,
            revision=arguments.revision,
            total_blocks=total,
            cached_blocks=cached,
            block_size=arguments.block_size,
            warmups=arguments.warmups,
            repeats=arguments.repeats,
            gpu_memory_utilization=arguments.gpu_memory_utilization,
            max_model_len=arguments.max_model_len,
            cache_block_override=arguments.cache_block_override,
            salt=arguments.salt,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.dry_run:
        return _drive(arguments)
    if not arguments.revision:
        parser.error("--revision is required for a reproducible measurement")
    return _drive(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
