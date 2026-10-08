"""Measure first-request and warm local inference latency without downloading weights."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from llama_cpp_live import create_live_runtime, smoke_request


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Name one useful way to organize a film library.")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()
    if not 2 <= arguments.runs <= 20:
        parser.error("--runs must be between 2 and 20")
    return arguments


async def _run(arguments: argparse.Namespace) -> None:
    runtime = create_live_runtime(
        arguments.models_dir,
        arguments.model,
        context_size=arguments.context,
        output_tokens=arguments.output_tokens,
    )
    request = smoke_request(
        arguments.model,
        arguments.prompt,
        context_size=arguments.context,
        output_tokens=arguments.output_tokens,
    )
    await runtime.open()
    durations: list[float] = []
    try:
        for _ in range(arguments.runs):
            started = time.perf_counter()
            await runtime.complete(request)
            durations.append(time.perf_counter() - started)
    finally:
        await runtime.close()
    warm = durations[1:]
    print(
        json.dumps(
            {
                "model": arguments.model,
                "runs": len(durations),
                "firstRequestSeconds": round(durations[0], 3),
                "warmAverageSeconds": round(statistics.mean(warm), 3),
                "warmMedianSeconds": round(statistics.median(warm), 3),
            },
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
