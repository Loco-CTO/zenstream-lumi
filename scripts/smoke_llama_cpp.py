"""Run one opt-in local inference check against an already installed Qwen GGUF."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from llama_cpp_live import create_live_runtime, smoke_request


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Reply with a short sentence confirming that local inference works.",
    )
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=64)
    return parser.parse_args()


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
    started = time.perf_counter()
    try:
        response = await runtime.complete(request)
    finally:
        await runtime.close()
    print(
        json.dumps(
            {
                "model": arguments.model,
                "durationSeconds": round(time.perf_counter() - started, 3),
                "answer": response.message.content,
                "toolCalls": len(response.message.tool_calls),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
