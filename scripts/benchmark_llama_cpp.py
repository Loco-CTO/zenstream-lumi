"""Compare local CPU and automatic llama.cpp modes with visible-stream timings.

The script uses an already installed and verified GGUF. It never downloads a model.
Run it on an idle machine for the most useful RAM and VRAM measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from llama_cpp_live import create_live_runtime, smoke_request

_DEFAULT_MODES = ("cpu_only", "automatic")
_VALID_MODES = ("automatic", "cpu_only", "gpu_preferred")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument(
        "--mode",
        choices=_VALID_MODES,
        action="append",
        help="Repeat to compare selected modes; defaults to CPU-only and automatic.",
    )
    parser.add_argument(
        "--prompt",
        default="请用简短的中文说明在本地运行模型为何有助于保护隐私。",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking while measuring only visible answer text.",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Offer a read-only catalog_search schema to exercise tool-capable streaming.",
    )
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path)
    arguments = parser.parse_args()
    if not 2 <= arguments.runs <= 20:
        parser.error("--runs must be between 2 and 20 (including the cold first run)")
    if not 0.1 <= arguments.sample_seconds <= 10:
        parser.error("--sample-seconds must be between 0.1 and 10")
    if not arguments.prompt.strip():
        parser.error("--prompt must not be empty")
    arguments.modes = tuple(dict.fromkeys(arguments.mode or _DEFAULT_MODES))
    return arguments


def _process_rss_bytes() -> int | None:
    """Read current resident memory with standard OS interfaces when available."""

    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            current_process = kernel32.GetCurrentProcess
            current_process.restype = wintypes.HANDLE
            get_memory = psapi.GetProcessMemoryInfo
            get_memory.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if get_memory(current_process(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return None

    statm_path = Path("/proc/self/statm")
    if statm_path.is_file():
        try:
            fields = statm_path.read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            pass

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, RuntimeError):
        return None


def _system_nvidia_vram_used_bytes() -> int | None:
    """Sample system-wide NVIDIA use; nvidia-smi does not attribute process use here."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        try:
            values.append(int(line.strip()) * 1024 * 1024)
        except ValueError:
            continue
    return sum(values) if values else None


class _ResourceSampler:
    """Best-effort periodic RAM and NVIDIA VRAM sampling for one mode trial."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_before: int | None = None
        self.rss_peak: int | None = None
        self.vram_before: int | None = None
        self.vram_peak: int | None = None

    def _sample(self) -> None:
        rss = _process_rss_bytes()
        vram = _system_nvidia_vram_used_bytes()
        if rss is not None:
            self.rss_peak = max(self.rss_peak or rss, rss)
        if vram is not None:
            self.vram_peak = max(self.vram_peak or vram, vram)

    def start(self) -> None:
        self.rss_before = _process_rss_bytes()
        self.vram_before = _system_nvidia_vram_used_bytes()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="lumi-benchmark-sampler")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval_seconds * 3))
        self._sample()

    def result(self) -> dict[str, int | None]:
        rss_delta = (
            max(0, self.rss_peak - self.rss_before)
            if self.rss_peak is not None and self.rss_before is not None
            else None
        )
        vram_delta = (
            max(0, self.vram_peak - self.vram_before)
            if self.vram_peak is not None and self.vram_before is not None
            else None
        )
        return {
            "processRssBeforeBytes": self.rss_before,
            "processRssPeakBytes": self.rss_peak,
            "processRssPeakIncreaseBytes": rss_delta,
            "systemNvidiaVramUsedBeforeBytes": self.vram_before,
            "systemNvidiaVramUsedPeakBytes": self.vram_peak,
            "systemNvidiaVramUsedIncreaseBytes": vram_delta,
        }


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _milliseconds(nanoseconds: int | None) -> float | None:
    if nanoseconds is None or nanoseconds < 0:
        return None
    return round(nanoseconds / 1_000_000, 3)


def _tokens_per_second(tokens: int | None, duration_ns: int | None) -> float | None:
    if tokens is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(tokens * 1_000_000_000 / duration_ns, 3)


def _model_provenance(models_directory: Path, model_id: str) -> dict[str, Any]:
    model_directory = models_directory.expanduser().resolve() / model_id.replace(":", "-")
    manifest_path = model_directory / "lumi-model-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = manifest.get("source") if isinstance(manifest, dict) else None
    source = source if isinstance(source, dict) else {}
    return {
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "repository": source.get("repositoryId"),
        "revision": source.get("revision"),
        "quantization": source.get("quantization"),
        "ggufSha256": source.get("sha256"),
    }


async def _measure_run(runtime: Any, request: Any, index: int) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    first_delta_ns: int | None = None
    first_non_whitespace_ns: int | None = None
    visible_parts: list[str] = []
    reset_count = 0
    response: Any | None = None
    async for event in runtime.stream(request):
        now_ns = time.perf_counter_ns()
        if event.kind == "reset":
            reset_count += 1
            visible_parts.clear()
            first_delta_ns = None
            first_non_whitespace_ns = None
        elif event.kind == "delta":
            if first_delta_ns is None:
                first_delta_ns = now_ns
            if first_non_whitespace_ns is None and any(
                not character.isspace() for character in event.text
            ):
                first_non_whitespace_ns = now_ns
            visible_parts.append(event.text)
        elif event.kind == "complete":
            response = event.response

    completed_ns = time.perf_counter_ns()
    if response is None:
        raise RuntimeError("The inference stream ended without a completion event")

    message = response.message
    prompt_tokens = _optional_int(getattr(response, "prompt_tokens", None))
    completion_tokens = _optional_int(getattr(response, "completion_tokens", None))
    prompt_duration_ns = _optional_int(getattr(response, "prompt_duration_ns", None))
    generation_duration_ns = _optional_int(
        getattr(response, "generation_duration_ns", None)
    )
    load_duration_ns = _optional_int(getattr(response, "load_duration_ns", None))
    total_duration_ns = _optional_int(getattr(response, "total_duration_ns", None))
    assembled_text = "".join(visible_parts)
    stream_matches_completion = assembled_text == getattr(message, "content", None)
    if not stream_matches_completion:
        raise RuntimeError(f"Visible stream differs from completed response in run {index}")

    return {
        "run": index,
        "coldRuntimeRun": index == 1,
        "timeToFirstTextDeltaMs": _milliseconds(
            first_delta_ns - started_ns if first_delta_ns is not None else None
        ),
        "timeToFirstVisibleTokenMs": _milliseconds(
            first_non_whitespace_ns - started_ns
            if first_non_whitespace_ns is not None
            else None
        ),
        "runtimeEndToEndMs": _milliseconds(completed_ns - started_ns),
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "promptProcessingMs": _milliseconds(prompt_duration_ns),
        "promptTokensPerSecond": _tokens_per_second(prompt_tokens, prompt_duration_ns),
        "generationMs": _milliseconds(generation_duration_ns),
        "generationTokensPerSecond": _tokens_per_second(
            completion_tokens, generation_duration_ns
        ),
        "modelLoadMs": _milliseconds(load_duration_ns),
        "runtimeReportedTotalMs": _milliseconds(total_duration_ns),
        "visibleCharacterCount": len(assembled_text),
        "streamTextMatchesCompletion": stream_matches_completion,
        "partialStreamResets": reset_count,
        "toolCalls": len(getattr(message, "tool_calls", ()) or ()),
    }


def _warm_summary(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    warm_runs = runs[1:]
    fields = (
        "timeToFirstTextDeltaMs",
        "timeToFirstVisibleTokenMs",
        "runtimeEndToEndMs",
        "promptProcessingMs",
        "generationMs",
        "generationTokensPerSecond",
    )
    summary: dict[str, float | None] = {}
    for field in fields:
        values = [value[field] for value in warm_runs if isinstance(value[field], (int, float))]
        summary[field] = round(statistics.mean(values), 3) if values else None
    return summary


def _write_utf8_report(serialized: str, stream: Any | None = None) -> None:
    """Write JSON as UTF-8 even when a Windows console defaults to cp1252."""

    output = stream or sys.stdout
    payload = (serialized + "\n").encode("utf-8")
    binary = getattr(output, "buffer", None)
    if binary is not None:
        binary.write(payload)
        binary.flush()
        return

    # StringIO and other test/embedding streams may not expose a byte buffer.
    try:
        output.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass
    output.write(payload.decode("utf-8"))
    output.flush()


def _host_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logicalCpuCount": os.cpu_count(),
        "python": platform.python_version(),
    }


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    provenance = _model_provenance(arguments.models_dir, arguments.model)
    request = smoke_request(
        arguments.model,
        arguments.prompt,
        context_size=arguments.context,
        output_tokens=arguments.output_tokens,
        thinking=arguments.thinking,
        with_tools=arguments.with_tools,
    )
    mode_results: list[dict[str, Any]] = []
    for mode in arguments.modes:
        runtime = create_live_runtime(
            arguments.models_dir,
            arguments.model,
            context_size=arguments.context,
            output_tokens=arguments.output_tokens,
            acceleration_mode=mode,
        )
        await runtime.open()
        sampler = _ResourceSampler(arguments.sample_seconds)
        sampler.start()
        runs: list[dict[str, Any]] = []
        try:
            for run_index in range(1, arguments.runs + 1):
                runs.append(await _measure_run(runtime, request, run_index))
            acceleration_status = runtime.acceleration_status()
        finally:
            sampler.stop()
            await runtime.close()
        mode_results.append(
            {
                "mode": mode,
                "accelerationStatus": acceleration_status,
                "coldRun": runs[0],
                "warmAverage": _warm_summary(runs),
                "runs": runs,
                "resources": sampler.result(),
            }
        )

    return {
        "schemaVersion": 2,
        "model": {
            "id": arguments.model,
            **provenance,
        },
        "request": {
            "promptCharacters": len(arguments.prompt),
            "contextTokens": arguments.context,
            "maxOutputTokens": arguments.output_tokens,
            "runsPerMode": arguments.runs,
            "thinking": arguments.thinking,
            "toolsOffered": len(request.tools),
        },
        "host": _host_info(),
        "resourceMeasurementNotes": [
            "Process RSS is sampled from the OS when the platform exposes it.",
            "NVIDIA VRAM is sampled system-wide with nvidia-smi and may include other processes.",
            "AMD/ROCm VRAM is unavailable; this harness does not currently have a rocm-smi probe.",
            (
                "llama-cpp-python 0.3.35 does not expose streamed chat completion usage counts; "
                "completion token count and tokens per second are unmeasured."
            ),
            (
                "Null resource or runtime timing values mean that platform or binding support "
                "was unavailable."
            ),
            (
                "Runtime end-to-end time covers embedded model inference only, not Lumi tools, "
                "research, "
                "network transport, or browser paint."
            ),
            (
                "Compare warm averages across modes; the cold run also includes lazy model loading "
                "and "
                "OS file-cache effects."
            ),
        ],
        "modes": mode_results,
    }


def main() -> None:
    arguments = _arguments()
    report = asyncio.run(_run(arguments))
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.json_out is not None:
        arguments.json_out.expanduser().write_text(serialized + "\n", encoding="utf-8")
    _write_utf8_report(serialized)


if __name__ == "__main__":
    main()
