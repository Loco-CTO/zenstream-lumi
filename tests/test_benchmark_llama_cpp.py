from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

import benchmark_llama_cpp  # noqa: E402
from llama_cpp_live import smoke_request  # noqa: E402


class _FakeRuntime:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events
        self._loaded = SimpleNamespace(model=_FakeModel())

    async def stream(self, request: object):
        del request
        for event in self._events:
            yield event


class _FakeModel:
    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[str]:
        del add_bos, special
        return text.decode("utf-8").split()


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(kind="delta", text=text)


def _reset(reason: str) -> SimpleNamespace:
    return SimpleNamespace(kind="reset", reason=reason)


def _complete(content: str) -> SimpleNamespace:
    response = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=[]),
        prompt_tokens=3,
        completion_tokens=2,
        prompt_duration_ns=None,
        generation_duration_ns=2_000_000,
        load_duration_ns=None,
        total_duration_ns=None,
    )
    return SimpleNamespace(kind="complete", response=response)


def _run_immediate(coroutine):
    """Drive the fake stream coroutine without opening a platform event loop."""

    try:
        result = coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise AssertionError(f"fake runtime unexpectedly suspended: {result!r}")


class BenchmarkStreamMeasurementTests(unittest.TestCase):
    def test_json_report_uses_utf8_bytes_with_cp1252_console(self) -> None:
        buffer = io.BytesIO()
        console = io.TextIOWrapper(buffer, encoding="cp1252")

        benchmark_llama_cpp._write_utf8_report('{"text":"本地推理"}', console)

        self.assertEqual(buffer.getvalue(), '{"text":"本地推理"}\n'.encode())

    def test_smoke_request_can_use_chinese_tool_capable_thinking_path(self) -> None:
        prompt = "请用简短的中文说明本地推理的隐私优势。"
        request = smoke_request(
            "qwen3.5:2b",
            prompt,
            context_size=2048,
            output_tokens=48,
            thinking=True,
            with_tools=True,
        )

        self.assertEqual(request.messages[0].content, prompt)
        self.assertTrue(request.thinking)
        self.assertEqual([tool.name for tool in request.tools], ["catalog_search"])
        self.assertEqual(request.tools[0].data_scope, "local")
        self.assertTrue(request.tools[0].read_only)

    def test_reset_discards_prefix_and_restarts_first_delta_timestamps(self) -> None:
        runtime = _FakeRuntime(
            [
                _delta("gpu prefix"),
                _reset("intermediate"),
                _delta("tool prefix"),
                _reset("cpu_fallback"),
                _delta(" "),
                _delta("answer"),
                _complete(" answer"),
            ]
        )
        timestamps = [
            1_000_000,  # run start
            1_500_000,  # discarded GPU text
            2_000_000,  # first reset
            2_500_000,  # discarded tool text
            3_000_000,  # second reset
            5_000_000,  # first delta after the final reset
            6_000_000,  # first visible non-whitespace text
            7_000_000,  # completion event
            8_000_000,  # run end
        ]

        with patch.object(
            benchmark_llama_cpp.time, "perf_counter_ns", side_effect=timestamps
        ):
            result = _run_immediate(
                benchmark_llama_cpp._measure_run(runtime, object(), 1)
            )

        self.assertEqual(result["timeToFirstTextDeltaMs"], 4.0)
        self.assertEqual(result["timeToFirstVisibleTokenMs"], 5.0)
        self.assertEqual(result["partialStreamResets"], 2)
        self.assertEqual(result["visibleCharacterCount"], len(" answer"))
        self.assertEqual(result["visibleOutputTokens"], 1)
        self.assertEqual(result["visibleOutputTokensPerSecond"], 500.0)
        self.assertTrue(result["streamTextMatchesCompletion"])

    def test_stream_completion_text_mismatch_fails_acceptance(self) -> None:
        runtime = _FakeRuntime([_delta("streamed"), _complete("normalized")])
        timestamps = [1_000_000, 1_500_000, 2_000_000, 3_000_000]

        with patch.object(
            benchmark_llama_cpp.time, "perf_counter_ns", side_effect=timestamps
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Visible stream differs from completed response"
            ):
                _run_immediate(
                    benchmark_llama_cpp._measure_run(runtime, object(), 2)
                )


class AmdGpuMemoryMeasurementTests(unittest.TestCase):
    def test_amd_smi_json_sums_reported_device_memory_in_megabytes(self) -> None:
        output = json.dumps(
            [
                {"gpu": 0, "vram_used": {"value": 12, "unit": "MB"}},
                {"gpu": 1, "vram_used": {"value": 20, "unit": "MiB"}},
            ]
        )

        used_bytes = benchmark_llama_cpp._parse_amd_gpu_memory_json(
            output, default_unit="MB"
        )

        self.assertEqual(used_bytes, 32 * 1024**2)

    def test_nested_apu_gtt_memory_is_reported_instead_of_claiming_vram(self) -> None:
        output = json.dumps(
            {"gpu": 0, "gtt_usage": {"used": {"value": 1.5, "unit": "GB"}}}
        )

        used_bytes = benchmark_llama_cpp._parse_amd_gpu_memory_json(
            output, default_unit="MB"
        )

        self.assertEqual(used_bytes, round(1.5 * 1024**3))

    def test_legacy_rocm_smi_byte_fields_are_supported(self) -> None:
        output = json.dumps(
            {
                "card0": {
                    "VRAM Total Memory (B)": 8_000_000,
                    "VRAM Used Memory (B)": 1_000_000,
                },
                "card1": {
                    "VRAM Total Memory (B)": 8_000_000,
                    "VRAM Used Memory (B)": 2_000_000,
                },
            }
        )

        used_bytes = benchmark_llama_cpp._parse_amd_gpu_memory_json(
            output, default_unit="B"
        )

        self.assertEqual(used_bytes, 3_000_000)

    def test_amd_probe_uses_amd_smi_and_reports_its_source(self) -> None:
        output = json.dumps([{"gpu": 0, "vram_used": {"value": 7, "unit": "MB"}}])
        with (
            patch.object(
                benchmark_llama_cpp.shutil,
                "which",
                side_effect=lambda command: "amd-smi.exe" if command == "amd-smi" else None,
            ),
            patch.object(
                benchmark_llama_cpp.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=output),
            ) as run,
        ):
            sample = benchmark_llama_cpp._system_amd_gpu_memory_sample()

        self.assertEqual(sample, ("amd-smi", 7 * 1024**2))
        self.assertEqual(run.call_args.args[0][1:], ["monitor", "--vram-usage", "--json"])
        self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_amd_probe_falls_back_to_legacy_rocm_smi(self) -> None:
        output = json.dumps({"card0": {"VRAM Used Memory (B)": 1_234}})
        with (
            patch.object(
                benchmark_llama_cpp.shutil,
                "which",
                side_effect=lambda command: "rocm-smi.exe" if command == "rocm-smi" else None,
            ),
            patch.object(
                benchmark_llama_cpp.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=output),
            ) as run,
        ):
            sample = benchmark_llama_cpp._system_amd_gpu_memory_sample()

        self.assertEqual(sample, ("rocm-smi", 1_234))
        self.assertEqual(
            run.call_args.args[0][1:], ["--showmeminfo", "vram", "--json"]
        )

    def test_unavailable_or_unrecognized_amd_metrics_stay_unmeasured(self) -> None:
        with patch.object(benchmark_llama_cpp.shutil, "which", return_value=None):
            self.assertIsNone(benchmark_llama_cpp._system_amd_gpu_memory_sample())

        self.assertIsNone(
            benchmark_llama_cpp._parse_amd_gpu_memory_json(
                '{"gpu": 0, "vram_percent": 50}', default_unit="MB"
            )
        )

    def test_resource_report_identifies_amd_source_and_systemwide_values(self) -> None:
        sampler = benchmark_llama_cpp._ResourceSampler(interval_seconds=1)
        sampler.amd_memory_before = 1_000
        sampler.amd_memory_peak = 2_500
        sampler.amd_memory_sources.add("amd-smi")

        result = sampler.result()

        self.assertEqual(result["systemAmdGpuMemoryUsedBeforeBytes"], 1_000)
        self.assertEqual(result["systemAmdGpuMemoryUsedPeakBytes"], 2_500)
        self.assertEqual(result["systemAmdGpuMemoryUsedIncreaseBytes"], 1_500)
        self.assertEqual(result["systemAmdGpuMemoryMeasurementSource"], "amd-smi")


if __name__ == "__main__":
    unittest.main()
