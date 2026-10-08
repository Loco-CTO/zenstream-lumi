# Runtime acceleration and streaming

## Release targets

The release workflow builds one `llama-cpp-python==0.3.35` wheel for each Python ABI and supported platform. Each wheel contains the llama.cpp CPU backend. Windows x64 and Linux x64 wheels also build CUDA and Vulkan backends into the same wheel. Linux ARM64 is CPU-only. This keeps CPU as the universal fallback and lets an installed GGUF model file move between CPU and accelerated modes without conversion.

The Windows build uses a pinned Visual Studio 2022 generator, CUDA 12.8.1, and Vulkan SDK 1.4.309.0. The Linux x64 build uses the manylinux 2.28 baseline with CUDA 12.8 and pinned Vulkan headers/loader dependencies. Its `glslc` build tool is compiled inside that same manylinux image from Shaderc v2025.4 commit `73743588fe9c39f2f1c780a087d94afac691a189` and its pinned dependency commits; it does not use LunarG's prebuilt Linux binaries, which require a newer glibc than the manylinux baseline. Linux ARM64 uses the manylinux 2.28 baseline and compiles the CPU backend only. All wheel builds explicitly set `GGML_BACKEND_DL=OFF`; the CUDA/Vulkan backend shared libraries are linked into the ggml core and their registry entries must be present even on driverless runners. Each release wheelhouse job installs its repaired runtime wheel from local wheel directories in a fresh virtual environment, imports `llama_cpp`, initializes and checks the ggml backend registry, and requires the CPU backend. Windows/Linux x64 jobs always require the CUDA and Vulkan shared libraries plus their statically linked registry entries. The smoke check probes for matching GPU hardware and reports a skip when no device is exposed; it does not mistake unavailable CI hardware for a missing packaged backend. Windows dependency repair bundles the CUDA and Vulkan runtime libraries used by the native backend libraries; the NVIDIA driver DLL is intentionally excluded. Linux wheel repair bundles redistributable CUDA/Vulkan libraries and leaves the driver (`libcuda.so.1`) to the host.

GPU inference still requires the appropriate GPU driver installed by the operating system or hardware vendor. That driver requirement is separate from installing a CUDA toolkit, Vulkan SDK, model server, or Lumi-specific runtime by hand. The release wheel supplies the compiled backend and its redistributable user-space libraries. A GPU-less host, unavailable device, unsafe VRAM budget, or failed GPU initialization continues with CPU inference.

CUDA is preferred for NVIDIA devices. Vulkan provides the cross-vendor path where the installed driver exposes a usable device, including supported AMD devices. HIP/ROCm is not packaged: the current installer contract selects one runtime wheel per platform, so a separate same-version ROCm wheel would compete with the Linux CUDA+Vulkan wheel. Safe support needs backend-specific wheel selection and verified dependency closure; the existing release matrix does not provide those. This repository currently has no macOS release target, so Metal is not built or advertised either.

## Administrator modes

- **Automatic** detects available compiled backends and devices, prefers the best usable backend, chooses partial layer offload within a conservative free-VRAM budget, and falls back to CPU when acceleration is unavailable or fails.
- **CPU only** skips Lumi's GPU device-selection probe and disables GPU layer offload, KQV offload, and operation offload while using the same GGUF. The pinned llama-cpp-python 0.3.35 wrapper still calls `llama_backend_init()` inside `Llama.__init__` and does not expose llama.cpp's `--device none` filter or a usable `devices` parameter. The runtime selects CPU and disables all exposed offload paths, but the wrapper may initialize compiled backend registries during model construction; the current binding does not provide a strict guarantee that GPU libraries or drivers are never initialized.
- **GPU preferred** asks Lumi to use the best safe accelerator; CPU remains the recovery path if the device cannot initialize or run inference.

The status API reports the selected backend/device, layer counts, runtime state, and a safe fallback reason. It does not expose llama.cpp command-line flags or model/tool internals to normal users.

## Visible streaming

The runtime yields only visible answer-text deltas while generation is active, then one normalized completion. The visible-text filter withholds analysis/thinking and special model tokens; tool calls stay on Lumi's structured tool path. A reset event contains only a safe `reason`, whose value is `intermediate` or `cpu_fallback`. Both reasons tell the client to clear the current partial assistant turn: `intermediate` marks text that must be discarded before the model continues through hidden or tool control flow, while `cpu_fallback` means GPU inference failed and the runtime is retrying that turn on CPU. Only `cpu_fallback` indicates a CPU retry. Cancellation stops generation at native token boundaries and keeps the model single-flight slot occupied until the worker exits.

Sanitized `delta.content` is emitted as soon as it is generated, including when tool schemas are available. Structured tool names and arguments are never included in deltas. If a later tool call follows visible draft text, Lumi emits an `intermediate` reset before returning the normalized tool call, so the client clears that draft before tool execution and the next assistant turn. Tool-enabled turns therefore keep incremental first-visible-token behavior while preserving the reset transition.

Clients should append delta text exactly as received, clear only the active turn on either reset reason, and use the complete event to finalize the assistant turn. Markdown can be incomplete while it streams (for example, an open code fence); render it incrementally with the same Markdown renderer and finalize the existing content on completion instead of appending the final answer a second time.

## Benchmarking

Install the optional embedded dependency and use an already installed 2B model for a baseline comparison:

```powershell
python -m pip install ".[embedded]"
python scripts/benchmark_llama_cpp.py `
  --models-dir C:\path\to\lumi-models `
  --model qwen3.5:2b `
  --thinking `
  --with-tools `
  --runs 4 `
  --json-out lumi-benchmark.json
```

The default prompt is Chinese. Add `--thinking` to exercise Qwen3.5's hidden reasoning path and `--with-tools` to offer the read-only `catalog_search` schema while measuring incremental text and tool-call resets. The runtime benchmark does not execute tool implementations; use the Orchestrator/UI benchmark for that. The default comparison runs `cpu_only` and `automatic` with the same model and prompt. Add repeated `--mode automatic --mode cpu_only --mode gpu_preferred` arguments to choose a different order or include GPU-preferred mode. The first run includes lazy model loading and cold filesystem effects; compare the reported warm averages for inference performance. The report includes:

- time to first non-whitespace visible text and time to the first text delta;
- prompt processing and generation duration/token rates when the binding supplies those metrics;
- end-to-end embedded-runtime latency and exact model/revision/quantization provenance;
- peak process resident memory where the OS exposes it;
- best-effort NVIDIA system-wide VRAM use when `nvidia-smi` is available; this may include other processes and is not per-process attribution;
- selected backend and fallback status for each mode.

System-wide NVIDIA VRAM changes observed during CPU-only trials are unattributed host measurements. They may reflect other processes, device initialization, or system activity and must not be reported as Lumi's CPU-mode VRAM use.

Unavailable native timing or resource probes are reported as `null`, not as zero. `runtimeEndToEndMs` measures the embedded model call only. Lumi tool execution, web retrieval, Orchestrator persistence, HTTP/SSE delivery, and browser paint need a separate service/UI benchmark on the running application. Measure on an otherwise idle host when comparing memory. Model weights and benchmark outputs belong in local data/output directories, not the repository.
