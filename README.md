# Lumi

Lumi is ZenStream's local, read-only conversational media assistant. It uses the user's permission-filtered ZenStream catalog and viewing context together with optional bounded web research, then answers with Markdown, validated ZenStream references, and inspectable sources.

Lumi is developed and released from its own repository as the `zenstream-lumi` Python package. ZenStream Orchestrator installs a supported Lumi release into its managed runtime directory only after an administrator explicitly enables the integration. Orchestrator then imports the installed package and calls it in-process. Lumi source remains in this repository; it is not copied into the Orchestrator source tree, and Lumi does not require a separately deployed service.

The runtime loads supported Qwen3.5 GGUF files lazily into the Orchestrator process through the CPU-only `llama-cpp-python` binding. Its pinned native wheels are optional release assets, so an Orchestrator that has not enabled Lumi does not need the inference runtime. Model weights stay in the host-supplied data directory and are never committed to this repository.

## Runtime package contract

The in-process runtime exposes `LUMI_RUNTIME_API_VERSION`, `LlamaCppChatRuntime`, and `LlamaCppConfig` from `lumi.runtime`. Orchestrator can reject an incompatible Lumi release before making the integration available. The runtime accepts only configured Qwen3.5 model IDs and locally installed, hash-verified GGUF files; it performs no model download and makes no HTTP request for inference.

Install the runtime dependency in an opted-in development environment with:

```powershell
python -m pip install '.[embedded]'
```

The Lumi installer downloads one GGUF from a commit-pinned repository for each supported model and verifies its exact size and SHA-256 before activation. It writes a schema 2 `lumi-model-manifest.json` containing the model identity, source repository, revision, quantization, and GGUF digest. The host passes Lumi a `VerifiedModelArtifact` created by `InstalledModelArtifact.as_verified_model_artifact()`. Before its first native load, Lumi checks the manifest digest, model identity, exact file set, safe paths, and pinned GGUF hash. It caches verified file identities and change timestamps in process, rehashing the model if its fingerprint changes. The native Qwen3.5 chat template handles per-request thinking and tool formatting; only read-only tool definitions are passed to Qwen, and native tool-call output is normalized into Lumi's `ToolCall` contract.

## Building and releasing

The `Lumi release` GitHub Actions workflow runs the package suite on pull requests and publishes a GitHub Release when a matching stable `vMAJOR.MINOR.PATCH` tag is pushed. The tag must match `project.version` in `pyproject.toml`. Each release contains `lumi-runtime.zip` and host-specific, SHA-256 recorded dependency wheels; it never contains Qwen checkpoints or model weights.

The release manifest keeps native runtime wheels separate from the deferred model-download dependency wheels. Orchestrator downloads and loads the CPU runtime only after an administrator enables Lumi. It downloads the small Hugging Face client dependency set only after an administrator explicitly starts a model installation. No checkpoint conversion, PyTorch, Transformers, model API URL, or remote inference service is used. Web research remains optional and is enabled only by the administrator's SearXNG setting.

For local native runtime development, install the optional extra (building llama.cpp from source may require CMake and a C++ compiler):

```powershell
python -m pip install ".[embedded]"
```

Run `python -m unittest discover -s tests -v` for the package suite, `python -m unittest discover -s tests -p "test_release_build.py" -v` for release archive tests, and `ruff check lumi tests scripts` for lint.

## Optional Qwen3.5 model installation

Lumi owns its supported model catalog and installer. `supported_models()` is the single source for supported model IDs, display labels, and thinking support. It currently exposes the pinned official `qwen3.5:0.8b`, `qwen3.5:2b`, and `qwen3.5:4b` models. `Qwen35ModelInstaller.list_models()` adds local installation state and provenance.

The host supplies a model data directory outside the Lumi package and repository. Install the optional model-download dependency with `python -m pip install ".[model-install]"`, then call the public package API only after an administrator explicitly requests installation:

```python
from lumi import Qwen35ModelInstaller, supported_models

options = supported_models()
installer = Qwen35ModelInstaller(data_directory / "lumi-models")
installed = installer.list_models()
artifact = installer.install_model("qwen3.5:0.8b", progress=report_progress)
runtime_artifact = artifact.as_verified_model_artifact()
installer.remove_model("qwen3.5:0.8b")
```

`install_model` is synchronous because it downloads and hashes a multi-gigabyte GGUF. Async hosts should call it from a bounded worker lane. Progress events carry a monotonic `current` value bounded by `total=10000`. Importing Lumi or starting the runtime never downloads weights. The installed GGUF files stay outside this repository.

The release workflow builds a host-specific CPU `llama-cpp-python` wheel for Windows x64, Linux x64, and Linux ARM64 with CPython 3.12, 3.13, and 3.14. The optional live smoke and benchmark scripts use an already installed model and never download one:

```powershell
python scripts/smoke_llama_cpp.py --models-dir C:\path\to\lumi-models --model qwen3.5:0.8b
python scripts/benchmark_llama_cpp.py --models-dir C:\path\to\lumi-models --model qwen3.5:0.8b --runs 3
```

## Conversation and privacy

The web and Android clients continue to use authenticated Orchestrator APIs. Orchestrator derives the current account from its normal session and supplies Lumi only the internal, permission-scoped context it needs. Lumi has no playback, download, delete, metadata-edit, watched-state, favourite-write, or library-write tools. It does not connect directly to the Orchestrator database or media filesystem.

Retrieved webpage content is untrusted evidence. Lumi never sends browser credentials, complete watch history, usernames, private library inventories, or conversation transcripts to external websites. Local model inference does not send prompts, conversation history, or user media context to an external model provider.

## Web research

Web research is optional. When no SearXNG URL is configured, `build_web_research_tools()` returns no tools, while local catalog tools, chat, inference, and model installation remain available. Orchestrator may enable bounded search and page retrieval by supplying an administrator-configured trusted SearXNG origin. Lumi submits focused queries derived from the current question, not private conversation history or ZenStream results. Retrieved pages are untrusted; page requests validate public destinations and redirects, send no Orchestrator credentials, and apply size and timeout limits.

## Development status

The repository contains the Lumi chat core, read-only tool contracts, optional web research, the Qwen3.5 model catalog and installer, local runtime adapters, and tests. Orchestrator release installation, admin controls, client experience, and end-to-end evaluation are developed in their respective repositories and remain separate stages.
