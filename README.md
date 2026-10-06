# Lumi

Lumi is ZenStream's local, read-only conversational media assistant. It uses the user's permission-filtered ZenStream catalog and viewing context together with bounded web research, then answers with Markdown, validated ZenStream references, and inspectable sources.

Lumi is developed and released from its own repository as the `zenstream-lumi` Python package. ZenStream Orchestrator installs a supported Lumi release into its managed runtime directory only after an administrator explicitly enables the integration. Orchestrator then imports the installed package and calls it in-process. Lumi source remains in this repository; it is not copied into the Orchestrator source tree, and Lumi does not require a separately deployed service.

The runtime loads supported Qwen3.5 model files lazily into the Orchestrator process. The ONNX Runtime GenAI dependency is an optional package extra, so an Orchestrator installation that has not enabled Lumi does not need the inference runtime. Model weights stay in Orchestrator-managed data storage and are never committed to this repository.

## Runtime package contract

The in-process runtime exposes `LUMI_RUNTIME_API_VERSION` and `OrtGenAIChatRuntime` from `lumi.runtime`. Orchestrator can reject an incompatible Lumi release before making the integration available. The runtime accepts only configured Qwen3.5 model IDs and local model directories; it performs no model download and makes no HTTP request for inference.

Install the runtime dependency in an opted-in development environment with:

```powershell
python -m pip install '.[embedded]'
```

The Orchestrator model installer verifies every model file against its supported catalog before marking a model ready. It writes a `lumi-model-manifest.json` containing schema version, model ID, format, and each relative file path, byte size, and SHA-256 digest. The Orchestrator passes Lumi a `VerifiedModelArtifact` with the expected manifest digest. Before its first native load, Lumi checks the manifest digest, model identity, exact file set, safe paths, and every declared file hash. It caches the verified file identities and change timestamps in process, rehashing a file when that fingerprint changes; this avoids a full multi-gigabyte reread on an ordinary idle unload/reload cycle. The first model load therefore includes one integrity pass over the installed files. The manifest must include `genai_config.json` and either `chat_template.jinja` or `tokenizer_config.json`. Only read-only tool definitions are passed to Qwen, and native Qwen3.5 tool-call output is normalized into Lumi's `ToolCall` contract.

## Conversation and privacy

The web and Android clients continue to use authenticated Orchestrator APIs. Orchestrator derives the current account from its normal session and supplies Lumi only the internal, permission-scoped context it needs. Lumi has no playback, download, delete, metadata-edit, watched-state, favourite-write, or library-write tools. It does not connect directly to the Orchestrator database or media filesystem.

Retrieved webpage content is untrusted evidence. Lumi never sends browser credentials, complete watch history, usernames, private library inventories, or conversation transcripts to external websites. Local model inference does not send prompts, conversation history, or user media context to an external model provider.

## Web research

Web search remains disabled unless Orchestrator supplies an administrator-configured trusted SearXNG origin. Lumi submits focused queries derived from the current question, not private conversation history or ZenStream results. Retrieved pages are untrusted; page requests validate public destinations and redirects, send no Orchestrator credentials, and apply size and timeout limits.

## Development status

The repository contains the Lumi chat core, read-only tool contracts, local runtime adapters, web research, and tests. Orchestrator release installation, model catalog/download management, admin controls, client experience, and end-to-end evaluation are developed in their respective repositories and remain separate stages.
