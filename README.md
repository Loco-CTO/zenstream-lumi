<div align="center">
  <h1 align="center">Lumi</h1>
  <p>ZenStream’s optional, locally run conversational media assistant.</p>
</div>

Lumi answers with permission-filtered catalog context, bounded public web research, Markdown, validated media references, and source links.

## How it fits together

Lumi is a Python package imported in-process by a compatible ZenStream Orchestrator release after an administrator enables it. Orchestrator owns authentication, account context, permissions, installation, and the user interface. Lumi owns chat orchestration, read-only tools, model runtime integration, and answer validation. It is not a separate service and has no user interface.

Model weights live in the host’s data directory outside this repository. Inference uses a locally installed Qwen3.5 GGUF through the optional `llama-cpp-python` runtime; Lumi does not call a hosted inference API or download weights during startup or inference.

## Configuration

Configuration and administrator controls belong to Orchestrator. Lumi receives the authenticated account context and bounded read-only adapters from the host. Model selection, context limits, backend mode, and model installation are configured through the host; see [architecture](docs/architecture.md) and [model installation](docs/model-installation.md).

## Quick start

For local package development, install the optional embedded runtime extra when native inference is needed:

```powershell
python -m pip install ".[embedded]"
```

Model installation is a separate, explicit operation and requires the `model-install` extra. It downloads a pinned, hash-verified model to a host-supplied data directory. See [model installation and provenance](docs/model-installation.md) before installing models.

## Deployment

Deploy Lumi only as a compatible Orchestrator-managed package release. The host downloads and verifies the release and optional native runtime wheels; model data is installed separately after an administrator requests it. See [release process](docs/releasing.md).

## Development checks

```powershell
python -m unittest discover -s tests -v
ruff check lumi tests scripts
python -m unittest discover -s tests -p "test_release_build.py" -v
```

The synthetic quality suite uses already installed models and local fixtures. Its results and model files must stay outside Git; see [quality evaluation](docs/quality-evaluation.md).

## Developer documentation

- [Documentation index](docs/README.md)
- [Architecture and ownership](docs/architecture.md)
- [Runtime acceleration and streaming](docs/runtime-acceleration-and-streaming.md)
- [Model installation and provenance](docs/model-installation.md)
- [Quality evaluation and benchmarking](docs/quality-evaluation.md)
- [Privacy and web research](docs/privacy-and-web-research.md)
- [Release process](docs/releasing.md)

## Troubleshooting

- **Lumi is unavailable:** confirm the Orchestrator release supports the Lumi runtime API and that an administrator enabled the integration.
- **No model is available:** install a supported model through the host and check its managed model-data directory.
- **Native backend fails to load:** confirm the host platform, Python ABI, runtime release, and GPU driver match the published support matrix. CPU remains the fallback.

## Releases

Stable package tags use `vMAJOR.MINOR.PATCH` and match `project.version` in `pyproject.toml`. Release assets contain the Lumi runtime and platform-specific dependency wheels, never model weights. See [release process](docs/releasing.md).

## License

See [LICENSE](LICENSE).
