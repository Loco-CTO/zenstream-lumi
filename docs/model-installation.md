# Model installation and provenance

## Ownership

Lumi owns its supported model catalog and installer in `lumi/model_catalog.py` and `lumi/model_installation.py`. The compatible Orchestrator host owns the user-facing administrator action, the configured data directory, dependency installation, and permission to begin a download.

## Install boundary

- The model catalog is the source for supported model IDs, labels, and capabilities. Do not maintain a second list in UI or host code.
- Model weights live in the host-supplied data directory outside the package and repository. Importing Lumi and starting inference do not download weights.
- Install the optional dependency only for an explicit model-installation operation: `python -m pip install '.[model-install]'`.
- `Qwen35ModelInstaller.install_model()` downloads from a commit-pinned source and verifies exact file size and SHA-256 before activation. Installation is synchronous because the model is large; async hosts must call it on a bounded worker lane and forward bounded progress.
- The installer writes a schema-versioned manifest with model identity, repository revision, quantization, and digest. The host passes a `VerifiedModelArtifact`; the runtime independently validates its manifest, exact file set, safe paths, model identity, and pinned digest before native load.

## Runtime boundary

The embedded runtime consumes an already installed and verified GGUF. It does not download a model, convert checkpoints, or call a remote inference endpoint. Keep build-time runtime wheels and deferred model-download dependencies separate; see [release process](releasing.md).

## Source and tests

Start with `lumi/model_catalog.py`, `lumi/model_installation.py`, the `VerifiedModelArtifact` contract, and `tests/test_model_installation.py`. Keep provenance and integrity checks covered when changing supported IDs, manifests, or installer paths.
