# AGENTS.md

## Ownership and model boundaries

- Lumi is independently released; hosts dynamically load a supported release instead of vendoring Lumi.
- Optional installs live in `lumi/model_installation.py`. Import and startup never download checkpoints.
- Require explicit admin action; install only IDs from `lumi.supported_models()`.
- Store files outside the package and repository. The host registers the returned digest-bound artifact.
- Keep source pins, integrity checks, activation, and removal owned by Lumi. Never commit model files.
- Register web research tools only when `WebResearchConfig.searxng_url` is set; no search URL must leave local chat, inference, and model installation available without web tools.

## Verification

- Installer tests use fake fetchers/converters and temporary directories; they never download weights.
- Run `python -m unittest tests.test_model_installation` after installer changes.
- The wider suite also needs the optional service dependencies.

## Release artifacts

- `scripts/build_release.py` assembles `lumi-runtime.zip` and its versioned manifest from the Lumi Python package and a wheelhouse. It rejects tags that do not exactly match `project.version` in `pyproject.toml`.
- `.github/workflows/lumi-release.yml` validates pull requests and publishes stable `vMAJOR.MINOR.PATCH` releases. Pin GitHub Actions to full commit SHAs.
- Keep runtime wheels separate from installer wheels in `lumi-release.json`. Orchestrator must fetch installer wheels only after an explicit model-install action.
- Release assets contain package files and dependency wheels only. Never include model checkpoints, model caches, benchmark output, or temporary conversion data.
- Before a release change is merged, run `python -m unittest discover -s tests -v`, `python scripts/build_release.py --tag v0.1.0 --check-tag`, and `ruff check lumi tests scripts`.
