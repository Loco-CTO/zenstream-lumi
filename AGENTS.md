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
