# Contributing

This repository owns the ZenStream Optional local assistant runtime. Start with the [README](README.md) and [documentation index](docs/README.md).

## Choose the repository

- [zenstream](https://github.com/Loco-CTO/zenstream/issues): Web client.
- [zenstream-mobile](https://github.com/Loco-CTO/zenstream-mobile/issues): Android client.
- [zenstream-orchestrator](https://github.com/Loco-CTO/zenstream-orchestrator/issues): API, services, dashboard, and Windows launcher.
- [zenstream-lumi](https://github.com/Loco-CTO/zenstream-lumi/issues): Optional local assistant runtime.

Report problems where the behavior is owned. If ownership is unclear, include the affected clients and server versions; maintainers can route the report. Link companion issues instead of copying the same report into all repositories.

## Report issues and ask questions

Search existing open and closed issues first. Add evidence to an existing matching report, or use the bug, feature, documentation, or question form in [New issue](https://github.com/Loco-CTO/zenstream-lumi/issues/new/choose). Questions are welcome through the question form; no separate discussion service is required.

For bugs, provide versions, reproduction steps, expected and actual behavior, and relevant redacted evidence. Unknown technical details are acceptable. For features, explain the problem, alternatives, and how the result could be verified.

Remove credentials, access tokens, private media paths, personal data, and conversation content from public logs and attachments. Do not post exploit details or secrets in public issues. Consult the repository's GitHub Security tab for available private reporting options.

## Set up development

Use a supported Python version from pyproject.toml. For package development run `python -m pip install -e ".[dev,service]"`. Native inference and model installation are optional; follow [model installation](docs/model-installation.md) before adding those extras. Keep weights outside the repository.

Read [architecture and ownership](docs/architecture.md) and the focused guides for the affected area. The README remains the source for configuration and deployment instructions.

## Make a focused change

Keep changes scoped to one problem. Discuss substantial new capabilities or architectural changes in an issue before investing in a large implementation; small fixes and documentation corrections may go directly to a PR.

Use a task branch for contributions. Agents must follow the applicable workspace Git instructions and [AGENTS.md](AGENTS.md). Preserve unrelated changes and keep generated artifacts reproducible with the existing generators.

Use concise conventional commit and PR titles such as `fix(playback): preserve the selected audio track` or `docs: clarify installation`. Use lowercase, present-tense subjects without a final period. Explain the problem and resulting behavior in the description.

Describe runtime API/host compatibility for integration changes. For inference changes, record backend/model and relevant quality or performance evidence; see docs/quality-evaluation.md. Do not download weights as part of routine validation. Lumi has no UI.

## Validate the change

Run the checks relevant to the affected code from the repository root unless noted:

```sh
python -m unittest discover -s tests -v
ruff check lumi tests scripts
python -m unittest discover -s tests -p "test_release_build.py" -v
```

Use focused tests while developing and the relevant existing CI checks before submission. Add regression coverage when it meaningfully verifies behavior. Documentation-only changes need link, formatting, and template validation rather than application builds. Record checks that could not run and why; do not claim unexecuted checks passed.

## Submit and review

Use the PR template to explain the problem, resulting behavior, related work, and validation. Include UI evidence where applicable and describe compatibility, migration, or release-order requirements. Remove optional sections that do not apply. Contributors are responsible for understanding and verifying all submitted changes, including generated code.

For coordinated changes, link companion issues/PRs with full repository URLs and identify the owning contract and release order. A merge does not imply that the change is already in a released version.

Maintainers use the [triage guide](docs/triage.md) to classify and prioritize work. Labels describe scope and assessment; they do not guarantee acceptance or a release date.
