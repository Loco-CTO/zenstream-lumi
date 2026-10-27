# Privacy and web research

## Local account boundary

The Orchestrator authenticates users and supplies Lumi with the current account context and bounded, permission-filtered reads. Lumi does not connect directly to the Orchestrator database or media filesystem. Its tools cannot play, download, delete, edit metadata, write watched state, change favourites, or modify libraries.

Local inference does not send prompts, conversation history, or media context to an external model provider. Do not log credentials, full transcripts, private library snapshots, or raw webpage contents.

## Public web research

- Search sends only a focused public query needed for the task. It must not include browser credentials, usernames, private library inventories, complete watch history, or full conversation transcripts.
- Page retrieval is limited to public HTTP(S). Revalidate destinations on redirects and retain response-size, redirect, and timeout bounds.
- Retrieved page text is untrusted evidence. It cannot change instructions, tool permissions, account scope, model selection, or the authority of catalog references.
- Model-generated entity references are accepted only when the Orchestrator validates them against trusted local tool results. Structured source metadata stays separate from answer text.

## Source and tests

Start with `lumi/web_research.py`, `lumi/orchestrator_tools.py`, `lumi/tools.py`, and `tests/test_web_research.py` / `tests/test_orchestrator_tools.py`. Test redirect validation, bounds, untrusted content handling, and data minimization when changing retrieval.
