"""Shared helpers for opt-in live llama.cpp checks using installed local models."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumi.contracts import ChatMessage, ModelRequest, ToolDefinition  # noqa: E402
from lumi.model_installation import MODEL_MANIFEST_FILENAME, supported_models  # noqa: E402
from lumi.runtime import (  # noqa: E402
    LlamaCppChatRuntime,
    LlamaCppConfig,
    VerifiedModelArtifact,
)


def create_live_runtime(
    models_directory: Path,
    model_id: str,
    *,
    context_size: int,
    output_tokens: int,
    acceleration_mode: str = "automatic",
) -> LlamaCppChatRuntime:
    """Create a runtime for one already installed and locally verified model."""

    supported = {option.model_id for option in supported_models()}
    if model_id not in supported:
        raise ValueError("Choose a model from Lumi's supported Qwen3.5 catalog")
    if not 256 <= context_size <= 32_768:
        raise ValueError("Context size must be between 256 and 32768 tokens")
    if not 1 <= output_tokens <= 8_192:
        raise ValueError("Output tokens must be between 1 and 8192")

    model_directory = models_directory.expanduser().resolve() / model_id.replace(":", "-")
    manifest_path = model_directory / MODEL_MANIFEST_FILENAME
    if model_directory.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Install the selected Lumi model before running this check")
    try:
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("The selected Lumi model manifest cannot be read") from error

    artifact = VerifiedModelArtifact(model_id, model_directory, manifest_digest)
    return LlamaCppChatRuntime(
        LlamaCppConfig(
            model_artifacts={model_id: artifact},
            max_context_tokens=context_size,
            max_output_tokens=output_tokens,
            n_ctx=context_size,
            n_batch=min(512, context_size),
            n_ubatch=min(512, context_size),
            acceleration_mode=acceleration_mode,
        )
    )


def smoke_request(
    model_id: str,
    prompt: str,
    *,
    context_size: int,
    output_tokens: int,
    thinking: bool = False,
    with_tools: bool = False,
) -> ModelRequest:
    tools = (
        (
            ToolDefinition(
                "catalog_search",
                "Search the local media catalog for matching titles.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                data_scope="local",
                read_only=True,
            ),
        )
        if with_tools
        else ()
    )
    return ModelRequest(
        model=model_id,
        messages=[ChatMessage(role="user", content=prompt)],
        tools=tools,
        thinking=thinking,
        context_size=context_size,
        output_tokens=output_tokens,
    )
