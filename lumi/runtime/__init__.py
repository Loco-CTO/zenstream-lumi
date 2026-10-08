"""Local model runtime adapters supported by Lumi."""

from lumi.runtime.llama_cpp import (
    LUMI_RUNTIME_API_VERSION,
    SUPPORTED_QWEN35_MODELS,
    LlamaCppChatRuntime,
    LlamaCppConfig,
    LlamaCppProtocolError,
    LlamaCppRuntimeError,
    VerifiedModelArtifact,
)
from lumi.runtime.ollama import (
    DEFAULT_QWEN35_MODELS,
    OllamaChatRuntime,
    OllamaHTTPError,
    OllamaModelStatus,
    OllamaProtocolError,
    OllamaRuntimeConfig,
    OllamaRuntimeError,
    OllamaRuntimeStatus,
    OllamaTransportError,
)

__all__ = [
    "DEFAULT_QWEN35_MODELS",
    "LUMI_RUNTIME_API_VERSION",
    "SUPPORTED_QWEN35_MODELS",
    "LlamaCppChatRuntime",
    "LlamaCppConfig",
    "LlamaCppProtocolError",
    "LlamaCppRuntimeError",
    "VerifiedModelArtifact",
    "OllamaChatRuntime",
    "OllamaHTTPError",
    "OllamaModelStatus",
    "OllamaProtocolError",
    "OllamaRuntimeConfig",
    "OllamaRuntimeError",
    "OllamaRuntimeStatus",
    "OllamaTransportError",
]
