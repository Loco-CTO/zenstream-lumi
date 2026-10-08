"""Lumi's local conversational media assistant core."""

from lumi.model_installation import (
    INSTALL_PROGRESS_TOTAL,
    MODEL_INSTALL_API_VERSION,
    InstalledModelArtifact,
    ModelInstallationCancelledError,
    ModelInstallationError,
    ModelInstallationUnavailableError,
    ModelInstallOption,
    ModelInstallProgress,
    Qwen35ModelInstaller,
    UnsupportedModelError,
    supported_models,
)

LUMI_PLUGIN_API_VERSION = 1

__all__ = [
    "INSTALL_PROGRESS_TOTAL",
    "InstalledModelArtifact",
    "LUMI_PLUGIN_API_VERSION",
    "MODEL_INSTALL_API_VERSION",
    "ModelInstallOption",
    "ModelInstallationCancelledError",
    "ModelInstallationError",
    "ModelInstallationUnavailableError",
    "ModelInstallProgress",
    "Qwen35ModelInstaller",
    "UnsupportedModelError",
    "supported_models",
]
