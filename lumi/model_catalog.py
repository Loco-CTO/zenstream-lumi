"""Admin-selected vanilla Qwen3.5 model options exposed to Lumi users."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

_OFFICIAL_MODEL_TAGS = frozenset({"0.8b", "2b", "4b", "9b", "27b", "35b-a3b"})


class ModelConfigurationError(ValueError):
    """A requested model or thinking choice is unavailable under current settings."""


@dataclass(frozen=True, slots=True)
class QwenModelOption:
    id: str
    label: str
    supports_thinking: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        family, separator, tag = self.id.partition(":")
        if not separator or family != "qwen3.5" or tag not in _OFFICIAL_MODEL_TAGS:
            raise ValueError("Lumi only supports official vanilla Qwen3.5 model tags")
        if not self.label.strip() or len(self.label) > 80:
            raise ValueError("Model labels must contain between 1 and 80 characters")


class ModelCatalog:
    """A read-only snapshot of the model options currently enabled by an administrator."""

    def __init__(
        self,
        models: Iterable[QwenModelOption],
        default_model: str,
        default_thinking: bool = False,
    ) -> None:
        entries = tuple(models)
        if not entries:
            raise ValueError("At least one Qwen3.5 model must be configured")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("Configured Lumi model IDs must be unique")
        selectable = {entry.id: entry for entry in entries if entry.enabled}
        if default_model not in selectable:
            raise ValueError("The Lumi default model must be enabled")
        self._models = entries
        self._selectable = selectable
        self._default_model = default_model
        default = selectable[default_model]
        self._default_thinking = bool(default_thinking and default.supports_thinking)

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_thinking(self) -> bool:
        return self._default_thinking

    @property
    def selectable_models(self) -> tuple[QwenModelOption, ...]:
        return tuple(model for model in self._models if model.enabled)

    def require(self, model_id: str) -> QwenModelOption:
        model = self._selectable.get(model_id)
        if model is None:
            raise ModelConfigurationError("The selected Lumi model is unavailable")
        return model

    def validate_choice(self, model_id: str, thinking: bool) -> QwenModelOption:
        model = self.require(model_id)
        if thinking and not model.supports_thinking:
            raise ModelConfigurationError("Thinking is unavailable for the selected model")
        return model
