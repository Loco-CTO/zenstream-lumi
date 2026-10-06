"""Private, service-authenticated API between the Orchestrator and Lumi."""

from __future__ import annotations

import hmac
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Path, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lumi.agent import InferenceError
from lumi.contracts import EntityReference, Source
from lumi.delegation import DelegationError
from lumi.model_catalog import ModelConfigurationError
from lumi.service import ChatTurn, LumiConversationService, LumiServiceBusy
from lumi.storage import (
    ConversationNotFound,
    ConversationSnapshot,
    StoredConversation,
    StoredMessage,
    UserModelPreference,
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=6_000)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    thinking: bool | None = Field(default=None, strict=True)

    @field_validator("message")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A conversation message cannot be empty")
        return value

    @model_validator(mode="after")
    def require_complete_model_choice(self) -> ChatRequest:
        if (self.model is None) != (self.thinking is None):
            raise ValueError("Model and thinking must be supplied together")
        return self


class ModelChoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=200)
    thinking: bool = Field(strict=True)


def create_app(
    service: LumiConversationService,
    internal_service_token: str,
) -> FastAPI:
    """Create a private API; bind it only to the configured local service network."""

    if len(internal_service_token) < 32 or internal_service_token.strip() != internal_service_token:
        raise ValueError("Lumi's internal service token must contain at least 32 characters")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="Lumi internal service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def authenticate_internal_requests(request: Request, call_next):
        if request.url.path.startswith("/internal/"):
            authorization = request.headers.get("authorization", "")
            scheme, separator, supplied_token = authorization.partition(" ")
            if (
                not separator
                or scheme.lower() != "bearer"
                or not supplied_token.isascii()
                or not hmac.compare_digest(supplied_token, internal_service_token)
            ):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"Cache-Control": "no-store"},
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DelegationError)
    async def delegation_error_handler(_: Request, __: DelegationError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    @app.exception_handler(ConversationNotFound)
    async def conversation_not_found_handler(
        _: Request, __: ConversationNotFound
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})

    @app.exception_handler(ModelConfigurationError)
    async def model_configuration_handler(
        _: Request, error: ModelConfigurationError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(LumiServiceBusy)
    async def busy_handler(_: Request, __: LumiServiceBusy) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": "Lumi is busy"})

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_: Request, __: InferenceError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Lumi is temporarily unavailable"})

    @app.exception_handler(sqlite3.Error)
    async def storage_error_handler(_: Request, __: sqlite3.Error) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Lumi storage is unavailable"})

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/internal/models")
    async def available_models(
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        models = service.available_models(x_lumi_delegation)
        return {
            "models": [
                {
                    "id": model.id,
                    "label": model.label,
                    "supportsThinking": model.supports_thinking,
                }
                for model in models
            ],
            "defaultModel": service.default_model,
            "defaultThinking": service.default_thinking,
        }

    @app.get("/internal/conversations")
    async def list_conversations(
        limit: int = Query(default=50, ge=1, le=100),
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        conversations = await service.list_conversations(x_lumi_delegation, limit)
        return {"conversations": [_conversation_json(item) for item in conversations]}

    @app.post("/internal/conversations/{conversation_id}/turns")
    async def chat(
        body: ChatRequest,
        conversation_id: str = Path(min_length=1, max_length=100),
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        turn = await service.chat(
            x_lumi_delegation,
            conversation_id,
            body.message,
            model=body.model,
            thinking=body.thinking,
        )
        return _turn_json(turn)

    @app.get("/internal/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str = Path(min_length=1, max_length=100),
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        snapshot = await service.get_conversation(x_lumi_delegation, conversation_id)
        return _snapshot_json(snapshot)

    @app.patch("/internal/conversations/{conversation_id}/choice")
    async def update_choice(
        body: ModelChoiceRequest,
        conversation_id: str = Path(min_length=1, max_length=100),
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        conversation = await service.update_choice(
            x_lumi_delegation,
            conversation_id,
            body.model,
            body.thinking,
        )
        return {"conversation": _conversation_json(conversation)}

    @app.put("/internal/preferences/model")
    async def update_preference(
        body: ModelChoiceRequest,
        x_lumi_delegation: str = Header(alias="X-Lumi-Delegation", max_length=8_192),
    ) -> dict[str, object]:
        preference = await service.update_user_preference(
            x_lumi_delegation,
            body.model,
            body.thinking,
        )
        return {"preference": _preference_json(preference)}

    return app


def _turn_json(turn: ChatTurn) -> dict[str, object]:
    return {
        "conversation": _conversation_json(turn.conversation),
        "answer": {
            "markdown": turn.answer.markdown,
            "references": [_reference_json(item) for item in turn.answer.references],
            "sources": [_source_json(item) for item in turn.answer.sources],
        },
    }


def _snapshot_json(snapshot: ConversationSnapshot) -> dict[str, object]:
    return {
        "conversation": _conversation_json(snapshot.conversation),
        "messages": [_message_json(item) for item in snapshot.messages],
    }


def _conversation_json(item: StoredConversation) -> dict[str, object]:
    return {
        "id": item.id,
        "title": item.title,
        "model": item.model,
        "thinking": item.thinking,
        "createdAt": item.created_at,
        "updatedAt": item.updated_at,
    }


def _message_json(item: StoredMessage) -> dict[str, object]:
    return {
        "id": item.id,
        "role": item.role,
        "content": item.content,
        "createdAt": item.created_at,
        "references": [_reference_json(reference) for reference in item.references],
        "sources": [_source_json(source) for source in item.sources],
    }


def _reference_json(item: EntityReference) -> dict[str, str]:
    return {"type": item.type, "id": item.id, "title": item.title}


def _source_json(item: Source) -> dict[str, str | None]:
    return {
        "url": item.url,
        "websiteName": item.website_name,
        "title": item.title,
        "faviconUrl": item.favicon_url,
    }


def _preference_json(item: UserModelPreference) -> dict[str, object]:
    return {"model": item.model, "thinking": item.thinking}
