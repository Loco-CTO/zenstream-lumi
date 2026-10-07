"""Verification for short-lived Orchestrator-signed Lumi delegations."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Protocol
from collections.abc import Callable


class DelegationError(PermissionError):
    """A signed delegation is invalid, expired, or missing required claims."""


@dataclass(frozen=True, slots=True)
class DelegationClaims:
    subject: str
    session_id: str
    conversation_id: str | None
    audience: str
    issuer: str
    scopes: frozenset[str]
    issued_at: int
    expires_at: int
    token_id: str


class DelegationVerifier(Protocol):
    def verify(self, token: str) -> DelegationClaims:
        """Verify a token and return claims that have passed signature and time checks."""


class Ed25519DelegationVerifier:
    """Verify compact EdDSA JWTs using only the Orchestrator's public key."""

    def __init__(
        self,
        public_key_pem: str | bytes,
        *,
        audience: str = "zenstream-lumi",
        issuer: str = "zenstream-orchestrator",
        maximum_lifetime_seconds: int = 780,
        clock_skew_seconds: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        if maximum_lifetime_seconds <= 0 or clock_skew_seconds < 0:
            raise ValueError("Delegation time limits are invalid")
        encoded_key = (
            public_key_pem.encode("ascii")
            if isinstance(public_key_pem, str)
            else public_key_pem
        )
        try:
            public_key = load_pem_public_key(encoded_key)
        except (TypeError, ValueError) as error:
            raise ValueError("The Lumi delegation public key is invalid") from error
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("The Lumi delegation public key must use Ed25519")
        if not audience or not issuer:
            raise ValueError("Delegation audience and issuer are required")

        self._public_key = public_key
        self._audience = audience
        self._issuer = issuer
        self._maximum_lifetime = maximum_lifetime_seconds
        self._clock_skew = clock_skew_seconds
        self._clock = clock

    def verify(self, token: str) -> DelegationClaims:
        from cryptography.exceptions import InvalidSignature

        if not isinstance(token, str) or not token or len(token) > 8_192:
            raise DelegationError("Invalid Lumi delegation.")
        parts = token.split(".")
        if len(parts) != 3:
            raise DelegationError("Invalid Lumi delegation.")
        header_segment, payload_segment, signature_segment = parts
        try:
            header = _decode_json_segment(header_segment)
            payload = _decode_json_segment(payload_segment)
            signature = _decode_segment(signature_segment)
            if set(header) != {"alg", "typ"} or header.get("alg") != "EdDSA":
                raise DelegationError("Invalid Lumi delegation.")
            if header.get("typ") != "JWT":
                raise DelegationError("Invalid Lumi delegation.")
            signed = f"{header_segment}.{payload_segment}".encode("ascii")
            self._public_key.verify(signature, signed)
        except (InvalidSignature, UnicodeError, ValueError, TypeError) as error:
            raise DelegationError("Invalid Lumi delegation.") from error

        claims = self._claims(payload)
        self._validate_time(claims, payload)
        return claims

    def _claims(self, payload: dict[str, object]) -> DelegationClaims:
        audience_claim = payload.get("aud")
        audiences = (
            {audience_claim}
            if isinstance(audience_claim, str)
            else set(audience_claim)
            if isinstance(audience_claim, list)
            and all(isinstance(item, str) for item in audience_claim)
            else set()
        )
        subject = _bounded_claim(payload, "sub", 200)
        session_id = _bounded_claim(payload, "sid", 200)
        token_id = _bounded_claim(payload, "jti", 200)
        issuer = payload.get("iss")
        conversation_id = payload.get("cid")
        raw_scopes = payload.get("scope")
        issued_at = payload.get("iat")
        expires_at = payload.get("exp")

        if self._audience not in audiences or issuer != self._issuer:
            raise DelegationError("Invalid Lumi delegation.")
        if conversation_id is not None and (
            not isinstance(conversation_id, str)
            or not conversation_id
            or len(conversation_id) > 100
        ):
            raise DelegationError("Invalid Lumi delegation.")
        if (
            not isinstance(raw_scopes, list)
            or not raw_scopes
            or len(raw_scopes) > 32
            or not all(isinstance(scope, str) and 0 < len(scope) <= 100 for scope in raw_scopes)
            or len(set(raw_scopes)) != len(raw_scopes)
        ):
            raise DelegationError("Invalid Lumi delegation.")
        if not _is_timestamp(issued_at) or not _is_timestamp(expires_at):
            raise DelegationError("Invalid Lumi delegation.")
        return DelegationClaims(
            subject=subject,
            session_id=session_id,
            conversation_id=conversation_id,
            audience=self._audience,
            issuer=self._issuer,
            scopes=frozenset(raw_scopes),
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=token_id,
        )

    def _validate_time(self, claims: DelegationClaims, payload: dict[str, object]) -> None:
        now = self._clock()
        not_before = payload.get("nbf", claims.issued_at)
        if not _is_timestamp(not_before):
            raise DelegationError("Invalid Lumi delegation.")
        if claims.expires_at <= now - self._clock_skew:
            raise DelegationError("Expired Lumi delegation.")
        if claims.issued_at > now + self._clock_skew or not_before > now + self._clock_skew:
            raise DelegationError("Invalid Lumi delegation.")
        if claims.expires_at <= claims.issued_at:
            raise DelegationError("Invalid Lumi delegation.")
        if claims.expires_at - claims.issued_at > self._maximum_lifetime:
            raise DelegationError("Invalid Lumi delegation.")
        if now - claims.issued_at > self._maximum_lifetime + self._clock_skew:
            raise DelegationError("Expired Lumi delegation.")


def _decode_segment(segment: str) -> bytes:
    if not segment or any(char.isspace() for char in segment):
        raise ValueError("Invalid base64url segment")
    padded = segment + "=" * (-len(segment) % 4)
    return base64.b64decode(padded, altchars=b"-_", validate=True)


def _decode_json_segment(segment: str) -> dict[str, object]:
    value = json.loads(
        _decode_segment(segment),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("JWT parts must be JSON objects")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JWT field")
        result[key] = value
    return result


def _bounded_claim(payload: dict[str, object], name: str, maximum: int) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DelegationError("Invalid Lumi delegation.")
    return value


def _is_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
