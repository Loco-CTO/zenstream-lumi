from __future__ import annotations

import base64
import json
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lumi.delegation import DelegationError, Ed25519DelegationVerifier


def encode_segment(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def make_token(private_key: Ed25519PrivateKey, claims: dict[str, object]) -> str:
    header = encode_segment(b'{"alg":"EdDSA","typ":"JWT"}')
    payload = encode_segment(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signed = f"{header}.{payload}".encode("ascii")
    signature = encode_segment(private_key.sign(signed))
    return f"{header}.{payload}.{signature}"


class DelegationVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1_800_000_000
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.verifier = Ed25519DelegationVerifier(
            public_key,
            clock=lambda: self.now,
        )

    def claims(self, **overrides: object) -> dict[str, object]:
        return {
            "iss": "zenstream-orchestrator",
            "aud": "zenstream-lumi",
            "sub": "account-7",
            "sid": "session-9",
            "cid": "conversation-11",
            "scope": ["lumi.chat.write", "catalog.read"],
            "iat": self.now,
            "exp": self.now + 120,
            "jti": "grant-13",
            **overrides,
        }

    def test_verifies_signed_scoped_conversation_claims(self) -> None:
        claims = self.verifier.verify(make_token(self.private_key, self.claims()))

        self.assertEqual(claims.subject, "account-7")
        self.assertEqual(claims.session_id, "session-9")
        self.assertEqual(claims.conversation_id, "conversation-11")
        self.assertEqual(claims.scopes, frozenset({"lumi.chat.write", "catalog.read"}))

    def test_rejects_tampered_signature(self) -> None:
        token = make_token(self.private_key, self.claims())
        header, payload, encoded_signature = token.split(".")
        padded_signature = encoded_signature + "=" * (-len(encoded_signature) % 4)
        signature = bytearray(base64.urlsafe_b64decode(padded_signature))
        signature[0] ^= 1
        changed = f"{header}.{payload}.{encode_segment(bytes(signature))}"

        with self.assertRaises(DelegationError):
            self.verifier.verify(changed)

    def test_rejects_expired_or_overlong_delegations(self) -> None:
        expired = self.claims(exp=self.now - 1)
        overlong = self.claims(exp=self.now + 781)

        with self.assertRaises(DelegationError):
            self.verifier.verify(make_token(self.private_key, expired))
        with self.assertRaises(DelegationError):
            self.verifier.verify(make_token(self.private_key, overlong))

    def test_accepts_the_maximum_delegation_lifetime(self) -> None:
        token = make_token(self.private_key, self.claims(exp=self.now + 780))

        claims = self.verifier.verify(token)

        self.assertEqual(claims.expires_at - claims.issued_at, 780)

    def test_rejects_wrong_audience_issuer_and_future_issue_time(self) -> None:
        invalid_claims = (
            self.claims(aud="other-service"),
            self.claims(iss="other-issuer"),
            self.claims(iat=self.now + 20, exp=self.now + 100),
        )

        for claims in invalid_claims:
            with self.subTest(claims=claims):
                with self.assertRaises(DelegationError):
                    self.verifier.verify(make_token(self.private_key, claims))

    def test_rejects_missing_identity_scope_and_duplicate_scope(self) -> None:
        invalid_claims = (
            self.claims(sub=""),
            self.claims(scope=[]),
            self.claims(scope=["catalog.read", "catalog.read"]),
            self.claims(cid=123),
        )

        for claims in invalid_claims:
            with self.subTest(claims=claims):
                with self.assertRaises(DelegationError):
                    self.verifier.verify(make_token(self.private_key, claims))

    def test_rejects_duplicate_json_claims(self) -> None:
        header = encode_segment(b'{"alg":"EdDSA","typ":"JWT"}')
        payload = encode_segment(
            (
                '{"iss":"zenstream-orchestrator","iss":"attacker",'
                '"aud":"zenstream-lumi","sub":"account-7","sid":"session-9",'
                '"cid":"conversation-11","scope":["catalog.read"],"iat":1800000000,'
                '"exp":1800000120,"jti":"grant-13"}'
            ).encode("utf-8")
        )
        signed = f"{header}.{payload}".encode("ascii")
        token = f"{header}.{payload}.{encode_segment(self.private_key.sign(signed))}"

        with self.assertRaises(DelegationError):
            self.verifier.verify(token)


if __name__ == "__main__":
    unittest.main()
