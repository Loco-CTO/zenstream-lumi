"""Validate the versioned, non-executable Lumi evaluation capability registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(__file__).with_name("capabilities.json")
_REGISTRY_ID = re.compile(r"^[a-z][a-z0-9-]*$")
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EFFECTS = {"read_only", "state_changing"}
_INTEGRATION_STATUSES = {"implemented", "partial", "proposed"}
_REVIEW_STATUSES = {"draft", "approved"}
_EVIDENCE = {"openapi", "route_implementation", "client_implementation"}
_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class CapabilityRegistryError(ValueError):
    """Raised when a capability registry is malformed or internally unsafe."""


def _require_keys(value: dict[str, Any], expected: set[str], prefix: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing:
        raise CapabilityRegistryError(
            f"{prefix}: missing field(s): {', '.join(missing)}"
        )
    if extra:
        raise CapabilityRegistryError(
            f"{prefix}: unsupported field(s): {', '.join(extra)}"
        )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_argument_schema(value: Any, prefix: str) -> None:
    if not isinstance(value, dict) or value.get("type") != "object":
        raise CapabilityRegistryError(
            f"{prefix}: argument_schema must describe an object"
        )
    properties = value.get("properties")
    required = value.get("required", [])
    if not isinstance(properties, dict):
        raise CapabilityRegistryError(
            f"{prefix}: argument_schema properties must be an object"
        )
    if not isinstance(required, list) or any(
        not isinstance(item, str) for item in required
    ):
        raise CapabilityRegistryError(
            f"{prefix}: argument_schema required must be a string array"
        )
    unknown_required = sorted(set(required) - properties.keys())
    if unknown_required:
        raise CapabilityRegistryError(
            f"{prefix}: required argument(s) lack schemas: {', '.join(unknown_required)}"
        )
    if value.get("additionalProperties") is not False:
        raise CapabilityRegistryError(
            f"{prefix}: argument_schema must reject additional properties"
        )
    if any(not isinstance(schema, dict) for schema in properties.values()):
        raise CapabilityRegistryError(
            f"{prefix}: every argument property must have a schema object"
        )


def validate_registry(registry: Any) -> dict[str, Any]:
    """Validate a registry document and return a concise non-sensitive summary."""
    if not isinstance(registry, dict):
        raise CapabilityRegistryError("registry must be a JSON object")
    _require_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "review_status",
            "created_date",
            "source_snapshot",
            "capabilities",
        },
        "registry",
    )
    version = registry["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise CapabilityRegistryError("registry schema_version must be 1")
    if not _nonempty_string(registry["registry_id"]) or not _REGISTRY_ID.fullmatch(
        registry["registry_id"]
    ):
        raise CapabilityRegistryError("registry_id must be a lowercase slug")
    if (
        not isinstance(registry["review_status"], str)
        or registry["review_status"] not in _REVIEW_STATUSES
    ):
        raise CapabilityRegistryError("registry review_status is invalid")
    if not _nonempty_string(registry["created_date"]):
        raise CapabilityRegistryError("created_date must be a nonempty string")
    try:
        date.fromisoformat(registry["created_date"])
    except ValueError as exc:
        raise CapabilityRegistryError("created_date must be an ISO date") from exc

    snapshot = registry["source_snapshot"]
    if not isinstance(snapshot, dict):
        raise CapabilityRegistryError("source_snapshot must be an object")
    _require_keys(
        snapshot,
        {
            "repository",
            "commit",
            "contract_path",
            "contract_sha256",
            "implementation_path",
            "implementation_sha256",
        },
        "source_snapshot",
    )
    for field in ("repository", "contract_path", "implementation_path"):
        if not _nonempty_string(snapshot[field]):
            raise CapabilityRegistryError(
                f"source_snapshot {field} must be a nonempty string"
            )
    if not isinstance(snapshot["commit"], str) or not _COMMIT.fullmatch(
        snapshot["commit"]
    ):
        raise CapabilityRegistryError(
            "source_snapshot commit must be a lowercase 40-character commit"
        )
    for field in ("contract_sha256", "implementation_sha256"):
        if not isinstance(snapshot[field], str) or not _SHA256.fullmatch(
            snapshot[field]
        ):
            raise CapabilityRegistryError(
                f"source_snapshot {field} must be a lowercase SHA-256 digest"
            )

    capabilities = registry["capabilities"]
    if not isinstance(capabilities, list) or not capabilities:
        raise CapabilityRegistryError("capabilities must be a nonempty array")
    ids: set[str] = set()
    for index, capability in enumerate(capabilities, start=1):
        prefix = f"capability {index}"
        if not isinstance(capability, dict):
            raise CapabilityRegistryError(f"{prefix} must be an object")
        _require_keys(
            capability,
            {
                "id",
                "title",
                "effect",
                "integration_status",
                "review_status",
                "description",
                "argument_schema",
                "operations",
                "limitations",
            },
            prefix,
        )
        capability_id = capability["id"]
        if (
            not isinstance(capability_id, str)
            or not _CAPABILITY_ID.fullmatch(capability_id)
        ):
            raise CapabilityRegistryError(f"{prefix} id must be a lowercase dotted id")
        if capability_id in ids:
            raise CapabilityRegistryError(
                f"{prefix}: duplicate capability id {capability_id}"
            )
        ids.add(capability_id)
        for field in ("title", "description"):
            if not _nonempty_string(capability[field]):
                raise CapabilityRegistryError(
                    f"{prefix} {field} must be a nonempty string"
                )
        if (
            not isinstance(capability["effect"], str)
            or capability["effect"] not in _EFFECTS
        ):
            raise CapabilityRegistryError(f"{prefix} effect is invalid")
        if (
            not isinstance(capability["integration_status"], str)
            or capability["integration_status"] not in _INTEGRATION_STATUSES
        ):
            raise CapabilityRegistryError(f"{prefix} integration_status is invalid")
        if (
            not isinstance(capability["review_status"], str)
            or capability["review_status"] not in _REVIEW_STATUSES
        ):
            raise CapabilityRegistryError(f"{prefix} review_status is invalid")
        if (
            registry["review_status"] == "approved"
            and capability["review_status"] != "approved"
        ):
            raise CapabilityRegistryError(
                f"{prefix}: an approved registry cannot contain an unapproved capability"
            )
        _validate_argument_schema(capability["argument_schema"], prefix)

        operations = capability["operations"]
        if not isinstance(operations, list) or not operations:
            raise CapabilityRegistryError(f"{prefix} operations must be nonempty")
        for operation_index, operation in enumerate(operations, start=1):
            operation_prefix = f"{prefix} operation {operation_index}"
            if not isinstance(operation, dict):
                raise CapabilityRegistryError(f"{operation_prefix} must be an object")
            _require_keys(
                operation,
                {"method", "path", "evidence"},
                operation_prefix,
            )
            if (
                not isinstance(operation["method"], str)
                or operation["method"] not in _METHODS
            ):
                raise CapabilityRegistryError(f"{operation_prefix} method is invalid")
            if (
                not isinstance(operation["path"], str)
                or not operation["path"].startswith("/api/")
            ):
                raise CapabilityRegistryError(
                    f"{operation_prefix} path must start with /api/"
                )
            if (
                not isinstance(operation["evidence"], str)
                or operation["evidence"] not in _EVIDENCE
            ):
                raise CapabilityRegistryError(f"{operation_prefix} evidence is invalid")

        limitations = capability["limitations"]
        if not isinstance(limitations, list) or not limitations or any(
            not _nonempty_string(item) for item in limitations
        ):
            raise CapabilityRegistryError(
                f"{prefix} limitations must be a nonempty string array"
            )

    return {
        "status": "valid",
        "registry_id": registry["registry_id"],
        "review_status": registry["review_status"],
        "capability_count": len(capabilities),
        "capability_ids": sorted(ids),
    }


def validate_file(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityRegistryError(f"cannot load registry: {exc}") from exc
    return validate_registry(registry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="capability registry JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the validation summary here; stdout if omitted",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_file(args.registry)
    except CapabilityRegistryError as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
