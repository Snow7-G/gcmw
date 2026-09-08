"""Read-only tool whitelist declarations (issue #57).

The ToolGateway only ever admits tools whose names appear in the sealed
read-only whitelist below — the gateway is the single access path and it never
exposes file/DB/shell/network/agent-to-agent operations (V2.3 §6.2). Schemas
are authored against the subset implemented in ``app.tools.validation`` and
never echo values in error text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Sealed read-only whitelist (ordered — first registration wins ordering).
READONLY_TOOL_NAMES: tuple[str, ...] = (
    "knowledge.search",
    "knowledge.get_fragment",
    "department.search",
    "staff.search",
    "video.search",
    "memory.read_short",
)

# name -> (description, input_schema)
TOOL_INPUT_SCHEMAS: dict[str, tuple[str, dict[str, Any]]] = {
    "knowledge.search": (
        "Search approved in-window knowledge items (tenant-scoped when tenant_id is given); returns titles + snippets.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "knowledge.get_fragment": (
        "Read one content fragment of an approved knowledge item for evidence grounding (read-only).",
        {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "fragment_index": {"type": "integer", "minimum": 0},
                "fragment_chars": {
                    "type": "integer",
                    "minimum": 128,
                    "maximum": 16384,
                },
            },
            "required": ["source_id"],
            "additionalProperties": False,
        },
    ),
    "department.search": (
        "Read-only department directory search (directory placeholder until the real index lands).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "staff.search": (
        "Read-only staff directory search (directory placeholder until the real index lands).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "video.search": (
        "Read-only video catalog search (directory placeholder until the real index lands).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 64},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "memory.read_short": (
        "Read the caller's short-term memory summary (writes are orchestrated outside the gateway).",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "minLength": 1, "maxLength": 128}
            },
            "additionalProperties": False,
        },
    ),
}

# All whitelisted outputs are JSON objects in v1.
OUTPUT_SCHEMA: dict[str, Any] = {"type": "object"}


class WhitelistError(ValueError):
    """Registration-time violation of the read-only whitelist (programming or
    configuration error — never routed to clients as an audit event)."""


@dataclass(frozen=True)
class ToolSpec:
    """Declarative spec of one whitelisted read-only tool.

    ``executor`` receives the validated arguments and returns a JSON-ready
    value; it must not mutate shared state. Domain-level failures are raised
    as :class:`~app.tools.gateway.ToolGatewayError` with a registry ErrorCode.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    max_result_bytes: int | None = None  # None -> gateway default
    executor: Callable[[dict[str, Any]], Any] | None = None

    def __post_init__(self) -> None:
        if self.name not in READONLY_TOOL_NAMES:
            raise WhitelistError(
                f"{self.name!r} is not in the read-only whitelist: "
                f"{READONLY_TOOL_NAMES}"
            )


def canonical_spec(
    name: str,
    executor: Callable[[dict[str, Any]], Any] | None = None,
    *,
    output_schema: dict[str, Any] | None = None,
) -> ToolSpec:
    """Build the canonical whitelist spec for ``name`` (see #57 issue body)."""
    if name not in TOOL_INPUT_SCHEMAS:
        raise WhitelistError(f"no canonical declaration for {name!r}")
    description, input_schema = TOOL_INPUT_SCHEMAS[name]
    return ToolSpec(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema if output_schema is not None else OUTPUT_SCHEMA,
        executor=executor,
    )
