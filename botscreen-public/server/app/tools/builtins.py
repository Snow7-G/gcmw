"""Reference read-only executors for the whitelisted tools (issue #57).

All executors are pure reads over injectable sources:

- ``knowledge.search`` / ``knowledge.get_fragment`` read the production view
  of any object exposing ``production_items(tenant_id) -> list`` whose items
  carry ``source_id/title/content/...`` attributes — the #56 KnowledgeStore
  satisfies this structurally and its production view is the only view #53
  RAG may query;
- ``department/staff/video.search`` run a deterministic substring search over
  an optional in-memory directory index (placeholder until real indexes land
  in a later issue). A record is visible ONLY to its own tenant — a record
  without an explicit ``tenant_id`` matches nobody — and results are rebuilt
  from a FIELD WHITELIST, so private columns (``internal_note`` …) and nested
  structures can never ride out to an agent;
- ``memory.read_short`` reads a short-term summary through an optional
  read-only callable, which receives the trusted ``(tenant, device, session)``
  triple: two tenants (or two devices) sharing a session id can never read each
  other''s memory;
- a tool whose dependency was never injected fails closed with
  ``E_UNAVAILABLE_MAINTENANCE`` — an unconfigured source is a deployment fault,
  never an empty result.

Executors never write, never touch files/network and raise
:class:`~app.tools.gateway.ToolGatewayError` (registry ErrorCode) for
domain-level outcomes such as a missing knowledge source.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app.contracts.errors import ErrorCode
from app.tools.gateway import ToolGateway, ToolGatewayError
from app.tools.specs import canonical_spec

_TOKEN_SPLIT = re.compile(r"[\s,，。；;：:、|/\\()\[\]{}<>«»\"'“”‘’!?！？.．\-—_]+")
_SNIPPET_CHARS = 240

Directories = Mapping[str, list[Mapping[str, Any]]]

#: Fields an agent may ever see per directory domain (everything else — private
#: columns, nested structures — is dropped before the result is built).
DIRECTORY_FIELDS: dict[str, tuple[str, ...]] = {
    "department": ("name", "location"),
    "staff": ("name", "title", "department"),
    "video": ("title", "duration_s"),
}

#: Explicit, reviewed marker for records that are public on purpose. A record
#: WITHOUT a tenant id is NOT public: it matches nobody.
PUBLIC_VISIBILITY = "public"

#: per-field cap for the projection (defence against oversized fields)
_DIRECTORY_FIELD_CHARS = 512


@dataclass(frozen=True)
class MemoryKey:
    """The trusted key space of short-term memory: tenant+device+session."""

    tenant_id: str
    device_id: str
    session_id: str


#: A memory reader receives the TRUSTED key (never a bare session id).
MemoryReader = Callable[[MemoryKey], str | None]


def memory_key(context: Any) -> MemoryKey | None:
    """Build the trusted memory key, or ``None`` when the context lacks one."""
    fields = {
        name: getattr(context, name, "")
        for name in ("tenant_id", "device_id", "session_id")
    }
    if not all(isinstance(value, str) and value for value in fields.values()):
        return None
    return MemoryKey(**fields)


def _directory_visible(record: Mapping[str, Any], tenant_id: str) -> bool:
    """A record is visible only to its OWN tenant, or when explicitly public.

    * ``tenant_id`` mismatch -> invisible;
    * a record with NO ``tenant_id`` -> invisible (it is never adopted by
      whichever tenant happens to search);
    * ``visibility == "public"`` -> visible (an explicit, reviewed marker).
    """
    owner = record.get("tenant_id")
    if isinstance(owner, str) and owner and owner == tenant_id:
        return True
    return record.get("visibility") == PUBLIC_VISIBILITY


def _project_directory_record(
    record: Mapping[str, Any], allowed: tuple[str, ...]
) -> dict[str, Any] | None:
    """Rebuild a record from the FIELD WHITELIST (no raw record ever escapes).

    Only scalar values survive: nested dict/list values are dropped, so a
    tampered record cannot smuggle a structure (or an identity) out to a model.
    """
    if not isinstance(record, Mapping):
        return None
    projected: dict[str, Any] = {}
    for field in allowed:
        value = _value(record.get(field))
        if isinstance(value, str):
            projected[field] = value[:_DIRECTORY_FIELD_CHARS]
        elif isinstance(value, int) and not isinstance(value, bool):
            projected[field] = value
    return projected or None


def _value(member: Any) -> Any:
    """Enum members expose .value; plain strings pass through unchanged."""
    return getattr(member, "value", member)


def _attr(item: Any, name: str, default: Any = "") -> Any:
    return getattr(item, name, default)


def _require_dependency(tool_name: str) -> None:
    """Fail CLOSED when a tool's server-side dependency is not configured.

    Returning an empty result would silently look like "no data"; an
    unconfigured dependency is a deployment fault and must be visible.
    """
    raise ToolGatewayError(
        ErrorCode.UNAVAILABLE_MAINTENANCE,
        f"tool {tool_name!r} has no dependency configured on this server",
    )


def _arg(args: dict[str, Any], name: str, default: Any) -> Any:
    return args.get(name, default)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_SPLIT.split(text) if t]


def _item_summary(item: Any) -> dict[str, Any]:
    content = _attr(item, "content")
    return {
        "source_id": _attr(item, "source_id"),
        "source_type": _value(_attr(item, "source_type")),
        "title": _attr(item, "title"),
        "medical_domain": _attr(item, "medical_domain"),
        "knowledge_version": _attr(item, "knowledge_version"),
        "content_hash": _attr(item, "content_hash"),
        "source_uri": _attr(item, "source_uri"),
        "snippet": content[:_SNIPPET_CHARS],
    }


def make_knowledge_search(store: Any | None) -> Callable[[Any, dict], Any]:
    """Deterministic relevance search over a production-view source."""

    def search(context: Any, args: dict[str, Any]) -> dict[str, Any]:
        query = _arg(args, "query", "")
        top_k = _arg(args, "top_k", 5)
        if store is None:
            _require_dependency("knowledge.search")
        tokens = set(_tokens(query))
        scored: list[tuple[int, int, Any]] = []
        # identity ALWAYS comes from the injected context, never from arguments
        for index, item in enumerate(store.production_items(context)):
            haystack = f"{_attr(item, 'title')} {_attr(item, 'content')}".lower()
            score = sum(1 for t in tokens if t in haystack)
            if score:
                scored.append((-score, index, item))
        scored.sort()
        ranked = scored[:top_k]
        return {
            "items": [_item_summary(item) for _, _, item in ranked],
            "total": len(ranked),
        }

    return search


def make_knowledge_get_fragment(store: Any | None) -> Callable[[Any, dict], Any]:
    def get_fragment(context: Any, args: dict[str, Any]) -> dict[str, Any]:
        source_id = args["source_id"]
        fragment_index = _arg(args, "fragment_index", 0)
        fragment_chars = _arg(args, "fragment_chars", 2000)
        if store is None:
            _require_dependency("knowledge.get_fragment")
        item = next(
            (
                it
                for it in store.production_items(context)
                if getattr(it, "source_id", None) == source_id
            ),
            None,
        )
        if item is None:
            raise ToolGatewayError(
                ErrorCode.NOT_FOUND_KNOWLEDGE,
                f"source {source_id!r} not available",
            )
        content = _attr(item, "content")
        fragments = [
            content[index : index + fragment_chars]
            for index in range(0, len(content), fragment_chars)
        ]
        if not fragments:
            fragments = [""]
        if fragment_index >= len(fragments):
            raise ToolGatewayError(
                ErrorCode.NOT_FOUND_KNOWLEDGE,
                f"fragment {fragment_index} out of range for {source_id!r}",
            )
        return {
            "source_id": _attr(item, "source_id"),
            "knowledge_version": _attr(item, "knowledge_version"),
            "content_hash": _attr(item, "content_hash"),
            "fragment_index": fragment_index,
            "total_fragments": len(fragments),
            "text": fragments[fragment_index],
        }

    return get_fragment


def make_directory_search(
    domain: str, directories: Directories | None
) -> Callable[[Any, dict], Any]:
    def search(context: Any, args: dict[str, Any]) -> dict[str, Any]:
        query = _arg(args, "query", "")
        top_k = _arg(args, "top_k", 5)
        if directories is None:
            _require_dependency(f"{domain}.search")
        # scoping follows the INJECTED context; the caller's tenant can never
        # be widened or redirected through arguments
        tenant_id = getattr(context, "tenant_id", "")
        allowed = DIRECTORY_FIELDS[domain]
        records = [
            visible
            for visible in (
                _project_directory_record(record, allowed)
                for record in (directories or {}).get(domain, [])
                if _directory_visible(record, tenant_id)
            )
            if visible is not None
        ]
        tokens = _tokens(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            haystack = " ".join(str(v) for v in record.values()).lower()
            hits = sum(1 for t in tokens if t in haystack)
            if hits:
                scored.append((-hits, index, record))
        scored.sort()
        items = [record for _, _, record in scored[:top_k]]
        return {"items": items, "total": len(items)}

    return search


def make_memory_read_short(reader: MemoryReader | None) -> Callable[[Any, dict], Any]:
    def read_short(context: Any, args: dict[str, Any]) -> dict[str, Any]:
        max_chars = _arg(args, "max_chars", 2000)
        if reader is None:
            _require_dependency("memory.read_short")
        key = memory_key(context)
        if key is None:
            # without a device+session identity the summary cannot be
            # attributed, so nothing is read (it is never guessed from args)
            raise ToolGatewayError(
                ErrorCode.AUTHZ_FORBIDDEN,
                "memory.read_short needs a device+session identity from the server",
            )
        summary = reader(key)
        if isinstance(summary, str) and max_chars:
            summary = summary[: int(max_chars)]
        if not summary:
            return {"summary": "", "available": False}
        return {"summary": summary, "available": True}

    return read_short


def build_gateway(
    *,
    knowledge_store: Any | None = None,
    directories: Directories | None = None,
    memory_reader: MemoryReader | None = None,
    audit_sink: Callable[[Any], None] | None = None,
    clock: Callable[[], Any] | None = None,
    default_timeout_ms: int = 5_000,
    default_max_result_bytes: int = 64 * 1024,
) -> ToolGateway:
    """Assemble a gateway with the six canonical read-only tools bound to the
    given (read-only) sources.

    A missing source is NOT silently empty: calls to that tool fail closed with
    ``E_UNAVAILABLE_MAINTENANCE`` (see ``_require_dependency``)."""

    gateway = ToolGateway(
        audit_sink=audit_sink,
        clock=clock,
        default_timeout_ms=default_timeout_ms,
        default_max_result_bytes=default_max_result_bytes,
    )
    bindings = {
        "knowledge.search": make_knowledge_search(knowledge_store),
        "knowledge.get_fragment": make_knowledge_get_fragment(knowledge_store),
        "department.search": make_directory_search("department", directories),
        "staff.search": make_directory_search("staff", directories),
        "video.search": make_directory_search("video", directories),
        "memory.read_short": make_memory_read_short(memory_reader),
    }
    for name, executor in bindings.items():
        gateway.register(canonical_spec(name, executor=executor))
    return gateway
