"""Reference read-only executors for the whitelisted tools (issue #57).

All executors are pure reads over injectable sources:

- ``knowledge.search`` / ``knowledge.get_fragment`` read the production view
  of any object exposing ``production_items(tenant_id) -> list`` whose items
  carry ``source_id/title/content/...`` attributes — the #56 KnowledgeStore
  satisfies this structurally and its production view is the only view #53
  RAG may query;
- ``department/staff/video.search`` run a deterministic substring search over
  an optional in-memory directory index (placeholder until real indexes land
  in a later issue);
- ``memory.read_short`` reads a short-term summary through an optional
  read-only callable.

Executors never write, never touch files/network and raise
:class:`~app.tools.gateway.ToolGatewayError` (registry ErrorCode) for
domain-level outcomes such as a missing knowledge source.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from app.contracts.errors import ErrorCode
from app.tools.gateway import ToolGateway, ToolGatewayError
from app.tools.specs import canonical_spec

_TOKEN_SPLIT = re.compile(r"[\s,，。；;：:、|/\\()\[\]{}<>«»\"'“”‘’!?！？.．\-—_]+")
_SNIPPET_CHARS = 240

Directories = Mapping[str, list[Mapping[str, Any]]]
MemoryReader = Callable[[str | None], str | None]


def _value(member: Any) -> Any:
    """Enum members expose .value; plain strings pass through unchanged."""
    return getattr(member, "value", member)


def _attr(item: Any, name: str, default: Any = "") -> Any:
    return getattr(item, name, default)


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


def make_knowledge_search(store: Any | None) -> Callable[[dict], Any]:
    """Deterministic relevance search over a production-view source."""

    def search(args: dict[str, Any]) -> dict[str, Any]:
        query = _arg(args, "query", "")
        top_k = _arg(args, "top_k", 5)
        tenant_id = _arg(args, "tenant_id", None)
        if store is None:
            return {"items": [], "total": 0}
        tokens = set(_tokens(query))
        scored: list[tuple[int, int, Any]] = []
        for index, item in enumerate(store.production_items(tenant_id)):
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


def make_knowledge_get_fragment(store: Any | None) -> Callable[[dict], Any]:
    def get_fragment(args: dict[str, Any]) -> dict[str, Any]:
        source_id = args["source_id"]
        fragment_index = _arg(args, "fragment_index", 0)
        fragment_chars = _arg(args, "fragment_chars", 2000)
        item = None
        if store is not None:
            item = next(
                (
                    it
                    for it in store.production_items(_arg(args, "tenant_id", None))
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
) -> Callable[[dict], Any]:
    def search(args: dict[str, Any]) -> dict[str, Any]:
        query = _arg(args, "query", "")
        top_k = _arg(args, "top_k", 5)
        tenant_id = _arg(args, "tenant_id", None)
        records = list((directories or {}).get(domain, []))
        if tenant_id is not None:
            records = [r for r in records if r.get("tenant_id", tenant_id) == tenant_id]
        tokens = _tokens(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            haystack = " ".join(str(v) for v in record.values()).lower()
            hits = sum(1 for t in tokens if t in haystack)
            if hits:
                scored.append((-hits, index, dict(record)))
        scored.sort()
        items = [record for _, _, record in scored[:top_k]]
        return {"items": items, "total": len(items)}

    return search


def make_memory_read_short(reader: MemoryReader | None) -> Callable[[dict], Any]:
    def read_short(args: dict[str, Any]) -> dict[str, Any]:
        session_id = _arg(args, "session_id", None)
        if reader is None:
            return {"summary": "", "available": False}
        summary = reader(session_id)
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
    given (read-only) sources. Sources default to empty, deterministic stubs."""
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
