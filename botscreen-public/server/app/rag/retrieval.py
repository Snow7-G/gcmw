"""RAG retrieval service (issue #53) — deterministic, evidence-grade.

Cascade implemented in v1 (V2.3 §6.2 subset):
1. FAQ exact match      — normalized whole-text equivalence wins;
2. structured filters   — source_type / medical_domain / audience narrowing;
3. lexical ranking      — token-overlap scoring over title+content (BM25-style
                          inverse-frequency weighting, deterministic);
4. lightweight rerank   — title hits and phrase containment boost, stable
                          insertion-order tie-break.

Only the *production view* of a knowledge source may be queried — the source
is any object exposing ``production_items(tenant_id) -> list`` of items with
``source_id/title/content/...`` attributes (the #56 KnowledgeStore satisfies
this structurally; its production view already gates review status and
validity windows, so unreviewed content is unreachable here by construction).
PostgreSQL reconciliation and vector recall arrive with the storage layer
(#40) behind the same interface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_PHRASE_SPLIT = re.compile(r"\s+")
_SNIPPET_CHARS = 240


def _value(member: Any) -> Any:
    return getattr(member, "value", member)


def _attr(item: Any, name: str, default: Any = "") -> Any:
    return getattr(item, name, default)


def _is_cjk(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def tokenize(text: str) -> list[str]:
    """Deterministic token stream: ASCII words plus CJK unigrams and bigrams.

    CJK text carries no spaces, so plain word splitting would never match a
    query term to a compound (e.g. ``咳嗽`` inside ``呼吸内科诊治咳嗽``).
    Every CJK character is emitted as a token together with its preceding CJK
    bigram — substring recall comes from the bigram, precision ordering from
    the phrase/exact cascade stages that follow.
    """
    lowered = (text or "").lower()
    tokens: list[str] = []
    word = ""
    prev_cjk = ""
    for ch in lowered:
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            word += ch
            prev_cjk = ""
            continue
        if word:
            tokens.append(word)
            word = ""
        if _is_cjk(ch):
            if prev_cjk:
                tokens.append(prev_cjk + ch)
            tokens.append(ch)
            prev_cjk = ch
        else:
            prev_cjk = ""
    if word:
        tokens.append(word)
    return tokens


def normalize_text(text: str) -> str:
    """Whitespace-normalized lowercase text for exact-match stage."""
    return " ".join(_PHRASE_SPLIT.split((text or "").lower())).strip()


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked, filter-passing hit over the production view."""

    source_id: str
    score: int
    source_type: str
    title: str
    snippet: str
    content_hash: str = ""
    source_uri: str = ""
    medical_domain: str = ""
    audience: str = ""
    knowledge_version: str = ""


def _document(item: Any) -> str:
    return f"{_attr(item, 'title')} {_attr(item, 'content')}".lower()


def rank_items(items: list[Any], query: str) -> list[tuple[int, int, Any]]:
    """Deterministic relevance ranking (token overlap, doc-frequency
    weighted, title boost, phrase containment boost; tie = insertion order).

    Returns ``(negative_score, insertion_index, item)`` triples so the caller
    can sort stably without ever touching item internals.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    norm_query = normalize_text(query)
    frequencies: dict[str, int] = {}
    token_sets: list[set[str]] = []
    for item in items:
        tokens = set(tokenize(_document(item)))
        token_sets.append(tokens)
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

    total = max(len(items), 1)
    scored: list[tuple[int, int, Any]] = []
    for index, (item, tokens) in enumerate(zip(items, token_sets)):
        title_tokens = set(tokenize(_attr(item, "title")))
        score = 0
        matched_any = False
        for token in query_tokens:
            if token not in tokens:
                continue
            matched_any = True
            idf = math.log(1 + total / (1 + frequencies.get(token, 0)))
            score += idf
            if token in title_tokens:
                score += 2 * idf  # lightweight rerank: title hits weigh more
        if not matched_any:
            continue
        if norm_query and norm_query in normalize_text(_document(item)):
            score += 10  # phrase containment boost
        if norm_query and norm_query == normalize_text(_attr(item, "content")):
            score += 5  # FAQ exact-match cascade stage
        scored.append((-score, index, item))
    return sorted(scored)


def _hit(item: Any, score: int, snippet_chars: int = _SNIPPET_CHARS) -> RetrievalHit:
    content = _attr(item, "content")
    return RetrievalHit(
        source_id=_attr(item, "source_id"),
        score=score,
        source_type=str(_value(_attr(item, "source_type"))),
        title=_attr(item, "title"),
        snippet=content[:snippet_chars],
        content_hash=_attr(item, "content_hash"),
        source_uri=_attr(item, "source_uri"),
        medical_domain=_attr(item, "medical_domain"),
        audience=_attr(item, "audience"),
        knowledge_version=_attr(item, "knowledge_version"),
    )


class RetrievalService:
    """Cascade retrieval over one production-view source (deterministic)."""

    def __init__(self, source: Any, *, snippet_chars: int = _SNIPPET_CHARS) -> None:
        self._source = source
        self._snippet_chars = snippet_chars

    def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        top_k: int = 5,
        source_type: str | None = None,
        medical_domain: str | None = None,
        audience: str | None = None,
    ) -> list[RetrievalHit]:
        """Query the production view only. Filters apply before ranking;
        results are sorted by descending score with stable ties."""
        if not (query or "").strip():
            return []
        if self._source is None:
            return []
        production = list(self._source.production_items(tenant_id))

        # structured-filter stage
        if source_type is not None:
            production = [
                it
                for it in production
                if str(_value(_attr(it, "source_type"))) == source_type
            ]
        if medical_domain is not None:
            production = [
                it for it in production if _attr(it, "medical_domain") == medical_domain
            ]
        if audience is not None:
            production = [it for it in production if _attr(it, "audience") == audience]

        ranked = rank_items(production, query)[:top_k]
        return [_hit(item, -neg, self._snippet_chars) for neg, _, item in ranked]
