"""RAG retrieval service (issue #53) — deterministic, evidence-grade.

Cascade implemented in v1 (V2.3 §6.2 subset):
1. FAQ exact match      — normalized whole-text equivalence wins;
2. structured filters   — source_type / medical_domain / audience narrowing;
3. lexical ranking      — token-overlap scoring over title+content (BM25-style
                          inverse-frequency weighting, deterministic);
4. lightweight rerank   — title hits and phrase containment boost, stable
                          insertion-order tie-break.

CANDIDACY is gated by relevance, not by scoring: a document enters the result
set only when a CJK bigram, an ASCII/numeric token or the normalized phrase
matches. A single shared CJK character ("痛" in 腹痛 vs 眼痛) may contribute to
the score but can never justify calling the record evidence, so an unrelated
approved item cannot produce a "grounded" answer.

Only the *production view* of a knowledge source may be queried — the source is
any object exposing ``production_items(context) -> list`` of items with
``source_id/title/content/...`` attributes (the #56 KnowledgeStore satisfies this
structurally; its production view already gates review status, validity windows
and supersession, so unreviewed/expired/revoked content is unreachable here by
construction). PostgreSQL reconciliation and vector recall arrive with the
storage layer (#40) behind the same interface.

TENANCY: ``search`` takes a trusted
:class:`app.contracts.common.TenantContext` — never a tenant id string. The
tenant of a query is therefore decided by the server-injected context, and the
production view is asked for exactly that tenant, so the same ``source_id`` in
another tenant is invisible here. A caller without a trusted context cannot
query at all (fail loud, not empty results).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.contracts.common import TenantContext

_PHRASE_SPLIT = re.compile(r"\s+")
_SNIPPET_CHARS = 240


def _value(member: Any) -> Any:
    return getattr(member, "value", member)


def _attr(item: Any, name: str, default: Any = "") -> Any:
    return getattr(item, name, default)


def _is_cjk(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def _token_class(token: str) -> str:
    """What a matched token is allowed to PROVE.

    ``cjk1`` — a single CJK character. It may contribute to the score but can
    never make a document a candidate on its own: one shared common character
    ("痛" in 腹痛 vs 眼痛) would otherwise turn an unrelated approved record into
    "evidence". ``cjk`` — a CJK bigram or longer. ``word`` — an ASCII/numeric
    token (any length: #57's substring recall relies on short tokens finding long
    unbroken runs).
    """
    if all(_is_cjk(ch) for ch in token):
        return "cjk1" if len(token) == 1 else "cjk"
    return "word"


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
        document = _document(item)
        score = 0
        # CANDIDACY (relevance gate) is separate from SCORING: a hit must be
        # proven by a CJK bigram, an ASCII/numeric token, or the normalized
        # phrase — never by a lone CJK character.
        eligible = False
        for token in query_tokens:
            # recall is TOKEN-set membership or plain substring containment: a
            # long unbroken run ("xxxx…") is one ASCII token, so a short query
            # token must still be able to find it inside the text
            if token not in tokens and token not in document:
                continue
            if _token_class(token) != "cjk1":
                eligible = True
            idf = math.log(1 + total / (1 + frequencies.get(token, 0)))
            score += idf
            if token in title_tokens:
                score += 2 * idf  # lightweight rerank: title hits weigh more
        if len(norm_query) >= 2 and norm_query in document:
            eligible = True
            score += 10  # phrase containment boost
        if norm_query and norm_query == normalize_text(_attr(item, "content")):
            score += 5  # FAQ exact-match cascade stage
        if not eligible:
            continue
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
        context: TenantContext,
        top_k: int = 5,
        source_type: str | None = None,
        medical_domain: str | None = None,
        audience: str | None = None,
    ) -> list[RetrievalHit]:
        """Query the production view of ``context.tenant_id`` only.

        Filters apply before ranking; results are sorted by descending score with
        stable ties. The context is REQUIRED and must be a trusted
        :class:`TenantContext`: tenancy is an input to this function, not an
        argument it will guess, and a bare tenant id (or ``None``) is refused
        rather than silently widening the query.
        """
        if not isinstance(context, TenantContext):
            raise TypeError(
                "retrieval requires a trusted TenantContext, not "
                f"{type(context).__name__}"
            )
        if not (query or "").strip():
            return []
        if self._source is None:
            return []
        production = list(self._source.production_items(context))

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
