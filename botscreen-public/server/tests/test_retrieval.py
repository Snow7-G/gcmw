"""Tests for the RAG RetrievalService cascade (issue #53).

Coverage: FAQ exact match stage, structured filters, deterministic lexical
ranking with title/phrase boosts and stable tie-breaks, empty-query handling,
the never-null source guard, and the TENANCY contract — a trusted
``TenantContext`` is required (a bare tenant id is refused), the production view
receives exactly that context, and the same ``source_id`` in another tenant is
invisible.
"""

from typing import ClassVar

import pytest
from pytest import mark

from app.contracts.common import TenantContext
from app.rag.retrieval import (
    RetrievalHit,
    RetrievalService,
    normalize_core_query,
    rank_items,
)

T1 = TenantContext(tenant_id="t1")
T2 = TenantContext(tenant_id="t2")


class _Item:
    def __init__(
        self,
        source_id,
        title,
        content,
        source_type="faq",
        tenant_id="t1",
        medical_domain="general",
        audience="public",
        knowledge_version=None,
    ) -> None:
        self.source_id = source_id
        self.title = title
        self.content = content
        self.source_type = source_type
        self.tenant_id = tenant_id
        self.medical_domain = medical_domain
        self.audience = audience
        self.knowledge_version = knowledge_version or f"{source_id}-v1"
        self.content_hash = f"hash-{source_id}"
        self.source_uri = f"kbase://{source_id}"


class _Source:
    """Production-view stand-in: scoped by the TRUSTED context it is handed."""

    def __init__(self, items) -> None:
        self._items = list(items)
        self.contexts: list[object] = []

    def production_items(self, context):
        self.contexts.append(context)
        return [it for it in self._items if it.tenant_id == context.tenant_id]


_FAQ = [
    _Item(
        "faq-fever",
        "发热就诊指南",
        "成人发热超过三天或持续高热建议门诊就诊。",
    ),
    _Item(
        "faq-cough",
        "咳嗽护理",
        "咳嗽伴随呼吸困难或胸痛需要立即就医。",
        medical_domain="respiratory",
    ),
    _Item(
        "dept-resp",
        "呼吸内科简介",
        "呼吸内科诊治咳嗽、哮喘与慢性阻塞性肺疾病。",
        source_type="department",
        medical_domain="respiratory",
    ),
]


def _service(items=None):
    return RetrievalService(_Source(items if items is not None else _FAQ))


class TestRanking:
    def test_returns_stable_sorted_triples(self):
        ranked = rank_items(_FAQ, "咳嗽")
        assert ranked == sorted(ranked)  # deterministic order
        assert ranked[0][2].source_id in {"faq-cough", "dept-resp"}

    def test_title_hit_outranks_content_only(self):
        items = [
            _Item("content-only", "无关标题", "内容是 发热 处理 方法"),
            _Item("title-hit", "发热 处理", "正文没有关键词"),
        ]
        ranked = rank_items(items, "发热")
        assert ranked[0][2].source_id == "title-hit"

    def test_exact_faq_content_boost(self):
        items = [
            _Item("exact", "标题", "体温多少算发热"),
            _Item("partial", "标题", "体温超过 38.5 通常算发热"),
        ]
        ranked = rank_items(items, "体温多少算发热")
        assert ranked[0][2].source_id == "exact"

    def test_empty_query_returns_empty(self):
        assert rank_items(_FAQ, "  ") == []


class TestRelevanceGate:
    """Review P1: one shared CJK character is not evidence.

    Candidacy needs a CJK bigram, an ASCII/numeric token or the normalized
    phrase; a lone CJK character may only contribute to the score.
    """

    _EYE: ClassVar[list] = [
        _Item("eye-pain", "眼痛就诊指南", "眼痛伴视力下降请及时到眼科就诊。"),
    ]

    def test_a_single_shared_character_is_not_a_match(self):
        svc = _service(self._EYE)
        assert svc.search("腹痛怎么办", context=T1) == []

    def test_a_single_character_query_is_refused_by_default(self):
        svc = _service(self._EYE)
        assert svc.search("痛", context=T1) == []  # fail closed
        assert svc.search("眼", context=T1) == []

    def test_a_shared_bigram_still_matches(self):
        svc = _service(self._EYE)
        assert [h.source_id for h in svc.search("眼痛怎么办", context=T1)] == [
            "eye-pain"
        ]

    def test_a_natural_question_matches_a_title_term(self):
        svc = _service(_FAQ)
        assert [h.source_id for h in svc.search("发热怎么办", context=T1)] == [
            "faq-fever"
        ]

    @mark.parametrize("query", ["xx", "needle"])
    def test_multi_character_ascii_substring_recall_is_preserved(self, query):
        """#57 relies on a >=2-char ASCII token finding a long unbroken run."""
        content = "x" * 400 if query == "xx" else "a" * 100 + "needle" + "b" * 100
        items = [_Item("long", "标题", content)]
        hits = _service(items).search(query, context=T1)
        assert [h.source_id for h in hits] == ["long"]

    def test_a_single_ascii_character_is_not_evidence(self):
        items = [_Item("long", "标题", "x" * 400)]
        assert _service(items).search("x", context=T1) == []
        assert _service(items).search("1", context=T1) == []

    @mark.parametrize(
        ("query", "title", "content"),
        [
            ("发热怎么办", "腹泻怎么办", "腹泻怎么办？"),
            ("腹痛怎么办", "眼痛怎么办", "眼痛怎么办？"),
            ("1型糖尿病怎么办", "1号楼眼科", "1号楼眼科门诊安排。"),
            ("A型流感怎么办", "维生素A说明", "维生素A说明与用法。"),
        ],
    )
    def test_scaffolding_and_single_characters_are_not_evidence(
        self, query, title, content
    ):
        """Review P1: a shared 怎么办, digit or letter proves nothing.

        These pairs share ONLY scaffolding or a single ASCII/digit character, and
        each of them produced a "grounded" answer before the content-word gate.
        """
        assert _service([_Item("only", title, content)]).search(query, context=T1) == []

    @mark.parametrize(
        ("query", "title", "content"),
        [
            ("发热怎么办", "发热指南", "成人发热超过三天建议门诊就诊。"),
            ("腹泻怎么办", "腹泻处理指南", "腹泻伴脱水表现需及时就诊。"),
            ("1型糖尿病怎么办", "1型糖尿病指南", "1型糖尿病需规范监测血糖。"),
            ("B超怎么做", "B超检查指南", "B超检查前需空腹八小时。"),
        ],
    )
    def test_content_word_hits_still_match(self, query, title, content):
        """The gate must not be over-broad: real content words still hit."""
        hits = _service([_Item("hit", title, content)]).search(query, context=T1)
        assert [h.source_id for h in hits] == ["hit"]

    def test_scaffolding_is_stripped_before_scoring(self):
        assert normalize_core_query("发热怎么办") == "发热"
        assert normalize_core_query("请问B超怎么做") == "b超"
        assert normalize_core_query("怎么处理发热") == "发热"
        assert normalize_core_query("可以吗") == ""  # only scaffolding: no content

    def test_a_single_character_never_decides_candidacy_but_still_scores(self):
        """Two eligible docs: the one also sharing the lone character ranks first."""
        items = [
            _Item("with-char", "标题", "发热 痛 处理"),
            _Item("without", "标题", "发热 处理"),
        ]
        ranked = rank_items(items, "痛 发热")
        assert next(item.source_id for _, _, item in ranked) == "with-char"
        # …and with no eligible token at all, the character alone proves nothing
        assert rank_items(items, "痛") == []


class TestSearchCascade:
    def test_structured_filter_before_ranking(self):
        hits = _service().search("咳嗽", source_type="department", context=T1)
        assert all(h.source_type == "department" for h in hits)
        assert hits[0].source_id == "dept-resp"

    def test_medical_domain_and_audience_filters(self):
        service = _service(
            [
                _Item("a", "x", "咳嗽 护理", medical_domain="respiratory"),
                _Item("b", "y", "咳嗽 护理", medical_domain="oncology"),
            ]
        )
        hits = service.search("咳嗽", context=T1, medical_domain="respiratory")
        assert [h.source_id for h in hits] == ["a"]

    def test_another_tenants_item_is_invisible(self):
        service = _service([_Item("a", "x", "发热 处理", tenant_id="t2")])
        assert service.search("发热", context=T1) == []
        assert [h.source_id for h in service.search("发热", context=T2)] == ["a"]

    def test_the_same_source_id_is_isolated_per_tenant(self):
        """Identical source_id in two tenants: each sees only its own record."""
        shared = [
            _Item("faq-fever", "发热指南", "t1 版：成人发热三天就诊", tenant_id="t1"),
            _Item("faq-fever", "发热指南", "t2 版：儿童发热两天就诊", tenant_id="t2"),
        ]
        service = _service(shared)
        t1_hits = service.search("发热", context=T1)
        t2_hits = service.search("发热", context=T2)
        assert [h.source_id for h in t1_hits] == ["faq-fever"]
        assert "t1 版" in t1_hits[0].snippet
        assert "t2 版" in t2_hits[0].snippet

    def test_a_bare_tenant_id_is_refused(self):
        """Tenancy is an input, not a guess: no context -> no query."""
        for bogus in ("t1", None, 1):
            with pytest.raises(TypeError):
                _service().search("发热", context=bogus)  # type: ignore[arg-type]

    def test_the_production_view_receives_the_trusted_context(self):
        source = _Source(_FAQ)
        RetrievalService(source).search("咳嗽", context=T1)
        assert source.contexts == [T1]  # the very object, not a derived id

    def test_top_k_limits_hits(self):
        hits = _service().search("内科", context=T1, top_k=1)
        assert len(hits) <= 1

    def test_no_source_returns_empty(self):
        assert RetrievalService(None).search("发热", context=T1) == []

    def test_blank_query_returns_empty(self):
        assert _service().search("   ", context=T1) == []

    def test_hit_exposes_evidence_fields(self):
        hits = _service().search("咳嗽 哮喘", context=T1)
        assert hits
        hit = hits[0]
        assert isinstance(hit, RetrievalHit)
        assert hit.snippet
        assert hit.content_hash.startswith("hash-")
        assert hit.knowledge_version
        assert hit.source_uri
