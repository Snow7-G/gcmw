"""Tests for the RAG RetrievalService cascade (issue #53).

Coverage: FAQ exact match stage, structured filters, deterministic lexical
ranking with title/phrase boosts and stable tie-breaks, empty-query handling
and the never-null source guard.
"""

from app.rag.retrieval import RetrievalHit, RetrievalService, rank_items


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
    def __init__(self, items) -> None:
        self._items = list(items)

    def production_items(self, tenant_id=None):
        if tenant_id is None:
            return list(self._items)
        return [it for it in self._items if it.tenant_id == tenant_id]


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


class TestSearchCascade:
    def test_structured_filter_before_ranking(self):
        hits = _service().search("咳嗽", source_type="department", tenant_id="t1")
        assert all(h.source_type == "department" for h in hits)
        assert hits[0].source_id == "dept-resp"

    def test_medical_domain_and_audience_filters(self):
        service = _service(
            [
                _Item("a", "x", "咳嗽 护理", medical_domain="respiratory"),
                _Item("b", "y", "咳嗽 护理", medical_domain="oncology"),
            ]
        )
        hits = service.search("咳嗽", medical_domain="respiratory")
        assert [h.source_id for h in hits] == ["a"]

    def test_tenant_narrowing_delegated_to_source(self):
        service = _service([_Item("a", "x", "发热 处理", tenant_id="t2")])
        assert service.search("发热", tenant_id="t1") == []

    def test_top_k_limits_hits(self):
        hits = _service().search("科", top_k=1)
        assert len(hits) <= 1

    def test_no_source_returns_empty(self):
        assert RetrievalService(None).search("发热") == []

    def test_blank_query_returns_empty(self):
        assert _service().search("   ") == []

    def test_hit_exposes_evidence_fields(self):
        hits = _service().search("咳嗽 哮喘")
        assert hits
        hit = hits[0]
        assert isinstance(hit, RetrievalHit)
        assert hit.snippet
        assert hit.content_hash.startswith("hash-")
        assert hit.knowledge_version
        assert hit.source_uri
