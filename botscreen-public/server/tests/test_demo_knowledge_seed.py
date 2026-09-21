"""演示语料的可重复装载流程（app.knowledge.demo_seed）测试。

要钉住的三件事：
1. 装载**只走**既有审核发布流程（candidate → in_review → approve），不直写生产视图；
2. **重复执行零写入**：内容未变时不得 mint 新版本、不得新增审计事件
   （"重复启动不得重复造数据"）；
3. 内容变化时走 store 文档化的重发布路径（revoke → 重新发布），旧版本留在
   history 里可审计。

外加两条语料自身的硬约束（都是踩过的坑）：
- 每条正文必须是**单句**：支持门是"claim 归一化后等于某个 unit"的相等判定，
  句界只有 ``。！？；``，带内部句界的句子会在模型断句时失去支持 → 拒答。
- MockProvider 的 needle 取"第一个命中"，所以 needle 之间不得互相包含——
  旧的 ``眼睛`` 会截胡"眼睛干涩怎么办"并返回不匹配的证据句。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pytest import mark

from app.contracts.common import TenantContext
from app.contracts.errors import ErrorCode
from app.contracts.knowledge import (
    AuditAction,
    CandidateInput,
    KnowledgeSourceType,
    ReviewStatus,
)
from app.contracts.model import ModelRequest
from app.knowledge import demo_seed
from app.knowledge.demo_seed import (
    DEMO_CORPUS,
    DEMO_TENANT_ID,
    KNOWLEDGE_VALID_FROM,
    canned_replies,
    seed_demo_knowledge,
)
from app.knowledge.store import KnowledgeStore
from app.providers.mock import MockProvider

#: 支持门的句界字符（逗号不是句界，所以用 ``、`` 承载并列是安全的）
_SENTENCE_TERMINATORS = "。！？；!?;\n"


def _context(tenant_id: str = DEMO_TENANT_ID) -> TenantContext:
    return TenantContext(tenant_id=tenant_id)


def _request(text: str) -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": text}],
        trace_id="t-demo-seed",
        deadline_ms=5000,
        token_budget=400,
    )


class TestSeedGoesThroughTheReviewLifecycle:
    def test_first_seed_publishes_every_entry_via_approve(self):
        store = KnowledgeStore()
        report = seed_demo_knowledge(store)

        assert report.created == tuple(e["source_id"] for e in DEMO_CORPUS)
        assert report.refreshed == () and report.unchanged == ()
        for entry in DEMO_CORPUS:
            item = store.get(_context(), entry["source_id"])
            assert item.review_status is ReviewStatus.APPROVED
            assert item.content == entry["content"]
            assert item.knowledge_version == f"{entry['source_id']}-v1"
            assert item.reviewed_by == demo_seed.DEMO_REVIEWER
            # 三个审计事件，一个不多：candidate → in_review → approved
            actions = [
                event.action
                for event in store.audit_trail(_context(), entry["source_id"])
            ]
            assert actions == [
                AuditAction.CANDIDATE_ADDED,
                AuditAction.IN_REVIEW,
                AuditAction.APPROVED,
            ]

    def test_production_view_holds_exactly_the_corpus(self):
        store = KnowledgeStore()
        seed_demo_knowledge(store)
        items = store.production_items(_context())
        assert [i.source_id for i in items] == [e["source_id"] for e in DEMO_CORPUS]

    def test_draft_only_entry_is_lifted_to_approved(self):
        # 半成品状态（只进了 candidate）也要被收敛，而不是抛冲突
        store = KnowledgeStore()
        entry = DEMO_CORPUS[0]
        store.add_candidate(
            _context(),
            CandidateInput(
                source_id=entry["source_id"],
                source_type=KnowledgeSourceType.FAQ,
                title=entry["title"],
                content=entry["content"],
                source_uri=f"kbase://{entry['source_id']}",
            ),
            actor=demo_seed.DEMO_ACTOR,
        )
        report = seed_demo_knowledge(store)
        assert report.refreshed == (entry["source_id"],)
        assert (
            store.get(_context(), entry["source_id"]).review_status
            is ReviewStatus.APPROVED
        )


class TestReseedingIsIdempotent:
    def test_second_seed_writes_nothing_and_mints_no_version(self):
        store = KnowledgeStore()
        seed_demo_knowledge(store)
        before = {
            entry["source_id"]: (
                store.get(_context(), entry["source_id"]).knowledge_version,
                len(store.audit_trail(_context(), entry["source_id"])),
            )
            for entry in DEMO_CORPUS
        }

        report = seed_demo_knowledge(store)

        assert report.unchanged == tuple(e["source_id"] for e in DEMO_CORPUS)
        assert report.wrote is False
        after = {
            entry["source_id"]: (
                store.get(_context(), entry["source_id"]).knowledge_version,
                len(store.audit_trail(_context(), entry["source_id"])),
            )
            for entry in DEMO_CORPUS
        }
        assert after == before  # 版本号与审计事件数都不变：零写入

    def test_repeated_seeding_keeps_one_item_per_source(self):
        store = KnowledgeStore()
        for _ in range(3):
            seed_demo_knowledge(store)
        assert [i.source_id for i in store.production_items(_context())] == [
            e["source_id"] for e in DEMO_CORPUS
        ]

    def test_changed_content_is_republished_and_old_version_kept(self, monkeypatch):
        store = KnowledgeStore()
        seed_demo_knowledge(store)
        target = DEMO_CORPUS[0]["source_id"]
        new_content = "体温超过39度建议尽快门诊就诊。"

        patched = tuple(
            {**entry, "content": new_content} if entry["source_id"] == target else entry
            for entry in DEMO_CORPUS
        )
        monkeypatch.setattr(demo_seed, "DEMO_CORPUS", patched)

        report = seed_demo_knowledge(store)

        assert report.refreshed == (target,)
        current = store.get(_context(), target)
        assert current.content == new_content
        assert current.knowledge_version == f"{target}-v2"  # 版本前进，不是原地改
        # 旧版本仍在 history 里（撤销留痕），生产视图只给新版本
        versions = [
            item.knowledge_version for item in store.history(_context(), target)
        ]
        assert f"{target}-v1" in versions
        assert [
            i.content
            for i in store.production_items(_context())
            if i.source_id == target
        ] == [new_content]


class TestTenantIsolation:
    def test_other_tenant_sees_nothing(self):
        store = KnowledgeStore()
        seed_demo_knowledge(store)
        assert store.production_items(_context("t2")) == []


class TestCorpusHygiene:
    @pytest.mark.parametrize("entry", DEMO_CORPUS, ids=lambda e: e["source_id"])
    def test_content_is_a_single_sentence(self, entry):
        body = entry["content"].rstrip(_SENTENCE_TERMINATORS)
        assert not any(ch in body for ch in _SENTENCE_TERMINATORS), (
            f"{entry['source_id']} 的正文含内部句界：{entry['content']!r} —— "
            "支持门是相等判定，模型一旦在此断句就会判 evidence_unsupported"
        )

    @pytest.mark.parametrize("entry", DEMO_CORPUS, ids=lambda e: e["source_id"])
    def test_content_ends_with_a_terminator(self, entry):
        assert entry["content"][-1] in "。！？"

    def test_source_ids_and_needles_are_unique(self):
        ids = [e["source_id"] for e in DEMO_CORPUS]
        needles = [e["question"] for e in DEMO_CORPUS]
        assert len(set(ids)) == len(ids)
        assert len(set(needles)) == len(needles)

    def test_needles_never_contain_each_other(self):
        # MockProvider 取"第一个命中的 needle"；互相包含会让某个条目永远取不到
        for a in DEMO_CORPUS:
            for b in DEMO_CORPUS:
                if a is b:
                    continue
                assert a["question"] not in b["question"], (
                    f"needle {a['question']!r} 被 {b['question']!r} 包含"
                )


class TestCannedRepliesMatchTheCorpus:
    def test_reply_quotes_the_approved_sentence_verbatim(self):
        replies = canned_replies()
        for entry in DEMO_CORPUS:
            reply = replies[entry["question"]]
            assert reply.startswith(entry["content"].rstrip("。"))
            assert "（资料[1]）" in reply
            assert reply.endswith("。")

    @mark.asyncio
    async def test_each_demo_question_gets_its_own_cited_sentence(self):
        provider = MockProvider(canned=canned_replies())
        for entry in DEMO_CORPUS:
            response = await provider.chat(_request(f"{entry['question']}怎么办"))
            assert response.content == canned_replies()[entry["question"]]

    @mark.asyncio
    async def test_dry_eye_question_is_not_hijacked_by_the_eye_needle(self):
        """回归：旧 needle「眼睛」会截胡"眼睛干涩怎么办"。

        那种情况下照抄给模型的是**眼科**那句，而检索命中的是干涩那份资料 →
        支持门判不支持 → 用户侧表现为"答不出来"。
        """
        provider = MockProvider(canned=canned_replies())
        dry_eye = next(e for e in DEMO_CORPUS if e["source_id"] == "faq-dry-eye")
        response = await provider.chat(_request("眼睛干涩怎么办"))
        assert response.content == canned_replies()[dry_eye["question"]]
        assert "眼部不适需及时就诊" not in response.content

    @mark.asyncio
    async def test_unknown_question_falls_through_to_a_non_answer(self):
        # 没有 needle 命中时不得编造语料之外的句子
        provider = MockProvider(canned=canned_replies())
        response = await provider.chat(_request("今天天气怎么样"))
        assert response.content not in canned_replies().values()


class TestAssemblyUsesTheRepeatableFlow:
    def test_assembly_calls_the_seed_once_and_reuses_the_constants(self, monkeypatch):
        from app.orchestration import assembly

        calls: list[int] = []
        real = assembly.seed_demo_knowledge

        def recorder(store):
            calls.append(1)
            return real(store)

        monkeypatch.setattr(assembly, "seed_demo_knowledge", recorder)
        from app.config import Settings

        executor = assembly.build_agent_executor(
            repository=object(), settings=Settings(environment="test")
        )

        assert executor is not None
        assert len(calls) == 1, "装配必须且只能调用一次可重复装载流程"
        assert assembly.DEMO_TENANT_ID == DEMO_TENANT_ID
        assert KNOWLEDGE_VALID_FROM == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_assembly_re_exports_the_seeded_constants(self):
        from app.orchestration import assembly

        # 语料搬走后，历史引用（__all__ 里的 DEMO_TENANT_ID）必须仍然可见
        assert "DEMO_TENANT_ID" in assembly.__all__


class TestErrorCodeSurfaceUnchanged:
    def test_refusal_codes_still_exist_for_the_offline_demo(self):
        # 语料扩充不该动到错误码面（顺手钉住，避免误改）
        assert ErrorCode.VALIDATION_INVALID_INPUT.value == "E_VALIDATION_INVALID_INPUT"
