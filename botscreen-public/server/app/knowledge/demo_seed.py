"""合成演示知识的**可重复执行**装载流程。

为什么单独成模块：这份语料是演示数据，装载必须**每次启动都能安全重跑**。

- 入口只有一个：既有的 #56 治理生命周期（candidate → in_review → approve）。
  绝不直接把记录塞进 production 视图。
- **内容未变时零写入**。否则每次启动都会 mint 一个新的 ``<id>-v<N>``：审计里
  堆满噪声，且"知识版本"这个演示卖点失去意义（重复启动不得重复造数据）。
- **内容变化时**才走 store 自己文档化的重发布路径
  ``revoke → candidate → in_review → approve``，旧版本留在 history 里可审计，
  生产视图也不会出现"已被替换却没留痕"的窗口。

语料本身是**合成**句子（不是真实医疗知识），并且刻意写成**单句**——这是硬约束：
支持门是"claim 归一化后必须等于某个 unit"的相等判定，而句界只有 ``。！？；``
（逗号不是句界，归一化时被删）。带内部句界的句子只有在模型原样照抄成一句时才
通过；一旦模型在逗号处断句，两段都失去支持 → ``evidence_unsupported`` → 拒答。
因此新增条目一律单句，用 ``、`` 承载并列（它会被归一化掉且不是句界）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..contracts.common import TenantContext
from ..contracts.knowledge import (
    ApprovalDecision,
    CandidateInput,
    KnowledgeSourceType,
    ReviewStatus,
    RevocationDecision,
)
from ..knowledge.store import KnowledgeGovernanceError, KnowledgeStore

#: the tenant the synthetic demo knowledge is published for. A run from any
#: other tenant must find no evidence and be refused (isolation is real).
DEMO_TENANT_ID = "t1"

#: knowledge validity window for the synthetic corpus (fixed, aware UTC).
KNOWLEDGE_VALID_FROM = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: 演示语料的动作主体（固定，便于审计追踪到"这是演示数据"）。
DEMO_ACTOR = "demo-content-owner"
DEMO_REVIEWER = "demo-reviewer"

#: the demo corpus. ``content`` MUST be a single sentence (see module docstring);
#: ``question`` doubles as the MockProvider needle, and MockProvider matches the
#: FIRST needle contained in the request text — so needles must be mutually
#: non-overlapping ("眼部" rather than "眼睛", which would also capture
#: "眼睛干涩怎么办" and answer it with the wrong, unsupported sentence).
DEMO_CORPUS: tuple[dict[str, str], ...] = (
    {
        "source_id": "faq-fever",
        "title": "发热护理须知",
        "content": "体温超过38.5建议门诊就诊。",
        "question": "发热",
    },
    {
        "source_id": "faq-eye",
        "title": "用眼卫生须知",
        "content": "眼部不适需及时就诊。",
        "question": "眼部",
    },
    {
        "source_id": "faq-myopia",
        "title": "近视矫正须知",
        "content": "近视可通过配镜或手术矫正但眼轴增长不可逆。",
        "question": "近视",
    },
    {
        "source_id": "faq-dry-eye",
        "title": "眼睛干涩护理须知",
        "content": "眼睛干涩可先注意休息与热敷、持续不缓解需就诊。",
        "question": "干涩",
    },
)


@dataclass(frozen=True)
class DemoSeedReport:
    """一次装载的可核对结果（审计/测试用）。"""

    created: tuple[str, ...]
    refreshed: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def wrote(self) -> bool:
        return bool(self.created or self.refreshed)


def _cite(sentence: str) -> str:
    """Compose a model reply whose citation sits INSIDE the sentence.

    The support gate splits claims on sentence terminators, so a marker placed
    after the closing 。 would land in a bodyless fragment and be read as a
    dangling citation. A citation always goes BEFORE the sentence-ending
    punctuation — which is also what the #53 prompt asks for."""
    body = sentence.rstrip("。")
    return f"{body}（资料[1]）。"


def canned_replies() -> dict[str, str]:
    """MockProvider 的 needle → 回复映射（复用物化语料）。

    MockProvider 取"第一个命中的 needle"，所以顺序即优先级；语料本身保证
    needle 互不包含（见 ``DEMO_CORPUS`` 注释）。
    """
    return {entry["question"]: _cite(entry["content"]) for entry in DEMO_CORPUS}


def _approve(
    store: KnowledgeStore, context: TenantContext, entry: dict[str, str]
) -> None:
    """candidate → in_review → approve：唯一的发布入口。"""
    store.add_candidate(
        context,
        CandidateInput(
            source_id=entry["source_id"],
            source_type=KnowledgeSourceType.FAQ,
            title=entry["title"],
            content=entry["content"],
            source_uri=f"kbase://{entry['source_id']}",
        ),
        actor=DEMO_ACTOR,
    )
    store.mark_in_review(context, entry["source_id"], actor=DEMO_ACTOR)
    store.approve(
        context,
        entry["source_id"],
        ApprovalDecision(reviewer=DEMO_REVIEWER, valid_from=KNOWLEDGE_VALID_FROM),
    )


def seed_demo_knowledge(
    store: KnowledgeStore, *, tenant_id: str = DEMO_TENANT_ID
) -> DemoSeedReport:
    """把演示语料收敛到"已批准且内容与代码一致"，可重复执行。

    对每个条目：已批准且正文相同 → **零写入**；否则（缺失 / 草稿 / 审核中 /
    已撤销 / 正文变化）先按治理流程撤销（若存在）再重新发布，旧版本进 history。
    """
    context = TenantContext(tenant_id=tenant_id)
    created: list[str] = []
    refreshed: list[str] = []
    unchanged: list[str] = []

    for entry in DEMO_CORPUS:
        source_id = entry["source_id"]
        try:
            current = store.get(context, source_id)
        except KnowledgeGovernanceError:
            _approve(store, context, entry)
            created.append(source_id)
            continue

        if (
            current.review_status is ReviewStatus.APPROVED
            and current.content == entry["content"]
        ):
            unchanged.append(source_id)  # 重复启动：零写入，不 mint 新版本
            continue

        # 内容变了 / 还停在草稿或审核中：走 store 文档化的重发布路径
        store.revoke(
            context,
            source_id,
            RevocationDecision(actor=DEMO_ACTOR, reason="demo corpus refresh"),
        )
        _approve(store, context, entry)
        refreshed.append(source_id)

    return DemoSeedReport(
        created=tuple(created), refreshed=tuple(refreshed), unchanged=tuple(unchanged)
    )
