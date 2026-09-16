"""Agent manifest, context and result contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from .common import Channel
from .errors import ErrorCode


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    supported_intents: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    required_data_scopes: list[str] = Field(default_factory=list)
    default_timeout_ms: int = Field(10000, gt=0)
    risk_level: RiskLevel = RiskLevel.LOW
    enabled: bool = True


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., min_length=1, max_length=64)
    device_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    run_id: str = Field(..., min_length=1, max_length=128)
    channel: Channel
    normalized_input: str = ""
    short_memory_summary: str = ""
    permitted_long_memory: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    deadline: AwareDatetime | None = None
    # call-time tool quota (#55A-B): the MAX tool calls the sub-agent may make
    # in THIS engagement, granted by the Manager from the run-cumulative
    # budget (revisions see what is left). ``None`` = unrestricted (legacy
    # callers); the Manager always grants an explicit number.
    tool_budget_granted: int | None = Field(default=None, ge=0)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    deadline: AwareDatetime | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, max_length=128)
    ok: bool = True
    data: Any = None
    error_code: ErrorCode | None = None
    error_message: str | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=128)
    source_type: str = Field(..., min_length=1, max_length=64)
    title: str = ""
    content: str = ""
    source_uri: str = ""
    content_hash: str = ""
    knowledge_version: str = ""


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=1, max_length=128)
    status: AgentStatus
    answer_candidate: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    confidence_band: str | None = None
    safety_status: str = "unknown"
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    public_trace: list[dict[str, Any]] = Field(default_factory=list)
    # trusted model provenance (#55A-B): ONLY the server-side
    # ModelGateway/Provider configuration, never model-reported values. The
    # Manager overwrites whatever the sub-agent claimed before this result is
    # built, so a forged provider/model identity can never reach the SSE layer.
    model: dict[str, str] | None = None
