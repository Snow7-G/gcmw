"""Unified error contract: stable error codes, safe envelopes and mappings.

Scope (issue #35): the contract layer only. FastAPI exception handlers are
added in #36 and the SSE protocol is unchanged (#30).

External error responses MUST use :class:`ErrorEnvelope` — it carries only
safe fields (code, safe message, request/trace ids, retry hints) and forbids
free-form ``details`` so provider output, local paths or stack traces can
never leak through the model.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# ErrorCode: stable, enumerable error identifiers. Free-form strings are not
# accepted anywhere in the contracts; unknown codes fail validation.
# ---------------------------------------------------------------------------


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    PROVIDER = "provider"
    TOOL = "tool"
    MODEL = "model"
    SAFETY = "safety"
    MEMORY = "memory"
    INTERNAL = "internal"


class ErrorCode(str, Enum):
    # validation
    VALIDATION_INVALID_INPUT = "E_VALIDATION_INVALID_INPUT"
    VALIDATION_PAYLOAD_TOO_LARGE = "E_VALIDATION_PAYLOAD_TOO_LARGE"
    VALIDATION_UNSUPPORTED_MODALITY = "E_VALIDATION_UNSUPPORTED_MODALITY"
    VALIDATION_UNSUPPORTED_FORMAT = "E_VALIDATION_UNSUPPORTED_FORMAT"
    # authentication / authorization
    AUTH_MISSING_CREDENTIALS = "E_AUTH_MISSING_CREDENTIALS"
    AUTH_INVALID_CREDENTIALS = "E_AUTH_INVALID_CREDENTIALS"
    AUTH_DEVICE_NOT_REGISTERED = "E_AUTH_DEVICE_NOT_REGISTERED"
    AUTHZ_FORBIDDEN = "E_AUTHZ_FORBIDDEN"
    # rate limiting / conflicts
    RATE_LIMIT_EXCEEDED = "E_RATE_LIMIT_EXCEEDED"
    CONFLICT_ACTIVE_RUN = "E_CONFLICT_ACTIVE_RUN"
    CONFLICT_IDEMPOTENCY = "E_CONFLICT_IDEMPOTENCY"
    # resources
    NOT_FOUND_SESSION = "E_NOT_FOUND_SESSION"
    NOT_FOUND_RUN = "E_NOT_FOUND_RUN"
    NOT_FOUND_KNOWLEDGE = "E_NOT_FOUND_KNOWLEDGE"
    NOT_FOUND_AGENT = "E_NOT_FOUND_AGENT"
    # timeouts / availability
    TIMEOUT_AGENT = "E_TIMEOUT_AGENT"
    TIMEOUT_PROVIDER = "E_TIMEOUT_PROVIDER"
    UNAVAILABLE_OVERLOADED = "E_UNAVAILABLE_OVERLOADED"
    UNAVAILABLE_MAINTENANCE = "E_UNAVAILABLE_MAINTENANCE"
    # provider / tool / model
    PROVIDER_UNREACHABLE = "E_PROVIDER_UNREACHABLE"
    PROVIDER_CAPABILITY_UNSUPPORTED = "E_PROVIDER_CAPABILITY_UNSUPPORTED"
    TOOL_DISABLED = "E_TOOL_DISABLED"
    TOOL_SCHEMA_REJECTED = "E_TOOL_SCHEMA_REJECTED"
    TOOL_TIMEOUT = "E_TOOL_TIMEOUT"
    TOOL_OVER_LIMIT = "E_TOOL_OVER_LIMIT"
    MODEL_GENERATION_FAILED = "E_MODEL_GENERATION_FAILED"
    MODEL_OUTPUT_UNPARSEABLE = "E_MODEL_OUTPUT_UNPARSEABLE"
    MODEL_JSON_SCHEMA_INVALID = "E_MODEL_JSON_SCHEMA_INVALID"
    # safety / memory
    SAFETY_BLOCKED = "E_SAFETY_BLOCKED"
    SAFETY_ESCALATE = "E_SAFETY_ESCALATE"
    MEMORY_UNAVAILABLE = "E_MEMORY_UNAVAILABLE"
    MEMORY_WRITE_DENIED = "E_MEMORY_WRITE_DENIED"
    # terminal unknown
    INTERNAL_UNKNOWN = "E_INTERNAL_UNKNOWN"


# ---------------------------------------------------------------------------
# Registry: ErrorCode -> HTTP status / retryable / terminal / safe message.
# ---------------------------------------------------------------------------


class ErrorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    category: ErrorCategory
    http_status: int = Field(..., ge=400, le=599)
    retryable: bool = False
    terminal: bool = True
    # Safe, user-presentable message (no provider text, no paths, no stack).
    message: str = Field(..., min_length=1)


# code: spec. Registry completeness is enforced by tests (and by lookup()).
ERROR_REGISTRY: dict[ErrorCode, ErrorSpec] = {}


def _register(
    code: ErrorCode,
    category: ErrorCategory,
    http: int,
    retryable: bool,
    terminal: bool,
    message: str,
) -> None:
    if code in ERROR_REGISTRY:
        raise ValueError(f"error code {code.value!r} is registered twice")
    spec = ErrorSpec(
        code=code,
        category=category,
        http_status=http,
        retryable=retryable,
        terminal=terminal,
        message=message,
    )
    ERROR_REGISTRY[code] = spec


_register(
    ErrorCode.VALIDATION_INVALID_INPUT,
    ErrorCategory.VALIDATION,
    400,
    False,
    True,
    "请求参数无效",
)
_register(
    ErrorCode.VALIDATION_UNSUPPORTED_MODALITY,
    ErrorCategory.VALIDATION,
    400,
    False,
    True,
    "不支持的输入类型",
)
_register(
    ErrorCode.VALIDATION_PAYLOAD_TOO_LARGE,
    ErrorCategory.VALIDATION,
    413,
    False,
    True,
    "请求体过大",
)
_register(
    ErrorCode.VALIDATION_UNSUPPORTED_FORMAT,
    ErrorCategory.VALIDATION,
    415,
    False,
    True,
    "不支持的媒体格式",
)
_register(
    ErrorCode.AUTH_MISSING_CREDENTIALS,
    ErrorCategory.AUTHENTICATION,
    401,
    False,
    True,
    "缺少认证凭据",
)
_register(
    ErrorCode.AUTH_INVALID_CREDENTIALS,
    ErrorCategory.AUTHENTICATION,
    401,
    False,
    True,
    "认证失败",
)
_register(
    ErrorCode.AUTH_DEVICE_NOT_REGISTERED,
    ErrorCategory.AUTHENTICATION,
    401,
    False,
    True,
    "设备未注册",
)
_register(
    ErrorCode.AUTHZ_FORBIDDEN,
    ErrorCategory.AUTHORIZATION,
    403,
    False,
    True,
    "无权执行该操作",
)
_register(
    ErrorCode.RATE_LIMIT_EXCEEDED,
    ErrorCategory.RATE_LIMIT,
    429,
    True,
    False,
    "请求过于频繁，请稍后再试",
)
_register(
    ErrorCode.CONFLICT_ACTIVE_RUN,
    ErrorCategory.CONFLICT,
    409,
    False,
    True,
    "当前会话已有进行中的任务",
)
_register(
    ErrorCode.CONFLICT_IDEMPOTENCY,
    ErrorCategory.CONFLICT,
    409,
    False,
    True,
    "请求标识冲突",
)
_register(
    ErrorCode.NOT_FOUND_SESSION,
    ErrorCategory.NOT_FOUND,
    404,
    False,
    True,
    "会话不存在或已结束",
)
_register(
    ErrorCode.NOT_FOUND_RUN,
    ErrorCategory.NOT_FOUND,
    404,
    False,
    True,
    "任务不存在或已过期",
)
_register(
    ErrorCode.NOT_FOUND_KNOWLEDGE,
    ErrorCategory.NOT_FOUND,
    404,
    False,
    True,
    "未找到匹配的已审核资料",
)
_register(
    ErrorCode.NOT_FOUND_AGENT,
    ErrorCategory.NOT_FOUND,
    404,
    False,
    True,
    "Agent 不存在或已停用",
)
_register(
    ErrorCode.TIMEOUT_AGENT, ErrorCategory.TIMEOUT, 504, True, True, "处理超时，请重试"
)
_register(
    ErrorCode.TIMEOUT_PROVIDER, ErrorCategory.TIMEOUT, 504, True, True, "模型服务超时"
)
_register(
    ErrorCode.UNAVAILABLE_OVERLOADED,
    ErrorCategory.UNAVAILABLE,
    503,
    True,
    False,
    "服务繁忙，请稍后再试",
)
_register(
    ErrorCode.UNAVAILABLE_MAINTENANCE,
    ErrorCategory.UNAVAILABLE,
    503,
    False,
    False,
    "服务维护中",
)
_register(
    ErrorCode.PROVIDER_UNREACHABLE,
    ErrorCategory.PROVIDER,
    502,
    True,
    True,
    "模型服务不可达",
)
_register(
    ErrorCode.PROVIDER_CAPABILITY_UNSUPPORTED,
    ErrorCategory.PROVIDER,
    501,
    False,
    True,
    "模型服务不支持该能力",
)
_register(ErrorCode.TOOL_DISABLED, ErrorCategory.TOOL, 403, False, True, "工具未启用")
_register(
    ErrorCode.TOOL_SCHEMA_REJECTED,
    ErrorCategory.TOOL,
    400,
    False,
    True,
    "工具参数校验失败",
)
_register(ErrorCode.TOOL_TIMEOUT, ErrorCategory.TOOL, 504, True, True, "工具调用超时")
_register(
    ErrorCode.TOOL_OVER_LIMIT, ErrorCategory.TOOL, 429, True, True, "工具调用超出限制"
)
_register(
    ErrorCode.MODEL_GENERATION_FAILED,
    ErrorCategory.MODEL,
    502,
    True,
    True,
    "答案生成失败",
)
_register(
    ErrorCode.MODEL_OUTPUT_UNPARSEABLE,
    ErrorCategory.MODEL,
    502,
    True,
    True,
    "模型输出无法解析",
)
_register(
    ErrorCode.MODEL_JSON_SCHEMA_INVALID,
    ErrorCategory.MODEL,
    502,
    True,
    True,
    "模型输出不符合约定结构",
)
_register(
    ErrorCode.SAFETY_BLOCKED,
    ErrorCategory.SAFETY,
    451,
    False,
    True,
    "该问题超出可回答范围",
)
_register(
    ErrorCode.SAFETY_ESCALATE, ErrorCategory.SAFETY, 451, False, False, "建议转人工咨询"
)
_register(
    ErrorCode.MEMORY_UNAVAILABLE,
    ErrorCategory.MEMORY,
    503,
    True,
    False,
    "会话记忆暂不可用",
)
_register(
    ErrorCode.MEMORY_WRITE_DENIED,
    ErrorCategory.MEMORY,
    403,
    False,
    True,
    "不允许写入该记忆",
)
_register(
    ErrorCode.INTERNAL_UNKNOWN, ErrorCategory.INTERNAL, 500, False, True, "服务内部错误"
)


def lookup(code: ErrorCode) -> ErrorSpec:
    """Registry lookup. Missing entries are a programming error and fail loudly."""
    spec = ERROR_REGISTRY.get(code)
    if spec is None:
        raise LookupError(f"error code {code.value!r} is not registered")
    return spec


def http_status_for(code: ErrorCode) -> int:
    return lookup(code).http_status


def is_retryable(code: ErrorCode) -> bool:
    return lookup(code).retryable


def missing_registry_entries() -> list[ErrorCode]:
    """All ErrorCode members that lack a registry entry (used by tests)."""
    return [code for code in ErrorCode if code not in ERROR_REGISTRY]


# ---------------------------------------------------------------------------
# ErrorEnvelope: the only error shape allowed across API boundaries.
# ---------------------------------------------------------------------------


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    # message is optional on input and is always taken from the registry: a
    # caller-supplied message that differs from the registry entry is rejected,
    # so free-form (potentially leaking) text can never ride through an envelope.
    message: str | None = None
    request_id: str = Field(..., min_length=1, max_length=128)
    trace_id: str = Field(..., min_length=1, max_length=128)
    retryable: bool = False
    retry_after_ms: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _registry_message_only(self) -> ErrorEnvelope:
        spec = lookup(self.code)
        if self.message is not None and self.message != spec.message:
            raise ValueError("envelope message must come from the error registry")
        self.message = spec.message
        self.retryable = spec.retryable
        return self

    @classmethod
    def build(
        cls,
        code: ErrorCode,
        request_id: str,
        trace_id: str,
        retry_after_ms: int | None = None,
    ) -> ErrorEnvelope:
        """Build a safe envelope from a registry entry (message comes from the registry)."""
        return cls(
            code=code,
            request_id=request_id,
            trace_id=trace_id,
            retry_after_ms=retry_after_ms,
        )
