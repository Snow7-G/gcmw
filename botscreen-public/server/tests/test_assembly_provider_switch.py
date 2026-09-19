"""#53 云模型接入：demo 装配按 GCMW_ACTIVE_PROVIDER 切换 Provider 的测试。"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.orchestration.assembly import build_agent_executor


def make_settings(active_provider: str) -> Settings:
    return Settings(environment="test", active_provider=active_provider)


@pytest.fixture()
def cloud_key():
    env_name = "GCMW_CLOUD_API_KEY"
    previous = os.environ.get(env_name)
    os.environ[env_name] = "test-cloud-key-000001"
    yield env_name
    if previous is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = previous


def test_default_mock_keeps_offline_demo():
    executor = build_agent_executor(repository=object(), settings=make_settings("mock"))
    assert executor is not None
    assert executor.models.active_provider_id == "mock"


def test_cloud_provider_registered_and_active(cloud_key):
    executor = build_agent_executor(
        repository=object(), settings=make_settings("cloud")
    )
    assert executor is not None
    assert executor.models.active_provider_id == "cloud"
    # 两个适配器都在册：mock（五场景自检/离线演示）+ cloud
    assert "mock" in executor.models._adapters
    assert "cloud" in executor.models._adapters
    assert executor.models._adapters["cloud"].model_id == "qwen-plus"


def test_cloud_without_key_fails_closed(cloud_key):
    # 第一道门在 Settings 校验层：active_provider=cloud 但 Key 为空 → 拒绝构造
    os.environ["GCMW_CLOUD_API_KEY"] = ""
    with pytest.raises(RuntimeError, match="GCMW_CLOUD_API_KEY"):
        make_settings("cloud")
    # 第二道门在装配层（防御纵深）：Key 环境变量整个缺失时，绕过校验直接
    # 调用装配也必须拒绝
    os.environ.pop("GCMW_CLOUD_API_KEY", None)
    settings = Settings.model_construct(environment="test", active_provider="cloud")
    with pytest.raises(RuntimeError, match="GCMW_ACTIVE_PROVIDER=cloud"):
        build_agent_executor(repository=object(), settings=settings)


def test_local_provider_stays_mock_in_demo_slice():
    # "local" 不在本切片范围：demo 装配仍用 mock（fail-safe 默认）
    executor = build_agent_executor(
        repository=object(), settings=make_settings("local")
    )
    assert executor.models.active_provider_id == "mock"
