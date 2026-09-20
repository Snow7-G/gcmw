"""Shared API test harness (not a test module).

Every test gets its OWN application: the run service lives in the FastAPI
lifespan scope, so each ``TestClient`` context builds a fresh service on a fresh
event loop — no asyncio primitive is ever shared between loops, and no test can
leak session/idempotency state into another.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.auth import DevicePrincipal
from app.config import Settings
from app.contracts.errors import ErrorEnvelope
from app.main import create_app
from app.storage.run_repository import (
    MemoryRunRepository,
    RunRepositoryError,
    RunRepositoryFault,
)

PRINCIPAL = DevicePrincipal(tenant_id="t1", device_id="d1")
OTHER_DEVICE = DevicePrincipal(tenant_id="t1", device_id="OTHER")
OTHER_TENANT = DevicePrincipal(tenant_id="t2", device_id="d1")

#: deterministic low-entropy test credentials (never real secrets)
PRIMARY_TOKEN = "dev-primary-000000000"
OTHER_DEVICE_TOKEN = "dev-other-device-00000000"
OTHER_TENANT_TOKEN = "dev-other-tenant-00000000"

_HARNESS_PRINCIPALS = {
    PRIMARY_TOKEN: PRINCIPAL,
    OTHER_DEVICE_TOKEN: OTHER_DEVICE,
    OTHER_TENANT_TOKEN: OTHER_TENANT,
}


class HarnessCredentialStore:
    """Test double for the credential store: token -> fixed principal.

    The entry guard authenticates BEFORE routing, so tests authenticate the way
    production does — with an ``Authorization: Bearer`` header — instead of
    overriding a dependency that the guard has already satisfied.
    """

    def resolve(self, presented: str):
        principal = _HARNESS_PRINCIPALS.get(presented)
        if principal is None:
            return None
        return SimpleNamespace(
            tenant_id=principal.tenant_id, device_id=principal.device_id
        )

    @property
    def configured(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(_HARNESS_PRINCIPALS)


class FakeClock:
    """Injectable clock for session TTL tests."""

    def __init__(self) -> None:
        self.value = datetime.now(timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value = self.value + timedelta(seconds=seconds)


@dataclass
class Harness:
    client: TestClient
    app: FastAPI
    clock: FakeClock
    repository: Any

    @property
    def service(self) -> Any:
        return self.app.state.agent_service

    def env(self, response) -> ErrorEnvelope:
        return ErrorEnvelope.model_validate(response.json())

    @staticmethod
    def auth(token: str) -> dict[str, str]:
        """Header for one of the harness identities."""
        return {"Authorization": f"Bearer {token}"}

    @contextmanager
    def as_token(self, token: str) -> Iterator[None]:
        """Act as another harness identity for the duration of the block.

        The client's default Authorization header is swapped (a per-request
        header does not reliably override it in this TestClient), so the entry
        guard really authenticates that credential.
        """
        headers = self.client.headers
        previous = headers.get("Authorization")
        headers["Authorization"] = f"Bearer {token}"
        try:
            yield
        finally:
            if previous is None:
                headers.pop("Authorization", None)
            else:
                headers["Authorization"] = previous


@contextmanager
def running_app(
    repository: Any | None = None,
    environment: str = "test",
    *,
    credentials: list[dict] | None = None,
    settings_kwargs: dict | None = None,
    default_credential: str | None = PRIMARY_TOKEN,
    agent_executor: bool = False,
    app_kwargs: dict[str, Any] | None = None,
) -> Iterator[Harness]:
    """Run the application under test.

    Authentication is NEVER bypassed: the entry guard resolves the credential
    before routing, so the client carries a real ``Authorization`` header (the
    harness identities by default, ``default_credential=None`` for anonymous
    tests) and ``credentials`` seeds the store from the environment exactly as
    an operator would.

    ``app_kwargs`` is forwarded to :func:`app.main.create_app` (e.g. the
    session-TTL sweeper interval for P1-2 tests).
    """
    repository = repository if repository is not None else MemoryRunRepository()
    settings = Settings(environment=environment, **(settings_kwargs or {}))
    env_name = settings.auth_credentials_env
    previous = os.environ.get(env_name)
    if credentials is not None:
        os.environ[env_name] = json.dumps(credentials)
    try:
        app = create_app(
            settings=settings,
            repository_factory=lambda _settings: repository,
            agent_executor=agent_executor,
            **(app_kwargs or {}),
        )
        clock = FakeClock()
        headers = (
            {"Authorization": f"Bearer {default_credential}"}
            if default_credential
            else None
        )
        with TestClient(app, raise_server_exceptions=False, headers=headers) as client:
            # install the clock once the lifespan has built the service
            if app.state.agent_service is not None:
                app.state.agent_service._clock = clock
            if credentials is None:
                # the guard authenticates before routing: give it the harness
                # identities unless the test brought its own credential store
                app.state.credentials = HarnessCredentialStore()
            yield Harness(client=client, app=app, clock=clock, repository=repository)
        app.dependency_overrides.clear()
    finally:
        if credentials is not None:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous


def new_session(harness: Harness, channel: str = "text") -> dict:
    res = harness.client.post("/api/v1/sessions", json={"channel": channel})
    assert res.status_code == 201, res.text
    return res.json()


def new_run(
    harness: Harness,
    session_id: str,
    text: str = "hi",
    key: str = "k",
    headers: dict[str, str] | None = None,
) -> dict:
    res = harness.client.post(
        "/api/v1/agent/runs",
        json={
            "session_id": session_id,
            "input": {"type": "text", "text": text},
            "idempotency_key": key,
        },
        headers=headers,
    )
    assert res.status_code == 200, res.text
    return res.json()


class ForcedStateRepository(MemoryRunRepository):
    """Memory repository with a test-controlled durable state override.

    Simulates an EXTERNAL writer (another worker, or the future ManagerAgent
    advancing a run) without touching the event loop from the test thread: the
    map is plain data, read inside the loop on the next ``state()`` call.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forced: dict[str, Any] = {}

    async def state(self, identity):
        forced = self.forced.get(identity.run_id)
        if forced is not None:
            return forced
        return await super().state(identity)


class VanishingRepository(MemoryRunRepository):
    """Memory repository where selected runs expire (NOT_FOUND) on read.

    Models the repository TTL evicting a run while the admission layer still
    holds its bookkeeping entry.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vanished: set[str] = set()

    async def state(self, identity):
        if identity.run_id in self.vanished:
            raise RunRepositoryError(RunRepositoryFault.NOT_FOUND, "expired")
        return await super().state(identity)
