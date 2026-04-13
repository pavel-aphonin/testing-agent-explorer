"""Shared pytest fixtures for the explorer test suite.

Most tests in this package are in-memory unit tests that don't need any
of this machinery. The fixtures below exist for
``test_integration_mesh.py``, which talks to a live backend and needs
a handful of things resolved from the environment:

    - where the backend lives (TA_BACKEND_URL)
    - what credentials to use for the user-facing JWT flow
    - what shared-secret to use for the worker's /api/internal/* auth
    - whether the backend is actually up — if not, we skip cleanly
      instead of failing, so ``pytest tests/`` stays green on a
      developer machine that isn't running docker compose at the moment

All defaults match ``testing-agent-infra/.env.example``, so the common
case is: dev runs ``docker compose up -d`` from infra/, then runs
pytest in explorer/ without setting any env vars.
"""

from __future__ import annotations

import os

import httpx
import pytest


DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "change_me_strong_password"
DEFAULT_WORKER_TOKEN = "change_me_worker_token_long_random_string"


@pytest.fixture(scope="session")
def backend_url() -> str:
    return os.environ.get("TA_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


@pytest.fixture(scope="session")
def admin_credentials() -> tuple[str, str]:
    email = os.environ.get("TA_ADMIN_EMAIL", DEFAULT_ADMIN_EMAIL)
    password = os.environ.get("TA_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
    return email, password


@pytest.fixture(scope="session")
def worker_token() -> str:
    return os.environ.get("TA_WORKER_TOKEN", DEFAULT_WORKER_TOKEN)


@pytest.fixture(scope="session")
def backend_available(backend_url: str) -> None:
    """Skip the calling test(s) if the backend is not reachable.

    One-shot GET /health with a 1-second timeout. We deliberately use
    sync httpx here — this fixture runs before any event loop is set
    up and we don't want to pull pytest-asyncio into a session-scope
    concern.
    """
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{backend_url}/health")
    except httpx.HTTPError as exc:
        pytest.skip(f"backend not reachable at {backend_url}: {exc}")

    if resp.status_code != 200:
        pytest.skip(
            f"backend at {backend_url}/health returned {resp.status_code}"
        )


@pytest.fixture
async def admin_jwt(
    backend_url: str,
    admin_credentials: tuple[str, str],
    backend_available: None,
) -> str:
    """Login as the seed admin and return a fresh access token.

    Uses the fastapi-users /auth/jwt/login endpoint, which accepts
    form-encoded ``username=<email>&password=<pwd>`` and returns a JWT
    in the ``access_token`` field. This is exactly what the frontend
    does on login.
    """
    email, password = admin_credentials
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            f"{backend_url}/auth/jwt/login",
            data={"username": email, "password": password},
        )
    if resp.status_code != 200:
        pytest.skip(
            f"admin login failed with {resp.status_code}: {resp.text}"
        )
    token = resp.json().get("access_token")
    if not token:
        pytest.skip("admin login did not return access_token")
    return token
