"""Startup and honest unavailable-state checks; no upstream or model credentials."""

import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from certiroute.config import INSECURE_LOCAL_API_KEY, CalibrationConfig, Settings
from certiroute.gateway.app import create_app


def _open(settings: Settings) -> TestClient:
    def upstream(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "from strong"}}]})

    return TestClient(create_app(settings, transport=httpx.MockTransport(upstream)))


@pytest.fixture
def client():
    settings = Settings(
        _env_file=None,
        config_file=Path("configs/gateway.yaml"),
        api_key=SecretStr("test-key"),
    )

    def upstream(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "from strong"}}]})

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as test_client:
        yield test_client


def test_alive_but_not_calibrated(client):
    assert client.get("/healthz").status_code == 200
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_models_require_correct_key(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "certiroute/auto"


def test_uncalibrated_chat_forwards_to_strong(client):
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    assert client.post("/v1/chat/completions", json=payload).status_code == 401
    response = client.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from strong"
    assert response.headers["x-certiroute-tier"] == "strong"
    assert response.headers["x-certiroute-calibration-id"] == "none"


@pytest.mark.parametrize("epsilon", ["0.03", "nan", "inf", "invalid"])
def test_chat_rejects_invalid_epsilon(client, epsilon):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
        headers={"Authorization": "Bearer test-key", "x-certiroute-epsilon": epsilon},
    )
    assert response.status_code == 400


def test_config_rejects_default_outside_grid():
    with pytest.raises(ValidationError):
        CalibrationConfig(
            artifact=Path("missing.json"), allowed_epsilons=[0.01], default_epsilon=0.02
        )


def test_missing_api_key_rejects_published_default(monkeypatch, caplog):
    monkeypatch.delenv("CERTIROUTE_API_KEY", raising=False)
    monkeypatch.delenv("CERTIROUTE_ALLOW_INSECURE_LOCAL_KEY", raising=False)
    caplog.set_level(logging.ERROR, logger="certiroute.gateway.app")
    settings = Settings(_env_file=None)
    assert settings.api_key is None
    assert settings.resolved_api_key() is None
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    published = {"Authorization": f"Bearer {INSECURE_LOCAL_API_KEY}"}
    with _open(settings) as client:
        assert client.get("/healthz").status_code == 200
        models = client.get("/v1/models", headers=published)
        chat = client.post("/v1/chat/completions", json=payload, headers=published)
    for response in (models, chat):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "api_key_not_configured"
        assert "CERTIROUTE_API_KEY" in response.json()["error"]["message"]
    assert any(
        record.levelno >= logging.ERROR and "CERTIROUTE_API_KEY" in record.message
        for record in caplog.records
    )


def test_blank_api_key_and_disabled_opt_in_do_not_authenticate(monkeypatch):
    monkeypatch.setenv("CERTIROUTE_API_KEY", "   ")
    monkeypatch.setenv("CERTIROUTE_ALLOW_INSECURE_LOCAL_KEY", "0")
    settings = Settings(_env_file=None)
    assert settings.api_key is None
    assert settings.allow_insecure_local_key is False
    assert settings.resolved_api_key() is None


def test_insecure_local_opt_in_accepts_published_key_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("CERTIROUTE_API_KEY", raising=False)
    monkeypatch.setenv("CERTIROUTE_ALLOW_INSECURE_LOCAL_KEY", "1")
    caplog.set_level(logging.WARNING, logger="certiroute.gateway.app")
    settings = Settings(_env_file=None)
    assert settings.api_key is None
    assert settings.resolved_api_key().get_secret_value() == INSECURE_LOCAL_API_KEY
    with _open(settings) as client:
        accepted = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {INSECURE_LOCAL_API_KEY}"}
        )
        rejected = client.get("/v1/models", headers={"Authorization": "Bearer other-key"})
    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "invalid_api_key"
    assert any(
        record.levelno == logging.WARNING and "INSECURE LOCAL KEY" in record.message
        for record in caplog.records
    )


def test_configured_api_key_overrides_insecure_opt_in(monkeypatch, caplog):
    monkeypatch.setenv("CERTIROUTE_API_KEY", "env-configured-key")
    monkeypatch.setenv("CERTIROUTE_ALLOW_INSECURE_LOCAL_KEY", "1")
    caplog.set_level(logging.WARNING, logger="certiroute.gateway.app")
    settings = Settings(_env_file=None)
    with _open(settings) as client:
        accepted = client.get("/v1/models", headers={"Authorization": "Bearer env-configured-key"})
        rejected = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {INSECURE_LOCAL_API_KEY}"}
        )
    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "invalid_api_key"
    assert not any("INSECURE LOCAL KEY" in record.message for record in caplog.records)
