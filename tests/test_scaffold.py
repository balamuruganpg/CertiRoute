"""Startup and honest unavailable-state checks; no upstream or model credentials."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from certiroute.config import CalibrationConfig, Settings
from certiroute.gateway.app import create_app


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
