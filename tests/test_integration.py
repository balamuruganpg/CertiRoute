"""Synthetic contract fixtures only; not evidence for evaluation-plan claims."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr

from certiroute.calibration.artifact import (
    CalibrationRow,
    DataProvenance,
    build_artifact,
    content_hash,
    load_artifact,
    model_identity,
    save_artifact,
)
from certiroute.config import Settings, load_config
from certiroute.gateway.app import create_app

AUTH = {"Authorization": "Bearer gateway-test-key"}


def payload(text="Hi", **kwargs):
    return {"model": "certiroute/auto", "messages": [{"role": "user", "content": text}], **kwargs}


def completion(model):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"from {model}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config_path = tmp_path / "gateway.yaml"
    artifact_path = tmp_path / "calibration.json"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "cheap": {
                        "base_url": "https://cheap.test/v1",
                        "model": "cheap-v1",
                        "api_key_env": "TEST_CHEAP_KEY",
                    },
                    "strong": {
                        "base_url": "https://strong.test/v1",
                        "model": "strong-v1",
                        "api_key_env": "TEST_STRONG_KEY",
                    },
                },
                "calibration": {
                    "artifact": str(artifact_path),
                    "allowed_epsilons": [0.01, 0.02],
                    "default_epsilon": 0.02,
                },
                "router": {"kind": "length_v1"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    # 100 easy short prompts and 20 hard prompts with tied scores of exactly 0.5.
    rows = [
        CalibrationRow(request=payload(f"short-{i}"), q_strong=1, q_cheap=1) for i in range(100)
    ]
    rows += [
        CalibrationRow(request=payload(f"hard-{i:03d}".ljust(1000, "x")), q_strong=1, q_cheap=0)
        for i in range(20)
    ]
    provenance = DataProvenance(
        sources=["synthetic-test-only"], sha256="a" * 64, collected_at=datetime.now(UTC)
    )
    artifact = build_artifact(rows, config, provenance)
    assert artifact.threshold(0.02) == 0.5
    save_artifact(artifact, artifact_path)
    settings = Settings(
        _env_file=None, config_file=config_path, api_key=SecretStr("gateway-test-key")
    )
    monkeypatch.setenv("TEST_CHEAP_KEY", "cheap-secret")
    monkeypatch.setenv("TEST_STRONG_KEY", "strong-secret")
    return settings, config, artifact, rows


def test_calibration_to_gateway_happy_path_both_tiers(setup):
    settings, config, artifact, _ = setup
    calls = []

    def upstream(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        assert request.url.path == "/v1/chat/completions"
        tier = "cheap" if body["model"] == "cheap-v1" else "strong"
        assert request.url.host == f"{tier}.test"
        assert request.headers["Authorization"] == f"Bearer {tier}-secret"
        assert body["temperature"] == 0 and body["max_tokens"] == 256 and body["top_p"] == 1
        return httpx.Response(200, json=completion(body["model"]))

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        assert client.get("/readyz").status_code == 200
        for text, tier in (("Hi", "cheap"), ("x" * 2000, "strong"), ("x" * 1000, "strong")):
            response = client.post("/v1/chat/completions", json=payload(text), headers=AUTH)
            assert response.status_code == 200
            assert response.json() == completion(f"{tier}-v1")
            assert response.headers["x-certiroute-tier"] == tier
            assert response.headers["x-certiroute-calibration-id"] == artifact.calibration_id
            assert response.headers["x-certiroute-scope"] == "in"
            assert response.headers["x-certiroute-degraded"] == "false"
    assert calls == ["cheap-v1", "strong-v1", "strong-v1"]


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 1},
        {"max_tokens": 512},
        {"tools": []},
        {"n": 2},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:test"}}],
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ]
        },
    ],
)
def test_out_of_scope_goes_strong(setup, changes):
    settings, _, _, _ = setup

    def upstream(request):
        body = json.loads(request.content)
        assert body["model"] == "strong-v1"
        for key, value in changes.items():
            assert body[key] == value
        return httpx.Response(200, json=completion(body["model"]))

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post("/v1/chat/completions", json=payload(**changes), headers=AUTH)
        assert response.status_code == 200
        assert response.headers["x-certiroute-scope"] == "out"
        assert response.headers["x-certiroute-calibration-id"] == "none"


@pytest.mark.parametrize(
    "failure", ["missing", "corrupt", "tampered", "expired", "model", "router", "scope", "metric"]
)
def test_invalid_artifact_is_unready_and_strong_only(setup, failure):
    settings, config, artifact, _ = setup
    path = config.calibration.artifact
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_text("not json", encoding="utf-8")
    else:
        fields = artifact.model_dump(mode="json")
        if failure == "tampered":
            fields["lambda_by_epsilon"]["0.02"] = 0
        elif failure == "expired":
            fields["created_at"] = (
                (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
            )
            fields["expires_at"] = (
                (datetime.now(UTC) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
            )
        elif failure == "model":
            fields["models"]["cheap"] = "different-model"
        elif failure == "router":
            fields["router"]["sha256"] = "b" * 64
        elif failure == "scope":
            fields["scope"]["max_tokens"] = 512
        else:
            fields["quality_metric"]["version"] = "2"
        if failure != "tampered":
            fields["calibration_id"] = "cal_" + content_hash(
                {k: v for k, v in fields.items() if k != "calibration_id"}
            )
        path.write_text(json.dumps(fields), encoding="utf-8")

    def upstream(request):
        assert json.loads(request.content)["model"] == "strong-v1"
        return httpx.Response(200, json=completion("strong-v1"))

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        assert client.get("/readyz").status_code == 503
        response = client.post("/v1/chat/completions", json=payload(), headers=AUTH)
        assert response.status_code == 200
        assert response.headers["x-certiroute-calibration-id"] == "none"


def test_cheap_error_falls_back_to_strong(setup):
    settings, _, _, _ = setup
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return (
            httpx.Response(429)
            if model == "cheap-v1"
            else httpx.Response(200, json=completion(model))
        )

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post("/v1/chat/completions", json=payload(), headers=AUTH)
        assert response.status_code == 200
        assert response.headers["x-certiroute-tier"] == "strong"
        assert response.headers["x-certiroute-reason"] == "cheap_upstream_failure"
    assert calls == ["cheap-v1", "strong-v1"]


@pytest.mark.parametrize("allow", [False, True])
def test_strong_never_downgrades_without_opt_in(setup, allow):
    settings, _, _, _ = setup
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return (
            httpx.Response(503, text="do not leak secrets")
            if model == "strong-v1"
            else httpx.Response(200, json=completion(model))
        )

    headers = {**AUTH, "x-certiroute-allow-downgrade": str(allow).lower()}
    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post("/v1/chat/completions", json=payload("x" * 2000), headers=headers)
        assert response.status_code == (200 if allow else 502)
        assert "do not leak secrets" not in response.text
        if allow:
            assert response.headers["x-certiroute-degraded"] == "true"
            assert response.headers["x-certiroute-calibration-id"] == "none"
    assert calls == (["strong-v1", "cheap-v1"] if allow else ["strong-v1"])


class Chunks(httpx.AsyncByteStream):
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.closed = False

    async def __aiter__(self):
        for index, part in enumerate(
            [b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n', b"data: [DONE]\n\n"]
        ):
            if self.fail_at == index:
                raise httpx.ReadError("disconnect")
            yield part

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("fail_at", [None, 0, 1])
def test_streaming_passthrough_and_failure_boundary(setup, fail_at):
    settings, _, _, _ = setup
    calls, streams = [], []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        stream = Chunks(fail_at if model == "cheap-v1" else None)
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post("/v1/chat/completions", json=payload(stream=True), headers=AUTH)
        assert response.status_code == 200
        if fail_at == 1:
            assert "event: error" in response.text
            assert "[DONE]" not in response.text
        else:
            assert (
                response.content
                == b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
            )
        assert response.headers["x-certiroute-tier"] == ("strong" if fail_at == 0 else "cheap")
    assert calls == (["cheap-v1", "strong-v1"] if fail_at == 0 else ["cheap-v1"])
    assert all(stream.closed for stream in streams)


def test_offline_cli_builds_loadable_immutable_artifact(setup, tmp_path):
    settings, config, artifact, rows = setup
    dataset = {
        "models": model_identity(config),
        "quality_metric": config.quality_metric.model_dump(),
        "scope": config.scope.model_dump(),
        "sources": ["synthetic-test-only"],
        "collected_at": datetime.now(UTC).isoformat(),
        "rows": [row.model_dump() for row in rows],
    }
    source, output = tmp_path / "paired.json", tmp_path / "cli-artifact.json"
    source.write_text(json.dumps(dataset), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "eval.calibrate",
        "--config",
        str(settings.config_file),
        "--input",
        str(source),
        "--out",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    generated = load_artifact(output, config)
    assert generated.lambda_by_epsilon == artifact.lambda_by_epsilon
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True, timeout=30).returncode == 2
    assert output.read_bytes() == before


def test_artifact_rejects_duplicate_calibration_prompts(setup):
    _, config, artifact, rows = setup
    with pytest.raises(ValueError, match="Duplicate"):
        build_artifact([rows[0], rows[0]], config, artifact.data)


@pytest.mark.parametrize("failure", ["timeout", "bad_json", "bad_shape"])
def test_provider_failures_are_normalized_and_fall_upward(setup, failure):
    settings, _, _, _ = setup
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "strong-v1":
            return httpx.Response(200, json=completion(model))
        if failure == "timeout":
            raise httpx.ReadTimeout("sensitive provider details")
        if failure == "bad_json":
            return httpx.Response(200, text="invalid json")
        return httpx.Response(200, json={"error": "unexpected response"})

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post("/v1/chat/completions", json=payload(), headers=AUTH)
        assert response.status_code == 200
        assert response.headers["x-certiroute-tier"] == "strong"
        assert "sensitive" not in response.text
    assert calls == ["cheap-v1", "strong-v1"]


def test_mode_override_bypass_and_conflict(setup):
    settings, _, _, _ = setup
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200, json=completion(model))

    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload("x" * 2000),
            headers={**AUTH, "x-certiroute-mode": "cheap"},
        )
        assert response.status_code == 200
        assert response.headers["x-certiroute-calibration-id"] == "none"
        assert response.headers["x-certiroute-degraded"] == "true"
        response = client.post(
            "/v1/chat/completions",
            json=payload(model="certiroute/strong"),
            headers={**AUTH, "x-certiroute-mode": "cheap"},
        )
        assert response.status_code == 400
    assert calls == ["cheap-v1"]
