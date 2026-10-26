"""Minimal calibrated gateway with pooled provider forwarding and strong fallback."""

import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Annotated
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask

from certiroute.calibration.artifact import CalibrationArtifact, load_artifact
from certiroute.config import Settings, load_config
from certiroute.gateway.schemas import ChatRequest
from certiroute.providers.base import ProviderError, ProviderStream
from certiroute.providers.openai_compat import OpenAICompatProvider
from certiroute.router.policy import RoutingDecision, route
from certiroute.router.predictor import Predictor, load_predictor
from certiroute.router.scope import with_generation_defaults

bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def error_response(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "param": None, "code": code}},
    )


def routing_headers(decision: RoutingDecision, request_id: str) -> dict[str, str]:
    return {
        "x-certiroute-request-id": request_id,
        "x-certiroute-tier": decision.tier,
        "x-certiroute-score": str(decision.score) if decision.score is not None else "unavailable",
        "x-certiroute-epsilon": str(decision.epsilon),
        "x-certiroute-calibration-id": decision.calibration_id or "none",
        "x-certiroute-degraded": str(decision.degraded).lower(),
        "x-certiroute-scope": decision.scope,
        "x-certiroute-reason": decision.reason,
    }


def create_app(
    settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    settings = settings if settings is not None else Settings()
    config = load_config(settings.config_file)
    artifact: CalibrationArtifact | None = None
    predictor: Predictor | None = None
    providers: dict[str, OpenAICompatProvider] = {}

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal artifact, predictor
        try:
            predictor = load_predictor(config)
            artifact = load_artifact(config.calibration.artifact, config, predictor)
        except (OSError, ValueError):
            # Do not log validation inputs (they may contain sensitive metadata).
            logger.warning("Calibration unavailable or invalid; serving strong-only.")
            artifact = None
        env_values = dotenv_values(".env")
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            for tier in ("cheap", "strong"):
                provider_config = getattr(config.models, tier)
                key = os.environ.get(
                    provider_config.api_key_env, env_values.get(provider_config.api_key_env)
                )
                providers[tier] = OpenAICompatProvider(provider_config, client, key)
            yield
        providers.clear()

    application = FastAPI(title="CertiRoute", version="0.0.1", lifespan=lifespan)

    def authenticated(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> bool:
        return credentials is not None and secrets.compare_digest(
            credentials.credentials.encode(), settings.api_key.get_secret_value().encode()
        )

    def unauthorized() -> JSONResponse:
        response = error_response(401, "A valid gateway bearer key is required.", "invalid_api_key")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @application.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def ready() -> JSONResponse:
        try:
            if artifact is None:
                raise ValueError("No artifact")
            artifact.validate_runtime(config, predictor)
        except ValueError:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "reason": "No valid calibration; serving strong-only."},
            )
        return JSONResponse(content={"ready": True, "calibration_id": artifact.calibration_id})

    @application.get("/v1/models")
    async def models(authorized: Annotated[bool, Depends(authenticated)]) -> JSONResponse:
        if not authorized:
            return unauthorized()
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": f"certiroute/{tier}",
                        "object": "model",
                        "created": 0,
                        "owned_by": "certiroute",
                    }
                    for tier in ("auto", "strong", "cheap")
                ],
            }
        )

    @application.post("/v1/chat/completions", response_model=None)
    async def chat(
        request: ChatRequest,
        authorized: Annotated[bool, Depends(authenticated)],
        epsilon_header: Annotated[str | None, Header(alias="x-certiroute-epsilon")] = None,
        mode: Annotated[str | None, Header(alias="x-certiroute-mode")] = None,
        allow_downgrade: Annotated[str | None, Header(alias="x-certiroute-allow-downgrade")] = None,
    ) -> JSONResponse | StreamingResponse:
        if not authorized:
            return unauthorized()
        try:
            epsilon = (
                config.calibration.default_epsilon
                if epsilon_header is None
                else float(epsilon_header)
            )
        except ValueError:
            return error_response(
                400, "Epsilon must be a number on the allowed grid.", "invalid_epsilon"
            )
        if epsilon not in config.calibration.allowed_epsilons:
            return error_response(400, "Epsilon is not on the allowed grid.", "invalid_epsilon")
        alias = request.model.removeprefix("certiroute/")
        if mode is not None and (
            mode not in ("auto", "cheap", "strong") or (alias != "auto" and mode != alias)
        ):
            return error_response(400, "Invalid or conflicting routing mode.", "invalid_mode")
        if allow_downgrade not in (None, "true", "false"):
            return error_response(
                400, "Downgrade header must be true or false.", "invalid_downgrade"
            )
        selected_mode = mode or alias
        payload = with_generation_defaults(request.model_dump(), config.scope)
        if selected_mode == "cheap":
            decision = RoutingDecision("cheap", epsilon, "explicit_bypass", "bypass", degraded=True)
        elif selected_mode == "strong":
            decision = RoutingDecision("strong", epsilon, "explicit_bypass", "bypass")
        else:
            decision = route(
                payload, epsilon=epsilon, config=config, artifact=artifact, predictor=predictor
            )
        request_id = str(uuid4())
        try:
            try:
                result = await providers[decision.tier].chat(payload)
            except ProviderError:
                if decision.tier == "cheap":
                    decision = replace(
                        decision, tier="strong", reason="cheap_upstream_failure", degraded=False
                    )
                elif allow_downgrade == "true":
                    decision = RoutingDecision(
                        "cheap", epsilon, "explicit_downgrade", "bypass", degraded=True
                    )
                else:
                    raise
                result = await providers[decision.tier].chat(payload)
        except ProviderError as exc:
            response = error_response(exc.status_code, str(exc), "upstream_error")
            response.headers.update(routing_headers(decision, request_id))
            return response

        headers = routing_headers(decision, request_id)
        if isinstance(result, ProviderStream):
            stream = result

            async def stream_body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in stream.chunks:
                        yield chunk
                except ProviderError as exc:
                    # Once any bytes are committed, never splice in another model.
                    logger.warning("Upstream stream failed for request %s", request_id)
                    error = json.dumps({"error": {"message": str(exc), "type": "upstream_error"}})
                    yield f"\n\nevent: error\ndata: {error}\n\n".encode()
                finally:
                    await stream.close()

            return StreamingResponse(
                stream_body(),
                media_type="text/event-stream",
                headers=headers,
                background=BackgroundTask(stream.close),
            )
        return JSONResponse(content=result, headers=headers)

    return application


app = create_app()
