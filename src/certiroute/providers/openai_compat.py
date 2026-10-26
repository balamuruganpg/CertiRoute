"""Pooled OpenAI-compatible JSON and SSE forwarding with bounded timeouts."""

import asyncio
from collections.abc import AsyncIterator
from json import JSONDecodeError

import httpx

from certiroute.config import ProviderConfig
from certiroute.providers.base import ChatResponse, ProviderError, ProviderStream


class OpenAICompatProvider:
    def __init__(
        self, config: ProviderConfig, client: httpx.AsyncClient, api_key: str | None = None
    ) -> None:
        self.config = config
        self.client = client
        self.api_key = api_key

    async def chat(self, request: dict[str, object]) -> ChatResponse:
        payload = {**request, "model": self.config.model}
        headers = {"Accept": "text/event-stream" if request.get("stream") else "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        upstream_request = self.client.build_request(
            "POST",
            str(self.config.base_url).rstrip("/") + "/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(self.config.read_timeout, connect=self.config.connect_timeout),
        )
        deadline = asyncio.get_running_loop().time() + self.config.total_timeout
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout_at(deadline):
                response = await self.client.send(upstream_request, stream=True)
                if not 200 <= response.status_code < 300:
                    status = response.status_code
                    raise ProviderError(
                        f"Upstream returned HTTP {status}.",
                        status_code=status if status in (400, 422, 429) else 502,
                    )
                if request.get("stream"):
                    if not response.headers.get("content-type", "").startswith("text/event-stream"):
                        raise ProviderError("Upstream did not return an SSE stream.")
                    iterator = response.aiter_bytes()
                    # Fail upward before committing downstream headers/bytes.
                    first = await anext(iterator, None)
                    if first is None:
                        raise ProviderError("Upstream returned an empty stream.")
                    return ProviderStream(
                        self._stream(response, iterator, first, deadline), response.aclose
                    )
                await response.aread()
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                    raise ProviderError("Upstream returned an invalid chat completion.")
                await response.aclose()
                return data
        except BaseException as exc:
            if response is not None:
                await response.aclose()
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                raise ProviderError("Upstream timed out.", 504) from exc
            if isinstance(exc, (httpx.HTTPError, JSONDecodeError, UnicodeDecodeError)):
                raise ProviderError("Upstream connection or response failed.") from exc
            raise

    async def _stream(
        self,
        response: httpx.Response,
        iterator: AsyncIterator[bytes],
        first: bytes,
        deadline: float,
    ) -> AsyncIterator[bytes]:
        try:
            async with asyncio.timeout_at(deadline):
                yield first
                async for chunk in iterator:
                    yield chunk
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ProviderError("Upstream stream timed out.", 504) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Upstream stream disconnected.") from exc
        finally:
            await response.aclose()
