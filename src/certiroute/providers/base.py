"""Provider contracts and errors safe to return without upstream secrets."""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ProviderStream:
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


ChatResponse = dict[str, object] | ProviderStream


class Provider(Protocol):
    async def chat(self, request: dict[str, object]) -> ChatResponse: ...
