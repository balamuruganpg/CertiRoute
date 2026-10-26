"""Minimal request envelope, not yet the complete OpenAI wire contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # Preserve additional parameters for the future adapter and scope guard.
    model_config = ConfigDict(extra="allow")

    model: Literal["certiroute/auto", "certiroute/strong", "certiroute/cheap"] = "certiroute/auto"
    messages: list[dict[str, object]] = Field(min_length=1)
    stream: bool = False
