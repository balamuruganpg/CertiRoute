"""Stable identities and prompt fingerprints shared by training and calibration."""

import hashlib
import json
import re

from certiroute.config import GatewayConfig


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def model_identity(config: GatewayConfig) -> dict[str, str]:
    return {
        tier: f"{str(provider.base_url).rstrip('/')}::{provider.model}"
        for tier, provider in (("cheap", config.models.cheap), ("strong", config.models.strong))
    }


def prompt_signature(request: dict[str, object], prefix: str = "") -> frozenset[str]:
    """Hashed token trigrams for lexical (not semantic) near-duplicate checking."""
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected messages.")
    words = []
    stripped = not prefix
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Expected text messages.")
        content = message["content"]
        if prefix and message.get("role") == "user":
            if not content.startswith(prefix):
                raise ValueError("Prompt does not match the bound signature prefix.")
            content = content[len(prefix) :]
            stripped = True
        words.extend([str(message.get("role")), *re.findall(r"\w+|[+*/=-]", content.lower())])
    if not stripped:
        raise ValueError("Missing user question for the bound signature prefix.")
    if len(words) < 3:
        return frozenset([content_hash(words)])
    return frozenset(content_hash(words[i : i + 3]) for i in range(len(words) - 2))


def overlaps(left: frozenset[str], right: frozenset[str]) -> bool:
    return bool(left and right) and len(left & right) / len(left | right) >= 0.9
