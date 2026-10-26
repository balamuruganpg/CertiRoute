"""Known settings, single-turn text and bounded prompt length."""

from certiroute.config import ScopeConfig
from certiroute.router.features import extract_features


def with_generation_defaults(request: dict[str, object], scope: ScopeConfig) -> dict[str, object]:
    result = dict(request)
    for key in ("temperature", "max_tokens", "top_p"):
        result.setdefault(key, getattr(scope, key))
    if scope.stop is not None:
        result.setdefault("stop", scope.stop)
    return result


def in_calibrated_scope(request: dict[str, object], scope: ScopeConfig) -> bool:
    # Unknown generation options can change quality: only forward them to strong.
    allowed = {"model", "messages", "stream", "temperature", "max_tokens", "top_p", "n"}
    if scope.stop is not None:
        allowed.add("stop")
        if request.get("stop") != scope.stop:
            return False
    if set(request) - allowed or request.get("n", 1) != 1:
        return False
    if any(
        request.get(key) != getattr(scope, key) for key in ("temperature", "max_tokens", "top_p")
    ):
        return False
    messages = request.get("messages")
    if not isinstance(messages, list):
        return False
    if any(not isinstance(m, dict) or set(m) != {"role", "content"} for m in messages):
        return False
    roles = [m["role"] for m in messages]
    if roles not in (["user"], ["system", "user"]):
        return False
    try:
        return extract_features(request)[0] <= scope.max_chars
    except ValueError:
        return False
