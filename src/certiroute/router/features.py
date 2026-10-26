"""Small local prompt features; never uses outputs, labels or provider calls."""

FEATURE_NAMES = (
    "characters",
    "words",
    "digit_fraction",
    "plus_signs",
    "multiply_signs",
    "code_markers",
    "lines",
)


def extract_features(request: dict[str, object]) -> list[float]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty messages list.")
    texts = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Features require text-only messages.")
        texts.append(message["content"])
    length = sum(map(len, texts))
    text = "\n".join(texts)
    return [
        float(length),
        float(len(text.split())),
        sum(c.isdigit() for c in text) / max(length, 1),
        float(text.count("+")),
        float(text.count("*")),
        float(sum(text.count(marker) for marker in (chr(96) * 3, "def ", "class ", "import "))),
        float(sum(t.count("\n") + 1 for t in texts)),
    ]
