"""Generate deterministic simulated arithmetic outputs, never model benchmark results."""

import argparse
import re
from pathlib import Path

from eval.fixtures import FixtureRow, TokenCounts


def generate_fixture(path: Path, n: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        for i in range(n):
            a, b = 2 * i + 3, (17 * i) % 89 + 2
            operation = "+" if i % 2 == 0 else "*"
            expected = a + b if operation == "+" else a * b
            # Simulators produce paired observations; the router never sees i or these rules.
            cheap = expected if operation == "+" or i % 5 == 0 else a + b
            strong = expected + 1 if i % 37 == 0 else expected
            prompt = f"Return only the integer. Compute {a} {operation} {b}."
            # Invented tokenizer for the fixture; NOT provider-reported tokens.
            input_tokens = len(re.findall(r"\w+|[^\w\s]", prompt))
            row = FixtureRow(
                id=f"arithmetic-{i:04d}",
                source="synthetic_arithmetic_v1",
                request={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256,
                    "top_p": 1,
                },
                expected=str(expected),
                cheap_output=str(cheap),
                strong_output=str(strong),
                cheap_usage=TokenCounts(
                    prompt_tokens=input_tokens, completion_tokens=max(1, (len(str(cheap)) + 3) // 4)
                ),
                strong_usage=TokenCounts(
                    prompt_tokens=input_tokens,
                    completion_tokens=max(1, (len(str(strong)) + 3) // 4),
                ),
                q_cheap=float(cheap == expected),
                q_strong=float(strong == expected),
            )
            target.write(row.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/eval/sample.jsonl"))
    args = parser.parse_args()
    generate_fixture(args.out)
    print(f"Wrote 240 synthetic paired records to {args.out}")


if __name__ == "__main__":
    main()
