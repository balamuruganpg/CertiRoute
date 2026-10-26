"""Freeze public RouteLLM GSM8K pairs; only --download permits networking."""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from certiroute.config import GatewayConfig, load_config
from certiroute.provenance import model_identity, overlaps, prompt_signature
from eval.fixtures import FixtureRow, TokenCounts, last_integer
from eval.manifest import DatasetManifest, SourceFile

REVISION = "0b64fdafe049e596a3f5657c219329f24af24198"
ROOT = f"https://raw.githubusercontent.com/lm-sys/RouteLLM/{REVISION}/"
FILES = {
    "gsm8k_responses.csv": (
        ROOT + "routellm/evals/gsm8k/gsm8k_responses.csv",
        "1f38ed03db354a020f71e92022383dc3cf37bbf2694dea4daaeea7b6c43fe57b",
    ),
    "test.jsonl": (
        ROOT + "routellm/evals/gsm8k/test.jsonl",
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    ),
    "train.jsonl": (
        ROOT + "routellm/evals/gsm8k/train.jsonl",
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    ),
    "generate_responses.py": (
        ROOT + "routellm/evals/gsm8k/generate_responses.py",
        "c22165fa06a913658b10262c20cd03d1d7246b669d48fdd22153171b21156ca5",
    ),
    "LICENSE-RouteLLM.txt": (
        ROOT + "LICENSE",
        "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
    ),
    "LICENSE-GSM8K.txt": (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
        "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b",
    ),
}


def estimate_tokens(text: str) -> int:
    """Explicit tokenizer-free approximation; excludes chat framing tokens."""
    return math.ceil(len(text) / 4)


def build(source_dir: Path, out: Path, download: bool = False, *, all_rows: bool = False) -> None:
    raw = {}
    for name, (url, digest) in FILES.items():
        path = source_dir / name
        if download and not path.exists():
            response = httpx.get(url, timeout=60, follow_redirects=True)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != digest:
                raise ValueError(f"Unexpected upstream checksum: {name}")
            source_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
        raw[name] = content
    config = load_config(Path("configs/offline.gsm8k.yaml"))
    train = [json.loads(line) for line in raw["train.jsonl"].decode().splitlines()]
    test = [json.loads(line) for line in raw["test.jsonl"].decode().splitlines()]
    cached = list(csv.DictReader(io.StringIO(raw["gsm8k_responses.csv"].decode())))
    if len(test) != len(cached):
        raise ValueError("Source row counts disagree.")
    prefix = "".join(
        "Question: " + row["question"] + "\nAnswer: " + row["answer"] + "\n\n" for row in train[:8]
    )
    if all_rows:
        fields = config.model_dump(mode="json")
        fields["router"]["signature_prefix"] = prefix
        config = GatewayConfig.model_validate(fields)
    # Selection consults prompts only, before looking at outputs or quality.
    indices = sorted(
        range(len(cached)), key=lambda i: hashlib.sha256(cached[i]["prompt"].encode()).hexdigest()
    )
    selected: list[FixtureRow] = []
    selected_indices: list[int] = []
    excluded_indices: list[int] = []
    signatures: list[frozenset[str]] = []
    for index in indices:
        record, truth = cached[index], test[index]
        if record["prompt"] != truth["question"]:
            raise ValueError("Cache prompt does not match ground-truth row.")
        prompt = prefix + "Question: " + record["prompt"] + "\nAnswer:"
        request: dict[str, object] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1024,
            "top_p": 1,
            "stop": ["Question"],
        }
        signature = prompt_signature(request, config.router.signature_prefix)
        if any(overlaps(signature, old) for old in signatures):
            excluded_indices.append(index)
            continue
        expected = last_integer(truth["answer"])
        if expected is None:
            raise ValueError("Missing ground truth.")
        cheap = record[config.models.cheap.model + "_response"]
        strong = record[config.models.strong.model + "_response"]
        q_cheap = float(last_integer(cheap) == expected)
        q_strong = float(last_integer(strong) == expected)
        if (record[config.models.cheap.model] == "True") != bool(q_cheap) or (
            record[config.models.strong.model] == "True"
        ) != bool(q_strong):
            raise ValueError("Recomputed scores differ from upstream labels.")
        selected.append(
            FixtureRow(
                id=f"gsm8k-test-{index}",
                source="routellm_gsm8k_v1",
                request=request,
                expected=expected,
                cheap_output=cheap,
                strong_output=strong,
                q_cheap=q_cheap,
                q_strong=q_strong,
                cheap_usage=TokenCounts(
                    prompt_tokens=estimate_tokens(prompt), completion_tokens=estimate_tokens(cheap)
                ),
                strong_usage=TokenCounts(
                    prompt_tokens=estimate_tokens(prompt), completion_tokens=estimate_tokens(strong)
                ),
            )
        )
        selected_indices.append(index)
        signatures.append(signature)
        if not all_rows and len(selected) == 100:
            break
    if not all_rows and len(selected) != 100:
        raise ValueError("Not enough distinct prompts.")
    out.mkdir(parents=True, exist_ok=False)
    dataset = out / "paired.jsonl"
    dataset.write_bytes(("\n".join(row.model_dump_json() for row in selected) + "\n").encode())
    manifest = DatasetManifest(
        source="routellm_gsm8k_v1",
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        frozen_at=datetime(2026, 10, 24, tzinfo=UTC),  # Release, not generation date.
        models=model_identity(config),
        quality_metric=config.quality_metric,
        scope=config.scope,
        source_files={
            name: SourceFile(url=url, sha256=digest) for name, (url, digest) in FILES.items()
        },
        selection="Sort all 1319 prompts by SHA256(UTF-8 question); greedily retain first 100 "
        "with full-prompt token-trigram Jaccard < 0.9 against retained prompts. "
        "No selection on labels or outputs.",
        selected_source_indices=selected_indices,
        prompt_template="Single user message: first 8 GSM8K training examples formatted as "
        "'Question: {question}\\nAnswer: {answer}\\n\\n', then "
        "'Question: {test_question}\\nAnswer:'. Full text is in every row.",
        generation={
            "temperature": 0,
            "max_tokens": 1024,
            "stop": ["Question"],
            "top_p": "Not explicit upstream; replay uses 1 (assumed default).",
            "provider": "Upstream SGLang OpenAI-compatible router; original host not recorded.",
            "generated_at": None,
            "finish_reason": None,
            "latency": None,
            "provenance": "Settings/model IDs from pinned generation script; original "
            "wire requests, provider revisions and generation dates unavailable.",
        },
        token_counts="ESTIMATED independently for prompt and each completion: "
        "ceil(Unicode character count / 4), including all 8 few-shot examples. "
        "Same approximation for both models; excludes chat framing. "
        "Not model-tokenizer counts or billed provider usage.",
        limitations=[
            "Real frozen model outputs and recomputed correctness; estimated tokens and costs.",
            "Original generation date, provider deployment and exact top_p are not recorded. "
            "These artifacts are bound to .invalid replay endpoints, not live deployment.",
            "Full-prompt deduplication includes a long common few-shot prefix; selection can "
            "favor longer or lexically distinctive questions. This is not a random GSM8K sample.",
            "Metric copies RouteLLM's last unsigned integer rule, including its limitations "
            "for decimals, signs and trailing numbers; it is not a general math verifier.",
            "GSM8K may be in LLM pretraining data; benchmark contamination is unmeasured.",
        ],
    )
    if all_rows:
        fields = manifest.model_dump(mode="json")
        fields.update(
            signature_prefix=prefix,
            excluded_source_indices=excluded_indices,
            selection=f"Process all {len(cached)} prompts in SHA256(UTF-8 question) order. "
            "Greedy token-trigram Jaccard >= 0.9 deduplication on the test question only, "
            "after removing the exact shared eight-example prefix. No selection on outcomes.",
        )
        fields["limitations"][2] = (
            "Question-only lexical deduplication does not detect all semantic duplicates. "
            "This is one finite public benchmark, not a representative sample of live traffic."
        )
        manifest = DatasetManifest.model_validate(fields)
        (out / "gateway-template.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    dataset.with_suffix(".manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    for name in ("LICENSE-RouteLLM.txt", "LICENSE-GSM8K.txt"):
        (out / name).write_bytes(raw[name])
    (out / "NOTICE.md").write_text(
        "# Attribution and modifications\n\n"
        "Paired outputs: LMSYS RouteLLM (Apache-2.0), revision " + REVISION + ".\n"
        "Questions, reference answers and few-shot examples: OpenAI GSM8K (MIT).\n"
        "See the accompanying license files and paired.manifest.json for pinned sources.\n\n"
        f"CertiRoute modifications: selected and deduplicated {len(selected)} rows; "
        "reconstructed full "
        "prompts; rescored answers; added estimated usage and provenance; converted CSV "
        "to JSONL. Model response text is unchanged.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(selected)} frozen pairs to {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/eval/routellm-source"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/gsm8k"))
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned source files")
    parser.add_argument("--all", action="store_true", help="All usable pairs, question-only dedup")
    args = parser.parse_args()
    build(args.source_dir, args.out, args.download, all_rows=args.all)


if __name__ == "__main__":
    main()
