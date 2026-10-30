"""Import pinned public ARC pairs and MBPP generations; never execute downloaded code.

RouterBench's pickle is decoded symbolically: every GLOBAL is replaced by an inert
recording node. No pandas/numpy imports, reducers or downloaded callables execute.
Only the known object-array block layout is accepted.
"""

import argparse
import ast
import hashlib
import io
import json
import math
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from eval.offline import write_json

ROUTERBENCH_REV = "784021482c3f320c6619ed4b3bb3b41a21424fcb"


class Symbol:
    args: tuple[Any, ...] = ()
    state: Any = None

    def __init__(self, *args: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state


class SymbolicUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type[Symbol]:
        return type(name, (Symbol,), {})

    def persistent_load(self, pid: object) -> Any:
        raise ValueError("Persistent references are forbidden")


def routerbench_columns(raw: bytes) -> dict[str, list[Any]]:
    frame = SymbolicUnpickler(io.BytesIO(raw)).load()
    blocks, axes = frame.state["_mgr"].args
    names = axes[0].args[1]["data"].state[4]
    array, location, _ = blocks[0].args
    _, shape, _, fortran, values = array.state
    if location.args != (0, 24, 1) or shape != (24, 36497) or fortran:
        raise ValueError("Unexpected RouterBench layout")
    if len(values) != math.prod(shape):
        raise ValueError("Truncated RouterBench block")
    return {names[i]: values[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])}


def single_text(value: str) -> str:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], str):
        raise ValueError("Expected one frozen text response")
    return parsed[0]


def fetch(url: str, path: Path) -> bytes:
    lock_path = Path(__file__).resolve().parents[1] / "data/eval/multibench/sources.lock.json"
    expected = json.loads(lock_path.read_bytes()).get(url)
    if expected is None:
        raise ValueError("Source URL is not in the committed checksum lock")
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Downloaded source checksum mismatch")
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Cached source checksum mismatch")
    return raw


def usage(prompt: str, output: str) -> dict[str, int]:
    return {
        "prompt_tokens": math.ceil(len(prompt) / 4),
        "completion_tokens": math.ceil(len(output) / 4),
    }


def record(identifier: str, task: str, prompt: str, cheap: str, strong: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "task": task,
        "request": {"model": "certiroute/auto", "messages": [{"role": "user", "content": prompt}]},
        "cheap_output": cheap,
        "strong_output": strong,
        "cheap_usage": usage(prompt, cheap),
        "strong_usage": usage(prompt, strong),
    }


def save_dataset(
    out: Path, name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    path = out / (name + ".jsonl")
    raw = "".join(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n" for r in rows).encode()
    with path.open("xb") as stream:
        stream.write(raw)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            **metadata,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(rows),
            "token_counts": "Estimated ceil(characters/4); excludes hidden chat-template tokens.",
            "frozen_at": "2026-10-26",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("build/expansion-sources"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/multibench"))
    parser.add_argument("--mbpp-revision", required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/withmartian/routerbench/resolve/"
        + ROUTERBENCH_REV
        + "/routerbench_0shot.pkl"
    )
    raw = fetch(url, args.cache / "routerbench_0shot.pkl")
    columns = routerbench_columns(raw)
    cheap_model, strong_model = "mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"
    rows = []
    for i, task in enumerate(columns["eval_name"]):
        if task != "arc-challenge":
            continue
        row = record(
            columns["sample_id"][i],
            task,
            single_text(columns["prompt"][i]),
            single_text(columns[cheap_model + "|model_response"][i]),
            single_text(columns[strong_model + "|model_response"][i]),
        )
        row.update(q_cheap=float(columns[cheap_model][i]), q_strong=float(columns[strong_model][i]))
        rows.append(row)
    save_dataset(
        args.out,
        "arc",
        rows,
        {
            "models": {"cheap": cheap_model, "strong": strong_model},
            "source_files": [{"url": url, "sha256": hashlib.sha256(raw).hexdigest()}],
            "selection": "All ARC-Challenge rows, original order; no outcome selection.",
            "quality": "RouterBench published exact-match correctness; retained, not regraded.",
            "generation": "Frozen zero-shot prompts/responses; "
            "upstream decoding metadata unavailable.",
            "pricing": {
                "as_of": "2024-06-26",
                "cheap": [0.24, 0.24],
                "strong": [10, 30],
                "description": "Historical scenario, USD per million estimated tokens.",
            },
        },
    )
    url = (
        "https://raw.githubusercontent.com/google-research/google-research/"
        + args.mbpp_revision
        + "/mbpp/sanitized-mbpp.json"
    )
    tasks_raw = fetch(url, args.cache / ("mbpp-" + args.mbpp_revision + ".json"))
    tasks = json.loads(tasks_raw)
    sources = [{"url": url, "sha256": hashlib.sha256(tasks_raw).hexdigest()}]
    archives = []
    for model in ("gpt-3.5-turbo", "gpt-4-1106-preview"):
        name = model + "_temp_0.0.zip"
        url = "https://github.com/evalplus/evalplus/releases/download/v0.2.0/" + name
        raw = fetch(url, args.cache / name)
        sources.append({"url": url, "sha256": hashlib.sha256(raw).hexdigest()})
        archives.append(zipfile.ZipFile(io.BytesIO(raw)))
    rows = []
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        paths = [
            f"{m}_temp_0.0/Mbpp_{task['task_id']}/0.py"
            for m in ("gpt-3.5-turbo", "gpt-4-1106-preview")
        ]
        if any(p not in z.namelist() for p, z in zip(paths, archives, strict=True)):
            continue
        outputs = [z.read(p).decode() for p, z in zip(paths, archives, strict=True)]
        row = record("mbpp/" + str(task["task_id"]), "mbpp", task["prompt"], *outputs)
        row.update(tests=task["test_list"], test_setup=task.get("test_imports", []))
        # Deliberately no quality labels until isolated tests run.
        rows.append(row)
    save_dataset(
        args.out,
        "mbpp-unscored",
        rows,
        {
            "models": {"cheap": "gpt-3.5-turbo", "strong": "gpt-4-1106-preview"},
            "source_files": sources,
            "selection": "Intersection of sanitized MBPP and both archives.",
            "quality": "Pending isolated base-test execution; not EvalPlus extended tests.",
            "generation": "EvalPlus v0.2.0 greedy chat generations; "
            "gpt-3.5 alias snapshot unknown. "
            "Prompt field is task text; original chat wrapper not recoverable here.",
            "pricing": {
                "as_of": "2024-06-26",
                "cheap": [0.5, 1.5],
                "strong": [10, 30],
                "description": "Historical scenario using GPT-3.5 Feb-2024 rates; "
                "generation snapshot unknown, not actual bills.",
            },
        },
    )
    print("Saved ARC pairs and unscored MBPP generations; run isolated code scoring next.")


if __name__ == "__main__":
    main()
