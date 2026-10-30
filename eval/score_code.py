"""Execute frozen MBPP generations in a restricted Docker container and freeze labels."""

import argparse
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from eval.import_benchmarks import save_dataset

IMAGE = "python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval/multibench"))
    args = parser.parse_args()
    source = args.data / "mbpp-unscored.jsonl"
    manifest = json.loads(source.with_suffix(".manifest.json").read_bytes())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Input checksum mismatch")
    if (args.data / "mbpp.jsonl").exists():
        raise ValueError("Output already exists; choose a fresh data directory")
    name = "certiroute-score-" + uuid.uuid4().hex
    worker = Path(__file__).with_name("code_worker.py").resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--mount",
        f"type=bind,source={source.resolve().parent},target=/input,readonly",
        "--mount",
        f"type=bind,source={worker},target=/worker.py,readonly",
        IMAGE,
        "python",
        "-I",
        "-u",
        "/worker.py",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=13000, check=True)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
    verdicts = [json.loads(line) for line in result.stdout.splitlines()]
    rows = [json.loads(line) for line in source.read_bytes().splitlines()]
    if [r["id"] for r in rows] != [v["id"] for v in verdicts]:
        raise ValueError("Incomplete or misaligned execution results")
    for row, verdict in zip(rows, verdicts, strict=True):
        row["test_setup"] = (
            "\n".join(row["test_setup"])
            if isinstance(row["test_setup"], list)
            else row["test_setup"]
        )
        for tier in ("cheap", "strong"):
            if verdict[tier] not in ("pass", "fail", "timeout"):
                raise ValueError("Invalid execution verdict")
            row["q_" + tier] = float(verdict[tier] == "pass")
    manifest.update(
        quality="Pass all sanitized MBPP base assertions after fenced-code extraction; "
        "timeout=15s.",
        execution={
            "image": IMAGE,
            "network": "none",
            "memory_mb": 512,
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "unscored_sha256": manifest["dataset_sha256"],
            "verdicts": verdicts,
        },
    )
    save_dataset(args.data, "mbpp", rows, manifest)
    print(f"Scored {len(rows)} paired programs in Docker; saved immutable MBPP snapshot.")


if __name__ == "__main__":
    main()
