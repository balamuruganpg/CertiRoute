"""Standalone container worker. Never invoke this file directly on the host."""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if not Path("/.dockerenv").exists():
        raise SystemExit("Container execution required")
    rows = [
        json.loads(line) for line in Path("/input/mbpp-unscored.jsonl").read_text().splitlines()
    ]
    for row in rows:
        result: dict[str, object] = {"id": row["id"]}
        for tier in ("cheap", "strong"):
            output = row[tier + "_output"]
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", output, re.DOTALL)
            code = "\n".join(blocks) if blocks else output
            setup = row["test_setup"]
            if isinstance(setup, list):
                setup = "\n".join(setup)
            source = setup + "\n" + code + "\n" + "\n".join(row["tests"]) + "\n"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "candidate.py"
                path.write_text(source)
                proc = subprocess.Popen(
                    [sys.executable, "-I", str(path)],
                    cwd=directory,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=15)
                    verdict = "pass" if returncode == 0 else "fail"
                except subprocess.TimeoutExpired:
                    if sys.platform != "win32":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
                    verdict = "timeout"
                result[tier] = verdict
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
