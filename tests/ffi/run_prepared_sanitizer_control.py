"""Check a deliberate sanitizer failure and retain its original output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verify_failure(mode: str, returncode: int, output: str) -> None:
    expected = {
        "memory": ("AddressSanitizer: heap-use-after-free", "decode_prepared_descriptors"),
        "undefined": ("runtime error: signed integer overflow",),
    }[mode]
    if returncode == 0 or any(marker not in output for marker in expected):
        raise ValueError(f"{mode} control did not fail with the required sanitizer diagnostic (exit={returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("mode", choices=("memory", "undefined"))
    parser.add_argument("artifact_root", type=Path)
    arguments = parser.parse_args()
    arguments.artifact_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=arguments.mode + "-", dir=arguments.artifact_root))
    executable = arguments.executable.resolve(strict=True)
    before = hashlib.sha256(executable.read_bytes()).hexdigest()
    try:
        result = subprocess.run([str(executable), arguments.mode], capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"") + (error.stderr or b"")
        (directory / "output.log").write_bytes(output)
        (directory / "result.json").write_text(json.dumps({
            "command": [str(executable), arguments.mode], "status": "timeout",
            "executable_sha256": before, "output_sha256": hashlib.sha256(output).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    output = result.stdout + result.stderr
    (directory / "output.log").write_bytes(output)
    record = {"command": [str(executable), arguments.mode], "returncode": result.returncode,
              "executable_sha256": before, "output_sha256": hashlib.sha256(output).hexdigest()}
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Sanitizer control evidence: {directory}", flush=True)
    if hashlib.sha256(executable.read_bytes()).hexdigest() != before:
        raise ValueError("sanitizer control executable changed during execution")
    verify_failure(arguments.mode, result.returncode, output.decode("utf-8", errors="replace"))
    print(f"{arguments.mode}: required sanitizer failure observed (exit={result.returncode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
