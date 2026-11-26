"""Check the exit status and diagnostic of a deliberate sanitizer control."""

import argparse
import subprocess
import sys
from pathlib import Path


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
    args = parser.parse_args()
    command = [str(args.executable.resolve(strict=True)), args.mode]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False)
        output = result.stdout.decode("utf-8", errors="replace")
        print(output, end="")
        verify_failure(args.mode, result.returncode, output)
    except subprocess.TimeoutExpired as error:
        print((error.output or b"").decode("utf-8", errors="replace"), end="")
        print(error, file=sys.stderr)
        return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
