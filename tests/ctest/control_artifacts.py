"""Outer control outcomes and bounded CI evidence selection; never dispose output."""

import argparse
import json
import sys
from pathlib import Path
import traceback
import uuid


MARKER = Path("out/control-fixture.json")
FAMILIES = ("checkpoint-controls", "build-evidence-controls", "registration-controls", "ctest-repetition-controls")


def record_fixture(root, test, outcome=None):
    result = test._outcome.result if test._outcome else None
    messages = {
        kind: [trace for owner, trace in getattr(result, kind, ())
               if owner is test or getattr(owner, "test_case", None) is test]
        for kind in ("failures", "errors")
    }
    if outcome is None:
        outcome = ("interrupted" if getattr(test, "interrupted", False) else
                   "incomplete" if result is None else
                   "failed" if any(messages.values()) else "passed")
    record = {
        "test": test.id(), "fixture": str(root), "outcome": outcome, **messages,
        "blocked_owners": [str(p) for p in root.glob("out/verification-locks/*/owner.json")],
    }
    if sys.exception() is not None:
        record["cleanup_exception"] = traceback.format_exc()
    path = root / MARKER
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Control outcome could not be saved: test={test.id()} path={path} error={error}", file=sys.stderr)
    return outcome


def selected_paths(preset_root):
    paths = set()
    for parent in (preset_root, preset_root / "verification-controls"):
        for family in FAMILIES:
            directory = parent / family
            for fixture in directory.glob("*"):
                if not fixture.is_dir() or fixture.resolve().parent != directory.resolve():
                    continue
                marker = fixture / MARKER
                try:
                    record = json.loads(marker.read_bytes())
                except (OSError, ValueError):
                    # Missing/interrupted writes are unproved evidence.
                    paths.add(fixture)
                    continue
                if not isinstance(record, dict) or record.get("outcome") != "passed":
                    paths.add(fixture)
    npy_root = preset_root / "rust-core"
    paths.update(p for p in npy_root.glob("npy-*.npy")
                 if p.is_file() and p.resolve().is_relative_to(npy_root.resolve()))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", nargs="+", required=True, choices=(
        "linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"))
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = [p for preset in args.preset for p in selected_paths(root / "out/validation" / preset)]
    value = "\n".join(p.as_posix() + ("/**" if p.is_dir() else "") for p in paths)
    if args.github_output:
        delimiter = uuid.uuid4().hex
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"paths<<{delimiter}\n{value}\n{delimiter}\n")
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
