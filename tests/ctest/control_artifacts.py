"""Outer control outcomes and bounded CI evidence selection; never dispose output."""

import argparse
import json
import stat
import sys
from pathlib import Path
import traceback
import uuid


MARKER = Path("out/control-fixture.json")
FAMILIES = ("checkpoint-controls", "build-evidence-controls", "registration-controls", "ctest-repetition-controls")


def read_fixture_marker(path):
    """Reject ambiguous JSON object keys anywhere in a fixture marker."""

    def unique_object(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError(f"duplicate fixture marker key: {key}")
            record[key] = value
        return record

    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)


def _directory_children(path):
    """Read one directory without following links; absent directories are empty."""
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return ()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise OSError("control directory is linked or not a directory: " + str(path))
    return tuple(path.iterdir())


def current_fixture_blockers(owner_root):
    """Find current output leases and named NativeRun locks under one owner."""
    owner_root = Path(owner_root)
    blockers = set()

    def children(path):
        try:
            return _directory_children(path)
        except OSError:
            blockers.add(path)
            return ()

    # Native dependency controls keep their command logs beside out/.
    for run_file in children(owner_root / "logs"):
        if run_file.name.endswith(".running.lock"):
            blockers.add(run_file)

    # Visit each directory component before descending so links cannot move
    # this bounded inspection outside the fixture's output owner.
    for output in children(owner_root / "out"):
        if output.name == "verification-locks":
            # Empty lease directories also protect work during acquisition.
            blockers.update(children(output))
        elif output.name == "validation":
            for preset_root in children(output):
                for result_directory in children(preset_root):
                    if result_directory.name != "results":
                        continue
                    for result_root in children(result_directory):
                        for run_file in children(result_root):
                            if run_file.name.endswith(".running.lock"):
                                blockers.add(run_file)
    return sorted(str(path) for path in blockers)


def record_fixture(root, test, outcome=None, *, owner_root=None):
    result = test._outcome.result if test._outcome else None
    messages = {
        kind: [
            trace
            for owner, trace in getattr(result, kind, ())
            if owner is test or getattr(owner, "test_case", None) is test
        ]
        for kind in ("failures", "errors")
    }
    if outcome is None:
        outcome = (
            "interrupted"
            if getattr(test, "interrupted", False)
            else "incomplete"
            if result is None
            else "failed"
            if any(messages.values())
            else "passed"
        )
    fixture_root = Path(root).resolve()
    owner_root = Path(owner_root or root).resolve()
    try:
        blocked_owners = current_fixture_blockers(owner_root)
    except OSError:
        # Selection treats a nonempty blocker list as unproved evidence.
        blocked_owners = [str(owner_root)]
    record = {
        "test": test.id(),
        "fixture": str(fixture_root),
        "outcome": outcome,
        "failures": messages["failures"],
        "errors": messages["errors"],
        "owner_root": str(owner_root),
        "blocked_owners": blocked_owners,
    }
    if sys.exception() is not None:
        record["cleanup_exception"] = traceback.format_exc()
    path = fixture_root / MARKER
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
            try:
                children = _directory_children(directory)
            except OSError:
                paths.add(directory)
                continue
            if not children:
                continue
            try:
                directory_root = directory.resolve(strict=True)
            except OSError:
                paths.add(directory)
                continue
            for fixture in children:
                try:
                    info = fixture.lstat()
                    if (
                        not stat.S_ISDIR(info.st_mode)
                        or stat.S_ISLNK(info.st_mode)
                        or getattr(info, "st_file_attributes", 0) & 0x400
                    ):
                        continue
                    fixture_root = fixture.resolve(strict=True)
                    if fixture_root.parent != directory_root:
                        continue
                except OSError:
                    paths.add(fixture)
                    continue
                marker = fixture / MARKER
                try:
                    for path, directory_expected in ((marker.parent, True), (marker, False)):
                        info = path.lstat()
                        correct_type = stat.S_ISDIR(info.st_mode) if directory_expected else stat.S_ISREG(info.st_mode)
                        if (
                            not correct_type
                            or stat.S_ISLNK(info.st_mode)
                            or getattr(info, "st_file_attributes", 0) & 0x400
                        ):
                            raise ValueError("linked or invalid fixture marker path")
                    record = read_fixture_marker(marker)
                except (OSError, ValueError, TypeError):
                    # Missing/interrupted writes are unproved evidence.
                    paths.add(fixture)
                    continue
                required = {"test", "fixture", "outcome", "failures", "errors", "owner_root", "blocked_owners"}
                allowed = required | {"cleanup_exception"}
                if (
                    not isinstance(record, dict)
                    or not required.issubset(record)
                    or not set(record).issubset(allowed)
                    or not isinstance(record.get("test"), str)
                    or not record["test"]
                    or record.get("outcome") != "passed"
                    or not isinstance(record.get("failures"), list)
                    or not isinstance(record.get("errors"), list)
                    or not isinstance(record.get("blocked_owners"), list)
                    or any(
                        not isinstance(value, str) or not value
                        for value in (
                            *record.get("failures", []),
                            *record.get("errors", []),
                            *record.get("blocked_owners", []),
                        )
                    )
                    or record.get("failures")
                    or record.get("errors")
                    or record.get("blocked_owners")
                    or "cleanup_exception" in record
                    or not isinstance(record.get("fixture"), str)
                    or not isinstance(record.get("owner_root"), str)
                ):
                    paths.add(fixture)
                    continue
                try:
                    recorded_fixture = Path(record["fixture"]).resolve(strict=True)
                    owner_root = Path(record["owner_root"]).resolve(strict=True)
                    if recorded_fixture != fixture_root or owner_root != fixture_root or not owner_root.is_dir():
                        raise ValueError("fixture marker path binding changed")
                except (OSError, TypeError, ValueError):
                    paths.add(fixture)
                    continue
                try:
                    blockers = current_fixture_blockers(owner_root)
                except OSError:
                    paths.add(fixture)
                    continue
                if blockers:
                    paths.add(fixture)
    npy_root = preset_root / "rust-core"
    paths.update(
        p for p in npy_root.glob("npy-*.npy") if p.is_file() and p.resolve().is_relative_to(npy_root.resolve())
    )
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        nargs="+",
        required=True,
        choices=("linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"),
    )
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
