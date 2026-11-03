"""Outer control outcomes, bounded CI evidence selection and held-fixture release."""

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path
import traceback
import uuid


MARKER = Path("out/control-fixture.json")
FAMILIES = ("checkpoint-controls", "build-evidence-controls", "registration-controls", "ctest-repetition-controls")


def read_fixture_marker(path):
    """Read a marker while rejecting duplicate keys at every object depth."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate fixture marker key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)


def current_fixture_blockers(owner_root):
    """Return current lease owners and campaign running locks under one project."""
    owner_root = Path(owner_root)
    locks_root = owner_root / "out/verification-locks"
    owners = []
    if locks_root.is_dir():
        try:
            for child in locks_root.iterdir():
                owner = child / "owner.json"
                owners.append(owner if owner.exists() or owner.is_symlink() else child)
        except OSError:
            owners.append(locks_root)
    elif locks_root.exists() or locks_root.is_symlink():
        owners.append(locks_root)
    running = [path for path in owner_root.glob("out/validation/*/*/running.lock")
               if path.exists() or path.is_symlink()]
    return sorted(set(owners + running))


def record_fixture(root, test, outcome=None, *, owner_root=None):
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
    owner_root = Path(owner_root or root)
    record = {
        "test": test.id(), "fixture": str(root), "outcome": outcome, **messages,
        "holds": sorted(set(getattr(test, "fixture_holds", ()))),
        "owner_root": str(owner_root),
        "diagnostic_retain": bool(os.environ.get("JUICER_TEST_RETAIN_FIXTURES")
                                   or getattr(test, "retain", False)),
        "blocked_owners": [str(p) for p in current_fixture_blockers(owner_root)],
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
                    record = read_fixture_marker(marker)
                except (OSError, ValueError):
                    # Missing/interrupted writes are unproved evidence.
                    paths.add(fixture)
                    continue
                if not isinstance(record, dict) or record.get("outcome") != "passed":
                    paths.add(fixture)
                    continue
                try:
                    recorded_fixture = Path(record.get("fixture", "")).resolve(strict=True)
                    if recorded_fixture != fixture.resolve(strict=True):
                        raise ValueError("fixture marker owner binding changed")
                except (OSError, TypeError, ValueError):
                    paths.add(fixture)
                    continue
                # A passed outer control can still have an unproved downstream
                # owner. Its workspace is retained for diagnosis and must remain
                # visible to CI selection. Treat malformed owner metadata as
                # unproved too; older or partial receipts cannot authorize loss.
                blocked_owners = record.get("blocked_owners")
                if (not isinstance(blocked_owners, list)
                        or any(not isinstance(owner, str) or not owner for owner in blocked_owners)
                        or blocked_owners):
                    paths.add(fixture)
                    continue
                owner_root = record.get("owner_root")
                try:
                    owner_root = Path(owner_root).resolve(strict=True)
                    fixture_root = fixture.resolve(strict=True)
                    if owner_root not in (fixture_root, fixture_root / "project") or not owner_root.is_dir():
                        raise ValueError("fixture owner root is outside its fixed owner shape")
                except (OSError, TypeError, ValueError):
                    paths.add(fixture)
                    continue
                if current_fixture_blockers(owner_root):
                    paths.add(fixture)
                    continue
                holds = record.get("holds", [])
                if (not isinstance(holds, list)
                        or any(not isinstance(hold, str) or not hold for hold in holds)
                        or len(holds) != len(set(holds)) or holds):
                    paths.add(fixture)
    npy_root = preset_root / "rust-core"
    paths.update(p for p in npy_root.glob("npy-*.npy")
                 if p.is_file() and p.resolve().is_relative_to(npy_root.resolve()))
    return sorted(paths)


def release_fixture_hold(root, name, consumer, *, resolve=Path):
    """Release one completed downstream consumer and remove its held fixture."""
    root = Path(root).resolve(strict=True)
    if (root.parent.name != "checkpoint-controls"
            or not isinstance(name, str) or not name.strip()
            or not isinstance(consumer, str) or not consumer.strip()):
        raise ValueError("fixture hold is outside its fixed policy-control owner")
    marker = root / MARKER
    record = read_fixture_marker(marker)
    if not isinstance(record, dict):
        raise ValueError("fixture hold record is malformed")
    recorded_root = Path(resolve(record.get("fixture", ""))).resolve(strict=True)
    holds = record.get("holds")
    owners = record.get("blocked_owners")
    releases = record.get("hold_releases", [])
    if ("owner_root" not in record or not isinstance(record["owner_root"], str)
            or not record["owner_root"] or "diagnostic_retain" not in record
            or not isinstance(record["diagnostic_retain"], bool)):
        raise ValueError("fixture hold lacks explicit owner-root or diagnostic-retention metadata")
    try:
        owner_root = Path(resolve(record["owner_root"])).resolve(strict=True)
        if owner_root != root / "project":
            raise ValueError("fixture owner root is outside its fixed project owner")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("fixture hold owner root is malformed or outside its fixture") from error
    if not owner_root.is_dir():
        raise ValueError("fixture hold owner root is not a directory")
    if (recorded_root != root or record.get("outcome") != "passed"
            or not isinstance(holds, list)
            or any(not isinstance(hold, str) or not hold for hold in holds)
            or len(holds) != len(set(holds)) or name not in holds
            or not isinstance(owners, list) or owners
            or not isinstance(releases, list)
            or any(not isinstance(release, dict) or set(release) != {"name", "consumer"}
                   or not isinstance(release.get("name"), str) or not release["name"].strip()
                   or not isinstance(release.get("consumer"), str) or not release["consumer"].strip()
                   for release in releases)):
        raise ValueError("fixture hold cannot release an incomplete, blocked or differently owned fixture")
    if current_fixture_blockers(owner_root):
        raise ValueError("fixture hold cannot release while current owner records or campaign work remain")
    holds.remove(name)
    record.setdefault("hold_releases", []).append({"name": name, "consumer": consumer})
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if (holds or record.get("diagnostic_retain", False)
            or os.environ.get("JUICER_TEST_RETAIN_FIXTURES")):
        return False
    try:
        def remove_readonly(function, path, error_info):
            error = error_info[1]
            if not isinstance(error, PermissionError):
                raise error
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(root, onerror=remove_readonly)
    except BaseException:
        root.mkdir(parents=True, exist_ok=True)
        record["outcome"] = "failed"
        record["cleanup_exception"] = traceback.format_exc()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise
    return True


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
