#!/usr/bin/env python3
"""Scoped local development and explicit native checkpoint verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dependencies import CompilationInputs, file_identities, header_consumers
from build_evidence import (
    SCOPED_PROTOCOL,
    check_scoped_preparation,
    prepare_build,
    run_ctest,
)
from execution import Campaign, OutputLease, UnresolvedScope, digest, write_json

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {"src/ScanRoute.cpp", "src/ScanRoute.h", "rust/film-juicer-core/src/route.rs", "rust/film-juicer-core/src/color/output.rs"}
OWNERS = ("tests/route/development.json", "tests/ffi/output-color-development.json")


def staged_changes(root):
    rows = subprocess.check_output(
        ["git", "diff", "--cached", "--raw", "--no-abbrev", "--no-renames", "-z", "--"], cwd=root
    ).decode().split("\0")
    return {rows[index + 1]: rows[index] for index in range(0, len(rows) - 1, 2)}


def dirty_paths(root, staged=None):
    if staged is None:
        staged = staged_changes(root)
    commands = [
        ["git", "diff", "--name-only", "--no-renames", "-z", "--"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    ]
    return set(staged) | {name for command in commands for name in subprocess.check_output(command, cwd=root).decode().split("\0") if name}


def is_documentation(name):
    """Identify repository prose for broad discovery, never declared inputs."""
    path = Path(name)
    if path.suffix != ".md":
        return False
    if len(path.parts) == 1:
        return name in {"README.md", "CONTRIBUTING.md"}
    if path.parts[0] == "docs":
        return True
    if path.parts[0] != "tests":
        return False
    fixture = any(part == "fixtures" or part.endswith("-fixtures") for part in path.parts[:-1])
    return not fixture or path.name == "README.md" or path.name.endswith("-README.md")


def source_snapshot(root):
    """Observe owned source/configuration content without reading the fixture corpus."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode().split("\0")
    staged = staged_changes(root)
    dirty = dirty_paths(root, staged)
    names += ["rust/film-juicer-plugin/src/cuda/sys.rs", "src/GeneratedColorSpaces.cpp", "src/GeneratedColorSpaces.h"]
    observed = {}
    for name in sorted(set(names) | dirty):
        if (not name or name.startswith((".juicer-local/", "out/"))
                or "__pycache__" in Path(name).parts or is_documentation(name)):
            continue
        path = root / name
        payload = (name.startswith(("Resources/", "external/")) or "/fixtures/" in name
                   or path.suffix.lower() in {".npy", ".npz", ".png", ".bin", ".zip", ".exr"})
        if payload:
            if name in dirty:
                # Unowned payloads cannot pass this scope. Keep their membership
                # and index state pending without reading their scientific bytes.
                observed[name] = {"unresolved": True, "exists": path.is_file()}
            continue
        observed[name] = digest(path) if path.is_file() else None
    # Keep the index identity when its bytes differ from HEAD, even if the
    # worktree was restored. This also observes a second staged-only edit.
    for name in observed.keys() & staged.keys():
        observed[name] = {"worktree": observed[name], "index": staged[name]}
    return observed


def edit_selection(current, previous, explicit=()):
    old = previous["observed"]
    changed = {name for name in old.keys() | current.keys()
               if not is_documentation(name)
               and (old.get(name) != current.get(name) or (name in old) != (name in current))}
    return sorted(set(previous.get("pending", ())) | changed)


def rust_body_edit(root, name, revision):
    """Prove a narrow existing expression edit; uncertainty broadens package checks."""
    if name not in {"rust/film-juicer-core/src/route.rs", "rust/film-juicer-core/src/color/output.rs"}:
        return False
    try:
        original = subprocess.check_output(["git", "show", revision + ":" + name], cwd=root, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return False
    current = (root / name).read_text(encoding="utf-8")
    if name.endswith("/color/output.rs"):
        def normalize(text):
            start = text.find("const SOURCE: Literal = Literal {")
            end = text.find("\nfn subhashes(", start)
            if start < 0 or end < 0:
                return None
            literals = re.sub(r"(?<![A-Za-z0-9_])-?[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?", "FLOAT", text[start:end])
            return text[:start] + literals + text[end:]
        baseline = normalize(original)
        return baseline is not None and baseline == normalize(current)
    # These four existing match arms are inside resolve, never declarations.
    pattern = r"(\(CapturePolarity::(?:Negative|Positive), RouteSelection::(?:Direct|Print)\) => )ScanRoute::(?:NegativeDirect|NegativePrint|PositiveDirect|PositivePrint)(,)"
    before, count = re.subn(pattern, r"\1ROUTE\2", original)
    after, actual = re.subn(pattern, r"\1ROUTE\2", current)
    return count == actual == 4 and before == after


def select_consumers(root, build, selected, *, require_compiled=False, compilation=None):
    unknown = set(selected) - SUPPORTED
    if unknown:
        raise UnresolvedScope("unresolved development scope: " + ", ".join(sorted(unknown)))
    if not selected:
        raise UnresolvedScope("empty development selection has no successful scope to reuse")
    if any(not (root / name).is_file() for name in selected):
        raise UnresolvedScope("unresolved deleted or renamed development input")
    targets, tests, inputs, packages, filters = set(), set(), set(selected), set(), set()
    compiler = {name for name in selected if name.endswith(".cpp")}
    body_only = {}
    for owner_path in OWNERS:
        owner = json.loads((root / owner_path).read_text(encoding="utf-8"))
        owned = set(owner.get("sources", [owner.get("source")])) & set(selected)
        if not owned:
            continue
        targets.add(owner["target"])
        tests.add(owner["test"])
        inputs.add(owner_path)
        if owner.get("fixture"):
            inputs.add(owner["fixture"])
        if any(name.endswith(".rs") for name in owned):
            packages.add("film-juicer-core")
            filters.add(owner["rust_filter"])
            tests.add("Quality.RustBoundaries")
            inputs.update(["tests/quality/test_rust_boundaries.py",
                           "tests/ffi/color_preparation_test.cpp"])
            inputs.update(path.relative_to(root).as_posix() for path in (root / "tests/quality/fixtures/rust_boundaries").glob("*.rs"))
            body_only.update({name: rust_body_edit(root, name, owner["rust_interface_revision"]) for name in owned if name.endswith(".rs")})
    if "src/ScanRoute.h" in selected:
        objects, sources = header_consumers(root, build, ["src/ScanRoute.h"], require_compiled=require_compiled, compilation=compilation)
        targets.update(objects)
        compiler.update(sources)
    if any(not proven for proven in body_only.values()):
        packages.add("film-juicer-plugin")
        filters = {""}  # Public interface or uncertain impact: cover both packages.
    if not targets:
        raise UnresolvedScope("no explicit development owner")
    return {"files": sorted(selected), "compiler_consumers": sorted(compiler),
            "targets": sorted(targets), "tests": sorted(tests), "inputs": sorted(inputs),
            "rust_packages": sorted(packages), "rust_test_filters": sorted(filters),
            "checks": ["source hygiene", "selected native/Rust analysis", *sorted(tests)],
            "reasons": ["explicit owners beside CTest registration; all configured header variants; linkage only determines build dependencies"]}


def check_receipt_payloads(root, receipt):
    record = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if record.get("status") != "passed" or record.get("returncode") != 0 or not record.get("process_tree", {}).get("drained"):
        raise ValueError("failed, partial or undrained evidence cannot be reused")
    if digest(root / record["output"]) != record["output_sha256"]:
        raise ValueError("command evidence payload changed")
    executable = shutil.which(record["identity"]["command"][0])
    if executable is None or digest(Path(executable)) != record["identity"]["executable_sha256"]:
        raise ValueError("consumed command executable changed")
    process = json.loads(Path(record["process_tree"]["receipt"]).read_text(encoding="utf-8"))
    if any(record["process_tree"].get(key) != value for key, value in process.items()):
        raise ValueError("process evidence payload changed")
    return record


def load_development_state(root, build, previous_path, explicit):
    state = build / "verification-development.json"
    latest = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else None
    if latest:
        path = Path(latest["receipt"])
        if digest(path) != latest["sha256"]:
            raise ValueError("latest development receipt changed; pending work cannot be discarded")
        if previous_path and Path(previous_path).resolve() != path.resolve():
            raise ValueError("previous receipt is not the latest development attempt; retain pending work")
        previous_path = path
    if not previous_path:
        current = source_snapshot(root)
        # Observation alone is never a successful verification baseline.
        return {"observed": current,
                "pending": sorted(name for name in dirty_paths(root) if not is_documentation(name)),
                "established": True}
    previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    if previous.get("preset") != build.name or previous.get("mode") not in {"development", "preparation", "qualification"}:
        raise ValueError("previous receipt must be development evidence for this preset")
    if previous.get("protocol") != SCOPED_PROTOCOL:
        current = source_snapshot(root)
        # Retain failures, but never import a legacy preparation/qualification pass.
        pending = set(previous.get("pending", ())) | {name for name in dirty_paths(root) if not is_documentation(name)}
        return {"observed": current, "pending": sorted(pending), "established": True}
    return previous


def quality_command(preset, selection):
    command = [sys.executable, "-B", "scripts/check-quality.py", "--preset", preset,
               "--development", "--files", *selection["files"]]
    if selection["compiler_consumers"]:
        command += ["--compiler-consumers", *selection["compiler_consumers"]]
    if selection["rust_packages"]:
        command += ["--rust-packages", *selection["rust_packages"],
                    "--rust-test-filters", *selection["rust_test_filters"]]
    return command


def scope_key(selection):
    return json.dumps(selection["files"], separators=(",", ":"))


def verify_development_receipt(campaign, build, previous, *, selection=None):
    if previous.get("protocol") != SCOPED_PROTOCOL or previous.get("scope_status") != "passed":
        raise ValueError("only successful scoped development evidence can be reused")
    compilation = CompilationInputs(campaign.root, build) if "src/ScanRoute.h" in previous["selection"]["files"] else None
    if selection is None:
        selection = select_consumers(campaign.root, build, previous["selection"]["files"], compilation=compilation)
    if selection != previous["selection"]:
        raise ValueError("development selection changed")
    check_scoped_preparation(campaign, build, Path(previous["build"]), compilation=compilation)
    quality = check_receipt_payloads(campaign.root, previous["quality_command"])
    tests = check_receipt_payloads(campaign.root, previous["tests_command"])
    if quality["identity"]["command"] != quality_command(build.name, selection):
        raise ValueError("development quality command changed")
    pattern = "^(" + "|".join(re.escape(name) for name in selection["tests"]) + ")$"
    command = tests["identity"]["command"]
    expected = ["ctest", "--preset", build.name, "-R", pattern, "--output-on-failure", "--output-junit", previous["junit"]]
    if command != expected:
        raise ValueError("development CTest command/membership changed")
    if file_identities([Path(name) for name in previous["evidence"]]) != previous["evidence"]:
        raise ValueError("development evidence or consumed tool/input changed")


def _development_owned(args, *, started=None, lease_seconds=0.0):
    started = time.monotonic() if started is None else started
    timings = {"lease_wait_seconds": lease_seconds}
    phase = time.monotonic()
    if args.preset != "linux-debug":
        raise ValueError("development supports the existing linux-debug tree only")
    build = ROOT / "out/build" / args.preset
    previous = load_development_state(ROOT, build, args.previous, args.files or ())
    timings["state_load_seconds"] = time.monotonic() - phase
    scopes = dict(previous.get("successful_scopes", {}))
    pointer = build / "verification-development.json"
    if previous.get("scope_status") == "passed" and pointer.is_file():
        scopes[scope_key(previous["selection"])] = json.loads(pointer.read_text())
    explicit = set(args.files or previous.get("selection", {}).get("files", ()))
    selected = sorted(explicit & SUPPORTED)
    if not selected:
        raise UnresolvedScope("select a supported route/header/output-color input")
    phase = time.monotonic()
    current = source_snapshot(ROOT)
    # Unsupported changes never become verified merely because their bytes are
    # unnecessary to this selected owner, including a second already-dirty edit.
    pending = set(edit_selection(current, previous)) | (explicit - SUPPORTED)
    for name, identity in current.items():
        if isinstance(identity, dict):
            worktree = identity.get("worktree", identity)
            if isinstance(worktree, dict) and worktree.get("unresolved"):
                pending.add(name)
    timings["observation_entry_seconds"] = time.monotonic() - phase
    campaign = Campaign(ROOT, args.preset, "development-" + uuid.uuid4().hex, ["prepare", "quality", "tests"])
    result_path = campaign.directory / "development.json"
    record = {"mode": "development", "protocol": SCOPED_PROTOCOL, "preset": args.preset,
              "status": "running", "scope_status": "running", "observed": current,
              "pending": sorted(pending | set(selected)), "selection": {"files": selected}, "successful_scopes": scopes,
              "timings": timings}
    print(f"Development receipt: {result_path}", flush=True)
    try:
        candidate = previous
        saved = scopes.get(scope_key({"files": selected}))
        if saved:
            if digest(Path(saved["receipt"])) != saved["sha256"]:
                raise ValueError("successful scoped record changed")
            candidate = json.loads(Path(saved["receipt"]).read_text())
        reusable = (candidate.get("scope_status") == "passed" and not set(selected) & pending
                    and candidate.get("selection", {}).get("files") == selected)
        if reusable:
            try:
                phase = time.monotonic()
                # No build/analyzer/test runs on this branch. Admit the stored
                # pass once, with fresh current inputs, immediately before reuse.
                verify_development_receipt(campaign, build, candidate)
            except ValueError:
                reusable = False
            finally:
                timings["reuse_validation_seconds"] = time.monotonic() - phase
        if reusable:
            selection = candidate["selection"]
            record["selection"] = selection
            record.update({name: candidate[name] for name in ("build", "quality_command", "tests_command", "quality", "junit", "evidence")})
            record["reused_from"] = str(saved["receipt"] if saved else args.previous or json.loads(pointer.read_text())["receipt"])
            print("Reused previous scoped pass; no build, analyzer or tests executed.", flush=True)
        else:
            phase = time.monotonic()
            selection = select_consumers(ROOT, build, selected)
            record["selection"] = selection
            timings["selection_entry_seconds"] = time.monotonic() - phase
            print(json.dumps(selection, indent=2), flush=True)
            phase = time.monotonic()
            _, built = prepare_build(campaign, "prepare", source=ROOT, build=build,
                                     inputs=[*selection["inputs"], "scripts/verify.py", "scripts/check-quality.py"], timeout=1200,
                                     configure_args=["-DBUILD_TESTING=ON"], targets=selection["targets"])
            timings["preparation_seconds"] = time.monotonic() - phase
            record["build"] = str(built)
            selection = select_consumers(ROOT, build, selected, require_compiled=True)
            record["selection"] = selection
            phase = time.monotonic()
            quality = campaign.run("quality", quality_command(args.preset, selection), inputs=selected,
                                   timeout=1200, repeat_reason="analyze selected development scope")
            timings["quality_seconds"] = time.monotonic() - phase
            junit = campaign.directory / "tests.xml"
            phase = time.monotonic()
            tests = run_ctest(campaign, "tests", build=build, prepared=built,
                              inputs=selection["inputs"], timeout=600, tests=selection["tests"], junit=junit)
            timings["ctest_seconds"] = time.monotonic() - phase
            payloads = [built, quality, tests, junit]
            for receipt in (built, quality, tests):
                stage = check_receipt_payloads(ROOT, receipt)
                payloads += [ROOT / stage["output"], Path(stage["process_tree"]["receipt"])]
            output = (ROOT / check_receipt_payloads(ROOT, quality)["output"]).read_text(encoding="utf-8")
            qpath = Path(re.search(r"^Quality run: (.+)$", output, re.M)[1]) / "run.json"
            q = json.loads(qpath.read_text(encoding="utf-8"))
            if q.get("status") != "passed" or q.get("mode") != "development" or q["selected"] != selected:
                raise ValueError("development quality did not complete its declared scope")
            payloads += [qpath, *(qpath.parent / name for name in q["files"]), *(Path(name) for name in q["consumed_inputs"]),
                         *(ROOT / name for name, checksum in q["inputs"]["files"].items() if checksum is not None)]
            for name, checksum in q["files"].items():
                if digest(qpath.parent / name) != checksum:
                    raise ValueError("inner quality payload changed")
            for name, checksum in q["consumed_inputs"].items():
                if digest(Path(name)) != checksum:
                    raise ValueError("consumed quality input changed")
            record.update(quality_command=str(quality), tests_command=str(tests), quality=str(qpath), junit=str(junit), evidence=file_identities(payloads))
        phase = time.monotonic()
        if source_snapshot(ROOT) != current:
            raise ValueError("source membership or contents changed during development")
        timings["observation_final_seconds"] = time.monotonic() - phase
        if not reusable:
            phase = time.monotonic()
            compilation = CompilationInputs(ROOT, build) if "src/ScanRoute.h" in selected else None
            if select_consumers(ROOT, build, selected, compilation=compilation) != selection:
                raise ValueError("source membership, contents or selection changed during development")
            timings["selection_final_seconds"] = time.monotonic() - phase
            phase = time.monotonic()
            check_scoped_preparation(campaign, build, Path(record["build"]), compilation=compilation)
            timings["preparation_final_seconds"] = time.monotonic() - phase
        pending -= set(selected)
        record.update(scope_status="passed", status="incomplete" if pending else "passed", pending=sorted(pending))
        print(f"Selected scope passed; {len(pending)} other paths remain unresolved.", flush=True)
    except BaseException as error:
        record.update(status="unresolved" if isinstance(error, UnresolvedScope) else "failed", scope_status="failed", error=str(error))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["timing_boundary"] = "development entry before lease through start of publication; supervised command wall time includes publication and lease release"
        if record.get("reused_from"):
            record["bookkeeping_before_publication_seconds"] = record["seconds"]
        phase = time.monotonic()
        write_json(result_path, record)
        write_json(build / "verification-development.json", {"receipt": str(result_path), "sha256": digest(result_path)})
        print(json.dumps({"publication_seconds": time.monotonic() - phase,
                          "entry_through_publication_seconds": time.monotonic() - started}), flush=True)
    return result_path


def development(args):
    started = time.monotonic()
    with OutputLease(ROOT, ROOT / "out/build" / args.preset):
        return _development_owned(args, started=started, lease_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    preparation = modes.add_parser("prepare", help="prepare explicit owning targets and dependencies only")
    preparation.add_argument("--preset", required=True)
    preparation.add_argument("--files", nargs="+", required=True)
    dev = modes.add_parser("development")
    dev.add_argument("--preset", required=True)
    choice = dev.add_mutually_exclusive_group(required=True)
    choice.add_argument("--files", nargs="+")
    choice.add_argument("--previous", type=Path)
    from qualification import GROUPS, PRESETS
    checkpoint = modes.add_parser("qualification", help="explicit native checkpoint; no unknown-scope fallback")
    checkpoint.add_argument("--preset", required=True, choices=PRESETS)
    checkpoint.add_argument("--reason", required=True)
    checkpoint.add_argument("--selection", nargs="+", required=True, choices=sorted(GROUPS))
    checkpoint.add_argument("--runtime", choices=("host", "all"), required=True)
    checkpoint.add_argument("--base")
    checkpoint.add_argument("--campaign")
    checkpoint.add_argument("--repeat-reason", default="")
    collection = modes.add_parser("collect-qualification", help="aggregate complete retained native/profile checkpoints")
    collection.add_argument("--receipts", nargs="+", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "collect-qualification":
        from qualification import collect_qualification
        result = collect_qualification(ROOT, args.receipts)
        if args.output.exists():
            raise ValueError("retain the existing aggregate and use a new output path")
        write_json(args.output, result)
        print("Complete retained qualification:", args.output)
        return 0
    if args.mode == "qualification":
        from qualification import qualify
        result = qualify(ROOT, args)
        return 2 if json.loads(result.read_text())["status"] == "incomplete" else 0
    if args.mode == "prepare":
        if args.preset != "linux-debug":
            raise ValueError("preparation supports linux-debug only")
        build = ROOT / "out/build" / args.preset
        selection = select_consumers(ROOT, build, args.files)
        with OutputLease(ROOT, build):
            campaign = Campaign(ROOT, args.preset, "preparation-" + uuid.uuid4().hex, ["prepare"])
            _, built = prepare_build(campaign, "prepare", source=ROOT, build=build,
                                     inputs=[*selection["inputs"], "scripts/verify.py", "scripts/check-quality.py"], timeout=1200,
                                     configure_args=["-DBUILD_TESTING=ON"], targets=selection["targets"])
            print("Scoped preparation only; pending development state preserved:", built)
    else:
        result = development(args)
        print(result)
        if json.loads(result.read_text())["status"] == "incomplete":
            return 2
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
