"""Native command containment, output ownership and result publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


class UnresolvedScope(ValueError):
    """The requested verification scope has no complete supported selection."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command_environment(environment: Mapping[str, str]) -> dict[str, str]:
    # The desktop call ID identifies orchestration, not a compiler/test input.
    # Remove it from child processes too, so an ignored value cannot affect output.
    return {key: value for key, value in environment.items() if key != "CODEX_TOOL_CALL_ID"}


def windows_vctip(environment) -> str | None:
    if os.name != "nt":
        return None
    tools = next(
        (value for key, value in environment.items() if key.upper() == "VCTOOLSINSTALLDIR"),
        None,
    )
    if not tools:
        raise ValueError("Windows build needs the selected VCToolsInstallDir")
    return str(Path(tools) / "bin/Hostx64/x64/VCTIP.EXE")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def name_component(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise ValueError(f"expected a lowercase directory name: {value}")
    return value


def run_process_tree(
    command,
    root,
    environment,
    output,
    timeout,
    receipt,
    lock,
    tree,
    stderr_output=None,
    vctip_cleanup=None,
):
    OutputLease.check_held()
    supervisor = Path(__file__).with_name("process_tree.py")
    policy = ["--vctip-cleanup", json.dumps(vctip_cleanup)] if vctip_cleanup else []
    supervisor_command = [sys.executable, "-B", str(supervisor), str(receipt), str(timeout), *policy, *command]
    process = subprocess.Popen(
        supervisor_command,
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=output,
        stderr=subprocess.STDOUT if stderr_output is None else stderr_output,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    tree.update(drained=False, receipt=str(receipt))
    tree["supervisor_pid"] = process.pid
    try:
        write_json(lock, {"owner_pid": os.getpid(), **tree})
        try:
            process.wait(timeout=None if timeout is None else timeout + 15)
        except BaseException:
            process.stdin.close()  # EOF requests termination and complete draining.
            process.wait(timeout=15)
            raise
    finally:
        process.stdin.close()
        if receipt.is_file():
            observed = json.loads(receipt.read_text(encoding="utf-8"))
            tree.update(observed)
    if process.returncode or not tree["drained"]:
        raise RuntimeError(
            "process-tree completion could not be established; native owner remains blocked; "
            f"supervisor exit={process.returncode}, status={tree.get('status', 'unreported')}, "
            f"error={tree.get('error', 'unreported')}; process receipt={receipt}"
        )
    if tree["status"] == "timeout":
        raise subprocess.TimeoutExpired(command, timeout)
    if tree["status"] not in {"completed", "failed-parent"}:
        raise RuntimeError(f"process tree ended with {tree['status']}")
    return tree["returncode"]


class OutputLease:
    """Exclude cross-caller build/Cargo readers and writers, or the shared GPU.

    Build leases precede the GPU lease. An unproved process tree retains every
    enclosing lease; owner death never implies recovery. Delegated child commands
    may borrow the exact token, while unrelated callers wait for release.
    """

    held = {}

    def __init__(self, root: Path, build: Path | None = None, *, timeout=None):
        self.root = root.resolve()
        key = (
            "gpu"
            if build is None
            else "build-"
            + hashlib.sha256(build.resolve().relative_to(self.root).as_posix().casefold().encode()).hexdigest()[:24]
        )
        self.path = self.root / "out/verification-locks" / key
        self.timeout = timeout
        self.blocked = False
        self.role = "owned"
        self.wait_seconds = 0.0

    def check_admission(self):
        try:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            if self.blocked or owner["status"] != "owned" or owner["token"] != self.token:
                raise RuntimeError(
                    "output owner is blocked or changed; establish drainage before recovery: " + str(self.path)
                )
        except BaseException:
            self.blocked = True
            raise

    @classmethod
    def check_held(cls):
        for lease in tuple(cls.held.values()):
            lease.check_admission()

    def __enter__(self):
        import uuid

        self.check_held()
        started = time.monotonic()
        key = hashlib.sha256(os.path.normcase(str(self.path)).encode()).hexdigest()
        inherited = json.loads(os.environ.get("JUICER_OUTPUT_LEASES", "{}"))
        if self.path in self.held:
            self.role = "nested"
            return self.held[self.path]
        if key in inherited:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            if owner["token"] != inherited[key]:
                raise ValueError("output lease delegation changed")
            self.token = inherited[key]
            self.check_admission()
            self.role = "delegated"
            self.held[self.path] = self
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                try:
                    owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    owner = {}
                if owner.get("status") == "blocked":
                    raise RuntimeError(f"output owner is blocked; inspect drainage before recovery: {self.path}")
                if self.timeout is not None and time.monotonic() - started >= self.timeout:
                    raise TimeoutError(f"output remains owned; inspect drainage before recovery: {self.path}")
                time.sleep(0.05)
        self.wait_seconds = time.monotonic() - started
        self.token = str(uuid.uuid4())
        write_json(
            self.path / "owner.json",
            {
                "owner_pid": os.getpid(),
                "token": self.token,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "status": "owned",
            },
        )
        self.previous = os.environ.get("JUICER_OUTPUT_LEASES")
        os.environ["JUICER_OUTPUT_LEASES"] = json.dumps({**inherited, key: self.token}, sort_keys=True)
        self.held[self.path] = self
        return self

    def __exit__(self, *_):
        if self.role == "nested":
            return
        self.held.pop(self.path)
        owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
        self.blocked |= owner["status"] == "blocked"
        if self.role == "delegated":
            if self.blocked:
                write_json(self.path / "owner.json", {**owner, "status": "blocked"})
            return
        if self.previous is None:
            os.environ.pop("JUICER_OUTPUT_LEASES", None)
        else:
            os.environ["JUICER_OUTPUT_LEASES"] = self.previous
        if self.blocked:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            write_json(self.path / "owner.json", {**owner, "status": "blocked"})
        else:
            (self.path / "owner.json").unlink()
            self.path.rmdir()


class NativeRun:
    """Commands for one continuously owned native build generation.

    No historic receipts, tool hashing or reuse. The caller keeps OutputLease
    across configuration, build, tests and completed-result publication.
    """

    def __init__(self, root: Path, directory: Path):
        self.root = root
        self.directory = directory
        self.commands = []

    def command(self, name, command, *, env=None, timeout=None, vctip_executable=None, stdout_only=False, cwd=None):
        OutputLease.check_held()
        cleanup = None
        if vctip_executable is not None:
            path = Path(vctip_executable).resolve(strict=True)
            if os.name != "nt" or path.name.lower() != "vctip.exe":
                raise ValueError("native compiler cleanup requires the selected Windows VCTIP.EXE")
            # Only CMake callers opt in. The existing supervisor waits for the
            # command to exit and confines exact-path telemetry cleanup to its job.
            cleanup = {"executable": str(path), "grace_seconds": 1, "drain_cap_seconds": None}
        name_component(name)
        log = self.directory / f"{name}.log"
        process_record = self.directory / f"{name}.process.json"
        lock = self.directory / f"{name}.running.lock"
        tree = {"drained": True}
        record = {
            "name": name,
            "command": list(map(str, command)),
            "directory": str(cwd or self.root),
            "returncode": None,
        }
        if cleanup:
            record["vctip_cleanup"] = cleanup
        self.commands.append(record)
        started = time.perf_counter()
        try:
            from contextlib import ExitStack

            with ExitStack() as streams:
                stream = streams.enter_context(log.open("xb"))
                errors = (
                    streams.enter_context((self.directory / f"{name}.stderr.log").open("xb")) if stdout_only else None
                )
                options = {}
                if cleanup:
                    options["vctip_cleanup"] = cleanup
                if errors is not None:
                    options["stderr_output"] = errors
                record["returncode"] = run_process_tree(
                    record["command"],
                    cwd or self.root,
                    command_environment(os.environ if env is None else env),
                    stream,
                    timeout,
                    process_record,
                    lock,
                    tree,
                    **options,
                )
            if record["returncode"]:
                raise subprocess.CalledProcessError(record["returncode"], command)
            return log
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["output_bytes"] = log.stat().st_size if log.exists() else 0
            if stdout_only:
                errors = self.directory / f"{name}.stderr.log"
                record["stderr_bytes"] = errors.stat().st_size if errors.exists() else 0
            record.update(
                seconds=time.perf_counter() - started,
                drained=tree["drained"],
                process_status=tree.get("status"),
                command_seconds=tree.get("command_seconds"),
                drain_seconds=tree.get("drain_seconds"),
            )
            if "resource_usage" in tree:
                record["resource_usage"] = tree["resource_usage"]
            if "resource_usage_error" in tree:
                record["resource_usage_error"] = tree["resource_usage_error"]
            if cleanup:
                record["vctip_terminated"] = tree.get("vctip_terminated", [])
            if tree["drained"]:
                lock.unlink(missing_ok=True)
            else:
                for lease in OutputLease.held.values():
                    lease.blocked = True
                write_json(lock, {"status": "blocked", **tree})

    def publish(self, result):
        OutputLease.check_held()
        if any(not c["drained"] for c in self.commands):
            raise RuntimeError("cannot publish while descendants remain unproved")
        target = self.directory / "result.json"
        if target.exists():
            raise ValueError("completed result already exists")
        write_json(target, result)


def _preset_owner(root, preset):
    if preset not in {"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"}:
        raise ValueError("unsupported native preset")
    requested = OutputLease(Path(root), Path(root) / "out/build" / preset, timeout=0)
    owner = OutputLease.held.get(requested.path)
    if owner is None:
        raise RuntimeError("closeout and expiry require the exact preset owner")
    owner.check_admission()
    return owner


def _plain_path(root, path, *, directory=False, missing=False):
    """Check each existing component without following a link or Windows reparse."""
    parts = path.absolute().relative_to(root).parts
    if not parts or ".." in parts:
        raise ValueError("unconfined closeout path")
    current = root
    for index, part in enumerate(parts):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing and (directory or index == len(parts) - 1):
                return None
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked or reparse closeout path: " + str(current))
        is_directory = index < len(parts) - 1 or directory
        if not (stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode)):
            raise ValueError("unexpected closeout path type: " + str(current))
    return info


def _read_closeout_json(root, path, costs):
    _plain_path(root, path)
    data = path.read_bytes()
    costs["read_bytes"] += len(data)
    return json.loads(data), data


def _write_closeout_json(root, path, value):
    _plain_path(root, path, missing=True)
    _plain_path(root, path.with_suffix(".tmp"), missing=True)
    write_json(path, value)
    return path.stat().st_size


def _closeout_state(value, result_bytes):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "native-closeout-v1"
        or value.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest()
        or type(value.get("accepted")) is not bool
        or value.get("review") not in {"open", "closed"}
    ):
        raise ValueError("invalid or mismatched closeout")
    closed = value.get("closed_at")
    if closed is not None and (type(closed) not in (int, float) or not math.isfinite(closed) or closed < 0):
        raise ValueError("invalid closeout time")
    if (closed is not None) != (value["accepted"] and value["review"] == "closed"):
        raise ValueError("incomplete acceptance/review closeout")
    holds = value.get("holds")
    if (
        not isinstance(holds, list)
        or any(hold not in {"review", "failure", "reproduction"} for hold in holds)
        or len(holds) != len(set(holds))
    ):
        raise ValueError("invalid closeout holds")
    disposition = value.get("disposition", {})
    if (
        not isinstance(disposition, dict)
        or type(disposition.get("deleted_bytes", 0)) is not int
        or disposition.get("deleted_bytes", 0) < 0
    ):
        raise ValueError("invalid closeout disposition")
    if disposition and disposition.get("status") not in {"pending", "deleted", "absent", "failed"}:
        raise ValueError("invalid closeout disposition status")
    if disposition.get("status") == "pending" and (
        type(disposition.get("pending_bytes")) is not int or disposition["pending_bytes"] < 0
    ):
        raise ValueError("invalid pending diagnostic size")
    return value


def _expiry_result(root, directory, result, preset):
    if (
        not isinstance(result, dict)
        or result.get("schema")
        not in {"native-tests-v1", "native-quality-v1", "native-build-v1", "native-spatial-optics-v1"}
        or result.get("preset") != preset
    ):
        raise ValueError("unsupported closeout result")
    commands = result.get("commands")
    if not isinstance(commands, list) or any(not isinstance(command, dict) for command in commands):
        raise ValueError("invalid closeout command metadata")
    if (
        result.get("status") != "passed"
        or result.get("candidate_unchanged") is not True
        or not commands
        or any(
            command.get("drained") is not True
            or command.get("returncode") != 0
            or command.get("process_status") != "completed"
            for command in commands
        )
    ):
        return False
    members = result.get("members", {})
    if not isinstance(members, dict) or not {"candidate.json", "candidate.patch"} <= members.keys():
        raise ValueError("missing mandatory closeout members")
    for name, member in members.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("unconfined mandatory closeout member")
        if not isinstance(member, dict) or type(member.get("bytes")) is not int or member["bytes"] < 0:
            raise ValueError("invalid mandatory closeout member")
        try:
            info = _plain_path(root, directory / name)
        except FileNotFoundError:
            raise ValueError("missing mandatory closeout member: " + name) from None
        if info.st_size != member.get("bytes"):
            raise ValueError("missing or incomplete mandatory closeout member: " + name)
    # Publication has exactly one optional stream. Do not generalize this into
    # an arbitrary result-directed file deletion API.
    if result.get("diagnostics") != {"path": "diagnostics.gz", "optional": True} or "diagnostics.gz" in members:
        raise ValueError("invalid optional diagnostic declaration")
    return True


def closeout_run(root, preset, run_id, *, accept=False, review=None, hold=(), release_hold=(), now=None):
    """Mutate review/hold metadata under the same owner used by expiry/builds."""
    owner = _preset_owner(root, preset)
    root = owner.root
    directory = root / "out/validation" / preset / "results" / name_component(run_id)
    _plain_path(root, directory, directory=True)
    costs = {"read_bytes": 0}
    result, raw = _read_closeout_json(root, directory / "result.json", costs)
    eligible = _expiry_result(root, directory, result, preset)
    path = directory / "closeout.json"
    if _plain_path(root, path, missing=True) is None:
        state = {
            "schema": "native-closeout-v1",
            "result_sha256": hashlib.sha256(raw).hexdigest(),
            "accepted": False,
            "review": "open",
            "closed_at": None,
            "holds": [],
        }
    else:
        state = _closeout_state(_read_closeout_json(root, path, costs)[0], raw)
    if review is not None and review not in {"open", "closed"}:
        raise ValueError("review must be open or closed")
    if any(value not in {"review", "failure", "reproduction"} for value in (*hold, *release_hold)):
        raise ValueError("unknown closeout hold")
    if set(hold) & set(release_hold):
        raise ValueError("cannot acquire and release the same hold")
    if accept and not eligible:
        raise ValueError("cannot accept a failed, changed or undrained result")
    state["accepted"] |= accept
    if review is not None:
        state["review"] = review
    state["holds"] = sorted((set(state["holds"]) | set(hold)) - set(release_hold))
    if state["accepted"] and state["review"] == "closed":
        if state["closed_at"] is None:
            state["closed_at"] = time.time() if now is None else now
    else:
        state["closed_at"] = None
    _closeout_state(state, raw)
    written = _write_closeout_json(root, path, state)
    return {"run": run_id, "closeout": state, "write_bytes": written, **costs}


def expire_diagnostics(root, preset, *, now=None):
    """Inspect at most 32 direct runs; never read payloads or a legacy tree."""
    started = time.perf_counter()
    owner = _preset_owner(root, preset)
    root = owner.root
    summary = {
        "visited": 0,
        "directory_entries": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "deleted_bytes": 0,
        "deleted_runs": 0,
        "errors": [],
        "more": False,
    }
    runs = root / "out/validation" / preset / "results"
    # Missing roots are ordinary on the first invocation; create no history tree.
    if _plain_path(root, runs, directory=True, missing=True) is None:
        summary["seconds"] = time.perf_counter() - started
        return summary
    cursor_path = owner.path.with_suffix(".expiry.json")
    after = ""
    if _plain_path(root, cursor_path, missing=True) is not None:
        cursor, _ = _read_closeout_json(root, cursor_path, summary)
        if (
            not isinstance(cursor, dict)
            or cursor.get("schema") != "native-expiry-cursor-v1"
            or not isinstance(cursor.get("after"), str)
        ):
            raise ValueError("invalid diagnostic expiry cursor")
        after = cursor["after"]
    ready = not after
    last = ""
    timestamp = time.time() if now is None else now
    # Stream names to the previous entry, then visit one batch. No sorted listing,
    # census or payload read is needed. A removed cursor restarts on the next pass.
    with os.scandir(runs) as entries:
        for entry in entries:
            summary["directory_entries"] += 1
            if not ready:
                ready = entry.name == after
                continue
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", entry.name):
                continue  # Report-only JSON files are not native run directories.
            if summary["visited"] == 32:
                summary["more"] = True
                break
            last = entry.name
            summary["visited"] += 1
            directory = runs / entry.name
            path = directory / "closeout.json"
            try:
                _plain_path(root, directory, directory=True)
                if _plain_path(root, path, missing=True) is None:
                    continue  # No closeout means no inspection of old result bytes.
                state, _ = _read_closeout_json(root, path, summary)
                result, raw = _read_closeout_json(root, directory / "result.json", summary)
                state = _closeout_state(state, raw)
                if state["closed_at"] is None or state["holds"] or timestamp < state["closed_at"] + 7 * 24 * 60 * 60:
                    continue
                if not _expiry_result(root, directory, result, preset):
                    continue
                target = directory / "diagnostics.gz"
                info = _plain_path(root, target, missing=True)
                disposition = state.get("disposition", {})
                previous = disposition.get("deleted_bytes", 0)
                if info is None and disposition.get("status") == "pending":
                    previous += disposition["pending_bytes"]
                try:
                    if info is not None:
                        # Persist intent first so interruption after unlink does
                        # not lose the measured byte count on a later retry.
                        state["disposition"] = {
                            "status": "pending",
                            "deleted_bytes": previous,
                            "pending_bytes": info.st_size,
                        }
                        summary["write_bytes"] += _write_closeout_json(root, path, state)
                        owner.check_admission()
                        _plain_path(root, target)  # Recheck confinement at deletion.
                        target.unlink()
                        summary["deleted_bytes"] += info.st_size
                        summary["deleted_runs"] += 1
                    state["disposition"] = {
                        "status": "deleted" if info is not None or previous else "absent",
                        "deleted_bytes": previous + (info.st_size if info is not None else 0),
                    }
                except OSError as error:
                    state["disposition"] = {"status": "failed", "deleted_bytes": previous, "error": str(error)}
                    summary["errors"].append({"run": entry.name, "error": str(error)})
                summary["write_bytes"] += _write_closeout_json(root, path, state)
            except (OSError, ValueError, KeyError, TypeError) as error:
                summary["errors"].append({"run": entry.name, "error": str(error)})
    summary["cursor_reset"] = not ready
    summary["more"] |= not ready
    summary["write_bytes"] += _write_closeout_json(
        root, cursor_path, {"schema": "native-expiry-cursor-v1", "after": last if summary["more"] else ""}
    )
    summary["seconds"] = time.perf_counter() - started
    print("Diagnostic expiry:", json.dumps(summary, sort_keys=True))
    return summary
