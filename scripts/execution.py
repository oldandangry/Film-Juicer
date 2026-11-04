"""Native command containment, output ownership and result publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    command, root, environment, output, timeout, receipt, lock, tree,
    stderr_output=None, vctip_cleanup=None,
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
            + hashlib.sha256(
                build.resolve().relative_to(self.root).as_posix().casefold().encode()
            ).hexdigest()[:24]
        )
        self.path = self.root / "out/verification-locks" / key
        self.timeout = timeout
        self.blocked = False
        self.role = "owned"
        self.wait_seconds = 0.0

    def check_admission(self):
        try:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            if (
                self.blocked
                or owner["status"] != "owned"
                or owner["token"] != self.token
            ):
                raise RuntimeError(
                    "output owner is blocked or changed; establish drainage before recovery: "
                    + str(self.path)
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
                    raise TimeoutError(
                        f"output remains owned; inspect drainage before recovery: {self.path}"
                    )
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
        os.environ["JUICER_OUTPUT_LEASES"] = json.dumps(
            {**inherited, key: self.token}, sort_keys=True
        )
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
        record = {"name": name, "command": list(map(str, command)), "directory": str(cwd or self.root), "returncode": None}
        if cleanup:
            record["vctip_cleanup"] = cleanup
        self.commands.append(record)
        started = time.perf_counter()
        try:
            from contextlib import ExitStack
            with ExitStack() as streams:
                stream = streams.enter_context(log.open("xb"))
                errors = streams.enter_context((self.directory / f"{name}.stderr.log").open("xb")) if stdout_only else None
                options = {}
                if cleanup:
                    options["vctip_cleanup"] = cleanup
                if errors is not None:
                    options["stderr_output"] = errors
                record["returncode"] = run_process_tree(
                    record["command"], cwd or self.root,
                    command_environment(os.environ if env is None else env),
                    stream, timeout, process_record, lock, tree,
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
            record.update(seconds=time.perf_counter() - started,
                          drained=tree["drained"], process_status=tree.get("status"),
                          command_seconds=tree.get("command_seconds"),
                          drain_seconds=tree.get("drain_seconds"))
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
