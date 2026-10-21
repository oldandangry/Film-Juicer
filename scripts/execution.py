"""Small, sequential command receipts for implementation verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


class UnresolvedScope(ValueError):
    """The requested verification scope has no complete supported selection."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot(root: Path, paths: Sequence[str]) -> dict[str, str]:
    """Hash declared files only. Never discover previous campaigns recursively."""
    names = sorted(set(paths))
    if not names:
        return {}
    resolved_root = root.resolve()
    result = {}
    for name in names:
        path = root / name
        path.resolve().relative_to(resolved_root)
        if not path.is_file():
            raise ValueError(f"declared input is not a file: {name}")
        result[name] = digest(path)
        path.resolve().relative_to(resolved_root)
    return result


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def name_component(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise ValueError(f"expected a lowercase directory name: {value}")
    return value


def check_windows_launcher(executable: Path) -> None:
    """App execution aliases activate through a broker outside Job inheritance."""
    paths = [executable]
    if executable.name.lower().startswith("python"):
        for directory in (executable.parent, executable.parent.parent):
            config = directory / "pyvenv.cfg"
            if config.is_file():
                for line in config.read_text(encoding="utf-8").splitlines():
                    key, separator, value = line.partition("=")
                    if separator and key.strip() == "home":
                        paths.append(Path(value.strip()) / "python.exe")
                break
    for path in paths:
        if (
            getattr(path.lstat(), "st_reparse_tag", 0) == 0x8000001B
        ):  # IO_REPARSE_TAG_APPEXECLINK
            raise ValueError(
                f"brokered app execution alias cannot be owned by the campaign job: {path}; use the installed executable directly"
            )


def python_command(
    script: Path, *arguments: str, python: str = sys.executable
) -> list[str]:
    """Run a file with its sibling imports, including under embedded Python."""
    script = script.resolve(strict=True)
    bootstrap = (
        "import runpy,sys; from pathlib import Path; "
        "p=sys.argv.pop(1); sys.path.insert(0,str(Path(p).parent)); "
        "sys.argv[0]=p; runpy.run_path(p,run_name='__main__')"
    )
    return [python, "-B", "-c", bootstrap, str(script), *arguments]


def rustc_command(
    arguments: Sequence[str], env: Mapping[str, str] | None = None
) -> list[str]:
    """Direct probes use the same scoped Windows linker as Cargo and CMake."""
    command = ["rustc", *arguments]
    if os.name == "nt":
        environment = os.environ if env is None else env
        linker = environment.get("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER", "")
        if not linker or not Path(linker).is_file():
            raise ValueError(
                "run Windows probes through with-windows-toolchain.ps1; qualified linker is missing"
            )
        options = [argument.removeprefix("-C") for argument in arguments]
        if any(option.startswith("linker=") for option in options):
            raise ValueError(
                "the shared launcher owns the linker; remove the per-command linker override"
            )
        command[1:1] = ["-C", f"linker={linker}"]
    return command


def tool_snapshot(paths: Sequence[str], *, allow_missing=False) -> dict[str, str | None]:
    """Explicit tools/configuration files; checkpoints can bind absent configs."""
    result = {}
    for name in sorted(set(paths)):
        path = Path(name).resolve(strict=not allow_missing)
        if allow_missing and not path.exists():
            result[str(path)] = None
        elif not path.is_file():
            raise ValueError("tool/configuration input is not a file: " + str(path))
        else:
            result[str(path)] = digest(path)
    return result



def run_process_tree(
    command, root, environment, output, timeout, receipt, lock, tree,
    stderr_output=None, vctip_cleanup=None,
):
    OutputLease.check_held()
    supervisor = Path(__file__).with_name("process_tree.py")
    policy = ["--vctip-cleanup", json.dumps(vctip_cleanup)] if vctip_cleanup else []
    process = subprocess.Popen(
        [sys.executable, "-B", str(supervisor), str(receipt), str(timeout), *policy, *command],
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=output,
        stderr=subprocess.STDOUT if stderr_output is None else stderr_output,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    tree.update(drained=False, supervisor_pid=process.pid, receipt=str(receipt))
    try:
        write_json(lock, {"owner_pid": os.getpid(), **tree})
        try:
            process.wait(timeout=timeout + 15)
        except BaseException:
            process.stdin.close()  # EOF requests termination and complete draining.
            process.wait(timeout=15)
            raise
    finally:
        process.stdin.close()
        if receipt.is_file():
            tree.update(json.loads(receipt.read_text(encoding="utf-8")))
    if process.returncode or not tree["drained"]:
        raise RuntimeError(
            "process-tree completion could not be established; campaign remains locked; "
            f"supervisor exit={process.returncode}, status={tree.get('status', 'unreported')}, "
            f"error={tree.get('error', 'unreported')}; process receipt={receipt}"
        )
    if tree["status"] == "timeout":
        raise subprocess.TimeoutExpired(command, timeout)
    if tree["status"] not in {"completed", "failed-parent"}:
        raise RuntimeError(f"process tree ended with {tree['status']}")
    return tree["returncode"]


class Campaign:
    """One owner, declared gates, immutable attempts; no automatic cached passes."""

    def __init__(
        self, root: Path, preset: str, name: str, required: Sequence[str]
    ) -> None:
        self.root = root.resolve()
        if preset not in {
            "linux-debug",
            "linux-release",
            "windows-clang-debug",
            "windows-clang-release",
        }:
            raise ValueError(f"unsupported preset: {preset}")
        self.directory = self.root / "out/validation" / preset / name_component(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.required = list(required)
        if not self.required or len(set(self.required)) != len(self.required):
            raise ValueError("required gates must be nonempty and unique")
        for stage in self.required:
            name_component(stage)
        contract = {"root": str(self.root), "preset": preset, "required": self.required}
        path = self.directory / "campaign.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(contract, indent=2) + "\n")
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != contract:
                raise ValueError(
                    "existing campaign has a different root, preset or required gate list"
                )

    def attempts(self, stage: str) -> list[dict]:
        name_component(stage)
        paths = list((self.directory / "attempts").glob(f"{stage}-*/result.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        return sorted(
            (record for record in records if record["stage"] == stage),
            key=lambda record: (record.get("attempt_number", 0), record["started_at"]),
        )

    def check_run_request(
        self, stage: str, command: Sequence[str], *, inputs: Sequence[str], timeout: float
    ) -> None:
        """Reject invalid requests before callers create pending ownership or reuse results."""
        name_component(stage)
        if (
            stage not in self.required
            or not command
            or not inputs
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "a declared stage, command, input files and finite positive timeout are required"
            )

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        inputs: Sequence[str],
        timeout: float,
        env: Mapping[str, str] | None = None,
        repeat_reason: str = "",
        tools: Sequence[str] = (),
        expected_inputs: Mapping[str, str] | None = None,
        expected_tools: Mapping[str, str] | None = None,
        vctip_executable: str | None = None,
        vctip_grace_seconds: float = 30,
        vctip_drain_cap_seconds: float = 300,
    ) -> Path:
        OutputLease.check_held()
        self.check_run_request(stage, command, inputs=inputs, timeout=timeout)
        environment = dict(os.environ if env is None else env)
        if "JUICER_OUTPUT_LEASES" in os.environ:
            environment["JUICER_OUTPUT_LEASES"] = os.environ["JUICER_OUTPUT_LEASES"]
        executable = shutil.which(command[0], path=environment.get("PATH", ""))
        if executable is None:
            raise ValueError(f"command is unavailable: {command[0]}")
        if os.name == "nt":
            check_windows_launcher(Path(executable))
        vctip_cleanup = None
        if vctip_executable is not None:
            if (
                os.name != "nt"
                or Path(vctip_executable).name.lower() != "vctip.exe"
                or not math.isfinite(vctip_grace_seconds)
                or vctip_grace_seconds < 0
                or not math.isfinite(vctip_drain_cap_seconds)
                or vctip_drain_cap_seconds <= vctip_grace_seconds
            ):
                raise ValueError(
                    "VCTIP cleanup requires Windows, VCTIP.EXE and finite drain cap greater than grace"
                )
            vctip_path = Path(vctip_executable).resolve(strict=True)
            vctip_cleanup = {
                "executable": str(vctip_path),
                "grace_seconds": vctip_grace_seconds,
                "drain_cap_seconds": vctip_drain_cap_seconds,
            }
        elif vctip_grace_seconds != 30 or vctip_drain_cap_seconds != 300:
            raise ValueError("VCTIP drain policy requires a VCTIP executable")
        preparation_start = time.perf_counter()
        selected_tools = [*tools, *([str(vctip_path)] if vctip_cleanup else [])]
        identity = {
            "command": list(command),
            "inputs": snapshot(self.root, inputs),
            "tools": tool_snapshot(selected_tools),
            "supervisor_sha256": digest(Path(__file__).with_name("process_tree.py")),
            "executable_sha256": digest(Path(executable)),
            "environment_sha256": hashlib.sha256(
                json.dumps(environment, sort_keys=True).encode()
            ).hexdigest(),
        }
        if vctip_cleanup:
            identity["vctip_cleanup"] = vctip_cleanup
        preparation_seconds = time.perf_counter() - preparation_start
        if any(
            identity["inputs"].get(name) != value
            for name, value in (expected_inputs or {}).items()
        ):
            raise ValueError(
                "prepared inputs changed before launch; repeat preparation"
            )
        if any(
            identity["tools"].get(name) != value
            for name, value in (expected_tools or {}).items()
        ):
            raise ValueError("prepared tools changed before launch; repeat preparation")
        lock = self.directory / "running.lock"
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} stage={stage}\n")
        tree = {"drained": True}
        try:
            prior = self.attempts(stage)
            if (
                prior
                and prior[-1]["status"] == "passed"
                and prior[-1]["identity"] == identity
                and not repeat_reason.strip()
            ):
                raise ValueError(
                    "stage already passed for these inputs; reference its receipt or supply a repeat reason"
                )
            attempts = self.directory / "attempts"
            attempts.mkdir(exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=stage + "-", dir=attempts))
            record = {
                "stage": stage,
                "identity": identity,
                "status": "running",
                "returncode": None,
                "attempt_number": max(
                    (item.get("attempt_number", 0) for item in prior), default=0
                )
                + 1,
                "repeat_reason": repeat_reason,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "directory": str(self.root),
                "timeout": timeout,
                "timings": {"initial_hash_seconds": preparation_seconds},
                "output": (directory / "output.log").relative_to(self.root).as_posix(),
            }
            receipt = directory / "result.json"
            write_json(receipt, record)
            start = time.perf_counter()
            try:
                with (directory / "output.log").open("xb") as output:
                    arguments = [
                        list(command), self.root, environment, output, timeout,
                        directory / "process.json", lock, tree,
                    ]
                    if vctip_cleanup:
                        record["returncode"] = run_process_tree(
                            *arguments, vctip_cleanup=vctip_cleanup,
                        )
                    else:
                        record["returncode"] = run_process_tree(*arguments)
                if record["returncode"]:
                    raise subprocess.CalledProcessError(record["returncode"], command)
                OutputLease.check_held()
                verification_start = time.perf_counter()
                unchanged = (
                    snapshot(self.root, inputs) == identity["inputs"]
                    and tool_snapshot(selected_tools) == identity["tools"]
                )
                record["timings"]["final_input_hash_seconds"] = (
                    time.perf_counter() - verification_start
                )
                if not unchanged:
                    raise ValueError(
                        "declared inputs changed during the command; result cannot qualify them"
                    )
                record["status"] = "passed"
            except BaseException as exc:
                record.update(
                    status="failed" if tree["drained"] else "blocked", error=str(exc)
                )
                raise
            finally:
                record.update(
                    seconds=time.perf_counter() - start,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                record["process_tree"] = tree
                if tree["drained"]:
                    output_start = time.perf_counter()
                    record["output_sha256"] = digest(directory / "output.log")
                    record["timings"]["output_hash_seconds"] = (
                        time.perf_counter() - output_start
                    )
                write_json(receipt, record)
            return receipt
        finally:
            if tree["drained"]:
                lock.unlink()
            else:
                for lease in OutputLease.held.values():
                    lease.blocked = True
                write_json(
                    lock,
                    {
                        "owner_pid": os.getpid(),
                        "stage": stage,
                        "status": "blocked",
                        **tree,
                    },
                )

    def report(self) -> dict:
        """Report evidence, never rerun a command or certify undeclared gates."""
        required = tuple(self.required)
        locked = (self.directory / "running.lock").exists()
        latest = {}
        stages = {}
        for stage in required:
            records = self.attempts(stage)
            latest[stage] = records[-1] if records else None
            stages[stage] = records[-1]["status"] if records else "outstanding"

        def read_files(records):
            local = set()
            tools = set()
            for record in records.values():
                local.update(record["identity"]["inputs"])
                local.add(record["output"])
                local.add(Path(record["output"]).with_name("result.json").as_posix())
                tools.update(record["identity"].get("tools", {}))
            try:
                resolved_root = self.root.resolve()
            except (OSError, ValueError):
                resolved_root = None
            values = {}
            for is_tool, names in ((False, local), (True, tools)):
                for name in sorted(names):
                    path = Path(name) if is_tool else self.root / name
                    try:
                        resolved = path.resolve(strict=is_tool)
                        if not is_tool:
                            if resolved_root is None:
                                raise ValueError("campaign root is unavailable")
                            resolved.relative_to(resolved_root)
                            if not path.is_file():
                                raise ValueError(
                                    f"declared input is not a file: {name}"
                                )
                        values[is_tool, name] = (str(resolved), digest(path))
                    except (OSError, ValueError):
                        values[is_tool, name] = None
            return values

        def status(record, values, previous=None):
            # Preserve stage-specific expectations and input/tool/output precedence.
            for is_tool, expected, changed in (
                (False, record["identity"]["inputs"], "inputs-changed"),
                (True, record["identity"].get("tools", {}), "inputs-changed"),
                (
                    False,
                    {record["output"]: record["output_sha256"]},
                    "evidence-changed",
                ),
            ):
                if any(values[is_tool, name] is None for name in expected):
                    return "inputs-or-evidence-unavailable"
                for name, checksum in expected.items():
                    current = values[is_tool, name]
                    if (
                        current[1] != checksum
                        or (is_tool and current[0] != name)
                        or (previous is not None and current != previous[is_tool, name])
                    ):
                        return changed
            receipt = Path(record["output"]).with_name("result.json").as_posix()
            if values[False, receipt] is None:
                return "inputs-or-evidence-unavailable"
            if (
                previous is not None
                and values[False, receipt] != previous[False, receipt]
            ):
                return "evidence-changed"
            return "passed"

        candidates = {
            stage: record
            for stage, record in latest.items()
            if stages[stage] == "passed"
        }
        initial = read_files(candidates)
        for stage, record in candidates.items():
            stages[stage] = status(record, initial)
        candidates = {
            stage: record
            for stage, record in candidates.items()
            if stages[stage] == "passed"
        }
        # This is a second fresh read, not reuse of the entry boundary's hashes.
        final = read_files(candidates)
        for stage, record in candidates.items():
            stages[stage] = status(record, final, initial)
        for stage in required:
            try:
                records = self.attempts(stage)
                if (records[-1] if records else None) != latest[stage]:
                    stages[stage] = "evidence-changed"
            except (OSError, ValueError):
                stages[stage] = "inputs-or-evidence-unavailable"
        if tuple(self.required) != required:
            stages = {stage: "evidence-changed" for stage in required}
        locked = locked or (self.directory / "running.lock").exists()
        return {
            "stages": stages,
            "locked": locked,
            "all_recorded_gates_passed": not locked
            and all(value == "passed" for value in stages.values()),
        }


class OutputLease:
    """Exclude cross-caller build/Cargo readers and writers, or the shared GPU.

    Build leases precede the GPU lease. An unproved process tree retains every
    enclosing lease; owner death never implies recovery. Delegated child commands
    may borrow the exact token, while unrelated callers wait for release.
    """

    held = {}

    def __init__(self, root: Path, build: Path | None = None, *, timeout=30):
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
                if time.monotonic() - started >= self.timeout:
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
