"""Small, sequential command receipts for implementation verification."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence


COMMAND_TRANSCRIPT_PROTOCOL = "command-transcript-v1"
COMMAND_TRANSCRIPT_GZIP_THRESHOLD = 256 * 1024
COMMAND_RECEIPT_PROTOCOL = "command-receipt-v2"
COMMAND_IDENTITY_MAP_PROTOCOL = "command-identity-map-v1"
_TRANSCRIPT_REFERENCE_FIELDS = frozenset({
    "protocol", "path", "encoding", "encoded_bytes", "encoded_sha256",
    "decoded_bytes", "decoded_sha256",
})
_FILE_OBSERVATIONS = ContextVar("verification_file_observations", default=None)


class UnresolvedScope(ValueError):
    """The requested verification scope has no complete supported selection."""


def _file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class FileObservations:
    """Share file reads only within one explicitly owned freshness boundary."""

    def __init__(self):
        self.bindings = {}
        self.digests = {}
        self.payloads = {}
        self.documents = {}
        self.content_bytes = 0
        self.content_reads = 0
        self.transcript_decoded_bytes = 0
        self.used = False

    def __enter__(self):
        if self.used:
            raise ValueError("file observations cannot cross freshness boundaries")
        self.used = True
        self.token = _FILE_OBSERVATIONS.set(self)
        return self

    def __exit__(self, kind, value, traceback):
        _FILE_OBSERVATIONS.reset(self.token)
        self.payloads.clear()
        self.documents.clear()

    def _binding(self, path):
        if _FILE_OBSERVATIONS.get() is not self:
            raise ValueError("file observations are outside their freshness boundary")
        path = Path(path)
        lexical = str(path.absolute())
        resolved = str(path.resolve(strict=True))
        if self.bindings.setdefault(lexical, resolved) != resolved:
            raise ValueError("input path changed during file observation")
        return path, resolved

    def digest(self, path):
        path, resolved = self._binding(path)
        if resolved not in self.digests:
            checksum = hashlib.sha256()
            parts = []
            size = 0
            self.content_reads += 1
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    checksum.update(block)
                    size += len(block)
                    self.content_bytes += len(block)
                    if parts is not None:
                        if size <= 1024 * 1024:
                            parts.append(block)
                        else:
                            parts = None
            if self._binding(path)[1] != resolved:
                raise ValueError("input path changed while reading")
            self.digests[resolved] = checksum.hexdigest()
            if parts is not None:
                self.payloads[resolved] = b"".join(parts)
        return self.digests[resolved]

    def read_bytes(self, path):
        path, resolved = self._binding(path)
        if resolved not in self.payloads:
            payload = path.read_bytes()
            self.content_reads += 1
            self.content_bytes += len(payload)
            checksum = hashlib.sha256(payload).hexdigest()
            if (self._binding(path)[1] != resolved
                    or self.digests.get(resolved, checksum) != checksum):
                raise ValueError("input changed while reading file observation")
            self.digests[resolved] = checksum
            self.payloads[resolved] = payload
        return self.payloads[resolved]

    def read_json(self, path):
        _, resolved = self._binding(path)
        if resolved not in self.documents:
            from build_dependencies import identity_json
            self.documents[resolved] = identity_json(self.read_bytes(path).decode("utf-8"))
        return deepcopy(self.documents[resolved])

    def counts(self):
        return {"content_bytes": self.content_bytes, "content_reads": self.content_reads,
                "transcript_decoded_bytes": self.transcript_decoded_bytes}


def digest(path: Path) -> str:
    observations = _FILE_OBSERVATIONS.get()
    return observations.digest(path) if observations is not None else _file_digest(path)


def read_json(path: Path, *, resolve=Path):
    """Read metadata, resolving command references from its original logical path."""
    logical_path = path
    path = resolve(path)
    observations = _FILE_OBSERVATIONS.get()
    if observations is not None:
        document = observations.read_json(path)
        return _decode_command_receipt(document, logical_path, resolve=resolve, physical_receipt=path)
    from build_dependencies import identity_json
    path = Path(path)
    document = identity_json(read_text(path, encoding="utf-8"))
    return _decode_command_receipt(document, logical_path, resolve=resolve, physical_receipt=path)


def read_bytes(path: Path) -> bytes:
    observations = _FILE_OBSERVATIONS.get()
    return observations.read_bytes(path) if observations is not None else Path(path).read_bytes()


def read_text(path: Path, encoding=None, errors=None, newline=None) -> str:
    observations = _FILE_OBSERVATIONS.get()
    if observations is None:
        return Path(path).read_text(encoding=encoding, errors=errors, newline=newline)
    with io.TextIOWrapper(io.BytesIO(observations.read_bytes(path)), encoding=encoding,
                          errors=errors, newline=newline) as stream:
        return stream.read()


def write_transcript(path: Path, payload: bytes) -> dict:
    """Write one immutable command stream, compressing only large payloads."""
    if not isinstance(payload, bytes):
        raise TypeError("command transcript payload must be bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decoded_hash = hashlib.sha256(payload).hexdigest()
    encoded_hash = hashlib.sha256()
    encoded_bytes = 0
    encoding = "gzip" if len(payload) > COMMAND_TRANSCRIPT_GZIP_THRESHOLD else "identity"

    class EncodedSink:
        def write(self, block):
            nonlocal encoded_bytes
            written = stream.write(block)
            if written != len(block):
                raise OSError("short write while saving command transcript")
            encoded_hash.update(block)
            encoded_bytes += written
            return written

        def flush(self):
            return stream.flush()

    with path.open("wb") as stream:
        if encoding == "identity":
            EncodedSink().write(payload)
        else:
            sink = EncodedSink()
            with gzip.GzipFile(filename="", mode="wb", fileobj=sink, compresslevel=9, mtime=0) as compressed:
                for offset in range(0, len(payload), 64 * 1024):
                    compressed.write(payload[offset:offset + 64 * 1024])

    return {
        "protocol": COMMAND_TRANSCRIPT_PROTOCOL,
        "path": str(path),
        "encoding": encoding,
        "encoded_bytes": encoded_bytes,
        "encoded_sha256": encoded_hash.hexdigest(),
        "decoded_bytes": len(payload),
        "decoded_sha256": decoded_hash,
    }


def read_transcript(reference: Mapping, resolve=Path) -> bytes:
    """Read a transcript only after checking its strict encoded and decoded identity."""
    if not isinstance(reference, Mapping) or set(reference) != _TRANSCRIPT_REFERENCE_FIELDS:
        raise ValueError("invalid command transcript reference fields")
    if reference["protocol"] != COMMAND_TRANSCRIPT_PROTOCOL:
        raise ValueError("unsupported command transcript protocol")
    if reference["encoding"] not in {"identity", "gzip"}:
        raise ValueError("unsupported command transcript encoding")
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise ValueError("invalid command transcript path")
    for field in ("encoded_bytes", "decoded_bytes"):
        if type(reference[field]) is not int or reference[field] < 0:
            raise ValueError("invalid command transcript byte count: " + field)
    for field in ("encoded_sha256", "decoded_sha256"):
        if not isinstance(reference[field], str) or not re.fullmatch(r"[0-9a-f]{64}", reference[field]):
            raise ValueError("invalid command transcript digest: " + field)

    path = resolve(reference["path"])
    encoded_hash = hashlib.sha256()
    decoded_hash = hashlib.sha256()
    encoded_bytes = 0
    decoded_bytes = 0
    decoded_parts = []
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if reference["encoding"] == "gzip" else None

    def retain_decoded(decoded):
        nonlocal decoded_bytes
        observations = _FILE_OBSERVATIONS.get()
        if observations is not None:
            observations.transcript_decoded_bytes += len(decoded)
        if decoded_bytes + len(decoded) > reference["decoded_bytes"]:
            raise ValueError("command transcript decoded byte count exceeds reference")
        if decoded:
            decoded_parts.append(decoded)
            decoded_hash.update(decoded)
            decoded_bytes += len(decoded)

    try:
        observations = _FILE_OBSERVATIONS.get()
        source = io.BytesIO(observations.read_bytes(path)) if observations is not None else path.open("rb")
        with source as stream:
            while block := stream.read(64 * 1024):
                encoded_hash.update(block)
                encoded_bytes += len(block)
                if decompressor is None:
                    retain_decoded(block)
                else:
                    pending = block
                    while pending:
                        remaining = reference["decoded_bytes"] - decoded_bytes
                        output_limit = min(64 * 1024, remaining + 1)
                        decoded = decompressor.decompress(pending, output_limit)
                        retain_decoded(decoded)
                        if decompressor.unused_data:
                            raise ValueError("trailing data in command transcript gzip stream")
                        next_pending = decompressor.unconsumed_tail
                        if next_pending == pending and not decoded:
                            raise ValueError("command transcript gzip decoder made no progress")
                        pending = next_pending
            if decompressor is not None:
                while True:
                    remaining = reference["decoded_bytes"] - decoded_bytes
                    output_limit = min(64 * 1024, remaining + 1)
                    decoded = decompressor.decompress(b"", output_limit)
                    retain_decoded(decoded)
                    if decompressor.unused_data:
                        raise ValueError("trailing data in command transcript gzip stream")
                    if not decoded:
                        break
                retain_decoded(decompressor.flush(64 * 1024))
                if not decompressor.eof:
                    raise ValueError("truncated command transcript gzip stream")
                if decompressor.unused_data or decompressor.unconsumed_tail:
                    raise ValueError("incomplete command transcript gzip stream")
    except zlib.error as error:
        raise ValueError("corrupt command transcript gzip stream") from error

    if encoded_bytes != reference["encoded_bytes"]:
        raise ValueError("command transcript encoded byte count mismatch")
    if encoded_hash.hexdigest() != reference["encoded_sha256"]:
        raise ValueError("command transcript encoded digest mismatch")
    if decoded_bytes != reference["decoded_bytes"]:
        raise ValueError("command transcript decoded byte count mismatch")
    if decoded_hash.hexdigest() != reference["decoded_sha256"]:
        raise ValueError("command transcript decoded digest mismatch")
    return b"".join(decoded_parts)


def command_environment(environment: Mapping[str, str]) -> dict[str, str]:
    # The desktop call ID identifies orchestration, not a compiler/test input.
    # Remove it from child processes too, so an ignored value cannot affect output.
    return {key: value for key, value in environment.items() if key != "CODEX_TOOL_CALL_ID"}


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
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_logical_owner(receipt):
    name = str(receipt)
    receipt = PureWindowsPath(name) if PureWindowsPath(name).is_absolute() else Path(name).absolute()
    if (receipt.name != "result.json" or receipt.parent.parent.name != "attempts"
            or ".." in receipt.parts):
        raise ValueError("command receipt is outside its owning campaign attempts")
    return receipt.parent.parent.parent


def _command_campaign_root(receipt: Path, *, reading: bool) -> Path:
    _command_logical_owner(receipt)
    receipt = Path(receipt).absolute()
    if (receipt.name != "result.json" or receipt.parent.parent.name != "attempts"
            or receipt.parent.is_symlink() or receipt.parent.parent.is_symlink()
            or reading and receipt.is_symlink()):
        raise ValueError("command receipt is outside its owning campaign attempts")
    try:
        return receipt.parent.parent.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError("command receipt campaign owner is unavailable") from error


def _identity_map_bytes(values: Mapping) -> bytes:
    if not isinstance(values, Mapping) or any(
        not isinstance(name, str) or not isinstance(checksum, str)
        or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        for name, checksum in values.items()
    ):
        raise ValueError("command receipt identity map is malformed")
    document = {"protocol": COMMAND_IDENTITY_MAP_PROTOCOL, "values": dict(values)}
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _decode_command_receipt(document, receipt, *, resolve=Path, physical_receipt=None):
    if not isinstance(document, dict):
        return document
    protocol = document.get("protocol")
    command_envelope = "stage" in document and "identity" in document
    if protocol != COMMAND_RECEIPT_PROTOCOL and not command_envelope:
        return document
    if protocol not in (None, COMMAND_RECEIPT_PROTOCOL):
        raise ValueError("unsupported command receipt protocol")
    if protocol is None:
        identity = document.get("identity")
        if isinstance(identity, dict) and any(name.endswith("_ref") for name in identity):
            raise ValueError("command receipt references require an explicit storage protocol")
        if not isinstance(identity, dict) or not {"inputs", "tools"} <= set(identity):
            raise ValueError("legacy command receipt identity maps are incomplete")
        for name in ("inputs", "tools"):
            _identity_map_bytes(identity[name])
        return document

    decoded = deepcopy(document)
    identity = decoded.get("identity")
    if (not isinstance(identity, dict) or "inputs" in identity or "tools" in identity
            or not {"inputs_ref", "tools_ref"} <= set(identity)):
        raise ValueError("command receipt identity-map references are incomplete")
    if any(name.endswith("_ref") and name not in {"inputs_ref", "tools_ref"} for name in identity):
        raise ValueError("command receipt contains an unknown identity-map reference")
    owner = _command_logical_owner(receipt)
    physical_receipt = Path(receipt if physical_receipt is None else physical_receipt).absolute()
    if (physical_receipt.is_symlink() or not physical_receipt.is_file()
            or physical_receipt.resolve(strict=True) != physical_receipt):
        raise ValueError("command receipt must be a regular nonsymlink file")
    if resolve is Path:
        owner = _command_campaign_root(receipt, reading=True)
        identity_directory = owner / "identities"
        if identity_directory.is_symlink() or not identity_directory.is_dir():
            raise ValueError("command receipt identity-map directory is unavailable or unsafe")

    parsed_maps = {}

    def read_map(reference):
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("command receipt identity-map reference fields changed")
        checksum = reference["sha256"]
        expected_path = f"identities/{checksum}.json"
        if (not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or reference["path"] != expected_path):
            raise ValueError("command receipt identity-map reference path changed")
        logical_path = owner / "identities" / (checksum + ".json")
        cache_key = str(logical_path)
        if cache_key in parsed_maps:
            return parsed_maps[cache_key]
        path = Path(resolve(str(logical_path))).absolute()
        if path.is_symlink():
            raise ValueError("command receipt identity-map blob cannot be a symlink")
        try:
            resolved = path.resolve(strict=True)
            if resolve is Path:
                resolved.relative_to(owner)
        except (OSError, ValueError) as error:
            raise ValueError("command receipt identity-map blob is missing or escapes its campaign") from error
        if resolved != path:
            raise ValueError("command receipt identity-map blob path changed")
        raw = read_bytes(path)
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("command receipt identity-map content hash changed")
        observations = _FILE_OBSERVATIONS.get()
        if observations is not None:
            blob = observations.read_json(path)
        else:
            from build_dependencies import identity_json
            blob = identity_json(raw.decode("utf-8"))
        if (not isinstance(blob, dict) or set(blob) != {"protocol", "values"}
                or blob["protocol"] != COMMAND_IDENTITY_MAP_PROTOCOL
                or not isinstance(blob["values"], dict)
                or _identity_map_bytes(blob["values"]) != raw):
            raise ValueError("command receipt identity-map schema or canonical encoding changed")
        parsed_maps[cache_key] = blob["values"]
        return blob["values"]

    for name in ("inputs", "tools"):
        identity[name] = deepcopy(read_map(identity.pop(name + "_ref")))
    decoded.pop("protocol")
    return decoded


def command_result_payloads(receipt, *, resolve=Path):
    """Validate and enumerate the immutable maps needed by a retained receipt."""
    from build_dependencies import identity_json
    physical_receipt = resolve(receipt)
    document = identity_json(read_text(physical_receipt, encoding="utf-8"))
    _decode_command_receipt(document, receipt, resolve=resolve, physical_receipt=physical_receipt)
    if not isinstance(document, dict) or document.get("protocol") != COMMAND_RECEIPT_PROTOCOL:
        return {}
    owner = _command_logical_owner(receipt)
    return {str(owner / reference["path"]): reference["sha256"]
            for name in ("inputs_ref", "tools_ref")
            for reference in (document["identity"][name],)}


def write_command_result(path: Path, record: Mapping) -> None:
    """Publish a compact receipt backed by immutable campaign-owned identity maps."""
    if not isinstance(record, Mapping) or not isinstance(record.get("identity"), Mapping):
        raise ValueError("command result identity is malformed")
    if "protocol" in record:
        raise ValueError("command result already has a storage protocol")
    receipt = Path(path)
    owner = _command_campaign_root(receipt, reading=False)
    stored = deepcopy(dict(record))
    identity = stored["identity"]
    if any(name.endswith("_ref") for name in identity):
        raise ValueError("command result contains an unknown identity-map reference")
    identity_directory = owner / "identities"
    if identity_directory.is_symlink():
        raise ValueError("command result identity-map directory cannot be a symlink")
    identity_directory.mkdir(exist_ok=True)
    identity_directory_resolved = identity_directory.resolve(strict=True)
    try:
        identity_directory_resolved.relative_to(owner.resolve(strict=True))
    except ValueError as error:
        raise ValueError("command result identity-map directory escapes its campaign") from error

    for name in ("inputs", "tools"):
        if name not in identity or name + "_ref" in identity:
            raise ValueError("command result identity maps are incomplete")
        blob = _identity_map_bytes(identity.pop(name))
        checksum = hashlib.sha256(blob).hexdigest()
        blob_path = identity_directory / (checksum + ".json")
        if blob_path.is_symlink():
            raise ValueError("command result identity-map blob cannot be a symlink")
        try:
            with blob_path.open("xb") as stream:
                if stream.write(blob) != len(blob):
                    raise OSError("short write while publishing command identity map")
        except FileExistsError:
            existing = read_bytes(blob_path)
            if (hashlib.sha256(existing).hexdigest() != checksum or existing != blob):
                raise ValueError("existing command identity-map blob changed")
        identity[name + "_ref"] = {"path": f"identities/{checksum}.json", "sha256": checksum}
    stored["protocol"] = COMMAND_RECEIPT_PROTOCOL
    write_json(receipt, stored)


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
                for line in read_text(config, encoding="utf-8").splitlines():
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
    """Hash each resolved file once per call; checkpoints can bind absent configs."""
    result = {}
    for name in sorted(set(paths)):
        path = Path(name).resolve(strict=not allow_missing)
        if allow_missing and not path.exists():
            result[str(path)] = None
        elif not path.is_file():
            raise ValueError("tool/configuration input is not a file: " + str(path))
        elif result.get(str(path)) is None:
            result[str(path)] = digest(path)
    return result


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def mapped_windows_path(path):
    if not PureWindowsPath(path).is_absolute():
        raise ValueError("Windows identity interpreter/path must be absolute")
    return Path(subprocess.check_output(["wslpath", "-u", str(path)], text=True).strip())


def windows_venv_home(config):
    homes = [line.partition("=")[2].strip() for line in read_text(config).splitlines()
             if line.partition("=")[0].strip() == "home"]
    if len(homes) != 1 or not PureWindowsPath(homes[0]).is_absolute():
        raise ValueError("Windows identity venv has no explicit base interpreter")
    return str(PureWindowsPath(homes[0]) / "python.exe")


def windows_runtime_paths():
    python = Path(sys.executable)
    configs = [python.parent / "pyvenv.cfg", python.parent.parent / "pyvenv.cfg"]
    config = next((p for p in configs if p.is_file()), None)
    base = Path(windows_venv_home(config)) if config else Path(sys.base_prefix) / "python.exe"
    if base.resolve(strict=True) != (Path(sys.base_prefix) / "python.exe").resolve(strict=True):
        raise ValueError("Windows runtime base interpreter disagrees with venv configuration")
    return {"python": python, "base_python": base, "config": config}


def windows_identity_interpreter(python):
    if os.name != "posix":
        raise ValueError("Windows identity interop requires the Linux collector")
    python = str(PureWindowsPath(python))
    executable = mapped_windows_path(python)
    if not executable.is_file():
        raise ValueError("Windows identity interpreter is unavailable")
    config = next((p for p in (executable.parent / "pyvenv.cfg", executable.parent.parent / "pyvenv.cfg")
                   if p.is_file()), None)
    base = windows_venv_home(config) if config else python
    return {"python": python, "base_python": base, "executable": executable, "config": config}


def prepare_windows_identity_launch(root, python, request_path, *, base_python=None):
    from build_dependencies import read_identity_request

    interpreter = windows_identity_interpreter(python)
    python, base, executable, config = (interpreter[n] for n in ("python", "base_python", "executable", "config"))
    base_executable = mapped_windows_path(base)
    base_target = str(PureWindowsPath(base_python)) if base_python is not None else base
    supervisor_executable = mapped_windows_path(base_target)
    if digest(supervisor_executable) != digest(base_executable):
        raise ValueError("Windows identity qualified base interpreter differs from venv configuration")
    checksum = digest(request_path)
    request = read_identity_request(request_path, checksum)
    helper = root / "scripts/build_dependencies.py"
    supervisor = root / "scripts/process_tree.py"
    project_tools = [helper, supervisor, root / "scripts/execution.py"]
    tools = [executable, base_executable, supervisor_executable, *project_tools, *([config] if config else [])]
    if any(not p.is_file() for p in tools):
        raise ValueError("Windows identity launch tool/configuration is missing")
    native_request = windows_path(request_path)
    return {"python": python, "base_python": base, "executable": str(executable),
            "base_target": base_target, "supervisor_executable": str(supervisor_executable),
            "request": request_path.relative_to(root).as_posix(), "request_sha256": checksum,
            "collection_id": request["collection_id"], "phase": request["phase"],
            "supervisor": windows_path(supervisor), "tools": [str(p) for p in tools],
            "command": [python, "-B", windows_path(helper), "--file-identities", native_request,
                        "--request-sha256", checksum], "native_request": native_request}



def run_process_tree(
    command, root, environment, output, timeout, receipt, lock, tree,
    stderr_output=None, vctip_cleanup=None, windows_identity=None,
):
    OutputLease.check_held()
    supervisor = Path(__file__).with_name("process_tree.py")
    policy = ["--vctip-cleanup", json.dumps(vctip_cleanup)] if vctip_cleanup else []
    supervisor_command = [sys.executable, "-B", str(supervisor), str(receipt), str(timeout), *policy, *command]
    if windows_identity is not None:
        supervisor_command = [windows_identity["supervisor_executable"], "-B", windows_identity["supervisor"],
            windows_path(receipt), str(timeout), "--identity-request", windows_identity["native_request"],
            windows_identity["request_sha256"], *command]
        tree.update(drained=False)
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
    tree["interop_pid" if windows_identity is not None else "supervisor_pid"] = process.pid
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
            observed = json.loads(read_text(receipt, encoding="utf-8"))
            if windows_identity is not None:
                expected = {"collection_id": windows_identity["collection_id"],
                            "phase": windows_identity["phase"],
                            "request_sha256": windows_identity["request_sha256"]}
                seal_path = receipt.with_suffix(".seal.json")
                seal = json.loads(read_text(seal_path)) if seal_path.is_file() else None
                if (observed.get("identity_request") != expected or observed.get("command") != command
                        or seal != {"receipt_sha256": digest(receipt), "identity_request": expected}
                        or set(observed.get("creation", {})) != {"windows_creation_time_100ns"}
                        or type(observed["creation"]["windows_creation_time_100ns"]) is not int
                        or observed["creation"]["windows_creation_time_100ns"] <= 0
                        or any(type(observed.get(k)) is not int or observed[k] <= 0
                               for k in ("supervisor_pid", "command_pid", "gated_pid"))):
                    tree.update(drained=False, status="blocked")
                    raise RuntimeError("Windows identity native process evidence is missing or contradictory")
            tree.update(observed)
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
            if json.loads(read_text(path, encoding="utf-8")) != contract:
                raise ValueError(
                    "existing campaign has a different root, preset or required gate list"
                )

    def attempts(self, stage: str) -> list[dict]:
        name_component(stage)
        paths = list((self.directory / "attempts").glob(f"{stage}-*/result.json"))
        records = [read_json(path) for path in paths]
        return sorted(
            (record for record in records if record["stage"] == stage),
            key=lambda record: (record.get("attempt_number", 0), record["started_at"]),
        )

    def check_run_request(
        self, stage: str, command: Sequence[str], *, inputs: Sequence[str], timeout: float | None
    ) -> None:
        """Reject invalid requests before callers create pending ownership or reuse results."""
        name_component(stage)
        if (
            stage not in self.required
            or not command
            or not inputs
            or timeout is not None and (not math.isfinite(timeout) or timeout <= 0)
        ):
            raise ValueError(
                "a declared stage, command, input files and finite positive timeout or None are required"
            )

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        inputs: Sequence[str],
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        repeat_reason: str = "",
        tools: Sequence[str] = (),
        expected_inputs: Mapping[str, str] | None = None,
        expected_tools: Mapping[str, str] | None = None,
        vctip_executable: str | None = None,
        vctip_grace_seconds: float = 30,
        vctip_drain_cap_seconds: float = 300,
        windows_python: str | None = None,
        identity_request: Path | None = None,
        windows_base_python: str | None = None,
    ) -> Path:
        OutputLease.check_held()
        native = None
        if windows_python is not None or identity_request is not None or windows_base_python is not None:
            if (windows_python is None or identity_request is None
                    or stage not in {"windows-entry", "windows-terminal"}
                    or vctip_executable is not None):
                raise ValueError("Windows identity route requires its bound collection stage/request")
            native = prepare_windows_identity_launch(self.root, windows_python, identity_request,
                                                     base_python=windows_base_python)
            if list(command) != native["command"] or stage != "windows-" + native["phase"]:
                raise ValueError("Windows identity command/stage does not match its request")
        self.check_run_request(stage, command, inputs=inputs, timeout=timeout)
        environment = command_environment(os.environ if env is None else env)
        if "JUICER_OUTPUT_LEASES" in os.environ:
            environment["JUICER_OUTPUT_LEASES"] = os.environ["JUICER_OUTPUT_LEASES"]
        executable = native["executable"] if native else shutil.which(command[0], path=environment.get("PATH", ""))
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
                "drain_cap_seconds": vctip_drain_cap_seconds if timeout is not None else None,
            }
        elif vctip_grace_seconds != 30 or vctip_drain_cap_seconds != 300:
            raise ValueError("VCTIP drain policy requires a VCTIP executable")
        preparation_start = time.perf_counter()
        selected_tools = [*tools, *([str(vctip_path)] if vctip_cleanup else []), *(native["tools"] if native else [])]
        with FileObservations() as initial_reads:
            identity = {
                "command": list(command),
                "inputs": snapshot(self.root, inputs),
                "tools": tool_snapshot(selected_tools),
                "supervisor_sha256": digest(self.root / "scripts/process_tree.py" if native
                                             else Path(__file__).with_name("process_tree.py")),
                "executable_sha256": digest(Path(executable)),
                "environment_sha256": hashlib.sha256(
                    json.dumps(environment, sort_keys=True).encode()
                ).hexdigest(),
            }
        if vctip_cleanup:
            identity["vctip_cleanup"] = vctip_cleanup
        if native is not None:
            identity["windows_identity"] = native
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
                "timings": {"initial_hash_seconds": preparation_seconds,
                            "initial_reads": initial_reads.counts()},
                "output": (directory / "output.log").relative_to(self.root).as_posix(),
            }
            receipt = directory / "result.json"
            write_command_result(receipt, record)
            start = time.perf_counter()
            try:
                with (directory / "output.log").open("xb") as output:
                    arguments = [
                        list(command), self.root, environment, output, timeout,
                        directory / "process.json", lock, tree,
                    ]
                    if native is not None:
                        record["returncode"] = run_process_tree(*arguments, windows_identity=native)
                    elif vctip_cleanup:
                        record["returncode"] = run_process_tree(
                            *arguments, vctip_cleanup=vctip_cleanup,
                        )
                    else:
                        record["returncode"] = run_process_tree(*arguments)
                if record["returncode"]:
                    raise subprocess.CalledProcessError(record["returncode"], command)
                OutputLease.check_held()
                verification_start = time.perf_counter()
                with FileObservations() as final_reads:
                    unchanged = (
                        snapshot(self.root, inputs) == identity["inputs"]
                        and tool_snapshot(selected_tools) == identity["tools"]
                    )
                record["timings"]["final_reads"] = final_reads.counts()
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
                write_command_result(receipt, record)
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
            with FileObservations():
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
                                if path.name == "result.json":
                                    read_json(path)
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
            owner = json.loads(read_text(self.path / "owner.json", encoding="utf-8"))
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
            owner = json.loads(read_text(self.path / "owner.json", encoding="utf-8"))
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
                    owner = json.loads(read_text(self.path / "owner.json", encoding="utf-8"))
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
        owner = json.loads(read_text(self.path / "owner.json", encoding="utf-8"))
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
            owner = json.loads(read_text(self.path / "owner.json", encoding="utf-8"))
            write_json(self.path / "owner.json", {**owner, "status": "blocked"})
        else:
            (self.path / "owner.json").unlink()
            self.path.rmdir()
