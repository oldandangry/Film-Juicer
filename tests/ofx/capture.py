#!/usr/bin/env python3
"""Capture one real OFX render from an early-instrumented Resolve process."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
execution = importlib.import_module("execution")


class ProcessHandle:
    """Observe the lifetime of the one process spawned by this capture."""

    def __init__(self, pid: int):
        if os.name != "nt":
            raise ValueError("Resolve process capture requires Windows")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        self.kernel.GetProcessTimes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
        ]
        self.kernel.GetProcessTimes.restype = ctypes.c_int
        self.kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel.WaitForSingleObject.restype = ctypes.c_uint32
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int
        self.handle = self.kernel.OpenProcess(0x1000 | 0x00100000, 0, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        created = ctypes.wintypes.FILETIME()
        exited = ctypes.wintypes.FILETIME()
        kernel = ctypes.wintypes.FILETIME()
        user = ctypes.wintypes.FILETIME()
        if not self.kernel.GetProcessTimes(
            self.handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        unix_seconds = (ticks - 116444736000000000) / 10_000_000
        self.created_at = datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat()

    def exited(self) -> bool:
        return self.kernel.WaitForSingleObject(self.handle, 0) == 0

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class PixelChunks:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.intervals = {}
        self.totals = {}

    def add(self, kind: str, offset: int, total: int, data: bytes):
        if kind not in {"source", "output"}:
            raise ValueError(f"unexpected image kind: {kind}")
        if type(offset) is not int or type(total) is not int or offset < 0 or total <= 0:
            raise ValueError("invalid pixel chunk bounds")
        if not data or offset + len(data) > total:
            raise ValueError("pixel chunk is empty or outside its image")
        if kind in self.totals and self.totals[kind] != total:
            raise ValueError(f"{kind} image size changed between chunks")
        intervals = self.intervals.setdefault(kind, [])
        end = offset + len(data)
        if any(offset < previous_end and previous_start < end for previous_start, previous_end in intervals):
            raise ValueError(f"overlapping or duplicate {kind} pixel chunk")
        path = self.directory / f"{kind}.f32.part"
        stream = self.files.get(kind)
        if stream is None:
            stream = path.open("w+b")
            stream.truncate(total)
            self.files[kind] = stream
            self.totals[kind] = total
        stream.seek(offset)
        stream.write(data)
        stream.flush()
        intervals.append((offset, end))

    def finish(self, kind: str, metadata: dict) -> dict:
        intervals = sorted(self.intervals.get(kind, ()))
        total = self.totals.get(kind)
        cursor = 0
        for start, end in intervals:
            if start != cursor:
                raise ValueError(f"missing {kind} pixel chunk at byte {cursor}")
            cursor = end
        if total is None or cursor != total:
            raise ValueError(f"incomplete {kind} pixels: received {cursor} of {total} bytes")
        bounds = metadata.get("bounds")
        components = metadata.get("components")
        channel_count = {"OfxImageComponentRGB": 3, "OfxImageComponentRGBA": 4}.get(components)
        if not isinstance(bounds, list) or len(bounds) != 4 or channel_count is None:
            raise ValueError(f"invalid {kind} image metadata")
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if width <= 0 or height <= 0 or total != width * height * channel_count * 4:
            raise ValueError(f"{kind} byte count does not match bounds and components")
        stream = self.files[kind]
        stream.flush()
        stream.close()
        del self.files[kind]
        partial = self.directory / f"{kind}.f32.part"
        final = self.directory / f"{kind}.f32"
        partial.replace(final)
        return {
            "path": final.name,
            "bytes": total,
            "sha256": execution.digest(final),
            "shape": [height, width, channel_count],
        }

    def close(self):
        for stream in self.files.values():
            stream.close()
        self.files.clear()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def bundle_inventory(bundle: Path) -> dict:
    if not bundle.is_dir():
        raise ValueError(f"bundle root is not a directory: {bundle}")
    return {
        path.relative_to(bundle).as_posix(): execution.digest(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    }


def inventory_digest(inventory: dict) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def module_in_bundle(module: Path, bundle: Path):
    try:
        module.relative_to(bundle)
    except ValueError as error:
        raise ValueError("module must be inside the supplied bundle root") from error


def image_capture(report: dict, chunks: PixelChunks) -> dict:
    if report.get("schema") != "juicer-ofx-capture-v1":
        raise ValueError("unexpected capture schema")
    if report.get("complete") is not True or report.get("status") != 0:
        raise ValueError("observer reported an incomplete or failed OFX render")
    images = {}
    for kind in ("source", "output"):
        item = report.get(kind)
        if not isinstance(item, dict):
            raise ValueError(f"capture omitted {kind} image metadata")
        images[kind] = chunks.finish(kind, item)
    return {"report": report, "images": images}


def run(args):
    if os.name != "nt":
        raise ValueError("this capture launcher is for Windows Resolve")
    try:
        import frida
    except ImportError as error:
        raise RuntimeError("install Frida in the isolated Windows capture environment first") from error

    executable = Path(args.executable).resolve(strict=True)
    module = Path(args.module).resolve(strict=True)
    bundle = Path(args.bundle_root).resolve(strict=True)
    agent = Path(__file__).with_name("capture-agent.js").resolve(strict=True)
    module_in_bundle(module, bundle)
    agent_bytes = agent.read_bytes()
    agent_sha256 = hashlib.sha256(agent_bytes).hexdigest()
    bundle_before = bundle_inventory(bundle)
    module_sha256 = execution.digest(module)
    bundle_sha256 = inventory_digest(bundle_before)

    result_parent = Path(args.result_parent).resolve()
    result_parent.mkdir(parents=True, exist_ok=True)
    directory = result_parent / execution.name_component(args.run_id)
    directory.mkdir(exist_ok=False)
    control_path = directory / "control.json"
    messages_path = directory / "messages.jsonl"
    (directory / "capture-agent.js").write_bytes(agent_bytes)

    manifest = {
        "schema": "juicer-ofx-process-capture-v1",
        "status": "starting",
        "started_at": timestamp(),
        "runner": {"path": str(Path(__file__).resolve()), "sha256": execution.digest(Path(__file__))},
        "frida": {"version": getattr(frida, "__version__", "unknown")},
        "agent": {"path": str(agent), "sha256": agent_sha256},
        "executable": str(executable),
        "arguments": args.target_arg,
        "module": {"path": str(module), "sha256": module_sha256},
        "bundle": {
            "root": str(bundle),
            "tree_sha256": bundle_sha256,
            "files": bundle_before,
            "hash_definition": "SHA-256 of sorted relative-path to file-SHA-256 JSON object",
        },
        "control_file": str(control_path),
        "capture": None,
        "observer_complete": False,
        "descendant_drainage_claimed": False,
    }
    execution.write_json(directory / "manifest.json", manifest)
    event_queue = queue.Queue()
    chunk_sets = {}
    pending_arm = None
    seen_module = False
    seen_plugin = False
    script_loaded = False
    observer_unloaded = False
    resumed = False
    capture_terminal = False
    last_control_digest = None
    process_handle = None
    session = None
    script = None
    pid = None
    session_detached = False
    errors = []

    def record(event: dict):
        with messages_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()

    def update_manifest(**values):
        manifest.update(values)
        execution.write_json(directory / "manifest.json", manifest)

    def unload_observer(reason: str):
        nonlocal observer_unloaded
        if script_loaded and not observer_unloaded:
            try:
                script.unload()
            except frida.InvalidOperationError:
                if process_handle is None or not process_handle.exited():
                    raise
            observer_unloaded = True
            update_manifest(observer_complete=True, observer_unloaded_at=timestamp(), observer_unload_reason=reason)

    def arm(options: dict):
        nonlocal pending_arm
        if not seen_module or not seen_plugin:
            pending_arm = options
            record({"at": timestamp(), "event": "arm-pending", "options": options})
            return
        script.exports_sync.arm(options)
        update_manifest(armed_at=timestamp(), arm_options=options, status="armed")
        record({"at": timestamp(), "event": "armed", "options": options})
        pending_arm = None

    def on_message(message, data):
        event_queue.put((message, data, timestamp()))

    def on_detached(reason, crash=None):
        event_queue.put(
            ({"type": "detached", "reason": str(reason), "crash": str(crash) if crash else None}, None, timestamp())
        )

    def process_message(message, data, observed_at):
        nonlocal seen_module, seen_plugin, capture_terminal, session_detached
        if message.get("type") == "send":
            payload = message.get("payload")
            if not isinstance(payload, dict):
                record(
                    {"at": observed_at, "event": "message", "payload": payload, "data_bytes": len(data) if data else 0}
                )
                return
            payload = dict(payload)
            if payload.get("event") == "pixels":
                capture_id = payload.get("capture_id")
                if type(capture_id) is not int or capture_id < 1:
                    raise ValueError("invalid pixel capture id")
                chunks = chunk_sets.setdefault(capture_id, PixelChunks(directory / f"capture-{capture_id:04d}"))
                if data is None:
                    raise ValueError("pixel message omitted binary data")
                payload["binary_sha256"] = hashlib.sha256(data).hexdigest()
                payload["binary_bytes"] = len(data)
            record({"at": observed_at, "event": payload.get("event", "message"), "payload": payload})
            if payload.get("event") == "pixels":
                try:
                    chunks.add(payload.get("kind"), payload.get("offset"), payload.get("total"), bytes(data))
                except BaseException as error:
                    record({"at": observed_at, "event": "pixel-chunk-error", "error": str(error)})
                    raise
            if payload.get("event") == "module":
                seen_module = True
                update_manifest(loaded_module=payload)
            elif payload.get("event") == "plugin":
                seen_plugin = True
                update_manifest(plugin=payload)
            elif payload.get("event") == "begin":
                update_manifest(render_begin=payload)
            elif payload.get("event") == "error":
                errors.append(payload)
                update_manifest(errors=errors, status="observer-error")
                capture_terminal = True
                unload_observer("observer-error")
            elif payload.get("event") == "capture":
                capture_terminal = True
                capture_id = payload.get("capture_id")
                chunks = chunk_sets.get(capture_id)
                if chunks is None:
                    raise ValueError("capture completion has no pixel chunks")
                capture = image_capture(payload, chunks)
                capture_document = dict(payload)
                capture_document["image_artifacts"] = capture["images"]
                execution.write_json(directory / f"capture-{capture_id:04d}" / "capture.json", capture_document)
                update_manifest(capture=capture, observer_complete=True, status="capture-complete")
                unload_observer("capture-complete")
            if pending_arm is not None and seen_module and seen_plugin and not capture_terminal:
                arm(pending_arm)
        elif message.get("type") == "error":
            failure = {
                "at": observed_at,
                "event": "frida-script-error",
                "description": message.get("description"),
                "stack": message.get("stack"),
            }
            errors.append(failure)
            record(failure)
            update_manifest(errors=errors, status="observer-error")
            capture_terminal = True
            unload_observer("frida-script-error")
        elif message.get("type") == "detached":
            session_detached = True
            record(
                {
                    "at": observed_at,
                    "event": "frida-session-detached",
                    "reason": message.get("reason"),
                    "crash": message.get("crash"),
                }
            )
            update_manifest(
                process_detached_at=observed_at, detach_reason=message.get("reason"), crash=message.get("crash")
            )
        else:
            record({"at": observed_at, "event": "frida-message", "message": message})

    def consume_control():
        nonlocal last_control_digest
        if not control_path.is_file():
            return
        raw = control_path.read_bytes()
        control_digest = hashlib.sha256(raw).hexdigest()
        if control_digest == last_control_digest:
            return
        last_control_digest = control_digest
        try:
            command = json.loads(raw)
            if not isinstance(command, dict) or command.get("action") not in {"arm", "stop"}:
                raise ValueError("control must be an object with action arm or stop")
            record({"at": timestamp(), "event": "control", "sha256": control_digest, "command": command})
            if command["action"] == "arm":
                options = {key: command[key] for key in ("time", "width", "height", "effect") if key in command}
                if args.time is not None:
                    options.setdefault("time", args.time)
                if args.width is not None:
                    options.setdefault("width", args.width)
                if args.height is not None:
                    options.setdefault("height", args.height)
                if args.effect is not None:
                    options.setdefault("effect", args.effect)
                arm(options)
            else:
                update_manifest(status="stopped-by-owner", stop_requested_at=timestamp())
                unload_observer("explicit-stop")
                return True
        except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
            record({"at": timestamp(), "event": "control-error", "sha256": control_digest, "error": str(error)})
            update_manifest(last_control_error=str(error))
        return False

    gpu_lease = execution.OutputLease(ROOT)
    gpu_lease.__enter__()
    try:
        device = frida.get_local_device()
        argv = [str(executable), *args.target_arg]
        pid = device.spawn(str(executable), argv=argv)
        process_handle = ProcessHandle(pid)
        update_manifest(
            status="process-suspended",
            process={"pid": pid, "created_at": process_handle.created_at, "spawned_by_runner": True},
            control_file=str(control_path),
        )
        session = device.attach(pid)
        session.on("detached", on_detached)
        script = session.create_script(agent_bytes.decode("utf-8"))
        script.on("message", on_message)
        script.load()
        script_loaded = True
        script.exports_sync.watch(str(module))
        update_manifest(observer_installed_at=timestamp(), status="observer-installed")
        if args.arm_on_start:
            options = {
                key: value
                for key, value in (
                    ("time", args.time),
                    ("width", args.width),
                    ("height", args.height),
                    ("effect", args.effect),
                )
                if value is not None
            }
            script.exports_sync.arm(options)
            update_manifest(armed_at=timestamp(), arm_options=options, status="armed-before-resume")
            record({"at": timestamp(), "event": "armed-before-resume", "options": options})
        device.resume(pid)
        resumed = True
        update_manifest(resumed_at=timestamp(), status="waiting-for-capture")
        print(f"Capture directory: {directory}", flush=True)
        print(
            f"Write {control_path} with an arm command after the project is ready; close Resolve to release the GPU lease.",
            flush=True,
        )
        while True:
            execution.OutputLease.check_held()
            try:
                message, data, observed_at = event_queue.get(timeout=0.1)
                process_message(message, data, observed_at)
            except queue.Empty:
                pass
            if not capture_terminal:
                consume_control()
            if process_handle.exited() and session_detached:
                break
        while not event_queue.empty():
            message, data, observed_at = event_queue.get_nowait()
            process_message(message, data, observed_at)
        for chunks in chunk_sets.values():
            chunks.close()
        final_inventory = bundle_inventory(bundle)
        unchanged = final_inventory == bundle_before and execution.digest(module) == module_sha256
        agent_unchanged = hashlib.sha256(agent.read_bytes()).hexdigest() == agent_sha256
        if not unchanged:
            errors.append({"error": "bundle changed while Resolve process was live"})
        if not agent_unchanged:
            errors.append({"error": "observer source changed while capture was live"})
        status = manifest.get("status")
        if status == "capture-complete" and unchanged and agent_unchanged:
            status = "passed"
        elif status == "stopped-by-owner" and unchanged and agent_unchanged:
            status = "stopped"
        elif not capture_terminal:
            status = "process-exited-before-capture"
        else:
            status = "failed"
        update_manifest(
            status=status,
            errors=errors,
            process_exited_at=timestamp(),
            bundle_unchanged=unchanged,
            agent_unchanged=agent_unchanged,
            final_bundle_tree_sha256=inventory_digest(final_inventory),
            completed_at=timestamp(),
        )
    except BaseException as error:
        errors.append({"type": type(error).__name__, "error": str(error), "at": timestamp()})
        update_manifest(status="failed", errors=errors, completed_at=timestamp())
        if pid is not None and not resumed:
            try:
                if session is not None:
                    session.detach()
                frida.get_local_device().kill(pid)
                if process_handle is not None:
                    while not process_handle.exited():
                        time.sleep(0.1)
            except BaseException as cleanup_error:
                update_manifest(
                    status="failed-owner-unresolved",
                    errors=[*errors, {"type": type(cleanup_error).__name__, "error": str(cleanup_error)}],
                )
                raise
        elif pid is not None and resumed and process_handle is not None and not process_handle.exited():
            unload_observer("runner-error")
            while not process_handle.exited():
                time.sleep(0.1)
        raise
    finally:
        if process_handle is not None:
            process_handle.close()
        gpu_lease.__exit__(*sys.exc_info())
    return 0 if manifest.get("status") == "passed" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("run", help="spawn Resolve suspended and observe one exact-module render")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--result-parent", default=str(ROOT / "out/validation/windows-clang-release/results"))
    capture.add_argument("--executable", "--resolve", dest="executable", required=True)
    capture.add_argument("--module", required=True)
    capture.add_argument("--bundle-root", required=True)
    capture.add_argument("--time", type=float)
    capture.add_argument("--width", type=int)
    capture.add_argument("--height", type=int)
    capture.add_argument("--effect")
    capture.add_argument("--arg", dest="target_arg", action="append", default=[])
    capture.add_argument("--arm-on-start", action="store_true")
    capture.set_defaults(action=run)
    return result


def main():
    args = parser().parse_args()
    return args.action(args)


if __name__ == "__main__":
    raise SystemExit(main())
