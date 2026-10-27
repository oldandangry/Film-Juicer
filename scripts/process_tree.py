"""Process-tree supervisor. Its receipt proves all writers have exited."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class LinuxChildren:
    def __init__(self):
        # Isolate subreaping in this supervisor, never in the campaign caller.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot enable child subreaping")

    def start(self, command, gate_path):
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def active(self):
        self.process.poll()
        children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
        for child in children.read_text().split():
            if int(child) != self.process.pid:
                os.waitpid(int(child), os.WNOHANG)
        return bool(children.read_text().strip())

    def stop(self):
        # Stop parents before enumerating children so they cannot keep forking.
        # Orphans (including setsid children) are adopted by this subreaper.
        def kill_children(pid):
            children = Path(f"/proc/{pid}/task/{pid}/children")
            try:
                pids = [int(value) for value in children.read_text().split()]
            except FileNotFoundError:
                return
            for child in pids:
                try:
                    os.kill(child, signal.SIGSTOP)
                    kill_children(child)
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        kill_children(os.getpid())

    def close(self):
        pass


class WindowsJob:
    def __init__(self):
        from ctypes import wintypes

        class Limits(ctypes.Structure):
            _fields_ = [
                ("user_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum", ctypes.c_size_t),
                ("maximum", ctypes.c_size_t),
                ("processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", Limits),
                ("io", ctypes.c_uint64 * 6),
                ("memory", ctypes.c_size_t * 4),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("faults", wintypes.DWORD),
                ("total", wintypes.DWORD),
                ("active", wintypes.DWORD),
                ("terminated", wintypes.DWORD),
            ]

        self.accounting_type = Accounting
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            (
                "SetInformationJobObject",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            (
                "QueryInformationJobObject",
                [
                    wintypes.HANDLE,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                ],
                wintypes.BOOL,
            ),
            (
                "AssignProcessToJobObject",
                [wintypes.HANDLE, wintypes.HANDLE],
                wintypes.BOOL,
            ),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            (
                "OpenProcess",
                [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
                wintypes.HANDLE,
            ),
            (
                "IsProcessInJob",
                [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
                wintypes.BOOL,
            ),
            (
                "QueryFullProcessImageNameW",
                [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            ("TerminateProcess", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        ):
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.job = self.api.CreateJobObjectW(None, None)
        self.check(self.job)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway
        try:
            self.check(
                self.api.SetInformationJobObject(
                    self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def start(self, command, gate_path):
        # The gate prevents any command descendants before job assignment.
        gate = (
            "import os,subprocess,sys; from pathlib import Path; "
            "p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(str(os.getpid())); t.replace(p); "
            "sys.exit(subprocess.call(sys.argv[2:], stdin=subprocess.DEVNULL) if sys.stdin.buffer.read(1)==b'g' else 125)"
        )
        # Bypass venv and app-alias redirectors for the gated interpreter.
        interpreter = Path(sys.base_prefix) / "python.exe"
        self.process = subprocess.Popen(
            [str(interpreter), "-B", "-c", gate, str(gate_path), *command],
            stdin=subprocess.PIPE,
        )
        try:
            self.check(
                self.api.AssignProcessToJobObject(self.job, int(self.process._handle))
            )
            # App execution aliases may add a redirector even for a base path.
            # Assign the actual gated interpreter too, before releasing stdin.
            deadline = time.monotonic() + 5
            read_error = None

            def check_gate_waiting():
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    detail = ": " + str(read_error) if read_error is not None else ""
                    raise RuntimeError(
                        "gated interpreter did not report a readable process identity "
                        "before exit or deadline" + detail
                    ) from read_error

            while True:
                check_gate_waiting()
                try:
                    pid = int(gate_path.read_text())
                except (FileNotFoundError, PermissionError) as error:
                    read_error = error
                    time.sleep(0.01)
                    continue
                check_gate_waiting()
                break
            if pid != self.process.pid:
                handle = self.api.OpenProcess(0x0101, False, pid)
                self.check(handle)
                try:
                    self.check(self.api.AssignProcessToJobObject(self.job, handle))
                finally:
                    self.check(self.api.CloseHandle(handle))
            check_gate_waiting()
            self.process.stdin.write(b"g")
            self.process.stdin.close()
        except BaseException:
            self.process.kill()
            self.process.wait()
            raise

    def active(self):
        self.process.poll()
        accounting = self.accounting_type()
        self.check(
            self.api.QueryInformationJobObject(
                self.job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
            )
        )
        return accounting.active != 0

    def stop(self):
        self.check(self.api.TerminateJobObject(self.job, 125))

    def stop_vctip(self, expected_path, record_termination):
        from ctypes import wintypes

        expected = os.path.normcase(os.path.realpath(expected_path))
        capacity = 16
        for _ in range(16):
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if self.api.QueryInformationJobObject(
                self.job, 3, buffer, ctypes.sizeof(buffer), None
            ):
                assigned = ctypes.c_uint32.from_buffer(buffer, 0).value
                count = ctypes.c_uint32.from_buffer(buffer, 4).value
                if count > capacity:
                    raise RuntimeError("job process list exceeded its returned capacity")
                # A successful query can still return only part of the PID list.
                if count >= assigned:
                    break
                capacity = max(capacity * 2, assigned)
            else:
                if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                    raise ctypes.WinError(ctypes.get_last_error())
                capacity *= 2
        else:
            raise RuntimeError("job process list did not stabilize")
        pids = list((ctypes.c_size_t * count).from_buffer(buffer, 8))
        for pid in pids:
            # QUERY_LIMITED_INFORMATION | TERMINATE | SYNCHRONIZE: the wait
            # must be authorized so an exit between enumeration and inspection
            # can be distinguished from an invalid handle.
            handle = self.api.OpenProcess(0x101001, False, pid)
            if not handle:
                if ctypes.get_last_error() == 87:  # Process exited after enumeration.
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                in_job = wintypes.BOOL()
                self.check(self.api.IsProcessInJob(handle, self.job, ctypes.byref(in_job)))
                if not in_job.value:
                    continue
                wait = self.api.WaitForSingleObject(handle, 0)
                if wait == 0:  # WAIT_OBJECT_0
                    continue
                if wait != 0x102:  # WAIT_TIMEOUT
                    raise ctypes.WinError(ctypes.get_last_error())
                image = ctypes.create_unicode_buffer(32768)
                length = wintypes.DWORD(len(image))
                self.check(self.api.QueryFullProcessImageNameW(
                    handle, 0, image, ctypes.byref(length)
                ))
                path = image.value[:length.value]
                if os.path.normcase(os.path.realpath(path)) != expected:
                    continue
                if not self.api.TerminateProcess(handle, 125):
                    error = ctypes.get_last_error()
                    if self.api.WaitForSingleObject(handle, 0) == 0:
                        continue  # Exited after the image check.
                    raise ctypes.WinError(error)
                record_termination({"pid": pid, "path": path})
            finally:
                self.check(self.api.CloseHandle(handle))

    def close(self):
        self.check(self.api.CloseHandle(self.job))


def main():
    receipt = Path(sys.argv[1])
    timeout = None if sys.argv[2] == "None" else float(sys.argv[2])
    cleanup = json.loads(sys.argv[4]) if sys.argv[3] == "--vctip-cleanup" else None
    command = sys.argv[5:] if cleanup else sys.argv[3:]
    interrupted = threading.Event()

    def watch_owner():
        os.read(sys.stdin.fileno(), 1)  # Stop byte or EOF when the owner disappears.
        interrupted.set()

    threading.Thread(target=watch_owner, daemon=True).start()
    owner = WindowsJob() if os.name == "nt" else LinuxChildren()
    if os.name == "nt":
        times = [ctypes.c_uint64() for _ in range(4)]
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.GetProcessTimes.argtypes = [
            ctypes.c_void_p,
            *([ctypes.POINTER(ctypes.c_uint64)] * 4),
        ]
        if not api.GetProcessTimes(
            ctypes.c_void_p(-1), *(ctypes.byref(value) for value in times)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        creation = {"windows_creation_time_100ns": times[0].value}
    else:
        fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
        creation = {
            "linux_start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    record = {
        "supervisor_pid": os.getpid(),
        "creation": creation,
        "status": "running",
        "drained": False,
    }
    if cleanup:
        record.update(vctip_cleanup=cleanup, vctip_terminated=[])

    def save():
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt)

    save()
    try:
        started = time.monotonic()
        owner.start(command, receipt.with_suffix(".gate.json"))
        record["command_pid"] = owner.process.pid
        save()
        deadline = None if timeout is None else time.monotonic() + timeout
        stopping = None
        command_end = None
        failed_parent = None
        next_vctip_check = None
        while True:
            active = owner.active()
            if command_end is None and owner.process.poll() is not None:
                command_end = time.monotonic()
                if owner.process.returncode:
                    failed_parent = command_end + 0.5
                    record["first_returncode"] = owner.process.returncode
                    record["failure_descendants"] = True
                    save()
                elif cleanup:
                    next_vctip_check = command_end + cleanup["grace_seconds"]
                    if cleanup["drain_cap_seconds"] is not None:
                        cap = command_end + cleanup["drain_cap_seconds"]
                        deadline = cap if deadline is None else min(deadline, cap)
            if not active:
                break
            now = time.monotonic()
            if (
                cleanup and next_vctip_check is not None
                and now >= next_vctip_check and (deadline is None or now < deadline)
                and not interrupted.is_set()
            ):
                def record_termination(child):
                    record["vctip_terminated"].append({
                        **child, "seconds_after_parent": time.monotonic() - command_end,
                    })
                    save()

                owner.stop_vctip(cleanup["executable"], record_termination)
                next_vctip_check = now + 0.5
            if stopping is None and (
                interrupted.is_set()
                or deadline is not None and now >= deadline
                or failed_parent is not None
                and now >= failed_parent
            ):
                record["status"] = (
                    "interrupted"
                    if interrupted.is_set()
                    else "failed-parent"
                    if failed_parent is not None
                    else "timeout"
                )
                stopping = now + 10
                save()
            if stopping is not None:
                owner.stop()
                if time.monotonic() >= stopping:
                    raise RuntimeError("process tree did not drain after termination")
            time.sleep(0.02)
        owner.process.wait()
        finished = time.monotonic()
        record.update(
            command_seconds=(command_end or finished) - started,
            drain_seconds=finished - (command_end or finished),
        )
        record.update(
            drained=True,
            returncode=record.get("first_returncode", owner.process.returncode),
        )
        if record["status"] == "running":
            record["status"] = "completed"
    except BaseException as exc:
        record.update(status="blocked", error=str(exc))
        raise
    finally:
        owner.close()
        save()


if __name__ == "__main__":
    main()
