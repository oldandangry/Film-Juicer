"""Native command ownership, drainage, cancellation and Windows Job controls."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import execution
import process_tree


class ReceiptPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="process receipt ü ")
        self.addCleanup(self.temporary.cleanup)
        self.receipt = Path(self.temporary.name) / "command.process.json"
        self.previous = {"status": "running", "drained": False}
        self.record = {"status": "completed", "drained": True, "returncode": 101}
        self.receipt.write_text(json.dumps(self.previous), encoding="utf-8")

    def test_complete_record_replaces_previous_receipt(self):
        process_tree.save_receipt(self.receipt, self.record)
        self.assertEqual(json.loads(self.receipt.read_text()), self.record)
        self.assertFalse(self.receipt.with_suffix(".tmp").exists())

    def test_write_failure_cannot_publish_or_retry(self):
        with (
            patch.object(Path, "write_text", side_effect=PermissionError("write denied")),
            patch.object(Path, "replace") as replace,
            patch.object(process_tree.time, "sleep") as sleep,
            self.assertRaises(PermissionError),
        ):
            process_tree.save_receipt(self.receipt, self.record)
        replace.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(json.loads(self.receipt.read_text()), self.previous)

    def test_unrelated_replacement_error_propagates_without_retry(self):
        error = OSError("invalid handle")
        error.winerror = 6
        with (
            patch.object(Path, "replace", side_effect=error) as replace,
            patch.object(process_tree.time, "sleep") as sleep,
            self.assertRaises(OSError) as failure,
        ):
            process_tree.save_receipt(self.receipt, self.record)
        self.assertIs(failure.exception, error)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(json.loads(self.receipt.read_text()), self.previous)
        self.assertEqual(json.loads(self.receipt.with_suffix(".tmp").read_text()), self.record)

    @unittest.skipIf(os.name == "nt", "non-Windows error policy")
    def test_non_windows_denial_is_not_retried(self):
        error = PermissionError("denied")
        error.winerror = 5
        with (
            patch.object(Path, "replace", side_effect=error) as replace,
            patch.object(process_tree.time, "sleep") as sleep,
            self.assertRaises(PermissionError),
        ):
            process_tree.save_receipt(self.receipt, self.record)
        replace.assert_called_once()
        sleep.assert_not_called()

    def hold_reader(self):
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        api.CreateFileW.restype = wintypes.HANDLE
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING: replacement is denied.
        handle = api.CreateFileW(str(self.receipt), 0x80000000, 1, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        def release():
            nonlocal handle
            if handle is not None:
                self.assertTrue(api.CloseHandle(handle))
                handle = None

        self.addCleanup(release)
        return release

    @unittest.skipUnless(os.name == "nt", "native Windows file sharing")
    def test_transient_reader_allows_complete_publication_after_retry(self):
        release = self.hold_reader()
        real_sleep = time.sleep

        def release_after_denial(delay):
            self.assertEqual(json.loads(self.receipt.read_text()), self.previous)
            release()
            real_sleep(delay)

        with patch.object(process_tree.time, "sleep", side_effect=release_after_denial) as sleep:
            process_tree.save_receipt(self.receipt, self.record)
        self.assertGreaterEqual(sleep.call_count, 1)
        self.assertEqual(json.loads(self.receipt.read_text()), self.record)
        self.assertFalse(self.receipt.with_suffix(".tmp").exists())

    @unittest.skipUnless(os.name == "nt", "native Windows file sharing")
    def test_persistent_reader_fails_with_both_records_intact(self):
        release = self.hold_reader()
        started = time.monotonic()
        with self.assertRaises(OSError) as failure:
            process_tree.save_receipt(self.receipt, self.record)
        elapsed = time.monotonic() - started
        self.assertIn(failure.exception.winerror, (5, 32))
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertLess(elapsed, 5)
        self.assertEqual(json.loads(self.receipt.read_text()), self.previous)
        self.assertEqual(json.loads(self.receipt.with_suffix(".tmp").read_text()), self.record)
        release()

    @unittest.skipUnless(os.name == "nt", "native Windows error classification")
    def test_sharing_violation_retries_the_atomic_replacement(self):
        original_replace = Path.replace
        attempts = 0

        def replace(path, target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ctypes.WinError(32)
            return original_replace(path, target)

        with patch.object(Path, "replace", replace):
            process_tree.save_receipt(self.receipt, self.record)
        self.assertGreaterEqual(attempts, 2)
        self.assertEqual(json.loads(self.receipt.read_text()), self.record)


class NativeRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native run ü ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "out/build" / ("windows-clang-debug" if os.name == "nt" else "linux-debug")
        self.build.mkdir(parents=True)
        self.directory = self.root / "out/validation" / self.build.name / "results/run-one"
        self.directory.mkdir(parents=True)
        self.lease = execution.OutputLease(self.root, self.build)
        self.lease.__enter__()
        self.addCleanup(self.lease.__exit__, None, None, None)
        self.native_run = execution.NativeRun(self.root, self.directory)
        self.python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable

    def test_parent_exit_waits_for_descendant_output_before_publication(self):
        child = "import time;time.sleep(.25);print('last descendant output',flush=True)"
        command = [
            self.python,
            "-B",
            "-c",
            "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "])",
        ]
        log = self.native_run.command("build", command, timeout=10)
        process = json.loads((self.directory / "build.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "completed")
        self.assertIn("last descendant output", log.read_text())
        self.native_run.publish({"schema": "native-run-v1", "commands": self.native_run.commands})
        self.assertEqual(json.loads((self.directory / "result.json").read_text())["schema"], "native-run-v1")

    def test_native_run_reports_advisory_child_io_counters(self):
        payload_bytes = 64 * 1024
        output = self.root / "resource-usage-payload.bin"
        code = (
            "from pathlib import Path;"
            f"Path({str(output)!r}).write_bytes(b'x' * {payload_bytes});"
            "print('io control complete', flush=True)"
        )
        self.native_run.command("io-control", [self.python, "-B", "-c", code], timeout=10)

        process = json.loads((self.directory / "io-control.process.json").read_text())
        record = self.native_run.commands[0]
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "completed")
        self.assertEqual(record["drained"], process["drained"])
        usage = process.get("resource_usage")
        self.assertIsNotNone(usage, process.get("resource_usage_error"))
        self.assertEqual(record["resource_usage"], usage)
        self.assertTrue(
            all(
                type(value) is int and value >= 0
                for key, value in usage.items()
                if key not in {"kind", "scope", "transfer_semantics"}
            )
        )
        if os.name == "nt":
            self.assertEqual(usage["kind"], "windows_job_io_accounting")
            self.assertEqual(usage["scope"], "all_job_members_including_terminated")
            self.assertEqual(
                usage["transfer_semantics"],
                "OS job logical transfers, not physical device bytes",
            )
            self.assertGreaterEqual(usage["write_transfer_bytes"], payload_bytes)
        else:
            self.assertEqual(usage["kind"], "linux_rusage_children")
            self.assertEqual(usage["scope"], "children_reaped_by_process_tree_supervisor")
            self.assertIn("input_blocks", usage)
            self.assertIn("output_blocks", usage)
        self.assertEqual(output.stat().st_size, payload_bytes)

    def test_nonzero_parent_drains_descendant_and_keeps_failure_visible(self):
        code = (
            "import subprocess,sys;"
            "subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)']);"
            "sys.exit(7)"
        )
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.native_run.command("test", [self.python, "-B", "-c", code], timeout=10)
        self.assertEqual(failure.exception.returncode, 7)
        process = json.loads((self.directory / "test.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "failed-parent")
        self.assertEqual(self.native_run.commands[0]["returncode"], 7)
        self.assertFalse((self.directory / "result.json").exists())

    def test_timeout_cancels_and_drains_descendant_writers(self):
        marker = self.root / "writer.txt"
        child = (
            "import time;from pathlib import Path;p=Path(" + repr(str(marker)) + ");"
            "\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        )
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "]);time.sleep(20)"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.native_run.command("test", [self.python, "-B", "-c", "import time;" + code], timeout=0.4)
        process = json.loads((self.directory / "test.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "timeout")
        final = marker.read_bytes()
        time.sleep(0.05)
        self.assertEqual(marker.read_bytes(), final)
        self.assertFalse((self.directory / "running.lock").exists())

    def test_keyboard_interrupt_drains_real_descendant_before_returning(self):
        marker = self.root / "interrupt-writer.txt"
        child = (
            "import time;from pathlib import Path;p=Path(" + repr(str(marker)) + ");"
            "\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        )
        code = (
            "import subprocess,sys,time;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "]);time.sleep(20)"
        )
        original_wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_after_child(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                interrupted = True
                raise KeyboardInterrupt()
            return original_wait(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", interrupt_after_child), self.assertRaises(KeyboardInterrupt):
            self.native_run.command("test", [self.python, "-B", "-c", code], timeout=10)
        process = json.loads((self.directory / "test.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "interrupted")
        final = marker.read_bytes()
        time.sleep(0.05)
        self.assertEqual(marker.read_bytes(), final)
        self.assertFalse((self.directory / "running.lock").exists())
        self.assertFalse((self.directory / "result.json").exists())

    def test_blocked_owner_rejects_nested_and_new_commands(self):
        self.lease.blocked = True
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            execution.OutputLease(self.root, self.build).__enter__()
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            self.native_run.command("must-not-launch", [self.python, "-c", "raise SystemExit(0)"])
        self.assertEqual(self.native_run.commands, [])
        owner = json.loads((self.lease.path / "owner.json").read_text())
        self.assertEqual(owner["status"], "owned")

    def test_publication_is_atomic_and_rejects_duplicate_or_unproved_runs(self):
        with self.assertRaisesRegex(ValueError, "lowercase directory"):
            self.native_run.command("Bad Name", [self.python, "-c", "pass"])
        self.assertEqual(self.native_run.commands, [])
        self.native_run.publish({"schema": "native-run-v1", "commands": []})
        self.assertFalse((self.directory / "result.tmp").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.native_run.publish({"schema": "native-run-v1", "commands": []})


class NativeCloseoutTests(unittest.TestCase):
    retention_seconds = 7 * 24 * 60 * 60

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native closeout ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir(parents=True)
        self.runs = self.root / "out/validation" / self.preset / "results"
        self.runs.mkdir(parents=True)
        self.lease = execution.OutputLease(self.root, self.build)
        self.lease.__enter__()
        self.addCleanup(self.release_lease)

    def release_lease(self):
        if self.lease is not None and self.lease.path in execution.OutputLease.held:
            self.lease.__exit__(None, None, None)

    def create_run(self, run_id, *, status="passed", candidate_unchanged=True, commands=None, diagnostics=True):
        directory = self.runs / run_id
        directory.mkdir()
        members = {}
        for name, data in (("candidate.json", b"{}\n"), ("candidate.patch", b"")):
            (directory / name).write_bytes(data)
            members[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if diagnostics:
            (directory / "diagnostics.gz").write_bytes(b"compressed diagnostic bytes")
        result = {
            "schema": "native-build-v1",
            "status": status,
            "preset": self.preset,
            "candidate_unchanged": candidate_unchanged,
            "commands": commands
            if commands is not None
            else [{"returncode": 0, "process_status": "completed", "drained": True}],
            "members": members,
            "diagnostics": {"path": "diagnostics.gz", "optional": True},
        }
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return directory

    def accept(self, run_id, *, now=1_000_000):
        return execution.closeout_run(self.root, self.preset, run_id, accept=True, review="closed", now=now)

    def expire(self, *, now=None):
        return execution.expire_diagnostics(
            self.root, self.preset, now=1_000_000 + self.retention_seconds if now is None else now
        )

    def test_closeout_time_controls_expiry_independently_of_file_timestamps(self):
        directory = self.create_run("run-clock")
        result_bytes = (directory / "result.json").read_bytes()
        old = 100
        os.utime(directory / "diagnostics.gz", (old, old))
        closeout = self.accept("run-clock", now=1_000_000)
        before = self.expire(now=1_000_000 + self.retention_seconds - 1)
        self.assertTrue((directory / "diagnostics.gz").exists())
        self.assertEqual(before["deleted_bytes"], 0)
        expired = self.expire(now=1_000_000 + self.retention_seconds)
        self.assertFalse((directory / "diagnostics.gz").exists())
        self.assertEqual(expired["deleted_bytes"], len(b"compressed diagnostic bytes"))
        self.assertEqual(closeout["closeout"]["closed_at"], 1_000_000)
        self.assertEqual((directory / "result.json").read_bytes(), result_bytes)
        repeated = self.expire()
        state = json.loads((directory / "closeout.json").read_text(encoding="utf-8"))
        self.assertEqual(repeated["deleted_bytes"], 0)
        self.assertEqual(state["disposition"]["deleted_bytes"], len(b"compressed diagnostic bytes"))

    def test_absent_invalid_and_reopened_closeout_do_not_expire(self):
        case_ids = ("absent", "invalid", "reopened")
        self.subtest_cases = case_ids
        for case_id in case_ids:
            with self.subTest(case=case_id):
                run_id = "run-" + case_id
                directory = self.create_run(run_id)
                if case_id == "invalid":
                    (directory / "closeout.json").write_text("{", encoding="utf-8")
                elif case_id == "reopened":
                    self.accept(run_id)
                    execution.closeout_run(self.root, self.preset, run_id, review="open")
                before_reads = 0
                if case_id == "absent":
                    summary = self.expire()
                    self.assertEqual(summary["read_bytes"], before_reads)
                else:
                    self.expire()
                self.assertTrue((directory / "diagnostics.gz").exists())
                if case_id == "invalid":
                    self.assertTrue(self.expire()["errors"])
                if case_id == "reopened":
                    state = json.loads((directory / "closeout.json").read_text(encoding="utf-8"))
                    self.assertIsNone(state["closed_at"])

    def test_each_hold_blocks_expiry_until_released(self):
        hold_kinds = ("review", "failure", "reproduction")
        self.subtest_cases = tuple("hold/" + value for value in hold_kinds)
        for hold_kind in hold_kinds:
            with self.subTest(case="hold/" + hold_kind):
                run_id = "run-hold-" + hold_kind
                directory = self.create_run(run_id)
                self.accept(run_id, now=1_000_000)
                execution.closeout_run(self.root, self.preset, run_id, hold=(hold_kind,))
                summary = self.expire()
                self.assertEqual(summary["deleted_bytes"], 0)
                self.assertTrue((directory / "diagnostics.gz").exists())
                execution.closeout_run(self.root, self.preset, run_id, release_hold=(hold_kind,))
                self.expire()
                self.assertFalse((directory / "diagnostics.gz").exists())

    def test_failed_changed_and_undrained_results_cannot_be_accepted_or_expired(self):
        cases = (
            ("failed", {"status": "failed"}),
            ("changed", {"candidate_unchanged": False}),
            ("undrained", {"commands": [{"returncode": 0, "process_status": "completed", "drained": False}]}),
        )
        self.subtest_cases = tuple(case for case, _ in cases)
        for case_id, overrides in cases:
            with self.subTest(case=case_id):
                run_id = "run-" + case_id
                directory = self.create_run(run_id, **overrides)
                with self.assertRaises(ValueError):
                    self.accept(run_id)
                execution.closeout_run(self.root, self.preset, run_id, review="closed", now=1_000_000)
                self.expire()
                self.assertTrue((directory / "diagnostics.gz").exists())

    def test_missing_optional_diagnostic_is_accepted_but_missing_mandatory_member_is_rejected(self):
        cases = ("optional-missing", "mandatory-missing")
        self.subtest_cases = cases
        for case_id in cases:
            with self.subTest(case=case_id):
                run_id = "run-" + case_id
                directory = self.create_run(run_id)
                if case_id == "optional-missing":
                    (directory / "diagnostics.gz").unlink()
                    self.accept(run_id)
                    self.expire()
                    state = json.loads((directory / "closeout.json").read_text(encoding="utf-8"))
                    self.assertEqual(state["disposition"]["status"], "absent")
                else:
                    (directory / "candidate.patch").unlink()
                    with self.assertRaisesRegex(ValueError, "mandatory"):
                        self.accept(run_id)

    def test_member_and_run_directory_links_are_rejected(self):
        if os.name == "nt":
            cases = ("junction",)
        else:
            cases = ("diagnostic-symlink", "run-directory-symlink")
        self.subtest_cases = cases
        for case_id in cases:
            with self.subTest(case=case_id):
                run_id = "run-link-" + case_id
                directory = self.create_run(run_id)
                self.accept(run_id)
                outside = self.root / ("outside-" + case_id)
                if case_id == "diagnostic-symlink":
                    outside.write_bytes(b"protected outside bytes")
                    (directory / "diagnostics.gz").unlink()
                    (directory / "diagnostics.gz").symlink_to(outside)
                else:
                    directory.rename(outside)
                    link = self.runs / run_id
                    if os.name == "nt":
                        created = subprocess.run(
                            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(created.returncode, 0, created.stderr)
                    else:
                        link.symlink_to(outside, target_is_directory=True)
                summary = self.expire()
                self.assertTrue(summary["errors"])
                if case_id == "diagnostic-symlink":
                    self.assertEqual(outside.read_bytes(), b"protected outside bytes")
                else:
                    self.assertTrue((outside / "diagnostics.gz").exists())

    def test_unlink_failure_is_reported_and_retry_preserves_actual_deleted_bytes(self):
        run_id = "run-unlink-failure"
        directory = self.create_run(run_id)
        self.accept(run_id)
        diagnostic = directory / "diagnostics.gz"
        actual_bytes = diagnostic.stat().st_size
        other = self.create_run("run-unlink-success")
        self.accept("run-unlink-success")
        other_bytes = (other / "diagnostics.gz").stat().st_size
        original_unlink = Path.unlink

        def fail_diagnostic(path, *args, **kwargs):
            if path == diagnostic:
                raise OSError("diagnostic is temporarily locked")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_diagnostic):
            failed = self.expire()
        self.assertTrue(failed["errors"])
        self.assertTrue(diagnostic.exists())
        self.assertEqual(failed["deleted_bytes"], other_bytes)
        self.assertFalse((other / "diagnostics.gz").exists())
        state = json.loads((directory / "closeout.json").read_text(encoding="utf-8"))
        self.assertEqual(state["disposition"]["status"], "failed")
        succeeded = self.expire()
        self.assertFalse(diagnostic.exists())
        self.assertEqual(succeeded["deleted_bytes"], actual_bytes)
        self.assertEqual(succeeded["deleted_runs"], 1)

    def test_only_the_declared_optional_stream_can_expire(self):
        cases = ("parent-path", "required-member", "not-optional", "undeclared", "missing-required-after-closeout")
        self.subtest_cases = cases
        for case_id in cases:
            with self.subTest(case=case_id):
                run_id = "run-" + case_id
                directory = self.create_run(run_id)
                self.accept(run_id)
                result_path = directory / "result.json"
                result = json.loads(result_path.read_bytes())
                if case_id == "missing-required-after-closeout":
                    (directory / "candidate.patch").unlink()
                else:
                    if case_id == "parent-path":
                        result["diagnostics"]["path"] = "../protected.gz"
                    elif case_id == "required-member":
                        result["diagnostics"]["path"] = "candidate.json"
                    elif case_id == "not-optional":
                        result["diagnostics"]["optional"] = False
                    else:
                        result.pop("diagnostics")
                    raw = json.dumps(result).encode()
                    result_path.write_bytes(raw)
                    closeout = directory / "closeout.json"
                    state = json.loads(closeout.read_bytes())
                    state["result_sha256"] = hashlib.sha256(raw).hexdigest()
                    closeout.write_text(json.dumps(state), encoding="utf-8")
                before = result_path.read_bytes()
                summary = self.expire()
                self.assertTrue(any(error["run"] == run_id for error in summary["errors"]))
                self.assertTrue((directory / "diagnostics.gz").exists())
                self.assertEqual(result_path.read_bytes(), before)

    def test_final_disposition_write_failure_is_recovered_after_unlink(self):
        run_id = "run-final-write-failure"
        directory = self.create_run(run_id)
        self.accept(run_id)
        diagnostic = directory / "diagnostics.gz"
        actual_bytes = diagnostic.stat().st_size
        original_replace = Path.replace
        failed_once = False

        def fail_final_write(path, target, *args, **kwargs):
            nonlocal failed_once
            if path.name == "closeout.tmp" and Path(target).name == "closeout.json" and not failed_once:
                failed_once = True
                raise OSError("interrupted closeout disposition")
            return original_replace(path, target, *args, **kwargs)

        closeout_path = directory / "closeout.json"
        replace_count = 0

        def fail_after_pending(path, target, *args, **kwargs):
            nonlocal replace_count
            if Path(target) == closeout_path:
                replace_count += 1
                if replace_count == 2:
                    return fail_final_write(path, target, *args, **kwargs)
            return original_replace(path, target, *args, **kwargs)

        with patch.object(Path, "replace", fail_after_pending):
            first = self.expire()
        self.assertTrue(first["errors"])
        self.assertFalse(diagnostic.exists())
        second = self.expire()
        state = json.loads((directory / "closeout.json").read_text(encoding="utf-8"))
        self.assertEqual(state["disposition"]["deleted_bytes"], actual_bytes)
        self.assertEqual(state["disposition"]["status"], "deleted")
        self.assertEqual(second["deleted_bytes"], 0)

    def test_expiry_visits_at_most_32_runs_and_rotation_survives_owner_release(self):
        report = self.runs / "verifier-scope-fixture.json"
        report.write_bytes(b"report-only evidence; never a run directory")
        run_ids = [f"run-batch-{index:02d}" for index in range(40)]
        for run_id in run_ids:
            self.create_run(run_id)
            self.accept(run_id)
        first = self.expire()
        self.assertEqual(first["visited"], 32)
        self.assertTrue(first["more"])
        self.assertLessEqual(first["visited"], 32)
        self.lease.__exit__(None, None, None)
        self.lease = execution.OutputLease(self.root, self.build)
        self.lease.__enter__()
        second = self.expire()
        self.assertEqual(second["visited"], 8)
        self.assertFalse(second["more"])
        self.assertEqual(first["deleted_runs"] + second["deleted_runs"], 40)
        self.assertTrue(all(not (self.runs / run_id / "diagnostics.gz").exists() for run_id in run_ids))
        self.assertEqual(report.read_bytes(), b"report-only evidence; never a run directory")

    def test_hold_and_expiry_share_the_preset_lease_across_processes(self):
        run_id = "run-serialized"
        directory = self.create_run(run_id)
        self.accept(run_id)
        script = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(Path(execution.__file__).parent)!r}); "
            "from execution import OutputLease; "
            f"root=Path({str(self.root)!r}); build=root/'out/build'/{self.preset!r}; "
            "\ntry:\n with OutputLease(root, build, timeout=0): print('acquired')"
            "\nexcept TimeoutError: print('excluded')"
        )
        environment = {key: value for key, value in os.environ.items() if key != "JUICER_OUTPUT_LEASES"}
        contender = subprocess.run(
            [sys.executable, "-B", "-c", script], env=environment, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(contender.returncode, 0, contender.stderr)
        self.assertEqual(contender.stdout.strip(), "excluded")
        execution.closeout_run(self.root, self.preset, run_id, hold=("reproduction",))
        retained = self.expire()
        self.assertEqual(retained["deleted_bytes"], 0)
        self.assertTrue((directory / "diagnostics.gz").exists())
        execution.closeout_run(self.root, self.preset, run_id, release_hold=("reproduction",))
        self.expire()
        self.assertFalse((directory / "diagnostics.gz").exists())

    def test_closeout_and_expiry_require_the_exact_preset_owner(self):
        run_id = "run-no-owner"
        self.create_run(run_id)
        self.lease.__exit__(None, None, None)
        self.lease = None
        with self.assertRaisesRegex(RuntimeError, "exact preset owner"):
            execution.closeout_run(self.root, self.preset, run_id)
        with self.assertRaisesRegex(RuntimeError, "exact preset owner"):
            execution.expire_diagnostics(self.root, self.preset)

    def test_blocked_owner_prevents_hold_changes_and_expiry(self):
        directory = self.create_run("run-blocked")
        self.accept("run-blocked")
        self.lease.blocked = True
        try:
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                execution.closeout_run(self.root, self.preset, "run-blocked", hold=("review",))
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                self.expire()
        finally:
            self.lease.blocked = False
        self.assertTrue((directory / "diagnostics.gz").exists())


NativeCloseoutTests.test_absent_invalid_and_reopened_closeout_do_not_expire.required_subtests = True
NativeCloseoutTests.test_each_hold_blocks_expiry_until_released.required_subtests = True
NativeCloseoutTests.test_failed_changed_and_undrained_results_cannot_be_accepted_or_expired.required_subtests = True
NativeCloseoutTests.test_missing_optional_diagnostic_is_accepted_but_missing_mandatory_member_is_rejected.required_subtests = True
NativeCloseoutTests.test_member_and_run_directory_links_are_rejected.required_subtests = True
NativeCloseoutTests.test_only_the_declared_optional_stream_can_expire.required_subtests = True


class WindowsVctipTests(unittest.TestCase):
    @staticmethod
    def required_subtests(method):
        method.required_subtests = True
        return method

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vctip controls ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    @unittest.skipUnless(os.name == "nt", "Windows compiler toolchain path")
    def test_selected_vctools_installation_resolves_exact_vctip_path(self):
        tools = Path(os.environ["VCToolsInstallDir"])
        expected = tools / "bin/Hostx64/x64/VCTIP.EXE"
        self.assertEqual(Path(execution.windows_vctip(os.environ)), expected)

    @unittest.skipUnless(os.name == "nt", "Windows Job VCTIP cleanup")
    def test_cleanup_only_terminates_the_selected_path_and_drains_other_children(self):
        system_root = Path(os.environ["SystemRoot"])
        ping = system_root / "System32/ping.exe"
        self.assertTrue(ping.is_file())
        vctip = self.root / "VCTIP.EXE"
        vctip.write_bytes(ping.read_bytes())
        build = self.root / "out/build/windows-clang-debug"
        build.mkdir(parents=True)
        directory = self.root / "out/validation/windows-clang-debug/results/run-one"
        directory.mkdir(parents=True)
        lease = execution.OutputLease(self.root, build)
        child = "import time;time.sleep(1.5);print('other child done',flush=True)"
        code = (
            "import subprocess,sys;"
            f"subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1']);"
            f"subprocess.Popen([sys.executable,'-B','-c',{child!r}])"
        )
        with lease:
            run = execution.NativeRun(self.root, directory)
            log = run.command(
                "cmake", [str(Path(sys.executable)), "-B", "-c", code], timeout=10, vctip_executable=str(vctip)
            )
        process = json.loads((directory / "cmake.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(len(process["vctip_terminated"]), 1)
        self.assertEqual(Path(process["vctip_terminated"][0]["path"]), vctip)
        self.assertIn("other child done", log.read_text())

    @unittest.skipUnless(os.name == "nt", "Windows Job VCTIP path scoping")
    def test_same_executable_name_at_another_path_is_not_terminated(self):
        ping = Path(os.environ["SystemRoot"]) / "System32/ping.exe"
        expected = self.root / "selected/VCTIP.EXE"
        other = self.root / "other/VCTIP.EXE"
        expected.parent.mkdir()
        other.parent.mkdir()
        expected.write_bytes(ping.read_bytes())
        other.write_bytes(ping.read_bytes())
        build = self.root / "out/build/windows-clang-debug"
        build.mkdir(parents=True)
        directory = self.root / "out/validation/windows-clang-debug/results/run-one"
        directory.mkdir(parents=True)
        with execution.OutputLease(self.root, build):
            run = execution.NativeRun(self.root, directory)
            code = f"import subprocess;subprocess.Popen([{str(other)!r},'-n','30','127.0.0.1'])"
            with self.assertRaises(subprocess.TimeoutExpired):
                run.command("test", [sys.executable, "-B", "-c", code], timeout=0.7, vctip_executable=str(expected))
        process = json.loads((directory / "test.process.json").read_text())
        self.assertEqual(process["vctip_terminated"], [])
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "timeout")


class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="Windows Job controls ")
        self.addCleanup(self.temporary.cleanup)
        self.job = process_tree.WindowsJob.__new__(process_tree.WindowsJob)
        self.job.job = 9
        self.job.api = Mock()
        self.job.api.OpenProcess.return_value = 99
        self.process = Mock(pid=42, _handle=7)
        self.process.poll.return_value = None
        self.popen = self.enterContext(patch.object(process_tree.subprocess, "Popen", return_value=self.process))

    def test_readable_pid_is_assigned_before_one_gate_release(self):
        errors = (FileNotFoundError("not published"), PermissionError("not readable"))
        self.subtest_cases = ("pid-file-not-published", "pid-file-not-readable")
        for case_id, error in zip(self.subtest_cases, errors):
            with self.subTest(case=case_id):
                self.popen.reset_mock()
                self.job.api.reset_mock()
                self.process.stdin.reset_mock()

                def assigned(*_):
                    self.assertFalse(self.process.stdin.write.called)
                    return True

                self.job.api.AssignProcessToJobObject.side_effect = assigned
                with (
                    patch.object(Path, "read_text", side_effect=[error, "43"]) as read,
                    patch.object(process_tree.time, "sleep") as sleep,
                ):
                    self.job.start(["command"], Path("gate.json"))
                self.assertEqual(read.call_count, 2)
                sleep.assert_called_once_with(0.01)
                self.popen.assert_called_once()
                self.job.api.AssignProcessToJobObject.assert_any_call(9, 7)
                self.job.api.AssignProcessToJobObject.assert_any_call(9, 99)
                self.assertEqual(self.job.api.AssignProcessToJobObject.call_count, 2)
                self.job.api.OpenProcess.assert_called_once_with(0x0101, False, 43)
                self.process.stdin.write.assert_called_once_with(b"g")

    def test_late_pid_or_job_assignment_cannot_release_gate(self):
        timing_cases = ((0.0, 0.0, 6.0), (0.0, 0.0, 0.0, 6.0))
        self.subtest_cases = tuple(f"timing-{index}" for index in range(len(timing_cases)))
        for case_id, times in zip(self.subtest_cases, timing_cases):
            with self.subTest(case=case_id):
                self.popen.reset_mock()
                self.process.reset_mock()
                with (
                    patch.object(Path, "read_text", return_value="43"),
                    patch.object(process_tree.time, "monotonic", side_effect=times),
                    self.assertRaisesRegex(RuntimeError, "readable process identity"),
                ):
                    self.job.start(["command"], Path("gate.json"))
                self.popen.assert_called_once()
                self.process.stdin.write.assert_not_called()
                self.process.kill.assert_called_once()
                self.process.wait.assert_called_once()

    def test_expired_gate_does_not_attempt_another_pid_read(self):
        with (
            patch.object(Path, "read_text", side_effect=[PermissionError("not ready"), "43"]) as read,
            patch.object(process_tree.time, "monotonic", side_effect=[0.0, 0.0, 6.0]),
            patch.object(process_tree.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "readable process identity"),
        ):
            self.job.start(["command"], Path("gate.json"))
        read.assert_called_once()
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_exit_during_pid_read_cannot_release_gate(self):
        self.process.poll.side_effect = [None, 7]
        with (
            patch.object(Path, "read_text", return_value="43"),
            self.assertRaisesRegex(RuntimeError, "readable process identity"),
        ):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_unreadable_pid_expires_without_releasing_or_relaunching(self):
        with (
            patch.object(Path, "read_text", side_effect=PermissionError("still unreadable")),
            patch.object(process_tree.time, "monotonic", side_effect=[0.0, 0.0, 6.0]),
            patch.object(process_tree.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "readable process identity.*still unreadable"),
        ):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_malformed_pid_fails_without_releasing_or_relaunching(self):
        with (
            patch.object(Path, "read_text", return_value="invalid"),
            patch.object(process_tree.time, "sleep") as sleep,
            self.assertRaises(ValueError),
        ):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()
        sleep.assert_not_called()

    def test_exited_gate_fails_without_releasing_or_relaunching(self):
        self.process.poll.return_value = 7
        with (
            patch.object(Path, "read_text", side_effect=FileNotFoundError("not published")),
            patch.object(process_tree.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "readable process identity"),
        ):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()
        sleep.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows Job process-list handling")
    def test_successful_partial_pid_list_is_requeried_before_termination(self):
        capacities = []

        def query(_job, _kind, buffer, size, _returned):
            capacity = (size - 8) // ctypes.sizeof(ctypes.c_size_t)
            capacities.append(capacity)
            first = len(capacities) == 1
            ctypes.c_uint32.from_buffer(buffer, 0).value = 17 if first else 0
            ctypes.c_uint32.from_buffer(buffer, 4).value = 16 if first else 0
            if first:
                pids = (ctypes.c_size_t * 16).from_buffer(buffer, 8)
                pids[:] = range(1, 17)
            return True

        self.job.api.QueryInformationJobObject.side_effect = query
        recorded = []
        self.job.stop_vctip(str(Path(self.temporary.name) / "VCTIP.EXE"), recorded.append)
        self.assertEqual(capacities, [16, 32])
        self.assertEqual(recorded, [])
        self.job.api.OpenProcess.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows Job termination recording")
    def test_vctip_termination_is_recorded_before_later_process_error(self):
        from ctypes import wintypes

        vctip = Path(self.temporary.name) / "VCTIP.EXE"
        vctip.write_bytes(b"selected executable identity")

        def query(_job, _kind, buffer, _size, _returned):
            ctypes.c_uint32.from_buffer(buffer, 0).value = 2
            ctypes.c_uint32.from_buffer(buffer, 4).value = 2
            pids = (ctypes.c_size_t * 2).from_buffer(buffer, 8)
            pids[:] = (111, 222)
            return True

        def open_process(_access, _inherit, pid):
            if pid == 111:
                return 101
            ctypes.set_last_error(5)
            return 0

        def is_in_job(_handle, _job, result):
            ctypes.cast(result, ctypes.POINTER(wintypes.BOOL))[0] = 1
            return True

        def image(_handle, _flags, buffer, length):
            buffer.value = str(vctip)
            ctypes.cast(length, ctypes.POINTER(wintypes.DWORD))[0] = len(str(vctip))
            return True

        self.job.api.QueryInformationJobObject.side_effect = query
        self.job.api.OpenProcess.side_effect = open_process
        self.job.api.IsProcessInJob.side_effect = is_in_job
        self.job.api.WaitForSingleObject.return_value = 0x102
        self.job.api.QueryFullProcessImageNameW.side_effect = image
        self.job.api.TerminateProcess.return_value = True
        self.job.api.CloseHandle.return_value = True
        recorded = []
        with self.assertRaises(OSError):
            self.job.stop_vctip(str(vctip), recorded.append)
        self.assertEqual(recorded, [{"pid": 111, "path": str(vctip)}])
        self.job.api.TerminateProcess.assert_called_once_with(101, 125)


WindowsJobTests.test_readable_pid_is_assigned_before_one_gate_release.required_subtests = True
WindowsJobTests.test_readable_pid_is_assigned_before_one_gate_release.subtest_cases = (
    "pid-file-not-published",
    "pid-file-not-readable",
)
WindowsJobTests.test_late_pid_or_job_assignment_cannot_release_gate.required_subtests = True
WindowsJobTests.test_late_pid_or_job_assignment_cannot_release_gate.subtest_cases = ("timing-0", "timing-1")


if __name__ == "__main__":
    unittest.main()
