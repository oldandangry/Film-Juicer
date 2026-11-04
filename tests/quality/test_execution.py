"""Native command ownership, drainage, cancellation and Windows Job controls."""

from __future__ import annotations

import ctypes
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


class NativeRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native run ü ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "out/build" / (
            "windows-clang-debug" if os.name == "nt" else "linux-debug"
        )
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
        command = [self.python, "-B", "-c",
                   "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "])"]
        log = self.native_run.command("build", command, timeout=10)
        process = json.loads((self.directory / "build.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "completed")
        self.assertIn("last descendant output", log.read_text())
        self.native_run.publish({"schema": "native-run-v1", "commands": self.native_run.commands})
        self.assertEqual(json.loads((self.directory / "result.json").read_text())["schema"], "native-run-v1")

    def test_nonzero_parent_drains_descendant_and_keeps_failure_visible(self):
        code = ("import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)']);"
                "sys.exit(7)")
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
        child = ("import time;from pathlib import Path;p=Path(" + repr(str(marker)) + ");"
                 "\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)")
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "]);time.sleep(20)"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.native_run.command("test", [self.python, "-B", "-c", "import time;" + code], timeout=.4)
        process = json.loads((self.directory / "test.process.json").read_text())
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "timeout")
        final = marker.read_bytes()
        time.sleep(.05)
        self.assertEqual(marker.read_bytes(), final)
        self.assertFalse((self.directory / "running.lock").exists())

    def test_keyboard_interrupt_drains_real_descendant_before_returning(self):
        marker = self.root / "interrupt-writer.txt"
        child = ("import time;from pathlib import Path;p=Path(" + repr(str(marker)) + ");"
                 "\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)")
        code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "]);time.sleep(20)"
        original_wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_after_child(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
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
        time.sleep(.05)
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
        code = ("import subprocess,sys;"
                f"subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1']);"
                f"subprocess.Popen([sys.executable,'-B','-c',{child!r}])")
        with lease:
            run = execution.NativeRun(self.root, directory)
            log = run.command("cmake", [str(Path(sys.executable)), "-B", "-c", code],
                              timeout=10, vctip_executable=str(vctip))
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
                run.command("test", [sys.executable, "-B", "-c", code], timeout=.7,
                            vctip_executable=str(expected))
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
                with patch.object(Path, "read_text", side_effect=[error, "43"]) as read, \
                     patch.object(process_tree.time, "sleep") as sleep:
                    self.job.start(["command"], Path("gate.json"))
                self.assertEqual(read.call_count, 2)
                sleep.assert_called_once_with(.01)
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
                with patch.object(Path, "read_text", return_value="43"), \
                     patch.object(process_tree.time, "monotonic", side_effect=times), \
                     self.assertRaisesRegex(RuntimeError, "readable process identity"):
                    self.job.start(["command"], Path("gate.json"))
                self.popen.assert_called_once()
                self.process.stdin.write.assert_not_called()
                self.process.kill.assert_called_once()
                self.process.wait.assert_called_once()

    def test_expired_gate_does_not_attempt_another_pid_read(self):
        with patch.object(Path, "read_text", side_effect=[PermissionError("not ready"), "43"]) as read, \
             patch.object(process_tree.time, "monotonic", side_effect=[0.0, 0.0, 6.0]), \
             patch.object(process_tree.time, "sleep"), \
             self.assertRaisesRegex(RuntimeError, "readable process identity"):
            self.job.start(["command"], Path("gate.json"))
        read.assert_called_once()
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_exit_during_pid_read_cannot_release_gate(self):
        self.process.poll.side_effect = [None, 7]
        with patch.object(Path, "read_text", return_value="43"), \
             self.assertRaisesRegex(RuntimeError, "readable process identity"):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_unreadable_pid_expires_without_releasing_or_relaunching(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("still unreadable")), \
             patch.object(process_tree.time, "monotonic", side_effect=[0.0, 0.0, 6.0]), \
             patch.object(process_tree.time, "sleep"), \
             self.assertRaisesRegex(RuntimeError, "readable process identity.*still unreadable"):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()

    def test_malformed_pid_fails_without_releasing_or_relaunching(self):
        with patch.object(Path, "read_text", return_value="invalid"), \
             patch.object(process_tree.time, "sleep") as sleep, self.assertRaises(ValueError):
            self.job.start(["command"], Path("gate.json"))
        self.popen.assert_called_once()
        self.process.stdin.write.assert_not_called()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once()
        sleep.assert_not_called()

    def test_exited_gate_fails_without_releasing_or_relaunching(self):
        self.process.poll.return_value = 7
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("not published")), \
             patch.object(process_tree.time, "sleep") as sleep, \
             self.assertRaisesRegex(RuntimeError, "readable process identity"):
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
    "pid-file-not-published", "pid-file-not-readable"
)
WindowsJobTests.test_late_pid_or_job_assignment_cannot_release_gate.required_subtests = True
WindowsJobTests.test_late_pid_or_job_assignment_cannot_release_gate.subtest_cases = (
    "timing-0", "timing-1"
)


if __name__ == "__main__":
    unittest.main()
