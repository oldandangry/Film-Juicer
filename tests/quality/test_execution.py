from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[2] / "tests/ctest"))
from control_artifacts import record_fixture

SPEC = importlib.util.spec_from_file_location(
    "campaign", Path(__file__).parents[2] / "scripts/execution.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
PROCESS_SPEC = importlib.util.spec_from_file_location(
    "process_tree", Path(campaign.__file__).with_name("process_tree.py")
)
process_tree = importlib.util.module_from_spec(PROCESS_SPEC)
PROCESS_SPEC.loader.exec_module(process_tree)


class CampaignTests(unittest.TestCase):
    def test_windows_identity_route_requires_bound_request_and_collection_stage(self):
        with patch.object(campaign.subprocess, "Popen", side_effect=AssertionError("unbound launch")):
            with self.assertRaisesRegex(ValueError, "Windows identity"):
                self.campaign.run("quality", self.command("pass"), inputs=["source.txt"],
                                  windows_python=r"C:\Python\python.exe")

    def test_default_supervision_allows_progress_and_quiet_work_without_elapsed_kill(self):
        code = ("import time\n"
                "for i in range(8):\n"
                " print(i, flush=True);time.sleep(.04)\n"
                "time.sleep(.2)\nprint('finished', flush=True)\n")
        result = self.campaign.run("quality", self.command(code), inputs=["source.txt"])
        raw = json.loads(result.read_text())
        self.assertIsNone(raw["timeout"])
        self.assertEqual(raw["process_tree"]["status"], "completed")
        self.assertTrue(raw["process_tree"]["drained"])
        self.assertIn("finished", (self.root / raw["output"]).read_text())

    @unittest.skipUnless(os.name == "nt", "Windows Job descendant drainage")
    def test_unbounded_build_waits_for_progressing_descendants_after_parent_exit(self):
        child = "import time;print('child start',flush=True);time.sleep(.8);print('child done',flush=True)"
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-u','-c'," + repr(child) + "])"
        result = self.campaign.run("quality", self.command(code), inputs=["source.txt"],
            vctip_executable=str(self.vctip_fixture()), vctip_grace_seconds=.1, vctip_drain_cap_seconds=.3)
        raw = json.loads(result.read_text())
        self.assertEqual(raw["process_tree"]["status"], "completed")
        self.assertIsNone(raw["identity"]["vctip_cleanup"]["drain_cap_seconds"])
        self.assertIn("child done", (self.root / raw["output"]).read_text())

    def test_unbounded_lease_rejects_known_blocked_owner_without_waiting(self):
        lease = campaign.OutputLease(self.root, self.root / "blocked-build")
        lease.path.mkdir(parents=True)
        (lease.path / "owner.json").write_text(json.dumps({"status": "blocked"}))
        with self.assertRaisesRegex(RuntimeError, "owner is blocked"):
            lease.__enter__()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "source.txt").write_text("original", encoding="utf-8")
        self.campaign = campaign.Campaign(
            self.root, "linux-debug", "example", ["quality", "tests"]
        )

    def command(self, code):
        # Store app aliases activate through an external broker; qualify native
        # Job inheritance with the actual installed interpreter.
        python = (
            str(Path(sys.base_prefix) / "python.exe")
            if os.name == "nt"
            else sys.executable
        )
        return [python, "-c", code]

    def vctip_fixture(self, directory=None):
        source = Path(os.environ["SystemRoot"]) / "System32/ping.exe"
        target_dir = self.root if directory is None else directory
        target_dir.mkdir(exist_ok=True)
        target = target_dir / "VCTIP.EXE"
        shutil.copy2(source, target)
        return target

    def execute(self, code, **kwargs):
        return self.campaign.run(
            "quality", self.command(code), inputs=["source.txt"], timeout=10, **kwargs
        )

    def test_failure_resume_retains_original_evidence_and_required_gates(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute("print('original failure'); raise SystemExit(7)")
        before = {
            str(p): p.read_bytes()
            for p in self.campaign.directory.glob("attempts/*/*")
            if p.is_file()
        }
        resumed = campaign.Campaign(
            self.root, "linux-debug", "example", ["quality", "tests"]
        )
        resumed.run(
            "quality", self.command("print('pass')"), inputs=["source.txt"], timeout=10
        )
        self.assertEqual(
            resumed.report()["stages"], {"quality": "passed", "tests": "outstanding"}
        )
        self.assertFalse(resumed.report()["all_recorded_gates_passed"])
        for path, content in before.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_repeated_pass_requires_reason_and_never_returns_a_cached_pass(self):
        self.execute("print('pass')")
        with self.assertRaisesRegex(ValueError, "already passed"):
            self.execute("print('pass')")
        second = self.execute(
            "print('pass')", repeat_reason="fresh-process determinism"
        )
        self.assertEqual(
            json.loads(second.read_text())["repeat_reason"], "fresh-process determinism"
        )
        self.assertEqual(len(self.campaign.attempts("quality")), 2)

    def test_clock_rollback_does_not_select_an_older_attempt(self):
        first = self.execute("print('first')")
        preserved = {p: p.read_bytes() for p in first.parent.iterdir() if p.is_file()}
        with patch.object(campaign, "datetime") as clock:
            clock.now.return_value = datetime(2000, 1, 1, tzinfo=timezone.utc)
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute("raise SystemExit(7)")
        attempts = self.campaign.attempts("quality")
        self.assertEqual([a["attempt_number"] for a in attempts], [1, 2])
        self.assertGreater(attempts[0]["started_at"], attempts[1]["started_at"])
        self.assertEqual(self.campaign.report()["stages"]["quality"], "failed")
        self.execute("print('retry')")
        self.assertEqual(self.campaign.report()["stages"]["quality"], "passed")
        self.assertEqual(
            [a["attempt_number"] for a in self.campaign.attempts("quality")], [1, 2, 3]
        )
        for path, content in preserved.items():
            self.assertEqual(path.read_bytes(), content)

    def test_orchestration_id_cannot_reach_child_processes(self):
        with patch.dict(os.environ, {"CODEX_TOOL_CALL_ID": "desktop-call"}):
            result = self.campaign.run("quality", self.command(
                "import os; assert 'CODEX_TOOL_CALL_ID' not in os.environ; print('checked')"
            ), inputs=["source.txt"])
        self.assertEqual(json.loads(result.read_text())["status"], "passed")

    def test_source_changes_invalidate_report_and_allow_fresh_execution(self):
        self.execute("print('pass')")
        (self.root / "source.txt").write_text("changed", encoding="utf-8")
        self.assertEqual(self.campaign.report()["stages"]["quality"], "inputs-changed")
        self.execute("print('pass')")
        self.assertEqual(self.campaign.report()["stages"]["quality"], "passed")

    def test_mutation_during_a_command_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.execute(
                "from pathlib import Path; Path('source.txt').write_text('changed')"
            )
        self.assertEqual(self.campaign.report()["stages"]["quality"], "failed")

    def test_tool_mutation_during_a_command_cannot_pass(self):
        tool = self.root / "tool.txt"
        tool.write_text("first", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.execute(
                "import os; from pathlib import Path; p=Path('tool.txt'); "
                "s=p.stat(); p.write_text('other', encoding='utf-8'); "
                "os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))",
                tools=[str(tool), str(self.root) + "/./tool.txt"],
            )
        self.assertEqual(self.campaign.report()["stages"]["quality"], "failed")

    def test_changed_prepared_tool_is_rejected_before_launch(self):
        tool = self.root / "tool.txt"
        tool.write_text("first", encoding="utf-8")
        tools = [str(tool), str(self.root) + "/./tool.txt"]
        prepared = campaign.tool_snapshot(tools)
        stamp = tool.stat()
        tool.write_text("other", encoding="utf-8")
        os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "prepared tools changed before launch"):
            self.execute(
                "from pathlib import Path; Path('launched').touch()",
                tools=tools,
                expected_tools=prepared,
            )
        self.assertFalse((self.root / "launched").exists())
        self.assertFalse(self.campaign.attempts("quality"))

    def test_modified_output_does_not_remain_passing_evidence(self):
        receipt = self.execute("print('pass')")
        (receipt.parent / "output.log").write_text("changed", encoding="utf-8")
        self.assertEqual(
            self.campaign.report()["stages"]["quality"], "evidence-changed"
        )

    def test_scope_is_explicit_and_missing_files_do_not_pass(self):
        with self.assertRaisesRegex(ValueError, "not a file"):
            campaign.snapshot(self.root, ["out"])
        with self.assertRaisesRegex(ValueError, "not a file"):
            campaign.snapshot(self.root, ["missing"])
        self.assertEqual(
            list(campaign.snapshot(self.root, ["source.txt"])), ["source.txt"]
        )

    def test_existing_lock_and_changed_gate_contract_are_rejected(self):
        lock = self.campaign.directory / "running.lock"
        lock.write_text("other process", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.execute("print('pass')")
        self.assertEqual(lock.read_text(), "other process")
        with self.assertRaisesRegex(ValueError, "different"):
            campaign.Campaign(self.root, "linux-debug", "example", ["quality"])

    def test_stage_prefixes_do_not_borrow_each_others_results(self):
        run = campaign.Campaign(
            self.root, "linux-debug", "prefixes", ["quality", "quality-extra"]
        )
        run.run(
            "quality-extra",
            self.command("print('pass')"),
            inputs=["source.txt"],
            timeout=10,
        )
        self.assertEqual(run.attempts("quality"), [])
        self.assertEqual(run.report()["stages"]["quality"], "outstanding")
        run.run(
            "quality", self.command("print('pass')"), inputs=["source.txt"], timeout=10
        )
        self.assertTrue(run.report()["all_recorded_gates_passed"])

    def test_timeout_is_failed_and_releases_the_campaign_lock(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.campaign.run(
                "quality",
                self.command("import time; time.sleep(5)"),
                inputs=["source.txt"],
                timeout=0.1,
            )
        self.assertEqual(self.campaign.report()["stages"]["quality"], "failed")
        self.assertFalse((self.campaign.directory / "running.lock").exists())

    def writer_command(self, parent_wait=True):
        child = (
            "import os,time; from pathlib import Path; Path('child.pid').write_text(str(os.getpid())); "
            "[(print('child writer',flush=True),time.sleep(0.02)) for _ in range(1500)]"
        )
        options = ", start_new_session=True" if os.name != "nt" else ""
        code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]{options}); "
        return self.command(code + ("time.sleep(30)" if parent_wait else ""))

    def assert_drained(self):
        record = self.campaign.attempts("quality")[-1]
        self.assertTrue(record["process_tree"]["drained"])
        self.assertFalse((self.campaign.directory / "running.lock").exists())
        log = self.root / record["output"]
        before = log.read_bytes()
        time.sleep(0.2)
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(campaign.digest(log), record["output_sha256"])
        self.assertIn(b"child writer", before)
        pid = int((self.root / "child.pid").read_text())
        if os.name != "nt":
            self.assertFalse(Path(f"/proc/{pid}").exists(), "descendant was not reaped")
        else:
            import ctypes

            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.OpenProcess.restype = ctypes.c_void_p
            api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            api.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = api.OpenProcess(0x100000, False, pid)
            if handle:
                try:
                    self.assertEqual(api.WaitForSingleObject(handle, 0), 0)
                finally:
                    api.CloseHandle(handle)

    def test_timeout_drains_descendant_writers_even_after_parent_exit(self):
        for parent_wait in (True, False):
            with (
                self.subTest(parent_wait=parent_wait),
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                self.campaign.run(
                    "quality",
                    self.writer_command(parent_wait),
                    inputs=["source.txt"],
                    timeout=1,
                )
            self.assert_drained()
        self.execute("print('safe retry')")

    def test_interruption_drains_real_descendant_before_unlock(self):
        original = subprocess.Popen.wait
        interrupted = False

        def wait(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                deadline = time.monotonic() + 5
                while (
                    not (self.root / "child.pid").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                interrupted = True
                raise KeyboardInterrupt()
            return original(process, *args, **kwargs)

        with (
            patch.object(subprocess.Popen, "wait", wait),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.campaign.run(
                "quality", self.writer_command(), inputs=["source.txt"], timeout=10
            )
        self.assert_drained()

    def test_normal_parent_exit_waits_for_descendant_completion(self):
        child = (
            "import time; time.sleep(.3); print('last descendant output',flush=True)"
        )
        receipt = self.execute(
            f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
        )
        self.assertIn(
            "last descendant output", (receipt.parent / "output.log").read_text()
        )
        self.assertTrue(json.loads(receipt.read_text())["process_tree"]["drained"])

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_alone_finishes_without_reaching_drain_cap(self):
        vctip = self.vctip_fixture()
        code = f"import subprocess;subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1'])"
        receipt = self.campaign.run(
            "quality", self.command(code), inputs=["source.txt"],
            timeout=3, vctip_executable=str(vctip), vctip_grace_seconds=0.1,
        )
        result = json.loads(receipt.read_text())
        tree = result["process_tree"]
        self.assertEqual(result["status"], "passed")
        self.assertTrue(tree["drained"])
        self.assertEqual(len(tree["vctip_terminated"]), 1)
        self.assertLess(tree["drain_seconds"], 2)
        self.assertEqual(tree["vctip_cleanup"]["drain_cap_seconds"], 300)

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_cleanup_is_scoped_and_waits_for_other_child(self):
        vctip = self.vctip_fixture()
        other = "import time;time.sleep(1.0);print('other child done',flush=True)"
        code = (
            "import subprocess,sys;"
            f"subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1']);"
            f"subprocess.Popen([sys.executable,'-c',{other!r}])"
        )
        receipt = self.campaign.run(
            "quality", self.command(code), inputs=["source.txt"],
            timeout=5, vctip_executable=str(vctip), vctip_grace_seconds=0.1,
        )
        result = json.loads(receipt.read_text())
        tree = result["process_tree"]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(tree["status"], "completed")
        self.assertTrue(tree["drained"])
        self.assertEqual(len(tree["vctip_terminated"]), 1)
        self.assertEqual(Path(tree["vctip_terminated"][0]["path"]), vctip)
        self.assertGreater(tree["drain_seconds"], 0.7)
        self.assertIn("other child done", (receipt.parent / "output.log").read_text())

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_cleanup_rejects_same_name_at_other_path(self):
        expected = self.vctip_fixture()
        other = self.vctip_fixture(self.root / "other")
        code = f"import subprocess;subprocess.Popen([{str(other)!r},'-n','30','127.0.0.1'])"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.campaign.run(
                "quality", self.command(code), inputs=["source.txt"],
                timeout=0.6, vctip_executable=str(expected),
                vctip_grace_seconds=0.1,
            )
        result = self.campaign.attempts("quality")[-1]
        self.assertEqual(result["process_tree"]["vctip_terminated"], [])
        self.assertEqual(result["process_tree"]["status"], "timeout")
        self.assertTrue(result["process_tree"]["drained"])
        self.assertFalse((self.campaign.directory / "running.lock").exists())

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_cleanup_caps_other_child_drain_after_parent(self):
        vctip = self.vctip_fixture()
        code = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.campaign.run(
                "quality", self.command(code), inputs=["source.txt"],
                timeout=5, vctip_executable=str(vctip),
                vctip_grace_seconds=0.1, vctip_drain_cap_seconds=0.6,
            )
        result = self.campaign.attempts("quality")[-1]
        tree = result["process_tree"]
        self.assertEqual(tree["status"], "timeout")
        self.assertTrue(tree["drained"])
        self.assertEqual(tree["vctip_terminated"], [])
        self.assertLess(result["seconds"], 3)

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_cleanup_retries_successful_partial_process_list(self):
        job = process_tree.WindowsJob.__new__(process_tree.WindowsJob)
        job.job = 1
        job.api = Mock()
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

        job.api.QueryInformationJobObject.side_effect = query
        recorded = []
        job.stop_vctip(str(self.root / "VCTIP.EXE"), recorded.append)
        self.assertEqual(capacities, [16, 32])
        self.assertEqual(recorded, [])
        job.api.OpenProcess.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_vctip_termination_is_recorded_before_later_process_error(self):
        from ctypes import wintypes

        vctip = self.vctip_fixture()
        job = process_tree.WindowsJob.__new__(process_tree.WindowsJob)
        job.job = 1
        job.api = Mock()

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

        job.api.QueryInformationJobObject.side_effect = query
        job.api.OpenProcess.side_effect = open_process
        job.api.IsProcessInJob.side_effect = is_in_job
        job.api.WaitForSingleObject.return_value = 0x102
        job.api.QueryFullProcessImageNameW.side_effect = image
        job.api.TerminateProcess.return_value = True
        job.api.CloseHandle.return_value = True
        recorded = []
        with self.assertRaises(OSError):
            job.stop_vctip(str(vctip), recorded.append)
        self.assertEqual(recorded, [{"pid": 111, "path": str(vctip)}])
        job.api.TerminateProcess.assert_called_once_with(101, 125)

    @unittest.skipUnless(os.name == "nt", "Windows Job policy")
    def test_failed_parent_never_cleans_vctip_as_a_pass(self):
        vctip = self.vctip_fixture()
        code = (
            "import subprocess,sys;"
            f"subprocess.Popen([{str(vctip)!r},'-n','30','127.0.0.1']);"
            "sys.exit(7)"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            self.campaign.run(
                "quality", self.command(code), inputs=["source.txt"],
                timeout=5, vctip_executable=str(vctip),
                vctip_grace_seconds=0.1,
            )
        result = self.campaign.attempts("quality")[-1]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["process_tree"]["status"], "failed-parent")
        self.assertEqual(result["process_tree"]["vctip_terminated"], [])
        self.assertTrue(result["process_tree"]["drained"])
        self.assertFalse((self.campaign.directory / "running.lock").exists())

    def test_unknown_completion_retains_blocked_lock_without_final_hash(self):
        def unknown(command, root, environment, output, timeout, receipt, lock, tree):
            tree.update(drained=False, supervisor_pid=123, receipt=str(receipt))
            raise RuntimeError("cannot establish completion")

        with (
            patch.object(campaign, "run_process_tree", unknown),
            self.assertRaises(RuntimeError),
        ):
            self.execute("print('unused')")
        record = self.campaign.attempts("quality")[-1]
        self.assertEqual(record["status"], "blocked")
        self.assertNotIn("output_sha256", record)
        lock = json.loads((self.campaign.directory / "running.lock").read_text())
        self.assertEqual(lock["supervisor_pid"], 123)
        with self.assertRaises(FileExistsError):
            self.execute("print('retry')")
        self.assertFalse(self.campaign.report()["all_recorded_gates_passed"])

    def test_nonfinite_timeout_is_rejected_before_launch(self):
        for timeout in (float("nan"), float("inf"), -float("inf")):
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                self.campaign.run(
                    "quality",
                    self.command("print('unused')"),
                    inputs=["source.txt"],
                    timeout=timeout,
                )
        self.assertEqual(self.campaign.attempts("quality"), [])

    def test_app_execution_alias_is_rejected_before_launch(self):
        from types import SimpleNamespace

        with patch.object(
            Path, "lstat", return_value=SimpleNamespace(st_reparse_tag=0x8000001B)
        ):
            with self.assertRaisesRegex(ValueError, "app execution alias"):
                campaign.check_windows_launcher(Path("broker.exe"))

    def test_failed_parent_cancels_owned_lingering_writers_within_fifteen_seconds(self):
        command = self.writer_command(False)
        command[-1] += " time.sleep(.2); sys.exit(7)"
        started = time.monotonic()
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.campaign.run("quality", command, inputs=["source.txt"], timeout=30)
        self.assertEqual(failure.exception.returncode, 7)
        self.assertLess(time.monotonic() - started, 15)
        self.assert_drained()
        tree = self.campaign.attempts("quality")[-1]["process_tree"]
        self.assertEqual(tree["first_returncode"], 7)
        self.assertEqual(tree["status"], "failed-parent")

    def test_blocked_held_owner_rejects_nested_and_new_command_admission(self):
        owner = campaign.OutputLease(self.root, self.root / "build")
        with owner:
            owner.blocked = True
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                with campaign.OutputLease(self.root, self.root / "build"):
                    self.fail("blocked owner admitted another operation")
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                self.execute("print('must not run')")
            self.assertEqual(self.campaign.attempts("quality"), [])
        self.assertTrue(owner.path.exists())

    def test_blocked_delegated_owner_rejects_child_admission(self):
        owner = campaign.OutputLease(self.root, self.root / "build")
        with owner:
            path = owner.path / "owner.json"
            record = json.loads(path.read_text())
            campaign.write_json(path, {**record, "status": "blocked"})
            source = Path(__file__).resolve().parents[2] / "scripts"
            code = (
                "import sys;sys.path.insert(0,"
                + repr(str(source))
                + ");from pathlib import Path;"
                "from execution import OutputLease;OutputLease(Path("
                + repr(str(self.root))
                + "),Path("
                + repr(str(self.root / "build"))
                + ")).__enter__();print('admitted')"
            )
            result = subprocess.run(
                self.command(code), capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output owner is blocked", result.stderr)
            self.assertNotIn("admitted", result.stdout)
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                with campaign.OutputLease(self.root, self.root / "build"):
                    self.fail("child block was not observed")
        self.assertTrue(owner.path.exists())

    def test_unproved_drainage_retains_enclosing_output_lease(self):
        def blocked(command, root, environment, output, timeout, receipt, lock, tree):
            tree["drained"] = False
            raise RuntimeError("controlled unresolved drainage")

        build = self.root / "build"
        lease = campaign.OutputLease(self.root, build)
        with (
            self.assertRaises(RuntimeError),
            lease,
            patch.object(campaign, "run_process_tree", blocked),
        ):
            self.execute("print('unreachable')")
        self.assertTrue(lease.path.is_dir())
        self.assertEqual(
            json.loads((lease.path / "owner.json").read_text())["status"], "blocked"
        )

    def test_owner_death_requests_cleanup_but_retains_recovery_lock(self):
        module = Path(campaign.__file__)
        owner_script = self.root / "owner.py"
        owner_script.write_text(
            "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "from execution import Campaign,OutputLease;"
            'root=Path(sys.argv[2]);c=Campaign(root,"linux-debug","dead-owner",["run"]);'
            'lease=OutputLease(root,root/"build");lease.__enter__();'
            'c.run("run",[sys.executable,"-c","import time;time.sleep(30)"],'
            'inputs=["source.txt"],timeout=60)',
            encoding="utf-8",
        )
        owner = subprocess.Popen(
            [
                self.command("")[0],
                "-B",
                str(owner_script),
                str(module.parent),
                str(self.root),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        directory = self.root / "out/validation/linux-debug/dead-owner"
        deadline = time.monotonic() + 10
        receipts = []
        while time.monotonic() < deadline:
            receipts = list(directory.glob("attempts/*/process.json"))
            if receipts and json.loads(receipts[0].read_text()).get("command_pid"):
                break
            time.sleep(0.05)
        self.assertTrue(receipts)
        owner.kill()
        owner.wait(timeout=5)
        while time.monotonic() < deadline:
            process = json.loads(receipts[0].read_text())
            if process["drained"]:
                break
            time.sleep(0.05)
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "interrupted")
        self.assertTrue((directory / "running.lock").exists())
        lease = campaign.OutputLease(self.root, self.root / "build", timeout=0.1)
        with self.assertRaises(TimeoutError):
            lease.__enter__()


def link_directory(link, target):
    if os.name == "nt":
        command = Path(os.environ["SystemRoot"]) / "System32/cmd.exe"
        subprocess.run(
            [str(command), "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=True,
            timeout=10,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def unlink_directory(link):
    if os.name == "nt":
        link.rmdir()
    else:
        link.unlink()


@unittest.skipUnless(os.name == "posix" and os.environ.get("JUICER_WINDOWS_PYTHON"),
                     "requires explicit qualified Windows Python under WSL")
class WindowsIdentityInteropTests(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[2]
        parent = repository / "out/validation/linux-debug/checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="identity-", dir=parent)).resolve()
        print("Windows identity fixture:", self.root, file=sys.stderr, flush=True)
        record_fixture(self.root, self, "running")
        self.addCleanup(lambda: record_fixture(self.root, self, owner_root=self.root))
        (self.root / "scripts").mkdir()
        for name in ("execution.py", "process_tree.py", "build_dependencies.py"):
            shutil.copy2(repository / "scripts" / name, self.root / "scripts" / name)
        self.python = os.environ["JUICER_WINDOWS_PYTHON"]
        self.input = self.root / "input.txt"
        self.input.write_text("original")
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({"protocol": "windows-file-identities-v1",
            "collection_id": "a" * 32, "phase": "entry", "proofs": {"b" * 64: {
                "files": [campaign.windows_path(self.input)], "rust_configuration": []}}}))
        self.owner = campaign.Campaign(self.root, "linux-debug", "collection",
                                       ["windows-entry", "windows-terminal"])

    def reader_prefix(self, source):
        helper = self.root / "scripts/build_dependencies.py"
        helper.write_text(helper.read_text().replace("from __future__ import annotations\n",
            "from __future__ import annotations\n\nif __name__ == '__main__':\n"
            "    import subprocess,sys,time\n    from pathlib import Path\n" +
            "\n".join("    " + line for line in source.splitlines()) + "\n", 1))

    def execute(self, timeout=10):
        launch = campaign.prepare_windows_identity_launch(self.root, self.python, self.request)
        return self.owner.run("windows-entry", launch["command"],
            inputs=["request.json", "scripts/execution.py", "scripts/process_tree.py",
                    "scripts/build_dependencies.py"], tools=launch["tools"], timeout=timeout,
            windows_python=self.python, identity_request=self.request)

    def test_native_identity_and_descendants_complete_before_release(self):
        marker = campaign.windows_path(self.root / "descendant.txt")
        child = "import time;from pathlib import Path;time.sleep(.8);Path(" + repr(marker) + ").write_text('finished')"
        self.reader_prefix("subprocess.Popen([sys.executable, '-B', '-c', " + repr(child) + "])")
        with campaign.OutputLease(self.root, self.root / "build"):
            path = self.execute()
        record = json.loads(path.read_text())
        tree = record["process_tree"]
        self.assertEqual(tree["status"], "completed")
        self.assertTrue(tree["drained"])
        self.assertEqual(set(tree["creation"]), {"windows_creation_time_100ns"})
        self.assertIn("interop_pid", tree)
        launch = record["identity"]["windows_identity"]
        self.assertEqual(launch["supervisor_executable"], str(campaign.mapped_windows_path(launch["base_target"])))
        self.assertEqual(tree["identity_request"]["request_sha256"], campaign.digest(self.request))
        self.assertEqual((self.root / "descendant.txt").read_text(), "finished")
        response = json.loads(path.with_name("output.log").read_text())
        self.assertEqual(response["proofs"]["b" * 64]["files"][campaign.windows_path(self.input)]["sha256"],
                         campaign.digest(self.input))
        self.assertFalse((self.owner.directory / "running.lock").exists())

    def test_native_failed_parent_drains_descendant_writer(self):
        marker = self.root / "grandchild.txt"
        grandchild = "import time;from pathlib import Path;p=Path(" + repr(campaign.windows_path(marker)) + ");\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        child = "import subprocess,sys;subprocess.Popen([sys.executable,'-B','-c'," + repr(grandchild) + "])"
        self.reader_prefix("subprocess.Popen([sys.executable, '-B', '-c', " + repr(child) + "])\n"
            "deadline=time.monotonic()+5\n"
            "while not Path(" + repr(campaign.windows_path(marker)) + ").exists():\n"
            "    if time.monotonic()>deadline: raise RuntimeError('grandchild did not start')\n"
            "    time.sleep(.02)\nraise SystemExit(7)")
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute()
        record = self.owner.attempts("windows-entry")[-1]
        self.assertEqual(record["returncode"], 7)
        self.assertEqual(record["process_tree"]["status"], "failed-parent")
        self.assertTrue(record["process_tree"]["drained"])
        before = marker.read_bytes()
        time.sleep(.1)
        self.assertEqual(marker.read_bytes(), before)

    def test_native_timeout_drains_writer_and_releases_proved_lease(self):
        marker = campaign.windows_path(self.root / "writer.txt")
        child = "import time;from pathlib import Path;p=Path(" + repr(marker) + ");\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        self.reader_prefix("subprocess.Popen([sys.executable, '-B', '-c', " + repr(child) + "])\ntime.sleep(30)")
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.execute(timeout=1)
        record = self.owner.attempts("windows-entry")[-1]
        self.assertEqual(record["process_tree"]["status"], "timeout")
        self.assertTrue(record["process_tree"]["drained"])
        marker = self.root / "writer.txt"
        before = marker.read_bytes()
        time.sleep(.1)
        self.assertEqual(marker.read_bytes(), before)

    def native_state(self, target=None):
        import qualification

        name = campaign.windows_path(self.input if target is None else target)
        expected = {"b" * 64: {"files": {name: {
            "resolved": campaign.windows_path(self.input), "sha256": campaign.digest(self.input)}},
            "rust_configuration": {}}}
        self.bind_runtime(expected["b" * 64]["files"])
        return qualification, {"campaign": self.owner, "python": self.python,
            "collection_id": "a" * 32, "expected": expected, "receipts": {}, "stages": []}

    def bind_runtime(self, files):
        launch = campaign.prepare_windows_identity_launch(self.root, self.python, self.request)
        for role in ("python", "base_python"):
            name = launch[role]
            files[name] = {"resolved": name, "sha256": campaign.digest(campaign.mapped_windows_path(name))}

    def copy_runtime(self):
        runtime = campaign.windows_identity_interpreter(self.python)
        self.assertIsNotNone(runtime["config"])
        executable = self.root / "runtime/Scripts/python.exe"
        executable.parent.mkdir(parents=True)
        shutil.copy2(runtime["executable"], executable)
        config = executable.parent.parent / "pyvenv.cfg"
        shutil.copy2(runtime["config"], config)
        self.python = campaign.windows_path(executable)
        return config

    def bind_runtime_fixture(self, config):
        qualification, native = self.native_state()
        files = native["expected"]["b" * 64]["files"]
        for path in (config, self.root / "scripts/build_dependencies.py"):
            name = campaign.windows_path(path)
            files[name] = {"resolved": name, "sha256": campaign.digest(path)}
        facts = self.root / "runtime-facts.json"
        facts.write_text(json.dumps({"scope": "isolated runtime substitution control",
                                     "proofs": native["expected"]}))
        native["receipts"] = {"runtime-facts.json": campaign.digest(facts)}
        return qualification, native

    def test_native_interpreter_substitution_rejects_original_producer_binding(self):
        config = self.copy_runtime()
        qualification, native = self.bind_runtime_fixture(config)
        executable = campaign.mapped_windows_path(self.python)
        executable.write_bytes(executable.read_bytes() + b"\0identity-control-overlay")
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaisesRegex(ValueError, "original proof maps"):
                qualification.run_native_identity_phase(self.root, native, "entry")
        process = self.owner.attempts("windows-entry")[-1]["process_tree"]
        self.assertTrue(process["drained"])
        self.assertEqual(process["status"], "completed")
        self.assertEqual(native["stages"], [])

    def test_native_config_mutation_during_reader_rejects_tool_seal(self):
        config = self.copy_runtime()
        self.reader_prefix("p=Path(" + repr(campaign.windows_path(config)) + ")\n"
                           "p.write_bytes(p.read_bytes()+b'\\n# changed during native reader\\n')")
        qualification, native = self.bind_runtime_fixture(config)
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                qualification.run_native_identity_phase(self.root, native, "entry")
        record = self.owner.attempts("windows-entry")[-1]
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["process_tree"]["drained"])
        self.assertEqual(record["process_tree"]["status"], "completed")

    def test_native_helper_mutation_during_reader_rejects_tool_seal(self):
        config = self.copy_runtime()
        helper = self.root / "scripts/build_dependencies.py"
        self.reader_prefix("p=Path(" + repr(campaign.windows_path(helper)) + ")\n"
                           "p.write_bytes(p.read_bytes()+b'\\n# changed during native reader\\n')")
        qualification, native = self.bind_runtime_fixture(config)
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                qualification.run_native_identity_phase(self.root, native, "entry")
        record = self.owner.attempts("windows-entry")[-1]
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["process_tree"]["drained"])
        self.assertEqual(record["process_tree"]["status"], "completed")

    def test_native_base_target_rejects_equal_byte_alias_result(self):
        launch = campaign.prepare_windows_identity_launch(self.root, self.python, self.request)
        other = self.root / "other-python.exe"
        shutil.copy2(campaign.mapped_windows_path(launch["base_python"]), other)
        helper = self.root / "scripts/build_dependencies.py"
        change = ("original_reader = read_windows_identities\n"
                  "def substituted(path, checksum):\n"
                  "    result = original_reader(path, checksum)\n"
                  "    result['runtime']['base_python']['resolved'] = " + repr(campaign.windows_path(other)) + "\n"
                  "    return result\nread_windows_identities = substituted\n\n")
        helper.write_text(helper.read_text().replace('if __name__ == "__main__":\n',
                                                     change + 'if __name__ == "__main__":\n'))
        qualification, native = self.native_state()
        with campaign.OutputLease(self.root, self.root / "build"):
            with self.assertRaisesRegex(ValueError, "interpreter binding changed"):
                qualification.run_native_identity_phase(self.root, native, "entry")

    def test_native_evidence_closure_rejects_missing_tampered_and_reordered_payloads(self):
        qualification, native = self.native_state()
        with campaign.OutputLease(self.root, self.root / "build"):
            qualification.run_native_identity_phase(self.root, native, "entry")
            qualification.run_native_identity_phase(self.root, native, "terminal")
            qualification.check_native_identity_evidence(self.root, native["stages"], native["expected"])
            for entries in (native["stages"][:1], list(reversed(native["stages"]))):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    qualification.check_native_identity_evidence(self.root, entries, native["expected"])
            for name in native["stages"][0]["payloads"]:
                path = Path(name)
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.subTest(payload=path.name), self.assertRaises(ValueError):
                        qualification.check_native_identity_evidence(self.root, native["stages"], native["expected"])
                finally:
                    path.write_bytes(original)

    def test_native_terminal_scan_rejects_same_byte_junction_retarget(self):
        self.assert_terminal_retarget_rejected(hardlink=False)

    def test_native_terminal_scan_rejects_same_file_hardlink_target_retarget(self):
        self.assert_terminal_retarget_rejected(hardlink=True)

    def assert_terminal_retarget_rejected(self, *, hardlink):
        first, second, alias = [self.root / n for n in ("first", "second", "alias")]
        for path in (first, second):
            path.mkdir()
        (first / "input.txt").write_text("original")
        if hardlink:
            os.link(first / "input.txt", second / "input.txt")
            self.assertEqual((first / "input.txt").stat().st_ino, (second / "input.txt").stat().st_ino)
        else:
            (second / "input.txt").write_text("original")
        self.input = first / "input.txt"
        marker = self.root / "retarget.txt"
        self.reader_prefix("target = " + repr(campaign.windows_path(second)) + " if Path(" +
            repr(campaign.windows_path(marker)) + ").exists() else " + repr(campaign.windows_path(first)) +
            "\nalias = " + repr(campaign.windows_path(alias)) +
            "\nif Path(alias).exists(): subprocess.run(['cmd.exe', '/c', 'rmdir', alias], check=True, capture_output=True)" +
            "\nsubprocess.run(['cmd.exe', '/c', 'mklink', '/J', alias, target], check=True, capture_output=True)")
        qualification, native = self.native_state(alias / "input.txt")
        with campaign.OutputLease(self.root, self.root / "build"):
            qualification.run_native_identity_phase(self.root, native, "entry")
            marker.touch()
            with self.assertRaisesRegex(ValueError, "original proof maps"):
                qualification.run_native_identity_phase(self.root, native, "terminal")
        self.assertEqual(campaign.digest(first / "input.txt"), campaign.digest(second / "input.txt"))

    def test_native_supervisor_early_exit_quarantines_unproved_lease(self):
        supervisor = self.root / "scripts/process_tree.py"
        supervisor.write_text(supervisor.read_text().replace('    main()\n', '    os._exit(0)\n'))
        lease = campaign.OutputLease(self.root, self.root / "build")
        with lease, self.assertRaisesRegex(RuntimeError, "completion could not be established"):
            self.execute()
        self.assertEqual(self.owner.attempts("windows-entry")[-1]["status"], "blocked")
        self.assertEqual(json.loads((lease.path / "owner.json").read_text())["status"], "blocked")

    def test_native_supervisor_death_with_live_job_quarantines_evidence(self):
        marker = self.root / "writer.txt"
        child = "import time;from pathlib import Path;p=Path(" + repr(campaign.windows_path(marker)) + ");\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        self.reader_prefix("subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "])\ntime.sleep(30)")
        supervisor = self.root / "scripts/process_tree.py"
        supervisor.write_text(supervisor.read_text().replace('\n        while True:\n',
            '\n        while True:\n            if Path(' + repr(campaign.windows_path(marker)) +
            ').is_file(): os._exit(23)\n', 1))
        lease = campaign.OutputLease(self.root, self.root / "build")
        with lease, self.assertRaisesRegex(RuntimeError, "native process evidence"):
            self.execute()
        self.assertEqual(self.owner.attempts("windows-entry")[-1]["status"], "blocked")
        self.assertEqual(json.loads((lease.path / "owner.json").read_text())["status"], "blocked")
        before = marker.read_bytes()
        time.sleep(.1)
        self.assertEqual(marker.read_bytes(), before)

    @unittest.skipUnless(os.environ.get("JUICER_WINDOWS_IDENTITY_PROOF"),
                         "requires explicit retained Windows preparation proof")
    def test_native_compiler_alias_spellings_match_exactly(self):
        import qualification

        repository = Path(__file__).resolve().parents[2]
        source = Path(os.environ["JUICER_WINDOWS_IDENTITY_PROOF"])
        retained = self.root / "historical-preparation.json"
        shutil.copy2(source, retained)
        proof = json.loads(retained.read_text())["scoped_preparation"]
        aliases = [(name, value["resolved"]) for name, value in proof["context"]["files"].items()
                   if name != value["resolved"]]
        self.assertTrue(any("~" in name for name, _ in aliases))
        self.assertTrue(any(name.casefold() == target.casefold() for name, target in aliases))
        fixture = self.root / "compiler-aliases.json"
        groups = {"files": {name: proof["context"]["files"][name] for name, _ in aliases},
                  "rust_configuration": {}}
        self.bind_runtime(groups["files"])
        fixture.write_text(json.dumps({"scope": "compiler alias identity control, not preparation admission",
            "source_receipt_sha256": campaign.digest(retained), "groups": groups}))
        expected = {campaign.digest(fixture): groups}
        native = {"campaign": self.owner, "python": self.python, "collection_id": "a" * 32,
                  "expected": expected, "receipts": {"compiler-aliases.json": campaign.digest(fixture)}, "stages": []}
        with campaign.OutputLease(repository, campaign.mapped_windows_path(proof["build"])), \
                campaign.OutputLease(self.root, self.root / "build"):
            qualification.run_native_identity_phase(self.root, native, "entry")
            qualification.run_native_identity_phase(self.root, native, "terminal")
            qualification.check_native_identity_evidence(self.root, native["stages"], expected)

    def test_native_parent_loss_closes_control_pipe_and_drains_job(self):
        marker = self.root / "writer.txt"
        child = "import time;from pathlib import Path;p=Path(" + repr(campaign.windows_path(marker)) + ");\nwhile True:p.write_text(str(time.monotonic()));time.sleep(.02)"
        self.reader_prefix("subprocess.Popen([sys.executable, '-B', '-c', " + repr(child) + "])\ntime.sleep(30)")
        worker = self.root / "worker.py"
        worker.write_text("import sys\nfrom pathlib import Path\nroot=Path(" + repr(str(self.root)) + ")\n"
            "sys.path.insert(0,str(root/'scripts'))\n"
            "from execution import Campaign,OutputLease,prepare_windows_identity_launch\n"
            "python=" + repr(self.python) + "\nrequest=root/'request.json'\n"
            "launch=prepare_windows_identity_launch(root,python,request)\n"
            "owner=Campaign(root,'linux-debug','owner-loss',['windows-entry'])\n"
            "with OutputLease(root,root/'build'):\n"
            " owner.run('windows-entry',launch['command'],inputs=['request.json','scripts/execution.py',"
            "'scripts/process_tree.py','scripts/build_dependencies.py'],tools=launch['tools'],timeout=10,"
            "windows_python=python,identity_request=request)\n")
        with (self.root / "worker.log").open("wb") as output:
            process = subprocess.Popen([sys.executable, "-B", str(worker)], stdout=output, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(marker.exists())
            finally:
                process.kill()
                process.wait(timeout=15)
        parent = self.root / "out/validation/linux-debug/owner-loss/attempts"
        deadline = time.monotonic() + 15
        native = None
        while time.monotonic() < deadline:
            paths = list(parent.glob("windows-entry-*/process.seal.json"))
            if paths:
                try:
                    native = json.loads(paths[0].with_name("process.json").read_text())
                except (OSError, ValueError):
                    pass
            if native and native.get("drained"):
                break
            time.sleep(.02)
        self.assertTrue(native and native.get("drained"))
        self.assertEqual(native["status"], "interrupted")
        before = marker.read_bytes()
        time.sleep(.1)
        self.assertEqual(marker.read_bytes(), before)
        self.assertTrue((parent.parent / "running.lock").is_file())
        self.assertTrue(list((self.root / "out/verification-locks").glob("*/owner.json")))


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.root = self.base / "root"
        self.root.mkdir()
        self.source = self.root / "source.txt"
        self.source.write_text("first", encoding="utf-8")
        self.owner = campaign.Campaign(
            self.root, "linux-debug", "report", ["one", "two"]
        )

    def prepare(self, stage, inputs=None, tools=()):
        python = (
            str(Path(sys.base_prefix) / "python.exe")
            if os.name == "nt"
            else sys.executable
        )
        return self.owner.run(
            stage,
            [python, "-c", "print('pass')"],
            inputs=["source.txt"] if inputs is None else inputs,
            tools=tools,
            timeout=10,
        )

    def change_after_read(self, target, change):
        original = campaign.digest
        changed = False

        def read(path):
            nonlocal changed
            value = original(path)
            if path == target and not changed:
                changed = True
                change()
            return value

        return patch.object(campaign, "digest", side_effect=read)

    def overwrite_source(self):
        stamp = self.source.stat()
        self.source.write_text("other", encoding="utf-8")
        os.utime(self.source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(self.source.stat().st_size, stamp.st_size)
        self.assertEqual(self.source.stat().st_mtime_ns, stamp.st_mtime_ns)

    def test_overlapping_stages_keep_different_expected_identities(self):
        self.prepare("one")
        self.overwrite_source()
        self.prepare("two")
        self.assertEqual(
            self.owner.report()["stages"], {"one": "inputs-changed", "two": "passed"}
        )

    def test_missing_unique_input_does_not_poison_other_stages(self):
        extra = self.root / "extra.txt"
        extra.write_text("extra", encoding="utf-8")
        self.prepare("one", ["source.txt", "extra.txt"])
        self.prepare("two")
        extra.unlink()
        self.assertEqual(
            self.owner.report()["stages"],
            {"one": "inputs-or-evidence-unavailable", "two": "passed"},
        )

    def test_reports_rehash_same_size_edits_with_restored_timestamp(self):
        self.prepare("one")
        self.prepare("two")
        self.assertTrue(self.owner.report()["all_recorded_gates_passed"])
        self.overwrite_source()
        self.assertEqual(
            self.owner.report()["stages"],
            {"one": "inputs-changed", "two": "inputs-changed"},
        )

    def test_source_change_during_report_cannot_pass(self):
        self.prepare("one")
        self.prepare("two")
        with self.change_after_read(self.source, self.overwrite_source):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "inputs-changed", "two": "inputs-changed"},
            )

    def test_output_change_during_report_cannot_pass(self):
        receipt = self.prepare("one")
        self.prepare("two")
        output = receipt.with_name("output.log")
        with self.change_after_read(
            output, lambda: output.write_text("changed", encoding="utf-8")
        ):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "evidence-changed", "two": "passed"},
            )

    def test_shared_file_keeps_its_input_and_output_roles(self):
        receipt = self.prepare("one")
        output = receipt.with_name("output.log")
        self.prepare("two", [output.relative_to(self.root).as_posix()])
        output.write_text("changed", encoding="utf-8")
        self.assertEqual(
            self.owner.report()["stages"],
            {"one": "evidence-changed", "two": "inputs-changed"},
        )

    def test_tool_change_during_report_cannot_pass(self):
        tool = self.base / "tool.txt"
        tool.write_text("first", encoding="utf-8")
        self.prepare("one", tools=[str(tool)])
        self.prepare("two")
        with self.change_after_read(
            tool, lambda: tool.write_text("other", encoding="utf-8")
        ):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "inputs-changed", "two": "passed"},
            )

    def test_missing_unique_tool_does_not_poison_other_stages(self):
        tool = self.base / "tool.txt"
        tool.write_text("tool", encoding="utf-8")
        self.prepare("one", tools=[str(tool)])
        self.prepare("two")
        tool.unlink()
        self.assertEqual(
            self.owner.report()["stages"],
            {"one": "inputs-or-evidence-unavailable", "two": "passed"},
        )

    def test_receipt_byte_change_during_report_cannot_pass(self):
        receipt = self.prepare("one")
        self.prepare("two")
        with self.change_after_read(
            receipt, lambda: receipt.write_bytes(receipt.read_bytes() + b"\n")
        ):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "evidence-changed", "two": "passed"},
            )

    def test_latest_attempt_change_during_report_cannot_pass(self):
        first = self.prepare("one")
        self.prepare("two")
        preserved = first.read_bytes()

        def add_attempt():
            command = self.owner.attempts("one")[-1]["identity"]["command"]
            self.owner.run(
                "one",
                command,
                inputs=["source.txt"],
                timeout=10,
                repeat_reason="latest-attempt membership witness",
            )

        with self.change_after_read(self.source, add_attempt):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "evidence-changed", "two": "passed"},
            )
        self.assertEqual(first.read_bytes(), preserved)

    def test_declared_membership_change_during_report_cannot_pass(self):
        receipt = self.prepare("one")
        self.prepare("two")
        extra = self.root / "extra.txt"
        extra.write_text("extra", encoding="utf-8")
        original = receipt.read_bytes()
        for addition in (True, False):
            with self.subTest(addition=addition):
                receipt.write_bytes(original)

                def change_membership():
                    record = json.loads(receipt.read_text())
                    if addition:
                        record["identity"]["inputs"]["extra.txt"] = campaign.digest(
                            extra
                        )
                    else:
                        del record["identity"]["inputs"]["source.txt"]
                    campaign.write_json(receipt, record)

                with self.change_after_read(self.source, change_membership):
                    self.assertEqual(
                        self.owner.report()["stages"],
                        {"one": "evidence-changed", "two": "passed"},
                    )

    def test_same_byte_input_link_retargeting_during_report_cannot_pass(self):
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        for directory in (first, second):
            (directory / "source.txt").write_text("same", encoding="utf-8")
        alias = self.root / "alias"
        link_directory(alias, first)
        self.prepare("one", ["alias/source.txt"])
        self.prepare("two")

        def retarget():
            unlink_directory(alias)
            link_directory(alias, second)

        with self.change_after_read(alias / "source.txt", retarget):
            self.assertEqual(
                self.owner.report()["stages"],
                {"one": "inputs-changed", "two": "passed"},
            )

    def test_input_link_cannot_escape_root_between_reports(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "source.txt").write_text("first", encoding="utf-8")
        alias = self.root / "alias"
        link_directory(alias, self.root)
        self.prepare("one", ["alias/source.txt"])
        self.prepare("two")
        self.assertTrue(self.owner.report()["all_recorded_gates_passed"])
        unlink_directory(alias)
        link_directory(alias, outside)
        self.assertEqual(
            self.owner.report()["stages"],
            {"one": "inputs-or-evidence-unavailable", "two": "passed"},
        )

    def test_a_lock_seen_at_either_boundary_blocks_qualification(self):
        self.prepare("one")
        self.prepare("two")
        lock = self.owner.directory / "running.lock"
        for present_at_entry in (False, True):
            with self.subTest(present_at_entry=present_at_entry):
                if present_at_entry:
                    lock.write_text("writer", encoding="utf-8")
                    change = lock.unlink
                else:

                    def change():
                        return lock.write_text("writer", encoding="utf-8")

                with self.change_after_read(self.source, change):
                    report = self.owner.report()
                self.assertTrue(report["locked"])
                self.assertFalse(report["all_recorded_gates_passed"])
                lock.unlink(missing_ok=True)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.root = self.base / "root"
        self.root.mkdir()
        (self.root / "a.txt").write_text("first", encoding="utf-8")
        (self.root / "b.txt").write_text("second", encoding="utf-8")

    def test_tool_snapshot_hashes_each_resolved_file_once_per_call(self):
        alias = self.base / "alias"
        link_directory(alias, self.root)
        self.addCleanup(unlink_directory, alias)
        first, second = self.root / "a.txt", self.root / "b.txt"
        expected = {str(path): campaign.digest(path) for path in (first, second)}
        names = [str(first), str(alias / "a.txt"), str(second)]
        for allow_missing in (False, True):
            with self.subTest(allow_missing=allow_missing), patch.object(
                campaign, "digest", wraps=campaign.digest
            ) as reads:
                self.assertEqual(
                    campaign.tool_snapshot(names, allow_missing=allow_missing), expected
                )
                self.assertCountEqual(
                    [call.args[0] for call in reads.call_args_list], [first, second]
                )

    def test_tool_snapshot_rehashes_edits_and_observes_removal_across_calls(self):
        tool = self.root / "a.txt"
        names = [str(tool), str(self.root) + "/./a.txt"]
        before = campaign.tool_snapshot(names)
        stamp = tool.stat()
        tool.write_text("other", encoding="utf-8")
        os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(tool.stat().st_size, stamp.st_size)
        self.assertEqual(tool.stat().st_mtime_ns, stamp.st_mtime_ns)
        self.assertNotEqual(campaign.tool_snapshot(names), before)
        tool.unlink()
        self.assertEqual(campaign.tool_snapshot(names, allow_missing=True), {str(tool): None})
        with self.assertRaises(FileNotFoundError):
            campaign.tool_snapshot(names)
        tool.mkdir()
        with self.assertRaisesRegex(ValueError, "not a file"):
            campaign.tool_snapshot(names, allow_missing=True)

    def test_tool_snapshot_resolves_aliases_again_after_retargeting(self):
        other = self.base / "other"
        other.mkdir()
        (other / "a.txt").write_text("first", encoding="utf-8")
        alias = self.base / "alias"
        link_directory(alias, self.root)
        self.addCleanup(unlink_directory, alias)
        first = campaign.tool_snapshot([str(alias / "a.txt")])
        unlink_directory(alias)
        link_directory(alias, other)
        second = campaign.tool_snapshot([str(alias / "a.txt")])
        self.assertEqual(list(first.values()), list(second.values()))
        self.assertEqual(list(first), [str(self.root / "a.txt")])
        self.assertEqual(list(second), [str(other / "a.txt")])

    def test_tool_snapshot_keeps_distinct_hardlink_paths(self):
        first = self.root / "a.txt"
        second = self.root / "hardlink.txt"
        os.link(first, second)
        expected = {str(path): campaign.digest(path) for path in (first, second)}
        with patch.object(campaign, "digest", wraps=campaign.digest) as reads:
            self.assertEqual(campaign.tool_snapshot(list(expected)), expected)
            self.assertCountEqual(
                [call.args[0] for call in reads.call_args_list], [first, second]
            )

    def test_declared_keys_stay_sorted_unique_and_unmodified(self):
        absolute = str(self.root / "a.txt")
        result = campaign.snapshot(self.root, ["b.txt", "a.txt", "b.txt", absolute])
        self.assertEqual(list(result), sorted(["a.txt", "b.txt", absolute]))
        self.assertEqual(result["a.txt"], result[absolute])
        self.assertEqual(result["b.txt"], campaign.digest(self.root / "b.txt"))
        self.assertEqual(campaign.snapshot(self.root, []), {})

    def test_same_size_edit_with_restored_timestamp_is_rehashed(self):
        source = self.root / "a.txt"
        before = campaign.snapshot(self.root, ["a.txt"])
        stamp = source.stat()
        source.write_text("other", encoding="utf-8")
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(source.stat().st_size, stamp.st_size)
        self.assertEqual(source.stat().st_mtime_ns, stamp.st_mtime_ns)
        self.assertNotEqual(campaign.snapshot(self.root, ["a.txt"]), before)

    def test_deleted_input_does_not_reuse_previous_hash(self):
        campaign.snapshot(self.root, ["a.txt"])
        (self.root / "a.txt").unlink()
        with self.assertRaisesRegex(ValueError, "not a file"):
            campaign.snapshot(self.root, ["a.txt"])

    def test_parent_traversal_cannot_escape_root(self):
        (self.base / "outside.txt").write_text("outside", encoding="utf-8")
        with self.assertRaises(ValueError):
            campaign.snapshot(self.root, ["../outside.txt"])

    def test_input_link_is_resolved_again_after_retargeting(self):
        inside = self.root / "inside"
        outside = self.base / "outside"
        inside.mkdir()
        outside.mkdir()
        for directory in (inside, outside):
            (directory / "source.txt").write_text("same", encoding="utf-8")
        alias = self.root / "alias"
        link_directory(alias, inside)
        self.assertEqual(
            campaign.snapshot(self.root, ["alias/source.txt"]),
            {"alias/source.txt": campaign.digest(inside / "source.txt")},
        )
        unlink_directory(alias)
        link_directory(alias, outside)
        with self.assertRaises(ValueError):
            campaign.snapshot(self.root, ["alias/source.txt"])

    def test_root_link_is_resolved_fresh_for_each_call(self):
        other = self.base / "other"
        other.mkdir()
        (other / "a.txt").write_text("other", encoding="utf-8")
        alias = self.base / "root-alias"
        link_directory(alias, self.root)
        first = campaign.snapshot(alias, ["a.txt"])
        unlink_directory(alias)
        link_directory(alias, other)
        second = campaign.snapshot(alias, ["a.txt"])
        self.assertNotEqual(first, second)
        self.assertEqual(second, {"a.txt": campaign.digest(other / "a.txt")})

    def test_root_retargeting_cannot_mix_namespaces_in_one_snapshot(self):
        other = self.base / "other"
        other.mkdir()
        (other / "b.txt").write_text("outside", encoding="utf-8")
        alias = self.base / "root-alias"
        link_directory(alias, self.root)
        original = campaign.digest

        def retarget(source):
            value = original(source)
            if source.name == "a.txt":
                unlink_directory(alias)
                link_directory(alias, other)
            return value

        with (
            patch.object(campaign, "digest", side_effect=retarget),
            self.assertRaises(ValueError),
        ):
            campaign.snapshot(alias, ["b.txt", "a.txt"])


class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        self.job = process_tree.WindowsJob.__new__(process_tree.WindowsJob)
        self.job.job = 9
        self.job.api = Mock()
        self.job.api.OpenProcess.return_value = 99
        self.process = Mock(pid=42, _handle=7)
        self.process.poll.return_value = None
        self.popen = self.enterContext(
            patch.object(process_tree.subprocess, "Popen", return_value=self.process)
        )

    def test_readable_pid_is_assigned_before_one_gate_release(self):
        for error in (FileNotFoundError("not published"), PermissionError("not readable")):
            with self.subTest(error=type(error).__name__):
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
                sleep.assert_called_once_with(0.01)
                self.popen.assert_called_once()
                self.job.api.AssignProcessToJobObject.assert_any_call(9, 7)
                self.job.api.AssignProcessToJobObject.assert_any_call(9, 99)
                self.assertEqual(self.job.api.AssignProcessToJobObject.call_count, 2)
                self.job.api.OpenProcess.assert_called_once_with(0x0101, False, 43)
                self.process.stdin.write.assert_called_once_with(b"g")

    def test_late_pid_or_job_assignment_cannot_release_gate(self):
        for times in ([0.0, 0.0, 6.0], [0.0, 0.0, 0.0, 6.0]):
            with self.subTest(times=times):
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


if __name__ == "__main__":
    unittest.main()
