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
