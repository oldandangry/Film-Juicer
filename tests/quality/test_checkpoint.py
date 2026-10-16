"""Checkpoint ownership, cross-caller reuse and exact CTest membership controls."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import qualification
from execution import Campaign, OutputLease, digest, tool_snapshot

ROOT = Path(__file__).resolve().parents[2]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = ROOT / "out/validation" / preset / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "out/build" / preset
        self.build.mkdir(parents=True)
        self.state_path = self.build / "verification-checkpoint.json"
        self.state = {}
        self.preset = preset
        self.environment = dict(os.environ)
        self.python = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
        self.source = self.root / "input.txt"
        self.source.write_text("first")
        self.other = self.root / "other.txt"
        self.other.write_text("unrelated")
        self.lease = OutputLease(self.root, self.build)
        self.lease.__enter__()
        def finish():
            self.lease.__exit__(None, None, None)
            result = self._outcome.result
            if any(test is self for test, _ in [*result.errors, *result.failures]):
                print("Retained checkpoint fixture:", self.root, file=sys.stderr)
            else:
                self.temporary.cleanup()
        self.addCleanup(finish)

    def campaign(self, name):
        return Campaign(self.root, self.preset, name, ["boundary", "tooling", "prepare", "corpus", "production"])

    def run_stage(self, name, *, campaign="first", inputs=None, command=None, tools=None, **kwargs):
        return qualification.run_stage(self.campaign(campaign), self.state_path, self.state, name,
            command or [self.python, "-c", "print('checked')"], inputs=inputs or ["input.txt"],
            tools=tools or [self.python], environment=self.environment, timeout=10, **kwargs)

    def test_new_campaign_reuses_original_success_without_execution(self):
        first = self.run_stage("boundary")
        result = self.run_stage("boundary", campaign="renamed")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(result["sha256"], digest(Path(first["receipt"])))
        self.assertFalse((self.campaign("renamed").directory / "attempts").exists())

    def test_known_output_destinations_do_not_change_gate_identity(self):
        first = self.run_stage("boundary", semantic_command=[self.python, "same", "<junit>"])
        result = self.run_stage("boundary", campaign="renamed", command=[self.python, "unused-output-path"],
                                semantic_command=[self.python, "same", "<junit>"])
        self.assertEqual(result["reused_from"], first["receipt"])

    def test_unrelated_repair_preserves_success_and_pending_failure(self):
        first = self.run_stage("boundary")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", inputs=["other.txt"], command=[self.python, "-c", "raise SystemExit(1)"])
        self.other.write_text("repair")
        result = self.run_stage("boundary", campaign="new-caller")
        self.assertEqual(result["reused_from"], first["receipt"])
        self.assertEqual(json.loads(self.state_path.read_text())["tooling"]["status"], "failed")
        self.run_stage("prepare", campaign="preparation", inputs=["other.txt"])
        self.assertEqual(self.state["tooling"]["status"], "failed")

    def test_dependency_content_with_restored_timestamp_invalidates_reuse(self):
        first = self.run_stage("boundary")
        stamp = self.source.stat()
        self.source.write_text("other")
        os.utime(self.source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        result = self.run_stage("boundary", campaign="changed")
        self.assertNotIn("reused_from", result)
        self.assertNotEqual(result["receipt"], first["receipt"])

    def test_failed_gate_cannot_be_reset_by_changing_caller(self):
        command = [self.python, "-c", "raise SystemExit(1)"]
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", command=command)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("boundary", campaign="renamed", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_reuse_validation_failure_stays_pending_across_other_gates(self):
        first = self.run_stage("prepare")
        original = Path(first["receipt"]).read_bytes()
        def reject(path):
            raise ValueError("required output missing")
        with self.assertRaisesRegex(ValueError, "required output"):
            self.run_stage("prepare", campaign="missing-output", validate=reject)
        self.run_stage("tooling", campaign="different-caller", inputs=["other.txt"])
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(Path(first["receipt"]).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            self.run_stage("prepare", campaign="another-caller")

    def test_input_mutation_during_execution_cannot_publish_success(self):
        command = [self.python, "-c", "from pathlib import Path;Path('input.txt').write_text('mutated')"]
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_stage("boundary", command=command)
        self.assertEqual(self.state["boundary"]["status"], "failed")

    def test_old_pass_cannot_bypass_new_failure_during_collection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("boundary", campaign="failure", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending"):
            qualification.collect_qualification(self.root, [path])

    def test_other_pending_gate_cannot_be_hidden_by_collecting_old_selection(self):
        first = self.run_stage("boundary")
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["boundary"], "required": ["boundary"], "stages": {"boundary": first}}
        path = self.root / "old-checkpoint.json"
        path.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_stage("tooling", command=[self.python, "-c", "raise SystemExit(1)"])
        with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
            qualification.collect_qualification(self.root, [path])

    def test_preparation_binds_its_owned_post_configuration_without_a_second_run(self):
        config = self.build / "CMakeCache.txt"
        config.write_text("before")
        inputs = ["input.txt", str(config.relative_to(self.root))]
        def prepare():
            path = self.campaign("first").run("prepare", [self.python, "-c", "print('configured')"], inputs=["input.txt"], timeout=10)
            config.write_text("after")
            return path
        first = self.run_stage("prepare", inputs=inputs, action=prepare)
        reused = self.run_stage("prepare", campaign="renamed", inputs=inputs, action=prepare)
        self.assertEqual(first["receipt"], reused["reused_from"])

    def test_boundary_inputs_bind_native_membership_and_foreign_fixture(self):
        for name in ("src/RenderRecipe.cpp", "src/JuicerState.cpp", "tests/ffi/color_preparation_test.cpp"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("source")
        (self.build / "CMakeCache.txt").write_text("configured")
        (self.build / "CTestTestfile.cmake").write_text("")
        first = qualification.checkpoint_inputs(self.root, self.build, "boundary")
        self.assertIn("src/RenderRecipe.cpp", first)
        self.assertIn("tests/ffi/color_preparation_test.cpp", first)
        added = self.root / "src/SpectralProcessing.h"
        added.write_text("new member")
        self.assertIn("src/SpectralProcessing.h", qualification.checkpoint_inputs(self.root, self.build, "boundary"))
        self.assertNotIn("src/SpectralProcessing.h", first)

    def test_cargo_configuration_content_and_membership_are_tool_inputs(self):
        (self.build / "CMakeCache.txt").write_text("")
        home = self.root / "cargo-home"
        home.mkdir()
        config = home / "config.toml"
        environment = {"PATH": "", "CARGO_HOME": str(home), "RUSTUP_HOME": str(self.root / "rustup")}
        before = qualification.checkpoint_tools(self.build, environment)
        absent = self.run_stage("boundary", tools=before)
        config.write_text('[build]\n')
        after = qualification.checkpoint_tools(self.build, environment)
        self.assertIn(str(config), before)
        self.assertIn(str(config), after)
        first = self.run_stage("boundary", tools=after)
        self.assertNotEqual(absent["receipt"], first["receipt"])
        config.write_text('[build]\nrustflags=["--cfg", "changed"]\n')
        changed = self.run_stage("boundary", campaign="config-changed", tools=qualification.checkpoint_tools(self.build, environment))
        self.assertNotEqual(first["receipt"], changed["receipt"])
        self.assertNotIn("reused_from", changed)

    def native_project(self):
        (self.root / ".gitignore").write_text("out/\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True, timeout=10)
        (self.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint NONE)\ninclude(CTest)\n'
            'add_custom_target(material ALL COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
            '"${CMAKE_SOURCE_DIR}/input.txt" "${CMAKE_BINARY_DIR}/material.txt" '
            'DEPENDS "${CMAKE_SOURCE_DIR}/input.txt" BYPRODUCTS "${CMAKE_BINARY_DIR}/material.txt")\n'
            'if(BUILD_TESTING)\nadd_test(NAME boundary COMMAND "${CMAKE_COMMAND}" -E true)\n'
            'set_tests_properties(boundary PROPERTIES LABELS "host;boundary")\nendif()\n')
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [{
            "name": self.preset, "generator": "Ninja", "binaryDir": str(self.build)}]}))
        subprocess.run(["cmake", "--preset", self.preset, "-DBUILD_TESTING=OFF"], cwd=self.root,
                       check=True, capture_output=True, timeout=30)

    def test_real_native_dispatcher_executes_and_reuses_exact_ctest_membership(self):
        self.native_project()
        subprocess.run(["cmake", "--preset", self.preset, "-DBUILD_TESTING=ON"], cwd=self.root,
                       check=True, capture_output=True, timeout=30)
        args = SimpleNamespace(reason="native caller regression", selection=["boundary"], preset=self.preset,
                               runtime="host", campaign="caller-one", repeat_reason="", base=None)
        import verify
        argv = ["verify.py", "qualification", "--preset", self.preset, "--reason", args.reason,
                "--selection", "boundary", "--runtime", "host", "--campaign", args.campaign]
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", argv):
            self.assertEqual(verify.main(), 0)
        first = json.loads((self.root / "out/validation" / self.preset / "caller-one/qualification.json").read_text())
        args.campaign = "caller-two"
        second = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual(first["membership"], {"boundary": ["boundary"]})
        self.assertEqual(second["status"], "passed")
        self.assertEqual(second["stages"]["boundary"]["reused_from"], first["stages"]["boundary"]["receipt"])
        self.assertFalse((self.root / "out/validation" / self.preset / "caller-two/attempts").exists())

    def mixed_project(self):
        self.native_project()
        probe = self.root / "probe.py"
        probe.write_text(
            "import json,sys,time\nfrom pathlib import Path\n"
            "root=Path(__file__).parent\nrole=sys.argv[1]\n"
            f"state=json.loads((root/'out/build/{self.preset}/verification-checkpoint.json').read_text())\n"
            "assert state['boundary']['status']=='pending'\n"
            "leases=list((root/'out/verification-locks').glob('*/owner.json'))\n"
            "assert any(p.parent.name.startswith('build-') for p in leases)\n"
            "assert (root/'out/verification-locks/gpu/owner.json').is_file()==(role=='gpu')\n"
            "with (root/'out/order.txt').open('a') as f:f.write(role+'\\n')\n"
            "mode=(root/'failure.txt').read_text() if (root/'failure.txt').exists() else ''\n"
            "if mode=='timeout' and role=='gpu':time.sleep(30)\n"
            "raise SystemExit(1 if mode==role else 0)\n")
        cmake = self.root / "CMakeLists.txt"
        text = cmake.read_text()
        text += '\n'
        for role in ("host", "gpu"):
            text += (f'add_test(NAME {role} COMMAND "{Path(self.python).as_posix()}" "{probe.as_posix()}" {role})\n'
                     f'set_tests_properties({role} PROPERTIES LABELS "{role};boundary")\n')
        cmake.write_text(text)
        subprocess.run(["cmake", "--preset", self.preset, "-DBUILD_TESTING=ON"], cwd=self.root,
                       check=True, capture_output=True, timeout=30)
        return SimpleNamespace(reason="mixed runtime regression", selection=["boundary"], preset=self.preset,
                               runtime="all", campaign="mixed", repeat_reason="", base=None)

    def test_mixed_ctest_owns_partitions_and_reuses_without_gpu_lease(self):
        args = self.mixed_project()
        actual_lease = qualification.OutputLease
        def bounded_lease(root, build=None, **kwargs):
            if build is None:
                self.assertGreater(kwargs["timeout"], 30, "GPU wait must use the remaining stage budget")
                self.assertLessEqual(kwargs["timeout"], 1800)
                self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "pending")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "OutputLease", side_effect=bounded_lease):
            first = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        stage = first["stages"]["boundary"]
        self.assertEqual(len(stage["additional_receipts"]), 1)
        self.assertEqual(first["runtime_membership"]["boundary"], {"host": ["boundary", "host"], "gpu": ["gpu"]})
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
        args.campaign = "reused-mixed"
        actual_lease = qualification.OutputLease
        def build_lease(root, build=None, **kwargs):
            self.assertIsNotNone(build, "unchanged reuse attempted to acquire the GPU lease")
            return actual_lease(root, build, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=AssertionError("executed reused stage")), \
                patch.object(qualification, "OutputLease", side_effect=build_lease):
            reused = json.loads(qualification.qualify(self.root, args).read_text())
        self.assertEqual(reused["stages"]["boundary"]["reused_from"], stage["receipt"])
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host", "gpu"])
        secondary = Path(next(iter(stage["additional_receipts"])))
        original = secondary.read_bytes()
        secondary.write_bytes(original + b" ")
        args.campaign = "changed-secondary"
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            qualification.qualify(self.root, args)
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        secondary.write_bytes(original)
        args.campaign = "cannot-hide-failure"
        with self.assertRaisesRegex(ValueError, "pending checkpoint"):
            qualification.qualify(self.root, args)

    def test_mixed_ctest_host_failure_stops_gpu_and_gpu_failure_stays_pending(self):
        args = self.mixed_project()
        for role in ("host", "gpu"):
            (self.root / "failure.txt").write_text(role)
            args.campaign = "fail-" + role
            args.repeat_reason = "exercise distinct partition failure"
            (self.root / "out/order.txt").unlink(missing_ok=True)
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            self.assertEqual((self.root / "out/order.txt").read_text().splitlines(),
                             ["host"] if role == "host" else ["host", "gpu"])
            self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
            self.assertFalse((OutputLease(self.root).path / "owner.json").exists())
            args.campaign = "new-caller-" + role
            args.repeat_reason = ""
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)

    def test_zero_exit_host_with_unrun_case_never_schedules_gpu(self):
        args = self.mixed_project()
        actual = Campaign.run
        def incomplete(campaign, stage, command, **kwargs):
            path = actual(campaign, stage, command, **kwargs)
            if "-LE" in command:
                import xml.etree.ElementTree as ET
                report = Path(command[command.index("--output-junit") + 1])
                document = ET.parse(report)
                next(document.getroot().iter("testcase")).set("status", "notrun")
                document.write(report)
            return path
        with patch.object(Campaign, "run", autospec=True, side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, "did not execute"):
                qualification.qualify(self.root, args)
        self.assertEqual((self.root / "out/order.txt").read_text().splitlines(), ["host"])
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        self.assertEqual(len(Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")), 1)

    def test_mixed_ctest_timeout_drains_gpu_and_keeps_whole_stage_failed(self):
        args = self.mixed_project()
        (self.root / "failure.txt").write_text("timeout")
        actual = qualification.run_ctest_parts
        def bounded(*args, **kwargs):
            kwargs["timeout"] = 15
            return actual(*args, **kwargs)
        with patch.object(qualification, "run_ctest_parts", side_effect=bounded):
            with self.assertRaises(subprocess.TimeoutExpired):
                qualification.qualify(self.root, args)
        self.assertEqual(json.loads(self.state_path.read_text())["boundary"]["status"], "failed")
        raw = Campaign(self.root, self.preset, "mixed", ["boundary"]).attempts("boundary")
        self.assertEqual([r["status"] for r in raw], ["passed", "failed"])
        self.assertTrue(all(r["process_tree"]["drained"] for r in raw))
        self.assertFalse((OutputLease(self.root).path / "owner.json").exists())

    def test_collection_binds_all_partition_receipts_and_aggregate_membership(self):
        args = self.mixed_project()
        path = qualification.qualify(self.root, args)
        record = json.loads(path.read_text())
        # The valid partial checkpoint reaches only the final full-matrix gate.
        with self.assertRaisesRegex(ValueError, "membership is incomplete"):
            qualification.collect_qualification(self.root, [path])
        for change in ("secondary", "overlap", "missing", "swapped"):
            candidate = json.loads(json.dumps(record))
            if change == "secondary":
                candidate["stages"]["boundary"]["additional_receipts"] = {}
            elif change == "overlap":
                candidate["runtime_membership"]["boundary"]["host"].append("gpu")
            elif change == "missing":
                del candidate["runtime_membership"]["boundary"]["gpu"]
            else:
                parts = candidate["runtime_membership"]["boundary"]
                parts["host"], parts["gpu"] = parts["gpu"], parts["host"]
            path.write_text(json.dumps(candidate))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "ownership|partition"):
                qualification.collect_qualification(self.root, [path])

    def test_real_off_to_on_preparation_owns_generated_configuration(self):
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "product")
        source_inputs = [p for p in inputs if not p.startswith("out/build/")]
        def prepare():
            return qualification.prepare_build(self.campaign("first"), "prepare", source=self.root,
                build=self.build, inputs=source_inputs, timeout=30, targets=["all"],
                configure_args=["-DBUILD_TESTING=ON"], checkpoint=True)[1]
        first = self.run_stage("prepare", inputs=inputs, action=prepare,
            refresh_inputs=lambda: qualification.checkpoint_inputs(self.root, self.build, "product"),
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("first"), self.build, p))
        self.assertIn("BUILD_TESTING:BOOL=ON", (self.build / "CMakeCache.txt").read_text())
        self.assertTrue((self.build / "CTestTestfile.cmake").is_file())
        refreshed = qualification.checkpoint_inputs(self.root, self.build, "product")
        second = self.run_stage("prepare", campaign="next-caller", inputs=refreshed, action=prepare,
            validate=lambda p: qualification.check_scoped_preparation(self.campaign("next-caller"), self.build, p))
        self.assertEqual(second["reused_from"], first["receipt"])

    def test_production_inputs_do_not_require_test_registration(self):
        self.native_project()
        self.assertFalse((self.build / "CTestTestfile.cmake").exists())
        inputs = qualification.checkpoint_inputs(self.root, self.build, "production")
        self.assertIn("CMakeLists.txt", inputs)
        self.assertIn((self.build / "CMakeCache.txt").relative_to(self.root).as_posix(), inputs)

    @unittest.skipUnless(os.name == "nt", "Windows production PDB")
    def test_production_pdb_changes_and_missing_repair_cannot_reuse_old_pass(self):
        self.native_project()
        self.preset = "windows-clang-release"
        self.build = self.root / "out/build" / self.preset
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.root.as_posix()}\nBUILD_TESTING:BOOL=OFF\n")
        module = self.build / "juicer.ofx"
        module.write_text("module")
        pdb = self.build / "juicer.pdb"
        pdb.write_text("first")
        driver = self.root / "tests/ffi/production_contract.py"
        driver.parent.mkdir(parents=True)
        driver.write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser();p.add_argument('--preset');p.add_argument('--artifacts')\n"
            "a=p.parse_args();root=Path.cwd();out=Path(a.artifacts);out.mkdir(parents=True,exist_ok=True)\n"
            "count=root/'out/production-count.txt'\n"
            "count.write_text(str(int(count.read_text())+1) if count.exists() else '1')\n"
            "assert (root/'out/build'/a.preset/'juicer.pdb').is_file()\n"
            "(out/'production.json').write_text('{}')\n")
        args = SimpleNamespace(reason="production PDB identity regression", selection=["production"],
                               preset=self.preset, runtime="host", campaign="first", repeat_reason="", base=None)
        count = self.root / "out/production-count.txt"
        with patch("build_dependencies.NinjaGraph", return_value=SimpleNamespace(outputs={str(module)})), \
                patch.object(qualification, "checkpoint_tools", return_value=[self.python]):
            first = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertEqual(first["identity"]["inputs"][pdb.relative_to(self.root).as_posix()], digest(pdb))
            args.campaign = "unchanged"
            reused = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertEqual(reused["reused_from"], first["receipt"])
            self.assertEqual(count.read_text(), "1")
            stamp = pdb.stat()
            pdb.write_text("other")
            os.utime(pdb, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            args.campaign = "pdb-changed"
            changed = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertNotIn("reused_from", changed)
            self.assertEqual(count.read_text(), "2")
            pdb.unlink()
            args.campaign = "pdb-missing"
            with self.assertRaises(subprocess.CalledProcessError):
                qualification.qualify(self.root, args)
            state = self.build / "verification-checkpoint.json"
            self.assertEqual(json.loads(state.read_text())["production"]["status"], "failed")
            self.assertEqual(count.read_text(), "3")
            args.campaign = "missing-still-pending"
            with self.assertRaisesRegex(ValueError, "pending checkpoint"):
                qualification.qualify(self.root, args)
            self.assertEqual(count.read_text(), "3")
            pdb.write_text("other")
            args.campaign = "pdb-repaired"
            repaired = json.loads(qualification.qualify(self.root, args).read_text())["stages"]["production"]
            self.assertNotIn("reused_from", repaired)
            self.assertEqual(count.read_text(), "4")

    @unittest.skipUnless(os.name == "nt", "Windows production symbol tools")
    def test_production_symbol_tool_changes_invalidate_reuse(self):
        (self.build / "CMakeCache.txt").write_text("")
        llvm = self.root / "llvm"
        llvm.mkdir()
        for name in ("llvm-nm.exe", "clang-cl.exe", "llvm-readobj.exe", "llvm-pdbutil.exe"):
            (llvm / name).write_text("first")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home"),
                       "RUSTUP_HOME": str(self.root / "rustup")}
        def local_path(value, *parts):
            if value == "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin":
                return llvm
            return Path(value, *parts)
        with patch.object(qualification, "Path", side_effect=local_path) as factory:
            factory.home.return_value = Path.home()
            tools = qualification.checkpoint_tools(self.build, environment, "production")
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            self.assertIn(str(llvm / name), tools)
        previous = self.run_stage("production", tools=tools)
        for name in ("llvm-readobj.exe", "llvm-pdbutil.exe"):
            tool = llvm / name
            stamp = tool.stat()
            tool.write_text("other")
            os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            changed = self.run_stage("production", campaign=name.removesuffix(".exe"), tools=tools)
            self.assertNotIn("reused_from", changed)
            self.assertNotEqual(changed["receipt"], previous["receipt"])
            previous = changed

    def retained_record(self, stage):
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "retained",
                  "preset": self.preset, "selection": ["corpus"], "required": ["corpus"], "stages": {"corpus": stage}}
        path = self.root / "checkpoint.json"
        path.write_text(json.dumps(record))
        return path

    def test_collection_rechecks_authored_data_and_input_membership(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = ["Resources/data.txt"]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "membership is incomplete"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source/data inputs changed"):
                qualification.collect_qualification(self.root, [path])
            data.write_text("authored")
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs + ["Resources/new.txt"]):
            with self.assertRaisesRegex(ValueError, "source/data membership changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rejects_previously_absent_tool_configuration(self):
        config = self.root / "cargo-config.toml"
        stage = self.run_stage("corpus", tools=[self.python, str(config)])
        path = self.retained_record(stage)
        config.write_text('[build]\n')
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_binds_actual_final_production_artifact(self):
        artifact = self.build / "plugin.a"
        artifact.write_bytes(b"inspected final artifact")
        inputs = ["input.txt", artifact.relative_to(self.root).as_posix()]
        stage = self.run_stage("production", inputs=inputs)
        record = {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "production",
                  "preset": self.preset, "selection": ["production"], "required": ["production"],
                  "stages": {"production": stage}}
        path = self.root / "production-checkpoint.json"
        path.write_text(json.dumps(record))
        artifact.write_bytes(b"different final artifact")
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs):
            with self.assertRaisesRegex(ValueError, "final inspected production artifacts changed"):
                qualification.collect_qualification(self.root, [path])

    def test_probe_artifact_ancestor_configuration_invalidates_checkpoint(self):
        (self.build / "CMakeCache.txt").write_text("")
        environment = {"PATH": "", "CARGO_HOME": str(self.root / "cargo-home")}
        tools = qualification.checkpoint_tools(self.build, environment)
        config = self.root / "out/.cargo/config.toml"
        first = self.run_stage("boundary", tools=tools)
        self.assertIn(str(config), tools)
        config.parent.mkdir(parents=True)
        config.write_text('[build]\n')
        changed = self.run_stage("boundary", campaign="artifact-config", tools=tools)
        self.assertNotEqual(first["receipt"], changed["receipt"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("checkpoint_quality", ROOT / "scripts/check-quality.py")
        quality = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quality
        spec.loader.exec_module(quality)
        with self.assertRaisesRegex(quality.QualityError, "different Cargo configuration"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")
        config.unlink()
        config.mkdir()
        with self.assertRaisesRegex(quality.QualityError, "not a file"):
            quality.check_probe_configuration(self.root, self.root / "out/probe")

    def test_tool_snapshot_distinguishes_absent_files_from_invalid_directories(self):
        config = self.root / "config.toml"
        with self.assertRaises(FileNotFoundError):
            tool_snapshot([str(config)])
        self.assertEqual(tool_snapshot([str(config)], allow_missing=True), {str(config): None})
        config.mkdir()
        for allow_missing in (False, True):
            with self.subTest(allow_missing=allow_missing), self.assertRaisesRegex(ValueError, "not a file"):
                tool_snapshot([str(config)], allow_missing=allow_missing)


class MembershipTests(unittest.TestCase):
    @staticmethod
    def inventory(*rows):
        return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]} for name, labels in rows]}

    def test_relevance_is_disjoint_and_runtime_is_orthogonal(self):
        inventory = self.inventory(("behavior", ["host", "product"]), ("boundary", ["host", "boundary"]),
                                   ("runner", ["host", "tooling"]), ("data", ["host", "corpus"]),
                                   ("device", ["gpu", "product"]))
        result = qualification.membership(inventory, ["product", "boundary", "tooling", "corpus"], "all")
        self.assertEqual(sorted(n for names in result.values() for n in names), ["behavior", "boundary", "data", "device", "runner"])
        self.assertNotIn("device", qualification.membership(inventory, ["product"], "host")["product"])

    def test_missing_duplicate_unclassified_and_empty_membership_fail(self):
        for inventory in (self.inventory(), self.inventory(("a", ["host"])),
                          self.inventory(("a", ["host", "product", "tooling"])),
                          self.inventory(("a", ["host", "product"]), ("a", ["host", "product"]))):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                qualification.membership(inventory, ["product"], "host")

    def test_disabled_required_case_is_rejected(self):
        inventory = self.inventory(("a", ["host", "boundary"]))
        inventory["tests"][0]["properties"].append({"name": "DISABLED", "value": True})
        with self.assertRaisesRegex(ValueError, "disabled"):
            qualification.membership(inventory, ["boundary"], "host")

    def test_global_serialization_requires_a_named_owner(self):
        inventory = self.inventory(("a", ["host", "product"]))
        inventory["tests"][0]["properties"].append({"name": "RUN_SERIAL", "value": True})
        with self.assertRaisesRegex(ValueError, "global serialization"):
            qualification.membership(inventory, ["product"], "host")
        inventory["tests"][0]["properties"].pop()
        inventory["tests"][0]["properties"].append({"name": "RESOURCE_LOCK", "value": ["owned_resource"]})
        self.assertEqual(qualification.membership(inventory, ["product"], "host"), {"product": ["a"]})

    def test_explicit_scratch_roots_cannot_collide_across_selected_groups(self):
        inventory = self.inventory(("a", ["host", "product"]), ("b", ["host", "boundary"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inventory["tests"][0]["command"] = ["case", "--scratch-root", str(root / "scratch/../shared")]
            inventory["tests"][1]["command"] = ["case", "--scratch-root", str(root / "shared")]
            with self.assertRaisesRegex(ValueError, "colliding scratch"):
                qualification.membership(inventory, ["product", "boundary"], "all")
            inventory["tests"][1]["command"][-1] = str(root / "other")
            result = qualification.membership(inventory, ["product", "boundary"], "all")
            self.assertEqual(result, {"product": ["a"], "boundary": ["b"]})
            for arguments in (["--scratch-root", "shared"], ["--scratch-root"],
                              ["--scratch-root", str(root / "other"), "--scratch-root", str(root / "shared")]):
                inventory["tests"][1]["command"] = ["case", *arguments]
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "scratch destination"):
                    qualification.membership(inventory, ["product", "boundary"], "all")

    def test_ctest_exit_zero_cannot_hide_missing_skipped_or_unrun_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "tests.xml"
            for body in ('', '<testcase name="required" status="notrun"/>',
                         '<testcase name="required" status="run"><skipped/></testcase>',
                         '<testcase name="required" status="run"><failure/></testcase>'):
                report.write_text('<testsuite>' + body + '</testsuite>')
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, "did not execute"):
                    qualification.validate_ctest_report(report, ["required"])
            report.write_text('<testsuite><testcase name="required" status="run"/></testsuite>')
            qualification.validate_ctest_report(report, ["required"])

    def test_both_native_ci_callers_use_registered_membership_without_publisher(self):
        for family in ("linux", "windows-clang"):
            path = ROOT / ".github/workflows" / ("tests-linux-host.yml" if family == "linux" else "tests-windows-host.yml")
            text = path.read_text()
            self.assertNotIn("--probe-manifest", text)
            self.assertNotIn("JUICER_RUST_PROBE_MANIFEST", text)
            self.assertNotIn("probe_evidence.py", text)
            self.assertIn('product|boundary|tooling', text)
            self.assertIn('product|boundary" -LE gpu', text)
            self.assertIn(f"out/validation/{family}-release/quality/**/*.log", text)
        self.assertIn('product|boundary|tooling|corpus', (ROOT / ".github/workflows/tests-linux-host.yml").read_text())

    def test_malformed_checkpoint_selection_and_empty_collection_schedule_no_work(self):
        import verify
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        with patch.object(qualification, "OutputLease") as lease:
            for changes in ({"reason": ""}, {"selection": []}, {"selection": ["unknown"]},
                            {"preset": "unsupported-debug"}, {"runtime": "unsupported"}):
                args = SimpleNamespace(**({"reason": "requested checkpoint", "selection": ["boundary"],
                                          "preset": preset, "runtime": "host"} | changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    qualification.qualify(ROOT, args)
            with self.assertRaisesRegex(ValueError, "checkpoint receipts are required"):
                qualification.collect_qualification(ROOT, [])
            argv = ["verify.py", "qualification", "--preset", preset]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stopped:
                verify.main()
            self.assertEqual(stopped.exception.code, 2)
            lease.assert_not_called()

    def test_partial_development_and_failed_results_cannot_supply_checkpoint(self):
        for record in ({"mode": "development", "status": "passed"}, {"mode": "checkpoint", "status": "failed"},
                       {"mode": "checkpoint", "status": "passed", "scope_status": "passed", "reason": "partial", "required": ["boundary"], "stages": {}}):
            with self.assertRaises(ValueError):
                qualification.require_checkpoint(record)


if __name__ == "__main__":
    unittest.main()
