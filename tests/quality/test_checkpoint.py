"""Checkpoint ownership, cross-caller reuse and exact CTest membership controls."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import qualification
from execution import Campaign, OutputLease, digest, snapshot, tool_snapshot, write_json

ROOT = Path(__file__).resolve().parents[2]


class CheckpointTests(unittest.TestCase):
    def run(self, result=None):
        try:
            return super().run(result)
        except KeyboardInterrupt:
            # unittest skips its cleanup callbacks when cancellation escapes.
            self.retain = True
            if hasattr(self, "lease") and self.lease.path in OutputLease.held:
                # Completed earlier commands cannot prove an interrupted fixture drained.
                self.lease.blocked = True
            self.doCleanups()
            raise

    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = ROOT / "out/validation" / preset / "checkpoint-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()
        self.retain = bool(os.environ.get("JUICER_TEST_ARTIFACT_DIR"))
        if self.retain:
            print("Checkpoint fixture:", self.root, file=sys.stderr, flush=True)
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
            result = self._outcome.result if self._outcome else None
            if (self.lease.blocked or getattr(self, "retain", False) or result is None
                    or any(test is self for test, _ in [*result.errors, *result.failures])):
                print("Retained checkpoint fixture:", self.root, file=sys.stderr)
            else:
                self.temporary.cleanup()
        self.addCleanup(finish)

    def campaign(self, name):
        return Campaign(self.root, self.preset, name, ["boundary", "tooling", "prepare", "corpus", "production"])

    def configure_fixture(self, preset, testing, name, *, timeout=30):
        from build_evidence import windows_vctip
        environment = dict(os.environ)
        campaign = Campaign(self.root, preset, name, ["prepare"])
        return campaign.run("prepare", ["cmake", "--preset", preset, "-DBUILD_TESTING=" + ("ON" if testing else "OFF")],
            inputs=[p for p in ("CMakeLists.txt", "CMakePresets.json", "binding.h.in") if (self.root / p).is_file()],
            env=environment, timeout=timeout, vctip_executable=windows_vctip(environment))

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
        self.configure_fixture(self.preset, False, "fixture-base")

    def test_real_native_dispatcher_executes_and_reuses_exact_ctest_membership(self):
        self.native_project()
        self.configure_fixture(self.preset, True, "fixture-on")
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

    def compiled_project(self):
        self.native_project()
        files = {
            "CMakeLists.txt": 'cmake_minimum_required(VERSION 3.24)\nproject(Checkpoint LANGUAGES C CXX)\n'
                'set(CMAKE_C_STANDARD 11)\nset(CMAKE_CXX_STANDARD 20)\n'
                'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\ninclude(CTest)\n'
                'configure_file(binding.h.in generated/binding.h COPYONLY)\n'
                'add_library(juicer STATIC src/native.cpp src/abi.c)\n'
                'target_include_directories(juicer PUBLIC "${CMAKE_BINARY_DIR}/generated")\n'
                'add_executable(behavior src/main.cpp)\ntarget_link_libraries(behavior PRIVATE juicer)\n'
                'if(BUILD_TESTING)\nforeach(group product boundary tooling corpus)\n'
                'add_test(NAME mini.${group} COMMAND behavior 10)\n'
                'set_tests_properties(mini.${group} PROPERTIES LABELS "host;${group}")\nendforeach()\nendif()\n',
            "binding.h.in": "#define BINDING 4\n",
            "src/abi.h": 'typedef struct Witness { int first; int second; } Witness;\n',
            "src/abi.c": '#include <stddef.h>\n#include "abi.h"\n'
                '_Static_assert(sizeof(Witness)==8,"C11 ABI");\n'
                'int c_size(void){return sizeof(Witness);}\nint c_offset(void){return offsetof(Witness,second);}\n',
            "src/native.cpp": '#include "binding.h"\nint native(){return 6+BINDING;}\n',
            "src/main.cpp": '#include <cstdio>\n#include <cstdlib>\n#include <cstddef>\n#include "abi.h"\n'
                'static_assert(sizeof(Witness)==8);\nint native();\nextern "C" int c_size(void);\nextern "C" int c_offset(void);\n'
                'int main(int argc,char** argv){int v=native();const char* offset=std::getenv("JUICER_P0_A_RUNTIME_OFFSET");'
                'if(offset)v+=std::atoi(offset);std::printf("value=%d abi=%d offset=%d\\n",v,c_size(),c_offset());'
                'return argc!=2 || v!=std::atoi(argv[1]) || c_size()!=sizeof(Witness) || c_offset()!=offsetof(Witness,second);}\n',
            "scripts/check-quality.py": 'import sys\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\np=Path("out")/(preset+"-quality-count.txt")\n'
                'p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nprint("miniature quality scheduling witness")\n',
            "tests/ffi/production_contract.py": 'import json,sys,subprocess\nfrom pathlib import Path\n'
                'preset=sys.argv[sys.argv.index("--preset")+1]\nout=Path(sys.argv[sys.argv.index("--artifacts")+1])\n'
                'build=Path("out/build")/preset\nexe=build/("behavior.exe" if sys.platform=="win32" else "behavior")\n'
                'value=subprocess.check_output([str(exe.resolve()),"10"],text=True,timeout=10)\n'
                'assert "value=10 abi=8 offset=4" in value\nout.mkdir(parents=True)\n'
                '(out/"production.json").write_text(json.dumps({"scope":"miniature protocol only","behavior":value}))\n',
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        compiler = ({"CMAKE_C_COMPILER": "clang-cl.exe", "CMAKE_CXX_COMPILER": "clang-cl.exe", "CMAKE_LINKER_TYPE": "LLD",
                     "CMAKE_C_FLAGS": "/Brepro", "CMAKE_CXX_FLAGS": "/Brepro"}
                    if os.name == "nt" else {"CMAKE_C_COMPILER": "/usr/bin/gcc-13", "CMAKE_CXX_COMPILER": "/usr/bin/g++-13"})
        (self.root / "CMakePresets.json").write_text(json.dumps({"version": 6, "configurePresets": [
            {"name": p, "generator": "Ninja", "binaryDir": str(self.root / "out/build" / p),
             "cacheVariables": {**compiler, "CMAKE_BUILD_TYPE": "Release" if p.endswith("release") else "Debug"}}
            for p in qualification.PRESETS]}))
        return files

    def replay_preparation_renewal(self):
        """Four protocol lanes use this OS's tiny binaries, never product qualification."""
        self.retain = True
        calls, deletions, artifact_reads, input_checks, tool_checks = [], [], [], [], []
        started = time.monotonic()
        progress = {"scope": "native miniature protocol only", "complete": False, "phases": []}
        progress_path = self.root / "out/validation" / self.preset / "replay-progress.json"
        progress_path.parent.mkdir(parents=True)
        def record_phase(name, action, *args, **kwargs):
            row = {"name": name, "status": "running"}
            progress["phases"].append(row)
            write_json(progress_path, progress)
            start = time.monotonic()
            try:
                result = action(*args, **kwargs)
                row["status"] = "passed"
                return result
            except BaseException as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "raised",
                           error=type(error).__name__)
                raise
            finally:
                row["seconds"] = time.monotonic() - start
                progress.update(seconds=time.monotonic() - started, commands=calls, deleted=deletions,
                                preparation_input_checks=len(input_checks),
                                preparation_input_check_seconds=sum(input_checks),
                                tool_checks=len(tool_checks), tool_check_seconds=sum(tool_checks),
                                artifact_hash_reads=len(artifact_reads),
                                artifact_hash_seconds=sum(value[0] for value in artifact_reads))
                write_json(progress_path, progress)
        files = record_phase("fixture.configure", self.compiled_project)
        artifact_paths = set()
        actual_digest = qualification.digest
        def measured_digest(path):
            path = Path(path).resolve()
            started = time.monotonic()
            result = actual_digest(path)
            if path in artifact_paths:
                artifact_reads.append((time.monotonic() - started, path.stat().st_size))
            return result
        actual_run = Campaign.run
        from build_dependencies import NinjaGraph
        actual_invalidate = NinjaGraph.invalidate
        def counted(campaign, stage, command, **kwargs):
            started = time.monotonic()
            try:
                return actual_run(campaign, stage, command, **kwargs)
            finally:
                calls.append({"stage": stage, "command": list(command), "seconds": time.monotonic() - started})
        def invalidate(graph, names):
            removed = actual_invalidate(graph, names)
            deletions.extend(removed)
            return removed
        actual_check = qualification.check_preparation_inputs
        def measured_inputs(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_check(*args, **kwargs)
            finally:
                input_checks.append(time.monotonic() - started)
        actual_tools = qualification.tool_snapshot
        def measured_tools(*args, **kwargs):
            started = time.monotonic()
            try:
                return actual_tools(*args, **kwargs)
            finally:
                tool_checks.append(time.monotonic() - started)
        def checkpoint(preset, campaign, selection):
            args = SimpleNamespace(reason="miniature complete protocol replay", selection=selection, preset=preset,
                                   runtime="all", campaign=campaign, repeat_reason="", base="fixture-baseline")
            # Other names model collector membership with native binaries on this OS.
            # The public native-platform admission remains exercised for the native lane.
            if preset == self.preset:
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification.qualify, self.root, args)
            with OutputLease(self.root, self.root / "out/build" / preset):
                return record_phase(f"checkpoint.{preset}.{campaign}", qualification._qualify_owned,
                                    self.root, self.root / "out/build" / preset, args)
        paths = []
        with patch.object(Campaign, "run", autospec=True, side_effect=counted), \
                patch.object(NinjaGraph, "invalidate", autospec=True, side_effect=invalidate), \
                patch.object(qualification, "digest", side_effect=measured_digest), \
                patch.object(qualification, "check_preparation_inputs", side_effect=measured_inputs), \
                patch.object(qualification, "tool_snapshot", side_effect=measured_tools), \
                patch("execution.tool_snapshot", side_effect=measured_tools):
            for preset in qualification.PRESETS:
                build = self.root / "out/build" / preset
                campaign = Campaign(self.root, preset, "cold", ["prepare"])
                if preset.endswith("release"):
                    record_phase(f"prepare.{preset}.cold", qualification.prepare_build, campaign, "prepare", source=self.root, build=build,
                        inputs=list(files), timeout=60, targets=["all"], configure_args=["-DBUILD_TESTING=OFF"], checkpoint=True)
                    paths.append(checkpoint(preset, "off", ["production"]))
                else:
                    record_phase(f"configure.{preset}.cold", self.configure_fixture, preset, True, "fixture-on", timeout=60)
                groups = ["product", "boundary"]
                if preset.endswith("debug"):
                    groups += ["tooling"]
                if preset == "linux-debug":
                    groups += ["corpus"]
                paths.append(checkpoint(preset, "initial", groups))
            initial = self.root / "out/validation" / self.preset / "initial/qualification.json"
            selection = json.loads(initial.read_text())["selection"]
            # CMake may resolve preset compiler spellings during cold setup.
            # Admit the resulting tool inventory before the proof-only comparison.
            stable = checkpoint(self.preset, "stable", selection)
            paths = [stable if p == initial else p for p in paths]
            initial = stable
            initial_collection = record_phase("collect.initial", qualification.collect_qualification, self.root, paths)
            self.assertEqual(initial_collection["status"], "passed")
            first = json.loads(initial.read_text())
            proof = json.loads(Path(first["stages"]["prepare"]["receipt"]).read_text())["scoped_preparation"]
            artifacts = {p: digest(Path(p)) for p in proof["outputs"]}
            before_source = snapshot(self.root, list(files))
            originals = {str(p): digest(p) for p in (self.root / "out/validation").rglob("*")
                         if p.is_file() and p != progress_path}
            for path in paths:
                artifact_paths.update(Path(p).resolve() for p in json.loads(path.read_text())["stages"].get("prepare", {}).get("payloads", {}))
            warm = []
            for trial in range(3):
                start = time.monotonic()
                call_start, delete_start, hash_start = len(calls), len(deletions), len(artifact_reads)
                check_start = len(input_checks)
                tool_start = len(tool_checks)
                with patch.object(qualification, "check_scoped_preparation", side_effect=ValueError("injected checkpoint proof validation failure")):
                    with self.assertRaisesRegex(ValueError, "injected checkpoint"):
                        checkpoint(self.preset, f"failed-{trial}", first["selection"])
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, paths)
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                state = json.loads(self.state_path.read_text())
                repair = self.campaign(f"repair-{trial}")
                repair_start = time.monotonic()
                renewed = record_phase(f"prepare.repair-{trial}", qualification.prepare_checkpoint,
                    repair, self.build, self.state_path, state, dict(os.environ),
                    timeout=60, repeat_reason="diagnosed checkpoint validation failure; completed artifact producer intact")
                repair_seconds = time.monotonic() - repair_start
                self.assertNotEqual(renewed["sha256"], first["stages"]["prepare"]["sha256"])
                current = checkpoint(self.preset, f"continued-{trial}", first["selection"])
                paths = [current if p == initial or p.parent.name.startswith("continued-") and p.parent.parent.name == self.preset else p for p in paths]
                collection_start = time.monotonic()
                collected = record_phase(f"collect.continued-{trial}", qualification.collect_qualification, self.root, paths)
                collection_seconds = time.monotonic() - collection_start
                self.assertEqual(collected["status"], "passed")
                self.assertEqual(before_source, snapshot(self.root, list(files)))
                self.assertEqual(artifacts, {p: digest(Path(p)) for p in artifacts})
                value = subprocess.check_output([str(self.build / ("behavior.exe" if os.name == "nt" else "behavior")), "10"], text=True, timeout=10)
                self.assertIn("value=10 abi=8 offset=4", value)
                logs = [self.root / json.loads(p.read_text())["output"] for p in repair.directory.glob("attempts/*/result.json")]
                self.assertTrue(any("no work to do" in p.read_text() for p in logs))
                warm.append({"seconds": time.monotonic() - start, "commands": calls[call_start:], "deleted": deletions[delete_start:],
                             "repair_seconds": repair_seconds, "collection_seconds": collection_seconds,
                             "artifact_hash_reads": len(artifact_reads) - hash_start,
                             "artifact_hash_seconds": sum(row[0] for row in artifact_reads[hash_start:]),
                             "artifact_bytes_hashed": sum(row[1] for row in artifact_reads[hash_start:]),
                             "preparation_input_checks": len(input_checks) - check_start,
                             "preparation_input_check_seconds": sum(input_checks[check_start:]),
                             "tool_checks": len(tool_checks) - tool_start,
                             "tool_check_seconds": sum(tool_checks[tool_start:]),
                             "renewed": renewed, "checkpoint": str(current), "collection": collected})
            self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
            summary = {"scope": "native miniature protocol; four synthetic lane names, no Film-Juicer/GPU/OFX qualification",
                       "fixture": str(self.root), "native": os.name, "artifacts": artifacts, "inputs": before_source,
                       "initial_collection": initial_collection, "warm": warm, "seconds": time.monotonic() - started,
                       "tool_checks": len(tool_checks), "tool_check_seconds": sum(tool_checks),
                       "rust_recovery": "not applicable; no Rust fixture", "cold_and_warm_commands": calls}
            for trial in warm:
                self.assertFalse(trial["deleted"])
                self.assertEqual([c["stage"] for c in trial["commands"]], ["prepare"],
                                 "proof renewal repeated unchanged quality/product work")
            write_json(self.root / "out/validation" / self.preset / "replay.json", summary)
            progress.update(complete=True, seconds=time.monotonic() - started)
            write_json(progress_path, progress)

    def prepare_compiled_project(self):
        self.retain = True
        self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        return qualification.prepare_checkpoint(self.campaign("compiled"), self.build, self.state_path, self.state,
                                                dict(os.environ), timeout=60)

    def test_preparation_output_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("behavior.exe" if os.name == "nt" else "behavior")

    def test_collection_rechecks_current_preparation_inputs(self):
        self.assert_collection_input_mutations([
            ("generated/binding.h", lambda b: b.replace(b"BINDING 4", b"BINDING 5"), "dependency contents"),
            ("CTestTestfile.cmake", lambda b: b + b'\nadd_test(unrun "missing")\n', "test registration"),
            ("generated/new-shadow.h", lambda b: b"#define SHADOW 1\n", "include membership"),
        ])

    def test_collection_rechecks_changed_cache(self):
        # Retain this mutation in its own fixture. DrvFS can round timestamps
        # when restoring them, so it cannot be another case's fresh baseline.
        self.assert_collection_input_mutations([
            ("CMakeCache.txt", lambda b: b.replace(b"CMAKE_CXX_FLAGS:STRING=", b"CMAKE_CXX_FLAGS:STRING=-DABI_CHANGED=1 "), "configuration inputs"),
        ], retain_mutation=True)

    def assert_collection_input_mutations(self, mutations, *, retain_mutation=False):
        self.prepare_compiled_project()
        args = SimpleNamespace(reason="collection input drift control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="admitted", repeat_reason="", base="fixture-baseline")
        path = qualification.qualify(self.root, args)
        record = json.loads(path.read_text())
        stage = record["stages"]["prepare"]
        identity = qualification.preparation_identity(self.build, stage)
        # Other fixture lanes are deliberately absent; the complete replay owns
        # positive aggregation. This control must reach that completeness check.
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [path])
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        outcomes = []
        for name, mutate, diagnostic in mutations:
            with self.subTest(input=name):
                target = self.build / name
                content = target.read_bytes() if target.exists() else None
                stamp = target.stat() if target.exists() else None
                try:
                    changed = mutate(content or b"")
                    self.assertNotEqual(content, changed)
                    target.write_bytes(changed)
                    if stamp:
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.preparation_identity(self.build, stage)
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        qualification.collect_qualification(self.root, [path])
                    value = subprocess.check_output([str(executable), "10"], text=True, timeout=10)
                    self.assertIn("value=10 abi=8 offset=4", value)
                    outcomes.append({"input": name, "rejected": diagnostic,
                                     "mtime_restoration_exact": bool(stamp) and target.stat().st_mtime_ns == stamp.st_mtime_ns,
                                     "behavior": value.strip(), "mutation_retained": retain_mutation})
                finally:
                    if retain_mutation:
                        pass
                    elif content is None:
                        target.unlink()
                    else:
                        target.write_bytes(content)
                        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            if not retain_mutation:
                self.assertEqual(identity, qualification.preparation_identity(self.build, stage))
        write_json(path.parent / "collection-controls.json", outcomes)

    def test_consumed_environment_change_cannot_reuse_product(self):
        self.retain = True
        files = self.compiled_project()
        self.configure_fixture(self.preset, True, "fixture-on", timeout=60)
        args = SimpleNamespace(reason="consumed runtime environment control", selection=["product"], preset=self.preset,
                               runtime="all", campaign="environment-baseline", repeat_reason="", base="fixture-baseline")
        outcomes = {}
        with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "0"}):
            first = qualification.qualify(self.root, args)
            original = first.read_bytes()
            source = snapshot(self.root, list(files))
            args.campaign = "environment-changed"
            with patch.dict(os.environ, {"JUICER_P0_A_RUNTIME_OFFSET": "1"}):
                executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
                stale = subprocess.run([str(executable), "10"], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(stale.returncode, 0)
                self.assertIn("value=11 abi=8 offset=4", stale.stdout)
                with self.assertRaises(subprocess.CalledProcessError):
                    qualification.qualify(self.root, args)
                with self.assertRaisesRegex(ValueError, "pending checkpoint work"):
                    qualification.collect_qualification(self.root, [first])
                self.assertEqual((self.root / "out" / (self.preset + "-quality-count.txt")).read_text(), "2")
                failed = self.root / "out/validation" / self.preset / args.campaign / "qualification.json"
                outcomes.update(changed_checkpoint=str(failed), witness=stale.stdout.strip(), variable="JUICER_P0_A_RUNTIME_OFFSET")
            self.assertEqual(original, first.read_bytes())
            self.assertEqual(source, snapshot(self.root, list(source)))
            args.campaign = "environment-restored"
            current = qualification.qualify(self.root, args)
            self.assertEqual(json.loads(current.read_text())["status"], "passed")
            outcomes.update(baseline=str(first), restored=str(current), status="changed environment executed quality and failed real product; restored environment passed")
            write_json(current.parent / "environment-control.json", outcomes)

    def test_preparation_archive_bytes_are_required_for_reuse(self):
        self.assert_output_mutation("juicer.lib" if os.name == "nt" else "libjuicer.a")

    def assert_output_mutation(self, name):
        first = self.prepare_compiled_project()
        original = Path(first["receipt"]).read_bytes()
        artifact = self.build / name
        content, stamp = artifact.read_bytes(), artifact.stat()
        artifact.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "payload changed|artifacts changed"):
            qualification.prepare_checkpoint(self.campaign("mutated"), self.build, self.state_path, self.state,
                                             dict(os.environ), timeout=60)
        self.assertEqual(self.state["prepare"]["status"], "failed")
        self.assertEqual(original, Path(first["receipt"]).read_bytes())
        for attempt in range(2):
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                qualification.prepare_checkpoint(self.campaign(f"repair-mutated-{attempt}"), self.build, self.state_path,
                    self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed artifact admission failure")
            self.assertEqual(self.state["prepare"]["status"], "failed")
            self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])

    def test_preparation_repair_rejects_unproved_process_receipt(self):
        first = self.prepare_compiled_project()
        record = json.loads(Path(first["receipt"]).read_text())
        process = Path(record["process_tree"]["receipt"])
        document = json.loads(process.read_text())
        document["drained"] = False
        process.write_text(json.dumps(document))
        with patch.object(Campaign, "run", side_effect=AssertionError("dispatched with unproved drainage")):
            for trial in range(2):
                with self.assertRaisesRegex(ValueError, "process evidence"):
                    qualification.prepare_checkpoint(self.campaign(f"undrained-{trial}"), self.build, self.state_path,
                        self.state, dict(os.environ), timeout=60, repeat_reason="diagnosed process proof failure")
                self.assertEqual(self.state["prepare"]["status"], "failed")
                self.assertEqual(first["payloads"], self.state["prepare"]["payloads"])

    def test_preparation_identity_requires_exact_artifacts_and_current_producer(self):
        first = self.prepare_compiled_project()
        receipt = Path(first["receipt"]).read_bytes()
        expected = qualification.preparation_identity(self.build, first)
        candidate = {**first, "payloads": dict(first["payloads"])}
        candidate["payloads"].pop(next(iter(candidate["payloads"])))
        with self.assertRaisesRegex(ValueError, "exact artifact identities"):
            qualification.preparation_identity(self.build, candidate)
        from build_evidence import preparation_paths
        baseline, pending = preparation_paths(self.build, ["all"])
        pending.write_text('{"test": "unproved artifact writer"}')
        with self.assertRaisesRegex(ValueError, "pending preparation"):
            qualification.preparation_identity(self.build, first)
        # Only this fixture's injected pending marker is removed; no writer ran.
        pending.unlink()
        self.assertEqual(expected, qualification.preparation_identity(self.build, first))
        original = baseline.read_bytes()
        changed = json.loads(original)
        changed["producer"] = "superseded"
        baseline.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "latest completed preparation"):
            qualification.preparation_identity(self.build, first)
        self.assertEqual(Path(first["receipt"]).read_bytes(), receipt)

    def test_restored_time_edit_exposes_stale_behavior_before_preparation(self):
        first = self.prepare_compiled_project()
        executable = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        subprocess.run([str(executable), "10"], check=True, capture_output=True, timeout=10)
        source = self.root / "src/native.cpp"
        content, stamp = source.read_bytes(), source.stat()
        source.write_bytes(content.replace(b"6+BINDING", b"7+BINDING"))
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        old = subprocess.run([str(executable), "11"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(old.returncode, 0)
        self.assertIn("value=10 abi=8 offset=4", old.stdout)
        with self.assertRaisesRegex(ValueError, "dependency contents"):
            qualification.check_scoped_preparation(self.campaign("stale"), self.build, Path(first["receipt"]))
        qualification.prepare_checkpoint(self.campaign("rebuilt"), self.build, self.state_path, self.state,
                                         dict(os.environ), timeout=60)
        new = subprocess.run([str(executable), "11"], check=True, capture_output=True, text=True, timeout=10)
        self.assertIn("value=11 abi=8 offset=4", new.stdout)

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
        self.configure_fixture(self.preset, True, "fixture-on")
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

    def test_collection_rechecks_tool_bytes_at_completion(self):
        self.assert_collection_tool_seal("bytes")

    def test_collection_rechecks_source_data_at_completion(self):
        data = self.root / "Resources/data.txt"
        data.parent.mkdir()
        data.write_text("authored")
        inputs = [data.relative_to(self.root).as_posix()]
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        actual_digest = qualification.digest
        def mutate_during_read(target):
            value = actual_digest(target)
            if str(target) == stage["receipt"]:
                stamp = data.stat()
                data.write_text("modified")
                os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=inputs), \
                patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data inputs changed during collection"):
                qualification.collect_qualification(self.root, [path])

    def test_collection_rechecks_source_membership_at_completion(self):
        import verify
        self.retain = True
        self.native_project()
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = json.loads(path.read_text())
        target = self.root / "out/validation" / self.preset / "corpus-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_digest = qualification.digest
        def mutate_during_read(path):
            value = actual_digest(path)
            if str(path) == stage["receipt"]:
                (self.root / "new-consumed.cpp").write_text("int added();\n")
            return value
        with patch.object(qualification, "digest", side_effect=mutate_during_read):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])

    def test_collection_rechecks_ignored_input_membership_at_completion(self):
        import verify
        self.retain = True
        self.native_project()
        added = self.root / "tests/ffi/out/added.bin"
        added.parent.mkdir(parents=True)
        inputs = qualification.checkpoint_inputs(self.root, self.build, "corpus")
        stage = self.run_stage("corpus", inputs=inputs)
        path = self.retained_record(stage)
        record = json.loads(path.read_text())
        target = self.root / "out/validation" / self.preset / "ignored-membership-record.json"
        path.rename(target)
        record["source"] = verify.source_snapshot(self.root)
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
            qualification.collect_qualification(self.root, [target])
        actual_inputs = qualification.checkpoint_inputs
        changed = []
        def mutate_after_discovery(*args, **kwargs):
            result = actual_inputs(*args, **kwargs)
            if not changed:
                added.write_bytes(b"new declared test input")
                changed.append(added.relative_to(self.root).as_posix())
            return result
        with patch.object(qualification, "checkpoint_inputs", side_effect=mutate_after_discovery):
            with self.assertRaisesRegex(ValueError, "source/data membership changed during collection"):
                qualification.collect_qualification(self.root, [target])
        self.assertEqual(changed, [added.relative_to(self.root).as_posix()])
        subprocess.run(["git", "check-ignore", "--quiet", "--", changed[0]], cwd=self.root,
                       check=True, capture_output=True, timeout=10)
        self.assertEqual(record["source"], verify.source_snapshot(self.root))
        self.assertIn(changed[0], actual_inputs(self.root, self.build, "corpus"))
        self.assertEqual(digest(Path(stage["receipt"])), stage["sha256"])
        write_json(target.parent / "ignored-membership-control.json",
                   {"added": changed[0], "git_ignored": True, "candidate_unchanged": True,
                    "rejected": "source/data membership changed during collection",
                    "receipt": stage["receipt"], "sha256": stage["sha256"]})

    def test_collection_rechecks_absent_tool_configuration_at_completion(self):
        self.assert_collection_tool_seal("absent")

    def test_collection_rechecks_tool_path_binding_at_completion(self):
        self.assert_collection_tool_seal("binding")

    def assert_collection_tool_seal(self, change):
        self.retain = True
        directory = self.root / "out/validation" / self.preset / "tool-seal"
        first, second, alias = (directory / name for name in ("first", "second", "alias"))
        for target in (first, second):
            target.mkdir(parents=True)
            (target / "tool").write_text("first")
        def bind(target):
            if os.name == "nt":
                subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
                               check=True, capture_output=True, timeout=10)
            else:
                alias.symlink_to(target, target_is_directory=True)
        def unbind():
            if os.name == "nt":
                alias.rmdir()
            else:
                alias.unlink()
        bind(first)
        config = directory / "absent.toml"
        tools = [self.python, str(first / "tool"), str(second / "tool"), str(alias / "tool"), str(config)]
        stages = {name: self.run_stage(name, tools=tools) for name in ("boundary", "corpus", "tooling")}
        path = self.retained_record(stages["corpus"])
        record = json.loads(path.read_text())
        record.update(selection=list(stages), required=list(stages), stages=stages)
        path.write_text(json.dumps(record))
        originals = {stage["receipt"]: digest(Path(stage["receipt"])) for stage in stages.values()}
        actual_digest = qualification.digest
        changed = []
        def mutate_during_read(path):
            value = actual_digest(path)
            if not changed and str(path) in originals:
                if change == "bytes":
                    tool = first / "tool"
                    stamp = tool.stat()
                    tool.write_text("other")
                    os.utime(tool, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                elif change == "absent":
                    config.write_text("[build]\n")
                else:
                    unbind()
                    bind(second)
                changed.append(change)
            return value
        with patch.object(qualification, "checkpoint_inputs", return_value=["input.txt"]):
            with patch.object(qualification, "tool_snapshot", wraps=qualification.tool_snapshot) as reads:
                with self.assertRaisesRegex(ValueError, "full checkpoint membership"):
                    qualification.collect_qualification(self.root, [path])
                self.assertEqual(reads.call_count, 2, "one opening and one closing tool scan for all stages")
            with patch.object(qualification, "digest", side_effect=mutate_during_read):
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed during collection"):
                    qualification.collect_qualification(self.root, [path])
            self.assertEqual(changed, [change])
            if change != "binding":
                # A later collection must observe the retained mutation afresh.
                with self.assertRaisesRegex(ValueError, "toolchain/configuration changed"):
                    qualification.collect_qualification(self.root, [path])
        self.assertEqual(originals, {p: digest(Path(p)) for p in originals})
        write_json(directory / "control.json", {"change": change, "rejected": True,
                   "scope": "real collector; incomplete lane membership remains rejected",
                   "original_receipts": originals, "tool_reads_per_collection": 2})

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


class FixtureLifetimeTests(unittest.TestCase):
    def test_cancellation_keeps_fixture_owner_after_completed_or_unproved_command(self):
        for drained in (True, False):
            with self.subTest(drained=drained):
                case = CheckpointTests("test_new_campaign_reuses_original_success_without_execution")
                previous = os.environ.get("JUICER_OUTPUT_LEASES")
                def cancel():
                    case.retain = True
                    if drained:
                        case.run_stage("boundary")
                        raise KeyboardInterrupt
                    # No process is launched: cancellation has no terminal proof.
                    def unproved_launch(*args, **kwargs):
                        args[7].update(drained=False, status="running")
                        raise KeyboardInterrupt
                    with patch("execution.run_process_tree", side_effect=unproved_launch):
                        case.run_stage("boundary")
                setattr(case, case._testMethodName, cancel)
                with self.assertRaises(KeyboardInterrupt):
                    case.run(unittest.TestResult())
                self.assertNotIn(case.lease.path, OutputLease.held)
                self.assertEqual(os.environ.get("JUICER_OUTPUT_LEASES"), previous)
                receipt = next((case.root / "out/validation").glob("*/first/attempts/*/result.json"))
                record = json.loads(receipt.read_text())
                owner = case.lease.path / "owner.json"
                self.assertEqual(record["process_tree"]["drained"], drained)
                self.assertTrue(owner.exists())
                self.assertEqual(json.loads(owner.read_text())["status"], "blocked")
                if drained:
                    self.assertEqual(record["status"], "passed")
                    process = json.loads(Path(record["process_tree"]["receipt"]).read_text())
                    self.assertTrue(process["drained"])
                    self.assertEqual(process["status"], "completed")
                else:
                    self.assertEqual(record["status"], "blocked")
                    self.assertTrue((case.root / "out/validation" / case.preset / "first/running.lock").exists())
                write_json(case.root / "out/validation" / case.preset / "cancellation.json",
                           {"scope": "cancelled fixture remains blocked; unproved branch launches no child",
                            "command_drained": drained, "receipt": str(receipt), "sha256": digest(receipt),
                            "owner_retained": owner.exists()})


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
