"""Behavior controls through real native/Cargo preparation and CTest entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_evidence
from execution import Campaign, OutputLease, digest

ROOT = Path(__file__).resolve().parents[2]


class CompilationInputsTests(unittest.TestCase):
    def test_fresh_boundary_observes_include_directory_symlink_retarget(self):
        from build_dependencies import CompilationInputs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            build = root / "build"
            build.mkdir()
            for name in ("first", "second"):
                (root / name).mkdir()
            (root / "first/entry.h").write_text("// no selected consumer\n")
            (root / "second/entry.h").write_text('#include "../selected.h"\n')
            (root / "selected.h").write_text("// selected\n")
            (root / "main.cpp").write_text('#include <entry.h>\n')
            link = root / "current"

            def link_to(directory):
                if os.name == "nt":
                    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(directory)], check=True, capture_output=True, timeout=10)
                else:
                    link.symlink_to(directory, target_is_directory=True)

            link_to(root / "first")
            row = {"file": "main.cpp", "directory": str(root), "arguments": ["c++", "-I" + str(link)]}
            (build / "compile_commands.json").write_text(json.dumps([row]))
            first = CompilationInputs(root, build)
            self.assertEqual(first.paths(row), {"main.cpp", "first/entry.h"})
            os.rmdir(link) if os.name == "nt" else link.unlink()
            link_to(root / "second")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(row), {"main.cpp", "second/entry.h", "selected.h"})

    def test_ordered_variants_and_fresh_boundary_observe_changed_includes(self):
        from build_dependencies import CompilationInputs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            build = root / "build"
            build.mkdir()
            for directory in ("first", "second"):
                (root / directory).mkdir()
                (root / directory / "chosen.h").write_text("// " + directory)
            source = root / "main.cpp"
            source.write_text('#include <chosen.h>\n')
            forced = root / "forced.h"
            forced.write_text("// forced\n")
            rows = [
                {"file": "main.cpp", "directory": str(root), "arguments": ["c++", "-I" + str(root / a), "-I" + str(root / b), *extra]}
                for a, b, extra in [("first", "second", []), ("second", "first", ["-include", "forced.h"])]
            ]
            (build / "compile_commands.json").write_text(json.dumps(rows))
            boundary = CompilationInputs(root, build)
            self.assertEqual(boundary.paths(rows[0]), {"main.cpp", "first/chosen.h"})
            self.assertEqual(boundary.paths(rows[1]), {"main.cpp", "second/chosen.h", "forced.h"})
            self.assertEqual(boundary.paths(rows[0]), {"main.cpp", "first/chosen.h"})
            stamp = source.stat()
            source.write_text('#include <other.h>\n')
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            (root / "first/other.h").write_text("// next boundary\n")
            fresh = CompilationInputs(root, build)
            self.assertEqual(fresh.paths(rows[0]), {"main.cpp", "first/other.h"})
            # Membership in the earlier namespace changes lookup even when the
            # source bytes and the formerly resolved header remain unchanged.
            (root / "second/other.h").write_text("// new earlier match\n")
            final = CompilationInputs(root, build)
            self.assertEqual(final.paths(rows[1]), {"main.cpp", "second/other.h", "forced.h"})


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        preset = "windows-clang-debug" if os.name == "nt" else "linux-debug"
        parent = ROOT / "out/validation" / preset / "build-evidence-controls"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, delete=False)
        self.root = Path(self.temporary.name).resolve()

        def finish_fixture():
            result = self._outcome.result if self._outcome else None
            failed = result and any(
                test is self or getattr(test, "test_case", None) is self
                for test, _ in [*result.errors, *result.failures]
            )
            blocked = list(self.root.glob("out/verification-locks/*/owner.json"))
            if failed or blocked:
                # Keep configured paths and unproved owners in place for diagnosis.
                retained = self.root / "retained-failure.json"
                retained.write_text(json.dumps({"test": self.id(), "fixture": str(self.root), "blocked_owners": [str(path) for path in blocked]}, indent=2) + "\n")
                print(f"Retained build-evidence fixture: {retained}", file=sys.stderr)
            else:
                self.temporary.cleanup()

        self.addCleanup(finish_fixture)
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.campaign = Campaign(self.root, preset, "control", ["prepare", "tests"])
        files = {
            "CMakeLists.txt": "cmake_minimum_required(VERSION 3.24)\nproject(ContentProof LANGUAGES C CXX)\n"
            "set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_C_STANDARD 11)\n"
            'enable_testing()\nif(ENABLE_RUST)\ninclude("'
            + (self.source / "cmake/JuicerRust.cmake").as_posix()
            + '")\nendif()\n'
            "configure_file(binding.h.in generated/binding.h COPYONLY)\n"
            "add_executable(behavior main.cpp native.cpp abi.c)\n"
            'target_include_directories(behavior PRIVATE "${CMAKE_BINARY_DIR}/shadow" "${CMAKE_BINARY_DIR}/generated")\n'
            "if(ENABLE_RUST)\ntarget_link_libraries(behavior PRIVATE film_juicer_rust)\nelse()\ntarget_sources(behavior PRIVATE rust_stub.c)\nendif()\n"
            "add_test(NAME behavior COMMAND behavior 10)\n",
            "main.cpp": '#include <cstdio>\n#include <cstdlib>\n#include "middle.h"\n'
            'int native();\nextern "C" int rust_value();\n'
            "int main(int argc,char** argv){int value=native()+header()+rust_value();"
            'std::printf("value=%d\\n",value);return argc!=2 || value!=std::atoi(argv[1]);}\n',
            "rust_stub.c": "int rust_value(void){return 3;}\n",
            "native.cpp": "int native(){return 1;}\n",
            "middle.h": '#include "detail.h"\n',
            "detail.h": '#include "binding.h"\ninline int header(){return 2+BINDING;}\n',
            "binding.h.in": "#define BINDING 4\n",
            "Cargo.toml": '[workspace]\nmembers=["rust/film-juicer-core","rust/film-juicer-plugin"]\nresolver="3"\n',
            "rust/film-juicer-core/Cargo.toml": '[package]\nname="film-juicer-core"\nversion="0.1.0"\nedition="2024"\n',
            "rust/film-juicer-core/src/lib.rs": "pub fn value()->i32{3}\n",
            "rust/film-juicer-plugin/Cargo.toml": '[package]\nname="film-juicer-plugin"\nversion="0.1.0"\nedition="2024"\n'
            '[lib]\ncrate-type=["staticlib"]\n[dependencies]\nfilm-juicer-core={path="../film-juicer-core"}\n'
            "[features]\ntest-support=[]\n",
            "rust/film-juicer-plugin/src/lib.rs": '#[unsafe(no_mangle)]\npub extern "C" fn rust_value()->i32{film_juicer_core::value()}\n',
            "abi.c": 'extern int rust_value(void);\n_Static_assert(_Generic(&rust_value,int(*)(void):1,default:0),"C11 witness");\n',
            "rust-toolchain.toml": '[toolchain]\nchannel="1.98.1"\n',
            "rustfmt.toml": "",
            "clippy.toml": "",
        }
        (self.source / "cmake").mkdir()
        (self.source / "cmake/JuicerRust.cmake").write_bytes(
            (ROOT / "cmake/JuicerRust.cmake").read_bytes()
        )
        for name, content in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        subprocess.run(
            [
                "cargo",
                "generate-lockfile",
                "--manifest-path",
                str(self.source / "Cargo.toml"),
            ],
            check=True,
            capture_output=True,
        )
        self.args = [
            "-G",
            "Ninja",
            "-DBUILD_TESTING=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DENABLE_RUST=OFF",
        ]
        if os.name == "nt":
            self.args += [
                "-DCMAKE_C_COMPILER=clang-cl.exe",
                "-DCMAKE_CXX_COMPILER=clang-cl.exe",
                "-DCMAKE_LINKER_TYPE=LLD",
            ]
        else:
            self.args += [
                "-DCMAKE_C_COMPILER=/usr/bin/gcc-13",
                "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13",
            ]

    def inputs(self):
        return [
            path.relative_to(self.root).as_posix()
            for path in self.source.rglob("*")
            if path.is_file()
        ]

    def prepare(self, **kwargs):
        if "-DENABLE_RUST=ON" in self.args:
            # Model the supported tagged Cargo cache; production keeps Cargo's
            # refusal to clean arbitrary untagged output directories.
            cargo_cache = self.build / "cargo"
            cargo_cache.mkdir(parents=True, exist_ok=True)
            (cargo_cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55\n")
        _, self.prepared = build_evidence.prepare_build(
            self.campaign,
            "prepare",
            source=self.source,
            build=self.build,
            inputs=self.inputs(),
            configure_args=self.args,
            timeout=120,
            **{"targets": ["behavior"], **kwargs},
        )
        return self.prepared

    def execute(self, **kwargs):
        receipt = build_evidence.run_ctest(
            self.campaign,
            "tests",
            build=self.build,
            prepared=self.prepared,
            inputs=self.inputs(),
            timeout=30,
            **{"tests": ["behavior"], **kwargs},
        )
        output = (self.root / json.loads(receipt.read_text())["output"]).read_text()
        self.assertIn("100% tests passed", output)
        return receipt

    def test_native_transitive_rust_generated_and_configuration_edits_execute_new_behavior(
        self,
    ):
        self.prepare()
        self.execute()
        # Finite declared controls: each semantic mutation must fail the original
        # independent value witness after a same-size/restored-time source edit.
        for name, before, after in [
            ("native.cpp", "return 1", "return 8"),
            ("detail.h", "return 2", "return 8"),
            ("binding.h.in", "BINDING 4", "BINDING 8"),
            ("CMakeLists.txt", "behavior 10", "behavior 11"),
        ]:
            with self.subTest(name=name):
                path = self.source / name
                original, stat = path.read_bytes(), path.stat()
                path.write_bytes(original.replace(before.encode(), after.encode()))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.prepare()
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute()
                path.write_bytes(original)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.prepare()
                self.execute()

    def test_intervening_preparation_rejects_earlier_pass_after_source_restore(self):
        first = self.prepare()
        self.execute()
        path = self.source / "native.cpp"
        original, stamp = path.read_bytes(), path.stat()
        try:
            path.write_bytes(original.replace(b"return 1", b"return 8"))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.prepare()
        finally:
            path.write_bytes(original)
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        attempts = len(self.campaign.attempts("tests"))
        with OutputLease(self.root, self.build):
            with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                build_evidence.run_ctest(self.campaign, "tests", build=self.build, prepared=first,
                                         inputs=self.inputs(), timeout=30, tests=["behavior"])
        self.assertEqual(len(self.campaign.attempts("tests")), attempts)
        self.prepare()
        self.execute()

    def test_pending_or_missing_preparation_owner_rejects_prior_pass(self):
        first = self.prepare()
        baseline = next(self.build.glob("verification-build-*.json"))
        original = baseline.read_bytes()
        pending = baseline.with_suffix(".pending.json")
        with OutputLease(self.root, self.build):
            try:
                pending.write_text("unfinished preparation")
                with self.assertRaisesRegex(ValueError, "pending preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            finally:
                pending.unlink(missing_ok=True)
            try:
                baseline.unlink()
                with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
                baseline.write_text("unreadable owner")
                with self.assertRaisesRegex(ValueError, "latest completed preparation"):
                    build_evidence.check_scoped_preparation(self.campaign, self.build, first)
            finally:
                baseline.write_bytes(original)
        self.execute()

    def test_restored_timestamp_rust_body_is_observed_through_native_executable(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        self.prepare()
        self.execute()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original, stamp = path.read_bytes(), path.stat()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        self.assertEqual(proof["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        clean = [row["identity"]["command"] for row in self.campaign.attempts("prepare") if Path(row["identity"]["command"][0]).name in {"cargo", "cargo.exe"} and row["identity"]["command"][1] == "clean"]
        self.assertEqual(len(clean), 1)
        self.assertIn("--target", clean[0])
        self.assertEqual(clean[0][clean[0].index("--profile") + 1], "dev")
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        self.execute()

    def test_overlapping_target_preparation_cannot_leave_a_reusable_owner(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        first = self.prepare()
        self.execute()
        path = self.source / "native.cpp"
        original, stamp = path.read_bytes(), path.stat()
        try:
            path.write_bytes(original.replace(b"return 1", b"return 8"))
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.prepare(targets=["overlap"])
        finally:
            path.write_bytes(original)
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "latest completed preparation"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.execute()

    def test_new_scope_restored_clean_rust_uses_shared_archive_inputs(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        self.execute()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original, stamp = path.read_bytes(), path.stat()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare(targets=["overlap"])
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_header_selection_keeps_cold_configured_variants(self):
        from build_dependencies import header_consumers

        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT main.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        targets, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertEqual(sources, ["main.cpp"])
        self.assertEqual(len(targets), 2)
        self.assertTrue(any("extra.dir" in name for name in targets))
        self.prepare(targets=targets)
        self.assertTrue(all((self.build / name).is_file() for name in targets))
        (self.source / "unconsumed.h").write_text("// no consumer\n")
        with self.assertRaisesRegex(ValueError, "without configured consumers"):
            header_consumers(self.source, self.build, ["unconsumed.h"])

    def test_old_nonconsumer_becomes_header_consumer_without_header_edit(self):
        from build_dependencies import header_consumers

        other = self.source / "other.cpp"
        other.write_text("int other(){return 0;}\n")
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT other.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        before, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertNotIn("other.cpp", sources)
        header = (self.source / "detail.h").read_bytes()
        other.write_text('#include "detail.h"\nint other(){return header();}\n')
        after, sources = header_consumers(self.source, self.build, ["detail.h"])
        self.assertIn("other.cpp", sources)
        self.assertGreater(len(after), len(before))
        self.assertEqual((self.source / "detail.h").read_bytes(), header)

    def test_identity_reads_reject_symlink_retarget_and_restored_time_content(self):
        from build_dependencies import file_identities

        first, second, alias = [self.root / name for name in ("first", "second", "alias")]
        for directory, text in [(first, "first"), (second, "other")]:
            directory.mkdir()
            (directory / "input.txt").write_text(text)

        def link_to(target):
            if os.name == "nt":
                subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(alias), str(target)], check=True, capture_output=True, timeout=10)
            else:
                alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)

        def unlink():
            os.rmdir(alias) if os.name == "nt" else alias.unlink()

        link_to(first)
        self.addCleanup(unlink)
        path = alias / "input.txt"
        baseline = file_identities([path])
        traversal = alias / ".." / "first" / "input.txt"
        self.assertEqual(file_identities([traversal])[str(traversal)]["resolved"], str((first / "input.txt").resolve()))
        self.assertEqual(file_identities([self.root / "absent"]), {})
        if os.name != "nt":
            leaf = self.root / "leaf.txt"
            leaf.symlink_to("first/input.txt")
            self.assertEqual(file_identities([leaf])[str(leaf)]["resolved"], str((first / "input.txt").resolve()))
            dangling = self.root / "dangling.txt"
            dangling.symlink_to("absent")
            self.assertEqual(file_identities([dangling]), {})
        stamp = path.stat()
        path.write_text("third")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertNotEqual(file_identities([path]), baseline)
        original = build_evidence.digest
        (second / "input.txt").write_text("third")

        def retarget(value):
            checksum = original(value)
            unlink()
            link_to(second)
            return checksum

        with patch("build_dependencies.digest", side_effect=retarget), self.assertRaisesRegex(ValueError, "input path changed"):
            file_identities([path])
        self.assertEqual(file_identities([path])[str(path)]["resolved"], str((second / "input.txt").resolve()))

    def test_selected_probe_fixture_change_invalidates_pass_without_cleaning_rust(self):
        path = self.source / "probes/reject.rs"
        path.parent.mkdir()
        path.write_text("// compiler assertion fixture\n")
        first = self.prepare()
        path.write_text("// changed compiler assertion\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], [])
        self.execute()

    def test_cargo_configuration_change_invalidates_reuse_and_active_packages(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        first = self.prepare()
        config = self.source / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text("[build]\nrustflags = []\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_macro_candidates_compile_before_consumer_publication(self):
        from build_dependencies import header_consumers

        middle = self.source / "middle.h"
        middle.write_text('#define DETAIL "detail.h"\n#include DETAIL\n')
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'add_library(extra OBJECT main.cpp)\ntarget_include_directories(extra PRIVATE "${CMAKE_BINARY_DIR}/generated")\n')
        self.prepare()
        targets, _ = header_consumers(self.source, self.build, ["detail.h"], require_compiled=False)
        self.assertEqual(len(targets), 2)
        self.prepare(targets=targets)
        self.assertEqual(len(header_consumers(self.source, self.build, ["detail.h"])[0]), 2)
        # A warm restored-time macro edit must discover a previously absent header.
        (self.source / "new.h").write_text('#include "detail.h"\n')
        original, stamp = middle.read_bytes(), middle.stat()
        middle.write_bytes(original.replace(b'"detail.h"', b'"new.h"'))
        os.utime(middle, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "without configured consumers"):
            header_consumers(self.source, self.build, ["new.h"])
        candidates, _ = header_consumers(self.source, self.build, ["new.h"], require_compiled=False)
        self.assertEqual(len(candidates), 2)
        self.prepare(targets=candidates)
        self.assertEqual(header_consumers(self.source, self.build, ["new.h"])[1], ["main.cpp"])

    def test_new_scope_cargo_configuration_invalidates_active_packages(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + "add_custom_target(overlap DEPENDS behavior)\n")
        self.prepare()
        config = self.source / ".cargo/config.toml"
        config.parent.mkdir()
        config.write_text("[build]\nrustflags = []\n")
        self.prepare(targets=["overlap"])
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
        self.execute()

    def test_rust_tool_identity_change_invalidates_active_packages(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        self.prepare()
        inventory = build_evidence.preparation_inventory
        def changed_tools(*args):
            value = inventory(*args)
            for name in value["tools"]:
                if Path(name).name in {"rustc", "rustc.exe"}:
                    value["tools"][name] = "replacement tool identity"
            return value
        with patch.object(build_evidence, "preparation_inventory", side_effect=changed_tools):
            self.prepare()
            self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], ["film-juicer-core", "film-juicer-plugin"])
            self.execute()

    def test_normal_timestamp_rust_edit_uses_incremental_cargo(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        self.prepare()
        path = self.source / "rust/film-juicer-core/src/lib.rs"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"{3}", b"{8}"))
        self.prepare()
        self.assertEqual(json.loads(self.prepared.read_text())["scoped_preparation"]["rust_invalidated_packages"], [])
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        path.write_bytes(original)
        self.prepare()
        self.execute()

    def test_unchanged_repeat_reuses_objects_and_archive_without_touching(self):
        self.args[self.args.index("-DENABLE_RUST=OFF")] = "-DENABLE_RUST=ON"
        self.prepare()
        record = json.loads(self.prepared.read_text())
        paths = [Path(name) for name in record["scoped_preparation"]["outputs"]]
        before = {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths}
        self.prepare()
        after = {str(path): (digest(path), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(
            json.loads(self.prepared.read_text())["scoped_preparation"]["invalidated"], []
        )
        # A missing successful-Cargo stamp must preserve the unchanged archive
        # and native consumers; order-only prerequisites do not carry data.
        (self.build / "cargo-build.stamp").unlink()
        self.prepare()
        repeated = {
            str(path): (digest(path), path.stat().st_mtime_ns)
            for path in paths
            if path.name != "cargo-build.stamp"
        }
        self.assertEqual(
            {
                name: value
                for name, value in before.items()
                if Path(name).name != "cargo-build.stamp"
            },
            repeated,
        )
        self.execute()

    def test_missing_preparation_owner_rebuilds_existing_outputs(self):
        self.prepare()
        owners = list(self.build.glob("verification-build-*.json"))
        self.assertTrue(owners)
        for path in owners:
            path.unlink()
        binary = self.build / ("behavior.exe" if os.name == "nt" else "behavior")
        binary.write_bytes(b"stale executable")
        self.prepare()
        self.execute()

    def test_complete_proof_preserves_unrelated_objects_for_selected_target(self):
        self.prepare()
        record = json.loads(self.prepared.read_text())
        objects = {
            Path(name): (digest(Path(name)), Path(name).stat().st_mtime_ns)
            for name in record["scoped_preparation"]["outputs"]
            if "main.cpp." in name and name.endswith((".o", ".obj"))
        }
        self.assertTrue(objects)
        path = self.source / "native.cpp"
        stamp = path.stat()
        path.write_bytes(path.read_bytes().replace(b"return 1", b"return 8"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare(targets=["behavior"])
        self.assertEqual(
            objects, {path: (digest(path), path.stat().st_mtime_ns) for path in objects}
        )
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(tests=["behavior"])
        failure = self.campaign.attempts("tests")[-1]
        output = (self.root / failure["output"]).read_text()
        self.assertIn("value=17", output)
        self.assertNotIn("RegularExpression::compile", output)

    def test_changed_compile_configuration_cannot_select_old_dependency_logs(self):
        first = self.prepare()
        with OutputLease(self.root, self.build):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        path = self.source / "CMakeLists.txt"
        path.write_text(path.read_text() + "set(CMAKE_CXX_FLAGS -DNEW_VARIANT)\n")
        with OutputLease(self.root, self.build), self.assertRaisesRegex(ValueError, "dependency contents"):
            build_evidence.check_scoped_preparation(self.campaign, self.build, first)
        self.prepare()
        self.execute()

    def test_warm_failed_build_resumes_without_recompiling_native(self):
        script = self.source / "fail_once.py"
        script.write_text(
            "from pathlib import Path\nimport sys\np=Path(sys.argv[1])\nif not p.exists():\n p.write_text('failed')\n raise SystemExit(7)\n"
        )
        with (self.source / "CMakeLists.txt").open("a") as stream:
            stream.write(
                'add_custom_target(fail_once ALL COMMAND "'
                + sys.executable.replace("\\", "/")
                + '" "${CMAKE_SOURCE_DIR}/fail_once.py" "${CMAKE_BINARY_DIR}/failed-once" DEPENDS behavior)\n'
            )
        self.build.mkdir()
        marker = self.build / "failed-once"
        marker.write_text("warm setup")
        self.prepare()
        record = json.loads(self.prepared.read_text())
        outputs = {
            Path(name): (digest(Path(name)), Path(name).stat().st_mtime_ns)
            for name in record["scoped_preparation"]["outputs"]
        }
        marker.unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        failure = self.campaign.attempts("prepare")[-1]
        failed = self.root / Path(failure["output"]).with_name("result.json")
        retained = {
            path: path.read_bytes()
            for path in failed.parent.iterdir()
            if path.is_file()
        }
        self.prepare(resume=failed)
        self.assertEqual(
            outputs, {path: (digest(path), path.stat().st_mtime_ns) for path in outputs}
        )
        self.assertTrue(
            all(path.read_bytes() == value for path, value in retained.items())
        )
        self.execute()

    def test_failed_compile_preserves_baseline_until_a_repaired_build(self):
        self.prepare()
        self.execute()
        baseline = next(self.build.glob("verification-build-*.json"))
        saved = baseline.read_bytes()
        path = self.source / "native.cpp"
        main = self.source / "main.cpp"
        original, stamp = path.read_bytes(), path.stat()
        main_original, main_stamp = main.read_bytes(), main.stat()
        path.write_text("this cannot compile\n")
        main.write_bytes(main_original.replace(b"native()+header()", b"native()+7+header()"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), saved)
        self.assertTrue(list(self.build.glob("verification-build-*.pending.json")))
        path.write_bytes(original)
        main.write_bytes(main_original)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        os.utime(main, ns=(main_stamp.st_atime_ns, main_stamp.st_mtime_ns))
        self.prepare()
        self.assertFalse(list(self.build.glob("verification-build-*.pending.json")))
        self.execute()

    def test_mutation_does_not_advance_successful_baseline(self):
        self.prepare()
        baseline = next(self.build.glob("verification-build-*.json"))
        original = baseline.read_bytes()
        run = self.campaign.run

        def mutate(stage, command, **kwargs):
            result = run(stage, command, **kwargs)
            if "--build" in command:
                (self.source / "native.cpp").write_text("int native(){return 8;}\n")
            return result

        with (
            patch.object(self.campaign, "run", mutate),
            self.assertRaisesRegex(ValueError, "mutated"),
        ):
            self.prepare()
        self.assertEqual(baseline.read_bytes(), original)

    def test_preparation_publication_rejects_changed_process_proof_or_missing_output(self):
        write = build_evidence.write_json
        for fault in ("process", "output"):
            with self.subTest(fault=fault):
                self.prepare()
                baseline = next(path for path in self.build.glob("verification-build-*.json") if not path.name.endswith(".pending.json"))
                original = baseline.read_bytes()

                def corrupt(path, document):
                    write(path, document)
                    if "scoped_preparation" in document and document.get("status") == "passed":
                        if fault == "process":
                            write(Path(document["process_tree"]["receipt"]), {"drained": False})
                        else:
                            output = next(Path(name) for name in document["scoped_preparation"]["outputs"] if name.endswith((".o", ".obj")))
                            output.unlink()

                with patch.object(build_evidence, "write_json", side_effect=corrupt), self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(baseline.read_bytes(), original)
                self.assertTrue(list(self.build.glob("verification-build-*.pending.json")))
        self.prepare()
        self.execute()

    def test_symbolic_utility_preserves_real_output_checks(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() +
                         'add_custom_target(stage COMMAND "${CMAKE_COMMAND}" -E copy_if_different '
                         '"${CMAKE_CURRENT_SOURCE_DIR}/native.cpp" "${CMAKE_BINARY_DIR}/stage-data" '
                         'DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/native.cpp" '
                         'BYPRODUCTS "${CMAKE_BINARY_DIR}/stage-data")\nadd_dependencies(behavior stage)\n')
        self.prepare()
        proof = json.loads(self.prepared.read_text())["scoped_preparation"]
        symbolic = self.build / "CMakeFiles/stage"
        self.assertFalse(symbolic.exists())
        self.assertNotIn(str(symbolic), proof["outputs"])
        real = self.build / "stage-data"
        self.assertIn(str(real), proof["outputs"])
        self.assertEqual(real.read_bytes(), (self.source / "native.cpp").read_bytes())
        self.execute()
        for path in [next(Path(name) for name in proof["outputs"] if name.endswith((".o", ".obj"))), real]:
            path.unlink()
            with self.assertRaises(ValueError):
                self.execute()
            self.prepare()
            self.execute()

    def test_fetched_header_content_invalidates_native_value(self):
        # CMake FetchContent sources live in _deps; they are content-bound
        # inputs, unlike undeclared generated headers elsewhere in the tree.
        header = self.build / "_deps/fetched-src/value.h"
        header.parent.mkdir(parents=True)
        header.write_text("#define FETCHED_VALUE 1\n")
        (self.source / "native.cpp").write_text('#include "value.h"\nint native(){return FETCHED_VALUE;}\n')
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text() + 'target_include_directories(behavior PRIVATE "${CMAKE_BINARY_DIR}/_deps/fetched-src")\n')
        self.prepare()
        self.execute()
        stamp = header.stat()
        header.write_text("#define FETCHED_VALUE 8\n")
        os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "dependency contents or membership changed"):
            self.execute()
        self.prepare()
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        header.write_text("#define FETCHED_VALUE 1\n")
        os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.prepare()
        self.execute()

    def test_generated_shadow_header_addition_cannot_reuse_old_executable(self):
        self.prepare()
        shadow = self.build / "shadow/binding.h"
        shadow.parent.mkdir(exist_ok=True)
        shadow.write_text("#define BINDING 8\n")
        with self.assertRaisesRegex(
            ValueError, "dependency contents or membership changed"
        ):
            self.execute()
        with self.assertRaisesRegex(ValueError, "unresolved generated build input"):
            self.prepare()
        shadow.unlink()
        self.prepare()
        self.execute()

    def test_partial_build_cannot_supply_unfiltered_qualification(self):
        self.prepare(targets=["behavior"])
        with self.assertRaisesRegex(ValueError, "legacy sealed-build"):
            self.execute(tests=[])

    def test_output_lease_spans_readers_and_conflicting_callers(self):
        with OutputLease(self.root, self.build):
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "JUICER_OUTPUT_LEASES"
            }
            command = [
                sys.executable,
                "-B",
                "-c",
                "import sys;sys.path.insert(0,sys.argv[1]);from execution import OutputLease;from pathlib import Path;"
                "with_lease=OutputLease(Path(sys.argv[2]),Path(sys.argv[3]),timeout=.2);with_lease.__enter__()",
                str(ROOT / "scripts"),
                str(self.root),
                str(self.build),
            ]
            result = subprocess.run(
                command, env=environment, capture_output=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"output remains owned", result.stderr)
        with OutputLease(self.root, self.build):
            pass


if __name__ == "__main__":
    unittest.main()
