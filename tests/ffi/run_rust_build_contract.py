"""Exercise the native production and test-support Rust build contracts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RustBuild(unittest.TestCase):
    def test_debug_two_variant_build_contract(self):
        self.subtest_cases = self.required_cases
        self.exercise("Debug")

    def test_release_two_variant_build_contract(self):
        self.subtest_cases = self.required_cases
        self.exercise("Release")

    @property
    def required_cases(self):
        return (
            "configure/off-production-only",
            "build/off-production-consumer",
            "configure/on-add-test-support",
            "build/on-preserves-production",
            "build/test-support-consumer",
            "build/noop",
            "build/support-input-isolation",
            "link/production-facade-absent",
            "configure/mixed-variants-rejected",
            "recover/production-stamp-and-byproduct",
            "recover/test-support-stamp-and-byproduct",
            "change/common-rust-producer",
            "change/rust-test-only-source",
            "change/rust-source-glob-membership",
            "change/native-header-consumer",
            "failure/production-cargo",
            "recover/production-cargo",
            "failure/test-support-cargo",
            "recover/test-support-cargo",
        )

    def exercise(self, configuration):
        options = OPTIONS
        artifact = Path(tempfile.mkdtemp(prefix=configuration.lower() + "-", dir=options.artifacts))
        source = artifact / "source"
        build = artifact / "build"
        core = source / "rust/film-juicer-core"
        plugin = source / "rust/film-juicer-plugin"
        (core / "src").mkdir(parents=True)
        (plugin / "src").mkdir(parents=True)
        (source / "Cargo.toml").write_text(
            '[workspace]\nmembers=["rust/film-juicer-core","rust/film-juicer-plugin"]\nresolver="3"\n')
        (source / "Cargo.lock").write_text(
            'version = 4\n[[package]]\nname="film-juicer-core"\nversion="0.1.0"\n'
            '[[package]]\nname="film-juicer-plugin"\nversion="0.1.0"\ndependencies=["film-juicer-core"]\n')
        for name in ("clippy.toml", "rustfmt.toml"):
            (source / name).write_text("")
        (source / "rust-toolchain.toml").write_text('[toolchain]\nchannel="1.98.1"\nprofile="minimal"\n')
        (core / "Cargo.toml").write_text(
            '[package]\nname="film-juicer-core"\nversion="0.1.0"\nedition="2024"\n'
            '[features]\ntest-support=[]\n')
        (plugin / "Cargo.toml").write_text(
            '[package]\nname="film-juicer-plugin"\nversion="0.1.0"\nedition="2024"\n'
            '[lib]\ncrate-type=["staticlib"]\n[features]\ntest-support=["film-juicer-core/test-support"]\n'
            '[dependencies]\nfilm-juicer-core={path="../film-juicer-core"}\n')
        original = '#[cfg(test)] mod unit;\npub fn value() -> u32 { 7 }\n'
        (core / "src/lib.rs").write_text(original)
        unit = core / "src/unit.rs"
        unit.write_text('#[test] fn value() { assert_eq!(super::value(), 7); }\n')
        (plugin / "src/lib.rs").write_text(
            '#[unsafe(no_mangle)] pub extern "C" fn fj_fixture_value() -> u32 { film_juicer_core::value() }\n'
            '#[cfg(feature="test-support")] #[unsafe(no_mangle)] pub extern "C" fn fj_fixture_test() -> u32 { 1 }\n')
        (source / "fixture.h").write_text(
            "#ifndef FIXTURE_H\n#define FIXTURE_H\n#include <stdint.h>\n"
            "uint32_t fj_fixture_value(void);\nuint32_t fj_fixture_test(void);\n"
            "uint32_t fixture_native_value(void);\n#endif\n")
        (source / "native_shared.c").write_text(
            '#include "fixture.h"\nuint32_t fixture_native_value(void) { return fj_fixture_value(); }\n')
        (source / "consumer.c").write_text(
            '#include <stdio.h>\n#include "fixture.h"\n'
            'int main(void) { printf("%u\\n", (unsigned)fixture_native_value()); return 0; }\n')
        (source / "test_consumer.c").write_text(
            '#include <stdio.h>\n#include "fixture.h"\n'
            'int main(void) { printf("%u %u\\n", (unsigned)fixture_native_value(), '
            '(unsigned)fj_fixture_test()); return 0; }\n')
        (source / "forbidden_consumer.c").write_text(
            '#include "fixture.h"\nint main(void) { return (int)fj_fixture_test(); }\n')

        def write_cmake(path, mixed=False):
            lines = [
                "cmake_minimum_required(VERSION 3.28)",
                "project(RustBuildContract LANGUAGES C)",
                f'include("{options.module.as_posix()}")',
                "add_library(native_shared OBJECT native_shared.c)",
                "target_include_directories(native_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})",
                "set_target_properties(native_shared PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)",
                "add_executable(production_consumer consumer.c $<TARGET_OBJECTS:native_shared>)",
                "target_include_directories(production_consumer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})",
                "set_target_properties(production_consumer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)",
                "target_link_libraries(production_consumer PRIVATE film_juicer_rust)",
                "add_executable(production_facade_rejection forbidden_consumer.c)",
                "target_include_directories(production_facade_rejection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})",
                "set_target_properties(production_facade_rejection PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)",
                "target_link_libraries(production_facade_rejection PRIVATE film_juicer_rust)",
            ]
            if mixed:
                lines.extend((
                    "add_executable(mixed_consumer consumer.c)",
                    "target_link_libraries(mixed_consumer PRIVATE film_juicer_rust film_juicer_rust_test_support)",
                ))
            else:
                lines.extend((
                    "if(BUILD_TESTING)",
                    "  add_executable(test_consumer test_consumer.c $<TARGET_OBJECTS:native_shared>)",
                    "  target_include_directories(test_consumer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})",
                    "  set_target_properties(test_consumer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)",
                    "  target_link_libraries(test_consumer PRIVATE film_juicer_rust_test_support)",
                    "endif()",
                ))
            (path / "CMakeLists.txt").write_text("\n".join(lines) + "\n")

        write_cmake(source)
        commands = []

        def run(command, *, success=True, build_dir=build):
            process = subprocess.run(command, capture_output=True, text=True)
            log = artifact / f"command-{len(commands):03}.log"
            with log.open("x", encoding="utf-8") as stream:
                stream.write(process.stdout + process.stderr)
            commands.append({"command": command, "returncode": process.returncode, "log": str(log)})
            (artifact / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            output = process.stdout + process.stderr
            if success:
                self.assertEqual(process.returncode, 0, log.read_text())
            else:
                self.assertNotEqual(process.returncode, 0, log.read_text())
            return output

        def configure(source_dir, build_dir, enabled):
            run([options.cmake, "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=" + configuration, "-DBUILD_TESTING=" + enabled,
                 "-DCMAKE_MAKE_PROGRAM=" + options.ninja, "-DCMAKE_C_COMPILER=" + options.cc,
                 "-DJUICER_CARGO_EXECUTABLE=" + options.cargo], build_dir=build_dir)

        def build_target(target, *, success=True):
            return run([options.cmake, "--build", str(build), "--target", target, "--parallel", "2"],
                       success=success)

        target = "x86_64-pc-windows-msvc" if os.name == "nt" else "x86_64-unknown-linux-gnu"
        archive_name = "film_juicer_plugin.lib" if os.name == "nt" else "libfilm_juicer_plugin.a"
        profile = configuration.lower()
        production_dir = build / "cargo/production"
        support_dir = build / "cargo/test-support"
        production_archive = production_dir / target / profile / archive_name
        support_archive = support_dir / target / profile / archive_name
        production_stamp = production_dir / "build.stamp"
        support_stamp = support_dir / "build.stamp"
        production_features = production_dir / "features.txt"
        support_features = support_dir / "features.txt"
        production_tag = production_dir / "CACHEDIR.TAG"
        support_tag = support_dir / "CACHEDIR.TAG"
        production_consumer = build / ("production_consumer.exe" if os.name == "nt" else "production_consumer")
        test_consumer = build / ("test_consumer.exe" if os.name == "nt" else "test_consumer")

        def latest_row(build_dir, output_name):
            rows = [line.split("\t") for line in (build_dir / ".ninja_log").read_text().splitlines()]
            matches = [row for row in rows if len(row) == 5 and row[3].replace("\\", "/").endswith(output_name)]
            self.assertTrue(matches, f"Ninja has no completed row for {output_name}")
            return matches[-1]

        def native_object():
            suffix = ".obj" if os.name == "nt" else ".o"
            matches = list(build.glob(f"CMakeFiles/native_shared.dir/native_shared.c{suffix}"))
            if not matches:
                matches = list(build.rglob(f"native_shared.c{suffix}"))
            self.assertEqual(len(matches), 1, f"expected one shared native object, found {matches}")
            return matches[0]

        def variant_state(name):
            archive = production_archive if name == "production" else support_archive
            stamp = production_stamp if name == "production" else support_stamp
            consumer = production_consumer if name == "production" else test_consumer
            return {
                "archive": digest(archive),
                "archive_mtime": archive.stat().st_mtime_ns,
                "stamp_mtime": stamp.stat().st_mtime_ns,
                "link": latest_row(build, consumer.name),
            }

        def native_state():
            obj = native_object()
            return {"digest": digest(obj), "mtime": obj.stat().st_mtime_ns,
                    "compile": latest_row(build, obj.name)}

        def feature_value(path):
            values = [line.partition("=")[2] for line in path.read_text().splitlines()
                      if line.startswith("features=")]
            self.assertEqual(len(values), 1, f"{path} must declare one feature selection")
            return values[0]

        def subcase(case_id):
            return self.subTest(case=case_id)

        with subcase("configure/off-production-only"):
            configure(source, build, "OFF")
            self.assertEqual(production_tag.read_text(),
                             "Signature: 8a477f597d28d172789f06886806bc55\n")
            self.assertFalse(support_dir.exists(), "OFF configure must not create test-support output")

        with subcase("build/off-production-consumer"):
            build_target("production_consumer")
            self.assertEqual(feature_value(production_features), "none")
            self.assertTrue(production_archive.is_file())
            self.assertEqual(run([str(production_consumer)]).strip(), "7")
            production_before_on = variant_state("production")
            native_before_on = native_state()

        with subcase("configure/on-add-test-support"):
            configure(source, build, "ON")
            self.assertEqual(production_tag.read_text(),
                             "Signature: 8a477f597d28d172789f06886806bc55\n")
            self.assertEqual(support_tag.read_text(),
                             "Signature: 8a477f597d28d172789f06886806bc55\n")
            self.assertEqual(feature_value(production_features), "none")
            self.assertEqual(feature_value(support_features), "test-support")
            build_target("production_consumer")
            self.assertEqual(variant_state("production"), production_before_on,
                             "enabling tests must not replace or relink production")

        with subcase("build/on-preserves-production"):
            build_target("juicer_rust_production_build")
            self.assertEqual(variant_state("production"), production_before_on)
            self.assertEqual(native_state(), native_before_on)

        with subcase("build/test-support-consumer"):
            build_target("test_consumer")
            self.assertEqual(feature_value(support_features), "test-support")
            self.assertEqual(run([str(test_consumer)]).strip(), "7 1")
            self.assertNotEqual(production_archive.resolve(), support_archive.resolve())
            production_after_support = variant_state("production")
            support_initial = variant_state("test-support")

        with subcase("build/noop"):
            native_before = native_state()
            build_target("production_consumer")
            build_target("test_consumer")
            self.assertEqual(variant_state("production"), production_after_support)
            self.assertEqual(variant_state("test-support"), support_initial)
            self.assertEqual(native_state(), native_before)

        with subcase("build/support-input-isolation"):
            production_before = variant_state("production")
            support_before = variant_state("test-support")
            support_stamp_before = support_before["stamp_mtime"]
            support_features.write_bytes(support_features.read_bytes())
            build_target("test_consumer")
            self.assertEqual(variant_state("production"), production_before,
                             "a test-support input change must not touch production")
            self.assertGreater(variant_state("test-support")["stamp_mtime"], support_stamp_before,
                               "a test-support input change must rebuild only its variant")
            self.assertEqual(variant_state("test-support")["archive"], support_before["archive"])
            self.assertEqual(variant_state("test-support")["link"], support_before["link"],
                             "unchanged test-support archive must not relink its consumer")

        with subcase("link/production-facade-absent"):
            output = build_target("production_facade_rejection", success=False)
            self.assertIn("fj_fixture_test", output,
                          "production archive unexpectedly supplied the facade-only symbol")

        with subcase("configure/mixed-variants-rejected"):
            mixed_source = artifact / "mixed-source"
            mixed_build = artifact / "mixed-build"
            mixed_source.mkdir()
            for path in ("Cargo.toml", "Cargo.lock", "clippy.toml", "rustfmt.toml", "rust-toolchain.toml"):
                shutil.copy2(source / path, mixed_source / path)
            shutil.copytree(source / "rust", mixed_source / "rust")
            for path in ("consumer.c", "native_shared.c", "fixture.h", "forbidden_consumer.c"):
                shutil.copy2(source / path, mixed_source / path)
            write_cmake(mixed_source, mixed=True)
            output = run([options.cmake, "-S", str(mixed_source), "-B", str(mixed_build), "-G", "Ninja",
                          "-DCMAKE_BUILD_TYPE=" + configuration, "-DBUILD_TESTING=ON",
                          "-DCMAKE_MAKE_PROGRAM=" + options.ninja, "-DCMAKE_C_COMPILER=" + options.cc,
                          "-DJUICER_CARGO_EXECUTABLE=" + options.cargo], success=False)
            self.assertIn("JUICER_RUST_VARIANT", output)

        with subcase("recover/production-stamp-and-byproduct"):
            production_before = variant_state("production")
            legacy_stamp = build / "cargo-build.stamp"
            legacy_stamp.write_text("obsolete single-archive stamp\n")
            production_stamp.unlink()
            build_target("production_consumer")
            self.assertGreater(production_stamp.stat().st_mtime_ns, production_before["stamp_mtime"])
            self.assertEqual(variant_state("production")["link"], production_before["link"])
            production_archive.unlink()
            build_target("production_consumer")
            self.assertTrue(production_archive.is_file(), "old aggregate stamp must not mask the production byproduct")
            self.assertEqual(run([str(production_consumer)]).strip(), "7")

        with subcase("recover/test-support-stamp-and-byproduct"):
            support_before = variant_state("test-support")
            support_stamp.unlink()
            build_target("test_consumer")
            self.assertGreater(support_stamp.stat().st_mtime_ns, support_before["stamp_mtime"])
            self.assertEqual(variant_state("test-support")["link"], support_before["link"])
            support_archive.unlink()
            build_target("test_consumer")
            self.assertTrue(support_archive.is_file(), "missing test-support byproduct must be recovered")
            self.assertEqual(run([str(test_consumer)]).strip(), "7 1")

        with subcase("change/common-rust-producer"):
            production_before = variant_state("production")
            support_before = variant_state("test-support")
            native_before = native_state()
            (core / "src/lib.rs").write_text(original.replace("{ 7 }", "{ 9 }"))
            build_target("production_consumer")
            build_target("test_consumer")
            production_after = variant_state("production")
            support_after = variant_state("test-support")
            self.assertNotEqual(production_after["archive"], production_before["archive"])
            self.assertNotEqual(support_after["archive"], support_before["archive"])
            self.assertNotEqual(production_after["link"], production_before["link"])
            self.assertNotEqual(support_after["link"], support_before["link"])
            self.assertEqual(native_state(), native_before, "Rust producer edits must not rebuild native objects")
            self.assertEqual(run([str(production_consumer)]).strip(), "9")
            self.assertEqual(run([str(test_consumer)]).strip(), "9 1")

        with subcase("change/rust-test-only-source"):
            unit.write_text(unit.read_text() + "// Changed cfg(test)-only source.\n")
            build_target("production_consumer")
            build_target("test_consumer")
            # Debug archive bytes can change with Rust's source fingerprint even
            # when this cfg(test)-only module is not part of either product build.
            # Keep this case focused on executable behavior and variant features.
            self.assertEqual(feature_value(production_features), "none")
            self.assertEqual(feature_value(support_features), "test-support")
            self.assertEqual(run([str(production_consumer)]).strip(), "9")
            self.assertEqual(run([str(test_consumer)]).strip(), "9 1")

        with subcase("change/rust-source-glob-membership"):
            extra = core / "src/unused.rs"
            production_before = variant_state("production")
            support_before = variant_state("test-support")
            extra.write_text("// New glob member.\n")
            build_target("production_consumer")
            build_target("test_consumer")
            self.assertEqual(variant_state("production")["archive"], production_before["archive"])
            self.assertEqual(variant_state("test-support")["archive"], support_before["archive"])
            extra.unlink()
            build_target("production_consumer")
            build_target("test_consumer")
            self.assertEqual(run([str(production_consumer)]).strip(), "9")

        with subcase("change/native-header-consumer"):
            production_before = variant_state("production")
            support_before = variant_state("test-support")
            native_before = native_state()
            header = source / "fixture.h"
            header.write_text(header.read_text() + "/* header dependency changed */\n")
            build_target("production_consumer")
            build_target("test_consumer")
            self.assertEqual(variant_state("production")["archive"], production_before["archive"])
            self.assertEqual(variant_state("test-support")["archive"], support_before["archive"])
            self.assertNotEqual(native_state()["compile"], native_before["compile"])
            self.assertNotEqual(variant_state("production")["link"], production_before["link"])
            self.assertNotEqual(variant_state("test-support")["link"], support_before["link"])

        with subcase("failure/production-cargo"):
            before = variant_state("production")
            valid = (core / "src/lib.rs").read_text()
            broken = valid + '#[cfg(not(feature="test-support"))] compile_error!("production Cargo failure");\n'
            (core / "src/lib.rs").write_text(broken)
            shutil.copyfile(core / "src/lib.rs", artifact / "failed-production-input.rs")
            output = build_target("juicer_rust_production_build", success=False)
            self.assertIn("production Cargo failure", output)
            self.assertEqual(production_stamp.stat().st_mtime_ns, before["stamp_mtime"],
                             "failed production Cargo must not complete its variant stamp")
            (core / "src/lib.rs").write_text(valid)

        with subcase("recover/production-cargo"):
            before = production_stamp.stat().st_mtime_ns
            build_target("production_consumer")
            self.assertGreater(production_stamp.stat().st_mtime_ns, before)
            self.assertEqual(run([str(production_consumer)]).strip(), "9")

        with subcase("failure/test-support-cargo"):
            before = variant_state("test-support")
            valid = (core / "src/lib.rs").read_text()
            broken = valid + '#[cfg(feature="test-support")] compile_error!("test-support Cargo failure");\n'
            (core / "src/lib.rs").write_text(broken)
            shutil.copyfile(core / "src/lib.rs", artifact / "failed-test-support-input.rs")
            output = build_target("juicer_rust_test_support_build", success=False)
            self.assertIn("test-support Cargo failure", output)
            self.assertEqual(support_stamp.stat().st_mtime_ns, before["stamp_mtime"],
                             "failed test-support Cargo must not complete its variant stamp")
            (core / "src/lib.rs").write_text(valid)

        with subcase("recover/test-support-cargo"):
            before = support_stamp.stat().st_mtime_ns
            build_target("test_consumer")
            self.assertGreater(support_stamp.stat().st_mtime_ns, before)
            self.assertEqual(run([str(test_consumer)]).strip(), "9 1")

        (artifact / "result.json").write_text(json.dumps({
            "configuration": configuration,
            "passed": True,
            "required_cases": self.subtest_cases,
            "production": variant_state("production"),
            "test_support": variant_state("test-support"),
            "native": native_state(),
        }, indent=2) + "\n")
        # Keep commands and failed source snapshots; successful fixture trees are disposable.
        shutil.rmtree(build)
        shutil.rmtree(source)
        shutil.rmtree(mixed_build, ignore_errors=True)
        shutil.rmtree(mixed_source, ignore_errors=True)


RustBuild.test_debug_two_variant_build_contract.required_subtests = True
RustBuild.test_release_two_variant_build_contract.required_subtests = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--module", type=Path, default=ROOT / "cmake/JuicerRust.cmake")
    OPTIONS = parser.parse_args()
    OPTIONS.artifacts.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=[__file__], verbosity=2)
