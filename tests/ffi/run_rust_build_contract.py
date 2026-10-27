"""Run native CMake/Cargo regressions with an isolated consumer and real tools."""

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
    def test_completion_and_archive_propagation(self):
        for configuration in ("Debug", "Release"):
            with self.subTest(configuration=configuration):
                self.exercise(configuration)

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
        (source / "consumer.c").write_text(
            '#include <stdint.h>\n#include <stdio.h>\nextern uint32_t fj_fixture_value(void);\n'
            'int main(void) { printf("%u\\n", (unsigned)fj_fixture_value()); return 0; }\n')
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.28)\nproject(RustBuildContract LANGUAGES C)\n'
            f'include("{options.module.as_posix()}")\n'
            'add_executable(consumer consumer.c)\n'
            'set_target_properties(consumer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)\n'
            'target_link_libraries(consumer PRIVATE film_juicer_rust)\n')
        commands = []

        def run(command, *, success=True):
            process = subprocess.run(command, capture_output=True, text=True)
            log = artifact / f"command-{len(commands):02}.log"
            with log.open("x", encoding="utf-8") as stream:
                stream.write(process.stdout + process.stderr)
            commands.append({"command": command, "returncode": process.returncode, "log": str(log)})
            (artifact / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            if success:
                self.assertEqual(process.returncode, 0, log.read_text())
            else:
                self.assertNotEqual(process.returncode, 0, log.read_text())
            return process.stdout + process.stderr

        def configure(enabled):
            run([options.cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=" + configuration, "-DBUILD_TESTING=" + enabled,
                 "-DCMAKE_MAKE_PROGRAM=" + options.ninja, "-DCMAKE_C_COMPILER=" + options.cc,
                 "-DJUICER_CARGO_EXECUTABLE=" + options.cargo])

        command = [options.cmake, "--build", str(build), "--parallel", "2"]
        target = "x86_64-pc-windows-msvc" if os.name == "nt" else "x86_64-unknown-linux-gnu"
        archive = build / "cargo" / target / configuration.lower() / (
            "film_juicer_plugin.lib" if os.name == "nt" else "libfilm_juicer_plugin.a")
        stamp = build / "cargo-build.stamp"
        consumer = build / ("consumer.exe" if os.name == "nt" else "consumer")

        def state():
            rows = [line.split("\t") for line in (build / ".ninja_log").read_text().splitlines()]
            links = [row for row in rows if len(row) == 5 and row[3] == consumer.name]
            return {"archive": digest(archive), "mtime": archive.stat().st_mtime_ns,
                    "stamp": stamp.stat().st_mtime_ns, "link": links[-1]}

        configure("ON")
        run(command)
        self.assertEqual(run([str(consumer)]).strip(), "7")
        before = state()
        run(command)
        self.assertEqual(state(), before, "warm build must perform no Cargo or native work")

        unit.write_text(unit.read_text() + "// Changed test-only input.\n")
        output = run(command)
        after = state()
        self.assertIn("Building Film-Juicer Rust static library", output)
        self.assertEqual(after["archive"], before["archive"])
        self.assertEqual(after["mtime"], before["mtime"], "test-only edit must not touch the archive")
        self.assertEqual(after["link"], before["link"], "test-only edit must not relink")
        self.assertGreater(after["stamp"], before["stamp"])
        test_only = {"before": before, "after": after}
        run(command)
        self.assertEqual(state(), after, "completion must prevent repeated fresh Cargo dispatch")

        stamp.unlink()
        run(command)
        self.assertEqual(state()["link"], after["link"], "missing stamp must not force native relinking")
        archive.unlink()
        run(command)
        self.assertTrue(archive.is_file(), "missing byproduct must be recovered")
        self.assertEqual(run([str(consumer)]).strip(), "7")

        before = state()
        (core / "src/lib.rs").write_text(original.replace("{ 7 }", "{ 9 }"))
        run(command)
        self.assertNotEqual(state()["archive"], before["archive"])
        self.assertNotEqual(state()["link"], before["link"])
        self.assertEqual(run([str(consumer)]).strip(), "9")

        for enabled in ("OFF", "ON"):
            before = state()
            configure(enabled)
            run(command)
            self.assertNotEqual(state()["archive"], before["archive"])
            self.assertNotEqual(state()["link"], before["link"])
            self.assertEqual((build / "cargo-features.txt").read_text(), "test-support=" + enabled + "\n")
            self.assertEqual(run([str(consumer)]).strip(), "9")

        extra = core / "src/unused.rs"
        extra.write_text("// New glob member.\n")
        before = state()
        run(command)
        self.assertGreater(state()["stamp"], before["stamp"])
        self.assertEqual(state()["link"], before["link"])
        extra.unlink()
        run(command)
        self.assertEqual(state()["link"], before["link"])

        before = state()
        broken = core / "src/lib.rs"
        valid = broken.read_text()
        broken.write_text(valid + 'compile_error!("intentional Cargo failure");\n')
        shutil.copyfile(broken, artifact / "failed-input.rs")
        for _ in range(2):
            self.assertIn("intentional Cargo failure", run(command, success=False))
            self.assertEqual(stamp.stat().st_mtime_ns, before["stamp"], "failed Cargo must not complete preparation")
        broken.write_text(valid)
        run(command)
        self.assertGreater(stamp.stat().st_mtime_ns, before["stamp"])
        self.assertEqual(run([str(consumer)]).strip(), "9")
        (artifact / "result.json").write_text(json.dumps({"configuration": configuration, "passed": True,
            "test_only": test_only, "final": state()}, indent=2) + "\n")
        # Logs and the failing input survive; this isolated compilation cache is disposable.
        shutil.rmtree(build)
        shutil.rmtree(source)


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
