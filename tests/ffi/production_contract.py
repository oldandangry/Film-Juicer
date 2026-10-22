"""Complete testing-OFF Release isolation, C11 values/ABI and OFX loading gate."""

from pathlib import Path
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from execution import digest

root = Path(__file__).resolve().parents[2]
fixtures = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--preset", choices=["linux-release", "windows-clang-release"], required=True
)
parser.add_argument("--artifacts", type=Path, required=True)
arguments = parser.parse_args()
windows = os.name == "nt"
suffix = ".exe" if windows else ""
build = root / "out/build" / arguments.preset
llvm = (
    Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin")
    if windows
    else Path("/usr/lib/llvm-22/bin")
)


def execute(arguments):
    out = arguments.artifacts.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(name, command, timeout=300):
        result = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
        (out / (name + ".stdout")).write_bytes(result.stdout)
        (out / (name + ".stderr")).write_bytes(result.stderr)
        commands.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "stdout_sha256": digest(out / (name + ".stdout")),
                "stderr_sha256": digest(out / (name + ".stderr")),
            }
        )
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result.check_returncode()
        return result.stdout.decode("utf-8", errors="replace")

    build_hashes = {
        str(build / name): digest(build / name)
        for name in ["CMakeCache.txt", "build.ninja", "compile_commands.json", "cargo-features.txt"]
    }

    def check_build_inputs():
        assert {path: digest(Path(path)) for path in build_hashes} == build_hashes, "production build input changed"

    assert "BUILD_TESTING:BOOL=OFF" in (build / "CMakeCache.txt").read_text()
    assert (build / "cargo-features.txt").read_text().strip() == "test-support=OFF"
    compdb = json.loads((build / "compile_commands.json").read_text())
    assert not any(
        (re.search("JUICER_\\w*TEST_HOOK", row["command"]) for row in compdb)
    )
    cargo = [
        line
        for line in (build / "build.ninja").read_text().splitlines()
        if "--package film-juicer-plugin" in line
    ]
    assert cargo and all(
        ("--no-default-features" in line and "--features" not in line for line in cargo)
    )
    check_build_inputs()
    archive = (
        build
        / "cargo"
        / (
            "x86_64-pc-windows-msvc/release/film_juicer_plugin.lib"
            if windows
            else "x86_64-unknown-linux-gnu/release/libfilm_juicer_plugin.a"
        )
    )
    native = build / ("juicer_core.lib" if windows else "libjuicer_core.a")
    module = build / "juicer.ofx"
    build_hashes.update({
        str(path): digest(path)
        for path in [native, archive, module, *([build / "juicer.pdb"] if windows else [])]
    })
    forbidden = re.compile(
        "fj_test_|::test_support::|film_bridge::faults::|film::input::testing::|film_input_bridge::(?:BOUNDARY_FAULT|boundary_fault|arm_boundary_fault)|film::test_support::|\\b(?:FAULT|[A-Z_]+_FAULT|FAULT_[A-Z_]+|[A-Z_]+_FAULT_[A-Z_]+)\\b|\\b[A-Z_]+_FACADE(?:_[A-Z_]+)?\\b|ContextDrainTestFailure|TestSupport::|(?:Profile|Catalog|Terminal|ContextDrain|HostAdmission|ExecutorFailure|PinnedUpload|Illuminant|Render|Spectral|TcLut|Color|OutputGamut|Development|Noise)Test::|\\b\\w*Test::|::fixtures::|::output_fixtures::|::faults::|TC_CONSUMED|TC_TRANSFERS|TC_RELEASES|EXPOSURE_CONSUMED|TC_FACADE_CONSUMED|TC_FACADE_RELEASES|ScannerOracle::|pendingCaptureTest|set_pending_capture_test_hook|COPY_METRICS|SCRATCH_CAPACITIES|gamut_bridge::(?:output_)?fixtures::|exposure_bridge::fixtures::|spectral_bridge::(?:inject|arm_fault|clear_fault)"
    )
    strings = [
        b"film input boundary fixture panic",
        b"film development acquire fixture panic",
        b"film development view fixture panic",
        b"film development release fixture panic",
        b"film development retain fixture panic",
        b"visual grain fixture panic",
        b"physical effects fixture panic",
        b"spatial optics fixture panic",
        b"print illuminant fixture panic",
        b"print preflash fixture panic",
        b"print balance fixture panic",
        b"scanner correction fixture panic",
        b"film DIR acquire fixture panic",
        b"film DIR view fixture panic",
        b"film DIR release fixture panic",
        b"film density bounds fixture panic",
        b"film recipe diagnostic fixture panic",
        b"output gamut boundary fault",
        b"closed operation-specific output gamut fault",
        b"input hull boundary fault",
        b"closed input hull fault",
        b"fixed spectrum fixture fault",
        b"fixed exposure preparation category fault",
        b"direct color facade fault",
        b"exposure production boundary fault",
        b"CAT16 production boundary fault",
        b"color production boundary fault",
        b"illuminant injected panic",
        b"illuminant fixture-only fault",
        b"spectral injected panic",
        b"spectral fixture-only fault",
        b"TC production boundary fault",
        b"TC fixture-only fault",
    ]
    consumed_paths = [root / "native/juicer_legacy_api.h", fixtures / "production-film.c.in",
                      fixtures / "production-output-color.c.in", Path(__file__),
                      out / "production-film-expected.h"]
    for directory in [root / "Resources", root / "tests/film/fixtures",
                      root / "tests/film/input-fixtures", root / "tests/film/dir-fixtures"]:
        consumed_paths.extend(path for path in directory.rglob("*") if path.is_file())
    grain_directory = root / "tests/film/grain-fixtures"
    grain_manifest = grain_directory / "manifest.json"
    grain_payload = grain_directory / json.loads(grain_manifest.read_text())["presets"][arguments.preset]["values"]
    consumed_paths.extend([grain_manifest, grain_payload])
    effects_directory = root / "tests/film/effects-fixtures"
    effects_manifest = effects_directory / "manifest.json"
    effects_spec = json.loads(effects_manifest.read_text())
    effects_payload = effects_directory / effects_spec["presets"][arguments.preset]["literal"]
    assert digest(effects_payload) == effects_spec["payload_sha256"][effects_payload.name]
    consumed_paths.extend([effects_manifest, effects_payload])
    optics_directory = root / "tests/film/optics-fixtures"
    optics_manifest = optics_directory / "manifest.json"
    optics_spec = json.loads(optics_manifest.read_text())
    optics_payload = optics_directory / optics_spec["presets"][arguments.preset]["literal"]
    assert digest(optics_payload) == optics_spec["payload_sha256"][optics_payload.name]
    consumed_paths.extend([optics_manifest, optics_payload])
    print_directory = root / "tests/print/illuminant-fixtures"
    print_manifest = print_directory / "rust-manifest.json"
    print_spec = json.loads(print_manifest.read_text())
    print_payload = print_directory / print_spec["presets"][arguments.preset]["literal"]
    assert digest(print_payload) == print_spec["files"][print_payload.name]["sha256"]
    native_print_manifest = print_directory / print_spec["native_manifest"]["path"]
    assert digest(native_print_manifest) == print_spec["native_manifest"]["sha256"]
    consumed_paths.extend([print_manifest, print_payload, native_print_manifest])
    preflash_manifest, preflash_payload, _, _, _ = preflash_fixture(arguments.preset)
    consumed_paths.extend([preflash_manifest, preflash_payload])
    balance_manifest, balance_payload, _, _, _ = balance_fixture(arguments.preset)
    consumed_paths.extend([balance_manifest, balance_payload])
    correction_manifest, correction_payload, correction_checked, _, _, _ = scanner_correction_fixture(arguments.preset)
    consumed_paths.extend([correction_manifest, correction_payload, correction_checked])
    consumed = {str(path): digest(path) for path in sorted(set(consumed_paths))}

    def check_consumed():
        assert {path: digest(Path(path)) for path in consumed} == consumed, "production input changed"

    scanned = []

    def scan(path, name, tool, flags):
        text = run(name, [str(llvm / (tool + suffix)), *flags, str(path)], 600)
        match = forbidden.search(text)
        assert match is None, (name, match.group(0) if match else None)
        raw = path.read_bytes()
        assert not any((value in raw for value in strings)), name
        assert digest(path) == build_hashes[str(path)], "production build input changed during scan"
        scanned.append(
            {
                "file": str(path),
                "sha256": build_hashes[str(path)],
                "symbols_and_fault_strings_absent": True,
            }
        )
        return text

    scan(native, "native-symbols", "llvm-nm", ["--defined-only", "--demangle"])
    symbols = scan(archive, "rust-symbols", "llvm-nm", ["--defined-only", "--demangle"])
    for name in [
        "fj_legacy_film_input_prepare",
        "fj_legacy_film_development_acquire",
        "fj_legacy_film_development_view",
        "fj_legacy_film_development_release",
        "fj_legacy_film_development_retain",
        "fj_legacy_film_density_minimum_prepare",
        "fj_legacy_film_dir_acquire",
        "fj_legacy_film_dir_view",
        "fj_legacy_film_dir_release",
        "fj_legacy_film_density_bounds_prepare",
        "fj_legacy_visual_grain_prepare",
        "fj_legacy_physical_effects_prepare",
        "fj_legacy_spatial_optics_prepare",
        "fj_legacy_print_illuminant_prepare",
        "fj_legacy_print_preflash_prepare",
        "fj_legacy_print_balance_prepare",
        "fj_legacy_scanner_direct_correction_prepare",
        "fj_legacy_scanner_print_correction_prepare",
        "fj_legacy_scanner_correction_input_demand",
        "fj_legacy_output_color",
        "fj_legacy_film_profile_acquire",
        "fj_legacy_print_profile_acquire",
        "fj_legacy_resolve_route",
    ]:
        assert re.search("\\b" + name + "\\b", symbols), name
    if windows:
        exports = scan(module, "module-exports", "llvm-readobj", ["--coff-exports"])
        module_export_names = re.findall("^\\s*Name: (.+)$", exports, re.M)
        scan(
            build / "juicer.pdb",
            "pdb-symbols",
            "llvm-pdbutil",
            ["dump", "-globals", "-publics"],
        )
    else:
        exports = scan(
            module, "module-exports", "llvm-nm", ["--dynamic", "--defined-only"]
        )
        module_export_names = [line.split()[-1] for line in exports.splitlines() if line.strip()]
        scan(module, "module-symbols", "llvm-nm", ["--defined-only", "--demangle"])
    assert sorted(module_export_names) == ["OfxGetNumberOfPlugins", "OfxGetPlugin"], module_export_names
    probes = [
        ("film", fixtures / "production-film.c.in"),
        ("accepted-output-color", fixtures / "production-output-color.c.in"),
    ]
    probe_records = []
    grain_results = []
    effects_results = []
    optics_results = []
    print_illuminant_results = []
    preflash_results = []
    balance_results = []
    scanner_correction_results = []
    scanner_demand_results = []
    for name, source in probes:
        executable = out / (name + suffix)
        target_source = out / (name + ".c")
        target_source.write_bytes(source.read_bytes())
        source = target_source
        source_sha256 = digest(source)
        if windows:
            command = [
                str(llvm / "clang-cl.exe"),
                "/nologo",
                "/TC",
                "/std:c11",
                "/MD",
                "/W4",
                "/WX",
                "/Inative",
                "/Itests/ffi",
                "/I" + str(out),
                str(source),
                "/Fe:" + str(executable),
                "/Fo:" + str(out / (name + ".obj")),
                "/link",
                str(archive),
                "kernel32.lib",
                "ntdll.lib",
                "userenv.lib",
                "ws2_32.lib",
                "dbghelp.lib",
                "advapi32.lib",
                "/defaultlib:msvcrt",
            ]
        else:
            command = [
                "gcc-13",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Inative",
                "-Itests/ffi",
                "-I" + str(out),
                str(source),
                str(archive),
                "-lgcc_s",
                "-lutil",
                "-lrt",
                "-lpthread",
                "-lm",
                "-ldl",
                "-lc",
                "-o",
                str(executable),
            ]
        check_consumed()
        run(name + "-compile", command, 300)
        linked = digest(executable)
        assert digest(archive) == next(row["sha256"] for row in scanned if row["file"] == str(archive))
        check_consumed()
        before = digest(executable)
        assert before == linked
        output = run(
            name + "-values",
            [str(executable), *([str(root / "Resources"), str(root / "tests/film/fixtures/synthetic/duplicates")] if name == "film" else [])],
            120,
        )
        after = digest(executable)
        assert after == before
        assert digest(source) == source_sha256, "generated probe source changed during execution"
        check_consumed()
        probe_records.append({"name": name, "executable": str(executable),
                              "linked_sha256": linked, "before_run_sha256": before,
                              "after_run_sha256": after, "source": str(source),
                              "source_sha256": source_sha256, "archive": str(archive),
                              "archive_sha256": digest(archive)})
        if name == "film":
            pattern = r"^GRAIN_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) channel=(\d+) sample=(\d+) fields=pass$"
            expected = [("grain-active-layers", 0, 0), ("grain-inactive", 0, 0),
                        ("grain-raw-boolean", 2, 1), ("grain-raw-owner", 2, 1),
                        ("grain-controls-area", 3, 17), ("grain-missing-layers", 3, 18)]
            lines = [line for line in output.splitlines() if line.startswith("GRAIN_CASE")]
            assert len(lines) == len(expected), lines
            for line, (case, category, reason) in zip(lines, expected):
                match = re.fullmatch(pattern, line)
                assert match is not None, line
                actual = [int(value) for value in match.groups()[1:]]
                assert match[1] == case and actual == [category, 0, 0, reason, 3, 2**64 - 1], line
                grain_results.append({"case": case, "category": actual[0], "api": actual[1],
                                      "native_code": actual[2], "reason": actual[3],
                                      "channel": actual[4], "sample": actual[5], "fields": "pass"})
            effects_pattern = r"^EFFECTS_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) channel=(\d+) sample=(\d+) fields=156-pass$"
            case_names = ["effects-active", "effects-inactive", "effects-debug-suppressed",
                     "effects-grain-off-debug", "effects-sanitized", "effects-weave-negative-zero",
                     "effects-weave-subfloat", "effects-raw-boolean", "effects-raw-params"]
            lines = [line for line in output.splitlines() if line.startswith("EFFECTS_CASE")]
            assert len(lines) == 9, lines
            for index, (line, case) in enumerate(zip(lines, case_names)):
                match = re.fullmatch(effects_pattern, line)
                assert match is not None and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == ([0, 0, 0, 0, 3, 2**64 - 1] if index < 7 else [2, 0, 0, 1, 3, 2**64 - 1]), line
                effects_results.append({"case": case, "category": actual[0], "api": actual[1],
                                        "native_code": actual[2], "reason": actual[3],
                                        "channel": actual[4], "sample": actual[5], "fields": "156-pass"})
            optics_pattern = r"^OPTICS_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) component=(\d+) field=(\d+) channel=(\d+) fields=pass$"
            optics_case_names = ["optics-active-print", "optics-active-direct", "optics-inactive",
                     "optics-inactive-negative-zero", "optics-lens-only", "optics-direct-ignored-enlarger",
                     "optics-off-invalid-camera", "optics-family-before-disabled", "optics-print-invalid-enlarger",
                     "optics-camera-before-lens", "optics-lens-before-halation", "optics-halation-total-strength",
                     "optics-diffusion-centers", "optics-raw-boolean", "optics-raw-params"]
            envelopes = [[0, 0, 0, 0, 0, 0, 3]] * 6 + [
                [3, 0, 0, 2, 1, 2, 3], [3, 0, 0, 2, 1, 1, 3], [3, 0, 0, 2, 4, 1, 3],
                [3, 0, 0, 2, 1, 2, 3], [3, 0, 0, 3, 2, 15, 3], [3, 0, 0, 4, 3, 16, 0],
                [3, 0, 0, 2, 1, 13, 3], [2, 0, 0, 1, 0, 0, 3], [2, 0, 0, 1, 0, 0, 3]]
            lines = [line for line in output.splitlines() if line.startswith("OPTICS_CASE")]
            assert len(lines) == 15, lines
            for line, case, expected_envelope in zip(lines, optics_case_names, envelopes):
                match = re.fullmatch(optics_pattern, line)
                assert match is not None and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == expected_envelope, line
                optics_results.append({"case": case, "category": actual[0], "api": actual[1],
                                       "native_code": actual[2], "reason": actual[3],
                                       "component": actual[4], "field": actual[5], "channel": actual[6],
                                       "fields": "pass"})
            print_pattern = r"^PRINT_ILLUMINANT_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) sample=(\d+) fields=81-pass$"
            names = ["print-illum-neutral", "print-illum-main", "print-illum-preflash",
                     "print-illum-negative-zero", "print-illum-sample-negative", "print-illum-sample-nan",
                     "print-illum-sample-infinite", "print-illum-computed-sample", "print-illum-energy-zero",
                     "print-illum-energy-overflow", "print-illum-raw-params", "print-illum-raw-span"]
            envelopes = [[0, 0, 0, 0, 4294967295]] * 4 + [
                [3, 0, 0, 2, 0], [3, 0, 0, 2, 37], [3, 0, 0, 2, 80],
                [3, 0, 0, 2, 24], [3, 0, 0, 3, 4294967295], [3, 0, 0, 3, 4294967295],
                [2, 0, 0, 1, 4294967295], [2, 0, 0, 1, 4294967295]]
            lines = [line for line in output.splitlines() if line.startswith("PRINT_ILLUMINANT_CASE")]
            assert len(lines) == 12, lines
            for line, case, envelope in zip(lines, names, envelopes, strict=True):
                match = re.fullmatch(print_pattern, line)
                assert match is not None and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == envelope, line
                print_illuminant_results.append({"case": case, "category": actual[0], "api": actual[1],
                                                  "native_code": actual[2], "reason": actual[3],
                                                  "sample": actual[4], "fields": "81-pass"})
            _, _, names, _, envelopes = preflash_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("PRINT_PREFLASH_CASE")]
            assert len(lines) == 12, lines
            for line, case, envelope in zip(lines, names, envelopes, strict=True):
                match = re.fullmatch(r"PRINT_PREFLASH_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) channel=(\d+) fields=3-pass", line)
                assert match is not None and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == [envelope[0], 0, 0, envelope[1], envelope[2]], line
                preflash_results.append({"case": case, "category": actual[0], "api": actual[1],
                                        "native_code": actual[2], "reason": actual[3],
                                        "channel": actual[4], "fields": "3-pass"})
            _, _, names, _, envelopes = balance_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("PRINT_BALANCE_CASE")]
            assert len(lines) == 16, lines
            for line, case, envelope in zip(lines, names, envelopes, strict=True):
                match = re.fullmatch(r"PRINT_BALANCE_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) channel=(\d+) fields=11-pass", line)
                assert match is not None and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == [envelope[0], 0, 0, envelope[1], envelope[2]], line
                balance_results.append({"case": case, "category": actual[0], "api": actual[1],
                                        "native_code": actual[2], "reason": actual[3],
                                        "channel": actual[4], "fields": "11-pass"})
            _, _, _, names, rows, envelopes = scanner_correction_fixture(arguments.preset)
            for prefix, results in (("SCANNER_CORRECTION_CASE", scanner_correction_results),
                                    ("SCANNER_DEMAND_CASE", scanner_demand_results)):
                lines = [line for line in output.splitlines() if line.startswith(prefix)]
                assert len(lines) == 28, lines
                for line, case, row, envelope in zip(lines, names, rows, envelopes, strict=True):
                    match = re.fullmatch(prefix + r" (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) fields=pass", line)
                    assert match is not None and match[1] == case, line
                    actual = [int(value) for value in match.groups()[1:]]
                    expected = envelope if prefix == "SCANNER_CORRECTION_CASE" else correction_demand(case, row)[0]
                    assert actual == [expected[0], 0, 0, expected[1]], line
                    results.append({"case": case, "category": actual[0], "api": actual[1],
                                    "native_code": actual[2], "reason": actual[3], "fields": "pass"})
    check_build_inputs()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if windows:
        cuda = Path(os.environ["CUDA_PATH_V13_2"])
        _dll_directories = [
            os.add_dll_directory(str(path))
            for path in [cuda / "bin", cuda / "bin/x64"]
            if path.is_dir()
        ]
    loaded = ctypes.CDLL(str(module))
    loaded.OfxGetNumberOfPlugins.restype = ctypes.c_int
    loaded.OfxGetPlugin.argtypes = [ctypes.c_int]
    loaded.OfxGetPlugin.restype = ctypes.c_void_p
    assert loaded.OfxGetNumberOfPlugins() == 1 and loaded.OfxGetPlugin(0)
    check_consumed()
    retained_build_inputs = {}
    for original, expected in build_hashes.items():
        path = Path(original)
        retained = out / "build" / path.relative_to(build)
        retained.parent.mkdir(parents=True, exist_ok=True)
        assert digest(path) == expected, "production build input changed before retention"
        shutil.copyfile(path, retained)
        assert digest(retained) == expected, "production build input changed during retention"
        retained_build_inputs[original] = {"file": str(retained), "sha256": expected}
    check_consumed()
    check_build_inputs()
    (out / "production.json").write_text(
        json.dumps(
            {
                "preset": arguments.preset,
                "build_testing": "OFF",
                "test_support": "OFF",
                "scanned": scanned,
                "build_inputs": retained_build_inputs,
                "consumed_inputs": consumed,
                "grain_fixture": {"manifest": str(grain_manifest), "payload": str(grain_payload),
                                  "logical_preset": arguments.preset},
                "probes": probe_records,
                "grain_results": grain_results,
                "effects_results": effects_results,
                "optics_results": optics_results,
                "print_illuminant_results": print_illuminant_results,
                "preflash_results": preflash_results,
                "balance_results": balance_results,
                "scanner_correction_results": scanner_correction_results,
                "scanner_demand_results": scanner_demand_results,
                "scanner_correction_fixture": {"manifest": str(correction_manifest), "payload": str(correction_payload),
                                               "normative": str(correction_checked), "logical_preset": arguments.preset},
                "balance_fixture": {"manifest": str(balance_manifest), "payload": str(balance_payload), "logical_preset": arguments.preset},
                "preflash_fixture": {"manifest": str(preflash_manifest), "payload": str(preflash_payload),
                                     "logical_preset": arguments.preset},
                "print_illuminant_fixture": {"manifest": str(print_manifest), "payload": str(print_payload),
                                             "logical_preset": arguments.preset},
                "optics_fixture": {"manifest": str(optics_manifest), "payload": str(optics_payload),
                                   "logical_preset": arguments.preset},
                "effects_fixture": {"manifest": str(effects_manifest), "payload": str(effects_payload),
                                    "logical_preset": arguments.preset},
                "commands": {"file": str(out / "commands.json"), "sha256": digest(out / "commands.json")},
                "command_outputs": {str(out / (row["name"] + suffix)): digest(out / (row["name"] + suffix))
                                    for row in commands for suffix in [".stdout", ".stderr"]},
                "loaded_module": {"file": str(module), "sha256": build_hashes[str(module)]},
                "exports": sorted(module_export_names),
                "hidden_device_load": True,
                "limitation": "CUDA_VISIBLE_DEVICES=-1 checks hidden discovery; an absent installed driver is not established.",
            },
            indent=2,
        )
        + "\n"
    )
    print("PRODUCTION_MANIFEST_SHA256", digest(out / "production.json"), flush=True)
    print(
        "PASS complete OFF film and retained output-color gate, archive/module/PDB isolation, two exports and hidden-device loading",
        flush=True,
    )


def expected_film_header(out):
    fixture = root / "tests/film/fixtures" / (arguments.preset + ".json")
    document = json.loads(fixture.read_text())
    table = document["profiles"]["kodak_portra_400"]
    case = next(
        (
            row
            for row in document["cases"]
            if row["key"] == "kodak_portra_400"
            and row["route"] == 0
            and (row["grain_policy"] == 2)
            and (row["development"]["gamma"][0] == 1065353216)
        )
    )

    def flatten(value):
        return (
            [word for row in value for word in flatten(row)]
            if isinstance(value, list)
            else [value]
        )

    header = ["#define FILM_N " + str(len(table["axis"])) + "U"]
    for name, value in [
        ("axis", table["axis"]),
        ("authored", table["authored"]),
        ("normalized", table["normalized"]),
        ("layers", table["layers"]),
        ("gamma", case["development"]["gamma"]),
        ("minimum", table["minimum"]),
        ("maximum", table["maximum"]),
    ]:
        header.append(
            "static const uint32_t film_"
            + name
            + "[] = {"
            + ",".join((str(word) + "U" for word in flatten(value)))
            + "};"
        )
    for name, value in [
        ("NORMALIZED_HASH", table["normalized_hash"]),
        ("LAYERS_HASH", table["layers_hash"]),
        ("HASH", case["development"]["hash"]),
    ]:
        header.append("#define FILM_" + name + " UINT64_C(" + str(value) + ")")
    input_directory = root / "tests/film/input-fixtures"
    input_manifest = json.loads((input_directory / "manifest.json").read_text())
    input_payload = input_manifest["presets"][arguments.preset]
    input_path = input_directory / input_payload
    assert digest(input_path) == input_manifest["payloads"][input_payload]["sha256"]
    input_document = json.loads(input_path.read_text())
    raw = next(row["raw"] for row in input_document["rows"]
               if row.get("control") is None and row["route"] == 0
               and row["space"] == 0 and row["method"] == 1 and row["decode"] == 0)
    for name in ("rgb_to_xyz", "xyz_adapt", "xyz_to_linear_srgb", "nominal_white", "projection_white", "sensitivity", "mallett_scale"):
        header.append("static const uint32_t film_input_" + name + "[] = {"
                      + ",".join(str(word) + "U" for word in flatten(raw[name])) + "};")
    for name in ("hash", "sensitivity_hash"):
        header.append("#define FILM_INPUT_" + name.upper() + " UINT64_C(" + str(raw[name]) + ")")
    (out / "production-film-expected.h").write_text("\n".join(header) + "\n")
    directory = root / "tests/film/dir-fixtures"
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = directory / manifest["presets"][arguments.preset]["values"]
    assert digest(payload) == manifest["payload_sha256"][payload.name]
    cases = json.loads(payload.read_text())["cases"]
    rows = [next(row for row in cases if row["key"] == "kodak_portra_400"
                 and row["route"] == route and row["kind"] == 2
                 and row["gamma"] == [1065353216]) for route in (0, 1)]
    extra = []
    for name, value in [("dir_matrix", rows[0]["dir"]["matrix"]),
                        ("dir_axes", rows[0]["dir"]["axes"]),
                        ("bounds_minimum", rows[0]["bounds"]["minimum"]),
                        ("bounds_maximum", rows[0]["bounds"]["maximum"]),
                        ("bounds_inverse", rows[0]["bounds"]["inverse"])]:
        extra.append("static const uint32_t film_" + name + "[] = {"
                     + ",".join(str(word) + "U" for word in flatten(value)) + "};")
    for name, value in [("DIR_HASH", rows[0]["dir"]["hash"]),
                        ("DIR_AXES_HASH", rows[0]["dir"]["axes_hash"])]:
        extra.append("#define FILM_" + name + " UINT64_C(" + str(value) + ")")
    extra.append("static const uint64_t film_bounds_hash[] = {"
                 + ",".join("UINT64_C(" + str(row["bounds"]["hash"]) + ")" for row in rows) + "};")
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("\n".join(extra) + "\n")

    grain_directory = root / "tests/film/grain-fixtures"
    grain_manifest = json.loads((grain_directory / "manifest.json").read_text())
    grain_payload = grain_directory / grain_manifest["presets"][arguments.preset]["values"]
    assert digest(grain_payload) == grain_manifest["payload_sha256"][grain_payload.name]
    grain_document = json.loads(grain_payload.read_text())
    selected = [row for row in grain_document["products"]
                if row["key"] == "kodak_portra_400" and row["route"] == 0
                and row["gamma"] == [1065353216] and row["controls"]["active"]
                and row["controls"]["sublayers_active"]]
    assert len(selected) == 1
    grain = selected[0]
    # Constants come solely from the independent native capture; no grain producer runs here.
    with (out / "production-film-expected.h").open("a") as stream:
        for prefix, record in [("grain_params", grain["controls"]), ("grain_recipe", grain["grain"])]:
            for name, value in record.items():
                if name in {"hash", "layers_hash"}:
                    stream.write("#define " + prefix.upper() + "_" + name.upper() + " UINT64_C(" + str(value) + ")\n")
                elif isinstance(value, list):
                    stream.write("static const uint32_t " + prefix + "_" + name + "[] = {"
                                 + ",".join(str(int(word)) + "U" for word in flatten(value)) + "};\n")
                else:
                    stream.write("#define " + prefix.upper() + "_" + name.upper() + " " + str(int(value)) + "\n")

    effects_directory = root / "tests/film/effects-fixtures"
    effects_manifest = json.loads((effects_directory / "manifest.json").read_text())
    effects_payload = effects_directory / effects_manifest["presets"][arguments.preset]["literal"]
    assert digest(effects_payload) == effects_manifest["payload_sha256"][effects_payload.name]
    effects = json.loads(effects_payload.read_text())["cases"]
    names = ["effects-active", "effects-inactive", "effects-debug-suppressed",
             "effects-grain-off-debug", "effects-sanitized", "effects-weave-negative-zero",
             "effects-weave-subfloat", "effects-raw-boolean", "effects-raw-params"]
    rows = [next(row for row in effects if row["id"] == name) for name in names[:7]]
    rows += [{**rows[0], "controls": {**rows[0]["controls"], "grain_active": 2}, "recipe": rows[1]["recipe"]},
             {**rows[0], "recipe": rows[1]["recipe"]}]
    assert len({name for name in names}) == 9 and len(rows) == 9
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* effects_names[9] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        stream.write("static const uint32_t effects_amounts[9][4] = {" + ",".join("{" + ",".join(str(word) + "U" for word in row["controls"]["amounts"]) + "}" for row in rows) + "};\n")
        for prefix, values, ctype in [
            ("effects_weave_input", [row["controls"]["weave"] for row in rows], "uint64_t"),
            ("effects_grain_active", [int(row["controls"]["grain_active"]) for row in rows], "uint32_t"),
            ("effects_debug", [row["controls"]["grain_debug_view"] for row in rows], "int32_t"),
            ("effects_active", [int(row["recipe"]["active"]) for row in rows], "uint32_t"),
            ("effects_weave", [row["recipe"]["weave"] for row in rows], "uint64_t"),
            ("effects_hash", [row["recipe"]["hash"] for row in rows], "uint64_t")]:
            stream.write("static const " + ctype + " " + prefix + "[9] = {" + ",".join(("UINT64_C(" + str(word) + ")") if ctype == "uint64_t" else str(word) for word in values) + "};\n")
        stream.write("static const uint32_t effects_coefficients[9][156] = {" + ",".join("{" + ",".join(str(word) + "U" for family in ["film_dust", "film_scratch", "gate_dust", "gate_scratch"] for word in row["recipe"][family]) + "}" for row in rows) + "};\n")


    optics_directory = root / "tests/film/optics-fixtures"
    optics_manifest = json.loads((optics_directory / "manifest.json").read_text())
    optics_payload = optics_directory / optics_manifest["presets"][arguments.preset]["literal"]
    assert digest(optics_payload) == optics_manifest["payload_sha256"][optics_payload.name]
    captured = json.loads(optics_payload.read_text())["cases"]
    names = ["optics-active-print", "optics-active-direct", "optics-inactive",
             "optics-inactive-negative-zero", "optics-lens-only", "optics-direct-ignored-enlarger",
             "optics-off-invalid-camera", "optics-family-before-disabled", "optics-print-invalid-enlarger",
             "optics-camera-before-lens", "optics-lens-before-halation", "optics-halation-total-strength",
             "optics-diffusion-centers", "optics-raw-boolean", "optics-raw-params"]
    rows = [next(row for row in captured if row["id"] == name) for name in names[:13]]
    # Raw admission clearing is the normative ABI contract; no old partial native failure output is used.
    rows += [{**rows[0], "input": {**rows[0]["input"],
                                 "camera": {**rows[0]["input"]["camera"], "active": 2}}, "success": False},
             {**rows[0], "success": False}]
    arrays = {name: [] for name in ["input_doubles", "input_floats", "input_flags",
                                  "doubles", "floats", "flags", "hashes"]}
    for row in rows:
        v = row["input"]
        arrays["input_doubles"].append(v["camera"]["f64"] + v["enlarger"]["f64"])
        arrays["input_floats"].append([v["lens_f32"]] + v["halation_f32"] + v["first_sigma_f32"] + v["primary_f32"])
        arrays["input_flags"].append([int(v["camera"]["active"]), v["camera"]["family"],
                                     int(v["enlarger"]["active"]), v["enlarger"]["family"],
                                     int(v["halation_active"]), int(v["print"])])
        if row["success"]:
            recipe = row["recipe"]
            c = recipe["camera"]
            e = recipe["enlarger"]
            h = recipe["halation"]
            arrays["doubles"].append(c["f64"] + e["f64"])
            arrays["floats"].append([recipe["lens_f32"]] + h["f32"])
            arrays["flags"].append([c["domain"], int(c["active"]), c["family"], e["domain"],
                                    int(e["active"]), e["family"], int(h["scatter_active"]), int(h["back_reflection_active"])])
            arrays["hashes"].append([c["resolved_hash"], c["hash"], e["resolved_hash"], e["hash"], recipe["lens_hash"], h["hash"], recipe["hash"]])
        else:
            for key, count in [("doubles", 18), ("floats", 10), ("flags", 8), ("hashes", 7)]:
                arrays[key].append([0] * count)
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* optics_names[15] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        for name, data in arrays.items():
            wide = name in {"input_doubles", "doubles", "hashes"}
            def literal(word):
                return "UINT64_C(" + str(word) + ")" if wide else str(word) + "U"
            stream.write("static const " + ("uint64_t" if wide else "uint32_t") + " optics_" + name
                         + "[15][" + str(len(data[0])) + "] = {" + ",".join("{" + ",".join(literal(word) for word in row) + "}" for row in data) + "};\n")

    print_directory = root / "tests/print/illuminant-fixtures"
    print_manifest = json.loads((print_directory / "rust-manifest.json").read_text())
    print_payload = print_directory / print_manifest["presets"][arguments.preset]["literal"]
    assert digest(print_payload) == print_manifest["files"][print_payload.name]["sha256"]
    literals = json.loads(print_payload.read_text())["cases"]
    names = ["print-illum-neutral", "print-illum-main", "print-illum-preflash",
             "print-illum-negative-zero", "print-illum-sample-negative", "print-illum-sample-nan",
             "print-illum-sample-infinite", "print-illum-computed-sample", "print-illum-energy-zero",
             "print-illum-energy-overflow", "print-illum-raw-params", "print-illum-raw-span"]
    rows = []
    for name in names[:10]:
        matched = [row for row in literals if row["id"] == name]
        assert len(matched) == 1, name
        rows.append(matched[0])
    rows += [rows[0], rows[0]]
    # Owner-approved pinned-erff words are frozen separately from native evidence.
    # Header generation evaluates neither Rust nor native mathematics.
    envelopes = [[0, 0, 4294967295]] * 4 + [[3, 2, 0], [3, 2, 37], [3, 2, 80],
                  [3, 2, 24], [3, 3, 4294967295], [3, 3, 4294967295],
                  [2, 1, 4294967295], [2, 1, 4294967295]]
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* print_illuminant_names[12] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        for name, data in [("edges", [row["edges_nm"] for row in rows]),
                           ("transitions", [row["transitions_nm"] for row in rows]),
                           ("cmy", [row["cmy_cc"] for row in rows]),
                           ("source", [row["source"] for row in rows]),
                           ("samples", [row["samples"] if i < 4 else [0] * 81 for i, row in enumerate(rows)]),
                           ("envelopes", envelopes)]:
            stream.write("static const uint32_t print_illuminant_" + name + "[12][" + str(len(data[0])) + "] = {" + ",".join("{" + ",".join(str(word) + "U" for word in row) + "}" for row in data) + "};\n")


def preflash_fixture(preset):
    directory = root / "tests/print/preflash-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    payload = directory / spec["presets"][preset]["native"]
    assert digest(payload) == spec["files"][payload.name]["sha256"]
    literals = json.loads(payload.read_text())["cases"]
    names = ["preflash-unit", "preflash-channels", "preflash-pattern", "preflash-zero",
             "preflash-light-nan", "preflash-sensitivity-skip", "preflash-narrow-overflow",
             "preflash-negative-m", "preflash-nonfinite-y", "preflash-raw-input",
             "preflash-raw-count", "preflash-raw-pointer"]
    rows = []
    for name in names[:9]:
        selected = [row for row in literals if row["id"] == name]
        assert len(selected) == 1, name
        rows.append(selected[0])
    rows += [rows[0]] * 3
    envelopes = [[0, 0, 4294967295] for _ in range(7)] + [[3, 2, 1], [3, 2, 2]] + [[2, 1, 4294967295] for _ in range(3)]
    return manifest, payload, names, rows, envelopes


def expected_preflash_header(out):
    _, _, names, rows, envelopes = preflash_fixture(arguments.preset)
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* preflash_names[12] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        for name, data in [("base", [row["film_base_density"] for row in rows]),
                           ("light", [row["filtered_illuminant"] for row in rows]),
                           ("sensitivity", [row["print_sensitivity_cmy"] for row in rows]),
                           ("raw", [row["raw"] if i < 7 else [0] * 3 for i, row in enumerate(rows)]),
                           ("envelopes", envelopes)]:
            stream.write("static const uint32_t preflash_" + name + "[12][" + str(len(data[0])) + "] = {" + ",".join("{" + ",".join(str(word) + "U" for word in row) + "}" for row in data) + "};\n")


def balance_fixture(preset):
    directory = root / "tests/print/balance-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    selected = spec["presets"][preset]
    payload = directory / selected["native"]
    assert digest(payload) == selected["sha256"]
    literals = json.loads(payload.read_text())["cases"]
    names = ["balance-none", "balance-compensation", "balance-normalize", "balance-both",
             "balance-zero", "balance-signed-zero", "balance-sensitivity-skip",
             "balance-baseline-failure", "balance-compensated-overflow", "balance-unknown-mode",
             "balance-main-hash-failure", "balance-normalizer-overflow", "balance-normalizer-underflow",
             "balance-raw-input", "balance-raw-span", "balance-raw-mode-width"]
    rows = []
    envelopes = []
    for name in names[:13]:
        selected_rows = [row for row in literals if row["id"] == name]
        assert len(selected_rows) == 1, name
        row = selected_rows[0]
        outcome = spec["outcomes"][name]
        assert row["success"] == (outcome["category"] == 0)
        rows.append(row)
        envelopes.append([outcome[key] for key in ("category", "reason", "channel")])
    rows += [rows[0]] * 3
    envelopes += [[2, 1, 4294967295] for _ in range(3)]
    return manifest, payload, names, rows, envelopes


def expected_balance_header(out):
    _, _, names, rows, envelopes = balance_fixture(arguments.preset)
    # Only frozen native words/identities and normative failure clearing enter this header.
    results = [row["balance"]["baseline_raw_rgb"] + row["balance"]["compensated_raw_rgb"] + row["balance"]["scalars"]
               if i < 13 and row["success"] else [0] * 9 for i, row in enumerate(rows)]
    hashes = [[row["balance"]["filtered_main_illuminant_hash"], row["balance"]["hash"]]
              if i < 13 and row["success"] else [0, 0] for i, row in enumerate(rows)]
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* balance_names[16] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        for name, data in [("base", [row["film_base_density"] for row in rows]),
                           ("dye", [row["film_channel_density_cmy"] for row in rows]),
                           ("sensitivity", [row["print_sensitivity_rgb"] for row in rows]),
                           ("light", [row["filtered_main_illuminant"] for row in rows]),
                           ("baseline", [row["baseline_density_cmy"] for row in rows]),
                           ("compensated", [row["compensated_density_cmy"] for row in rows]),
                           ("words", results), ("envelopes", envelopes)]:
            stream.write("static const uint32_t balance_" + name + "[16][" + str(len(data[0])) + "] = {" + ",".join("{" + ",".join(str(word) + "U" for word in row) + "}" for row in data) + "};\n")
        for name, data in [("hashes", hashes), ("reference", [[row["reference_hash"]] for row in rows])]:
            stream.write("static const uint64_t balance_" + name + "[16][" + str(len(data[0])) + "] = {" + ",".join("{" + ",".join("UINT64_C(" + str(word) + ")" for word in row) + "}" for row in data) + "};\n")
        stream.write("static const uint32_t balance_modes[16] = {" + ",".join(str(row["normalization_mode"]) + "U" for row in rows) + "};\n")


def scanner_correction_fixture(preset):
    directory = root / "tests/scanner/correction-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    payload = directory / spec["presets"][preset]["native"]
    checked = directory / spec["normative"]
    for path in (payload, checked):
        assert digest(path) == spec["files"][path.name]["sha256"]
    native = {row["id"]: row for row in json.loads(payload.read_text())["cases"]}
    normative = json.loads(checked.read_text())["cases"]
    # Exact revision-1 membership. Raw-only cases borrow the frozen bounded
    # literal and change only the named wire admission in the C caller.
    membership = [
        ("direct-both", "direct-none"), ("direct-black", "direct-black"),
        ("direct-white", "direct-white"), ("negative-print", "print-none"),
        ("positive-print", "positive-print"), ("print-preflash", "print-preflash"),
        ("direct-off", "direct-route-2-flags-0"), ("print-off", "print-route-1-flags-0"),
        ("negative-direct", "direct-route-0-flags-3"), ("literal-off", "direct-route-255-flags-0"),
        ("direct-route", "direct-route-1-flags-3"), ("print-route", "print-route-2-flags-3"),
        ("direct-reference", "direct-reference"), ("direct-scan", "direct-scan"),
        ("direct-midgray", "direct-midgray"), ("direct-lookup", "direct-direct-lookup"),
        ("print-reference-lookup", "print-print-lookup"), ("print-scan", "print-scan"),
        ("print-midgray", "print-midgray"), ("final-range", "direct-range"),
        ("print-normalizer", "print-normalizer-nan"),
        ("raw-params", "direct-none"), ("raw-span", "direct-none"),
        ("raw-flags", "direct-none"), ("raw-route-width", "direct-none"),
        ("print-literal-off", "print-route-255-flags-0"),
        ("print-byte-route-4", "print-route-4-flags-3"),
        ("print-byte-route-255", "print-route-255-flags-3"),
    ]
    assert len(membership) == 28 and len({name for name, _ in membership}) == 28
    rows = [native[key] for _, key in membership]
    envelopes = [[2, 1] if name.startswith("raw-") else
                 [normative[key]["category"], normative[key]["reason"]]
                 for name, key in membership]
    for (_, key), row, envelope in zip(membership, rows, envelopes, strict=True):
        if envelope[0] == 0:
            assert row["native"]["native_outcome"] == "returned" and row["native"]["native_bool"]
        elif normative[key]["intentional_exception_replacement"]:
            assert row["native"]["native_outcome"] == "exception" and row["native"]["native_bool"] is None
    return manifest, payload, checked, ["correction-" + name for name, _ in membership], rows, envelopes


def correction_demand(name, row):
    if name in {"correction-raw-params", "correction-raw-flags", "correction-raw-route-width"}:
        return [2, 1], 0
    params = row["input"]
    route = params["route"]
    if params["flags"] == 0 or not row["spec"]["print"] and route == 0:
        return [0, 0], 0
    if route in ((1, 3) if row["spec"]["print"] else (2,)):
        return [0, 0], 1
    return [3, 7 if row["spec"]["print"] else 2], 0


def expected_scanner_correction_header(out):
    _, _, _, names, rows, envelopes = scanner_correction_fixture(arguments.preset)
    inputs = [row["input"] for row in rows]
    assert all(row["tables"]["k"] == 1 and not row["tables"]["has_baseline"] for row in inputs)
    results = [row["native"]["descriptor"] if envelope[0] == 0 else
               {"route": 0, "active": 0, "black": 0, "white": 0, "schema": 0, "hash": 0, "words": [0] * 7}
               for row, envelope in zip(rows, envelopes, strict=True)]
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const char* correction_names[28] = {" + ",".join(json.dumps(name) for name in names) + "};\n")
        diagnostics = ["scanner correction raw input" if envelope[0] == 2 else
                       "ResourceDescriptorMismatch phase=8B print correction inputs" if envelope[1] == 7 else
                       row["native"]["diagnostic"] for row, envelope in zip(rows, envelopes, strict=True)]
        stream.write("static const char* correction_diagnostics[28] = {" + ",".join(json.dumps(text) for text in diagnostics) + "};\n")

        def words(name, data):
            width = max(1, max(len(row) for row in data))
            assert len(data) == 28
            stream.write("static const uint32_t correction_" + name + "[28][" + str(width) + "] = {" +
                         ",".join("{" + ",".join(str(int(word)) + "U" for word in (row or [0])) + "}" for row in data) + "};\n")

        words("params", [[row["route"], row["flags"]] for row in inputs])
        words("levels", [row["levels"] for row in inputs])
        for field in ("film_base", "film_dyes", "sensitivity", "light", "print_base", "film_axis", "authored_curves", "print_axis", "print_curves",
                      "minimum", "maximum", "inverse", "baseline_density", "film_minimum", "film_maximum", "preflash"):
            words(field, [row.get(field, []) for row in inputs])
        for field in ("c", "m", "y", "ay"):
            words(field, [row["tables"][field] for row in inputs])
        words("scalars", [[row["tables"]["inv_yn"], row["normalizer"], row["preflash_exposure"]] for row in inputs])
        words("counts", [[len(row.get("film_axis", [])), len(row["authored_curves"]), len(row.get("print_axis", [])), len(row.get("print_curves", []))] for row in inputs])
        words("medium", [[row["medium"]] for row in inputs])
        words("print", [[row["spec"]["print"]] for row in rows])
        words("tags", [[row[key] for key in ("route", "active", "black", "white", "schema")] for row in results])
        words("words", [row["words"] for row in results])
        words("envelopes", envelopes)
        words("demand", [envelope + [required] for envelope, required in (correction_demand(name, row) for name, row in zip(names, rows, strict=True))])
        words("trace", [[1 if result["active"] else 0] if envelope[0] == 0 else
                        [2 if envelope[1] in (5, 10) else 3 if envelope[1] == 12 else 0]
                        for result, envelope in zip(results, envelopes, strict=True)])
        for field, data in (("hashes", [row["hash"] for row in results]),
                            ("reference", [row["reference_hash"] for row in inputs])):
            stream.write("static const uint64_t correction_" + field + "[28] = {" +
                         ",".join("UINT64_C(" + str(word) + ")" for word in data) + "};\n")


if __name__ == "__main__":
    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    expected_film_header(arguments.artifacts)
    expected_preflash_header(arguments.artifacts)
    expected_balance_header(arguments.artifacts)
    expected_scanner_correction_header(arguments.artifacts)
    execute(arguments)
