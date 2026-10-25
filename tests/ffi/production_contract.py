"""Complete testing-OFF Release isolation, C11 values/ABI and OFX loading gate."""

from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

from shared_fixture import load_shared_fixture, original_capture_bytes

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
    forbidden = re.compile(forbidden.pattern + r"|reconstruction::testing::|film::reference::testing::")
    forbidden = re.compile(forbidden.pattern + r"|SpectralAllocationProbe::|gaussian_faults::|halation_frame_faults::|diffusion_frame_faults::|dir_frame_faults::|DirFrameObservation|rejectNativeAllocation|rejectedNativeBytes")
    strings = [
        b"spatial DIR frame fixture panic",
        b"DirFrameObservation",
        b"fj_test_dir_frame",
        b"capture-dir-frames",
        b"--dir-frame",
        b"scatter/halation frame fixture panic",
        b"HalationObservation",
        b"DiffusionObservation",
        b"diffusion construction fixture panic",
        b"fj_test_diffusion_",
        b"capture-diffusion-frames",
        b"fj_test_scatter_halation_frame",
        b"capture-halation-frames",
        b"frame geometry fixture panic",
        b"FrameGeometryTest",
        b"--capture-frame-geometry",
        b"--frame-geometry",
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
        b"scanner LUT fixture panic",
        b"scanner LUT descriptor fixture panic",
        b"scanner post-effects fixture panic",
        b"Gaussian kernel fixture panic",
        b"Gaussian kernel fixture rendezvous canceled",
        b"film DIR acquire fixture panic",
        b"film DIR view fixture panic",
        b"film DIR release fixture panic",
        b"film density bounds fixture panic",
        b"film recipe diagnostic fixture panic",
        b"film reference dependency boundary fault",
        b"film reference dependency allocation failed",
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
    reference_manifest, reference_payload, _ = reference_fixture(arguments.preset)
    consumed_paths.extend([reference_manifest, reference_payload])
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
    descriptor_manifest, descriptor_payload, descriptor_context, _, _ = scanner_descriptor_fixture(arguments.preset)
    post_manifest, post_payload, post_context, _, _ = scanner_post_fixture(arguments.preset)
    frame_manifest, frame_payload, frame_context, _ = frame_fixture(arguments.preset)
    consumed_paths.extend([frame_manifest, frame_payload, frame_context, frame_manifest.parent / "inputs.json"])
    halation_manifest, halation_payload, halation_context, halation_checked, _, halation_spec = halation_frame_fixture(arguments.preset)
    consumed_paths.extend([halation_manifest, halation_payload, halation_context, halation_checked, halation_manifest.parent / "inputs.json"])
    diffusion_manifest, diffusion_payload, diffusion_context, diffusion_checked, _, diffusion_spec = diffusion_frame_fixture(arguments.preset)
    consumed_paths.extend([diffusion_manifest, diffusion_payload, diffusion_context, diffusion_checked, diffusion_manifest.parent / "inputs.json"])
    edge_logical = diffusion_spec["presets"][arguments.preset]["edges"]
    diffusion_edges_payload = diffusion_manifest.parent / diffusion_spec["payloads"][edge_logical]
    diffusion_edges_context = diffusion_manifest.parent / diffusion_spec["contexts"][edge_logical]
    diffusion_edges_inputs = diffusion_manifest.parent / "edge-inputs.json"
    consumed_paths.extend([diffusion_edges_payload, diffusion_edges_context, diffusion_edges_inputs])
    dir_manifest, dir_payload, dir_checked, _, dir_spec = dir_frame_fixture(arguments.preset)
    consumed_paths.extend([dir_manifest, dir_payload, dir_checked, dir_manifest.parent / "inputs.json"])
    gaussian_manifest, gaussian_payload, gaussian_context, _, _ = gaussian_fixture(arguments.preset)
    consumed_paths.extend([gaussian_manifest, gaussian_payload, gaussian_context])
    consumed_paths.extend([post_manifest, post_payload, post_context])
    consumed_paths.extend([descriptor_manifest, descriptor_payload, descriptor_context])
    lut_manifest, lut_payload, lut_context, lut_checked, _ = scanner_lut_fixture(arguments.preset)
    consumed_paths.extend([lut_manifest, lut_payload, lut_context, lut_checked, fixtures / "shared_fixture.py",
                           fixtures / "compact_exposure_fixtures.py"])
    correction_manifest, correction_payload, correction_checked, _, _, _ = scanner_correction_fixture(arguments.preset)
    consumed_paths.extend([correction_manifest, correction_payload, correction_checked])
    color_manifest, color_payload, color_checked, _, _ = scanner_color_fixture(arguments.preset)
    consumed_paths.extend([color_manifest, color_payload, color_checked])
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
        "fj_legacy_dir_frame_prepare",
        "fj_legacy_scatter_halation_frame_prepare",
        "fj_legacy_diffusion_sample_prepare",
        "fj_legacy_diffusion_components_prepare",
        "fj_legacy_diffusion_frame_prepare",
        "fj_legacy_film_input_prepare",
        "fj_legacy_film_development_acquire",
        "fj_legacy_film_development_view",
        "fj_legacy_film_development_release",
        "fj_legacy_film_development_retain",
        "fj_legacy_film_density_minimum_prepare",
        "fj_legacy_film_reference_prepare",
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
        "fj_legacy_scanner_color_prepare",
        "fj_legacy_scanner_lut_describe",
        "fj_legacy_scanner_post_describe",
        "fj_legacy_frame_domain_prepare",
        "fj_legacy_frame_geometry_prepare",
        "fj_legacy_gaussian_kernel_acquire",
        "fj_legacy_gaussian_kernel_view",
        "fj_legacy_gaussian_kernel_release",
        "fj_legacy_scanner_lut_acquire",
        "fj_legacy_scanner_lut_view",
        "fj_legacy_scanner_lut_release",
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
    scanner_lut_results = []
    scanner_color_results = []
    scanner_descriptor_results = []
    scanner_post_results = []
    gaussian_results = []
    frame_results = []
    halation_frame_results = []
    diffusion_frame_results = []
    dir_frame_results = []
    reference_results = []
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
            dir_frame_results = [json.loads(line.removeprefix("DIR_FRAME_RESULT "))
                                 for line in output.splitlines() if line.startswith("DIR_FRAME_RESULT ")]
            assert len(dir_frame_results) == 121 and dir_frame_results == dir_spec["production_cases"], "complete DIR production values/failure trace membership"
            diffusion_frame_results = [json.loads(line.removeprefix("DIFFUSION_FRAME_RESULT "))
                                       for line in output.splitlines() if line.startswith("DIFFUSION_FRAME_RESULT ")]
            assert len(diffusion_frame_results) == 507 and diffusion_frame_results == diffusion_spec["production_cases"], "complete diffusion production values/failure membership"
            halation_frame_results = [json.loads(line.removeprefix("HALATION_FRAME_RESULT "))
                                      for line in output.splitlines() if line.startswith("HALATION_FRAME_RESULT ")]
            assert len(halation_frame_results) == 86 and halation_frame_results == halation_spec["production_cases"], "complete halation production values/failure membership"
            _, _, reference_rows = reference_fixture(arguments.preset)
            expected = []
            for row in reference_rows:
                method = row["method"]
                for case in range(8 if method == 1 else 7):
                    envelope = ([0, 0, 0, 0, 0, 0, 0] if case in (0, 5) else
                                [2, 0, 0, 1, 0, 0, 0] if case in (1, 4) else
                                [3, 0, 0, 2, 1, 0, 0] if case == 2 else
                                [3, 0, 0, 2, 2, 1, 0] if case == 3 else
                                [3, 0, 0, 3, 0, 7, 0] if case == 6 else
                                [3, 0, 0, 4, 1, 8, 1])
                    expected.append((f"reference-{method}-{case}", envelope,
                                     row["reference"]["hash"] if case in (0, 5) else 0))
            lines = [line for line in output.splitlines() if line.startswith("REFERENCE_CASE")]
            assert len(lines) == len(expected), lines
            for line, (case, envelope, identity) in zip(lines, expected):
                match = re.fullmatch(r"REFERENCE_CASE (\S+) category=(\d+) api=(\d+) native_code=(\d+) reason=(\d+) exposure=(\d+) operation=(\d+) channel=(\d+) fields=13-pass hash=(\d+)", line)
                assert match and match[1] == case, line
                actual = [int(value) for value in match.groups()[1:]]
                assert actual == [*envelope, identity], line
                reference_results.append({"case": case, "envelope": envelope,
                                          "hash": identity, "fields": "13-pass"})
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
            lut_names = ["LUT-A2", "LUT-A3", "LUT-FAIL-RESOLUTION-1", "LUT-FAIL-CAPACITY", "LUT-FAIL-MISSING-SPAN", "LUT-FAIL-MISMATCHED-SPAN", "LUT-FAIL-MEDIUM-256", "LUT-FAIL-BOOLEAN-2", "LUT-MEDIUM-K0", "LUT-SPECTRAL-EARLY-X", "LUT-SPECTRAL-EARLY-Y", "LUT-SPECTRAL-EARLY-Z", "LUT-FAIL-NULL-VIEW", "LUT-FAIL-INDEPENDENT-OUTPUT-CLEAR", "LUT-FAIL-MALFORMED-DIAGNOSTIC", "LUT-LIFE-NULL-RELEASE", "LUT-LIFE-CONSUMED-INVALID-DIAGNOSTIC"]
            lines = [line for line in output.splitlines() if line.startswith("SCANNER_LUT_CASE ")]
            assert len(lines) == len(lut_names)
            envelopes = [(0, 0, 2**32-1, 2**64-1, 0)] * 2 + [(3, 2, 2**32-1, 2**64-1, 0), (4, 3, 2**32-1, 2**64-1, 1)] + [(2, 1, 2**32-1, 2**64-1, 0)] * 4 + [(3, 5, 0, 0, 0)] + [(3, 5, channel, 0, 0) for channel in range(3)] + [(2, 1, 2**32-1, 2**64-1, 0)] * 3 + [(0, 0, 2**32-1, 2**64-1, 0), (2, 0, 2**32-1, 2**64-1, 0)]
            for line, name, envelope in zip(lines, lut_names, envelopes, strict=True):
                match = re.fullmatch(r"SCANNER_LUT_CASE (\S+) category=(\d+) api=(\d+) native_code=(-?\d+) reason=(\d+) channel=(\d+) index=(\d+) buffer=(\d+) fields=pass", line)
                assert match and match[1] == name, line
                actual = [int(x) for x in match.groups()[1:]]
                assert actual == [envelope[0], 0, 0, *envelope[1:]], line
                scanner_lut_results.append({"case": name, "fields": actual})
            _, _, _, _, color_cases = scanner_color_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("SCANNER_COLOR_CASE ")]
            assert len(lines) == len(color_cases)
            for line, expected in zip(lines, color_cases, strict=True):
                match = re.fullmatch(r"SCANNER_COLOR_CASE (\S+) category=(\d+) api=(\d+) native_code=(-?\d+) reason=(\d+) stage=(\d+) fields=pass", line)
                assert match and match[1] == expected["id"], line
                actual = [int(x) for x in match.groups()[1:]]
                assert actual == expected["envelope"], line
                scanner_color_results.append({"case": expected["id"], "fields": actual})
            _, _, _, _, descriptor_cases = scanner_descriptor_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("SCANNER_DESCRIPTOR_CASE ")]
            assert len(lines) == len(descriptor_cases)
            for line, expected in zip(lines, descriptor_cases, strict=True):
                match = re.fullmatch(r"SCANNER_DESCRIPTOR_CASE (\S+) category=(\d+) api=(\d+) native_code=(-?\d+) reason=(\d+) lane=(\d+) fields=10-pass hash=(\d+)", line)
                assert match and match[1] == expected["case"], line
                actual = [int(x) for x in match.groups()[1:]]
                assert actual == [*expected["envelope"], expected["hash"]], line
                scanner_descriptor_results.append(expected)
            _, _, _, _, post_cases = scanner_post_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("SCANNER_POST_CASE ")]
            assert len(lines) == len(post_cases)
            for line, expected in zip(lines, post_cases, strict=True):
                match = re.fullmatch(r"SCANNER_POST_CASE (\S+) category=(\d+) api=(\d+) native_code=(-?\d+) reason=(\d+) lane=(\d+) fields=13-pass hash=(\d+)", line)
                assert match and match[1] == expected["case"], line
                assert [int(x) for x in match.groups()[1:]] == [*expected["envelope"], expected["hash"]], line
                scanner_post_results.append(expected)
            _, _, _, frame_spec = frame_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("FRAME_CASE ")]
            assert len(lines) == len(frame_spec["production_cases"])
            for line, expected in zip(lines, frame_spec["production_cases"], strict=True):
                actual = json.loads(line.removeprefix("FRAME_CASE "))
                assert actual == expected, actual["case"]
                frame_results.append(actual)
            _, _, _, _, gaussian_cases = gaussian_fixture(arguments.preset)
            lines = [line for line in output.splitlines() if line.startswith("GAUSSIAN_CASE ")]
            assert len(lines) == len(gaussian_cases)
            for line, expected in zip(lines, gaussian_cases, strict=True):
                actual = json.loads(line.removeprefix("GAUSSIAN_CASE "))
                assert actual == expected, actual["case"]
                gaussian_results.append(actual)
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
                "scanner_lut_results": scanner_lut_results,
                "scanner_color_results": scanner_color_results,
                "scanner_descriptor_results": scanner_descriptor_results,
                "scanner_post_results": scanner_post_results,
                "frame_results": frame_results,
                "dir_frame_results": dir_frame_results,
                "dir_frame_fixture": {"manifest": str(dir_manifest), "payload": str(dir_payload), "normative": str(dir_checked),
                                      "input_spec": str(dir_manifest.parent / "inputs.json"), "logical_preset": arguments.preset},
                "diffusion_frame_results": diffusion_frame_results,
                "diffusion_frame_fixture": {"edges_payload": str(diffusion_edges_payload), "edges_context": str(diffusion_edges_context), "edges_input_spec": str(diffusion_edges_inputs), "manifest": str(diffusion_manifest), "payload": str(diffusion_payload), "context": str(diffusion_context),
                                            "normative": str(diffusion_checked), "input_spec": str(diffusion_manifest.parent / "inputs.json"), "logical_preset": arguments.preset},
                "halation_frame_results": halation_frame_results,
                "halation_frame_fixture": {"manifest": str(halation_manifest), "payload": str(halation_payload), "context": str(halation_context),
                                           "normative": str(halation_checked), "input_spec": str(halation_manifest.parent / "inputs.json"), "logical_preset": arguments.preset},
                "frame_fixture": {"manifest": str(frame_manifest), "payload": str(frame_payload), "context": str(frame_context), "logical_preset": arguments.preset},
                "gaussian_results": gaussian_results,
                "gaussian_fixture": {"manifest": str(gaussian_manifest), "payload": str(gaussian_payload), "context": str(gaussian_context), "logical_preset": arguments.preset},
                "scanner_post_fixture": {"manifest": str(post_manifest), "payload": str(post_payload), "context": str(post_context), "logical_preset": arguments.preset},
                "scanner_descriptor_fixture": {"manifest": str(descriptor_manifest), "payload": str(descriptor_payload), "context": str(descriptor_context), "logical_preset": arguments.preset},
                "reference_results": reference_results,
                "reference_fixture": {"manifest": str(reference_manifest), "payload": str(reference_payload), "logical_preset": arguments.preset},
                "scanner_color_fixture": {"manifest": str(color_manifest), "payload": str(color_payload), "normative": str(color_checked), "logical_preset": arguments.preset},
                "scanner_lut_fixture": {"manifest": str(lut_manifest), "payload": str(lut_payload), "context": str(lut_context), "normative": str(lut_checked), "logical_preset": arguments.preset},
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


def reference_fixture(preset):
    directory = root / "tests/film/reference-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    payload = directory / spec["presets"][preset]
    assert digest(payload) == spec["payload_sha256"][payload.name]
    rows = [row for row in json.loads(payload.read_text())["products"] if "input" in row]
    assert len(rows) == 2 and [row["method"] for row in rows] == [0, 1]
    return manifest, payload, rows


def expected_reference_header(out):
    _, _, rows = reference_fixture(arguments.preset)
    header = []
    for index, row in enumerate(rows):
        source = row["input"]
        assert source["key"] == "kodak_portra_400" and row["route"] == 0
        for name in ("basis", "illuminant", "sensitivity", "tc", "white", "gamma"):
            values = source[name]
            header.append(f"static const uint32_t reference_{index}_{name}[] = {{"
                          + ",".join(str(word) + "U" for word in values or [0]) + "};")
        reference = row["reference"]
        values = [word for name in ("baseline_raw_rgb", "compensated_raw_rgb", "baseline_density_cmy", "compensated_density_cmy") for word in reference[name]]
        values.append(reference["meter_scale"])
        header.append(f"static const uint32_t reference_{index}_expected[] = {{"
                      + ",".join(str(word) + "U" for word in values) + "};")
        for name, value in (("method", row["method"]), ("auto", int(source["auto"])),
                            ("ev", source["manual_ev"]), ("green", source["green_scale"]),
                            ("hash", reference["hash"])):
            header.append(f"#define REFERENCE_{index}_{name.upper()} UINT64_C({value})")
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("\n".join(header) + "\n")


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
    expected_reference_header(out)
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


def scanner_lut_fixture(preset):
    directory = root / "tests/scanner/lut-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    entry = spec["presets"][preset]
    logical = entry["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    for path in (payload, context):
        assert digest(path) == spec["storage_files"][path.name]
    native = load_shared_fixture(directory / logical)
    assert native["preset"] == preset
    assert hashlib.sha256(original_capture_bytes(native, spec["original_line_endings"][logical])).hexdigest() == entry["sha256"]
    checked = directory / "checked.json"
    normative = json.loads(checked.read_text())
    assert normative["failure_defaults"] == {"reason": 0, "channel": 2**32 - 1, "index": 2**64 - 1, "buffer": 0}
    assert normative["reasons"] == {"RAW": [2, 0, 0], "RESOLUTION": [3, 0, 0], "SIZE": [4, 0, 0], "ALLOCATION": [4, 0, 0], "NONFINITE": [3, 0, 0], "PANIC": [8, 0, 0]}
    rows = [next(row for row in native["cases"] if row["id"] == name) for name in ("LUT-A2", "LUT-A3")]
    for row in rows:
        r = row["input"]["resolution"]
        assert row["native"]["success"] and row["native"]["resolution"] == r
        for i, name in enumerate(("log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max")):
            side = r if i < 4 else r - 1
            words = row["native"][name]
            assert len(words) == 3 * side ** 3 and all(type(word) is int and 0 <= word < 2**32 for word in words)
        assert row["native"]["diagnostic"] == ""
        assert row["native"]["failed_voxel"] == 2**64 - 1 and row["native"]["failed_channel"] == 2**32 - 1
    return manifest, payload, context, checked, rows


def scanner_color_fixture(preset):
    directory = root / "tests/scanner/color-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_text())
    entry = spec["presets"][preset]
    payload = directory / entry["native"]
    assert digest(payload) == entry["sha256"]
    checked = directory / "checked.json"
    cases = json.loads(checked.read_text())["production_cases"]
    native = json.loads(payload.read_text())
    rows = [next(row for row in native["cases"] if row["id"] == case["id"]) for case in cases[:4]]
    for row in rows:
        expected = row["expected"]
        assert expected["params"][2] is True and expected["encoding"]["schema"] == 1
        assert len(expected["cat02"]) == 9 and len(expected["xyz_to_rgb"]) == 9
        assert len(expected["illuminant_xyz"]) == 3 and len(expected["encoding"]["coefficients"]) == 6
        assert len(expected["encoding"]["dwg_to_output"]) == 9
    return manifest, payload, checked, rows, cases


def expected_scanner_color_header(out):
    _, _, _, rows, _ = scanner_color_fixture(arguments.preset)
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const FjScannerColorInput scanner_color_inputs[] = {\n")
        for row in rows:
            i = row["input"]
            stream.write("{%dU,%dU,%dU,{0,0,0},UINT64_C(%d),UINT64_C(%d)},\n" % (i["medium"], i["output_space"], i["apply_cctf"], i["illuminant_hash"], i["active_gamut_recipe_hash"]))
        stream.write("};\nstatic const char* const scanner_color_names[] = {" + ",".join(json.dumps(row["id"]) for row in rows) + "};\n")
        stream.write("static const uint32_t scanner_color_words[][36] = {\n")
        for row in rows:
            e = row["expected"]
            words = e["cat02"] + e["xyz_to_rgb"] + e["illuminant_xyz"] + e["encoding"]["coefficients"] + e["encoding"]["dwg_to_output"]
            assert len(words) == 36
            stream.write("{" + ",".join(str(w) + "U" for w in words) + "},\n")
        stream.write("};\nstatic const uint32_t scanner_color_kinds[] = {" + ",".join(str(row["expected"]["encoding"]["kind"]) + "U" for row in rows) + "};\n")
        stream.write("static const uint64_t scanner_color_hashes[] = {" + ",".join("UINT64_C(%d)" % row["expected"]["hash"] for row in rows) + "};\n")


def dir_frame_fixture(preset):
    directory = root / "tests/dir/frame-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    checked = directory / spec["checked"]
    for path in (payload, checked, directory / "inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset
    assert [len(captured[key]) for key in ("cases", "products")] == [74, 24]
    assert len(spec["production_cases"]) == 121 and len(spec["production_inputs"]) == 121
    assert {row["case"] for row in spec["production_cases"]} == set(spec["production_inputs"])
    return manifest, payload, checked, captured, spec


def expected_dir_frame_header(out):
    _, _, _, _, spec = dir_frame_fixture(arguments.preset)
    def words(values):
        return "{" + ",".join("UINT64_C(" + str(v) + ")" for v in values) + "}"

    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("typedef struct { const char* name; uint32_t mode; uint64_t input[14]; FjStatus status; uint64_t output[79], failure[82]; int output_valid, failure_valid, error_valid; const char* diagnostic; } DirFrameRow;\n")
        stream.write("static const DirFrameRow dir_frame_rows[] = {\n")
        for row in spec["production_cases"]:
            declaration = spec["production_inputs"][row["case"]]
            assert len(declaration["words"]) == 14
            assert row["output"] is None or len(row["output"]) == 79
            assert row["failure"] is None or len(row["failure"]) == 82
            stream.write("{%s,%d,%s,%s,%s,%s,%d,%d,%d,%s},\n" % (
                json.dumps(row["case"]), declaration["mode"], words(declaration["words"]),
                "{" + ",".join(str(v) for v in row["envelope"]) + "}", words(row["output"] or [0]), words(row["failure"] or [0]),
                row["output"] is not None, row["failure"] is not None, row["diagnostic"] is not None,
                json.dumps(row["diagnostic"]) if row["diagnostic"] is not None else "NULL"))
        stream.write("};\n")


def diffusion_frame_fixture(preset):
    directory = root / "tests/diffusion/frame-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    checked = directory / spec["checked"]
    edge_logical = spec["presets"][preset]["edges"]
    edge_payload = directory / spec["payloads"][edge_logical]
    edge_context = directory / spec["contexts"][edge_logical]
    for path in (payload, context, checked, directory / "inputs.json", edge_payload, edge_context, directory / "edge-inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = load_shared_fixture(directory / logical)
    edges = load_shared_fixture(directory / edge_logical)
    assert captured["preset"] == preset and edges["preset"] == preset
    assert len(edges["samples"]) == 2
    captured["samples"].extend(edges["samples"])
    assert [len(captured[key]) for key in ("samples", "frames", "products", "primitives")] == [96, 35, 92, 8]
    assert len(spec["production_cases"]) == 507 and len(spec["production_inputs"]) == 507
    assert {row["case"] for row in spec["production_cases"]} == set(spec["production_inputs"])
    return manifest, payload, context, checked, captured, spec


def expected_diffusion_frame_header(out):
    _, _, _, _, _, spec = diffusion_frame_fixture(arguments.preset)
    def words(values):
        return "{" + ",".join("UINT64_C(" + str(v) + ")" for v in values) + "}"
    def integers(values):
        return "{" + ",".join(str(v) for v in values) + "}"
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("typedef struct { const char* name; uint32_t operation, mode, input_count; uint64_t input[34]; FjStatus status; uint32_t failure[3]; uint32_t output_count; uint64_t output[36]; int output_valid, failure_valid, error_valid; const char* diagnostic; } DiffusionFrameRow;\n")
        stream.write("static const DiffusionFrameRow diffusion_frame_rows[] = {\n")
        for row in spec["production_cases"]:
            input_record = spec["production_inputs"][row["case"]]
            operation = input_record["operation"]
            assert len(input_record["words"]) == (34 if operation == 3 else 12)
            count = {1: 12, 2: 36, 3: 35}[operation]
            assert row["output"] is None or len(row["output"]) == count
            stream.write("{%s,%d,%d,%d,%s,%s,%s,%d,%s,%d,%d,%d,%s},\n" % (
                json.dumps(row["case"]), operation, input_record["mode"], len(input_record["words"]), words(input_record["words"]),
                integers(row["envelope"]), integers(row["failure"] or [0, 0, 0]), count, words(row["output"] or [0]),
                row["output"] is not None, row["failure"] is not None, row["diagnostic"] is not None,
                json.dumps(row["diagnostic"]) if row["diagnostic"] is not None else "NULL"))
        stream.write("};\n")


def halation_frame_fixture(preset):
    directory = root / "tests/scatter_halation/frame-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    checked = directory / spec["checked"]
    for path in (payload, context, checked, directory / "inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset
    assert len(captured["cases"]) == 52 and len(captured["products"]) == 20
    assert len(spec["production_cases"]) == 86
    assert len({row["case"] for row in spec["production_cases"]}) == 86
    return manifest, payload, context, checked, captured, spec


def expected_halation_frame_header(out):
    _, _, _, _, captured, spec = halation_frame_fixture(arguments.preset)
    rows = captured["cases"] + captured["products"]
    by_name = {row["input"]["name"]: row for row in rows}
    malformed = {row["case"]: row for row in spec["production_malformed"]}
    def words(values):
        return "{" + ",".join(str(value) + "U" for value in values) + "}"

    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("typedef struct { const char* name; uint32_t mode, flags[2], input[10]; uint64_t recipe_hash; FjStatus status; uint32_t failure[4]; uint64_t sigma_bits; int sigma_nan, output_valid, failure_valid, error_valid; uint64_t expected_hash; const uint32_t* words; const char* diagnostic; } HalationFrameRow;\n")
        for index, expected in enumerate(spec["production_cases"]):
            if expected["output"] is not None:
                assert len(expected["output"]["words"]) == 529
                stream.write(f"static const uint32_t halation_frame_words_{index}[529] = {{" + ",".join(str(v) + "U" for v in expected["output"]["words"]) + "};\n")
        stream.write("static const HalationFrameRow halation_frame_rows[] = {\n")
        for index, expected in enumerate(spec["production_cases"]):
            ordinary = index < len(rows)
            declaration = None if ordinary else malformed[expected["case"]]
            row = rows[index] if ordinary else by_name[declaration["source_case"]]
            input_record = row["input"]
            values = input_record["scalars"] + input_record["first_sigma_um"] + input_record["total_strength"] + [input_record["pixel_size_um"]]
            output = expected["output"]
            stream.write("{%s,%dU,%s,%s,UINT64_C(%d),%s,%s,UINT64_C(%d),%d,%d,%d,%d,UINT64_C(%d),%s,%s},\n" % (
                json.dumps(expected["case"]), 0 if ordinary else declaration["mode"],
                words([int(input_record["scatter_active"]), int(input_record["back_reflection_active"])]), words(values), input_record["recipe_hash"],
                words(expected["envelope"]), words(expected["failure"] or [0] * 4), expected["sigma_bits"] or 0,
                bool(expected["sigma_nan"]), output is not None, expected["failure"] is not None, expected["diagnostic"] is not None,
                output["recipe_hash"] if output else 0, f"halation_frame_words_{index}" if output else "NULL",
                json.dumps(expected["diagnostic"]) if expected["diagnostic"] is not None else "NULL"))
        stream.write("};\n")


def frame_fixture(preset):
    directory = root / "tests/frame/geometry-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    for path in (payload, context, directory / "inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset
    assert len(captured["domains"]) == 15 and len(captured["geometry"]) == 22
    assert len(spec["production_cases"]) == len(spec["production_inputs"]) == 69
    return manifest, payload, context, spec


def expected_frame_header(out):
    _, _, _, spec = frame_fixture(arguments.preset)
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("static const struct { const char* name; int operation, kind; int32_t input[4], expected[4]; uint32_t film, pixel, category, reason; uint64_t wide; int output_valid, failure_valid; const char* diagnostic; } frame_rows[] = {\n")
        for source, expected in zip(spec["production_inputs"], spec["production_cases"], strict=True):
            assert source["case"] == expected["case"]
            fields = [json.dumps(source["case"]), str(source["operation"]), str(source["kind"]),
                      "{" + ",".join(map(str, source["input"])) + "}",
                      "{" + ",".join(map(str, expected["domain"] or [0]*4)) + "}",
                      str(source["film_bits"]) + "U", str(expected["pixel_bits"] or 0) + "U",
                      str(expected["envelope"][0]) + "U", str(expected["envelope"][3] or 0) + "U",
                      "UINT64_C(" + str(expected["diffusion_bits"] or 0) + ")",
                      str(int(expected["domain"] is not None)), str(int(expected["envelope"][3] is not None)),
                      "NULL" if expected["diagnostic"] is None else json.dumps(expected["diagnostic"])]
            stream.write("{" + ",".join(fields) + "},\n")
        stream.write("};\n")


def gaussian_fixture(preset):
    directory = root / "tests/optics/gaussian-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload, context = (directory / spec[k][logical] for k in ("payloads", "contexts"))
    assert all(digest(path) == spec["sha256"][path.name] for path in (payload, context))
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset and len(captured["cases"]) == 188
    cases = []
    for row in captured["cases"]:
        radius, sigma = row["radius"], row["sigma_bits"]
        reason = 2 if radius < 1 or radius > 2048 else (3 if sigma & 0x80000000 or sigma == 0 or sigma >= 0x7f800000 else 0)
        cases.append({"case": row["name"], "envelope": [3 if reason else 0, 0, 0, reason],
                      "radius": 0 if reason else radius, "sigma_bits": 0 if reason else sigma,
                      "weights": [] if reason else row["weights"]})
    for name in spec["production_cases"][188:]:
        reason = 2 if name == "radius-upper-reject" else (0 if name in {"zero-capacity", "null-release"} else 1)
        cases.append({"case": name, "envelope": [3 if reason == 2 else 2 if reason else 0, 0, 0, reason],
                      "radius": 0, "sigma_bits": 0, "weights": []})
    assert [v["case"] for v in cases] == spec["production_cases"]
    return manifest, payload, context, captured, cases


def expected_gaussian_header(out):
    _, _, _, captured, cases = gaussian_fixture(arguments.preset)
    with (out / "production-film-expected.h").open("a") as stream:
        for i, row in enumerate(captured["cases"]):
            stream.write(f"static const uint32_t gaussian_words_{i}[] = {{" + ",".join(str(w) + "U" for w in row["weights"] or [0]) + "};\n")
        stream.write("static const struct { const char* name; int32_t radius; uint32_t sigma, reason; size_t count; const uint32_t* words; } gaussian_rows[] = {\n")
        for i, (row, expected) in enumerate(zip(captured["cases"], cases[:188], strict=True)):
            stream.write("{%s,%d,%dU,%dU,%d,gaussian_words_%d},\n" % (json.dumps(row["name"]), row["radius"], row["sigma_bits"], expected["envelope"][3], len(expected["weights"]), i))
        stream.write("};\n")


def scanner_post_fixture(preset):
    directory = root / "tests/scanner/post-effects-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    assert all(digest(path) == spec["sha256"][path.name] for path in (payload, context))
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset and len(captured["cases"]) == 400
    cases = spec["production_cases"]
    assert len(cases) == 410 and len({row["case"] for row in cases}) == 410
    return manifest, payload, context, captured, cases


def expected_scanner_post_header(out):
    _, _, _, captured, cases = scanner_post_fixture(arguments.preset)
    rows = captured["cases"]
    with (out / "production-film-expected.h").open("a") as stream:
        def array(name, values):
            stream.write(f"static const uint32_t {name}[][{len(values[0])}] = {{\n")
            for words in values:
                stream.write("{" + ",".join(str(word) + "U" for word in words) + "},\n")
            stream.write("};\n")
        array("post_input_words", [[row["input"]["route"], int(row["input"]["glare_active"]), *row["input"]["values"]] for row in rows])
        projected = []
        for row in rows:
            d = row["native"]["descriptor"]
            projected.append([d["route"], int(d["glare_active"]), *d["values"], *d["radii"], d["schema"]] if not row["native"]["exception"] else [0] * 12)
        array("post_expected_words", projected)
        array("post_envelopes", [row["envelope"] for row in cases[:400]])
        stream.write("static const uint64_t post_hashes[] = {" + ",".join(f"UINT64_C({row['hash']})" for row in cases[:400]) + "};\n")
        for name, values in (("post_names", [row["id"] for row in rows]), ("post_diagnostics", ["invalid scanner post-effects arguments" if row["native"]["exception"] else row["native"]["diagnostic"] for row in rows])):
            stream.write(f"static const char* const {name}[] = {{" + ",".join(json.dumps(v) for v in values) + "};\n")


def scanner_descriptor_fixture(preset):
    directory = root / "tests/scanner/descriptor-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    logical = spec["presets"][preset]["native"]
    payload = directory / spec["payloads"][logical]
    context = directory / spec["contexts"][logical]
    assert all(digest(path) == spec["sha256"][path.name] for path in (payload, context))
    captured = load_shared_fixture(directory / logical)
    assert captured["preset"] == preset and len(captured["cases"]) == 520 and len(captured["products"]) == 4
    cases = spec["production_cases"] + spec["production_malformed"]
    assert len(cases) == 41 and len({v["case"] for v in cases}) == len(cases)
    return manifest, payload, context, captured, cases


def expected_scanner_descriptor_header(out):
    _, _, _, captured, cases = scanner_descriptor_fixture(arguments.preset)
    by_id = {row["id"]: row for row in captured["cases"]}
    selected = [by_id[row["case"]] for row in cases[:24]]
    with (out / "production-film-expected.h").open("a") as stream:
        def array(name, rows):
            stream.write("static const uint32_t " + name + "[][" + str(len(rows[0])) + "] = {\n")
            for row in rows:
                stream.write("{" + ",".join(str(v) + "U" for v in row) + "},\n")
            stream.write("};\n")
        array("descriptor_channel_words", [row["input"]["channel_density_cmy"] for row in selected])
        array("descriptor_base_words", [row["input"]["base_density"] for row in selected])
        for i, row in enumerate(selected):
            for key in ("viewing_illuminant", "profile_viewing_illuminant", "observer_identity", "print_profile_key"):
                stream.write(f"static const uint8_t descriptor_{i}_{key}[] = {{" + ",".join(str(v) for v in row["input"][key] or [0]) + "};\n")
        stream.write("static const FjScannerLutDescriptorInput descriptor_inputs[] = {\n")
        for i, row in enumerate(selected):
            v = row["input"]
            prefix = [str(int(v[k])) + "U" for k in ("mode", "route", "capture_polarity", "bounds_medium", "bounds_source", "output_medium", "profile_present", "resolution")]
            prefix += ["UINT64_C(%d)" % v[k] for k in ("density_bounds_hash", "print_profile_token")]
            prefix += ["{NULL,243}", "{NULL,81}"]
            prefix += [f"{{descriptor_{i}_{key},{len(v[key])}}}" for key in ("viewing_illuminant", "profile_viewing_illuminant", "observer_identity", "print_profile_key")]
            stream.write("{" + ",".join(prefix) + "},\n")
        stream.write("};\nstatic const FjScannerLutDescriptor descriptor_expected[] = {\n")
        for row in selected:
            d = row["native"]["descriptor"] if row["native"]["success"] else dict.fromkeys(row["native"]["descriptor"], 0)
            values = [str(d[k]) + "U" for k in ("route", "medium", "capture_polarity", "resolution")]
            values += ["UINT64_C(%d)" % d[k] for k in ("density_bounds_hash", "channel_density_hash", "base_density_hash", "viewing_illuminant_hash", "observer_hash", "hash")]
            stream.write("{" + ",".join(values) + "},\n")
        stream.write("};\nstatic const char* const descriptor_names[] = {" + ",".join(json.dumps(v["case"]) for v in cases) + "};\n")
        stream.write("static const char* const descriptor_diagnostics[] = {" + ",".join(json.dumps(v["native"]["diagnostic"]) for v in selected) + "};\n")
        array("descriptor_envelopes", [v["envelope"] for v in cases])


def expected_scanner_lut_header(out):
    _, _, _, _, rows = scanner_lut_fixture(arguments.preset)
    first = rows[0]["input"]
    assert {k: val for k, val in first.items() if k != "resolution"} == {k: val for k, val in rows[1]["input"].items() if k != "resolution"}
    with (out / "production-film-expected.h").open("a") as stream:
        def words(name, values):
            stream.write("static const uint32_t " + name + "[] = {" + ",".join(str(word) + "U" for word in values) + "};\n")
        for field in ("eps_c", "eps_m", "eps_y", "weighted_x", "weighted_y", "weighted_z", "baseline", "range_min_cmy", "inverse_span_cmy"):
            words("lut_input_" + field, first[field])
        words("lut_input_inv_yn", [first["inv_yn"]])
        for row in rows:
            for field in ("log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"):
                words("lut_" + str(row["input"]["resolution"]) + "_" + field, row["native"][field])


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
    expected_scanner_lut_header(arguments.artifacts)
    expected_scanner_color_header(arguments.artifacts)
    expected_scanner_descriptor_header(arguments.artifacts)
    expected_scanner_post_header(arguments.artifacts)
    expected_gaussian_header(arguments.artifacts)
    expected_frame_header(arguments.artifacts)
    expected_halation_frame_header(arguments.artifacts)
    expected_diffusion_frame_header(arguments.artifacts)
    expected_dir_frame_header(arguments.artifacts)
    execute(arguments)
