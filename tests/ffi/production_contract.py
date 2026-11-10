"""Generate C11 expectations or run the testing-OFF Release production gate."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

from shared_fixture import load_shared_fixture
from effects_frame_probe_contract import (
    EFFECTS_FRAME_CASE_COUNT, EFFECTS_FRAME_CASE_IDS,
    EFFECTS_FRAME_INPUT_WORD_COUNT, EFFECTS_FRAME_OUTPUT_WORD_COUNT,
    EFFECTS_FRAME_FAILURE_WORD_COUNT,
)
from grain_probe_contract import (
    GRAIN_FRAME_CASE_COUNT,
    GRAIN_FRAME_CASE_IDS,
    GRAIN_FRAME_CURVE_COUNT,
    GRAIN_FRAME_CURVE_SAMPLE_COUNT,
    GRAIN_FRAME_FAILURE_WORD_COUNT,
    GRAIN_FRAME_INPUT_WORD_COUNT,
    GRAIN_FRAME_OUTPUT_WORD_COUNT,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

root = Path(__file__).resolve().parents[2]
fixtures = Path(__file__).resolve().parent
arguments = None
windows = os.name == "nt"
suffix = ".exe" if windows else ""
build = None
llvm = (
    Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin")
    if windows
    else Path("/usr/lib/llvm-22/bin")
)


PRODUCTION_FORBIDDEN_SYMBOLS = re.compile(
    "fj_test_|::test_support::|film_bridge::faults::|film::input::testing::|film_input_bridge::(?:BOUNDARY_FAULT|boundary_fault|arm_boundary_fault)|film::test_support::|\\b(?:FAULT|[A-Z_]+_FAULT|FAULT_[A-Z_]+|[A-Z_]+_FAULT_[A-Z_]+)\\b|\\b[A-Z_]+_FACADE(?:_[A-Z_]+)?\\b|ContextDrainTestFailure|TestSupport::|(?:Profile|Catalog|Terminal|ContextDrain|HostAdmission|ExecutorFailure|PinnedUpload|Illuminant|Render|Spectral|TcLut|Color|OutputGamut|Development|Noise)Test::|\\b\\w*Test::|::fixtures::|::output_fixtures::|::faults::|TC_CONSUMED|TC_TRANSFERS|TC_RELEASES|EXPOSURE_CONSUMED|TC_FACADE_CONSUMED|TC_FACADE_RELEASES|ScannerOracle::|pendingCaptureTest|set_pending_capture_test_hook|COPY_METRICS|SCRATCH_CAPACITIES|gamut_bridge::(?:output_)?fixtures::|exposure_bridge::fixtures::|spectral_bridge::(?:inject|arm_fault|clear_fault)"
)
PRODUCTION_FORBIDDEN_SYMBOLS = re.compile(
    PRODUCTION_FORBIDDEN_SYMBOLS.pattern
    + r"|reconstruction::testing::|film::reference::testing::"
    + r"|SpectralAllocationProbe::|gaussian_faults::|halation_frame_faults::|diffusion_frame_faults::|dir_frame_faults::|DirFrameObservation|rejectNativeAllocation|rejectedNativeBytes"
    + r"|WorkspaceTransitionObservation|WorkspaceCandidateObservation|WorkspaceRejection"
)
PRODUCTION_FORBIDDEN_STRINGS = [
    b"workspace candidate capacity rejection",
    b"workspace candidate invalid-value rejection",
    b"workspace candidate context-loss rejection",
    b"--workspace-transitions",
    b"spatial DIR boundary fixture panic",
    b"spatial DIR boundary candidate rendezvous canceled",
    b"JUICER_DIR_BOUNDARY_TEST_HOOK",
    b"DirBoundaryTest::",
    b"fj_test_dir_boundary",
    b"capture-dir-boundaries",
    b"visual grain frame fixture entry panic",
    b"visual grain frame fixture layer panic",
    b"JUICER_GRAIN_FRAME_CAPTURE_TEST_HOOK",
    b"JuicerGrainCapture",
    b"GrainFrameObservation",
    b"EffectsFrameObservation",
    b"physical effects frame fixture entry panic",
    b"fj_test_effects_frame",
    b"capture-effects-frames",
    b"JUICER_EFFECTS_FRAME_CAPTURE_TEST_HOOK",
    b"--effects-frame",
    b"fj_test_grain_frame",
    b"capture-grain-frames",
    b"--grain-frame",
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
PRODUCTION_MODULE_EXPORTS = ("OfxGetNumberOfPlugins", "OfxGetPlugin")

# Count the bytes already read for identity checks; no second inventory pass.
IO_COSTS = {"hash_read_bytes": 0, "hash_reads": 0, "copy_read_bytes": 0,
            "copy_write_bytes": 0, "command_output_write_bytes": 0}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            IO_COSTS["hash_read_bytes"] += len(block)
            value.update(block)
    IO_COSTS["hash_reads"] += 1
    return value.hexdigest()


def production_module_export_names(output, windows):
    if windows:
        return re.findall(r"^\s*Name: (.+)$", output, re.MULTILINE)
    return [line.split()[-1] for line in output.splitlines() if line.strip()]

def execute(arguments):
    from execution import NativeRun, OutputLease, windows_vctip
    from qualification import validate_production_contract
    import math

    if arguments.preset not in {"linux-release", "windows-clang-release"}:
        raise ValueError("production contract requires a Release preset")
    out = arguments.artifacts.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inherited = json.loads(os.environ.get("JUICER_OUTPUT_LEASES", "{}"))
    leases = (OutputLease(root, build), OutputLease(root))
    for lease in leases:
        key = hashlib.sha256(os.path.normcase(str(lease.path)).encode()).hexdigest()
        if key not in inherited:
            raise RuntimeError("production contract requires delegated build and GPU owners")

    def execute_owned():
        run = NativeRun(root, out)
        commands = []
        commands_path = out / "commands.json"

        def command(name, argv, *, env=None, compiler=False):
            try:
                log = run.command(name, argv, env=env, stdout_only=True,
                                  vctip_executable=windows_vctip(os.environ) if compiler else None)
            finally:
                record = dict(run.commands[-1])
                stdout = out / (name + ".log")
                stderr = out / (name + ".stderr.log")
                if stdout.is_file():
                    record["stdout_sha256"] = digest(stdout)
                if stderr.is_file():
                    record["stderr_sha256"] = digest(stderr)
                commands.append(record)
                commands_path.write_text(json.dumps(commands, indent=2) + "\n")
                IO_COSTS["command_output_write_bytes"] += record.get("output_bytes", 0) + record.get("stderr_bytes", 0)
            return log.read_text(encoding="utf-8", errors="replace")

        cache = build / "CMakeCache.txt"
        ninja = build / "build.ninja"
        compdb_path = build / "compile_commands.json"
        assert "BUILD_TESTING:BOOL=OFF" in cache.read_text()
        features = build / "cargo/production/features.txt"
        stamp = build / "cargo/production/build.stamp"
        assert features.read_text().splitlines()[0] == "features=none" and stamp.is_file()
        cargo_commands = [line for line in ninja.read_text().splitlines() if "--package film-juicer-plugin" in line]
        assert cargo_commands and all("--no-default-features" in line and "--features" not in line
                                      and "/cargo/production" in line.replace("\\", "/") for line in cargo_commands)
        compdb = json.loads(compdb_path.read_text())
        assert not any(re.search(r"JUICER_\w*TEST_HOOK", row.get("command", " ".join(row.get("arguments", []))))
                       for row in compdb)
        archive = build / "cargo/production" / ("x86_64-pc-windows-msvc/release/film_juicer_plugin.lib"
                                                    if windows else "x86_64-unknown-linux-gnu/release/libfilm_juicer_plugin.a")
        native = build / ("juicer_core.lib" if windows else "libjuicer_core.a")
        build_module = build / "juicer.ofx"
        contents = root / "out/stage" / arguments.preset / "juicer.ofx.bundle/Contents"
        module = contents / ("Win64" if windows else "Linux-x86-64") / "juicer.ofx"
        resources = contents / "Resources"
        render = build / "tests/bin" / ("JuicerProductionPreparedRenderProbe" + suffix)
        binary_inputs = [cache, ninja, compdb_path, features, stamp, archive, native, build_module, module, render]
        if windows:
            binary_inputs.append(build / "juicer.pdb")
        before = {str(path): digest(path) for path in binary_inputs}
        assert before[str(build_module)] == before[str(module)], "staged and build modules differ"
        processor_fixture = root / "tests/ofx/fixtures/exposure" / f"processor-{arguments.preset}.dat"
        updates = root / "tests/print/illuminant-fixtures/product-updates.json"
        source_inputs = [Path(__file__), root / "scripts/execution.py", root / "scripts/process_tree.py",
                         root / "scripts/qualification.py", root / "CMakeLists.txt",
                         fixtures / "production-ofx.c", fixtures / "production-cuda-abi.c",
                         fixtures / "production-ofx-render.cpp", root / "tests/ofx/instance_host.cpp",
                         root / "tests/ofx/instance_host.h", root / "tests/ofx/probe.py",
                         root / "tests/ofx/descriptor_manifest.json", processor_fixture, updates,
                         root / "tests/ofx/fixtures/exposure/manifest.json",
                         root / "tests/print/illuminant-fixtures/manifest.json"]
        source_inputs.extend(path for path in (root / "Resources").rglob("*") if path.is_file())
        assert resources.is_dir(), "staged resources missing"
        source_inputs.extend(path for path in resources.rglob("*") if path.is_file())
        consumed_before = {str(path): digest(path) for path in sorted(set(source_inputs))}
        forbidden = re.compile(PRODUCTION_FORBIDDEN_SYMBOLS.pattern +
                               r"|\bfj_legacy_\w*\b|\b(?:JuicerEffect|JuicerProcessor|ProcessRuntime|PreparedState|ParamSnapshot)\b|\bOFX::|Ofx(?:PluginFactory|ImageEffect)Helper")
        scans = []

        def scan(name, path, tool, flags):
            output = command(name, [str(llvm / (tool + suffix)), *flags, str(path)])
            match = forbidden.search(output)
            assert match is None, (name, match.group(0) if match else None)
            data = path.read_bytes()
            assert not any(value in data for value in PRODUCTION_FORBIDDEN_STRINGS), name
            assert hashlib.sha256(data).hexdigest() == before[str(path)]
            scans.append({"name": name, "file": str(path), "sha256": before[str(path)],
                          "symbols_and_fault_strings_absent": True, "legacy_symbols_absent": True})
            return output

        scan("native-symbols", native, "llvm-nm", ["--defined-only", "--demangle"])
        scan("rust-symbols", archive, "llvm-nm", ["--defined-only", "--demangle"])
        if windows:
            exports = scan("module-exports", module, "llvm-readobj", ["--coff-exports"])
            scan("module-symbols", build / "juicer.pdb", "llvm-pdbutil", ["dump", "-globals", "-publics"])
        else:
            exports = scan("module-exports", module, "llvm-nm", ["--dynamic", "--defined-only"])
            scan("module-symbols", module, "llvm-nm", ["--defined-only", "--demangle"])
        export_names = production_module_export_names(exports, windows)
        assert sorted(export_names) == sorted(PRODUCTION_MODULE_EXPORTS), export_names

        no_device_env = os.environ.copy()
        no_device_env["CUDA_VISIBLE_DEVICES"] = "-1"
        if windows:
            cuda = Path(no_device_env["CUDA_PATH_V13_2"])
            no_device_env["PATH"] = os.pathsep.join([str(cuda / "bin"), str(cuda / "bin/x64"),
                                                     no_device_env.get("PATH", "")])
        probe_records = []
        for name, source, target in (("ofx", fixtures / "production-ofx.c", "JuicerProductionOfxProbe"),
                                     ("cuda-abi", fixtures / "production-cuda-abi.c", "JuicerProductionCudaAbiProbe")):
            entry = next(row for row in compdb if Path(row["file"]).resolve() == source.resolve())
            compile_command = entry.get("command") or " ".join(entry["arguments"])
            normalized = compile_command.replace("\\", "/").lower()
            assert re.search(r"(?:[/-]std:c11|-std=c11)", normalized), (name, compile_command)
            if windows:
                assert "clang-cl" in normalized and re.search(r"(?:^|\s)[/-]tc(?:\s|$)", normalized)
            else:
                assert re.search(r"(?:^|[/ ])gcc-?13(?:[ /]|$)", normalized)
            command(name + "-probe-build", ["cmake", "--build", str(build), "--config", "Release",
                                             "--target", target, "--verbose"], compiler=True)
            executable = build / "tests/bin" / (target + suffix)
            linked = digest(executable)
            before[str(executable)] = linked
            command(name + "-probe", [str(executable), *([str(module)] if name == "ofx" else [])],
                    env=no_device_env)
            after = digest(executable)
            assert linked == after
            probe_records.append({"name": name, "executable": str(executable), "linked_sha256": linked,
                                  "before_run_sha256": linked, "after_run_sha256": after,
                                  "source": str(source), "compile_command": compile_command})

        module_command = [sys.executable, "-B", str(root / "tests/ofx/probe.py"), str(module),
                          "--expected-descriptor", str(root / "tests/ofx/descriptor_manifest.json")]
        command("module-probe", module_command, env=no_device_env)
        render_command = [str(render), str(module), str(resources), str(processor_fixture),
                          arguments.preset, str(updates)]
        document = json.loads(command("prepared-render", render_command))
        assert set(document) == {"rows", "lifecycle"}
        fixture_tokens = iter(processor_fixture.read_text().split())
        fixture_cases = ("negative-direct", "negative-print", "positive-direct", "positive-print",
                         "combined-print", "glare-plus-zero", "glare-minus-zero")
        patch_spec = json.loads(updates.read_text())
        assert patch_spec["schema"] == 1 and patch_spec["policy"] == "print-erff-0.2.15-2026-10-20"
        expected_words = {}
        for case in fixture_cases:
            assert next(fixture_tokens) == case
            count = int(next(fixture_tokens))
            values = [struct.unpack("<I", struct.pack("<f", float(next(fixture_tokens))))[0]
                      for _ in range(count)]
            logical = f"ofx/fixtures/exposure/{processor_fixture.name}.{case}"
            patches = patch_spec["fixtures"].get(logical, [])
            assert len(patches) % 2 == 0
            for test, replace in zip(patches[::2], patches[1::2], strict=True):
                assert test["op"] == "test" and replace["op"] == "replace" and test["path"] == replace["path"]
                index = int(test["path"].removeprefix("/"))
                assert values[index] == test["value"]
                values[index] = replace["value"]
            expected_words[case] = values
        assert next(fixture_tokens, None) is None
        rows = []
        for raw in document["rows"]:
            row = dict(raw)
            expected_count = 105 if row["route"] == "negative-direct" else 140
            for key, target in (("input_f32_words", "input_sha256"),
                                ("output_f32_words", "output_f32_words_sha256")):
                words = row.pop(key)
                assert isinstance(words, list) and len(words) == expected_count
                assert all(type(word) is int and 0 <= word <= 0xffffffff for word in words)
                packed = b"".join(struct.pack("<I", word) for word in words)
                if key == "output_f32_words":
                    values = [value[0] for value in struct.iter_unpack("<f", packed)]
                    expected = [struct.unpack("<f", struct.pack("<I", word))[0]
                                for word in expected_words[row["route"]]]
                    assert len(values) == len(expected)
                    assert all(math.isfinite(actual) and math.isfinite(want)
                               and abs(actual - want) <= 2.0e-4 + 3.0e-4 * abs(want)
                               for actual, want in zip(values, expected, strict=True))
                row[target] = hashlib.sha256(packed).hexdigest()
            rows.append(row)
        current = {path: digest(path) for path in before}
        assert current == before, "production build input changed"
        consumed_after = {path: digest(path) for path in consumed_before}
        assert consumed_after == consumed_before, "production source or fixture changed"
        build_inputs = {path: {"sha256_before": value, "sha256_after": current[path]}
                        for path, value in before.items()}
        for original, identity in build_inputs.items():
            source = Path(original)
            retained = out / "inputs" / source.relative_to(root)
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, retained)
            IO_COSTS["copy_read_bytes"] += source.stat().st_size
            IO_COSTS["copy_write_bytes"] += retained.stat().st_size
            assert digest(retained) == identity["sha256_before"], "retained production input changed"
            identity["file"] = str(retained)
        consumed = {path: {"sha256_before": value, "sha256_after": consumed_after[path]}
                    for path, value in consumed_before.items()}
        render_record = commands[-1]
        prepared = {"module": str(module), "module_sha256": before[str(module)],
                    "executable": str(render), "executable_sha256_before": before[str(render)],
                    "executable_sha256_after": current[str(render)], "fixture": str(processor_fixture),
                    "fixture_sha256": consumed_before[str(processor_fixture)], "command": render_command,
                    "stdout_sha256": render_record["stdout_sha256"], "stderr_sha256": render_record["stderr_sha256"],
                    "rows": rows, "lifecycle": document["lifecycle"]}
        report = {"schema": "production-contract-v2", "preset": arguments.preset,
                  "build_testing": "OFF", "test_support": "OFF", "hidden_device_load": True,
                  "io_costs": IO_COSTS, "exports": sorted(export_names), "scanned": scans,
                  "build_inputs": build_inputs, "consumed_inputs": consumed,
                  "loaded_module": {"file": str(module), "sha256": before[str(module)],
                                    "build_file": str(build_module)},
                  "probes": probe_records, "prepared_render": prepared,
                  "commands": {"file": str(commands_path), "sha256": digest(commands_path)}}
        report_data = (json.dumps(report, indent=2) + "\n").encode()
        validate_production_contract(arguments.preset, {"production-contract.json": report_data,
                                                       "production-commands.json": commands_path.read_bytes()})
        (out / "production.json").write_bytes(report_data)
        print("PRODUCTION_MANIFEST_SHA256", digest(out / "production.json"), flush=True)

    with leases[0], leases[1]:
        execute_owned()


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


def dir_boundary_fixture(preset):
    directory = root / "tests/dir/boundary-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    payload = directory / spec["presets"][preset]["native"]
    checked = directory / spec["checked"]
    for path in (payload, checked, directory / "inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = json.loads(payload.read_bytes())
    assert captured["preset"] == preset
    assert len(captured["cases"]) == 114 and len(captured["primitives"]) == 56
    assert len(spec["production_cases"]) == 171 and len(spec["production_inputs"]) == 171
    assert len({row["case"] for row in spec["production_cases"]}) == 171
    assert {row["case"] for row in spec["production_cases"]} == set(spec["production_inputs"])
    return manifest, payload, checked, captured, spec


def expected_dir_boundary_header(out):
    _, _, _, _, spec = dir_boundary_fixture(arguments.preset)
    def words(values):
        return "{" + ",".join("UINT64_C(" + str(v) + ")" for v in values) + "}"
    def integers(values):
        return "{" + ",".join(str(v) for v in values) + "}"
    values = [row for row in spec["production_cases"] if spec["production_inputs"][row["case"]]["kind"] == "values"]
    raw = [row for row in spec["production_cases"] if spec["production_inputs"][row["case"]]["kind"] == "raw"]
    assert len(values) == 117 and len(raw) == 51
    with (out / "production-film-expected.h").open("a") as stream:
        stream.write("#define DIR_BOUNDARY_MAX_WEIGHTS %d\n" % max(len(row["weights"] or []) for row in values))
        stream.write("typedef struct { const char* name; uint64_t input[18]; FjStatus status; uint64_t key, failure[7]; int present; int32_t shape[6]; size_t count; const uint64_t* weights; uint64_t terminal[18]; const char* diagnostic; size_t diagnostic_length; } DirBoundaryRow;\n")
        for index, row in enumerate(values):
            stream.write("static const uint64_t dir_boundary_weights_%d[] = %s;\n" % (index, words(row["weights"] or [0])))
        stream.write("static const DirBoundaryRow dir_boundary_rows[] = {\n")
        for index, row in enumerate(values):
            declaration = spec["production_inputs"][row["case"]]
            assert len(declaration["words"]) == 18 and len(row["failure"]) == 7
            stream.write("{%s,%s,%s,UINT64_C(%d),%s,%d,%s,%d,dir_boundary_weights_%d,%s,%s,%d},\n" % (
                json.dumps(row["case"]), words(declaration["words"]), integers(row["envelope"]), row["key"], words(row["failure"]),
                row["owner"], integers(row["shape"] or [0]), len(row["weights"] or []), index, words(row["terminal"] or [0]), json.dumps(row["diagnostic"]), row["diagnostic_length"]))
        stream.write("};\n")
        stream.write("typedef struct { const char* name; uint32_t operation,mode; int output_valid,failure_valid,error_record,error_span; size_t capacity; const char* diagnostic; size_t diagnostic_length; } DirBoundaryRawRow;\n")
        stream.write("static const DirBoundaryRawRow dir_boundary_raw_rows[] = {\n")
        for row in raw:
            d = spec["production_inputs"][row["case"]]
            stream.write("{%s,%d,%d,%d,%d,%d,%d,%d,%s,%d},\n" % (json.dumps(row["case"]), d["operation"], d["mode"], d["output_valid"], d["failure_valid"], d["error_record"], d["error_span"], d["capacity"], json.dumps(row["diagnostic"]) if row["diagnostic"] is not None else "NULL", row["diagnostic_length"] or 0))
        stream.write("};\n")



def effects_frame_fixture(preset):
    directory = root / "tests/film/effects-frame-fixtures"
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
    assert [len(captured[key]) for key in ("cases", "products", "primitives")] == [347, 72, 57]
    rows = spec["production_cases"][preset]
    assert tuple(row["case"] for row in rows) == EFFECTS_FRAME_CASE_IDS
    assert len(rows) == EFFECTS_FRAME_CASE_COUNT and len({row["case"] for row in rows}) == EFFECTS_FRAME_CASE_COUNT
    assert set(spec["production_inputs"][preset]) == set(EFFECTS_FRAME_CASE_IDS)
    return manifest, payload, context, checked, spec


def expected_effects_frame_header(out):
    _, _, _, _, spec = effects_frame_fixture(arguments.preset)

    def words(values):
        return "{" + ",".join("UINT64_C(" + str(v) + ")" for v in values) + "}"

    with (out / "production-film-expected.h").open("a") as stream:
        stream.write(f"#define FJ_EFFECTS_FRAME_CASE_COUNT {EFFECTS_FRAME_CASE_COUNT}U\n")
        stream.write(f"#define FJ_EFFECTS_FRAME_INPUT_WORD_COUNT {EFFECTS_FRAME_INPUT_WORD_COUNT}U\n")
        stream.write(f"#define FJ_EFFECTS_FRAME_OUTPUT_WORD_COUNT {EFFECTS_FRAME_OUTPUT_WORD_COUNT}U\n")
        stream.write(f"#define FJ_EFFECTS_FRAME_FAILURE_WORD_COUNT {EFFECTS_FRAME_FAILURE_WORD_COUNT}U\n")
        stream.write("typedef struct { const char* name; uint32_t mode; size_t capacity; uint64_t input[FJ_EFFECTS_FRAME_INPUT_WORD_COUNT]; FjStatus status; uint64_t output[FJ_EFFECTS_FRAME_OUTPUT_WORD_COUNT], failure[FJ_EFFECTS_FRAME_FAILURE_WORD_COUNT]; int output_valid, failure_valid, error_valid; const char* diagnostic; } EffectsFrameRow;\n")
        stream.write("static const EffectsFrameRow effects_frame_rows[] = {\n")
        for row in spec["production_cases"][arguments.preset]:
            source = spec["production_inputs"][arguments.preset][row["case"]]
            assert len(source["words"]) == EFFECTS_FRAME_INPUT_WORD_COUNT
            assert row["output"] is None or len(row["output"]) == EFFECTS_FRAME_OUTPUT_WORD_COUNT
            assert row["failure"] is None or len(row["failure"]) == EFFECTS_FRAME_FAILURE_WORD_COUNT
            stream.write("{%s,%d,%d,%s,%s,%s,%s,%d,%d,%d,%s},\n" % (
                json.dumps(row["case"]), source["mode"], source["capacity"], words(source["words"]),
                "{" + ",".join(str(v) for v in row["envelope"]) + "}",
                words(row["output"] or [0]), words(row["failure"] or [0]),
                row["output"] is not None, row["failure"] is not None, row["diagnostic"] is not None,
                json.dumps(row["diagnostic"]) if row["diagnostic"] is not None else "NULL"))
        stream.write("};\n")


def grain_frame_fixture(preset):
    directory = root / "tests/grain/frame-fixtures"
    manifest = directory / "manifest.json"
    spec = json.loads(manifest.read_bytes())
    payload = directory / spec["presets"][preset]["native"]
    checked = directory / spec["checked"]
    for path in (payload, checked, directory / "inputs.json"):
        assert digest(path) == spec["sha256"][path.name]
    captured = json.loads(payload.read_bytes())
    assert captured["preset"] == preset
    assert [len(captured[key]) for key in ("cases", "products", "primitives")] == [153, 44, 50]
    case_ids = tuple(row["case"] for row in spec["production_cases"])
    assert case_ids == GRAIN_FRAME_CASE_IDS, "reviewed complete grain-frame production case membership"
    assert len(case_ids) == GRAIN_FRAME_CASE_COUNT and len(set(case_ids)) == GRAIN_FRAME_CASE_COUNT
    assert set(spec["production_inputs"]) == set(GRAIN_FRAME_CASE_IDS)
    return manifest, payload, checked, captured, spec


def expected_grain_frame_header(out):
    _, _, _, _, spec = grain_frame_fixture(arguments.preset)

    def words(values):
        return "{" + ",".join("UINT64_C(" + str(v) + ")" for v in values) + "}"

    with (out / "production-film-expected.h").open("a") as stream:
        stream.write(f"#define FJ_GRAIN_FRAME_CASE_COUNT {GRAIN_FRAME_CASE_COUNT}U\n")
        stream.write(f"#define FJ_GRAIN_FRAME_INPUT_WORD_COUNT {GRAIN_FRAME_INPUT_WORD_COUNT}U\n")
        stream.write(f"#define FJ_GRAIN_FRAME_CURVE_COUNT {GRAIN_FRAME_CURVE_COUNT}U\n")
        stream.write(f"#define FJ_GRAIN_FRAME_CURVE_SAMPLE_COUNT {GRAIN_FRAME_CURVE_SAMPLE_COUNT}U\n")
        stream.write(f"#define FJ_GRAIN_FRAME_OUTPUT_WORD_COUNT {GRAIN_FRAME_OUTPUT_WORD_COUNT}U\n")
        stream.write(f"#define FJ_GRAIN_FRAME_FAILURE_WORD_COUNT {GRAIN_FRAME_FAILURE_WORD_COUNT}U\n")
        stream.write("typedef struct { const char* name; uint32_t mode; size_t capacity; uint64_t input[FJ_GRAIN_FRAME_INPUT_WORD_COUNT]; uint32_t counts[FJ_GRAIN_FRAME_CURVE_COUNT], curves[FJ_GRAIN_FRAME_CURVE_COUNT][FJ_GRAIN_FRAME_CURVE_SAMPLE_COUNT]; FjStatus status; uint64_t output[FJ_GRAIN_FRAME_OUTPUT_WORD_COUNT], failure[FJ_GRAIN_FRAME_FAILURE_WORD_COUNT]; int output_valid, failure_valid, error_valid; const char* diagnostic; } GrainFrameRow;\n")
        stream.write("static const GrainFrameRow grain_frame_rows[] = {\n")
        for row in spec["production_cases"]:
            source = spec["production_inputs"][row["case"]]
            assert len(source["words"]) == GRAIN_FRAME_INPUT_WORD_COUNT and len(source["curves"]) == GRAIN_FRAME_CURVE_COUNT
            assert all(len(curve) <= GRAIN_FRAME_CURVE_SAMPLE_COUNT for curve in source["curves"])
            assert row["output"] is None or len(row["output"]) == GRAIN_FRAME_OUTPUT_WORD_COUNT
            curves = "{" + ",".join("{" + ",".join("UINT32_C(" + str(v) + ")" for v in (curve or [0])) + "}" for curve in source["curves"]) + "}"
            counts = "{" + ",".join(str(len(curve)) for curve in source["curves"]) + "}"
            stream.write("{%s,%d,%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%s},\n" % (
                json.dumps(row["case"]), source["mode"], source["capacity"], words(source["words"]), counts, curves,
                "{" + ",".join(str(v) for v in row["envelope"]) + "}", words(row["output"] or [0]), words(row["failure"] or [0]),
                row["output"] is not None, row["failure"] is not None, row["diagnostic"] is not None,
                json.dumps(row["diagnostic"]) if row["diagnostic"] is not None else "NULL"))
        stream.write("};\n")

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=["linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"], required=True
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--generate-expected-header", action="store_true",
                        help="write only production-film-expected.h; no native commands or driver access")
    global arguments, build
    arguments = parser.parse_args(argv)
    build = root / "out/build" / arguments.preset
    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    if arguments.generate_expected_header:
        expected_film_header(arguments.artifacts)
        expected_balance_header(arguments.artifacts)
        expected_scanner_correction_header(arguments.artifacts)
        expected_scanner_color_header(arguments.artifacts)
        expected_scanner_descriptor_header(arguments.artifacts)
        expected_scanner_post_header(arguments.artifacts)
        expected_frame_header(arguments.artifacts)
        expected_halation_frame_header(arguments.artifacts)
        expected_diffusion_frame_header(arguments.artifacts)
        expected_grain_frame_header(arguments.artifacts)
        expected_effects_frame_header(arguments.artifacts)
        expected_dir_frame_header(arguments.artifacts)
        expected_dir_boundary_header(arguments.artifacts)
    else:
        execute(arguments)


if __name__ == "__main__":
    main()
