"""Complete testing-OFF Release isolation, C11 values/ABI and OFX loading gate."""

from pathlib import Path
import argparse
import ctypes
import json
import os
import re
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
    out = arguments.artifacts
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
    forbidden = re.compile(
        "fj_test_|::test_support::|film_bridge::faults::|film::input::testing::|film_input_bridge::(?:BOUNDARY_FAULT|boundary_fault|arm_boundary_fault)|film::test_support::|\\b(?:FAULT|[A-Z_]+_FAULT|FAULT_[A-Z_]+|[A-Z_]+_FAULT_[A-Z_]+)\\b|\\b[A-Z_]+_FACADE(?:_[A-Z_]+)?\\b|ContextDrainTestFailure|TestSupport::|(?:Profile|Catalog|Terminal|ContextDrain|HostAdmission|ExecutorFailure|PinnedUpload|Illuminant|Render|Spectral|TcLut|Color|OutputGamut|Development|Noise)Test::|\\b\\w*Test::|::fixtures::|::output_fixtures::|::faults::|TC_CONSUMED|TC_TRANSFERS|TC_RELEASES|EXPOSURE_CONSUMED|TC_FACADE_CONSUMED|TC_FACADE_RELEASES|ScannerOracle::|pendingCaptureTest|set_pending_capture_test_hook|COPY_METRICS|SCRATCH_CAPACITIES|gamut_bridge::(?:output_)?fixtures::|exposure_bridge::fixtures::|spectral_bridge::(?:inject|arm_fault|clear_fault)"
    )
    strings = [
        b"film input boundary fixture panic",
        b"film development acquire fixture panic",
        b"film development view fixture panic",
        b"film development release fixture panic",
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
    scanned = []

    def scan(path, name, tool, flags):
        text = run(name, [str(llvm / (tool + suffix)), *flags, str(path)], 600)
        match = forbidden.search(text)
        assert match is None, (name, match.group(0) if match else None)
        raw = path.read_bytes()
        assert not any((value in raw for value in strings)), name
        scanned.append(
            {
                "file": str(path),
                "sha256": digest(path),
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
        "fj_legacy_output_color",
        "fj_legacy_film_profile_acquire",
        "fj_legacy_print_profile_acquire",
        "fj_legacy_resolve_route",
    ]:
        assert re.search("\\b" + name + "\\b", symbols), name
    if windows:
        exports = scan(module, "module-exports", "llvm-readobj", ["--coff-exports"])
        names = re.findall("^\\s*Name: (.+)$", exports, re.M)
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
        names = [line.split()[-1] for line in exports.splitlines() if line.strip()]
        scan(module, "module-symbols", "llvm-nm", ["--defined-only", "--demangle"])
    assert sorted(names) == ["OfxGetNumberOfPlugins", "OfxGetPlugin"], names
    probes = [
        ("film", fixtures / "production-film.c.in"),
        ("accepted-output-color", fixtures / "production-output-color.c.in"),
    ]
    for name, source in probes:
        executable = out / (name + suffix)
        target_source = out / (name + ".c")
        target_source.write_bytes(source.read_bytes())
        source = target_source
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
        run(name + "-compile", command, 300)
        run(
            name + "-values",
            [str(executable), *([str(root / "Resources")] if name == "film" else [])],
            120,
        )
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
    (out / "production.json").write_text(
        json.dumps(
            {
                "preset": arguments.preset,
                "build_testing": "OFF",
                "test_support": "OFF",
                "scanned": scanned,
                "exports": sorted(names),
                "hidden_device_load": True,
                "limitation": "CUDA_VISIBLE_DEVICES=-1 checks hidden discovery; an absent installed driver is not established.",
            },
            indent=2,
        )
        + "\n"
    )
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


if __name__ == "__main__":
    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    expected_film_header(arguments.artifacts)
    execute(arguments)
