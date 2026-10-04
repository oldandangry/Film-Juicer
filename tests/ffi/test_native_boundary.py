"""Enforce the current native/host and production operation boundaries."""

from pathlib import Path
import json
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = set(json.loads(
    (ROOT / "scripts/source_file_policy.json").read_text(encoding="utf-8")
)["supportedExtensions"])


def native_sources(root):
    return sorted(
        path
        for directory in (root / "native", root / "src/Cuda")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )


def forbidden_uses(paths, pattern):
    findings = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(pattern, source):
            number = source.count("\n", 0, match.start()) + 1
            findings.append(f"{path}:{number}: {match.group().strip()}")
    return findings


HOST_MESSAGES = r"\b(?:DirFailureMessage|sendMessage)\b|OFX::"
CONTEXT_RESET = r"\b(?:cudaDeviceReset|cuDevicePrimaryCtxReset|cuCtxReset)\s*\("
RETIRED_SPECTRAL = r"\b(?:NpySpectraLUT|NpyFloat2D|load_npy_spectra_lut|load_npy_float2d|load_csv_triplets|load_hanatos_spectra_lut|load_arctic2026beta04_spectra_lut|load_mallett2019_basis_npy|sourceElementBytes|hanatosAssetHash|arcticAssetHash|kExpectedDecodedAssetHash|data_file_string)\b|NpyLoader\.h"
RETIRED_ILLUMINANTS_CALIBRATION = r"\b(?:load_csv_pairs|build_curve_from_csv_pinned|build_curve_D65_pinned|build_curve_D55_pinned|build_curve_D50_pinned|build_curve_T_pinned|build_curve_K75P_pinned|build_curve_TH_KG3_pinned|build_curve_TH_KG3_L_pinned|IlluminantFilterAssetSet|IlluminantFilterCurveCacheEntry|NeutralPrintCalibrationSnapshot|NeutralPrintCalibrationCacheState|load_neutral_print_calibration_snapshot|neutral_print_calibration_path|read_file_bytes)\b"
RETIRED_PROFILES = r"\b(?:ProfileAssetStore|ProfileJSONLoader)\b|ProfileAssets\.cpp"


class NativeBoundary(unittest.TestCase):
    def test_executor_has_no_host_message_surface(self):
        self.assertEqual(forbidden_uses(native_sources(ROOT), HOST_MESSAGES), [])

    def test_production_uses_one_admitted_c_record_render_body(self):
        source = (ROOT / "src/mainProcessing.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"JuicerCuda::execute_(?:direct|print)\s*\(")
        self.assertNotIn("cudaStreamSynchronize", source)
        self.assertEqual(source.count("JuicerCuda::project_and_render("), 1)
        self.assertIn("FJ_RENDER_DEFERRED_DIR_ERROR", source)
        self.assertNotIn('find("component=dir")', source)
        self.assertIn('"FilmJuicerDeferredCudaFailure"', source)
        projection = (ROOT / "src/CudaRenderProjection.cpp").read_text(encoding="utf-8")
        self.assertEqual(projection.count("call.render("), 1)
        self.assertNotRegex(projection, r'#include\s+".*(?:tests/|prepared_boundary|prepared_descriptors\.h)')
        boundary = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertEqual(boundary.count("JuicerCuda::execute_prepared_host_data("), 1)
        self.assertEqual(boundary.count("FjRenderOutcome JuicerCuda::NativeCall::render("), 1)
        self.assertIn("call.render(context, frame, submission, prepared", boundary)

    def test_only_accepted_operations_are_defined(self):
        definitions = []
        for path in (ROOT / "native").glob("*.cpp"):
            definitions.extend(re.findall(r"^(?:FjStatus|FjRenderOutcome) (fj_cuda_\w+)\([^;]*?\)\s*\{", path.read_text(encoding="utf-8"), re.MULTILINE))
        self.assertCountEqual(definitions, ["fj_cuda_create", "fj_cuda_inspect", "fj_cuda_render", "fj_cuda_retire_instance", "fj_cuda_shutdown", "fj_cuda_destroy"])

    def test_resource_destruction_does_not_reenter_root(self):
        source = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text(encoding="utf-8")
        self.assertNotIn("ProcessRoot.h", source)
        self.assertNotRegex(source, r"JuicerProcess::|retire_idle_context")
        source = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertNotIn("SF_TEMP_BRIDGE_release_cuda_owner", source)
        self.assertNotIn("shutdown_if_initialized", source)

    def test_native_code_does_not_reset_host_contexts(self):
        paths = native_sources(ROOT)
        paths.append(ROOT / "src/ProcessRoot.cpp")
        self.assertEqual(forbidden_uses(paths, CONTEXT_RESET), [])

    def test_retired_native_profile_authority_stays_deleted(self):
        paths = [
            path for directory in (ROOT / "src", ROOT / "native", ROOT / "cmake")
            for path in directory.rglob("*")
            if path.is_file() and (
                path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake" or path.name == "CMakeLists.txt"
            )
        ]
        paths.append(ROOT / "CMakeLists.txt")
        self.assertEqual(forbidden_uses(paths, RETIRED_PROFILES), [])
        for stem in ("ProfileJSONLoader", "ProfileAssets"):
            self.assertFalse((ROOT / "src" / f"{stem}.cpp").exists())
        self.assertFalse((ROOT / "src/ProfileJSONLoader.h").exists())

    def test_retired_spectral_source_authority_stays_deleted(self):
        paths = [
            path for directory in (ROOT / "src", ROOT / "native", ROOT / "cmake", ROOT / "tests")
            for path in directory.rglob("*")
            if path.is_file() and (
                path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake" or path.name == "CMakeLists.txt"
            )
        ]
        paths.append(ROOT / "CMakeLists.txt")
        self.assertEqual(forbidden_uses(paths, RETIRED_SPECTRAL), [])
        self.assertFalse((ROOT / "src/NpyLoader.h").exists())

    def test_retired_illuminant_and_calibration_authority_stays_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native", ROOT / "cmake", ROOT / "tests")
                 for path in directory.rglob("*") if path.is_file() and
                 (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake")]
        self.assertEqual(forbidden_uses(paths, RETIRED_ILLUMINANTS_CALIBRATION), [])



class NativeBoundaryControls(unittest.TestCase):
    def test_new_nested_sources_and_headers_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("native/new/nested.hpp", "src/Cuda/new/operation.cuh", "src/Cuda/new/body.cpp"):
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("OFX::effect();\ncudaDeviceReset\n();\n", encoding="utf-8")
            paths = native_sources(root)
            self.assertEqual(len(paths), 3)
            for pattern in (HOST_MESSAGES, CONTEXT_RESET):
                self.assertEqual(len(forbidden_uses(paths, pattern)), 3)
            for path in paths:
                path.write_text("execute_prepared_frame();\n", encoding="utf-8")
            for pattern in (HOST_MESSAGES, CONTEXT_RESET):
                self.assertEqual(forbidden_uses(paths, pattern), [])

    def test_retired_authority_and_build_entry_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "CMakeLists.txt"
            for text in ("class ProfileAssetStore {};", '#include "ProfileJSONLoader.h"', "src/ProfileAssets.cpp"):
                source.write_text(text, encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_PROFILES)), 1)
            source.write_text("src/RustAssetBridge.cpp\n", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_PROFILES), [])

    def test_spectral_deletions_reject_definitions_includes_and_fixture_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hpp"
            for text in (
                '#include "NpyLoader.h"', "struct NpySpectraLUT {};", "NpyFloat2D basis;",
                "load_csv_triplets(path);", "load_hanatos_spectra_lut(path);",
                "load_arctic2026beta04_spectra_lut(path);", "load_mallett2019_basis_npy(path);",
                "load_npy_float2d(path);", "load_npy_spectra_lut(path);",
                "sourceElementBytes = 2;", "hanatosAssetHash = 1;", "arcticAssetHash = 2;",
                "kExpectedDecodedAssetHash = 3;", "data_file_string(root, name);",
            ):
                with self.subTest(text=text):
                    source.write_text(text, encoding="utf-8")
                    self.assertEqual(len(forbidden_uses([source], RETIRED_SPECTRAL)), 1)
            source.write_text(
                "ReconstructionLut lut; MallettBasis basis; copy_cmf_triplets(); "
                "build_film_tc_lut(); build_illuminant_curve(rows, label); set_cie_1931_2deg_cmf(x, y, z);",
                encoding="utf-8",
            )
            self.assertEqual(forbidden_uses([source], RETIRED_SPECTRAL), [])

    def test_illuminant_calibration_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hpp"
            for name in ("load_csv_pairs", "build_curve_D65_pinned", "build_curve_TH_KG3_pinned",
                         "build_curve_TH_KG3_L_pinned", "IlluminantFilterAssetSet", "NeutralPrintCalibrationSnapshot",
                         "load_neutral_print_calibration_snapshot", "neutral_print_calibration_path", "read_file_bytes"):
                source.write_text(f"// retired: {name}();\n", encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_ILLUMINANTS_CALIBRATION)), 1)
            source.write_text("Json noise; load_static_noise_payloads(); build_illuminant_curve(rows, label); "
                              "prepare_tungsten_kg3_lens_input(rows, label); copy_csv_pairs(source);", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_ILLUMINANTS_CALIBRATION), [])



if __name__ == "__main__":
    unittest.main()
