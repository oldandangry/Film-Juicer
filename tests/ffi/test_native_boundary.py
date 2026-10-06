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


def color_sources(root):
    return sorted(
        path for directory in (root / "src", root / "native", root / "cmake", root / "tests")
        for path in directory.rglob("*")
        if path.is_file() and (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake")
    ) + [root / "CMakeLists.txt"]


RETIRED_CAT02 = r"\b(?:chromatic_adapt_XYZ_CAT02|build_chromatic_adaptation_matrix|prepare_film_raw_config|mat3_has_only_finite|sanitize_nonnegative_triplet|normalize_triplet_to_unit_y|triplet_has_positive_finite_sum|sanitize_white_or_dwg|whites_approximately_equal)\b"

RETIRED_INPUT_COLOR = r"\b(?:matrix_input_rgb_to_xyz|input_colorspace_white_xyz|kRGB_to_XYZ_DWG|kRGB_to_XYZ_BT2020|kRGB_to_XYZ_ACES2065|kRGB_to_XYZ_sRGB_Rec709|kInputD65WhiteXYZ|kInputAcesWhiteXYZ|gDWG_RGB_to_XYZ|gDWG_XYZ_to_RGB|XYZ_to_DWG_linear|DWG_linear_to_XYZ|decode_BT2020_nonnegative|decode_BT2020_channel|decode_sRGB_nonnegative|decode_sRGB_channel|apply_input_cctf_decoding|convert_input_rgb_to_DWG|convert_input_rgb_to_sRGB_linear|copy_triplet_sanitized|clamp_triplet_nonnegative_inplace|mul_3x3_vec3)\b"
INPUT_MATRIX_METHODS = r"\b(?:Mat3::(?:mul|inverse)|void\s+mul|Mat3\s+inverse)\s*\("

RETIRED_CAT16 = r"\b(?:chromatic_adapt_XYZ_CAT16|build_chromatic_adaptation_matrix_CAT16)\b"


HOST_MESSAGES = r"\b(?:DirFailureMessage|sendMessage)\b|OFX::"
CONTEXT_RESET = r"\b(?:cudaDeviceReset|cuDevicePrimaryCtxReset|cuCtxReset)\s*\("
RETIRED_SPECTRAL = r"\b(?:NpySpectraLUT|NpyFloat2D|load_npy_spectra_lut|load_npy_float2d|load_csv_triplets|load_hanatos_spectra_lut|load_arctic2026beta04_spectra_lut|load_mallett2019_basis_npy|sourceElementBytes|hanatosAssetHash|arcticAssetHash|kExpectedDecodedAssetHash|data_file_string)\b|NpyLoader\.h"
RETIRED_ILLUMINANTS_CALIBRATION = r"\b(?:load_csv_pairs|build_curve_from_csv_pinned|build_curve_D65_pinned|build_curve_D55_pinned|build_curve_D50_pinned|build_curve_T_pinned|build_curve_K75P_pinned|build_curve_TH_KG3_pinned|build_curve_TH_KG3_L_pinned|IlluminantFilterAssetSet|IlluminantFilterCurveCacheEntry|NeutralPrintCalibrationSnapshot|NeutralPrintCalibrationCacheState|load_neutral_print_calibration_snapshot|neutral_print_calibration_path|read_file_bytes)\b"
RETIRED_PROFILES = r"\b(?:ProfileAssetStore|ProfileJSONLoader)\b|ProfileAssets\.cpp"

RETIRED_NOISE = r"\b(?:StbnNoisePayload|WangNoisePayload|StaticNoisePayloadSet|StaticNoiseAssetSet|StaticNoisePayloadCacheState|static_noise_payloads|load_stbn_noise_payload|load_wang_noise_payload|load_static_noise_payloads|ensure_static_noise_assets|load_static_noise_assets|make_static_noise_assets|noise_asset_path|build_static_noise_input|compatibility_data_directory|_staticNoiseOnce|_staticNoiseAssets|_staticNoisePayloadCache|_dataDir)\b"


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

    def test_retired_cat16_producers_stay_deleted(self):
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_CAT16), [])

    def test_retired_cat02_producers_stay_deleted(self):
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_CAT02), [])

    def test_retired_input_color_producers_stay_deleted(self):
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_INPUT_COLOR), [])
        self.assertEqual(forbidden_uses([ROOT / "src/ColorTransforms.h"], INPUT_MATRIX_METHODS), [])

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

    def test_input_color_deletion_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            for relative in ("src/new/input.hpp", "native/new/input.cpp", "tests/ffi/new/setup.c", "cmake/new/input.cmake"):
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                for text in ("matrix_input_rgb_to_xyz(space);", "gDWG_XYZ_to_RGB.mul(xyz, rgb);", "decode_sRGB_channel(value);", "convert_input_rgb_to_DWG(config, rgb, out);", "mul_3x3_vec3(m, v, out);"):
                    source.write_text(text, encoding="utf-8")
                    self.assertEqual(len(forbidden_uses(color_sources(root), RETIRED_INPUT_COLOR)), 1)
                source.write_text("fj_legacy_input_matrices(space, out);\nfj_test_dwg_to_xyz(rgb, out);\nJuicerColor::input_to_dwg(config, rgb, false);\nfloat gDWG_WhitePoint_XYZ[3];\n", encoding="utf-8")
                self.assertEqual(forbidden_uses(color_sources(root), RETIRED_INPUT_COLOR), [])
            owner = root / "src/ColorTransforms.h"
            for text in ("void mul(const float v[3], float out[3]);", "Mat3 inverse(float fallback) const;", "Mat3::mul(v, out);"):
                owner.write_text(text, encoding="utf-8")
                self.assertEqual(len(forbidden_uses([owner], INPUT_MATRIX_METHODS)), 1)
            owner.write_text("struct Mat3 { float m[9]; };\nMat3 make_identity_mat3();\n", encoding="utf-8")
            other = root / "src/OutputColor.h"
            other.write_text("struct Mat3 { void mul(); Mat3 inverse(); };\n", encoding="utf-8")
            cuda = root / "src/Cuda/device.cuh"
            cuda.parent.mkdir(parents=True, exist_ok=True)
            cuda.write_text("decode_BT2020_channel_device(value);\nconvert_input_rgb_to_DWG_device(config, rgb, out);\n", encoding="utf-8")
            self.assertEqual(forbidden_uses([owner], INPUT_MATRIX_METHODS), [])
            self.assertEqual(forbidden_uses(color_sources(root), RETIRED_INPUT_COLOR), [])

    def test_cat16_deletion_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            for relative in ("src/new/nested.hpp", "native/new/body.cpp", "tests/ffi/new/fixture.c", "cmake/new/check.cmake"):
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                for text in ("void chromatic_adapt_XYZ_CAT16();", "build_chromatic_adaptation_matrix_CAT16(whites);"):
                    source.write_text(text, encoding="utf-8")
                    self.assertEqual(len(forbidden_uses(color_sources(root), RETIRED_CAT16)), 1)
                source.write_text("chromatic_adapt_XYZ_CAT02_device(); JuicerColor::cat02_matrix(whites); "
                                  "Mat3 matrix; fj_legacy_cat16_matrix(source, destination, out); "
                                  "JuicerColor::adapt_cat16(xyz, whites);", encoding="utf-8")
                self.assertEqual(forbidden_uses(color_sources(root), RETIRED_CAT16), [])

    def test_cat02_deletion_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            for relative in ("src/new/nested.hpp", "src/Cuda/new/body.cuh", "native/new/body.cpp",
                             "tests/ffi/new/fixture.c", "cmake/new/check.cmake"):
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                for text in ("void chromatic_adapt_XYZ_CAT02();", "build_chromatic_adaptation_matrix(whites);",
                             "prepare_film_raw_config(config);", "mat3_has_only_finite(matrix);",
                             "sanitize_nonnegative_triplet(out, in);", "normalize_triplet_to_unit_y(xyz);",
                             "triplet_has_positive_finite_sum(xyz);", "sanitize_white_or_dwg(in, out);",
                             "whites_approximately_equal(whites);"):
                    source.write_text(text, encoding="utf-8")
                    self.assertEqual(len(forbidden_uses(color_sources(root), RETIRED_CAT02)), 1)
                source.write_text("chromatic_adapt_XYZ_CAT02_device(xyz, whites, out); "
                                  "Mat3 matrix; make_identity_mat3(); JuicerColor::input_matrices(space); "
                                  "JuicerColor::input_to_dwg(config, rgb, false); "
                                  "JuicerColor::cat02_matrix(whites); JuicerColor::adapt_cat02(xyz, whites); "
                                  "fj_legacy_cat16_matrix(source, destination, out);", encoding="utf-8")
                self.assertEqual(forbidden_uses(color_sources(root), RETIRED_CAT02), [])

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
            source.write_text("NoiseSource noise; build_illuminant_curve(rows, label); "
                              "prepare_tungsten_kg3_lens_input(rows, label); copy_csv_pairs(source);", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_ILLUMINANTS_CALIBRATION), [])

    def test_noise_source_and_path_authority_stays_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native", ROOT / "tests/ffi")
                 for path in directory.rglob("*")
                 if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, RETIRED_NOISE), [])
        source = (ROOT / "src/ResourceAssetLibrary.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"nlohmann|\bJson\b|fstream|ifstream|read_bytes")
        projection = (ROOT / "src/CudaRenderProjection.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(projection, r"(?:assets\(\)\.noise|fj_legacy_noise_|NoiseSource)")
        fixture = (ROOT / "tests/ffi/prepared_boundary.cpp").read_text(encoding="utf-8")
        self.assertIn("fj_test_noise_acquire", fixture)
        self.assertNotIn("fj_legacy_noise_", fixture)

    def test_noise_deletion_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hpp"
            for name in ("StbnNoisePayload", "WangNoisePayload", "StaticNoisePayloadSet",
                         "StaticNoiseAssetSet", "StaticNoisePayloadCacheState", "static_noise_payloads",
                         "load_stbn_noise_payload", "load_wang_noise_payload", "load_static_noise_payloads",
                         "build_static_noise_input", "compatibility_data_directory", "_dataDir"):
                source.write_text(f"// retired {name};\n", encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_NOISE)), 1)
            source.write_text("NoiseSource source; StaticNoiseInput view; Library(nativePath); "
                              "fj_legacy_noise_view(owner, out, error); ensure_grain_static_assets_uploaded(view);",
                              encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_NOISE), [])



if __name__ == "__main__":
    unittest.main()
