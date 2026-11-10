"""Enforce the current native/host and production operation boundaries."""

from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import re
import tempfile
import unittest
from unittest.mock import patch

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
        if (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake") and path.is_file()
    ) + [root / "CMakeLists.txt"]


def product_sources(root):
    return sorted(
        path for directory in (root / "src", root / "native", root / "cmake")
        for path in directory.rglob("*")
        if path.is_file() and (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake")
    ) + [root / "CMakeLists.txt"]


RETIRED_CAT02 = r"\b(?:chromatic_adapt_XYZ_CAT02|build_chromatic_adaptation_matrix|prepare_film_raw_config|mat3_has_only_finite|sanitize_nonnegative_triplet|normalize_triplet_to_unit_y|triplet_has_positive_finite_sum|sanitize_white_or_dwg|whites_approximately_equal)\b"

RETIRED_INPUT_COLOR = r"\b(?:matrix_input_rgb_to_xyz|input_colorspace_white_xyz|kRGB_to_XYZ_DWG|kRGB_to_XYZ_BT2020|kRGB_to_XYZ_ACES2065|kRGB_to_XYZ_sRGB_Rec709|kInputD65WhiteXYZ|kInputAcesWhiteXYZ|gDWG_RGB_to_XYZ|gDWG_XYZ_to_RGB|XYZ_to_DWG_linear|DWG_linear_to_XYZ|decode_BT2020_nonnegative|decode_BT2020_channel|decode_sRGB_nonnegative|decode_sRGB_channel|apply_input_cctf_decoding|convert_input_rgb_to_DWG|convert_input_rgb_to_sRGB_linear|copy_triplet_sanitized|clamp_triplet_nonnegative_inplace|mul_3x3_vec3)\b"
INPUT_MATRIX_METHODS = r"\b(?:Mat3::(?:mul|inverse)|void\s+mul|Mat3\s+inverse)\s*\("

RETIRED_CAT16 = r"\b(?:chromatic_adapt_XYZ_CAT16|build_chromatic_adaptation_matrix_CAT16)\b"
RETIRED_ILLUMINANT_MATH = r"\b(?:PlanckBlackbodySample|planck_blackbody|build_blackbody_curve|build_illuminant_curve|build_tungsten_kg3_curve|prepare_tungsten_kg3_lens_input|build_tungsten_kg3_lens_curve|TungstenKg3LensInput|build_curve_equal_energy_pinned|SpectralResampleDetail|scalar_akima_resample|akima_resample_agx|resample_pairs_akima_to_reference_axis|mean_power_normalize)\b"


RETIRED_EXPOSURE_DEFINITIONS = r"\b(?:bool\s+(?:build_hanatos_reconstructed_reference_white|derive_final_sensitivity)|float\s+(?:camera_filter_sample|hanatos_window_sample))\s*\([^;{}]*\)\s*\{"
RETIRED_INPUT_HULL_DEFINITIONS = r"\b(?:bool\s+(?:build_input_compression_hull|build_locus|build_d65_center|curve_has_reference_samples)|void\s+fourier_lowpass|double\s+ray_polygon_distance)\s*\([^;{}]*\)\s*\{"


HOST_MESSAGES = r"\b(?:DirFailureMessage|sendMessage)\b|OFX::"
CONTEXT_RESET = r"\b(?:cudaDeviceReset|cuDevicePrimaryCtxReset|cuCtxReset)\s*\("
RETIRED_SPECTRAL = r"\b(?:NpySpectraLUT|NpyFloat2D|load_npy_spectra_lut|load_npy_float2d|load_csv_triplets|load_hanatos_spectra_lut|load_arctic2026beta04_spectra_lut|load_mallett2019_basis_npy|sourceElementBytes|hanatosAssetHash|arcticAssetHash|kExpectedDecodedAssetHash|data_file_string)\b|NpyLoader\.h"

RETIRED_SPECTRAL_COPIES = (
    r"\b(?:SpectralContext|ReconstructionLut|MallettBasis|SpectralTables|CMFTriplets|"
    r"gShape|gXBar|gYBar|gZBar|gHanatosAvailable|gHanSpectra|gMallettAvailable|gMallettBasis|"
    r"hanatos_available|set_hanatos_available|hanatos_matches_reference_shape|"
    r"arctic_available|set_arctic_available|mallett_available|set_mallett_available|mallett_basis_matches_reference_shape|"
    r"copy_hanatos_lut|copy_arctic_lut|copy_mallett_basis|copy_cmf_triplets|"
    r"load_process_spectral_assets|ensure_bootstrap|_bootstrapOnce|log_spectral_warning|log_resample_failure|"
    r"samples_follow_reference_axis|make_reference_spectral_shape|lock_shape_to_reference_axis|"
    r"cmf_triplets_match_reference_axis|set_cie_1931_2deg_cmf|spectral_tables_view|JuicerSpectral|JuicerExposure)\b"
    r"|\bSpectral::(?:context|FilmTcLut)\b"
    r"|Rust(?:Spectral|Reconstruction|Exposure)Bridge\.(?:cpp|h)|(?:SpectralProcessing|Scanner)\.cpp"
)
RETIRED_ILLUMINANT_COPIES = (
    r"\b(?:CsvSource|CsvRows|IlluminantCurveView|IlluminantFilterCurveSet|copy_csv_pairs|csv_rows|"
    r"illuminant_filter_curves|JuicerIlluminant|IlluminantKeys)\b|Illuminants\.(?:cpp|h)"
)

RETIRED_ILLUMINANTS_CALIBRATION = r"\b(?:load_csv_pairs|build_curve_from_csv_pinned|build_curve_D65_pinned|build_curve_D55_pinned|build_curve_D50_pinned|build_curve_T_pinned|build_curve_K75P_pinned|build_curve_TH_KG3_pinned|build_curve_TH_KG3_L_pinned|IlluminantFilterAssetSet|IlluminantFilterCurveCacheEntry|NeutralPrintCalibrationSnapshot|NeutralPrintCalibrationCacheState|load_neutral_print_calibration_snapshot|neutral_print_calibration_path|read_file_bytes)\b"
RETIRED_PROFILES = r"\b(?:ProfileAssetStore|ProfileJSONLoader|FilmSource|PrintProfileSource|ConversionSlots|FilmFoundationBuildInput|DevelopmentHold|FilmDevelopRecipe)\b|ProfileAssets\.[ch](?:pp)?"

RETIRED_NOISE = r"\b(?:NoiseSource|StbnNoisePayload|WangNoisePayload|StaticNoisePayloadSet|StaticNoiseAssetSet|StaticNoisePayloadCacheState|static_noise_payloads|load_stbn_noise_payload|load_wang_noise_payload|load_static_noise_payloads|ensure_static_noise_assets|load_static_noise_assets|make_static_noise_assets|noise_asset_path|build_static_noise_input|compatibility_data_directory|_staticNoiseOnce|_staticNoiseAssets|_staticNoisePayloadCache|_dataDir)\b"


RETIRED_OUTPUT_GAMUT = r"\b(?:kOutputBoundaryCubeSlack|kOutputBoundaryMaximumChroma|build_output_gamut_transform|output_boundary_contract_hash|build_output_boundary_table|native_rgb_to_oklab|oklab_to_native_rgb|evaluate_cubic|append_interior_derivative_roots|first_channel_exit|build_boundary_cubics|multiply_float_matrix_vector)\b|GamutCompression\.cpp"


RETIRED_OUTPUT_COLOR_DEFINITIONS = (
    r"\b(?:Entry\s+(?:gDWG|gEntries)\b|namespace\s+GeneratedColorSpaces\s*\{)"
    r"|\b(?:void\s+(?:ensure_hashes|applyEncoding)|std::uint64_t\s+hash_entry|"
    r"(?:Matrix3x3|Mat3)\s+(?:dwg_to_output_matrix|mat_mul)|"
    r"const\s+ColorSpaceEntry\s*&\s*(?:get|getDWG)|"
    r"(?:float|double)\s+(?:encode_channel|encode_channel_double|encode_gamma_signed|encode_gammad_signed|encode_sRGB|encode_sRGBd|encode_BT2020|encode_BT2020d|encode_ProPhoto|encode_ProPhotod|encode_DaVinciIntermediate|encode_DaVinciIntermediated|clamp01|clamp01d))"
    r"\s*\([^;{}]*\)\s*\{"
)


RETIRED_FILM_INPUT_DEFINITIONS = (
    r"\b(?:std::uint64_t\s+(?:hash_film_raw_recipe|hash_tc_lut_recipe)|"
    r"std::array<float,\s*3>\s+copy_filter_triplet|"
    r"Spektrafilm::AutoExposureMethod\s+auto_exposure_method_from_index)"
    r"\s*\([^;{}]*\)\s*\{"
)

RETIRED_S5C2_AMBIENT_OWNERS = (
    r"\b(?:JuicerCuda::Owner|borrowed_owner|Root::instance)\b|"
    r"\bJuicerProcess::root\s*\(|"
    r"\b(?:static|inline\s+static)\s+(?:JuicerProcess::)?Root\b|"
    r"\b(?:JuicerProcess::)?Root\s+g_root\b|"
    r"\bHostCleanup\b|"
    r"\b(?:JuicerAssets::(?:Library|AssetBridge)|(?:ResourceAssetLibrary|RustAssetBridge))\b|"
    r"\bFjAssets\s*\*\s*_assets\b|"
    r"\b(?:borrow_rust_assets|release_cached_payloads|detach_assets)\s*\(|"
    r"\b_dataDirectory\b|"
    r"\bRoot\s*\(\s*(?:const\s+)?std::filesystem::path\s*&"
)


def find_llvm_nm():
    return (shutil.which("llvm-nm") if os.name == "nt"
            else shutil.which("llvm-nm-22") or shutil.which("llvm-nm"))


def product_build():
    if selected := os.environ.get("JUICER_PRODUCT_BUILD"):
        return Path(selected)
    build = Path.cwd().parent
    if (build / "CMakeCache.txt").is_file():
        return build
    return ROOT / "out/build" / ("windows-clang-debug" if os.name == "nt" else "linux-debug")


class NativeBoundary(unittest.TestCase):
    def test_explicit_product_build_does_not_fall_back_to_debug(self):
        selected = ROOT / "out/build" / ("windows-clang-release" if os.name == "nt" else "linux-release")
        with (patch.dict(os.environ, {"JUICER_PRODUCT_BUILD": str(selected)}),
              patch.object(Path, "cwd", side_effect=AssertionError("explicit build must take precedence"))):
            self.assertEqual(product_build(), selected)

    def test_prepared_execution_binds_resolved_grain_matrices_and_print_scalars(self):
        header = (ROOT / "native/juicer_cuda_descriptors.h").read_text(encoding="utf-8")
        decoder = (ROOT / "native/juicer_cuda_descriptors.cpp").read_text(encoding="utf-8")
        prepared = (ROOT / "native/juicer_cuda_prepared.cpp").read_text(encoding="utf-8")
        retired = r"\b(?:grainRecipe|FilmRawConfig|PrintExposureRecipe)\b"
        self.assertNotRegex(header + decoder + prepared, retired)
        for text in ("VisualGrainRecipe grainRecipe;", "FilmRawConfig config;", "PrintExposureRecipe exposure;"):
            self.assertRegex(text, retired)
        self.assertNotRegex("VisualGrainPayloadInput grainPayload; float printExposure;", retired)
        self.assertIn("VisualGrainPayloadInput grainPayload", header)
        self.assertIn("coefficients.flags = source.flags & kCoefficientFlags", decoder)
        self.assertIn("descriptor.requiresFullFrame = (source.flags & FJ_GRAIN_FULL_FRAME) != 0", decoder)
        self.assertIn("visual_grain_scratch_shape(coefficients)", decoder)
        root = (ROOT / "src/ProcessRoot.cpp").read_text(encoding="utf-8")
        self.assertNotIn("focusedFilmRawConfig", root)
        for wire, native, retained in (("input_rgb_to_xyz", "inputRgbToXyz", "focusedInputRgbToXyz"),
                                       ("input_xyz_adapt", "inputXyzAdapt", "focusedInputXyzAdapt"),
                                       ("xyz_to_linear_srgb", "xyzToLinearSrgb", "focusedXyzToLinearSrgb")):
            self.assertIn(f"std::copy_n(exposure.{wire}, 9, focused.{native}.begin())", prepared)
            self.assertIn(f"frame._state->{retained} = request.focused.{native}", root)
        self.assertIn("focused.applyInputChromaticAdapt = (exposure.flags & FJ_FILM_ADAPT_INPUT) != 0", prepared)
        self.assertIn("source.print.exposure, source.print.preflash_exposure", prepared)
        self.assertRegex(prepared, r"decode_prepared_descriptors\(source, descriptors, diagnostic\)[\s\S]*execute_prepared\(")
        execution = (ROOT / "src/Cuda/JuicerCudaExecutor.h").read_text(encoding="utf-8")
        self.assertIn("float printExposure = 0.0f", execution)
        self.assertIn("float preflashExposure = 0.0f", execution)
        boundary = (ROOT / "tests/ffi/prepared_boundary.cpp").read_text(encoding="utf-8")
        model = r"\b(?:RenderRecipe|FocusedRenderPayload|encode_film|encode_print|build_focused_route_resource_input|build_print_resource_input)\b"
        self.assertNotRegex(boundary, model)
        for text in ("RenderRecipe recipe;", "FocusedRenderPayload payload;", "encode_film()", "encode_print()",
                     "build_focused_route_resource_input()", "build_print_resource_input()"):
            self.assertRegex(text, model)
        self.assertIn("bind_state(state, prepared)", boundary)
        self.assertIn("PreparedFrameFixture::with_prepared_frame_inputs(", boundary)
        self.assertIn("encode_prepared_descriptors(inputs.descriptors, prepared)", boundary)
        binding = r"check_frame_bindings\([^;]*&state,\s*inputs\.hostPreparation(?:,|\))"
        self.assertRegex(boundary, binding)
        for invalid in ("check_frame_bindings(cuda, frame, &state, nullptr, caller)",
                        "check_frame_bindings(cuda, frame, nullptr, inputs.hostPreparation, caller)",
                        "other_helper(&state, inputs.hostPreparation, caller)"):
            self.assertNotRegex(invalid, binding)
        self.assertIn("&error, inputs.hostPreparation)", boundary)
        raw = (ROOT / "tests/ffi/prepared_raw_render_fixture.h").read_text(encoding="utf-8")
        self.assertIn("const FjHostPreparation* hostPreparation = nullptr", raw)
        self.assertIn("render_with_borrowed_preparation(&consumer, hostPreparation, error)", raw)
        self.assertIn("fj_test_with_host_preparation(&consumer, &PreparedRawRenderFixture::render_with_borrowed_preparation, error)", raw)
        self.assertIn("fj_test_execute_prepared_c(consumer->access", raw)
        build = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("juicer_illuminant_preflash_root", build)
        for provider in ("juicer_render_test_objects", "juicer_executor_failure_test_objects", "juicer_pinned_upload_test_objects"):
            self.assertIn(provider, build)
        executor_objects = build.split("add_library(juicer_executor_failure_test_objects OBJECT", 1)[1].split("add_test(NAME Ofx.Gpu.NoiseSourceLifetime", 1)[0]
        self.assertNotIn("JUICER_NOISE_TEST_HOOK", executor_objects)
        self.assertNotIn("../src/RustAssetBridge.cpp", executor_objects)
        self.assertIn("JUICER_EXECUTOR_FAILURE_TEST_HOOK=1", executor_objects)

    def test_prepared_state_oracle_uses_frozen_expectations_and_raw_cache_holds(self):
        source = (ROOT / "tests/ffi/prepared_state_test.cpp").read_text(encoding="utf-8")
        retired = r"prepared_state_native|SF_TEMP_BRIDGE_native_prepared_view|FocusedRenderStateBuildProduct|build_(?:direct|print)_render_state_product|--capture-state-projection"
        self.assertNotRegex(source, retired)
        for text in ('#include "prepared_state_native.h"', "SF_TEMP_BRIDGE_native_prepared_view()",
                     "FocusedRenderStateBuildProduct product;", "build_direct_render_state_product()",
                     "build_print_render_state_product()", "--capture-state-projection"):
            self.assertRegex(text, retired)
        self.assertNotRegex('PreparedStateOracle::compare(row.at("expected"), actual);', retired)
        for name in ("prepared_state_native.h", "prepared_state_native.cpp"):
            self.assertFalse((ROOT / "tests/ffi" / name).exists())
        self.assertNotIn("prepared_state_native.cpp", (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8"))
        for text in ('PreparedStateOracle::compare(row.at("expected"), PreparedStateOracle::view(actual))',
                     'oracle_row("NegativePrintMallett").at("expected")', "--check-state-projection",
                     "fj_test_assets_input_hull_acquire", "fj_test_assets_output_boundary_acquire",
                     "input.entry_identity == actual.input_hull.entry_identity", "input.xy.data == actual.input_hull.xy.data",
                     "output.entry_identity == actual.output_boundary.entry_identity", "output.cmax.data == actual.output_boundary.cmax.data",
                     "observe_encoding_relation(first.output_boundary.entry_identity == second.output_boundary.entry_identity)"):
            self.assertIn(text, source)

    def test_workspace_layout_owns_coupled_storage_and_scanner_inputs_are_closed(self):
        root = (ROOT / "src/ProcessRoot.cpp").read_text()
        header = (ROOT / "src/ProcessRoot.h").read_text()
        core = (ROOT / "src/Cuda/ResourceManager/JuicerCudaResourceCore.h").read_text()
        executor = (ROOT / "src/Cuda/JuicerCudaExecutor.cpp").read_text()
        retired = r"\b(?:WorkspaceRequest|derive_workspace_request|grain_reuses_spatial_dir_raw|make_workspace_scratch_request_descriptor)\b|\w*UseFocusedSplit|\w*UseFusedScannerPostSpatialDirHandoff|\w*CaptureDensityReady"
        self.assertNotRegex(root + header + executor, retired)
        for text in ("WorkspaceRequest request;", "derive_workspace_request()", "grain_reuses_spatial_dir_raw()",
                     "make_workspace_scratch_request_descriptor()", "directUseFocusedSplit", "printUseFusedScannerPostSpatialDirHandoff", "directCaptureDensityReady"):
            self.assertRegex(text, retired)
        self.assertIn("class WorkspaceLayout final", core)
        self.assertIn("const ScratchRequestDescriptor& allocation() const noexcept", core)
        self.assertIn("WorkspaceLayout::create", root)
        self.assertIn("WorkspaceDerivation requestedWorkspace", root)
        for name in ("admittedLayout", "postFrameLayout"):
            self.assertIn("std::optional<WorkspaceLayout> " + name, root)
        self.assertIn("with_dir_layout", root)
        self.assertIn("std::variant<MonolithicScan, FusedDirScanner, MaterializedDensity>", executor)
        self.assertEqual(executor.count("const ScannerInput scannerInput ="), 2)
        self.assertEqual(executor.count("dirBuild = build_"), 2)

    def test_synthetic_reference_has_one_safe_owner_and_native_completion_only(self):
        native = [path for directory in (ROOT / "src", ROOT / "native")
                  for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(native, r"\bsample_synthetic_density(?:_for_test)?\s*\("), [])
        source = (ROOT / "src/RenderValues.cpp").read_text()
        retired = (r"\b(?:struct|class)\s+(?:RenderRecipe|ProfileRoute|ScannerOutputRecipe|PrintRecipe|"
                   r"PrintFilterRecipe|PrintExposureRecipe|PrintIlluminantRecipe|PrintDevelopRecipe)\b|"
                   r"\busing\s+(?:::)?(?:RenderRecipe|ProfileRoute|ScannerOutputRecipe|PrintRecipe|"
                   r"PrintFilterRecipe|PrintExposureRecipe|PrintIlluminantRecipe|PrintDevelopRecipe)\b|"
                   r"\b(?:Direct|Print)RecipeBuild(?:Input|Result)\b|"
                   r"\b(?:make_render_recipe|build_film_foundation|build_direct_render_recipe|"
                   r"build_print_render_recipe|finish_synthetic_reference_recipes|hash_profile_route|"
                   r"hash_print_recipe|hash_final_render_recipe)\s*\(")
        self.assertEqual(forbidden_uses(color_sources(ROOT), retired), [])
        for text in ("struct RenderRecipe {};", "struct ProfileRoute {};", "using ::RenderRecipe;", "using ::ProfileRoute;",
                     "struct ScannerOutputRecipe {};", "struct PrintRecipe {};", "struct PrintFilterRecipe {};",
                     "struct PrintExposureRecipe {};", "struct PrintIlluminantRecipe {};", "struct PrintDevelopRecipe {};",
                     "using ::ScannerOutputRecipe;", "using ::PrintRecipe;", "using ::PrintFilterRecipe;",
                     "using ::PrintExposureRecipe;", "using ::PrintIlluminantRecipe;", "using ::PrintDevelopRecipe;",
                     "DirectRecipeBuildInput input;", "DirectRecipeBuildResult result;",
                     "PrintRecipeBuildInput input;", "PrintRecipeBuildResult result;", "make_render_recipe()",
                     "build_film_foundation()", "build_direct_render_recipe()", "build_print_render_recipe()",
                     "finish_synthetic_reference_recipes()", "hash_profile_route()", "hash_print_recipe()",
                     "hash_final_render_recipe()"):
            self.assertRegex(text, retired)
        for text in ('#include "RenderValues.h"', "FilmFoundationBuildInput input;", "PrintResourceInput input;",
                     "struct PrintBalanceRecipe {};", "struct DichroicFilterRecipe {};", "struct CmyCcTriplet {};"):
            self.assertNotRegex(text, retired)
        for name in ("RenderRecipe.h", "RenderRecipe.cpp", "RustFilmRecipeBridge.h", "RustFilmRecipeBridge.cpp"):
            self.assertFalse((ROOT / "src" / name).exists(), name)
        cmake = (ROOT / "cmake/JuicerSources.cmake").read_text() + (ROOT / "tests/CMakeLists.txt").read_text()
        retired_hooks = r"\b(?:JUICER_FILM_RECIPE_TEST_HOOK|juicer_film_recipe_test_objects)\b"
        self.assertNotRegex(cmake, retired_hooks)
        for token in ("JUICER_FILM_RECIPE_TEST_HOOK", "juicer_film_recipe_test_objects"):
            self.assertRegex(token, retired_hooks)
        abi = (ROOT / "tests/ffi/film_recipe_abi_c.c").read_text()
        raw = "\n".join(path.read_text(encoding="utf-8")
                        for path in sorted((ROOT / "rust/film-juicer-plugin/src").rglob("*.rs")))
        for family in ("spatial_optics", "print_illuminant", "print_preflash"):
            legacy = "fj_legacy_" + family + "_prepare"
            self.assertEqual(forbidden_uses(native, r"\b" + legacy + r"\s*\("), [])
            retired_export = r"\b" + legacy + r"\b"
            self.assertNotRegex(raw, retired_export)
            for mutation in ("pub extern C fn " + legacy + "() {}",
                             '#[export_name = "' + legacy + '"] fn replacement() {}'):
                self.assertRegex(mutation, retired_export)
            self.assertIn("fj_test_" + family + "_c_prepare", abi)
            self.assertIn("return fj_test_" + family + "_prepare(", abi)
        retired_values = r"\b(?:struct|class|enum\s+class)\s+(?:ProfileDigest|DensityMedium|DensityBoundsSource|NeutralCalibrationStatus|PrintNormalizationMode|DirCouplersRecipe|VisualGrainRecipe|GrainContract|DensityBoundsRecipe|DichroicFilterRecipe|PrintBalanceRecipe)\b"
        self.assertEqual(forbidden_uses(native, retired_values), [])
        for name in ("ProfileDigest", "DensityMedium", "DensityBoundsSource", "NeutralCalibrationStatus", "PrintNormalizationMode", "DirCouplersRecipe", "VisualGrainRecipe", "GrainContract", "DensityBoundsRecipe", "DichroicFilterRecipe", "PrintBalanceRecipe"):
            self.assertRegex("struct " + name + " {};", retired_values)
        values = (ROOT / "src/RenderValues.h").read_text()
        self.assertIn("struct CmyCcTriplet", values)
        for name in ("PhysicalEffectsControls", "SpatialOpticsControls"):
            pattern = r"\b(?:struct|class)\s+" + name + r"\b"
            self.assertEqual(forbidden_uses(product_sources(ROOT), pattern), [])
            self.assertRegex("struct " + name + " {};", pattern)
            self.assertNotRegex(values, pattern)
        inputs = (ROOT / "tests/ffi/preparation_fixture.h").read_text()
        self.assertIn("FjPreparationParams values", inputs)
        fixture = (ROOT / "tests/ffi/film_reference_test.inc").read_text()
        preparers = (("fj_test_film_reference_prepare", "facade"),
                     ("fj_test_film_reference_c_prepare", "c11"))
        self.subtest_cases = tuple(f"caller/{caller}" for _, caller in preparers)
        for function, caller in preparers:
            with self.subTest(case=f"caller/{caller}"):
                self.assertIsNotNone(
                    re.search(r'\{\s*' + function + r',\s*"' + caller + r'"\s*\}', fixture),
                    f"film-reference fixture must bind its {caller} consumer")
        self.assertIn("for (const auto& [prepare, api] : preparers)", fixture)
        self.assertIn("prepare(development.get(), &inputs.wire", fixture)
        self.assertIn('project(out) == row.at("reference")', fixture)
        recipe = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text()
        complete = recipe.split("pub fn prepare(", 1)[1].split("\n}\n", 1)[0]
        sequence = ["trace_stage!(PrintMainIlluminant)", "reference::Reference::new(",
                    "print::Balance::new(", "trace_stage!(ScannerPayload)", "Ok(RenderState {"]
        positions = [complete.index(value) for value in sequence]
        self.assertEqual(positions, sorted(positions))
        state = "\n".join(path.read_text(encoding="utf-8") for path in product_sources(ROOT))
        self.assertNotIn("developmentHold", state)
        self.assertNotIn("prepare_published_state(", state)
        for old in ("src/JuicerState.cpp", "src/JuicerState.h", "src/ProcessRuntime.cpp", "src/ProcessRuntime.h"):
            self.assertFalse((ROOT / old).exists(), old)
        instance = (ROOT / "rust/film-juicer-plugin/src/instance.rs").read_text()
        self.assertIn("publication: Arc<state::Publication>", instance)
        self.assertIn("self.publication.admit(|params| {", instance)
        self.assertIn("let outcome = recipe::prepare(assets, params);", instance)
        self.assertNotIn("DevelopmentHold", state)
        self.assertNotIn("film_reference_samples", state)
        reader = (ROOT / "tests/ffi/illuminant_calibration_test.cpp").read_text()
        self.assertNotIn("foundationReference", reader)
        self.assertIn("fj_test_exposure_sensitivity(&foundationInput", reader)
        self.assertIn("foundationInput.reference_illuminant = shared.tungsten_kg3_lens", reader)
        self.assertIn("foundationSensitivity.mallett_green_scale", reader)
        self.assertNotIn("result.developmentHold = std::move(film.developmentHold)", source)
        preparation = (ROOT / "rust/film-juicer-plugin/src/preparation.rs").read_text()
        self.assertIn("let completed = recipe::prepare(assets, params)", preparation)
        self.assertIn("FjPreparedState::from_state(Arc::new(completed))", preparation)
        self.assertIn("fn project_state(owner: &FjPreparedState)", preparation)
        self.assertNotIn("pub(crate) fn project_state(", preparation)
        owner = preparation.split("pub(crate) struct FjPreparedState {", 1)[1].split("\n}", 1)[0]
        self.assertIn("state: Arc<recipe::RenderState>", owner)
        public_state = r"\bpub(?:\([^)]*\))?\s+state\s*:"
        self.assertNotRegex(owner, public_state)
        self.assertRegex("pub(crate) state: Arc<recipe::RenderState>", public_state)

        contents_owner = recipe.split("pub struct RenderState {", 1)[1].split("\n}", 1)[0]
        self.assertIn("contents: Vec<Contents>", contents_owner)
        self.assertNotRegex(contents_owner, r"\bpub(?:\([^)]*\))?\s+contents\s*:")
        contents = recipe.split("struct Contents {", 1)[1].split("\n}", 1)[0]
        self.assertNotRegex(contents, r"\bpub(?:\([^)]*\))?\s+")
        self.assertIn(".try_reserve_exact(if reject_reservation { usize::MAX } else { 1 })", recipe)
        self.assertIn(".map_err(PrepareError::ContentsAllocation)?;", recipe)
        self.assertIn("backing.push(contents);", recipe)

    def test_film_input_construction_and_hash_definitions_stay_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native")
                 for path in directory.rglob("*")
                 if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, RETIRED_FILM_INPUT_DEFINITIONS), [])
        for rejected in (
            "std::uint64_t hash_film_raw_recipe(const FilmRawRecipe& recipe) {}",
            "std::uint64_t hash_tc_lut_recipe(const FilmRawRecipe& recipe) {}",
            "std::array<float, 3> copy_filter_triplet(const std::array<double, 3>& values, bool active) {}",
            "Spektrafilm::AutoExposureMethod auto_exposure_method_from_index(int index) {}",
        ):
            self.assertRegex(rejected, RETIRED_FILM_INPUT_DEFINITIONS)
            self.assertNotRegex(rejected.replace(" {}", ";"), RETIRED_FILM_INPUT_DEFINITIONS)
        self.assertNotRegex("bool auto_exposure_method_index_valid(int index) {}", RETIRED_FILM_INPUT_DEFINITIONS)
        recipe = (ROOT / "src/RenderValues.cpp").read_text()
        self.assertNotIn("build_film_foundation(", recipe)
        self.assertNotIn("auto_exposure_method_index_valid(", recipe)
        self.assertFalse((ROOT / "src/RustFilmInputBridge.cpp").exists())
        cmake = (ROOT / "cmake/JuicerSources.cmake").read_text() + (ROOT / "tests/CMakeLists.txt").read_text()
        self.assertNotIn("RustFilmInputBridge.cpp", cmake)
        self.assertNotIn("juicer_film_input_test_objects", cmake)
        self.assertNotIn("FilmFoundationBuildInput", (ROOT / "src/RenderValues.h").read_text())
        self.assertFalse((ROOT / "src/RustExposureBridge.cpp").exists())
        fixture = (ROOT / "tests/ffi/film_input_test.cpp").read_text()
        self.assertIn("fj_test_film_input_prepare", fixture)
        self.assertIn("selectedLogExposureWords", fixture)
        self.assertNotIn("InputTest::translate_failure", fixture)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text()
        self.assertIn("pub fn prepare(assets: &Assets, params: &Params) -> Result<RenderState, PrepareError>", complete)
        self.assertIn("let input = input::Recipe::new(", complete)
        calls = []
        for path in (ROOT / "rust/film-juicer-core/src").rglob("*.rs"):
            production = path.read_text().split("#[cfg(test)]", 1)[0]
            if "InputColor::from_matrices(" in production:
                calls.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(calls, ["rust/film-juicer-core/src/film/input.rs"])

    def test_output_color_authority_is_retired_and_packing_only_binds_values(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native")
                 for path in directory.rglob("*")
                 if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, RETIRED_OUTPUT_COLOR_DEFINITIONS), [])
        for rejected in (
            "inline Entry gDWG = {};", "inline Entry gEntries[9] = {};",
            "namespace GeneratedColorSpaces {", "void ensure_hashes() {}",
            "std::uint64_t hash_entry(const Entry& e) {}",
            "const ColorSpaceEntry& getDWG() {}", "const ColorSpaceEntry& get(Tag cs) {}",
            "Matrix3x3 dwg_to_output_matrix(ColorSpace cs) {}", "Mat3 mat_mul(const float a[9], const float b[9]) {}",
            "void applyEncoding(const Params& p, double rgb[3]) {}", "void applyEncoding(const Params& p, float rgb[3]) {}",
            "float encode_channel(const Params& p, float x) {}", "double encode_channel_double(const Params& p, double x) {}",
            "float clamp01(float x) {}", "double clamp01d(double x) {}",
        ):
            self.assertRegex(rejected, RETIRED_OUTPUT_COLOR_DEFINITIONS)
        for result, name in (("float", "encode_gamma_signed"), ("double", "encode_gammad_signed"),
                             ("float", "encode_sRGB"), ("double", "encode_sRGBd"),
                             ("float", "encode_BT2020"), ("double", "encode_BT2020d"),
                             ("float", "encode_ProPhoto"), ("double", "encode_ProPhotod"),
                             ("float", "encode_DaVinciIntermediate"), ("double", "encode_DaVinciIntermediated")):
            self.assertRegex(f"{result} {name}({result} x) {{ return x; }}", RETIRED_OUTPUT_COLOR_DEFINITIONS)
            self.assertNotRegex(f"{result} {name}({result} x);", RETIRED_OUTPUT_COLOR_DEFINITIONS)
        for retained in ("enum class ColorSpace : std::uint8_t {};", "struct Params {};",
                         "constexpr ColorSpace colorSpaceFromIndex(int index) {}", "const char* kColorSpaceLabels[] = {};",
                         "FjOutputColorPreparation JuicerColor::output_color(ColorSpace space) {}"):
            self.assertNotRegex(retained, RETIRED_OUTPUT_COLOR_DEFINITIONS)
        executor = (ROOT / "src/Cuda/JuicerCudaExecutor.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(executor, r"JuicerColor::output_color|fj_legacy_output_color|dwg_to_output_matrix|GeneratedColorSpaces")
        self.assertIn("color.preparedEncoding", executor)
        self.assertFalse((ROOT / "src/Scanner.cpp").exists())
        scanner = (ROOT / "src/Scanner.h").read_text(encoding="utf-8")
        self.assertNotIn("hash_color_runtime(", scanner)
        definition = r"\bColorRuntime\s+build_color_runtime\s*\([^;{}]*\)\s*\{"
        self.assertNotRegex(scanner, definition)
        self.assertRegex("ColorRuntime build_color_runtime() {}", definition)
        self.assertNotRegex("ColorRuntime build_color_runtime();", definition)
        core = (ROOT / "rust/film-juicer-core/src/scanner/color.rs").read_text(encoding="utf-8")
        self.assertLess(core.index("if input.illuminant_hash == 0"), core.index("color::output_color"))
        self.assertLess(core.index("if output.hash() == 0"), core.index("color::checked_cat02_matrix"))
        color = (ROOT / "rust/film-juicer-core/src/color.rs").read_text(encoding="utf-8")
        checked = color.split("pub fn checked_cat02_matrix", 1)[1].split("\n}", 1)[0]
        self.assertIn("cat02_matrix(", checked)
        for suffix in ("cpp", "h"):
            self.assertFalse((ROOT / f"src/RustScannerBridge.{suffix}").exists())
        scanner_recipe = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        self.assertIn("scanner::Color::new(", scanner_recipe)
        self.assertIn("color::output_color(space)", scanner_recipe)
        # The executor also binds the existing frame-local print-glare seed.
        # Retire color hashing everywhere while retaining that exact consumer.
        glare_seed = """Hash::hash_uint64_values(
                {recipeHash,
                 snapshot.frameToken.value,
                 static_cast<std::uint64_t>(srcBounds.x1),
                 static_cast<std::uint64_t>(srcBounds.y1)})"""
        self.assertEqual(executor.count(glare_seed), 1)
        self.assertNotRegex(executor.replace(glare_seed, ""), r"JuicerColor::|hash_color_runtime|Hash::hash_uint64_values")
        prepared = (ROOT / "native/juicer_cuda_descriptors.cpp").read_text(encoding="utf-8")
        self.assertIn("color.color_space > FJ_OUTPUT_REC709", prepared)
        oracle = (ROOT / "tests/scanner/output_encoding_oracle.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(oracle, r"JuicerColor::|fj_(?:test(?:_boundary)?|legacy)_output_color|GeneratedColorSpaces")
        self.assertIn('"output_encoding_native.inc"', oracle)

    def test_native_output_gamut_math_is_retired(self):
        self.assertFalse((ROOT / "src/GamutCompression.cpp").exists())
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_OUTPUT_GAMUT), [])
        header = (ROOT / "src/GamutCompression.h").read_text(encoding="utf-8")
        for survivor in ("kOklabXyzToLms", "kOklabLmsToXyz", "kOklabLmsRootToLab", "kOklabLabToLmsRoot", "sample_output_boundary", "compress_output_oklch"):
            self.assertIn(survivor, header)
        fixture = (ROOT / "tests/ffi/gamut_fixture.h").read_text(encoding="utf-8")
        fixture_owner = r"\b(?:struct|class)\s+OutputBoundaryTable\s*\{"
        self.assertRegex(fixture, fixture_owner)
        self.assertEqual(forbidden_uses(product_sources(ROOT), fixture_owner), [])
        self.assertRegex("struct OutputBoundaryTable {};", fixture_owner)
        self.assertNotRegex("struct OutputBoundaryTable;", fixture_owner)
        self.assertIn("struct OutputGamutTransform", header)
        self.assertIn("struct OutputCompressionView", header)
        for removed in ("build_output_gamut_transform", "output_boundary_contract_hash", "build_output_boundary_table", "native_rgb_to_oklab", "oklab_to_native_rgb", "evaluate_cubic", "append_interior_derivative_roots", "first_channel_exit", "build_boundary_cubics", "multiply_float_matrix_vector"):
            self.assertRegex(f"void {removed}() {{}}", RETIRED_OUTPUT_GAMUT)
        self.assertRegex("src/GamutCompression.cpp", RETIRED_OUTPUT_GAMUT)
        self.assertNotRegex("compress_output_oklch(); sample_output_boundary(); JuicerGamut::output_boundary();", RETIRED_OUTPUT_GAMUT)

    def test_retired_tc_math_definitions_stay_deleted(self):
        self.assertFalse((ROOT / "src/SpectralProcessing.cpp").exists())
        tc = r"\b(?:bool\s+(?:build_film_tc_lut|build_blurred_hanatos_spectra|build_gaussian_kernel|projected_white_to_tc)|(?:float|int|void|std::array<\s*float\s*,\s*3\s*>)\s+(?:sample_film_tc_lut|sample_mitchell_2d|sample_integrated|sample_spectrum|mitchell_weight|mitchell_coordinate|mitchell_reflect_index|gaussian_reflect_index|eval_hanatos_surface))\s*\([^;{}]*\)\s*\{"
        remap = r"\b(?:bool\s+(?:compress_input_xy|remap_film_tc_lut_for_input_compression)|(?:float|double)\s+(?:bilinear_sample|reinhard_knee))\s*\([^;{}]*\)\s*\{"
        self.assertEqual(forbidden_uses(color_sources(ROOT), tc), [])
        self.assertEqual(forbidden_uses(color_sources(ROOT), remap), [])
        for body in ("bool build_film_tc_lut() {}", "float eval_hanatos_surface() {}", "int gaussian_reflect_index() {}", "void mitchell_coordinate() {}"):
            self.assertRegex(body, tc)
        for body in ("bool compress_input_xy() {}", "bool remap_film_tc_lut_for_input_compression() {}", "double reinhard_knee() {}", "float bilinear_sample() {}"):
            self.assertRegex(body, remap)
        self.assertNotRegex("bool build_film_tc_lut();", tc)
        self.assertNotRegex("double ray_polygon_distance() {}", remap)
        core = (ROOT / "rust/film-juicer-core/src/reconstruction.rs").read_text(encoding="utf-8")
        for name in ("checked_reference_white", "build_tc_lut_checked", "sample_tc_lut_checked"):
            self.assertIn(f"pub fn {name}(", core)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        self.assertIn("reconstruction::checked_reference_white(", complete)
        self.assertIn("let observer = Observer::acquire(assets)?;", complete)

    def test_native_input_hull_math_is_retired(self):
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_INPUT_HULL_DEFINITIONS), [])
        self.assertFalse((ROOT / "src/GamutCompression.cpp").exists())
        self.assertNotIn("build_input_compression_hull", (ROOT / "src/GamutCompression.h").read_text(encoding="utf-8"))
        fixture = (ROOT / "tests/ffi/gamut_fixture.h").read_text(encoding="utf-8")
        fixture_owner = r"\b(?:struct|class)\s+InputCompressionHull\s*\{"
        self.assertRegex(fixture, fixture_owner)
        self.assertEqual(forbidden_uses(product_sources(ROOT), fixture_owner), [])
        self.assertRegex("struct InputCompressionHull {};", fixture_owner)
        self.assertNotRegex("struct InputCompressionHull;", fixture_owner)
        library = (ROOT / "tests/ffi/test_asset_library.cpp").read_text(encoding="utf-8")
        self.assertNotIn("JuicerGamut::input_hull(", library)
        self.assertIn("return _bridge.input_hull();", library)
        bridge = (ROOT / "tests/ffi/test_asset_bridge.cpp").read_text(encoding="utf-8")
        self.assertEqual(bridge.count("fj_test_assets_input_hull_acquire("), 1)
        core = (ROOT / "rust/film-juicer-core/src/gamut.rs").read_text(encoding="utf-8")
        self.assertEqual(core.count("fn ray_polygon_distance<"), 1)

    def test_input_hull_deletion_controls(self):
        for result, name in (("bool", "build_input_compression_hull"), ("bool", "build_locus"),
                             ("bool", "build_d65_center"), ("bool", "curve_has_reference_samples"),
                             ("void", "fourier_lowpass"), ("double", "ray_polygon_distance")):
            self.assertRegex(f"{result} {name}(const auto& source) {{ return {{}}; }}", RETIRED_INPUT_HULL_DEFINITIONS)
            self.assertNotRegex(f"{result} {name}(const Source&);", RETIRED_INPUT_HULL_DEFINITIONS)
            self.assertNotRegex(f"{name}(source);", RETIRED_INPUT_HULL_DEFINITIONS)
        self.assertNotRegex("double evaluate_cubic(const Cubic& c, double x) { return x; }", RETIRED_INPUT_HULL_DEFINITIONS)

    def test_retired_exposure_definitions_stay_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native")
                 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, RETIRED_EXPOSURE_DEFINITIONS), [])
        for declaration in ("bool derive_final_sensitivity() {}", "float camera_filter_sample(float x) { return x; }",
                            "float hanatos_window_sample(float x) { return x; }",
                            "bool build_hanatos_reconstructed_reference_white() {}"):
            self.assertRegex(declaration, RETIRED_EXPOSURE_DEFINITIONS)
        for survivor in ("float sample_spectrum() {}", "bool build_gaussian_kernel() {}",
                         "float hanatos_window_sample_for_test(float x) { return x; }",
                         "JuicerExposure::reference_white(x, y, z, out, diagnostic);"):
            self.assertNotRegex(survivor, RETIRED_EXPOSURE_DEFINITIONS)
        recipe = (ROOT / "src/RenderValues.cpp").read_text(encoding="utf-8")
        self.assertEqual(recipe.count("std::uint64_t hash_tc_lut_recipe("), 0)
        self.assertEqual(recipe.count("JuicerExposure::prepare_sensitivity("), 0)
        family = (ROOT / "rust/film-juicer-core/src/film/input.rs").read_text(encoding="utf-8")
        self.assertIn("exposure::prepare_sensitivity(", family)
        self.assertEqual(forbidden_uses(product_sources(ROOT), r"JuicerExposure::reference_white\s*\("), [])
        self.assertFalse((ROOT / "src/JuicerState.cpp").exists())
        retired_lookup = r"\b(?:JUICER_ASSET_LOOKUP_TEST_HOOK|juicer_asset_lookup_test_objects)\b"
        self.assertEqual(forbidden_uses(color_sources(ROOT) + [ROOT / "tests/CMakeLists.txt"], retired_lookup), [])
        for token in ("JUICER_ASSET_LOOKUP_TEST_HOOK", "juicer_asset_lookup_test_objects"):
            self.assertRegex(token, retired_lookup)
        self.assertNotRegex("JuicerAssetLookupTests fj_test_exposure_window_sample", retired_lookup)
        self.assertNotIn('"juicer_test_api.h"', recipe)
        self.assertNotIn("hanatos_window_sample_for_test(", recipe)
        consumer = (ROOT / "tests/ffi/asset_lookup_test.cpp").read_text(encoding="utf-8")
        self.assertIn('#include "asset_lookup_test.h"', consumer)
        self.assertIn('#include "juicer_test_api.h"', consumer)
        sample = consumer.split("    float hanatos_window_sample_for_test(", 1)[1].split("\n    }", 1)[0]
        self.assertIn("fj_test_exposure_window_sample(wavelength, {params.data(), params.size()}, &result, &error)", sample)

    def test_native_illuminant_math_is_retired(self):
        self.assertEqual(forbidden_uses(color_sources(ROOT), RETIRED_ILLUMINANT_MATH), [])
    def test_executor_has_no_host_message_surface(self):
        self.assertEqual(forbidden_uses(native_sources(ROOT), HOST_MESSAGES), [])

    def test_production_uses_one_admitted_c_record_render_body(self):
        source = (ROOT / "rust/film-juicer-plugin/src/ofx/render.rs").read_text(encoding="utf-8")
        stages = [source.index(token) for token in ("owner.state.admit(assets)", "cuda::FramePolicy::prepare(state, full)", "callback.render_native(")]
        self.assertEqual(stages, sorted(stages))
        self.assertIn("let state = admitted.state();", source)
        admission = (ROOT / "rust/film-juicer-core/src/state.rs").read_text(encoding="utf-8")
        self.assertRegex(admission, r"pub struct Admission\s*\{\s*state: Arc<RenderState>")
        self.assertNotIn("cudaStreamSynchronize", source)
        self.assertNotRegex(source, r"JuicerCuda::(?:execute_direct|execute_print|project_and_render)\s*\(")
        render = (ROOT / "rust/film-juicer-plugin/src/cuda/render.rs").read_text(encoding="utf-8")
        self.assertIn(".select_submission(submission::Key", render)
        self.assertIn("self.owner.complete(", render)
        self.assertIn("UnavailableKind::MissingState", render)
        self.assertIn("UnavailableKind::MissingInstance", render)
        callback = (ROOT / "rust/film-juicer-plugin/src/ofx/host/image.rs").read_text(encoding="utf-8")
        self.assertIn('self.message(c"FilmJuicerDeferredCudaFailure", completion.diagnostic())', callback)
        self.assertIn("completion.deferred_dir_error()", callback)
        self.assertNotIn('find("component=dir")', callback)
        completion = (ROOT / "rust/film-juicer-plugin/src/cuda/edge/ofx.rs").read_text(encoding="utf-8")
        self.assertIn("FJ_RENDER_DEFERRED_DIR_ERROR", completion)
        fixture = (ROOT / "tests/ffi/owner_fixture.cpp").read_text(encoding="utf-8")
        self.assertIn("FJ_UNAVAILABLE_STATE", fixture)
        self.assertIn("FJ_UNAVAILABLE_INSTANCE", fixture)
        for old in ("src/mainProcessing.cpp", "src/JuicerEffect.cpp", "src/JuicerEffect.h", "src/PreparedState.cpp"):
            self.assertFalse((ROOT / old).exists(), old)
        for path in ("src/CudaRenderProjection.cpp", "src/CudaRenderProjection.h"):
            self.assertFalse((ROOT / path).exists())
        execution = "".join((ROOT / path).read_text(encoding="utf-8") for path in (
            "src/Cuda/JuicerCudaExecutor.h", "src/Cuda/JuicerCudaExecutor.cpp"))
        retired = r"\b(?:project_and_render|execute_direct|execute_print|describe_execution|DirectExecutionInput|PrintExecutionInput|PrintCorrectionSource)\b"
        self.assertNotRegex(execution, retired)
        for text in ("project_and_render()", "execute_direct()", "execute_print()", "describe_execution()",
                     "DirectExecutionInput input;", "PrintExecutionInput input;", "PrintCorrectionSource source;"):
            self.assertRegex(text, retired)
        for live in ("execute_prepared(", "execute_direct_schedule(", "execute_print_schedule("):
            self.assertIn(live, execution)
        state = "\n".join(path.read_text(encoding="utf-8") for path in product_sources(ROOT))
        model = r"\b(?:FocusedRenderStateBuildProduct|FocusedRenderPayload|CudaFramePreparationRequest|FocusedRouteResourcePreparation|PrintResourcePreparation|build_direct_render_state_product|build_print_render_state_product|build_focused_route_resource_input|derive_print_resource_input)\b"
        root = "".join((ROOT / path).read_text(encoding="utf-8") for path in (
            "src/ProcessRoot.cpp", "src/ProcessRoot.h", "src/Cuda/JuicerCudaResources.cpp",
            "src/Cuda/JuicerCudaResources.h", "src/Cuda/JuicerCudaHostViews.h"))
        self.assertNotRegex(state + root, model)
        for text in ("FocusedRenderStateBuildProduct product;", "FocusedRenderPayload payload;",
                     "CudaFramePreparationRequest request;", "FocusedRouteResourcePreparation input;",
                     "PrintResourcePreparation input;", "build_direct_render_state_product()",
                     "build_focused_route_resource_input()", "derive_print_resource_input()"):
            self.assertRegex(text, model)
        self.assertFalse((ROOT / "src/FocusedRenderPayload.h").exists())
        retired_publications = (
            r"\b(?:InstanceState|PendingParamsState|DirectRenderState|PrintRenderState|JuicerAtomic)\b|"
            r"\b(?:admit_pending_render_state|prepare_published_state|rebuild_direct_render_state|"
            r"rebuild_print_render_state|set_pending_capture_test_hook)\s*\(")
        self.assertEqual(forbidden_uses(native_sources(ROOT), retired_publications), [])
        for name in ("InstanceState", "PendingParamsState", "DirectRenderState", "PrintRenderState"):
            self.assertRegex("struct " + name + " {};", retired_publications)
        self.assertRegex("namespace JuicerAtomic {}", retired_publications)
        for name in ("admit_pending_render_state", "prepare_published_state", "rebuild_direct_render_state",
                     "rebuild_print_render_state", "set_pending_capture_test_hook"):
            self.assertRegex(name + "()", retired_publications)
        cmake = (ROOT / "tests/CMakeLists.txt").read_text()
        retired_hooks = r"\b(?:JUICER_ADMISSION_TEST_HOOK|juicer_admission_test_state)\b"
        self.assertNotRegex(cmake, retired_hooks)
        for name in ("JUICER_ADMISSION_TEST_HOOK", "juicer_admission_test_state"):
            self.assertRegex(name, retired_hooks)
        rust_instance = (ROOT / "rust/film-juicer-plugin/src/instance.rs").read_text(encoding="utf-8")
        self.assertIn("submission: Submission", rust_instance)
        self.assertIn("crate::submission::{self, Submission}", rust_instance)
        owner = (ROOT / "rust/film-juicer-plugin/src/ofx/instance.rs").read_text(encoding="utf-8")
        self.assertIn("pub(crate) struct Owner", owner)
        self.assertIn("state: crate::instance::Instance", owner)
        self.assertIn("std::optional<JuicerCuda::PrintResourceInput> print;", root)
        self.assertIn("const FocusedRouteResourceInput& input", root)
        self.assertIn("const PrintResourceInput& input", root)
        writer = (ROOT / "tests/ffi/prepared_descriptors.cpp").read_text(encoding="utf-8")
        self.assertIn("void encode_prepared_descriptors(", writer)
        self.assertNotRegex(writer, r"\b(?:RenderRecipe|FocusedRenderPayload)\b")
        provider = (ROOT / "native/juicer_cuda_call_scope.cpp").read_text(encoding="utf-8")
        self.assertEqual(provider.count("_call->render("), 1)
        edge = (ROOT / "rust/film-juicer-plugin/src/cuda/edge.rs").read_text(encoding="utf-8")
        self.assertIn("render_prepared_with_instance", edge)
        self.assertNotIn("fj_legacy_prepared_state_render", edge)
        boundary = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertEqual(boundary.count("JuicerCuda::execute_prepared_host_data("), 1)
        self.assertEqual(boundary.count("FjRenderOutcome JuicerCuda::NativeCall::render("), 1)
        self.assertIn("call.render(context, frame, submission, prepared", boundary)

    def test_only_accepted_operations_are_defined(self):
        definitions = []
        for path in (ROOT / "native").glob("*.cpp"):
            definitions.extend(re.findall(r"^(?:FjStatus|FjRenderOutcome) (fj_cuda_\w+)\([^;]*?\)\s*\{", path.read_text(encoding="utf-8"), re.MULTILINE))
        self.assertCountEqual(definitions, ["fj_cuda_create", "fj_cuda_inspect", "fj_cuda_render", "fj_cuda_retire_instance", "fj_cuda_shutdown", "fj_cuda_destroy", "fj_cuda_check_terminal"])

    def test_resource_destruction_does_not_reenter_root(self):
        source = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text(encoding="utf-8")
        self.assertNotIn("ProcessRoot.h", source)
        self.assertNotRegex(source, r"JuicerProcess::|retire_idle_context")
        source = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        self.assertNotIn("SF_TEMP_BRIDGE_release_cuda_owner", source)
        self.assertNotIn("shutdown_if_initialized", source)

    def test_retired_scanner_descriptor_producers_stay_deleted(self):
        self.assertFalse((ROOT / "src/Scanner.cpp").exists())
        header = (ROOT / "src/Scanner.h").read_text(encoding="utf-8")
        bridge = (ROOT / "native/juicer_cuda_descriptors.cpp").read_text(encoding="utf-8")
        retired = r"\b(?:hash_scanner_spectral_lut_descriptor|hash_nan_preserving_floats|kScannerSpectralLutSchemaVersion|DirectScannerLutInput|PrintScannerLutInput|PrintCorrectionDerivationInput|build_direct_scanner_spectral_lut_descriptor|build_print_scanner_spectral_lut_descriptor|build_direct_scanner_color_correction_descriptor|build_print_scanner_color_correction_descriptor|SF_TEMP_BRIDGE_SCANNER_LUT_DESCRIPTOR)\b|_native_baseline"
        self.assertNotRegex(header + bridge, retired)
        for text in ("DirectScannerLutInput input;", "PrintCorrectionDerivationInput input;",
                     "build_direct_scanner_spectral_lut_descriptor()", "SF_TEMP_BRIDGE_SCANNER_LUT_DESCRIPTOR"):
            self.assertRegex(text, retired)
        self.assertNotIn("Hash::", bridge)
        self.assertIn("struct ScannerSpectralLutDescriptor", header)
        self.assertIn("struct ScannerColorCorrectionDescriptor", header)
        self.assertIn("decoded.scanner.hash = source.scanner_lut.hash", (ROOT / "native/juicer_cuda_descriptors.cpp").read_text(encoding="utf-8"))
        for name, facade, owner in (("scanner_lut_descriptor", "fj_test_scanner_lut_describe", "prepare_state("),
                                    ("scanner_correction", "fj_test_scanner_direct_correction_prepare", "JuicerTestSupport::prepare_product(")):
            reader = (ROOT / f"tests/ffi/{name}_test.inc").read_text(encoding="utf-8")
            self.assertIn(facade, reader)
            self.assertIn(owner, reader)
            if name == "scanner_lut_descriptor":
                self.assertIn("representableRows == (products ? 4u : 500u)", reader)
                self.assertIn("retiredPointerRows == (products ? 0u : 20u)", reader)
                self.assertIn('"null-preempts-route-255"', reader)

    def test_native_gaussian_coefficient_producer_and_capture_access_stay_deleted(self):
        source = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text()
        header = (ROOT / "src/Cuda/JuicerCudaResources.h").read_text()
        for name in ("build_gaussian_weights_cpu", "capture_native"):
            self.assertNotIn(name, source + header)
        self.assertNotIn("capture-gaussian-kernels", (ROOT / "tests/ffi/film_recipe_test.cpp").read_text())
        for name in ("RustGaussianBridge.cpp", "RustGaussianBridge.h"):
            self.assertFalse((ROOT / "src" / name).exists())
        retired = (
            r"\b(?:JuicerGaussian::Kernel|JuicerScanner::Lut|SF_TEMP_BRIDGE_GAUSSIAN_KERNEL|"
            r"SF_TEMP_BRIDGE_SCANNER_LUT|fj_legacy_(?:scanner_lut|gaussian_kernel)_(?:acquire|view|release))\b"
            r"|RustGaussianBridge\.(?:cpp|h)"
        )
        paths = color_sources(ROOT) + [ROOT / "tests/CMakeLists.txt"]
        paths += [path for directory in (ROOT / "rust/film-juicer-core/src", ROOT / "rust/film-juicer-plugin/src")
                  for path in directory.rglob("*.rs")]
        self.assertEqual(forbidden_uses(paths, retired), [])
        for name in ("JuicerGaussian::Kernel", "JuicerScanner::Lut", "SF_TEMP_BRIDGE_GAUSSIAN_KERNEL",
                     "SF_TEMP_BRIDGE_SCANNER_LUT", "RustGaussianBridge.cpp", "RustGaussianBridge.h"):
            self.assertRegex(name, retired)
        for family in ("scanner_lut", "gaussian_kernel"):
            for operation in ("acquire", "view", "release"):
                self.assertRegex(f"fj_legacy_{family}_{operation}", retired)
        self.assertNotRegex("SF_TEMP_BRIDGE_SCANNER_COLOR fj_test_gaussian_kernel_acquire", retired)
        body = source.split("    static bool ensure_cached_gaussian_kernel(", 1)[1].split("    bool ensure_spatial_dir_kernel(", 1)[0]
        stages = [body.index(token) for token in (
            "if (!sigmaOk || radius <= 0)", "gaussianKernelCache.find(key)",
            "const FjHostPreparation emptyPreparation{};",
            "const auto& preparation = hostPreparation ? *hostPreparation : emptyPreparation;",
            "prepare_gaussian_kernel(preparation,", "allocate_owned_device(")]
        self.assertEqual(stages, sorted(stages))
        self.assertIn("return true;", body[stages[0]:stages[1]])
        self.assertIn("return true;", body[stages[1]:stages[2]])
        dispatch = (ROOT / "native/juicer_host_preparation.cpp").read_text()
        self.assertNotIn("std::exp", dispatch)
        self.assertNotIn("Hash::", dispatch)
        gaussian = dispatch.split("    bool prepare_gaussian_kernel(", 1)[1]
        self.assertIn("!preparation.user || !preparation.scanner_lut || !preparation.gaussian", gaussian)
        self.assertIn("preparation.gaussian(preparation.user, &input, &view, &failure, &error)", gaussian)
        self.assertIn('{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "Gaussian kernel projection invalid"', gaussian)
        callback = (ROOT / "rust/film-juicer-plugin/src/cuda/host_preparation.rs").read_text()
        for token in ("unsafe extern \"C\" fn gaussian(", "optics_bridge::prepare_host_gaussian(", "&mut backing.gaussians"):
            self.assertIn(token, callback)
        owner = (ROOT / "rust/film-juicer-plugin/src/optics_bridge.rs").read_text()
        facades = (ROOT / "rust/film-juicer-plugin/src/test_support.rs").read_text()
        c11 = (ROOT / "tests/ffi/film_recipe_abi_c.c").read_text()
        c11_gaussian = c11.split("int fj_test_gaussian_c_call(", 1)[1].split("\n}", 1)[0]
        for operation in ("acquire", "view", "release"):
            self.assertIn(f"pub(crate) unsafe fn gaussian_{operation}(", owner)
            self.assertIn(f"fn fj_test_gaussian_kernel_{operation}(", facades)
            self.assertIn(f"crate::optics_bridge::gaussian_{operation}(", facades)
            self.assertIn(f"fj_test_gaussian_kernel_{operation}(", c11_gaussian)
        fixture = (ROOT / "tests/ffi/gaussian_kernel_test.inc").read_text()
        for token in ("with_host_preparation(", "prepare_gaussian_kernel(", "fj_test_gaussian_c_call(", "rows.size() == 188"):
            self.assertIn(token, fixture)
        cases = (ROOT / "tests/resource/gaussian_kernel_test.cpp").read_text()
        for token in (
            "ensure_gaussian_kernel(r, r.scannerLensBlurKernel, 1.0f, 3, failure, nullptr)",
            "failure.status.category == FJ_STATUS_PREPARATION_FAILURE", "failure.status.api == FJ_API_NONE",
            'failure.diagnostic == "Gaussian kernel projection invalid"', "equal(snapshot(r), empty)",
            "fj_test_gaussian_kernel_acquires() == coldAcquires",
            "ensure_gaussian_kernel(r, r.scannerUnsharpKernel, 1.0f, 3, failure, nullptr)",
            "fj_test_gaussian_kernel_acquires() == warmAcquires", "equal(snapshot(r), initial)",
        ):
            self.assertIn(token, cases)
        cmake = (ROOT / "tests/CMakeLists.txt").read_text()
        self.assertIn("add_library(juicer_gaussian_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)", cmake)
        self.assertIn("target_compile_definitions(juicer_gaussian_test_objects PRIVATE JUICER_GAUSSIAN_TEST_HOOK=1)", cmake)

    def test_reflected_dir_boundary_has_one_rust_producer_and_native_cache_owner(self):
        retired = r"\b(?:DirDoubleDouble|dir_dd_\w+|SpatialDirBoundaryAxisCpu|build_spatial_dir_boundary_axis_cpu|spatial_dir_boundary_identity)\b"
        paths = [path for directory in (ROOT / "src", ROOT / "native")
                 for path in directory.rglob("*")
                 if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, retired), [])
        for definition in ("struct DirDoubleDouble {};", "double dir_dd_multiply() {}",
                           "struct SpatialDirBoundaryAxisCpu {};",
                           "bool build_spatial_dir_boundary_axis_cpu() {}",
                           "uint64_t spatial_dir_boundary_identity() {}"):
            self.assertRegex(definition, retired)
        source = (ROOT / "src/Cuda/JuicerCudaResources.cpp").read_text()
        body = source.split("    bool ensure_spatial_dir_boundary(", 1)[1].split("    bool clear_spatial_dir_boundary_binding(", 1)[0]
        stages = [body.index(token) for token in (
            "JuicerDirBoundary::describe(", "boundary.weights && boundary.identity == identity",
            "JuicerDirBoundary::Boundary::acquire(", "hostBoundary->view()", "alloc_and_upload_bytes(",
            "bool keepCandidate", "boundary.weights = static_cast<double*>(candidateRaw)",
            "if (!keepCandidate)")]
        self.assertEqual(stages, sorted(stages))
        self.assertEqual(body.count("validate_resource_owner_locked(resources, outError, true)"), 2)
        self.assertEqual(body.count("boundary.weights && boundary.identity == identity"), 2)
        self.assertIn("view.weights", body)
        self.assertIn("view.horizontal_terminal", body)
        self.assertNotIn("identity == 0", body)
        self.assertNotIn("std::vector<double>", body)
        hooks = "".join((ROOT / path).read_text() for path in (
            "tests/dir/boundary_test_hooks.h", "tests/dir/boundary_test_hooks.cpp", "tests/ffi/film_recipe_test.cpp"))
        for token in ("capture-dir-boundaries", "capture_inputs", "key_bytes", "DirDoubleDouble"):
            self.assertNotIn(token, hooks)
        bridge = (ROOT / "src/JuicerDirBoundary.cpp").read_text()
        self.assertNotRegex(bridge, r"std::fma|std::exp|Hash::|hash_bytes_update")
        header = (ROOT / "src/JuicerDirBoundary.h").read_text()
        self.assertIn('#include "juicer_dir_boundary_api.h"', header)
        boundary = (ROOT / "native/juicer_dir_boundary_api.h").read_text()
        for operation in ("describe", "acquire", "view", "release"):
            self.assertIn("fj_dir_boundary_" + operation + "(", boundary)
            self.assertIn("fj_dir_boundary_" + operation + "(", bridge)
        retired_transport = r"\bfj_legacy_dir_boundary_\w+\b|juicer_legacy_api\.h"
        self.assertNotRegex(header + bridge + boundary, retired_transport)
        for declaration in ("fj_legacy_dir_boundary_acquire()", '#include "juicer_legacy_api.h"'):
            with self.assertRaises(AssertionError):
                self.assertNotRegex(header + bridge + boundary + declaration, retired_transport)

    def test_dir_boundary_hooks_are_absent_from_ordinary_objects(self):
        build = product_build()
        compdb = json.loads((build / "compile_commands.json").read_bytes())
        isolated = [row for row in compdb if "juicer_dir_boundary_test_objects.dir" in row["command"]]
        self.assertEqual(len(isolated), 1)
        self.assertIn("JUICER_DIR_BOUNDARY_TEST_HOOK=1", isolated[0]["command"])
        nm = find_llvm_nm()
        self.assertTrue(nm, "qualified LLVM nm required for ordinary-object isolation")
        for owner, source in (("juicer_native", "Cuda/JuicerCudaResources.cpp"), ("juicer_host", "JuicerDirBoundary.cpp")):
            rows = [row for row in compdb if owner + ".dir" in row["command"] and row["file"].endswith(source)]
            self.assertEqual(len(rows), 1)
            self.assertNotIn("JUICER_DIR_BOUNDARY_TEST_HOOK", rows[0]["command"])
            path = build / "CMakeFiles" / (owner + ".dir") / "src" / (source + (".obj" if os.name == "nt" else ".o"))
            data = path.read_bytes()
            symbols = subprocess.check_output([nm, "--demangle", "--defined-only", str(path)]).decode(errors="replace")
            self.assertNotRegex(symbols, r"DirBoundaryTest::|fj_test_dir_boundary|dir_boundary_bridge::faults::")
            for text in ("DirBoundaryTest", "spatial DIR boundary fixture panic", "spatial DIR boundary candidate rendezvous canceled", "capture-dir-boundaries", "fj_test_dir_boundary"):
                for encoding in ("utf-8", "utf-16-le"):
                    pattern = text.encode(encoding)
                    self.assertNotIn(pattern, data)
                    with self.assertRaises(AssertionError):
                        self.assertNotIn(pattern, data + pattern)
            print("DIR boundary ordinary-object isolation", path, hashlib.sha256(data).hexdigest())

    def test_effects_frame_checked_edge_has_no_native_mathematical_or_resource_owner(self):
        core = (ROOT / "rust/film-juicer-core/src/effects/frame.rs").read_text().split("#[cfg(test)]", 1)[0]
        binder = (ROOT / "src/FilmEffectsFrameDescriptors.cpp").read_text()
        self.assertNotRegex(core, r"\b(?:unsafe|Vec|Box|Arc|Mutex|Cell)\b|mul_add\s*\(")
        self.assertEqual(core.count("libm::fma("), 3)
        self.assertNotRegex(binder, r"std::(?:fma|floor|ceil|sin|cos|sqrt)\s*\(|cudaMalloc|cudaFree")
        effects_binder = binder.split("bool compose_frame_descriptor(", 1)[1].split("\n    }", 1)[0]
        self.assertIn("math.present != 1", effects_binder)
        self.assertIn("(math.flags & ~31u) != 0", effects_binder)
        self.assertIn("out.recipeHash = math.recipe_hash", effects_binder)
        self.assertNotRegex(effects_binder, r"Hash::|hash_bytes|cudaMalloc|cudaFree")
        frame = (ROOT / "rust/film-juicer-plugin/src/cuda/frame.rs").read_text()
        self.assertIn("let recipe = state.effects();", frame)
        self.assertIn("recipe_hash: recipe.hash()", frame)
        self.assertIn("effects::frame::Input", frame)
        self.assertIn("crate::effects_frame_bridge::checked_frame(input, fault)", frame)
        projection = (ROOT / "rust/film-juicer-plugin/src/effects_frame_bridge.rs").read_text()
        self.assertIn("recipe_hash: v.recipe_hash()", projection)
        scope = (ROOT / "native/juicer_cuda_call_scope.cpp").read_text()
        self.assertIn("JuicerEffects::compose_frame_descriptor(", scope)
        self.assertIn("input->effects.session_seed != frame.session_seed", scope)
        self.assertIn("input->effects.clip_token != frame.clip_token", scope)
        self.assertIn("same_extent(input->effects.full_frame_extent, frame.effects_geometry.pixel_definition)", scope)
        self.assertFalse((ROOT / "src/RustEffectsFrameBridge.cpp").exists())
        self.assertFalse((ROOT / "src/RustEffectsFrameBridge.h").exists())

    def test_retired_effects_frame_mathematics_and_capture_access_stay_deleted(self):
        source = (ROOT / "src/FilmEffectsFrameDescriptors.cpp").read_text()
        header = (ROOT / "src/FilmEffectsFrameDescriptors.h").read_text()
        readers = "".join((ROOT / path).read_text() for path in (
            "tests/CMakeLists.txt", "tests/ffi/effects_frame_test.inc", "tests/ffi/film_recipe_test.cpp"))
        retired = r"\b(?:hash_film_juicer_effects_descriptor|hash_to_unit|phase_from_seed|sin_sum|compute_gate_weave|split_defect_origin|valid_defect_grid|kSeedPassWeave|JUICER_EFFECTS_FRAME_CAPTURE_TEST_HOOK|juicer_capture_effects_primitive|juicer_effects_frame_capture_objects)\b|capture-effects-frames|effects_frame_capture\.h"
        self.assertNotRegex(source + header + readers, retired)
        self.assertNotRegex(source, r"std::(?:fma|floor|ceil|sin|cos|sqrt)\s*\(")
        self.assertFalse((ROOT / "tests/ffi/effects_frame_capture.h").exists())
        for text in ("void compute_gate_weave() {}", "auto hash_film_juicer_effects_descriptor() {}",
                     "JUICER_EFFECTS_FRAME_CAPTURE_TEST_HOOK", "capture-effects-frames", "effects_frame_capture.h"):
            self.assertRegex(text, retired)
        self.assertNotRegex(source, r"bool build_film_juicer_effects_frame_descriptor\s*\(")
        self.assertIn("JuicerEffects::compose_frame_descriptor(", (ROOT / "native/juicer_cuda_call_scope.cpp").read_text())
        executor = (ROOT / "src/Cuda/JuicerCudaExecutor.cpp").read_text()
        self.assertNotIn("build_film_juicer_effects_descriptor_for_frame(", executor)
        self.assertNotIn("ExecutorTest::observe_effects_frame(", executor)
        scope = (ROOT / "native/juicer_cuda_call_scope.cpp").read_text()
        self.assertEqual(scope.count("JuicerCuda::ExecutorTest::observe_effects_frame("), 1)
        self.assertIn("JuicerEffects::compose_frame_descriptor(", scope)
        self.assertIn("#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)", scope)
        self.assertIn("JuicerTestSupport::prepare_effects_frame", (ROOT / "tests/scatter_halation/recipe_descriptor_test.cpp").read_text())

    def test_effects_frame_seams_are_absent_from_ordinary_native_objects(self):
        build = product_build()
        compdb = json.loads((build / "compile_commands.json").read_bytes())
        nm = find_llvm_nm()
        self.assertTrue(nm, "qualified LLVM nm required for effects frame ordinary-object isolation")
        for owner, source in (("juicer_host", "src/FilmEffectsFrameDescriptors.cpp"),
                              ("juicer_native", "native/juicer_cuda_call_scope.cpp"),
                              ("juicer_native", "src/Cuda/JuicerCudaExecutor.cpp")):
            rows = [row for row in compdb if owner + ".dir" in row["command"] and row["file"].replace("\\", "/").endswith(source)]
            self.assertEqual(len(rows), 1)
            self.assertNotIn("JUICER_EFFECTS_FRAME_CAPTURE_TEST_HOOK", rows[0]["command"])
            path = build / "CMakeFiles" / (owner + ".dir") / (source + (".obj" if os.name == "nt" else ".o"))
            data = path.read_bytes()
            symbols = subprocess.check_output([nm, "--demangle", "--defined-only", str(path)]).decode(errors="replace")
            self.assertNotRegex(symbols, r"EffectsFrameObservation|effects_frame_observation|juicer_capture_effects|fj_test_effects_frame")
            for text in ("EffectsFrameObservation", "physical effects frame fixture entry panic",
                         "fj_test_effects_frame", "capture-effects-frames", "effects_frame_capture.h"):
                for encoding in ("utf-8", "utf-16-le"):
                    pattern = text.encode(encoding)
                    self.assertNotIn(pattern, data)
                    with self.assertRaises(AssertionError):
                        self.assertNotIn(pattern, data + pattern)
            print("Effects frame ordinary-object isolation", path, hashlib.sha256(data).hexdigest())

    def test_retired_grain_frame_mathematics_and_capture_access_stay_deleted(self):
        source = (ROOT / "src/FilmEffectsFrameDescriptors.cpp").read_text()
        grain = source.split("constexpr std::uint64_t kSeedPassWeave", 1)[0]
        header = (ROOT / "src/FilmEffectsFrameDescriptors.h").read_text()
        setup = {path: (ROOT / path).read_text() for path in (
            "tests/grain/scratch_reuse_test.cpp", "tests/ffi/gaussian_kernel_test.inc",
            "tests/ffi/cuda_render_test.cpp")}
        readers = "".join((ROOT / path).read_text() for path in (
            "tests/CMakeLists.txt", "tests/ffi/grain_frame_test.inc", "tests/ffi/film_recipe_test.cpp"))
        retired = r"\b(?:nanmax|make_gaussian|kernel_energy_2d|make_seed_base|kSeedPassGrain|kMaxGrainGaussianRadius|JUICER_GRAIN_FRAME_CAPTURE_TEST_HOOK|JuicerGrainCapture|juicer_grain_frame_capture_objects)\b|capture-grain-frames|grain_frame_capture\.h"
        self.assertNotRegex(grain + header + readers, retired)
        self.assertNotRegex(grain, r"std::(?:floor|llround|sqrt|exp)\s*\(")
        self.assertFalse((ROOT / "tests/ffi/grain_frame_capture.h").exists())
        for text in ("float nanmax() {}", "auto make_gaussian() {}", "float kernel_energy_2d() {}", "auto make_seed_base() {}", "JUICER_GRAIN_FRAME_CAPTURE_TEST_HOOK", "capture-grain-frames", "grain_frame_capture.h"):
            self.assertRegex(grain + text, retired)
        model = r"\b(?:VisualGrainFrameDescriptorInput|build_visual_grain_frame_descriptor|SF_TEMP_BRIDGE_GRAIN_FRAME)\b|RustGrainFrameBridge"
        self.assertNotRegex(grain + header + readers, model)
        for text in ("VisualGrainFrameDescriptorInput input;", "build_visual_grain_frame_descriptor()", "RustGrainFrameBridge.h"):
            self.assertRegex(text, model)
        for suffix in ("h", "cpp"):
            self.assertFalse((ROOT / f"src/RustGrainFrameBridge.{suffix}").exists())
        self.assertIn("compose_visual_grain_frame_descriptor(", grain)
        self.assertIn("visual_grain_scratch_shape(const FjGrainCoefficients&", grain)
        raw = (ROOT / "tests/ffi/grain_frame_test.inc").read_text()
        self.assertIn("fj_test_grain_frame_prepare(", raw)
        self.assertIn("representableRows == 150 && retiredPointerRows == 3", raw)
        self.assertEqual(grain.count("hash_descriptor("), 2)
        self.assertEqual(grain.count("out = descriptor;"), 1)
        for path, consumer in setup.items():
            self.assertNotIn("build_visual_grain_frame_descriptor", consumer)
            if path == "tests/ffi/gaussian_kernel_test.inc":
                self.assertRegex(consumer, r"JuicerTestSupport::prepare_product\([^,\n]+,\s*p\)")
                self.assertNotIn("GrainFrameTest::product_input", consumer)
                self.assertIn("JuicerTestSupport::grain_frame_input(view, grainFrame)", consumer)
                self.assertIn("fj_test_grain_frame_prepare(&rawGrain", consumer)
            elif path == "tests/grain/scratch_reuse_test.cpp":
                self.assertRegex(consumer, r"JuicerTestSupport::prepare_product\([^,\n]+,\s*make_controls\(route, variant\)\)")
                self.assertIn("FrameFixture::with_prepared_frame_inputs(", consumer)
                helper = (ROOT / "tests/ffi/prepared_frame_fixture.cpp").read_text()
                self.assertIn("fj_test_grain_frame_prepare(", helper)
                self.assertIn("compose_visual_grain_frame_descriptor(", helper)
            else:
                self.assertNotIn("JuicerTestSupport::prepare_grain_frame", consumer)
                self.assertIn("JuicerTestSupport::grain_frame_input(*state, facts)", consumer)
                self.assertIn("JuicerTestSupport::grain_coefficients(state->grain)", consumer)
                self.assertIn("fj_test_grain_frame_prepare(&raw", consumer)
                self.assertIn("compose_visual_grain_frame_descriptor(", consumer)

    def test_retired_frame_geometry_mathematics_and_capture_access_stay_deleted(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in product_sources(ROOT))
        hooks = "".join((ROOT / "tests/ofx" / name).read_text() for name in ("frame_geometry_test_hooks.h", "frame_geometry_test_hooks.cpp", "adapter_trace_test.cpp"))
        pattern = r"\b(?:native_frame_domain|native_pixel_size|capture_domain|capture_pixel_size)\s*\(|filmFormatMm\s*\*\s*1000\.0|static_cast<std::int64_t>\(bounds\.x2\)"
        self.assertNotRegex(source + hooks, pattern)
        self.assertNotIn("--capture-frame-geometry", hooks)
        for retired in ("float native_pixel_size() {}", "auto capture_domain() {}", "filmFormatMm * 1000.0", "static_cast<std::int64_t>(bounds.x2)"):
            self.assertRegex(source + retired, pattern)
        for old in ("src/RustFrameBridge.cpp", "src/RustFrameBridge.h", "src/JuicerEffect.cpp"):
            self.assertFalse((ROOT / old).exists(), old)
        frame = (ROOT / "rust/film-juicer-plugin/src/cuda/frame_policy.rs").read_text()
        self.assertIn("crate::frame_bridge::checked_geometry(", frame)
        self.assertIn("FramePolicy::prepare(state, full)", (ROOT / "rust/film-juicer-plugin/src/ofx/render.rs").read_text())

    def test_frame_geometry_hooks_are_absent_from_ordinary_objects(self):
        build = product_build()
        compdb = json.loads((build / "compile_commands.json").read_bytes())
        nm = find_llvm_nm()
        self.assertTrue(nm, "qualified LLVM nm required for ordinary-object isolation")
        self.assertFalse(any(row["file"].endswith("RustGrainFrameBridge.cpp") for row in compdb))
        lib = (ROOT / "rust/film-juicer-plugin/src/lib.rs").read_text(encoding="utf-8")
        self.assertIn('#[cfg(feature = "test-support")]\nmod frame_observer;', lib)
        for owner, source in (("juicer", "native/ofx_exports.cpp"), ("juicer_host", "src/FilmEffectsFrameDescriptors.cpp"), ("juicer_host", "src/JuicerDirBoundary.cpp"), ("juicer_host", "src/DiffusionHostBehavior.cpp"), ("juicer_native", "src/Cuda/JuicerCudaExecutor.cpp")):
            rows = [row for row in compdb if owner + ".dir" in row["command"] and row["file"].replace("\\", "/").endswith(source)]
            self.assertEqual(len(rows), 1)
            self.assertNotIn("JUICER_FRAME_GEOMETRY_TEST_HOOK", rows[0]["command"])
            self.assertNotIn("JUICER_EXECUTOR_FAILURE_TEST_HOOK", rows[0]["command"])
            self.assertNotIn("JUICER_GRAIN_FRAME_CAPTURE_TEST_HOOK", rows[0]["command"])
            path = build / "CMakeFiles" / (owner + ".dir") / (source + (".obj" if os.name == "nt" else ".o"))
            data = path.read_bytes()
            symbols = subprocess.check_output([nm, "--demangle", str(path)]).decode(errors="replace")
            self.assertNotRegex(symbols, r"FrameGeometryTest::|frame_bridge::faults::|halation_frame_faults::|diffusion_frame_faults::|dir_frame_faults::|grain_frame_bridge::faults::|JuicerGrainCapture::|ExecutorTest::|fj_test_grain_frame|fj_test_dir_frame|fj_test_diffusion_|fj_test_scatter_halation_frame|fj_test_frame|native_frame_domain|native_pixel_size")
            for text in ("FrameGeometryTest", "frame geometry fixture panic", "--capture-frame-geometry", "--frame-geometry", "fj_test_frame", "scatter/halation frame fixture panic", "fj_test_scatter_halation_frame", "HalationObservation", "DiffusionObservation", "diffusion construction fixture panic", "fj_test_diffusion_", "capture-diffusion-frames", "DirFrameObservation", "spatial DIR frame fixture panic", "fj_test_dir_frame", "capture-dir-frames", "--dir-frame", "GrainFrameObservation", "visual grain frame fixture entry panic", "visual grain frame fixture layer panic", "fj_test_grain_frame", "capture-grain-frames", "--grain-frame"):
                for encoding in ("utf-8", "utf-16-le"):
                    pattern = text.encode(encoding)
                    self.assertNotIn(pattern, data)
                    with self.assertRaises(AssertionError):
                        self.assertNotIn(pattern, data + pattern)
            print("Frame geometry ordinary-object isolation", path, hashlib.sha256(data).hexdigest())

    def test_gaussian_hooks_and_allocator_are_absent_from_ordinary_objects(self):
        build = product_build()
        compdb = json.loads((build / "compile_commands.json").read_bytes())
        ordinary = [row for row in compdb if "juicer_native.dir" in row["command"] or "juicer_host.dir" in row["command"]]
        self.assertTrue(ordinary)
        self.assertFalse(any("JUICER_GAUSSIAN_TEST_HOOK" in row["command"] for row in ordinary))
        self.assertFalse(any(Path(row["file"]).name == "RustGaussianBridge.cpp" for row in compdb))
        self.assertEqual(sum(Path(row["file"]).name == "JuicerCudaResources.cpp" for row in ordinary), 1)
        nm = find_llvm_nm()
        self.assertTrue(nm, "qualified LLVM nm required for ordinary-object isolation")
        for owner, source in (("juicer_native", "Cuda/JuicerCudaResources.cpp"),):
            path = build / "CMakeFiles" / (owner + ".dir") / "src" / (source + (".obj" if os.name == "nt" else ".o"))
            data = path.read_bytes()
            output = subprocess.check_output([nm, "--demangle", "--defined-only", str(path)]).decode(errors="replace")
            self.assertNotRegex(output, r"GaussianKernelTest::|SpectralAllocationProbe::|gaussian_faults::|fj_test_gaussian|rejectNativeAllocation|rejectedNativeBytes")
            for text in ("Gaussian kernel fixture", "GaussianKernelTest", "SpectralAllocationProbe"):
                for encoding in ("utf-8", "utf-16-le"):
                    pattern = text.encode(encoding)
                    self.assertNotIn(pattern, data)
                    with self.assertRaises(AssertionError):
                        self.assertNotIn(pattern, data + pattern)
            print("Gaussian ordinary-object isolation", path, hashlib.sha256(data).hexdigest())

    def test_retired_scanner_post_constructor_stays_deleted(self):
        scanner = "".join((ROOT / path).read_text(encoding="utf-8") for path in (
            "src/Scanner.h", "src/Cuda/JuicerCudaExecutor.cpp", "native/juicer_cuda_descriptors.cpp"))
        retired = r"\b(?:build_scanner_post_effects_descriptor|scanner_gaussian_radius_or_zero|scanner_gaussian_sigma_with_device_kernel_or_zero|kScannerGaussianTruncate|kMaxScannerGaussianRadius|SF_TEMP_BRIDGE_SCANNER_POST_EFFECTS|fj_legacy_scanner_post_describe)\b"
        self.assertNotRegex(scanner, retired)
        for text in ("build_scanner_post_effects_descriptor()", "SF_TEMP_BRIDGE_SCANNER_POST_EFFECTS"):
            self.assertRegex(text, retired)
        self.assertIn("struct ScannerPostEffectsDescriptor", scanner)
        reader = (ROOT / "tests/ffi/scanner_post_effects_test.inc").read_text(encoding="utf-8")
        self.assertIn("fj_test_scanner_post_describe", reader)
        self.assertIn("JuicerTestSupport::prepare_product(", reader)
        self.assertNotIn("capture-scanner-post-effects", (ROOT / "tests/ffi/film_recipe_test.cpp").read_text(encoding="utf-8"))

    def test_native_code_does_not_reset_host_contexts(self):
        paths = native_sources(ROOT)
        paths.append(ROOT / "src/ProcessRoot.cpp")
        self.assertEqual(forbidden_uses(paths, CONTEXT_RESET), [])

    def test_submission_policy_has_one_safe_owner_and_no_native_selector(self):
        retired = r"\b(?:NativeSubmissionState|FjSubmissionAccess|SubmissionLatch|SubmissionSelection)\b|\bselect_submission\s*\("
        self.assertEqual(forbidden_uses(native_sources(ROOT), retired), [])
        for name in ("NativeSubmissionState", "FjSubmissionAccess", "SubmissionLatch", "SubmissionSelection"):
            self.assertRegex("struct " + name + " {};", retired)
        self.assertRegex("select_submission()", retired)
        edge = (ROOT / "rust/film-juicer-plugin/src/cuda/edge.rs").read_text()
        self.assertNotRegex(edge, r"\b(?:FjSubmissionAccess|SubmissionLatch)\b")
        owner = (ROOT / "rust/film-juicer-plugin/src/submission.rs").read_text()
        self.assertIn("#![forbid(unsafe_code)]", owner)
        render = (ROOT / "rust/film-juicer-plugin/src/cuda/render.rs").read_text()
        self.assertIn(".select_submission(submission::Key", render)
        ofx = (ROOT / "rust/film-juicer-plugin/src/ofx/render.rs").read_text()
        self.assertIn("frame_position(args.time)", ofx)
        self.assertIn("callback.render_native(", ofx)
        self.assertFalse((ROOT / "src/mainProcessing.cpp").exists())
        self.assertEqual(forbidden_uses(product_sources(ROOT), r"SF_TEMP_BRIDGE_(?:FRAME_IDENTITY_INPUTS|NATIVE_SUBMISSION_STATE)"), [])

    def test_s5c2_process_ownership_is_explicit_and_test_helpers_stay_test_only(self):
        paths = product_sources(ROOT)
        self.assertEqual(forbidden_uses(paths, RETIRED_S5C2_AMBIENT_OWNERS), [])
        retired_d = (r"\b(?:class|struct)\s+(?:JuicerEffect|JuicerProcessor|ParamSnapshot|ProcessRuntime|RenderInstance|PreparedState)\b|"
                     r"\bOFX::|\bjuicer_ofxs\b|ofxs[A-Za-z0-9_]*\.h")
        self.assertEqual(forbidden_uses(paths, retired_d), [])
        for spelling in ("class JuicerEffect {};", "class JuicerProcessor {};", "struct ParamSnapshot {};",
                         "class ProcessRuntime {};", "class RenderInstance {};", "class PreparedState {};",
                         "OFX::ImageEffect* owner;", "juicer_ofxs", '#include "ofxsImageEffect.h"'):
            self.assertRegex(spelling, retired_d)

        for relative in (
            "src/ResourceAssetLibrary.h", "src/ResourceAssetLibrary.cpp",
            "src/RustAssetBridge.h", "src/RustAssetBridge.cpp",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)

        for old in ("src/ProcessRuntime.h", "src/ProcessRuntime.cpp", "src/JuicerEffect.cpp", "src/mainProcessing.cpp", "src/main.cpp"):
            self.assertFalse((ROOT / old).exists(), old)
        process_root = (ROOT / "src/ProcessRoot.h").read_text(encoding="utf-8")
        runtime = (ROOT / "rust/film-juicer-plugin/src/runtime.rs").read_text(encoding="utf-8")
        module = (ROOT / "rust/film-juicer-plugin/src/ofx/mod.rs").read_text(encoding="utf-8")
        effect = (ROOT / "rust/film-juicer-plugin/src/ofx/render.rs").read_text(encoding="utf-8")
        self.assertIn("Runtime::new(directory)", module)
        self.assertIn("loaded.runtime.check_close()", module)
        self.assertIn("runtime.close()", module)
        self.assertIn("self.native.take()", runtime)
        self.assertIn("self.assets.take()", runtime)
        self.assertIn("owner.loaded.runtime().assets()", effect)
        self.assertIn("callback.render_native(", effect)
        entry = (ROOT / "native/ofx_exports.cpp").read_text(encoding="utf-8")
        self.assertIn("OfxGetNumberOfPlugins(", entry)
        self.assertIn("OfxGetPlugin(", entry)
        self.assertNotIn("static Root& instance()", process_root)
        self.assertNotIn("JuicerAssets::Library", process_root)
        self.assertNotRegex(process_root, r"\bFjAssets\s*\*|\b_assets\b|\b_dataDirectory\b")
        self.assertNotIn("borrow_rust_assets", process_root)

        for relative in (
            "tests/ffi/native_path.h", "tests/ffi/native_path.cpp",
            "tests/ffi/test_asset_bridge.h", "tests/ffi/test_asset_bridge.cpp",
            "tests/ffi/test_asset_library.h", "tests/ffi/test_asset_library.cpp",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        bridge_header = (ROOT / "tests/ffi/test_asset_bridge.h").read_text(encoding="utf-8")
        bridge_source = (ROOT / "tests/ffi/test_asset_bridge.cpp").read_text(encoding="utf-8")
        library_source = (ROOT / "tests/ffi/test_asset_library.cpp").read_text(encoding="utf-8")
        bridge_consumer = (ROOT / "tests/ffi/production_profile_owner_test.cpp").read_text(encoding="utf-8")
        library_consumer = (ROOT / "tests/ffi/input_hull_test.cpp").read_text(encoding="utf-8")
        native_path_header = (ROOT / "tests/ffi/native_path.h").read_text(encoding="utf-8")
        native_path_source = (ROOT / "tests/ffi/native_path.cpp").read_text(encoding="utf-8")
        native_owner = (ROOT / "native/juicer_cuda_owner.h").read_text(encoding="utf-8")
        self.assertIn("class InputHullHold", bridge_header)
        self.assertIn("class OutputBoundaryHold", bridge_header)
        self.assertIn("InputHullHold::~InputHullHold()", bridge_source)
        self.assertIn("OutputBoundaryHold::~OutputBoundaryHold()", bridge_source)
        self.assertIn("fj_test_assets_input_hull_acquire(", bridge_source)
        self.assertNotIn("fj_legacy_assets_input_hull_acquire(", bridge_source)
        self.assertIn("fj_test_assets_output_boundary_acquire(", bridge_source)
        self.assertNotIn("fj_legacy_assets_output_boundary_acquire(", bridge_source)
        self.assertIn('#include "juicer_test_api.h"', bridge_consumer)
        for operation in ("fj_test_assets_film_defaults(", "fj_test_assets_film_acquire(",
                          "fj_test_assets_film_view(", "fj_test_assets_film_release("):
            self.assertIn(operation, bridge_consumer)
        self.assertIn('#include "test_asset_library.h"', library_consumer)
        self.assertIn("JuicerAssets::Library", library_consumer)
        self.assertIn("class NativePathArgument", native_path_header)
        self.assertIn("NativePathArgument::view()", native_path_source)
        self.assertNotRegex(native_path_header + native_path_source, r"copy_native_path|path_diagnostic|fj_legacy_assets_create")
        self.assertIn("JuicerProcess::Root& borrow_root(FjCuda* cuda)", native_owner)
        self.assertNotIn("fj_legacy_runtime_create(", library_source + bridge_source)

        build = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8") + (ROOT / "cmake/JuicerSources.cmake").read_text(encoding="utf-8")
        tests_build = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("src/ResourceAssetLibrary.cpp", build + tests_build)
        self.assertNotIn("src/RustAssetBridge.cpp", build + tests_build)
        self.assertNotIn("tests/ffi/test_asset_bridge.cpp", build)
        self.assertNotIn("tests/ffi/test_asset_library.cpp", build)
        path_owner = r"\bNativePathArgument\b"
        self.assertEqual(forbidden_uses(product_sources(ROOT), path_owner), [])
        self.assertRegex("class NativePathArgument {};", path_owner)
        self.assertFalse((ROOT / "src/NativePath.h").exists())
        self.assertFalse((ROOT / "src/NativePath.cpp").exists())
        self.assertNotIn("src/NativePath.cpp", build + tests_build)
        self.assertIn("native_path.cpp", tests_build)
        self.assertIn("test_asset_bridge.cpp", tests_build)
        self.assertIn("test_asset_library.cpp", tests_build)

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
        self.assertFalse((ROOT / "src/ProfileAssets.h").exists())

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
        self.assertEqual(forbidden_uses(paths, RETIRED_SPECTRAL_COPIES), [])
        for name in ("RustSpectralBridge.h", "RustSpectralBridge.cpp", "RustReconstructionBridge.h",
                     "RustReconstructionBridge.cpp", "RustExposureBridge.h", "RustExposureBridge.cpp",
                     "SpectralProcessing.cpp", "Scanner.cpp"):
            self.assertFalse((ROOT / "src" / name).exists())
        values = (ROOT / "src/SpectralData.h").read_text(encoding="utf-8")
        for token in ("struct Curve", "struct SpectralShape", "inline constexpr SpectralShape kReferenceShape{};",
                      "kNumSamples == 81", "kFilmTcLutExtent = 192", "kFilmTcLutChannels = 4",
                      "lambda.assign(kReferenceShape.wavelengths.begin(), kReferenceShape.wavelengths.end())"):
            self.assertIn(token, values)
        assets = (ROOT / "rust/film-juicer-core/src/assets.rs").read_text(encoding="utf-8")
        constructor = assets.split("    pub fn new(resource_dir: PathBuf) -> Self {", 1)[1].split("    fn snapshot(", 1)[0]
        for family in ("hanatos", "arctic", "mallett", "cmf"):
            self.assertIn(f"{family}: OnceLock::new()", constructor)
            self.assertNotIn(f"self.{family}(", constructor)
        for family in ("hanatos", "arctic", "mallett"):
            owner = assets.split(f"    fn {family}_with(", 1)[1].split("\n    }", 1)[0]
            self.assertIn(f"if self.{family}.get().is_none()", owner)
            self.assertIn(f"drop(self.{family}.set(prepared))", owner)
            self.assertIn(".map(Arc::clone)", owner)
        complete = (ROOT / "rust/film-juicer-core/src/recipe.rs").read_text(encoding="utf-8")
        foundation = complete.split("fn acquire_foundation(", 1)[1].split("fn enlarger_illuminant(", 1)[0]
        stages = [foundation.index(token) for token in (
            "let illuminants = assets.illuminants()?;", "let observer = Observer::acquire(assets)?;",
            "match params.spectral_upsampling_mode")]
        self.assertEqual(stages, sorted(stages))
        for token in ("let rows = assets.cmf()?;", "let source = assets.hanatos()?;",
                      "1 => mallett = Some(assets.mallett()?)", "tc_source = Some(assets.arctic()?)"):
            self.assertIn(token, complete)
        preparation = (ROOT / "rust/film-juicer-plugin/src/preparation.rs").read_text(encoding="utf-8")
        self.assertIn("let completed = recipe::prepare(assets, params)", preparation)

    def test_retired_illuminant_and_calibration_authority_stays_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native", ROOT / "cmake", ROOT / "tests")
                 for path in directory.rglob("*") if path.is_file() and
                 (path.suffix.lower() in SOURCE_SUFFIXES or path.suffix == ".cmake")]
        self.assertEqual(forbidden_uses(paths, RETIRED_ILLUMINANTS_CALIBRATION), [])
        self.assertEqual(forbidden_uses(paths + [ROOT / "CMakeLists.txt", ROOT / "tests/CMakeLists.txt"], RETIRED_ILLUMINANT_COPIES), [])
        for name in ("Illuminants.h", "Illuminants.cpp"):
            self.assertFalse((ROOT / "src" / name).exists())
        core = (ROOT / "rust/film-juicer-core/src/illuminant.rs").read_text(encoding="utf-8")
        for name in ("blackbody", "checked_equal_energy", "tungsten_kg3", "prepare_lens", "finish_lens"):
            self.assertIn(f"pub fn {name}(", core)
        preparation = (ROOT / "rust/film-juicer-plugin/src/preparation.rs").read_text(encoding="utf-8")
        owner = preparation.split("unsafe fn illuminants_acquire_with(", 1)[1].split("pub(crate) unsafe fn illuminants_view(", 1)[0]
        for token in (".illuminants_observed(&IlluminantObserver(&*hooks))", "entry: owner.illuminants().map_err(Failure::Asset)?"):
            self.assertIn(token, owner)
        bridge = (ROOT / "tests/ffi/test_asset_bridge.cpp").read_text(encoding="utf-8")
        self.assertIn("IlluminantTest::calibration_result(raw)", bridge)
        self.assertIn("fj_test_neutral_calibration_lookup(", bridge)
        self.assertNotIn("fj_legacy_neutral_calibration_lookup(", bridge)

    test_synthetic_reference_has_one_safe_owner_and_native_completion_only.required_subtests = True


class NativeBoundaryControls(unittest.TestCase):
    def test_s5c2_ownership_guard_rejects_ambient_and_native_asset_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            source = root / "src/ProcessRoot.cpp"
            source.parent.mkdir(parents=True)
            for text in (
                "JuicerCuda::Owner owner;",
                "Root::instance();",
                "JuicerProcess::root();",
                "borrowed_owner();",
                "HostCleanup cleanup;",
                "JuicerAssets::Library library(path);",
                "FjAssets* _assets;",
                "borrow_rust_assets();",
                "release_cached_payloads();",
                "detach_assets();",
                "JuicerProcess::Root g_root;",
                "Root(const std::filesystem::path& path) {}",
            ):
                source.write_text(text, encoding="utf-8")
                self.assertEqual(len(forbidden_uses(product_sources(root), RETIRED_S5C2_AMBIENT_OWNERS)), 1, text)
            source.write_text(
                "Root& root = JuicerCuda::borrow_root(cuda);\n"
                "const FjHostPreparation* preparation = nullptr;\n",
                encoding="utf-8",
            )
            self.assertEqual(forbidden_uses(product_sources(root), RETIRED_S5C2_AMBIENT_OWNERS), [])

    def test_llvm_nm_uses_active_toolchain_path(self):
        names = ("llvm-nm.exe",) if os.name == "nt" else ("llvm-nm-22", "llvm-nm")
        self.subtest_cases = tuple(f"tool/{name}" for name in names)
        for name in names:
            with self.subTest(case=f"tool/{name}", name=name), tempfile.TemporaryDirectory() as directory:
                toolchain = Path(directory) / "Microsoft Visual Studio/18/Enterprise/VC/Tools/Llvm/x64/bin"
                toolchain.mkdir(parents=True)
                executable = toolchain / name
                executable.write_bytes(b"")
                executable.chmod(0o700)
                with patch.dict(os.environ, {"PATH": str(toolchain)}):
                    self.assertEqual(Path(find_llvm_nm()), executable)

    def test_missing_llvm_nm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PATH": directory}):
            self.assertIsNone(find_llvm_nm())

    def test_illuminant_deletion_guard_rejects_each_retired_definition(self):
        symbols = ("PlanckBlackbodySample", "planck_blackbody", "build_blackbody_curve", "build_illuminant_curve", "build_tungsten_kg3_curve", "prepare_tungsten_kg3_lens_input", "build_tungsten_kg3_lens_curve", "TungstenKg3LensInput", "build_curve_equal_energy_pinned", "SpectralResampleDetail", "scalar_akima_resample", "akima_resample_agx", "resample_pairs_akima_to_reference_axis", "mean_power_normalize")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/Illuminants.cpp"
            source.parent.mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            for symbol in symbols:
                source.write_text(f"void {symbol}();\n", encoding="utf-8")
                self.assertEqual(len(forbidden_uses(color_sources(root), RETIRED_ILLUMINANT_MATH)), 1)
            source.write_text("JuicerIlluminant::blackbody(3200);\nSpectral::samples_follow_reference_axis(rows);\n", encoding="utf-8")
            self.assertEqual(forbidden_uses(color_sources(root), RETIRED_ILLUMINANT_MATH), [])
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
            for text in ("class ProfileAssetStore {};", '#include "ProfileJSONLoader.h"', "src/ProfileAssets.cpp", '#include "ProfileAssets.h"', "class FilmSource {};", "class ConversionSlots {};"):
                source.write_text(text, encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_PROFILES)), 1)
            source.write_text("src/RustAssetBridge.cpp\n", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_PROFILES), [])
            self.assertEqual(len(forbidden_uses([source], RETIRED_S5C2_AMBIENT_OWNERS)), 1)

    def test_spectral_deletions_reject_definitions_includes_and_fixture_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hpp"
            controls = (
                '#include "NpyLoader.h"', "struct NpySpectraLUT {};", "NpyFloat2D basis;",
                "load_csv_triplets(path);", "load_hanatos_spectra_lut(path);",
                "load_arctic2026beta04_spectra_lut(path);", "load_mallett2019_basis_npy(path);",
                "load_npy_float2d(path);", "load_npy_spectra_lut(path);",
                "sourceElementBytes = 2;", "hanatosAssetHash = 1;", "arcticAssetHash = 2;",
                "kExpectedDecodedAssetHash = 3;", "data_file_string(root, name);",
            )
            self.subtest_cases = tuple(f"retired-use/{index:02d}" for index in range(len(controls))) + ("retained-control",)
            for index, text in enumerate(controls):
                with self.subTest(case=f"retired-use/{index:02d}", text=text):
                    source.write_text(text, encoding="utf-8")
                    self.assertEqual(len(forbidden_uses([source], RETIRED_SPECTRAL)), 1)
            with self.subTest(case="retained-control"):
                source.write_text(
                    "ReconstructionLut lut; MallettBasis basis; copy_cmf_triplets(); "
                    "build_film_tc_lut(); build_illuminant_curve(rows, label); set_cie_1931_2deg_cmf(x, y, z);",
                    encoding="utf-8",
                )
                self.assertEqual(forbidden_uses([source], RETIRED_SPECTRAL), [])

            for name in (
                "SpectralContext", "ReconstructionLut", "MallettBasis", "SpectralTables", "CMFTriplets",
                "gShape", "gXBar", "gYBar", "gZBar", "gHanatosAvailable", "gHanSpectra", "gMallettAvailable", "gMallettBasis",
                "hanatos_available", "set_hanatos_available", "hanatos_matches_reference_shape", "arctic_available", "set_arctic_available",
                "mallett_available", "set_mallett_available", "mallett_basis_matches_reference_shape", "copy_hanatos_lut", "copy_arctic_lut",
                "copy_mallett_basis", "copy_cmf_triplets", "load_process_spectral_assets", "ensure_bootstrap", "_bootstrapOnce",
                "log_spectral_warning", "log_resample_failure", "samples_follow_reference_axis", "make_reference_spectral_shape",
                "lock_shape_to_reference_axis", "cmf_triplets_match_reference_axis", "set_cie_1931_2deg_cmf", "spectral_tables_view",
                "JuicerSpectral", "JuicerExposure", "JuicerExposure::reference_white", "JuicerExposure::mallett_midgray",
                "JuicerExposure::reference_source", "JuicerExposure::mallett_reference_raw", "Spectral::MallettBasis",
                "Spectral::context", "Spectral::FilmTcLut",
                "RustSpectralBridge.h", "RustSpectralBridge.cpp", "RustReconstructionBridge.h", "RustReconstructionBridge.cpp",
                "RustExposureBridge.h", "RustExposureBridge.cpp", "SpectralProcessing.cpp", "Scanner.cpp",
            ):
                source.write_text(name, encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_SPECTRAL_COPIES)), 1, name)
            source.write_text("Spectral::Curve curve; Spectral::SpectralShape shape; Spectral::kReferenceShape; "
                              "FjOwnedFilmTcLut owner; FjSpectralTables view; SpectralTablesView tables; "
                              "gDWG_WhitePoint_XYZ; assign_reference_axis(axis); spectral_shape_matches_reference(shape);", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_SPECTRAL_COPIES), [])

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

            for name in ("CsvSource", "CsvRows", "IlluminantCurveView", "IlluminantFilterCurveSet", "copy_csv_pairs",
                         "csv_rows", "illuminant_filter_curves", "JuicerIlluminant", "IlluminantKeys", "Illuminants.h", "Illuminants.cpp"):
                source.write_text(name, encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_ILLUMINANT_COPIES)), 1, name)
            source.write_text("FjCsvPairs* rows; FJ_CSV_D65; FjIlluminantsView view; "
                              "fj_test_assets_illuminants_acquire_observed(); fj_test_illuminants_view(); "
                              "fj_legacy_neutral_calibration_lookup();", encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_ILLUMINANT_COPIES), [])

    def test_noise_source_and_path_authority_stays_deleted(self):
        paths = [path for directory in (ROOT / "src", ROOT / "native", ROOT / "tests/ffi")
                 for path in directory.rglob("*")
                 if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
        self.assertEqual(forbidden_uses(paths, RETIRED_NOISE), [])
        self.assertFalse((ROOT / "src/ProcessRuntime.cpp").exists())
        root_source = (ROOT / "src/ProcessRoot.cpp").read_text(encoding="utf-8")
        self.assertNotRegex(root_source, r"nlohmann|\bJson\b|fstream|ifstream|read_bytes")
        self.assertNotIn("assets().noise()", root_source)
        noise_preflight = root_source.index("if (request.identity.grainActive && !request.noise)")
        self.assertLess(noise_preflight, root_source.index("if (!begin_submission(", noise_preflight))
        self.assertIn("MissingRequiredResource phase=grain_static field=noise", root_source)
        self.assertFalse((ROOT / "src/CudaRenderProjection.cpp").exists())
        fixture = (ROOT / "tests/ffi/prepared_boundary.cpp").read_text(encoding="utf-8")
        self.assertNotIn("fj_test_noise_acquire", fixture)
        self.assertNotIn("fj_legacy_noise_", fixture)
        self.assertIn("PreparedFrameFixture::with_prepared_frame_inputs(", fixture)
        self.assertIn("*inputs.preparation.noise", fixture)
        prepared_fixture = (ROOT / "tests/ffi/prepared_frame_fixture.cpp").read_text(encoding="utf-8")
        self.assertIn("class NoiseHold final", prepared_fixture)
        self.assertIn("fj_test_assets_noise_acquire(assets, &_owner, error)", prepared_fixture)
        self.assertNotIn("fj_legacy_noise_acquire(", prepared_fixture)
        self.assertIn("noiseOwner.acquire(assets, &error)", prepared_fixture)
        self.assertIn("release_noise(noiseOwner, diagnostic)", prepared_fixture)
        native_owner = (ROOT / "native/juicer_cuda_owner.h").read_text(encoding="utf-8")
        native_api = (ROOT / "native/juicer_cuda_api.cpp").read_text(encoding="utf-8")
        noise_test = (ROOT / "tests/ffi/noise_owner_test.cpp").read_text(encoding="utf-8")
        build = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("#if defined(JUICER_NATIVE_CALL_TEST_HOOK)", native_owner)
        self.assertIn("#if defined(JUICER_NATIVE_CALL_TEST_HOOK)", native_api)
        self.assertIn("NativeCallReentryPrecheck", noise_test)
        self.assertIn("calling_thread_native_gate_active()", noise_test)
        self.assertIn("juicer_noise_test_objects PRIVATE JUICER_NATIVE_CALL_TEST_HOOK=1", build)
        self.assertIn("JuicerNoiseOwnerTests PRIVATE JUICER_NATIVE_CALL_TEST_HOOK=1", build)

    def test_noise_deletion_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hpp"
            for name in ("Noise" + "Source", "WangNoisePayload", "StaticNoisePayloadSet",
                         "StaticNoiseAssetSet", "StaticNoisePayloadCacheState", "static_noise_payloads",
                         "load_stbn_noise_payload", "load_wang_noise_payload", "load_static_noise_payloads",
                         "build_static_noise_input", "compatibility_data_directory", "_dataDir"):
                source.write_text(f"// retired {name};\n", encoding="utf-8")
                self.assertEqual(len(forbidden_uses([source], RETIRED_NOISE)), 1)
            source.write_text("StaticNoiseInput view; Library(nativePath); "
                              "fj_legacy_noise_view(owner, out, error); ensure_grain_static_assets_uploaded(view);",
                              encoding="utf-8")
            self.assertEqual(forbidden_uses([source], RETIRED_NOISE), [])

    test_llvm_nm_uses_active_toolchain_path.required_subtests = True
    test_spectral_deletions_reject_definitions_includes_and_fixture_dependencies.required_subtests = True


if __name__ == "__main__":
    unittest.main()
