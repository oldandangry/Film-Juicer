#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustAssetBridge.h"
#include "RustExposureBridge.h"
#include "RustSpectralBridge.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"

extern "C" const std::size_t* fj_test_film_input_abi_c_facts(std::size_t* count);
extern "C" FjStatus fj_test_film_input_c_call(const FjFilmProfile*, const FjFilmInputParams*, const FjFilmInputResources*, FjFilmInput*, FjFilmInputFailure*, FjErrorBuffer*);
static_assert(std::is_same_v<decltype(&fj_legacy_film_input_prepare), FjStatus (*)(const FjFilmProfile*, const FjFilmInputParams*, const FjFilmInputResources*, FjFilmInput*, FjFilmInputFailure*, FjErrorBuffer*)>);

static_assert(std::is_same_v<decltype(FjFilmInputParams::input_space), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::decode_cctf), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::method), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::auto_exposure_enabled), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::meter), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::manual_exposure_ev), float>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::film_format_long_edge_mm), float>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::filter_override), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::uv), double[3]>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::ir), double[3]>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::apply_hanatos_window), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::apply_hanatos_surface), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputParams::input_compression), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::reference_illuminant), FjFloatSpan>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::reference_white_xyz), FjFloatSpan>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::integrated_d65_xyz), FjFloatSpan>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::reconstructed_reference_white), FjFloatSpan>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::tc_source_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInputResources::input_hull_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::color), FjInputColorConversion>);
static_assert(std::is_same_v<decltype(FjFilmInput::xyz_to_linear_srgb), float[9]>);
static_assert(std::is_same_v<decltype(FjFilmInput::nominal_white_xyz), float[3]>);
static_assert(std::is_same_v<decltype(FjFilmInput::projection_white_xyz), float[3]>);
static_assert(std::is_same_v<decltype(FjFilmInput::method), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::auto_exposure_enabled), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::meter), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::manual_exposure_ev), float>);
static_assert(std::is_same_v<decltype(FjFilmInput::film_format_long_edge_mm), float>);
static_assert(std::is_same_v<decltype(FjFilmInput::band_pass_active), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::uv), float[3]>);
static_assert(std::is_same_v<decltype(FjFilmInput::ir), float[3]>);
static_assert(std::is_same_v<decltype(FjFilmInput::apply_hanatos_window), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::apply_hanatos_surface), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::hanatos_spectral_blur), float>);
static_assert(std::is_same_v<decltype(FjFilmInput::hanatos_window), float[4]>);
static_assert(std::is_same_v<decltype(FjFilmInput::hanatos_surface_rgb), float[3][15]>);
static_assert(std::is_same_v<decltype(FjFilmInput::projection_illuminant), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::input_compression_active), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::input_hull_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::tc_source_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::sensitivity_rgb), float[81][3]>);
static_assert(std::is_same_v<decltype(FjFilmInput::sensitivity_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::recipe_mallett_green_scale), float>);
static_assert(std::is_same_v<decltype(FjFilmInput::tc_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInput::hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjFilmInputFailure::origin), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputFailure::reason), uint32_t>);
static_assert(std::is_same_v<decltype(FjFilmInputFailure::sensitivity), FjSensitivityFailure>);

namespace JuicerAssets::InputTest {
    thread_local bool releaseCaches = false;
    thread_local unsigned failureMode = 0;
    thread_local const Profiles::FilmProfile* selectedProfile = nullptr;
    thread_local std::weak_ptr<const Gamut::InputCompressionHull> selectedHull;
    thread_local InstanceState* supersedeState = nullptr;
    thread_local ParamSnapshot newer;
    void before_prepare(const FilmSource& source, const Spektrafilm::FilmFoundationBuildInput& input) {
        if (releaseCaches) {
            releaseCaches = false;
            selectedProfile = source.profile().get();
            selectedHull = input.inputCompressionHull;
            if (JuicerProcess::root().assets().release_cached_payloads().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("cache detach while selected input source/hull remain held");
            }
        }
    }
    void after_prepare(FjStatus& status, FjFilmInputFailure& failure) {
        if (failureMode == 1) {
            failureMode = 0;
            status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            failure = {FJ_FILM_INPUT_ORIGIN_RESOURCE, FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE, {0, 0}};
            if (supersedeState) {
                std::lock_guard<std::mutex> lock(supersedeState->pending.m);
                supersedeState->pending.value = PendingParamsState::Valid{newer, hash_params(newer)};
                supersedeState = nullptr;
            }
        }
    }
    bool translate_failure(FjStatus, const FjFilmInputFailure&, const FjFilmInputParams&, std::string_view, std::string&);
} // namespace JuicerAssets::InputTest

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }
    void status(FjStatus value, std::uint32_t category = FJ_STATUS_SUCCESS) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "closed input status");
    }
    Json words(const float* values, std::size_t count) {
        Json result = Json::array();
        for (std::size_t i = 0; i < count; ++i) {
            result.push_back(std::bit_cast<std::uint32_t>(values[i]));
        }
        return result;
    }
    template <typename Range>
    Json words(const Range& values) {
        Json result = Json::array();
        for (float value : values) {
            result.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return result;
    }
    Json native_values(const FilmRawRecipe& v) {
        Json sensitivity = Json::array(), surface = Json::array();
        for (const auto& row : v.finalSensitivity)
            sensitivity.push_back(words(row));
        for (const auto& row : v.hanatos.surfaceParams)
            surface.push_back(words(row));
        return {{"input_space", v.inputColorSpace}, {"decode_cctf", v.inputCctfDecoding}, {"method", static_cast<unsigned>(v.rgbToRawMethod)}, {"auto_exposure", v.autoExposureEnabled}, {"meter", static_cast<unsigned>(v.autoExposureMethod)}, {"ev_mm", words(std::array{v.manualExposureCompensationEv, v.filmFormatLongEdgeMm})}, {"filter_active", v.cameraBandPass.active}, {"uv", words(v.cameraBandPass.uv)}, {"ir", words(v.cameraBandPass.ir)}, {"rgb_to_xyz", words(v.inputRgbToXyz)}, {"xyz_adapt", words(v.inputXyzAdapt)}, {"xyz_to_linear_srgb", words(v.xyzToLinearSrgb)}, {"nominal_white", words(v.inputNominalWhiteXYZ)}, {"projection_white", words(v.projectionWhiteXYZ)}, {"reference_label", v.referenceIlluminant}, {"projection_label", v.projectionIlluminant}, {"hanatos_label", v.hanatos.referenceIlluminant}, {"window_enabled", v.hanatos.applyWindow}, {"surface_enabled", v.hanatos.applySurface}, {"blur", words(std::array{v.hanatos.spectralGaussianBlur})}, {"window", words(v.hanatos.windowParams)}, {"surface", surface}, {"compression", v.inputCompressionActive}, {"hull_hash", v.inputCompressionHullHash}, {"source_hash", v.tcSourceAssetHash}, {"sensitivity", sensitivity}, {"sensitivity_hash", v.finalSensitivityHash}, {"mallett_scale", words(std::array{v.mallettGreenMidgrayScale})}, {"tc_hash", v.tcLutHash}, {"hash", v.hash}};
    }
    Json projected(const FjFilmInput& v) {
        Json sensitivity = Json::array(), surface = Json::array();
        for (const auto& row : v.sensitivity_rgb) {
            sensitivity.push_back(words(row, 3));
        }
        for (const auto& row : v.hanatos_surface_rgb) {
            surface.push_back(words(row, 15));
        }
        return {{"input_space", v.color.input_space}, {"decode_cctf", bool(v.color.decode_cctf)}, {"method", v.method}, {"auto_exposure", bool(v.auto_exposure_enabled)}, {"meter", v.meter}, {"ev_mm", Json::array({std::bit_cast<std::uint32_t>(v.manual_exposure_ev), std::bit_cast<std::uint32_t>(v.film_format_long_edge_mm)})}, {"filter_active", bool(v.band_pass_active)}, {"uv", words(v.uv, 3)}, {"ir", words(v.ir, 3)}, {"rgb_to_xyz", words(v.color.rgb_to_xyz, 9)}, {"xyz_adapt", words(v.color.xyz_adapt, 9)}, {"xyz_to_linear_srgb", words(v.xyz_to_linear_srgb, 9)}, {"nominal_white", words(v.nominal_white_xyz, 3)}, {"projection_white", words(v.projection_white_xyz, 3)}, {"reference_label", "D55"}, {"projection_label", v.projection_illuminant == FJ_FILM_INPUT_ILLUMINANT_PROFILE_REFERENCE ? "D55" : "D65"}, {"hanatos_label", v.method == FJ_RAW_HANATOS_2025 ? "D55" : ""}, {"window_enabled", bool(v.apply_hanatos_window)}, {"surface_enabled", bool(v.apply_hanatos_surface)}, {"blur", words(&v.hanatos_spectral_blur, 1)}, {"window", words(v.hanatos_window, 4)}, {"surface", surface}, {"compression", bool(v.input_compression_active)}, {"hull_hash", v.input_hull_hash}, {"source_hash", v.tc_source_hash}, {"sensitivity", sensitivity}, {"sensitivity_hash", v.sensitivity_hash}, {"mallett_scale", words(&v.recipe_mallett_green_scale, 1)}, {"tc_hash", v.tc_hash}, {"hash", v.hash}};
    }
    void layouts() {
        const std::size_t cpp[] = {
#define FJ_INPUT_TYPE(type) sizeof(type), alignof(type),
#define FJ_INPUT_FIELD(type, field) offsetof(type, field),
#define FJ_INPUT_EXTENT(type, field, element) sizeof(type::field) / sizeof(element),
#define FJ_INPUT_VALUE(value) value,
#include "film_input_abi_facts.inc"
#undef FJ_INPUT_TYPE
#undef FJ_INPUT_FIELD
#undef FJ_INPUT_EXTENT
#undef FJ_INPUT_VALUE
        };
        std::size_t cCount = 0, rustCount = 0;
        const auto* c = fj_test_film_input_abi_c_facts(&cCount);
        const auto* rust = fj_test_film_input_abi_facts(&rustCount);
        require(cCount == std::size(cpp) && rustCount == cCount, "complete layout membership");
        for (std::size_t i = 0; i < cCount; ++i) {
            require(cpp[i] == c[i] && c[i] == rust[i], "C11/C++20/Rust fields/extents/tags agree");
        }
        std::printf("PASS %zu coordinated film-input ABI facts\n", cCount);
    }
    struct Profile {
        FjAssets* assets = nullptr;
        FjFilmProfile* film = nullptr;
        Profile(const std::filesystem::path& root, const std::string& key) {
            const JuicerAssets::NativePathArgument path(root);
            status(fj_legacy_assets_create(path.view(), &assets, nullptr));
            status(fj_legacy_film_profile_acquire(assets, {key.data(), key.size()}, &film, nullptr));
        }
        ~Profile() {
            if (fj_legacy_film_profile_release(film, nullptr).category != FJ_STATUS_SUCCESS ||
                fj_legacy_assets_destroy(assets, nullptr).category != FJ_STATUS_SUCCESS) {
                std::terminate();
            }
        }
    };
    FjFilmInputParams params(const Json& row) {
        FjFilmInputParams out{};
        out.input_space = row.at("space");
        out.decode_cctf = row.at("decode");
        out.method = row.at("method");
        out.film_format_long_edge_mm = 35.0f;
        out.uv[0] = 1.0;
        out.uv[1] = 410.0;
        out.uv[2] = 8.0;
        out.ir[0] = 1.0;
        out.ir[1] = 675.0;
        out.ir[2] = 15.0;
        out.apply_hanatos_window = 1;
        if (row.contains("control")) {
            const unsigned test = row.at("control");
            out.auto_exposure_enabled = test % 2;
            out.input_compression = test % 2;
            out.meter = test % 7;
            out.manual_exposure_ev = static_cast<float>(test % 2 ? 1.234567890123 : -0.0);
            out.film_format_long_edge_mm = static_cast<float>(test % 2 ? 36.123456789012 : 35.0);
            out.apply_hanatos_window = (test / 2) % 2;
            out.apply_hanatos_surface = test % 2;
            out.filter_override = test % 2;
            out.uv[0] = 0.314159265358979;
            out.uv[1] = 413.123456789012;
            out.uv[2] = 9.123456789012;
            out.ir[0] = 0.271828182845904;
            out.ir[1] = 679.123456789012;
            out.ir[2] = 17.123456789012;
        }
        return out;
    }
    FjFilmInput poisoned() {
        FjFilmInput value{};
        value.color.input_space = value.color.decode_cctf = value.color.adapt_xyz = 99;
        std::fill_n(value.color.rgb_to_xyz, 9, 1.0f);
        std::fill_n(value.color.xyz_adapt, 9, 1.0f);
        std::fill_n(value.xyz_to_linear_srgb, 9, 1.0f);
        std::fill_n(value.nominal_white_xyz, 3, 1.0f);
        std::fill_n(value.projection_white_xyz, 3, 1.0f);
        value.method = 99;
        value.auto_exposure_enabled = 99;
        value.meter = 99;
        value.manual_exposure_ev = 1.0f;
        value.film_format_long_edge_mm = 1.0f;
        value.band_pass_active = 99;
        std::fill_n(value.uv, 3, 1.0f);
        std::fill_n(value.ir, 3, 1.0f);
        value.apply_hanatos_window = 99;
        value.apply_hanatos_surface = 99;
        value.hanatos_spectral_blur = 1.0f;
        std::fill_n(value.hanatos_window, 4, 1.0f);
        for (auto& row : value.hanatos_surface_rgb) {
            std::fill_n(row, 15, 1.0f);
        }
        value.projection_illuminant = 99;
        value.input_compression_active = 99;
        value.input_hull_hash = 99;
        value.tc_source_hash = 99;
        for (auto& row : value.sensitivity_rgb) {
            std::fill_n(row, 3, 1.0f);
        }
        value.sensitivity_hash = 99;
        value.recipe_mallett_green_scale = 1.0f;
        value.tc_hash = 99;
        value.hash = 99;
        return value;
    }
    void cleared(const FjFilmInput& out) {
        const auto value = projected(out);
        for (auto i = value.begin(); i != value.end(); ++i) {
            if (i.value().is_string()) {
                continue;
            }
            const auto all_zero = [](const auto& self, const Json& value) -> bool {
                if (value.is_array()) {
                    for (const auto& element : value) {
                        if (!self(self, element)) {
                            return false;
                        }
                    }
                    return true;
                }
                return value == 0 || value == false;
            };
            require(all_zero(all_zero, i.value()), "every semantic result field clears to positive zero");
        }
        require(out.color.adapt_xyz == 0 && out.projection_illuminant == 0, "adaptation and projection tag clear");
    }
    void native_products(const Json& fixture) {
        for (const auto& row : fixture.at("rows")) {
            ParamSnapshot controls;
            controls.filmProfileKey = row.at("key");
            controls.printProfileKey = "kodak_portra_endura";
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            controls.inputColorSpace = row.at("space");
            controls.inputCctfDecoding = row.at("decode");
            controls.spectralUpsamplingMode = row.at("method");
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.grainControls.active = false;
            if (row.contains("control")) {
                const unsigned test = row.at("control");
                controls.cameraAutoExposureEnabled = static_cast<int>(test % 2);
                controls.inputCompressionEnabled = static_cast<int>(test % 2);
                controls.cameraMeteringMethod = static_cast<int>(test % 7);
                controls.cameraExposureCompensationEv = test % 2 ? 1.234567890123 : -0.0;
                controls.cameraFilmFormatLongEdgeMm = test % 2 ? 36.123456789012 : 35.0;
                controls.hanatos2025AdaptationWindow = static_cast<int>((test / 2) % 2);
                controls.hanatos2025AdaptationSurface = static_cast<int>(test % 2);
                controls.cameraFilterOverride = test % 2;
                controls.cameraFilterUV = {0.314159265358979, 413.123456789012, 9.123456789012};
                controls.cameraFilterIR = {0.271828182845904, 679.123456789012, 17.123456789012};
            }
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            const bool built = Spektrafilm::scan_route_is_print(controls.scanRoute)
                                   ? build_print_render_state_product(controls, product, diagnostic)
                                   : build_direct_render_state_product(controls, product, diagnostic);
            require(built, diagnostic.c_str());
            require(native_values(product.recipe.filmRaw) == row.at("raw"), "actual native family exactly matches independent compound capture");
            require(product.recipe.hash == row.at("final_hash") && hash_params(controls) == row.at("snapshot_hash"), "complete native product and captured-control identities remain exact");
        }
        std::puts("PASS 140 actual native direct/print family, final and snapshot products");
    }
    void lifetime_and_admission() {
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot controls;
            controls.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 1;
            controls.outputGamutCompressionEnabled = 0;
            controls.grainControls.active = false;
            InstanceState state;
            const auto pending = [&state](const ParamSnapshot& snapshot) {
                std::lock_guard<std::mutex> lock(state.pending.m);
                state.pending.value = PendingParamsState::Valid{snapshot, hash_params(snapshot)};
            };
            JuicerAssets::InputTest::releaseCaches = true;
            pending(controls);
            const auto old = admit_pending_render_state(state);
            require(route % 2 ? bool(old.printState) : bool(old.directState), "source/hull release during synchronous preparation still completes");
            const auto& heldRecipe = route % 2 ? old.printState->recipe : old.directState->recipe;
            require(heldRecipe.profileRoute.filmProfile.get() == JuicerAssets::InputTest::selectedProfile &&
                        heldRecipe.filmRaw.inputCompressionHull == JuicerAssets::InputTest::selectedHull.lock(),
                    "same captured source/profile and hull hold survive cache detachment and success binding");
            const auto heldHash = heldRecipe.hash;
            const auto counter = state.buildCounterNext.load();
            const auto identity = state.lastHash.load();
            auto changed = controls;
            changed.cameraExposureCompensationEv = 0.25;
            pending(changed);
            JuicerAssets::InputTest::failureMode = 1;
            const auto rejected = admit_pending_render_state(state);
            require(rejected.status == PendingRenderAdmissionStatus::RebuildFailed && !rejected.directState && !rejected.printState, "returned input failure prevents old-state admission");
            require(!JuicerAtomic::load_shared_ptr(&state.activeDirectState) && !JuicerAtomic::load_shared_ptr(&state.activePrintState) &&
                        state.lastHash.load() == identity && state.buildCounterNext.load() == counter && heldRecipe.hash == heldHash,
                    "ordinary input failure clears selected slot without publishing or revoking admitted holds");
            pending(controls);
            const auto recovered = admit_pending_render_state(state);
            require(route % 2 ? bool(recovered.printState) : bool(recovered.directState), "ordinary input failure recovers");
            const auto direct = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
            const auto print = JuicerAtomic::load_shared_ptr(&state.activePrintState);
            pending(changed);
            status(fj_test_color_arm_fault(5, 1, 1));
            const auto thrown = admit_pending_render_state(state);
            require(thrown.status == PendingRenderAdmissionStatus::RebuildFailed && !thrown.directState && !thrown.printState &&
                        JuicerAtomic::load_shared_ptr(&state.activeDirectState) == direct && JuicerAtomic::load_shared_ptr(&state.activePrintState) == print,
                    "thrown input failure retains publication but cannot admit it for changed controls");
            auto newer = changed;
            newer.cameraExposureCompensationEv = 0.5;
            JuicerAssets::InputTest::supersedeState = &state;
            JuicerAssets::InputTest::newer = newer;
            JuicerAssets::InputTest::failureMode = 1;
            const auto superseded = admit_pending_render_state(state);
            require(route % 2 ? bool(superseded.printState) : bool(superseded.directState), "superseded input failure recovers to a complete newer state");
            const auto& current = route % 2 ? superseded.printState->recipe : superseded.directState->recipe;
            require(current.filmRaw.manualExposureCompensationEv == 0.5f && heldRecipe.hash == heldHash, "superseded failed input preparation retries only newer controls and retains old immutable holds");
            for (unsigned selector = 0; selector < 3; ++selector) {
                auto invalid = controls;
                if (selector == 0) {
                    invalid.inputColorSpace = -1;
                }
                if (selector == 1) {
                    invalid.spectralUpsamplingMode = -1;
                }
                if (selector == 2) {
                    invalid.cameraMeteringMethod = -1;
                }
                status(fj_test_color_arm_fault(5, 1, 1));
                FocusedRenderStateBuildProduct product;
                std::string diagnostic;
                const bool built = route % 2 ? build_print_render_state_product(invalid, product, diagnostic) : build_direct_render_state_product(invalid, product, diagnostic);
                require(!built && diagnostic.starts_with("UnsupportedMode phase=3B field="), "native signed selectors retain ordinary precheck failure before raw packing");
                bool propagated = false;
                try {
                    (void)(route % 2 ? build_print_render_state_product(controls, product, diagnostic) : build_direct_render_state_product(controls, product, diagnostic));
                } catch (const JuicerCuda::ExecutionFailure& error) {
                    propagated = error.failure.diagnostic == "Input matrix preparation failed";
                }
                require(propagated, "each native selector precheck preserves the armed matrix fault for the next valid call");
            }
        }
        std::puts("PASS actual source/hull lifetimes, slot return/throw distinctions, admission, supersession, recovery and selector priority");
    }
    void native_resource_prechecks() {
        auto& assets = JuicerProcess::root().assets();
        for (unsigned route = 0; route < 4; ++route) {
            const auto selected = assets.selected_profiles_for_route({route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f", "kodak_portra_endura", static_cast<Spektrafilm::ScanRoute>(route)});
            require(selected.valid && selected.filmSource && (route % 2 == 0 || selected.printSource), "selected native precheck fixture sources");
            for (bool missingProfile : {true, false}) {
                status(fj_test_color_arm_fault(5, 1, 1));
                std::string diagnostic;
                bool valid = false;
                if (route % 2) {
                    Spektrafilm::PrintRecipeBuildInput input;
                    input.film.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                    input.film.filmSource = missingProfile ? nullptr : selected.filmSource.get();
                    input.film.inputColorSpace = -1;
                    input.printSource = selected.printSource;
                    const auto result = Spektrafilm::build_print_render_recipe(std::move(input));
                    valid = result.valid;
                    diagnostic = result.diagnostic;
                } else {
                    Spektrafilm::DirectRecipeBuildInput input;
                    input.film.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                    input.film.filmSource = missingProfile ? nullptr : selected.filmSource.get();
                    input.film.inputColorSpace = -1;
                    const auto result = Spektrafilm::build_direct_render_recipe(std::move(input));
                    valid = result.valid;
                    diagnostic = result.diagnostic;
                }
                const char* expected = missingProfile ? (route % 2 ? "MissingRequiredResource phase=4A field=selected_profile" : "MissingRequiredResource phase=3B field=film_profile") : "MissingRequiredResource phase=3B field=reference_illuminant";
                require(!valid && diagnostic == expected, "native missing profile/reference priority precedes selectors and matrix faults");
                FjInputColorMatrices matrices{};
                status(fj_legacy_input_matrices(FJ_INPUT_DWG, &matrices), FJ_STATUS_UNSUPPORTED_INPUT);
            }
        }
    }
    void routing() {
        enum class Domain : std::uint8_t {
            Any,
            Hanatos,
            Tc,
            Mallett,
            CompressedTc
        };
        struct Row {
            unsigned origin;
            unsigned reason;
            unsigned categories;
            Domain domain;
        };
        // Literal origin/reason/category codes independently pin the approved wire envelope.
        std::vector<Row> rows{{0, 0, 1, Domain::Any}};
        for (unsigned reason = 1; reason <= 7; ++reason) {
            rows.push_back({1, reason, 1u << 2, Domain::Any});
        }
        for (unsigned reason = 1; reason <= 6; ++reason) {
            rows.push_back({2, reason, 1u << 3, reason <= 3 ? Domain::Hanatos : reason == 6 ? Domain::CompressedTc
                                                                                            : Domain::Tc});
        }
        rows.push_back({3, 1, 0x1fe, Domain::Any});
        rows.push_back({3, 2, 0x1fe, Domain::Tc});
        rows.push_back({3, 3, 0x1fe, Domain::Mallett});
        rows.push_back({4, 1, 1u << 3, Domain::Any});
        rows.push_back({4, 2, 0x1fe, Domain::Any});
        rows.push_back({5, 1, 1u << 3, Domain::Any});
        for (unsigned reason = 1; reason <= 6; ++reason) {
            rows.push_back({6, reason, 1u << 8, reason == 4 ? Domain::Tc : reason == 5 ? Domain::Mallett
                                                                                       : Domain::Any});
        }
        rows.push_back({99, 99, 0, Domain::Any});
        rows.push_back({6, 99, 0, Domain::Any});
        for (const auto& row : rows) {
            for (unsigned method = 0; method < 3; ++method) {
                for (unsigned compression = 0; compression < 2; ++compression) {
                    for (unsigned category = 0; category <= 9; ++category) {
                        FjFilmInputParams params{};
                        params.method = method;
                        params.input_compression = compression;
                        const bool tc = method != FJ_RAW_MALLETT_2019;
                        const bool applicable = row.domain == Domain::Any || (row.domain == Domain::Hanatos && method == 0) ||
                                                (row.domain == Domain::Tc && tc) || (row.domain == Domain::Mallett && !tc) || (row.domain == Domain::CompressedTc && tc && compression);
                        const bool valid = applicable && (row.categories & (1u << category));
                        FjFilmInputFailure failure{row.origin, row.reason, {0, 0}};
                        std::string diagnostic;
                        bool returned = false, threw = false, allocated = false;
                        try {
                            returned = JuicerAssets::InputTest::translate_failure({category, FJ_API_NONE, 0}, failure, params, "opaque boundary diagnostic", diagnostic);
                        } catch (const std::bad_alloc&) {
                            allocated = true;
                        } catch (const JuicerCuda::ExecutionFailure& error) {
                            threw = true;
                            status(error.failure.status, valid ? category : FJ_STATUS_INTERNAL_FAILURE);
                            if (!valid) {
                                require(error.failure.diagnostic == "Film input preparation: invalid failure contract", "impossible failure tuple is terminal integrity failure");
                            }
                        }
                        const bool ordinary = valid && (row.origin == 2 || row.origin == 5 || (row.origin == 4 && category == 3));
                        const bool success = valid && category == 0;
                        const bool allocation = valid && row.origin == 4 && row.reason == 2 && category == 4;
                        require(returned == success && allocated == allocation && threw == (!ordinary && !success && !allocation), "every closed origin/reason/category/method/compression routing row");
                        if (ordinary) {
                            require(!diagnostic.empty(), "ordinary return preserves owning diagnostic");
                        }
                    }
                }
            }
        }
        for (const auto malformed : {FjSensitivityFailure{0, 1}, {1, 243}, {2, 1}, {3, 0}}) {
            FjFilmInputParams params{};
            FjFilmInputFailure failure{4, 1, malformed};
            std::string diagnostic;
            bool rejected = false;
            try {
                (void)JuicerAssets::InputTest::translate_failure({3, 0, 0}, failure, params, "ignored", diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected, "invalid sensitivity metadata cannot dispatch");
        }
        for (auto value : {FjStatus{0, 1, 0}, FjStatus{0, 0, 1}}) {
            FjFilmInputParams params{};
            FjFilmInputFailure failure{};
            std::string diagnostic;
            bool rejected = false;
            try {
                (void)JuicerAssets::InputTest::translate_failure(value, failure, params, "ignored", diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
            }
            require(rejected, "API and native-code fields are closed for CPU preparation");
        }
        std::puts("PASS complete native failure-envelope and origin-sensitive return/throw table");
    }
    void cases(const std::filesystem::path& root, const Json& fixture) {
        auto& assets = JuicerProcess::root().assets();
        const auto illuminants = assets.illuminant_filter_curves();
        Scanner::ScannerIlluminant reference, d65;
        require(JuicerSpectral::integrate_white(illuminants->d55, "D55", reference) &&
                    JuicerSpectral::integrate_white(illuminants->d65, "D65", d65),
                "integrated input whites");
        std::array<float, 81> reconstructed{};
        std::string diagnostic;
        require(JuicerExposure::reference_white(Spectral::gHanSpectra, 0.0f, {reference.whiteXYZ[0], reference.whiteXYZ[1], reference.whiteXYZ[2]}, reconstructed, diagnostic), "preceding reference leaf");
        const auto hull = assets.input_compression_hull();
        for (const auto& row : fixture.at("rows")) {
            Profile profile(root, row.at("key"));
            const auto input = params(row);
            const FjFilmInputResources resources{
                {illuminants->d55.linear.data(), 81}, {reference.whiteXYZ, 3}, {d65.whiteXYZ, 3}, {reconstructed.data(), 81}, input.method == FJ_RAW_ARCTIC_2026_BETA04 ? Spectral::context().arcticSpectra.assetHash : Spectral::gHanSpectra.assetHash, input.input_compression ? hull->hash : 0};
            for (auto call : {&fj_legacy_film_input_prepare, &fj_test_film_input_prepare, &fj_test_film_input_c_call}) {
                FjFilmInput result = poisoned();
                FjFilmInputFailure failure{99, 99, {99, 99}};
                status(call(profile.film, &input, &resources, &result, &failure, nullptr));
                require(result.color.adapt_xyz == 1 && failure.origin == 0 && failure.reason == 0 && failure.sensitivity.hash_failure == 0 && failure.sensitivity.hash_sample_index == 0, "completed adaptation/success envelope");
                const auto actual = projected(result);
                if (actual != row.at("raw")) {
                    for (auto i = actual.begin(); i != actual.end(); ++i) {
                        if (i.value() != row.at("raw").at(i.key())) {
                            std::fprintf(stderr, "mismatch key=%s route=%u space=%u method=%u field=%s\n", row.at("key").get<std::string>().c_str(), row.at("route").get<unsigned>(), input.input_space, input.method, i.key().c_str());
                        }
                    }
                    throw std::runtime_error("independent native compound expectation");
                }
                auto corrupted = row.at("raw");
                corrupted["xyz_to_linear_srgb"][0] = corrupted.at("xyz_to_linear_srgb")[0].get<std::uint32_t>() ^ 1u;
                require(actual != corrupted, "corrupted matrix expectation is rejected");
            }
        }
        Profile profile(root, "kodak_portra_400");
        auto input = params(fixture.at("rows").at(0));
        FjFilmInputResources resources{{illuminants->d55.linear.data(), 81}, {reference.whiteXYZ, 3}, {d65.whiteXYZ, 3}, {reconstructed.data(), 81}, Spectral::gHanSpectra.assetHash, 0};
        FjFilmInput result{};
        FjFilmInputFailure failure{};
        std::array<char, 8> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        auto inactiveHold = resources;
        inactiveHold.input_hull_hash = hull->hash;
        status(fj_legacy_film_input_prepare(profile.film, &input, &inactiveHold, &result, &failure, nullptr));
        require(result.input_hull_hash == hull->hash && result.hash == fixture.at("rows")[0].at("raw").at("hash"), "inactive TC hold identity is retained without a conditional hash contribution");
        for (unsigned test = 0; test < 13; ++test) {
            auto badInput = input;
            auto badResources = resources;
            FjFilmInput* out = &result;
            FjFilmInputFailure* failed = &failure;
            const FjFilmProfile* selected = profile.film;
            const FjFilmInputParams* controls = &badInput;
            const FjFilmInputResources* samples = &badResources;
            unsigned reason = FJ_FILM_INPUT_RAW_RECORDS;
            if (test == 0) {
                out = nullptr;
            }
            if (test == 1) {
                failed = nullptr;
            }
            if (test == 2) {
                selected = nullptr;
            }
            if (test == 3) {
                controls = nullptr;
            }
            if (test == 4) {
                samples = nullptr;
            }
            if (test == 5) {
                badInput.input_space = 4;
                reason = FJ_FILM_INPUT_RAW_INPUT_SPACE;
            }
            if (test == 6) {
                badInput.method = 3;
                reason = FJ_FILM_INPUT_RAW_METHOD;
            }
            if (test == 7) {
                badInput.meter = 7;
                reason = FJ_FILM_INPUT_RAW_METER;
            }
            if (test == 8) {
                badInput.decode_cctf = 2;
                reason = FJ_FILM_INPUT_RAW_FLAGS;
            }
            if (test == 9) {
                badResources.reference_illuminant.count = 80;
                reason = FJ_FILM_INPUT_RAW_SPANS;
            }
            if (test == 10) {
                badResources.reference_white_xyz.count = 0;
                reason = FJ_FILM_INPUT_RAW_SPANS;
            }
            if (test == 11) {
                badResources.integrated_d65_xyz.data = nullptr;
                reason = FJ_FILM_INPUT_RAW_SPANS;
            }
            if (test == 12) {
                badResources.reconstructed_reference_white.count = std::numeric_limits<std::size_t>::max();
                reason = FJ_FILM_INPUT_RAW_SPANS;
            }
            result = poisoned();
            failure = {99, 99, {99, 99}};
            status(fj_legacy_film_input_prepare(selected, controls, samples, out, failed, &error), FJ_STATUS_UNSUPPORTED_INPUT);
            if (out) {
                cleared(*out);
            }
            if (failed) {
                require(failure.origin == FJ_FILM_INPUT_ORIGIN_RAW && failure.reason == reason && failure.sensitivity.hash_failure == 0 && failure.sensitivity.hash_sample_index == 0, "raw precedence/reason and independent metadata clearing");
            }
            require(error.length == bytes.size() - 1 && bytes.back() == '\0', "bounded diagnostic truncation");
        }
        for (auto field : {&FjFilmInputParams::decode_cctf, &FjFilmInputParams::auto_exposure_enabled, &FjFilmInputParams::filter_override, &FjFilmInputParams::apply_hanatos_window, &FjFilmInputParams::apply_hanatos_surface, &FjFilmInputParams::input_compression}) {
            auto malformed = input;
            malformed.method = FJ_RAW_MALLETT_2019;
            malformed.*field = 2;
            status(fj_legacy_film_input_prepare(profile.film, &malformed, &resources, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(failure.origin == FJ_FILM_INPUT_ORIGIN_RAW && failure.reason == FJ_FILM_INPUT_RAW_FLAGS, "every inactive flag remains closed");
            cleared(result);
        }
        for (auto field : {&FjFilmInputResources::reference_illuminant, &FjFilmInputResources::reference_white_xyz, &FjFilmInputResources::integrated_d65_xyz, &FjFilmInputResources::reconstructed_reference_white}) {
            for (unsigned defect = 0; defect < 5; ++defect) {
                auto malformed = resources;
                if (defect == 0) {
                    (malformed.*field).count = 0;
                }
                if (defect == 1) {
                    (malformed.*field).data = nullptr;
                }
                if (defect == 2) {
                    ++(malformed.*field).count;
                }
                if (defect == 3) {
                    (malformed.*field).data = reinterpret_cast<const float*>(std::uintptr_t{1});
                }
                if (defect == 4) {
                    (malformed.*field).data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3);
                }
                status(fj_legacy_film_input_prepare(profile.film, &input, &malformed, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                require(failure.reason == FJ_FILM_INPUT_RAW_SPANS, "every supplied span is structurally admitted before borrowing");
                cleared(result);
            }
        }
        for (auto operation : {5u, 1u, 3u}) {
            auto selected = input;
            if (operation == 3) {
                selected.method = FJ_RAW_MALLETT_2019;
            }
            for (unsigned fault : {1u, 2u}) {
                status(fj_test_color_arm_fault(operation, 1, fault));
                status(fj_legacy_film_input_prepare(profile.film, &selected, &resources, &result, &failure, nullptr), fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                require(failure.origin == (fault == 1 ? FJ_FILM_INPUT_ORIGIN_COLOR : FJ_FILM_INPUT_ORIGIN_INTERNAL), "leaf error and caught panic retain distinct origins");
                const auto colorReason = operation == 5 ? FJ_FILM_INPUT_COLOR_MATRICES : operation == 1 ? FJ_FILM_INPUT_COLOR_CAT16
                                                                                                        : FJ_FILM_INPUT_COLOR_CAT02;
                require(failure.reason == (fault == 1 ? colorReason : colorReason + 2), "narrow color panic provenance");
                cleared(result);
                status(fj_legacy_film_input_prepare(profile.film, &selected, &resources, &result, &failure, nullptr));
            }
        }
        for (unsigned fault = 1; fault <= 4; ++fault) {
            status(fj_test_exposure_arm_fault(2, 1, fault));
            status(fj_legacy_film_input_prepare(profile.film, &input, &resources, &result, &failure, nullptr), fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                                                      : fault == 3   ? FJ_STATUS_ALLOCATION_FAILURE
                                                                                                                                                                     : FJ_STATUS_PREPARATION_FAILURE);
            require(failure.origin == (fault == 2 ? FJ_FILM_INPUT_ORIGIN_INTERNAL : FJ_FILM_INPUT_ORIGIN_SENSITIVITY) && failure.reason == (fault == 2 ? FJ_FILM_INPUT_INTERNAL_SENSITIVITY_PANIC : FJ_FILM_INPUT_SENSITIVITY_BOUNDARY), "sensitivity boundary provenance/category");
            require(fj_test_exposure_fault_consumed(2, fault) == 1, "fault consumed once at its sensitivity stage");
            cleared(result);
        }
        status(fj_test_color_arm_fault(5, 1, 1));
        auto invalid = input;
        invalid.meter = 99;
        status(fj_legacy_film_input_prepare(profile.film, &invalid, &resources, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(failure.origin == FJ_FILM_INPUT_ORIGIN_RAW, "raw tags reject before semantic faults");
        status(fj_legacy_film_input_prepare(profile.film, &input, &resources, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(failure.origin == FJ_FILM_INPUT_ORIGIN_COLOR && failure.reason == FJ_FILM_INPUT_COLOR_MATRICES, "raw rejection preserves the armed matrix fault");
        auto missing = resources;
        missing.reference_white_xyz = {nullptr, 0};
        status(fj_test_color_arm_fault(5, 1, 1));
        status(fj_legacy_film_input_prepare(profile.film, &input, &missing, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(failure.reason == FJ_FILM_INPUT_COLOR_MATRICES, "matrix failure precedes absent projection white");
        auto compression = input;
        compression.input_compression = 1;
        status(fj_test_color_arm_fault(1, 1, 1));
        status(fj_legacy_film_input_prepare(profile.film, &compression, &resources, &result, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(failure.reason == FJ_FILM_INPUT_COLOR_CAT16, "CAT16 precedes active hull requirement");
        missing = resources;
        missing.reconstructed_reference_white = {nullptr, 0};
        status(fj_legacy_film_input_prepare(profile.film, &compression, &missing, &result, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        require(failure.origin == FJ_FILM_INPUT_ORIGIN_RESOURCE && failure.reason == FJ_FILM_INPUT_RESOURCE_INPUT_HULL, "hull precedes reconstruction requirement");
        for (unsigned stage : {1u, 2u, 3u}) {
            fj_test_film_input_arm_boundary_fault(stage);
            status(fj_legacy_film_input_prepare(profile.film, &input, stage == 2 ? &missing : &resources, &result, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
            require(failure.origin == FJ_FILM_INPUT_ORIGIN_INTERNAL && failure.reason == FJ_FILM_INPUT_INTERNAL_PANIC, "projection/formatting/cleanup panic containment");
            cleared(result);
        }
        status(fj_legacy_assets_release_cached_payloads(profile.assets, nullptr));
        status(assets.release_cached_payloads());
        status(fj_legacy_film_input_prepare(profile.film, &input, &resources, &result, &failure, nullptr));
        require(result.hash == fixture.at("rows")[0].at("raw").at("hash"), "held source and arrays survive cache release");
        resources.reference_white_xyz = {nullptr, 0};
        status(fj_legacy_film_input_prepare(profile.film, &input, &resources, &result, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        require(failure.origin == FJ_FILM_INPUT_ORIGIN_RESOURCE && failure.reason == FJ_FILM_INPUT_RESOURCE_PROJECTION_WHITE, "semantic projection-white requirement");
        cleared(result);
        input.method = FJ_RAW_MALLETT_2019;
        resources.tc_source_hash = 0;
        resources.input_hull_hash = 0;
        resources.integrated_d65_xyz = {nullptr, 0};
        resources.reconstructed_reference_white = {nullptr, 0};
        status(fj_legacy_film_input_prepare(profile.film, &input, &resources, &result, &failure, nullptr));
        require(result.tc_hash == 0 && result.tc_source_hash == 0 && result.input_hull_hash == 0 && result.apply_hanatos_window == 1 && result.apply_hanatos_surface == 0, "inactive Mallett values");
        error.capacity = 0;
        error.data = reinterpret_cast<char*>(std::uintptr_t{1});
        status(fj_legacy_film_input_prepare(nullptr, nullptr, nullptr, &result, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(error.length == 0, "zero-capacity diagnostics do not access backing");
        error.capacity = 1;
        error.data = nullptr;
        status(fj_legacy_film_input_prepare(nullptr, nullptr, nullptr, &result, &failure, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(failure.reason == FJ_FILM_INPUT_RAW_DIAGNOSTIC && error.length == 0, "diagnostic defect precedes missing records");
    }
} // namespace

int main(int argc, char** argv) try { // NOLINT(bugprone-exception-escape): all fixture failures return nonzero.
    require(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "products"), "resource root and immutable preset fixture required");
    layouts();
    routing();
    std::ifstream stream(argv[2]);
    const auto fixture = Json::parse(stream);
    require(fixture.at("rows").size() == 140, "exact nonempty native case membership");
    JuicerCuda::Owner owner;
    owner.create(std::filesystem::path(argv[1]));
    JuicerProcess::root().ensure_bootstrap();
    if (argc == 4) {
        native_products(fixture);
        lifetime_and_admission();
        native_resource_prechecks();
    } else {
        cases(argv[1], fixture);
    }
    std::puts("PASS 140 native film-input cases through core facade, legacy wrapper and C11 caller; raw failures");
    return 0;
} catch (const JuicerCuda::ExecutionFailure& error) {
    std::fprintf(stderr, "unexpected native category=%u api=%u code=%d: %s\n", error.failure.status.category, error.failure.status.api, error.failure.status.native_code, error.failure.diagnostic.c_str());
    return 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    return 1;
}
