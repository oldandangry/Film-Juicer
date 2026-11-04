#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <cstring>
#include <vector>
#include <utility>

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "case_results.h"

#include "juicer_test_api.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaResources.h"
#include "JuicerState.h"
#include "PreparedState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "Logging.h"
#include "RustAssetBridge.h"
#include "RustScannerBridge.h"
#include "RustFrameBridge.h"
#include "RustHalationBridge.h"
#include "RustDiffusionBridge.h"
#include "RustDirBridge.h"
#include "RustEffectsFrameBridge.h"
#include "RustDirBoundaryBridge.h"
#include "juicer_cuda_owner.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_host_preparation.h"
#include "print_illuminant_fixture.h"
#include "exposure_fixture.h"
#include "spatial_optics_fixture.h"
#include "prepared_product_fixture.h"
#include "prepared_frame_fixture.h"
#include "../dir/boundary_test_hooks.h"

#include "film_recipe_abi_cpp.inc"
extern "C" const std::size_t* fj_test_film_recipe_abi_c_facts(std::size_t* count);
extern "C" FjStatus fj_test_spatial_optics_c_prepare(const FjSpatialOpticsParams* input, FjSpatialOpticsRecipe* out, FjSpatialOpticsFailure* failure, FjErrorBuffer* error);
extern "C" FjStatus fj_test_print_illuminant_c_prepare(const FjPrintIlluminantParams* input, FjFloatSpan source, FjIlluminant* out, FjPrintIlluminantFailure* failure, FjErrorBuffer* error);
extern "C" FjStatus fj_test_print_preflash_c_prepare(const FjPrintPreflashInput* input, FjPrintPreflashRaw* out, FjPrintPreflashFailure* failure, FjErrorBuffer* error);
extern "C" FjStatus fj_test_film_reference_c_prepare(const FjFilmDevelopmentOwner*, const FjFilmReferenceInput*, FjFilmReference*, FjFilmReferenceFailure*, FjErrorBuffer*);
extern "C" int fj_test_scanner_correction_c_call(unsigned, const FjScannerCorrectionParams*, const FjScannerDirectCorrectionInput*, const FjScannerPrintCorrectionInput*, const FjScannerColorCorrection*, unsigned, unsigned);

extern "C" int fj_test_gaussian_c_call(const FjGaussianKernelInput*, const uint32_t*, size_t);
extern "C" int fj_test_scanner_descriptor_c_call(const FjScannerLutDescriptorInput*, const FjScannerLutDescriptor*);
extern "C" FjStatus fj_test_scanner_lut_c_acquire(const FjScannerLutInput*, FjScannerLutOwner**, FjScannerLutFailure*, FjErrorBuffer*);
extern "C" FjStatus fj_test_scanner_lut_c_view(const FjScannerLutOwner*, FjScannerLutView*, FjScannerLutFailure*, FjErrorBuffer*);
extern "C" FjStatus fj_test_scanner_lut_c_release(FjScannerLutOwner*, FjErrorBuffer*);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const char* message);
    enum class OpticsOperationKind : std::uint8_t {
        Layout,
        RawInput,
        Fault
    };
    struct OpticsOperation {
        std::string_view id;
        OpticsOperationKind kind;
        std::size_t variant;
    };
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void require_equal(const Json& actual, const Json& expected, const char* message) {
        if (actual != expected)
            throw std::runtime_error(std::string(message) + ": first difference " + Json::diff(expected, actual).front().dump());
    }
    void status(FjStatus value, std::uint32_t category = FJ_STATUS_SUCCESS) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "closed film recipe status");
    }
    Json words(const float* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint32_t>(values[i]));
        return out;
    }
    Json read(const std::filesystem::path& path) {
        std::ifstream stream(path);
        require(bool(stream), "fixture open");
        return PrintIlluminantFixtures::apply(path, Json::parse(stream));
    }
    template <std::size_t N>
    void copy_words(const Json& value, float (&out)[N]) {
        require(value.size() == N, "fixture extent");
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(value[i].get<std::uint32_t>());
    }
    FjFilmDirParams params(const Json& v) {
        FjFilmDirParams out{};
        out.active = v.at("active").get<bool>() ? 1u : 0u;
        out.amount = std::bit_cast<float>(v.at("amount")[0].get<std::uint32_t>());
        out.inhibition_same_layer = std::bit_cast<float>(v.at("inhibition")[0].get<std::uint32_t>());
        out.inhibition_interlayer = std::bit_cast<float>(v.at("inhibition")[1].get<std::uint32_t>());
        out.diffusion_size_um = std::bit_cast<float>(v.at("diffusion")[0].get<std::uint32_t>());
        out.diffusion_tail_um = std::bit_cast<float>(v.at("diffusion")[1].get<std::uint32_t>());
        out.diffusion_tail_weight = std::bit_cast<float>(v.at("diffusion")[2].get<std::uint32_t>());
        out.gamma_use_stock = v.at("stock").get<bool>() ? 1u : 0u;
        copy_words(v.at("same"), out.gamma_same_layer_rgb);
        copy_words(v.at("red"), out.gamma_interlayer_r_to_gb);
        copy_words(v.at("green"), out.gamma_interlayer_g_to_rb);
        copy_words(v.at("blue"), out.gamma_interlayer_b_to_rg);
        copy_words(v.at("donor_k"), out.langmuir_donor_k_rgb);
        copy_words(v.at("receiver_k"), out.langmuir_receiver_k_rgb);
        return out;
    }
    Json project(const FjFilmDirView& v) {
        Json matrix = Json::array(), axes = Json::array();
        for (std::size_t i = 0; i < 3; ++i) {
            matrix.push_back(words(v.matrix_rgb + 3 * i, 3));
            axes.push_back(words(v.compensated_axes_rgb[i].data, v.compensated_axes_rgb[i].count));
        }
        const float diffusion[]{v.diffusion_size_um, v.diffusion_tail_um, v.diffusion_tail_weight};
        return {{"polarity", v.polarity}, {"active", v.mode != 0}, {"mode", v.mode}, {"matrix", matrix}, {"diffusion", words(diffusion, 3)}, {"maximum", words(v.density_max_rgb, 3)}, {"reference", words(v.density_ref_rgb, 3)}, {"donor_k", words(v.donor_k_rgb, 3)}, {"receiver_c_ref", words(v.receiver_c_ref_rgb, 3)}, {"receiver_kr", words(v.receiver_kr_rgb, 3)}, {"axes", axes}, {"axes_hash", v.compensated_axes_hash}, {"hash", v.hash}};
    }
    Json project(const FjFilmDensityBounds& v) {
        return {{"route", v.route}, {"medium", v.medium}, {"polarity", v.polarity}, {"source", v.source}, {"minimum", words(v.minimum_cmy, 3)}, {"maximum", words(v.maximum_cmy, 3)}, {"inverse", words(v.inv_span_cmy, 3)}, {"authored_minimum", words(v.authored_minimum_cmy, 3)}, {"authored_maximum", words(v.authored_maximum_cmy, 3)}, {"hash", v.hash}};
    }
    void metadata(const FjFilmRecipeFailure& failure, std::uint32_t reason = FJ_FILM_RECIPE_REASON_NONE, std::uint32_t channel = 3, std::uint64_t sample = UINT64_MAX) {
        require(failure.reason == reason && failure.channel == channel && failure.sample == sample, "exact film failure/success metadata");
    }
    struct Assets {
        FjAssets* value = nullptr;
        explicit Assets(const std::filesystem::path& path) {
#ifdef _WIN32
            const auto bytes = path.wstring();
            const FjPathView view{bytes.data(), bytes.size(), FJ_PATH_WINDOWS_WIDE};
#else
            const auto bytes = path.string();
            const FjPathView view{bytes.data(), bytes.size(), FJ_PATH_UNIX_BYTES};
#endif
            status(fj_legacy_assets_create(view, &value, nullptr));
        }
        ~Assets() {
            if (fj_legacy_assets_destroy(value, nullptr).category != 0)
                std::terminate();
        }
        FjFilmDevelopmentOwner* development(const Json& row) const {
            const auto key = row.at("key").get<std::string>();
            FjFilmProfile* profile = nullptr;
            status(fj_legacy_film_profile_acquire(value, {key.data(), key.size()}, &profile, nullptr));
            FjFilmDevelopmentOwner* owner = nullptr;
            status(fj_legacy_film_development_acquire(profile, row.at("route").get<std::uint32_t>(), std::bit_cast<float>(row.at("gamma")[0].get<std::uint32_t>()), 0, &owner, nullptr));
            status(fj_legacy_film_profile_release(profile, nullptr));
            return owner;
        }
    };
    void layouts() {
        const std::size_t cpp[]{
#define FJ_RECIPE_TYPE(t) sizeof(t), alignof(t),
#define FJ_RECIPE_FIELD(t, f) offsetof(t, f),
#define FJ_RECIPE_EXTENT(t, f, e) sizeof(t::f) / sizeof(e),
#define FJ_RECIPE_VALUE(v) v,
#include "film_recipe_abi_facts.inc"
#undef FJ_RECIPE_TYPE
#undef FJ_RECIPE_FIELD
#undef FJ_RECIPE_EXTENT
#undef FJ_RECIPE_VALUE
        };
        std::size_t cn = 0, rn = 0;
        const auto* c = fj_test_film_recipe_abi_c_facts(&cn);
        const auto* rust = fj_test_film_recipe_abi_facts(&rn);
        require(cn == std::size(cpp) && rn == cn, "exact ABI fact membership");
        for (std::size_t i = 0; i < cn; ++i)
            require(cpp[i] == c[i] && c[i] == rust[i], "C11/C++20/Rust layout agreement");
        std::printf("PASS %zu compiled ABI facts and every C/C++ field/signature\n", cn);
    }
    std::string film_case_id(std::string_view group, const Json& row) {
        return std::string(group) + "/" + row.at("key").get<std::string>() + "/route/" +
               std::to_string(row.at("route").get<unsigned>()) + "/gamma/" +
               std::to_string(row.at("gamma").at(0).get<std::uint32_t>()) + "/kind/" +
               std::to_string(row.at("kind").get<unsigned>());
    }
    void values(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        require(captured.at("cases").size() == 64, "64 independent native products");
        for (bool facade : {false, true})
            for (const auto& row : captured.at("cases")) {
                JuicerTests::CaseEvent case_event(facade ? "c11" : "cpp", film_case_id("film/default", row));
                auto* original = assets.development(row);
                FjFilmDevelopmentOwner* held = nullptr;
                status((facade ? fj_test_film_development_retain : fj_legacy_film_development_retain)(original, &held, nullptr));
                require(held == original && fj_test_film_development_live_owners() == 2, "equal address, independent holds");
                status(fj_legacy_film_development_release(original, nullptr));
                status(fj_legacy_assets_release_cached_payloads(assets.value, nullptr));
                FjFilmDevelopmentView development{};
                status(fj_legacy_film_development_view(held, &development, nullptr));
                auto controls = params(row.at("controls"));
                FjFilmDirOwner* dir = nullptr;
                FjFilmRecipeFailure failure{};
                status((facade ? fj_test_film_dir_acquire : fj_legacy_film_dir_acquire)(held, &controls, &dir, &failure, nullptr));
                metadata(failure);
                FjFilmDirView view{};
                status((facade ? fj_test_film_dir_view : fj_legacy_film_dir_view)(dir, &view, &failure, nullptr));
                metadata(failure);
                const auto copied = project(view);
                require(copied == row.at("dir"), "all frozen DIR values/axes/identities");
                for (auto span : view.compensated_axes_rgb)
                    require(span.count == (view.mode == 0 ? 0 : development.log_exposure.count), "RGB N-element axes");
                status((facade ? fj_test_film_dir_release : fj_legacy_film_dir_release)(dir, nullptr));
                require(copied == row.at("dir") && fj_test_film_dir_live_owners() == 0, "copy owns bytes after view expiry");
                const FjFilmDensityMinimum requested{{0.07f, 0.08f, 0.12f}};
                FjFilmDensityMinimum minimum{};
                status((facade ? fj_test_film_density_minimum_prepare : fj_legacy_film_density_minimum_prepare)(&requested, &minimum, &failure, nullptr));
                metadata(failure);
                require(words(minimum.cmy, 3) == row.at("grain_minimum"), "frozen minimum");
                FjFilmDensityBounds bounds{};
                status((facade ? fj_test_film_density_bounds_prepare : fj_legacy_film_density_bounds_prepare)(held, &requested, &bounds, &failure, nullptr));
                metadata(failure);
                require(project(bounds) == row.at("bounds"), "all frozen film bound values/identities");
                status(fj_legacy_film_development_release(held, nullptr));
                require(fj_test_film_development_live_owners() == 0, "held development release");
                case_event.passed();
            }
    }
    void failures(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        auto* development = assets.development(captured.at("cases")[2]);
        auto controls = params(captured.at("cases")[2].at("controls"));
        FjFilmDirOwner* owner = nullptr;
        FjFilmRecipeFailure failure{};
        JuicerTests::Cohort axis_faults("film_recipe.axis_fault", {"1", "2", "3"});
        JuicerTests::Cohort diagnostic_capacities("film_recipe.diagnostic_capacity", {"0", "1", "5", "256"});
        JuicerTests::Cohort panic_stages("film_recipe.panic_stage", {"0", "1", "2", "3", "4"});
        controls.active = 2;
        status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(!owner, "raw failure clears result");
        metadata(failure, FJ_FILM_RECIPE_RAW);
        controls.active = 0;
        controls.gamma_use_stock = 2;
        status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        metadata(failure, FJ_FILM_RECIPE_RAW);
        controls = params(captured.at("cases")[2].at("controls"));
        for (unsigned fault : {1u, 2u, 3u}) {
            auto event = axis_faults.case_event(std::to_string(fault));
            status(fj_test_film_recipe_fault(5, fault));
            status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr), fault == 3 ? FJ_STATUS_PREPARATION_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
            metadata(failure, fault == 1 ? FJ_FILM_RECIPE_SIZE : fault == 2 ? FJ_FILM_RECIPE_ALLOCATION
                                                                            : FJ_FILM_RECIPE_IDENTITY);
            require(!owner && fj_test_film_dir_live_owners() == 0, "real axis-stage fault has no owner");
            event.passed();
        }
        FjFilmDensityMinimum requested{{-1, 0, 0}}, out{{9, 9, 9}};
        const FjFilmDensityMinimum validMinimum{{0.07f, 0.08f, 0.12f}};
        FjFilmDensityBounds rejectedBounds{};
        status(fj_test_film_recipe_fault(6, 1));
        status(fj_legacy_film_density_bounds_prepare(development, &validMinimum, &rejectedBounds, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        metadata(failure, FJ_FILM_RECIPE_IDENTITY);
        require(rejectedBounds.hash == 0, "bounds identity failure clears fixed output");
        status(fj_legacy_film_density_minimum_prepare(&requested, &out, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        metadata(failure, FJ_FILM_RECIPE_MINIMUM);
        require(words(out.cmy, 3) == Json::array({0, 0, 0}), "minimum failure clear");
        FjErrorBuffer malformed{nullptr, 1, 99};
        status(fj_legacy_film_density_minimum_prepare(&requested, &out, &failure, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        metadata(failure, FJ_FILM_RECIPE_RAW);
        require(malformed.length == 0, "diagnostic admission priority");
        for (std::size_t capacity : {0u, 1u, 5u, 256u}) {
            auto event = diagnostic_capacities.case_event(std::to_string(capacity));
            std::array<char, 256> text{};
            FjErrorBuffer error{text.data(), capacity, 99};
            status(fj_legacy_film_density_minimum_prepare(&requested, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            require(error.length == (capacity == 0 ? 0 : std::min(capacity - 1, std::string("film recipe reason=2 channel=3 sample=18446744073709551615").size())) && (capacity == 0 || text[error.length] == '\0'), "bounded diagnostic/truncation");
            event.passed();
        }
        alignas(FjFilmRecipeFailure) std::array<std::byte, sizeof(FjFilmRecipeFailure) + 8> storage{};
        auto* invalid = reinterpret_cast<FjFilmRecipeFailure*>(storage.data() + 1);
        status(fj_legacy_film_density_minimum_prepare(&requested, &out, invalid, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(words(out.cmy, 3) == Json::array({0, 0, 0}), "invalid failure slot still clears result");
        status(fj_legacy_film_density_minimum_prepare(&requested, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        metadata(failure, FJ_FILM_RECIPE_RAW);
        for (unsigned stage : {0u, 1u, 2u, 3u, 4u}) {
            auto event = panic_stages.case_event(std::to_string(stage));
            status(fj_test_film_recipe_fault(stage, 1));
            if (stage == 0) {
                status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
                metadata(failure, FJ_FILM_RECIPE_PANIC);
                require(!owner, "panic clear");
            }
            if (stage == 1 || stage == 2) {
                status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr));
                if (stage == 1) {
                    FjFilmDirView view{};
                    view.hash = 99;
                    status(fj_legacy_film_dir_view(owner, &view, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
                    metadata(failure, FJ_FILM_RECIPE_PANIC);
                    require(view.hash == 0, "view panic clear");
                }
                status(fj_legacy_film_dir_release(owner, nullptr), stage == 2 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_SUCCESS);
            }
            if (stage == 3) {
                requested.cmy[0] = 0;
                FjFilmDensityBounds bounds{};
                status(fj_legacy_film_density_bounds_prepare(development, &requested, &bounds, &failure, nullptr), FJ_STATUS_INTERNAL_FAILURE);
                metadata(failure, FJ_FILM_RECIPE_PANIC);
                requested.cmy[0] = -1;
            }
            if (stage == 4) {
                std::array<char, 32> text{};
                FjErrorBuffer error{text.data(), text.size(), 0};
                status(fj_legacy_film_density_minimum_prepare(&requested, &out, &failure, &error), FJ_STATUS_INTERNAL_FAILURE);
                metadata(failure, FJ_FILM_RECIPE_PANIC);
            }
            event.passed();
        }
        FjFilmDevelopmentOwner* retained = nullptr;
        status(fj_test_film_development_fault(3, 1));
        status(fj_legacy_film_development_retain(development, &retained, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        require(!retained && fj_test_film_development_live_owners() == 1, "provisional retain unwinds without consuming original");
        status(fj_legacy_film_development_release(development, nullptr));
        status(fj_legacy_film_dir_release(nullptr, nullptr));
        status(fj_legacy_film_dir_release(reinterpret_cast<FjFilmDirOwner*>(std::uintptr_t{1}), nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "failure cleanup holds");
    }
} // namespace
#include "film_recipe_native_test.inc"
#include "visual_grain_test.inc"
#include "physical_effects_test.inc"
#include "spatial_optics_test.inc"
#include "print_illuminant_test.inc"
#include "print_preflash_test.inc"
#include "print_balance_test.inc"
#include "scanner_correction_test.inc"
#include "scanner_lut_test.inc"
#include "film_reference_test.inc"
#include "scanner_lut_descriptor_test.inc"
#include "scanner_post_effects_test.inc"
#include "frame_geometry_test.inc"
#include "halation_frame_test.inc"
#include "diffusion_frame_test.inc"
#include "dir_frame_test.inc"
#include "dir_boundary_test.inc"
#include "grain_frame_test.inc"
#include "effects_frame_test.inc"
#include "gaussian_kernel_test.inc"
int main(int argc, char** argv) try {
    const bool list_cases = JuicerTests::list_requested(argc, argv);
    if (list_cases)
        --argc;
    JuicerTests::CaseLog case_log{!list_cases, "FJ_OPTICS_CASES"};
    if (argc == 5 && (std::string_view(argv[3]) == "effects-frame" || std::string_view(argv[3]) == "effects-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "effects independent native context");
        const bool products = std::string_view(argv[3]) == "effects-frame-products";
        if (list_cases) {
            const auto add_rows = [](const Json& rows, bool includeC11) {
                for (const auto& row : rows) {
                    const auto id = row.at("input").at("id").get<std::string>();
                    JuicerTests::list_case("cpp", id);
                    EffectsFrameTest::Input input(row.at("input"));
                    FjEffectsFrameInput raw{};
                    std::string diagnostic;
                    if (includeC11 && JuicerEffects::project_input(input.frame, raw, diagnostic))
                        JuicerTests::list_case("c11", id);
                }
            };
            if (products)
                add_rows(captured.at("products"), false);
            else {
                add_rows(captured.at("cases"), true);
                add_rows(captured.at("products"), true);
                JuicerTests::list_case("cpp", "effects-frame/raw-and-faults");
            }
            return 0;
        }
        if (products) {
            const auto actual = EffectsFrameTest::product_rows(argv[1], read(directory / "inputs.json"), captured.at("products"));
            require_equal(actual, captured.at("products"), "complete genuine effects frame products");
        } else {
            EffectsFrameTest::values(captured, read(directory / "checked.json"));
            EffectsFrameTest::raw_and_faults(captured);
        }
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "grain-frame" || std::string_view(argv[3]) == "grain-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        const bool products = std::string_view(argv[3]) == "grain-frame-products";
        if (list_cases) {
            if (products) {
                const auto inputs = read(directory / "inputs.json").at("products");
                for (unsigned route = 0; route < 4; ++route)
                    for (std::size_t index = 0; index < inputs.size(); ++index)
                        JuicerTests::list_case("cpp", "grain-frame/product/" + std::to_string(route) + "/" + std::to_string(index));
            } else {
                for (const auto& row : captured.at("cases")) {
                    const auto& name = row.at("input").at("name").get_ref<const std::string&>();
                    if (name == "missing-recipe" || name == "missing-film" || name == "missing-inactive")
                        continue;
                    JuicerTests::list_case("cpp", name);
                    JuicerTests::list_case("c11", name);
                }
                JuicerTests::list_case("cpp", "grain-frame/raw-and-faults");
            }
            return 0;
        }
        if (products) {
            const auto actual = GrainFrameTest::product_rows(argv[1], read(directory / "inputs.json"), captured.at("products"));
            require(actual == captured.at("products"), "complete genuine grain frame products and native composition hashes");
        } else {
            layouts();
            GrainFrameTest::values(captured);
            GrainFrameTest::raw_and_faults(captured);
        }
        std::puts("PASS grain frame exact C11/full/native/facade/failure/default contracts");
        case_log.finish();
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "dir-boundary" || std::string_view(argv[3]) == "dir-boundary-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        require(captured.at("preset") == argv[4], "DIR boundary native context");
        const bool products = std::string_view(argv[3]) == "dir-boundary-products";
        if (list_cases) {
            for (const auto& row : captured.at("cases")) {
                const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                JuicerTests::list_case("cpp", id);
                if (!products)
                    JuicerTests::list_case("c11", id);
            }
            if (products)
                JuicerTests::list_case("cpp", "dir-boundary/product-faults");
            else
                JuicerTests::list_case("cpp", "dir-boundary/raw-and-faults");
            if (!products)
                for (unsigned operation = 0; operation < 4; ++operation)
                    JuicerTests::list_case("dir_boundary.rejected_fault_operation", std::to_string(operation));
            return 0;
        }
        layouts();
        if (!products) {
            DirBoundaryTest::raw_values({.native = captured, .checked = read(directory / "checked.json")});
            DirBoundaryTest::boundary(captured);
        } else {
            DirBoundaryTest::products(captured);
        }
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "dir-frame" || std::string_view(argv[3]) == "dir-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        require(captured.at("preset") == argv[4], "DIR frame native context");
        const bool products = std::string_view(argv[3]) == "dir-frame-products";
        if (list_cases) {
            for (const char* group : {"cases", "products"})
                for (const auto& row : captured.at(group)) {
                    const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                    JuicerTests::list_case("cpp", id);
                    JuicerTests::list_case("c11", id);
                }
            if (products) {
                for (const auto& row : captured.at("products"))
                    JuicerTests::list_case("cpp", row.at("input").at("name").get<std::string>() + "/product-check");
            } else
                JuicerTests::list_case("cpp", "dir-frame/raw-boundary");
            return 0;
        }
        layouts();
        DirFrameTest::values(captured, read(directory / "checked.json"));
        if (!products) {
            DirFrameTest::boundary(captured.at("cases"));
        } else {
            const auto actual = DirFrameTest::products(argv[1], read(directory / "inputs.json").at("products"), captured.at("products"));
            require(actual == captured.at("products"), "complete DIR route products and final native identities");
        }
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "diffusion-frame" || std::string_view(argv[3]) == "diffusion-frame-products")) {
        std::fprintf(stderr, "DiffusionFrame stage=fixture-read\n");
        std::fflush(stderr);
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        const auto edges = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("edges").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4] && edges.at("preset") == argv[4], "diffusion native contexts");
        for (const auto& row : edges.at("samples"))
            captured["samples"].push_back(row);
        const bool products = std::string_view(argv[3]) == "diffusion-frame-products";
        if (list_cases) {
            const auto list_rows = [](const Json& rows) {
                for (const auto& row : rows) {
                    const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                    JuicerTests::list_case("cpp", id);
                    JuicerTests::list_case("c11", id);
                }
            };
            list_rows(captured.at("samples"));
            list_rows(captured.at("frames"));
            for (const auto& row : captured.at("products")) {
                const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                JuicerTests::list_case("cpp", id);
                JuicerTests::list_case("c11", id);
                list_rows(row.at("samples"));
            }
            if (products) {
                for (const auto& row : captured.at("products")) {
                    const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                    JuicerTests::list_case("diffusion_frame.product_frame", id + "/product-check");
                    for (const auto& sample : row.at("samples"))
                        JuicerTests::list_case("diffusion_frame.product_stage", sample.at("input").at("name").get<std::string>() + "/product-check");
                }
            } else
                JuicerTests::list_case("cpp", "diffusion-frame/raw-boundary");
            return 0;
        }
        layouts();
        std::fprintf(stderr, "DiffusionFrame stage=values\n");
        std::fflush(stderr);
        DiffusionFrameTest::values(captured, read(directory / "checked.json"));
        std::fprintf(stderr, "DiffusionFrame stage=values-complete\n");
        std::fflush(stderr);
        if (!products)
            DiffusionFrameTest::boundary(captured, read(directory / "checked.json"));
        else
            require_equal(DiffusionFrameTest::products(argv[1], read(directory / "inputs.json").at("products"), captured.at("products")), captured.at("products"), "complete diffusion route products match unchanged native captures");
        std::fprintf(stderr, "DiffusionFrame stage=complete\n");
        std::fflush(stderr);
        case_log.finish();
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "halation-frame" || std::string_view(argv[3]) == "halation-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "halation frame native context");
        const bool products = std::string_view(argv[3]) == "halation-frame-products";
        if (list_cases) {
            for (const auto& group : {"cases", "products"})
                for (const auto& row : captured.at(group)) {
                    const auto& id = row.at("input").at("name").get_ref<const std::string&>();
                    JuicerTests::list_case("cpp", id);
                    JuicerTests::list_case("c11", id);
                }
            if (products) {
                for (const auto& row : captured.at("products"))
                    JuicerTests::list_case("cpp", row.at("input").at("name").get<std::string>() + "/product-check");
            } else
                JuicerTests::list_case("cpp", "halation-frame/raw-boundary");
            return 0;
        }
        layouts();
        HalationFrameTest::values(captured, read(directory / manifest.at("checked").get<std::string>()));
        if (!products)
            HalationFrameTest::boundary();
        else
            require(HalationFrameTest::products(argv[1], manifest.at("product_pixel_bits"), captured.at("products")) == captured.at("products"), "actual four-route Rust halation products match complete native captures");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "frame" || std::string_view(argv[3]) == "frame-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "frame fixture native context");
        const bool products = std::string_view(argv[3]) == "frame-products";
        if (list_cases) {
            for (const auto& [group, rows] : {std::pair{"domain", &captured.at("domains")}, std::pair{"geometry", &captured.at("geometry")}}) {
                std::size_t index = 0;
                for (const auto& row : *rows) {
                    const auto id = std::string("frame/") + group + "/" + (row.contains("id") ? row.at("id").get<std::string>() : std::to_string(index));
                    ++index;
                    if (products) {
                        JuicerTests::list_case("cpp", id);
                    } else {
                        JuicerTests::list_case("cpp", id + "/api-0");
                        JuicerTests::list_case("cpp", id + "/api-1");
                        JuicerTests::list_case("c11", id + "/api-2");
                    }
                }
            }
            if (!products) {
                JuicerTests::list_case("cpp", "frame/boundary");
                for (const auto* domain : {"domain", "geometry"})
                    for (const auto* api : {"legacy", "facade"}) {
                        for (unsigned kind = 0; kind < 14; ++kind)
                            JuicerTests::list_case("frame_geometry.raw_input", std::string(domain) + "/" + api + "/" + std::to_string(kind));
                        for (std::size_t capacity : {std::size_t{1}, std::size_t{8}})
                            JuicerTests::list_case("frame_geometry.diagnostic_capacity", std::string(domain) + "/" + api + "/" + std::to_string(capacity));
                        JuicerTests::list_case("frame_geometry.panic_operation", std::string(domain) + "/" + api);
                    }
            }
            case_log.finish();
            return 0;
        }
        layouts();
        FrameGeometryTest::values(captured, products);
        if (std::string_view(argv[3]) == "frame")
            FrameGeometryTest::boundary();
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "gaussian" || std::string_view(argv[3]) == "gaussian-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "Gaussian native preset context");
        const bool products = std::string_view(argv[3]) == "gaussian-products";
        if (list_cases) {
            if (products) {
                std::ifstream cohort_file(directory / "cohort.json");
                const auto cohort = Json::parse(cohort_file);
                const auto pixel_sizes = cohort.at("products").at("pixel_size_um").get<std::vector<float>>();
                for (unsigned route = 0; route < 4; ++route)
                    for (const auto pixel_size_um : pixel_sizes)
                        JuicerTests::list_case("gaussian.product_context", GaussianTest::product_context_id(route, pixel_size_um));
            }
            for (const auto& row : captured.at("cases")) {
                const auto& name = row.at("name").get_ref<const std::string&>();
                if (products) {
                    JuicerTests::list_case("cpp", name + "/values");
                    JuicerTests::list_case("cpp", name + "/product");
                    const int radius = row.at("radius");
                    const float sigma = std::bit_cast<float>(row.at("sigma_bits").get<std::uint32_t>());
                    if (radius > 0 && std::isfinite(sigma) && sigma > 0.0f)
                        JuicerTests::list_case("c11", name);
                } else {
                    JuicerTests::list_case("cpp", name + "/values");
                    const int radius = row.at("radius");
                    const float sigma = std::bit_cast<float>(row.at("sigma_bits").get<std::uint32_t>());
                    if (radius > 0 && std::isfinite(sigma) && sigma > 0.0f)
                        JuicerTests::list_case("c11", name);
                }
            }
            if (!products) {
                JuicerTests::list_case("cpp", "gaussian/boundary");
                for (unsigned probe = 0; probe < 8; ++probe)
                    JuicerTests::list_case("gaussian.domain_probe", std::to_string(probe));
                for (const auto* id : {"null-input", "null-owner", "null-failure", "malformed-error", "invalid-input"})
                    JuicerTests::list_case("gaussian.missing_peer", id);
                for (const auto* id : {"allocation-failed", "candidate-allocation-failed", "storage-unavailable", "empty-failure", "nonempty-success", "empty-success"})
                    JuicerTests::list_case("gaussian.callback_diagnostic", id);
            }
            case_log.finish();
            return 0;
        }
        layouts();
        GaussianTest::values(captured.at("cases"));
        if (products) {
            const auto actual = GaussianTest::products(argv[1], captured.at("cohort"), captured.at("cases"));
            auto expected = captured.at("cases");
            for (auto& row : expected)
                row.erase("primitives");
            require(actual == expected, "all actual Gaussian family descriptor products and every coefficient");
        } else
            GaussianTest::boundary();
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-post" || std::string_view(argv[3]) == "scanner-post-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4] && captured.at("cases").size() == 400, "post fixture preset and membership");
        if (list_cases) {
            if (std::string_view(argv[3]) == "scanner-post") {
                for (const auto& row : captured.at("cases")) {
                    JuicerTests::list_case("cpp", row.at("id").get_ref<const std::string&>());
                    JuicerTests::list_case("c11", row.at("id").get_ref<const std::string&>());
                }
                JuicerTests::list_case("cpp", "scanner-post/boundary");
                JuicerTests::list_case("cpp", "scanner-post/lifetime");
            } else {
                for (const auto& row : captured.at("cases")) {
                    JuicerTests::list_case("cpp", row.at("id").get_ref<const std::string&>());
                    JuicerTests::list_case("c11", row.at("id").get_ref<const std::string&>());
                }
                for (unsigned route = 0; route < 4; ++route)
                    JuicerTests::list_case("cpp", "scanner-post/route/" + std::to_string(route));
            }
            case_log.finish();
            return 0;
        }
        layouts();
        if (std::string_view(argv[3]) == "scanner-post") {
            ScannerPostTest::boundary(captured.at("cases"));
            ScannerPostTest::lifetime(captured.at("cases"));
        } else
            ScannerPostTest::values(captured.at("cases"));
        if (std::string_view(argv[3]) == "scanner-post-products")
            ScannerPostTest::callers(argv[1], captured.at("cases"));
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-descriptor" || std::string_view(argv[3]) == "scanner-descriptor-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "descriptor native preset context");
        const bool products = std::string_view(argv[3]) == "scanner-descriptor-products";
        if (list_cases) {
            const auto& rows = captured.at(products ? "products" : "cases");
            for (const auto& row : rows) {
                const auto id = products
                                    ? "scanner-descriptor/product-row/" + std::to_string(row.at("route").get<unsigned>())
                                    : row.at("id").get<std::string>();
                JuicerTests::list_case("cpp", id);
                const auto& input = row.at("input");
                const unsigned missing = input.at("missing").get<unsigned>();
                const bool retired = missing >= 1 && missing <= 3;
                if (!retired && row.at("native").at("success").get<bool>()) {
                    JuicerTests::list_case("c11", id + "/api-0");
                    JuicerTests::list_case("c11", id + "/api-1");
                }
            }
            if (products) {
                for (unsigned route = 0; route < 4; ++route)
                    JuicerTests::list_case("cpp", "scanner-descriptor/product/" + std::to_string(route));
                for (unsigned route : {0u, 2u})
                    JuicerTests::list_case("cpp", "scanner-descriptor/caller/" + std::to_string(route));
            } else {
                JuicerTests::list_case("cpp", "scanner-descriptor/boundary");
                JuicerTests::list_case("cpp", "scanner-descriptor/lifetime");
                JuicerTests::list_case("cpp", "scanner-descriptor/token-identity");
                for (unsigned route : {1u, 3u})
                    JuicerTests::list_case("scanner_lut_descriptor.print_token_route", std::to_string(route));
            }
            return 0;
        }
        layouts();
        ScannerDescriptorTest::values(captured.at(products ? "products" : "cases"), products);
        if (products)
            require_equal(ScannerDescriptorTest::products(argv[1], captured.at("products")), captured.at("products"), "complete actual descriptor products and enclosing identities");
        else {
            ScannerDescriptorTest::boundary(captured.at("cases"));
            ScannerDescriptorTest::lifetime(captured.at("cases"));
            ScannerDescriptorTest::admitted_token_change_preserves_identity_and_viewing_label_mismatch_is_rejected(captured.at("cases"));
        }
        if (products)
            ScannerDescriptorTest::callers(argv[1]);
        case_log.finish();
        return 0;
    }
    if (argc == 4 && std::string_view(argv[3]) == "reference-primitives") {
        if (list_cases) {
            JuicerTests::list_case("cpp", "film-reference/primitives");
            return 0;
        }
        FilmReferenceTest::check_primitives(read(argv[2]));
        std::puts("PASS exact native/Rust film reference log10 words");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "film-reference" || std::string_view(argv[3]) == "film-reference-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).get<std::string>());
        const bool products = std::string_view(argv[3]) == "film-reference-products";
        if (list_cases) {
            if (products) {
                for (const auto& row : captured.at("products"))
                    JuicerTests::list_case("cpp", "film-reference/" + std::to_string(row.at("route").get<unsigned>()) + "/" + std::to_string(row.at("method").get<unsigned>()) + "/" + std::to_string(row.at("variant").get<unsigned>()));
                for (unsigned method = 0; method < 3; ++method)
                    for (unsigned route = 0; route < 4; ++route)
                        JuicerTests::list_case("cpp", "film-reference/caller/" + std::to_string(method) + "/" + std::to_string(route));
                for (unsigned method = 0; method < 3; ++method)
                    for (unsigned route = 0; route < 4; ++route) {
                        const std::vector<unsigned> operations = method == 1 ? std::vector<unsigned>{5, 6} : std::vector<unsigned>{2};
                        for (unsigned operation : operations)
                            for (unsigned index : {1u, 2u})
                                for (unsigned fault : {1u, 2u, 3u, 4u}) {
                                    if (method == 1 && fault == 3)
                                        continue;
                                    JuicerTests::list_case("film_reference.actual_caller_fault", "method-" + std::to_string(method) + "/route-" + std::to_string(route) + "/operation-" + std::to_string(operation) + "/index-" + std::to_string(index) + "/fault-" + std::to_string(fault));
                                }
                        JuicerTests::list_case("film_reference.actual_caller_supersession", "method-" + std::to_string(method) + "/route-" + std::to_string(route));
                    }
            } else {
                JuicerTests::list_case("cpp", "film-reference/primitives");
                for (const auto& row : captured.at("products"))
                    if (row.contains("input")) {
                        const auto base = std::to_string(row.at("route").get<unsigned>()) + "/" + std::to_string(row.at("method").get<unsigned>()) + "/" + std::to_string(row.at("variant").get<unsigned>());
                        JuicerTests::list_case("cpp", "film-reference/" + base);
                        for (const auto* api : {"legacy", "facade", "c11"}) {
                            for (unsigned malformed = 0; malformed < 28; ++malformed)
                                JuicerTests::list_case("film_reference.raw_input", base + "/" + api + "/" + std::to_string(malformed));
                            const auto method = row.at("method").get<unsigned>();
                            const std::vector<unsigned> operations = method == 1 ? std::vector<unsigned>{1, 6} : std::vector<unsigned>{1, 2, 3, 4, 5};
                            for (unsigned operation : operations)
                                for (unsigned exposure = 1; exposure <= 2; ++exposure)
                                    for (unsigned fault = 1; fault <= 4; ++fault) {
                                        if ((operation >= 2 && operation <= 4 && fault > 2) || (fault == 3 && operation != 5))
                                            continue;
                                        JuicerTests::list_case("film_reference.operation_fault", base + "/" + api + "/" + std::to_string(operation) + "/" + std::to_string(exposure) + "/" + std::to_string(fault));
                                    }
                        }
                        if (row.at("method").get<unsigned>() == 1)
                            for (const auto* api : {"legacy", "facade", "c11"})
                                for (const auto& expected : captured.at("numerical_failures"))
                                    JuicerTests::list_case("film_reference.numerical_failure", base + "/" + api + "/mutation-" + std::to_string(expected.at("mutation").get<unsigned>()));
                    }
            }
            case_log.finish();
            return 0;
        }
        layouts();
        if (!products)
            FilmReferenceTest::boundary(argv[1], captured);
        else {
            require(FilmReferenceTest::products(argv[1], captured.at("products")) == captured.at("products"), "complete frozen reference products and downstream identities");
            FilmReferenceTest::callers(argv[1]);
        }
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-lut" || std::string_view(argv[3]) == "scanner-lut-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "scanner LUT fixture preset context");
        const bool products = std::string_view(argv[3]) == "scanner-lut-products";
        if (list_cases) {
            if (products) {
                for (const auto& row : captured.at("products")) {
                    const auto id = row.contains("id") ? row.at("id").get<std::string>() : "route/" + std::to_string(row.at("route").get<unsigned>());
                    JuicerTests::list_case("c11", id);
                    JuicerTests::list_case("cpp", "scanner-lut/callback/" + id);
                }
                for (unsigned route = 0; route < 4; ++route)
                    JuicerTests::list_case("cpp", "scanner-lut/product-route/" + std::to_string(route));
            } else {
                for (const auto& row : captured.at("cases")) {
                    const auto& id = row.at("id").get_ref<const std::string&>();
                    JuicerTests::list_case("c11", id);
                    JuicerTests::list_case("cpp", "scanner-lut/callback/" + id);
                }
                JuicerTests::list_case("cpp", "scanner-lut/boundary");
                for (const auto* id : {"allocation-failed", "storage-unavailable", "empty", "size-failure"})
                    JuicerTests::list_case("scanner_lut.callback_diagnostic", id);
                for (const auto* id : {"resolution-small", "resolution-overflow", "medium-enum", "baseline-bool", "eps-c-null", "weighted-z-count", "eps-m-pointer", "eps-y-overflow"})
                    JuicerTests::list_case("scanner_lut.raw_input", id);
                for (unsigned buffer = 1; buffer <= 10; ++buffer)
                    JuicerTests::list_case("scanner_lut.allocation_fault", std::to_string(buffer));
                for (unsigned operation = 0; operation < 3; ++operation)
                    JuicerTests::list_case("scanner_lut.panic_operation", std::to_string(operation));
                for (const auto* id : {"null-input", "invalid-failure-peer", "null-owner", "malformed-diagnostic", "null-view", "truncated-diagnostic", "zero-capacity-diagnostic"})
                    JuicerTests::list_case("scanner_lut.invalid_argument", id);
                for (const auto* id : {"disabled-baseline", "nonpositive-sample-count"})
                    JuicerTests::list_case("scanner_lut.ignored_span", id);
                JuicerTests::list_case("cpp", "scanner-lut/lifetime");
            }
            case_log.finish();
            return 0;
        }
        layouts();
        ScannerLutTest::values(captured.at(products ? "products" : "cases"));
        ScannerLutTest::callback_values(captured.at(products ? "products" : "cases"));
        if (!products) {
            ScannerLutTest::boundary(captured);
            ScannerLutTest::lifetime(captured);
        } else {
            ScannerLutTest::check_products(argv[1], captured);
        }
        std::puts("PASS scanner LUT complete words/counts, ABI, raw/facade, failures, reservation/panic and expiry");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-correction" || std::string_view(argv[3]) == "scanner-correction-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        const auto checked = read(directory / manifest.at("normative").get<std::string>());
        const bool products = std::string_view(argv[3]) == "scanner-correction-products";
        if (list_cases) {
            if (products) {
                for (const auto& row : captured.at("products")) {
                    const auto id = row.at("id").get<std::string>();
                    const auto first = id.find(':');
                    const auto second = id.find(':', first + 1);
                    JuicerTests::list_case("scanner_correction.recipe_tuple", id.substr(0, first) + "/" + id.substr(first + 1, second - first - 1) + "/" + id.substr(second + 1));
                }
            } else {
                for (const auto& row : captured.at("cases")) {
                    const auto id = row.at("id").get<std::string>();
                    if (checked.at("cases").at(id).at("native_only").get<bool>())
                        continue;
                    for (const std::string_view facade : {"/legacy-facade", "/test-facade"}) {
                        JuicerTests::list_case("cpp", id + std::string(facade));
                        JuicerTests::list_case("c11", id + std::string(facade) + "/c-call");
                    }
                }
                for (const auto& row : captured.at("products"))
                    for (const std::string_view facade : {"/legacy-facade", "/test-facade"})
                        JuicerTests::list_case("cpp", "scanner-correction/product/" + row.at("id").get<std::string>() + std::string(facade));
                JuicerTests::list_case("cpp", "scanner-correction/boundary");
                for (unsigned scenario = 0; scenario < 5; ++scenario)
                    JuicerTests::list_case("scanner_correction.demand_query_scenario", std::to_string(scenario));
            }
            case_log.finish();
            return 0;
        }
        if (products)
            ScannerCorrectionTest::check_products(argv[1], captured);
        else {
            layouts();
            ScannerCorrectionTest::values(captured, checked);
            ScannerCorrectionTest::boundary(captured);
        }
        std::puts("PASS scanner correction exact values, checked demand, ABI, clearing and complete held products");
        case_log.finish();
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "scanner-correction-primitives") {
        if (list_cases) {
            JuicerTests::list_case("cpp", "scanner-correction/primitives");
            return 0;
        }
        JuicerTests::CaseEvent primitive_event("cpp", "scanner-correction/primitives");
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"primitives", ScannerCorrectionTest::primitives()}});
        primitive_event.passed();
        case_log.finish();
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-balance-cache") {
        if (list_cases) {
            JuicerTests::list_case("cpp", "print-balance/cache");
            return 0;
        }
        PrintBalanceTest::cache(argv[1]);
        std::puts("PASS balance cache keys, resolved bindings, residency/accounting and no per-frame derivation");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "print-balance" || std::string_view(argv[3]) == "print-balance-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        const bool products = std::string_view(argv[3]) == "print-balance-products";
        if (list_cases) {
            if (products) {
                for (unsigned route = 0; route < 4; ++route)
                    for (int mode = 0; mode < 3; ++mode)
                        for (unsigned variant = 0; variant < 10; ++variant)
                            JuicerTests::list_case("print_balance.recipe_tuple", "print-balance/route/" + std::to_string(route) + "/mode/" + std::to_string(mode) + "/variant/" + std::to_string(variant));
                JuicerTests::list_case("cpp", "print-balance/actual-caller");
                for (int normalize : {0, 1})
                    for (int compensate : {0, 1})
                        JuicerTests::list_case("print_balance.normalize_compensate", "normalize-" + std::to_string(normalize) + "/compensate-" + std::to_string(compensate));
            } else {
                for (const auto& row : captured.at("cases")) {
                    const auto& id = row.at("id").get_ref<const std::string&>();
                    JuicerTests::list_case("cpp", id);
                    JuicerTests::list_case("c11", id);
                }
                for (unsigned caller = 0; caller < 2; ++caller)
                    for (unsigned malformed = 0; malformed < 3; ++malformed)
                        JuicerTests::list_case("print_balance.malformed_error", "caller-" + std::to_string(caller) + "/error-" + std::to_string(malformed));
                JuicerTests::list_case("cpp", "print-balance/transport");
            }
            case_log.finish();
            return 0;
        }
        if (std::string_view(argv[3]) == "print-balance-products")
            PrintBalanceTest::products(argv[1], captured);
        else {
            layouts();
            PrintBalanceTest::values(captured, argv[2]);
            PrintBalanceTest::transport(captured);
        }
        std::puts("PASS balance complete values, closed boundary, integrity and actual caller");
        case_log.finish();
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "balance-primitives") {
        if (list_cases) {
            JuicerTests::list_case("cpp", "print-balance/primitives");
            return 0;
        }
        JuicerTests::CaseEvent primitive_event("cpp", "print-balance/primitives");
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"cases", PrintBalanceTest::primitives()}});
        primitive_event.passed();
        case_log.finish();
        return 0;
    }


    if (argc == 5 && (std::string_view(argv[3]) == "print-preflash" || std::string_view(argv[3]) == "print-preflash-products")) {
        const auto manifest = read(argv[2]);
        const auto path = std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>();
        const auto captured = read(path);
        const bool products = std::string_view(argv[3]) == "print-preflash-products";
        if (list_cases) {
            if (products) {
                for (unsigned route = 0; route < 4; ++route)
                    for (int mode = 0; mode < 3; ++mode)
                        for (unsigned variant = 0; variant < 6; ++variant)
                            JuicerTests::list_case("print_preflash.recipe_tuple", "print-preflash/route/" + std::to_string(route) + "/mode/" + std::to_string(mode) + "/variant/" + std::to_string(variant));
                JuicerTests::list_case("cpp", "print-preflash/actual-caller");
            } else {
                for (const auto& row : captured.at("cases")) {
                    const auto& id = row.at("id").get_ref<const std::string&>();
                    JuicerTests::list_case("c11", id);
                    JuicerTests::list_case("cpp", id);
                }
                for (unsigned caller = 0; caller < 2; ++caller)
                    for (unsigned address = 0; address < 2; ++address)
                        JuicerTests::list_case("print_preflash.raw_address", "caller-" + std::to_string(caller) + "/address-" + std::to_string(address));
                JuicerTests::list_case("cpp", "print-preflash/transport");
            }
            case_log.finish();
            return 0;
        }
        if (products)
            PrintPreflashTest::products(argv[1], captured);
        else {
            layouts();
            PrintPreflashTest::values(captured, argv[2]);
            PrintPreflashTest::transport(captured);
        }
        std::puts("PASS preflash exact adopted values, closed transport and selected caller contracts");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-preflash-cache") {
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        if (list_cases) {
            JuicerTests::list_case("cpp", "print-preflash/cache");
            return 0;
        }
        PrintPreflashTest::cache(argv[1], captured);
        std::puts("PASS actual request/input preflash cache, fault nonconsumption, failure commit and recovery");
        case_log.finish();
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "preflash-primitives") {
        if (list_cases) {
            JuicerTests::list_case("cpp", "print-preflash/primitives");
            for (const auto& id : PrintPreflashTest::primitive_profile_ids())
                JuicerTests::list_case("print_preflash.primitive_profile", id);
            for (const auto& id : PrintPreflashTest::primitive_density_ids())
                JuicerTests::list_case("print_preflash.primitive_density_grid", id);
            return 0;
        }
        JuicerTests::CaseEvent primitive_event("cpp", "print-preflash/primitives");
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"cases", PrintPreflashTest::primitives(argv[1])}});
        primitive_event.passed();
        std::puts("PASS preflash primitive observations recorded; independent analysis required");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-illuminant-products") {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto products = read(directory / selected.at("products").get<std::string>());
        const auto literals = read(directory / selected.at("literal").get<std::string>());
        if (list_cases) {
            for (const auto& row : products.at("products"))
                JuicerTests::list_case("print_illuminant.recipe_tuple", "print-illuminant/route/" + std::to_string(row.at("route").get<unsigned>()) + "/mode/" + std::to_string(row.at("mode").get<int>()) + "/variant/" + std::to_string(row.at("variant").get<unsigned>()));
            JuicerTests::list_case("cpp", "print-illuminant/publication");
            case_log.finish();
            return 0;
        }
        PrintIlluminantTest::products(products, argv[1], literals);
        std::puts("PASS 72 genuine print illuminant products, direct omission and downstream identities");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "print-illuminant" || std::string_view(argv[3]) == "print-illuminant-transport")) {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / selected.at("literal").get<std::string>());
        if (list_cases) {
            if (std::string_view(argv[3]) == "print-illuminant")
                for (const auto& row : captured.at("cases")) {
                    const auto& id = row.at("id").get_ref<const std::string&>();
                    JuicerTests::list_case("c11", id);
                    JuicerTests::list_case("cpp", id);
                }
            JuicerTests::list_case("cpp", "print-illuminant/transport");
            case_log.finish();
            return 0;
        }
        layouts();
        PrintIlluminantTest::transport(captured);
        if (std::string_view(argv[3]) == "print-illuminant")
            PrintIlluminantTest::values(captured);
        std::puts("PASS print illuminant selected ABI/transport/value cohort");
        case_log.finish();
        return 0;
    }

    if (argc == 5 && std::string_view(argv[3]) == "optics-products") {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const OpticsTest::ProductFixtures fixtures{
            .products = read(directory / selected.at("products").get<std::string>()),
            .literals = read(directory / selected.at("literal").get<std::string>()),
            .ofx = read(directory / selected.at("ofx").get<std::string>())};
        if (list_cases) {
            for (const auto& row : fixtures.products.at("products"))
                JuicerTests::list_case("cpp", "optics/product/" + std::to_string(row.at("route").get<unsigned>()) + "/" + std::to_string(row.at("variant").get<unsigned>()));
            for (unsigned route = 0; route < 4; ++route)
                JuicerTests::list_case("cpp", "optics/publication/" + std::to_string(route));
            return 0;
        }
        OpticsTest::products(argv[1], fixtures);
        case_log.finish();
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "optics") {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / selected.at("literal").get<std::string>());
        if (list_cases) {
            OpticsTest::list_cases(captured);
            case_log.finish();
            return 0;
        }
        OpticsTest::run_operation(OpticsTest::layout_operation, "cpp", captured);
        OpticsTest::values(captured);
        OpticsTest::run_all_nonrow_cases(captured);
        case_log.finish();
        std::puts("PASS spatial optics exact values, compiled ABI, facade and failures");
        case_log.finish();
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "effects" || std::string_view(argv[3]) == "effects-products")) {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const bool products = std::string_view(argv[3]) == "effects-products";
        const auto captured = read(directory / selected.at(products ? "products" : "literal").get<std::string>());
        if (list_cases) {
            if (products) {
                for (const auto& row : captured.at("products"))
                    JuicerTests::list_case("cpp", "effects/product/" + row.at("key").get<std::string>() + "/" + std::to_string(row.at("route").get<unsigned>()) + "/" + row.at("id").get<std::string>());
                for (unsigned route = 0; route < 4; ++route)
                    JuicerTests::list_case("cpp", "effects/publication/" + std::to_string(route));
            } else {
                for (const auto& row : captured.at("cases")) {
                    JuicerTests::list_case("cpp", "effects/" + row.at("id").get<std::string>());
                    JuicerTests::list_case("c11", "effects/" + row.at("id").get<std::string>());
                }
                JuicerTests::list_case("cpp", "effects/failures");
            }
            return 0;
        }
        if (products)
            effects_products(argv[1], captured);
        else {
            layouts();
            effects_values(captured);
            effects_failures(captured.at("cases")[0].at("recipe"));
        }
        std::puts("PASS physical effects exact value/ABI/facade/failure/projection cohort");
        case_log.finish();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "grain" || std::string_view(argv[3]) == "grain-products")) {
        const auto manifest = read(argv[2]);
        const std::filesystem::path directory = std::filesystem::path(argv[2]).parent_path();
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto captured = read(directory / selected.at("values").get<std::string>());
        const bool products = std::string_view(argv[3]) == "grain-products";
        const auto synthetic = products ? Json::object() : read(directory / selected.at("synthetic").get<std::string>());
        if (list_cases) {
            if (products) {
                std::size_t index = 0;
                for (const auto& row : captured.at("products"))
                    JuicerTests::list_case("cpp", "grain/product/" + std::to_string(index++) + "/" + row.at("key").get<std::string>() + "/" + std::to_string(row.at("route").get<unsigned>()));
                for (unsigned route = 0; route < 4; ++route)
                    JuicerTests::list_case("cpp", "grain/publication/" + std::to_string(route));
                JuicerTests::list_case("cpp", "grain/product-owner-drain");
            } else {
                for (bool facade : {false, true}) {
                    const char* caller = facade ? "c11" : "cpp";
                    std::size_t product_index = 0;
                    for (const auto& row : captured.at("products"))
                        JuicerTests::list_case(caller, "grain/product-input/" + std::to_string(product_index++) + "/" + row.at("key").get<std::string>() + "/" + std::to_string(row.at("route").get<unsigned>()));
                    std::size_t control_index = 0;
                    for (const auto& row : captured.at("controls"))
                        JuicerTests::list_case(caller, "grain/control/" + std::to_string(control_index++) + "/" + row.at("id").get<std::string>());
                    std::size_t synthetic_index = 0;
                    for (const auto& row : synthetic.at("products"))
                        JuicerTests::list_case(caller, "grain/synthetic/" + std::to_string(synthetic_index++) + "/" + row.at("source").get<std::string>());
                    JuicerTests::list_case(caller, "grain/failures");
                }
                JuicerTests::list_case("cpp", "grain/owner-drain");
            }
            return 0;
        }
        if (products)
            grain_products(argv[1], captured);
        else {
            layouts();
            grain_values(argv[1], captured, synthetic, directory);
            grain_failures(argv[1], captured);
        }
        std::puts("PASS visual grain complete value/ABI/facade/failure/lifetime cohort");
        case_log.finish();
        return 0;
    }
    const bool product_mode = argc == 4 && std::string_view(argv[3]) == "products";
    require(argc == 3 || product_mode, "resource root and capture required");
    const auto captured = read(argv[2]);
    if (list_cases) {
        for (const auto& row : captured.at("cases")) {
            const auto id = film_case_id(product_mode ? "film/products" : "film/default", row);
            if (product_mode) {
                JuicerTests::list_case("cpp", id);
            } else {
                JuicerTests::list_case("cpp", id);
                JuicerTests::list_case("c11", id);
            }
        }
        if (!product_mode) {
            const auto f1 = read(std::filesystem::path(argv[2]).parent_path() /
                                 (std::filesystem::path(argv[2]).stem().string() + "-f1.json"));
            for (const auto& row : f1)
                JuicerTests::list_case("cpp", "film/default/raw/" + row.at("case").get<std::string>() + "/" + std::to_string(row.at("kind").get<unsigned>()));
            for (const auto* id : {"1", "2", "3"})
                JuicerTests::list_case("film_recipe.axis_fault", id);
            for (const auto* id : {"0", "1", "5", "256"})
                JuicerTests::list_case("film_recipe.diagnostic_capacity", id);
            for (const auto* id : {"0", "1", "2", "3", "4"})
                JuicerTests::list_case("film_recipe.panic_stage", id);
        } else {
            for (unsigned route = 0; route < 4; ++route)
                JuicerTests::list_case("film_dir.admission_route", std::to_string(route));
        }
        case_log.finish();
        return 0;
    }
    if (product_mode)
        products(argv[1], captured);
    else {
        layouts();
        values(argv[1], captured);
        failures(argv[1], captured);
        native_cases(argv[2], captured);
    }
    std::puts("PASS native characterizations, shared holds, raw failure/diagnostic and panic cleanup");
    case_log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("unknown film recipe fixture failure\n", stderr);
    return 1;
}
