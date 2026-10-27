#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <thread>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstring>
#include <vector>
#include <utility>

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

#include "juicer_test_api.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaResources.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "Logging.h"
#include "RustAssetBridge.h"
#include "RustScannerBridge.h"
#include "RustGaussianBridge.h"
#include "RustFrameBridge.h"
#include "RustHalationBridge.h"
#include "RustDiffusionBridge.h"
#include "RustDirBridge.h"
#include "RustGrainFrameBridge.h"
#include "RustEffectsFrameBridge.h"
#include "RustDirBoundaryBridge.h"
#include "juicer_cuda_owner.h"
#include "spectral_allocation_probe.h"
#include "print_illuminant_fixture.h"
#include "exposure_fixture.h"
#include "spatial_optics_fixture.h"
#include "../dir/boundary_test_hooks.h"

#include "film_recipe_abi_cpp.inc"
static_assert(!std::is_copy_constructible_v<JuicerAssets::DevelopmentHold>);
static_assert(!std::is_copy_constructible_v<JuicerAssets::FilmDevelopment>);
static_assert(!std::is_copy_constructible_v<Spektrafilm::DirectRecipeBuildResult>);
static_assert(!std::is_copy_constructible_v<Spektrafilm::PrintRecipeBuildResult>);
static_assert(std::is_nothrow_move_constructible_v<JuicerAssets::DevelopmentHold>);
extern "C" const std::size_t* fj_test_film_recipe_abi_c_facts(std::size_t* count);
extern "C" FjStatus fj_test_film_reference_c_prepare(const FjFilmDevelopmentOwner*, const FjFilmReferenceInput*, FjFilmReference*, FjFilmReferenceFailure*, FjErrorBuffer*);
extern "C" int fj_test_scanner_correction_c_call(unsigned, const FjScannerCorrectionParams*, const FjScannerDirectCorrectionInput*, const FjScannerPrintCorrectionInput*, const FjScannerColorCorrection*, unsigned, unsigned);

extern "C" int fj_test_gaussian_c_call(const FjGaussianKernelInput*, const uint32_t*, size_t);
extern "C" int fj_test_scanner_descriptor_c_call(const FjScannerLutDescriptorInput*, const FjScannerLutDescriptor*);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
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
    void values(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        require(captured.at("cases").size() == 64, "64 independent native products");
        for (bool facade : {false, true})
            for (const auto& row : captured.at("cases")) {
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
            }
    }
    void failures(const std::filesystem::path& resources, const Json& captured) {
        Assets assets(resources);
        auto* development = assets.development(captured.at("cases")[2]);
        auto controls = params(captured.at("cases")[2].at("controls"));
        FjFilmDirOwner* owner = nullptr;
        FjFilmRecipeFailure failure{};
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
            status(fj_test_film_recipe_fault(5, fault));
            status(fj_legacy_film_dir_acquire(development, &controls, &owner, &failure, nullptr), fault == 3 ? FJ_STATUS_PREPARATION_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
            metadata(failure, fault == 1 ? FJ_FILM_RECIPE_SIZE : fault == 2 ? FJ_FILM_RECIPE_ALLOCATION
                                                                            : FJ_FILM_RECIPE_IDENTITY);
            require(!owner && fj_test_film_dir_live_owners() == 0, "real axis-stage fault has no owner");
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
            std::array<char, 256> text{};
            FjErrorBuffer error{text.data(), capacity, 99};
            status(fj_legacy_film_density_minimum_prepare(&requested, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            require(error.length == (capacity == 0 ? 0 : std::min(capacity - 1, std::string("film recipe reason=2 channel=3 sample=18446744073709551615").size())) && (capacity == 0 || text[error.length] == '\0'), "bounded diagnostic/truncation");
        }
        alignas(FjFilmRecipeFailure) std::array<std::byte, sizeof(FjFilmRecipeFailure) + 8> storage{};
        auto* invalid = reinterpret_cast<FjFilmRecipeFailure*>(storage.data() + 1);
        status(fj_legacy_film_density_minimum_prepare(&requested, &out, invalid, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        require(words(out.cmy, 3) == Json::array({0, 0, 0}), "invalid failure slot still clears result");
        status(fj_legacy_film_density_minimum_prepare(&requested, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        metadata(failure, FJ_FILM_RECIPE_RAW);
        for (unsigned stage : {0u, 1u, 2u, 3u, 4u}) {
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
#include "gaussian_kernel_test.inc"
#include "frame_geometry_test.inc"
#include "halation_frame_test.inc"
#include "diffusion_frame_test.inc"
#include "dir_frame_test.inc"
#include "dir_boundary_test.inc"
#include "grain_frame_test.inc"
#include "effects_frame_test.inc"
int main(int argc, char** argv) try {
    if (argc == 5 && (std::string_view(argv[3]) == "effects-frame" || std::string_view(argv[3]) == "effects-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "effects independent native context");
        if (std::string_view(argv[3]) == "effects-frame-products")
            require(EffectsFrameTest::product_rows(argv[1], read(directory / "inputs.json")) == captured.at("products"), "complete genuine effects frame products");
        else {
            EffectsFrameTest::values(captured, read(directory / "checked.json"));
            EffectsFrameTest::raw_and_faults(captured);
        }
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "grain-frame" || std::string_view(argv[3]) == "grain-frame-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        if (std::string_view(argv[3]) == "grain-frame-products")
            require(GrainFrameTest::product_rows(argv[1], read(directory / "inputs.json")) == captured.at("products"), "complete genuine grain frame products and native composition hashes");
        else {
            layouts();
            GrainFrameTest::values(captured);
            GrainFrameTest::raw_and_faults(captured);
        }
        std::puts("PASS grain frame exact C11/full/native/facade/failure/default contracts");
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "dir-boundary" || std::string_view(argv[3]) == "dir-boundary-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        require(captured.at("preset") == argv[4], "DIR boundary native context");
        if (std::string_view(argv[3]) == "dir-boundary") {
            DirBoundaryTest::raw_values({.native = captured, .checked = read(directory / "checked.json")});
            DirBoundaryTest::boundary(captured);
        } else {
            DirBoundaryTest::products(captured);
        }
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "dir-frame" || std::string_view(argv[3]) == "dir-frame-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        require(captured.at("preset") == argv[4], "DIR frame native context");
        DirFrameTest::values(captured, read(directory / "checked.json"));
        if (std::string_view(argv[3]) == "dir-frame") {
            DirFrameTest::boundary(captured.at("cases"));
        } else {
            require(DirFrameTest::products(argv[1], read(directory / "inputs.json").at("products")) == captured.at("products"), "complete DIR route products and final native identities");
        }
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "diffusion-frame" || std::string_view(argv[3]) == "diffusion-frame-products")) {
        std::fprintf(stderr, "DiffusionFrame stage=fixture-read\n");
        std::fflush(stderr);
        layouts();
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        const auto edges = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("edges").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4] && edges.at("preset") == argv[4], "diffusion native contexts");
        for (const auto& row : edges.at("samples"))
            captured["samples"].push_back(row);
        std::fprintf(stderr, "DiffusionFrame stage=values\n");
        std::fflush(stderr);
        DiffusionFrameTest::values(captured, read(directory / "checked.json"));
        std::fprintf(stderr, "DiffusionFrame stage=values-complete\n");
        std::fflush(stderr);
        if (std::string_view(argv[3]) == "diffusion-frame")
            DiffusionFrameTest::boundary(captured, read(directory / "checked.json"));
        else
            require(DiffusionFrameTest::products(argv[1], read(directory / "inputs.json").at("products")) == captured.at("products"), "complete diffusion route products match unchanged native captures");
        std::fprintf(stderr, "DiffusionFrame stage=complete\n");
        std::fflush(stderr);
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "halation-frame" || std::string_view(argv[3]) == "halation-frame-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "halation frame native context");
        HalationFrameTest::values(captured, read(directory / manifest.at("checked").get<std::string>()));
        if (std::string_view(argv[3]) == "halation-frame")
            HalationFrameTest::boundary();
        else
            require(HalationFrameTest::products(argv[1], manifest.at("product_pixel_bits")) == captured.at("products"), "actual four-route Rust halation products match complete native captures");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "frame" || std::string_view(argv[3]) == "frame-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "frame fixture native context");
        FrameGeometryTest::values(captured, std::string_view(argv[3]) == "frame-products");
        if (std::string_view(argv[3]) == "frame")
            FrameGeometryTest::boundary();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "gaussian" || std::string_view(argv[3]) == "gaussian-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = load_shared_fixture(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "Gaussian native preset context");
        GaussianTest::values(captured.at("cases"));
        if (std::string_view(argv[3]) == "gaussian-products") {
            const auto actual = GaussianTest::products(argv[1], captured.at("cohort"));
            auto expected = captured.at("cases");
            for (auto& row : expected)
                row.erase("primitives");
            require(actual == expected, "all actual Gaussian family descriptor products and every coefficient");
        } else
            GaussianTest::boundary();
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-post" || std::string_view(argv[3]) == "scanner-post-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4] && captured.at("cases").size() == 400, "post fixture preset and membership");
        if (std::string_view(argv[3]) == "scanner-post") {
            ScannerPostTest::boundary(captured.at("cases"));
            ScannerPostTest::adapters(captured.at("cases"));
        } else
            ScannerPostTest::native_values(captured.at("cases"));
        if (std::string_view(argv[3]) == "scanner-post-products")
            ScannerPostTest::callers(argv[1], captured.at("cases"));
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-descriptor" || std::string_view(argv[3]) == "scanner-descriptor-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "descriptor native preset context");
        const bool products = std::string_view(argv[3]) == "scanner-descriptor-products";
        ScannerDescriptorTest::values(captured.at(products ? "products" : "cases"));
        if (products)
            require(ScannerDescriptorTest::products(argv[1]) == captured.at("products"), "complete actual descriptor products and enclosing identities");
        else {
            ScannerDescriptorTest::boundary(captured.at("cases"));
            ScannerDescriptorTest::adapters(captured.at("cases"));
            ScannerDescriptorTest::admitted_token_change_preserves_identity_and_viewing_label_mismatch_is_rejected(captured.at("cases"));
        }
        if (products)
            ScannerDescriptorTest::callers(argv[1]);
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "capture-scanner-descriptors") {
        PrintPreflashTest::save(argv[5], {{"schema", 1}, {"preset", argv[4]}, {"cases", ScannerDescriptorTest::cases()}, {"products", ScannerDescriptorTest::products(argv[1])}});
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "reference-capture") {
        PrintPreflashTest::save(argv[5], {{"schema", 1}, {"preset", argv[4]}, {"products", FilmReferenceTest::products(argv[1])}, {"numerical_failures", FilmReferenceTest::numerical_failures(argv[1])}});
        std::puts("PASS native film reference observations recorded; independent analysis required");
        return 0;
    }
    if (argc == 4 && std::string_view(argv[3]) == "reference-primitives") {
        FilmReferenceTest::check_primitives(read(argv[2]));
        std::puts("PASS exact native/Rust film reference log10 words");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "film-reference" || std::string_view(argv[3]) == "film-reference-products")) {
        layouts();
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).get<std::string>());
        if (std::string_view(argv[3]) == "film-reference")
            FilmReferenceTest::boundary(argv[1], captured);
        else {
            require(FilmReferenceTest::products(argv[1]) == captured.at("products") && FilmReferenceTest::numerical_failures(argv[1]) == captured.at("numerical_failures"), "complete frozen reference products, failures and downstream identities");
            FilmReferenceTest::callers(argv[1]);
        }
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-lut" || std::string_view(argv[3]) == "scanner-lut-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = load_shared_fixture(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>(), "manifest.json");
        require(captured.at("preset") == argv[4], "scanner LUT fixture preset context");
        layouts();
        const bool products = std::string_view(argv[3]) == "scanner-lut-products";
        ScannerLutTest::values(captured.at(products ? "products" : "cases"));
        ScannerLutTest::native_values(captured.at(products ? "products" : "cases"));
        if (!products) {
            ScannerLutTest::boundary(captured);
            ScannerLutTest::native_failures(captured);
        } else {
            ScannerLutTest::check_products(argv[1], captured);
        }
        std::puts("PASS scanner LUT complete words/counts, ABI, raw/facade, failures, reservation/panic and expiry");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "scanner-correction" || std::string_view(argv[3]) == "scanner-correction-products")) {
        const auto manifest = read(argv[2]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        const auto checked = read(directory / manifest.at("normative").get<std::string>());
        const auto tracePath = directory / manifest.at("presets").at(argv[4]).at("traces").get<std::string>();
        if (std::string_view(argv[3]) == "scanner-correction-products")
            ScannerCorrectionTest::check_products(argv[1], captured, read(tracePath));
        else {
            layouts();
            ScannerCorrectionTest::values(captured, checked);
            ScannerCorrectionTest::boundary(captured);
            ScannerCorrectionTest::native(captured, checked);
            ScannerCorrectionTest::callers(captured, checked, tracePath);
        }
        std::puts("PASS scanner correction exact values, checked demand, ABI, clearing and actual native binding");
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "scanner-correction-primitives") {
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"primitives", ScannerCorrectionTest::primitives()}});
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-balance-cache") {
        PrintBalanceTest::cache(argv[1]);
        std::puts("PASS balance cache keys, resolved bindings, residency/accounting and no per-frame derivation");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "print-balance" || std::string_view(argv[3]) == "print-balance-products")) {
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        if (std::string_view(argv[3]) == "print-balance-products")
            PrintBalanceTest::products(argv[1], captured);
        else {
            layouts();
            PrintBalanceTest::values(captured, argv[2]);
            PrintBalanceTest::transport(captured);
        }
        std::puts("PASS balance complete values, closed boundary, integrity and actual caller");
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "balance-primitives") {
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"cases", PrintBalanceTest::primitives()}});
        return 0;
    }


    if (argc == 5 && (std::string_view(argv[3]) == "print-preflash" || std::string_view(argv[3]) == "print-preflash-products")) {
        const auto manifest = read(argv[2]);
        const auto path = std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>();
        const auto captured = read(path);
        if (std::string_view(argv[3]) == "print-preflash-products")
            PrintPreflashTest::products(argv[1], captured);
        else {
            layouts();
            PrintPreflashTest::values(captured, argv[2]);
            PrintPreflashTest::transport(captured);
            PrintPreflashTest::native(captured);
        }
        std::puts("PASS preflash exact adopted values, closed transport, native projection and selected caller contracts");
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-preflash-cache") {
        const auto manifest = read(argv[2]);
        const auto captured = read(std::filesystem::path(argv[2]).parent_path() / manifest.at("presets").at(argv[4]).at("native").get<std::string>());
        PrintPreflashTest::cache(argv[1], captured);
        std::puts("PASS actual request/input preflash cache, fault nonconsumption, failure commit and recovery");
        return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "preflash-primitives") {
        PrintPreflashTest::save(argv[5], {{"preset", argv[4]}, {"cases", PrintPreflashTest::primitives(argv[1])}});
        std::puts("PASS preflash primitive observations recorded; independent analysis required");
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "print-illuminant-products") {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        PrintIlluminantTest::products(argv[1], read(directory / selected.at("products").get<std::string>()));
        std::puts("PASS 72 genuine print illuminant products, direct omission and downstream identities");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "print-illuminant" || std::string_view(argv[3]) == "print-illuminant-transport")) {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / selected.at("literal").get<std::string>());
        layouts();
        PrintIlluminantTest::transport(captured);
        PrintIlluminantTest::native(captured);
        if (std::string_view(argv[3]) == "print-illuminant")
            PrintIlluminantTest::values(captured);
        std::puts("PASS print illuminant selected ABI/transport/projection/value cohort");
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
        OpticsTest::products(argv[1], fixtures);
        return 0;
    }
    if (argc == 5 && std::string_view(argv[3]) == "optics") {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        const auto captured = read(directory / selected.at("literal").get<std::string>());
        layouts();
        OpticsTest::values(captured);
        OpticsTest::raw_and_faults(captured);
        OpticsTest::native(captured);
        std::puts("PASS spatial optics exact values, compiled ABI, facade, failures and native projection");
        return 0;
    }

    if (argc == 5 && (std::string_view(argv[3]) == "effects" || std::string_view(argv[3]) == "effects-products")) {
        const auto manifest = read(argv[2]);
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto directory = std::filesystem::path(argv[2]).parent_path();
        if (std::string_view(argv[3]) == "effects-products")
            effects_products(argv[1], read(directory / selected.at("products").get<std::string>()));
        else {
            const auto captured = read(directory / selected.at("literal").get<std::string>());
            layouts();
            effects_values(captured);
            effects_failures(captured.at("cases")[0].at("recipe"));
            effects_native(captured);
        }
        std::puts("PASS physical effects exact value/ABI/facade/failure/projection cohort");
        return 0;
    }
    if (argc == 5 && (std::string_view(argv[3]) == "grain" || std::string_view(argv[3]) == "grain-products")) {
        const auto manifest = read(argv[2]);
        const std::filesystem::path directory = std::filesystem::path(argv[2]).parent_path();
        const auto selected = manifest.at("presets").at(argv[4]);
        const auto captured = read(directory / selected.at("values").get<std::string>());
        if (std::string_view(argv[3]) == "grain-products")
            grain_products(argv[1], captured);
        else {
            layouts();
            grain_values(argv[1], captured, read(directory / selected.at("synthetic").get<std::string>()), directory);
            grain_failures(argv[1], captured);
            grain_native_bridge(argv[1], captured, read(directory / selected.at("synthetic").get<std::string>()), directory);
        }
        std::puts("PASS visual grain complete value/ABI/facade/failure/lifetime cohort");
        return 0;
    }
    require(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "products"), "resource root and capture required");
    const auto captured = read(argv[2]);
    if (argc == 4)
        products(argv[1], captured, argv[2]);
    else {
        layouts();
        values(argv[1], captured);
        failures(argv[1], captured);
        native_cases(argv[2], captured);
    }
    std::puts("PASS native characterizations, shared holds, raw failure/diagnostic and panic cleanup");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("unknown film recipe fixture failure\n", stderr);
    return 1;
}
