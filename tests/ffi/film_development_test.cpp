#include <array>
#include <barrier>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <new>
#include <string>
#include <string_view>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "RustAssetBridge.h"
#include "juicer_test_api.h"
#include "prepared_product_fixture.h"
#include "print_illuminant_fixture.h"
#include "juicer_cuda_owner.h"

extern "C" const std::size_t* fj_test_film_development_abi_c_facts(std::size_t* count);
static_assert(!std::is_same_v<FjFilmDevelopment, FjFilmDevelopmentOwner>);
static_assert(std::is_same_v<decltype(&fj_legacy_film_development_acquire), FjStatus (*)(const FjFilmProfile*, std::uint32_t, float, std::uint32_t, FjFilmDevelopmentOwner**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_film_development_view), FjStatus (*)(const FjFilmDevelopmentOwner*, FjFilmDevelopmentView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_film_development_release), FjStatus (*)(FjFilmDevelopmentOwner*, FjErrorBuffer*)>);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus actual, std::uint32_t expected = FJ_STATUS_SUCCESS) {
        require(actual.category == expected && actual.api == FJ_API_NONE && actual.native_code == 0, "typed development status");
    }
    Json words(const float* values, std::size_t count) {
        Json result = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            result.push_back(std::bit_cast<std::uint32_t>(values[i]));
        return result;
    }
    void compare(FjFilmDevelopmentView view, const Json& row, const Json& table) {
        const auto& expected = row.at("development");
        require(view.route == row.at("route") && view.polarity == table.at("polarity"), "route/polarity mapping");
        require(bool(view.layers_required) == expected.at("layers_required"), "layer policy");
        require(words(view.gamma_rgb, 3) == expected.at("gamma") && words(view.minimum_cmy, 3) == table.at("minimum") && words(view.maximum_cmy, 3) == table.at("maximum"), "gamma/extrema words");
        require(view.normalized_curves_hash == table.at("normalized_hash") && view.layers_hash == expected.at("layers_hash") && view.hash == expected.at("hash"), "native frozen identities");
        require(words(view.log_exposure.data, view.log_exposure.count) == table.at("axis"), "borrowed interpolation axis");
        require(view.authored_curves_cmy.count == 3 * view.log_exposure.count && view.normalized_curves_cmy.count == view.authored_curves_cmy.count, "3N shapes");
        for (std::size_t i = 0; i < view.log_exposure.count; ++i) {
            require(words(view.authored_curves_cmy.data + 3 * i, 3) == table.at("authored")[i] && words(view.normalized_curves_cmy.data + 3 * i, 3) == table.at("normalized")[i], "native frozen CMY rows");
        }
        for (std::size_t layer = 0; layer < 3; ++layer)
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const auto lane = view.density_curves_layers[layer][channel];
                if (view.layers_required)
                    require(words(lane.data, lane.count) == table.at("layers")[layer][channel], "layer/channel/sample order");
                else
                    require(!lane.data && lane.count == 0 && view.layers_hash == 0, "omitted layer coupling");
            }
    }
    void layouts() {
        const std::size_t cpp[] = {
#define FJ_DEVELOPMENT_TYPE(type) sizeof(type), alignof(type),
#define FJ_DEVELOPMENT_FIELD(type, field) offsetof(type, field),
#define FJ_DEVELOPMENT_EXTENT(field, element) sizeof(FjFilmDevelopmentView::field) / sizeof(element),
#define FJ_DEVELOPMENT_VALUE(value) value,
#include "film_development_abi_facts.inc"
#undef FJ_DEVELOPMENT_TYPE
#undef FJ_DEVELOPMENT_FIELD
#undef FJ_DEVELOPMENT_EXTENT
#undef FJ_DEVELOPMENT_VALUE
        };
        std::size_t cCount = 0, rustCount = 0;
        const auto* c = fj_test_film_development_abi_c_facts(&cCount);
        const auto* rust = fj_test_film_development_abi_facts(&rustCount);
        require(cCount == std::size(cpp) && rustCount == cCount, "complete coordinated layout fact membership");
        for (std::size_t i = 0; i < cCount; ++i)
            require(cpp[i] == c[i] && c[i] == rust[i], "C11/C++20/Rust layout/extent/tag equality");
        std::printf("PASS %zu C11/C++20/Rust ABI facts; view size=%zu\n", cCount, sizeof(FjFilmDevelopmentView));
    }
    struct Assets {
        FjAssets* owner = nullptr;
        explicit Assets(const std::filesystem::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            status(fj_legacy_assets_create(path.view(), &owner, nullptr));
        }
        ~Assets() {
            if (fj_legacy_assets_destroy(owner, nullptr).category != FJ_STATUS_SUCCESS)
                std::terminate();
        }
    };
    void clear(FjFilmDevelopmentView view) {
        require(view.route == 0 && view.polarity == 0 && view.layers_required == 0 && view.hash == 0 && view.layers_hash == 0 && view.normalized_curves_hash == 0, "failure clears scalar fields");
        for (auto span : {view.log_exposure, view.authored_curves_cmy, view.normalized_curves_cmy})
            require(!span.data && span.count == 0, "failure clears spans");
        for (const auto& layer : view.density_curves_layers)
            for (auto span : layer)
                require(!span.data && span.count == 0, "failure clears layers");
        require(words(view.gamma_rgb, 3) == Json::array({0, 0, 0}) && words(view.minimum_cmy, 3) == Json::array({0, 0, 0}) && words(view.maximum_cmy, 3) == Json::array({0, 0, 0}), "failure clears fixed arrays");
    }
    void raw_cases(const std::filesystem::path& root, const Json& fixture) {
        Assets assets(root);
        for (bool facade : {false, true}) {
            const auto acquire = facade ? fj_test_film_development_acquire : fj_legacy_film_development_acquire;
            const auto view = facade ? fj_test_film_development_view : fj_legacy_film_development_view;
            const auto release = facade ? fj_test_film_development_release : fj_legacy_film_development_release;
            bool viewFaultWitnessed = false;
            for (const auto& row : fixture.at("cases")) {
                const auto key = row.at("key").get<std::string>();
                FjFilmProfile* profile = nullptr;
                status(fj_legacy_film_profile_acquire(assets.owner, {key.data(), key.size()}, &profile, nullptr));
                FjFilmDevelopmentOwner* owner = nullptr;
                const auto gamma = std::bit_cast<float>(row.at("development").at("gamma")[0].get<std::uint32_t>());
                status(acquire(profile, row.at("route"), gamma, row.at("development").at("layers_required").get<bool>() ? 1u : 0u, &owner, nullptr));
                status(fj_legacy_film_profile_release(profile, nullptr));
                status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr));
                FjFilmDevelopmentView result{};
                status(view(owner, &result, nullptr));
                compare(result, row, fixture.at("profiles").at(key));
                if (!viewFaultWitnessed) {
                    viewFaultWitnessed = true;
                    status(fj_test_film_development_fault(1, 1));
                    FjFilmDevelopmentView failedView{};
                    failedView.route = 99;
                    failedView.hash = 99;
                    std::array<char, 128> viewMessage{};
                    FjErrorBuffer viewError{viewMessage.data(), viewMessage.size(), 0};
                    status(view(owner, &failedView, &viewError), FJ_STATUS_INTERNAL_FAILURE);
                    clear(failedView);
                    require(std::string_view(viewMessage.data(), viewError.length) == "asset boundary panic",
                            "native operation-1 panic category and diagnostic");
                    require(fj_test_film_development_live_owners() == 1,
                            "failed native view leaves its owner live for retry");
                    FjFilmDevelopmentView retriedView{};
                    status(view(owner, &retriedView, nullptr));
                    compare(retriedView, row, fixture.at("profiles").at(key));
                }
                status(release(owner, nullptr));
                require(fj_test_film_development_live_owners() == 0, "consume-once owner release");
            }
        }
        FjFilmProfile* profile = nullptr;
        status(fj_legacy_film_profile_acquire(assets.owner, {"kodak_portra_400", 16}, &profile, nullptr));
        for (auto acquire : {fj_legacy_film_development_acquire, fj_test_film_development_acquire}) {
            for (auto route : {2u, 3u, 4u, UINT32_MAX}) {
                auto* output = reinterpret_cast<FjFilmDevelopmentOwner*>(std::uintptr_t{1});
                status(acquire(profile, route, 1, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                require(!output, "route failure clears owner");
            }
            for (float gamma : {0.0f, 0.049f, 4.001f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                FjFilmDevelopmentOwner* output = nullptr;
                status(acquire(profile, 0, gamma, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                require(!output, "gamma failure has no owner");
            }
            FjFilmDevelopmentOwner* output = nullptr;
            status(acquire(profile, 0, 1, 2, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            status(acquire(nullptr, 0, 1, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            status(acquire(profile, 0, 1, 0, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                status(fj_test_film_development_fault(0, fault));
                std::array<char, 5> message{};
                FjErrorBuffer error{message.data(), message.size(), 0};
                status(acquire(profile, 0, 1, 0, &output, &error), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : (fault == 2 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_PREPARATION_FAILURE));
                require(!output && error.length < message.size() && message[error.length] == '\0', "failure output and bounded diagnostic");
                require(fj_test_film_development_live_owners() == 0, "failure cleanup");
            }
        }
        for (auto view : {fj_legacy_film_development_view, fj_test_film_development_view}) {
            FjFilmDevelopmentView output{};
            output.hash = 99;
            output.route = 99;
            status(view(nullptr, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            clear(output);
        }
        for (auto release : {fj_legacy_film_development_release, fj_test_film_development_release}) {
            status(release(nullptr, nullptr));
            for (bool malformed : {false, true}) {
                FjFilmDevelopmentOwner* output = nullptr;
                status(fj_legacy_film_development_acquire(profile, 0, 1, 1, &output, nullptr));
                if (!malformed)
                    status(fj_test_film_development_fault(2, 1));
                FjErrorBuffer error{nullptr, 1, 99};
                status(release(output, malformed ? &error : nullptr), malformed ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                require(fj_test_film_development_live_owners() == 0, "release failure still consumes owner");
            }
        }
        status(fj_legacy_film_profile_release(profile, nullptr));
    }
    void products(const std::filesystem::path& root, const Json& fixture) {
        JuicerCuda::Owner owner;
        owner.create(root);
        std::size_t count = 0;
        for (const auto& row : fixture.at("cases")) {
            ParamSnapshot controls;
            controls.filmProfileKey = row.at("key");
            controls.printProfileKey = "kodak_portra_endura";
            controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(row.at("route").get<unsigned>());
            controls.spectralUpsamplingMode = static_cast<int>(Spektrafilm::RgbToRawMethod::Mallett2019);
            controls.cameraAutoExposureEnabled = 0;
            controls.inputCompressionEnabled = 0;
            controls.outputGamutCompressionEnabled = 0;
            controls.filmGammaFactor = std::bit_cast<float>(row.at("development").at("gamma")[0].get<std::uint32_t>());
            controls.grainControls.active = row.at("grain_policy") != 0;
            controls.grainControls.sublayersActive = row.at("grain_policy") != 1;
            auto product = JuicerTestSupport::prepare_product(controls);
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            FjPreparedStateView view{};
            status(product.view(view, error));
            require(hash_params(controls) == row.at("snapshot_hash") && view.recipe_hash == row.at("final_hash"), "independent complete snapshot/final identity");
            require(view.film_profile_asset_version == row.at("profile_token") && view.film_development.hash == row.at("development").at("hash"), "profile/development identities survive actual consumers");
            require(view.scanner_recipe_hash == row.at("scanner_hash") && view.film_reference.hash == row.at("reference_hash"), "scanner/reference completion precedes success");
            ++count;
        }
        require(count == 480, "complete actual-consumer matrix");
        std::size_t failureRows = 0;
        for (unsigned route = 0; route < 4; ++route) {
            for (bool layers : {false, true}) {
                ParamSnapshot controls;
                controls.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                controls.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                controls.spectralUpsamplingMode = static_cast<int>(Spektrafilm::RgbToRawMethod::Mallett2019);
                controls.cameraAutoExposureEnabled = 0;
                controls.inputCompressionEnabled = 0;
                controls.outputGamutCompressionEnabled = 0;
                controls.grainControls.active = true;
                controls.grainControls.sublayersActive = layers;
                InstanceState state;
                const auto pending = [&state](const ParamSnapshot& snapshot) {
                    std::lock_guard<std::mutex> lock(state.pending.m);
                    state.pending.value = PendingParamsState::Valid{snapshot, hash_params(snapshot)};
                };
                pending(controls);
                auto publication = admit_pending_render_state(state);
                require(route % 2 ? bool(publication.printState) : bool(publication.directState), "failure baseline complete admission");
                require(fj_test_prepared_live_states() == 1 && fj_test_development_curve_count(0) == 1,
                        "one complete state owns the baseline normalized curves");
                std::uint64_t counter = state.buildCounterNext.load();
                std::uint64_t identity = state.lastHash.load();
                FjPreparedStateView oldView = JuicerTestSupport::published_view(publication);
                Json oldCurveWords = words(oldView.film_development.density_rgb.data,
                                           oldView.film_development.density_rgb.count);
                const auto layer_words = [](const FjPreparedStateView& view) {
                    Json result = Json::array();
                    for (const auto& layer : view.film_development.density_layers) {
                        Json channels = Json::array();
                        for (const auto span : layer)
                            channels.push_back(words(span.data, span.count));
                        result.push_back(std::move(channels));
                    }
                    return result;
                };
                Json oldLayerWords = layer_words(oldView);
                unsigned attempt = 0;
                for (const unsigned fault : {1u, 2u, 3u, 4u}) {
                    if (fault == 4 && !layers)
                        continue;
                    auto changed = controls;
                    changed.filmGammaFactor = 1.1f + 0.1f * static_cast<float>(attempt++);
                    const std::uint64_t changedHash = hash_params(changed);
                    pending(changed);
                    auto previousDirect = JuicerAtomic::load_shared_ptr(&state.activeDirectState);
                    auto previousPrint = JuicerAtomic::load_shared_ptr(&state.activePrintState);
                    const std::string_view expectedDiagnostic = fault == 1
                                                                    ? "film development normalized table reservation failed"
                                                                : fault == 4
                                                                    ? "MalformedRequiredProfileData phase=9A field=visual_grain"
                                                                    : "MalformedRequiredProfileData phase=3A field=data.density_curves hash";
                    status(fj_test_development_construction_fault(fault));
                    std::optional<PreparedState> failedOwner;
                    std::array<char, 4096> constructionText{};
                    FjErrorBuffer constructionError{constructionText.data(), constructionText.size(), 0};
                    status(prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), changed, failedOwner, constructionError),
                           fault == 1 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_PREPARATION_FAILURE);
                    require(!failedOwner && std::string_view(constructionText.data(), constructionError.length) == expectedDiagnostic,
                            "complete constructor reports the real leaf category/diagnostic and returns no partial owner");
                    const auto allocationsBefore = fj_test_development_curve_count(1);
                    const auto dropsBefore = fj_test_development_curve_count(2);
                    status(fj_test_development_construction_fault(fault));
                    bool allocationThrown = false;
                    PendingRenderAdmissionResult rejected;
                    try {
                        rejected = admit_pending_render_state(state);
                    } catch (const std::bad_alloc&) {
                        allocationThrown = true;
                    }
                    if (fault == 1) {
                        require(allocationThrown, "core development reservation maps to bad_alloc at actual admission");
                    } else {
                        require(!allocationThrown && rejected.status == PendingRenderAdmissionStatus::RebuildFailed &&
                                    !rejected.directState && !rejected.printState && rejected.diagnostic == expectedDiagnostic,
                                "core development identity/layer failure keeps its preparation diagnostic");
                    }
                    const std::size_t temporaryAllocations = fault == 1 ? 0 : 1;
                    require(fj_test_development_curve_count(0) == 1 &&
                                fj_test_development_curve_count(1) == allocationsBefore + temporaryAllocations &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations,
                            "failed reservation allocates nothing; later failures reclaim the actual temporary normalized curves");
                    ++failureRows;
                    const bool retainPublication = fault == 1;
                    auto expectedDirect = route % 2 == 0 && !retainPublication
                                              ? std::shared_ptr<const DirectRenderState>{}
                                              : previousDirect;
                    auto expectedPrint = route % 2 != 0 && !retainPublication
                                             ? std::shared_ptr<const PrintRenderState>{}
                                             : previousPrint;
                    require(state.buildCounterNext.load() == counter && state.lastHash.load() == identity &&
                                state.lastHash.load() != changedHash &&
                                JuicerAtomic::load_shared_ptr(&state.activeDirectState) == expectedDirect &&
                                JuicerAtomic::load_shared_ptr(&state.activePrintState) == expectedPrint,
                            "development failure preserves counters/hash and applies its route-slot policy");
                    const auto heldAfterFailure = JuicerTestSupport::published_view(publication);
                    require(heldAfterFailure.recipe_hash == oldView.recipe_hash &&
                                heldAfterFailure.film_development.hash == oldView.film_development.hash &&
                                heldAfterFailure.film_development.density_curves_hash == oldView.film_development.density_curves_hash &&
                                heldAfterFailure.film_development.density_layers_hash == oldView.film_development.density_layers_hash &&
                                words(heldAfterFailure.film_development.density_rgb.data,
                                      heldAfterFailure.film_development.density_rgb.count) == oldCurveWords &&
                                layer_words(heldAfterFailure) == oldLayerWords,
                            "failed construction leaves the admitted state and borrowed curves readable");
                    require(fj_test_prepared_live_states() == 1 && fj_test_film_development_live_owners() == 0,
                            "failed complete construction publishes no opaque state and creates no legacy foreign owner");

                    const auto recovered = admit_pending_render_state(state);
                    require(route % 2 ? bool(recovered.printState) : bool(recovered.directState),
                            "development one-shot failure retry publishes the selected route");
                    const auto recoveredView = JuicerTestSupport::published_view(recovered);
                    require(recoveredView.recipe_hash != oldView.recipe_hash &&
                                recoveredView.film_development.hash != oldView.film_development.hash &&
                                recoveredView.film_development.density_curves_hash == oldView.film_development.density_curves_hash &&
                                recoveredView.film_development.density_layers_hash == oldView.film_development.density_layers_hash &&
                                words(recoveredView.film_development.density_rgb.data,
                                      recoveredView.film_development.density_rgb.count) == oldCurveWords &&
                                layer_words(recoveredView) == oldLayerWords &&
                                state.buildCounterNext.load() == counter + 1 && state.lastHash.load() == changedHash,
                            "retry publishes once with exact curve/layer content and updated identity");
                    require(fj_test_development_curve_count(0) == 2 &&
                                fj_test_development_curve_count(1) == allocationsBefore + temporaryAllocations + 1 &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations,
                            "retry holds both old and new actual normalized allocations without reclaiming the admitted one");
                    require(fj_test_prepared_live_states() == 2,
                            "recovery publication coexists with the prior admitted state");
                    publication = recovered;
                    oldView = recoveredView;
                    oldCurveWords = words(oldView.film_development.density_rgb.data,
                                          oldView.film_development.density_rgb.count);
                    oldLayerWords = layer_words(oldView);
                    counter = state.buildCounterNext.load();
                    identity = state.lastHash.load();
                    expectedDirect.reset();
                    expectedPrint.reset();
                    previousDirect.reset();
                    previousPrint.reset();
                    require(fj_test_prepared_live_states() == 1 && fj_test_development_curve_count(0) == 1 &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations + 1,
                            "replacing the test hold releases only the prior complete state and its actual normalized curves");
                }
                const auto warm = admit_pending_render_state(state);
                require(warm.directState == publication.directState && warm.printState == publication.printState,
                        "identical current snapshot reuses the complete state");
            }
        }
        require(failureRows == 28, "complete actual development fault membership");
        std::puts("PASS 28 actual development fault/retry rows");
        std::puts("PASS 480 independent complete direct/print recipe and reference/scanner/final identities");
    }
} // namespace
int main(int argc, char** argv) try { // NOLINT(bugprone-exception-escape): all fixture failures become nonzero exits.
    require(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "products"), "resources and immutable native capture required");
    std::ifstream input{std::filesystem::path(argv[2])};
    const auto fixture = PrintIlluminantFixtures::apply(argv[2], Json::parse(input));
    if (argc == 4) {
        products(argv[1], fixture);
    } else {
        layouts();
        raw_cases(argv[1], fixture);
    }
    std::puts("PASS film development: exact raw/facade values, lifetime, failures and ABI");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("unknown film development failure\n", stderr);
    return 1;
}
