#include <array>
#include <barrier>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <new>
#include <string>
#include <string_view>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

#include "ProcessRuntime.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "NativePath.h"
#include "juicer_test_api.h"
#include "prepared_product_fixture.h"
#include "case_results.h"
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
                const std::string key = row.at("key").get<std::string>();
                const std::string row_id = key + "/route-" + std::to_string(row.at("route").get<unsigned>()) + "/gamma-" + std::to_string(row.at("development").at("gamma")[0].get<std::uint32_t>()) + "/layers-" + (row.at("development").at("layers_required").get<bool>() ? "1" : "0") + "/grain-" + std::to_string(row.at("grain_policy").get<int>());
                JuicerTests::CaseEvent event("film_development.raw", std::string(facade ? "facade/" : "legacy/") + row_id);
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
                event.passed();
            }
        }
        JuicerTests::CaseEvent failure_event("film_development.raw", "boundary_fault_order_and_consumption");
        FjFilmProfile* profile = nullptr;
        status(fj_legacy_film_profile_acquire(assets.owner, {"kodak_portra_400", 16}, &profile, nullptr));
        std::vector<std::string> route_ids;
        std::vector<std::string> gamma_ids;
        std::vector<std::string> acquire_fault_ids;
        for (const char* caller : {"legacy", "facade"}) {
            for (unsigned route : {2u, 3u, 4u, UINT32_MAX})
                route_ids.emplace_back(std::string(caller) + "/route-" + std::to_string(route));
            for (unsigned i = 0; i < 5; ++i)
                gamma_ids.emplace_back(std::string(caller) + "/gamma-" + std::to_string(i));
            for (unsigned fault = 1; fault <= 4; ++fault)
                acquire_fault_ids.emplace_back(std::string(caller) + "/fault-" + std::to_string(fault));
        }
        JuicerTests::Cohort invalid_routes("film_development.raw.invalid_route", route_ids);
        JuicerTests::Cohort invalid_gammas("film_development.raw.invalid_gamma", gamma_ids);
        JuicerTests::Cohort acquire_faults("film_development.raw.acquire_fault", acquire_fault_ids);
        std::size_t route_index = 0, gamma_index = 0, acquire_fault_index = 0;
        for (auto acquire : {fj_legacy_film_development_acquire, fj_test_film_development_acquire}) {
            for (auto route : {2u, 3u, 4u, UINT32_MAX}) {
                auto row_event = invalid_routes.case_event(route_ids[route_index++]);
                auto* output = reinterpret_cast<FjFilmDevelopmentOwner*>(std::uintptr_t{1});
                status(acquire(profile, route, 1, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                require(!output, "route failure clears owner");
                row_event.passed();
            }
            for (float gamma : {0.0f, 0.049f, 4.001f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                auto row_event = invalid_gammas.case_event(gamma_ids[gamma_index++]);
                FjFilmDevelopmentOwner* output = nullptr;
                status(acquire(profile, 0, gamma, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
                require(!output, "gamma failure has no owner");
                row_event.passed();
            }
            FjFilmDevelopmentOwner* output = nullptr;
            status(acquire(profile, 0, 1, 2, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            status(acquire(nullptr, 0, 1, 0, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            status(acquire(profile, 0, 1, 0, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                auto row_event = acquire_faults.case_event(acquire_fault_ids[acquire_fault_index++]);
                status(fj_test_film_development_fault(0, fault));
                std::array<char, 5> message{};
                FjErrorBuffer error{message.data(), message.size(), 0};
                status(acquire(profile, 0, 1, 0, &output, &error), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : (fault == 2 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_PREPARATION_FAILURE));
                require(!output && error.length < message.size() && message[error.length] == '\0', "failure output and bounded diagnostic");
                require(fj_test_film_development_live_owners() == 0, "failure cleanup");
                row_event.passed();
            }
        }
        JuicerTests::Cohort view_callers("film_development.raw.view_caller", {"legacy", "facade"});
        for (const auto& [caller, view] : {std::pair{"legacy", &fj_legacy_film_development_view}, std::pair{"facade", &fj_test_film_development_view}}) {
            auto row_event = view_callers.case_event(caller);
            FjFilmDevelopmentView output{};
            output.hash = 99;
            output.route = 99;
            status(view(nullptr, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            clear(output);
            row_event.passed();
        }
        JuicerTests::Cohort release_callers("film_development.raw.release_caller", {"legacy/ordinary", "legacy/malformed", "facade/ordinary", "facade/malformed"});
        for (const auto& [caller, release] : {std::pair{"legacy", &fj_legacy_film_development_release}, std::pair{"facade", &fj_test_film_development_release}}) {
            status(release(nullptr, nullptr));
            for (bool malformed : {false, true}) {
                auto row_event = release_callers.case_event(std::string(caller) + (malformed ? "/malformed" : "/ordinary"));
                FjFilmDevelopmentOwner* output = nullptr;
                status(fj_legacy_film_development_acquire(profile, 0, 1, 1, &output, nullptr));
                if (!malformed)
                    status(fj_test_film_development_fault(2, 1));
                FjErrorBuffer error{nullptr, 1, 99};
                status(release(output, malformed ? &error : nullptr), malformed ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                require(fj_test_film_development_live_owners() == 0, "release failure still consumes owner");
                row_event.passed();
            }
        }
        status(fj_legacy_film_profile_release(profile, nullptr));
        failure_event.passed();
    }
    void products(const std::filesystem::path& root, const Json& fixture) {
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        std::size_t count = 0;
        for (const auto& row : fixture.at("cases")) {
            const std::string row_id = row.at("key").get<std::string>() + "/route-" + std::to_string(row.at("route").get<unsigned>()) + "/grain-" + std::to_string(row.at("grain_policy").get<int>()) + "/gamma-" + std::to_string(row.at("development").at("gamma")[0].get<std::uint32_t>());
            JuicerTests::CaseEvent event("film_development.products", row_id);
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
            auto product = JuicerTestSupport::prepare_product(owner.borrow().assets, controls);
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            FjPreparedStateView view{};
            status(product.view(view, error));
            require(hash_params(controls) == row.at("snapshot_hash") && view.recipe_hash == row.at("final_hash"), "independent complete snapshot/final identity");
            require(view.film_profile_asset_version == row.at("profile_token") && view.film_development.hash == row.at("development").at("hash"), "profile/development identities survive actual consumers");
            require(view.scanner_recipe_hash == row.at("scanner_hash") && view.film_reference.hash == row.at("reference_hash"), "scanner/reference completion precedes success");
            ++count;
            event.passed();
        }
        require(count == 480, "complete actual-consumer matrix");
        JuicerTests::CaseEvent failure_event("film_development.products", "failure_retry_and_publication");
        std::size_t failureRows = 0;
        std::vector<std::string> fault_ids;
        for (unsigned route = 0; route < 4; ++route)
            for (bool layers : {false, true})
                for (unsigned fault = 1; fault <= 4; ++fault)
                    if (fault != 4 || layers)
                        fault_ids.emplace_back("route-" + std::to_string(route) + "/layers-" + (layers ? "1" : "0") + "/fault-" + std::to_string(fault));
        JuicerTests::Cohort product_faults("film_development.products.product_fault", fault_ids);
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
                RenderInstance instance;
                JuicerTestSupport::capture(instance, controls);
                auto publication = JuicerTestSupport::admit(owner.borrow().assets, instance);
                require(publication.kind == FJ_ADMISSION_READY && publication.state, "failure baseline complete admission");
                require(fj_test_development_curve_count(0) == 1,
                        "one complete state owns the baseline normalized curves");
                auto metrics = JuicerTestSupport::PublicationObserver::metrics(instance);
                std::uint64_t counter = metrics.build;
                std::uint64_t identity = metrics.publishedIdentity;
                std::size_t backingIdentity = metrics.backingIdentity;
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
                ParamSnapshot committedControls = controls;
                unsigned attempt = 0;
                for (const unsigned fault : {1u, 2u, 3u, 4u}) {
                    if (fault == 4 && !layers)
                        continue;
                    auto row_event = product_faults.case_event("route-" + std::to_string(route) + "/layers-" + (layers ? "1" : "0") + "/fault-" + std::to_string(fault));
                    auto changed = controls;
                    changed.filmGammaFactor = 1.1f + 0.1f * static_cast<float>(attempt++);
                    const std::uint64_t changedHash = hash_params(changed);
                    JuicerTestSupport::capture(instance, changed);
                    const std::string_view expectedDiagnostic = fault == 1
                                                                    ? "film development normalized table reservation failed"
                                                                : fault == 4
                                                                    ? "MalformedRequiredProfileData phase=9A field=visual_grain"
                                                                    : "MalformedRequiredProfileData phase=3A field=data.density_curves hash";
                    status(fj_test_development_construction_fault(fault));
                    std::optional<PreparedState> failedOwner;
                    std::array<char, 4096> constructionText{};
                    FjErrorBuffer constructionError{constructionText.data(), constructionText.size(), 0};
                    status(prepare_state(owner.borrow().assets, changed, failedOwner, constructionError),
                           fault == 1 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_PREPARATION_FAILURE);
                    require(!failedOwner && std::string_view(constructionText.data(), constructionError.length) == expectedDiagnostic,
                            "complete constructor reports the real leaf category/diagnostic and returns no partial owner");
                    const auto allocationsBefore = fj_test_development_curve_count(1);
                    const auto dropsBefore = fj_test_development_curve_count(2);
                    status(fj_test_development_construction_fault(fault));
                    bool allocationThrown = false;
                    JuicerTestSupport::PublicationAdmission rejected;
                    try {
                        rejected = JuicerTestSupport::admit(owner.borrow().assets, instance);
                    } catch (const std::bad_alloc&) {
                        allocationThrown = true;
                    }
                    if (fault == 1) {
                        require(allocationThrown, "core development reservation maps to bad_alloc at actual admission");
                    } else {
                        require(!allocationThrown && rejected.delivery.category == FJ_STATUS_PREPARATION_FAILURE &&
                                    rejected.kind == FJ_ADMISSION_PREPARATION_FAILED && !rejected.state && rejected.diagnostic == expectedDiagnostic,
                                "core development identity/layer failure keeps its preparation diagnostic");
                    }
                    const std::size_t temporaryAllocations = fault == 1 ? 0 : 1;
                    require(fj_test_development_curve_count(0) == 1 &&
                                fj_test_development_curve_count(1) == allocationsBefore + temporaryAllocations &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations,
                            "failed reservation allocates nothing; later failures reclaim the actual temporary normalized curves");
                    ++failureRows;
                    metrics = JuicerTestSupport::PublicationObserver::metrics(instance);
                    require(metrics.build == counter && metrics.publishedIdentity == identity &&
                                metrics.inputIdentity == changedHash && metrics.backingIdentity == backingIdentity,
                            "development failure preserves the old publication metadata while current controls remain changed");
                    const auto heldAfterFailure = JuicerTestSupport::published_view(publication);
                    require(heldAfterFailure.recipe_hash == oldView.recipe_hash &&
                                heldAfterFailure.film_development.hash == oldView.film_development.hash &&
                                heldAfterFailure.film_development.density_curves_hash == oldView.film_development.density_curves_hash &&
                                heldAfterFailure.film_development.density_layers_hash == oldView.film_development.density_layers_hash &&
                                words(heldAfterFailure.film_development.density_rgb.data,
                                      heldAfterFailure.film_development.density_rgb.count) == oldCurveWords &&
                                layer_words(heldAfterFailure) == oldLayerWords,
                            "failed construction leaves the admitted state and borrowed curves readable");
                    require(fj_test_film_development_live_owners() == 0,
                            "failed complete construction creates no legacy foreign owner");

                    JuicerTestSupport::capture(instance, committedControls);
                    auto reused = JuicerTestSupport::admit(owner.borrow().assets, instance);
                    require(reused.kind == FJ_ADMISSION_READY && reused.state &&
                                JuicerTestSupport::PublicationObserver::metrics(instance).build == counter &&
                                JuicerTestSupport::PublicationObserver::backing(*reused.state) == backingIdentity &&
                                JuicerTestSupport::published_view(reused).recipe_hash == oldView.recipe_hash,
                            "returning to the last admitted identity reuses its complete state after failure");
                    reused.state.reset();
                    JuicerTestSupport::capture(instance, changed);
                    auto recovered = JuicerTestSupport::admit(owner.borrow().assets, instance);
                    require(recovered.kind == FJ_ADMISSION_READY && recovered.state,
                            "development one-shot failure retry publishes the selected route");
                    const auto recoveredView = JuicerTestSupport::published_view(recovered);
                    require(recoveredView.recipe_hash != oldView.recipe_hash &&
                                recoveredView.film_development.hash != oldView.film_development.hash &&
                                recoveredView.film_development.density_curves_hash == oldView.film_development.density_curves_hash &&
                                recoveredView.film_development.density_layers_hash == oldView.film_development.density_layers_hash &&
                                words(recoveredView.film_development.density_rgb.data,
                                      recoveredView.film_development.density_rgb.count) == oldCurveWords &&
                                layer_words(recoveredView) == oldLayerWords &&
                                JuicerTestSupport::PublicationObserver::metrics(instance).build == counter + 1 &&
                                JuicerTestSupport::PublicationObserver::metrics(instance).publishedIdentity == changedHash,
                            "retry publishes once with exact curve/layer content and updated identity");
                    require(fj_test_development_curve_count(0) == 2 &&
                                fj_test_development_curve_count(1) == allocationsBefore + temporaryAllocations + 1 &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations,
                            "retry holds both old and new actual normalized allocations without reclaiming the admitted one");
                    publication = std::move(recovered);
                    committedControls = changed;
                    oldView = recoveredView;
                    oldCurveWords = words(oldView.film_development.density_rgb.data,
                                          oldView.film_development.density_rgb.count);
                    oldLayerWords = layer_words(oldView);
                    metrics = JuicerTestSupport::PublicationObserver::metrics(instance);
                    counter = metrics.build;
                    identity = metrics.publishedIdentity;
                    backingIdentity = metrics.backingIdentity;
                    require(fj_test_development_curve_count(0) == 1 &&
                                fj_test_development_curve_count(2) == dropsBefore + temporaryAllocations + 1,
                            "replacing the test hold releases only the prior complete state and its actual normalized curves");
                    row_event.passed();
                }
                const auto warm = JuicerTestSupport::admit(owner.borrow().assets, instance);
                require(warm.kind == FJ_ADMISSION_READY && warm.state &&
                            JuicerTestSupport::PublicationObserver::metrics(instance).build == counter &&
                            JuicerTestSupport::PublicationObserver::metrics(instance).backingIdentity == backingIdentity,
                        "identical current snapshot reuses the complete state");
            }
        }
        require(failureRows == 28, "complete actual development fault membership");
        failure_event.passed();
        std::puts("PASS 28 actual development fault/retry rows");
        std::puts("PASS 480 independent complete direct/print recipe and reference/scanner/final identities");
    }
} // namespace
int main(int argc, char** argv) try { // NOLINT(bugprone-exception-escape): all fixture failures become nonzero exits.
    JuicerTests::CaseLog case_log;
    if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
        std::ifstream listing{std::filesystem::path(argv[2])};
        const auto listed = Json::parse(listing);
        if (argc == 5) {
            for (const auto& row : listed.at("cases")) {
                const std::string id = row.at("key").get<std::string>() + "/route-" + std::to_string(row.at("route").get<unsigned>()) + "/grain-" + std::to_string(row.at("grain_policy").get<int>()) + "/gamma-" + std::to_string(row.at("development").at("gamma")[0].get<std::uint32_t>());
                JuicerTests::list_case("film_development.products", id);
            }
            JuicerTests::list_case("film_development.products", "failure_retry_and_publication");
        } else {
            for (const auto& row : listed.at("cases")) {
                const std::string id = row.at("key").get<std::string>() + "/route-" + std::to_string(row.at("route").get<unsigned>()) + "/gamma-" + std::to_string(row.at("development").at("gamma")[0].get<std::uint32_t>()) + "/layers-" + (row.at("development").at("layers_required").get<bool>() ? "1" : "0") + "/grain-" + std::to_string(row.at("grain_policy").get<int>());
                JuicerTests::list_case("film_development.raw", "legacy/" + id);
                JuicerTests::list_case("film_development.raw", "facade/" + id);
            }
            JuicerTests::list_case("film_development.raw", "boundary_fault_order_and_consumption");
            for (const auto* caller : {"legacy", "facade"})
                JuicerTests::list_case("film_development.raw.view_caller", caller);
            for (const auto* id : {"legacy/ordinary", "legacy/malformed", "facade/ordinary", "facade/malformed"})
                JuicerTests::list_case("film_development.raw.release_caller", id);
            JuicerTests::list_case("film_development", "layout_abi");
        }
        return 0;
    }
    require(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "products"), "resources and immutable native capture required");
    std::ifstream input{std::filesystem::path(argv[2])};
    const auto fixture = PrintIlluminantFixtures::apply(argv[2], Json::parse(input));
    if (argc == 4) {
        products(argv[1], fixture);
    } else {
        JuicerTests::CaseEvent layout_event("film_development", "layout_abi");
        layouts();
        layout_event.passed();
        raw_cases(argv[1], fixture);
    }
    case_log.finish();
    std::puts("PASS film development: exact raw/facade values, lifetime, failures and ABI");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("unknown film development failure\n", stderr);
    return 1;
}
