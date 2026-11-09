#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "capture_identity.h"
#include "owner_fixture.h"
#include "preparation_fixture.h"
#include "juicer_test_api.h"

namespace JuicerTestSupport {

    struct SubmissionObservation {
        bool valid = false;
        FjSubmission submission{};
        FjCudaContext context{};
    };

    inline SubmissionObservation submission_observation(const InstanceFixture& instance) {
        SubmissionObservation result{};
        std::uint32_t valid = 0;
        if (fj_test_instance_submission(instance.borrow(), &valid, &result.submission, &result.context, nullptr).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("cannot observe Rust submission");
        }
        result.valid = valid != 0;
        return result;
    }

    class PublicationObservation {
    public:
        PublicationObservation(const PublicationObservation&) = delete;
        PublicationObservation& operator=(const PublicationObservation&) = delete;
        PublicationObservation(PublicationObservation&& other) noexcept
            : view(other.view), owner_(std::exchange(other.owner_, nullptr)) {}
        ~PublicationObservation() {
            if (owner_) {
                (void)fj_test_instance_observation_release(owner_, nullptr);
            }
        }
        FjInstanceObservationView view{};

    private:
        explicit PublicationObservation(FjInstanceObservation* owner) : owner_(owner) {
            if (fj_test_instance_observation_view(owner_, &view, nullptr).category != FJ_STATUS_SUCCESS) {
                (void)fj_test_instance_observation_release(owner_, nullptr);
                owner_ = nullptr;
                throw std::runtime_error("cannot inspect Rust publication");
            }
        }
        FjInstanceObservation* owner_;
        friend class PublicationObserver;
    };

    struct PublicationMetrics {
        std::uint32_t pendingKind = 0;
        std::uint64_t inputIdentity = 0;
        std::uint64_t publishedIdentity = 0;
        std::uint64_t build = 0;
        std::size_t backingIdentity = 0;
        std::uint32_t route = 0;
        std::uint64_t recipeHash = 0;
        std::uint64_t issued = 0;
        std::uint64_t writeRevision = 0;
        std::uint64_t outstandingWrites = 0;
        std::string diagnostic;
    };

    class PublicationObserver {
    public:
        static std::size_t backing(const PreparedStateHold& state) {
            return fj_test_prepared_state_identity(state.borrow());
        }
        static PublicationObservation observe(const InstanceFixture& publication) {
            FjInstanceObservation* owner = nullptr;
            if (fj_test_instance_observation_acquire(publication.borrow(), &owner, nullptr).category != FJ_STATUS_SUCCESS || !owner) {
                throw std::runtime_error("cannot observe Rust publication");
            }
            return PublicationObservation(owner);
        }
        static PublicationMetrics metrics(const InstanceFixture& publication) {
            const auto observation = observe(publication);
            const auto& view = observation.view;
            return {view.pending_kind, view.input_identity, view.published_identity, view.build, view.backing_identity, view.published.route, view.published.recipe_hash, view.issued, view.write_revision, view.outstanding_writes, view.diagnostic.count ? std::string(view.diagnostic.data, view.diagnostic.count) : std::string{}};
        }
    };

    // Copy keys while the borrowed observation is live.
    inline PreparationInput captured_input(const FjPreparationParams& raw) {
        return PreparationInput(raw);
    }

    inline PreparationInput captured_input(const InstanceFixture& publication) {
        const auto observation = PublicationObserver::observe(publication);
        if (observation.view.pending_kind != 4) {
            throw std::runtime_error("publication has no captured input");
        }
        return PreparationInput(observation.view.captured);
    }

    struct PublicationAdmission {
        std::uint32_t kind = FJ_ADMISSION_NONE;
        FjStatus delivery{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        std::shared_ptr<const PreparedStateHold> state;
        std::string diagnostic;
    };

    inline void capture(InstanceFixture& publication, const PreparationInput& snapshot, std::optional<std::uint64_t> identity = std::nullopt) {
        FjErrorBuffer error{nullptr, 0, 0};
        FjCaptureStart capture{};
        FjSnapshotStart snapshotStart{};
        std::uint32_t update = FJ_UPDATE_NONE;
        if (publication.begin_capture(FJ_CAPTURE_CHANGE, capture, error).category != FJ_STATUS_SUCCESS ||
            capture.kind != FJ_CAPTURE_STARTED ||
            publication.begin_snapshot(capture.capture, snapshotStart, error).category != FJ_STATUS_SUCCESS ||
            snapshotStart.kind != FJ_SNAPSHOT_READY ||
            publication.install(capture.capture, snapshotStart.write_revision, snapshot, identity.value_or(hash_input_identity(snapshot)), update, error).category != FJ_STATUS_SUCCESS ||
            update != FJ_UPDATE_APPLIED) {
            throw std::runtime_error("fixture capture did not install its complete snapshot");
        }
    }

    inline void invalidate(InstanceFixture& publication, const std::string& diagnostic) {
        FjErrorBuffer error{nullptr, 0, 0};
        FjCaptureStart capture{};
        std::uint32_t update = FJ_UPDATE_NONE;
        if (publication.begin_capture(FJ_CAPTURE_CHANGE, capture, error).category != FJ_STATUS_SUCCESS ||
            publication.invalidate(capture.capture, FJ_CAPTURE_INVALID_CONTROLS, {diagnostic.data(), diagnostic.size()}, update, error).category != FJ_STATUS_SUCCESS ||
            update != FJ_UPDATE_APPLIED) {
            throw std::runtime_error("fixture invalidation did not complete capture");
        }
    }

    inline PublicationAdmission admit(const FjAssets* assets, InstanceFixture& publication) {
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        PublicationAdmission result;
        std::optional<PreparedStateHold> state;
        result.delivery = publication.admit(assets, result.kind, state, error);
        result.diagnostic.assign(text.data(), error.length);
        if (result.delivery.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (state) {
            result.state = std::make_shared<PreparedStateHold>(std::move(*state));
        }
        return result;
    }

    inline FjPreparedStateView published_view(const PublicationAdmission& publication) {
        if (!publication.state) {
            throw std::runtime_error("publication has no admitted prepared state");
        }
        FjPreparedStateView view{};
        FjErrorBuffer error{nullptr, 0, 0};
        if (publication.state->view(view, error).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("cannot view admitted Rust state");
        }
        return view;
    }

    inline PreparedStateHold prepare_product(const FjAssets* assets, const PreparationInput& snapshot) {
        if (assets == nullptr)
            throw std::runtime_error("prepared product has no process Rust assets owner");

        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedStateHold> state;
        const FjStatus result = prepare_state(assets, snapshot, state, error);
        if (result.category != FJ_STATUS_SUCCESS || !state) {
            const auto length = std::min(error.length, diagnostic.size());
            throw std::runtime_error(std::string(diagnostic.data(), length));
        }
        return std::move(*state);
    }

    // Legacy complete-result captures record three inactive native defaults.
    // Their metadata does not describe a PrintState on a direct route.
    inline std::array<float, 3> captured_print_factors(const FjPreparedStateView& state) {
        const auto& balance = state.print_balance;
        const std::array factors{balance.factor_midgray, balance.factor_midgray_comp, balance.normalizer};
        if ((state.route & 1u) != 0)
            return factors;

        const auto zero = [](float value) {
            return std::bit_cast<std::uint32_t>(value) == 0;
        };
        if (state.print_recipe_hash != 0 || state.print.hash != 0 || state.print.flags != 0 ||
            state.print_profile_asset_version != 0 || state.print_profile_key.count != 0 ||
            balance.hash != 0 || balance.filtered_main_illuminant_hash != 0 ||
            !std::all_of(factors.begin(), factors.end(), zero) ||
            !std::all_of(std::begin(balance.baseline_raw_rgb), std::end(balance.baseline_raw_rgb), zero) ||
            !std::all_of(std::begin(balance.compensated_raw_rgb), std::end(balance.compensated_raw_rgb), zero)) {
            throw std::runtime_error("direct print projection is not inactive/zero before capture metadata mapping");
        }
        for (const auto span : {state.print.film_density_cmy, state.print.film_base_density, state.print.sensitivity_cmy, state.print.log_exposure, state.print.density_cmy, state.print.main_illuminant, state.print.preflash_illuminant}) {
            if (span.data != nullptr || span.count != 0)
                throw std::runtime_error("direct print projection has backing before capture metadata mapping");
        }
        return {1.0f, 1.0f, 1.0f};
    }

} // namespace JuicerTestSupport
