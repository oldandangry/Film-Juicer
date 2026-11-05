#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "JuicerState.h"
#include "PreparedState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"

namespace JuicerTestSupport {

    inline FjPreparedStateView published_view(const PendingRenderAdmissionResult& publication) {
        const PreparedState* state = publication.printState ? &publication.printState->prepared : publication.directState ? &publication.directState->prepared
                                                                                                                          : nullptr;
        if (!state)
            throw std::runtime_error("publication has no prepared state");
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView view{};
        if (state->view(view, error).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error(std::string(diagnostic.data(), std::min(error.length, diagnostic.size())));
        return view;
    }

    inline PreparedState prepare_product(const ParamSnapshot& snapshot) {
        auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
        if (assets == nullptr)
            throw std::runtime_error("prepared product has no process Rust assets owner");

        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::optional<PreparedState> state;
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
