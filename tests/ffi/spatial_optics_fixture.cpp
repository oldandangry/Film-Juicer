#include "spatial_optics_fixture.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "juicer_test_api.h"

namespace {
    FjSpatialOpticsParams defaults() {
        FjSpatialOpticsParams input{};
        input.camera_diffusion.family = 1;
        input.enlarger_diffusion.family = 1;
        return input;
    }

    bool prepare(const FjSpatialOpticsParams& input, FjSpatialOpticsRecipe& out, std::string& diagnostic) {
        diagnostic.clear();
        FjSpatialOpticsFailure failure{};
        const FjStatus status = fj_test_spatial_optics_prepare(&input, &out, &failure, nullptr);
        if (status.category == FJ_STATUS_SUCCESS) {
            return true;
        }
        if (status.category != FJ_STATUS_PREPARATION_FAILURE) {
            throw std::runtime_error("spatial optics fixture boundary failure");
        }
        constexpr std::array<const char*, 16> fields{{"family", "strength", "spatial_scale", "halo_warmth", "core_intensity", "core_size", "halo_intensity", "halo_size", "bloom_intensity", "bloom_size", "scatter_fraction", "group_weights", "group_centers_um", "resolved_hash", "sigma_um", "total_strength"}};
        if (failure.field == 0 || failure.field > fields.size()) {
            throw std::runtime_error("spatial optics fixture unknown field");
        }
        if (failure.reason == FJ_OPTICS_REASON_DIFFUSION) {
            diagnostic = "InvalidDiffusionHostBehavior field=";
        } else if (failure.reason == FJ_OPTICS_REASON_HALATION) {
            diagnostic = "ResourceDescriptorMismatch component=scatter_halation field=";
        } else {
            throw std::runtime_error("spatial optics fixture unexpected component");
        }
        diagnostic += fields[failure.field - 1];
        if (failure.channel < 3) {
            diagnostic += " channel=" + std::to_string(failure.channel);
        }
        return false;
    }
} // namespace

namespace JuicerTestSupport {
    bool prepare_diffusion_filter(
        const Spektrafilm::DiffusionFilterAuthoredControls& authored,
        Spektrafilm::DiffusionFilterResolvedParameters& out,
        std::string& diagnostic) {
        out = {};
        auto input = defaults();
        input.camera_diffusion = {authored.active ? 1u : 0u,
                                  static_cast<std::uint32_t>(authored.family),
                                  authored.strength,
                                  authored.spatialScale,
                                  authored.haloWarmth,
                                  authored.coreIntensity,
                                  authored.coreSize,
                                  authored.haloIntensity,
                                  authored.haloSize,
                                  authored.bloomIntensity,
                                  authored.bloomSize};
        FjSpatialOpticsRecipe result{};
        if (!prepare(input, result, diagnostic)) {
            return false;
        }
        const auto& v = result.camera_diffusion;
        out.active = v.active != 0;
        out.family = static_cast<Spektrafilm::DiffusionFilterFamily>(v.family);
        out.scatterFraction = v.scatter_fraction;
        std::copy_n(v.group_weights, 3, out.groupWeightsCoreHaloBloom.begin());
        std::copy_n(v.group_lambda_um, 3, out.groupCenterLambdaUm.begin());
        out.effectiveWarmth = v.warmth;
        out.spatialScale = v.spatial_scale;
        out.hash = v.resolved_hash;
        return true;
    }

    bool prepare_scatter_halation(
        const Spektrafilm::ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profile,
        ScatterHalationOpticsRecipe& out,
        std::string& diagnostic) {
        out = {};
        auto input = defaults();
        auto& h = input.scatter_halation;
        h.active = controls.active ? 1u : 0u;
        h.scatter_amount = controls.scatterAmount;
        h.scatter_spatial_scale = controls.scatterSpatialScale;
        h.halation_amount = controls.halationAmount;
        h.halation_spatial_scale = controls.halationSpatialScale;
        std::copy_n(profile.halationFirstSigmaUm.begin(), 3, h.first_sigma_um);
        std::copy_n(profile.halationPrimaryAmount.begin(), 3, h.primary_amount);
        FjSpatialOpticsRecipe result{};
        if (!prepare(input, result, diagnostic)) {
            return false;
        }
        const auto& v = result.scatter_halation;
        out.scatterActive = v.scatter_active != 0;
        out.backReflectionActive = v.back_reflection_active != 0;
        out.scatterAmount = v.scatter_amount;
        out.scatterSpatialScale = v.scatter_spatial_scale;
        out.halationSpatialScale = v.halation_spatial_scale;
        std::copy_n(v.first_sigma_um, 3, out.halationFirstSigmaUm.begin());
        std::copy_n(v.total_strength, 3, out.totalStrength.begin());
        out.hash = v.hash;
        return true;
    }
} // namespace JuicerTestSupport
