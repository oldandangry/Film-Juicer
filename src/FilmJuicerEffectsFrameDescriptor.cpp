#include "FilmJuicerEffectsFrameDescriptor.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Hash.h"

namespace {

    constexpr std::uint64_t kSeedPassWeave = 3;

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    bool valid_extent(const Spektrafilm::FilmJuicerEffectsFrameExtent& extent) {
        return extent.width > 0 && extent.height > 0;
    }

    bool same_extent(
        const Spektrafilm::FilmJuicerEffectsFrameExtent& a,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& b) {
        return a.x == b.x && a.y == b.y &&
               a.width == b.width && a.height == b.height;
    }

    bool extent_contains(
        const Spektrafilm::FilmJuicerEffectsFrameExtent& outer,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& inner) {
        const std::int64_t outerRight =
            static_cast<std::int64_t>(outer.x) + outer.width;
        const std::int64_t outerBottom =
            static_cast<std::int64_t>(outer.y) + outer.height;
        const std::int64_t innerRight =
            static_cast<std::int64_t>(inner.x) + inner.width;
        const std::int64_t innerBottom =
            static_cast<std::int64_t>(inner.y) + inner.height;
        return inner.x >= outer.x && inner.y >= outer.y &&
               innerRight <= outerRight && innerBottom <= outerBottom;
    }

    double hash_to_unit(std::uint64_t hash) {
        constexpr double kInverseMantissaRange =
            1.0 / 9007199254740992.0;
        return static_cast<double>(hash >> 11) * kInverseMantissaRange;
    }

    double phase_from_seed(
        std::uint64_t sessionSeed,
        std::uint64_t passId,
        int axis,
        int component) {
        const std::uint64_t fields[4] = {
            sessionSeed,
            passId,
            static_cast<std::uint64_t>(axis),
            static_cast<std::uint64_t>(component)};
        std::uint64_t hash = Hash::hash_bytes(fields, sizeof(fields));
        if (hash == 0) {
            hash = 1;
        }
        constexpr double kTwoPi = 6.28318530717958647692;
        return hash_to_unit(hash) * kTwoPi;
    }

    // Argument order mirrors the reviewed Archive signal formula.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    double sin_sum(
        const double* frequencies,
        int count,
        std::uint64_t sessionSeed,
        std::uint64_t passId,
        int axis,
        int componentOffset,
        double timeSeconds) {
        constexpr double kTwoPi = 6.28318530717958647692;
        double sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double phase = phase_from_seed(
                sessionSeed,
                passId,
                axis,
                componentOffset + index);
            sum += std::sin(
                kTwoPi * frequencies[index] * timeSeconds + phase);
        }
        return sum;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    struct GateWeaveSignal final {
        float dxPx = 0.0f;
        float dyPx = 0.0f;
        float cosRot = 1.0f;
        float sinRot = 0.0f;
    };

    // Units are named and passed once from the descriptor builder.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    GateWeaveSignal compute_gate_weave(
        std::uint64_t sessionSeed,
        double timeSeconds,
        double translateRmsUm,
        double rotateRmsDeg,
        double pixelSizeUm,
        double amount) {
        GateWeaveSignal out{};
        if (!(amount > 0.0) || !std::isfinite(pixelSizeUm) ||
            !(pixelSizeUm > 0.0)) {
            return out;
        }

        const double translateRms = translateRmsUm * amount;
        const double rotateRms = rotateRmsDeg * amount;
        if (!(translateRms > 0.0 || rotateRms > 0.0)) {
            return out;
        }

        constexpr double kDriftFrequencies[] = {0.15, 0.35, 0.80};
        constexpr double kJitterFrequencies[] = {6.0, 12.0};
        constexpr int kDriftCount =
            static_cast<int>(std::size(kDriftFrequencies));
        constexpr int kJitterCount =
            static_cast<int>(std::size(kJitterFrequencies));
        const double driftNorm =
            1.0 / std::sqrt(0.5 * static_cast<double>(kDriftCount));
        const double jitterNorm =
            1.0 / std::sqrt(0.5 * static_cast<double>(kJitterCount));
        constexpr double kDriftWeight = 0.85;
        constexpr double kJitterWeight = 0.15;
        const double weightNorm = 1.0 / std::sqrt(
                                            kDriftWeight * kDriftWeight +
                                            kJitterWeight * kJitterWeight);

        for (int axis = 0; axis < 2; ++axis) {
            const double drift = sin_sum(
                                     kDriftFrequencies,
                                     kDriftCount,
                                     sessionSeed,
                                     kSeedPassWeave,
                                     axis,
                                     0,
                                     timeSeconds) *
                                 driftNorm;
            const double jitter = sin_sum(
                                      kJitterFrequencies,
                                      kJitterCount,
                                      sessionSeed,
                                      kSeedPassWeave,
                                      axis,
                                      10,
                                      timeSeconds) *
                                  jitterNorm;
            const double composite =
                (kDriftWeight * drift + kJitterWeight * jitter) *
                weightNorm;
            const float deltaPx = static_cast<float>(
                composite * translateRms / pixelSizeUm);
            if (axis == 0) {
                out.dxPx = deltaPx;
            } else {
                out.dyPx = deltaPx;
            }
        }

        if (rotateRms > 0.0) {
            const double rotationSignal = sin_sum(
                                              kDriftFrequencies,
                                              kDriftCount,
                                              sessionSeed,
                                              kSeedPassWeave,
                                              2,
                                              0,
                                              timeSeconds) *
                                          driftNorm;
            const double rotationDegrees = rotationSignal * rotateRms;
            const double rotationRadians =
                rotationDegrees * (3.14159265358979323846 / 180.0);
            out.cosRot = static_cast<float>(std::cos(rotationRadians));
            out.sinRot = static_cast<float>(std::sin(rotationRadians));
        }
        return out;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    void hash_extent(
        std::uint64_t& hash,
        const Spektrafilm::FilmJuicerEffectsFrameExtent& extent) {
        hash_value(hash, extent.x);
        hash_value(hash, extent.y);
        hash_value(hash, extent.width);
        hash_value(hash, extent.height);
    }

    std::uint64_t hash_descriptor(
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_extent(hash, descriptor.renderExtent);
        hash_extent(hash, descriptor.fullFrameExtent);
        hash_value(hash, descriptor.pixelSizeUm);
        hash_value(hash, descriptor.frame0);
        hash_value(hash, descriptor.frameAlpha);
        hash_value(hash, descriptor.sessionSeed);
        hash_value(hash, descriptor.clipToken);
        hash_value(hash, descriptor.pitchPx);
        hash_value(hash, descriptor.filmDustAmount);
        hash_value(hash, descriptor.filmScratchAmount);
        hash_value(hash, descriptor.gateDustAmount);
        hash_value(hash, descriptor.gateScratchAmount);
        hash_value(hash, descriptor.weaveActive);
        hash_value(hash, descriptor.weaveDxPx);
        hash_value(hash, descriptor.weaveDyPx);
        hash_value(hash, descriptor.weaveCosRot);
        hash_value(hash, descriptor.weaveSinRot);
        hash_value(hash, descriptor.filmActive);
        hash_value(hash, descriptor.gateMaskActive);
        hash_value(hash, descriptor.gateOutputActive);
        hash_value(hash, descriptor.requiresFullFrame);
        hash_value(hash, descriptor.recipeHash);
        return hash;
    }

} // namespace

namespace Spektrafilm {

    bool build_film_juicer_effects_frame_descriptor(
        const FilmJuicerEffectsFrameDescriptorInput& input,
        FilmJuicerEffectsFrameDescriptor& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = FilmJuicerEffectsFrameDescriptor{};
        if (!input.recipe) {
            diagnostic =
                "MissingRequiredResource phase=effects_descriptor field=recipe";
            return false;
        }
        if (!input.recipe->active) {
            return input.recipe->hash == 0;
        }
        if (input.recipe->hash == 0 ||
            !valid_extent(input.renderExtent) ||
            !valid_extent(input.fullFrameExtent) ||
            !extent_contains(input.fullFrameExtent, input.renderExtent) ||
            !std::isfinite(input.pixelSizeUm) ||
            !(input.pixelSizeUm > 0.0f) ||
            !std::isfinite(input.frameTime) ||
            !std::isfinite(input.frameRate) ||
            !(input.frameRate > 0.0)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=effects_descriptor field=frame_input";
            return false;
        }

        const double frameFloor = std::floor(input.frameTime);
        if (frameFloor <
                static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            frameFloor >=
                static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            diagnostic =
                "UnsupportedMode phase=effects_descriptor field=frame_time";
            return false;
        }

        out.renderExtent = input.renderExtent;
        out.fullFrameExtent = input.fullFrameExtent;
        out.pixelSizeUm = input.pixelSizeUm;
        out.frame0 = static_cast<std::int64_t>(frameFloor);
        out.frameAlpha = static_cast<float>(
            std::clamp(input.frameTime - frameFloor, 0.0, 1.0));
        out.sessionSeed = input.sessionSeed != 0 ? input.sessionSeed : 1;
        out.clipToken = input.clipToken;
        out.pitchPx = input.fullFrameExtent.height;
        out.filmDustAmount = input.recipe->filmDustAmount;
        out.filmScratchAmount = input.recipe->filmScratchAmount;
        out.gateDustAmount = input.recipe->gateDustAmount;
        out.gateScratchAmount = input.recipe->gateScratchAmount;
        out.filmActive =
            out.filmDustAmount > 0.0f || out.filmScratchAmount > 0.0f;
        out.gateMaskActive =
            out.gateDustAmount > 0.0f || out.gateScratchAmount > 0.0f;
        out.weaveActive = input.recipe->gateWeaveAmount > 0.0;
        out.gateOutputActive = out.weaveActive || out.gateMaskActive;
        out.requiresFullFrame = out.gateOutputActive;
        if (out.requiresFullFrame &&
            !same_extent(input.renderExtent, input.fullFrameExtent)) {
            diagnostic =
                "UnsupportedMode phase=effects_descriptor field=full_frame_required";
            out = FilmJuicerEffectsFrameDescriptor{};
            return false;
        }

        const double timeSeconds =
            (static_cast<double>(out.frame0) + out.frameAlpha) /
            input.frameRate;
        const GateWeaveSignal weave = compute_gate_weave(
            out.sessionSeed,
            timeSeconds,
            6.0,
            0.005,
            static_cast<double>(out.pixelSizeUm),
            input.recipe->gateWeaveAmount);
        out.weaveDxPx = weave.dxPx;
        out.weaveDyPx = weave.dyPx;
        out.weaveCosRot = weave.cosRot;
        out.weaveSinRot = weave.sinRot;
        out.recipeHash = input.recipe->hash;
        out.hash = hash_descriptor(out);
        if (out.hash == 0) {
            diagnostic =
                "ResourceDescriptorMismatch phase=effects_descriptor field=hash";
            out = FilmJuicerEffectsFrameDescriptor{};
            return false;
        }
        return true;
    }

} // namespace Spektrafilm
