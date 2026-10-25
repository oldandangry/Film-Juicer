#pragma once

#include <array>
#include <variant>

#include "RenderRecipe.h"
#include "Cuda/JuicerCudaFailure.h"
#include "juicer_legacy_api.h"

namespace JuicerDir {
    // SF_TEMP_BRIDGE_DIR_FRAME: remove mathematical preparation at its final
    // native model caller, no later than S6.D; S4.E narrows final execution inputs.
    // Native workspace composition and its prepared decoder remain native.
    class MathOutcome final {
    public:
        struct Inactive {};
        struct Active {
            FjDirFrameMath math;
        };
        struct MathematicalFailure {
            FjDirFrameFailure failure;
        };
        struct BoundaryFailure {
            JuicerCuda::Failure failure;
        };
        using Result = std::variant<Inactive, Active, MathematicalFailure, BoundaryFailure>;

        static MathOutcome admit(
            const FjDirFrameInput&, FjStatus, const FjDirFrameMath&, const FjDirFrameFailure&, const FjErrorBuffer&, const std::array<char, 256>&);
        const Result& result() const noexcept;
        const FjDirFrameInput& input() const noexcept;

    private:
        MathOutcome(const FjDirFrameInput&, Result);
        const FjDirFrameInput _input;
        const Result _result;
    };

    MathOutcome prepare_math(
        const DirCouplersRecipe&, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent);
} // namespace JuicerDir
