#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "juicer_legacy_api.h"

namespace JuicerCuda {
    struct Failure;
}
namespace Spektrafilm {
    struct DirGaussianComponentPlan;
}

namespace JuicerDirBoundary {
    // SF_TEMP_BRIDGE_DIR_BOUNDARY: literal model projection. S4.E narrows final
    // execution inputs; remove at the last native model caller, no later than S6.D.
    FjDirBoundaryInput project_input(const Spektrafilm::DirGaussianComponentPlan&, std::array<int, 2> dimensions);
    bool describe(const FjDirBoundaryInput&, std::uint64_t& key, JuicerCuda::Failure&);

    class Boundary final {
    public:
        static std::optional<Boundary> acquire(const FjDirBoundaryInput&, std::uint64_t key, JuicerCuda::Failure&);
        Boundary(Boundary&&) noexcept;
        Boundary& operator=(Boundary&&) noexcept;
        Boundary(const Boundary&) = delete;
        Boundary& operator=(const Boundary&) = delete;
        FjDirBoundaryView view() const noexcept;

    private:
        struct Release {
            void operator()(FjDirBoundaryOwner*) const noexcept;
        };
        Boundary(std::unique_ptr<FjDirBoundaryOwner, Release>, FjDirBoundaryView) noexcept;
        std::unique_ptr<FjDirBoundaryOwner, Release> _owner;
        FjDirBoundaryView _view;
    };
} // namespace JuicerDirBoundary
