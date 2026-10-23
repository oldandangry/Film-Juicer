#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "juicer_legacy_api.h"

struct RenderRecipe;
namespace Spectral {
    struct SpectralTables;
}
namespace Scanner {
    struct MediumView;
    struct ScannerColorCorrectionDescriptor;
    struct PrintCorrectionDerivationInput;
} // namespace Scanner
namespace JuicerCuda {
    struct Failure;
}

namespace JuicerScanner {
    // SF_TEMP_BRIDGE_SCANNER_LUT: scoped immutable host hold; remove S4.E.
    class Lut final {
    public:
        static std::optional<Lut> acquire(const Scanner::MediumView&, std::uint32_t, JuicerCuda::Failure&);
        Lut(Lut&&) noexcept;
        Lut& operator=(Lut&&) noexcept;
        Lut(const Lut&) = delete;
        Lut& operator=(const Lut&) = delete;
        std::span<const float> log2_xyz() const noexcept;
        std::span<const float> slope_c() const noexcept;
        std::span<const float> slope_m() const noexcept;
        std::span<const float> slope_y() const noexcept;
        std::span<const float> cell_min() const noexcept;
        std::span<const float> cell_max() const noexcept;

    private:
        struct Release {
            void operator()(FjScannerLutOwner*) const noexcept;
        };
        Lut(std::unique_ptr<FjScannerLutOwner, Release>, FjScannerLutView) noexcept;
        std::unique_ptr<FjScannerLutOwner, Release> _owner;
        FjScannerLutView _view;
    };
    // SF_TEMP_BRIDGE_SCANNER_CORRECTION: synchronous completed copy; remove S4.E.
    bool prepare_direct(const RenderRecipe&, const Spectral::SpectralTables&, Scanner::ScannerColorCorrectionDescriptor&, FjScannerCorrectionTrace&, std::string&);
    bool prepare_print(const Scanner::PrintCorrectionDerivationInput&,
                       Scanner::ScannerColorCorrectionDescriptor&,
                       FjScannerCorrectionTrace&,
                       std::string&);

#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        using Projection = void (*)(FjStatus&, FjScannerColorCorrection&, FjScannerCorrectionFailure&, FjScannerCorrectionTrace&);
        void set_projection(Projection value);
    } // namespace Test
#endif
} // namespace JuicerScanner
