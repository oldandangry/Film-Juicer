#pragma once

#include <string>

#include "juicer_legacy_api.h"

struct RenderRecipe;
namespace Spectral {
    struct SpectralTables;
}
namespace Scanner {
    struct ScannerColorCorrectionDescriptor;
    struct PrintCorrectionDerivationInput;
} // namespace Scanner

namespace JuicerScanner {
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
