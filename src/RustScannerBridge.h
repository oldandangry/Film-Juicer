#pragma once

#include "juicer_legacy_api.h"

namespace JuicerScanner {
#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        using ColorProjection = void (*)(FjStatus&, FjScannerColor&, FjScannerColorFailure&);
        void set_color_projection(ColorProjection value);
    } // namespace Test
#endif
} // namespace JuicerScanner
