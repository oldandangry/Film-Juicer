#pragma once

#include <cstdint>
#include "juicer_test_api.h"
#include "Scanner.h"

namespace JuicerTestFixture::Scanner::Hook {
#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        using ColorProjection = void (*)(FjStatus&, FjScannerColor&, FjScannerColorFailure&);
        void set_color_projection(ColorProjection value);
    } // namespace Test
#endif
} // namespace JuicerTestFixture::Scanner::Hook

namespace JuicerTestFixture::Scanner {
    ::Scanner::ColorRuntime build_color_runtime(::Scanner::ScannerMedium medium, const ::Scanner::ScannerIlluminant& illuminant, const ::OutputEncoding::Params& outputEncoding, std::uint64_t activeOutputGamutRecipeHash);
} // namespace JuicerTestFixture::Scanner
