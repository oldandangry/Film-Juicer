#include "RustScannerBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "Scanner.h"

namespace JuicerScanner {
#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        thread_local ColorProjection colorProjection = nullptr;
        void set_color_projection(ColorProjection value) {
            colorProjection = value;
        }
    } // namespace Test
#endif
    namespace {
        [[noreturn]] void fail(FjStatus status, const char* diagnostic) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, diagnostic);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
    } // namespace
} // namespace JuicerScanner

namespace Scanner {
    // SF_TEMP_BRIDGE_SCANNER_COLOR: bind one complete Rust value; remove S4.E.
    ColorRuntime build_color_runtime(ScannerMedium medium, const ScannerIlluminant& illuminant, const OutputEncoding::Params& outputEncoding, std::uint64_t activeOutputGamutRecipeHash) {
        const FjScannerColorInput input{static_cast<std::uint32_t>(medium),
                                        static_cast<std::uint32_t>(outputEncoding.colorSpace),
                                        std::uint32_t(outputEncoding.applyCctfEncoding),
                                        {illuminant.whiteXYZ[0], illuminant.whiteXYZ[1], illuminant.whiteXYZ[2]},
                                        illuminant.hash,
                                        activeOutputGamutRecipeHash};
        FjScannerColor wire{};
        FjScannerColorFailure failure{};
        std::array<char, 512> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjStatus status = fj_legacy_scanner_color_prepare(&input, &wire, &failure, &error);
#if defined(JUICER_SCANNER_TEST_HOOK)
        if (JuicerScanner::Test::colorProjection)
            JuicerScanner::Test::colorProjection(status, wire, failure);
#endif
        const auto invalid = [] {
            JuicerScanner::fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "scanner color projection invalid");
        };
        bool envelope = status.api == FJ_API_NONE && status.native_code == 0;
        switch (failure.reason) {
            case FJ_SCANNER_COLOR_REASON_NONE:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_SUCCESS;
                break;
            case FJ_SCANNER_COLOR_REASON_RAW:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                break;
            case FJ_SCANNER_COLOR_REASON_ILLUMINANT:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_PREPARATION_FAILURE;
                break;
            case FJ_SCANNER_COLOR_REASON_OUTPUT:
                envelope = envelope && failure.stage == 1 && status.category == FJ_STATUS_PREPARATION_FAILURE;
                break;
            case FJ_SCANNER_COLOR_REASON_BOUNDARY:
                envelope = envelope && (failure.stage == 1 || failure.stage == 2) && status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                break;
            case FJ_SCANNER_COLOR_REASON_PANIC:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_INTERNAL_FAILURE;
                break;
            default:
                envelope = false;
                break;
        }
        if (!envelope || error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
            invalid();
        if (status.category != FJ_STATUS_SUCCESS) {
            const std::array<unsigned char, sizeof(FjScannerColor)> zeros{};
            if (std::memcmp(&wire, zeros.data(), zeros.size()) != 0)
                invalid();
            if (failure.reason == FJ_SCANNER_COLOR_REASON_ILLUMINANT) {
                JTRACE("HASH", "FATAL: scanner illuminant hash invalid for color runtime");
                return {};
            }
            if (failure.reason == FJ_SCANNER_COLOR_REASON_OUTPUT) {
                JTRACE("HASH", "FATAL: generated color space hash invalid");
                return {};
            }
            JuicerScanner::fail(status, text.data());
        }
        if (error.length != 0 || wire.output_space != input.output_space || wire.flags != (2u | input.apply_cctf) || wire.active_gamut_recipe_hash != input.active_gamut_recipe_hash || wire.encoding.schema != FJ_OUTPUT_ENCODING_SCHEMA || wire.encoding.cctf_kind > FJ_CCTF_DWG_INTERMEDIATE)
            invalid();
        ColorRuntime out{};
        std::copy_n(wire.cat02, 9, out.cat02);
        std::copy_n(wire.xyz_to_rgb, 9, out.xyzToRgb);
        std::copy_n(wire.illuminant_xyz, 3, out.illuminantXYZ);
        out.encoding = {static_cast<OutputEncoding::ColorSpace>(wire.output_space), (wire.flags & 1u) != 0, true};
        out.preparedEncoding = wire.encoding;
        out.outputGamutRecipeHash = wire.active_gamut_recipe_hash;
        out.hash = wire.hash;
        return out;
    }
} // namespace Scanner
