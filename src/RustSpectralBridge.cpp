#include "RustSpectralBridge.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "Scanner.h"
#include "SpectralData.h"
#include "juicer_legacy_api.h"

namespace JuicerSpectral {
    namespace {
        struct Diagnostic {
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        };
        FjFloatSpan span(const std::vector<float>& samples) {
            return {samples.data(), samples.size()};
        }
        FjSpectralObserver observer() {
            return {span(Spectral::gXBar.linear), span(Spectral::gYBar.linear), span(Spectral::gZBar.linear)};
        }
        void require_success(FjStatus status, const Diagnostic& diagnostic, const char* operation) {
            if (status.category != FJ_STATUS_SUCCESS) {
                JuicerCuda::Failure failure;
                JuicerCuda::set_failure(failure, status, std::string(operation) + ": " + std::string(diagnostic.bytes.data(), diagnostic.error.length));
                throw JuicerCuda::ExecutionFailure{std::move(failure)};
            }
        }
        template <std::size_t N>
        void assign(std::vector<float>& out, const float (&samples)[N]) {
            out.assign(samples, samples + N);
        }
    } // namespace

    static_assert(sizeof(std::array<float, 3>) == 12 && alignof(std::array<float, 3>) == alignof(float));
    static_assert(sizeof(std::array<std::array<float, 3>, 81>) == 243 * sizeof(float));
    static_assert(sizeof(FjSpectralTables) == 4264 && alignof(FjSpectralTables) == 8);
    static_assert(sizeof(FjSpectralWhite) == 32 && alignof(FjSpectralWhite) == 8);
    static_assert(sizeof(FjSpectralWhiteFailure) == 16 && alignof(FjSpectralWhiteFailure) == 8);

    void build_tables(const std::array<std::array<float, 3>, 81>& dyesCmy,
                      const std::array<float, 81>& baselineMin,
                      const Spectral::Curve& illuminant,
                      std::uint64_t illuminantHash,
                      Spectral::SpectralTables& out) {
        // The nested fixed arrays have packed scalar layout; no std::pair projection.
        const FjSpectralInput input{{dyesCmy.front().data(), 243}, observer(), span(illuminant.linear), {baselineMin.data(), baselineMin.size()}, {nullptr, 0}, illuminantHash};
        FjSpectralTables tables{};
        Diagnostic diagnostic;
        require_success(fj_legacy_spectral_tables(&input, &tables, &diagnostic.error), diagnostic, "Spectral table preparation");
        // Native vector allocation still throws through the existing publication boundary.
        assign(out.lambda, tables.lambda_nm);
        assign(out.illum, tables.illuminant);
        assign(out.Xbar, tables.observer_xyz[0]);
        assign(out.Ybar, tables.observer_xyz[1]);
        assign(out.Zbar, tables.observer_xyz[2]);
        assign(out.Ax, tables.weighted_xyz[0]);
        assign(out.Ay, tables.weighted_xyz[1]);
        assign(out.Az, tables.weighted_xyz[2]);
        assign(out.epsC, tables.dyes_cmy[0]);
        assign(out.epsM, tables.dyes_cmy[1]);
        assign(out.epsY, tables.dyes_cmy[2]);
        assign(out.baseDensityMin, tables.baseline_min);
        assign(out.baseDensityMid, tables.baseline_mid);
        out.K = 81;
        out.deltaLambda = tables.delta_lambda;
        out.invYn = tables.inv_yn;
        std::copy_n(tables.white_xyz, 3, out.whiteXYZ);
        std::copy_n(tables.reference_white_xyz, 3, out.refIllumWhiteXYZ);
        out.hasBaseline = tables.has_baseline == 1;
        out.densityBaselineMixReference = 0.0f;
        out.illuminantHash = tables.illuminant_hash;
        out.tablesHash = tables.tables_hash;
    }

    std::array<float, 9> s_inverse(const Spectral::SpectralTables& tables) {
        const FjSpectralSInput input{span(tables.Ax), span(tables.Ay), span(tables.Az)};
        FjSpectralInverse out{};
        Diagnostic diagnostic;
        require_success(fj_legacy_spectral_s_inverse(&input, &out, &diagnostic.error), diagnostic, "Film S inverse preparation");
        std::array<float, 9> result{};
        std::copy_n(out.matrix, 9, result.begin());
        return result;
    }

    bool integrate_white(const Spectral::Curve& illuminant, const char* label, Scanner::ScannerIlluminant& out) {
        const FjSpectralWhiteInput input{observer(), span(illuminant.linear)};
        FjSpectralWhite white{};
        FjSpectralWhiteFailure failure{};
        Diagnostic diagnostic;
        const FjStatus status = fj_legacy_spectral_white(&input, &white, &failure, &diagnostic.error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            std::ostringstream message;
            switch (failure.reason) {
                case 1:
                    message << "FATAL: non-finite CMF/SPD sample in " << label << " illuminant";
                    break;
                case 2:
                    message << "FATAL: invalid luminance sum for " << label << " (Yn=" << failure.luminance_sum << ")";
                    break;
                case 3:
                    message << "FATAL: invalid white sum while building scanner illuminant";
                    break;
                case 4:
                    JTRACE("HASH", "FATAL: hash_float_span encountered non-finite sample at index 81");
                    [[fallthrough]];
                case 5:
                    message << "FATAL: failed to hash viewing illuminant for " << label;
                    break;
                default:
                    message << "Scanner white preparation (" << label << "): " << std::string(diagnostic.bytes.data(), diagnostic.error.length);
                    break;
            }
            JTRACE("ILLUM", message.str());
            return false;
        }
        require_success(status, diagnostic, "Scanner white preparation");
        out.normalization = white.normalization;
        std::copy_n(white.white_xyz, 3, out.whiteXYZ);
        std::copy_n(white.white_xy, 2, out.whiteXY);
        out.hash = white.hash;
        return true;
    }
} // namespace JuicerSpectral
