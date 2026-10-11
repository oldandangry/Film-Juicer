#include "RustGamutBridge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <new>
#include <string>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "SpectralData.h"
#include "juicer_legacy_api.h"

namespace JuicerGamut {
#if defined(JUICER_INPUT_HULL_TEST_HOOK)
    namespace Test {
        void before_build(Gamut::InputCompressionHull& out);
    }
#endif
    namespace {
        bool has_reference_samples(const Spectral::Curve& curve) {
            return curve.linear.size() == 81 && curve.lambda_nm.size() == 81;
        }
    } // namespace
    static_assert(sizeof(FjInputHullSource) == 64 && alignof(FjInputHullSource) == 8);
    static_assert(offsetof(FjInputHullSource, x) == 0 && offsetof(FjInputHullSource, y) == 16);
    static_assert(offsetof(FjInputHullSource, z) == 32 && offsetof(FjInputHullSource, d65) == 48);
    static_assert(sizeof(FjInputHull) == 8216 && alignof(FjInputHull) == 8);
    static_assert(offsetof(FjInputHull, center_xy) == 0 && offsetof(FjInputHull, xy) == 8 && offsetof(FjInputHull, hash) == 8208);

    bool input_hull(const Spectral::Curve& xBar,
                    const Spectral::Curve& yBar,
                    const Spectral::Curve& zBar,
                    const Spectral::Curve& d65,
                    Gamut::InputCompressionHull& out) {
        out = {};
        if (!has_reference_samples(xBar) || !has_reference_samples(yBar) ||
            !has_reference_samples(zBar) || !has_reference_samples(d65)) {
            out.diagnostic = "MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65";
            return false;
        }
        const FjInputHullSource source{{xBar.linear.data(), xBar.linear.size()},
                                       {yBar.linear.data(), yBar.linear.size()},
                                       {zBar.linear.data(), zBar.linear.size()},
                                       {d65.linear.data(), d65.linear.size()}};
        FjInputHull result{};
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
#if defined(JUICER_INPUT_HULL_TEST_HOOK)
        Test::before_build(out);
#endif
        const auto status = fj_legacy_gamut_input_hull(&source, &result, &error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            out.diagnostic.assign(bytes.data(), error.length);
            return false;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "Input hull preparation: " + std::string(bytes.data(), error.length));
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        std::copy_n(result.center_xy, 2, out.center.begin());
        for (std::size_t i = 0; i < out.xy.size(); ++i) {
            std::copy_n(result.xy + i * 2, 2, out.xy[i].begin());
        }
        out.hash = result.hash;
        out.valid = true;
        return true;
    }
} // namespace JuicerGamut
