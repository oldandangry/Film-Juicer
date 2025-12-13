// SpectralMathAVX.cpp
// AVX spectral integrators were removed; this TU remains as a placeholder
// to keep project build lists stable for future SIMD implementations.

#include "SpectralProcessing.h"


#ifdef JUICER_TESTS
#include <limits>
#include "Print.h"
#endif // JUICER_TESTS

namespace Spectral {

#ifdef JUICER_TESTS

namespace {

    bool approx_equal(float a, float b, float eps) {
        return std::fabs(a - b) <= eps;
    }

    bool approx_equal3(const float a[3], const float b[3], float eps) {
        return approx_equal(a[0], b[0], eps) &&
            approx_equal(a[1], b[1], eps) &&
            approx_equal(a[2], b[2], eps);
    }

    Curve make_constant_axis_curve(float v) {
        Curve c;
        assign_reference_axis(c.lambda_nm);
        c.linear.assign(static_cast<size_t>(gShape.K), v);
        return c;
    }

    void dyes_to_XYZ_manual_skip(
        const SpectralTables& T,
        const float dyes_cmy[3],
        bool skipInvalid,
        bool useBaseline,
        float XYZ[3])
    {
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            if (skipInvalid) {
                if (static_cast<size_t>(i) < T.epsValid.size() &&
                    T.epsValid[static_cast<size_t>(i)] == 0) {
                    continue;
                }
                if (useBaseline && T.hasBaseline &&
                    static_cast<size_t>(i) < T.baseMinValid.size() &&
                    T.baseMinValid[static_cast<size_t>(i)] == 0) {
                    continue;
                }
            }

            const float baseSpectral = (useBaseline && T.hasBaseline) ? T.baseMin[i] : 0.0f;
            const float Dlambda = dyes_cmy[0] * T.epsC[i]
                + dyes_cmy[1] * T.epsM[i]
                + dyes_cmy[2] * T.epsY[i]
                + baseSpectral;
            if (!std::isfinite(Dlambda)) {
                continue;
            }
            const float Tlambda = std::exp(-kLn10 * Dlambda);
            X += static_cast<double>(Tlambda) * static_cast<double>(T.Ax[i]);
            Y += static_cast<double>(Tlambda) * static_cast<double>(T.Ay[i]);
            Z += static_cast<double>(Tlambda) * static_cast<double>(T.Az[i]);
        }
        const float s = T.invYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
    }

} // namespace

bool run_followup_fix_tests() {
    bool ok = true;
    constexpr float kEps = 1e-6f;
    const float dyes[3] = { 1.0f, 1.0f, 1.0f };

    Curve xbar = make_constant_axis_curve(1.0f);
    Curve ybar = make_constant_axis_curve(1.0f);
    Curve zbar = make_constant_axis_curve(1.0f);
    Curve illum = make_constant_axis_curve(1.0f);

    // ---------------------------------------------------------------------
    // Test 1: eps missingness must skip contribution (no edge leakage).
    // ---------------------------------------------------------------------
    {
        Curve epsY = make_constant_axis_curve(0.1f);
        Curve epsM = make_constant_axis_curve(0.2f);
        Curve epsC = make_constant_axis_curve(0.3f);

        const int bad = 0;
        epsC.linear[static_cast<size_t>(bad)] = std::numeric_limits<float>::quiet_NaN();

        SpectralTables T;
        Curve baseMin = make_constant_axis_curve(0.0f);
        Curve baseMid = make_constant_axis_curve(0.0f);
        build_tables_from_curves_non_global(
            epsY, epsM, epsC,
            xbar, ybar, zbar,
            illum,
            baseMin, baseMid, /*hasBaseline*/false,
            /*baselineMixReference*/0.0f,
            T);

        float XYZ_impl[3];
        dyes_to_XYZ_given_tables(T, dyes, XYZ_impl);

        float XYZ_skip[3];
        dyes_to_XYZ_manual_skip(T, dyes, /*skipInvalid*/true, /*useBaseline*/false, XYZ_skip);

        float XYZ_leak[3];
        dyes_to_XYZ_manual_skip(T, dyes, /*skipInvalid*/false, /*useBaseline*/false, XYZ_leak);

        if (!approx_equal3(XYZ_impl, XYZ_skip, kEps)) {
            JTRACE("JUICER_TESTS", "FAIL: dyes_to_XYZ_given_tables did not match manual skip");
            ok = false;
        }
        if (approx_equal3(XYZ_impl, XYZ_leak, kEps)) {
            JTRACE("JUICER_TESTS", "FAIL: dyes_to_XYZ_given_tables matched leaky integration");
            ok = false;
        }

        // Print path coverage: negative_T_from_dyes must output 0 at invalid wavelengths.
        WorkingState ws;
        ws.tablesView = T;
        ws.hasBaseline = false;
        ws.baseMin.linear.clear();
        std::vector<float> Tneg;
        const float D0[3] = { 0.0f, 0.0f, 0.0f };
        Print::negative_T_from_dyes(ws, D0, Tneg);
        if (Tneg.size() != static_cast<size_t>(T.K) ||
            Tneg[static_cast<size_t>(bad)] != 0.0f) {
            JTRACE("JUICER_TESTS", "FAIL: negative_T_from_dyes did not zero eps-invalid wavelength");
            ok = false;
        }
    }

    // ---------------------------------------------------------------------
    // Test 2: baseline missingness must skip contribution when enabled.
    // ---------------------------------------------------------------------
    {
        Curve epsY = make_constant_axis_curve(0.1f);
        Curve epsM = make_constant_axis_curve(0.2f);
        Curve epsC = make_constant_axis_curve(0.3f);

        Curve baseMin = make_constant_axis_curve(0.05f);
        Curve baseMid = make_constant_axis_curve(0.0f);
        const int bad = 1;
        baseMin.linear[static_cast<size_t>(bad)] = std::numeric_limits<float>::quiet_NaN();

        SpectralTables T;
        build_tables_from_curves_non_global(
            epsY, epsM, epsC,
            xbar, ybar, zbar,
            illum,
            baseMin, baseMid, /*hasBaseline*/true,
            /*baselineMixReference*/0.0f,
            T);

        float XYZ_impl[3];
        dyes_to_XYZ_with_baseline_given_tables(T, dyes, XYZ_impl);

        float XYZ_skip[3];
        dyes_to_XYZ_manual_skip(T, dyes, /*skipInvalid*/true, /*useBaseline*/true, XYZ_skip);

        float XYZ_leak[3];
        dyes_to_XYZ_manual_skip(T, dyes, /*skipInvalid*/false, /*useBaseline*/true, XYZ_leak);

        if (!approx_equal3(XYZ_impl, XYZ_skip, kEps)) {
            JTRACE("JUICER_TESTS", "FAIL: dyes_to_XYZ_with_baseline_given_tables did not match manual skip");
            ok = false;
        }
        if (approx_equal3(XYZ_impl, XYZ_leak, kEps)) {
            JTRACE("JUICER_TESTS", "FAIL: dyes_to_XYZ_with_baseline_given_tables matched leaky integration");
            ok = false;
        }

        // Print path coverage: baseline-invalid wavelengths must yield 0 transmittance.
        WorkingState ws;
        ws.tablesView = T;
        ws.hasBaseline = true;
        ws.baseMin.linear.assign(static_cast<size_t>(T.K), 0.0f);
        std::vector<float> Tneg;
        const float D0[3] = { 0.0f, 0.0f, 0.0f };
        Print::negative_T_from_dyes(ws, D0, Tneg);
        if (Tneg.size() != static_cast<size_t>(T.K) ||
            Tneg[static_cast<size_t>(bad)] != 0.0f) {
            JTRACE("JUICER_TESTS", "FAIL: negative_T_from_dyes did not zero baseline-invalid wavelength");
            ok = false;
        }
    }

    // ---------------------------------------------------------------------
    // Test 3: tablesHash must change when epsValid changes (mask-aware hashing).
    // ---------------------------------------------------------------------
    {
        Curve epsY_nan = make_constant_axis_curve(0.1f);
        Curve epsM_nan = make_constant_axis_curve(0.2f);
        Curve epsC_nan = make_constant_axis_curve(0.3f);

        Curve epsY_zero = epsY_nan;
        Curve epsM_zero = epsM_nan;
        Curve epsC_zero = epsC_nan;

        const int bad = 2;
        epsC_nan.linear[static_cast<size_t>(bad)] = std::numeric_limits<float>::quiet_NaN();
        epsY_zero.linear[static_cast<size_t>(bad)] = 0.0f;
        epsM_zero.linear[static_cast<size_t>(bad)] = 0.0f;
        epsC_zero.linear[static_cast<size_t>(bad)] = 0.0f;

        Curve baseMin = make_constant_axis_curve(0.0f);
        Curve baseMid = make_constant_axis_curve(0.0f);

        SpectralTables Tnan;
        build_tables_from_curves_non_global(
            epsY_nan, epsM_nan, epsC_nan,
            xbar, ybar, zbar,
            illum,
            baseMin, baseMid, /*hasBaseline*/false,
            /*baselineMixReference*/0.0f,
            Tnan);

        SpectralTables Tzero;
        build_tables_from_curves_non_global(
            epsY_zero, epsM_zero, epsC_zero,
            xbar, ybar, zbar,
            illum,
            baseMin, baseMid, /*hasBaseline*/false,
            /*baselineMixReference*/0.0f,
            Tzero);

        if (Tnan.tablesHash == 0 || Tzero.tablesHash == 0) {
            JTRACE("JUICER_TESTS", "FAIL: tablesHash unexpectedly 0 (hash saw non-finite values)");
            ok = false;
        }
        if (Tnan.tablesHash == Tzero.tablesHash) {
            JTRACE("JUICER_TESTS", "FAIL: tablesHash did not change with epsValid mask changes");
            ok = false;
        }
    }

    if (ok) {
        JTRACE("JUICER_TESTS", "OK: follow-up NaN parity tests passed");
    }

    return ok;
}

#endif // JUICER_TESTS

} // namespace Spectral
