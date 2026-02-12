#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "AgxNanSemantics.h"
#include "Print.h"
#include "WorkingState.h"

namespace PrintPreflashShared {

inline float blend_dichroic_filter_linear(float curveVal, float normalizedAmount) {
    // agx-emulsion parity: do not treat non-finite curve samples as identity.
    // NaNs must propagate even when amount is 0 (NumPy semantics: NaN * 0 = NaN).
    const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
    return 1.0f - (1.0f - curveVal) * a;
}

inline float compose_dichroic_amount(float neutralAmount, float deltaSteps) {
    const float neutral = std::isfinite(neutralAmount)
        ? std::clamp(neutralAmount, 0.0f, 1.0f)
        : 0.0f;
    float ds = std::isfinite(deltaSteps) ? deltaSteps : 0.0f;
    ds = std::clamp(ds, -Print::kEnlargerSteps, Print::kEnlargerSteps);
    const float totalSteps = neutral * Print::kEnlargerSteps + ds;
    return totalSteps / Print::kEnlargerSteps;
}

inline bool compute_raw(
    const WorkingState& ws,
    const Print::Runtime& prt,
    float outRaw[3],
    int& outShapeK) {
    outShapeK = Spectral::gShape.K;
    if (!outRaw || outShapeK <= 0) {
        return false;
    }

    outRaw[0] = 0.0f;
    outRaw[1] = 0.0f;
    outRaw[2] = 0.0f;

    if (ws.tablesView.K <= 0 || ws.tablesView.K != outShapeK) {
        return false;
    }

    const Print::Profile& p = prt.profile;
    if (static_cast<int>(p.sensC_log.linear.size()) < outShapeK ||
        static_cast<int>(p.sensM_log.linear.size()) < outShapeK ||
        static_cast<int>(p.sensY_log.linear.size()) < outShapeK) {
        return false;
    }

    const bool hasBaseline =
        ws.hasBaseline &&
        static_cast<int>(ws.tablesView.baseMin.size()) == outShapeK;

    const float yAmount = compose_dichroic_amount(prt.neutralY, 0.0f);
    const float mAmount = compose_dichroic_amount(prt.neutralM, 0.0f);
    const float cAmount = compose_dichroic_amount(prt.neutralC, 0.0f);

    double accumC = 0.0;
    double accumM = 0.0;
    double accumY = 0.0;

    for (int i = 0; i < outShapeK; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float Ee = (prt.illumEnlarger.linear.size() > idx)
            ? prt.illumEnlarger.linear[idx]
            : 1.0f;
        const float fY = blend_dichroic_filter_linear(
            (prt.filterY.linear.size() > idx) ? prt.filterY.linear[idx] : 1.0f,
            yAmount);
        const float fM = blend_dichroic_filter_linear(
            (prt.filterM.linear.size() > idx) ? prt.filterM.linear[idx] : 1.0f,
            mAmount);
        const float fC = blend_dichroic_filter_linear(
            (prt.filterC.linear.size() > idx) ? prt.filterC.linear[idx] : 1.0f,
            cAmount);
        const float illumFiltered = Ee * (fY * fM * fC);

        const float baseDensity = hasBaseline ? ws.tablesView.baseMin[idx] : 0.0f;
        const double light = static_cast<double>(density_to_light_sample_agx(baseDensity, illumFiltered));

        const float sC = p.sensC_log.linear[idx];
        const float sM = p.sensM_log.linear[idx];
        const float sY = p.sensY_log.linear[idx];
        if (!std::isnan(sC)) {
            accumC += light * static_cast<double>(sC);
        }
        if (!std::isnan(sM)) {
            accumM += light * static_cast<double>(sM);
        }
        if (!std::isnan(sY)) {
            accumY += light * static_cast<double>(sY);
        }
    }

    outRaw[0] = static_cast<float>(accumC);
    outRaw[1] = static_cast<float>(accumM);
    outRaw[2] = static_cast<float>(accumY);
    return std::isfinite(outRaw[0]) && std::isfinite(outRaw[1]) && std::isfinite(outRaw[2]);
}

} // namespace PrintPreflashShared

