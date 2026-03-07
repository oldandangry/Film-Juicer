// Cuda/Scan/JuicerCudaScanStage.cuh
// Scan-stage device helpers (header-only).
#pragma once

#include "Cuda/JuicerCudaKernelsUtil.cuh"

static __device__ __forceinline__ void scan_spectral_to_log_xyz_device(
    const JuicerCuda::ScanTablesPayload& medium,
    const double D_norm[3],
    double logXYZ[3])
{
    if (!D_norm || !logXYZ) {
        return;
    }
    if (!medium.epsC || !medium.epsM || !medium.epsY || !medium.Ax || !medium.Ay || !medium.Az || medium.K <= 0) {
        logXYZ[0] = logXYZ[1] = logXYZ[2] = nan("");
        return;
    }

    double D_denorm0;
    double D_denorm1;
    double D_denorm2;
    if (medium.mediumIsNegative) {
        D_denorm0 = D_norm[0] / static_cast<double>(medium.inv_max_cmy[0]) - static_cast<double>(medium.min_cmy[0]);
        D_denorm1 = D_norm[1] / static_cast<double>(medium.inv_max_cmy[1]) - static_cast<double>(medium.min_cmy[1]);
        D_denorm2 = D_norm[2] / static_cast<double>(medium.inv_max_cmy[2]) - static_cast<double>(medium.min_cmy[2]);
    }
    else {
        D_denorm0 = D_norm[0] / static_cast<double>(medium.inv_max_cmy[0]);
        D_denorm1 = D_norm[1] / static_cast<double>(medium.inv_max_cmy[1]);
        D_denorm2 = D_norm[2] / static_cast<double>(medium.inv_max_cmy[2]);
    }

    double X = 0.0;
    double Y = 0.0;
    double Z = 0.0;
    for (int i = 0; i < medium.K; ++i) {
        const double baseSpectral = (medium.hasBaseline && medium.baseMin)
            ? static_cast<double>(ldg_f(medium.baseMin + i))
            : 0.0;
        const double Dlambda =
            D_denorm0 * static_cast<double>(ldg_f(medium.epsC + i)) +
            D_denorm1 * static_cast<double>(ldg_f(medium.epsM + i)) +
            D_denorm2 * static_cast<double>(ldg_f(medium.epsY + i)) +
            baseSpectral;

        const double transmittance = pow(10.0, -Dlambda);

        const double ax = static_cast<double>(ldg_f(medium.Ax + i));
        const double ay = static_cast<double>(ldg_f(medium.Ay + i));
        const double az = static_cast<double>(ldg_f(medium.Az + i));

        if (isfinite(ax)) {
            const double out = transmittance * ax;
            if (!isnan(out)) X += out;
        }
        if (isfinite(ay)) {
            const double out = transmittance * ay;
            if (!isnan(out)) Y += out;
        }
        if (isfinite(az)) {
            const double out = transmittance * az;
            if (!isnan(out)) Z += out;
        }
    }

    const double invNormalization = static_cast<double>(medium.invYn);
    const double XYZ0 = X * invNormalization;
    const double XYZ1 = Y * invNormalization;
    const double XYZ2 = Z * invNormalization;

    constexpr double kEps = 1e-10;
    logXYZ[0] = log10(XYZ0 + kEps);
    logXYZ[1] = log10(XYZ1 + kEps);
    logXYZ[2] = log10(XYZ2 + kEps);
}

static __device__ __forceinline__ void scan_log_xyz_device(
    const JuicerCuda::ScanStagePayload& scanStage,
    const double D_norm[3],
    double logXYZ[3])
{
    if (!logXYZ) {
        return;
    }

    const bool D_norm_finite = isfinite(D_norm[0]) && isfinite(D_norm[1]) && isfinite(D_norm[2]);
    if (scanStage.scannerUseLut && scanStage.scanLutLog2XYZ && scanStage.scanLutRes > 0 && D_norm_finite) {
        sample_cubic_scan_lut_device(scanStage.scanLutLog2XYZ, scanStage.scanLutRes, D_norm, logXYZ);
    }
    else {
        scan_spectral_to_log_xyz_device(scanStage.scanTables, D_norm, logXYZ);
    }
}
