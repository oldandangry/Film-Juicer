// Cuda/JuicerCudaValidation.cu
// CUDA validation probes (guarded).
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#if defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)

namespace {

    __global__ void probe_density_curve_kernel(
        const float* JUICER_RESTRICT x,
        const float* JUICER_RESTRICT y,
        int n,
        float gammaFactor,
        const float* JUICER_RESTRICT logE,
        int m,
        float* out) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m) {
            out[idx] = sample_density_at_logE_device(x, y, n, logE[idx], gammaFactor);
        }
    }

    __global__ void probe_density_curve_sanitize_inf_kernel(
        const float* JUICER_RESTRICT x,
        const float* JUICER_RESTRICT y,
        int n,
        float gammaFactor,
        const float* JUICER_RESTRICT logE,
        int m,
        float* out) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m) {
            const float le = sanitize_inf_logE_for_curve_device(logE[idx], x, n);
            out[idx] = sample_density_at_logE_device(x, y, n, le, gammaFactor);
        }
    }

    __global__ void probe_convert_input_to_DWG_kernel(
        const float* inRgb,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        int applyInputChromaticAdapt,
        const float* inputRGBToXYZ,
        const float* inputXYZAdapt,
        float* outRgbDWG) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        float linear[3];
        apply_input_cctf_decoding_device(inputColorSpaceIndex, applyCctfDecoding, inRgb, linear);

        float XYZ[3];
        mat3_mul9_device(inputRGBToXYZ, linear, XYZ);

        const float* xyzPtr = XYZ;
        float adapted[3];
        if (applyInputChromaticAdapt) {
            mat3_mul9_device(inputXYZAdapt, XYZ, adapted);
            xyzPtr = adapted;
        }

        float dwg[3];
        XYZ_to_DWG_linear_device(xyzPtr, dwg);
        outRgbDWG[0] = dwg[0];
        outRgbDWG[1] = dwg[1];
        outRgbDWG[2] = dwg[2];
    }

    __device__ void hanatos_linear_spectrum_scaled_device(
        const float rgbDWG[3],
        const float* JUICER_RESTRICT hanatosLut,
        int N,
        int K,
        const float refIllumWhiteXYZ[3],
        float* outEe /* K */) {
        // Convert DWG RGB to XYZ (D65).
        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f, 0.14877482f, 0.10105872f, 0.27411851f, 0.87363190f, -0.14775041f, -0.09896291f, -0.13789533f, 1.32591599f};

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
            DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
            DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]};
        XYZ[0] = device_sanitize_channel(XYZ[0]);
        XYZ[1] = device_sanitize_channel(XYZ[1]);
        XYZ[2] = device_sanitize_channel(XYZ[2]);

        const float D65[3] = {0.950455f, 1.0f, 1.089058f};

        float refWhite[3] = {
            device_sanitize_channel(refIllumWhiteXYZ[0]),
            device_sanitize_channel(refIllumWhiteXYZ[1]),
            device_sanitize_channel(refIllumWhiteXYZ[2])};
        if (!(refWhite[1] > 0.0f)) {
            refWhite[0] = D65[0];
            refWhite[1] = D65[1];
            refWhite[2] = D65[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = device_sanitize_channel(adaptedXYZ[0]);
        adaptedXYZ[1] = device_sanitize_channel(adaptedXYZ[1]);
        adaptedXYZ[2] = device_sanitize_channel(adaptedXYZ[2]);

        const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float denom = fmaxf(sumXYZ, 1e-10f);

        float x = adaptedXYZ[0] / denom;
        float y = adaptedXYZ[1] / denom;
        x = fminf(1.0f, fmaxf(0.0f, x));
        y = fminf(1.0f, fmaxf(0.0f, y));

        float qx, qy;
        tri2quad_device(x, y, qx, qy);

        const float fx = fminf(1.0f, fmaxf(0.0f, qx)) * static_cast<float>(N - 1);
        const float fy = fminf(1.0f, fmaxf(0.0f, qy)) * static_cast<float>(N - 1);
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        if (x0 < 0)
            x0 = 0;
        if (y0 < 0)
            y0 = 0;
        if (x0 > N - 1)
            x0 = N - 1;
        if (y0 > N - 1)
            y0 = N - 1;
        const int x1 = (x0 + 1 <= N - 1) ? (x0 + 1) : (N - 1);
        const int y1 = (y0 + 1 <= N - 1) ? (y0 + 1) : (N - 1);
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        for (int k = 0; k < K; ++k) {
            const float v00 = hanatos_bilinear_at(hanatosLut, N, K, x0, y0, k);
            const float v10 = hanatos_bilinear_at(hanatosLut, N, K, x1, y0, k);
            const float v01 = hanatos_bilinear_at(hanatosLut, N, K, x0, y1, k);
            const float v11 = hanatos_bilinear_at(hanatosLut, N, K, x1, y1, k);
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            const float raw = v0 * (1.0f - ty) + v1 * ty;
            outEe[k] = device_sanitize_channel(sumXYZ * raw);
        }
    }

    __global__ void probe_hanatos_layer_exposures_kernel(
        const float* JUICER_RESTRICT hanatosLut,
        int N,
        int K,
        float exposureScale,
        int applyDeltaLambda,
        float rgbDWG0,
        float rgbDWG1,
        float rgbDWG2,
        float refWhite0,
        float refWhite1,
        float refWhite2,
        const float* JUICER_RESTRICT sensB,
        const float* JUICER_RESTRICT sensG,
        const float* JUICER_RESTRICT sensR,
        float* outE3) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        float rgb[3] = {rgbDWG0, rgbDWG1, rgbDWG2};
        float refWhite[3] = {refWhite0, refWhite1, refWhite2};

        // Small fixed K (81). Keep on stack for the probe path.
        float Ee[81];
        hanatos_linear_spectrum_scaled_device(rgb, hanatosLut, N, K, refWhite, Ee);

        double Eb = 0.0;
        double Eg = 0.0;
        double Er = 0.0;
        for (int i = 0; i < K; ++i) {
            const float e = Ee[i];
            if (!device_isfinite(e)) {
                continue;
            }
            const float sb = sensB ? ldg_f(sensB + i) : 0.0f;
            const float sg = sensG ? ldg_f(sensG + i) : 0.0f;
            const float sr = sensR ? ldg_f(sensR + i) : 0.0f;
            const double e64 = static_cast<double>(e);
            if (device_isfinite(sb))
                Eb += e64 * static_cast<double>(sb);
            if (device_isfinite(sg))
                Eg += e64 * static_cast<double>(sg);
            if (device_isfinite(sr))
                Er += e64 * static_cast<double>(sr);
        }

        const float safeScale = fmaxf(0.0f, exposureScale);
        const double delta = applyDeltaLambda ? 5.0 : 1.0;
        const double dl = delta * static_cast<double>(safeScale);
        outE3[0] = device_isfinite(static_cast<float>(Eb * dl)) ? static_cast<float>(Eb * dl) : 0.0f;
        outE3[1] = device_isfinite(static_cast<float>(Eg * dl)) ? static_cast<float>(Eg * dl) : 0.0f;
        outE3[2] = device_isfinite(static_cast<float>(Er * dl)) ? static_cast<float>(Er * dl) : 0.0f;
    }

    struct TablesProbeParams {
        float S_inv[9];
        float refIllumWhiteXYZ[3];
    };

    __device__ __forceinline__ float sanitize_component_device(float v) {
        if (!isfinite(v)) {
            return 0.0f;
        }
        return fmaxf(0.0f, v);
    }

    __global__ void probe_tables_layer_exposures_kernel(
        TablesProbeParams params,
        const float* Ax,
        const float* Ay,
        const float* Az,
        int K,
        const float* sensB,
        const float* sensG,
        const float* sensR,
        float exposureScale,
        int applyDeltaLambda,
        float rgbDWG0,
        float rgbDWG1,
        float rgbDWG2,
        float* outE3) {
        if (!Ax || !Ay || !Az || !sensB || !sensG || !sensR || !outE3 || K <= 0) {
            return;
        }

        // Convert DWG RGB to XYZ (D65).
        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f, 0.14877482f, 0.10105872f, 0.27411851f, 0.87363190f, -0.14775041f, -0.09896291f, -0.13789533f, 1.32591599f};

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG0 + DWG_RGB_to_XYZ[1] * rgbDWG1 + DWG_RGB_to_XYZ[2] * rgbDWG2,
            DWG_RGB_to_XYZ[3] * rgbDWG0 + DWG_RGB_to_XYZ[4] * rgbDWG1 + DWG_RGB_to_XYZ[5] * rgbDWG2,
            DWG_RGB_to_XYZ[6] * rgbDWG0 + DWG_RGB_to_XYZ[7] * rgbDWG1 + DWG_RGB_to_XYZ[8] * rgbDWG2};

        float sanitizedXYZ[3] = {
            sanitize_component_device(XYZ[0]),
            sanitize_component_device(XYZ[1]),
            sanitize_component_device(XYZ[2])};

        const float D65[3] = {0.950455f, 1.0f, 1.089058f};

        float refWhite[3] = {
            sanitize_component_device(params.refIllumWhiteXYZ[0]),
            sanitize_component_device(params.refIllumWhiteXYZ[1]),
            sanitize_component_device(params.refIllumWhiteXYZ[2])};
        if (!(refWhite[1] > 0.0f)) {
            refWhite[0] = D65[0];
            refWhite[1] = D65[1];
            refWhite[2] = D65[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02_device(sanitizedXYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = fmaxf(0.0f, adaptedXYZ[0]);
        adaptedXYZ[1] = fmaxf(0.0f, adaptedXYZ[1]);
        adaptedXYZ[2] = fmaxf(0.0f, adaptedXYZ[2]);

        const float targetScale = (adaptedXYZ[1] > 0.0f) ? adaptedXYZ[1] : sanitizedXYZ[1];

        float cx =
            params.S_inv[0] * adaptedXYZ[0] +
            params.S_inv[1] * adaptedXYZ[1] +
            params.S_inv[2] * adaptedXYZ[2];
        float cy =
            params.S_inv[3] * adaptedXYZ[0] +
            params.S_inv[4] * adaptedXYZ[1] +
            params.S_inv[5] * adaptedXYZ[2];
        float cz =
            params.S_inv[6] * adaptedXYZ[0] +
            params.S_inv[7] * adaptedXYZ[1] +
            params.S_inv[8] * adaptedXYZ[2];
        cx = fmaxf(0.0f, cx);
        cy = fmaxf(0.0f, cy);
        cz = fmaxf(0.0f, cz);

        double Y_recon = 0.0;
        double Eb = 0.0;
        double Eg = 0.0;
        double Er = 0.0;

        for (int i = 0; i < K; ++i) {
            const float bx = fmaxf(0.0f, sanitize_component_device(Ax[i]));
            const float by = fmaxf(0.0f, sanitize_component_device(Ay[i]));
            const float bz = fmaxf(0.0f, sanitize_component_device(Az[i]));
            const float Ei = fmaxf(1e-6f, cx * bx + cy * by + cz * bz);

            Y_recon += static_cast<double>(Ei) * static_cast<double>(by);

            const float sb = sensB[i];
            const float sg = sensG[i];
            const float sr = sensR[i];
            const double e64 = static_cast<double>(Ei);
            if (isfinite(sb))
                Eb += e64 * static_cast<double>(sb);
            if (isfinite(sg))
                Eg += e64 * static_cast<double>(sg);
            if (isfinite(sr))
                Er += e64 * static_cast<double>(sr);
        }

        if (Y_recon > 1e-20 && targetScale > 0.0f) {
            const double s = static_cast<double>(targetScale) / Y_recon;
            Eb *= s;
            Eg *= s;
            Er *= s;
        } else if (!(targetScale > 0.0f)) {
            Eb = Eg = Er = 0.0;
        }

        const float safeScale = fmaxf(0.0f, exposureScale);
        const double delta = applyDeltaLambda ? 5.0 : 1.0;
        const double dl = delta * static_cast<double>(safeScale);

        outE3[0] = fmaxf(0.0f, static_cast<float>(Eb * dl));
        outE3[1] = fmaxf(0.0f, static_cast<float>(Eg * dl));
        outE3[2] = fmaxf(0.0f, static_cast<float>(Er * dl));
    }

    __global__ void probe_film_log_raw_kernel(const float* filmRaw3, float* outLogRaw3) {
        if (!filmRaw3 || !outLogRaw3) {
            return;
        }
        constexpr float kEps = 1e-10f;
        const float a = filmRaw3[0];
        const float b = filmRaw3[1];
        const float c = filmRaw3[2];
        outLogRaw3[0] = log10f(fmaxf(a, 0.0f) + kEps);
        outLogRaw3[1] = log10f(fmaxf(b, 0.0f) + kEps);
        outLogRaw3[2] = log10f(fmaxf(c, 0.0f) + kEps);
    }

    __global__ void probe_scan_spectral_to_log_xyz_kernel(
        double d0,
        double d1,
        double d2,
        int mediumIsNegative,
        float min0,
        float min1,
        float min2,
        float invMax0,
        float invMax1,
        float invMax2,
        const float* epsC,
        const float* epsM,
        const float* epsY,
        const float* Ax,
        const float* Ay,
        const float* Az,
        const float* baseDensityMin,
        int K,
        int hasBaseline,
        float invYn,
        double* outLogXYZ3) {
        if (!outLogXYZ3 || !epsC || !epsM || !epsY || !Ax || !Ay || !Az || K <= 0) {
            return;
        }

        double D_denorm0;
        double D_denorm1;
        double D_denorm2;
        if (mediumIsNegative) {
            D_denorm0 = d0 / static_cast<double>(invMax0) - static_cast<double>(min0);
            D_denorm1 = d1 / static_cast<double>(invMax1) - static_cast<double>(min1);
            D_denorm2 = d2 / static_cast<double>(invMax2) - static_cast<double>(min2);
        } else {
            D_denorm0 = d0 / static_cast<double>(invMax0);
            D_denorm1 = d1 / static_cast<double>(invMax1);
            D_denorm2 = d2 / static_cast<double>(invMax2);
        }

        double X = 0.0;
        double Y = 0.0;
        double Z = 0.0;
        for (int i = 0; i < K; ++i) {
            const double baseSpectral = (hasBaseline && baseDensityMin) ? static_cast<double>(baseDensityMin[i]) : 0.0;
            const double Dlambda =
                D_denorm0 * static_cast<double>(epsC[i]) +
                D_denorm1 * static_cast<double>(epsM[i]) +
                D_denorm2 * static_cast<double>(epsY[i]) +
                baseSpectral;

            const double transmittance = pow(10.0, -Dlambda);

            const double ax = static_cast<double>(Ax[i]);
            const double ay = static_cast<double>(Ay[i]);
            const double az = static_cast<double>(Az[i]);

            if (isfinite(ax)) {
                const double out = transmittance * ax;
                if (!isnan(out))
                    X += out;
            }
            if (isfinite(ay)) {
                const double out = transmittance * ay;
                if (!isnan(out))
                    Y += out;
            }
            if (isfinite(az)) {
                const double out = transmittance * az;
                if (!isnan(out))
                    Z += out;
            }
        }

        const double invNormalization = static_cast<double>(invYn);
        const double XYZ0 = X * invNormalization;
        const double XYZ1 = Y * invNormalization;
        const double XYZ2 = Z * invNormalization;

        constexpr double kEps = 1e-10;
        outLogXYZ3[0] = log10(XYZ0 + kEps);
        outLogXYZ3[1] = log10(XYZ1 + kEps);
        outLogXYZ3[2] = log10(XYZ2 + kEps);
    }

    __global__ void probe_clamp_logE_to_curve_domain_kernel(
        const float* x,
        int n,
        float logE,
        float* outLogE) {
        if (!x || n <= 0 || !outLogE) {
            return;
        }

        // Match Couplers::apply_runtime_logE_with_curves clamp_to:
        // - xmin/xmax are taken as curve.lambda_nm.front/back (no finite-domain scan)
        // - non-finite logE => xmin
        const float xmin = x[0];
        const float xmax = x[n - 1];

        if (!isfinite(logE)) {
            outLogE[0] = xmin;
            return;
        }

        float v = logE;
        v = fmaxf(v, xmin);
        v = fminf(v, xmax);
        outLogE[0] = v;
    }

    __global__ void probe_print_pipeline_kernel(
        JuicerCuda::PipelineRunParams params,
        const float* inNegCmy,
        float* outPrintCmy,
        int count) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        if (!inNegCmy || !outPrintCmy) {
            return;
        }
        float D_cmy[3] = {
            inNegCmy[idx * 3 + 0],
            inNegCmy[idx * 3 + 1],
            inNegCmy[idx * 3 + 2]};
        apply_print_pipeline_device(params.printExpose, params.printDevelop, D_cmy);
        outPrintCmy[idx * 3 + 0] = D_cmy[0];
        outPrintCmy[idx * 3 + 1] = D_cmy[1];
        outPrintCmy[idx * 3 + 2] = D_cmy[2];
    }

} // namespace

extern "C" cudaError_t juicer_cuda_probe_density_curve(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque) {
    if (!dX || !dY || n <= 0 || !hLogE || m <= 0 || !hOut) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dLogE = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLogE), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(dLogE, hLogE, static_cast<size_t>(m) * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    const int threads = 128;
    const int blocks = (m + threads - 1) / threads;
    probe_density_curve_kernel<<<blocks, threads, 0, stream>>>(dX, dY, n, gammaFactor, dLogE, m, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(hOut, dOut, static_cast<size_t>(m) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dLogE);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_density_curve_sanitize_inf(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque) {
    if (!dX || !dY || n <= 0 || !hLogE || m <= 0 || !hOut) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dLogE = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLogE), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), static_cast<size_t>(m) * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(dLogE, hLogE, static_cast<size_t>(m) * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    const int threads = 128;
    const int blocks = (m + threads - 1) / threads;
    probe_density_curve_sanitize_inf_kernel<<<blocks, threads, 0, stream>>>(dX, dY, n, gammaFactor, dLogE, m, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaMemcpyAsync(hOut, dOut, static_cast<size_t>(m) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dLogE);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dLogE);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_hanatos_layer_exposures(
    const float rgbDWG[3],
    const float* dHanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    int K,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque) {
    if (!rgbDWG || !dHanatosLut || hanatosN <= 0 || !refIllumWhiteXYZ || !outE3) {
        return cudaErrorInvalidValue;
    }
    if (K != 81) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }

    probe_hanatos_layer_exposures_kernel<<<1, 1, 0, stream>>>(
        dHanatosLut,
        hanatosN,
        K,
        exposureScale,
        applyDeltaLambda,
        rgbDWG[0],
        rgbDWG[1],
        rgbDWG[2],
        refIllumWhiteXYZ[0],
        refIllumWhiteXYZ[1],
        refIllumWhiteXYZ[2],
        dSensB,
        dSensG,
        dSensR,
        dOut);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaMemcpyAsync(outE3, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_tables_layer_exposures(
    const float rgbDWG[3],
    const float S_inv9[9],
    const float refIllumWhiteXYZ[3],
    const float* dAx,
    const float* dAy,
    const float* dAz,
    int K,
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque) {
    if (!rgbDWG || !S_inv9 || !refIllumWhiteXYZ || !dAx || !dAy || !dAz || !dSensB || !dSensG || !dSensR || !outE3 || K <= 0) {
        return cudaErrorInvalidValue;
    }
    if (K != 81) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    TablesProbeParams params{};
    for (int i = 0; i < 9; ++i)
        params.S_inv[i] = S_inv9[i];
    for (int i = 0; i < 3; ++i)
        params.refIllumWhiteXYZ[i] = refIllumWhiteXYZ[i];

    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }

    probe_tables_layer_exposures_kernel<<<1, 1, 0, stream>>>(
        params,
        dAx,
        dAy,
        dAz,
        K,
        dSensB,
        dSensG,
        dSensR,
        exposureScale,
        applyDeltaLambda,
        rgbDWG[0],
        rgbDWG[1],
        rgbDWG[2],
        dOut);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaMemcpyAsync(outE3, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_film_log_raw(
    const float filmRaw3[3],
    float outLogRaw3[3],
    void* cudaStreamOpaque) {
    if (!filmRaw3 || !outLogRaw3) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dIn = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIn), 3 * sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(dIn, filmRaw3, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    probe_film_log_raw_kernel<<<1, 1, 0, stream>>>(dIn, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(outLogRaw3, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dIn);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_scan_spectral_to_log_xyz(
    const double D_norm3[3],
    int mediumIsNegative,
    const float min_cmy[3],
    const float inv_max_cmy[3],
    const float* dEpsC,
    const float* dEpsM,
    const float* dEpsY,
    const float* dAx,
    const float* dAy,
    const float* dAz,
    const float* dBaseMin,
    int K,
    int hasBaseline,
    float invYn,
    double outLogXYZ3[3],
    void* cudaStreamOpaque) {
    if (!D_norm3 || !min_cmy || !inv_max_cmy || !dEpsC || !dEpsM || !dEpsY || !dAx || !dAy || !dAz || !outLogXYZ3) {
        return cudaErrorInvalidValue;
    }
    if (K != 81) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    double* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(double));
    if (err != cudaSuccess) {
        return err;
    }

    probe_scan_spectral_to_log_xyz_kernel<<<1, 1, 0, stream>>>(
        D_norm3[0],
        D_norm3[1],
        D_norm3[2],
        mediumIsNegative ? 1 : 0,
        min_cmy[0],
        min_cmy[1],
        min_cmy[2],
        inv_max_cmy[0],
        inv_max_cmy[1],
        inv_max_cmy[2],
        dEpsC,
        dEpsM,
        dEpsY,
        dAx,
        dAy,
        dAz,
        dBaseMin,
        K,
        hasBaseline ? 1 : 0,
        invYn,
        dOut);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaMemcpyAsync(outLogXYZ3, dOut, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_clamp_logE_to_curve_domain(
    const float* dX,
    int n,
    float logE,
    float* outLogE,
    void* cudaStreamOpaque) {
    if (!dX || n <= 0 || !outLogE) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dOut), sizeof(float));
    if (err != cudaSuccess) {
        return err;
    }

    probe_clamp_logE_to_curve_domain_kernel<<<1, 1, 0, stream>>>(dX, n, logE, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaMemcpyAsync(outLogE, dOut, sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_convert_input_to_DWG(
    const float rgbIn[3],
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    int applyInputChromaticAdapt,
    const float inputRGBToXYZ9[9],
    const float inputXYZAdapt9[9],
    float outRgbDWG[3],
    void* cudaStreamOpaque) {
    if (!rgbIn || !inputRGBToXYZ9 || !inputXYZAdapt9 || !outRgbDWG) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dIn = nullptr;
    float* dMatA = nullptr;
    float* dMatB = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIn), 3 * sizeof(float));
    if (err != cudaSuccess)
        return err;
    err = cudaMalloc(reinterpret_cast<void**>(&dMatA), 9 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dIn);
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dMatB), 9 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(dIn, rgbIn, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }
    err = cudaMemcpyAsync(dMatA, inputRGBToXYZ9, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }
    err = cudaMemcpyAsync(dMatB, inputXYZAdapt9, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }

    probe_convert_input_to_DWG_kernel<<<1, 1, 0, stream>>>(dIn, inputColorSpaceIndex, applyCctfDecoding, applyInputChromaticAdapt, dMatA, dMatB, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(outRgbDWG, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dMatB);
        cudaFree(dMatA);
        cudaFree(dIn);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dMatB);
    cudaFree(dMatA);
    cudaFree(dIn);
    return err;
}

extern "C" cudaError_t juicer_cuda_probe_print_pipeline(
    const float* hNegCmy,
    int count,
    const JuicerCuda::PipelineRunParams* hParams,
    float* hOutPrintCmy,
    void* cudaStreamOpaque) {
    if (!hNegCmy || count <= 0 || !hParams || !hOutPrintCmy) {
        return cudaErrorInvalidValue;
    }
    if (!hParams->printExpose.active) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dIn = nullptr;
    float* dOut = nullptr;
    const size_t bytes = static_cast<size_t>(count) * 3u * sizeof(float);

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIn), bytes);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&dOut), bytes);
    if (err != cudaSuccess) {
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(dIn, hNegCmy, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    probe_print_pipeline_kernel<<<blocks, threads, 0, stream>>>(*hParams, dIn, dOut, count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    err = cudaMemcpyAsync(hOutPrintCmy, dOut, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(dOut);
        cudaFree(dIn);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dIn);
    return err;
}

#endif // JUICER_CUDA_VALIDATE_PRIMITIVES
