// Cuda/JuicerCudaPrimitives.cu
//
// Phase 2: core CUDA math primitives with CPU-parity semantics.
//
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "Cuda/JuicerCudaPayloads.h"

namespace {

    __device__ __forceinline__ bool device_isfinite(float v) {
        return isfinite(v);
    }

    __device__ __forceinline__ float device_sanitize_channel(float v) {
        return device_isfinite(v) ? v : 0.0f;
    }

    __device__ __forceinline__ float device_sanitize_nonneg(float v) {
        return device_isfinite(v) ? fmaxf(0.0f, v) : 0.0f;
    }

    __device__ float sample_density_at_logE_device(const float* x, const float* y, int n, float logE, float gammaFactor) {
        if (!x || !y || n <= 0) {
            return 0.0f;
        }

        // Preserve NaN query behavior (matches CPU).
        if (!device_isfinite(logE)) {
            return nanf("");
        }

        const float gammaSafe = (device_isfinite(gammaFactor) && gammaFactor > 0.0f) ? gammaFactor : 1.0f;
        const float xq = logE * gammaSafe;

        int domainBegin = 0;
        while (domainBegin < n && !device_isfinite(x[domainBegin])) {
            ++domainBegin;
        }
        if (domainBegin >= n) {
            return 0.0f;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !device_isfinite(x[domainEnd])) {
            --domainEnd;
        }

        const float xmin = x[domainBegin];
        const float xmax = x[domainEnd];
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return y[domainBegin];
        }

        if (xq <= xmin) {
            return y[domainBegin];
        }
        if (xq >= xmax) {
            return y[domainEnd];
        }

        int i1 = domainBegin + 1;
        while (i1 <= domainEnd && x[i1] < xq) {
            ++i1;
        }
        if (i1 > domainEnd) {
            return y[domainEnd];
        }

        const int i0 = i1 - 1;
        const float x0 = x[i0];
        const float x1 = x[i1];
        const float y0 = y[i0];
        const float y1 = y[i1];

        const float denom = x1 - x0;
        if (!(denom > 0.0f) || !device_isfinite(denom)) {
            return y0;
        }

        const float t = (xq - x0) / denom;
        return y0 + t * (y1 - y0);
    }

    __device__ float sanitize_inf_logE_for_curve_device(float logE, const float* x, int n) {
        if (device_isfinite(logE) || isnan(logE)) {
            return logE;
        }
        if (!x || n <= 0) {
            return logE;
        }

        int begin = 0;
        while (begin < n && !device_isfinite(x[begin])) {
            ++begin;
        }
        if (begin >= n) {
            return logE;
        }
        int end = n - 1;
        while (end > begin && !device_isfinite(x[end])) {
            --end;
        }
        const float xmin = x[begin];
        const float xmax = x[end];
        if (!device_isfinite(xmin) || !device_isfinite(xmax) || !(xmax >= xmin)) {
            return logE;
        }
        return (logE > 0.0f) ? xmax : xmin;
    }

    __global__ void probe_density_curve_kernel(
        const float* x,
        const float* y,
        int n,
        float gammaFactor,
        const float* logE,
        int m,
        float* out)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m) {
            out[idx] = sample_density_at_logE_device(x, y, n, logE[idx], gammaFactor);
        }
    }

    __global__ void probe_density_curve_sanitize_inf_kernel(
        const float* x,
        const float* y,
        int n,
        float gammaFactor,
        const float* logE,
        int m,
        float* out)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m) {
            const float le = sanitize_inf_logE_for_curve_device(logE[idx], x, n);
            out[idx] = sample_density_at_logE_device(x, y, n, le, gammaFactor);
        }
    }

    struct Mat3 {
        float m[9];
    };

    __device__ __forceinline__ void mul3(const float m[9], const float v[3], float dst[3]) {
        dst[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        dst[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        dst[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    }

    __device__ void chromatic_adapt_XYZ_CAT02_device(
        const float XYZ[3],
        const float srcWhiteXYZ[3],
        const float dstWhiteXYZ[3],
        float outXYZ[3])
    {
        // Matches Spectral::chromatic_adapt_XYZ_CAT02 in ColorTransforms.h
        const float M[9] = {
             0.7328000f,  0.4296000f, -0.1624000f,
            -0.7036000f,  1.6975000f,  0.0061000f,
             0.0030000f,  0.0136000f,  0.9834000f
        };
        const float M_inv[9] = {
            1.0961238f, -0.2788690f,  0.1827452f,
            0.4543690f,  0.4735332f,  0.0720978f,
           -0.0096276f, -0.0056980f,  1.0153256f
        };

        float srcWhite[3] = {
            device_sanitize_nonneg(srcWhiteXYZ[0]),
            device_sanitize_nonneg(srcWhiteXYZ[1]),
            device_sanitize_nonneg(srcWhiteXYZ[2])
        };
        float dstWhite[3] = {
            device_sanitize_nonneg(dstWhiteXYZ[0]),
            device_sanitize_nonneg(dstWhiteXYZ[1]),
            device_sanitize_nonneg(dstWhiteXYZ[2])
        };

        const float srcY = (srcWhite[1] > 0.0f) ? srcWhite[1] : 1.0f;
        const float dstY = (dstWhite[1] > 0.0f) ? dstWhite[1] : 1.0f;
        const float srcScale = 1.0f / srcY;
        const float dstScale = 1.0f / dstY;
        srcWhite[0] *= srcScale; srcWhite[1] = 1.0f; srcWhite[2] *= srcScale;
        dstWhite[0] *= dstScale; dstWhite[1] = 1.0f; dstWhite[2] *= dstScale;

        float srcLMS[3];
        float dstLMS[3];
        float XYZ_LMS[3];
        mul3(M, srcWhite, srcLMS);
        mul3(M, dstWhite, dstLMS);
        mul3(M, XYZ, XYZ_LMS);

        const float scale0 = (srcLMS[0] > 1e-6f) ? (dstLMS[0] / srcLMS[0]) : 1.0f;
        const float scale1 = (srcLMS[1] > 1e-6f) ? (dstLMS[1] / srcLMS[1]) : 1.0f;
        const float scale2 = (srcLMS[2] > 1e-6f) ? (dstLMS[2] / srcLMS[2]) : 1.0f;

        float adaptedLMS[3] = {
            scale0 * XYZ_LMS[0],
            scale1 * XYZ_LMS[1],
            scale2 * XYZ_LMS[2]
        };
        mul3(M_inv, adaptedLMS, outXYZ);
    }

    __device__ __forceinline__ float decode_bt2020_channel_device(float v) {
        const float x = fmaxf(0.0f, device_sanitize_channel(v));
        constexpr float a = 1.09929681f;
        constexpr float threshold = 0.0812428791f;
        if (x < threshold) {
            return x / 4.5f;
        }
        return powf((x + (a - 1.0f)) / a, 1.0f / 0.45f);
    }

    __device__ __forceinline__ void apply_input_cctf_decoding_device(
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        const float inRgb[3],
        float outRgb[3])
    {
        if (!applyCctfDecoding) {
            outRgb[0] = device_sanitize_channel(inRgb[0]);
            outRgb[1] = device_sanitize_channel(inRgb[1]);
            outRgb[2] = device_sanitize_channel(inRgb[2]);
            return;
        }

        // InputColorSpace enum: 0=DWG, 1=BT2020, 2=ACES2065-1.
        if (inputColorSpaceIndex == 1) {
            outRgb[0] = decode_bt2020_channel_device(inRgb[0]);
            outRgb[1] = decode_bt2020_channel_device(inRgb[1]);
            outRgb[2] = decode_bt2020_channel_device(inRgb[2]);
            return;
        }

        outRgb[0] = device_sanitize_channel(inRgb[0]);
        outRgb[1] = device_sanitize_channel(inRgb[1]);
        outRgb[2] = device_sanitize_channel(inRgb[2]);
    }

    __device__ __forceinline__ void mat3_mul9_device(const float m[9], const float v[3], float out[3]) {
        out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    }

    __device__ __forceinline__ void XYZ_to_DWG_linear_device(const float XYZ[3], float RGB[3]) {
        const float DWG_XYZ_to_RGB[9] = {
            1.51667204f, -0.28147805f, -0.14696363f,
           -0.46491710f,  1.25142378f,  0.17488461f,
            0.07578536f,  0.08076209f,  0.76034476f
        };
        float out[3];
        mat3_mul9_device(DWG_XYZ_to_RGB, XYZ, out);
        for (int i = 0; i < 3; ++i) {
            float v = out[i];
            if (!device_isfinite(v) || v < 0.0f) {
                v = 0.0f;
            }
            RGB[i] = v;
        }
    }

    __global__ void probe_convert_input_to_DWG_kernel(
        const float* inRgb,
        int inputColorSpaceIndex,
        int applyCctfDecoding,
        int applyInputChromaticAdapt,
        const float* inputRGBToXYZ,
        const float* inputXYZAdapt,
        float* outRgbDWG)
    {
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

    __device__ __forceinline__ void tri2quad_device(float tx, float ty, float& qx, float& qy) {
        // Matches Spectral::tri2quad in SpectralProcessing.h
        const float denom = fmaxf(1.0f - tx, 1e-10f);
        float x = (1.0f - tx);
        x = x * x;
        float y = ty / denom;
        qx = fminf(1.0f, fmaxf(0.0f, x));
        qy = fminf(1.0f, fmaxf(0.0f, y));
    }

    __device__ __forceinline__ float hanatos_bilinear_at(
        const float* lut,
        int N,
        int K,
        int x,
        int y,
        int k)
    {
        const std::size_t idx = (static_cast<std::size_t>(x) * static_cast<std::size_t>(N) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k);
        return lut[idx];
    }

    __device__ void hanatos_linear_spectrum_scaled_device(
        const float rgbDWG[3],
        const float* hanatosLut,
        int N,
        int K,
        const float refIllumWhiteXYZ[3],
        float* outEe /* K */)
    {
        // Convert DWG RGB to XYZ (D65).
        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f,  0.14877482f,  0.10105872f,
            0.27411851f,  0.87363190f, -0.14775041f,
           -0.09896291f, -0.13789533f,  1.32591599f
        };

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
            DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
            DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
        };
        XYZ[0] = device_sanitize_nonneg(XYZ[0]);
        XYZ[1] = device_sanitize_nonneg(XYZ[1]);
        XYZ[2] = device_sanitize_nonneg(XYZ[2]);

        const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

        float refWhite[3] = {
            device_sanitize_nonneg(refIllumWhiteXYZ[0]),
            device_sanitize_nonneg(refIllumWhiteXYZ[1]),
            device_sanitize_nonneg(refIllumWhiteXYZ[2])
        };
        if (!(refWhite[1] > 0.0f)) {
            refWhite[0] = D65[0];
            refWhite[1] = D65[1];
            refWhite[2] = D65[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = device_sanitize_nonneg(adaptedXYZ[0]);
        adaptedXYZ[1] = device_sanitize_nonneg(adaptedXYZ[1]);
        adaptedXYZ[2] = device_sanitize_nonneg(adaptedXYZ[2]);

        const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float safeSum = (sumXYZ > 0.0f) ? sumXYZ : 0.0f;
        if (!(safeSum > 0.0f)) {
            for (int k = 0; k < K; ++k) {
                outEe[k] = 0.0f;
            }
            return;
        }

        float x = 1.0f / 3.0f;
        float y = 1.0f / 3.0f;
        if (sumXYZ > 1e-12f) {
            x = adaptedXYZ[0] / sumXYZ;
            y = adaptedXYZ[1] / sumXYZ;
        }
        x = fminf(1.0f, fmaxf(0.0f, x));
        y = fminf(1.0f, fmaxf(0.0f, y));

        float qx, qy;
        tri2quad_device(x, y, qx, qy);

        const float fx = fminf(1.0f, fmaxf(0.0f, qx)) * static_cast<float>(N - 1);
        const float fy = fminf(1.0f, fmaxf(0.0f, qy)) * static_cast<float>(N - 1);
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x0 > N - 1) x0 = N - 1;
        if (y0 > N - 1) y0 = N - 1;
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
            outEe[k] = fmaxf(0.0f, safeSum * raw);
        }
    }

    __global__ void probe_hanatos_layer_exposures_kernel(
        const float* hanatosLut,
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
        const float* sensB,
        const float* sensG,
        const float* sensR,
        float* outE3)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }

        float rgb[3] = { rgbDWG0, rgbDWG1, rgbDWG2 };
        float refWhite[3] = { refWhite0, refWhite1, refWhite2 };

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
            const float sb = sensB ? sensB[i] : 0.0f;
            const float sg = sensG ? sensG[i] : 0.0f;
            const float sr = sensR ? sensR[i] : 0.0f;
            const double e64 = static_cast<double>(e);
            if (device_isfinite(sb)) Eb += e64 * static_cast<double>(sb);
            if (device_isfinite(sg)) Eg += e64 * static_cast<double>(sg);
            if (device_isfinite(sr)) Er += e64 * static_cast<double>(sr);
        }

        const float safeScale = fmaxf(0.0f, exposureScale);
        const double delta = applyDeltaLambda ? 5.0 : 1.0;
        const double dl = delta * static_cast<double>(safeScale);
        outE3[0] = fmaxf(0.0f, static_cast<float>(Eb * dl));
        outE3[1] = fmaxf(0.0f, static_cast<float>(Eg * dl));
        outE3[2] = fmaxf(0.0f, static_cast<float>(Er * dl));
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
    void* cudaStreamOpaque)
{
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
    void* cudaStreamOpaque)
{
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
    void* cudaStreamOpaque)
{
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

namespace {

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
        float* outE3)
    {
        if (!Ax || !Ay || !Az || !sensB || !sensG || !sensR || !outE3 || K <= 0) {
            return;
        }

        // Convert DWG RGB to XYZ (D65).
        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f,  0.14877482f,  0.10105872f,
            0.27411851f,  0.87363190f, -0.14775041f,
           -0.09896291f, -0.13789533f,  1.32591599f
        };

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG0 + DWG_RGB_to_XYZ[1] * rgbDWG1 + DWG_RGB_to_XYZ[2] * rgbDWG2,
            DWG_RGB_to_XYZ[3] * rgbDWG0 + DWG_RGB_to_XYZ[4] * rgbDWG1 + DWG_RGB_to_XYZ[5] * rgbDWG2,
            DWG_RGB_to_XYZ[6] * rgbDWG0 + DWG_RGB_to_XYZ[7] * rgbDWG1 + DWG_RGB_to_XYZ[8] * rgbDWG2
        };

        float sanitizedXYZ[3] = {
            sanitize_component_device(XYZ[0]),
            sanitize_component_device(XYZ[1]),
            sanitize_component_device(XYZ[2])
        };

        const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

        float refWhite[3] = {
            sanitize_component_device(params.refIllumWhiteXYZ[0]),
            sanitize_component_device(params.refIllumWhiteXYZ[1]),
            sanitize_component_device(params.refIllumWhiteXYZ[2])
        };
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
            if (isfinite(sb)) Eb += e64 * static_cast<double>(sb);
            if (isfinite(sg)) Eg += e64 * static_cast<double>(sg);
            if (isfinite(sr)) Er += e64 * static_cast<double>(sr);
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

} // namespace

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
    void* cudaStreamOpaque)
{
    if (!rgbDWG || !S_inv9 || !refIllumWhiteXYZ || !dAx || !dAy || !dAz || !dSensB || !dSensG || !dSensR || !outE3 || K <= 0) {
        return cudaErrorInvalidValue;
    }
    if (K != 81) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    TablesProbeParams params{};
    for (int i = 0; i < 9; ++i) params.S_inv[i] = S_inv9[i];
    for (int i = 0; i < 3; ++i) params.refIllumWhiteXYZ[i] = refIllumWhiteXYZ[i];

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

namespace {

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

} // namespace

extern "C" cudaError_t juicer_cuda_probe_film_log_raw(
    const float filmRaw3[3],
    float outLogRaw3[3],
    void* cudaStreamOpaque)
{
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

namespace {

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
        const float* baseMin,
        int K,
        int hasBaseline,
        float invYn,
        double* outLogXYZ3)
    {
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
            const double baseSpectral = (hasBaseline && baseMin) ? static_cast<double>(baseMin[i]) : 0.0;
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

        const double invNormalization = static_cast<double>(invYn);
        const double XYZ0 = X * invNormalization;
        const double XYZ1 = Y * invNormalization;
        const double XYZ2 = Z * invNormalization;

        constexpr double kEps = 1e-10;
        outLogXYZ3[0] = log10(XYZ0 + kEps);
        outLogXYZ3[1] = log10(XYZ1 + kEps);
        outLogXYZ3[2] = log10(XYZ2 + kEps);
    }

} // namespace

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
    void* cudaStreamOpaque)
{
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

namespace {

    __global__ void probe_clamp_logE_to_curve_domain_kernel(
        const float* x,
        int n,
        float logE,
        float* outLogE)
    {
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

} // namespace

extern "C" cudaError_t juicer_cuda_probe_clamp_logE_to_curve_domain(
    const float* dX,
    int n,
    float logE,
    float* outLogE,
    void* cudaStreamOpaque)
{
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
    void* cudaStreamOpaque)
{
    if (!rgbIn || !inputRGBToXYZ9 || !inputXYZAdapt9 || !outRgbDWG) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    float* dIn = nullptr;
    float* dMatA = nullptr;
    float* dMatB = nullptr;
    float* dOut = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIn), 3 * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(reinterpret_cast<void**>(&dMatA), 9 * sizeof(float));
    if (err != cudaSuccess) { cudaFree(dIn); return err; }
    err = cudaMalloc(reinterpret_cast<void**>(&dMatB), 9 * sizeof(float));
    if (err != cudaSuccess) { cudaFree(dMatA); cudaFree(dIn); return err; }
    err = cudaMalloc(reinterpret_cast<void**>(&dOut), 3 * sizeof(float));
    if (err != cudaSuccess) { cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }

    err = cudaMemcpyAsync(dIn, rgbIn, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(dOut); cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }
    err = cudaMemcpyAsync(dMatA, inputRGBToXYZ9, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(dOut); cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }
    err = cudaMemcpyAsync(dMatB, inputXYZAdapt9, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(dOut); cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }

    probe_convert_input_to_DWG_kernel<<<1, 1, 0, stream>>>(dIn, inputColorSpaceIndex, applyCctfDecoding, applyInputChromaticAdapt, dMatA, dMatB, dOut);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(dOut); cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }

    err = cudaMemcpyAsync(outRgbDWG, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { cudaFree(dOut); cudaFree(dMatB); cudaFree(dMatA); cudaFree(dIn); return err; }

    err = cudaStreamSynchronize(stream);
    cudaFree(dOut);
    cudaFree(dMatB);
    cudaFree(dMatA);
    cudaFree(dIn);
    return err;
}

namespace {

    __device__ __forceinline__ double clamp01d_device(double v) {
        if (v <= 0.0) return 0.0;
        if (v >= 1.0) return 1.0;
        return v;
    }

    __device__ __forceinline__ double encode_sRGBd_device(double v) {
        if (v <= 0.0031308) {
            return 12.92 * v;
        }
        return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    }

    __device__ __forceinline__ double encode_gammad_device(double v, double exponent) {
        return pow(v, exponent);
    }

    __device__ __forceinline__ double encode_BT2020d_device(double v, double a, double b) {
        if (v < b) {
            return v * 4.5;
        }
        return a * pow(v, 0.45) - (a - 1.0);
    }

    __device__ __forceinline__ double encode_ProPhotod_device(double v, double threshold, double exponent) {
        if (v < threshold) {
            return v * 16.0;
        }
        return pow(v, exponent);
    }

    __device__ __forceinline__ double encode_DaVinciIntermediated_device(double v, const JuicerCuda::CctfPayload& cctf) {
        const double linear = fmax(0.0, v);
        if (linear <= static_cast<double>(cctf.linearCutoff)) {
            return linear * static_cast<double>(cctf.d);
        }
        // (log2(linear + a) + b) * c
        return (log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
    }

    __device__ __forceinline__ double encode_channel_double_device(const JuicerCuda::CctfPayload& cctf, double v) {
        switch (cctf.kind) {
        case 0: // Linear
            return v;
        case 1: // Gamma
            return encode_gammad_device(v, static_cast<double>(cctf.gamma));
        case 2: // SRGB
            return encode_sRGBd_device(v);
        case 3: // BT2020
            return encode_BT2020d_device(v, static_cast<double>(cctf.a), static_cast<double>(cctf.b));
        case 4: // ProPhoto
            return encode_ProPhotod_device(v, static_cast<double>(cctf.linearCutoff), static_cast<double>(cctf.gamma));
        case 5: // DaVinciIntermediate
            return encode_DaVinciIntermediated_device(v, cctf);
        default:
            return clamp01d_device(v);
        }
    }

    __device__ __forceinline__ void apply_output_encoding_device(const JuicerCuda::OutputEncodingPayload& enc, double rgb[3]) {
        if (!rgb) {
            return;
        }

        if (enc.preserveLinearRange) {
            return; // no encoding, no clamp
        }

        if (enc.applyCctfEncoding) {
            rgb[0] = encode_channel_double_device(enc.cctf, rgb[0]);
            rgb[1] = encode_channel_double_device(enc.cctf, rgb[1]);
            rgb[2] = encode_channel_double_device(enc.cctf, rgb[2]);
        }

        // agx parity: encode first, then clip
        rgb[0] = clamp01d_device(rgb[0]);
        rgb[1] = clamp01d_device(rgb[1]);
        rgb[2] = clamp01d_device(rgb[2]);
    }

    __device__ __forceinline__ void mat3_mul_vec_double_device(const float m9[9], const double v3[3], double out3[3]) {
        out3[0] =
            static_cast<double>(m9[0]) * v3[0] +
            static_cast<double>(m9[1]) * v3[1] +
            static_cast<double>(m9[2]) * v3[2];
        out3[1] =
            static_cast<double>(m9[3]) * v3[0] +
            static_cast<double>(m9[4]) * v3[1] +
            static_cast<double>(m9[5]) * v3[2];
        out3[2] =
            static_cast<double>(m9[6]) * v3[0] +
            static_cast<double>(m9[7]) * v3[1] +
            static_cast<double>(m9[8]) * v3[2];
    }

    __device__ __forceinline__ double mitchell_weight_device(double t) {
        const double B = 1.0 / 3.0;
        const double C = 1.0 / 3.0;
        const double x = fabs(t);
        if (x < 1.0) {
            return (1.0 / 6.0) * ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
                + (-18.0 + 12.0 * B + 6.0 * C) * x * x
                + (6.0 - 2.0 * B));
        }
        else if (x < 2.0) {
            return (1.0 / 6.0) * ((-B - 6.0 * C) * x * x * x
                + (6.0 * B + 30.0 * C) * x * x
                + (-12.0 * B - 48.0 * C) * x
                + (8.0 * B + 24.0 * C));
        }
        return 0.0;
    }

    __device__ __forceinline__ int reflect_index_device(int idx, int size) {
        if (size <= 1) return 0;
        if (idx < 0) return -idx;
        if (idx >= size) return 2 * (size - 1) - idx;
        return idx;
    }

    __device__ __forceinline__ double scan_lut_fetch_device(const double* lut, int res, size_t limit, int xi, int yi, int zi, int c) {
        const size_t sRes = static_cast<size_t>(res);
        const size_t idx = (static_cast<size_t>(zi) * sRes + static_cast<size_t>(yi)) * sRes + static_cast<size_t>(xi);
        const size_t base = idx * 3u + static_cast<size_t>(c);
        if (base >= limit) {
            return 0.0;
        }
        return lut[base];
    }

    __device__ __forceinline__ int reflect_index_repeat_device(int idx, int size) {
        if (size <= 1) {
            return 0;
        }
        while (idx < 0 || idx >= size) {
            if (idx < 0) {
                idx = -idx;
            }
            else {
                idx = 2 * size - idx - 2;
            }
        }
        return idx;
    }

    __device__ __forceinline__ void sample_cubic_scan_lut_device(const double* lut, int resRaw, const double D_norm[3], double out[3]) {
        if (!out) {
            return;
        }
        if (!lut || !D_norm) {
            out[0] = out[1] = out[2] = 0.0;
            return;
        }

        const int res = (resRaw > 1) ? resRaw : 1;
        const double scale = (res > 1) ? static_cast<double>(res - 1) : 1.0;
        const double fx = D_norm[0] * scale;
        const double fy = D_norm[1] * scale;
        const double fz = D_norm[2] * scale;

        const int xBase = static_cast<int>(floor(fx));
        const int yBase = static_cast<int>(floor(fy));
        const int zBase = static_cast<int>(floor(fz));
        const double tx = fx - static_cast<double>(xBase);
        const double ty = fy - static_cast<double>(yBase);
        const double tz = fz - static_cast<double>(zBase);

        double wx[4], wy[4], wz[4];
        wx[0] = mitchell_weight_device(tx + 1.0);
        wx[1] = mitchell_weight_device(tx);
        wx[2] = mitchell_weight_device(tx - 1.0);
        wx[3] = mitchell_weight_device(tx - 2.0);
        wy[0] = mitchell_weight_device(ty + 1.0);
        wy[1] = mitchell_weight_device(ty);
        wy[2] = mitchell_weight_device(ty - 1.0);
        wy[3] = mitchell_weight_device(ty - 2.0);
        wz[0] = mitchell_weight_device(tz + 1.0);
        wz[1] = mitchell_weight_device(tz);
        wz[2] = mitchell_weight_device(tz - 1.0);
        wz[3] = mitchell_weight_device(tz - 2.0);

        const size_t sRes = static_cast<size_t>(res);
        const size_t limit = sRes * sRes * sRes * 3u;

        double sum[3] = { 0.0, 0.0, 0.0 };
        double wsum = 0.0;
        for (int i = 0; i < 4; ++i) {
            const int xi = reflect_index_device(xBase - 1 + i, res);
            for (int j = 0; j < 4; ++j) {
                const int yj = reflect_index_device(yBase - 1 + j, res);
                for (int k = 0; k < 4; ++k) {
                    const int zk = reflect_index_device(zBase - 1 + k, res);
                    const double w = wx[i] * wy[j] * wz[k];
                    wsum += w;
                    sum[0] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 0);
                    sum[1] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 1);
                    sum[2] += w * scan_lut_fetch_device(lut, res, limit, xi, yj, zk, 2);
                }
            }
        }

        const double inv = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        out[0] = sum[0] * inv;
        out[1] = sum[1] * inv;
        out[2] = sum[2] * inv;
    }

    __device__ __forceinline__ void scan_spectral_to_log_xyz_device(
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
            const double baseSpectral = (medium.hasBaseline && medium.baseMin) ? static_cast<double>(medium.baseMin[i]) : 0.0;
            const double Dlambda =
                D_denorm0 * static_cast<double>(medium.epsC[i]) +
                D_denorm1 * static_cast<double>(medium.epsM[i]) +
                D_denorm2 * static_cast<double>(medium.epsY[i]) +
                baseSpectral;

            const double transmittance = pow(10.0, -Dlambda);

            const double ax = static_cast<double>(medium.Ax[i]);
            const double ay = static_cast<double>(medium.Ay[i]);
            const double az = static_cast<double>(medium.Az[i]);

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

    __device__ __forceinline__ void convert_input_to_DWG_device(
        const JuicerCuda::FilmRawPayload& cfg,
        const float rgbIn[3],
        float rgbDWG[3])
    {
        float linear[3];
        apply_input_cctf_decoding_device(cfg.inputColorSpaceIndex, cfg.applyCctfDecoding, rgbIn, linear);

        float XYZ[3];
        mat3_mul9_device(cfg.inputRGBToXYZ, linear, XYZ);

        const float* xyzPtr = XYZ;
        float adapted[3];
        if (cfg.applyInputChromaticAdapt) {
            mat3_mul9_device(cfg.inputXYZAdapt, XYZ, adapted);
            xyzPtr = adapted;
        }

        float dwg[3];
        XYZ_to_DWG_linear_device(xyzPtr, dwg);
        rgbDWG[0] = dwg[0];
        rgbDWG[1] = dwg[1];
        rgbDWG[2] = dwg[2];
    }

    __device__ void hanatos_layer_exposures_device(
        const float rgbDWG[3],
        const float* hanatosLut,
        int hanatosN,
        const float refIllumWhiteXYZ[3],
        const float* sensB,
        const float* sensG,
        const float* sensR,
        float E_out[3])
    {
        if (!E_out) {
            return;
        }
        if (!hanatosLut || hanatosN <= 0 || !sensB || !sensG || !sensR) {
            E_out[0] = E_out[1] = E_out[2] = 0.0f;
            return;
        }

        // Convert DWG RGB to XYZ (D65).
        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f,  0.14877482f,  0.10105872f,
            0.27411851f,  0.87363190f, -0.14775041f,
           -0.09896291f, -0.13789533f,  1.32591599f
        };

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
            DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
            DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
        };
        XYZ[0] = device_sanitize_nonneg(XYZ[0]);
        XYZ[1] = device_sanitize_nonneg(XYZ[1]);
        XYZ[2] = device_sanitize_nonneg(XYZ[2]);

        const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

        float refWhite[3] = {
            device_sanitize_nonneg(refIllumWhiteXYZ[0]),
            device_sanitize_nonneg(refIllumWhiteXYZ[1]),
            device_sanitize_nonneg(refIllumWhiteXYZ[2])
        };
        if (!(refWhite[1] > 0.0f)) {
            refWhite[0] = D65[0];
            refWhite[1] = D65[1];
            refWhite[2] = D65[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02_device(XYZ, D65, refWhite, adaptedXYZ);
        adaptedXYZ[0] = device_sanitize_nonneg(adaptedXYZ[0]);
        adaptedXYZ[1] = device_sanitize_nonneg(adaptedXYZ[1]);
        adaptedXYZ[2] = device_sanitize_nonneg(adaptedXYZ[2]);

        const float sumXYZ = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
        const float safeSum = (sumXYZ > 0.0f) ? sumXYZ : 0.0f;
        if (!(safeSum > 0.0f)) {
            E_out[0] = E_out[1] = E_out[2] = 0.0f;
            return;
        }

        float x = 1.0f / 3.0f;
        float y = 1.0f / 3.0f;
        if (sumXYZ > 1e-12f) {
            x = adaptedXYZ[0] / sumXYZ;
            y = adaptedXYZ[1] / sumXYZ;
        }
        x = fminf(1.0f, fmaxf(0.0f, x));
        y = fminf(1.0f, fmaxf(0.0f, y));

        float qx, qy;
        tri2quad_device(x, y, qx, qy);

        const int N = hanatosN;
        const int K = 81;
        const float fx = fminf(1.0f, fmaxf(0.0f, qx)) * static_cast<float>(N - 1);
        const float fy = fminf(1.0f, fmaxf(0.0f, qy)) * static_cast<float>(N - 1);
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x0 > N - 1) x0 = N - 1;
        if (y0 > N - 1) y0 = N - 1;
        const int x1 = (x0 + 1 <= N - 1) ? (x0 + 1) : (N - 1);
        const int y1 = (y0 + 1 <= N - 1) ? (y0 + 1) : (N - 1);
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

        double Eb = 0.0;
        double Eg = 0.0;
        double Er = 0.0;
        for (int k = 0; k < K; ++k) {
            const float v00 = hanatos_bilinear_at(hanatosLut, N, K, x0, y0, k);
            const float v10 = hanatos_bilinear_at(hanatosLut, N, K, x1, y0, k);
            const float v01 = hanatos_bilinear_at(hanatosLut, N, K, x0, y1, k);
            const float v11 = hanatos_bilinear_at(hanatosLut, N, K, x1, y1, k);
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            const float raw = v0 * (1.0f - ty) + v1 * ty;

            const float e = fmaxf(0.0f, safeSum * raw);
            if (!device_isfinite(e)) {
                continue;
            }
            const double e64 = static_cast<double>(e);

            const float sb = sensB[k];
            const float sg = sensG[k];
            const float sr = sensR[k];
            if (isfinite(sb)) Eb += e64 * static_cast<double>(sb);
            if (isfinite(sg)) Eg += e64 * static_cast<double>(sg);
            if (isfinite(sr)) Er += e64 * static_cast<double>(sr);
        }

        E_out[0] = fmaxf(0.0f, static_cast<float>(Eb));
        E_out[1] = fmaxf(0.0f, static_cast<float>(Eg));
        E_out[2] = fmaxf(0.0f, static_cast<float>(Er));
    }

    __device__ void tables_layer_exposures_device(
        const float rgbDWG[3],
        const float S_inv[9],
        const float refIllumWhiteXYZ[3],
        const float* Ax,
        const float* Ay,
        const float* Az,
        const float* sensB,
        const float* sensG,
        const float* sensR,
        float E_out[3])
    {
        if (!E_out) {
            return;
        }
        if (!rgbDWG || !S_inv || !refIllumWhiteXYZ || !Ax || !Ay || !Az || !sensB || !sensG || !sensR) {
            E_out[0] = E_out[1] = E_out[2] = 0.0f;
            return;
        }

        const float DWG_RGB_to_XYZ[9] = {
            0.70062239f,  0.14877482f,  0.10105872f,
            0.27411851f,  0.87363190f, -0.14775041f,
           -0.09896291f, -0.13789533f,  1.32591599f
        };

        float XYZ[3] = {
            DWG_RGB_to_XYZ[0] * rgbDWG[0] + DWG_RGB_to_XYZ[1] * rgbDWG[1] + DWG_RGB_to_XYZ[2] * rgbDWG[2],
            DWG_RGB_to_XYZ[3] * rgbDWG[0] + DWG_RGB_to_XYZ[4] * rgbDWG[1] + DWG_RGB_to_XYZ[5] * rgbDWG[2],
            DWG_RGB_to_XYZ[6] * rgbDWG[0] + DWG_RGB_to_XYZ[7] * rgbDWG[1] + DWG_RGB_to_XYZ[8] * rgbDWG[2]
        };

        float sanitizedXYZ[3] = {
            device_sanitize_nonneg(XYZ[0]),
            device_sanitize_nonneg(XYZ[1]),
            device_sanitize_nonneg(XYZ[2])
        };

        const float D65[3] = { 0.950455f, 1.0f, 1.089058f };

        float refWhite[3] = {
            device_sanitize_nonneg(refIllumWhiteXYZ[0]),
            device_sanitize_nonneg(refIllumWhiteXYZ[1]),
            device_sanitize_nonneg(refIllumWhiteXYZ[2])
        };
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
            S_inv[0] * adaptedXYZ[0] +
            S_inv[1] * adaptedXYZ[1] +
            S_inv[2] * adaptedXYZ[2];
        float cy =
            S_inv[3] * adaptedXYZ[0] +
            S_inv[4] * adaptedXYZ[1] +
            S_inv[5] * adaptedXYZ[2];
        float cz =
            S_inv[6] * adaptedXYZ[0] +
            S_inv[7] * adaptedXYZ[1] +
            S_inv[8] * adaptedXYZ[2];
        cx = fmaxf(0.0f, cx);
        cy = fmaxf(0.0f, cy);
        cz = fmaxf(0.0f, cz);

        double Y_recon = 0.0;
        double Eb = 0.0;
        double Eg = 0.0;
        double Er = 0.0;

        const int K = 81;
        for (int i = 0; i < K; ++i) {
            const float bx = fmaxf(0.0f, device_sanitize_nonneg(Ax[i]));
            const float by = fmaxf(0.0f, device_sanitize_nonneg(Ay[i]));
            const float bz = fmaxf(0.0f, device_sanitize_nonneg(Az[i]));
            const float Ei = fmaxf(1e-6f, cx * bx + cy * by + cz * bz);

            Y_recon += static_cast<double>(Ei) * static_cast<double>(by);

            const float sb = sensB[i];
            const float sg = sensG[i];
            const float sr = sensR[i];
            const double e64 = static_cast<double>(Ei);
            if (isfinite(sb)) Eb += e64 * static_cast<double>(sb);
            if (isfinite(sg)) Eg += e64 * static_cast<double>(sg);
            if (isfinite(sr)) Er += e64 * static_cast<double>(sr);
        }

        if (Y_recon > 1e-20 && targetScale > 0.0f) {
            const double s = static_cast<double>(targetScale) / Y_recon;
            Eb *= s;
            Eg *= s;
            Er *= s;
        }
        else if (!(targetScale > 0.0f)) {
            Eb = Eg = Er = 0.0;
        }

        // Mallett path uses applyDeltaLambda=true (Δλ=5nm) and exposureScale=1.0 for raw film exposure.
        const double dl = 5.0;
        E_out[0] = fmaxf(0.0f, static_cast<float>(Eb * dl));
        E_out[1] = fmaxf(0.0f, static_cast<float>(Eg * dl));
        E_out[2] = fmaxf(0.0f, static_cast<float>(Er * dl));
    }

    __device__ __forceinline__ float clamp_to_curve_domain_device(float logE, const JuicerCuda::DeviceCurveView& c) {
        if (!c.x || c.n <= 0) {
            return logE;
        }
        const float xmin = c.x[0];
        const float xmax = c.x[c.n - 1];
        if (!isfinite(logE)) {
            return xmin;
        }
        float v = logE;
        v = fmaxf(v, xmin);
        v = fminf(v, xmax);
        return v;
    }

    __device__ __forceinline__ void apply_dir_runtime_logE_device(
        float logE_BGR[3],
        const float layerD_BGR[3],
        const JuicerCuda::DirPayload& dir,
        const JuicerCuda::DeviceCurveView& densB,
        const JuicerCuda::DeviceCurveView& densG,
        const JuicerCuda::DeviceCurveView& densR)
    {
        if (!logE_BGR || !layerD_BGR) {
            return;
        }
        if (!dir.active) {
            return;
        }

        auto safe_norm = [](float D, float dmax) -> float {
            float Din = (!isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f;
            float n = Din / m;
            if (!isfinite(n) || n < 0.0f) n = 0.0f;
            return n;
        };

        float nB = safe_norm(layerD_BGR[0], dir.dMax[0]);
        float nG = safe_norm(layerD_BGR[1], dir.dMax[1]);
        float nR = safe_norm(layerD_BGR[2], dir.dMax[2]);

        auto high_boost = [&](float n) -> float {
            const float nb = n + dir.highShift * n * n;
            if (!isfinite(nb)) {
                return (n >= 0.0f && isfinite(n)) ? n : 0.0f;
            }
            return fmaxf(0.0f, nb);
        };
        nB = high_boost(nB);
        nG = high_boost(nG);
        nR = high_boost(nR);

        float aY = dir.M[0] * nB + dir.M[3] * nG + dir.M[6] * nR;
        float aM = dir.M[1] * nB + dir.M[4] * nG + dir.M[7] * nR;
        float aC = dir.M[2] * nB + dir.M[5] * nG + dir.M[8] * nR;

        if (!isfinite(aY)) aY = 0.0f;
        if (!isfinite(aM)) aM = 0.0f;
        if (!isfinite(aC)) aC = 0.0f;

        auto clamp_corr = [](float v) -> float {
            if (!isfinite(v)) return 0.0f;
            if (v < -10.0f) return -10.0f;
            if (v > 10.0f) return 10.0f;
            return v;
        };
        aY = clamp_corr(aY);
        aM = clamp_corr(aM);
        aC = clamp_corr(aC);

        logE_BGR[0] -= aY;
        logE_BGR[1] -= aM;
        logE_BGR[2] -= aC;

        logE_BGR[0] = clamp_to_curve_domain_device(logE_BGR[0], densB);
        logE_BGR[1] = clamp_to_curve_domain_device(logE_BGR[1], densG);
        logE_BGR[2] = clamp_to_curve_domain_device(logE_BGR[2], densR);
    }

    __global__ void phase3_negative_only_kernel(JuicerCuda::Phase3RunParams params) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);

        const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };

        float rgbDWG[3];
        convert_input_to_DWG_device(params.filmRaw, rgbIn, rgbDWG);

        float E_raw[3] = { 0.0f, 0.0f, 0.0f };
        const bool allowHanatos = (params.filmRaw.spectralUpsamplingMode == 0);
        const bool spdReady = params.tablesAx && params.tablesAy && params.tablesAz && params.tablesK == 81;
        const bool useHanatos = allowHanatos &&
            spdReady &&
            params.hanatosLut &&
            (params.hanatosN > 0) &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);
        const bool canTables =
            spdReady &&
            params.sensB.y && params.sensG.y && params.sensR.y &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);

        if (useHanatos) {
            hanatos_layer_exposures_device(
                rgbDWG,
                params.hanatosLut,
                params.hanatosN,
                params.filmRaw.refIllumWhiteXYZ,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }
        else if (canTables) {
            tables_layer_exposures_device(
                rgbDWG,
                params.spdSInv,
                params.filmRaw.refIllumWhiteXYZ,
                params.tablesAx,
                params.tablesAy,
                params.tablesAz,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }

        float midgrayScale = params.filmRaw.midgrayScale;
        if (!isfinite(midgrayScale) || !(midgrayScale > 0.0f)) {
            midgrayScale = 1.0f;
        }

        float exposureScale = params.exposureScale;
        if (!isfinite(exposureScale) || !(exposureScale > 0.0f)) {
            exposureScale = 1.0f;
        }

        float filmRaw[3];
        for (int i = 0; i < 3; ++i) {
            float v = E_raw[i];
            if (!isfinite(v) || v < 0.0f) v = 0.0f;
            v = fmaxf(0.0f, v * midgrayScale);
            v = fmaxf(0.0f, v * exposureScale);
            filmRaw[i] = v;
        }

        constexpr float kLogEps = 1e-10f;
        float logE[3] = {
            log10f(fmaxf(filmRaw[0], 0.0f) + kLogEps),
            log10f(fmaxf(filmRaw[1], 0.0f) + kLogEps),
            log10f(fmaxf(filmRaw[2], 0.0f) + kLogEps)
        };

        // Prevent +/-inf logE from collapsing to NaN in density curve sampling (CPU parity).
        logE[0] = sanitize_inf_logE_for_curve_device(logE[0], params.densB.x, params.densB.n);
        logE[1] = sanitize_inf_logE_for_curve_device(logE[1], params.densG.x, params.densG.n);
        logE[2] = sanitize_inf_logE_for_curve_device(logE[2], params.densR.x, params.densR.n);

        float layerPre[3];
        layerPre[0] = sample_density_at_logE_device(params.densB.x, params.densB.y, params.densB.n, logE[0], params.gammaFactorB);
        layerPre[1] = sample_density_at_logE_device(params.densG.x, params.densG.y, params.densG.n, logE[1], params.gammaFactorG);
        layerPre[2] = sample_density_at_logE_device(params.densR.x, params.densR.y, params.densR.n, logE[2], params.gammaFactorR);

        float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
        if (params.dir.active) {
            float logE_corr[3] = { logE[0], logE[1], logE[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, params.dir, params.densB, params.densG, params.densR);

            const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
            const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
            const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else {
            // Map B/G/R layer densities to C/M/Y dyes
            D_cmy[0] = layerPre[2];
            D_cmy[1] = layerPre[1];
            D_cmy[2] = layerPre[0];
        }

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        if (params.scannerUseLut && params.scanLutLogXYZ && params.scanLutRes > 0) {
            sample_cubic_scan_lut_device(params.scanLutLogXYZ, params.scanLutRes, D_norm, logXYZ);
        }
        else {
            scan_spectral_to_log_xyz_device(params.scan, D_norm, logXYZ);
        }
        const double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        // Output encoding + clamp
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);
        if (nC == 4) {
            dstPix[3] = srcPix[3];
        }
    }

    // --- Scanner glare parity (matches ScannerOptics.cpp) ---

    __device__ __forceinline__ std::uint64_t fnv1a_update_u64_device(std::uint64_t h, std::uint64_t v) {
        constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<std::uint64_t>((v >> (8 * i)) & 0xffULL);
            h *= kFnvPrime;
        }
        return h;
    }

    __device__ __forceinline__ std::uint64_t fnv1a_hash_u64_5_device(
        std::uint64_t a,
        std::uint64_t b,
        std::uint64_t c,
        std::uint64_t d,
        std::uint64_t e)
    {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        h = fnv1a_update_u64_device(h, a);
        h = fnv1a_update_u64_device(h, b);
        h = fnv1a_update_u64_device(h, c);
        h = fnv1a_update_u64_device(h, d);
        h = fnv1a_update_u64_device(h, e);
        return h;
    }

    __device__ __forceinline__ float hash_to_uniform_device(std::uint64_t h) {
        constexpr double kInvU64Max = 1.0 / 18446744073709551615.0;
        return static_cast<float>((static_cast<double>(h) + 0.5) * kInvU64Max);
    }

    __device__ __forceinline__ float box_muller_device(std::uint64_t h1, std::uint64_t h2) {
        float u1 = hash_to_uniform_device(h1);
        u1 = fminf(fmaxf(u1, 1e-7f), 1.0f);
        const float u2 = hash_to_uniform_device(h2);
        const float r = sqrtf(-2.0f * logf(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float theta = kTwoPi * u2;
        return r * cosf(theta);
    }

    __device__ __forceinline__ float lognormal_from_mean_std_device(float mean, float stddev, float normalSample) {
        const float m2 = mean * mean;
        const float s2 = stddev * stddev;
        const float sigmaSq = logf(1.0f + (s2 / m2));
        const float sigma = sqrtf(fmaxf(0.0f, sigmaSq));
        const float mu = logf(fmaxf(1e-12f, mean)) - 0.5f * sigmaSq;
        return expf(mu + sigma * normalSample);
    }

    __global__ void optics_glare_generate_kernel(
        float* out,
        int width,
        int height,
        std::uint64_t glareSeed,
        std::uint64_t mediumId,
        int originX,
        int originY,
        float percent,
        float roughness)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        if (!out) {
            return;
        }
        if (!device_isfinite(percent) || !(percent > 0.0f) || !device_isfinite(roughness)) {
            out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = 0.0f;
            return;
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        const std::uint64_t absX = static_cast<std::uint64_t>(originX + x);
        const std::uint64_t absY = static_cast<std::uint64_t>(originY + y);

        const std::uint64_t h1 = fnv1a_hash_u64_5_device(glareSeed, mediumId, absX, absY, 0ULL);
        const std::uint64_t h2 = fnv1a_hash_u64_5_device(glareSeed, mediumId, absX, absY, 1ULL);
        const float n = box_muller_device(h1, h2);

        const float mean = fmaxf(0.0f, percent);
        const float stddev = fmaxf(0.0f, roughness * percent);
        const float glare = lognormal_from_mean_std_device(mean, stddev, n);

        out[idx] = (device_isfinite(glare) && !isnan(glare)) ? glare : 0.0f;
    }

    __global__ void phase3_stageA_linear_rgb_kernel(
        JuicerCuda::Phase3RunParams params,
        float* rgbR,
        float* rgbG,
        float* rgbB,
        const float* glarePercent)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || params.srcRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
        const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
        if (!srcPix) {
            return;
        }

        const float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };

        float rgbDWG[3];
        convert_input_to_DWG_device(params.filmRaw, rgbIn, rgbDWG);

        float E_raw[3] = { 0.0f, 0.0f, 0.0f };
        const bool allowHanatos = (params.filmRaw.spectralUpsamplingMode == 0);
        const bool spdReady = params.tablesAx && params.tablesAy && params.tablesAz && params.tablesK == 81;
        const bool useHanatos = allowHanatos &&
            spdReady &&
            params.hanatosLut &&
            (params.hanatosN > 0) &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);
        const bool canTables =
            spdReady &&
            params.sensB.y && params.sensG.y && params.sensR.y &&
            (params.sensB.n >= 81) &&
            (params.sensG.n >= 81) &&
            (params.sensR.n >= 81);

        if (useHanatos) {
            hanatos_layer_exposures_device(
                rgbDWG,
                params.hanatosLut,
                params.hanatosN,
                params.filmRaw.refIllumWhiteXYZ,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }
        else if (canTables) {
            tables_layer_exposures_device(
                rgbDWG,
                params.spdSInv,
                params.filmRaw.refIllumWhiteXYZ,
                params.tablesAx,
                params.tablesAy,
                params.tablesAz,
                params.sensB.y,
                params.sensG.y,
                params.sensR.y,
                E_raw);
        }

        float midgrayScale = params.filmRaw.midgrayScale;
        if (!isfinite(midgrayScale) || !(midgrayScale > 0.0f)) {
            midgrayScale = 1.0f;
        }

        float exposureScale = params.exposureScale;
        if (!isfinite(exposureScale) || !(exposureScale > 0.0f)) {
            exposureScale = 1.0f;
        }

        float filmRaw[3];
        for (int i = 0; i < 3; ++i) {
            float v = E_raw[i];
            if (!isfinite(v) || v < 0.0f) v = 0.0f;
            v = fmaxf(0.0f, v * midgrayScale);
            v = fmaxf(0.0f, v * exposureScale);
            filmRaw[i] = v;
        }

        constexpr float kLogEps = 1e-10f;
        float logE[3] = {
            log10f(fmaxf(filmRaw[0], 0.0f) + kLogEps),
            log10f(fmaxf(filmRaw[1], 0.0f) + kLogEps),
            log10f(fmaxf(filmRaw[2], 0.0f) + kLogEps)
        };

        // Prevent +/-inf logE from collapsing to NaN in density curve sampling (CPU parity).
        logE[0] = sanitize_inf_logE_for_curve_device(logE[0], params.densB.x, params.densB.n);
        logE[1] = sanitize_inf_logE_for_curve_device(logE[1], params.densG.x, params.densG.n);
        logE[2] = sanitize_inf_logE_for_curve_device(logE[2], params.densR.x, params.densR.n);

        float layerPre[3];
        layerPre[0] = sample_density_at_logE_device(params.densB.x, params.densB.y, params.densB.n, logE[0], params.gammaFactorB);
        layerPre[1] = sample_density_at_logE_device(params.densG.x, params.densG.y, params.densG.n, logE[1], params.gammaFactorG);
        layerPre[2] = sample_density_at_logE_device(params.densR.x, params.densR.y, params.densR.n, logE[2], params.gammaFactorR);

        float D_cmy[3] = { 0.0f, 0.0f, 0.0f };
        if (params.dir.active) {
            float logE_corr[3] = { logE[0], logE[1], logE[2] };
            apply_dir_runtime_logE_device(logE_corr, layerPre, params.dir, params.densB, params.densG, params.densR);

            const JuicerCuda::DeviceCurveView cB = params.dirPrecorrected ? params.dirDensB : params.densB;
            const JuicerCuda::DeviceCurveView cG = params.dirPrecorrected ? params.dirDensG : params.densG;
            const JuicerCuda::DeviceCurveView cR = params.dirPrecorrected ? params.dirDensR : params.densR;

            const float DY = sample_density_at_logE_device(cB.x, cB.y, cB.n, logE_corr[0], params.gammaFactorB);
            const float DM = sample_density_at_logE_device(cG.x, cG.y, cG.n, logE_corr[1], params.gammaFactorG);
            const float DC = sample_density_at_logE_device(cR.x, cR.y, cR.n, logE_corr[2], params.gammaFactorR);

            D_cmy[0] = DC;
            D_cmy[1] = DM;
            D_cmy[2] = DY;
        }
        else {
            // Map B/G/R layer densities to C/M/Y dyes
            D_cmy[0] = layerPre[2];
            D_cmy[1] = layerPre[1];
            D_cmy[2] = layerPre[0];
        }

        // Scan: normalize density -> logXYZ
        double D_norm[3];
        if (params.scan.mediumIsNegative) {
            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(params.scan.min_cmy[0])) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(params.scan.min_cmy[1])) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(params.scan.min_cmy[2])) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }
        else {
            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(params.scan.inv_max_cmy[0]);
            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(params.scan.inv_max_cmy[1]);
            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(params.scan.inv_max_cmy[2]);
        }

        double logXYZ[3] = { 0.0, 0.0, 0.0 };
        if (params.scannerUseLut && params.scanLutLogXYZ && params.scanLutRes > 0) {
            sample_cubic_scan_lut_device(params.scanLutLogXYZ, params.scanLutRes, D_norm, logXYZ);
        }
        else {
            scan_spectral_to_log_xyz_device(params.scan, D_norm, logXYZ);
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);

        double xyz[3] = {
            pow(10.0, logXYZ[0]),
            pow(10.0, logXYZ[1]),
            pow(10.0, logXYZ[2])
        };

        if (glarePercent) {
            const double glare = static_cast<double>(glarePercent[idx]) * 0.01;
            xyz[0] += glare * static_cast<double>(params.scanColor.illuminantXYZ[0]);
            xyz[1] += glare * static_cast<double>(params.scanColor.illuminantXYZ[1]);
            xyz[2] += glare * static_cast<double>(params.scanColor.illuminantXYZ[2]);
        }

        double adapted[3];
        mat3_mul_vec_double_device(params.scanColor.cat02, xyz, adapted);
        double rgbOut[3];
        mat3_mul_vec_double_device(params.scanColor.xyzToRgb, adapted, rgbOut);

        rgbR[idx] = (isfinite(rgbOut[0]) && !isnan(rgbOut[0])) ? static_cast<float>(rgbOut[0]) : 0.0f;
        rgbG[idx] = (isfinite(rgbOut[1]) && !isnan(rgbOut[1])) ? static_cast<float>(rgbOut[1]) : 0.0f;
        rgbB[idx] = (isfinite(rgbOut[2]) && !isnan(rgbOut[2])) ? static_cast<float>(rgbOut[2]) : 0.0f;
    }

    __global__ void optics_blur_horizontal_kernel(
        const float* in,
        float* out,
        int width,
        int height,
        const float* k,
        int radius)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        if (!in || !out || !k || radius <= 0) {
            return;
        }

        double acc = 0.0;
        for (int j = -radius; j <= radius; ++j) {
            const int xx = reflect_index_repeat_device(x + j, width);
            const float v = in[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(xx)];
            const float w = k[j + radius];
            acc += static_cast<double>(v) * static_cast<double>(w);
        }
        out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
            (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
    }

    __global__ void optics_blur_vertical_kernel(
        const float* in,
        float* out,
        int width,
        int height,
        const float* k,
        int radius)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        if (!in || !out || !k || radius <= 0) {
            return;
        }

        double acc = 0.0;
        for (int j = -radius; j <= radius; ++j) {
            const int yy = reflect_index_repeat_device(y + j, height);
            const float v = in[static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(x)];
            const float w = k[j + radius];
            acc += static_cast<double>(v) * static_cast<double>(w);
        }
        out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
            (isfinite(acc) && !isnan(acc)) ? static_cast<float>(acc) : 0.0f;
    }

    __global__ void optics_unsharp_combine_kernel(float* inOut, const float* blurred, int n, float amount) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        if (!inOut || !blurred) {
            return;
        }
        const double v0 = static_cast<double>(inOut[idx]);
        const double vb = static_cast<double>(blurred[idx]);
        const double a = static_cast<double>(amount);
        const double v = v0 + a * (v0 - vb);
        inOut[idx] = (isfinite(v) && !isnan(v)) ? static_cast<float>(v) : 0.0f;
    }

    __global__ void phase3_stageD_encode_write_kernel(
        JuicerCuda::Phase3RunParams params,
        const float* rgbR,
        const float* rgbG,
        const float* rgbB)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= params.width || y >= params.height) {
            return;
        }

        if (!params.src || !params.dst || params.srcRowBytes == 0 || params.dstRowBytes == 0) {
            return;
        }
        if (!rgbR || !rgbG || !rgbB) {
            return;
        }

        const int nC = params.nComponents;
        if (!(nC == 3 || nC == 4)) {
            return;
        }

        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(params.width) + static_cast<size_t>(x);
        double rgbOut[3] = {
            static_cast<double>(rgbR[idx]),
            static_cast<double>(rgbG[idx]),
            static_cast<double>(rgbB[idx])
        };
        apply_output_encoding_device(params.scanColor.encoding, rgbOut);

        const std::size_t pixelBytes = static_cast<std::size_t>(nC) * sizeof(float);
        char* dstRow = reinterpret_cast<char*>(params.dst) + static_cast<std::size_t>(y) * params.dstRowBytes;
        float* dstPix = reinterpret_cast<float*>(dstRow + static_cast<std::size_t>(x) * pixelBytes);
        dstPix[0] = static_cast<float>(rgbOut[0]);
        dstPix[1] = static_cast<float>(rgbOut[1]);
        dstPix[2] = static_cast<float>(rgbOut[2]);

        if (nC == 4) {
            const char* srcRow = reinterpret_cast<const char*>(params.src) + static_cast<std::size_t>(y) * params.srcRowBytes;
            const float* srcPix = reinterpret_cast<const float*>(srcRow + static_cast<std::size_t>(x) * pixelBytes);
            dstPix[3] = srcPix ? srcPix[3] : 1.0f;
        }
    }

} // namespace

extern "C" cudaError_t juicer_cuda_phase3_negative_only(
    const JuicerCuda::Phase3RunParams* hParams,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads(16, 16);
    dim3 blocks(
        static_cast<unsigned int>((params.width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((params.height + threads.y - 1) / threads.y));
    phase3_negative_only_kernel<<<blocks, threads, 0, stream>>>(params);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_phase3_negative_only_optics(
    const JuicerCuda::Phase3RunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
    void* cudaStreamOpaque)
{
    if (!hParams) {
        return cudaErrorInvalidValue;
    }

    const JuicerCuda::Phase3RunParams params = *hParams;
    if (!params.src || !params.dst) {
        return cudaErrorInvalidValue;
    }
    if (params.width <= 0 || params.height <= 0) {
        return cudaSuccess;
    }
    if (!(params.nComponents == 3 || params.nComponents == 4)) {
        return cudaErrorInvalidValue;
    }
    if (params.srcRowBytes == 0 || params.dstRowBytes == 0) {
        return cudaErrorInvalidValue;
    }
    if (!dRgbR || !dRgbG || !dRgbB || !dTmp) {
        return cudaErrorInvalidValue;
    }

    cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

    dim3 threads2D(16, 16);
    dim3 blocks2D(
        static_cast<unsigned int>((params.width + threads2D.x - 1) / threads2D.x),
        static_cast<unsigned int>((params.height + threads2D.y - 1) / threads2D.y));

    auto blur_plane_in_place = [&](float* plane, float* tmpBuf, const float* k, int radius) -> cudaError_t {
        if (!plane || !tmpBuf || !k || radius <= 0) {
            return cudaSuccess;
        }
        optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, tmpBuf, params.width, params.height, k, radius);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return e;
        }
        optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(tmpBuf, plane, params.width, params.height, k, radius);
        return cudaGetLastError();
    };

    const bool doGlare =
        std::isfinite(static_cast<double>(glarePercent)) && (glarePercent > 0.0f) &&
        std::isfinite(static_cast<double>(glareRoughness));
    if (doGlare) {
        const std::uint64_t mediumId = params.scan.mediumIsNegative ? 0ULL : 1ULL;
        optics_glare_generate_kernel<<<blocks2D, threads2D, 0, stream>>>(
            dTmp,
            params.width,
            params.height,
            glareSeed,
            mediumId,
            glareOriginX,
            glareOriginY,
            glarePercent,
            glareRoughness);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (glareRadius > 0 && dGlareKernel) {
            err = blur_plane_in_place(dTmp, dRgbR, dGlareKernel, glareRadius);
            if (err != cudaSuccess) return err;
        }
    }

    phase3_stageA_linear_rgb_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB, doGlare ? dTmp : nullptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    if (lensBlurRadius > 0 && dLensBlurKernel) {
        err = blur_plane_in_place(dRgbR, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbG, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
        err = blur_plane_in_place(dRgbB, dTmp, dLensBlurKernel, lensBlurRadius);
        if (err != cudaSuccess) return err;
    }

    const bool doUnsharp =
        (unsharpRadius > 0) && dUnsharpKernel &&
        std::isfinite(static_cast<double>(unsharpAmount)) && (unsharpAmount != 0.0f);
    if (doUnsharp) {
        if (!dScratchBlurred) {
            return cudaErrorInvalidValue;
        }

        const int total = params.width * params.height;
        const int threads1D = 256;
        const int blocks1D = (total + threads1D - 1) / threads1D;

        auto unsharp_plane_in_place = [&](float* plane) -> cudaError_t {
            optics_blur_horizontal_kernel<<<blocks2D, threads2D, 0, stream>>>(plane, dTmp, params.width, params.height, dUnsharpKernel, unsharpRadius);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_blur_vertical_kernel<<<blocks2D, threads2D, 0, stream>>>(dTmp, dScratchBlurred, params.width, params.height, dUnsharpKernel, unsharpRadius);
            e = cudaGetLastError();
            if (e != cudaSuccess) {
                return e;
            }
            optics_unsharp_combine_kernel<<<blocks1D, threads1D, 0, stream>>>(plane, dScratchBlurred, total, unsharpAmount);
            return cudaGetLastError();
        };

        err = unsharp_plane_in_place(dRgbR);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbG);
        if (err != cudaSuccess) return err;
        err = unsharp_plane_in_place(dRgbB);
        if (err != cudaSuccess) return err;
    }

    phase3_stageD_encode_write_kernel<<<blocks2D, threads2D, 0, stream>>>(params, dRgbR, dRgbG, dRgbB);
    return cudaGetLastError();
}
