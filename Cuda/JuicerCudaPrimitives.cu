// Cuda/JuicerCudaPrimitives.cu
//
// Phase 2: core CUDA math primitives with CPU-parity semantics.
//
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

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
