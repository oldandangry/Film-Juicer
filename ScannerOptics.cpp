// ScannerOptics.cpp

#include "ScannerOptics.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>

#include "Hash.h"
#include "Logging.h"
#include "OutputEncoding.h"
#include "GeneratedColorSpaces.h"
#include "SpectralProcessing.h"
#include "ColorTransforms.h"

namespace {

    template <typename Scalar>
    inline void mat3_mul_vec(const Scalar m[9], const Scalar v[3], Scalar out[3]) {
        out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
        out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
        out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    }

    inline void mat3_mul(const float a[9], const float b[9], float out[9]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out[r * 3 + c] =
                    a[r * 3 + 0] * b[0 * 3 + c] +
                    a[r * 3 + 1] * b[1 * 3 + c] +
                    a[r * 3 + 2] * b[2 * 3 + c];
            }
        }
    }

    template <typename Scalar>
    inline void build_gaussian_kernel(float sigma, std::vector<Scalar>& kernel) {
        kernel.clear();
        if (!(std::isfinite(sigma)) || sigma <= 0.0f) {
            kernel.push_back(static_cast<Scalar>(1.0));
            return;
        }
        const int radiusRaw = std::max(1, int(std::ceil(4.0f * sigma)));
        const int radius = std::min(radiusRaw, 75);
        kernel.resize(size_t(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            kernel[size_t(i + radius)] = static_cast<Scalar>(w);
            wsum += w;
        }
        const Scalar invW = (wsum != 0.0) ? static_cast<Scalar>(1.0 / wsum) : static_cast<Scalar>(0.0);
        for (Scalar& w : kernel) w *= invW;
    }

    template <typename Scalar>
    void blur_separable(const std::vector<Scalar>& src, std::vector<Scalar>& tmp, std::vector<Scalar>& dst,
        int width, int height, const std::vector<Scalar>& k) {
        tmp.assign(size_t(width * height), static_cast<Scalar>(0.0));
        const int radius = int(k.size() / 2);
        auto reflectIndex = [](int idx, int size) -> int {
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
            };
        for (int y = 0; y < height; ++y) {
            const Scalar* srow = &src[size_t(y * width)];
            Scalar* trow = &tmp[size_t(y * width)];
            for (int x = 0; x < width; ++x) {
                Scalar acc = static_cast<Scalar>(0.0);
                for (int j = -radius; j <= radius; ++j) {
                    const int xx = reflectIndex(x + j, width);
                    acc += srow[xx] * k[size_t(j + radius)];
                }
                trow[x] = acc;
            }
        }
        dst.assign(size_t(width * height), static_cast<Scalar>(0.0));
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                Scalar acc = static_cast<Scalar>(0.0);
                for (int j = -radius; j <= radius; ++j) {
                    const int yy = reflectIndex(y + j, height);
                    acc += tmp[size_t(yy * width + x)] * k[size_t(j + radius)];
                }
                dst[size_t(y * width + x)] = acc;
            }
        }
    }

    inline double mitchell_weight(double t) {
        const double B = 1.0 / 3.0;
        const double C = 1.0 / 3.0;
        const double x = std::fabs(t);
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

    inline int reflect_index(int idx, int size) {
        if (size <= 1) return 0;
        if (idx < 0) return -idx;
        if (idx >= size) return 2 * (size - 1) - idx;
        return idx;
    }

    inline void sample_cubic(const Scanner::SpectralLutBuffer& lut, const double D_norm[3], double out[3]) {
        const int res = static_cast<int>(std::max(1u, lut.res));
        const double scale = (res > 1) ? static_cast<double>(res - 1) : 1.0;
        const double fx = D_norm[0] * scale;
        const double fy = D_norm[1] * scale;
        const double fz = D_norm[2] * scale;

        const int xBase = static_cast<int>(std::floor(fx));
        const int yBase = static_cast<int>(std::floor(fy));
        const int zBase = static_cast<int>(std::floor(fz));
        const double tx = fx - static_cast<double>(xBase);
        const double ty = fy - static_cast<double>(yBase);
        const double tz = fz - static_cast<double>(zBase);

        double wx[4], wy[4], wz[4];
        wx[0] = mitchell_weight(tx + 1.0);
        wx[1] = mitchell_weight(tx);
        wx[2] = mitchell_weight(tx - 1.0);
        wx[3] = mitchell_weight(tx - 2.0);
        wy[0] = mitchell_weight(ty + 1.0);
        wy[1] = mitchell_weight(ty);
        wy[2] = mitchell_weight(ty - 1.0);
        wy[3] = mitchell_weight(ty - 2.0);
        wz[0] = mitchell_weight(tz + 1.0);
        wz[1] = mitchell_weight(tz);
        wz[2] = mitchell_weight(tz - 1.0);
        wz[3] = mitchell_weight(tz - 2.0);

        auto lut_at = [&](int xi, int yi, int zi, int c) -> double {
            const size_t idx = (size_t(zi) * size_t(res) + size_t(yi)) * size_t(res) + size_t(xi);
            const size_t base = idx * 3 + size_t(c);
            if (base >= lut.cpu.size()) return 0.0;
            return lut.cpu[base];
            };

        double sum[3] = { 0.0, 0.0, 0.0 };
        double wsum = 0.0;
        for (int i = 0; i < 4; ++i) {
            const int xi = reflect_index(xBase - 1 + i, res);
            for (int j = 0; j < 4; ++j) {
                const int yj = reflect_index(yBase - 1 + j, res);
                for (int k = 0; k < 4; ++k) {
                    const int zk = reflect_index(zBase - 1 + k, res);
                    const double w = wx[i] * wy[j] * wz[k];
                    wsum += w;
                    for (int c = 0; c < 3; ++c) {
                        sum[c] += w * lut_at(xi, yj, zk, c);
                    }
                }
            }
        }
        const double inv = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        out[0] = sum[0] * inv;
        out[1] = sum[1] * inv;
        out[2] = sum[2] * inv;
    }

    inline void dyes_to_XYZ_given_tables_double(
        const Spectral::SpectralTables& T,
        const double dyes_cmy[3],
        double XYZ[3],
        bool useBaseline,
        double invNormalization)
    {
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        constexpr double kLn10d = 2.302585092994046;
        for (int i = 0; i < K; ++i) {
            if (useBaseline && T.hasBaseline &&
                static_cast<size_t>(i) < T.baseMinValid.size() &&
                T.baseMinValid[static_cast<size_t>(i)] == 0) {
                continue;
            }
            const double baseSpectral = (useBaseline && T.hasBaseline) ? static_cast<double>(T.baseMin[i]) : 0.0;
            const double Dlambda = dyes_cmy[0] * static_cast<double>(T.epsC[i])
                + dyes_cmy[1] * static_cast<double>(T.epsM[i])
                + dyes_cmy[2] * static_cast<double>(T.epsY[i])
                + baseSpectral;
            if (!std::isfinite(Dlambda)) {
                continue;
            }
            const double Tlambda = std::exp(-kLn10d * Dlambda);
            X += Tlambda * static_cast<double>(T.Ax[i]);
            Y += Tlambda * static_cast<double>(T.Ay[i]);
            Z += Tlambda * static_cast<double>(T.Az[i]);
        }
        XYZ[0] = X * invNormalization;
        XYZ[1] = Y * invNormalization;
        XYZ[2] = Z * invNormalization;
    }

    inline float hash_to_uniform(std::uint64_t h) {
        const double scale = 1.0 / static_cast<double>(std::numeric_limits<std::uint64_t>::max());
        return static_cast<float>((static_cast<double>(h) + 0.5) * scale);
    }

    inline float box_muller(std::uint64_t h1, std::uint64_t h2) {
        float u1 = std::clamp(hash_to_uniform(h1), 1e-7f, 1.0f);
        const float u2 = hash_to_uniform(h2);
        const float r = std::sqrt(-2.0f * std::log(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float theta = kTwoPi * u2;
        return r * std::cos(theta);
    }

    inline float lognormal_from_mean_std(float mean, float stddev, float normalSample) {
        const float m2 = mean * mean;
        const float s2 = stddev * stddev;
        const float sigmaSq = std::log(1.0f + (s2 / m2));
        const float sigma = std::sqrt(std::max(0.0f, sigmaSq));
        const float mu = std::log(std::max(1e-12f, mean)) - 0.5f * sigmaSq;
        return std::exp(mu + sigma * normalSample);
    }

    inline bool should_rebuild_lut(
        const ScannerOptics::Runtime& runtime,
        const Scanner::ScannerStaticKey& staticKey,
        bool staticKeyChanged)
    {
        if (!runtime.lut.valid) return true;
        if (staticKeyChanged) return true;
        const std::uint64_t expected =
            Hash::hash_bytes(&staticKey.hash, sizeof(staticKey.hash));
        if (runtime.lut.hash != expected) return true;
        return false;
    }

    inline std::uint64_t hash_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& encoding,
        const GeneratedColorSpaces::ColorSpaceEntry& outSpace)
    {
        const std::uint64_t encFields[] = {
            static_cast<std::uint64_t>(OutputEncoding::toIndex(encoding.colorSpace)),
            static_cast<std::uint64_t>(encoding.applyCctfEncoding),
            static_cast<std::uint64_t>(encoding.preserveLinearRange),
            static_cast<std::uint64_t>(encoding.inputIsOutputSpace)
        };
        const std::uint64_t encHash = Hash::hash_bytes(encFields, sizeof(encFields));
        const std::uint64_t fields[] = {
            outSpace.hash,
            medium.illuminant.hash,
            static_cast<std::uint64_t>(medium.medium),
            encHash
        };
        return Hash::hash_bytes(fields, sizeof(fields));
    }

    inline void log_scanner_keys(
        const char* label,
        const Scanner::ScannerKey& key,
        const Scanner::ScannerMediumRuntime& medium,
        const Scanner::Settings& settings,
        const Spectral::SpectralTables* tables)
    {
        std::ostringstream oss;
        oss << label
            << " medium=" << static_cast<int>(medium.medium)
            << " static=" << key.staticKey.hash
            << " tablesHash=" << key.staticKey.tablesHash
            << " densityRangeHash=" << key.staticKey.densityRangeHash
            << " glareHash=" << key.staticKey.glareHash
            << " colorHash=" << key.staticKey.colorRuntimeHash
            << " lutRes=" << key.staticKey.lutResolution
            << " useLut=" << (settings.useLut ? 1 : 0)
            << " settingsHash=" << key.runtimeKey.settingsHash
            << " frameBoundsV=" << key.runtimeKey.frameBoundsVersion;
        if (tables) {
            oss << " tables.tablesHash=" << tables->tablesHash
                << " tables.illumHash=" << tables->illuminantHash;
        }
        JTRACE("SCAN", oss.str());
    }

} // namespace

namespace ScannerOptics {

    Scanner::ColorRuntime build_color_runtime(
        const Scanner::ScannerMediumRuntime& medium,
        const OutputEncoding::Params& outputEncoding)
    {
        Scanner::ColorRuntime rt{};
        if (medium.illuminant.hash == 0) {
            JTRACE("HASH", "FATAL: scanner illuminant hash invalid for color runtime");
            return rt;
        }
        const GeneratedColorSpaces::ColorSpaceEntry& outSpace =
            GeneratedColorSpaces::get(outputEncoding.colorSpace);
        if (outSpace.hash == 0) {
            JTRACE("HASH", "FATAL: generated color space hash invalid");
            return rt;
        }
        Spectral::Mat3 adapt = Spectral::build_chromatic_adaptation_matrix(
            medium.illuminant.whiteXYZ,
            outSpace.whiteXYZ);
        for (int i = 0; i < 9; ++i) {
            rt.cat02[i] = adapt.m[i];
        }

        for (int i = 0; i < 9; ++i) {
            rt.xyzToRgb[i] = outSpace.xyzToRgb[i];
        }
        rt.encoding = outputEncoding;
        rt.encoding.inputIsOutputSpace = true;
        rt.illuminantXYZ[0] = medium.illuminant.whiteXYZ[0];
        rt.illuminantXYZ[1] = medium.illuminant.whiteXYZ[1];
        rt.illuminantXYZ[2] = medium.illuminant.whiteXYZ[2];
        rt.hash = hash_color_runtime(medium, rt.encoding, outSpace);
        return rt;
    }

    void render_density_to_rgb(const RenderContext& ctx) {
        if (!ctx.medium || !ctx.density || !ctx.runtime) {
            JTRACE("SCAN", "FATAL: scanner render context incomplete");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!ctx.color) {
            JTRACE("SCAN", "FATAL: scanner color runtime missing");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        Runtime& runtime = *ctx.runtime;
        const std::uint64_t prevStatic = runtime.key.staticKey.hash;
        const std::uint64_t prevGlareHash = runtime.key.staticKey.glareHash;
        const Scanner::ScannerRuntimeKey prevRuntimeKey = runtime.key.runtimeKey;
        runtime.key = ctx.scannerKey;
        const bool staticKeyChanged = (prevStatic != ctx.scannerKey.staticKey.hash);
        const bool settingsChanged =
            prevRuntimeKey.settingsHash != ctx.scannerKey.runtimeKey.settingsHash;
        const bool frameBoundsChanged =
            prevRuntimeKey.frameBoundsVersion != ctx.scannerKey.runtimeKey.frameBoundsVersion;

        const Scanner::ScannerMediumRuntime& medium = *ctx.medium;
        const Scanner::ScannerDensityBuffer& density = *ctx.density;
        const Spectral::SpectralTables* tables = medium.tables;
        if (staticKeyChanged || settingsChanged || frameBoundsChanged) {
            log_scanner_keys("scanner runtime rebuild",
                ctx.scannerKey, medium, ctx.settings, tables);
        }
        if (!tables || tables->K <= 0) {
            JTRACE("SCAN", "FATAL: scanner spectral tables unavailable");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (tables->tablesHash != medium.staticKey.tablesHash || medium.staticKey.hash == 0) {
            JTRACE("SCAN", "FATAL: scanner static key/table hash mismatch");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (density.medium != medium.medium) {
            JTRACE("SCAN", "FATAL: density slab medium does not match scanner medium runtime");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (tables->illuminantHash != 0 && medium.illuminant.hash != 0 &&
            tables->illuminantHash != medium.illuminant.hash) {
            JTRACE("SCAN", "FATAL: scanner illuminant hash mismatch");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (tables->illuminantHash == 0 || medium.illuminant.hash == 0) {
            JTRACE("SCAN", "FATAL: scanner illuminant hash missing");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (medium.range.digest == 0) {
            JTRACE("SCAN", "FATAL: scanner density range missing or invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        const double tablesInvYn = static_cast<double>(tables->invYn);
        if (!(std::isfinite(tablesInvYn) && tablesInvYn > 0.0)) {
            JTRACE("SCAN", "FATAL: scanner tables contain invalid invYn");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (!(std::isfinite(medium.illuminant.normalization) && medium.illuminant.normalization > 0.0f)) {
            JTRACE("SCAN", "FATAL: scanner illuminant normalization invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        const double invNormalization = 1.0 / static_cast<double>(medium.illuminant.normalization);
        if (!(std::isfinite(invNormalization) && invNormalization > 0.0)) {
            JTRACE("SCAN", "FATAL: scanner illuminant normalization invalid");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(medium.illuminant.whiteXYZ[i])) {
                JTRACE("SCAN", "FATAL: scanner illuminant whiteXYZ invalid");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }
        if (!(medium.illuminant.whiteXYZ[1] > 0.0f)) {
            JTRACE("SCAN", "FATAL: scanner illuminant whiteXYZ has invalid Y component");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        {
            constexpr double kTolInvYnRel = 1e-5;
            const double invYnFromIll = invNormalization;
            const double invYnFromTables = tablesInvYn;
            const double denom = std::max(1e-30, std::abs(invYnFromIll));
            const double rel = std::abs(invYnFromIll - invYnFromTables) / denom;
            if (!(std::isfinite(rel) && rel <= kTolInvYnRel)) {
                JTRACE("SCAN", "FATAL: scanner illuminant normalization disagrees with tables invYn");
                throw OFX::Exception::Suite(kOfxStatErrFatal);
            }
        }
        {
            constexpr double kTolWhiteAbs = 1e-3;
            for (int i = 0; i < 3; ++i) {
                const double a = static_cast<double>(tables->whiteXYZ[i]);
                const double b = static_cast<double>(medium.illuminant.whiteXYZ[i]);
                if (!(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= kTolWhiteAbs)) {
                    JTRACE("SCAN", "FATAL: scanner tables whiteXYZ disagrees with illuminant whiteXYZ");
                    throw OFX::Exception::Suite(kOfxStatErrFatal);
                }
            }
        }
        const bool useBaseline = tables->hasBaseline; // per-medium baseline; do not gate on film state
        const int width = ctx.bounds.x2 - ctx.bounds.x1;
        const int height = ctx.bounds.y2 - ctx.bounds.y1;
        const int originX = ctx.bounds.x1;
        const int originY = ctx.bounds.y1;
        if (width <= 0 || height <= 0) {
            return;
        }
        const size_t total = size_t(width) * size_t(height);
        const size_t channelSize = total;
        if (density.width != width || density.height != height || density.stride != width ||
            density.c.size() < total || density.m.size() < total || density.y.size() < total) {
            JTRACE("SCAN", "FATAL: density slab does not match render bounds");
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }

        auto spectral_to_logXYZ = [&](const double D_norm[3], double logXYZ[3]) {
            double D_denorm[3];
            if (medium.medium == Scanner::ScannerMedium::Negative) {
                D_denorm[0] = D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]) - static_cast<double>(medium.range.min_cmy[0]);
                D_denorm[1] = D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]) - static_cast<double>(medium.range.min_cmy[1]);
                D_denorm[2] = D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]) - static_cast<double>(medium.range.min_cmy[2]);
            }
            else {
                D_denorm[0] = D_norm[0] / static_cast<double>(medium.range.inv_max_cmy[0]);
                D_denorm[1] = D_norm[1] / static_cast<double>(medium.range.inv_max_cmy[1]);
                D_denorm[2] = D_norm[2] / static_cast<double>(medium.range.inv_max_cmy[2]);
            }
            double XYZ[3] = { 0.0, 0.0, 0.0 };
            dyes_to_XYZ_given_tables_double(*tables, D_denorm, XYZ, useBaseline, invNormalization);
            constexpr double kEps = 1e-10;
            logXYZ[0] = std::log10(std::max(0.0, XYZ[0]) + kEps);
            logXYZ[1] = std::log10(std::max(0.0, XYZ[1]) + kEps);
            logXYZ[2] = std::log10(std::max(0.0, XYZ[2]) + kEps);
        };

        // Prepare LUT if needed
        const bool useLut = ctx.settings.useLut;
        if (useLut && should_rebuild_lut(runtime, ctx.scannerKey.staticKey, staticKeyChanged)) {
            const std::uint32_t res = std::clamp(
                ctx.scannerKey.staticKey.lutResolution, 17u, 128u);
            runtime.lut.cpu.assign(size_t(res) * size_t(res) * size_t(res) * 3u, 0.0);
            runtime.lut.res = res;
            {
                std::ostringstream oss;
                oss << "build LUT res=" << res
                    << " staticKey=" << ctx.scannerKey.staticKey.hash
                    << " colorHash=" << ctx.scannerKey.staticKey.colorRuntimeHash
                    << " tablesHash=" << ctx.scannerKey.staticKey.tablesHash
                    << " densityRangeHash=" << ctx.scannerKey.staticKey.densityRangeHash
                    << " glareHash=" << ctx.scannerKey.staticKey.glareHash;
                JTRACE("SCAN", oss.str());
            }
            for (std::uint32_t z = 0; z < res; ++z) {
                const double nz = (res > 1u) ? double(z) / double(res - 1u) : 0.0;
                for (std::uint32_t y = 0; y < res; ++y) {
                    const double ny = (res > 1u) ? double(y) / double(res - 1u) : 0.0;
                    for (std::uint32_t x = 0; x < res; ++x) {
                        const double nx = (res > 1u) ? double(x) / double(res - 1u) : 0.0;
                        const size_t idx = (size_t(z) * size_t(res) + size_t(y)) * size_t(res) + size_t(x);
                        double logXYZ[3];
                        const double D_norm[3] = { nx, ny, nz };
                        spectral_to_logXYZ(D_norm, logXYZ);
                        if (!std::isfinite(logXYZ[0]) || !std::isfinite(logXYZ[1]) || !std::isfinite(logXYZ[2])) {
                            JTRACE("SCAN", "FATAL: invalid LUT sample during spectral computation");
                            throw OFX::Exception::Suite(kOfxStatErrFatal);
                        }
                        const size_t base = idx * 3u;
                        runtime.lut.cpu[base + 0] = logXYZ[0];
                        runtime.lut.cpu[base + 1] = logXYZ[1];
                        runtime.lut.cpu[base + 2] = logXYZ[2];
                    }
                }
            }
            runtime.lut.hash = Hash::hash_bytes(&ctx.scannerKey.staticKey.hash, sizeof(ctx.scannerKey.staticKey.hash));
            runtime.lut.valid = true;
        }
        if (!useLut) {
            runtime.lut.valid = false;
        }

        // Prepare blur/unsharp kernels
        if (staticKeyChanged || settingsChanged || runtime.blurKernel.empty()) {
            build_gaussian_kernel(ctx.options.lensBlurSigmaPx, runtime.blurKernel);
        }
        if (staticKeyChanged || settingsChanged || runtime.unsharpKernel.empty()) {
            build_gaussian_kernel(ctx.options.unsharpSigmaPx, runtime.unsharpKernel);
        }
        runtime.unsharpAmount = ctx.options.unsharpAmount;

        // Prepare glare cache when active
        const bool glareActive = medium.glare.active && medium.glare.percent > 0.0f;
        if (glareActive) {
            const std::uint64_t seedFields[4] = {
                ctx.seedBase,
                static_cast<std::uint64_t>(ctx.runtimeKey.frameBoundsVersion),
                medium.staticKey.glareHash,
                static_cast<std::uint64_t>(medium.medium)
            };
            const std::uint64_t glareSeed = Hash::hash_bytes(seedFields, sizeof(seedFields));
            const bool dimsChanged = runtime.glare.width != width || runtime.glare.height != height;
            const bool seedChanged = runtime.glare.seedHash != glareSeed;
            const bool glareParamsChanged = prevGlareHash != ctx.scannerKey.staticKey.glareHash;
            if (!runtime.glare.valid || dimsChanged || seedChanged || glareParamsChanged || frameBoundsChanged) {
                const size_t channelSize = total;
                runtime.glare.amount.assign(channelSize, 0.0f);
                runtime.glare.tmp.assign(channelSize, 0.0f);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const std::uint64_t absX = static_cast<std::uint64_t>(originX + x);
                        const std::uint64_t absY = static_cast<std::uint64_t>(originY + y);
                        const size_t idx = size_t(y) * size_t(width) + size_t(x);
                        const std::uint64_t mix1[5] = {
                            glareSeed,
                            static_cast<std::uint64_t>(medium.medium),
                            absX,
                            absY,
                            0ull
                        };
                        const std::uint64_t mix2[5] = {
                            glareSeed,
                            static_cast<std::uint64_t>(medium.medium),
                            absX,
                            absY,
                            1ull
                        };
                        const float n = box_muller(
                            Hash::hash_bytes(mix1, sizeof(mix1)),
                            Hash::hash_bytes(mix2, sizeof(mix2)));
                        const float mean = static_cast<float>(medium.glare.percent);
                        const float stddev = static_cast<float>(medium.glare.roughness * medium.glare.percent);
                        float glare = lognormal_from_mean_std(std::max(0.0f, mean), std::max(0.0f, stddev), n);
                        runtime.glare.amount[idx] = glare;
                    }
                }
                if (medium.glare.blur > 0.0f) {
                    std::vector<float> glareKernel;
                    build_gaussian_kernel(static_cast<float>(medium.glare.blur), glareKernel);
                    std::vector<float> glareBuf(runtime.glare.amount.begin(), runtime.glare.amount.end());
                    blur_separable(glareBuf, runtime.glare.tmp, runtime.glare.amount, width, height, glareKernel);
                }
                for (float& g : runtime.glare.amount) {
                    g = g / 100.0f;
                }
                runtime.glare.width = width;
                runtime.glare.height = height;
                runtime.glare.key = ctx.scannerKey;
                runtime.glare.seedHash = glareSeed;
                runtime.glare.valid = true;
            }
        }
        else {
            runtime.glare.valid = false;
        }

        const unsigned int nThreads = std::max(1u, ctx.threadCount);
        const int rowsPerThread = (height + int(nThreads) - 1) / int(nThreads);

        double cat02[9];
        double xyzToRgb[9];
        for (int i = 0; i < 9; ++i) {
            cat02[i] = static_cast<double>(ctx.color->cat02[i]);
            xyzToRgb[i] = static_cast<double>(ctx.color->xyzToRgb[i]);
        }

        std::vector<double> rgbR(total, 0.0);
        std::vector<double> rgbG(total, 0.0);
        std::vector<double> rgbB(total, 0.0);

        std::atomic<bool> abortFlag{ false };
        std::atomic<bool> failure{ false };
        std::vector<std::thread> threads;
        threads.reserve(nThreads);

        // Stage A: density -> linear RGB
        for (unsigned int t = 0; t < nThreads; ++t) {
            const int yStart = rowsPerThread * int(t);
            const int yEnd = std::min(height, rowsPerThread * int(t + 1));
            threads.emplace_back([&, yStart, yEnd]() {
                for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                    if (ctx.abort.abortRequested()) {
                        abortFlag.store(true, std::memory_order_relaxed);
                        break;
                    }
                    const size_t rowOffset = size_t(yOff) * size_t(width);
                    for (int xOff = 0; xOff < width; ++xOff) {
                        if (abortFlag.load(std::memory_order_relaxed)) {
                            break;
                        }
                        const size_t idx = rowOffset + size_t(xOff);
                        float D_cmy[3] = { density.c[idx], density.m[idx], density.y[idx] };
                        double D_norm[3];
                        if (medium.medium == Scanner::ScannerMedium::Negative) {
                            D_norm[0] = (static_cast<double>(D_cmy[0]) + static_cast<double>(medium.range.min_cmy[0])) * static_cast<double>(medium.range.inv_max_cmy[0]);
                            D_norm[1] = (static_cast<double>(D_cmy[1]) + static_cast<double>(medium.range.min_cmy[1])) * static_cast<double>(medium.range.inv_max_cmy[1]);
                            D_norm[2] = (static_cast<double>(D_cmy[2]) + static_cast<double>(medium.range.min_cmy[2])) * static_cast<double>(medium.range.inv_max_cmy[2]);
                        }
                        else {
                            D_norm[0] = static_cast<double>(D_cmy[0]) * static_cast<double>(medium.range.inv_max_cmy[0]);
                            D_norm[1] = static_cast<double>(D_cmy[1]) * static_cast<double>(medium.range.inv_max_cmy[1]);
                            D_norm[2] = static_cast<double>(D_cmy[2]) * static_cast<double>(medium.range.inv_max_cmy[2]);
                        }
                        if (!std::isfinite(D_norm[0]) || !std::isfinite(D_norm[1]) || !std::isfinite(D_norm[2])) {
                            abortFlag.store(true, std::memory_order_relaxed);
                            failure.store(true, std::memory_order_relaxed);
                            break;
                        }
                        double logXYZ[3];
                        if (useLut && runtime.lut.valid) {
                            sample_cubic(runtime.lut, D_norm, logXYZ);
                        }
                        else {
                            spectral_to_logXYZ(D_norm, logXYZ);
                        }
                        double xyz[3] = {
                            std::pow(10.0, logXYZ[0]),
                            std::pow(10.0, logXYZ[1]),
                            std::pow(10.0, logXYZ[2])
                        };
                        if (runtime.glare.valid) {
                            const float glare = runtime.glare.amount[idx];
                            xyz[0] += static_cast<double>(glare) * static_cast<double>(ctx.color->illuminantXYZ[0]);
                            xyz[1] += static_cast<double>(glare) * static_cast<double>(ctx.color->illuminantXYZ[1]);
                            xyz[2] += static_cast<double>(glare) * static_cast<double>(ctx.color->illuminantXYZ[2]);
                        }
                        double adapted[3];
                        mat3_mul_vec(cat02, xyz, adapted);
                        double rgbOut[3];
                        mat3_mul_vec(xyzToRgb, adapted, rgbOut);
                        if (!std::isfinite(rgbOut[0]) || !std::isfinite(rgbOut[1]) || !std::isfinite(rgbOut[2])) {
                            abortFlag.store(true, std::memory_order_relaxed);
                            failure.store(true, std::memory_order_relaxed);
                            break;
                        }
                        // agx-emulsion parity: keep signed linear RGB through optics; defer clipping to post-encoding.
                        rgbR[idx] = rgbOut[0];
                        rgbG[idx] = rgbOut[1];
                        rgbB[idx] = rgbOut[2];
                    }
                }
                });
        }

        for (std::thread& th : threads) {
            if (th.joinable()) th.join();
        }
        threads.clear();

        if (failure.load(std::memory_order_relaxed)) {
            throw OFX::Exception::Suite(kOfxStatErrFatal);
        }
        if (abortFlag.load(std::memory_order_relaxed)) {
            return;
        }

        // Stage B: lens blur
        if (ctx.options.lensBlurSigmaPx > 0.0f && runtime.blurKernel.size() > 1) {
            std::vector<double> tmp;
            blur_separable(rgbR, tmp, rgbR, width, height, runtime.blurKernel);
            blur_separable(rgbG, tmp, rgbG, width, height, runtime.blurKernel);
            blur_separable(rgbB, tmp, rgbB, width, height, runtime.blurKernel);
        }

        // Stage C: unsharp mask
        if (ctx.options.unsharpSigmaPx > 0.0f && std::isfinite(ctx.options.unsharpAmount) && ctx.options.unsharpAmount != 0.0f &&
            runtime.unsharpKernel.size() > 1) {
            std::vector<double> tmp;
            std::vector<double> blurred(total, 0.0);
            blur_separable(rgbR, tmp, blurred, width, height, runtime.unsharpKernel);
            for (size_t i = 0; i < total; ++i) {
                double v = rgbR[i] + static_cast<double>(ctx.options.unsharpAmount) * (rgbR[i] - blurred[i]);
                // agx-emulsion parity: allow overshoot/undershoot; clip only after encoding.
                rgbR[i] = std::isfinite(v) ? v : 0.0;
            }
            blur_separable(rgbG, tmp, blurred, width, height, runtime.unsharpKernel);
            for (size_t i = 0; i < total; ++i) {
                double v = rgbG[i] + static_cast<double>(ctx.options.unsharpAmount) * (rgbG[i] - blurred[i]);
                rgbG[i] = std::isfinite(v) ? v : 0.0;
            }
            blur_separable(rgbB, tmp, blurred, width, height, runtime.unsharpKernel);
            for (size_t i = 0; i < total; ++i) {
                double v = rgbB[i] + static_cast<double>(ctx.options.unsharpAmount) * (rgbB[i] - blurred[i]);
                rgbB[i] = std::isfinite(v) ? v : 0.0;
            }
        }

        // Stage D: write to destination with output encoding
        abortFlag.store(false, std::memory_order_relaxed);
        for (unsigned int t = 0; t < nThreads; ++t) {
            const int yStart = rowsPerThread * int(t);
            const int yEnd = std::min(height, rowsPerThread * int(t + 1));
            threads.emplace_back([&, yStart, yEnd]() {
                for (int yOff = yStart; yOff < yEnd && !abortFlag.load(std::memory_order_relaxed); ++yOff) {
                    if (ctx.abort.abortRequested()) {
                        abortFlag.store(true, std::memory_order_relaxed);
                        break;
                    }
                    const int y = originY + yOff;
                    const size_t rowOffset = size_t(yOff) * size_t(width);
                    for (int xOff = 0; xOff < width; ++xOff) {
                        if (abortFlag.load(std::memory_order_relaxed)) {
                            break;
                        }
                        const float* srcPix = nullptr;
                        if (ctx.srcImage) {
                            srcPix = reinterpret_cast<const float*>(ctx.srcImage->getPixelAddress(originX + xOff, y));
                        }
                        if (!ctx.dstView.r) {
                            continue;
                        }
                        const ptrdiff_t rowOffsetBytes =
                            ctx.dstView.strideBytes * static_cast<ptrdiff_t>(yOff);
                        float* dstRow = reinterpret_cast<float*>(
                            reinterpret_cast<char*>(ctx.dstView.r) + rowOffsetBytes);
                        float* dstPix = dstRow + size_t(xOff * ctx.nComponents);
                        const size_t idx = rowOffset + size_t(xOff);
                        double rgbOut[3] = { rgbR[idx], rgbG[idx], rgbB[idx] };
                        OutputEncoding::applyEncoding(ctx.color->encoding, rgbOut);
                        dstPix[0] = static_cast<float>(rgbOut[0]);
                        dstPix[1] = static_cast<float>(rgbOut[1]);
                        dstPix[2] = static_cast<float>(rgbOut[2]);
                        if (ctx.copyAlpha) {
                            dstPix[3] = (srcPix && ctx.copyAlpha) ? srcPix[3] : 1.0f;
                        }
                    }
                }
                });
        }

        for (std::thread& th : threads) {
            if (th.joinable()) th.join();
        }
    }

} // namespace ScannerOptics
