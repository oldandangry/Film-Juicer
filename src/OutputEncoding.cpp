#include "OutputEncoding.h"

#include <algorithm>
#include <cmath>

#include "GeneratedColorSpaces.h"

namespace OutputEncoding {
    namespace {

        struct Mat3 {
            float m[9];
            void mul(const float v[3], float out[3]) const {
                out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
                out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
                out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
            }
        };

        inline Mat3 mat_mul(const float a[9], const float b[9]) {
            Mat3 out{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out.m[r * 3 + c] =
                        a[r * 3 + 0] * b[0 * 3 + c] +
                        a[r * 3 + 1] * b[1 * 3 + c] +
                        a[r * 3 + 2] * b[2 * 3 + c];
                }
            }
            return out;
        }

        inline float clamp01(float v) {
            if (v <= 0.0f) return 0.0f;
            if (v >= 1.0f) return 1.0f;
            return v;
        }

        inline float encode_gamma_signed(float v, float exponent) {
            const float mag = static_cast<float>(std::pow(std::abs(v), exponent));
            return std::copysign(mag, v);
        }

        inline float encode_sRGB(float v) {
            if (v <= 0.0031308f) {
                return 12.92f * v;
            }
            return 1.055f * static_cast<float>(std::pow(v, 1.0f / 2.4f)) - 0.055f;
        }

        inline float encode_BT2020(float v, float a, float b) {
            if (v < b) {
                return v * 4.5f;
            }
            return a * static_cast<float>(std::pow(v, 0.45f)) - (a - 1.0f);
        }

        inline float encode_ProPhoto(float v, float threshold, float exponent) {
            if (v < threshold) {
                return v * 16.0f;
            }
            return static_cast<float>(std::pow(v, exponent));
        }

        inline float encode_DaVinciIntermediate(float v, const GeneratedColorSpaces::CctfParams& cctf) {
            const float linear = std::max(0.0f, v);
            if (linear <= cctf.linearCutoff) {
                return linear * cctf.d;
            }
            return (static_cast<float>(std::log2(linear + cctf.a)) + cctf.b) * cctf.c;
        }

        inline float encode_channel(const GeneratedColorSpaces::CctfParams& cctf, float v) {
            using Kind = GeneratedColorSpaces::CctfKind;
            switch (cctf.kind) {
            case Kind::Linear:
                return v;
            case Kind::Gamma:
                return encode_gamma_signed(v, cctf.gamma);
            case Kind::SRGB:
                return encode_sRGB(v);
            case Kind::BT2020:
                return encode_BT2020(v, cctf.a, cctf.b);
            case Kind::ProPhoto:
                return encode_ProPhoto(v, cctf.linearCutoff, cctf.gamma);
            case Kind::DaVinciIntermediate:
                return encode_DaVinciIntermediate(v, cctf);
            default:
                return clamp01(v);
            }
        }

        inline double clamp01d(double v) {
            if (v <= 0.0) return 0.0;
            if (v >= 1.0) return 1.0;
            return v;
        }

        inline double encode_gammad_signed(double v, double exponent) {
            const double mag = std::pow(std::abs(v), exponent);
            return std::copysign(mag, v);
        }

        inline double encode_sRGBd(double v) {
            if (v <= 0.0031308) {
                return 12.92 * v;
            }
            return 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
        }

        inline double encode_BT2020d(double v, double a, double b) {
            if (v < b) {
                return v * 4.5;
            }
            return a * std::pow(v, 0.45) - (a - 1.0);
        }

        inline double encode_ProPhotod(double v, double threshold, double exponent) {
            if (v < threshold) {
                return v * 16.0;
            }
            return std::pow(v, exponent);
        }

        inline double encode_DaVinciIntermediated(double v, const GeneratedColorSpaces::CctfParams& cctf) {
            const double linear = std::max(0.0, v);
            if (linear <= static_cast<double>(cctf.linearCutoff)) {
                return linear * static_cast<double>(cctf.d);
            }
            return (std::log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) * static_cast<double>(cctf.c);
        }

        inline double encode_channel_double(const GeneratedColorSpaces::CctfParams& cctf, double v) {
            using Kind = GeneratedColorSpaces::CctfKind;
            switch (cctf.kind) {
            case Kind::Linear:
                return v;
            case Kind::Gamma:
                return encode_gammad_signed(v, static_cast<double>(cctf.gamma));
            case Kind::SRGB:
                return encode_sRGBd(v);
            case Kind::BT2020:
                return encode_BT2020d(v, static_cast<double>(cctf.a), static_cast<double>(cctf.b));
            case Kind::ProPhoto:
                return encode_ProPhotod(v, static_cast<double>(cctf.linearCutoff), static_cast<double>(cctf.gamma));
            case Kind::DaVinciIntermediate:
                return encode_DaVinciIntermediated(v, cctf);
            default:
                return clamp01d(v);
            }
        }

    } // namespace

    Matrix3x3 dwg_to_output_matrix(ColorSpace cs) {
        const auto& dwg = GeneratedColorSpaces::getDWG();
        const auto& out = GeneratedColorSpaces::get(cs);
        Matrix3x3 result{};
        Mat3 mat = mat_mul(out.xyzToRgb, dwg.rgbToXyz);
        for (int i = 0; i < 9; ++i) {
            result.m[i] = mat.m[i];
        }
        return result;
    }

    void applyEncoding(const Params& params, float rgb[3]) {
        const auto& outSpace = GeneratedColorSpaces::get(params.colorSpace);
        float linear[3];
        if (params.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        }
        else {
            const Matrix3x3 m = dwg_to_output_matrix(params.colorSpace);
            Mat3 tmp{ { m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8] } };
            tmp.mul(rgb, linear);
        }

        if (params.preserveLinearRange) {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
            return;
        }

        if (params.applyCctfEncoding) {
            rgb[0] = encode_channel(outSpace.cctf, linear[0]);
            rgb[1] = encode_channel(outSpace.cctf, linear[1]);
            rgb[2] = encode_channel(outSpace.cctf, linear[2]);
        }
        else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        // Single post-encoding clamp (agx parity: encode first, then clip)
        rgb[0] = clamp01(rgb[0]);
        rgb[1] = clamp01(rgb[1]);
        rgb[2] = clamp01(rgb[2]);
    }

    void applyEncoding(const Params& params, double rgb[3]) {
        const auto& outSpace = GeneratedColorSpaces::get(params.colorSpace);
        double linear[3];
        if (params.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        }
        else {
            const Matrix3x3 m = dwg_to_output_matrix(params.colorSpace);
            double mat[9] = {
                static_cast<double>(m.m[0]), static_cast<double>(m.m[1]), static_cast<double>(m.m[2]),
                static_cast<double>(m.m[3]), static_cast<double>(m.m[4]), static_cast<double>(m.m[5]),
                static_cast<double>(m.m[6]), static_cast<double>(m.m[7]), static_cast<double>(m.m[8])
            };
            linear[0] = mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2];
            linear[1] = mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2];
            linear[2] = mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2];
        }

        if (params.preserveLinearRange) {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
            return;
        }

        if (params.applyCctfEncoding) {
            rgb[0] = encode_channel_double(outSpace.cctf, linear[0]);
            rgb[1] = encode_channel_double(outSpace.cctf, linear[1]);
            rgb[2] = encode_channel_double(outSpace.cctf, linear[2]);
        }
        else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        rgb[0] = clamp01d(rgb[0]);
        rgb[1] = clamp01d(rgb[1]);
        rgb[2] = clamp01d(rgb[2]);
    }

} // namespace OutputEncoding
