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
            if (!std::isfinite(v)) {
                return 0.0f;
            }
            if (v <= 0.0f) return 0.0f;
            if (v >= 1.0f) return 1.0f;
            return v;
        }

        inline float sanitizeSceneLinear(float v) {
            return std::isfinite(v) ? v : 0.0f;
        }

        inline float encode_gamma(float v, float exponent) {
            const float linear = clamp01(v);
            const float encoded = static_cast<float>(std::pow(linear, exponent));
            return clamp01(encoded);
        }

        inline float encode_sRGB(float v) {
            const float linear = clamp01(v);
            if (linear <= 0.0031308f) {
                return 12.92f * linear;
            }
            const float encoded = 1.055f * static_cast<float>(std::pow(linear, 1.0f / 2.4f)) - 0.055f;
            return clamp01(encoded);
        }

        inline float encode_BT2020(float v, float a, float b) {
            const float linear = clamp01(v);
            if (linear < b) {
                return clamp01(linear * 4.5f);
            }
            const float encoded = a * static_cast<float>(std::pow(linear, 0.45f)) - (a - 1.0f);
            return clamp01(encoded);
        }

        inline float encode_ProPhoto(float v, float threshold, float exponent) {
            const float linear = clamp01(v);
            if (linear < threshold) {
                return clamp01(linear * 16.0f);
            }
            const float encoded = static_cast<float>(std::pow(linear, exponent));
            return clamp01(encoded);
        }

        inline float encode_DaVinciIntermediate(float v, const GeneratedColorSpaces::CctfParams& cctf) {
            if (!std::isfinite(v)) {
                return 0.0f;
            }
            const float linear = std::max(0.0f, v);
            if (linear <= cctf.linearCutoff) {
                return clamp01(linear * cctf.d);
            }
            const float encoded = (static_cast<float>(std::log2(linear + cctf.a)) + cctf.b) * cctf.c;
            return clamp01(encoded);
        }

        inline float encode_channel(const GeneratedColorSpaces::CctfParams& cctf, float v) {
            using Kind = GeneratedColorSpaces::CctfKind;
            switch (cctf.kind) {
            case Kind::Linear:
                return sanitizeSceneLinear(v);
            case Kind::Gamma:
                return encode_gamma(v, cctf.gamma);
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
            linear[0] = sanitizeSceneLinear(rgb[0]);
            linear[1] = sanitizeSceneLinear(rgb[1]);
            linear[2] = sanitizeSceneLinear(rgb[2]);
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
            rgb[0] = clamp01(linear[0]);
            rgb[1] = clamp01(linear[1]);
            rgb[2] = clamp01(linear[2]);
        }
    }

} // namespace OutputEncoding
