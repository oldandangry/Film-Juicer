#include "output_encoding_oracle.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

// Preserve f64 literals versus promoted frozen f32 coefficients and native
// threshold/NaN/signed-zero/matrix-vector order. Candidate values never enter here.
namespace ScannerOracle {
    namespace {
        enum class CctfKind : std::uint8_t {
            Linear,
            Gamma,
            Srgb,
            Bt2020,
            ProPhoto,
            DavinciIntermediate
        };
        struct Cctf {
            CctfKind kind;
            float gamma, a, b, c, d, linearCutoff;
        };
        struct Entry {
            Cctf cctf;
            std::array<float, 9> dwgToOutput;
        };
#include "output_encoding_native.inc"
        inline double clamp01d(double v) {
            if (v <= 0.0)
                return 0.0;
            if (v >= 1.0)
                return 1.0;
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

        inline double encode_Bt2020d(double v, const Cctf& cctf) {
            if (v < static_cast<double>(cctf.b)) {
                return v * 4.5;
            }
            return static_cast<double>(cctf.a) * std::pow(v, 0.45) - (static_cast<double>(cctf.a) - 1.0);
        }

        inline double encode_ProPhotod(double v, const Cctf& cctf) {
            if (v < static_cast<double>(cctf.linearCutoff)) {
                return v * 16.0;
            }
            return std::pow(v, static_cast<double>(cctf.gamma));
        }

        inline double encode_DavinciIntermediated(double v, const Cctf& cctf) {
            const double linear = std::max(0.0, v);
            if (linear <= static_cast<double>(cctf.linearCutoff)) {
                return linear * static_cast<double>(cctf.d);
            }
            return (std::log2(linear + static_cast<double>(cctf.a)) + static_cast<double>(cctf.b)) *
                   static_cast<double>(cctf.c);
        }

        inline double encode_channel_double(const Cctf& cctf, double v) {
            using Kind = CctfKind;
            switch (cctf.kind) {
                case Kind::Linear:
                    return v;
                case Kind::Gamma:
                    return encode_gammad_signed(v, static_cast<double>(cctf.gamma));
                case Kind::Srgb:
                    return encode_sRGBd(v);
                case Kind::Bt2020:
                    return encode_Bt2020d(v, cctf);
                case Kind::ProPhoto:
                    return encode_ProPhotod(v, cctf);
                case Kind::DavinciIntermediate:
                    return encode_DavinciIntermediated(v, cctf);
                default:
                    return clamp01d(v);
            }
        }

    } // namespace

    void apply_encoding(const OutputEncoding::Params& params, double rgb[3]) {
        const auto& outSpace = kEntries[OutputEncoding::toIndex(OutputEncoding::colorSpaceFromIndex(OutputEncoding::toIndex(params.colorSpace)))];
        double linear[3];
        if (params.inputIsOutputSpace) {
            linear[0] = rgb[0];
            linear[1] = rgb[1];
            linear[2] = rgb[2];
        } else {
            const auto& m = outSpace.dwgToOutput;
            const double mat[9] = {
                static_cast<double>(m[0]), static_cast<double>(m[1]), static_cast<double>(m[2]), static_cast<double>(m[3]), static_cast<double>(m[4]), static_cast<double>(m[5]), static_cast<double>(m[6]), static_cast<double>(m[7]), static_cast<double>(m[8])};
            linear[0] = mat[0] * rgb[0] + mat[1] * rgb[1] + mat[2] * rgb[2];
            linear[1] = mat[3] * rgb[0] + mat[4] * rgb[1] + mat[5] * rgb[2];
            linear[2] = mat[6] * rgb[0] + mat[7] * rgb[1] + mat[8] * rgb[2];
        }

        if (params.applyCctfEncoding) {
            rgb[0] = encode_channel_double(outSpace.cctf, linear[0]);
            rgb[1] = encode_channel_double(outSpace.cctf, linear[1]);
            rgb[2] = encode_channel_double(outSpace.cctf, linear[2]);
        } else {
            rgb[0] = linear[0];
            rgb[1] = linear[1];
            rgb[2] = linear[2];
        }

        rgb[0] = clamp01d(rgb[0]);
        rgb[1] = clamp01d(rgb[1]);
        rgb[2] = clamp01d(rgb[2]);
    }

} // namespace ScannerOracle
