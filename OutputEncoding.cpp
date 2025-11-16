#include "OutputEncoding.h"

#include <algorithm>
#include <cmath>

namespace OutputEncoding {
    namespace {

        struct Mat3 {
            float m[9];
            void mul(const float in[3], float out[3]) const {
                out[0] = m[0] * in[0] + m[1] * in[1] + m[2] * in[2];
                out[1] = m[3] * in[0] + m[4] * in[1] + m[5] * in[2];
                out[2] = m[6] * in[0] + m[7] * in[1] + m[8] * in[2];
            }
        };

        constexpr Mat3 kDWG_to_BT709 = {
            {
                1.89861488f, -0.79217619f, -0.10643869f,
               -0.16894878f,  1.48897576f, -0.32002699f,
               -0.12153916f, -0.31567582f,  1.43721499f
            }
        };

        constexpr Mat3 kDWG_to_DCI_P3 = {
            {
                1.64213728f, -0.51275164f, -0.12938563f,
               -0.09589176f,  1.40815000f, -0.31225824f,
               -0.09146801f, -0.20840512f,  1.29987313f
            }
        };

        constexpr Mat3 kDWG_to_DisplayP3 = {
            {
                1.53154371f, -0.38718498f, -0.14435875f,
               -0.10031767f,  1.41325474f, -0.31293708f,
               -0.09046195f, -0.19316357f,  1.28362552f
            }
        };

        constexpr Mat3 kDWG_to_AdobeRGB = {
            {
                1.30961943f, -0.14233693f, -0.16728209f,
               -0.16894604f,  1.48897779f, -0.32002041f,
               -0.12349249f, -0.24139073f,  1.36487610f
            }
        };

        constexpr Mat3 kDWG_to_BT2020 = {
            {
                1.13030218f, -0.02039285f, -0.10990933f,
               -0.02554706f,  1.31084932f, -0.28530226f,
               -0.09259861f, -0.16465302f,  1.25725163f
            }
        };

        constexpr Mat3 kDWG_to_ProPhoto = {
            {
                0.93085883f,  0.03513168f,  0.03418863f,
                0.03394711f,  1.22406659f, -0.25805447f,
               -0.09352002f, -0.13118145f,  1.22497578f
            }
        };

        constexpr Mat3 kDWG_to_ACES2065 = {
            {
                0.74827029f,  0.16769466f,  0.08403505f,
                0.02084212f,  1.11190474f, -0.13274687f,
               -0.09151226f, -0.12774671f,  1.21925897f
            }
        };

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

        inline bool isSceneLinear(ColorSpace cs) {
            switch (cs) {
            case ColorSpace::ACES2065_1:
                return true;
            default:
                return false;
            }
        }

        inline float encode_Rec709(float v) {
            const float signal = clamp01(v);
            constexpr float kGamma = 2.4f; // Inverse of ITU-R BT.1886 EOTF (gamma 2.4)
            const float encoded = static_cast<float>(std::pow(signal, 1.0f / kGamma));
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

        inline float encode_DaVinciIntermediate(float v) {
            if (!std::isfinite(v)) {
                return 0.0f;
            }

            constexpr float kDI_A = 0.0075f;
            constexpr float kDI_B = 7.0f;
            constexpr float kDI_C = 0.07329248f;
            constexpr float kDI_M = 10.44426855f;
            constexpr float kDI_LIN_CUT = 0.00262409f;

            const float linear = std::max(0.0f, v);
            if (linear <= kDI_LIN_CUT) {
                return clamp01(linear * kDI_M);
            }

            const float encoded = (static_cast<float>(std::log2(linear + kDI_A)) + kDI_B) * kDI_C;
            return clamp01(encoded);
        }

        inline float encode_gamma(float v, float exponent) {
            const float linear = clamp01(v);
            const float encoded = static_cast<float>(std::pow(linear, exponent));
            return clamp01(encoded);
        }

        inline float encode_BT2020(float v) {
            const float linear = clamp01(v);
            constexpr float a = 1.09929681f;
            constexpr float b = 0.01805397f;
            if (linear < b) {
                return clamp01(linear * 4.5f);
            }
            const float encoded = a * static_cast<float>(std::pow(linear, 0.45f)) - (a - 1.0f);
            return clamp01(encoded);
        }

        inline float encode_ProPhoto(float v) {
            const float linear = clamp01(v);
            constexpr float threshold = 0.001953125f;
            if (linear < threshold) {
                return clamp01(linear * 16.0f);
            }
            const float encoded = static_cast<float>(std::pow(linear, 1.0f / 1.8f));
            return clamp01(encoded);
        }

        inline float applyEncodingChannel(ColorSpace cs, float v) {
            switch (cs) {
            case ColorSpace::sRGB:
                return encode_sRGB(v);
            case ColorSpace::Rec709:
                return encode_Rec709(v);
            case ColorSpace::DCI_P3:
                return encode_gamma(v, 1.0f / 2.6f);
            case ColorSpace::DisplayP3:
                return encode_sRGB(v);
            case ColorSpace::AdobeRGB:
                return encode_gamma(v, 1.0f / 2.19921875f);
            case ColorSpace::ITU_R_BT2020:
                return encode_BT2020(v);
            case ColorSpace::ProPhotoRGB:
                return encode_ProPhoto(v);
            case ColorSpace::ACES2065_1:
                return sanitizeSceneLinear(v);
            case ColorSpace::DaVinciWideGamutIntermediate:
                return encode_DaVinciIntermediate(v);
            default:
                return clamp01(v);
            }
        }

        void convertFromDWG(ColorSpace cs, const float in[3], float out[3]) {
            switch (cs) {
            case ColorSpace::sRGB:
            case ColorSpace::Rec709:
                kDWG_to_BT709.mul(in, out);
                break;
            case ColorSpace::DCI_P3:
                kDWG_to_DCI_P3.mul(in, out);
                break;
            case ColorSpace::DisplayP3:
                kDWG_to_DisplayP3.mul(in, out);
                break;
            case ColorSpace::AdobeRGB:
                kDWG_to_AdobeRGB.mul(in, out);
                break;
            case ColorSpace::ITU_R_BT2020:
                kDWG_to_BT2020.mul(in, out);
                break;
            case ColorSpace::ProPhotoRGB:
                kDWG_to_ProPhoto.mul(in, out);
                break;
            case ColorSpace::ACES2065_1:
                kDWG_to_ACES2065.mul(in, out);
                break;
            case ColorSpace::DaVinciWideGamutIntermediate:
                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                break;
            default:
                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                break;
            }
        }

        bool hasEncoding(ColorSpace cs) {
            switch (cs) {
            case ColorSpace::sRGB:
            case ColorSpace::Rec709:
            case ColorSpace::DCI_P3:
            case ColorSpace::DisplayP3:
            case ColorSpace::AdobeRGB:
            case ColorSpace::ITU_R_BT2020:
            case ColorSpace::ProPhotoRGB:
            case ColorSpace::ACES2065_1:
            case ColorSpace::DaVinciWideGamutIntermediate:
                return true;
            default:
                return false;
            }
        }
    }

    void applyEncoding(const Params& params, float rgb[3]) {
        float converted[3];
        convertFromDWG(params.colorSpace, rgb, converted);

        if (params.preserveLinearRange) {
            rgb[0] = converted[0];
            rgb[1] = converted[1];
            rgb[2] = converted[2];
            return;
        }

        if (params.applyCctfEncoding) {
            if (hasEncoding(params.colorSpace)) {
                rgb[0] = applyEncodingChannel(params.colorSpace, converted[0]);
                rgb[1] = applyEncodingChannel(params.colorSpace, converted[1]);
                rgb[2] = applyEncodingChannel(params.colorSpace, converted[2]);
            }
            else if (isSceneLinear(params.colorSpace)) {
                rgb[0] = sanitizeSceneLinear(converted[0]);
                rgb[1] = sanitizeSceneLinear(converted[1]);
                rgb[2] = sanitizeSceneLinear(converted[2]);
            }
            else {
                rgb[0] = clamp01(converted[0]);
                rgb[1] = clamp01(converted[1]);
                rgb[2] = clamp01(converted[2]);
            }
        }
        else {
            rgb[0] = clamp01(converted[0]);
            rgb[1] = clamp01(converted[1]);
            rgb[2] = clamp01(converted[2]);
        }
    }

} // namespace OutputEncoding
