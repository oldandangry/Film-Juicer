#include "RustGamutBridge.h"

#include "RustColorBridge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <new>
#include <string>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "SpectralData.h"
#include "juicer_legacy_api.h"

namespace JuicerGamut {
    namespace {
        bool has_reference_samples(const Spectral::Curve& curve) {
            return curve.linear.size() == 81 && curve.lambda_nm.size() == 81;
        }
    } // namespace
    static_assert(sizeof(FjInputHullSource) == 64 && alignof(FjInputHullSource) == 8);
    static_assert(offsetof(FjInputHullSource, x) == 0 && offsetof(FjInputHullSource, y) == 16);
    static_assert(offsetof(FjInputHullSource, z) == 32 && offsetof(FjInputHullSource, d65) == 48);
    static_assert(sizeof(FjInputHull) == 8216 && alignof(FjInputHull) == 8);
    static_assert(offsetof(FjInputHull, center_xy) == 0 && offsetof(FjInputHull, xy) == 8 && offsetof(FjInputHull, hash) == 8208);

    bool input_hull(const Spectral::Curve& xBar,
                    const Spectral::Curve& yBar,
                    const Spectral::Curve& zBar,
                    const Spectral::Curve& d65,
                    Gamut::InputCompressionHull& out) {
        out = {};
        if (!has_reference_samples(xBar) || !has_reference_samples(yBar) ||
            !has_reference_samples(zBar) || !has_reference_samples(d65)) {
            out.diagnostic = "MalformedRequiredResource component=input_gamut_hull requirement=cie1931_380_700_and_d65";
            return false;
        }
        const FjInputHullSource source{{xBar.linear.data(), xBar.linear.size()},
                                       {yBar.linear.data(), yBar.linear.size()},
                                       {zBar.linear.data(), zBar.linear.size()},
                                       {d65.linear.data(), d65.linear.size()}};
        FjInputHull result{};
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const auto status = fj_legacy_gamut_input_hull(&source, &result, &error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            out.diagnostic.assign(bytes.data(), error.length);
            return false;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc();
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "Input hull preparation: " + std::string(bytes.data(), error.length));
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        std::copy_n(result.center_xy, 2, out.center.begin());
        for (std::size_t i = 0; i < out.xy.size(); ++i) {
            std::copy_n(result.xy + i * 2, 2, out.xy[i].begin());
        }
        out.hash = result.hash;
        out.valid = true;
        return true;
    }
} // namespace JuicerGamut

namespace JuicerGamut {
    static_assert(sizeof(FjOutputGamutSource) == 100 && alignof(FjOutputGamutSource) == 4);
    static_assert(offsetof(FjOutputGamutSource, output_space) == 0);
    static_assert(offsetof(FjOutputGamutSource, rgb_to_xyz) == 4);
    static_assert(offsetof(FjOutputGamutSource, xyz_to_rgb) == 40);
    static_assert(offsetof(FjOutputGamutSource, white_xyz) == 76);
    static_assert(offsetof(FjOutputGamutSource, d65_white_xyz) == 88);
    static_assert(sizeof(FjOutputGamutTransform) == 96 && alignof(FjOutputGamutTransform) == 8);
    static_assert(offsetof(FjOutputGamutTransform, native_rgb_to_d65_xyz) == 0);
    static_assert(offsetof(FjOutputGamutTransform, d65_xyz_to_native_rgb) == 36);
    static_assert(offsetof(FjOutputGamutTransform, output_space) == 72);
    static_assert(offsetof(FjOutputGamutTransform, hash) == 80);
    static_assert(offsetof(FjOutputGamutTransform, contract_hash) == 88);
    static_assert(sizeof(FjOutputBoundarySource) == 48 && alignof(FjOutputBoundarySource) == 8);
    static_assert(offsetof(FjOutputBoundarySource, output_space) == 0);
    static_assert(offsetof(FjOutputBoundarySource, d65_xyz_to_native_rgb) == 4);
    static_assert(offsetof(FjOutputBoundarySource, transform_hash) == 40);
    static_assert(sizeof(FjOutputBoundaryIdentity) == 24 && alignof(FjOutputBoundaryIdentity) == 8);
    static_assert(offsetof(FjOutputBoundaryIdentity, transform_hash) == 0);
    static_assert(offsetof(FjOutputBoundaryIdentity, contract_hash) == 8);
    static_assert(offsetof(FjOutputBoundaryIdentity, hash) == 16);
    static_assert(sizeof(FjOutputBoundaryFailure) == 12 && alignof(FjOutputBoundaryFailure) == 4);
    static_assert(offsetof(FjOutputBoundaryFailure, requirement) == 0);
    static_assert(offsetof(FjOutputBoundaryFailure, lightness_index) == 4);
    static_assert(offsetof(FjOutputBoundaryFailure, hue_index) == 8);

    bool output_transform(OutputEncoding::ColorSpace space, Gamut::OutputGamutTransform& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const int index = OutputEncoding::toIndex(space);
        if (index < 0 || index >= static_cast<int>(OutputEncoding::kColorSpaceCount)) {
            diagnostic = "ResourceDescriptorMismatch component=output_gamut_transform field=output_color_space";
            return false;
        }
        const auto output = JuicerColor::output_color(space);
        const auto d65 = JuicerColor::output_color(OutputEncoding::ColorSpace::sRGB);
        FjOutputGamutSource source{};
        source.output_space = static_cast<std::uint32_t>(index);
        std::copy_n(output.rgb_to_xyz, 9, source.rgb_to_xyz);
        std::copy_n(output.xyz_to_rgb, 9, source.xyz_to_rgb);
        std::copy_n(output.white_xyz, 3, source.white_xyz);
        std::copy_n(d65.white_xyz, 3, source.d65_white_xyz);
        FjOutputGamutTransform result{};
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const auto status = fj_legacy_gamut_output_transform(&source, &result, &error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            diagnostic.assign(bytes.data(), error.length);
            return false;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc{};
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "Output gamut transform: " + std::string(bytes.data(), error.length));
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        std::copy_n(result.native_rgb_to_d65_xyz, 9, out.nativeRgbToD65Xyz.begin());
        std::copy_n(result.d65_xyz_to_native_rgb, 9, out.d65XyzToNativeRgb.begin());
        out.outputColorSpace = space;
        out.hash = result.hash;
        out.contractHash = result.contract_hash;
        out.valid = true;
        return true;
    }

    bool output_boundary(const Gamut::OutputGamutTransform& transform, Gamut::OutputBoundaryTable& out) {
        out.cmax.fill(0.0f);
        out.transformHash = out.contractHash = out.hash = 0;
        out.valid = false;
        out.diagnostic.clear();
        const int index = OutputEncoding::toIndex(transform.outputColorSpace);
        if (!transform.valid || transform.hash == 0 || transform.contractHash == 0 || index < 0 || index >= static_cast<int>(OutputEncoding::kColorSpaceCount)) {
            out.diagnostic = "ResourceDescriptorMismatch component=output_boundary_table field=transform";
            return false;
        }
        FjOutputBoundarySource source{};
        source.output_space = static_cast<std::uint32_t>(index);
        std::copy(transform.d65XyzToNativeRgb.begin(), transform.d65XyzToNativeRgb.end(), source.d65_xyz_to_native_rgb);
        source.transform_hash = transform.hash;
        FjOutputBoundaryIdentity identity{};
        FjOutputBoundaryFailure detail{};
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const auto status = fj_legacy_gamut_output_boundary(&source, out.cmax.data(), out.cmax.size(), &identity, &detail, &error);
        if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
            switch (detail.requirement) {
                case 1:
                    out.diagnostic = "ResourceDescriptorMismatch component=output_boundary_table field=transform";
                    break;
                case 2:
                    out.diagnostic = "MalformedRequiredResource component=output_boundary_table requirement=c2_outside output_space=" + std::to_string(index) + " lightness_index=" + std::to_string(detail.lightness_index) + " hue_index=" + std::to_string(detail.hue_index);
                    break;
                case 3:
                    out.diagnostic = "MalformedRequiredResource component=output_boundary_table requirement=finite_first_exit";
                    break;
                case 4:
                    out.diagnostic = "ResourceDescriptorMismatch component=output_boundary_table field=identity";
                    break;
                default: {
                    JuicerCuda::Failure failure;
                    JuicerCuda::set_failure(failure, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Output boundary missing typed preparation failure");
                    throw JuicerCuda::ExecutionFailure{std::move(failure)};
                }
            }
            return false;
        }
        if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
            throw std::bad_alloc{};
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "Output boundary: " + std::string(bytes.data(), error.length));
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        out.transformHash = identity.transform_hash;
        out.contractHash = identity.contract_hash;
        out.hash = identity.hash;
        out.valid = true;
        return true;
    }
} // namespace JuicerGamut
