#include "RustScannerBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"
#include "RenderRecipe.h"
#include "Scanner.h"

namespace JuicerScanner {
#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        thread_local Projection projection = nullptr;
        thread_local ColorProjection colorProjection = nullptr;
        thread_local DescriptorProjection descriptorProjection = nullptr;
        void set_descriptor_projection(DescriptorProjection value) {
            descriptorProjection = value;
        }
        void set_color_projection(ColorProjection value) {
            colorProjection = value;
        }
        void set_projection(Projection value) {
            projection = value;
        }
    } // namespace Test
#endif
    namespace {
        [[noreturn]] void fail(FjStatus status, const char* diagnostic) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, diagnostic);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        [[noreturn]] void integrity() {
            fail({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Scanner correction preparation: invalid transport contract");
        }
        FjScannerCorrectionParams params(const RenderRecipe& recipe) {
            const auto& output = recipe.scannerOutput;
            return {static_cast<std::uint32_t>(recipe.profileRoute.scanRoute),
                    std::uint32_t(output.blackCorrection) | (std::uint32_t(output.whiteCorrection) << 1u),
                    output.blackLevel,
                    output.whiteLevel};
        }
        bool valid_envelope(FjStatus status, const FjScannerCorrectionFailure& failure) {
            if (status.api != FJ_API_NONE || status.native_code != 0)
                return false;
            if (failure.reason == FJ_SCANNER_CORRECTION_REASON_NONE)
                return status.category == FJ_STATUS_SUCCESS;
            if (failure.reason == FJ_SCANNER_CORRECTION_REASON_RAW)
                return status.category == FJ_STATUS_UNSUPPORTED_INPUT;
            if (failure.reason == FJ_SCANNER_CORRECTION_REASON_PANIC)
                return status.category == FJ_STATUS_INTERNAL_FAILURE;
            return failure.reason >= FJ_SCANNER_CORRECTION_REASON_DIRECT_INPUTS &&
                   failure.reason <= FJ_SCANNER_CORRECTION_REASON_IDENTITY &&
                   status.category == FJ_STATUS_PREPARATION_FAILURE;
        }
        template <std::size_t N>
        void admit_diagnostic(const FjErrorBuffer& error, const std::array<char, N>& text) {
            if (error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
                integrity();
        }
        std::optional<bool> demand(const FjScannerCorrectionParams& params, std::uint32_t operation, std::string& diagnostic) {
            std::array<char, 512> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            FjScannerCorrectionFailure failure{};
            std::uint32_t required = 0;
            const FjStatus status = fj_legacy_scanner_correction_input_demand(&params, operation, &required, &failure, &error);
            admit_diagnostic(error, text);
            if (!valid_envelope(status, failure) || required > 1 ||
                (status.category != FJ_STATUS_SUCCESS && required != 0))
                integrity();
            if (status.category == FJ_STATUS_SUCCESS) {
                if (error.length != 0)
                    integrity();
                return required != 0;
            }
            if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
                if (failure.reason != FJ_SCANNER_CORRECTION_REASON_DIRECT_INPUTS && failure.reason != FJ_SCANNER_CORRECTION_REASON_PRINT_INPUTS)
                    integrity();
                diagnostic.assign(text.data(), error.length);
                return std::nullopt;
            }
            fail(status, text.data());
        }

        std::array<float, 7> values(const FjScannerColorCorrection& value) {
            return {value.target_black_linear, value.target_white_linear, value.reference_black_y, value.reference_white_y, value.xyz_slope, value.xyz_offset, value.exposure_scale};
        }
        bool same(float left, float right) {
            return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
        }
        bool zero_result(const FjScannerColorCorrection& value) {
            if (value.route != 0 || value.active != 0 || value.black_correction != 0 || value.white_correction != 0 || value.schema_version != 0 || value.hash != 0)
                return false;
            const auto floats = values(value);
            return std::all_of(floats.begin(), floats.end(), [](float value) {
                return same(value, 0.0f);
            });
        }
        std::uint32_t failure_trace(std::uint32_t reason) {
            if (reason == FJ_SCANNER_CORRECTION_REASON_DIRECT_MIDGRAY || reason == FJ_SCANNER_CORRECTION_REASON_PRINT_MIDGRAY)
                return FJ_SCANNER_CORRECTION_TRACE_MIDGRAY;
            if (reason == FJ_SCANNER_CORRECTION_REASON_INVALID)
                return FJ_SCANNER_CORRECTION_TRACE_MALFORMED;
            if (reason == FJ_SCANNER_CORRECTION_REASON_IDENTITY)
                return FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH;
            return FJ_SCANNER_CORRECTION_TRACE_NONE;
        }
        bool zero_trace(const FjScannerCorrectionTrace& trace) {
            const std::array floats{trace.reference_black_y, trace.reference_white_y, trace.target_black_linear, trace.target_white_linear, trace.xyz_slope, trace.xyz_offset, trace.corrected_midgray, trace.exposure_scale};
            return trace.hash == 0 && std::all_of(floats.begin(), floats.end(), [](float value) {
                       return same(value, 0.0f);
                   });
        }
        bool finish(FjStatus status, FjScannerColorCorrection& wire, FjScannerCorrectionFailure& failure, FjScannerCorrectionTrace& trace, const FjScannerCorrectionParams& params, bool required, const FjErrorBuffer& error, Scanner::ScannerColorCorrectionDescriptor& out, std::string& diagnostic) {
#if defined(JUICER_SCANNER_TEST_HOOK)
            if (Test::projection)
                Test::projection(status, wire, failure, trace);
#endif
            if (!valid_envelope(status, failure))
                integrity();
            if (status.category != FJ_STATUS_SUCCESS) {
                const auto expectedTrace = failure_trace(failure.reason);
                if (!zero_result(wire) || trace.outcome != expectedTrace || trace.hash != 0 ||
                    (trace.outcome == FJ_SCANNER_CORRECTION_TRACE_NONE && !zero_trace(trace)))
                    integrity();
                if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
                    diagnostic.assign(error.data, error.length);
                    return false;
                }
                fail(status, error.data);
            }
            if (error.length != 0 || wire.route != params.route || wire.schema_version != 1 || wire.active != std::uint32_t(required) ||
                wire.black_correction > 1 || wire.white_correction > 1)
                integrity();
            if (required) {
                const auto floats = values(wire);
                if (wire.black_correction != (params.flags & 1u) || wire.white_correction != ((params.flags >> 1u) & 1u) || wire.hash == 0 ||
                    !std::all_of(floats.begin(), floats.end(), [](float value) {
                        return std::isfinite(value);
                    }) ||
                    !(wire.reference_white_y - wire.reference_black_y > 1.0e-6f) || !(wire.exposure_scale > 0.0f) || !std::isfinite(trace.corrected_midgray) || !(trace.corrected_midgray > 0.0f) || trace.outcome != FJ_SCANNER_CORRECTION_TRACE_ACCEPTED || trace.hash != wire.hash || !same(trace.target_black_linear, wire.target_black_linear) || !same(trace.target_white_linear, wire.target_white_linear) || !same(trace.reference_black_y, wire.reference_black_y) || !same(trace.reference_white_y, wire.reference_white_y) || !same(trace.xyz_slope, wire.xyz_slope) || !same(trace.xyz_offset, wire.xyz_offset) || !same(trace.exposure_scale, wire.exposure_scale))
                    integrity();
            } else {
                constexpr std::array defaults{0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
                const auto floats = values(wire);
                if (wire.black_correction != 0 || wire.white_correction != 0 || wire.hash != 0 ||
                    !std::equal(floats.begin(), floats.end(), defaults.begin(), same) ||
                    trace.outcome != FJ_SCANNER_CORRECTION_TRACE_NONE || !zero_trace(trace))
                    integrity();
            }
            out.active = wire.active != 0;
            out.blackCorrection = wire.black_correction != 0;
            out.whiteCorrection = wire.white_correction != 0;
            out.targetBlackLinear = wire.target_black_linear;
            out.targetWhiteLinear = wire.target_white_linear;
            out.referenceBlackY = wire.reference_black_y;
            out.referenceWhiteY = wire.reference_white_y;
            out.xyzSlope = wire.xyz_slope;
            out.xyzOffset = wire.xyz_offset;
            out.exposureScale = wire.exposure_scale;
            out.schemaVersion = wire.schema_version;
            out.hash = wire.hash;
            return true;
        }
        bool bind_tables(const Spectral::SpectralTables& source, FjScannerCorrectionTables& out) {
            if (source.K <= 0)
                return false;
            const auto count = static_cast<std::size_t>(source.K);
            if (source.epsC.size() < count || source.epsM.size() < count || source.epsY.size() < count || source.Ay.size() < count ||
                (source.hasBaseline && source.baseDensityMin.size() < count))
                return false;
            out = {{source.epsC.data(), count}, {source.epsM.data(), count}, {source.epsY.data(), count}, {source.Ay.data(), count}, source.hasBaseline ? FjFloatSpan{source.baseDensityMin.data(), count} : FjFloatSpan{}, source.K, source.invYn, std::uint32_t(source.hasBaseline)};
            return true;
        }
        std::vector<float> flatten(const std::vector<std::array<float, 3>>& rows) {
            std::vector<float> values;
            if (rows.size() > values.max_size() / 3u)
                throw std::bad_alloc{};
            values.reserve(rows.size() * 3u);
            for (const auto& row : rows)
                values.insert(values.end(), row.begin(), row.end());
            return values;
        }
        std::array<float, 243> flatten(const std::array<std::array<float, 3>, 81>& rows) {
            std::array<float, 243> values{};
            for (std::size_t sample = 0; sample < rows.size(); ++sample)
                for (std::size_t channel = 0; channel < 3; ++channel)
                    values[3u * sample + channel] = rows[sample][channel];
            return values;
        }
    } // namespace

    bool prepare_direct(const RenderRecipe& recipe, const Spectral::SpectralTables& scannerTables, Scanner::ScannerColorCorrectionDescriptor& out, FjScannerCorrectionTrace& trace, std::string& diagnostic) {
        out = Scanner::ScannerColorCorrectionDescriptor{};
        out.route = recipe.profileRoute.scanRoute;
        trace = {};
        diagnostic.clear();
        const auto requested = params(recipe);
        const auto required = demand(requested, 0, diagnostic);
        if (!required)
            return false;
        try {
            FjScannerCorrectionTables tables{};
            FjScannerDirectCorrectionInput input{};
            std::vector<float> curves;
            const FjScannerDirectCorrectionInput* active = nullptr;
            if (*required && recipe.profileRoute.filmProfile && bind_tables(scannerTables, tables)) {
                const auto& film = recipe.profileRoute.filmProfile->data;
                curves = flatten(recipe.filmDevelop.authoredDensityCurves);
                input.tables = &tables;
                input.film_base_density = {film.baseDensity.data(), film.baseDensity.size()};
                input.film_log_exposure = {film.logExposure.data(), film.logExposure.size()};
                input.film_authored_density_curves_cmy = {curves.data(), curves.size()};
                std::copy(recipe.densityBounds.dataMinCmy.begin(), recipe.densityBounds.dataMinCmy.end(), input.minimum_density_cmy);
                std::copy(recipe.densityBounds.dataMaxCmy.begin(), recipe.densityBounds.dataMaxCmy.end(), input.maximum_density_cmy);
                std::copy(recipe.densityBounds.invSpanCmy.begin(), recipe.densityBounds.invSpanCmy.end(), input.inverse_span_cmy);
                const auto& reference = recipe.scannerOutput.syntheticFilmReference;
                std::copy(reference.baselineDensityCmy.begin(), reference.baselineDensityCmy.end(), input.baseline_density_cmy);
                input.reference_hash = reference.hash;
                input.medium = static_cast<std::uint32_t>(recipe.densityBounds.medium);
                active = &input;
            }
            FjScannerColorCorrection wire{};
            FjScannerCorrectionFailure failure{};
            std::array<char, 512> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            FjStatus status = fj_legacy_scanner_direct_correction_prepare(&requested, active, &wire, &failure, &trace, &error);
            admit_diagnostic(error, text);
            return finish(status, wire, failure, trace, requested, *required, error, out, diagnostic);
        } catch (const std::bad_alloc&) {
            fail({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "Scanner correction transport allocation");
        }
    }

    bool prepare_print(const Scanner::PrintCorrectionDerivationInput& source, Scanner::ScannerColorCorrectionDescriptor& out, FjScannerCorrectionTrace& trace, std::string& diagnostic) {
        out = Scanner::ScannerColorCorrectionDescriptor{};
        trace = {};
        diagnostic.clear();
        if (!source.recipe) {
            diagnostic = "ResourceDescriptorMismatch phase=8B print correction recipe";
            return false;
        }
        const auto& recipe = *source.recipe;
        out.route = recipe.profileRoute.scanRoute;
        const auto requested = params(recipe);
        const auto required = demand(requested, 1, diagnostic);
        if (!required)
            return false;
        try {
            FjScannerCorrectionTables tables{};
            FjScannerPrintCorrectionInput input{};
            std::array<float, 243> dyes{}, sensitivity{};
            std::vector<float> curves;
            const FjScannerPrintCorrectionInput* active = nullptr;
            if (*required && recipe.profileRoute.filmProfile && recipe.profileRoute.printProfile && source.scannerTables &&
                source.mainIlluminant && source.preflashRawCmy && source.spectralSampleCount == 81 && bind_tables(*source.scannerTables, tables)) {
                const auto& film = recipe.profileRoute.filmProfile->data;
                const auto& print = recipe.profileRoute.printProfile->data;
                dyes = flatten(film.channelDensity);
                sensitivity = flatten(print.linearSensitivity);
                curves = flatten(print.densityCurves);
                input.tables = &tables;
                input.film_base_density = {film.baseDensity.data(), film.baseDensity.size()};
                input.film_channel_density_cmy = {dyes.data(), dyes.size()};
                input.print_sensitivity_cmy = {sensitivity.data(), sensitivity.size()};
                input.filtered_main_illuminant = {source.mainIlluminant, 81};
                input.print_base_density = {print.baseDensity.data(), print.baseDensity.size()};
                input.print_log_exposure = {print.logExposure.data(), print.logExposure.size()};
                input.print_density_curves_cmy = {curves.data(), curves.size()};
                std::copy(recipe.densityBounds.dataMinCmy.begin(), recipe.densityBounds.dataMinCmy.end(), input.minimum_density_cmy);
                std::copy(recipe.densityBounds.invSpanCmy.begin(), recipe.densityBounds.invSpanCmy.end(), input.inverse_span_cmy);
                std::copy(recipe.enlargerFilmBounds.dataMinCmy.begin(), recipe.enlargerFilmBounds.dataMinCmy.end(), input.film_minimum_density_cmy);
                std::copy(recipe.enlargerFilmBounds.dataMaxCmy.begin(), recipe.enlargerFilmBounds.dataMaxCmy.end(), input.film_maximum_density_cmy);
                std::copy_n(source.preflashRawCmy, 3, input.preflash_raw_cmy);
                input.normalizer = source.normalizer;
                input.preflash_exposure = recipe.print.exposure.preflashExposure;
                input.medium = static_cast<std::uint32_t>(recipe.densityBounds.medium);
                active = &input;
            }
            FjScannerColorCorrection wire{};
            FjScannerCorrectionFailure failure{};
            std::array<char, 512> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            FjStatus status = fj_legacy_scanner_print_correction_prepare(&requested, active, &wire, &failure, &trace, &error);
            admit_diagnostic(error, text);
            return finish(status, wire, failure, trace, requested, *required, error, out, diagnostic);
        } catch (const std::bad_alloc&) {
            fail({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "Scanner correction transport allocation");
        }
    }
} // namespace JuicerScanner

namespace JuicerScanner {
    void Lut::Release::operator()(FjScannerLutOwner* owner) const noexcept {
        (void)fj_legacy_scanner_lut_release(owner, nullptr);
    }
    Lut::Lut(std::unique_ptr<FjScannerLutOwner, Release> owner, FjScannerLutView view) noexcept
        : _owner(std::move(owner)), _view(view) {}
    Lut::Lut(Lut&& other) noexcept
        : _owner(std::move(other._owner)), _view(std::exchange(other._view, {})) {}
    Lut& Lut::operator=(Lut&& other) noexcept {
        if (this != &other) {
            _owner = std::move(other._owner);
            _view = std::exchange(other._view, {});
        }
        return *this;
    }

    std::optional<Lut> Lut::acquire(const Scanner::MediumView& medium, std::uint32_t resolution, JuicerCuda::Failure& failure) {
        const auto span = [](std::span<const float> values) -> FjFloatSpan {
            return {values.empty() ? nullptr : values.data(), values.size()};
        };
        const auto& t = medium.tables;
        FjScannerLutInput input{span(t.epsC), span(t.epsM), span(t.epsY), span(t.Ax), span(t.Ay), span(t.Az), span(t.baseDensityMin), {}, {}, t.K, t.invYn, t.hasBaseline ? 1u : 0u, static_cast<std::uint32_t>(medium.medium), resolution};
        std::copy_n(medium.range.min_cmy, 3, input.range_min_cmy);
        std::copy_n(medium.range.inv_max_cmy, 3, input.inverse_span_cmy);
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjScannerLutOwner* raw = nullptr;
        FjScannerLutFailure fields{};
        const FjStatus status = fj_legacy_scanner_lut_acquire(&input, &raw, &fields, &error);
        std::unique_ptr<FjScannerLutOwner, Release> owner(raw);
        const auto contract = [&]() {
            fail({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Scanner LUT preparation: invalid transport contract");
        };
        if (error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0' || status.api != FJ_API_NONE || status.native_code != 0)
            contract();
        if (status.category != FJ_STATUS_SUCCESS) {
            std::uint32_t category = FJ_STATUS_PREPARATION_FAILURE;
            switch (fields.reason) {
                case FJ_SCANNER_LUT_REASON_RAW:
                    category = FJ_STATUS_UNSUPPORTED_INPUT;
                    break;
                case FJ_SCANNER_LUT_REASON_SIZE:
                case FJ_SCANNER_LUT_REASON_ALLOCATION:
                    category = FJ_STATUS_ALLOCATION_FAILURE;
                    break;
                case FJ_SCANNER_LUT_REASON_PANIC:
                    category = FJ_STATUS_INTERNAL_FAILURE;
                    break;
                default:
                    break;
            }
            if (owner || fields.reason < 1 || fields.reason > 6 || status.category != category ||
                (fields.reason == FJ_SCANNER_LUT_REASON_NONFINITE ? fields.channel > 2 || fields.index == UINT64_MAX : fields.channel != UINT32_MAX || fields.index != UINT64_MAX) ||
                ((fields.reason == FJ_SCANNER_LUT_REASON_SIZE || fields.reason == FJ_SCANNER_LUT_REASON_ALLOCATION) ? fields.buffer < 1 || fields.buffer > 10 : fields.buffer != 0))
                contract();
            JuicerCuda::set_failure(failure, status, std::string_view(text.data(), error.length));
            return std::nullopt;
        }
        if (!owner || error.length || fields.reason || fields.channel != UINT32_MAX || fields.index != UINT64_MAX || fields.buffer)
            contract();
        FjScannerLutView view{};
        const FjStatus viewed = fj_legacy_scanner_lut_view(owner.get(), &view, &fields, &error);
        if (viewed.api != FJ_API_NONE || viewed.native_code != 0 || error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
            contract();
        const std::array arrays{view.log2_xyz, view.slope_c, view.slope_m, view.slope_y, view.cell_min, view.cell_max};
        if (viewed.category != FJ_STATUS_SUCCESS) {
            if (viewed.category != FJ_STATUS_INTERNAL_FAILURE || fields.reason != FJ_SCANNER_LUT_REASON_PANIC || fields.channel != UINT32_MAX || fields.index != UINT64_MAX || fields.buffer || view.resolution ||
                std::any_of(arrays.begin(), arrays.end(), [](FjFloatSpan span) {
                    return span.data || span.count;
                }))
                contract();
            JuicerCuda::set_failure(failure, viewed, std::string_view(text.data(), error.length));
            return std::nullopt;
        }
        if (error.length || fields.reason || fields.channel != UINT32_MAX || fields.index != UINT64_MAX || fields.buffer || view.resolution != resolution || resolution < 2)
            contract();
        const auto r = static_cast<std::size_t>(resolution);
        const auto nodes = r * r * r * 3u;
        const auto cells = (r - 1u) * (r - 1u) * (r - 1u) * 3u;
        for (std::size_t i = 0; i < arrays.size(); ++i)
            if (!arrays[i].data || reinterpret_cast<std::uintptr_t>(arrays[i].data) % alignof(float) || arrays[i].count != (i < 4 ? nodes : cells))
                contract();
        failure = {};
        return Lut{std::move(owner), view};
    }
    std::span<const float> Lut::log2_xyz() const noexcept {
        return {_view.log2_xyz.data, _view.log2_xyz.count};
    }
    std::span<const float> Lut::slope_c() const noexcept {
        return {_view.slope_c.data, _view.slope_c.count};
    }
    std::span<const float> Lut::slope_m() const noexcept {
        return {_view.slope_m.data, _view.slope_m.count};
    }
    std::span<const float> Lut::slope_y() const noexcept {
        return {_view.slope_y.data, _view.slope_y.count};
    }
    std::span<const float> Lut::cell_min() const noexcept {
        return {_view.cell_min.data, _view.cell_min.count};
    }
    std::span<const float> Lut::cell_max() const noexcept {
        return {_view.cell_max.data, _view.cell_max.count};
    }
} // namespace JuicerScanner

namespace Scanner {
    // SF_TEMP_BRIDGE_SCANNER_COLOR: bind one complete Rust value; remove S4.E.
    ColorRuntime build_color_runtime(ScannerMedium medium, const ScannerIlluminant& illuminant, const OutputEncoding::Params& outputEncoding, std::uint64_t activeOutputGamutRecipeHash) {
        const FjScannerColorInput input{static_cast<std::uint32_t>(medium),
                                        static_cast<std::uint32_t>(outputEncoding.colorSpace),
                                        std::uint32_t(outputEncoding.applyCctfEncoding),
                                        {illuminant.whiteXYZ[0], illuminant.whiteXYZ[1], illuminant.whiteXYZ[2]},
                                        illuminant.hash,
                                        activeOutputGamutRecipeHash};
        FjScannerColor wire{};
        FjScannerColorFailure failure{};
        std::array<char, 512> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjStatus status = fj_legacy_scanner_color_prepare(&input, &wire, &failure, &error);
#if defined(JUICER_SCANNER_TEST_HOOK)
        if (JuicerScanner::Test::colorProjection)
            JuicerScanner::Test::colorProjection(status, wire, failure);
#endif
        const auto invalid = [] {
            JuicerScanner::fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "scanner color projection invalid");
        };
        bool envelope = status.api == FJ_API_NONE && status.native_code == 0;
        switch (failure.reason) {
            case FJ_SCANNER_COLOR_REASON_NONE:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_SUCCESS;
                break;
            case FJ_SCANNER_COLOR_REASON_RAW:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                break;
            case FJ_SCANNER_COLOR_REASON_ILLUMINANT:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_PREPARATION_FAILURE;
                break;
            case FJ_SCANNER_COLOR_REASON_OUTPUT:
                envelope = envelope && failure.stage == 1 && status.category == FJ_STATUS_PREPARATION_FAILURE;
                break;
            case FJ_SCANNER_COLOR_REASON_BOUNDARY:
                envelope = envelope && (failure.stage == 1 || failure.stage == 2) && status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                break;
            case FJ_SCANNER_COLOR_REASON_PANIC:
                envelope = envelope && failure.stage == 0 && status.category == FJ_STATUS_INTERNAL_FAILURE;
                break;
            default:
                envelope = false;
                break;
        }
        if (!envelope || error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0')
            invalid();
        if (status.category != FJ_STATUS_SUCCESS) {
            const std::array<unsigned char, sizeof(FjScannerColor)> zeros{};
            if (std::memcmp(&wire, zeros.data(), zeros.size()) != 0)
                invalid();
            if (failure.reason == FJ_SCANNER_COLOR_REASON_ILLUMINANT) {
                JTRACE("HASH", "FATAL: scanner illuminant hash invalid for color runtime");
                return {};
            }
            if (failure.reason == FJ_SCANNER_COLOR_REASON_OUTPUT) {
                JTRACE("HASH", "FATAL: generated color space hash invalid");
                return {};
            }
            JuicerScanner::fail(status, text.data());
        }
        if (error.length != 0 || wire.output_space != input.output_space || wire.flags != (2u | input.apply_cctf) || wire.active_gamut_recipe_hash != input.active_gamut_recipe_hash || wire.encoding.schema != FJ_OUTPUT_ENCODING_SCHEMA || wire.encoding.cctf_kind > FJ_CCTF_DWG_INTERMEDIATE)
            invalid();
        ColorRuntime out{};
        std::copy_n(wire.cat02, 9, out.cat02);
        std::copy_n(wire.xyz_to_rgb, 9, out.xyzToRgb);
        std::copy_n(wire.illuminant_xyz, 3, out.illuminantXYZ);
        out.encoding = {static_cast<OutputEncoding::ColorSpace>(wire.output_space), (wire.flags & 1u) != 0, true};
        out.preparedEncoding = wire.encoding;
        out.outputGamutRecipeHash = wire.active_gamut_recipe_hash;
        out.hash = wire.hash;
        return out;
    }
} // namespace Scanner

namespace {
    [[noreturn]] void descriptor_projection_failure() {
        JuicerCuda::Failure failure;
        JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "scanner LUT descriptor projection invalid");
        throw JuicerCuda::ExecutionFailure{std::move(failure)};
    }

    bool describe_scanner_lut(const ProfileRoute& route, const DensityBoundsRecipe& bounds, const ScannerOutputRecipe& output, std::string_view observer, bool print, Scanner::ScannerSpectralLutDescriptor& descriptor, std::string& diagnostic) {
        const auto bytes = [](std::string_view value) -> FjByteSpan {
            return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()};
        };
        // Preserve the old native invalid-route exception after top-pointer admission.
        if (static_cast<unsigned>(route.scanRoute) > 3)
            static_cast<void>(Spektrafilm::scan_route_metadata(route.scanRoute));
        FjScannerLutDescriptorInput input{};
        input.mode = print ? 1u : 0u;
        input.route = static_cast<std::uint32_t>(route.scanRoute);
        input.capture_polarity = static_cast<std::uint32_t>(route.capturePolarity);
        input.bounds_medium = static_cast<std::uint32_t>(bounds.medium);
        input.resolution = output.lutResolution;
        input.density_bounds_hash = bounds.hash;
        input.viewing_illuminant = bytes(output.viewingIlluminant);
        input.observer_identity = bytes(observer);
        if (print) {
            input.bounds_source = static_cast<std::uint32_t>(bounds.source);
            input.output_medium = static_cast<std::uint32_t>(output.medium);
            input.print_profile_token = route.printProfileAssetVersionToken;
            input.print_profile_key = bytes(route.printProfileKey);
            if (route.printProfile) {
                input.profile_present = 1;
                input.channel_density_cmy = {&route.printProfile->data.channelDensity[0][0], 243};
                input.base_density = {route.printProfile->data.baseDensity.data(), 81};
                input.profile_viewing_illuminant = bytes(route.printProfile->info.viewingIlluminant.value);
            }
        } else if (route.filmProfile) {
            input.profile_present = 1;
            input.channel_density_cmy = {&route.filmProfile->data.channelDensity[0][0], 243};
            input.base_density = {route.filmProfile->data.baseDensity.data(), 81};
        }
        FjScannerLutDescriptor wire{};
        FjScannerLutDescriptorFailure failure{};
        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjStatus status = fj_legacy_scanner_lut_describe(&input, &wire, &failure, &error);
#if defined(JUICER_SCANNER_TEST_HOOK)
        if (JuicerScanner::Test::descriptorProjection)
            JuicerScanner::Test::descriptorProjection(status, wire, failure, error);
#endif
        if (status.api != FJ_API_NONE || status.native_code != 0 ||
            error.data != message.data() || error.capacity != message.size() ||
            error.length >= message.size() || message[error.length] != '\0')
            descriptor_projection_failure();
        const auto reason = failure.reason;
        const bool hashFailure = reason == (print ? FJ_SCANNER_LUT_DESCRIPTOR_PRINT_HASH : FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_HASH);
        const bool ordinary = print ? reason == FJ_SCANNER_LUT_DESCRIPTOR_PRINT_RECIPE || hashFailure
                                    : (reason >= FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_RECIPE && reason <= FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_ILLUMINANT) || hashFailure;
        const auto expectedCategory = reason == 0                                 ? FJ_STATUS_SUCCESS
                                      : reason == FJ_SCANNER_LUT_DESCRIPTOR_RAW   ? FJ_STATUS_UNSUPPORTED_INPUT
                                      : reason == FJ_SCANNER_LUT_DESCRIPTOR_PANIC ? FJ_STATUS_INTERNAL_FAILURE
                                      : ordinary                                  ? FJ_STATUS_PREPARATION_FAILURE
                                                                                  : UINT32_MAX;
        if (status.category != expectedCategory ||
            (hashFailure ? failure.lane < 1 || failure.lane > 5 : failure.lane != 0))
            descriptor_projection_failure();
        if (status.category != FJ_STATUS_SUCCESS) {
            const FjScannerLutDescriptor zero{};
            constexpr std::array messages{
                "", "invalid scanner LUT descriptor arguments", "ResourceDescriptorMismatch phase=3A scanner direct recipe unavailable", "ResourceDescriptorMismatch phase=3A scanner density bounds unavailable", "ResourceDescriptorMismatch phase=3A scanner observer identity unavailable", "ResourceDescriptorMismatch phase=3A scanner illuminant identity unavailable", "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid", "ResourceDescriptorMismatch phase=4C scanner print recipe", "ResourceDescriptorMismatch phase=4C scanner print descriptor hash", "scanner LUT descriptor preparation panicked"};
            if (std::memcmp(&wire, &zero, sizeof wire) != 0 || reason >= messages.size() ||
                std::string_view(message.data(), error.length) != messages[reason])
                descriptor_projection_failure();
            if (ordinary) {
                diagnostic.assign(message.data(), error.length);
                return false;
            }
            JuicerCuda::Failure rejected;
            JuicerCuda::set_failure(rejected, status, message.data());
            throw JuicerCuda::ExecutionFailure{std::move(rejected)};
        }
        if (reason != 0 || error.length != 0 || wire.route != input.route ||
            wire.medium != input.mode || wire.capture_polarity != input.capture_polarity ||
            wire.resolution != input.resolution || wire.density_bounds_hash != input.density_bounds_hash ||
            !wire.density_bounds_hash || !wire.channel_density_hash || !wire.base_density_hash ||
            !wire.viewing_illuminant_hash || !wire.observer_hash || !wire.hash)
            descriptor_projection_failure();
        descriptor.route = static_cast<Spektrafilm::ScanRoute>(wire.route);
        descriptor.medium = static_cast<Scanner::ScannedMediumKind>(wire.medium);
        descriptor.polarity = static_cast<Spektrafilm::ProfilePolarity>(wire.capture_polarity);
        descriptor.lutResolution = wire.resolution;
        descriptor.densityBoundsHash = wire.density_bounds_hash;
        descriptor.channelDensityHash = wire.channel_density_hash;
        descriptor.baseDensityHash = wire.base_density_hash;
        descriptor.scanIlluminantHash = wire.viewing_illuminant_hash;
        descriptor.observerHash = wire.observer_hash;
        descriptor.hash = wire.hash;
        return true;
    }
} // namespace

namespace Scanner {
    // SF_TEMP_BRIDGE_SCANNER_LUT_DESCRIPTOR: mechanical synchronous binding; remove S4.E.
    bool build_direct_scanner_spectral_lut_descriptor(const DirectScannerSpectralLutDescriptorInput& input, ScannerSpectralLutDescriptor& descriptor, std::string& diagnostic) {
        descriptor = ScannerSpectralLutDescriptor{};
        diagnostic.clear();
        if (!input.profileRoute || !input.densityBounds || !input.scannerOutput) {
            diagnostic = "ResourceDescriptorMismatch phase=3A scanner direct inputs unavailable";
            return false;
        }
        return describe_scanner_lut(*input.profileRoute, *input.densityBounds, *input.scannerOutput, input.observerIdentity, false, descriptor, diagnostic);
    }
    // SF_TEMP_BRIDGE_SCANNER_LUT_DESCRIPTOR: mechanical synchronous binding; remove S4.E.
    bool build_print_scanner_spectral_lut_descriptor(const PrintScannerSpectralLutDescriptorInput& input, ScannerSpectralLutDescriptor& descriptor, std::string& diagnostic) {
        descriptor = ScannerSpectralLutDescriptor{};
        diagnostic.clear();
        if (!input.profileRoute || !input.densityBounds || !input.scannerOutput) {
            diagnostic = "ResourceDescriptorMismatch phase=4C scanner print inputs unavailable";
            return false;
        }
        return describe_scanner_lut(*input.profileRoute, *input.densityBounds, *input.scannerOutput, input.observerIdentity, true, descriptor, diagnostic);
    }
} // namespace Scanner
