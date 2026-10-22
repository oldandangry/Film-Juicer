#include "RustScannerBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "Cuda/JuicerCudaExecutor.h"
#include "RenderRecipe.h"
#include "Scanner.h"

namespace JuicerScanner {
#if defined(JUICER_SCANNER_TEST_HOOK)
    namespace Test {
        thread_local Projection projection = nullptr;
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
