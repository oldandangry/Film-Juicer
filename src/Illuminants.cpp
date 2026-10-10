#include "Illuminants.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "RustAssetBridge.h"
#include "Cuda/JuicerCudaExecutor.h"

namespace JuicerIlluminant {
    namespace {
        struct Diagnostic {
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        };
        bool available(FjStatus status, const Diagnostic& diagnostic, std::string_view label) {
            if (status.category == FJ_STATUS_SUCCESS) {
                return true;
            }
            if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
                if (JTRACE_ENABLED(1)) {
                    [[maybe_unused]] const std::string detail = "Illuminant preparation (" + std::string(label) + "): " + std::string(diagnostic.bytes.data(), diagnostic.error.length);
                    JTRACE("ILLUM", detail);
                }
                return false;
            }
            [[maybe_unused]] const std::string detail = "Illuminant preparation (" + std::string(label) + "): " + std::string(diagnostic.bytes.data(), diagnostic.error.length);
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, detail);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        void warning(const FjIlluminantCoverage& coverage, std::string_view label) {
            if (!coverage.warnings || !JTRACE_ENABLED(1)) {
                return;
            }
            std::ostringstream message;
            message << "CSV coverage warning (" << label << "): ";
            if (coverage.warnings & 1u) {
                message << "start=" << coverage.min_nm << "nm (need <= 370nm)";
            }
            if (coverage.warnings & 2u) {
                if (coverage.warnings & 1u) {
                    message << ", ";
                }
                message << "end=" << coverage.max_nm << "nm (need >= 790nm)";
            }
            JTRACE("ILLUM", message.str());
        }
        Spectral::Curve project(const FjIlluminant& value) {
            Spectral::Curve curve;
            Spectral::assign_reference_axis(curve.lambda_nm);
            curve.linear.assign(std::begin(value.samples), std::end(value.samples));
            return curve;
        }
    } // namespace

    Lens::Lens(FjIlluminantLens* owner) noexcept : _owner(owner) {}
    Lens::Lens(Lens&& lens) noexcept : _owner(std::exchange(lens._owner, nullptr)) {}
    Lens::~Lens() {
        if (_owner) {
            auto* owner = std::exchange(_owner, nullptr);
            Diagnostic diagnostic;
            const auto status = fj_legacy_illuminant_lens_release(&owner, &diagnostic.error);
            if (status.category != FJ_STATUS_SUCCESS) {
                try {
                    JTRACE("ILLUM", "illuminant lens cleanup failed");
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }
        }
    }
    Spectral::Curve from_samples(const JuicerAssets::CsvRows& rows, std::string_view label) {
        Diagnostic diagnostic;
        FjIlluminant value{};
        const auto status = fj_legacy_illuminant_from_samples(rows.view(), &value, &diagnostic.error);
        return available(status, diagnostic, label) ? project(value) : Spectral::Curve{};
    }
    Spectral::Curve blackbody(float temperatureKelvin) {
        Diagnostic diagnostic;
        FjIlluminant value{};
        const auto status = fj_legacy_illuminant_blackbody(temperatureKelvin, &value, &diagnostic.error);
        return available(status, diagnostic, "BB") ? project(value) : Spectral::Curve{};
    }
    Spectral::Curve equal_energy() {
        Diagnostic diagnostic;
        FjIlluminant value{};
        const auto status = fj_legacy_illuminant_equal_energy(&value, &diagnostic.error);
        return available(status, diagnostic, "EQUAL") ? project(value) : Spectral::Curve{};
    }
    Spectral::Curve tungsten_kg3(const JuicerAssets::CsvRows& rows, std::string_view label) {
        Diagnostic diagnostic;
        FjIlluminant value{};
        FjIlluminantCoverage coverage{};
        const auto status = fj_legacy_illuminant_tungsten_kg3(rows.view(), &value, &coverage, &diagnostic.error);
        warning(coverage, label);
        return available(status, diagnostic, label) ? project(value) : Spectral::Curve{};
    }
    std::optional<Lens> prepare_lens(const JuicerAssets::CsvRows& rows, std::string_view label) {
        Diagnostic diagnostic;
        FjIlluminantCoverage coverage{};
        FjIlluminantLens* owner = nullptr;
        const auto status = fj_legacy_illuminant_lens_prepare(rows.view(), &owner, &coverage, &diagnostic.error);
        Lens lens(owner);
        warning(coverage, label);
        if (!available(status, diagnostic, label)) {
            return std::nullopt;
        }
        return lens;
    }
    Spectral::Curve finish_lens(Lens lens, const JuicerAssets::CsvRows& rows, std::string_view label) {
        Diagnostic diagnostic;
        FjIlluminant value{};
        FjIlluminantCoverage coverage{};
        auto* owner = std::exchange(lens._owner, nullptr);
        const auto status = fj_legacy_illuminant_lens_finish(&owner, rows.view(), &value, &coverage, &diagnostic.error);
        warning(coverage, label);
        return available(status, diagnostic, label) ? project(value) : Spectral::Curve{};
    }
} // namespace JuicerIlluminant
