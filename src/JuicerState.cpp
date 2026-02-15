#include "JuicerState.h"

#include "LogExposureOffsets.h"
#include "RebuildWorkingStateInternals.h"
#include "WorkingStateCorePayload.h"
#include "WorkingStateCoreSharing.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
void JuicerCudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
    JuicerCuda::destroy(resources);
}
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <filesystem>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <sstream>
#include <system_error>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <iterator>

#include "nlohmann/json.hpp"

#include "Illuminants.h"
#include "IlluminantKeys.h"

namespace {
    namespace fs = std::filesystem;

    struct FilmStockDefinition {
        std::string optionLabel;
        std::string jsonKey;
    };

    struct PrintPaperDefinition {
        std::string optionLabel;
        std::string folderName;
        std::string jsonKey;
    };

    using FilmStockFallback = std::array<FilmStockDefinition, 5>;
    using PrintPaperFallback = std::array<PrintPaperDefinition, 2>;

    static const FilmStockFallback kFallbackFilmStocks{ {
        { "Vision3 250D", "kodak_vision3_250d_uc" },
        { "Vision3 50D",  "kodak_vision3_50d_uc" },
        { "Vision3 200T", "kodak_vision3_200t_uc" },
        { "Vision3 500T", "kodak_vision3_500t_uc" },
        { "Portra 400",   "kodak_portra_400_auc" }
    } };

    static const PrintPaperFallback kFallbackPrintPapers{ {
        { "2383", "kodak_2383", "kodak_2383_uc" },
        { "2393", "kodak_2393", "kodak_2393_uc" }
    } };

    std::vector<FilmStockDefinition>& film_stock_definitions() {
        static std::vector<FilmStockDefinition> defs;
        return defs;
    }

    std::vector<PrintPaperDefinition>& print_paper_definitions() {
        static std::vector<PrintPaperDefinition> defs;
        return defs;
    }

    std::once_flag gProfileCatalogOnce;

    struct FilterCatalog {
        std::vector<std::string> paperKeys;
        std::vector<std::string> filmKeys;
        std::unordered_set<std::string> paperKeySet;
        std::unordered_set<std::string> filmKeySet;
    };

    void trace_working_state_core_share(
        const WorkingStateSharing::AcquireCoreSharedResult& result,
        std::uint64_t buildCounter,
        const char* path)
    {
        if (!JTRACE_ENABLED(2) || result.keyHash == 0) {
            return;
        }
        std::string msg = std::string("event=core_share_shell")
            + " path=" + (path ? std::string(path) : std::string("unknown"))
            + " build=" + std::to_string(buildCounter)
            + " core_share_hash=" + std::to_string(result.keyHash)
            + " core_share_identity=" + std::to_string(result.identity)
            + " cache_hit=" + std::to_string(result.hit ? 1 : 0)
            + " cache_inserted=" + std::to_string(result.inserted ? 1 : 0)
            + " payload_present=" + std::to_string(result.payloadPresent ? 1 : 0)
            + " payload_backfilled=" + std::to_string(result.payloadBackfilled ? 1 : 0)
            + " cache_entries=" + std::to_string(static_cast<unsigned long long>(result.cacheEntries));
        JTRACE("MSWSC", msg);
    }

    void recompute_working_state_dir_overlay(InstanceState& S, const ParamSnapshot& P, WorkingState& target) {
        auto compute_curve_max = [](const Spectral::Curve& c) {
            float m = 0.0f;
            for (float v : c.linear) {
                if (std::isfinite(v) && v > m) {
                    m = v;
                }
            }
            if (!std::isfinite(m) || m <= 1e-4f) {
                m = 1.0f;
            }
            if (m > 1000.0f) {
                m = 1000.0f;
            }
            return m;
            };
        auto approx_equal_local = [](double a, double b, double eps = 1e-6) {
            return std::fabs(a - b) <= eps;
            };

        const Profiles::DirCouplersProfile& dirCfg = S.base.dirCouplers;

        int effectiveCouplersActive = P.couplersActive;
        double effectiveCouplersAmount = P.couplersAmount;
        double effectiveRatioB = P.ratioB;
        double effectiveRatioG = P.ratioG;
        double effectiveRatioR = P.ratioR;
        double effectiveCouplersSigma = P.sigma;
        double effectiveCouplersHigh = P.high;
        const bool spatialSigmaIsUiDefault = approx_equal_local(P.spatialSigmaMicrometers, kFactoryCouplersSpatialSigma);
        double effectiveSpatialSigma = P.spatialSigmaMicrometers;
        if (spatialSigmaIsUiDefault && S.couplerProfileSpatialSigmaValid) {
            effectiveSpatialSigma = S.couplerProfileSpatialSigmaMicrometers;
        }

#ifdef JUICER_ENABLE_COUPLERS
        if (dirCfg.hasData) {
            auto sanitize_profile = [](float value, double fallback, double lo, double hi) -> double {
                double v = static_cast<double>(value);
                if (!std::isfinite(v)) {
                    return fallback;
                }
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                return v;
                };

            if (!S.couplerDirty.active.load(std::memory_order_acquire) && effectiveCouplersActive == kFactoryCouplersActive) {
                effectiveCouplersActive = dirCfg.active ? 1 : 0;
            }
            if (!S.couplerDirty.amount.load(std::memory_order_acquire) && approx_equal_local(effectiveCouplersAmount, kFactoryCouplersAmount)) {
                effectiveCouplersAmount = sanitize_profile(dirCfg.amount, effectiveCouplersAmount, 0.0, 2.0);
            }
            if (!S.couplerDirty.ratioB.load(std::memory_order_acquire) && approx_equal_local(effectiveRatioB, kFactoryCouplersRatioB)) {
                effectiveRatioB = sanitize_profile(dirCfg.ratioRGB[0], effectiveRatioB, 0.0, 1.0);
            }
            if (!S.couplerDirty.ratioG.load(std::memory_order_acquire) && approx_equal_local(effectiveRatioG, kFactoryCouplersRatioG)) {
                effectiveRatioG = sanitize_profile(dirCfg.ratioRGB[1], effectiveRatioG, 0.0, 1.0);
            }
            if (!S.couplerDirty.ratioR.load(std::memory_order_acquire) && approx_equal_local(effectiveRatioR, kFactoryCouplersRatioR)) {
                effectiveRatioR = sanitize_profile(dirCfg.ratioRGB[2], effectiveRatioR, 0.0, 1.0);
            }
            if (!S.couplerDirty.sigma.load(std::memory_order_acquire) && approx_equal_local(effectiveCouplersSigma, kFactoryCouplersSigma)) {
                effectiveCouplersSigma = sanitize_profile(dirCfg.diffusionInterlayer, effectiveCouplersSigma, 0.0, 4.0);
            }
            if (!S.couplerDirty.high.load(std::memory_order_acquire) && approx_equal_local(effectiveCouplersHigh, kFactoryCouplersHigh)) {
                effectiveCouplersHigh = sanitize_profile(dirCfg.highExposureShift, effectiveCouplersHigh, 0.0, 1.0);
            }
            if (!S.couplerDirty.spatialSigma.load(std::memory_order_acquire) && spatialSigmaIsUiDefault) {
                const double profileSpatialSigma = S.couplerProfileSpatialSigmaValid
                    ? S.couplerProfileSpatialSigmaMicrometers
                    : static_cast<double>(dirCfg.diffusionSizeUm);
                effectiveSpatialSigma = sanitize_profile(static_cast<float>(profileSpatialSigma), effectiveSpatialSigma, 0.0, 50.0);
            }
        }
#endif

        const std::array<float, 3> densityMaxPostDir{
            compute_curve_max(target.densB),
            compute_curve_max(target.densG),
            compute_curve_max(target.densR)
        };

        bool precorrectApplied = false;
        Couplers::Runtime dirRT{};
        dirRT.active = (effectiveCouplersActive != 0);
        {
            auto clampRatio = [](double v) -> float {
                if (!std::isfinite(v) || v < 0.0) return 0.0f;
                if (v > 1.0) return 1.0f;
                return static_cast<float>(v);
                };
            auto clampAmount = [](double v) -> float {
                if (!std::isfinite(v) || v < 0.0) return 0.0f;
                if (v > 2.0) return 2.0f;
                return static_cast<float>(v);
                };
            const float amountScale = clampAmount(effectiveCouplersAmount);
            const float amount[3] = {
                amountScale * clampRatio(effectiveRatioB),
                amountScale * clampRatio(effectiveRatioG),
                amountScale * clampRatio(effectiveRatioR)
            };
#ifdef JUICER_ENABLE_COUPLERS
            Couplers::build_dir_matrix(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#else
            auto build_dir_matrix_stub = [](float M[3][3], const float amountValues[3], float layerSigma) {
                const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
                float amt[3] = { amountValues[0], amountValues[1], amountValues[2] };
                const float sigmaCapped = std::min(sigma, 3.0f);
                for (int i = 0; i < 3; ++i) {
                    if (!std::isfinite(amt[i])) amt[i] = 0.0f;
                    amt[i] = std::clamp(amt[i], 0.0f, 1.0f);
                }
                auto gauss = [sigmaCapped](int dx) -> float {
                    if (sigmaCapped <= 0.0f) {
                        return (dx == 0) ? 1.0f : 0.0f;
                    }
                    const float s2 = sigmaCapped * sigmaCapped;
                    return std::exp(-0.5f * (dx * dx) / s2);
                    };
                for (int r = 0; r < 3; ++r) {
                    float row[3];
                    float wsum = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        row[c] = gauss(c - r);
                        wsum += row[c];
                    }
                    if (wsum > 0.0f) {
                        for (int c = 0; c < 3; ++c) {
                            row[c] /= wsum;
                        }
                    }
                    for (int c = 0; c < 3; ++c) {
                        M[r][c] = amt[r] * row[c];
                    }
                }
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        if (!std::isfinite(M[r][c])) {
                            M[r][c] = 0.0f;
                        }
                    }
                }
                };
            build_dir_matrix_stub(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#endif
            dirRT.highShift = static_cast<float>(effectiveCouplersHigh);
            dirRT.spatialSigmaMicrometers = static_cast<float>(effectiveSpatialSigma);
            dirRT.spatialSigmaPixels = 0.0f;

#ifdef JUICER_ENABLE_COUPLERS
            if (dirRT.active) {
                auto has_nonfinite_density = [](const Spectral::Curve& c) -> bool {
                    for (float v : c.linear) {
                        if (!std::isfinite(v)) {
                            return true;
                        }
                    }
                    return false;
                    };

                if (has_nonfinite_density(target.densB) || has_nonfinite_density(target.densG) || has_nonfinite_density(target.densR)) {
                    precorrectApplied = false;
                }
                else {
                    Spectral::Curve densB_corr, densG_corr, densR_corr;
                    Couplers::precorrect_density_curves_before_DIR_into(
                        dirRT.M, dirRT.highShift,
                        target.densB, target.densG, target.densR,
                        densB_corr, densG_corr, densR_corr);
                    target.dirDensB = std::move(densB_corr);
                    target.dirDensG = std::move(densG_corr);
                    target.dirDensR = std::move(densR_corr);
                    precorrectApplied = true;
                }
            }
#endif
        }

        dirRT.dMax[0] = densityMaxPostDir[0];
        dirRT.dMax[1] = densityMaxPostDir[1];
        dirRT.dMax[2] = densityMaxPostDir[2];
        target.dirRT = dirRT;
        target.dirPrecorrected = precorrectApplied;
        if (!precorrectApplied) {
            target.dirDensB = target.densB;
            target.dirDensG = target.densG;
            target.dirDensR = target.densR;
        }
        target.dMax[0] = dirRT.dMax[0];
        target.dMax[1] = dirRT.dMax[1];
        target.dMax[2] = dirRT.dMax[2];
        target.negParams.DmaxY = target.dMax[0];
        target.negParams.DmaxM = target.dMax[1];
        target.negParams.DmaxC = target.dMax[2];

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float v = target.dirRT.M[r][c];
                if (!std::isfinite(v)) v = 0.0f;
                if (v < -10.0f) v = -10.0f;
                if (v > 10.0f)  v = 10.0f;
                target.dirRT.M[r][c] = v;
            }
        }
        for (int i = 0; i < 3; ++i) {
            float v = target.dMax[i];
            if (!std::isfinite(v) || v <= 1e-4f) v = 1.0f;
            if (v > 1000.0f) v = 1000.0f;
            target.dMax[i] = v;
            target.dirRT.dMax[i] = v;
        }
    }

    bool rebuild_working_state_scanner_output_runtime(const ParamSnapshot& P, WorkingState& target) {
        const bool printRuntimeOk = target.printScannerValid;
        if (!target.negativeScannerValid) {
            JTRACE("HASH", "FATAL: negative scanner runtime marked invalid");
            return false;
        }

        OutputEncoding::Params scannerEncoding{};
        scannerEncoding.colorSpace = OutputEncoding::colorSpaceFromIndex(P.outputColorSpace);
        scannerEncoding.applyCctfEncoding = (P.outputCctfEncoding != 0);
        scannerEncoding.preserveLinearRange = (P.outputLinearPassThrough != 0);
        scannerEncoding.inputIsOutputSpace = true;

        const std::uint32_t lutRes =
            static_cast<std::uint32_t>(std::clamp(P.scannerLutResolution, 17, 128));
        const std::uint64_t negGlareHash = Scanner::hash_glare(target.negativeGlare);
        const std::uint64_t printGlareHash = printRuntimeOk ? Scanner::hash_glare(target.printGlare) : 0;
        if (negGlareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash negative glare parameters");
            return false;
        }
        if (printRuntimeOk && printGlareHash == 0) {
            JTRACE("HASH", "FATAL: failed to hash print glare parameters");
            return false;
        }
        if (target.tablesScan.tablesHash == 0) {
            JTRACE("HASH", "FATAL: scanner table hash invalid for negative medium");
            return false;
        }
        if (printRuntimeOk && target.tablesPrint.tablesHash == 0) {
            JTRACE("HASH", "FATAL: scanner table hash invalid for print medium");
            return false;
        }

        target.negativeMediumRuntime = Scanner::ScannerMediumRuntime{};
        target.negativeMediumRuntime.medium = Scanner::ScannerMedium::Negative;
        target.negativeMediumRuntime.tables = (target.tablesScan.K > 0) ? &target.tablesScan : nullptr;
        target.negativeMediumRuntime.range = target.negativeDensityRange;
        target.negativeMediumRuntime.illuminant = target.negativeScannerIlluminant;
        target.negativeMediumRuntime.glare = target.negativeGlare;

        target.printMediumRuntime = Scanner::ScannerMediumRuntime{};
        target.printMediumRuntime.medium = Scanner::ScannerMedium::Print;
        target.printMediumRuntime.tables = (target.tablesPrint.K > 0) ? &target.tablesPrint : nullptr;
        target.printMediumRuntime.range = target.printDensityRange;
        target.printMediumRuntime.illuminant = target.printScannerIlluminant;
        target.printMediumRuntime.glare = target.printGlare;

        target.negativeColorRuntime = ScannerOptics::build_color_runtime(
            target.negativeMediumRuntime,
            scannerEncoding);
        if (target.negativeColorRuntime.hash == 0) {
            JTRACE("HASH", "FATAL: scanner color runtime hash invalid for negative medium");
            return false;
        }
        if (printRuntimeOk) {
            target.printColorRuntime = ScannerOptics::build_color_runtime(
                target.printMediumRuntime,
                scannerEncoding);
            if (target.printColorRuntime.hash == 0) {
                JTRACE("HASH", "FATAL: scanner color runtime hash invalid for print medium");
                return false;
            }
        }
        else {
            target.printColorRuntime = Scanner::ColorRuntime{};
        }

        target.negativeStaticKey = Scanner::ScannerStaticKey{};
        target.negativeStaticKey.medium = Scanner::ScannerMedium::Negative;
        target.negativeStaticKey.tablesHash = target.tablesScan.tablesHash;
        target.negativeStaticKey.densityRangeHash = target.negativeDensityRange.digest;
        target.negativeStaticKey.glareHash = negGlareHash;
        target.negativeStaticKey.colorRuntimeHash = target.negativeColorRuntime.hash;
        target.negativeStaticKey.lutResolution = lutRes;
        Scanner::finalize_static_key(target.negativeStaticKey);

        target.printStaticKey = Scanner::ScannerStaticKey{};
        if (printRuntimeOk) {
            target.printStaticKey.medium = Scanner::ScannerMedium::Print;
            target.printStaticKey.tablesHash = target.tablesPrint.tablesHash;
            target.printStaticKey.densityRangeHash = target.printDensityRange.digest;
            target.printStaticKey.glareHash = printGlareHash;
            target.printStaticKey.colorRuntimeHash = target.printColorRuntime.hash;
            target.printStaticKey.lutResolution = lutRes;
            Scanner::finalize_static_key(target.printStaticKey);
        }

        target.negativeMediumRuntime.color = &target.negativeColorRuntime;
        target.negativeMediumRuntime.staticKey = target.negativeStaticKey;

        target.printMediumRuntime.color = printRuntimeOk ? &target.printColorRuntime : nullptr;
        target.printMediumRuntime.staticKey = target.printStaticKey;

        const float glareCompensationFactor = target.printRT
            ? target.printRT->profile.glare.compensationRemovalFactor
            : target.printGlare.compensationRemovalFactor;
        target.printGlareCompensated = (printRuntimeOk && glareCompensationFactor > 0.0f);
        return true;
    }

    std::string sanitize_identifier(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (char ch : value) {
            unsigned char uc = static_cast<unsigned char>(ch);
            if (std::isalnum(uc)) {
                out.push_back(static_cast<char>(std::tolower(uc)));
            }
        }
        return out;
    }

    bool equals_ignore_case(const std::string& a, const std::string& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) !=
                std::tolower(static_cast<unsigned char>(b[i]))) {
                return false;;
            }
        }
        return true;
    }

    FilterCatalog load_filter_catalog(const fs::path& filterPath) {
        FilterCatalog catalog;
        std::error_code ec;
        if (!fs::exists(filterPath, ec) || fs::is_directory(filterPath, ec)) {
            return catalog;
        }

        std::ifstream file(filterPath, std::ios::binary);
        if (!file.is_open()) {
            return catalog;
        }

        nlohmann::json root = nlohmann::json::parse(file, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            return catalog;
        }

        for (auto it = root.begin(); it != root.end(); ++it) {
            if (!it.value().is_object()) {
                continue;
            }
            const std::string paperKey = it.key();
            if (catalog.paperKeySet.insert(paperKey).second) {
                catalog.paperKeys.push_back(paperKey);
            }
            for (auto illumIt = it.value().begin(); illumIt != it.value().end(); ++illumIt) {
                if (!illumIt.value().is_object()) {
                    continue;
                }
                for (auto filmIt = illumIt.value().begin(); filmIt != illumIt.value().end(); ++filmIt) {
                    const std::string filmKey = filmIt.key();
                    if (catalog.filmKeySet.insert(filmKey).second) {
                        catalog.filmKeys.push_back(filmKey);
                    }
                }
            }
        }
        return catalog;
    }

    void populate_profile_catalogs() {
        auto& filmDefs = film_stock_definitions();
        auto& paperDefs = print_paper_definitions();
        filmDefs.clear();
        paperDefs.clear();

        fs::path base = fs::path(gDataDir);
        fs::path profilesDir = base / "profiles";
        fs::path paperDir = base / "paper";

        std::unordered_map<std::string, Profiles::ProfileInfoSummary> infoByKey;
        std::vector<std::string> missingFilmKeys;
        std::vector<std::string> missingPaperKeys;
        std::error_code ec;
        if (!gDataDir.empty() && fs::exists(profilesDir, ec) && fs::is_directory(profilesDir, ec)) {
            for (fs::directory_iterator it(profilesDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (!it->is_regular_file(ec)) {
                    continue;
                }
                if (it->path().extension() != ".json") {
                    continue;
                }
                Profiles::ProfileInfoSummary info;
                const std::string jsonPath = it->path().string();
                if (Profiles::load_profile_info(jsonPath, info)) {
                    infoByKey[info.stock] = std::move(info);
                }
            }
        }

        FilterCatalog filters = load_filter_catalog(profilesDir / "enlarger_neutral_ymc_filters.json");

        auto pushFilm = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                missingFilmKeys.push_back(key + " (profile missing)");
                return;
            }
            if (!equals_ignore_case(it->second.type, "negative")) {
                std::string reason = key + " (type='" + it->second.type + "')";
                missingFilmKeys.push_back(std::move(reason));
                return;
            }
            std::string label = it->second.name.empty() ? it->second.stock : it->second.name;
            filmDefs.push_back({ std::move(label), it->second.stock });
            };

        for (const std::string& key : filters.filmKeys) {
            pushFilm(key);
        }

        if (filmDefs.empty()) {
            for (const auto& pair : infoByKey) {
                if (!equals_ignore_case(pair.second.type, "negative")) {
                    continue;
                }
                std::string label = pair.second.name.empty() ? pair.second.stock : pair.second.name;
                filmDefs.push_back({ std::move(label), pair.second.stock });
            }
            std::sort(filmDefs.begin(), filmDefs.end(),
                [](const FilmStockDefinition& a, const FilmStockDefinition& b) {
                    return a.optionLabel < b.optionLabel;
                });
        }

        if (filmDefs.empty()) {
            if (!missingFilmKeys.empty()) {
                std::ostringstream oss;
                oss << "catalog fallback: film profiles unavailable for keys: ";
                for (size_t i = 0; i < missingFilmKeys.size(); ++i) {
                    if (i > 0) {
                        oss << ", ";
                    }
                    oss << missingFilmKeys[i];
                }
                JTRACE("CATALOG", oss.str());
            }
            else {
                JTRACE("CATALOG", "catalog fallback: no film profiles discovered; using defaults");
            }
            filmDefs.assign(kFallbackFilmStocks.begin(), kFallbackFilmStocks.end());
        }

        struct PrintFolderInfo {
            std::string name;
            std::string sanitized;
            bool used = false;
        };

        std::vector<PrintFolderInfo> folders;
        if (!gDataDir.empty() && fs::exists(paperDir, ec) && fs::is_directory(paperDir, ec)) {
            for (fs::directory_iterator it(paperDir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                if (it->is_directory(ec)) {
                    std::string folder = it->path().filename().string();
                    if (!folder.empty()) {
                        folders.push_back({ folder, sanitize_identifier(folder), false });
                    }
                }
            }
        }

        auto claimFolder = [&](const Profiles::ProfileInfoSummary& info, const std::string& key) -> std::string {
            std::string keySan = sanitize_identifier(key);
            std::string nameSan = sanitize_identifier(info.name);
            size_t bestScore = 0;
            int bestIndex = -1;
            for (size_t i = 0; i < folders.size(); ++i) {
                if (folders[i].used) {
                    continue;
                }
                const std::string& folderSan = folders[i].sanitized;
                if (folderSan.empty()) {
                    continue;
                }
                size_t score = 0;
                bool match = false;
                if (!keySan.empty() && keySan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 4;
                }
                if (!match && !nameSan.empty() && nameSan.find(folderSan) != std::string::npos) {
                    match = true;
                    score = folderSan.size() * 3;
                }
                if (!match && !keySan.empty() && folderSan.find(keySan) != std::string::npos) {
                    match = true;
                    score = keySan.size() * 2;
                }
                if (!match && !nameSan.empty() && folderSan.find(nameSan) != std::string::npos) {
                    match = true;
                    score = nameSan.size();
                }
                if (match && score > bestScore) {
                    bestScore = score;
                    bestIndex = static_cast<int>(i);
                }
            }
            if (bestIndex >= 0) {
                folders[bestIndex].used = true;
                return folders[bestIndex].name;
            }
            return {};
            };

        auto pushPaper = [&](const std::string& key) {
            auto it = infoByKey.find(key);
            if (it == infoByKey.end()) {
                missingPaperKeys.push_back(key + " (profile missing)");
                return;
            }
            const auto& info = it->second;
            if (!equals_ignore_case(info.type, "paper")) {
                std::string reason = key + " (type='" + info.type + "')";
                missingPaperKeys.push_back(std::move(reason));
                return;
            }
            std::string folder = claimFolder(info, key);
            std::string label = info.name.empty() ? key : info.name;
            paperDefs.push_back({ std::move(label), std::move(folder), key });
            };

        for (const std::string& key : filters.paperKeys) {
            pushPaper(key);
        }

        if (paperDefs.empty()) {
            for (auto& folderInfo : folders) {
                if (folderInfo.used || folderInfo.sanitized.empty()) {
                    continue;
                }
                std::string bestKey;
                Profiles::ProfileInfoSummary bestInfo;
                size_t bestScore = 0;
                for (const auto& pair : infoByKey) {
                    if (!equals_ignore_case(pair.second.type, "paper")) {
                        continue;
                    }
                    std::string keySan = sanitize_identifier(pair.first);
                    std::string nameSan = sanitize_identifier(pair.second.name);
                    size_t score = 0;
                    bool match = false;
                    if (!keySan.empty() && keySan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 4;
                    }
                    if (!match && !nameSan.empty() && nameSan.find(folderInfo.sanitized) != std::string::npos) {
                        match = true;
                        score = folderInfo.sanitized.size() * 3;
                    }
                    if (!match && !keySan.empty() && folderInfo.sanitized.find(keySan) != std::string::npos) {
                        match = true;
                        score = keySan.size() * 2;
                    }
                    if (!match && !nameSan.empty() && folderInfo.sanitized.find(nameSan) != std::string::npos) {
                        match = true;
                        score = nameSan.size();
                    }
                    if (match && score > bestScore) {
                        bestScore = score;
                        bestKey = pair.first;
                        bestInfo = pair.second;
                    }
                }
                if (!bestKey.empty()) {
                    folderInfo.used = true;
                    std::string label = folderInfo.name;
                    paperDefs.push_back({ std::move(label), folderInfo.name, bestKey });
                }
            }
        }

        if (paperDefs.empty()) {
            if (!missingPaperKeys.empty()) {
                std::ostringstream oss;
                oss << "catalog fallback: print profiles unavailable for keys: ";
                for (size_t i = 0; i < missingPaperKeys.size(); ++i) {
                    if (i > 0) {
                        oss << ", ";
                    }
                    oss << missingPaperKeys[i];
                }
                JTRACE("CATALOG", oss.str());
            }
            else {
                JTRACE("CATALOG", "catalog fallback: no print profiles discovered; using defaults");
            }
            paperDefs.assign(kFallbackPrintPapers.begin(), kFallbackPrintPapers.end());
        }
    }

    void ensure_profile_catalogs() {
        std::call_once(gProfileCatalogOnce, populate_profile_catalogs);
        auto& films = film_stock_definitions();
        auto& papers = print_paper_definitions();
        if (films.empty()) {
            films.assign(kFallbackFilmStocks.begin(), kFallbackFilmStocks.end());
        }
        if (papers.empty()) {
            papers.assign(kFallbackPrintPapers.begin(), kFallbackPrintPapers.end());
        }
    }

    static const FilmStockDefinition& film_stock_for_index(int filmIndex) {
        ensure_profile_catalogs();
        auto& films = film_stock_definitions();
        if (films.empty()) {
            static const FilmStockDefinition dummy{ "", "" };
            return dummy;
        }
        if (filmIndex < 0 || filmIndex >= static_cast<int>(films.size())) {
            filmIndex = 0;
        }
        return films[filmIndex];
    }

    static const PrintPaperDefinition& print_paper_for_index(int index) {
        ensure_profile_catalogs();
        auto& papers = print_paper_definitions();
        if (papers.empty()) {
            static const PrintPaperDefinition dummy{ "", "", "" };
            return dummy;
        }
        if (index < 0 || index >= static_cast<int>(papers.size())) {
            index = 0;
        }
        return papers[index];
    }

    inline bool approx_equal(double a, double b, double eps = 1e-6) {
        return std::fabs(a - b) <= eps;
    }

    static Spectral::Curve build_blackbody_curve(float temperature) {
        Spectral::Curve curve;
        if (!(temperature > 0.0f)) {
            return curve;
        }
        Spectral::assign_reference_axis(curve.lambda_nm);
        curve.linear.resize(static_cast<size_t>(Spectral::gShape.K));
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            curve.linear[static_cast<size_t>(i)] = Spectral::planck_blackbody(curve.lambda_nm[static_cast<size_t>(i)], temperature);
        }
        Spectral::mean_power_normalize(curve.linear);
        return curve;
    }

    static std::string make_data_subpath(
        const std::string& baseDir,
        std::initializer_list<std::string_view> segments)
    {
        fs::path path(baseDir);
        for (std::string_view seg : segments) {
            if (!seg.empty()) {
                path /= fs::path(seg);
            }
        }
        path = path.lexically_normal();
        path.make_preferred();
        return path.string();
    }

    static Spectral::Curve build_illuminant_from_string(
        const std::string& dataDir,
        const std::string& source)
    {
        const std::string normalized = IlluminantKeys::normalize(source);

        auto build_or_log = [&](auto builder, const char* label) -> Spectral::Curve {
            try {
                return builder();
            }
            catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): " << e.what();
                JTRACE("ILLUM", oss.str());
            }
            catch (...) {
                std::ostringstream oss;
                oss << "failed to load illuminant '" << source << "' (" << label
                    << "): unknown error";
                JTRACE("ILLUM", oss.str());
            }
            return Spectral::Curve{};
            };

        if (IlluminantKeys::matches_any(normalized, { "D65" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D65_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D65.csv" }));
                }, "D65");
        }
        if (IlluminantKeys::matches_any(normalized, { "D55" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D55_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D55.csv" }));
                }, "D55");
        }
        if (IlluminantKeys::matches_any(normalized, { "D50" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_D50_pinned(
                    make_data_subpath(dataDir, { "illuminants", "D50.csv" }));
                }, "D50");
        }
        if (IlluminantKeys::matches_any(normalized, { "TH-KG3-L", "THKG3L", "TH-KG3L" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_TH_KG3_L_pinned(
                    make_data_subpath(dataDir, { "filters", "heat_absorbing", "schott", "KG3.csv" }),
                    make_data_subpath(dataDir, { "filters", "lens_transmission", "canon", "canon_24_f28_is.csv" }));
                }, "TH-KG3-L");
        }
        if (IlluminantKeys::matches_any(normalized, { "T", "INCANDESCENT" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_T_pinned(
                    make_data_subpath(dataDir, { "illuminants", "T.csv" }));
                }, "T");
        }
        if (IlluminantKeys::matches_any(normalized, { "K75P", "KINOTON75P" })) {
            return build_or_log([&]() {
                return Spectral::build_curve_K75P_pinned(
                    make_data_subpath(dataDir, { "illuminants", "K75P.csv" }));
                }, "K75P");
        }
        if (IlluminantKeys::matches_any(normalized, { "EQUAL", "EQUALENERGY", "EQUAL-ENERGY" })) {
            return Spectral::build_curve_equal_energy_pinned();
        }

        if (normalized.size() > 2 && normalized[0] == 'B' && normalized[1] == 'B') {
            const char* start = normalized.c_str() + 2;
            while (*start != '\0' && std::isspace(static_cast<unsigned char>(*start))) {
                ++start;
            }
            char* endPtr = nullptr;
            const double temperature = std::strtod(start, &endPtr);
            while (endPtr && *endPtr != '\0' && std::isspace(static_cast<unsigned char>(*endPtr))) {
                ++endPtr;
            }
            if (start != endPtr && endPtr && *endPtr == '\0' && temperature > 0.0) {
                return build_blackbody_curve(static_cast<float>(temperature));
            }
        }

        if (!normalized.empty()) {
            std::ostringstream oss;
            oss << "unknown illuminant string '" << source << "'; returning empty curve";
            JTRACE("ILLUM", oss.str());
        }
        return Spectral::Curve{};
    }

    static bool curve_matches_reference_axis(const Spectral::Curve& curve) {
        const size_t expected = static_cast<size_t>(Spectral::gShape.K);
        if (curve.linear.size() != expected || curve.lambda_nm.size() != expected) {
            return false;
        }
        constexpr float kAxisMatchTolerance = 1e-3f;
        for (size_t i = 0; i < expected; ++i) {
            const float lambda = curve.lambda_nm[i];
            if (!std::isfinite(lambda) ||
                std::abs(lambda - Spectral::gShape.wavelengths[i]) > kAxisMatchTolerance) {
                return false;
            }
        }
        return true;
    }

    static bool build_scanner_illuminant(
        const std::string& dataDir,
        const std::string& source,
        const char* label,
        Scanner::ScannerIlluminant& out)
    {
        out = Scanner::ScannerIlluminant{};
        if (source.empty()) {
            std::ostringstream oss;
            oss << "FATAL: missing viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }

        Spectral::Curve curve = build_illuminant_from_string(dataDir, source);
        if (!curve_matches_reference_axis(curve)) {
            std::ostringstream oss;
            oss << "FATAL: viewing illuminant '" << source
                << "' for " << label << " not pinned to agx axis";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        const int K = Spectral::gShape.K;
        const auto& xBar = Spectral::gXBar.linear;
        const auto& yBar = Spectral::gYBar.linear;
        const auto& zBar = Spectral::gZBar.linear;
        if (xBar.size() != static_cast<size_t>(K) ||
            yBar.size() != static_cast<size_t>(K) ||
            zBar.size() != static_cast<size_t>(K)) {
            std::ostringstream oss;
            oss << "FATAL: CMFs unavailable for " << label << " (K mismatch)";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;
        for (int i = 0; i < K; ++i) {
            const float spd = curve.linear[static_cast<size_t>(i)];
            const float xb = xBar[static_cast<size_t>(i)];
            const float yb = yBar[static_cast<size_t>(i)];
            const float zb = zBar[static_cast<size_t>(i)];
            if (!(std::isfinite(spd) && std::isfinite(xb) && std::isfinite(yb) && std::isfinite(zb))) {
                std::ostringstream oss;
                oss << "FATAL: non-finite CMF/SPD sample in " << label << " illuminant";
                JTRACE("ILLUM", oss.str());
                return false;
            }
            sumX += static_cast<double>(spd) * static_cast<double>(xb);
            sumY += static_cast<double>(spd) * static_cast<double>(yb);
            sumZ += static_cast<double>(spd) * static_cast<double>(zb);
        }

        if (!(std::isfinite(sumY) && sumY > 0.0)) {
            std::ostringstream oss;
            oss << "FATAL: invalid luminance sum for " << label << " (Yn=" << sumY << ")";
            JTRACE("ILLUM", oss.str());
            return false;
        }

        out.curve = std::move(curve);
        out.normalization = static_cast<float>(sumY);
        const double invYn = 1.0 / sumY;
        out.whiteXYZ[0] = static_cast<float>(sumX * invYn);
        out.whiteXYZ[1] = 1.0f;
        out.whiteXYZ[2] = static_cast<float>(sumZ * invYn);

        const double whiteSum = sumX + sumY + sumZ;
        if (!(std::isfinite(whiteSum) && whiteSum > 0.0)) {
            JTRACE("ILLUM", "FATAL: invalid white sum while building scanner illuminant");
            return false;
        }
        out.whiteXY[0] = static_cast<float>(sumX / whiteSum);
        out.whiteXY[1] = static_cast<float>(sumY / whiteSum);

        std::vector<float> hashSamples = out.curve.linear;
        hashSamples.push_back(out.normalization);
        out.hash = Hash::hash_float_span(hashSamples.data(), hashSamples.size());
        if (out.hash == 0) {
            std::ostringstream oss;
            oss << "FATAL: failed to hash viewing illuminant for " << label;
            JTRACE("ILLUM", oss.str());
            return false;
        }
        return true;
    }

    static bool nanmax_curve(const Spectral::Curve& curve, float& outMax) {
        if (curve.linear.empty()) {
            return false;
        }
        double m = -std::numeric_limits<double>::infinity();
        bool found = false;
        for (float v : curve.linear) {
            if (std::isfinite(v)) {
                m = std::max(m, static_cast<double>(v));
                found = true;
            }
        }
        if (!found || !std::isfinite(m)) {
            return false;
        }
        outMax = static_cast<float>(m);
        return std::isfinite(outMax);
    }

    static bool compute_negative_density_range(
        const Spectral::Curve& densB,
        const Spectral::Curve& densG,
        const Spectral::Curve& densR,
        const Profiles::GrainMetadata& grain,
        Scanner::ScannerDensityRange& outRange)
    {
        outRange = Scanner::ScannerDensityRange{};
        for (int i = 0; i < 3; ++i) {
            const float v = grain.densityMin[static_cast<size_t>(i)];
            if (!std::isfinite(v)) {
                JTRACE("BUILD", "FATAL: non-finite grain density_min for negative medium");
                return false;
            }
            outRange.min_cmy[i] = v;
        }

        float maxC = 0.0f, maxM = 0.0f, maxY = 0.0f;
        const bool okC = nanmax_curve(densR, maxC);
        const bool okM = nanmax_curve(densG, maxM);
        const bool okY = nanmax_curve(densB, maxY);
        if (!(okC && okM && okY)) {
            JTRACE("BUILD", "FATAL: failed to capture negative density maxima (post-glare)");
            return false;
        }

        outRange.max_cmy[0] = maxC + outRange.min_cmy[0];
        outRange.max_cmy[1] = maxM + outRange.min_cmy[1];
        outRange.max_cmy[2] = maxY + outRange.min_cmy[2];

        for (int i = 0; i < 3; ++i) {
            const float v = outRange.max_cmy[i];
            if (!(std::isfinite(v) && v > 0.0f)) {
                JTRACE("BUILD", "FATAL: invalid negative density range (non-positive max)");
                return false;
            }
            outRange.inv_max_cmy[i] = 1.0f / v;
        }

        float hashVals[6] = {
            outRange.min_cmy[0], outRange.min_cmy[1], outRange.min_cmy[2],
            outRange.max_cmy[0], outRange.max_cmy[1], outRange.max_cmy[2]
        };
        outRange.digest = Hash::hash_float_span(hashVals, std::size(hashVals));
        if (outRange.digest == 0) {
            JTRACE("HASH", "FATAL: failed to hash negative density range");
            return false;
        }
        return true;
    }

    static bool compute_print_density_range(
        const Print::Profile& profile,
        Scanner::ScannerDensityRange& outRange)
    {
        outRange = Scanner::ScannerDensityRange{};
        float maxC = 0.0f, maxM = 0.0f, maxY = 0.0f;
        const bool okC = nanmax_curve(profile.dcC, maxC);
        const bool okM = nanmax_curve(profile.dcM, maxM);
        const bool okY = nanmax_curve(profile.dcY, maxY);
        if (!(okC && okM && okY)) {
            JTRACE("BUILD", "FATAL: failed to capture print density maxima (post-glare)");
            return false;
        }

        outRange.max_cmy[0] = maxC;
        outRange.max_cmy[1] = maxM;
        outRange.max_cmy[2] = maxY;

        for (int i = 0; i < 3; ++i) {
            const float v = outRange.max_cmy[i];
            if (!(std::isfinite(v) && v > 0.0f)) {
                JTRACE("BUILD", "FATAL: invalid print density range (non-positive max)");
                return false;
            }
            outRange.inv_max_cmy[i] = 1.0f / v;
        }

        float hashVals[6] = {
            outRange.min_cmy[0], outRange.min_cmy[1], outRange.min_cmy[2],
            outRange.max_cmy[0], outRange.max_cmy[1], outRange.max_cmy[2]
        };
        outRange.digest = Hash::hash_float_span(hashVals, std::size(hashVals));
        if (outRange.digest == 0) {
            JTRACE("HASH", "FATAL: failed to hash print density range");
            return false;
        }
        return true;
    }
}

uint64_t hash_params(const ParamSnapshot& p) {
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
        };
    uint64_t h = 0;
    h = mix(h, static_cast<uint64_t>(p.filmStockIndex));
    h = mix(h, static_cast<uint64_t>(p.printPaperIndex));
    h = mix(h, static_cast<uint64_t>(p.spectralUpsamplingMode));
    h = mix(h, static_cast<uint64_t>(p.refIll));
    h = mix(h, static_cast<uint64_t>(p.enlIll));
    h = mix(h, static_cast<uint64_t>(p.enlDichroicSet));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalDensity * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalTransition * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.printDminFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.couplersActive));
    h = mix(h, static_cast<uint64_t>(p.couplersAmount * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioR * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioG * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioB * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.sigma * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.high * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.spatialSigmaMicrometers * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.scannerLutResolution));
    h = mix(h, static_cast<uint64_t>(p.inputColorSpace));
    h = mix(h, static_cast<uint64_t>(p.inputCctfDecoding));
    h = mix(h, static_cast<uint64_t>(p.outputColorSpace));
    h = mix(h, static_cast<uint64_t>(p.outputCctfEncoding));
    h = mix(h, static_cast<uint64_t>(p.outputLinearPassThrough));
    h = mix(h, static_cast<uint64_t>(p.cameraFilterOverride ? 1 : 0));
    if (p.cameraFilterOverride) {
        auto mix_triplet = [&](const std::array<double, 3>& triplet) {
            for (double v : triplet) {
                if (std::isfinite(v)) {
                    const int64_t scaled = static_cast<int64_t>(std::llround(v * 10000.0));
                    h = mix(h, static_cast<uint64_t>(scaled));
                }
            }
            };
        mix_triplet(p.cameraFilterUV);
        mix_triplet(p.cameraFilterIR);
    }
    return h;
}

uint64_t hash_params_core(const ParamSnapshot& p) {
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
        };
    uint64_t h = 0;
    h = mix(h, static_cast<uint64_t>(p.filmStockIndex));
    h = mix(h, static_cast<uint64_t>(p.printPaperIndex));
    h = mix(h, static_cast<uint64_t>(p.spectralUpsamplingMode));
    h = mix(h, static_cast<uint64_t>(p.refIll));
    h = mix(h, static_cast<uint64_t>(p.enlIll));
    h = mix(h, static_cast<uint64_t>(p.enlDichroicSet));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalDensity * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalTransition * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.printDminFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.scannerLutResolution));
    h = mix(h, static_cast<uint64_t>(p.inputColorSpace));
    h = mix(h, static_cast<uint64_t>(p.inputCctfDecoding));
    h = mix(h, static_cast<uint64_t>(p.outputColorSpace));
    h = mix(h, static_cast<uint64_t>(p.outputCctfEncoding));
    h = mix(h, static_cast<uint64_t>(p.outputLinearPassThrough));
    h = mix(h, static_cast<uint64_t>(p.cameraFilterOverride ? 1 : 0));
    if (p.cameraFilterOverride) {
        auto mix_triplet = [&](const std::array<double, 3>& triplet) {
            for (double v : triplet) {
                if (std::isfinite(v)) {
                    const int64_t scaled = static_cast<int64_t>(std::llround(v * 10000.0));
                    h = mix(h, static_cast<uint64_t>(scaled));
                }
            }
            };
        mix_triplet(p.cameraFilterUV);
        mix_triplet(p.cameraFilterIR);
    }
    return h;
}

static uint64_t hash_params_upload_core(const ParamSnapshot& p) {
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
        };
    uint64_t h = 0;
    h = mix(h, static_cast<uint64_t>(p.filmStockIndex));
    h = mix(h, static_cast<uint64_t>(p.printPaperIndex));
    h = mix(h, static_cast<uint64_t>(p.spectralUpsamplingMode));
    h = mix(h, static_cast<uint64_t>(p.refIll));
    h = mix(h, static_cast<uint64_t>(p.enlIll));
    h = mix(h, static_cast<uint64_t>(p.enlDichroicSet));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalDensity * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.glareCompRemovalTransition * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.printDminFactor * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.cameraFilterOverride ? 1 : 0));
    if (p.cameraFilterOverride) {
        auto mix_triplet = [&](const std::array<double, 3>& triplet) {
            for (double v : triplet) {
                if (std::isfinite(v)) {
                    const int64_t scaled = static_cast<int64_t>(std::llround(v * 10000.0));
                    h = mix(h, static_cast<uint64_t>(scaled));
                }
            }
            };
        mix_triplet(p.cameraFilterUV);
        mix_triplet(p.cameraFilterIR);
    }
    return h;
}

uint64_t hash_params_dir(const ParamSnapshot& p) {
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
        };
    uint64_t h = 0;
    h = mix(h, static_cast<uint64_t>(p.couplersActive));
    h = mix(h, static_cast<uint64_t>(p.couplersAmount * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioR * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioG * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioB * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.sigma * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.high * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.spatialSigmaMicrometers * 10000.0));
    return h;
}

std::string print_dir_for_index(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    if (paper.folderName.empty() || gDataDir.empty()) {
        return std::string();
    }

    std::filesystem::path base = std::filesystem::path(gDataDir);
    std::filesystem::path dir = base / "paper" / std::filesystem::path(paper.folderName);
    dir.make_preferred();
    std::string result = dir.string();
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif
    if (!result.empty() && result.back() != separator) {
        result.push_back(separator);
    }
    return result;
}

const char* print_paper_json_key_for_index(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    return paper.jsonKey.empty() ? nullptr : paper.jsonKey.c_str();
}

const char* negative_json_key_for_stock_index(int filmIndex) {
    const FilmStockDefinition& stock = film_stock_for_index(filmIndex);
    return stock.jsonKey.empty() ? nullptr : stock.jsonKey.c_str();
}

int film_stock_option_count() {
    ensure_profile_catalogs();
    return static_cast<int>(film_stock_definitions().size());
}

const char* film_stock_option_label(int index) {
    const FilmStockDefinition& stock = film_stock_for_index(index);
    return stock.optionLabel.empty() ? "" : stock.optionLabel.c_str();
}

int print_paper_option_count() {
    ensure_profile_catalogs();
    return static_cast<int>(print_paper_definitions().size());
}

const char* print_paper_option_label(int index) {
    const PrintPaperDefinition& paper = print_paper_for_index(index);
    return paper.optionLabel.empty() ? "" : paper.optionLabel.c_str();
}

bool load_film_stock_into_base(int filmIndex, InstanceState& S) {
    const FilmStockDefinition& stock = film_stock_for_index(filmIndex);
    JTRACE_SCOPE("STOCK", std::string("load_film_stock_into_base: ") + stock.jsonKey);

    std::vector<std::pair<float, float>> c_data;
    std::vector<std::pair<float, float>> m_data;
    std::vector<std::pair<float, float>> y_data;

    std::vector<std::pair<float, float>> r_sens;
    std::vector<std::pair<float, float>> g_sens;
    std::vector<std::pair<float, float>> b_sens;

    std::vector<std::pair<float, float>> dmin;
    std::vector<std::pair<float, float>> dmid;

    std::vector<std::pair<float, float>> dc_r;
    std::vector<std::pair<float, float>> dc_g;
    std::vector<std::pair<float, float>> dc_b;
    std::array<std::array<std::vector<std::pair<float, float>>, 3>, 3> dc_layers{};

    S.base.densityMidNeutral.clear();
    S.base.logExposureMidNeutral.clear();
    S.base.referenceIlluminant.clear();
    S.base.viewingIlluminant.clear();
    S.filmReferenceIlluminant.clear();
    S.base.dyeDensityMinFactor = 1.0f;
    S.base.cameraFilterUV = { {1.0f, 410.0f, 8.0f} };
    S.base.cameraFilterIR = { {1.0f, 675.0f, 15.0f} };
    S.base.cameraFilterDefined = false;
    S.base.grain = Profiles::GrainMetadata{};
    S.base.halation = Profiles::HalationMetadata{};
    S.base.glare = Profiles::ProfileGlare{};
    S.base.hasDensityCurvesLayers = false;
    for (auto& layer : S.base.densityCurvesLayers) {
        for (auto& ch : layer) ch.clear();
    }

    if (stock.jsonKey.empty()) {
        JTRACE("STOCK", "film stock missing JSON key; cannot load profile");
        return false;
    }
    const std::string jsonPath = data_dir_string("profiles", stock.jsonKey + ".json");
    Profiles::AgxFilmProfile profile;
    if (!Profiles::load_agx_film_profile_json(jsonPath, profile)) {
        JTRACE("STOCK", std::string("failed to load agx profile json: ") + stock.jsonKey);
        return false;
    }

    c_data = std::move(profile.dyeC);
    m_data = std::move(profile.dyeM);
    y_data = std::move(profile.dyeY);
    r_sens = std::move(profile.logSensR);
    g_sens = std::move(profile.logSensG);
    b_sens = std::move(profile.logSensB);
    dc_r = std::move(profile.densityCurveR);
    dc_g = std::move(profile.densityCurveG);
    dc_b = std::move(profile.densityCurveB);
    dmin = std::move(profile.baseMin);
    dmid = std::move(profile.baseMid);
    dc_layers = std::move(profile.densityCurvesLayers);
    if (std::isfinite(profile.dyeDensityMinFactor) && profile.dyeDensityMinFactor >= 0.0f) {
        S.base.dyeDensityMinFactor = profile.dyeDensityMinFactor;
    }
    if (profile.hasGammaFactor) {
        S.base.gammaFactor = profile.gammaFactor;
    }
    S.base.densityMidNeutral = std::move(profile.densityMidNeutral);
    S.base.logExposureMidNeutral = std::move(profile.logExposureMidNeutral);
    S.base.referenceIlluminant = profile.referenceIlluminant;
    S.base.viewingIlluminant = profile.viewingIlluminant;
    S.filmReferenceIlluminant = S.base.referenceIlluminant;
    S.base.dirCouplers = profile.dirCouplers;
    S.base.cameraFilterUV = profile.cameraFilterUV;
    S.base.cameraFilterIR = profile.cameraFilterIR;
    S.base.cameraFilterDefined = profile.hasCameraFilterUV || profile.hasCameraFilterIR;
    if (profile.dirCouplers.hasData && std::isfinite(profile.dirCouplers.diffusionSizeUm)) {
        const float spatialSigmaUm = std::clamp(profile.dirCouplers.diffusionSizeUm, 0.0f, 50.0f);
        S.couplerProfileSpatialSigmaMicrometers = static_cast<double>(spatialSigmaUm);
        S.couplerProfileSpatialSigmaValid = true;
    }
    else {
        S.couplerProfileSpatialSigmaMicrometers = 0.0;
        S.couplerProfileSpatialSigmaValid = false;
    }
    S.base.maskingCouplers = profile.maskingCouplers;
    S.base.grain = profile.grain;
    S.base.halation = profile.halation;
    S.base.glare = profile.glare;
    if (profile.hasDensityCurvesLayers) {
        S.base.hasDensityCurvesLayers = true;
        for (size_t layer = 0; layer < dc_layers.size(); ++layer) {
            for (size_t ch = 0; ch < dc_layers[layer].size(); ++ch) {
                S.base.densityCurvesLayers[layer][ch].clear();
                S.base.densityCurvesLayers[layer][ch].reserve(dc_layers[layer][ch].size());
                for (const auto& sample : dc_layers[layer][ch]) {
                    S.base.densityCurvesLayers[layer][ch].push_back(sample.second);
                }
            }
        }
    }
    JTRACE("STOCK", std::string("loaded agx profile json: ") + stock.jsonKey);
    if (!dc_r.empty() && !dc_g.empty() && !dc_b.empty()) {
        std::ostringstream oss;
        oss << "density curves loaded from JSON '" << stock.jsonKey << "' samples R/G/B="
            << dc_r.size() << "/" << dc_g.size() << "/" << dc_b.size();
        JTRACE("STOCK", oss.str());
    }
    else {
        JTRACE("STOCK", std::string("density curves missing in JSON profile: ") + stock.jsonKey);
    }

    {
        auto sz = [](const auto& v) { return static_cast<int>(v.size()); };
        std::ostringstream oss;
        oss << "c/m/y=" << sz(c_data) << "/" << sz(m_data) << "/" << sz(y_data)
            << " sens r/g/b=" << sz(r_sens) << "/" << sz(g_sens) << "/" << sz(b_sens)
            << " dens r/g/b=" << sz(dc_r) << "/" << sz(dc_g) << "/" << sz(dc_b)
            << " base min/mid=" << sz(dmin) << "/" << sz(dmid);
        JTRACE("STOCK", oss.str());
    }

    if (c_data.empty()) JTRACE("STOCK", "dye_density_c missing/empty");
    if (m_data.empty()) JTRACE("STOCK", "dye_density_m missing/empty");
    if (y_data.empty()) JTRACE("STOCK", "dye_density_y missing/empty");
    if (r_sens.empty()) JTRACE("STOCK", "log_sensitivity_r missing/empty");
    if (g_sens.empty()) JTRACE("STOCK", "log_sensitivity_g missing/empty");
    if (b_sens.empty()) JTRACE("STOCK", "log_sensitivity_b missing/empty");
    if (dc_r.empty()) JTRACE("STOCK", "density_curve_r missing/empty");
    if (dc_g.empty()) JTRACE("STOCK", "density_curve_g missing/empty");
    if (dc_b.empty()) JTRACE("STOCK", "density_curve_b missing/empty");

    const bool okCore =
        !c_data.empty() && !m_data.empty() && !y_data.empty() &&
        !r_sens.empty() && !g_sens.empty() && !b_sens.empty() &&
        !dc_r.empty() && !dc_g.empty() && !dc_b.empty();
    if (!okCore) {
        JTRACE("STOCK", "okCore=false (required assets missing)");
        JTRACE("STOCK", "film stock load aborted due to missing JSON density curves");
        return false;
    }

    const bool epsYOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsY, y_data);
    const bool epsMOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsM, m_data);
    const bool epsCOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.epsC, c_data);

    const bool sensBOk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensB, b_sens);
    const bool sensGOk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensG, g_sens);
    const bool sensROk = Spectral::build_curve_on_reference_axis_from_log10_aligned_pairs(S.base.sensR, r_sens);

    const bool densBOk = Spectral::build_curve_on_log_exposure_axis(S.base.densB, dc_b);
    const bool densGOk = Spectral::build_curve_on_log_exposure_axis(S.base.densG, dc_g);
    const bool densROk = Spectral::build_curve_on_log_exposure_axis(S.base.densR, dc_r);

    const bool dyeOk = epsYOk && epsMOk && epsCOk;
    const bool sensOk = sensBOk && sensGOk && sensROk;
    const bool densOk = densBOk && densGOk && densROk;

    if (!dyeOk) {
        std::ostringstream oss;
        oss << "dye epsilon resample failure (Y=" << (epsYOk ? "ok" : "empty")
            << ", M=" << (epsMOk ? "ok" : "empty")
            << ", C=" << (epsCOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }
    if (!sensOk) {
        std::ostringstream oss;
        oss << "log sensitivity resample failure (B=" << (sensBOk ? "ok" : "empty")
            << ", G=" << (sensGOk ? "ok" : "empty")
            << ", R=" << (sensROk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    if (!densOk) {
        std::ostringstream oss;
        oss << "density curve build failure (B=" << (densBOk ? "ok" : "empty")
            << ", G=" << (densGOk ? "ok" : "empty")
            << ", R=" << (densROk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    if (!(dyeOk && sensOk && densOk)) {
        std::ostringstream fatal;
        fatal << "FATAL: missing spectral data (film profile '" << stock.jsonKey << "')";
        JTRACE("STOCK", fatal.str());
        return false;
    }

    auto subtract_baseline_floor = [](Spectral::Curve& curve) {
        if (curve.linear.empty()) return;
        float minVal = FLT_MAX;
        for (float v : curve.linear) {
            if (std::isfinite(v) && v < minVal) {
                minVal = v;
            }
        }
        if (!std::isfinite(minVal) || minVal == FLT_MAX || minVal == 0.0f) {
            return;
        }
        for (float& v : curve.linear) {
            // agx-emulsion parity (density curves): preserve authored NaNs through sampling; do not
            // convert NaN -> 0 density (which would lift shadows). agx does `curve -= nanmin(curve)`.
            if (std::isfinite(v)) {
                v -= minVal;
                // Guard against tiny negatives from float error; keep NaNs untouched.
                if (v < 0.0f) {
                    v = 0.0f;
                }
            }
        }
        };
    subtract_baseline_floor(S.base.densB);
    subtract_baseline_floor(S.base.densG);
    subtract_baseline_floor(S.base.densR);

    bool baseMinOk = false;
    if (!dmin.empty()) {
        baseMinOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.baseMin, dmin);
        if (!baseMinOk) {
            S.base.baseMin.lambda_nm.clear();
            S.base.baseMin.linear.clear();
        }
    }
    else {
        S.base.baseMin.lambda_nm.clear();
        S.base.baseMin.linear.clear();
    }

    bool baseMidOk = true;
    if (!dmid.empty()) {
        baseMidOk = Spectral::build_curve_on_reference_axis_from_aligned_pairs(S.base.baseMid, dmid);
        if (!baseMidOk) {
            S.base.baseMid.lambda_nm.clear();
            S.base.baseMid.linear.clear();
        }
    }
    else {
        S.base.baseMid.lambda_nm.clear();
        S.base.baseMid.linear.clear();
    }

    S.base.hasBaseline = baseMinOk && !S.base.baseMin.linear.empty();
    if (!baseMinOk) {
        std::ostringstream oss;
        oss << "baseline resample failure (min=" << (baseMinOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }
    else if (!baseMidOk && !dmid.empty()) {
        std::ostringstream oss;
        oss << "baseline resample warning (mid=" << (baseMidOk ? "ok" : "empty") << ")";
        JTRACE("STOCK", oss.str());
    }

    {
        std::ostringstream oss;
        oss << "epsY/M/C K=" << static_cast<int>(S.base.epsY.linear.size())
            << "/" << static_cast<int>(S.base.epsM.linear.size())
            << "/" << static_cast<int>(S.base.epsC.linear.size())
            << " sensB/G/R K=" << static_cast<int>(S.base.sensB.linear.size())
            << "/" << static_cast<int>(S.base.sensG.linear.size())
            << "/" << static_cast<int>(S.base.sensR.linear.size())
            << " densB/G/R K=" << static_cast<int>(S.base.densB.linear.size())
            << "/" << static_cast<int>(S.base.densG.linear.size())
            << "/" << static_cast<int>(S.base.densR.linear.size())
            << " baseMin/baseMid K=" << static_cast<int>(S.base.baseMin.linear.size())
            << "/" << static_cast<int>(S.base.baseMid.linear.size())
            << " hasBaseline=" << (S.base.hasBaseline ? 1 : 0);
        JTRACE("STOCK", oss.str());
    }

    JTRACE("STOCK", std::string("loaded OK; baseline=") + (S.base.hasBaseline ? "1" : "0"));

    return true;
}

void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {
    auto sanitize_curve = [](Spectral::Curve& c) {
        for (float& v : c.linear) {
            if (!std::isfinite(v)) v = 0.0f;
            if (v < 0.0f) v = 0.0f;
        }
        };

    auto estimate_base_dyes = [](const Spectral::Curve& baseCurve,
        const Spectral::Curve& epsY,
        const Spectral::Curve& epsM,
        const Spectral::Curve& epsC) -> std::array<float, 3>
        {
            std::array<float, 3> result{ 0.0f, 0.0f, 0.0f };
            const size_t K = baseCurve.linear.size();
            if (K == 0 || epsY.linear.size() != K || epsM.linear.size() != K || epsC.linear.size() != K) {
                return result;
            }

            double ATA[3][3] = { {0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0} };
            double ATb[3] = { 0.0, 0.0, 0.0 };

            for (size_t i = 0; i < K; ++i) {
                const double ay = epsY.linear[i];
                const double am = epsM.linear[i];
                const double ac = epsC.linear[i];
                const double b = baseCurve.linear[i];
                if (!std::isfinite(ay) || !std::isfinite(am) || !std::isfinite(ac) || !std::isfinite(b)) {
                    continue;
                }
                const double vec[3] = { ay, am, ac };
                for (int r = 0; r < 3; ++r) {
                    ATb[r] += vec[r] * b;
                    for (int c = 0; c < 3; ++c) {
                        ATA[r][c] += vec[r] * vec[c];
                    }
                }
            }

            double mat[3][4];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    mat[r][c] = ATA[r][c];
                }
                mat[r][3] = ATb[r];
            }

            for (int i = 0; i < 3; ++i) {
                int pivot = i;
                double maxAbs = std::fabs(mat[i][i]);
                for (int r = i + 1; r < 3; ++r) {
                    const double absVal = std::fabs(mat[r][i]);
                    if (absVal > maxAbs) {
                        maxAbs = absVal;
                        pivot = r;
                    }
                }
                if (maxAbs < 1e-9) {
                    return result;
                }
                if (pivot != i) {
                    for (int c = i; c < 4; ++c) {
                        std::swap(mat[i][c], mat[pivot][c]);
                    }
                }
                const double inv = 1.0 / mat[i][i];
                for (int c = i; c < 4; ++c) {
                    mat[i][c] *= inv;
                }
                for (int r = 0; r < 3; ++r) {
                    if (r == i) continue;
                    const double factor = mat[r][i];
                    for (int c = i; c < 4; ++c) {
                        mat[r][c] -= factor * mat[i][c];
                    }
                }
            }

            for (int i = 0; i < 3; ++i) {
                float v = static_cast<float>(mat[i][3]);
                if (!std::isfinite(v) || v < 0.0f) {
                    v = 0.0f;
                }
                result[i] = v;
            }

            return result;
        };

    JTRACE_SCOPE("BUILD", "rebuild_working_state");

    std::unique_lock<std::mutex> lk(S.m);
    std::shared_ptr<WorkingState> next = std::make_shared<WorkingState>();
    WorkingState* target = next.get();

    {
        std::ostringstream oss;
        oss << "enter with baseLoaded=" << (S.baseLoaded ? 1 : 0);
        JTRACE("BUILD", oss.str());
    }

    const std::uint64_t coreShareHash = hash_params_core(P);
    WorkingStateSharing::AcquireCoreSharedResult coreShare =
        WorkingStateSharing::acquire_or_create_shared_core(coreShareHash);
    if (coreShare.sharedCore && coreShare.sharedCore->payload) {
        WorkingStateSharing::apply_working_state_core_payload(*coreShare.sharedCore->payload, *target);
        target->coreShareHash = coreShareHash;
        target->sharedCore = coreShare.sharedCore;

        recompute_working_state_dir_overlay(S, P, *target);
        S.spatialSigmaCacheValid.store(false, std::memory_order_release);
        if (rebuild_working_state_scanner_output_runtime(P, *target)) {
            target->fullHash = hash_params(P);
            target->uploadCoreHash = hash_params_upload_core(P);
            target->coreHash = coreShareHash;
            target->coreShareHash = coreShareHash;
            target->dirHash = hash_params_dir(P);
            target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
            trace_working_state_core_share(coreShare, target->buildCounter, "full_rebuild_payload_fast");

            if (JTRACE_ENABLED(3)) {
                const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
                const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
                const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(target->printRT.get());
                const float neutralY = target->printRT ? target->printRT->neutralY : 0.0f;
                const float neutralM = target->printRT ? target->printRT->neutralM : 0.0f;
                const float neutralC = target->printRT ? target->printRT->neutralC : 0.0f;
                std::string msg = std::string("working state commit build=") + std::to_string(target->buildCounter)
                    + " paper=" + std::string(paperKey ? paperKey : "<null>")
                    + " film=" + std::string(filmKey ? filmKey : "<null>")
                    + " printRT=" + std::to_string(prtPtr)
                    + " neutralY/M/C=" + std::to_string(neutralY) + "/" + std::to_string(neutralM) + "/" + std::to_string(neutralC)
                    + " printRef=" + (target->printRT ? target->printRT->referenceIlluminant : std::string("<null>"))
                    + " printView=" + (target->printRT ? target->printRT->viewingIlluminant : std::string("<null>"));
                JTRACE_VERBOSE("PRINTDBG", msg);
            }

            {
                std::ostringstream oss;
                oss << "WorkingState build #" << target->buildCounter;
                JTRACE("BUILD", oss.str());
            }

            JuicerAtomic::store_shared_ptr(&S.activeWorkingState, std::shared_ptr<const WorkingState>(next));
            {
                std::ostringstream oss;
                oss << "activeWorkingState swapped; buildCounter=" << static_cast<long long>(target->buildCounter);
                JTRACE("BUILD", oss.str());
            }
            S.activeBuildCounter = target->buildCounter;
            S.lastParams = P;
            S.lastHash.store(target->fullHash, std::memory_order_release);
            return;
        }
        JTRACE("MSWSC", "event=core_share_fastpath_fallback reason=scanner_runtime_rebuild_failed");
    }

    Print::build_illuminant_from_choice(P.enlIll, S.printRT, S.dataDir, /*forEnlarger*/true);
    Scanner::ScannerIlluminant printScannerIlluminant;
    if (!build_scanner_illuminant(S.dataDir, S.printRT.viewingIlluminant, "print viewing", printScannerIlluminant)) {
        lk.unlock();
        return;
    }
    S.printRT.illumView = printScannerIlluminant.curve;
    {
        std::ostringstream oss;
        oss << "Enl illum K=" << static_cast<int>(S.printRT.illumEnlarger.linear.size())
            << " View illum K=" << static_cast<int>(S.printRT.illumView.linear.size());
        JTRACE("BUILD", oss.str());
    }

    Scanner::ScannerIlluminant negativeScannerIlluminant;
    if (!build_scanner_illuminant(S.dataDir, S.base.viewingIlluminant, "negative viewing", negativeScannerIlluminant)) {
        lk.unlock();
        return;
    }
    Scanner::ScannerDensityRange negativeDensityRange;
    bool negativeRangeOk = false;

    Spectral::Curve epsY = S.base.epsY;
    Spectral::Curve epsM = S.base.epsM;
    Spectral::Curve epsC = S.base.epsC;
    Spectral::Curve sensB = S.base.sensB;
    Spectral::Curve sensG = S.base.sensG;
    Spectral::Curve sensR = S.base.sensR;
    Spectral::Curve densB = S.base.densB;
    Spectral::Curve densG = S.base.densG;
    Spectral::Curve densR = S.base.densR;
    Spectral::Curve dirDensB = densB;
    Spectral::Curve dirDensG = densG;
    Spectral::Curve dirDensR = densR;
    Spectral::Curve baseMin = S.base.baseMin;
    Spectral::Curve baseMid = S.base.baseMid;
    Spectral::Curve illumRef;
    bool hasRefIlluminant = false;
    const bool hasBaseline = S.base.hasBaseline;
    const float dyeDensityMinScale =
        (std::isfinite(S.base.dyeDensityMinFactor) && S.base.dyeDensityMinFactor >= 0.0f)
        ? S.base.dyeDensityMinFactor
        : 1.0f;
    if (hasBaseline && !approx_equal(dyeDensityMinScale, 1.0f)) {
        for (float& v : baseMin.linear) {
            if (std::isfinite(v)) {
                v *= dyeDensityMinScale;
                if (v < 0.0f) {
                    v = 0.0f;
                }
            }
        }
    }
    if (hasBaseline) {
        for (float& v : baseMin.linear) {
            if (std::isfinite(v) && v < 0.0f) {
                v = 0.0f;
            }
        }
        for (float& v : baseMid.linear) {
            if (std::isfinite(v) && v < 0.0f) {
                v = 0.0f;
            }
        }
    }
    // agx-emulsion parity: baseline NaNs are preserved in working-state curves and handled as
    // "0 contribution" during integration via SpectralTables baseline validity masks.


    // Per agx-emulsion parity: film profiles contain sensitivities that are ALREADY balanced
    // during profile generation (profiles/balance.py). Runtime rebalancing creates spectral
    // errors. We load the reference illuminant for metadata/debugging purposes only.
    {
        Print::Runtime tmpRT;

        Spectral::Curve profileRefIll;
        if (!S.illuminantOverride.reference && !S.filmReferenceIlluminant.empty()) {
            profileRefIll = build_illuminant_from_string(S.dataDir, S.filmReferenceIlluminant);
        }

        if (!profileRefIll.linear.empty() &&
            static_cast<int>(profileRefIll.linear.size()) == Spectral::gShape.K)
        {
            tmpRT.illumView = profileRefIll;
        }
        else {
            Print::build_illuminant_from_choice(P.refIll, tmpRT, S.dataDir, /*forEnlarger*/false);
        }

        illumRef = tmpRT.illumView;
        hasRefIlluminant =
            (!illumRef.linear.empty() && (int)illumRef.linear.size() == Spectral::gShape.K);

        // NOTE: Removed runtime balancing. Profiles are pre-balanced; rebalancing at runtime
        // violates agx-emulsion parity and causes red/magenta color shifts. See claude-review.md.
        if (hasRefIlluminant) {
            JTRACE("BUILD", "loaded reference illuminant for metadata (no runtime balancing applied)");
        }
        else {
            JTRACE("BUILD", "reference illuminant failed to load; SPD reconstruction will be disabled");
        }
    }

    {
        auto to_triplet = [](const std::array<double, 3>& src, const std::array<float, 3>& fallback) {
            std::array<float, 3> out = fallback;
            for (size_t i = 0; i < out.size(); ++i) {
                const double v = src[i];
                if (std::isfinite(v)) {
                    out[i] = static_cast<float>(v);
                }
            }
            return out;
            };

        std::array<float, 3> filterUV = S.base.cameraFilterUV;
        std::array<float, 3> filterIR = S.base.cameraFilterIR;
        if (P.cameraFilterOverride) {
            filterUV = to_triplet(P.cameraFilterUV, filterUV);
            filterIR = to_triplet(P.cameraFilterIR, filterIR);
        }

        const float ampUV = std::clamp(filterUV[0], 0.0f, 1.0f);
        const float ampIR = std::clamp(filterIR[0], 0.0f, 1.0f);
        if (ampUV > 0.0f || ampIR > 0.0f) {
            const std::vector<float> bandPass = Spectral::compute_band_pass_filter(filterUV, filterIR);
            if (bandPass.size() == static_cast<size_t>(Spectral::gShape.K)) {
                auto applyFilter = [&bandPass](Spectral::Curve& curve) {
                    if (curve.linear.size() != bandPass.size()) {
                        return;
                    }
                    for (size_t i = 0; i < bandPass.size(); ++i) {
                        curve.linear[i] *= bandPass[i];
                    }
                    };

                // Apply UV/IR band-pass filters to sensitivities (agx-emulsion parity)
                applyFilter(sensB);
                applyFilter(sensG);
                applyFilter(sensR);

                Spectral::mark_spectral_tables_dirty();
            }
        }
    }

    sanitize_curve(sensB);
    sanitize_curve(sensG);
    sanitize_curve(sensR);

    Spectral::Curve densBForCalibration = densB;
    Spectral::Curve densGForCalibration = densG;
    Spectral::Curve densRForCalibration = densR;
    auto compute_curve_max = [](const Spectral::Curve& c) {
        float m = 0.0f;
        for (float v : c.linear) {
            if (std::isfinite(v) && v > m) {
                m = v;
            }
        }
        if (!std::isfinite(m) || m <= 1e-4f) {
            m = 1.0f;
        }
        if (m > 1000.0f) {
            m = 1000.0f;
        }
        return m;
        };
    std::array<float, 3> densityMaxPostDir{
        compute_curve_max(densBForCalibration),
        compute_curve_max(densGForCalibration),
        compute_curve_max(densRForCalibration)
    };

    const Profiles::DirCouplersProfile& dirCfg = S.base.dirCouplers;

    int effectiveCouplersActive = P.couplersActive;
    double effectiveCouplersAmount = P.couplersAmount;
    double effectiveRatioB = P.ratioB;
    double effectiveRatioG = P.ratioG;
    double effectiveRatioR = P.ratioR;
    double effectiveCouplersSigma = P.sigma;
    double effectiveCouplersHigh = P.high;
    const bool spatialSigmaIsUiDefault = approx_equal(P.spatialSigmaMicrometers, kFactoryCouplersSpatialSigma);
    double effectiveSpatialSigma = P.spatialSigmaMicrometers;
    if (spatialSigmaIsUiDefault && S.couplerProfileSpatialSigmaValid) {
        effectiveSpatialSigma = S.couplerProfileSpatialSigmaMicrometers;
    }

#ifdef JUICER_ENABLE_COUPLERS
    if (dirCfg.hasData) {
        auto sanitize_profile = [](float value, double fallback, double lo, double hi) -> double {
            double v = static_cast<double>(value);
            if (!std::isfinite(v)) {
                return fallback;
            }
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            return v;
            };

        if (!S.couplerDirty.active.load(std::memory_order_acquire) && effectiveCouplersActive == kFactoryCouplersActive) {
            effectiveCouplersActive = dirCfg.active ? 1 : 0;
        }
        if (!S.couplerDirty.amount.load(std::memory_order_acquire) && approx_equal(effectiveCouplersAmount, kFactoryCouplersAmount)) {
            effectiveCouplersAmount = sanitize_profile(dirCfg.amount, effectiveCouplersAmount, 0.0, 2.0);
        }
        if (!S.couplerDirty.ratioB.load(std::memory_order_acquire) && approx_equal(effectiveRatioB, kFactoryCouplersRatioB)) {
            effectiveRatioB = sanitize_profile(dirCfg.ratioRGB[0], effectiveRatioB, 0.0, 1.0);
        }
        if (!S.couplerDirty.ratioG.load(std::memory_order_acquire) && approx_equal(effectiveRatioG, kFactoryCouplersRatioG)) {
            effectiveRatioG = sanitize_profile(dirCfg.ratioRGB[1], effectiveRatioG, 0.0, 1.0);
        }
        if (!S.couplerDirty.ratioR.load(std::memory_order_acquire) && approx_equal(effectiveRatioR, kFactoryCouplersRatioR)) {
            effectiveRatioR = sanitize_profile(dirCfg.ratioRGB[2], effectiveRatioR, 0.0, 1.0);
        }
        if (!S.couplerDirty.sigma.load(std::memory_order_acquire) && approx_equal(effectiveCouplersSigma, kFactoryCouplersSigma)) {
            effectiveCouplersSigma = sanitize_profile(dirCfg.diffusionInterlayer, effectiveCouplersSigma, 0.0, 4.0);
        }
        if (!S.couplerDirty.high.load(std::memory_order_acquire) && approx_equal(effectiveCouplersHigh, kFactoryCouplersHigh)) {
            effectiveCouplersHigh = sanitize_profile(dirCfg.highExposureShift, effectiveCouplersHigh, 0.0, 1.0);
        }
        if (!S.couplerDirty.spatialSigma.load(std::memory_order_acquire) && spatialSigmaIsUiDefault) {
            const double profileSpatialSigma = S.couplerProfileSpatialSigmaValid
                ? S.couplerProfileSpatialSigmaMicrometers
                : static_cast<double>(dirCfg.diffusionSizeUm);
            effectiveSpatialSigma = sanitize_profile(static_cast<float>(profileSpatialSigma), effectiveSpatialSigma, 0.0, 50.0);
        }
    }
#endif

    bool precorrectApplied = false;
    Couplers::Runtime dirRT{};
    dirRT.active = (effectiveCouplersActive != 0);
    {
        auto clampRatio = [](double v) -> float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            if (v > 1.0) return 1.0f;
            return static_cast<float>(v);
            };
        auto clampAmount = [](double v) -> float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            if (v > 2.0) return 2.0f;
            return static_cast<float>(v);
            };
        const float amountScale = clampAmount(effectiveCouplersAmount);
        const float amount[3] = {
            amountScale * clampRatio(effectiveRatioB),
            amountScale * clampRatio(effectiveRatioG),
            amountScale * clampRatio(effectiveRatioR)
        };
#ifdef JUICER_ENABLE_COUPLERS
        Couplers::build_dir_matrix(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#else
        auto build_dir_matrix_stub = [](float M[3][3], const float amountValues[3], float layerSigma) {
            const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
            float amt[3] = { amountValues[0], amountValues[1], amountValues[2] };
            const float sigmaCapped = std::min(sigma, 3.0f);
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(amt[i])) amt[i] = 0.0f;
                amt[i] = std::clamp(amt[i], 0.0f, 1.0f);
            }
            auto gauss = [sigmaCapped](int dx) -> float {
                if (sigmaCapped <= 0.0f) {
                    return (dx == 0) ? 1.0f : 0.0f;
                }
                const float s2 = sigmaCapped * sigmaCapped;
                return std::exp(-0.5f * (dx * dx) / s2);
                };
            for (int r = 0; r < 3; ++r) {
                float row[3];
                float wsum = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    row[c] = gauss(c - r);
                    wsum += row[c];
                }
                if (wsum > 0.0f) {
                    for (int c = 0; c < 3; ++c) {
                        row[c] /= wsum;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    M[r][c] = amt[r] * row[c];
                }
            }
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(M[r][c])) {
                        M[r][c] = 0.0f;
                    }
                }
            }
            };
        build_dir_matrix_stub(dirRT.M, amount, static_cast<float>(effectiveCouplersSigma));
#endif
        dirRT.highShift = static_cast<float>(effectiveCouplersHigh);
        dirRT.spatialSigmaMicrometers = static_cast<float>(effectiveSpatialSigma);
        dirRT.spatialSigmaPixels = 0.0f;

#ifdef JUICER_ENABLE_COUPLERS
        if (dirRT.active) {
            auto has_nonfinite_density = [](const Spectral::Curve& c) -> bool {
                for (float v : c.linear) {
                    if (!std::isfinite(v)) {
                        return true;
                    }
                }
                return false;
                };

            // agx-emulsion parity: density curves may contain intentional toe NaNs. DIR pre-correction
            // must not "heal" them into 0 densities; if authored NaNs exist, skip pre-correction and
            // let NaNs propagate through sampling to "0 transmitted light" downstream.
            if (has_nonfinite_density(densB) || has_nonfinite_density(densG) || has_nonfinite_density(densR)) {
                precorrectApplied = false;
                JTRACE("BUILD", "precorrect: skipped (density curves contain non-finite samples; NaN toe parity)");
            }
            else {
                Spectral::Curve densB_corr, densG_corr, densR_corr;
                Couplers::precorrect_density_curves_before_DIR_into(
                    dirRT.M, dirRT.highShift,
                    densB, densG, densR,
                    densB_corr, densG_corr, densR_corr);
                dirDensB = std::move(densB_corr);
                dirDensG = std::move(densG_corr);
                dirDensR = std::move(densR_corr);
                precorrectApplied = true;
            }
        }
#else
        (void)dirRT;
#endif

        dirRT.dMax[0] = densityMaxPostDir[0];
        dirRT.dMax[1] = densityMaxPostDir[1];
        dirRT.dMax[2] = densityMaxPostDir[2];
        {
            std::ostringstream oss;
            oss << "DIR active=" << (dirRT.active ? 1 : 0)
                << " sigma=" << static_cast<float>(effectiveCouplersSigma) << " high=" << static_cast<float>(effectiveCouplersHigh)
                << " dMax=" << dirRT.dMax[0] << "," << dirRT.dMax[1] << "," << dirRT.dMax[2]
                << " precorrect=" << (precorrectApplied ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }
    }
    dirRT.dMax[0] = densityMaxPostDir[0];
    dirRT.dMax[1] = densityMaxPostDir[1];
    dirRT.dMax[2] = densityMaxPostDir[2];

    Spectral::NegativeCouplerParams negParams;
    negParams.DmaxY = dirRT.dMax[0];
    negParams.DmaxM = dirRT.dMax[1];
    negParams.DmaxC = dirRT.dMax[2];
    negParams.baseY = 0.0f;
    negParams.baseM = 0.0f;
    negParams.baseC = 0.0f;
    negParams.kB = 6.0f;
    negParams.kG = 6.0f;
    negParams.kR = 6.0f;
    {
        const float defaultMask[9] = {
            0.98f, -0.06f, -0.02f,
           -0.03f,  0.98f, -0.05f,
           -0.02f, -0.04f,  0.98f };
        std::copy(std::begin(defaultMask), std::end(defaultMask), negParams.mask);
    }

    if (S.base.hasBaseline) {
        const auto baseCoeffs = estimate_base_dyes(baseMin, epsY, epsM, epsC);
        negParams.baseY = baseCoeffs[0];
        negParams.baseM = baseCoeffs[1];
        negParams.baseC = baseCoeffs[2];
    }

    if (dirCfg.hasData) {
        auto computeK = [&](int idx) -> float {
            float amount = std::isfinite(dirCfg.amount) ? dirCfg.amount : 1.0f;
            if (amount <= 0.0f) amount = 0.1f;
            float ratio = dirCfg.ratioRGB[idx];
            if (!std::isfinite(ratio) || ratio <= 0.0f) ratio = 1.0f;
            float k = 6.0f * amount * ratio;
            if (!std::isfinite(k) || k <= 0.0f) k = 6.0f;
            return std::clamp(k, 1.0f, 24.0f);
            };
        negParams.kB = computeK(0);
        negParams.kG = computeK(1);
        negParams.kR = computeK(2);
    }

    const bool hasMaskingData = S.base.maskingCouplers.hasData;
    if (dirCfg.hasData || hasMaskingData) {
        float amountRGB[3] = { 1.0f, 1.0f, 1.0f };
        if (dirCfg.hasData) {
            for (int i = 0; i < 3; ++i) {
                float ratio = dirCfg.ratioRGB[i];
                if (!std::isfinite(ratio)) ratio = 1.0f;
                if (ratio < 0.0f) ratio = 0.0f;
                float amount = std::isfinite(dirCfg.amount) ? dirCfg.amount : 1.0f;
                if (amount < 0.0f) amount = 0.0f;
                amountRGB[i] = std::clamp(amount * ratio, 0.0f, 1.0f);
            }
        }

        float dirMatrix[3][3] = { {0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
#ifdef JUICER_ENABLE_COUPLERS
        Couplers::build_dir_matrix(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#else
        auto build_dir_matrix_local = [](float M[3][3], const float amountValues[3], float layerSigma) {
            const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
            float amt[3] = { amountValues[0], amountValues[1], amountValues[2] };
            const float sigmaCapped = std::min(sigma, 3.0f);
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(amt[i])) amt[i] = 0.0f;
                amt[i] = std::clamp(amt[i], 0.0f, 1.0f);
            }
            auto gauss = [sigmaCapped](int dx) -> float {
                if (sigmaCapped <= 0.0f) {
                    return (dx == 0) ? 1.0f : 0.0f;
                }
                const float s2 = sigmaCapped * sigmaCapped;
                return std::exp(-0.5f * (dx * dx) / s2);
                };
            for (int r = 0; r < 3; ++r) {
                float row[3];
                float wsum = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    row[c] = gauss(c - r);
                    wsum += row[c];
                }
                if (wsum > 0.0f) {
                    for (int c = 0; c < 3; ++c) {
                        row[c] /= wsum;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    M[r][c] = amt[r] * row[c];
                }
            }
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(M[r][c])) {
                        M[r][c] = 0.0f;
                    }
                }
            }
            };
        build_dir_matrix_local(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#endif

        std::array<float, 3> maskScaleCh{ {1.0f, 1.0f, 1.0f} };
        std::array<float, 3> maskOffsetCh{ {0.0f, 0.0f, 0.0f} };
        if (hasMaskingData) {
            const auto& maskProfile = S.base.maskingCouplers;
            const auto& lambda = Spectral::gShape.wavelengths;
            const size_t K = lambda.size();
            const Spectral::Curve* epsCurves[3] = { &S.base.epsY, &S.base.epsM, &S.base.epsC };
            const float effectiveness = 1.0f;
            for (int ch = 0; ch < 3; ++ch) {
                const Spectral::Curve* eps = epsCurves[ch];
                if (!eps || eps->linear.size() != K || K == 0) {
                    continue;
                }
                const float cross = (maskProfile.crossOverPoints.size() > static_cast<size_t>(ch))
                    ? maskProfile.crossOverPoints[ch]
                    : std::numeric_limits<float>::quiet_NaN();
                const float widthRaw = (maskProfile.transitionWidths.size() > static_cast<size_t>(ch))
                    ? maskProfile.transitionWidths[ch]
                    : std::numeric_limits<float>::quiet_NaN();
                const float width = (std::isfinite(widthRaw) && std::fabs(widthRaw) > 1e-6f)
                    ? std::fabs(widthRaw)
                    : std::numeric_limits<float>::quiet_NaN();

                double weightSum = 0.0;
                double scaleSum = 0.0;
                double offsetSum = 0.0;
                for (size_t i = 0; i < K; ++i) {
                    const float weight = eps->linear[i];
                    if (!std::isfinite(weight) || weight <= 0.0f) {
                        continue;
                    }
                    const float lambda_nm = lambda[i];
                    float scaleSpectral = 1.0f;
                    if (std::isfinite(cross) && std::isfinite(width)) {
                        const float t = (lambda_nm - cross) / width;
                        scaleSpectral = (std::erf(t) + 1.0f + effectiveness) / (2.0f + effectiveness);
                    }
                    double gaussSpectral = 0.0;
                    const auto& gaussians = maskProfile.gaussianModel[ch];
                    for (const auto& tri : gaussians) {
                        float mu = tri[0];
                        float sigma = tri[1];
                        float amp = tri[2];
                        if (!std::isfinite(mu) || !std::isfinite(sigma) || !std::isfinite(amp)) {
                            continue;
                        }
                        sigma = std::fabs(sigma);
                        if (sigma <= 1e-6f) {
                            continue;
                        }
                        const float s = (lambda_nm - mu) / sigma;
                        gaussSpectral += static_cast<double>(amp) * std::exp(-0.5 * static_cast<double>(s) * static_cast<double>(s));
                    }
                    weightSum += static_cast<double>(weight);
                    scaleSum += static_cast<double>(weight) * static_cast<double>(scaleSpectral);
                    offsetSum += static_cast<double>(weight) * gaussSpectral;
                }
                if (weightSum > 0.0) {
                    float scaleAvg = static_cast<float>(scaleSum / weightSum);
                    float offsetAvg = static_cast<float>(offsetSum / weightSum);
                    if (!std::isfinite(scaleAvg) || scaleAvg <= 0.0f) {
                        scaleAvg = 1.0f;
                    }
                    if (!std::isfinite(offsetAvg) || offsetAvg < 0.0f) {
                        offsetAvg = 0.0f;
                    }
                    maskScaleCh[ch] = scaleAvg;
                    maskOffsetCh[ch] = offsetAvg;
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            float scale = maskScaleCh[i];
            if (!std::isfinite(scale) || scale <= 0.0f) scale = 1.0f;
            float offset = maskOffsetCh[i];
            if (!std::isfinite(offset) || offset < 0.0f) offset = 0.0f;
            negParams.maskScale[i] = scale;
            negParams.maskOffset[i] = offset;
        }

        const float maskScaleBase = 0.25f;
        const float maskScaleOffset = hasMaskingData ? 0.05f : 0.0f;
        for (int r = 0; r < 3; ++r) {
            float maskStrength = hasMaskingData
                ? ((1.0f - maskScaleCh[r]) + maskOffsetCh[r])
                : 0.0f;
            if (!std::isfinite(maskStrength)) maskStrength = 0.0f;
            float amountRow = dirCfg.hasData ? amountRGB[r] : 1.0f;
            if (!std::isfinite(amountRow) || amountRow < 0.0f) amountRow = 0.0f;
            float scale = maskScaleBase * (maskStrength + maskScaleOffset);
            if (dirCfg.hasData) {
                scale *= amountRow;
            }
            scale = std::clamp(scale, 0.0f, 0.25f);
            for (int c = 0; c < 3; ++c) {
                float val = dirMatrix[r][c];
                if (!std::isfinite(val)) {
                    val = (r == c) ? 1.0f : 0.0f;
                }
                const float delta = scale * val;
                if (r == c) {
                    float diag = 1.0f - delta;
                    if (!std::isfinite(diag)) diag = 1.0f;
                    if (diag < 0.0f) diag = 0.0f;
                    negParams.mask[r * 3 + c] = diag;
                }
                else {
                    float off = -delta;
                    if (!std::isfinite(off)) off = 0.0f;
                    off = std::clamp(off, -1.0f, 1.0f);
                    negParams.mask[r * 3 + c] = off;
                }
            }
        }
    }

    target->negativeScannerValid = false;
    target->printScannerValid = false;
    target->printGlareCompensated = false;

    target->negParams = negParams;
    target->grain = S.base.grain;
    target->halation = S.base.halation;
    target->negativeGlare = S.base.glare;
    target->hasDensityCurvesLayers = S.base.hasDensityCurvesLayers;
    for (size_t layer = 0; layer < target->densityCurvesLayers.size(); ++layer) {
        for (size_t ch = 0; ch < target->densityCurvesLayers[layer].size(); ++ch) {
            if (target->hasDensityCurvesLayers) {
                target->densityCurvesLayers[layer][ch] = S.base.densityCurvesLayers[layer][ch];
            }
            else {
                target->densityCurvesLayers[layer][ch].clear();
            }
        }
    }

    auto average_positive = [](const auto& values) -> float {
        float sum = 0.0f;
        int count = 0;
        for (float v : values) {
            if (std::isfinite(v) && v > 1e-6f) {
                sum += v;
                ++count;
            }
        }
        return (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
        };

    const float baselineMixReference = !S.base.densityMidNeutral.empty()
        ? average_positive(S.base.densityMidNeutral)
        : 0.0f;
    const float printBaselineMixReference =
        (S.printRT.profile.hasBaseline && S.printRT.hasMidNeutralDensity)
        ? average_positive(S.printRT.midNeutralDensity)
        : 0.0f;

        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ S.printRT.illumView,
            /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
            baselineMixReference,
            target->tablesView,
            printScannerIlluminant.hash);

    if (hasRefIlluminant) {
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ illumRef,
            /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
            baselineMixReference,
            target->tablesRef);

        const bool validWhite =
            std::isfinite(target->tablesRef.whiteXYZ[0]) &&
            std::isfinite(target->tablesRef.whiteXYZ[1]) &&
            std::isfinite(target->tablesRef.whiteXYZ[2]) &&
            target->tablesRef.whiteXYZ[1] > 0.0f;
        if (!validWhite) {
            JTRACE("BUILD", "reference illuminant produced invalid white XYZ; disabling SPD for safety");
            target->tablesRef = Spectral::SpectralTables{};
            hasRefIlluminant = false;
        }
    }
    else {
        target->tablesRef = Spectral::SpectralTables{};
    }

    std::shared_ptr<Print::Runtime> printRuntimeCopy = std::make_shared<Print::Runtime>(S.printRT);
    Print::Profile printProfile = printRuntimeCopy->profile;
    Print::DensityCurves printCurves = printRuntimeCopy->densityCurvesRaw;
    Scanner::ScannerDensityRange printDensityRange;
    bool printRangeOk = false;
    bool printDensityOk = !printCurves.cyan.empty() &&
        !printCurves.magenta.empty() &&
        !printCurves.yellow.empty();
    bool printRuntimeOk = false;
    if (printDensityOk) {
        const float factor = static_cast<float>(
            std::clamp(std::isfinite(P.glareCompRemovalFactor) ? P.glareCompRemovalFactor : 0.0, 0.0, 1.0));
        const float density = static_cast<float>(
            std::clamp(std::isfinite(P.glareCompRemovalDensity) ? P.glareCompRemovalDensity : 1.2, 0.0, 3.0));
        const float transition = static_cast<float>(
            std::clamp(std::isfinite(P.glareCompRemovalTransition) ? P.glareCompRemovalTransition : 0.3, 0.0, 2.0));

        printProfile.glare.compensationRemovalFactor = factor;
        printProfile.glare.compensationRemovalDensity = density;
        printProfile.glare.compensationRemovalTransition = transition;
        printProfile.glareCompensationFactor = factor;
        printProfile.glareCompensationDensity = density;
        printProfile.glareCompensationTransition = transition;
        printProfile.hasGlareCompensation = (factor > 0.0f);

        if (factor > 0.0f) {
            const bool removed = Print::remove_glare_compensation_from_curves(printProfile, printCurves);
            if (!removed) {
                JTRACE("PRINT", "FATAL: failed to remove viewing glare compensation from print curves");
                printDensityOk = false;
            }
        }
    }
    if (printDensityOk) {
        printDensityOk = Print::rebuild_density_curves(printProfile, printCurves);
        if (printDensityOk) {
            printRangeOk = compute_print_density_range(printProfile, printDensityRange);
            printDensityOk = printRangeOk;
        }
    }
    Print::recompute_mid_neutral(printProfile, printRuntimeCopy.get());
    printRuntimeCopy->profile = printProfile;
    printRuntimeCopy->glare = printProfile.glare;

    if (printDensityOk &&
        Print::profile_is_valid(printProfile) &&
        printRuntimeCopy->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K))
    {
        const float printDminFactor = std::isfinite(P.printDminFactor)
            ? static_cast<float>(std::clamp(P.printDminFactor, 0.0, 1.0))
            : 0.4f;
        if (printProfile.hasBaseline && !approx_equal(printDminFactor, 1.0f)) {
            for (float& v : printProfile.baseMin.linear) {
                if (std::isfinite(v)) {
                    v *= printDminFactor;
                }
            }
        }
        if (printProfile.hasBaseline) {
            for (float& v : printProfile.baseMin.linear) {
                if (std::isfinite(v) && v < 0.0f) {
                    v = 0.0f;
                }
            }
            for (float& v : printProfile.baseMid.linear) {
                if (std::isfinite(v) && v < 0.0f) {
                    v = 0.0f;
                }
            }
        }
        printRuntimeCopy->profile = printProfile;
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ printProfile.epsY,
            /*epsM*/ printProfile.epsM,
            /*epsC*/ printProfile.epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ printRuntimeCopy->illumView,
            /*baseMin*/ printProfile.baseMin,
            /*baseMid*/ printProfile.baseMid,
            /*hasBaseline*/ printProfile.hasBaseline,
            printBaselineMixReference,
            target->tablesPrint,
            printScannerIlluminant.hash);
        printRuntimeOk = true;
    }
    else {
        if (!printDensityOk) {
            JTRACE("BUILD", "FATAL: missing spectral data (print profile) after glare processing");
        }
        else {
            JTRACE("BUILD", "FATAL: print profile invalid or viewing illuminant missing");
        }
        target->tablesPrint = Spectral::SpectralTables{};
    }

    Spectral::Curve illumScan = negativeScannerIlluminant.curve;
    if (illumScan.linear.size() != static_cast<size_t>(Spectral::gShape.K)) {
        JTRACE("BUILD", "FATAL: scanner illuminant for negative medium is invalid");
        return;
    }
    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ illumScan,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        baselineMixReference,
        target->tablesScan,
        negativeScannerIlluminant.hash);

    if (hasRefIlluminant && target->tablesRef.K > 0) {
        const bool validWhite =
            std::isfinite(target->tablesRef.whiteXYZ[0]) &&
            std::isfinite(target->tablesRef.whiteXYZ[1]) &&
            std::isfinite(target->tablesRef.whiteXYZ[2]) &&
            target->tablesRef.whiteXYZ[1] > 0.0f &&
            approx_equal(static_cast<double>(target->tablesRef.whiteXYZ[1]), 1.0, 1e-4);

        const bool validRefWhite =
            std::isfinite(target->tablesRef.refIllumWhiteXYZ[0]) &&
            std::isfinite(target->tablesRef.refIllumWhiteXYZ[1]) &&
            std::isfinite(target->tablesRef.refIllumWhiteXYZ[2]) &&
            target->tablesRef.refIllumWhiteXYZ[1] > 0.0f &&
            approx_equal(static_cast<double>(target->tablesRef.refIllumWhiteXYZ[1]), 1.0, 1e-4);

        if (validWhite && validRefWhite) {
            Spectral::compute_S_inverse_from_tables(target->tablesRef, target->spdSInv);
            target->spdReady = true;
        }
        else {
            target->spdReady = false;
            JTRACE("BUILD", "Reference white XYZ validation failed; disabling SPD exposure");
        }
    }
    else {
        target->spdReady = false;
    }

    if (target->tablesView.K <= 0) {
        JTRACE("BUILD", "tablesView not ready; will cause wsReady=0 in render.");
    }
    if (hasRefIlluminant) {
        if (!target->spdReady || target->tablesRef.K <= 0) {
            JTRACE("BUILD", "tablesRef or spdSInv not ready; SPD exposure disabled.");
        }
    }
    else {
        JTRACE("BUILD", "reference illuminant missing; SPD exposure disabled.");
    }

    {
        auto all_finite_curve = [](const Spectral::Curve& c)->bool {
            for (float v : c.linear) {
                if (!std::isfinite(v)) {
                    return false;
                }
            }
            return true;
            };
        auto density_curve_ok = [](const Spectral::Curve& c)->bool {
            // agx-emulsion parity: density curves may contain toe NaNs; allow NaNs but reject
            // infinities and require at least one finite sample for calibration.
            bool anyFinite = false;
            for (float v : c.linear) {
                if (std::isinf(v)) {
                    return false;
                }
                if (std::isfinite(v)) {
                    anyFinite = true;
                }
            }
            return anyFinite;
            };

        const bool ok_dens = density_curve_ok(densB) && density_curve_ok(densG) && density_curve_ok(densR);
        const bool ok_sens = all_finite_curve(sensB) && all_finite_curve(sensG) && all_finite_curve(sensR);
        const bool baseMidOk = baseMid.linear.empty() ||
            static_cast<int>(baseMid.linear.size()) == Spectral::gShape.K;
        const bool ok_base = !hasBaseline ||
            (static_cast<int>(baseMin.linear.size()) == Spectral::gShape.K && baseMidOk);
        const bool ok_tables =
            (target->tablesView.K == Spectral::gShape.K) &&
            (target->tablesScan.K == Spectral::gShape.K) &&
            (!target->spdReady || target->tablesRef.K == Spectral::gShape.K);

        std::ostringstream oss;
        oss << "pre-Ecal: ok_dens=" << (ok_dens ? 1 : 0)
            << " ok_sens=" << (ok_sens ? 1 : 0)
            << " ok_base=" << (ok_base ? 1 : 0)
            << " ok_tables=" << (ok_tables ? 1 : 0)
            << " spdReady=" << (target->spdReady ? 1 : 0);
        JTRACE("BUILD", oss.str());

        if (!(ok_dens && ok_sens && ok_base && ok_tables)) {
            JTRACE("BUILD", "pre-Ecal: invalid inputs; aborting rebuild to avoid crash");
            return;
        }
    }

    {
        bool ok_spd = true;
        for (int i = 0; i < 9; ++i) {
            if (!std::isfinite(target->spdSInv[i])) { ok_spd = false; break; }
        }
        const bool ok_invYn =
            std::isfinite(target->tablesView.invYn) && target->tablesView.invYn > 0.0f &&
            std::isfinite(target->tablesScan.invYn) && target->tablesScan.invYn > 0.0f &&
            (!target->spdReady || (std::isfinite(target->tablesRef.invYn) && target->tablesRef.invYn > 0.0f));

        std::ostringstream oss;
        oss << "pre-Ecal: ok_spd=" << (ok_spd ? 1 : 0)
            << " ok_invYn=" << (ok_invYn ? 1 : 0);
        JTRACE("BUILD", oss.str());

        if (!ok_spd || !ok_invYn) {
            JTRACE("BUILD", "pre-Ecal: invalid S_inv or invYn; aborting rebuild");
            return;
        }
    }

    std::array<float, 3> densityMidRGB{ {0.0f, 0.0f, 0.0f} };
    bool hasDensityMid = !S.base.densityMidNeutral.empty();
    if (hasDensityMid) {
        float seed = 0.0f;
        if (std::isfinite(S.base.densityMidNeutral.front())) {
            seed = S.base.densityMidNeutral.front();
        }
        densityMidRGB.fill(seed);
        const size_t count = std::min<size_t>(static_cast<size_t>(3), S.base.densityMidNeutral.size());
        for (size_t i = 0; i < count; ++i) {
            const float v = S.base.densityMidNeutral[i];
            if (std::isfinite(v)) {
                densityMidRGB[i] = v;
            }
        }
    }
    std::array<float, 3> densityOffsets{ {0.0f, 0.0f, 0.0f} };
    if (hasDensityMid) {
        densityOffsets = RebuildWorkingState::compute_mid_neutral_logE_offsets_rgb(
            densRForCalibration, densGForCalibration, densBForCalibration, densityMidRGB);
    }

    std::array<float, 3> logMidRGB{ {0.0f, 0.0f, 0.0f} };
    bool hasLogEMid = !S.base.logExposureMidNeutral.empty();
    if (hasLogEMid) {
        float seed = 0.0f;
        bool seedValid = false;
        if (std::isfinite(S.base.logExposureMidNeutral.front())) {
            seed = S.base.logExposureMidNeutral.front();
            seedValid = true;
        }
        logMidRGB.fill(seed);
        size_t finiteCount = seedValid ? 1u : 0u;
        const size_t count = std::min<size_t>(static_cast<size_t>(3), S.base.logExposureMidNeutral.size());
        for (size_t i = 0; i < count; ++i) {
            const float v = S.base.logExposureMidNeutral[i];
            if (std::isfinite(v)) {
                logMidRGB[i] = v;
                ++finiteCount;
            }
        }
        if (!seedValid && finiteCount == 0u) {
            hasLogEMid = false;
            logMidRGB = { {0.0f, 0.0f, 0.0f} };
        }
    }

    std::array<float, 3> offsetsRGB{ {0.0f, 0.0f, 0.0f} };
    bool usingLogEMetadata = false;
    if (hasLogEMid) {
        offsetsRGB = logMidRGB;
        usingLogEMetadata = true;
    }
    else if (hasDensityMid) {
        offsetsRGB = densityOffsets;
    }

    const float offR = offsetsRGB[0];
    const float offG = offsetsRGB[1];
    const float offB = offsetsRGB[2];

    {
        std::ostringstream oss;
        if (usingLogEMetadata) {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (log_exposure_midscale_neutral)";
            if (hasDensityMid) {
                oss << " density_midscale_neutral=" << densityOffsets[2] << "/"
                    << densityOffsets[1] << "/" << densityOffsets[0];
            }
        }
        else if (hasDensityMid) {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (density_midscale_neutral)";
        }
        else {
            oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR
                << " (no midscale metadata)";
        }
        JTRACE("BUILD", oss.str());
    }

    target->densB = std::move(densB);
    target->densG = std::move(densG);
    target->densR = std::move(densR);
    if (precorrectApplied) {
        target->dirDensB = std::move(dirDensB);
        target->dirDensG = std::move(dirDensG);
        target->dirDensR = std::move(dirDensR);
    }
    else {
        target->dirDensB = target->densB;
        target->dirDensG = target->densG;
        target->dirDensR = target->densR;
    }
    target->sensB = std::move(sensB);
    target->sensG = std::move(sensG);
    target->sensR = std::move(sensR);
    // negSensB/G/R removed: profiles are pre-balanced, no "before balance" state needed.
    // Per agx-emulsion parity, sensitivities are used as-is from profiles.
    target->negSensB.linear.clear();
    target->negSensG.linear.clear();
    target->negSensR.linear.clear();
    target->baseMin = std::move(baseMin);
    target->baseMid = std::move(baseMid);
    target->hasBaseline = hasBaseline;
    target->baselineMixReference = baselineMixReference;
    target->printBaselineMixReference = printBaselineMixReference;

    // Copy per-channel gamma factors for density curve interpolation (agx-emulsion parity)
    target->gammaFactorB = S.base.gammaFactor[0];
    target->gammaFactorG = S.base.gammaFactor[1];
    target->gammaFactorR = S.base.gammaFactor[2];

    target->dirRT = dirRT;
    target->dirPrecorrected = precorrectApplied;
    target->dMax[0] = dirRT.dMax[0];
    target->dMax[1] = dirRT.dMax[1];
    target->dMax[2] = dirRT.dMax[2];

    negativeRangeOk = compute_negative_density_range(
        target->densB, target->densG, target->densR, target->grain, negativeDensityRange);
    if (!negativeRangeOk) {
        return;
    }

    S.spatialSigmaCacheValid.store(false, std::memory_order_release);

    {
        RebuildWorkingState::NegativeReuseContext reuseCtx;
        reuseCtx.activeBuildCounter = S.activeBuildCounter;
        reuseCtx.lastHash = S.lastHash.load(std::memory_order_acquire);
        reuseCtx.lastFilmStock = S.lastParams.filmStockIndex;
        reuseCtx.lastEnlargerIll = S.lastParams.enlIll;

        const std::shared_ptr<const WorkingState> prev = JuicerAtomic::load_shared_ptr(&S.activeWorkingState);
        if (RebuildWorkingState::can_reuse_negative_params(
            reuseCtx, prev.get(), *target, P.filmStockIndex, P.enlIll)) {
            // Only reuse metadata that is guaranteed to be identical. Density
            // curves and dMax are left untouched so freshly computed values stay
            // active after rebuilds (agx-emulsion parity for stock/illuminant swaps).
            target->negParams = prev->negParams;
        }
    }

    {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float v = target->dirRT.M[r][c];
                if (!std::isfinite(v)) v = 0.0f;
                if (v < -10.0f) v = -10.0f;
                if (v > 10.0f)  v = 10.0f;
                target->dirRT.M[r][c] = v;
            }
        }
        for (int i = 0; i < 3; ++i) {
            float v = target->dMax[i];
            if (!std::isfinite(v) || v <= 1e-4f) v = 1.0f;
            if (v > 1000.0f) v = 1000.0f;
            target->dMax[i] = v;
            target->dirRT.dMax[i] = v;
        }
    }

    target->filmRaw = Spectral::FilmRawConfig{};
    target->filmRaw.inputColorSpace = Spectral::inputColorSpaceFromIndex(P.inputColorSpace);
    target->filmRaw.applyCctfDecoding = (P.inputCctfDecoding != 0);
    target->filmRaw.spectralUpsamplingMode = Spectral::spectral_upsampling_mode_from_index(P.spectralUpsamplingMode);
    Spectral::prepare_film_raw_config(target->filmRaw);

    if (target->spdReady && target->tablesRef.K > 0) {
        for (int i = 0; i < 3; ++i) {
            target->filmRaw.refIllumWhiteXYZ[i] = target->tablesRef.refIllumWhiteXYZ[i];
        }
        target->filmRaw.hasRefIllumWhite = true;
    }
    else {
        target->filmRaw.refIllumWhiteXYZ[0] = Spectral::gDWG_WhitePoint_XYZ[0];
        target->filmRaw.refIllumWhiteXYZ[1] = Spectral::gDWG_WhitePoint_XYZ[1];
        target->filmRaw.refIllumWhiteXYZ[2] = Spectral::gDWG_WhitePoint_XYZ[2];
        target->filmRaw.hasRefIllumWhite = false;
    }

    // Per agx-emulsion parity: use the same sensitivities everywhere (no separate "before balance" state).
    // Profiles contain pre-balanced sensitivities; mid-gray computation must match actual render.
    const Spectral::SpectralTables* tablesSPD =
        (target->spdReady && target->tablesRef.K > 0) ? &target->tablesRef : nullptr;
    const float* sInv = target->spdReady ? target->spdSInv : nullptr;
    Spectral::compute_film_raw_midgray(
        target->filmRaw,
        tablesSPD,
        sInv,
        target->sensB,
        target->sensG,
        target->sensR);

    {
        std::ostringstream oss;
        oss << "film raw midgray scale=" << target->filmRaw.midgrayScale
            << " rawMidGreen=" << target->filmRaw.rawMidgrayGreen;
        JTRACE("BUILD", oss.str());
    }

    target->printRT = std::move(printRuntimeCopy);
    target->printGlare = target->printRT ? target->printRT->glare : Profiles::ProfileGlare{};
    target->negativeScannerIlluminant = negativeScannerIlluminant;
    target->printScannerIlluminant = printScannerIlluminant;
    target->negativeDensityRange = negativeDensityRange;
    target->printDensityRange = printDensityRange;

    target->negativeScannerValid = true;
    target->printScannerValid = printRuntimeOk;
    target->printGlareCompensated = (printRuntimeOk && printProfile.glare.compensationRemovalFactor > 0.0f);
    if (!rebuild_working_state_scanner_output_runtime(P, *target)) {
        return;
    }

    target->fullHash = hash_params(P);
    target->uploadCoreHash = hash_params_upload_core(P);
    target->coreHash = coreShareHash;
    target->coreShareHash = coreShareHash;
    target->dirHash = hash_params_dir(P);
    target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        auto corePayload = std::make_shared<WorkingStateSharing::WorkingStateCorePayload>();
        WorkingStateSharing::capture_working_state_core_payload(*target, *corePayload);
        const WorkingStateSharing::AcquireCoreSharedResult coreShareSeed =
            WorkingStateSharing::acquire_or_create_shared_core(target->coreShareHash, std::move(corePayload));
        target->sharedCore = coreShareSeed.sharedCore;
        trace_working_state_core_share(coreShareSeed, target->buildCounter, "full_rebuild");
    }
    if (JTRACE_ENABLED(3)) {
        const char* paperKey = print_paper_json_key_for_index(P.printPaperIndex);
        const char* filmKey = negative_json_key_for_stock_index(P.filmStockIndex);
        const std::uintptr_t prtPtr = reinterpret_cast<std::uintptr_t>(target->printRT.get());
        const float neutralY = target->printRT ? target->printRT->neutralY : 0.0f;
        const float neutralM = target->printRT ? target->printRT->neutralM : 0.0f;
        const float neutralC = target->printRT ? target->printRT->neutralC : 0.0f;
        std::string msg = std::string("working state commit build=") + std::to_string(target->buildCounter)
            + " paper=" + std::string(paperKey ? paperKey : "<null>")
            + " film=" + std::string(filmKey ? filmKey : "<null>")
            + " printRT=" + std::to_string(prtPtr)
            + " neutralY/M/C=" + std::to_string(neutralY) + "/" + std::to_string(neutralM) + "/" + std::to_string(neutralC)
            + " printRef=" + (target->printRT ? target->printRT->referenceIlluminant : std::string("<null>"))
            + " printView=" + (target->printRT ? target->printRT->viewingIlluminant : std::string("<null>"));
        JTRACE_VERBOSE("PRINTDBG", msg);
    }

    {
        std::ostringstream oss;
        oss << "WorkingState build #" << target->buildCounter;
        JTRACE("BUILD", oss.str());
    }

    JuicerAtomic::store_shared_ptr(&S.activeWorkingState, std::shared_ptr<const WorkingState>(next));
    {
        std::ostringstream oss;
        oss << "activeWorkingState swapped; buildCounter=" << static_cast<long long>(target->buildCounter);
        JTRACE("BUILD", oss.str());
    }
    S.activeBuildCounter = target->buildCounter;
    S.lastParams = P;
    S.lastHash.store(target->fullHash, std::memory_order_release);

}

void rebuild_working_state_couplers_only(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {
    (void)instance;

#ifndef JUICER_ENABLE_COUPLERS
    rebuild_working_state(instance, S, P);
    return;
#else
    JTRACE_SCOPE("BUILD", "rebuild_working_state_couplers_only");

    std::unique_lock<std::mutex> lk(S.m);

    std::shared_ptr<WorkingState> next = std::make_shared<WorkingState>();
    WorkingState* target = next.get();
    const std::uint64_t coreShareHash = hash_params_core(P);
    WorkingStateSharing::AcquireCoreSharedResult coreShare =
        WorkingStateSharing::acquire_or_create_shared_core(coreShareHash);
    std::shared_ptr<const WorkingState> src;
    if (!(coreShare.sharedCore && coreShare.sharedCore->payload)) {
        src = JuicerAtomic::load_shared_ptr(&S.activeWorkingState);
        if (!src || src->buildCounter == 0) {
            lk.unlock();
            rebuild_working_state(instance, S, P);
            return;
        }
        auto payloadSeed = std::make_shared<WorkingStateSharing::WorkingStateCorePayload>();
        WorkingStateSharing::capture_working_state_core_payload(*src, *payloadSeed);
        coreShare = WorkingStateSharing::acquire_or_create_shared_core(coreShareHash, std::move(payloadSeed));
    }
    if (coreShare.sharedCore && coreShare.sharedCore->payload) {
        WorkingStateSharing::apply_working_state_core_payload(*coreShare.sharedCore->payload, *target);
    }
    else {
        WorkingStateSharing::WorkingStateCorePayload fallbackPayload{};
        WorkingStateSharing::capture_working_state_core_payload(*src, fallbackPayload);
        WorkingStateSharing::apply_working_state_core_payload(fallbackPayload, *target);
    }
    target->coreShareHash = coreShareHash;
    target->sharedCore = coreShare.sharedCore;
    recompute_working_state_dir_overlay(S, P, *target);
    S.spatialSigmaCacheValid.store(false, std::memory_order_release);
    if (!rebuild_working_state_scanner_output_runtime(P, *target)) {
        lk.unlock();
        rebuild_working_state(instance, S, P);
        return;
    }

    target->fullHash = hash_params(P);
    target->uploadCoreHash = hash_params_upload_core(P);
    target->coreHash = coreShareHash;
    target->coreShareHash = coreShareHash;
    target->dirHash = hash_params_dir(P);
    target->buildCounter = S.buildCounterNext.fetch_add(1, std::memory_order_relaxed) + 1;
    target->sharedCore = coreShare.sharedCore;
    trace_working_state_core_share(coreShare, target->buildCounter, "couplers_only");

    JuicerAtomic::store_shared_ptr(&S.activeWorkingState, std::shared_ptr<const WorkingState>(next));
    S.activeBuildCounter = target->buildCounter;
    S.lastParams = P;
    S.lastHash.store(target->fullHash, std::memory_order_release);
#endif
}
