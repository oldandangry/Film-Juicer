#include "ProfileJSONLoader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <string_view>

#include "nlohmann/json.hpp"
#include "Logging.h"
#include "IlluminantKeys.h"
#include "SpectralTypes.h"
#include "SpectralData.h"

namespace Profiles {
    namespace {

        using Json = nlohmann::json;

        bool is_identifier_char(char c) {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '#';
        }

        bool has_separator_before(const std::string& text, std::size_t index) {
            if (index == 0) {
                return true;
            }
            const char prev = text[index - 1];
            const unsigned char uprev = static_cast<unsigned char>(prev);
            if (std::isspace(uprev)) {
                return true;
            }
            switch (prev) {
            case ',':
            case ':':
            case '[':
            case ']':
            case '{':
            case '}':
            case '(':
            case ')':
            case '+':
            case '-':
            case '=':
                return true;
            default:
                break;
            }
            return !is_identifier_char(prev);
        }

        bool matches_case_insensitive(const std::string& text, std::size_t index, std::string_view token) {
            if (index + token.size() > text.size()) {
                return false;
            }
            for (std::size_t j = 0; j < token.size(); ++j) {
                const unsigned char tc = static_cast<unsigned char>(text[index + j]);
                const unsigned char lc = static_cast<unsigned char>(token[j]);
                if (std::tolower(tc) != lc) {
                    return false;
                }
            }
            return true;
        }

        void sanitize_non_finite_literals(std::string& text) {
            // Replace bare non-finite tokens with string sentinels that will be
            // accepted by the JSON parser and decoded later.
            static constexpr std::array<std::pair<std::string_view, std::string_view>, 33> kReplacements{ {
                { "-infinity", "\"__-inf__\"" },
                { "+infinity", "\"__inf__\"" },
                { "infinity",  "\"__inf__\"" },
                { "-inf",      "\"__-inf__\"" },
                { "+inf",      "\"__inf__\"" },
                { "inf",       "\"__inf__\"" },
                { "-nan",      "\"__nan__\"" },
                { "+nan",      "\"__nan__\"" },
                { "nan",       "\"__nan__\"" },
                { "-1.#inf",   "\"__-inf__\"" },
                { "+1.#inf",   "\"__inf__\"" },
                { "1.#inf",    "\"__inf__\"" },
                { "-1.#ind",   "\"__-inf__\"" },
                { "+1.#ind",   "\"__inf__\"" },
                { "1.#ind",    "\"__inf__\"" },
                { "-1.#nan",   "\"__nan__\"" },
                { "+1.#nan",   "\"__nan__\"" },
                { "1.#nan",    "\"__nan__\"" },
                { "-1.#qnan",  "\"__nan__\"" },
                { "+1.#qnan",  "\"__nan__\"" },
                { "1.#qnan",   "\"__nan__\"" },
                { "-1.#snan",  "\"__nan__\"" },
                { "+1.#snan",  "\"__nan__\"" },
                { "1.#snan",   "\"__nan__\"" },
                { "-nan(ind)", "\"__nan__\"" },
                { "+nan(ind)", "\"__nan__\"" },
                { "nan(ind)",  "\"__nan__\"" },
                { "-nan(qnan)","\"__nan__\"" },
                { "+nan(qnan)","\"__nan__\"" },
                { "nan(qnan)", "\"__nan__\"" },
                { "-nan(snan)","\"__nan__\"" },
                { "+nan(snan)","\"__nan__\"" },
                { "nan(snan)", "\"__nan__\"" }
            } };

            bool inString = false;
            bool escaping = false;

            for (std::size_t i = 0; i < text.size();) {
                const char c = text[i];
                if (inString) {
                    if (escaping) {
                        escaping = false;
                        ++i;
                        continue;
                    }
                    if (c == '\\') {
                        escaping = true;
                        ++i;
                        continue;
                    }
                    if (c == '"') {
                        inString = false;
                    }
                    ++i;
                    continue;
                }

                if (c == '"') {
                    inString = true;
                    ++i;
                    continue;
                }

                bool replaced = false;
                for (const auto& [token, replacement] : kReplacements) {
                    if (!matches_case_insensitive(text, i, token)) {
                        continue;
                    }

                    std::size_t bodyStart = 0;
                    while (bodyStart < token.size() && (token[bodyStart] == '+' || token[bodyStart] == '-')) {
                        ++bodyStart;
                    }
                    const bool requiresSeparator = bodyStart < token.size() &&
                        (std::isdigit(static_cast<unsigned char>(token[bodyStart])) || token[bodyStart] == '#');
                    if (requiresSeparator && !has_separator_before(text, i)) {
                        continue;
                    }

                    const std::size_t end = i + token.size();
                    const bool hasPrev = i > 0 && is_identifier_char(text[i - 1]);
                    const bool hasNext = end < text.size() && is_identifier_char(text[end]);
                    if (!hasPrev && !hasNext) {
                        text.replace(i, token.size(), replacement);
                        i += replacement.size();
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    ++i;
                }
            }
        }

        bool parse_json_file(const std::string& path, Json& out) {
            errno = 0;
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                const int err = errno;
                std::string reason;
                if (err != 0) {
                    reason = std::system_category().message(err);
                }
                if (reason.empty()) {
                    reason = "unknown error";
                }
                JTRACE("PROFILE", std::string("failed to open profile '") + path + "': " + reason);
                return false;
            }
            std::ostringstream oss;
            oss << file.rdbuf();
            std::string text = oss.str();
            sanitize_non_finite_literals(text);

            // Upstream agx-emulsion allows NaN/Inf literals via Python json (allow_nan=True).
            // Keep them intact so downstream NaN handling mirrors agx.
            try {
                out = Json::parse(text,
                    /*cb*/nullptr,
                    /*allow_exceptions*/true,
                    /*ignore_comments*/true);
            }
            catch (const Json::parse_error& e) {
                JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': " + e.what());
                return false;
            }
            catch (const std::exception& e) {
                JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': " + e.what());
                return false;
            }
            catch (...) {
                JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': unknown exception");
                return false;
            }

            if (out.is_discarded()) {
                JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': parser discarded document");
                return false;
            }

            return true;
        }

        std::optional<float> parse_optional_float(const Json& node) {
            if (node.is_number_float() || node.is_number_integer()) {
                float v = static_cast<float>(node.get<double>());
                if (std::isfinite(v)) {
                    return v;
                }
            }
            else if (node.is_string()) {
                const std::string& str = node.get_ref<const std::string&>();
                const char* begin = str.c_str();
                char* end = nullptr;
                errno = 0;
                const float v = std::strtof(begin, &end);
                if (end != begin && end == begin + str.size() && errno == 0 && std::isfinite(v)) {
                    return v;
                }
            }
            else if (node.is_boolean()) {
                return node.get<bool>() ? 1.0f : 0.0f;
            }
            return std::nullopt;
        }

        std::optional<float> decode_nonfinite_sentinel(const Json& node) {
            if (!node.is_string()) {
                return std::nullopt;
            }
            const std::string& str = node.get_ref<const std::string&>();
            if (str == "__nan__") {
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (str == "__inf__") {
                return std::numeric_limits<float>::infinity();
            }
            if (str == "__-inf__") {
                return -std::numeric_limits<float>::infinity();
            }
            return std::nullopt;
        }

        std::optional<float> parse_optional_float_allow_nan(const Json& node) {
            if (auto special = decode_nonfinite_sentinel(node)) {
                return special;
            }
            if (node.is_number_float() || node.is_number_integer()) {
                float v = static_cast<float>(node.get<double>());
                if (std::isfinite(v) || std::isnan(v)) {
                    return v;
                }
            }
            else if (node.is_string()) {
                const std::string& str = node.get_ref<const std::string&>();
                const char* begin = str.c_str();
                char* end = nullptr;
                errno = 0;
                const float v = std::strtof(begin, &end);
                if (end != begin && end == begin + str.size() && errno == 0 &&
                    (std::isfinite(v) || std::isnan(v))) {
                    return v;
                }
            }
            else if (node.is_boolean()) {
                return node.get<bool>() ? 1.0f : 0.0f;
            }
            return std::nullopt;
        }

        std::string to_lower_ascii(std::string value) {
            for (char& c : value) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            return value;
        }

        bool json_wavelengths_match_reference_axis(const Json& wavelengths, std::string_view sourceLabel) {
            const std::string labelStr = sourceLabel.empty()
                ? std::string("profile JSON")
                : std::string(sourceLabel);

            if (!wavelengths.is_array()) {
                JTRACE("PROFILE", "JSON wavelengths not an array (" + labelStr + ')');
                return false;
            }
            if (wavelengths.size() != Spectral::kNumSamples) {
                std::ostringstream oss;
                oss << "JSON wavelengths mismatch (" << labelStr << "): expected "
                    << Spectral::kNumSamples << " samples, got " << wavelengths.size();
                JTRACE("PROFILE", oss.str());
                return false;
            }

            for (int i = 0; i < Spectral::kNumSamples; ++i) {
                std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
                if (!wlOpt) {
                    std::ostringstream oss;
                    oss << "JSON wavelength missing at index " << i
                        << " (" << labelStr << ')';
                    JTRACE("PROFILE", oss.str());
                    return false;
                }
                const float expected = Spectral::kLambdaMin + static_cast<float>(i) * Spectral::kDelta;
                if (*wlOpt != expected) {
                    std::ostringstream oss;
                    oss << "JSON wavelength mismatch at index " << i
                        << " (" << labelStr << "): expected "
                        << expected << "nm, got " << *wlOpt << "nm";
                    JTRACE("PROFILE", oss.str());
                    return false;
                }
            }

            return true;
        }

        bool append_pair_if_present(std::vector<std::pair<float, float>>& target, float x, const Json* node) {
            float value = 0.0f;
            bool ok = false;
            if (node && !node->is_null()) {
                if (auto opt = parse_optional_float(*node)) {
                    value = *opt;
                    ok = true;
                }
            }
            target.emplace_back(x, value);
            return ok;
        }

        bool append_pair_if_present(std::vector<std::pair<float, float>>& target, float x, const Json& node) {
            return append_pair_if_present(target, x, &node);
        }

        bool append_spectral_pair_with_nan_if_missing(
            std::vector<std::pair<float, float>>& target,
            float x,
            const Json* node)
        {
            if (node && !node->is_null()) {
                if (auto opt = parse_optional_float(*node)) {
                    target.emplace_back(x, *opt);
                    return true;
                }
            }
            target.emplace_back(x, std::numeric_limits<float>::quiet_NaN());
            return false;
        }

        size_t count_finite_samples(const std::vector<std::pair<float, float>>& samples) {
            size_t count = 0;
            for (const auto& sample : samples) {
                if (std::isfinite(sample.first) && std::isfinite(sample.second)) {
                    ++count;
                }
            }
            return count;
        }

        void parse_dir_couplers(const Json& node, DirCouplersProfile& outProfile) {
            outProfile = DirCouplersProfile{};
            if (!node.is_object()) {
                return;
            }

            bool any = false;

            if (node.contains("active") && node["active"].is_boolean()) {
                outProfile.active = node["active"].get<bool>();
                any = true;
            }
            if (auto amount = parse_optional_float(node.value("amount", Json{}))) {
                outProfile.amount = *amount;
                any = true;
            }
            if (node.contains("ratio_rgb") && node["ratio_rgb"].is_array()) {
                const Json& arr = node["ratio_rgb"];
                for (size_t i = 0; i < std::min<size_t>(3, arr.size()); ++i) {
                    if (auto val = parse_optional_float(arr[i])) {
                        outProfile.ratioRGB[i] = *val;
                        any = true;
                    }
                }
            }
            if (auto diff = parse_optional_float(node.value("diffusion_interlayer", Json{}))) {
                outProfile.diffusionInterlayer = *diff;
                any = true;
            }
            if (auto diffSize = parse_optional_float(node.value("diffusion_size_um", Json{}))) {
                outProfile.diffusionSizeUm = *diffSize;
                any = true;
            }
            if (auto high = parse_optional_float(node.value("high_exposure_shift", Json{}))) {
                outProfile.highExposureShift = *high;
                any = true;
            }

            outProfile.hasData = any;
        }

        void parse_masking_couplers(const Json& node, MaskingCouplersProfile& outProfile) {
            outProfile = MaskingCouplersProfile{};
            if (!node.is_object()) {
                return;
            }

            bool any = false;

            if (node.contains("cross_over_points") && node["cross_over_points"].is_array()) {
                const Json& arr = node["cross_over_points"];
                outProfile.crossOverPoints.clear();
                for (const auto& v : arr) {
                    if (auto val = parse_optional_float(v)) {
                        outProfile.crossOverPoints.push_back(*val);
                        any = true;
                    }
                }
            }

            if (node.contains("transition_widths") && node["transition_widths"].is_array()) {
                const Json& arr = node["transition_widths"];
                outProfile.transitionWidths.clear();
                for (const auto& v : arr) {
                    if (auto val = parse_optional_float(v)) {
                        outProfile.transitionWidths.push_back(*val);
                        any = true;
                    }
                }
            }

            if (node.contains("gaussian_model") && node["gaussian_model"].is_array()) {
                const Json& gm = node["gaussian_model"];
                for (size_t ch = 0; ch < std::min<size_t>(3, gm.size()); ++ch) {
                    const Json& channel = gm[ch];
                    auto& dest = outProfile.gaussianModel[ch];
                    dest.clear();
                    if (!channel.is_array()) {
                        continue;
                    }
                    for (const auto& peak : channel) {
                        if (!peak.is_array() || peak.size() < 3) {
                            continue;
                        }
                        std::array<float, 3> triplet{ 0.0f, 0.0f, 0.0f };
                        bool ok = true;
                        for (size_t k = 0; k < 3; ++k) {
                            if (auto val = parse_optional_float(peak[k])) {
                                triplet[k] = *val;
                            }
                            else {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            dest.push_back(triplet);
                            any = true;
                        }
                    }
                }
            }

            outProfile.hasData = any;
        }

        bool parse_filter_triplet(const Json& node, std::array<float, 3>& dest) {
            std::array<float, 3> result = dest;
            bool any = false;

            if (node.is_array()) {
                const size_t count = std::min<size_t>(3, node.size());
                for (size_t i = 0; i < count; ++i) {
                    if (auto val = parse_optional_float(node[i])) {
                        result[i] = *val;
                        any = true;
                    }
                }
            }
            else if (node.is_object()) {
                auto try_key = [&](const char* key, size_t idx) {
                    auto it = node.find(key);
                    if (it != node.end()) {
                        if (auto val = parse_optional_float(*it)) {
                            result[idx] = *val;
                            any = true;
                        }
                    }
                    };

                try_key("amplitude", 0);
                try_key("amp", 0);
                try_key("strength", 0);
                try_key("wavelength", 1);
                try_key("center", 1);
                try_key("cutoff", 1);
                try_key("nm", 1);
                try_key("sigma", 2);
                try_key("width", 2);
                try_key("transition", 2);
            }
            else if (auto scalar = parse_optional_float(node)) {
                result[0] = *scalar;
                any = true;
            }

            if (any) {
                dest = result;
            }
            return any;
        }

    } // namespace

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile) {
        outProfile = AgxFilmProfile{};

        Json root;
        if (!parse_json_file(jsonPath, root)) {
            return false;
        }

        if (root.contains("info")) {
            const Json& info = root["info"];
            if (info.contains("densitometer") && info["densitometer"].is_string()) {
                outProfile.densitometer = info["densitometer"].get<std::string>();
            }
            if (info.contains("type") && info["type"].is_string()) {
                outProfile.type = to_lower_ascii(info["type"].get<std::string>());
            }
            if (info.contains("reference_illuminant") && info["reference_illuminant"].is_string()) {
                const std::string raw = info["reference_illuminant"].get<std::string>();
                const std::string normalized = IlluminantKeys::normalize(raw);
                outProfile.referenceIlluminant = normalized.empty() ? raw : normalized;
            }
            if (info.contains("viewing_illuminant") && info["viewing_illuminant"].is_string()) {
                const std::string raw = info["viewing_illuminant"].get<std::string>();
                const std::string normalized = IlluminantKeys::normalize(raw);
                outProfile.viewingIlluminant = normalized.empty() ? raw : normalized;
            }
            if (info.contains("density_midscale_neutral")) {
                const Json& mid = info["density_midscale_neutral"];
                outProfile.densityMidNeutral.clear();
                if (mid.is_array()) {
                    for (const auto& v : mid) {
                        if (auto val = parse_optional_float(v)) {
                            outProfile.densityMidNeutral.push_back(*val);
                        }
                    }
                }
                else if (auto val = parse_optional_float(mid)) {
                    outProfile.densityMidNeutral.push_back(*val);
                }
            }
            if (info.contains("log_exposure_midscale_neutral")) {
                const Json& mid = info["log_exposure_midscale_neutral"];
                outProfile.logExposureMidNeutral.clear();
                if (mid.is_array()) {
                    for (const auto& v : mid) {
                        if (auto val = parse_optional_float(v)) {
                            outProfile.logExposureMidNeutral.push_back(*val);
                        }
                    }
                }
                else if (auto val = parse_optional_float(mid)) {
                    outProfile.logExposureMidNeutral.push_back(*val);
                }
            }
        }

        if (root.contains("camera") && root["camera"].is_object()) {
            const Json& camera = root["camera"];
            std::array<float, 3> uv = outProfile.cameraFilterUV;
            std::array<float, 3> ir = outProfile.cameraFilterIR;
            const bool hasUV = camera.contains("filter_uv")
                ? parse_filter_triplet(camera["filter_uv"], uv)
                : false;
            const bool hasIR = camera.contains("filter_ir")
                ? parse_filter_triplet(camera["filter_ir"], ir)
                : false;
            if (hasUV) {
                outProfile.cameraFilterUV = uv;
                outProfile.hasCameraFilterUV = true;
            }
            if (hasIR) {
                outProfile.cameraFilterIR = ir;
                outProfile.hasCameraFilterIR = true;
            }
        }

        if (!root.contains("data")) {
            return false;
        }
        const Json& data = root["data"];

        if (data.contains("tune")) {
            const Json& tune = data["tune"];
            if (tune.contains("dye_density_min_factor")) {
                if (auto val = parse_optional_float(tune["dye_density_min_factor"])) {
                    outProfile.dyeDensityMinFactor = *val;
                }
            }
            if (tune.contains("gamma_factor")) {
                const Json& gf = tune["gamma_factor"];
                std::array<float, 3> gamma{ {1.0f, 1.0f, 1.0f} };
                bool any = false;
                if (auto scalar = parse_optional_float(gf)) {
                    const float v = *scalar;
                    if (std::isfinite(v) && v > 0.0f) {
                        gamma.fill(v);
                        any = true;
                    }
                }
                else if (gf.is_array()) {
                    const size_t count = std::min<size_t>(3, gf.size());
                    for (size_t i = 0; i < count; ++i) {
                        if (auto val = parse_optional_float(gf[i])) {
                            const float v = *val;
                            if (std::isfinite(v) && v > 0.0f) {
                                gamma[i] = v;
                                any = true;
                            }
                        }
                    }
                }
                if (any) {
                    outProfile.gammaFactor = gamma;
                    outProfile.hasGammaFactor = true;
                }
            }
        }

        if (!data.contains("wavelengths") || !data.contains("dye_density")) {
            return false;
        }

        const Json& wavelengths = data["wavelengths"];
        const Json& dyeDensity = data["dye_density"];
        if (!wavelengths.is_array() || !dyeDensity.is_array()) {
            return false;
        }
        const std::string wavelengthLabel = jsonPath.empty() ? std::string("profile JSON") : jsonPath;
        if (!json_wavelengths_match_reference_axis(wavelengths, wavelengthLabel)) {
            return false;
        }
        if (dyeDensity.size() != Spectral::kNumSamples) {
            JTRACE("PROFILE", "dye_density sample count mismatch; expected 81 samples");
            return false;
        }

        outProfile.dyeC.reserve(Spectral::kNumSamples);
        outProfile.dyeM.reserve(Spectral::kNumSamples);
        outProfile.dyeY.reserve(Spectral::kNumSamples);
        outProfile.baseMin.reserve(Spectral::kNumSamples);
        outProfile.baseMid.reserve(Spectral::kNumSamples);

        for (size_t i = 0; i < Spectral::kNumSamples; ++i) {
            const float wl = Spectral::kLambdaMin + static_cast<float>(i) * Spectral::kDelta;
            std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
            if (!wlOpt || *wlOpt != wl) {
                JTRACE("PROFILE", "JSON wavelength mismatch during ingest; rejecting profile");
                return false;
            }
            const Json& row = dyeDensity[i];
            if (!row.is_array()) {
                JTRACE("PROFILE", "dye_density row missing or not array; inserting NaNs");
                outProfile.dyeC.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.dyeM.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.dyeY.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.baseMin.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.baseMid.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                continue;
            }

            auto get = [&](size_t idx)->std::optional<float> {
                if (idx >= row.size()) return std::nullopt;
                return parse_optional_float_allow_nan(row[idx]);
            };
            auto cVal = get(0);
            auto mVal = get(1);
            auto yVal = get(2);
            auto baseMinVal = get(3);
            auto baseMidVal = get(4);

            outProfile.dyeC.emplace_back(wl, cVal.value_or(std::numeric_limits<float>::quiet_NaN()));
            outProfile.dyeM.emplace_back(wl, mVal.value_or(std::numeric_limits<float>::quiet_NaN()));
            outProfile.dyeY.emplace_back(wl, yVal.value_or(std::numeric_limits<float>::quiet_NaN()));
            outProfile.baseMin.emplace_back(wl, baseMinVal.value_or(std::numeric_limits<float>::quiet_NaN()));
            outProfile.baseMid.emplace_back(wl, baseMidVal.value_or(std::numeric_limits<float>::quiet_NaN()));
        }

        // Log sensitivity curves (RGB order in the source profile).
        if (!data.contains("log_sensitivity")) {
            return false;
        }
        const Json& logSens = data["log_sensitivity"];
        if (!logSens.is_array() || logSens.size() != Spectral::kNumSamples) {
            JTRACE("PROFILE", "log_sensitivity must be an array of 81 rows");
            return false;
        }
        outProfile.logSensR.reserve(Spectral::kNumSamples);
        outProfile.logSensG.reserve(Spectral::kNumSamples);
        outProfile.logSensB.reserve(Spectral::kNumSamples);
        for (size_t i = 0; i < Spectral::kNumSamples; ++i) {
            const float wl = Spectral::kLambdaMin + static_cast<float>(i) * Spectral::kDelta;
            std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
            if (!wlOpt || *wlOpt != wl) {
                JTRACE("PROFILE", "log_sensitivity wavelength mismatch; rejecting profile");
                return false;
            }
            const Json& row = logSens[i];
            if (!row.is_array() || row.size() != 3) {
                JTRACE("PROFILE", "log_sensitivity rows must contain exactly 3 channels");
                return false;
            }
            auto rVal = parse_optional_float_allow_nan(row[0]);
            auto gVal = parse_optional_float_allow_nan(row[1]);
            auto bVal = parse_optional_float_allow_nan(row[2]);
            if (!(rVal && gVal && bVal)) {
                JTRACE("PROFILE", "non-numeric log_sensitivity entry; rejecting profile");
                return false;
            }
            outProfile.logSensR.emplace_back(wl, *rVal);
            outProfile.logSensG.emplace_back(wl, *gVal);
            outProfile.logSensB.emplace_back(wl, *bVal);
        }

        // Density curves (log exposure domain shared by all channels).
        if (!data.contains("log_exposure") || !data.contains("density_curves")) {
            return false;
        }
        const Json& logExposure = data["log_exposure"];
        const Json& densityCurves = data["density_curves"];
        if (!logExposure.is_array() || !densityCurves.is_array()) {
            return false;
        }
        if (logExposure.size() != Spectral::kLogExposureSamples ||
            densityCurves.size() != Spectral::kLogExposureSamples) {
            JTRACE("PROFILE", "log_exposure/density_curves length mismatch; expected 256 samples");
            return false;
        }

        outProfile.densityCurveR.reserve(Spectral::kLogExposureSamples);
        outProfile.densityCurveG.reserve(Spectral::kLogExposureSamples);
        outProfile.densityCurveB.reserve(Spectral::kLogExposureSamples);

        float prevLogE = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < Spectral::kLogExposureSamples; ++i) {
            auto logEOpt = parse_optional_float(logExposure[i]);
            if (!logEOpt) {
                JTRACE("PROFILE", "non-numeric log_exposure entry; rejecting profile");
                return false;
            }
            const float logE = *logEOpt;
            if (!(logE > prevLogE)) {
                JTRACE("PROFILE", "log_exposure not strictly increasing; rejecting profile");
                return false;
            }
            prevLogE = logE;

            const Json& row = densityCurves[i];
            if (!row.is_array() || row.size() != 3) {
                JTRACE("PROFILE", "density_curves rows must contain exactly 3 channels");
                return false;
            }
            auto rVal = parse_optional_float_allow_nan(row[0]);
            auto gVal = parse_optional_float_allow_nan(row[1]);
            auto bVal = parse_optional_float_allow_nan(row[2]);
            if (!(rVal && gVal && bVal)) {
                JTRACE("PROFILE", "non-numeric density_curves entry; rejecting profile");
                return false;
            }
            outProfile.densityCurveR.emplace_back(logE, *rVal);
            outProfile.densityCurveG.emplace_back(logE, *gVal);
            outProfile.densityCurveB.emplace_back(logE, *bVal);
        }

        // Optional density_curves_layers (256 x 3 x 3)
        if (data.contains("density_curves_layers")) {
            const Json& layers = data["density_curves_layers"];
            if (!layers.is_array()) {
                JTRACE("PROFILE", "density_curves_layers present but not an array; rejecting profile");
                return false;
            }
            if (layers.size() != Spectral::kLogExposureSamples) {
                JTRACE("PROFILE", "density_curves_layers length mismatch; rejecting profile");
                return false;
            }
            outProfile.hasDensityCurvesLayers = true;
            for (size_t i = 0; i < layers.size(); ++i) {
                auto logEOpt = parse_optional_float(logExposure[i]);
                if (!logEOpt) {
                    JTRACE("PROFILE", "density_curves_layers log_exposure missing; rejecting profile");
                    return false;
                }
                const float logE = *logEOpt;
                const Json& row = layers[i];
                if (!row.is_array() || row.size() != 3) {
                    JTRACE("PROFILE", "density_curves_layers rows must contain 3 sublayers");
                    return false;
                }
                for (size_t layer = 0; layer < 3; ++layer) {
                    const Json& sub = row[layer];
                    if (!sub.is_array() || sub.size() != 3) {
                        JTRACE("PROFILE", "density_curves_layers sublayer must contain 3 channels");
                        return false;
                    }
                    for (size_t ch = 0; ch < 3; ++ch) {
                        auto v = parse_optional_float_allow_nan(sub[ch]);
                        if (!v) {
                            JTRACE("PROFILE", "non-numeric density_curves_layers entry; rejecting profile");
                            return false;
                        }
                        outProfile.densityCurvesLayers[layer][ch].emplace_back(logE, *v);
                    }
                }
            }
        }

        // Require the core spectral assets to be present.
        const bool haveDyes = !outProfile.dyeC.empty() &&
            !outProfile.dyeM.empty() && !outProfile.dyeY.empty();
        const bool haveCurves = !outProfile.densityCurveR.empty() &&
            !outProfile.densityCurveG.empty() && !outProfile.densityCurveB.empty();
        const bool haveSens = !outProfile.logSensR.empty() &&
            !outProfile.logSensG.empty() && !outProfile.logSensB.empty();
        if (!(haveDyes && haveCurves && haveSens)) {
            return false;
        }

        outProfile.type = to_lower_ascii(outProfile.type);
        const bool isPaper = outProfile.type == "paper" ||
            outProfile.type == "print" || outProfile.type == "print_paper";
        const bool isNegative = !isPaper;

        if (root.contains("dir_couplers")) {
            parse_dir_couplers(root["dir_couplers"], outProfile.dirCouplers);
        }

        if (root.contains("masking_couplers")) {
            parse_masking_couplers(root["masking_couplers"], outProfile.maskingCouplers);
        }

        auto log_glare_failure = [&](const std::string& field) {
            std::ostringstream oss;
            oss << "FATAL: missing or invalid glare field '" << field << "' in profile '" << jsonPath << "'";
            JTRACE("PROFILE", oss.str());
        };

        if (!root.contains("glare") || !root["glare"].is_object()) {
            log_glare_failure("glare");
            return false;
        }

        const Json& glareNode = root["glare"];
        ProfileGlare glare{};
        auto require_bool = [&](const char* key, bool& dst)->bool {
            auto it = glareNode.find(key);
            if (it != glareNode.end() && it->is_boolean()) {
                dst = it->get<bool>();
                return true;
            }
            log_glare_failure(key);
            return false;
        };
        auto require_float = [&](const char* key, float& dst)->bool {
            auto val = parse_optional_float(glareNode.value(key, Json{}));
            if (val && std::isfinite(*val)) {
                dst = *val;
                return true;
            }
            log_glare_failure(key);
            return false;
        };

        if (!require_bool("active", glare.active)) return false;
        if (!require_float("percent", glare.percent)) return false;
        if (!require_float("roughness", glare.roughness)) return false;
        if (!require_float("blur", glare.blur)) return false;

        if (isPaper) {
            if (!require_float("compensation_removal_factor", glare.compensationRemovalFactor)) return false;
            if (!require_float("compensation_removal_density", glare.compensationRemovalDensity)) return false;
            if (!require_float("compensation_removal_transition", glare.compensationRemovalTransition)) return false;
        }
        else {
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_factor", Json{}))) {
                if (std::isfinite(*val)) glare.compensationRemovalFactor = *val;
            }
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_density", Json{}))) {
                if (std::isfinite(*val)) glare.compensationRemovalDensity = *val;
            }
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_transition", Json{}))) {
                if (std::isfinite(*val)) glare.compensationRemovalTransition = *val;
            }
        }

        outProfile.glare = glare;
        outProfile.hasGlare = true;
        outProfile.hasGlareCompensation = true;
        outProfile.glareCompensationFactor = glare.compensationRemovalFactor;
        outProfile.glareCompensationDensity = glare.compensationRemovalDensity;
        outProfile.glareCompensationTransition = glare.compensationRemovalTransition;

        if (isNegative) {
            auto log_grain_failure = [&](const std::string& field) {
                std::ostringstream oss;
                oss << "FATAL: missing or invalid grain field '" << field << "' in profile '" << jsonPath << "'";
                JTRACE("PROFILE", oss.str());
            };
            auto log_halation_failure = [&](const std::string& field) {
                std::ostringstream oss;
                oss << "FATAL: missing or invalid halation field '" << field << "' in profile '" << jsonPath << "'";
                JTRACE("PROFILE", oss.str());
            };

            if (!root.contains("grain") || !root["grain"].is_object()) {
                log_grain_failure("grain");
                return false;
            }
            const Json& grainNode = root["grain"];
            GrainMetadata grain{};

            auto require_bool = [&](const char* key, bool& dst)->bool {
                auto it = grainNode.find(key);
                if (it != grainNode.end() && it->is_boolean()) {
                    dst = it->get<bool>();
                    return true;
                }
                log_grain_failure(key);
                return false;
            };
            auto require_float = [&](const char* key, float& dst)->bool {
                auto val = parse_optional_float(grainNode.value(key, Json{}));
                if (val && std::isfinite(*val)) {
                    dst = *val;
                    return true;
                }
                log_grain_failure(key);
                return false;
            };
            auto require_float_array3 = [&](const char* key, std::array<float, 3>& dst)->bool {
                if (!grainNode.contains(key) || !grainNode[key].is_array() || grainNode[key].size() < 3) {
                    log_grain_failure(key);
                    return false;
                }
                for (size_t i = 0; i < 3; ++i) {
                    auto val = parse_optional_float(grainNode[key][i]);
                    if (!val || !std::isfinite(*val)) {
                        log_grain_failure(key);
                        return false;
                    }
                    dst[i] = *val;
                }
                return true;
            };
            auto require_float_array2 = [&](const char* key, std::array<float, 2>& dst)->bool {
                if (!grainNode.contains(key) || !grainNode[key].is_array() || grainNode[key].size() < 2) {
                    log_grain_failure(key);
                    return false;
                }
                for (size_t i = 0; i < 2; ++i) {
                    auto val = parse_optional_float(grainNode[key][i]);
                    if (!val || !std::isfinite(*val)) {
                        log_grain_failure(key);
                        return false;
                    }
                    dst[i] = *val;
                }
                return true;
            };
            auto require_int = [&](const char* key, int& dst)->bool {
                if (!grainNode.contains(key)) {
                    log_grain_failure(key);
                    return false;
                }
                const Json& node = grainNode[key];
                if (node.is_number_integer()) {
                    dst = node.get<int>();
                    return true;
                }
                if (node.is_number()) {
                    auto val = parse_optional_float(node);
                    if (val && std::isfinite(*val)) {
                        dst = static_cast<int>(std::lround(*val));
                        return true;
                    }
                }
                log_grain_failure(key);
                return false;
            };

            if (!require_bool("active", grain.active)) return false;
            if (!require_bool("sublayers_active", grain.sublayersActive)) return false;
            if (!require_float("agx_particle_area_um2", grain.agxParticleAreaUm2)) return false;
            if (!require_float_array3("agx_particle_scale", grain.agxParticleScale)) return false;
            if (!require_float_array3("agx_particle_scale_layers", grain.agxParticleScaleLayers)) return false;
            if (!require_float_array3("density_min", grain.densityMin)) return false;
            if (!require_float_array3("uniformity", grain.uniformity)) return false;
            if (!require_float("blur", grain.blur)) return false;
            if (!require_float("blur_dye_clouds_um", grain.blurDyeCloudsUm)) return false;
            if (!require_float_array2("micro_structure", grain.microStructure)) return false;
            if (!require_int("n_sub_layers", grain.nSubLayers)) return false;
            if (auto val = parse_optional_float(grainNode.value("clump_temporal_mix", Json{}))) {
                if (std::isfinite(*val)) {
                    grain.clumpTemporalMix = *val;
                }
            }
            if (auto val = parse_optional_float(grainNode.value("clump_morph_period_sec", Json{}))) {
                if (std::isfinite(*val)) {
                    grain.clumpMorphPeriodSec = *val;
                }
            }

            outProfile.grain = grain;
            outProfile.hasGrain = true;

            if (!root.contains("halation") || !root["halation"].is_object()) {
                log_halation_failure("halation");
                return false;
            }
            const Json& halationNode = root["halation"];
            HalationMetadata halation{};

            auto require_hal_bool = [&](const char* key, bool& dst)->bool {
                auto it = halationNode.find(key);
                if (it != halationNode.end() && it->is_boolean()) {
                    dst = it->get<bool>();
                    return true;
                }
                log_halation_failure(key);
                return false;
            };
            auto require_hal_array3 = [&](const char* key, std::array<float, 3>& dst)->bool {
                if (!halationNode.contains(key) || !halationNode[key].is_array() || halationNode[key].size() < 3) {
                    log_halation_failure(key);
                    return false;
                }
                for (size_t i = 0; i < 3; ++i) {
                    auto val = parse_optional_float(halationNode[key][i]);
                    if (!val || !std::isfinite(*val)) {
                        log_halation_failure(key);
                        return false;
                    }
                    dst[i] = *val;
                }
                return true;
            };

            if (!require_hal_bool("active", halation.active)) return false;
            if (!require_hal_array3("strength", halation.strength)) return false;
            if (!require_hal_array3("size_um", halation.sizeUm)) return false;
            if (!require_hal_array3("scattering_strength", halation.scatteringStrength)) return false;
            if (!require_hal_array3("scattering_size_um", halation.scatteringSizeUm)) return false;

            outProfile.halation = halation;
            outProfile.hasHalation = true;
        }
        else {
            outProfile.hasGrain = false;
            outProfile.hasHalation = false;
        }

        return true;
    }

    bool load_profile_info(const std::string& jsonPath, ProfileInfoSummary& outInfo) {
        Json root;
        if (!parse_json_file(jsonPath, root)) {
            return false;
        }
        if (!root.is_object() || !root.contains("info")) {
            return false;
        }

        const Json& infoNode = root["info"];
        if (!infoNode.is_object()) {
            return false;
        }

        auto readString = [](const Json& node) -> std::string {
            if (node.is_string()) {
                return node.get<std::string>();
            }
            return std::string();
            };

        outInfo = ProfileInfoSummary{};
        if (infoNode.contains("stock")) {
            outInfo.stock = readString(infoNode["stock"]);
        }
        if (infoNode.contains("name")) {
            outInfo.name = readString(infoNode["name"]);
        }
        if (infoNode.contains("type")) {
            outInfo.type = readString(infoNode["type"]);
        }

        return !outInfo.stock.empty();
    }

} // namespace Profiles
