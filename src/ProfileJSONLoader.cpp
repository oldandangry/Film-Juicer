#include "ProfileJSONLoader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <utility>
#include <string_view>

#include "nlohmann/json.hpp"
#include "Logging.h"
#include "Illuminants.h"
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
            // SF_TEMP_BRIDGE_AgxProfileNonFiniteLiteralParser owner=Phase2B remove=Phase3/Phase4:
            // retained only for the old AgxFilmProfile bridge behind the Phase 1A product-render cutoff.
            // Selected spektrafilm payload loading below uses field-gated parsing and rejects Inf.
            // Replace bare non-finite tokens with string sentinels that will be
            // accepted by the JSON parser and decoded later.
            static constexpr std::array<std::pair<std::string_view, std::string_view>, 33> kReplacements{{{"-infinity", "\"__-inf__\""},
                                                                                                          {"+infinity", "\"__inf__\""},
                                                                                                          {"infinity", "\"__inf__\""},
                                                                                                          {"-inf", "\"__-inf__\""},
                                                                                                          {"+inf", "\"__inf__\""},
                                                                                                          {"inf", "\"__inf__\""},
                                                                                                          {"-nan", "\"__nan__\""},
                                                                                                          {"+nan", "\"__nan__\""},
                                                                                                          {"nan", "\"__nan__\""},
                                                                                                          {"-1.#inf", "\"__-inf__\""},
                                                                                                          {"+1.#inf", "\"__inf__\""},
                                                                                                          {"1.#inf", "\"__inf__\""},
                                                                                                          {"-1.#ind", "\"__-inf__\""},
                                                                                                          {"+1.#ind", "\"__inf__\""},
                                                                                                          {"1.#ind", "\"__inf__\""},
                                                                                                          {"-1.#nan", "\"__nan__\""},
                                                                                                          {"+1.#nan", "\"__nan__\""},
                                                                                                          {"1.#nan", "\"__nan__\""},
                                                                                                          {"-1.#qnan", "\"__nan__\""},
                                                                                                          {"+1.#qnan", "\"__nan__\""},
                                                                                                          {"1.#qnan", "\"__nan__\""},
                                                                                                          {"-1.#snan", "\"__nan__\""},
                                                                                                          {"+1.#snan", "\"__nan__\""},
                                                                                                          {"1.#snan", "\"__nan__\""},
                                                                                                          {"-nan(ind)", "\"__nan__\""},
                                                                                                          {"+nan(ind)", "\"__nan__\""},
                                                                                                          {"nan(ind)", "\"__nan__\""},
                                                                                                          {"-nan(qnan)", "\"__nan__\""},
                                                                                                          {"+nan(qnan)", "\"__nan__\""},
                                                                                                          {"nan(qnan)", "\"__nan__\""},
                                                                                                          {"-nan(snan)", "\"__nan__\""},
                                                                                                          {"+nan(snan)", "\"__nan__\""},
                                                                                                          {"nan(snan)", "\"__nan__\""}}};

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
            const auto trace_parse_failure = [&](const char* reason) {
                if (!JTRACE_ENABLED(1)) {
                    return;
                }
                JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': " + (reason ? reason : "unknown exception"));
            };

            errno = 0;
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                if (JTRACE_ENABLED(1)) {
                    const int err = errno;
                    std::string reason;
                    if (err != 0) {
                        reason = std::system_category().message(err);
                    }
                    if (reason.empty()) {
                        reason = "unknown error";
                    }
                    JTRACE("PROFILE", std::string("failed to open profile '") + path + "': " + reason);
                }
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
                                  /*cb*/ nullptr,
                                  /*allow_exceptions*/ true,
                                  /*ignore_comments*/ true);
            } catch (const Json::parse_error& e) {
                trace_parse_failure(e.what());
                return false;
            } catch (const std::exception& e) {
                trace_parse_failure(e.what());
                return false;
            } catch (...) {
                trace_parse_failure(nullptr);
                return false;
            }

            if (out.is_discarded()) {
                if (JTRACE_ENABLED(1)) {
                    JTRACE("PROFILE", std::string("failed to parse profile '") + path + "': parser discarded document");
                }
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
            } else if (node.is_string()) {
                const std::string& str = node.get_ref<const std::string&>();
                const char* begin = str.c_str();
                char* end = nullptr;
                errno = 0;
                const float v = std::strtof(begin, &end);
                if (end != begin && end == begin + str.size() && errno == 0 && std::isfinite(v)) {
                    return v;
                }
            } else if (node.is_boolean()) {
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
            // SF_TEMP_BRIDGE_AgxProfileAllowNanParser owner=Phase2B remove=Phase3/Phase4:
            // allowed only inside parse_agx_film_profile_json; selected spektrafilm loading uses
            // parse_selected_number with schema-lock field policy and rejects Inf.
            if (auto special = decode_nonfinite_sentinel(node)) {
                return special;
            }
            if (node.is_number_float() || node.is_number_integer()) {
                float v = static_cast<float>(node.get<double>());
                if (std::isfinite(v) || std::isnan(v)) {
                    return v;
                }
            } else if (node.is_string()) {
                const std::string& str = node.get_ref<const std::string&>();
                const char* begin = str.c_str();
                char* end = nullptr;
                errno = 0;
                const float v = std::strtof(begin, &end);
                if (end != begin && end == begin + str.size() && errno == 0 &&
                    (std::isfinite(v) || std::isnan(v))) {
                    return v;
                }
            } else if (node.is_boolean()) {
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

        struct FileStamp {
            bool valid = false;
            std::uint64_t sizeBytes = 0;
            std::int64_t writeTimeTicks = 0;
        };

        struct CachedProfileInfoEntry {
            std::string cacheKey;
            FileStamp stamp;
            ProfileInfoSummary info;
        };

        constexpr std::size_t kProfileInfoCacheCapacity = 8;

        FileStamp read_profile_file_stamp(const std::string& jsonPath) {
            std::filesystem::path path(jsonPath);
            std::error_code ec;
            const auto sizeBytes = std::filesystem::file_size(path, ec);
            if (ec) {
                return {};
            }
            const auto writeTime = std::filesystem::last_write_time(path, ec);
            if (ec) {
                return {};
            }

            FileStamp stamp;
            stamp.valid = true;
            stamp.sizeBytes = static_cast<std::uint64_t>(sizeBytes);
            stamp.writeTimeTicks = static_cast<std::int64_t>(writeTime.time_since_epoch().count());
            return stamp;
        }

        bool same_file_stamp(const FileStamp& a, const FileStamp& b) {
            return a.valid &&
                   b.valid &&
                   a.sizeBytes == b.sizeBytes &&
                   a.writeTimeTicks == b.writeTimeTicks;
        }

        std::string normalize_profile_cache_key(const std::string& jsonPath) {
            std::filesystem::path path(jsonPath);
            path.make_preferred();
            return to_lower_ascii(path.lexically_normal().string());
        }

        std::mutex& profile_info_cache_mutex() {
            static std::mutex cacheMutex;
            return cacheMutex;
        }

        std::vector<CachedProfileInfoEntry>& profile_info_cache() {
            static std::vector<CachedProfileInfoEntry> cache;
            return cache;
        }

        bool try_load_cached_profile_info(
            const std::string& cacheKey,
            const FileStamp& stamp,
            ProfileInfoSummary& outInfo) {
            if (cacheKey.empty() || !stamp.valid) {
                return false;
            }

            std::lock_guard<std::mutex> lock(profile_info_cache_mutex());
            auto& cache = profile_info_cache();
            for (std::size_t i = 0; i < cache.size(); ++i) {
                CachedProfileInfoEntry& entry = cache[i];
                if (entry.cacheKey != cacheKey || !same_file_stamp(entry.stamp, stamp)) {
                    continue;
                }

                if (i != 0) {
                    std::swap(cache[0], cache[i]);
                }
                outInfo = cache[0].info;
                return true;
            }

            return false;
        }

        void store_cached_profile_info(
            std::string cacheKey,
            const FileStamp& stamp,
            const ProfileInfoSummary& info) {
            if (cacheKey.empty() || !stamp.valid) {
                return;
            }

            std::lock_guard<std::mutex> lock(profile_info_cache_mutex());
            auto& cache = profile_info_cache();
            for (std::size_t i = 0; i < cache.size(); ++i) {
                if (cache[i].cacheKey == cacheKey) {
                    cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(i));
                    break;
                }
            }

            cache.insert(
                cache.begin(),
                CachedProfileInfoEntry{
                    std::move(cacheKey),
                    stamp,
                    info});
            if (cache.size() > kProfileInfoCacheCapacity) {
                cache.resize(kProfileInfoCacheCapacity);
            }
        }

        bool extract_profile_info_from_root(const Json& root, ProfileInfoSummary& outInfo) {
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

        bool json_wavelengths_match_reference_axis(const Json& wavelengths, std::string_view sourceLabel) {
            if (!wavelengths.is_array()) {
                if (JTRACE_ENABLED(1)) {
                    const std::string_view label = sourceLabel.empty()
                                                       ? std::string_view("profile JSON")
                                                       : sourceLabel;
                    std::ostringstream oss;
                    oss << "JSON wavelengths not an array (" << label << ')';
                    JTRACE("PROFILE", oss.str());
                }
                return false;
            }
            if (wavelengths.size() != Spectral::kNumSamples) {
                if (JTRACE_ENABLED(1)) {
                    const std::string_view label = sourceLabel.empty()
                                                       ? std::string_view("profile JSON")
                                                       : sourceLabel;
                    std::ostringstream oss;
                    oss << "JSON wavelengths mismatch (" << label << "): expected "
                        << Spectral::kNumSamples << " samples, got " << wavelengths.size();
                    JTRACE("PROFILE", oss.str());
                }
                return false;
            }

            for (int i = 0; i < Spectral::kNumSamples; ++i) {
                std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
                if (!wlOpt) {
                    if (JTRACE_ENABLED(1)) {
                        const std::string_view label = sourceLabel.empty()
                                                           ? std::string_view("profile JSON")
                                                           : sourceLabel;
                        std::ostringstream oss;
                        oss << "JSON wavelength missing at index " << i
                            << " (" << label << ')';
                        JTRACE("PROFILE", oss.str());
                    }
                    return false;
                }
                const float expected = Spectral::kLambdaMin + static_cast<float>(i) * Spectral::kDelta;
                if (*wlOpt != expected) {
                    if (JTRACE_ENABLED(1)) {
                        const std::string_view label = sourceLabel.empty()
                                                           ? std::string_view("profile JSON")
                                                           : sourceLabel;
                        std::ostringstream oss;
                        oss << "JSON wavelength mismatch at index " << i
                            << " (" << label << "): expected "
                            << expected << "nm, got " << *wlOpt << "nm";
                        JTRACE("PROFILE", oss.str());
                    }
                    return false;
                }
            }

            return true;
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
                        std::array<float, 3> triplet{0.0f, 0.0f, 0.0f};
                        bool ok = true;
                        for (size_t k = 0; k < 3; ++k) {
                            if (auto val = parse_optional_float(peak[k])) {
                                triplet[k] = *val;
                            } else {
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
            } else if (node.is_object()) {
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
            } else if (auto scalar = parse_optional_float(node)) {
                result[0] = *scalar;
                any = true;
            }

            if (any) {
                dest = result;
            }
            return any;
        }

        enum class SelectedNumericPolicy : unsigned char {
            FiniteOnly,
            NullableNan
        };

        struct SelectedProfileContext {
            std::string path;
            std::string key;
            const char* role = "film";
        };

        std::string selected_profile_diagnostic(
            const SelectedProfileContext& ctx,
            std::string field,
            std::string expected,
            std::string actual) {
            std::ostringstream oss;
            oss << "MalformedRequiredProfileData phase=2"
                << " profile=" << (ctx.key.empty() ? "<unknown>" : ctx.key)
                << " role=" << (ctx.role ? ctx.role : "unknown")
                << " field=" << field
                << " expected=" << expected
                << " actual=" << actual;
            if (!ctx.path.empty()) {
                oss << " path=" << ctx.path;
            }
            return oss.str();
        }

        bool set_error(
            std::string& error,
            const SelectedProfileContext& ctx,
            const std::string& field,
            const std::string& expected,
            const std::string& actual) {
            error = selected_profile_diagnostic(ctx, field, expected, actual);
            return false;
        }

        std::string json_type_name(const Json& node) {
            if (node.is_null()) {
                return "null";
            }
            if (node.is_boolean()) {
                return "boolean";
            }
            if (node.is_number()) {
                return "number";
            }
            if (node.is_string()) {
                return "string";
            }
            if (node.is_array()) {
                return "array";
            }
            if (node.is_object()) {
                return "object";
            }
            return "unknown";
        }

        bool is_nan_string_token(const std::string& value) {
            const std::string lower = to_lower_ascii(value);
            return lower == "nan" || lower == "+nan" || lower == "-nan" || lower == "__nan__";
        }

        bool is_inf_string_token(const std::string& value) {
            const std::string lower = to_lower_ascii(value);
            return lower == "inf" || lower == "+inf" || lower == "-inf" ||
                   lower == "infinity" || lower == "+infinity" || lower == "-infinity" ||
                   lower == "__inf__" || lower == "__-inf__";
        }

        bool quote_bare_nan_literals_for_json_parse(std::string& text) {
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

                std::size_t tokenOffset = 0;
                if ((c == '+' || c == '-') && i + 1 < text.size()) {
                    tokenOffset = 1;
                }
                const std::size_t tokenIndex = i + tokenOffset;
                if (matches_case_insensitive(text, tokenIndex, "nan")) {
                    const std::size_t end = tokenIndex + 3u;
                    const bool hasPrev = i > 0 && is_identifier_char(text[i - 1]);
                    const bool hasNext = end < text.size() && is_identifier_char(text[end]);
                    if (!hasPrev && !hasNext) {
                        text.replace(i, end - i, "\"NaN\"");
                        i += 5u;
                        continue;
                    }
                }
                ++i;
            }

            return true;
        }

        bool parse_selected_json_file(const std::string& path, Json& out, std::string& error) {
            errno = 0;
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                const int err = errno;
                error = std::string("MissingRequiredResource phase=2 field=<file> expected=profile-json actual=open-failed path=") +
                        path + (err != 0 ? (" reason=" + std::system_category().message(err)) : std::string());
                return false;
            }

            std::ostringstream oss;
            oss << file.rdbuf();
            std::string text = oss.str();
            quote_bare_nan_literals_for_json_parse(text);

            try {
                out = Json::parse(text, nullptr, true, true);
            } catch (const Json::exception& ex) {
                error = std::string("MalformedRequiredProfileData phase=2 field=<json> expected=strict-json-or-field-gated-NaN actual=parse-error path=") +
                        path + " reason=" + ex.what();
                return false;
            }
            return out.is_object();
        }

        bool require_array_size(
            const Json& node,
            std::size_t expected,
            const SelectedProfileContext& ctx,
            const std::string& field,
            std::string& error) {
            if (!node.is_array() || node.size() != expected) {
                const std::string actual = node.is_array() ? std::to_string(node.size()) : json_type_name(node);
                return set_error(error, ctx, field, "array[" + std::to_string(expected) + "]", actual);
            }
            return true;
        }

        bool parse_selected_number(
            const Json& node,
            SelectedNumericPolicy policy,
            const SelectedProfileContext& ctx,
            const std::string& field,
            float& out,
            std::string& error) {
            if (node.is_null()) {
                if (policy == SelectedNumericPolicy::NullableNan) {
                    out = std::numeric_limits<float>::quiet_NaN();
                    return true;
                }
                return set_error(error, ctx, field, "finite-number", "null");
            }

            if (node.is_string()) {
                const std::string value = node.get<std::string>();
                if (is_nan_string_token(value) && policy == SelectedNumericPolicy::NullableNan) {
                    out = std::numeric_limits<float>::quiet_NaN();
                    return true;
                }
                if (is_inf_string_token(value)) {
                    return set_error(error, ctx, field, "finite-or-field-gated-NaN", value);
                }
                return set_error(error, ctx, field, "numeric-token", "string");
            }

            if (!node.is_number() || node.is_boolean()) {
                return set_error(error, ctx, field, "numeric-token", json_type_name(node));
            }

            const double raw = node.get<double>();
            if (std::isnan(raw) && policy == SelectedNumericPolicy::NullableNan) {
                out = std::numeric_limits<float>::quiet_NaN();
                return true;
            }
            if (!std::isfinite(raw)) {
                return set_error(error, ctx, field, "finite-number", std::isnan(raw) ? "NaN" : "Inf");
            }

            out = static_cast<float>(raw);
            return true;
        }

        bool parse_selected_vector(
            const Json& node,
            std::size_t expected,
            SelectedNumericPolicy policy,
            const SelectedProfileContext& ctx,
            const std::string& field,
            std::vector<float>& out,
            std::string& error) {
            if (!require_array_size(node, expected, ctx, field, error)) {
                return false;
            }
            out.assign(expected, 0.0f);
            for (std::size_t i = 0; i < expected; ++i) {
                if (!parse_selected_number(
                        node[i],
                        policy,
                        ctx,
                        field + "[" + std::to_string(i) + "]",
                        out[i],
                        error)) {
                    return false;
                }
            }
            return true;
        }

        bool parse_selected_triplet_matrix(
            const Json& node,
            std::size_t expectedRows,
            SelectedNumericPolicy policy,
            const SelectedProfileContext& ctx,
            const std::string& field,
            std::vector<std::array<float, 3>>& out,
            std::string& error) {
            if (!require_array_size(node, expectedRows, ctx, field, error)) {
                return false;
            }
            out.assign(expectedRows, std::array<float, 3>{0.0f, 0.0f, 0.0f});
            for (std::size_t row = 0; row < expectedRows; ++row) {
                const Json& rowNode = node[row];
                if (!require_array_size(rowNode, 3u, ctx, field + "[" + std::to_string(row) + "]", error)) {
                    return false;
                }
                for (std::size_t ch = 0; ch < 3u; ++ch) {
                    if (!parse_selected_number(
                            rowNode[ch],
                            policy,
                            ctx,
                            field + "[" + std::to_string(row) + "][" + std::to_string(ch) + "]",
                            out[row][ch],
                            error)) {
                        return false;
                    }
                }
            }
            return true;
        }

        template <std::size_t N>
        void copy_vector_to_array(const std::vector<float>& src, std::array<float, N>& dst) {
            for (std::size_t i = 0; i < N; ++i) {
                dst[i] = src[i];
            }
        }

        template <std::size_t N>
        void copy_matrix_to_array(
            const std::vector<std::array<float, 3>>& src,
            std::array<std::array<float, 3>, N>& dst) {
            for (std::size_t i = 0; i < N; ++i) {
                dst[i] = src[i];
            }
        }

        bool validate_reference_axis(
            const std::array<float, 81>& wavelengths,
            const SelectedProfileContext& ctx,
            std::string& error) {
            for (std::size_t i = 0; i < wavelengths.size(); ++i) {
                const float expected = Spectral::kLambdaMin + static_cast<float>(i) * Spectral::kDelta;
                if (wavelengths[i] != expected) {
                    return set_error(
                        error,
                        ctx,
                        "data.wavelengths[" + std::to_string(i) + "]",
                        std::to_string(expected),
                        std::to_string(wavelengths[i]));
                }
            }
            return true;
        }

        bool validate_log_exposure_axis(
            const std::vector<float>& logExposure,
            const SelectedProfileContext& ctx,
            std::string& error) {
            if (logExposure.empty()) {
                return set_error(error, ctx, "data.log_exposure", "non-empty finite array", "empty");
            }
            for (std::size_t i = 1; i < logExposure.size(); ++i) {
                if (logExposure[i] < logExposure[i - 1]) {
                    return set_error(
                        error,
                        ctx,
                        "data.log_exposure[" + std::to_string(i) + "]",
                        "non-decreasing",
                        std::to_string(logExposure[i]));
                }
            }
            return true;
        }

        struct SelectedStringMemberSpec {
            const char* key = "";
            const char* fallback = "";
            bool required = false;
        };

        bool string_member_or_default(
            const Json& object,
            const SelectedStringMemberSpec& spec,
            std::string& out,
            bool& defaulted,
            const SelectedProfileContext& ctx,
            std::string& error) {
            const char* key = spec.key;
            const auto it = object.find(key);
            defaulted = it == object.end();
            if (defaulted) {
                if (spec.required) {
                    return set_error(error, ctx, std::string("info.") + key, "string", "missing");
                }
                out = spec.fallback ? spec.fallback : "";
                return true;
            }
            if (!it->is_string()) {
                return set_error(error, ctx, std::string("info.") + key, "string", json_type_name(*it));
            }
            out = it->get<std::string>();
            return true;
        }

        Spektrafilm::ProfileSupport parse_selected_support(const std::string& value) {
            if (value == "film") {
                return Spektrafilm::ProfileSupport::Film;
            }
            if (value == "paper") {
                return Spektrafilm::ProfileSupport::Paper;
            }
            return Spektrafilm::ProfileSupport::Unsupported;
        }

        Spektrafilm::ProfileStage parse_selected_stage(const std::string& value) {
            if (value == "filming") {
                return Spektrafilm::ProfileStage::Filming;
            }
            if (value == "printing") {
                return Spektrafilm::ProfileStage::Printing;
            }
            return Spektrafilm::ProfileStage::Unsupported;
        }

        Spektrafilm::ProfilePolarity parse_selected_polarity(const std::string& value) {
            if (value == "negative") {
                return Spektrafilm::ProfilePolarity::Negative;
            }
            if (value == "positive") {
                return Spektrafilm::ProfilePolarity::Positive;
            }
            return Spektrafilm::ProfilePolarity::Unsupported;
        }

        ProfileUse parse_profile_use(const std::string& value) {
            if (value == "still") {
                return ProfileUse::Still;
            }
            if (value == "cine") {
                return ProfileUse::Cine;
            }
            return ProfileUse::Unsupported;
        }

        ProfileAntihalation parse_profile_antihalation(const std::string& value) {
            if (value == "strong") {
                return ProfileAntihalation::Strong;
            }
            if (value == "weak") {
                return ProfileAntihalation::Weak;
            }
            if (value == "no") {
                return ProfileAntihalation::No;
            }
            return ProfileAntihalation::Unsupported;
        }

        ProfileChannelModel parse_profile_channel_model(const std::string& value) {
            if (value == "color") {
                return ProfileChannelModel::Color;
            }
            if (value == "bw") {
                return ProfileChannelModel::Bw;
            }
            return ProfileChannelModel::Unsupported;
        }

        bool string_has_only_digits(const std::string& value, std::size_t first) {
            if (first >= value.size()) {
                return false;
            }
            for (std::size_t i = first; i < value.size(); ++i) {
                if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
                    return false;
                }
            }
            return true;
        }

        bool is_supported_profile_illuminant(const std::string& value) {
            const std::string normalized = IlluminantKeys::normalize(value);
            if (IlluminantKeys::matches_any(
                    normalized,
                    {"D50", "D55", "D65", "T", "K75P", "KINOTON75P", "TH-KG3", "THKG3", "TH-KG3-L", "THKG3L"})) {
                return true;
            }
            if (normalized.size() >= 2u && normalized[0] == 'D' && string_has_only_digits(normalized, 1u)) {
                return true;
            }
            if (normalized.size() > 2u && normalized[0] == 'B' && normalized[1] == 'B') {
                const char* start = normalized.c_str() + 2;
                char* endPtr = nullptr;
                const double temperature = std::strtod(start, &endPtr);
                return endPtr && *endPtr == '\0' && std::isfinite(temperature) && temperature > 0.0;
            }
            return false;
        }

        bool parse_profile_info(
            const Json& root,
            ProfileRole role,
            SpektrafilmProfileInfo& out,
            SelectedProfileContext& ctx,
            std::string& error) {
            const Json empty = Json::object();
            const Json& info = root.contains("info") && root["info"].is_object() ? root["info"] : empty;
            if (!root.contains("info") || !root["info"].is_object()) {
                return set_error(error, ctx, "info", "object", root.contains("info") ? json_type_name(root["info"]) : "missing");
            }

            std::string raw;
            bool stockDefaulted = false;
            if (!string_member_or_default(info, {"stock", "", true}, out.stock, stockDefaulted, ctx, error)) {
                return false;
            }
            ctx.key = out.stock;
            bool labelDefaulted = false;
            if (!string_member_or_default(info, {"name", out.stock.c_str()}, out.name, labelDefaulted, ctx, error)) {
                return false;
            }

            if (!string_member_or_default(info, {"support", "film"}, raw, out.supportDefaulted, ctx, error)) {
                return false;
            }
            out.support = parse_selected_support(raw);
            if (out.support == Spektrafilm::ProfileSupport::Unsupported) {
                return set_error(error, ctx, "info.support", "film|paper", raw);
            }

            if (!string_member_or_default(info, {"stage", "filming"}, raw, out.stageDefaulted, ctx, error)) {
                return false;
            }
            out.stage = parse_selected_stage(raw);
            if (out.stage == Spektrafilm::ProfileStage::Unsupported) {
                return set_error(error, ctx, "info.stage", "filming|printing", raw);
            }

            if (!string_member_or_default(info, {"type", "negative"}, raw, out.typeDefaulted, ctx, error)) {
                return false;
            }
            out.type = parse_selected_polarity(raw);
            if (out.type == Spektrafilm::ProfilePolarity::Unsupported) {
                return set_error(error, ctx, "info.type", "negative|positive", raw);
            }

            if (role == ProfileRole::Film &&
                !(out.support == Spektrafilm::ProfileSupport::Film && out.stage == Spektrafilm::ProfileStage::Filming)) {
                return set_error(error, ctx, "info", "support=film stage=filming", "role-mismatch");
            }
            if (role == ProfileRole::Print && out.stage != Spektrafilm::ProfileStage::Printing) {
                return set_error(error, ctx, "info.stage", "printing", "role-mismatch");
            }

            if (!string_member_or_default(info, {"use", "still"}, raw, out.useDefaulted, ctx, error)) {
                return false;
            }
            out.use = parse_profile_use(raw);
            if (out.use == ProfileUse::Unsupported) {
                return set_error(error, ctx, "info.use", "still|cine", raw);
            }

            if (!string_member_or_default(info, {"antihalation", "weak"}, raw, out.antihalationDefaulted, ctx, error)) {
                return false;
            }
            out.antihalation = parse_profile_antihalation(raw);
            if (out.antihalation == ProfileAntihalation::Unsupported) {
                return set_error(error, ctx, "info.antihalation", "strong|weak|no", raw);
            }

            if (!string_member_or_default(info, {"channel_model", "color"}, raw, out.channelModelDefaulted, ctx, error)) {
                return false;
            }
            out.channelModel = parse_profile_channel_model(raw);
            if (out.channelModel == ProfileChannelModel::Unsupported) {
                return set_error(error, ctx, "info.channel_model", "color|bw", raw);
            }

            if (!string_member_or_default(info, {"reference_illuminant", "D55"}, out.referenceIlluminant.value, out.referenceIlluminantDefaulted, ctx, error)) {
                return false;
            }
            if (!is_supported_profile_illuminant(out.referenceIlluminant.value)) {
                return set_error(error, ctx, "info.reference_illuminant", "supported illuminant key including BB<temperature>", out.referenceIlluminant.value);
            }
            if (!string_member_or_default(info, {"viewing_illuminant", "D50"}, out.viewingIlluminant.value, out.viewingIlluminantDefaulted, ctx, error)) {
                return false;
            }
            if (!is_supported_profile_illuminant(out.viewingIlluminant.value)) {
                return set_error(error, ctx, "info.viewing_illuminant", "supported illuminant key including BB<temperature>", out.viewingIlluminant.value);
            }

            return true;
        }

        void derive_linear_sensitivity(SpektrafilmProfileSamples& samples) {
            for (std::size_t row = 0; row < samples.logSensitivity.size(); ++row) {
                for (std::size_t ch = 0; ch < 3u; ++ch) {
                    const float authored = samples.logSensitivity[row][ch];
                    const float linear = std::pow(10.0f, authored);
                    samples.linearSensitivity[row][ch] = std::isfinite(linear) ? linear : 0.0f;
                }
            }
        }

        bool parse_profile_samples(
            const Json& root,
            SpektrafilmProfileSamples& out,
            const SelectedProfileContext& ctx,
            std::string& error) {
            if (!root.contains("data") || !root["data"].is_object()) {
                return set_error(error, ctx, "data", "object", root.contains("data") ? json_type_name(root["data"]) : "missing");
            }
            const Json& data = root["data"];

            std::vector<float> wavelengths;
            if (!parse_selected_vector(
                    data.value("wavelengths", Json{}),
                    Spectral::kNumSamples,
                    SelectedNumericPolicy::FiniteOnly,
                    ctx,
                    "data.wavelengths",
                    wavelengths,
                    error)) {
                return false;
            }
            copy_vector_to_array(wavelengths, out.wavelengths);
            if (!validate_reference_axis(out.wavelengths, ctx, error)) {
                return false;
            }

            std::vector<std::array<float, 3>> matrix;
            if (!parse_selected_triplet_matrix(
                    data.value("log_sensitivity", Json{}),
                    Spectral::kNumSamples,
                    SelectedNumericPolicy::NullableNan,
                    ctx,
                    "data.log_sensitivity",
                    matrix,
                    error)) {
                return false;
            }
            copy_matrix_to_array(matrix, out.logSensitivity);
            derive_linear_sensitivity(out);

            if (!parse_selected_triplet_matrix(
                    data.value("channel_density", Json{}),
                    Spectral::kNumSamples,
                    SelectedNumericPolicy::NullableNan,
                    ctx,
                    "data.channel_density",
                    matrix,
                    error)) {
                return false;
            }
            copy_matrix_to_array(matrix, out.channelDensity);

            std::vector<float> baseDensity;
            if (!parse_selected_vector(
                    data.value("base_density", Json{}),
                    Spectral::kNumSamples,
                    SelectedNumericPolicy::NullableNan,
                    ctx,
                    "data.base_density",
                    baseDensity,
                    error)) {
                return false;
            }
            copy_vector_to_array(baseDensity, out.baseDensity);

            const Json& logExposureNode = data.value("log_exposure", Json{});
            if (!logExposureNode.is_array()) {
                return set_error(error, ctx, "data.log_exposure", "array[N]", json_type_name(logExposureNode));
            }
            if (!parse_selected_vector(
                    logExposureNode,
                    logExposureNode.size(),
                    SelectedNumericPolicy::FiniteOnly,
                    ctx,
                    "data.log_exposure",
                    out.logExposure,
                    error)) {
                return false;
            }
            if (!validate_log_exposure_axis(out.logExposure, ctx, error)) {
                return false;
            }

            if (!parse_selected_triplet_matrix(
                    data.value("density_curves", Json{}),
                    out.logExposure.size(),
                    SelectedNumericPolicy::NullableNan,
                    ctx,
                    "data.density_curves",
                    out.densityCurves,
                    error)) {
                return false;
            }

            const auto windowIt = data.find("hanatos2025_adaptation_window_params");
            out.hasHanatos2025AdaptationWindowParams = false;
            if (windowIt != data.end() && windowIt->is_array() && !windowIt->empty()) {
                std::vector<float> window;
                if (!parse_selected_vector(
                        *windowIt,
                        4u,
                        SelectedNumericPolicy::FiniteOnly,
                        ctx,
                        "data.hanatos2025_adaptation_window_params",
                        window,
                        error)) {
                    return false;
                }
                copy_vector_to_array(window, out.hanatos2025AdaptationWindowParams);
                out.hasHanatos2025AdaptationWindowParams = true;
            } else if (windowIt != data.end() && !(windowIt->is_array() && windowIt->empty())) {
                return set_error(error, ctx, "data.hanatos2025_adaptation_window_params", "array[4] or []", json_type_name(*windowIt));
            }

            const auto surfaceIt = data.find("hanatos2025_adaptation_surface_params");
            out.hasHanatos2025AdaptationSurfaceParams = false;
            if (surfaceIt != data.end() && surfaceIt->is_array() && !surfaceIt->empty()) {
                std::vector<std::array<float, 3>> unusedTriplets;
                if (!require_array_size(*surfaceIt, 3u, ctx, "data.hanatos2025_adaptation_surface_params", error)) {
                    return false;
                }
                for (std::size_t ch = 0; ch < 3u; ++ch) {
                    const std::string rowField = "data.hanatos2025_adaptation_surface_params[" + std::to_string(ch) + "]";
                    if (!require_array_size((*surfaceIt)[ch], 15u, ctx, rowField, error)) {
                        return false;
                    }
                    for (std::size_t k = 0; k < 15u; ++k) {
                        if (!parse_selected_number(
                                (*surfaceIt)[ch][k],
                                SelectedNumericPolicy::FiniteOnly,
                                ctx,
                                rowField + "[" + std::to_string(k) + "]",
                                out.hanatos2025AdaptationSurfaceParams[ch][k],
                                error)) {
                            return false;
                        }
                    }
                }
                out.hasHanatos2025AdaptationSurfaceParams = true;
            } else if (surfaceIt != data.end() && !(surfaceIt->is_array() && surfaceIt->empty())) {
                return set_error(error, ctx, "data.hanatos2025_adaptation_surface_params", "array[3][15] or []", json_type_name(*surfaceIt));
            }

            return true;
        }

    } // namespace

    static bool parse_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile) {
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
                } else if (auto val = parse_optional_float(mid)) {
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
                } else if (auto val = parse_optional_float(mid)) {
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
                std::array<float, 3> gamma{{1.0f, 1.0f, 1.0f}};
                bool any = false;
                if (auto scalar = parse_optional_float(gf)) {
                    const float v = *scalar;
                    if (std::isfinite(v) && v > 0.0f) {
                        gamma.fill(v);
                        any = true;
                    }
                } else if (gf.is_array()) {
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
        const std::string_view wavelengthLabel = jsonPath.empty()
                                                     ? std::string_view("profile JSON")
                                                     : std::string_view(jsonPath);
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
                // SF_TEMP_BRIDGE_AgxProfileMalformedRowPlaceholder owner=Phase2B remove=Phase3/Phase4:
                // old AgxFilmProfile bridge behavior retained only behind the Phase 1A product-render cutoff.
                JTRACE("PROFILE", "dye_density row missing or not array; inserting NaNs");
                outProfile.dyeC.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.dyeM.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.dyeY.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.baseMin.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                outProfile.baseMid.emplace_back(wl, std::numeric_limits<float>::quiet_NaN());
                continue;
            }

            auto get = [&](size_t idx) -> std::optional<float> {
                if (idx >= row.size())
                    return std::nullopt;
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

        auto log_profile_field_failure = [&](const char* section, const std::string& field) {
            if (JTRACE_ENABLED(1)) {
                std::ostringstream oss;
                oss << "FATAL: missing or invalid " << (section ? section : "profile")
                    << " field '" << field << "' in profile '" << jsonPath << "'";
                JTRACE("PROFILE", oss.str());
            }
        };
        auto log_glare_failure = [&](const std::string& field) {
            log_profile_field_failure("glare", field);
        };

        if (!root.contains("glare") || !root["glare"].is_object()) {
            log_glare_failure("glare");
            return false;
        }

        const Json& glareNode = root["glare"];
        ProfileGlare glare{};
        auto require_bool = [&](const char* key, bool& dst) -> bool {
            auto it = glareNode.find(key);
            if (it != glareNode.end() && it->is_boolean()) {
                dst = it->get<bool>();
                return true;
            }
            log_glare_failure(key);
            return false;
        };
        auto require_float = [&](const char* key, float& dst) -> bool {
            auto val = parse_optional_float(glareNode.value(key, Json{}));
            if (val && std::isfinite(*val)) {
                dst = *val;
                return true;
            }
            log_glare_failure(key);
            return false;
        };

        if (!require_bool("active", glare.active))
            return false;
        if (!require_float("percent", glare.percent))
            return false;
        if (!require_float("roughness", glare.roughness))
            return false;
        if (!require_float("blur", glare.blur))
            return false;

        if (isPaper) {
            if (!require_float("compensation_removal_factor", glare.compensationRemovalFactor))
                return false;
            if (!require_float("compensation_removal_density", glare.compensationRemovalDensity))
                return false;
            if (!require_float("compensation_removal_transition", glare.compensationRemovalTransition))
                return false;
        } else {
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_factor", Json{}))) {
                if (std::isfinite(*val))
                    glare.compensationRemovalFactor = *val;
            }
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_density", Json{}))) {
                if (std::isfinite(*val))
                    glare.compensationRemovalDensity = *val;
            }
            if (auto val = parse_optional_float(glareNode.value("compensation_removal_transition", Json{}))) {
                if (std::isfinite(*val))
                    glare.compensationRemovalTransition = *val;
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
                log_profile_field_failure("grain", field);
            };
            auto log_halation_failure = [&](const std::string& field) {
                log_profile_field_failure("halation", field);
            };

            if (!root.contains("grain") || !root["grain"].is_object()) {
                log_grain_failure("grain");
                return false;
            }
            const Json& grainNode = root["grain"];
            GrainMetadata grain{};

            auto require_bool = [&](const char* key, bool& dst) -> bool {
                auto it = grainNode.find(key);
                if (it != grainNode.end() && it->is_boolean()) {
                    dst = it->get<bool>();
                    return true;
                }
                log_grain_failure(key);
                return false;
            };
            auto require_float = [&](const char* key, float& dst) -> bool {
                auto val = parse_optional_float(grainNode.value(key, Json{}));
                if (val && std::isfinite(*val)) {
                    dst = *val;
                    return true;
                }
                log_grain_failure(key);
                return false;
            };
            auto require_float_array3 = [&](const char* key, std::array<float, 3>& dst) -> bool {
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
            auto require_float_array2 = [&](const char* key, std::array<float, 2>& dst) -> bool {
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
            auto require_int = [&](const char* key, int& dst) -> bool {
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

            if (!require_bool("active", grain.active))
                return false;
            if (!require_bool("sublayers_active", grain.sublayersActive))
                return false;
            if (!require_float("agx_particle_area_um2", grain.agxParticleAreaUm2))
                return false;
            if (!require_float_array3("agx_particle_scale", grain.agxParticleScale))
                return false;
            if (!require_float_array3("agx_particle_scale_layers", grain.agxParticleScaleLayers))
                return false;
            if (!require_float_array3("density_min", grain.densityMin))
                return false;
            if (!require_float_array3("uniformity", grain.uniformity))
                return false;
            if (!require_float("blur", grain.blur))
                return false;
            if (!require_float("blur_dye_clouds_um", grain.blurDyeCloudsUm))
                return false;
            if (!require_float_array2("micro_structure", grain.microStructure))
                return false;
            if (!require_int("n_sub_layers", grain.nSubLayers))
                return false;
            if (auto val = parse_optional_float(grainNode.value("size_mix_weight_mid", Json{}))) {
                if (std::isfinite(*val)) {
                    grain.sizeMixWeightMid = *val;
                }
            }
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

            auto require_hal_bool = [&](const char* key, bool& dst) -> bool {
                auto it = halationNode.find(key);
                if (it != halationNode.end() && it->is_boolean()) {
                    dst = it->get<bool>();
                    return true;
                }
                log_halation_failure(key);
                return false;
            };
            auto require_hal_array3 = [&](const char* key, std::array<float, 3>& dst) -> bool {
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

            if (!require_hal_bool("active", halation.active))
                return false;
            if (!require_hal_array3("strength", halation.strength))
                return false;
            if (!require_hal_array3("size_um", halation.sizeUm))
                return false;
            if (!require_hal_array3("scattering_strength", halation.scatteringStrength))
                return false;
            if (!require_hal_array3("scattering_size_um", halation.scatteringSizeUm))
                return false;

            outProfile.halation = halation;
            outProfile.hasHalation = true;
        } else {
            outProfile.hasGrain = false;
            outProfile.hasHalation = false;
        }

        return true;
    }

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile) {
        return parse_agx_film_profile_json(jsonPath, outProfile);
    }

    bool load_profile_info(const std::string& jsonPath, ProfileInfoSummary& outInfo) {
        outInfo = ProfileInfoSummary{};

        const FileStamp stamp = read_profile_file_stamp(jsonPath);
        const std::string cacheKey = normalize_profile_cache_key(jsonPath);
        if (try_load_cached_profile_info(cacheKey, stamp, outInfo)) {
            return true;
        }

        Json root;
        if (!parse_json_file(jsonPath, root)) {
            return false;
        }
        if (!extract_profile_info_from_root(root, outInfo)) {
            return false;
        }

        store_cached_profile_info(cacheKey, stamp, outInfo);
        return true;
    }

    namespace {
        template <typename ProfileT, typename DataT>
        bool load_validated_profile_json_impl(
            const std::string& jsonPath,
            ProfileRole role,
            ProfileT& outProfile,
            std::string* outDiagnostic) {
            outProfile = ProfileT{};
            Json root;
            std::string error;
            if (!parse_selected_json_file(jsonPath, root, error)) {
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }

            SelectedProfileContext ctx;
            ctx.path = jsonPath;
            ctx.role = role == ProfileRole::Print ? "print" : "film";

            SpektrafilmProfileInfo info;
            if (!parse_profile_info(root, role, info, ctx, error)) {
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }

            DataT data;
            if (!parse_profile_samples(root, data, ctx, error)) {
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }

            outProfile.info = std::move(info);
            outProfile.data = std::move(data);
            outProfile.digest = build_profile_digest(outProfile.info, role);
            outProfile.digest.hanatosWindowAuthored = outProfile.data.hasHanatos2025AdaptationWindowParams;
            outProfile.digest.hanatosSurfaceAuthored = outProfile.data.hasHanatos2025AdaptationSurfaceParams;
            outProfile.sourcePath = jsonPath;
            outProfile.assetVersionToken = build_profile_asset_version_token(outProfile.info, outProfile.data);
            if (outDiagnostic) {
                outDiagnostic->clear();
            }
            return outProfile.assetVersionToken != 0;
        }
    } // namespace

    bool load_validated_film_profile_json(
        const std::string& jsonPath,
        ValidatedFilmProfile& outProfile,
        std::string* outDiagnostic) {
        return load_validated_profile_json_impl<ValidatedFilmProfile, SpektrafilmFilmData>(
            jsonPath,
            ProfileRole::Film,
            outProfile,
            outDiagnostic);
    }

    bool load_validated_print_profile_json(
        const std::string& jsonPath,
        ValidatedPrintProfile& outProfile,
        std::string* outDiagnostic) {
        return load_validated_profile_json_impl<ValidatedPrintProfile, SpektrafilmPrintData>(
            jsonPath,
            ProfileRole::Print,
            outProfile,
            outDiagnostic);
    }

} // namespace Profiles
