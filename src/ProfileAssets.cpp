#include "ProfileAssets.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hash.h"
#include "Illuminants.h"
#include "Logging.h"
#include "SpectralData.h"
#include "nlohmann/json.hpp"

namespace Profiles {
    namespace {

        using Json = nlohmann::json;

        enum class SelectedProfileKind : unsigned char {
            Film,
            Print
        };

        bool is_identifier_char(char c) {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '#';
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
        std::string to_lower_ascii(std::string value) {
            for (char& c : value) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            return value;
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
            const std::string& field,
            const std::string& expected,
            const std::string& actual) {
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

        bool parse_selected_density_curves_layers_required(
            const Json& node,
            std::size_t expectedRows,
            const SelectedProfileContext& ctx,
            std::array<std::array<std::vector<float>, 3>, 3>& out,
            std::string& error) {
            if (!require_array_size(node, expectedRows, ctx, "data.density_curves_layers", error)) {
                return false;
            }
            for (std::size_t layer = 0; layer < 3u; ++layer) {
                for (std::size_t ch = 0; ch < 3u; ++ch) {
                    out[layer][ch].assign(expectedRows, std::numeric_limits<float>::quiet_NaN());
                }
            }
            for (std::size_t row = 0; row < expectedRows; ++row) {
                const Json& rowNode = node[row];
                const std::string rowField =
                    "data.density_curves_layers[" + std::to_string(row) + "]";
                if (!require_array_size(rowNode, 3u, ctx, rowField, error)) {
                    return false;
                }
                for (std::size_t layer = 0; layer < 3u; ++layer) {
                    const Json& layerNode = rowNode[layer];
                    const std::string layerField =
                        rowField + "[" + std::to_string(layer) + "]";
                    if (!require_array_size(layerNode, 3u, ctx, layerField, error)) {
                        return false;
                    }
                    for (std::size_t ch = 0; ch < 3u; ++ch) {
                        if (!parse_selected_number(
                                layerNode[ch],
                                SelectedNumericPolicy::NullableNan,
                                ctx,
                                layerField + "[" + std::to_string(ch) + "]",
                                out[layer][ch][row],
                                error)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        void record_selected_density_curves_layers(
            const Json& data,
            SpektrafilmProfileSamples& out,
            const SelectedProfileContext& ctx) {
            out.hasDensityCurvesLayers = false;
            out.densityCurvesLayersMalformed = false;
            out.densityCurvesLayersDiagnostic.clear();
            for (auto& layer : out.densityCurvesLayers) {
                for (auto& channel : layer) {
                    channel.clear();
                }
            }

            const auto it = data.find("density_curves_layers");
            if (it == data.end() || (it->is_array() && it->empty())) {
                return;
            }

            std::array<std::array<std::vector<float>, 3>, 3> layers{};
            std::string layerError;
            if (!parse_selected_density_curves_layers_required(
                    *it,
                    out.logExposure.size(),
                    ctx,
                    layers,
                    layerError)) {
                out.densityCurvesLayersMalformed = true;
                out.densityCurvesLayersDiagnostic =
                    layerError.empty()
                        ? "MalformedRequiredProfileData phase=9B field=data.density_curves_layers"
                        : layerError;
                return;
            }

            out.densityCurvesLayers = std::move(layers);
            out.hasDensityCurvesLayers = true;
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
            SelectedProfileKind role,
            SpektrafilmProfileInfo& out,
            SelectedProfileContext& ctx,
            std::string& error) {
            const Json empty = Json::object();
            const Json& info = root.contains("info") && root["info"].is_object() ? root["info"] : empty;
            if (!root.contains("info") || !root["info"].is_object()) {
                return set_error(error, ctx, "info", "object", root.contains("info") ? json_type_name(root["info"]) : "missing");
            }

            std::string raw;
            bool ignoredDefaulted = false;
            bool stockDefaulted = false;
            if (!string_member_or_default(info, {"stock", "", true}, out.stock, stockDefaulted, ctx, error)) {
                return false;
            }
            ctx.key = out.stock;
            bool labelDefaulted = false;
            if (!string_member_or_default(info, {"name", out.stock.c_str()}, out.name, labelDefaulted, ctx, error)) {
                return false;
            }

            if (!string_member_or_default(info, {"support", "film"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.support = parse_selected_support(raw);
            if (out.support == Spektrafilm::ProfileSupport::Unsupported) {
                return set_error(error, ctx, "info.support", "film|paper", raw);
            }

            if (!string_member_or_default(info, {"stage", "filming"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.stage = parse_selected_stage(raw);
            if (out.stage == Spektrafilm::ProfileStage::Unsupported) {
                return set_error(error, ctx, "info.stage", "filming|printing", raw);
            }

            if (!string_member_or_default(info, {"type", "negative"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.type = parse_selected_polarity(raw);
            if (out.type == Spektrafilm::ProfilePolarity::Unsupported) {
                return set_error(error, ctx, "info.type", "negative|positive", raw);
            }

            if (role == SelectedProfileKind::Film &&
                !(out.support == Spektrafilm::ProfileSupport::Film && out.stage == Spektrafilm::ProfileStage::Filming)) {
                return set_error(error, ctx, "info", "support=film stage=filming", "role-mismatch");
            }
            if (role == SelectedProfileKind::Print && out.stage != Spektrafilm::ProfileStage::Printing) {
                return set_error(error, ctx, "info.stage", "printing", "role-mismatch");
            }

            if (!string_member_or_default(info, {"use", "still"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.use = parse_profile_use(raw);
            if (out.use == ProfileUse::Unsupported) {
                return set_error(error, ctx, "info.use", "still|cine", raw);
            }

            if (!string_member_or_default(info, {"antihalation", "weak"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.antihalation = parse_profile_antihalation(raw);
            if (out.antihalation == ProfileAntihalation::Unsupported) {
                return set_error(error, ctx, "info.antihalation", "strong|weak|no", raw);
            }

            if (!string_member_or_default(info, {"channel_model", "color"}, raw, ignoredDefaulted, ctx, error)) {
                return false;
            }
            out.channelModel = parse_profile_channel_model(raw);
            if (out.channelModel == ProfileChannelModel::Unsupported) {
                return set_error(error, ctx, "info.channel_model", "color|bw", raw);
            }

            if (!string_member_or_default(info, {"reference_illuminant", "D55"}, out.referenceIlluminant.value, ignoredDefaulted, ctx, error)) {
                return false;
            }
            if (!is_supported_profile_illuminant(out.referenceIlluminant.value)) {
                return set_error(error, ctx, "info.reference_illuminant", "supported illuminant key including BB<temperature>", out.referenceIlluminant.value);
            }
            if (!string_member_or_default(info, {"viewing_illuminant", "D50"}, out.viewingIlluminant.value, ignoredDefaulted, ctx, error)) {
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
            record_selected_density_curves_layers(data, out, ctx);

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

        const Spektrafilm::ProfileCatalogEntry* find_profile_entry(
            const std::vector<Spektrafilm::ProfileCatalogEntry>& entries,
            const std::string& key) {
            const auto it = std::find_if(
                entries.begin(),
                entries.end(),
                [&key](const Spektrafilm::ProfileCatalogEntry& entry) {
                    return entry.key == key;
                });
            return it == entries.end() ? nullptr : &*it;
        }

        void set_missing_profile_diagnostic(
            std::string* outDiagnostic,
            const char* role,
            const std::string& key) {
            if (outDiagnostic) {
                *outDiagnostic = std::string("MissingRequiredResource phase=2 field=") + role +
                                 "_profile key=" + key;
            }
        }

        void hash_u64_update(std::uint64_t& hash, std::uint64_t value) {
            Hash::hash_bytes_update(hash, &value, sizeof(value));
        }

        void hash_string_update(std::uint64_t& hash, const char* value) {
            Hash::hash_bytes_update(hash, value, std::strlen(value));
        }

        void hash_string_update(std::uint64_t& hash, const std::string& value) {
            Hash::hash_bytes_update(hash, value.data(), value.size());
        }

        void hash_float_values_allowing_nan(std::uint64_t& hash, const float* values, std::size_t count) {
            const Hash::FloatSpanHash pair = Hash::hash_float_span_with_nan_mask(values, count);
            hash_u64_update(hash, pair.valueHash);
            hash_u64_update(hash, pair.nanMaskHash);
        }

        void hash_consumed_profile_samples(std::uint64_t& hash, const SpektrafilmProfileSamples& data) {
            hash_string_update(hash, "data.wavelengths");
            hash_float_values_allowing_nan(hash, data.wavelengths.data(), data.wavelengths.size());
            hash_string_update(hash, "data.log_sensitivity");
            hash_float_values_allowing_nan(hash, &data.logSensitivity[0][0], data.logSensitivity.size() * 3u);
            hash_string_update(hash, "data.channel_density");
            hash_float_values_allowing_nan(hash, &data.channelDensity[0][0], data.channelDensity.size() * 3u);
            hash_string_update(hash, "data.base_density");
            hash_float_values_allowing_nan(hash, data.baseDensity.data(), data.baseDensity.size());
            hash_string_update(hash, "data.log_exposure");
            hash_float_values_allowing_nan(hash, data.logExposure.data(), data.logExposure.size());
            hash_string_update(hash, "data.density_curves");
            if (!data.densityCurves.empty()) {
                hash_float_values_allowing_nan(hash, &data.densityCurves[0][0], data.densityCurves.size() * 3u);
            }
            hash_string_update(hash, "data.hanatos2025_adaptation_window_params");
            hash_u64_update(hash, data.hasHanatos2025AdaptationWindowParams ? 1u : 0u);
            if (data.hasHanatos2025AdaptationWindowParams) {
                hash_float_values_allowing_nan(
                    hash,
                    data.hanatos2025AdaptationWindowParams.data(),
                    data.hanatos2025AdaptationWindowParams.size());
            }
            hash_string_update(hash, "data.hanatos2025_adaptation_surface_params");
            hash_u64_update(hash, data.hasHanatos2025AdaptationSurfaceParams ? 1u : 0u);
            if (data.hasHanatos2025AdaptationSurfaceParams) {
                hash_float_values_allowing_nan(hash, &data.hanatos2025AdaptationSurfaceParams[0][0], 45u);
            }
        }

        bool validate_halation_profile_digest(
            const ProfileDigest& digest,
            const SelectedProfileContext& ctx,
            std::string& error) {
            const auto validate = [&](const std::array<float, 3>& values,
                                      const char* field) {
                for (std::size_t channel = 0; channel < values.size(); ++channel) {
                    const float value = values[channel];
                    if (!std::isfinite(value) || value < 0.0f) {
                        return set_error(
                            error,
                            ctx,
                            std::string(field) + "[" + std::to_string(channel) + "]",
                            "finite-nonnegative-Float32",
                            std::to_string(value));
                    }
                }
                return true;
            };
            return validate(
                       digest.halationFirstSigmaUm,
                       "digest.halation_first_sigma_um") &&
                   validate(
                       digest.halationPrimaryAmount,
                       "digest.halation_primary_amount");
        }

        bool build_film_profile_digest(
            const SpektrafilmProfileInfo& info,
            const SelectedProfileContext& ctx,
            ProfileDigest& out,
            std::string& error) {
            ProfileDigest digest{};
            const bool positive = info.type == Spektrafilm::ProfilePolarity::Positive;
            if (positive) {
                digest.gammaSamelayerRgb = {{0.12f, 0.08f, 0.06f}};
                digest.gammaInterlayerRToGb = {{0.12f, 0.06f}};
                digest.gammaInterlayerGToRb = {{0.08f, 0.06f}};
                digest.gammaInterlayerBToRg = {{0.06f, 0.06f}};
            } else {
                digest.gammaSamelayerRgb = {{0.336f, 0.319f, 0.273f}};
                digest.gammaInterlayerRToGb = {{0.353f, 0.302f}};
                digest.gammaInterlayerGToRb = {{0.154f, 0.353f}};
                digest.gammaInterlayerBToRg = {{0.168f, 0.226f}};
            }

            if (info.stock == "fujifilm_velvia_100") {
                digest.gammaSamelayerRgb = {{0.108f, 0.072f, 0.054f}};
                digest.gammaInterlayerRToGb = {{0.108f, 0.054f}};
                digest.gammaInterlayerGToRb = {{0.072f, 0.054f}};
                digest.gammaInterlayerBToRg = {{0.054f, 0.054f}};
            } else if (info.stock == "fujifilm_provia_100f") {
                digest.gammaSamelayerRgb = {{0.156f, 0.104f, 0.078f}};
                digest.gammaInterlayerRToGb = {{0.156f, 0.078f}};
                digest.gammaInterlayerGToRb = {{0.104f, 0.078f}};
                digest.gammaInterlayerBToRg = {{0.078f, 0.078f}};
            }

            digest.halationFirstSigmaUm = info.use == ProfileUse::Cine
                                              ? std::array<float, 3>{{50.0f, 50.0f, 50.0f}}
                                              : std::array<float, 3>{{65.0f, 65.0f, 65.0f}};
            switch (info.antihalation) {
                case ProfileAntihalation::Strong:
                    digest.halationPrimaryAmount = {{0.015f, 0.005f, 0.0f}};
                    break;
                case ProfileAntihalation::Weak:
                    digest.halationPrimaryAmount = {{0.08f, 0.02f, 0.0f}};
                    break;
                case ProfileAntihalation::No:
                    digest.halationPrimaryAmount = {{0.30f, 0.10f, 0.015f}};
                    break;
                default:
                    return set_error(
                        error,
                        ctx,
                        "info.antihalation",
                        "strong|weak|no",
                        "unsupported-enum");
            }
            if (!validate_halation_profile_digest(digest, ctx, error)) {
                return false;
            }
            out = digest;
            return true;
        }

        std::uint64_t build_profile_asset_version_token(
            const SpektrafilmProfileInfo& info,
            const SpektrafilmProfileSamples& data) {
            std::uint64_t hash = Hash::kFnvOffset;
            hash_string_update(hash, info.stock);
            hash_u64_update(hash, static_cast<std::uint64_t>(info.support));
            hash_u64_update(hash, static_cast<std::uint64_t>(info.stage));
            hash_u64_update(hash, static_cast<std::uint64_t>(info.type));
            hash_u64_update(hash, static_cast<std::uint64_t>(info.use));
            hash_u64_update(hash, static_cast<std::uint64_t>(info.antihalation));
            hash_u64_update(hash, static_cast<std::uint64_t>(info.channelModel));
            hash_string_update(hash, info.referenceIlluminant.value);
            hash_string_update(hash, info.viewingIlluminant.value);
            hash_consumed_profile_samples(hash, data);
            return hash == 0 ? 1u : hash;
        }

        template <typename ProfileT>
        bool load_validated_profile_json_impl(
            const std::string& jsonPath,
            SelectedProfileKind kind,
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
            ctx.role = kind == SelectedProfileKind::Print ? "print" : "film";

            SpektrafilmProfileInfo info;
            if (!parse_profile_info(root, kind, info, ctx, error)) {
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }

            SpektrafilmProfileSamples data;
            if (!parse_profile_samples(root, data, ctx, error)) {
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }

            outProfile.info = std::move(info);
            outProfile.data = std::move(data);
            outProfile.assetVersionToken =
                build_profile_asset_version_token(outProfile.info, outProfile.data);
            if (outDiagnostic) {
                outDiagnostic->clear();
            }
            return outProfile.assetVersionToken != 0;
        }

        bool load_validated_film_profile_json(
            const std::string& jsonPath,
            ValidatedFilmProfile& outProfile,
            std::string* outDiagnostic) {
            if (!load_validated_profile_json_impl(
                    jsonPath,
                    SelectedProfileKind::Film,
                    outProfile,
                    outDiagnostic)) {
                return false;
            }
            SelectedProfileContext ctx;
            ctx.path = jsonPath;
            ctx.key = outProfile.info.stock;
            ctx.role = "film";
            std::string error;
            if (!build_film_profile_digest(
                    outProfile.info,
                    ctx,
                    outProfile.digest,
                    error)) {
                outProfile = ValidatedFilmProfile{};
                if (outDiagnostic) {
                    *outDiagnostic = error;
                }
                return false;
            }
            return true;
        }

        bool load_validated_print_profile_json(
            const std::string& jsonPath,
            ValidatedPrintProfile& outProfile,
            std::string* outDiagnostic) {
            return load_validated_profile_json_impl(
                jsonPath,
                SelectedProfileKind::Print,
                outProfile,
                outDiagnostic);
        }

    } // namespace

    struct ProfileAssetStore::CacheState {
        std::mutex mutex;
        std::unordered_map<std::string, std::shared_ptr<const ValidatedFilmProfile>> filmProfiles;
        std::unordered_map<std::string, std::shared_ptr<const ValidatedPrintProfile>> printProfiles;
    };

    ProfileAssetStore::ProfileAssetStore()
        : _cache(std::make_unique<CacheState>()) {
    }

    ProfileAssetStore::~ProfileAssetStore() = default;

    std::shared_ptr<const ValidatedFilmProfile> ProfileAssetStore::load_film_profile_by_key(
        const Spektrafilm::ProfileCatalog& catalog,
        const std::string& key,
        std::string* outDiagnostic) {
        const Spektrafilm::ProfileCatalogEntry* source =
            find_profile_entry(catalog.filmProfiles, key);
        if (!source) {
            set_missing_profile_diagnostic(outDiagnostic, "film", key);
            return {};
        }

        std::shared_ptr<const ValidatedFilmProfile> cached;
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            const auto it = _cache->filmProfiles.find(key);
            if (it != _cache->filmProfiles.end()) {
                cached = it->second;
            }
        }
        if (cached) {
            if (outDiagnostic) {
                outDiagnostic->clear();
            }
            return cached;
        }

        ValidatedFilmProfile parsed;
        std::string diagnostic;
        if (!load_validated_film_profile_json(source->sourcePath, parsed, &diagnostic)) {
            std::shared_ptr<const ValidatedFilmProfile> published;
            {
                std::lock_guard<std::mutex> lock(_cache->mutex);
                const auto it = _cache->filmProfiles.find(key);
                if (it != _cache->filmProfiles.end()) {
                    published = it->second;
                }
            }
            if (published) {
                if (outDiagnostic) {
                    outDiagnostic->clear();
                }
                return published;
            }
            if (outDiagnostic) {
                *outDiagnostic = diagnostic;
            }
            if (JTRACE_ENABLED(1)) {
                JTRACE("PROFILE", diagnostic);
            }
            return {};
        }

        std::shared_ptr<const ValidatedFilmProfile> loaded =
            std::make_shared<ValidatedFilmProfile>(std::move(parsed));
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            const auto [it, inserted] = _cache->filmProfiles.try_emplace(key, loaded);
            (void)inserted;
            loaded = it->second;
        }
        if (outDiagnostic) {
            outDiagnostic->clear();
        }
        return loaded;
    }

    std::shared_ptr<const ValidatedPrintProfile> ProfileAssetStore::load_print_profile_by_key(
        const Spektrafilm::ProfileCatalog& catalog,
        const std::string& key,
        std::string* outDiagnostic) {
        const Spektrafilm::ProfileCatalogEntry* source =
            find_profile_entry(catalog.printProfiles, key);
        if (!source) {
            set_missing_profile_diagnostic(outDiagnostic, "print", key);
            return {};
        }

        std::shared_ptr<const ValidatedPrintProfile> cached;
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            const auto it = _cache->printProfiles.find(key);
            if (it != _cache->printProfiles.end()) {
                cached = it->second;
            }
        }
        if (cached) {
            if (outDiagnostic) {
                outDiagnostic->clear();
            }
            return cached;
        }

        ValidatedPrintProfile parsed;
        std::string diagnostic;
        if (!load_validated_print_profile_json(source->sourcePath, parsed, &diagnostic)) {
            std::shared_ptr<const ValidatedPrintProfile> published;
            {
                std::lock_guard<std::mutex> lock(_cache->mutex);
                const auto it = _cache->printProfiles.find(key);
                if (it != _cache->printProfiles.end()) {
                    published = it->second;
                }
            }
            if (published) {
                if (outDiagnostic) {
                    outDiagnostic->clear();
                }
                return published;
            }
            if (outDiagnostic) {
                *outDiagnostic = diagnostic;
            }
            if (JTRACE_ENABLED(1)) {
                JTRACE("PROFILE", diagnostic);
            }
            return {};
        }

        std::shared_ptr<const ValidatedPrintProfile> loaded =
            std::make_shared<ValidatedPrintProfile>(std::move(parsed));
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            const auto [it, inserted] = _cache->printProfiles.try_emplace(key, loaded);
            (void)inserted;
            loaded = it->second;
        }
        if (outDiagnostic) {
            outDiagnostic->clear();
        }
        return loaded;
    }

    SelectedProfileResult ProfileAssetStore::selected_profiles_for_route(
        const Spektrafilm::ProfileCatalog& catalog,
        const SelectedProfileRequest& request) {
        SelectedProfileResult result;
        result.filmProfile =
            load_film_profile_by_key(catalog, request.filmProfileKey, &result.diagnostic);
        if (!result.filmProfile) {
            return result;
        }

        if (!Spektrafilm::scan_route_is_print(request.scanRoute)) {
            result.valid = true;
            result.diagnostic.clear();
            return result;
        }

        result.printProfile =
            load_print_profile_by_key(catalog, request.printProfileKey, &result.diagnostic);
        if (!result.printProfile) {
            return result;
        }
        result.valid = true;
        result.diagnostic.clear();
        return result;
    }

    void ProfileAssetStore::release_cached_payloads() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            _cache->filmProfiles.clear();
            _cache->printProfiles.clear();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace Profiles
