#include "ResourceAssetLibrary.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <system_error>
#include <utility>

#include "Logging.h"
#include "Illuminants.h"
#include "nlohmann/json.hpp"

namespace JuicerAssets {

    struct Library::StaticNoiseAssetSet {
        std::string stbnPath;
        std::string wangTilesPath;
        std::string wangMetadataPath;
        std::uint64_t version = 0;
    };

    struct Library::IlluminantFilterAssetSet {
        std::string d65Path;
        std::string d55Path;
        std::string d50Path;
        std::string tungstenPath;
        std::string kinoton75PPath;
        std::string kg3Path;
        std::string lensTransmissionPath;
        std::uint64_t version = 0;
    };

    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;

        constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
        constexpr std::uint64_t kFnvPrime64 = 1099511628211ull;

        struct IlluminantFilterCurveCacheEntry {
            IlluminantFilterCurveSet curves;
            bool ready = false;
        };

        std::uint64_t fnv1a_append(std::uint64_t hash, const void* data, size_t sizeBytes) {
            const unsigned char* bytes = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < sizeBytes; ++i) {
                hash ^= static_cast<std::uint64_t>(bytes[i]);
                hash *= kFnvPrime64;
            }
            return hash;
        }

        void hash_string(std::uint64_t& hash, const std::string& value) {
            hash = fnv1a_append(hash, value.data(), value.size());
        }

        template <typename T>
        void hash_value(std::uint64_t& hash, const T& value) {
            hash = fnv1a_append(hash, &value, sizeof(value));
        }

        bool read_file_bytes(
            const std::string& path,
            std::string& out,
            std::uint64_t* outHash = nullptr) {
            out.clear();
            if (outHash) {
                *outHash = 0;
            }
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) {
                return false;
            }
            const std::streamsize size = file.tellg();
            if (size < 0) {
                return false;
            }
            out.resize(static_cast<std::size_t>(size));
            file.seekg(0, std::ios::beg);
            if (size > 0 && !file.read(out.data(), size)) {
                out.clear();
                return false;
            }
            if (outHash) {
                *outHash = fnv1a_append(kFnvOffsetBasis64, out.data(), out.size());
            }
            return true;
        }

        std::string data_path_string(const std::string& dataDir, std::initializer_list<const char*> segments) {
            fs::path path(dataDir);
            for (const char* segment : segments) {
                if (segment && *segment) {
                    path /= segment;
                }
            }
            path.make_preferred();
            return path.string();
        }

        std::string neutral_print_calibration_path(const std::string& dataDir) {
            return data_path_string(dataDir, {"filters", "neutral_print_filters.json"});
        }

        struct NeutralPrintCalibrationSnapshot {
            NeutralPrintCalibrationStatus rootStatus =
                NeutralPrintCalibrationStatus::MissingFile;
            Json root;
            std::string diagnostic;
        };

        std::shared_ptr<const NeutralPrintCalibrationSnapshot>
        load_neutral_print_calibration_snapshot(const std::string& path) {
            auto snapshot = std::make_shared<NeutralPrintCalibrationSnapshot>();
            std::string bytes;
            if (!read_file_bytes(path, bytes)) {
                std::error_code ec;
                if (fs::exists(path, ec) && !ec) {
                    snapshot->rootStatus = NeutralPrintCalibrationStatus::Malformed;
                    snapshot->diagnostic =
                        "MalformedNeutralPrintCalibration phase=4A field=resource_read";
                }
                return snapshot;
            }
            Json root = Json::parse(bytes, nullptr, false);
            if (root.is_discarded() || !root.is_object()) {
                snapshot->rootStatus = NeutralPrintCalibrationStatus::Malformed;
                snapshot->diagnostic =
                    "MalformedNeutralPrintCalibration phase=4A field=root";
                return snapshot;
            }
            snapshot->rootStatus = NeutralPrintCalibrationStatus::Found;
            snapshot->root = std::move(root);
            return snapshot;
        }

        std::string measured_dichroic_relative_path(const std::string& setKey, const char* channel) {
            return "Resources/filters/dichroics/" + setKey + "/filter_" + channel + ".csv";
        }

        struct MeasuredDichroicChannelRequest {
            const std::string& path;
            const std::string& relativePath;
        };

        bool parse_measured_dichroic_channel(
            const MeasuredDichroicChannelRequest& request,
            std::uint64_t& outHash,
            std::array<float, 81>* outTransmittance,
            std::string& diagnostic) {
            std::string bytes;
            std::uint64_t fileHash = 0;
            if (!read_file_bytes(request.path, bytes, &fileHash)) {
                diagnostic = "SelectedDichroicResourceMissing phase=4A resource=" + request.relativePath;
                return false;
            }

            std::vector<std::pair<float, float>> pairs;
            std::istringstream input(bytes);
            std::string line;
            std::size_t lineNumber = 0;
            while (std::getline(input, line)) {
                ++lineNumber;
                const std::size_t comment = line.find('#');
                if (comment != std::string::npos) {
                    line.erase(comment);
                }
                const std::size_t first = line.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) {
                    continue;
                }

                std::istringstream row(line.substr(first));
                float wavelength = 0.0f;
                float percentTransmittance = 0.0f;
                if (!(row >> wavelength)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                while (row.peek() == ',' || row.peek() == ';') {
                    row.get();
                }
                if (!(row >> percentTransmittance) ||
                    !std::isfinite(wavelength) ||
                    !std::isfinite(percentTransmittance)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                row >> std::ws;
                if (!row.eof()) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A line=" + std::to_string(lineNumber);
                    return false;
                }
                pairs.emplace_back(wavelength, percentTransmittance);
            }

            const std::vector<std::pair<float, float>> resampled =
                Spectral::resample_pairs_akima_to_reference_axis(pairs);
            if (resampled.size() != 81u) {
                diagnostic = "MalformedSelectedDichroicResource phase=4A field=akima_reference_axis";
                return false;
            }

            std::array<float, 81> transmittance{};
            for (std::size_t i = 0; i < resampled.size(); ++i) {
                const float value = resampled[i].second * 0.01f;
                if (!std::isfinite(value)) {
                    diagnostic = "MalformedSelectedDichroicResource phase=4A field=canonical_axis_coverage";
                    return false;
                }
                transmittance[i] = value;
            }

            std::uint64_t hash = kFnvOffsetBasis64;
            constexpr std::uint32_t kSchemaVersion = 1u;
            hash_value(hash, kSchemaVersion);
            hash_string(hash, request.relativePath);
            hash_value(hash, fileHash);
            hash = fnv1a_append(hash, transmittance.data(), sizeof(transmittance));
            outHash = hash;
            if (outTransmittance) {
                *outTransmittance = transmittance;
            }
            return outHash != 0;
        }

        std::string noise_asset_path(const std::string& dataDir, std::initializer_list<const char*> segments) {
            if (dataDir.empty()) {
                return {};
            }
            return data_path_string(dataDir, segments);
        }

        Library::StaticNoiseAssetSet make_static_noise_assets(const std::string& dataDir) {
            Library::StaticNoiseAssetSet asset;
            asset.stbnPath = noise_asset_path(dataDir, {"Noise", "stbn_scalar_512x512x256_u8.bin"});
            asset.wangTilesPath = noise_asset_path(dataDir, {"Noise", "Wang", "wang_tiles_256x256x16_u8.bin"});
            asset.wangMetadataPath = noise_asset_path(dataDir, {"Noise", "Wang", "tiles.json"});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        StbnNoisePayload load_stbn_noise_payload(const Library::StaticNoiseAssetSet& assets) {
            StbnNoisePayload payload;
            payload.width = 512;
            payload.height = 512;
            payload.frames = 256;
            payload.version = assets.version;

            if (assets.stbnPath.empty()) {
                payload.error = "STBN load failed: data directory missing";
                return payload;
            }

            fs::path path = fs::path(assets.stbnPath);
            path.make_preferred();

            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) {
                payload.error = "STBN load failed: cannot open logical noise asset";
                return payload;
            }

            const std::streamsize size = file.tellg();
            if (size <= 0) {
                payload.error = "STBN load failed: logical noise asset is empty";
                return payload;
            }

            const std::size_t expected = static_cast<std::size_t>(payload.width) *
                                         static_cast<std::size_t>(payload.height) *
                                         static_cast<std::size_t>(payload.frames);
            if (static_cast<std::size_t>(size) != expected) {
                payload.error = "STBN load failed: logical noise asset has unexpected size";
                return payload;
            }

            payload.data.resize(expected);
            file.seekg(0, std::ios::beg);
            if (!file.read(reinterpret_cast<char*>(payload.data.data()), size)) {
                payload.error = "STBN load failed: logical noise asset read error";
                payload.data.clear();
                return payload;
            }

            payload.valid = true;
            return payload;
        }

        struct WangTileEdges {
            int left = 0;
            int right = 0;
            int top = 0;
            int bottom = 0;
        };

        std::size_t wang_lut_index(const WangTileEdges& edges, int colors) {
            const std::size_t c = static_cast<std::size_t>(colors);
            return (((static_cast<std::size_t>(edges.left) * c + static_cast<std::size_t>(edges.right)) * c +
                     static_cast<std::size_t>(edges.top)) *
                        c +
                    static_cast<std::size_t>(edges.bottom));
        }

        WangNoisePayload load_wang_noise_payload(const Library::StaticNoiseAssetSet& assets) {
            WangNoisePayload payload;
            payload.version = assets.version;

            if (assets.wangTilesPath.empty() || assets.wangMetadataPath.empty()) {
                payload.error = "Wang tiles load failed: data directory missing";
                return payload;
            }

            fs::path binPath = fs::path(assets.wangTilesPath);
            fs::path jsonPath = fs::path(assets.wangMetadataPath);
            binPath.make_preferred();
            jsonPath.make_preferred();

            if (!fs::exists(binPath) || !fs::exists(jsonPath)) {
                payload.error = "Wang tiles load failed: logical noise asset set is incomplete";
                return payload;
            }

            std::ifstream jf(jsonPath);
            if (!jf) {
                payload.error = "Wang tiles load failed: cannot open logical metadata asset";
                return payload;
            }

            Json root;
            try {
                jf >> root;
            } catch (const std::exception& e) {
                payload.error = std::string("Wang tiles load failed: invalid JSON ") + e.what();
                return payload;
            }

            if (!root.contains("resolution") || !root.contains("tiles") || !root.contains("colors") ||
                !root.contains("mapping")) {
                payload.error = "Wang tiles load failed: tiles.json missing required fields";
                return payload;
            }

            const int width = root.value("resolution", 0);
            const int height = width;
            const int count = root.value("tiles", 0);
            const int colors = root.value("colors", 0);
            if (width <= 0 || height <= 0 || count <= 0 || colors <= 0) {
                payload.error = "Wang tiles load failed: invalid metadata in tiles.json";
                return payload;
            }

            const std::size_t lutSize = static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors) *
                                        static_cast<std::size_t>(colors);
            std::vector<std::uint8_t> lut(lutSize, 0);

            const auto& mapping = root["mapping"];
            if (!mapping.is_array()) {
                payload.error = "Wang tiles load failed: mapping is not an array";
                return payload;
            }

            for (const auto& entry : mapping) {
                if (!entry.contains("index") || !entry.contains("labels")) {
                    continue;
                }
                const int idx = entry.value("index", 0);
                const auto& labels = entry["labels"];
                const int l = labels.value("L", 0);
                const int r = labels.value("R", 0);
                const int t = labels.value("T", 0);
                const int b = labels.value("B", 0);
                if (l < 0 || r < 0 || t < 0 || b < 0 ||
                    l >= colors || r >= colors || t >= colors || b >= colors) {
                    continue;
                }
                const std::size_t lutIndex = wang_lut_index(WangTileEdges{l, r, t, b}, colors);
                if (lutIndex < lut.size() && idx >= 0 && idx < count) {
                    lut[lutIndex] = static_cast<std::uint8_t>(idx);
                }
            }

            std::ifstream bin(binPath, std::ios::binary | std::ios::ate);
            if (!bin) {
                payload.error = "Wang tiles load failed: cannot open logical tile asset";
                return payload;
            }
            const std::streamsize size = bin.tellg();
            if (size <= 0) {
                payload.error = "Wang tiles load failed: logical tile asset is empty";
                return payload;
            }
            const std::size_t expected = static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(count);
            if (static_cast<std::size_t>(size) != expected) {
                payload.error = "Wang tiles load failed: logical tile asset has unexpected size";
                return payload;
            }

            std::vector<std::uint8_t> tiles(expected);
            bin.seekg(0, std::ios::beg);
            if (!bin.read(reinterpret_cast<char*>(tiles.data()), size)) {
                payload.error = "Wang tiles load failed: logical tile asset read error";
                return payload;
            }

            payload.tiles = std::move(tiles);
            payload.lut = std::move(lut);
            payload.width = width;
            payload.height = height;
            payload.count = count;
            payload.colors = colors;
            payload.valid = true;
            return payload;
        }

        StaticNoisePayloadSet load_static_noise_payloads(const Library::StaticNoiseAssetSet& assets) {
            StaticNoisePayloadSet payloads;
            payloads.stbn = load_stbn_noise_payload(assets);
            payloads.wang = load_wang_noise_payload(assets);
            payloads.version = assets.version;
            return payloads;
        }

        Library::IlluminantFilterAssetSet make_illuminant_filter_assets(const std::string& dataDir) {
            Library::IlluminantFilterAssetSet asset;
            asset.d65Path = data_path_string(dataDir, {"illuminants", "D65.csv"});
            asset.d55Path = data_path_string(dataDir, {"illuminants", "D55.csv"});
            asset.d50Path = data_path_string(dataDir, {"illuminants", "D50.csv"});
            asset.tungstenPath = data_path_string(dataDir, {"illuminants", "T.csv"});
            asset.kinoton75PPath = data_path_string(dataDir, {"illuminants", "K75P.csv"});
            asset.kg3Path = data_path_string(dataDir, {"filters", "heat_absorbing", "schott", "KG3.csv"});
            asset.lensTransmissionPath = data_path_string(
                dataDir,
                {"filters", "lens_transmission", "canon", "canon_24_f28_is.csv"});
            asset.version = Library::kProcessAssetVersion;
            return asset;
        }

        IlluminantFilterCurveSet load_illuminant_filter_curves(const Library::IlluminantFilterAssetSet& asset) {
            IlluminantFilterCurveSet curves;
            curves.d65 = Spectral::build_curve_D65_pinned(asset.d65Path);
            curves.d55 = Spectral::build_curve_D55_pinned(asset.d55Path);
            curves.d50 = Spectral::build_curve_D50_pinned(asset.d50Path);
            curves.tungsten = Spectral::build_curve_T_pinned(asset.tungstenPath);
            curves.kinoton75P = Spectral::build_curve_K75P_pinned(asset.kinoton75PPath);
            curves.tungstenKg3 = Spectral::build_curve_TH_KG3_pinned(asset.kg3Path);
            curves.tungstenKg3Lens = Spectral::build_curve_TH_KG3_L_pinned(
                asset.kg3Path,
                asset.lensTransmissionPath);
            curves.version = asset.version;
            return curves;
        }

        bool curve_is_on_reference_axis(const Spectral::Curve& curve) {
            const size_t expected = static_cast<size_t>(Spectral::gShape.K);
            return curve.lambda_nm.size() == expected && curve.linear.size() == expected;
        }

        bool illuminant_filter_curves_complete(const IlluminantFilterCurveSet& curves) {
            return curve_is_on_reference_axis(curves.d65) &&
                   curve_is_on_reference_axis(curves.d55) &&
                   curve_is_on_reference_axis(curves.d50) &&
                   curve_is_on_reference_axis(curves.tungsten) &&
                   curve_is_on_reference_axis(curves.kinoton75P) &&
                   curve_is_on_reference_axis(curves.tungstenKg3) &&
                   curve_is_on_reference_axis(curves.tungstenKg3Lens);
        }

    } // namespace

    struct Library::StaticNoisePayloadCacheState {
        std::mutex mutex;
        std::shared_ptr<const StaticNoisePayloadSet> payloads;
    };

    struct Library::IlluminantFilterCurveCacheState {
        std::mutex mutex;
        IlluminantFilterCurveCacheEntry entry;
    };

    struct Library::NeutralPrintCalibrationCacheState {
        std::mutex mutex;
        std::shared_ptr<const NeutralPrintCalibrationSnapshot> snapshot;
    };

    Library::Library(std::string dataDir)
        : _dataDir(std::move(dataDir)),
          _staticNoiseAssets(std::make_unique<StaticNoiseAssetSet>()),
          _illuminantFilterAssets(std::make_unique<IlluminantFilterAssetSet>()),
          _staticNoisePayloadCache(std::make_unique<StaticNoisePayloadCacheState>()),
          _illuminantFilterCurveCache(
              std::make_unique<IlluminantFilterCurveCacheState>()),
          _neutralPrintCalibrationCache(
              std::make_unique<NeutralPrintCalibrationCacheState>()),
          _selectedProfileAssets(
              std::make_unique<Profiles::ProfileAssetStore>()) {
    }

    Library::~Library() = default;

    void Library::ensure_catalogs() {
        std::call_once(_catalogOnce, [this]() {
            load_catalogs();
        });
    }

    void Library::ensure_static_noise_assets() {
        std::call_once(_staticNoiseOnce, [this]() {
            load_static_noise_assets();
        });
    }

    void Library::ensure_illuminant_filter_assets() {
        std::call_once(_illuminantFilterOnce, [this]() {
            load_illuminant_filter_assets();
        });
    }

    void Library::load_catalogs() {
        const bool traceCatalog = JTRACE_ENABLED(1);
        _spektrafilmProfileCatalog =
            Spektrafilm::build_profile_catalog(_dataDir);
        if (!_spektrafilmProfileCatalog.valid) {
            if (traceCatalog) {
                JTRACE(
                    "CATALOG",
                    "spektrafilm profile catalog unavailable: " +
                        _spektrafilmProfileCatalog.failure);
            }
            return;
        }

        if (traceCatalog) {
            std::ostringstream oss;
            oss << "spektrafilm profile catalog film="
                << _spektrafilmProfileCatalog.filmProfiles.size()
                << " print="
                << _spektrafilmProfileCatalog.printProfiles.size()
                << " defaultFilm="
                << (_spektrafilmProfileCatalog.defaultFilmPresent ? 1 : 0)
                << " defaultPrint="
                << (_spektrafilmProfileCatalog.defaultPrintPresent ? 1 : 0);
            JTRACE("CATALOG", oss.str());
        }
    }

    void Library::load_static_noise_assets() {
        *_staticNoiseAssets = make_static_noise_assets(_dataDir);
    }

    void Library::load_illuminant_filter_assets() {
        *_illuminantFilterAssets = make_illuminant_filter_assets(_dataDir);
    }

    const Spektrafilm::ProfileCatalog& Library::spektrafilm_profile_catalog() {
        ensure_catalogs();
        return _spektrafilmProfileCatalog;
    }

    std::shared_ptr<const Profiles::ValidatedFilmProfile>
    Library::selected_film_profile_for_key(const std::string& key) {
        ensure_catalogs();
        return _selectedProfileAssets->load_film_profile_by_key(
            _spektrafilmProfileCatalog,
            key);
    }

    SelectedProfileResult Library::selected_profiles_for_route(
        const SelectedProfileRequest& request) {
        ensure_catalogs();
        return _selectedProfileAssets->selected_profiles_for_route(
            _spektrafilmProfileCatalog,
            request);
    }

    std::shared_ptr<const StaticNoisePayloadSet>
    Library::static_noise_payloads() {
        ensure_static_noise_assets();
        std::lock_guard<std::mutex> lock(_staticNoisePayloadCache->mutex);
        if (!_staticNoisePayloadCache->payloads) {
            _staticNoisePayloadCache->payloads =
                std::make_shared<StaticNoisePayloadSet>(
                    load_static_noise_payloads(*_staticNoiseAssets));
        }
        return _staticNoisePayloadCache->payloads;
    }

    const IlluminantFilterCurveSet& Library::illuminant_filter_curves() {
        ensure_illuminant_filter_assets();
        std::lock_guard<std::mutex> lock(
            _illuminantFilterCurveCache->mutex);
        IlluminantFilterCurveCacheEntry& entry =
            _illuminantFilterCurveCache->entry;
        if (!entry.ready) {
            entry.curves =
                load_illuminant_filter_curves(*_illuminantFilterAssets);
            entry.ready =
                illuminant_filter_curves_complete(entry.curves);
        }
        return entry.curves;
    }

    MeasuredDichroicResourceIdentity Library::measured_dichroic_resource_identity(const std::string& setKey) {
        MeasuredDichroicResourceIdentity result;
        result.setKey = setKey;
        constexpr std::array<const char*, 3> kChannelsCmy{{"c", "m", "y"}};
        for (std::size_t channel = 0; channel < kChannelsCmy.size(); ++channel) {
            const char* channelKey = kChannelsCmy[channel];
            result.resourcePathsCmy[channel] = measured_dichroic_relative_path(setKey, channelKey);
            const std::string fileName = std::string("filter_") + channelKey + ".csv";
            const std::string path =
                data_path_string(_dataDir, {"filters", "dichroics", setKey.c_str(), fileName.c_str()});
            if (!parse_measured_dichroic_channel(
                    MeasuredDichroicChannelRequest{path, result.resourcePathsCmy[channel]},
                    result.resourceHashesCmy[channel],
                    nullptr,
                    result.diagnostic)) {
                return result;
            }
        }

        std::uint64_t hash = kFnvOffsetBasis64;
        constexpr std::uint32_t kSchemaVersion = 1u;
        hash_value(hash, kSchemaVersion);
        hash_string(hash, result.setKey);
        for (std::size_t channel = 0; channel < result.resourcePathsCmy.size(); ++channel) {
            hash_string(hash, result.resourcePathsCmy[channel]);
            hash_value(hash, result.resourceHashesCmy[channel]);
        }
        result.hash = hash;
        result.valid = result.hash != 0;
        return result;
    }

    MeasuredDichroicCurveResult Library::measured_dichroic_curves(const std::string& setKey) {
        MeasuredDichroicCurveResult result;
        constexpr std::array<const char*, 3> kChannelsCmy{{"c", "m", "y"}};
        std::uint64_t hash = kFnvOffsetBasis64;
        constexpr std::uint32_t kSchemaVersion = 1u;
        hash_value(hash, kSchemaVersion);
        hash_string(hash, setKey);
        for (std::size_t channel = 0; channel < kChannelsCmy.size(); ++channel) {
            const char* channelKey = kChannelsCmy[channel];
            const std::string relativePath = measured_dichroic_relative_path(setKey, channelKey);
            const std::string fileName = std::string("filter_") + channelKey + ".csv";
            const std::string path =
                data_path_string(_dataDir, {"filters", "dichroics", setKey.c_str(), fileName.c_str()});
            if (!parse_measured_dichroic_channel(
                    MeasuredDichroicChannelRequest{path, relativePath},
                    result.resourceHashesCmy[channel],
                    &result.transmittanceCmy[channel],
                    result.diagnostic)) {
                return result;
            }
            hash_string(hash, relativePath);
            hash_value(hash, result.resourceHashesCmy[channel]);
        }
        result.hash = hash;
        result.valid = result.hash != 0;
        return result;
    }

    NeutralPrintCalibrationResult Library::neutral_print_calibration(
        const std::string& printProfileKey,
        const std::string& printIlluminantKey,
        const std::string& filmProfileKey) {
        NeutralPrintCalibrationResult result;
        std::shared_ptr<const NeutralPrintCalibrationSnapshot> snapshot;
        {
            std::lock_guard<std::mutex> lock(_neutralPrintCalibrationCache->mutex);
            snapshot = _neutralPrintCalibrationCache->snapshot;
        }
        if (!snapshot) {
            std::shared_ptr<const NeutralPrintCalibrationSnapshot> loaded =
                load_neutral_print_calibration_snapshot(
                    neutral_print_calibration_path(_dataDir));
            {
                std::lock_guard<std::mutex> lock(_neutralPrintCalibrationCache->mutex);
                if (!_neutralPrintCalibrationCache->snapshot) {
                    _neutralPrintCalibrationCache->snapshot = std::move(loaded);
                }
                snapshot = _neutralPrintCalibrationCache->snapshot;
            }
        }
        if (snapshot->rootStatus != NeutralPrintCalibrationStatus::Found) {
            result.status = snapshot->rootStatus;
            result.diagnostic = snapshot->diagnostic;
            return result;
        }

        const Json& root = snapshot->root;
        const auto printIt = root.find(printProfileKey);
        if (printIt == root.end()) {
            result.status = NeutralPrintCalibrationStatus::MissingEntry;
        } else if (!printIt->is_object()) {
            result.status = NeutralPrintCalibrationStatus::Malformed;
            result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=print_profile";
        } else {
            const auto illuminantIt = printIt->find(printIlluminantKey);
            if (illuminantIt == printIt->end()) {
                result.status = NeutralPrintCalibrationStatus::MissingEntry;
            } else if (!illuminantIt->is_object()) {
                result.status = NeutralPrintCalibrationStatus::Malformed;
                result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=print_illuminant";
            } else {
                const auto filmIt = illuminantIt->find(filmProfileKey);
                if (filmIt == illuminantIt->end()) {
                    result.status = NeutralPrintCalibrationStatus::MissingEntry;
                } else if (!filmIt->is_array() || filmIt->size() != result.cmyCc.size()) {
                    result.status = NeutralPrintCalibrationStatus::Malformed;
                    result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                } else {
                    result.status = NeutralPrintCalibrationStatus::Found;
                    for (std::size_t channel = 0; channel < result.cmyCc.size(); ++channel) {
                        const Json& value = (*filmIt)[channel];
                        if (!value.is_number()) {
                            result.status = NeutralPrintCalibrationStatus::Malformed;
                            result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                            break;
                        }
                        const double cc = value.get<double>();
                        if (!std::isfinite(cc)) {
                            result.status = NeutralPrintCalibrationStatus::Malformed;
                            result.diagnostic = "MalformedNeutralPrintCalibration phase=4A field=cmy_cc";
                            break;
                        }
                        result.cmyCc[channel] = static_cast<float>(cc);
                    }
                }
            }
        }

        return result;
    }

    void Library::release_cached_payloads() noexcept {
        try {
            if (_staticNoisePayloadCache) {
                std::lock_guard<std::mutex> lock(
                    _staticNoisePayloadCache->mutex);
                _staticNoisePayloadCache->payloads.reset();
            }
            if (_illuminantFilterCurveCache) {
                std::lock_guard<std::mutex> lock(
                    _illuminantFilterCurveCache->mutex);
                _illuminantFilterCurveCache->entry =
                    IlluminantFilterCurveCacheEntry{};
            }
            if (_neutralPrintCalibrationCache) {
                std::lock_guard<std::mutex> lock(
                    _neutralPrintCalibrationCache->mutex);
                _neutralPrintCalibrationCache->snapshot.reset();
            }
            if (_selectedProfileAssets) {
                _selectedProfileAssets->release_cached_payloads();
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerAssets
