#include "ProfileAssets.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "Hash.h"
#include "Logging.h"
#include "ProfileJSONLoader.h"

namespace Profiles {
    namespace {

        constexpr std::size_t kSelectedProfileCacheCapacity = 4;

        template <typename ProfileT>
        struct CacheEntry {
            std::string sourcePath;
            std::uint64_t sourceVersion = 0;
            std::shared_ptr<const ProfileT> profile;
        };

        template <typename ProfileT>
        std::shared_ptr<const ProfileT> find_cached_profile(
            std::vector<CacheEntry<ProfileT>>& cache,
            const Spektrafilm::ProfileCatalogEntry& source) {
            for (std::size_t i = 0; i < cache.size(); ++i) {
                CacheEntry<ProfileT>& entry = cache[i];
                if (entry.sourcePath != source.sourcePath || entry.sourceVersion != source.sourceVersion) {
                    continue;
                }
                if (i != 0) {
                    std::swap(cache[0], cache[i]);
                }
                return cache[0].profile;
            }
            return {};
        }

        template <typename ProfileT>
        void store_cached_profile(
            std::vector<CacheEntry<ProfileT>>& cache,
            const Spektrafilm::ProfileCatalogEntry& source,
            std::shared_ptr<const ProfileT> profile) {
            cache.erase(
                std::remove_if(
                    cache.begin(),
                    cache.end(),
                    [&source](const CacheEntry<ProfileT>& entry) {
                        return entry.sourcePath == source.sourcePath;
                    }),
                cache.end());
            cache.insert(
                cache.begin(),
                CacheEntry<ProfileT>{source.sourcePath, source.sourceVersion, std::move(profile)});
            if (cache.size() > kSelectedProfileCacheCapacity) {
                cache.resize(kSelectedProfileCacheCapacity);
            }
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

    } // namespace

    struct ProfileAssetStore::CacheState {
        std::mutex mutex;
        std::vector<CacheEntry<ValidatedFilmProfile>> filmProfiles;
        std::vector<CacheEntry<ValidatedPrintProfile>> printProfiles;
    };

    ProfileDigest build_profile_digest(const SpektrafilmProfileInfo& info, ProfileRole role) {
        ProfileDigest digest{};
        digest.profileRole = role;

        const bool positive = info.type == Spektrafilm::ProfilePolarity::Positive;
        if (positive) {
            digest.gammaSamelayerRgb = {{0.12f, 0.08f, 0.06f}};
            digest.gammaInterlayerRToGb = {{0.12f, 0.06f}};
            digest.gammaInterlayerGToRb = {{0.08f, 0.06f}};
            digest.gammaInterlayerBToRg = {{0.06f, 0.06f}};
            digest.dirGammaSource = "positive-default";
        } else {
            digest.gammaSamelayerRgb = {{0.336f, 0.319f, 0.273f}};
            digest.gammaInterlayerRToGb = {{0.353f, 0.302f}};
            digest.gammaInterlayerGToRb = {{0.154f, 0.353f}};
            digest.gammaInterlayerBToRg = {{0.168f, 0.226f}};
            digest.dirGammaSource = "negative-default";
        }

        if (info.stock == "fujifilm_velvia_100") {
            digest.gammaSamelayerRgb = {{0.108f, 0.072f, 0.054f}};
            digest.gammaInterlayerRToGb = {{0.108f, 0.054f}};
            digest.gammaInterlayerGToRb = {{0.072f, 0.054f}};
            digest.gammaInterlayerBToRg = {{0.054f, 0.054f}};
            digest.dirGammaSource = "stock-override";
        } else if (info.stock == "fujifilm_provia_100f") {
            digest.gammaSamelayerRgb = {{0.156f, 0.104f, 0.078f}};
            digest.gammaInterlayerRToGb = {{0.156f, 0.078f}};
            digest.gammaInterlayerGToRb = {{0.104f, 0.078f}};
            digest.gammaInterlayerBToRg = {{0.078f, 0.078f}};
            digest.dirGammaSource = "stock-override";
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
                digest.halationPresetApplied = false;
                break;
        }
        return digest;
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

    ProfileAssetStore::ProfileAssetStore()
        : _cache(std::make_unique<CacheState>()) {
    }

    ProfileAssetStore::~ProfileAssetStore() = default;

    std::shared_ptr<const ValidatedFilmProfile> ProfileAssetStore::load_film_profile_by_key(
        const Spektrafilm::ProfileCatalog& catalog,
        const std::string& key,
        std::string* outDiagnostic) {
        const Spektrafilm::ProfileCatalogEntry* source = find_profile_entry(catalog.filmProfiles, key);
        if (!source) {
            set_missing_profile_diagnostic(outDiagnostic, "film", key);
            return {};
        }
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            if (std::shared_ptr<const ValidatedFilmProfile> cached =
                    find_cached_profile(_cache->filmProfiles, *source)) {
                if (outDiagnostic) {
                    outDiagnostic->clear();
                }
                return cached;
            }
        }

        ValidatedFilmProfile parsed;
        std::string diagnostic;
        if (!load_validated_film_profile_json(source->sourcePath, parsed, &diagnostic)) {
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
            store_cached_profile(_cache->filmProfiles, *source, loaded);
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
        const Spektrafilm::ProfileCatalogEntry* source = find_profile_entry(catalog.printProfiles, key);
        if (!source) {
            set_missing_profile_diagnostic(outDiagnostic, "print", key);
            return {};
        }
        {
            std::lock_guard<std::mutex> lock(_cache->mutex);
            if (std::shared_ptr<const ValidatedPrintProfile> cached =
                    find_cached_profile(_cache->printProfiles, *source)) {
                if (outDiagnostic) {
                    outDiagnostic->clear();
                }
                return cached;
            }
        }

        ValidatedPrintProfile parsed;
        std::string diagnostic;
        if (!load_validated_print_profile_json(source->sourcePath, parsed, &diagnostic)) {
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
            store_cached_profile(_cache->printProfiles, *source, loaded);
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
        result.filmProfile = load_film_profile_by_key(catalog, request.filmProfileKey, &result.diagnostic);
        if (!result.filmProfile) {
            return result;
        }

        if (!Spektrafilm::scan_route_is_print(request.scanRoute)) {
            result.valid = true;
            result.diagnostic.clear();
            return result;
        }

        result.printProfile = load_print_profile_by_key(catalog, request.printProfileKey, &result.diagnostic);
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
