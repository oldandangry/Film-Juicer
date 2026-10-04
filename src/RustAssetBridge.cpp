#include "RustAssetBridge.h"

#include <algorithm>
#include <array>
#include <span>
#include <mutex>
#include <exception>
#include <cstddef>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Logging.h"

namespace JuicerAssets {
    namespace {
        [[noreturn]] void fail(FjStatus status, std::string_view message) {
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc();
            }
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, message);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }

        [[noreturn]] void invalid_path() {
            fail({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "invalid native resource path extent or encoding");
        }

        void report_cleanup(FjStatus result) noexcept {
            if (result.category != FJ_STATUS_SUCCESS) {
                try {
                    JTRACE("ASSETS", "unexpected Rust asset owner release failure category=" + std::to_string(result.category));
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }
        }
        struct NoiseDeleter {
            void operator()(FjNoise* owner) const noexcept {
#if defined(JUICER_NOISE_TEST_HOOK)
                NoiseTest::observe(NoiseTest::Operation::Release);
#endif
                report_cleanup(fj_legacy_noise_release(owner, nullptr));
            }
        };
        struct FilmDeleter {
            void operator()(FjFilmProfile* owner) const noexcept {
                report_cleanup(fj_legacy_film_profile_release(owner, nullptr));
            }
        };
        struct PrintDeleter {
            void operator()(FjPrintProfile* owner) const noexcept {
                report_cleanup(fj_legacy_print_profile_release(owner, nullptr));
            }
        };
        struct DensityDeleter {
            void operator()(FjPrintDensityCurves* owner) const noexcept {
                report_cleanup(fj_legacy_print_density_release(owner, nullptr));
            }
        };
        struct SpectraDeleter {
            void operator()(FjSpectraLut* owner) const noexcept {
                report_cleanup(fj_legacy_spectra_lut_release(owner, nullptr));
            }
        };
        struct MallettDeleter {
            void operator()(FjMallettBasis* owner) const noexcept {
                report_cleanup(fj_legacy_mallett_release(owner, nullptr));
            }
        };
        struct CmfDeleter {
            void operator()(FjCmf* owner) const noexcept {
                report_cleanup(fj_legacy_cmf_release(owner, nullptr));
            }
        };
        struct CsvDeleter {
            void operator()(FjCsvPairs* owner) const noexcept {
                report_cleanup(fj_legacy_csv_release(owner, nullptr));
            }
        };
        std::uint32_t csv_tag(CsvSource source) {
            switch (source) {
                case CsvSource::D65:
                    return FJ_CSV_D65;
                case CsvSource::D55:
                    return FJ_CSV_D55;
                case CsvSource::D50:
                    return FJ_CSV_D50;
                case CsvSource::T:
                    return FJ_CSV_T;
                case CsvSource::K75p:
                    return FJ_CSV_K75P;
                case CsvSource::Kg3:
                    return FJ_CSV_KG3;
                case CsvSource::Canon24F28Is:
                    return FJ_CSV_CANON_24_F28_IS;
            }
            fail({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "unsupported native CSV source");
        }
        [[noreturn]] void invalid_view() {
            fail({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalid Rust asset view shape, tag or token");
        }
        std::span<const float> float_span(FjFloatSpan view) {
            if (view.count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float) ||
                (view.count == 0 ? view.data != nullptr : !view.data || reinterpret_cast<std::uintptr_t>(view.data) % alignof(float) != 0)) {
                invalid_view();
            }
            return {view.data, view.count};
        }
        void copy_fixed(FjFloatSpan view, std::span<float> target) {
            const auto values = float_span(view);
            if (values.size() != target.size()) {
                invalid_view();
            }
            std::copy(values.begin(), values.end(), target.begin());
        }
        template <std::size_t Rows, std::size_t Channels>
        void copy_rows(FjFloatSpan view, std::array<std::array<float, Channels>, Rows>& target) {
            const auto values = float_span(view);
            if (values.size() != Rows * Channels) {
                invalid_view();
            }
            for (std::size_t row = 0; row < Rows; ++row) {
                std::copy_n(values.data() + row * Channels, Channels, target[row].begin());
            }
        }
        // FJ_TEMP_BRIDGE: spectral source conversion; remove S4.E.
        Spectral::ReconstructionLut copy_lut(
            const FjAssets* assets,
            FjStatus (*acquire)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*),
            const char* family) {
            std::array<char, 512> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            FjSpectraLut* acquired = nullptr;
            const FjStatus result = acquire(assets, &acquired, &error);
            const std::unique_ptr<FjSpectraLut, SpectraDeleter> owner(acquired);
            if (result.category != FJ_STATUS_SUCCESS) {
                fail(result, {diagnostic.data(), error.length});
            }
            FjSpectraLutView view{};
            const FjStatus viewed = fj_legacy_spectra_lut_view(owner.get(), &view, &error);
            if (viewed.category != FJ_STATUS_SUCCESS) {
                fail(viewed, {diagnostic.data(), error.length});
            }
#if defined(JUICER_SPECTRAL_TEST_HOOK)
            SpectralTest::lut_view(family, view);
#else
            (void)family;
#endif
            const auto samples = float_span(view.samples);
            constexpr std::size_t kCount = std::size_t{192} * 192u * 81u;
            if (samples.size() != kCount || view.asset_hash == 0) {
                invalid_view();
            }
            Spectral::ReconstructionLut copy;
            if (samples.size() > copy.data.max_size()) {
                invalid_view();
            }
            copy.data.resize(kCount);
#if defined(JUICER_SPECTRAL_TEST_HOOK)
            SpectralTest::copy_allocated(family, copy.data.capacity());
#endif
            std::copy(samples.begin(), samples.end(), copy.data.begin());
            copy.size = 192;
            copy.numSamples = 81;
            copy.assetHash = view.asset_hash;
#if defined(JUICER_SPECTRAL_TEST_HOOK)
            SpectralTest::copy_complete(family, copy.data.size(), copy.data.capacity());
#endif
            return copy;
        }

        Profiles::ProfileIlluminant copy_illuminant(FjIlluminantView view);

        template <typename Samples>
        void copy_tables(FjProfileTablesView view, Samples& data) {
            copy_rows(view.linear_sensitivity_rgb, data.linearSensitivity);
            copy_rows(view.channel_density_cmy, data.channelDensity);
            copy_fixed(view.base_density, data.baseDensity);
            const auto axis = float_span(view.log_exposure);
            const auto totals = float_span(view.density_curves_cmy);
            if (axis.empty() || axis.size() > std::numeric_limits<std::size_t>::max() / 3 || totals.size() != axis.size() * 3) {
                invalid_view();
            }
            if (axis.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || axis.size() > data.logExposure.max_size() || axis.size() > data.densityCurves.max_size()) {
                fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "profile exposure count exceeds native representation");
            }
            data.logExposure.assign(axis.begin(), axis.end());
            data.densityCurves.resize(axis.size());
            for (std::size_t row = 0; row < axis.size(); ++row) {
                std::copy_n(totals.data() + row * 3, 3, data.densityCurves[row].begin());
            }
        }

        struct CatalogDeleter {
            void operator()(FjCatalog* catalog) const noexcept {
                report_cleanup(fj_legacy_catalog_release(catalog, nullptr));
            }
        };

        std::string copy_text(FjStringView view) {
            if (view.count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) || (view.count == 0 ? view.data != nullptr : !view.data)) {
                invalid_view();
            }
            if (view.count > std::string().max_size()) {
                fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "profile text exceeds native representation");
            }
            return view.count == 0 ? std::string() : std::string(view.data, view.count);
        }

        Profiles::ProfileIlluminant copy_illuminant(FjIlluminantView view) {
            auto label = copy_text(view.label);
            switch (view.kind) {
                case FJ_ILLUMINANT_NAMED:
                    return {std::move(label), Profiles::NamedIlluminant{}};
                case FJ_ILLUMINANT_BLACKBODY:
                    return {std::move(label), Profiles::BlackbodyIlluminant{view.temperature_kelvin}};
                default:
                    invalid_view();
            }
        }

        struct FilmSource final {
            FjFilmProfile* owner = nullptr;
            std::shared_ptr<const Profiles::FilmProfile> profile;
            ~FilmSource() {
                report_cleanup(close());
            }
            FjStatus close(FjErrorBuffer* error = nullptr) noexcept {
                return owner ? fj_legacy_film_profile_release(std::exchange(owner, nullptr), error) : FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            }
        };

        void escaped_unit(unsigned unit, std::string& out, char prefix) {
            static constexpr char kHex[] = "0123456789abcdef";
            out += '\\';
            out += prefix;
            for (unsigned shift = prefix == 'u' ? 16u : 8u; shift != 0;) {
                shift -= 4;
                out += kHex[(unit >> shift) & 15];
            }
        }

        void append_utf8(std::string& out, std::uint32_t code) {
            if (code < 0x80) {
                out += static_cast<char>(code);
            } else if (code < 0x800) {
                out += static_cast<char>(0xc0 | (code >> 6));
                out += static_cast<char>(0x80 | (code & 63));
            } else if (code < 0x10000) {
                out += static_cast<char>(0xe0 | (code >> 12));
                out += static_cast<char>(0x80 | ((code >> 6) & 63));
                out += static_cast<char>(0x80 | (code & 63));
            } else {
                out += static_cast<char>(0xf0 | (code >> 18));
                out += static_cast<char>(0x80 | ((code >> 12) & 63));
                out += static_cast<char>(0x80 | ((code >> 6) & 63));
                out += static_cast<char>(0x80 | (code & 63));
            }
        }
    } // namespace

    NoiseSource::NoiseSource(FjNoise* owner, const FjStaticNoise& view) noexcept
        : _owner(owner), _view(view) {
    }

    NoiseSource::NoiseSource(NoiseSource&& source) noexcept
        : _owner(std::exchange(source._owner, nullptr)), _view(source._view) {
    }

    NoiseSource::~NoiseSource() {
        if (_owner) {
            NoiseDeleter{}(_owner);
        }
    }

    JuicerCuda::StaticNoiseInput NoiseSource::view() const& {
        return {{_view.stbn.data, _view.stbn.count}, {_view.wang_tiles.data, _view.wang_tiles.count}, {_view.wang_lut.data, _view.wang_lut.count}, _view.stbn_width, _view.stbn_height, _view.stbn_frames, _view.wang_width, _view.wang_height, static_cast<int>(_view.wang_tile_count), _view.wang_colors};
    }

    NoiseSource AssetBridge::noise() {
        std::array<char, 512> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjNoise* acquired = nullptr;
#if defined(JUICER_NOISE_TEST_HOOK)
        NoiseTest::observe(NoiseTest::Operation::Acquire);
#endif
        const auto result = fj_legacy_noise_acquire(_assets, &acquired, &error);
        std::unique_ptr<FjNoise, NoiseDeleter> owner(acquired);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, message.data());
        }
        FjStaticNoise view{};
#if defined(JUICER_NOISE_TEST_HOOK)
        NoiseTest::observe(NoiseTest::Operation::View);
#endif
        const auto viewed = fj_legacy_noise_view(owner.get(), &view, &error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            fail(viewed, message.data());
        }
#if defined(JUICER_NOISE_TEST_HOOK)
        NoiseTest::view(view);
#endif
        if (!owner || view.stbn_width != 512 || view.stbn_height != 512 || view.stbn_frames != 256 ||
            !view.stbn.data || view.stbn.count != std::size_t{512} * 512 * 256 ||
            view.wang_width != 256 || view.wang_height != 256 || view.wang_tile_count != 16 ||
            view.wang_tile_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) || view.wang_colors != 2 ||
            !view.wang_tiles.data || view.wang_tiles.count != std::size_t{256} * 256 * 16 ||
            !view.wang_lut.data || view.wang_lut.count != 16) {
            fail({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "invalid complete Rust noise view extent");
        }
        return NoiseSource(owner.release(), view);
    }

    NativePathArgument::NativePathArgument(const std::filesystem::path& path)
#if !defined(_WIN32)
        : _units(path.native())
#endif
    {
#if defined(_WIN32)
        _units.reserve(path.native().size());
        for (wchar_t unit : path.native()) {
            _units.push_back(static_cast<std::uint16_t>(unit));
        }
#endif
    }

    FjPathView NativePathArgument::view() const noexcept {
#if defined(_WIN32)
        return {_units.data(), _units.size(), FJ_PATH_WINDOWS_WIDE};
#else
        return {_units.data(), _units.size(), FJ_PATH_UNIX_BYTES};
#endif
    }

    std::filesystem::path copy_native_path(FjPathView view) {
        constexpr auto kLimit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        if (!view.data || view.count == 0) {
            invalid_path();
        }
#if defined(_WIN32)
        if (view.encoding != FJ_PATH_WINDOWS_WIDE || view.count > kLimit / sizeof(std::uint16_t) ||
            reinterpret_cast<std::uintptr_t>(view.data) % alignof(std::uint16_t) != 0) {
            invalid_path();
        }
        const auto* units = static_cast<const std::uint16_t*>(view.data);
        std::wstring native;
        native.reserve(view.count);
        for (std::size_t i = 0; i < view.count; ++i) {
            if (units[i] == 0) {
                invalid_path();
            }
            native.push_back(static_cast<wchar_t>(units[i]));
        }
        return std::filesystem::path(std::move(native));
#else
        if (view.encoding != FJ_PATH_UNIX_BYTES || view.count > kLimit) {
            invalid_path();
        }
        const std::string_view native(static_cast<const char*>(view.data), view.count);
        if (native.find('\0') != std::string_view::npos) {
            invalid_path();
        }
        return std::filesystem::path(native);
#endif
    }

    std::string path_diagnostic(const std::filesystem::path& path) {
        const auto& native = path.native();
        std::string out;
        bool escaped = false;
#if defined(_WIN32)
        for (std::size_t i = 0; i < native.size(); ++i) {
            const auto unit = static_cast<std::uint16_t>(native[i]);
            if (unit >= 0xd800 && unit <= 0xdbff && i + 1 < native.size() &&
                native[i + 1] >= 0xdc00 && native[i + 1] <= 0xdfff) {
                append_utf8(out, 0x10000u + ((unit - 0xd800u) << 10u) + (static_cast<std::uint16_t>(native[++i]) - 0xdc00u));
            } else if (unit >= 0xd800 && unit <= 0xdfff) {
                escaped_unit(unit, out, 'u');
                escaped = true;
            } else {
                if (unit == '\\') {
                    out += "\\\\";
                } else {
                    append_utf8(out, unit);
                }
            }
        }
#else
        for (std::size_t i = 0; i < native.size();) {
            const auto first = static_cast<unsigned char>(native[i]);
            const std::size_t length = first < 0x80 ? 1 : first >= 0xc2 && first <= 0xdf ? 2
                                                      : first >= 0xe0 && first <= 0xef   ? 3
                                                      : first >= 0xf0 && first <= 0xf4   ? 4
                                                                                         : 0;
            bool valid = length != 0 && length <= native.size() - i;
            std::uint32_t code = first & (length == 2 ? 31u : length == 3 ? 15u
                                                          : length == 4   ? 7u
                                                                          : 127u);
            for (std::size_t j = 1; valid && j < length; ++j) {
                const auto next = static_cast<unsigned char>(native[i + j]);
                valid = (next & 0xc0) == 0x80;
                code = (code << 6u) | (next & 63u);
            }
            valid = valid && (length < 2 || code >= (length == 2 ? 0x80u : length == 3 ? 0x800u
                                                                                       : 0x10000u)) &&
                    code <= 0x10ffff && (code < 0xd800 || code > 0xdfff);
            if (valid) {
                if (code == '\\') {
                    out += "\\\\";
                } else {
                    append_utf8(out, code);
                }
                i += length;
            } else {
                escaped_unit(first, out, 'x');
                escaped = true;
                ++i;
            }
        }
#endif
        if (escaped) {
            return out;
        }
        std::string readable;
        for (std::size_t i = 0; i < out.size(); ++i) {
            readable += out[i];
            if (out[i] == '\\') {
                ++i;
            }
        }
        return readable;
    }

    struct AssetBridge::ConversionSlots {
        std::mutex mutex;
        std::vector<std::string> filmKeys;
        std::vector<std::string> printKeys;
        std::vector<std::shared_ptr<FilmSource>> films;
        std::vector<std::shared_ptr<const PrintProfileSource>> prints;
    };

    PrintProfileSource::PrintProfileSource(FjPrintProfile* owner, std::shared_ptr<const Profiles::PrintProfile> profile)
        : _owner(owner), _profile(std::move(profile)) {}
    PrintProfileSource::~PrintProfileSource() {
        report_cleanup(close());
    }
    const std::shared_ptr<const Profiles::PrintProfile>& PrintProfileSource::profile() const noexcept {
        return _profile;
    }
    FjStatus PrintProfileSource::close(FjErrorBuffer* error) const noexcept {
        return _owner ? fj_legacy_print_profile_release(std::exchange(_owner, nullptr), error) : FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    }
    Profiles::PrintDensityCurves PrintProfileSource::sample_density_curves(double gamma) const {
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjPrintDensityCurves* acquired = nullptr;
        auto result = fj_legacy_print_profile_sample_density(_owner, gamma, &acquired, &error);
        std::unique_ptr<FjPrintDensityCurves, DensityDeleter> owner(acquired);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {message.data(), error.length});
        }
        FjPrintDensityView view{};
        result = fj_legacy_print_density_view(owner.get(), &view, &error);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {message.data(), error.length});
        }
        const auto totals = float_span(view.totals_cmy);
        if (totals.size() != _profile->data.logExposure.size() * 3 || view.hash == 0) {
            invalid_view();
        }
        Profiles::PrintDensityCurves curves;
        curves.totals.resize(_profile->data.logExposure.size());
        for (std::size_t row = 0; row < curves.totals.size(); ++row) {
            std::copy_n(totals.data() + row * 3, 3, curves.totals[row].begin());
        }
        curves.hash = view.hash;
        result = fj_legacy_print_density_release(owner.release(), &error);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {message.data(), error.length});
        }
        return curves;
    }

    std::shared_ptr<const Profiles::FilmProfile> AssetBridge::film(const std::string& key) {
        if (!_slots) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "film catalog unavailable");
        }
        const auto found = std::find(_slots->filmKeys.begin(), _slots->filmKeys.end(), key);
        if (found == _slots->filmKeys.end()) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "missing selected film profile key");
        }
        const auto index = static_cast<std::size_t>(found - _slots->filmKeys.begin());
        std::shared_ptr<FilmSource> winner;
        {
            std::lock_guard<std::mutex> lock(_slots->mutex);
            winner = _slots->films[index];
        }
        if (winner) {
            return winner->profile;
        }
        std::shared_ptr<FilmSource> candidate;
        try {
            std::array<char, 1024> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            FjFilmProfile* acquired = nullptr;
            auto result = fj_legacy_film_profile_acquire(_assets, {key.empty() ? nullptr : key.data(), key.size()}, &acquired, &error);
            std::unique_ptr<FjFilmProfile, FilmDeleter> owner(acquired);
            if (result.category != FJ_STATUS_SUCCESS) {
                fail(result, {message.data(), error.length});
            }
            FjFilmProfileView view{};
            result = fj_legacy_film_profile_view(owner.get(), &view, &error);
            if (result.category != FJ_STATUS_SUCCESS) {
                fail(result, {message.data(), error.length});
            }
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::film_view(view);
#endif
            if (!view.asset_token || view.support != FJ_PROFILE_SUPPORT_FILM || view.stage != FJ_PROFILE_STAGE_FILMING) {
                invalid_view();
            }
            auto profile = std::make_shared<Profiles::FilmProfile>();
            profile->info.stock = copy_text(view.stock);
            profile->info.referenceIlluminant = copy_illuminant(view.reference_illuminant);
            profile->info.viewingIlluminant = copy_illuminant(view.viewing_illuminant);
            switch (view.polarity) {
                case FJ_POLARITY_NEGATIVE:
                    profile->info.type = Spektrafilm::ProfilePolarity::Negative;
                    break;
                case FJ_POLARITY_POSITIVE:
                    profile->info.type = Spektrafilm::ProfilePolarity::Positive;
                    break;
                default:
                    invalid_view();
            }
            copy_tables(view.tables, profile->data);
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::after_tables();
#endif
            copy_fixed(view.wavelengths, profile->data.wavelengths);
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto values = float_span(view.density_curves_layers[layer][channel]);
                    auto& target = profile->data.densityCurvesLayers[layer][channel];
                    if (values.size() != profile->data.logExposure.size()) {
                        invalid_view();
                    }
                    if (values.size() > target.max_size()) {
                        fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "profile layer count exceeds native representation");
                    }
                    target.assign(values.begin(), values.end());
                }
            }
            auto& data = profile->data;
            if (view.hanatos_window.count != 0) {
                copy_fixed(view.hanatos_window, data.hanatos2025AdaptationWindowParams);
                data.hasHanatos2025AdaptationWindowParams = true;
            } else {
                (void)float_span(view.hanatos_window);
            }
            if (view.hanatos_surface_rgb.count != 0) {
                copy_rows(view.hanatos_surface_rgb, data.hanatos2025AdaptationSurfaceParams);
                data.hasHanatos2025AdaptationSurfaceParams = true;
            } else {
                (void)float_span(view.hanatos_surface_rgb);
            }
            auto& digest = profile->digest;
            std::copy_n(view.digest.gamma_samelayer_rgb, 3, digest.gammaSamelayerRgb.begin());
            std::copy_n(view.digest.gamma_interlayer_r_to_gb, 2, digest.gammaInterlayerRToGb.begin());
            std::copy_n(view.digest.gamma_interlayer_g_to_rb, 2, digest.gammaInterlayerGToRb.begin());
            std::copy_n(view.digest.gamma_interlayer_b_to_rg, 2, digest.gammaInterlayerBToRg.begin());
            std::copy_n(view.digest.halation_first_sigma_um, 3, digest.halationFirstSigmaUm.begin());
            std::copy_n(view.digest.halation_primary_amount, 3, digest.halationPrimaryAmount.begin());
            digest.hanatosSpectralGaussianBlurDefault = view.digest.hanatos_spectral_gaussian_blur_default;
            profile->assetVersionToken = view.asset_token;
            candidate = std::make_shared<FilmSource>();
            candidate->profile = std::move(profile);
            candidate->owner = owner.release();
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::before_publication();
#endif
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(_slots->mutex);
                winner = _slots->films[index];
            }
            if (winner) {
                return winner->profile;
            }
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(_slots->mutex);
            if (!_slots->films[index]) {
                _slots->films[index] = candidate;
            }
            winner = _slots->films[index];
        }
        return winner->profile;
    }

    std::shared_ptr<const PrintProfileSource> AssetBridge::print(const std::string& key) {
        if (!_slots) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "print catalog unavailable");
        }
        const auto found = std::find(_slots->printKeys.begin(), _slots->printKeys.end(), key);
        if (found == _slots->printKeys.end()) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "missing selected print profile key");
        }
        const auto index = static_cast<std::size_t>(found - _slots->printKeys.begin());
        std::shared_ptr<const PrintProfileSource> winner;
        {
            std::lock_guard<std::mutex> lock(_slots->mutex);
            winner = _slots->prints[index];
        }
        if (winner) {
            return winner;
        }
        std::shared_ptr<PrintProfileSource> candidate;
        try {
            std::array<char, 1024> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            FjPrintProfile* acquired = nullptr;
            auto result = fj_legacy_print_profile_acquire(_assets, {key.empty() ? nullptr : key.data(), key.size()}, &acquired, &error);
            std::unique_ptr<FjPrintProfile, PrintDeleter> owner(acquired);
            if (result.category != FJ_STATUS_SUCCESS) {
                fail(result, {message.data(), error.length});
            }
            FjPrintProfileView view{};
            result = fj_legacy_print_profile_view(owner.get(), &view, &error);
            if (result.category != FJ_STATUS_SUCCESS) {
                fail(result, {message.data(), error.length});
            }
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::print_view(view);
#endif
            if (!view.asset_token || view.stage != FJ_PROFILE_STAGE_PRINTING) {
                invalid_view();
            }
            auto profile = std::make_shared<Profiles::PrintProfile>();
            profile->info.stock = copy_text(view.stock);
            profile->info.viewingIlluminant = copy_illuminant(view.viewing_illuminant);
            copy_tables(view.tables, profile->data);
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::after_tables();
#endif
            profile->assetVersionToken = view.asset_token;
            candidate = std::shared_ptr<PrintProfileSource>(new PrintProfileSource(nullptr, std::move(profile)));
            candidate->_owner = owner.release();
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
            ProfileTest::before_publication();
#endif
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(_slots->mutex);
                winner = _slots->prints[index];
            }
            if (winner) {
                return winner;
            }
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(_slots->mutex);
            if (!_slots->prints[index]) {
                _slots->prints[index] = candidate;
            }
            winner = _slots->prints[index];
        }
        return winner;
    }

    FjStatus AssetBridge::release_cached_payloads(FjErrorBuffer* error) noexcept {
        FjStatus first{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        try {
            ConversionSlots* slots = nullptr;
            {
                std::lock_guard<std::mutex> lock(_slotsMutex);
                slots = _slots.get();
            }
            // Published slots remain fixed until terminal close excludes readers.
            if (slots) {
                std::vector<std::shared_ptr<FilmSource>> films(slots->films.size());
                std::vector<std::shared_ptr<const PrintProfileSource>> prints(slots->prints.size());
                {
                    std::lock_guard<std::mutex> lock(slots->mutex);
                    for (std::size_t i = 0; i < films.size(); ++i) {
                        films[i].swap(slots->films[i]);
                    }
                    for (std::size_t i = 0; i < prints.size(); ++i) {
                        prints[i].swap(slots->prints[i]);
                    }
                }
            }
        } catch (const std::bad_alloc&) {
            first = JuicerCuda::write_status({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "native profile cache detachment allocation failed", error);
        } catch (const std::exception& detail) {
            first = JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what(), error);
        } catch (...) {
            first = JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "native profile cache detachment failed", error);
        }
        const auto result = fj_legacy_assets_release_cached_payloads(_assets, first.category == FJ_STATUS_SUCCESS ? error : nullptr);
        return first.category == FJ_STATUS_SUCCESS ? result : first;
    }

    AssetBridge::AssetBridge(const std::filesystem::path& resourceRoot) {
        const NativePathArgument argument(resourceRoot);
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const auto result = fj_legacy_assets_create(argument.view(), &_assets, &error);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {message.data(), error.length});
        }
    }

    AssetBridge::~AssetBridge() {
        report_cleanup(close());
    }

    FjStatus AssetBridge::close(FjErrorBuffer* error) noexcept {
        FjStatus first{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        const auto collect = [&](FjStatus result) {
            if (first.category == FJ_STATUS_SUCCESS) {
                first = result;
            }
        };
        // Terminal close excludes all host builds, source leases and views.
        auto slots = std::move(_slots);
        if (slots) {
            for (const auto& film : slots->films) {
                if (film) {
                    collect(film->close(first.category == FJ_STATUS_SUCCESS ? error : nullptr));
                }
            }
            for (const auto& print : slots->prints) {
                if (print) {
                    collect(print->close(first.category == FJ_STATUS_SUCCESS ? error : nullptr));
                }
            }
        }
        slots.reset();
        if (_catalog) {
            collect(fj_legacy_catalog_release(std::exchange(_catalog, nullptr), first.category == FJ_STATUS_SUCCESS ? error : nullptr));
        }
        if (_assets) {
            collect(fj_legacy_assets_destroy(std::exchange(_assets, nullptr), first.category == FJ_STATUS_SUCCESS ? error : nullptr));
        }
        return first;
    }

    Spektrafilm::ProfileCatalog AssetBridge::load_catalog() {
        std::array<char, 4096> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjCatalog* acquired = nullptr;
        FjCatalogCounts counts{};
        const auto result = fj_legacy_catalog_acquire(_assets, &acquired, &counts, &error);
        std::unique_ptr<FjCatalog, CatalogDeleter> owner(acquired);
        Spektrafilm::ProfileCatalog candidate;
        if (result.category == FJ_STATUS_PREPARATION_FAILURE) {
#if defined(JUICER_CATALOG_TEST_HOOK)
            CatalogTest::before_source_diagnostic();
#endif
            candidate.failure.assign(message.data(), error.length);
            return candidate;
        }
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {message.data(), error.length});
        }
#if defined(JUICER_CATALOG_TEST_HOOK)
        CatalogTest::before_conversion(counts);
#endif
        if (counts.film_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            counts.print_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "catalog count exceeds OFX option index width");
        }
        const auto copy_entries = [&](std::uint32_t role, std::vector<Spektrafilm::ProfileCatalogEntry>& entries, std::size_t count) {
            entries.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                FjCatalogEntryView view{};
                const auto entryResult = fj_legacy_catalog_entry(owner.get(), role, i, &view, &error);
                if (entryResult.category != FJ_STATUS_SUCCESS) {
                    fail(entryResult, {message.data(), error.length});
                }
                Spektrafilm::ProfilePolarity polarity;
                switch (view.polarity) {
                    case FJ_POLARITY_NEGATIVE:
                        polarity = Spektrafilm::ProfilePolarity::Negative;
                        break;
                    case FJ_POLARITY_POSITIVE:
                        polarity = Spektrafilm::ProfilePolarity::Positive;
                        break;
                    default:
                        fail({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalid Rust catalog polarity");
                }
                entries.push_back({copy_text(view.key), copy_text(view.label), polarity});
            }
        };
        copy_entries(FJ_PROFILE_ROLE_FILM, candidate.filmProfiles, counts.film_count);
        copy_entries(FJ_PROFILE_ROLE_PRINT, candidate.printProfiles, counts.print_count);
        candidate.valid = true;
        candidate.defaultFilmPresent = true;
        candidate.defaultPrintPresent = true;
        auto slots = std::make_unique<ConversionSlots>();
        slots->filmKeys.reserve(candidate.filmProfiles.size());
        slots->printKeys.reserve(candidate.printProfiles.size());
        for (const auto& entry : candidate.filmProfiles) {
            slots->filmKeys.push_back(entry.key);
        }
        for (const auto& entry : candidate.printProfiles) {
            slots->printKeys.push_back(entry.key);
        }
        slots->films.resize(candidate.filmProfiles.size());
        slots->prints.resize(candidate.printProfiles.size());
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
        ProfileTest::before_catalog_publication();
#endif
        {
            std::lock_guard<std::mutex> lock(_slotsMutex);
            _slots.swap(slots);
        }
        _catalog = owner.release();
        return candidate;
    }
} // namespace JuicerAssets

namespace JuicerAssets {
    Spectral::ReconstructionLut AssetBridge::copy_hanatos_lut() {
        return copy_lut(_assets, fj_legacy_hanatos_acquire, "hanatos");
    }
    Spectral::ReconstructionLut AssetBridge::copy_arctic_lut() {
        return copy_lut(_assets, fj_legacy_arctic_acquire, "arctic");
    }
    Spectral::MallettBasis AssetBridge::copy_mallett_basis() {
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjMallettBasis* acquired = nullptr;
        const auto result = fj_legacy_mallett_acquire(_assets, &acquired, &error);
        const std::unique_ptr<FjMallettBasis, MallettDeleter> owner(acquired);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {diagnostic.data(), error.length});
        }
        FjFloatSpan view{};
        const auto viewed = fj_legacy_mallett_view(owner.get(), &view, &error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            fail(viewed, {diagnostic.data(), error.length});
        }
#if defined(JUICER_SPECTRAL_TEST_HOOK)
        SpectralTest::mallett_view(view);
#endif
        const auto samples = float_span(view);
        if (samples.size() != 243u) {
            invalid_view();
        }
        Spectral::MallettBasis copy;
        copy.data.resize(243u);
#if defined(JUICER_SPECTRAL_TEST_HOOK)
        SpectralTest::copy_allocated("mallett", copy.data.capacity());
#endif
        std::copy(samples.begin(), samples.end(), copy.data.begin());
        copy.rows = 81;
        copy.cols = 3;
#if defined(JUICER_SPECTRAL_TEST_HOOK)
        SpectralTest::copy_complete("mallett", copy.data.size(), copy.data.capacity());
#endif
        return copy;
    }
    Spectral::CMFTriplets AssetBridge::copy_cmf_triplets() {
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjCmf* acquired = nullptr;
        const auto result = fj_legacy_cmf_acquire(_assets, &acquired, &error);
        const std::unique_ptr<FjCmf, CmfDeleter> owner(acquired);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {diagnostic.data(), error.length});
        }
        FjFloatSpan view{};
        const auto viewed = fj_legacy_cmf_view(owner.get(), &view, &error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            fail(viewed, {diagnostic.data(), error.length});
        }
#if defined(JUICER_SPECTRAL_TEST_HOOK)
        SpectralTest::cmf_view(view);
#endif
        const auto samples = float_span(view);
        if (samples.size() % 4u != 0) {
            invalid_view();
        }
        const auto rows = samples.size() / 4u;
        Spectral::CMFTriplets copy;
        if (rows > copy.xbar.max_size()) {
            invalid_view();
        }
        for (auto* channel : {&copy.xbar, &copy.ybar, &copy.zbar}) {
            channel->resize(rows);
#if defined(JUICER_SPECTRAL_TEST_HOOK)
            SpectralTest::copy_allocated("cmf", channel->capacity());
#endif
        }
        for (std::size_t row = 0; row < rows; ++row) {
            const auto base = row * 4u;
            copy.xbar[row] = {samples[base], samples[base + 1u]};
            copy.ybar[row] = {samples[base], samples[base + 2u]};
            copy.zbar[row] = {samples[base], samples[base + 3u]};
        }
#if defined(JUICER_SPECTRAL_TEST_HOOK)
        SpectralTest::copy_complete("cmf", rows * 3u, copy.xbar.capacity() + copy.ybar.capacity() + copy.zbar.capacity());
#endif
        return copy;
    }

    std::vector<std::pair<float, float>> AssetBridge::copy_csv_pairs(CsvSource source) {
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::before_csv_acquisition(source);
#endif
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjCsvPairs* acquired = nullptr;
        const auto result = fj_legacy_csv_acquire(_assets, csv_tag(source), &acquired, &error);
        const std::unique_ptr<FjCsvPairs, CsvDeleter> owner(acquired);
        if (result.category != FJ_STATUS_SUCCESS) {
            fail(result, {diagnostic.data(), error.length});
        }
        FjFloatSpan view{};
        const auto viewed = fj_legacy_csv_view(owner.get(), &view, &error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            fail(viewed, {diagnostic.data(), error.length});
        }
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::csv_view(source, view);
#endif
        const auto samples = float_span(view);
        if (samples.size() % 2u != 0) {
            invalid_view();
        }
        const auto rows = samples.size() / 2u;
        std::vector<std::pair<float, float>> copy;
        if (rows > copy.max_size()) {
            invalid_view();
        }
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::before_csv_copy(source, rows);
#endif
        copy.resize(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            copy[row] = {samples[row * 2u], samples[row * 2u + 1u]};
        }
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::after_csv_copy(source, rows, copy.capacity());
#endif
        return copy;
    }

    NeutralPrintCalibrationResult AssetBridge::neutral_print_calibration(const std::string& printStock,
                                                                         const std::string& illuminant,
                                                                         const std::string& filmStock) {
        std::array<char, 512> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjNeutralCalibrationResult raw{};
        const auto text = [](const std::string& value) -> FjStringView {
            return {value.empty() ? nullptr : value.data(), value.size()};
        };
        const auto status = fj_legacy_neutral_calibration_lookup(_assets, text(printStock), text(illuminant), text(filmStock), &raw, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            fail(status, {diagnostic.data(), error.length});
        }
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::calibration_result(raw);
#endif
        NeutralPrintCalibrationResult result;
        if (raw.outcome == FJ_CALIBRATION_FOUND) {
            if (raw.field != FJ_CALIBRATION_FIELD_NONE) {
                invalid_view();
            }
            result.status = NeutralPrintCalibrationStatus::Found;
            std::copy(std::begin(raw.cmy_cc), std::end(raw.cmy_cc), result.cmyCc.begin());
            return result;
        }
        if (!std::all_of(std::begin(raw.cmy_cc), std::end(raw.cmy_cc), [](float cc) {
                return cc == 0.0f;
            })) {
            invalid_view();
        }
        if (raw.outcome == FJ_CALIBRATION_MISSING_FILE || raw.outcome == FJ_CALIBRATION_MISSING_ENTRY) {
            if (raw.field != FJ_CALIBRATION_FIELD_NONE) {
                invalid_view();
            }
            result.status = raw.outcome == FJ_CALIBRATION_MISSING_FILE ? NeutralPrintCalibrationStatus::MissingFile : NeutralPrintCalibrationStatus::MissingEntry;
            return result;
        }
        if (raw.outcome != FJ_CALIBRATION_MALFORMED) {
            invalid_view();
        }
        const char* field = nullptr;
        switch (raw.field) {
            case FJ_CALIBRATION_FIELD_RESOURCE_READ:
                field = "resource_read";
                break;
            case FJ_CALIBRATION_FIELD_ROOT:
                field = "root";
                break;
            case FJ_CALIBRATION_FIELD_PRINT_PROFILE:
                field = "print_profile";
                break;
            case FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT:
                field = "print_illuminant";
                break;
            case FJ_CALIBRATION_FIELD_CMY_CC:
                field = "cmy_cc";
                break;
            default:
                invalid_view();
        }
        result.status = NeutralPrintCalibrationStatus::Malformed;
        result.diagnostic = std::string("MalformedNeutralPrintCalibration phase=4A field=") + field;
        return result;
    }
} // namespace JuicerAssets
