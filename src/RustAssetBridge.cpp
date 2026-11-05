#include "RustAssetBridge.h"

#include <algorithm>
#include <array>
#include <span>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "Logging.h"

namespace JuicerAssets {
    FjAssets* AssetBridge::borrow_assets() const noexcept {
        return _assets;
    }

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
        struct HullDeleter {
            void operator()(FjInputHullOwner* owner) const noexcept {
                report_cleanup(fj_legacy_input_hull_release(owner, nullptr));
            }
        };
        struct BoundaryDeleter {
            void operator()(FjOutputBoundaryOwner* owner) const noexcept {
                report_cleanup(fj_legacy_output_boundary_release(owner, nullptr));
            }
        };
        [[noreturn]] void invalid_view() {
            fail({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalid Rust asset view shape, tag or token");
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

    FjStatus AssetBridge::film_defaults(
        const std::string& key,
        FjFilmDigest& out,
        FjErrorBuffer& error) {
        out = FjFilmDigest{};

        FjFilmProfile* owner = nullptr;
        FjStatus status = fj_legacy_film_profile_acquire(
            _assets,
            {key.empty() ? nullptr : key.data(), key.size()},
            &owner,
            &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            if (owner) {
                FjErrorBuffer ignored{nullptr, 0, 0};
                (void)fj_legacy_film_profile_release(owner, &ignored);
            }
            return status;
        }
        if (!owner) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }

        FjFilmProfileView view{};
        status = fj_legacy_film_profile_view(owner, &view, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            FjErrorBuffer ignored{nullptr, 0, 0};
            (void)fj_legacy_film_profile_release(owner, &ignored);
            return status;
        }

        const FjFilmDigest defaults = view.digest;
        status = fj_legacy_film_profile_release(owner, &error);
        if (status.category == FJ_STATUS_SUCCESS) {
            out = defaults;
        }
        return status;
    }

    FjStatus AssetBridge::release_cached_payloads(FjErrorBuffer* error) noexcept {
        // Rust owns the derived and source caches; preserve their cleanup order.
        FjStatus first = fj_legacy_assets_release_derived_payloads(_assets, error);
        const auto sources = fj_legacy_assets_release_source_payloads(_assets, first.category == FJ_STATUS_SUCCESS ? error : nullptr);
        if (first.category == FJ_STATUS_SUCCESS)
            first = sources;
        return first;
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
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
        ProfileTest::before_catalog_publication();
#endif
        _catalog = owner.release();
        return candidate;
    }
} // namespace JuicerAssets

namespace JuicerAssets {
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

namespace JuicerAssets {
    std::shared_ptr<const Gamut::InputHullHold> AssetBridge::input_hull() {
        FjInputHullOwner* raw = nullptr;
        std::array<char, 1024> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const auto result = fj_legacy_assets_input_hull_acquire(_assets, &raw, &error);
        if (result.category == FJ_STATUS_PREPARATION_FAILURE)
            return {};
        if (result.category != FJ_STATUS_SUCCESS)
            fail(result, "Input hull preparation: " + std::string(bytes.data(), error.length));
        std::unique_ptr<FjInputHullOwner, HullDeleter> hold(raw);
        FjInputHullView view{};
        const auto projected = fj_legacy_input_hull_view(raw, &view, &error);
        if (projected.category != FJ_STATUS_SUCCESS)
            fail(projected, std::string_view(bytes.data(), error.length));
        return std::shared_ptr<const Gamut::InputHullHold>(new Gamut::InputHullHold(hold.release(), view));
    }

    std::shared_ptr<const Gamut::OutputBoundaryHold> AssetBridge::output_boundary(const Gamut::OutputGamutTransform& transform, std::string& diagnostic) {
        diagnostic.clear();
        // valid is the accepted transform's native transport marker. No cache
        // selection/generation or table derivation takes place on this side.
        const int index = OutputEncoding::toIndex(transform.outputColorSpace);
        if (!transform.valid || transform.hash == 0 || transform.contractHash == 0 || index < 0 || index >= static_cast<int>(OutputEncoding::kColorSpaceCount)) {
            diagnostic = "ResourceDescriptorMismatch component=output_boundary_cache field=transform";
            return {};
        }
        FjOutputBoundarySource source{};
        source.output_space = static_cast<std::uint32_t>(transform.outputColorSpace);
        std::copy(transform.d65XyzToNativeRgb.begin(), transform.d65XyzToNativeRgb.end(), source.d65_xyz_to_native_rgb);
        source.transform_hash = transform.hash;
        FjOutputBoundaryOwner* raw = nullptr;
        std::array<char, 1024> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        const auto result = fj_legacy_assets_output_boundary_acquire(_assets, &source, transform.contractHash, &raw, &error);
        if (result.category == FJ_STATUS_PREPARATION_FAILURE || result.category == FJ_STATUS_UNSUPPORTED_INPUT) {
            diagnostic.assign(bytes.data(), error.length);
            return {};
        }
        if (result.category != FJ_STATUS_SUCCESS)
            fail(result, "Output boundary: " + std::string(bytes.data(), error.length));
        std::unique_ptr<FjOutputBoundaryOwner, BoundaryDeleter> hold(raw);
        FjOutputBoundaryView view{};
        const auto projected = fj_legacy_output_boundary_view(raw, &view, &error);
        if (projected.category != FJ_STATUS_SUCCESS)
            fail(projected, std::string_view(bytes.data(), error.length));
        return std::shared_ptr<const Gamut::OutputBoundaryHold>(new Gamut::OutputBoundaryHold(hold.release(), view));
    }
} // namespace JuicerAssets

namespace Gamut {
    InputHullHold::InputHullHold(FjInputHullOwner* owner, const FjInputHullView& view) noexcept
        : xy(std::span<const float>(view.xy.data, view.xy.count)), center{view.center_xy[0], view.center_xy[1]}, hash(view.hash), entryIdentity(view.entry_identity), _owner(owner) {}
    InputHullHold::~InputHullHold() {
        (void)fj_legacy_input_hull_release(_owner, nullptr);
    }
    OutputBoundaryHold::OutputBoundaryHold(FjOutputBoundaryOwner* owner, const FjOutputBoundaryView& view) noexcept
        : cmax(view.cmax.data, view.cmax.count), transformHash(view.transform_hash), contractHash(view.contract_hash), hash(view.hash), entryIdentity(view.entry_identity), _owner(owner) {}
    OutputBoundaryHold::~OutputBoundaryHold() {
        (void)fj_legacy_output_boundary_release(_owner, nullptr);
    }
} // namespace Gamut
