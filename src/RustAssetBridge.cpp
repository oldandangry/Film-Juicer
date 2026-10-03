#include "RustAssetBridge.h"

#include <array>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaExecutor.h"

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

        struct CatalogDeleter {
            void operator()(FjCatalog* catalog) const noexcept {
                (void)fj_legacy_catalog_release(catalog, nullptr);
            }
        };

        std::string copy_text(FjStringView view) {
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
        (void)close();
    }

    FjStatus AssetBridge::close(FjErrorBuffer* error) noexcept {
        const auto catalog = _catalog ? fj_legacy_catalog_release(std::exchange(_catalog, nullptr), error)
                                      : FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        const auto assets = _assets ? fj_legacy_assets_destroy(std::exchange(_assets, nullptr), catalog.category == FJ_STATUS_SUCCESS ? error : nullptr)
                                    : FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        return catalog.category == FJ_STATUS_SUCCESS ? assets : catalog;
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
                entries.push_back({copy_text(view.key), copy_text(view.label), copy_native_path(view.source_path), polarity});
            }
        };
        copy_entries(FJ_PROFILE_ROLE_FILM, candidate.filmProfiles, counts.film_count);
        copy_entries(FJ_PROFILE_ROLE_PRINT, candidate.printProfiles, counts.print_count);
        candidate.valid = true;
        candidate.defaultFilmPresent = true;
        candidate.defaultPrintPresent = true;
        _catalog = owner.release();
        return candidate;
    }
} // namespace JuicerAssets
