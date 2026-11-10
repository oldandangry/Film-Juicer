#include "native_path.h"

namespace JuicerAssets {

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

} // namespace JuicerAssets
