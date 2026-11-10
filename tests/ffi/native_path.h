#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "juicer_cuda_api.h"

namespace JuicerAssets {

    // Immediate foreign argument: Unix bytes borrow the path; Windows UTF-16
    // units are copied without aliasing wchar_t and uint16_t storage.
    class NativePathArgument final {
    public:
        explicit NativePathArgument(const std::filesystem::path& path);
        FjPathView view() const noexcept;

    private:
#if defined(_WIN32)
        std::vector<std::uint16_t> _units;
#else
        const std::string& _units;
#endif
    };

} // namespace JuicerAssets
