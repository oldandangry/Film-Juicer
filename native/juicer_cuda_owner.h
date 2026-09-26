#pragma once

#include <string_view>

#include "juicer_cuda_api.h"

namespace JuicerCuda {

    // Temporary C++ runtime holder; replaced by the Rust owner at its cutover.
    // Host load/unload serializes lifetime against all borrowed render calls.
    class Owner final {
    public:
        Owner() noexcept = default;
        ~Owner();

        Owner(const Owner&) = delete;
        Owner& operator=(const Owner&) = delete;

        void create(std::string_view dataDirectory);
        // S2.D replaces the bool bridge with the typed C terminal operations.
        // Consumes the handle even on failure; destruction never retries it.
        bool close() noexcept;

    private:
        FjCuda* _cuda = nullptr;
    };

} // namespace JuicerCuda
