// Hash.h
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <cmath>
#include <sstream>

#include "Logging.h"

namespace Hash {

    inline constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    inline constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

    inline uint64_t hash_bytes(const void* data, std::size_t numBytes) {
        if (numBytes == 0) {
            return kFnvOffset;
        }
        if (!data) {
            JTRACE("HASH", "FATAL: hash_bytes called with null data");
            return 0ULL;
        }
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        uint64_t h = kFnvOffset;
        for (std::size_t i = 0; i < numBytes; ++i) {
            h ^= static_cast<uint64_t>(bytes[i]);
            h *= kFnvPrime;
        }
        return h;
    }

    inline uint64_t hash_float_span(const float* data, std::size_t count) {
        if (count == 0) {
            return kFnvOffset;
        }
        if (!data) {
            JTRACE("HASH", "FATAL: hash_float_span called with null data");
            return 0ULL;
        }

        uint64_t h = kFnvOffset;
        for (std::size_t i = 0; i < count; ++i) {
            float v = data[i];
            if (!std::isfinite(v)) {
                std::ostringstream oss;
                oss << "FATAL: hash_float_span encountered non-finite sample at index " << i;
                JTRACE("HASH", oss.str());
                return 0ULL;
            }
            if (v == 0.0f) {
                v = 0.0f; // canonicalize -0.0f to +0.0f
            }
            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&bits);
            for (std::size_t b = 0; b < sizeof(bits); ++b) {
                h ^= static_cast<uint64_t>(bytes[b]);
                h *= kFnvPrime;
            }
        }
        return h;
    }

    template <std::size_t N>
    inline uint64_t hash_float_span(const std::array<float, N>& values) {
        return hash_float_span(values.data(), values.size());
    }

} // namespace Hash

