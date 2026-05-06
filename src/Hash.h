// Hash.h
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <cmath>
#include <sstream>

#include "Logging.h"

namespace Hash {

    inline constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    inline constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

    inline void hash_bytes_update(uint64_t& h, const void* data, std::size_t numBytes) {
        if (numBytes == 0) {
            return;
        }
        if (!data) {
            JTRACE("HASH", "FATAL: hash_bytes_update called with null data");
            h = 0ULL;
            return;
        }
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < numBytes; ++i) {
            h ^= static_cast<uint64_t>(bytes[i]);
            h *= kFnvPrime;
        }
    }

    inline uint64_t hash_bytes(const void* data, std::size_t numBytes) {
        if (numBytes == 0) {
            return kFnvOffset;
        }
        if (!data) {
            JTRACE("HASH", "FATAL: hash_bytes called with null data");
            return 0ULL;
        }
        uint64_t h = kFnvOffset;
        hash_bytes_update(h, data, numBytes);
        return h;
    }

    inline uint64_t hash_uint64_values(std::initializer_list<std::uint64_t> values) {
        uint64_t h = kFnvOffset;
        for (std::uint64_t value : values) {
            hash_bytes_update(h, &value, sizeof(value));
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

    struct FloatSpanHash {
        std::uint64_t valueHash = kFnvOffset;
        std::uint64_t nanMaskHash = kFnvOffset;
    };

    inline FloatSpanHash hash_float_span_with_nan_mask(const float* data, std::size_t count) {
        FloatSpanHash out{};
        if (count == 0) {
            return out;
        }
        if (!data) {
            JTRACE("HASH", "FATAL: hash_float_span_with_nan_mask called with null data");
            out.valueHash = 0ULL;
            out.nanMaskHash = 0ULL;
            return out;
        }

        uint64_t valueHash = kFnvOffset;
        uint64_t maskHash = kFnvOffset;
        std::uint64_t word = 0;
        unsigned bit = 0;

        for (std::size_t i = 0; i < count; ++i) {
            float v = data[i];
            const bool isNan = std::isnan(v);
            if (isNan) {
                word |= (1ULL << bit);
                v = 0.0f;
            }
            if (v == 0.0f) {
                v = 0.0f; // canonicalize -0.0f to +0.0f
            }

            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            hash_bytes_update(valueHash, &bits, sizeof(bits));

            ++bit;
            if (bit == 64) {
                hash_bytes_update(maskHash, &word, sizeof(word));
                word = 0;
                bit = 0;
            }
        }
        if (bit != 0) {
            hash_bytes_update(maskHash, &word, sizeof(word));
        }

        out.valueHash = valueHash;
        out.nanMaskHash = maskHash;
        return out;
    }

    template <std::size_t N>
    inline uint64_t hash_float_span(const std::array<float, N>& values) {
        return hash_float_span(values.data(), values.size());
    }

} // namespace Hash
