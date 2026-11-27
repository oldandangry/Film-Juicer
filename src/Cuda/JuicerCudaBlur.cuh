#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace JuicerCuda {
    static __device__ __forceinline__ int reflect_index_half_sample_wide_device(std::int64_t idx, int size) {
        if (size <= 1) {
            return 0;
        }
        while (idx < 0 || idx >= static_cast<std::int64_t>(size)) {
            if (idx < 0) {
                idx = -idx - 1;
            } else {
                idx = 2 * static_cast<std::int64_t>(size) - idx - 1;
            }
        }
        return static_cast<int>(idx);
    }
} // namespace JuicerCuda
