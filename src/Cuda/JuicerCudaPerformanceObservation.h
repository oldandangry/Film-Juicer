#pragma once

#include <cstdint>

namespace JuicerCuda::PerformanceObservation {
    struct LedgerSnapshot {
        std::uint64_t reserved_bytes;
        std::uint64_t committed_bytes;
        std::uint64_t retiring_bytes;
        std::uint64_t records;
    };
    struct UploadPoolSnapshot {
        std::uint64_t bytes[4];
        std::uint64_t counts[4];
    };
    void context(int device, const void* opaque, std::uint64_t epoch) noexcept;
    void ledger(int device, const LedgerSnapshot& snapshot) noexcept;
    bool upload_snapshot(UploadPoolSnapshot& snapshot) noexcept;
    void upload_allocate(std::uint64_t bytes) noexcept;
    void upload_free(std::uint64_t bytes) noexcept;
} // namespace JuicerCuda::PerformanceObservation
