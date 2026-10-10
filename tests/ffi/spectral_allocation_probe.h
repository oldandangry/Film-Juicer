#pragma once

#include <cstddef>

// Fixture-only interception of the actual native vector allocation. Separate
// compilation keeps replacement new/delete outside allocator consumers.
namespace SpectralAllocationProbe {
    void arm();
    void clear();
    std::size_t bytes();
} // namespace SpectralAllocationProbe
