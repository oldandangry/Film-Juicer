#include "spectral_allocation_probe.h"

#include <cstdlib>
#include <new>

namespace {
    thread_local bool rejectNativeAllocation = false;
    thread_local std::size_t rejectedNativeBytes = 0;
} // namespace
// Fixture-only rejection of the real native vector allocation, after Rust's
// fixed-array result returns. No product allocator or policy seam is added.
// CRT replacement declarations use platform-specific parameter spellings.
// Keep this exception on the fixture's three replacement operators only.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t bytes) {
    if (rejectNativeAllocation) {
        rejectNativeAllocation = false;
        rejectedNativeBytes = bytes;
        throw std::bad_alloc{};
    }
    if (void* storage = std::malloc(bytes == 0 ? 1 : bytes))
        return storage;
    throw std::bad_alloc{};
}
void operator delete(void* storage) noexcept {
    std::free(storage);
}
void operator delete(void* storage, std::size_t) noexcept {
    std::free(storage);
}

// NOLINTEND(readability-inconsistent-declaration-parameter-name)

namespace SpectralAllocationProbe {
    void arm() {
        rejectedNativeBytes = 0;
        rejectNativeAllocation = true;
    }
    void clear() {
        rejectNativeAllocation = false;
    }
    std::size_t bytes() {
        return rejectedNativeBytes;
    }
} // namespace SpectralAllocationProbe
