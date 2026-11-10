#include "spectral_allocation_probe.h"

#include <cstdlib>
#include <new>

namespace {
    thread_local unsigned rejectNativeAllocation = 0;
    thread_local std::size_t rejectedNativeBytes = 0;
} // namespace
// Fixture-only rejection of real native projection/cache allocations.
// No product allocator or policy seam is added.
// CRT replacement declarations use platform-specific parameter spellings.
// Keep this exception on the fixture's three replacement operators only.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t bytes) {
    if (rejectNativeAllocation != 0 && --rejectNativeAllocation == 0) {
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
    void arm(unsigned allocation) {
        rejectedNativeBytes = 0;
        rejectNativeAllocation = allocation;
    }
    void clear() {
        rejectNativeAllocation = 0;
    }
    std::size_t bytes() {
        return rejectedNativeBytes;
    }
} // namespace SpectralAllocationProbe
