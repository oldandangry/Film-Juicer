#include "allocation_fault.h"
#include <cstdlib>
#include <new>
#include <optional>
#include <utility>

namespace {
    thread_local std::optional<std::size_t> allocationsBeforeFault;
    thread_local bool faultDelivered = false;
} // namespace

namespace JuicerTest {
    void fail_allocation_after(std::size_t successfulAllocations) noexcept {
        allocationsBeforeFault = successfulAllocations;
        faultDelivered = false;
    }

    bool take_allocation_fault() noexcept {
        allocationsBeforeFault.reset();
        return std::exchange(faultDelivered, false);
    }
} // namespace JuicerTest

void* operator new(std::size_t size) {
    if (allocationsBeforeFault && (*allocationsBeforeFault)-- == 0) {
        allocationsBeforeFault.reset();
        faultDelivered = true;
        throw std::bad_alloc{};
    }
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        return pointer;
    }
    throw std::bad_alloc{};
}

void operator delete(void* block) noexcept {
    std::free(block);
}

void operator delete(void* block, std::size_t) noexcept {
    ::operator delete(block);
}
