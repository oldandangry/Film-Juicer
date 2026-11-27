#pragma once

#include <cstddef>

namespace JuicerTest {
    void fail_allocation_after(std::size_t successfulAllocations) noexcept;
    bool take_allocation_fault() noexcept;
} // namespace JuicerTest
