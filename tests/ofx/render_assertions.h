#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace RenderAssertions {
    void compare_pixels(const std::string& name, std::span<const float> actual, std::span<const float> expected);
    void require_same_bits(const std::string& name, float actual, float expected);
    void compare_bits(const std::string& name, std::span<const float> actual, std::span<const float> expected);
    bool identities_match(const std::array<std::uint64_t, 4>& actual, const std::array<std::uint64_t, 4>& expected);
} // namespace RenderAssertions
