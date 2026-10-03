#include "render_assertions.h"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace RenderAssertions {
    void compare_pixels(const std::string& name, std::span<const float> actual, std::span<const float> expected) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(name + ": pixel count mismatch");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                throw std::runtime_error(name + ": non-finite pixel " + std::to_string(i));
            }
            const float allowed = 2e-4f + 3e-4f * std::abs(expected[i]);
            if (std::abs(actual[i] - expected[i]) > allowed) {
                throw std::runtime_error(name + ": pixel " + std::to_string(i) + " differs: " +
                                         std::to_string(actual[i]) + " versus " + std::to_string(expected[i]));
            }
        }
    }

    void require_same_bits(const std::string& name, float actual, float expected) {
        if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
            throw std::runtime_error(name + ": float bits differ");
        }
    }

    void compare_bits(const std::string& name, std::span<const float> actual, std::span<const float> expected) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(name + ": pixel count mismatch");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require_same_bits(name + ": pixel " + std::to_string(i), actual[i], expected[i]);
        }
    }

    bool identities_match(const std::array<std::uint64_t, 4>& actual, const std::array<std::uint64_t, 4>& expected) {
        return actual == expected;
    }
} // namespace RenderAssertions
