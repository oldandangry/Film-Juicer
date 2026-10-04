#include "render_assertions.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(RenderAssertions, RejectsNumericalAndChannelRegressions) {
    const std::array<float, 3> expected{0.1f, 0.3f, 0.7f};
    EXPECT_NO_THROW(RenderAssertions::compare_pixels("unchanged", expected, expected));
    auto actual = expected;
    actual[1] += 0.001f;
    EXPECT_THROW(RenderAssertions::compare_pixels("one corrupted pixel", actual, expected), std::runtime_error);
    EXPECT_THROW(RenderAssertions::compare_pixels("channel order", std::array<float, 3>{0.7f, 0.3f, 0.1f}, expected), std::runtime_error);
    EXPECT_THROW(RenderAssertions::compare_pixels("truncated", std::span(actual).first(2), expected), std::runtime_error);
    actual = expected;
    actual[1] += 0.0001f;
    EXPECT_NO_THROW(RenderAssertions::compare_pixels("inside retained bound", actual, expected));
}

TEST(RenderAssertions, RejectsNonFiniteActualAndExpected) {
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        const std::array<float, 1> finite{0.3f};
        const std::array<float, 1> corrupt{invalid};
        EXPECT_THROW(RenderAssertions::compare_pixels("actual", corrupt, finite), std::runtime_error);
        EXPECT_THROW(RenderAssertions::compare_pixels("expected", finite, corrupt), std::runtime_error);
        EXPECT_THROW(RenderAssertions::compare_pixels("both", corrupt, corrupt), std::runtime_error);
    }
}

TEST(RenderAssertions, RejectsAlphaPaddingAndPathBitChanges) {
    EXPECT_NO_THROW(RenderAssertions::require_same_bits("alpha", 0.25f, 0.25f));
    EXPECT_THROW(RenderAssertions::require_same_bits("alpha", std::nextafter(0.25f, 1.0f), 0.25f), std::runtime_error);
    EXPECT_THROW(RenderAssertions::require_same_bits("padding", -7.0f, -8.0f), std::runtime_error);
    EXPECT_THROW(RenderAssertions::require_same_bits("signed zero", -0.0f, 0.0f), std::runtime_error);
    const std::array<float, 1> expected{0.3f};
    const std::array<float, 1> corrupt{std::nextafter(0.3f, 1.0f)};
    EXPECT_THROW(RenderAssertions::compare_bits("path", corrupt, expected), std::runtime_error);
}

TEST(RenderAssertions, RejectsEveryHashAndSeedLane) {
    const std::array<std::uint64_t, 4> expected{0x1298, 0x7846, 0x5123, 0x9367};
    EXPECT_TRUE(RenderAssertions::identities_match(expected, expected));
    for (std::size_t lane = 0; lane < expected.size(); ++lane) {
        auto corrupt = expected;
        corrupt[lane] ^= 1u;
        EXPECT_FALSE(RenderAssertions::identities_match(corrupt, expected));
    }
}
