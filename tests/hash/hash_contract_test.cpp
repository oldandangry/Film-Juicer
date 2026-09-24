#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "Hash.h"

TEST(HashContract, ByteAndWordOrderRetainWraparound) {
    constexpr std::array<std::uint8_t, 4> bytes{{0, 1, 0xfe, 0xff}};
    EXPECT_EQ(Hash::hash_bytes(bytes.data(), bytes.size()), 0x4651de7f9a7611e9ULL);
    EXPECT_EQ(
        Hash::hash_uint64_values({0x0123456789abcdefULL, 0xffffffffffffffffULL}),
        0x000c8698d4e9656dULL);
}

TEST(HashContract, FloatSpanCanonicalizesSignedZero) {
    constexpr std::array<float, 2> signs{{0.0f, -0.0f}};
    constexpr std::array<float, 2> positives{{0.0f, 0.0f}};
    EXPECT_EQ(Hash::hash_float_span(signs), 0xa8c7f832281a39c5ULL);
    EXPECT_EQ(Hash::hash_float_span(signs), Hash::hash_float_span(positives));
}

TEST(HashContract, NanMaskRetainsTheSixtyFourSampleBoundary) {
    std::array<float, 65> samples{};
    samples[63] = std::numeric_limits<float>::quiet_NaN();
    const Hash::FloatSpanHash firstWord =
        Hash::hash_float_span_with_nan_mask(samples.data(), 64);
    EXPECT_EQ(firstWord.valueHash, 0xd80ac658736bb725ULL);
    EXPECT_EQ(firstWord.nanMaskHash, 0xa8c7783228196045ULL);

    samples[64] = std::numeric_limits<float>::quiet_NaN();
    const Hash::FloatSpanHash secondWord =
        Hash::hash_float_span_with_nan_mask(samples.data(), samples.size());
    EXPECT_EQ(secondWord.valueHash, 0xde65f6d7d32ae7f5ULL);
    EXPECT_EQ(secondWord.nanMaskHash, 0xb3faa9e6089510c4ULL);
}
