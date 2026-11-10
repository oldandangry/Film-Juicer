#pragma once

#include <cstdint>
#include <string>

#include "GamutCompression.h"

namespace Spectral {
    struct Curve;
}
namespace Gamut {
    struct OutputBoundaryTable {
        std::array<float, kOutputBoundaryValueCount> cmax{};
        std::uint64_t transformHash = 0;
        std::uint64_t contractHash = 0;
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };

    struct InputCompressionHull {
        std::array<std::array<float, 2>, kInputHullDirectionCount + 1> xy{};
        std::array<float, 2> center{};
        std::uint64_t hash = 0;
        bool valid = false;
        std::string diagnostic;
    };
} // namespace Gamut

namespace OutputEncoding {
    enum class ColorSpace : std::uint8_t;
}

// Sources live through synchronous return. This binding owns no math or cache.
namespace JuicerTestFixture::Gamut {
    bool output_transform(OutputEncoding::ColorSpace space, ::Gamut::OutputGamutTransform& out, std::string& diagnostic);
    bool output_boundary(const ::Gamut::OutputGamutTransform& transform, ::Gamut::OutputBoundaryTable& out);
    bool input_hull(const Spectral::Curve& xBar,
                    const Spectral::Curve& yBar,
                    const Spectral::Curve& zBar,
                    const Spectral::Curve& d65,
                    ::Gamut::InputCompressionHull& out);
} // namespace JuicerTestFixture::Gamut
