#pragma once

#include <cstdint>
#include <string>

namespace Spectral {
    struct Curve;
}
namespace Gamut {
    struct InputCompressionHull;
    struct OutputGamutTransform;
    struct OutputBoundaryTable;
} // namespace Gamut

namespace OutputEncoding {
    enum class ColorSpace : std::uint8_t;
}

// FJ_TEMP_BRIDGE: completed fixed hull into native Library/recipe holds; remove S4.E.
// Sources live through synchronous return. This binding owns no math or cache.
namespace JuicerGamut {
    bool output_transform(OutputEncoding::ColorSpace space, Gamut::OutputGamutTransform& out, std::string& diagnostic);
    bool output_boundary(const Gamut::OutputGamutTransform& transform, Gamut::OutputBoundaryTable& out);
    bool input_hull(const Spectral::Curve& xBar,
                    const Spectral::Curve& yBar,
                    const Spectral::Curve& zBar,
                    const Spectral::Curve& d65,
                    Gamut::InputCompressionHull& out);
} // namespace JuicerGamut
