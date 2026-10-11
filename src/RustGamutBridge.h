#pragma once

namespace Spectral {
    struct Curve;
}
namespace Gamut {
    struct InputCompressionHull;
}

// FJ_TEMP_BRIDGE: completed fixed hull into native Library/recipe holds; remove S4.E.
// Sources live through synchronous return. This binding owns no math or cache.
namespace JuicerGamut {
    bool input_hull(const Spectral::Curve& xBar,
                    const Spectral::Curve& yBar,
                    const Spectral::Curve& zBar,
                    const Spectral::Curve& d65,
                    Gamut::InputCompressionHull& out);
}
