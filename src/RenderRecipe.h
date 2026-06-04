#pragma once

#include <string>
#include <utility>

#include "ScanRoute.h"

// RenderRecipe owner map:
// - ProfileRoute owns selected profile keys and resolved ScanRoute.
// - FilmRawRecipe will own input color, RGB-to-raw, and exposure ordering when consumed.
// - FilmDevelopRecipe and DirCouplersRecipe will own film development and DIR policy.
// - DensityBoundsRecipe will own direct-film and print-route density bounds.
// - PrintRecipe will own print filters, neutral calibration, preflash, and print exposure.
// - ScannerOutputRecipe will own scanner LUT, correction, post effects, output color, and CCTF.
// - SpatialOptics will own optics activation, backend identity, and exactness failure policy.
// - GrainContract will own spektrafilm grain/layer semantics and density_min source.
// - FrameRequest owns frame-local extent, pixel size, temporal tokens, and metering request facts.
// - PreparedCudaFrame/context resource internals own durable GPU handles, scratch, staging, and views.
// This is source-adjacent orientation, not a runtime registry.
struct ProfileRoute {
    std::string filmProfileKey;
    std::string printProfileKey;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
};

struct RenderRecipe {
    ProfileRoute profileRoute;
};

namespace Spektrafilm {

    using ::ProfileRoute;
    using ::RenderRecipe;

    inline RenderRecipe make_render_recipe(ProfileRoute profileRoute) {
        RenderRecipe recipe{};
        recipe.profileRoute = std::move(profileRoute);
        return recipe;
    }

    inline RenderRecipe make_render_recipe(
        std::string filmProfileKey,
        std::string printProfileKey,
        ScanRoute scanRoute) {
        ProfileRoute profileRoute{};
        profileRoute.filmProfileKey = std::move(filmProfileKey);
        profileRoute.printProfileKey = std::move(printProfileKey);
        profileRoute.scanRoute = scanRoute;
        return make_render_recipe(std::move(profileRoute));
    }

} // namespace Spektrafilm
