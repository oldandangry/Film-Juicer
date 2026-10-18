#pragma once

#include <cstddef>
#include <string>

#include "ScanRoute.h"

struct FjFilmDevelopmentOwner;
struct GrainContract;
struct DirCouplersControls;
struct DirCouplersRecipe;
struct DensityBoundsRecipe;
struct VisualGrainControls;
struct VisualGrainRecipe;
struct FilmJuicerEffectsRecipe;
struct PhysicalEffectsControls;
struct SpatialOptics;
namespace Spektrafilm {
    struct SpatialOpticsControls;
}
namespace Profiles {
    struct FilmProfile;
    struct ProfileDigest;
} // namespace Profiles

namespace JuicerAssets {
    // SF_TEMP_BRIDGE_FILM_DIR_DENSITY: synchronous additional hold; remove S4.E.
    // Moves transfer the consuming obligation and leave the source handle null.
    class DevelopmentHold final {
    public:
        DevelopmentHold() noexcept = default;
        explicit DevelopmentHold(FjFilmDevelopmentOwner* owner) noexcept;
        ~DevelopmentHold();
        DevelopmentHold(const DevelopmentHold&) = delete;
        DevelopmentHold& operator=(const DevelopmentHold&) = delete;
        DevelopmentHold(DevelopmentHold&& source) noexcept;
        DevelopmentHold& operator=(DevelopmentHold&& source) noexcept;

        const FjFilmDevelopmentOwner* get() const noexcept {
            return _owner;
        }

    private:
        FjFilmDevelopmentOwner* _owner = nullptr;
    };

    // SF_TEMP_BRIDGE_VISUAL_GRAIN: copy one completed fixed Rust value; remove S4.E.
    bool prepare_visual_grain(const DevelopmentHold& development, const VisualGrainControls& controls, VisualGrainRecipe& out, std::string& diagnostic);
    // SF_TEMP_BRIDGE_PHYSICAL_EFFECTS: copy one completed fixed value; remove S4.E.
    bool prepare_physical_effects(const PhysicalEffectsControls& controls, const VisualGrainRecipe& grain, FilmJuicerEffectsRecipe& out, std::string& diagnostic);
    // SF_TEMP_BRIDGE_SPATIAL_OPTICS: complete fixed projection; remove S4.E.
    bool prepare_spatial_optics(const Spektrafilm::SpatialOpticsControls& controls, const Profiles::ProfileDigest& digest, bool printDomain, SpatialOptics& out, std::string& diagnostic);
    bool prepare_film_density_minimum(const GrainContract& requested, GrainContract& out, std::string& diagnostic);
    bool prepare_film_dir(const DevelopmentHold& development, const DirCouplersControls& controls, const Profiles::FilmProfile& profile, Spektrafilm::ScanRoute route, std::size_t samples, DirCouplersRecipe& out, std::string& diagnostic);
    bool prepare_film_density_bounds(const DevelopmentHold& development, const GrainContract& requested, const Profiles::FilmProfile& profile, Spektrafilm::ScanRoute route, DensityBoundsRecipe& out, std::string& diagnostic);
} // namespace JuicerAssets
