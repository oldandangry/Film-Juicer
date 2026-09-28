#pragma once

#include <optional>
#include <string>

#include "juicer_cuda_api.h"
#include "FilmEffectsFrameDescriptors.h"
#include "Scanner.h"

namespace JuicerCuda {

    // Invocation-local fixed values. Large host tables remain separate borrowed spans.
    struct PreparedDescriptors {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        Spektrafilm::ProfilePolarity capturePolarity = Spektrafilm::ProfilePolarity::Unsupported;
        Scanner::ScannerSpectralLutDescriptor scanner{};
        Scanner::ScannerColorCorrectionDescriptor correction{};
        Scanner::ScannerPostEffectsDescriptor post{};
        Scanner::ColorRuntime color{};
        OutputGamutRecipe outputGamut{};
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        std::optional<ScatterHalationFrameDescriptor> halation;
        Spektrafilm::SpatialDirDescriptor spatialDir{};
        Spektrafilm::VisualGrainRecipe grainRecipe{};
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
    };

    bool decode_prepared_descriptors(
        const FjPreparedHostData& source,
        PreparedDescriptors& out,
        std::string& diagnostic);

} // namespace JuicerCuda
