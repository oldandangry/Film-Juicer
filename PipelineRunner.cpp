#include "PipelineRunner.h"

#include "DevelopFilmStage.h"
#include "DevelopPrintStage.h"
#include "ExposeFilmStage.h"
#include "ExposePrintStage.h"
#include "Print.h"

namespace Pipeline {

    PipelineRunner::PipelineRunner(const PipelineRunnerConfig& cfg) : cfg_(cfg) {}

    float PipelineRunner::compute_midgray_factor(
        const WorkingState& ws,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        const Couplers::Runtime& dirRT,
        float exposureCompScale)
    {
        return ExposePrintStage::compute_midgray_factor(
            ws,
            printRuntime,
            printParams,
            dirRT,
            exposureCompScale);
    }

    bool PipelineRunner::run_density_pixel(
        const WorkingState& ws,
        const DensityPixelInputs& in,
        DensityPixelOutputs& out) const
    {
        out = DensityPixelOutputs{};

        FilmRaw filmRaw{};
        if (in.useFilmRawOverride) {
            filmRaw = in.filmRawOverride;
        }
        else {
            ExposeFilmInputs exposeIn{};
            exposeIn.rgb = in.rgb;
            exposeIn.exposureScale = in.exposureScale;

            ExposeFilmOutputs exposeOut{};
            if (!ExposeFilmStage::run(ws, exposeIn, exposeOut)) {
                return false;
            }
            filmRaw = exposeOut.filmRaw;
        }

        DevelopFilmInputs devIn{};
        devIn.filmRaw = filmRaw;
        devIn.dirRuntime = in.dirRuntime;
        devIn.applyDirRuntime = in.applyDirRuntime;
        devIn.useSpatialDIR = in.useSpatialDIR;
        devIn.spatialLogECorrectionsYMC[0] = in.spatialLogECorrectionsYMC[0];
        devIn.spatialLogECorrectionsYMC[1] = in.spatialLogECorrectionsYMC[1];
        devIn.spatialLogECorrectionsYMC[2] = in.spatialLogECorrectionsYMC[2];

        DevelopFilmOutputs devOut{};
        if (!DevelopFilmStage::run(ws, devIn, devOut)) {
            return false;
        }

        out.filmRaw = filmRaw;
        out.filmLogRaw = devOut.filmLogRaw;
        out.negativeDensity = devOut.negativeDensity;
        out.medium = DensityMedium::Negative;

        if (!cfg_.enablePrint) {
            return true;
        }

        if (!in.printRuntime || !in.printParams || in.printParams->bypass) {
            return true;
        }
        if (!in.printScratch) {
            return false;
        }

        ExposePrintInputs exposePrintIn{};
        exposePrintIn.printRuntime = in.printRuntime;
        exposePrintIn.printParams = in.printParams;
        exposePrintIn.negativeDensity = devOut.negativeDensity;
        exposePrintIn.midgrayFactor = in.midgrayFactor;

        ExposePrintOutputs exposePrintOut{};
        if (!ExposePrintStage::run(ws, exposePrintIn, exposePrintOut, *in.printScratch)) {
            return false;
        }

        DevelopPrintInputs developPrintIn{};
        developPrintIn.printRuntime = in.printRuntime;
        developPrintIn.printLogRaw = exposePrintOut.printLogRaw;

        DevelopPrintOutputs developPrintOut{};
        if (!DevelopPrintStage::run(developPrintIn, developPrintOut)) {
            return false;
        }

        out.printDensity = developPrintOut.printDensity;
        out.medium = DensityMedium::Print;
        return true;
    }

} // namespace Pipeline
