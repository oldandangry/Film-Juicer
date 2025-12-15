#include "PipelineRunner.h"

#include "DevelopFilmStage.h"
#include "DevelopPrintStage.h"
#include "ExposeFilmStage.h"
#include "ExposePrintStage.h"
#include "ScanStage.h"

namespace Pipeline {

    PipelineRunner::PipelineRunner(const PipelineRunnerConfig& cfg) : cfg_(cfg) {}

    bool PipelineRunner::run_pixel(const WorkingState& ws, const RgbLinear& inRgb, float outRgb[3]) const {
        (void)ws;

        if (!outRgb) {
            return false;
        }

        outRgb[0] = inRgb.v[0];
        outRgb[1] = inRgb.v[1];
        outRgb[2] = inRgb.v[2];

        return false;
    }

} // namespace Pipeline

