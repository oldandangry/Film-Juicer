#include "ScanStage.h"

namespace Pipeline {

    bool ScanStage::run(const WorkingState& ws, const ScanInputs& in, ScanOutputs& out) {
        (void)ws;
        (void)in;
        out.logXyz = LogXyz{};
        out.xyz = Xyz{};
        return false;
    }

} // namespace Pipeline

