#include "ProcessRoot.h"

#include <string>
#include <tuple>

enum class NeutralFilterThreadClass : unsigned char {
    Control = 0,
    RenderWorker = 1
};

namespace {

    JuicerAssets::NeutralFilterLookupThread to_asset_lookup_thread(NeutralFilterThreadClass threadClass) {
        switch (threadClass) {
            case NeutralFilterThreadClass::Control:
                return JuicerAssets::NeutralFilterLookupThread::Control;
            case NeutralFilterThreadClass::RenderWorker:
                return JuicerAssets::NeutralFilterLookupThread::RenderWorker;
            default:
                return JuicerAssets::NeutralFilterLookupThread::Control;
        }
    }

} // namespace

bool load_enlarger_neutral_filters(
    const std::string& jsonPath,
    const std::string& paperKey,
    const std::string& illuminantKey,
    const std::string& negativeKey,
    std::tuple<float, float, float>& outYMC,
    NeutralFilterThreadClass threadClass,
    std::string* outSelectedDbVersionHash) {
    if (outSelectedDbVersionHash) {
        outSelectedDbVersionHash->clear();
    }

    const JuicerAssets::NeutralFilterLookupResult result =
        JuicerProcess::root().assets().lookup_neutral_filters(
            jsonPath,
            paperKey,
            illuminantKey,
            negativeKey,
            to_asset_lookup_thread(threadClass));
    if (!result.found) {
        return false;
    }

    outYMC = result.ymc;
    if (outSelectedDbVersionHash) {
        *outSelectedDbVersionHash = result.selectedDbVersionHash;
    }
    return true;
}
