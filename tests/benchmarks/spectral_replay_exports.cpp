#include <cstdint>

#include "../ofx/module_test.h"
#include "ofxCore.h"

#if defined(_WIN32)
#define FJ_SPECTRAL_REPLAY_EXPORT OfxExport
#else
#define FJ_SPECTRAL_REPLAY_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
FJ_SPECTRAL_REPLAY_EXPORT void fj_spectral_replay_set_tokens(
    std::uint64_t session, std::uint64_t instance, std::uint64_t clip) {
    fj_test_ofx_render_tokens(session, instance, clip);
}
}
