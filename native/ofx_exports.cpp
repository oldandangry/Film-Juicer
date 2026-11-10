#include "juicer_ofx_api.h"

#if defined(_WIN32)
#define FJ_OFX_DISCOVERY OfxExport
#else
#define FJ_OFX_DISCOVERY OfxExport __attribute__((visibility("default")))
#endif

extern "C" {

FJ_OFX_DISCOVERY int OfxGetNumberOfPlugins(void) {
    return fj_ofx_plugin_count();
}

FJ_OFX_DISCOVERY OfxPlugin* OfxGetPlugin(int index) {
    return fj_ofx_plugin(index);
}
}
