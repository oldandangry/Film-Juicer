#pragma once

#include "ofx_bindings.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private candidate discovery. Shipping discovery continues to select OFXS.
 * The returned descriptor is immutable and binary-lived. Callback arguments
 * follow OFX's live-pointer contract; setHost's handle and fetched APIs remain
 * valid until binary unload. B1 implements only Load/Describe/Unload. */
int fj_ofx_plugin_count(void);
OfxPlugin* fj_ofx_plugin(int index);

#ifdef __cplusplus
}
#endif
