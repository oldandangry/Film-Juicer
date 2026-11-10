#pragma once

#include "ofx_bindings.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private descriptor used by the standard discovery facade. The returned
 * descriptor is immutable and binary-lived. Callback arguments follow OFX's
 * live-pointer contract; setHost's handle and fetched APIs remain valid until
 * binary unload. Rust owns the module, instance and synchronous render actions. */
int fj_ofx_plugin_count(void);
OfxPlugin* fj_ofx_plugin(int index);

#ifdef __cplusplus
}
#endif
