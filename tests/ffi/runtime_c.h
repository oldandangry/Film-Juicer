#pragma once

#include "juicer_legacy_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Real creation/lifecycle consumer; linked only with the product native engine. */
FjStatus fj_test_runtime_c_consumer(FjPathView resource_dir);

#ifdef __cplusplus
}
#endif
