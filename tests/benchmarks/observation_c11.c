#include "observation.h"

/* Compile this translation unit as C11 to keep the fixture ABI usable by C. */
_Static_assert(sizeof(((FjMeasurementRecord*)0)->construction_starts) == 8,
               "measurement counters require 64-bit storage");
_Static_assert(sizeof(((FjMeasurementRecord*)0)->upload_state_bytes) / sizeof(uint64_t) == 4,
               "upload pool exposes four states");

int fj_measurement_c11_record_size(void) {
    return (int)sizeof(FjMeasurementRecord);
}
