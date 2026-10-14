#pragma once

#include "juicer_cuda_descriptors.h"

namespace JuicerCudaTest {

    FjOutputEncoding prepare_output_encoding(std::uint32_t space);

    // Writes fixed families only; the caller retains and projects the large table owners.
    void encode_prepared_descriptors(
        const JuicerCuda::PreparedDescriptors& source,
        FjPreparedHostData& out);

} // namespace JuicerCudaTest
