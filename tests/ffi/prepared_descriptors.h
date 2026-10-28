#pragma once

#include "juicer_cuda_descriptors.h"

namespace JuicerCudaTest {

    FjOutputEncoding prepare_output_encoding(std::uint32_t space);

    void encode_spatial_dir_descriptor(const Spektrafilm::SpatialDirDescriptor&, FjSpatialDir&);

    // Writes fixed families only; the caller retains and projects the large table owners.
    void encode_prepared_descriptors(
        const JuicerCuda::PreparedDescriptors& source,
        FjPreparedHostData& out);

} // namespace JuicerCudaTest
