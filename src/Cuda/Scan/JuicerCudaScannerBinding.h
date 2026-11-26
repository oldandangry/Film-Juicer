#pragma once

#include "Cuda/JuicerCudaPayloads.h"
#include "Scanner.h"

namespace JuicerCuda {
    // The prepared descriptor has already admitted the fixed wavelength spans.
    ScannerSpectra bind_scanner_spectra(const Scanner::SpectralTablesView& tables);
} // namespace JuicerCuda
