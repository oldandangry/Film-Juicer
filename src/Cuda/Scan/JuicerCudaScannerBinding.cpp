#include "Cuda/Scan/JuicerCudaScannerBinding.h"

#include <iterator>

namespace JuicerCuda {
    ScannerSpectra bind_scanner_spectra(const Scanner::SpectralTablesView& tables) {
        ScannerSpectra spectra{};
        for (std::size_t i = 0; i < std::size(spectra.wavelengths); ++i) {
            spectra.wavelengths[i] = {
                tables.epsC[i], tables.epsM[i], tables.epsY[i], tables.baseDensityMin.empty() ? 0.0F : tables.baseDensityMin[i], tables.Ax[i], tables.Ay[i], tables.Az[i]};
        }
        spectra.inverseYNormalization = tables.invYn;
        return spectra;
    }
} // namespace JuicerCuda
