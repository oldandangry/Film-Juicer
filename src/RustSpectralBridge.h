#pragma once

#include <array>
#include <cstdint>

namespace Spectral {
    struct Curve;
    struct SpectralTables;
} // namespace Spectral
namespace Scanner {
    struct ScannerIlluminant;
} // namespace Scanner

// FJ_TEMP_BRIDGE: spectral source borrows and native value projection; remove S4.E.
namespace JuicerSpectral {
    void build_tables(const std::array<std::array<float, 3>, 81>& dyesCmy,
                      const std::array<float, 81>& baselineMin,
                      const Spectral::Curve& illuminant,
                      std::uint64_t illuminantHash,
                      Spectral::SpectralTables& out);
    std::array<float, 9> s_inverse(const Spectral::SpectralTables& tables);
    bool integrate_white(const Spectral::Curve& illuminant, const char* label, Scanner::ScannerIlluminant& out);
} // namespace JuicerSpectral
