#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "ColorTransforms.h"
#include "GamutCompression.h"
#include "Scanner.h"
#include "SpectralData.h"

struct FocusedRenderPayload {
    Spectral::SpectralTables exposureTables;
    std::array<float, 9> spdSInv{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    Spectral::FilmRawConfig filmRawConfig;
    std::optional<Spectral::FilmTcLut> filmTcLut;
    std::optional<std::array<float, Spectral::kNumSamples>> printMainIlluminant;
    Spectral::SpectralTables scannerTables;
    Scanner::ColorRuntime scannerColor;
    std::shared_ptr<const Gamut::OutputBoundaryTable> outputBoundaryTable;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t scannerHash = 0;
};
