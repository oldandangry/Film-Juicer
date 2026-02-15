#pragma once

#include "WorkingState.h"

namespace WorkingStateSharing {

struct WorkingStateCorePayload {
    Spectral::Curve densB;
    Spectral::Curve densG;
    Spectral::Curve densR;
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{};
    bool hasDensityCurvesLayers = false;

    Profiles::GrainMetadata grain;
    Profiles::HalationMetadata halation;
    Profiles::ProfileGlare negativeGlare;
    Profiles::ProfileGlare printGlare;

    Spectral::Curve sensB;
    Spectral::Curve sensG;
    Spectral::Curve sensR;
    Spectral::Curve negSensB;
    Spectral::Curve negSensG;
    Spectral::Curve negSensR;

    Spectral::Curve baseMin;
    Spectral::Curve baseMid;
    bool hasBaseline = false;
    float baselineMixReference = 0.0f;
    float printBaselineMixReference = 0.0f;
    float gammaFactorB = 1.0f;
    float gammaFactorG = 1.0f;
    float gammaFactorR = 1.0f;

    Spectral::SpectralTables tablesView;
    Spectral::SpectralTables tablesPrint;
    Spectral::SpectralTables tablesRef;
    Spectral::SpectralTables tablesScan;

    Scanner::ScannerIlluminant negativeScannerIlluminant;
    Scanner::ScannerDensityRange negativeDensityRange;
    Scanner::ScannerStaticKey negativeStaticKey;
    Scanner::ColorRuntime negativeColorRuntime;
    Scanner::ScannerMediumRuntime negativeMediumRuntime;

    Scanner::ScannerIlluminant printScannerIlluminant;
    Scanner::ScannerDensityRange printDensityRange;
    Scanner::ScannerStaticKey printStaticKey;
    Scanner::ColorRuntime printColorRuntime;
    Scanner::ScannerMediumRuntime printMediumRuntime;

    bool negativeScannerValid = false;
    bool printScannerValid = false;
    bool printGlareCompensated = false;

    float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
    bool spdReady = false;
    Spectral::FilmRawConfig filmRaw;
    std::shared_ptr<const Print::Runtime> printRT;
    Spectral::NegativeCouplerParams negParams{};
    std::uint64_t coreShareHash = 0;
};

inline void capture_working_state_core_payload(const WorkingState& in, WorkingStateCorePayload& out) {
    out.densB = in.densB;
    out.densG = in.densG;
    out.densR = in.densR;
    out.densityCurvesLayers = in.densityCurvesLayers;
    out.hasDensityCurvesLayers = in.hasDensityCurvesLayers;

    out.grain = in.grain;
    out.halation = in.halation;
    out.negativeGlare = in.negativeGlare;
    out.printGlare = in.printGlare;

    out.sensB = in.sensB;
    out.sensG = in.sensG;
    out.sensR = in.sensR;
    out.negSensB = in.negSensB;
    out.negSensG = in.negSensG;
    out.negSensR = in.negSensR;

    out.baseMin = in.baseMin;
    out.baseMid = in.baseMid;
    out.hasBaseline = in.hasBaseline;
    out.baselineMixReference = in.baselineMixReference;
    out.printBaselineMixReference = in.printBaselineMixReference;
    out.gammaFactorB = in.gammaFactorB;
    out.gammaFactorG = in.gammaFactorG;
    out.gammaFactorR = in.gammaFactorR;

    out.tablesView = in.tablesView;
    out.tablesPrint = in.tablesPrint;
    out.tablesRef = in.tablesRef;
    out.tablesScan = in.tablesScan;

    out.negativeScannerIlluminant = in.negativeScannerIlluminant;
    out.negativeDensityRange = in.negativeDensityRange;
    out.negativeStaticKey = in.negativeStaticKey;
    out.negativeColorRuntime = in.negativeColorRuntime;
    out.negativeMediumRuntime = in.negativeMediumRuntime;

    out.printScannerIlluminant = in.printScannerIlluminant;
    out.printDensityRange = in.printDensityRange;
    out.printStaticKey = in.printStaticKey;
    out.printColorRuntime = in.printColorRuntime;
    out.printMediumRuntime = in.printMediumRuntime;

    out.negativeScannerValid = in.negativeScannerValid;
    out.printScannerValid = in.printScannerValid;
    out.printGlareCompensated = in.printGlareCompensated;

    for (int i = 0; i < 9; ++i) {
        out.spdSInv[i] = in.spdSInv[i];
    }
    out.spdReady = in.spdReady;
    out.filmRaw = in.filmRaw;
    out.printRT = in.printRT;
    out.negParams = in.negParams;
    out.coreShareHash = in.coreShareHash;
}

inline void apply_working_state_core_payload(const WorkingStateCorePayload& in, WorkingState& out) {
    out.densB = in.densB;
    out.densG = in.densG;
    out.densR = in.densR;
    out.densityCurvesLayers = in.densityCurvesLayers;
    out.hasDensityCurvesLayers = in.hasDensityCurvesLayers;

    out.grain = in.grain;
    out.halation = in.halation;
    out.negativeGlare = in.negativeGlare;
    out.printGlare = in.printGlare;

    out.sensB = in.sensB;
    out.sensG = in.sensG;
    out.sensR = in.sensR;
    out.negSensB = in.negSensB;
    out.negSensG = in.negSensG;
    out.negSensR = in.negSensR;

    out.baseMin = in.baseMin;
    out.baseMid = in.baseMid;
    out.hasBaseline = in.hasBaseline;
    out.baselineMixReference = in.baselineMixReference;
    out.printBaselineMixReference = in.printBaselineMixReference;
    out.gammaFactorB = in.gammaFactorB;
    out.gammaFactorG = in.gammaFactorG;
    out.gammaFactorR = in.gammaFactorR;

    out.tablesView = in.tablesView;
    out.tablesPrint = in.tablesPrint;
    out.tablesRef = in.tablesRef;
    out.tablesScan = in.tablesScan;

    out.negativeScannerIlluminant = in.negativeScannerIlluminant;
    out.negativeDensityRange = in.negativeDensityRange;
    out.negativeStaticKey = in.negativeStaticKey;
    out.negativeColorRuntime = in.negativeColorRuntime;
    out.negativeMediumRuntime = in.negativeMediumRuntime;

    out.printScannerIlluminant = in.printScannerIlluminant;
    out.printDensityRange = in.printDensityRange;
    out.printStaticKey = in.printStaticKey;
    out.printColorRuntime = in.printColorRuntime;
    out.printMediumRuntime = in.printMediumRuntime;

    out.negativeScannerValid = in.negativeScannerValid;
    out.printScannerValid = in.printScannerValid;
    out.printGlareCompensated = in.printGlareCompensated;

    for (int i = 0; i < 9; ++i) {
        out.spdSInv[i] = in.spdSInv[i];
    }
    out.spdReady = in.spdReady;
    out.filmRaw = in.filmRaw;
    out.printRT = in.printRT;
    out.negParams = in.negParams;
    out.coreShareHash = in.coreShareHash;

    out.negativeMediumRuntime.tables = (out.tablesScan.K > 0) ? &out.tablesScan : nullptr;
    out.negativeMediumRuntime.color = &out.negativeColorRuntime;
    out.negativeMediumRuntime.staticKey = out.negativeStaticKey;
    if (out.printScannerValid) {
        out.printMediumRuntime.tables = (out.tablesPrint.K > 0) ? &out.tablesPrint : nullptr;
        out.printMediumRuntime.color = &out.printColorRuntime;
    }
    else {
        out.printMediumRuntime.tables = nullptr;
        out.printMediumRuntime.color = nullptr;
    }
    out.printMediumRuntime.staticKey = out.printStaticKey;
}

} // namespace WorkingStateSharing

