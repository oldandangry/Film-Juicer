#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ColorTransforms.h"
#include "ProfileCatalog.h"
#include "RenderRecipe.h"
#include "ScanRoute.h"
#include "Scanner.h"
#include "ScatterHalation.h"
#include "SpectralData.h"
#include "ofxImageEffect.h"

namespace JuicerAtomic {

    template <typename T>
    inline std::shared_ptr<T> load_shared_ptr(const std::shared_ptr<T>* p) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        std::shared_ptr<T> v = std::atomic_load(p);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        return v;
    }

    template <typename T>
    inline void store_shared_ptr(std::shared_ptr<T>* p, std::shared_ptr<T> value) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        std::atomic_store(p, std::move(value));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

} // namespace JuicerAtomic

#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

struct FocusedRenderPayload {
    Spectral::SpectralTables exposureTables;
    std::array<float, 9> spdSInv{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    Spectral::FilmRawConfig filmRawConfig;
    Spectral::SpectralTables scannerTables;
    Scanner::ColorRuntime scannerColor;
    std::uint64_t uploadCoreHash = 0;
    std::uint64_t scannerHash = 0;
};

struct FocusedRenderStateBuildProduct {
    Spektrafilm::RenderRecipe recipe;
    FocusedRenderPayload payload;
};

struct DirectRenderState {
    Spektrafilm::RenderRecipe recipe;
    FocusedRenderPayload payload;
    std::uint64_t buildCounter = 0;
};

struct PrintRenderState {
    Spektrafilm::RenderRecipe recipe;
    FocusedRenderPayload payload;
    std::uint64_t buildCounter = 0;
};

struct ParamSnapshot {
    std::uint64_t filmProfileAssetVersionToken = 0;
    std::uint64_t printProfileAssetVersionToken = 0;

    double printExposure = 1.0;
    double printPreflashExposure = 0.0;
    double preflashMFilterCc = 0.0;
    double preflashYFilterCc = 0.0;
    double printShadowCompensationFactor = 0.0;
    double printShadowCompensationDensity = 1.2;
    double printShadowCompensationTransition = 0.3;
    double glarePercent = 0.03;
    double glareRoughness = 0.7;
    double glareBlurSigmaPx = 0.5;
    double printDminFactor = 0.4;
    double couplersAmount = 1.0;
    double couplersInhibitionSameLayer = 1.0;
    double couplersInhibitionInterlayer = 1.0;
    double couplersDiffusionSizeUm = 20.0;
    double cameraExposureCompensationEv = 0.0;
    float cameraFilmFormatLongEdgeMm = 35.0f;
    double scannerLensBlurSigmaPx = 0.0;
    double scannerBlackLevel = 0.01;
    double scannerWhiteLevel = 0.98;
    double gateWeaveAmount = 1.0;

    std::array<double, 2> couplersGammaInterlayerRToGb{{0.353, 0.302}};
    std::array<double, 2> couplersGammaInterlayerGToRb{{0.154, 0.353}};
    std::array<double, 2> couplersGammaInterlayerBToRg{{0.168, 0.226}};
    std::array<double, 2> scannerUnsharpMask{{0.7, 0.7}};
    std::array<double, 3> printUiYmcCc{};
    std::array<double, 3> couplersGammaSameLayerRgb{{0.336, 0.319, 0.273}};
    std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
    std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};

    Spektrafilm::DiffusionFilterAuthoredControls cameraDiffusion;
    Spektrafilm::DiffusionFilterAuthoredControls enlargerDiffusion;

    std::string filmProfileKey = Spektrafilm::kDefaultFilmProfileKey;
    std::string printProfileKey = Spektrafilm::kDefaultPrintProfileKey;

    int spectralUpsamplingMode = 0;
    int refIll = 0;
    int enlIll = 3;
    int enlDichroicSet = 0;
    int normalizePrintExposure = 1;
    int printExposureCompensation = 1;
    int couplersActive = 1;
    int couplersGammaUseStock = 1;
    int inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::DaVinciWideGamut);
    int inputCctfDecoding = 0;
    int hanatos2025AdaptationWindow = 1;
    int hanatos2025AdaptationSurface = 0;
    int cameraAutoExposureEnabled = 1;
    int cameraMeteringMethod = 0;
    ScatterHalationControls scatterHalationControls;
    int scannerBlackCorrection = 0;
    int scannerWhiteCorrection = 0;
    int scannerUseLut = 1;
    int scannerLutResolution = 17;
    int outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
    int outputCctfEncoding = 1;
    int outputLinearPassThrough = 0;
    Spektrafilm::VisualGrainControls grainControls;
    float filmDustAmount = 0.0f;
    float filmScratchAmount = 0.0f;
    float gateDustAmount = 0.0f;
    float gateScratchAmount = 0.0f;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;

    bool glareActive = true;
    bool cameraFilterOverride = false;
};

uint64_t hash_params(const ParamSnapshot& p);

struct PendingParamsState {
    struct Uninitialized {};

    struct Valid {
        ParamSnapshot params;
        std::uint64_t fullHash = 0;
    };

    struct InvalidSnapshotControls {
        std::string diagnostic;
    };

    // Leaf lock for the coalesced pending-params snapshot. Do not nest with InstanceState::m.
    std::mutex m;
    std::variant<Uninitialized, Valid, InvalidSnapshotControls> value{Uninitialized{}};
};

struct InstanceState {
    // Lock-order rule for P6: rebuildMutex may precede m during rebuild snapshot/publication.
    // m is otherwise a short publication/snapshot lock and must not be held across heavy rebuild
    // work, sleep/wait paths, or while acquiring the leaf mutexes below.
    std::mutex m;
    // Serializes focused-state rebuilds so heavy construction runs outside InstanceState::m.
    std::mutex rebuildMutex;
    // Published route state. Readers use std::atomic_load; writers use std::atomic_store.
    std::shared_ptr<const DirectRenderState> activeDirectState;
    std::shared_ptr<const PrintRenderState> activePrintState;
    std::atomic<std::uint64_t> buildCounterNext{0};
    std::atomic<std::uint32_t> frameBoundsVersion{0};
    OfxRectI cachedFrameBounds{0, 0, 0, 0};

    std::atomic<std::uint64_t> lastHash{0};

    // Latest parameter snapshot observed from UI callbacks; consumed/coalesced on render thread.
    PendingParamsState pending;

    bool suppressParamEvents = false;

    std::uint64_t sessionSeed = 0;
    std::uint64_t instanceToken = 0;
    std::atomic<std::uint64_t> submissionSnapshotIdNext{1};

    // Snapshot latch: all submissions for the same frame token reuse one immutable snapshot payload.
    std::mutex submissionSnapshotLatchMutex;
    bool submissionSnapshotLatchValid = false;
    JuicerCuda::ResourceManager::SubmissionSnapshot submissionSnapshotLatch{};
};

enum class PendingRenderAdmissionStatus : std::uint8_t {
    NeedsSnapshotAcquisition = 0,
    InvalidSnapshotControls,
    RebuildFailed,
    AdmittedDirect,
    AdmittedPrint
};

struct PendingRenderAdmissionResult {
    PendingRenderAdmissionStatus status =
        PendingRenderAdmissionStatus::NeedsSnapshotAcquisition;
    ParamSnapshot snapshot;
    std::shared_ptr<const DirectRenderState> directState;
    std::shared_ptr<const PrintRenderState> printState;
    std::string diagnostic;
};

bool spektrafilm_profile_catalog_ready();
const char* spektrafilm_profile_catalog_failure();
int film_profile_option_count();
const char* film_profile_option_key(int index);
const char* film_profile_option_label(int index);
int print_profile_option_count();
const char* print_profile_option_key(int index);
const char* print_profile_option_label(int index);
bool rebuild_direct_render_state(InstanceState& S, const ParamSnapshot& P);
bool rebuild_print_render_state(InstanceState& S, const ParamSnapshot& P);
bool build_direct_render_state_product(
    const ParamSnapshot& snapshot,
    FocusedRenderStateBuildProduct& out,
    std::string& outError);
bool build_print_render_state_product(
    const ParamSnapshot& snapshot,
    FocusedRenderStateBuildProduct& out,
    std::string& outError);
PendingRenderAdmissionResult admit_pending_render_state(InstanceState& state);
