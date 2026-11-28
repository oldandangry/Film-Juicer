#include "PreparedCudaFrameState.h"

#include "ProcessRoot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Logging.h"
#include "Scanner.h"

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/Scan/JuicerCudaScannerBinding.h"

namespace JuicerProcess {

    using JuicerCuda::ResourceManager::WorkspaceLayout;

    bool spatial_dir_scratch_has_required_roles(
        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
        Spektrafilm::DirScratchTier tier,
        const Spektrafilm::DirScratchPlaneRoles& roles) noexcept {
        if (!spatial_dir_roles_match_tier(tier, roles)) {
            return false;
        }
        if (tier == Spektrafilm::DirScratchTier::Tier0) {
            return true;
        }
        const bool hasThreeRaw =
            scratch.rawCorrectionY && scratch.rawCorrectionM && scratch.rawCorrectionC;
        const bool rawCorrectionMatch =
            (roles.rawCorrectionPlanes == 3 && hasThreeRaw) ||
            (roles.rawCorrectionPlanes == 1 && scratch.rawCorrectionY);
        const bool filterTempMatch =
            roles.filterTempPlanes == 0 || scratch.filterTemp;
        const bool tier1Base =
            rawCorrectionMatch &&
            scratch.filteredCorrectionY && scratch.filteredCorrectionM && scratch.filteredCorrectionC &&
            filterTempMatch;
        if (!tier1Base) {
            return false;
        }
        const bool hasAliasedForwardTemps =
            scratch.filterTempM && scratch.filterTempC;
        const bool channelTempsMatch =
            roles.filterTempPlanes == 1 ||
            (roles.filterTempPlanes == 2 && scratch.filterTempM) ||
            (roles.filterTempPlanes == 3 && hasAliasedForwardTemps);
        const bool cachedLogRawMatch =
            roles.cachedLogRawPlanes == 0 ||
            (roles.cachedLogRawPlanes == 2 &&
             scratch.logRawB && scratch.logRawG) ||
            (roles.cachedLogRawPlanes == 3 &&
             scratch.logRawB && scratch.logRawG && scratch.logRawR);
        return channelTempsMatch && cachedLogRawMatch;
    }

    bool spatial_dir_scratch_has_final_develop_roles(
        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
        if (!scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
            !scratch.filteredCorrectionC) {
            return false;
        }
        const bool cachedLogRawMatch =
            targetRoles.cachedLogRawPlanes == 0 ||
            (targetRoles.cachedLogRawPlanes == 2 &&
             scratch.logRawB && scratch.logRawG) ||
            (targetRoles.cachedLogRawPlanes == 3 &&
             scratch.logRawB && scratch.logRawG && scratch.logRawR);
        if (!cachedLogRawMatch) {
            return false;
        }
        const bool filterTempsCompatible =
            targetRoles.filterTempPlanes == 3 ||
            (targetRoles.filterTempPlanes == 2 && scratch.filterTempM) ||
            targetRoles.filterTempPlanes <= 1;
        return filterTempsCompatible;
    }

    namespace {

#if JUICER_DIAGNOSTICS_COMPILED
        void trace_scanner_post_effects_view_inactive(
            const JuicerCuda::ResourceManager::SubmissionTransaction* transaction,
            const char* reason,
            std::uint64_t requestedDescriptorHash,
            const Scanner::ScannerPostEffectsDescriptor* descriptor,
            const Root::PreparedCudaFrame::ScannerWorkspaceView* scratch,
            const Root::PreparedCudaFrame::KernelView* lensBlur,
            const Root::PreparedCudaFrame::KernelView* unsharp,
            const Root::PreparedCudaFrame::KernelView* glare) noexcept {
            if (!JTRACE_ENABLED(1)) {
                return;
            }

            try {
                std::string msg = "event=scanner_post_effects_view_inactive";
                if (transaction) {
                    msg += " transaction_id=";
                    msg += std::to_string(static_cast<unsigned long long>(transaction->transactionId));
                    msg += " snapshot_id=";
                    msg += std::to_string(static_cast<unsigned long long>(transaction->snapshot.snapshotId));
                } else {
                    msg += " transaction_id=0 snapshot_id=0";
                }
                msg += " reason=";
                msg += reason && reason[0] ? reason : "unspecified";
                msg += " requested_hash=";
                msg += std::to_string(static_cast<unsigned long long>(requestedDescriptorHash));
                msg += " prepared_hash=";
                msg += std::to_string(static_cast<unsigned long long>(descriptor ? descriptor->hash : 0));
                msg += " glare_active=";
                msg += descriptor && descriptor->glareActive ? "1" : "0";
                msg += " glare_blur_sigma_px=";
                msg += descriptor ? std::to_string(descriptor->glareBlurSigmaPx) : "0";
                msg += " lens_blur_sigma_px=";
                msg += descriptor ? std::to_string(descriptor->lensBlurSigmaPx) : "0";
                msg += " unsharp_sigma_px=";
                msg += descriptor ? std::to_string(descriptor->unsharpSigmaPx) : "0";
                msg += " unsharp_amount=";
                msg += descriptor ? std::to_string(descriptor->unsharpAmount) : "0";

                const auto append_scratch_bool = [&](const char* name, const void* value) {
                    msg += " scratch_";
                    msg += name;
                    msg += "=";
                    msg += value ? "1" : "0";
                };
                msg += " scratch_active=";
                msg += scratch && scratch->active ? "1" : "0";
                msg += " scratch_rgb_aliased_from_spatial_dir_filtered=";
                msg += scratch && scratch->rgbAliasedFromSpatialDirFiltered ? "1" : "0";
                append_scratch_bool("rgb_r", scratch ? scratch->rgbR : nullptr);
                append_scratch_bool("rgb_g", scratch ? scratch->rgbG : nullptr);
                append_scratch_bool("rgb_b", scratch ? scratch->rgbB : nullptr);
                append_scratch_bool("tmp", scratch ? scratch->tmp : nullptr);
                append_scratch_bool("blurred", scratch ? scratch->blurred : nullptr);
                append_scratch_bool("aux", scratch ? scratch->aux : nullptr);
                append_scratch_bool("gate_transmittance", scratch ? scratch->gateTransmittance : nullptr);
                msg += " scratch_gate_transmittance_width=";
                msg += std::to_string(scratch ? scratch->gateTransmittanceWidth : 0);
                msg += " scratch_gate_transmittance_height=";
                msg += std::to_string(scratch ? scratch->gateTransmittanceHeight : 0);

                const auto append_kernel =
                    [&](const char* prefix, const Root::PreparedCudaFrame::KernelView* kernel) {
                        msg += " ";
                        msg += prefix;
                        msg += "_kernel=";
                        msg += kernel && kernel->weights ? "1" : "0";
                        msg += " ";
                        msg += prefix;
                        msg += "_radius=";
                        msg += std::to_string(kernel ? kernel->radius : 0);
                    };
                append_kernel("lens", lensBlur);
                append_kernel("unsharp", unsharp);
                append_kernel("glare", glare);
                JTRACE("CUDA", msg);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }

        void trace_scanner_post_rgb_alias(
            const JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            std::uint64_t descriptorHash,
            std::uint64_t spatialDirDescriptorHash) noexcept {
            if (!JTRACE_ENABLED(1)) {
                return;
            }

            try {
                std::string msg = "event=scanner_post_rgb_alias";
                msg += " transaction_id=";
                msg += std::to_string(static_cast<unsigned long long>(transaction.transactionId));
                msg += " snapshot_id=";
                msg += std::to_string(static_cast<unsigned long long>(transaction.snapshot.snapshotId));
                msg += " descriptor_hash=";
                msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
                msg += " spatial_dir_descriptor_hash=";
                msg += std::to_string(static_cast<unsigned long long>(spatialDirDescriptorHash));
                msg += " rgb_source=spatial_dir_filtered_alias alias_planes=3";
                JTRACE("DIR_DESCRIPTOR", msg);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
#endif

    } // namespace

    JuicerCuda::PreparedLegacyGrainView
    Root::PreparedCudaFrame::legacy_grain_resources() const noexcept {
        JuicerCuda::PreparedLegacyGrainView view{};
        if (!_state || !_state->legacyGrainPrepared ||
            !_state->legacyGrainDescriptor.has_value() ||
            !_state->grainStaticResources ||
            !_state->transaction.active ||
            _state->transaction.committed) {
            return view;
        }

        const Spektrafilm::LegacyGrainFrameDescriptor& descriptor =
            *_state->legacyGrainDescriptor;
        const JuicerCuda::Resources& staticResources =
            *_state->grainStaticResources;
        view.staticNoise.stbn = staticResources.stbnData;
        view.staticNoise.stbnWidth = staticResources.stbnWidth;
        view.staticNoise.stbnHeight = staticResources.stbnHeight;
        view.staticNoise.stbnFrames = staticResources.stbnFrames;
        view.staticNoise.wangTiles = staticResources.wangTilesData;
        view.staticNoise.wangLut = staticResources.wangLutData;
        view.staticNoise.wangWidth = staticResources.wangWidth;
        view.staticNoise.wangHeight = staticResources.wangHeight;
        view.staticNoise.wangCount = staticResources.wangCount;
        view.staticNoise.wangColors = staticResources.wangColors;
        view.correlation = _state->preparedGrainCorrelation;
        for (int layer = 0; layer < 3; ++layer) {
            for (int channel = 0; channel < 3; ++channel) {
                view.dyeCloud[layer][channel] =
                    _state->preparedGrainDyeCloud[layer][channel];
            }
        }
        view.densityLayers = _state->preparedGrainDensityLayers;
        view.descriptor = &descriptor;

        bool ready =
            descriptor.active && descriptor.hash != 0 &&
            view.staticNoise.stbn &&
            view.staticNoise.stbnWidth > 0 &&
            view.staticNoise.stbnHeight > 0 &&
            view.staticNoise.stbnFrames > 0 &&
            view.staticNoise.wangTiles &&
            view.staticNoise.wangLut &&
            view.staticNoise.wangWidth > 0 &&
            view.staticNoise.wangHeight > 0 &&
            view.staticNoise.wangCount > 0 &&
            view.staticNoise.wangColors > 0;
        auto gaussian_ready =
            [](const Spektrafilm::LegacyGrainGaussian& gaussian,
               const JuicerCuda::LegacyGrainPreparedGaussianView& prepared) {
                if (gaussian.radius <= 0) {
                    return gaussian.hash == 0 && !prepared.active &&
                           !prepared.weights &&
                           prepared.radius == 0 &&
                           prepared.descriptorHash == 0;
                }
                return gaussian.hash != 0 && prepared.active &&
                       prepared.descriptorHash == gaussian.hash &&
                       prepared.weights &&
                       prepared.radius == gaussian.radius &&
                       prepared.sigma == gaussian.sigmaPx;
            };
        for (std::size_t index = 0;
             index < descriptor.correlation.size();
             ++index) {
            ready = ready &&
                    gaussian_ready(
                        descriptor.correlation[index],
                        view.correlation[index]);
        }
        for (std::size_t layer = 0; layer < 3; ++layer) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                ready = ready &&
                        gaussian_ready(
                            descriptor.dyeCloud[layer][channel],
                            view.dyeCloud[layer][channel]);
            }
        }
        if (descriptor.densityCurvesLayersHash != 0) {
            ready =
                ready && view.densityLayers.active &&
                view.densityLayers.hash ==
                    descriptor.densityCurvesLayersHash;
            for (int layer = 0; layer < 3; ++layer) {
                for (int channel = 0; channel < 3; ++channel) {
                    ready = ready &&
                            view.densityLayers
                                .curves[layer][channel];
                }
            }
            for (int channel = 0; channel < 3; ++channel) {
                const JuicerCuda::DeviceCurveView& baseCurve =
                    view.densityLayers.baseCurvesCmy[channel];
                ready = ready && baseCurve.x && baseCurve.y &&
                        baseCurve.n > 0 && baseCurve.domainBegin >= 0 &&
                        baseCurve.domainEnd >= baseCurve.domainBegin &&
                        baseCurve.domainEnd < baseCurve.n;
            }
        } else {
            ready = ready && !view.densityLayers.active &&
                    view.densityLayers.hash == 0;
        }

        view.active = ready;
        return view.active
                   ? view
                   : JuicerCuda::PreparedLegacyGrainView{};
    }

    const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
    Root::PreparedCudaFrame::film_juicer_effects_descriptor() const noexcept {
        if (!active() || !_state->effectsDescriptor.has_value()) {
            return nullptr;
        }
        return &*_state->effectsDescriptor;
    }

    Root::PreparedCudaFrame::LegacyGrainWorkspaceView
    Root::PreparedCudaFrame::legacy_grain_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        LegacyGrainWorkspaceView view{};
        if (!_state || !_state->legacyGrainPrepared ||
            !_state->legacyGrainDescriptor.has_value() ||
            !workspace_marker_matches_current_frame(workspace) ||
            !_state->scratchWorkspace.retainedLeaseActive || !_state->scratchWorkspace.admittedLayout) {
            return view;
        }
        const WorkspaceLayout& request =
            active_workspace_layout(workspace);
        if (!request.allocation().needOptics) {
            return view;
        }
        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->resources->scannerScratch;
        view.filterTemp =
            request.allocation().needSharedTmp ? scratch.tmp : nullptr;
        view.scaleWork =
            request.allocation().needBlurred ? scratch.blurred : nullptr;
        view.deltaAccum = (request.grain_storage() != JuicerCuda::ResourceManager::GrainStorage::Absent) ? scratch.aux : nullptr;
        view.layerWork =
            request.grain_uses_layers() ? scratch.grainTmp : nullptr;
        view.sharedDelta =
            request.grain_uses_shared() ? scratch.grainTmpShared : nullptr;
        if (request.grain_storage() == JuicerCuda::ResourceManager::GrainStorage::DirRaw) {
            const auto& dir = _state->resources->spatialDirScratch;
            const std::size_t requiredElements =
                static_cast<std::size_t>(request.allocation().requestedWidth) * request.allocation().requestedHeight;
            if (dir.capacityElements < requiredElements) {
                return LegacyGrainWorkspaceView{};
            }
            const std::array<float*, 3> raw{
                dir.rawCorrectionY, dir.rawCorrectionM, dir.rawCorrectionC};
            const std::array<float*, 8> live{
                view.filterTemp, view.scaleWork, dir.filteredCorrectionY, dir.filteredCorrectionM, dir.filteredCorrectionC, dir.logRawB, dir.logRawG, dir.logRawR};
            for (std::array<float*, 3>::const_iterator current = raw.begin();
                 current != raw.end();
                 ++current) {
                if (!*current ||
                    std::find(raw.begin(), current, *current) != current ||
                    std::find(live.begin(), live.end(), *current) != live.end()) {
                    return LegacyGrainWorkspaceView{};
                }
            }
            view.deltaAccum = raw[0];
            view.layerWork = request.grain_uses_layers() ? raw[1] : nullptr;
            view.sharedDelta = request.grain_uses_shared() ? raw[2] : nullptr;
        }
        view.frameUniforms = (request.grain_storage() != JuicerCuda::ResourceManager::GrainStorage::Absent)
                                 ? scratch.grainFrameUniforms
                                 : nullptr;
        view.scratchShape =
            _state->legacyGrainDescriptor->scratchShape;

        bool ready = view.filterTemp && view.scaleWork &&
                     view.deltaAccum && view.frameUniforms;
        switch (view.scratchShape) {
            case Spektrafilm::LegacyGrainScratchShape::Streamed:
                ready = ready && !view.layerWork &&
                        !view.sharedDelta &&
                        !request.grain_uses_layers() &&
                        !request.grain_uses_shared();
                break;
            case Spektrafilm::LegacyGrainScratchShape::StreamedShared:
                ready = ready && !view.layerWork &&
                        view.sharedDelta &&
                        !request.grain_uses_layers() &&
                        request.grain_uses_shared();
                break;
            case Spektrafilm::LegacyGrainScratchShape::StreamedLayers:
                ready = ready && view.layerWork &&
                        !view.sharedDelta &&
                        request.grain_uses_layers() &&
                        !request.grain_uses_shared();
                break;
            case Spektrafilm::LegacyGrainScratchShape::StreamedLayersShared:
                ready = ready && view.layerWork &&
                        view.sharedDelta &&
                        request.grain_uses_layers() &&
                        request.grain_uses_shared();
                break;
            case Spektrafilm::LegacyGrainScratchShape::None:
            default:
                ready = false;
                break;
        }
        view.active = ready;
        return view.active ? view : LegacyGrainWorkspaceView{};
    }

    const JuicerCuda::GrainV2::FilmQuad* Root::PreparedCudaFrame::grain_v2_bank() const noexcept {
        return _state ? _state->grainV2Bank : nullptr;
    }

    JuicerCuda::GrainV2::ProxyWorkspace* Root::PreparedCudaFrame::grain_v2_proxy_workspace() const noexcept {
        return _state ? static_cast<JuicerCuda::GrainV2::ProxyWorkspace*>(_state->grainV2ProxyWorkspace) : nullptr;
    }

    Root::PreparedCudaFrame::CaptureFilmDensityWorkspaceView
    Root::PreparedCudaFrame::capture_film_density_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        CaptureFilmDensityWorkspaceView view{};
        if (!_state || !workspace_marker_matches_current_frame(workspace) ||
            !_state->scratchWorkspace.retainedLeaseActive ||
            (_state->capturePolarity !=
                 Spektrafilm::ProfilePolarity::Negative &&
             _state->capturePolarity !=
                 Spektrafilm::ProfilePolarity::Positive)) {
            return view;
        }
        const WorkspaceLayout& request = active_workspace_layout(workspace);
        if (!request.allocation().needOptics) {
            return view;
        }
        const ScannerWorkspaceView shared = scanner_workspace(workspace);
        if (!shared.active) {
            return view;
        }
        view.c = shared.rgbR;
        view.m = shared.rgbG;
        view.y = shared.rgbB;
        view.polarity = _state->capturePolarity;
        view.active = view.c && view.m && view.y;
        return view.active ? view : CaptureFilmDensityWorkspaceView{};
    }

    Root::PreparedCudaFrame::FocusedRgbWorkspaceView
    Root::PreparedCudaFrame::focused_rgb_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        FocusedRgbWorkspaceView view{};
        if (!_state || !workspace_marker_matches_current_frame(workspace) ||
            !_state->scratchWorkspace.retainedLeaseActive || !_state->scratchWorkspace.admittedLayout) {
            return view;
        }
        const WorkspaceLayout& request = active_workspace_layout(workspace);
        if (!request.allocation().needOptics) {
            return view;
        }
        const ScannerWorkspaceView shared = scanner_workspace(workspace);
        if (!shared.active) {
            return view;
        }
        view.r = shared.rgbR;
        view.g = shared.rgbG;
        view.b = shared.rgbB;
        view.active = view.r && view.g && view.b;
        return view.active ? view : FocusedRgbWorkspaceView{};
    }

    Root::PreparedCudaFrame::AutoExposureBufferView Root::PreparedCudaFrame::auto_exposure_buffers() const noexcept {
        AutoExposureBufferView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return view;
        }

        const State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        view.scratch = workspace.scratch;
        view.deviceState = workspace.deviceState;
        view.weightsWidth = workspace.weightsWidth;
        view.weightsHeight = workspace.weightsHeight;
        view.keyHash = workspace.keyHash;
        const bool histogramReady =
            workspace.method != Spektrafilm::AutoExposureMethod::Median ||
            (view.scratch.maxYBits && view.scratch.histogram);
        const bool partialsReady =
            workspace.method == Spektrafilm::AutoExposureMethod::Median ||
            (view.scratch.partialsA && view.scratch.partialsB && view.scratch.partialCapacity > 0);
        const bool weightsReady =
            workspace.method != Spektrafilm::AutoExposureMethod::CenterWeighted ||
            (view.scratch.weightsX && view.scratch.weightsY);
        view.active =
            workspace.active &&
            histogramReady &&
            partialsReady &&
            weightsReady &&
            view.deviceState.exposureScale;
        return view;
    }

    Root::PreparedCudaFrame::FocusedPreparedView Root::PreparedCudaFrame::focused_resources() const noexcept {
        FocusedPreparedView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed ||
            !_state->focusedScannerColor) {
            return view;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        view.film.finalSensB = {resources.sensB.x, resources.sensB.y, resources.sensB.n, resources.sensB.domainBegin, resources.sensB.domainEnd};
        view.film.finalSensG = {resources.sensG.x, resources.sensG.y, resources.sensG.n, resources.sensG.domainBegin, resources.sensG.domainEnd};
        view.film.finalSensR = {resources.sensR.x, resources.sensR.y, resources.sensR.n, resources.sensR.domainBegin, resources.sensR.domainEnd};
        view.film.normalizedDensB = {resources.densB.x, resources.densB.y, resources.densB.n, resources.densB.domainBegin, resources.densB.domainEnd};
        view.film.normalizedDensG = {resources.densG.x, resources.densG.y, resources.densG.n, resources.densG.domainBegin, resources.densG.domainEnd};
        view.film.normalizedDensR = {resources.densR.x, resources.densR.y, resources.densR.n, resources.densR.domainBegin, resources.densR.domainEnd};
        if (resources.filmDirHash != 0) {
            view.film.dirDensB = {resources.dirDensB.x, resources.dirDensB.y, resources.dirDensB.n, resources.dirDensB.domainBegin, resources.dirDensB.domainEnd};
            view.film.dirDensG = {resources.dirDensG.x, resources.dirDensG.y, resources.dirDensG.n, resources.dirDensG.domainBegin, resources.dirDensG.domainEnd};
            view.film.dirDensR = {resources.dirDensR.x, resources.dirDensR.y, resources.dirDensR.n, resources.dirDensR.domainBegin, resources.dirDensR.domainEnd};
        }
        view.film.tablesIllum = resources.tablesIllum;
        view.film.tablesK = resources.tablesK;
        view.film.filmTcLut = resources.filmTcLut;
        view.film.filmTcLutExtent = resources.filmTcLutExtent;
        view.film.mallettBasis = resources.mallettBasis;
        view.film.mallettBasisK = resources.mallettBasisK;
        std::copy_n(_state->focusedInputRgbToXyz.data(), 9, view.film.inputRGBToXYZ);
        std::copy_n(_state->focusedInputXyzAdapt.data(), 9, view.film.inputXYZAdapt);
        std::copy_n(
            _state->focusedXyzToLinearSrgb.data(),
            9,
            view.film.xyzToLinearSrgb);
        view.film.applyInputChromaticAdapt = _state->focusedApplyInputChromaticAdapt ? 1 : 0;
        view.film.finalSensitivityHash = resources.filmFinalSensitivityHash;
        view.film.normalizedDensityCurvesHash = resources.filmDensityCurvesHash;
        view.film.dirCouplersHash = resources.filmDirHash;
        view.scannerSpectra = &_state->focusedScannerSpectra;
        view.scannerColor = &*_state->focusedScannerColor;
        view.outputGamutTransform = _state->focusedOutputGamutTransform ? &*_state->focusedOutputGamutTransform : nullptr;
        view.outputGamutCmax = resources.outputGamutCmax;
        view.selectedMethod = resources.filmRgbToRawMethod;
        const bool outputGamutExpected =
            view.scannerColor->outputGamutRecipeHash != 0;
        const bool outputGamutReady = outputGamutExpected
                                          ? view.outputGamutTransform &&
                                                view.outputGamutCmax &&
                                                resources.outputGamutRecipeHash ==
                                                    view.scannerColor->outputGamutRecipeHash
                                          : !view.outputGamutTransform &&
                                                !view.outputGamutCmax &&
                                                resources.outputGamutRecipeHash == 0;
        view.active = outputGamutReady;
        return view;
    }

    JuicerCuda::Diffusion::DiffusionPreparedView
    Root::PreparedCudaFrame::diffusion_resources() const noexcept {
        if (!active() || !_state->diffusionLease.active()) {
            return {};
        }
        return _state->diffusionLease.view();
    }

    JuicerCuda::ScatterHalationPreparedView
    Root::PreparedCudaFrame::scatter_halation_resources() const noexcept {
        JuicerCuda::ScatterHalationPreparedView view{};
        if (!active() || !_state->scatterHalationDescriptor) {
            return view;
        }
        view.descriptor = &*_state->scatterHalationDescriptor;
        view.fullFrameWidth = _state->scatterHalationFrameWidth;
        view.fullFrameHeight = _state->scatterHalationFrameHeight;
        view.carrierSource = _state->scatterHalationCarrierSource;
        view.currentCarrier = _state->scatterHalationCarrier;
        view.filterTemp = _state->scatterHalationFilterTemp;
        view.weightedAccumulation =
            _state->scatterHalationWeightedAccumulation;
        return view;
    }

    JuicerCuda::CameraFilmLinearExposurePlanes Root::PreparedCudaFrame::camera_film_linear_carrier() const noexcept {
        return active() ? _state->scatterHalationCarrier : JuicerCuda::CameraFilmLinearExposurePlanes{};
    }

    Root::PreparedCudaFrame::PrintPreparedView Root::PreparedCudaFrame::print_resources() const noexcept {
        PrintPreparedView view{};
        if (!_state || !_state->resources || !Spektrafilm::scan_route_is_print(_state->scanRoute) ||
            !_state->transaction.active || _state->transaction.committed) {
            return view;
        }
        const JuicerCuda::Resources& resources = *_state->resources;
        const JuicerCuda::PrintResourceDescriptors& descriptors = _state->printDescriptors;
        view.filmDensityTables.epsC = resources.printFilmDensityTables.epsC;
        view.filmDensityTables.epsM = resources.printFilmDensityTables.epsM;
        view.filmDensityTables.epsY = resources.printFilmDensityTables.epsY;
        view.filmDensityTables.baseDensityMin = resources.printFilmDensityTables.baseDensityMin;
        view.filmDensityTables.K = resources.printFilmDensityTables.K;
        view.printSensC = {resources.printSensC.x, resources.printSensC.y, resources.printSensC.n, resources.printSensC.domainBegin, resources.printSensC.domainEnd};
        view.printSensM = {resources.printSensM.x, resources.printSensM.y, resources.printSensM.n, resources.printSensM.domainBegin, resources.printSensM.domainEnd};
        view.printSensY = {resources.printSensY.x, resources.printSensY.y, resources.printSensY.n, resources.printSensY.domainBegin, resources.printSensY.domainEnd};
        view.printDcC = {resources.printDcC.x, resources.printDcC.y, resources.printDcC.n, resources.printDcC.domainBegin, resources.printDcC.domainEnd};
        view.printDcM = {resources.printDcC.x, resources.printDcM.y, resources.printDcM.n, resources.printDcM.domainBegin, resources.printDcM.domainEnd};
        view.printDcY = {resources.printDcC.x, resources.printDcY.y, resources.printDcY.n, resources.printDcY.domainBegin, resources.printDcY.domainEnd};
        view.mainIlluminant = resources.printIllumFiltered;
        view.preflashIlluminant = resources.printPreflashIllumFiltered;
        view.spectralSampleCount = resources.printIllumK;
        std::copy_n(_state->printPreflashRawCmy.data(), 3, view.preflashRawCmy);
        view.normalizer = _state->printNormalizer;
        view.filmDensityTablesHash = resources.printFilmDensityTablesDescriptorHash;
        view.profileTablesHash = resources.printProfileTablesDescriptorHash;
        view.mainIlluminantHash = resources.printMainIlluminantDescriptorHash;
        view.preflashIlluminantHash = resources.printPreflashIlluminantDescriptorHash;
        view.preflashActive = descriptors.preflashActive;
        view.active =
            view.filmDensityTablesHash != 0 && view.filmDensityTablesHash == descriptors.filmDensityTablesHash &&
            view.filmDensityTables.K == Spectral::kNumSamples &&
            view.filmDensityTables.epsC && view.filmDensityTables.epsM && view.filmDensityTables.epsY &&
            view.filmDensityTables.baseDensityMin &&
            view.profileTablesHash != 0 && view.profileTablesHash == descriptors.profileTablesHash &&
            view.printDcC.x && view.printDcC.y && view.printDcM.y && view.printDcY.y &&
            view.printSensC.y && view.printSensM.y && view.printSensY.y &&
            view.mainIlluminantHash != 0 && view.mainIlluminantHash == descriptors.mainIlluminantHash &&
            view.mainIlluminant && view.spectralSampleCount == Spectral::kNumSamples &&
            (view.preflashActive
                 ? view.preflashIlluminantHash != 0 && view.preflashIlluminantHash == descriptors.preflashIlluminantHash &&
                       view.preflashIlluminant && resources.printPreflashIllumK == Spectral::kNumSamples
                 : !view.preflashIlluminant && view.preflashIlluminantHash == 0 &&
                       resources.printPreflashIllumK == 0);
        return view;
    }

    Root::PreparedCudaFrame::SpatialDirScratchView Root::PreparedCudaFrame::spatial_dir_scratch(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        SpatialDirScratchView view{};
        if (!workspace_marker_matches_current_frame(workspace) || !workspace.layout().allocation().needSpatialDir) {
            return view;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive || !_state->scratchWorkspace.admittedLayout) {
            return view;
        }

        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch =
            _state->resources->spatialDirScratch;
        const WorkspaceLayout& effectiveRequest =
            (_state->scratchWorkspace.admittedLayout->allocation().needSpatialDir &&
             _state->scratchWorkspace.admittedLayout->allocation().spatialDirDescriptorHash ==
                 workspace.layout().allocation().spatialDirDescriptorHash)
                ? *_state->scratchWorkspace.admittedLayout
                : workspace.layout();
        const Spektrafilm::DirScratchPlaneRoles& roles =
            effectiveRequest.allocation().spatialDirPlaneRoles;
        view.rawCorrectionY =
            roles.rawCorrectionPlanes >= 1 ? scratch.rawCorrectionY : nullptr;
        view.rawCorrectionM =
            roles.rawCorrectionPlanes >= 3 ? scratch.rawCorrectionM : nullptr;
        view.rawCorrectionC =
            roles.rawCorrectionPlanes >= 3 ? scratch.rawCorrectionC : nullptr;
        view.filteredCorrectionY =
            roles.filteredCorrectionPlanes >= 1 ? scratch.filteredCorrectionY : nullptr;
        view.filteredCorrectionM =
            roles.filteredCorrectionPlanes >= 2 ? scratch.filteredCorrectionM : nullptr;
        view.filteredCorrectionC =
            roles.filteredCorrectionPlanes >= 3 ? scratch.filteredCorrectionC : nullptr;
        view.filterTemp =
            roles.filterTempPlanes >= 1 ? scratch.filterTemp : nullptr;
        view.filterTempM =
            roles.filterTempPlanes >= 2 ? scratch.filterTempM : nullptr;
        view.filterTempC =
            roles.filterTempPlanes >= 3 ? scratch.filterTempC : nullptr;
        view.logRawB =
            roles.cachedLogRawPlanes >= 1 ? scratch.logRawB : nullptr;
        view.logRawG =
            roles.cachedLogRawPlanes >= 2 ? scratch.logRawG : nullptr;
        view.logRawR =
            roles.cachedLogRawPlanes >= 3 ? scratch.logRawR : nullptr;
        view.descriptorHash = effectiveRequest.allocation().spatialDirDescriptorHash;
        view.scratchTier = effectiveRequest.allocation().spatialDirScratchTier;
        view.planeRoles = effectiveRequest.allocation().spatialDirPlaneRoles;
        view.targetScratchTier = effectiveRequest.allocation().spatialDirTargetScratchTier;
        view.targetPlaneRoles = effectiveRequest.allocation().spatialDirTargetPlaneRoles;
        view.active =
            spatial_dir_scratch_has_required_roles(
                scratch,
                view.scratchTier,
                view.planeRoles) ||
            spatial_dir_scratch_has_final_develop_roles(
                scratch,
                view.targetPlaneRoles);
        return view;
    }

    Root::PreparedCudaFrame::SpatialDirPreparedView Root::PreparedCudaFrame::spatial_dir_resources(
        const WorkspaceLeaseMarker& workspace,
        std::uint64_t descriptorHash) const noexcept {
        SpatialDirPreparedView view{};
        if (!workspace_marker_matches_current_frame(workspace) ||
            !workspace.layout().allocation().needSpatialDir ||
            !_state ||
            _state->spatialDirDescriptor.hash == 0 ||
            _state->spatialDirDescriptor.hash != descriptorHash) {
            return view;
        }
        const auto& kernels = _state->resources->spatialDirKernels;
        const auto& boundaries = _state->resources->spatialDirBoundaries;
        const Spektrafilm::DirFilterPlan& filterPlan =
            _state->spatialDirDescriptor.filterPlan;
        view.gaussian = {
            kernels[0].weights,
            kernels[0].radius,
            filterPlan.components[0].sigmaPixels};
        for (int slot = 0; slot < 3; ++slot) {
            const int component = slot + 1;
            view.exponential[slot] = {
                kernels[static_cast<std::size_t>(component)].weights,
                kernels[static_cast<std::size_t>(component)].radius,
                component < filterPlan.componentCount
                    ? filterPlan.components[static_cast<std::size_t>(component)].sigmaPixels
                    : 0.0f};
        }
        view.descriptorHash = descriptorHash;
        view.active = true;
        for (int slot = 0;
             slot < _state->spatialDirDescriptor.filterPlan.componentCount;
             ++slot) {
            const Spektrafilm::DirGaussianComponentPlan& component =
                _state->spatialDirDescriptor.filterPlan.components[static_cast<std::size_t>(slot)];
            if (component.weight > 0.0f &&
                component.referenceOperator ==
                    Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect) {
                const auto& kernel = kernels[static_cast<std::size_t>(slot)];
                view.active = view.active && kernel.weights &&
                              kernel.radius == component.radius;
            } else if (component.weight > 0.0f &&
                       component.referenceOperator ==
                           Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect) {
                const auto& boundary = boundaries[static_cast<std::size_t>(slot)];
                auto& boundaryView = view.boundaries[static_cast<std::size_t>(slot)];
                boundaryView.horizontalWeights = boundary.weights;
                boundaryView.verticalWeights =
                    boundary.weights
                        ? boundary.weights +
                              static_cast<std::ptrdiff_t>(3 * boundary.horizontalLength)
                        : nullptr;
                boundaryView.horizontalLength = boundary.horizontalLength;
                boundaryView.verticalLength = boundary.verticalLength;
                boundaryView.horizontalTerminalSize = boundary.horizontalTerminalSize;
                boundaryView.verticalTerminalSize = boundary.verticalTerminalSize;
                boundaryView.horizontalTerminalMatrix =
                    boundary.horizontalTerminalMatrix;
                boundaryView.verticalTerminalMatrix = boundary.verticalTerminalMatrix;
                view.active = view.active && boundary.weights &&
                              boundary.horizontalLength > 0 &&
                              boundary.verticalLength > 0 &&
                              boundary.horizontalTerminalSize > 0 &&
                              boundary.verticalTerminalSize > 0;
            }
        }
        return view;
    }

    Root::PreparedCudaFrame::ScannerWorkspaceView Root::PreparedCudaFrame::scanner_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        ScannerWorkspaceView view{};
        if (!workspace_marker_matches_current_frame(workspace)) {
            return view;
        }
        const WorkspaceLayout& activeRequest = active_workspace_layout(workspace);
        if (!activeRequest.allocation().needOptics) {
            return view;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive || !_state->scratchWorkspace.admittedLayout) {
            return view;
        }

        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->resources->scannerScratch;
        if (activeRequest.allocation().aliasScannerRgbFromSpatialDirFiltered) {
            const JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir =
                _state->resources->spatialDirScratch;
            view.rgbR = spatialDir.filteredCorrectionC;
            view.rgbG = spatialDir.filteredCorrectionM;
            view.rgbB = spatialDir.filteredCorrectionY;
            view.rgbAliasedFromSpatialDirFiltered =
                view.rgbR && view.rgbG && view.rgbB;
        } else {
            view.rgbR = scratch.rgbR;
            view.rgbG = scratch.rgbG;
            view.rgbB = scratch.rgbB;
        }
        view.tmp = scratch.tmp;
        view.blurred = scratch.blurred;
        view.aux = scratch.aux;
        view.grainTmp = scratch.grainTmp;
        view.grainTmpShared = scratch.grainTmpShared;
        view.gateTransmittance = scratch.gateTransmittance;
        view.filmDustTransmittance = scratch.filmDustTransmittance;
        view.gateTransmittanceWidth = scratch.gateWidth;
        view.gateTransmittanceHeight = scratch.gateHeight;
        view.active = view.rgbR && view.rgbG && view.rgbB;
        view.hasGateTransmittance = view.gateTransmittance && view.gateTransmittanceWidth > 0 && view.gateTransmittanceHeight > 0;
        return view;
    }

    Root::PreparedCudaFrame::ScannerPostEffectsPreparedView
    Root::PreparedCudaFrame::scanner_post_effects_resources(
        const WorkspaceLeaseMarker& workspace,
        std::uint64_t descriptorHash) const noexcept {
        ScannerPostEffectsPreparedView view{};
        if (!_state) {
#if JUICER_DIAGNOSTICS_COMPILED
            trace_scanner_post_effects_view_inactive(
                nullptr,
                "missing_state",
                descriptorHash,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
#endif
            return view;
        }
        const Scanner::ScannerPostEffectsDescriptor& descriptor =
            _state->scannerPostEffectsDescriptor;
        if (descriptorHash == 0) {
#if JUICER_DIAGNOSTICS_COMPILED
            trace_scanner_post_effects_view_inactive(
                &_state->transaction,
                "requested_hash_zero",
                descriptorHash,
                &descriptor,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
#endif
            return view;
        }
        if (descriptor.hash != descriptorHash) {
#if JUICER_DIAGNOSTICS_COMPILED
            trace_scanner_post_effects_view_inactive(
                &_state->transaction,
                "descriptor_hash_mismatch",
                descriptorHash,
                &descriptor,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
#endif
            return view;
        }
        view.scratch = scanner_workspace(workspace);
        if (!view.scratch.active) {
#if JUICER_DIAGNOSTICS_COMPILED
            trace_scanner_post_effects_view_inactive(
                &_state->transaction,
                "scanner_scratch_inactive",
                descriptorHash,
                &descriptor,
                &view.scratch,
                nullptr,
                nullptr,
                nullptr);
#endif
            return view;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (view.scratch.rgbAliasedFromSpatialDirFiltered) {
            trace_scanner_post_rgb_alias(
                _state->transaction,
                descriptorHash,
                workspace.layout().allocation().spatialDirDescriptorHash);
        }
#endif
        const JuicerCuda::Resources& resources = *_state->resources;
        if (descriptor.lensBlurSigmaPx > 0.0f) {
            view.lensBlur = {
                resources.scannerLensBlurKernel.weights,
                resources.scannerLensBlurKernel.radius,
                resources.scannerLensBlurKernel.sigma};
        }
        if (descriptor.unsharpSigmaPx > 0.0f) {
            view.unsharp = {
                resources.scannerUnsharpKernel.weights,
                resources.scannerUnsharpKernel.radius,
                resources.scannerUnsharpKernel.sigma};
        }
        if (descriptor.glareActive && descriptor.glareBlurSigmaPx > 0.0f) {
            view.glare = {
                resources.scannerGlareKernel.weights,
                resources.scannerGlareKernel.radius,
                resources.scannerGlareKernel.sigma};
        }
        view.descriptorHash = descriptor.hash;
        const bool sharedTmpReady = view.scratch.tmp != nullptr;
        const bool glareScratchReady =
            !descriptor.glareActive ||
            descriptor.glareBlurSigmaPx <= 0.0f ||
            view.scratch.blurred != nullptr;
        const bool lensReady =
            descriptor.lensBlurSigmaPx <= 0.0f ||
            (view.lensBlur.weights &&
             view.lensBlur.radius == descriptor.lensBlurRadius &&
             view.lensBlur.sigma == descriptor.lensBlurSigmaPx);
        const bool unsharpReady =
            descriptor.unsharpSigmaPx <= 0.0f ||
            (view.unsharp.weights &&
             view.unsharp.radius == descriptor.unsharpRadius &&
             view.unsharp.sigma == descriptor.unsharpSigmaPx);
        const bool glareReady =
            !descriptor.glareActive ||
            descriptor.glareBlurSigmaPx <= 0.0f ||
            (view.glare.weights &&
             view.glare.radius == descriptor.glareBlurRadius &&
             view.glare.sigma == descriptor.glareBlurSigmaPx);
        view.active = sharedTmpReady && glareScratchReady &&
                      lensReady && unsharpReady && glareReady;
#if JUICER_DIAGNOSTICS_COMPILED
        if (!view.active) {
            trace_scanner_post_effects_view_inactive(
                &_state->transaction,
                "scanner_kernel_missing",
                descriptorHash,
                &descriptor,
                &view.scratch,
                &view.lensBlur,
                &view.unsharp,
                &view.glare);
        }
#endif
        return view;
    }


} // namespace JuicerProcess
