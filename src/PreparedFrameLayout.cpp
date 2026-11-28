#include "PreparedFrameLayout.h"

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

    bool checked_size_multiply(
        std::size_t left,
        std::size_t right,
        std::size_t& out) noexcept {
        if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
            return false;
        }
        out = left * right;
        return true;
    }

    bool checked_size_add(
        std::size_t left,
        std::size_t right,
        std::size_t& out) noexcept {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            return false;
        }
        out = left + right;
        return true;
    }

    const char* scatter_halation_carrier_source_label(
        JuicerCuda::ScatterHalationCarrierSource source) noexcept {
        switch (source) {
            case JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes:
                return "DedicatedPreparedPlanes";
            case JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes:
                return "CameraDiffusionStagePlanes";
            default:
                return "Unknown";
        }
    }

    bool derive_scatter_halation_allocation_shape(
        int width,
        int height,
        JuicerCuda::ScatterHalationCarrierSource carrierSource,
        bool spatial,
        ScatterHalationAllocationShape& out,
        const char*& failedFact) noexcept {
        out = ScatterHalationAllocationShape{};
        failedFact = nullptr;
        out.physicalAllocationCount = static_cast<std::size_t>(spatial) +
                                      static_cast<std::size_t>(carrierSource == JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes);

        std::size_t pixelCount = 0;
        if (!checked_size_multiply(
                static_cast<std::size_t>(width),
                static_cast<std::size_t>(height),
                pixelCount)) {
            failedFact = "pixel_count";
            return false;
        }
        if (!checked_size_multiply(pixelCount, sizeof(float), out.planeBytes)) {
            failedFact = "plane_bytes";
            return false;
        }
        if (spatial) {
            out.filterSecondPlaneOffset = out.planeBytes;
            if (!checked_size_multiply(2u, out.planeBytes, out.filterBytes)) {
                failedFact = "filter_bytes";
                return false;
            }
        }

        if (carrierSource ==
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes) {
            out.carrierGreenPlaneOffset = out.planeBytes;
            if (!checked_size_add(
                    out.carrierGreenPlaneOffset,
                    out.planeBytes,
                    out.carrierBluePlaneOffset)) {
                failedFact = "carrier_blue_plane_offset";
                return false;
            }
            if (!checked_size_multiply(3u, out.planeBytes, out.carrierBytes)) {
                failedFact = "carrier_bytes";
                return false;
            }
            std::size_t carrierEnd = 0;
            if (!checked_size_add(
                    out.carrierBluePlaneOffset,
                    out.planeBytes,
                    carrierEnd) ||
                carrierEnd != out.carrierBytes) {
                failedFact = "carrier_end_offset";
                return false;
            }
        }
        if (!checked_size_add(
                out.filterBytes,
                out.carrierBytes,
                out.requestedBytes)) {
            failedFact = "requested_bytes";
            return false;
        }
        return true;
    }

    void set_scatter_halation_exact_admission_failure(
        const Root::PreparationIdentity& identity,
        const ScatterHalationFrameDescriptor& descriptor,
        int width,
        int height,
        JuicerCuda::ScatterHalationCarrierSource carrierSource,
        const ScatterHalationAllocationShape& shape,
        std::size_t admittedBytes,
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        const char* failedFact,
        const std::string& cause,
        std::string& outError) {
        outError = "ExactAdmissionFailure route=";
        outError += Spektrafilm::scan_route_key(identity.scanRoute);
        outError += " domain=FilmLinearExposure film_profile_key=";
        outError += identity.filmProfileKey;
        outError += " film_profile_asset_version_token=" +
                    std::to_string(identity.filmProfileAssetVersionToken);
        outError += " backend=Exact descriptor_recipe_hash=" +
                    std::to_string(
                        static_cast<unsigned long long>(descriptor.recipeHash));
        outError += " full_frame_width=" + std::to_string(width);
        outError += " full_frame_height=" + std::to_string(height);
        if (shape.filterBytes != 0) {
            outError += " logical_filter_scratch_bytes=" +
                        std::to_string(
                            static_cast<unsigned long long>(shape.filterBytes));
        }
        outError += " carrier_source=";
        outError += scatter_halation_carrier_source_label(carrierSource);
        if (shape.carrierBytes != 0 ||
            carrierSource ==
                JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes) {
            outError += " logical_carrier_bytes=" +
                        std::to_string(
                            static_cast<unsigned long long>(shape.carrierBytes));
        }
        if (shape.requestedBytes != 0) {
            outError += " physical_bytes_requested=" +
                        std::to_string(
                            static_cast<unsigned long long>(shape.requestedBytes));
        }
        outError += " bytes_admitted_before_failure=" +
                    std::to_string(
                        static_cast<unsigned long long>(admittedBytes));
        outError += " device_id=" + std::to_string(contextKey.deviceId);
        outError += " context=" +
                    std::to_string(static_cast<unsigned long long>(
                        reinterpret_cast<std::uintptr_t>(
                            contextKey.contextOpaque)));
        outError += " context_epoch=" +
                    std::to_string(
                        static_cast<unsigned long long>(contextEpoch));
        outError += " attempted_block_shape=";
        outError +=
            carrierSource ==
                    JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes
                ? "filter_2_planes+carrier_3_planes"
                : "filter_2_planes";
        outError += " physical_allocation_count=" +
                    std::to_string(
                        static_cast<unsigned long long>(
                            shape.physicalAllocationCount));
        outError += " failed_fact=";
        outError += failedFact && failedFact[0] ? failedFact : "unknown";
        if (!cause.empty()) {
            outError += " cause=" + cause;
        }
    }

    bool derive_workspace_layout(
        const Root::PreparationIdentity& identity,
        const Spektrafilm::SpatialDirDescriptor* spatialDir,
        const Scanner::ScannerPostEffectsDescriptor* scannerPostEffects,
        const Spektrafilm::DiffusionFrameSetDescriptor* diffusionFrameSet,
        const ScatterHalationFrameDescriptor* scatterHalationDescriptor,
        const std::optional<Spektrafilm::LegacyGrainFrameDescriptor>& legacyGrain,
        bool grainV2,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& effects,
        Spektrafilm::ProfilePolarity capturePolarity,
        int requestedWidth,
        int requestedHeight,
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        Root::PreparedCudaFrame::WorkspaceDerivation& out,
        std::string& outError) {
        out = std::monostate{};
        JuicerCuda::ResourceManager::ScratchRequestAttachments attachments;
        Spektrafilm::LegacyGrainScratchShape grainShape = Spektrafilm::LegacyGrainScratchShape::None;
        JuicerCuda::ResourceManager::ActiveDirScratch::Layout source;
        JuicerCuda::ResourceManager::ActiveDirScratch::Layout target;
        std::uint64_t dirHash = 0;
        if (requestedWidth <= 0 || requestedHeight <= 0) {
            outError = "ResourceDescriptorMismatch phase=workspace field=extent";
            return false;
        }

        const bool routeDiffusionActive =
            diffusionFrameSet &&
            (diffusionFrameSet->camera.has_value() ||
             diffusionFrameSet->enlarger.has_value());
        if (diffusionFrameSet &&
            (diffusionFrameSet->hash == 0 ||
             diffusionFrameSet->fullFrame.width != requestedWidth ||
             diffusionFrameSet->fullFrame.height != requestedHeight)) {
            outError =
                "ResourceDescriptorMismatch phase=workspace field=diffusion_extent";
            return false;
        }
        if (routeDiffusionActive) {
            attachments.needOptics = true;
        }

        const std::uint64_t scatterHalationRecipeHash =
            identity.scatterHalationHash;
        if (scatterHalationRecipeHash == 0) {
            if (scatterHalationDescriptor) {
                outError =
                    "ResourceDescriptorMismatch phase=scatter_halation_admission field=recipe_hash expected=0 actual=" +
                    std::to_string(static_cast<unsigned long long>(
                        scatterHalationDescriptor->recipeHash));
                return false;
            }
        } else {
            if (!scatterHalationDescriptor) {
                outError =
                    "MissingRequiredResource phase=scatter_halation_admission field=descriptor recipe_hash=" +
                    std::to_string(static_cast<unsigned long long>(
                        scatterHalationRecipeHash));
                return false;
            }
            if (scatterHalationDescriptor->recipeHash == 0 ||
                scatterHalationDescriptor->recipeHash !=
                    scatterHalationRecipeHash) {
                outError =
                    "ResourceDescriptorMismatch phase=scatter_halation_admission field=recipe_hash expected=" +
                    std::to_string(static_cast<unsigned long long>(
                        scatterHalationRecipeHash)) +
                    " actual=" +
                    std::to_string(static_cast<unsigned long long>(
                        scatterHalationDescriptor->recipeHash));
                return false;
            }

            const bool cameraCarrier =
                diffusionFrameSet && diffusionFrameSet->camera.has_value();
            const JuicerCuda::ScatterHalationCarrierSource carrierSource =
                cameraCarrier
                    ? JuicerCuda::ScatterHalationCarrierSource::
                          CameraDiffusionStagePlanes
                    : JuicerCuda::ScatterHalationCarrierSource::
                          DedicatedPreparedPlanes;
            ScatterHalationAllocationShape shape{};
            const char* failedFact = nullptr;
            if (!derive_scatter_halation_allocation_shape(
                    requestedWidth,
                    requestedHeight,
                    carrierSource,
                    true,
                    shape,
                    failedFact)) {
                set_scatter_halation_exact_admission_failure(
                    identity,
                    *scatterHalationDescriptor,
                    requestedWidth,
                    requestedHeight,
                    carrierSource,
                    shape,
                    0,
                    contextKey,
                    contextEpoch,
                    failedFact,
                    {},
                    outError);
                return false;
            }
            if (cameraCarrier &&
                diffusionFrameSet->camera->stage !=
                    Spektrafilm::DiffusionLinearStage::CameraFilmLinear) {
                set_scatter_halation_exact_admission_failure(
                    identity,
                    *scatterHalationDescriptor,
                    requestedWidth,
                    requestedHeight,
                    carrierSource,
                    shape,
                    0,
                    contextKey,
                    contextEpoch,
                    "camera_carrier_stage",
                    {},
                    outError);
                return false;
            }
            attachments.needOptics = true;
        }

        if (spatialDir && spatialDir->hash != 0) {
            if (spatialDir->renderExtent.width != requestedWidth ||
                spatialDir->renderExtent.height != requestedHeight) {
                outError =
                    "ResourceDescriptorMismatch phase=workspace field=spatial_dir_extent";
                return false;
            }
            dirHash = spatialDir->hash;
            source = {spatialDir->scratchTier, spatialDir->planeRoles};
            target = {spatialDir->targetScratchTier, spatialDir->targetPlaneRoles};
            attachments.needSharedTmp = spatialDir->planeRoles.filterTempPlanes > 0;
        }
        if (grainV2) {
            attachments.needOptics = true;
        }
        if (legacyGrain.has_value()) {
            if (legacyGrain->renderExtent.width != requestedWidth ||
                legacyGrain->renderExtent.height != requestedHeight) {
                outError =
                    "ResourceDescriptorMismatch phase=workspace field=grain_extent";
                return false;
            }
            if (!legacyGrain->active || legacyGrain->hash == 0 ||
                legacyGrain->scratchShape == Spektrafilm::LegacyGrainScratchShape::None) {
                outError = "ResourceDescriptorMismatch phase=grain_workspace field=descriptor";
                return false;
            }
            switch (legacyGrain->scratchShape) {
                case Spektrafilm::LegacyGrainScratchShape::Streamed:
                case Spektrafilm::LegacyGrainScratchShape::StreamedShared:
                case Spektrafilm::LegacyGrainScratchShape::StreamedLayers:
                case Spektrafilm::LegacyGrainScratchShape::StreamedLayersShared:
                    break;
                default:
                    outError = "ResourceDescriptorMismatch phase=grain_workspace field=scratch_shape";
                    return false;
            }
            grainShape = legacyGrain->scratchShape;
            attachments.needOptics = true;
            attachments.needSharedTmp = true;
            attachments.needBlurred = true;
        }
        if (effects.has_value()) {
            if (effects->renderExtent.width != requestedWidth ||
                effects->renderExtent.height != requestedHeight ||
                (!effects->filmActive && !effects->gateOutputActive)) {
                outError =
                    "ResourceDescriptorMismatch phase=effects_workspace field=descriptor";
                return false;
            }
            attachments.needOptics = true;
            attachments.needGateTransmittance = effects->gateTransmittanceActive;
            attachments.needFilmDustTransmittance = effects->filmDust.slotProbability > 0.0f;
            attachments.gateWidth = effects->gateWidth;
            attachments.gateHeight = effects->gateHeight;
        }
        if (scannerPostEffects && scannerPostEffects->active()) {
            attachments.needOptics = true;
            attachments.needSharedTmp = true;
            attachments.needBlurred =
                attachments.needBlurred ||
                (scannerPostEffects->glareActive &&
                 scannerPostEffects->glareBlurSigmaPx > 0.0f);
        }
        // Positive final develop retains DIR correction as an input while the scanner writes RGB.
        if (!routeDiffusionActive && attachments.needOptics && dirHash != 0 &&
            capturePolarity != Spektrafilm::ProfilePolarity::Positive) {
            attachments.aliasScannerRgbFromSpatialDirFiltered = true;
            target = source;
            target.roles.cachedLogRawPlanes = 0;
        }
        if (!attachments.needOptics && dirHash == 0) {
            return true;
        }
        JuicerCuda::ResourceManager::DirScratchRequest dir;
        if (dirHash != 0) {
            const auto active = JuicerCuda::ResourceManager::ActiveDirScratch::create(dirHash, source, target);
            if (!active) {
                outError = "invalid frame scratch request";
                return false;
            }
            dir = *active;
        }
        const auto layout = WorkspaceLayout::create({requestedWidth, requestedHeight}, dir, attachments, grainShape);
        if (!layout) {
            // Other construction failures retain checkpoint timing and earlier preparation failures.
            out = Root::PreparedCudaFrame::WorkspaceConstructionFailure{};
            return true;
        }
        out = *layout;
        return true;
    }


} // namespace JuicerProcess
