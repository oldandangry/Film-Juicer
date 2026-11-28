#pragma once

#include "ProcessRoot.h"

namespace JuicerProcess {

    using JuicerCuda::ResourceManager::WorkspaceLayout;

    struct ScatterHalationAllocationShape {
        std::size_t planeBytes = 0;
        std::size_t filterBytes = 0;
        std::size_t carrierBytes = 0;
        std::size_t requestedBytes = 0;
        std::size_t filterSecondPlaneOffset = 0;
        std::size_t carrierGreenPlaneOffset = 0;
        std::size_t carrierBluePlaneOffset = 0;
        std::size_t physicalAllocationCount = 0;
    };

    bool checked_size_multiply(
        std::size_t left,
        std::size_t right,
        std::size_t& out) noexcept;
    bool checked_size_add(
        std::size_t left,
        std::size_t right,
        std::size_t& out) noexcept;
    bool derive_scatter_halation_allocation_shape(
        int width,
        int height,
        JuicerCuda::ScatterHalationCarrierSource carrierSource,
        bool spatial,
        ScatterHalationAllocationShape& out,
        const char*& failedFact) noexcept;
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
        std::string& outError);
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
        std::string& outError);

} // namespace JuicerProcess
