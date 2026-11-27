#include "Cuda/Resources/ResourceInternals.h"
#include "PreparedCudaFrameState.h"
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

    namespace {

        bool dir_plane_roles_equal(
            const Spektrafilm::DirScratchPlaneRoles& a,
            const Spektrafilm::DirScratchPlaneRoles& b) noexcept {
            return a.rawCorrectionPlanes == b.rawCorrectionPlanes &&
                   a.filteredCorrectionPlanes == b.filteredCorrectionPlanes &&
                   a.filterTempPlanes == b.filterTempPlanes &&
                   a.cachedLogRawPlanes == b.cachedLogRawPlanes;
        }

        bool spatial_dir_descriptor_has_strict_yvv(
            const Spektrafilm::SpatialDirDescriptor& descriptor) noexcept {
            if (descriptor.scratchTier != Spektrafilm::DirScratchTier::Tier1IChannels ||
                descriptor.targetScratchTier != Spektrafilm::DirScratchTier::Tier2) {
                return false;
            }
            for (int component = 0; component < descriptor.filterPlan.componentCount; ++component) {
                const Spektrafilm::DirGaussianComponentPlan& plan =
                    descriptor.filterPlan.components[static_cast<std::size_t>(component)];
                if (plan.weight > 0.0f &&
                    plan.referenceOperator == Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect) {
                    return true;
                }
            }
            return false;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_low_scratch_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 3;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 2;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_single_temp_sequential_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 3;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 1;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_component_streamed_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 1;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 1;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_cached_lograw_alias_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 3;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 3;
            roles.cachedLogRawPlanes = 3;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_cached_lograw_bg_alias_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles = strict_yvv_cached_lograw_alias_plane_roles();
            roles.cachedLogRawPlanes = 2;
            return roles;
        }

        std::optional<WorkspaceLayout> strict_yvv_workspace_candidate(
            const WorkspaceLayout& base, Spektrafilm::DirScratchTier buildTier, const Spektrafilm::DirScratchPlaneRoles& buildRoles, Spektrafilm::DirScratchTier targetTier, const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
            return base.with_dir_layout({buildTier, buildRoles}, {targetTier, targetRoles});
        }

        bool spatial_dir_alias_target_uses_build_roles(const WorkspaceLayout& layout) noexcept {
            const auto& request = layout.allocation();
            return layout.scanner_storage() == JuicerCuda::ResourceManager::ScannerStorage::DirFiltered &&
                   request.spatialDirTargetScratchTier == request.spatialDirScratchTier &&
                   dir_plane_roles_equal(request.spatialDirTargetPlaneRoles, request.spatialDirPlaneRoles);
        }

        [[maybe_unused]] const char* strict_yvv_candidate_label(
            const Spektrafilm::DirScratchPlaneRoles& roles) noexcept {
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 3 &&
                roles.cachedLogRawPlanes == 3) {
                return "strict_yvv_channels_aliased_forward_cached_lograw";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 3 &&
                roles.cachedLogRawPlanes == 2) {
                return "strict_yvv_channels_aliased_forward_cached_lograw_bg";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 3 &&
                roles.cachedLogRawPlanes == 0) {
                return "strict_yvv_channels_aliased_forward";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 2) {
                return "strict_yvv_low_scratch_pair";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 1) {
                return "strict_yvv_single_temp_sequential";
            }
            if (roles.rawCorrectionPlanes == 1 &&
                roles.filterTempPlanes == 1) {
                return "strict_yvv_component_streamed";
            }
            return "unknown";
        }

#if JUICER_DIAGNOSTICS_COMPILED
        void append_workspace_request_trace_fields(
            std::string& msg,
            const char* prefix,
            const WorkspaceLayout& request) {
            const Spektrafilm::DirScratchPlaneRoles& roles = request.allocation().spatialDirPlaneRoles;
            const Spektrafilm::DirScratchPlaneRoles& targetRoles =
                request.allocation().spatialDirTargetPlaneRoles;
            const std::string fieldPrefix = prefix && prefix[0] ? std::string(prefix) : "request";
            auto append_i32 = [&](const char* name, int value) {
                msg += " ";
                msg += fieldPrefix;
                msg += "_";
                msg += name;
                msg += "=";
                msg += std::to_string(value);
            };
            auto append_u64 = [&](const char* name, std::uint64_t value) {
                msg += " ";
                msg += fieldPrefix;
                msg += "_";
                msg += name;
                msg += "=";
                msg += std::to_string(static_cast<unsigned long long>(value));
            };
            auto append_bool = [&](const char* name, bool value) {
                append_i32(name, value ? 1 : 0);
            };
            auto append_cstr = [&](const char* name, const char* value) {
                msg += " ";
                msg += fieldPrefix;
                msg += "_";
                msg += name;
                msg += "=";
                msg += value && value[0] ? value : "unspecified";
            };

            append_bool("need_optics", request.allocation().needOptics);
            append_bool("need_spatial_dir", request.allocation().needSpatialDir);
            append_u64("spatial_dir_descriptor_hash", request.allocation().spatialDirDescriptorHash);
            append_cstr("spatial_dir_scratch_tier", Spektrafilm::to_cstr(request.allocation().spatialDirScratchTier));
            append_i32("raw_correction_planes", roles.rawCorrectionPlanes);
            append_i32("filtered_correction_planes", roles.filteredCorrectionPlanes);
            append_i32("filter_temp_planes", roles.filterTempPlanes);
            append_i32("cached_log_raw_planes", roles.cachedLogRawPlanes);
            append_cstr("spatial_dir_target_scratch_tier", Spektrafilm::to_cstr(request.allocation().spatialDirTargetScratchTier));
            append_i32("target_raw_correction_planes", targetRoles.rawCorrectionPlanes);
            append_i32("target_filtered_correction_planes", targetRoles.filteredCorrectionPlanes);
            append_i32("target_filter_temp_planes", targetRoles.filterTempPlanes);
            append_i32("target_cached_log_raw_planes", targetRoles.cachedLogRawPlanes);
            append_i32("requested_width", request.allocation().requestedWidth);
            append_i32("requested_height", request.allocation().requestedHeight);
            append_bool("need_blurred", request.allocation().needBlurred);
            append_bool(
                "alias_scanner_rgb_from_spatial_dir_filtered",
                request.allocation().aliasScannerRgbFromSpatialDirFiltered);
            append_bool("need_aux", (request.grain_storage() != JuicerCuda::ResourceManager::GrainStorage::Absent));
            append_bool("need_shared_tmp", request.allocation().needSharedTmp);
            append_bool(
                "need_grain_frame_uniforms",
                (request.grain_storage() != JuicerCuda::ResourceManager::GrainStorage::Absent));
            append_bool("need_grain_layer_work", request.grain_uses_layers());
            append_bool("need_grain_shared", request.grain_uses_shared());
            append_bool("need_gate_transmittance", request.allocation().needGateTransmittance);
        }

        void trace_workspace_request_mismatch(
            const JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const WorkspaceLayout& activeRequest,
            const WorkspaceLayout& requestedRequest) {
            if (!JTRACE_ENABLED(1)) {
                return;
            }
            std::string msg = "event=frame_scratch_request_mismatch transaction_id=";
            msg += std::to_string(static_cast<unsigned long long>(transaction.transactionId));
            msg += " snapshot_id=";
            msg += std::to_string(static_cast<unsigned long long>(transaction.snapshot.snapshotId));
            append_workspace_request_trace_fields(msg, "active", activeRequest);
            append_workspace_request_trace_fields(msg, "requested", requestedRequest);
            JTRACE("MSADM", msg);
        }
#endif

    } // namespace

    static bool valid_film_grain_frame(const FjFilmGrain& input, int width, int height) {
        using namespace JuicerCuda::FilmGrain;
        const auto axis_supported = [renderedTexels = static_cast<double>(input.pitch)](int size) {
            return (size - 0.5) * renderedTexels < kPlacementCenterLimit;
        };
        const bool calibrated = input.sample_step[0] == 0.0 && input.sample_step[1] == 0.0;
        if (std::ranges::any_of(input.calibrated_border, [calibrated](std::uint8_t cells) {
                return cells > (calibrated ? 1 : 0);
            }))
            return false;
        const auto& [left, top, right, bottom] = input.calibrated_border;
        if (!std::isfinite(input.pitch) || input.pitch < kMinRenderedFootprint || input.pitch > kMaxFootprint ||
            !std::isfinite(input.temporal_fraction) || input.temporal_fraction < 0.F || input.temporal_fraction >= 1.F ||
            !std::isfinite(input.gain) || input.gain < 0.F ||
            !std::isfinite(input.chroma_fraction) || input.chroma_fraction < 0.F || input.chroma_fraction > 1.F ||
            width <= left + right || height <= top + bottom ||
            !axis_supported(width - left - right) ||
            !axis_supported(height - top - bottom)) {
            return false;
        }
        if (calibrated)
            return input.sample_pitch == input.pitch;
        if (!std::isfinite(input.sample_pitch) || input.sample_pitch < kMinProxySampleFootprint ||
            input.sample_pitch > kMaxFootprint)
            return false;

        enum class Axis : std::uint8_t {
            X,
            Y
        };
        const auto sample_axis_supported = [&input, width, height](Axis axis) {
            const auto index = static_cast<std::size_t>(axis);
            const int size = axis == Axis::X ? width : height;
            const double extent = input.sample_extent[index];
            const double step = input.sample_step[index];
            const double start = input.sample_origin[index];
            if (!std::isfinite(extent) || extent <= 0.0 ||
                !std::isfinite(step) || step <= 0.0 || !std::isfinite(start) ||
                (std::ceil(extent) - 0.5) * input.sample_pitch >= kPlacementCenterLimit)
                return false;

            const auto raw_edge = [step, start](double pixel) {
                return std::fma(pixel, step, start);
            };
            if (!std::isfinite(raw_edge(0.0)) || !std::isfinite(raw_edge(size)))
                return false;
            const auto edge = [&raw_edge, extent](double pixel) {
                return std::clamp(raw_edge(pixel), 0.0, extent);
            };
            const double first = edge(0.0);
            const double last = edge(size);
            const double firstEnd = edge(1.0);
            const double lastStart = edge(size - 1.0);
            // One outward-rounded empty border can abut either Full edge,
            // but the axis and the neighbouring pixel must have support.
            return last > first &&
                   (firstEnd > first || (size > 1 && edge(2.0) > first)) &&
                   (lastStart < last || (size > 1 && edge(size - 2.0) < last));
        };
        return sample_axis_supported(Axis::X) && sample_axis_supported(Axis::Y);
    }

    bool validate_visual_grain_descriptor(
        const Root::PreparationIdentity& identity,
        const std::optional<Spektrafilm::VisualGrainFrameDescriptor>& descriptor,
        std::string& outError) {
        if (!identity.grainActive) {
            if (descriptor.has_value()) {
                outError =
                    "ResourceDescriptorMismatch phase=grain_descriptor field=inactive_descriptor";
                return false;
            }
            return true;
        }
        if (!descriptor.has_value()) {
            outError =
                "MissingRequiredResource phase=grain_descriptor field=active_descriptor";
            return false;
        }
        const Spektrafilm::VisualGrainFrameDescriptor& frame = *descriptor;
        if (!frame.active ||
            frame.hash == 0 ||
            frame.recipeHash != identity.grainHash ||
            frame.capturePolarity != identity.capturePolarity ||
            frame.densityCurvesLayersHash !=
                identity.grainDensityLayersHash ||
            frame.scratchShape ==
                Spektrafilm::VisualGrainScratchShape::None) {
            outError =
                "ResourceDescriptorMismatch phase=grain_descriptor field=identity";
            return false;
        }
        return true;
    }

    bool validate_film_juicer_effects_descriptor(
        const Root::PreparationIdentity& identity,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        std::string& outError) {
        if (!identity.effectsActive) {
            if (identity.effectsHash != 0 || descriptor.has_value()) {
                outError =
                    "ResourceDescriptorMismatch phase=effects_descriptor field=inactive_descriptor";
                return false;
            }
            return true;
        }
        if (!descriptor.has_value()) {
            outError =
                "MissingRequiredResource phase=effects_descriptor field=active_descriptor";
            return false;
        }
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& frame =
            *descriptor;
        if (frame.hash == 0 || frame.recipeHash != identity.effectsHash) {
            outError =
                "ResourceDescriptorMismatch phase=effects_descriptor field=identity";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::State::allocate_device_bytes(
        void** outPtr,
        std::size_t bytes,
        const char* label,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!outPtr || *outPtr || bytes == 0 || !label || !label[0] ||
            !resources || !resources->deviceLedger) {
            outError.diagnostic = "invalid prepared-frame CUDA allocation request";
            return false;
        }

        JuicerCuda::TrackedDeviceAllocations::node_type node;
        if (!JuicerCuda::allocate_tracked_device_buffer(*resources, bytes, label, node, outError)) {
            return false;
        }
        void* const allocated = node.key();
        auto insertion = deviceAllocationRecords.insert(std::move(node));
        if (!insertion.inserted) {
            JuicerCuda::Failure ignored;
            (void)insertion.node.mapped().free_after_completion(ignored);
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "prepared-frame allocation pointer collision");
            return false;
        }
        *outPtr = allocated;
        return true;
    }

    bool Root::PreparedCudaFrame::State::retire_device_bytes(
        void* ptr,
        void* cudaStreamOpaque,
        const char* label,
        JuicerCuda::Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!ptr) {
            return true;
        }
        const auto allocation = deviceAllocationRecords.find(ptr);
        if (allocation == deviceAllocationRecords.end()) {
            outError.diagnostic = "unregistered prepared-frame CUDA retirement";
            return false;
        }
        const std::size_t recordedBytes = static_cast<std::size_t>(
            allocation->second.bytes());
        if (recordedBytes == 0) {
            outError.diagnostic = "invalid prepared-frame CUDA retirement record";
            return false;
        }
        if (!JuicerCuda::retire_frame_scratch_allocation(
                *resources,
                ptr,
                recordedBytes,
                allocation->second,
                cudaStreamOpaque,
                label,
                outError)) {
            return false;
        }
        deviceAllocationRecords.erase(allocation);
        return true;
    }

    bool Root::PreparedCudaFrame::State::free_device_bytes_now(
        void* ptr,
        JuicerCuda::Failure& outError) noexcept {
        outError = {};
        if (!ptr) {
            return true;
        }
        try {
            const auto allocation = deviceAllocationRecords.find(ptr);
            if (allocation == deviceAllocationRecords.end()) {
                outError.diagnostic = "unregistered prepared-frame CUDA free";
                return false;
            }
            if (!allocation->second.free_after_completion(outError)) {
                return false;
            }
            deviceAllocationRecords.erase(allocation);
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError.diagnostic = "prepared-frame CUDA free bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool Root::PreparedCudaFrame::State::transfer_remaining_device_allocations(
        JuicerCuda::Failure& outError) noexcept {
        try {
            outError = {};
            if (deviceAllocationRecords.empty()) {
                return true;
            }
            if (!resources) {
                outError.diagnostic =
                    "CUDA resources unavailable for remaining frame allocation transfer";
                return false;
            }
            while (!deviceAllocationRecords.empty()) {
                const auto allocation = deviceAllocationRecords.begin();
                void* ptr = allocation->first;
                JuicerCuda::Failure retireError;
                if (retire_device_bytes(
                        ptr,
                        lastCudaStreamOpaque,
                        "remaining frame allocation",
                        retireError)) {
                    continue;
                }

                const cudaStream_t stream = lastCudaStreamOpaque
                                                ? reinterpret_cast<cudaStream_t>(
                                                      lastCudaStreamOpaque)
                                                : nullptr;
                const bool completionCertain =
                    cudaStreamSynchronize(stream) == cudaSuccess;
                if (completionCertain) {
                    JuicerCuda::Failure freeError;
                    if (free_device_bytes_now(ptr, freeError)) {
                        continue;
                    }
                }

                auto allocationRecord = deviceAllocationRecords.extract(ptr);
                JuicerCuda::Failure adoptionError;
                if (!JuicerCuda::adopt_failed_frame_allocation_record(
                        *resources,
                        allocationRecord,
                        completionCertain,
                        adoptionError)) {
                    if (!allocationRecord.empty()) {
                        deviceAllocationRecords.insert(
                            std::move(allocationRecord));
                    }
                    outError.status = adoptionError.status;
                    outError.diagnostic = adoptionError.diagnostic.empty()
                                              ? "remaining frame allocation adoption failed"
                                              : adoptionError.diagnostic;
                    return false;
                }
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError.diagnostic =
                    "remaining frame allocation transfer bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool Root::PreparedCudaFrame::State::prepare_diffusion_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        const Root::PreparationIdentity& identity,
        const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
        int requestedWidth,
        int requestedHeight,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        diffusionExecutionDescriptor = {};
        if (!frameSet) {
            if (!resources) {
                outError.diagnostic =
                    "MissingRequiredResource component=diffusion field=context_resources";
                return false;
            }
            return JuicerCuda::Diffusion::trim_inactive_diffusion_resources(
                resources->diffusion,
                contextKey,
                outError);
        }
        const Spektrafilm::ScanRoute route = identity.scanRoute;
        const Spektrafilm::ScanRouteMetadata& routeMetadata =
            Spektrafilm::scan_route_metadata(route);
        const bool printRoute = routeMetadata.printRoute;
        if (!resources || !resources->deviceLedger ||
            transaction.contextEpoch == 0 ||
            transaction.resolvedMemoryBudget.allocationCapBytes == 0 ||
            frameSet->route != route ||
            frameSet->fullFrame.width != requestedWidth ||
            frameSet->fullFrame.height != requestedHeight ||
            requestedWidth <= 0 || requestedHeight <= 0 ||
            (!printRoute && frameSet->enlarger.has_value())) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=diffusion field=preparation_request";
            return false;
        }

        const auto appendDiffusionFailureIdentity = [&]() noexcept {
            try {
                const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                    resources->deviceLedger->snapshot();
                outError.diagnostic += " device_id=" + std::to_string(contextKey.deviceId);
                outError.diagnostic += " context=" + std::to_string(
                                                         static_cast<unsigned long long>(
                                                             reinterpret_cast<std::uintptr_t>(
                                                                 contextKey.contextOpaque)));
                outError.diagnostic += " epoch=" + std::to_string(
                                                       static_cast<unsigned long long>(
                                                           transaction.contextEpoch));
                outError.diagnostic += " device_budget_bytes=" + std::to_string(
                                                                     static_cast<unsigned long long>(
                                                                         transaction.resolvedMemoryBudget.deviceBudgetBytes));
                outError.diagnostic += " allocation_cap_bytes=" + std::to_string(
                                                                      static_cast<unsigned long long>(
                                                                          transaction.resolvedMemoryBudget.allocationCapBytes));
                outError.diagnostic += " ledger_reserved_bytes=" + std::to_string(
                                                                       static_cast<unsigned long long>(
                                                                           ledgerSnapshot.reservedBytes));
                outError.diagnostic += " ledger_committed_bytes=" + std::to_string(
                                                                        static_cast<unsigned long long>(
                                                                            ledgerSnapshot.committedBytes));
                outError.diagnostic += " ledger_retiring_bytes=" + std::to_string(
                                                                       static_cast<unsigned long long>(
                                                                           ledgerSnapshot.retiringBytes));
                outError.diagnostic += " frame_set_hash=" + std::to_string(
                                                                static_cast<unsigned long long>(
                                                                    frameSet->hash));
                outError.diagnostic += " execution_hash=" + std::to_string(
                                                                static_cast<unsigned long long>(
                                                                    diffusionExecutionDescriptor.hash));
                outError.diagnostic += " fft_width=" +
                                       std::to_string(diffusionExecutionDescriptor.layout.width);
                outError.diagnostic += " fft_height=" +
                                       std::to_string(diffusionExecutionDescriptor.layout.height);
                outError.diagnostic += " unique_spectra=" + std::to_string(
                                                                diffusionExecutionDescriptor.uniqueSpectrumCount);
                for (std::size_t index = 0;
                     index < diffusionExecutionDescriptor.uniqueSpectrumCount;
                     ++index) {
                    outError.diagnostic += " spectrum_key_" + std::to_string(index) + "=" +
                                           std::to_string(static_cast<unsigned long long>(
                                               diffusionExecutionDescriptor.spectrumKeys[index].hash));
                }
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        };
        if (!Spektrafilm::build_diffusion_execution_descriptor(
                *frameSet,
                transaction.contextEpoch,
                transaction.resolvedMemoryBudget.allocationCapBytes,
                diffusionExecutionDescriptor,
                outError.diagnostic)) {
            appendDiffusionFailureIdentity();
            return false;
        }
        if (!JuicerCuda::Diffusion::prepare_diffusion_resources(
                resources->diffusion,
                contextKey,
                transaction.contextEpoch,
                resources->deviceLedger,
                *frameSet,
                diffusionExecutionDescriptor,
                cudaStreamOpaque,
                diffusionLease,
                outError)) {
            appendDiffusionFailureIdentity();
            diffusionExecutionDescriptor = {};
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::State::prepare_scatter_halation_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        const Root::PreparationIdentity& identity,
        const ScatterHalationFrameDescriptor* descriptor,
        const JuicerCuda::HighlightBoostDescriptor* boost,
        const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
        int requestedWidth,
        int requestedHeight,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!descriptor && !boost) {
            return true;
        }
        if (!resources || !transaction.active || transaction.committed ||
            scatterHalationDescriptor || scatterHalationFilterBlock ||
            scatterHalationCarrierBlock) {
            outError.diagnostic =
                "prepared scatter-halation resource state is not empty";
            return false;
        }

        const bool cameraCarrier = frameSet && frameSet->camera.has_value();
        const JuicerCuda::ScatterHalationCarrierSource carrierSource =
            cameraCarrier
                ? JuicerCuda::ScatterHalationCarrierSource::
                      CameraDiffusionStagePlanes
                : JuicerCuda::ScatterHalationCarrierSource::
                      DedicatedPreparedPlanes;
        ScatterHalationAllocationShape shape{};
        const auto fail_admission = [&](const char* fact, std::size_t allocated, const JuicerCuda::Failure& underlying) {
            outError.status = underlying.status;
            if (descriptor) {
                set_scatter_halation_exact_admission_failure(
                    identity, *descriptor, requestedWidth, requestedHeight, carrierSource, shape, allocated, contextKey, transaction.contextEpoch, fact, underlying.diagnostic, outError.diagnostic);
            } else {
                outError.diagnostic = "HighlightBoostAdmissionFailure field=" + std::string(fact);
                if (!underlying.diagnostic.empty()) {
                    outError.diagnostic += " | " + underlying.diagnostic;
                }
            }
            return false;
        };
        const char* failedFact = nullptr;
        if (!derive_scatter_halation_allocation_shape(
                requestedWidth,
                requestedHeight,
                carrierSource,
                descriptor != nullptr,
                shape,
                failedFact)) {
            return fail_admission(failedFact, 0, {});
        }

        if (cameraCarrier) {
            const JuicerCuda::Diffusion::DiffusionPreparedView diffusionView =
                diffusionLease.view();
            if (!diffusionLease.active() || !diffusionView.active ||
                diffusionView.executionDescriptor.contextEpoch !=
                    transaction.contextEpoch ||
                diffusionView.executionDescriptor.frameSetHash !=
                    frameSet->hash ||
                resources->ownerContextKey != contextKey ||
                resources->contextEpoch != transaction.contextEpoch) {
                return fail_admission("camera_diffusion_exact_context_lease", 0, {});
            }
        }

        void* filterBlock = nullptr;
        JuicerCuda::Failure allocationError;
        if (descriptor && !allocate_device_bytes(
                              &filterBlock,
                              shape.filterBytes,
                              "scatter halation filter scratch",
                              allocationError)) {
            return fail_admission("filter_allocation_block", 0, allocationError);
        }

        void* carrierBlock = nullptr;
        if (carrierSource ==
                JuicerCuda::ScatterHalationCarrierSource::
                    DedicatedPreparedPlanes &&
            !allocate_device_bytes(
                &carrierBlock,
                shape.carrierBytes,
                descriptor ? "scatter halation dedicated carrier" : "highlight boost dedicated carrier",
                allocationError)) {
            return fail_admission("carrier_allocation_block", shape.filterBytes, allocationError);
        }

        if (filterBlock) {
            auto* filterBytes = static_cast<std::byte*>(filterBlock);
            scatterHalationFilterBlock = filterBlock;
            scatterHalationFilterTemp = reinterpret_cast<float*>(filterBytes);
            scatterHalationWeightedAccumulation = reinterpret_cast<float*>(
                filterBytes + shape.filterSecondPlaneOffset);
        }
        scatterHalationCarrierSource = carrierSource;
        if (carrierBlock) {
            auto* carrierBytes = static_cast<std::byte*>(carrierBlock);
            scatterHalationCarrierBlock = carrierBlock;
            scatterHalationCarrier.redSensitive =
                reinterpret_cast<float*>(carrierBytes);
            scatterHalationCarrier.greenSensitive = reinterpret_cast<float*>(
                carrierBytes + shape.carrierGreenPlaneOffset);
            scatterHalationCarrier.blueSensitive = reinterpret_cast<float*>(
                carrierBytes + shape.carrierBluePlaneOffset);
            scatterHalationCarrier.rowStrideFloats =
                static_cast<std::size_t>(requestedWidth);
        }
        scatterHalationFrameWidth = requestedWidth;
        scatterHalationFrameHeight = requestedHeight;
        if (descriptor) {
            scatterHalationDescriptor = *descriptor;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::State::prepare_highlight_boost(
        const JuicerCuda::HighlightBoostDescriptor& descriptor,
        JuicerCuda::Failure& outError) {
        auto& stage = highlightBoostStage;
        void* workspace = nullptr;
        if (!allocate_device_bytes(&workspace, sizeof(JuicerCuda::HighlightBoostWorkspace), "highlight boost reduction and status", outError)) {
            return false;
        }
        stage.workspace = static_cast<JuicerCuda::HighlightBoostWorkspace*>(workspace);
        void* pinned = nullptr;
        if (!JuicerCuda::allocate_frame_status_staging(
                *resources, sizeof(JuicerCuda::HighlightBoostDisposition), pinned, stage.event, outError)) {
            return false;
        }
        stage.hostStatus = static_cast<JuicerCuda::HighlightBoostDisposition*>(pinned);
        stage.descriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scatter_halation_resources(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        auto& boost = highlightBoostStage;
        if (!scatterHalationDescriptor && !scatterHalationFilterBlock &&
            !scatterHalationCarrierBlock && !boost.workspace) {
            return true;
        }
        if (!resources) {
            outError.diagnostic =
                "CUDA resources unavailable for scatter-halation release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque =
            cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        if (boost.hostStatus) {
            // Move the status event to the end of this frame's use before
            // transferring it. Freeing pinned memory can wait on later work.
            auto* const retireStream = reinterpret_cast<cudaStream_t>(retireStreamOpaque);
            const bool needsFence = boost.eventRecorded || boost.copyPending;
            const auto recordStatus = needsFence ? cudaEventRecord(boost.event, retireStream) : cudaSuccess;
            const bool released = JuicerCuda::retire_frame_status_staging(
                *resources, boost.hostStatus, needsFence && recordStatus == cudaSuccess, recordStatus != cudaSuccess, outError);
            boost.hostStatus = nullptr;
            boost.event = nullptr;
            if (!released) {
                return false;
            }
            if (recordStatus != cudaSuccess) {
                JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(recordStatus), "highlight boost retirement event record failed");
                return false;
            }
        }
        if (boost.workspace) {
            if (!retire_device_bytes(boost.workspace, retireStreamOpaque, "highlight boost reduction and status", outError)) {
                return false;
            }
            boost = {};
        }
        if (scatterHalationCarrierBlock) {
            void* carrierBlock = scatterHalationCarrierBlock;
            if (!retire_device_bytes(
                    carrierBlock,
                    retireStreamOpaque,
                    scatterHalationDescriptor ? "scatter halation dedicated carrier" : "highlight boost dedicated carrier",
                    outError)) {
                return false;
            }
            scatterHalationCarrierBlock = nullptr;
            scatterHalationCarrier = {};
        }
        if (scatterHalationFilterBlock) {
            void* filterBlock = scatterHalationFilterBlock;
            if (!retire_device_bytes(
                    filterBlock,
                    retireStreamOpaque,
                    "scatter halation filter scratch",
                    outError)) {
                return false;
            }
            scatterHalationFilterBlock = nullptr;
            scatterHalationFilterTemp = nullptr;
            scatterHalationWeightedAccumulation = nullptr;
        }
        scatterHalationDescriptor.reset();
        scatterHalationFrameWidth = 0;
        scatterHalationFrameHeight = 0;
        scatterHalationCarrierSource =
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes;
        return true;
    }

    bool Root::PreparedCudaFrame::State::allocate_scan_error_stage(JuicerCuda::Failure& outError) {
        outError = {};
        return JuicerCuda::acquire_scan_error_buffer(*resources, scanErrorStage.buffer, outError);
    }

    bool Root::PreparedCudaFrame::State::release_scan_error_stage_after_use(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        ScanErrorFrameStage& stage = scanErrorStage;
        if (!stage.buffer.host) {
            return true;
        }
        if (!resources) {
            outError.diagnostic = "CUDA resources unavailable for frame scan-error stage release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        auto* const stream = reinterpret_cast<cudaStream_t>(lastCudaStreamOpaque);
        auto* const event = stage.buffer.event;
        outError.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        auto waitResult = cudaSuccess;
        if (stage.flagUsed) {
            const auto recordStatus = stage.readbackPending ? cudaSuccess : cudaEventRecord(event, stream);
            if (recordStatus == cudaSuccess) {
                if (JuicerCuda::retain_scan_error_readback(
                        *resources, stage.buffer, stage.identity, stage.readbackPending, outError)) {
                    stage = {};
                    return true;
                }
                waitResult = cudaEventSynchronize(event);
            } else {
                JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(recordStatus), "record scan error buffer retirement failed");
                waitResult = JuicerCuda::ResourceOperations::synchronize_stream(stream);
            }
        }
        if (waitResult == cudaSuccess) {
            JuicerCuda::Failure releaseError;
            if (!JuicerCuda::release_scan_error_buffer(*resources, stage.buffer.host, releaseError)) {
                JuicerCuda::retain_first_failure(outError, std::move(releaseError));
            }
        } else {
            JuicerCuda::record_cleanup_context_loss(outError, JuicerCuda::runtime_failure_status(waitResult));
            JuicerCuda::quarantine_unfenced_frame_use(*resources);
        }
        stage = {};
        return outError.status.category == FJ_STATUS_SUCCESS;
    }

    bool Root::PreparedCudaFrame::State::allocate_auto_exposure_workspace(
        const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
        JuicerCuda::Failure& outError) {
        outError = {};
        free_auto_exposure_workspace_now();
        const int meterWidth = descriptor.previewWidth;
        const int meterHeight = descriptor.previewHeight;
        if (meterWidth <= 0 || meterHeight <= 0) {
            outError.diagnostic = "auto-exposure meter dimensions invalid";
            return false;
        }

        const int blockX = 16;
        const int blockY = 16;
        const int gridX = (meterWidth + blockX - 1) / blockX;
        const int gridY = (meterHeight + blockY - 1) / blockY;
        const bool needsHistogram = descriptor.method == Spektrafilm::AutoExposureMethod::Median;
        const bool needsWeights = descriptor.method == Spektrafilm::AutoExposureMethod::CenterWeighted;
        const bool needsPartials = !needsHistogram;
        const int neededPartials = needsPartials ? gridX * gridY : 0;
        if (needsPartials && neededPartials <= 0) {
            outError.diagnostic = "auto-exposure partial count invalid";
            return false;
        }

        auto alloc_device = [&](auto*& ptr, std::size_t bytes, const char* label) -> bool {
            return allocate_device_bytes(
                reinterpret_cast<void**>(&ptr),
                bytes,
                label,
                outError);
        };

        AutoExposureFrameWorkspace next{};
        if (!alloc_device(next.deviceState.exposureScale, sizeof(float), "frame auto-exposure scale")) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsHistogram &&
            (!alloc_device(next.scratch.maxYBits, sizeof(unsigned int), "frame auto-exposure maxYBits") ||
             !alloc_device(next.scratch.histogram, sizeof(unsigned int) * 2048u, "frame auto-exposure histogram"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsWeights &&
            (!alloc_device(next.scratch.weightsX, static_cast<std::size_t>(meterWidth) * sizeof(float), "frame auto-exposure weightsX") ||
             !alloc_device(next.scratch.weightsY, static_cast<std::size_t>(meterHeight) * sizeof(float), "frame auto-exposure weightsY"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsPartials &&
            (!alloc_device(
                 next.scratch.partialsA,
                 static_cast<std::size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial),
                 "frame auto-exposure partialsA") ||
             !alloc_device(
                 next.scratch.partialsB,
                 static_cast<std::size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial),
                 "frame auto-exposure partialsB"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }

        next.scratch.partialCapacity = neededPartials;
        next.weightsXCapacity = needsWeights ? meterWidth : 0;
        next.weightsYCapacity = needsWeights ? meterHeight : 0;
        next.weightsWidth = 0;
        next.weightsHeight = 0;
        next.method = descriptor.method;
        next.keyHash = 0;
        next.active = true;
        autoExposureWorkspace = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_auto_exposure_workspace_after_use(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        if (!workspace.active) {
            return true;
        }

        try {
            if (!resources) {
                outError.diagnostic = "CUDA resources unavailable for frame auto-exposure retire";
                return false;
            }

            remember_stream(cudaStreamOpaque);
            void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
            bool retiredAll = true;
            auto retire_ptr = [&](auto*& ptr, const char* label) {
                if (!ptr || !retiredAll) {
                    return;
                }
                void* raw = ptr;
                JuicerCuda::Failure localError;
                if (retire_device_bytes(
                        raw,
                        retireStreamOpaque,
                        label,
                        localError)) {
                    ptr = nullptr;
                    return;
                }
                retiredAll = false;
                outError.status = localError.status;
                if (localError.diagnostic.empty()) {
                    outError.diagnostic = std::string(label ? label : "frame auto-exposure buffer") +
                                          " retire failed";
                } else {
                    outError.diagnostic = localError.diagnostic;
                }
            };

            retire_ptr(workspace.deviceState.exposureScale, "frame auto-exposure scale");
            retire_ptr(workspace.scratch.maxYBits, "frame auto-exposure maxYBits");
            retire_ptr(workspace.scratch.histogram, "frame auto-exposure histogram");
            retire_ptr(
                workspace.scratch.weightsX,
                "frame auto-exposure weightsX");
            retire_ptr(
                workspace.scratch.weightsY,
                "frame auto-exposure weightsY");
            retire_ptr(
                workspace.scratch.partialsA,
                "frame auto-exposure partialsA");
            retire_ptr(
                workspace.scratch.partialsB,
                "frame auto-exposure partialsB");

            if (retiredAll) {
                workspace = AutoExposureFrameWorkspace{};
                return true;
            }
        } catch (...) {
            outError.diagnostic = "frame auto-exposure retire threw";
        }
        if (outError.diagnostic.empty()) {
            outError.diagnostic = "frame auto-exposure retire failed";
        }
        return false;
    }

    void Root::PreparedCudaFrame::State::free_auto_exposure_workspace_now() noexcept {
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        JuicerCuda::Failure freeError;
        if (workspace.scratch.partialsA) {
            (void)free_device_bytes_now(workspace.scratch.partialsA, freeError);
        }
        if (workspace.scratch.partialsB) {
            (void)free_device_bytes_now(workspace.scratch.partialsB, freeError);
        }
        if (workspace.scratch.maxYBits) {
            (void)free_device_bytes_now(workspace.scratch.maxYBits, freeError);
        }
        if (workspace.scratch.histogram) {
            (void)free_device_bytes_now(workspace.scratch.histogram, freeError);
        }
        if (workspace.scratch.weightsX) {
            (void)free_device_bytes_now(workspace.scratch.weightsX, freeError);
        }
        if (workspace.scratch.weightsY) {
            (void)free_device_bytes_now(workspace.scratch.weightsY, freeError);
        }
        if (workspace.deviceState.exposureScale) {
            (void)free_device_bytes_now(
                workspace.deviceState.exposureScale,
                freeError);
        }
        workspace = AutoExposureFrameWorkspace{};
    }

    bool Root::PreparedCudaFrame::State::submit_frame_use_event(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!resources || !transaction.active || transaction.committed) {
            outError.diagnostic = "prepared frame is not active for use-event submission";
            return false;
        }
        if (frameUseEventSubmitted) {
            return true;
        }
        remember_stream(cudaStreamOpaque);

        outError.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        const auto fence = [&](JuicerCuda::Resources& target, const char* label, bool& fenced) {
            if (fenced) {
                return;
            }
            JuicerCuda::Failure recordError;
            if (JuicerCuda::record_frame_use_event(target, cudaStreamOpaque, label, recordError)) {
                fenced = true;
                return;
            }
            const auto syncError = JuicerCuda::ResourceOperations::synchronize_stream(
                reinterpret_cast<cudaStream_t>(cudaStreamOpaque));
            fenced = syncError == cudaSuccess;
            if (fenced && !JuicerCuda::context_loss(recordError)) {
#if JUICER_DIAGNOSTICS_COMPILED
                if (JTRACE_ENABLED(1)) {
                    try {
                        JTRACE("CUDALIFE", std::string("event=frame_use_event_fallback_sync label=") + label);
                    } catch (...) {
                        JuicerLogging::discard_current_exception();
                    }
                }
#endif
                return;
            }
            if (!fenced) {
                JuicerCuda::quarantine_unfenced_frame_use(target);
                std::array<char, 768> diagnostic{};
                const auto* reason = cudaGetErrorString(syncError);
                std::snprintf(diagnostic.data(), diagnostic.size(), "%s | cudaStreamSynchronize(frame use) failed: %s", recordError.diagnostic.empty() ? "frame use event submission failed" : recordError.diagnostic.c_str(), reason ? reason : "(unknown)");
                JuicerCuda::set_failure(recordError, recordError.status, diagnostic.data());
                JuicerCuda::record_cleanup_context_loss(recordError, JuicerCuda::runtime_failure_status(syncError));
            }
            JuicerCuda::retain_first_failure(outError, std::move(recordError));
        };

        fence(*resources, "frame use", frameResourcesFenced);
        if (grainStaticResources && visualGrainPrepared) {
            fence(*grainStaticResources, "grain-static use", grainStaticResourcesFenced);
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        frameUseEventSubmitted = true;
#if JUICER_DIAGNOSTICS_COMPILED
        if (grainStaticResources && visualGrainPrepared &&
            JTRACE_ENABLED(1)) {
            JTRACE(
                "GRAINLIFE",
                "event=grain_static_use_event_recorded before_owner_reset=1");
        }
#endif
        return true;
    }

    bool Root::PreparedCudaFrame::State::ensure_scratch_workspace(
        const WorkspaceLayout& request,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!resources) {
            outError.diagnostic = "CUDA resources unavailable for frame scratch lease";
            return false;
        }
        if (scratchWorkspace.retainedLeaseActive) {
            if (scratchWorkspace.admittedLayout && !(*scratchWorkspace.admittedLayout == request)) {
#if JUICER_DIAGNOSTICS_COMPILED
                trace_workspace_request_mismatch(transaction, *scratchWorkspace.admittedLayout, request);
#endif
                outError.diagnostic = "prepared frame scratch workspace request mismatch";
                return false;
            }
            return true;
        }

        remember_stream(cudaStreamOpaque);
        if (!JuicerCuda::acquire_retained_frame_scratch_lease(
                *resources,
                transaction.leaseGeneration,
                cudaStreamOpaque,
                outError)) {
            return false;
        }
        // Cleanup keeps its projection even if the allocation command fails.
        scratchWorkspace.postFrameLayout = request;
        scratchWorkspace.retainedLeaseActive = true;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scratch_workspace_after_use(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.retainedLeaseActive) {
            return true;
        }
        if (!resources) {
            outError.diagnostic = "CUDA resources unavailable for frame scratch release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        if (!JuicerCuda::release_retained_frame_scratch_lease(
                *resources,
                transaction.leaseGeneration,
                outError)) {
            if (outError.diagnostic.empty()) {
                outError.diagnostic = "retained frame scratch lease release failed";
            }
            return false;
        }
        workspace = FrameScratchWorkspace{};
        return true;
    }

    Root::PreparedCudaFrame::PreparedCudaFrame() noexcept = default;

    Root::PreparedCudaFrame::PreparedCudaFrame(std::unique_ptr<State> state) noexcept
        : _state(std::move(state)) {
    }

    Root::PreparedCudaFrame::~PreparedCudaFrame() {
        abort();
    }

    Root::PreparedCudaFrame::PreparedCudaFrame(PreparedCudaFrame&& other) noexcept
        : _state(std::move(other._state)) {
    }

    Root::PreparedCudaFrame& Root::PreparedCudaFrame::operator=(PreparedCudaFrame&& other) noexcept {
        if (this != &other) {
            abort();
            _state = std::move(other._state);
        }
        return *this;
    }

    Root::PreparedCudaFrame::WorkspaceLeaseMarker::WorkspaceLeaseMarker(
        const WorkspaceDerivation& request,
        std::uint64_t leaseGeneration) noexcept
        : _request(request), _leaseGeneration(leaseGeneration) {
    }

    bool Root::PreparedCudaFrame::WorkspaceLeaseMarker::active() const noexcept {
        return !std::holds_alternative<std::monostate>(_request) && _leaseGeneration != 0;
    }

    const WorkspaceLayout& Root::PreparedCudaFrame::WorkspaceLeaseMarker::layout() const noexcept {
        return *std::get_if<WorkspaceLayout>(&_request);
    }

    const JuicerCuda::ResourceManager::SubmissionSnapshot& Root::PreparedCudaFrame::admitted_snapshot() const noexcept {
        return _state->transaction.snapshot;
    }

    std::uint64_t Root::PreparedCudaFrame::admitted_context_epoch() const noexcept {
        return _state->transaction.contextEpoch;
    }

    bool Root::PreparedCudaFrame::active() const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed;
    }

    JuicerCuda::ResourceManager::ScratchRequestDescriptor
    Root::PreparedCudaFrame::State::post_frame_scratch_request_descriptor() const noexcept {
        if (!scratchWorkspace.retainedLeaseActive || !scratchWorkspace.postFrameLayout) {
            return JuicerCuda::ResourceManager::ScratchRequestDescriptor{};
        }
        return scratchWorkspace.postFrameLayout->allocation();
    }

    Root::PreparedCudaFrame::WorkspaceLeaseMarker
    Root::PreparedCudaFrame::workspace_lease() const noexcept {
        if (!active() || std::holds_alternative<std::monostate>(_state->requestedWorkspace)) {
            return WorkspaceLeaseMarker{};
        }
        return WorkspaceLeaseMarker(_state->requestedWorkspace, _state->transaction.leaseGeneration);
    }

    const WorkspaceLayout& Root::PreparedCudaFrame::active_workspace_layout(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        if (_state && _state->scratchWorkspace.retainedLeaseActive && _state->scratchWorkspace.admittedLayout &&
            _state->scratchWorkspace.admittedLayout->allocation().needSpatialDir &&
            workspace.layout().allocation().needSpatialDir &&
            _state->scratchWorkspace.admittedLayout->allocation().spatialDirDescriptorHash ==
                workspace.layout().allocation().spatialDirDescriptorHash) {
            return *_state->scratchWorkspace.admittedLayout;
        }
        return workspace.layout();
    }

    bool Root::PreparedCudaFrame::workspace_marker_matches_current_frame(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed &&
               workspace.active() && std::holds_alternative<WorkspaceLayout>(workspace._request) &&
               workspace._leaseGeneration == _state->transaction.leaseGeneration;
    }

    bool Root::PreparedCudaFrame::validate_workspace_lease_marker(
        const WorkspaceLeaseMarker& workspace,
        std::string& outError) const {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        if (!workspace.active()) {
            outError = "prepared frame workspace marker is not active";
            return false;
        }
        if (workspace._leaseGeneration != _state->transaction.leaseGeneration) {
            outError = "prepared frame workspace marker does not match current lease";
            return false;
        }
        if (std::holds_alternative<WorkspaceConstructionFailure>(workspace._request)) {
            outError = "invalid frame scratch request";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::finish(void* cudaStreamOpaque, JuicerCuda::Failure& outError) {
        outError = {};
        if (!_state || !_state->root || !_state->transaction.active || _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active";
            return false;
        }
        if (_state->highlightBoostStage.workspace && !_state->highlightBoostStage.accepted) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "highlight boost status was not accepted");
            return false;
        }
        _state->remember_stream(cudaStreamOpaque);
        if (!_state->submit_frame_use_event(cudaStreamOpaque, outError)) {
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "frame use-event submission failed");
            }
            return false;
        }
        const auto postFrameScratchRequest = _state->post_frame_scratch_request_descriptor();
        JuicerCuda::Failure releaseError;
        if (!_state->release_scatter_halation_resources(
                cudaStreamOpaque,
                releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "scatter-halation resource release failed");
            }
            return false;
        }
        if (_state->diffusionLease.active() &&
            !JuicerCuda::Diffusion::release_diffusion_resources(
                _state->resources->diffusion,
                _state->diffusionLease,
                cudaStreamOpaque,
                false,
                releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "diffusion resource release failed");
            }
            return false;
        }
        if (!_state->release_scan_error_stage_after_use(cudaStreamOpaque, releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "frame scan-error stage release failed");
            }
            return false;
        }
        if (!_state->release_scratch_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "frame scratch workspace release failed");
            }
            return false;
        }
        if (!_state->release_auto_exposure_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "frame auto-exposure retire failed");
            }
            if (JTRACE_ENABLED(1)) {
                std::string msg = "frame_auto_exposure_retire_failed finish=1 error=";
                msg += outError.diagnostic;
                JTRACE("CUDA", msg);
            }
            return false;
        }
        if (!_state->transfer_remaining_device_allocations(releaseError)) {
            outError = std::move(releaseError);
            if (outError.diagnostic.empty()) {
                JuicerCuda::set_failure(outError, outError.status, "remaining frame allocation transfer failed");
            }
            return false;
        }
        if (!JuicerCuda::ResourceManager::command_shed_post_frame_scratch(
                _state->transaction,
                *_state->resources,
                postFrameScratchRequest,
                cudaStreamOpaque,
                "command_shed_post_frame_scratch_finish",
                outError)) {
            return false;
        }

        if (!JuicerCuda::ResourceManager::commit_submission(
                _state->transaction,
                cudaStreamOpaque,
                outError.diagnostic)) {
            return false;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (_state->grainStaticOwner && JTRACE_ENABLED(1)) {
            const std::string message =
                std::string("event=grain_static_finish_owner_reset use_event_recorded=") +
                (_state->frameUseEventSubmitted ? "1" : "0");
            JTRACE("GRAINLIFE", message);
        }
#endif
        _state->resources = nullptr;
        _state->grainStaticResources = nullptr;
        _state->resourceOwner.reset();
        _state->grainStaticOwner.reset();
        return true;
    }

    void Root::PreparedCudaFrame::abort() noexcept {
        if (_state) {
            const auto cleanup = [this](const char* stage, auto&& release) noexcept {
                JuicerCuda::Failure failure;
                try {
                    if (release(failure)) {
                        return;
                    }
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
                _state->recovery.observe(stage, failure);
#if JUICER_DIAGNOSTICS_COMPILED
                if (JTRACE_ENABLED(1)) {
                    try {
                        JTRACE("CUDA", std::string(stage) + " abort=1 error=" + failure.diagnostic);
                    } catch (...) {
                        JuicerLogging::discard_current_exception();
                    }
                }
#endif
            };
            auto* const stream = _state->lastCudaStreamOpaque;
            if (_state->resources && _state->transaction.active && !_state->transaction.committed) {
                cleanup("abort_frame_use_event", [&](auto& error) {
                    return _state->submit_frame_use_event(stream, error);
                });
            }
            const auto postFrameScratchRequest = _state->post_frame_scratch_request_descriptor();
            cleanup("scatter_halation_resource_release_failed", [&](auto& error) {
                return _state->release_scatter_halation_resources(stream, error);
            });
            if (_state->resources && _state->diffusionLease.active()) {
                cleanup("diffusion_resource_release_failed", [&](auto& error) {
                    return JuicerCuda::Diffusion::release_diffusion_resources(
                        _state->resources->diffusion, _state->diffusionLease, stream, true, error);
                });
            }
            cleanup("frame_scan_error_release_failed", [&](auto& error) {
                return _state->release_scan_error_stage_after_use(stream, error);
            });
            cleanup("frame_scratch_workspace_release_failed", [&](auto& error) {
                return _state->release_scratch_workspace_after_use(stream, error);
            });
            cleanup("frame_auto_exposure_retire_failed", [&](auto& error) {
                return _state->release_auto_exposure_workspace_after_use(stream, error);
            });
            cleanup("frame_remaining_allocation_transfer_failed", [&](auto& error) {
                return _state->transfer_remaining_device_allocations(error);
            });
            if (_state->root && _state->resources && _state->transaction.active && !_state->transaction.committed) {
                cleanup("frame_post_frame_scratch_shed_failed", [&](auto& error) {
                    return JuicerCuda::ResourceManager::command_shed_post_frame_scratch(
                        _state->transaction, *_state->resources, postFrameScratchRequest, stream, "command_shed_post_frame_scratch_abort", error);
                });
            }
        }
        if (_state && _state->root && _state->transaction.active && !_state->transaction.committed) {
            JuicerCuda::ResourceManager::rollback_submission(_state->transaction);
#if JUICER_DIAGNOSTICS_COMPILED
            try {
                if (!_state->transaction.active && JTRACE_ENABLED(3)) {
                    const auto& snapshot = _state->transaction.snapshot;
                    JTRACE_VERBOSE("CUDALIFE",
                                   "event=frame_rollback snapshot_id=" + std::to_string(snapshot.snapshotId) +
                                       " instance_token=" + std::to_string(snapshot.instanceToken.value) +
                                       " use_fenced=" + (_state->frameUseEventSubmitted ? "1" : "0"));
                }
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
#endif
        }
        if (_state) {
#if JUICER_DIAGNOSTICS_COMPILED
            try {
                if (_state->grainStaticOwner && JTRACE_ENABLED(1)) {
                    const std::string message =
                        std::string("event=grain_static_abort_owner_reset use_event_recorded=") +
                        (_state->frameUseEventSubmitted ? "1" : "0");
                    JTRACE("GRAINLIFE", message);
                }
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
#endif
            _state->resources = nullptr;
            _state->grainStaticResources = nullptr;
            _state->resourceOwner.reset();
            _state->grainStaticOwner.reset();
        }
    }

    bool Root::PreparedCudaFrame::stage_optical_workspace(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!validate_workspace_lease_marker(workspace, outError.diagnostic)) {
            return false;
        }
        const WorkspaceLayout& activeRequest = active_workspace_layout(workspace);
        if (!_state->ensure_scratch_workspace(activeRequest, cudaStreamOpaque, outError)) {
            const bool nonCapacityFailure =
                !JuicerCuda::allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"acquire_frame_scratch_workspace"},
                nonCapacityFailure
                    ? "CUDA frame scratch workspace acquisition failed"
                    : "CUDA frame scratch allocation capacity exhausted");
            return false;
        }
        const auto& scratchRequest = (activeRequest).allocation();
        if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            const bool nonCapacityFailure =
                !JuicerCuda::allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_optics_scratch"},
                nonCapacityFailure
                    ? "CUDA optics scratch allocation failed"
                    : "CUDA optics scratch allocation capacity exhausted");
            return false;
        }
        _state->scratchWorkspace.admittedLayout = activeRequest;
        _state->scratchWorkspace.postFrameLayout = activeRequest;
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_spatial_dir_resources(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!validate_workspace_lease_marker(workspace, outError.diagnostic) || !workspace.layout().allocation().needSpatialDir) {
            if (outError.diagnostic.empty()) {
                outError.diagnostic = "spatial DIR workspace was not admitted";
            }
            return false;
        }
        if (descriptor.hash == 0 || descriptor.dirRecipeHash == 0 ||
            descriptor.filterPlan.componentCount <= 0) {
            outError.diagnostic = "spatial DIR descriptor is invalid";
            return false;
        }
        const bool targetMatchesDescriptor =
            workspace.layout().allocation().spatialDirTargetScratchTier == descriptor.targetScratchTier &&
            dir_plane_roles_equal(
                workspace.layout().allocation().spatialDirTargetPlaneRoles,
                descriptor.targetPlaneRoles);
        const bool targetIsFusedScannerPostAlias =
            spatial_dir_alias_target_uses_build_roles(workspace.layout());
        if (workspace.layout().allocation().spatialDirDescriptorHash != descriptor.hash ||
            workspace.layout().allocation().spatialDirScratchTier != descriptor.scratchTier ||
            !dir_plane_roles_equal(workspace.layout().allocation().spatialDirPlaneRoles, descriptor.planeRoles) ||
            (!targetMatchesDescriptor && !targetIsFusedScannerPostAlias)) {
            outError.diagnostic = "spatial DIR workspace descriptor identity mismatch";
            return false;
        }
        if (!spatial_dir_roles_match_tier(descriptor.scratchTier, descriptor.planeRoles)) {
            outError.diagnostic = "spatial DIR descriptor requested unsupported scratch tier or plane roles";
            return false;
        }
        if (!spatial_dir_roles_match_tier(descriptor.targetScratchTier, descriptor.targetPlaneRoles)) {
            outError.diagnostic = "spatial DIR descriptor requested unsupported target scratch tier or plane roles";
            return false;
        }

        std::array<std::optional<WorkspaceLayout>, 6> scratchCandidates{};
        int scratchCandidateCount = 0;
        auto add_scratch_candidate = [&](const std::optional<WorkspaceLayout>& selected) {
            if (!selected) {
                return;
            }
            const auto& candidate = *selected;
            if (scratchCandidateCount >= static_cast<int>(scratchCandidates.size())) {
                return;
            }
            for (int index = 0; index < scratchCandidateCount; ++index) {
                const WorkspaceLayout& existing = *scratchCandidates[static_cast<std::size_t>(index)];
                if (existing.allocation().spatialDirScratchTier == candidate.allocation().spatialDirScratchTier &&
                    existing.allocation().spatialDirTargetScratchTier == candidate.allocation().spatialDirTargetScratchTier &&
                    dir_plane_roles_equal(existing.allocation().spatialDirPlaneRoles, candidate.allocation().spatialDirPlaneRoles) &&
                    dir_plane_roles_equal(
                        existing.allocation().spatialDirTargetPlaneRoles,
                        candidate.allocation().spatialDirTargetPlaneRoles)) {
                    return;
                }
            }
            scratchCandidates[static_cast<std::size_t>(scratchCandidateCount)] = candidate;
            ++scratchCandidateCount;
        };
        if (spatial_dir_descriptor_has_strict_yvv(descriptor)) {
            const bool aliasTarget =
                spatial_dir_alias_target_uses_build_roles(workspace.layout());
            if (aliasTarget) {
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBuild =
                    strict_yvv_cached_lograw_alias_plane_roles();
                Spektrafilm::DirScratchPlaneRoles cachedLogRawTarget = cachedLogRawBuild;
                add_scratch_candidate(strict_yvv_workspace_candidate(
                    workspace.layout(),
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBuild,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawTarget));
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBgBuild =
                    strict_yvv_cached_lograw_bg_alias_plane_roles();
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBgTarget = cachedLogRawBgBuild;
                add_scratch_candidate(strict_yvv_workspace_candidate(
                    workspace.layout(),
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBgBuild,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBgTarget));
            }
            add_scratch_candidate(workspace.layout());
            Spektrafilm::DirScratchPlaneRoles lowScratchBuild =
                strict_yvv_low_scratch_plane_roles();
            Spektrafilm::DirScratchPlaneRoles lowScratchTarget = lowScratchBuild;
            lowScratchTarget.cachedLogRawPlanes =
                aliasTarget ? 0 : descriptor.targetPlaneRoles.cachedLogRawPlanes;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace.layout(),
                Spektrafilm::DirScratchTier::Tier1IChannels,
                lowScratchBuild,
                aliasTarget ? Spektrafilm::DirScratchTier::Tier1IChannels : descriptor.targetScratchTier,
                lowScratchTarget));

            Spektrafilm::DirScratchPlaneRoles singleTempBuild =
                strict_yvv_single_temp_sequential_plane_roles();
            Spektrafilm::DirScratchPlaneRoles singleTempTarget = singleTempBuild;
            singleTempTarget.cachedLogRawPlanes =
                aliasTarget ? 0 : descriptor.targetPlaneRoles.cachedLogRawPlanes;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace.layout(),
                Spektrafilm::DirScratchTier::Tier1IChannels,
                singleTempBuild,
                aliasTarget ? Spektrafilm::DirScratchTier::Tier1IChannels : descriptor.targetScratchTier,
                singleTempTarget));

            Spektrafilm::DirScratchPlaneRoles streamedBuild =
                strict_yvv_component_streamed_plane_roles();
            Spektrafilm::DirScratchPlaneRoles streamedTarget = streamedBuild;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace.layout(),
                Spektrafilm::DirScratchTier::Tier1IChannels,
                streamedBuild,
                Spektrafilm::DirScratchTier::Tier1IChannels,
                streamedTarget));
        } else {
            add_scratch_candidate(workspace.layout());
        }
        if (scratchCandidateCount <= 0) {
            outError.diagnostic = "spatial DIR descriptor produced no supported scratch candidates";
            return false;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const Spektrafilm::DirScratchPlaneRoles& roles = descriptor.planeRoles;
            const Spektrafilm::DirScratchPlaneRoles& targetRoles = descriptor.targetPlaneRoles;
            std::string msg = "event=spatial_dir_admission descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
            msg += " dir_recipe_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.dirRecipeHash));
            msg += " route=";
            msg += descriptor.traceRouteLabel ? descriptor.traceRouteLabel : "unknown";
            msg += " scratch_tier=";
            msg += Spektrafilm::to_cstr(descriptor.scratchTier);
            msg += " target_scratch_tier=";
            msg += Spektrafilm::to_cstr(descriptor.targetScratchTier);
            msg += " scratch_source=pending";
            msg += " component_count=";
            msg += std::to_string(descriptor.filterPlan.componentCount);
            msg += " raw_correction_planes=";
            msg += std::to_string(roles.rawCorrectionPlanes);
            msg += " filtered_correction_planes=";
            msg += std::to_string(roles.filteredCorrectionPlanes);
            msg += " filter_temp_planes=";
            msg += std::to_string(roles.filterTempPlanes);
            msg += " cached_log_raw_planes=";
            msg += std::to_string(roles.cachedLogRawPlanes);
            msg += " target_raw_correction_planes=";
            msg += std::to_string(targetRoles.rawCorrectionPlanes);
            msg += " target_filtered_correction_planes=";
            msg += std::to_string(targetRoles.filteredCorrectionPlanes);
            msg += " target_filter_temp_planes=";
            msg += std::to_string(targetRoles.filterTempPlanes);
            msg += " target_cached_log_raw_planes=";
            msg += std::to_string(targetRoles.cachedLogRawPlanes);
            msg += " render_extent=";
            msg += std::to_string(descriptor.renderExtent.x) + "," +
                   std::to_string(descriptor.renderExtent.y) + "," +
                   std::to_string(descriptor.renderExtent.width) + "x" +
                   std::to_string(descriptor.renderExtent.height);
            msg += " full_frame_extent=";
            msg += std::to_string(descriptor.fullFrameExtent.x) + "," +
                   std::to_string(descriptor.fullFrameExtent.y) + "," +
                   std::to_string(descriptor.fullFrameExtent.width) + "x" +
                   std::to_string(descriptor.fullFrameExtent.height);
            msg += " filter_domain_extent=";
            msg += std::to_string(descriptor.filterDomainExtent.x) + "," +
                   std::to_string(descriptor.filterDomainExtent.y) + "," +
                   std::to_string(descriptor.filterDomainExtent.width) + "x" +
                   std::to_string(descriptor.filterDomainExtent.height);
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        bool scratchPrepared = false;
        std::optional<WorkspaceLayout> admittedLayout;
        JuicerCuda::Failure scratchAdmissionError;
        for (int candidateIndex = 0; candidateIndex < scratchCandidateCount; ++candidateIndex) {
            const WorkspaceLayout& candidate =
                *scratchCandidates[static_cast<std::size_t>(candidateIndex)];
#if JUICER_DIAGNOSTICS_COMPILED
            if (JTRACE_ENABLED(1)) {
                const Spektrafilm::DirScratchPlaneRoles& roles = candidate.allocation().spatialDirPlaneRoles;
                const Spektrafilm::DirScratchPlaneRoles& targetRoles =
                    candidate.allocation().spatialDirTargetPlaneRoles;
                std::string msg = "event=spatial_dir_scratch_candidate descriptor_hash=";
                msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
                msg += " candidate_index=";
                msg += std::to_string(candidateIndex);
                msg += " strict_yvv_shape=";
                msg += strict_yvv_candidate_label(roles);
                msg += " scratch_tier=";
                msg += Spektrafilm::to_cstr(candidate.allocation().spatialDirScratchTier);
                msg += " target_scratch_tier=";
                msg += Spektrafilm::to_cstr(candidate.allocation().spatialDirTargetScratchTier);
                msg += " raw_correction_planes=";
                msg += std::to_string(roles.rawCorrectionPlanes);
                msg += " filtered_correction_planes=";
                msg += std::to_string(roles.filteredCorrectionPlanes);
                msg += " filter_temp_planes=";
                msg += std::to_string(roles.filterTempPlanes);
                msg += " cached_log_raw_planes=";
                msg += std::to_string(roles.cachedLogRawPlanes);
                msg += " target_raw_correction_planes=";
                msg += std::to_string(targetRoles.rawCorrectionPlanes);
                msg += " target_filtered_correction_planes=";
                msg += std::to_string(targetRoles.filteredCorrectionPlanes);
                msg += " target_filter_temp_planes=";
                msg += std::to_string(targetRoles.filterTempPlanes);
                msg += " target_cached_log_raw_planes=";
                msg += std::to_string(targetRoles.cachedLogRawPlanes);
                JTRACE("DIR_DESCRIPTOR", msg);
            }
#endif
            if (!_state->scratchWorkspace.retainedLeaseActive &&
                !_state->ensure_scratch_workspace(candidate, cudaStreamOpaque, scratchAdmissionError)) {
                if (JuicerCuda::allocation_capacity_exhausted(
                        scratchAdmissionError) &&
                    candidateIndex + 1 < scratchCandidateCount) {
                    continue;
                }
                outError = scratchAdmissionError;
                break;
            }

            const auto& scratchRequest = candidate.allocation();
            _state->scratchWorkspace.postFrameLayout = candidate;
            const bool candidateSucceeded = JuicerCuda::ResourceManager::command_ensure_spatial_dir_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                scratchAdmissionError);
            if (!candidateSucceeded) {
                if (JuicerCuda::allocation_capacity_exhausted(
                        scratchAdmissionError) &&
                    candidateIndex + 1 < scratchCandidateCount) {
                    continue;
                }
                outError = scratchAdmissionError;
                break;
            }
            admittedLayout = candidate;
            _state->scratchWorkspace.admittedLayout = candidate;
            _state->scratchWorkspace.postFrameLayout = candidate;
            scratchPrepared = true;
            outError = {};
            break;
        }
        if (!scratchPrepared) {
            if (outError.diagnostic.empty()) {
                outError.diagnostic = scratchAdmissionError.diagnostic.empty()
                                          ? "spatial DIR scratch admission failed"
                                          : scratchAdmissionError.diagnostic;
            }
            const bool nonCapacityFailure =
                !JuicerCuda::allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_spatial_dir_scratch"},
                nonCapacityFailure
                    ? "CUDA spatial DIR scratch allocation failed"
                    : "CUDA spatial DIR scratch allocation capacity exhausted");
            return false;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const Spektrafilm::DirScratchPlaneRoles& roles = admittedLayout->allocation().spatialDirPlaneRoles;
            std::string msg = "event=spatial_dir_scratch_admission descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
            msg += " scratch_source=";
            msg += "retained";
            msg += " scratch_tier=";
            msg += Spektrafilm::to_cstr(admittedLayout->allocation().spatialDirScratchTier);
            msg += " strict_yvv_shape=";
            msg += strict_yvv_candidate_label(roles);
            msg += " raw_correction_planes=";
            msg += std::to_string(roles.rawCorrectionPlanes);
            msg += " filtered_correction_planes=";
            msg += std::to_string(roles.filteredCorrectionPlanes);
            msg += " filter_temp_planes=";
            msg += std::to_string(roles.filterTempPlanes);
            msg += " cached_log_raw_planes=";
            msg += std::to_string(roles.cachedLogRawPlanes);
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        for (int slot = 0;
             slot < Spektrafilm::DirFilterPlan::kMaxComponents;
             ++slot) {
            const bool componentPresent =
                slot < descriptor.filterPlan.componentCount;
            const Spektrafilm::DirGaussianComponentPlan& component =
                descriptor.filterPlan.components[static_cast<std::size_t>(slot)];
            const bool activeFir =
                componentPresent && component.weight > 0.0f &&
                component.referenceOperator ==
                    Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect;
            const bool activeIir =
                componentPresent && component.weight > 0.0f &&
                component.referenceOperator ==
                    Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect;
            if (!activeFir) {
                if (!JuicerCuda::clear_spatial_dir_kernel_binding(
                        *_state->resources,
                        _state->resources
                            ->spatialDirKernels[static_cast<std::size_t>(slot)],
                        outError)) {
                    _state->set_failure(
                        PreparedCudaFailureStage{
                            "command_clear_spatial_dir_kernel"},
                        "CUDA spatial DIR kernel binding clear failed");
                    return false;
                }
            } else {
                if (!JuicerCuda::ensure_spatial_dir_kernel(
                        *_state->resources,
                        _state->resources->spatialDirKernels[static_cast<std::size_t>(slot)],
                        component.radius,
                        component.sigmaPixels,
                        outError,
                        hostPreparation)) {
                    _state->set_failure(
                        PreparedCudaFailureStage{"command_ensure_spatial_dir_kernel"},
                        "CUDA spatial DIR kernel upload failed");
                    return false;
                }
            }
            if (!activeIir) {
                if (!JuicerCuda::clear_spatial_dir_boundary_binding(
                        *_state->resources,
                        _state->resources
                            ->spatialDirBoundaries[static_cast<std::size_t>(slot)],
                        cudaStreamOpaque,
                        outError)) {
                    _state->set_failure(
                        PreparedCudaFailureStage{
                            "command_clear_spatial_dir_boundary"},
                        "CUDA spatial DIR boundary binding clear failed");
                    return false;
                }
            } else if (!JuicerCuda::ensure_spatial_dir_boundary(
                           *_state->resources,
                           _state->resources
                               ->spatialDirBoundaries[static_cast<std::size_t>(slot)],
                           component,
                           descriptor.filterDomainExtent.width,
                           descriptor.filterDomainExtent.height,
                           cudaStreamOpaque,
                           outError,
                           hostPreparation)) {
                _state->set_failure(
                    PreparedCudaFailureStage{
                        "command_ensure_spatial_dir_boundary"},
                    "CUDA spatial DIR reflected-boundary preparation failed");
                return false;
            }
        }
        _state->spatialDirDescriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::stage_spatial_dir_cached_log_raw_for_final_develop(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!validate_workspace_lease_marker(workspace, outError.diagnostic)) {
            return false;
        }
        const WorkspaceLayout& effectiveRequest = active_workspace_layout(workspace);
        if (!effectiveRequest.allocation().needSpatialDir ||
            effectiveRequest.allocation().spatialDirTargetPlaneRoles.cachedLogRawPlanes == 0) {
            return true;
        }
        if (effectiveRequest.allocation().spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
            outError.diagnostic = "spatial DIR cached log raw final target roles invalid";
            return false;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive || !_state->scratchWorkspace.admittedLayout) {
            outError.diagnostic = "spatial DIR cached log raw staging requires an active scratch workspace";
            return false;
        }

        JuicerCuda::SpatialDirCachedLogRawStageStats stageStats{};
        _state->remember_stream(cudaStreamOpaque);
        const auto& scratchRequest = (effectiveRequest).allocation();
        const bool ok = JuicerCuda::ResourceManager::command_ensure_spatial_dir_cached_log_raw_stage(
            _state->transaction,
            *_state->resources,
            scratchRequest,
            cudaStreamOpaque,
            stageStats,
            outError);
        if (ok) {
            const auto& target = effectiveRequest.allocation();
            const auto staged = effectiveRequest.with_dir_layout(
                {target.spatialDirTargetScratchTier, target.spatialDirTargetPlaneRoles},
                {target.spatialDirTargetScratchTier, target.spatialDirTargetPlaneRoles});
            if (!staged) {
                outError.diagnostic = "invalid frame scratch request";
                return false;
            }
            _state->scratchWorkspace.admittedLayout = *staged;
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace.layout().allocation().spatialDirDescriptorHash;
            std::string msg = "event=spatial_dir_cached_log_raw_stage_for_final_develop";
            msg += " descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
            msg += " scratch_source=";
            msg += "retained";
            msg += " ok=";
            msg += ok ? "1" : "0";
            msg += " pending_scratch_bytes_before=";
            msg += std::to_string(static_cast<unsigned long long>(stageStats.pendingScratchBytesBefore));
            msg += " cached_log_raw_allocated_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(stageStats.cachedLogRawAllocatedBytes));
            if (!outError.diagnostic.empty()) {
                msg += " error=";
                msg += outError.diagnostic;
            }
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        if (!ok) {
            _state->set_failure(
                PreparedCudaFailureStage{"stage_spatial_dir_cached_log_raw_for_final_develop"},
                "CUDA spatial DIR cached log raw staging failed");
        }
        return ok;
    }

    bool Root::PreparedCudaFrame::prepare_scanner_post_effects(
        const Scanner::ScannerPostEffectsDescriptor& descriptor,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError = {};
        if (!descriptor.active()) {
            return true;
        }
        if (descriptor.lensBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerLensBlurKernel,
                descriptor.lensBlurSigmaPx,
                descriptor.lensBlurRadius,
                outError,
                hostPreparation)) {
            return false;
        }
        if (descriptor.unsharpSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerUnsharpKernel,
                descriptor.unsharpSigmaPx,
                descriptor.unsharpRadius,
                outError,
                hostPreparation)) {
            return false;
        }
        if (descriptor.glareActive && descriptor.glareBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerGlareKernel,
                descriptor.glareBlurSigmaPx,
                descriptor.glareBlurRadius,
                outError,
                hostPreparation)) {
            return false;
        }
        _state->scannerPostEffectsDescriptor = descriptor;
        return true;
    }

    Root::PreparedCudaFrame::ScanErrorStageResult Root::PreparedCudaFrame::prepare_scan_error_stage(
        int*& outScanErrorFlag,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        outScanErrorFlag = nullptr;
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active";
            return ScanErrorStageResult::Failed;
        }

        JuicerCuda::ScanErrorReadbackResult previousError{};
        if (!JuicerCuda::poll_scan_error_readbacks(
                *_state->resources,
                previousError,
                outError)) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_pending_readback"},
                "CUDA scan error validation failed");
            return ScanErrorStageResult::Failed;
        }
        if (previousError.status != 0) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_previous_readback"},
                "CUDA scan error validation failed");
            const bool deferredDirFailure = (previousError.status & (1 << 8)) != 0;
            const auto& producing = previousError.identity;
            const auto producingRoute = static_cast<Spektrafilm::ScanRoute>(producing.scanRoute);
            if (deferredDirFailure) {
                const int channels = (previousError.status >> 9) & 0x7;
                outError.diagnostic =
                    "earlier CUDA DIR submission failed component=dir route=" +
                    std::string(Spektrafilm::scan_route_key(producingRoute)) +
                    " profile=" + producing.profileKey +
                    " polarity=" + std::to_string(producing.capturePolarity) +
                    " recipe_hash=" + std::to_string(producing.dirRecipeHash) +
                    " descriptor_hash=" +
                    std::to_string(producing.dirDescriptorHash) +
                    " submission_id=" + std::to_string(producing.submissionId) +
                    " context_epoch=" +
                    std::to_string(producing.contextEpoch) +
                    " channel_mask=" + std::to_string(channels) +
                    " requirement=finite_defined_receiver_arithmetic";
            } else {
                const bool paper = Spektrafilm::scan_route_is_print(producingRoute);
                outError.diagnostic =
                    "earlier CUDA scan produced non-finite RGB route=" +
                    std::string(Spektrafilm::scan_route_key(producingRoute)) +
                    (paper ? " medium=paper profile=" : " medium=film profile=") +
                    (paper ? producing.paperProfileKey : producing.profileKey) +
                    " profile_asset_version=" + std::to_string(producing.scannedProfileAssetVersion) +
                    " submission_id=" + std::to_string(producing.submissionId) +
                    " frame_token=" + std::to_string(producing.frameToken) +
                    " scanner_hash=" + std::to_string(producing.scannerHash) +
                    " context_epoch=" + std::to_string(producing.contextEpoch);
            }
            JTRACE("CUDA", std::string("FATAL: ") + outError.diagnostic);
            return deferredDirFailure ? ScanErrorStageResult::DeferredDirFailure : ScanErrorStageResult::Failed;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        stage.identity.dirRecipeHash = _state->resources->filmDirHash;
        stage.identity.dirDescriptorHash = _state->spatialDirDescriptor.hash;
        outScanErrorFlag = stage.buffer.device;
        if (!outScanErrorFlag) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_missing"},
                "CUDA scan error validation failed");
            outError.diagnostic = "scan error flag missing after allocation";
            return ScanErrorStageResult::Failed;
        }

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        stage.flagUsed = true;
        const cudaError_t flagErr = cudaMemsetAsync(outScanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_memset"},
                "CUDA scan error validation failed");
            outError.status = JuicerCuda::runtime_failure_status(flagErr);
            outError.diagnostic = "CUDA scan error flag memset failed";
            return ScanErrorStageResult::Failed;
        }
        return ScanErrorStageResult::Ready;
    }

    bool Root::PreparedCudaFrame::finalize_scan_error_stage(
        int* scanErrorFlag,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active";
            return false;
        }
        if (!scanErrorFlag) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_missing"},
                "CUDA scan error validation failed");
            outError.diagnostic = "scan error flag missing after allocation";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        if (scanErrorFlag != stage.buffer.device) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_mismatch"},
                "CUDA scan error validation failed");
            outError.diagnostic = "scan error flag does not match prepared frame stage";
            return false;
        }

        cudaEvent_t scanEvent = stage.buffer.event;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        if (stage.buffer.host && scanEvent) {
            const cudaError_t flagErr = JuicerCuda::ResourceOperations::read_scan_error(stage.buffer.host, scanErrorFlag, stream);
            if (flagErr != cudaSuccess) {
                _state->set_failure(
                    PreparedCudaFailureStage{"scan_error_flag_readback"},
                    "CUDA scan error validation failed");
                outError.status = JuicerCuda::runtime_failure_status(flagErr);
                outError.diagnostic = "CUDA scan error flag readback failed";
                return false;
            }
            const cudaError_t evErr = cudaEventRecord(scanEvent, stream);
            if (evErr != cudaSuccess) {
                _state->set_failure(
                    PreparedCudaFailureStage{"scan_error_event_record"},
                    "CUDA scan error validation failed");
                outError.status = JuicerCuda::runtime_failure_status(evErr);
                outError.diagnostic = "CUDA scan error event record failed";
                return false;
            }
            stage.readbackPending = true;
        } else {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_staging_missing"},
                "CUDA scan error validation failed");
            outError.diagnostic = "scan error host/event staging missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::checkpoint_large_scratch_transition(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        const char* stageTag,
        JuicerCuda::Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!validate_workspace_lease_marker(workspace, outError.diagnostic)) {
            return false;
        }

        _state->remember_stream(cudaStreamOpaque);
        const auto& scratchRequest = workspace.layout().allocation();
        const char* failureStageTag =
            stageTag ? stageTag : "command_checkpoint_large_scratch_transition";
        if (!JuicerCuda::ResourceManager::command_checkpoint_large_scratch_transition(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                failureStageTag,
                outError)) {
            _state->set_failure(
                PreparedCudaFailureStage{failureStageTag},
                "CUDA large scratch transition checkpoint failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::build_gaussian_kernel_slot(
        JuicerCuda::Resources::DeviceGaussianKernel& kernel,
        float sigma,
        int radius,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError = {};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ensure_gaussian_kernel(
                *_state->resources,
                kernel,
                sigma,
                radius,
                outError,
                hostPreparation)) {
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_gaussian_kernel"},
                "CUDA gaussian kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_visual_grain_resources(
        Root& root,
        const JuicerCuda::StaticNoiseInput* noise,
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError = {};
        if (!_state || !_state->resources || !_state->transaction.active ||
            _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active";
            return false;
        }

        const bool active = _state->visualGrainDescriptor.has_value();
        if (active && !noise) {
            outError.diagnostic = "MissingRequiredResource phase=grain_static field=noise";
            return false;
        }

        const auto& snapshot = _state->transaction.snapshot;
        detail::GrainStaticMembershipChange membershipChange;
        if (!root.apply_grain_static_membership_and_copy_owner(
                deviceContextKey,
                _state->transaction.contextEpoch,
                _state->resources->deviceLedger,
                snapshot.instanceToken.value,
                snapshot.snapshotId,
                active,
                membershipChange,
                _state->grainStaticOwner,
                _state->grainStaticResources,
                outError)) {
            return false;
        }
        if (!active) {
            return true;
        }

        const ContextCudaResourceKey contextKey{
            deviceContextKey,
            _state->transaction.contextEpoch};
        auto fail_preparation = [&](const char* stageTag,
                                    const char* failurePrefix) {
            root.rollback_grain_static_membership(
                contextKey,
                membershipChange);
            _state->grainStaticResources = nullptr;
            _state->grainStaticOwner.reset();
            _state->set_failure(PreparedCudaFailureStage{stageTag}, failurePrefix);
            return false;
        };

        const Spektrafilm::VisualGrainFrameDescriptor& descriptor =
            *_state->visualGrainDescriptor;
        const bool uploaded = JuicerCuda::ensure_grain_static_assets_uploaded(*_state->grainStaticResources, *noise, cudaStreamOpaque, outError);
        if (!uploaded) {
            return fail_preparation(
                "ensure_grain_static_assets_uploaded",
                "CUDA grain-static upload failed");
        }

        const JuicerCuda::Resources& staticResources =
            *_state->grainStaticResources;
        if (!staticResources.stbnData || !staticResources.wangTilesData ||
            !staticResources.wangLutData ||
            staticResources.stbnWidth <= 0 ||
            staticResources.stbnHeight <= 0 ||
            staticResources.stbnFrames <= 0 ||
            staticResources.wangWidth <= 0 ||
            staticResources.wangHeight <= 0 ||
            staticResources.wangCount <= 0 ||
            staticResources.wangColors <= 0) {
            outError.diagnostic =
                "ResourceDescriptorMismatch phase=grain_static field=prepared_identity";
            return fail_preparation(
                "validate_grain_static_resources",
                "CUDA grain-static identity validation failed");
        }

        std::array<JuicerCuda::VisualGrainPreparedGaussianView, 12>
            preparedByIdentity{};
        std::array<std::uint64_t, 12> preparedIdentityHashes{};
        std::size_t preparedIdentityCount = 0;
        auto prepare_gaussian =
            [&](const Spektrafilm::VisualGrainGaussian& gaussian,
                JuicerCuda::Resources::DeviceGaussianKernel& kernel,
                JuicerCuda::VisualGrainPreparedGaussianView& outView) {
                outView =
                    JuicerCuda::VisualGrainPreparedGaussianView{};
                if (gaussian.radius <= 0) {
                    return gaussian.hash == 0;
                }
                if (gaussian.hash == 0) {
                    outError.diagnostic =
                        "ResourceDescriptorMismatch phase=grain_gaussian field=descriptor";
                    return false;
                }
                for (std::size_t index = 0;
                     index < preparedIdentityCount;
                     ++index) {
                    if (preparedIdentityHashes[index] == gaussian.hash) {
                        outView = preparedByIdentity[index];
                        return true;
                    }
                }
                if (!build_gaussian_kernel_slot(
                        kernel,
                        gaussian.sigmaPx,
                        gaussian.radius,
                        outError,
                        hostPreparation) ||
                    !kernel.weights || kernel.radius != gaussian.radius ||
                    kernel.sigma != gaussian.sigmaPx) {
                    if (outError.diagnostic.empty()) {
                        outError.diagnostic =
                            "ResourceDescriptorMismatch phase=grain_gaussian field=prepared_identity";
                    }
                    return false;
                }
                outView.weights = kernel.weights;
                outView.radius = kernel.radius;
                outView.sigma = kernel.sigma;
                outView.descriptorHash = gaussian.hash;
                outView.active = true;
                preparedIdentityHashes[preparedIdentityCount] =
                    gaussian.hash;
                preparedByIdentity[preparedIdentityCount] = outView;
                ++preparedIdentityCount;
                return true;
            };

        std::array<JuicerCuda::Resources::DeviceGaussianKernel*, 3>
            correlationSlots{
                &_state->resources->grainBlurKernel,
                &_state->resources->grainBlurKernelMid,
                &_state->resources->grainBlurKernelCoarse};
        for (std::size_t index = 0;
             index < correlationSlots.size();
             ++index) {
            if (!prepare_gaussian(
                    descriptor.correlation[index],
                    *correlationSlots[index],
                    _state->preparedGrainCorrelation[index])) {
                return fail_preparation(
                    "prepare_visual_grain_correlation",
                    "CUDA grain correlation Gaussian preparation failed");
            }
        }
        for (int layer = 0; layer < 3; ++layer) {
            for (int channel = 0; channel < 3; ++channel) {
                if (!prepare_gaussian(
                        descriptor.dyeCloud[static_cast<std::size_t>(layer)]
                                           [static_cast<std::size_t>(channel)],
                        _state->resources->grainDyeKernel[layer][channel],
                        _state->preparedGrainDyeCloud[layer][channel])) {
                    return fail_preparation(
                        "prepare_visual_grain_dye_cloud",
                        "CUDA grain dye-cloud Gaussian preparation failed");
                }
            }
        }

        _state->preparedGrainDensityLayers =
            JuicerCuda::VisualGrainPreparedDensityLayersView{};
        if (descriptor.densityCurvesLayersHash != 0) {
            JuicerCuda::Resources& resources = *_state->resources;
            if (!resources.hasDensityCurvesLayers ||
                resources.filmDensityLayersHash !=
                    descriptor.densityCurvesLayersHash) {
                outError.diagnostic =
                    "MissingRequiredResource phase=grain_density_layers field=prepared_identity";
                return fail_preparation(
                    "prepare_visual_grain_density_layers",
                    "CUDA grain density-layer preparation failed");
            }
            const JuicerCuda::DeviceCurve* baseCurvesCmy[3] = {
                &resources.densR,
                &resources.densG,
                &resources.densB};
            for (int channel = 0; channel < 3; ++channel) {
                const JuicerCuda::DeviceCurve& baseCurve =
                    *baseCurvesCmy[channel];
                if (!baseCurve.x || !baseCurve.y || baseCurve.n <= 0 ||
                    baseCurve.domainBegin < 0 ||
                    baseCurve.domainEnd < baseCurve.domainBegin ||
                    baseCurve.domainEnd >= baseCurve.n ||
                    resources.densityCurvesLayersChannelN[channel] !=
                        baseCurve.n) {
                    outError.diagnostic =
                        "ResourceDescriptorMismatch phase=grain_density_layers field=base_curve";
                    return fail_preparation(
                        "prepare_visual_grain_density_layers",
                        "CUDA grain density-layer preparation failed");
                }
                _state->preparedGrainDensityLayers
                    .baseCurvesCmy[channel] = {
                    baseCurve.x,
                    baseCurve.y,
                    baseCurve.n,
                    baseCurve.domainBegin,
                    baseCurve.domainEnd};
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int channel = 0; channel < 3; ++channel) {
                    const float* curve =
                        resources.densityCurvesLayers[layer][channel];
                    if (!curve) {
                        outError.diagnostic =
                            "MissingRequiredResource phase=grain_density_layers field=curve";
                        return fail_preparation(
                            "prepare_visual_grain_density_layers",
                            "CUDA grain density-layer preparation failed");
                    }
                    _state->preparedGrainDensityLayers
                        .curves[layer][channel] = curve;
                }
            }
            _state->preparedGrainDensityLayers.hash =
                descriptor.densityCurvesLayersHash;
            _state->preparedGrainDensityLayers.active = true;
        }
        _state->visualGrainPrepared = true;
        return true;
    }
    void Root::PreparedCudaFrame::mark_diffusion_work_enqueued() noexcept {
        if (!active() || !_state->diffusionLease.active()) {
            return;
        }
        JuicerCuda::Diffusion::mark_diffusion_work_enqueued(
            _state->diffusionLease);
    }

    bool Root::PreparedCudaFrame::release_diffusion_resources_after_use(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!active() || !_state->resources ||
            !_state->diffusionLease.active()) {
            outError.diagnostic = "prepared diffusion lease is not active";
            return false;
        }
        _state->remember_stream(cudaStreamOpaque);
        return JuicerCuda::Diffusion::release_diffusion_resources(
            _state->resources->diffusion,
            _state->diffusionLease,
            cudaStreamOpaque,
            false,
            outError);
    }

    bool Root::PreparedCudaFrame::enqueue_highlight_boost(
        JuicerCuda::CameraFilmLinearExposurePlanes carrier,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        if (!active()) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "highlight boost frame is inactive");
            return false;
        }
        auto& stage = _state->highlightBoostStage;
        if (!stage.workspace) {
            return true;
        }
        if (!stage.hostStatus || !stage.event || stage.eventRecorded || stage.copyPending) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "highlight boost frame status staging is invalid");
            return false;
        }
        _state->remember_stream(cudaStreamOpaque);
        auto* const stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const auto launch = JuicerCuda::launch_highlight_boost(
            stage.descriptor, carrier, _state->scatterHalationFrameWidth, _state->scatterHalationFrameHeight, stage.workspace, stream);
        if (launch.status != cudaSuccess) {
            using JuicerCuda::HighlightBoostLaunchStage;
            const char* label = "highlight boost launch failed";
            switch (launch.stage) {
                case HighlightBoostLaunchStage::Binding:
                    label = "highlight boost carrier binding failed";
                    break;
                case HighlightBoostLaunchStage::PartialReduction:
                    label = "highlight boost block reduction failed";
                    break;
                case HighlightBoostLaunchStage::FinalReduction:
                    label = "highlight boost final reduction failed";
                    break;
                case HighlightBoostLaunchStage::Apply:
                    label = "highlight boost apply failed";
                    break;
                case HighlightBoostLaunchStage::None:
                    break;
            }
            JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(launch.status), label);
            return false;
        }
        // Publish the staging dependency before the copy. After an API error,
        // retirement still needs a completion witness for that stream.
        stage.copyPending = true;
        cudaError_t copyStatus = cudaSuccess;
        if (copyStatus == cudaSuccess) {
            copyStatus = cudaMemcpyAsync(stage.hostStatus, &stage.workspace->disposition, sizeof(*stage.hostStatus), cudaMemcpyDeviceToHost, stream);
        }
        if (copyStatus != cudaSuccess) {
            JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(copyStatus), "highlight boost status readback failed");
            return false;
        }
        cudaError_t eventStatus = cudaSuccess;
        if (eventStatus == cudaSuccess) {
            eventStatus = cudaEventRecord(stage.event, stream);
        }
        if (eventStatus != cudaSuccess) {
            JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(eventStatus), "highlight boost status event record failed");
            return false;
        }
        stage.eventRecorded = true;
        return true;
    }

    bool Root::PreparedCudaFrame::check_highlight_boost(JuicerCuda::Failure& outError) {
        if (!active()) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "highlight boost frame is inactive");
            return false;
        }
        auto& stage = _state->highlightBoostStage;
        if (!stage.workspace) {
            return true;
        }
        if (!stage.hostStatus || !stage.eventRecorded) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "highlight boost status was not submitted");
            return false;
        }
        if (const auto status = cudaEventSynchronize(stage.event); status != cudaSuccess) {
            JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(status), "highlight boost status event wait failed");
            return false;
        }
        stage.copyPending = false;
        using JuicerCuda::HighlightBoostDisposition;
        switch (*stage.hostStatus) {
            case HighlightBoostDisposition::Applied:
            case HighlightBoostDisposition::Bypass:
                stage.accepted = true;
                return true;
            case HighlightBoostDisposition::InvalidCarrier:
                JuicerCuda::set_failure(outError, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "HighlightBoostInvalidCarrier");
                return false;
            case HighlightBoostDisposition::PeakOverflow:
                JuicerCuda::set_failure(outError, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "HighlightBoostPeakOverflow");
                return false;
        }
        JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "invalid highlight boost disposition");
        return false;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_weights_built(
        const AutoExposureWeightsExtent& weights) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        workspace.weightsWidth = weights.width;
        workspace.weightsHeight = weights.height;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_metered(
        const AutoExposureMeteredResult& result) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        workspace.keyHash = result.keyHash;
    }

    bool Root::PreparedCudaFrame::record_use(
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError) {
        outError = {};
        if (!_state || !_state->resources || !_state->transaction.active ||
            _state->transaction.committed) {
            outError.diagnostic = "prepared frame is not active for use-event submission";
            return false;
        }
        if (_state->frameUseEventSubmitted) {
            return true;
        }
        if (!_state->submit_frame_use_event(cudaStreamOpaque, outError)) {
            _state->set_failure(
                PreparedCudaFailureStage{"submit_frame_use_event"},
                "CUDA prepared-frame use fencing failed");
            return false;
        }
        return true;
    }

    const char* Root::PreparedCudaFrame::failure_stage_tag() const noexcept {
        return _state ? _state->failureStageTag : "prepare_frame";
    }

    const char* Root::PreparedCudaFrame::failure_prefix() const noexcept {
        return _state ? _state->failurePrefix : "CUDA prepared frame failed";
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const PreparedFrameInput& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        JuicerCuda::PendingContextLossRecovery& recovery,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        PreparedCudaFrame frame = create_prepared_frame(cudaStreamOpaque, recovery, outError);
        if (!frame._state) {
            return frame;
        }
        return prepare_cuda_frame(std::move(frame), deviceContextKey, snapshot, request, autoExposureBufferRequest, cudaStreamOpaque, outError, hostPreparation);
    }

    Root::PreparedCudaFrame Root::create_prepared_frame(
        void* cudaStreamOpaque, JuicerCuda::PendingContextLossRecovery& recovery, JuicerCuda::Failure& outError) {
        outError = {};
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>(recovery);
        } catch (...) {
            outError.diagnostic = "failed to allocate CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->remember_stream(cudaStreamOpaque);
        frame._state->set_failure(
            PreparedCudaFailureStage{"prepare_cuda_frame"},
            "CUDA prepared frame failed");
        return frame;
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        PreparedCudaFrame frame,
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const PreparedFrameInput& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        JuicerCuda::Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const Spektrafilm::ScanRoute route = request.identity.scanRoute;
        const Spektrafilm::ScanRouteMetadata& routeMetadata =
            Spektrafilm::scan_route_metadata(route);
        const bool printRoute = routeMetadata.printRoute;
        const auto appendRoute = [&]() noexcept {
            try {
                if (!outError.diagnostic.empty()) {
                    outError.diagnostic += ' ';
                }
                outError.diagnostic += "route=";
                outError.diagnostic += routeMetadata.key;
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        };
        const auto recordFailure = [&](const char* stageTag, const char* prefix) {
            frame._state->set_failure(
                PreparedCudaFailureStage{stageTag},
                prefix);
            appendRoute();
        };
        const OutputGamutRecipe& outputGamut = request.outputGamut;
        // Explicit views supply these identities independently; reject conflicting bindings before submission.
        if (request.focused.route != route || request.identity.capturePolarity != routeMetadata.capturePolarity) {
            outError.diagnostic = "ResourceDescriptorMismatch component=cuda_frame_preparation field=route_identity";
            recordFailure(
                "validate_cuda_frame_preparation_request",
                "CUDA frame preparation request validation failed");
            return frame;
        }
        if (request.highlightBoostDescriptor
                ? request.highlightBoostDescriptor->recipeHash == 0 || request.highlightBoostDescriptor->recipeHash != request.identity.highlightBoostHash
                : request.identity.highlightBoostHash != 0) {
            outError.diagnostic = "ResourceDescriptorMismatch component=highlight_boost field=recipe_hash";
            recordFailure("validate_highlight_boost", "CUDA highlight boost request validation failed");
            return frame;
        }
        if (request.focused.outputGamut.enabled != outputGamut.enabled ||
            request.focused.outputGamut.recipeHash != outputGamut.hash ||
            (outputGamut.enabled && request.focused.outputGamut.cmax.empty()) ||
            (!outputGamut.enabled && !request.focused.outputGamut.cmax.empty())) {
            outError.diagnostic = "ResourceDescriptorMismatch component=cuda_frame_preparation field=output_gamut_binding";
            recordFailure(
                "validate_cuda_frame_preparation_request",
                "CUDA frame preparation request validation failed");
            return frame;
        }
        if (!validate_visual_grain_descriptor(
                request.identity,
                request.visualGrainDescriptor,
                outError.diagnostic)) {
            recordFailure(
                "validate_visual_grain_descriptor",
                "CUDA visual grain descriptor validation failed");
            return frame;
        }
        if (!validate_film_juicer_effects_descriptor(
                request.identity,
                request.effectsDescriptor,
                outError.diagnostic)) {
            recordFailure(
                "validate_film_juicer_effects_descriptor",
                "CUDA effects descriptor validation failed");
            return frame;
        }
        frame._state->visualGrainDescriptor =
            request.visualGrainDescriptor;
        frame._state->effectsDescriptor = request.effectsDescriptor;
        if (printRoute) {
            if (!request.print) {
                outError.diagnostic = "MissingRequiredResource phase=4B field=print_preparation_request";
                recordFailure(
                    "build_print_resource_descriptors",
                    "CUDA print resource descriptor construction failed");
                return frame;
            }
            frame._state->printDescriptors = request.print->descriptors;
            frame._state->printPreflashRawCmy = request.print->preflashRawCmy;
            frame._state->printNormalizer = request.print->normalizer;
        }
        if (request.identity.grainActive && !request.noise) {
            outError.diagnostic = "MissingRequiredResource phase=grain_static field=noise";
            appendRoute();
            frame.abort();
            return frame;
        }
        if (request.filmGrain && !valid_film_grain_frame(*request.filmGrain, request.requestedWidth, request.requestedHeight)) {
            const char* prefix = printRoute ? "print Film grain binding failed: invalid frame" : "direct Film grain binding failed: invalid frame";
            const char* detail = cudaGetErrorString(cudaErrorInvalidValue);
            std::array<char, 128> diagnostic{};
            const auto length = std::snprintf(diagnostic.data(), diagnostic.size(), "%s: %s", prefix, detail ? detail : "(unknown)");
            JuicerCuda::set_failure(outError, JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), length < 0 ? prefix : diagnostic.data());
            frame._state->set_failure(
                PreparedCudaFailureStage{printRoute ? "print_film_grain_binding" : "direct_film_grain_binding"},
                prefix);
            return frame;
        }
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> deviceLedger;
        if (!begin_submission(
                frame._state->transaction,
                snapshot,
                deviceLedger,
                outError)) {
            recordFailure(
                "begin_submission",
                "CUDA frame begin_submission failed");
            return frame;
        }
        frame._state->recovery.submissionAdmitted = true;
        if (!derive_workspace_layout(
                request.identity,
                request.spatialDirDescriptor,
                request.scannerPostEffects,
                request.diffusionFrameSetDescriptor,
                request.scatterHalationDescriptor,
                request.visualGrainDescriptor,
                request.filmGrain != nullptr,
                request.effectsDescriptor,
                request.identity.capturePolarity,
                request.requestedWidth,
                request.requestedHeight,
                deviceContextKey,
                frame._state->transaction.contextEpoch,
                frame._state->requestedWorkspace,
                outError.diagnostic)) {
            recordFailure(
                "derive_cuda_frame_workspace_request",
                "CUDA frame workspace derivation failed");
            frame.abort();
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.contextEpoch,
                frame._state->transaction.resolvedMemoryBudget,
                deviceLedger,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            recordFailure(
                "resolve_cuda_resources",
                "CUDA frame resource acquisition failed");
            frame.abort();
            return frame;
        }
        if (request.filmGrain && !JuicerCuda::ensure_film_grain_uploaded(
                                     *frame._state->resources, *request.filmGrain, frame._state->filmGrainBank, outError)) {
            recordFailure("prepare_film_grain", "CUDA Film grain bank preparation failed");
            frame.abort();
            return frame;
        }
        if (request.filmGrain &&
            JuicerCuda::FilmGrain::proxy_sample_bound(request.filmGrain->sample_extent, request.filmGrain->sample_step) >
                JuicerCuda::FilmGrain::kProxySerialSamples) {
            if (!frame._state->allocate_device_bytes(&frame._state->filmProxyWorkspace,
                                                     sizeof(JuicerCuda::FilmGrain::ProxyWorkspace),
                                                     "Film proxy transmission reduction",
                                                     outError)) {
                recordFailure("prepare_film_proxy", "CUDA Film proxy workspace preparation failed");
                frame.abort();
                return frame;
            }
        }

        if (!frame._state->prepare_diffusion_resources(
                deviceContextKey,
                request.identity,
                request.diffusionFrameSetDescriptor,
                request.requestedWidth,
                request.requestedHeight,
                cudaStreamOpaque,
                outError)) {
            recordFailure(
                "prepare_diffusion_resources",
                "CUDA diffusion resource preparation failed");
            frame.abort();
            return frame;
        }
        if (!frame._state->prepare_scatter_halation_resources(
                deviceContextKey,
                request.identity,
                request.scatterHalationDescriptor,
                request.highlightBoostDescriptor,
                request.diffusionFrameSetDescriptor,
                request.requestedWidth,
                request.requestedHeight,
                outError)) {
            recordFailure(
                "prepare_scatter_halation_resources",
                "CUDA scatter-halation resource preparation failed");
            frame.abort();
            return frame;
        }
        if (request.highlightBoostDescriptor &&
            !frame._state->prepare_highlight_boost(*request.highlightBoostDescriptor, outError)) {
            recordFailure("prepare_highlight_boost", "CUDA highlight boost resource preparation failed");
            frame.abort();
            return frame;
        }
        if (!JuicerCuda::prepare_focused_route_resources(
                *frame._state->resources,
                request.focused,
                cudaStreamOpaque,
                outError)) {
            recordFailure(
                "prepare_focused_route_resources",
                "CUDA focused route resource preparation failed");
            frame.abort();
            return frame;
        }
        if (printRoute) {
            const bool printPrepared = JuicerCuda::prepare_print_resources(
                *frame._state->resources, *request.print, cudaStreamOpaque, outError);
            if (!printPrepared) {
                recordFailure(
                    "prepare_print_resources",
                    "CUDA print resource preparation failed");
                frame.abort();
                return frame;
            }
        }
        frame._state->capturePolarity =
            request.identity.capturePolarity;
        frame._state->scanRoute =
            request.identity.scanRoute;
        frame._state->focusedInputRgbToXyz = request.focused.inputRgbToXyz;
        frame._state->focusedInputXyzAdapt = request.focused.inputXyzAdapt;
        frame._state->focusedXyzToLinearSrgb = request.focused.xyzToLinearSrgb;
        frame._state->focusedApplyInputChromaticAdapt = request.focused.applyInputChromaticAdapt;
        frame._state->focusedScannerColor = request.scannerColor;
        frame._state->focusedScannerSpectra = JuicerCuda::bind_scanner_spectra(request.focused.scannerTables);
        frame._state->focusedOutputGamutTransform =
            outputGamut.enabled ? std::optional{outputGamut.transform} : std::nullopt;
        if (request.scannerPostEffects) {
            if (!frame.prepare_scanner_post_effects(
                    *request.scannerPostEffects,
                    outError,
                    hostPreparation)) {
                recordFailure(
                    "prepare_scanner_post_effects",
                    "CUDA scanner post-effect preparation failed");
                frame.abort();
                return frame;
            }
        }
        if (!frame._state->allocate_scan_error_stage(outError)) {
            recordFailure(
                "allocate_scan_error_stage",
                "CUDA scan error staging allocation failed");
            frame.abort();
            return frame;
        }
        auto& scanIdentity = frame._state->scanErrorStage.identity;
        const auto& admitted = frame._state->transaction;
        scanIdentity.contextEpoch = admitted.contextEpoch;
        scanIdentity.profileKey = request.identity.filmProfileKey;
        scanIdentity.scanRoute = static_cast<int>(request.identity.scanRoute);
        scanIdentity.capturePolarity = static_cast<int>(request.identity.capturePolarity);
        scanIdentity.submissionId = admitted.snapshot.snapshotId;
        scanIdentity.frameToken = admitted.snapshot.frameToken.value;
        scanIdentity.scannerHash = admitted.snapshot.keyDigests.scannerHash;
        scanIdentity.scannedProfileAssetVersion = printRoute ? request.identity.paperProfileAssetVersionToken
                                                             : request.identity.filmProfileAssetVersionToken;
        scanIdentity.paperProfileKey = request.identity.paperProfileKey;
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(
                autoExposureBufferRequest.descriptor,
                outError)) {
            recordFailure(
                "allocate_auto_exposure_workspace",
                "CUDA auto-exposure workspace allocation failed");
            frame.abort();
            return frame;
        }
        if (!frame.prepare_visual_grain_resources(
                *this,
                request.noise,
                deviceContextKey,
                cudaStreamOpaque,
                outError,
                hostPreparation)) {
            appendRoute();
            frame.abort();
            return frame;
        }
        return frame;
    }


} // namespace JuicerProcess
