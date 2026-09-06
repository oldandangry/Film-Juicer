#include "ProcessRoot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "Illuminants.h"
#include "JuicerState.h"
#include "Logging.h"
#include "SpectralData.h"
#include "SpectralProcessing.h"

#include <cuda_runtime.h>
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

namespace JuicerProcess {

    namespace {

        bool query_cuda_device_total_bytes(
            int deviceId,
            std::uint64_t& outDeviceBudgetBytes,
            std::string& outError) {
            outDeviceBudgetBytes = 0;
            outError.clear();
            if (deviceId < 0) {
                outError = "CUDA device-total query requires a valid device id";
                return false;
            }

            int previousDevice = -1;
            cudaError_t status = cudaGetDevice(&previousDevice);
            if (status != cudaSuccess) {
                outError = std::string("cudaGetDevice failed during device-total query: ") +
                           cudaGetErrorString(status);
                return false;
            }

            const bool switchedDevice = previousDevice != deviceId;
            if (switchedDevice) {
                status = cudaSetDevice(deviceId);
                if (status != cudaSuccess) {
                    outError = std::string("cudaSetDevice failed during device-total query: ") +
                               cudaGetErrorString(status);
                    return false;
                }
            }

            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            const cudaError_t queryStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
            const cudaError_t restoreStatus = switchedDevice
                                                  ? cudaSetDevice(previousDevice)
                                                  : cudaSuccess;
            if (queryStatus != cudaSuccess) {
                outError = std::string("cudaMemGetInfo failed during device-total query: ") +
                           cudaGetErrorString(queryStatus);
                return false;
            }
            if (restoreStatus != cudaSuccess) {
                outError = std::string("cudaSetDevice failed restoring device after device-total query: ") +
                           cudaGetErrorString(restoreStatus);
                return false;
            }

            outDeviceBudgetBytes = static_cast<std::uint64_t>(
                std::min<std::size_t>(
                    totalBytes,
                    static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
            if (outDeviceBudgetBytes <= 1) {
                outError = "CUDA device-total query returned an invalid budget";
                outDeviceBudgetBytes = 0;
                return false;
            }
            return true;
        }

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
            if (descriptor.approximation != Spektrafilm::DirApproximationMarker::SpektrafilmStrict ||
                descriptor.scratchTier != Spektrafilm::DirScratchTier::Tier1IChannels ||
                descriptor.targetScratchTier != Spektrafilm::DirScratchTier::Tier2) {
                return false;
            }
            for (int component = 0; component < descriptor.filterPlan.componentCount; ++component) {
                const Spektrafilm::DirGaussianComponentPlan& plan =
                    descriptor.filterPlan.components[static_cast<std::size_t>(component)];
                if (plan.weight > 0.0f &&
                    plan.referenceOperator == Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReplicate) {
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

        Root::PreparedCudaFrame::WorkspaceRequest strict_yvv_workspace_candidate(
            const Root::PreparedCudaFrame::WorkspaceRequest& base,
            Spektrafilm::DirScratchTier buildTier,
            const Spektrafilm::DirScratchPlaneRoles& buildRoles,
            Spektrafilm::DirScratchTier targetTier,
            const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
            Root::PreparedCudaFrame::WorkspaceRequest candidate = base;
            candidate.spatialDirScratchTier = buildTier;
            candidate.spatialDirPlaneRoles = buildRoles;
            candidate.spatialDirTargetScratchTier = targetTier;
            candidate.spatialDirTargetPlaneRoles = targetRoles;
            return candidate;
        }

        bool spatial_dir_alias_target_uses_build_roles(
            const Root::PreparedCudaFrame::WorkspaceRequest& request) noexcept {
            return request.aliasScannerRgbFromSpatialDirFiltered &&
                   request.needSpatialDir &&
                   request.spatialDirTargetScratchTier == request.spatialDirScratchTier &&
                   dir_plane_roles_equal(
                       request.spatialDirTargetPlaneRoles,
                       request.spatialDirPlaneRoles);
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


        std::string compute_process_data_dir() {
            namespace fs = std::filesystem;

#if defined(_WIN32)
            HMODULE module = nullptr;
            if (!GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(&compute_process_data_dir),
                    &module)) {
                return std::string();
            }

            std::wstring buffer(MAX_PATH, L'\0');
            DWORD length = 0;
            for (;;) {
                SetLastError(ERROR_SUCCESS);
                length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
                if (length == 0) {
                    return std::string();
                }
                if (length < buffer.size()) {
                    buffer.resize(length);
                    break;
                }
                if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
                    buffer.resize(length);
                    break;
                }
                buffer.resize(buffer.size() * 2);
            }

            fs::path modulePath(buffer);
            fs::path moduleDir = modulePath.parent_path();
            if (moduleDir.empty()) {
                return std::string();
            }
            fs::path contentsDir = moduleDir.parent_path();
            if (contentsDir.empty()) {
                return std::string();
            }

            fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
            resourcesDir.make_preferred();
            std::wstring native = resourcesDir.native();
            if (!native.empty() && native.back() != L'\\') {
                native.push_back(L'\\');
            }

            if (native.empty()) {
                return std::string();
            }

            int required = WideCharToMultiByte(
                CP_UTF8,
                0,
                native.c_str(),
                static_cast<int>(native.size()),
                nullptr,
                0,
                nullptr,
                nullptr);
            if (required <= 0) {
                return std::string();
            }

            std::string path(static_cast<size_t>(required), '\0');
            WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), path.data(), required, nullptr, nullptr);
            return path;
#else
            Dl_info info{};
            if (dladdr(reinterpret_cast<const void*>(&compute_process_data_dir), &info) == 0 || info.dli_fname == nullptr) {
                return std::string();
            }

            fs::path modulePath(info.dli_fname);
            fs::path moduleDir = modulePath.parent_path();
            if (moduleDir.empty()) {
                return std::string();
            }
            fs::path contentsDir = moduleDir.parent_path();
            if (contentsDir.empty()) {
                return std::string();
            }

            fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
            resourcesDir.make_preferred();
            std::string path = resourcesDir.u8string();
            if (!path.empty() && path.back() != '/') {
                path.push_back('/');
            }
            return path;
#endif
        }

        template <typename... Parts>
        std::string data_file_string(const std::string& dataDir, Parts&&... parts) {
            std::filesystem::path path(dataDir);
            ((path /= std::filesystem::path(std::forward<Parts>(parts))), ...);
            path.make_preferred();
            return path.string();
        }

        void load_process_spectral_assets(const std::string& dataDir) {
            try {
                Spectral::lock_shape_to_reference_axis();
                const auto cmf = Spectral::load_csv_triplets(data_file_string(dataDir, "cie1931_2deg.csv"));
                if (!Spectral::cmf_triplets_match_reference_axis(cmf)) {
                    JTRACE("INIT", "FATAL: CMF wavelengths do not match 380-780@5nm grid");
                    throw std::runtime_error("CMF grid mismatch");
                }
                Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
            } catch (const std::exception& ex) {
                Spectral::set_hanatos_available(false);
                Spectral::set_mallett_available(false);
                (void)ex;
#if JUICER_DIAGNOSTICS_COMPILED
                if (JTRACE_ENABLED(1)) {
                    std::string msg = "FATAL: spectral bootstrap failed: ";
                    msg += ex.what();
                    JTRACE("INIT", msg);
                }
#endif
            } catch (...) {
                Spectral::set_hanatos_available(false);
                Spectral::set_mallett_available(false);
                JTRACE("INIT", "FATAL: spectral bootstrap failed with unknown error");
            }

            try {
                const std::string lutPath = data_file_string(
                    dataDir,
                    "luts",
                    "spectral_upsampling",
                    "irradiance_xy_tc.npy");
                Spectral::load_hanatos_spectra_lut(lutPath);
            } catch (...) {
                Spectral::set_hanatos_available(false);
            }

            try {
                const std::string basisPath = data_file_string(
                    dataDir,
                    "luts",
                    "spectral_upsampling",
                    "mallett2019_basis.npy");
                Spectral::load_mallett2019_basis_npy(basisPath);
            } catch (...) {
                Spectral::set_mallett_available(false);
            }
        }

        JuicerCuda::ResourceManager::ScratchRequestDescriptor make_workspace_scratch_request_descriptor(
            const Root::PreparedCudaFrame::WorkspaceRequest& workspace) noexcept {
            JuicerCuda::ResourceManager::ScratchRequestBuildRequest request{};
            request.families.needOptics = workspace.needOptics;
            request.families.needSpatialDir = workspace.needSpatialDir;
            request.spatialDirDescriptorHash = workspace.spatialDirDescriptorHash;
            request.spatialDirScratchTier = workspace.spatialDirScratchTier;
            request.spatialDirPlaneRoles = workspace.spatialDirPlaneRoles;
            request.spatialDirTargetScratchTier = workspace.spatialDirTargetScratchTier;
            request.spatialDirTargetPlaneRoles = workspace.spatialDirTargetPlaneRoles;
            request.extent.requestedWidth = workspace.requestedWidth;
            request.extent.requestedHeight = workspace.requestedHeight;
            request.attachments.needBlurred = workspace.needBlurred;
            request.attachments.aliasScannerRgbFromSpatialDirFiltered =
                workspace.aliasScannerRgbFromSpatialDirFiltered;
            request.attachments.needAux = workspace.needAux;
            request.attachments.needSharedTmp = workspace.needSharedTmp;
            request.attachments.needGrainFrameUniforms =
                workspace.needGrainFrameUniforms;
            request.attachments.needGrainLayerWork = workspace.needGrainLayerWork;
            request.attachments.needGrainShared = workspace.needGrainShared;
            request.attachments.needGateTransmittance = workspace.needGateTransmittance;
            request.attachments.needFilmDustTransmittance = workspace.needFilmDustTransmittance;
            request.attachments.gateWidth = workspace.gateWidth;
            request.attachments.gateHeight = workspace.gateHeight;
            return JuicerCuda::ResourceManager::make_scratch_request_descriptor(
                request);
        }

#if JUICER_DIAGNOSTICS_COMPILED
        void append_workspace_request_trace_fields(
            std::string& msg,
            const char* prefix,
            const Root::PreparedCudaFrame::WorkspaceRequest& request) {
            const Spektrafilm::DirScratchPlaneRoles& roles = request.spatialDirPlaneRoles;
            const Spektrafilm::DirScratchPlaneRoles& targetRoles =
                request.spatialDirTargetPlaneRoles;
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

            append_bool("need_optics", request.needOptics);
            append_bool("need_spatial_dir", request.needSpatialDir);
            append_u64("spatial_dir_descriptor_hash", request.spatialDirDescriptorHash);
            append_cstr("spatial_dir_scratch_tier", Spektrafilm::to_cstr(request.spatialDirScratchTier));
            append_i32("raw_correction_planes", roles.rawCorrectionPlanes);
            append_i32("filtered_correction_planes", roles.filteredCorrectionPlanes);
            append_i32("filter_temp_planes", roles.filterTempPlanes);
            append_i32("cached_log_raw_planes", roles.cachedLogRawPlanes);
            append_cstr("spatial_dir_target_scratch_tier", Spektrafilm::to_cstr(request.spatialDirTargetScratchTier));
            append_i32("target_raw_correction_planes", targetRoles.rawCorrectionPlanes);
            append_i32("target_filtered_correction_planes", targetRoles.filteredCorrectionPlanes);
            append_i32("target_filter_temp_planes", targetRoles.filterTempPlanes);
            append_i32("target_cached_log_raw_planes", targetRoles.cachedLogRawPlanes);
            append_i32("requested_width", request.requestedWidth);
            append_i32("requested_height", request.requestedHeight);
            append_bool("need_blurred", request.needBlurred);
            append_bool(
                "alias_scanner_rgb_from_spatial_dir_filtered",
                request.aliasScannerRgbFromSpatialDirFiltered);
            append_bool("need_aux", request.needAux);
            append_bool("need_shared_tmp", request.needSharedTmp);
            append_bool(
                "need_grain_frame_uniforms",
                request.needGrainFrameUniforms);
            append_bool("need_grain_layer_work", request.needGrainLayerWork);
            append_bool("need_grain_shared", request.needGrainShared);
            append_bool("need_gate_transmittance", request.needGateTransmittance);
        }

        void trace_workspace_request_mismatch(
            const JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
            const Root::PreparedCudaFrame::WorkspaceRequest& activeRequest,
            const Root::PreparedCudaFrame::WorkspaceRequest& requestedRequest) {
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

    Root::FramePreparationToken::FramePreparationToken(Root* root) noexcept
        : _root(root) {
    }

    Root::FramePreparationToken::~FramePreparationToken() {
        reset();
    }

    Root::FramePreparationToken::FramePreparationToken(FramePreparationToken&& other) noexcept
        : _root(std::exchange(other._root, nullptr)) {
    }

    Root::FramePreparationToken& Root::FramePreparationToken::operator=(FramePreparationToken&& other) noexcept {
        if (this != &other) {
            reset();
            _root = std::exchange(other._root, nullptr);
        }
        return *this;
    }

    bool Root::FramePreparationToken::active() const noexcept {
        return _root != nullptr;
    }

    void Root::FramePreparationToken::reset() noexcept {
        Root* root = _root;
        _root = nullptr;
        if (root) {
            root->finish_frame_preparation();
        }
    }

    struct PreparedCudaFailureStage {
        const char* tag = nullptr;
    };

    struct Root::PreparedCudaFrame::State {
        struct AutoExposureFrameWorkspace {
            JuicerCudaAutoExposureScratch scratch{};
            JuicerCudaAutoExposureDeviceState deviceState{};
            int weightsWidth = 0;
            int weightsHeight = 0;
            int weightsXCapacity = 0;
            int weightsYCapacity = 0;
            Spektrafilm::AutoExposureMethod method = Spektrafilm::AutoExposureMethod::CenterWeighted;
            std::uint64_t keyHash = 0;
            bool active = false;
        };

        struct ScanErrorFrameStage {
            int* deviceFlag = nullptr;
            int* hostFlag = nullptr;
            void* eventOpaque = nullptr;
            bool active = false;
            bool readbackPending = false;
        };

        struct FrameScratchWorkspace {
            WorkspaceRequest request{};
            WorkspaceRequest postFrameRequest{};
            bool postFrameRequestActive = false;
            bool retainedLeaseActive = false;
        };

        Root* root = nullptr;
        Root::CudaResourceOwner resourceOwner;
        Root::CudaResourceOwner grainStaticOwner;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Resources* grainStaticResources = nullptr;
        Spektrafilm::DiffusionExecutionDescriptor diffusionExecutionDescriptor{};
        JuicerCuda::Diffusion::PreparedDiffusionLease diffusionLease;
        std::optional<ScatterHalationFrameDescriptor> scatterHalationDescriptor;
        void* scatterHalationFilterBlock = nullptr;
        void* scatterHalationCarrierBlock = nullptr;
        JuicerCuda::ScatterHalationCarrierSource scatterHalationCarrierSource =
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes;
        JuicerCuda::CameraFilmLinearExposurePlanes scatterHalationCarrier{};
        float* scatterHalationFilterTemp = nullptr;
        float* scatterHalationWeightedAccumulation = nullptr;
        const Spectral::FilmRawConfig* focusedFilmRawConfig = nullptr;
        const Scanner::ColorRuntime* focusedScannerColor = nullptr;
        const JuicerCuda::Resources::DeviceScanMedium* focusedScanMedium = nullptr;
        const JuicerCuda::Resources::DeviceSpectralLut* focusedScanLut = nullptr;
        const PrintRecipe* printRecipe = nullptr;
        JuicerCuda::PrintResourceDescriptors printDescriptors{};
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        ScanErrorFrameStage scanErrorStage{};
        AutoExposureFrameWorkspace autoExposureWorkspace{};
        FrameScratchWorkspace scratchWorkspace{};
        std::map<void*, JuicerCuda::DeviceByteReservation>
            deviceAllocationRecords;
        WorkspaceRequest workspaceRequest{};
        Spektrafilm::SpatialDirDescriptor spatialDirDescriptor{};
        Scanner::ScannerPostEffectsDescriptor scannerPostEffectsDescriptor{};
        Spektrafilm::ProfilePolarity capturePolarity =
            Spektrafilm::ProfilePolarity::Unsupported;
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> visualGrainDescriptor;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effectsDescriptor;
        std::array<JuicerCuda::VisualGrainPreparedGaussianView, 3>
            preparedGrainCorrelation{};
        JuicerCuda::VisualGrainPreparedGaussianView
            preparedGrainDyeCloud[3][3] = {};
        JuicerCuda::VisualGrainPreparedDensityLayersView
            preparedGrainDensityLayers{};
        bool visualGrainPrepared = false;
        void* lastCudaStreamOpaque = nullptr;
        bool frameUseEventSubmitted = false;
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";

        void set_failure(
            PreparedCudaFailureStage stage,
            const char* prefix) noexcept {
            failureStageTag = stage.tag;
            failurePrefix = prefix;
        }

        void remember_stream(void* cudaStreamOpaque) noexcept {
            if (cudaStreamOpaque) {
                lastCudaStreamOpaque = cudaStreamOpaque;
            }
        }

        bool allocate_device_bytes(
            void** outPtr,
            std::size_t bytes,
            const char* label,
            std::string& outError);
        bool retire_device_bytes(
            void* ptr,
            void* cudaStreamOpaque,
            const char* label,
            std::string& outError);
        bool free_device_bytes_now(void* ptr, std::string& outError) noexcept;
        bool transfer_remaining_device_allocations(
            std::string& outError) noexcept;
        bool prepare_diffusion_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            const Spektrafilm::RenderRecipe& recipe,
            const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
            int requestedWidth,
            int requestedHeight,
            void* cudaStreamOpaque,
            std::string& outError);
        bool prepare_scatter_halation_resources(
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            const Spektrafilm::RenderRecipe& recipe,
            const ScatterHalationFrameDescriptor* descriptor,
            const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
            int requestedWidth,
            int requestedHeight,
            std::string& outError);
        bool release_scatter_halation_resources(
            void* cudaStreamOpaque,
            std::string& outError);

        bool allocate_auto_exposure_workspace(
            const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
            std::string& outError);
        bool allocate_scan_error_stage(std::string& outError);
        bool release_scan_error_stage_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool release_auto_exposure_workspace_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        JuicerCuda::ResourceManager::ScratchRequestDescriptor post_frame_scratch_request_descriptor() const noexcept;
        bool ensure_scratch_workspace(
            const WorkspaceRequest& request,
            void* cudaStreamOpaque,
            std::string& outError);
        bool release_scratch_workspace_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool submit_frame_use_event(
            void* cudaStreamOpaque,
            std::string& outError);
        void free_scan_error_stage_now() noexcept;
        void free_auto_exposure_workspace_now() noexcept;
    };

    bool validate_visual_grain_descriptor(
        const Spektrafilm::RenderRecipe* recipe,
        const std::optional<Spektrafilm::VisualGrainFrameDescriptor>& descriptor,
        std::string& outError) {
        if (!recipe) {
            outError =
                "MissingRequiredResource phase=grain_descriptor field=render_recipe";
            return false;
        }
        const Spektrafilm::VisualGrainRecipe& visualGrain =
            recipe->visualGrain;
        if (!visualGrain.active) {
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
            frame.recipeHash != visualGrain.hash ||
            frame.capturePolarity != recipe->profileRoute.capturePolarity ||
            frame.densityCurvesLayersHash !=
                visualGrain.densityCurvesLayersHash ||
            frame.staticNoiseVersion == 0 ||
            frame.scratchShape ==
                Spektrafilm::VisualGrainScratchShape::None) {
            outError =
                "ResourceDescriptorMismatch phase=grain_descriptor field=identity";
            return false;
        }
        return true;
    }

    bool validate_film_juicer_effects_descriptor(
        const Spektrafilm::RenderRecipe* recipe,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& descriptor,
        std::string& outError) {
        if (!recipe) {
            outError =
                "MissingRequiredResource phase=effects_descriptor field=render_recipe";
            return false;
        }
        const Spektrafilm::FilmJuicerEffectsRecipe& effects =
            recipe->filmJuicerEffects;
        if (!effects.active) {
            if (effects.hash != 0 || descriptor.has_value()) {
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
        if (frame.hash == 0 || frame.recipeHash != effects.hash ||
            (!frame.filmActive && !frame.gateOutputActive) ||
            frame.gateOutputActive !=
                (frame.weaveActive || frame.gateTransmittanceActive) ||
            frame.requiresFullFrame != frame.weaveActive ||
            !Spektrafilm::validate_film_juicer_effects_frame_descriptor(frame)) {
            outError =
                "ResourceDescriptorMismatch phase=effects_descriptor field=identity";
            return false;
        }
        return true;
    }

    bool derive_visual_grain_workspace_request(
        const Spektrafilm::VisualGrainFrameDescriptor& descriptor,
        Root::PreparedCudaFrame::WorkspaceRequest& request,
        std::string& outError) {
        if (!descriptor.active || descriptor.hash == 0 ||
            descriptor.scratchShape == Spektrafilm::VisualGrainScratchShape::None) {
            outError =
                "ResourceDescriptorMismatch phase=grain_workspace field=descriptor";
            return false;
        }
        request.needOptics = true;
        request.needSharedTmp = true;
        request.needBlurred = true;
        request.needAux = true;
        request.needGrainFrameUniforms = true;
        switch (descriptor.scratchShape) {
            case Spektrafilm::VisualGrainScratchShape::Streamed:
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedShared:
                request.needGrainShared = true;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayers:
                request.needGrainLayerWork = true;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayersShared:
                request.needGrainLayerWork = true;
                request.needGrainShared = true;
                break;
            case Spektrafilm::VisualGrainScratchShape::None:
            default:
                outError =
                    "ResourceDescriptorMismatch phase=grain_workspace field=scratch_shape";
                return false;
        }
        return true;
    }

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
        ScatterHalationAllocationShape& out,
        const char*& failedFact) noexcept {
        out = ScatterHalationAllocationShape{};
        failedFact = nullptr;
        out.physicalAllocationCount =
            carrierSource ==
                    JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes
                ? 2u
                : 1u;

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
        out.filterSecondPlaneOffset = out.planeBytes;
        if (!checked_size_multiply(2u, out.planeBytes, out.filterBytes)) {
            failedFact = "filter_bytes";
            return false;
        }
        std::size_t filterEnd = 0;
        if (!checked_size_add(
                out.filterSecondPlaneOffset,
                out.planeBytes,
                filterEnd) ||
            filterEnd != out.filterBytes) {
            failedFact = "filter_second_plane_offset";
            return false;
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
        const Spektrafilm::RenderRecipe& recipe,
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
        outError += Spektrafilm::scan_route_key(recipe.profileRoute.scanRoute);
        outError += " domain=FilmLinearExposure film_profile_key=";
        outError += recipe.profileRoute.filmProfileKey;
        outError += " film_profile_asset_version_token=" +
                    std::to_string(recipe.profileRoute.filmProfileAssetVersionToken);
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

    bool derive_workspace_request(
        const Spektrafilm::RenderRecipe& recipe,
        const Spektrafilm::SpatialDirDescriptor* spatialDir,
        const Scanner::ScannerPostEffectsDescriptor* scannerPostEffects,
        const Spektrafilm::DiffusionFrameSetDescriptor* diffusionFrameSet,
        const ScatterHalationFrameDescriptor* scatterHalationDescriptor,
        const std::optional<Spektrafilm::VisualGrainFrameDescriptor>& visualGrain,
        const std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor>& effects,
        Spektrafilm::ProfilePolarity capturePolarity,
        int requestedWidth,
        int requestedHeight,
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        Root::PreparedCudaFrame::WorkspaceRequest& out,
        std::string& outError) {
        out = Root::PreparedCudaFrame::WorkspaceRequest{};
        if (requestedWidth <= 0 || requestedHeight <= 0) {
            outError = "ResourceDescriptorMismatch phase=workspace field=extent";
            return false;
        }
        out.requestedWidth = requestedWidth;
        out.requestedHeight = requestedHeight;

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
            out.needOptics = true;
        }

        const std::uint64_t scatterHalationRecipeHash =
            recipe.spatialOptics.scatterHalation.hash;
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
                    shape,
                    failedFact)) {
                set_scatter_halation_exact_admission_failure(
                    recipe,
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
                    recipe,
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
            out.needOptics = true;
        }

        if (spatialDir && spatialDir->hash != 0) {
            if (spatialDir->renderExtent.width != requestedWidth ||
                spatialDir->renderExtent.height != requestedHeight) {
                outError =
                    "ResourceDescriptorMismatch phase=workspace field=spatial_dir_extent";
                return false;
            }
            out.needSpatialDir = true;
            out.spatialDirDescriptorHash = spatialDir->hash;
            out.spatialDirScratchTier = spatialDir->scratchTier;
            out.spatialDirPlaneRoles = spatialDir->planeRoles;
            out.spatialDirTargetScratchTier = spatialDir->targetScratchTier;
            out.spatialDirTargetPlaneRoles = spatialDir->targetPlaneRoles;
            out.needSharedTmp = spatialDir->planeRoles.filterTempPlanes > 0;
        }
        if (visualGrain.has_value()) {
            if (visualGrain->renderExtent.width != requestedWidth ||
                visualGrain->renderExtent.height != requestedHeight) {
                outError =
                    "ResourceDescriptorMismatch phase=workspace field=grain_extent";
                return false;
            }
            if (!derive_visual_grain_workspace_request(
                    *visualGrain,
                    out,
                    outError)) {
                return false;
            }
        }
        if (effects.has_value()) {
            if (effects->renderExtent.width != requestedWidth ||
                effects->renderExtent.height != requestedHeight ||
                (!effects->filmActive && !effects->gateOutputActive)) {
                outError =
                    "ResourceDescriptorMismatch phase=effects_workspace field=descriptor";
                return false;
            }
            out.needOptics = true;
            out.needGateTransmittance = effects->gateTransmittanceActive;
            out.needFilmDustTransmittance = effects->filmDust.slotProbability > 0.0f;
            out.gateWidth = effects->gateWidth;
            out.gateHeight = effects->gateHeight;
        }
        if (scannerPostEffects && scannerPostEffects->active()) {
            out.needOptics = true;
            out.needSharedTmp = true;
            out.needBlurred =
                out.needBlurred ||
                (scannerPostEffects->glareActive &&
                 scannerPostEffects->glareBlurSigmaPx > 0.0f);
        }
        // Positive final develop retains DIR correction as an input while the scanner writes RGB.
        if (!routeDiffusionActive && out.needOptics && out.needSpatialDir &&
            capturePolarity != Spektrafilm::ProfilePolarity::Positive) {
            out.aliasScannerRgbFromSpatialDirFiltered = true;
            out.spatialDirTargetScratchTier = out.spatialDirScratchTier;
            out.spatialDirTargetPlaneRoles = out.spatialDirPlaneRoles;
            out.spatialDirTargetPlaneRoles.cachedLogRawPlanes = 0;
        }
        if (!out.has_any_family()) {
            out = Root::PreparedCudaFrame::WorkspaceRequest{};
        }
        return true;
    }

    bool Root::PreparedCudaFrame::State::allocate_device_bytes(
        void** outPtr,
        std::size_t bytes,
        const char* label,
        std::string& outError) {
        outError.clear();
        if (!outPtr || *outPtr || bytes == 0 || !label || !label[0] ||
            !resources || !resources->deviceLedger) {
            outError = "invalid prepared-frame CUDA allocation request";
            return false;
        }

        std::map<void*, JuicerCuda::DeviceByteReservation> stagedRecords;
        std::map<void*, JuicerCuda::DeviceByteReservation>::node_type stagedNode;
        try {
            stagedRecords.emplace(nullptr, JuicerCuda::DeviceByteReservation{});
            stagedNode = stagedRecords.extract(stagedRecords.begin());
        } catch (...) {
            JuicerLogging::discard_current_exception();
            outError = "failed to stage prepared-frame allocation record";
            return false;
        }

        JuicerCuda::DeviceByteReservation reservation;
        if (!resources->deviceLedger->reserve(
                JuicerCuda::DeviceReservationRequest{
                    .contextKey = resources->ownerContextKey,
                    .contextEpoch = resources->contextEpoch,
                    .bytes = static_cast<std::uint64_t>(bytes)},
                reservation,
                outError)) {
            return false;
        }
        void* allocated = nullptr;
        const cudaError_t allocationError = cudaMalloc(&allocated, bytes);
        if (allocationError != cudaSuccess || !allocated) {
            outError = std::string("cudaMalloc(") + label + ") failed: " +
                       (cudaGetErrorString(allocationError)
                            ? cudaGetErrorString(allocationError)
                            : "(unknown)");
            return false;
        }
        if (!reservation.commit(static_cast<std::uint64_t>(bytes), outError)) {
            (void)cudaFree(allocated);
            return false;
        }

        stagedNode.key() = allocated;
        stagedNode.mapped() = std::move(reservation);
        const auto inserted = deviceAllocationRecords.insert(std::move(stagedNode));
        if (!inserted.inserted) {
            JuicerCuda::DeviceByteReservation duplicateReservation =
                std::move(inserted.node.mapped());
            std::string retireError;
            (void)duplicateReservation.mark_retiring(retireError);
            const cudaError_t freeError = cudaFree(allocated);
            (void)duplicateReservation.release_after_physical_free(
                freeError == cudaSuccess,
                retireError);
            outError = "prepared-frame allocation pointer collision";
            return false;
        }
        *outPtr = allocated;
        return true;
    }

    bool Root::PreparedCudaFrame::State::retire_device_bytes(
        void* ptr,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        if (!ptr) {
            return true;
        }
        const auto allocation = deviceAllocationRecords.find(ptr);
        if (allocation == deviceAllocationRecords.end()) {
            outError = "unregistered prepared-frame CUDA retirement";
            return false;
        }
        const std::size_t recordedBytes = static_cast<std::size_t>(
            allocation->second.bytes());
        if (recordedBytes == 0) {
            outError = "invalid prepared-frame CUDA retirement record";
            return false;
        }
        if (!JuicerCuda::retire_frame_scratch_allocation(
                *resources,
                ptr,
                recordedBytes,
                std::move(allocation->second),
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
        std::string& outError) noexcept {
        outError.clear();
        if (!ptr) {
            return true;
        }
        try {
            const auto allocation = deviceAllocationRecords.find(ptr);
            if (allocation == deviceAllocationRecords.end()) {
                outError = "unregistered prepared-frame CUDA free";
                return false;
            }
            JuicerCuda::DeviceByteReservation& reservation = allocation->second;
            if (reservation.state() ==
                    JuicerCuda::DeviceReservationState::Committed &&
                !reservation.mark_retiring(outError)) {
                return false;
            }
            const cudaError_t freeError = cudaFree(ptr);
            if (freeError != cudaSuccess) {
                outError = std::string("cudaFree(prepared frame) failed: ") +
                           (cudaGetErrorString(freeError)
                                ? cudaGetErrorString(freeError)
                                : "(unknown)");
                return false;
            }
            if (!reservation.release_after_physical_free(true, outError)) {
                return false;
            }
            deviceAllocationRecords.erase(allocation);
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "prepared-frame CUDA free bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool Root::PreparedCudaFrame::State::transfer_remaining_device_allocations(
        std::string& outError) noexcept {
        try {
            outError.clear();
            if (deviceAllocationRecords.empty()) {
                return true;
            }
            if (!resources) {
                outError =
                    "CUDA resources unavailable for remaining frame allocation transfer";
                return false;
            }
            while (!deviceAllocationRecords.empty()) {
                const auto allocation = deviceAllocationRecords.begin();
                void* ptr = allocation->first;
                std::string retireError;
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
                    std::string freeError;
                    if (free_device_bytes_now(ptr, freeError)) {
                        continue;
                    }
                }

                auto allocationRecord = deviceAllocationRecords.extract(ptr);
                std::string adoptionError;
                if (!JuicerCuda::adopt_failed_frame_allocation_record(
                        *resources,
                        allocationRecord,
                        completionCertain,
                        adoptionError)) {
                    if (!allocationRecord.empty()) {
                        deviceAllocationRecords.insert(
                            std::move(allocationRecord));
                    }
                    outError = adoptionError.empty()
                                   ? "remaining frame allocation adoption failed"
                                   : adoptionError;
                    return false;
                }
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError =
                    "remaining frame allocation transfer bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool Root::PreparedCudaFrame::State::prepare_diffusion_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        const Spektrafilm::RenderRecipe& recipe,
        const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
        int requestedWidth,
        int requestedHeight,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        diffusionExecutionDescriptor = {};
        if (!frameSet) {
            if (!resources) {
                outError =
                    "MissingRequiredResource component=diffusion field=context_resources";
                return false;
            }
            return JuicerCuda::Diffusion::trim_inactive_diffusion_resources(
                resources->diffusion,
                contextKey,
                outError);
        }
        const Spektrafilm::ScanRoute route = recipe.profileRoute.scanRoute;
        const Spektrafilm::ScanRouteMetadata& routeMetadata =
            Spektrafilm::scan_route_metadata(route);
        const bool printRoute = routeMetadata.printRoute;
        if (!resources || !resources->deviceLedger ||
            transaction.snapshot.contextEpoch == 0 ||
            transaction.resolvedMemoryBudget.allocationCapBytes == 0 ||
            routeMetadata.route != route || frameSet->hash == 0 ||
            frameSet->route != route ||
            frameSet->fullFrame.width != requestedWidth ||
            frameSet->fullFrame.height != requestedHeight ||
            requestedWidth <= 0 || requestedHeight <= 0 ||
            (!printRoute && frameSet->enlarger.has_value())) {
            outError =
                "ResourceDescriptorMismatch component=diffusion field=preparation_request";
            return false;
        }

        const auto appendDiffusionFailureIdentity = [&] {
            const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                resources->deviceLedger->snapshot();
            outError += " device_id=" + std::to_string(contextKey.deviceId);
            outError += " context=" + std::to_string(
                                          static_cast<unsigned long long>(
                                              reinterpret_cast<std::uintptr_t>(
                                                  contextKey.contextOpaque)));
            outError += " epoch=" + std::to_string(
                                        static_cast<unsigned long long>(
                                            transaction.snapshot.contextEpoch));
            outError += " device_budget_bytes=" + std::to_string(
                                                      static_cast<unsigned long long>(
                                                          transaction.resolvedMemoryBudget.deviceBudgetBytes));
            outError += " allocation_cap_bytes=" + std::to_string(
                                                       static_cast<unsigned long long>(
                                                           transaction.resolvedMemoryBudget.allocationCapBytes));
            outError += " ledger_reserved_bytes=" + std::to_string(
                                                        static_cast<unsigned long long>(
                                                            ledgerSnapshot.reservedBytes));
            outError += " ledger_committed_bytes=" + std::to_string(
                                                         static_cast<unsigned long long>(
                                                             ledgerSnapshot.committedBytes));
            outError += " ledger_retiring_bytes=" + std::to_string(
                                                        static_cast<unsigned long long>(
                                                            ledgerSnapshot.retiringBytes));
            outError += " frame_set_hash=" + std::to_string(
                                                 static_cast<unsigned long long>(
                                                     frameSet->hash));
            outError += " execution_hash=" + std::to_string(
                                                 static_cast<unsigned long long>(
                                                     diffusionExecutionDescriptor.hash));
            outError += " fft_width=" +
                        std::to_string(diffusionExecutionDescriptor.layout.width);
            outError += " fft_height=" +
                        std::to_string(diffusionExecutionDescriptor.layout.height);
            outError += " unique_spectra=" + std::to_string(
                                                 diffusionExecutionDescriptor.uniqueSpectrumCount);
            for (std::size_t index = 0;
                 index < diffusionExecutionDescriptor.uniqueSpectrumCount;
                 ++index) {
                outError += " spectrum_key_" + std::to_string(index) + "=" +
                            std::to_string(static_cast<unsigned long long>(
                                diffusionExecutionDescriptor.spectrumKeys[index].hash));
            }
        };
        if (frameSet->camera &&
            frameSet->camera->stage !=
                Spektrafilm::DiffusionLinearStage::CameraFilmLinear) {
            outError =
                "ResourceDescriptorMismatch component=diffusion field=camera_stage_order";
            return false;
        }
        if (frameSet->enlarger &&
            frameSet->enlarger->stage !=
                Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear) {
            outError =
                "ResourceDescriptorMismatch component=diffusion field=enlarger_stage_order";
            return false;
        }

        if (!Spektrafilm::build_diffusion_execution_descriptor(
                *frameSet,
                transaction.snapshot.contextEpoch,
                transaction.resolvedMemoryBudget.allocationCapBytes,
                diffusionExecutionDescriptor,
                outError)) {
            appendDiffusionFailureIdentity();
            return false;
        }
        if (!JuicerCuda::Diffusion::prepare_diffusion_resources(
                resources->diffusion,
                contextKey,
                transaction.snapshot.contextEpoch,
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
        const Spektrafilm::RenderRecipe& recipe,
        const ScatterHalationFrameDescriptor* descriptor,
        const Spektrafilm::DiffusionFrameSetDescriptor* frameSet,
        int requestedWidth,
        int requestedHeight,
        std::string& outError) {
        outError.clear();
        if (!descriptor) {
            return true;
        }
        if (!resources || !transaction.active || transaction.committed ||
            scatterHalationDescriptor || scatterHalationFilterBlock ||
            scatterHalationCarrierBlock) {
            outError =
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
        const char* failedFact = nullptr;
        if (!derive_scatter_halation_allocation_shape(
                requestedWidth,
                requestedHeight,
                carrierSource,
                shape,
                failedFact)) {
            set_scatter_halation_exact_admission_failure(
                recipe,
                *descriptor,
                requestedWidth,
                requestedHeight,
                carrierSource,
                shape,
                0,
                contextKey,
                transaction.snapshot.contextEpoch,
                failedFact,
                {},
                outError);
            return false;
        }

        if (cameraCarrier) {
            const JuicerCuda::Diffusion::DiffusionPreparedView diffusionView =
                diffusionLease.view();
            if (!diffusionLease.active() || !diffusionView.active ||
                diffusionView.executionDescriptor.contextEpoch !=
                    transaction.snapshot.contextEpoch ||
                diffusionView.executionDescriptor.frameSetHash !=
                    frameSet->hash ||
                resources->ownerContextKey != contextKey ||
                resources->contextEpoch != transaction.snapshot.contextEpoch) {
                set_scatter_halation_exact_admission_failure(
                    recipe,
                    *descriptor,
                    requestedWidth,
                    requestedHeight,
                    carrierSource,
                    shape,
                    0,
                    contextKey,
                    transaction.snapshot.contextEpoch,
                    "camera_diffusion_exact_context_lease",
                    {},
                    outError);
                return false;
            }
        }

        void* filterBlock = nullptr;
        std::string allocationError;
        if (!allocate_device_bytes(
                &filterBlock,
                shape.filterBytes,
                "scatter halation filter scratch",
                allocationError)) {
            set_scatter_halation_exact_admission_failure(
                recipe,
                *descriptor,
                requestedWidth,
                requestedHeight,
                carrierSource,
                shape,
                0,
                contextKey,
                transaction.snapshot.contextEpoch,
                "filter_allocation_block",
                allocationError,
                outError);
            return false;
        }

        void* carrierBlock = nullptr;
        if (carrierSource ==
                JuicerCuda::ScatterHalationCarrierSource::
                    DedicatedPreparedPlanes &&
            !allocate_device_bytes(
                &carrierBlock,
                shape.carrierBytes,
                "scatter halation dedicated carrier",
                allocationError)) {
            set_scatter_halation_exact_admission_failure(
                recipe,
                *descriptor,
                requestedWidth,
                requestedHeight,
                carrierSource,
                shape,
                shape.filterBytes,
                contextKey,
                transaction.snapshot.contextEpoch,
                "carrier_allocation_block",
                allocationError,
                outError);
            return false;
        }

        auto* filterBytes = static_cast<std::byte*>(filterBlock);
        scatterHalationFilterBlock = filterBlock;
        scatterHalationFilterTemp = reinterpret_cast<float*>(filterBytes);
        scatterHalationWeightedAccumulation = reinterpret_cast<float*>(
            filterBytes + shape.filterSecondPlaneOffset);
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
        scatterHalationDescriptor = *descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scatter_halation_resources(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!scatterHalationDescriptor && !scatterHalationFilterBlock &&
            !scatterHalationCarrierBlock) {
            return true;
        }
        if (!resources) {
            outError =
                "CUDA resources unavailable for scatter-halation release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque =
            cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        if (scatterHalationCarrierBlock) {
            void* carrierBlock = scatterHalationCarrierBlock;
            if (!retire_device_bytes(
                    carrierBlock,
                    retireStreamOpaque,
                    "scatter halation dedicated carrier",
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
        scatterHalationCarrierSource =
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes;
        return true;
    }

    bool Root::PreparedCudaFrame::State::allocate_scan_error_stage(std::string& outError) {
        outError.clear();
        free_scan_error_stage_now();

        ScanErrorFrameStage next{};
        if (!allocate_device_bytes(
                reinterpret_cast<void**>(&next.deviceFlag),
                sizeof(int),
                "frame scan error flag",
                outError)) {
            return false;
        }

        cudaError_t err =
            cudaMallocHost(reinterpret_cast<void**>(&next.hostFlag), sizeof(int));
        if (err != cudaSuccess) {
            std::string freeError;
            (void)free_device_bytes_now(next.deviceFlag, freeError);
            outError = std::string("cudaMallocHost(frame scan error staging) failed: ") +
                       (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        cudaEvent_t ev = nullptr;
        err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess || !ev) {
            if (ev) {
                cudaEventDestroy(ev);
            }
            cudaFreeHost(next.hostFlag);
            std::string freeError;
            (void)free_device_bytes_now(next.deviceFlag, freeError);
            outError = std::string("cudaEventCreateWithFlags(frame scan error staging) failed: ") +
                       (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        next.eventOpaque = reinterpret_cast<void*>(ev);

        next.active = true;
        scanErrorStage = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scan_error_stage_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        ScanErrorFrameStage& stage = scanErrorStage;
        if (!stage.active) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scan-error stage release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        if (stage.readbackPending && stage.hostFlag && stage.eventOpaque) {
            if (!JuicerCuda::retain_scan_error_readback(
                    *resources,
                    stage.hostFlag,
                    stage.eventOpaque,
                    outError)) {
                if (outError.empty()) {
                    outError = "frame scan-error readback retention failed";
                }
                return false;
            }
            stage.readbackPending = false;
        }

        if (stage.deviceFlag) {
            int* flag = stage.deviceFlag;
            if (!retire_device_bytes(
                    flag,
                    retireStreamOpaque,
                    "frame scan error flag",
                    outError)) {
                if (outError.empty()) {
                    outError = "frame scan-error flag retire failed";
                }
                return false;
            }
            stage.deviceFlag = nullptr;
        }

        if (stage.hostFlag) {
            cudaFreeHost(stage.hostFlag);
            stage.hostFlag = nullptr;
        }
        if (stage.eventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(stage.eventOpaque);
            cudaEventDestroy(ev);
            stage.eventOpaque = nullptr;
        }
        stage = ScanErrorFrameStage{};
        return true;
    }

    void Root::PreparedCudaFrame::State::free_scan_error_stage_now() noexcept {
        ScanErrorFrameStage& stage = scanErrorStage;
        if (stage.deviceFlag) {
            std::string freeError;
            (void)free_device_bytes_now(stage.deviceFlag, freeError);
        }
        if (stage.hostFlag) {
            cudaFreeHost(stage.hostFlag);
        }
        if (stage.eventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(stage.eventOpaque);
            cudaEventDestroy(ev);
        }
        stage = ScanErrorFrameStage{};
    }

    bool Root::PreparedCudaFrame::State::allocate_auto_exposure_workspace(
        const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
        std::string& outError) {
        outError.clear();
        free_auto_exposure_workspace_now();
        const int meterWidth = descriptor.previewWidth;
        const int meterHeight = descriptor.previewHeight;
        if (meterWidth <= 0 || meterHeight <= 0) {
            outError = "auto-exposure meter dimensions invalid";
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
            outError = "auto-exposure partial count invalid";
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
        std::string& outError) {
        outError.clear();
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        if (!workspace.active) {
            return true;
        }

        try {
            if (!resources) {
                outError = "CUDA resources unavailable for frame auto-exposure retire";
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
                std::string localError;
                if (retire_device_bytes(
                        raw,
                        retireStreamOpaque,
                        label,
                        localError)) {
                    ptr = nullptr;
                    return;
                }
                retiredAll = false;
                if (localError.empty()) {
                    outError = std::string(label ? label : "frame auto-exposure buffer") +
                               " retire failed";
                } else {
                    outError = localError;
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
            outError = "frame auto-exposure retire threw";
        }
        if (outError.empty()) {
            outError = "frame auto-exposure retire failed";
        }
        return false;
    }

    void Root::PreparedCudaFrame::State::free_auto_exposure_workspace_now() noexcept {
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        std::string freeError;
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
        std::string& outError) {
        outError.clear();
        if (!resources || !transaction.active || transaction.committed) {
            outError = "prepared frame is not active for use-event submission";
            return false;
        }
        if (frameUseEventSubmitted) {
            return true;
        }
        remember_stream(cudaStreamOpaque);

        auto record_use_event = [&](JuicerCuda::Resources& target,
                                    const char* label) {
            return JuicerCuda::record_frame_use_event(
                target,
                cudaStreamOpaque,
                label,
                outError);
        };

        if (!record_use_event(*resources, "frame use")) {
            return false;
        }
        if (grainStaticResources && visualGrainPrepared) {
            if (!record_use_event(*grainStaticResources, "grain-static use")) {
                return false;
            }
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
        const WorkspaceRequest& request,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!request.needOptics && !request.needSpatialDir) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scratch lease";
            return false;
        }
        if (request.requestedWidth <= 0 || request.requestedHeight <= 0) {
            outError = "frame scratch dimensions invalid";
            return false;
        }
        if (request.needSpatialDir) {
            if (request.spatialDirDescriptorHash == 0) {
                outError = "spatial DIR workspace missing descriptor hash";
                return false;
            }
            if (!spatial_dir_roles_match_tier(
                    request.spatialDirScratchTier,
                    request.spatialDirPlaneRoles)) {
                outError = "spatial DIR workspace plane roles do not match scratch tier";
                return false;
            }
            if (!spatial_dir_roles_match_tier(
                    request.spatialDirTargetScratchTier,
                    request.spatialDirTargetPlaneRoles)) {
                outError = "spatial DIR workspace target plane roles do not match scratch tier";
                return false;
            }
        }
        if (request.aliasScannerRgbFromSpatialDirFiltered &&
            (!request.needOptics ||
             !request.needSpatialDir ||
             request.spatialDirPlaneRoles.filteredCorrectionPlanes != 3 ||
             (request.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 0 &&
              request.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 2 &&
              request.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3))) {
            outError = "scanner RGB spatial DIR alias request is invalid";
            return false;
        }

        auto requests_match = [](const WorkspaceRequest& a, const WorkspaceRequest& b) noexcept {
            return a.needOptics == b.needOptics &&
                   a.needSpatialDir == b.needSpatialDir &&
                   a.spatialDirDescriptorHash == b.spatialDirDescriptorHash &&
                   a.spatialDirScratchTier == b.spatialDirScratchTier &&
                   a.spatialDirPlaneRoles.rawCorrectionPlanes == b.spatialDirPlaneRoles.rawCorrectionPlanes &&
                   a.spatialDirPlaneRoles.filteredCorrectionPlanes == b.spatialDirPlaneRoles.filteredCorrectionPlanes &&
                   a.spatialDirPlaneRoles.filterTempPlanes == b.spatialDirPlaneRoles.filterTempPlanes &&
                   a.spatialDirPlaneRoles.cachedLogRawPlanes == b.spatialDirPlaneRoles.cachedLogRawPlanes &&
                   a.spatialDirTargetScratchTier == b.spatialDirTargetScratchTier &&
                   a.spatialDirTargetPlaneRoles.rawCorrectionPlanes == b.spatialDirTargetPlaneRoles.rawCorrectionPlanes &&
                   a.spatialDirTargetPlaneRoles.filteredCorrectionPlanes == b.spatialDirTargetPlaneRoles.filteredCorrectionPlanes &&
                   a.spatialDirTargetPlaneRoles.filterTempPlanes == b.spatialDirTargetPlaneRoles.filterTempPlanes &&
                   a.spatialDirTargetPlaneRoles.cachedLogRawPlanes == b.spatialDirTargetPlaneRoles.cachedLogRawPlanes &&
                   a.requestedWidth == b.requestedWidth &&
                   a.requestedHeight == b.requestedHeight &&
                   a.needBlurred == b.needBlurred &&
                   a.aliasScannerRgbFromSpatialDirFiltered ==
                       b.aliasScannerRgbFromSpatialDirFiltered &&
                   a.needAux == b.needAux &&
                   a.needSharedTmp == b.needSharedTmp &&
                   a.needGrainFrameUniforms == b.needGrainFrameUniforms &&
                   a.needGrainLayerWork == b.needGrainLayerWork &&
                   a.needGrainShared == b.needGrainShared &&
                   a.needGateTransmittance == b.needGateTransmittance &&
                   a.needFilmDustTransmittance == b.needFilmDustTransmittance &&
                   a.gateWidth == b.gateWidth && a.gateHeight == b.gateHeight;
        };

        if (scratchWorkspace.retainedLeaseActive) {
            if (!requests_match(scratchWorkspace.request, request)) {
#if JUICER_DIAGNOSTICS_COMPILED
                trace_workspace_request_mismatch(transaction, scratchWorkspace.request, request);
#endif
                outError = "prepared frame scratch workspace request mismatch";
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
        scratchWorkspace.request = request;
        scratchWorkspace.postFrameRequest = request;
        scratchWorkspace.postFrameRequestActive = true;
        scratchWorkspace.retainedLeaseActive = true;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scratch_workspace_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.retainedLeaseActive) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scratch release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        if (!submit_frame_use_event(retireStreamOpaque, outError)) {
            if (outError.empty()) {
                outError = "frame scratch use-event submission failed";
            }
            return false;
        }
        if (!JuicerCuda::release_retained_frame_scratch_lease(
                *resources,
                transaction.leaseGeneration,
                outError)) {
            if (outError.empty()) {
                outError = "retained frame scratch lease release failed";
            }
            return false;
        }
        workspace = FrameScratchWorkspace{};
        return true;
    }

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
        const WorkspaceRequest& request,
        std::uint64_t leaseGeneration) noexcept
        : _request(request), _leaseGeneration(leaseGeneration), _active(leaseGeneration != 0) {
    }

    bool Root::PreparedCudaFrame::WorkspaceLeaseMarker::active() const noexcept {
        return _active;
    }

    bool Root::PreparedCudaFrame::active() const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed;
    }

    JuicerCuda::ResourceManager::ScratchRequestDescriptor
    Root::PreparedCudaFrame::State::post_frame_scratch_request_descriptor() const noexcept {
        if (!scratchWorkspace.retainedLeaseActive) {
            return JuicerCuda::ResourceManager::ScratchRequestDescriptor{};
        }
        const WorkspaceRequest& request = scratchWorkspace.postFrameRequestActive
                                              ? scratchWorkspace.postFrameRequest
                                              : scratchWorkspace.request;
        return make_workspace_scratch_request_descriptor(request);
    }

    Root::PreparedCudaFrame::WorkspaceLeaseMarker
    Root::PreparedCudaFrame::workspace_lease() const noexcept {
        if (!active() || !_state->workspaceRequest.has_any_family()) {
            return WorkspaceLeaseMarker{};
        }
        return WorkspaceLeaseMarker(
            _state->workspaceRequest,
            _state->transaction.leaseGeneration);
    }

    const Root::PreparedCudaFrame::WorkspaceRequest&
    Root::PreparedCudaFrame::active_workspace_request(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        if (_state &&
            _state->scratchWorkspace.retainedLeaseActive &&
            _state->scratchWorkspace.request.needSpatialDir &&
            workspace._request.needSpatialDir &&
            _state->scratchWorkspace.request.spatialDirDescriptorHash ==
                workspace._request.spatialDirDescriptorHash) {
            return _state->scratchWorkspace.request;
        }
        return workspace._request;
    }

    bool Root::PreparedCudaFrame::workspace_marker_matches_current_frame(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed &&
               workspace.active() &&
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
        return true;
    }

    bool Root::PreparedCudaFrame::finish(void* cudaStreamOpaque, std::string& outError) {
        outError.clear();
        if (!_state || !_state->root || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        _state->remember_stream(cudaStreamOpaque);
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor postFrameScratchRequest =
            _state->post_frame_scratch_request_descriptor();
        std::string releaseError;
        if (!_state->release_scatter_halation_resources(
                cudaStreamOpaque,
                releaseError)) {
            outError = releaseError.empty()
                           ? "scatter-halation resource release failed"
                           : releaseError;
            return false;
        }
        if (_state->diffusionLease.active() &&
            !JuicerCuda::Diffusion::release_diffusion_resources(
                _state->resources->diffusion,
                _state->diffusionLease,
                cudaStreamOpaque,
                false,
                releaseError)) {
            outError = releaseError.empty()
                           ? "diffusion resource release failed"
                           : releaseError;
            return false;
        }
        if (!_state->release_scan_error_stage_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame scan-error stage release failed" : releaseError;
            return false;
        }
        if (!_state->release_scratch_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame scratch workspace release failed" : releaseError;
            return false;
        }
        if (!_state->release_auto_exposure_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame auto-exposure retire failed" : releaseError;
            if (JTRACE_ENABLED(1)) {
                std::string msg = "frame_auto_exposure_retire_failed finish=1 error=";
                msg += outError;
                JTRACE("CUDA", msg);
            }
            return false;
        }
        if (!_state->transfer_remaining_device_allocations(releaseError)) {
            outError = releaseError.empty()
                           ? "remaining frame allocation transfer failed"
                           : releaseError;
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
                outError)) {
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
            try {
                std::string releaseError;
                const JuicerCuda::ResourceManager::ScratchRequestDescriptor postFrameScratchRequest =
                    _state->post_frame_scratch_request_descriptor();
                if (!_state->release_scatter_halation_resources(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg =
                        "scatter_halation_resource_release_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (_state->resources && _state->diffusionLease.active() &&
                    !JuicerCuda::Diffusion::release_diffusion_resources(
                        _state->resources->diffusion,
                        _state->diffusionLease,
                        _state->lastCudaStreamOpaque,
                        true,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg =
                        "diffusion_resource_release_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                (void)_state->release_scan_error_stage_after_use(
                    _state->lastCudaStreamOpaque,
                    releaseError);
                if (!_state->release_scratch_workspace_after_use(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_scratch_workspace_release_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (!_state->release_auto_exposure_workspace_after_use(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_auto_exposure_retire_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (!_state->transfer_remaining_device_allocations(
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg =
                        "frame_remaining_allocation_transfer_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (_state->root && _state->resources && _state->transaction.active &&
                    !_state->transaction.committed &&
                    !JuicerCuda::ResourceManager::command_shed_post_frame_scratch(
                        _state->transaction,
                        *_state->resources,
                        postFrameScratchRequest,
                        _state->lastCudaStreamOpaque,
                        "command_shed_post_frame_scratch_abort",
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_post_frame_scratch_shed_failed abort=1";
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
        if (_state && _state->root && _state->transaction.active && !_state->transaction.committed) {
            JuicerCuda::ResourceManager::rollback_submission(_state->transaction);
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
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        const WorkspaceRequest& activeRequest = active_workspace_request(workspace);
        if (!_state->ensure_scratch_workspace(activeRequest, cudaStreamOpaque, outError)) {
            const bool marksContextLoss =
                !JuicerCuda::ResourceManager::error_is_allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"acquire_frame_scratch_workspace"},
                marksContextLoss
                    ? "CUDA frame scratch workspace acquisition failed"
                    : "CUDA frame scratch allocation capacity exhausted");
            return false;
        }
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(activeRequest);
        if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            const bool marksContextLoss =
                !JuicerCuda::ResourceManager::error_is_allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_optics_scratch"},
                marksContextLoss
                    ? "CUDA optics scratch allocation failed"
                    : "CUDA optics scratch allocation capacity exhausted");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_spatial_dir_resources(
        const Spektrafilm::SpatialDirDescriptor& descriptor,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError) || !workspace._request.needSpatialDir) {
            if (outError.empty()) {
                outError = "spatial DIR workspace was not admitted";
            }
            return false;
        }
        if (descriptor.hash == 0 || descriptor.dirRecipeHash == 0 ||
            !(descriptor.gaussianSigmaPixels > 0.0f)) {
            outError = "spatial DIR descriptor is invalid";
            return false;
        }
        const bool targetMatchesDescriptor =
            workspace._request.spatialDirTargetScratchTier == descriptor.targetScratchTier &&
            dir_plane_roles_equal(
                workspace._request.spatialDirTargetPlaneRoles,
                descriptor.targetPlaneRoles);
        const bool targetIsFusedScannerPostAlias =
            spatial_dir_alias_target_uses_build_roles(workspace._request);
        if (workspace._request.spatialDirDescriptorHash != descriptor.hash ||
            workspace._request.spatialDirScratchTier != descriptor.scratchTier ||
            !dir_plane_roles_equal(workspace._request.spatialDirPlaneRoles, descriptor.planeRoles) ||
            (!targetMatchesDescriptor && !targetIsFusedScannerPostAlias)) {
            outError = "spatial DIR workspace descriptor identity mismatch";
            return false;
        }
        if (!spatial_dir_roles_match_tier(descriptor.scratchTier, descriptor.planeRoles)) {
            outError = "spatial DIR descriptor requested unsupported scratch tier or plane roles";
            return false;
        }
        if (!spatial_dir_roles_match_tier(descriptor.targetScratchTier, descriptor.targetPlaneRoles)) {
            outError = "spatial DIR descriptor requested unsupported target scratch tier or plane roles";
            return false;
        }

        std::array<WorkspaceRequest, 6> scratchCandidates{};
        int scratchCandidateCount = 0;
        auto add_scratch_candidate = [&](const WorkspaceRequest& candidate) {
            if (scratchCandidateCount >= static_cast<int>(scratchCandidates.size())) {
                return;
            }
            if (!spatial_dir_roles_match_tier(
                    candidate.spatialDirScratchTier,
                    candidate.spatialDirPlaneRoles) ||
                !spatial_dir_roles_match_tier(
                    candidate.spatialDirTargetScratchTier,
                    candidate.spatialDirTargetPlaneRoles)) {
                return;
            }
            for (int index = 0; index < scratchCandidateCount; ++index) {
                const WorkspaceRequest& existing = scratchCandidates[static_cast<std::size_t>(index)];
                if (existing.spatialDirScratchTier == candidate.spatialDirScratchTier &&
                    existing.spatialDirTargetScratchTier == candidate.spatialDirTargetScratchTier &&
                    dir_plane_roles_equal(existing.spatialDirPlaneRoles, candidate.spatialDirPlaneRoles) &&
                    dir_plane_roles_equal(
                        existing.spatialDirTargetPlaneRoles,
                        candidate.spatialDirTargetPlaneRoles)) {
                    return;
                }
            }
            scratchCandidates[static_cast<std::size_t>(scratchCandidateCount)] = candidate;
            ++scratchCandidateCount;
        };
        if (spatial_dir_descriptor_has_strict_yvv(descriptor)) {
            const bool aliasTarget =
                spatial_dir_alias_target_uses_build_roles(workspace._request);
            if (aliasTarget) {
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBuild =
                    strict_yvv_cached_lograw_alias_plane_roles();
                Spektrafilm::DirScratchPlaneRoles cachedLogRawTarget = cachedLogRawBuild;
                add_scratch_candidate(strict_yvv_workspace_candidate(
                    workspace._request,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBuild,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawTarget));
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBgBuild =
                    strict_yvv_cached_lograw_bg_alias_plane_roles();
                Spektrafilm::DirScratchPlaneRoles cachedLogRawBgTarget = cachedLogRawBgBuild;
                add_scratch_candidate(strict_yvv_workspace_candidate(
                    workspace._request,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBgBuild,
                    Spektrafilm::DirScratchTier::Tier2,
                    cachedLogRawBgTarget));
            }
            add_scratch_candidate(workspace._request);
            Spektrafilm::DirScratchPlaneRoles lowScratchBuild =
                strict_yvv_low_scratch_plane_roles();
            Spektrafilm::DirScratchPlaneRoles lowScratchTarget = lowScratchBuild;
            lowScratchTarget.cachedLogRawPlanes =
                aliasTarget ? 0 : descriptor.targetPlaneRoles.cachedLogRawPlanes;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace._request,
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
                workspace._request,
                Spektrafilm::DirScratchTier::Tier1IChannels,
                singleTempBuild,
                aliasTarget ? Spektrafilm::DirScratchTier::Tier1IChannels : descriptor.targetScratchTier,
                singleTempTarget));

            Spektrafilm::DirScratchPlaneRoles streamedBuild =
                strict_yvv_component_streamed_plane_roles();
            Spektrafilm::DirScratchPlaneRoles streamedTarget = streamedBuild;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace._request,
                Spektrafilm::DirScratchTier::Tier1IChannels,
                streamedBuild,
                Spektrafilm::DirScratchTier::Tier1IChannels,
                streamedTarget));
        } else {
            add_scratch_candidate(workspace._request);
        }
        if (scratchCandidateCount <= 0) {
            outError = "spatial DIR descriptor produced no supported scratch candidates";
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
            msg += " source_contract=";
            msg += Spektrafilm::to_cstr(descriptor.sourceContract);
            msg += " boundary_mode=";
            msg += Spektrafilm::to_cstr(descriptor.boundaryMode);
            msg += " scratch_tier=";
            msg += Spektrafilm::to_cstr(descriptor.scratchTier);
            msg += " target_scratch_tier=";
            msg += Spektrafilm::to_cstr(descriptor.targetScratchTier);
            msg += " approximation=";
            msg += Spektrafilm::to_cstr(descriptor.approximation);
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
        WorkspaceRequest admittedRequest{};
        std::string scratchAdmissionError;
        for (int candidateIndex = 0; candidateIndex < scratchCandidateCount; ++candidateIndex) {
            const WorkspaceRequest& candidate =
                scratchCandidates[static_cast<std::size_t>(candidateIndex)];
#if JUICER_DIAGNOSTICS_COMPILED
            if (JTRACE_ENABLED(1)) {
                const Spektrafilm::DirScratchPlaneRoles& roles = candidate.spatialDirPlaneRoles;
                const Spektrafilm::DirScratchPlaneRoles& targetRoles =
                    candidate.spatialDirTargetPlaneRoles;
                std::string msg = "event=spatial_dir_scratch_candidate descriptor_hash=";
                msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
                msg += " candidate_index=";
                msg += std::to_string(candidateIndex);
                msg += " strict_yvv_shape=";
                msg += strict_yvv_candidate_label(roles);
                msg += " scratch_tier=";
                msg += Spektrafilm::to_cstr(candidate.spatialDirScratchTier);
                msg += " target_scratch_tier=";
                msg += Spektrafilm::to_cstr(candidate.spatialDirTargetScratchTier);
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
            if (_state->scratchWorkspace.retainedLeaseActive) {
                _state->scratchWorkspace.request = candidate;
                _state->scratchWorkspace.postFrameRequest = candidate;
                _state->scratchWorkspace.postFrameRequestActive = true;
            } else if (!_state->ensure_scratch_workspace(candidate, cudaStreamOpaque, scratchAdmissionError)) {
                if (JuicerCuda::ResourceManager::error_is_allocation_capacity_exhausted(
                        scratchAdmissionError) &&
                    candidateIndex + 1 < scratchCandidateCount) {
                    continue;
                }
                outError = scratchAdmissionError;
                break;
            }

            const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
                make_workspace_scratch_request_descriptor(candidate);
            if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_scratch(
                    _state->transaction,
                    *_state->resources,
                    scratchRequest,
                    cudaStreamOpaque,
                    scratchAdmissionError)) {
                if (JuicerCuda::ResourceManager::error_is_allocation_capacity_exhausted(
                        scratchAdmissionError) &&
                    candidateIndex + 1 < scratchCandidateCount) {
                    continue;
                }
                outError = scratchAdmissionError;
                break;
            }
            admittedRequest = candidate;
            scratchPrepared = true;
            outError.clear();
            break;
        }
        if (!scratchPrepared) {
            if (outError.empty()) {
                outError = scratchAdmissionError.empty()
                               ? "spatial DIR scratch admission failed"
                               : scratchAdmissionError;
            }
            const bool marksContextLoss =
                !JuicerCuda::ResourceManager::error_is_allocation_capacity_exhausted(outError);
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_spatial_dir_scratch"},
                marksContextLoss
                    ? "CUDA spatial DIR scratch allocation failed"
                    : "CUDA spatial DIR scratch allocation capacity exhausted");
            return false;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const Spektrafilm::DirScratchPlaneRoles& roles = admittedRequest.spatialDirPlaneRoles;
            std::string msg = "event=spatial_dir_scratch_admission descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
            msg += " scratch_source=";
            msg += "retained";
            msg += " scratch_tier=";
            msg += Spektrafilm::to_cstr(admittedRequest.spatialDirScratchTier);
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
        const float sigmas[4] = {
            descriptor.gaussianSigmaPixels,
            descriptor.exponentialSigmaPixels[0],
            descriptor.exponentialSigmaPixels[1],
            descriptor.exponentialSigmaPixels[2]};
        for (int slot = 0; slot < 4; ++slot) {
            if (slot > 0 && !(descriptor.exponentialWeights[slot - 1] > 0.0f)) {
                continue;
            }
            if (sigmas[slot] >= 3.0f) {
                continue;
            }
            if (!JuicerCuda::ensure_spatial_dir_kernel(
                    *_state->resources,
                    _state->resources->spatialDirKernels[static_cast<std::size_t>(slot)],
                    sigmas[slot],
                    outError)) {
                _state->set_failure(
                    PreparedCudaFailureStage{"command_ensure_spatial_dir_kernel"},
                    "CUDA spatial DIR kernel upload failed");
                return false;
            }
        }
        _state->spatialDirDescriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::stage_spatial_dir_cached_log_raw_for_final_develop(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        const WorkspaceRequest& effectiveRequest =
            (_state->scratchWorkspace.request.needSpatialDir &&
             _state->scratchWorkspace.request.spatialDirDescriptorHash ==
                 workspace._request.spatialDirDescriptorHash)
                ? _state->scratchWorkspace.request
                : workspace._request;
        if (!effectiveRequest.needSpatialDir ||
            effectiveRequest.spatialDirTargetPlaneRoles.cachedLogRawPlanes == 0) {
            return true;
        }
        if (effectiveRequest.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
            outError = "spatial DIR cached log raw final target roles invalid";
            return false;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive) {
            outError = "spatial DIR cached log raw staging requires an active scratch workspace";
            return false;
        }

        JuicerCuda::SpatialDirCachedLogRawStageStats stageStats{};
        _state->remember_stream(cudaStreamOpaque);
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(effectiveRequest);
        const bool ok = JuicerCuda::ResourceManager::command_ensure_spatial_dir_cached_log_raw_stage(
            _state->transaction,
            *_state->resources,
            scratchRequest,
            cudaStreamOpaque,
            stageStats,
            outError);
        if (ok) {
            WorkspaceRequest& activeRequest = _state->scratchWorkspace.request;
            activeRequest.spatialDirScratchTier =
                activeRequest.spatialDirTargetScratchTier;
            activeRequest.spatialDirPlaneRoles =
                activeRequest.spatialDirTargetPlaneRoles;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace._request.spatialDirDescriptorHash;
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
            if (!outError.empty()) {
                msg += " error=";
                msg += outError;
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
        std::string& outError) {
        outError.clear();
        if (!descriptor.active()) {
            return true;
        }
        if (descriptor.lensBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerLensBlurKernel,
                descriptor.lensBlurSigmaPx,
                descriptor.lensBlurRadius,
                outError)) {
            return false;
        }
        if (descriptor.unsharpSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerUnsharpKernel,
                descriptor.unsharpSigmaPx,
                descriptor.unsharpRadius,
                outError)) {
            return false;
        }
        if (descriptor.glareActive && descriptor.glareBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerGlareKernel,
                descriptor.glareBlurSigmaPx,
                descriptor.glareBlurRadius,
                outError)) {
            return false;
        }
        _state->scannerPostEffectsDescriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_scan_error_stage(
        int*& outScanErrorFlag,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        outScanErrorFlag = nullptr;
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        bool previousScanErrorDetected = false;
        if (!JuicerCuda::poll_scan_error_readbacks(
                *_state->resources,
                cudaStreamOpaque,
                previousScanErrorDetected,
                outError)) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_pending_readback"},
                "CUDA scan error validation failed");
            return false;
        }
        if (previousScanErrorDetected) {
            JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_previous_readback"},
                "CUDA scan error validation failed");
            outError = "previous scan produced non-finite RGB";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        outScanErrorFlag = stage.deviceFlag;
        if (!outScanErrorFlag) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_missing"},
                "CUDA scan error validation failed");
            outError = "scan error flag missing after allocation";
            return false;
        }

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const cudaError_t flagErr = cudaMemsetAsync(outScanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_memset"},
                "CUDA scan error validation failed");
            outError = "CUDA scan error flag memset failed";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::finalize_scan_error_stage(
        int* scanErrorFlag,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        if (!scanErrorFlag) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_missing"},
                "CUDA scan error validation failed");
            outError = "scan error flag missing after allocation";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        if (scanErrorFlag != stage.deviceFlag) {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_flag_mismatch"},
                "CUDA scan error validation failed");
            outError = "scan error flag does not match prepared frame stage";
            return false;
        }

        cudaEvent_t scanEvent = stage.eventOpaque
                                    ? reinterpret_cast<cudaEvent_t>(stage.eventOpaque)
                                    : nullptr;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        if (stage.hostFlag && scanEvent) {
            cudaError_t flagErr = cudaMemcpyAsync(
                stage.hostFlag,
                scanErrorFlag,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream);
            if (flagErr != cudaSuccess) {
                _state->set_failure(
                    PreparedCudaFailureStage{"scan_error_flag_readback"},
                    "CUDA scan error validation failed");
                outError = "CUDA scan error flag readback failed";
                return false;
            }
            const cudaError_t evErr = cudaEventRecord(scanEvent, stream);
            if (evErr != cudaSuccess) {
                _state->set_failure(
                    PreparedCudaFailureStage{"scan_error_event_record"},
                    "CUDA scan error validation failed");
                outError = "CUDA scan error event record failed";
                return false;
            }
            stage.readbackPending = true;
        } else {
            _state->set_failure(
                PreparedCudaFailureStage{"scan_error_staging_missing"},
                "CUDA scan error validation failed");
            outError = "scan error host/event staging missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::checkpoint_large_scratch_transition(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        const char* stageTag,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        _state->remember_stream(cudaStreamOpaque);
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(workspace._request);
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
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ensure_gaussian_kernel(
                *_state->resources,
                kernel,
                sigma,
                radius,
                outError)) {
            _state->set_failure(
                PreparedCudaFailureStage{"command_ensure_gaussian_kernel"},
                "CUDA gaussian kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_visual_grain_resources(
        Root& root,
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active ||
            _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        const bool active = _state->visualGrainDescriptor.has_value();
        std::shared_ptr<const JuicerAssets::StaticNoisePayloadSet> payloads;
        if (active) {
            payloads = root.assets().static_noise_payloads();
            if (!payloads || !payloads->stbn.valid || !payloads->wang.valid) {
                outError = payloads
                               ? (!payloads->stbn.valid
                                      ? payloads->stbn.error
                                      : payloads->wang.error)
                               : "MissingRequiredResource phase=grain_static field=payloads";
                return false;
            }
            const Spektrafilm::VisualGrainFrameDescriptor& descriptor =
                *_state->visualGrainDescriptor;
            if (payloads->version != descriptor.staticNoiseVersion ||
                payloads->stbn.version != descriptor.staticNoiseVersion ||
                payloads->wang.version != descriptor.staticNoiseVersion ||
                payloads->stbn.width <= 0 || payloads->stbn.height <= 0 ||
                payloads->stbn.frames <= 0 || payloads->wang.width <= 0 ||
                payloads->wang.height <= 0 || payloads->wang.count <= 0 ||
                payloads->wang.colors <= 0) {
                outError =
                    "ResourceDescriptorMismatch phase=grain_static field=asset_identity";
                return false;
            }
        }

        const auto& snapshot = _state->transaction.snapshot;
        detail::GrainStaticMembershipChange membershipChange;
        if (!root.apply_grain_static_membership_and_copy_owner(
                deviceContextKey,
                snapshot.contextEpoch,
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
            snapshot.contextEpoch};
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
        if (!JuicerCuda::ensure_grain_static_assets_uploaded(
                *_state->grainStaticResources,
                *payloads,
                descriptor.staticNoiseVersion,
                cudaStreamOpaque,
                outError)) {
            return fail_preparation(
                "ensure_grain_static_assets_uploaded",
                "CUDA grain-static upload failed");
        }

        const JuicerCuda::Resources& staticResources =
            *_state->grainStaticResources;
        if (!staticResources.stbnData || !staticResources.wangTilesData ||
            !staticResources.wangLutData ||
            staticResources.grainStaticAssetVersion !=
                descriptor.staticNoiseVersion ||
            staticResources.stbnWidth != payloads->stbn.width ||
            staticResources.stbnHeight != payloads->stbn.height ||
            staticResources.stbnFrames != payloads->stbn.frames ||
            staticResources.wangWidth != payloads->wang.width ||
            staticResources.wangHeight != payloads->wang.height ||
            staticResources.wangCount != payloads->wang.count ||
            staticResources.wangColors != payloads->wang.colors) {
            outError =
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
                    outError =
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
                        outError) ||
                    !kernel.weights || kernel.radius != gaussian.radius ||
                    kernel.sigma != gaussian.sigmaPx) {
                    if (outError.empty()) {
                        outError =
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
                outError =
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
                    outError =
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
                        outError =
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
    JuicerCuda::PreparedVisualGrainView
    Root::PreparedCudaFrame::visual_grain_resources() const noexcept {
        JuicerCuda::PreparedVisualGrainView view{};
        if (!_state || !_state->visualGrainPrepared ||
            !_state->visualGrainDescriptor.has_value() ||
            !_state->grainStaticResources ||
            !_state->transaction.active ||
            _state->transaction.committed) {
            return view;
        }

        const Spektrafilm::VisualGrainFrameDescriptor& descriptor =
            *_state->visualGrainDescriptor;
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
        view.staticNoise.version =
            staticResources.grainStaticAssetVersion;
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
            view.staticNoise.wangColors > 0 &&
            view.staticNoise.version == descriptor.staticNoiseVersion;
        auto gaussian_ready =
            [](const Spektrafilm::VisualGrainGaussian& gaussian,
               const JuicerCuda::VisualGrainPreparedGaussianView& prepared) {
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
                   : JuicerCuda::PreparedVisualGrainView{};
    }

    const Spektrafilm::FilmJuicerEffectsFrameDescriptor*
    Root::PreparedCudaFrame::film_juicer_effects_descriptor() const noexcept {
        if (!active() || !_state->effectsDescriptor.has_value()) {
            return nullptr;
        }
        return &*_state->effectsDescriptor;
    }

    Root::PreparedCudaFrame::VisualGrainWorkspaceView
    Root::PreparedCudaFrame::visual_grain_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        VisualGrainWorkspaceView view{};
        if (!_state || !_state->visualGrainPrepared ||
            !_state->visualGrainDescriptor.has_value() ||
            !workspace_marker_matches_current_frame(workspace) ||
            !_state->scratchWorkspace.retainedLeaseActive) {
            return view;
        }
        const WorkspaceRequest& request =
            active_workspace_request(workspace);
        if (!request.needOptics) {
            return view;
        }
        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->resources->scannerScratch;
        view.filterTemp =
            request.needSharedTmp ? scratch.tmp : nullptr;
        view.scaleWork =
            request.needBlurred ? scratch.blurred : nullptr;
        view.deltaAccum = request.needAux ? scratch.aux : nullptr;
        view.layerWork =
            request.needGrainLayerWork ? scratch.grainTmp : nullptr;
        view.sharedDelta =
            request.needGrainShared ? scratch.grainTmpShared : nullptr;
        view.frameUniforms = request.needGrainFrameUniforms
                                 ? scratch.grainFrameUniforms
                                 : nullptr;
        view.scratchShape =
            _state->visualGrainDescriptor->scratchShape;

        bool ready = view.filterTemp && view.scaleWork &&
                     view.deltaAccum && view.frameUniforms;
        switch (view.scratchShape) {
            case Spektrafilm::VisualGrainScratchShape::Streamed:
                ready = ready && !view.layerWork &&
                        !view.sharedDelta &&
                        !request.needGrainLayerWork &&
                        !request.needGrainShared;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedShared:
                ready = ready && !view.layerWork &&
                        view.sharedDelta &&
                        !request.needGrainLayerWork &&
                        request.needGrainShared;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayers:
                ready = ready && view.layerWork &&
                        !view.sharedDelta &&
                        request.needGrainLayerWork &&
                        !request.needGrainShared;
                break;
            case Spektrafilm::VisualGrainScratchShape::StreamedLayersShared:
                ready = ready && view.layerWork &&
                        view.sharedDelta &&
                        request.needGrainLayerWork &&
                        request.needGrainShared;
                break;
            case Spektrafilm::VisualGrainScratchShape::None:
            default:
                ready = false;
                break;
        }
        view.active = ready;
        return view.active ? view : VisualGrainWorkspaceView{};
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
        const WorkspaceRequest& request = active_workspace_request(workspace);
        if (!request.needOptics) {
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
            !_state->scratchWorkspace.retainedLeaseActive) {
            return view;
        }
        const WorkspaceRequest& request = active_workspace_request(workspace);
        if (!request.needOptics) {
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
            !_state->focusedFilmRawConfig || !_state->focusedScannerColor ||
            !_state->focusedScanMedium || !_state->focusedScanLut) {
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
        view.film.tablesAx = resources.tablesAx;
        view.film.tablesAy = resources.tablesAy;
        view.film.tablesAz = resources.tablesAz;
        view.film.tablesIllum = resources.tablesIllum;
        view.film.tablesK = resources.tablesK;
        std::copy_n(resources.spdSInv, 9, view.film.spdSInv);
        view.film.hanatosLut = resources.hanatosLut;
        view.film.hanatosN = resources.hanatosN;
        view.film.hanatosLutIntegrated = resources.hanatosLutIntegrated;
        view.film.hanatosNIntegrated = resources.hanatosNIntegrated;
        view.film.mallettBasis = resources.mallettBasis;
        view.film.mallettBasisK = resources.mallettBasisK;
        std::copy_n(_state->focusedFilmRawConfig->inputRGBToXYZ.m, 9, view.film.inputRGBToXYZ);
        std::copy_n(_state->focusedFilmRawConfig->inputXYZAdapt.m, 9, view.film.inputXYZAdapt);
        view.film.applyInputChromaticAdapt = _state->focusedFilmRawConfig->applyInputChromaticAdapt ? 1 : 0;
        std::copy_n(resources.refIllumWhiteXYZ, 3, view.film.refIllumWhiteXYZ);
        view.film.finalSensitivityHash = resources.filmFinalSensitivityHash;
        view.film.normalizedDensityCurvesHash = resources.filmDensityCurvesHash;
        view.film.dirCouplersHash = resources.filmDirHash;
        view.scanMedium = _state->focusedScanMedium;
        view.scanLut = _state->focusedScanLut;
        view.scannerColor = _state->focusedScannerColor;
        view.densityBoundsHash = resources.routeDensityBoundsHash;
        view.scannerDescriptorHash = resources.routeScannerDescriptorHash;
        view.selectedMethod = resources.filmRgbToRawMethod;
        view.active = view.scanMedium && view.scanLut->canonical_ready() &&
                      view.densityBoundsHash != 0 && view.scannerDescriptorHash != 0;
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
        view.fullFrameWidth = _state->workspaceRequest.requestedWidth;
        view.fullFrameHeight = _state->workspaceRequest.requestedHeight;
        view.carrierSource = _state->scatterHalationCarrierSource;
        view.currentCarrier = _state->scatterHalationCarrier;
        view.filterTemp = _state->scatterHalationFilterTemp;
        view.weightedAccumulation =
            _state->scatterHalationWeightedAccumulation;
        return view;
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
        std::string& outError) {
        outError.clear();
        if (!active() || !_state->resources ||
            !_state->diffusionLease.active()) {
            outError = "prepared diffusion lease is not active";
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

    Root::PreparedCudaFrame::PrintPreparedView Root::PreparedCudaFrame::print_resources() const noexcept {
        PrintPreparedView view{};
        if (!_state || !_state->resources || !_state->printRecipe ||
            !_state->transaction.active || _state->transaction.committed ||
            _state->printDescriptors.hash == 0) {
            return view;
        }
        const JuicerCuda::Resources& resources = *_state->resources;
        const JuicerCuda::PrintResourceDescriptors& descriptors = _state->printDescriptors;
        view.filmDensityTables.epsC = resources.printFilmDensityTables.epsC;
        view.filmDensityTables.epsM = resources.printFilmDensityTables.epsM;
        view.filmDensityTables.epsY = resources.printFilmDensityTables.epsY;
        view.filmDensityTables.baseDensityMin = resources.printFilmDensityTables.baseDensityMin;
        view.filmDensityTables.K = resources.printFilmDensityTables.K;
        view.filmDensityTables.hasBaseline = resources.printFilmDensityTables.hasBaseline;
        view.filmDensityTables.invYn = resources.printFilmDensityTables.invYn;
        view.printSensC = {resources.printSensC.x, resources.printSensC.y, resources.printSensC.n, resources.printSensC.domainBegin, resources.printSensC.domainEnd};
        view.printSensM = {resources.printSensM.x, resources.printSensM.y, resources.printSensM.n, resources.printSensM.domainBegin, resources.printSensM.domainEnd};
        view.printSensY = {resources.printSensY.x, resources.printSensY.y, resources.printSensY.n, resources.printSensY.domainBegin, resources.printSensY.domainEnd};
        view.printDcC = {resources.printDcC.x, resources.printDcC.y, resources.printDcC.n, resources.printDcC.domainBegin, resources.printDcC.domainEnd};
        view.printDcM = {resources.printDcM.x, resources.printDcM.y, resources.printDcM.n, resources.printDcM.domainBegin, resources.printDcM.domainEnd};
        view.printDcY = {resources.printDcY.x, resources.printDcY.y, resources.printDcY.n, resources.printDcY.domainBegin, resources.printDcY.domainEnd};
        view.mainIlluminant = resources.printIllumFiltered;
        view.mainIlluminantHost =
            resources.printIllumFilteredHostValid ? resources.printIllumFilteredHost.data() : nullptr;
        view.preflashIlluminant = resources.printPreflashIllumFiltered;
        view.spectralSampleCount = resources.printIllumK;
        std::copy_n(resources.printPreflashRaw, 3, view.preflashRawCmy);
        view.normalizer = resources.printBalanceNormalizer;
        view.filmDensityTablesHash = resources.printFilmDensityTablesDescriptorHash;
        view.profileTablesHash = resources.printProfileTablesDescriptorHash;
        view.mainIlluminantHash = resources.printMainIlluminantDescriptorHash;
        view.preflashIlluminantHash = resources.printPreflashIlluminantDescriptorHash;
        view.preflashRawHash = resources.printPreflashRawDescriptorHash;
        view.balanceHash = resources.printBalanceDescriptorHash;
        view.preparationHash = resources.printPreparationDescriptorHash;
        view.preflashActive = descriptors.preflashActive;
        view.active =
            view.filmDensityTablesHash == descriptors.filmDensityTables.hash &&
            view.profileTablesHash == descriptors.profileTables.hash &&
            view.mainIlluminantHash == descriptors.mainIlluminant.hash &&
            view.balanceHash == descriptors.balance.hash &&
            view.preparationHash == descriptors.hash &&
            (!view.preflashActive ||
             (view.preflashIlluminantHash == descriptors.preflashIlluminant.hash &&
              view.preflashRawHash == descriptors.preflashRaw.hash));
        return view;
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

    Root::PreparedCudaFrame::SpatialDirScratchView Root::PreparedCudaFrame::spatial_dir_scratch(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        SpatialDirScratchView view{};
        if (!workspace_marker_matches_current_frame(workspace) || !workspace._request.needSpatialDir) {
            return view;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch =
            _state->resources->spatialDirScratch;
        const WorkspaceRequest& effectiveRequest =
            (_state->scratchWorkspace.request.needSpatialDir &&
             _state->scratchWorkspace.request.spatialDirDescriptorHash ==
                 workspace._request.spatialDirDescriptorHash)
                ? _state->scratchWorkspace.request
                : workspace._request;
        const Spektrafilm::DirScratchPlaneRoles& roles =
            effectiveRequest.spatialDirPlaneRoles;
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
        view.descriptorHash = effectiveRequest.spatialDirDescriptorHash;
        view.scratchTier = effectiveRequest.spatialDirScratchTier;
        view.planeRoles = effectiveRequest.spatialDirPlaneRoles;
        view.targetScratchTier = effectiveRequest.spatialDirTargetScratchTier;
        view.targetPlaneRoles = effectiveRequest.spatialDirTargetPlaneRoles;
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
            !workspace._request.needSpatialDir ||
            !_state ||
            _state->spatialDirDescriptor.hash == 0 ||
            _state->spatialDirDescriptor.hash != descriptorHash) {
            return view;
        }
        const auto& kernels = _state->resources->spatialDirKernels;
        view.gaussian = {
            kernels[0].weights,
            kernels[0].radius,
            _state->spatialDirDescriptor.gaussianSigmaPixels};
        for (int slot = 0; slot < 3; ++slot) {
            view.exponential[slot] = {
                kernels[static_cast<std::size_t>(slot) + 1].weights,
                kernels[static_cast<std::size_t>(slot) + 1].radius,
                _state->spatialDirDescriptor.exponentialSigmaPixels[slot]};
        }
        view.descriptorHash = descriptorHash;
        view.active = view.gaussian.sigma >= 3.0f ||
                      (view.gaussian.weights && view.gaussian.radius > 0);
        for (int slot = 0; slot < 3; ++slot) {
            const bool required = _state->spatialDirDescriptor.exponentialWeights[slot] > 0.0f;
            view.active = view.active &&
                          (!required ||
                           view.exponential[slot].sigma >= 3.0f ||
                           (view.exponential[slot].weights && view.exponential[slot].radius > 0));
        }
        return view;
    }

    Root::PreparedCudaFrame::ScannerWorkspaceView Root::PreparedCudaFrame::scanner_workspace(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        ScannerWorkspaceView view{};
        if (!workspace_marker_matches_current_frame(workspace)) {
            return view;
        }
        const WorkspaceRequest& activeRequest = active_workspace_request(workspace);
        if (!activeRequest.needOptics) {
            return view;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->resources->scannerScratch;
        if (activeRequest.aliasScannerRgbFromSpatialDirFiltered) {
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
                workspace._request.spatialDirDescriptorHash);
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


    bool Root::PreparedCudaFrame::record_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active ||
            _state->transaction.committed) {
            outError = "prepared frame is not active for use-event submission";
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


    Root::ShutdownToken::ShutdownToken(Root* root) noexcept
        : _root(root) {
    }

    Root::ShutdownToken::~ShutdownToken() {
        reset();
    }

    Root::ShutdownToken::ShutdownToken(ShutdownToken&& other) noexcept
        : _root(std::exchange(other._root, nullptr)) {
    }

    Root::ShutdownToken& Root::ShutdownToken::operator=(ShutdownToken&& other) noexcept {
        if (this != &other) {
            reset();
            _root = std::exchange(other._root, nullptr);
        }
        return *this;
    }

    void Root::ShutdownToken::reset() noexcept {
        Root* root = _root;
        _root = nullptr;
        if (root) {
            root->finish_shutdown();
        }
    }

    Root& Root::instance() noexcept {
        static Root root;
        return root;
    }

    Root& root() noexcept {
        return Root::instance();
    }

    Root::Root()
        : _dataDir(compute_process_data_dir()), _assets(_dataDir) {
    }

    void Root::ensure_bootstrap() {
        resume_frame_preparation();
        std::call_once(_bootstrapOnce, [this]() {
            load_process_spectral_assets(_dataDir);
        });
    }

    void Root::shutdown() noexcept {
        try {
            ShutdownToken shutdown = begin_shutdown();
            (void)shutdown;
            wait_for_frame_preparation();
            std::string retireError;
            if (!retire_known_contexts(retireError)) {
                set_shutdown_retire_blocked(true);
                if (JTRACE_ENABLED(1)) {
                    std::string msg;
                    msg.reserve(160);
                    msg = "process_shutdown_retire_failed release_host_services=0";
                    if (!retireError.empty()) {
                        msg += " error=";
                        msg += retireError;
                    }
                    JTRACE("MSLCY", msg);
                }
                return;
            }
            set_shutdown_retire_blocked(false);
            release_cuda_context_resource_owners();
            release_process_host_services();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::retire_grain_static_instance(
        std::uint64_t instanceToken) noexcept {
        if (instanceToken == 0) {
            return;
        }
        try {
            std::vector<CudaResourceOwner> retiredOwners;
            {
                std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
                for (auto& item : _cudaContextResources) {
                    CudaContextResourceEntry& entry = item.second;
                    entry.grainInstances.erase(instanceToken);
                    if (!detail::grain_static_has_active_instance(
                            entry.grainInstances) &&
                        entry.grainOwner) {
                        retiredOwners.emplace_back(std::move(entry.grainOwner));
                    }
                }
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    JuicerAssets::Library& Root::assets() noexcept {
        return _assets;
    }

    Root::FramePreparationToken Root::begin_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (!_acceptFramePreparation) {
                return FramePreparationToken{};
            }
            ++_activeFramePreparations;
            return FramePreparationToken(this);
        } catch (...) {
            return FramePreparationToken{};
        }
    }

    bool Root::retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept {
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return retire_cuda_context(key, false, outError);
    }

    bool Root::retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept {
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return retire_cuda_context(key, true, outError);
    }

    bool Root::retire_cuda_context(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        bool contextReset,
        std::string& outError) noexcept {
        try {
            outError.clear();
            if (deviceContextKey.deviceId < 0 ||
                !deviceContextKey.contextOpaque) {
                outError = "invalid CUDA context retirement key";
                return false;
            }
            JuicerCuda::ResourceManager::RegistryContextSnapshot contextSnapshot{};
            if (!JuicerCuda::ResourceManager::registry_begin_owner_retire(
                    deviceContextKey,
                    contextSnapshot)) {
                outError = "context owner-retire rejected";
                return false;
            }
            if (contextSnapshot.contextEpoch == 0) {
                outError = "context owner-retire epoch invalid";
                return false;
            }

            std::vector<CudaResourceOwner> owners;
            std::vector<std::uint64_t> epochs;
            std::shared_ptr<JuicerCuda::DeviceAllocationLedger> deviceLedger;
            {
                std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
                const auto ledger =
                    _cudaDeviceLedgers.find(deviceContextKey.deviceId);
                if (ledger != _cudaDeviceLedgers.end()) {
                    deviceLedger = ledger->second;
                }
                auto collect_epoch = [&](std::uint64_t epoch) {
                    if (epoch != 0 &&
                        std::find(epochs.begin(), epochs.end(), epoch) ==
                            epochs.end()) {
                        epochs.push_back(epoch);
                    }
                };
                collect_epoch(contextSnapshot.contextEpoch);
                for (const auto& [key, entry] : _cudaContextResources) {
                    if (key.deviceContextKey == deviceContextKey) {
                        collect_epoch(key.contextEpoch);
                        if (entry.frameOwner) {
                            owners.push_back(entry.frameOwner);
                        }
                        if (entry.grainOwner) {
                            owners.push_back(entry.grainOwner);
                        }
                    }
                }
            }

            bool physicalDrainSucceeded = true;
            bool provenContextLoss = false;
            for (const CudaResourceOwner& owner : owners) {
                if (!owner ||
                    !JuicerCuda::drain_for_context_retire(*owner, outError)) {
                    physicalDrainSucceeded = false;
                    break;
                }
            }
            if (!physicalDrainSucceeded) {
                if (!contextReset || !deviceLedger) {
                    if (outError.empty()) {
                        outError = "CUDA context owner drain failed";
                    }
                    return false;
                }
                provenContextLoss = true;
                for (std::uint64_t epoch : epochs) {
                    std::uint64_t releasedBytes = 0;
                    if (!deviceLedger->release_context_after_proven_loss(
                            deviceContextKey,
                            epoch,
                            releasedBytes,
                            outError)) {
                        return false;
                    }
                }
                for (const CudaResourceOwner& owner : owners) {
                    if (owner) {
                        JuicerCuda::invalidate_resources_after_proven_context_loss(
                            *owner);
                    }
                }
                outError.clear();
            }
            for (std::uint64_t epoch : epochs) {
                if (deviceLedger &&
                    deviceLedger->record_count_for_context(
                        deviceContextKey,
                        epoch) != 0) {
                    outError =
                        "CUDA context retirement retained ledger records epoch=" +
                        std::to_string(epoch);
                    return false;
                }
            }

            {
                std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
                for (auto it = _cudaContextResources.begin();
                     it != _cudaContextResources.end();) {
                    if (it->first.deviceContextKey == deviceContextKey) {
                        it = _cudaContextResources.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            if (!JuicerCuda::ResourceManager::registry_retire(
                    deviceContextKey)) {
                outError = "context registry retire rejected";
                return false;
            }
            JuicerCuda::purge_pinned_upload_staging_for_context(
                deviceContextKey.deviceId,
                deviceContextKey.contextOpaque,
                provenContextLoss
                    ? JuicerCuda::PinnedUploadPurgeDisposition::ProvenContextLoss
                    : JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire);
            owners.clear();
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "CUDA context retirement failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    void Root::CudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
        JuicerCuda::destroy(resources);
    }

    bool Root::resolve_context_cuda_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError.clear();
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque ||
            !deviceLedger) {
            outError = "invalid CUDA context key";
            return false;
        }
        if (contextEpoch == 0) {
            outError = "invalid CUDA context epoch";
            return false;
        }

        const ContextCudaResourceKey resourceKey{deviceContextKey, contextEpoch};
        CudaResourceOwner candidateOwner;
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaContextResourceEntry& entry = _cudaContextResources[resourceKey];
            CudaResourceOwner& resourceOwner = entry.frameOwner;
            if (!resourceOwner) {
                candidateOwner = CudaResourceOwner(
                    JuicerCuda::create(
                        deviceContextKey,
                        contextEpoch,
                        JuicerCuda::Resources::kAllocationOwnershipSchemaVersion,
                        deviceLedger,
                        outError),
                    CudaResourcesDeleter{});
                if (!candidateOwner) {
                    JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                    if (!entry.grainOwner && entry.grainInstances.empty()) {
                        _cudaContextResources.erase(resourceKey);
                    }
                    if (outError.empty()) {
                        outError = "failed to allocate CUDA resources";
                    }
                    return false;
                }
                if (!(candidateOwner->ownerContextKey == deviceContextKey) ||
                    candidateOwner->contextEpoch != contextEpoch ||
                    candidateOwner->deviceLedger != deviceLedger) {
                    if (!entry.grainOwner && entry.grainInstances.empty()) {
                        _cudaContextResources.erase(resourceKey);
                    }
                    outError = "CUDA resources resolved for a different context";
                    return false;
                }
                resourceOwner = std::move(candidateOwner);
            }

            outResourceOwner = resourceOwner;
        }

        outResources = outResourceOwner.get();
        if (!outResources) {
            outError = "CUDA resources missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::resolve_cuda_device_ledger(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
        std::string& outError) {
        outDeviceLedger.reset();
        outError.clear();
        if (deviceContextKey.deviceId < 0) {
            outError =
                "UnsupportedCudaExecutionProfile component=device_ledger field=device_id";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            const auto ledgerIt = _cudaDeviceLedgers.find(deviceContextKey.deviceId);
            if (ledgerIt != _cudaDeviceLedgers.end() && ledgerIt->second) {
                outDeviceLedger = ledgerIt->second;
                return true;
            }
        }

        std::uint64_t deviceBudgetBytes = 0;
        if (!query_cuda_device_total_bytes(
                deviceContextKey.deviceId,
                deviceBudgetBytes,
                outError)) {
            return false;
        }
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> candidate =
            JuicerCuda::DeviceAllocationLedger::create(
                JuicerCuda::DeviceLedgerBudget{
                    .deviceId = deviceContextKey.deviceId,
                    .bytes = deviceBudgetBytes},
                outError);
        if (!candidate) {
            if (outError.empty()) {
                outError =
                    "UnsupportedCudaExecutionProfile component=device_ledger field=create";
            }
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            auto [ledgerIt, inserted] =
                _cudaDeviceLedgers.try_emplace(deviceContextKey.deviceId, candidate);
            if (!inserted && !ledgerIt->second) {
                ledgerIt->second = std::move(candidate);
            }
            outDeviceLedger = ledgerIt->second;
        }
        return outDeviceLedger != nullptr;
    }

    bool Root::resolve_cuda_frame_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const JuicerCuda::ResourceManager::ResolvedMemoryBudget& memoryBudget,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError.clear();
        if (!deviceLedger ||
            memoryBudget.deviceId != deviceContextKey.deviceId ||
            memoryBudget.deviceBudgetBytes == 0 ||
            memoryBudget.allocationCapBytes == 0 ||
            deviceLedger->device_budget_bytes() != memoryBudget.deviceBudgetBytes) {
            outError =
                "UnsupportedCudaExecutionProfile component=device_ledger field=memory_budget";
            return false;
        }

        if (!deviceLedger->bind_or_validate_cap(
                memoryBudget.allocationCapBytes,
                outError)) {
            const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                deviceLedger->snapshot();
            outError =
                "UnsupportedCudaExecutionProfile component=device_ledger field=cap" +
                std::string(" device=") + std::to_string(deviceContextKey.deviceId) +
                " requested_cap=" +
                std::to_string(memoryBudget.allocationCapBytes) +
                " current_cap=" + std::to_string(ledgerSnapshot.capBytes) +
                " reserved=" + std::to_string(ledgerSnapshot.reservedBytes) +
                " committed=" + std::to_string(ledgerSnapshot.committedBytes) +
                " retiring=" + std::to_string(ledgerSnapshot.retiringBytes);
            return false;
        }
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            deviceLedger,
            outResourceOwner,
            outResources,
            outError);
    }

    bool Root::apply_grain_static_membership_and_copy_owner(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        std::uint64_t instanceToken,
        std::uint64_t snapshotId,
        bool active,
        detail::GrainStaticMembershipChange& outChange,
        CudaResourceOwner& outOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        outChange = detail::GrainStaticMembershipChange{};
        outOwner.reset();
        outResources = nullptr;
        outError.clear();
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque ||
            contextEpoch == 0 || instanceToken == 0 || snapshotId == 0 ||
            !deviceLedger) {
            outError =
                "ResourceDescriptorMismatch phase=grain_static field=membership_identity";
            return false;
        }

        const ContextCudaResourceKey contextKey{
            deviceContextKey,
            contextEpoch};
        CudaResourceOwner candidateOwner;
        CudaResourceOwner retiredOwner;
#if JUICER_DIAGNOSTICS_COMPILED
        bool ownerCopied = false;
        bool rootOwnerRetired = false;
#endif
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaContextResourceEntry& entry = _cudaContextResources[contextKey];
            detail::GrainStaticMembershipChange change;
            const detail::GrainStaticMembershipResult result =
                detail::apply_grain_static_membership(
                    entry.grainInstances,
                    instanceToken,
                    snapshotId,
                    active,
                    change);
            switch (result) {
                case detail::GrainStaticMembershipResult::Applied:
                case detail::GrainStaticMembershipResult::Idempotent:
                    break;
                case detail::GrainStaticMembershipResult::StaleSnapshot:
                    outError =
                        "ResourceDescriptorMismatch phase=grain_static field=stale_snapshot";
                    return false;
                case detail::GrainStaticMembershipResult::ConflictingEqualSnapshot:
                    outError =
                        "ResourceDescriptorMismatch phase=grain_static field=equal_snapshot_conflict";
                    return false;
                case detail::GrainStaticMembershipResult::InvalidInput:
                default:
                    outError =
                        "ResourceDescriptorMismatch phase=grain_static field=membership_update";
                    return false;
            }

            if (active) {
                if (!entry.grainOwner) {
                    candidateOwner = CudaResourceOwner(
                        JuicerCuda::create(
                            deviceContextKey,
                            contextEpoch,
                            JuicerCuda::Resources::kAllocationOwnershipSchemaVersion,
                            deviceLedger,
                            outError),
                        CudaResourcesDeleter{});
                    if (!candidateOwner) {
                        if (change.applied) {
                            (void)detail::rollback_grain_static_membership(
                                entry.grainInstances,
                                change);
                        }
                        if (!entry.frameOwner && entry.grainInstances.empty()) {
                            _cudaContextResources.erase(contextKey);
                        }
                        outError = "failed to allocate grain-static CUDA resources";
                        return false;
                    }
                    if (!(candidateOwner->ownerContextKey == deviceContextKey) ||
                        candidateOwner->contextEpoch != contextEpoch ||
                        candidateOwner->deviceLedger != deviceLedger) {
                        if (change.applied) {
                            (void)detail::rollback_grain_static_membership(
                                entry.grainInstances,
                                change);
                        }
                        if (!entry.frameOwner && entry.grainInstances.empty()) {
                            _cudaContextResources.erase(contextKey);
                        }
                        outError =
                            "grain-static CUDA resources resolved for a different context";
                        return false;
                    }
                    entry.grainOwner = std::move(candidateOwner);
                }
                if (!(entry.grainOwner->ownerContextKey == deviceContextKey) ||
                    entry.grainOwner->contextEpoch != contextEpoch ||
                    entry.grainOwner->deviceLedger != deviceLedger) {
                    if (change.applied) {
                        (void)detail::rollback_grain_static_membership(
                            entry.grainInstances,
                            change);
                    }
                    outError =
                        "grain-static CUDA resources resolved for a different context";
                    return false;
                }
                outOwner = entry.grainOwner;
#if JUICER_DIAGNOSTICS_COMPILED
                ownerCopied = static_cast<bool>(outOwner);
#endif
            } else if (!detail::grain_static_has_active_instance(
                           entry.grainInstances)) {
                retiredOwner = std::move(entry.grainOwner);
#if JUICER_DIAGNOSTICS_COMPILED
                rootOwnerRetired = static_cast<bool>(retiredOwner);
#endif
            }
            outChange = change;
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            std::string message =
                "event=grain_static_membership active=";
            message += active ? "1" : "0";
            message += " owner_copied_before_unlock=";
            message += ownerCopied ? "1" : "0";
            message += " root_owner_retired_after_unlock=";
            message += rootOwnerRetired ? "1" : "0";
            message += " snapshot_id=";
            message += std::to_string(
                static_cast<unsigned long long>(snapshotId));
            JTRACE("GRAINLIFE", message);
        }
#endif

        outResources = outOwner.get();
        if (active && !outResources) {
            outError =
                "grain-static CUDA resources missing after owner copy";
            return false;
        }
        return true;
    }

    void Root::rollback_grain_static_membership(
        const ContextCudaResourceKey& contextKey,
        const detail::GrainStaticMembershipChange& change) noexcept {
        if (!change.applied) {
            return;
        }
        try {
            CudaResourceOwner retiredOwner;
#if JUICER_DIAGNOSTICS_COMPILED
            bool rolledBack = false;
#endif
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            const auto entryIt = _cudaContextResources.find(contextKey);
            if (entryIt == _cudaContextResources.end() ||
                !detail::rollback_grain_static_membership(
                    entryIt->second.grainInstances,
                    change)) {
                return;
            }
#if JUICER_DIAGNOSTICS_COMPILED
            rolledBack = true;
#endif
            if (!detail::grain_static_has_active_instance(
                    entryIt->second.grainInstances)) {
                retiredOwner = std::move(entryIt->second.grainOwner);
            }
            if (!entryIt->second.frameOwner &&
                !entryIt->second.grainOwner &&
                entryIt->second.grainInstances.empty()) {
                _cudaContextResources.erase(entryIt);
            }
#if JUICER_DIAGNOSTICS_COMPILED
            if (rolledBack && JTRACE_ENABLED(1)) {
                JTRACE(
                    "GRAINLIFE",
                    "event=grain_static_conditional_rollback applied=1");
            }
#endif
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }
    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const CudaFramePreparationRequest& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>();
        } catch (...) {
            outError = "failed to allocate CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->remember_stream(cudaStreamOpaque);
        frame._state->set_failure(
            PreparedCudaFailureStage{"prepare_cuda_frame"},
            "CUDA prepared frame failed");
        if (!request.recipe) {
            outError =
                "MissingRequiredResource component=cuda_frame_preparation field=recipe";
            return frame;
        }
        const Spektrafilm::ScanRoute route =
            request.recipe->profileRoute.scanRoute;
        const Spektrafilm::ScanRouteMetadata& routeMetadata =
            Spektrafilm::scan_route_metadata(route);
        if (routeMetadata.route != route) {
            outError =
                "ResourceDescriptorMismatch component=cuda_frame_preparation field=scan_route";
            return frame;
        }
        const bool printRoute = routeMetadata.printRoute;
        const auto appendRoute = [&] {
            if (!outError.empty()) {
                outError += ' ';
            }
            outError += "route=";
            outError += routeMetadata.key;
        };
        const auto recordFailure = [&](const char* stageTag, const char* prefix) {
            frame._state->set_failure(
                PreparedCudaFailureStage{stageTag},
                prefix);
            appendRoute();
        };
        if (!request.exposureTables || !request.spdSInv ||
            !request.filmRawConfig || !request.scannerTables || !request.scannerColor ||
            !request.scannerLutDescriptor) {
            outError =
                "MissingRequiredResource component=cuda_frame_preparation field=serving_inputs";
            recordFailure(
                "validate_cuda_frame_preparation_request",
                "CUDA frame preparation request validation failed");
            return frame;
        }
        if (!validate_visual_grain_descriptor(
                request.recipe,
                request.visualGrainDescriptor,
                outError)) {
            recordFailure(
                "validate_visual_grain_descriptor",
                "CUDA visual grain descriptor validation failed");
            return frame;
        }
        if (!validate_film_juicer_effects_descriptor(
                request.recipe,
                request.effectsDescriptor,
                outError)) {
            recordFailure(
                "validate_film_juicer_effects_descriptor",
                "CUDA effects descriptor validation failed");
            return frame;
        }
        frame._state->visualGrainDescriptor =
            request.visualGrainDescriptor;
        frame._state->effectsDescriptor = request.effectsDescriptor;
        if (printRoute) {
            if (!JuicerCuda::build_print_resource_descriptors(
                    *request.recipe,
                    frame._state->printDescriptors,
                    outError)) {
                recordFailure(
                    "build_print_resource_descriptors",
                    "CUDA print resource descriptor construction failed");
                return frame;
            }
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
        if (!derive_workspace_request(
                *request.recipe,
                request.spatialDirDescriptor,
                request.scannerPostEffects,
                request.diffusionFrameSetDescriptor,
                request.scatterHalationDescriptor,
                request.visualGrainDescriptor,
                request.effectsDescriptor,
                request.recipe->profileRoute.capturePolarity,
                request.requestedWidth,
                request.requestedHeight,
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->workspaceRequest,
                outError)) {
            recordFailure(
                "derive_cuda_frame_workspace_request",
                "CUDA frame workspace derivation failed");
            frame.abort();
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
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
        if (!frame._state->prepare_diffusion_resources(
                deviceContextKey,
                *request.recipe,
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
                *request.recipe,
                request.scatterHalationDescriptor,
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
        JuicerCuda::FocusedRouteResourcePreparation focusedPreparation{};
        focusedPreparation.recipe = request.recipe;
        focusedPreparation.exposureTables = request.exposureTables;
        focusedPreparation.spdSInv = request.spdSInv;
        focusedPreparation.filmRawConfig = request.filmRawConfig;
        focusedPreparation.scannerTables = request.scannerTables;
        focusedPreparation.scannerColor = request.scannerColor;
        focusedPreparation.scannerLutDescriptor = request.scannerLutDescriptor;
        if (!JuicerCuda::prepare_focused_route_resources(
                *frame._state->resources,
                focusedPreparation,
                cudaStreamOpaque,
                outError)) {
            recordFailure(
                "prepare_focused_route_resources",
                "CUDA focused route resource preparation failed");
            frame.abort();
            return frame;
        }
        if (printRoute) {
            JuicerCuda::PrintResourcePreparation preparation{};
            preparation.recipe = request.recipe;
            preparation.assets = &_assets;
            if (!JuicerCuda::prepare_print_resources(
                    *frame._state->resources,
                    preparation,
                    cudaStreamOpaque,
                    outError)) {
                recordFailure(
                    "prepare_print_resources",
                    "CUDA print resource preparation failed");
                frame.abort();
                return frame;
            }
            frame._state->printRecipe = &request.recipe->print;
            frame._state->focusedScanMedium =
                &frame._state->resources->scanPrint;
            frame._state->focusedScanLut =
                &frame._state->resources->scanPrintLut;
        } else {
            frame._state->focusedScanMedium =
                &frame._state->resources->scanNegative;
            frame._state->focusedScanLut =
                &frame._state->resources->scanNegativeLut;
        }
        frame._state->capturePolarity =
            request.recipe->profileRoute.capturePolarity;
        frame._state->focusedFilmRawConfig = request.filmRawConfig;
        frame._state->focusedScannerColor = request.scannerColor;
        if (request.scannerPostEffects) {
            if (!frame.prepare_scanner_post_effects(
                    *request.scannerPostEffects,
                    outError)) {
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
                deviceContextKey,
                cudaStreamOpaque,
                outError)) {
            appendRoute();
            frame.abort();
            return frame;
        }
        return frame;
    }

    bool Root::begin_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
        std::string& outError) {
        if (!resolve_cuda_device_ledger(
                snapshot.deviceContextKey,
                outDeviceLedger,
                outError)) {
            return false;
        }
        return JuicerCuda::ResourceManager::begin_submission(
            transaction,
            snapshot,
            outDeviceLedger->device_budget_bytes(),
            outError);
    }


    bool Root::retire_known_contexts(std::string& outError) noexcept {
        outError.clear();
        try {
            std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
            JuicerCuda::ResourceManager::registry_snapshot_context_keys(keys);
            {
                std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
                for (const auto& [resourceKey, entry] : _cudaContextResources) {
                    (void)entry;
                    if (std::find(
                            keys.begin(),
                            keys.end(),
                            resourceKey.deviceContextKey) == keys.end()) {
                        keys.push_back(resourceKey.deviceContextKey);
                    }
                }
            }
            for (const auto& key : keys) {
                if (!retire_cuda_context(key, false, outError)) {
                    return false;
                }
            }
            keys.clear();
            JuicerCuda::ResourceManager::registry_snapshot_context_keys(keys);
            if (!keys.empty()) {
                outError = "context retire incomplete; live_contexts=" +
                           std::to_string(keys.size());
                return false;
            }
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            for (const auto& [deviceId, ledger] : _cudaDeviceLedgers) {
                if (!ledger) {
                    continue;
                }
                const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                    ledger->snapshot();
                if (ledgerSnapshot.chargedBytes != 0 ||
                    ledgerSnapshot.recordCount != 0) {
                    outError =
                        "shutdown retained CUDA ledger records device=" +
                        std::to_string(deviceId) +
                        " charged=" +
                        std::to_string(ledgerSnapshot.chargedBytes) +
                        " records=" +
                        std::to_string(ledgerSnapshot.recordCount);
                    return false;
                }
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "registry-wide context retire threw";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    void Root::release_cuda_context_resource_owners() noexcept {
        try {
            CudaContextResourceMap contextResources;
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            contextResources.swap(_cudaContextResources);
            _cudaDeviceLedgers.clear();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::release_process_host_services() noexcept {
        _assets.release_cached_payloads();
    }

    Root::ShutdownToken Root::begin_shutdown() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _acceptFramePreparation = false;
            ++_activeShutdowns;
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
            return ShutdownToken(this);
        } catch (...) {
            return ShutdownToken{};
        }
    }

    void Root::finish_shutdown() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeShutdowns > 0) {
                --_activeShutdowns;
            }
            _framePreparationCv.notify_all();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::finish_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeFramePreparations > 0) {
                --_activeFramePreparations;
            }
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::resume_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeShutdowns == 0 && !_shutdownRetireBlocked) {
                _acceptFramePreparation = true;
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::wait_for_frame_preparation() noexcept {
        try {
            std::unique_lock<std::mutex> lock(_framePreparationMutex);
            _framePreparationCv.wait(lock, [this]() {
                return _activeFramePreparations == 0;
            });
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::set_shutdown_retire_blocked(bool blocked) noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _shutdownRetireBlocked = blocked;
            if (blocked) {
                _acceptFramePreparation = false;
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerProcess
