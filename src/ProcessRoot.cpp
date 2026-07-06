#include "ProcessRoot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace JuicerProcess {

    namespace {

        bool spatial_dir_roles_match_tier(
            Spektrafilm::DirScratchTier tier,
            const Spektrafilm::DirScratchPlaneRoles& roles) noexcept {
            switch (tier) {
                case Spektrafilm::DirScratchTier::Tier0:
                    return roles.total_float_planes() == 0;
                case Spektrafilm::DirScratchTier::Tier1F:
                    return roles.filteredCorrectionPlanes == 3 &&
                           ((roles.rawCorrectionPlanes == 1 &&
                             roles.filterTempPlanes == 0 &&
                             roles.iirForwardTempPlanes == 0) ||
                            (roles.rawCorrectionPlanes == 3 &&
                             roles.filterTempPlanes == 1 &&
                             roles.iirForwardTempPlanes == 0)) &&
                           roles.cachedLogRawPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_corrPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_mixPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_tmpPlanes == 0;
                case Spektrafilm::DirScratchTier::Tier1IChannels:
                    return roles.filteredCorrectionPlanes == 3 &&
                           ((roles.rawCorrectionPlanes == 1 &&
                             roles.filterTempPlanes == 1 &&
                             roles.iirForwardTempPlanes == 0) ||
                            (roles.rawCorrectionPlanes == 3 &&
                             ((roles.filterTempPlanes == 1 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 1 &&
                               roles.iirForwardTempPlanes == 1) ||
                              (roles.filterTempPlanes == 2 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 3 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 3 &&
                               roles.iirForwardTempPlanes == 3)))) &&
                           roles.cachedLogRawPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_corrPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_mixPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_tmpPlanes == 0;
                case Spektrafilm::DirScratchTier::Tier2:
                    return roles.filteredCorrectionPlanes == 3 &&
                           ((roles.rawCorrectionPlanes == 1 &&
                             (roles.filterTempPlanes == 0 ||
                              roles.filterTempPlanes == 1) &&
                             roles.iirForwardTempPlanes == 0) ||
                            (roles.rawCorrectionPlanes == 3 &&
                             ((roles.filterTempPlanes == 1 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 1 &&
                               roles.iirForwardTempPlanes == 1) ||
                              (roles.filterTempPlanes == 2 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 3 &&
                               roles.iirForwardTempPlanes == 0) ||
                              (roles.filterTempPlanes == 3 &&
                               roles.iirForwardTempPlanes == 3)))) &&
                           roles.cachedLogRawPlanes == 3 &&
                           roles.SF_TEMP_BRIDGE_corrPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_mixPlanes == 0 &&
                           roles.SF_TEMP_BRIDGE_tmpPlanes == 0;
                case Spektrafilm::DirScratchTier::Tier3:
                case Spektrafilm::DirScratchTier::Unsupported:
                case Spektrafilm::DirScratchTier::SF_TEMP_BRIDGE_LegacySpatialDirScratch:
                    return false;
            }
            return false;
        }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
            const bool hasStreamedRaw =
                scratch.rawCorrectionY && scratch.rawCorrectionM == nullptr && scratch.rawCorrectionC == nullptr;
            const bool rawCorrectionMatch =
                (roles.rawCorrectionPlanes == 3 && hasThreeRaw) ||
                (roles.rawCorrectionPlanes == 1 && hasStreamedRaw);
            const bool filterTempMatch =
                roles.filterTempPlanes == 0 || scratch.filterTemp;
            const bool tier1Base =
                rawCorrectionMatch &&
                scratch.filteredCorrectionY && scratch.filteredCorrectionM && scratch.filteredCorrectionC &&
                filterTempMatch;
            if (!tier1Base) {
                return false;
            }
            const bool hasChannelTemps =
                scratch.filterTempM && scratch.filterTempC &&
                scratch.iirForwardTemp && scratch.iirForwardTempM && scratch.iirForwardTempC;
            const bool hasAliasedForwardTemps =
                scratch.filterTempM && scratch.filterTempC &&
                scratch.iirForwardTemp == nullptr && scratch.iirForwardTempM == nullptr &&
                scratch.iirForwardTempC == nullptr;
            const bool hasCompactSequentialTemps =
                scratch.iirForwardTemp &&
                scratch.filterTempM == nullptr && scratch.filterTempC == nullptr &&
                scratch.iirForwardTempM == nullptr && scratch.iirForwardTempC == nullptr;
            const bool hasLowScratchPairTemps =
                scratch.filterTempM && scratch.filterTempC == nullptr &&
                scratch.iirForwardTemp == nullptr && scratch.iirForwardTempM == nullptr &&
                scratch.iirForwardTempC == nullptr;
            const bool hasNoChannelTemps =
                scratch.filterTempM == nullptr && scratch.filterTempC == nullptr &&
                scratch.iirForwardTemp == nullptr && scratch.iirForwardTempM == nullptr &&
                scratch.iirForwardTempC == nullptr;
            const bool channelTempsMatch =
                (roles.filterTempPlanes == 0 && roles.iirForwardTempPlanes == 0 && hasNoChannelTemps) ||
                (roles.filterTempPlanes == 1 && roles.iirForwardTempPlanes == 0 && hasNoChannelTemps) ||
                (roles.filterTempPlanes == 1 && roles.iirForwardTempPlanes == 1 && hasCompactSequentialTemps) ||
                (roles.filterTempPlanes == 2 && roles.iirForwardTempPlanes == 0 && hasLowScratchPairTemps) ||
                (roles.filterTempPlanes == 3 && roles.iirForwardTempPlanes == 0 && hasAliasedForwardTemps) ||
                (roles.filterTempPlanes == 3 && roles.iirForwardTempPlanes == 3 && hasChannelTemps);
            const bool hasCachedLogRaw = scratch.logRawB && scratch.logRawG && scratch.logRawR;
            const bool hasNoCachedLogRaw =
                scratch.logRawB == nullptr && scratch.logRawG == nullptr && scratch.logRawR == nullptr;
            const bool cachedLogRawMatch =
                (roles.cachedLogRawPlanes == 0 && hasNoCachedLogRaw) ||
                (roles.cachedLogRawPlanes == 3 && hasCachedLogRaw);
            return channelTempsMatch && cachedLogRawMatch;
        }

        bool spatial_dir_scratch_has_final_develop_roles(
            const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch,
            const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
            if (!scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
                !scratch.filteredCorrectionC) {
                return false;
            }
            const bool hasCachedLogRaw = scratch.logRawB && scratch.logRawG && scratch.logRawR;
            const bool hasNoCachedLogRaw =
                scratch.logRawB == nullptr && scratch.logRawG == nullptr && scratch.logRawR == nullptr;
            const bool cachedLogRawMatch =
                (targetRoles.cachedLogRawPlanes == 0 && hasNoCachedLogRaw) ||
                (targetRoles.cachedLogRawPlanes == 3 && hasCachedLogRaw);
            if (!cachedLogRawMatch) {
                return false;
            }
            const bool filterTempsCompatible =
                targetRoles.filterTempPlanes == 3 ||
                (targetRoles.filterTempPlanes == 2 && !scratch.filterTempC) ||
                (!scratch.filterTempM && !scratch.filterTempC);
            const bool iirForwardTempsCompatible =
                targetRoles.iirForwardTempPlanes == 3 ||
                (!scratch.iirForwardTemp && !scratch.iirForwardTempM && !scratch.iirForwardTempC);
            return filterTempsCompatible && iirForwardTempsCompatible;
        }

        bool dir_plane_roles_equal(
            const Spektrafilm::DirScratchPlaneRoles& a,
            const Spektrafilm::DirScratchPlaneRoles& b) noexcept {
            return a.rawCorrectionPlanes == b.rawCorrectionPlanes &&
                   a.filteredCorrectionPlanes == b.filteredCorrectionPlanes &&
                   a.filterTempPlanes == b.filterTempPlanes &&
                   a.iirForwardTempPlanes == b.iirForwardTempPlanes &&
                   a.cachedLogRawPlanes == b.cachedLogRawPlanes &&
                   a.SF_TEMP_BRIDGE_corrPlanes == b.SF_TEMP_BRIDGE_corrPlanes &&
                   a.SF_TEMP_BRIDGE_mixPlanes == b.SF_TEMP_BRIDGE_mixPlanes &&
                   a.SF_TEMP_BRIDGE_tmpPlanes == b.SF_TEMP_BRIDGE_tmpPlanes;
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
            roles.iirForwardTempPlanes = 0;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_single_temp_sequential_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 3;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 1;
            roles.iirForwardTempPlanes = 0;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Spektrafilm::DirScratchPlaneRoles strict_yvv_component_streamed_plane_roles() noexcept {
            Spektrafilm::DirScratchPlaneRoles roles{};
            roles.rawCorrectionPlanes = 1;
            roles.filteredCorrectionPlanes = 3;
            roles.filterTempPlanes = 1;
            roles.iirForwardTempPlanes = 0;
            roles.cachedLogRawPlanes = 0;
            return roles;
        }

        Root::PreparedCudaFrame::WorkspaceRequest strict_yvv_workspace_candidate(
            const Root::PreparedCudaFrame::WorkspaceRequest& base,
            const Spektrafilm::DirScratchPlaneRoles& buildRoles,
            Spektrafilm::DirScratchTier targetTier,
            const Spektrafilm::DirScratchPlaneRoles& targetRoles) noexcept {
            Root::PreparedCudaFrame::WorkspaceRequest candidate = base;
            candidate.spatialDirScratchTier = Spektrafilm::DirScratchTier::Tier1IChannels;
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
                   request.spatialDirTargetPlaneRoles.cachedLogRawPlanes == 0 &&
                   dir_plane_roles_equal(
                       request.spatialDirTargetPlaneRoles,
                       request.spatialDirPlaneRoles);
        }

        [[maybe_unused]] const char* strict_yvv_candidate_label(
            const Spektrafilm::DirScratchPlaneRoles& roles) noexcept {
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 3 &&
                roles.iirForwardTempPlanes == 0) {
                return "strict_yvv_channels_aliased_forward";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 3 &&
                roles.iirForwardTempPlanes == 3) {
                return "strict_yvv_channels";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 2 &&
                roles.iirForwardTempPlanes == 0) {
                return "strict_yvv_low_scratch_pair";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 1 &&
                roles.iirForwardTempPlanes == 0) {
                return "strict_yvv_single_temp_sequential";
            }
            if (roles.rawCorrectionPlanes == 1 &&
                roles.filterTempPlanes == 0 &&
                roles.iirForwardTempPlanes == 0) {
                return "accepted_fft_component_streamed";
            }
            if (roles.rawCorrectionPlanes == 1 &&
                roles.filterTempPlanes == 1 &&
                roles.iirForwardTempPlanes == 0) {
                return "strict_yvv_component_streamed";
            }
            if (roles.rawCorrectionPlanes == 3 &&
                roles.filterTempPlanes == 1 &&
                roles.iirForwardTempPlanes == 1) {
                return "strict_yvv_compact_sequential";
            }
            return "unknown";
        }

        bool workspace_request_needs_shared_tmp_plane(
            const Root::PreparedCudaFrame::WorkspaceRequest& request) noexcept {
            return request.needOptics ||
                   (request.needSpatialDir &&
                    request.spatialDirPlaneRoles.filterTempPlanes > 0);
        }
#endif

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

        void load_spectral_globals(const std::string& dataDir) {
            Spectral::SpectralMutationScope mutationScope(
                Spectral::SpectralMutationStage::Bootstrap,
                "process_bootstrap");
            (void)mutationScope;

            try {
                Spectral::lock_shape_to_reference_axis();
                const auto cmf = Spectral::load_csv_triplets(data_file_string(dataDir, "cie1931_2deg.csv"));
                if (!Spectral::cmf_triplets_match_reference_axis(cmf)) {
                    JTRACE("INIT", "FATAL: CMF wavelengths do not match 380-780@5nm grid");
                    throw std::runtime_error("CMF grid mismatch");
                }
                Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
                Spectral::ensure_precomputed_up_to_date();
                Spectral::disable_hanatos_if_reference_mismatch();
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

            std::vector<std::pair<float, float>> kg3Pairs;
            try {
                kg3Pairs = Spectral::load_csv_pairs(data_file_string(
                    dataDir,
                    "filters",
                    "heat_absorbing",
                    "schott",
                    "KG3.csv"));
            } catch (...) {
                kg3Pairs.clear();
            }
            if (kg3Pairs.empty()) {
                kg3Pairs = {
                    {Spectral::gShape.lambdaMin, 1.0f},
                    {Spectral::gShape.lambdaMax, 1.0f}};
            }
            Spectral::set_filter_KG3_from_pairs(kg3Pairs);
        }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
            request.attachments.needGrainTriplet = workspace.needGrainTriplet;
            request.attachments.needGrainShared = workspace.needGrainShared;
            request.attachments.needGateMask = workspace.needGateMask;
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
            append_i32("iir_forward_temp_planes", roles.iirForwardTempPlanes);
            append_i32("cached_log_raw_planes", roles.cachedLogRawPlanes);
            append_cstr("spatial_dir_target_scratch_tier", Spektrafilm::to_cstr(request.spatialDirTargetScratchTier));
            append_i32("target_raw_correction_planes", targetRoles.rawCorrectionPlanes);
            append_i32("target_filtered_correction_planes", targetRoles.filteredCorrectionPlanes);
            append_i32("target_filter_temp_planes", targetRoles.filterTempPlanes);
            append_i32("target_iir_forward_temp_planes", targetRoles.iirForwardTempPlanes);
            append_i32("target_cached_log_raw_planes", targetRoles.cachedLogRawPlanes);
            append_i32("requested_width", request.requestedWidth);
            append_i32("requested_height", request.requestedHeight);
            append_bool("need_blurred", request.needBlurred);
            append_bool(
                "alias_scanner_rgb_from_spatial_dir_filtered",
                request.aliasScannerRgbFromSpatialDirFiltered);
            append_bool("need_aux", request.needAux);
            append_bool("need_grain_triplet", request.needGrainTriplet);
            append_bool("need_grain_shared", request.needGrainShared);
            append_bool("need_gate_mask", request.needGateMask);
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
            msg += " trace_schema=";
            msg += std::to_string(transaction.snapshot.traceSchemaVersion);
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
                    msg += " trace_schema=";
                    msg += std::to_string(transaction->snapshot.traceSchemaVersion);
                } else {
                    msg += " transaction_id=0 snapshot_id=0 trace_schema=0";
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
                append_scratch_bool("gate_mask", scratch ? scratch->gateMask : nullptr);
                msg += " scratch_gate_mask_width=";
                msg += std::to_string(scratch ? scratch->gateMaskWidth : 0);
                msg += " scratch_gate_mask_height=";
                msg += std::to_string(scratch ? scratch->gateMaskHeight : 0);

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
                msg += " trace_schema=";
                msg += std::to_string(transaction.snapshot.traceSchemaVersion);
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
            JuicerCuda::Resources::DeviceOpticsScratch optics{};
            JuicerCuda::Resources::DeviceSpatialDirScratch spatialDir{};
            float* sharedTmpPlane = nullptr;
            int sharedTmpWidth = 0;
            int sharedTmpHeight = 0;
            std::size_t sharedTmpCapacityElements = 0;
            bool postFrameRequestActive = false;
            bool retainedLeaseActive = false;
            bool overflowActive = false;
        };

        Root* root = nullptr;
        Root::CudaResourceOwner resourceOwner;
        Root::CudaResourceOwner grainStaticOwner;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Resources* grainStaticResources = nullptr;
        const Spectral::FilmRawConfig* focusedFilmRawConfig = nullptr;
        const Scanner::ColorRuntime* focusedScannerColor = nullptr;
        const PrintRecipe* printRecipe = nullptr;
        JuicerCuda::PrintResourceDescriptors printDescriptors{};
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        ScanErrorFrameStage scanErrorStage{};
        AutoExposureFrameWorkspace autoExposureWorkspace{};
        FrameScratchWorkspace scratchWorkspace{};
        Spektrafilm::SpatialDirDescriptor spatialDirDescriptor{};
        Scanner::ScannerPostEffectsDescriptor scannerPostEffectsDescriptor{};
        void* lastCudaStreamOpaque = nullptr;
        bool frameUseEventSubmitted = false;
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";
        bool failureMarksContextLoss = true;

        void set_failure(const char* stageTag, const char* prefix, bool marksContextLoss = true) noexcept {
            failureStageTag = stageTag;
            failurePrefix = prefix;
            failureMarksContextLoss = marksContextLoss;
        }

        void remember_stream(void* cudaStreamOpaque) noexcept {
            if (cudaStreamOpaque) {
                lastCudaStreamOpaque = cudaStreamOpaque;
            }
        }

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
        bool release_overflow_spatial_dir_build_scratch_after_build(
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirBuildScratchReleaseStats& outStats,
            std::string& outError);
        bool stage_overflow_spatial_dir_cached_log_raw_for_final_develop(
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirCachedLogRawStageStats& outStats,
            std::string& outError);
        bool release_overflow_spatial_dir_cached_log_raw_after_final_develop(
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirCachedLogRawReleaseStats& outStats,
            std::string& outError);
        bool release_overflow_spatial_dir_stage_after_scan_linear(
            void* cudaStreamOpaque,
            JuicerCuda::SpatialDirStageReleaseStats& outStats,
            std::string& outError);
        bool release_scratch_workspace_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool submit_frame_use_event(
            void* cudaStreamOpaque,
            std::string& outError);
        void free_scan_error_stage_now() noexcept;
        void free_auto_exposure_workspace_now() noexcept;
        void free_scratch_workspace_now() noexcept;
    };

    bool Root::PreparedCudaFrame::State::allocate_scan_error_stage(std::string& outError) {
        outError.clear();
        free_scan_error_stage_now();

        ScanErrorFrameStage next{};
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&next.deviceFlag), sizeof(int));
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(frame scan error flag) failed: ") +
                       (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            next.deviceFlag = nullptr;
            return false;
        }

        err = cudaMallocHost(reinterpret_cast<void**>(&next.hostFlag), sizeof(int));
        if (err != cudaSuccess) {
            next.hostFlag = nullptr;
        }

        if (next.hostFlag) {
            cudaEvent_t ev = nullptr;
            err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err == cudaSuccess && ev) {
                next.eventOpaque = reinterpret_cast<void*>(ev);
            }
        }

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
            if (!JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    flag,
                    sizeof(int),
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
            cudaFree(stage.deviceFlag);
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
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(") + label + ") failed: " +
                           (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                ptr = nullptr;
                return false;
            }
            return true;
        };

        AutoExposureFrameWorkspace next{};
        if (!alloc_device(next.deviceState.exposureScale, sizeof(float), "frame auto-exposure scale") ||
            !alloc_device(next.deviceState.autoEV, sizeof(double), "frame auto-exposure autoEV") ||
            !alloc_device(next.deviceState.valid, sizeof(int), "frame auto-exposure valid")) {
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
            auto retire_ptr = [&](auto*& ptr, std::size_t bytes, const char* label) {
                if (!ptr || !retiredAll) {
                    return;
                }
                void* raw = ptr;
                std::string localError;
                if (JuicerCuda::retire_frame_scratch_allocation(
                        *resources,
                        raw,
                        bytes,
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

            retire_ptr(workspace.deviceState.exposureScale, sizeof(float), "frame auto-exposure scale");
            retire_ptr(workspace.deviceState.autoEV, sizeof(double), "frame auto-exposure autoEV");
            retire_ptr(workspace.deviceState.valid, sizeof(int), "frame auto-exposure valid");
            retire_ptr(workspace.scratch.maxYBits, sizeof(unsigned int), "frame auto-exposure maxYBits");
            retire_ptr(workspace.scratch.histogram, sizeof(unsigned int) * 2048u, "frame auto-exposure histogram");
            retire_ptr(
                workspace.scratch.weightsX,
                static_cast<std::size_t>(std::max(0, workspace.weightsXCapacity)) * sizeof(float),
                "frame auto-exposure weightsX");
            retire_ptr(
                workspace.scratch.weightsY,
                static_cast<std::size_t>(std::max(0, workspace.weightsYCapacity)) * sizeof(float),
                "frame auto-exposure weightsY");
            retire_ptr(
                workspace.scratch.partialsA,
                static_cast<std::size_t>(std::max(0, workspace.scratch.partialCapacity)) *
                    sizeof(JuicerCudaAutoExposurePartial),
                "frame auto-exposure partialsA");
            retire_ptr(
                workspace.scratch.partialsB,
                static_cast<std::size_t>(std::max(0, workspace.scratch.partialCapacity)) *
                    sizeof(JuicerCudaAutoExposurePartial),
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
        if (workspace.scratch.partialsA) {
            cudaFree(workspace.scratch.partialsA);
        }
        if (workspace.scratch.partialsB) {
            cudaFree(workspace.scratch.partialsB);
        }
        if (workspace.scratch.maxYBits) {
            cudaFree(workspace.scratch.maxYBits);
        }
        if (workspace.scratch.histogram) {
            cudaFree(workspace.scratch.histogram);
        }
        if (workspace.scratch.weightsX) {
            cudaFree(workspace.scratch.weightsX);
        }
        if (workspace.scratch.weightsY) {
            cudaFree(workspace.scratch.weightsY);
        }
        if (workspace.deviceState.exposureScale) {
            cudaFree(workspace.deviceState.exposureScale);
        }
        if (workspace.deviceState.autoEV) {
            cudaFree(workspace.deviceState.autoEV);
        }
        if (workspace.deviceState.valid) {
            cudaFree(workspace.deviceState.valid);
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

        cudaEvent_t ev = nullptr;
        const cudaError_t createErr = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (createErr != cudaSuccess || !ev) {
            outError = std::string("cudaEventCreateWithFlags(frame use) failed: ") +
                       (cudaGetErrorString(createErr) ? cudaGetErrorString(createErr) : "(unknown)");
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t recordErr = cudaEventRecord(ev, stream);
        if (recordErr != cudaSuccess) {
            cudaEventDestroy(ev);
            outError = std::string("cudaEventRecord(frame use) failed: ") +
                       (cudaGetErrorString(recordErr) ? cudaGetErrorString(recordErr) : "(unknown)");
            return false;
        }

        void* eventOpaque = reinterpret_cast<void*>(ev);
        if (!JuicerCuda::retain_frame_use_event(*resources, eventOpaque, outError)) {
            if (eventOpaque) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(eventOpaque));
            }
            if (outError.empty()) {
                outError = "frame use event retention failed";
            }
            return false;
        }
        frameUseEventSubmitted = true;
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
             request.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 0)) {
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
                   a.spatialDirPlaneRoles.iirForwardTempPlanes == b.spatialDirPlaneRoles.iirForwardTempPlanes &&
                   a.spatialDirPlaneRoles.cachedLogRawPlanes == b.spatialDirPlaneRoles.cachedLogRawPlanes &&
                   a.spatialDirPlaneRoles.SF_TEMP_BRIDGE_corrPlanes == b.spatialDirPlaneRoles.SF_TEMP_BRIDGE_corrPlanes &&
                   a.spatialDirPlaneRoles.SF_TEMP_BRIDGE_mixPlanes == b.spatialDirPlaneRoles.SF_TEMP_BRIDGE_mixPlanes &&
                   a.spatialDirPlaneRoles.SF_TEMP_BRIDGE_tmpPlanes == b.spatialDirPlaneRoles.SF_TEMP_BRIDGE_tmpPlanes &&
                   a.spatialDirTargetScratchTier == b.spatialDirTargetScratchTier &&
                   a.spatialDirTargetPlaneRoles.rawCorrectionPlanes == b.spatialDirTargetPlaneRoles.rawCorrectionPlanes &&
                   a.spatialDirTargetPlaneRoles.filteredCorrectionPlanes == b.spatialDirTargetPlaneRoles.filteredCorrectionPlanes &&
                   a.spatialDirTargetPlaneRoles.filterTempPlanes == b.spatialDirTargetPlaneRoles.filterTempPlanes &&
                   a.spatialDirTargetPlaneRoles.iirForwardTempPlanes == b.spatialDirTargetPlaneRoles.iirForwardTempPlanes &&
                   a.spatialDirTargetPlaneRoles.cachedLogRawPlanes == b.spatialDirTargetPlaneRoles.cachedLogRawPlanes &&
                   a.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_corrPlanes == b.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_corrPlanes &&
                   a.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_mixPlanes == b.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_mixPlanes &&
                   a.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_tmpPlanes == b.spatialDirTargetPlaneRoles.SF_TEMP_BRIDGE_tmpPlanes &&
                   a.requestedWidth == b.requestedWidth &&
                   a.requestedHeight == b.requestedHeight &&
                   a.needBlurred == b.needBlurred &&
                   a.aliasScannerRgbFromSpatialDirFiltered ==
                       b.aliasScannerRgbFromSpatialDirFiltered &&
                   a.needAux == b.needAux &&
                   a.needGrainTriplet == b.needGrainTriplet &&
                   a.needGrainShared == b.needGrainShared &&
                   a.needGateMask == b.needGateMask;
        };

        if (scratchWorkspace.retainedLeaseActive || scratchWorkspace.overflowActive) {
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
        bool retainedAcquired = false;
        if (!JuicerCuda::try_acquire_retained_frame_scratch_lease(
                *resources,
                transaction.leaseGeneration,
                cudaStreamOpaque,
                retainedAcquired,
                outError)) {
            return false;
        }
        if (retainedAcquired) {
            scratchWorkspace.request = request;
            scratchWorkspace.postFrameRequest = request;
            scratchWorkspace.postFrameRequestActive = true;
            scratchWorkspace.retainedLeaseActive = true;
            return true;
        }

        const std::size_t width = static_cast<std::size_t>(request.requestedWidth);
        const std::size_t height = static_cast<std::size_t>(request.requestedHeight);
        if (width > (std::numeric_limits<std::size_t>::max() / height)) {
            outError = "frame scratch element count overflow";
            return false;
        }
        const std::size_t requiredElements = width * height;
        if (requiredElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            outError = "frame scratch byte count overflow";
            return false;
        }
        const std::size_t planeBytes = requiredElements * sizeof(float);
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(request);
        if (!JuicerCuda::ResourceManager::command_admit_frame_scratch_overflow(
                transaction,
                *resources,
                scratchRequest,
                outError)) {
            return false;
        }

        FrameScratchWorkspace next{};
        next.request = request;
        next.postFrameRequest = request;
        next.postFrameRequestActive = true;
        next.overflowActive = true;

        auto alloc_float = [&](float*& ptr, std::size_t bytes, const char* label) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
            if (err == cudaSuccess && ptr) {
                return true;
            }
            outError = std::string("cudaMalloc(") + (label ? label : "frame scratch") + ") failed: " +
                       (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            ptr = nullptr;
            return false;
        };
        auto fail_after_partial_alloc = [&]() {
            scratchWorkspace = next;
            free_scratch_workspace_now();
            return false;
        };

        const bool needSharedTmp = workspace_request_needs_shared_tmp_plane(request);
        if (needSharedTmp) {
            if (!alloc_float(next.sharedTmpPlane, planeBytes, "frame shared tmp plane")) {
                return fail_after_partial_alloc();
            }
            next.sharedTmpWidth = request.requestedWidth;
            next.sharedTmpHeight = request.requestedHeight;
            next.sharedTmpCapacityElements = requiredElements;
        }

        if (request.needOptics) {
            JuicerCuda::Resources::DeviceOpticsScratch& optics = next.optics;
            if (!next.sharedTmpPlane) {
                outError = "frame optics scratch requires shared tmp plane";
                return fail_after_partial_alloc();
            }
            if (!request.aliasScannerRgbFromSpatialDirFiltered) {
                if (!alloc_float(optics.rgbR, planeBytes, "frame scannerScratch.rgbR") ||
                    !alloc_float(optics.rgbG, planeBytes, "frame scannerScratch.rgbG") ||
                    !alloc_float(optics.rgbB, planeBytes, "frame scannerScratch.rgbB")) {
                    return fail_after_partial_alloc();
                }
            }
            optics.tmp = next.sharedTmpPlane;
            optics.width = request.requestedWidth;
            optics.height = request.requestedHeight;
            optics.capacityElements = requiredElements;
            if (request.needBlurred &&
                !alloc_float(optics.blurred, planeBytes, "frame scannerScratch.blurred")) {
                return fail_after_partial_alloc();
            }
            if (request.needAux &&
                !alloc_float(optics.aux, planeBytes, "frame scannerScratch.aux")) {
                return fail_after_partial_alloc();
            }
            if (request.needGrainTriplet) {
                if (!alloc_float(optics.grainTmp, planeBytes, "frame scannerScratch.grainTmp") ||
                    !alloc_float(optics.grainTmpMid, planeBytes, "frame scannerScratch.grainTmpMid") ||
                    !alloc_float(optics.grainTmpCoarse, planeBytes, "frame scannerScratch.grainTmpCoarse")) {
                    return fail_after_partial_alloc();
                }
            }
            if (request.needGrainShared &&
                !alloc_float(optics.grainTmpShared, planeBytes, "frame scannerScratch.grainTmpShared")) {
                return fail_after_partial_alloc();
            }
            if (request.needGateMask) {
                const int gateWidth = (request.requestedWidth + 1) / 2;
                const int gateHeight = (request.requestedHeight + 1) / 2;
                const std::size_t gateElements =
                    static_cast<std::size_t>(gateWidth) * static_cast<std::size_t>(gateHeight);
                if (gateElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
                    outError = "frame gate-mask byte count overflow";
                    return fail_after_partial_alloc();
                }
                if (!alloc_float(
                        optics.gateMask,
                        gateElements * sizeof(float),
                        "frame scannerScratch.gateMask")) {
                    return fail_after_partial_alloc();
                }
                optics.gateWidth = gateWidth;
                optics.gateHeight = gateHeight;
                optics.gateMaskCapacityElements = gateElements;
                optics.gateMaskHash = 0;
            }
        }

        if (request.needSpatialDir) {
            JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = next.spatialDir;
            if (!spatial_dir_roles_match_tier(
                    request.spatialDirScratchTier,
                    request.spatialDirPlaneRoles)) {
                outError = "frame spatial DIR scratch tier/roles invalid";
                return fail_after_partial_alloc();
            }
            if (!alloc_float(spatialDir.rawCorrectionY, planeBytes, "frame spatial DIR rawCorrectionY") ||
                !alloc_float(
                    spatialDir.filteredCorrectionY,
                    planeBytes,
                    "frame spatial DIR filteredCorrectionY") ||
                !alloc_float(
                    spatialDir.filteredCorrectionM,
                    planeBytes,
                    "frame spatial DIR filteredCorrectionM") ||
                !alloc_float(
                    spatialDir.filteredCorrectionC,
                    planeBytes,
                    "frame spatial DIR filteredCorrectionC")) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.rawCorrectionPlanes == 3 &&
                (!alloc_float(spatialDir.rawCorrectionM, planeBytes, "frame spatial DIR rawCorrectionM") ||
                 !alloc_float(spatialDir.rawCorrectionC, planeBytes, "frame spatial DIR rawCorrectionC"))) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.filterTempPlanes > 0) {
                if (!next.sharedTmpPlane) {
                    outError = "frame spatial DIR filter temp requires shared tmp plane";
                    return fail_after_partial_alloc();
                }
                spatialDir.filterTemp = next.sharedTmpPlane;
            }
            if (request.spatialDirPlaneRoles.filterTempPlanes >= 2 &&
                !alloc_float(
                    spatialDir.filterTempM,
                    planeBytes,
                    "frame spatial DIR filterTempM")) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.filterTempPlanes == 3 &&
                !alloc_float(
                    spatialDir.filterTempC,
                    planeBytes,
                    "frame spatial DIR filterTempC")) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.iirForwardTempPlanes == 3 &&
                (!alloc_float(
                     spatialDir.iirForwardTemp,
                     planeBytes,
                     "frame spatial DIR iirForwardTemp") ||
                 !alloc_float(
                     spatialDir.iirForwardTempM,
                     planeBytes,
                     "frame spatial DIR iirForwardTempM") ||
                 !alloc_float(
                     spatialDir.iirForwardTempC,
                     planeBytes,
                     "frame spatial DIR iirForwardTempC"))) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.iirForwardTempPlanes == 1 &&
                !alloc_float(
                    spatialDir.iirForwardTemp,
                    planeBytes,
                    "frame spatial DIR iirForwardTemp")) {
                return fail_after_partial_alloc();
            }
            if (request.spatialDirPlaneRoles.cachedLogRawPlanes == 3 &&
                (!alloc_float(
                     spatialDir.logRawB,
                     planeBytes,
                     "frame spatial DIR logRawB") ||
                 !alloc_float(
                     spatialDir.logRawG,
                     planeBytes,
                     "frame spatial DIR logRawG") ||
                 !alloc_float(
                     spatialDir.logRawR,
                     planeBytes,
                     "frame spatial DIR logRawR"))) {
                return fail_after_partial_alloc();
            }
            spatialDir.width = request.requestedWidth;
            spatialDir.height = request.requestedHeight;
            spatialDir.capacityElements = requiredElements;
        }

        scratchWorkspace = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_overflow_spatial_dir_build_scratch_after_build(
        void* cudaStreamOpaque,
        JuicerCuda::SpatialDirBuildScratchReleaseStats& outStats,
        std::string& outError) {
        outStats = JuicerCuda::SpatialDirBuildScratchReleaseStats{};
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.overflowActive || !workspace.request.needSpatialDir) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for overflow spatial DIR build scratch release";
            return false;
        }
        JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = workspace.spatialDir;
        const std::size_t planeBytes = spatialDir.capacityElements * sizeof(float);
        if (planeBytes == 0) {
            return true;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        bool retiredAll = true;
        auto retire_ptr = [&](float*& ptr, const char* label, std::size_t& retiredBytes) {
            if (!ptr || !retiredAll) {
                return;
            }
            void* raw = ptr;
            std::string localError;
            if (JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    planeBytes,
                    retireStreamOpaque,
                    label,
                    localError)) {
                ptr = nullptr;
                retiredBytes += planeBytes;
                return;
            }
            retiredAll = false;
            outError = localError.empty()
                           ? std::string(label ? label : "overflow spatial DIR build scratch") +
                                 " retire failed"
                           : localError;
        };

        retire_ptr(
            spatialDir.rawCorrectionY,
            "overflow spatial DIR build rawCorrectionY",
            outStats.rawCorrectionRetiredBytes);
        retire_ptr(
            spatialDir.rawCorrectionM,
            "overflow spatial DIR build rawCorrectionM",
            outStats.rawCorrectionRetiredBytes);
        retire_ptr(
            spatialDir.rawCorrectionC,
            "overflow spatial DIR build rawCorrectionC",
            outStats.rawCorrectionRetiredBytes);
        retire_ptr(
            spatialDir.filterTempM,
            "overflow spatial DIR build filterTempM",
            outStats.filterTempRetiredBytes);
        retire_ptr(
            spatialDir.filterTempC,
            "overflow spatial DIR build filterTempC",
            outStats.filterTempRetiredBytes);
        retire_ptr(
            spatialDir.iirForwardTemp,
            "overflow spatial DIR build iirForwardTemp",
            outStats.iirForwardTempRetiredBytes);
        retire_ptr(
            spatialDir.iirForwardTempM,
            "overflow spatial DIR build iirForwardTempM",
            outStats.iirForwardTempRetiredBytes);
        retire_ptr(
            spatialDir.iirForwardTempC,
            "overflow spatial DIR build iirForwardTempC",
            outStats.iirForwardTempRetiredBytes);
        if (!retiredAll) {
            if (outError.empty()) {
                outError = "overflow spatial DIR build scratch retire failed";
            }
            return false;
        }

        const float* sharedTmp = workspace.sharedTmpPlane;
        spatialDir.filterTemp = nullptr;
        const bool opticsKeepsSharedTmp =
            sharedTmp && workspace.optics.tmp == sharedTmp &&
            (workspace.optics.rgbR || workspace.optics.rgbG || workspace.optics.rgbB ||
             workspace.optics.blurred || workspace.optics.aux || workspace.optics.grainTmp ||
             workspace.optics.grainTmpShared || workspace.optics.grainTmpMid ||
             workspace.optics.grainTmpCoarse || workspace.optics.gateMask);
        if (sharedTmp && !opticsKeepsSharedTmp) {
            void* raw = workspace.sharedTmpPlane;
            std::string localError;
            if (!JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    workspace.sharedTmpCapacityElements * sizeof(float),
                    retireStreamOpaque,
                    "overflow spatial DIR build shared tmp plane",
                    localError)) {
                outError = localError.empty()
                               ? "overflow spatial DIR build shared tmp retire failed"
                               : localError;
                return false;
            }
            outStats.sharedTmpRetiredBytes =
                workspace.sharedTmpCapacityElements * sizeof(float);
            workspace.sharedTmpPlane = nullptr;
            workspace.sharedTmpWidth = 0;
            workspace.sharedTmpHeight = 0;
            workspace.sharedTmpCapacityElements = 0;
            if (workspace.optics.tmp == sharedTmp) {
                workspace.optics.tmp = nullptr;
            }
        }

        const bool syncBeforeReap =
            outStats.rawCorrectionRetiredBytes > 0 ||
            outStats.filterTempRetiredBytes > 0 ||
            outStats.iirForwardTempRetiredBytes > 0 ||
            outStats.sharedTmpRetiredBytes > 0;
        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = retireStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(retireStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(overflow spatial DIR build scratch) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return JuicerCuda::reap_retired_allocations(
            *resources,
            outStats.reclaimedBytes,
            outError);
    }

    bool Root::PreparedCudaFrame::State::stage_overflow_spatial_dir_cached_log_raw_for_final_develop(
        void* cudaStreamOpaque,
        JuicerCuda::SpatialDirCachedLogRawStageStats& outStats,
        std::string& outError) {
        outStats = JuicerCuda::SpatialDirCachedLogRawStageStats{};
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.overflowActive || !workspace.request.needSpatialDir) {
            return true;
        }
        if (workspace.request.spatialDirTargetPlaneRoles.cachedLogRawPlanes != 3) {
            outError = "overflow spatial DIR cached log raw target roles invalid";
            return false;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for overflow spatial DIR cached log raw stage";
            return false;
        }

        JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = workspace.spatialDir;
        if (!spatialDir.filteredCorrectionY || !spatialDir.filteredCorrectionM ||
            !spatialDir.filteredCorrectionC || spatialDir.capacityElements == 0) {
            outError = "overflow spatial DIR cached log raw stage missing filtered correction residency";
            return false;
        }
        if (spatialDir.logRawB && spatialDir.logRawG && spatialDir.logRawR) {
            return true;
        }
        const std::size_t planeBytes = spatialDir.capacityElements * sizeof(float);
        if (planeBytes == 0) {
            outError = "overflow spatial DIR cached log raw stage byte count invalid";
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(workspace.request);
        if (!JuicerCuda::ResourceManager::command_admit_spatial_dir_cached_log_raw_overflow(
                transaction,
                *resources,
                scratchRequest,
                outError)) {
            return false;
        }

        auto free_log_raw = [&]() noexcept {
            if (spatialDir.logRawB) {
                cudaFree(spatialDir.logRawB);
                spatialDir.logRawB = nullptr;
            }
            if (spatialDir.logRawG) {
                cudaFree(spatialDir.logRawG);
                spatialDir.logRawG = nullptr;
            }
            if (spatialDir.logRawR) {
                cudaFree(spatialDir.logRawR);
                spatialDir.logRawR = nullptr;
            }
        };
        auto alloc_float = [&](float*& ptr, const char* label) {
            if (ptr) {
                return true;
            }
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), planeBytes);
            if (err == cudaSuccess && ptr) {
                outStats.cachedLogRawAllocatedBytes += planeBytes;
                return true;
            }
            outError = std::string("cudaMalloc(") + (label ? label : "overflow spatial DIR logRaw") +
                       ") failed: " +
                       (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            ptr = nullptr;
            return false;
        };

        if (!alloc_float(spatialDir.logRawB, "overflow spatial DIR final logRawB") ||
            !alloc_float(spatialDir.logRawG, "overflow spatial DIR final logRawG") ||
            !alloc_float(spatialDir.logRawR, "overflow spatial DIR final logRawR")) {
            free_log_raw();
            return false;
        }
        remember_stream(cudaStreamOpaque);
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_overflow_spatial_dir_cached_log_raw_after_final_develop(
        void* cudaStreamOpaque,
        JuicerCuda::SpatialDirCachedLogRawReleaseStats& outStats,
        std::string& outError) {
        outStats = JuicerCuda::SpatialDirCachedLogRawReleaseStats{};
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.overflowActive || !workspace.request.needSpatialDir) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for overflow spatial DIR cached log raw release";
            return false;
        }
        JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = workspace.spatialDir;
        const std::size_t planeBytes = spatialDir.capacityElements * sizeof(float);
        if (planeBytes == 0) {
            return true;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        bool retiredAll = true;
        auto retire_ptr = [&](float*& ptr, const char* label) {
            if (!ptr || !retiredAll) {
                return;
            }
            void* raw = ptr;
            std::string localError;
            if (JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    planeBytes,
                    retireStreamOpaque,
                    label,
                    localError)) {
                ptr = nullptr;
                outStats.cachedLogRawRetiredBytes += planeBytes;
                return;
            }
            retiredAll = false;
            outError = localError.empty()
                           ? std::string(label ? label : "overflow spatial DIR cached log raw") +
                                 " retire failed"
                           : localError;
        };

        retire_ptr(spatialDir.logRawB, "overflow spatial DIR final logRawB");
        retire_ptr(spatialDir.logRawG, "overflow spatial DIR final logRawG");
        retire_ptr(spatialDir.logRawR, "overflow spatial DIR final logRawR");
        if (!retiredAll) {
            if (outError.empty()) {
                outError = "overflow spatial DIR cached log raw retire failed";
            }
            return false;
        }
        if (outStats.cachedLogRawRetiredBytes == 0) {
            return true;
        }

        const cudaStream_t stream = retireStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(retireStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(overflow spatial DIR cached log raw) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return JuicerCuda::reap_retired_allocations(
            *resources,
            outStats.reclaimedBytes,
            outError);
    }

    bool Root::PreparedCudaFrame::State::release_overflow_spatial_dir_stage_after_scan_linear(
        void* cudaStreamOpaque,
        JuicerCuda::SpatialDirStageReleaseStats& outStats,
        std::string& outError) {
        outStats = JuicerCuda::SpatialDirStageReleaseStats{};
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.overflowActive || !workspace.request.needSpatialDir) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for overflow spatial DIR stage release";
            return false;
        }
        JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = workspace.spatialDir;
        const std::size_t planeBytes = spatialDir.capacityElements * sizeof(float);
        if (planeBytes == 0) {
            workspace.request.needSpatialDir = false;
            spatialDir = JuicerCuda::Resources::DeviceSpatialDirScratch{};
            return true;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        bool retiredAll = true;
        auto retire_ptr = [&](float*& ptr, const char* label) {
            if (!ptr || !retiredAll) {
                return;
            }
            void* raw = ptr;
            std::string localError;
            if (JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    planeBytes,
                    retireStreamOpaque,
                    label,
                    localError)) {
                ptr = nullptr;
                outStats.retiredBytes += planeBytes;
                return;
            }
            retiredAll = false;
            outError = localError.empty()
                           ? std::string(label ? label : "overflow spatial DIR stage") +
                                 " retire failed"
                           : localError;
        };

        retire_ptr(spatialDir.rawCorrectionY, "overflow spatial DIR rawCorrectionY");
        retire_ptr(spatialDir.rawCorrectionM, "overflow spatial DIR rawCorrectionM");
        retire_ptr(spatialDir.rawCorrectionC, "overflow spatial DIR rawCorrectionC");
        retire_ptr(spatialDir.filteredCorrectionY, "overflow spatial DIR filteredCorrectionY");
        retire_ptr(spatialDir.filteredCorrectionM, "overflow spatial DIR filteredCorrectionM");
        retire_ptr(spatialDir.filteredCorrectionC, "overflow spatial DIR filteredCorrectionC");
        retire_ptr(spatialDir.filterTempM, "overflow spatial DIR filterTempM");
        retire_ptr(spatialDir.filterTempC, "overflow spatial DIR filterTempC");
        retire_ptr(spatialDir.iirForwardTemp, "overflow spatial DIR iirForwardTemp");
        retire_ptr(spatialDir.iirForwardTempM, "overflow spatial DIR iirForwardTempM");
        retire_ptr(spatialDir.iirForwardTempC, "overflow spatial DIR iirForwardTempC");
        retire_ptr(spatialDir.logRawB, "overflow spatial DIR logRawB");
        retire_ptr(spatialDir.logRawG, "overflow spatial DIR logRawG");
        retire_ptr(spatialDir.logRawR, "overflow spatial DIR logRawR");
        if (!retiredAll) {
            if (outError.empty()) {
                outError = "overflow spatial DIR stage retire failed";
            }
            return false;
        }

        const cudaStream_t stream = retireStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(retireStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(overflow spatial DIR stage) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        if (!JuicerCuda::reap_retired_allocations(
                *resources,
                outStats.reclaimedBytes,
                outError)) {
            return false;
        }
        spatialDir = JuicerCuda::Resources::DeviceSpatialDirScratch{};
        workspace.request.needSpatialDir = false;
        spatialDirDescriptor = Spektrafilm::SpatialDirDescriptor{};
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scratch_workspace_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.retainedLeaseActive && !workspace.overflowActive) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scratch release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;

        if (workspace.retainedLeaseActive) {
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

        bool retiredAll = true;
        auto retire_ptr = [&](float*& ptr, std::size_t bytes, const char* label) {
            if (!ptr || !retiredAll) {
                return;
            }
            void* raw = ptr;
            std::string localError;
            if (JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    bytes,
                    retireStreamOpaque,
                    label,
                    localError)) {
                ptr = nullptr;
                return;
            }
            retiredAll = false;
            outError = localError.empty()
                           ? std::string(label ? label : "frame scratch") + " retire failed"
                           : localError;
        };

        const std::size_t planeBytes = workspace.optics.capacityElements > 0
                                           ? workspace.optics.capacityElements * sizeof(float)
                                           : workspace.spatialDir.capacityElements * sizeof(float);
        retire_ptr(workspace.optics.rgbR, planeBytes, "frame scannerScratch.rgbR");
        retire_ptr(workspace.optics.rgbG, planeBytes, "frame scannerScratch.rgbG");
        retire_ptr(workspace.optics.rgbB, planeBytes, "frame scannerScratch.rgbB");
        retire_ptr(workspace.optics.blurred, planeBytes, "frame scannerScratch.blurred");
        retire_ptr(workspace.optics.aux, planeBytes, "frame scannerScratch.aux");
        retire_ptr(workspace.optics.grainTmp, planeBytes, "frame scannerScratch.grainTmp");
        retire_ptr(workspace.optics.grainTmpShared, planeBytes, "frame scannerScratch.grainTmpShared");
        retire_ptr(workspace.optics.grainTmpMid, planeBytes, "frame scannerScratch.grainTmpMid");
        retire_ptr(workspace.optics.grainTmpCoarse, planeBytes, "frame scannerScratch.grainTmpCoarse");
        retire_ptr(
            workspace.optics.gateMask,
            workspace.optics.gateMaskCapacityElements * sizeof(float),
            "frame scannerScratch.gateMask");
        retire_ptr(workspace.spatialDir.rawCorrectionY, planeBytes, "frame spatial DIR rawCorrectionY");
        retire_ptr(workspace.spatialDir.rawCorrectionM, planeBytes, "frame spatial DIR rawCorrectionM");
        retire_ptr(workspace.spatialDir.rawCorrectionC, planeBytes, "frame spatial DIR rawCorrectionC");
        retire_ptr(
            workspace.spatialDir.filteredCorrectionY,
            planeBytes,
            "frame spatial DIR filteredCorrectionY");
        retire_ptr(
            workspace.spatialDir.filteredCorrectionM,
            planeBytes,
            "frame spatial DIR filteredCorrectionM");
        retire_ptr(
            workspace.spatialDir.filteredCorrectionC,
            planeBytes,
            "frame spatial DIR filteredCorrectionC");
        retire_ptr(
            workspace.spatialDir.filterTempM,
            planeBytes,
            "frame spatial DIR filterTempM");
        retire_ptr(
            workspace.spatialDir.filterTempC,
            planeBytes,
            "frame spatial DIR filterTempC");
        retire_ptr(
            workspace.spatialDir.iirForwardTemp,
            planeBytes,
            "frame spatial DIR iirForwardTemp");
        retire_ptr(
            workspace.spatialDir.iirForwardTempM,
            planeBytes,
            "frame spatial DIR iirForwardTempM");
        retire_ptr(
            workspace.spatialDir.iirForwardTempC,
            planeBytes,
            "frame spatial DIR iirForwardTempC");
        retire_ptr(workspace.spatialDir.logRawB, planeBytes, "frame spatial DIR logRawB");
        retire_ptr(workspace.spatialDir.logRawG, planeBytes, "frame spatial DIR logRawG");
        retire_ptr(workspace.spatialDir.logRawR, planeBytes, "frame spatial DIR logRawR");
        retire_ptr(
            workspace.sharedTmpPlane,
            workspace.sharedTmpCapacityElements * sizeof(float),
            "frame shared tmp plane");

        if (retiredAll) {
            workspace = FrameScratchWorkspace{};
            return true;
        }
        if (outError.empty()) {
            outError = "frame scratch retire failed";
        }
        return false;
    }

    void Root::PreparedCudaFrame::State::free_scratch_workspace_now() noexcept {
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (workspace.optics.rgbR) {
            cudaFree(workspace.optics.rgbR);
        }
        if (workspace.optics.rgbG) {
            cudaFree(workspace.optics.rgbG);
        }
        if (workspace.optics.rgbB) {
            cudaFree(workspace.optics.rgbB);
        }
        if (workspace.optics.blurred) {
            cudaFree(workspace.optics.blurred);
        }
        if (workspace.optics.aux) {
            cudaFree(workspace.optics.aux);
        }
        if (workspace.optics.grainTmp) {
            cudaFree(workspace.optics.grainTmp);
        }
        if (workspace.optics.grainTmpShared) {
            cudaFree(workspace.optics.grainTmpShared);
        }
        if (workspace.optics.grainTmpMid) {
            cudaFree(workspace.optics.grainTmpMid);
        }
        if (workspace.optics.grainTmpCoarse) {
            cudaFree(workspace.optics.grainTmpCoarse);
        }
        if (workspace.optics.gateMask) {
            cudaFree(workspace.optics.gateMask);
        }
        if (workspace.spatialDir.rawCorrectionY) {
            cudaFree(workspace.spatialDir.rawCorrectionY);
        }
        if (workspace.spatialDir.rawCorrectionM) {
            cudaFree(workspace.spatialDir.rawCorrectionM);
        }
        if (workspace.spatialDir.rawCorrectionC) {
            cudaFree(workspace.spatialDir.rawCorrectionC);
        }
        if (workspace.spatialDir.filteredCorrectionY) {
            cudaFree(workspace.spatialDir.filteredCorrectionY);
        }
        if (workspace.spatialDir.filteredCorrectionM) {
            cudaFree(workspace.spatialDir.filteredCorrectionM);
        }
        if (workspace.spatialDir.filteredCorrectionC) {
            cudaFree(workspace.spatialDir.filteredCorrectionC);
        }
        if (workspace.spatialDir.filterTempM) {
            cudaFree(workspace.spatialDir.filterTempM);
        }
        if (workspace.spatialDir.filterTempC) {
            cudaFree(workspace.spatialDir.filterTempC);
        }
        if (workspace.spatialDir.iirForwardTemp) {
            cudaFree(workspace.spatialDir.iirForwardTemp);
        }
        if (workspace.spatialDir.iirForwardTempM) {
            cudaFree(workspace.spatialDir.iirForwardTempM);
        }
        if (workspace.spatialDir.iirForwardTempC) {
            cudaFree(workspace.spatialDir.iirForwardTempC);
        }
        if (workspace.spatialDir.logRawB) {
            cudaFree(workspace.spatialDir.logRawB);
        }
        if (workspace.spatialDir.logRawG) {
            cudaFree(workspace.spatialDir.logRawG);
        }
        if (workspace.spatialDir.logRawR) {
            cudaFree(workspace.spatialDir.logRawR);
        }
        if (workspace.sharedTmpPlane) {
            cudaFree(workspace.sharedTmpPlane);
        }
        workspace = FrameScratchWorkspace{};
    }

    Root::PreparedCudaFrame::PreparedCudaFrame(std::unique_ptr<State> state) noexcept
        : _state(std::move(state)) {
    }

    Root::PreparedCudaFrame::~PreparedCudaFrame() {
        abort("prepared_frame_scope_exit");
    }

    Root::PreparedCudaFrame::PreparedCudaFrame(PreparedCudaFrame&& other) noexcept
        : _state(std::move(other._state)) {
    }

    Root::PreparedCudaFrame& Root::PreparedCudaFrame::operator=(PreparedCudaFrame&& other) noexcept {
        if (this != &other) {
            abort("prepared_frame_move_assignment");
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

    bool Root::PreparedCudaFrame::WorkspaceLeaseMarker::has_any_family() const noexcept {
        return _active && (_request.needOptics || _request.needSpatialDir);
    }

    std::uint64_t Root::PreparedCudaFrame::WorkspaceLeaseMarker::lease_generation() const noexcept {
        return _active ? _leaseGeneration : 0;
    }

    bool Root::PreparedCudaFrame::active() const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed;
    }

    JuicerCuda::ResourceManager::ScratchRequestDescriptor
    Root::PreparedCudaFrame::State::post_frame_scratch_request_descriptor() const noexcept {
        if (!scratchWorkspace.retainedLeaseActive && !scratchWorkspace.overflowActive) {
            return JuicerCuda::ResourceManager::ScratchRequestDescriptor{};
        }
        const WorkspaceRequest& request = scratchWorkspace.postFrameRequestActive
                                              ? scratchWorkspace.postFrameRequest
                                              : scratchWorkspace.request;
        return make_workspace_scratch_request_descriptor(request);
    }

    Root::PreparedCudaFrame::WorkspaceLeaseMarker Root::PreparedCudaFrame::bind_workspace_request(
        const WorkspaceRequest& request) const noexcept {
        if (!active()) {
            return WorkspaceLeaseMarker{};
        }
        return WorkspaceLeaseMarker(request, _state->transaction.leaseGeneration);
    }

    JuicerCuda::ResourceManager::ScratchRequestDescriptor Root::PreparedCudaFrame::make_scratch_request_descriptor(
        const WorkspaceLeaseMarker& workspace) noexcept {
        return make_workspace_scratch_request_descriptor(workspace._request);
    }

    const Root::PreparedCudaFrame::WorkspaceRequest&
    Root::PreparedCudaFrame::active_workspace_request(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        if (_state &&
            (_state->scratchWorkspace.retainedLeaseActive ||
             _state->scratchWorkspace.overflowActive) &&
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
               workspace.lease_generation() == _state->transaction.leaseGeneration;
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
        if (workspace.lease_generation() != _state->transaction.leaseGeneration) {
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
        if (!_state->root->shed_post_frame_scratch(
                _state->transaction,
                *_state->resources,
                postFrameScratchRequest,
                cudaStreamOpaque,
                "command_shed_post_frame_scratch_finish",
                outError)) {
            return false;
        }

        if (!_state->root->commit_submission(_state->transaction, cudaStreamOpaque, outError)) {
            return false;
        }
        return true;
    }

    void Root::PreparedCudaFrame::abort(const char* reason) noexcept {
        if (_state) {
            try {
                std::string releaseError;
                const JuicerCuda::ResourceManager::ScratchRequestDescriptor postFrameScratchRequest =
                    _state->post_frame_scratch_request_descriptor();
                (void)_state->release_scan_error_stage_after_use(
                    _state->lastCudaStreamOpaque,
                    releaseError);
                if (!_state->release_scratch_workspace_after_use(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_scratch_workspace_release_failed abort=1";
                    if (reason && reason[0]) {
                        msg += " reason=";
                        msg += reason;
                    }
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
                    if (reason && reason[0]) {
                        msg += " reason=";
                        msg += reason;
                    }
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (_state->root && _state->resources && _state->transaction.active &&
                    !_state->transaction.committed &&
                    !_state->root->shed_post_frame_scratch(
                        _state->transaction,
                        *_state->resources,
                        postFrameScratchRequest,
                        _state->lastCudaStreamOpaque,
                        "command_shed_post_frame_scratch_abort",
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_post_frame_scratch_shed_failed abort=1";
                    if (reason && reason[0]) {
                        msg += " reason=";
                        msg += reason;
                    }
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
            _state->root->rollback_submission(_state->transaction, reason);
        }
    }

    bool Root::PreparedCudaFrame::prepare_current_medium(
        const WorkingState& workingState,
        bool negativeMedium,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        const char* stageTag = negativeMedium ? "current_medium_upload_negative"
                                              : "current_medium_upload_print";
        if (!JuicerCuda::ResourceManager::command_ensure_current_medium_uploaded(
                _state->transaction,
                *_state->resources,
                workingState,
                negativeMedium,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(stageTag, "CUDA current-medium upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_scan_lut(
        const WorkingState& workingState,
        bool negativeMedium,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        const char* stageTag = negativeMedium ? "scan_lut_negative"
                                              : "scan_lut_print";
        const char* failurePrefix = negativeMedium ? "CUDA scan LUT upload failed"
                                                   : "CUDA print scan LUT upload failed";
        if (!JuicerCuda::ResourceManager::command_ensure_scan_lut(
                _state->transaction,
                *_state->resources,
                workingState,
                negativeMedium,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(stageTag, failurePrefix);
            return false;
        }
        return true;
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
            const bool marksContextLoss = !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
            _state->set_failure(
                "acquire_frame_scratch_workspace",
                marksContextLoss
                    ? "CUDA frame scratch workspace acquisition failed"
                    : "CUDA frame scratch workspace admission failed",
                marksContextLoss);
            return false;
        }
        if (_state->scratchWorkspace.overflowActive) {
            return true;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(activeRequest);
        if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            const bool marksContextLoss = !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
            _state->set_failure(
                "command_ensure_optics_scratch",
                marksContextLoss
                    ? "CUDA optics scratch allocation failed"
                    : "CUDA optics scratch deferred by contention policy",
                marksContextLoss);
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::try_stage_profile_optical_workspace(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        const WorkspaceRequest& activeRequest = active_workspace_request(workspace);
        if (!_state->ensure_scratch_workspace(activeRequest, cudaStreamOpaque, outError)) {
            return false;
        }
        if (_state->scratchWorkspace.overflowActive) {
            return true;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(activeRequest);
        return JuicerCuda::ResourceManager::command_ensure_optics_scratch(
            _state->transaction,
            *_state->resources,
            scratchRequest,
            cudaStreamOpaque,
            outError);
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

        std::array<WorkspaceRequest, 4> scratchCandidates{};
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
        add_scratch_candidate(workspace._request);
        if (spatial_dir_descriptor_has_strict_yvv(descriptor)) {
            const bool aliasTarget =
                spatial_dir_alias_target_uses_build_roles(workspace._request);
            Spektrafilm::DirScratchPlaneRoles lowScratchBuild =
                strict_yvv_low_scratch_plane_roles();
            Spektrafilm::DirScratchPlaneRoles lowScratchTarget = lowScratchBuild;
            lowScratchTarget.cachedLogRawPlanes =
                aliasTarget ? 0 : descriptor.targetPlaneRoles.cachedLogRawPlanes;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace._request,
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
                singleTempBuild,
                aliasTarget ? Spektrafilm::DirScratchTier::Tier1IChannels : descriptor.targetScratchTier,
                singleTempTarget));

            Spektrafilm::DirScratchPlaneRoles streamedBuild =
                strict_yvv_component_streamed_plane_roles();
            Spektrafilm::DirScratchPlaneRoles streamedTarget = streamedBuild;
            add_scratch_candidate(strict_yvv_workspace_candidate(
                workspace._request,
                streamedBuild,
                Spektrafilm::DirScratchTier::Tier1IChannels,
                streamedTarget));
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
            msg += " legacy_compatibility_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.legacyCompatibilityHash));
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
            msg += " fft_mode=";
            msg += descriptor.approximation == Spektrafilm::DirApproximationMarker::AcceptedFftReplicatePadSmooth
                       ? "accepted_fft_replicate_pad_smooth"
                       : "none";
            msg += " fft_pad_pixels=";
            msg += std::to_string(descriptor.fftPadPixels);
            msg += " fft_pad_sigma=";
            msg += std::to_string(descriptor.fftPadSigma);
            msg += " fft_width=";
            msg += std::to_string(descriptor.fftWidth);
            msg += " fft_height=";
            msg += std::to_string(descriptor.fftHeight);
            msg += " fft_complex_width=";
            msg += std::to_string(descriptor.fftComplexWidth);
            msg += " scratch_source=pending";
            msg += " component_count=";
            msg += std::to_string(descriptor.filterPlan.componentCount);
            msg += " raw_correction_planes=";
            msg += std::to_string(roles.rawCorrectionPlanes);
            msg += " filtered_correction_planes=";
            msg += std::to_string(roles.filteredCorrectionPlanes);
            msg += " filter_temp_planes=";
            msg += std::to_string(roles.filterTempPlanes);
            msg += " iir_forward_temp_planes=";
            msg += std::to_string(roles.iirForwardTempPlanes);
            msg += " cached_log_raw_planes=";
            msg += std::to_string(roles.cachedLogRawPlanes);
            msg += " SF_TEMP_BRIDGE_corr_planes=";
            msg += std::to_string(roles.SF_TEMP_BRIDGE_corrPlanes);
            msg += " SF_TEMP_BRIDGE_mix_planes=";
            msg += std::to_string(roles.SF_TEMP_BRIDGE_mixPlanes);
            msg += " SF_TEMP_BRIDGE_tmp_planes=";
            msg += std::to_string(roles.SF_TEMP_BRIDGE_tmpPlanes);
            msg += " target_raw_correction_planes=";
            msg += std::to_string(targetRoles.rawCorrectionPlanes);
            msg += " target_filtered_correction_planes=";
            msg += std::to_string(targetRoles.filteredCorrectionPlanes);
            msg += " target_filter_temp_planes=";
            msg += std::to_string(targetRoles.filterTempPlanes);
            msg += " target_iir_forward_temp_planes=";
            msg += std::to_string(targetRoles.iirForwardTempPlanes);
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
                msg += " iir_forward_temp_planes=";
                msg += std::to_string(roles.iirForwardTempPlanes);
                msg += " cached_log_raw_planes=";
                msg += std::to_string(roles.cachedLogRawPlanes);
                msg += " target_raw_correction_planes=";
                msg += std::to_string(targetRoles.rawCorrectionPlanes);
                msg += " target_filtered_correction_planes=";
                msg += std::to_string(targetRoles.filteredCorrectionPlanes);
                msg += " target_filter_temp_planes=";
                msg += std::to_string(targetRoles.filterTempPlanes);
                msg += " target_iir_forward_temp_planes=";
                msg += std::to_string(targetRoles.iirForwardTempPlanes);
                msg += " target_cached_log_raw_planes=";
                msg += std::to_string(targetRoles.cachedLogRawPlanes);
                JTRACE("DIR_DESCRIPTOR", msg);
            }
#endif
            if (_state->scratchWorkspace.retainedLeaseActive &&
                !_state->scratchWorkspace.overflowActive) {
                _state->scratchWorkspace.request = candidate;
                _state->scratchWorkspace.postFrameRequest = candidate;
                _state->scratchWorkspace.postFrameRequestActive = true;
            } else if (!_state->ensure_scratch_workspace(candidate, cudaStreamOpaque, scratchAdmissionError)) {
                if (JuicerCuda::ResourceManager::error_is_scratch_exhausted(scratchAdmissionError) &&
                    candidateIndex + 1 < scratchCandidateCount) {
                    continue;
                }
                outError = scratchAdmissionError;
                break;
            }

            const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
                make_workspace_scratch_request_descriptor(candidate);
            if (!_state->scratchWorkspace.overflowActive) {
                if (descriptor.approximation == Spektrafilm::DirApproximationMarker::AcceptedFftReplicatePadSmooth &&
                    !JuicerCuda::ResourceManager::command_preflight_spatial_dir_fft_peak(
                        _state->transaction,
                        *_state->resources,
                        descriptor,
                        scratchRequest,
                        cudaStreamOpaque,
                        scratchAdmissionError)) {
                    outError = scratchAdmissionError;
                    break;
                }
                if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_scratch(
                        _state->transaction,
                        *_state->resources,
                        scratchRequest,
                        cudaStreamOpaque,
                        scratchAdmissionError)) {
                    if (JuicerCuda::ResourceManager::error_is_scratch_exhausted(scratchAdmissionError) &&
                        candidateIndex + 1 < scratchCandidateCount) {
                        continue;
                    }
                    outError = scratchAdmissionError;
                    break;
                }
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
            const bool marksContextLoss = !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
            _state->set_failure(
                "command_ensure_spatial_dir_scratch",
                marksContextLoss
                    ? "CUDA spatial DIR scratch allocation failed"
                    : "CUDA spatial DIR scratch deferred by contention policy",
                marksContextLoss);
            return false;
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const Spektrafilm::DirScratchPlaneRoles& roles = admittedRequest.spatialDirPlaneRoles;
            std::string msg = "event=spatial_dir_scratch_admission descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptor.hash));
            msg += " scratch_source=";
            msg += _state->scratchWorkspace.overflowActive ? "overflow" : "retained";
            msg += " scratch_tier=";
            msg += Spektrafilm::to_cstr(admittedRequest.spatialDirScratchTier);
            msg += " strict_yvv_shape=";
            msg += strict_yvv_candidate_label(roles);
            msg += " fft_mode=";
            msg += descriptor.approximation == Spektrafilm::DirApproximationMarker::AcceptedFftReplicatePadSmooth
                       ? "accepted_fft_replicate_pad_smooth"
                       : "none";
            msg += " fft_pad_pixels=";
            msg += std::to_string(descriptor.fftPadPixels);
            msg += " fft_width=";
            msg += std::to_string(descriptor.fftWidth);
            msg += " fft_height=";
            msg += std::to_string(descriptor.fftHeight);
            msg += " fft_complex_width=";
            msg += std::to_string(descriptor.fftComplexWidth);
            msg += " raw_correction_planes=";
            msg += std::to_string(roles.rawCorrectionPlanes);
            msg += " filtered_correction_planes=";
            msg += std::to_string(roles.filteredCorrectionPlanes);
            msg += " filter_temp_planes=";
            msg += std::to_string(roles.filterTempPlanes);
            msg += " iir_forward_temp_planes=";
            msg += std::to_string(roles.iirForwardTempPlanes);
            msg += " cached_log_raw_planes=";
            msg += std::to_string(roles.cachedLogRawPlanes);
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_workspace_scratch_request_descriptor(admittedRequest);
        if (descriptor.approximation == Spektrafilm::DirApproximationMarker::AcceptedFftReplicatePadSmooth) {
            if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_fft(
                    _state->transaction,
                    *_state->resources,
                    descriptor,
                    scratchRequest,
                    cudaStreamOpaque,
                    outError)) {
                const bool marksContextLoss =
                    !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
                _state->set_failure(
                    "command_ensure_spatial_dir_fft",
                    marksContextLoss
                        ? "CUDA spatial DIR FFT preparation failed"
                        : "CUDA spatial DIR FFT admission failed",
                    marksContextLoss);
                return false;
            }
        } else {
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
                if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_kernel(
                        _state->transaction,
                        *_state->resources,
                        _state->resources->spatialDirKernels[static_cast<std::size_t>(slot)],
                        sigmas[slot],
                        cudaStreamOpaque,
                        outError)) {
                    _state->set_failure(
                        "command_ensure_spatial_dir_kernel",
                        "CUDA spatial DIR kernel upload failed");
                    return false;
                }
            }
        }
        _state->spatialDirDescriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::release_spatial_dir_build_scratch_after_build(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        if (!workspace._request.needSpatialDir) {
            return true;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return true;
        }

        JuicerCuda::SpatialDirBuildScratchReleaseStats releaseStats{};
        const bool overflow = _state->scratchWorkspace.overflowActive;
        bool ok = false;
        if (overflow) {
            ok = _state->release_overflow_spatial_dir_build_scratch_after_build(
                cudaStreamOpaque,
                releaseStats,
                outError);
        } else {
            _state->remember_stream(cudaStreamOpaque);
            ok = JuicerCuda::ResourceManager::command_release_spatial_dir_build_scratch_stage(
                _state->transaction,
                *_state->resources,
                cudaStreamOpaque,
                releaseStats,
                outError);
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace._request.spatialDirDescriptorHash;
            std::string msg = "event=spatial_dir_build_scratch_release_after_build";
            msg += " descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
            msg += " scratch_source=";
            msg += overflow ? "overflow" : "retained";
            msg += " ok=";
            msg += ok ? "1" : "0";
            msg += " pending_scratch_bytes_before=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.pendingScratchBytesBefore));
            msg += " raw_correction_retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.rawCorrectionRetiredBytes));
            msg += " filter_temp_retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.filterTempRetiredBytes));
            msg += " iir_forward_temp_retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.iirForwardTempRetiredBytes));
            msg += " shared_tmp_retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.sharedTmpRetiredBytes));
            msg += " reclaimed_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.reclaimedBytes));
            if (!outError.empty()) {
                msg += " error=";
                msg += outError;
            }
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        if (!ok) {
            _state->set_failure(
                "release_spatial_dir_build_scratch_after_build",
                "CUDA spatial DIR build scratch release failed");
        }
        return ok;
    }

    bool Root::PreparedCudaFrame::release_spatial_dir_stage_after_scan_linear(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        if (!workspace._request.needSpatialDir) {
            return true;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return true;
        }

        JuicerCuda::SpatialDirStageReleaseStats releaseStats{};
        const bool overflow = _state->scratchWorkspace.overflowActive;
        bool ok = false;
        if (overflow) {
            ok = _state->release_overflow_spatial_dir_stage_after_scan_linear(
                cudaStreamOpaque,
                releaseStats,
                outError);
        } else {
            _state->remember_stream(cudaStreamOpaque);
            ok = JuicerCuda::ResourceManager::command_release_spatial_dir_scratch_stage(
                _state->transaction,
                *_state->resources,
                cudaStreamOpaque,
                releaseStats,
                outError);
            if (ok) {
                _state->scratchWorkspace.request.needSpatialDir = false;
                _state->spatialDirDescriptor = Spektrafilm::SpatialDirDescriptor{};
            }
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace._request.spatialDirDescriptorHash;
            std::string msg = "event=spatial_dir_stage_release_after_scan_linear";
            msg += " descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
            msg += " scratch_source=";
            msg += overflow ? "overflow" : "retained";
            msg += " ok=";
            msg += ok ? "1" : "0";
            msg += " retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.retiredBytes));
            msg += " reclaimed_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.reclaimedBytes));
            if (!outError.empty()) {
                msg += " error=";
                msg += outError;
            }
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        if (!ok) {
            _state->set_failure(
                "release_spatial_dir_stage_after_scan_linear",
                "CUDA spatial DIR stage release failed");
        }
        return ok;
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
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            outError = "spatial DIR cached log raw staging requires an active scratch workspace";
            return false;
        }

        JuicerCuda::SpatialDirCachedLogRawStageStats stageStats{};
        const bool overflow = _state->scratchWorkspace.overflowActive;
        bool ok = false;
        if (overflow) {
            ok = _state->stage_overflow_spatial_dir_cached_log_raw_for_final_develop(
                cudaStreamOpaque,
                stageStats,
                outError);
        } else {
            _state->remember_stream(cudaStreamOpaque);
            const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
                make_workspace_scratch_request_descriptor(effectiveRequest);
            ok = JuicerCuda::ResourceManager::command_ensure_spatial_dir_cached_log_raw_stage(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                stageStats,
                outError);
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace._request.spatialDirDescriptorHash;
            std::string msg = "event=spatial_dir_cached_log_raw_stage_for_final_develop";
            msg += " descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
            msg += " scratch_source=";
            msg += overflow ? "overflow" : "retained";
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
                "stage_spatial_dir_cached_log_raw_for_final_develop",
                "CUDA spatial DIR cached log raw staging failed");
        }
        return ok;
    }

    bool Root::PreparedCudaFrame::release_spatial_dir_cached_log_raw_after_final_develop(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }
        if (!workspace._request.needSpatialDir) {
            return true;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return true;
        }

        JuicerCuda::SpatialDirCachedLogRawReleaseStats releaseStats{};
        const bool overflow = _state->scratchWorkspace.overflowActive;
        bool ok = false;
        if (overflow) {
            ok = _state->release_overflow_spatial_dir_cached_log_raw_after_final_develop(
                cudaStreamOpaque,
                releaseStats,
                outError);
        } else {
            _state->remember_stream(cudaStreamOpaque);
            ok = JuicerCuda::ResourceManager::command_release_spatial_dir_cached_log_raw_stage(
                _state->transaction,
                *_state->resources,
                cudaStreamOpaque,
                releaseStats,
                outError);
        }
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            const std::uint64_t descriptorHash = workspace._request.spatialDirDescriptorHash;
            std::string msg = "event=spatial_dir_cached_log_raw_release_after_final_develop";
            msg += " descriptor_hash=";
            msg += std::to_string(static_cast<unsigned long long>(descriptorHash));
            msg += " scratch_source=";
            msg += overflow ? "overflow" : "retained";
            msg += " ok=";
            msg += ok ? "1" : "0";
            msg += " pending_scratch_bytes_before=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.pendingScratchBytesBefore));
            msg += " cached_log_raw_retired_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.cachedLogRawRetiredBytes));
            msg += " reclaimed_bytes=";
            msg += std::to_string(static_cast<unsigned long long>(releaseStats.reclaimedBytes));
            if (!outError.empty()) {
                msg += " error=";
                msg += outError;
            }
            JTRACE("DIR_DESCRIPTOR", msg);
        }
#endif
        if (!ok) {
            _state->set_failure(
                "release_spatial_dir_cached_log_raw_after_final_develop",
                "CUDA spatial DIR cached log raw release failed");
        }
        return ok;
    }

    bool Root::PreparedCudaFrame::prepare_scanner_post_effects(
        const Scanner::ScannerPostEffectsDescriptor& descriptor,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!descriptor.active()) {
            return true;
        }
        if (descriptor.lensBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerLensBlurKernel,
                descriptor.lensBlurSigmaPx,
                cudaStreamOpaque,
                outError)) {
            return false;
        }
        if (descriptor.unsharpSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerUnsharpKernel,
                descriptor.unsharpSigmaPx,
                cudaStreamOpaque,
                outError)) {
            return false;
        }
        if (descriptor.glareActive && descriptor.glareBlurSigmaPx > 0.0f &&
            !build_gaussian_kernel_slot(
                _state->resources->scannerGlareKernel,
                descriptor.glareBlurSigmaPx,
                cudaStreamOpaque,
                outError)) {
            return false;
        }
        _state->scannerPostEffectsDescriptor = descriptor;
        return true;
    }

    bool Root::PreparedCudaFrame::build_print_illuminant_filter_curve(
        const WorkingState& workingState,
        const Print::Runtime& printRt,
        const Print::Params& printParams,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        if (!JuicerCuda::ResourceManager::command_ensure_print_illuminant_filtered(
                _state->transaction,
                *_state->resources,
                workingState,
                printRt,
                printParams,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_print_illuminant_filtered",
                "CUDA print illuminant upload failed");
            return false;
        }
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
                "scan_error_pending_readback",
                "CUDA scan error validation failed");
            return false;
        }
        if (previousScanErrorDetected) {
            JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
            _state->set_failure(
                "scan_error_previous_readback",
                "CUDA scan error validation failed",
                false);
            outError = "previous scan produced non-finite RGB";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        outScanErrorFlag = stage.deviceFlag;
        if (!outScanErrorFlag) {
            _state->set_failure(
                "scan_error_flag_missing",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag missing after allocation";
            return false;
        }

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const cudaError_t flagErr = cudaMemsetAsync(outScanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            _state->set_failure(
                "scan_error_flag_memset",
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
                "scan_error_flag_missing",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag missing after allocation";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        if (scanErrorFlag != stage.deviceFlag) {
            _state->set_failure(
                "scan_error_flag_mismatch",
                "CUDA scan error validation failed",
                false);
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
                    "scan_error_flag_readback",
                    "CUDA scan error validation failed");
                outError = "CUDA scan error flag readback failed";
                return false;
            }
            const cudaError_t evErr = cudaEventRecord(scanEvent, stream);
            if (evErr != cudaSuccess) {
                _state->set_failure(
                    "scan_error_event_record",
                    "CUDA scan error validation failed");
                outError = "CUDA scan error event record failed";
                return false;
            }
            stage.readbackPending = true;
        } else {
            static std::atomic<bool> sScanErrorReadbackUnavailableWarned{false};
            if (!sScanErrorReadbackUnavailableWarned.exchange(true)) {
                JTRACE("CUDA", "scan error host/event staging unavailable; skipping asynchronous scan-error readback validation");
            }
        }
        return true;
    }

    bool Root::PreparedCudaFrame::checkpoint_scratch_phase(
        const WorkspaceLeaseMarker& workspace,
        const char* stageTag,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        const char* failureStageTag = stageTag ? stageTag : "command_checkpoint_scratch_phase";
        if (!JuicerCuda::ResourceManager::command_checkpoint_scratch_phase(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                failureStageTag,
                outError)) {
            _state->set_failure(
                failureStageTag,
                "CUDA scratch phase checkpoint failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::checkpoint_large_scratch_transition(
        const WorkspaceLeaseMarker& workspace,
        bool usesSpatialDirFft,
        void* cudaStreamOpaque,
        const char* stageTag,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        _state->remember_stream(cudaStreamOpaque);
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        const char* failureStageTag =
            stageTag ? stageTag : "command_checkpoint_large_scratch_transition";
        if (!JuicerCuda::ResourceManager::command_checkpoint_large_scratch_transition(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                usesSpatialDirFft,
                cudaStreamOpaque,
                failureStageTag,
                outError)) {
            _state->set_failure(
                failureStageTag,
                "CUDA large scratch transition checkpoint failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::build_gaussian_kernel_slot(
        JuicerCuda::Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                _state->transaction,
                *_state->resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_gaussian_kernel",
                "CUDA gaussian kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::build_halation_kernel_slot(
        JuicerCuda::Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                _state->transaction,
                *_state->resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_halation_kernel",
                "CUDA halation kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::build_lens_blur_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->scannerLensBlurKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::build_unsharp_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->scannerUnsharpKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::build_glare_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->scannerGlareKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->grainBlurKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_mid_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->grainBlurKernelMid,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_coarse_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->grainBlurKernelCoarse,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_dye_kernel(
        int layer,
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (layer < 0 || layer >= 3 || channel < 0 || channel >= 3) {
            outError = "invalid grain dye kernel slot";
            _state->set_failure(
                "prepare_grain_dye_kernel",
                "CUDA grain dye-cloud kernel upload failed");
            return false;
        }
        return build_gaussian_kernel_slot(
            _state->resources->grainDyeKernel[layer][channel],
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::build_halation_kernel(
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (channel < 0 || channel >= 3) {
            outError = "invalid halation kernel slot";
            _state->set_failure(
                "build_halation_kernel",
                "CUDA halation kernel upload failed");
            return false;
        }
        return build_halation_kernel_slot(
            _state->resources->halationKernel[channel],
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::build_halation_scatter_kernel(
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (channel < 0 || channel >= 3) {
            outError = "invalid halation scatter kernel slot";
            _state->set_failure(
                "build_halation_scatter_kernel",
                "CUDA halation scatter kernel upload failed");
            return false;
        }
        return build_halation_kernel_slot(
            _state->resources->halationScatterKernel[channel],
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::launch_base_pipeline_graph(
        JuicerCuda::PipelineRunParams& run,
        int renderModeKey,
        void* cudaStreamOpaque,
        int& outCudaErrorCode,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_launch_base_pipeline_graph(
                _state->transaction,
                run,
                renderModeKey,
                cudaStreamOpaque,
                outCudaErrorCode,
                outError)) {
            _state->set_failure(
                "command_launch_base_pipeline_graph",
                "CUDA base graph launch command failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::validate_density_primitives(
        const WorkingState& workingState,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        return JuicerCuda::validate_density_primitives(
            *_state->resources,
            workingState,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::validate_print_primitives(
        const WorkingState& workingState,
        const Print::Runtime& printRt,
        const Print::Params& printParams,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        return JuicerCuda::validate_print_primitives(
            *_state->resources,
            workingState,
            printRt,
            printParams,
            midgrayFactor,
            cudaStreamOpaque,
            outError);
    }

    Root::PreparedCudaFrame::GrainStaticAssets Root::PreparedCudaFrame::grain_static_assets() const noexcept {
        GrainStaticAssets assets{};
        if (!_state || !_state->grainStaticResources || !_state->transaction.active || _state->transaction.committed) {
            return assets;
        }

        const JuicerCuda::Resources& resources = *_state->grainStaticResources;
        if (resources.stbnData && resources.stbnWidth > 0 && resources.stbnHeight > 0 && resources.stbnFrames > 0) {
            assets.stbn = resources.stbnData;
            assets.stbnWidth = resources.stbnWidth;
            assets.stbnHeight = resources.stbnHeight;
            assets.stbnFrames = resources.stbnFrames;
        }
        if (resources.wangTilesData && resources.wangLutData &&
            resources.wangWidth > 0 && resources.wangHeight > 0 &&
            resources.wangCount > 0 && resources.wangColors > 0) {
            assets.wangTiles = resources.wangTilesData;
            assets.wangLut = resources.wangLutData;
            assets.wangWidth = resources.wangWidth;
            assets.wangHeight = resources.wangHeight;
            assets.wangCount = resources.wangCount;
            assets.wangColors = resources.wangColors;
        }
        return assets;
    }

    Root::PreparedCudaFrame::DurableBundleView Root::PreparedCudaFrame::durable_bundle() const noexcept {
        DurableBundleView bundle{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return bundle;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        bundle.film.densB = &resources.densB;
        bundle.film.densG = &resources.densG;
        bundle.film.densR = &resources.densR;
        bundle.film.dirDensB = &resources.dirDensB;
        bundle.film.dirDensG = &resources.dirDensG;
        bundle.film.dirDensR = &resources.dirDensR;
        bundle.film.sensB = &resources.sensB;
        bundle.film.sensG = &resources.sensG;
        bundle.film.sensR = &resources.sensR;
        bundle.film.hasDensityCurvesLayers = resources.hasDensityCurvesLayers;
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                bundle.film.densityCurvesLayers[layer][ch] = resources.densityCurvesLayers[layer][ch];
            }
        }
        bundle.film.tablesAx = resources.tablesAx;
        bundle.film.tablesAy = resources.tablesAy;
        bundle.film.tablesAz = resources.tablesAz;
        bundle.film.tablesIllum = resources.tablesIllum;
        bundle.film.tablesK = resources.tablesK;
        for (int i = 0; i < 9; ++i) {
            bundle.film.spdSInv[i] = resources.spdSInv[i];
        }
        bundle.film.hanatosLut = resources.hanatosLut;
        bundle.film.hanatosN = resources.hanatosN;
        bundle.film.hanatosLutIntegrated = resources.hanatosLutIntegrated;
        bundle.film.hanatosNIntegrated = resources.hanatosNIntegrated;
        bundle.film.mallettBasis = resources.mallettBasis;
        bundle.film.mallettBasisK = resources.mallettBasisK;

        bundle.scan.negativeMedium = &resources.scanNegative;
        bundle.scan.printMedium = &resources.scanPrint;
        bundle.scan.negativeLut = &resources.scanNegativeLut;
        bundle.scan.printLut = &resources.scanPrintLut;

        bundle.print.printIllumFiltered = resources.printIllumFiltered;
        bundle.print.printIllumK = resources.printIllumK;
        bundle.print.printSensC = &resources.printSensC;
        bundle.print.printSensM = &resources.printSensM;
        bundle.print.printSensY = &resources.printSensY;
        bundle.print.printDcC = &resources.printDcC;
        bundle.print.printDcM = &resources.printDcM;
        bundle.print.printDcY = &resources.printDcY;
        bundle.print.printGammaC = resources.printGammaC;
        bundle.print.printGammaM = resources.printGammaM;
        bundle.print.printGammaY = resources.printGammaY;
        for (int i = 0; i < 3; ++i) {
            bundle.print.printPreflashRaw[i] = resources.printPreflashRaw[i];
        }
        return bundle;
    }

    Root::PreparedCudaFrame::OpticsKernelView Root::PreparedCudaFrame::optics_kernels() const noexcept {
        OpticsKernelView kernels{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return kernels;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        kernels.spatialDir = {
            resources.spatialDirKernels[0].weights,
            resources.spatialDirKernels[0].radius,
            resources.spatialDirKernels[0].sigma};
        kernels.scannerLensBlur = {resources.scannerLensBlurKernel.weights, resources.scannerLensBlurKernel.radius};
        kernels.scannerUnsharp = {resources.scannerUnsharpKernel.weights, resources.scannerUnsharpKernel.radius};
        kernels.scannerGlare = {resources.scannerGlareKernel.weights, resources.scannerGlareKernel.radius};
        kernels.grainBlur = {resources.grainBlurKernel.weights, resources.grainBlurKernel.radius};
        kernels.grainBlurMid = {resources.grainBlurKernelMid.weights, resources.grainBlurKernelMid.radius};
        kernels.grainBlurCoarse = {resources.grainBlurKernelCoarse.weights, resources.grainBlurKernelCoarse.radius};
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                kernels.grainDye[layer][ch] = {
                    resources.grainDyeKernel[layer][ch].weights,
                    resources.grainDyeKernel[layer][ch].radius};
            }
        }
        for (int i = 0; i < 3; ++i) {
            kernels.halation[i] = {resources.halationKernel[i].weights, resources.halationKernel[i].radius};
            kernels.halationScatter[i] = {
                resources.halationScatterKernel[i].weights,
                resources.halationScatterKernel[i].radius};
        }
        return kernels;
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
            view.deviceState.exposureScale &&
            view.deviceState.autoEV &&
            view.deviceState.valid;
        return view;
    }

    Root::PreparedCudaFrame::UploadTraceView Root::PreparedCudaFrame::upload_trace_view() const noexcept {
        UploadTraceView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return view;
        }

        std::lock_guard<std::mutex> lock(_state->resources->m);
        view.uploadedBuildCounter = _state->resources->uploadedBuildCounter;
        view.printIllumBuildCounter = _state->resources->printIllumBuildCounter;
        view.printIllumCoreHash = _state->resources->printIllumCoreHash;
        view.printIllumNeutralFilterHash = _state->resources->printIllumNeutralFilterHash;
        view.printIllumYShiftSteps = _state->resources->printIllumYShiftSteps;
        view.printIllumMShiftSteps = _state->resources->printIllumMShiftSteps;
        view.printIllumCShiftSteps = _state->resources->printIllumCShiftSteps;
        view.printPreflashValid = _state->resources->printPreflashValid;
        view.printPreflashKeyHash = _state->resources->printPreflashKeyHash;
        view.directUploadCounter = _state->resources->directUploadCounter;
        view.finalSensitivityHash = _state->resources->directFinalSensitivityHash;
        view.densityCurvesHash = _state->resources->directDensityCurvesHash;
        view.densityBoundsHash = _state->resources->directDensityBoundsHash;
        view.scannerDescriptorHash = _state->resources->directScannerDescriptorHash;
        view.printPreparationCounter = _state->resources->printPreparationCounter;
        view.printPreparationDescriptorHash =
            _state->resources->printPreparationDescriptorHash;
        view.active = true;
        return view;
    }

    Root::PreparedCudaFrame::DirectPreparedView Root::PreparedCudaFrame::direct_resources() const noexcept {
        DirectPreparedView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed ||
            !_state->focusedFilmRawConfig || !_state->focusedScannerColor) {
            return view;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        view.film.finalSensB = {resources.sensB.x, resources.sensB.y, resources.sensB.n, resources.sensB.domainBegin, resources.sensB.domainEnd};
        view.film.finalSensG = {resources.sensG.x, resources.sensG.y, resources.sensG.n, resources.sensG.domainBegin, resources.sensG.domainEnd};
        view.film.finalSensR = {resources.sensR.x, resources.sensR.y, resources.sensR.n, resources.sensR.domainBegin, resources.sensR.domainEnd};
        view.film.normalizedDensB = {resources.densB.x, resources.densB.y, resources.densB.n, resources.densB.domainBegin, resources.densB.domainEnd};
        view.film.normalizedDensG = {resources.densG.x, resources.densG.y, resources.densG.n, resources.densG.domainBegin, resources.densG.domainEnd};
        view.film.normalizedDensR = {resources.densR.x, resources.densR.y, resources.densR.n, resources.densR.domainBegin, resources.densR.domainEnd};
        if (resources.directDirHash != 0) {
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
        if (resources.hasDensityCurvesLayers && resources.directDensityLayersHash != 0) {
            view.film.hasDensityCurvesLayers = true;
            view.film.densityCurvesLayersHash = resources.directDensityLayersHash;
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    view.film.densityCurvesLayers[layer][ch] = resources.densityCurvesLayers[layer][ch];
                }
            }
        }
        std::copy_n(_state->focusedFilmRawConfig->inputRGBToXYZ.m, 9, view.film.inputRGBToXYZ);
        std::copy_n(_state->focusedFilmRawConfig->inputXYZAdapt.m, 9, view.film.inputXYZAdapt);
        view.film.applyInputChromaticAdapt = _state->focusedFilmRawConfig->applyInputChromaticAdapt ? 1 : 0;
        std::copy_n(resources.refIllumWhiteXYZ, 3, view.film.refIllumWhiteXYZ);
        view.film.finalSensitivityHash = resources.directFinalSensitivityHash;
        view.film.normalizedDensityCurvesHash = resources.directDensityCurvesHash;
        view.film.dirCouplersHash = resources.directDirHash;
        view.scanMedium = &resources.scanNegative;
        view.scanLut = &resources.scanNegativeLut;
        view.scannerColor = _state->focusedScannerColor;
        view.densityBoundsHash = resources.directDensityBoundsHash;
        view.scannerDescriptorHash = resources.directScannerDescriptorHash;
        view.selectedMethod = resources.directSelectedMethod;
        view.active = view.scanMedium && view.scanLut->canonical_ready() &&
                      view.densityBoundsHash != 0 && view.scannerDescriptorHash != 0;
        return view;
    }

    Root::PreparedCudaFrame::PrintRoutePreparedView Root::PreparedCudaFrame::print_route_resources() const noexcept {
        PrintRoutePreparedView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed ||
            !_state->focusedFilmRawConfig || !_state->focusedScannerColor) {
            return view;
        }
        const JuicerCuda::Resources& resources = *_state->resources;
        view.film.finalSensB = {resources.sensB.x, resources.sensB.y, resources.sensB.n, resources.sensB.domainBegin, resources.sensB.domainEnd};
        view.film.finalSensG = {resources.sensG.x, resources.sensG.y, resources.sensG.n, resources.sensG.domainBegin, resources.sensG.domainEnd};
        view.film.finalSensR = {resources.sensR.x, resources.sensR.y, resources.sensR.n, resources.sensR.domainBegin, resources.sensR.domainEnd};
        view.film.normalizedDensB = {resources.densB.x, resources.densB.y, resources.densB.n, resources.densB.domainBegin, resources.densB.domainEnd};
        view.film.normalizedDensG = {resources.densG.x, resources.densG.y, resources.densG.n, resources.densG.domainBegin, resources.densG.domainEnd};
        view.film.normalizedDensR = {resources.densR.x, resources.densR.y, resources.densR.n, resources.densR.domainBegin, resources.densR.domainEnd};
        if (resources.directDirHash != 0) {
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
        if (resources.hasDensityCurvesLayers && resources.directDensityLayersHash != 0) {
            view.film.hasDensityCurvesLayers = true;
            view.film.densityCurvesLayersHash = resources.directDensityLayersHash;
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    view.film.densityCurvesLayers[layer][ch] = resources.densityCurvesLayers[layer][ch];
                }
            }
        }
        std::copy_n(_state->focusedFilmRawConfig->inputRGBToXYZ.m, 9, view.film.inputRGBToXYZ);
        std::copy_n(_state->focusedFilmRawConfig->inputXYZAdapt.m, 9, view.film.inputXYZAdapt);
        view.film.applyInputChromaticAdapt = _state->focusedFilmRawConfig->applyInputChromaticAdapt ? 1 : 0;
        std::copy_n(resources.refIllumWhiteXYZ, 3, view.film.refIllumWhiteXYZ);
        view.film.finalSensitivityHash = resources.directFinalSensitivityHash;
        view.film.normalizedDensityCurvesHash = resources.directDensityCurvesHash;
        view.film.dirCouplersHash = resources.directDirHash;
        view.scanMedium = &resources.scanPrint;
        view.scanLut = &resources.scanPrintLut;
        view.scannerColor = _state->focusedScannerColor;
        view.densityBoundsHash = resources.directDensityBoundsHash;
        view.scannerDescriptorHash = resources.directScannerDescriptorHash;
        view.selectedMethod = resources.directSelectedMethod;
        view.active = view.scanLut->canonical_ready() &&
                      view.densityBoundsHash != 0 && view.scannerDescriptorHash != 0;
        return view;
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
        view.factorMidgray = resources.printBalanceFactorMidgray;
        view.factorMidgrayComp = resources.printBalanceFactorMidgrayComp;
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
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch =
            _state->scratchWorkspace.overflowActive
                ? _state->scratchWorkspace.spatialDir
                : _state->resources->spatialDirScratch;
        const WorkspaceRequest& effectiveRequest =
            (_state->scratchWorkspace.request.needSpatialDir &&
             _state->scratchWorkspace.request.spatialDirDescriptorHash ==
                 workspace._request.spatialDirDescriptorHash)
                ? _state->scratchWorkspace.request
                : workspace._request;
        view.rawCorrectionY = scratch.rawCorrectionY;
        view.rawCorrectionM = scratch.rawCorrectionM;
        view.rawCorrectionC = scratch.rawCorrectionC;
        view.filteredCorrectionY = scratch.filteredCorrectionY;
        view.filteredCorrectionM = scratch.filteredCorrectionM;
        view.filteredCorrectionC = scratch.filteredCorrectionC;
        view.filterTemp = scratch.filterTemp;
        view.filterTempM = scratch.filterTempM;
        view.filterTempC = scratch.filterTempC;
        view.iirForwardTemp = scratch.iirForwardTemp;
        view.iirForwardTempM = scratch.iirForwardTempM;
        view.iirForwardTempC = scratch.iirForwardTempC;
        view.logRawB = scratch.logRawB;
        view.logRawG = scratch.logRawG;
        view.logRawR = scratch.logRawR;
        view.descriptorHash = effectiveRequest.spatialDirDescriptorHash;
        view.scratchTier = effectiveRequest.spatialDirScratchTier;
        view.planeRoles = effectiveRequest.spatialDirPlaneRoles;
        view.targetScratchTier = effectiveRequest.spatialDirTargetScratchTier;
        view.targetPlaneRoles = effectiveRequest.spatialDirTargetPlaneRoles;
        view.overflow = _state->scratchWorkspace.overflowActive;
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
        if (_state->spatialDirDescriptor.approximation ==
            Spektrafilm::DirApproximationMarker::AcceptedFftReplicatePadSmooth) {
            const auto& fft = _state->resources->spatialDirFft;
            view.fftForwardPlan = fft.forwardPlan;
            view.fftInversePlan = fft.inversePlan;
            view.fftRealBuffer = fft.realBuffer;
            view.fftSpectrum = fft.spectrum;
            view.fftTransfer = fft.transfer;
            view.fftWidth = fft.width;
            view.fftHeight = fft.height;
            view.fftPadPixels = fft.padPixels;
            view.fftComplexWidth = fft.complexWidth;
            view.fftRealBufferBytes = fft.realBufferBytes;
            view.fftSpectrumBytes = fft.spectrumBytes;
            view.fftTransferBytes = fft.transferBytes;
            view.fftWorkAreaBytes = fft.workAreaBytes;
            view.fftForwardWorkBytes = fft.forwardWorkBytes;
            view.fftInverseWorkBytes = fft.inverseWorkBytes;
            view.fftSetupMs = fft.lastSetupMs;
            view.fftSetupCreated = fft.lastSetupCreated;
            view.fftActive =
                fft.descriptorHash == descriptorHash &&
                fft.forwardPlan != 0 &&
                fft.inversePlan != 0 &&
                fft.realBuffer &&
                fft.spectrum &&
                fft.transfer &&
                fft.width == _state->spatialDirDescriptor.fftWidth &&
                fft.height == _state->spatialDirDescriptor.fftHeight &&
                fft.padPixels == _state->spatialDirDescriptor.fftPadPixels &&
                fft.complexWidth == _state->spatialDirDescriptor.fftComplexWidth;
            view.active = view.fftActive;
            return view;
        }
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
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->scratchWorkspace.overflowActive
                ? _state->scratchWorkspace.optics
                : _state->resources->scannerScratch;
        if (activeRequest.aliasScannerRgbFromSpatialDirFiltered) {
            const JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir =
                _state->scratchWorkspace.overflowActive
                    ? _state->scratchWorkspace.spatialDir
                    : _state->resources->spatialDirScratch;
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
        view.grainTmpMid = scratch.grainTmpMid;
        view.grainTmpCoarse = scratch.grainTmpCoarse;
        view.gateMask = scratch.gateMask;
        view.gateMaskWidth = scratch.gateWidth;
        view.gateMaskHeight = scratch.gateHeight;
        view.gateMaskHash = scratch.gateMaskHash;
        view.active = view.rgbR && view.rgbG && view.rgbB && view.tmp;
        view.hasGateMask = view.gateMask && view.gateMaskWidth > 0 && view.gateMaskHeight > 0;
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
        const bool lensReady =
            descriptor.lensBlurSigmaPx <= 0.0f ||
            (view.lensBlur.weights && view.lensBlur.radius > 0);
        const bool unsharpReady =
            descriptor.unsharpSigmaPx <= 0.0f ||
            (view.unsharp.weights && view.unsharp.radius > 0);
        const bool glareReady =
            !descriptor.glareActive ||
            descriptor.glareBlurSigmaPx <= 0.0f ||
            (view.glare.weights && view.glare.radius > 0);
        view.active = lensReady && unsharpReady && glareReady;
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

    void Root::PreparedCudaFrame::mark_gate_mask_built(std::uint64_t gateMaskHash) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        if (_state->scratchWorkspace.overflowActive) {
            _state->scratchWorkspace.optics.gateMaskHash = gateMaskHash;
            return;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive) {
            return;
        }
        _state->resources->scannerScratch.gateMaskHash = gateMaskHash;
    }

    void Root::PreparedCudaFrame::record_use(void* cudaStreamOpaque) noexcept {
        try {
            if (!_state ||
                !_state->resources ||
                !_state->transaction.active ||
                _state->transaction.committed ||
                _state->frameUseEventSubmitted) {
                return;
            }
            std::string ignoredError;
            (void)_state->submit_frame_use_event(cudaStreamOpaque, ignoredError);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    const char* Root::PreparedCudaFrame::failure_stage_tag() const noexcept {
        return _state ? _state->failureStageTag : "prepare_frame";
    }

    const char* Root::PreparedCudaFrame::failure_prefix() const noexcept {
        return _state ? _state->failurePrefix : "CUDA prepared frame failed";
    }

    bool Root::PreparedCudaFrame::failure_marks_context_loss() const noexcept {
        return _state ? _state->failureMarksContextLoss : true;
    }
#endif

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

    const std::string& Root::data_dir() const noexcept {
        return _dataDir;
    }

    void Root::ensure_bootstrap() {
        resume_frame_preparation();
        std::call_once(_bootstrapOnce, [this]() {
            load_spectral_globals(_dataDir);
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
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return JuicerCuda::ResourceManager::command_retire_context_idle(key, outError);
#else
        (void)deviceId;
        (void)contextOpaque;
        outError.clear();
        return true;
#endif
    }

    bool Root::retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return JuicerCuda::ResourceManager::command_retire_context_reset(key, outError);
#else
        (void)deviceId;
        (void)contextOpaque;
        outError.clear();
        return true;
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    void Root::CudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
        JuicerCuda::destroy(resources);
    }

    bool Root::resolve_context_cuda_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        ContextCudaResourceMap& contextMap,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError.clear();
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque) {
            outError = "invalid CUDA context key";
            return false;
        }
        if (contextEpoch == 0) {
            outError = "invalid CUDA context epoch";
            return false;
        }

        const ContextCudaResourceKey resourceKey{deviceContextKey, contextEpoch};
        std::vector<CudaResourceOwner> retiredOwners;
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaResourceOwner& resourceOwner = contextMap[resourceKey];
            if (!resourceOwner) {
                CudaResourceOwner resources(JuicerCuda::create(), CudaResourcesDeleter{});
                if (!resources) {
                    JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                    outError = "failed to allocate CUDA resources";
                    return false;
                }
                if (resources->deviceId < 0) {
                    resources->deviceId = deviceContextKey.deviceId;
                }
                if (!resources->ownerContextOpaque) {
                    resources->ownerContextOpaque = deviceContextKey.contextOpaque;
                }
                if (resources->deviceId != deviceContextKey.deviceId ||
                    resources->ownerContextOpaque != deviceContextKey.contextOpaque) {
                    outError = "CUDA resources resolved for a different context";
                    return false;
                }
                resourceOwner = std::move(resources);
            }

            outResourceOwner = resourceOwner;
            for (auto it = contextMap.begin(); it != contextMap.end();) {
                if (!(it->first.deviceContextKey == deviceContextKey) ||
                    it->first.contextEpoch >= contextEpoch) {
                    ++it;
                    continue;
                }
                if (it->second) {
                    retiredOwners.emplace_back(std::move(it->second));
                }
                it = contextMap.erase(it);
            }
        }

        outResources = outResourceOwner.get();
        if (!outResources) {
            outError = "CUDA resources missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::resolve_cuda_frame_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            _cudaResourcesByContext,
            outResourceOwner,
            outResources,
            outError);
    }

    bool Root::resolve_cuda_grain_static_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            _cudaGrainStaticByContext,
            outResourceOwner,
            outResources,
            outError);
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const WorkingState& workingState,
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
        frame._state->set_failure("prepare_frame", "CUDA prepared frame failed");
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure("begin_submission", "begin_submission failed");
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            frame._state->set_failure("resolve_cuda_resources", "CUDA resource acquisition failed");
            frame.abort("prepared_frame_resource_acquire_failed");
            return frame;
        }
        if (!acquire_submission_plan(frame._state->transaction, outError)) {
            frame._state->set_failure("acquire_plan", "acquire_plan failed");
            frame.abort("prepared_frame_acquire_failed");
            return frame;
        }
        if (!JuicerCuda::ResourceManager::command_ensure_uploaded(
                frame._state->transaction,
                *frame._state->resources,
                workingState,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure("command_ensure_uploaded", "CUDA WorkingState upload failed");
            frame.abort("prepared_frame_upload_failed");
            return frame;
        }
        if (!frame._state->allocate_scan_error_stage(outError)) {
            frame._state->set_failure(
                "allocate_scan_error_stage",
                "CUDA scan error staging allocation failed");
            frame.abort("prepared_frame_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(
                autoExposureBufferRequest.descriptor,
                outError)) {
            frame._state->set_failure(
                "allocate_auto_exposure_workspace",
                "CUDA auto-exposure workspace allocation failed");
            frame.abort("prepared_frame_auto_exposure_failed");
            return frame;
        }
        return frame;
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const PrintCudaPreparationRequest& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>();
        } catch (...) {
            outError = "failed to allocate Phase 4B print CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->remember_stream(cudaStreamOpaque);
        frame._state->set_failure(
            "prepare_cuda_frame_print_phase4B",
            "CUDA Phase 4B print prepared frame failed",
            false);
        if (!request.recipe || !request.exposureTables || !request.spdSInv ||
            !request.filmRawConfig || !request.scannerTables || !request.scannerColor ||
            !request.scannerLutDescriptor) {
            outError = "MissingRequiredResource phase=4B field=print_recipe";
            return frame;
        }
        if (!JuicerCuda::build_print_resource_descriptors(
                *request.recipe,
                frame._state->printDescriptors,
                outError)) {
            return frame;
        }
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure(
                "begin_submission_print_phase4B",
                "Phase 4B print begin_submission failed");
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            frame._state->set_failure(
                "resolve_cuda_print_resources_phase4B",
                "Phase 4B print CUDA resource acquisition failed");
            frame.abort("print_phase4B_resource_acquire_failed");
            return frame;
        }
        if (!acquire_submission_plan(frame._state->transaction, outError)) {
            frame._state->set_failure(
                "acquire_print_plan_phase4B",
                "Phase 4B print acquire_plan failed");
            frame.abort("print_phase4B_acquire_failed");
            return frame;
        }
        JuicerCuda::PrintRouteResourcePreparation focusedPreparation{};
        focusedPreparation.recipe = request.recipe;
        focusedPreparation.exposureTables = request.exposureTables;
        focusedPreparation.spdSInv = request.spdSInv;
        focusedPreparation.filmRawConfig = request.filmRawConfig;
        focusedPreparation.scannerTables = request.scannerTables;
        focusedPreparation.scannerColor = request.scannerColor;
        focusedPreparation.scannerLutDescriptor = request.scannerLutDescriptor;
        if (!JuicerCuda::prepare_print_route_resources(
                *frame._state->resources,
                focusedPreparation,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure(
                "prepare_print_route_resources_phase4C",
                "Phase 4C focused print-route resource preparation failed",
                false);
            frame.abort("print_phase4C_focused_preparation_failed");
            return frame;
        }
        JuicerCuda::PrintResourcePreparation preparation{};
        preparation.recipe = request.recipe;
        preparation.assets = &_assets;
        if (!JuicerCuda::prepare_print_resources(
                *frame._state->resources,
                preparation,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure(
                "prepare_print_resources_phase4B",
                "Phase 4B print resource preparation failed",
                false);
            frame.abort("print_phase4B_preparation_failed");
            return frame;
        }
        frame._state->printRecipe = &request.recipe->print;
        frame._state->focusedFilmRawConfig = request.filmRawConfig;
        frame._state->focusedScannerColor = request.scannerColor;
        if (request.scannerPostEffects) {
            if (!frame.prepare_scanner_post_effects(
                    *request.scannerPostEffects,
                    cudaStreamOpaque,
                    outError)) {
                frame._state->set_failure(
                    "prepare_print_scanner_post_effects_phase8",
                    "CUDA print scanner post-effect preparation failed",
                    false);
                frame.abort("print_phase8_scanner_post_effect_preparation_failed");
                return frame;
            }
        }
        if (!frame._state->allocate_scan_error_stage(outError)) {
            frame._state->set_failure(
                "allocate_print_scan_error_stage_phase4C",
                "CUDA print scan error staging allocation failed");
            frame.abort("print_phase4C_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(
                autoExposureBufferRequest.descriptor,
                outError)) {
            frame._state->set_failure(
                "allocate_print_auto_exposure_workspace_phase4C",
                "CUDA print auto-exposure workspace allocation failed");
            frame.abort("print_phase4C_auto_exposure_failed");
            return frame;
        }
        return frame;
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const DirectCudaPreparationRequest& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>();
        } catch (...) {
            outError = "failed to allocate direct CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->remember_stream(cudaStreamOpaque);
        frame._state->set_failure("prepare_cuda_frame_direct", "CUDA direct prepared frame failed");
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure("begin_submission_direct", "direct begin_submission failed");
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            frame._state->set_failure("resolve_cuda_direct_resources", "CUDA direct resource acquisition failed");
            frame.abort("direct_prepared_frame_resource_acquire_failed");
            return frame;
        }
        if (!acquire_submission_plan(frame._state->transaction, outError)) {
            frame._state->set_failure("acquire_direct_plan", "direct acquire_plan failed");
            frame.abort("direct_prepared_frame_acquire_failed");
            return frame;
        }

        JuicerCuda::DirectResourcePreparation directRequest{};
        directRequest.recipe = request.recipe;
        directRequest.exposureTables = request.exposureTables;
        directRequest.spdSInv = request.spdSInv;
        directRequest.filmRawConfig = request.filmRawConfig;
        directRequest.scannerTables = request.scannerTables;
        directRequest.scannerColor = request.scannerColor;
        directRequest.scannerLutDescriptor = request.scannerLutDescriptor;
        if (!JuicerCuda::prepare_direct_resources(
                *frame._state->resources,
                directRequest,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure("prepare_direct_resources", "CUDA direct resource preparation failed");
            frame.abort("direct_prepared_frame_upload_failed");
            return frame;
        }
        frame._state->focusedFilmRawConfig = request.filmRawConfig;
        frame._state->focusedScannerColor = request.scannerColor;
        if (request.scannerPostEffects) {
            if (!frame.prepare_scanner_post_effects(
                    *request.scannerPostEffects,
                    cudaStreamOpaque,
                    outError)) {
                frame._state->set_failure(
                    "prepare_direct_scanner_post_effects_phase8",
                    "CUDA direct scanner post-effect preparation failed",
                    false);
                frame.abort("direct_phase8_scanner_post_effect_preparation_failed");
                return frame;
            }
        }
        if (!frame._state->allocate_scan_error_stage(outError)) {
            frame._state->set_failure("allocate_direct_scan_error_stage", "CUDA direct scan error staging allocation failed");
            frame.abort("direct_prepared_frame_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(autoExposureBufferRequest.descriptor, outError)) {
            frame._state->set_failure("allocate_direct_auto_exposure_workspace", "CUDA direct auto-exposure workspace allocation failed");
            frame.abort("direct_prepared_frame_auto_exposure_failed");
            return frame;
        }
        return frame;
    }

    bool Root::begin_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& outError) {
        return JuicerCuda::ResourceManager::begin_submission(transaction, snapshot, outError);
    }

    bool Root::acquire_submission_plan(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        std::string& outError) {
        return JuicerCuda::ResourceManager::acquire_plan(transaction, outError);
    }

    bool Root::commit_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        void* cudaStreamOpaque,
        std::string& outError) {
        return JuicerCuda::ResourceManager::commit_submission(transaction, cudaStreamOpaque, outError);
    }

    bool Root::shed_post_frame_scratch(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        JuicerCuda::Resources& resources,
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        const char* commandName,
        std::string& outError) {
        return JuicerCuda::ResourceManager::command_shed_post_frame_scratch(
            transaction,
            resources,
            scratchRequest,
            cudaStreamOpaque,
            commandName,
            outError);
    }

    void Root::rollback_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const char* reason) noexcept {
        JuicerCuda::ResourceManager::rollback_submission(transaction, reason);
    }
#endif

    bool Root::retire_known_contexts(std::string& outError) noexcept {
        outError.clear();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        try {
            return JuicerCuda::ResourceManager::command_retire_all_contexts_idle(outError);
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "registry-wide context retire threw";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
#else
        return true;
#endif
    }

    void Root::release_cuda_context_resource_owners() noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        try {
            ContextCudaResourceMap frameResources;
            ContextCudaResourceMap grainStaticResources;
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            frameResources.swap(_cudaResourcesByContext);
            grainStaticResources.swap(_cudaGrainStaticByContext);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
#endif
    }

    void Root::release_cuda_host_asset_caches() noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::purge_host_asset_caches_if_registry_idle("process_shutdown");
#endif
    }

    void Root::release_process_host_services() noexcept {
        release_cuda_host_asset_caches();
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
