#include "Cuda/Resources/ResourceInternals.h"

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "SpectralData.h"
#include "GamutCompression.h"
#include "Scanner.h"
#include "JuicerDirBoundary.h"
#include "juicer_host_preparation.h"

#include "Logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <numeric>
#include <atomic>
#include <span>
#include <sstream>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace JuicerCuda {

    static void clear_gaussian_kernel(
        Resources::DeviceGaussianKernel& kernel) noexcept {
        kernel = Resources::DeviceGaussianKernel{};
    }

    void clear_gaussian_kernel_views(Resources& resources) noexcept {
        clear_gaussian_kernel(resources.scannerLensBlurKernel);
        clear_gaussian_kernel(resources.scannerUnsharpKernel);
        clear_gaussian_kernel(resources.scannerGlareKernel);
        clear_gaussian_kernel(resources.grainBlurKernel);
        clear_gaussian_kernel(resources.grainBlurKernelMid);
        clear_gaussian_kernel(resources.grainBlurKernelCoarse);
        for (auto& kernels : resources.grainDyeKernel) {
            for (auto& kernel : kernels) {
                clear_gaussian_kernel(kernel);
            }
        }
        for (auto& kernel : resources.spatialDirKernels) {
            clear_gaussian_kernel(kernel);
        }
    }

    void free_gaussian_cache(Resources& resources) noexcept {
        for (auto& entry : resources.gaussianKernelCache) {
            free_owned_device_noexcept(resources, entry.second);
        }
        resources.gaussianKernelCache.clear();
    }

    static bool ensure_cached_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        int radius,
        float sigma,
        const char* label,
        Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError = {};
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        if (!sigmaOk || radius <= 0) {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            clear_gaussian_kernel(kernel);
            return true;
        }

        std::uint32_t sigmaBits = 0;
        std::memcpy(&sigmaBits, &sigma, sizeof(sigmaBits));
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(radius)) << 32u) |
            static_cast<std::uint64_t>(sigmaBits);
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            const auto cached = resources.gaussianKernelCache.find(key);
            if (cached != resources.gaussianKernelCache.end()) {
                kernel = Resources::DeviceGaussianKernel{
                    cached->second,
                    radius,
                    sigma};
                return true;
            }
        }

        FjGaussianKernelView view{};
        const FjHostPreparation emptyPreparation{};
        const auto& preparation = hostPreparation ? *hostPreparation : emptyPreparation;
        if (!prepare_gaussian_kernel(preparation, {radius, sigma}, view, outError))
            return false;
        const std::span<const float> weights{view.weights.data, view.weights.count};
        const std::size_t bytes = weights.size_bytes();
        void* candidateRaw = nullptr;
        if (!allocate_owned_device(
                resources,
                &candidateRaw,
                bytes,
                label ? label : "gaussian kernel",
                outError)) {
            return false;
        }
        auto* candidateWeights = static_cast<float*>(candidateRaw);
        const cudaError_t copyError = cudaMemcpy(
            candidateWeights,
            weights.data(),
            bytes,
            cudaMemcpyHostToDevice);
        if (copyError != cudaSuccess) {
            outError.status = runtime_failure_status(copyError);
            outError.diagnostic = std::string("cudaMemcpy(") +
                                  (label ? label : "gaussian kernel") + ") failed: " +
                                  (cudaGetErrorString(copyError)
                                       ? cudaGetErrorString(copyError)
                                       : "(unknown)");
            free_owned_device_noexcept(resources, candidateWeights);
            return false;
        }

        bool cacheOwnsCandidate = false;
        bool success = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (validate_resource_owner_locked(resources, outError)) {
                reap_retire_queue_locked(resources);
                try {
                    const auto inserted = resources.gaussianKernelCache.emplace(
                        key,
                        candidateWeights);
                    cacheOwnsCandidate = inserted.second;
                    kernel = Resources::DeviceGaussianKernel{
                        inserted.first->second,
                        radius,
                        sigma};
                    success = true;
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                    outError.diagnostic = "Gaussian cache bookkeeping failed";
                }
            }
        }
        if (!cacheOwnsCandidate) {
            free_owned_device_noexcept(resources, candidateWeights);
        }
        return success;
    }

    bool ensure_spatial_dir_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        int radius,
        float sigma,
        Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        constexpr int kMaxRadius = 2048;
        if (!std::isfinite(sigma) || sigma <= 0.0f || radius <= 0 || radius > kMaxRadius) {
            outError.diagnostic = "spatial DIR kernel radius exceeds prepared-frame limit";
            return false;
        }
        return ensure_cached_gaussian_kernel(
            resources,
            kernel,
            radius,
            sigma,
            "spatial DIR kernel",
            outError,
            hostPreparation);
    }

    bool ensure_spatial_dir_boundary(
        Resources& resources,
        Resources::DeviceSpatialDirBoundary& boundary,
        const Spektrafilm::DirGaussianComponentPlan& component,
        int width,
        int height,
        void* cudaStreamOpaque,
        Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError = {};
        if (component.referenceOperator !=
                Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect ||
            !(component.weight > 0.0f) || component.boundaryIdentity == 0 || width <= 0 || height <= 0) {
            outError.diagnostic = "spatial DIR reflected-boundary request is invalid";
            return false;
        }
        const auto input = JuicerDirBoundary::project_input(component, {width, height});
        const auto identity = component.boundaryIdentity;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            if (boundary.weights && boundary.identity == identity) {
                return true;
            }
        }

        FjDirBoundaryView view{};
        if (!JuicerDirBoundary::prepare(hostPreparation, input, view, outError)) {
            return false;
        }
        const std::size_t bytes = view.count * sizeof(double);
        void* candidateRaw = nullptr;
        if (!alloc_and_upload_bytes(
                resources,
                candidateRaw,
                view.weights,
                bytes,
                cudaStreamOpaque,
                "spatial DIR reflected boundary",
                outError)) {
            return false;
        }

        bool keepCandidate = false;
        bool success = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (validate_resource_owner_locked(resources, outError)) {
                reap_retire_queue_locked(resources);
                if (boundary.weights && boundary.identity == identity) {
                    keepCandidate = false;
                    success = true;
                } else if (!boundary.weights ||
                           retire_ptr_locked(
                               resources,
                               boundary.weights,
                               boundary.bytes,
                               Resources::RetireKind::DeviceFree,
                               cudaStreamOpaque,
                               "spatial DIR reflected boundary",
                               outError)) {
                    boundary.weights = static_cast<double*>(candidateRaw);
                    boundary.bytes = bytes;
                    boundary.horizontalLength = view.horizontal_length;
                    boundary.verticalLength = view.vertical_length;
                    boundary.horizontalTerminalSize = view.horizontal_terminal_size;
                    boundary.verticalTerminalSize = view.vertical_terminal_size;
                    std::copy_n(view.horizontal_terminal, boundary.horizontalTerminalMatrix.size(), boundary.horizontalTerminalMatrix.begin());
                    std::copy_n(view.vertical_terminal, boundary.verticalTerminalMatrix.size(), boundary.verticalTerminalMatrix.begin());
                    boundary.identity = identity;
                    keepCandidate = true;
                    success = true;
                }
            }
        }
        if (!keepCandidate) {
            free_owned_device_noexcept(resources, candidateRaw);
        }
        return success;
    }

    bool clear_spatial_dir_boundary_binding(
        Resources& resources,
        Resources::DeviceSpatialDirBoundary& boundary,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);
        if (boundary.weights &&
            !retire_ptr_locked(
                resources,
                boundary.weights,
                boundary.bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "spatial DIR reflected boundary clear",
                outError)) {
            return false;
        }
        boundary = Resources::DeviceSpatialDirBoundary{};
        return true;
    }

    bool clear_spatial_dir_kernel_binding(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        Failure& outError) {
        outError = {};
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);
        clear_gaussian_kernel(kernel);
        return true;
    }

    bool ensure_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        float sigma,
        int radius,
        Failure& outError,
        const FjHostPreparation* hostPreparation) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        constexpr int kMaxRadius = 75;
        if (!std::isfinite(sigma) || sigma <= 0.0f || radius <= 0 ||
            radius > kMaxRadius) {
            outError.diagnostic = "gaussian descriptor is invalid";
            return false;
        }
        return ensure_cached_gaussian_kernel(
            resources,
            kernel,
            radius,
            sigma,
            "gaussian kernel",
            outError,
            hostPreparation);
    }

} // namespace JuicerCuda
