#include "Cuda/JuicerCudaGrainV2.cuh"

#include <algorithm>
#include <cmath>

namespace JuicerCuda::GrainV2 {
    // The arrays are paired Full extents and action steps in axis order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::uint64_t proxy_sample_bound(const double (&extent)[2], const double (&step)[2]) noexcept {
        constexpr double maxAxisCells = kPlacementCenterLimit / kMinProxySampleFootprint + 0.5;
        std::uint64_t bound = 1;
        for (std::size_t axis = 0; axis < 2; ++axis) {
            if (!(extent[axis] > 0.0 && extent[axis] <= maxAxisCells) ||
                !(step[axis] > 0.0 && std::isfinite(step[axis])))
                return 0;
            // Allow an unaligned cell and independent rounding of the two
            // edges. Clipping to the Full image can only reduce the count.
            const double cells = std::min(std::ceil(extent[axis]), std::ceil(step[axis]) + 2.0);
            bound *= static_cast<std::uint64_t>(cells);
        }
        return bound;
    }

    cudaError_t apply(const FilmQuad* sums, FilmTables tables, FilmFrame frame, FilmPlanes planes, void* stream) {
        if (!sums || !planes.cyan || !planes.magenta || !planes.yellow || frame.width <= 0 || frame.height <= 0 || frame.stride < frame.width)
            return cudaErrorInvalidValue;
        auto* const cudaStream = reinterpret_cast<cudaStream_t>(stream);
        const dim3 threads{32, 8};
        if (frame.proxy.step[0] == 0.0 && frame.proxy.step[1] == 0.0) {
            const auto& [left, top, right, bottom] = frame.calibratedBorder;
            frame.width -= left + right;
            frame.height -= top + bottom;
            if (frame.width <= 0 || frame.height <= 0)
                return cudaErrorInvalidValue;
            // Only Film sees the cropped view. Kernel-local coordinates start
            // at the first supported pixel; the density planes keep their stride.
            const auto offset = static_cast<std::size_t>(top) * frame.stride + left;
            planes.cyan += offset;
            planes.magenta += offset;
            planes.yellow += offset;
            const dim3 blocks{(static_cast<unsigned>(frame.width) + 31U) / 32U, (static_cast<unsigned>(frame.height) + 7U) / 8U};
            apply_quad<<<blocks, threads, 0, cudaStream>>>(sums, tables, frame, planes);
            return cudaGetLastError();
        }
        const auto samples = proxy_sample_bound(frame.proxy.extent, frame.proxy.step);
        if (samples == 0)
            return cudaErrorInvalidValue;
        if (samples <= kProxySerialSamples) {
            const dim3 blocks{(static_cast<unsigned>(frame.width) + 31U) / 32U, (static_cast<unsigned>(frame.height) + 7U) / 8U};
            apply_proxy<<<blocks, threads, 0, cudaStream>>>(sums, tables, frame, planes);
            return cudaGetLastError();
        }
        if (!frame.proxyWorkspace)
            return cudaErrorInvalidValue;

        // Each pass integrates disjoint Full cells. Reuse the frame-owned
        // storage in stream order after folding its partials into pixel totals.
        constexpr auto samplesPerPass = static_cast<std::uint64_t>(kProxyBlocksPerPixel) * kProxySamplesPerBlock;
        const auto pixels = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height);
        for (std::size_t pixel = 0; pixel < pixels; pixel += kProxyPixelsPerBatch) {
            const auto count = static_cast<unsigned int>(std::min<std::size_t>(pixels - pixel, kProxyPixelsPerBatch));
            for (std::uint64_t first = 0; first < samples; first += samplesPerPass) {
                const auto remaining = samples - first;
                const auto blockCount = static_cast<unsigned int>(
                    (std::min(remaining, samplesPerPass) + kProxySamplesPerBlock - 1) / kProxySamplesPerBlock);
                const ProxyBatch batch{pixel, first, blockCount, remaining <= samplesPerPass};
                integrate_proxy_batch<<<dim3{blockCount, count}, kProxyThreads, 0, cudaStream>>>(sums, tables, frame, planes, batch);
                if (const auto status = cudaGetLastError(); status != cudaSuccess)
                    return status;
                combine_proxy_batch<<<1, count, 0, cudaStream>>>(frame, planes, batch);
                if (const auto status = cudaGetLastError(); status != cudaSuccess)
                    return status;
            }
        }
        return cudaSuccess;
    }
} // namespace JuicerCuda::GrainV2
