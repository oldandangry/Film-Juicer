#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "ProcessRoot.h"

extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload*, const JuicerCuda::GrainKernelPayload*, int, int, float*, float*, float*, float*, float*, float*, float*, float*, void*);

namespace {
    using Frame = JuicerProcess::Root::PreparedCudaFrame;

    void require(bool ok, const std::string& detail) {
        if (!ok) {
            throw std::runtime_error(detail);
        }
    }

    void require_cuda(cudaError_t status) {
        require(status == cudaSuccess, cudaGetErrorString(status));
    }

    struct GrainVariant {
        bool layers = true;
        bool shared = true;
        int debug = 0;
        double time = 12.0;
        float pixelSize = 6.0f;
        bool cameraDiffusion = false;
    };

    struct Inputs {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scanner;
        Spektrafilm::SpatialDirDescriptor dir;
        Spektrafilm::VisualGrainFrameDescriptor grain;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        int width;
        int height;

        Inputs(int w, int h, Spektrafilm::ScanRoute route, const GrainVariant& variant = {})
            : width(w), height(h) {
            ParamSnapshot controls;
            controls.scanRoute = route;
            if (Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Positive) {
                controls.filmProfileKey = "fujifilm_provia_100f";
            }
            controls.scatterHalationControls = {};
            controls.cameraDiffusion.active = variant.cameraDiffusion;
            controls.gateWeaveAmount = 0.0;
            controls.grainControls.active = true;
            controls.grainControls.sublayersActive = variant.layers;
            controls.grainControls.chromaSharedWeight = variant.shared ? 1.0f : 0.0f;
            controls.grainControls.chromaIndependentWeight = variant.shared ? 0.0f : 1.0f;
            controls.grainControls.debugView = variant.debug;
            controls.dirCouplers.diffusionSizeUm = 20.0f;
            controls.dirCouplers.diffusionTailUm = 200.0f;
            controls.dirCouplers.diffusionTailWeight = 0.03f;
            std::string error;
            const bool print = Spektrafilm::scan_route_is_print(route);
            require(print ? build_print_render_state_product(controls, product, error)
                          : build_direct_render_state_product(controls, product, error),
                    error);
            if (print) {
                require(Scanner::build_print_scanner_spectral_lut_descriptor(
                            {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error),
                        error);
            } else {
                require(Scanner::build_direct_scanner_spectral_lut_descriptor(
                            {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error),
                        error);
            }
            if (variant.cameraDiffusion) {
                require(Spektrafilm::build_diffusion_frame_set_descriptor(
                            product.recipe.spatialOptics, route, variant.pixelSize, {0, 0, w, h}, diffusion, error),
                        error);
            }
            const Spektrafilm::DirFrameExtent extent{0, 0, w, h};
            require(Spektrafilm::build_spatial_dir_descriptor(
                        product.recipe.dirCouplers, variant.pixelSize, extent, extent, "grain-scratch-test", dir),
                    "DIR descriptor failed");
            Spektrafilm::VisualGrainFrameDescriptorInput input;
            input.recipe = &product.recipe.visualGrain;
            input.filmDevelop = &product.recipe.filmDevelop;
            input.capturePolarity = product.recipe.profileRoute.capturePolarity;
            input.renderExtent = {0, 0, w, h};
            input.fullFrameExtent = input.renderExtent;
            input.pixelSizeUm = variant.pixelSize;
            input.frameTime = variant.time;
            input.frameRate = 24.0;
            input.sessionSeed = 17;
            input.clipToken = 23;
            require(Spektrafilm::build_visual_grain_frame_descriptor(input, grain, error), error);
        }

        JuicerProcess::Root::CudaFramePreparationRequest request(bool withDir = true) const {
            JuicerProcess::Root::CudaFramePreparationRequest r;
            r.recipe = &product.recipe;
            r.exposureTables = &product.payload.exposureTables;
            r.filmRawConfig = &product.payload.filmRawConfig;
            r.filmTcLut = product.payload.filmTcLut ? &*product.payload.filmTcLut : nullptr;
            r.printMainIlluminant = product.payload.printMainIlluminant ? &*product.payload.printMainIlluminant : nullptr;
            r.scannerTables = &product.payload.scannerTables;
            r.scannerColor = &product.payload.scannerColor;
            r.scannerLutDescriptor = &scanner;
            r.outputBoundaryTable = product.payload.outputBoundaryTable.get();
            r.spatialDirDescriptor = withDir ? &dir : nullptr;
            r.visualGrainDescriptor = grain;
            r.diffusionFrameSetDescriptor = diffusion ? &*diffusion : nullptr;
            r.requestedWidth = width;
            r.requestedHeight = height;
            return r;
        }
    };

    class GrainScratch : public testing::Test {
    protected:
        void SetUp() override {
            require_cuda(cudaSetDevice(0));
            require_cuda(cudaFree(nullptr));
            CUcontext context = nullptr;
            require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "No CUDA context");
            key = {0, context};
            require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }

        void TearDown() override {
            require_cuda(cudaStreamSynchronize(stream));
            std::string error;
            const bool retired = contextRetired ||
                                 JuicerProcess::root().retire_idle_context(0, key.contextOpaque, error);
            const cudaError_t destroyStatus = cudaStreamDestroy(stream);
            stream = nullptr;
            require(retired, error);
            require_cuda(destroyStatus);
        }

        Frame prepare(const Inputs& inputs, bool withDir = true) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x475241494eull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.contextEpoch = 1;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.product.payload.uploadCoreHash, inputs.product.recipe.dirCouplers.hash, inputs.product.payload.scannerHash, 0);
            std::string error;
            auto frame = JuicerProcess::root().prepare_cuda_frame(
                key, snapshot, inputs.request(withDir), {}, stream, error);
            require(frame.active(), error);
            const auto lease = frame.workspace_lease();
            if (withDir) {
                require(frame.prepare_spatial_dir_resources(inputs.dir, lease, stream, error), error);
            }
            require(frame.stage_optical_workspace(lease, stream, error), error);
            return frame;
        }

        void finish(Frame& frame) {
            std::string error;
            require(frame.finish(stream, error), error);
        }

        void compare_grain(const Inputs& inputs, Frame& frame) {
            const auto lease = frame.workspace_lease();
            const auto work = frame.visual_grain_workspace(lease);
            const auto density = frame.capture_film_density_workspace(lease);
            require(work.active && density.active, "Missing grain/density workspace");
            JuicerCuda::GrainPayload payload;
            JuicerCuda::GrainKernelPayload kernels;
            std::string error;
            require(JuicerCuda::pack_visual_grain_payload(inputs.product.recipe.visualGrain,
                                                          frame.visual_grain_resources(),
                                                          payload,
                                                          kernels,
                                                          error),
                    error);
            payload.frameUniforms = work.frameUniforms;
            const std::size_t count = static_cast<std::size_t>(inputs.width) * inputs.height;
            const std::size_t bytes = count * sizeof(float);
            struct DevicePlanes {
                float* data = nullptr;
                explicit DevicePlanes(std::size_t size) {
                    require_cuda(cudaMalloc(&data, size));
                }
                ~DevicePlanes() {
                    (void)cudaFree(data);
                }
                DevicePlanes(const DevicePlanes&) = delete;
                DevicePlanes& operator=(const DevicePlanes&) = delete;
            } reference(8 * bytes);
            std::vector<float> source(count);
            const std::array<float*, 3> planes{density.c, density.m, density.y};
            for (std::size_t channel = 0; channel < 3; ++channel) {
                for (std::size_t i = 0; i < count; ++i) {
                    source[i] = (0.02f + static_cast<float>((i * 17 + channel * 31) % 997) / 1000.0f) * payload.densityMax[channel];
                }
                require_cuda(cudaMemcpyAsync(planes[channel], source.data(), bytes, cudaMemcpyHostToDevice, stream));
                require_cuda(cudaMemcpyAsync(reference.data + channel * count, source.data(), bytes, cudaMemcpyHostToDevice, stream));
                require_cuda(cudaStreamSynchronize(stream));
            }
            // Simulate dirty contents left by DIR to catch reads-before-writes,
            // including optional/debug grain paths.
            for (float* pointer : {work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta}) {
                if (pointer) {
                    require_cuda(cudaMemsetAsync(pointer, 0xff, bytes, stream));
                }
            }
            require_cuda(cudaMemsetAsync(reference.data + 3 * count, 0x3c, 5 * bytes, stream));
            require_cuda(juicer_cuda_apply_visual_grain(&payload, &kernels, inputs.width, inputs.height, reference.data, reference.data + count, reference.data + 2 * count, reference.data + 3 * count, reference.data + 4 * count, reference.data + 5 * count, work.layerWork ? reference.data + 6 * count : nullptr, work.sharedDelta ? reference.data + 7 * count : nullptr, stream));
            require_cuda(juicer_cuda_apply_visual_grain(&payload, &kernels, inputs.width, inputs.height, density.c, density.m, density.y, work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta, stream));
            std::vector<float> actual(count), expected(count);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                require_cuda(cudaMemcpyAsync(actual.data(), planes[channel], bytes, cudaMemcpyDeviceToHost, stream));
                require_cuda(cudaMemcpyAsync(expected.data(), reference.data + channel * count, bytes, cudaMemcpyDeviceToHost, stream));
                require_cuda(cudaStreamSynchronize(stream));
                for (std::size_t i = 0; i < count; ++i) {
                    ASSERT_TRUE(std::isfinite(actual[i])) << "channel=" << channel << " pixel=" << i;
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i]))
                        << "channel=" << channel << " pixel=" << i;
                }
            }
        }

        inline static std::uint64_t nextIdentity = 1;
        JuicerCuda::ResourceManager::DeviceContextKey key;
        cudaStream_t stream = nullptr;
        bool contextRetired = false;
    };

    TEST_F(GrainScratch, ReusesFinishedRawCorrectionPlanesWithoutDedicatedGrainAllocations) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan,
                           Spektrafilm::ScanRoute::NegativePrintScan}) {
            Inputs inputs(128, 96, route);
            auto frame = prepare(inputs);
            const auto lease = frame.workspace_lease();
            const auto grain = frame.visual_grain_workspace(lease);
            const auto dir = frame.spatial_dir_scratch(lease);
            const auto optics = frame.scanner_workspace(lease);
            ASSERT_TRUE(grain.active);
            ASSERT_TRUE(dir.active);
            ASSERT_EQ(dir.planeRoles.rawCorrectionPlanes, 3);
            EXPECT_EQ(grain.deltaAccum, dir.rawCorrectionY);
            EXPECT_EQ(grain.layerWork, dir.rawCorrectionM);
            EXPECT_EQ(grain.sharedDelta, dir.rawCorrectionC);
            EXPECT_EQ(optics.aux, nullptr);
            EXPECT_EQ(optics.grainTmp, nullptr);
            EXPECT_EQ(optics.grainTmpShared, nullptr);
            EXPECT_NE(grain.deltaAccum, optics.rgbR);
            EXPECT_NE(grain.layerWork, optics.rgbG);
            EXPECT_NE(grain.sharedDelta, optics.rgbB);
            finish(frame);
        }
    }

    TEST_F(GrainScratch, KeepsDedicatedWorkspaceWithoutSpatialDir) {
        Inputs inputs(64, 48, Spektrafilm::ScanRoute::NegativeDirectScan);
        auto frame = prepare(inputs, false);
        const auto grain = frame.visual_grain_workspace(frame.workspace_lease());
        const auto optics = frame.scanner_workspace(frame.workspace_lease());
        ASSERT_TRUE(grain.active);
        EXPECT_EQ(grain.deltaAccum, optics.aux);
        EXPECT_EQ(grain.layerWork, optics.grainTmp);
        EXPECT_EQ(grain.sharedDelta, optics.grainTmpShared);
        finish(frame);
    }
    TEST_F(GrainScratch, GrainMatchesDedicatedScratchBitForBitAcrossRoutesShapesTimesAndDebugViews) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            for (bool layers : {false, true}) {
                for (bool shared : {false, true}) {
                    for (double time : {12.0, 12.375}) {
                        for (int debug = 0; debug <= 6; ++debug) {
                            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route)
                                                            << " layers=" << layers << " shared=" << shared << " time=" << time << " debug=" << debug);
                            Inputs inputs(48, 32, route, {.layers = layers, .shared = shared, .debug = debug, .time = time});
                            auto frame = prepare(inputs);
                            const auto work = frame.visual_grain_workspace(frame.workspace_lease());
                            ASSERT_TRUE(work.active);
                            EXPECT_EQ(work.layerWork != nullptr, layers);
                            EXPECT_EQ(work.sharedDelta != nullptr, shared && debug <= 1);
                            compare_grain(inputs, frame);
                            finish(frame);
                        }
                    }
                }
            }
        }
    }

    TEST_F(GrainScratch, ReusesOnlyWithinEligibleRouteAndRetainedTier) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            for (float pixelSize : {6.0f, 200.0f}) {
                Inputs inputs(64, 48, route, {.pixelSize = pixelSize});
                auto frame = prepare(inputs);
                auto lease = frame.workspace_lease();
                const auto before = frame.visual_grain_workspace(lease);
                const auto dir = frame.spatial_dir_scratch(lease);
                const bool negative = inputs.product.recipe.profileRoute.capturePolarity == Spektrafilm::ProfilePolarity::Negative;
                const auto optics = frame.scanner_workspace(lease);
                EXPECT_EQ(before.deltaAccum, negative ? dir.rawCorrectionY : optics.aux);
                std::string error;
                if (dir.targetPlaneRoles.cachedLogRawPlanes == 3) {
                    require(frame.stage_spatial_dir_cached_log_raw_for_final_develop(lease, stream, error), error);
                }
                const auto after = frame.visual_grain_workspace(lease);
                ASSERT_TRUE(after.active);
                EXPECT_EQ(after.deltaAccum, before.deltaAccum);
                compare_grain(inputs, frame);
                finish(frame);
            }
        }
    }

    TEST_F(GrainScratch, SurvivesResolutionChangesQueuedAbortAndContextRetirement) {
        for (int width : {64, 320, 48, 128}) {
            Inputs inputs(width, width / 2, Spektrafilm::ScanRoute::NegativePrintScan);
            auto frame = prepare(inputs);
            const auto marker = frame.workspace_lease();
            const auto grain = frame.visual_grain_workspace(marker);
            ASSERT_TRUE(grain.active);
            require_cuda(cudaMemsetAsync(grain.deltaAccum, 0x7f, static_cast<std::size_t>(inputs.width) * inputs.height * sizeof(float), stream));
            frame.abort();
            EXPECT_FALSE(frame.visual_grain_workspace(marker).active);
            auto next = prepare(inputs);
            compare_grain(inputs, next);
            finish(next);
        }
        require_cuda(cudaStreamSynchronize(stream));
        std::string error;
        require(JuicerProcess::root().retire_idle_context(0, key.contextOpaque, error), error);
        Inputs inputs(96, 64, Spektrafilm::ScanRoute::NegativeDirectScan);
        auto recreated = prepare(inputs);
        compare_grain(inputs, recreated);
        finish(recreated);
    }

    TEST_F(GrainScratch, RejectsConcurrentRetainedLeasesAndOrdersLaterStreamReuse) {
        Inputs inputs(96, 64, Spektrafilm::ScanRoute::NegativePrintScan);
        auto first = prepare(inputs);
        try {
            auto overlap = prepare(inputs);
            FAIL() << "Concurrent retained scratch admission unexpectedly succeeded";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("serialized render contract"), std::string::npos);
        }
        first.abort();
        require_cuda(cudaStreamSynchronize(stream));
        std::string error;
        require(JuicerProcess::root().retire_idle_context(0, key.contextOpaque, error), error);
        auto queued = prepare(inputs);
        const auto work = queued.visual_grain_workspace(queued.workspace_lease());
        require_cuda(cudaMemsetAsync(work.deltaAccum, 0xff, static_cast<std::size_t>(inputs.width) * inputs.height * sizeof(float), stream));
        finish(queued);
        cudaStream_t original = stream;
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto second = prepare(inputs);
        compare_grain(inputs, second);
        finish(second);
        require_cuda(cudaStreamSynchronize(stream));
        require_cuda(cudaStreamDestroy(original));
    }

    TEST_F(GrainScratch, GrainToggleReleasesLogicalWorkspaceAndReenablesReuse) {
        Inputs inputs(128, 96, Spektrafilm::ScanRoute::NegativePrintScan);
        auto first = prepare(inputs);
        finish(first);
        auto request = inputs.request();
        request.visualGrainDescriptor.reset();
        auto recipe = inputs.product.recipe;
        recipe.visualGrain = {};
        request.recipe = &recipe;
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        snapshot.instanceToken.value = 0x475241494eull;
        snapshot.frameToken.value = nextIdentity;
        snapshot.snapshotId = nextIdentity++;
        snapshot.deviceContextKey = key;
        snapshot.contextEpoch = 1;
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            inputs.product.payload.uploadCoreHash, recipe.dirCouplers.hash, inputs.product.payload.scannerHash, 0);
        std::string error;
        auto off = JuicerProcess::root().prepare_cuda_frame(key, snapshot, request, {}, stream, error);
        require(off.active(), error);
        EXPECT_FALSE(off.visual_grain_workspace(off.workspace_lease()).active);
        finish(off);
        auto on = prepare(inputs);
        compare_grain(inputs, on);
        finish(on);
    }

    TEST_F(GrainScratch, AdmittedTierPreservesGrainBindingAndOutput) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan}) {
            Inputs inputs(96, 64, route);
            auto frame = prepare(inputs);
            const auto lease = frame.workspace_lease();
            const auto work = frame.visual_grain_workspace(lease);
            const auto dir = frame.spatial_dir_scratch(lease);
            const auto optics = frame.scanner_workspace(lease);
            ASSERT_TRUE(work.active);
            if (dir.planeRoles.rawCorrectionPlanes == 3) {
                EXPECT_EQ(work.deltaAccum, dir.rawCorrectionY);
                EXPECT_EQ(work.layerWork, dir.rawCorrectionM);
                EXPECT_EQ(work.sharedDelta, dir.rawCorrectionC);
                EXPECT_EQ(optics.aux, nullptr);
            } else {
                EXPECT_EQ(dir.planeRoles.rawCorrectionPlanes, 1);
                EXPECT_EQ(work.deltaAccum, optics.aux);
                EXPECT_EQ(work.layerWork, optics.grainTmp);
                EXPECT_EQ(work.sharedDelta, optics.grainTmpShared);
                EXPECT_NE(work.deltaAccum, dir.rawCorrectionY);
            }
            std::cout << "ADMITTED_DIR_PLANES=" << dir.planeRoles.total_float_planes() << '\n';
            compare_grain(inputs, frame);
            finish(frame);
        }
    }

    TEST_F(GrainScratch, DiffusionKeepsDedicatedGrainScratch) {
        Inputs inputs(64, 48, Spektrafilm::ScanRoute::NegativeDirectScan, {.cameraDiffusion = true});
        auto frame = prepare(inputs);
        const auto lease = frame.workspace_lease();
        const auto work = frame.visual_grain_workspace(lease);
        const auto optics = frame.scanner_workspace(lease);
        const auto dir = frame.spatial_dir_scratch(lease);
        ASSERT_TRUE(work.active);
        EXPECT_EQ(work.deltaAccum, optics.aux);
        EXPECT_EQ(work.layerWork, optics.grainTmp);
        EXPECT_EQ(work.sharedDelta, optics.grainTmpShared);
        EXPECT_NE(work.deltaAccum, dir.rawCorrectionY);
        compare_grain(inputs, frame);
        finish(frame);
    }

    TEST_F(GrainScratch, ResetRetiresReusedWorkspace) {
        Inputs inputs(64, 48, Spektrafilm::ScanRoute::NegativePrintScan);
        auto frame = prepare(inputs);
        compare_grain(inputs, frame);
        finish(frame);
        require_cuda(cudaStreamSynchronize(stream));
        std::string error;
        require(JuicerProcess::root().retire_reset_context(0, key.contextOpaque, error), error);
        contextRetired = true;
    }

    // Explicit maintenance benchmark, excluded from ordinary CTest runs.
    TEST_F(GrainScratch, DISABLED_SixKMemoryAndTiming) {
        Inputs inputs(6048, 4032, Spektrafilm::ScanRoute::NegativePrintScan);
        auto frame = prepare(inputs);
        const auto lease = frame.workspace_lease();
        const auto work = frame.visual_grain_workspace(lease);
        const auto dir = frame.spatial_dir_scratch(lease);
        const auto optics = frame.scanner_workspace(lease);
        const auto density = frame.capture_film_density_workspace(lease);
        const std::set<const float*> pointers{
            dir.rawCorrectionY, dir.rawCorrectionM, dir.rawCorrectionC, dir.filteredCorrectionY, dir.filteredCorrectionM, dir.filteredCorrectionC, dir.filterTemp, dir.filterTempM, dir.filterTempC, dir.logRawB, dir.logRawG, dir.logRawR, optics.rgbR, optics.rgbG, optics.rgbB, optics.tmp, optics.blurred, optics.aux, optics.grainTmp, optics.grainTmpShared};
        std::uint64_t bytes = 0;
        std::size_t planes = 0;
        for (const float* pointer : pointers) {
            if (!pointer) {
                continue;
            }
            CUdeviceptr base = 0;
            std::size_t size = 0;
            require(cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(pointer)) == CUDA_SUCCESS,
                    "CUDA allocation range unavailable");
            EXPECT_EQ(base, reinterpret_cast<CUdeviceptr>(pointer));
            bytes += size;
            ++planes;
        }
        EXPECT_TRUE(planes == 13 || planes == 16);
        compare_grain(inputs, frame);
        JuicerCuda::GrainPayload payload;
        JuicerCuda::GrainKernelPayload kernels;
        std::string error;
        require(JuicerCuda::pack_visual_grain_payload(inputs.product.recipe.visualGrain,
                                                      frame.visual_grain_resources(),
                                                      payload,
                                                      kernels,
                                                      error),
                error);
        payload.frameUniforms = work.frameUniforms;
        cudaEvent_t start = nullptr, end = nullptr;
        require_cuda(cudaEventCreate(&start));
        require_cuda(cudaEventCreate(&end));
        std::vector<float> times;
        const std::size_t planeBytes = static_cast<std::size_t>(inputs.width) * inputs.height * sizeof(float);
        for (int sample = -10; sample < 30; ++sample) {
            for (float* pointer : {density.c, density.m, density.y}) {
                require_cuda(cudaMemsetAsync(pointer, 0x3f, planeBytes, stream));
            }
            require_cuda(cudaEventRecord(start, stream));
            require_cuda(juicer_cuda_apply_visual_grain(&payload, &kernels, inputs.width, inputs.height, density.c, density.m, density.y, work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta, stream));
            require_cuda(cudaEventRecord(end, stream));
            require_cuda(cudaEventSynchronize(end));
            float elapsed = 0;
            require_cuda(cudaEventElapsedTime(&elapsed, start, end));
            if (sample >= 0) {
                times.push_back(elapsed);
            }
        }
        std::sort(times.begin(), times.end());
        std::cout << "GRAIN_REUSE_BENCHMARK planes=" << planes << " scratch_bytes=" << bytes
                  << " median_ms=" << (times[14] + times[15]) * 0.5f
                  << " min_ms=" << times.front() << " max_ms=" << times.back() << '\n';
        require_cuda(cudaEventDestroy(start));
        require_cuda(cudaEventDestroy(end));
        finish(frame);
    }

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    JuicerProcess::root().ensure_bootstrap();
    const int result = RUN_ALL_TESTS();
    JuicerProcess::shutdown_if_initialized();
    return result;
}
