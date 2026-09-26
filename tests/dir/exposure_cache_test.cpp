#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir(const JuicerCuda::DirectPipelineRunParams*, JuicerCuda::SpatialDirBuildRequest);
extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir_cached_log_raw(const JuicerCuda::DirectPipelineRunParams*, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(const JuicerCuda::DirectPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_direct_focused_capture_density(const JuicerCuda::DirectPipelineRunParams*, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_direct_focused_capture_density_from_camera_film_linear(const JuicerCuda::DirectPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_build_print_spatial_dir(const JuicerCuda::PrintPipelineRunParams*, JuicerCuda::SpatialDirBuildRequest);
extern "C" cudaError_t juicer_cuda_build_print_spatial_dir_cached_log_raw(const JuicerCuda::PrintPipelineRunParams*, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_build_print_spatial_dir_cached_log_raw_from_camera_film_linear(const JuicerCuda::PrintPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_capture_density(const JuicerCuda::PrintPipelineRunParams*, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_capture_density_from_camera_film_linear(const JuicerCuda::PrintPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);

namespace {
    using Frame = JuicerProcess::Root::PreparedCudaFrame;

    void require(bool ok, const std::string& detail) {
        if (!ok) {
            throw std::runtime_error(detail);
        }
    }

    void require_cuda(cudaError_t result) {
        require(result == cudaSuccess, cudaGetErrorString(result));
    }

    class DeviceBuffer {
    public:
        explicit DeviceBuffer(std::size_t bytes) {
            require_cuda(cudaMalloc(&data, bytes));
        }
        ~DeviceBuffer() {
            (void)cudaFree(data);
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        float* data = nullptr;
    };

    struct Inputs {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scanner;
        Spektrafilm::SpatialDirDescriptor dir;
        int width;
        int height;

        Inputs(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method, float pixelSize, Spektrafilm::DirFrameExtent extent)
            : width(extent.width), height(extent.height) {
            ParamSnapshot controls;
            controls.scanRoute = route;
            controls.spectralUpsamplingMode = static_cast<int>(method);
            controls.grainControls.active = false;
            controls.scatterHalationControls = {};
            if (Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Positive) {
                controls.filmProfileKey = "fujifilm_provia_100f";
            }
            controls.dirCouplers.diffusionSizeUm = 20.0f;
            controls.dirCouplers.diffusionTailUm = 200.0f;
            controls.dirCouplers.diffusionTailWeight = 0.03f;
            std::string error;
            const bool print = Spektrafilm::scan_route_is_print(route);
            require(print ? build_print_render_state_product(controls, product, error)
                          : build_direct_render_state_product(controls, product, error),
                    error);
            require(print ? Scanner::build_print_scanner_spectral_lut_descriptor(
                                {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error)
                          : Scanner::build_direct_scanner_spectral_lut_descriptor(
                                {&product.recipe.profileRoute, &product.recipe.densityBounds, &product.recipe.scannerOutput}, scanner, error),
                    error);
            require(Spektrafilm::build_spatial_dir_descriptor(product.recipe.dirCouplers, pixelSize, extent, extent, "dir-exposure-cache-test", dir), "DIR descriptor failed");
        }

        JuicerProcess::Root::CudaFramePreparationRequest request() const {
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
            r.spatialDirDescriptor = &dir;
            r.requestedWidth = width;
            r.requestedHeight = height;
            return r;
        }
    };

    void bind_filter(JuicerCuda::SpatialDirFilterSpec& out,
                     const Spektrafilm::DirGaussianComponentPlan& plan,
                     const Frame::KernelView& kernel,
                     const Frame::SpatialDirPreparedView::BoundaryView& boundary) {
        out.kernel = kernel.weights;
        out.radius = kernel.radius;
        out.sigma = kernel.sigma;
        out.weight = plan.weight;
        switch (plan.referenceOperator) {
            case Spektrafilm::DirReferenceOperator::Identity:
                out.filterOperator = JuicerCuda::SpatialDirFilterOperator::Identity;
                break;
            case Spektrafilm::DirReferenceOperator::SpektrafilmSmallFirReflect:
                out.filterOperator = JuicerCuda::SpatialDirFilterOperator::FirReflect;
                break;
            case Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect:
                out.filterOperator = JuicerCuda::SpatialDirFilterOperator::YvvReflect;
                break;
            default:
                out.filterOperator = JuicerCuda::SpatialDirFilterOperator::None;
                break;
        }
        out.iir.feedforward = plan.iir.feedforward;
        std::copy(plan.iir.feedback.begin(), plan.iir.feedback.end(), out.iir.feedback);
        out.horizontalBoundary.initialWeights = boundary.horizontalWeights;
        out.horizontalBoundary.initialWeightLength = boundary.horizontalLength;
        out.horizontalBoundary.terminalSize = boundary.horizontalTerminalSize;
        std::copy(boundary.horizontalTerminalMatrix.begin(), boundary.horizontalTerminalMatrix.end(), out.horizontalBoundary.terminalMatrix);
        out.verticalBoundary.initialWeights = boundary.verticalWeights;
        out.verticalBoundary.initialWeightLength = boundary.verticalLength;
        out.verticalBoundary.terminalSize = boundary.verticalTerminalSize;
        std::copy(boundary.verticalTerminalMatrix.begin(), boundary.verticalTerminalMatrix.end(), out.verticalBoundary.terminalMatrix);
    }

    class ExposureCache : public testing::Test {
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
            const bool retired = JuicerProcess::root().retire_idle_context(0, key.contextOpaque, error);
            const auto status = cudaStreamDestroy(stream);
            require(retired, error);
            require_cuda(status);
        }

        template <typename Params>
        void compare(const Inputs& inputs, bool carrierInput, int cachePlanes, bool benchmark = false) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x4558504f53555245ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.contextEpoch = 1;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.product.payload.uploadCoreHash, inputs.product.recipe.dirCouplers.hash, inputs.product.payload.scannerHash, 0);
            std::string error;
            auto frame = JuicerProcess::root().prepare_cuda_frame(key, snapshot, inputs.request(), {}, stream, error);
            require(frame.active(), error);
            const auto lease = frame.workspace_lease();
            require(frame.prepare_spatial_dir_resources(inputs.dir, lease, stream, error), error);
            const auto scratch = frame.spatial_dir_scratch(lease);
            const auto resources = frame.spatial_dir_resources(lease, inputs.dir.hash);
            require(scratch.active && resources.active && scratch.rawCorrectionM && scratch.rawCorrectionC, "Missing three-channel DIR workspace");
            JuicerCuda::FilmPayloadPack payload;
            const auto& recipe = inputs.product.recipe;
            require(JuicerCuda::pack_film_payloads(recipe.filmRaw, recipe.filmDevelop, recipe.dirCouplers, recipe.densityBounds, frame.focused_resources().film, nullptr, 1.25f, payload, error), error);
            const std::size_t count = static_cast<std::size_t>(inputs.width) * inputs.height;
            const std::size_t bytes = count * sizeof(float);
            DeviceBuffer source(4 * bytes), camera(3 * bytes), fusedCache(3 * bytes), separateCache(3 * bytes), actualDensity(3 * bytes), expectedDensity(3 * bytes), failures(sizeof(int));
            std::vector<float> input(4 * count);
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    input[4 * i + channel] = i % 17 == 0 ? 0.0f : static_cast<float>((i * 17 + channel * 89) % 997) / 500.0f - 0.02f;
                }
                input[4 * i + 3] = 0.37f;
            }
            require_cuda(cudaMemcpyAsync(source.data, input.data(), 4 * bytes, cudaMemcpyHostToDevice, stream));
            require_cuda(cudaMemsetAsync(failures.data, 0, sizeof(int), stream));
            Params params;
            params.width = inputs.width;
            params.height = inputs.height;
            params.nComponents = 4;
            params.src = source.data;
            params.srcRowBytes = static_cast<std::size_t>(inputs.width) * 4 * sizeof(float);
            params.filmRaw = payload.filmRaw;
            params.filmExpose = payload.filmExposure;
            params.filmDevelop = payload.filmDevelop;
            params.scanStage.scanErrorFlag = reinterpret_cast<int*>(failures.data);
            JuicerCuda::SpatialDirBuildRequest request;
            request.streamOpaque = stream;
            request.planes.rawCorrectionY = scratch.rawCorrectionY;
            request.planes.rawCorrectionM = scratch.rawCorrectionM;
            request.planes.rawCorrectionC = scratch.rawCorrectionC;
            request.planes.filteredCorrectionY = scratch.filteredCorrectionY;
            request.planes.filteredCorrectionM = scratch.filteredCorrectionM;
            request.planes.filteredCorrectionC = scratch.filteredCorrectionC;
            request.planes.filterTemp = scratch.filterTemp;
            request.planes.filterTempM = scratch.filterTempM;
            request.planes.filterTempC = scratch.filterTempC;
            bind_filter(request.gaussian, inputs.dir.filterPlan.components[0], resources.gaussian, resources.boundaries[0]);
            for (std::size_t tail = 0; tail < 3; ++tail) {
                bind_filter(request.tails[tail], inputs.dir.filterPlan.components[tail + 1], resources.exponential[tail], resources.boundaries[tail + 1]);
            }
            if (carrierInput) {
                std::vector<float> exposures(3 * count);
                for (std::size_t i = 0; i < exposures.size(); ++i) {
                    exposures[i] = i % 13 == 0 ? 0.0f : std::exp2(static_cast<float>(i % 97) / 5.0f - 12.0f);
                }
                require_cuda(cudaMemcpyAsync(camera.data, exposures.data(), 3 * bytes, cudaMemcpyHostToDevice, stream));
                require_cuda(cudaStreamSynchronize(stream));
                request.cameraFilmLinear = {camera.data, camera.data + count, camera.data + 2 * count, static_cast<std::size_t>(inputs.width)};
            }
            const auto build = [&]() {
                if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
                    require_cuda(juicer_cuda_build_direct_spatial_dir(&params, request));
                } else {
                    require_cuda(juicer_cuda_build_print_spatial_dir(&params, request));
                }
            };
            const auto build_separate_cache = [&]() {
                if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
                    require_cuda(carrierInput ? juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(&params, request.cameraFilmLinear, separateCache.data, separateCache.data + count, separateCache.data + 2 * count, stream)
                                              : juicer_cuda_build_direct_spatial_dir_cached_log_raw(&params, separateCache.data, separateCache.data + count, separateCache.data + 2 * count, stream));
                } else {
                    require_cuda(carrierInput ? juicer_cuda_build_print_spatial_dir_cached_log_raw_from_camera_film_linear(&params, request.cameraFilmLinear, separateCache.data, separateCache.data + count, separateCache.data + 2 * count, stream)
                                              : juicer_cuda_build_print_spatial_dir_cached_log_raw(&params, separateCache.data, separateCache.data + count, separateCache.data + 2 * count, stream));
                }
            };
            const auto develop = [&](float* cache, float* density) {
                params.filmDevelop.spatialDir = {1, scratch.filteredCorrectionY, scratch.filteredCorrectionM, scratch.filteredCorrectionC, cache, cache + count, cachePlanes == 3 ? cache + 2 * count : nullptr};
                if constexpr (std::is_same_v<Params, JuicerCuda::DirectPipelineRunParams>) {
                    require_cuda(carrierInput ? juicer_cuda_direct_focused_capture_density_from_camera_film_linear(&params, request.cameraFilmLinear, density, density + count, density + 2 * count, stream)
                                              : juicer_cuda_direct_focused_capture_density(&params, density, density + count, density + 2 * count, stream));
                } else {
                    require_cuda(carrierInput ? juicer_cuda_print_focused_capture_density_from_camera_film_linear(&params, request.cameraFilmLinear, density, density + count, density + 2 * count, stream)
                                              : juicer_cuda_print_focused_capture_density(&params, density, density + count, density + 2 * count, stream));
                }
            };
            const auto launch = [&](bool reuse) {
                params.filmDevelop.spatialDir = {};
                request.planes.logRawB = reuse ? fusedCache.data : nullptr;
                request.planes.logRawG = reuse ? fusedCache.data + count : nullptr;
                request.planes.logRawR = reuse && cachePlanes == 3 ? fusedCache.data + 2 * count : nullptr;
                build();
                if (!reuse) {
                    build_separate_cache();
                }
                develop(reuse ? fusedCache.data : separateCache.data, reuse ? actualDensity.data : expectedDensity.data);
            };
            // Distinct poison catches unwritten output. Filtering and development
            // execute between cache production and readback, as they do in rendering.
            require_cuda(cudaMemsetAsync(fusedCache.data, 0xff, 3 * bytes, stream));
            require_cuda(cudaMemsetAsync(separateCache.data, 0x3c, 3 * bytes, stream));
            launch(false);
            launch(true);
            std::vector<float> actual(3 * count), expected(3 * count);
            const auto compare_planes = [&](const float* a, const float* b, std::size_t elements, bool finite) {
                require_cuda(cudaMemcpyAsync(actual.data(), a, elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
                require_cuda(cudaMemcpyAsync(expected.data(), b, elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
                require_cuda(cudaStreamSynchronize(stream));
                for (std::size_t i = 0; i < elements; ++i) {
                    if (finite) {
                        ASSERT_TRUE(std::isfinite(actual[i])) << "element=" << i;
                    }
                    ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i])) << "element=" << i;
                }
            };
            compare_planes(fusedCache.data, separateCache.data, static_cast<std::size_t>(cachePlanes) * count, false);
            compare_planes(actualDensity.data, expectedDensity.data, 3 * count, true);
            int failureBits = -1;
            require_cuda(cudaMemcpy(&failureBits, failures.data, sizeof(int), cudaMemcpyDeviceToHost));
            EXPECT_EQ(failureBits, 0);
            if (benchmark) {
                cudaEvent_t start = nullptr, stop = nullptr;
                require_cuda(cudaEventCreate(&start));
                require_cuda(cudaEventCreate(&stop));
                std::array<std::vector<float>, 2> times;
                for (int sample = -10; sample < 30; ++sample) {
                    // Alternate pair order to avoid assigning clock drift to one path.
                    for (int turn = 0; turn < 2; ++turn) {
                        const int mode = (sample + 10 + turn) % 2;
                        require_cuda(cudaEventRecord(start, stream));
                        launch(mode == 1);
                        require_cuda(cudaEventRecord(stop, stream));
                        require_cuda(cudaEventSynchronize(stop));
                        float ms = 0.0f;
                        require_cuda(cudaEventElapsedTime(&ms, start, stop));
                        if (sample >= 0) {
                            times[static_cast<std::size_t>(mode)].push_back(ms);
                        }
                    }
                }
                for (std::size_t mode = 0; mode < times.size(); ++mode) {
                    auto& samples = times[mode];
                    std::cout << "DIR_EXPOSURE_SAMPLES reuse=" << mode;
                    for (float ms : samples) {
                        std::cout << ' ' << ms;
                    }
                    std::sort(samples.begin(), samples.end());
                    std::cout << "\nDIR_EXPOSURE_BENCHMARK reuse=" << mode << " median_ms=" << (samples[14] + samples[15]) * 0.5f << '\n';
                }
                require_cuda(cudaEventDestroy(start));
                require_cuda(cudaEventDestroy(stop));
            }
            require(frame.finish(stream, error), error);
        }

        JuicerCuda::ResourceManager::DeviceContextKey key;
        cudaStream_t stream = nullptr;
        inline static std::uint64_t nextIdentity = 1;
    };

    TEST_F(ExposureCache, SourceBuildMatchesSeparateCacheAndDevelopedDensityBitForBit) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            for (auto method : {Spektrafilm::RgbToRawMethod::Hanatos2025, Spektrafilm::RgbToRawMethod::Mallett2019, Spektrafilm::RgbToRawMethod::Arctic2026beta04}) {
                for (float pixelSize : {6.0f, 200.0f}) {
                    Inputs inputs(route, method, pixelSize, {0, 0, 64, 48});
                    for (bool carrier : {false, true}) {
                        for (int cachePlanes : {2, 3}) {
                            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route) << " method=" << static_cast<int>(method) << " pixelSize=" << pixelSize << " carrier=" << carrier << " cachePlanes=" << cachePlanes);
                            if (Spektrafilm::scan_route_is_print(route)) {
                                compare<JuicerCuda::PrintPipelineRunParams>(inputs, carrier, cachePlanes);
                            } else {
                                compare<JuicerCuda::DirectPipelineRunParams>(inputs, carrier, cachePlanes);
                            }
                        }
                    }
                }
            }
        }
    }

    // Explicit CUDA-event measurement; never part of ordinary CTest.
    TEST_F(ExposureCache, DISABLED_SixKTiming) {
        Inputs inputs(Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::RgbToRawMethod::Hanatos2025, 36000.0f / 6048.0f, {0, 0, 6048, 4032});
        compare<JuicerCuda::PrintPipelineRunParams>(inputs, false, 3, true);
    }
} // namespace

int main(int argc, char** argv) {
    JuicerCuda::Owner cudaOwner;
    cudaOwner.create(JuicerProcess::data_directory());
    testing::InitGoogleTest(&argc, argv);
    JuicerProcess::root().ensure_bootstrap();
    const int result = RUN_ALL_TESTS();
    JuicerProcess::shutdown_if_initialized();
    return result;
}
