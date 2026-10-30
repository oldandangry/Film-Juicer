#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <memory>
#include <limits>
#include <mutex>
#include <thread>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/JuicerCudaFailure.h"
#include "gtest/gtest.h"

#include "ColorTransforms.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "RustDirBoundaryBridge.h"
#include "boundary_test_hooks.h"
#include "juicer_cuda_owner.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/prepared_frame_fixture.h"
#include "../ffi/prepared_product_fixture.h"

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
    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

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
        PreparedState state;
        Spektrafilm::SpatialDirDescriptor dir;
        JuicerCuda::FilmPayloadInput filmPayload{};
        std::uint64_t uploadCoreHash = 0;
        std::uint64_t scannerHash = 0;
        std::uint64_t dirRecipeHash = 0;
        float pixelSizeUm = 0.0f;
        int width;
        int height;

        Inputs(Spektrafilm::ScanRoute route, Spektrafilm::RgbToRawMethod method, float pixelSize, Spektrafilm::DirFrameExtent extent, float tailWeight = 0.03f)
            : state(JuicerTestSupport::prepare_product(controls(route, method, tailWeight))),
              pixelSizeUm(pixelSize), width(extent.width), height(extent.height) {
            FrameFixture::FrameFacts facts;
            facts.renderExtent = {extent.x, extent.y, extent.width, extent.height};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.renderExtent;
            facts.pixelSizeUm = pixelSize;
            Capture captured{};
            std::string diagnostic;
            require(FrameFixture::with_prepared_frame_inputs(
                        state, facts, &captured, &capture_values, diagnostic),
                    diagnostic);
            dir = captured.dir;
            filmPayload = captured.filmPayload;
            uploadCoreHash = captured.uploadCoreHash;
            scannerHash = captured.scannerHash;
            dirRecipeHash = captured.dirRecipeHash;
        }

        static ParamSnapshot controls(Spektrafilm::ScanRoute route,
                                      Spektrafilm::RgbToRawMethod method,
                                      float tailWeight) {
            ParamSnapshot result;
            result.scanRoute = route;
            result.spectralUpsamplingMode = static_cast<int>(method);
            result.grainControls.active = false;
            result.scatterHalationControls = {};
            if (Spektrafilm::scan_route_metadata(route).capturePolarity ==
                Spektrafilm::ProfilePolarity::Positive) {
                result.filmProfileKey = "fujifilm_provia_100f";
            }
            result.dirCouplers.diffusionSizeUm = 20.0f;
            result.dirCouplers.diffusionTailUm = 200.0f;
            result.dirCouplers.diffusionTailWeight = tailWeight;
            return result;
        }

        struct Capture {
            Spektrafilm::SpatialDirDescriptor dir;
            JuicerCuda::FilmPayloadInput filmPayload;
            std::uint64_t uploadCoreHash = 0;
            std::uint64_t scannerHash = 0;
            std::uint64_t dirRecipeHash = 0;
        };

        static bool capture_values(void* user,
                                   const FrameFixture::Inputs& prepared,
                                   std::string&) {
            auto& out = *static_cast<Capture*>(user);
            out.dir = prepared.descriptors.spatialDir;
            out.filmPayload = prepared.filmPayload;
            out.uploadCoreHash = prepared.state.upload_core_hash;
            out.scannerHash = prepared.state.scanner_hash;
            out.dirRecipeHash = prepared.state.dir_recipe.hash;
            return true;
        }
    };

    struct RootFrameCall {
        const Inputs* inputs = nullptr;
        const JuicerCuda::ResourceManager::DeviceContextKey* key = nullptr;
        const JuicerCuda::ResourceManager::SubmissionSnapshot* snapshot = nullptr;
        cudaStream_t stream = nullptr;
        JuicerCuda::Failure failure;
        Frame frame;
    };

    bool prepare_root_frame(void* user,
                            const FrameFixture::Inputs& prepared,
                            std::string& diagnostic) {
        auto& call = *static_cast<RootFrameCall*>(user);
        JuicerProcess::Root& root = JuicerProcess::root();
        Frame frame = root.prepare_cuda_frame(
            *call.key, *call.snapshot, prepared.preparation, {}, call.stream, call.failure, prepared.hostPreparation);
        if (!frame.active()) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        if (!frame.prepare_spatial_dir_resources(
                call.inputs->dir, frame.workspace_lease(), call.stream, call.failure, prepared.hostPreparation)) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        call.frame = std::move(frame);
        return true;
    }

    Frame prepare_frame(const Inputs& inputs,
                        const JuicerCuda::ResourceManager::DeviceContextKey& key,
                        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
                        cudaStream_t stream,
                        JuicerCuda::Failure& failure) {
        RootFrameCall call{&inputs, &key, &snapshot, stream, {}, {}};
        std::string diagnostic;
        const bool ready = FrameFixture::with_prepared_frame_inputs(
            inputs.state,
            {{0, 0, inputs.width, inputs.height},
             {0, 0, inputs.width, inputs.height},
             {0, 0, inputs.width, inputs.height},
             inputs.pixelSizeUm},
            &call,
            &prepare_root_frame,
            diagnostic);
        failure = call.failure;
        require(ready, diagnostic.empty() ? failure.diagnostic : diagnostic);
        return std::move(call.frame);
    }

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

    using Json = nlohmann::json;
    Json boundary_fixture() {
        std::ifstream input(JUICER_DIR_BOUNDARY_FIXTURE_PATH);
        require(bool(input), "DIR boundary fixture open");
        return Json::parse(input);
    }
    Json boundary_words(const double* values, std::size_t count) {
        Json words = Json::array();
        for (std::size_t index = 0; index < count; ++index)
            words.push_back(std::bit_cast<std::uint64_t>(values[index]));
        return words;
    }
    std::uint64_t boundary_key(const Spektrafilm::DirGaussianComponentPlan& component, int width, int height) {
        JuicerCuda::Failure failure;
        std::uint64_t key = 0;
        require(JuicerDirBoundary::describe(JuicerDirBoundary::project_input(component, {width, height}), key, failure), failure.diagnostic);
        return key;
    }
    const Json& expected_boundary(const Json& fixture, std::uint64_t key) {
        for (const auto& row : fixture.at("cases"))
            if (row.at("success").get<bool>() && row.at("key") == key)
                return row;
        throw std::runtime_error("actual DIR boundary has no frozen native expectation identity=" + std::to_string(key));
    }
    Json expected_weights(const Json& row) {
        Json words = row.at("horizontal").at("weights");
        for (const auto& word : row.at("vertical").at("weights"))
            words.push_back(word);
        return words;
    }
    void compare_boundary(const Frame::SpatialDirPreparedView::BoundaryView& actual, const Json& row) {
        const auto& horizontal = row.at("horizontal");
        const auto& vertical = row.at("vertical");
        require(actual.horizontalLength == horizontal.at("length") && actual.verticalLength == vertical.at("length"), "Root boundary retained lengths");
        require(actual.horizontalTerminalSize == horizontal.at("terminal_size") && actual.verticalTerminalSize == vertical.at("terminal_size"), "Root boundary terminal sizes");
        require(actual.horizontalWeights && actual.verticalWeights == actual.horizontalWeights + 3 * static_cast<std::size_t>(actual.horizontalLength), "Root vertical offset binds the combined allocation");
        const auto expected = expected_weights(row);
        std::vector<double> downloaded(expected.size());
        require_cuda(cudaMemcpy(downloaded.data(), actual.horizontalWeights, downloaded.size() * sizeof(double), cudaMemcpyDeviceToHost));
        require(boundary_words(downloaded.data(), downloaded.size()) == expected, "entire actual Root boundary array matches frozen native words");
        require(boundary_words(actual.horizontalTerminalMatrix.data(), 9) == horizontal.at("terminal") && boundary_words(actual.verticalTerminalMatrix.data(), 9) == vertical.at("terminal"), "Root terminal packing is exact");
    }
    Frame::SpatialDirPreparedView::BoundaryView boundary_view(const JuicerCuda::Resources::DeviceSpatialDirBoundary& boundary) {
        return {boundary.weights, boundary.weights ? boundary.weights + 3 * static_cast<std::size_t>(boundary.horizontalLength) : nullptr, boundary.horizontalLength, boundary.verticalLength, boundary.horizontalTerminalSize, boundary.verticalTerminalSize, boundary.horizontalTerminalMatrix, boundary.verticalTerminalMatrix};
    }
    void same_binding(const JuicerCuda::Resources::DeviceSpatialDirBoundary& actual, const JuicerCuda::Resources::DeviceSpatialDirBoundary& old) {
        require(actual.weights == old.weights && actual.bytes == old.bytes && actual.identity == old.identity &&
                    actual.horizontalLength == old.horizontalLength && actual.verticalLength == old.verticalLength &&
                    actual.horizontalTerminalSize == old.horizontalTerminalSize && actual.verticalTerminalSize == old.verticalTerminalSize &&
                    boundary_words(actual.horizontalTerminalMatrix.data(), 9) == boundary_words(old.horizontalTerminalMatrix.data(), 9) &&
                    boundary_words(actual.verticalTerminalMatrix.data(), 9) == boundary_words(old.verticalTerminalMatrix.data(), 9),
                "failed boundary preparation preserves every old binding field");
    }
    struct BoundaryWorkers {
        JuicerCuda::DirBoundaryTest::Controls& controls;
        JuicerCuda::Resources::DeviceSpatialDirBoundary& boundary;
        const Spektrafilm::DirGaussianComponentPlan component;
        const int width, height;
        void* stream;
        std::array<std::thread, 2> threads;
        std::array<std::exception_ptr, 2> exceptions{};
        std::array<JuicerCuda::Failure, 2> failures{};
        BoundaryWorkers(JuicerCuda::DirBoundaryTest::Controls& value, JuicerCuda::Resources::DeviceSpatialDirBoundary& target, Spektrafilm::DirGaussianComponentPlan input, std::array<int, 2> dimensions, void* uploadStream)
            : controls(value), boundary(target), component(input), width(dimensions[0]), height(dimensions[1]), stream(uploadStream) {}
        ~BoundaryWorkers() {
            controls.cancel();
            join();
        }
        void join() {
            for (auto& thread : threads)
                if (thread.joinable())
                    thread.join();
        }
        void start(unsigned count) {
            for (unsigned index = 0; index < count; ++index)
                threads[index] = std::thread([&, index] {
                    bool success = false;
                    try {
                        require(cuCtxSetCurrent(static_cast<CUcontext>(controls.context.contextOpaque)) == CUDA_SUCCESS, "DIR worker exact current context");
                        JuicerCuda::DirBoundaryTest::Binding binding(controls, index);
                        success = JuicerCuda::ensure_spatial_dir_boundary(controls.resources, boundary, component, width, height, stream, failures[index]);
                    } catch (...) {
                        exceptions[index] = std::current_exception();
                    }
                    std::lock_guard lock(controls.mutex);
                    controls.participants[index].success = success;
                    controls.participants[index].done = true;
                    controls.changed.notify_all();
                });
        }
        void check() {
            join();
            for (const auto& exception : exceptions)
                if (exception)
                    std::rethrow_exception(exception);
            require(!controls.timedOut, "DIR rendezvous timeout is a failure");
        }
    };

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
            if (trackedLedger) {
                const auto state = trackedLedger->snapshot();
                require(state.chargedBytes == 0 && state.reservedBytes == 0 && state.recordCount == 0, "DIR context retirement drains physical allocation ownership");
            }
        }

        Frame prepare_boundary(const Inputs& inputs) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x444952424f554e44ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.uploadCoreHash, inputs.dirRecipeHash, inputs.scannerHash, 0);
            JuicerCuda::Failure failure;
            return prepare_frame(inputs, key, snapshot, stream, failure);
        }
        JuicerCuda::Resources& boundary_owner() {
            Inputs inputs(Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::RgbToRawMethod::Hanatos2025, 1.0f, {0, 0, 64, 48});
            const auto identity = boundary_key(inputs.dir.filterPlan.components[0], 64, 48);
            JuicerCuda::DirBoundaryTest::Watch watch(key, identity);
            auto frame = prepare_boundary(inputs);
            require(watch.resources && watch.epoch != 0 && watch.resources->ownerContextKey == key, "observe the actual Root exact-context owner");
            trackedLedger = watch.resources->deviceLedger;
            JuicerCuda::Failure failure;
            require(frame.finish(stream, failure), failure.diagnostic);
            require_cuda(cudaStreamSynchronize(stream));
            // Root retains this owner until this fixture's drained context retirement.
            return *watch.resources;
        }

        template <typename Params>
        void compare(const Inputs& inputs, bool carrierInput, int cachePlanes, bool benchmark = false) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x4558504f53555245ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.uploadCoreHash, inputs.dirRecipeHash, inputs.scannerHash, 0);
            JuicerCuda::Failure error;
            auto frame = prepare_frame(inputs, key, snapshot, stream, error);
            const auto lease = frame.workspace_lease();
            const auto scratch = frame.spatial_dir_scratch(lease);
            const auto resources = frame.spatial_dir_resources(lease, inputs.dir.hash);
            require(scratch.active && resources.active && scratch.rawCorrectionM && scratch.rawCorrectionC, "Missing three-channel DIR workspace");
            JuicerCuda::FilmPayloadPack payload;
            require(JuicerCuda::pack_film_payloads(
                        inputs.filmPayload, frame.focused_resources().film, nullptr, 1.25f, payload, error.diagnostic),
                    error.diagnostic);
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
            require(frame.finish(stream, error), error.diagnostic);
        }

        JuicerCuda::ResourceManager::DeviceContextKey key;
        cudaStream_t stream = nullptr;
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> trackedLedger;
        inline static std::uint64_t nextIdentity = 1;
    };

    TEST_F(ExposureCache, GaussianFirArraysMatchFrozenWordsAndWarmHitsDoNoConstruction) {
        for (auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan, Spektrafilm::ScanRoute::PositiveDirectScan, Spektrafilm::ScanRoute::PositivePrintScan}) {
            Inputs inputs(route, Spektrafilm::RgbToRawMethod::Hanatos2025, 12.0f, {0, 0, 64, 48});
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x4741555353ull;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.uploadCoreHash, inputs.dirRecipeHash, inputs.scannerHash, 0);
            JuicerCuda::Failure error;
            const auto prepareFir = [&] {
                return prepare_frame(inputs, key, snapshot, stream, error);
            };
            auto frame = prepareFir();
            const auto prepared = frame.spatial_dir_resources(frame.workspace_lease(), inputs.dir.hash);
            ASSERT_TRUE(prepared.active);
            ASSERT_NE(prepared.gaussian.weights, nullptr);
            const auto expected = gaussian_kernel_fixture_words(JUICER_GAUSSIAN_FIXTURE_PATH, prepared.gaussian.radius, prepared.gaussian.sigma);
            std::vector<float> actual(expected.size());
            require_cuda(cudaMemcpy(actual.data(), prepared.gaussian.weights, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::bit_cast<std::uint32_t>(actual[i]) == expected[i], "all actual FIR DIR Gaussian words match frozen native coefficients");
            require(frame.finish(stream, error), error.diagnostic);
            const auto acquisitions = fj_test_gaussian_kernel_acquires();
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            auto warm = prepareFir();
            EXPECT_EQ(fj_test_gaussian_kernel_acquires(), acquisitions);
            EXPECT_EQ(warm.spatial_dir_resources(warm.workspace_lease(), inputs.dir.hash).gaussian.weights, prepared.gaussian.weights);
            require(warm.finish(stream, error), error.diagnostic);
        }
    }

    TEST_F(ExposureCache, ReflectedBoundaryRootArraysWarmHitsClearFinishAndAbort) {
        const auto fixture = boundary_fixture();
        struct Case {
            Spektrafilm::ScanRoute route;
            float pixel, tail;
            int width, height;
        };
        const std::array cases{
            Case{Spektrafilm::ScanRoute::NegativeDirectScan, 1.0f, 0.0f, 64, 48},
            Case{Spektrafilm::ScanRoute::NegativeDirectScan, 8.0f, 0.03f, 64, 48},
            Case{Spektrafilm::ScanRoute::NegativePrintScan, 8.0f, 0.03f, 65, 63},
            Case{Spektrafilm::ScanRoute::PositiveDirectScan, 20.0f, 0.03f, 64, 48},
            Case{Spektrafilm::ScanRoute::PositivePrintScan, 1.0f, 0.0f, 65, 63},
            Case{Spektrafilm::ScanRoute::PositivePrintScan, 100.0f, 0.03f, 64, 48},
            Case{Spektrafilm::ScanRoute::NegativePrintScan, 200.0f, 0.03f, 64, 48},
            Case{Spektrafilm::ScanRoute::PositiveDirectScan, 8.0f, 0.03f, 65, 63}};
        for (const auto& spec : cases) {
            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(spec.route) << " pixel=" << spec.pixel << " tail=" << spec.tail << " width=" << spec.width);
            Inputs inputs(spec.route, Spektrafilm::RgbToRawMethod::Hanatos2025, spec.pixel, {0, 0, spec.width, spec.height}, spec.tail);
            auto frame = prepare_boundary(inputs);
            const auto actual = frame.spatial_dir_resources(frame.workspace_lease(), inputs.dir.hash);
            ASSERT_TRUE(actual.active);
            for (std::size_t slot = 0; slot < actual.boundaries.size(); ++slot) {
                const auto& component = inputs.dir.filterPlan.components[slot];
                const bool active = slot < static_cast<std::size_t>(inputs.dir.filterPlan.componentCount) && component.weight > 0.0f && component.referenceOperator == Spektrafilm::DirReferenceOperator::SpektrafilmLargeYvvReflect;
                if (active)
                    compare_boundary(actual.boundaries[slot], expected_boundary(fixture, boundary_key(component, spec.width, spec.height)));
                else
                    require(!actual.boundaries[slot].horizontalWeights && !actual.boundaries[slot].verticalWeights && !actual.boundaries[slot].horizontalLength && !actual.boundaries[slot].verticalLength, "Root clears nonparticipating boundary views");
            }
            require(fj_test_dir_boundary_count(3) == 0, "no host owner retained in prepared frame/cache");
            JuicerCuda::Failure failure;
            require(frame.finish(stream, failure), failure.diagnostic);
            const auto acquires = fj_test_dir_boundary_count(0);
            auto warm = prepare_boundary(inputs);
            const auto reused = warm.spatial_dir_resources(warm.workspace_lease(), inputs.dir.hash);
            require(fj_test_dir_boundary_count(0) == acquires, "warm Root preparation performs no Rust acquire/vector construction");
            for (std::size_t slot = 0; slot < actual.boundaries.size(); ++slot)
                require(actual.boundaries[slot].horizontalWeights == reused.boundaries[slot].horizontalWeights, "warm Root reuses each device binding");
            warm.abort();
            require(fj_test_dir_boundary_count(3) == 0, "abort retains no Rust host owner");
        }
    }

    TEST_F(ExposureCache, ReflectedBoundaryFailuresStagingAndIndependentEpoch) {
        auto& resources = boundary_owner();
        const auto fixture = boundary_fixture();
        auto& binding = resources.spatialDirBoundaries[0];
        const auto old = binding;
        const auto& oldExpected = expected_boundary(fixture, old.identity);
        auto ignored = JuicerCuda::DirBoundaryTest::component(oldExpected.at("input").at("component"));
        ignored.sigmaPixels = std::numeric_limits<float>::quiet_NaN();
        ignored.radius = -1;
        ignored.weight = std::numeric_limits<float>::infinity();
        JuicerCuda::Failure failure;
        auto acquires = fj_test_dir_boundary_count(0);
        require(JuicerCuda::ensure_spatial_dir_boundary(resources, binding, ignored, 64, 48, stream, failure), failure.diagnostic);
        require(fj_test_dir_boundary_count(0) == acquires, "ignored diagnostic fields preserve warm-hit opportunity");
        same_binding(binding, old);
        const auto& candidate = fixture.at("cases").at(7);
        const auto component = JuicerCuda::DirBoundaryTest::component(candidate.at("input").at("component"));
        const auto identity = candidate.at("key").get<std::uint64_t>();
        for (const auto& [operation, fault] : std::array<std::pair<unsigned, unsigned>, 7>{{{1, 1}, {1, 2}, {2, 1}, {2, 9}, {2, 10}, {2, 11}, {2, 12}}}) {
            const auto releases = fj_test_dir_boundary_count(2);
            require(fj_test_dir_boundary_fault(operation, fault).category == 0, "arm bounded real boundary failure");
            if (operation == 2)
                require(fj_test_dir_boundary_fault(3, 1).category == 0, "cleanup panic must preserve primary failure");
            require(!JuicerCuda::ensure_spatial_dir_boundary(resources, binding, component, 9, 6, stream, failure), "real resource miss rejects failed Rust boundary");
            const auto expectedCategory = operation == 1 ? (fault == 1 ? 8u : 4u) : fault == 1 ? 8u
                                                                                               : 3u;
            require(failure.status.category == expectedCategory && failure.status.api == 0 && failure.status.native_code == 0, "primary boundary status survives release panic");
            same_binding(binding, old);
            require(fj_test_dir_boundary_count(3) == 0 && fj_test_dir_boundary_count(2) == releases + (operation == 2 ? 1u : 0u), "real miss consumes its only host owner once");
        }
        {
            const auto before = resources.deviceLedger->snapshot();
            JuicerCuda::DirBoundaryTest::Controls controls(resources, identity);
            controls.participants[0].failUpload = true;
            {
                JuicerCuda::DirBoundaryTest::Binding observe(controls, 0);
                require(!JuicerCuda::ensure_spatial_dir_boundary(resources, binding, component, 9, 6, stream, failure), "bounded actual upload failure");
            }
            const auto& p = controls.participants[0];
            require(p.uploadRejected && p.copies == 1 && p.candidate && p.frees == 1 && p.freed == p.candidate && p.freeSucceeded && !p.observationFailed, "failed real candidate is physically freed exactly once");
            same_binding(binding, old);
            const auto after = resources.deviceLedger->snapshot();
            require(after.chargedBytes == before.chargedBytes && after.reservedBytes == before.reservedBytes && after.recordCount == before.recordCount, "failed upload preserves physical accounting");
            require(fj_test_dir_boundary_count(3) == 0, "failed upload releases host boundary");
        }
        {
            JuicerCuda::DirBoundaryTest::Controls controls(resources, identity);
            {
                JuicerCuda::DirBoundaryTest::Binding observe(controls, 0);
                require(JuicerCuda::ensure_spatial_dir_boundary(resources, binding, component, 9, 6, stream, failure), failure.diagnostic);
            }
            const auto& p = controls.participants[0];
            require(p.copies == 1 && p.candidate == binding.weights && p.hostOwners >= 1 && p.staging && reinterpret_cast<std::uintptr_t>(p.staging) != p.sourceAddress && !p.observationFailed, "actual upload copies into independent native pinned staging while host owner is live");
            require(fj_test_dir_boundary_count(3) == 0, "successful ensure releases host owner before later native staging read");
            require(boundary_words(p.stagedWeights.data(), p.stagedWeights.size()) == expected_weights(candidate) && boundary_words(p.staging, p.stagedWeights.size()) == expected_weights(candidate), "actual staged bytes remain intact after host owner release");
            compare_boundary(boundary_view(binding), candidate);
        }
        acquires = fj_test_dir_boundary_count(0);
        std::unique_ptr<JuicerCuda::Resources, decltype(&JuicerCuda::destroy)> epoch(
            JuicerCuda::create(key, resources.contextEpoch + 1, resources.deviceLedger, failure), &JuicerCuda::destroy);
        require(bool(epoch), failure.diagnostic);
        require(JuicerCuda::ensure_spatial_dir_boundary(*epoch, epoch->spatialDirBoundaries[0], component, 9, 6, stream, failure), failure.diagnostic);
        require(epoch->spatialDirBoundaries[0].weights != binding.weights && fj_test_dir_boundary_count(0) == acquires + 1, "independent native epoch constructs an independent device binding");
        compare_boundary(boundary_view(epoch->spatialDirBoundaries[0]), candidate);
        require(JuicerCuda::drain_for_context_retire(*epoch, failure), failure.diagnostic);
        require(epoch->deviceAllocationRecords.empty(), "independent epoch drains all physical records");
    }

    TEST_F(ExposureCache, ReflectedBoundarySameKeyRaceAndPrepublicationCancellation) {
        auto& resources = boundary_owner();
        auto& target = resources.spatialDirBoundaries[0];
        const auto fixture = boundary_fixture();
        const auto& candidate = fixture.at("cases").at(7);
        const auto component = JuicerCuda::DirBoundaryTest::component(candidate.at("input").at("component"));
        const auto identity = candidate.at("key").get<std::uint64_t>();
        const auto old = target;
        const auto acquires = fj_test_dir_boundary_count(0);
        {
            JuicerCuda::DirBoundaryTest::Controls controls(resources, identity);
            controls.rendezvous = true;
            BoundaryWorkers workers(controls, target, component, {9, 6}, stream);
            workers.start(2);
            require(controls.await_arrivals(2), "two real same-key uploaded candidates rendezvous");
            require(controls.participants[0].candidate && controls.participants[1].candidate && controls.participants[0].candidate != controls.participants[1].candidate, "distinct real same-key allocations");
            require(fj_test_dir_boundary_count(3) == 2, "both miss-local host owners survive synchronous staging and prepublication wait");
            controls.release(0);
            require(controls.await_done(0) && controls.participants[0].success, "first candidate publishes before second resumes");
            controls.release(1);
            require(controls.await_done(1) && controls.participants[1].success, "same-key loser succeeds by reusing winner");
            workers.check();
            const auto& winner = controls.participants[0];
            const auto& loser = controls.participants[1];
            require(target.weights == winner.candidate && target.identity == identity && winner.frees == 0 && loser.frees == 1 && loser.freed == loser.candidate && loser.freeSucceeded, "publication keeps winner and physically frees the actual losing allocation");
            require(fj_test_dir_boundary_count(3) == 0 && fj_test_dir_boundary_count(0) == acquires + 2, "same-key race releases both host owners");
            compare_boundary(boundary_view(target), candidate);
        }
        const auto& next = fixture.at("cases").at(6);
        const auto nextComponent = JuicerCuda::DirBoundaryTest::component(next.at("input").at("component"));
        const auto current = target;
        JuicerCuda::DirBoundaryTest::Controls controls(resources, next.at("key"));
        controls.rendezvous = true;
        BoundaryWorkers workers(controls, target, nextComponent, {next.at("input").at("width").get<int>(), next.at("input").at("height").get<int>()}, stream);
        workers.start(1);
        require(controls.await_arrivals(1), "canceled candidate reaches real prepublication boundary");
        controls.cancel();
        workers.check();
        const auto& rejected = controls.participants[0];
        require(!rejected.success && rejected.frees == 1 && rejected.freed == rejected.candidate && rejected.freeSucceeded, "prepublication cancellation frees candidate");
        same_binding(target, current);
        require(target.weights != old.weights && fj_test_dir_boundary_count(3) == 0, "race replacement and cancellation preserve native ownership");
    }

    TEST_F(ExposureCache, ReflectedBoundarySecondOwnerValidationRejectsQuarantinedPublication) {
        auto& resources = boundary_owner();
        auto& target = resources.spatialDirBoundaries[0];
        const auto old = target;
        const auto fixture = boundary_fixture();
        const auto& candidate = fixture.at("cases").at(7);
        const auto component = JuicerCuda::DirBoundaryTest::component(candidate.at("input").at("component"));
        JuicerCuda::DirBoundaryTest::Controls controls(resources, candidate.at("key"));
        controls.rendezvous = true;
        BoundaryWorkers workers(controls, target, component, {9, 6}, stream);
        workers.start(1);
        require(controls.await_arrivals(1), "candidate uploaded before second exact owner validation");
        // Exercise the existing owner transition, without changing context/epoch,
        // allocator policy or the publication implementation.
        JuicerCuda::quarantine_unfenced_frame_use(resources, stream);
        controls.release(0);
        require(controls.await_done(0), "revalidated candidate returns");
        workers.check();
        const auto& rejected = controls.participants[0];
        require(!rejected.success && workers.failures[0].diagnostic == "CUDA resources quarantined after unfenced frame use", "actual second owner validation blocks publication");
        require(rejected.frees == 1 && rejected.freed == rejected.candidate && rejected.freeSucceeded && fj_test_dir_boundary_count(3) == 0, "owner rejection frees candidate and releases host boundary");
        same_binding(target, old);
    }

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
    const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
    return closed.category == FJ_STATUS_SUCCESS ? result : 1;
}
