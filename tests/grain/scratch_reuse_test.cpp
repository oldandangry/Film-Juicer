#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "ColorTransforms.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/prepared_frame_fixture.h"
#include "../ffi/prepared_product_fixture.h"
#include "../ffi/spatial_optics_fixture.h"
#include "../ffi/case_results.h"
#include "../ffi/gtest_cases.h"

extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload*, const JuicerCuda::GrainKernelPayload*, int, int, float*, float*, float*, float*, float*, float*, float*, float*, void*);

namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
        struct GrainSnapshot {
            std::weak_ptr<JuicerCuda::Resources> frame;
            std::weak_ptr<JuicerCuda::Resources> grain;
            std::size_t members = 0;
            bool first = false;
            bool second = false;
            std::uint64_t epoch = 0;
            std::uint64_t grainRecords = 0;
            JuicerCuda::DeviceLedgerSnapshot ledger;
        };

        static GrainSnapshot grain_snapshot(Root& root,
                                            const JuicerCuda::ResourceManager::DeviceContextKey& context,
                                            std::uint64_t first,
                                            std::uint64_t second) {
            GrainSnapshot result{};
            std::lock_guard<std::mutex> lock(root._cudaResourcesMutex);
            for (const auto& [key, entry] : root._cudaContextResources) {
                if (key.deviceContextKey == context) {
                    result.frame = entry.frameOwner;
                    result.grain = entry.grainOwner;
                    result.members = entry.grainInstances.size();
                    result.first = entry.grainInstances.contains(first);
                    result.second = entry.grainInstances.contains(second);
                    result.epoch = key.contextEpoch;
                    if (entry.grainOwner) {
                        std::lock_guard<std::mutex> allocations(entry.grainOwner->deviceAllocationRecordsMutex);
                        result.grainRecords = entry.grainOwner->deviceAllocationRecords.size();
                    }
                }
            }
            const auto ledger = root._cudaDeviceLedgers.find(context.deviceId);
            if (ledger != root._cudaDeviceLedgers.end()) {
                result.ledger = ledger->second->snapshot();
            }
            return result;
        }
    };
} // namespace JuicerProcess::TestSupport

namespace {
    using Frame = JuicerProcess::Root::PreparedCudaFrame;
    namespace FrameFixture = JuicerCudaTest::PreparedFrameFixture;

    std::string route_case_key(Spektrafilm::ScanRoute route) {
        return "route-" + std::to_string(static_cast<int>(route));
    }

    constexpr std::array kCurveStorageRoutes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                             Spektrafilm::ScanRoute::NegativePrintScan,
                                             Spektrafilm::ScanRoute::PositiveDirectScan,
                                             Spektrafilm::ScanRoute::PositivePrintScan};
    constexpr std::array kCurveStorageLayerModes{false, true};

    std::string curve_storage_case_key(Spektrafilm::ScanRoute route, bool layers) {
        return route_case_key(route) + "/layers-" + (layers ? "on" : "off");
    }

    std::vector<std::string> curve_storage_case_ids() {
        std::vector<std::string> ids;
        for (const auto route : kCurveStorageRoutes)
            for (bool layers : kCurveStorageLayerModes)
                ids.push_back(curve_storage_case_key(route, layers));
        return ids;
    }

    void require(bool ok, const std::string& detail) {
        if (!ok) {
            throw std::runtime_error(detail);
        }
    }

    void require_cuda(cudaError_t status) {
        require(status == cudaSuccess, cudaGetErrorString(status));
    }

    struct GrainVariant {
        bool active = true;
        bool layers = true;
        bool shared = true;
        int debug = 0;
        double time = 12.0;
        float pixelSize = 6.0f;
        bool cameraDiffusion = false;
        float filmGamma = 1.0f;
        float dirAmount = 1.0f;
    };

    struct CurvePointers {
        std::array<JuicerCuda::DeviceCurveView, 3> normalized{};
        std::array<JuicerCuda::DeviceCurveView, 3> dir{};
        std::array<JuicerCuda::DeviceCurveView, 3> layerBase{};
        std::array<std::array<const float*, 3>, 3> layers{};
        std::uint64_t normalizedHash = 0;
        std::uint64_t layersHash = 0;
        bool layersActive = false;
    };

    struct Inputs {
        JuicerTestSupport::RuntimeFixture& runtime;
        JuicerTestSupport::PreparedStateHold state;
        JuicerTestSupport::PreparationInput controls;
        Spektrafilm::VisualGrainFrameDescriptor grain{};
        JuicerCuda::VisualGrainPayloadInput grainPayload{};
        Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
        std::uint64_t uploadCoreHash = 0;
        std::uint64_t scannerHash = 0;
        std::uint64_t dirRecipeHash = 0;
        std::uint64_t recipeHash = 0;
        std::uint64_t filmDevelopmentHash = 0;
        std::uint64_t normalizedDensityCurvesHash = 0;
        std::uint64_t densityCurvesLayersHash = 0;
        int width;
        int height;
        FrameFixture::FrameFacts facts{};

        Inputs(JuicerTestSupport::RuntimeFixture& processRuntime, int w, int h, Spektrafilm::ScanRoute route, const GrainVariant& variant = {})
            : runtime(processRuntime), state(JuicerTestSupport::prepare_product(runtime.borrow().assets, make_controls(route, variant))),
              controls(make_controls(route, variant)), width(w), height(h) {
            facts.renderExtent = {0, 0, w, h};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.renderExtent;
            facts.pixelSizeUm = variant.pixelSize;
            facts.filmFormatLongEdgeMm = controls.values.camera_film_format_long_edge_mm;
            facts.frameTime = variant.time;
            facts.frameRate = 24.0;
            facts.sessionSeed = 17;
            facts.clipToken = 23;
            capture_frame_inputs();
        }

        static JuicerTestSupport::PreparationInput make_controls(Spektrafilm::ScanRoute route, const GrainVariant& variant) {
            JuicerTestSupport::PreparationInput result;
            result.values.scan_route = static_cast<std::uint32_t>(route);
            result.values.film_gamma_factor = variant.filmGamma;
            if (Spektrafilm::scan_route_metadata(route).capturePolarity == Spektrafilm::ProfilePolarity::Positive)
                result.film_key = "fujifilm_provia_100f";
            result.values.scatter_halation = {};
            result.values.camera_diffusion.active = variant.cameraDiffusion;
            result.values.gate_weave_amount = 0.0;
            result.values.grain.active = variant.active;
            result.values.grain.sublayers_active = variant.layers;
            result.values.grain.chroma_shared_weight = variant.shared ? 1.0f : 0.0f;
            result.values.grain.chroma_independent_weight = variant.shared ? 0.0f : 1.0f;
            result.values.grain.debug_view = variant.debug;
            result.values.dir_couplers.amount = variant.dirAmount;
            result.values.dir_couplers.diffusion_size_um = 20.0f;
            result.values.dir_couplers.diffusion_tail_um = 200.0f;
            result.values.dir_couplers.diffusion_tail_weight = 0.03f;
            return result;
        }

        struct Capture {
            Inputs* inputs = nullptr;
        };

        static bool capture_values(void* user, const FrameFixture::Inputs& prepared, std::string&) {
            auto& inputs = *static_cast<Capture*>(user)->inputs;
            inputs.grain = prepared.descriptors.grain.value_or(Spektrafilm::VisualGrainFrameDescriptor{});
            inputs.grainPayload = prepared.grainPayload;
            inputs.polarity = prepared.descriptors.capturePolarity;
            inputs.uploadCoreHash = prepared.state.upload_core_hash;
            inputs.scannerHash = prepared.state.scanner_hash;
            inputs.dirRecipeHash = prepared.state.dir_recipe.hash;
            inputs.recipeHash = prepared.state.recipe_hash;
            inputs.filmDevelopmentHash = prepared.state.film_development.hash;
            inputs.normalizedDensityCurvesHash = prepared.state.film_development.density_curves_hash;
            inputs.densityCurvesLayersHash = prepared.state.film_development.density_layers_hash;
            return true;
        }

        void capture_frame_inputs() {
            Capture capture{this};
            std::string diagnostic;
            require(FrameFixture::with_prepared_frame_inputs(
                        runtime.borrow().assets, state, facts, &capture, &capture_values, diagnostic),
                    diagnostic);
        }

        void build_grain_frame(double time) {
            facts.frameTime = time;
            capture_frame_inputs();
        }
    };

    struct FrameCall {
        JuicerProcess::Root& root;
        const bool withDir;
        const JuicerCuda::ResourceManager::DeviceContextKey& key;
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot;
        cudaStream_t stream;
        JuicerCuda::Failure failure;
        Frame frame;
        CurvePointers* curves = nullptr;
        const CurvePointers* expectedCurves = nullptr;
    };

    void audit_density_curve(const JuicerCuda::DeviceCurveView& curve,
                             FjFloatSpan logExposure,
                             FjFloatSpan densityRgb,
                             std::size_t channel) {
        const std::size_t count = logExposure.count;
        require(count != 0 && curve.n == static_cast<int>(count) && densityRgb.count == count * 3,
                "device and held density curve extents match");
        std::vector<float> axis(count), density(count);
        require_cuda(cudaMemcpy(axis.data(), curve.x, count * sizeof(float), cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(density.data(), curve.y, count * sizeof(float), cudaMemcpyDeviceToHost));
        require(std::memcmp(axis.data(), logExposure.data, count * sizeof(float)) == 0, "device exposure axis matches held state");
        for (std::size_t row = 0; row < count; ++row) {
            require(std::bit_cast<std::uint32_t>(density[row]) == std::bit_cast<std::uint32_t>(densityRgb.data[row * 3 + channel]),
                    "device density matches held state");
        }
    }

    void audit_curve_storage(const FrameFixture::Inputs& prepared,
                             Frame& frame,
                             CurvePointers* output,
                             const CurvePointers* expected) {
        if (!output && !expected)
            return;
        const auto& development = prepared.state.film_development;
        const auto& dirRecipe = prepared.state.dir_recipe;
        const auto film = frame.focused_resources().film;
        const auto layers = frame.visual_grain_resources().densityLayers;
        const std::size_t count = development.log_exposure.count;
        require(film.normalizedDensityCurvesHash == development.density_curves_hash, "published normalized identity matches held state");
        require(film.dirCouplersHash == dirRecipe.hash, "published DIR identity matches held state");
        const std::array<JuicerCuda::DeviceCurveView, 3> normalized{{film.normalizedDensR, film.normalizedDensG, film.normalizedDensB}};
        const std::array<JuicerCuda::DeviceCurveView, 3> dir{{film.dirDensR, film.dirDensG, film.dirDensB}};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            {
                SCOPED_TRACE("normalized density");
                audit_density_curve(normalized[channel], development.log_exposure, development.density_rgb, channel);
            }
            if (dirRecipe.mode != FJ_DIR_INACTIVE) {
                SCOPED_TRACE("DIR-compensated density");
                audit_density_curve(dir[channel], dirRecipe.compensated_axes_rgb[channel], development.density_rgb, channel);
            }
            if (prepared.state.grain.sublayers_active != 0) {
                for (std::size_t layer = 0; layer < 3; ++layer) {
                    const auto* actual = layers.curves[layer][channel];
                    require(actual != nullptr, "active layer curve present");
                    std::vector<float> values(count);
                    require_cuda(cudaMemcpy(values.data(), actual, count * sizeof(float), cudaMemcpyDeviceToHost));
                    for (std::size_t row = 0; row < count; ++row) {
                        const auto& source = development.density_layers[layer][channel];
                        require(std::bit_cast<std::uint32_t>(values[row]) == std::bit_cast<std::uint32_t>(source.data[row]), "device layer density matches held state");
                    }
                }
            }
        }
        if (output) {
            output->normalized = normalized;
            output->dir = dir;
            std::copy(std::begin(layers.baseCurvesCmy), std::end(layers.baseCurvesCmy), output->layerBase.begin());
            for (std::size_t layer = 0; layer < 3; ++layer)
                for (std::size_t channel = 0; channel < 3; ++channel)
                    output->layers[layer][channel] = layers.curves[layer][channel];
            output->normalizedHash = development.density_curves_hash;
            output->layersHash = layers.hash;
            output->layersActive = layers.active;
        }
        if (expected) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                require(expected->normalized[channel].x == normalized[channel].x && expected->normalized[channel].y == normalized[channel].y, "exact normalized curve storage reused");
                require(expected->dir[channel].x == dir[channel].x && expected->dir[channel].y == dir[channel].y, "exact DIR curve storage reused");
                if (expected->layersActive) {
                    require(expected->layerBase[channel].x == layers.baseCurvesCmy[channel].x && expected->layerBase[channel].y == layers.baseCurvesCmy[channel].y, "exact layer base storage reused");
                    for (std::size_t layer = 0; layer < 3; ++layer)
                        require(expected->layers[layer][channel] == layers.curves[layer][channel], "exact layer curve storage reused");
                }
            }
            require(expected->layersActive == layers.active && expected->layersHash == layers.hash, "layer storage identity retained");
        }
    }

    bool prepare_root_frame(void* user, const FrameFixture::Inputs& prepared, std::string& diagnostic) {
        auto& call = *static_cast<FrameCall*>(user);
        auto preparation = prepared.preparation;
        if (!call.withDir)
            preparation.spatialDirDescriptor = nullptr;
        Frame frame = call.root.prepare_cuda_frame(
            call.key, call.snapshot, preparation, {}, call.stream, call.failure, prepared.hostPreparation);
        if (!frame.active()) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        const auto lease = frame.workspace_lease();
        if (call.withDir && !frame.prepare_spatial_dir_resources(
                                prepared.descriptors.spatialDir, lease, call.stream, call.failure, prepared.hostPreparation)) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        if (!frame.stage_optical_workspace(lease, call.stream, call.failure)) {
            diagnostic = call.failure.diagnostic;
            return false;
        }
        if (call.curves || call.expectedCurves)
            require_cuda(cudaStreamSynchronize(call.stream));
        audit_curve_storage(prepared, frame, call.curves, call.expectedCurves);
        call.frame = std::move(frame);
        return true;
    }

    struct ConsumerThrowSentinel final : std::exception {
        const char* what() const noexcept override {
            return "prepared-frame consumer sentinel";
        }
    };

    struct NoiseFaultReset {
        NoiseFaultReset() {
            fj_test_noise_fault(0);
        }
        ~NoiseFaultReset() {
            fj_test_noise_fault(0);
        }
    };

    bool throw_after_release_fault(void*, const FrameFixture::Inputs&, std::string&) {
        fj_test_noise_fault(3);
        throw ConsumerThrowSentinel{};
    }

    bool prepare_then_arm_release_fault(void* user,
                                        const FrameFixture::Inputs& prepared,
                                        std::string& diagnostic) {
        fj_test_noise_fault(3);
        return prepare_root_frame(user, prepared, diagnostic);
    }

    class GrainScratch : public testing::Test {
    protected:
        inline static JuicerTestSupport::RuntimeFixture runtime;

        static void SetUpTestSuite() {
            runtime.create(JuicerProcess::data_directory());
        }
        static void TearDownTestSuite() {
            require(runtime.close().category == FJ_STATUS_SUCCESS, "grain Runtime close failed");
        }

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
                                 JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, error);
            const cudaError_t destroyStatus = cudaStreamDestroy(stream);
            stream = nullptr;
            require(retired, error);
            require_cuda(destroyStatus);
        }

        Frame prepare(const Inputs& inputs,
                      bool withDir = true,
                      std::uint64_t instanceToken = 0x475241494eull,
                      CurvePointers* curves = nullptr,
                      const CurvePointers* expectedCurves = nullptr) {
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = instanceToken;
            snapshot.frameToken.value = nextIdentity;
            snapshot.snapshotId = nextIdentity++;
            snapshot.deviceContextKey = key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.uploadCoreHash, inputs.dirRecipeHash, inputs.scannerHash, 0);
            FrameCall call{JuicerCuda::borrow_root(runtime.borrow().cuda), withDir, key, snapshot, stream, {}, {}, curves, expectedCurves};
            std::string diagnostic;
            const bool ready = FrameFixture::with_prepared_frame_inputs(
                inputs.runtime.borrow().assets, inputs.state, inputs.facts, &call, &prepare_root_frame, diagnostic);
            require(ready, diagnostic.empty() ? call.failure.diagnostic : diagnostic);
            return std::move(call.frame);
        }

        void finish(Frame& frame) {
            JuicerCuda::Failure error;
            require(frame.finish(stream, error), error.diagnostic);
        }

        void compare_grain(const Inputs& inputs, Frame& frame) {
            const auto lease = frame.workspace_lease();
            const auto work = frame.visual_grain_workspace(lease);
            const auto density = frame.capture_film_density_workspace(lease);
            require(work.active && density.active, "Missing grain/density workspace");
            JuicerCuda::GrainPayload payload;
            JuicerCuda::GrainKernelPayload kernels;
            std::string error;
            require(JuicerCuda::pack_visual_grain_payload(inputs.grainPayload,
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

    TEST_F(GrainScratch, PreparedFrameNoiseReleaseFailurePreservesConsumerAndRootOutcomes) {
        JuicerTests::GtestCohort cases("noise-release-operation", {"consumer-exception-preserved", "release-failure-preserved", "retry-after-release-failure"});
        Inputs inputs(runtime, 64, 48, Spektrafilm::ScanRoute::NegativeDirectScan);
        const std::size_t baseline = fj_test_live_noise_owners();
        NoiseFaultReset resetFault;
        FrameFixture::FrameFacts facts = inputs.facts;

        auto consumerException = cases.case_event("consumer-exception-preserved");
        bool sentinelRethrown = false;
        std::string diagnostic;
        try {
            (void)FrameFixture::with_prepared_frame_inputs(
                inputs.runtime.borrow().assets, inputs.state, facts, nullptr, &throw_after_release_fault, diagnostic);
        } catch (const ConsumerThrowSentinel&) {
            sentinelRethrown = true;
        } catch (...) {
            FAIL() << "prepared-frame cleanup replaced the consumer exception";
        }
        EXPECT_TRUE(sentinelRethrown);
        EXPECT_EQ(fj_test_live_noise_owners(), baseline);
        consumerException.passed();

        auto releaseFailure = cases.case_event("release-failure-preserved");
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        snapshot.instanceToken.value = 0x475241494eull;
        snapshot.frameToken.value = nextIdentity;
        snapshot.snapshotId = nextIdentity++;
        snapshot.deviceContextKey = key;
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            inputs.uploadCoreHash, inputs.dirRecipeHash, inputs.scannerHash, 0);
        FrameCall call{JuicerCuda::borrow_root(runtime.borrow().cuda), true, key, snapshot, stream, {}, {}};
        diagnostic.clear();
        const bool releaseFailed = FrameFixture::with_prepared_frame_inputs(
            inputs.runtime.borrow().assets, inputs.state, facts, &call, &prepare_then_arm_release_fault, diagnostic);
        EXPECT_FALSE(releaseFailed);
        EXPECT_TRUE(call.frame.active());
        EXPECT_NE(diagnostic.find("asset boundary panic"), std::string::npos);
        EXPECT_EQ(fj_test_live_noise_owners(), baseline);
        if (call.frame.active())
            finish(call.frame);
        releaseFailure.passed();

        auto retry_event = cases.case_event("retry-after-release-failure");
        fj_test_noise_fault(0);
        FrameCall retry{JuicerCuda::borrow_root(runtime.borrow().cuda), true, key, snapshot, stream, {}, {}};
        diagnostic.clear();
        EXPECT_TRUE(FrameFixture::with_prepared_frame_inputs(
            inputs.runtime.borrow().assets, inputs.state, facts, &retry, &prepare_root_frame, diagnostic))
            << diagnostic;
        EXPECT_TRUE(retry.frame.active());
        if (retry.frame.active())
            finish(retry.frame);
        EXPECT_EQ(fj_test_live_noise_owners(), baseline);
        retry_event.passed();
    }

    TEST(GrainUpload, RetainsCapacityFailureAfterRelock) {
        require_cuda(cudaSetDevice(0));
        require_cuda(cudaFree(nullptr));
        CUcontext context = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
        ASSERT_NE(context, nullptr);
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, context};
        std::string diagnostic;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({key.deviceId, 1024}, diagnostic);
        ASSERT_NE(ledger, nullptr) << diagnostic;
        ASSERT_TRUE(ledger->bind_or_validate_cap(4, diagnostic)) << diagnostic;
        JuicerCuda::Resources resources(key, 1, ledger);
        const std::array<std::uint8_t, 8> bytes{};
        const JuicerCuda::StaticNoiseInput input{bytes, bytes, bytes, 1, 1, 1, 1, 1, 1, 1};
        JuicerCuda::Failure failure;
        ASSERT_FALSE(JuicerCuda::ensure_grain_static_assets_uploaded(resources, input, nullptr, failure));
        EXPECT_EQ(failure.status.category, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(failure.status.api, FJ_API_NONE);
        EXPECT_EQ(failure.status.native_code, 0);
        EXPECT_EQ(failure.diagnostic, "device_cap_exceeded | Wang upload failed: device_cap_exceeded");
        EXPECT_EQ(ledger->snapshot().chargedBytes, 0);
        EXPECT_EQ(resources.stbnData, nullptr);
        EXPECT_EQ(resources.wangTilesData, nullptr);
    }

    TEST_F(GrainScratch, GaussianPreparedArraysMatchFrozenWordsAndWarmHitsDoNoConstruction) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        const std::array pixelSizes{2.0f, 6.0f};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * pixelSizes.size());
        for (const auto route : routes)
            for (const auto pixelSize : pixelSizes)
                ids.push_back(route_case_key(route) + "/pixel-" +
                              std::to_string(std::bit_cast<std::uint32_t>(pixelSize)));
        JuicerTests::GtestCohort cases("route-pixel-size", ids);
        for (auto route : routes)
            for (float pixelSize : pixelSizes) {
                auto row = cases.case_event(
                    route_case_key(route) + "/pixel-" +
                    std::to_string(std::bit_cast<std::uint32_t>(pixelSize)));
                Inputs inputs(runtime, 64, 48, route, {.pixelSize = pixelSize});
                auto frame = prepare(inputs);
                const auto kernels = frame.visual_grain_resources();
                ASSERT_TRUE(kernels.active);
                const auto check = [](const auto& kernel) {
                    if (!kernel.weights)
                        return;
                    const auto expected = gaussian_kernel_fixture_words(JUICER_GAUSSIAN_FIXTURE_PATH, kernel.radius, kernel.sigma);
                    std::vector<float> actual(expected.size());
                    require_cuda(cudaMemcpy(actual.data(), kernel.weights, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        require(std::bit_cast<std::uint32_t>(actual[i]) == expected[i], "all grain Gaussian words match independent native coefficients");
                };
                for (const auto& kernel : kernels.correlation)
                    check(kernel);
                for (const auto& layer : kernels.dyeCloud)
                    for (const auto& kernel : layer)
                        check(kernel);
                finish(frame);
                const auto acquisitions = fj_test_gaussian_kernel_acquires();
                auto warm = prepare(inputs);
                EXPECT_EQ(fj_test_gaussian_kernel_acquires(), acquisitions);
                const auto reused = warm.visual_grain_resources();
                for (std::size_t c = 0; c < 3; ++c) {
                    EXPECT_EQ(reused.correlation[c].weights, kernels.correlation[c].weights);
                    for (std::size_t layer = 0; layer < 3; ++layer)
                        EXPECT_EQ(reused.dyeCloud[layer][c].weights, kernels.dyeCloud[layer][c].weights);
                }
                finish(warm);
                row.passed();
            }
    }

    TEST_F(GrainScratch, ReusesFinishedRawCorrectionPlanesWithoutDedicatedGrainAllocations) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            Inputs inputs(runtime, 128, 96, route);
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
            row.passed();
        }
    }

    TEST_F(GrainScratch, FilmGammaRebuildReusesExactCurveAndLayerDeviceStorage) {
        JuicerTests::GtestCohort cases("route-layer-mode", curve_storage_case_ids());
        for (const auto route : kCurveStorageRoutes) {
            for (bool layers : kCurveStorageLayerModes) {
                auto row = cases.case_event(curve_storage_case_key(route, layers));
                GrainVariant variant;
                variant.layers = layers;
                Inputs before(runtime, 64, 48, route, variant);
                variant.filmGamma = 1.1f;
                Inputs changed(runtime, 64, 48, route, variant);
                EXPECT_NE(before.filmDevelopmentHash, changed.filmDevelopmentHash);
                EXPECT_NE(before.recipeHash, changed.recipeHash);
                EXPECT_EQ(before.normalizedDensityCurvesHash, changed.normalizedDensityCurvesHash);
                EXPECT_EQ(before.densityCurvesLayersHash, changed.densityCurvesLayersHash);
                CurvePointers firstCurves;
                auto first = prepare(before, false, 0x475241494eull, &firstCurves);
                require_cuda(cudaStreamSynchronize(stream));
                ASSERT_EQ(firstCurves.layersActive, layers);
                finish(first);
                CurvePointers changedCurves;
                auto second = prepare(changed, false, 0x475241494eull, &changedCurves, &firstCurves);
                require_cuda(cudaStreamSynchronize(stream));
                EXPECT_EQ(changedCurves.layersActive, layers);
                EXPECT_EQ(firstCurves.normalizedHash, changedCurves.normalizedHash);
                EXPECT_EQ(firstCurves.layersHash, changedCurves.layersHash);
                finish(second);
                row.passed();
            }
        }
    }

    TEST_F(GrainScratch, DirAmountChangeOverwritesCurvesInPlace) {
        JuicerTests::GtestCohort cases("route-layer-mode", curve_storage_case_ids());
        for (const auto route : kCurveStorageRoutes) {
            for (bool layers : kCurveStorageLayerModes) {
                const auto caseKey = curve_storage_case_key(route, layers);
                SCOPED_TRACE(caseKey);
                auto row = cases.case_event(caseKey);
                GrainVariant variant;
                variant.layers = layers;
                const Inputs before(runtime, 64, 48, route, variant);
                variant.dirAmount = 0.5f;
                const Inputs changed(runtime, 64, 48, route, variant);

                FjPreparedStateView firstView{}, changedView{};
                FjErrorBuffer error{};
                ASSERT_EQ(before.state.view(firstView, error).category, FJ_STATUS_SUCCESS);
                ASSERT_EQ(changed.state.view(changedView, error).category, FJ_STATUS_SUCCESS);
                const auto& firstDir = firstView.dir_recipe;
                const auto& changedDir = changedView.dir_recipe;
                ASSERT_NE(firstDir.mode, FJ_DIR_INACTIVE);
                ASSERT_EQ(changedDir.mode, firstDir.mode);
                ASSERT_NE(firstDir.hash, 0u);
                ASSERT_NE(changedDir.hash, 0u);
                ASSERT_NE(firstDir.hash, changedDir.hash);

                const auto& firstDevelopment = firstView.film_development;
                const auto& changedDevelopment = changedView.film_development;
                const std::size_t count = firstDevelopment.log_exposure.count;
                ASSERT_GT(count, 0u);
                ASSERT_EQ(changedDevelopment.log_exposure.count, count);
                ASSERT_EQ(firstDevelopment.density_rgb.count, count * 3);
                ASSERT_EQ(changedDevelopment.density_rgb.count, count * 3);
                ASSERT_NE(firstDevelopment.density_curves_hash, 0u);
                ASSERT_EQ(changedDevelopment.density_curves_hash, firstDevelopment.density_curves_hash);
                ASSERT_EQ(firstView.grain.sublayers_active != 0, layers);
                ASSERT_EQ(changedView.grain.sublayers_active != 0, layers);
                ASSERT_EQ(firstDevelopment.density_layers_hash != 0, layers);
                ASSERT_EQ(changedDevelopment.density_layers_hash, firstDevelopment.density_layers_hash);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto& firstAxis = firstDir.compensated_axes_rgb[channel];
                    const auto& changedAxis = changedDir.compensated_axes_rgb[channel];
                    ASSERT_EQ(firstAxis.count, count);
                    ASSERT_EQ(changedAxis.count, count);
                    ASSERT_NE(std::memcmp(firstAxis.data, changedAxis.data, count * sizeof(float)), 0) << "RGB channel " << channel;
                    for (std::size_t layer = 0; layer < 3; ++layer) {
                        ASSERT_EQ(firstDevelopment.density_layers[layer][channel].count, layers ? count : 0u);
                        ASSERT_EQ(changedDevelopment.density_layers[layer][channel].count, layers ? count : 0u);
                    }
                }

                CurvePointers firstCurves;
                auto first = prepare(before, false, 0x475241494eull, &firstCurves);
                ASSERT_EQ(firstCurves.layersActive, layers);
                ASSERT_EQ(firstCurves.layersHash, firstDevelopment.density_layers_hash);
                finish(first);
                // The resident DIR identity differs from the second request's
                // identity checked by the focused resource warm predicate.
                auto second = prepare(changed, false, 0x475241494eull, nullptr, &firstCurves);
                finish(second);
                row.passed();
            }
        }
    }

    TEST_F(GrainScratch, KeepsDedicatedWorkspaceWithoutSpatialDir) {
        Inputs inputs(runtime, 64, 48, Spektrafilm::ScanRoute::NegativeDirectScan);
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
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        const std::array layerModes{false, true};
        const std::array sharedModes{false, true};
        const std::array times{12.0, 12.375};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * layerModes.size() * sharedModes.size() * times.size() * 7);
        for (const auto route : routes)
            for (const auto layers : layerModes)
                for (const auto shared : sharedModes)
                    for (const auto time : times)
                        for (int debug = 0; debug <= 6; ++debug)
                            ids.push_back(route_case_key(route) + "/layers-" + (layers ? "on" : "off") +
                                          "/shared-" + (shared ? "on" : "off") + "/time-" +
                                          std::to_string(std::bit_cast<std::uint64_t>(time)) +
                                          "/debug-" + std::to_string(debug));
        JuicerTests::GtestCohort cases("route-layer-shared-time-debug", ids);
        for (auto route : routes) {
            for (bool layers : layerModes) {
                for (bool shared : sharedModes) {
                    // Only frame time changes between passes. Retain one host
                    // product per debug view; every CUDA frame remains fresh.
                    std::array<std::optional<Inputs>, 7> inputsByDebug;
                    for (double time : times) {
                        for (int debug = 0; debug <= 6; ++debug) {
                            auto row = cases.case_event(route_case_key(route) + "/layers-" + (layers ? "on" : "off") +
                                                        "/shared-" + (shared ? "on" : "off") + "/time-" +
                                                        std::to_string(std::bit_cast<std::uint64_t>(time)) +
                                                        "/debug-" + std::to_string(debug));
                            SCOPED_TRACE(testing::Message() << "route=" << static_cast<int>(route)
                                                            << " layers=" << layers << " shared=" << shared << " time=" << time << " debug=" << debug);
                            auto& retained = inputsByDebug[debug];
                            if (!retained) {
                                retained.emplace(runtime, 48, 32, route, GrainVariant{.layers = layers, .shared = shared, .debug = debug, .time = time});
                            } else {
                                retained->build_grain_frame(time);
                            }
                            const auto& inputs = *retained;
                            ASSERT_EQ(inputs.grainPayload.coefficients.debug_view, static_cast<std::uint32_t>(debug));
                            ASSERT_EQ(inputs.grain.frame0, 12);
                            ASSERT_EQ(std::bit_cast<std::uint32_t>(inputs.grain.frameAlpha),
                                      std::bit_cast<std::uint32_t>(static_cast<float>(time - 12.0)));
                            auto frame = prepare(inputs);
                            const auto work = frame.visual_grain_workspace(frame.workspace_lease());
                            ASSERT_TRUE(work.active);
                            EXPECT_EQ(work.layerWork != nullptr, layers);
                            EXPECT_EQ(work.sharedDelta != nullptr, shared && debug <= 1);
                            compare_grain(inputs, frame);
                            finish(frame);
                            row.passed();
                        }
                    }
                }
            }
        }
    }

    TEST_F(GrainScratch, ReusesOnlyWithinEligibleRouteAndRetainedTier) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::PositiveDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan,
                                Spektrafilm::ScanRoute::PositivePrintScan};
        const std::array pixelSizes{6.0f, 200.0f};
        std::vector<std::string> ids;
        ids.reserve(routes.size() * pixelSizes.size());
        for (const auto route : routes)
            for (const auto pixelSize : pixelSizes)
                ids.push_back(route_case_key(route) + "/pixel-" +
                              std::to_string(std::bit_cast<std::uint32_t>(pixelSize)));
        JuicerTests::GtestCohort cases("route-pixel-size", ids);
        for (auto route : routes) {
            for (float pixelSize : pixelSizes) {
                auto row = cases.case_event(route_case_key(route) + "/pixel-" +
                                            std::to_string(std::bit_cast<std::uint32_t>(pixelSize)));
                Inputs inputs(runtime, 64, 48, route, {.pixelSize = pixelSize});
                auto frame = prepare(inputs);
                auto lease = frame.workspace_lease();
                const auto before = frame.visual_grain_workspace(lease);
                const auto dir = frame.spatial_dir_scratch(lease);
                const bool negative = inputs.polarity == Spektrafilm::ProfilePolarity::Negative;
                const auto optics = frame.scanner_workspace(lease);
                EXPECT_EQ(before.deltaAccum, negative ? dir.rawCorrectionY : optics.aux);
                JuicerCuda::Failure error;
                if (dir.targetPlaneRoles.cachedLogRawPlanes == 3) {
                    require(frame.stage_spatial_dir_cached_log_raw_for_final_develop(lease, stream, error), error.diagnostic);
                }
                const auto after = frame.visual_grain_workspace(lease);
                ASSERT_TRUE(after.active);
                EXPECT_EQ(after.deltaAccum, before.deltaAccum);
                compare_grain(inputs, frame);
                finish(frame);
                row.passed();
            }
        }
    }

    TEST_F(GrainScratch, SurvivesResolutionChangesQueuedAbortAndContextRetirement) {
        const std::array widths{64, 320, 48, 128};
        std::vector<std::string> ids;
        ids.reserve(widths.size() + 1);
        for (const int width : widths)
            ids.push_back("width-" + std::to_string(width));
        ids.push_back("context-retirement-recreation");
        JuicerTests::GtestCohort cases("width-context-transition", ids);
        for (int width : widths) {
            auto row = cases.case_event("width-" + std::to_string(width));
            Inputs inputs(runtime, width, width / 2, Spektrafilm::ScanRoute::NegativePrintScan);
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
            row.passed();
        }
        auto retirement = cases.case_event("context-retirement-recreation");
        require_cuda(cudaStreamSynchronize(stream));
        std::string error;
        require(JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, error), error);
        Inputs inputs(runtime, 96, 64, Spektrafilm::ScanRoute::NegativeDirectScan);
        auto recreated = prepare(inputs);
        compare_grain(inputs, recreated);
        finish(recreated);
        retirement.passed();
    }

    TEST_F(GrainScratch, RejectsConcurrentRetainedLeasesAndOrdersLaterStreamReuse) {
        Inputs inputs(runtime, 96, 64, Spektrafilm::ScanRoute::NegativePrintScan);
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
        require(JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, error), error);
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
        Inputs inputs(runtime, 128, 96, Spektrafilm::ScanRoute::NegativePrintScan);
        auto first = prepare(inputs);
        finish(first);
        Inputs disabled(runtime, 128, 96, Spektrafilm::ScanRoute::NegativePrintScan, GrainVariant{.active = false});
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        snapshot.instanceToken.value = 0x475241494eull;
        snapshot.frameToken.value = nextIdentity;
        snapshot.snapshotId = nextIdentity++;
        snapshot.deviceContextKey = key;
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            disabled.uploadCoreHash, disabled.dirRecipeHash, disabled.scannerHash, 0);
        FrameCall call{JuicerCuda::borrow_root(runtime.borrow().cuda), true, key, snapshot, stream, {}, {}};
        std::string diagnostic;
        const bool ready = FrameFixture::with_prepared_frame_inputs(
            disabled.runtime.borrow().assets, disabled.state, disabled.facts, &call, &prepare_root_frame, diagnostic);
        require(ready, diagnostic.empty() ? call.failure.diagnostic : diagnostic);
        auto off = std::move(call.frame);
        EXPECT_FALSE(off.visual_grain_workspace(off.workspace_lease()).active);
        finish(off);
        auto on = prepare(inputs);
        compare_grain(inputs, on);
        finish(on);
    }

    TEST_F(GrainScratch, AdmittedTierPreservesGrainBindingAndOutput) {
        const std::array routes{Spektrafilm::ScanRoute::NegativeDirectScan,
                                Spektrafilm::ScanRoute::NegativePrintScan};
        std::vector<std::string> ids;
        ids.reserve(routes.size());
        for (const auto route : routes)
            ids.push_back(route_case_key(route));
        JuicerTests::GtestCohort cases("route", ids);
        for (auto route : routes) {
            auto row = cases.case_event(route_case_key(route));
            Inputs inputs(runtime, 96, 64, route);
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
            row.passed();
        }
    }

    TEST_F(GrainScratch, DiffusionKeepsDedicatedGrainScratch) {
        Inputs inputs(runtime, 64, 48, Spektrafilm::ScanRoute::NegativeDirectScan, {.cameraDiffusion = true});
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

    TEST_F(GrainScratch, RetiresOnlyEligibleInstanceMembershipAcrossExactContexts) {
        constexpr std::uint64_t firstToken = 0x475241494eull;
        constexpr std::uint64_t secondToken = firstToken + 1;
        using Observer = JuicerProcess::TestSupport::RootLifetimeObserver;
        auto& root = JuicerCuda::borrow_root(runtime.borrow().cuda);
        Inputs inputs(runtime, 64, 48, Spektrafilm::ScanRoute::NegativeDirectScan);
        const auto originalKey = key;
        auto* const originalStream = stream;
        {
            auto first = prepare(inputs, true, firstToken);
            finish(first);
        }
        {
            auto second = prepare(inputs, true, secondToken);
            finish(second);
        }
        require_cuda(cudaStreamSynchronize(stream));
        const auto before = Observer::grain_snapshot(root, key, firstToken, secondToken);
        ASSERT_EQ(before.members, 2u);
        ASSERT_GT(before.grainRecords, 0u);
        ASSERT_FALSE(before.frame.expired());

        CUcontext other = nullptr;
        require(cuCtxCreate(&other, nullptr, 0, 0) == CUDA_SUCCESS, "create fixture context");
        key = {0, other};
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        {
            auto foreign = prepare(inputs, true, firstToken);
            finish(foreign);
        }
        require_cuda(cudaStreamSynchronize(stream));
        const auto otherBefore = Observer::grain_snapshot(root, key, firstToken, secondToken);
        require(otherBefore.members == 1 && otherBefore.grainRecords != 0, "other context lacks grain ownership");

        EXPECT_EQ(fj_cuda_retire_instance(runtime.borrow().cuda, firstToken, nullptr).category, FJ_STATUS_SUCCESS);
        const auto remaining = Observer::grain_snapshot(root, originalKey, firstToken, secondToken);
        const auto otherAfter = Observer::grain_snapshot(root, key, firstToken, secondToken);
        EXPECT_FALSE(remaining.first);
        EXPECT_TRUE(remaining.second);
        EXPECT_EQ(remaining.members, 1u);
        EXPECT_EQ(remaining.epoch, before.epoch);
        EXPECT_EQ(remaining.grain.lock().get(), before.grain.lock().get());
        EXPECT_EQ(remaining.frame.lock().get(), before.frame.lock().get());
        EXPECT_EQ(remaining.grainRecords, before.grainRecords);
        EXPECT_EQ(otherAfter.members, 0u);
        EXPECT_TRUE(otherAfter.grain.expired());
        EXPECT_EQ(otherAfter.frame.lock().get(), otherBefore.frame.lock().get());
        EXPECT_EQ(otherAfter.epoch, otherBefore.epoch);

        // Last primary-context member is retired while the other exact context
        // is current. Its physical ownership must enter deferred destruction,
        // without reentering Root or retiring either context implicitly.
        const auto chargedBefore = otherAfter.ledger.chargedBytes;
        EXPECT_EQ(fj_cuda_retire_instance(runtime.borrow().cuda, secondToken, nullptr).category, FJ_STATUS_SUCCESS);
        const auto deferred = Observer::grain_snapshot(root, originalKey, firstToken, secondToken);
        EXPECT_EQ(deferred.members, 0u);
        EXPECT_TRUE(deferred.grain.expired());
        EXPECT_EQ(deferred.frame.lock().get(), before.frame.lock().get());
        EXPECT_EQ(deferred.epoch, before.epoch);
        EXPECT_EQ(deferred.ledger.chargedBytes, chargedBefore);
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> live;
        JuicerCuda::ResourceManager::registry_snapshot_context_keys(live);
        EXPECT_NE(std::find(live.begin(), live.end(), originalKey), live.end());
        EXPECT_NE(std::find(live.begin(), live.end(), key), live.end());

        std::string diagnostic;
        require(root.retire_idle_context(0, other, diagnostic), diagnostic);
        require_cuda(cudaStreamDestroy(stream));
        require(cuCtxDestroy(other) == CUDA_SUCCESS, "destroy fixture context");
        require(cuCtxSetCurrent(static_cast<CUcontext>(originalKey.contextOpaque)) == CUDA_SUCCESS, "restore fixture context");
        key = originalKey;
        stream = originalStream;
        require(root.retire_idle_context(0, key.contextOpaque, diagnostic), diagnostic);
        contextRetired = true;
        const auto retired = Observer::grain_snapshot(root, key, firstToken, secondToken);
        EXPECT_EQ(retired.ledger.chargedBytes, 0u);
        EXPECT_EQ(retired.ledger.recordCount, 0u);
        EXPECT_TRUE(before.frame.expired());
    }

    TEST_F(GrainScratch, ResetRetiresDeferredOwnershipAfterFixtureContextDestruction) {
        constexpr std::uint64_t primaryToken = 0x475241494eull;
        constexpr std::uint64_t foreignToken = primaryToken + 1;
        using Observer = JuicerProcess::TestSupport::RootLifetimeObserver;
        auto& root = JuicerCuda::borrow_root(runtime.borrow().cuda);
        Inputs inputs(runtime, 32, 24, Spektrafilm::ScanRoute::NegativeDirectScan);
        const auto primaryKey = key;
        auto* const primaryStream = stream;
        {
            auto frame = prepare(inputs, true, primaryToken);
            finish(frame);
        }
        require_cuda(cudaStreamSynchronize(stream));
        const auto primary = Observer::grain_snapshot(root, key, primaryToken, foreignToken);
        CUcontext other = nullptr;
        require(cuCtxCreate(&other, nullptr, 0, 0) == CUDA_SUCCESS, "create reset fixture context");
        key = {0, other};
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        {
            auto frame = prepare(inputs, true, foreignToken);
            finish(frame);
        }
        require_cuda(cudaStreamSynchronize(stream));
        require_cuda(cudaStreamDestroy(stream));
        const auto foreignKey = key;
        require(cuCtxSetCurrent(static_cast<CUcontext>(primaryKey.contextOpaque)) == CUDA_SUCCESS, "restore primary fixture context");
        key = primaryKey;
        stream = primaryStream;
        EXPECT_EQ(fj_cuda_retire_instance(runtime.borrow().cuda, foreignToken, nullptr).category, FJ_STATUS_SUCCESS);
        const auto deferred = Observer::grain_snapshot(root, foreignKey, primaryToken, foreignToken);
        EXPECT_TRUE(deferred.grain.expired());
        EXPECT_GT(deferred.ledger.chargedBytes, primary.ledger.chargedBytes);
        // Only this fixture owns 'other'. Its actual destruction establishes the
        // reset fact; no live host context is reset and no driver fault is claimed.
        require(cuCtxDestroy(other) == CUDA_SUCCESS, "destroy owned reset fixture context");
        std::string diagnostic;
        require(root.retire_reset_context(0, other, diagnostic), diagnostic);
        const auto after = Observer::grain_snapshot(root, key, primaryToken, foreignToken);
        EXPECT_EQ(after.ledger.chargedBytes, primary.ledger.chargedBytes);
        EXPECT_EQ(after.ledger.recordCount, primary.ledger.recordCount);
        EXPECT_EQ(after.epoch, primary.epoch);
        EXPECT_TRUE(after.first);
        EXPECT_EQ(after.frame.lock().get(), primary.frame.lock().get());
        EXPECT_EQ(after.grain.lock().get(), primary.grain.lock().get());
    }

    TEST_F(GrainScratch, ResetRetiresReusedWorkspace) {
        Inputs inputs(runtime, 64, 48, Spektrafilm::ScanRoute::NegativePrintScan);
        auto frame = prepare(inputs);
        compare_grain(inputs, frame);
        finish(frame);
        require_cuda(cudaStreamSynchronize(stream));
        std::string error;
        require(JuicerCuda::borrow_root(runtime.borrow().cuda).retire_reset_context(0, key.contextOpaque, error), error);
        contextRetired = true;
    }

    // Explicit maintenance benchmark, excluded from ordinary CTest runs.
    TEST_F(GrainScratch, DISABLED_SixKMemoryAndTiming) {
        Inputs inputs(runtime, 6048, 4032, Spektrafilm::ScanRoute::NegativePrintScan);
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
        require(JuicerCuda::pack_visual_grain_payload(inputs.grainPayload,
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

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    JuicerTests::CaseLog case_log;
    const int result = RUN_ALL_TESTS();
    case_log.finish();
    return result;
} catch (...) {
    return 1;
}
