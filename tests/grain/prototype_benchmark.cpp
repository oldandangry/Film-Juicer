#include "prototype_benchmark.h"
#include "film_reference.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaFilmPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "juicer_ofx_api.h"
#include "../ffi/prepared_frame_fixture.h"
#include "../ffi/prepared_product_fixture.h"
#include "../ofx/instance_host.h"

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_apply_visual_grain(
    const JuicerCuda::GrainPayload*, const JuicerCuda::GrainKernelPayload*, int, int, float*, float*, float*, float*, float*, float*, float*, float*, void*);
extern "C" FjStatus fj_test_ofx_instance_observe(OfxImageEffectHandle, FjInstanceObservation**, FjErrorBuffer*);

namespace GrainPrototype {
    namespace {
        constexpr int width = 3840;
        constexpr int height = 2160;
        constexpr std::size_t count = static_cast<std::size_t>(width) * height;
        constexpr std::size_t planeBytes = count * sizeof(float);
        namespace Fixture = JuicerCudaTest::PreparedFrameFixture;
        using Clock = std::chrono::steady_clock;

        void require(bool condition, const std::string& message) {
            if (!condition)
                throw std::runtime_error(message);
        }
        void cuda_check(cudaError_t status) {
            require(status == cudaSuccess, cudaGetErrorString(status));
        }
        double milliseconds(Clock::time_point begin) {
            return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        }

        FjVisualGrainParams current_grain_controls(std::string_view look) {
            require(look == "default" || look == "expensive" || look == "simple", "unsupported Current grain look");
            ofx_test::InstanceHost host;
            auto* plugin = OfxGetPlugin(0);
            require(plugin != nullptr, "missing OFX test plugin");
            plugin->setHost(host.binding());
            require(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "load OFX defaults");
            const auto unload = [plugin](OfxPlugin*) {
                static_cast<void>(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr));
            };
            std::unique_ptr<OfxPlugin, decltype(unload)> loaded(plugin, unload);
            auto* descriptor = host.descriptor_handle();
            require(plugin->mainEntry(kOfxActionDescribe, descriptor, nullptr, nullptr) == kOfxStatOK, "describe OFX defaults");
            require(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, descriptor, host.context_args(), nullptr) == kOfxStatOK,
                    "describe OFX default context");
            auto* effect = host.create_effect();
            require(plugin->mainEntry(kOfxActionCreateInstance, effect, nullptr, nullptr) == kOfxStatOK, "create OFX default instance");
            const auto destroy = [plugin](auto* handle) {
                static_cast<void>(plugin->mainEntry(kOfxActionDestroyInstance, handle, nullptr, nullptr));
            };
            std::unique_ptr<std::remove_pointer_t<OfxImageEffectHandle>, decltype(destroy)> instance(effect, destroy);
            if (look == "expensive") {
                host.set_value(effect, "GrainSublayersActive", {1});
                host.set_value(effect, "GrainSizeMixWeight", {1.0 / 3.0});
                host.set_value(effect, "GrainSizeMixWeightMid", {1.0 / 3.0});
                host.set_value(effect, "GrainSizeMixScale", {4.0});
                host.set_value(effect, "GrainChroma", {0.5});
            } else if (look == "simple") {
                host.set_value(effect, "GrainSublayersActive", {0});
                host.set_value(effect, "GrainSizeMixWeight", {0.0});
                host.set_value(effect, "GrainSizeMixWeightMid", {0.0});
                host.set_value(effect, "GrainBlur", {0.0});
                host.set_value(effect, "GrainBlurDyeCloudsUm", {0.0});
                host.set_value(effect, "GrainChroma", {1.0});
            }
            host.set_value(effect, "GrainActive", {1});
            auto* args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "GrainActive"});
            require(plugin->mainEntry(kOfxActionInstanceChanged, effect, args, nullptr) == kOfxStatReplyDefault,
                    "capture enabled OFX grain defaults");
            FjInstanceObservation* observation = nullptr;
            const auto status = fj_test_ofx_instance_observe(effect, &observation, nullptr);
            const auto release = [](FjInstanceObservation* value) {
                static_cast<void>(fj_test_instance_observation_release(value, nullptr));
            };
            std::unique_ptr<FjInstanceObservation, decltype(release)> owner(observation, release);
            FjInstanceObservationView view{};
            require(status.category == FJ_STATUS_SUCCESS && owner &&
                        fj_test_instance_observation_view(owner.get(), &view, nullptr).category == FJ_STATUS_SUCCESS &&
                        view.captured.grain.active == 1,
                    "read enabled OFX grain defaults");
            const auto defaults = view.captured.grain;
            owner.reset();
            require(plugin->mainEntry(kOfxActionDestroyInstance, instance.release(), nullptr, nullptr) == kOfxStatOK,
                    "destroy OFX default instance");
            require(loaded.release()->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "unload OFX defaults");
            return defaults;
        }
        class Buffer {
        public:
            explicit Buffer(std::size_t bytes) {
                cuda_check(cudaMalloc(&data, bytes));
            }
            ~Buffer() {
                (void)cudaFree(data);
            }
            Buffer(const Buffer&) = delete;
            Buffer& operator=(const Buffer&) = delete;
            float* data = nullptr;
        };
        class Event {
        public:
            Event() {
                cuda_check(cudaEventCreate(&event));
            }
            ~Event() {
                (void)cudaEventDestroy(event);
            }
            Event(const Event&) = delete;
            Event& operator=(const Event&) = delete;
            cudaEvent_t event = nullptr;
        };
        struct Run {
            JuicerTestSupport::RuntimeFixture runtime;
            JuicerTestSupport::PreparationInput controls;
            std::optional<JuicerTestSupport::PreparedStateHold> state;
            JuicerCuda::ResourceManager::DeviceContextKey key{};
            cudaStream_t stream = nullptr;
            std::filesystem::path path;
            FilmLaunch film = nullptr;
            void* filmOwner = nullptr;
            std::ostream* results = nullptr;
            bool capture = false;

            Run(const std::filesystem::path& resources, bool captureInput)
                : controls(), state(prepare(resources, captureInput)) {}

            Run(const std::filesystem::path& resources, const JuicerTestSupport::PreparationInput& input)
                : controls(input) {
                runtime.create(resources);
                state.emplace(JuicerTestSupport::prepare_product(runtime.borrow().assets, controls));
            }
            JuicerTestSupport::PreparedStateHold prepare(const std::filesystem::path& resources, bool captureInput) {
                // The OFX module and the stage fixture each own the process
                // Root while live. Close the defaults capture before opening
                // the fixture's dedicated stage owner.
                if (!captureInput)
                    controls.values.grain = current_grain_controls("default");
                runtime.create(resources);
                controls.film_key = "kodak_portra_400";
                controls.values.camera_film_format_long_edge_mm = 36.0f;
                controls.values.dir_couplers.amount = 0.0f;
                controls.values.scatter_halation = {};
                return JuicerTestSupport::prepare_product(runtime.borrow().assets, controls);
            }
            void open() {
                cuda_check(cudaSetDevice(0));
                cuda_check(cudaFree(nullptr));
                void* context = nullptr;
                std::string diagnostic;
                require(JuicerCuda::query_current_cuda_context(context, diagnostic), diagnostic);
                key = {0, context};
                cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }
            void close() {
                cuda_check(cudaStreamSynchronize(stream));
                std::string diagnostic;
                require(JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, diagnostic), diagnostic);
                cuda_check(cudaStreamDestroy(stream));
                stream = nullptr;
                state.reset();
                require(runtime.close().category == FJ_STATUS_SUCCESS, "runtime close failed");
            }
            ~Run() {
                if (stream) {
                    (void)cudaStreamSynchronize(stream);
                    std::string diagnostic;
                    (void)JuicerCuda::borrow_root(runtime.borrow().cuda).retire_idle_context(0, key.contextOpaque, diagnostic);
                    (void)cudaStreamDestroy(stream);
                }
            }
        };
        bool execute(void* user, const Fixture::Inputs& prepared, std::string& diagnostic) {
            auto& run = *static_cast<Run*>(user);
            require(prepared.state.dir_recipe.mode == FJ_DIR_INACTIVE, "G1 requires DIR off");
            JuicerCuda::Failure failure;
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = 0x473142454e4348ull;
            snapshot.frameToken.value = 1;
            snapshot.snapshotId = 1;
            snapshot.deviceContextKey = run.key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                prepared.state.upload_core_hash, prepared.state.dir_recipe.hash, prepared.state.scanner_hash, 0);
            const auto setupStart = Clock::now();
            auto frame = JuicerCuda::borrow_root(run.runtime.borrow().cuda).prepare_cuda_frame(run.key, snapshot, prepared.preparation, {}, run.stream, failure, prepared.hostPreparation);
            if (!frame.active()) {
                diagnostic = failure.diagnostic;
                return false;
            }
            const auto lease = frame.workspace_lease();
            if (!frame.stage_optical_workspace(lease, run.stream, failure)) {
                diagnostic = failure.diagnostic;
                frame.abort();
                return false;
            }
            try {
                if (run.results) {
                    cuda_check(cudaStreamSynchronize(run.stream));
                    *run.results << "current_frame_setup_ms," << milliseconds(setupStart) << '\n';
                }
                Buffer input(3 * planeBytes);
                if (run.capture) {
                    JuicerCuda::FilmPayloadPack payload;
                    require(JuicerCuda::pack_film_payloads(prepared.filmPayload, frame.focused_resources().film, nullptr, prepared.filmRouteCorrectionScale, payload, diagnostic), diagnostic);
                    // Exactly representable positive camera-linear stimulus, planar red/green/blue sensitivity.
                    std::vector<float> camera(3 * count);
                    for (std::size_t channel = 0; channel < 3; ++channel)
                        for (std::size_t i = 0; i < count; ++i)
                            camera[channel * count + i] = static_cast<float>(1 + (i * 17 + channel * 89) % 1024) / 512.0f;
                    Buffer deviceCamera(3 * planeBytes);
                    cuda_check(cudaMemcpyAsync(deviceCamera.data, camera.data(), 3 * planeBytes, cudaMemcpyHostToDevice, run.stream));
                    JuicerCuda::DirectPipelineRunParams params{};
                    params.width = width;
                    params.height = height;
                    params.filmRaw = payload.filmRaw;
                    params.filmExpose = payload.filmExposure;
                    params.filmDevelop = payload.filmDevelop;
                    JuicerCuda::CameraFilmLinearExposurePlanes cameraPlanes{deviceCamera.data, deviceCamera.data + count, deviceCamera.data + 2 * count, width};
                    cuda_check(juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
                        &params, cameraPlanes, input.data, input.data + count, input.data + 2 * count, run.stream));
                    cuda_check(cudaMemcpyAsync(camera.data(), input.data, 3 * planeBytes, cudaMemcpyDeviceToHost, run.stream));
                    cuda_check(cudaStreamSynchronize(run.stream));
                    require(!std::filesystem::exists(run.path), "developed input already exists");
                    std::ofstream file(run.path, std::ios::binary | std::ios::trunc);
                    require(static_cast<bool>(file), "cannot open developed input output");
                    file.write(reinterpret_cast<const char*>(camera.data()), static_cast<std::streamsize>(3 * planeBytes));
                    require(static_cast<bool>(file), "cannot write developed input output");
                } else {
                    const auto readStart = Clock::now();
                    std::vector<float> host(3 * count);
                    std::ifstream file(run.path, std::ios::binary | std::ios::ate);
                    require(file && file.tellg() == static_cast<std::streamoff>(3 * planeBytes), "wrong developed input size");
                    file.seekg(0);
                    file.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(3 * planeBytes));
                    require(static_cast<bool>(file), "cannot read developed input");
                    for (float value : host)
                        require(std::isfinite(value), "developed input is not finite");
                    *run.results << "input_read_ms," << milliseconds(readStart) << '\n';
                    const auto uploadStart = Clock::now();
                    cuda_check(cudaMemcpyAsync(input.data, host.data(), 3 * planeBytes, cudaMemcpyHostToDevice, run.stream));
                    cuda_check(cudaStreamSynchronize(run.stream));
                    *run.results << "input_upload_ms," << milliseconds(uploadStart) << '\n';
                    const auto work = frame.visual_grain_workspace(lease);
                    require(work.active, "missing grain workspace");
                    JuicerCuda::GrainPayload grain;
                    JuicerCuda::GrainKernelPayload kernels;
                    require(JuicerCuda::pack_visual_grain_payload(prepared.grainPayload, frame.visual_grain_resources(), grain, kernels, diagnostic), diagnostic);
                    require(grain.active == 1 && grain.frameIndex == 0 && grain.timeAlpha == 0,
                            "G1 Current recipe must be active at integer frame zero");
                    grain.frameUniforms = work.frameUniforms;
                    Buffer carrier(3 * planeBytes);
                    Event begin, end;
                    const std::array buffers{work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta};
                    std::size_t workspaceBytes = 0;
                    for (std::size_t i = 0; i < buffers.size(); ++i) {
                        const auto* endOfPrevious = buffers.data() + i;
                        if (buffers[i] != nullptr && std::find(buffers.data(), endOfPrevious, buffers[i]) == endOfPrevious)
                            workspaceBytes += planeBytes;
                    }
                    *run.results << "input_bytes," << 3 * planeBytes << "\ncarrier_bytes," << 3 * planeBytes
                                 << "\ncurrent_workspace_plane_bytes," << workspaceBytes
                                 << "\ncurrent_workspace_uniform_bytes," << (work.frameUniforms ? sizeof(*work.frameUniforms) : 0U)
                                 << "\ncurrent_seed_base," << grain.seedBase << "\ncurrent_seed_next," << grain.seedBaseNext
                                 << "\ncurrent_recipe_hash," << prepared.state.grain.hash
                                 << "\ncurrent_frame_index," << grain.frameIndex << "\ncurrent_amplitude," << grain.amplitude
                                 << "\ncurrent_sublayers," << grain.nSubLayers << "\n";
                    *run.results << "current_active," << grain.active << "\ncurrent_sublayers_active," << grain.sublayersActive
                                 << "\ncurrent_time_alpha," << grain.timeAlpha << "\ncurrent_origin," << grain.originX << ',' << grain.originY
                                 << "\ncurrent_stbn," << grain.stbnWidth << ',' << grain.stbnHeight << ',' << grain.stbnFrames
                                 << ',' << grain.stbnOffsetX << ',' << grain.stbnOffsetY << ',' << grain.stbnFrame
                                 << ',' << grain.stbnSessionSeed << ',' << grain.clipToken
                                 << "\ncurrent_wang," << grain.wangWidth << ',' << grain.wangHeight << ',' << grain.wangCount
                                 << ',' << grain.wangColors << ',' << grain.wangCellMm
                                 << "\ncurrent_breathing," << grain.breathingPeriodFrames << ',' << grain.breathingAmplitude
                                 << ',' << grain.breathingCellUmSmall << ',' << grain.breathingCellUmLarge << ','
                                 << grain.breathingMix << ',' << grain.breathingDriftUmPerFrame
                                 << "\ncurrent_clump," << grain.microStructure[0] << ',' << grain.microStructure[1] << ','
                                 << grain.clumpTemporalMix << ',' << grain.clumpMorphPeriodFrames << ',' << grain.pitchPx
                                 << "\ncurrent_filter_radii," << kernels.blurRadius << ',' << kernels.blurRadiusMid << ',' << kernels.blurRadiusCoarse
                                 << "\ncurrent_pixel_um," << grain.pixelSizeUm << "\ncurrent_chroma_mix," << grain.chromaMix
                                 << "\ncurrent_shared_weight," << grain.chromaSharedWeight << "\ncurrent_independent_weight," << grain.chromaIndWeight
                                 << "\ncurrent_size_fine," << grain.sizeMixWeightFine << "\ncurrent_size_mid," << grain.sizeMixWeightMid
                                 << "\ncurrent_size_coarse," << grain.sizeMixWeight << "\ncurrent_size_scale," << grain.sizeMixScale
                                 << "\ncurrent_size_gain," << grain.sizeMixGain << "\ncurrent_debug_view," << grain.debugView << '\n';
                    for (int channel = 0; channel < 3; ++channel) {
                        *run.results << "current_channel," << channel << ',' << grain.densityMin[channel] << ','
                                     << grain.densityMax[channel] << ',' << grain.uniformity[channel] << ','
                                     << grain.nParticles[channel] << ',' << grain.odParticle[channel] << '\n';
                        for (int layer = 0; layer < 3; ++layer) {
                            *run.results << "current_layer," << channel << ',' << layer << ',' << grain.densityMinLayers[layer][channel]
                                         << ',' << grain.densityMaxLayers[layer][channel] << ',' << grain.nParticlesLayers[layer][channel]
                                         << ',' << grain.odParticleLayers[layer][channel] << ',' << kernels.dyeRadius[layer][channel] << '\n';
                        }
                    }
                    auto apply = [&](bool current) {
                        if (current) {
                            cuda_check(juicer_cuda_apply_visual_grain(&grain, &kernels, width, height, carrier.data, carrier.data + count, carrier.data + 2 * count, work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta, run.stream));
                        } else {
                            cuda_check(run.film(run.filmOwner, carrier.data, carrier.data + count, carrier.data + 2 * count, run.stream));
                        }
                    };
                    auto reset_carrier = [&] {
                        cuda_check(cudaMemcpyAsync(carrier.data, input.data, 3 * planeBytes, cudaMemcpyDeviceToDevice, run.stream));
                    };
                    auto measure = [&](bool current, int block, int sample, bool warmup) {
                        const auto resetStart = Clock::now();
                        reset_carrier();
                        cuda_check(cudaStreamSynchronize(run.stream));
                        const double resetMs = milliseconds(resetStart);
                        cuda_check(cudaEventRecord(begin.event, run.stream));
                        const auto start = Clock::now();
                        apply(current);
                        const double submitMs = milliseconds(start);
                        cuda_check(cudaEventRecord(end.event, run.stream));
                        cuda_check(cudaEventSynchronize(end.event));
                        const double totalMs = milliseconds(start);
                        float deviceMs = 0;
                        cuda_check(cudaEventElapsedTime(&deviceMs, begin.event, end.event));
                        *run.results << (warmup ? "warmup" : "sample") << ',' << block << ',' << sample << ','
                                     << (current ? "current" : "film") << ',' << deviceMs << ',' << submitMs << ',' << totalMs << ',' << resetMs << '\n';
                    };
                    // One first-use observation is kept separate from warmup and samples.
                    measure(true, -1, 0, true);
                    measure(false, -1, 0, true);
                    for (int block = 0; block < 3; ++block) {
                        for (const bool isCurrent : {block != 1, block == 1}) {
                            for (int pass = 0; pass < 40; ++pass)
                                measure(isCurrent, block, pass, pass < 10);
                        }
                    }
                    // Observe both outputs outside timing and repeat Current at
                    // the same immutable input and frame identity.
                    std::vector<float> first(3 * count), second(3 * count);
                    auto download = [&](bool current, std::vector<float>& out) {
                        reset_carrier();
                        apply(current);
                        cuda_check(cudaMemcpyAsync(out.data(), carrier.data, 3 * planeBytes, cudaMemcpyDeviceToHost, run.stream));
                        cuda_check(cudaStreamSynchronize(run.stream));
                    };
                    download(true, first);
                    download(true, second);
                    std::size_t mismatches = 0;
                    for (std::size_t i = 0; i < first.size(); ++i)
                        if (std::bit_cast<std::uint32_t>(first[i]) != std::bit_cast<std::uint32_t>(second[i]))
                            ++mismatches;
                    *run.results << "current_repeat_bit_mismatches," << mismatches << '\n';
                    require(mismatches == 0, "current grain repeatability failed");
                    download(false, second);
                    *run.results << "driver_host_plane_bytes," << 9 * planeBytes << '\n';
                    for (const bool current : {true, false}) {
                        const auto& output = current ? first : second;
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            double sum = 0;
                            double sumSquares = 0;
                            double minimum = std::numeric_limits<double>::infinity();
                            double maximum = -std::numeric_limits<double>::infinity();
                            std::size_t changed = 0;
                            for (std::size_t i = channel * count; i < (channel + 1) * count; ++i) {
                                const double delta = static_cast<double>(output[i]) - host[i];
                                require(std::isfinite(delta), "nonfinite grain output");
                                sum += delta;
                                sumSquares += delta * delta;
                                minimum = std::min(minimum, delta);
                                maximum = std::max(maximum, delta);
                                changed += delta != 0;
                            }
                            require(changed != 0, "active grain left a density plane unchanged");
                            *run.results << "density_delta," << (current ? "current" : "film") << ',' << channel
                                         << ',' << changed << ',' << minimum << ',' << maximum << ',' << sum / count
                                         << ',' << std::sqrt(sumSquares / count) << '\n';
                        }
                    }
                }
                require(frame.finish(run.stream, failure), failure.diagnostic);
                cuda_check(cudaStreamSynchronize(run.stream));
                return true;
            } catch (...) {
                frame.abort();
                throw;
            }
        }
        // Same resource and carrier path order as the two explicit operations.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        void run(const std::filesystem::path& resources, const std::filesystem::path& path, bool capture, FilmLaunch film, void* owner, std::ostream* output) {
            const auto setupStart = Clock::now();
            Run context(resources, capture);
            context.path = path;
            context.capture = capture;
            context.film = film;
            context.filmOwner = owner;
            context.results = output;
            context.open();
            if (output)
                *output << "runtime_context_setup_ms," << milliseconds(setupStart) << '\n';
            Fixture::FrameFacts facts;
            facts.renderExtent = {0, 0, width, height};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.renderExtent;
            facts.pixelSizeUm = 9.375f;
            facts.filmFormatLongEdgeMm = 36.0f;
            facts.frameTime = 0.0;
            facts.frameRate = 24.0;
            facts.sessionSeed = 17;
            facts.clipToken = 23;
            std::string diagnostic;
            require(Fixture::with_prepared_frame_inputs(context.runtime.borrow().assets, *context.state, facts, &context, &execute, diagnostic), diagnostic);
            context.close();
        }

        using Json = nlohmann::json;

        struct CurrentCapture {
            Run& run;
            const Fixture::FrameFacts& facts;
            std::uint64_t submission;
            FilmPlanes planes;
            Json resolved;
        };

        bool render_current(void* user, const Fixture::Inputs& prepared, std::string& diagnostic) {
            auto& capture = *static_cast<CurrentCapture*>(user);
            auto& run = capture.run;
            require(prepared.state.dir_recipe.mode == FJ_DIR_INACTIVE, "research requires DIR off");
            JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
            snapshot.instanceToken.value = capture.facts.clipToken;
            snapshot.frameToken.value = capture.submission;
            snapshot.snapshotId = capture.submission;
            snapshot.deviceContextKey = run.key;
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                prepared.state.upload_core_hash, prepared.state.dir_recipe.hash, prepared.state.scanner_hash, 0);
            JuicerCuda::Failure failure;
            auto frame = JuicerCuda::borrow_root(run.runtime.borrow().cuda).prepare_cuda_frame(run.key, snapshot, prepared.preparation, {}, run.stream, failure, prepared.hostPreparation);
            if (!frame.active()) {
                diagnostic = failure.diagnostic;
                return false;
            }
            const auto lease = frame.workspace_lease();
            if (!frame.stage_optical_workspace(lease, run.stream, failure)) {
                diagnostic = failure.diagnostic;
                frame.abort();
                return false;
            }
            try {
                const auto work = frame.visual_grain_workspace(lease);
                JuicerCuda::GrainPayload grain;
                JuicerCuda::GrainKernelPayload kernels;
                const bool bound = work.active && JuicerCuda::pack_visual_grain_payload(prepared.grainPayload,
                                                                                        frame.visual_grain_resources(),
                                                                                        grain,
                                                                                        kernels,
                                                                                        diagnostic);
                require(bound, "cannot bind Current grain: " + diagnostic);
                grain.frameUniforms = work.frameUniforms;
                const auto& extent = capture.facts.fullFrameExtent;
                require(grain.active == 1 && grain.pixelSizeUm == capture.facts.pixelSizeUm, "incorrect Current frame geometry");
                cuda_check(juicer_cuda_apply_visual_grain(&grain, &kernels, extent.width, extent.height, capture.planes.cyan, capture.planes.magenta, capture.planes.yellow, work.filterTemp, work.scaleWork, work.deltaAccum, work.layerWork, work.sharedDelta, run.stream));
                const auto& recipe = prepared.state.grain;
                auto& resolved = capture.resolved;
                resolved = {
                    {"recipe_hash", recipe.hash},
                    {"layer_hash", recipe.layers_hash},
                    {"seed", grain.seedBase},
                    {"next_seed", grain.seedBaseNext},
                    {"frame_index", grain.frameIndex},
                    {"fraction", grain.timeAlpha},
                    {"route", prepared.state.route},
                    {"positive", grain.positiveFilm},
                    {"origin", {grain.originX, grain.originY}},
                    {"sublayers_active", grain.sublayersActive},
                    {"sublayers", grain.nSubLayers},
                    {"amount", grain.amplitude},
                    {"debug_view", grain.debugView},
                    {"particle_area_um2", recipe.particle_area_um2},
                    {"dye_cloud_blur_um", recipe.dye_cloud_blur_um},
                    {"correlation_sigma_px", recipe.correlation_sigma_px},
                    {"chroma", {grain.chromaMix, grain.chromaSharedWeight, grain.chromaIndWeight}},
                    {"band_weights", {grain.sizeMixWeightFine, grain.sizeMixWeightMid, grain.sizeMixWeight}},
                    {"band_scale", grain.sizeMixScale},
                    {"band_gain", grain.sizeMixGain},
                    {"filter_radii", {kernels.blurRadius, kernels.blurRadiusMid, kernels.blurRadiusCoarse}},
                    {"width_height_frames_stbn", {grain.stbnWidth, grain.stbnHeight, grain.stbnFrames}},
                    {"offset_x_y_frame_stbn", {grain.stbnOffsetX, grain.stbnOffsetY, grain.stbnFrame}},
                    {"wang", {grain.wangWidth, grain.wangHeight, grain.wangCount, grain.wangColors, grain.wangCellMm}},
                    {"breathing", {grain.breathingPeriodFrames, grain.breathingAmplitude, grain.breathingCellUmSmall, grain.breathingCellUmLarge, grain.breathingMix, grain.breathingDriftUmPerFrame}},
                    {"clump", {grain.microStructure[0], grain.microStructure[1], grain.clumpTemporalMix, grain.clumpMorphPeriodFrames, grain.pitchPx}},
                    {"records", Json::array()},
                };
                for (int record = 0; record < kRecords; ++record) {
                    Json layers = Json::array();
                    for (int layer = 0; layer < 3; ++layer) {
                        layers.push_back({grain.densityMinLayers[layer][record], grain.densityMaxLayers[layer][record], grain.nParticlesLayers[layer][record], grain.odParticleLayers[layer][record], kernels.dyeRadius[layer][record]});
                    }
                    resolved["records"].push_back({{"density_min_max", {grain.densityMin[record], grain.densityMax[record]}},
                                                   {"uniformity", grain.uniformity[record]},
                                                   {"particles", grain.nParticles[record]},
                                                   {"particle_density", grain.odParticle[record]},
                                                   {"layers", layers}});
                }
                require(frame.finish(run.stream, failure), failure.diagnostic);
                cuda_check(cudaStreamSynchronize(run.stream));
                return true;
            } catch (...) {
                frame.abort();
                throw;
            }
        }

        std::uint32_t film_seed(const Fixture::FrameFacts& facts, std::uint64_t index) {
            // SplitMix64 permutation of frame/clip/session identity, truncated
            // to the upstream placement seed width. Coordinates do not enter.
            std::uint64_t word = (facts.sessionSeed ^ std::rotl(facts.clipToken, 17)) + (index + 1) * 0x9e3779b97f4a7c15ULL;
            word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
            word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
            return static_cast<std::uint32_t>(word ^ (word >> 31));
        }

        JuicerTestSupport::PreparationInput research_controls(const Json& request) {
            JuicerTestSupport::PreparationInput controls;
            controls.values.grain = current_grain_controls(request.at("look").get<std::string>());
            controls.film_key = request.at("stock").get<std::string>();
            controls.values.camera_film_format_long_edge_mm = request.at("long_edge_mm").get<float>();
            controls.values.film_gamma_factor = 1.0f;
            controls.values.dir_couplers.amount = 0.0f;
            controls.values.scatter_halation = {};
            controls.values.scan_route = request.at("route").get<std::uint32_t>();
            return controls;
        }

        void fill_research_input(std::vector<float>& host, const Fixture::FrameFacts& facts, const Json& sample) {
            const auto densities = sample.at("density").get<std::array<float, kRecords>>();
            const int w = facts.fullFrameExtent.width;
            const int h = facts.fullFrameExtent.height;
            const auto elements = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            const auto stimulus = sample.value("stimulus", "flat");
            require(stimulus == "flat" || stimulus == "edge" || stimulus == "patches", "invalid density stimulus");
            if (stimulus == "patches") {
                const auto values = sample.at("patches").get<std::vector<std::array<float, kRecords>>>();
                const int columns = sample.at("columns").get<int>();
                const int rows = sample.at("rows").get<int>();
                require(columns > 0 && columns <= w && rows > 0 && rows <= h &&
                            values.size() == static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows),
                        "invalid density patch layout");
                for (int row = 0; row < rows; ++row) {
                    for (int column = 0; column < columns; ++column) {
                        const auto index = static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(column);
                        const auto& cmy = values.at(index);
                        const int begin = column * w / columns;
                        const int end = (column + 1) * w / columns;
                        for (std::size_t record = 0; record < cmy.size(); ++record) {
                            require(std::isfinite(cmy[record]), "nonfinite patch density");
                            auto* plane = host.data() + record * elements;
                            for (int y = row * h / rows; y < (row + 1) * h / rows; ++y)
                                std::fill(plane + static_cast<std::size_t>(y * w + begin),
                                          plane + static_cast<std::size_t>(y * w + end),
                                          cmy[record]);
                        }
                    }
                }
                return;
            }
            for (std::size_t record = 0; record < densities.size(); ++record) {
                require(std::isfinite(densities[record]), "nonfinite input density");
                auto* plane = host.data() + record * elements;
                std::fill_n(plane, elements, densities[record]);
                if (stimulus == "flat")
                    continue;
                const int edge = w / 2 - w / 8 + static_cast<int>(facts.frameTime) % (w / 4);
                for (int y = 0; y < h; ++y) {
                    auto* row = plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
                    for (int x = 0; x < w; ++x) {
                        const float ramp = static_cast<float>(y) / static_cast<float>(h);
                        const float detail = ((x / 4 + y / 4) % 2 == 0) ? 0.05f : 0.0f;
                        row[x] += x < edge ? ramp * 0.2f : (0.25f + 0.08f * static_cast<float>(record) + detail);
                    }
                }
            }
        }
    } // namespace

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void capture_research(const std::filesystem::path& resources, const std::filesystem::path& requestPath, std::ostream& results) {
        std::ifstream requestFile(requestPath);
        require(static_cast<bool>(requestFile), "cannot open research request");
        auto request = Json::parse(requestFile);
        const auto engine = request.at("engine").get<std::string>();
        require(engine == "current" || engine == "film" || engine == "off", "invalid grain engine");
        const auto w = request.at("width").get<int>();
        const auto h = request.at("height").get<int>();
        const auto cropWidth = request.at("crop_width").get<int>();
        const auto cropHeight = request.at("crop_height").get<int>();
        const auto cropX = (w - cropWidth) / 2;
        const auto cropY = (h - cropHeight) / 2;
        require(w >= 8 && h >= 8 && w <= 4096 && h <= 4096 && cropWidth > 0 && cropHeight > 0 &&
                    cropWidth <= w && cropHeight <= h,
                "unsupported research frame/crop");
        Fixture::FrameFacts facts;
        facts.renderExtent = {0, 0, w, h};
        facts.fullFrameExtent = facts.renderExtent;
        facts.pixelDefinition = facts.renderExtent;
        facts.pixelSizeUm = request.at("pitch_um").get<float>();
        facts.filmFormatLongEdgeMm = request.at("long_edge_mm").get<float>();
        facts.frameRate = request.at("fps").get<double>();
        facts.sessionSeed = request.at("session_seed").get<std::uint64_t>();
        facts.clipToken = request.at("clip_token").get<std::uint64_t>();
        require(facts.pixelSizeUm > 0.0f && facts.frameRate == 24.0 && facts.clipToken != 0, "invalid research frame identity");
        const auto elements = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const auto rowBytes = static_cast<std::size_t>(w) * sizeof(float);
        const auto cropRowBytes = static_cast<std::size_t>(cropWidth) * sizeof(float);
        const auto cropElements = static_cast<std::size_t>(cropWidth) * static_cast<std::size_t>(cropHeight);
        const auto framesPath = requestPath.parent_path() / "frames.f32";
        const auto manifestPath = requestPath.parent_path() / "manifest.json";
        require(!std::filesystem::exists(framesPath) && !std::filesystem::exists(manifestPath), "research output exists");

        std::optional<Run> current;
        if (engine == "current") {
            current.emplace(resources, research_controls(request));
            current->open();
        } else {
            cuda_check(cudaSetDevice(0));
            cuda_check(cudaFree(nullptr));
        }
        auto* const stream = current ? current->stream : nullptr;
        DeviceFilmBank device;
        FilmTables tables{};
        FilmFrame film{};
        film.width = w;
        film.height = h;
        film.stride = w;
        film.enabled = engine == "film";
        if (film.enabled) {
            const auto size = request.at("size").get<float>();
            const auto amounts = request.at("amounts").get<std::array<float, kRecords>>();
            require(size >= 0.5f && size <= 4.0f, "Film size outside experiment");
            film.chromaFraction = request.value("chroma_fraction", 1.F);
            require(film.chromaFraction >= 0.F && film.chromaFraction <= 1.F, "Film chroma fraction outside experiment");
            for (std::size_t record = 0; record < amounts.size(); ++record) {
                require(amounts[record] >= 0.125f && amounts[record] <= 8.0f, "Film amount outside experiment");
                film.amounts[record] = amounts[record] / size;
            }
            film.pitch = facts.pixelSizeUm / size;
            const auto bankPath = request.at("bank").get<std::string>();
            const auto bank = read_film_bank(bankPath, 1.0f);
            tables = read_film_tables(request.value("tables", bankPath), film.pitch);
            cuda_check(device.upload(bank));
            request["footprint"] = film.pitch;
            request["resolved_chroma_fraction"] = film.chromaFraction;
            request["device_bank_bytes"] = bank.sums.size() * sizeof(float);
        }
        Buffer carrier(kRecords * elements * sizeof(float));
        FilmPlanes planes{carrier.data, carrier.data + elements, carrier.data + 2 * elements};
        std::vector<float> host(kRecords * elements);
        std::vector<float> crop(kRecords * cropElements);
        std::ofstream frames(framesPath, std::ios::binary);
        require(static_cast<bool>(frames), "cannot open research frame output");
        request["resolved"] = Json::array();
        std::uint64_t submission = 0;
        for (const auto& sample : request.at("requests")) {
            facts.frameTime = sample.at("time").get<double>();
            require(facts.frameTime >= 0.0 && facts.frameTime <= 1000000.0, "unsupported research frame time");
            fill_research_input(host, facts, sample);
            cuda_check(cudaMemcpyAsync(carrier.data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            if (current) {
                CurrentCapture capture{*current, facts, ++submission, planes, {}};
                std::string diagnostic;
                const bool rendered = Fixture::with_prepared_frame_inputs(current->runtime.borrow().assets, *current->state, facts, &capture, &render_current, diagnostic);
                require(rendered, "Current research preparation: " + diagnostic);
                request["resolved"].push_back(std::move(capture.resolved));
            } else {
                const auto index = static_cast<std::uint64_t>(std::floor(facts.frameTime));
                film.temporalFraction = static_cast<float>(facts.frameTime - std::floor(facts.frameTime));
                film.seed = film_seed(facts, index);
                film.nextSeed = film.temporalFraction == 0.0f ? 0 : film_seed(facts, index + 1);
                cuda_check(apply_film_reference(device, tables, film, planes, stream));
                request["resolved"].push_back({{"seed", film.seed}, {"next_seed", film.nextSeed}, {"fraction", film.temporalFraction}});
            }
            for (std::size_t record = 0; record < kRecords; ++record) {
                const auto origin = static_cast<std::size_t>(cropY) * static_cast<std::size_t>(w) + static_cast<std::size_t>(cropX);
                cuda_check(cudaMemcpy2DAsync(crop.data() + record * cropElements, cropRowBytes, carrier.data + record * elements + origin, rowBytes, cropRowBytes, static_cast<std::size_t>(cropHeight), cudaMemcpyDeviceToHost, stream));
            }
            cuda_check(cudaStreamSynchronize(stream));
            require(std::all_of(crop.begin(), crop.end(), [](float value) {
                        return std::isfinite(value);
                    }),
                    "nonfinite research output");
            frames.write(reinterpret_cast<const char*>(crop.data()), static_cast<std::streamsize>(crop.size() * sizeof(float)));
            require(static_cast<bool>(frames), "cannot write research frame");
        }
        frames.close();
        require(frames.good() && !request["resolved"].empty(), "incomplete research capture");
        if (current)
            current->close();
        request["crop_x"] = cropX;
        request["crop_y"] = cropY;
        std::ofstream manifest(manifestPath);
        manifest << request.dump(2) << '\n';
        manifest.close();
        require(manifest.good(), "cannot write research manifest");
        results << "research captured " << request["resolved"].size() << " full frames\n";
    }

    void capture_developed_input(const std::filesystem::path& resources, const std::filesystem::path& output) {
        run(resources, output, true, nullptr, nullptr, nullptr);
    }
    void benchmark(const std::filesystem::path& resources, const std::filesystem::path& input, FilmLaunch film, void* filmOwner, std::ostream& results) {
        require(film != nullptr, "missing Film launch");
        run(resources, input, false, film, filmOwner, &results);
    }
} // namespace GrainPrototype
