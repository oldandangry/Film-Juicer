#include "production_state_contract.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "case_results.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_owner.h"

namespace {
    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr int kPadding = 5;
    constexpr float kSourceCanary = -7.0f;
    constexpr float kDestinationCanary = -8.0f;
    constexpr std::uint64_t kSessionSeed = 0x20260923;
    constexpr std::uint64_t kInstanceToken = 0x641207;
    constexpr std::uint64_t kClipToken = 0x5312;
    constexpr std::uint64_t kFrameToken = 37;

    void require(bool value, std::string_view diagnostic) {
        if (!value)
            throw std::runtime_error(std::string(diagnostic));
    }
    void check(FjStatus status, std::string_view operation, const FjErrorBuffer& error) {
        if (status.category != FJ_STATUS_SUCCESS) {
            const std::string detail = error.data && error.length ? std::string(error.data, error.length) : "no diagnostic";
            throw std::runtime_error(std::string(operation) + ": " + detail);
        }
    }
    void cuda_check(cudaError_t status, std::string_view operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
    void driver_check(CUresult status, std::string_view operation) {
        if (status != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(status, &name);
            throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver failure"));
        }
    }
    class ContextRestore final {
    public:
        explicit ContextRestore(CUcontext context) noexcept : original_(context) {}
        ~ContextRestore() {
            if (!restored_)
                (void)cuCtxSetCurrent(original_);
        }
        void restore() {
            driver_check(cuCtxSetCurrent(original_), "restore original context");
            CUcontext current = nullptr;
            driver_check(cuCtxGetCurrent(&current), "inspect restored context");
            require(current == original_, "original context was not restored");
            restored_ = true;
        }

    private:
        CUcontext original_;
        bool restored_ = false;
    };
    class DeviceBuffers final {
    public:
        DeviceBuffers() = default;
        DeviceBuffers(const DeviceBuffers&) = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;
        ~DeviceBuffers() {
            if (stream)
                (void)cudaStreamSynchronize(stream);
            if (destination)
                (void)cudaFree(destination);
            if (source)
                (void)cudaFree(source);
            if (stream)
                (void)cudaStreamDestroy(stream);
        }
        void close() {
            cuda_check(cudaStreamSynchronize(stream), "complete stream before buffer release");
            auto* destinationOwner = destination;
            destination = nullptr;
            cuda_check(cudaFree(destinationOwner), "release destination");
            auto* sourceOwner = source;
            source = nullptr;
            cuda_check(cudaFree(sourceOwner), "release source");
            auto* streamOwner = stream;
            stream = nullptr;
            cuda_check(cudaStreamDestroy(streamOwner), "release stream");
        }
        cudaStream_t stream = nullptr;
        float* source = nullptr;
        float* destination = nullptr;
    };
    class InstanceOwner final {
    public:
        InstanceOwner() = default;
        InstanceOwner(const InstanceOwner&) = delete;
        InstanceOwner& operator=(const InstanceOwner&) = delete;
        ~InstanceOwner() {
            FjErrorBuffer ignored{nullptr, 0, 0};
            if (instance_)
                (void)fj_test_instance_destroy(instance_, &ignored);
        }
        void create(FjErrorBuffer& error) {
            const auto status = fj_test_instance_create(&instance_, &error);
            check(status, "create instance", error);
            require(instance_ != nullptr, "created instance was null");
        }
        FjInstance* get() const noexcept {
            return instance_;
        }
        void close(FjErrorBuffer& error) {
            require(instance_ != nullptr, "instance already consumed");
            auto* owner = instance_;
            instance_ = nullptr;
            check(fj_test_instance_destroy(owner, &error), "destroy instance", error);
            require(instance_ == nullptr, "consumed instance slot remained populated");
        }

    private:
        FjInstance* instance_ = nullptr;
    };
    class StateOwner final {
    public:
        StateOwner() = default;
        StateOwner(const StateOwner&) = delete;
        StateOwner& operator=(const StateOwner&) = delete;
        ~StateOwner() {
            FjErrorBuffer ignored{nullptr, 0, 0};
            if (state_)
                (void)fj_test_prepared_state_release(state_, &ignored);
        }
        void take(FjPreparedState*& state) noexcept {
            state_ = state;
            state = nullptr;
        }
        FjPreparedState* get() const noexcept {
            return state_;
        }
        void release(FjErrorBuffer& error) {
            require(state_ != nullptr, "prepared state already consumed");
            auto* owner = state_;
            state_ = nullptr;
            check(fj_test_prepared_state_release(owner, &error), "release prepared state", error);
            require(state_ == nullptr, "consumed prepared state slot remained populated");
        }

    private:
        FjPreparedState* state_ = nullptr;
    };
    struct Completion {
        FjCuda* cuda = nullptr;
        unsigned calls = 0;
        bool exact = false;
    };
    FjCallCompletionResult complete(void* user, const FjCallCompletionInput* input) noexcept {
        auto& state = *static_cast<Completion*>(user);
        ++state.calls;
        if (!input)
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        try {
            JuicerCuda::check_native_call_admission(state.cuda);
            state.exact = input->render_outcome_valid == 1 && input->has_inspected_context == 1 &&
                          input->recovery_scope_entered == 1 && input->force_fatal == 0 && input->abort_failed == 0 &&
                          input->outcome.status.category == FJ_STATUS_SUCCESS;
        } catch (...) {
            state.exact = false;
        }
        return {input->outcome, state.exact ? 0u : 1u};
    }
    struct SubmissionObservation {
        unsigned calls = 0;
        FjSubmission submission{};
        std::uint32_t reused = 0;
    };
    thread_local SubmissionObservation* currentSubmission = nullptr;
    void trace_submission(void*, const FjSubmission* submission, std::uint32_t reused) noexcept {
        if (currentSubmission && submission) {
            ++currentSubmission->calls;
            currentSubmission->submission = *submission;
            currentSubmission->reused = reused;
        }
    }
    class SubmissionTrace final {
    public:
        explicit SubmissionTrace(SubmissionObservation& observation) {
            require(currentSubmission == nullptr, "nested submission trace");
            currentSubmission = &observation;
        }
        ~SubmissionTrace() {
            currentSubmission = nullptr;
        }
        SubmissionTrace(const SubmissionTrace&) = delete;
        SubmissionTrace& operator=(const SubmissionTrace&) = delete;
    };
    std::vector<float> source_pixels(int components) {
        const int pitch = kWidth * components + kPadding;
        std::vector<float> source(static_cast<std::size_t>(pitch) * kHeight, kSourceCanary);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(components);
                for (int c = 0; c < components; ++c) {
                    source[pixel + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
                }
            }
        }
        return source;
    }
    struct PixelImages {
        const std::vector<float>& output;
        const std::vector<float>& source;
        const std::vector<float>& expected;
    };
    std::vector<std::uint32_t> check_pixels(PixelImages images, int components) {
        const auto& [output, source, expected] = images;
        const int pitch = kWidth * components + kPadding;
        const auto lanes = static_cast<std::size_t>(components);
        const auto rowSamples = static_cast<std::size_t>(pitch);
        require(expected.size() == static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) * lanes, "expected pixel count");
        std::vector<std::uint32_t> words;
        words.reserve(expected.size());
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                for (int c = 0; c < components; ++c) {
                    const auto offset = static_cast<std::size_t>(y) * rowSamples + static_cast<std::size_t>(x) * lanes + static_cast<std::size_t>(c);
                    const auto index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kWidth) + static_cast<std::size_t>(x)) * lanes + static_cast<std::size_t>(c);
                    const float actual = output[offset], wanted = expected[index];
                    require(std::isfinite(actual) && std::isfinite(wanted) &&
                                std::abs(actual - wanted) <= 2.0e-4f + 3.0e-4f * std::abs(wanted),
                            "render differs from accepted pixel fixture");
                    if (c == 3)
                        require(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(source[offset]), "alpha bits changed");
                    words.push_back(std::bit_cast<std::uint32_t>(actual));
                }
            }
            for (int sample = kWidth * components; sample < pitch; ++sample) {
                const auto offset = static_cast<std::size_t>(y) * rowSamples + static_cast<std::size_t>(sample);
                require(std::bit_cast<std::uint32_t>(output[offset]) == std::bit_cast<std::uint32_t>(kDestinationCanary), "destination padding changed");
            }
        }
        return words;
    }
} // namespace

namespace JuicerCudaTest {
    void check_production_state_contract(FjRuntimeView runtime, const FjPreparationParams& params, std::uint64_t expectedInputIdentity, std::uint32_t route, const std::vector<float>& expectedPixels, const JuicerTests::Cohort& rows) {
        require(runtime.assets && runtime.cuda, "production state contract requires a live Runtime");
        require(route == FJ_ROUTE_NEGATIVE_DIRECT || route == FJ_ROUTE_NEGATIVE_PRINT, "production state contract route");
        const bool print = route == FJ_ROUTE_NEGATIVE_PRINT;
        require(params.scan_route == route && params.spectral_upsampling_mode == (print ? 1 : 0) &&
                    params.camera_auto_exposure_enabled == 0 && params.scanner_unsharp_mask[0] == 0.0 &&
                    params.scanner_unsharp_mask[1] == 0.0 && params.gate_weave_amount == 0.0,
                "production state contract typed inputs");
        const int components = print ? 4 : 3;
        const auto source = source_pixels(components);
        const std::size_t bytes = source.size() * sizeof(float);
        const std::string prefix = print ? "negative-print/" : "negative-direct/";
        const std::uint64_t token = kInstanceToken + route;
        std::array<char, 2048> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};

        driver_check(cuInit(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver_check(cuCtxGetCurrent(&original), "inspect original context");
        ContextRestore restore(original);
        cuda_check(cudaSetDevice(0), "select CUDA device");
        InstanceOwner producer;
        producer.create(error);
        InstanceOwner submission;
        submission.create(error);
        check(fj_test_instance_bind_identity(submission.get(), kSessionSeed, token, &error), "bind submission identity", error);
        struct Retirement final {
            Retirement(FjCuda& owner, std::uint64_t instanceToken) : cuda(&owner), token(instanceToken) {}
            FjCuda* cuda;
            std::uint64_t token;
            bool pending = true;
            ~Retirement() {
                if (pending) {
                    FjErrorBuffer ignored{nullptr, 0, 0};
                    (void)fj_cuda_retire_instance(cuda, token, &ignored);
                }
            }
            void close(FjErrorBuffer& error) {
                require(pending, "submission membership already retired");
                pending = false;
                check(fj_cuda_retire_instance(cuda, token, &error), "retire submission membership", error);
            }
        } retirement{*runtime.cuda, token};
        DeviceBuffers device;
        cuda_check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create production stream");
        cuda_check(cudaMalloc(&device.source, bytes), "allocate source");
        cuda_check(cudaMalloc(&device.destination, bytes), "allocate destination");
        cuda_check(cudaMemcpy(device.source, source.data(), bytes, cudaMemcpyHostToDevice), "upload source");

        const int pitch = kWidth * components + kPadding;
        const FjRect bounds{0, 0, kWidth, kHeight};
        FjFrame frame{};
        frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, static_cast<std::ptrdiff_t>(pitch) * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(components), FJ_DEPTH_FLOAT32};
        frame.destination = frame.source;
        frame.destination.address = reinterpret_cast<std::uintptr_t>(device.destination);
        frame.render_window = bounds;
        frame.full_frame_extent = bounds;
        frame.stream = reinterpret_cast<std::uintptr_t>(device.stream);
        frame.flags = FJ_FRAME_STREAM_PRESENT;
        frame.pixel_size_um = 35'000.0f / static_cast<float>(kWidth);
        frame.time_frames = 37.0;
        frame.frame_rate = 24.0;
        frame.session_seed = kSessionSeed;
        frame.clip_token = kClipToken;
        const FjExtent fullExtent{0, 0, kWidth, kHeight};
        std::vector<std::uint32_t> coldWords;
        std::uint64_t coldRecipe = 0;
        for (const bool warm : {false, true}) {
            auto row = rows.case_event(prefix + (warm ? "warm" : "cold"));
            std::vector<float> initial(source.size(), kDestinationCanary);
            std::vector<float> output(source.size(), kDestinationCanary);
            cuda_check(cudaMemcpyAsync(device.destination, initial.data(), bytes, cudaMemcpyHostToDevice, device.stream), "reset destination");

            FjCaptureStart capture{};
            check(fj_test_instance_begin_capture(producer.get(), FJ_CAPTURE_CHANGE, &capture, &error), "begin capture", error);
            require(capture.kind == FJ_CAPTURE_STARTED, "change capture did not start");
            FjSnapshotStart snapshot{};
            check(fj_test_instance_begin_snapshot(producer.get(), capture.capture, &snapshot, &error), "begin snapshot", error);
            require(snapshot.kind == FJ_SNAPSHOT_READY, "capture snapshot was not ready");
            std::uint32_t update = FJ_UPDATE_NONE;
            check(fj_test_instance_install(producer.get(), capture.capture, snapshot.write_revision, &params, expectedInputIdentity, &update, &error), "install snapshot", error);
            require(update == FJ_UPDATE_APPLIED, "capture snapshot was not applied");
            FjAdmission admission{};
            const FjStatus admitted = fj_test_instance_admit(producer.get(), runtime.assets, &admission, &error);
            StateOwner held;
            held.take(admission.state);
            check(admitted, "admit prepared state", error);
            require(admission.kind == FJ_ADMISSION_READY && held.get() != nullptr &&
                        admission.input_identity == expectedInputIdentity && admission.build == 1 && admission.route == route,
                    "admission identity, build or route differs from captured inputs");
            if (warm)
                producer.close(error);

            std::optional<FjPreparedStateView> view(std::in_place);
            check(fj_test_prepared_state_view(held.get(), &*view, &error), "inspect held state", error);
            FjPreparedFramePolicy policy{};
            check(fj_test_prepared_state_frame_policy(held.get(), &fullExtent, &policy, &error), "inspect held frame policy", error);
            require(view->route == route && view->recipe_hash == policy.recipe_hash, "held route or policy hash differs");
            if (warm)
                require(policy.recipe_hash == coldRecipe, "warm recipe differs from cold recipe");
            else
                coldRecipe = policy.recipe_hash;

            Completion completion{runtime.cuda};
            SubmissionObservation observed;
            JuicerCuda::CallScope scope(runtime.cuda, route, complete, &completion);
            FjCudaCallAccess access = scope.access();
            access.trace_submission = trace_submission;
            CUcontext before = nullptr, after = nullptr;
            driver_check(cuCtxGetCurrent(&before), "inspect context before held render");
            FjCallCompletionResult result{};
            {
                SubmissionTrace trace(observed);
                result = fj_test_prepared_state_render(runtime.assets, held.get(), access, &frame, &policy, submission.get(), kFrameToken, {}, &error);
            }
            view.reset();
            require(!view.has_value(), "held view did not expire before release");
            held.release(error);
            driver_check(cuCtxGetCurrent(&after), "inspect context after held render");
            require(before == after, "held render changed current context");
            require(result.outcome.status.category == FJ_STATUS_SUCCESS && result.fatal == 0 &&
                        completion.calls == 1 && completion.exact && !scope.recovery().pending,
                    scope.diagnostic().empty() ? "held render completion tuple failed" : scope.diagnostic());
            require(observed.calls == 1 && observed.submission.instance_token == token &&
                        observed.submission.frame_token == kFrameToken && observed.submission.submission_id == 1 &&
                        observed.reused == (warm ? 1u : 0u),
                    "held render submission identity or reuse differs");
            cuda_check(cudaStreamSynchronize(device.stream), "complete held render stream");
            cuda_check(cudaMemcpy(output.data(), device.destination, bytes, cudaMemcpyDeviceToHost), "read held render");
            const auto outputWords = check_pixels({.output = output, .source = source, .expected = expectedPixels}, components);
            if (warm) {
                require(outputWords == coldWords, "warm output differs from cold output");
                retirement.close(error);
                submission.close(error);
                device.close();
                restore.restore();
            } else {
                coldWords = outputWords;
            }
            row.passed();
        }
    }
} // namespace JuicerCudaTest
