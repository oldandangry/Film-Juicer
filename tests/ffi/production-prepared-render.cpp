#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_owner.h"
#include "JuicerState.h"
#include "PreparedState.h"
#include "ProcessRoot.h"

namespace {

    using Json = nlohmann::json;

    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr int kPitchPadding = 5;
    constexpr float kSourceCanary = -7.0f;
    constexpr float kDestinationCanary = -8.0f;
    constexpr std::uint64_t kInstanceToken = 0x641207;
    constexpr std::uint64_t kSessionSeed = 0x20260923;
    constexpr std::uint64_t kClipToken = 0x5312;

    void require(bool condition, std::string_view message) {
        if (!condition) {
            throw std::runtime_error(std::string(message));
        }
    }

    void require_cuda(cudaError_t result, std::string_view operation) {
        if (result != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
        }
    }

    void require_driver(CUresult result, std::string_view operation) {
        if (result != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(result, &name);
            throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver failure"));
        }
    }

    struct ContextRestore final {
        CUcontext original = nullptr;
        bool restored = false;

        explicit ContextRestore(CUcontext value) noexcept : original(value) {}

        ~ContextRestore() {
            if (!restored) {
                (void)cuCtxSetCurrent(original);
            }
        }

        void restore() {
            require_driver(cuCtxSetCurrent(original), "restore original CUDA context");
            CUcontext current = nullptr;
            require_driver(cuCtxGetCurrent(&current), "check restored CUDA context");
            restored = current == original;
            require(restored, "original CUDA context was not restored");
        }
    };

    struct DeviceBuffers final {
        cudaStream_t stream = nullptr;
        float* source = nullptr;
        float* destination = nullptr;

        ~DeviceBuffers() {
            if (stream) {
                (void)cudaStreamSynchronize(stream);
            }
            if (destination) {
                (void)cudaFree(destination);
            }
            if (source) {
                (void)cudaFree(source);
            }
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }
    };

    class PreparedOwner final {
    public:
        PreparedOwner() = default;
        PreparedOwner(const PreparedOwner&) = delete;
        PreparedOwner& operator=(const PreparedOwner&) = delete;
        ~PreparedOwner() {
            release();
        }

        void create(FjAssets* assets, const ParamSnapshot& parameters, FjErrorBuffer& error) {
            const FjPreparationParams input = preparation_params(parameters);
            FjPreparedState* created = nullptr;
            const FjStatus status = fj_legacy_prepared_state_create(assets, &input, &created, &error);
            if (status.category != FJ_STATUS_SUCCESS || !created) {
                if (created) {
                    FjErrorBuffer ignored{nullptr, 0, 0};
                    (void)fj_legacy_prepared_state_release(created, &ignored);
                }
                throw std::runtime_error(error.length ? std::string(error.data, error.length) : "prepared state creation failed");
            }
            state_ = created;
        }

        FjPreparedState* get() const noexcept {
            return state_;
        }

        bool release() noexcept {
            if (!state_) {
                return true;
            }
            FjPreparedState* state = state_;
            state_ = nullptr;
            FjErrorBuffer ignored{nullptr, 0, 0};
            return fj_legacy_prepared_state_release(state, &ignored).category == FJ_STATUS_SUCCESS;
        }

    private:
        FjPreparedState* state_ = nullptr;
    };

    struct Case final {
        const char* id;
        const char* routeName;
        Spektrafilm::ScanRoute route;
        int reconstruction;
        int components;
    };

    constexpr std::array<Case, 2> kCases{{{"negative-direct", "negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3},
                                          {"negative-print", "negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4}}};

    ParamSnapshot parameters_for(const Case& test) {
        ParamSnapshot parameters;
        parameters.scanRoute = test.route;
        parameters.spectralUpsamplingMode = test.reconstruction;
        parameters.cameraAutoExposureEnabled = 0;
        parameters.scannerUnsharpMask = {0.0, 0.0};
        parameters.gateWeaveAmount = 0.0;
        return parameters;
    }

    std::vector<float> initial_source(const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        std::vector<float> source(static_cast<std::size_t>(pitch) * static_cast<std::size_t>(kHeight), kSourceCanary);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    source[pixel + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
                }
            }
        }
        return source;
    }

    std::vector<float> active_pixels(const std::vector<float>& pitched, const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        std::vector<float> result;
        result.reserve(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < kHeight; ++y) {
            const auto begin = pitched.begin() + static_cast<std::ptrdiff_t>(y) * pitch;
            const auto rowSamples = static_cast<std::ptrdiff_t>(kWidth) * test.components;
            result.insert(result.end(), begin, begin + rowSamples);
        }
        return result;
    }

    Json words_json(const std::vector<float>& values) {
        Json words = Json::array();
        for (const float value : values) {
            words.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return words;
    }

    std::vector<float> words_floats(const Json& words) {
        std::vector<float> result;
        result.reserve(words.size());
        for (const auto& word : words) {
            result.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
        }
        return result;
    }

    std::array<std::vector<float>, 2> expected_pixels(const std::filesystem::path& fixture,
                                                      const Json& updates) {
        std::ifstream input(fixture);
        require(static_cast<bool>(input), "processor fixture is unavailable");
        require(updates.at("schema") == 1 && updates.at("policy") == "print-erff-0.2.15-2026-10-20",
                "unsupported print illuminant expectation policy");
        constexpr std::array<std::string_view, 7> fixtureCases{
            "negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"};
        std::array<std::vector<float>, 2> result;
        for (std::size_t index = 0; index < fixtureCases.size(); ++index) {
            std::string name;
            std::size_t count = 0;
            input >> name >> count;
            require(static_cast<bool>(input) && name == fixtureCases[index] && count > 0,
                    "processor fixture membership or ordering changed");
            std::vector<float> values(count);
            for (float& value : values) {
                input >> value;
            }
            require(static_cast<bool>(input), "processor fixture is truncated");
            if (index < result.size()) {
                Json words = words_json(values);
                const auto logicalPath = "ofx/fixtures/exposure/" + fixture.filename().string() + "." + name;
                const auto& fixtures = updates.at("fixtures");
                if (fixtures.contains(logicalPath)) {
                    const auto& patch = fixtures.at(logicalPath);
                    require(patch.is_array() && patch.size() % 2 == 0,
                            "incomplete print illuminant expectation supplement");
                    for (std::size_t i = 0; i < patch.size(); i += 2) {
                        require(patch[i].at("op") == "test" && patch[i + 1].at("op") == "replace" &&
                                    patch[i].at("path") == patch[i + 1].at("path"),
                                "unbounded print illuminant expectation supplement");
                    }
                    words = words.patch(patch);
                }
                result[index] = words_floats(words);
            }
        }
        std::string trailing;
        require(!(input >> trailing), "processor fixture has unexpected trailing rows");
        return result;
    }

    bool compare_fixture(const std::vector<float>& actual, const std::vector<float>& expected) {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float tolerance = 2.0e-4f + 3.0e-4f * std::abs(expected[i]);
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    bool canaries_preserved(const std::vector<float>& initial, const std::vector<float>& actual, const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = kWidth * test.components; x < pitch; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x);
                if (std::bit_cast<std::uint32_t>(actual[index]) != std::bit_cast<std::uint32_t>(initial[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool alpha_preserved(const std::vector<float>& source, const std::vector<float>& output, const Case& test) {
        if (test.components != 4) {
            return true;
        }
        const int pitch = kWidth * test.components + kPitchPadding;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + 3u;
                if (std::bit_cast<std::uint32_t>(source[index]) != std::bit_cast<std::uint32_t>(output[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    struct CompletionState final {
        bool called = false;
        bool valid = false;
    };

    FjCallCompletionResult complete_call(void* user, const FjCallCompletionInput* input) noexcept {
        auto& state = *static_cast<CompletionState*>(user);
        state.called = true;
        state.valid = input && input->render_outcome_valid == 1 && input->has_inspected_context == 1 &&
                      input->recovery_scope_entered == 1 && input->force_fatal == 0 && input->abort_failed == 0 &&
                      input->outcome.status.category == FJ_STATUS_SUCCESS;
        if (!input) {
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        return {input->outcome, state.valid ? 0u : 1u};
    }

    FjStatus select_submission(void* user, const FjCudaContext*, const FjSubmission* requested, FjSubmission* selected, FjErrorBuffer*) noexcept {
        if (!requested || !selected) {
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        }
        *selected = *requested;
        selected->submission_id = *static_cast<const std::uint64_t*>(user);
        return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    }

    struct PassPixels final {
        const std::vector<float>& source;
        const std::vector<float>& expected;
    };

    Json run_pass(const Case& test, std::string_view pass, const PassPixels& pixels, std::vector<std::uint32_t>* coldWords, std::uint64_t* coldRecipeHash, DeviceBuffers& device) {
        const auto& source = pixels.source;
        const auto& expected = pixels.expected;
        const int pitch = kWidth * test.components + kPitchPadding;
        const std::size_t samples = static_cast<std::size_t>(pitch) * static_cast<std::size_t>(kHeight);
        const std::size_t byteCount = samples * sizeof(float);
        std::vector<float> initialDestination(samples, kDestinationCanary);
        std::vector<float> hostOutput(samples, kDestinationCanary);
        require_cuda(cudaMemcpyAsync(device.destination, initialDestination.data(), byteCount, cudaMemcpyHostToDevice, device.stream), "reset destination buffer");

        FjRect bounds{0, 0, kWidth, kHeight};
        FjFrame frame{};
        frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, static_cast<std::ptrdiff_t>(pitch) * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
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

        std::array<char, 2048> errorStorage{};
        FjErrorBuffer error{errorStorage.data(), errorStorage.size(), 0};
        auto& assets = JuicerProcess::root().assets();
        PreparedOwner prepared;
        prepared.create(assets.borrow_rust_assets(), parameters_for(test), error);
        std::optional<FjPreparedStateView> view(std::in_place);
        require(fj_legacy_prepared_state_view(prepared.get(), &*view, &error).category == FJ_STATUS_SUCCESS,
                "prepared state scalar view failed");
        const FjExtent fullExtent{0, 0, kWidth, kHeight};
        FjPreparedFramePolicy policy{};
        const FjStatus policyStatus = fj_legacy_prepared_state_frame_policy(prepared.get(), &fullExtent, &policy, &error);
        require(policyStatus.category == FJ_STATUS_SUCCESS, error.length ? std::string(error.data, error.length) : "prepared frame policy failed");
        require(view->recipe_hash == policy.recipe_hash && view->route == static_cast<std::uint32_t>(test.route),
                "prepared state policy differs from held route");

        CompletionState completion;
        const auto route = static_cast<std::uint32_t>(test.route);
        JuicerCuda::CallScope scope(JuicerCuda::borrowed_owner(), route, complete_call, &completion);
        std::uint64_t submissionId = 1;
        const FjSubmissionAccess submission{kInstanceToken + route, 37, &submissionId, select_submission};
        CUcontext contextBefore = nullptr;
        require_driver(cuCtxGetCurrent(&contextBefore), "inspect CUDA context before submission");
        const FjCallCompletionResult result = fj_legacy_prepared_state_render(
            assets.borrow_rust_assets(), prepared.get(), scope.access(), &frame, &policy, submission, {}, &error);
        view.reset();
        const bool viewsReleased = !view.has_value();
        const bool stateReleased = prepared.release() && prepared.get() == nullptr;
        CUcontext contextAfterSubmit = nullptr;
        require_driver(cuCtxGetCurrent(&contextAfterSubmit), "inspect CUDA context after submission");

        const bool submitted = result.outcome.status.category == FJ_STATUS_SUCCESS && result.fatal == 0 && completion.called && completion.valid;
        require(submitted, scope.diagnostic().empty() ? (error.length ? std::string(error.data, error.length) : "prepared render submission failed") : scope.diagnostic());
        require(stateReleased, "prepared state release after submission failed");

        require_cuda(cudaStreamSynchronize(device.stream), "wait for submitted CUDA stream");
        const bool streamCompleted = true;
        require_cuda(cudaMemcpy(hostOutput.data(), device.destination, byteCount, cudaMemcpyDeviceToHost), "read rendered pixels");
        const bool readbackComplete = true;
        const bool canaries = canaries_preserved(initialDestination, hostOutput, test);
        const bool alpha = alpha_preserved(source, hostOutput, test);
        require(canaries, "render changed pitched destination canaries");
        require(alpha, "render changed RGBA alpha");
        const auto activeOutput = active_pixels(hostOutput, test);
        const bool fixtureMatch = compare_fixture(activeOutput, expected);
        require(fixtureMatch, std::string(test.id) + ": rendered pixels differ from accepted fixture");
        auto words = words_json(activeOutput);
        std::vector<std::uint32_t> outputWords;
        outputWords.reserve(activeOutput.size());
        for (float value : activeOutput) {
            outputWords.push_back(std::bit_cast<std::uint32_t>(value));
        }
        if (pass == "cold") {
            *coldWords = outputWords;
            *coldRecipeHash = policy.recipe_hash;
        } else {
            require(coldWords && *coldWords == outputWords, "warm output words differ from cold output words");
            require(coldRecipeHash && *coldRecipeHash == policy.recipe_hash,
                    "fresh warm prepared state recipe hash differs from cold state");
        }
        Json row{
            {"case_id", std::string(test.id) + "/" + std::string(pass)},
            {"route", test.routeName},
            {"pass", std::string(pass)},
            {"recipe_hash", policy.recipe_hash},
            {"status", result.outcome.status.category},
            {"api", result.outcome.status.api},
            {"native_code", result.outcome.status.native_code},
            {"diagnostic", scope.diagnostic()},
            {"input_shape", Json::array({kHeight, kWidth, test.components})},
            {"output_shape", Json::array({kHeight, kWidth, test.components})},
            {"accepted_fixture_match", fixtureMatch},
            {"submitted", submitted},
            {"readback_complete", readbackComplete},
            {"state_released_after_submit", stateReleased},
            {"views_released_after_submit", viewsReleased},
            {"stream_completed", streamCompleted},
            {"context_preserved", contextBefore == contextAfterSubmit},
            {"context_restored", false},
            {"input_f32_words", words_json(active_pixels(source, test))},
            {"output_f32_words", std::move(words)}};
        return row;
    }

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: JuicerProductionPreparedRenderProbe Resources processor-${preset}.dat preset product-updates.json");
        const std::filesystem::path resources(argv[1]);
        const std::filesystem::path fixture(argv[2]);
        const std::string preset(argv[3]);
        const std::filesystem::path updates(argv[4]);
        require(fixture.filename().string() == "processor-" + preset + ".dat", "processor fixture does not match the selected preset");
        require(std::filesystem::exists(updates), "print expectation updates file is unavailable");
        std::ifstream updatesInput(updates);
        require(static_cast<bool>(updatesInput), "print expectation updates file could not be opened");
        const Json updatesJson = Json::parse(updatesInput);

        require_driver(cuInit(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        require_driver(cuCtxGetCurrent(&original), "read original CUDA context");
        ContextRestore contextRestore(original);

        require_cuda(cudaSetDevice(0), "select CUDA device");
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();

        const auto expected = expected_pixels(fixture, updatesJson);
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create production stream");

        Json rows = Json::array();
        for (std::size_t caseIndex = 0; caseIndex < kCases.size(); ++caseIndex) {
            const Case& test = kCases[caseIndex];
            const auto source = initial_source(test);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate source buffer");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate destination buffer");
            require_cuda(cudaMemcpy(device.source, source.data(), bytes, cudaMemcpyHostToDevice), "upload deterministic source");
            std::vector<float> destination(source.size(), kDestinationCanary);
            require_cuda(cudaMemcpy(device.destination, destination.data(), bytes, cudaMemcpyHostToDevice), "initialize destination canaries");

            std::vector<std::uint32_t> coldWords;
            std::uint64_t coldRecipeHash = 0;
            const PassPixels pixels{.source = source, .expected = expected[caseIndex]};
            rows.push_back(run_pass(test, "cold", pixels, &coldWords, &coldRecipeHash, device));
            rows.push_back(run_pass(test, "warm", pixels, &coldWords, &coldRecipeHash, device));

            require_cuda(cudaFree(device.destination), "free destination buffer");
            device.destination = nullptr;
            require_cuda(cudaFree(device.source), "free source buffer");
            device.source = nullptr;
        }

        require_cuda(cudaStreamSynchronize(device.stream), "complete all production stream work");
        require_cuda(cudaStreamDestroy(device.stream), "destroy production stream");
        device.stream = nullptr;
        require(owner.close().category == FJ_STATUS_SUCCESS, "normal CUDA owner shutdown failed");
        contextRestore.restore();
        for (auto& row : rows) {
            row["context_restored"] = contextRestore.restored;
        }
        const std::string output = Json{{"rows", std::move(rows)}}.dump();
        if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() || std::fputc('\n', stdout) == EOF) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        (void)std::fputs(error.what(), stderr);
        (void)std::fputc('\n', stderr);
        return 1;
    } catch (...) {
        (void)std::fputs("unknown production prepared-render failure\n", stderr);
        return 1;
    }
}
