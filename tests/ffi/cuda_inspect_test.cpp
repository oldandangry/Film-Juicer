#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"

namespace {
    FjFrame frame_at() {
        FjFrame frame{};
        frame.source = {0x1000, {-2, -3, 6, 5}, 144, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
        frame.destination = {0x2000, {-4, -5, 4, 3}, 160, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
        frame.render_window = {-1, -2, 3, 2};
        frame.full_frame_extent = frame.source.bounds;
        return frame;
    }

    class InspectHost : public testing::Test {
    protected:
        void SetUp() override {
            owner.create("inspection-does-not-load-assets");
        }
        void TearDown() override {
            EXPECT_EQ(owner.close().category, FJ_STATUS_SUCCESS);
        }
        void reject(const FjFrame& frame) {
            FjCudaContext context{73, 91};
            std::array<char, 9> text{};
            FjErrorBuffer error{text.data(), text.size(), 42};
            const auto result = fj_cuda_inspect(JuicerCuda::borrowed_owner(), &frame, &context, &error);
            EXPECT_EQ(result.category, FJ_STATUS_UNSUPPORTED_INPUT) << text.data();
            EXPECT_EQ(result.api, FJ_API_NONE);
            EXPECT_EQ(result.native_code, 0);
            EXPECT_EQ(context.device_id, 73);
            EXPECT_EQ(context.context, 91U);
            EXPECT_EQ(error.length, text.size() - 1);
            EXPECT_EQ(text.back(), '\0');
        }
        JuicerCuda::Owner owner;
    };

    TEST_F(InspectHost, InvalidArgumentsLeaveOutputUnchanged) {
        auto frame = frame_at();
        FjCudaContext context{73, 91};
        EXPECT_EQ(fj_cuda_inspect(nullptr, &frame, &context, nullptr).category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), nullptr, &context, nullptr).category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), &frame, nullptr, nullptr).category, FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer error{nullptr, 5, 42};
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), &frame, &context, &error).category, FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(context.device_id, 73);
        EXPECT_EQ(context.context, 91U);
    }

    TEST_F(InspectHost, RenderRejectsInvalidArgumentsBeforeCudaDiscovery) {
        auto* const cuda = JuicerCuda::borrowed_owner();
        auto frame = frame_at();
        frame.source.address = 0;
        const FjCudaContext context{0, 1};
        const FjSubmission submission{1, 0, 1, 1, 1, 1, 1};
        const FjPreparedHostData prepared{};
        const auto category = [&](FjCuda* inputOwner, const FjCudaContext* inputContext, const FjFrame* inputFrame, const FjSubmission* inputSubmission, const FjPreparedHostData* inputPrepared, FjErrorBuffer* error) {
            return fj_cuda_render(inputOwner, inputContext, inputFrame, inputSubmission, inputPrepared, {}, error).category;
        };
        EXPECT_EQ(category(nullptr, &context, &frame, &submission, &prepared, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(category(cuda, nullptr, &frame, &submission, &prepared, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(category(cuda, &context, nullptr, &submission, &prepared, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(category(cuda, &context, &frame, nullptr, &prepared, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(category(cuda, &context, &frame, &submission, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(category(reinterpret_cast<FjCuda*>(std::uintptr_t{1}), &context, &frame, &submission, &prepared, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(category(cuda, &context, &frame, &submission, &prepared, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer malformed{nullptr, 1, 99};
        EXPECT_EQ(category(cuda, &context, &frame, &submission, &prepared, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(malformed.length, 0U);
    }

    TEST_F(InspectHost, FlagsWindowsAndComponents) {
        auto frame = frame_at();
        frame.flags = 8;
        reject(frame);
        frame = frame_at();
        frame.stream = 1;
        reject(frame);
        for (const FjRect window : {FjRect{0, 0, 0, 1}, FjRect{0, 0, 1, 0}, FjRect{2, 0, 1, 1}, FjRect{-3, -2, 3, 2}, FjRect{-1, -2, 5, 2}, FjRect{-1, -4, 3, 2}, FjRect{-1, -2, 3, 4}}) {
            frame = frame_at();
            frame.render_window = window;
            reject(frame);
        }
        frame = frame_at();
        frame.full_frame_extent = {};
        reject(frame);
        frame = frame_at();
        frame.destination.components = FJ_COMPONENTS_RGB;
        reject(frame);
    }

    TEST_F(InspectHost, EachImageIsValidatedIndependently) {
        for (bool source : {true, false}) {
            for (int fault = 0; fault < 14; ++fault) {
                SCOPED_TRACE(std::to_string(source) + ":" + std::to_string(fault));
                auto frame = frame_at();
                auto& image = source ? frame.source : frame.destination;
                switch (fault) {
                    case 0:
                        image.address = 0;
                        break;
                    case 1:
                        image.components = 1;
                        break;
                    case 2:
                        image.depth = 2;
                        break;
                    case 3:
                        image.bounds.x2 = image.bounds.x1;
                        break;
                    case 4:
                        image.bounds.y2 = image.bounds.y1 - 1;
                        break;
                    case 5:
                        image.row_bytes = 0;
                        break;
                    case 6:
                        image.row_bytes = -144;
                        break;
                    case 7:
                        image.row_bytes = 124;
                        break;
                    case 8:
                        image.row_bytes = std::numeric_limits<std::ptrdiff_t>::max() - 3;
                        break;
                    case 9:
                        image.address = std::numeric_limits<std::uintptr_t>::max() - 3;
                        break;
                    case 10:
                        image.bounds.x1 = std::numeric_limits<std::int32_t>::min();
                        image.bounds.x2 = std::numeric_limits<std::int32_t>::max();
                        break;
                    case 11:
                        image.address += 1;
                        break;
                    case 12:
                        image.row_bytes += 1;
                        break;
                    case 13:
                        image.bounds.y1 = frame.render_window.y1 + 1;
                        break;
                }
                reject(frame);
            }
        }
    }

    TEST_F(InspectHost, EqualAndPartialAccessOverlap) {
        auto frame = frame_at();
        frame.destination = frame.source;
        reject(frame);
        frame.destination.address += 16;
        reject(frame);
    }

#if defined(JUICER_CUDA_INSPECT_GPU_TEST)
    class InspectGpu : public InspectHost {
    protected:
        void SetUp() override {
            InspectHost::SetUp();
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&source, 4096), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&destination, 4096), cudaSuccess);
        }
        void TearDown() override {
            EXPECT_EQ(cudaFree(source), cudaSuccess);
            EXPECT_EQ(cudaFree(destination), cudaSuccess);
            InspectHost::TearDown();
        }
        FjFrame frame() const {
            auto result = frame_at();
            result.source.address = reinterpret_cast<std::uintptr_t>(source);
            result.destination.address = reinterpret_cast<std::uintptr_t>(destination);
            return result;
        }
        void* source = nullptr;
        void* destination = nullptr;
    };

    TEST_F(InspectGpu, ExactIdentityAndDistinctOrigins) {
        const auto input = frame();
        FjCudaContext context{};
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), &input, &context, nullptr).category, FJ_STATUS_SUCCESS);
        CUcontext current = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
        EXPECT_EQ(context.device_id, 0);
        EXPECT_EQ(context.context, reinterpret_cast<std::uintptr_t>(current));
    }

    TEST_F(InspectGpu, AllocationCoverageIncludesFullSourceAndDestinationBounds) {
        for (bool sourceImage : {true, false}) {
            auto input = frame();
            auto& image = sourceImage ? input.source : input.destination;
            image.address += 4096 - 128;
            reject(input);
        }
        auto input = frame();
        input.source.address += 4096 - (7 * 144 + 128);
        FjCudaContext context{};
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), &input, &context, nullptr).category, FJ_STATUS_SUCCESS);
        input.source.address += sizeof(float);
        reject(input);
    }

    TEST_F(InspectGpu, RejectsHostMemoryAndForeignContext) {
        std::array<float, 400> host{};
        auto input = frame();
        input.destination.address = reinterpret_cast<std::uintptr_t>(host.data());
        reject(input);
        CUcontext original = nullptr;
        CUcontext other = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&original), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxCreate(&other, nullptr, 0, 0), CUDA_SUCCESS);
        input = frame();
        reject(input);
        EXPECT_EQ(cuCtxDestroy(other), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxSetCurrent(original), CUDA_SUCCESS);
    }

    TEST_F(InspectGpu, RejectsManagedMemory) {
        void* managed = nullptr;
        ASSERT_EQ(cudaMallocManaged(&managed, 4096), cudaSuccess);
        auto input = frame();
        input.destination.address = reinterpret_cast<std::uintptr_t>(managed);
        reject(input);
        EXPECT_EQ(cudaFree(managed), cudaSuccess);
    }

    TEST_F(InspectGpu, DeviceMismatchWhenTwoDevicesAreAvailable) {
        int devices = 0;
        ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
        if (devices < 2) {
            GTEST_SKIP() << "device mismatch requires two physical CUDA devices";
        }
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        void* other = nullptr;
        ASSERT_EQ(cudaMalloc(&other, 4096), cudaSuccess);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        auto input = frame();
        input.destination.address = reinterpret_cast<std::uintptr_t>(other);
        reject(input);
        EXPECT_EQ(cudaSetDevice(1), cudaSuccess);
        EXPECT_EQ(cudaFree(other), cudaSuccess);
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    TEST_F(InspectGpu, SameAllocationDisjointRowsAreSupported) {
        auto input = frame();
        input.source = {reinterpret_cast<std::uintptr_t>(source), {0, 0, 2, 2}, 128, FJ_COMPONENTS_RGBA, FJ_DEPTH_FLOAT32};
        input.destination = input.source;
        input.destination.address += 32;
        input.render_window = input.source.bounds;
        input.full_frame_extent = input.source.bounds;
        FjCudaContext context{};
        EXPECT_EQ(fj_cuda_inspect(JuicerCuda::borrowed_owner(), &input, &context, nullptr).category, FJ_STATUS_SUCCESS);
        input.destination.address -= sizeof(float);
        reject(input);
    }
#endif
} // namespace
