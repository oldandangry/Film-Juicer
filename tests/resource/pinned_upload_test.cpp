#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "../ffi/juicer_test_api.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResources.h"

namespace JuicerCuda::PinnedUploadTest {
    thread_local std::array<std::int32_t, 5> errors{};
    thread_local std::array<int, 5> calls{};
    thread_local bool failDiagnosticAllocation = false;
    thread_local int diagnosticCalls = 0;

    std::int32_t injected_error(Operation operation) noexcept {
        const auto index = static_cast<std::size_t>(operation);
        ++calls[index];
        return std::exchange(errors[index], 0);
    }

    void before_diagnostic() {
        ++diagnosticCalls;
        if (std::exchange(failDiagnosticAllocation, false)) {
            throw std::bad_alloc{};
        }
    }
} // namespace JuicerCuda::PinnedUploadTest

namespace {
    using JuicerCuda::PinnedUploadPurgeDisposition;
    using JuicerCuda::PinnedUploadTest::Operation;
    constexpr std::size_t kCap = std::size_t{128} * 1024u * 1024u;

    void require_cuda(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    void inject(Operation operation, cudaError_t error) {
        JuicerCuda::PinnedUploadTest::errors[static_cast<std::size_t>(operation)] = error;
    }

    // Only warmed, explicitly supplied streams enter this gate. It never wraps
    // rendering, allocations, absent-stream completion or exceptional-sync tests.
    class UploadGate final {
    public:
        explicit UploadGate(cudaStream_t stream) : _stream(stream) {
            require_cuda(cudaLaunchHostFunc(stream, &UploadGate::wait, this));
        }
        ~UploadGate() {
            release();
            // The fixture checks CUDA completion; teardown must not throw while
            // ensuring that the callback no longer refers to this gate.
            (void)cudaStreamSynchronize(_stream);
        }
        void release() noexcept {
            try {
                {
                    std::lock_guard lock(_mutex);
                    _released = true;
                }
                _condition.notify_all();
            } catch (...) {
                _timedOut.store(true);
            }
        }
        bool timed_out() const {
            return _timedOut.load();
        }

    private:
        static void CUDART_CB wait(void* pointer) {
            auto& gate = *static_cast<UploadGate*>(pointer);
            try {
                std::unique_lock lock(gate._mutex);
                if (!gate._condition.wait_for(lock, std::chrono::seconds(10), [&] {
                        return gate._released;
                    })) {
                    gate._timedOut.store(true);
                }
            } catch (...) {
                gate._timedOut.store(true);
            }
        }
        std::mutex _mutex;
        std::condition_variable _condition;
        bool _released = false;
        cudaStream_t _stream = nullptr;
        std::atomic<bool> _timedOut{false};
    };

    class PinnedUpload : public ::testing::Test {
    protected:
        void SetUp() override {
            JuicerCuda::PinnedUploadTest::errors = {};
            JuicerCuda::PinnedUploadTest::calls = {};
            JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = false;
            JuicerCuda::PinnedUploadTest::diagnosticCalls = 0;
            require_cuda(cudaSetDevice(0));
            require_cuda(cudaFree(nullptr));
            CUcontext context = nullptr;
            ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
            ASSERT_NE(context, nullptr);
            key = {0, context};
            std::string diagnostic;
            ledger = JuicerCuda::DeviceAllocationLedger::create({0, 1024}, diagnostic);
            ASSERT_NE(ledger, nullptr) << diagnostic;
            ASSERT_TRUE(ledger->bind_or_validate_cap(1024, diagnostic)) << diagnostic;
            resources = std::make_unique<JuicerCuda::Resources>(key, 1, ledger);
            require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            require_cuda(cudaMalloc(&destination, kCap));
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
        }
        void TearDown() override {
            JuicerCuda::PinnedUploadTest::errors = {};
            JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = false;
            if (stream) {
                EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            }
            resources.reset();
            if (ledger) {
                EXPECT_EQ(ledger->snapshot().chargedBytes, 0);
            }
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
            if (destination) {
                EXPECT_EQ(cudaFree(destination), cudaSuccess);
            }
            if (stream) {
                EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
            }
        }
        bool upload(const void* source, std::size_t bytes) {
            return JuicerCuda::PinnedUploadTest::upload(*resources, destination, source, bytes, stream, failure);
        }
        void warm(const std::vector<std::uint8_t>& source) {
            if (!upload(source.data(), source.size())) {
                throw std::runtime_error(failure.diagnostic);
            }
            require_cuda(cudaStreamSynchronize(stream));
            JuicerCuda::PinnedUploadTest::poll(key);
            if (JuicerCuda::PinnedUploadTest::snapshot(key).available != 1) {
                throw std::runtime_error("staging warmup did not leave one available block");
            }
            JuicerCuda::PinnedUploadTest::calls = {};
        }
        void expect_status(FjStatus expected, bool diagnosticExpected = true) {
            EXPECT_EQ(failure.status.category, expected.category) << failure.diagnostic;
            EXPECT_EQ(failure.status.api, expected.api);
            EXPECT_EQ(failure.status.native_code, expected.native_code);
            EXPECT_EQ(!failure.diagnostic.empty(), diagnosticExpected);
        }
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> ledger;
        std::unique_ptr<JuicerCuda::Resources> resources;
        cudaStream_t stream = nullptr;
        void* destination = nullptr;
        JuicerCuda::Failure failure;
    };

    TEST_F(PinnedUpload, CallerStorageExpiresWhileNativeUploadIsOutstanding) {
        std::vector<std::uint8_t> source(4096, 0x16);
        warm(source);
        std::fill(source.begin(), source.end(), 0x69);
        const auto original = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        UploadGate gate(stream);
        ASSERT_TRUE(upload(source.data(), source.size())) << failure.diagnostic;
        const auto pending = JuicerCuda::PinnedUploadTest::snapshot(key);
        ASSERT_EQ(pending.inFlight, 1);
        EXPECT_EQ(pending.reserved, 0);
        EXPECT_EQ(pending.blocks.front().id, original.id);
        EXPECT_NE(pending.blocks.front().pointer, source.data());
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Allocate)], 0);
        auto* const event = static_cast<cudaEvent_t>(pending.blocks.front().event);
        ASSERT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        std::fill(source.begin(), source.end(), 0x16);
        std::vector<std::uint8_t>().swap(source);
        ASSERT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        JuicerCuda::PinnedUploadTest::poll(key);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).inFlight, 1);
        gate.release();
        require_cuda(cudaEventSynchronize(event));
        std::array<std::uint8_t, 4096> actual{};
        require_cuda(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](auto value) {
            return value == 0x69;
        }));
        EXPECT_FALSE(gate.timed_out());
        JuicerCuda::PinnedUploadTest::poll(key);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).available, 1);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).inFlight, 0);
    }

    TEST_F(PinnedUpload, RustNoiseOwnerExpiresBeforeNativeStagingCompletes) {
        std::vector<std::uint8_t> initial(4096, 0);
        warm(initial);
        const std::string root = JUICER_TEST_RESOURCE_DIR;
        FjNoise* source = nullptr;
        ASSERT_EQ(fj_test_noise_acquire({root.data(), root.size()}, &source, nullptr).category, FJ_STATUS_SUCCESS);
        struct Release {
            FjNoise* owner;
            ~Release() {
                (void)fj_test_noise_release(owner, nullptr);
            }
        } owner{source};
        FjStaticNoise view{};
        ASSERT_EQ(fj_test_noise_view(source, &view, nullptr).category, FJ_STATUS_SUCCESS);
        const std::vector<std::uint8_t> expected(view.stbn.data, view.stbn.data + 4096);
        UploadGate gate(stream);
        ASSERT_TRUE(upload(view.stbn.data, expected.size())) << failure.diagnostic;
        const auto pending = JuicerCuda::PinnedUploadTest::snapshot(key);
        ASSERT_EQ(pending.inFlight, 1);
        const auto* block = pending.blocks.front().pointer;
        EXPECT_NE(block, view.stbn.data);
        auto* event = static_cast<cudaEvent_t>(pending.blocks.front().event);
        ASSERT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        ASSERT_EQ(fj_test_noise_release(std::exchange(owner.owner, nullptr), nullptr).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_live_noise_owners(), 0u);
        view = {};
        ASSERT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        gate.release();
        require_cuda(cudaEventSynchronize(event));
        std::array<std::uint8_t, 4096> actual{};
        require_cuda(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin()));
        EXPECT_FALSE(gate.timed_out());
        JuicerCuda::PinnedUploadTest::poll(key);
    }

    TEST_F(PinnedUpload, OversizedAndBusyPoolFailuresDoNotSelectDeviceReclaim) {
        const std::uint8_t byte = 0;
        ASSERT_FALSE(upload(&byte, kCap + 1));
        expect_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
        std::vector<std::uint8_t> source(kCap, 0x31);
        warm(source);
        UploadGate gate(stream);
        ASSERT_TRUE(upload(source.data(), source.size()));
        ASSERT_FALSE(upload(&byte, 1));
        expect_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
        const auto pending = JuicerCuda::PinnedUploadTest::snapshot(key);
        EXPECT_EQ(pending.totalBytes, kCap);
        EXPECT_EQ(pending.inFlight, 1);
        EXPECT_EQ(pending.reserved, 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 1);
        EXPECT_EQ(cudaEventQuery(static_cast<cudaEvent_t>(pending.blocks.front().event)), cudaErrorNotReady);
        gate.release();
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_FALSE(gate.timed_out());
    }

    TEST_F(PinnedUpload, PressureTrimReplacesOnlyCompletedStorage) {
        std::vector<std::uint8_t> source(kCap / 2 + 1, 0x2a);
        ASSERT_TRUE(upload(source.data(), kCap / 2));
        require_cuda(cudaStreamSynchronize(stream));
        JuicerCuda::PinnedUploadTest::poll(key);
        ASSERT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).available, 1);
        ASSERT_TRUE(upload(source.data(), source.size()));
        const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
        ASSERT_EQ(state.blocks.size(), 1);
        EXPECT_EQ(state.totalBytes, source.size());
        EXPECT_EQ(state.blocks.front().capacity, source.size());
        EXPECT_EQ(state.reserved, 0);
        require_cuda(cudaStreamSynchronize(stream));
    }

    TEST_F(PinnedUpload, ReservationFailuresPreserveRuntimeCodesWithoutDeviceReclaim) {
        const std::array<std::uint8_t, 64> source{};
        for (const auto operation : {Operation::Allocate, Operation::CreateEvent}) {
            for (const auto error : {cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorContextIsDestroyed}) {
                inject(operation, error);
                ASSERT_FALSE(upload(source.data(), source.size()));
                expect_status({error == cudaErrorContextIsDestroyed ? FJ_STATUS_CONTEXT_LOSS : FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, error});
                EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
            }
        }
    }

    TEST_F(PinnedUpload, FailedStagedCopyReturnsReservationAndNeverBorrowsCallerStorage) {
        const std::array<std::uint8_t, 64> source{};
        for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorContextIsDestroyed}) {
            JuicerCuda::PinnedUploadTest::calls = {};
            inject(Operation::Copy, error);
            ASSERT_FALSE(upload(source.data(), source.size()));
            const auto expected = JuicerCuda::runtime_failure_status(error);
            expect_status(expected);
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 1);
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::RecordEvent)], 0);
            EXPECT_NE(failure.diagnostic.find("prepared span pinned staging"), std::string::npos);
            const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
            EXPECT_EQ(state.available, 1);
            EXPECT_EQ(state.reserved + state.inFlight + state.quarantined, 0);
        }
        ASSERT_TRUE(upload(source.data(), source.size()));
    }

    TEST_F(PinnedUpload, GrainAllocationRollsBackAfterStagingFailure) {
        const std::array<std::uint8_t, 8> bytes{};
        const JuicerCuda::StaticNoiseInput input{bytes, bytes, bytes, 1, 1, 1, 1, 1, 1, 1};
        inject(Operation::Copy, cudaErrorInvalidValue);
        ASSERT_FALSE(JuicerCuda::ensure_grain_static_assets_uploaded(*resources, input, stream, failure));
        expect_status({FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, cudaErrorInvalidValue});
        EXPECT_EQ(resources->stbnData, nullptr);
        require_cuda(cudaStreamSynchronize(stream));
        resources.reset();
        EXPECT_EQ(ledger->snapshot().chargedBytes, 0);
        EXPECT_EQ(ledger->snapshot().recordCount, 0);
        JuicerCuda::PinnedUploadTest::poll(key);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).quarantined, 0);
    }

    TEST_F(PinnedUpload, ReservationDiagnosticFailurePreservesNativeStatus) {
        const std::array<std::uint8_t, 64> source{};
        for (const auto operation : {Operation::Allocate, Operation::CreateEvent}) {
            for (const auto error : {cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorContextIsDestroyed}) {
                inject(operation, error);
                JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
                bool uploaded = true;
                ASSERT_NO_THROW(uploaded = upload(source.data(), source.size()));
                EXPECT_FALSE(uploaded);
                expect_status({error == cudaErrorContextIsDestroyed ? FJ_STATUS_CONTEXT_LOSS : FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, error}, false);
                EXPECT_FALSE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
            }
        }
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::diagnosticCalls, 6);
    }

    TEST_F(PinnedUpload, CopyDiagnosticFailureReturnsReservationAndPreservesNativeStatus) {
        const std::vector<std::uint8_t> source(64, 0x52);
        warm(source);
        const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorContextIsDestroyed}) {
            inject(Operation::Copy, error);
            JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
            bool uploaded = true;
            ASSERT_NO_THROW(uploaded = upload(source.data(), source.size()));
            EXPECT_FALSE(uploaded);
            expect_status(JuicerCuda::runtime_failure_status(error), false);
            EXPECT_FALSE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
            const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
            EXPECT_EQ(state.available, 1);
            EXPECT_EQ(state.reserved + state.inFlight + state.quarantined, 0);
            ASSERT_EQ(state.blocks.size(), 1);
            EXPECT_EQ(state.blocks.front().id, block.id);
        }
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::diagnosticCalls, 3);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 3);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::RecordEvent)], 0);
        ASSERT_TRUE(upload(source.data(), source.size()));
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Allocate)], 0);
    }

    TEST_F(PinnedUpload, GrainAllocationRollsBackWhenDiagnosticAllocationFails) {
        const std::array<std::uint8_t, 8> bytes{};
        const JuicerCuda::StaticNoiseInput input{bytes, bytes, bytes, 1, 1, 1, 1, 1, 1, 1};
        // Establish the two Wang allocations while leaving STBN unprepared.
        inject(Operation::Copy, cudaErrorInvalidValue);
        ASSERT_FALSE(JuicerCuda::ensure_grain_static_assets_uploaded(*resources, input, stream, failure));
        ASSERT_EQ(resources->stbnData, nullptr);
        ASSERT_NE(resources->wangTilesData, nullptr);
        ASSERT_NE(resources->wangLutData, nullptr);
        require_cuda(cudaStreamSynchronize(stream));
        JuicerCuda::PinnedUploadTest::poll(key);
        const auto before = ledger->snapshot();
        ASSERT_EQ(before.chargedBytes, 2 * bytes.size());
        ASSERT_EQ(before.recordCount, 2);

        inject(Operation::Copy, cudaErrorInvalidValue);
        JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
        bool prepared = true;
        ASSERT_NO_THROW(prepared = JuicerCuda::ensure_grain_static_assets_uploaded(*resources, input, stream, failure));
        EXPECT_FALSE(prepared);
        expect_status({FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, cudaErrorInvalidValue});
        EXPECT_FALSE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
        EXPECT_EQ(resources->stbnData, nullptr);
        // Rollback must have happened before resource-owner teardown.
        EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
        EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
        const auto staging = JuicerCuda::PinnedUploadTest::snapshot(key);
        EXPECT_EQ(staging.reserved + staging.inFlight + staging.quarantined, 0);
        ASSERT_TRUE(JuicerCuda::ensure_grain_static_assets_uploaded(*resources, input, stream, failure));
        EXPECT_NE(resources->stbnData, nullptr);
        EXPECT_EQ(ledger->snapshot().chargedBytes, 3 * bytes.size());
    }

    TEST_F(PinnedUpload, CompletedExceptionalSyncNeedsNoFailureDiagnosticAllocation) {
        const std::vector<std::uint8_t> source(64, 0x6d);
        warm(source);
        inject(Operation::RecordEvent, cudaErrorInvalidResourceHandle);
        JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
        bool uploaded = false;
        ASSERT_NO_THROW(uploaded = upload(source.data(), source.size()));
        EXPECT_TRUE(uploaded);
        EXPECT_TRUE(failure.diagnostic.empty());
        EXPECT_TRUE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::diagnosticCalls, 0);
        JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = false;
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Synchronize)], 1);
        const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
        EXPECT_EQ(state.available, 1);
        EXPECT_EQ(state.reserved + state.inFlight + state.quarantined, 0);
        std::array<std::uint8_t, 64> actual{};
        require_cuda(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::equal(actual.begin(), actual.end(), source.begin()));
        ASSERT_TRUE(upload(source.data(), source.size()));
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Allocate)], 0);
    }

    TEST_F(PinnedUpload, CompletionDiagnosticFailurePreservesQuarantineAndContextLoss) {
        const std::vector<std::uint8_t> source(64, 0x48);
        for (const bool recordLosesContext : {false, true}) {
            warm(source);
            const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
            UploadGate gate(stream);
            inject(Operation::RecordEvent, recordLosesContext ? cudaErrorContextIsDestroyed : cudaErrorInvalidValue);
            inject(Operation::Synchronize, recordLosesContext ? cudaErrorInvalidValue : cudaErrorContextIsDestroyed);
            JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
            bool uploaded = true;
            EXPECT_NO_THROW(uploaded = upload(source.data(), source.size()));
            EXPECT_FALSE(uploaded);
            expect_status({FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, cudaErrorContextIsDestroyed}, false);
            EXPECT_FALSE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
            const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
            EXPECT_EQ(state.quarantined, 1);
            EXPECT_EQ(state.reserved + state.available + state.inFlight, 0);
            EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
            gate.release();
            require_cuda(cudaStreamSynchronize(stream));
            EXPECT_FALSE(gate.timed_out());
            JuicerCuda::PinnedUploadTest::poll(key);
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).quarantined, 1);
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
            require_cuda(cudaFreeHost(block.pointer));
        }
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::diagnosticCalls, 2);
    }

    TEST_F(PinnedUpload, InvalidInputsFailWithoutAReservation) {
        const std::uint8_t byte = 0;
        EXPECT_TRUE(upload(nullptr, 0));
        EXPECT_FALSE(upload(nullptr, 1));
        expect_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        resources->deviceId = 1;
        EXPECT_FALSE(upload(&byte, 1));
        resources->deviceId = key.deviceId;
        expect_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0});
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
    }

    TEST_F(PinnedUpload, CompletionFailurePreservesLaterContextLossAndBothDiagnostics) {
        const std::array<std::uint8_t, 64> source{};
        inject(Operation::RecordEvent, cudaErrorInvalidResourceHandle);
        inject(Operation::Synchronize, cudaErrorContextIsDestroyed);
        ASSERT_FALSE(upload(source.data(), source.size()));
        expect_status({FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, cudaErrorContextIsDestroyed});
        EXPECT_NE(failure.diagnostic.find("cudaEventRecord"), std::string::npos);
        EXPECT_NE(failure.diagnostic.find("cudaStreamSynchronize"), std::string::npos);
        const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
        ASSERT_EQ(state.quarantined, 1);
        EXPECT_EQ(state.reserved, 0);
        require_cuda(cudaStreamSynchronize(stream));
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        require_cuda(cudaFreeHost(state.blocks.front().pointer));
    }

    TEST_F(PinnedUpload, EventRecordFailurePreservesExceptionalCompletion) {
        const std::array<std::uint8_t, 64> source{};
        inject(Operation::RecordEvent, cudaErrorInvalidResourceHandle);
        ASSERT_TRUE(upload(source.data(), source.size()));
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Synchronize)], 1);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).available, 1);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
        EXPECT_TRUE(failure.diagnostic.empty());
    }

    TEST_F(PinnedUpload, QuarantinedUploadCannotUseAnOlderEventAsCompletionProof) {
        std::vector<std::uint8_t> source(4096, 0x39);
        warm(source);
        const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        UploadGate gate(stream);
        inject(Operation::RecordEvent, cudaErrorContextIsDestroyed);
        inject(Operation::Synchronize, cudaErrorInvalidValue);
        ASSERT_FALSE(upload(source.data(), source.size()));
        expect_status({FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, cudaErrorContextIsDestroyed});
        EXPECT_NE(failure.diagnostic.find("cudaEventRecord"), std::string::npos);
        EXPECT_NE(failure.diagnostic.find("cudaStreamSynchronize"), std::string::npos);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).quarantined, 1);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
        // This event belongs to the prior completed upload, so success proves nothing.
        EXPECT_EQ(cudaEventQuery(static_cast<cudaEvent_t>(block.event)), cudaSuccess);
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, block.pointer), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
        gate.release();
        require_cuda(cudaStreamSynchronize(stream));
        std::array<std::uint8_t, 4096> actual{};
        require_cuda(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual.front(), 0x39);
        EXPECT_EQ(actual.back(), 0x39);
        EXPECT_FALSE(gate.timed_out());
        // The isolated fixture can prove completion now; production deliberately abandoned it.
        require_cuda(cudaFreeHost(block.pointer));
    }

    TEST_F(PinnedUpload, NormalPurgeWaitsForTheCurrentUploadFence) {
        std::vector<std::uint8_t> source(4096, 0x73);
        warm(source);
        UploadGate gate(stream);
        ASSERT_TRUE(upload(source.data(), source.size()));
        auto* const event = static_cast<cudaEvent_t>(JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front().event);
        ASSERT_EQ(cudaEventQuery(event), cudaErrorNotReady);
        auto purge = std::async(std::launch::async, [&] {
            require_cuda(cudaSetDevice(key.deviceId));
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        });
        // Release before any fatal assertion or future destruction can wait on the gate.
        const bool returnedEarly = purge.wait_for(std::chrono::milliseconds(50)) == std::future_status::ready;
        gate.release();
        EXPECT_FALSE(returnedEarly);
        purge.get();
        EXPECT_FALSE(gate.timed_out());
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
    }

    TEST_F(PinnedUpload, ProvenLossPurgeAbandonsStorageWithoutFreeingOutstandingBytes) {
        std::vector<std::uint8_t> source(4096, 0x27);
        warm(source);
        const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        UploadGate gate(stream);
        ASSERT_TRUE(upload(source.data(), source.size()));
        ASSERT_EQ(cudaEventQuery(static_cast<cudaEvent_t>(block.event)), cudaErrorNotReady);
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::ProvenContextLoss);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, block.pointer), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
        gate.release();
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_FALSE(gate.timed_out());
        require_cuda(cudaEventDestroy(static_cast<cudaEvent_t>(block.event)));
        require_cuda(cudaFreeHost(block.pointer));
    }
} // namespace
