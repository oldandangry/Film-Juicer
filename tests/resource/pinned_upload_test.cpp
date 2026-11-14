#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "../ffi/gtest_cases.h"
#include "../ffi/juicer_test_api.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaResources.h"

namespace JuicerCuda::PinnedUploadTest {
    thread_local std::array<std::int32_t, 5> errors{};
    thread_local std::array<int, 5> calls{};
    thread_local bool failDiagnosticAllocation = false;
    thread_local int diagnosticCalls = 0;
    thread_local std::function<void(Resources&)> overwriteStaged;
    thread_local std::exception_ptr overwriteHookError;
    thread_local std::function<void(Operation)> observeOperation;
    thread_local std::exception_ptr operationHookError;

    std::int32_t injected_error(Operation operation) noexcept {
        const auto index = static_cast<std::size_t>(operation);
        ++calls[index];
        if (observeOperation) {
            try {
                observeOperation(operation);
            } catch (...) {
                operationHookError = std::current_exception();
            }
        }
        return std::exchange(errors[index], 0);
    }

    void before_diagnostic() {
        ++diagnosticCalls;
        if (std::exchange(failDiagnosticAllocation, false)) {
            throw std::bad_alloc{};
        }
    }

    void after_overwrite_staging(Resources& resources) noexcept {
        if (auto observe = std::exchange(overwriteStaged, {})) {
            try {
                observe(resources);
            } catch (...) {
                overwriteHookError = std::current_exception();
            }
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
            ledger = JuicerCuda::DeviceAllocationLedger::create({0, 8192}, diagnostic);
            ASSERT_NE(ledger, nullptr) << diagnostic;
            ASSERT_TRUE(ledger->bind_or_validate_cap(8192, diagnostic)) << diagnostic;
            resources = std::make_unique<JuicerCuda::Resources>(key, 1, ledger);
            require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            require_cuda(cudaMalloc(&destination, kCap));
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
        }
        void TearDown() override {
            JuicerCuda::PinnedUploadTest::errors = {};
            JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = false;
            JuicerCuda::PinnedUploadTest::overwriteStaged = {};
            JuicerCuda::PinnedUploadTest::overwriteHookError = nullptr;
            JuicerCuda::PinnedUploadTest::observeOperation = {};
            JuicerCuda::PinnedUploadTest::operationHookError = nullptr;
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
        bool prepare_print_observing(
            const JuicerCuda::PrintResourceInput& input,
            std::function<void(Operation)> observe) {
            JuicerCuda::PinnedUploadTest::observeOperation = std::move(observe);
            const bool prepared = JuicerCuda::prepare_print_resources(*resources, input, stream, failure);
            JuicerCuda::PinnedUploadTest::observeOperation = {};
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::operationHookError, nullptr);
            return prepared;
        }
        void expect_print_curve(
            const JuicerCuda::DeviceCurve& retained,
            const std::array<float, 2>& expectedAxis,
            const std::array<float, 2>& expectedSamples) {
            EXPECT_EQ(resources->printDcC.x, retained.x);
            EXPECT_EQ(resources->printDcC.y, retained.y);
            EXPECT_EQ(resources->printDcC.n, retained.n);
            std::array<float, 2> actualAxis{};
            std::array<float, 2> actualSamples{};
            require_cuda(cudaMemcpy(actualAxis.data(), retained.x, sizeof(actualAxis), cudaMemcpyDeviceToHost));
            require_cuda(cudaMemcpy(actualSamples.data(), retained.y, sizeof(actualSamples), cudaMemcpyDeviceToHost));
            EXPECT_EQ(actualAxis, expectedAxis);
            EXPECT_EQ(actualSamples, expectedSamples);
        }
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> ledger;
        std::unique_ptr<JuicerCuda::Resources> resources;
        cudaStream_t stream = nullptr;
        void* destination = nullptr;
        JuicerCuda::Failure failure;
    };

    struct PrintUploadInput {
        std::array<float, 2> axis{-2.0f, 0.0f};
        std::array<std::array<float, 3>, 2> density{{{0.25f, 0.5f, 0.75f}, {1.0f, 1.25f, 1.5f}}};
        std::array<std::array<float, 3>, Spectral::kNumSamples> spectra{};
        std::array<float, Spectral::kNumSamples> illuminant{};
        JuicerCuda::PrintResourceInput input;

        PrintUploadInput() {
            illuminant.fill(1.0f);
            input.descriptors.hash = 1;
            input.descriptors.profileTables.hash = 1;
            input.descriptors.filmDensityTables.hash = 1;
            input.descriptors.mainIlluminant.hash = 1;
            input.descriptors.balance.hash = 1;
            input.filmChannelDensityCmy = JuicerCuda::ThreeChannelSamplesView(spectra);
            input.filmBaseDensity = illuminant;
            input.printLogExposure = axis;
            input.printDensityCurvesCmy = JuicerCuda::ThreeChannelSamplesView(density);
            input.printSensitivityCmy = JuicerCuda::ThreeChannelSamplesView(spectra);
            input.mainIlluminant = illuminant;
        }

        void change_curve(std::uint64_t hash) {
            axis = {-1.5f, 0.5f};
            density[0][0] = 0.375f;
            density[1][0] = 1.125f;
            input.descriptors.hash = hash;
            input.descriptors.profileTables.hash = hash;
        }

        std::array<float, 2> cyan_samples() const {
            return {density[0][0], density[1][0]};
        }
    };

    TEST_F(PinnedUpload, FrameUseRegisteredDuringStagingFencesBothCurveCopies) {
        PrintUploadInput oldPrint;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, oldPrint.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto curve = resources->printDcC;
        PrintUploadInput nextPrint;
        nextPrint.change_curve(2);

        cudaStream_t reader = nullptr;
        require_cuda(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking));
        const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> readerOwner(reader, &cudaStreamDestroy);
        std::promise<void> staged;
        std::promise<void> registered;
        auto registration = registered.get_future();
        auto overwrite = std::async(std::launch::async, [&] {
            require_cuda(cudaSetDevice(key.deviceId));
            JuicerCuda::PinnedUploadTest::overwriteStaged = [&](JuicerCuda::Resources&) {
                staged.set_value();
                EXPECT_EQ(registration.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            };
            JuicerCuda::Failure uploadError;
            EXPECT_TRUE(JuicerCuda::prepare_print_resources(*resources, nextPrint.input, stream, uploadError)) << uploadError.diagnostic;
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::overwriteHookError, nullptr);
        });
        ASSERT_EQ(staged.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
        UploadGate readerGate(reader);
        auto* const previousWords = static_cast<float*>(destination);
        require_cuda(cudaMemcpyAsync(previousWords, curve.x, sizeof(oldPrint.axis), cudaMemcpyDeviceToDevice, reader));
        require_cuda(cudaMemcpyAsync(previousWords + 2, curve.y, sizeof(oldPrint.axis), cudaMemcpyDeviceToDevice, reader));
        EXPECT_TRUE(JuicerCuda::record_frame_use_event(*resources, reader, "overwrite old curve use", failure)) << failure.diagnostic;
        registered.set_value();
        overwrite.get();
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
        readerGate.release();
        require_cuda(cudaStreamSynchronize(reader));
        require_cuda(cudaStreamSynchronize(stream));
        EXPECT_FALSE(readerGate.timed_out());
        EXPECT_EQ(resources->printDcC.x, curve.x);
        EXPECT_EQ(resources->printDcC.y, curve.y);
        std::array<float, 4> previous{};
        require_cuda(cudaMemcpy(previous.data(), previousWords, sizeof(previous), cudaMemcpyDeviceToHost));
        const std::array<float, 4> expectedPrevious{oldPrint.axis[0], oldPrint.axis[1], oldPrint.density[0][0], oldPrint.density[1][0]};
        EXPECT_EQ(previous, expectedPrevious);
        std::array<float, 2> actualAxis{};
        std::array<float, 2> actualDensity{};
        require_cuda(cudaMemcpy(actualAxis.data(), curve.x, sizeof(actualAxis), cudaMemcpyDeviceToHost));
        require_cuda(cudaMemcpy(actualDensity.data(), curve.y, sizeof(actualDensity), cudaMemcpyDeviceToHost));
        EXPECT_EQ(actualAxis, nextPrint.axis);
        const std::array<float, 2> expectedDensity{nextPrint.density[0][0], nextPrint.density[1][0]};
        EXPECT_EQ(actualDensity, expectedDensity);

        JuicerCuda::PinnedUploadTest::calls = {};
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, nextPrint.input, stream, failure)) << failure.diagnostic;
        const std::array<int, 5> noUpload{};
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls, noUpload);
    }

    TEST_F(PinnedUpload, FrameUseRegistrationIsExcludedAtBothCurveCopySeams) {
        const std::vector<std::string> seams{"before-x", "before-y"};
        JuicerTests::GtestCohort cases("copy-seam", seams);
        PrintUploadInput print;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));

        for (std::size_t seam = 0; seam < seams.size(); ++seam) {
            auto row = cases.case_event(seams[seam]);
            print.change_curve(seam + 2);
            cudaStream_t reader = nullptr;
            require_cuda(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking));
            const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> readerOwner(reader, &cudaStreamDestroy);
            UploadGate readerGate(reader);
            ASSERT_TRUE(JuicerCuda::record_frame_use_event(*resources, reader, "overwrite fence seed", failure)) << failure.diagnostic;
            std::promise<void> copyPaused;
            auto pause = copyPaused.get_future();
            std::promise<bool> resourceLockAvailable;
            auto lockProbe = resourceLockAvailable.get_future();
            std::atomic<int> submittedCopies{0};
            auto contender = std::async(std::launch::async, [&] {
                require_cuda(cudaSetDevice(key.deviceId));
                if (pause.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                    throw std::runtime_error("overwrite did not reach the selected copy seam");
                }
                // Probe from the registration thread while the copy is paused.
                // The serving lock is irrelevant to frame-use registration.
                std::unique_lock lock(resources->m, std::try_to_lock);
                const bool available = lock.owns_lock();
                if (available) {
                    lock.unlock();
                }
                resourceLockAvailable.set_value(available);
                JuicerCuda::Failure registrationError;
                EXPECT_TRUE(JuicerCuda::record_frame_use_event(*resources, reader, "overwrite registration contender", registrationError))
                    << registrationError.diagnostic;
                return submittedCopies.load();
            });
            std::size_t copyIndex = 0;
            const bool prepared = prepare_print_observing(print.input, [&](Operation operation) {
                if (operation == Operation::RecordEvent) {
                    ++submittedCopies;
                }
                if (operation != Operation::Copy || copyIndex++ != seam) {
                    return;
                }
                copyPaused.set_value();
                if (lockProbe.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                    ADD_FAILURE() << "registration thread did not reach the resource lock";
                    return;
                }
                const bool available = lockProbe.get();
                EXPECT_FALSE(available);
                // If the lock was released, give the real registration time to
                // finish before resuming the copy. The probe itself must fail.
                const auto completion = contender.wait_for(available ? std::chrono::seconds(5) : std::chrono::seconds(0));
                EXPECT_EQ(completion, std::future_status::timeout);
            });
            EXPECT_GE(contender.get(), 2);
            ASSERT_TRUE(prepared) << failure.diagnostic;
            readerGate.release();
            require_cuda(cudaStreamSynchronize(reader));
            require_cuda(cudaStreamSynchronize(stream));
            EXPECT_FALSE(readerGate.timed_out());
            row.passed();
        }
    }

    TEST_F(PinnedUpload, CurveSamplesRetryReusesAxisReservationAfterCapPressure) {
        PrintUploadInput print;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto retained = resources->printDcC;
        const auto before = ledger->snapshot();
        const auto oldSamples = print.cyan_samples();
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        warm(std::vector<std::uint8_t>(kCap, 0));
        const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        ASSERT_EQ(block.capacity, kCap);
        print.change_curve(2);

        bool staged = false;
        JuicerCuda::PinnedUploadTest::overwriteStaged = [&](JuicerCuda::Resources&) {
            const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
            EXPECT_EQ(state.totalBytes, kCap);
            EXPECT_EQ(state.reserved, 1);
            EXPECT_EQ(state.available + state.inFlight + state.quarantined, 0);
            staged = true;
        };
        std::vector<Operation> operations;
        std::size_t copies = 0;
        ASSERT_TRUE(prepare_print_observing(print.input, [&](Operation operation) {
            operations.push_back(operation);
            if (operation == Operation::Copy && ++copies == 2) {
                const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
                EXPECT_EQ(state.reserved, 1);
                ASSERT_EQ(state.blocks.size(), 1);
                EXPECT_EQ(state.blocks.front().id, block.id);
                EXPECT_EQ(state.blocks.front().pointer, block.pointer);
                EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
                expect_print_curve(retained, print.axis, oldSamples);
            }
            // Prove completion on the real stream before each reuse of the sole block.
            if (operation == Operation::RecordEvent) {
                inject(operation, cudaErrorInvalidValue);
            }
        })) << failure.diagnostic;
        EXPECT_TRUE(staged);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::overwriteHookError, nullptr);
        const std::array perCopy{Operation::Copy, Operation::RecordEvent, Operation::Synchronize};
        constexpr std::size_t expectedCopies = 9;
        ASSERT_EQ(operations.size(), expectedCopies * perCopy.size());
        for (std::size_t copy = 0; copy < expectedCopies; ++copy) {
            EXPECT_TRUE(std::equal(perCopy.begin(), perCopy.end(), operations.begin() + static_cast<std::ptrdiff_t>(copy * perCopy.size())));
        }
        const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
        EXPECT_EQ(state.available, 1);
        EXPECT_EQ(state.reserved + state.inFlight + state.quarantined, 0);
        EXPECT_EQ(resources->printProfileTablesDescriptorHash, 2u);
        EXPECT_EQ(resources->printPreparationDescriptorHash, 2u);
        EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
        EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
        expect_print_curve(retained, print.axis, print.cyan_samples());
    }

    TEST_F(PinnedUpload, CurveAxisSubmissionDeterminesSamplesFailurePrecedence) {
        const std::vector<std::string> orders{"axis-copy-and-samples-stage-fail", "samples-stage-fails", "samples-copy-fails"};
        JuicerTests::GtestCohort cases("failure-order", orders);
        PrintUploadInput oldPrint;
        PrintUploadInput nextPrint;
        nextPrint.change_curve(2);

        for (std::size_t order = 0; order < orders.size(); ++order) {
            auto row = cases.case_event(orders[order]);
            ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, oldPrint.input, stream, failure)) << failure.diagnostic;
            require_cuda(cudaStreamSynchronize(stream));
            const auto retained = resources->printDcC;
            const auto before = ledger->snapshot();
            JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);

            const bool failAxis = order == 0;
            const bool failSamplesCopy = order == 2;
            std::vector<Operation> operations;
            std::size_t allocations = 0;
            std::size_t copies = 0;
            ASSERT_FALSE(prepare_print_observing(nextPrint.input, [&](Operation operation) {
                operations.push_back(operation);
                if (operation == Operation::Allocate && ++allocations == 2 && !failSamplesCopy) {
                    const auto state = JuicerCuda::PinnedUploadTest::snapshot(key);
                    EXPECT_EQ(state.reserved, 1);
                    EXPECT_EQ(state.totalBytes, sizeof(oldPrint.axis));
                    inject(operation, cudaErrorMemoryAllocation);
                }
                if (operation == Operation::Copy && (++copies == 1 ? failAxis : failSamplesCopy)) {
                    inject(operation, cudaErrorInvalidValue);
                }
            }));
            std::vector expectedOperations{Operation::Allocate, Operation::CreateEvent, Operation::Allocate};
            if (failSamplesCopy) {
                expectedOperations.push_back(Operation::CreateEvent);
            }
            expectedOperations.push_back(Operation::Copy);
            if (!failAxis) {
                expectedOperations.push_back(Operation::RecordEvent);
            }
            if (failSamplesCopy) {
                expectedOperations.push_back(Operation::Copy);
            }
            EXPECT_EQ(operations, expectedOperations);
            const auto expectedCode = failAxis || failSamplesCopy ? cudaErrorInvalidValue : cudaErrorMemoryAllocation;
            expect_status({FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, expectedCode});
            EXPECT_NE(failure.diagnostic.find(failAxis ? "dcC.x upload failed" : "dcC.y upload failed"), std::string::npos);
            EXPECT_EQ(resources->printProfileTablesDescriptorHash, 1u);
            EXPECT_EQ(resources->printPreparationDescriptorHash, 1u);
            EXPECT_FALSE(resources->frameUseFenceQuarantined);
            const auto staging = JuicerCuda::PinnedUploadTest::snapshot(key);
            EXPECT_EQ(staging.reserved + staging.quarantined, 0);
            EXPECT_EQ(staging.available, failAxis || failSamplesCopy ? 1u : 0u);
            EXPECT_EQ(staging.inFlight, failAxis ? 0u : 1u);
            EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
            EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);

            require_cuda(cudaStreamSynchronize(stream));
            JuicerCuda::PinnedUploadTest::poll(key);
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).available, failSamplesCopy ? 2u : 1u);
            expect_print_curve(retained, failAxis ? oldPrint.axis : nextPrint.axis, oldPrint.cyan_samples());

            ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, nextPrint.input, stream, failure)) << failure.diagnostic;
            require_cuda(cudaStreamSynchronize(stream));
            EXPECT_EQ(resources->printProfileTablesDescriptorHash, 2u);
            EXPECT_EQ(resources->printPreparationDescriptorHash, 2u);
            EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
            EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
            expect_print_curve(retained, nextPrint.axis, nextPrint.cyan_samples());
            row.passed();
        }
    }

    TEST_F(PinnedUpload, OverwriteRejectsStaleEpochAndChangedDestination) {
        PrintUploadInput print;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto curve = resources->printDcC;
        const auto epoch = resources->contextEpoch;
        const auto before = ledger->snapshot();
        print.input.descriptors.hash = 2;
        print.input.descriptors.profileTables.hash = 2;
        print.axis = {-1.75f, 0.25f};

        JuicerCuda::PinnedUploadTest::calls = {};
        JuicerCuda::PinnedUploadTest::overwriteStaged = [](JuicerCuda::Resources& stagedResources) {
            std::lock_guard lock(stagedResources.m);
            ++stagedResources.contextEpoch;
        };
        const bool acceptedEpoch = JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure);
        resources->contextEpoch = epoch;
        EXPECT_FALSE(acceptedEpoch);
        EXPECT_NE(failure.diagnostic.find("owner changed"), std::string::npos);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);

        JuicerCuda::PinnedUploadTest::overwriteStaged = [](JuicerCuda::Resources& stagedResources) {
            std::lock_guard lock(stagedResources.m);
            std::swap(stagedResources.printDcC.x, stagedResources.printDcC.y);
        };
        const bool acceptedPointer = JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure);
        resources->printDcC = curve;
        EXPECT_FALSE(acceptedPointer);
        EXPECT_NE(failure.diagnostic.find("destination changed"), std::string::npos);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::overwriteHookError, nullptr);
        EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
        EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
        std::array<float, 2> actual{};
        require_cuda(cudaMemcpy(actual.data(), curve.x, sizeof(actual), cudaMemcpyDeviceToHost));
        const std::array<float, 2> expected{-2.0f, 0.0f};
        EXPECT_EQ(actual, expected);
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
    }

    TEST_F(PinnedUpload, OverwriteReturnsReservationsOnAllocationEventAndCopyFailure) {
        PrintUploadInput print;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        JuicerCuda::purge_pinned_upload_staging_for_context(key.deviceId, key.contextOpaque, PinnedUploadPurgeDisposition::NormalRetire);
        const auto before = ledger->snapshot();
        float* const illuminant = resources->printIllumFiltered;
        print.illuminant.fill(2.0f);
        print.input.descriptors.hash = 2;
        print.input.descriptors.mainIlluminant.hash = 2;
        const std::vector<std::string> operations{"allocate", "create-event", "copy"};
        JuicerTests::GtestCohort faults("operation", operations);
        const std::array faultOperations{Operation::Allocate, Operation::CreateEvent, Operation::Copy};
        for (std::size_t index = 0; index < faultOperations.size(); ++index) {
            auto row = faults.case_event(operations[index]);
            const auto operation = faultOperations[index];
            inject(operation, cudaErrorInvalidValue);
            ASSERT_FALSE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure));
            expect_status({FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, cudaErrorInvalidValue});
            EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
            EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
            EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
            EXPECT_EQ(resources->printIllumFiltered, illuminant);
            float actual = 0.0f;
            require_cuda(cudaMemcpy(&actual, illuminant, sizeof(actual), cudaMemcpyDeviceToHost));
            EXPECT_EQ(actual, 1.0f);
            row.passed();
        }
        inject(Operation::RecordEvent, cudaErrorInvalidValue);
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        EXPECT_FALSE(resources->frameUseFenceQuarantined);
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
        float actual = 0.0f;
        require_cuda(cudaMemcpy(&actual, illuminant, sizeof(actual), cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual, 2.0f);
    }

    TEST_F(PinnedUpload, UnfencedOverwriteRetainsDestinationUntilExactContextDrain) {
        PrintUploadInput print;
        ASSERT_TRUE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure)) << failure.diagnostic;
        require_cuda(cudaStreamSynchronize(stream));
        const auto before = ledger->snapshot();
        float* const illuminant = resources->printIllumFiltered;
        print.illuminant.fill(3.0f);
        print.input.descriptors.hash = 2;
        print.input.descriptors.mainIlluminant.hash = 2;
        inject(Operation::RecordEvent, cudaErrorInvalidValue);
        inject(Operation::Synchronize, cudaErrorContextIsDestroyed);
        ASSERT_FALSE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure));
        expect_status({FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_RUNTIME, cudaErrorContextIsDestroyed});
        EXPECT_TRUE(resources->frameUseFenceQuarantined);
        EXPECT_EQ(resources->unfencedFrameUseStreamOpaque, stream);
        EXPECT_EQ(resources->printIllumFiltered, illuminant);
        EXPECT_EQ(resources->printMainIlluminantDescriptorHash, 1u);
        const auto staging = JuicerCuda::PinnedUploadTest::snapshot(key);
        EXPECT_EQ(staging.reserved, 0);
        EXPECT_EQ(staging.quarantined, 1);
        EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
        EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);
        EXPECT_FALSE(JuicerCuda::prepare_print_resources(*resources, print.input, stream, failure));

        ASSERT_EQ(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
        CUcontext other = nullptr;
        const auto createOther = cuCtxCreate(&other, nullptr, 0, key.deviceId);
        if (createOther != CUDA_SUCCESS) {
            (void)cuCtxSetCurrent(static_cast<CUcontext>(key.contextOpaque));
        }
        ASSERT_EQ(createOther, CUDA_SUCCESS);
        const bool drainedOther = JuicerCuda::drain_for_context_retire(*resources, failure);
        const auto destroyOther = cuCtxDestroy(other);
        ASSERT_EQ(cuCtxSetCurrent(static_cast<CUcontext>(key.contextOpaque)), CUDA_SUCCESS);
        EXPECT_EQ(destroyOther, CUDA_SUCCESS);
        EXPECT_FALSE(drainedOther);
        EXPECT_TRUE(resources->frameUseFenceQuarantined);
        EXPECT_EQ(ledger->snapshot().chargedBytes, before.chargedBytes);
        EXPECT_EQ(ledger->snapshot().recordCount, before.recordCount);

        ASSERT_TRUE(JuicerCuda::drain_for_context_retire(*resources, failure)) << failure.diagnostic;
        EXPECT_FALSE(resources->frameUseFenceQuarantined);
        EXPECT_EQ(resources->printIllumFiltered, nullptr);
        EXPECT_EQ(ledger->snapshot().chargedBytes, 0);
        EXPECT_EQ(ledger->snapshot().recordCount, 0);
    }

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
        JuicerTests::GtestCohort cases("reservation-failure", std::vector<std::string>{"allocate-memory", "allocate-invalid", "allocate-context-loss", "event-memory", "event-invalid", "event-context-loss"});
        for (const auto operation : {Operation::Allocate, Operation::CreateEvent}) {
            for (const auto error : {cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorContextIsDestroyed}) {
                const auto* const operationName = operation == Operation::Allocate ? "allocate" : "event";
                const auto* const errorName = error == cudaErrorMemoryAllocation ? "memory" : error == cudaErrorInvalidValue ? "invalid"
                                                                                                                             : "context-loss";
                auto row = cases.case_event(std::string(operationName) + "-" + errorName);
                inject(operation, error);
                ASSERT_FALSE(upload(source.data(), source.size()));
                expect_status({error == cudaErrorContextIsDestroyed ? FJ_STATUS_CONTEXT_LOSS : FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, error});
                EXPECT_FALSE(JuicerCuda::allocation_capacity_exhausted(failure));
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).reserved, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
                row.passed();
            }
        }
    }

    TEST_F(PinnedUpload, FailedStagedCopyReturnsReservationAndNeverBorrowsCallerStorage) {
        const std::array<std::uint8_t, 64> source{};
        JuicerTests::GtestCohort cases("copy-failure", std::vector<std::string>{"invalid", "memory", "context-loss"});
        for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorContextIsDestroyed}) {
            const auto* const errorName = error == cudaErrorInvalidValue ? "invalid" : error == cudaErrorMemoryAllocation ? "memory"
                                                                                                                          : "context-loss";
            auto row = cases.case_event(errorName);
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
            row.passed();
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
        JuicerTests::GtestCohort cases("reservation-diagnostic-failure", std::vector<std::string>{"allocate-memory", "allocate-invalid", "allocate-context-loss", "event-memory", "event-invalid", "event-context-loss"});
        for (const auto operation : {Operation::Allocate, Operation::CreateEvent}) {
            for (const auto error : {cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorContextIsDestroyed}) {
                const auto* const operationName = operation == Operation::Allocate ? "allocate" : "event";
                const auto* const errorName = error == cudaErrorMemoryAllocation ? "memory" : error == cudaErrorInvalidValue ? "invalid"
                                                                                                                             : "context-loss";
                auto row = cases.case_event(std::string(operationName) + "-" + errorName);
                inject(operation, error);
                JuicerCuda::PinnedUploadTest::failDiagnosticAllocation = true;
                bool uploaded = true;
                ASSERT_NO_THROW(uploaded = upload(source.data(), source.size()));
                EXPECT_FALSE(uploaded);
                expect_status({error == cudaErrorContextIsDestroyed ? FJ_STATUS_CONTEXT_LOSS : FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, error}, false);
                EXPECT_FALSE(JuicerCuda::PinnedUploadTest::failDiagnosticAllocation);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::snapshot(key).totalBytes, 0);
                EXPECT_EQ(JuicerCuda::PinnedUploadTest::calls[static_cast<std::size_t>(Operation::Copy)], 0);
                row.passed();
            }
        }
        EXPECT_EQ(JuicerCuda::PinnedUploadTest::diagnosticCalls, 6);
    }

    TEST_F(PinnedUpload, CopyDiagnosticFailureReturnsReservationAndPreservesNativeStatus) {
        const std::vector<std::uint8_t> source(64, 0x52);
        warm(source);
        const auto block = JuicerCuda::PinnedUploadTest::snapshot(key).blocks.front();
        JuicerTests::GtestCohort cases("copy-diagnostic-failure", std::vector<std::string>{"invalid", "memory", "context-loss"});
        for (const auto error : {cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorContextIsDestroyed}) {
            const auto* const errorName = error == cudaErrorInvalidValue ? "invalid" : error == cudaErrorMemoryAllocation ? "memory"
                                                                                                                          : "context-loss";
            auto row = cases.case_event(errorName);
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
            row.passed();
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
        JuicerTests::GtestCohort cases("completion-diagnostic-failure", std::vector<std::string>{"record-valid-sync-loss", "record-loss-sync-invalid"});
        for (const bool recordLosesContext : {false, true}) {
            auto row = cases.case_event(recordLosesContext ? "record-loss-sync-invalid" : "record-valid-sync-loss");
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
            row.passed();
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
