#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/Diffusion/DiffusionInternals.h"
#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/Resources/ResourceInternals.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

namespace {
    bool failFree = false;
    unsigned freeAttempts = 0;
    void* failBuffer = nullptr;
    unsigned bufferFailures = 0;
    unsigned hostFrees = 0;
    unsigned eventDestroys = 0;
    unsigned planFailures = 0;
    bool failHost = false;
    bool failEventDestroy = false;
    bool failPlanDestroy = false;
    cudaEvent_t pendingQuery = nullptr;
    cudaError_t recordFault = cudaSuccess;
    cudaError_t waitFault = cudaSuccess;
    cudaStream_t failWaitStream = nullptr;
    cudaError_t contextWaitFault = cudaSuccess;
    void* contextLossBuffer = nullptr;
    cufftResult transformFault = CUFFT_SUCCESS;

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    struct PendingWork {
        std::mutex mutex;
        std::condition_variable wake;
        bool released = false;
        std::atomic<bool> completed{false};
        std::promise<void> barrierEntered;
        std::atomic<bool> earlyFree{false};

        static void CUDART_CB hold(void* data) {
            auto& work = *static_cast<PendingWork*>(data);
            std::unique_lock lock(work.mutex);
            (void)work.wake.wait_for(lock, std::chrono::seconds(10), [&] {
                return work.released;
            });
        }

        static void CUDART_CB complete(void* data) noexcept {
            static_cast<PendingWork*>(data)->completed.store(true);
        }

        ~PendingWork() {
            release();
            (void)cudaStreamSynchronize(cudaStreamPerThread);
        }

        void release() {
            {
                const std::lock_guard lock(mutex);
                released = true;
            }
            wake.notify_all();
        }
    };

    PendingWork* pendingWork = nullptr;

    template <typename Drain>
    void check_cross_thread_drain(CUcontext context, PendingWork& work, const JuicerCuda::DeviceAllocationLedger& ledger, Drain drain) {
        const auto charged = ledger.snapshot().chargedBytes;
        pendingWork = &work;
        auto barrier = work.barrierEntered.get_future();
        auto retired = std::async(std::launch::async, [&] {
            return cuCtxSetCurrent(context) == CUDA_SUCCESS && drain();
        });
        const bool waitedForProducer = barrier.wait_for(std::chrono::seconds(2)) == std::future_status::ready &&
                                       retired.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout &&
                                       !work.completed.load();
        const bool heldCharge = ledger.snapshot().chargedBytes == charged;
        work.release();
        const bool success = retired.get();
        pendingWork = nullptr;
        require(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "finish per-thread producer stream");
        require(waitedForProducer && heldCharge && !work.earlyFree.load(),
                "retirement did not prove producer completion before freeing or uncharging");
        require(success && ledger.snapshot().recordCount == 0, "retirement after producer completion failed");
    }

    JuicerCuda::Resources* allocate(const JuicerCuda::ResourceManager::DeviceContextKey& key,
                                    const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& ledger) {
        auto resources = std::make_unique<JuicerCuda::Resources>(key, 1, ledger);
        JuicerCuda::Failure failure;
        JuicerCuda::DeviceByteReservation reservation;
        require(ledger->reserve({key, 1, 4096}, reservation, failure), "reserve 4 KiB");
        void* allocation = nullptr;
        require(cudaMalloc(&allocation, 4096) == cudaSuccess, "allocate 4 KiB");
        std::string diagnostic;
        require(reservation.commit(4096, diagnostic), "commit 4 KiB");
        resources->deviceAllocationRecords.try_emplace(allocation, allocation, std::move(reservation));
        return resources.release();
    }

    void check() {
        require(cuInit(0) == CUDA_SUCCESS, "initialize CUDA driver");
        CUdevice device{};
        require(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "select device");
        std::array<CUcontext, 3> contexts{};
        for (auto& context : contexts) {
            require(cuCtxCreate(&context, nullptr, 0, device) == CUDA_SUCCESS, "create fixture context");
        }
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, contexts[0]};
        const JuicerCuda::ResourceManager::DeviceContextKey unrelated{0, contexts[1]};
        std::string diagnostic;
        auto ledger = JuicerCuda::DeviceAllocationLedger::create({0, 65536}, diagnostic);
        require(ledger && ledger->bind_or_validate_cap(65536, diagnostic), "bind ledger cap");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "make owner current");
        std::array<JuicerCuda::Resources*, 2> resources{allocate(key, ledger), allocate(key, ledger)};
        require(cuCtxSetCurrent(contexts[1]) == CUDA_SUCCESS, "make unrelated owner current");
        auto* const other = allocate(unrelated, ledger);
        require(cuCtxSetCurrent(contexts[2]) == CUDA_SUCCESS, "make foreign context current");
        for (auto* resource : resources) {
            JuicerCuda::destroy(resource);
        }
        JuicerCuda::destroy(other);
        require(ledger->snapshot().chargedBytes == 12288 && ledger->snapshot().recordCount == 3, "deferred accounting before drain");
        JuicerCuda::Failure failure;
        require(!JuicerCuda::drain_deferred_resources(key, 1, failure), "foreign context drained the owning epoch");
        require(ledger->snapshot().recordCount == 3 && freeAttempts == 0, "foreign context discarded quarantined ownership");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "restore exact owner");
        require(JuicerCuda::drain_deferred_resources(key, 2, failure), "empty successor epoch drain failed");
        require(ledger->snapshot().recordCount == 3 && freeAttempts == 0, "successor epoch drained older ownership");
        // A runtime call fails after the exact-context batch has been detached.
        failFree = true;
        require(!JuicerCuda::drain_deferred_resources(key, 1, failure), "faulted drain succeeded");
        require(failure.status.category == FJ_STATUS_CUDA_FAILURE &&
                    failure.status.api == FJ_API_CUDA_RUNTIME &&
                    failure.status.native_code == cudaErrorInvalidValue,
                "drain lost the typed CUDA free failure");
        require(ledger->snapshot().chargedBytes == 12288 && ledger->snapshot().recordCount == 3,
                "failure discarded the batch or its ledger charges");
        require(freeAttempts == 1, "failed drain retried or freed a later entry");
        require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "restore owner after fault");
        require(JuicerCuda::drain_deferred_resources(key, 1, failure), "explicit drain after clearing fault");
        std::printf("after retry charged=%llu records=%llu attempts=%u\n",
                    static_cast<unsigned long long>(ledger->snapshot().chargedBytes),
                    static_cast<unsigned long long>(ledger->snapshot().recordCount),
                    freeAttempts);
        require(ledger->snapshot().chargedBytes == 4096 && ledger->snapshot().recordCount == 1, "failed or unprocessed owner lost, duplicated or unrelated owner freed");
        require(ledger->record_count_for_context(unrelated, 1) == 1, "unrelated exact-context entry changed");
        require(JuicerCuda::drain_deferred_resources(key, 1, failure), "empty drain failed");
        require(cuCtxSetCurrent(contexts[1]) == CUDA_SUCCESS, "restore unrelated owner");
        require(JuicerCuda::drain_deferred_resources(unrelated, 1, failure), "unrelated deferred drain");
        require(ledger->snapshot().chargedBytes == 0 && ledger->snapshot().recordCount == 0, "deferred ownership did not drain to zero");
        require(freeAttempts == 4u, "deferred entries duplicated or reprocessed");
        for (auto* context : contexts) {
            require(cuCtxDestroy(context) == CUDA_SUCCESS, "destroy fixture-owned context");
        }
    }
    void check_event_and_host_retirement() {
        using namespace JuicerCuda;
        CUcontext context = nullptr;
        require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "current event fixture context");
        const ResourceManager::DeviceContextKey key{0, context};
        std::string diagnostic;
        auto ledger = DeviceAllocationLedger::create({0, 65536}, diagnostic);
        require(ledger && ledger->bind_or_validate_cap(65536, diagnostic), "bind event fixture ledger");

        Failure failure;
        constexpr std::uint64_t epoch = 17;
        cudaStream_t stream = nullptr;
        require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "create resource stream");
        auto* resources = create(key, epoch, ledger, failure);
        require(resources != nullptr, "create event owner");

        recordFault = cudaErrorInvalidResourceHandle;
        failEventDestroy = true;
        require(!record_frame_use_event(*resources, stream, "fixture", failure), "failed record succeeded");
        require(failure.status.native_code == cudaErrorInvalidResourceHandle, "failed event cleanup replaced record status");
        require(resources->pendingFrameUseEvents.size() == 1 &&
                    resources->pendingFrameUseEvents.front().event &&
                    !resources->pendingFrameUseEvents.front().recorded,
                "unrecorded event escaped its owner");
        const auto beforeEventRetry = eventDestroys;
        require(!drain_for_context_retire(*resources, failure), "persistent event destroy failure drained");
        require(eventDestroys == beforeEventRetry && resources->pendingFrameUseEvents.front().event,
                "event handle disappeared while disposal failed");
        failEventDestroy = false;
        CUcontext foreign = nullptr;
        require(cuCtxCreate(&foreign, nullptr, 0, 0) == CUDA_SUCCESS, "create foreign event context");
        require(!record_frame_use_event(*resources, stream, "fixture", failure), "foreign context recorded a frame use event");
        require(eventDestroys == beforeEventRetry && resources->pendingFrameUseEvents.front().event,
                "foreign context attempted event disposal before owner validation");
        require(cuCtxDestroy(foreign) == CUDA_SUCCESS && cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore event owner");
        require(drain_for_context_retire(*resources, failure), "event destruction retry failed");
        require(eventDestroys == beforeEventRetry + 1 && resources->pendingFrameUseEvents.empty(),
                "event destruction retry was not exactly once");

        ScanErrorBuffer scan{};
        require(acquire_scan_error_buffer(*resources, scan, failure), "acquire scan staging");
        failEventDestroy = true;
        require(!release_scan_error_buffer(*resources, scan.host, failure), "failed scan event destroy succeeded");
        require(resources->scanErrorBuffers.front().event.get() == scan.event,
                "failed scan event destroy dropped its handle");
        failEventDestroy = false;
        require(acquire_scan_error_buffer(*resources, scan, failure), "retry scan event disposal");

        void* highlight = nullptr;
        cudaEvent_t highlightEvent = nullptr;
        require(allocate_frame_status_staging(*resources, sizeof(int), highlight, highlightEvent, failure),
                "allocate highlight staging");
        require(cudaStreamSynchronize(stream) == cudaSuccess, "complete fixture work before synthetic loss");
        std::uint64_t released = 0;
        require(ledger->release_context_after_proven_loss(key, epoch, released, diagnostic) && released == 2 * sizeof(int),
                "invalidate only the two scan flag charges");
        invalidate_resources_after_proven_context_loss(*resources);

        failHost = true;
        const auto beforeHostRetry = hostFrees;
        const auto beforeDeviceRetry = freeAttempts;
        const auto beforeInvalidatedEvents = eventDestroys;
        require(!drain_for_context_retire(*resources, failure), "failed pinned frees drained");
        require(resources->retireQueue.size() == 1 && resources->retireBytes == sizeof(int) &&
                    resources->retireQueue.front().pinnedBuffer.get() == highlight &&
                    resources->scanErrorBuffers.front().host.get() == scan.host,
                "failed pinned frees discarded host handles or accounting");
        destroy(resources);
        auto* successor = create(key, epoch + 1, ledger, failure);
        require(successor != nullptr && ledger->record_count_for_context(key, epoch) == 0, "create successor epoch");
        require(hostFrees == beforeHostRetry && freeAttempts == beforeDeviceRetry && eventDestroys == beforeInvalidatedEvents,
                "new epoch disposed invalidated device resources or lost the host fault");

        failHost = false;
        require(drain_deferred_resources(key, epoch + 1, failure), "empty successor drain failed");
        require(hostFrees == beforeHostRetry, "successor drain consumed earlier epoch's host allocations");
        auto* trigger = create(key, epoch + 1, ledger, failure);
        require(trigger != nullptr && hostFrees == beforeHostRetry + 3 && freeAttempts == beforeDeviceRetry &&
                    eventDestroys == beforeInvalidatedEvents,
                "host retry was not isolated from invalidated device handles");
        destroy(trigger);
        destroy(successor);
        require(hostFrees == beforeHostRetry + 3, "host retry disposed a handle twice");
        require(cudaStreamDestroy(stream) == cudaSuccess, "destroy resource stream");
    }

    void check_frame_and_scratch_retirement(CUcontext context) {
        using namespace JuicerCuda;
        const ResourceManager::DeviceContextKey key{0, context};
        std::string diagnostic;
        auto ledger = DeviceAllocationLedger::create({0, 65536}, diagnostic);
        require(ledger && ledger->bind_or_validate_cap(65536, diagnostic), "bind frame fixture ledger");
        Failure failure;
        {
            Resources resources(key, 29, ledger);
            void* allocation = nullptr;
            require(allocate_owned_device(resources, &allocation, 4096, "fixture frame allocation", failure), "allocate tracked frame buffer");
            PendingWork work;
            require(cudaLaunchHostFunc(cudaStreamPerThread, PendingWork::hold, &work) == cudaSuccess, "gate per-thread frame use");
            require(cudaMemsetAsync(allocation, 0, 4096, cudaStreamPerThread) == cudaSuccess, "queue device use of tracked frame buffer");
            require(cudaLaunchHostFunc(cudaStreamPerThread, PendingWork::complete, &work) == cudaSuccess, "observe frame device work completion");
            recordFault = cudaErrorInvalidResourceHandle;
            require(!record_frame_use_event(resources, cudaStreamPerThread, "fixture frame", failure), "fault the frame fence");
            failWaitStream = cudaStreamPerThread;
            waitFault = cudaErrorInvalidValue;
            require(ResourceOperations::synchronize_stream(cudaStreamPerThread) == cudaErrorInvalidValue, "fault the immediate frame wait");
            quarantine_unfenced_frame_use(resources);
            waitFault = cudaSuccess;
            check_cross_thread_drain(context, work, *ledger, [&] {
                return drain_for_context_retire(resources, failure);
            });
        }
        {
            Resources resources(key, 30, ledger);
            void* attachment = nullptr;
            require(allocate_owned_device(resources, &attachment, 4096, "fixture dust attachment", failure), "allocate tracked attachment");
            resources.scannerScratch.filmDustTransmittance = static_cast<float*>(attachment);
            resources.scannerScratch.filmDustCapacityElements = 4096 / sizeof(float);
            LargeScratchTransitionReclaimStats stats{};
            recordFault = cudaErrorInvalidResourceHandle;
            failEventDestroy = true;
            require(!reclaim_large_scratch_transition(resources, {}, nullptr, stats, failure), "fault the defect attachment fence");
            require(failure.status.native_code == cudaErrorInvalidResourceHandle &&
                        resources.defectRetirementEvents.front() &&
                        resources.scannerScratch.filmDustTransmittance == attachment && ledger->snapshot().chargedBytes == 4096,
                    "failed defect rollback lost the event, old allocation or charge");
            const auto beforeRetry = eventDestroys;
            failEventDestroy = false;
            require(drain_for_context_retire(resources, failure), "retry defect event and attachment retirement");
            require(eventDestroys == beforeRetry + 1 && !resources.defectRetirementEvents.front() && ledger->snapshot().recordCount == 0,
                    "defect retirement did not dispose of its event and allocation exactly once");
        }
    }

    void check_diffusion_retirement() {
        using namespace JuicerCuda;
        using namespace JuicerCuda::Diffusion;
        CUcontext context = nullptr;
        require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "current diffusion fixture context");
        const ResourceManager::DeviceContextKey key{0, context};
        constexpr std::uint64_t epoch = 23;
        constexpr std::uint64_t cap = std::uint64_t{64} * 1024 * 1024;
        std::string diagnostic;
        auto ledger = DeviceAllocationLedger::create({0, cap}, diagnostic);
        require(ledger && ledger->bind_or_validate_cap(cap, diagnostic), "bind diffusion fixture ledger");

        Spektrafilm::DiffusionFrameSetDescriptor frame{};
        frame.hash = 1;
        frame.fullFrame = {0, 0, 8, 6};
        auto& camera = frame.camera.emplace();
        camera.hash = 2;
        camera.sample.hash = 3;
        camera.sample.radiusPixels = 1;
        camera.sample.components.lambdaPixels.fill(1.);
        camera.sample.components.redWeights.fill(1. / 9.);
        camera.sample.components.greenWeights = camera.sample.components.redWeights;
        camera.sample.components.blueWeights = camera.sample.components.redWeights;

        Spektrafilm::DiffusionExecutionDescriptor execution{};
        require(Spektrafilm::make_plan_layout(32, 32, execution.layout), "make diffusion layout");
        execution.contextEpoch = epoch;
        execution.frameSetHash = frame.hash;
        execution.stageCount = execution.uniqueSpectrumCount = 1;
        execution.spectrumKeys[0] = {camera.sample.hash, {32, 32}, 4};
        execution.planKey = {{32, 32}, 5};
        execution.stagePlaneBytes = std::uint64_t{8} * 6 * 4 * sizeof(float);
        execution.reservedSharedWorkBytes = std::uint64_t{16} * 1024 * 1024;
        execution.planAllowanceBytes = std::uint64_t{128} * 1024;

        std::array<cudaStream_t, 2> streams{};
        for (auto& stream : streams) {
            require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "create diffusion stream");
        }
        Failure failure;
        const auto prepare = [&](DiffusionContextResources& owner, PreparedDiffusionLease& lease, cudaStream_t stream) {
            return prepare_diffusion_resources(owner, key, epoch, ledger, frame, execution, stream, lease, failure);
        };
        {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            recordFault = cudaErrorInvalidResourceHandle;
            failWaitStream = streams[0];
            waitFault = cudaErrorContextIsDestroyed;
            require(!prepare(owner, lease, streams[0]), "unrecorded spectrum build succeeded");
            require(failure.status.api == FJ_API_CUDA_RUNTIME && failure.status.native_code == cudaErrorInvalidResourceHandle &&
                        context_loss(failure) && !owner.hostLeaseActive && !lease.active(),
                    "unrecorded build lost primary status, cleanup loss or host claim");
            const auto& first = owner.spectra.front();
            const auto charge = ledger->snapshot().chargedBytes;
            auto* const firstSpectrum = first.spectra.red;
            require(charge > 0 && firstSpectrum && first.buildEvent && !first.buildEventRecorded && first.pendingWorkspaceMask == 1,
                    "failed build lost its producing workspace dependency");
            require(prepare(owner, lease, streams[1]), failure.diagnostic.c_str());
            require(first.spectra.red == firstSpectrum, "opportunistic reap freed an unfenced spectrum");
            recordFault = cudaErrorInvalidResourceHandle;
            require(release_diffusion_resources(owner, lease, streams[1], true, failure), "second stream fallback failed");
            require(first.spectra.red == firstSpectrum && first.pendingWorkspaceMask == 1,
                    "second stream wait discharged another stream's spectrum dependency");
            contextWaitFault = cudaErrorInvalidValue;
            require(!drain_diffusion_resources(owner, failure), "drain accepted an unfenced workspace");
            require(first.spectra.red == firstSpectrum && first.pendingWorkspaceMask == 1 && ledger->snapshot().chargedBytes == charge,
                    "blocking drain released the unfenced spectrum or its charge");
            waitFault = contextWaitFault = cudaSuccess;
            require(drain_diffusion_resources(owner, failure), "diffusion exact-owner retry failed");
            require(ledger->snapshot().recordCount == 0 && !first.spectra.red, "retry left spectrum ownership behind");
        }
        {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            transformFault = CUFFT_EXEC_FAILED;
            waitFault = cudaErrorContextIsDestroyed;
            require(!prepare(owner, lease, streams[0]), "faulted cuFFT build succeeded");
            require(failure.status.category == FJ_STATUS_CUFFT_FAILURE && failure.status.api == FJ_API_CUFFT &&
                        failure.status.native_code == CUFFT_EXEC_FAILED && context_loss(failure) && !owner.hostLeaseActive,
                    "preparation rollback replaced cuFFT failure or stranded the claim");
            waitFault = cudaSuccess;
            require(drain_diffusion_resources(owner, failure), "faulted cuFFT build did not retire");
            require(ledger->snapshot().recordCount == 0, "cuFFT rollback retained charges after completion");
        }
        for (const bool failFallback : {false, true}) {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            require(prepare(owner, lease, streams[0]), "prepare shared spectrum producer");
            const auto& spectrum = owner.spectra.front();
            auto* const shared = spectrum.spectra.red;
            require(release_diffusion_resources(owner, lease, streams[0], false, failure), "record first workspace completion");
            require(cudaStreamSynchronize(streams[0]) == cudaSuccess, "complete first workspace fixture commands");
            // Keep the first workspace in flight from admission's perspective,
            // without relying on a callback delay or on CUDA allocation timing.
            pendingQuery = owner.workspaces.front().completionEvent.get();
            require(prepare(owner, lease, streams[1]), "prepare second workspace with shared spectrum");
            require(lease.view().spectra.front().spectra.red == shared && spectrum.leased &&
                        spectrum.pendingWorkspaceMask == 1,
                    "second workspace did not lease the first workspace's spectrum");
            require(cudaMemcpyAsync(lease.view().execution.execution.transformBuffer, shared, execution.layout.transformBytes, cudaMemcpyDeviceToDevice, streams[1]) == cudaSuccess,
                    "enqueue shared spectrum use");
            mark_diffusion_work_enqueued(lease);
            recordFault = cudaErrorInvalidResourceHandle;
            failWaitStream = streams[1];
            waitFault = failFallback ? cudaErrorContextIsDestroyed : cudaSuccess;
            require(release_diffusion_resources(owner, lease, streams[1], true, failure) != failFallback,
                    "shared spectrum release missed its fallback result");
            require(spectrum.spectra.red == shared && spectrum.pendingWorkspaceMask == (failFallback ? 3 : 1),
                    "shared spectrum lost a dependency after fallback");
            if (failFallback) {
                require(failure.status.api == FJ_API_CUDA_RUNTIME && failure.status.native_code == cudaErrorInvalidResourceHandle &&
                            context_loss(failure),
                        "release fallback replaced record status or lost cleanup context loss");
                contextWaitFault = cudaErrorInvalidValue;
                require(!drain_diffusion_resources(owner, failure) && spectrum.spectra.red == shared &&
                            spectrum.pendingWorkspaceMask == 2,
                        "blocking drain discharged the unfenced consumer with the other workspace");
                waitFault = contextWaitFault = cudaSuccess;
            }
            pendingQuery = nullptr;
            require(drain_diffusion_resources(owner, failure), "retire both shared spectrum workspaces");
            require(ledger->snapshot().recordCount == 0 && !spectrum.spectra.red, "shared spectrum retry left ownership behind");
        }
        {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            require(prepare(owner, lease, cudaStreamPerThread), "prepare per-thread diffusion workspace");
            PendingWork work;
            require(cudaLaunchHostFunc(cudaStreamPerThread, PendingWork::hold, &work) == cudaSuccess, "gate per-thread spectrum use");
            const auto view = lease.view();
            require(cudaMemcpyAsync(view.execution.execution.transformBuffer, view.spectra.front().spectra.red, execution.layout.transformBytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread) == cudaSuccess,
                    "queue device use of diffusion spectrum and workspace");
            require(cudaLaunchHostFunc(cudaStreamPerThread, PendingWork::complete, &work) == cudaSuccess, "observe diffusion device work completion");
            mark_diffusion_work_enqueued(lease);
            recordFault = cudaErrorInvalidResourceHandle;
            failWaitStream = cudaStreamPerThread;
            waitFault = cudaErrorInvalidValue;
            const bool released = release_diffusion_resources(owner, lease, cudaStreamPerThread, false, failure);
            waitFault = cudaSuccess;
            const bool retained = !released && owner.spectra.front().pendingWorkspaceMask == 1 &&
                                  owner.spectra.front().spectra.red && ledger->snapshot().chargedBytes > 0;
            check_cross_thread_drain(context, work, *ledger, [&] {
                return drain_diffusion_resources(owner, failure);
            });
            require(retained && owner.spectra.front().pendingWorkspaceMask == 0, "per-thread release lost workspace dependency");
        }
        for (const bool evict : {false, true}) {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            const auto use_spectrum = [&](std::uint64_t hash) {
                camera.sample.hash = hash;
                execution.spectrumKeys[0].sampleHash = hash;
                execution.spectrumKeys[0].hash = hash;
                require(prepare(owner, lease, streams[0]), "prepare retained spectrum");
                require(release_diffusion_resources(owner, lease, streams[0], false, failure), "release retained spectrum");
                require(cudaStreamSynchronize(streams[0]) == cudaSuccess, "complete retained spectrum");
            };
            use_spectrum(11);
            use_spectrum(12);
            require(reap_completed(owner, failure), "reap completed spectrum producers");
            auto& first = owner.spectra.front();
            failBuffer = first.spectra.red;
            contextLossBuffer = first.spectra.green;
            if (evict) {
                camera.sample.hash = 13;
                execution.spectrumKeys[0].sampleHash = execution.spectrumKeys[0].hash = 13;
                require(!prepare(owner, lease, streams[0]), "eviction hid context loss by selecting a vacant slot");
                require(owner.spectra[2].state == SpectrumEntryState::Vacant, "eviction populated a spare slot after context loss");
            } else {
                require(!trim_inactive_diffusion_resources(owner, key, failure), "spectrum trim ignored retirement failure");
            }
            require(failure.status.api == FJ_API_CUDA_RUNTIME && failure.status.native_code == cudaErrorInvalidValue &&
                        context_loss(failure) && !owner.hostLeaseActive &&
                        first.spectra.red == failBuffer && first.spectra.green == contextLossBuffer,
                    "spectrum disposal lost the primary status, context loss, or failed handles");
            failBuffer = contextLossBuffer = nullptr;
            require(drain_diffusion_resources(owner, failure), "drain spectrum after mixed disposal errors");
            require(ledger->snapshot().recordCount == 0, "spectrum retry left ledger charges");
        }
        for (const bool failPlan : {false, true}) {
            DiffusionContextResources owner;
            PreparedDiffusionLease lease;
            require(prepare(owner, lease, streams[0]), failure.diagnostic.c_str());
            const auto& workspace = owner.workspaces.front();
            const auto r2c = workspace.r2cPlan.get();
            failBuffer = failPlan ? nullptr : workspace.transform.get();
            failPlanDestroy = failPlan;
            const auto previousFailures = failPlan ? planFailures : bufferFailures;
            require(release_diffusion_resources(owner, lease, streams[0], true, failure), "record workspace completion");
            require(cudaStreamSynchronize(streams[0]) == cudaSuccess, "complete workspace");
            for (unsigned attempt = 1; attempt <= 2; ++attempt) {
                require(!prepare(owner, lease, streams[1]), "persistent workspace disposal failure admitted work");
                require(failure.status.api == (failPlan ? FJ_API_CUFFT : FJ_API_CUDA_RUNTIME) &&
                            failure.status.native_code == (failPlan ? static_cast<int>(CUFFT_INTERNAL_ERROR) : static_cast<int>(cudaErrorInvalidValue)) &&
                            !owner.hostLeaseActive && !lease.active(),
                        "workspace disposal did not surface the typed failure or release the claim");
                require((failPlan ? planFailures : bufferFailures) == previousFailures + attempt * (failPlan ? 2u : 1u),
                        "admission retried a failed disposal");
                require((failPlan ? workspace.r2cPlan.get() == r2c && workspace.planAllowanceReservation.active() : workspace.transform.get() == failBuffer) &&
                            ledger->snapshot().chargedBytes > 0,
                        "failed disposal dropped a handle or its ledger charge");
            }
            failBuffer = nullptr;
            failPlanDestroy = false;
            require(prepare(owner, lease, streams[1]), failure.diagnostic.c_str());
            require(release_diffusion_resources(owner, lease, streams[1], false, failure), "release retry lease");
            require(drain_diffusion_resources(owner, failure), "drain retry lease");
            require(ledger->snapshot().recordCount == 0, "workspace retry did not release charges");
        }
        for (auto* stream : streams) {
            require(cudaStreamDestroy(stream) == cudaSuccess, "destroy diffusion stream");
        }
    }

    void check_retirement_failures() {
        CUcontext context = nullptr;
        require(cuCtxCreate(&context, nullptr, 0, 0) == CUDA_SUCCESS, "create regression fixture context");
        check_event_and_host_retirement();
        check_frame_and_scratch_retirement(context);
        check_diffusion_retirement();
        require(cuCtxDestroy(context) == CUDA_SUCCESS, "destroy regression fixture context");
    }
} // namespace

namespace JuicerCuda::ResourceOperations {
    cudaError_t allocate_upload(void** destination, std::size_t bytes) {
        return cudaMallocHost(destination, bytes);
    }
    cudaError_t create_upload_event(cudaEvent_t* event) {
        return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    }
    cudaError_t copy_upload(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
        return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    }
    cudaError_t read_scan_error(int* host, const int* device, cudaStream_t stream) {
        return cudaMemcpyAsync(host, device, sizeof(*host), cudaMemcpyDeviceToHost, stream);
    }
    cudaError_t query_event(cudaEvent_t event) noexcept {
        return event == pendingQuery ? cudaErrorNotReady : cudaEventQuery(event);
    }

    cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
        const auto fault = std::exchange(recordFault, cudaSuccess);
        return fault == cudaSuccess ? cudaEventRecord(event, stream) : fault;
    }
    cudaError_t synchronize_stream(cudaStream_t stream) {
        return stream == failWaitStream && waitFault != cudaSuccess ? waitFault : cudaStreamSynchronize(stream);
    }

    cudaError_t synchronize_context() noexcept {
        if (pendingWork) {
            try {
                pendingWork->barrierEntered.set_value();
            } catch (...) {
                return cudaErrorUnknown;
            }
        }
        return contextWaitFault == cudaSuccess ? cudaDeviceSynchronize() : contextWaitFault;
    }

    cudaError_t free_device(void* pointer) {
        ++freeAttempts;
        if (pendingWork && !pendingWork->completed.load()) {
            pendingWork->earlyFree.store(true);
            return cudaErrorInvalidValue;
        }
        if (pointer == contextLossBuffer) {
            return cudaErrorContextIsDestroyed;
        }
        if (pointer == failBuffer) {
            ++bufferFailures;
            return cudaErrorInvalidValue;
        }
        return std::exchange(failFree, false) ? cudaErrorInvalidValue : cudaFree(pointer);
    }

    cudaError_t free_host(void* pointer) noexcept {
        if (failHost) {
            return cudaErrorInvalidValue;
        }
        const auto result = cudaFreeHost(pointer);
        hostFrees += result == cudaSuccess;
        return result;
    }

    cudaError_t destroy_event(cudaEvent_t event) noexcept {
        if (failEventDestroy) {
            return cudaErrorInvalidValue;
        }
        const auto result = cudaEventDestroy(event);
        eventDestroys += result == cudaSuccess;
        return result;
    }

    cufftResult destroy_plan(cufftHandle plan) noexcept {
        if (failPlanDestroy) {
            ++planFailures;
            return CUFFT_INTERNAL_ERROR;
        }
        return cufftDestroy(plan);
    }

    cufftResult execute_spectrum_r2c(cufftHandle plan, cufftReal* input, cufftComplex* output) noexcept {
        const auto fault = std::exchange(transformFault, CUFFT_SUCCESS);
        return fault == CUFFT_SUCCESS ? cufftExecR2C(plan, input, output) : fault;
    }
} // namespace JuicerCuda::ResourceOperations

int main() {
    try {
        check();
        check_retirement_failures();
        std::puts("PASS deferred resource ownership");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL deferred resource ownership: %s\n", error.what());
        return 1;
    }
}
