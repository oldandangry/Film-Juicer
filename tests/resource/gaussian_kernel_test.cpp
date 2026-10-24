#include <array>
#include <bit>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "gaussian_kernel_test_hooks.h"
#include "../ffi/juicer_test_api.h"

namespace {
    using namespace JuicerCuda;
    using GaussianKernelTest::Binding;
    using GaussianKernelTest::Controls;
    using Json = nlohmann::json;
    void require(bool value, const char* text) {
        if (!value)
            throw std::runtime_error(text);
    }
    constexpr int kRadius = 7;
    const float kSigma = 1.234567f;
    std::uint64_t key(int radius, float sigma) {
        return (std::uint64_t(std::uint32_t(radius)) << 32u) | std::bit_cast<std::uint32_t>(sigma);
    }
    struct Fixture {
        CUcontext context = nullptr;
        std::shared_ptr<DeviceAllocationLedger> ledger;
        Resources* resources = nullptr;
        Fixture() {
            require(cuInit(0) == CUDA_SUCCESS, "CUDA init");
            CUdevice device{};
            require(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "CUDA device");
            require(cuCtxCreate(&context, nullptr, 0, device) == CUDA_SUCCESS, "CUDA context");
            std::string diagnostic;
            ledger = DeviceAllocationLedger::create({0, 65536}, diagnostic);
            require(bool(ledger) && ledger->bind_or_validate_cap(65536, diagnostic), "Gaussian ledger cap bound");
            Failure failure;
            resources = create({0, context}, 1, ledger, failure);
            require(resources != nullptr, "Gaussian exact-context owner");
        }
        ~Fixture() {
            if (context) {
                (void)cuCtxSetCurrent(context);
                destroy(resources);
                (void)cuCtxDestroy(context);
            }
        }
        void close() {
            Failure failure;
            require(drain_for_context_retire(*resources, failure), "Gaussian normal drained retirement");
            require(resources->gaussianKernelCache.empty() && resources->deviceAllocationRecords.empty(), "drained cache and allocation records");
            const auto ledgerState = ledger->snapshot();
            require(ledgerState.chargedBytes == 0 && ledgerState.reservedBytes == 0 && ledgerState.recordCount == 0, "retirement physically frees all committed kernels");
        }
    };
    struct State {
        std::size_t entries;
        std::size_t allocations;
        DeviceLedgerSnapshot ledger;
    };
    State snapshot(Resources& resources) {
        State state{};
        {
            std::lock_guard lock(resources.m);
            state.entries = resources.gaussianKernelCache.size();
        }
        {
            std::lock_guard lock(resources.deviceAllocationRecordsMutex);
            state.allocations = resources.deviceAllocationRecords.size();
        }
        state.ledger = resources.deviceLedger->snapshot();
        return state;
    }
    void equal(State a, State b) {
        require(a.entries == b.entries && a.allocations == b.allocations &&
                    a.ledger.chargedBytes == b.ledger.chargedBytes && a.ledger.reservedBytes == b.ledger.reservedBytes &&
                    a.ledger.committedBytes == b.ledger.committedBytes && a.ledger.retiringBytes == b.ledger.retiringBytes &&
                    a.ledger.recordCount == b.ledger.recordCount,
                "Gaussian failed transition preserves cache/record/ledger balance");
    }
    void download(const Resources::DeviceGaussianKernel& kernel, const Json& fixture) {
        const Json* expected = nullptr;
        for (const auto& row : fixture.at("cases"))
            if (row.at("radius") == kernel.radius && row.at("sigma_bits") == std::bit_cast<std::uint32_t>(kernel.sigma)) {
                expected = &row;
                break;
            }
        require(expected != nullptr, "download has independently frozen expectation");
        std::vector<float> weights(static_cast<std::size_t>(kernel.radius) * 2u + 1u);
        require(cudaMemcpy(weights.data(), kernel.weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "download real cached coefficients");
        require(expected->at("weights").size() == weights.size(), "exact downloaded kernel extent");
        for (std::size_t i = 0; i < weights.size(); ++i)
            require(std::bit_cast<std::uint32_t>(weights[i]) == expected->at("weights")[i].get<std::uint32_t>(), "every downloaded coefficient matches independent native word");
    }
    void seeded_failure(const Json& expected, bool copy) {
        Fixture fixture;
        auto& r = *fixture.resources;
        Failure failure;
        require(ensure_gaussian_kernel(r, r.scannerLensBlurKernel, 1.0f, 3, failure), "seed old Gaussian slot/cache");
        const auto old = r.scannerLensBlurKernel;
        const auto before = snapshot(r);
        const auto owners = fj_test_gaussian_kernel_live_owners();
        Controls controls(r, key(kRadius, kSigma));
        controls.participants[0].failCopy = copy;
        controls.participants[0].failInsertion = !copy;
        {
            Binding binding(controls, 0);
            require(!ensure_gaussian_kernel(r, r.scannerLensBlurKernel, kSigma, kRadius, failure), "selected native Gaussian failure");
        }
        const auto& p = controls.participants[0];
        require(p.copies == 1 && p.candidate && p.frees == 1 && p.freed == p.candidate && p.freeSucceeded, "exactly one successful real candidate free");
        require(r.scannerLensBlurKernel.weights == old.weights && r.scannerLensBlurKernel.radius == old.radius && r.scannerLensBlurKernel.sigma == old.sigma, "old slot preserved");
        equal(snapshot(r), before);
        {
            std::lock_guard lock(r.m);
            require(r.gaussianKernelCache.at(key(old.radius, old.sigma)) == old.weights && !r.gaussianKernelCache.contains(key(kRadius, kSigma)), "old cache entry remains and failed key is absent");
        }
        {
            std::lock_guard lock(r.deviceAllocationRecordsMutex);
            require(r.deviceAllocationRecords.contains(old.weights) && !r.deviceAllocationRecords.contains(p.candidate), "physical cleanup removes only candidate allocation record");
        }
        download(old, expected);
        require(fj_test_gaussian_kernel_live_owners() == owners, "failure releases Rust coefficient owner");
        if (copy) {
            require(p.copyConsumed && failure.status.category == 5 && failure.status.api == 1 && failure.status.native_code == int(cudaErrorInvalidValue), "actual copy error status");
            const auto text = std::string("cudaMemcpy(gaussian kernel) failed: ") + cudaGetErrorString(cudaErrorInvalidValue);
            require(failure.diagnostic == text, "actual copy error exact diagnostic");
        } else {
            require(p.insertionConsumed && p.rejectedBytes != 0 && failure.status.category == 3 && failure.status.api == 0 && failure.status.native_code == 0 && failure.diagnostic == "Gaussian cache bookkeeping failed", "real cache-node allocation rejected with unchanged bookkeeping envelope");
        }
        require(ensure_gaussian_kernel(r, r.scannerLensBlurKernel, kSigma, kRadius, failure), "successful recovery after clearing one-shot fault");
        download(r.scannerLensBlurKernel, expected);
        fixture.close();
        std::printf("PASS Gaussian %s failure: actual candidate cleanup, old state, ledger/Rust balance and recovery\n", copy ? "copy" : "node-allocation");
    }
    void preparation_keys_and_contexts(const Json& expected) {
        Fixture fixture;
        auto& r = *fixture.resources;
        Failure failure;
        require(ensure_gaussian_kernel(r, r.scannerLensBlurKernel, 1.0f, 3, failure), "key witness seed");
        const auto old = r.scannerLensBlurKernel;
        const auto initial = snapshot(r);
        for (auto fault : {std::array<unsigned, 2>{0, 1}, {0, 2}, {1, 1}, {1, 2}, {1, 3}, {1, 4}}) {
            require(fj_test_gaussian_kernel_fault(fault[0], fault[1]).category == 0, "arm owning Rust failure");
            require(!ensure_gaussian_kernel(r, r.scannerLensBlurKernel, kSigma, kRadius, failure), "actual cache miss propagates Rust preparation/projection failure");
            require(r.scannerLensBlurKernel.weights == old.weights, "failed Rust preparation keeps old slot");
            equal(snapshot(r), initial);
            require(fj_test_gaussian_kernel_live_owners() == 0, "failed Rust cache miss releases owner");
        }
        download(old, expected);
        const auto beforeGuard = fj_test_gaussian_kernel_acquires();
        require(!ensure_gaussian_kernel(r, r.scannerUnsharpKernel, 19.0f, 76, failure), "standard family radius guard preserved");
        require(failure.diagnostic == "gaussian descriptor is invalid" && fj_test_gaussian_kernel_acquires() == beforeGuard, "guard precedes Rust acquisition");
        require(ensure_spatial_dir_kernel(r, r.spatialDirKernels[1], 2048, 512.0f, failure), "DIR upper bound actual cache entry");
        download(r.spatialDirKernels[1], expected);
        const auto beforeInvalidDir = fj_test_gaussian_kernel_acquires();
        require(!ensure_spatial_dir_kernel(r, r.spatialDirKernels[2], 2049, 512.0f, failure) && fj_test_gaussian_kernel_acquires() == beforeInvalidDir, "DIR upper guard before Rust");
        require(failure.diagnostic == "spatial DIR kernel radius exceeds prepared-frame limit", "DIR guard exact diagnostic");
        require(ensure_gaussian_kernel(r, r.scannerUnsharpKernel, std::bit_cast<float>(std::uint32_t{0x3f800001}), 3, failure), "adjacent sigma bits create distinct exact key");
        require(r.scannerUnsharpKernel.weights != old.weights, "sigma bit change does not reuse old key");
        download(r.scannerUnsharpKernel, expected);
        const auto beforeEpoch = snapshot(r);
        auto* epoch = create(r.ownerContextKey, 2, fixture.ledger, failure);
        require(epoch != nullptr, "second epoch exact owner");
        const auto epochAcquires = fj_test_gaussian_kernel_acquires();
        const bool epochReady = ensure_gaussian_kernel(*epoch, epoch->scannerLensBlurKernel, 1, 3, failure);
        const bool independent = epochReady && epoch->scannerLensBlurKernel.weights != old.weights && epoch->gaussianKernelCache.size() == 1 && fj_test_gaussian_kernel_acquires() == epochAcquires + 1;
        destroy(epoch);
        require(independent, "new epoch constructs independent device entry");
        equal(snapshot(r), beforeEpoch);
        {
            Fixture other;
            const auto beforeContext = fj_test_gaussian_kernel_acquires();
            require(other.resources->ownerContextKey != r.ownerContextKey, "different real exact contexts");
            require(ensure_gaussian_kernel(*other.resources, other.resources->scannerLensBlurKernel, 1, 3, failure), "new exact context acquires independent coefficients");
            require(fj_test_gaussian_kernel_acquires() == beforeContext + 1 && other.resources->gaussianKernelCache.size() == 1, "no cross-context host/device reuse");
            download(other.resources->scannerLensBlurKernel, expected);
            other.close();
        }
        require(cuCtxSetCurrent(fixture.context) == CUDA_SUCCESS, "restore first exact context");
        equal(snapshot(r), beforeEpoch);
        fixture.close();
        std::puts("PASS Gaussian preparation failures, family guards, upper-bound download, sigma bits, exact context and epoch separation");
    }

    struct Workers {
        Controls& controls;
        std::array<std::thread, 2> threads;
        std::array<Failure, 2> failures;
        std::array<std::exception_ptr, 2> exceptions;
        explicit Workers(Controls& value) : controls(value) {}
        ~Workers() {
            controls.cancel();
            join();
        }
        void join() {
            for (auto& thread : threads)
                if (thread.joinable())
                    thread.join();
        }
        void start(Fixture& fixture) {
            for (unsigned index = 0; index < 2; ++index)
                threads[index] = std::thread([&, index] {
                    bool ok = false;
                    try {
                        require(cuCtxSetCurrent(fixture.context) == CUDA_SUCCESS, "worker current exact context");
                        Binding binding(controls, index);
                        auto& r = *fixture.resources;
                        ok = index == 0 ? ensure_gaussian_kernel(r, r.scannerLensBlurKernel, kSigma, kRadius, failures[index])
                                        : ensure_spatial_dir_kernel(r, r.spatialDirKernels[0], kRadius, kSigma, failures[index]);
                    } catch (...) {
                        exceptions[index] = std::current_exception();
                    }
                    std::lock_guard lock(controls.mutex);
                    controls.participants[index].success = ok;
                    controls.participants[index].done = true;
                    controls.changed.notify_all();
                });
        }
    };
    void collision(const Json& expected, bool cancel) {
        Fixture fixture;
        auto& r = *fixture.resources;
        Failure failure;
        require(ensure_gaussian_kernel(r, r.scannerLensBlurKernel, 1.0f, 3, failure), "seed collision old state");
        const auto old = r.scannerLensBlurKernel;
        const auto before = snapshot(r);
        const auto owners = fj_test_gaussian_kernel_live_owners();
        const auto acquisitions = fj_test_gaussian_kernel_acquires();
        Controls controls(r, key(kRadius, kSigma));
        controls.rendezvous = true;
        Workers workers(controls);
        workers.start(fixture);
        require(controls.await_arrivals(2), "both real uploaded candidates reach bounded rendezvous");
        require(controls.participants[0].candidate && controls.participants[1].candidate && controls.participants[0].candidate != controls.participants[1].candidate, "two distinct actual uploaded allocations");
        require(fj_test_gaussian_kernel_live_owners() == owners + 2, "both real uploads retain their local Rust coefficient owners until insertion");
        if (cancel)
            controls.cancel();
        else {
            controls.release(0);
            require(controls.await_done(0), "A returns after real cache adoption before B release");
            require(controls.participants[0].success, "A successful adoption");
            controls.release(1);
            require(controls.await_done(1), "B returns after real insertion loser cleanup");
        }
        workers.join();
        for (const auto& exception : workers.exceptions)
            if (exception)
                std::rethrow_exception(exception);
        require(!controls.timedOut, "unexpected rendezvous timeout always fails");
        require(fj_test_gaussian_kernel_live_owners() == owners && fj_test_gaussian_kernel_acquires() == acquisitions + 2, "two miss-local Rust owners released");
        if (cancel) {
            for (unsigned i = 0; i < 2; ++i) {
                const auto& p = controls.participants[i];
                const auto& f = workers.failures[i];
                require(!p.success && p.frees == 1 && p.freed == p.candidate && p.freeSucceeded, "canceled candidate physically freed once");
                require(f.status.category == 3 && f.status.api == 0 && f.status.native_code == 0 && f.diagnostic == "Gaussian kernel fixture rendezvous canceled", "fixture cancellation exact envelope");
            }
            equal(snapshot(r), before);
            require(r.scannerLensBlurKernel.weights == old.weights && !r.spatialDirKernels[0].weights, "cancellation preserves old committed views");
        } else {
            const auto& a = controls.participants[0];
            const auto& b = controls.participants[1];
            require(a.success && b.success && a.frees == 0 && b.frees == 1 && b.freed == b.candidate && b.freeSucceeded, "real winner retention and once-only loser free");
            require(r.scannerLensBlurKernel.weights == a.candidate && r.spatialDirKernels[0].weights == a.candidate, "cross-family slots converge on A");
            const auto after = snapshot(r);
            const auto bytes = (2u * kRadius + 1u) * sizeof(float);
            require(after.entries == before.entries + 1 && after.allocations == before.allocations + 1 && after.ledger.recordCount == before.ledger.recordCount + 1 && after.ledger.chargedBytes == before.ledger.chargedBytes + bytes && after.ledger.reservedBytes == 0, "one new adopted device/map/ledger entry with exact byte charge");
            download(r.scannerLensBlurKernel, expected);
            const auto warmAcquires = fj_test_gaussian_kernel_acquires();
            {
                Binding binding(controls, 0);
                require(ensure_gaussian_kernel(r, r.scannerUnsharpKernel, kSigma, kRadius, failure), "real warm hit");
            }
            require(fj_test_gaussian_kernel_acquires() == warmAcquires && a.copies == 1 && r.scannerUnsharpKernel.weights == a.candidate, "warm hit has zero Rust acquire, host coefficient allocation, device allocation and copy");
            equal(snapshot(r), after);
        }
        fixture.close();
        std::printf("PASS Gaussian %s: bounded workers joined, exact accounting, frozen download and drained retirement\n", cancel ? "cancellation" : "insertion-loser");
    }
} // namespace
int main(int argc, char** argv) try {
    require(argc == 2, "Gaussian frozen fixture payload argument");
    std::ifstream stream(argv[1]);
    require(bool(stream), "Gaussian fixture open");
    const auto expected = Json::parse(stream);
    preparation_keys_and_contexts(expected);
    seeded_failure(expected, true);
    seeded_failure(expected, false);
    collision(expected, false);
    collision(expected, true);
    require(fj_test_gaussian_kernel_live_owners() == 0, "no escaped Gaussian owners");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL Gaussian: %s\n", error.what());
    return 1;
}
