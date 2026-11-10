#include <array>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <system_error>
#include <cstring>
#include <utility>
#include <vector>

#include "ProcessRoot.h"
#include "owner_fixture.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "case_results.h"

namespace {
    std::atomic<std::size_t> allocations{0};
    thread_local bool rejectDiagnosticAllocation = false;
    std::atomic<unsigned> shutdownAttempts{0};
    unsigned destructions = 0;
    std::string injection;
    thread_local bool failOwnerLock = false;
    const std::system_error ownerLockFailure{std::make_error_code(std::errc::invalid_argument)};
    std::atomic<bool> holdShutdown{false};
    std::atomic<bool> ownerLockAttempted{false};

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    // This is a test-owned catalog hold; Runtime itself retains no FjCatalog.
    class CatalogHold final {
    public:
        explicit CatalogHold(const FjAssets* assets) : assets_(assets) {
            FjCatalogCounts counts{};
            const FjStatus acquired = fj_test_catalog_acquire(assets_, &owner_, &counts, nullptr);
            if (acquired.category != FJ_STATUS_SUCCESS || !owner_ || !counts.film_count || !counts.print_count) {
                if (owner_)
                    (void)fj_test_catalog_release(std::exchange(owner_, nullptr), nullptr);
                throw std::runtime_error("cold terminal catalog failed");
            }
            if (fj_test_catalog_entry(owner_, FJ_PROFILE_ROLE_FILM, 0, &entry_, nullptr).category != FJ_STATUS_SUCCESS ||
                !entry_.key.data || !entry_.label.data ||
                fj_test_catalog_backing_identity(owner_, &backing_, nullptr).category != FJ_STATUS_SUCCESS || !backing_) {
                (void)fj_test_catalog_release(std::exchange(owner_, nullptr), nullptr);
                throw std::runtime_error("terminal catalog entry unavailable");
            }
        }
        CatalogHold(const CatalogHold&) = delete;
        CatalogHold& operator=(const CatalogHold&) = delete;
        ~CatalogHold() {
            if (owner_)
                (void)fj_test_catalog_release(owner_, nullptr);
        }
        bool stable() const {
            FjCatalogEntryView current{};
            FjCatalogEntryView reacquired{};
            FjCatalog* again = nullptr;
            FjCatalogCounts counts{};
            std::uintptr_t backing = 0;
            if (fj_test_catalog_entry(owner_, FJ_PROFILE_ROLE_FILM, 0, &current, nullptr).category != FJ_STATUS_SUCCESS)
                return false;
            const FjStatus acquired = fj_test_catalog_acquire(assets_, &again, &counts, nullptr);
            if (acquired.category != FJ_STATUS_SUCCESS || !again) {
                if (again)
                    (void)fj_test_catalog_release(again, nullptr);
                return false;
            }
            const bool same = fj_test_catalog_entry(again, FJ_PROFILE_ROLE_FILM, 0, &reacquired, nullptr).category == FJ_STATUS_SUCCESS &&
                              fj_test_catalog_backing_identity(again, &backing, nullptr).category == FJ_STATUS_SUCCESS &&
                              backing == backing_ && current.key.data == entry_.key.data && current.key.count == entry_.key.count &&
                              current.label.data == entry_.label.data && current.label.count == entry_.label.count &&
                              reacquired.key.data == entry_.key.data && reacquired.key.count == entry_.key.count &&
                              reacquired.label.data == entry_.label.data && reacquired.label.count == entry_.label.count;
            return fj_test_catalog_release(again, nullptr).category == FJ_STATUS_SUCCESS && same;
        }
        void close() {
            FjCatalog* owner = std::exchange(owner_, nullptr);
            require(fj_test_catalog_release(owner, nullptr).category == FJ_STATUS_SUCCESS, "terminal catalog release failed");
        }

    private:
        const FjAssets* assets_;
        FjCatalog* owner_ = nullptr;
        FjCatalogEntryView entry_{};
        std::uintptr_t backing_ = 0;
    };

    FjCuda* native(JuicerTestSupport::RuntimeFixture& owner) {
        return owner.borrow().cuda;
    }

    bool live(JuicerTestSupport::RuntimeFixture& owner) {
        FjRuntimeView view{};
        return owner.borrow(view).category == FJ_STATUS_SUCCESS;
    }

    JuicerProcess::Root& root(JuicerTestSupport::RuntimeFixture& owner) {
        return JuicerCuda::borrow_root(native(owner));
    }

    bool same_status(FjStatus a, FjStatus b) {
        return a.category == b.category && a.api == b.api && a.native_code == b.native_code;
    }

    void inject() {
        if (injection == "allocation") {
            throw std::bad_alloc();
        }
        if (injection == "standard") {
            throw std::runtime_error("terminal standard exception");
        }
        if (injection == "unknown") {
            throw 7;
        }
        if (injection == "diagnostic-allocation") {
            JuicerCuda::Failure failure;
            rejectDiagnosticAllocation = true;
            JuicerCuda::set_failure(failure, {FJ_STATUS_CUFFT_FAILURE, FJ_API_CUFFT, 13}, "A long terminal diagnostic that requires string storage beyond the inline capacity; allocation failure must not change the cuFFT origin.");
            rejectDiagnosticAllocation = false;
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        if (injection == "runtime") {
            throw JuicerCuda::ExecutionFailure{{{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, 719}, "runtime failure"}};
        }
        if (injection == "context-loss") {
            throw JuicerCuda::ExecutionFailure{{{FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_DRIVER, 201}, "context failure"}};
        }
        if (injection == "typed" || injection == "wording") {
            throw JuicerCuda::ExecutionFailure{{{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701},
                                                injection == "typed" ? "terminal driver diagnostic" : "different words: allocation capacity context lost"}};
        }
    }

    void require_host_detached() {
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0 && fj_test_profile_live_owners() == 0 && fj_test_print_density_live_owners() == 0, "retained native graph owns Rust assets");
    }

    void reject_accepting_calls(FjCuda* cuda) {
        FjFrame frame{};
        FjCudaContext context{0, 1};
        FjSubmission submission{};
        submission.instance_token = 1;
        submission.submission_id = 1;
        FjPreparedHostData prepared{};
        require(fj_cuda_inspect(cuda, &frame, &context, nullptr).category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted inspection");
        require(fj_cuda_render(cuda, &context, &frame, &submission, &prepared, {}, nullptr).status.category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted render");
        require(fj_cuda_retire_instance(cuda, 1, nullptr).category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted retirement");
        require(!JuicerCuda::borrow_root(cuda).begin_frame_preparation().active(), "closed Root admitted preparation");
    }

    void check_success() {
        JuicerTestSupport::RuntimeFixture empty;
        require(empty.close().category == FJ_STATUS_SUCCESS, "empty close failed");
        for (unsigned cycle = 0; cycle != 3; ++cycle) {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create("metadata-only");
            FjCuda* borrow = native(owner);
            require(fj_cuda_retire_instance(borrow, 0, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "zero token accepted");
            require(fj_cuda_retire_instance(borrow, 123, nullptr).category == FJ_STATUS_SUCCESS, "valid unused token rejected");
            require(owner.shutdown().category == FJ_STATUS_SUCCESS, "metadata shutdown failed");
            require(native(owner) == borrow && destructions == cycle, "shutdown consumed caller ownership");
            reject_accepting_calls(borrow);
            const auto attempts = shutdownAttempts.load();
            require(owner.shutdown().category == FJ_STATUS_SUCCESS && shutdownAttempts == attempts, "closed shutdown retried");
            require(owner.close().category == FJ_STATUS_SUCCESS, "closed destroy failed");
            require(!live(owner) && destructions == cycle + 1, "closed destroy did not release exactly once");
            require(owner.close().category == FJ_STATUS_SUCCESS && destructions == cycle + 1, "empty wrapper consumed twice");
        }
        const auto attempts = shutdownAttempts.load();
        {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create("accepting-destroy");
        }
        require(shutdownAttempts == attempts + 1 && destructions == 4,
                "accepting destructor did not make exactly one shutdown attempt");
    }

    void check_host_cache_result(bool nativeFailure) {
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(JUICER_TEST_RESOURCE_DIR);
        CatalogHold catalog(owner.borrow().assets);
        require(fj_test_catalog_live_owners() == 1, "unexpected terminal catalog holds");
        fj_test_cache_release_failure();
        auto* borrowed = native(owner);
        if (nativeFailure) {
            injection = "typed";
            require(owner.shutdown().native_code == -701, "host failure erased native close failure");
            require(fj_test_runtime_cache_attempts() == 0 && fj_test_catalog_live_owners() == 1, "native failure started host purge");
            require(catalog.stable(), "native failure invalidated catalog backing");
            injection.clear();
            catalog.close();
            require(owner.close().native_code == -701, "terminal cleanup erased native failure");
            require_host_detached();
            require(shutdownAttempts == 1 && destructions == 0, "failed native close was retired again or deleted");
            return;
        }
        fj_test_runtime_cache_hold(1);
        std::array<FjStatus, 8> results{};
        std::array<std::array<char, 256>, 8> diagnostics{};
        std::array<std::thread, 8> workers;
        std::atomic<unsigned> completed{0};
        const auto call = [&](std::size_t i) {
            FjErrorBuffer error{diagnostics[i].data(), diagnostics[i].size(), 0};
            results[i] = owner.shutdown(&error);
            completed.fetch_add(1);
        };
        workers[0] = std::thread(call, 0);
        while (fj_test_runtime_cache_attempts() == 0) {
            std::this_thread::yield();
        }
        for (std::size_t i = 1; i < workers.size(); ++i) {
            workers[i] = std::thread(call, i);
        }
        require(completed == 0, "shutdown returned before cleanup result publication");
        fj_test_runtime_cache_hold(0);
        for (auto& worker : workers) {
            worker.join();
        }
        for (std::size_t i = 0; i < workers.size(); ++i) {
            require(results[i].category == FJ_STATUS_INTERNAL_FAILURE && same_status(results[i], results[0]) && std::strcmp(diagnostics[i].data(), diagnostics[0].data()) == 0, "concurrent shutdown saw another cleanup result");
        }
        require(fj_test_runtime_cache_attempts() == 1 && shutdownAttempts == 1 && fj_test_catalog_live_owners() == 1, "cache release repeated or changed the test catalog hold");
        require(catalog.stable(), "cache cleanup invalidated catalog backing");
        reject_accepting_calls(borrowed);
        require(same_status(owner.shutdown(), results[0]) && fj_test_runtime_cache_attempts() == 1 && shutdownAttempts == 1, "repeated shutdown retried cleanup/native retirement");
        fj_test_runtime_fault(3);
        catalog.close();
        std::array<char, 256> terminalText{};
        FjErrorBuffer terminalError{terminalText.data(), terminalText.size(), 0};
        require(same_status(owner.close(&terminalError), results[0]) && destructions == 1 && !live(owner) &&
                    std::strcmp(terminalText.data(), diagnostics[0].data()) == 0,
                "terminal host cleanup erased the saved cache result or retained the closed native graph");
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0 && fj_test_profile_live_owners() == 0, "terminal close retained host owners");
    }

    void check_serialization() {
        JuicerTestSupport::RuntimeFixture owner;
        owner.create("admission-order");
        auto outerGuard = JuicerCuda::borrow_root(native(owner)).begin_frame_preparation();
        FjCuda* borrow = native(owner);
        FjStatus result{};
        std::thread terminal([&] {
            result = owner.shutdown();
        });
        shutdownAttempts.wait(0);
        bool rejected = false;
        try {
            JuicerCuda::NativeCall call(borrow);
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            rejected = failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
        }
        outerGuard = {};
        terminal.join();
        require(rejected && result.category == FJ_STATUS_SUCCESS, "shutdown deadlocked or admitted a waiting callback");
        require(owner.close().category == FJ_STATUS_SUCCESS, "serialized shutdown destroy failed");
    }

    void check_unregistered(const std::string& mode) {
        JuicerTestSupport::RuntimeFixture owner;
        owner.create("unregistered-terminal-handle");
        auto* const invalid = reinterpret_cast<FjCuda*>(std::uintptr_t{8});
        if (mode == "unregistered") {
            require(fj_cuda_shutdown(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "unregistered shutdown accepted");
            require(fj_cuda_destroy(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "unregistered destroy accepted");
        } else if (mode == "unregistered-reentry") {
            JuicerCuda::NativeCall call(native(owner));
            require(fj_cuda_shutdown(invalid, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "reentrant shutdown accepted");
            require(fj_cuda_destroy(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "reentrant unregistered destroy accepted");
        } else {
            failOwnerLock = true;
            const auto result = mode == "unregistered-shutdown-lock" ? fj_cuda_shutdown(invalid, nullptr) : fj_cuda_destroy(invalid, nullptr);
            if (mode == "unregistered-destroy-lock") {
                require(failOwnerLock && result.category == FJ_STATUS_PREPARATION_FAILURE, "unregistered destroy reached throwable admission");
                failOwnerLock = false;
            } else {
                require(!failOwnerLock && result.category == FJ_STATUS_INTERNAL_FAILURE, "unregistered admission exception not contained");
            }
        }
        require(fj_cuda_retire_instance(native(owner), 1, nullptr).category == FJ_STATUS_SUCCESS, "invalid handle changed registered admission");
        require(owner.close().category == FJ_STATUS_SUCCESS && destructions == 1, "valid owner was lost");
    }

    void check_admission_failure(const std::string& mode) {
        const bool saved = mode.starts_with("saved-");
        const bool consuming = mode.ends_with("destroy-lock");
        const FjStatus expected = saved ? FjStatus{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701}
                                        : FjStatus{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create("terminal-admission-exception");
            FjCuda* borrow = native(owner);
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            if (saved) {
                injection = "typed";
                require(same_status(owner.shutdown(&error), expected), "initial typed shutdown failed");
                injection.clear();
            }
            failOwnerLock = true;
            const auto before = allocations.load();
            const auto result = consuming ? owner.close(&error) : owner.shutdown(&error);
            const auto after = allocations.load();
            require(!failOwnerLock && same_status(result, expected), "admission exception replaced selected status");
            require(before == after, "admission failure retention allocated");
            if (saved) {
                require(std::strcmp(diagnostic.data(), "terminal driver diagnostic") == 0, "admission exception replaced selected diagnostic");
            }
            reject_accepting_calls(borrow);
            if (!consuming) {
                require(same_status(owner.close(&error), expected), "consumption replaced admission failure");
                if (saved) {
                    require(std::strcmp(diagnostic.data(), "terminal driver diagnostic") == 0, "consumption replaced selected diagnostic");
                }
            }
            require(owner.close().category == FJ_STATUS_SUCCESS, "wrapper consumed twice");
        }
        require_host_detached();
        require(destructions == 0 && shutdownAttempts == (saved ? 1u : 0u), "destructor retried retained admission failure");
    }

    void check_buffers(const std::string& mode) {
        const bool blocked = mode == "malformed-blocked-destroy";
        {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create("terminal-error-buffer");
            FjCuda* borrow = native(owner);
            FjErrorBuffer malformed{nullptr, 1, 99};
            if (mode == "buffers") {
                require(owner.shutdown(&malformed).category == FJ_STATUS_UNSUPPORTED_INPUT, "malformed borrowed shutdown accepted");
                require(fj_cuda_retire_instance(borrow, 1, nullptr).category == FJ_STATUS_SUCCESS, "malformed shutdown closed admission");
                require(shutdownAttempts == 0, "malformed shutdown attempted drain");
                FjErrorBuffer zero{nullptr, 0, 99};
                require(owner.shutdown(&zero).category == FJ_STATUS_SUCCESS && zero.length == 0, "zero-capacity shutdown rejected");
                require(owner.close().category == FJ_STATUS_SUCCESS, "valid destroy failed");
            } else {
                if (blocked) {
                    injection = "typed";
                    require(owner.shutdown(nullptr).native_code == -701, "typed setup failed");
                    injection.clear();
                }
                const auto result = owner.close(&malformed);
                const FjStatus expected = blocked ? FjStatus{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701}
                                                  : FjStatus{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
                require(same_status(result, expected), "malformed consuming output lost status precedence");
                require(owner.close().category == FJ_STATUS_SUCCESS, "malformed output caused second consume");
                if (blocked) {
                    reject_accepting_calls(borrow);
                }
            }
        }
        if (blocked) {
            require_host_detached();
        }
        require(fj_test_assets_live_owners() == 0, "malformed diagnostics retained Rust Assets");
        require(shutdownAttempts == 1 && destructions == (blocked ? 0u : 1u), "malformed output abandoned or retried owner");
    }

    void check_shutdown_progress() {
        JuicerTestSupport::RuntimeFixture owner;
        owner.create("shutdown-progress");
        injection = "typed";
        holdShutdown = true;
        FjStatus first{}, second{};
        std::array<char, 128> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::thread terminal([&] {
            first = owner.shutdown(nullptr);
        });
        shutdownAttempts.wait(0);
        std::thread rejected([&] {
            failOwnerLock = true;
            second = owner.shutdown(&error);
        });
        ownerLockAttempted.wait(false);
        holdShutdown = false;
        holdShutdown.notify_all();
        terminal.join();
        rejected.join();
        const FjStatus expected{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701};
        require(same_status(first, expected) && same_status(second, expected), "in-progress result was read or overwritten");
        require(std::strcmp(diagnostic.data(), "terminal driver diagnostic") == 0, "in-progress diagnostic was read or overwritten");
        require(same_status(owner.close(), expected) && destructions == 0 && shutdownAttempts == 1, "in-progress failure was retried");
    }

    void check_failure(const std::string& mode, bool destroyAccepting) {
        FjStatus expected{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        if (mode == "allocation") {
            expected.category = FJ_STATUS_ALLOCATION_FAILURE;
        } else if (mode == "diagnostic-allocation") {
            expected = {FJ_STATUS_CUFFT_FAILURE, FJ_API_CUFFT, 13};
        } else if (mode == "runtime") {
            expected = {FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, 719};
        } else if (mode == "context-loss") {
            expected = {FJ_STATUS_CONTEXT_LOSS, FJ_API_CUDA_DRIVER, 201};
        } else if (mode == "typed" || mode == "wording") {
            expected = {FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701};
        }
        {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create("retained-terminal-owner");
            FjCuda* borrow = native(owner);
            injection = mode;
            // The same exception seam also proves the non-consuming instance C
            // entry contains exceptions while preserving accepting admission.
            require(same_status(fj_cuda_retire_instance(borrow, 1, nullptr), expected), "retire exception lost typed origin");
            require(root(owner).begin_frame_preparation().active(), "retire exception closed owner admission");
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto result = destroyAccepting ? owner.close(&error) : owner.shutdown(&error);
            require(same_status(result, expected) && error.length != 0, "terminal failure lost category/API/code/diagnostic");
            require(shutdownAttempts == 1 && destructions == 0 &&
                        (destroyAccepting ? !live(owner) : native(owner) == borrow),
                    "failed terminal call retried, freed or incorrectly retained the runtime owner");
            require(fj_test_assets_live_owners() == (destroyAccepting ? 0u : 1u), "borrowed versus terminal host lifetime mismatch");
            if (!destroyAccepting) {
                reject_accepting_calls(borrow);
                const std::string original(diagnostic.data());
                injection.clear();
                const std::array<std::size_t, 3> capacities{0u, 1u, 8u};
                std::vector<std::string> capacity_ids;
                capacity_ids.reserve(capacities.size());
                for (const auto capacity : capacities)
                    capacity_ids.push_back(mode + "/" + std::to_string(capacity));
                JuicerTests::Cohort capacity_cases("cuda_terminal.capacity", capacity_ids);
                for (const std::size_t capacity : capacities) {
                    auto capacity_event = capacity_cases.case_event(mode + "/" + std::to_string(capacity));
                    error = {diagnostic.data(), capacity, 99};
                    require(same_status(owner.shutdown(&error), expected), "diagnostic capacity changed terminal status");
                    require(error.length == (capacity ? capacity - 1 : 0), "terminal diagnostic length wrong");
                    if (capacity) {
                        require(diagnostic[capacity - 1] == '\0', "terminal diagnostic missing terminator");
                        require(original.compare(0, error.length, diagnostic.data(), error.length) == 0, "terminal diagnostic changed after failure");
                    }
                    capacity_event.passed();
                }
                error = {diagnostic.data(), diagnostic.size(), 0};
                const auto before = allocations.load();
                const auto consumed = owner.close(&error);
                const auto after = allocations.load();
                require(before == after, "blocked destroy allocated during retention");
                require(same_status(consumed, expected), "blocked destroy lost original terminal failure");
            }
            require(owner.close().category == FJ_STATUS_SUCCESS, "consumed wrapper close failed");
        }
        require(shutdownAttempts == 1 && destructions == 0, "destructor retried or walked retained graph");
        FjCuda* replacement = nullptr;
        require(fj_cuda_create(&replacement, nullptr).category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "retention permitted a replacement graph");
    }
} // namespace

// Executable-local allocation counting proves retained destroy makes no C++
// allocation. One scoped diagnostic assignment also exercises allocation failure;
// this fixture does not exhaust memory or add an allocator to product objects.
void* operator new(std::size_t size) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (rejectDiagnosticAllocation) {
        throw std::bad_alloc();
    }
    if (void* value = std::malloc(size ? size : 1)) {
        return value;
    }
    throw std::bad_alloc();
}
// Keep the replacement deallocator's backing free opaque to GCC's inlined
// new/delete mismatch analysis. Allocation counting still uses the real calls.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline))
#endif
void operator delete(void* block) noexcept {
    std::free(block);
}
void* operator new[](std::size_t size) {
    return ::operator new(size);
}
void operator delete[](void* block) noexcept {
    ::operator delete(block);
}
#if defined(__cpp_sized_deallocation)
void operator delete(void* block, std::size_t) noexcept {
    ::operator delete(block);
}
void operator delete[](void* block, std::size_t) noexcept {
    ::operator delete(block);
}
#endif

namespace JuicerCuda::TerminalTest {
    void after_construct() {
        if (injection == "construction") {
            throw std::bad_alloc();
        }
    }
    void before_owner_lock() {
        if (std::exchange(failOwnerLock, false)) {
            ownerLockAttempted = true;
            ownerLockAttempted.notify_all();
            throw std::system_error{ownerLockFailure};
        }
    }
    void before_shutdown() {
        require(fj_test_assets_live_owners() == 1, "Runtime released Assets before the native terminal attempt");
        shutdownAttempts.fetch_add(1);
        shutdownAttempts.notify_all();
        holdShutdown.wait(true);
        inject();
    }
    void before_retire_instance() {
        inject();
    }
    void owner_destroyed() noexcept {
        ++destructions;
    }
} // namespace JuicerCuda::TerminalTest

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        const bool listing = argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases";
        const std::string mode = argc > 1 && (!listing || argc > 2) ? argv[1] : "success";
        if (listing) {
            JuicerTests::list_case("cuda_terminal", mode);
            return 0;
        }
        JuicerTests::CaseEvent event("cuda_terminal", mode);
        if (mode == "partial-library") {
            JuicerTestSupport::RuntimeFixture owner;
            fj_test_runtime_fault(1);
            bool failed = false;
            try {
                owner.create("runtime-assets-before-native");
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            require(failed && !live(owner) && fj_test_assets_live_owners() == 0 && shutdownAttempts == 0 && destructions == 0,
                    "Runtime stage 1 did not release partial Assets before native construction");

            fj_test_runtime_fault(2);
            failed = false;
            try {
                owner.create("runtime-native-before-publication");
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            require(failed && !live(owner) && fj_test_assets_live_owners() == 0 && shutdownAttempts == 1 && destructions == 1,
                    "Runtime stage 2 did not consume the guarded native owner before releasing Assets");

            for (const unsigned stage : {4u, 5u}) {
                fj_test_runtime_fault(stage);
                bool contained = false;
                try {
                    owner.create("runtime-construction-unwind");
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    contained = failure.failure.status.category == FJ_STATUS_INTERNAL_FAILURE;
                }
                const unsigned attempts = stage == 4 ? 1u : 2u;
                require(contained && !live(owner) && fj_test_assets_live_owners() == 0 &&
                            shutdownAttempts == attempts && destructions == attempts,
                        "Runtime construction panic escaped or leaked/retried an owner");
            }

            owner.create("runtime-terminal-host-cleanup");
            fj_test_runtime_fault(3);
            const FjStatus terminal = owner.close();
            require(terminal.category == FJ_STATUS_INTERNAL_FAILURE && !live(owner) &&
                        fj_test_assets_live_owners() == 0 && shutdownAttempts == 3 && destructions == 3,
                    "Runtime stage 3 did not report after terminal host cleanup");
        } else if (mode == "construction") {
            injection = "construction";
            JuicerTestSupport::RuntimeFixture owner;
            bool failed = false;
            try {
                owner.create("partial-host-construction");
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            require(failed && !live(owner) && fj_test_assets_live_owners() == 0 && destructions == 1 && shutdownAttempts == 0, "partial construction leaked/published host owner");
            injection.clear();
            owner.create("retry-after-partial-construction");
            require(owner.close().category == FJ_STATUS_SUCCESS && fj_test_assets_live_owners() == 0, "construction retry failed");
        } else if (mode == "catalog-retention") {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create(JUICER_TEST_RESOURCE_DIR);
            CatalogHold catalog(owner.borrow().assets);
            require(fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 1, "cold terminal catalog failed");
            injection = "typed";
            require(owner.shutdown().native_code == -701, "borrowed close lost injected failure");
            require(fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 1 && catalog.stable(), "failed borrowed close expired catalog");
            injection.clear();
            catalog.close();
            const auto before = allocations.load();
            require(owner.close().native_code == -701, "catalog cleanup lost primary native failure");
            require(allocations.load() == before, "terminal Runtime cleanup allocated");
            require_host_detached();
            require(shutdownAttempts == 1 && destructions == 0, "catalog cleanup retried uncertain CUDA teardown");
        } else if (mode == "registered-reentry") {
            JuicerTestSupport::RuntimeFixture owner;
            owner.create(JUICER_TEST_RESOURCE_DIR);
            CatalogHold catalog(owner.borrow().assets);
            FjCuda* borrowed = native(owner);
            {
                JuicerCuda::NativeCall call(borrowed);
                require(fj_cuda_destroy(borrowed, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "registered native reentry accepted");
                require(fj_test_assets_live_owners() == 1 && fj_test_catalog_live_owners() == 1, "native reentry detached active host borrow");
                require(catalog.stable(), "reentry invalidated catalog borrow");
            }
            // Reentry is explicitly an illegal terminal overlap, not evidence
            // of cleanup. After the active borrow ends, a legal terminal call
            // may consume its retained host owner without retrying native work.
            catalog.close();
            fj_test_runtime_fault(3);
            require(owner.close().category == FJ_STATUS_UNSUPPORTED_INPUT, "saved reentry failure lost");
            require_host_detached();
            require(shutdownAttempts == 0 && destructions == 0, "reentry recovery retried or deleted native metadata");
            require(fj_cuda_check_terminal(borrowed, nullptr).category == FJ_STATUS_SUCCESS,
                    "consuming reentry cleanup lost retained native registration");
            reject_accepting_calls(borrowed);
        } else if (mode == "host-cache-failure") {
            check_host_cache_result(false);
        } else if (mode == "host-cache-native-precedence") {
            check_host_cache_result(true);
        } else if (mode == "success") {
            check_success();
        } else if (mode == "serialization") {
            check_serialization();
        } else if (mode.starts_with("unregistered")) {
            check_unregistered(mode);
        } else if (mode.ends_with("-lock")) {
            check_admission_failure(mode);
        } else if (mode == "buffers" || mode.starts_with("malformed-")) {
            check_buffers(mode);
        } else if (mode == "shutdown-progress") {
            check_shutdown_progress();
        } else if (mode == "accepting-failure") {
            check_failure("typed", true);
        } else {
            check_failure(mode, false);
        }
        std::printf("PASS terminal ownership: %s\n", mode.c_str());
        event.passed();
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL terminal ownership: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("FAIL terminal ownership: escaped exception\n", stderr);
        return 1;
    }
}
