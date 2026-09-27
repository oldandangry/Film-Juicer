#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <system_error>
#include <cstring>
#include <utility>

#include "ProcessRoot.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "juicer_cuda_owner.h"

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

    void reject_accepting_calls(FjCuda* cuda) {
        FjFrame frame{};
        FjCudaContext context{0, 1};
        FjSubmission submission{};
        submission.instance_token = 1;
        submission.submission_id = 1;
        FjPreparedHostData prepared{};
        require(fj_cuda_inspect(cuda, &frame, &context, nullptr).category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted inspection");
        require(fj_cuda_render(cuda, &context, &frame, &submission, &prepared, {}, nullptr).category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted render");
        require(fj_cuda_retire_instance(cuda, 1, nullptr).category == FJ_STATUS_PREPARATION_FAILURE,
                "closed owner admitted retirement");
        require(!JuicerProcess::root().begin_frame_preparation().active(), "closed Root admitted preparation");
    }

    void check_success() {
        JuicerCuda::Owner empty;
        require(empty.close().category == FJ_STATUS_SUCCESS, "empty close failed");
        for (unsigned cycle = 0; cycle != 3; ++cycle) {
            JuicerCuda::Owner owner;
            owner.create("metadata-only");
            FjCuda* borrow = JuicerCuda::borrowed_owner();
            require(fj_cuda_retire_instance(borrow, 0, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "zero token accepted");
            require(fj_cuda_retire_instance(borrow, 123, nullptr).category == FJ_STATUS_SUCCESS, "valid unused token rejected");
            require(fj_cuda_shutdown(borrow, nullptr).category == FJ_STATUS_SUCCESS, "metadata shutdown failed");
            require(JuicerCuda::borrowed_owner() == borrow && destructions == cycle, "shutdown consumed caller ownership");
            reject_accepting_calls(borrow);
            const auto attempts = shutdownAttempts.load();
            require(fj_cuda_shutdown(borrow, nullptr).category == FJ_STATUS_SUCCESS && shutdownAttempts == attempts, "closed shutdown retried");
            require(owner.close().category == FJ_STATUS_SUCCESS, "closed destroy failed");
            require(!JuicerCuda::borrowed_owner() && destructions == cycle + 1, "closed destroy did not release exactly once");
            require(owner.close().category == FJ_STATUS_SUCCESS && destructions == cycle + 1, "empty wrapper consumed twice");
        }
        const auto attempts = shutdownAttempts.load();
        {
            JuicerCuda::Owner owner;
            owner.create("accepting-destroy");
        }
        require(shutdownAttempts == attempts + 1 && destructions == 4 && !JuicerCuda::borrowed_owner(),
                "accepting destructor did not make exactly one shutdown attempt");
    }

    void check_serialization() {
        JuicerCuda::Owner owner;
        owner.create("admission-order");
        auto outerGuard = JuicerProcess::root().begin_frame_preparation();
        FjCuda* borrow = JuicerCuda::borrowed_owner();
        FjStatus result{};
        std::thread terminal([&] {
            result = fj_cuda_shutdown(borrow, nullptr);
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
        JuicerCuda::Owner owner;
        owner.create("unregistered-terminal-handle");
        auto* const invalid = reinterpret_cast<FjCuda*>(std::uintptr_t{8});
        if (mode == "unregistered") {
            require(fj_cuda_shutdown(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "unregistered shutdown accepted");
            require(fj_cuda_destroy(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "unregistered destroy accepted");
        } else if (mode == "unregistered-reentry") {
            JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
            require(fj_cuda_shutdown(invalid, nullptr).category == FJ_STATUS_UNSUPPORTED_INPUT, "reentrant shutdown accepted");
            require(fj_cuda_destroy(invalid, nullptr).category == FJ_STATUS_PREPARATION_FAILURE, "reentrant unregistered destroy accepted");
        } else {
            failOwnerLock = true;
            const auto result = mode == "unregistered-shutdown-lock" ? fj_cuda_shutdown(invalid, nullptr) : fj_cuda_destroy(invalid, nullptr);
            require(!failOwnerLock && result.category == FJ_STATUS_INTERNAL_FAILURE, "unregistered admission exception not contained");
        }
        require(fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), 1, nullptr).category == FJ_STATUS_SUCCESS, "invalid handle changed registered admission");
        require(owner.close().category == FJ_STATUS_SUCCESS && destructions == 1, "valid owner was lost");
    }

    void check_admission_failure(const std::string& mode) {
        const bool saved = mode.starts_with("saved-");
        const bool consuming = mode.ends_with("destroy-lock");
        const FjStatus expected = saved ? FjStatus{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_DRIVER, -701}
                                        : FjStatus{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        {
            JuicerCuda::Owner owner;
            owner.create("terminal-admission-exception");
            FjCuda* borrow = JuicerCuda::borrowed_owner();
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            if (saved) {
                injection = "typed";
                require(same_status(fj_cuda_shutdown(borrow, &error), expected), "initial typed shutdown failed");
                injection.clear();
            }
            failOwnerLock = true;
            const auto before = allocations.load();
            const auto result = consuming ? owner.close(&error) : fj_cuda_shutdown(borrow, &error);
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
        require(destructions == 0 && shutdownAttempts == (saved ? 1u : 0u), "destructor retried retained admission failure");
    }

    void check_buffers(const std::string& mode) {
        const bool blocked = mode == "malformed-blocked-destroy";
        {
            JuicerCuda::Owner owner;
            owner.create("terminal-error-buffer");
            FjCuda* borrow = JuicerCuda::borrowed_owner();
            FjErrorBuffer malformed{nullptr, 1, 99};
            if (mode == "buffers") {
                require(fj_cuda_shutdown(borrow, &malformed).category == FJ_STATUS_UNSUPPORTED_INPUT, "malformed borrowed shutdown accepted");
                require(fj_cuda_retire_instance(borrow, 1, nullptr).category == FJ_STATUS_SUCCESS, "malformed shutdown closed admission");
                require(shutdownAttempts == 0, "malformed shutdown attempted drain");
                FjErrorBuffer zero{nullptr, 0, 99};
                require(fj_cuda_shutdown(borrow, &zero).category == FJ_STATUS_SUCCESS && zero.length == 0, "zero-capacity shutdown rejected");
                require(owner.close().category == FJ_STATUS_SUCCESS, "valid destroy failed");
            } else {
                if (blocked) {
                    injection = "typed";
                    require(fj_cuda_shutdown(borrow, nullptr).native_code == -701, "typed setup failed");
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
        require(shutdownAttempts == 1 && destructions == (blocked ? 0u : 1u), "malformed output abandoned or retried owner");
    }

    void check_shutdown_progress() {
        JuicerCuda::Owner owner;
        owner.create("shutdown-progress");
        FjCuda* borrow = JuicerCuda::borrowed_owner();
        injection = "typed";
        holdShutdown = true;
        FjStatus first{}, second{};
        std::array<char, 128> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        std::thread terminal([&] {
            first = fj_cuda_shutdown(borrow, nullptr);
        });
        shutdownAttempts.wait(0);
        std::thread rejected([&] {
            failOwnerLock = true;
            second = fj_cuda_shutdown(borrow, &error);
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
            JuicerCuda::Owner owner;
            owner.create("retained-terminal-owner");
            FjCuda* borrow = JuicerCuda::borrowed_owner();
            injection = mode;
            // The same exception seam also proves the non-consuming instance C
            // entry contains exceptions while preserving accepting admission.
            require(same_status(fj_cuda_retire_instance(borrow, 1, nullptr), expected), "retire exception lost typed origin");
            require(JuicerProcess::root().begin_frame_preparation().active(), "retire exception closed owner admission");
            std::array<char, 128> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto result = destroyAccepting ? owner.close(&error) : fj_cuda_shutdown(borrow, &error);
            require(same_status(result, expected) && error.length != 0, "terminal failure lost category/API/code/diagnostic");
            require(shutdownAttempts == 1 && destructions == 0 && JuicerCuda::borrowed_owner() == borrow,
                    "failed terminal call retried, freed or unregistered graph");
            if (!destroyAccepting) {
                reject_accepting_calls(borrow);
                const std::string original(diagnostic.data());
                injection.clear();
                for (std::size_t capacity : {0u, 1u, 8u}) {
                    error = {diagnostic.data(), capacity, 99};
                    require(same_status(fj_cuda_shutdown(borrow, &error), expected), "diagnostic capacity changed terminal status");
                    require(error.length == (capacity ? capacity - 1 : 0), "terminal diagnostic length wrong");
                    if (capacity) {
                        require(diagnostic[capacity - 1] == '\0', "terminal diagnostic missing terminator");
                        require(original.compare(0, error.length, diagnostic.data(), error.length) == 0, "terminal diagnostic changed after failure");
                    }
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
        require(fj_cuda_create({"new", 3}, &replacement, nullptr).category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
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
    void before_owner_lock() {
        if (std::exchange(failOwnerLock, false)) {
            ownerLockAttempted = true;
            ownerLockAttempted.notify_all();
            throw std::system_error{ownerLockFailure};
        }
    }
    void before_shutdown() {
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
        const std::string mode = argc > 1 ? argv[1] : "success";
        if (mode == "success") {
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
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL terminal ownership: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("FAIL terminal ownership: escaped exception\n", stderr);
        return 1;
    }
}
