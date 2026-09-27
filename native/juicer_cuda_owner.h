#pragma once

#include <mutex>
#include <string_view>

#include "juicer_cuda_api.h"

namespace JuicerCuda {

    // Callback-local borrow of the registered owner; never constructs one.
    FjCuda* borrowed_owner() noexcept;

    // Private callback-local admission. Owns no runtime or CUDA context. The
    // host lifetime excludes owner destruction until this borrow ends.
    class NativeCall final {
    public:
        explicit NativeCall(FjCuda* cuda);
        ~NativeCall();
        NativeCall(const NativeCall&) = delete;
        NativeCall& operator=(const NativeCall&) = delete;

        FjStatus inspect(const FjFrame* frame, FjCudaContext* outContext, FjErrorBuffer* error);

    private:
        std::unique_lock<std::mutex> _lock;
    };

#if defined(JUICER_CUDA_RENDER_TEST_HOOK)
    namespace RenderTest {
        void before_execute();
        void after_execute();
    } // namespace RenderTest
#endif

#if defined(JUICER_CUDA_TERMINAL_TEST_HOOK)
    namespace TerminalTest {
        void before_owner_lock();
        void before_shutdown();
        void before_retire_instance();
        void owner_destroyed() noexcept;
    } // namespace TerminalTest
#endif

    // Temporary C++ runtime holder; replaced by the Rust owner at its cutover.
    // Host load/unload serializes lifetime against all borrowed render calls.
    class Owner final {
    public:
        Owner() noexcept = default;
        ~Owner();

        Owner(const Owner&) = delete;
        Owner& operator=(const Owner&) = delete;

        void create(std::string_view dataDirectory);
        // Consumes the handle even on failure; destruction never retries it.
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        FjCuda* _cuda = nullptr;
    };

} // namespace JuicerCuda
