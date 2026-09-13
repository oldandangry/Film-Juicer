#include "Cuda/JuicerCudaDriver.h"

#include <cuda.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace JuicerCuda {

    namespace {

        using CuCtxGetCurrentFn = CUresult(CUDAAPI*)(CUcontext*);

        struct CudaDriverDispatch {
            CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
            const char* loadError = nullptr;
        };

        const CudaDriverDispatch& cuda_driver_dispatch() {
            static const CudaDriverDispatch dispatch = []() {
                CudaDriverDispatch result;
                // Retain the loader reference for process lifetime: deferred cleanup
                // can query the driver after individual instances have been destroyed.
#if defined(_WIN32)
                HMODULE module = LoadLibraryA("nvcuda.dll");
                if (!module) {
                    result.loadError = "nvcuda.dll not available";
                    return result;
                }
                result.cuCtxGetCurrent =
                    reinterpret_cast<CuCtxGetCurrentFn>(GetProcAddress(module, "cuCtxGetCurrent"));
#elif defined(__linux__)
                void* module = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
                if (!module) {
                    result.loadError = "libcuda.so.1 not available";
                    return result;
                }
                result.cuCtxGetCurrent =
                    reinterpret_cast<CuCtxGetCurrentFn>(dlsym(module, "cuCtxGetCurrent"));
#else
                result.loadError = "cuCtxGetCurrent loader unsupported on this platform";
                return result;
#endif
                if (!result.cuCtxGetCurrent) {
                    result.loadError = "cuCtxGetCurrent symbol not found";
                }
                return result;
            }();
            return dispatch;
        }

    } // namespace

    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError) {
        outContextOpaque = nullptr;
        outError.clear();

        const CudaDriverDispatch& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuCtxGetCurrent) {
            outError = dispatch.loadError ? dispatch.loadError : "driver dispatch unavailable";
            return false;
        }

        CUcontext currentContext = nullptr;
        const CUresult result = dispatch.cuCtxGetCurrent(&currentContext);
        if (result != CUDA_SUCCESS) {
            outError = std::string("cuCtxGetCurrent failed (code=") + std::to_string(static_cast<int>(result)) + ")";
            return false;
        }
        if (!currentContext) {
            outError = "current CUDA context is null";
            return false;
        }

        outContextOpaque = reinterpret_cast<void*>(currentContext);
        return true;
    }

} // namespace JuicerCuda
