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
        using CuPointerGetAttributeFn = CUresult(CUDAAPI*)(void*, CUpointer_attribute, CUdeviceptr);
        using CuMemGetAddressRangeFn = CUresult(CUDAAPI*)(CUdeviceptr*, std::size_t*, CUdeviceptr);

        struct CudaDriverDispatch {
            CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
            CuPointerGetAttributeFn cuPointerGetAttribute = nullptr;
            CuMemGetAddressRangeFn cuMemGetAddressRange = nullptr;
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
                result.cuPointerGetAttribute = reinterpret_cast<CuPointerGetAttributeFn>(GetProcAddress(module, "cuPointerGetAttribute"));
                result.cuMemGetAddressRange = reinterpret_cast<CuMemGetAddressRangeFn>(GetProcAddress(module, "cuMemGetAddressRange_v2"));
#elif defined(__linux__)
                void* module = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
                if (!module) {
                    result.loadError = "libcuda.so.1 not available";
                    return result;
                }
                result.cuCtxGetCurrent =
                    reinterpret_cast<CuCtxGetCurrentFn>(dlsym(module, "cuCtxGetCurrent"));
                result.cuPointerGetAttribute = reinterpret_cast<CuPointerGetAttributeFn>(dlsym(module, "cuPointerGetAttribute"));
                result.cuMemGetAddressRange = reinterpret_cast<CuMemGetAddressRangeFn>(dlsym(module, "cuMemGetAddressRange_v2"));
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

    bool query_current_cuda_context(void*& outContextOpaque, std::string& outError, int* nativeCode) {
        outContextOpaque = nullptr;
        outError.clear();
        if (nativeCode) {
            *nativeCode = 0;
        }

        const CudaDriverDispatch& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuCtxGetCurrent) {
            outError = dispatch.loadError ? dispatch.loadError : "driver dispatch unavailable";
            return false;
        }

        CUcontext currentContext = nullptr;
        const CUresult result = dispatch.cuCtxGetCurrent(&currentContext);
        if (result != CUDA_SUCCESS) {
            if (nativeCode) {
                *nativeCode = static_cast<int>(result);
            }
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

    bool query_cuda_pointer_context(std::uintptr_t address, void*& outContext, int& nativeCode, std::string& outError) {
        outContext = nullptr;
        nativeCode = 0;
        outError.clear();
        const auto& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuPointerGetAttribute) {
            outError = "cuPointerGetAttribute unavailable";
            return false;
        }
        CUcontext context = nullptr;
        const CUresult result = dispatch.cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, address);
        if (result != CUDA_SUCCESS) {
            nativeCode = static_cast<int>(result);
            outError = "cuPointerGetAttribute(CONTEXT) failed";
            return false;
        }
        outContext = reinterpret_cast<void*>(context);
        return true;
    }

    bool query_cuda_allocation(std::uintptr_t address, CudaAllocation& allocation, int& nativeCode, std::string& outError) {
        allocation = {};
        nativeCode = 0;
        outError.clear();
        const auto& dispatch = cuda_driver_dispatch();
        if (!dispatch.cuPointerGetAttribute || !dispatch.cuMemGetAddressRange) {
            outError = "CUDA allocation query symbols unavailable";
            return false;
        }
        CUdeviceptr base = 0;
        std::size_t bytes = 0;
        const CUresult result = dispatch.cuMemGetAddressRange(&base, &bytes, address);
        if (result != CUDA_SUCCESS) {
            nativeCode = static_cast<int>(result);
            outError = "cuMemGetAddressRange failed";
            return false;
        }
        allocation = {static_cast<std::uintptr_t>(base), bytes};
        return true;
    }

} // namespace JuicerCuda
