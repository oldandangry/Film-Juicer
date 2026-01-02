// Cuda/JuicerCudaSelfCheck.h
//
// Runtime CUDA self-check.
//
// Intent:
// - Provide a fast "is CUDA actually usable?" probe in the real host environment (Resolve / OFX),
//   without introducing a separate test harness (JUICER_TESTS is intentionally not used).
// - Keep this self-contained and easy to remove later: delete this file + .cu implementation and
//   remove the call site guarded by JUICER_CUDA_SELF_CHECK.
//
// Notes:
// - The OFX CUDA stream pointer is passed through as an opaque void* to avoid leaking CUDA headers
//   into non-CUDA compilation units.
// - Returns true on success; false on failure and fills outError with a stable message pointer.
//
#pragma once

// Returns false and sets *outError on failure (message pointer stays valid for process lifetime).
bool juicer_cuda_runtime_self_check(void* cudaStreamOpaque, const char** outError);
