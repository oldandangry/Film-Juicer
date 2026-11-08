#ifndef _WIN32
// glibc exposes Dl_info/dladdr under this prescribed feature-test macro.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _GNU_SOURCE
#endif

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "module_test.h"

#ifdef _WIN32
_Static_assert((DWORD)-1 > 0, "DWORD unsigned");
_Static_assert((BOOL)-1 < 0, "BOOL signed");
_Static_assert((WCHAR)-1 > 0, "WCHAR unsigned");
_Static_assert(_Generic(&GetModuleHandleExW, BOOL(WINAPI*)(DWORD, LPCWSTR, HMODULE*): 1, default: 0), "GetModuleHandleExW signature");
_Static_assert(_Generic(&GetModuleFileNameW, DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD): 1, default: 0), "GetModuleFileNameW signature");
_Static_assert(_Generic(&SetLastError, VOID(WINAPI*)(DWORD): 1, default: 0), "SetLastError signature");
_Static_assert(_Generic(&GetLastError, DWORD(WINAPI*)(void): 1, default: 0), "GetLastError signature");
_Static_assert(sizeof(LARGE_INTEGER) == sizeof(long long), "QPC output width");
_Static_assert(_Alignof(LARGE_INTEGER) == _Alignof(long long), "QPC output alignment");
_Static_assert(_Generic(&QueryPerformanceCounter, BOOL(WINAPI*)(LARGE_INTEGER*): 1, default: 0), "QueryPerformanceCounter signature");
_Static_assert(_Generic(&QueryPerformanceFrequency, BOOL(WINAPI*)(LARGE_INTEGER*): 1, default: 0), "QueryPerformanceFrequency signature");
#else
_Static_assert(_Generic(&dladdr, int (*)(const void*, Dl_info*): 1, default: 0), "dladdr signature");
_Static_assert(_Generic(((Dl_info*)0)->dli_fname, const char*: 1, default: 0), "Dl_info filename");
_Static_assert(_Generic(((Dl_info*)0)->dli_fbase, void*: 1, default: 0), "Dl_info base");
_Static_assert(_Generic(((Dl_info*)0)->dli_sname, const char*: 1, default: 0), "Dl_info symbol");
_Static_assert(_Generic(((Dl_info*)0)->dli_saddr, void*: 1, default: 0), "Dl_info address");
#endif

int fj_test_ofx_platform_c(void) {
#ifdef _WIN32
    const size_t facts[] = {sizeof(WCHAR), sizeof(DWORD), sizeof(BOOL), sizeof(HMODULE), _Alignof(HMODULE), GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, sizeof(LARGE_INTEGER), _Alignof(LARGE_INTEGER)};
#else
    const size_t facts[] = {sizeof(Dl_info), _Alignof(Dl_info), offsetof(Dl_info, dli_fname), offsetof(Dl_info, dli_fbase), offsetof(Dl_info, dli_sname), offsetof(Dl_info, dli_saddr)};
#endif
    return fj_test_ofx_platform_abi(facts, sizeof(facts) / sizeof(facts[0]));
}
