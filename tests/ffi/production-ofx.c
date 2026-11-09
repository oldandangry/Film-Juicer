#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "ofx_bindings.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int (*PluginCount)(void);
typedef OfxPlugin* (*PluginAt)(int);
_Static_assert(sizeof(OfxStatus) == 4 && sizeof(OfxTime) == 8, "OFX scalar ABI");
_Static_assert(sizeof(OfxPlugin) == 48 && _Alignof(OfxPlugin) == 8, "OFX descriptor ABI");
_Static_assert(offsetof(OfxPlugin, setHost) == 32 && offsetof(OfxPlugin, mainEntry) == 40, "OFX callback ABI");

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    if (argc != 2)
        return 1;
    HMODULE module = LoadLibraryW(argv[1]);
    if (!module)
        return 2;
    PluginCount count;
    PluginAt plugin_at;
    FARPROC count_symbol = GetProcAddress(module, "OfxGetNumberOfPlugins");
    FARPROC plugin_symbol = GetProcAddress(module, "OfxGetPlugin");
#else
int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    void* module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!module)
        return 2;
    PluginCount count;
    PluginAt plugin_at;
    void* count_symbol = dlsym(module, "OfxGetNumberOfPlugins");
    void* plugin_symbol = dlsym(module, "OfxGetPlugin");
#endif
    _Static_assert(sizeof(count) == sizeof(count_symbol) && sizeof(plugin_at) == sizeof(plugin_symbol), "platform function-pointer ABI");
    memcpy(&count, &count_symbol, sizeof(count));
    memcpy(&plugin_at, &plugin_symbol, sizeof(plugin_at));
    int result = 0;
    if (!count || !plugin_at || count() != 1 || plugin_at(-1) || plugin_at(1) || plugin_at(INT_MIN) || plugin_at(INT_MAX)) {
        result = 3;
    } else {
        const OfxPlugin* plugin = plugin_at(0);
        if (!plugin || plugin != plugin_at(0) || !plugin->setHost || !plugin->mainEntry ||
            strcmp(plugin->pluginApi, kOfxImageEffectPluginApi) != 0 ||
            plugin->apiVersion != kOfxImageEffectPluginApiVersion ||
            strcmp(plugin->pluginIdentifier, "com.juicer.Juicer") != 0 ||
            plugin->pluginVersionMajor != 1 || plugin->pluginVersionMinor != 0)
            result = 4;
    }
#if defined(_WIN32)
    if (!FreeLibrary(module))
        return 5;
#else
    if (dlclose(module) != 0)
        return 5;
#endif
    return result;
}
