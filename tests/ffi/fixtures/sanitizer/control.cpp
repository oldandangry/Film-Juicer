#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "juicer_cuda_prepared.h"

// Intentional sanitizer failures only. Never link this entry into the plug-in.
int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    const std::string_view mode(argv[1]);
    if (mode == "memory") {
        auto storage = std::make_unique<FjPreparedHostData>();
        const auto* borrowed = storage.get();
        storage.reset();
        JuicerCuda::PreparedDescriptors descriptors;
        std::string diagnostic;
        // The report must originate in the instrumented production decoder.
        return JuicerCuda::decode_prepared_descriptors(*borrowed, descriptors, diagnostic) ? 0 : 1;
    }
    if (mode == "undefined") {
        volatile int maximum = std::numeric_limits<int>::max();
        volatile int one = 1;
        return maximum + one;
    }
    return mode == "clean" ? 0 : 2;
}
