#pragma once

#include <cstdint>

namespace JuicerTestSupport {
    struct PreparationInput;

    // Independent captured-input oracle; keys are owned by input. Scaled shadow
    // controls retain the original finite, representable integer-cast domain.
    std::uint64_t hash_input_identity(const PreparationInput& input);
} // namespace JuicerTestSupport
