#pragma once

#include <array>
#include <stdexcept>
#include <string>

#include "juicer_test_api.h"

namespace JuicerTestSupport {

    // Values are transport only. record() borrows the two owned keys through an
    // immediate consuming call; mutating or destroying this holder expires them.
    struct PreparationInput final {
        FjPreparationParams values{};
        std::string film_key = "kodak_portra_400";
        std::string print_key = "kodak_portra_endura";

        PreparationInput() {
            std::array<char, 2048> text{};
            FjErrorBuffer error{text.data(), text.size(), 0};
            const FjStatus status = fj_test_preparation_defaults(&values, &error);
            if (status.category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error(error.length ? std::string(text.data(), error.length)
                                                      : "cannot obtain preparation defaults");
            }
            if (values.film_profile_key.data || values.film_profile_key.count ||
                values.print_profile_key.data || values.print_profile_key.count) {
                throw std::runtime_error("preparation defaults contain borrowed keys");
            }
        }

        // The source and both key extents must remain live during construction.
        explicit PreparationInput(const FjPreparationParams& source)
            : values(source), film_key(copy_key(source.film_profile_key)), print_key(copy_key(source.print_profile_key)) {
            values.film_profile_key = {};
            values.print_profile_key = {};
        }

        FjPreparationParams record() const& noexcept {
            auto result = values;
            result.film_profile_key = {film_key.data(), film_key.size()};
            result.print_profile_key = {print_key.data(), print_key.size()};
            return result;
        }
        FjPreparationParams record() && = delete;
        FjPreparationParams record() const&& = delete;

    private:
        static std::string copy_key(FjStringView view) {
            if (view.count == 0)
                return {};
            if (!view.data)
                throw std::runtime_error("preparation key has no backing");
            return {view.data, view.count};
        }
    };

} // namespace JuicerTestSupport
