#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>

#include "juicer_legacy_api.h"

struct FilmRawRecipe;

namespace Spectral {
    struct ReconstructionLut;

    // FJ_TEMP_BRIDGE: single completed Rust TC allocation owner; remove S4.E.
    struct FilmTcLut {
        static constexpr int kSize = 192;
        static constexpr int kChannels = 4;

        FilmTcLut(const FilmTcLut&) = delete;
        FilmTcLut& operator=(const FilmTcLut&) = delete;
        FilmTcLut(FilmTcLut&& other) noexcept;
        FilmTcLut& operator=(FilmTcLut&& other) noexcept;
        ~FilmTcLut() noexcept;
        std::span<const float> samples() const noexcept;

    private:
        explicit FilmTcLut(FjOwnedFilmTcLut& owned) noexcept;
        void release() noexcept;
        FjOwnedFilmTcLut owned_{};
        friend bool build_film_tc_lut(const FilmRawRecipe&, const ReconstructionLut&, const std::array<float, 81>&, std::optional<FilmTcLut>&, std::string&);
    };

    // FJ_TEMP_BRIDGE: native selection/key guard and Rust TC bindings; remove S4.E.
    bool build_film_tc_lut(const FilmRawRecipe& recipe, const ReconstructionLut& spectra, const std::array<float, 81>& referenceIlluminant, std::optional<FilmTcLut>& out, std::string& diagnostic);
    bool sample_film_tc_lut(const FilmTcLut& lut, const std::array<float, 3>& projectedXyz, std::array<float, 3>& out, std::string& diagnostic);

#if defined(JUICER_TC_LUT_TEST_HOOK)
    namespace TcLutTest {
        using ReleaseObserver = void (*)(const float*) noexcept;
        void set_release_observer(ReleaseObserver observer) noexcept;
    } // namespace TcLutTest
#endif
} // namespace Spectral
