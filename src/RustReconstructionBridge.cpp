#include "RustReconstructionBridge.h"

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "GamutCompression.h"
#include "RenderRecipe.h"
#include "SpectralData.h"

namespace Spectral {
    namespace {
        struct Diagnostic {
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        };
        bool completed(FjStatus status, const Diagnostic& error, std::string& diagnostic, const char* operation) {
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc();
            }
            if (status.category == FJ_STATUS_PREPARATION_FAILURE) {
                diagnostic.assign(error.bytes.data(), error.error.length);
                return false;
            }
            if (status.category != FJ_STATUS_SUCCESS) {
                JuicerCuda::Failure failure;
                JuicerCuda::set_failure(failure, status, std::string(operation) + ": " + std::string(error.bytes.data(), error.error.length));
                throw JuicerCuda::ExecutionFailure{std::move(failure)};
            }
            return true;
        }
#if defined(JUICER_TC_LUT_TEST_HOOK)
        std::atomic<TcLutTest::ReleaseObserver> releaseObserver{nullptr};
#endif
    } // namespace

    static_assert(sizeof(FilmTcLut) == sizeof(FjOwnedFilmTcLut));
    static_assert(!std::is_default_constructible_v<FilmTcLut> && !std::is_copy_constructible_v<FilmTcLut> && !std::is_copy_assignable_v<FilmTcLut>);
    static_assert(std::is_nothrow_move_constructible_v<FilmTcLut> && std::is_nothrow_move_assignable_v<FilmTcLut>);
    static_assert(std::is_same_v<decltype(std::declval<const FilmTcLut&>().samples()), std::span<const float>>);
    static_assert(sizeof(FjFilmTcLutInput) == 120 && alignof(FjFilmTcLutInput) == 8);
    static_assert(sizeof(FjOwnedFilmTcLut) == 24 && alignof(FjOwnedFilmTcLut) == 8);
    static_assert(sizeof(std::array<std::array<float, 3>, 81>) == 243 * sizeof(float));
    static_assert(sizeof(std::array<std::array<float, 15>, 3>) == 45 * sizeof(float));
    static_assert(sizeof(std::array<std::array<float, 2>, 1025>) == 2050 * sizeof(float));

    FilmTcLut::FilmTcLut(FjOwnedFilmTcLut& owned) noexcept : owned_(std::exchange(owned, {})) {}
    FilmTcLut::FilmTcLut(FilmTcLut&& other) noexcept : owned_(std::exchange(other.owned_, {})) {}
    FilmTcLut& FilmTcLut::operator=(FilmTcLut&& other) noexcept {
        if (this != &other) {
            release();
            owned_ = std::exchange(other.owned_, {});
        }
        return *this;
    }
    FilmTcLut::~FilmTcLut() noexcept {
        release();
    }
    void FilmTcLut::release() noexcept {
        if (owned_.samples.data) {
#if defined(JUICER_TC_LUT_TEST_HOOK)
            if (const auto observer = releaseObserver.load(std::memory_order_acquire)) {
                observer(owned_.samples.data);
            }
#endif
            // The exact module-owned token has no recoverable release failure.
            // Native code never deletes the allocation or introduces a retry.
            (void)fj_legacy_reconstruction_release_tc_lut(&owned_);
        }
    }
    std::span<const float> FilmTcLut::samples() const noexcept {
        return {owned_.samples.data, owned_.samples.count};
    }
#if defined(JUICER_TC_LUT_TEST_HOOK)
    void TcLutTest::set_release_observer(ReleaseObserver observer) noexcept {
        releaseObserver.store(observer, std::memory_order_release);
    }
#endif

    bool build_film_tc_lut(const FilmRawRecipe& recipe, const ReconstructionLut& spectra, const std::array<float, 81>& referenceIlluminant, std::optional<FilmTcLut>& out, std::string& diagnostic) {
        out.reset();
        diagnostic.clear();
        if (recipe.tcLutHash == 0 ||
            (recipe.rgbToRawMethod != Spektrafilm::RgbToRawMethod::Hanatos2025 &&
             recipe.rgbToRawMethod != Spektrafilm::RgbToRawMethod::Arctic2026beta04)) {
            diagnostic = "MalformedRequiredResource component=film_tc_lut requirement=selected_finite_192x192x81_spectra";
            return false;
        }
        const bool surface = recipe.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 && recipe.hanatos.applySurface;
        const auto* hull = recipe.inputCompressionActive ? recipe.inputCompressionHull.get() : nullptr;
        const bool available = hull && hull->valid && hull->hash != 0;
        const FjFilmTcLutInput input{
            {spectra.data.data(), spectra.data.size()},
            {recipe.finalSensitivity.front().data(), 243},
            {referenceIlluminant.data(), referenceIlluminant.size()},
            {recipe.projectionWhiteXYZ[0], recipe.projectionWhiteXYZ[1], recipe.projectionWhiteXYZ[2]},
            recipe.hanatos.spectralGaussianBlur,
            static_cast<std::uint32_t>(recipe.rgbToRawMethod),
            recipe.hanatos.applySurface ? 1u : 0u,
            {surface ? recipe.hanatos.surfaceParams.front().data() : nullptr, surface ? 45u : 0u},
            recipe.inputCompressionActive ? 1u : 0u,
            available ? 1u : 0u,
            {available ? hull->center[0] : 0.0f, available ? hull->center[1] : 0.0f},
            {available ? hull->xy.front().data() : nullptr, available ? 2050u : 0u}};
        FjOwnedFilmTcLut result{};
        Diagnostic error;
        if (!completed(fj_legacy_reconstruction_tc_lut(&input, &result, &error.error), error, diagnostic, "Film TC preparation")) {
            return false;
        }
        FilmTcLut owner(result);
        out.emplace(std::move(owner));
        return true;
    }
    bool sample_film_tc_lut(const FilmTcLut& lut, const std::array<float, 3>& projectedXyz, std::array<float, 3>& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const auto samples = lut.samples();
        Diagnostic error;
        return completed(fj_legacy_reconstruction_sample_tc_lut({samples.data(), samples.size()}, projectedXyz.data(), out.data(), &error.error), error, diagnostic, "Film TC sampling");
    }
} // namespace Spectral
