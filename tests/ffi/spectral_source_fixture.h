#pragma once

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "RustAssetBridge.h"
#include "juicer_legacy_api.h"

namespace JuicerTestSupport {
    inline void require_spectral_source_status(FjStatus status, const char* operation) {
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
            throw std::runtime_error(std::string("spectral source ") + operation + " failed");
        }
    }

    class SpectralSourceAssets {
    public:
        explicit SpectralSourceAssets(const std::filesystem::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            require_spectral_source_status(fj_legacy_assets_create(path.view(), &owner_, nullptr), "asset creation");
        }

        ~SpectralSourceAssets() noexcept {
            if (owner_ != nullptr) {
                const FjStatus status = fj_legacy_assets_destroy(std::exchange(owner_, nullptr), nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }

        SpectralSourceAssets(const SpectralSourceAssets&) = delete;
        SpectralSourceAssets& operator=(const SpectralSourceAssets&) = delete;

        FjAssets* get() const noexcept {
            return owner_;
        }

    private:
        FjAssets* owner_ = nullptr;
    };

    class HanatosSource {
    public:
        explicit HanatosSource(const FjAssets* assets) {
            require_spectral_source_status(fj_legacy_hanatos_acquire(assets, &owner_, nullptr), "Hanatos acquire");
            try {
                require_spectral_source_status(fj_legacy_spectra_lut_view(owner_, &view_, nullptr), "Hanatos view");
                if (view_.samples.data == nullptr || view_.samples.count != kLutSamples) {
                    throw std::runtime_error("spectral source Hanatos extent mismatch");
                }
            } catch (...) {
                release();
                throw;
            }
        }

        ~HanatosSource() noexcept {
            release();
        }

        HanatosSource(const HanatosSource&) = delete;
        HanatosSource& operator=(const HanatosSource&) = delete;

        const FjSpectraLutView& view() const& noexcept {
            return view_;
        }

        const FjSpectraLutView& view() const&& = delete;

    private:
        static constexpr std::size_t kLutSamples = std::size_t{192} * 192u * 81u;
        FjSpectraLut* owner_ = nullptr;
        FjSpectraLutView view_{};

        void release() noexcept {
            if (owner_ != nullptr) {
                FjSpectraLut* owner = std::exchange(owner_, nullptr);
                const FjStatus status = fj_legacy_spectra_lut_release(owner, nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }
    };

    class ArcticSource {
    public:
        explicit ArcticSource(const FjAssets* assets) {
            require_spectral_source_status(fj_legacy_arctic_acquire(assets, &owner_, nullptr), "Arctic acquire");
            try {
                require_spectral_source_status(fj_legacy_spectra_lut_view(owner_, &view_, nullptr), "Arctic view");
                if (view_.samples.data == nullptr || view_.samples.count != kLutSamples) {
                    throw std::runtime_error("spectral source Arctic extent mismatch");
                }
            } catch (...) {
                release();
                throw;
            }
        }

        ~ArcticSource() noexcept {
            release();
        }

        ArcticSource(const ArcticSource&) = delete;
        ArcticSource& operator=(const ArcticSource&) = delete;

        const FjSpectraLutView& view() const& noexcept {
            return view_;
        }

        const FjSpectraLutView& view() const&& = delete;

    private:
        static constexpr std::size_t kLutSamples = std::size_t{192} * 192u * 81u;
        FjSpectraLut* owner_ = nullptr;
        FjSpectraLutView view_{};

        void release() noexcept {
            if (owner_ != nullptr) {
                FjSpectraLut* owner = std::exchange(owner_, nullptr);
                const FjStatus status = fj_legacy_spectra_lut_release(owner, nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }
    };

    class MallettSource {
    public:
        explicit MallettSource(const FjAssets* assets) {
            require_spectral_source_status(fj_legacy_mallett_acquire(assets, &owner_, nullptr), "Mallett acquire");
            try {
                require_spectral_source_status(fj_legacy_mallett_view(owner_, &samples_, nullptr), "Mallett view");
                if (samples_.data == nullptr || samples_.count != 243u) {
                    throw std::runtime_error("spectral source Mallett extent mismatch");
                }
            } catch (...) {
                release();
                throw;
            }
        }

        ~MallettSource() noexcept {
            release();
        }

        MallettSource(const MallettSource&) = delete;
        MallettSource& operator=(const MallettSource&) = delete;

        const FjFloatSpan& samples() const& noexcept {
            return samples_;
        }

        const FjFloatSpan& samples() const&& = delete;

    private:
        FjMallettBasis* owner_ = nullptr;
        FjFloatSpan samples_{};

        void release() noexcept {
            if (owner_ != nullptr) {
                FjMallettBasis* owner = std::exchange(owner_, nullptr);
                const FjStatus status = fj_legacy_mallett_release(owner, nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }
    };

    class CmfSource {
    public:
        explicit CmfSource(const FjAssets* assets) {
            require_spectral_source_status(fj_legacy_cmf_acquire(assets, &owner_, nullptr), "CMF acquire");
            try {
                require_spectral_source_status(fj_legacy_cmf_view(owner_, &rows_, nullptr), "CMF view");
                if (rows_.data == nullptr || rows_.count != 4u * kSamples) {
                    throw std::runtime_error("spectral source CMF extent mismatch");
                }
                for (std::size_t i = 0; i < kSamples; ++i) {
                    x_[i] = rows_.data[4u * i + 1u];
                    y_[i] = rows_.data[4u * i + 2u];
                    z_[i] = rows_.data[4u * i + 3u];
                }
            } catch (...) {
                release();
                throw;
            }
        }

        ~CmfSource() noexcept {
            release();
        }

        CmfSource(const CmfSource&) = delete;
        CmfSource& operator=(const CmfSource&) = delete;

        const FjFloatSpan& rows() const& noexcept {
            return rows_;
        }

        const FjFloatSpan& rows() const&& = delete;

        FjFloatSpan x() const& noexcept {
            return {x_.data(), x_.size()};
        }

        FjFloatSpan x() const&& = delete;

        FjFloatSpan y() const& noexcept {
            return {y_.data(), y_.size()};
        }

        FjFloatSpan y() const&& = delete;

        FjFloatSpan z() const& noexcept {
            return {z_.data(), z_.size()};
        }

        FjFloatSpan z() const&& = delete;

    private:
        static constexpr std::size_t kSamples = 81;
        FjCmf* owner_ = nullptr;
        FjFloatSpan rows_{};
        std::array<float, kSamples> x_{}, y_{}, z_{};

        void release() noexcept {
            if (owner_ != nullptr) {
                FjCmf* owner = std::exchange(owner_, nullptr);
                const FjStatus status = fj_legacy_cmf_release(owner, nullptr);
                if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                    std::abort();
                }
            }
        }
    };
} // namespace JuicerTestSupport
