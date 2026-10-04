#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ProfileCatalog.h"
#include "ProfileAssets.h"
#include "SpectralData.h"
#include "juicer_legacy_api.h"

namespace JuicerCuda {
    struct StaticNoiseInput;
}

namespace JuicerAssets {
    struct IlluminantFilterCurveSet;

    enum class NeutralPrintCalibrationStatus : unsigned char {
        MissingFile,
        MissingEntry,
        Found,
        Malformed
    };

    struct NeutralPrintCalibrationResult {
        NeutralPrintCalibrationStatus status =
            NeutralPrintCalibrationStatus::MissingEntry;
        std::array<float, 3> cmyCc{};
        std::string diagnostic;
    };

    enum class CsvSource : unsigned char {
        D65,
        D55,
        D50,
        T,
        K75p,
        Kg3,
        Canon24F28Is
    };

#if defined(JUICER_ILLUMINANT_TEST_HOOK)
    namespace IlluminantTest {
        void before_csv_acquisition(CsvSource source);
        void csv_view(CsvSource source, FjFloatSpan& view);
        void before_csv_copy(CsvSource source, std::size_t rows);
        void after_csv_copy(CsvSource source, std::size_t rows, std::size_t capacity);
        void calibration_result(FjNeutralCalibrationResult& result);
        void film_reference_samples(const std::array<float, 81>& samples);
        void before_curve_publication(const std::shared_ptr<const IlluminantFilterCurveSet>& candidate);
    } // namespace IlluminantTest
#endif

    // Synchronous argument: native path bytes borrow the path; Windows units
    // are copied by value, never aliased between wchar_t and uint16_t.
    class NativePathArgument final {
    public:
        explicit NativePathArgument(const std::filesystem::path& path);
        FjPathView view() const noexcept;

    private:
#if defined(_WIN32)
        std::vector<std::uint16_t> _units;
#else
        const std::string& _units;
#endif
    };

    std::filesystem::path copy_native_path(FjPathView view);
    std::string path_diagnostic(const std::filesystem::path& path);

#if defined(JUICER_CATALOG_TEST_HOOK)
    namespace CatalogTest {
        void before_conversion(FjCatalogCounts& counts);
        void before_source_diagnostic();
    } // namespace CatalogTest
#endif

#if defined(JUICER_NOISE_TEST_HOOK)
    namespace NoiseTest {
        enum class Operation : unsigned char {
            Acquire,
            View,
            Release
        };
        void observe(Operation operation) noexcept;
        void view(FjStaticNoise& view);
        void projection_step() noexcept;
        void outcome(FjStatus status) noexcept;
    } // namespace NoiseTest
#endif

    // FJ_TEMP_BRIDGE: noise source borrow; remove S4.E.
    class NoiseSource final {
    public:
        ~NoiseSource();
        NoiseSource(NoiseSource&& source) noexcept;
        NoiseSource(const NoiseSource&) = delete;
        NoiseSource& operator=(const NoiseSource&) = delete;
        NoiseSource& operator=(NoiseSource&&) = delete;
        JuicerCuda::StaticNoiseInput view() const&;
        JuicerCuda::StaticNoiseInput view() const&& = delete;

    private:
        friend class AssetBridge;
        NoiseSource(FjNoise* owner, const FjStaticNoise& view) noexcept;
        FjNoise* _owner;
        FjStaticNoise _view;
    };

    class PrintProfileSource final {
    public:
        ~PrintProfileSource();
        PrintProfileSource(const PrintProfileSource&) = delete;
        PrintProfileSource& operator=(const PrintProfileSource&) = delete;
        const std::shared_ptr<const Profiles::PrintProfile>& profile() const noexcept;
        Profiles::PrintDensityCurves sample_density_curves(double gamma) const;

    private:
        friend class AssetBridge;
        FjStatus close(FjErrorBuffer* error = nullptr) const noexcept;
        PrintProfileSource(FjPrintProfile* owner, std::shared_ptr<const Profiles::PrintProfile> profile);
        mutable FjPrintProfile* _owner = nullptr;
        std::shared_ptr<const Profiles::PrintProfile> _profile;
    };

#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
    namespace ProfileTest {
        void film_view(FjFilmProfileView& view);
        void print_view(FjPrintProfileView& view);
        void after_tables();
        void before_publication();
        void before_catalog_publication();
    } // namespace ProfileTest
#endif

#if defined(JUICER_SPECTRAL_TEST_HOOK)
    namespace SpectralTest {
        void lut_view(const char* family, FjSpectraLutView& view);
        void mallett_view(FjFloatSpan& view);
        void cmf_view(FjFloatSpan& view);
        void copy_allocated(const char* family, std::size_t capacity);
        void copy_complete(const char* family, std::size_t length, std::size_t capacity);
        void before_curve_allocation();
    } // namespace SpectralTest
#endif

    // FJ_TEMP_BRIDGE: asset conversion; remove S4.E.
    // All retained foreign owners nest inside the detachable native Library.
    class AssetBridge final {
    public:
        explicit AssetBridge(const std::filesystem::path& resourceRoot);
        ~AssetBridge();
        AssetBridge(const AssetBridge&) = delete;
        AssetBridge& operator=(const AssetBridge&) = delete;

        NoiseSource noise();
        Spectral::ReconstructionLut copy_hanatos_lut();
        Spectral::ReconstructionLut copy_arctic_lut();
        Spectral::MallettBasis copy_mallett_basis();
        Spectral::CMFTriplets copy_cmf_triplets();
        std::vector<std::pair<float, float>> copy_csv_pairs(CsvSource source);
        NeutralPrintCalibrationResult neutral_print_calibration(const std::string& printStock,
                                                                const std::string& illuminant,
                                                                const std::string& filmStock);
        Spektrafilm::ProfileCatalog load_catalog();
        std::shared_ptr<const Profiles::FilmProfile> film(const std::string& key);
        std::shared_ptr<const PrintProfileSource> print(const std::string& key);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        struct ConversionSlots;
        // Catalog readers use Library's call_once; cache release uses this lock.
        std::mutex _slotsMutex;
        std::unique_ptr<ConversionSlots> _slots;
        FjAssets* _assets = nullptr;
        FjCatalog* _catalog = nullptr;
    };

} // namespace JuicerAssets
