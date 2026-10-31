#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <mutex>
#include <cstdlib>
#include <optional>
#include <string>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaResources.h"
#include "Illuminants.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ResourceAssetLibrary.h"
#include "RustAssetBridge.h"
#include "ColorTransforms.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "spectral_allocation_probe.h"
#include "prepared_boundary.h"
#include "prepared_product_fixture.h"

static_assert(sizeof(FjNeutralCalibrationResult) == 20 && alignof(FjNeutralCalibrationResult) == 4);
static_assert(offsetof(FjNeutralCalibrationResult, outcome) == 0 && offsetof(FjNeutralCalibrationResult, field) == 4 && offsetof(FjNeutralCalibrationResult, cmy_cc) == 8);
static_assert(sizeof(FjFloatSpan) == 16 && alignof(FjFloatSpan) == 8);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_acquire), FjStatus (*)(const FjAssets*, uint32_t, FjCsvPairs**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_view), FjStatus (*)(const FjCsvPairs*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_release), FjStatus (*)(FjCsvPairs*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_neutral_calibration_lookup), FjStatus (*)(const FjAssets*, FjStringView, FjStringView, FjStringView, FjNeutralCalibrationResult*, FjErrorBuffer*)>);


namespace IlluminantConstruction {
    void initialize(const std::filesystem::path&, const std::filesystem::path&, const std::filesystem::path&);
}


namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using JuicerAssets::CsvSource;
    fs::path resources, fixtures, scratch;
    thread_local unsigned viewFault = 0;
    thread_local unsigned calibrationFault = 0;
    thread_local bool failCopy = false;
    thread_local bool failLensAcquisition = false;
    thread_local std::vector<CsvSource>* acquisitions = nullptr;
    std::barrier<>* publicationBarrier = nullptr;
    std::atomic<unsigned> publications{0};
    thread_local bool failPublication = false;
    std::atomic<std::size_t> pairCapacity{0};
    std::barrier<>* copyBarrier = nullptr;
    std::barrier<>* calibrationBarrier = nullptr;
    std::array<float, 81> foundationReference{};
    std::mutex rustCandidatesMutex;
    std::vector<std::uintptr_t> rustCandidates;

    FjStringView text(const std::string& value) {
        return {value.empty() ? nullptr : value.data(), value.size()};
    }
    void expect_status(FjStatus status, std::uint32_t category) {
        EXPECT_EQ(status.category, category);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
    }
    struct CurveView {
        std::span<const float> lambda_nm, linear;
    };
    struct HeldIlluminants {
        FjIlluminantsOwner* owner = nullptr;
        std::uintptr_t identity = 0;
        CurveView d65, d55, d50, tungsten, kinoton75P, tungstenKg3, tungstenKg3Lens;
        ~HeldIlluminants() {
            (void)fj_test_illuminants_release(owner, nullptr);
        }
    };
    class IlluminantsHold {
        std::shared_ptr<const HeldIlluminants> value;

    public:
        IlluminantsHold() = default;
        explicit IlluminantsHold(std::shared_ptr<const HeldIlluminants> source) : value(std::move(source)) {}
        const HeldIlluminants* operator->() const {
            return value.get();
        }
        explicit operator bool() const {
            return bool(value);
        }
        void reset() {
            value.reset();
        }
        friend bool operator==(const IlluminantsHold& a, const IlluminantsHold& b) {
            return (!a && !b) || (a && b && a->identity == b->identity);
        }
    };
    class IlluminantCacheProbe {
        std::unique_ptr<JuicerAssets::Library> owned;
        JuicerAssets::Library* native;
        static std::uint32_t before_source(std::uint32_t source, std::uint32_t lensPrepared) noexcept {
            try {
                if (source == FJ_CSV_CANON_24_F28_IS && lensPrepared != 1)
                    return FJ_STATUS_INTERNAL_FAILURE;
                if (source == FJ_CSV_CANON_24_F28_IS && std::exchange(failLensAcquisition, false)) {
                    EXPECT_EQ(lensPrepared, 1u);
                    return FJ_STATUS_ALLOCATION_FAILURE;
                }
                if (acquisitions)
                    acquisitions->push_back(static_cast<CsvSource>(source - FJ_CSV_D65));
                if (source == FJ_CSV_D55 && std::exchange(viewFault, 0) != 0)
                    return FJ_STATUS_INTERNAL_FAILURE;
                return FJ_STATUS_SUCCESS;
            } catch (const std::bad_alloc&) {
                return FJ_STATUS_ALLOCATION_FAILURE;
            } catch (...) {
                return FJ_STATUS_INTERNAL_FAILURE;
            }
        }
        static std::uint32_t before_publication(std::uintptr_t entry, std::size_t bytes) noexcept {
            try {
                if (entry == 0 || bytes < std::size_t{7} * 81u * sizeof(float))
                    return FJ_STATUS_INTERNAL_FAILURE;
                if (std::exchange(failPublication, false))
                    return FJ_STATUS_ALLOCATION_FAILURE;
                {
                    std::lock_guard<std::mutex> lock(rustCandidatesMutex);
                    rustCandidates.push_back(entry);
                }
                publications.fetch_add(1);
                if (publicationBarrier) {
                    publicationBarrier->arrive_and_wait();
                    publicationBarrier->arrive_and_wait();
                }
                return FJ_STATUS_SUCCESS;
            } catch (const std::bad_alloc&) {
                return FJ_STATUS_ALLOCATION_FAILURE;
            } catch (...) {
                return FJ_STATUS_INTERNAL_FAILURE;
            }
        }

    public:
        explicit IlluminantCacheProbe(const fs::path& root) : owned(std::make_unique<JuicerAssets::Library>(root)), native(owned.get()) {}
        explicit IlluminantCacheProbe(JuicerAssets::Library& source) : native(&source) {}
        FjAssets* owner() const {
            return native->borrow_rust_assets();
        }
        FjCacheMetrics metrics() const {
            FjCacheMetrics out{};
            const auto result = fj_test_cache_metrics(owner(), FJ_TEST_CACHE_ILLUMINANTS, &out, nullptr);
            if (result.category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("read actual illuminant owner metrics");
            return out;
        }
        IlluminantsHold acquire() const {
            FjIlluminantsOwner* acquired = nullptr;
            const FjIlluminantCacheHooks hooks{before_source, before_publication};
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            const auto result = fj_test_assets_illuminants_acquire_observed(owner(), &hooks, &acquired, &error);
            if (result.category != FJ_STATUS_SUCCESS) {
                EXPECT_EQ(acquired, nullptr);
                throw JuicerCuda::ExecutionFailure{{result, std::string(bytes.data(), error.length)}};
            }
            std::unique_ptr<FjIlluminantsOwner, decltype([](FjIlluminantsOwner* value) {
                                (void)fj_test_illuminants_release(value, nullptr);
                            })>
                guard(acquired);
            FjIlluminantsView view{};
            if (fj_test_illuminants_view(acquired, &view, nullptr).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("project actual illuminant entry");
            auto snapshot = std::make_shared<HeldIlluminants>();
            snapshot->owner = guard.release();
            snapshot->identity = view.entry_identity;
            static constexpr Spectral::SpectralShape kAxis;
            const auto curve = [](FjFloatSpan samples) {
                return samples.count == 0 ? CurveView{} : CurveView{kAxis.wavelengths, {samples.data, samples.count}};
            };
            snapshot->d65 = curve(view.d65);
            snapshot->d55 = curve(view.d55);
            snapshot->d50 = curve(view.d50);
            snapshot->tungsten = curve(view.tungsten);
            snapshot->kinoton75P = curve(view.k75p);
            snapshot->tungstenKg3 = curve(view.tungsten_kg3);
            snapshot->tungstenKg3Lens = curve(view.tungsten_kg3_lens);
            return IlluminantsHold(std::move(snapshot));
        }
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) const {
            return fj_legacy_assets_release_cached_payloads(owner(), error);
        }
    };
    template <typename Operation>
    void expect_cache_failure(Operation operation, std::uint32_t category) {
        bool failed = false;
        try {
            operation();
        } catch (const JuicerCuda::ExecutionFailure& error) {
            failed = true;
            expect_status(error.failure.status, category);
            EXPECT_FALSE(error.failure.diagnostic.empty());
        }
        EXPECT_TRUE(failed);
    }
    struct Assets {
        FjAssets* owner = nullptr;
        explicit Assets(const fs::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            expect_status(fj_legacy_assets_create(path.view(), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            const auto status = fj_legacy_assets_destroy(owner, nullptr);
            if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0)
                std::abort();
        }
    };
    fs::path empty_root(const char* name) {
        const auto root = scratch / name;
        fs::remove_all(root);
        fs::create_directories(root / "illuminants");
        fs::create_directories(root / "filters");
        return root;
    }
    fs::path full_root(const char* name) {
        const auto root = empty_root(name);
        fs::copy(resources, root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        return root;
    }
    void write(const fs::path& path, const std::string& bytes) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        out << bytes;
        ASSERT_TRUE(out.good());
    }
    std::uint64_t hash_floats(const float* data, std::size_t count) {
        std::uint64_t hash = 14695981039346656037ull;
        for (std::size_t i = 0; i < count; ++i) {
            const auto bits = std::bit_cast<std::uint32_t>(data[i]);
            for (unsigned byte = 0; byte < 4; ++byte) {
                hash ^= (bits >> (8u * byte)) & 0xffu;
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }
    FjNeutralCalibrationResult lookup(const Assets& assets, const std::string& print = "paper", const std::string& illuminant = "light", const std::string& film = "film") {
        FjNeutralCalibrationResult result{};
        expect_status(fj_legacy_neutral_calibration_lookup(assets.owner, text(print), text(illuminant), text(film), &result, nullptr), FJ_STATUS_SUCCESS);
        return result;
    }
    ParamSnapshot controls(bool print = false) {
        ParamSnapshot result;
        result.scanRoute = print ? Spektrafilm::ScanRoute::NegativePrintScan : Spektrafilm::ScanRoute::NegativeDirectScan;
        result.spectralUpsamplingMode = 1;
        result.dirCouplers.active = false;
        result.outputGamutCompressionEnabled = 0;
        return result;
    }

    TEST(IlluminantAbi, AllSourcesRetainExactBitsAcrossAssetsReleaseAndDestruction) {
        struct Expected {
            std::uint32_t tag;
            std::size_t rows;
            std::uint64_t hash;
        };
        const std::array<Expected, 7> expected{{{FJ_CSV_D65, 81, 0xadbdb3d2c250ef10ull}, {FJ_CSV_D55, 81, 0xbc4634b46165cfffull}, {FJ_CSV_D50, 81, 0xf70ca3af96e9a5e8ull}, {FJ_CSV_T, 81, 0x2013ac33b7e2aaecull}, {FJ_CSV_K75P, 81, 0x1adba11b770daaa9ull}, {FJ_CSV_KG3, 146, 0xe040af29a7935a51ull}, {FJ_CSV_CANON_24_F28_IS, 107, 0x74308f00cb53f384ull}}};
        std::array<FjCsvPairs*, 7> held{};
        std::array<FjFloatSpan, 7> views{};
        {
            Assets assets(resources);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expect_status(fj_legacy_csv_acquire(assets.owner, expected[i].tag, &held[i], nullptr), FJ_STATUS_SUCCESS);
                expect_status(fj_legacy_csv_view(held[i], &views[i], nullptr), FJ_STATUS_SUCCESS);
                EXPECT_EQ(views[i].count, expected[i].rows * 2);
                EXPECT_EQ(hash_floats(views[i].data, views[i].count), expected[i].hash);
                FjCsvPairs* warm = nullptr;
                FjFloatSpan warmView{};
                expect_status(fj_legacy_csv_acquire(assets.owner, expected[i].tag, &warm, nullptr), FJ_STATUS_SUCCESS);
                expect_status(fj_legacy_csv_view(warm, &warmView, nullptr), FJ_STATUS_SUCCESS);
                EXPECT_EQ(warmView.data, views[i].data);
                expect_status(fj_legacy_csv_release(warm, nullptr), FJ_STATUS_SUCCESS);
            }
            expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 7u);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(hash_floats(views[i].data, views[i].count), expected[i].hash);
            expect_status(fj_legacy_csv_release(held[i], nullptr), FJ_STATUS_SUCCESS);
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(IlluminantAbi, ClearedOutputsOptionalDiagnosticsAndConsumeOnceRelease) {
        Assets assets(resources);
        FjCsvPairs* owner = nullptr;
        for (auto tag : {0u, 8u, 0xffffffffu}) {
            owner = reinterpret_cast<FjCsvPairs*>(1);
            expect_status(fj_legacy_csv_acquire(assets.owner, tag, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);
        }
        FjFloatSpan view{reinterpret_cast<const float*>(1), 99};
        FjErrorBuffer invalid{nullptr, 1, 99};
        expect_status(fj_legacy_csv_view(nullptr, &view, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(view.data, nullptr);
        EXPECT_EQ(view.count, 0u);
        EXPECT_EQ(invalid.length, 0u);
        expect_status(fj_legacy_csv_acquire(nullptr, FJ_CSV_D65, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D65, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_csv_view(nullptr, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_csv_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        FjErrorBuffer zero{reinterpret_cast<char*>(1), 0, 99};
        expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D65, &owner, &zero), FJ_STATUS_SUCCESS);
        EXPECT_EQ(zero.length, 0u);
        expect_status(fj_legacy_csv_view(owner, &view, &zero), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_csv_release(owner, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(IlluminantAbi, AcceptedCsvRepresentationEmptySuccessAndFailureRetry) {
        const auto root = empty_root("csv-representation");
        write(root / "illuminants/D65.csv", "# comment\n380,1 trailing\n385 1e-999\n0x1p2 4\nbad\n390 2;note\n395;3\n");
        JuicerAssets::AssetBridge bridge(root);
        const auto rows = bridge.copy_csv_pairs(CsvSource::D65);
        ASSERT_EQ(rows.size(), 3u);
        EXPECT_EQ(rows[0], (std::pair<float, float>{380, 1}));
        EXPECT_EQ(rows[1], (std::pair<float, float>{385, 0}));
        EXPECT_EQ(rows[2], (std::pair<float, float>{390, 2}));
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        write(root / "illuminants/D55.csv", "# no rows\n");
        EXPECT_TRUE(bridge.copy_csv_pairs(CsvSource::D55).empty());
        Assets assets(root);
        FjCsvPairs* owner = nullptr;
        expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D55, &owner, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan view{};
        expect_status(fj_legacy_csv_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(view.data, nullptr);
        EXPECT_EQ(view.count, 0u);
        expect_status(fj_legacy_csv_release(owner, nullptr), FJ_STATUS_SUCCESS);
        write(root / "illuminants/D50.csv", std::string(65536, '1'));
        expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D50, &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        write(root / "illuminants/D50.csv", "380,1\n");
        expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D50, &owner, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_csv_release(owner, nullptr), FJ_STATUS_SUCCESS);
    }

    TEST(IlluminantAbi, CapacityPoisonAndPanicTransportClearsOutput) {
        Assets assets(resources);
        for (const auto& test : std::array<std::pair<std::uint32_t, std::uint32_t>, 3>{{{1, FJ_STATUS_INTERNAL_FAILURE}, {2, FJ_STATUS_ALLOCATION_FAILURE}, {3, FJ_STATUS_INTERNAL_FAILURE}}}) {
            fj_test_csv_fault(test.first);
            FjCsvPairs* owner = reinterpret_cast<FjCsvPairs*>(1);
            std::array<char, 8> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 99};
            expect_status(fj_legacy_csv_acquire(assets.owner, FJ_CSV_D65, &owner, &error), test.second);
            EXPECT_EQ(owner, nullptr);
            EXPECT_GT(error.length, 0u);
            EXPECT_EQ(bytes[error.length], '\0');
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        }
    }

    TEST(IlluminantCopy, InvalidSpansAndAllocationUnwindReleaseOwners) {
        JuicerAssets::AssetBridge bridge(resources);
        for (unsigned fault = 1; fault <= 4; ++fault) {
            viewFault = fault;
            EXPECT_THROW(bridge.copy_csv_pairs(CsvSource::D55), JuicerCuda::ExecutionFailure);
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        }
        failCopy = true;
        EXPECT_THROW(bridge.copy_csv_pairs(CsvSource::D55), std::bad_alloc);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        const auto rows = bridge.copy_csv_pairs(CsvSource::D55);
        expect_status(bridge.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(rows.size(), 81u);
        EXPECT_FLOAT_EQ(rows[0].first, 380);
    }

    TEST(CalibrationAbi, SelectedFieldsExactUtf8KeysAndClearedInvalidInputs) {
        const auto root = empty_root("calibration-branches");
        const auto path = root / "filters/neutral_print_filters.json";
        Assets assets(root);
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MISSING_FILE);
        const std::array<std::pair<const char*, std::uint32_t>, 5> malformed{{{"[]", FJ_CALIBRATION_FIELD_ROOT}, {"{\"paper\":[]}", FJ_CALIBRATION_FIELD_PRINT_PROFILE}, {"{\"paper\":{\"light\":[]}}", FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT}, {"{\"paper\":{\"light\":{\"film\":[1,2]}}}", FJ_CALIBRATION_FIELD_CMY_CC}, {"{\"paper\":{\"light\":{\"film\":[1,null,3]}}}", FJ_CALIBRATION_FIELD_CMY_CC}}};
        for (const auto& item : malformed) {
            write(path, item.first);
            expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            const auto result = lookup(assets);
            EXPECT_EQ(result.outcome, FJ_CALIBRATION_MALFORMED);
            EXPECT_EQ(result.field, item.second);
            for (float cc : result.cmy_cc)
                EXPECT_EQ(std::bit_cast<std::uint32_t>(cc), 0u);
        }
        write(path, "\xef\xbb\xbf{\"paper\":{\"light\":{\"film\":[1,2,3],\"unused\":false}},\"bad-unused\":[]}");
        // Cached malformed source/root persists across repair until release.
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MALFORMED);
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_FOUND);
        EXPECT_EQ(lookup(assets).cmy_cc[2], 3.0f);
        EXPECT_EQ(lookup(assets, "Paper").outcome, FJ_CALIBRATION_MISSING_ENTRY);
        EXPECT_EQ(lookup(assets, "").outcome, FJ_CALIBRATION_MISSING_ENTRY);
        EXPECT_EQ(lookup(assets, std::string("paper\0", 6)).outcome, FJ_CALIBRATION_MISSING_ENTRY);
        const std::string badUtf8(1, static_cast<char>(0xff));
        FjNeutralCalibrationResult output{99, 99, {1, 2, 3}};
        expect_status(fj_legacy_neutral_calibration_lookup(assets.owner, text(badUtf8), text("light"), text("film"), &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(output.outcome, 0u);
        EXPECT_EQ(output.field, 0u);
        for (float cc : output.cmy_cc)
            EXPECT_EQ(cc, 0.0f);
        expect_status(fj_legacy_neutral_calibration_lookup(nullptr, {}, {}, {}, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_neutral_calibration_lookup(assets.owner, {}, {}, {}, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const FjStringView badExtent{nullptr, 1};
        expect_status(fj_legacy_neutral_calibration_lookup(assets.owner, badExtent, {}, {}, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
    }

    TEST(CalibrationAbi, ReadAllocationClassifierSurvivesAssetsAndTransportUntilRelease) {
        Assets assets(resources);
        for (std::uint32_t mode = 1; mode <= 7; ++mode) {
            expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            fj_test_calibration_read_fault(mode);
            for (std::size_t capacity : {0u, 1u, 8u, 512u}) {
                std::array<char, 512> bytes{};
                FjErrorBuffer error{bytes.data(), capacity, 99};
                FjNeutralCalibrationResult output{99, 99, {1, 2, 3}};
                const auto status = fj_legacy_neutral_calibration_lookup(assets.owner, text("paper"), text("light"), text("film"), &output, &error);
                expect_status(status, mode <= 4 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_SUCCESS);
                if (mode <= 4) {
                    EXPECT_EQ(output.outcome, 0u);
                    EXPECT_EQ(output.field, 0u);
                    for (float cc : output.cmy_cc)
                        EXPECT_EQ(std::bit_cast<std::uint32_t>(cc), 0u);
                    if (capacity > 1) {
                        EXPECT_GT(error.length, 0u);
                    }
                } else {
                    EXPECT_EQ(output.outcome, mode == 5 ? FJ_CALIBRATION_MALFORMED : FJ_CALIBRATION_MISSING_FILE);
                    EXPECT_EQ(output.field, mode == 5 ? FJ_CALIBRATION_FIELD_RESOURCE_READ : FJ_CALIBRATION_FIELD_NONE);
                }
                if (capacity) {
                    EXPECT_EQ(bytes[error.length], '\0');
                }
            }
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            EXPECT_EQ(fj_test_calibration_probes(), mode <= 3 ? 0u : 1u);
            fj_test_calibration_read_fault(0);
            FjNeutralCalibrationResult output{};
            expect_status(fj_legacy_neutral_calibration_lookup(assets.owner, {}, {}, {}, &output, nullptr), mode <= 4 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MISSING_ENTRY);
        }
    }

    TEST(CalibrationAbi, Bundled160CmyTripletsMatchIndependentCapture) {
        std::ifstream input(resources / "filters/neutral_print_filters.json");
        const auto root = Json::parse(input);
        Assets assets(resources);
        std::vector<float> coefficients;
        for (const auto& print : root.items()) {
            for (const auto& illuminant : print.value().items()) {
                for (const auto& film : illuminant.value().items()) {
                    const auto result = lookup(assets, print.key(), illuminant.key(), film.key());
                    ASSERT_EQ(result.outcome, FJ_CALIBRATION_FOUND);
                    coefficients.insert(coefficients.end(), std::begin(result.cmy_cc), std::end(result.cmy_cc));
                }
            }
        }
        ASSERT_EQ(coefficients.size(), 480u);
        EXPECT_EQ(hash_floats(coefficients.data(), coefficients.size()), 0x3ef43845b118f2f9ull);
    }

    TEST(IlluminantLibrary, NativeHeldViewAllocationFailuresReleaseForeignHolds) {
        for (unsigned allocation = 1; allocation <= 2; ++allocation) {
            JuicerAssets::Library native(resources);
            (void)native.illuminant_filter_curves();
            SpectralAllocationProbe::arm(allocation);
            EXPECT_THROW((void)native.illuminant_filter_curves(), std::bad_alloc);
            EXPECT_GT(SpectralAllocationProbe::bytes(), 0u);
            SpectralAllocationProbe::clear();
            EXPECT_LT(SpectralAllocationProbe::bytes(), std::size_t{7} * 81u * sizeof(float));
            expect_status(native.release_cached_payloads(), FJ_STATUS_SUCCESS);
            FjCacheMetrics metrics{};
            expect_status(fj_test_cache_metrics(native.borrow_rust_assets(), FJ_TEST_CACHE_ILLUMINANTS, &metrics, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(metrics.allocations, 1u);
            EXPECT_EQ(metrics.destructions, 1u);
            EXPECT_EQ(metrics.live_bytes, 0u);
        }
    }

    TEST(IlluminantLibrary, SevenDerivedCurvesMatchAcceptedNativeProducerAndRetainSnapshots) {
        IlluminantCacheProbe library(resources);
        std::vector<CsvSource> order;
        acquisitions = &order;
        auto curves = library.acquire();
        acquisitions = nullptr;
        EXPECT_EQ(order, (std::vector<CsvSource>{CsvSource::D65, CsvSource::D55, CsvSource::D50, CsvSource::T, CsvSource::K75p, CsvSource::Kg3, CsvSource::Canon24F28Is}));
        EXPECT_EQ(pairCapacity.load(), 0u);
#if defined(_WIN32)
        std::ifstream expected(fixtures / "windows.bits");
#else
        std::ifstream expected(fixtures / "linux.bits");
#endif
        ASSERT_TRUE(expected.good());
        std::size_t sampleBytes = 0;
        for (const auto* curve : {&curves->d65, &curves->d55, &curves->d50, &curves->tungsten, &curves->kinoton75P, &curves->tungstenKg3, &curves->tungstenKg3Lens}) {
            ASSERT_EQ(curve->linear.size(), 81u);
            ASSERT_EQ(curve->lambda_nm.size(), 81u);
            sampleBytes += curve->linear.size() * sizeof(float);
            for (const auto* samples : {&curve->lambda_nm, &curve->linear})
                for (float sample : *samples) {
                    std::uint32_t bits = 0;
                    ASSERT_TRUE(static_cast<bool>(expected >> bits));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(sample), bits);
                }
        }
        EXPECT_EQ(sampleBytes, 2268u);
        const auto warm = library.metrics();
        for (unsigned index = 0; index < 100; ++index) {
            const auto held = library.acquire();
            EXPECT_EQ(held, curves);
            EXPECT_EQ(held->d65.linear.data(), curves->d65.linear.data());
        }
        EXPECT_EQ(library.metrics().allocations, warm.allocations);
        EXPECT_EQ(library.metrics().derivations, warm.derivations);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        auto newer = library.acquire();
        EXPECT_NE(newer, curves);
        EXPECT_TRUE(std::equal(newer->tungstenKg3Lens.linear.begin(), newer->tungstenKg3Lens.linear.end(), curves->tungstenKg3Lens.linear.begin(), curves->tungstenKg3Lens.linear.end()));
        const auto before = library.metrics();
        curves.reset();
        EXPECT_EQ(library.metrics().destructions, before.destructions + 1);
        EXPECT_LT(library.metrics().live_bytes, before.live_bytes);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        std::printf("actual Rust retained curve bytes=%llu; logical samples=%zu; canonical axis shared; no retained foreign CSV owner\n", static_cast<unsigned long long>(library.metrics().live_bytes), sampleBytes);
    }

    TEST(IlluminantLibrary, PartialResultsRetryAndLensAcquisitionIsConditional) {
        const auto root = full_root("partial-curves");
        fs::remove(root / "illuminants/D55.csv");
        write(root / "filters/heat_absorbing/schott/KG3.csv", "# empty decoded success\n");
        IlluminantCacheProbe library(root);
        std::vector<CsvSource> order;
        acquisitions = &order;
        const auto partial = library.acquire();
        acquisitions = nullptr;
        EXPECT_EQ(partial->d65.linear.size(), 81u);
        EXPECT_TRUE(partial->d55.linear.empty());
        EXPECT_TRUE(partial->tungstenKg3Lens.linear.empty());
        EXPECT_EQ(order.size(), 6u);
        EXPECT_EQ(order.back(), CsvSource::Kg3);
        fs::copy_file(resources / "illuminants/D55.csv", root / "illuminants/D55.csv");
        const auto later = library.acquire();
        EXPECT_NE(partial, later);
        EXPECT_EQ(later->d55.linear.size(), 81u);
        fs::copy_file(resources / "filters/heat_absorbing/schott/KG3.csv", root / "filters/heat_absorbing/schott/KG3.csv", fs::copy_options::overwrite_existing);
        EXPECT_TRUE(library.acquire()->tungstenKg3.linear.empty());
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(library.acquire()->tungstenKg3Lens.linear.size(), 81u);
    }

    TEST(IlluminantLibrary, ConcurrentColdWinnersAndReleaseDuringBuild) {
        IlluminantCacheProbe library(resources);
        std::barrier barrier(2);
        publicationBarrier = &barrier;
        publications = 0;
        rustCandidates.clear();
        rustCandidates.reserve(2);
        expect_status(fj_test_cache_drop_reentry(library.owner(), FJ_TEST_CACHE_ILLUMINANTS, nullptr), FJ_STATUS_SUCCESS);
        std::array<IlluminantsHold, 2> held;
        std::array<std::thread, 2> workers;
        for (std::size_t i = 0; i < workers.size(); ++i)
            workers[i] = std::thread([&, i] {
                held[i] = library.acquire();
            });
        for (auto& worker : workers)
            worker.join();
        publicationBarrier = nullptr;
        const auto metrics = library.metrics();
        EXPECT_EQ(metrics.reentries, 1u);
        EXPECT_EQ(metrics.reentry_failures, 0u);
        ASSERT_EQ(rustCandidates.size(), 2u);
        EXPECT_NE(rustCandidates[0], rustCandidates[1]);
        EXPECT_EQ(metrics.allocations, 2u);
        EXPECT_EQ(metrics.destructions, 1u);
        EXPECT_GT(metrics.peak_bytes, metrics.live_bytes);
        rustCandidates.clear();
        EXPECT_EQ(held[0], held[1]);
        EXPECT_EQ(publications.load(), 2u);
        std::printf("real Rust candidate high-water=%llu current=%llu; one loser reclaimed and reentered outside cache locks\n", static_cast<unsigned long long>(metrics.peak_bytes), static_cast<unsigned long long>(metrics.live_bytes));
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        std::barrier releaseBarrier(2);
        publicationBarrier = &releaseBarrier;
        std::thread loader([&] {
            held[0] = library.acquire();
        });
        releaseBarrier.arrive_and_wait();
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        releaseBarrier.arrive_and_wait();
        loader.join();
        publicationBarrier = nullptr;
        EXPECT_EQ(library.acquire(), held[0]);
    }

    TEST(IlluminantLibrary, RequestedAllocationReclaimsOldSnapshotsAndPrepublicationFailures) {
        IlluminantCacheProbe library(resources);
        auto old = library.acquire();
        const auto one = library.metrics().live_bytes;
        EXPECT_GE(one, 2268u);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(library.metrics().live_bytes, one);
        auto current = library.acquire();
        EXPECT_EQ(library.metrics().live_bytes, 2 * one);
        old.reset();
        EXPECT_EQ(library.metrics().live_bytes, one);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        expect_status(fj_test_cache_drop_reentry(library.owner(), FJ_TEST_CACHE_ILLUMINANTS, nullptr), FJ_STATUS_SUCCESS);
        current.reset();
        EXPECT_EQ(library.metrics().live_bytes, 0u);
        EXPECT_EQ(library.metrics().reentries, 1u);
        EXPECT_EQ(library.metrics().reentry_failures, 0u);
        failPublication = true;
        expect_cache_failure([&] {
            (void)library.acquire();
        },
                             FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        EXPECT_EQ(library.metrics().live_bytes, 0u);
        const auto recovered = library.acquire();
        EXPECT_EQ(recovered->d65.linear.size(), 81u);
        const auto before = library.metrics();
        SpectralAllocationProbe::arm();
        bool wrapperFailure = false;
        try {
            (void)library.acquire();
        } catch (const std::bad_alloc&) {
            wrapperFailure = true;
        }
        EXPECT_TRUE(wrapperFailure);
        EXPECT_GT(SpectralAllocationProbe::bytes(), 0u);
        EXPECT_EQ(library.metrics().allocations, before.allocations);
        EXPECT_EQ(library.acquire(), recovered);
        std::printf("Rust retained bytes one=%llu overlap=%llu; real final-drop reentry; native wrapper rejection remains separate\n", static_cast<unsigned long long>(one), 2ull * static_cast<unsigned long long>(one));
    }

    TEST(IlluminantNative, LensAcquisitionUnwindConsumesPreparedOwnerAndPreservesSourceFailure) {
        for (bool cleanupFault : {false, true}) {
            // The standalone native mathematical wrapper keeps its original
            // foreign-lens unwind witness, separate from the moved cache owner.
            EXPECT_THROW(([&] {
                             JuicerAssets::AssetBridge bridge(resources);
                             auto kg3 = bridge.csv_rows(CsvSource::Kg3);
                             auto lens = JuicerIlluminant::prepare_lens(kg3, "KG3");
                             ASSERT_TRUE(lens);
                             if (cleanupFault)
                                 expect_status(fj_test_illuminant_arm_fault(FJ_TEST_ILLUMINANT_LENS_RELEASE, 1, 2), FJ_STATUS_SUCCESS);
                             failLensAcquisition = true;
                             (void)bridge.csv_rows(CsvSource::Canon24F28Is);
                         })(),
                         std::bad_alloc);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);
            // Observe the intended destructor export before clearing any slot.
            std::array<char, 128> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjIlluminantLens* empty = nullptr;
            expect_status(fj_legacy_illuminant_lens_release(&empty, &error), FJ_STATUS_SUCCESS);
            IlluminantCacheProbe probe(resources);
            expect_status(fj_test_cache_drop_reentry(probe.owner(), FJ_TEST_CACHE_ILLUMINANTS, nullptr), FJ_STATUS_SUCCESS);
            expect_status(fj_test_cache_clear_failures(probe.owner(), 0, cleanupFault ? 1u : 0u, nullptr), FJ_STATUS_SUCCESS);
            failLensAcquisition = true;
            expect_cache_failure([&] {
                (void)probe.acquire();
            },
                                 FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_EQ(probe.metrics().live_bytes, 0u);
            EXPECT_EQ(probe.metrics().reentries, 1u);
            EXPECT_EQ(probe.metrics().reentry_failures, cleanupFault ? 1u : 0u);
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 0u);
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);
            ASSERT_TRUE(probe.acquire());
        }
    }

    TEST(IlluminantMath, DirectConstraintsAndAsymmetricCoverage) {
        JuicerAssets::AssetBridge bridge(resources);
        auto direct = bridge.copy_csv_pairs(CsvSource::D65);
        const auto check = [](const std::vector<std::pair<float, float>>& pairs) {
            std::vector<float> rows;
            for (const auto& pair : pairs) {
                rows.push_back(pair.first);
                rows.push_back(pair.second);
            }
            std::array<char, 256> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjIlluminant curve{};
            expect_status(fj_legacy_illuminant_from_samples({rows.data(), rows.size()}, &curve, &error), FJ_STATUS_PREPARATION_FAILURE);
        };
        direct[0].first += 0.01f;
        check(direct);
        direct = bridge.copy_csv_pairs(CsvSource::D65);
        direct[0].second = std::numeric_limits<float>::infinity();
        check(direct);
        direct = bridge.copy_csv_pairs(CsvSource::D65);
        direct[0].second += 1;
        check(direct);
        const std::array<float, 4> narrow{380, 1, 780, 1};
        std::array<char, 256> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        FjIlluminant curve{};
        FjIlluminantCoverage coverage{};
        FjIlluminantLens* owner = nullptr;
        expect_status(fj_legacy_illuminant_tungsten_kg3({narrow.data(), narrow.size()}, &curve, &coverage, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(coverage.warnings, 3u);
        expect_status(fj_legacy_illuminant_lens_prepare({narrow.data(), narrow.size()}, &owner, &coverage, &error), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_illuminant_lens_finish(&owner, {narrow.data(), narrow.size()}, &curve, &coverage, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(owner, nullptr);
        expect_status(fj_legacy_illuminant_lens_prepare({}, &owner, &coverage, &error), FJ_STATUS_PREPARATION_FAILURE);
    }

    TEST(IlluminantCopy, BorrowedCopyCompletesAcrossSourceCacheRelease) {
        JuicerAssets::AssetBridge bridge(resources);
        std::barrier barrier(2);
        copyBarrier = &barrier;
        std::vector<std::pair<float, float>> copied;
        std::thread reader([&] {
            copied = bridge.copy_csv_pairs(CsvSource::D65);
        });
        barrier.arrive_and_wait();
        expect_status(bridge.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_csv_live_owners(), 1u);
        barrier.arrive_and_wait();
        reader.join();
        copyBarrier = nullptr;
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        EXPECT_EQ(copied.size(), 81u);
    }

    TEST(CalibrationNative, LookupCompletesAcrossCacheReleaseAndStockKeysOwnRecipeSelection) {
        const auto root = full_root("stock-recipe");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        // Retain catalog keys before changing copied metadata for cold profile decoding.
        ASSERT_TRUE(assets.spektrafilm_profile_catalog().valid);
        const auto load = [](const fs::path& path) {
            std::ifstream file(path);
            return Json::parse(file);
        };
        auto film = load(root / "profiles/kodak_portra_400.json");
        auto print = load(root / "profiles/kodak_portra_endura.json");
        film["info"]["stock"] = "film-stock";
        print["info"]["stock"] = "print-stock";
        write(root / "profiles/kodak_portra_400.json", film.dump());
        write(root / "profiles/kodak_portra_endura.json", print.dump());
        const auto path = root / "filters/neutral_print_filters.json";
        write(path, R"({"print-stock":{"TH-KG3":{"film-stock":[11,22,33],"unused":false}},"kodak_portra_endura":{"TH-KG3":{"kodak_portra_400":[101,202,303]}},"bad-unused":[]})");
        const auto params = controls(true);
        auto product = JuicerTestSupport::prepare_product(params);
        std::array<char, 2048> productText{};
        FjErrorBuffer productError{productText.data(), productText.size(), 0};
        FjPreparedStateView productView{};
        ASSERT_EQ(product.view(productView, productError).category, FJ_STATUS_SUCCESS);
        const auto selectedProfiles = assets.selected_profiles_for_route({params.filmProfileKey, params.printProfileKey, params.scanRoute});
        ASSERT_TRUE(selectedProfiles.valid);
        EXPECT_NE(selectedProfiles.filmSource->profile()->info.stock, params.filmProfileKey);
        EXPECT_NE(selectedProfiles.printSource->profile()->info.stock, params.printProfileKey);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[0], 11);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[1], 22);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[2], 33);
        FjStatus preparedStatus{};
        std::string diagnostic;
        auto preparedState = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), params, preparedStatus, diagnostic);
        expect_status(preparedStatus, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(preparedState) << diagnostic;
        FjPreparedStateView prepared{};
        productError = {productText.data(), productText.size(), 0};
        expect_status(preparedState->view(prepared, productError), FJ_STATUS_SUCCESS);
        EXPECT_EQ(prepared.recipe_hash, productView.recipe_hash);
        EXPECT_EQ(prepared.print_filter_main_cmy_cc[0], 11);
        EXPECT_EQ(prepared.print_filter_main_cmy_cc[1], 22);
        EXPECT_EQ(prepared.print_filter_main_cmy_cc[2], 33);
        std::barrier barrier(2);
        calibrationBarrier = &barrier;
        JuicerAssets::NeutralPrintCalibrationResult selected;
        std::thread reader([&] {
            selected = assets.neutral_print_calibration("print-stock", "TH-KG3", "film-stock");
        });
        barrier.arrive_and_wait();
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        barrier.arrive_and_wait();
        reader.join();
        calibrationBarrier = nullptr;
        EXPECT_EQ(selected.status, JuicerAssets::NeutralPrintCalibrationStatus::Found);
        EXPECT_EQ(selected.cmyCc, (std::array<float, 3>{11, 22, 33}));
        std::optional<PreparedState> candidate;
        for (const std::string& bytes : {std::string("{}"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1e100,2,3]}}})"), std::string(R"({"print-stock":[]})"), std::string("[]"), std::string(R"({"print-stock":{"TH-KG3":[]}})"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1,2]}}})")}) {
            write(path, bytes);
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            productError = {productText.data(), productText.size(), 0};
            const auto result = prepare_state(assets.borrow_rust_assets(), params, candidate, productError);
            const bool built = result.category == FJ_STATUS_SUCCESS;
            if (bytes == "{}") {
                ASSERT_TRUE(built) << std::string(productText.data(), productError.length);
                FjPreparedStateView candidateView{};
                ASSERT_EQ(candidate->view(candidateView, productError).category, FJ_STATUS_SUCCESS);
                EXPECT_EQ(candidateView.print_filter_main_cmy_cc[0], 0);
                EXPECT_EQ(candidateView.print_filter_main_cmy_cc[1], 65);
                EXPECT_EQ(candidateView.print_filter_main_cmy_cc[2], 55);
            } else {
                EXPECT_FALSE(built);
                EXPECT_GT(productError.length, 0u);
            }
            auto cCandidate = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), params, preparedStatus, diagnostic);
            expect_status(preparedStatus, built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(bool(cCandidate), built);
            if (cCandidate) {
                productError = {productText.data(), productText.size(), 0};
                expect_status(cCandidate->view(prepared, productError), FJ_STATUS_SUCCESS);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[0], 0);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[1], 65);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[2], 55);
            }
        }
        fs::remove(path);
        fs::create_directory(path);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        productError = {productText.data(), productText.size(), 0};
        auto malformed = prepare_state(assets.borrow_rust_assets(), params, candidate, productError);
        EXPECT_NE(malformed.category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(std::string(productText.data(), productError.length), "MalformedNeutralPrintCalibration phase=4A field=resource_read");
        fs::remove(path);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        auto recovered = JuicerTestSupport::prepare_product(params);
        productError = {productText.data(), productText.size(), 0};
        ASSERT_EQ(recovered.view(productView, productError).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[1], 65);
        auto missingFile = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), params, preparedStatus, diagnostic);
        expect_status(preparedStatus, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(missingFile) << diagnostic;
        productError = {productText.data(), productText.size(), 0};
        expect_status(missingFile->view(prepared, productError), FJ_STATUS_SUCCESS);
        EXPECT_EQ(prepared.print_filter_main_cmy_cc[1], 65);
    }

    TEST(IlluminantNative, FilmReferenceLensAliasAndPreflashUseRetainedSourceCopies) {
        const auto root = full_root("aliases-preflash");
        std::ifstream file(root / "profiles/kodak_portra_400.json");
        auto film = Json::parse(file);
        film["info"]["reference_illuminant"] = "TH-KG3";
        write(root / "profiles/kodak_portra_400.json", film.dump());
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        auto authored = controls(true);
        authored.printPreflashExposure = 0.1;
        // SF_TEMP_BRIDGE_NATIVE_ILLUMINANT_REFERENCE: retain the original
        // native source-copy witness through I7; this build owns the hook call.
        FocusedRenderStateBuildProduct SF_TEMP_BRIDGE_nativeProduct;
        std::string SF_TEMP_BRIDGE_nativeDiagnostic;
        ASSERT_TRUE(build_print_render_state_product(authored, SF_TEMP_BRIDGE_nativeProduct, SF_TEMP_BRIDGE_nativeDiagnostic))
            << SF_TEMP_BRIDGE_nativeDiagnostic;
        auto product = JuicerTestSupport::prepare_product(authored);
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView productView{};
        ASSERT_EQ(product.view(productView, error).category, FJ_STATUS_SUCCESS);
        const auto curves = assets.illuminant_filter_curves();
        FjIlluminantsOwner* selected = nullptr;
        expect_status(fj_test_assets_illuminants_acquire(assets.borrow_rust_assets(), &selected, nullptr), FJ_STATUS_SUCCESS);
        const auto releaseIlluminants = [](FjIlluminantsOwner* held) {
            (void)fj_test_illuminants_release(held, nullptr);
        };
        std::unique_ptr<FjIlluminantsOwner, decltype(releaseIlluminants)> selectedHold(selected, releaseIlluminants);
        FjIlluminantsView shared{};
        expect_status(fj_test_illuminants_view(selected, &shared, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(curves->entryIdentity, shared.entry_identity);
        EXPECT_EQ(curves->d65.linear.data(), shared.d65.data);
        EXPECT_EQ(curves->tungstenKg3Lens.linear.data(), shared.tungsten_kg3_lens.data);
        EXPECT_TRUE(std::equal(foundationReference.begin(), foundationReference.end(), curves->tungstenKg3Lens.linear.begin()));
        ASSERT_EQ(productView.exposure_tables.illuminant.count, curves->tungstenKg3.linear.size());
        EXPECT_TRUE(std::equal(productView.exposure_tables.illuminant.data, productView.exposure_tables.illuminant.data + productView.exposure_tables.illuminant.count, curves->tungstenKg3.linear.begin()));
        ASSERT_EQ(productView.print.main_illuminant.count, 81u);
        FjStatus result{};
        std::string stateDiagnostic;
        auto state = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(state) << stateDiagnostic;
        FjPreparedStateView prepared{};
        error = {diagnostic.data(), diagnostic.size(), 0};
        expect_status(state->view(prepared, error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(prepared.recipe_hash, productView.recipe_hash);
        EXPECT_EQ(prepared.film_reference.hash, productView.film_reference.hash);

        FjFilmProfile* profileOwner = nullptr;
        expect_status(fj_legacy_film_profile_acquire(assets.borrow_rust_assets(), text("kodak_portra_400"), &profileOwner, &error), FJ_STATUS_SUCCESS);
        const auto releaseProfile = [](FjFilmProfile* held) {
            (void)fj_legacy_film_profile_release(held, nullptr);
        };
        std::unique_ptr<FjFilmProfile, decltype(releaseProfile)> profileHold(profileOwner, releaseProfile);
        FjFilmProfileView profileView{};
        expect_status(fj_legacy_film_profile_view(profileOwner, &profileView, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(profileView.reference_illuminant.kind, FJ_ILLUMINANT_NAMED);
        EXPECT_EQ(std::string(profileView.reference_illuminant.label.data, profileView.reference_illuminant.label.count), "TH-KG3");
        EXPECT_EQ(profileView.tables.linear_sensitivity_rgb.count, 243u);

        FjFilmDevelopmentOwner* developmentOwner = nullptr;
        expect_status(fj_legacy_film_development_acquire(
                          profileOwner,
                          static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativePrintScan),
                          authored.filmGammaFactor,
                          0,
                          &developmentOwner,
                          &error),
                      FJ_STATUS_SUCCESS);
        const auto releaseDevelopment = [](FjFilmDevelopmentOwner* held) {
            (void)fj_legacy_film_development_release(held, nullptr);
        };
        std::unique_ptr<FjFilmDevelopmentOwner, decltype(releaseDevelopment)> developmentHold(developmentOwner, releaseDevelopment);

        ASSERT_EQ(prepared.film_exposure.mallett_basis.count, 243u);
        ASSERT_EQ(prepared.film_exposure.sensitivity_rgb.count, 243u);
        FjFilmReferenceInput referenceInput{};
        referenceInput.method = static_cast<std::uint32_t>(authored.spectralUpsamplingMode);
        referenceInput.auto_exposure_enabled = authored.cameraAutoExposureEnabled != 0 ? 1u : 0u;
        referenceInput.manual_exposure_ev = static_cast<float>(authored.cameraExposureCompensationEv);
        referenceInput.mallett_green_scale = prepared.film_exposure.mallett_green_midgray_scale;
        std::copy_n(productView.input_policy.projection_white_xyz, 3, referenceInput.projection_white_xyz);
        referenceInput.basis_rgb = prepared.film_exposure.mallett_basis;
        referenceInput.film_illuminant = prepared.exposure_tables.illuminant;
        referenceInput.sensitivity_rgb = prepared.film_exposure.sensitivity_rgb;
        FjFilmReference heldReference{};
        FjFilmReferenceFailure referenceFailure{};
        expect_status(fj_test_film_reference_prepare(developmentOwner, &referenceInput, &heldReference, &referenceFailure, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(referenceFailure.reason, FJ_FILM_REFERENCE_REASON_NONE);
        EXPECT_EQ(std::memcmp(heldReference.baseline_raw_rgb, productView.film_reference.baseline_raw_rgb, sizeof(heldReference.baseline_raw_rgb)), 0);
        EXPECT_EQ(std::memcmp(heldReference.compensated_raw_rgb, productView.film_reference.compensated_raw_rgb, sizeof(heldReference.compensated_raw_rgb)), 0);
        EXPECT_EQ(std::memcmp(heldReference.baseline_density_cmy, productView.film_reference.baseline_density_cmy, sizeof(heldReference.baseline_density_cmy)), 0);
        EXPECT_EQ(std::memcmp(heldReference.compensated_density_cmy, productView.film_reference.compensated_density_cmy, sizeof(heldReference.compensated_density_cmy)), 0);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(heldReference.meter_scale), std::bit_cast<std::uint32_t>(productView.film_reference.meter_scale));
        EXPECT_EQ(heldReference.hash, productView.film_reference.hash);

        ASSERT_EQ(prepared.exposure_tables.illuminant.count, curves->tungstenKg3.linear.size());
        EXPECT_TRUE(std::equal(curves->tungstenKg3.linear.begin(), curves->tungstenKg3.linear.end(), prepared.exposure_tables.illuminant.data));

        FjPrintIlluminantParams illuminantParams{};
        const std::array<float, 4> edges{{516.0f, 500.0f, 610.0f, 607.0f}};
        const std::array<float, 4> transitions{{12.0f, 8.0f, 8.0f, 8.0f}};
        std::copy(edges.begin(), edges.end(), illuminantParams.edges_nm);
        std::copy(transitions.begin(), transitions.end(), illuminantParams.transitions_nm);
        std::copy_n(productView.print_filter_preflash_cmy_cc, 3, illuminantParams.cmy_cc);
        FjIlluminant independentPreflashIlluminant{};
        FjPrintIlluminantFailure illuminantFailure{};
        expect_status(fj_test_print_illuminant_prepare(
                          &illuminantParams,
                          productView.print_source_illuminant,
                          &independentPreflashIlluminant,
                          &illuminantFailure,
                          &error),
                      FJ_STATUS_SUCCESS);
        EXPECT_EQ(illuminantFailure.reason, FJ_PRINT_ILLUMINANT_REASON_NONE);
        std::array<float, 81> savedPreflashIlluminant{};
        std::copy_n(independentPreflashIlluminant.samples, savedPreflashIlluminant.size(), savedPreflashIlluminant.begin());
        EXPECT_EQ(std::memcmp(savedPreflashIlluminant.data(), prepared.print.preflash_illuminant.data, sizeof(savedPreflashIlluminant)), 0);

        FjPrintPreflashInput input{
            productView.print.film_base_density,
            {savedPreflashIlluminant.data(), savedPreflashIlluminant.size()},
            productView.print.sensitivity_cmy};
        FjPrintPreflashRaw preflashRaw{};
        FjPrintPreflashFailure preflashFailure{};
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        expect_status(fj_test_print_preflash_prepare(&input, &preflashRaw, &preflashFailure, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(preflashFailure.reason, FJ_PRINT_PREFLASH_REASON_NONE);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        const auto savedPreflashRawCmy = std::array<float, 3>{preflashRaw.cmy[0], preflashRaw.cmy[1], preflashRaw.cmy[2]};
        EXPECT_EQ(std::memcmp(prepared.print.preflash_raw_cmy, preflashRaw.cmy, sizeof(preflashRaw.cmy)), 0);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(std::memcmp(savedPreflashIlluminant.data(), prepared.print.preflash_illuminant.data, sizeof(savedPreflashIlluminant)), 0);
        EXPECT_EQ(savedPreflashRawCmy, (std::array<float, 3>{prepared.print.preflash_raw_cmy[0], prepared.print.preflash_raw_cmy[1], prepared.print.preflash_raw_cmy[2]}));
        expect_status(fj_test_assets_curve_fault(assets.borrow_rust_assets(), FJ_CSV_D55, 1, nullptr), FJ_STATUS_SUCCESS);
        stateDiagnostic.clear();
        auto failed = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_FALSE(failed);
        expect_status(fj_test_assets_curve_fault(assets.borrow_rust_assets(), FJ_CSV_D55, 3, nullptr), FJ_STATUS_SUCCESS);
        stateDiagnostic.clear();
        failed = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_FALSE(failed);
        EXPECT_EQ(std::memcmp(savedPreflashIlluminant.data(), prepared.print.preflash_illuminant.data, sizeof(savedPreflashIlluminant)), 0);
        EXPECT_EQ(savedPreflashRawCmy, (std::array<float, 3>{prepared.print.preflash_raw_cmy[0], prepared.print.preflash_raw_cmy[1], prepared.print.preflash_raw_cmy[2]}));
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(CalibrationNative, ImpossibleTagsAreInternalAndAllocationCannotSelectRecipeFallback) {
        JuicerAssets::AssetBridge bridge(resources);
        for (unsigned fault = 1; fault <= 4; ++fault) {
            calibrationFault = fault;
            EXPECT_THROW(bridge.neutral_print_calibration("missing", "light", "film"), JuicerCuda::ExecutionFailure);
        }
        const auto root = full_root("native-allocation");
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        std::string diagnostic;
        for (unsigned mode = 1; mode <= 4; ++mode) {
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            fj_test_calibration_read_fault(mode);
            std::optional<PreparedState> product;
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            const auto failedProduct = prepare_state(assets.borrow_rust_assets(), controls(true), product, error);
            EXPECT_EQ(failedProduct.category, FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_FALSE(product);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            FjStatus result{};
            auto failed = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), controls(true), result, diagnostic);
            expect_status(result, FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_FALSE(failed);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            error = {errorText.data(), errorText.size(), 0};
            const auto recovered = prepare_state(assets.borrow_rust_assets(), controls(true), product, error);
            expect_status(recovered, FJ_STATUS_SUCCESS);
            ASSERT_TRUE(product);
        }
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        fj_test_calibration_read_fault(1);
        std::optional<PreparedState> product;
        std::array<char, 2048> errorText{};
        FjErrorBuffer error{errorText.data(), errorText.size(), 0};
        const auto directResult = prepare_state(assets.borrow_rust_assets(), controls(), product, error);
        expect_status(directResult, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(product);
        EXPECT_EQ(fj_test_calibration_reads(), 0u);
        FjStatus result{};
        auto direct = JuicerCudaTest::prepare_state(assets.borrow_rust_assets(), controls(), result, diagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(direct) << diagnostic;
        EXPECT_EQ(fj_test_calibration_reads(), 0u);
        fj_test_calibration_read_fault(0);
    }

    TEST(IlluminantNative, UnrelatedAllocationAndInternalFaultAbortColdConstruction) {
        auto params = controls();
        const auto root = full_root("native-curve-faults");
        std::ifstream file(root / "profiles/kodak_portra_400.json");
        auto film = Json::parse(file);
        file.close();
        film["info"]["reference_illuminant"] = "D65";
        write(root / "profiles/kodak_portra_400.json", film.dump());
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        auto& assets = JuicerProcess::root().assets();
        IlluminantCacheProbe probe(assets);
        std::string diagnostic;
        FjStatus result{};
        expect_status(fj_test_assets_curve_fault(probe.owner(), FJ_CSV_D55, 1, nullptr), FJ_STATUS_SUCCESS);
        auto failed = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_FALSE(failed);
        EXPECT_FALSE(diagnostic.empty());
        EXPECT_EQ(probe.metrics().live_bytes, 0u);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        expect_status(fj_test_assets_curve_fault(probe.owner(), FJ_CSV_D55, 3, nullptr), FJ_STATUS_SUCCESS);
        failed = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_FALSE(failed);
        EXPECT_EQ(probe.metrics().live_bytes, 0u);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        for (const unsigned fault : {1u, 3u}) {
            expect_status(fj_test_assets_curve_fault(probe.owner(), FJ_CSV_D55, fault, nullptr), FJ_STATUS_SUCCESS);
            std::optional<PreparedState> rejected;
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            const auto failure = prepare_state(probe.owner(), params, rejected, error);
            EXPECT_EQ(failure.category, fault == 1 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            EXPECT_FALSE(rejected);
            EXPECT_EQ(probe.metrics().live_bytes, 0u);
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        }
        auto state = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(state) << diagnostic;
        FjPreparedStateView before{};
        std::array<char, 2048> beforeText{};
        FjErrorBuffer beforeError{beforeText.data(), beforeText.size(), 0};
        expect_status(state->view(before, beforeError), FJ_STATUS_SUCCESS);
        std::optional<PreparedState> native;
        std::array<char, 2048> nativeText{};
        FjErrorBuffer nativeError{nativeText.data(), nativeText.size(), 0};
        expect_status(prepare_state(probe.owner(), params, native, nativeError), FJ_STATUS_SUCCESS);
        ASSERT_TRUE(native);
        FjPreparedStateView nativeView{};
        expect_status(native->view(nativeView, nativeError), FJ_STATUS_SUCCESS);
        EXPECT_EQ(before.recipe_hash, nativeView.recipe_hash);
        ASSERT_EQ(before.exposure_tables.illuminant.count, nativeView.exposure_tables.illuminant.count);
        EXPECT_TRUE(std::equal(nativeView.exposure_tables.illuminant.data, nativeView.exposure_tables.illuminant.data + nativeView.exposure_tables.illuminant.count, before.exposure_tables.illuminant.data));
        FjInputHullOwner* acquired = nullptr;
        expect_status(fj_test_assets_input_hull_acquire(probe.owner(), &acquired, nullptr), FJ_STATUS_SUCCESS);
        const auto releaseHull = [](FjInputHullOwner* value) {
            (void)fj_test_input_hull_release(value, nullptr);
        };
        std::unique_ptr<FjInputHullOwner, decltype(releaseHull)> hull(acquired, releaseHull);
        FjInputHullView input{};
        expect_status(fj_test_input_hull_view(hull.get(), &input, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(before.input_hull.entry_identity, 0u); // Mallett does not acquire a hull.
        EXPECT_NE(input.entry_identity, 0u);
        expect_status(probe.release_cached_payloads(), FJ_STATUS_SUCCESS);
        expect_status(fj_test_assets_input_hull_acquire(probe.owner(), &acquired, nullptr), FJ_STATUS_SUCCESS);
        std::unique_ptr<FjInputHullOwner, decltype(releaseHull)> warmHull(acquired, releaseHull);
        FjInputHullView warm{};
        expect_status(fj_test_input_hull_view(warmHull.get(), &warm, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(input.entry_identity, warm.entry_identity);
        EXPECT_EQ(input.xy.data, warm.xy.data);
        fs::remove(root / "illuminants/D55.csv");
        expect_status(probe.release_cached_payloads(), FJ_STATUS_SUCCESS);
        auto partial = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(partial) << diagnostic;
        EXPECT_TRUE(probe.acquire()->d55.linear.empty());
        fs::remove(root / "illuminants/D65.csv");
        expect_status(probe.release_cached_payloads(), FJ_STATUS_SUCCESS);
        auto missing = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_FALSE(missing);
        EXPECT_FALSE(diagnostic.empty());
    }
} // namespace

namespace JuicerAssets::IlluminantTest {
    void before_csv_acquisition(CsvSource source) {
        if (source == CsvSource::Canon24F28Is && std::exchange(failLensAcquisition, false)) {
            EXPECT_EQ(fj_test_illuminant_live_lenses(), 1u);
            throw std::bad_alloc();
        }

        if (acquisitions)
            acquisitions->push_back(source);
    }
    void csv_view(CsvSource source, FjFloatSpan& view) {
        if (copyBarrier && source == CsvSource::D65) {
            copyBarrier->arrive_and_wait();
            copyBarrier->arrive_and_wait();
        }
        // Library faults target the unrelated D55 after the selected D65 succeeded.
        if (source != CsvSource::D55)
            return;
        switch (std::exchange(viewFault, 0)) {
            case 1:
                --view.count;
                break;
            case 2:
                view.count = std::numeric_limits<std::size_t>::max();
                break;
            case 3:
                view.data = nullptr;
                break;
            case 4:
                view.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(view.data) + 1);
                break;
            default:
                break;
        }
    }
    void before_csv_copy(CsvSource source, std::size_t) {
        if (source == CsvSource::D55 && std::exchange(failCopy, false))
            throw std::bad_alloc();
    }
    void after_csv_copy(CsvSource, std::size_t, std::size_t capacity) {
        pairCapacity.fetch_add(capacity);
    }
    void film_reference_samples(const std::array<float, 81>& samples) {
        foundationReference = samples;
    }
    void calibration_result(FjNeutralCalibrationResult& result) {
        if (calibrationBarrier) {
            calibrationBarrier->arrive_and_wait();
            calibrationBarrier->arrive_and_wait();
        }
        switch (std::exchange(calibrationFault, 0)) {
            case 1:
                result.outcome = 0;
                break;
            case 2:
                result.field = FJ_CALIBRATION_FIELD_ROOT;
                break;
            case 3:
                result.cmy_cc[0] = 1;
                break;
            case 4:
                result.outcome = FJ_CALIBRATION_MALFORMED;
                result.field = 99;
                break;
            default:
                break;
        }
    }

} // namespace JuicerAssets::IlluminantTest

int main(int argc, char** argv) try {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc != 4)
        return 2;
    resources = argv[1];
    fixtures = argv[2];
    scratch = argv[3];
    fs::create_directories(scratch);
    IlluminantConstruction::initialize(resources, fixtures, scratch);
    return RUN_ALL_TESTS();
} catch (const std::exception& error) {
    std::fprintf(stderr, "illuminant/calibration test setup: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("illuminant/calibration test setup: unknown exception\n", stderr);
    return 1;
}
