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
#include <string>
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
#include "SpectralProcessing.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"

static_assert(sizeof(FjNeutralCalibrationResult) == 20 && alignof(FjNeutralCalibrationResult) == 4);
static_assert(offsetof(FjNeutralCalibrationResult, outcome) == 0 && offsetof(FjNeutralCalibrationResult, field) == 4 && offsetof(FjNeutralCalibrationResult, cmy_cc) == 8);
static_assert(sizeof(FjFloatSpan) == 16 && alignof(FjFloatSpan) == 8);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_acquire), FjStatus (*)(const FjAssets*, uint32_t, FjCsvPairs**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_view), FjStatus (*)(const FjCsvPairs*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_csv_release), FjStatus (*)(FjCsvPairs*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_neutral_calibration_lookup), FjStatus (*)(const FjAssets*, FjStringView, FjStringView, FjStringView, FjNeutralCalibrationResult*, FjErrorBuffer*)>);


namespace CurveAllocationTest {
    struct alignas(std::max_align_t) Header {
        std::size_t bytes;
        bool counted;
    };
    std::atomic<std::size_t> liveBytes{0};
    std::atomic<std::size_t> peakBytes{0};
    std::atomic<unsigned> reentries{0};
    thread_local bool counting = false;
    thread_local JuicerAssets::Library* releaseOnCurveFree = nullptr;
} // namespace CurveAllocationTest
// The two standard libraries spell allocator parameters differently.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(CurveAllocationTest::Header))
        throw std::bad_alloc();
    auto* block = static_cast<CurveAllocationTest::Header*>(std::malloc(sizeof(CurveAllocationTest::Header) + bytes));
    if (!block)
        throw std::bad_alloc();
    block->bytes = bytes;
    block->counted = CurveAllocationTest::counting;
    if (block->counted) {
        const auto live = CurveAllocationTest::liveBytes.fetch_add(bytes) + bytes;
        auto peak = CurveAllocationTest::peakBytes.load();
        while (peak < live && !CurveAllocationTest::peakBytes.compare_exchange_weak(peak, live)) {
        }
    }
    return block + 1;
}
void operator delete(void* pointer) noexcept {
    if (!pointer)
        return;
    auto* block = static_cast<CurveAllocationTest::Header*>(pointer) - 1;
    // The analyzer models standard new without this executable-local prefix.
    if (block->counted) {                                       // NOLINT(clang-analyzer-security.ArrayBound)
        CurveAllocationTest::liveBytes.fetch_sub(block->bytes); // NOLINT(clang-analyzer-security.ArrayBound)
    }
    if (block->bytes == 81 * sizeof(float) && CurveAllocationTest::releaseOnCurveFree) { // NOLINT(clang-analyzer-security.ArrayBound)
        auto* library = std::exchange(CurveAllocationTest::releaseOnCurveFree, nullptr);
        if (library->release_cached_payloads().category != FJ_STATUS_SUCCESS)
            std::abort();
        CurveAllocationTest::reentries.fetch_add(1);
    }
    std::free(block);
}
void* operator new[](std::size_t bytes) {
    return ::operator new(bytes);
}
void operator delete[](void* pointer) noexcept {
    ::operator delete(pointer);
}
#if defined(__cpp_sized_deallocation)
void operator delete(void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}
#endif
// NOLINTEND(readability-inconsistent-declaration-parameter-name)

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using JuicerAssets::CsvSource;
    fs::path resources, fixtures, scratch;
    thread_local unsigned viewFault = 0;
    thread_local unsigned calibrationFault = 0;
    thread_local bool failCopy = false;
    thread_local std::vector<CsvSource>* acquisitions = nullptr;
    std::barrier<>* publicationBarrier = nullptr;
    std::atomic<unsigned> publications{0};
    thread_local bool failPublication = false;
    JuicerAssets::Library* destructionLibrary = nullptr;
    std::mutex candidatesMutex;
    std::vector<std::weak_ptr<const JuicerAssets::IlluminantFilterCurveSet>> candidates;
    std::atomic<std::size_t> pairCapacity{0};
    std::barrier<>* copyBarrier = nullptr;
    std::barrier<>* calibrationBarrier = nullptr;
    std::array<float, 81> foundationReference{};

    FjStringView text(const std::string& value) {
        return {value.empty() ? nullptr : value.data(), value.size()};
    }
    void expect_status(FjStatus status, std::uint32_t category) {
        EXPECT_EQ(status.category, category);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
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

    TEST(IlluminantLibrary, SevenDerivedCurvesMatchAcceptedNativeProducerAndRetainSnapshots) {
        JuicerAssets::Library library(resources, resources.string());
        pairCapacity = 0;
        std::vector<CsvSource> order;
        acquisitions = &order;
        auto curves = library.illuminant_filter_curves();
        acquisitions = nullptr;
        EXPECT_EQ(order, (std::vector<CsvSource>{CsvSource::D65, CsvSource::D55, CsvSource::D50, CsvSource::T, CsvSource::K75p, CsvSource::Kg3, CsvSource::Canon24F28Is}));
        EXPECT_EQ(pairCapacity.load(), 658u);
#if defined(_WIN32)
        std::ifstream expected(fixtures / "windows.bits");
#else
        std::ifstream expected(fixtures / "linux.bits");
#endif
        ASSERT_TRUE(expected.good());
        std::size_t curveBytes = 0;
        for (const auto* curve : {&curves->d65, &curves->d55, &curves->d50, &curves->tungsten, &curves->kinoton75P, &curves->tungstenKg3, &curves->tungstenKg3Lens}) {
            ASSERT_EQ(curve->linear.size(), 81u);
            ASSERT_EQ(curve->lambda_nm.size(), 81u);
            curveBytes += (curve->linear.capacity() + curve->lambda_nm.capacity()) * sizeof(float);
            for (const auto* samples : {&curve->lambda_nm, &curve->linear}) {
                for (float sample : *samples) {
                    std::uint32_t bits = 0;
                    ASSERT_TRUE(static_cast<bool>(expected >> bits));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(sample), bits);
                }
            }
        }
        EXPECT_EQ(curveBytes, 4536u);
        EXPECT_EQ(library.illuminant_filter_curves().get(), curves.get());
        std::weak_ptr<const JuicerAssets::IlluminantFilterCurveSet> old = curves;
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        auto newer = library.illuminant_filter_curves();
        EXPECT_NE(newer.get(), curves.get());
        EXPECT_EQ(newer->tungstenKg3Lens.linear, curves->tungstenKg3Lens.linear);
        EXPECT_FALSE(old.expired());
        curves.reset();
        EXPECT_TRUE(old.expired());
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        std::printf("native pairs capacity: 658 rows/5264 bytes cumulatively; derived curves capacity=%zu per snapshot; retained old+new=%zu; snapshot inline=%zu; no retained CSV handle\n", curveBytes, 2 * curveBytes, sizeof(*newer));
    }

    TEST(IlluminantLibrary, PartialResultsRetryAndLensAcquisitionIsConditional) {
        const auto root = full_root("partial-curves");
        fs::remove(root / "illuminants/D55.csv");
        write(root / "filters/heat_absorbing/schott/KG3.csv", "# empty decoded success\n");
        JuicerAssets::Library library(root, root.string());
        std::vector<CsvSource> order;
        acquisitions = &order;
        const auto partial = library.illuminant_filter_curves();
        acquisitions = nullptr;
        EXPECT_EQ(partial->d65.linear.size(), 81u);
        EXPECT_TRUE(partial->d55.linear.empty());
        EXPECT_TRUE(partial->tungstenKg3Lens.linear.empty());
        EXPECT_EQ(order.size(), 6u);
        EXPECT_EQ(order.back(), CsvSource::Kg3);
        fs::copy_file(resources / "illuminants/D55.csv", root / "illuminants/D55.csv");
        const auto later = library.illuminant_filter_curves();
        EXPECT_NE(partial.get(), later.get());
        EXPECT_EQ(later->d55.linear.size(), 81u);
        // Successful empty source is retained until explicit release.
        fs::copy_file(resources / "filters/heat_absorbing/schott/KG3.csv", root / "filters/heat_absorbing/schott/KG3.csv", fs::copy_options::overwrite_existing);
        EXPECT_TRUE(library.illuminant_filter_curves()->tungstenKg3.linear.empty());
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(library.illuminant_filter_curves()->tungstenKg3Lens.linear.size(), 81u);
    }

    TEST(IlluminantLibrary, ConcurrentColdWinnersAndReleaseDuringBuild) {
        JuicerAssets::Library library(resources, resources.string());
        std::barrier barrier(2);
        publicationBarrier = &barrier;
        publications = 0;
        candidates.clear();
        candidates.reserve(2);
        CurveAllocationTest::peakBytes = 0;
        destructionLibrary = &library;
        CurveAllocationTest::reentries = 0;
        std::array<std::shared_ptr<const JuicerAssets::IlluminantFilterCurveSet>, 2> held;
        std::array<std::thread, 2> workers;
        for (std::size_t i = 0; i < workers.size(); ++i)
            workers[i] = std::thread([&, i] {
                CurveAllocationTest::counting = true;
                held[i] = library.illuminant_filter_curves();
                CurveAllocationTest::counting = false;
            });
        for (auto& worker : workers)
            worker.join();
        publicationBarrier = nullptr;
        destructionLibrary = nullptr;
        EXPECT_EQ(CurveAllocationTest::reentries.load(), 1u);
        ASSERT_EQ(candidates.size(), 2u);
        EXPECT_EQ(static_cast<unsigned>(candidates[0].expired()) + static_cast<unsigned>(candidates[1].expired()), 1u);
        const auto concurrentPeak = CurveAllocationTest::peakBytes.load();
        candidates.clear();
        EXPECT_GT(concurrentPeak, CurveAllocationTest::liveBytes.load());
        std::printf("concurrent native requested high-water=%zu after loser weak cleanup=%zu; complete winner shared; loser curve storage reclaimed outside locks\n", concurrentPeak, CurveAllocationTest::liveBytes.load());
        EXPECT_EQ(held[0].get(), held[1].get());
        EXPECT_EQ(publications.load(), 2u);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        std::barrier releaseBarrier(2);
        publicationBarrier = &releaseBarrier;
        std::thread loader([&] {
            held[0] = library.illuminant_filter_curves();
        });
        releaseBarrier.arrive_and_wait();
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        releaseBarrier.arrive_and_wait();
        loader.join();
        publicationBarrier = nullptr;
        EXPECT_EQ(library.illuminant_filter_curves().get(), held[0].get());
    }

    TEST(IlluminantLibrary, RequestedAllocationReclaimsOldSnapshotsAndPrepublicationFailures) {
        JuicerAssets::Library library(resources, resources.string());
        CurveAllocationTest::counting = true;
        auto old = library.illuminant_filter_curves();
        CurveAllocationTest::counting = false;
        const auto one = CurveAllocationTest::liveBytes.load();
        EXPECT_GT(one, 4536u);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(CurveAllocationTest::liveBytes.load(), one);
        CurveAllocationTest::counting = true;
        auto current = library.illuminant_filter_curves();
        CurveAllocationTest::counting = false;
        EXPECT_EQ(CurveAllocationTest::liveBytes.load(), 2 * one);
        old.reset();
        EXPECT_EQ(CurveAllocationTest::liveBytes.load(), one);
        expect_status(library.release_cached_payloads(), FJ_STATUS_SUCCESS);
        CurveAllocationTest::releaseOnCurveFree = &library;
        current.reset(); // reentrant release during actual curve deletion proves the lock is free.
        EXPECT_EQ(CurveAllocationTest::liveBytes.load(), 0u);
        failPublication = true;
        EXPECT_THROW(library.illuminant_filter_curves(), std::bad_alloc);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        EXPECT_EQ(CurveAllocationTest::liveBytes.load(), 0u);
        EXPECT_EQ(library.illuminant_filter_curves()->d65.linear.size(), 81u);
        std::printf("requested native retained bytes: one=%zu overlap=%zu final=0; excludes allocator header/Rust/RSS; old and loser deletion reentered cache release outside locks\n", one, 2 * one);
    }

    TEST(IlluminantMath, DirectConstraintsAndAsymmetricCoverage) {
        JuicerAssets::AssetBridge bridge(resources);
        auto direct = bridge.copy_csv_pairs(CsvSource::D65);
        direct[0].first += 0.01f;
        EXPECT_TRUE(Spectral::build_illuminant_curve(direct, "axis").linear.empty());
        direct = bridge.copy_csv_pairs(CsvSource::D65);
        direct[0].second = std::numeric_limits<float>::infinity();
        EXPECT_TRUE(Spectral::build_illuminant_curve(direct, "finite").linear.empty());
        direct = bridge.copy_csv_pairs(CsvSource::D65);
        direct[0].second += 1;
        EXPECT_TRUE(Spectral::build_illuminant_curve(direct, "mean").linear.empty());
        const std::vector<std::pair<float, float>> narrow{{380, 1}, {780, 1}};
        EXPECT_TRUE(Spectral::build_tungsten_kg3_curve(narrow, "coverage").linear.empty());
        auto prepared = Spectral::prepare_tungsten_kg3_lens_input(narrow, "coverage-warning");
        ASSERT_TRUE(prepared);
        EXPECT_EQ(Spectral::build_tungsten_kg3_lens_curve(std::move(*prepared), narrow, "lens-warning").linear.size(), 81u);
        EXPECT_FALSE(Spectral::prepare_tungsten_kg3_lens_input({}, "empty"));
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
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        ASSERT_TRUE(build_print_render_state_product(params, product, diagnostic)) << diagnostic;
        const auto selectedProfiles = assets.selected_profiles_for_route({params.filmProfileKey, params.printProfileKey, params.scanRoute});
        ASSERT_TRUE(selectedProfiles.valid);
        EXPECT_NE(selectedProfiles.filmProfile->info.stock, params.filmProfileKey);
        EXPECT_NE(selectedProfiles.printSource->profile()->info.stock, params.printProfileKey);
        const auto& cc = product.recipe.print.filters.mainCmyCc;
        EXPECT_EQ(cc.c, 11);
        EXPECT_EQ(cc.m, 22);
        EXPECT_EQ(cc.y, 33);
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
        for (const std::string& bytes : {std::string("{}"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1e100,2,3]}}})"), std::string(R"({"print-stock":[]})"), std::string("[]"), std::string(R"({"print-stock":{"TH-KG3":[]}})"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1,2]}}})")}) {
            write(path, bytes);
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            const bool built = build_print_render_state_product(params, product, diagnostic);
            if (bytes == "{}") {
                ASSERT_TRUE(built) << diagnostic;
                EXPECT_EQ(product.recipe.print.filters.mainCmyCc.c, 0);
                EXPECT_EQ(product.recipe.print.filters.mainCmyCc.m, 65);
                EXPECT_EQ(product.recipe.print.filters.mainCmyCc.y, 55);
            } else {
                EXPECT_FALSE(built);
                EXPECT_FALSE(diagnostic.empty());
            }
        }
        fs::remove(path);
        fs::create_directory(path);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_FALSE(build_print_render_state_product(params, product, diagnostic));
        EXPECT_EQ(diagnostic, "MalformedNeutralPrintCalibration phase=4A field=resource_read");
        fs::remove(path);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        ASSERT_TRUE(build_print_render_state_product(params, product, diagnostic)) << diagnostic;
        EXPECT_EQ(product.recipe.print.filters.mainCmyCc.m, 65);
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
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        auto authored = controls(true);
        authored.printPreflashExposure = 0.1;
        ASSERT_TRUE(build_print_render_state_product(authored, product, diagnostic)) << diagnostic;
        const auto curves = assets.illuminant_filter_curves();
        EXPECT_TRUE(std::equal(foundationReference.begin(), foundationReference.end(), curves->tungstenKg3Lens.linear.begin()));
        EXPECT_EQ(product.payload.exposureTables.illum, curves->tungstenKg3.linear);
        ASSERT_TRUE(product.payload.printMainIlluminant);
        JuicerCuda::PrintResourceInput input;
        std::array<float, 81> preflash{};
        const JuicerCuda::PrintResourcePreparation request{&product.recipe, &assets, &*product.payload.printMainIlluminant};
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        ASSERT_TRUE(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic)) << diagnostic;
        EXPECT_EQ(input.preflashIlluminant.data(), preflash.data());
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        const auto saved = preflash;
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(preflash, saved);
        failCopy = true;
        EXPECT_THROW(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), std::bad_alloc);
        viewFault = 1;
        EXPECT_THROW(JuicerCuda::build_print_resource_input(request, input, preflash, diagnostic), JuicerCuda::ExecutionFailure);
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
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        for (unsigned mode = 1; mode <= 4; ++mode) {
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            fj_test_calibration_read_fault(mode);
            EXPECT_THROW(build_print_render_state_product(controls(true), product, diagnostic), std::bad_alloc);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
            EXPECT_THROW(build_print_render_state_product(controls(true), product, diagnostic), std::bad_alloc);
            expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
            ASSERT_TRUE(build_print_render_state_product(controls(true), product, diagnostic)) << diagnostic;
        }
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        fj_test_calibration_read_fault(1);
        ASSERT_TRUE(build_direct_render_state_product(controls(), product, diagnostic)) << diagnostic;
        EXPECT_EQ(fj_test_calibration_reads(), 0u);
        fj_test_calibration_read_fault(0);
    }

    TEST(IlluminantNative, UnrelatedAllocationAndInternalFaultAbortColdConstruction) {
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
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        failCopy = true;
        EXPECT_THROW(build_direct_render_state_product(controls(), product, diagnostic), std::bad_alloc);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        viewFault = 1;
        EXPECT_THROW(build_direct_render_state_product(controls(), product, diagnostic), JuicerCuda::ExecutionFailure);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        ASSERT_TRUE(build_direct_render_state_product(controls(), product, diagnostic)) << diagnostic;
        const auto hull = assets.input_compression_hull();
        ASSERT_TRUE(hull);
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_EQ(hull.get(), assets.input_compression_hull().get());
        fs::remove(root / "illuminants/D55.csv");
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        ASSERT_TRUE(build_direct_render_state_product(controls(), product, diagnostic)) << diagnostic;
        EXPECT_TRUE(assets.illuminant_filter_curves()->d55.linear.empty());
        fs::remove(root / "illuminants/D65.csv");
        expect_status(assets.release_cached_payloads(), FJ_STATUS_SUCCESS);
        EXPECT_FALSE(build_direct_render_state_product(controls(), product, diagnostic));
        EXPECT_FALSE(diagnostic.empty());
    }
} // namespace

namespace JuicerAssets::IlluminantTest {
    void before_csv_acquisition(CsvSource source) {
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
    void before_curve_publication(const std::shared_ptr<const IlluminantFilterCurveSet>& candidate) {
        if (std::exchange(failPublication, false))
            throw std::bad_alloc();
        if (destructionLibrary) {
            std::lock_guard<std::mutex> lock(candidatesMutex);
            candidates.emplace_back(candidate);
            CurveAllocationTest::releaseOnCurveFree = destructionLibrary;
        }
        publications.fetch_add(1);
        if (publicationBarrier) {
            publicationBarrier->arrive_and_wait();
            publicationBarrier->arrive_and_wait();
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
    return RUN_ALL_TESTS();
} catch (const std::exception& error) {
    std::fprintf(stderr, "illuminant/calibration test setup: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("illuminant/calibration test setup: unknown exception\n", stderr);
    return 1;
}
