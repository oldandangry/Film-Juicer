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

#include "gtest_cases.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaResources.h"
#include "ProcessRoot.h"
#include "test_asset_library.h"
#include "test_asset_workspace.h"
#include "native_path.h"
#include "test_asset_bridge.h"
#include "ColorTransforms.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "prepared_boundary.h"
#include "prepared_product_fixture.h"

static_assert(sizeof(FjNeutralCalibrationResult) == 20 && alignof(FjNeutralCalibrationResult) == 4);
static_assert(offsetof(FjNeutralCalibrationResult, outcome) == 0 && offsetof(FjNeutralCalibrationResult, field) == 4 && offsetof(FjNeutralCalibrationResult, cmy_cc) == 8);
static_assert(sizeof(FjFloatSpan) == 16 && alignof(FjFloatSpan) == 8);
static_assert(std::is_same_v<decltype(&fj_test_csv_acquire), FjStatus (*)(const FjAssets*, uint32_t, FjCsvPairs**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_csv_view), FjStatus (*)(const FjCsvPairs*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_csv_release), FjStatus (*)(FjCsvPairs*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_neutral_calibration_lookup), FjStatus (*)(const FjAssets*, FjStringView, FjStringView, FjStringView, FjNeutralCalibrationResult*, FjErrorBuffer*)>);


namespace IlluminantConstruction {
    void initialize(const std::filesystem::path&, const std::filesystem::path&, const std::filesystem::path&);
}


namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    fs::path resources, fixtures, scratch;
    thread_local unsigned calibrationFault = 0;
    thread_local bool failLensAcquisition = false;
    thread_local std::vector<std::uint32_t>* acquisitions = nullptr;
    std::barrier<>* publicationBarrier = nullptr;
    std::atomic<unsigned> publications{0};
    thread_local bool failPublication = false;
    std::barrier<>* calibrationBarrier = nullptr;
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
        const FjAssets* borrowed = nullptr;
        static std::uint32_t before_source(std::uint32_t source, std::uint32_t lensPrepared) noexcept {
            try {
                if (source == FJ_CSV_CANON_24_F28_IS && lensPrepared != 1)
                    return FJ_STATUS_INTERNAL_FAILURE;
                if (source == FJ_CSV_CANON_24_F28_IS && std::exchange(failLensAcquisition, false)) {
                    EXPECT_EQ(lensPrepared, 1u);
                    return FJ_STATUS_ALLOCATION_FAILURE;
                }
                if (acquisitions)
                    acquisitions->push_back(source);
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
        explicit IlluminantCacheProbe(const FjAssets* source) : native(nullptr), borrowed(source) {}
        const FjAssets* owner() const {
            return borrowed ? borrowed : native->borrow_rust_assets();
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
            return fj_test_assets_release_cached_payloads(owner(), error);
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
            expect_status(fj_test_assets_create(path.view(), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            const auto status = fj_test_assets_destroy(owner, nullptr);
            if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0)
                std::abort();
        }
    };
    fs::path empty_root(const char* name) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / name);
        fs::create_directories(root / "illuminants");
        fs::create_directories(root / "filters");
        return root;
    }
    fs::path full_root(const char* name) {
        const auto root = empty_root(name);
        JuicerTestSupport::copy_print_preparation_assets(resources, root);
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
        expect_status(fj_test_neutral_calibration_lookup(assets.owner, text(print), text(illuminant), text(film), &result, nullptr), FJ_STATUS_SUCCESS);
        return result;
    }
    JuicerTestSupport::PreparationInput controls(bool print = false) {
        JuicerTestSupport::PreparationInput result;
        result.values.scan_route = print ? FJ_ROUTE_NEGATIVE_PRINT : FJ_ROUTE_NEGATIVE_DIRECT;
        result.values.spectral_upsampling_mode = 1;
        result.values.dir_couplers.active = false;
        result.values.output_gamut_compression_enabled = 0;
        return result;
    }

    TEST(IlluminantAbi, AllSourcesRetainExactBitsAcrossAssetsReleaseAndDestruction) {
        struct Expected {
            std::uint32_t tag;
            std::size_t rows;
            std::uint64_t hash;
        };
        const std::array<Expected, 7> expected{{{FJ_CSV_D65, 81, 0xadbdb3d2c250ef10ull}, {FJ_CSV_D55, 81, 0xbc4634b46165cfffull}, {FJ_CSV_D50, 81, 0xf70ca3af96e9a5e8ull}, {FJ_CSV_T, 81, 0x2013ac33b7e2aaecull}, {FJ_CSV_K75P, 81, 0x1adba11b770daaa9ull}, {FJ_CSV_KG3, 146, 0xe040af29a7935a51ull}, {FJ_CSV_CANON_24_F28_IS, 107, 0x74308f00cb53f384ull}}};
        std::vector<std::string> source_ids;
        source_ids.reserve(expected.size());
        for (const auto& source : expected)
            source_ids.push_back("csv-tag/" + std::to_string(source.tag));
        JuicerTests::GtestCohort source_cases("csv-source-resources", source_ids);
        std::vector<std::unique_ptr<JuicerTests::GtestCase>> source_rows;
        source_rows.reserve(source_ids.size());
        for (const auto& id : source_ids)
            source_rows.push_back(source_cases.deferred_case(id));
        std::array<FjCsvPairs*, 7> held{};
        std::array<FjFloatSpan, 7> views{};
        {
            Assets assets(resources);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expect_status(fj_test_csv_acquire(assets.owner, expected[i].tag, &held[i], nullptr), FJ_STATUS_SUCCESS);
                expect_status(fj_test_csv_view(held[i], &views[i], nullptr), FJ_STATUS_SUCCESS);
                EXPECT_EQ(views[i].count, expected[i].rows * 2);
                EXPECT_EQ(hash_floats(views[i].data, views[i].count), expected[i].hash);
                FjCsvPairs* warm = nullptr;
                FjFloatSpan warmView{};
                expect_status(fj_test_csv_acquire(assets.owner, expected[i].tag, &warm, nullptr), FJ_STATUS_SUCCESS);
                expect_status(fj_test_csv_view(warm, &warmView, nullptr), FJ_STATUS_SUCCESS);
                EXPECT_EQ(warmView.data, views[i].data);
                expect_status(fj_test_csv_release(warm, nullptr), FJ_STATUS_SUCCESS);
            }
            expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 7u);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(hash_floats(views[i].data, views[i].count), expected[i].hash);
            expect_status(fj_test_csv_release(held[i], nullptr), FJ_STATUS_SUCCESS);
            source_rows[i]->passed();
        }
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(IlluminantAbi, ClearedOutputsOptionalDiagnosticsAndConsumeOnceRelease) {
        Assets assets(resources);
        FjCsvPairs* owner = nullptr;
        JuicerTests::GtestCohort invalid_source_tag_cases("invalid-source-tags", 3u);
        std::size_t invalid_source_tag_index = 0;
        for (auto tag : {0u, 8u, 0xffffffffu}) {
            auto invalid_source_tag_case = invalid_source_tag_cases.case_event(std::to_string(invalid_source_tag_index++));
            owner = reinterpret_cast<FjCsvPairs*>(1);
            expect_status(fj_test_csv_acquire(assets.owner, tag, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(owner, nullptr);

            invalid_source_tag_case.passed();
        }
        FjFloatSpan view{reinterpret_cast<const float*>(1), 99};
        FjErrorBuffer invalid{nullptr, 1, 99};
        expect_status(fj_test_csv_view(nullptr, &view, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(view.data, nullptr);
        EXPECT_EQ(view.count, 0u);
        EXPECT_EQ(invalid.length, 0u);
        expect_status(fj_test_csv_acquire(nullptr, FJ_CSV_D65, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D65, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_csv_view(nullptr, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_csv_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        FjErrorBuffer zero{reinterpret_cast<char*>(1), 0, 99};
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D65, &owner, &zero), FJ_STATUS_SUCCESS);
        EXPECT_EQ(zero.length, 0u);
        expect_status(fj_test_csv_view(owner, &view, &zero), FJ_STATUS_SUCCESS);
        expect_status(fj_test_csv_release(owner, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(IlluminantAbi, AcceptedCsvRepresentationEmptySuccessAndFailureRetry) {
        const auto root = empty_root("csv-representation");
        write(root / "illuminants/D65.csv", "# comment\n380,1 trailing\n385 1e-999\n0x1p2 4\nbad\n390 2;note\n395;3\n");
        Assets assets(root);
        FjCsvPairs* owner = nullptr;
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D65, &owner, nullptr), FJ_STATUS_SUCCESS);
        const auto releaseCsv = [](FjCsvPairs* held) {
            expect_status(fj_test_csv_release(held, nullptr), FJ_STATUS_SUCCESS);
        };
        std::unique_ptr<FjCsvPairs, decltype(releaseCsv)> rowsHold(owner, releaseCsv);
        FjFloatSpan rows{};
        expect_status(fj_test_csv_view(rowsHold.get(), &rows, nullptr), FJ_STATUS_SUCCESS);
        ASSERT_EQ(rows.count, 6u);
        EXPECT_EQ((std::array{rows.data[0], rows.data[1]}), (std::array<float, 2>{380, 1}));
        EXPECT_EQ((std::array{rows.data[2], rows.data[3]}), (std::array<float, 2>{385, 0}));
        EXPECT_EQ((std::array{rows.data[4], rows.data[5]}), (std::array<float, 2>{390, 2}));
        rowsHold.reset();
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        write(root / "illuminants/D55.csv", "# no rows\n");
        owner = nullptr;
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D55, &owner, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan view{};
        expect_status(fj_test_csv_view(owner, &view, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(view.data, nullptr);
        EXPECT_EQ(view.count, 0u);
        expect_status(fj_test_csv_release(owner, nullptr), FJ_STATUS_SUCCESS);
        write(root / "illuminants/D50.csv", std::string(65536, '1'));
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D50, &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        write(root / "illuminants/D50.csv", "380,1\n");
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D50, &owner, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_csv_release(owner, nullptr), FJ_STATUS_SUCCESS);
    }

    TEST(IlluminantAbi, CapacityPoisonAndPanicTransportClearsOutput) {
        Assets assets(resources);
        JuicerTests::GtestCohort transport_fault_cases("transport-faults", 3u);
        std::size_t transport_fault_index = 0;
        for (const auto& test : std::array<std::pair<std::uint32_t, std::uint32_t>, 3>{{{1, FJ_STATUS_INTERNAL_FAILURE}, {2, FJ_STATUS_ALLOCATION_FAILURE}, {3, FJ_STATUS_INTERNAL_FAILURE}}}) {
            auto transport_fault_case = transport_fault_cases.case_event(std::to_string(transport_fault_index++));
            fj_test_csv_fault(test.first);
            FjCsvPairs* owner = reinterpret_cast<FjCsvPairs*>(1);
            std::array<char, 8> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 99};
            expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D65, &owner, &error), test.second);
            EXPECT_EQ(owner, nullptr);
            EXPECT_GT(error.length, 0u);
            EXPECT_EQ(bytes[error.length], '\0');
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);

            transport_fault_case.passed();
        }
    }

    TEST(CalibrationAbi, SelectedFieldsExactUtf8KeysAndClearedInvalidInputs) {
        const auto root = empty_root("calibration-branches");
        const auto path = root / "filters/neutral_print_filters.json";
        Assets assets(root);
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MISSING_FILE);
        const std::array<std::pair<const char*, std::uint32_t>, 5> malformed{{{"[]", FJ_CALIBRATION_FIELD_ROOT}, {"{\"paper\":[]}", FJ_CALIBRATION_FIELD_PRINT_PROFILE}, {"{\"paper\":{\"light\":[]}}", FJ_CALIBRATION_FIELD_PRINT_ILLUMINANT}, {"{\"paper\":{\"light\":{\"film\":[1,2]}}}", FJ_CALIBRATION_FIELD_CMY_CC}, {"{\"paper\":{\"light\":{\"film\":[1,null,3]}}}", FJ_CALIBRATION_FIELD_CMY_CC}}};
        JuicerTests::GtestCohort malformed_key_cases("malformed-keys", std::size(malformed));
        std::size_t malformed_key_index = 0;
        for (const auto& item : malformed) {
            auto malformed_key_case = malformed_key_cases.case_event(std::to_string(malformed_key_index++));
            write(path, item.first);
            expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            const auto result = lookup(assets);
            EXPECT_EQ(result.outcome, FJ_CALIBRATION_MALFORMED);
            EXPECT_EQ(result.field, item.second);
            for (float cc : result.cmy_cc)
                EXPECT_EQ(std::bit_cast<std::uint32_t>(cc), 0u);

            malformed_key_case.passed();
        }
        write(path, "\xef\xbb\xbf{\"paper\":{\"light\":{\"film\":[1,2,3],\"unused\":false}},\"bad-unused\":[]}");
        // Cached malformed source/root persists across repair until release.
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MALFORMED);
        expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_FOUND);
        EXPECT_EQ(lookup(assets).cmy_cc[2], 3.0f);
        EXPECT_EQ(lookup(assets, "Paper").outcome, FJ_CALIBRATION_MISSING_ENTRY);
        EXPECT_EQ(lookup(assets, "").outcome, FJ_CALIBRATION_MISSING_ENTRY);
        EXPECT_EQ(lookup(assets, std::string("paper\0", 6)).outcome, FJ_CALIBRATION_MISSING_ENTRY);
        const std::string badUtf8(1, static_cast<char>(0xff));
        FjNeutralCalibrationResult output{99, 99, {1, 2, 3}};
        expect_status(fj_test_neutral_calibration_lookup(assets.owner, text(badUtf8), text("light"), text("film"), &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(output.outcome, 0u);
        EXPECT_EQ(output.field, 0u);
        for (float cc : output.cmy_cc)
            EXPECT_EQ(cc, 0.0f);
        expect_status(fj_test_neutral_calibration_lookup(nullptr, {}, {}, {}, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_neutral_calibration_lookup(assets.owner, {}, {}, {}, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const FjStringView badExtent{nullptr, 1};
        expect_status(fj_test_neutral_calibration_lookup(assets.owner, badExtent, {}, {}, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
    }

    TEST(CalibrationAbi, ReadAllocationClassifierSurvivesAssetsAndTransportUntilRelease) {
        Assets assets(resources);
        constexpr std::array<std::size_t, 4> capacities{0u, 1u, 8u, 512u};
        std::vector<std::string> capacity_ids;
        std::vector<std::string> mode_ids;
        for (std::uint32_t mode = 1; mode <= 7; ++mode) {
            mode_ids.push_back("mode/" + std::to_string(mode));
            for (const auto capacity : capacities)
                capacity_ids.push_back("mode/" + std::to_string(mode) + "/capacity/" + std::to_string(capacity));
        }
        JuicerTests::GtestCohort read_fault_capacity_cases("read-fault-capacities", capacity_ids);
        JuicerTests::GtestCohort read_fault_mode_cases("read-fault-modes", mode_ids);
        for (std::uint32_t mode = 1; mode <= 7; ++mode) {
            auto mode_case = read_fault_mode_cases.case_event("mode/" + std::to_string(mode));
            std::vector<std::unique_ptr<JuicerTests::GtestCase>> capacity_cases;
            capacity_cases.reserve(capacities.size());
            for (const auto capacity : capacities)
                capacity_cases.push_back(read_fault_capacity_cases.deferred_case("mode/" + std::to_string(mode) + "/capacity/" + std::to_string(capacity)));
            expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            std::array<bool, capacities.size()> visited{};
            fj_test_calibration_read_fault(mode);
            for (std::size_t index = 0; index < capacities.size(); ++index) {
                const auto capacity = capacities[index];
                std::array<char, 512> bytes{};
                FjErrorBuffer error{bytes.data(), capacity, 99};
                FjNeutralCalibrationResult output{99, 99, {1, 2, 3}};
                const auto status = fj_test_neutral_calibration_lookup(assets.owner, text("paper"), text("light"), text("film"), &output, &error);
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
                visited[index] = true;
            }
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            EXPECT_EQ(fj_test_calibration_probes(), mode <= 3 ? 0u : 1u);
            fj_test_calibration_read_fault(0);
            FjNeutralCalibrationResult output{};
            expect_status(fj_test_neutral_calibration_lookup(assets.owner, {}, {}, {}, &output, nullptr), mode <= 4 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
            EXPECT_EQ(lookup(assets).outcome, FJ_CALIBRATION_MISSING_ENTRY);
            for (std::size_t index = 0; index < capacity_cases.size(); ++index)
                if (visited[index])
                    capacity_cases[index]->passed();
            mode_case.passed();
        }
    }

    TEST(CalibrationAbi, Bundled160CmyTripletsMatchIndependentCapture) {
        std::ifstream input(resources / "filters/neutral_print_filters.json");
        const auto root = Json::parse(input);
        std::vector<std::string> expected_ids;
        for (const auto& print : root.items())
            for (const auto& illuminant : print.value().items())
                for (const auto& film : illuminant.value().items())
                    expected_ids.push_back(print.key() + "/" + illuminant.key() + "/" + film.key());
        JuicerTests::GtestCohort calibration_rows("calibrations", expected_ids);
        std::vector<std::unique_ptr<JuicerTests::GtestCase>> calibration_events;
        Assets assets(resources);
        std::vector<float> coefficients;
        for (const auto& print : root.items()) {
            for (const auto& illuminant : print.value().items()) {
                for (const auto& film : illuminant.value().items()) {
                    calibration_events.push_back(calibration_rows.deferred_case(print.key() + "/" + illuminant.key() + "/" + film.key()));
                    const auto result = lookup(assets, print.key(), illuminant.key(), film.key());
                    ASSERT_EQ(result.outcome, FJ_CALIBRATION_FOUND);
                    coefficients.insert(coefficients.end(), std::begin(result.cmy_cc), std::end(result.cmy_cc));
                }
            }
        }
        ASSERT_EQ(coefficients.size(), 480u);
        EXPECT_EQ(hash_floats(coefficients.data(), coefficients.size()), 0x3ef43845b118f2f9ull);
        for (const auto& row : calibration_events)
            row->passed();
    }

    TEST(IlluminantLibrary, SevenDerivedCurvesMatchAcceptedNativeProducerAndRetainSnapshots) {
        IlluminantCacheProbe library(resources);
        std::vector<std::uint32_t> order;
        acquisitions = &order;
        auto curves = library.acquire();
        acquisitions = nullptr;
        EXPECT_EQ(order, (std::vector<std::uint32_t>{FJ_CSV_D65, FJ_CSV_D55, FJ_CSV_D50, FJ_CSV_T, FJ_CSV_K75P, FJ_CSV_KG3, FJ_CSV_CANON_24_F28_IS}));
#if defined(_WIN32)
        std::ifstream expected(fixtures / "windows.bits");
#else
        std::ifstream expected(fixtures / "linux.bits");
#endif
        ASSERT_TRUE(expected.good());
        std::size_t sampleBytes = 0;
        JuicerTests::GtestCohort derived_curve_cases("derived-curves", 7u);
        std::size_t derived_curve_index = 0;
        for (const auto* curve : {&curves->d65, &curves->d55, &curves->d50, &curves->tungsten, &curves->kinoton75P, &curves->tungstenKg3, &curves->tungstenKg3Lens}) {
            auto derived_curve_case = derived_curve_cases.case_event(std::to_string(derived_curve_index++));
            ASSERT_EQ(curve->linear.size(), 81u);
            ASSERT_EQ(curve->lambda_nm.size(), 81u);
            sampleBytes += curve->linear.size() * sizeof(float);
            for (const auto* samples : {&curve->lambda_nm, &curve->linear})
                for (float sample : *samples) {
                    std::uint32_t bits = 0;
                    ASSERT_TRUE(static_cast<bool>(expected >> bits));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(sample), bits);
                }

            derived_curve_case.passed();
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
        std::vector<std::uint32_t> order;
        acquisitions = &order;
        const auto partial = library.acquire();
        acquisitions = nullptr;
        EXPECT_EQ(partial->d65.linear.size(), 81u);
        EXPECT_TRUE(partial->d55.linear.empty());
        EXPECT_TRUE(partial->tungstenKg3Lens.linear.empty());
        EXPECT_EQ(order.size(), 6u);
        EXPECT_EQ(order.back(), FJ_CSV_KG3);
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
        EXPECT_EQ(library.acquire(), recovered);
        EXPECT_EQ(library.metrics().allocations, before.allocations);
        std::printf("Rust retained bytes one=%llu overlap=%llu; real final-drop reentry\n", static_cast<unsigned long long>(one), 2ull * static_cast<unsigned long long>(one));
    }

    TEST(IlluminantNative, LensAcquisitionUnwindConsumesPreparedOwnerAndPreservesSourceFailure) {
        JuicerTests::GtestCohort cleanup_fault_cases("cleanup-faults", 2u);
        std::size_t cleanup_fault_index = 0;
        for (bool cleanupFault : {false, true}) {
            auto cleanup_fault_case = cleanup_fault_cases.case_event(std::to_string(cleanup_fault_index++));
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

            cleanup_fault_case.passed();
        }
    }

    TEST(IlluminantMath, DirectConstraintsAndAsymmetricCoverage) {
        Assets assets(resources);
        FjCsvPairs* acquired = nullptr;
        expect_status(fj_test_csv_acquire(assets.owner, FJ_CSV_D65, &acquired, nullptr), FJ_STATUS_SUCCESS);
        const auto releaseCsv = [](FjCsvPairs* held) {
            (void)fj_test_csv_release(held, nullptr);
        };
        std::unique_ptr<FjCsvPairs, decltype(releaseCsv)> source(acquired, releaseCsv);
        FjFloatSpan original{};
        expect_status(fj_test_csv_view(source.get(), &original, nullptr), FJ_STATUS_SUCCESS);
        ASSERT_EQ(original.count, 162u);
        auto direct = std::vector<float>(original.data, original.data + original.count);
        const auto check = [](const std::vector<float>& rows) {
            std::array<char, 256> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjIlluminant curve{};
            expect_status(fj_test_illuminant_from_samples({rows.data(), rows.size()}, &curve, &error), FJ_STATUS_PREPARATION_FAILURE);
        };
        direct[0] += 0.01f;
        check(direct);
        direct.assign(original.data, original.data + original.count);
        direct[1] = std::numeric_limits<float>::infinity();
        check(direct);
        direct.assign(original.data, original.data + original.count);
        direct[1] += 1;
        check(direct);
        const std::array<float, 4> narrow{380, 1, 780, 1};
        std::array<char, 256> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
        FjIlluminant curve{};
        FjIlluminantCoverage coverage{};
        FjIlluminantLens* owner = nullptr;
        expect_status(fj_test_illuminant_tungsten_kg3({narrow.data(), narrow.size()}, &curve, &coverage, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(coverage.warnings, 3u);
        expect_status(fj_test_illuminant_lens_prepare({narrow.data(), narrow.size()}, &owner, &coverage, &error), FJ_STATUS_SUCCESS);
        expect_status(fj_test_illuminant_lens_finish(&owner, {narrow.data(), narrow.size()}, &curve, &coverage, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(owner, nullptr);
        expect_status(fj_test_illuminant_lens_prepare({}, &owner, &coverage, &error), FJ_STATUS_PREPARATION_FAILURE);
    }

    TEST(CalibrationNative, LookupCompletesAcrossCacheReleaseAndStockKeysOwnRecipeSelection) {
        const auto root = full_root("stock-recipe");
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        // Retain catalog keys before changing stock metadata for cold profile decoding.
        FjCatalog* rawCatalog = nullptr;
        FjCatalogCounts catalogCounts{};
        ASSERT_EQ(fj_test_catalog_acquire(assets, &rawCatalog, &catalogCounts, nullptr).category, FJ_STATUS_SUCCESS);
        const auto releaseCatalog = [](FjCatalog* catalog) {
            if (catalog)
                (void)fj_test_catalog_release(catalog, nullptr);
        };
        std::unique_ptr<FjCatalog, decltype(releaseCatalog)> catalog(rawCatalog, releaseCatalog);
        ASSERT_GT(catalogCounts.film_count, 0u);
        ASSERT_GT(catalogCounts.print_count, 0u);
        bool retainedFilmKey = false;
        bool retainedPrintKey = false;
        for (const auto& [role, count] : {std::pair{FJ_PROFILE_ROLE_FILM, catalogCounts.film_count},
                                          std::pair{FJ_PROFILE_ROLE_PRINT, catalogCounts.print_count}}) {
            for (std::size_t i = 0; i < count; ++i) {
                FjCatalogEntryView entry{};
                ASSERT_EQ(fj_test_catalog_entry(catalog.get(), role, i, &entry, nullptr).category, FJ_STATUS_SUCCESS);
                const std::string key(entry.key.data, entry.key.count);
                retainedFilmKey = retainedFilmKey || (role == FJ_PROFILE_ROLE_FILM && key == "kodak_portra_400");
                retainedPrintKey = retainedPrintKey || (role == FJ_PROFILE_ROLE_PRINT && key == "kodak_portra_endura");
            }
        }
        ASSERT_TRUE(retainedFilmKey);
        ASSERT_TRUE(retainedPrintKey);
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
        auto product = JuicerTestSupport::prepare_product(assets, params);
        std::array<char, 2048> productText{};
        FjErrorBuffer productError{productText.data(), productText.size(), 0};
        FjPreparedStateView productView{};
        ASSERT_EQ(product.view(productView, productError).category, FJ_STATUS_SUCCESS);

        EXPECT_EQ(productView.print_filter_main_cmy_cc[0], 11);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[1], 22);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[2], 33);
        FjStatus preparedStatus{};
        std::string diagnostic;
        auto preparedState = JuicerCudaTest::prepare_state(assets, params, preparedStatus, diagnostic);
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
        const auto lookup = [&] {
            const auto text = [](const std::string& value) -> FjStringView {
                return {value.empty() ? nullptr : value.data(), value.size()};
            };
            FjNeutralCalibrationResult raw{};
            std::array<char, 512> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto result = fj_test_neutral_calibration_lookup(
                assets, text("print-stock"), text("TH-KG3"), text("film-stock"), &raw, &error);
            expect_status(result, FJ_STATUS_SUCCESS);
            JuicerAssets::IlluminantTest::calibration_result(raw);
            JuicerAssets::NeutralPrintCalibrationResult value;
            if (raw.outcome == FJ_CALIBRATION_FOUND) {
                EXPECT_EQ(raw.field, FJ_CALIBRATION_FIELD_NONE);
                value.status = JuicerAssets::NeutralPrintCalibrationStatus::Found;
                std::copy(std::begin(raw.cmy_cc), std::end(raw.cmy_cc), value.cmyCc.begin());
            }
            return value;
        };
        std::thread reader([&] {
            selected = lookup();
        });
        barrier.arrive_and_wait();
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        barrier.arrive_and_wait();
        reader.join();
        calibrationBarrier = nullptr;
        EXPECT_EQ(selected.status, JuicerAssets::NeutralPrintCalibrationStatus::Found);
        EXPECT_EQ(selected.cmyCc, (std::array<float, 3>{11, 22, 33}));
        std::optional<JuicerTestSupport::PreparedStateHold> candidate;
        JuicerTests::GtestCohort calibration_shape_cases("calibration-shapes", 6u);
        std::size_t calibration_shape_index = 0;
        for (const std::string& bytes : {std::string("{}"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1e100,2,3]}}})"), std::string(R"({"print-stock":[]})"), std::string("[]"), std::string(R"({"print-stock":{"TH-KG3":[]}})"), std::string(R"({"print-stock":{"TH-KG3":{"film-stock":[1,2]}}})")}) {
            auto calibration_shape_case = calibration_shape_cases.case_event(std::to_string(calibration_shape_index++));
            write(path, bytes);
            expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            productError = {productText.data(), productText.size(), 0};
            const auto result = JuicerTestSupport::prepare_state(assets, params, candidate, productError);
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
            auto cCandidate = JuicerCudaTest::prepare_state(assets, params, preparedStatus, diagnostic);
            expect_status(preparedStatus, built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(bool(cCandidate), built);
            if (cCandidate) {
                productError = {productText.data(), productText.size(), 0};
                expect_status(cCandidate->view(prepared, productError), FJ_STATUS_SUCCESS);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[0], 0);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[1], 65);
                EXPECT_EQ(prepared.print_filter_main_cmy_cc[2], 55);
            }

            calibration_shape_case.passed();
        }
        fs::remove(path);
        fs::create_directory(path);
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        productError = {productText.data(), productText.size(), 0};
        auto malformed = JuicerTestSupport::prepare_state(assets, params, candidate, productError);
        EXPECT_NE(malformed.category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(std::string(productText.data(), productError.length), "MalformedNeutralPrintCalibration phase=4A field=resource_read");
        fs::remove(path);
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        auto recovered = JuicerTestSupport::prepare_product(assets, params);
        productError = {productText.data(), productText.size(), 0};
        ASSERT_EQ(recovered.view(productView, productError).category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(productView.print_filter_main_cmy_cc[1], 65);
        auto missingFile = JuicerCudaTest::prepare_state(assets, params, preparedStatus, diagnostic);
        expect_status(preparedStatus, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(missingFile) << diagnostic;
        productError = {productText.data(), productText.size(), 0};
        expect_status(missingFile->view(prepared, productError), FJ_STATUS_SUCCESS);
        EXPECT_EQ(prepared.print_filter_main_cmy_cc[1], 65);
    }

    TEST(IlluminantNative, FilmReferenceLensAliasAndPreflashRetainOwners) {
        const auto root = full_root("aliases-preflash");
        std::ifstream file(root / "profiles/kodak_portra_400.json");
        auto film = Json::parse(file);
        film["info"]["reference_illuminant"] = "TH-KG3";
        write(root / "profiles/kodak_portra_400.json", film.dump());
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        auto authored = controls(true);
        authored.values.print_preflash_exposure = 0.1;
        auto product = JuicerTestSupport::prepare_product(assets, authored);
        std::array<char, 2048> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
        FjPreparedStateView productView{};
        ASSERT_EQ(product.view(productView, error).category, FJ_STATUS_SUCCESS);
        IlluminantCacheProbe cache(assets);
        const auto curves = cache.acquire();
        FjIlluminantsOwner* selected = nullptr;
        expect_status(fj_test_assets_illuminants_acquire(assets, &selected, nullptr), FJ_STATUS_SUCCESS);
        const auto releaseIlluminants = [](FjIlluminantsOwner* held) {
            (void)fj_test_illuminants_release(held, nullptr);
        };
        std::unique_ptr<FjIlluminantsOwner, decltype(releaseIlluminants)> selectedHold(selected, releaseIlluminants);
        FjIlluminantsView shared{};
        expect_status(fj_test_illuminants_view(selected, &shared, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(curves->identity, shared.entry_identity);
        EXPECT_EQ(curves->d65.linear.data(), shared.d65.data);
        EXPECT_EQ(curves->tungstenKg3Lens.linear.data(), shared.tungsten_kg3_lens.data);
        ASSERT_EQ(productView.exposure_tables.illuminant.count, curves->tungstenKg3.linear.size());
        EXPECT_TRUE(std::equal(productView.exposure_tables.illuminant.data, productView.exposure_tables.illuminant.data + productView.exposure_tables.illuminant.count, curves->tungstenKg3.linear.begin()));
        ASSERT_EQ(productView.print.main_illuminant.count, 81u);
        FjStatus result{};
        std::string stateDiagnostic;
        auto state = JuicerCudaTest::prepare_state(assets, authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(state) << stateDiagnostic;
        FjPreparedStateView prepared{};
        error = {diagnostic.data(), diagnostic.size(), 0};
        expect_status(state->view(prepared, error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(prepared.recipe_hash, productView.recipe_hash);
        EXPECT_EQ(prepared.film_reference.hash, productView.film_reference.hash);

        FjFilmProfile* profileOwner = nullptr;
        expect_status(fj_test_assets_film_acquire(assets, text("kodak_portra_400"), &profileOwner, &error), FJ_STATUS_SUCCESS);
        const auto releaseProfile = [](FjFilmProfile* held) {
            (void)fj_test_assets_film_release(held, nullptr);
        };
        std::unique_ptr<FjFilmProfile, decltype(releaseProfile)> profileHold(profileOwner, releaseProfile);
        FjFilmProfileView profileView{};
        expect_status(fj_test_assets_film_view(profileOwner, &profileView, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(profileView.reference_illuminant.kind, FJ_ILLUMINANT_NAMED);
        EXPECT_EQ(std::string(profileView.reference_illuminant.label.data, profileView.reference_illuminant.label.count), "TH-KG3");
        EXPECT_EQ(profileView.tables.linear_sensitivity_rgb.count, 243u);

        ASSERT_EQ(productView.film_exposure.method, 1u);
        ASSERT_EQ(shared.tungsten_kg3_lens.count, 81u);
        FjSensitivityInput foundationInput{};
        foundationInput.linear_sensitivity_rgb = profileView.tables.linear_sensitivity_rgb;
        foundationInput.reference_illuminant = shared.tungsten_kg3_lens;
        foundationInput.method = productView.film_exposure.method;
        foundationInput.band_pass_active = productView.input_policy.band_pass_active;
        std::copy_n(productView.input_policy.uv, 3, foundationInput.uv);
        std::copy_n(productView.input_policy.ir, 3, foundationInput.ir);
        FjSensitivity foundationSensitivity{};
        FjSensitivityFailure foundationFailure{};
        expect_status(fj_test_exposure_sensitivity(&foundationInput, &foundationSensitivity, &foundationFailure, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(foundationFailure.hash_failure, 0u);
        EXPECT_EQ(foundationFailure.hash_sample_index, 0u);
        ASSERT_EQ(productView.film_exposure.sensitivity_rgb.count, 243u);
        ASSERT_NE(productView.film_exposure.sensitivity_rgb.data, nullptr);
        EXPECT_EQ(std::memcmp(foundationSensitivity.values_rgb, productView.film_exposure.sensitivity_rgb.data, sizeof(foundationSensitivity.values_rgb)), 0);
        EXPECT_EQ(foundationSensitivity.hash, productView.film_exposure.sensitivity_hash);
        // Mallett's recipe scale distinguishes the foundation lens alias from the exposure light.
        EXPECT_EQ(std::bit_cast<std::uint32_t>(foundationSensitivity.mallett_green_scale),
                  std::bit_cast<std::uint32_t>(productView.film_exposure.mallett_green_midgray_scale));

        FjFilmDevelopmentOwner* developmentOwner = nullptr;
        expect_status(fj_test_film_development_acquire(
                          profileOwner,
                          static_cast<std::uint32_t>(Spektrafilm::ScanRoute::NegativePrintScan),
                          authored.values.film_gamma_factor,
                          0,
                          &developmentOwner,
                          &error),
                      FJ_STATUS_SUCCESS);
        const auto releaseDevelopment = [](FjFilmDevelopmentOwner* held) {
            (void)fj_test_film_development_release(held, nullptr);
        };
        std::unique_ptr<FjFilmDevelopmentOwner, decltype(releaseDevelopment)> developmentHold(developmentOwner, releaseDevelopment);

        ASSERT_EQ(prepared.film_exposure.mallett_basis.count, 243u);
        ASSERT_EQ(prepared.film_exposure.sensitivity_rgb.count, 243u);
        FjFilmReferenceInput referenceInput{};
        referenceInput.method = static_cast<std::uint32_t>(authored.values.spectral_upsampling_mode);
        referenceInput.auto_exposure_enabled = authored.values.camera_auto_exposure_enabled != 0 ? 1u : 0u;
        referenceInput.manual_exposure_ev = static_cast<float>(authored.values.camera_exposure_compensation_ev);
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
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_print_preflash_prepare(&input, &preflashRaw, &preflashFailure, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(preflashFailure.reason, FJ_PRINT_PREFLASH_REASON_NONE);
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
        const auto savedPreflashRawCmy = std::array<float, 3>{preflashRaw.cmy[0], preflashRaw.cmy[1], preflashRaw.cmy[2]};
        EXPECT_EQ(std::memcmp(prepared.print.preflash_raw_cmy, preflashRaw.cmy, sizeof(preflashRaw.cmy)), 0);
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(std::memcmp(savedPreflashIlluminant.data(), prepared.print.preflash_illuminant.data, sizeof(savedPreflashIlluminant)), 0);
        EXPECT_EQ(savedPreflashRawCmy, (std::array<float, 3>{prepared.print.preflash_raw_cmy[0], prepared.print.preflash_raw_cmy[1], prepared.print.preflash_raw_cmy[2]}));
        expect_status(fj_test_assets_curve_fault(assets, FJ_CSV_D55, 1, nullptr), FJ_STATUS_SUCCESS);
        stateDiagnostic.clear();
        auto failed = JuicerCudaTest::prepare_state(assets, authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_ALLOCATION_FAILURE);
        EXPECT_FALSE(failed);
        expect_status(fj_test_assets_curve_fault(assets, FJ_CSV_D55, 3, nullptr), FJ_STATUS_SUCCESS);
        stateDiagnostic.clear();
        failed = JuicerCudaTest::prepare_state(assets, authored, result, stateDiagnostic);
        expect_status(result, FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_FALSE(failed);
        EXPECT_EQ(std::memcmp(savedPreflashIlluminant.data(), prepared.print.preflash_illuminant.data, sizeof(savedPreflashIlluminant)), 0);
        EXPECT_EQ(savedPreflashRawCmy, (std::array<float, 3>{prepared.print.preflash_raw_cmy[0], prepared.print.preflash_raw_cmy[1], prepared.print.preflash_raw_cmy[2]}));
        EXPECT_EQ(fj_test_csv_live_owners(), 0u);
    }

    TEST(CalibrationNative, ImpossibleTagsAreInternalAndAllocationCannotSelectRecipeFallback) {
        JuicerAssets::AssetBridge bridge(resources);
        JuicerTests::GtestCohort impossible_tag_cases("impossible-tags", 4u);
        std::size_t impossible_tag_index = 0;
        for (unsigned fault = 1; fault <= 4; ++fault) {
            auto impossible_tag_case = impossible_tag_cases.case_event(std::to_string(impossible_tag_index++));
            calibrationFault = fault;
            EXPECT_THROW(bridge.neutral_print_calibration("missing", "light", "film"), JuicerCuda::ExecutionFailure);

            impossible_tag_case.passed();
        }
        const auto root = full_root("native-allocation");
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        std::string diagnostic;
        JuicerTests::GtestCohort allocation_fallback_cases("allocation-fallback", 4u);
        std::size_t allocation_fallback_index = 0;
        for (unsigned mode = 1; mode <= 4; ++mode) {
            auto allocation_fallback_case = allocation_fallback_cases.case_event(std::to_string(allocation_fallback_index++));
            expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            fj_test_calibration_read_fault(mode);
            std::optional<JuicerTestSupport::PreparedStateHold> product;
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            const auto failedProduct = JuicerTestSupport::prepare_state(assets, controls(true), product, error);
            EXPECT_EQ(failedProduct.category, FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_FALSE(product);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            FjStatus result{};
            auto failed = JuicerCudaTest::prepare_state(assets, controls(true), result, diagnostic);
            expect_status(result, FJ_STATUS_ALLOCATION_FAILURE);
            EXPECT_FALSE(failed);
            EXPECT_EQ(fj_test_calibration_reads(), 1u);
            fj_test_calibration_read_fault(0);
            expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
            error = {errorText.data(), errorText.size(), 0};
            const auto recovered = JuicerTestSupport::prepare_state(assets, controls(true), product, error);
            expect_status(recovered, FJ_STATUS_SUCCESS);
            ASSERT_TRUE(product);

            allocation_fallback_case.passed();
        }
        expect_status(fj_test_assets_release_cached_payloads(assets, nullptr), FJ_STATUS_SUCCESS);
        fj_test_calibration_read_fault(1);
        std::optional<JuicerTestSupport::PreparedStateHold> product;
        std::array<char, 2048> errorText{};
        FjErrorBuffer error{errorText.data(), errorText.size(), 0};
        const auto directResult = JuicerTestSupport::prepare_state(assets, controls(), product, error);
        expect_status(directResult, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(product);
        EXPECT_EQ(fj_test_calibration_reads(), 0u);
        FjStatus result{};
        auto direct = JuicerCudaTest::prepare_state(assets, controls(), result, diagnostic);
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
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
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
        JuicerTests::GtestCohort cold_fault_cases("cold-faults", 2u);
        std::size_t cold_fault_index = 0;
        for (const unsigned fault : {1u, 3u}) {
            auto cold_fault_case = cold_fault_cases.case_event(std::to_string(cold_fault_index++));
            expect_status(fj_test_assets_curve_fault(probe.owner(), FJ_CSV_D55, fault, nullptr), FJ_STATUS_SUCCESS);
            std::optional<JuicerTestSupport::PreparedStateHold> rejected;
            std::array<char, 2048> errorText{};
            FjErrorBuffer error{errorText.data(), errorText.size(), 0};
            const auto failure = JuicerTestSupport::prepare_state(probe.owner(), params, rejected, error);
            EXPECT_EQ(failure.category, fault == 1 ? FJ_STATUS_ALLOCATION_FAILURE : FJ_STATUS_INTERNAL_FAILURE);
            EXPECT_FALSE(rejected);
            EXPECT_EQ(probe.metrics().live_bytes, 0u);
            EXPECT_EQ(fj_test_csv_live_owners(), 0u);

            cold_fault_case.passed();
        }
        auto state = JuicerCudaTest::prepare_state(probe.owner(), params, result, diagnostic);
        expect_status(result, FJ_STATUS_SUCCESS);
        ASSERT_TRUE(state) << diagnostic;
        FjPreparedStateView before{};
        std::array<char, 2048> beforeText{};
        FjErrorBuffer beforeError{beforeText.data(), beforeText.size(), 0};
        expect_status(state->view(before, beforeError), FJ_STATUS_SUCCESS);
        std::optional<JuicerTestSupport::PreparedStateHold> native;
        std::array<char, 2048> nativeText{};
        FjErrorBuffer nativeError{nativeText.data(), nativeText.size(), 0};
        expect_status(JuicerTestSupport::prepare_state(probe.owner(), params, native, nativeError), FJ_STATUS_SUCCESS);
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
    if (::testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    JuicerTests::CaseLog case_log;
    if (argc != 4)
        return 2;
    resources = argv[1];
    fixtures = argv[2];
    scratch = argv[3];
    fs::create_directories(scratch);
    IlluminantConstruction::initialize(resources, fixtures, scratch);
    const int result = RUN_ALL_TESTS();
    case_log.finish();
    const bool ownersDrained = fj_test_illuminant_live_lenses() == 0 && fj_test_csv_live_owners() == 0 && fj_test_assets_live_owners() == 0;
    return JuicerTestSupport::test_asset_workspaces.finish(result, ownersDrained);
} catch (const std::exception& error) {
    std::fprintf(stderr, "illuminant/calibration test setup: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("illuminant/calibration test setup: unknown exception\n", stderr);
    return 1;
}
