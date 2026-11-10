#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_cases.h"
#include "juicer_test_api.h"

extern "C" const std::size_t* fj_test_spectral_abi_c_facts(std::size_t* count);
#define FJ_ABI_TYPE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_ABI_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "spectral_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
static_assert(std::is_same_v<decltype(&fj_test_hanatos_acquire), FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_arctic_acquire), FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_spectra_lut_view), FjStatus (*)(const FjSpectraLut*, FjSpectraLutView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_spectra_lut_release), FjStatus (*)(FjSpectraLut*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_mallett_acquire), FjStatus (*)(const FjAssets*, FjMallettBasis**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_mallett_view), FjStatus (*)(const FjMallettBasis*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_mallett_release), FjStatus (*)(FjMallettBasis*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_cmf_acquire), FjStatus (*)(const FjAssets*, FjCmf**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_cmf_view), FjStatus (*)(const FjCmf*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_cmf_release), FjStatus (*)(FjCmf*, FjErrorBuffer*)>);

#include "spectral_test_resources.h"
#include "test_asset_workspace.h"

namespace {
    namespace fs = std::filesystem;
    fs::path resources;
    fs::path scratch;

    void expect_status(FjStatus status, std::uint32_t expected) {
        EXPECT_EQ(status.category, expected);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
    }
    void release_status(FjStatus status) noexcept {
        if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
            std::abort();
        }
    }
    struct Assets {
        FjAssets* owner = nullptr;
        explicit Assets(const fs::path& root) {
#if defined(_WIN32)
            std::vector<std::uint16_t> units;
            for (wchar_t unit : root.native()) {
                units.push_back(static_cast<std::uint16_t>(unit));
            }
            const FjPathView path{units.data(), units.size(), FJ_PATH_WINDOWS_WIDE};
#else
            const FjPathView path{root.native().data(), root.native().size(), FJ_PATH_UNIX_BYTES};
#endif
            expect_status(fj_test_assets_create(path, &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            release_status(fj_test_assets_destroy(owner, nullptr));
        }
        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;
    };
    struct Lut {
        FjSpectraLut* owner = nullptr;
        Lut(const Assets& assets, bool arctic = false) {
            expect_status((arctic ? fj_test_arctic_acquire : fj_test_hanatos_acquire)(assets.owner, &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Lut() {
            release_status(fj_test_spectra_lut_release(owner, nullptr));
        }
        Lut(const Lut&) = delete;
        Lut& operator=(const Lut&) = delete;
        FjSpectraLutView view() const {
            FjSpectraLutView result{};
            expect_status(fj_test_spectra_lut_view(owner, &result, nullptr), FJ_STATUS_SUCCESS);
            return result;
        }
    };
    std::uint64_t fingerprint(FjFloatSpan samples) {
        std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
        for (std::size_t index = 0; index < samples.count; ++index) {
            const auto bits = std::bit_cast<std::uint32_t>(samples.data[index]);
            for (unsigned shift = 0; shift < 32; shift += 8) {
                hash = (hash ^ ((bits >> shift) & 255u)) * UINT64_C(0x100000001b3);
            }
        }
        return hash;
    }
    void expect_empty(FjFloatSpan span) {
        EXPECT_EQ(span.data, nullptr);
        EXPECT_EQ(span.count, 0u);
    }

    TEST(SpectralOwner, FailedOrUndrainedOutcomePreservesGeneratedWorkspace) {
        JuicerTestSupport::TestAssetWorkspaces workspaces;
        struct RetainEnvironment final {
            const std::optional<std::string> saved = JuicerTestSupport::test_fixture_retention_setting();
            bool restored = false;

            void restore() noexcept {
                if (restored)
                    return;
#if defined(_WIN32)
                _putenv_s("JUICER_TEST_RETAIN_FIXTURES", saved ? saved->c_str() : "");
#else
                if (saved)
                    setenv("JUICER_TEST_RETAIN_FIXTURES", saved->c_str(), 1);
                else
                    unsetenv("JUICER_TEST_RETAIN_FIXTURES");
#endif
                restored = true;
            }

            ~RetainEnvironment() {
                restore();
            }
        } retainEnvironment;

        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto parent = scratch / ("asset-workspace-lifecycle-" + std::to_string(stamp));
        ASSERT_TRUE(fs::create_directory(parent));
        const auto workspace = workspaces.create(parent / "failed-outcome");
        const auto marker = workspace / "failure-evidence.txt";
        std::ofstream(marker) << "retain until the outer run succeeds";
        ASSERT_TRUE(fs::exists(marker));
        const auto sibling = parent / "unregistered-sibling";
        ASSERT_TRUE(fs::create_directory(sibling));
        EXPECT_EQ(workspaces.finish(7, true), 7);
        EXPECT_TRUE(fs::exists(marker));
        EXPECT_EQ(workspaces.finish(0, false), 1);
        EXPECT_TRUE(fs::exists(marker));

#if defined(_WIN32)
        _putenv_s("JUICER_TEST_RETAIN_FIXTURES", "0");
#else
        unsetenv("JUICER_TEST_RETAIN_FIXTURES");
#endif
        EXPECT_EQ(workspaces.finish(0, true), 0);
        EXPECT_FALSE(fs::exists(workspace));
        EXPECT_TRUE(fs::exists(sibling));
        if (::testing::Test::HasFailure())
            return;
        std::error_code siblingError;
        fs::remove_all(sibling, siblingError);
        EXPECT_FALSE(siblingError);
        std::error_code parentError;
        fs::remove(parent, parentError);
        EXPECT_FALSE(parentError);

        const auto retained = workspaces.create(scratch / "retained-by-environment");
#if defined(_WIN32)
        _putenv_s("JUICER_TEST_RETAIN_FIXTURES", "1");
#else
        setenv("JUICER_TEST_RETAIN_FIXTURES", "1", 1);
#endif
        EXPECT_EQ(workspaces.finish(0, true), 0);
        EXPECT_TRUE(fs::exists(retained));
        retainEnvironment.restore();
        if (!::testing::Test::HasFailure()) {
            EXPECT_EQ(workspaces.finish(0, true), 0);
            EXPECT_EQ(fs::exists(retained), retainEnvironment.saved == "1");
        }
    }

    TEST(SpectralOwner, NativeLayoutsAndBundledSources) {
        static constexpr std::size_t expected[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#include "spectral_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
        };
        std::size_t count = 0;
        const auto* actual = fj_test_spectral_abi_c_facts(&count);
        ASSERT_EQ(count, std::size(expected));
        for (std::size_t index = 0; index < count; ++index) {
            EXPECT_EQ(actual[index], expected[index]);
        }
        const Assets assets(resources);
        const Lut hanatos(assets);
        const Lut arctic(assets, true);
        EXPECT_EQ(hanatos.view().samples.count, SpectralFixture::kLutCount);
        EXPECT_EQ(fingerprint(hanatos.view().samples), UINT64_C(0x81ebefd4e4cc9926));
        EXPECT_EQ(hanatos.view().asset_hash, UINT64_C(0x81ebefd4e4cc9926));
        EXPECT_EQ(arctic.view().samples.count, SpectralFixture::kLutCount);
        EXPECT_EQ(fingerprint(arctic.view().samples), UINT64_C(0x9262ffb765e3289e));
        EXPECT_EQ(arctic.view().asset_hash, UINT64_C(0x9262ffb765e3289e));
        FjMallettBasis* basis = nullptr;
        FjCmf* cmf = nullptr;
        expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan values{};
        expect_status(fj_test_mallett_view(basis, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.count, 243u);
        EXPECT_EQ(fingerprint(values), UINT64_C(0xb2ce998405c55a48));
        expect_status(fj_test_cmf_view(cmf, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.count, 324u);
        EXPECT_EQ(fingerprint(values), UINT64_C(0xc0ea4ea30d5fd861));
        release_status(fj_test_mallett_release(basis, nullptr));
        release_status(fj_test_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, SnapshotsShareBytesAndSurviveAssetsDrop) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / "snapshot");
        SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount});
        SpectralFixture::write_npy(root, SpectralFixture::kMallett, {.elementBytes = 4, .sampleCount = 243, .shape = "81,3"});
        SpectralFixture::write_cmf(root);
        Assets assets(root);
        const Lut first(assets);
        const Lut second(assets);
        EXPECT_NE(first.owner, second.owner);
        EXPECT_EQ(first.view().samples.data, second.view().samples.data);
        FjMallettBasis* basis = nullptr;
        FjCmf* cmf = nullptr;
        expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        fs::remove_all(root);
        expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        const Lut retained(assets);
        EXPECT_EQ(retained.view().samples.data, first.view().samples.data);
        release_status(fj_test_assets_destroy(std::exchange(assets.owner, nullptr), nullptr));
        std::array<FjSpectraLutView, 8> views{};
        std::vector<std::thread> threads;
        threads.reserve(views.size());
        for (auto& view : views) {
            threads.emplace_back([&first, &view] {
                expect_status(fj_test_spectra_lut_view(first.owner, &view, nullptr), FJ_STATUS_SUCCESS);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (const auto& view : views) {
            EXPECT_EQ(view.samples.data, first.view().samples.data);
            EXPECT_EQ(view.asset_hash, first.view().asset_hash);
            EXPECT_EQ(view.samples.data[0], 1.0f);
        }
        FjFloatSpan values{};
        expect_status(fj_test_mallett_view(basis, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.data[242], 1.0f);
        expect_status(fj_test_cmf_view(cmf, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.data[323], 3.0f);
        release_status(fj_test_mallett_release(basis, nullptr));
        release_status(fj_test_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, LazyIndependentFamiliesAndFirstFailure) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / "lazy");
        Assets assets(root);
        SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 2, .sampleCount = SpectralFixture::kLutCount});
        const Lut hanatos(assets);
        SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = 8, .sampleCount = SpectralFixture::kLutCount});
        const Lut arctic(assets, true);
        EXPECT_EQ(hanatos.view().asset_hash, arctic.view().asset_hash);
        FjMallettBasis* basis = nullptr;
        expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(basis, nullptr);
        SpectralFixture::write_npy(root, SpectralFixture::kMallett, {.elementBytes = 4, .sampleCount = 243, .shape = "81,3"});
        expect_status(fj_test_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        SpectralFixture::write_cmf(root, {.rows = 0});
        FjCmf* cmf = nullptr;
        expect_status(fj_test_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan rows{};
        expect_status(fj_test_cmf_view(cmf, &rows, nullptr), FJ_STATUS_SUCCESS);
        expect_empty(rows);
        release_status(fj_test_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, ClearedOutputsDiagnosticsAndConsumingRelease) {
        const Assets assets(resources);
        JuicerTests::GtestCohort capacities_cases("capacities", 4u);
        std::size_t capacities_index = 0;
        for (std::size_t capacity : {0u, 1u, 9u, 256u}) {
            auto capacities_case = capacities_cases.case_event(std::to_string(capacities_index++));
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), capacity, 999};
            FjSpectraLut* lut = nullptr;
            expect_status(fj_test_hanatos_acquire(nullptr, &lut, &error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(lut, nullptr);
            EXPECT_LT(error.length, capacity == 0 ? 1u : capacity);
            if (capacity > 0) {
                EXPECT_EQ(message[error.length], '\0');
            }

            capacities_case.passed();
        }
        JuicerTests::GtestCohort lut_families_cases("lut-families", 2u);
        std::size_t lut_families_index = 0;
        JuicerTests::GtestCohort lut_acquire_faults_cases("lut-acquire-faults", 4u);
        std::size_t lut_acquire_faults_index = 0;
        JuicerTests::GtestCohort lut_view_faults_cases("lut-view-faults", 4u);
        std::size_t lut_view_faults_index = 0;
        for (auto acquire : {fj_test_hanatos_acquire, fj_test_arctic_acquire}) {
            auto lut_families_case = lut_families_cases.case_event(std::to_string(lut_families_index++));
            expect_status(acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            for (unsigned fault : {1u, 2u}) {
                auto lut_acquire_faults_case = lut_acquire_faults_cases.case_event(std::to_string(lut_acquire_faults_index++));
                FjSpectraLut* lut = nullptr;
                fj_test_spectral_fault(fault);
                expect_status(acquire(assets.owner, &lut, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
                EXPECT_EQ(lut, nullptr);

                lut_acquire_faults_case.passed();
            }
            const auto before = fj_test_spectral_live_owners();
            FjSpectraLut* lut = nullptr;
            expect_status(acquire(assets.owner, &lut, nullptr), FJ_STATUS_SUCCESS);
            for (unsigned fault : {1u, 2u}) {
                auto lut_view_faults_case = lut_view_faults_cases.case_event(std::to_string(lut_view_faults_index++));
                FjSpectraLutView view{{nullptr, 999}, 999};
                fj_test_spectral_fault(fault);
                expect_status(fj_test_spectra_lut_view(lut, &view, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
                expect_empty(view.samples);
                EXPECT_EQ(view.asset_hash, 0u);

                lut_view_faults_case.passed();
            }
            FjErrorBuffer malformed{nullptr, 1, 999};
            FjSpectraLutView view{{nullptr, 999}, 999};
            expect_status(fj_test_spectra_lut_view(lut, &view, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_empty(view.samples);
            EXPECT_EQ(view.asset_hash, 0u);
            expect_status(fj_test_spectra_lut_release(std::exchange(lut, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(fj_test_spectral_live_owners(), before);

            lut_families_case.passed();
        }
        FjSpectraLutView view{{nullptr, 999}, 999};
        expect_status(fj_test_spectra_lut_view(nullptr, &view, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(view.samples);
        EXPECT_EQ(view.asset_hash, 0u);
        release_status(fj_test_spectra_lut_release(nullptr, nullptr));
    }

    TEST(SpectralOwner, MallettAndCmfFailureClearingAndConsumption) {
        const Assets assets(resources);
        const auto before = fj_test_spectral_live_owners();
        FjMallettBasis* basis = nullptr;
        FjCmf* cmf = nullptr;
        expect_status(fj_test_mallett_acquire(nullptr, &basis, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_cmf_acquire(nullptr, &cmf, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_mallett_acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_cmf_acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        JuicerTests::GtestCohort mallett_cmf_acquire_cases("mallett-cmf-acquire", 2u);
        std::size_t mallett_cmf_acquire_index = 0;
        for (unsigned fault : {1u, 2u}) {
            auto mallett_cmf_acquire_case = mallett_cmf_acquire_cases.case_event(std::to_string(mallett_cmf_acquire_index++));
            const auto expected = fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE;
            fj_test_spectral_fault(fault);
            expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), expected);
            EXPECT_EQ(basis, nullptr);
            fj_test_spectral_fault(fault);
            expect_status(fj_test_cmf_acquire(assets.owner, &cmf, nullptr), expected);
            EXPECT_EQ(cmf, nullptr);

            mallett_cmf_acquire_case.passed();
        }
        expect_status(fj_test_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        JuicerTests::GtestCohort mallett_cmf_view_cases("mallett-cmf-view", 2u);
        std::size_t mallett_cmf_view_index = 0;
        for (unsigned fault : {1u, 2u}) {
            auto mallett_cmf_view_case = mallett_cmf_view_cases.case_event(std::to_string(mallett_cmf_view_index++));
            const auto expected = fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE;
            FjFloatSpan span{nullptr, 999};
            fj_test_spectral_fault(fault);
            expect_status(fj_test_mallett_view(basis, &span, nullptr), expected);
            expect_empty(span);
            fj_test_spectral_fault(fault);
            span.count = 999;
            expect_status(fj_test_cmf_view(cmf, &span, nullptr), expected);
            expect_empty(span);

            mallett_cmf_view_case.passed();
        }
        FjErrorBuffer malformed{nullptr, 1, 999};
        expect_status(fj_test_mallett_release(std::exchange(basis, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_cmf_release(std::exchange(cmf, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_spectral_live_owners(), before);
        FjFloatSpan span{nullptr, 999};
        expect_status(fj_test_mallett_view(nullptr, &span, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(span);
        span.count = 999;
        expect_status(fj_test_cmf_view(nullptr, &span, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(span);
        release_status(fj_test_mallett_release(nullptr, nullptr));
        release_status(fj_test_cmf_release(nullptr, nullptr));
    }
} // namespace

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (::testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    JuicerTests::CaseLog case_log;
    if (argc != 3) {
        return 2;
    }
    resources = fs::absolute(argv[1]);
    scratch = fs::absolute(argv[2]);
    fs::create_directories(scratch);
    const int result = RUN_ALL_TESTS();
    case_log.finish();
    const bool ownersDrained = fj_test_spectral_live_owners() == 0 && fj_test_assets_live_owners() == 0;
    return JuicerTestSupport::test_asset_workspaces.finish(result, ownersDrained);
} catch (const std::exception& error) {
    std::fprintf(stderr, "spectral fixture setup failed: %s\n", error.what());
    return 2;
} catch (...) {
    std::fputs("spectral fixture setup failed\n", stderr);
    return 2;
}
