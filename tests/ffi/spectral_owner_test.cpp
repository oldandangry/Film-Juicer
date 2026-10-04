#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "juicer_legacy_api.h"
#include "juicer_test_api.h"

extern "C" const std::size_t* fj_test_spectral_abi_c_facts(std::size_t* count);
#define FJ_ABI_TYPE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_ABI_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "spectral_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
static_assert(std::is_same_v<decltype(&fj_legacy_hanatos_acquire), FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_arctic_acquire), FjStatus (*)(const FjAssets*, FjSpectraLut**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_spectra_lut_view), FjStatus (*)(const FjSpectraLut*, FjSpectraLutView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_spectra_lut_release), FjStatus (*)(FjSpectraLut*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_mallett_acquire), FjStatus (*)(const FjAssets*, FjMallettBasis**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_mallett_view), FjStatus (*)(const FjMallettBasis*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_mallett_release), FjStatus (*)(FjMallettBasis*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_cmf_acquire), FjStatus (*)(const FjAssets*, FjCmf**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_cmf_view), FjStatus (*)(const FjCmf*, FjFloatSpan*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_legacy_cmf_release), FjStatus (*)(FjCmf*, FjErrorBuffer*)>);

#include "spectral_test_resources.h"

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
            expect_status(fj_legacy_assets_create(path, &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            release_status(fj_legacy_assets_destroy(owner, nullptr));
        }
        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;
    };
    struct Lut {
        FjSpectraLut* owner = nullptr;
        Lut(const Assets& assets, bool arctic = false) {
            expect_status((arctic ? fj_legacy_arctic_acquire : fj_legacy_hanatos_acquire)(assets.owner, &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Lut() {
            release_status(fj_legacy_spectra_lut_release(owner, nullptr));
        }
        Lut(const Lut&) = delete;
        Lut& operator=(const Lut&) = delete;
        FjSpectraLutView view() const {
            FjSpectraLutView result{};
            expect_status(fj_legacy_spectra_lut_view(owner, &result, nullptr), FJ_STATUS_SUCCESS);
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
        expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan values{};
        expect_status(fj_legacy_mallett_view(basis, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.count, 243u);
        EXPECT_EQ(fingerprint(values), UINT64_C(0xb2ce998405c55a48));
        expect_status(fj_legacy_cmf_view(cmf, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.count, 324u);
        EXPECT_EQ(fingerprint(values), UINT64_C(0xc0ea4ea30d5fd861));
        release_status(fj_legacy_mallett_release(basis, nullptr));
        release_status(fj_legacy_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, SnapshotsShareBytesAndSurviveAssetsDrop) {
        const auto root = scratch / "snapshot";
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
        expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        fs::remove_all(root);
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        const Lut retained(assets);
        EXPECT_EQ(retained.view().samples.data, first.view().samples.data);
        release_status(fj_legacy_assets_destroy(std::exchange(assets.owner, nullptr), nullptr));
        std::array<FjSpectraLutView, 8> views{};
        std::vector<std::thread> threads;
        threads.reserve(views.size());
        for (auto& view : views) {
            threads.emplace_back([&first, &view] {
                expect_status(fj_legacy_spectra_lut_view(first.owner, &view, nullptr), FJ_STATUS_SUCCESS);
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
        expect_status(fj_legacy_mallett_view(basis, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.data[242], 1.0f);
        expect_status(fj_legacy_cmf_view(cmf, &values, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(values.data[323], 3.0f);
        release_status(fj_legacy_mallett_release(basis, nullptr));
        release_status(fj_legacy_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, LazyIndependentFamiliesAndFirstFailure) {
        const auto root = scratch / "lazy";
        fs::remove_all(root);
        Assets assets(root);
        SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 2, .sampleCount = SpectralFixture::kLutCount});
        const Lut hanatos(assets);
        SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = 8, .sampleCount = SpectralFixture::kLutCount});
        const Lut arctic(assets, true);
        EXPECT_EQ(hanatos.view().asset_hash, arctic.view().asset_hash);
        FjMallettBasis* basis = nullptr;
        expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(basis, nullptr);
        SpectralFixture::write_npy(root, SpectralFixture::kMallett, {.elementBytes = 4, .sampleCount = 243, .shape = "81,3"});
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        SpectralFixture::write_cmf(root, {.rows = 0});
        FjCmf* cmf = nullptr;
        expect_status(fj_legacy_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        FjFloatSpan rows{};
        expect_status(fj_legacy_cmf_view(cmf, &rows, nullptr), FJ_STATUS_SUCCESS);
        expect_empty(rows);
        release_status(fj_legacy_cmf_release(cmf, nullptr));
    }

    TEST(SpectralOwner, ClearedOutputsDiagnosticsAndConsumingRelease) {
        const Assets assets(resources);
        for (std::size_t capacity : {0u, 1u, 9u, 256u}) {
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), capacity, 999};
            FjSpectraLut* lut = nullptr;
            expect_status(fj_legacy_hanatos_acquire(nullptr, &lut, &error), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(lut, nullptr);
            EXPECT_LT(error.length, capacity == 0 ? 1u : capacity);
            if (capacity > 0) {
                EXPECT_EQ(message[error.length], '\0');
            }
        }
        for (auto acquire : {fj_legacy_hanatos_acquire, fj_legacy_arctic_acquire}) {
            expect_status(acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            for (unsigned fault : {1u, 2u}) {
                FjSpectraLut* lut = nullptr;
                fj_test_spectral_fault(fault);
                expect_status(acquire(assets.owner, &lut, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
                EXPECT_EQ(lut, nullptr);
            }
            const auto before = fj_test_spectral_live_owners();
            FjSpectraLut* lut = nullptr;
            expect_status(acquire(assets.owner, &lut, nullptr), FJ_STATUS_SUCCESS);
            for (unsigned fault : {1u, 2u}) {
                FjSpectraLutView view{{nullptr, 999}, 999};
                fj_test_spectral_fault(fault);
                expect_status(fj_legacy_spectra_lut_view(lut, &view, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE);
                expect_empty(view.samples);
                EXPECT_EQ(view.asset_hash, 0u);
            }
            FjErrorBuffer malformed{nullptr, 1, 999};
            FjSpectraLutView view{{nullptr, 999}, 999};
            expect_status(fj_legacy_spectra_lut_view(lut, &view, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
            expect_empty(view.samples);
            EXPECT_EQ(view.asset_hash, 0u);
            expect_status(fj_legacy_spectra_lut_release(std::exchange(lut, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(fj_test_spectral_live_owners(), before);
        }
        FjSpectraLutView view{{nullptr, 999}, 999};
        expect_status(fj_legacy_spectra_lut_view(nullptr, &view, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(view.samples);
        EXPECT_EQ(view.asset_hash, 0u);
        release_status(fj_legacy_spectra_lut_release(nullptr, nullptr));
    }

    TEST(SpectralOwner, MallettAndCmfFailureClearingAndConsumption) {
        const Assets assets(resources);
        const auto before = fj_test_spectral_live_owners();
        FjMallettBasis* basis = nullptr;
        FjCmf* cmf = nullptr;
        expect_status(fj_legacy_mallett_acquire(nullptr, &basis, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_cmf_acquire(nullptr, &cmf, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_mallett_acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_cmf_acquire(assets.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        for (unsigned fault : {1u, 2u}) {
            const auto expected = fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE;
            fj_test_spectral_fault(fault);
            expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), expected);
            EXPECT_EQ(basis, nullptr);
            fj_test_spectral_fault(fault);
            expect_status(fj_legacy_cmf_acquire(assets.owner, &cmf, nullptr), expected);
            EXPECT_EQ(cmf, nullptr);
        }
        expect_status(fj_legacy_mallett_acquire(assets.owner, &basis, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_cmf_acquire(assets.owner, &cmf, nullptr), FJ_STATUS_SUCCESS);
        for (unsigned fault : {1u, 2u}) {
            const auto expected = fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_ALLOCATION_FAILURE;
            FjFloatSpan span{nullptr, 999};
            fj_test_spectral_fault(fault);
            expect_status(fj_legacy_mallett_view(basis, &span, nullptr), expected);
            expect_empty(span);
            fj_test_spectral_fault(fault);
            span.count = 999;
            expect_status(fj_legacy_cmf_view(cmf, &span, nullptr), expected);
            expect_empty(span);
        }
        FjErrorBuffer malformed{nullptr, 1, 999};
        expect_status(fj_legacy_mallett_release(std::exchange(basis, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_legacy_cmf_release(std::exchange(cmf, nullptr), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_spectral_live_owners(), before);
        FjFloatSpan span{nullptr, 999};
        expect_status(fj_legacy_mallett_view(nullptr, &span, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(span);
        span.count = 999;
        expect_status(fj_legacy_cmf_view(nullptr, &span, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(span);
        release_status(fj_legacy_mallett_release(nullptr, nullptr));
        release_status(fj_legacy_cmf_release(nullptr, nullptr));
    }
} // namespace

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 3) {
        return 2;
    }
    resources = fs::absolute(argv[1]);
    scratch = fs::absolute(argv[2]);
    fs::create_directories(scratch);
    const int result = RUN_ALL_TESTS();
    return result == 0 && fj_test_spectral_live_owners() == 0 && fj_test_assets_live_owners() == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "spectral fixture setup failed: %s\n", error.what());
    return 2;
} catch (...) {
    std::fputs("spectral fixture setup failed\n", stderr);
    return 2;
}
