#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ColorTransforms.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "RustAssetBridge.h"
#include "SpectralProcessing.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "spectral_test_resources.h"

namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
        static bool has_no_cuda(const Root& root) {
            return root._cudaContextResources.empty() && root._cudaDeviceLedgers.empty() && root._activeFramePreparations == 0;
        }
    };
} // namespace JuicerProcess::TestSupport

namespace AllocationTest {
    // Executable-local counting observes requested bytes and actual cleanup.
    // Product and Rust allocators are unchanged; allocator bookkeeping is excluded.
    struct alignas(std::max_align_t) Header {
        std::size_t bytes;
        bool counted;
    };
    std::atomic<std::size_t> liveBytes{0};
    thread_local bool counting = false;
    thread_local std::size_t failBytes = 0;
} // namespace AllocationTest

// Standard-library allocation declarations use different parameter names on
// GNU/MSVC. These replacements retain portable local names.
// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t bytes) {
    if (AllocationTest::failBytes == bytes && bytes != 0) {
        AllocationTest::failBytes = 0;
        throw std::bad_alloc();
    }
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(AllocationTest::Header)) {
        throw std::bad_alloc();
    }
    auto* block = static_cast<AllocationTest::Header*>(std::malloc(sizeof(AllocationTest::Header) + bytes));
    if (!block) {
        throw std::bad_alloc();
    }
    block->bytes = bytes;
    block->counted = AllocationTest::counting;
    if (block->counted) {
        AllocationTest::liveBytes.fetch_add(bytes);
    }
    return block + 1;
}
void operator delete(void* pointer) noexcept {
    if (pointer) {
        auto* block = static_cast<AllocationTest::Header*>(pointer) - 1;
        // The replacement new returns one Header past malloc's base. The
        // analyzer models standard new without that prefix allocation.
        if (block->counted) { // NOLINT(clang-analyzer-security.ArrayBound)
            AllocationTest::liveBytes.fetch_sub(block->bytes);
        }
        std::free(block);
    }
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
    fs::path resources;
    fs::path scratch;
    struct Observation {
        const char* family = nullptr;
        std::size_t length = 0;
        std::size_t capacity = 0;
        std::size_t priorNativeBytes = 0;
    };
    std::array<Observation, 128> observations{};
    std::size_t observationCount = 0;
    unsigned viewFault = 0;
    const char* faultFamily = "";
    unsigned allocationFault = 0;
    unsigned curveAllocation = 0;
    unsigned failCurve = 0;
    const float* borrowedLut = nullptr;
    std::barrier<>* copyBarrier = nullptr;
    std::vector<std::string> acquisitions;

    std::size_t native_bytes() {
        const auto& context = Spectral::context();
        return sizeof(float) * (context.hanSpectra.data.capacity() + context.arcticSpectra.data.capacity() + context.mallettBasis.data.capacity());
    }
    void reset_observations() {
        observationCount = 0;
        acquisitions.clear();
        acquisitions.reserve(8);
        viewFault = 0;
        faultFamily = "";
        allocationFault = 0;
        curveAllocation = 0;
        failCurve = 0;
    }
    fs::path source_root(const fs::path& name) {
        const auto root = scratch / name;
        fs::remove_all(root);
        fs::create_directories(root);
        fs::copy_file(resources / "cie1931_2deg.csv", root / "cie1931_2deg.csv");
        for (const auto* file : {SpectralFixture::kHanatos, SpectralFixture::kArctic, SpectralFixture::kMallett}) {
            const auto path = SpectralFixture::source_path(root, file);
            fs::create_directories(path.parent_path());
            fs::copy_file(SpectralFixture::source_path(resources, file), path);
        }
        return root;
    }
    fs::path consumer_root(const std::string& name) {
        const auto root = source_root(name);
        for (const char* directory : {"profiles", "illuminants", "filters", "gamut"}) {
            if (fs::exists(resources / directory)) {
                fs::copy(resources / directory, root / directory, fs::copy_options::recursive);
            }
        }
        return root;
    }
    void expect_no_cuda() {
        EXPECT_TRUE(JuicerProcess::TestSupport::RootLifetimeObserver::has_no_cuda(JuicerProcess::root()));
    }
    void expect_empty_cmf() {
        for (const auto* curve : {&Spectral::gXBar, &Spectral::gYBar, &Spectral::gZBar}) {
            EXPECT_TRUE(curve->lambda_nm.empty());
            EXPECT_TRUE(curve->linear.empty());
            EXPECT_EQ(curve->lambda_nm.capacity(), 0u);
            EXPECT_EQ(curve->linear.capacity(), 0u);
        }
    }
    ParamSnapshot controls(unsigned method) {
        ParamSnapshot snapshot;
        snapshot.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        snapshot.spectralUpsamplingMode = static_cast<int>(method);
        snapshot.dirCouplers.active = false;
        snapshot.outputGamutCompressionEnabled = 0;
        snapshot.inputCompressionEnabled = 0;
        snapshot.hanatos2025AdaptationWindow = 0;
        snapshot.hanatos2025AdaptationSurface = 0;
        return snapshot;
    }

    TEST(SpectralBootstrap, ColdOrderWarmReuseAndNativeLifetime) {
        const auto root = source_root("warm-reuse");
        reset_observations();
        JuicerCuda::Owner owner;
        owner.create(root);
        EXPECT_TRUE(acquisitions.empty());
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        std::vector<std::thread> callers;
        callers.reserve(8);
        for (unsigned i = 0; i < 8; ++i) {
            callers.emplace_back([] {
                JuicerProcess::root().ensure_bootstrap();
            });
        }
        for (auto& caller : callers) {
            caller.join();
        }
        ASSERT_EQ(acquisitions, (std::vector<std::string>{"cmf", "hanatos", "arctic", "mallett"}));
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        ASSERT_TRUE(Spectral::hanatos_available());
        ASSERT_TRUE(Spectral::arctic_available());
        ASSERT_TRUE(Spectral::mallett_available());
        EXPECT_EQ(Spectral::gHanSpectra.assetHash, UINT64_C(0x81ebefd4e4cc9926));
        EXPECT_EQ(Spectral::context().arcticSpectra.assetHash, UINT64_C(0x9262ffb765e3289e));
        const auto* hanatos = Spectral::gHanSpectra.data.data();
        const auto* arctic = Spectral::context().arcticSpectra.data.data();
        const auto* basis = Spectral::gMallettBasis.data.data();
        const auto copies = observationCount;
        fs::remove_all(root);
        EXPECT_EQ(JuicerProcess::root().assets().release_cached_payloads().category, FJ_STATUS_SUCCESS);
        for (unsigned i = 0; i < 8; ++i) {
            JuicerProcess::root().ensure_bootstrap();
        }
        EXPECT_EQ(observationCount, copies);
        EXPECT_EQ(Spectral::gHanSpectra.data.data(), hanatos);
        EXPECT_EQ(Spectral::context().arcticSpectra.data.data(), arctic);
        EXPECT_EQ(Spectral::gMallettBasis.data.data(), basis);
        expect_no_cuda();
        EXPECT_EQ(owner.close().category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_assets_live_owners(), 0u);
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        EXPECT_EQ(Spectral::gHanSpectra.data.front(), 5.84375f);
        EXPECT_EQ(Spectral::gMallettBasis.data.back(), std::bit_cast<float>(UINT32_C(0x3eaa8d89)));
        EXPECT_EQ(Spectral::gXBar.linear.size(), 81u);
    }

    TEST(SpectralBootstrap, SequentialOwnerCmfFailuresClearAllCurves) {
        for (const char* failure : {"missing", "empty", "axis", "token", "copy", "construction"}) {
            SCOPED_TRACE(failure);
            reset_observations();
            {
                JuicerCuda::Owner first;
                first.create(resources);
                JuicerProcess::root().ensure_bootstrap();
                ASSERT_EQ(Spectral::gXBar.linear.size(), 81u);
            }
            auto root = consumer_root(std::string("cmf-") + failure);
            if (std::strcmp(failure, "missing") == 0) {
                fs::remove(root / "cie1931_2deg.csv");
            }
            if (std::strcmp(failure, "empty") == 0) {
                SpectralFixture::write_cmf(root, {.rows = 0});
            }
            if (std::strcmp(failure, "axis") == 0) {
                SpectralFixture::write_cmf(root, {.wavelengthOffsetNm = 1});
            }
            if (std::strcmp(failure, "token") == 0) {
                std::ofstream file(root / "cie1931_2deg.csv", std::ios::trunc);
                file << "380,nan,1,2\n";
            }
            if (std::strcmp(failure, "copy") == 0) {
                faultFamily = "cmf";
                allocationFault = 2;
            }
            if (std::strcmp(failure, "construction") == 0) {
                curveAllocation = 0;
                failCurve = 2;
            }
            JuicerCuda::Owner second;
            second.create(root);
            JuicerProcess::root().ensure_bootstrap();
            expect_empty_cmf();
            EXPECT_TRUE(Spectral::hanatos_available());
            EXPECT_TRUE(Spectral::arctic_available());
            EXPECT_TRUE(Spectral::mallett_available());
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            EXPECT_FALSE(build_direct_render_state_product(controls(1), product, diagnostic));
            EXPECT_FALSE(diagnostic.empty());
            expect_no_cuda();
        }
    }

    TEST(SpectralBootstrap, IndependentMissingFamiliesAndSelectedConsumers) {
        for (const char* missing : {SpectralFixture::kHanatos, SpectralFixture::kArctic, SpectralFixture::kMallett}) {
            SCOPED_TRACE(missing);
            const auto root = consumer_root(std::string("missing-") + missing);
            fs::remove(SpectralFixture::source_path(root, missing));
            reset_observations();
            JuicerCuda::Owner owner;
            owner.create(root);
            JuicerProcess::root().ensure_bootstrap();
            const unsigned absent = missing == SpectralFixture::kHanatos ? 0u : (missing == SpectralFixture::kArctic ? 2u : 1u);
            for (unsigned method : {0u, 1u, 2u}) {
                FocusedRenderStateBuildProduct product;
                std::string diagnostic;
                const bool success = build_direct_render_state_product(controls(method), product, diagnostic);
                EXPECT_EQ(success, method != absent) << "method=" << method << ' ' << diagnostic;
                if (method == absent) {
                    EXPECT_FALSE(diagnostic.empty());
                }
            }
            if (absent == 0) {
                EXPECT_FALSE(Spectral::hanatos_available());
                EXPECT_TRUE(Spectral::gHanSpectra.data.empty());
                EXPECT_EQ(Spectral::gHanSpectra.assetHash, 0u);
            } else if (absent == 2) {
                EXPECT_FALSE(Spectral::arctic_available());
                EXPECT_TRUE(Spectral::context().arcticSpectra.data.empty());
                EXPECT_EQ(Spectral::context().arcticSpectra.assetHash, 0u);
            } else {
                EXPECT_FALSE(Spectral::mallett_available());
                EXPECT_TRUE(Spectral::gMallettBasis.data.empty());
            }
            expect_no_cuda();
        }
    }

    TEST(SpectralBootstrap, NonfiniteHanatosReachesSelectedComputation) {
        const auto root = consumer_root("nonfinite-hanatos");
        SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount}, {0x7fc12345u});
        reset_observations();
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        ASSERT_TRUE(Spectral::hanatos_available());
        EXPECT_EQ(std::bit_cast<std::uint32_t>(Spectral::gHanSpectra.data[0]), 0x7fc12345u);
        EXPECT_NE(Spectral::gHanSpectra.assetHash, 0u);
        FilmRawRecipe recipe;
        recipe.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        recipe.tcLutHash = 1;
        recipe.finalSensitivity.fill({1.0f, 1.0f, 1.0f});
        std::array<float, 81> illuminant{};
        illuminant.fill(1.0f);
        std::optional<Spectral::FilmTcLut> result;
        std::string diagnostic;
        EXPECT_FALSE(Spectral::build_film_tc_lut(recipe, Spectral::gHanSpectra, illuminant, result, diagnostic));
        EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
        EXPECT_FALSE(result.has_value());
        FocusedRenderStateBuildProduct product;
        EXPECT_FALSE(build_direct_render_state_product(controls(0), product, diagnostic));
        EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
        expect_no_cuda();
    }

    TEST(SpectralBootstrap, SupportedWidthsChangedExtremaAndSignedSpecials) {
        const std::vector<std::uint32_t> decoded{0u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x7f800000u, 0xff800000u, 0x7fc46000u, 0xffc6a000u};
        for (unsigned width : {2u, 4u, 8u}) {
            const auto root = source_root("width-" + std::to_string(width));
            std::vector<std::uint64_t> bits;
            if (width == 2) {
                bits = {0u, 0x8000u, 0x3c00u, 0xc000u, 0x7c00u, 0xfc00u, 0x7e23u, 0xfe35u};
            } else {
                for (auto sample : decoded) {
                    bits.push_back(width == 4 ? sample : std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<float>(sample))));
                }
            }
            SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = width, .sampleCount = SpectralFixture::kLutCount}, bits);
            SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = width, .sampleCount = SpectralFixture::kLutCount}, bits);
            SpectralFixture::write_npy(root, SpectralFixture::kMallett, {.elementBytes = width, .sampleCount = 243, .shape = "81,3"}, bits);
            reset_observations();
            JuicerCuda::Owner owner;
            owner.create(root);
            JuicerProcess::root().ensure_bootstrap();
            ASSERT_TRUE(Spectral::hanatos_available());
            ASSERT_TRUE(Spectral::arctic_available());
            ASSERT_TRUE(Spectral::mallett_available());
            for (std::size_t index = 0; index < decoded.size(); ++index) {
                EXPECT_EQ(std::bit_cast<std::uint32_t>(Spectral::gHanSpectra.data[index]), decoded[index]);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(Spectral::context().arcticSpectra.data[index]), decoded[index]);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(Spectral::gMallettBasis.data[index]), decoded[index]);
            }
            // Independent FNV-1a stream: zero signs -> 0, NaNs -> 0x7fc00000,
            // other decoded bits unchanged; eight authored samples then ones.
            EXPECT_EQ(Spectral::gHanSpectra.assetHash, UINT64_C(0x4e766e5978ce36d8));
            EXPECT_EQ(Spectral::gHanSpectra.assetHash, Spectral::context().arcticSpectra.assetHash);
            Spectral::SpectralTables tables;
            tables.K = 81;
            tables.illum.assign(81, 1.0f);
            Spectral::Curve sensitivity;
            sensitivity.linear.assign(81, 1.0f);
            const float rgb[3]{1.0f, 1.0f, 1.0f};
            float exposure[3]{};
            Spectral::mallett2019_exposures_from_linear_srgb(rgb, tables, sensitivity, sensitivity, sensitivity, exposure);
            // Row zero contributes 1, rows one/two have nonfinite SPD and are
            // skipped by the retained computation; the remaining 78 contribute 3.
            for (float channel : exposure) {
                EXPECT_EQ(channel, 235.0f);
            }
            FilmRawRecipe recipe;
            recipe.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Arctic2026beta04;
            recipe.tcLutHash = 1;
            recipe.projectionWhiteXYZ = {0.950455f, 1.0f, 1.089058f};
            recipe.finalSensitivity.fill({1.0f, 1.0f, 1.0f});
            std::array<float, 81> illuminant{};
            illuminant.fill(1.0f);
            std::optional<Spectral::FilmTcLut> integrated;
            std::string diagnostic;
            EXPECT_FALSE(Spectral::build_film_tc_lut(recipe, Spectral::context().arcticSpectra, illuminant, integrated, diagnostic));
            EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
        }
    }

    TEST(SpectralBootstrap, AlteredFiniteSourcesAndOverflowReachComputation) {
        for (bool overflow : {false, true}) {
            const auto root = consumer_root(overflow ? "narrowing-overflow" : "changed-finite");
            if (overflow) {
                SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 8, .sampleCount = SpectralFixture::kLutCount}, {std::bit_cast<std::uint64_t>(1e300)});
            } else {
                SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount}, {0x42c80000u, 0xc2480000u});
            }
            reset_observations();
            JuicerCuda::Owner owner;
            owner.create(root);
            JuicerProcess::root().ensure_bootstrap();
            EXPECT_TRUE(Spectral::hanatos_available());
            EXPECT_TRUE(Spectral::arctic_available());
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            if (overflow) {
                EXPECT_TRUE(std::isinf(Spectral::gHanSpectra.data.front()));
                EXPECT_FALSE(build_direct_render_state_product(controls(0), product, diagnostic));
                EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
            } else {
                EXPECT_EQ(Spectral::context().arcticSpectra.data[0], 100.0f);
                EXPECT_EQ(Spectral::context().arcticSpectra.data[1], -50.0f);
                EXPECT_NE(Spectral::context().arcticSpectra.assetHash, UINT64_C(0x9262ffb765e3289e));
                EXPECT_TRUE(build_direct_render_state_product(controls(2), product, diagnostic)) << diagnostic;
            }
        }
    }

    TEST(SpectralBootstrap, MalformedSourcesFailWithoutTranspositionOrPartialPublication) {
        for (unsigned failure = 0; failure < 6; ++failure) {
            const auto root = source_root("malformed-" + std::to_string(failure));
            if (failure == 0) {
                SpectralFixture::write_npy(root, SpectralFixture::kMallett, {.elementBytes = 4, .sampleCount = 243, .shape = "3,81"});
            }
            if (failure == 1) {
                SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount, .shape = "192,192,81", .version = 2});
            }
            if (failure == 2) {
                fs::resize_file(SpectralFixture::source_path(root, SpectralFixture::kArctic), 512);
            }
            if (failure == 3) {
                std::ofstream file(SpectralFixture::source_path(root, SpectralFixture::kArctic), std::ios::binary | std::ios::app);
                file.put('x');
            }
            if (failure >= 4) {
                SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount});
                std::fstream file(SpectralFixture::source_path(root, SpectralFixture::kArctic), std::ios::binary | std::ios::in | std::ios::out);
                std::array<char, 128> header{};
                file.read(header.data(), static_cast<std::streamsize>(header.size()));
                const std::string_view text(header.data(), header.size());
                const auto offset = text.find(failure == 4 ? "<f4" : "False");
                ASSERT_NE(offset, std::string_view::npos);
                file.seekp(static_cast<std::streamoff>(offset));
                file.write(failure == 4 ? ">f4" : "True ", failure == 4 ? 3 : 5);
            }
            reset_observations();
            JuicerCuda::Owner owner;
            owner.create(root);
            JuicerProcess::root().ensure_bootstrap();
            EXPECT_EQ(Spectral::mallett_available(), failure != 0);
            EXPECT_EQ(Spectral::hanatos_available(), failure != 1);
            EXPECT_EQ(Spectral::arctic_available(), failure < 2);
            if (failure >= 2) {
                EXPECT_EQ(Spectral::context().arcticSpectra.assetHash, 0u);
                EXPECT_TRUE(Spectral::context().arcticSpectra.data.empty());
            }
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        }
    }

    TEST(SpectralCopy, StructuralForeignViewsFailAndHandlesExpire) {
        JuicerAssets::AssetBridge bridge(resources);
        for (unsigned fault : {1u, 2u, 3u, 4u}) {
            viewFault = fault;
            faultFamily = "hanatos";
            EXPECT_THROW((void)bridge.copy_hanatos_lut(), JuicerCuda::ExecutionFailure);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            viewFault = fault;
            faultFamily = "mallett";
            EXPECT_THROW((void)bridge.copy_mallett_basis(), JuicerCuda::ExecutionFailure);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            viewFault = fault;
            faultFamily = "cmf";
            EXPECT_THROW((void)bridge.copy_cmf_triplets(), JuicerCuda::ExecutionFailure);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        }
        viewFault = 0;
        const auto copy = bridge.copy_hanatos_lut();
        EXPECT_NE(copy.data.data(), borrowedLut);
        EXPECT_EQ(copy.assetHash, UINT64_C(0x81ebefd4e4cc9926));
    }

    TEST(SpectralCopy, FailedNativeAllocationsReclaimCopiesAndKeepSuccessfulSources) {
        const auto root = source_root("failed-copy");
        JuicerAssets::AssetBridge bridge(root);
        reset_observations();
        faultFamily = "hanatos";
        allocationFault = 1;
        AllocationTest::counting = true;
        bool failed = false;
        try {
            (void)bridge.copy_hanatos_lut();
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        AllocationTest::counting = false;
        EXPECT_TRUE(failed);
        EXPECT_EQ(AllocationTest::liveBytes.load(), 0u);
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        fs::remove(SpectralFixture::source_path(root, SpectralFixture::kHanatos));
        const auto copy = bridge.copy_hanatos_lut();
        EXPECT_EQ(copy.assetHash, UINT64_C(0x81ebefd4e4cc9926));
        for (unsigned nth : {1u, 2u, 3u}) {
            faultFamily = "cmf";
            allocationFault = nth;
            AllocationTest::counting = true;
            failed = false;
            try {
                (void)bridge.copy_cmf_triplets();
            } catch (const std::bad_alloc&) {
                failed = true;
            }
            AllocationTest::counting = false;
            EXPECT_TRUE(failed);
            EXPECT_EQ(AllocationTest::liveBytes.load(), 0u);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        }
        EXPECT_EQ(bridge.copy_cmf_triplets().xbar.size(), 81u);
    }

    TEST(SpectralCopy, CacheReleaseOverlapsColdCopyWithoutRevokingView) {
        JuicerAssets::AssetBridge bridge(resources);
        std::barrier barrier(2);
        copyBarrier = &barrier;
        Spectral::ReconstructionLut copy;
        std::exception_ptr failure;
        std::thread loader([&] {
            try {
                copy = bridge.copy_hanatos_lut();
            } catch (...) {
                failure = std::current_exception();
            }
        });
        barrier.arrive_and_wait();
        EXPECT_EQ(bridge.release_cached_payloads().category, FJ_STATUS_SUCCESS);
        barrier.arrive_and_wait();
        loader.join();
        copyBarrier = nullptr;
        if (failure) {
            std::rethrow_exception(failure);
        }
        EXPECT_EQ(copy.assetHash, UINT64_C(0x81ebefd4e4cc9926));
        EXPECT_EQ(copy.data.front(), 5.84375f);
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
    }

    TEST(SpectralBootstrap, FailedNativeReconstructionCopyClearsPreviousRecordAndCompletesAttempt) {
        for (const char* family : {"hanatos", "arctic", "mallett"}) {
            reset_observations();
            {
                JuicerCuda::Owner previous;
                previous.create(resources);
                JuicerProcess::root().ensure_bootstrap();
            }
            const auto root = source_root(std::string("failed-native-") + family);
            faultFamily = family;
            allocationFault = 1;
            JuicerCuda::Owner owner;
            owner.create(root);
            JuicerProcess::root().ensure_bootstrap();
            if (std::strcmp(family, "hanatos") == 0) {
                EXPECT_FALSE(Spectral::hanatos_available());
                EXPECT_TRUE(Spectral::gHanSpectra.data.empty());
                EXPECT_EQ(Spectral::gHanSpectra.assetHash, 0u);
                fs::remove(SpectralFixture::source_path(root, SpectralFixture::kHanatos));
                EXPECT_EQ(JuicerProcess::root().assets().copy_hanatos_lut().assetHash, UINT64_C(0x81ebefd4e4cc9926));
            } else if (std::strcmp(family, "arctic") == 0) {
                EXPECT_FALSE(Spectral::arctic_available());
                EXPECT_TRUE(Spectral::context().arcticSpectra.data.empty());
                EXPECT_EQ(Spectral::context().arcticSpectra.assetHash, 0u);
                fs::remove(SpectralFixture::source_path(root, SpectralFixture::kArctic));
                EXPECT_EQ(JuicerProcess::root().assets().copy_arctic_lut().assetHash, UINT64_C(0x9262ffb765e3289e));
            } else {
                EXPECT_FALSE(Spectral::mallett_available());
                EXPECT_TRUE(Spectral::gMallettBasis.data.empty());
                fs::remove(SpectralFixture::source_path(root, SpectralFixture::kMallett));
                EXPECT_EQ(JuicerProcess::root().assets().copy_mallett_basis().data.size(), 243u);
            }
            const auto copied = observationCount;
            JuicerProcess::root().ensure_bootstrap();
            EXPECT_EQ(observationCount, copied);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            expect_no_cuda();
        }
    }

    TEST(SpectralBootstrap, CopyCapacityAndSequentialOwnerOverlapRemainBounded) {
        reset_observations();
        for (unsigned repetition = 0; repetition < 3; ++repetition) {
            JuicerCuda::Owner owner;
            owner.create(resources);
            JuicerProcess::root().ensure_bootstrap();
            EXPECT_EQ(native_bytes(), 2u * SpectralFixture::kLutCount * sizeof(float) + 243u * sizeof(float));
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            for (unsigned warm = 0; warm < 8; ++warm) {
                JuicerProcess::root().ensure_bootstrap();
            }
            EXPECT_EQ(observationCount, (repetition + 1u) * 4u);
        }
        for (std::size_t index = 0; index < observationCount; ++index) {
            const auto& observation = observations[index];
            EXPECT_EQ(observation.length, observation.capacity);
            std::printf("copy family=%s length=%zu capacity=%zu prior_native_bytes=%zu\n", observation.family, observation.length, observation.capacity, observation.priorNativeBytes);
        }
        std::printf("native LUT bytes=%zu Mallett=%zu curve logical/capacity bytes=%zu; one new LUT overlaps the preceding native family during replacement\n",
                    2u * SpectralFixture::kLutCount * sizeof(float),
                    Spectral::gMallettBasis.data.capacity() * sizeof(float),
                    (Spectral::gXBar.lambda_nm.capacity() + Spectral::gXBar.linear.capacity() + Spectral::gYBar.lambda_nm.capacity() + Spectral::gYBar.linear.capacity() + Spectral::gZBar.lambda_nm.capacity() + Spectral::gZBar.linear.capacity()) * sizeof(float));
    }

    TEST(SpectralBootstrap, PlatformNativeRootsOpenActualSpectralFiles) {
#if defined(_WIN32)
        const auto root = source_root(fs::path(L"spectral-\u00e5-\u6f22"));
#else
        std::array<char, 64> pattern{};
        const auto prefix = (fs::temp_directory_path() / "juicer-spectral-XXXXXX").native();
        ASSERT_LT(prefix.size(), pattern.size());
        std::copy(prefix.begin(), prefix.end(), pattern.begin());
        const char* created = mkdtemp(pattern.data());
        ASSERT_NE(created, nullptr);
        const fs::path nativeTemporary(created);
        const auto root = source_root(nativeTemporary / fs::path(std::string("spectral-byte-") + static_cast<char>(0xff)));
#endif
        reset_observations();
        JuicerCuda::Owner owner;
        owner.create(root);
        JuicerProcess::root().ensure_bootstrap();
        EXPECT_TRUE(Spectral::hanatos_available());
        EXPECT_TRUE(Spectral::arctic_available());
        EXPECT_TRUE(Spectral::mallett_available());
        EXPECT_EQ(Spectral::gXBar.linear.size(), 81u);
        EXPECT_EQ(Spectral::gHanSpectra.assetHash, UINT64_C(0x81ebefd4e4cc9926));
        expect_no_cuda();
        EXPECT_EQ(owner.close().category, FJ_STATUS_SUCCESS);
#if !defined(_WIN32)
        fs::remove_all(nativeTemporary);
#endif
    }
} // namespace

namespace JuicerAssets::SpectralTest {
    void lut_view(const char* family, FjSpectraLutView& view) {
        acquisitions.emplace_back(family);
        borrowedLut = view.samples.data;
        if (copyBarrier && std::strcmp(family, "hanatos") == 0) {
            copyBarrier->arrive_and_wait();
            copyBarrier->arrive_and_wait();
        }
        if (std::strcmp(family, faultFamily) != 0) {
            return;
        }
        const auto fault = std::exchange(viewFault, 0u);
        if (fault == 1) {
            view.samples.count = 1;
        }
        if (fault == 2) {
            view.samples.data = nullptr;
        }
        if (fault == 3) {
            view.samples.data = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(view.samples.data) + 1u);
        }
        if (fault == 4) {
            view.asset_hash = 0;
        }
    }
    void mallett_view(FjFloatSpan& view) {
        acquisitions.emplace_back("mallett");
        if (std::strcmp(faultFamily, "mallett") != 0) {
            return;
        }
        const auto fault = std::exchange(viewFault, 0u);
        if (fault == 1) {
            view.count = 1;
        }
        if (fault == 2) {
            view.data = nullptr;
        }
        if (fault == 3) {
            view.data = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(view.data) + 1u);
        }
        if (fault == 4) {
            view.count = std::numeric_limits<std::size_t>::max();
        }
    }
    void cmf_view(FjFloatSpan& view) {
        acquisitions.emplace_back("cmf");
        if (std::strcmp(faultFamily, "cmf") != 0) {
            return;
        }
        const auto fault = std::exchange(viewFault, 0u);
        if (fault == 1) {
            view.count = 1;
        }
        if (fault == 2) {
            view.data = nullptr;
        }
        if (fault == 3) {
            view.data = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(view.data) + 1u);
        }
        if (fault == 4) {
            view.count = std::numeric_limits<std::size_t>::max();
        }
    }
    void copy_allocated(const char* family, std::size_t capacity) {
        (void)capacity;
        if (std::strcmp(family, faultFamily) == 0 && allocationFault != 0 && --allocationFault == 0) {
            throw std::bad_alloc();
        }
    }
    void copy_complete(const char* family, std::size_t length, std::size_t capacity) {
        if (observationCount >= observations.size()) {
            throw std::runtime_error("spectral fixture observation capacity");
        }
        observations[observationCount++] = {family, length, capacity, native_bytes()};
    }
    void before_curve_allocation() {
        if (++curveAllocation == failCurve) {
            AllocationTest::failBytes = 81u * sizeof(float);
        }
    }
} // namespace JuicerAssets::SpectralTest

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 3) {
        return 2;
    }
    resources = fs::absolute(argv[1]);
    scratch = fs::absolute(argv[2]);
    fs::create_directories(scratch);
    acquisitions.reserve(32);
    const int result = RUN_ALL_TESTS();
    return result == 0 && fj_test_spectral_live_owners() == 0 && fj_test_assets_live_owners() == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "spectral fixture setup failed: %s\n", error.what());
    return 2;
} catch (...) {
    std::fputs("spectral fixture setup failed\n", stderr);
    return 2;
}
