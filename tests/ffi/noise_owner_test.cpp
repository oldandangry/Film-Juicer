#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "RustAssetBridge.h"
#include "juicer_cuda_owner.h"
#include "juicer_legacy_api.h"
#include "juicer_test_api.h"

namespace {
    namespace fs = std::filesystem;
    fs::path resources;
    fs::path scratch;
    std::array<unsigned, 3> operations{};
    bool gateViolation = false;
    bool corruptView = false;

    void expect_status(FjStatus result, std::uint32_t category) {
        EXPECT_EQ(result.category, category);
        EXPECT_EQ(result.api, FJ_API_NONE);
        EXPECT_EQ(result.native_code, 0);
    }
    struct Assets {
        FjAssets* owner = nullptr;
        explicit Assets(const fs::path& root) {
            const JuicerAssets::NativePathArgument path(root);
            expect_status(fj_legacy_assets_create(path.view(), &owner, nullptr), FJ_STATUS_SUCCESS);
        }
        ~Assets() {
            (void)fj_legacy_assets_destroy(owner, nullptr);
        }
        Assets(const Assets&) = delete;
        Assets& operator=(const Assets&) = delete;
    };
    struct NoiseRelease {
        void operator()(FjNoise* owner) const noexcept {
            (void)fj_legacy_noise_release(owner, nullptr);
        }
    };
    using Noise = std::unique_ptr<FjNoise, NoiseRelease>;
    Noise acquire(const Assets& assets) {
        FjNoise* owner = nullptr;
        expect_status(fj_legacy_noise_acquire(assets.owner, &owner, nullptr), FJ_STATUS_SUCCESS);
        return Noise(owner);
    }
    FjStaticNoise view(const Noise& owner) {
        FjStaticNoise result{};
        expect_status(fj_legacy_noise_view(owner.get(), &result, nullptr), FJ_STATUS_SUCCESS);
        return result;
    }
    void expect_empty(const FjStaticNoise& view) {
        EXPECT_EQ(view.stbn.data, nullptr);
        EXPECT_EQ(view.stbn.count, 0u);
        EXPECT_EQ(view.wang_tiles.data, nullptr);
        EXPECT_EQ(view.wang_tiles.count, 0u);
        EXPECT_EQ(view.wang_lut.data, nullptr);
        EXPECT_EQ(view.wang_lut.count, 0u);
        EXPECT_EQ(view.stbn_width | view.stbn_height | view.stbn_frames | view.wang_width | view.wang_height | view.wang_colors, 0);
        EXPECT_EQ(view.wang_tile_count, 0u);
    }
    void copy_noise(const fs::path& root) {
        fs::create_directories(root / "Noise/Wang");
        for (const auto* file : {"Noise/stbn_scalar_512x512x256_u8.bin", "Noise/Wang/wang_tiles_256x256x16_u8.bin", "Noise/Wang/tiles.json"}) {
            fs::copy_file(resources / file, root / file, fs::copy_options::overwrite_existing);
        }
    }

    std::uint64_t fingerprint(FjByteSpan bytes) {
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (std::size_t index = 0; index < bytes.count; ++index) {
            hash = (hash ^ bytes.data[index]) * UINT64_C(1099511628211);
        }
        return hash;
    }

    TEST(NoiseOwner, CompleteSharedViewsSurviveCacheAndAssetsDestruction) {
        const auto before = fj_test_live_noise_owners();
        Noise first;
        Noise second;
        FjStaticNoise original{};
        {
            const Assets assets(resources);
            first = acquire(assets);
            second = acquire(assets);
            original = view(first);
            const auto repeated = view(second);
            EXPECT_EQ(original.stbn.data, repeated.stbn.data);
            EXPECT_EQ(original.wang_tiles.data, repeated.wang_tiles.data);
            EXPECT_EQ(original.wang_lut.data, repeated.wang_lut.data);
            EXPECT_EQ(original.stbn.count, 67108864u);
            EXPECT_EQ(original.wang_tiles.count, 1048576u);
            EXPECT_EQ(original.wang_lut.count, 16u);
            EXPECT_EQ(original.stbn_width, 512);
            EXPECT_EQ(original.stbn_height, 512);
            EXPECT_EQ(original.stbn_frames, 256);
            EXPECT_EQ(original.wang_width, 256);
            EXPECT_EQ(original.wang_height, 256);
            EXPECT_EQ(original.wang_tile_count, 16u);
            EXPECT_EQ(original.wang_colors, 2);
            EXPECT_EQ(fingerprint(original.stbn), UINT64_C(16781862639737418621));
            EXPECT_EQ(fingerprint(original.wang_tiles), UINT64_C(10252691589253770981));
            constexpr std::array<std::uint8_t, 16> lut{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
            EXPECT_TRUE(std::equal(lut.begin(), lut.end(), original.wang_lut.data));
            expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        }
        EXPECT_EQ(view(first).stbn.data, original.stbn.data);
        std::array<FjStaticNoise, 2> reads{};
        std::thread a([&] {
            reads[0] = view(first);
        });
        std::thread b([&] {
            reads[1] = view(second);
        });
        a.join();
        b.join();
        EXPECT_EQ(reads[0].wang_lut.data, reads[1].wang_lut.data);
        first.reset();
        EXPECT_EQ(view(second).stbn.data, original.stbn.data);
        second.reset();
        EXPECT_EQ(fj_test_live_noise_owners(), before);
    }

    TEST(NoiseOwner, ClearedOutputsDiagnosticsAndConsumingRelease) {
        FjNoise* owner = reinterpret_cast<FjNoise*>(1);
        std::array<char, 8> message{};
        FjErrorBuffer error{message.data(), message.size(), 999};
        expect_status(fj_legacy_noise_acquire(nullptr, &owner, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(owner, nullptr);
        EXPECT_EQ(error.length, message.size() - 1);
        EXPECT_EQ(message.back(), '\0');
        expect_status(fj_legacy_noise_acquire(nullptr, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        FjStaticNoise output{};
        output.stbn.count = 999;
        output.wang_width = 8;
        expect_status(fj_legacy_noise_view(nullptr, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(output);
        expect_status(fj_legacy_noise_view(nullptr, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const Assets assets(resources);
        FjErrorBuffer malformed{nullptr, 1, 999};
        expect_status(fj_legacy_noise_acquire(assets.owner, &owner, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(owner, nullptr);
        auto acquired = acquire(assets);
        output = view(acquired);
        expect_status(fj_legacy_noise_view(acquired.get(), &output, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(output);
        const auto before = fj_test_live_noise_owners();
        expect_status(fj_legacy_noise_release(acquired.release(), &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(fj_test_live_noise_owners(), before - 1);
        expect_status(fj_legacy_noise_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_legacy_noise_release(nullptr, &malformed), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer zero{nullptr, 0, 999};
        expect_status(fj_legacy_noise_release(nullptr, &zero), FJ_STATUS_SUCCESS);
        EXPECT_EQ(zero.length, 0u);
    }

    TEST(NoiseOwner, ContainedPanicAndFailedNativeConstructionReleaseOnce) {
        const Assets assets(resources);
        FjNoise* output = reinterpret_cast<FjNoise*>(1);
        fj_test_noise_fault(1);
        expect_status(fj_legacy_noise_acquire(assets.owner, &output, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(output, nullptr);
        fj_test_noise_fault(0);
        auto owner = acquire(assets);
        FjStaticNoise result{};
        result.stbn.count = 99;
        fj_test_noise_fault(2);
        expect_status(fj_legacy_noise_view(owner.get(), &result, nullptr), FJ_STATUS_INTERNAL_FAILURE);
        expect_empty(result);
        fj_test_noise_fault(3);
        const auto before = fj_test_live_noise_owners();
        expect_status(fj_legacy_noise_release(owner.release(), nullptr), FJ_STATUS_INTERNAL_FAILURE);
        EXPECT_EQ(fj_test_live_noise_owners(), before - 1);
        fj_test_noise_fault(0);
        JuicerAssets::AssetBridge bridge(resources);
        corruptView = true;
        EXPECT_THROW((void)bridge.noise(), JuicerCuda::ExecutionFailure);
        corruptView = false;
        EXPECT_EQ(fj_test_live_noise_owners(), before - 1);
        EXPECT_FALSE(gateViolation);
    }

    TEST(NoiseOwner, OrdinarySourceErrorStaysCachedUntilRelease) {
        const auto root = scratch / "failure-cache";
        fs::remove_all(root);
        fs::create_directories(root);
        const Assets assets(root);
        FjNoise* owner = nullptr;
        expect_status(fj_legacy_noise_acquire(assets.owner, &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        copy_noise(root);
        expect_status(fj_legacy_noise_acquire(assets.owner, &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        expect_status(fj_legacy_assets_release_cached_payloads(assets.owner, nullptr), FJ_STATUS_SUCCESS);
        auto success = acquire(assets);
        EXPECT_NE(success, nullptr);
    }

    TEST(NoiseOwner, NativePathAndFacadeUseCompleteProducer) {
        const auto root = scratch / fs::path(std::u8string(u8"noise-å-film"));
        copy_noise(root);
        {
            const Assets assets(root);
            auto owner = acquire(assets);
            EXPECT_EQ(view(owner).stbn.count, 67108864u);
        }
        const auto utf8 = resources.u8string();
        const FjStringView text{reinterpret_cast<const char*>(utf8.data()), utf8.size()};
        FjNoise* owner = nullptr;
        expect_status(fj_test_noise_acquire(text, &owner, nullptr), FJ_STATUS_SUCCESS);
        FjStaticNoise result{};
        expect_status(fj_test_noise_view(owner, &result, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(result.wang_lut.count, 16u);
        expect_status(fj_test_noise_release(owner, nullptr), FJ_STATUS_SUCCESS);
        owner = reinterpret_cast<FjNoise*>(1);
        expect_status(fj_test_noise_acquire({}, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(owner, nullptr);
        const char bad[] = {'x', '\0', 'y'};
        expect_status(fj_test_noise_acquire({bad, sizeof(bad)}, &owner, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(owner, nullptr);
    }

    TEST(NoiseOwner, MoveOnlyBorrowAndNonlockingReentryPrecheck) {
        static_assert(!std::is_default_constructible_v<JuicerAssets::NoiseSource>);
        static_assert(!std::is_copy_constructible_v<JuicerAssets::NoiseSource>);
        static_assert(std::is_nothrow_move_constructible_v<JuicerAssets::NoiseSource>);
        static_assert(!std::is_move_assignable_v<JuicerAssets::NoiseSource>);
        operations = {};
        {
            JuicerAssets::AssetBridge bridge(resources);
            auto source = bridge.noise();
            const auto* const address = source.view().stbn.data();
            auto moved = std::move(source);
            EXPECT_EQ(moved.view().stbn.data(), address);
            EXPECT_EQ(operations[0], 1u);
            EXPECT_EQ(operations[1], 1u);
            EXPECT_EQ(operations[2], 0u);
        }
        EXPECT_EQ(operations[2], 1u);
        EXPECT_FALSE(gateViolation);
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerCuda::check_native_call_admission(JuicerCuda::borrowed_owner());
        EXPECT_FALSE(JuicerCuda::calling_thread_native_gate_active());
        {
            JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
            EXPECT_TRUE(JuicerCuda::calling_thread_native_gate_active());
            EXPECT_THROW(JuicerCuda::check_native_call_admission(JuicerCuda::borrowed_owner()), JuicerCuda::ExecutionFailure);
        }
        EXPECT_FALSE(JuicerCuda::calling_thread_native_gate_active());
        expect_status(owner.close(), FJ_STATUS_SUCCESS);
        std::printf("NoiseSource=%zu FjStaticNoise=%zu StaticNoiseInput=%zu; shared source=68157456 bytes\n", sizeof(JuicerAssets::NoiseSource), sizeof(FjStaticNoise), sizeof(JuicerCuda::StaticNoiseInput));
    }
} // namespace

namespace JuicerAssets::NoiseTest {
    void observe(Operation operation) noexcept {
        ++operations[static_cast<unsigned>(operation)];
        gateViolation = gateViolation || JuicerCuda::calling_thread_native_gate_active();
    }
    void view(FjStaticNoise& view) {
        if (corruptView) {
            view.stbn.count = 1;
        }
    }
} // namespace JuicerAssets::NoiseTest

int main(int argc, char** argv) try {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 3) {
        return 2;
    }
    resources = fs::absolute(argv[1]);
    scratch = fs::absolute(argv[2]);
    fs::create_directories(scratch);
    const int result = RUN_ALL_TESTS();
    return result == 0 && fj_test_live_noise_owners() == 0 && !gateViolation ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "noise fixture setup failed: %s\n", error.what());
    return 2;
}
