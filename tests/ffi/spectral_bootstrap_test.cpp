#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "ProcessRuntime.h"
#include "gtest_cases.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "prepared_product_fixture.h"
#include "spectral_source_fixture.h"
#include "spectral_test_resources.h"
#include "test_asset_workspace.h"

namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
        static bool has_no_cuda(const Root& root) {
            return root._cudaContextResources.empty() && root._cudaDeviceLedgers.empty() && root._activeFramePreparations == 0;
        }
    };
} // namespace JuicerProcess::TestSupport

namespace {
    namespace fs = std::filesystem;
    fs::path resources;
    fs::path scratch;
    fs::path source_root(const fs::path& name) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(name.is_absolute() ? name : scratch / name);
        JuicerTestSupport::copy_reconstruction_assets(resources, root);
        return root;
    }
    fs::path consumer_root(const std::string& name) {
        const auto root = JuicerTestSupport::test_asset_workspaces.create(scratch / name);
        JuicerTestSupport::copy_direct_preparation_assets(resources, root);
        return root;
    }
    void expect_no_cuda(const JuicerProcess::ProcessRuntime& runtime) {
        EXPECT_TRUE(JuicerProcess::TestSupport::RootLifetimeObserver::has_no_cuda(
            JuicerCuda::borrow_root(runtime.borrow().cuda)));
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
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        const auto* assets = owner.borrow().assets;
        {
            // Explicit raw acquisitions observe source cache ownership, not production stage order.
            std::array<std::array<const float*, 4>, 8> backing{};
            std::array<std::exception_ptr, 8> failures{};
            std::vector<std::jthread> callers;
            callers.reserve(failures.size());
            for (std::size_t i = 0; i < failures.size(); ++i) {
                callers.emplace_back([&, i] {
                    try {
                        const JuicerTestSupport::CmfSource sharedCmf(assets);
                        const JuicerTestSupport::HanatosSource sharedHanatos(assets);
                        const JuicerTestSupport::ArcticSource sharedArctic(assets);
                        const JuicerTestSupport::MallettSource sharedBasis(assets);
                        // Retain only pointer identities; matching live holds below supply all reads.
                        backing[i] = {sharedCmf.rows().data, sharedHanatos.view().samples.data, sharedArctic.view().samples.data, sharedBasis.samples().data};
                    } catch (...) {
                        failures[i] = std::current_exception();
                    }
                });
            }
            for (auto& caller : callers) {
                caller.join();
            }
            for (const auto& failure : failures) {
                if (failure) {
                    std::rethrow_exception(failure);
                }
            }
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            const JuicerTestSupport::CmfSource cmf(assets);
            const JuicerTestSupport::HanatosSource hanatos(assets);
            const JuicerTestSupport::ArcticSource arctic(assets);
            const JuicerTestSupport::MallettSource basis(assets);
            const std::array<const float*, 4> retained{cmf.rows().data, hanatos.view().samples.data, arctic.view().samples.data, basis.samples().data};
            for (const auto& shared : backing) {
                EXPECT_EQ(shared, retained);
            }
            EXPECT_EQ(fj_test_spectral_live_owners(), 4u);
            EXPECT_EQ(hanatos.view().asset_hash, UINT64_C(0x81ebefd4e4cc9926));
            EXPECT_EQ(arctic.view().asset_hash, UINT64_C(0x9262ffb765e3289e));
            // Removing these source inputs is the existing cache-lifetime stimulus.
            fs::remove_all(root);
            EXPECT_EQ(fj_legacy_assets_release_cached_payloads(owner.borrow().assets, nullptr).category, FJ_STATUS_SUCCESS);
            for (unsigned i = 0; i < 8; ++i) {
                const JuicerTestSupport::CmfSource warmCmf(assets);
                const JuicerTestSupport::HanatosSource warmHanatos(assets);
                const JuicerTestSupport::ArcticSource warmArctic(assets);
                const JuicerTestSupport::MallettSource warmBasis(assets);
                EXPECT_EQ(warmCmf.rows().data, cmf.rows().data);
                EXPECT_EQ(warmHanatos.view().samples.data, hanatos.view().samples.data);
                EXPECT_EQ(warmArctic.view().samples.data, arctic.view().samples.data);
                EXPECT_EQ(warmBasis.samples().data, basis.samples().data);
            }
            expect_no_cuda(owner);
            EXPECT_EQ(owner.close().category, FJ_STATUS_SUCCESS);
            EXPECT_EQ(fj_test_assets_live_owners(), 0u);
            EXPECT_EQ(fj_test_spectral_live_owners(), 4u);
            EXPECT_EQ(hanatos.view().samples.data[0], 5.84375f);
            EXPECT_EQ(basis.samples().data[basis.samples().count - 1u], std::bit_cast<float>(UINT32_C(0x3eaa8d89)));
            EXPECT_EQ(cmf.x().count, 81u);
            EXPECT_EQ(cmf.rows().count, 324u);
        }
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
    }

    TEST(SpectralBootstrap, SequentialOwnerCmfFailuresClearAllCurves) {
        JuicerTests::GtestCohort cmf_failures_cases("cmf-failures", 4u);
        std::size_t cmf_failures_index = 0;
        for (const char* failure : {"missing", "empty", "axis", "token"}) {
            auto cmf_failures_case = cmf_failures_cases.case_event(std::to_string(cmf_failures_index++));
            SCOPED_TRACE(failure);
            {
                JuicerProcess::ProcessRuntime first;
                first.create(resources);
                const JuicerTestSupport::CmfSource previousCmf(first.borrow().assets);
                ASSERT_EQ(previousCmf.x().count, 81u);
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
            JuicerProcess::ProcessRuntime second;
            second.create(root);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);

            std::optional<PreparedState> product;
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            const auto* assets = second.borrow().assets;
            const auto result = prepare_state(assets, controls(1), product, error);
            const bool rustSourceSucceeds = std::strcmp(failure, "axis") == 0;
            if (rustSourceSucceeds) {
                ASSERT_EQ(result.category, FJ_STATUS_SUCCESS);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                ASSERT_TRUE(product);
                FjPreparedStateView view{};
                error = {diagnostic.data(), diagnostic.size(), 0};
                ASSERT_EQ(product->view(view, error).category, FJ_STATUS_SUCCESS);
                JuicerTestSupport::CmfSource cmf(assets);
                const std::array<FjFloatSpan, 3> source{{cmf.x(), cmf.y(), cmf.z()}};
                for (std::size_t channel = 0; channel < source.size(); ++channel) {
                    ASSERT_EQ(view.exposure_tables.observer_xyz[channel].count, 81u);
                    EXPECT_EQ(std::memcmp(
                                  view.exposure_tables.observer_xyz[channel].data,
                                  source[channel].data,
                                  81u * sizeof(float)),
                              0);
                }
            } else {
                EXPECT_EQ(result.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                EXPECT_FALSE(product);
                EXPECT_GT(error.length, 0u);
            }
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
            expect_no_cuda(second);

            cmf_failures_case.passed();
        }
    }

    TEST(SpectralBootstrap, IndependentMissingFamiliesAndSelectedConsumers) {
        JuicerTests::GtestCohort missing_family_method_cases("missing-family-method", 9u);
        std::size_t missing_family_method_index = 0;
        for (const char* missing : {SpectralFixture::kHanatos, SpectralFixture::kArctic, SpectralFixture::kMallett}) {
            SCOPED_TRACE(missing);
            const auto root = consumer_root(std::string("missing-") + missing);
            fs::remove(SpectralFixture::source_path(root, missing));
            JuicerProcess::ProcessRuntime owner;
            owner.create(root);
            const unsigned absent = missing == SpectralFixture::kHanatos ? 0u : (missing == SpectralFixture::kArctic ? 2u : 1u);
            for (unsigned method : {0u, 1u, 2u}) {
                auto missing_family_method_case = missing_family_method_cases.case_event(std::to_string(missing_family_method_index++));
                bool success = false;
                std::string diagnostic;
                if (method != absent) {
                    auto state = JuicerTestSupport::prepare_product(owner.borrow().assets, controls(method));
                    std::array<char, 2048> viewText{};
                    FjErrorBuffer viewError{viewText.data(), viewText.size(), 0};
                    FjPreparedStateView view{};
                    const auto viewed = state.view(view, viewError);
                    EXPECT_EQ(viewed.category, FJ_STATUS_SUCCESS);
                    success = viewed.category == FJ_STATUS_SUCCESS;
                } else {
                    std::optional<PreparedState> state;
                    std::array<char, 2048> errorText{};
                    FjErrorBuffer error{errorText.data(), errorText.size(), 0};
                    const auto result = prepare_state(owner.borrow().assets, controls(method), state, error);
                    success = result.category == FJ_STATUS_SUCCESS;
                    diagnostic.assign(errorText.data(), error.length);
                }
                EXPECT_EQ(success, method != absent) << "method=" << method << ' ' << diagnostic;
                if (method == absent) {
                    EXPECT_FALSE(diagnostic.empty());
                }

                missing_family_method_case.passed();
            }
            expect_no_cuda(owner);
        }
    }

    TEST(SpectralBootstrap, NonfiniteHanatosReachesSelectedComputation) {
        const auto root = consumer_root("nonfinite-hanatos");
        SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount}, {0x7fc12345u});
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        const FjAssets* assets = owner.borrow().assets;
        const JuicerTestSupport::HanatosSource source(assets);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(source.view().samples.data[0]), 0x7fc12345u);
        EXPECT_NE(source.view().asset_hash, 0u);
        std::array<float, 243> sensitivity{};
        sensitivity.fill(1.0f);
        std::array<float, 81> illuminant{};
        illuminant.fill(1.0f);
        const FjFilmTcLutInput input{
            source.view().samples,
            {sensitivity.data(), sensitivity.size()},
            {illuminant.data(), illuminant.size()},
            {0.9504559f, 1.0f, 1.0890578f},
            0.0f,
            0,
            0,
            {},
            1,
            0,
            {},
            {}};
        FjOwnedFilmTcLut result{};
        std::array<char, 2048> tcText{};
        FjErrorBuffer tcError{tcText.data(), tcText.size(), 0};
        const auto tcStatus = fj_test_reconstruction_tc_lut(&input, &result, &tcError);
        EXPECT_EQ(tcStatus.category, FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(tcStatus.api, FJ_API_NONE);
        EXPECT_EQ(tcStatus.native_code, 0);
        std::string diagnostic(tcText.data(), tcError.length);
        EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
        EXPECT_EQ(result.samples.data, nullptr);
        EXPECT_EQ(result.samples.count, 0u);
        EXPECT_EQ(result.capacity, 0u);
        EXPECT_EQ(fj_test_reconstruction_release_tc_lut(&result).category, FJ_STATUS_SUCCESS);
        std::optional<PreparedState> product;
        std::array<char, 2048> errorText{};
        FjErrorBuffer error{errorText.data(), errorText.size(), 0};
        const auto preparedStatus = prepare_state(assets, controls(0), product, error);
        diagnostic.assign(errorText.data(), error.length);
        EXPECT_NE(preparedStatus.category, FJ_STATUS_SUCCESS);
        EXPECT_FALSE(product);
        EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
        expect_no_cuda(owner);
    }

    TEST(SpectralBootstrap, SupportedWidthsChangedExtremaAndSignedSpecials) {
        const std::vector<std::uint32_t> decoded{0u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x7f800000u, 0xff800000u, 0x7fc46000u, 0xffc6a000u};
        JuicerTests::GtestCohort source_widths_cases("source-widths", 3u);
        std::size_t source_widths_index = 0;
        for (unsigned width : {2u, 4u, 8u}) {
            auto source_widths_case = source_widths_cases.case_event(std::to_string(source_widths_index++));
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
            JuicerProcess::ProcessRuntime owner;
            owner.create(root);
            const FjAssets* assets = owner.borrow().assets;
            const JuicerTestSupport::HanatosSource hanatos(assets);
            const JuicerTestSupport::ArcticSource arctic(assets);
            const JuicerTestSupport::MallettSource mallettSource(assets);
            for (std::size_t index = 0; index < decoded.size(); ++index) {
                EXPECT_EQ(std::bit_cast<std::uint32_t>(hanatos.view().samples.data[index]), decoded[index]);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(arctic.view().samples.data[index]), decoded[index]);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(mallettSource.samples().data[index]), decoded[index]);
            }
            // Independent FNV-1a stream: zero signs -> 0, NaNs -> 0x7fc00000,
            // other decoded bits unchanged; eight authored samples then ones.
            EXPECT_EQ(hanatos.view().asset_hash, UINT64_C(0x4e766e5978ce36d8));
            EXPECT_EQ(hanatos.view().asset_hash, arctic.view().asset_hash);
            std::array<float, 81> leafIlluminant{};
            leafIlluminant.fill(1.0f);
            std::array<std::array<float, 3>, 81> sensitivity{};
            for (auto& row : sensitivity)
                row.fill(1.0f);
            const FjMallettRawInput request{{1.0f, 1.0f, 1.0f}, mallettSource.samples(), {leafIlluminant.data(), 81}, {sensitivity.front().data(), 243}};
            float exposure[3]{};
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            EXPECT_EQ(fj_test_exposure_mallett_raw(&request, exposure, &error).category, FJ_STATUS_SUCCESS);
            // Row zero contributes 1, rows one/two have nonfinite SPD and are
            // skipped by the retained computation; the remaining 78 contribute 3.
            for (float channel : exposure) {
                EXPECT_EQ(channel, 235.0f);
            }
            const FjFilmTcLutInput tcInput{
                arctic.view().samples,
                {sensitivity.front().data(), 243},
                {leafIlluminant.data(), leafIlluminant.size()},
                {0.950455f, 1.0f, 1.089058f},
                0.0f,
                2,
                0,
                {},
                1,
                0,
                {},
                {}};
            FjOwnedFilmTcLut integrated{};
            error = {bytes.data(), bytes.size(), 0};
            const auto tcStatus = fj_test_reconstruction_tc_lut(&tcInput, &integrated, &error);
            EXPECT_EQ(tcStatus.category, FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(tcStatus.api, FJ_API_NONE);
            EXPECT_EQ(tcStatus.native_code, 0);
            const std::string diagnostic(bytes.data(), error.length);
            EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
            EXPECT_EQ(integrated.samples.data, nullptr);
            EXPECT_EQ(integrated.samples.count, 0u);
            EXPECT_EQ(integrated.capacity, 0u);
            EXPECT_EQ(fj_test_reconstruction_release_tc_lut(&integrated).category, FJ_STATUS_SUCCESS);
            expect_no_cuda(owner);

            source_widths_case.passed();
        }
    }

    TEST(SpectralBootstrap, AlteredFiniteSourcesAndOverflowReachComputation) {
        JuicerTests::GtestCohort source_overflow_cases("source-overflow", 2u);
        std::size_t source_overflow_index = 0;
        for (bool overflow : {false, true}) {
            auto source_overflow_case = source_overflow_cases.case_event(std::to_string(source_overflow_index++));
            const auto root = consumer_root(overflow ? "narrowing-overflow" : "changed-finite");
            if (overflow) {
                SpectralFixture::write_npy(root, SpectralFixture::kHanatos, {.elementBytes = 8, .sampleCount = SpectralFixture::kLutCount}, {std::bit_cast<std::uint64_t>(1e300)});
            } else {
                SpectralFixture::write_npy(root, SpectralFixture::kArctic, {.elementBytes = 4, .sampleCount = SpectralFixture::kLutCount}, {0x42c80000u, 0xc2480000u});
            }
            JuicerProcess::ProcessRuntime owner;
            owner.create(root);
            const FjAssets* assets = owner.borrow().assets;
            const JuicerTestSupport::HanatosSource hanatos(assets);
            const JuicerTestSupport::ArcticSource arctic(assets);
            if (overflow) {
                EXPECT_TRUE(std::isinf(hanatos.view().samples.data[0]));
                std::optional<PreparedState> product;
                std::array<char, 2048> errorText{};
                FjErrorBuffer error{errorText.data(), errorText.size(), 0};
                const auto result = prepare_state(owner.borrow().assets, controls(0), product, error);
                const std::string diagnostic(errorText.data(), error.length);
                EXPECT_NE(result.category, FJ_STATUS_SUCCESS);
                EXPECT_FALSE(product);
                EXPECT_NE(diagnostic.find("finite_integrated_samples"), std::string::npos) << diagnostic;
            } else {
                EXPECT_EQ(arctic.view().samples.data[0], 100.0f);
                EXPECT_EQ(arctic.view().samples.data[1], -50.0f);
                EXPECT_NE(arctic.view().asset_hash, UINT64_C(0x9262ffb765e3289e));
                auto product = JuicerTestSupport::prepare_product(owner.borrow().assets, controls(2));
                std::array<char, 2048> errorText{};
                FjErrorBuffer error{errorText.data(), errorText.size(), 0};
                FjPreparedStateView view{};
                EXPECT_EQ(product.view(view, error).category, FJ_STATUS_SUCCESS);
            }
            expect_no_cuda(owner);

            source_overflow_case.passed();
        }
    }

    TEST(SpectralBootstrap, MalformedSourcesFailWithoutTranspositionOrPartialPublication) {
        JuicerTests::GtestCohort malformed_sources_cases("malformed-sources", 6u);
        std::size_t malformed_sources_index = 0;
        for (unsigned failure = 0; failure < 6; ++failure) {
            auto malformed_sources_case = malformed_sources_cases.case_event(std::to_string(malformed_sources_index++));
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
            JuicerProcess::ProcessRuntime owner;
            owner.create(root);
            const FjAssets* assets = owner.borrow().assets;
            if (failure == 0) {
                FjMallettBasis* rejected = nullptr;
                const auto result = fj_legacy_mallett_acquire(assets, &rejected, nullptr);
                EXPECT_EQ(result.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                EXPECT_EQ(rejected, nullptr);
                EXPECT_EQ(fj_legacy_mallett_release(rejected, nullptr).category, FJ_STATUS_SUCCESS);
                const JuicerTestSupport::HanatosSource hanatos(assets);
                const JuicerTestSupport::ArcticSource arctic(assets);
                EXPECT_NE(hanatos.view().asset_hash, 0u);
                EXPECT_NE(arctic.view().asset_hash, 0u);
            } else if (failure == 1) {
                FjSpectraLut* rejected = nullptr;
                const auto result = fj_legacy_hanatos_acquire(assets, &rejected, nullptr);
                EXPECT_EQ(result.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                EXPECT_EQ(rejected, nullptr);
                EXPECT_EQ(fj_legacy_spectra_lut_release(rejected, nullptr).category, FJ_STATUS_SUCCESS);
                const JuicerTestSupport::ArcticSource arctic(assets);
                const JuicerTestSupport::MallettSource basis(assets);
                EXPECT_NE(arctic.view().asset_hash, 0u);
                EXPECT_EQ(basis.samples().count, 243u);
            } else {
                FjSpectraLut* rejected = nullptr;
                const auto result = fj_legacy_arctic_acquire(assets, &rejected, nullptr);
                EXPECT_EQ(result.category, FJ_STATUS_PREPARATION_FAILURE);
                EXPECT_EQ(result.api, FJ_API_NONE);
                EXPECT_EQ(result.native_code, 0);
                EXPECT_EQ(rejected, nullptr);
                EXPECT_EQ(fj_legacy_spectra_lut_release(rejected, nullptr).category, FJ_STATUS_SUCCESS);
                const JuicerTestSupport::HanatosSource hanatos(assets);
                const JuicerTestSupport::MallettSource basis(assets);
                EXPECT_NE(hanatos.view().asset_hash, 0u);
                EXPECT_EQ(basis.samples().count, 243u);
            }
            expect_no_cuda(owner);
            EXPECT_EQ(fj_test_spectral_live_owners(), 0u);

            malformed_sources_case.passed();
        }
    }

    TEST(SpectralBootstrap, PlatformNativeRootsOpenActualSpectralFiles) {
#if defined(_WIN32)
        const auto root = source_root(fs::path(L"spectral-\u00e5-\u6f22"));
#else
        const auto root = source_root(fs::temp_directory_path() / fs::path(std::string("juicer-spectral-byte-") + static_cast<char>(0xff)));
#endif
        JuicerProcess::ProcessRuntime owner;
        owner.create(root);
        {
            const FjAssets* assets = owner.borrow().assets;
            const JuicerTestSupport::HanatosSource hanatos(assets);
            const JuicerTestSupport::ArcticSource arctic(assets);
            const JuicerTestSupport::MallettSource basis(assets);
            const JuicerTestSupport::CmfSource cmf(assets);
            EXPECT_EQ(hanatos.view().asset_hash, UINT64_C(0x81ebefd4e4cc9926));
            EXPECT_EQ(arctic.view().asset_hash, UINT64_C(0x9262ffb765e3289e));
            EXPECT_EQ(basis.samples().count, 243u);
            EXPECT_EQ(cmf.x().count, 81u);
            expect_no_cuda(owner);
        }
        EXPECT_EQ(owner.close().category, FJ_STATUS_SUCCESS);
        EXPECT_EQ(fj_test_spectral_live_owners(), 0u);
        EXPECT_EQ(fj_test_assets_live_owners(), 0u);
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
