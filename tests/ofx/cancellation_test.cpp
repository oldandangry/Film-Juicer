#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include "../resource/observed_operations.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "juicer_ofx_api.h"
#include "production_images.h"

namespace {
    constexpr ofx_test::Case kDirect{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 4};
    constexpr float kDestinationCanary = -9.F;

    class ImplicitStreamOfx : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            plugin = fj_ofx_plugin(0);
            ASSERT_NE(plugin, nullptr);
            plugin->setHost(host.binding());
            ASSERT_EQ(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr), kOfxStatOK);
            loaded = true;
            ASSERT_EQ(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args(), nullptr), kOfxStatOK);
            auto* const candidate = host.create_effect();
            ofx_test::author_parameters(host, candidate, kDirect);
            host.set_value(candidate, "CouplersActive", {0});
            host.set_value(candidate, "GlareActive", {0});
            ASSERT_EQ(plugin->mainEntry(kOfxActionCreateInstance, candidate, nullptr, nullptr), kOfxStatOK);
            effect = candidate;
            ASSERT_EQ(cudaMalloc(&images.source, bytes()), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&images.destination, bytes()), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(images.source, source.data(), bytes(), cudaMemcpyHostToDevice), cudaSuccess);
            ofx_test::configure(host, effect, kDirect, images);
            host.set_property(host.action_args(), kOfxImageEffectPropCudaStream, {});
        }

        void TearDown() override {
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            if (effect) {
                EXPECT_EQ(plugin->mainEntry(kOfxActionDestroyInstance, effect, nullptr, nullptr), kOfxStatOK);
            }
            if (loaded) {
                EXPECT_EQ(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr), kOfxStatOK);
            }
            EXPECT_EQ(cudaFree(images.destination), cudaSuccess);
            EXPECT_EQ(cudaFree(images.source), cudaSuccess);
        }

        std::size_t bytes() const {
            return source.size() * sizeof(float);
        }

        void render_and_read(OfxStatus expected, std::vector<float>& output) {
            ASSERT_EQ(cudaMemcpy(images.destination, canaries.data(), bytes(), cudaMemcpyHostToDevice), cudaSuccess);
            const auto sourceReleases = host.image_release_count(effect, "Source");
            const auto outputReleases = host.image_release_count(effect, "Output");
            EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), expected);
            EXPECT_EQ(host.image_release_count(effect, "Source"), sourceReleases + 1);
            EXPECT_EQ(host.image_release_count(effect, "Output"), outputReleases + 1);
            EXPECT_FALSE(host.callback_error());
            // The injected failure skips the real stream wait.
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(output.data(), images.destination, bytes(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_TRUE(ofx_test::canaries_preserved(canaries, output, kDirect));
            EXPECT_TRUE(ofx_test::alpha_preserved(source, output, kDirect));
        }

        ofx_test::InstanceHost host;
        OfxPlugin* plugin = nullptr;
        OfxImageEffectHandle effect = nullptr;
        ofx_test::ImageBuffers images;
        const std::vector<float> source = ofx_test::initial_source(kDirect);
        const std::vector<float> canaries = std::vector<float>(source.size(), kDestinationCanary);
        bool loaded = false;
    };

    TEST_F(ImplicitStreamOfx, CancellationBeforeNativeAdmissionPreservesThePublishedId) {
        std::vector<FjSubmission> selections;
        auto& observe = JuicerTest::calls.beforeRenderAdmission;
        observe = [&](FjCuda*, const FjCudaContext&, const FjSubmission& selected) {
            selections.push_back(selected);
        };

        std::vector<float> expected(source.size()), actual(source.size());
        render_and_read(kOfxStatOK, expected);
        unsigned queries = 0;
        (void)host.set_abort_callback([&] {
            return ++queries == 3 ? 1 : 0;
        });
        host.set_property(host.action_args(), kOfxPropTime, {1.0});
        const auto syncs = JuicerTest::calls.synchronizations;
        EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatOK);
        EXPECT_EQ(queries, 3U);
        EXPECT_EQ(JuicerTest::calls.synchronizations, syncs);
        (void)host.set_abort_callback({});
        host.set_property(host.action_args(), kOfxPropTime, {kDirect.time});
        render_and_read(kOfxStatOK, actual);
        observe = {};

        ASSERT_EQ(selections.size(), 3U);
        EXPECT_NE(selections[1].frame_token, selections[0].frame_token);
        EXPECT_NE(selections[1].submission_id, selections[0].submission_id);
        EXPECT_EQ(selections[2].frame_token, selections[0].frame_token);
        EXPECT_EQ(selections[2].submission_id, selections[0].submission_id);
        EXPECT_EQ(actual, expected);
    }

    enum class Pause : std::uint8_t {
        BeforeSelection,
        AfterSelection
    };

    class StaleOfxSelection : public ImplicitStreamOfx, public testing::WithParamInterface<Pause> {};

    TEST_P(StaleOfxSelection, LeavesTheSuccessorSubmissionAndResourcesIntact) {
        namespace Registry = JuicerCuda::ResourceManager;
        struct Selection {
            FjCudaContext context;
            FjSubmission submission;
        };
        auto& observe = JuicerTest::calls.beforeRenderAdmission;
        FjCudaRetirement idle{};
        FjCuda* native = nullptr;
        observe = [&](FjCuda* owner, const FjCudaContext& context, const FjSubmission&) {
            native = owner;
            idle = {context, 0, 0};
        };
        unsigned queries = 0;
        (void)host.set_abort_callback([&] {
            return ++queries == 3 ? 1 : 0;
        });
        EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatOK);
        (void)host.set_abort_callback({});
        observe = {};
        ASSERT_EQ(queries, 3U);
        ASSERT_NE(native, nullptr);
        ASSERT_NE(idle.context.epoch, 0U);

        const auto operations = [] {
            const auto& calls = JuicerTest::calls;
            return std::array{calls.copies, calls.eventRecords, calls.synchronizations, calls.scanErrorReadbacks};
        };
        const auto nextTransaction = [] {
            return Registry::global_state().nextTransactionId.load(std::memory_order_relaxed);
        };
        const bool beforeSelection = GetParam() == Pause::BeforeSelection;
        std::optional<Selection> stale, successor;
        std::vector<float> expected(source.size()), actual(source.size());
        bool renderingSuccessor = false;
        auto successorOperations = operations();
        auto successorNextTransaction = nextTransaction();
        const auto retireAndRenderSuccessor = [&] {
            ASSERT_EQ(fj_cuda_retire_context(native, &idle, nullptr).category, FJ_STATUS_SUCCESS);
            renderingSuccessor = true;
            render_and_read(kOfxStatOK, expected);
            renderingSuccessor = false;
            successorOperations = operations();
            successorNextTransaction = nextTransaction();
        };
        observe = [&](FjCuda* owner, const FjCudaContext& context, const FjSubmission& submission) {
            EXPECT_EQ(owner, native);
            if (renderingSuccessor) {
                successor = {context, submission};
            } else {
                stale = {context, submission};
                if (!beforeSelection) {
                    retireAndRenderSuccessor();
                }
            }
        };
        queries = 0;
        (void)host.set_abort_callback([&] {
            if (!renderingSuccessor && ++queries == 2 && beforeSelection) {
                // The second Rust poll follows inspection and precedes ID selection.
                retireAndRenderSuccessor();
            }
            return 0;
        });
        const auto sourceReleases = host.image_release_count(effect, "Source");
        const auto outputReleases = host.image_release_count(effect, "Output");
        EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatErrFatal);
        (void)host.set_abort_callback({});
        observe = {};

        ASSERT_TRUE(stale);
        ASSERT_TRUE(successor);
        EXPECT_EQ(queries, 2U);
        EXPECT_EQ(host.image_release_count(effect, "Source"), sourceReleases + 2);
        EXPECT_EQ(host.image_release_count(effect, "Output"), outputReleases + 2);
        EXPECT_FALSE(host.callback_error());
        EXPECT_EQ(stale->context.epoch, idle.context.epoch);
        EXPECT_NE(successor->context.epoch, idle.context.epoch);
        EXPECT_EQ(stale->context.context, successor->context.context);
        EXPECT_EQ(stale->submission.instance_token, successor->submission.instance_token);
        EXPECT_EQ(stale->submission.frame_token, successor->submission.frame_token);
        if (beforeSelection) {
            EXPECT_GT(stale->submission.submission_id, successor->submission.submission_id);
        } else {
            EXPECT_LT(stale->submission.submission_id, successor->submission.submission_id);
        }
        // Replay the retirement that removed A while B holds the same raw context.
        for (int replay = 0; replay < 2; ++replay) {
            EXPECT_EQ(fj_cuda_retire_context(native, &idle, nullptr).category, FJ_STATUS_PREPARATION_FAILURE);
        }
        EXPECT_EQ(operations(), successorOperations);
        EXPECT_EQ(nextTransaction(), successorNextTransaction);
        ASSERT_EQ(cudaMemcpy(actual.data(), images.destination, bytes(), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(actual, expected);

        std::optional<Selection> retry;
        observe = [&](FjCuda*, const FjCudaContext& context, const FjSubmission& submission) {
            retry = {context, submission};
        };
        render_and_read(kOfxStatOK, actual);
        observe = {};
        ASSERT_TRUE(retry);
        EXPECT_EQ(retry->context.epoch, successor->context.epoch);
        EXPECT_EQ(retry->submission.submission_id, successor->submission.submission_id);
        EXPECT_EQ(JuicerTest::calls.copies, successorOperations[0]);
        EXPECT_EQ(actual, expected);
    }

    INSTANTIATE_TEST_SUITE_P(
        Delay, StaleOfxSelection, testing::Values(Pause::BeforeSelection, Pause::AfterSelection), [](const testing::TestParamInfo<Pause>& info) {
            return info.param == Pause::BeforeSelection ? "BeforeSelection" : "AfterSelection";
        });

    TEST_F(ImplicitStreamOfx, ReportsFailedCompletionThroughTheProductionHost) {
        std::vector<float> expected(source.size()), actual(source.size());
        ASSERT_NO_FATAL_FAILURE(render_and_read(kOfxStatOK, expected));
        for (const auto& [fault, status] :
             std::array{std::pair{cudaSuccess, kOfxStatOK}, std::pair{cudaErrorInvalidValue, kOfxStatErrFatal}}) {
            SCOPED_TRACE(fault);
            unsigned queries = 0;
            (void)host.set_abort_callback([&] {
                if (++queries != 5) {
                    return 0;
                }
                JuicerTest::calls.nextSynchronization = fault;
                return 1;
            });
            const auto syncs = JuicerTest::calls.synchronizations;
            const auto records = JuicerTest::calls.eventRecords;
            render_and_read(status, actual);
            (void)host.set_abort_callback({});
            EXPECT_EQ(queries, 5U);
            EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
            EXPECT_EQ(JuicerTest::calls.eventRecords, records + 1);
            EXPECT_EQ(JuicerTest::calls.nextSynchronization, cudaSuccess);
            EXPECT_EQ(actual, expected);
        }
        ASSERT_NO_FATAL_FAILURE(render_and_read(kOfxStatOK, actual));
        EXPECT_EQ(actual, expected);
    }
    TEST_F(ImplicitStreamOfx, CompletionReentryCannotAdmitTheFailedEpoch) {
        std::vector<float> expected(source.size()), actual(source.size());
        ASSERT_NO_FATAL_FAILURE(render_and_read(kOfxStatOK, expected));
        unsigned queries = 0;
        (void)host.set_abort_callback([&] {
            if (++queries != 5) {
                return 0;
            }
            JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
            return 1;
        });
        unsigned messages = 0;
        (void)host.set_message_callback([&](std::string_view, std::string_view id, std::string_view) {
            EXPECT_EQ(id, "FilmJuicerCudaContextLoss");
            if (++messages == 1) {
                const auto input = host.image_release_count(effect, "Source");
                const auto output = host.image_release_count(effect, "Output");
                EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatErrFatal);
                EXPECT_EQ(host.image_release_count(effect, "Source"), input + 1);
                EXPECT_EQ(host.image_release_count(effect, "Output"), output + 1);
            }
            return kOfxStatOK;
        });
        const auto input = host.image_release_count(effect, "Source");
        const auto output = host.image_release_count(effect, "Output");
        EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatErrFatal);
        EXPECT_EQ(host.image_release_count(effect, "Source"), input + 2);
        EXPECT_EQ(host.image_release_count(effect, "Output"), output + 2);
        EXPECT_EQ(queries, 5U);
        EXPECT_EQ(messages, 1U);
        EXPECT_FALSE(host.callback_error());
        (void)host.set_abort_callback({});
        (void)host.set_message_callback({});
        ASSERT_NO_FATAL_FAILURE(render_and_read(kOfxStatOK, actual));
        EXPECT_EQ(actual, expected);
    }

    TEST_F(ImplicitStreamOfx, DeferredDirCompletionRejectsReentryAfterImplicitWaitLosesContext) {
        namespace Registry = JuicerCuda::ResourceManager;
        host.set_value(effect, "CouplersActive", {1});
        auto* const change = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "CouplersActive"});
        ASSERT_EQ(plugin->mainEntry(kOfxActionInstanceChanged, effect, change, nullptr), kOfxStatReplyDefault);

        std::vector<float> expected(source.size()), actual(source.size());
        const auto producerReadbacks = JuicerTest::calls.scanErrorReadbacks;
        JuicerTest::calls.nextScanDirFailure = true;
        render_and_read(kOfxStatOK, expected);
        ASSERT_FALSE(std::exchange(JuicerTest::calls.nextScanDirFailure, false));
        ASSERT_EQ(JuicerTest::calls.scanErrorReadbacks, producerReadbacks + 1);

        CUcontext context = nullptr;
        ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
        ASSERT_NE(context, nullptr);
        const Registry::DeviceContextKey key{0, context};
        const auto registered = [&] {
            std::vector<Registry::DeviceContextKey> keys;
            Registry::registry_snapshot_context_keys(keys);
            return std::ranges::find(keys, key) != keys.end();
        };
        ASSERT_TRUE(registered());
        const auto failedEpoch = Registry::registry_inspect_context(key);
        ASSERT_NE(failedEpoch, 0U);
        ASSERT_TRUE(Registry::registry_accepts_epoch(key, failedEpoch));

        unsigned queries = 0;
        (void)host.set_abort_callback([&] {
            ++queries;
            return 0;
        });
        const auto sourceReleases = host.image_release_count(effect, "Source");
        const auto outputReleases = host.image_release_count(effect, "Output");
        const auto syncs = JuicerTest::calls.synchronizations;
        const auto records = JuicerTest::calls.eventRecords;
        unsigned messages = 0;
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): The OFX host fixture fixes the callback argument order.
        (void)host.set_message_callback([&](std::string_view, std::string_view id, std::string_view diagnostic) {
            EXPECT_EQ(id, "FilmJuicerDeferredCudaFailure");
            EXPECT_EQ(diagnostic, cudaGetErrorString(cudaErrorContextIsDestroyed));
            if (++messages != 1) {
                return kOfxStatOK;
            }
            EXPECT_TRUE(registered());
            EXPECT_FALSE(Registry::registry_accepts_epoch(key, failedEpoch));
            EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
            EXPECT_EQ(JuicerTest::calls.eventRecords, records + 1);
            EXPECT_EQ(JuicerTest::calls.scanErrorReadbacks, producerReadbacks + 1);
            EXPECT_EQ(host.image_release_count(effect, "Source"), sourceReleases);
            EXPECT_EQ(host.image_release_count(effect, "Output"), outputReleases);

            const auto outerQueries = queries;
            EXPECT_GT(outerQueries, 0U);
            EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatErrFatal);
            EXPECT_EQ(queries, outerQueries);
            EXPECT_EQ(JuicerTest::calls.synchronizations, syncs + 1);
            EXPECT_EQ(JuicerTest::calls.eventRecords, records + 1);
            EXPECT_EQ(JuicerTest::calls.scanErrorReadbacks, producerReadbacks + 1);
            EXPECT_EQ(host.image_release_count(effect, "Source"), sourceReleases + 1);
            EXPECT_EQ(host.image_release_count(effect, "Output"), outputReleases + 1);
            EXPECT_TRUE(registered());
            return kOfxStatOK;
        });
        JuicerTest::calls.nextSynchronization = cudaErrorContextIsDestroyed;
        EXPECT_EQ(plugin->mainEntry(kOfxImageEffectActionRender, effect, host.action_args(), nullptr), kOfxStatErrFatal);
        (void)host.set_message_callback({});
        (void)host.set_abort_callback({});
        EXPECT_EQ(std::exchange(JuicerTest::calls.nextSynchronization, cudaSuccess), cudaSuccess);
        EXPECT_EQ(messages, 1U);
        EXPECT_EQ(host.image_release_count(effect, "Source"), sourceReleases + 2);
        EXPECT_EQ(host.image_release_count(effect, "Output"), outputReleases + 2);
        EXPECT_FALSE(host.callback_error());
        EXPECT_FALSE(registered());

        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        ASSERT_NO_FATAL_FAILURE(render_and_read(kOfxStatOK, actual));
        EXPECT_EQ(actual, expected);
        ASSERT_TRUE(registered());
        const auto recoveredEpoch = Registry::registry_inspect_context(key);
        EXPECT_NE(recoveredEpoch, failedEpoch);
        EXPECT_TRUE(Registry::registry_accepts_epoch(key, recoveredEpoch));
    }
} // namespace
