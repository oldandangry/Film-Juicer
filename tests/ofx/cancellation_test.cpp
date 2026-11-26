#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../resource/observed_operations.h"
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
} // namespace
