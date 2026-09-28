#include <array>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>

#include "gtest/gtest.h"

#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

namespace {
    using JuicerCuda::ResourceManager::ActiveDirScratch;
    using JuicerCuda::ResourceManager::DirScratchRequest;
    using JuicerCuda::ResourceManager::make_scratch_request_descriptor;
    using JuicerCuda::ResourceManager::scratch_request_descriptor_is_valid;
    using JuicerCuda::ResourceManager::ScratchRequestAttachments;
    using JuicerCuda::ResourceManager::ScratchRequestDescriptor;
    using JuicerCuda::ResourceManager::ScratchRequestExtent;
    using Spektrafilm::DirScratchTier;

    constexpr ScratchRequestExtent kExtent{1920, 1080};
    constexpr std::uint64_t kDescriptorHash = 0x12345678;
    constexpr ActiveDirScratch::Layout kFir{DirScratchTier::Tier1F, {3, 3, 1, 0}};
    constexpr ActiveDirScratch::Layout kYvv{DirScratchTier::Tier1IChannels, {3, 3, 3, 0}};
    constexpr ActiveDirScratch::Layout kCached{DirScratchTier::Tier2, {3, 3, 3, 3}};

    static_assert(!std::is_default_constructible_v<ActiveDirScratch>);
    static_assert(!std::is_aggregate_v<ActiveDirScratch>);
    static_assert(!std::is_constructible_v<DirScratchRequest, std::optional<ActiveDirScratch>>);
    static_assert(std::is_standard_layout_v<ScratchRequestDescriptor>);
    static_assert(std::is_trivially_copyable_v<ScratchRequestDescriptor>);

    TEST(ScratchRequest, InactiveDirHasNoDependentFields) {
        ScratchRequestAttachments attachments;
        attachments.needOptics = true;
        const auto request = make_scratch_request_descriptor(kExtent, std::monostate{}, attachments);
        ASSERT_TRUE(request);
        EXPECT_FALSE(request->needSpatialDir);
        EXPECT_EQ(request->spatialDirDescriptorHash, 0u);
        EXPECT_EQ(request->spatialDirScratchTier, DirScratchTier::Tier0);
        EXPECT_EQ(request->spatialDirTargetScratchTier, DirScratchTier::Tier0);
        EXPECT_EQ(request->spatialDirPlaneRoles.total_float_planes(), 0);
        EXPECT_EQ(request->spatialDirTargetPlaneRoles.total_float_planes(), 0);
        EXPECT_TRUE(scratch_request_descriptor_is_valid(*request));

        auto invalid = *request;
        invalid.spatialDirDescriptorHash = kDescriptorHash;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirScratchTier = DirScratchTier::Tier1F;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirPlaneRoles = kFir.roles;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirTargetScratchTier = DirScratchTier::Tier1F;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirTargetPlaneRoles = kFir.roles;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirPlaneRoles = {3, -3, 0, 0};
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
    }

    TEST(ScratchRequest, ActiveDirRequiresHashAndSourceStorage) {
        EXPECT_FALSE(ActiveDirScratch::create(0, kFir, kFir));
        EXPECT_FALSE(ActiveDirScratch::create(kDescriptorHash, {}, kFir));
    }

    TEST(ScratchRequest, ActiveDirRejectsMismatchedSourceAndTargetLayouts) {
        const std::array<ActiveDirScratch::Layout, 5> invalidLayouts{{{DirScratchTier::Tier1F, {3, 3, 3, 0}},
                                                                      {DirScratchTier::Tier1IChannels, {3, 3, 1, 3}},
                                                                      {DirScratchTier::Tier2, {3, 3, 3, 0}},
                                                                      {DirScratchTier::Unsupported, {3, 3, 3, 3}},
                                                                      {DirScratchTier::Tier0, {3, -3, 0, 0}}}};
        for (const auto& invalid : invalidLayouts) {
            EXPECT_FALSE(ActiveDirScratch::create(kDescriptorHash, invalid, kCached));
            EXPECT_FALSE(ActiveDirScratch::create(kDescriptorHash, kYvv, invalid));
        }
    }

    TEST(ScratchRequest, ScannerAliasRequiresOpticsAndActiveDir) {
        const auto active = ActiveDirScratch::create(kDescriptorHash, kFir, kFir);
        ASSERT_TRUE(active);
        ScratchRequestAttachments attachments;
        attachments.aliasScannerRgbFromSpatialDirFiltered = true;
        attachments.needSharedTmp = true;
        EXPECT_FALSE(make_scratch_request_descriptor(kExtent, *active, attachments));
        attachments.needOptics = true;
        EXPECT_FALSE(make_scratch_request_descriptor(kExtent, std::monostate{}, attachments));
        const auto request = make_scratch_request_descriptor(kExtent, *active, attachments);
        ASSERT_TRUE(request);
        auto invalid = *request;
        invalid.spatialDirPlaneRoles.filteredCorrectionPlanes = 1;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
        invalid = *request;
        invalid.spatialDirTargetPlaneRoles.cachedLogRawPlanes = 1;
        EXPECT_FALSE(scratch_request_descriptor_is_valid(invalid));
    }

    TEST(ScratchRequest, FrameExtentMustBePositive) {
        ScratchRequestAttachments attachments;
        attachments.needOptics = true;
        for (const ScratchRequestExtent extent : {ScratchRequestExtent{0, 1080}, {-1, 1080}, {1920, 0}, {1920, -1}}) {
            EXPECT_FALSE(make_scratch_request_descriptor(extent, std::monostate{}, attachments));
        }
    }

    TEST(ScratchRequest, GateAttachmentRequiresPositiveGateExtent) {
        ScratchRequestAttachments attachments;
        attachments.needOptics = true;
        attachments.needGateTransmittance = true;
        for (const ScratchRequestExtent extent : {ScratchRequestExtent{0, 720}, {-1, 720}, {1280, 0}, {1280, -1}}) {
            attachments.gateWidth = extent.requestedWidth;
            attachments.gateHeight = extent.requestedHeight;
            EXPECT_FALSE(make_scratch_request_descriptor(kExtent, std::monostate{}, attachments));
        }
        attachments.gateWidth = 1280;
        attachments.gateHeight = 720;
        const auto request = make_scratch_request_descriptor(kExtent, std::monostate{}, attachments);
        ASSERT_TRUE(request);
        EXPECT_EQ(request->gateWidth, 1280);
        EXPECT_EQ(request->gateHeight, 720);
        attachments.needGateTransmittance = false;
        const auto withoutGate = make_scratch_request_descriptor(kExtent, std::monostate{}, attachments);
        ASSERT_TRUE(withoutGate);
        EXPECT_EQ(withoutGate->gateWidth, 0);
        EXPECT_EQ(withoutGate->gateHeight, 0);
    }

    TEST(ScratchRequest, SharedTemporaryStorageRequiresAConsumer) {
        const auto active = ActiveDirScratch::create(kDescriptorHash, kYvv, kCached);
        ASSERT_TRUE(active);
        ScratchRequestAttachments attachments;
        EXPECT_FALSE(make_scratch_request_descriptor(kExtent, *active, attachments));
        attachments.needSharedTmp = true;
        EXPECT_TRUE(make_scratch_request_descriptor(kExtent, *active, attachments));
        EXPECT_FALSE(make_scratch_request_descriptor(kExtent, std::monostate{}, attachments));
        attachments.needOptics = true;
        EXPECT_TRUE(make_scratch_request_descriptor(kExtent, std::monostate{}, attachments));
    }

    TEST(ScratchRequest, OpticsAttachmentsAreRejectedWithoutOptics) {
        const auto active = ActiveDirScratch::create(kDescriptorHash, kFir, kFir);
        ASSERT_TRUE(active);
        const std::array<bool ScratchRequestAttachments::*, 7> opticsFlags{{&ScratchRequestAttachments::needBlurred,
                                                                            &ScratchRequestAttachments::needAux,
                                                                            &ScratchRequestAttachments::needGrainFrameUniforms,
                                                                            &ScratchRequestAttachments::needGrainLayerWork,
                                                                            &ScratchRequestAttachments::needGrainShared,
                                                                            &ScratchRequestAttachments::needGateTransmittance,
                                                                            &ScratchRequestAttachments::needFilmDustTransmittance}};
        for (auto flag : opticsFlags) {
            ScratchRequestAttachments attachments;
            attachments.needSharedTmp = true;
            attachments.gateWidth = 1280;
            attachments.gateHeight = 720;
            attachments.*flag = true;
            EXPECT_FALSE(make_scratch_request_descriptor(kExtent, *active, attachments));
        }
    }

    TEST(ScratchRequest, NoFamilySentinelRemainsDistinctFromConstructionFailure) {
        const ScratchRequestDescriptor absent{};
        EXPECT_FALSE(absent.has_any_family());
        EXPECT_EQ(absent.generation, 0u);
        EXPECT_FALSE(make_scratch_request_descriptor(kExtent, std::monostate{}, {}));
    }

    TEST(ScratchRequest, DirectAndPrintTierCompositionsKeepTheirGeneration) {
        struct Composition {
            const char* name = nullptr;
            ActiveDirScratch::Layout source;
            ActiveDirScratch::Layout target;
            bool aliasScanner = false;
            bool print = false;
            std::uint64_t generation = 0;
        };
        // Product characterization captured from the accepted 7415a8c constructor.
        // Each row names source/target storage and the direct/print attachment shape.
        const std::array<Composition, 8> compositions{{{"direct-fir", kFir, kFir, true, false, 13369519279501012274ull},
                                                       {"print-fir", kFir, kFir, false, true, 6993612232329871157ull},
                                                       {"direct-yvv-alias", kYvv, kYvv, true, false, 16398215074749439696ull},
                                                       {"print-yvv-cached", kYvv, kCached, false, true, 8834133510794582629ull},
                                                       {"direct-yvv-cached-alias", kCached, kCached, true, false, 10089080395616178954ull},
                                                       {"print-yvv-single-temp", {DirScratchTier::Tier1IChannels, {3, 3, 1, 0}}, {DirScratchTier::Tier2, {3, 3, 1, 3}}, false, true, 12556773535806495577ull},
                                                       {"direct-yvv-streamed", {DirScratchTier::Tier1IChannels, {1, 3, 1, 0}}, {DirScratchTier::Tier1IChannels, {1, 3, 1, 0}}, true, false, 13344895294117419568ull},
                                                       {"print-yvv-bg-cache", {DirScratchTier::Tier2, {3, 3, 3, 2}}, {DirScratchTier::Tier2, {3, 3, 3, 2}}, true, true, 15096511039551998049ull}}};
        for (const auto& composition : compositions) {
            SCOPED_TRACE(composition.name);
            const auto active = ActiveDirScratch::create(kDescriptorHash, composition.source, composition.target);
            ASSERT_TRUE(active);
            ScratchRequestAttachments attachments;
            attachments.needOptics = true;
            attachments.needSharedTmp = true;
            attachments.aliasScannerRgbFromSpatialDirFiltered = composition.aliasScanner;
            attachments.needBlurred = composition.print;
            attachments.needAux = !composition.aliasScanner;
            attachments.needGrainFrameUniforms = true;
            attachments.needGrainLayerWork = !composition.aliasScanner;
            attachments.needGrainShared = !composition.aliasScanner;
            attachments.needGateTransmittance = true;
            attachments.needFilmDustTransmittance = true;
            attachments.gateWidth = 1280;
            attachments.gateHeight = 720;
            const auto request = make_scratch_request_descriptor(kExtent, *active, attachments);
            ASSERT_TRUE(request);
            EXPECT_TRUE(scratch_request_descriptor_is_valid(*request));
            EXPECT_EQ(request->spatialDirDescriptorHash, kDescriptorHash);
            EXPECT_EQ(request->spatialDirScratchTier, composition.source.tier);
            EXPECT_EQ(request->spatialDirTargetScratchTier, composition.target.tier);
            EXPECT_EQ(request->spatialDirPlaneRoles.rawCorrectionPlanes, composition.source.roles.rawCorrectionPlanes);
            EXPECT_EQ(request->spatialDirPlaneRoles.filteredCorrectionPlanes, composition.source.roles.filteredCorrectionPlanes);
            EXPECT_EQ(request->spatialDirPlaneRoles.filterTempPlanes, composition.source.roles.filterTempPlanes);
            EXPECT_EQ(request->spatialDirPlaneRoles.cachedLogRawPlanes, composition.source.roles.cachedLogRawPlanes);
            EXPECT_EQ(request->spatialDirTargetPlaneRoles.rawCorrectionPlanes, composition.target.roles.rawCorrectionPlanes);
            EXPECT_EQ(request->spatialDirTargetPlaneRoles.filteredCorrectionPlanes, composition.target.roles.filteredCorrectionPlanes);
            EXPECT_EQ(request->spatialDirTargetPlaneRoles.filterTempPlanes, composition.target.roles.filterTempPlanes);
            EXPECT_EQ(request->spatialDirTargetPlaneRoles.cachedLogRawPlanes, composition.target.roles.cachedLogRawPlanes);
            EXPECT_EQ(request->generation, composition.generation);
        }
    }
} // namespace
