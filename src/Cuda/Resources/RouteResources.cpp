#include "Cuda/Resources/ResourceInternals.h"

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Cuda/JuicerCudaDriver.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "SpectralData.h"
#include "GamutCompression.h"
#include "Scanner.h"
#include "JuicerDirBoundary.h"
#include "juicer_host_preparation.h"

#include "Logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <numeric>
#include <atomic>
#include <span>
#include <sstream>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace JuicerCuda {

    void free_curve(Resources& resources, DeviceCurve& c) noexcept {
        if (c.x) {
            free_owned_device_noexcept(resources, c.x);
            c.x = nullptr;
        }
        if (c.y) {
            free_owned_device_noexcept(resources, c.y);
            c.y = nullptr;
        }
        c.n = 0;
        c.domainBegin = 0;
        c.domainEnd = 0;
    }

    bool retire_curve_locked(Resources& resources, DeviceCurve& c, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const size_t bytes = static_cast<size_t>(std::max(0, c.n)) * sizeof(float);
        if (c.x) {
            if (!retire_ptr_locked(resources, c.x, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.x = nullptr;
        }
        if (c.y) {
            if (!retire_ptr_locked(resources, c.y, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.y = nullptr;
        }
        c.n = 0;
        c.domainBegin = 0;
        c.domainEnd = 0;
        return true;
    }

    void free_density_layers(Resources& resources) noexcept {
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    free_owned_device_noexcept(
                        resources,
                        resources.densityCurvesLayers[layer][ch]);
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
        }
        for (int ch = 0; ch < 3; ++ch) {
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
        resources.filmDensityLayersHash = 0;
    }

    static bool retire_density_layers_locked(Resources& resources, void* cudaStreamOpaque, const char* label, Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        for (int ch = 0; ch < 3; ++ch) {
            const size_t bytes =
                static_cast<size_t>(std::max(0, resources.densityCurvesLayersChannelN[ch])) *
                sizeof(float);
            for (int layer = 0; layer < 3; ++layer) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    if (!retire_ptr_locked(resources, resources.densityCurvesLayers[layer][ch], bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                        return false;
                    }
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
        resources.filmDensityLayersHash = 0;
        return true;
    }

    void free_film_tc_lut(Resources& resources) noexcept {
        if (resources.filmTcLut) {
            free_owned_device_noexcept(resources, resources.filmTcLut);
            resources.filmTcLut = nullptr;
        }
        resources.filmTcLutExtent = 0;
        resources.filmTcLutKeyHash = 0;
    }

    void free_output_gamut_table(Resources& resources) noexcept {
        if (resources.outputGamutCmax) {
            free_owned_device_noexcept(resources, resources.outputGamutCmax);
            resources.outputGamutCmax = nullptr;
        }
        resources.outputGamutTableHash = 0;
        resources.outputGamutRecipeHash = 0;
    }

    void free_mallett_basis(Resources& resources) noexcept {
        if (resources.mallettBasis) {
            free_owned_device_noexcept(resources, resources.mallettBasis);
            resources.mallettBasis = nullptr;
        }
        resources.mallettBasisK = 0;
    }


    static void free_film_density_tables(
        Resources& resources,
        Resources::DeviceFilmDensityTables& t) noexcept;

    void free_print_payloads(Resources& resources) noexcept {
        free_curve(resources, resources.printDcC);
        free_curve(resources, resources.printDcM);
        free_curve(resources, resources.printDcY);
        free_curve(resources, resources.printSensC);
        free_curve(resources, resources.printSensM);
        free_curve(resources, resources.printSensY);
        free_film_density_tables(resources, resources.printFilmDensityTables);

        if (resources.printIllumFiltered) {
            free_owned_device_noexcept(resources, resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
        }
        if (resources.printPreflashIllumFiltered) {
            free_owned_device_noexcept(
                resources,
                resources.printPreflashIllumFiltered);
            resources.printPreflashIllumFiltered = nullptr;
        }
        resources.printIllumK = 0;
        resources.printPreflashIllumK = 0;
        resources.printFilmDensityTablesDescriptorHash = 0;
        resources.printProfileTablesDescriptorHash = 0;
        resources.printMainIlluminantDescriptorHash = 0;
        resources.printPreflashIlluminantDescriptorHash = 0;
    }

    static bool prepare_output_gamut_table_locked(
        Resources& resources,
        const OutputGamutResourceInput& input,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>& resourcesLock,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        constexpr std::size_t kTableBytes =
            Gamut::kOutputBoundaryValueCount * sizeof(float);
        if (!input.enabled) {
            if (resources.outputGamutCmax) {
                if (!retire_ptr_locked(
                        resources,
                        resources.outputGamutCmax,
                        kTableBytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        "inactive output gamut Cmax",
                        outError)) {
                    return false;
                }
            }
            resources.outputGamutCmax = nullptr;
            resources.outputGamutTableHash = 0;
            resources.outputGamutRecipeHash = 0;
            return true;
        }
        if (input.cmax.empty()) {
            outError.diagnostic =
                "MissingRequiredResource component=output_gamut_cmax field=boundary_table";
            return false;
        }
        if (resources.outputGamutCmax &&
            resources.outputGamutTableHash == input.tableHash &&
            resources.outputGamutRecipeHash == input.recipeHash) {
            return true;
        }

        void* candidateRaw = nullptr;
        if (!alloc_and_upload_bytes_locked(
                resources,
                candidateRaw,
                input.cmax.data(),
                kTableBytes,
                cudaStreamOpaque,
                &resourcesLock,
                "output gamut Cmax",
                outError)) {
            return false;
        }
        float* candidate = static_cast<float*>(candidateRaw);
        if (resources.outputGamutCmax) {
            if (!retire_ptr_locked(
                    resources,
                    resources.outputGamutCmax,
                    kTableBytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "replaced output gamut Cmax",
                    outError)) {
                free_owned_device_noexcept(resources, candidate);
                return false;
            }
        }
        resources.outputGamutCmax = candidate;
        resources.outputGamutTableHash = input.tableHash;
        resources.outputGamutRecipeHash = input.recipeHash;
        return true;
    }

    void free_stbn(Resources& resources) noexcept {
        if (resources.stbnData) {
            free_owned_device_noexcept(resources, resources.stbnData);
            resources.stbnData = nullptr;
        }
        resources.stbnWidth = 0;
        resources.stbnHeight = 0;
        resources.stbnFrames = 0;
    }

    void free_wang(Resources& resources) noexcept {
        if (resources.wangTilesData) {
            free_owned_device_noexcept(resources, resources.wangTilesData);
            resources.wangTilesData = nullptr;
        }
        if (resources.wangLutData) {
            free_owned_device_noexcept(resources, resources.wangLutData);
            resources.wangLutData = nullptr;
        }
        resources.wangWidth = 0;
        resources.wangHeight = 0;
        resources.wangCount = 0;
        resources.wangColors = 0;
    }

    bool ensure_grain_v2_uploaded(Resources& resources, const FjFilmGrain& input, const GrainV2::FilmQuad*& out, Failure& error) {
        out = nullptr;
        error = {};
        if (!input.bank_hash || !input.sums.data || input.sums.count != GrainV2::kBankEntries) {
            error.diagnostic = "MissingRequiredResource phase=grain_v2 field=bank";
            return false;
        }
        std::shared_ptr<Resources::GrainV2Slot> slot;
        {
            std::unique_lock<std::mutex> lock(resources.m);
            auto found = resources.grainV2Banks.find(input.bank_hash);
            if (found != resources.grainV2Banks.end()) {
                slot = found->second;
                slot->ready.wait(lock, [&] {
                    return !slot->loading;
                });
                if (slot->bank) {
                    out = slot->bank;
                    return true;
                }
                set_failure(error, slot->failure, "Grain v2 bank acquisition failed");
                return false;
            }
            slot = std::make_shared<Resources::GrainV2Slot>();
            resources.grainV2Banks.emplace(input.bank_hash, slot);
        }
        const auto publish = [&](GrainV2::FilmQuad* bank) {
            std::lock_guard<std::mutex> lock(resources.m);
            slot->bank = bank;
            slot->failure = error.status;
            slot->loading = false;
            if (!bank) {
                auto found = resources.grainV2Banks.find(input.bank_hash);
                if (found != resources.grainV2Banks.end() && found->second == slot)
                    resources.grainV2Banks.erase(found);
            }
            slot->ready.notify_all();
        };
        void* device = nullptr;
        constexpr std::size_t bytes = GrainV2::kQuadEntries * sizeof(GrainV2::FilmQuad);
        try {
            if (!allocate_owned_device(resources, &device, bytes, "Grain v2 bank", error)) {
                publish(nullptr);
                return false;
            }
            std::vector<GrainV2::FilmQuad> packed(GrainV2::kQuadEntries);
            for (int record = 0; record < GrainV2::kRecords; ++record) {
                for (int level = 0; level < GrainV2::kLevels; ++level) {
                    const auto index = static_cast<std::size_t>(record) * GrainV2::kLevels + level;
                    const float* source = input.sums.data + index * GrainV2::kTileEntries;
                    auto* target = packed.data() + index * GrainV2::kTileSide * GrainV2::kTileSide;
                    for (int y = 0; y < GrainV2::kTileSide; ++y) {
                        for (int x = 0; x < GrainV2::kTileSide; ++x) {
                            const auto at = static_cast<std::size_t>(y) * GrainV2::kTileStride + x;
                            target[y * GrainV2::kTileSide + x] = {source[at], source[at + 1], source[at + GrainV2::kTileStride], source[at + GrainV2::kTileStride + 1]};
                        }
                    }
                }
            }
            cudaError_t status = cudaMemcpy(device, packed.data(), bytes, cudaMemcpyHostToDevice);
            if (status == cudaSuccess) {
                // Publish only after the copy is visible to other streams.
                status = cudaStreamSynchronize(nullptr);
            }
            if (status != cudaSuccess) {
                set_failure(error, runtime_failure_status(status), "CUDA Grain v2 bank upload failed");
                free_owned_device_noexcept(resources, device);
                publish(nullptr);
                return false;
            }
            auto* quads = static_cast<GrainV2::FilmQuad*>(device);
            publish(quads);
            out = quads;
            return true;
        } catch (const std::bad_alloc&) {
            free_owned_device_noexcept(resources, device);
            set_failure(error, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "Grain v2 bank packing allocation failed");
            publish(nullptr);
            return false;
        } catch (...) {
            free_owned_device_noexcept(resources, device);
            set_failure(error, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Grain v2 bank packing failed");
            publish(nullptr);
            return false;
        }
    }

    bool ensure_grain_static_assets_uploaded(
        Resources& resources,
        const StaticNoiseInput& input,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError = {};
        if (input.stbn.empty() || input.stbnWidth <= 0 || input.stbnHeight <= 0 ||
            input.stbnFrames <= 0 || input.wangTiles.empty() || input.wangLut.empty() ||
            input.wangWidth <= 0 || input.wangHeight <= 0 || input.wangCount <= 0 ||
            input.wangColors <= 0) {
            outError.diagnostic = "grain static host asset payload is incomplete";
            return false;
        }

        std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
        std::unique_lock<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);

        const bool needStbnUpload = !resources.stbnData;
        const bool needWangUpload =
            !resources.wangTilesData || !resources.wangLutData;
        if (!needStbnUpload && !needWangUpload) {
            if (resources.stbnWidth <= 0 || resources.stbnHeight <= 0 ||
                resources.stbnFrames <= 0 || resources.wangWidth <= 0 ||
                resources.wangHeight <= 0 || resources.wangCount <= 0 ||
                resources.wangColors <= 0) {
                outError.diagnostic = "grain static resource identity mismatch";
                return false;
            }
            return true;
        }

        bool stbnFailed = false;
        bool wangFailed = false;
        Failure stbnError;
        if (!resources.stbnData) {
            void* stbnData = nullptr;
            if (!alloc_and_upload_bytes_locked(
                    resources,
                    stbnData,
                    input.stbn.data(),
                    input.stbn.size(),
                    cudaStreamOpaque,
                    &lock,
                    "STBN",
                    stbnError)) {
                stbnFailed = true;
                free_stbn(resources);
            } else if (!resources.stbnData) {
                resources.stbnData =
                    reinterpret_cast<std::uint8_t*>(stbnData);
                resources.stbnWidth = input.stbnWidth;
                resources.stbnHeight = input.stbnHeight;
                resources.stbnFrames = input.stbnFrames;
            } else if (stbnData) {
                free_owned_device_noexcept(resources, stbnData);
            }
        }

        Failure wangError;
        if (!resources.wangTilesData || !resources.wangLutData) {
            if (resources.wangTilesData || resources.wangLutData) {
                free_wang(resources);
            }
            void* wangTilesData = nullptr;
            void* wangLutData = nullptr;
            const bool tilesOk = alloc_and_upload_bytes_locked(
                resources,
                wangTilesData,
                input.wangTiles.data(),
                input.wangTiles.size(),
                cudaStreamOpaque,
                &lock,
                "Wang.tiles",
                wangError);
            const bool lutOk = tilesOk && alloc_and_upload_bytes_locked(
                                              resources,
                                              wangLutData,
                                              input.wangLut.data(),
                                              input.wangLut.size(),
                                              cudaStreamOpaque,
                                              &lock,
                                              "Wang.lut",
                                              wangError);
            if (!tilesOk || !lutOk) {
                wangFailed = true;
                if (wangTilesData) {
                    free_owned_device_noexcept(resources, wangTilesData);
                }
                if (wangLutData) {
                    free_owned_device_noexcept(resources, wangLutData);
                }
                free_wang(resources);
            } else if (!resources.wangTilesData && !resources.wangLutData) {
                resources.wangTilesData =
                    reinterpret_cast<std::uint8_t*>(wangTilesData);
                resources.wangLutData =
                    reinterpret_cast<std::uint8_t*>(wangLutData);
                resources.wangWidth = input.wangWidth;
                resources.wangHeight = input.wangHeight;
                resources.wangCount = input.wangCount;
                resources.wangColors = input.wangColors;
            } else {
                if (wangTilesData) {
                    free_owned_device_noexcept(resources, wangTilesData);
                }
                if (wangLutData) {
                    free_owned_device_noexcept(resources, wangLutData);
                }
            }
        }

        const bool ready =
            resources.stbnData && resources.stbnWidth > 0 &&
            resources.stbnHeight > 0 && resources.stbnFrames > 0 &&
            resources.wangTilesData && resources.wangLutData &&
            resources.wangWidth > 0 && resources.wangHeight > 0 &&
            resources.wangCount > 0 && resources.wangColors > 0;
        if (!ready) {
            outError.status = stbnFailed ? stbnError.status : wangFailed ? wangError.status
                                                                         : FjStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            // Either upload can invalidate the shared context, even after an ordinary failure.
            if (wangFailed && context_loss(wangError) && !context_loss(outError)) {
                outError.status = wangError.status;
            }
            outError.diagnostic = !stbnError.diagnostic.empty()
                                      ? stbnError.diagnostic
                                      : (!wangError.diagnostic.empty()
                                             ? wangError.diagnostic
                                             : "grain static resource upload incomplete");
            if (!stbnError.diagnostic.empty() && !wangError.diagnostic.empty()) {
                outError.diagnostic += " | Wang upload failed: ";
                outError.diagnostic += wangError.diagnostic;
            }
            return false;
        }
        return true;
    }

    bool prepare_focused_route_resources(
        Resources& resources,
        const FocusedRouteResourceInput& input,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError = {};
        const FilmExposureResourceInput& filmRaw = input.filmRaw;
        const bool tcMethod =
            filmRaw.rgbToRawMethod ==
                Spektrafilm::RgbToRawMethod::Hanatos2025 ||
            filmRaw.rgbToRawMethod ==
                Spektrafilm::RgbToRawMethod::Arctic2026beta04;
        constexpr std::size_t kFilmTcValueCount =
            static_cast<std::size_t>(Spectral::kFilmTcLutExtent) *
            static_cast<std::size_t>(Spectral::kFilmTcLutExtent) *
            Spectral::kFilmTcLutChannels;
        if ((tcMethod &&
             (input.filmTcLut.size() != kFilmTcValueCount ||
              filmRaw.tcLutHash == 0)) ||
            (!tcMethod && !input.filmTcLut.empty())) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=focused_route field=film_tc_lut";
            return false;
        }
        const FilmDevelopmentResourceInput& filmDevelop = input.filmDevelop;
        const DirResourceInput& dirCouplers = input.dirCouplers;
        const OutputGamutResourceInput& outputGamut = input.outputGamut;
        if ((outputGamut.enabled && outputGamut.cmax.size() != Gamut::kOutputBoundaryValueCount) ||
            (!outputGamut.enabled && !outputGamut.cmax.empty())) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=focused_route field=output_gamut_binding";
            return false;
        }
        const bool wantDensityLayers =
            input.wantDensityLayers;
        if (input.exposureSampleCount != Spectral::kNumSamples ||
            filmDevelop.logExposure.empty() ||
            filmDevelop.logExposure.size() * 3u != filmDevelop.normalizedDensityCurvesRgb.scalar_count()) {
            outError.diagnostic = "focused route resource host derivation shape mismatch";
            return false;
        }

        std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
        std::unique_lock<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError)) {
            return false;
        }
        reap_retire_queue_locked(resources);

        const bool coreAlreadyPrepared =
            resources.filmFinalSensitivityHash == filmRaw.finalSensitivityHash &&
            resources.filmDensityCurvesHash == filmDevelop.normalizedDensityCurvesHash &&
            resources.filmDensityLayersHash == (wantDensityLayers ? filmDevelop.densityCurvesLayersHash : 0) &&
            resources.filmDirHash == (dirCouplers.active ? dirCouplers.hash : 0) &&
            resources.filmRgbToRawMethod == filmRaw.rgbToRawMethod &&
            resources.sensB.x && resources.sensG.x && resources.sensR.x &&
            resources.densB.x && resources.densG.x && resources.densR.x &&
            (!wantDensityLayers ||
             (resources.hasDensityCurvesLayers &&
              resources.densityCurvesLayers[0][0] && resources.densityCurvesLayers[0][1] &&
              resources.densityCurvesLayers[0][2] && resources.densityCurvesLayers[1][0] &&
              resources.densityCurvesLayers[1][1] && resources.densityCurvesLayers[1][2] &&
              resources.densityCurvesLayers[2][0] && resources.densityCurvesLayers[2][1] &&
              resources.densityCurvesLayers[2][2])) &&
            (wantDensityLayers || !resources.hasDensityCurvesLayers) &&
            (!dirCouplers.active ||
             (resources.dirDensB.x && resources.dirDensG.x && resources.dirDensR.x)) &&
            (tcMethod || resources.tablesIllum) &&
            ((tcMethod &&
              resources.filmTcLut &&
              resources.filmTcLutExtent == Spectral::kFilmTcLutExtent &&
              resources.filmTcLutKeyHash == filmRaw.tcLutHash &&
              !resources.mallettBasis) ||
             (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 &&
              resources.mallettBasis && !resources.filmTcLut));
        if (coreAlreadyPrepared) {
            return prepare_output_gamut_table_locked(
                resources,
                outputGamut,
                cudaStreamOpaque,
                lock,
                outError);
        }
        const std::span<const float> sensitivityWavelengthsNm = Spectral::kReferenceShape.wavelengths;
        std::vector<float> sensB(Spectral::kNumSamples);
        std::vector<float> sensG(Spectral::kNumSamples);
        std::vector<float> sensR(Spectral::kNumSamples);
        for (int sample = 0; sample < Spectral::kNumSamples; ++sample) {
            const auto rgb = filmRaw.finalSensitivityRgb.sample(static_cast<std::size_t>(sample));
            sensB[static_cast<std::size_t>(sample)] = rgb[2];
            sensG[static_cast<std::size_t>(sample)] = rgb[1];
            sensR[static_cast<std::size_t>(sample)] = rgb[0];
        }

        const auto filmLogExposure = filmDevelop.logExposure;
        const std::size_t densitySampleCount = filmDevelop.normalizedDensityCurvesRgb.sample_count();
        std::vector<float> densB(densitySampleCount);
        std::vector<float> densG(densitySampleCount);
        std::vector<float> densR(densitySampleCount);
        for (std::size_t sample = 0; sample < densitySampleCount; ++sample) {
            const auto rgb = filmDevelop.normalizedDensityCurvesRgb.sample(sample);
            densB[sample] = rgb[2];
            densG[sample] = rgb[1];
            densR[sample] = rgb[0];
        }

        resources.filmFinalSensitivityHash = 0;
        resources.filmDensityCurvesHash = 0;
        if (!upload_curve_locked(resources, resources.sensB, {sensitivityWavelengthsNm, sensB}, cudaStreamOpaque, lock, "focused film finalSensB", outError) ||
            !upload_curve_locked(resources, resources.sensG, {sensitivityWavelengthsNm, sensG}, cudaStreamOpaque, lock, "focused film finalSensG", outError) ||
            !upload_curve_locked(resources, resources.sensR, {sensitivityWavelengthsNm, sensR}, cudaStreamOpaque, lock, "focused film finalSensR", outError) ||
            !upload_curve_locked(resources, resources.densB, {filmLogExposure, densB}, cudaStreamOpaque, lock, "focused film normalizedDensB", outError) ||
            !upload_curve_locked(resources, resources.densG, {filmLogExposure, densG}, cudaStreamOpaque, lock, "focused film normalizedDensG", outError) ||
            !upload_curve_locked(resources, resources.densR, {filmLogExposure, densR}, cudaStreamOpaque, lock, "focused film normalizedDensR", outError)) {
            return false;
        }

        if (wantDensityLayers) {
            const int densitySamples = static_cast<int>(filmDevelop.logExposure.size());
            if (!filmDevelop.densityCurvesLayersRequired ||
                filmDevelop.densityCurvesLayersHash == 0 ||
                densitySamples <= 0) {
                outError.diagnostic = "MissingRequiredResource phase=9B field=density_curves_layers";
                return false;
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    if (static_cast<int>(filmDevelop.densityCurvesLayers[layer][ch].size()) !=
                        densitySamples) {
                        outError.diagnostic = "MalformedRequiredProfileData phase=9B field=density_curves_layers shape";
                        return false;
                    }
                }
            }
            bool canReuse =
                resources.hasDensityCurvesLayers &&
                resources.filmDensityLayersHash == filmDevelop.densityCurvesLayersHash &&
                resources.densityCurvesLayersChannelN[0] == densitySamples &&
                resources.densityCurvesLayersChannelN[1] == densitySamples &&
                resources.densityCurvesLayersChannelN[2] == densitySamples;
            for (int layer = 0; layer < 3 && canReuse; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    canReuse = canReuse && resources.densityCurvesLayers[layer][ch] != nullptr;
                }
            }
            resources.filmDensityLayersHash = 0;
            if (!canReuse) {
                if (!retire_density_layers_locked(
                        resources,
                        cudaStreamOpaque,
                        "focused density_curves_layers",
                        outError)) {
                    return false;
                }
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    if (!upload_array_locked(
                            resources,
                            resources.densityCurvesLayers[layer][ch],
                            canReuse ? resources.densityCurvesLayersChannelN[ch] : 0,
                            filmDevelop.densityCurvesLayers[layer][ch].data(),
                            densitySamples,
                            cudaStreamOpaque,
                            lock,
                            "focused density_curves_layers",
                            outError)) {
                        return false;
                    }
                }
            }
            for (int ch = 0; ch < 3; ++ch) {
                resources.densityCurvesLayersChannelN[ch] = densitySamples;
            }
            resources.hasDensityCurvesLayers = 1;
        } else if (resources.hasDensityCurvesLayers || resources.filmDensityLayersHash != 0) {
            resources.filmDensityLayersHash = 0;
            if (!retire_density_layers_locked(
                    resources,
                    cudaStreamOpaque,
                    "inactive focused density_curves_layers",
                    outError)) {
                return false;
            }
        }

        resources.filmDirHash = 0;
        if (dirCouplers.active) {
            if (dirCouplers.hash == 0 ||
                dirCouplers.compensatedDensityCurveAxesHash == 0 ||
                std::any_of(
                    dirCouplers.compensatedDensityCurveAxesRgb.begin(),
                    dirCouplers.compensatedDensityCurveAxesRgb.end(),
                    [&](const auto& axis) {
                        return axis.size() != filmDevelop.logExposure.size();
                    })) {
                outError.diagnostic = "focused film DIR resource descriptor mismatch";
                return false;
            }
            const auto& compensatedLogExposureRgb = dirCouplers.compensatedDensityCurveAxesRgb;
            if (!upload_curve_locked(resources, resources.dirDensB, {compensatedLogExposureRgb[2], densB}, cudaStreamOpaque, lock, "focused film DIR densB", outError) ||
                !upload_curve_locked(resources, resources.dirDensG, {compensatedLogExposureRgb[1], densG}, cudaStreamOpaque, lock, "focused film DIR densG", outError) ||
                !upload_curve_locked(resources, resources.dirDensR, {compensatedLogExposureRgb[0], densR}, cudaStreamOpaque, lock, "focused film DIR densR", outError)) {
                return false;
            }
        } else {
            if (!retire_curve_locked(resources, resources.dirDensB, cudaStreamOpaque, "disabled focused film DIR densB", outError) ||
                !retire_curve_locked(resources, resources.dirDensG, cudaStreamOpaque, "disabled focused film DIR densG", outError) ||
                !retire_curve_locked(resources, resources.dirDensR, cudaStreamOpaque, "disabled focused film DIR densR", outError)) {
                return false;
            }
        }

        const int exposureK = input.exposureSampleCount;
        if (tcMethod) {
            if (resources.tablesIllum) {
                const std::size_t bytes =
                    static_cast<std::size_t>(resources.tablesK) * sizeof(float);
                if (!retire_ptr_locked(resources, resources.tablesIllum, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected Mallett illuminant", outError)) {
                    return false;
                }
                resources.tablesIllum = nullptr;
                resources.tablesK = 0;
            }
            constexpr int extent = Spectral::kFilmTcLutExtent;
            constexpr int integratedCount =
                extent * extent * Spectral::kFilmTcLutChannels;
            if (!upload_array_locked(resources, resources.filmTcLut, resources.filmTcLutExtent * resources.filmTcLutExtent * Spectral::kFilmTcLutChannels, input.filmTcLut.data(), integratedCount, cudaStreamOpaque, lock, "film TC LUT", outError)) {
                return false;
            }
            resources.filmTcLutExtent = extent;
            resources.filmTcLutKeyHash = filmRaw.tcLutHash;
            if (resources.mallettBasis) {
                const std::size_t bytes = static_cast<std::size_t>(resources.mallettBasisK) * 3u * sizeof(float);
                if (!retire_ptr_locked(resources, resources.mallettBasis, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected Mallett basis", outError)) {
                    return false;
                }
                resources.mallettBasis = nullptr;
                resources.mallettBasisK = 0;
            }
        } else if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
            if (!upload_array_locked(resources, resources.tablesIllum, resources.tablesK, input.exposureIlluminant.data(), exposureK, cudaStreamOpaque, lock, "focused film tablesIllum", outError)) {
                return false;
            }
            resources.tablesK = exposureK;
            const int k = input.mallettRows;
            if (!input.mallettAvailable ||
                k != Spectral::kNumSamples || input.mallettColumns != 3 ||
                input.mallettBasis.empty()) {
                outError.diagnostic = "selected Mallett resource family is unavailable";
                return false;
            }
            const int count = k * 3;
            if (!upload_array_locked(resources, resources.mallettBasis, resources.mallettBasisK * 3, input.mallettBasis.data(), count, cudaStreamOpaque, lock, "film Mallett basis", outError)) {
                return false;
            }
            resources.mallettBasisK = k;
            if (resources.filmTcLut) {
                const std::size_t bytes = static_cast<std::size_t>(resources.filmTcLutExtent) *
                                          static_cast<std::size_t>(resources.filmTcLutExtent) * Spectral::kFilmTcLutChannels * sizeof(float);
                if (!retire_ptr_locked(resources, resources.filmTcLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected film TC LUT", outError)) {
                    return false;
                }
                resources.filmTcLut = nullptr;
                resources.filmTcLutExtent = 0;
                resources.filmTcLutKeyHash = 0;
            }
        } else {
            outError.diagnostic = "selected RGB-to-raw method is unsupported";
            return false;
        }

        resources.filmFinalSensitivityHash = filmRaw.finalSensitivityHash;
        resources.filmDensityCurvesHash = filmDevelop.normalizedDensityCurvesHash;
        resources.filmDensityLayersHash = wantDensityLayers ? filmDevelop.densityCurvesLayersHash : 0;
        resources.filmDirHash = dirCouplers.active ? dirCouplers.hash : 0;
        resources.filmRgbToRawMethod = filmRaw.rgbToRawMethod;
        JTRACE_VERBOSE("CUDA_UPLOAD", "filmCurveHash=" + std::to_string(resources.filmDensityCurvesHash));
        return prepare_output_gamut_table_locked(
            resources,
            outputGamut,
            cudaStreamOpaque,
            lock,
            outError);
    }

    namespace {

        bool print_curve_ready(const DeviceCurveView& curve, int expectedSamples, bool requireX) {
            return curve.y && (!requireX || curve.x) && curve.n == expectedSamples &&
                   curve.domainBegin >= 0 && curve.domainEnd >= curve.domainBegin &&
                   curve.domainEnd < curve.n;
        }

        struct PrintPreparationReceipt {
            const char* cacheOutcome = nullptr;
            const char* mainIlluminantOutcome = nullptr;
            std::uint64_t mainHash = 0;
            CmyCcTriplet cmy;
            std::array<std::uint64_t, 4> resident{};
            std::array<std::uintptr_t, 2> slots{};
            std::uint64_t chargedBytes = 0;
            std::array<float, 3> raw{};
        };

#if JUICER_DIAGNOSTICS_COMPILED
        void trace_print_preparation(const PrintPreparationReceipt& receipt) noexcept {
            try {
                std::string msg = "event=print_resource_preparation cacheOutcome=";
                msg += receipt.cacheOutcome;
                msg += " mainIlluminantOutcome=";
                msg += receipt.mainIlluminantOutcome;
                msg += " filteredMainIlluminantDescriptorHash=" + std::to_string(receipt.mainHash);
                msg += " preparedMainIlluminantDescriptorHash=" + std::to_string(receipt.resident[2]);
                msg += " descriptorCmyCc(C/M/Y)=" + std::to_string(receipt.cmy.c) + "/" +
                       std::to_string(receipt.cmy.m) + "/" + std::to_string(receipt.cmy.y);
                msg += " resident=" + std::to_string(receipt.resident[0]) + "/" + std::to_string(receipt.resident[1]) +
                       "/" + std::to_string(receipt.resident[2]) + "/" + std::to_string(receipt.resident[3]);
                msg += " slots=" + std::to_string(receipt.slots[0]) + "/" + std::to_string(receipt.slots[1]);
                msg += " chargedBytes=" + std::to_string(receipt.chargedBytes);
                msg += " raw=" + std::to_string(std::bit_cast<std::uint32_t>(receipt.raw[0])) +
                       "/" + std::to_string(std::bit_cast<std::uint32_t>(receipt.raw[1])) +
                       "/" + std::to_string(std::bit_cast<std::uint32_t>(receipt.raw[2]));
                JuicerLogging::write("PHASE4C_BALANCE", msg, true);
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
#endif

        bool prepare_print_resources_impl(
            Resources& resources,
            const PrintResourceInput& input,
            void* cudaStreamOpaque,
            Failure& outError,
            PrintPreparationReceipt* receipt) {
            const PrintResourceDescriptors& descriptors = input.descriptors;
            outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
            std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
            std::unique_lock<std::mutex> lock(resources.m);
            if (!validate_resource_owner_locked(resources, outError)) {
                return false;
            }
            reap_retire_queue_locked(resources);
            const bool filmDensityHit =
                resources.printFilmDensityTablesDescriptorHash != 0 &&
                resources.printFilmDensityTablesDescriptorHash == descriptors.filmDensityTablesHash &&
                resources.printFilmDensityTables.K == Spectral::kNumSamples &&
                resources.printFilmDensityTables.epsC &&
                resources.printFilmDensityTables.epsM &&
                resources.printFilmDensityTables.epsY &&
                resources.printFilmDensityTables.baseDensityMin;
            const bool profileHit =
                resources.printProfileTablesDescriptorHash != 0 &&
                resources.printProfileTablesDescriptorHash == descriptors.profileTablesHash &&
                resources.printDcC.x && resources.printDcC.y &&
                resources.printDcM.y && resources.printDcY.y &&
                resources.printSensC.y && resources.printSensM.y && resources.printSensY.y;
            const bool mainHit =
                resources.printMainIlluminantDescriptorHash != 0 &&
                resources.printMainIlluminantDescriptorHash == descriptors.mainIlluminantHash &&
                resources.printIllumFiltered &&
                resources.printIllumK == Spectral::kNumSamples;
            const bool preflashIlluminantHit = descriptors.preflashActive
                                                   ? (resources.printPreflashIlluminantDescriptorHash != 0 &&
                                                      resources.printPreflashIlluminantDescriptorHash == descriptors.preflashIlluminantHash &&
                                                      resources.printPreflashIllumFiltered &&
                                                      resources.printPreflashIllumK == Spectral::kNumSamples)
                                                   : (!resources.printPreflashIllumFiltered &&
                                                      resources.printPreflashIlluminantDescriptorHash == 0 &&
                                                      resources.printPreflashIllumK == 0);
            const auto capture_receipt = [&](const char* cacheOutcome,
                                             const char* mainIlluminantOutcome) {
                if (!receipt) {
                    return;
                }
                *receipt = {
                    cacheOutcome, mainIlluminantOutcome, descriptors.mainIlluminantHash, descriptors.mainCmyCc, {resources.printFilmDensityTablesDescriptorHash, resources.printProfileTablesDescriptorHash, resources.printMainIlluminantDescriptorHash, resources.printPreflashIlluminantDescriptorHash}, {reinterpret_cast<std::uintptr_t>(resources.printIllumFiltered), reinterpret_cast<std::uintptr_t>(resources.printPreflashIllumFiltered)}, resources.deviceLedger->snapshot().chargedBytes, input.preflashRawCmy};
            };
            if (filmDensityHit && profileHit && mainHit && preflashIlluminantHit) {
                capture_receipt("hit", "hit");
                return true;
            }

            if (!filmDensityHit) {
                std::array<float, Spectral::kNumSamples> epsC{};
                std::array<float, Spectral::kNumSamples> epsM{};
                std::array<float, Spectral::kNumSamples> epsY{};
                for (std::size_t sample = 0; sample < input.filmChannelDensityCmy.sample_count(); ++sample) {
                    const auto cmy = input.filmChannelDensityCmy.sample(sample);
                    epsC[sample] = cmy[0];
                    epsM[sample] = cmy[1];
                    epsY[sample] = cmy[2];
                }
                Resources::DeviceFilmDensityTables& tables = resources.printFilmDensityTables;
                const int currentK = tables.K;
                resources.printFilmDensityTablesDescriptorHash = 0;
                if (!upload_array_locked(resources, tables.epsC, currentK, epsC.data(), Spectral::kNumSamples, cudaStreamOpaque, lock, "phase4B print film density epsC", outError) ||
                    !upload_array_locked(resources, tables.epsM, currentK, epsM.data(), Spectral::kNumSamples, cudaStreamOpaque, lock, "phase4B print film density epsM", outError) ||
                    !upload_array_locked(resources, tables.epsY, currentK, epsY.data(), Spectral::kNumSamples, cudaStreamOpaque, lock, "phase4B print film density epsY", outError) ||
                    !upload_array_locked(resources, tables.baseDensityMin, currentK, input.filmBaseDensity.data(), Spectral::kNumSamples, cudaStreamOpaque, lock, "phase4B print film density base", outError)) {
                    return false;
                }
                tables.K = Spectral::kNumSamples;
                resources.printFilmDensityTablesDescriptorHash = descriptors.filmDensityTablesHash;
            }
            if (!profileHit) {
                const auto printLogExposure = input.printLogExposure;
                const std::size_t densitySampleCount = input.printDensityCurvesCmy.sample_count();
                std::vector<float> dcC(densitySampleCount);
                std::vector<float> dcM(densitySampleCount);
                std::vector<float> dcY(densitySampleCount);
                std::vector<float> sensC(Spectral::kNumSamples);
                std::vector<float> sensM(Spectral::kNumSamples);
                std::vector<float> sensY(Spectral::kNumSamples);
                for (std::size_t sample = 0; sample < densitySampleCount; ++sample) {
                    const auto cmy = input.printDensityCurvesCmy.sample(sample);
                    dcC[sample] = cmy[0];
                    dcM[sample] = cmy[1];
                    dcY[sample] = cmy[2];
                }
                for (int sample = 0; sample < Spectral::kNumSamples; ++sample) {
                    const auto cmy = input.printSensitivityCmy.sample(static_cast<std::size_t>(sample));
                    sensC[static_cast<std::size_t>(sample)] = cmy[0];
                    sensM[static_cast<std::size_t>(sample)] = cmy[1];
                    sensY[static_cast<std::size_t>(sample)] = cmy[2];
                }
                resources.printProfileTablesDescriptorHash = 0;
                if (!upload_curve_locked(resources, resources.printDcC, {printLogExposure, dcC}, cudaStreamOpaque, lock, "phase4B print dcC", outError) ||
                    !upload_spectral_samples_locked(resources, resources.printDcM, dcM, cudaStreamOpaque, lock, "phase4B print dcM", outError) ||
                    !upload_spectral_samples_locked(resources, resources.printDcY, dcY, cudaStreamOpaque, lock, "phase4B print dcY", outError) ||
                    !upload_spectral_samples_locked(resources, resources.printSensC, sensC, cudaStreamOpaque, lock, "phase4B print sensC", outError) ||
                    !upload_spectral_samples_locked(resources, resources.printSensM, sensM, cudaStreamOpaque, lock, "phase4B print sensM", outError) ||
                    !upload_spectral_samples_locked(resources, resources.printSensY, sensY, cudaStreamOpaque, lock, "phase4B print sensY", outError)) {
                    return false;
                }
                resources.printProfileTablesDescriptorHash = descriptors.profileTablesHash;
            }
            if (!mainHit) {
                if (resources.printIllumFiltered && resources.printIllumK == Spectral::kNumSamples) {
                    resources.printMainIlluminantDescriptorHash = 0;
                }
                if (!upload_array_locked(
                        resources,
                        resources.printIllumFiltered,
                        resources.printIllumK,
                        input.mainIlluminant.data(),
                        Spectral::kNumSamples,
                        cudaStreamOpaque,
                        lock,
                        "phase4B filtered main print illuminant",
                        outError)) {
                    return false;
                }
                resources.printIllumK = Spectral::kNumSamples;
                resources.printMainIlluminantDescriptorHash = descriptors.mainIlluminantHash;
            }
            if (descriptors.preflashActive) {
                if (!preflashIlluminantHit) {
                    if (resources.printPreflashIllumFiltered && resources.printPreflashIllumK == Spectral::kNumSamples) {
                        resources.printPreflashIlluminantDescriptorHash = 0;
                    }
                    if (!upload_array_locked(
                            resources,
                            resources.printPreflashIllumFiltered,
                            resources.printPreflashIllumK,
                            input.preflashIlluminant.data(),
                            Spectral::kNumSamples,
                            cudaStreamOpaque,
                            lock,
                            "phase4B filtered preflash print illuminant",
                            outError)) {
                        return false;
                    }
                    resources.printPreflashIllumK = Spectral::kNumSamples;
                    resources.printPreflashIlluminantDescriptorHash =
                        descriptors.preflashIlluminantHash;
                }
            } else {
                if (resources.printPreflashIllumFiltered) {
                    const std::size_t bytes =
                        static_cast<std::size_t>(std::max(0, resources.printPreflashIllumK)) * sizeof(float);
                    if (!retire_ptr_locked(
                            resources,
                            resources.printPreflashIllumFiltered,
                            bytes,
                            Resources::RetireKind::DeviceFree,
                            cudaStreamOpaque,
                            "disabled phase4B preflash illuminant",
                            outError)) {
                        return false;
                    }
                    resources.printPreflashIllumFiltered = nullptr;
                }
                resources.printPreflashIllumK = 0;
                resources.printPreflashIlluminantDescriptorHash = 0;
            }
            capture_receipt("rebuilt", mainHit ? "retained" : "rebuilt");
            return true;
        }

    } // namespace


    bool prepare_print_resources(
        Resources& resources,
        const PrintResourceInput& input,
        void* cudaStreamOpaque,
        Failure& outError) {
        outError = {};
#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(3)) {
            PrintPreparationReceipt receipt;
            if (!prepare_print_resources_impl(resources, input, cudaStreamOpaque, outError, &receipt)) {
                return false;
            }
            // Both resource locks have left scope. The receipt owns every
            // value needed for formatting and the synchronous flush.
            trace_print_preparation(receipt);
            return true;
        }
#endif
        return prepare_print_resources_impl(resources, input, cudaStreamOpaque, outError, nullptr);
    }

    bool pack_print_cuda_payloads(
        float printExposure,
        float preflashExposure,
        const PrintPreparedView& prepared,
        float routeCorrectionScale,
        PrintCudaPayloadPack& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = PrintCudaPayloadPack{};
        if (!prepared.active ||
            prepared.filmDensityTablesHash == 0 ||
            prepared.filmDensityTables.K != Spectral::kNumSamples ||
            !prepared.filmDensityTables.epsC ||
            !prepared.filmDensityTables.epsM ||
            !prepared.filmDensityTables.epsY ||
            !prepared.filmDensityTables.baseDensityMin ||
            prepared.profileTablesHash == 0 ||
            prepared.mainIlluminantHash == 0 ||
            !prepared.mainIlluminant ||
            prepared.spectralSampleCount != Spectral::kNumSamples ||
            !print_curve_ready(prepared.printSensC, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printSensM, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printSensY, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printDcC, prepared.printDcC.n, true) ||
            !print_curve_ready(prepared.printDcM, prepared.printDcC.n, true) ||
            !print_curve_ready(prepared.printDcY, prepared.printDcC.n, true) ||
            prepared.printDcC.x != prepared.printDcM.x ||
            prepared.printDcC.x != prepared.printDcY.x) {
            diagnostic = "ResourceDescriptorMismatch phase=4B field=print_prepared_view";
            return false;
        }
        if (prepared.preflashActive &&
            (!prepared.preflashIlluminant ||
             prepared.preflashIlluminantHash == 0)) {
            diagnostic = "MissingRequiredResource phase=4B field=preflash_prepared_view";
            return false;
        }
        if (!std::isfinite(printExposure) ||
            printExposure < 0.0f ||
            !std::isfinite(prepared.normalizer) ||
            !(prepared.normalizer > 0.0f) ||
            !std::isfinite(routeCorrectionScale) ||
            !(routeCorrectionScale > 0.0f)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=4B field=print_exposure_scale";
            return false;
        }
        if (prepared.preflashActive) {
            if (!std::isfinite(preflashExposure) ||
                preflashExposure < 0.0f) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=4B field=preflash_exposure";
                return false;
            }
            for (float raw : prepared.preflashRawCmy) {
                if (!std::isfinite(raw) || raw < 0.0f) {
                    diagnostic =
                        "ResourceDescriptorMismatch phase=4B field=preflash_raw";
                    return false;
                }
            }
        }

        out.expose.active = 1;
        out.expose.negTables = prepared.filmDensityTables;
        out.expose.printIllumFiltered = prepared.mainIlluminant;
        out.expose.printIllumK = prepared.spectralSampleCount;
        out.expose.printSensC = prepared.printSensC;
        out.expose.printSensM = prepared.printSensM;
        out.expose.printSensY = prepared.printSensY;
        out.expose.printExposure = printExposure;
        out.expose.routeCorrectionScale = routeCorrectionScale;
        out.expose.printPreflashExposure =
            prepared.preflashActive ? preflashExposure : 0.0f;
        out.expose.printMidgrayFactor = prepared.normalizer;
        if (prepared.preflashActive) {
            std::copy_n(prepared.preflashRawCmy, 3, out.expose.printPreflashRaw);
        }
        out.develop.printDcC = prepared.printDcC;
        out.develop.printDcM = prepared.printDcM;
        out.develop.printDcY = prepared.printDcY;
        const float printRawScale = printExposure * prepared.normalizer;
        if (!(std::isfinite(printRawScale) && printRawScale >= 0.0f)) {
            diagnostic = "MalformedRequiredResource phase=4B field=print_raw_scale";
            return false;
        }
        return true;
    }

    void free_tables(Resources& resources) noexcept {
        if (resources.tablesIllum) {
            free_owned_device_noexcept(resources, resources.tablesIllum);
            resources.tablesIllum = nullptr;
        }
        resources.tablesK = 0;
    }

    static void free_film_density_tables(
        Resources& resources,
        Resources::DeviceFilmDensityTables& t) noexcept {
        if (t.epsC) {
            free_owned_device_noexcept(resources, t.epsC);
            t.epsC = nullptr;
        }
        if (t.epsM) {
            free_owned_device_noexcept(resources, t.epsM);
            t.epsM = nullptr;
        }
        if (t.epsY) {
            free_owned_device_noexcept(resources, t.epsY);
            t.epsY = nullptr;
        }
        if (t.baseDensityMin) {
            free_owned_device_noexcept(resources, t.baseDensityMin);
            t.baseDensityMin = nullptr;
        }
        t.K = 0;
    }


} // namespace JuicerCuda
