#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <cuda_runtime_api.h>

namespace GrainPrototype {
    constexpr int kTileSide = 256;
    constexpr int kTileStride = kTileSide + 1;
    constexpr int kTileEntries = kTileStride * kTileStride;
    constexpr int kLevels = 17;
    constexpr int kRecords = 3;
    constexpr int kPlacementBlock = 128;
    constexpr int kPlacementMargin = 64;
    constexpr int kSeam = 16;
    constexpr int kLayings = 2;
    constexpr std::size_t kBankEntries = static_cast<std::size_t>(kRecords) * kLevels * kTileEntries;

    struct FilmLevel {
        float meanLight;
        float meanDensity;
        float correlation;
    };

    struct FilmRecord {
        float minimumGross;
        float maximumGross;
        float offset;
        FilmLevel levels[kLevels];
    };

    // Frozen single-footprint view of a bank. The planes and SAT are C/M/Y.
    struct FilmTables {
        FilmRecord records[kRecords];
        float footprint;
    };

    struct FilmBank {
        std::vector<float> sums;
        FilmTables tables;
    };

    struct alignas(16) FilmQuad {
        float topLeft;
        float topRight;
        float bottomLeft;
        float bottomRight;
    };
    static_assert(sizeof(FilmQuad) == 16 && alignof(FilmQuad) == 16);
    constexpr std::size_t kQuadEntries = static_cast<std::size_t>(kRecords) * kLevels * kTileSide * kTileSide;
    std::vector<FilmQuad> pack_film_quads(const FilmBank& bank);

    struct FilmFrame {
        float pitch = 0.F;
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        std::uint32_t seed = 0;
        bool enabled = false;
        float amounts[kRecords]{1.F, 1.F, 1.F};
        std::uint32_t nextSeed = 0;
        float temporalFraction = 0.F;
        float chromaFraction = 1.F;
    };

    enum class FilmVariant : std::uint8_t {
        Reference,
        Scalar,
        Reuse,
        Paired,
        Quad
    };
    enum class FilmBlock : std::uint8_t {
        Block32x4,
        Block16x8,
        Block32x8
    };

    struct FilmPlanes {
        float* cyan;
        float* magenta;
        float* yellow;
    };

    // Owner is scoped to a single current CUDA context. The caller drains every
    // stream that borrowed this immutable bank before the owner leaves scope.
    class DeviceFilmBank {
    public:
        DeviceFilmBank() = default;
        DeviceFilmBank(const DeviceFilmBank&) = delete;
        DeviceFilmBank& operator=(const DeviceFilmBank&) = delete;
        ~DeviceFilmBank();

        cudaError_t upload(const FilmBank& bank);
        [[nodiscard]] bool resident() const {
            return sums_ != nullptr;
        }
        [[nodiscard]] const float* sums() const {
            return sums_;
        }
        [[nodiscard]] const FilmTables& tables() const {
            return tables_;
        }
        [[nodiscard]] void* context() const {
            return context_;
        }

    private:
        float* sums_ = nullptr;
        void* context_ = nullptr;
        FilmTables tables_{};
    };

    // Immutable quads belong to the exact CUDA context that received them.
    class DeviceFilmQuads {
    public:
        DeviceFilmQuads() = default;
        DeviceFilmQuads(const DeviceFilmQuads&) = delete;
        DeviceFilmQuads& operator=(const DeviceFilmQuads&) = delete;
        ~DeviceFilmQuads();
        cudaError_t upload(const std::vector<FilmQuad>& quads);
        [[nodiscard]] bool resident() const {
            return quads_ != nullptr;
        }
        [[nodiscard]] const FilmQuad* quads() const {
            return quads_;
        }
        [[nodiscard]] void* context() const {
            return context_;
        }

    private:
        FilmQuad* quads_ = nullptr;
        void* context_ = nullptr;
    };

    cudaError_t apply_film_quads(const DeviceFilmQuads& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream, FilmBlock block);

    // Only the three density planes are written. A disabled frame needs neither
    // a CUDA context nor an uploaded bank.
    cudaError_t apply_film_reference(const DeviceFilmBank& bank, FilmFrame frame, FilmPlanes planes, cudaStream_t stream);

    cudaError_t apply_film_reference(const DeviceFilmBank& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream);

    cudaError_t apply_film_variant(const DeviceFilmBank& bank, const FilmTables& tables, FilmFrame frame, FilmPlanes planes, cudaStream_t stream, FilmVariant variant, FilmBlock block);

    FilmTables read_film_tables(const std::filesystem::path& directory, float footprint);

    // Fixed little-endian research capture. Throws on I/O or representation failure.
    FilmBank read_film_bank(const std::filesystem::path& directory, float footprint, const std::filesystem::path& tableDirectory = {});
} // namespace GrainPrototype
