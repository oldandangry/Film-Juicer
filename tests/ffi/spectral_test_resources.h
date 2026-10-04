#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpectralFixture {
    inline constexpr std::size_t kLutCount = std::size_t{192} * 192u * 81u;
    inline constexpr const char* kHanatos = "irradiance_xy_tc.npy";
    inline constexpr const char* kArctic = "arctic2026beta04_reflectance_xy_tc.npy";
    inline constexpr const char* kMallett = "mallett2019_basis.npy";

    inline std::filesystem::path source_path(const std::filesystem::path& root, const char* name) {
        return root / "luts" / "spectral_upsampling" / name;
    }

    struct NpyRepresentation {
        unsigned elementBytes;
        std::size_t sampleCount;
        std::string shape = "192,192,81";
        unsigned version = 1;
    };
    struct CmfRepresentation {
        int rows = 81;
        int wavelengthOffsetNm = 0;
    };

    // Synthetic representation contracts, independent of the product decoder.
    inline void write_npy(const std::filesystem::path& root, const char* name, const NpyRepresentation& representation, const std::vector<std::uint64_t>& prefix = {}) {
        const auto width = representation.elementBytes;
        const auto count = representation.sampleCount;
        const auto& shape = representation.shape;
        const auto version = representation.version;
        const auto path = source_path(root, name);
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const std::string header = "{\"shape\": (" + shape + "), \"fortran_order\": False, \"descr\": \"<f" + std::to_string(width) + "\"}\n";
        const std::array<unsigned char, 10> magic{0x93, 'N', 'U', 'M', 'P', 'Y', static_cast<unsigned char>(version), 0, static_cast<unsigned char>(header.size() & 255u), static_cast<unsigned char>(header.size() >> 8u)};
        file.write(reinterpret_cast<const char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        const std::uint64_t one = width == 2 ? 0x3c00u : (width == 4 ? 0x3f800000u : UINT64_C(0x3ff0000000000000));
        std::array<char, 4096> chunk{};
        std::size_t used = 0;
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint64_t bits = index < prefix.size() ? prefix[index] : one;
            for (unsigned byte = 0; byte < width; ++byte) {
                chunk[used++] = static_cast<char>((bits >> (byte * 8u)) & 255u);
            }
            if (used == chunk.size()) {
                file.write(chunk.data(), static_cast<std::streamsize>(used));
                used = 0;
            }
        }
        file.write(chunk.data(), static_cast<std::streamsize>(used));
    }

    inline void write_cmf(const std::filesystem::path& root, CmfRepresentation representation = {}) {
        std::filesystem::create_directories(root);
        std::ofstream file(root / "cie1931_2deg.csv");
        file.exceptions(std::ios::badbit | std::ios::failbit);
        for (int row = 0; row < representation.rows; ++row) {
            file << 380 + row * 5 + representation.wavelengthOffsetNm << ",1,2,3 # source row\n";
        }
    }
} // namespace SpectralFixture
