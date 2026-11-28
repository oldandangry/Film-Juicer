#include "film_reference.h"

#include <array>
#include <bit>
#include <cmath>
#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace GrainPrototype {
    namespace {
        void read_words(const std::filesystem::path& path, float* data, std::size_t count) {
            static_assert(std::endian::native == std::endian::little);
            static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
            if (!input)
                throw std::ios_base::failure("open film capture " + path.string());
            if (input.tellg() != bytes)
                throw std::invalid_argument("unexpected film capture length " + path.string());
            input.seekg(0);
            if (!input.read(reinterpret_cast<char*>(data), bytes))
                throw std::ios_base::failure("read film capture " + path.string());
            for (std::size_t i = 0; i < count; ++i) {
                if (!std::isfinite(data[i]))
                    throw std::invalid_argument("nonfinite word in film capture " + path.string());
            }
        }

        std::string footprint_file(float width) {
            if (!std::isfinite(width) || width < 0.25F || width > 37.5F)
                throw std::invalid_argument("unsupported film footprint width");
            std::array<char, 32> buffer{};
            const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), width);
            if (result.ec != std::errc{})
                throw std::invalid_argument("invalid film footprint width");
            return "footprint-" + std::string(buffer.data(), result.ptr) + ".f32";
        }
    } // namespace

    static FilmTables read_bank_tables(const std::filesystem::path& directory,
                                       const std::filesystem::path& tableDirectory,
                                       float footprint) {
        const auto tableFile = footprint_file(footprint);
        constexpr std::size_t recordWords = 3 + kLevels;
        constexpr std::size_t footprintWords = std::size_t{2} * kLevels;
        std::array<float, kRecords * recordWords> records{};
        std::array<float, kRecords * footprintWords> pitch{};
        read_words(directory / "records.f32", records.data(), records.size());
        read_words(tableDirectory / tableFile, pitch.data(), pitch.size());
        FilmTables tables{};
        tables.footprint = footprint;
        for (std::size_t channel = 0; channel < kRecords; ++channel) {
            const auto recordStart = channel * recordWords;
            const auto pitchStart = channel * footprintWords;
            auto& record = tables.records[channel];
            record.minimumGross = records.at(recordStart);
            record.maximumGross = records.at(recordStart + 1);
            record.offset = records.at(recordStart + 2);
            if (record.maximumGross - record.minimumGross < 1e-6F)
                throw std::invalid_argument("invalid film capture density range");
            for (std::size_t i = 0; i < kLevels; ++i) {
                const FilmLevel level{records.at(recordStart + 3 + i), pitch.at(pitchStart + i), pitch.at(pitchStart + kLevels + i)};
                // Swift and Halide have different floors for variance normalization.
                // Only admit tables whose convex blends stay above both floors.
                if (level.meanLight <= 0 || level.meanLight > 1 || level.correlation < -0.9997F ||
                    level.correlation > 1)
                    throw std::invalid_argument("film capture outside light or correlation domain");
                record.levels[i] = level;
            }
        }
        return tables;
    }

    FilmTables read_film_tables(const std::filesystem::path& directory, float footprint, const std::filesystem::path& tableDirectory) {
        return read_bank_tables(directory, tableDirectory.empty() ? directory : tableDirectory, footprint);
    }

    FilmBank read_film_bank(const std::filesystem::path& directory, float footprint, const std::filesystem::path& tableDirectory) {
        FilmBank bank{};
        bank.tables = read_film_tables(directory, footprint, tableDirectory);
        bank.sums.resize(kBankEntries);
        read_words(directory / "sums.f32", bank.sums.data(), bank.sums.size());
        return bank;
    }
} // namespace GrainPrototype
