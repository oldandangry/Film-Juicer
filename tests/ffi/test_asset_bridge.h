#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "ProcessRuntime.h"
#include "ProfileCatalog.h"
#include "juicer_legacy_api.h"

struct FjInputHullOwner;
struct FjInputHullView;
struct FjOutputBoundaryOwner;
struct FjOutputBoundaryView;

namespace Gamut {
    struct OutputGamutTransform;
}

namespace JuicerAssets {
    enum class NeutralPrintCalibrationStatus : unsigned char {
        MissingFile,
        MissingEntry,
        Found,
        Malformed
    };

    struct NeutralPrintCalibrationResult {
        NeutralPrintCalibrationStatus status =
            NeutralPrintCalibrationStatus::MissingEntry;
        std::array<float, 3> cmyCc{};
        std::string diagnostic;
    };

#if defined(JUICER_ILLUMINANT_TEST_HOOK)
    namespace IlluminantTest {
        void calibration_result(FjNeutralCalibrationResult& result);
        void frame_abort() noexcept;
    } // namespace IlluminantTest
#endif

    std::string path_diagnostic(const std::filesystem::path& path);

    class AssetBridge;
} // namespace JuicerAssets

namespace Gamut {
    class InputHullPoints final {
    public:
        struct Iterator {
            const float* point;
            std::span<const float, 2> operator*() const noexcept {
                return std::span<const float, 2>(point, 2);
            }
            Iterator& operator++() noexcept {
                point += 2;
                return *this;
            }
            bool operator==(const Iterator&) const = default;
        };
        Iterator begin() const noexcept {
            return {_coordinates.data()};
        }
        Iterator end() const noexcept {
            return {_coordinates.data() + _coordinates.size()};
        }
        std::span<const float, 2> operator[](std::size_t index) const noexcept {
            return std::span<const float, 2>(_coordinates.data() + 2 * index, 2);
        }
        const float* data() const noexcept {
            return _coordinates.data();
        }
        std::size_t size() const noexcept {
            return _coordinates.size() / 2;
        }

    private:
        friend class InputHullHold;
        explicit InputHullPoints(std::span<const float> coordinates) noexcept : _coordinates(coordinates) {}
        std::span<const float> _coordinates;
    };

    class InputHullHold final {
    public:
        ~InputHullHold();
        InputHullHold(const InputHullHold&) = delete;
        InputHullHold& operator=(const InputHullHold&) = delete;
        const InputHullPoints xy;
        const std::array<float, 2> center;
        const std::uint64_t hash;
        const std::uintptr_t entryIdentity;

    private:
        friend class JuicerAssets::AssetBridge;
        InputHullHold(FjInputHullOwner*, const FjInputHullView&) noexcept;
        FjInputHullOwner* _owner;
    };

    class OutputBoundaryHold final {
    public:
        ~OutputBoundaryHold();
        OutputBoundaryHold(const OutputBoundaryHold&) = delete;
        OutputBoundaryHold& operator=(const OutputBoundaryHold&) = delete;
        const std::span<const float> cmax;
        const std::uint64_t transformHash, contractHash, hash;
        const std::uintptr_t entryIdentity;

    private:
        friend class JuicerAssets::AssetBridge;
        OutputBoundaryHold(FjOutputBoundaryOwner*, const FjOutputBoundaryView&) noexcept;
        FjOutputBoundaryOwner* _owner;
    };
} // namespace Gamut

namespace JuicerAssets {
    class AssetBridge final {
    public:
        explicit AssetBridge(const std::filesystem::path& resourceRoot);
        ~AssetBridge();
        AssetBridge(const AssetBridge&) = delete;
        AssetBridge& operator=(const AssetBridge&) = delete;

        std::shared_ptr<const Gamut::InputHullHold> input_hull();
        std::shared_ptr<const Gamut::OutputBoundaryHold> output_boundary(const Gamut::OutputGamutTransform&, std::string& diagnostic);
        NeutralPrintCalibrationResult neutral_print_calibration(const std::string& printStock,
                                                                const std::string& illuminant,
                                                                const std::string& filmStock);
        Spektrafilm::ProfileCatalog load_catalog();
        FjStatus film_defaults(const std::string& key, FjFilmDigest& out, FjErrorBuffer& error);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;
        FjAssets* borrow_assets() const noexcept;

    private:
        FjAssets* _assets = nullptr;
        FjCatalog* _catalog = nullptr;
    };
} // namespace JuicerAssets
