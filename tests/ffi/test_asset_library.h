#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "test_asset_bridge.h"

namespace JuicerAssets {
    class Library {
    public:
        explicit Library(const std::filesystem::path& resourceRoot);
        ~Library();

        std::shared_ptr<const Gamut::InputHullHold>
        input_compression_hull();
        std::shared_ptr<const Gamut::OutputBoundaryHold>
        output_boundary_table(
            const Gamut::OutputGamutTransform& transform,
            std::string& diagnostic);
        NeutralPrintCalibrationResult neutral_print_calibration(
            const std::string& printStock,
            const std::string& printIlluminantKey,
            const std::string& filmStock);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;
        FjAssets* borrow_rust_assets() const noexcept;

    private:
        AssetBridge _bridge;
    };
} // namespace JuicerAssets
