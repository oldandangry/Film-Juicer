#include "test_asset_library.h"

#include <filesystem>
namespace JuicerAssets {
    Library::Library(const std::filesystem::path& resourceRoot) : _bridge(resourceRoot) {}

    Library::~Library() = default;

    FjAssets* Library::borrow_rust_assets() const noexcept {
        return _bridge.borrow_assets();
    }

    FjStatus Library::close(FjErrorBuffer* error) noexcept {
        return _bridge.close(error);
    }

    std::shared_ptr<const Gamut::InputHullHold> Library::input_compression_hull() {
        return _bridge.input_hull();
    }

    std::shared_ptr<const Gamut::OutputBoundaryHold> Library::output_boundary_table(
        const Gamut::OutputGamutTransform& transform, std::string& diagnostic) {
        return _bridge.output_boundary(transform, diagnostic);
    }

    NeutralPrintCalibrationResult Library::neutral_print_calibration(const std::string& printStock,
                                                                     const std::string& printIlluminantKey,
                                                                     const std::string& filmStock) {
        return _bridge.neutral_print_calibration(printStock, printIlluminantKey, filmStock);
    }

    FjStatus Library::release_cached_payloads(FjErrorBuffer* error) noexcept {
        return _bridge.release_cached_payloads(error);
    }

} // namespace JuicerAssets
