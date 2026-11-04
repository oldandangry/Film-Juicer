#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace JuicerTestSupport {
    namespace fs = std::filesystem;

    inline std::optional<std::string> test_fixture_retention_setting() {
#if defined(_WIN32)
        std::size_t length = 0;
        if (getenv_s(&length, nullptr, 0, "JUICER_TEST_RETAIN_FIXTURES") != 0)
            throw std::runtime_error("read test fixture retention setting");
        if (length == 0)
            return std::nullopt;
        std::string value(length, '\0');
        if (getenv_s(&length, value.data(), value.size(), "JUICER_TEST_RETAIN_FIXTURES") != 0)
            throw std::runtime_error("read test fixture retention setting");
        value.resize(length - 1);
        return value;
#else
        const char* const value = std::getenv("JUICER_TEST_RETAIN_FIXTURES");
        return value ? std::optional<std::string>(value) : std::nullopt;
#endif
    }

    class TestAssetWorkspaces final {
    public:
        TestAssetWorkspaces() = default;
        TestAssetWorkspaces(const TestAssetWorkspaces&) = delete;
        TestAssetWorkspaces& operator=(const TestAssetWorkspaces&) = delete;
        TestAssetWorkspaces(TestAssetWorkspaces&&) = delete;
        TestAssetWorkspaces& operator=(TestAssetWorkspaces&&) = delete;

        fs::path create(const fs::path& prefix) {
            const auto scratch = prefix.parent_path();
            const auto label = prefix.filename();
            fs::create_directories(scratch);
            const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto sequence = sequence_++;
            for (unsigned attempt = 0;; ++attempt) {
                auto name = label;
                name += fs::path("-" + std::to_string(tick) + "-" + std::to_string(sequence) + "-" + std::to_string(attempt));
                const auto workspace = scratch / name;
                std::error_code error;
                if (fs::create_directory(workspace, error)) {
                    workspaces_.push_back(workspace);
                    return workspace;
                }
                if (error)
                    throw std::filesystem::filesystem_error("create unique test asset workspace", workspace, error);
            }
        }

        int finish(int testResult, bool ownersDrained) {
            if (testResult != 0)
                return testResult;
            if (!ownersDrained)
                return 1;

            if (test_fixture_retention_setting() == "1")
                return 0;

            bool removed = true;
            for (const auto& workspace : workspaces_) {
                std::error_code error;
                fs::remove_all(workspace, error);
                if (error) {
                    removed = false;
                    std::fprintf(stderr, "test asset workspace cleanup failed: %s: %s\n", workspace.string().c_str(), error.message().c_str());
                }
            }
            if (removed)
                workspaces_.clear();
            return removed ? 0 : 1;
        }

    private:
        std::vector<fs::path> workspaces_;
        std::uint64_t sequence_{};
    };

    inline TestAssetWorkspaces test_asset_workspaces;

    inline void copy_resource_files(
        const fs::path& source,
        const fs::path& destination,
        std::span<const std::string_view> resources) {
        for (const auto resource : resources) {
            const auto relative = fs::path(resource);
            const auto target = destination / relative;
            fs::create_directories(target.parent_path());
            fs::copy_file(source / relative, target, fs::copy_options::overwrite_existing);
        }
    }

    inline void copy_reconstruction_assets(const fs::path& source, const fs::path& destination) {
        constexpr std::array<std::string_view, 4> resources{
            "cie1931_2deg.csv",
            "luts/spectral_upsampling/irradiance_xy_tc.npy",
            "luts/spectral_upsampling/arctic2026beta04_reflectance_xy_tc.npy",
            "luts/spectral_upsampling/mallett2019_basis.npy"};
        copy_resource_files(source, destination, resources);
    }

    inline void copy_direct_preparation_assets(const fs::path& source, const fs::path& destination) {
        // These consumers select the negative defaults and build the complete
        // illuminant table, which reads both filter source files as well.
        constexpr std::array<std::string_view, 9> resources{
            "profiles/kodak_portra_400.json",
            "profiles/kodak_portra_endura.json",
            "illuminants/D50.csv",
            "illuminants/D55.csv",
            "illuminants/D65.csv",
            "illuminants/K75P.csv",
            "illuminants/T.csv",
            "filters/heat_absorbing/schott/KG3.csv",
            "filters/lens_transmission/canon/canon_24_f28_is.csv"};
        copy_resource_files(source, destination, resources);
        copy_reconstruction_assets(source, destination);
    }

    inline void copy_print_preparation_assets(const fs::path& source, const fs::path& destination) {
        copy_direct_preparation_assets(source, destination);
        copy_resource_files(source, destination, std::array<std::string_view, 1>{"filters/neutral_print_filters.json"});
    }

    inline void copy_positive_print_preparation_assets(const fs::path& source, const fs::path& destination) {
        copy_print_preparation_assets(source, destination);
        copy_resource_files(source, destination, std::array<std::string_view, 1>{"profiles/fujifilm_provia_100f.json"});
    }

} // namespace JuicerTestSupport
