#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "DiffusionHostBehavior.h"
#include "ProcessRoot.h"
#include "ScatterHalation.h"

namespace ScatterHalationValidation {

    class PreparedRouteInputs final {
    public:
        struct Impl;

        explicit PreparedRouteInputs(std::unique_ptr<Impl> impl);
        ~PreparedRouteInputs();
        PreparedRouteInputs(PreparedRouteInputs&&) noexcept;
        PreparedRouteInputs& operator=(PreparedRouteInputs&&) noexcept;
        PreparedRouteInputs(const PreparedRouteInputs&) = delete;
        PreparedRouteInputs& operator=(const PreparedRouteInputs&) = delete;

        [[nodiscard]] JuicerProcess::Root::CudaFramePreparationRequest request(
            int width,
            int height) const;
        [[nodiscard]] JuicerCuda::ResourceManager::SubmissionSnapshot
        submission_snapshot(
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t identity) const;
        [[nodiscard]] const ScatterHalationFrameDescriptor*
        scatter_descriptor() const noexcept;
        [[nodiscard]] const Spektrafilm::DiffusionFrameSetDescriptor*
        diffusion_frame_set() const noexcept;

    private:
        std::unique_ptr<Impl> _impl;
    };

    PreparedRouteInputs build_prepared_route_inputs(
        Spektrafilm::ScanRoute route,
        const ScatterHalationControls& controls,
        const Spektrafilm::DiffusionFilterAuthoredControls& cameraDiffusion,
        float pixelSizeUm,
        int width,
        int height);

    struct Arguments {
        std::string caseGroup;
        std::filesystem::path fixtureRoot;
        std::filesystem::path resourceRoot;
        std::filesystem::path scratchRoot;
        std::filesystem::path performanceOutput;
        std::string performanceMode = "active";
        int deviceIndex = 0;
        int performanceWidth = 1920;
        int performanceHeight = 1080;
        int warmupCount = 10;
        int sampleCount = 30;
        int holdMilliseconds = 0;
    };

    struct CaseResult {
        std::string name;
        bool passed = false;
        std::string detail;
    };

    class Results final {
    public:
        void record(std::string name, bool passed, std::string detail);
        [[nodiscard]] int failure_count() const noexcept;
        [[nodiscard]] const std::vector<CaseResult>& cases() const noexcept;

    private:
        std::vector<CaseResult> _cases;
    };

    void run_focused_cuda_gate_3_rows(
        const Arguments& arguments,
        Results& results);
    void run_unpublished_route_rows(
        const Arguments& arguments,
        Results& results);
    void run_scanner_post_effect_cuda_rows(Results& results);
    void run_captured_carrier_rows(
        const Arguments& arguments,
        Results& results);
    void run_zero_work_rows(const Arguments& arguments, Results& results);
    void run_lifecycle_rows(const Arguments& arguments, Results& results);
    void run_performance_rows(const Arguments& arguments, Results& results);

} // namespace ScatterHalationValidation
