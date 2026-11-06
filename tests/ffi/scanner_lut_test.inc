#include "host_preparation_fixture.h"

namespace ScannerLutTest {
    constexpr std::array kArrayNames{"log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"};
    struct Literal {
        std::array<std::vector<float>, 7> spectra;
        Scanner::MediumView medium;
        std::uint32_t resolution;

        explicit Literal(const Json& input)
            : resolution(input.at("resolution")) {
            constexpr std::array names{"eps_c", "eps_m", "eps_y", "weighted_x", "weighted_y", "weighted_z", "baseline"};
            for (std::size_t i = 0; i < spectra.size(); ++i)
                for (const auto& word : input.at(names[i]))
                    spectra[i].push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
            medium.medium = static_cast<Scanner::ScannerMedium>(input.at("medium").get<std::uint8_t>());
            medium.tables = {spectra[0], spectra[1], spectra[2], spectra[3], spectra[4], spectra[5], spectra[6], input.at("sample_count").get<int>(), std::bit_cast<float>(input.at("inv_yn").get<std::uint32_t>()), input.at("has_baseline").get<bool>()};
            copy_words(input.at("range_min_cmy"), medium.range.min_cmy);
            copy_words(input.at("inverse_span_cmy"), medium.range.inv_max_cmy);
        }
    };

    void compare(const Json& actual, const Json& expected) {
        for (const char* field : {"success", "resolution", "diagnostic", "failed_voxel", "failed_channel"})
            require(actual.at(field) == expected.at(field), "exact scanner LUT result/failure metadata");
        const std::size_t resolution = expected.at("resolution");
        for (std::size_t i = 0; i < kArrayNames.size(); ++i) {
            const auto* name = kArrayNames[i];
            const auto side = i < 4 ? resolution : (resolution ? resolution - 1 : 0);
            require(expected.at(name).size() == 3 * side * side * side, "scanner LUT expected complete array count");
            require(actual.at(name) == expected.at(name), "scanner LUT every stored word including signed zero");
        }
    }

    Json input(const Scanner::MediumView& medium, std::uint32_t resolution) {
        const auto& t = medium.tables;
        return {{"resolution", resolution}, {"medium", static_cast<unsigned>(medium.medium)}, {"sample_count", t.K}, {"inv_yn", std::bit_cast<std::uint32_t>(t.invYn)}, {"has_baseline", t.hasBaseline}, {"range_min_cmy", words(medium.range.min_cmy, 3)}, {"inverse_span_cmy", words(medium.range.inv_max_cmy, 3)}, {"eps_c", words(t.epsC.data(), t.epsC.size())}, {"eps_m", words(t.epsM.data(), t.epsM.size())}, {"eps_y", words(t.epsY.data(), t.epsY.size())}, {"weighted_x", words(t.Ax.data(), t.Ax.size())}, {"weighted_y", words(t.Ay.data(), t.Ay.size())}, {"weighted_z", words(t.Az.data(), t.Az.size())}, {"baseline", words(t.baseDensityMin.data(), t.baseDensityMin.size())}};
    }

    FjScannerLutInput wire(const Literal& literal) {
        const auto& m = literal.medium;
        const auto span = [](std::span<const float> v) -> FjFloatSpan {
            return {v.empty() ? nullptr : v.data(), v.size()};
        };
        const auto& t = m.tables;
        FjScannerLutInput out{span(t.epsC), span(t.epsM), span(t.epsY), span(t.Ax), span(t.Ay), span(t.Az), span(t.baseDensityMin), {}, {}, t.K, t.invYn, t.hasBaseline ? 1u : 0u, static_cast<std::uint32_t>(m.medium), literal.resolution};
        std::copy_n(m.range.min_cmy, 3, out.range_min_cmy);
        std::copy_n(m.range.inv_max_cmy, 3, out.inverse_span_cmy);
        return out;
    }
    Json project(const FjScannerLutView& view) {
        const std::array arrays{view.log2_xyz, view.slope_c, view.slope_m, view.slope_y, view.cell_min, view.cell_max};
        Json out{{"success", true}, {"resolution", view.resolution}, {"diagnostic", ""}, {"failed_voxel", UINT64_MAX}, {"failed_channel", UINT32_MAX}};
        for (std::size_t i = 0; i < arrays.size(); ++i)
            out[kArrayNames[i]] = words(arrays[i].data, arrays[i].count);
        return out;
    }
    void metadata(const FjScannerLutFailure& f, unsigned reason = 0, unsigned buffer = 0, std::uint64_t index = UINT64_MAX, unsigned channel = UINT32_MAX) {
        require(f.reason == reason && f.buffer == buffer && f.index == index && f.channel == channel, "exact scanner LUT failure fields");
    }
    void values(const Json& rows) {
        for (const auto& row : rows) {
            const auto id = row.contains("id") ? row.at("id").get<std::string>() : "route/" + std::to_string(row.at("route").get<unsigned>());
            JuicerTests::CaseEvent case_event("c11", id);
            const Literal literal(row.at("input"));
            const auto input = wire(literal);
            FjScannerLutOwner* owner = reinterpret_cast<FjScannerLutOwner*>(std::uintptr_t{0x5555});
            FjScannerLutFailure f{99, 99, 99, 99};
            std::array<char, 256> text{};
            FjErrorBuffer error{text.data(), text.size(), 99};
            const auto result = fj_test_scanner_lut_c_acquire(&input, &owner, &f, &error);
            const auto& expected = row.at("native");
            if (!expected.at("success").get<bool>()) {
                status(result, FJ_STATUS_PREPARATION_FAILURE);
                require(!owner, "failed LUT publishes no owner");
                const bool resolution = input.resolution < 2;
                metadata(f, resolution ? 2u : 5u, 0, expected.at("failed_voxel"), expected.at("failed_channel"));
                require(std::string(text.data(), error.length) == expected.at("diagnostic").get<std::string>(), "native LUT diagnostic continuity");
                case_event.passed();
                continue;
            }
            status(result);
            metadata(f);
            require(owner && error.length == 0 && text[0] == 0, "successful complete LUT owner");
            FjScannerLutView borrowed{};
            status(fj_test_scanner_lut_c_view(owner, &borrowed, &f, &error));
            metadata(f);
            compare(project(borrowed), expected);
            FjScannerLutView repeated{};
            status(fj_test_scanner_lut_c_view(owner, &repeated, &f, nullptr));
            metadata(f);
            require(borrowed.log2_xyz.data == repeated.log2_xyz.data, "immutable repeated view does no copy");
            compare(project(repeated), expected);
            status(fj_test_scanner_lut_c_release(owner, nullptr));
            owner = nullptr;
            case_event.passed();
        }
        require(fj_test_scanner_lut_live_owners() == 0, "value cohort drops every host owner");
    }
    void callback_failure_diagnostics(const Literal& literal) {
        struct Reply {
            FjScannerLutFailure failure{FJ_SCANNER_LUT_REASON_ALLOCATION, UINT32_MAX, UINT64_MAX, 11};
            std::string_view diagnostic;
        } reply;
        const FjHostPreparation preparation{
            &reply,
            [](void* user, const FjScannerLutInput*, FjScannerLutView* view, FjScannerLutFailure* failure, FjErrorBuffer* error) -> FjStatus {
                const auto& reply = *static_cast<Reply*>(user);
                *view = {};
                *failure = reply.failure;
                std::copy(reply.diagnostic.begin(), reply.diagnostic.end(), error->data);
                error->length = reply.diagnostic.size();
                error->data[error->length] = '\0';
                return {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0};
            },
            [](void*, const FjGaussianKernelInput*, FjGaussianKernelView*, FjGaussianKernelFailure*, FjErrorBuffer*) -> FjStatus {
                return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
            }};
        JuicerTests::Cohort diagnostic_rows("scanner_lut.callback_diagnostic", {"allocation-failed", "storage-unavailable", "empty", "size-failure"});
        std::size_t diagnostic_index = 0;
        for (const char* diagnostic : {"scanner LUT candidate storage allocation failed", "scanner LUT candidate storage is unavailable", ""}) {
            auto row_event = diagnostic_rows.case_event(diagnostic_index == 0 ? "allocation-failed" : diagnostic_index == 1 ? "storage-unavailable"
                                                                                                                            : "empty");
            reply.diagnostic = diagnostic;
            FjScannerLutView view{};
            JuicerCuda::Failure failure;
            require(!JuicerCuda::prepare_scanner_lut(preparation, literal.medium, literal.resolution, view, failure), "typed scanner allocation callback rejects preparation");
            status(failure.status, FJ_STATUS_ALLOCATION_FAILURE);
            require(failure.diagnostic == diagnostic, "scanner allocation category is independent of forwarded diagnostic wording");
            require(view.resolution == 0 && !view.log2_xyz.data && !view.log2_xyz.count, "scanner callback failure leaves no published view");
            row_event.passed();
            ++diagnostic_index;
        }
        reply.failure.reason = FJ_SCANNER_LUT_REASON_SIZE;
        reply.diagnostic = "scanner LUT candidate storage allocation failed";
        auto size_event = diagnostic_rows.case_event("size-failure");
        FjScannerLutView view{};
        JuicerCuda::Failure failure;
        bool rejected = false;
        try {
            (void)JuicerCuda::prepare_scanner_lut(preparation, literal.medium, literal.resolution, view, failure);
        } catch (const JuicerCuda::ExecutionFailure& error) {
            status(error.failure.status, FJ_STATUS_INTERNAL_FAILURE);
            rejected = true;
        }
        require(rejected, "scanner buffer 11 remains invalid for a SIZE failure despite the diagnostic");
        require(reply.failure.reason == FJ_SCANNER_LUT_REASON_SIZE, "scanner callback SIZE setup");
        size_event.passed();
    }
    void boundary(const Json& captured) {
        JuicerTests::CaseEvent boundary_event("cpp", "scanner-lut/boundary");
        const auto& row = captured.at("cases")[0];
        Literal literal(row.at("input"));
        callback_failure_diagnostics(literal);
        const auto original = wire(literal);
        struct Rejection {
            unsigned category = FJ_STATUS_SUCCESS;
            unsigned reason = FJ_SCANNER_LUT_REASON_NONE;
            unsigned buffer = 0;
        };
        const auto reject = [&](FjScannerLutInput input, Rejection expected) {
            FjScannerLutOwner* owner = reinterpret_cast<FjScannerLutOwner*>(std::uintptr_t{0x5555});
            FjScannerLutFailure f{99, 99, 99, 99};
            std::array<char, 256> text{};
            FjErrorBuffer error{text.data(), text.size(), 99};
            status(fj_test_scanner_lut_acquire(&input, &owner, &f, &error), expected.category);
            require(!owner, "rejected LUT clears owner");
            metadata(f, expected.reason, expected.buffer);
            require(error.length < error.capacity && text[error.length] == 0, "LUT diagnostics bounded and terminated");
        };
        JuicerTests::Cohort raw_input_rows("scanner_lut.raw_input", {"resolution-small", "resolution-overflow", "medium-enum", "baseline-bool", "eps-c-null", "weighted-z-count", "eps-m-pointer", "eps-y-overflow"});
        auto input = original;
        input.resolution = 1;
        {
            auto raw_event = raw_input_rows.case_event("resolution-small");
            reject(input, {FJ_STATUS_PREPARATION_FAILURE, FJ_SCANNER_LUT_REASON_RESOLUTION});
            raw_event.passed();
        }
        input = original;
        input.resolution = UINT32_MAX;
        {
            auto raw_event = raw_input_rows.case_event("resolution-overflow");
            reject(input, {FJ_STATUS_ALLOCATION_FAILURE, FJ_SCANNER_LUT_REASON_SIZE, 1});
            raw_event.passed();
        }
        input = original;
        input.medium = 256;
        {
            auto raw_event = raw_input_rows.case_event("medium-enum");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        input = original;
        input.has_baseline = 2;
        {
            auto raw_event = raw_input_rows.case_event("baseline-bool");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        input = original;
        input.eps_c = {nullptr, 3};
        {
            auto raw_event = raw_input_rows.case_event("eps-c-null");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        input = original;
        input.weighted_z.count = 2;
        {
            auto raw_event = raw_input_rows.case_event("weighted-z-count");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        input = original;
        input.eps_m = {reinterpret_cast<const float*>(std::uintptr_t{1}), 3};
        {
            auto raw_event = raw_input_rows.case_event("eps-m-pointer");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        input = original;
        input.eps_y.count = SIZE_MAX;
        {
            auto raw_event = raw_input_rows.case_event("eps-y-overflow");
            reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
            raw_event.passed();
        }
        JuicerTests::Cohort allocation_faults("scanner_lut.allocation_fault", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"});
        for (unsigned buffer = 1; buffer <= 10; ++buffer) {
            auto row_event = allocation_faults.case_event(std::to_string(buffer));
            status(fj_test_scanner_lut_fault(0, buffer + 1));
            reject(original, {FJ_STATUS_ALLOCATION_FAILURE, FJ_SCANNER_LUT_REASON_ALLOCATION, buffer});
            require(fj_test_scanner_lut_live_owners() == 0, "every failed reservation drops provisional buffers");
            row_event.passed();
        }
        JuicerTests::Cohort panic_operations("scanner_lut.panic_operation", 3);
        for (unsigned operation = 0; operation < 3; ++operation) {
            auto row_event = panic_operations.case_event(std::to_string(operation));
            FjScannerLutOwner* owner = nullptr;
            FjScannerLutFailure f{};
            FjScannerLutView view{};
            if (operation)
                status(fj_test_scanner_lut_acquire(&original, &owner, &f, nullptr));
            status(fj_test_scanner_lut_fault(operation, 1));
            status(fj_test_scanner_lut_fault(operation, 12), 2); // Invalid pair preserves the armed panic.
            if (operation == 0) {
                status(fj_test_scanner_lut_acquire(&original, &owner, &f, nullptr), 8);
                metadata(f, 6);
                require(!owner, "acquire panic no publication");
            } else if (operation == 1) {
                status(fj_test_scanner_lut_view(owner, &view, &f, nullptr), 8);
                metadata(f, 6);
                require(view.resolution == 0 && view.cell_max.data == nullptr, "view panic clears all storage");
                status(fj_test_scanner_lut_release(owner, nullptr));
            } else
                status(fj_test_scanner_lut_release(owner, nullptr), 8);
            require(fj_test_scanner_lut_live_owners() == 0, "contained panic consumes/drops its owned storage");
            row_event.passed();
        }
        JuicerTests::Cohort invalid_arguments("scanner_lut.invalid_argument", {"null-input", "invalid-failure-peer", "null-owner", "malformed-diagnostic", "null-view", "truncated-diagnostic", "zero-capacity-diagnostic"});
        FjScannerLutOwner* owner = nullptr;
        FjScannerLutFailure f{99, 99, 99, 99};
        FjScannerLutView view{};
        {
            auto invalid_event = invalid_arguments.case_event("null-input");
            status(fj_test_scanner_lut_acquire(nullptr, &owner, &f, nullptr), 2);
            metadata(f, 1);
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("invalid-failure-peer");
            owner = reinterpret_cast<FjScannerLutOwner*>(std::uintptr_t{0x5555});
            status(fj_test_scanner_lut_acquire(&original, &owner, reinterpret_cast<FjScannerLutFailure*>(std::uintptr_t{1}), nullptr), 2);
            require(!owner, "malformed peer cannot prevent owner clearing");
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("null-owner");
            f = {99, 99, 99, 99};
            status(fj_test_scanner_lut_acquire(&original, nullptr, &f, nullptr), 2);
            metadata(f, 1);
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("malformed-diagnostic");
            FjErrorBuffer malformed{nullptr, 1, 99};
            status(fj_test_scanner_lut_acquire(&original, &owner, &f, &malformed), 2);
            metadata(f, 1);
            require(!owner && malformed.length == 0, "diagnostic admission clears independent outputs");
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("null-view");
            std::memset(&view, 0x55, sizeof(view));
            status(fj_test_scanner_lut_view(nullptr, &view, &f, nullptr), 2);
            metadata(f, 1);
            require(view.resolution == 0 && view.log2_xyz.data == nullptr && view.slope_c.count == 0 && view.slope_m.data == nullptr && view.slope_y.count == 0 && view.cell_min.data == nullptr && view.cell_max.count == 0, "all six view fields independently cleared");
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("truncated-diagnostic");
            std::array<char, 4> text{};
            FjErrorBuffer truncated{text.data(), text.size(), 99};
            status(fj_test_scanner_lut_acquire(nullptr, &owner, &f, &truncated), 2);
            require(truncated.length == 3 && text[3] == 0, "truncated diagnostic NUL");
            invalid_event.passed();
        }
        {
            auto invalid_event = invalid_arguments.case_event("zero-capacity-diagnostic");
            FjErrorBuffer zero{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
            status(fj_test_scanner_lut_acquire(nullptr, &owner, &f, &zero), 2);
            require(zero.length == 0, "zero-capacity diagnostic ignores backing");
            invalid_event.passed();
        }
        // Ignored baseline and nonpositive-K backing are never admitted or read.
        JuicerTests::Cohort ignored_spans("scanner_lut.ignored_span", {"disabled-baseline", "nonpositive-sample-count"});
        {
            auto ignored_event = ignored_spans.case_event("disabled-baseline");
            input = original;
            input.has_baseline = 0;
            input.baseline_min = {reinterpret_cast<const float*>(std::uintptr_t{1}), SIZE_MAX};
            status(fj_test_scanner_lut_acquire(&input, &owner, &f, nullptr));
            status(fj_test_scanner_lut_release(owner, nullptr));
            ignored_event.passed();
        }
        {
            auto ignored_event = ignored_spans.case_event("nonpositive-sample-count");
            input = original;
            input.sample_count = 0;
            input.eps_c = input.eps_m = input.eps_y = input.weighted_x = input.weighted_y = input.weighted_z = input.baseline_min = {reinterpret_cast<const float*>(std::uintptr_t{1}), SIZE_MAX};
            status(fj_test_scanner_lut_fault(0, 11));
            status(fj_test_scanner_lut_acquire(&input, &owner, &f, nullptr), 3);
            metadata(f, 5, 0, 0, 0);
            status(fj_test_scanner_lut_acquire(&original, &owner, &f, nullptr)); // Unreached reservation fault was consumed at admission.
            FjErrorBuffer malformed{nullptr, 1, 99};
            status(fj_test_scanner_lut_release(owner, &malformed), 2);
            require(fj_test_scanner_lut_live_owners() == 0, "invalid diagnostics consume legitimate release");
            ignored_event.passed();
        }
        status(fj_test_scanner_lut_release(nullptr, nullptr));
        {
            Literal expiring(row.at("input"));
            const auto wireInput = wire(expiring);
            status(fj_test_scanner_lut_acquire(&wireInput, &owner, &f, nullptr));
            for (auto& spectrum : expiring.spectra) {
                std::fill(spectrum.begin(), spectrum.end(), -999.0f);
                spectrum.clear();
                spectrum.shrink_to_fit();
            }
        }
        status(fj_test_scanner_lut_view(owner, &view, &f, nullptr));
        compare(project(view), row.at("native"));
        status(fj_test_scanner_lut_release(owner, nullptr));
        boundary_event.passed();
    }
    void callback_values(const Json& rows) {
        for (const auto& row : rows) {
            const auto id = row.contains("id") ? row.at("id").get<std::string>() : "route/" + std::to_string(row.at("route").get<unsigned>());
            JuicerTests::CaseEvent case_event("cpp", "scanner-lut/callback/" + id);
            const Literal literal(row.at("input"));
            JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                // Distinct prior values make dropped error forwarding observable.
                JuicerCuda::Failure failure{{FJ_STATUS_CUDA_FAILURE, FJ_API_CUDA_RUNTIME, 1}, "previous failure"};
                FjScannerLutView view{};
                const bool built = JuicerCuda::prepare_scanner_lut(*preparation, literal.medium, literal.resolution, view, failure);
                require(built == row.at("native").at("success").get<bool>(), "callback adapter success/failure");
                if (!built) {
                    status(failure.status, FJ_STATUS_PREPARATION_FAILURE);
                    require(failure.diagnostic == row.at("native").at("diagnostic").get<std::string>(), "callback adapter preserves scanner LUT failure diagnostic");
                } else {
                    compare(project(view), row.at("native"));
                }
                return true;
            });
            case_event.passed();
        }
        require(fj_test_scanner_lut_live_owners() == 0, "callback values release every host owner");
    }
} // namespace ScannerLutTest

namespace ScannerLutTest {
    void check_products(const std::filesystem::path& resources, const Json& captured) {
        JuicerProcess::ProcessRuntime owner;
        owner.create(resources.string().c_str());
        require(captured.at("products").size() == 4, "exact actual scanner product membership");
        for (unsigned route = 0; route < 4; ++route) {
            JuicerTests::CaseEvent case_event("cpp", "scanner-lut/product-route/" + std::to_string(route));
            auto params = PrintPreflashTest::controls();
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.scannerLutResolution = 17;
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> product;
            const auto prepared = prepare_state(owner.borrow().assets, params, product, error);
            const bool print = route % 2 != 0;
            require(prepared.category == FJ_STATUS_SUCCESS && product, diagnostic.data());
            FjPreparedStateView view{};
            status(product->view(view, error));
            const auto& descriptor = view.scanner_lut;
            Scanner::MediumView medium;
            medium.medium = print ? Scanner::ScannerMedium::Print : Scanner::ScannerMedium::Negative;
            const auto& tables = view.scanner_tables;
            const auto as_span = [](FjFloatSpan values) {
                return std::span<const float>(values.data, values.count);
            };
            medium.tables = {as_span(tables.dyes_cmy[0]), as_span(tables.dyes_cmy[1]), as_span(tables.dyes_cmy[2]), as_span(tables.weighted_xyz[0]), as_span(tables.weighted_xyz[1]), as_span(tables.weighted_xyz[2]), as_span(tables.baseline_min), 81, tables.inv_yn, tables.has_baseline != 0};
            for (std::size_t channel = 0; channel < 3; ++channel) {
                medium.range.min_cmy[channel] = print ? view.scanner_bounds.minimum_cmy[channel] : -view.scanner_bounds.minimum_cmy[channel];
                medium.range.inv_max_cmy[channel] = view.scanner_bounds.inv_span_cmy[channel];
            }
            const auto& expected = captured.at("products")[route];
            require(expected.at("route").get<unsigned>() == route && expected.at("descriptor_hash").get<std::uint64_t>() == descriptor.hash, "actual scanner product descriptor identity");
            require(input(medium, descriptor.resolution) == expected.at("input"), "every actual selected scanner table and bound word");
            JuicerTestSupport::with_host_preparation([&](const FjHostPreparation* preparation) {
                JuicerCuda::Failure failure;
                FjScannerLutView lutView{};
                const bool prepared = JuicerCuda::prepare_scanner_lut(*preparation, medium, descriptor.resolution, lutView, failure);
                require(prepared, failure.diagnostic.c_str());
                // All six borrowed arrays are compared before callback backing expires.
                compare(project(lutView), expected.at("native"));
                return true;
            });
            case_event.passed();
        }
    }
    void lifetime(const Json& captured) {
        JuicerTests::CaseEvent lifetime_event("cpp", "scanner-lut/lifetime");
        const Literal literal(captured.at("cases")[0].at("input"));
        const auto in = wire(literal);
        FjScannerLutOwner* first = nullptr;
        FjScannerLutOwner* second = nullptr;
        FjScannerLutFailure f{};
        status(fj_test_scanner_lut_acquire(&in, &first, &f, nullptr));
        status(fj_test_scanner_lut_acquire(&in, &second, &f, nullptr));
        require(first != second && fj_test_scanner_lut_live_owners() == 2, "independent LUT owners");
        status(fj_test_scanner_lut_release(std::exchange(first, nullptr), nullptr));
        FjScannerLutView view{};
        status(fj_test_scanner_lut_view(second, &view, &f, nullptr));
        compare(project(view), captured.at("cases")[0].at("native"));
        status(fj_test_scanner_lut_release(std::exchange(second, nullptr), nullptr));
        require(fj_test_scanner_lut_live_owners() == 0, "independent raw owners release exactly once");
        lifetime_event.passed();
    }
} // namespace ScannerLutTest
