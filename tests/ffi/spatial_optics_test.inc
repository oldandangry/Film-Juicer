namespace OpticsTest {
    using Prepare = FjStatus (*)(const FjSpatialOpticsParams*, FjSpatialOpticsRecipe*, FjSpatialOpticsFailure*, FjErrorBuffer*);
    Json double_words(const double* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint64_t>(values[i]));
        return out;
    }
    FjDiffusionParams diffusion_params(const Json& input) {
        FjDiffusionParams out{};
        out.active = input.at("active").get<bool>() ? 1u : 0u;
        out.family = input.at("family").get<std::uint32_t>();
        const auto& v = input.at("f64");
        const std::array<double*, 9> fields{&out.strength, &out.spatial_scale, &out.halo_warmth, &out.core_intensity, &out.core_size, &out.halo_intensity, &out.halo_size, &out.bloom_intensity, &out.bloom_size};
        for (std::size_t i = 0; i < fields.size(); ++i)
            *fields[i] = std::bit_cast<double>(v[i].get<std::uint64_t>());
        return out;
    }
    FjSpatialOpticsParams params(const Json& input) {
        FjSpatialOpticsParams out{};
        out.camera_diffusion = diffusion_params(input.at("camera"));
        out.enlarger_diffusion = diffusion_params(input.at("enlarger"));
        out.camera_lens_blur_um = std::bit_cast<float>(input.at("lens_f32").get<std::uint32_t>());
        out.print_domain = input.at("print").get<bool>() ? 1u : 0u;
        auto& h = out.scatter_halation;
        h.active = input.at("halation_active").get<bool>() ? 1u : 0u;
        const auto& v = input.at("halation_f32");
        h.scatter_amount = std::bit_cast<float>(v[0].get<std::uint32_t>());
        h.scatter_spatial_scale = std::bit_cast<float>(v[1].get<std::uint32_t>());
        h.halation_amount = std::bit_cast<float>(v[2].get<std::uint32_t>());
        h.halation_spatial_scale = std::bit_cast<float>(v[3].get<std::uint32_t>());
        copy_words(input.at("first_sigma_f32"), h.first_sigma_um);
        copy_words(input.at("primary_f32"), h.primary_amount);
        return out;
    }
    Json project(const FjDiffusionRecipe& v) {
        const double values[]{v.scatter_fraction, v.group_weights[0], v.group_weights[1], v.group_weights[2], v.group_lambda_um[0], v.group_lambda_um[1], v.group_lambda_um[2], v.warmth, v.spatial_scale};
        return {{"domain", v.domain}, {"active", v.active != 0}, {"family", v.family}, {"f64", double_words(values, 9)}, {"resolved_hash", v.resolved_hash}, {"hash", v.hash}};
    }
    Json project(const FjSpatialOpticsRecipe& v) {
        const auto& h = v.scatter_halation;
        const float values[]{h.scatter_amount, h.scatter_spatial_scale, h.halation_spatial_scale, h.first_sigma_um[0], h.first_sigma_um[1], h.first_sigma_um[2], h.total_strength[0], h.total_strength[1], h.total_strength[2]};
        return {{"camera", project(v.camera_diffusion)}, {"enlarger", project(v.enlarger_diffusion)}, {"lens_f32", std::bit_cast<std::uint32_t>(v.camera_lens_blur.sigma_um)}, {"lens_hash", v.camera_lens_blur.hash}, {"halation", {{"scatter_active", h.scatter_active != 0}, {"back_reflection_active", h.back_reflection_active != 0}, {"f32", words(values, 9)}, {"hash", h.hash}}}, {"hash", v.hash}};
    }
    void metadata(const FjSpatialOpticsFailure& actual, std::uint32_t reason = FJ_OPTICS_REASON_NONE, std::uint32_t component = FJ_OPTICS_COMPONENT_NONE, std::uint32_t field = FJ_OPTICS_FIELD_NONE, std::uint32_t channel = 3) {
        require(actual.reason == reason && actual.component == component && actual.field == field && actual.channel == channel, "complete optics failure envelope");
    }
    FjSpatialOpticsFailure expected_failure(const std::string& diagnostic) {
        FjSpatialOpticsFailure out{FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, 0, 3};
        if (diagnostic.find("component=enlarger_diffusion") != std::string::npos)
            out.component = FJ_OPTICS_COMPONENT_ENLARGER_DIFFUSION;
        if (diagnostic.find("component=camera_lens_blur") != std::string::npos) {
            out.reason = FJ_OPTICS_REASON_LENS_BLUR;
            out.component = FJ_OPTICS_COMPONENT_CAMERA_LENS_BLUR;
        }
        if (diagnostic.find("component=scatter_halation") != std::string::npos) {
            out.reason = FJ_OPTICS_REASON_HALATION;
            out.component = FJ_OPTICS_COMPONENT_SCATTER_HALATION;
        }
        const std::array<const char*, 16> fields{"family", "strength", "spatial_scale", "halo_warmth", "core_intensity", "core_size", "halo_intensity", "halo_size", "bloom_intensity", "bloom_size", "scatter_fraction", "group_weights", "group_centers_um", "resolved_hash", "sigma_um", "total_strength"};
        for (std::size_t i = 0; i < fields.size(); ++i)
            if (diagnostic.find(std::string("field=") + fields[i]) != std::string::npos)
                out.field = static_cast<std::uint32_t>(i + 1);
        for (unsigned i = 0; i < 3; ++i)
            if (diagnostic.find("channel=" + std::to_string(i)) != std::string::npos)
                out.channel = i;
        require(out.field != 0, "native capture has a known ordinary failure field");
        return out;
    }
    void zero(const FjSpatialOpticsRecipe& out) {
        require(project(out) == project(FjSpatialOpticsRecipe{}), "every seeded optics result field clears to integer/+float zero");
    }
    void seed(FjSpatialOpticsRecipe& out) {
        std::memset(&out, 0x55, sizeof(out));
    }
    void representations(const FjSpatialOpticsRecipe& out) {
        require(out.camera_diffusion.active <= 1 && out.enlarger_diffusion.active <= 1 && out.scatter_halation.scatter_active <= 1 && out.scatter_halation.back_reflection_active <= 1, "exact boolean output representations");
    }
    void values(const Json& captured) {
        require(captured.at("cases").size() == 553, "exact frozen optics literal membership");
        const std::array<std::pair<std::string_view, Prepare>, 2> callers{{{"c11", fj_test_spatial_optics_c_prepare},
                                                                           {"cpp", fj_test_spatial_optics_prepare}}};
        for (const auto& [caller, prepare] : callers)
            for (const auto& row : captured.at("cases")) {
                JuicerTests::CaseEvent case_event(caller, row.at("id").get_ref<const std::string&>());
                auto input = params(row.at("input"));
                FjSpatialOpticsRecipe out{};
                seed(out);
                FjSpatialOpticsFailure failure{99, 98, 97, 96};
                const bool success = row.at("success");
                status(prepare(&input, &out, &failure, nullptr), success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (success) {
                    metadata(failure);
                    representations(out);
                    if (out.camera_lens_blur.hash != 0) {
                        std::array<char, 256> text{};
                        FjErrorBuffer error{text.data(), text.size(), 0};
                        const auto rejected = fj_test_camera_lens_blur_preflight(out.camera_lens_blur.sigma_um, &error);
                        require(rejected.category == FJ_STATUS_PREPARATION_FAILURE && rejected.api == FJ_API_NONE && rejected.native_code == 0 &&
                                    error.length < text.size() && text[error.length] == '\0' &&
                                    std::string_view(text.data(), error.length) == "ExactOpticsNotImplementedForPhase6 component=camera_lens_blur",
                                "positive lens remains rejected by render preflight");
                    }
                    require(project(out) == row.at("recipe"), "all frozen f32/f64 fields, flags/tags and seven identities");
                } else {
                    const auto expected = expected_failure(row.at("diagnostic").get<std::string>());
                    metadata(failure, expected.reason, expected.component, expected.field, expected.channel);
                    zero(out);
                }
                case_event.passed();
            }
    }
    enum class RawVariant : std::uint8_t {
        NullInput,
        NullResult,
        NullFailure,
        MisalignedInput,
        NearNullInput,
        MisalignedResult,
        MisalignedFailure,
        MisalignedError,
        NullErrorBacking,
        OverflowErrorExtent,
        CameraFlag,
        HalationFlag,
        EnlargerFlag,
        PrintFlag,
        CompetingFailure,
        ZeroCapacity,
        DiagnosticCapacityOne,
        DiagnosticCapacityThree,
        DiagnosticCapacityFull
    };
    constexpr std::array<OpticsOperation, 19> raw_operations{{{"optics:raw:null-input", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::NullInput)},
                                                              {"optics:raw:null-result", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::NullResult)},
                                                              {"optics:raw:null-failure", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::NullFailure)},
                                                              {"optics:raw:misaligned-input", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::MisalignedInput)},
                                                              {"optics:raw:overflowing-input-address", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::NearNullInput)},
                                                              {"optics:raw:misaligned-result", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::MisalignedResult)},
                                                              {"optics:raw:misaligned-failure", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::MisalignedFailure)},
                                                              {"optics:raw:misaligned-error", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::MisalignedError)},
                                                              {"optics:raw:null-error-backing", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::NullErrorBacking)},
                                                              {"optics:raw:overflow-error-extent", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::OverflowErrorExtent)},
                                                              {"optics:raw:camera-active-flag", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::CameraFlag)},
                                                              {"optics:raw:halation-active-flag", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::HalationFlag)},
                                                              {"optics:raw:enlarger-active-flag", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::EnlargerFlag)},
                                                              {"optics:raw:print-domain-flag", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::PrintFlag)},
                                                              {"optics:raw:competing-invalid-fields", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::CompetingFailure)},
                                                              {"optics:raw:zero-error-capacity", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::ZeroCapacity)},
                                                              {"optics:raw:diagnostic-capacity-1", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::DiagnosticCapacityOne)},
                                                              {"optics:raw:diagnostic-capacity-3", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::DiagnosticCapacityThree)},
                                                              {"optics:raw:diagnostic-capacity-256", OpticsOperationKind::RawInput, static_cast<std::size_t>(RawVariant::DiagnosticCapacityFull)}}};
    enum class FaultVariant : std::uint8_t {
        Panic,
        PanicOneShot,
        RawNullInputPanic,
        RawNullOutputPanic,
        ResolvedHash,
        ResolvedHashOneShot,
        RawNullInputResolvedHash,
        RawNullOutputResolvedHash,
        InactiveThenActiveWithResolvedHash,
        ClearPendingPanic,
        ThreadLocalResolvedHash
    };
    constexpr std::array<OpticsOperation, 11> fault_operations{{{"optics:fault:panic", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::Panic)},
                                                                {"optics:fault:panic-one-shot", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::PanicOneShot)},
                                                                {"optics:fault:raw-null-input-clears-panic", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::RawNullInputPanic)},
                                                                {"optics:fault:raw-null-output-clears-panic", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::RawNullOutputPanic)},
                                                                {"optics:fault:resolved-hash", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::ResolvedHash)},
                                                                {"optics:fault:resolved-hash-one-shot", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::ResolvedHashOneShot)},
                                                                {"optics:fault:raw-null-input-clears-resolved-hash", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::RawNullInputResolvedHash)},
                                                                {"optics:fault:raw-null-output-clears-resolved-hash", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::RawNullOutputResolvedHash)},
                                                                {"optics:fault:inactive-then-active-with-resolved-hash", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::InactiveThenActiveWithResolvedHash)},
                                                                {"optics:fault:clear-pending-panic", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::ClearPendingPanic)},
                                                                {"optics:fault:thread-local-resolved-hash", OpticsOperationKind::Fault, static_cast<std::size_t>(FaultVariant::ThreadLocalResolvedHash)}}};
    const OpticsOperation layout_operation{"optics:abi-layout", OpticsOperationKind::Layout, 0};
    Prepare caller_prepare(std::string_view caller) {
        return caller == "c11" ? fj_test_spatial_optics_c_prepare : fj_test_spatial_optics_prepare;
    }
    void raw_input_variant(const Json& captured, Prepare prepare, RawVariant variant) {
        const auto& rows = captured.at("cases");
        const auto find = [&](const char* id) -> const Json& {
            for (const auto& row : rows)
                if (row.at("id") == id)
                    return row;
            throw std::runtime_error("missing optics case");
        };
        const auto inactive = params(find("optics-inactive").at("input"));
        FjSpatialOpticsRecipe out{};
        FjSpatialOpticsFailure failure{};
        alignas(FjSpatialOpticsParams) std::array<std::byte, sizeof(FjSpatialOpticsParams) + 8> storage{};
        const auto raw = [&](const FjSpatialOpticsParams* input, FjSpatialOpticsRecipe* result, FjSpatialOpticsFailure* rejected, FjErrorBuffer* error) {
            seed(out);
            failure = {99, 98, 97, 96};
            status(prepare(input, result, rejected, error), FJ_STATUS_UNSUPPORTED_INPUT);
            if (result == &out)
                zero(out);
            if (rejected == &failure)
                metadata(failure, FJ_OPTICS_REASON_RAW);
        };
        switch (variant) {
            case RawVariant::NullInput:
                raw(nullptr, &out, &failure, nullptr);
                break;
            case RawVariant::NullResult:
                raw(&inactive, nullptr, &failure, nullptr);
                break;
            case RawVariant::NullFailure:
                raw(&inactive, &out, nullptr, nullptr);
                break;
            case RawVariant::MisalignedInput:
                raw(reinterpret_cast<const FjSpatialOpticsParams*>(storage.data() + 1), &out, &failure, nullptr);
                break;
            case RawVariant::NearNullInput:
                raw(reinterpret_cast<const FjSpatialOpticsParams*>(UINTPTR_MAX - 7), &out, &failure, nullptr);
                break;
            case RawVariant::MisalignedResult:
                raw(&inactive, reinterpret_cast<FjSpatialOpticsRecipe*>(storage.data() + 1), &failure, nullptr);
                break;
            case RawVariant::MisalignedFailure:
                raw(&inactive, &out, reinterpret_cast<FjSpatialOpticsFailure*>(storage.data() + 1), nullptr);
                break;
            case RawVariant::MisalignedError:
                raw(&inactive, &out, &failure, reinterpret_cast<FjErrorBuffer*>(storage.data() + 1));
                break;
            case RawVariant::NullErrorBacking: {
                FjErrorBuffer error{nullptr, 1, 99};
                raw(&inactive, &out, &failure, &error);
                require(error.length == 0, "admitted diagnostic length clears before invalid backing");
                break;
            }
            case RawVariant::OverflowErrorExtent: {
                FjErrorBuffer error{reinterpret_cast<char*>(1), SIZE_MAX, 99};
                raw(&inactive, &out, &failure, &error);
                require(error.length == 0, "overflowing diagnostic extent rejected before backing");
                break;
            }
            case RawVariant::CameraFlag:
            case RawVariant::HalationFlag:
            case RawVariant::EnlargerFlag:
            case RawVariant::PrintFlag: {
                auto input = inactive;
                std::array<std::uint32_t*, 4> flags{&input.camera_diffusion.active, &input.scatter_halation.active, &input.enlarger_diffusion.active, &input.print_domain};
                *flags[static_cast<std::size_t>(variant) - static_cast<std::size_t>(RawVariant::CameraFlag)] = 2;
                raw(&input, &out, &failure, nullptr);
                break;
            }
            case RawVariant::CompetingFailure: {
                auto input = inactive;
                input.camera_diffusion.strength = std::numeric_limits<double>::quiet_NaN();
                input.enlarger_diffusion.active = 2;
                raw(&input, &out, &failure, nullptr);
                break;
            }
            case RawVariant::ZeroCapacity: {
                FjErrorBuffer error{reinterpret_cast<char*>(1), 0, 99};
                status(prepare(&inactive, &out, &failure, &error));
                metadata(failure);
                require(error.length == 0 && project(out) == find("optics-inactive").at("recipe"), "capacity zero never reads backing and differs from raw clearing");
                break;
            }
            case RawVariant::DiagnosticCapacityOne:
            case RawVariant::DiagnosticCapacityThree:
            case RawVariant::DiagnosticCapacityFull: {
                auto invalid = inactive;
                invalid.camera_diffusion.family = 255;
                const std::array<std::size_t, 3> capacities{1, 3, 256};
                const auto capacity = capacities[static_cast<std::size_t>(variant) - static_cast<std::size_t>(RawVariant::DiagnosticCapacityOne)];
                std::array<char, 256> text{};
                text.fill('x');
                FjErrorBuffer error{text.data(), capacity, 99};
                seed(out);
                status(prepare(&invalid, &out, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
                metadata(failure, FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, FJ_OPTICS_FIELD_FAMILY);
                zero(out);
                require(error.length < capacity && text[error.length] == '\0' && (capacity == 1 ? error.length == 0 : error.length > 0), "bounded terminated/truncated diagnostics preserve the complete outcome");
                break;
            }
        }
    }
    void fault_variant(const Json& captured, Prepare prepare, FaultVariant variant) {
        const auto& rows = captured.at("cases");
        const auto find = [&](const char* id) -> const Json& {
            for (const auto& row : rows)
                if (row.at("id") == id)
                    return row;
            throw std::runtime_error("missing optics case");
        };
        const auto inactive = params(find("optics-inactive").at("input"));
        const auto active = params(find("optics-active-print").at("input"));
        FjSpatialOpticsRecipe out{};
        FjSpatialOpticsFailure failure{};
        const auto raw = [&](const FjSpatialOpticsParams* input, FjSpatialOpticsRecipe* result, FjSpatialOpticsFailure* rejected, FjErrorBuffer* error) {
            seed(out);
            failure = {99, 98, 97, 96};
            status(prepare(input, result, rejected, error), FJ_STATUS_UNSUPPORTED_INPUT);
            if (result == &out)
                zero(out);
            if (rejected == &failure)
                metadata(failure, FJ_OPTICS_REASON_RAW);
        };
        const auto verify_success = [&] {
            status(prepare(&active, &out, &failure, nullptr));
            metadata(failure);
            require(project(out) == find("optics-active-print").at("recipe"), "fault is one shot");
        };
        const auto verify_success_after_raw = [&] {
            status(prepare(&active, &out, &failure, nullptr));
            metadata(failure);
        };
        switch (variant) {
            case FaultVariant::Panic:
            case FaultVariant::ResolvedHash: {
                const unsigned fault = variant == FaultVariant::Panic ? 1u : 2u;
                status(fj_test_film_recipe_fault(9, fault));
                seed(out);
                status(prepare(&active, &out, &failure, nullptr), fault == 1 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_PREPARATION_FAILURE);
                if (fault == 1)
                    metadata(failure, FJ_OPTICS_REASON_PANIC);
                else
                    metadata(failure, FJ_OPTICS_REASON_DIFFUSION, FJ_OPTICS_COMPONENT_CAMERA_DIFFUSION, FJ_OPTICS_FIELD_RESOLVED_HASH);
                zero(out);
                break;
            }
            case FaultVariant::PanicOneShot:
            case FaultVariant::ResolvedHashOneShot:
                verify_success();
                break;
            case FaultVariant::RawNullInputPanic:
            case FaultVariant::RawNullOutputPanic:
            case FaultVariant::RawNullInputResolvedHash:
            case FaultVariant::RawNullOutputResolvedHash: {
                const bool panic = variant == FaultVariant::RawNullInputPanic || variant == FaultVariant::RawNullOutputPanic;
                const bool null_input = variant == FaultVariant::RawNullInputPanic || variant == FaultVariant::RawNullInputResolvedHash;
                status(fj_test_film_recipe_fault(9, panic ? 1u : 2u));
                if (null_input)
                    raw(nullptr, &out, &failure, nullptr);
                else
                    raw(&active, nullptr, &failure, nullptr);
                verify_success_after_raw();
                break;
            }
            case FaultVariant::InactiveThenActiveWithResolvedHash:
                status(fj_test_film_recipe_fault(9, 2));
                status(prepare(&inactive, &out, &failure, nullptr));
                status(prepare(&active, &out, &failure, nullptr));
                metadata(failure);
                break;
            case FaultVariant::ClearPendingPanic:
                status(fj_test_film_recipe_fault(9, 1));
                status(fj_test_film_recipe_fault(9, 0));
                verify_success();
                break;
            case FaultVariant::ThreadLocalResolvedHash: {
                std::atomic<bool> okay{true};
                std::thread isolated([&] {
                    FjSpatialOpticsRecipe value{};
                    FjSpatialOpticsFailure rejected{};
                    status(fj_test_film_recipe_fault(9, 2));
                    const auto result = prepare(&active, &value, &rejected, nullptr);
                    if (result.category != FJ_STATUS_PREPARATION_FAILURE || rejected.field != FJ_OPTICS_FIELD_RESOLVED_HASH)
                        okay = false;
                });
                status(prepare(&active, &out, &failure, nullptr));
                metadata(failure);
                isolated.join();
                require(okay.load(), "raw fault and core guard are thread local");
                break;
            }
        }
    }
    void run_operation(const OpticsOperation& operation, std::string_view caller, const Json& captured) {
        JuicerTests::CaseEvent case_event(caller, operation.id);
        switch (operation.kind) {
            case OpticsOperationKind::Layout:
                layouts();
                break;
            case OpticsOperationKind::RawInput:
                raw_input_variant(captured, caller_prepare(caller), static_cast<RawVariant>(operation.variant));
                break;
            case OpticsOperationKind::Fault:
                fault_variant(captured, caller_prepare(caller), static_cast<FaultVariant>(operation.variant));
                break;
        }
        case_event.passed();
    }
    void list_cases(const Json& captured) {
        const auto& cases = captured.at("cases");
        require(cases.size() == 553, "exact frozen optics literal membership");
        std::set<std::string> ids;
        for (const auto& row : cases) {
            const auto id = row.at("id").get<std::string>();
            require(!id.empty() && ids.insert(id).second, "unique frozen optics fixture IDs");
        }
        for (const auto caller : {std::string_view("c11"), std::string_view("cpp")})
            for (const auto& row : cases)
                JuicerTests::list_case(caller, row.at("id").get_ref<const std::string&>());
        JuicerTests::list_case("cpp", layout_operation.id);
        for (const auto caller : {std::string_view("c11"), std::string_view("cpp")}) {
            for (const auto& operation : raw_operations)
                JuicerTests::list_case(caller, operation.id);
            for (const auto& operation : fault_operations)
                JuicerTests::list_case(caller, operation.id);
        }
    }
    void run_all_nonrow_cases(const Json& captured) {
        for (const auto caller : {std::string_view("c11"), std::string_view("cpp")}) {
            for (const auto& operation : raw_operations)
                run_operation(operation, caller, captured);
            for (const auto& operation : fault_operations)
                run_operation(operation, caller, captured);
        }
    }
    struct ProductFixtures {
        Json products;
        Json literals;
        Json ofx;
    };
    void products(const std::filesystem::path& resources, const ProductFixtures& fixtures) {
        const auto& captured = fixtures.products;
        const auto& literals = fixtures.literals;
        const auto& ofx = fixtures.ofx;
        JuicerTestSupport::RuntimeFixture owner;
        owner.create(resources.string().c_str());
        const auto find = [&](const char* id) -> const Json& {
            for (const auto& row : literals.at("cases"))
                if (row.at("id") == id)
                    return row;
            throw std::runtime_error("missing frozen product input");
        };
        const auto active = params(find("optics-active-print").at("input"));
        const auto inactive = params(find("optics-inactive").at("input"));
        const double cameraStrength = params(find("strength-0-21").at("input")).camera_diffusion.strength;
        const double enlargerStrength = std::bit_cast<double>(ofx.at("ofx")[0].at("input").at("enlarger").at("f64")[0].get<std::uint64_t>());
        const double invalidStrength = params(find("optics-off-invalid-camera").at("input")).camera_diffusion.strength;
        require(captured.at("products").size() == 32, "exact genuine optics product membership");
        for (const auto& row : captured.at("products")) {
            const auto route = row.at("route").get<unsigned>();
            const auto variant = row.at("variant").get<unsigned>();
            JuicerTests::CaseEvent event("cpp", "optics/product/" + std::to_string(route) + "/" + std::to_string(variant));
            auto input = variant == 0 ? inactive : active;
            input.camera_diffusion.family = variant % 4;
            input.camera_diffusion.strength = cameraStrength;
            if (variant == 5)
                input.camera_diffusion.active = 0;
            if (variant == 6)
                input.enlarger_diffusion.strength = enlargerStrength;
            if (variant == 7) {
                input.camera_diffusion.active = 0;
                input.camera_diffusion.strength = invalidStrength;
            }
            JuicerTestSupport::PreparationInput params;
            params.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.print_key = "kodak_portra_endura";
            params.values.scan_route = static_cast<std::uint32_t>(route);
            params.values.camera_auto_exposure_enabled = 0;
            params.values.input_compression_enabled = 0;
            params.values.output_gamut_compression_enabled = 0;
            params.values.camera_diffusion = input.camera_diffusion;
            params.values.enlarger_diffusion = input.enlarger_diffusion;
            params.values.scatter_halation = {input.scatter_halation.active,
                                              input.scatter_halation.scatter_amount,
                                              input.scatter_halation.scatter_spatial_scale,
                                              input.scatter_halation.halation_amount,
                                              input.scatter_halation.halation_spatial_scale};
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> product;
            const auto* assets = owner.borrow().assets;
            const FjStatus result = JuicerTestSupport::prepare_state(assets, params, product, error);
            const bool built = result.category == FJ_STATUS_SUCCESS && product.has_value();
            const std::string observedDiagnostic(diagnostic.data(), error.length);
            FjPreparedStateView view{};
            if (built)
                status(product->view(view, error));
            require(built == row.at("success").get<bool>() && observedDiagnostic == row.at("diagnostic").get<std::string>(), "genuine held-state route product outcome and first failure");
            Json actual{{"snapshot_hash", JuicerTestSupport::hash_input_identity(params)}, {"final_hash", built ? view.recipe_hash : 0}};
            Json expected{{"snapshot_hash", row.at("snapshot_hash")}, {"final_hash", row.at("final_hash")}};
            if (built) {
                actual["recipe"] = project(view.optics);
                expected["recipe"] = row.at("recipe");
            } else {
                require(result.category == FJ_STATUS_PREPARATION_FAILURE && result.api == FJ_API_NONE &&
                            result.native_code == 0 && !product,
                        "failed optics preparation returns no complete owner");
            }
            require_equal(actual, expected, "native authored snapshot, complete optics and final route identities");
            event.passed();
        }
        for (unsigned route = 0; route < 4; ++route) {
            JuicerTests::CaseEvent event("cpp", "optics/publication/" + std::to_string(route));
            JuicerTestSupport::PreparationInput baseline;
            baseline.film_key = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            baseline.print_key = "kodak_portra_endura";
            baseline.values.scan_route = static_cast<std::uint32_t>(route);
            baseline.values.camera_auto_exposure_enabled = 0;
            baseline.values.input_compression_enabled = 0;
            baseline.values.output_gamut_compression_enabled = 0;
            baseline.values.grain.active = false;
            baseline.values.camera_diffusion.active = true;
            JuicerTestSupport::InstanceFixture state;
            JuicerTestSupport::capture(state, baseline);
            const auto admitted = JuicerTestSupport::admit(owner.borrow().assets, state);
            require(admitted.kind == FJ_ADMISSION_READY && admitted.state,
                    "complete optics publication on both polarities/routes");
            const auto held = JuicerTestSupport::published_view(admitted);
            const auto hash = held.recipe_hash;
            const auto baselineMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            auto invalid = baseline;
            invalid.values.camera_diffusion.active = false;
            invalid.values.camera_diffusion.strength = invalidStrength;
            JuicerTestSupport::capture(state, invalid);
            const auto rejected = JuicerTestSupport::admit(owner.borrow().assets, state);
            const auto failedMetrics = JuicerTestSupport::PublicationObserver::metrics(state);
            require(rejected.delivery.category == FJ_STATUS_PREPARATION_FAILURE &&
                        rejected.kind == FJ_ADMISSION_PREPARATION_FAILED && !rejected.state &&
                        rejected.diagnostic == "ResourceDescriptorMismatch phase=6A component=camera_diffusion field=strength" &&
                        held.recipe_hash == hash && failedMetrics.build == baselineMetrics.build &&
                        failedMetrics.publishedIdentity == baselineMetrics.publishedIdentity &&
                        failedMetrics.backingIdentity == baselineMetrics.backingIdentity,
                    "disabled invalid camera blocks admission without partial publication or revoking held state");
            JuicerTestSupport::capture(state, baseline);
            const auto recovered = JuicerTestSupport::admit(owner.borrow().assets, state);
            require(recovered.kind == FJ_ADMISSION_READY && recovered.state &&
                        JuicerTestSupport::PublicationObserver::metrics(state).build == baselineMetrics.build,
                    "optics failure recovery returns to the last complete state");
            const auto repeated = JuicerTestSupport::admit(owner.borrow().assets, state);
            require(repeated.kind == FJ_ADMISSION_READY && repeated.state &&
                        JuicerTestSupport::PublicationObserver::backing(*repeated.state) == baselineMetrics.backingIdentity &&
                        JuicerTestSupport::PublicationObserver::metrics(state).build == baselineMetrics.build,
                    "unchanged recovered publication reuses state");
            auto unused = baseline;
            unused.values.enlarger_diffusion.family = static_cast<std::uint32_t>(255);
            unused.values.enlarger_diffusion.strength = invalidStrength;
            std::string diagnostic;
            FjStatus delivery{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            const auto build = [&](const JuicerTestSupport::PreparationInput& input) {
                JuicerTestSupport::capture(state, input);
                const auto result = JuicerTestSupport::admit(owner.borrow().assets, state);
                delivery = result.delivery;
                diagnostic = result.diagnostic;
                return result.kind == FJ_ADMISSION_READY && bool(result.state);
            };
            const bool success = build(unused);
            if (route % 2)
                require(!success && diagnostic == "ResourceDescriptorMismatch phase=6A component=enlarger_diffusion field=family", "print domain validates enlarger family before finite controls");
            else {
                const auto direct = JuicerTestSupport::admit(owner.borrow().assets, state);
                const auto directView = JuicerTestSupport::published_view(direct);
                require(success && directView.optics.hash == held.optics.hash && JuicerTestSupport::hash_input_identity(unused) != JuicerTestSupport::hash_input_identity(baseline), "direct ignores enlarger policy while native authored snapshot still changes");
            }
            auto badInput = baseline;
            badInput.values.input_color_space = -1;
            status(fj_test_prepared_optics_resolved_hash_fault());
            const std::string bareInputDiagnostic = "UnsupportedMode phase=3B field=input_color_space";
            require(!build(badInput) &&
                        delivery.category == FJ_STATUS_UNSUPPORTED_INPUT && delivery.api == FJ_API_NONE &&
                        delivery.native_code == 0 && diagnostic == bareInputDiagnostic,
                    "actual admission preserves the bare unsupported-input diagnostic and leaves optics fault pending");
            std::array<char, 2048> rawDiagnosticText{};
            FjErrorBuffer rawError{rawDiagnosticText.data(), rawDiagnosticText.size(), 0};
            std::optional<JuicerTestSupport::PreparedStateHold> rawProduct;
            const auto rawStatus = JuicerTestSupport::prepare_state(owner.borrow().assets, badInput, rawProduct, rawError);
            const std::string observedRawDiagnostic(rawDiagnosticText.data(), rawError.length);
            require(rawStatus.category == FJ_STATUS_UNSUPPORTED_INPUT && rawStatus.api == FJ_API_NONE &&
                        rawStatus.native_code == 0 && !rawProduct && observedRawDiagnostic == bareInputDiagnostic,
                    "direct constructor retains the bare unsupported-input status and diagnostic without an owner");
            auto badGrain = baseline;
            badGrain.values.grain.active = true;
            badGrain.values.grain.particle_area_um2 = 0;
            require(!build(badGrain) && diagnostic == "ResourceDescriptorMismatch phase=6A component=camera_diffusion field=resolved_hash", "actual optics guard precedes later grain construction");
            require(!build(badGrain) && diagnostic == "MalformedRequiredProfileData phase=9A field=visual_grain", "consumed optics fault permits the later grain failure");
            require(held.recipe_hash == hash && fj_test_film_development_live_owners() == 0 && fj_test_film_dir_live_owners() == 0, "failures preserve publications and drain temporary construction holds");
            event.passed();
        }
        std::puts("PASS 32 genuine optics products and four-route publication, ordering, recovery and identity contracts");
    }
} // namespace OpticsTest
