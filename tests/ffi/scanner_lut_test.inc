namespace ScannerLutTest {
    constexpr std::array kArrayNames{"log2_xyz", "slope_c", "slope_m", "slope_y", "cell_min", "cell_max"};
    struct Literal {
        std::array<std::vector<float>, 7> spectra;
        Scanner::MediumView medium;
        std::uint32_t resolution;

        explicit Literal(const Json& input) : resolution(input.at("resolution")) {
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
        for (bool facade : {false, true})
            for (const auto& row : rows) {
                const Literal literal(row.at("input"));
                const auto input = wire(literal);
                FjScannerLutOwner* owner = reinterpret_cast<FjScannerLutOwner*>(std::uintptr_t{0x5555});
                FjScannerLutFailure f{99, 99, 99, 99};
                std::array<char, 256> text{};
                FjErrorBuffer error{text.data(), text.size(), 99};
                const auto acquire = facade ? fj_test_scanner_lut_acquire : fj_legacy_scanner_lut_acquire;
                const auto view = facade ? fj_test_scanner_lut_view : fj_legacy_scanner_lut_view;
                const auto release = facade ? fj_test_scanner_lut_release : fj_legacy_scanner_lut_release;
                const auto result = acquire(&input, &owner, &f, &error);
                const auto& expected = row.at("native");
                if (!expected.at("success").get<bool>()) {
                    status(result, FJ_STATUS_PREPARATION_FAILURE);
                    require(!owner, "failed LUT publishes no owner");
                    const bool resolution = input.resolution < 2;
                    metadata(f, resolution ? 2u : 5u, 0, expected.at("failed_voxel"), expected.at("failed_channel"));
                    require(std::string(text.data(), error.length) == expected.at("diagnostic").get<std::string>(), "native LUT diagnostic continuity");
                    continue;
                }
                status(result);
                metadata(f);
                require(owner && error.length == 0 && text[0] == 0, "successful complete LUT owner");
                FjScannerLutView borrowed{};
                status(view(owner, &borrowed, &f, &error));
                metadata(f);
                compare(project(borrowed), expected);
                FjScannerLutView repeated{};
                status(view(owner, &repeated, &f, nullptr));
                metadata(f);
                require(borrowed.log2_xyz.data == repeated.log2_xyz.data, "immutable repeated view does no copy");
                compare(project(repeated), expected);
                status(release(owner, nullptr));
                owner = nullptr;
            }
        require(fj_test_scanner_lut_live_owners() == 0, "value cohort drops every host owner");
    }
    void boundary(const Json& captured) {
        const auto& row = captured.at("cases")[0];
        Literal literal(row.at("input"));
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
            status(fj_legacy_scanner_lut_acquire(&input, &owner, &f, &error), expected.category);
            require(!owner, "rejected LUT clears owner");
            metadata(f, expected.reason, expected.buffer);
            require(error.length < error.capacity && text[error.length] == 0, "LUT diagnostics bounded and terminated");
        };
        auto input = original;
        input.resolution = 1;
        reject(input, {FJ_STATUS_PREPARATION_FAILURE, FJ_SCANNER_LUT_REASON_RESOLUTION});
        input = original;
        input.resolution = UINT32_MAX;
        reject(input, {FJ_STATUS_ALLOCATION_FAILURE, FJ_SCANNER_LUT_REASON_SIZE, 1});
        input = original;
        input.medium = 256;
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        input = original;
        input.has_baseline = 2;
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        input = original;
        input.eps_c = {nullptr, 3};
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        input = original;
        input.weighted_z.count = 2;
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        input = original;
        input.eps_m = {reinterpret_cast<const float*>(std::uintptr_t{1}), 3};
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        input = original;
        input.eps_y.count = SIZE_MAX;
        reject(input, {FJ_STATUS_UNSUPPORTED_INPUT, FJ_SCANNER_LUT_REASON_RAW});
        for (unsigned buffer = 1; buffer <= 10; ++buffer) {
            status(fj_test_scanner_lut_fault(0, buffer + 1));
            reject(original, {FJ_STATUS_ALLOCATION_FAILURE, FJ_SCANNER_LUT_REASON_ALLOCATION, buffer});
            require(fj_test_scanner_lut_live_owners() == 0, "every failed reservation drops provisional buffers");
        }
        for (unsigned operation = 0; operation < 3; ++operation) {
            FjScannerLutOwner* owner = nullptr;
            FjScannerLutFailure f{};
            FjScannerLutView view{};
            if (operation)
                status(fj_legacy_scanner_lut_acquire(&original, &owner, &f, nullptr));
            status(fj_test_scanner_lut_fault(operation, 1));
            status(fj_test_scanner_lut_fault(operation, 12), 2); // Invalid pair preserves the armed panic.
            if (operation == 0) {
                status(fj_legacy_scanner_lut_acquire(&original, &owner, &f, nullptr), 8);
                metadata(f, 6);
                require(!owner, "acquire panic no publication");
            } else if (operation == 1) {
                status(fj_legacy_scanner_lut_view(owner, &view, &f, nullptr), 8);
                metadata(f, 6);
                require(view.resolution == 0 && view.cell_max.data == nullptr, "view panic clears all storage");
                status(fj_legacy_scanner_lut_release(owner, nullptr));
            } else
                status(fj_legacy_scanner_lut_release(owner, nullptr), 8);
            require(fj_test_scanner_lut_live_owners() == 0, "contained panic consumes/drops its owned storage");
        }
        FjScannerLutOwner* owner = nullptr;
        FjScannerLutFailure f{99, 99, 99, 99};
        FjScannerLutView view{};
        status(fj_legacy_scanner_lut_acquire(nullptr, &owner, &f, nullptr), 2);
        metadata(f, 1);
        owner = reinterpret_cast<FjScannerLutOwner*>(std::uintptr_t{0x5555});
        status(fj_legacy_scanner_lut_acquire(&original, &owner, reinterpret_cast<FjScannerLutFailure*>(std::uintptr_t{1}), nullptr), 2);
        require(!owner, "malformed peer cannot prevent owner clearing");
        f = {99, 99, 99, 99};
        status(fj_legacy_scanner_lut_acquire(&original, nullptr, &f, nullptr), 2);
        metadata(f, 1);
        FjErrorBuffer malformed{nullptr, 1, 99};
        status(fj_legacy_scanner_lut_acquire(&original, &owner, &f, &malformed), 2);
        metadata(f, 1);
        require(!owner && malformed.length == 0, "diagnostic admission clears independent outputs");
        std::memset(&view, 0x55, sizeof(view));
        status(fj_legacy_scanner_lut_view(nullptr, &view, &f, nullptr), 2);
        metadata(f, 1);
        require(view.resolution == 0 && view.log2_xyz.data == nullptr && view.slope_c.count == 0 && view.slope_m.data == nullptr && view.slope_y.count == 0 && view.cell_min.data == nullptr && view.cell_max.count == 0, "all six view fields independently cleared");
        std::array<char, 4> text{};
        FjErrorBuffer truncated{text.data(), text.size(), 99};
        status(fj_legacy_scanner_lut_acquire(nullptr, &owner, &f, &truncated), 2);
        require(truncated.length == 3 && text[3] == 0, "truncated diagnostic NUL");
        FjErrorBuffer zero{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
        status(fj_legacy_scanner_lut_acquire(nullptr, &owner, &f, &zero), 2);
        require(zero.length == 0, "zero-capacity diagnostic ignores backing");
        // Ignored baseline and nonpositive-K backing are never admitted or read.
        input = original;
        input.has_baseline = 0;
        input.baseline_min = {reinterpret_cast<const float*>(std::uintptr_t{1}), SIZE_MAX};
        status(fj_legacy_scanner_lut_acquire(&input, &owner, &f, nullptr));
        status(fj_legacy_scanner_lut_release(owner, nullptr));
        input = original;
        input.sample_count = 0;
        input.eps_c = input.eps_m = input.eps_y = input.weighted_x = input.weighted_y = input.weighted_z = input.baseline_min = {reinterpret_cast<const float*>(std::uintptr_t{1}), SIZE_MAX};
        status(fj_test_scanner_lut_fault(0, 11));
        status(fj_legacy_scanner_lut_acquire(&input, &owner, &f, nullptr), 3);
        metadata(f, 5, 0, 0, 0);
        status(fj_legacy_scanner_lut_acquire(&original, &owner, &f, nullptr)); // Unreached reservation fault was consumed at admission.
        status(fj_legacy_scanner_lut_release(owner, &malformed), 2);
        require(fj_test_scanner_lut_live_owners() == 0, "invalid diagnostics consume legitimate release");
        status(fj_legacy_scanner_lut_release(nullptr, nullptr));
        {
            Literal expiring(row.at("input"));
            const auto wireInput = wire(expiring);
            status(fj_legacy_scanner_lut_acquire(&wireInput, &owner, &f, nullptr));
            for (auto& spectrum : expiring.spectra) {
                std::fill(spectrum.begin(), spectrum.end(), -999.0f);
                spectrum.clear();
                spectrum.shrink_to_fit();
            }
        }
        status(fj_legacy_scanner_lut_view(owner, &view, &f, nullptr));
        compare(project(view), row.at("native"));
        status(fj_legacy_scanner_lut_release(owner, nullptr));
    }
    void native_values(const Json& rows) {
        static_assert(!std::is_copy_constructible_v<JuicerScanner::Lut>);
        static_assert(std::is_nothrow_move_constructible_v<JuicerScanner::Lut>);
        for (const auto& row : rows) {
            const Literal literal(row.at("input"));
            JuicerCuda::Failure failure;
            auto owner = JuicerScanner::Lut::acquire(literal.medium, literal.resolution, failure);
            require(owner.has_value() == row.at("native").at("success").get<bool>(), "native adapter success/failure");
            if (!owner)
                continue;
            auto moved = std::move(*owner);
            owner.reset();
            const auto span = [](std::span<const float> v) -> FjFloatSpan {
                return {v.data(), v.size()};
            };
            const FjScannerLutView view{literal.resolution, span(moved.log2_xyz()), span(moved.slope_c()), span(moved.slope_m()), span(moved.slope_y()), span(moved.cell_min()), span(moved.cell_max())};
            compare(project(view), row.at("native"));
        }
        require(fj_test_scanner_lut_live_owners() == 0, "native moves and error cleanup consume once");
    }
} // namespace ScannerLutTest

namespace ScannerLutTest {
    void check_products(const std::filesystem::path& resources, const Json& captured) {
        JuicerCuda::Owner owner;
        owner.create(resources.string().c_str());
        JuicerProcess::root().ensure_bootstrap();
        require(captured.at("products").size() == 4, "exact actual scanner product membership");
        for (unsigned route = 0; route < 4; ++route) {
            auto params = PrintPreflashTest::controls();
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.scannerLutResolution = 17;
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            const bool print = route % 2 != 0;
            require(print ? build_print_render_state_product(params, product, diagnostic) : build_direct_render_state_product(params, product, diagnostic), diagnostic.c_str());
            Scanner::ScannerSpectralLutDescriptor descriptor;
            const auto& recipe = product.recipe;
            require(print ? Scanner::build_print_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, descriptor, diagnostic)
                          : Scanner::build_direct_scanner_spectral_lut_descriptor({&recipe.profileRoute, &recipe.densityBounds, &recipe.scannerOutput}, descriptor, diagnostic),
                    diagnostic.c_str());
            Scanner::MediumView medium;
            medium.medium = print ? Scanner::ScannerMedium::Print : Scanner::ScannerMedium::Negative;
            medium.tables = Scanner::spectral_tables_view(product.payload.scannerTables);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                medium.range.min_cmy[channel] = print ? recipe.densityBounds.dataMinCmy[channel] : -recipe.densityBounds.dataMinCmy[channel];
                medium.range.inv_max_cmy[channel] = recipe.densityBounds.invSpanCmy[channel];
            }
            const auto& expected = captured.at("products")[route];
            require(expected.at("route").get<unsigned>() == route && expected.at("descriptor_hash").get<std::uint64_t>() == descriptor.hash, "actual scanner product descriptor identity");
            require(input(medium, descriptor.lutResolution) == expected.at("input"), "every actual selected scanner table and bound word");
            JuicerCuda::Failure failure;
            auto owner = JuicerScanner::Lut::acquire(medium, descriptor.lutResolution, failure);
            require(owner.has_value(), failure.diagnostic.c_str());
            const auto span = [](std::span<const float> values) -> FjFloatSpan {
                return {values.data(), values.size()};
            };
            const FjScannerLutView view{descriptor.lutResolution, span(owner->log2_xyz()), span(owner->slope_c()), span(owner->slope_m()), span(owner->slope_y()), span(owner->cell_min()), span(owner->cell_max())};
            compare(project(view), expected.at("native"));
        }
    }
    void native_failures(const Json& captured) {
        const Literal literal(captured.at("cases")[0].at("input"));
        for (unsigned buffer = 1; buffer <= 10; ++buffer) {
            status(fj_test_scanner_lut_fault(0, buffer + 1));
            JuicerCuda::Failure failure;
            auto owner = JuicerScanner::Lut::acquire(literal.medium, literal.resolution, failure);
            require(!owner && failure.status.category == FJ_STATUS_ALLOCATION_FAILURE && failure.status.api == FJ_API_NONE && failure.status.native_code == 0 && failure.diagnostic == "scanner LUT host allocation failed", "native adapter preserves typed host allocation failure");
        }
        for (unsigned operation : {0u, 1u}) {
            status(fj_test_scanner_lut_fault(operation, 1));
            JuicerCuda::Failure failure;
            auto owner = JuicerScanner::Lut::acquire(literal.medium, literal.resolution, failure);
            require(!owner && failure.status.category == FJ_STATUS_INTERNAL_FAILURE && failure.diagnostic == "scanner LUT preparation panicked", "native adapter preserves acquire/view panic and drops provisional owner");
        }
        const auto in = wire(literal);
        FjScannerLutOwner* first = nullptr;
        FjScannerLutOwner* second = nullptr;
        FjScannerLutFailure f{};
        status(fj_legacy_scanner_lut_acquire(&in, &first, &f, nullptr));
        status(fj_legacy_scanner_lut_acquire(&in, &second, &f, nullptr));
        require(first != second && fj_test_scanner_lut_live_owners() == 2, "independent LUT owners");
        status(fj_legacy_scanner_lut_release(std::exchange(first, nullptr), nullptr));
        FjScannerLutView view{};
        status(fj_legacy_scanner_lut_view(second, &view, &f, nullptr));
        compare(project(view), captured.at("cases")[0].at("native"));
        status(fj_legacy_scanner_lut_release(std::exchange(second, nullptr), nullptr));
        require(fj_test_scanner_lut_live_owners() == 0, "native errors and independent owners release exactly once");
    }
} // namespace ScannerLutTest
