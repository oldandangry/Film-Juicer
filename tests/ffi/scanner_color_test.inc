// Independent whole-builder values and preserved CAT02/catalogue expectations.
namespace ScannerColorTests {
    using Prepare = FjStatus (*)(const FjScannerColorInput*, FjScannerColor*, FjScannerColorFailure*, FjErrorBuffer*);
    static_assert(std::is_same_v<decltype(&fj_legacy_scanner_color_prepare), Prepare>);
    static_assert(std::is_same_v<decltype(&fj_test_scanner_color_prepare), Prepare>);
    static_assert(sizeof(FjScannerColorInput) == 40 && alignof(FjScannerColorInput) == 8);
    static_assert(offsetof(FjScannerColorInput, medium) == 0 && offsetof(FjScannerColorInput, output_space) == 4 && offsetof(FjScannerColorInput, apply_cctf) == 8);
    static_assert(offsetof(FjScannerColorInput, illuminant_xyz) == 12 && offsetof(FjScannerColorInput, illuminant_hash) == 24 && offsetof(FjScannerColorInput, active_gamut_recipe_hash) == 32);
    static_assert(sizeof(FjScannerColor) == 176 && alignof(FjScannerColor) == 8);
    static_assert(offsetof(FjScannerColor, cat02) == 0 && offsetof(FjScannerColor, xyz_to_rgb) == 36 && offsetof(FjScannerColor, illuminant_xyz) == 72);
    static_assert(offsetof(FjScannerColor, output_space) == 84 && offsetof(FjScannerColor, flags) == 88 && offsetof(FjScannerColor, encoding) == 92);
    static_assert(offsetof(FjScannerColor, active_gamut_recipe_hash) == 160 && offsetof(FjScannerColor, hash) == 168);
    static_assert(sizeof(FjScannerColorFailure) == 8 && alignof(FjScannerColorFailure) == 4);
    static_assert(offsetof(FjScannerColorFailure, reason) == 0 && offsetof(FjScannerColorFailure, stage) == 4);
    FjScannerColorInput input(const Json& i) {
        FjScannerColorInput out{i.at("medium"), i.at("output_space"), i.at("apply_cctf").get<bool>() ? 1u : 0u, {}, i.at("illuminant_hash"), i.at("active_gamut_recipe_hash")};
        for (int c = 0; c < 3; ++c)
            out.illuminant_xyz[c] = std::bit_cast<float>(i.at("illuminant_xyz")[c].get<std::uint32_t>());
        return out;
    }
    Json result(const FjScannerColor& v) {
        const auto& e = v.encoding;
        const std::array coefficients{e.gamma, e.a, e.b, e.c, e.d, e.linear_cutoff};
        return {{"cat02", words(v.cat02, 9)}, {"xyz_to_rgb", words(v.xyz_to_rgb, 9)}, {"illuminant_xyz", words(v.illuminant_xyz, 3)}, {"params", {v.output_space, bool(v.flags & 1u), bool(v.flags & 2u)}}, {"encoding", {{"schema", e.schema}, {"kind", e.cctf_kind}, {"coefficients", words(coefficients.data(), 6)}, {"dwg_to_output", words(e.dwg_to_output, 9)}}}, {"active_gamut_recipe_hash", v.active_gamut_recipe_hash}, {"hash", v.hash}};
    }
    FjScannerColor prepare(Prepare operation, const FjScannerColorInput& in) {
        FjScannerColor out{};
        FjScannerColorFailure failure{99, 99};
        std::array<char, 128> text{};
        FjErrorBuffer error{text.data(), text.size(), 99};
        const auto status = operation(&in, &out, &failure, &error);
        require(status.category == 0 && status.api == 0 && status.native_code == 0 && failure.reason == 0 && failure.stage == 0 && error.length == 0 && text[0] == '\0', "complete scanner success envelope");
        return out;
    }
    void encoding(const FjOutputEncoding& e, const Json& entry) {
        const std::array c{e.gamma, e.a, e.b, e.c, e.d, e.linear_cutoff};
        require(e.schema == 1 && e.cctf_kind == entry.at("kind") && words(c.data(), 6) == entry.at("cctf") && words(e.dwg_to_output, 9) == entry.at("dwg_to_output"), "every prepared encoding field from frozen catalogue, including CCTF off");
    }
    thread_local int corruption = 0;
    void corrupt(FjStatus& status, FjScannerColor& value, FjScannerColorFailure& failure) {
        switch (corruption) {
            case 1:
                value.output_space ^= 1u;
                break;
            case 2:
                value.flags ^= 1u;
                break;
            case 3:
                value.active_gamut_recipe_hash ^= 1u;
                break;
            case 4:
                value.encoding.schema = 0;
                break;
            case 5:
                value.encoding.cctf_kind = 6;
                break;
            case 6:
                status.category = 99;
                break;
            case 7:
                status.api = 1;
                break;
            case 8:
                failure.reason = 1;
                break;
            case 9:
                failure.stage = 1;
                break;
            case 10:
                status.category = 3;
                failure.reason = 2;
                break;
            case 11:
                value.hash = 0;
                break;
        }
    }
    void integrity() {
        Scanner::ScannerIlluminant illuminant{};
        illuminant.whiteXYZ[0] = 0.950455f;
        illuminant.whiteXYZ[1] = 1.0f;
        illuminant.whiteXYZ[2] = 1.089058f;
        illuminant.hash = 1234567;
        for (corruption = 1; corruption <= 10; ++corruption) {
            JuicerScanner::Test::set_color_projection(corrupt);
            bool rejected = false;
            try {
                (void)Scanner::build_color_runtime(Scanner::ScannerMedium::Negative, illuminant, {}, 0);
            } catch (const JuicerCuda::ExecutionFailure& error) {
                rejected = error.failure.status.category == FJ_STATUS_PREPARATION_FAILURE && error.failure.status.api == FJ_API_NONE && error.failure.status.native_code == 0 && error.failure.diagnostic == "scanner color projection invalid";
            }
            JuicerScanner::Test::set_color_projection(nullptr);
            require(rejected, "new foreign projection integrity fails as typed preparation failure");
        }
        corruption = 11;
        JuicerScanner::Test::set_color_projection(corrupt);
        const auto zero = Scanner::build_color_runtime(Scanner::ScannerMedium::Negative, illuminant, {}, 0);
        JuicerScanner::Test::set_color_projection(nullptr);
        corruption = 0;
        require(zero.hash == 0 && zero.encoding.inputIsOutputSpace, "final zero identity remains caller publication responsibility");
    }
    void cold_first_access(const Json& supplement) {
        std::barrier barrier(16);
        std::array<FjScannerColor, 16> values{};
        std::array<FjStatus, 16> statuses{};
        std::vector<std::thread> threads;
        threads.reserve(values.size());
        const auto in = input(supplement.at("cases")[0].at("input"));
        for (std::size_t i = 0; i < values.size(); ++i)
            threads.emplace_back([&, i] {
                FjScannerColorFailure failure{};
                FjErrorBuffer error{nullptr, 0, 0};
                barrier.arrive_and_wait();
                statuses[i] = (i % 2 ? fj_test_scanner_color_prepare : fj_legacy_scanner_color_prepare)(&in, &values[i], &failure, &error);
            });
        for (auto& thread : threads)
            thread.join();
        for (std::size_t i = 0; i < values.size(); ++i)
            require(statuses[i].category == 0 && result(values[i]) == supplement.at("cases")[0].at("expected"), "scanner cold concurrent whole-constructor first access");
    }
    struct Expectations {
        const Json& supplement;
        const Json& catalogue;
        const Json& cat02;
    };
    void run(const Expectations& expected) {
        const auto& supplement = expected.supplement;
        const auto& catalogue = expected.catalogue;
        const auto& cat02 = expected.cat02;
        integrity();
        std::vector<std::pair<FjScannerColor, Json>> held;
        for (auto operation : {fj_legacy_scanner_color_prepare, fj_test_scanner_color_prepare}) {
            for (const auto& row : supplement.at("cases")) {
                auto in = input(row.at("input"));
                if (in.illuminant_hash == 0)
                    continue;
                auto v = prepare(operation, in);
                require(result(v) == row.at("expected"), "all supplement words and identities");
                held.emplace_back(v, row.at("expected"));
            }
            for (const auto& row : catalogue.at("scanner")) {
                if (row.at("illuminant_hash") == 0)
                    continue;
                FjScannerColorInput in{row.at("medium"), row.at("tag"), row.at("encoded"), {0.950455f, 1.0f, 1.089058f}, row.at("illuminant_hash"), 7654321};
                auto v = prepare(operation, in);
                require(words(v.cat02, 9) == row.at("cat02") && words(v.xyz_to_rgb, 9) == row.at("xyz_to_rgb") && words(v.illuminant_xyz, 3) == row.at("illuminant_xyz") && v.hash == row.at("hash") && v.active_gamut_recipe_hash == row.at("gamut_recipe_hash"), "existing whole-builder raw/facade values");
                require(v.output_space == row.at("tag") && v.flags == (2u | in.apply_cctf), "original byte and forced input flags");
                encoding(v.encoding, catalogue.at("entries")[in.output_space < 9 ? in.output_space : 0]);
            }
            // Reuse accepted source-white threshold/nonfinite rows; do not reserialize fixtures.
            std::size_t whites = 0;
            for (const auto* family : {"leaf", "threshold"}) {
                for (const auto& row : cat02.at(family)) {
                    const auto id = row.at("id").get<std::string>();
                    const bool selected = id == "threshold-row-1-offset--1" || id == "threshold-row-1-offset-0" || id == "threshold-row-1-offset-1" || id == "source-special-0-3-xyz-0" || id == "source-special-0-4-xyz-0" || id == "source-special-0-5-xyz-0";
                    if (!selected)
                        continue;
                    require(row.at("destination") == Json::array({catalogue.at("entries")[5].at("base")[8], catalogue.at("entries")[5].at("base")[9], catalogue.at("entries")[5].at("base")[10]}), "frozen CAT02 destination is selected catalogue white");
                    FjScannerColorInput in{0, 5, 1, {}, 1234567, 7654321};
                    for (int c = 0; c < 3; ++c)
                        in.illuminant_xyz[c] = std::bit_cast<float>(row.at("source")[c].get<std::uint32_t>());
                    const auto v = prepare(operation, in);
                    require(words(v.cat02, 9) == row.at("expected").at("matrix") && words(v.illuminant_xyz, 3) == row.at("source"), "frozen threshold/nonfinite CAT02 and original source words");
                    encoding(v.encoding, catalogue.at("entries")[5]);
                    require(words(v.xyz_to_rgb, 9) == catalogue.at("entries")[5].at("xyz_to_rgb") && v.output_space == 5 && v.flags == 3 && v.active_gamut_recipe_hash == 7654321, "complete threshold/nonfinite projection fields");
                    const auto native = std::find_if(catalogue.at("scanner").begin(), catalogue.at("scanner").end(), [](const Json& r) {
                        return r.at("tag") == 5 && r.at("medium") == 0 && r.at("encoded") == 1 && r.at("illuminant_hash") == 1234567;
                    });
                    require(native != catalogue.at("scanner").end() && v.hash == native->at("hash"), "source white words do not independently enter runtime hash");
                    ++whites;
                }
            }
            require(whites == 6, "nonempty exact accepted white case membership");
        }
        for (const auto& [v, expected] : held)
            require(result(v) == expected, "caller-owned values survive subsequent preparations and input expiry");
    }
} // namespace ScannerColorTests
