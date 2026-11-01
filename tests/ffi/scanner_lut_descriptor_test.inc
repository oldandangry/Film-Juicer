namespace ScannerDescriptorTest {
    Json bytes(const std::string& text) {
        Json out = Json::array();
        for (unsigned char byte : text)
            out.push_back(byte);
        return out;
    }
    std::string text(const Json& value) {
        std::string out;
        for (const auto& byte : value)
            out.push_back(static_cast<char>(byte.get<unsigned>()));
        return out;
    }
    Json prepared_input(const FjPreparedStateView& prepared, std::string_view observer) {
        auto* assets = JuicerProcess::root().assets().borrow_rust_assets();
        require(assets != nullptr, "prepared scanner descriptor has process Rust assets");
        FjFilmProfile* filmOwner = nullptr;
        FjPrintProfile* printOwner = nullptr;
        const bool printing = prepared.route % 2 != 0;
        status(fj_legacy_film_profile_acquire(assets, prepared.film_profile_key, &filmOwner, nullptr));
        if (printing)
            status(fj_legacy_print_profile_acquire(assets, prepared.print_profile_key, &printOwner, nullptr));
        FjFilmProfileView film{};
        FjPrintProfileView print{};
        const auto filmStatus = fj_legacy_film_profile_view(filmOwner, &film, nullptr);
        status(filmStatus);
        if (printing)
            status(fj_legacy_print_profile_view(printOwner, &print, nullptr));
        require(film.asset_token == prepared.film_profile_asset_version &&
                    (printing ? print.asset_token == prepared.print_profile_asset_version : prepared.print_profile_asset_version == 0),
                "scanner descriptor source profiles match selected held-state asset versions");
        const auto& tables = printing ? print.tables : film.tables;
        const auto& viewing = printing ? print.viewing_illuminant.label : film.viewing_illuminant.label;
        Json value{{"mode", printing ? 1 : 0},
                   {"route", prepared.route},
                   {"capture_polarity", prepared.polarity},
                   {"bounds_medium", prepared.scanner_bounds.medium},
                   {"bounds_source", prepared.scanner_bounds.source},
                   {"output_medium", prepared.scanner_bounds.medium},
                   {"profile_present", true},
                   {"missing", 0},
                   {"density_bounds_hash", prepared.scanner_bounds.hash},
                   {"print_profile_token", prepared.print_profile_asset_version},
                   {"print_profile_key", prepared.print_profile_key.count == 0 ? Json::array() : bytes(std::string(prepared.print_profile_key.data, prepared.print_profile_key.count))},
                   {"channel_density_cmy", words(tables.channel_density_cmy.data, tables.channel_density_cmy.count)},
                   {"base_density", words(tables.base_density.data, tables.base_density.count)},
                   {"viewing_illuminant", bytes(std::string(viewing.data, viewing.count))},
                   {"profile_viewing_illuminant", bytes(std::string(viewing.data, viewing.count))},
                   {"observer_identity", bytes(std::string(observer))},
                   {"resolution", prepared.scanner_lut.resolution}};
        status(fj_legacy_film_profile_release(filmOwner, nullptr));
        if (printOwner)
            status(fj_legacy_print_profile_release(printOwner, nullptr));
        return value;
    }
    struct Borrowed {
        std::array<float, 243> channel{};
        std::array<float, 81> base{};
        std::array<std::string, 4> labels;
        FjScannerLutDescriptorInput wire{};
        explicit Borrowed(const Json& in) {
            for (std::size_t i = 0; i < channel.size(); ++i)
                channel[i] = std::bit_cast<float>(in.at("channel_density_cmy")[i].get<std::uint32_t>());
            for (std::size_t i = 0; i < base.size(); ++i)
                base[i] = std::bit_cast<float>(in.at("base_density")[i].get<std::uint32_t>());
            constexpr std::array names{"viewing_illuminant", "profile_viewing_illuminant", "observer_identity", "print_profile_key"};
            for (std::size_t i = 0; i < labels.size(); ++i)
                labels[i] = text(in.at(names[i]));
            const auto bytes = [&](std::size_t i) -> FjByteSpan {
                return {reinterpret_cast<const std::uint8_t*>(labels[i].data()), labels[i].size()};
            };
            wire = {in.at("mode"), in.at("route"), in.at("capture_polarity"), in.at("bounds_medium"), in.at("bounds_source"), in.at("output_medium"), in.at("profile_present").get<bool>() ? 1u : 0u, in.at("resolution"), in.at("density_bounds_hash"), in.at("print_profile_token"), {channel.data(), channel.size()}, {base.data(), base.size()}, bytes(0), bytes(1), bytes(2), bytes(3)};
        }
        Borrowed(const Borrowed&) = delete;
    };
    Json project(const FjScannerLutDescriptor& d) {
        return {{"route", d.route}, {"medium", d.medium}, {"capture_polarity", d.capture_polarity}, {"resolution", d.resolution}, {"density_bounds_hash", d.density_bounds_hash}, {"channel_density_hash", d.channel_density_hash}, {"base_density_hash", d.base_density_hash}, {"viewing_illuminant_hash", d.viewing_illuminant_hash}, {"observer_hash", d.observer_hash}, {"hash", d.hash}};
    }
    FjScannerLutDescriptor describe(const Json& in) {
        const Borrowed borrowed(in);
        FjScannerLutDescriptor result{};
        FjScannerLutDescriptorFailure failure{};
        std::array<char, 256> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        status(fj_test_scanner_lut_describe(&borrowed.wire, &result, &failure, &error));
        require(failure.reason == 0 && failure.lane == 0 && error.length == 0 && message[0] == '\0',
                "raw descriptor product has a complete success envelope");
        return result;
    }
    Json products(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        Json rows = Json::array();
        for (unsigned r = 0; r < 4; ++r) {
            auto params = FilmReferenceTest::controls(1, static_cast<Spektrafilm::ScanRoute>(r), 0);
            std::array<char, 2048> diagnostic{};
            FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 0};
            std::optional<PreparedState> product;
            const auto result = prepare_state(JuicerProcess::root().assets().borrow_rust_assets(), params, product, error);
            require(result.category == FJ_STATUS_SUCCESS && product, diagnostic.data());
            FjPreparedStateView view{};
            status(product->view(view, error));
            const auto value = prepared_input(view, "CIE1931_2deg");
            const Json described{{"success", true}, {"exception", false}, {"diagnostic", ""}, {"descriptor", project(describe(value))}};
            rows.push_back({{"route", r}, {"input", value}, {"native", described}, {"snapshot_hash", hash_params(params)}, {"final_hash", view.recipe_hash}, {"scanner_hash", view.scanner_recipe_hash}, {"upload_hash", view.upload_core_hash}});
        }
        return rows;
    }

    bool cleared(const FjScannerLutDescriptor& result) {
        const FjScannerLutDescriptor zero{};
        return std::memcmp(&result, &zero, sizeof result) == 0;
    }
    unsigned reason(const std::string& diagnostic, bool print) {
        if (print)
            return FJ_SCANNER_LUT_DESCRIPTOR_PRINT_RECIPE;
        if (diagnostic.find("direct recipe") != std::string::npos)
            return FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_RECIPE;
        if (diagnostic.find("density bounds") != std::string::npos)
            return FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_BOUNDS;
        if (diagnostic.find("observer identity") != std::string::npos)
            return FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_OBSERVER;
        if (diagnostic.find("illuminant identity") != std::string::npos)
            return FJ_SCANNER_LUT_DESCRIPTOR_DIRECT_ILLUMINANT;
        throw std::runtime_error("unknown frozen scanner descriptor diagnostic");
    }
    void values(const Json& rows, bool products) {
        require(rows.size() == (products ? 4u : 520u), "scanner descriptor frozen membership");
        constexpr std::array<const char*, 5> retiredNames{
            "missing-1", "missing-2", "missing-3", "null-preempts-route-4", "null-preempts-route-255"};
        std::array<std::array<bool, 5>, 4> retiredRows{};
        std::size_t representableRows = 0;
        std::size_t retiredPointerRows = 0;
        for (const auto& row : rows) {
            const auto& in = row.at("input");
            if (in.at("missing").get<unsigned>() >= 1 && in.at("missing").get<unsigned>() <= 3) {
                // These recorded null model-pointer outcomes have no raw ABI equivalent.
                bool identified = false;
                for (unsigned route = 0; route < 4; ++route) {
                    for (std::size_t index = 0; index < retiredNames.size(); ++index) {
                        if (row.at("id") != std::to_string(route) + "/" + retiredNames[index])
                            continue;
                        require(!products && !retiredRows[route][index], "each retired pointer row is named once");
                        require(in.at("missing") == (index < 3 ? index + 1 : 1), "retired pointer marker is unchanged");
                        const auto& expected = row.at("native");
                        require(!expected.at("success").get<bool>() && !expected.at("exception").get<bool>() &&
                                    expected.at("diagnostic") == (route % 2 == 0
                                                                      ? "ResourceDescriptorMismatch phase=3A scanner direct inputs unavailable"
                                                                      : "ResourceDescriptorMismatch phase=4C scanner print inputs unavailable"),
                                "retained historical pointer-only disposition");
                        retiredRows[route][index] = true;
                        identified = true;
                    }
                }
                require(identified, "only the twenty named model-pointer rows retire");
                ++retiredPointerRows;
                continue;
            }
            ++representableRows;
            for (auto call : {fj_legacy_scanner_lut_describe, fj_test_scanner_lut_describe}) {
                const Borrowed borrowed(in);
                FjScannerLutDescriptor out;
                std::memset(&out, 0xa5, sizeof out);
                FjScannerLutDescriptorFailure failure{99, 99};
                std::array<char, 256> message{};
                FjErrorBuffer error{message.data(), message.size(), 99};
                const auto result = call(&borrowed.wire, &out, &failure, &error);
                const auto& expected = row.at("native");
                if (expected.at("success").get<bool>()) {
                    status(result);
                    require(failure.reason == 0 && failure.lane == 0 && error.length == 0 && message[0] == 0, "descriptor success envelope");
                    require(project(out) == expected.at("descriptor"), "all descriptor fields exact");
                    require(fj_test_scanner_descriptor_c_call(&borrowed.wire, &out), "compiled C11 descriptor call");
                } else {
                    const bool invalidRoute = in.at("route").get<unsigned>() > 3;
                    status(result, invalidRoute ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_PREPARATION_FAILURE);
                    require(cleared(out) && failure.lane == 0, "raw failure is all-zero independently of native defaults");
                    require(failure.reason == (invalidRoute ? 1 : reason(expected.at("diagnostic"), in.at("mode") == 1)), "exact descriptor rejection reason");
                    require(std::string(message.data(), error.length) == (invalidRoute ? "invalid scanner LUT descriptor arguments" : expected.at("diagnostic").get<std::string>()), "descriptor diagnostic continuity");
                }
            }
        }
        require(representableRows == (products ? 4u : 500u) &&
                    retiredPointerRows == (products ? 0u : 20u),
                "raw descriptor rows and retired model-pointer membership");
    }
    struct Rejection {
        unsigned category = FJ_STATUS_UNSUPPORTED_INPUT;
        unsigned reason = 1;
        unsigned lane = 0;
    };
    void boundary(const Json& rows) {
        for (unsigned mode = 0; mode < 2; ++mode) {
            const Borrowed borrowed(rows.at(mode ? rows.size() / 4 : 0).at("input"));
            const auto good = borrowed.wire;
            const auto reject = [&](const FjScannerLutDescriptorInput* in, Rejection expected = {}) {
                FjScannerLutDescriptor out;
                std::memset(&out, 0xa5, sizeof out);
                FjScannerLutDescriptorFailure failure{99, 99};
                char message[8]{};
                FjErrorBuffer error{message, sizeof message, 99};
                status(fj_legacy_scanner_lut_describe(in, &out, &failure, &error), expected.category);
                require(cleared(out) && failure.reason == expected.reason && failure.lane == expected.lane, "closed descriptor failure and zero result");
                require(error.length < sizeof message && message[error.length] == 0, "bounded descriptor diagnostic");
            };
            reject(nullptr);
            reject(reinterpret_cast<const FjScannerLutDescriptorInput*>(std::uintptr_t{1}));
            for (unsigned which = 0; which < 16; ++which) {
                auto bad = good;
                switch (which) {
                    case 0:
                        bad.mode = 2;
                        break;
                    case 1:
                        bad.route = 4;
                        break;
                    case 2:
                        bad.capture_polarity = 256;
                        break;
                    case 3:
                        bad.bounds_medium = 256;
                        break;
                    case 4:
                        bad.profile_present = 2;
                        break;
                    case 5:
                        bad.channel_density_cmy.count = 242;
                        break;
                    case 6:
                        bad.base_density.count = 82;
                        break;
                    case 7:
                        bad.channel_density_cmy.data = nullptr;
                        break;
                    case 8:
                        bad.base_density.data = reinterpret_cast<const float*>(std::uintptr_t{1});
                        break;
                    case 9:
                        bad.viewing_illuminant = {nullptr, 1};
                        break;
                    case 10:
                        bad.observer_identity.count = SIZE_MAX;
                        break;
                    case 11:
                        bad.observer_identity = {reinterpret_cast<const std::uint8_t*>(UINTPTR_MAX - 4), 10};
                        break;
                    case 12:
                        bad.base_density.count = SIZE_MAX;
                        break;
                    case 13:
                        bad.channel_density_cmy.data = reinterpret_cast<const float*>(UINTPTR_MAX - 3);
                        break;
                    case 14:
                        bad.mode = 1;
                        bad.print_profile_key = {nullptr, 1};
                        break;
                    case 15:
                        bad.mode = 1;
                        bad.profile_viewing_illuminant = {nullptr, 1};
                        break;
                }
                reject(&bad);
            }
            FjScannerLutDescriptor out;
            FjScannerLutDescriptorFailure failure{99, 99};
            status(fj_legacy_scanner_lut_describe(&good, nullptr, &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(failure.reason == 1 && failure.lane == 0, "failure independently cleared/admitted without result");
            std::memset(&out, 0xa5, sizeof out);
            status(fj_legacy_scanner_lut_describe(&good, &out, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(out), "result independently cleared without failure");
            alignas(8) std::array<std::uint8_t, 80> unaligned{};
            unaligned.fill(0xa5);
            status(fj_legacy_scanner_lut_describe(&good, reinterpret_cast<FjScannerLutDescriptor*>(unaligned.data() + 1), &failure, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(failure.reason == 1 && failure.lane == 0, "unaligned result leaves independent failure");
            std::memset(&out, 0xa5, sizeof out);
            status(fj_legacy_scanner_lut_describe(&good, &out, reinterpret_cast<FjScannerLutDescriptorFailure*>(unaligned.data() + 1), nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(out), "unaligned failure leaves independently cleared result");
            status(fj_legacy_scanner_lut_describe(&good, &out, &failure, reinterpret_cast<FjErrorBuffer*>(unaligned.data() + 1)), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(out) && failure.reason == 1 && std::all_of(unaligned.begin(), unaligned.end(), [](auto byte) {
                        return byte == 0xa5;
                    }),
                    "unaligned diagnostic leaves no backing writes");
            FjErrorBuffer invalid{nullptr, 1, 99};
            std::memset(&out, 0xa5, sizeof out);
            status(fj_legacy_scanner_lut_describe(&good, &out, &failure, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(out) && failure.reason == 1 && failure.lane == 0, "invalid diagnostic clears both outputs");
            FjErrorBuffer zero{reinterpret_cast<char*>(std::uintptr_t{1}), 0, 99};
            status(fj_legacy_scanner_lut_describe(&good, &out, &failure, &zero));
            require(zero.length == 0, "zero capacity diagnostic never reads backing");
            auto ignored = good;
            if (!mode) {
                ignored.bounds_source = ignored.output_medium = UINT32_MAX;
                ignored.profile_viewing_illuminant = ignored.print_profile_key = {nullptr, SIZE_MAX};
                status(fj_legacy_scanner_lut_describe(&ignored, &out, &failure, nullptr));
            } else {
                ignored.bounds_source = 256;
                reject(&ignored);
                ignored = good;
                ignored.output_medium = 256;
                reject(&ignored);
            }
            auto absent = good;
            absent.profile_present = 0;
            absent.channel_density_cmy = absent.base_density = {nullptr, SIZE_MAX};
            absent.profile_viewing_illuminant = {nullptr, SIZE_MAX};
            reject(&absent, {FJ_STATUS_PREPARATION_FAILURE, mode ? 7u : 2u});
            for (unsigned fault = 1; fault <= 6; ++fault) {
                status(fj_test_scanner_lut_descriptor_fault(fault));
                reject(&absent, {FJ_STATUS_PREPARATION_FAILURE, mode ? 7u : 2u});
                reject(&good, {fault == 6 ? FJ_STATUS_INTERNAL_FAILURE : FJ_STATUS_PREPARATION_FAILURE, fault == 6 ? 9u : mode ? 8u
                                                                                                                               : 6u,
                               fault == 6 ? 0u : fault});
                status(fj_legacy_scanner_lut_describe(&good, &out, &failure, nullptr));
                require(failure.reason == 0 && failure.lane == 0, "one-shot descriptor fault recovers");
            }
            status(fj_test_scanner_lut_descriptor_fault(7), FJ_STATUS_UNSUPPORTED_INPUT);
        }
    }

    void lifetime(const Json& rows) {
        const Borrowed borrowed(rows[0].at("input"));
        FjScannerLutDescriptor retained{};
        {
            Borrowed expiring(rows[0].at("input"));
            FjScannerLutDescriptorFailure failure{};
            status(fj_legacy_scanner_lut_describe(&expiring.wire, &retained, &failure, nullptr));
        }
        status(fj_test_scanner_lut_descriptor_fault(1));
        std::thread independent([&] {
            FjScannerLutDescriptor result{};
            FjScannerLutDescriptorFailure failure{};
            status(fj_legacy_scanner_lut_describe(&borrowed.wire, &result, &failure, nullptr));
            require(project(result) == project(retained), "owned result survives source expiry and later cross-thread preparation");
        });
        independent.join();
        FjScannerLutDescriptor result{};
        FjScannerLutDescriptorFailure failure{};
        status(fj_legacy_scanner_lut_describe(&borrowed.wire, &result, &failure, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        require(failure.reason == 6 && failure.lane == 1, "other thread did not consume local descriptor fault");
    }
    void callers(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        struct ClearReferenceFault {
            ~ClearReferenceFault() {
                fj_test_exposure_clear_fault();
            }
        } clearReferenceFault;
        for (unsigned r : {0u, 2u}) {
            auto controls = FilmReferenceTest::controls(1, static_cast<Spektrafilm::ScanRoute>(r), 0);

            // The accepted construction path is the held Rust state. Arm an
            // actual early scanner descriptor fault and the later reference
            // source fault together: the first create must consume only the
            // descriptor operation, and the next must reach the reference.
            status(fj_test_exposure_arm_fault(5, 1, 1));
            status(fj_test_scanner_lut_descriptor_fault(1));
            std::array<char, 2048> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            std::optional<PreparedState> prepared;
            auto* const assets = JuicerProcess::root().assets().borrow_rust_assets();
            auto preparedStatus = prepare_state(assets, controls, prepared, error);
            require(preparedStatus.category == FJ_STATUS_PREPARATION_FAILURE && !prepared,
                    "actual Rust owner rejects the early scanner descriptor fault");
            require(std::string(message.data(), error.length) ==
                        "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid",
                    "actual Rust owner preserves the early descriptor diagnostic");
            require(fj_test_exposure_fault_consumed(5, 1) == 0,
                    "early scanner descriptor failure leaves later reference fault pending");
            require(fj_test_film_development_live_owners() == 0,
                    "failed Rust construction leaves no legacy development owner");

            error.length = 0;
            message[0] = 0;
            preparedStatus = prepare_state(assets, controls, prepared, error);
            require(preparedStatus.category == FJ_STATUS_UNSUPPORTED_INPUT && !prepared,
                    "one-shot descriptor recovery reaches the later reference source fault");
            require(std::string(message.data(), error.length) ==
                        "Reference source preparation: exposure production boundary fault",
                    "later reference source retains its exact diagnostic");
            require(fj_test_exposure_fault_consumed(5, 1) == 1 &&
                        fj_test_film_development_live_owners() == 0,
                    "later source fault is consumed once without a leaked legacy owner");

            error.length = 0;
            message[0] = 0;
            preparedStatus = prepare_state(assets, controls, prepared, error);
            require(preparedStatus.category == FJ_STATUS_SUCCESS && prepared,
                    "actual Rust state construction recovers after both one-shot faults");
            FjPreparedStateView actual{};
            status(prepared->view(actual, error));
            const auto described = describe(prepared_input(actual, "CIE1931_2deg_380_780_5nm"));
            require(project(actual.scanner_lut) == project(described),
                    "actual held-state scanner descriptor matches the raw descriptor consumer");

            const auto retainedDescriptor = project(actual.scanner_lut);
            const auto originalBoundsHash = actual.scanner_bounds.hash;
            auto changedControls = controls;
            changedControls.filmProfileKey = r == 0 ? "kodak_portra_800" : "fujifilm_velvia_100";
            std::optional<PreparedState> changed;
            error.length = 0;
            message[0] = 0;
            preparedStatus = prepare_state(assets, changedControls, changed, error);
            require(preparedStatus.category == FJ_STATUS_SUCCESS && changed,
                    "actual Rust owner accepts the selected replacement profile");
            FjPreparedStateView changedView{};
            status(changed->view(changedView, error));
            require(originalBoundsHash != changedView.scanner_bounds.hash &&
                        actual.scanner_lut.density_bounds_hash !=
                            changedView.scanner_lut.density_bounds_hash &&
                        actual.scanner_lut.hash != changedView.scanner_lut.hash,
                    "actual held-state profile change propagates its bound dependency through the scanner descriptor");
            FjPreparedStateView retained{};
            status(prepared->view(retained, error));
            require(project(retained.scanner_lut) == retainedDescriptor,
                    "held original Rust state keeps its descriptor readable after a distinct profile owner is constructed");
            changed.reset();
            prepared.reset();
        }
    }

    void admitted_token_change_preserves_identity_and_viewing_label_mismatch_is_rejected(const Json& rows) {
        for (unsigned route : {1u, 3u}) {
            const auto prefix = std::to_string(route) + "/";
            const auto find = [&](const char* name) -> const Json& {
                return *std::find_if(rows.begin(), rows.end(), [&](const Json& row) {
                    return row.at("id") == prefix + name;
                });
            };
            const auto baseline = describe(find("normal").at("input"));
            const auto changed = describe(find("admitted-token-key-change").at("input"));
            require(project(changed) == project(baseline), "admitted nonzero print token/key changes preserve complete raw identity");
            const Borrowed rejected(find("empty-profile_viewing_illuminant").at("input"));
            FjScannerLutDescriptor result;
            std::memset(&result, 0xa5, sizeof result);
            FjScannerLutDescriptorFailure failure{};
            std::array<char, 256> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            status(fj_test_scanner_lut_describe(&rejected.wire, &result, &failure, &error), FJ_STATUS_PREPARATION_FAILURE);
            require(failure.reason == FJ_SCANNER_LUT_DESCRIPTOR_PRINT_RECIPE && failure.lane == 0 && cleared(result) &&
                        std::string(message.data(), error.length) == "ResourceDescriptorMismatch phase=4C scanner print recipe",
                    "raw viewing-label mismatch retains typed failure, exact diagnostic and zero output");
        }
    }
} // namespace ScannerDescriptorTest
