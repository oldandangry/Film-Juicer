namespace ScannerDescriptorTest {
    Json project(const Scanner::ScannerSpectralLutDescriptor& d) {
        return {{"route", static_cast<unsigned>(d.route)}, {"medium", static_cast<unsigned>(d.medium)}, {"capture_polarity", static_cast<unsigned>(d.polarity)}, {"resolution", d.lutResolution}, {"density_bounds_hash", d.densityBoundsHash}, {"channel_density_hash", d.channelDensityHash}, {"base_density_hash", d.baseDensityHash}, {"viewing_illuminant_hash", d.scanIlluminantHash}, {"observer_hash", d.observerHash}, {"hash", d.hash}};
    }
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
    Json input(const ProfileRoute& route, const DensityBoundsRecipe& bounds, const ScannerOutputRecipe& output, const std::string& observer, bool print, unsigned missing = 0) {
        const auto& channel = print ? route.printProfile->data.channelDensity : route.filmProfile->data.channelDensity;
        const auto& base = print ? route.printProfile->data.baseDensity : route.filmProfile->data.baseDensity;
        return {{"mode", print ? 1 : 0}, {"route", static_cast<unsigned>(route.scanRoute)}, {"capture_polarity", static_cast<unsigned>(route.capturePolarity)}, {"bounds_medium", static_cast<unsigned>(bounds.medium)}, {"bounds_source", static_cast<unsigned>(bounds.source)}, {"output_medium", static_cast<unsigned>(output.medium)}, {"profile_present", missing != 4}, {"missing", missing}, {"density_bounds_hash", bounds.hash}, {"print_profile_token", route.printProfileAssetVersionToken}, {"print_profile_key", bytes(route.printProfileKey)}, {"channel_density_cmy", words(&channel[0][0], 243)}, {"base_density", words(base.data(), 81)}, {"viewing_illuminant", bytes(output.viewingIlluminant)}, {"profile_viewing_illuminant", bytes(print ? route.printProfile->info.viewingIlluminant.value : route.filmProfile->info.viewingIlluminant.value)}, {"observer_identity", bytes(observer)}, {"resolution", output.lutResolution}};
    }
    Json native(const Json& in) {
        auto film = std::make_shared<Profiles::FilmProfile>();
        auto print = std::make_shared<Profiles::PrintProfile>();
        for (std::size_t i = 0; i < 243; ++i) {
            film->data.channelDensity[i / 3][i % 3] = print->data.channelDensity[i / 3][i % 3] = std::bit_cast<float>(in.at("channel_density_cmy")[i].get<std::uint32_t>());
        }
        for (std::size_t i = 0; i < 81; ++i)
            film->data.baseDensity[i] = print->data.baseDensity[i] = std::bit_cast<float>(in.at("base_density")[i].get<std::uint32_t>());
        film->info.viewingIlluminant.value = print->info.viewingIlluminant.value = text(in.at("profile_viewing_illuminant"));
        ProfileRoute route;
        route.scanRoute = static_cast<Spektrafilm::ScanRoute>(in.at("route").get<unsigned>());
        route.capturePolarity = static_cast<Spektrafilm::ProfilePolarity>(in.at("capture_polarity").get<unsigned>());
        if (in.at("profile_present").get<bool>()) {
            route.filmProfile = film;
            route.printProfile = print;
        }
        route.printProfileKey = text(in.at("print_profile_key"));
        route.printProfileAssetVersionToken = in.at("print_profile_token");
        DensityBoundsRecipe bounds;
        bounds.medium = static_cast<Spektrafilm::DensityMedium>(in.at("bounds_medium").get<unsigned>());
        bounds.source = static_cast<Spektrafilm::DensityBoundsSource>(in.at("bounds_source").get<unsigned>());
        bounds.hash = in.at("density_bounds_hash");
        ScannerOutputRecipe output;
        output.medium = static_cast<Spektrafilm::DensityMedium>(in.at("output_medium").get<unsigned>());
        output.viewingIlluminant = text(in.at("viewing_illuminant"));
        output.lutResolution = in.at("resolution");
        const auto observer = text(in.at("observer_identity"));
        const unsigned missing = in.at("missing");
        Scanner::ScannerSpectralLutDescriptor result;
        result.hash = 99;
        std::string diagnostic = "seed";
        bool success = false;
        bool exception = false;
        try {
            success = in.at("mode") == 1 ? Scanner::build_print_scanner_spectral_lut_descriptor({missing == 1 ? nullptr : &route, missing == 2 ? nullptr : &bounds, missing == 3 ? nullptr : &output, observer}, result, diagnostic)
                                         : Scanner::build_direct_scanner_spectral_lut_descriptor({missing == 1 ? nullptr : &route, missing == 2 ? nullptr : &bounds, missing == 3 ? nullptr : &output, observer}, result, diagnostic);
        } catch (const std::out_of_range&) {
            exception = true;
        }
        return {{"success", success}, {"exception", exception}, {"diagnostic", diagnostic}, {"descriptor", project(result)}};
    }
    Json cases() {
        auto film = std::make_shared<Profiles::FilmProfile>();
        auto print = std::make_shared<Profiles::PrintProfile>();
        for (unsigned i = 0; i < 243; ++i)
            film->data.channelDensity[i / 3][i % 3] = print->data.channelDensity[i / 3][i % 3] = static_cast<float>(i + 1) / 256;
        for (unsigned i = 0; i < 81; ++i)
            film->data.baseDensity[i] = print->data.baseDensity[i] = static_cast<float>(i) / 128;
        Json rows = Json::array();
        for (unsigned r = 0; r < 4; ++r) {
            const bool printing = r % 2;
            ProfileRoute route;
            route.scanRoute = static_cast<Spektrafilm::ScanRoute>(r);
            route.capturePolarity = static_cast<Spektrafilm::ProfilePolarity>(r / 2);
            route.filmProfile = film;
            route.printProfile = print;
            route.printProfileKey = "selected";
            route.printProfileAssetVersionToken = 42;
            DensityBoundsRecipe bounds;
            bounds.medium = printing ? Spektrafilm::DensityMedium::Print : Spektrafilm::DensityMedium::Film;
            bounds.source = Spektrafilm::DensityBoundsSource::PrintMediaAuthoredCurves;
            bounds.hash = 0x123456789abcdef0;
            ScannerOutputRecipe output;
            output.medium = bounds.medium;
            output.viewingIlluminant = "D50";
            output.lutResolution = 17;
            const auto original = input(route, bounds, output, "CIE1931_2deg", printing);
            const auto add = [&](const std::string& id, Json value) {
                rows.push_back({{"id", std::to_string(r) + "/" + id}, {"input", value}, {"native", native(value)}});
            };
            add("normal", original);
            for (unsigned resolution : {0u, 1u, 33u, UINT32_MAX}) {
                auto value = original;
                value["resolution"] = resolution;
                add("resolution-" + std::to_string(resolution), value);
            }
            for (unsigned polarity : {2u, 255u, 1u - r / 2}) {
                auto value = original;
                value["capture_polarity"] = polarity;
                add("polarity-" + std::to_string(polarity), value);
            }
            for (unsigned missing = 1; missing <= 4; ++missing) {
                auto value = original;
                value["missing"] = missing;
                value["profile_present"] = missing != 4;
                add("missing-" + std::to_string(missing), value);
            }
            for (const char* field : {"density_bounds_hash", "print_profile_token", "bounds_medium", "bounds_source", "output_medium", "route"}) {
                auto value = original;
                value[field] = std::string_view(field) == "route" ? (r ^ 1) : std::string_view(field) == "density_bounds_hash" || std::string_view(field) == "print_profile_token" ? 0
                                                                                                                                                                                   : 255;
                add(std::string("predicate-") + field, value);
            }
            for (const char* field : {"print_profile_key", "viewing_illuminant", "observer_identity", "profile_viewing_illuminant"}) {
                auto value = original;
                value[field] = Json::array();
                add(std::string("empty-") + field, value);
            }
            auto value = original;
            value["viewing_illuminant"] = value["profile_viewing_illuminant"] = Json::array();
            add("both-empty", value);
            value["observer_identity"] = Json::array();
            value["density_bounds_hash"] = 0;
            add("priority-bounds-labels", value);
            value = original;
            value["viewing_illuminant"] = value["observer_identity"] = Json::array();
            add("priority-observer-label", value);
            value = original;
            value["viewing_illuminant"] = value["profile_viewing_illuminant"] = {68, 0, 255, 32};
            value["observer_identity"] = {0, 255};
            add("exact-bytes", value);
            for (unsigned invalid : {4u, 255u}) {
                value = original;
                value["route"] = invalid;
                add("invalid-route-" + std::to_string(invalid), value);
                value["missing"] = 1;
                add("null-preempts-route-" + std::to_string(invalid), value);
            }
            value = original;
            value["print_profile_token"] = 43;
            value["print_profile_key"] = {99};
            add("admitted-token-key-change", value);
            value = original;
            value["density_bounds_hash"] = 9;
            add("bounds-dependency", value);
            for (const char* field : {"channel_density_cmy", "base_density"}) {
                const std::vector<unsigned> positions = std::string_view(field) == "base_density" ? std::vector<unsigned>{0, 63, 64, 80} : std::vector<unsigned>{0, 1, 2, 63, 64, 127, 128, 191, 192, 242};
                for (unsigned position : positions)
                    for (std::uint32_t word : {0u, 0x80000000u, 0x7fc00001u, 0xffc00123u, 0x7f800000u, 0xff800000u, 0x3f012345u}) {
                        value = original;
                        value[field][position] = word;
                        add(std::string(field) + "-" + std::to_string(position) + "-" + std::to_string(word), value);
                    }
            }
        }
        return rows;
    }
    Json products(const std::filesystem::path& resources) {
        JuicerCuda::Owner owner;
        owner.create(resources);
        JuicerProcess::root().ensure_bootstrap();
        Json rows = Json::array();
        for (unsigned r = 0; r < 4; ++r) {
            auto params = FilmReferenceTest::controls(1, static_cast<Spektrafilm::ScanRoute>(r), 0);
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            require(r % 2 ? build_print_render_state_product(params, product, diagnostic) : build_direct_render_state_product(params, product, diagnostic), diagnostic.c_str());
            const auto& recipe = product.recipe;
            const auto value = input(recipe.profileRoute, recipe.densityBounds, recipe.scannerOutput, "CIE1931_2deg", r % 2);
            rows.push_back({{"route", r}, {"input", value}, {"native", native(value)}, {"snapshot_hash", hash_params(params)}, {"final_hash", recipe.hash}, {"scanner_hash", recipe.scannerOutput.hash}, {"upload_hash", product.payload.uploadCoreHash}});
        }
        return rows;
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
    void values(const Json& rows) {
        for (const auto& row : rows) {
            const auto& in = row.at("input");
            require(native(in) == row.at("native"), row.value("id", "product").c_str());
            if (in.at("missing").get<unsigned>() >= 1 && in.at("missing").get<unsigned>() <= 3)
                continue;
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

    thread_local unsigned projectionFault = 0;
    void corrupt(FjStatus& status, FjScannerLutDescriptor& result, FjScannerLutDescriptorFailure& failure, FjErrorBuffer& error) {
        switch (std::exchange(projectionFault, 0u)) {
            case 1:
                status.api = 1;
                break;
            case 2:
                status.native_code = 1;
                break;
            case 3:
                status.category = 99;
                break;
            case 4:
                failure.reason = 99;
                break;
            case 5:
                failure.lane = 1;
                break;
            case 6:
                ++result.route;
                break;
            case 7:
                result.medium ^= 1;
                break;
            case 8:
                ++result.capture_polarity;
                break;
            case 9:
                ++result.resolution;
                break;
            case 10:
                ++result.density_bounds_hash;
                break;
            case 11:
                result.channel_density_hash = 0;
                break;
            case 12:
                result.base_density_hash = 0;
                break;
            case 13:
                result.viewing_illuminant_hash = 0;
                break;
            case 14:
                result.observer_hash = 0;
                break;
            case 15:
                result.hash = 0;
                break;
            case 16:
                error.length = error.capacity;
                break;
            case 17:
                error.length = 1;
                break;
            case 18:
                failure.reason = 7;
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                break;
            case 19:
                failure.reason = 6;
                failure.lane = 6;
                status.category = FJ_STATUS_PREPARATION_FAILURE;
                break;
            case 20:
                error.data = nullptr;
                break;
            default:
                break;
        }
    }
    void adapters(const Json& rows) {
        struct Reset {
            ~Reset() {
                JuicerScanner::Test::set_descriptor_projection(nullptr);
            }
        } reset;
        JuicerScanner::Test::set_descriptor_projection(corrupt);
        for (unsigned mode = 0; mode < 2; ++mode) {
            const auto& in = rows.at(mode ? rows.size() / 4 : 0).at("input");
            for (unsigned fault = 1; fault <= 20; ++fault) {
                projectionFault = fault;
                bool caught = false;
                try {
                    native(in);
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    caught = true;
                    require(failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE && failure.failure.status.api == FJ_API_NONE && failure.failure.status.native_code == 0 && failure.failure.diagnostic == "scanner LUT descriptor projection invalid", "corrupt descriptor envelope typed failure");
                }
                require(caught && projectionFault == 0, "every corrupt descriptor projection rejected");
                require(native(in) == rows.at(mode ? rows.size() / 4 : 0).at("native"), "projection recovers");
            }
            status(fj_test_scanner_lut_descriptor_fault(6));
            bool caught = false;
            try {
                native(in);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                caught = true;
                require(failure.failure.status.category == FJ_STATUS_INTERNAL_FAILURE && failure.failure.status.api == 0 && failure.failure.status.native_code == 0 && failure.failure.diagnostic == "scanner LUT descriptor preparation panicked", "native descriptor contained panic");
            }
            require(caught, "panic remains typed at native adapter");
            for (unsigned fault = 1; fault <= 5; ++fault) {
                status(fj_test_scanner_lut_descriptor_fault(fault));
                const auto rejected = native(in);
                require(!rejected.at("success").get<bool>() && rejected.at("descriptor") == project(Scanner::ScannerSpectralLutDescriptor{}), "native hash failures retain default carrier");
                require(rejected.at("diagnostic") == (mode ? "ResourceDescriptorMismatch phase=4C scanner print descriptor hash" : "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid"), "native hash diagnostic");
            }
        }
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
        JuicerProcess::root().ensure_bootstrap();
        for (unsigned r : {0u, 2u}) {
            auto controls = FilmReferenceTest::controls(1, static_cast<Spektrafilm::ScanRoute>(r), 0);
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            FilmReferenceTest::observeHold = true;
            FilmReferenceTest::nextProjection = 1;
            status(fj_test_scanner_lut_descriptor_fault(1));
            require(!build_direct_render_state_product(controls, product, diagnostic), "actual direct state rejects descriptor before reference completion");
            require(diagnostic == "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid", "direct state keeps early descriptor failure");
            require(FilmReferenceTest::nextProjection == 1, "early descriptor failure precedes the armed later reference projection");
            FilmReferenceTest::nextProjection = 0;
            require(fj_test_film_development_live_owners() == 0, "early descriptor failure drops moved preparation hold");
            require(build_direct_render_state_product(controls, product, diagnostic), diagnostic.c_str());
            FilmReferenceTest::observeHold = false;
            const auto before = product.recipe.densityBounds.hash;
            controls.filmProfileKey = r == 0 ? "kodak_portra_800" : "fujifilm_velvia_100";
            FocusedRenderStateBuildProduct changed;
            require(build_direct_render_state_product(controls, changed, diagnostic), diagnostic.c_str());
            require(before != changed.recipe.densityBounds.hash, "actual selected profile changes bound dependency");
            Scanner::ScannerSpectralLutDescriptor first, second;
            const auto& a = product.recipe;
            const auto& b = changed.recipe;
            require(Scanner::build_direct_scanner_spectral_lut_descriptor({&a.profileRoute, &a.densityBounds, &a.scannerOutput}, first, diagnostic) && Scanner::build_direct_scanner_spectral_lut_descriptor({&b.profileRoute, &b.densityBounds, &b.scannerOutput}, second, diagnostic), "actual dependency descriptors");
            require(first.densityBoundsHash != second.densityBoundsHash && first.hash != second.hash, "actual caller propagates upstream bound identity");
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
            const auto baseline = native(find("normal").at("input"));
            const auto changed = native(find("admitted-token-key-change").at("input"));
            require(baseline.at("success").get<bool>() && changed == baseline, "admitted nonzero print token/key changes preserve complete identity");
            const auto rejected = native(find("empty-profile_viewing_illuminant").at("input"));
            require(!rejected.at("success").get<bool>() && rejected.at("diagnostic") == "ResourceDescriptorMismatch phase=4C scanner print recipe" && rejected.at("descriptor") == project(Scanner::ScannerSpectralLutDescriptor{}), "public viewing-label mismatch replacement rejects with native defaults");
        }
    }
} // namespace ScannerDescriptorTest
