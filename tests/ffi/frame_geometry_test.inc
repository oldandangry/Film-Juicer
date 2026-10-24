namespace FrameGeometryTest {
    Json fields(FjExtent d) {
        return Json::array({d.x, d.y, d.width, d.height});
    }
    FjExtent extent(const Json& a) {
        return {a.at(0), a.at(1), a.at(2), a.at(3)};
    }
    bool cleared(FjExtent d) {
        return fields(d) == Json::array({0, 0, 0, 0});
    }
    void values(const Json& captured, bool products) {
        for (const auto& row : captured.at("domains")) {
            const auto& b = row.at("bounds");
            const FjRect input{b.at(0), b.at(1), b.at(2), b.at(3)};
            if (!products)
                for (auto prepare : {fj_legacy_frame_domain_prepare, fj_test_frame_domain_prepare, fj_test_frame_domain_c_prepare}) {
                    FjExtent out{7, 8, 9, 10};
                    FjFrameGeometryFailure failure{99};
                    char text[128] = "stale";
                    FjErrorBuffer error{text, sizeof(text), 99};
                    const bool range = row.at("range");
                    status(prepare(&input, &out, &failure, &error), range ? 3u : 0u);
                    require(failure.reason == (range ? 2u : 0u), "frame domain reason");
                    require(range ? cleared(out) : fields(out) == row.at("domain"), "every frozen signed domain field");
                    require(std::string_view(text, error.length) == (range ? "frame descriptor dimensions exceed signed integer range" : ""), "exact domain diagnostic");
                }
            if (products) {
                JuicerFrame::Domain out{};
                JuicerFrame::Failure failure;
                const bool ready = JuicerFrame::domain_from_bounds({input.x1, input.y1, input.x2, input.y2}, out, failure);
                require(ready != row.at("range").get<bool>(), "native domain wrapper result");
                if (ready)
                    require(Json::array({out.x, out.y, out.width, out.height}) == row.at("domain"), "native wrapper complete signed domain");
                else
                    require(failure.reason == JuicerFrame::FailureReason::DomainRange, "native typed range failure");
            }
        }
        for (const auto& row : captured.at("geometry")) {
            const auto d = extent(row.at("domain"));
            const FjFrameGeometryInput input{d, std::bit_cast<float>(row.at("film_bits").get<std::uint32_t>())};
            if (!products)
                for (auto prepare : {fj_legacy_frame_geometry_prepare, fj_test_frame_geometry_prepare, fj_test_frame_geometry_c_prepare}) {
                    FjFrameGeometry out{{7, 8, 9, 10}, -9};
                    FjFrameGeometryFailure failure{99};
                    status(prepare(&input, &out, &failure, nullptr));
                    require(failure.reason == 0 && fields(out.full_frame) == row.at("domain"), "complete geometry domain fields");
                    require(std::bit_cast<std::uint32_t>(out.pixel_size_um) == row.at("pixel_bits"), "frozen final f32 pixel word");
                    require(std::bit_cast<std::uint64_t>(static_cast<double>(out.pixel_size_um)) == row.at("diffusion_bits"), "frozen widened diffusion word");
                }
            if (products) {
                JuicerFrame::Geometry out{};
                JuicerFrame::Failure failure;
                require(JuicerFrame::prepare_geometry({d.x, d.y, d.width, d.height}, input.film_format_long_edge_mm, out, failure), "native scalar wrapper accepts every contract result");
                require(Json::array({out.fullFrame.x, out.fullFrame.y, out.fullFrame.width, out.fullFrame.height}) == row.at("domain") &&
                            std::bit_cast<std::uint32_t>(out.pixelSizeUm) == row.at("pixel_bits") &&
                            std::bit_cast<std::uint64_t>(static_cast<double>(out.pixelSizeUm)) == row.at("diffusion_bits"),
                        "native full geometry projection");
            }
        }
    }

    template <typename Input, typename Output, typename Prepare>
    void raw(const Input& input, Prepare prepare, unsigned operation) {
        for (unsigned kind = 0; kind < 14; ++kind) {
            Output out{};
            std::memset(&out, 0x55, sizeof(out));
            FjFrameGeometryFailure failure{99};
            char text[128] = "stale";
            FjErrorBuffer error{text, sizeof(text), 99};
            auto* diagnostic = &error;
            const Input* in = &input;
            Output* output = &out;
            auto* rejected = &failure;
            switch (kind) {
                case 0:
                    in = nullptr;
                    break;
                case 1:
                    output = nullptr;
                    break;
                case 2:
                    rejected = nullptr;
                    break;
                case 3:
                    in = reinterpret_cast<const Input*>(std::uintptr_t{1});
                    break;
                case 4:
                    output = reinterpret_cast<Output*>(std::uintptr_t{1});
                    break;
                case 5:
                    rejected = reinterpret_cast<FjFrameGeometryFailure*>(std::uintptr_t{1});
                    break;
                case 6:
                    in = reinterpret_cast<const Input*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
                    break;
                case 7:
                    output = reinterpret_cast<Output*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
                    break;
                case 8:
                    rejected = reinterpret_cast<FjFrameGeometryFailure*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
                    break;
                case 9:
                    error.data = nullptr;
                    error.capacity = 1;
                    break;
                case 10:
                    diagnostic = reinterpret_cast<FjErrorBuffer*>(std::uintptr_t{1});
                    break;
                case 11:
                    error.data = reinterpret_cast<char*>(std::numeric_limits<std::uintptr_t>::max() - 1u);
                    error.capacity = 8;
                    break;
                case 12:
                    diagnostic = nullptr;
                    break;
                case 13:
                    error.data = reinterpret_cast<char*>(std::uintptr_t{1});
                    error.capacity = 0;
                    break;
            }
            const bool success = kind >= 12;
            status(prepare(in, output, rejected, diagnostic), success ? 0u : 2u);
            if (rejected == &failure)
                require(failure.reason == (success ? 0u : 1u), "raw failure peer independently cleared then assigned");
            if (output == &out && !success) {
                if constexpr (std::is_same_v<Output, FjExtent>)
                    require(cleared(out), "domain cleared despite malformed peer");
                else
                    require(cleared(out.full_frame) && std::bit_cast<std::uint32_t>(out.pixel_size_um) == 0, "geometry cleared despite malformed peer");
            }
            if (kind < 9)
                require(std::string_view(text, error.length) == "invalid frame geometry arguments", "exact raw diagnostic");
            if (kind == 13)
                require(error.length == 0, "zero capacity diagnostics never touch backing");
        }
        for (std::size_t capacity : {std::size_t{1}, std::size_t{8}}) {
            Output out{};
            FjFrameGeometryFailure failure{};
            std::array<char, 128> text{};
            text.fill('!');
            FjErrorBuffer error{text.data(), capacity, 99};
            status(prepare(nullptr, &out, &failure, &error), 2);
            require(error.length == capacity - 1 && text[capacity - 1] == '\0' && text[capacity] == '!', "bounded diagnostics include NUL and preserve canary");
        }
        status(fj_test_frame_geometry_fault(operation, 1));
        Output out{};
        FjFrameGeometryFailure failure{};
        char text[128]{};
        FjErrorBuffer error{text, sizeof(text), 0};
        status(prepare(&input, &out, &failure, &error), 8);
        require(failure.reason == 3 && std::string_view(text) == "frame geometry preparation panicked", "whole-operation contained frame panic");
        if constexpr (std::is_same_v<Output, FjExtent>)
            require(cleared(out), "panic domain clear");
        else
            require(cleared(out.full_frame) && std::bit_cast<std::uint32_t>(out.pixel_size_um) == 0, "panic geometry clear");
        status(prepare(&input, &out, &failure, nullptr));
    }

    void boundary() {
        const FjRect bounds{0, 0, 7, 5};
        const FjFrameGeometryInput input{{0, 0, 7, 5}, 36};
        for (auto prepare : {fj_legacy_frame_domain_prepare, fj_test_frame_domain_prepare})
            raw<FjRect, FjExtent>(bounds, prepare, 0);
        for (auto prepare : {fj_legacy_frame_geometry_prepare, fj_test_frame_geometry_prepare})
            raw<FjFrameGeometryInput, FjFrameGeometry>(input, prepare, 1);
        status(fj_test_frame_geometry_fault(2, 1), 2);
        status(fj_test_frame_geometry_fault(0, 2), 2);
        status(fj_test_frame_geometry_fault(0, 1));
        std::thread other([&] {
            FjExtent out{};
            FjFrameGeometryFailure failure{};
            status(fj_legacy_frame_domain_prepare(&bounds, &out, &failure, nullptr));
        });
        other.join();
        FjExtent out{};
        FjFrameGeometryFailure failure{};
        status(fj_legacy_frame_domain_prepare(&bounds, &out, &failure, nullptr), 8);
        status(fj_test_frame_geometry_fault(0, 0));
        status(fj_test_frame_geometry_fault(1, 0));
    }
} // namespace FrameGeometryTest
