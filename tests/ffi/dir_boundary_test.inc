namespace DirBoundaryTest {
    struct Fixture {
        const Json& native;
        const Json& checked;
    };
    Json words64(const double* values, std::size_t count) {
        Json result = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            result.push_back(std::bit_cast<std::uint64_t>(values[i]));
        return result;
    }
    Json metadata(const FjDirBoundaryFailure& value) {
        return Json::array({value.reason, value.axis, value.row, value.length, std::bit_cast<std::uint32_t>(value.sigma_px), std::bit_cast<std::uint64_t>(value.sum), std::bit_cast<std::uint64_t>(value.tolerance)});
    }
    FjDirBoundaryInput input(const Json& value) {
        return JuicerDirBoundary::project_input(JuicerCuda::DirBoundaryTest::component(value.at("component")), {value.at("width").get<int>(), value.at("height").get<int>()});
    }
    void compare(const FjDirBoundaryView& view, const Json& expected) {
        const auto& horizontal = expected.at("horizontal");
        const auto& vertical = expected.at("vertical");
        require(view.width == expected.at("input").at("width") && view.height == expected.at("input").at("height"), "boundary dimensions");
        require(view.identity == expected.at("key"), "boundary key");
        require(view.horizontal_length == horizontal.at("length") && view.vertical_length == vertical.at("length"), "boundary retained lengths");
        require(view.horizontal_terminal_size == horizontal.at("terminal_size") && view.vertical_terminal_size == vertical.at("terminal_size"), "boundary terminal sizes");
        Json weights = horizontal.at("weights");
        for (const auto& word : vertical.at("weights"))
            weights.push_back(word);
        require(view.count == weights.size() && view.weights && words64(view.weights, view.count) == weights, "entire combined boundary weights");
        require(words64(view.horizontal_terminal, 9) == horizontal.at("terminal") && words64(view.vertical_terminal, 9) == vertical.at("terminal"), "all packed terminal words including positive-zero suffix");
    }
    bool empty(const FjDirBoundaryView& view) {
        return !view.weights && !view.count && !view.identity && !view.width && !view.height && !view.horizontal_length && !view.vertical_length && !view.horizontal_terminal_size && !view.vertical_terminal_size &&
               words64(view.horizontal_terminal, 9) == Json::array({0, 0, 0, 0, 0, 0, 0, 0, 0}) && words64(view.vertical_terminal, 9) == Json::array({0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    FjDirBoundaryView dirty_view() {
        FjDirBoundaryView view{reinterpret_cast<const double*>(std::uintptr_t{1}), 1, 1, 1, 1, 1, 1, 1, 1, {}, {}};
        std::fill_n(view.horizontal_terminal, 9, 1.0);
        std::fill_n(view.vertical_terminal, 9, 1.0);
        return view;
    }
    void raw_values(Fixture fixture) {
        const auto& captured = fixture.native;
        const auto& checked = fixture.checked;
        require(captured.at("cases").size() == 114, "nonempty exact boundary membership");
        for (unsigned useC = 0; useC < 2; ++useC) {
            const auto describe = useC ? fj_test_dir_boundary_c_describe : fj_test_dir_boundary_describe;
            const auto acquire = useC ? fj_test_dir_boundary_c_acquire : fj_test_dir_boundary_acquire;
            const auto getView = useC ? fj_test_dir_boundary_c_view : fj_test_dir_boundary_view;
            const auto release = useC ? fj_test_dir_boundary_c_release : fj_test_dir_boundary_release;
            for (const auto& row : captured.at("cases")) {
                const auto raw = input(row.at("input"));
                const auto& expected = checked.at("ordinary").at(row.at("input").at("name").get<std::string>());
                FjDirBoundaryFailure failure{};
                std::array<char, 256> text{};
                FjErrorBuffer error{text.data(), text.size(), 777};
                std::uint64_t key = 777;
                const bool outer = expected.at(0) != FJ_DIR_BOUNDARY_REQUEST;
                status(describe(&raw, &key, &failure, &error), outer ? 0u : 3u);
                require(key == row.at("key") && metadata(failure) == (outer ? Json::array({0, 0, 3, 0, 0, 0, 0}) : expected), "describe admits only outer request before key/warm hit");
                FjDirBoundaryOwner* owner = reinterpret_cast<FjDirBoundaryOwner*>(std::uintptr_t{1});
                status(acquire(&raw, &owner, &failure, &error), row.at("success").get<bool>() ? 0u : 3u);
                require(metadata(failure) == expected, "typed native failure metadata");
                require(std::string_view(text.data(), error.length) == checked.at("messages").at(failure.reason).get<std::string>(), "exact base boundary diagnostic");
                if (row.at("success").get<bool>()) {
                    require(owner, "complete boundary owner");
                    FjDirBoundaryView view = dirty_view();
                    status(getView(owner, &view, &failure, &error));
                    compare(view, row);
                    const auto* weights = view.weights;
                    const auto copy = words64(weights, view.count);
                    status(getView(owner, &view, &failure, nullptr));
                    require(view.weights == weights && copy == words64(view.weights, view.count), "stable repeated view and NULL diagnostics");
                    status(release(owner, nullptr));
                } else {
                    require(!owner, "failed acquire clears owner");
                }
            }
        }
        require(fj_test_dir_boundary_count(3) == 0, "no leaked complete owners");
    }
    void products(const Json& captured) {
        static_assert(!std::is_copy_constructible_v<JuicerDirBoundary::Boundary>);
        for (const auto& row : captured.at("cases")) {
            JuicerCuda::Failure failure;
            const auto raw = input(row.at("input"));
            std::uint64_t key = 777;
            if (!JuicerDirBoundary::describe(raw, key, failure)) {
                require(!row.at("success").get<bool>() && failure.diagnostic == row.at("diagnostic").get<std::string>(), "native outer adapter failure");
                continue;
            }
            require(key == row.at("key"), "native adapter key");
            const auto owner = JuicerDirBoundary::Boundary::acquire(raw, key, failure);
            require(bool(owner) == row.at("success").get<bool>(), "native complete owner admission");
            if (owner)
                compare(owner->view(), row);
            else
                require(failure.diagnostic == row.at("diagnostic").get<std::string>(), "original native numerical diagnostic including stream formatting");
        }
        const auto& row = captured.at("cases").at(7);
        for (unsigned fault = 9; fault <= 12; ++fault) {
            const auto releases = fj_test_dir_boundary_count(2);
            status(fj_test_dir_boundary_fault(2, fault));
            JuicerCuda::Failure failure;
            require(!JuicerDirBoundary::Boundary::acquire(input(row.at("input")), row.at("key"), failure), "malformed view rejected");
            require(failure.status.category == 3 && failure.status.api == 0 && failure.status.native_code == 0 && failure.diagnostic == "spatial DIR reflected-boundary view is invalid", "native invalid-view envelope");
            require(fj_test_dir_boundary_count(2) == releases + 1 && fj_test_dir_boundary_count(3) == 0, "exactly one release on native rejection");
        }
    }
    void boundary(const Json& captured) {
        const auto raw = input(captured.at("cases").at(7).at("input"));
        const Json defaults = Json::array({0, 0, 3, 0, 0, 0, 0});
        FjDirBoundaryFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 99};
        std::uint64_t key = 999;
        FjDirBoundaryOwner* owner = reinterpret_cast<FjDirBoundaryOwner*>(std::uintptr_t{1});
        status(fj_test_dir_boundary_describe(nullptr, &key, &failure, &error), 2);
        require(key == 0 && failure.reason == 1, "cleared key on raw failure");
        status(fj_test_dir_boundary_acquire(nullptr, &owner, &failure, &error), 2);
        require(!owner && failure.reason == 1, "cleared owner on raw failure");
        auto view = dirty_view();
        status(fj_test_dir_boundary_view(nullptr, &view, &failure, &error), 2);
        require(empty(view) && failure.reason == 1, "cleared view on raw failure");
        status(fj_test_dir_boundary_describe(&raw, nullptr, &failure, &error), 2);
        require(failure.reason == 1 && failure.row == 3, "independent failure peer");
        status(fj_test_dir_boundary_acquire(&raw, &owner, nullptr, &error), 2);
        require(!owner, "independent owner peer");
        alignas(16) std::array<std::byte, sizeof(FjDirBoundaryInput) + 16> misaligned{};
        status(fj_test_dir_boundary_acquire(reinterpret_cast<const FjDirBoundaryInput*>(misaligned.data() + 1), &owner, &failure, &error), 2);
        require(!owner && failure.reason == 1, "misaligned input admission");
        const auto overflow = std::numeric_limits<std::uintptr_t>::max() - 7;
        status(fj_test_dir_boundary_describe(reinterpret_cast<const FjDirBoundaryInput*>(overflow), &key, &failure, &error), 2);
        require(key == 0 && failure.reason == 1, "overflowing record admission without dereference");
        const std::string message = "spatial DIR reflected-boundary arguments are invalid";
        for (const std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}, message.size(), message.size() + 1}) {
            text.fill('x');
            FjErrorBuffer bounded{capacity ? text.data() : reinterpret_cast<char*>(std::uintptr_t{1}), capacity, 999};
            owner = reinterpret_cast<FjDirBoundaryOwner*>(std::uintptr_t{1});
            status(fj_test_dir_boundary_acquire(nullptr, &owner, &failure, &bounded), 2);
            const auto written = capacity ? std::min(message.size(), capacity - 1) : 0;
            require(!owner && failure.reason == 1 && failure.axis == 0 && failure.row == 3 && failure.length == 0, "truncation preserves typed failure");
            require(bounded.length == written, "R1 written diagnostic bytes exclude NUL");
            if (capacity)
                require(text[written] == '\0' && std::string_view(text.data(), written) == std::string_view(message).substr(0, written) && text[capacity] == 'x', "bounded diagnostic NUL and guard byte");
            else
                require(text[0] == 'x', "zero capacity ignores backing");
        }
        FjErrorBuffer malformed{nullptr, 1, 999};
        status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, &malformed), 2);
        require(!owner && failure.reason == 1 && malformed.length == 0, "malformed diagnostic admission clears peers");
        malformed = {text.data(), std::numeric_limits<std::size_t>::max(), 999};
        status(fj_test_dir_boundary_describe(&raw, &key, &failure, &malformed), 2);
        require(!key && failure.reason == 1 && malformed.length == 0, "oversized diagnostic span");
        malformed = {reinterpret_cast<char*>(overflow), 16, 999};
        status(fj_test_dir_boundary_describe(&raw, &key, &failure, &malformed), 2);
        require(!key && failure.reason == 1 && malformed.length == 0, "overflowing diagnostic span");
        for (unsigned operation = 0; operation < 4; ++operation) {
            status(fj_test_dir_boundary_fault(operation, 1));
            if (operation == 0) {
                status(fj_test_dir_boundary_describe(&raw, &key, &failure, &error), 8);
                require(!key && failure.reason == 13, "describe panic clearing");
            } else if (operation == 1) {
                status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, &error), 8);
                require(!owner && failure.reason == 13, "acquire panic clearing");
            } else {
                status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, nullptr));
                if (operation == 2) {
                    view = dirty_view();
                    status(fj_test_dir_boundary_view(owner, &view, &failure, &error), 8);
                    require(empty(view) && failure.reason == 13, "view panic clearing");
                    status(fj_test_dir_boundary_release(owner, nullptr));
                } else {
                    status(fj_test_dir_boundary_release(owner, &error), 8);
                    require(std::string_view(text.data(), error.length) == "spatial DIR reflected-boundary release panicked", "release panic diagnostic");
                }
            }
            require(fj_test_dir_boundary_count(3) == 0, "panic leaves no escaped owner");
        }
        for (unsigned fault = 2; fault <= 8; ++fault) {
            status(fj_test_dir_boundary_fault(1, fault));
            status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, &error), 4);
            require(!owner && failure.reason == 12 && failure.axis == (fault <= 4 ? 1u : fault <= 7 ? 2u
                                                                                                    : 0u),
                    "seven bounded fallible reservations");
            require(failure.row == 3 && failure.length == (failure.axis == 1 ? raw.width : failure.axis == 2 ? raw.height
                                                                                                             : 0),
                    "reservation axis metadata");
            status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, nullptr));
            status(fj_test_dir_boundary_release(owner, nullptr));
        }
        for (unsigned operation = 0; operation < 4; ++operation) {
            status(fj_test_dir_boundary_fault(operation, 1));
            status(fj_test_dir_boundary_fault(operation, 99), 2);
            if (operation == 0)
                status(fj_test_dir_boundary_describe(nullptr, &key, &failure, nullptr), 2);
            else if (operation == 1)
                status(fj_test_dir_boundary_acquire(nullptr, &owner, &failure, nullptr), 2);
            else if (operation == 2)
                status(fj_test_dir_boundary_view(nullptr, &view, &failure, nullptr), 2);
            else {
                malformed = {nullptr, 1, 999};
                status(fj_test_dir_boundary_release(nullptr, &malformed), 2);
            }
            status(fj_test_dir_boundary_describe(&raw, &key, &failure, nullptr));
            status(fj_test_dir_boundary_acquire(&raw, &owner, &failure, nullptr));
            status(fj_test_dir_boundary_view(owner, &view, &failure, nullptr));
            status(fj_test_dir_boundary_release(owner, nullptr));
        }
        FjDirBoundaryOwner* independent = nullptr;
        {
            auto expired = raw;
            status(fj_test_dir_boundary_acquire(&expired, &owner, &failure, nullptr));
            status(fj_test_dir_boundary_acquire(&expired, &independent, &failure, nullptr));
        }
        status(fj_test_dir_boundary_view(owner, &view, &failure, nullptr));
        compare(view, captured.at("cases").at(7));
        auto second = dirty_view();
        status(fj_test_dir_boundary_view(independent, &second, &failure, nullptr));
        require(owner != independent && view.weights != second.weights && metadata(failure) == defaults, "independent owners survive caller input expiry");
        malformed = {nullptr, 1, 999};
        const auto released = fj_test_dir_boundary_count(2);
        status(fj_test_dir_boundary_release(owner, &malformed), 2);
        require(fj_test_dir_boundary_count(3) == 1 && fj_test_dir_boundary_count(2) == released + 1, "malformed diagnostic consumes exactly one owner");
        status(fj_test_dir_boundary_fault(3, 1));
        status(fj_test_dir_boundary_release(independent, &malformed), 2);
        require(fj_test_dir_boundary_count(3) == 0, "malformed diagnostic still consumes owner with pending release panic");
        status(fj_test_dir_boundary_release(nullptr, nullptr));
        auto oversized = raw;
        oversized.width = std::numeric_limits<int>::max() / 2 + 1;
        status(fj_test_dir_boundary_acquire(&oversized, &owner, &failure, nullptr), 3);
        require(!owner && failure.reason == 3 && failure.axis == 1, "checked horizontal period overflow");
        oversized = raw;
        oversized.height = std::numeric_limits<int>::max() / 2 + 1;
        status(fj_test_dir_boundary_acquire(&oversized, &owner, &failure, nullptr), 3);
        require(!owner && failure.reason == 3 && failure.axis == 2, "checked vertical overflow after horizontal product");
        oversized.component.boundary_derivation_version = 0;
        status(fj_test_dir_boundary_acquire(&oversized, &owner, &failure, nullptr), 3);
        require(!owner && failure.reason == 4 && failure.axis == 1, "horizontal policy precedes later axis size");
        require(fj_test_dir_boundary_count(3) == 0, "boundary lifetime closure");
    }
} // namespace DirBoundaryTest
