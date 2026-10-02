#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "juicer_test_api.h"

extern "C" const std::size_t* fj_test_profile_abi_c_facts(std::size_t* count);

#define FJ_ABI_TYPE(type, size, alignment)     \
    static_assert(sizeof(type) == size);       \
    static_assert(alignof(type) == alignment); \
    static_assert(std::is_standard_layout_v<type>);
#define FJ_ABI_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#define FJ_ABI_VALUE(tag, value) static_assert(tag == value);
#include "profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE

static_assert(std::is_same_v<decltype(&fj_test_film_profile_acquire), FjStatus (*)(FjStringView, FjStringView, FjFilmProfile**, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_film_profile_view), FjStatus (*)(const FjFilmProfile*, FjFilmProfileView*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_film_profile_release), FjStatus (*)(FjFilmProfile*, FjErrorBuffer*)>);
static_assert(std::is_same_v<decltype(&fj_test_profile_abi_facts), const std::size_t* (*)(std::size_t*)>);

namespace {

    std::filesystem::path scratchRoot;

    FjStringView text(const std::string& value) {
        return {value.data(), value.size()};
    }

    void expect_status(FjStatus status, std::uint32_t category) {
        EXPECT_EQ(status.category, category);
        EXPECT_EQ(status.api, FJ_API_NONE);
        EXPECT_EQ(status.native_code, 0);
    }

    struct ProfileInput {
        const char* metadata = "";
        const char* exposure = "[-1e40,-0.0,0.0,1e40]";
        const char* amplitudes = "[[1,2,3],[4,5,6],[7,8,9]]";
        bool include_data = true;
    };

    void write_profile(const std::filesystem::path& root, const std::string& key, const ProfileInput& input = {}) {
        std::filesystem::create_directories(root / "profiles");
        std::ofstream file(root / "profiles" / (key + ".json"));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << "{\"info\":{\"stock\":\"" << key << "\"" << input.metadata << "},\"data\":{";
        if (input.include_data) {
            file << "\"log_exposure\":" << input.exposure << ",\"density_curves_model\":{"
                 << "\"centers\":[[0,0,0],[0,0,0],[0,0,0]],\"sigmas\":[[1,1,1],[1,1,1],[1,1,1]],\"amplitudes\":"
                 << input.amplitudes << "},\"wavelengths\":[";
            for (int sample = 0; sample < 81; ++sample) {
                file << (sample == 0 ? "" : ",") << 380 + sample * 5;
            }
            file << "],\"log_sensitivity\":[";
            for (int sample = 0; sample < 81; ++sample) {
                file << (sample == 0 ? "" : ",") << "[0,0,0]";
            }
            file << "],\"channel_density\":[";
            for (int sample = 0; sample < 81; ++sample) {
                file << (sample == 0 ? "" : ",") << (sample == 0 ? "[null,-0.0,2]" : "[1,2,3]");
            }
            file << "],\"base_density\":[";
            for (int sample = 0; sample < 81; ++sample) {
                file << (sample == 0 ? "" : ",") << (sample == 0 ? "null" : "-0.0");
            }
            file << ']';
        }
        file << "}}\n";
        file.close();
    }

    std::filesystem::path catalog(const char* name) {
        const auto root = scratchRoot / name;
        write_profile(root, "kodak_portra_400");
        write_profile(root, "kodak_portra_endura", {.metadata = ",\"support\":\"paper\",\"stage\":\"printing\""});
        return root;
    }

    class FilmLease {
    public:
        FilmLease(const std::filesystem::path& root, const std::string& key) {
            const std::string path = root.generic_string();
            std::array<char, 256> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            const FjStatus status = fj_test_film_profile_acquire(text(path), text(key), &owner, &error);
            if (status.category != FJ_STATUS_SUCCESS || owner == nullptr) {
                fj_test_film_profile_release(std::exchange(owner, nullptr), nullptr);
                throw std::runtime_error(bytes.data());
            }
        }
        ~FilmLease() noexcept {
            const FjStatus result = fj_test_film_profile_release(std::exchange(owner, nullptr), nullptr);
            if (result.category != FJ_STATUS_SUCCESS || result.api != FJ_API_NONE || result.native_code != 0) {
                // Cleanup cannot allocate/format a GTest failure while unwinding.
                std::fprintf(stderr, "profile fixture release failed: category=%u api=%u code=%d\n", result.category, result.api, result.native_code);
                std::abort();
            }
        }
        FilmLease(const FilmLease&) = delete;
        FilmLease& operator=(const FilmLease&) = delete;
        FilmLease(FilmLease&&) = delete;
        FilmLease& operator=(FilmLease&&) = delete;

        FjFilmProfile* owner = nullptr;
    };

    void expect_empty(const FjFilmProfileView& view) {
        EXPECT_EQ(view.use, 0u);
        EXPECT_EQ(view.antihalation, 0u);
        EXPECT_EQ(view.asset_token, 0u);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            EXPECT_EQ(view.halation_first_sigma_um[channel], 0.0f);
            EXPECT_EQ(view.halation_primary_amount[channel], 0.0f);
        }
        EXPECT_EQ(view.source_log_exposure.data, nullptr);
        EXPECT_EQ(view.source_log_exposure.count, 0u);
        const auto check = [](FjFloatSpan span) {
            EXPECT_EQ(span.data, nullptr);
            EXPECT_EQ(span.count, 0u);
        };
        check(view.log_exposure);
        check(view.density_curves_cmy);
        check(view.channel_density_cmy);
        check(view.base_density);
        for (const auto& layer : view.density_curves_layers) {
            for (const auto& channel : layer) {
                check(channel);
            }
        }
    }

    TEST(ProfileOwner, CAndRustLayoutsAndSignaturesAgree) {
        const std::size_t cppFacts[] = {
#define FJ_ABI_TYPE(type, size, alignment) sizeof(type), alignof(type),
#define FJ_ABI_FIELD(type, field, offset) offsetof(type, field),
#define FJ_ABI_VALUE(tag, value) tag,
#include "profile_abi_facts.inc"
#undef FJ_ABI_TYPE
#undef FJ_ABI_FIELD
#undef FJ_ABI_VALUE
        };
        std::size_t cCount = 0;
        std::size_t rustCount = 0;
        const auto* cFacts = fj_test_profile_abi_c_facts(&cCount);
        const auto* rustFacts = fj_test_profile_abi_facts(&rustCount);
        ASSERT_EQ(cCount, std::size(cppFacts));
        ASSERT_EQ(rustCount, cCount);
        for (std::size_t index = 0; index < cCount; ++index) {
            EXPECT_EQ(cppFacts[index], cFacts[index]) << index;
            EXPECT_EQ(cFacts[index], rustFacts[index]) << index;
        }
        EXPECT_EQ(fj_test_profile_abi_facts(nullptr), nullptr);
        std::printf("C/C++/Rust agree on %zu profile ABI facts and four signatures.\n", cCount);
    }

    TEST(ProfileOwner, ExactProjectionOutlivesAssetsAndCopyOutlivesOwner) {
        const auto root = catalog("projection");
        FilmLease lease(root, "kodak_portra_400");
        FjFilmProfileView view{};
        expect_status(fj_test_film_profile_view(lease.owner, &view, nullptr), FJ_STATUS_SUCCESS);
        ASSERT_EQ(view.source_log_exposure.count, 4u);
        ASSERT_EQ(view.log_exposure.count, 4u);
        ASSERT_EQ(view.density_curves_cmy.count, 12u);
        ASSERT_EQ(view.channel_density_cmy.count, 243u);
        ASSERT_EQ(view.base_density.count, 81u);
        EXPECT_EQ(view.use, FJ_PROFILE_USE_STILL);
        EXPECT_EQ(view.antihalation, FJ_PROFILE_ANTIHALATION_WEAK);
        EXPECT_NE(view.asset_token, 0u);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(view.source_log_exposure.data[0]), std::bit_cast<std::uint64_t>(-1e40));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(view.source_log_exposure.data[3]), std::bit_cast<std::uint64_t>(1e40));
        EXPECT_TRUE(std::signbit(view.source_log_exposure.data[1]));
        EXPECT_FALSE(std::signbit(view.source_log_exposure.data[2]));
        EXPECT_EQ(view.log_exposure.data[0], -std::numeric_limits<float>::infinity());
        EXPECT_EQ(view.log_exposure.data[3], std::numeric_limits<float>::infinity());
        EXPECT_TRUE(std::signbit(view.log_exposure.data[1]));
        EXPECT_FALSE(std::signbit(view.log_exposure.data[2]));
        // Independent identities: erfc(+inf)=0, erfc(0)=1, erfc(-inf)=2.
        // Model amplitudes are [channel][layer]; projected curves transpose to
        // [layer][channel][exposure]. No candidate capture supplies expectations.
        for (std::size_t channel = 0; channel < 3; ++channel) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(view.halation_first_sigma_um[channel]), std::bit_cast<std::uint32_t>(65.0f));
            constexpr std::array<float, 3> kAmounts{0.08f, 0.02f, 0.0f};
            EXPECT_EQ(std::bit_cast<std::uint32_t>(view.halation_primary_amount[channel]), std::bit_cast<std::uint32_t>(kAmounts[channel]));
            for (std::size_t exposure = 0; exposure < 4; ++exposure) {
                const float factor = exposure == 0 ? 0.0f : (exposure == 3 ? 1.0f : 0.5f);
                const float sum = static_cast<float>(6 + channel * 9);
                EXPECT_EQ(view.density_curves_cmy.data[exposure * 3 + channel], sum * factor);
                for (std::size_t layer = 0; layer < 3; ++layer) {
                    const auto span = view.density_curves_layers[layer][channel];
                    ASSERT_EQ(span.count, 4u);
                    EXPECT_EQ(span.data[exposure], static_cast<float>(channel * 3 + layer + 1) * factor);
                }
            }
        }
        EXPECT_TRUE(std::isnan(view.channel_density_cmy.data[0]));
        EXPECT_TRUE(std::signbit(view.channel_density_cmy.data[1]));
        EXPECT_EQ(view.channel_density_cmy.data[2], 2.0f);
        EXPECT_EQ(view.channel_density_cmy.data[3], 1.0f);
        EXPECT_TRUE(std::isnan(view.base_density.data[0]));
        EXPECT_TRUE(std::signbit(view.base_density.data[1]));

        const std::vector<double> sourceCopy(view.source_log_exposure.data, view.source_log_exposure.data + view.source_log_exposure.count);
        const std::vector<float> totalsCopy(view.density_curves_cmy.data, view.density_curves_cmy.data + view.density_curves_cmy.count);
        FjFilmProfileView repeat{};
        expect_status(fj_test_film_profile_view(lease.owner, &repeat, nullptr), FJ_STATUS_SUCCESS);
        EXPECT_EQ(repeat.asset_token, view.asset_token);
        EXPECT_EQ(repeat.source_log_exposure.data, view.source_log_exposure.data);
        EXPECT_EQ(repeat.density_curves_cmy.data, view.density_curves_cmy.data);
        expect_status(fj_test_film_profile_release(std::exchange(lease.owner, nullptr), nullptr), FJ_STATUS_SUCCESS);
        // Only independent C++ copies are accessed after release.
        EXPECT_TRUE(std::signbit(sourceCopy[1]));
        EXPECT_EQ(totalsCopy[9], 6.0f);
        const auto copiedBytes = sourceCopy.size() * sizeof(double) + totalsCopy.size() * sizeof(float);
        EXPECT_EQ(copiedBytes, 80u);
        std::printf("Borrowed projection duplicates 0 profile-table bytes; view is %zu bytes including 24 digest bytes; local independent copies own %zu bytes.\n", sizeof(FjFilmProfileView), copiedBytes);
    }

    TEST(ProfileOwner, AllMetadataTagsAndIndependentHalationBits) {
        const auto root = catalog("metadata");
        constexpr std::array<const char*, 2> kUses{"still", "cine"};
        constexpr std::array<const char*, 3> kAntihalation{"strong", "weak", "no"};
        constexpr std::array<std::array<float, 3>, 3> kAmounts{{{0.015f, 0.005f, 0.0f}, {0.08f, 0.02f, 0.0f}, {0.3f, 0.1f, 0.015f}}};
        for (std::size_t use = 0; use < 2; ++use) {
            for (std::size_t antihalation = 0; antihalation < 3; ++antihalation) {
                const std::string key = std::string(kUses[use]) + "_" + kAntihalation[antihalation];
                const std::string metadata = ",\"use\":\"" + std::string(kUses[use]) + "\",\"antihalation\":\"" + kAntihalation[antihalation] + "\"";
                write_profile(root, key, {.metadata = metadata.c_str()});
                FilmLease lease(root, key);
                FjFilmProfileView view{};
                expect_status(fj_test_film_profile_view(lease.owner, &view, nullptr), FJ_STATUS_SUCCESS);
                EXPECT_EQ(view.use, use);
                EXPECT_EQ(view.antihalation, antihalation);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(view.halation_first_sigma_um[channel]), std::bit_cast<std::uint32_t>(use == 0 ? 65.0f : 50.0f));
                    EXPECT_EQ(std::bit_cast<std::uint32_t>(view.halation_primary_amount[channel]), std::bit_cast<std::uint32_t>(kAmounts[antihalation][channel]));
                }
            }
        }
    }

    TEST(ProfileOwner, FailureOutputsAndBoundedDiagnostics) {
        const auto root = catalog("failures");
        const std::string path = root.generic_string();
        const std::string key = "missing";
        // Seed output with a distinct live owner, without transferring that hold.
        FilmLease retained(root, "kodak_portra_400");
        FjFilmProfile* output = retained.owner;
        std::array<char, 8> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 99};
        expect_status(fj_test_film_profile_acquire(text(path), text(key), &output, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(output, nullptr);
        EXPECT_EQ(error.length, 7u);
        EXPECT_EQ(bytes[7], '\0');
        error = {nullptr, 0, 99};
        expect_status(fj_test_film_profile_acquire(text(path), text(key), &output, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(error.length, 0u);
        error = {nullptr, 1, 99};
        output = retained.owner;
        expect_status(fj_test_film_profile_acquire(text(path), text(key), &output, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(output, nullptr);
        EXPECT_EQ(error.length, 0u);
        FjFilmProfileView view{};
        expect_status(fj_test_film_profile_view(retained.owner, &view, nullptr), FJ_STATUS_SUCCESS);
        expect_status(fj_test_film_profile_view(nullptr, &view, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(view);
        expect_status(fj_test_film_profile_view(retained.owner, &view, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_empty(view);
        expect_status(fj_test_film_profile_view(retained.owner, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_film_profile_acquire(text(path), text(key), nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const std::array<FjStringView, 5> invalid{{{nullptr, 0}, {nullptr, 1}, {path.data(), std::numeric_limits<std::size_t>::max()}, {"\xff", 1}, {"a\0b", 3}}};
        for (const auto input : invalid) {
            expect_status(fj_test_film_profile_acquire(input, text(key), &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(output, nullptr);
            expect_status(fj_test_film_profile_acquire(text(path), input, &output, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
            EXPECT_EQ(output, nullptr);
        }
        bytes.fill('x');
        error = {bytes.data(), bytes.size(), 99};
        expect_status(fj_test_film_profile_view(retained.owner, &view, &error), FJ_STATUS_SUCCESS);
        EXPECT_EQ(error.length, 0u);
        EXPECT_EQ(bytes[0], '\0');
        char terminator = 'x';
        error = {&terminator, 1, 99};
        expect_status(fj_test_film_profile_acquire(text(path), text(key), &output, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(error.length, 0u);
        EXPECT_EQ(terminator, '\0');
    }

    TEST(ProfileOwner, TypedResourceAndCompletionFailuresPublishNoOwner) {
        const auto root = catalog("resource-failures");
        write_profile(root, "decode", {.include_data = false});
        write_profile(root, "descending", {.exposure = "[1,0]"});
        write_profile(root, "computed", {.exposure = "[0]", .amplitudes = "[[1e40,2,3],[4,5,6],[7,8,9]]"});
        write_profile(root, "unsupported", {.metadata = ",\"use\":\"unsupported\""});
        for (const char* key : {"decode", "descending", "computed", "unsupported"}) {
            const std::string path = root.generic_string();
            FjFilmProfile* owner = nullptr;
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            expect_status(fj_test_film_profile_acquire(text(path), text(key), &owner, &error), FJ_STATUS_PREPARATION_FAILURE);
            EXPECT_EQ(owner, nullptr);
            EXPECT_GT(error.length, 0u);
            EXPECT_NE(std::string(bytes.data()).find(key), std::string::npos);
        }
        const std::string missingRoot = (scratchRoot / "no-catalog").generic_string();
        FjFilmProfile* owner = nullptr;
        expect_status(fj_test_film_profile_acquire(text(missingRoot), text("kodak_portra_400"), &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        const auto incompleteRoot = scratchRoot / "missing-print-default";
        write_profile(incompleteRoot, "kodak_portra_400");
        const std::string incompletePath = incompleteRoot.generic_string();
        expect_status(fj_test_film_profile_acquire(text(incompletePath), text("kodak_portra_400"), &owner, nullptr), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
    }

    TEST(ProfileOwner, WritableDiagnosticOutputsNeedNoPriorInitialization) {
        const auto root = catalog("uninitialized-output");
        const std::string path = root.generic_string();
        char bytes[8];
        FjErrorBuffer error;
        error.data = bytes;
        error.capacity = sizeof(bytes);
        FjFilmProfile* owner = nullptr;
        expect_status(fj_test_film_profile_acquire(text(path), text("missing"), &owner, &error), FJ_STATUS_PREPARATION_FAILURE);
        EXPECT_EQ(owner, nullptr);
        EXPECT_EQ(error.length, 7u);
        EXPECT_EQ(bytes[7], '\0');
        EXPECT_EQ(std::string(bytes), "missing");
        bytes[0] = 'x';
        error.capacity = std::numeric_limits<std::size_t>::max();
        expect_status(fj_test_film_profile_acquire(text(path), text("missing"), &owner, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(error.length, 0u);
        EXPECT_EQ(bytes[0], 'x');
    }

    TEST(ProfileOwner, ConcurrentReadsUseDisjointOutputsAndExcludeRelease) {
        const auto root = catalog("concurrent-read");
        FilmLease lease(root, "kodak_portra_400");
        std::array<FjStatus, 4> statuses{};
        std::array<FjFilmProfileView, 4> views{};
        {
            std::array<std::jthread, 4> readers;
            for (std::size_t index = 0; index < readers.size(); ++index) {
                readers[index] = std::jthread([&, index] {
                    statuses[index] = fj_test_film_profile_view(lease.owner, &views[index], nullptr);
                });
            }
        }
        // jthread joins before this scope reads outputs or the lease releases.
        for (std::size_t index = 0; index < views.size(); ++index) {
            expect_status(statuses[index], FJ_STATUS_SUCCESS);
            EXPECT_EQ(views[index].asset_token, views[0].asset_token);
            EXPECT_EQ(views[index].log_exposure.data, views[0].log_exposure.data);
            EXPECT_EQ(views[index].density_curves_cmy.data, views[0].density_curves_cmy.data);
            EXPECT_EQ(views[index].density_curves_cmy.count, 12u);
        }
    }

    TEST(ProfileOwner, ReleaseConsumesWithMalformedDiagnostic) {
        const auto root = catalog("release");
        FilmLease lease(root, "kodak_portra_400");
        FjErrorBuffer error{nullptr, 1, 99};
        expect_status(fj_test_film_profile_release(std::exchange(lease.owner, nullptr), &error), FJ_STATUS_UNSUPPORTED_INPUT);
        EXPECT_EQ(error.length, 0u);
        expect_status(fj_test_film_profile_release(nullptr, &error), FJ_STATUS_UNSUPPORTED_INPUT);
        expect_status(fj_test_film_profile_release(nullptr, nullptr), FJ_STATUS_SUCCESS);
        // Reclamation itself is checked with Weak in Rust.Bridge, not by probing
        // the consumed handle or dereferencing an expired view.
    }
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("profile owner test requires its isolated artifact root");
        }
        scratchRoot = std::filesystem::path(argv[1]);
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "profile owner fixture: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "profile owner fixture: unknown exception\n");
        return 1;
    }
}
