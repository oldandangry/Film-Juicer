#include <algorithm>
#include <array>
#include <type_traits>
#include <barrier>
#include <bit>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "exposure_fixture.h"
#include "OutputColor.h"
#include "color_fixture.h"
#include "scanner_fixture.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Scanner.h"
#include "output_encoding_oracle.h"
#include "juicer_test_api.h"

extern "C" int output_color_abi_c(void);
extern "C" int scanner_color_abi_c(void);
namespace {
    using Json = nlohmann::json;
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    std::uint64_t read_fixture_word(const Json& value, const char* message) {
        require(value.is_number_unsigned(), message);
        return value.get<std::uint64_t>();
    }
    int read_fixture_key(const Json& value, int max, const char* message) {
        const auto word = read_fixture_word(value, message);
        require(std::cmp_less_equal(word, max), message);
        return static_cast<int>(word);
    }
    Json words(const float* values, std::size_t count) {
        Json out = Json::array();
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(std::bit_cast<std::uint32_t>(values[i]));
        return out;
    }
    void compare(const FjOutputColorPreparation& out, const Json& row) {
        require(out.output_space == row.at("tag") && out.entry_hash == row.at("hash"), "selected tag/entry identity");
        require(words(out.rgb_to_xyz, 9) == row.at("rgb_to_xyz") && words(out.xyz_to_rgb, 9) == row.at("xyz_to_rgb"), "native source matrix words");
        const auto& base = row.at("base");
        require(words(out.white_xyz, 3) == Json::array({base[8], base[9], base[10]}), "distinct native output white");
        const auto& encoding = out.encoding;
        const std::array coefficients{encoding.gamma, encoding.a, encoding.b, encoding.c, encoding.d, encoding.linear_cutoff};
        require(encoding.schema == 1 && encoding.cctf_kind == row.at("kind"), "complete encoding schema/kind");
        require(words(coefficients.data(), 6) == row.at("cctf") && words(encoding.dwg_to_output, 9) == row.at("dwg_to_output"), "native CCTF/DWG words");
    }
    Json scanner_color_result(const Scanner::ColorRuntime& runtime) {
        const auto& e = runtime.preparedEncoding;
        const std::array coefficients{e.gamma, e.a, e.b, e.c, e.d, e.linear_cutoff};
        return {{"cat02", words(runtime.cat02, 9)}, {"xyz_to_rgb", words(runtime.xyzToRgb, 9)}, {"illuminant_xyz", words(runtime.illuminantXYZ, 3)}, {"params", {OutputEncoding::toIndex(runtime.encoding.colorSpace), runtime.encoding.applyCctfEncoding, runtime.encoding.inputIsOutputSpace}}, {"encoding", {{"schema", e.schema}, {"kind", e.cctf_kind}, {"coefficients", words(coefficients.data(), 6)}, {"dwg_to_output", words(e.dwg_to_output, 9)}}}, {"active_gamut_recipe_hash", runtime.outputGamutRecipeHash}, {"hash", runtime.hash}};
    }
    Scanner::ColorRuntime scanner_color_case(const Json& input) {
        Scanner::ScannerIlluminant illuminant{};
        for (int c = 0; c < 3; ++c)
            illuminant.whiteXYZ[c] = std::bit_cast<float>(input.at("illuminant_xyz")[c].get<std::uint32_t>());
        illuminant.hash = input.at("illuminant_hash");
        return JuicerTestFixture::Scanner::build_color_runtime(static_cast<Scanner::ScannerMedium>(input.at("medium").get<int>()), illuminant, {static_cast<OutputEncoding::ColorSpace>(input.at("output_space").get<int>()), input.at("apply_cctf").get<bool>(), input.at("input_is_output_space").get<bool>()}, input.at("active_gamut_recipe_hash"));
    }
    int verify_scanner_color(const char* fixture) {
        std::ifstream stream{std::filesystem::path(fixture)};
        Json rows;
        stream >> rows;
        for (const auto& row : rows.at("cases")) {
            require(scanner_color_result(scanner_color_case(row.at("input"))) == row.at("expected"), "independent complete scanner supplement");
        }
        std::cout << "Complete scanner supplement passed\n";
        return 0;
    }
} // namespace
#include "scanner_color_test.inc"
// VS LLVM 22 reports the JSON parser's internal type_error despite both handlers below.
int main(int argc, char** argv) try { // NOLINT(bugprone-exception-escape): all fixture exceptions become a failing exit status.
    if (argc == 5 && std::string_view(argv[1]) == "scanner-color") {
        std::ifstream a{std::filesystem::path(argv[2])}, b{std::filesystem::path(argv[3])};
        Json supplement, catalogue;
        a >> supplement;
        b >> catalogue;
        const auto cat02 = load_exposure_qualified_fixture(argv[4]);
        ScannerColorTests::cold_first_access(supplement);
        (void)verify_scanner_color(argv[2]);
        ScannerColorTests::run({.supplement = supplement, .catalogue = catalogue, .cat02 = cat02});
        require(scanner_color_abi_c() == 0, "scanner C11 clearing/fault/width/layout");
        return 0;
    }
    require(argc == 2, "immutable native fixture required");
    // First access in this fresh process is concurrent across both actual exports.
    std::barrier barrier(16);
    std::array<FjOutputColorPreparation, 16> results{};
    std::array<std::uint32_t, 16> statuses{};
    std::vector<std::thread> threads;
    threads.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&, i] {
            barrier.arrive_and_wait();
            statuses[i] = (i % 2 ? fj_test_output_color : fj_test_boundary_output_color)(static_cast<std::uint32_t>(i % 9), &results[i]).category;
        });
    }
    for (auto& thread : threads)
        thread.join();
    std::ifstream input{std::filesystem::path(argv[1])};
    Json fixture;
    input >> fixture;
    for (std::size_t i = 0; i < results.size(); ++i) {
        require(statuses[i] == FJ_STATUS_SUCCESS, "concurrent first-access success");
        compare(results[i], fixture.at("entries")[i % 9]);
    }
    {
        require(output_color_abi_c() == 0, "C11 layouts/value/clearing/canary/consume-once faults");
    }
    for (int tag = 0; tag < 9; ++tag) {
        require(fixture.at("entries")[tag].at("name") == OutputEncoding::kColorSpaceLabels[tag], "native UI tag/label correspondence");
        const auto out = JuicerTestFixture::Color::output_color(static_cast<OutputEncoding::ColorSpace>(tag));
        compare(out, fixture.at("entries")[tag]);
        FjOutputColorPreparation direct{};
        require(fj_test_output_color(tag, &direct).category == FJ_STATUS_SUCCESS, "independent direct facade");
        compare(direct, fixture.at("entries")[tag]);
    }
    {
        compare(JuicerTestFixture::Color::output_color(static_cast<OutputEncoding::ColorSpace>(255)), fixture.at("entries")[0]);
    }
    constexpr std::array scannerTags{0, 1, 2, 3, 4, 5, 6, 7, 8, 255};
    std::set<std::tuple<int, int, int, std::uint64_t>> scannerRemaining;
    for (int medium : {0, 1})
        for (int tag : scannerTags)
            for (int encoded : {0, 1})
                for (std::uint64_t hash : {0, 1234567})
                    scannerRemaining.emplace(medium, tag, encoded, hash);
    for (const auto& row : fixture.at("scanner").get_ref<const Json::array_t&>()) {
        Scanner::ScannerIlluminant illuminant{};
        illuminant.whiteXYZ[0] = 0.950455f;
        illuminant.whiteXYZ[1] = 1.0f;
        illuminant.whiteXYZ[2] = 1.089058f;
        illuminant.hash = read_fixture_word(row.at("illuminant_hash"), "scanner fixture illuminant hash must be an unsigned integer");
        const auto tag = read_fixture_key(row.at("tag"), 255, "scanner fixture tag must be an unsigned integer in 0..255");
        const auto medium = read_fixture_key(row.at("medium"), 1, "scanner fixture medium must be an unsigned integer in 0..1");
        const auto encoded = read_fixture_key(row.at("encoded"), 1, "scanner fixture encoded flag must be an unsigned integer in 0..1");
        const auto key = std::tuple{medium, tag, encoded, illuminant.hash};
        require(scannerRemaining.contains(key), "unexpected or duplicate scanner fixture key");
        const auto runtime = JuicerTestFixture::Scanner::build_color_runtime(
            static_cast<Scanner::ScannerMedium>(medium), illuminant, {static_cast<OutputEncoding::ColorSpace>(tag), encoded != 0, false}, 7654321);
        require(words(runtime.cat02, 9) == row.at("cat02") && words(runtime.xyzToRgb, 9) == row.at("xyz_to_rgb"), "frozen scanner arrays and fallback");
        require(words(runtime.illuminantXYZ, 3) == row.at("illuminant_xyz") && runtime.hash == row.at("hash") && runtime.outputGamutRecipeHash == row.at("gamut_recipe_hash"), "frozen scanner white/identities");
        require(Json::array({OutputEncoding::toIndex(runtime.encoding.colorSpace), runtime.encoding.applyCctfEncoding ? 1 : 0, runtime.encoding.inputIsOutputSpace ? 1 : 0}) == row.at("params"), "original tag and forced scanner input space");
        if (illuminant.hash != 0) {
            const auto& encoding = runtime.preparedEncoding;
            const auto& entry = fixture.at("entries")[tag < 9 ? tag : 0];
            const std::array coefficients{encoding.gamma, encoding.a, encoding.b, encoding.c, encoding.d, encoding.linear_cutoff};
            require(encoding.schema == 1 && encoding.cctf_kind == entry.at("kind") && words(coefficients.data(), 6) == entry.at("cctf") && words(encoding.dwg_to_output, 9) == entry.at("dwg_to_output"), "scanner retains complete independent prepared words for both encoding states");
        }
        scannerRemaining.erase(key);
    }
    require(scannerRemaining.empty(), "missing scanner fixture comparisons");
    // Invalid illuminants still fail before either newly migrated preparation or CAT02.
    for (const auto operation : {FJ_TEST_OUTPUT_COLOR, FJ_TEST_CAT02_MATRIX}) {
        require(fj_test_color_arm_fault(operation, 1, FJ_TEST_COLOR_PANIC).category == FJ_STATUS_SUCCESS, "arm early-order witness");
        require(JuicerTestFixture::Scanner::build_color_runtime(Scanner::ScannerMedium::Negative, {}, {}, 0).hash == 0, "invalid illuminant returns before color work");
        FjOutputColorPreparation prepared{};
        const float white[3]{1.0f, 1.0f, 1.0f};
        float matrix[9]{};
        const auto status = operation == FJ_TEST_OUTPUT_COLOR ? fj_test_boundary_output_color(0, &prepared) : fj_test_boundary_cat02_matrix(white, white, matrix);
        require(status.category == FJ_STATUS_INTERNAL_FAILURE, "early failure preserves operation-specific fault");
        (void)fj_test_color_clear_fault();
    }
    // The frozen oracle crosses every tag and flag combination with these
    // input words at offsets 0, 3 and 7 around the same cyclic sequence.
    constexpr std::array<std::uint64_t, 28> oracleWords{
        0x8000000000000000,
        0x0000000000000000,
        0x3ff0000000000000,
        0xbff0000000000000,
        0x40effc0000000000,
        0xc0effc0000000000,
        0x36a0000000000000,
        0x3810000000000000,
        0x7ff8000000000000,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x3fe8353f80000000,
        0x3ffc305540000000,
        0x3f69a5c380000000,
        0x3f69a5c360000000,
        0x3f69a5c3a0000000,
        0x3f927cbd60000000,
        0x3f927cbd40000000,
        0x3f927cbd80000000,
        0x3f60000000000000,
        0x3f5fffffe0000000,
        0x3f60000020000000,
        0x3f657f1da0000000,
        0x3f657f1d80000000,
        0x3f657f1dc0000000,
        0x3ff0000000000000,
        0x3fefffffe0000000,
        0x3ff0000020000000,
    };
    std::set<std::tuple<int, int, int, std::array<std::uint64_t, 3>>> oracleRemaining;
    for (int tag = 0; tag < 9; ++tag)
        for (int encoded : {0, 1})
            for (int native : {0, 1})
                for (std::size_t i = 0; i < oracleWords.size(); ++i)
                    oracleRemaining.emplace(tag, encoded, native, std::array{oracleWords[i], oracleWords[(i + 3) % oracleWords.size()], oracleWords[(i + 7) % oracleWords.size()]});
    for (const auto& row : fixture.at("oracle").get_ref<const Json::array_t&>()) {
        const auto tag = read_fixture_key(row.at("tag"), 8, "encoding-oracle fixture tag must be an unsigned integer in 0..8");
        const auto encoded = read_fixture_key(row.at("encoded"), 1, "encoding-oracle fixture encoded flag must be an unsigned integer in 0..1");
        const auto native = read_fixture_key(row.at("native"), 1, "encoding-oracle fixture native flag must be an unsigned integer in 0..1");
        const auto& input = row.at("input");
        std::array<std::uint64_t, 3> inputWords{};
        require(input.is_array() && input.size() == inputWords.size(), "encoding-oracle fixture input must contain exactly three words");
        for (std::size_t c = 0; c < inputWords.size(); ++c)
            inputWords[c] = read_fixture_word(input[c], "encoding-oracle fixture input word must be an unsigned integer");
        const auto key = std::tuple{tag, encoded, native, inputWords};
        require(oracleRemaining.contains(key), "unexpected or duplicate encoding-oracle fixture key");
        double rgb[3];
        for (int c = 0; c < 3; ++c)
            rgb[c] = std::bit_cast<double>(inputWords[c]);
        ScannerOracle::apply_encoding({static_cast<OutputEncoding::ColorSpace>(tag), encoded != 0, native != 0}, rgb);
        for (int c = 0; c < 3; ++c) {
            const auto expected = std::bit_cast<double>(row.at("expected")[c].get<std::uint64_t>());
            require(std::isnan(expected) ? std::isnan(rgb[c]) : std::bit_cast<std::uint64_t>(rgb[c]) == std::bit_cast<std::uint64_t>(expected), "frozen native double oracle semantics");
        }
        oracleRemaining.erase(key);
    }
    require(oracleRemaining.empty(), "missing encoding-oracle fixture comparisons");
    // Earlier caller-owned values remain usable after every subsequent preparation.
    for (std::size_t i = 0; i < results.size(); ++i)
        compare(results[i], fixture.at("entries")[i % 9]);
    std::cout << "Output color: complete native values, ABI, first access, lifetime and faults passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Unknown output-color test failure\n";
    return 1;
}
