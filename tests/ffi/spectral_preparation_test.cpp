#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "Cuda/JuicerCudaExecutor.h"
#include "RustSpectralBridge.h"
#include "RustAssetBridge.h"
#include "spectral_allocation_probe.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ColorTransforms.h"
#include "juicer_test_api.h"
#include "exposure_fixture.h"
#include "juicer_cuda_owner.h"

extern "C" int fj_test_spectral_preparation_abi_c(void);
void fj_test_spectral_products(const nlohmann::json&, const std::filesystem::path&, const std::filesystem::path&, bool, void (*)(const nlohmann::json&));
void fj_test_spectral_native_allocation(const nlohmann::json&);
namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& why) {
        if (!condition)
            throw std::runtime_error(why);
    }
    void status(FjStatus actual, std::uint32_t expected) {
        require(actual.category == expected && actual.api == FJ_API_NONE && actual.native_code == 0, "spectral typed status " + std::to_string(actual.category) + " expected " + std::to_string(expected));
    }
    struct Diagnostic {
        std::array<char, 512> bytes{};
        FjErrorBuffer error{bytes.data(), bytes.size(), 0};
    };
    template <typename Range>
    Json bits(const Range& values) {
        Json out = Json::array();
        for (float v : values)
            out.push_back(std::bit_cast<std::uint32_t>(v));
        return out;
    }
    template <std::size_t N>
    std::array<float, N> decode(const Json& values) {
        require(values.size() == N, "fixed input size");
        std::array<float, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(values[i].get<std::uint32_t>());
        return out;
    }
    template <std::size_t N>
    FjFloatSpan span(const std::array<float, N>& values) {
        return {values.data(), N};
    }
    void exact(const Json& actual, const Json& expected, const std::string& why) {
        require(actual.size() == expected.size(), why + " count");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto a = actual[i].get<std::uint32_t>(), b = expected[i].get<std::uint32_t>();
            if (a != b)
                require(std::isnan(std::bit_cast<float>(a)) && std::isnan(std::bit_cast<float>(b)), why + " first difference at " + std::to_string(i) + ": " + std::to_string(a) + " expected " + std::to_string(b));
        }
    }
    struct Input {
        std::array<float, 243> dyes{};
        std::array<std::array<float, 81>, 3> observer{};
        std::array<float, 81> illuminant{}, minimum{}, midpoint{};
        FjSpectralInput view{};
        explicit Input(const Json& j) {
            for (std::size_t i = 0; i < 81; ++i) {
                auto row = decode<3>(j.at("dyes")[i]);
                std::copy(row.begin(), row.end(), dyes.data() + i * 3);
            }
            for (std::size_t c = 0; c < 3; ++c)
                observer[c] = decode<81>(j.at("observer")[c]);
            illuminant = decode<81>(j.at("illuminant"));
            view = {span(dyes), {span(observer[0]), span(observer[1]), span(observer[2])}, span(illuminant), {nullptr, 0}, {nullptr, 0}, j.at("illuminant_hash").get<std::uint64_t>()};
            if (!j.at("baseline_min").empty()) {
                minimum = decode<81>(j.at("baseline_min"));
                view.baseline_min = span(minimum);
            }
            if (!j.at("baseline_mid").empty()) {
                midpoint = decode<81>(j.at("baseline_mid"));
                view.baseline_mid = span(midpoint);
            }
        }
    };
    void compare_tables(const FjSpectralTables& t, const Json& j) {
        exact(bits(t.lambda_nm), j.at("lambda"), "lambda");
        exact(bits(t.illuminant), j.at("illum"), "illuminant");
        const char* obs[] = {"xbar", "ybar", "zbar"};
        const char* weighted[] = {"ax", "ay", "az"};
        const char* dyes[] = {"eps_c", "eps_m", "eps_y"};
        for (std::size_t c = 0; c < 3; ++c) {
            exact(bits(t.observer_xyz[c]), j.at(obs[c]), obs[c]);
            exact(bits(t.weighted_xyz[c]), j.at(weighted[c]), weighted[c]);
            exact(bits(t.dyes_cmy[c]), j.at(dyes[c]), dyes[c]);
        }
        exact(bits(t.baseline_min), j.at("base_density_min"), "baseline min");
        exact(bits(t.baseline_mid), j.at("base_density_mid"), "baseline mid");
        exact(bits(std::array{t.delta_lambda}), j.at("delta_lambda"), "delta");
        exact(bits(std::array{t.inv_yn}), j.at("inv_yn"), "invYn");
        exact(bits(t.white_xyz), j.at("white_xyz"), "table white");
        exact(bits(t.reference_white_xyz), j.at("ref_illum_white_xyz"), "reference white");
        require(t.has_baseline == (j.at("has_baseline").get<bool>() ? 1u : 0u) && t.illuminant_hash == j.at("illuminant_hash") && t.tables_hash == j.at("tables_hash"), "exact table flags/identities");
    }
    void tables(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("tables")) {
            Input in(row.at("input"));
            for (bool facade : {false, true}) {
                FjSpectralTables out{};
                Diagnostic d;
                status(facade ? fj_test_spectral_tables(&in.view, &out, &d.error) : fj_legacy_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
                compare_tables(out, row.at("expected"));
                require(d.error.length == 0, "success clears diagnostic");
            }
        }
        Input reference(fixture.at("leaf").at("tables")[0].at("input"));
        FjSpectralTables baseline{};
        Diagnostic d;
        status(fj_legacy_spectral_tables(&reference.view, &baseline, &d.error), FJ_STATUS_SUCCESS);
        for (const auto& row : fixture.at("dependencies")) {
            Input in(row.at("input"));
            FjSpectralTables out{};
            status(fj_legacy_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
            compare_tables(out, row.at("expected"));
            require(out.illuminant_hash == baseline.illuminant_hash && out.tables_hash != baseline.tables_hash, "isolated source dependency changes table family identity with supplied identity preserved");
        }
        require(fixture.at("leaf").at("bool_encoding") == Json::array({1, 0, 1}), "frozen bool byte encoding");
    }
    void whites(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("tables")) {
            Input in(row.at("input"));
            const FjSpectralWhiteInput input{in.view.observer, in.view.illuminant};
            for (bool facade : {false, true}) {
                FjSpectralWhite out{};
                FjSpectralWhiteFailure failure{};
                Diagnostic d;
                const bool built = row.at("white").at("built");
                status(facade ? fj_test_spectral_white(&input, &out, &failure, &d.error) : fj_legacy_spectral_white(&input, &out, &failure, &d.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built) {
                    const auto& w = row.at("white");
                    exact(bits(std::array{out.normalization}), w.at("normalization"), "white normalization");
                    exact(bits(out.white_xyz), w.at("xyz"), "white XYZ");
                    exact(bits(out.white_xy), w.at("xy"), "white xy");
                    require(out.hash == w.at("hash") && failure.reason == 0 && failure.luminance_sum == 0, "white identity/completion");
                } else {
                    require(out.hash == 0 && out.normalization == 0 && out.white_xyz[0] == 0 && out.white_xyz[1] == 0 && out.white_xyz[2] == 0 && out.white_xy[0] == 0 && out.white_xy[1] == 0, "failed white cleared");
                    require(failure.reason == row.at("white_failure").at("reason") && std::bit_cast<std::uint64_t>(failure.luminance_sum) == row.at("white_failure").at("luminance_sum"), "closed native white failure/sum");
                    require(d.error.length > 0, "computed white failure diagnostic");
                }
            }
        }
    }
    void inverses(const Json& fixture) {
        for (const auto& row : fixture.at("leaf").at("inverse")) {
            const auto x = decode<81>(row.at("input")[0]), y = decode<81>(row.at("input")[1]), z = decode<81>(row.at("input")[2]);
            const FjSpectralSInput input{span(x), span(y), span(z)};
            for (bool facade : {false, true}) {
                FjSpectralInverse out{};
                Diagnostic d;
                status(facade ? fj_test_spectral_s_inverse(&input, &out, &d.error) : fj_legacy_spectral_s_inverse(&input, &out, &d.error), FJ_STATUS_SUCCESS);
                exact(bits(out.matrix), row.at("expected"), "S inverse " + std::to_string(row.at("id").get<int>()));
            }
        }
    }
    void native_allocation(const Json& fixture) {
        Input in(fixture.at("leaf").at("tables")[0].at("input"));
        std::array<std::array<float, 3>, 81> dyes{};
        for (std::size_t i = 0; i < 81; ++i)
            std::copy_n(in.dyes.data() + i * 3, 3, dyes[i].data());
        Spectral::Curve illuminant;
        illuminant.linear.assign(in.illuminant.begin(), in.illuminant.end());
        Spectral::SpectralTables out;
        SpectralAllocationProbe::arm();
        bool rejected = false;
        try {
            JuicerSpectral::build_tables(dyes, in.minimum, illuminant, in.view.illuminant_hash, out);
        } catch (const std::bad_alloc&) {
            rejected = true;
        }
        SpectralAllocationProbe::clear();
        require(rejected && SpectralAllocationProbe::bytes() == 81 * sizeof(float) && out.tablesHash == 0, "real native projection allocation throws before result identity publication");
        JuicerSpectral::build_tables(dyes, in.minimum, illuminant, in.view.illuminant_hash, out);
        std::size_t capacity = 0;
        for (const auto* array : {&out.lambda, &out.illum, &out.Xbar, &out.Ybar, &out.Zbar, &out.Ax, &out.Ay, &out.Az, &out.epsC, &out.epsM, &out.epsY, &out.baseDensityMin, &out.baseDensityMid}) {
            require(array->size() == 81, "native table complete extent");
            capacity += array->capacity() * sizeof(float);
        }
        require(capacity >= std::size_t{13} * 81 * sizeof(float), "real native vector capacities");
        auto* const address = out.lambda.data();
        const auto hash = out.tablesHash;
        JuicerSpectral::build_tables(dyes, in.minimum, illuminant, in.view.illuminant_hash, out);
        require(out.lambda.data() == address && out.tablesHash == hash, "existing vector capacity can be reused without a derived cache");
        std::printf("Native spectral projection: 13 arrays, %zu capacity bytes; rejected %zu-byte allocation; fixed C table %zu bytes. These are capacities, not RSS or a live-peak claim.\n", capacity, SpectralAllocationProbe::bytes(), sizeof(FjSpectralTables));
    }
    void edge(const Json& fixture, const std::filesystem::path& resources) {
        require(fj_test_spectral_preparation_abi_c() == 1, "compiled C11 prototypes/layouts");
        Input in(fixture.at("leaf").at("tables")[0].at("input"));
        FjSpectralTables out{};
        Diagnostic d;
        auto malformed = in.view;
        for (auto count : {0u, 80u, 82u}) {
            malformed.illuminant.count = count;
            out.tables_hash = 123;
            status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(out.tables_hash == 0 && out.lambda_nm[0] == 0 && d.error.length > 0, "malformed tables cleared");
        }
        malformed = in.view;
        malformed.baseline_min = {nullptr, 0};
        malformed.baseline_mid = span(in.midpoint);
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        malformed = in.view;
        malformed.dyes_cmy.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(in.dyes.data()) + 1);
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        malformed = in.view;
        malformed.illuminant.count = std::numeric_limits<std::size_t>::max();
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        malformed = in.view;
        malformed.illuminant.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3);
        status(fj_legacy_spectral_tables(&malformed, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_spectral_tables(nullptr, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_spectral_tables(&in.view, nullptr, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_spectral_tables(&in.view, &out, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        const FjSpectralWhiteInput w{in.view.observer, in.view.illuminant};
        FjSpectralWhite white{};
        FjSpectralWhiteFailure failure{};
        const FjSpectralSInput s{in.view.observer.x, in.view.observer.y, in.view.observer.z};
        FjSpectralInverse inverse{};
        const auto call = [&](std::uint32_t op) {
            switch (op) {
                case 1:
                    return fj_legacy_spectral_tables(&in.view, &out, &d.error);
                case 2:
                    return fj_legacy_spectral_white(&w, &white, &failure, &d.error);
                default:
                    return fj_legacy_spectral_s_inverse(&s, &inverse, &d.error);
            }
        };
        for (std::uint32_t op = 1; op <= 3; ++op)
            for (std::uint32_t fault = 1; fault <= 4; ++fault) {
                status(fj_test_spectral_arm_fault(op, 2, fault), FJ_STATUS_SUCCESS);
                status(call(op), FJ_STATUS_SUCCESS);
                const std::uint32_t category[] = {0, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_INTERNAL_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE};
                status(call(op), category[fault]);
                require(out.tables_hash == 0 || op != 1, "fault tables clear");
                require((white.hash == 0 && failure.reason == 0 && failure.luminance_sum == 0) || op != 2, "fault white/meta clear");
                require(inverse.matrix[0] == 0 || op != 3, "fault inverse clear");
                status(call(op), FJ_STATUS_SUCCESS);
            }
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        fj_test_spectral_arm_facade_fault();
        status(call(1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        status(fj_test_spectral_tables(&in.view, &out, &d.error), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        std::thread other([&] {
            FjSpectralTables result{};
            Diagnostic diag;
            status(fj_legacy_spectral_tables(&in.view, &result, &diag.error), FJ_STATUS_SUCCESS);
        });
        other.join();
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_legacy_spectral_tables(nullptr, &out, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        status(call(1), FJ_STATUS_SUCCESS);
        struct CsvOwners {
            FjAssets* assets = nullptr;
            FjCsvPairs* pairs = nullptr;
            ~CsvOwners() {
                if (pairs && fj_legacy_csv_release(pairs, nullptr).category != FJ_STATUS_SUCCESS)
                    std::abort();
                if (assets && fj_legacy_assets_destroy(assets, nullptr).category != FJ_STATUS_SUCCESS)
                    std::abort();
            }
        } owners;
        const JuicerAssets::NativePathArgument path(resources);
        status(fj_legacy_assets_create(path.view(), &owners.assets, &d.error), FJ_STATUS_SUCCESS);
        status(fj_test_spectral_arm_fault(1, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_test_color_arm_fault(FJ_TEST_CAT02_MATRIX, 1, FJ_TEST_COLOR_UNSUPPORTED_INPUT), FJ_STATUS_SUCCESS);
        status(fj_test_illuminant_arm_fault(FJ_TEST_ILLUMINANT_EQUAL_ENERGY, 1, 1), FJ_STATUS_SUCCESS);
        fj_test_csv_fault(1);
        status(call(1), FJ_STATUS_UNSUPPORTED_INPUT);
        const std::array<float, 3> nominal{1.0f, 1.0f, 1.0f};
        std::array<float, 9> matrix{};
        status(fj_legacy_cat02_matrix(nominal.data(), nominal.data(), matrix.data()), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_cat02_matrix(nominal.data(), nominal.data(), matrix.data()), FJ_STATUS_SUCCESS);
        FjIlluminant light{};
        status(fj_legacy_illuminant_equal_energy(&light, &d.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_illuminant_equal_energy(&light, &d.error), FJ_STATUS_SUCCESS);
        status(fj_legacy_csv_acquire(owners.assets, FJ_CSV_D65, &owners.pairs, &d.error), FJ_STATUS_INTERNAL_FAILURE);
        require(owners.pairs == nullptr, "CSV fault publishes no owner independently of spectral math");
        status(fj_legacy_csv_acquire(owners.assets, FJ_CSV_D65, &owners.pairs, &d.error), FJ_STATUS_SUCCESS);
        status(call(1), FJ_STATUS_SUCCESS);
        fj_test_spectral_clear_fault();
    }
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 3 || argc == 5, "group, preset fixture and optional product resource/scratch required");
        const Json fixture = load_exposure_qualified_fixture(argv[2]);
        const std::string group = argv[1];
        if (group == "tables")
            tables(fixture);
        else if (group == "white")
            whites(fixture);
        else if (group == "inverse")
            inverses(fixture);
        else if (group == "edge" && argc == 5)
            edge(fixture, argv[3]);
        else if ((group == "products" || group == "admission") && argc == 5)
            fj_test_spectral_products(fixture, argv[3], argv[4], group == "admission", fj_test_spectral_native_allocation);
        else
            throw std::runtime_error("unknown spectral group");
        std::puts("PASS spectral frozen native values/contract");
        return 0;
    } catch (const JuicerCuda::ExecutionFailure& e) {
        std::fprintf(stderr, "spectral execution failure category=%u: %s\n", e.failure.status.category, e.failure.diagnostic.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

void fj_test_spectral_native_allocation(const nlohmann::json& fixture) {
    native_allocation(fixture);
}
