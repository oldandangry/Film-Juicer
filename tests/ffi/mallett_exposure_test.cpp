#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ColorTransforms.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "RustExposureBridge.h"
#include "RustAssetBridge.h"
#include "juicer_test_api.h"

#define FJ_MALLETT_SIZE(type, size, alignment) static_assert(sizeof(type) == size && alignof(type) == alignment);
#define FJ_MALLETT_FIELD(type, field, offset) static_assert(offsetof(type, field) == offset);
#include "mallett_exposure_abi_facts.inc"
#undef FJ_MALLETT_SIZE
#undef FJ_MALLETT_FIELD
extern "C" int fj_test_mallett_exposure_abi_c(const FjMallettMidgrayInput*, FjMallettMidgray*, float, FjMidgrayNormalization*, float, float*, const FjMallettReferenceInput*, FjReferenceRaw*, FjErrorBuffer*);

extern "C" int fj_test_spectrum_abi_c(const FjHanatosSpectrumInput*, FjSpectrumFixture*, const FjTablesSpectrumInput*, FjSpectrumFixture*, const FjMallettRawInput*, float*, FjErrorBuffer*);

void fj_test_mallett_callers(const std::filesystem::path&, const nlohmann::json&, const std::filesystem::path&, bool);

namespace {
    using Json = nlohmann::json;
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void status(FjStatus value, std::uint32_t category) {
        require(value.category == category && value.api == FJ_API_NONE && value.native_code == 0, "fixed exposure status " + std::to_string(value.category) + " expected " + std::to_string(category));
    }
    template <std::size_t N>
    std::array<float, N> decode(const Json& row) {
        require(row.size() == N, "fixed fixture extent");
        std::array<float, N> out{};
        for (std::size_t i = 0; i < N; ++i)
            out[i] = std::bit_cast<float>(row[i].get<std::uint32_t>());
        return out;
    }
    template <std::size_t N>
    void exact(const float* actual, const Json& expected) {
        require(expected.size() == N, "fixed expected extent");
        for (std::size_t i = 0; i < N; ++i) {
            const auto word = expected[i].get<std::uint32_t>();
            require(std::bit_cast<std::uint32_t>(actual[i]) == word || (std::isnan(actual[i]) && std::isnan(std::bit_cast<float>(word))), "fixed exposure bit mismatch at " + std::to_string(i));
        }
    }
    struct Diagnostic {
        std::array<char, 514> bytes{};
        FjErrorBuffer error{bytes.data() + 1, 512, 0};
        Diagnostic() {
            bytes.front() = 'L';
            bytes.back() = 'R';
        }
        void guards() const {
            require(bytes.front() == 'L' && bytes.back() == 'R', "diagnostic canaries");
        }
    };
    struct Operands {
        std::array<std::array<float, 3>, 81> basis{}, sensitivity{};
        std::array<float, 81> illuminant{};
        explicit Operands(const Json& row) {
            for (std::size_t i = 0; i < 81; ++i) {
                basis[i] = decode<3>(row.at("basis")[i]);
                sensitivity[i] = decode<3>(row.at("sensitivity")[i]);
            }
            illuminant = decode<81>(row.at("illuminant"));
        }
        FjFloatSpan basis_span() const {
            return {basis.front().data(), 243};
        }
        FjFloatSpan sensitivity_span() const {
            return {sensitivity.front().data(), 243};
        }
        FjFloatSpan illuminant_span() const {
            return {illuminant.data(), 81};
        }
        Spectral::MallettBasis native_basis() const {
            Spectral::MallettBasis result{81, 3, {}};
            for (const auto& rgb : basis)
                result.data.insert(result.data.end(), rgb.begin(), rgb.end());
            return result;
        }
    };
    struct FocusedInput : Operands {
        FjMallettMidgrayInput input{};
        explicit FocusedInput(const Json& row) : Operands(row.at("input")) {
            const auto& value = row.at("input");
            input.color.input_space = value.at("space").get<std::uint32_t>();
            input.color.decode_cctf = value.at("decode").get<bool>() ? 1u : 0u;
            input.color.adapt_xyz = value.at("adapt").get<bool>() ? 1u : 0u;
            const auto rgb = decode<9>(value.at("rgb_to_xyz")), adapt = decode<9>(value.at("xyz_adapt")), final = decode<9>(value.at("xyz_to_srgb"));
            std::copy(rgb.begin(), rgb.end(), input.color.rgb_to_xyz);
            std::copy(adapt.begin(), adapt.end(), input.color.xyz_adapt);
            std::copy(final.begin(), final.end(), input.xyz_to_linear_srgb);
            input.basis_rgb = basis_span();
            input.illuminant = illuminant_span();
            input.sensitivity_rgb = sensitivity_span();
        }
        Spectral::FilmRawConfig config() const {
            Spectral::FilmRawConfig result;
            result.inputColorSpace = static_cast<Spectral::InputColorSpace>(input.color.input_space);
            result.applyCctfDecoding = input.color.decode_cctf != 0;
            result.applyInputChromaticAdapt = input.color.adapt_xyz != 0;
            std::copy_n(input.color.rgb_to_xyz, 9, result.inputRGBToXYZ.m);
            std::copy_n(input.color.xyz_adapt, 9, result.inputXYZAdapt.m);
            std::copy_n(input.xyz_to_linear_srgb, 9, result.xyzToLinearSrgb.m);
            return result;
        }
    };
    struct ReferenceInput : Operands {
        FjMallettReferenceInput input{};
        explicit ReferenceInput(const Json& row) : Operands(row) {
            const auto source = decode<2>(row.at("source_scale"));
            input = {basis_span(), illuminant_span(), sensitivity_span(), source[0], source[1]};
        }
    };
    void check(const FjMallettMidgray& actual, const Json& expected) {
        exact<3>(actual.midgray_dwg_rgb, expected.at("dwg"));
        exact<3>(actual.raw_midgray_bgr, expected.at("raw_bgr"));
        const float normalization[]{actual.raw_green, actual.scale};
        exact<2>(normalization, expected.at("normalization"));
    }
    void numerical(const Json& fixture) {
        for (const auto& row : fixture.at("focused")) {
            FocusedInput input(row);
            Diagnostic error;
            for (bool facade : {false, true}) {
                FjMallettMidgray result{};
                status((facade ? fj_test_exposure_mallett_midgray : fj_legacy_exposure_mallett_midgray)(&input.input, &result, &error.error), FJ_STATUS_SUCCESS);
                check(result, row.at("expected"));
            }
            auto config = input.config();
            auto basis = input.native_basis();
            std::string diagnostic;
            require(JuicerExposure::mallett_midgray(basis, input.illuminant, input.sensitivity, config, diagnostic), "native focused binding");
            exact<3>(config.midgrayDWG, row.at("expected").at("dwg"));
            exact<3>(config.rawMidgray, row.at("expected").at("raw_bgr"));
            const float normalization[]{config.rawMidgrayGreen, config.midgrayScale};
            exact<2>(normalization, row.at("expected").at("normalization"));
            error.guards();
        }
        for (const auto& row : fixture.at("normalizations")) {
            const float green = std::bit_cast<float>(row.at("green").get<std::uint32_t>());
            const Json expected = Json::array({row.at("expected")[0], row.at("expected")[2]});
            for (bool facade : {false, true}) {
                FjMidgrayNormalization result{};
                Diagnostic error;
                status((facade ? fj_test_exposure_tc_midgray : fj_legacy_exposure_tc_midgray)(green, &result, &error.error), FJ_STATUS_SUCCESS);
                const float values[]{result.raw_green, result.scale};
                exact<2>(values, expected);
            }
            Spectral::FilmRawConfig config;
            std::string diagnostic;
            require(JuicerExposure::tc_midgray(green, config, diagnostic), "native TC binding");
            const float values[]{config.rawMidgrayGreen, config.midgrayScale};
            exact<2>(values, expected);
        }
        for (const auto& row : fixture.at("sources")) {
            const float ev = std::bit_cast<float>(row.at("ev").get<std::uint32_t>());
            const bool built = row.at("built").get<bool>();
            for (bool facade : {false, true}) {
                float result = -1;
                Diagnostic error;
                status((facade ? fj_test_exposure_reference_source : fj_legacy_exposure_reference_source)(ev, &result, &error.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built)
                    exact<1>(&result, Json::array({row.at("exp2_source")[1]}));
                else
                    require(std::bit_cast<std::uint32_t>(result) == 0, "failed source clear");
            }
            float result = -1;
            std::string diagnostic;
            require(JuicerExposure::reference_source(ev, result, diagnostic) == built, "native source disposition");
            if (built)
                exact<1>(&result, Json::array({row.at("exp2_source")[1]}));
            else
                require(result == 0 && !diagnostic.empty(), "native source failure");
        }
        for (const auto& row : fixture.at("references")) {
            ReferenceInput input(row);
            const bool built = row.at("built").get<bool>();
            for (bool facade : {false, true}) {
                FjReferenceRaw result{};
                Diagnostic error;
                status((facade ? fj_test_exposure_mallett_reference_raw : fj_legacy_exposure_mallett_reference_raw)(&input.input, &result, &error.error), built ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
                if (built)
                    exact<3>(result.rgb, row.at("expected"));
                else
                    require(std::all_of(std::begin(result.rgb), std::end(result.rgb), [](float v) {
                                return std::bit_cast<std::uint32_t>(v) == 0;
                            }),
                            "failed reference clear");
            }
            auto basis = input.native_basis();
            std::array<float, 3> result{};
            std::string diagnostic;
            require(JuicerExposure::mallett_reference_raw(basis, input.illuminant, input.sensitivity, input.input.source, input.input.green_scale, result, diagnostic) == built, "native reference disposition");
            if (built)
                exact<3>(result.data(), row.at("expected"));
        }
        FocusedInput focused(fixture.at("focused")[0]);
        ReferenceInput reference(fixture.at("references")[0]);
        FjMallettMidgray result{};
        FjMidgrayNormalization normalization{};
        FjReferenceRaw raw{};
        float source = 0;
        Diagnostic error;
        require(fj_test_mallett_exposure_abi_c(&focused.input, &result, 1.0f, &normalization, 0.0f, &source, &reference.input, &raw, &error.error) == 0, "C11 prototypes/status transport");
        check(result, fixture.at("focused")[0].at("expected"));
        exact<3>(raw.rgb, fixture.at("references")[0].at("expected"));
        require(normalization.raw_green == 1.0f && normalization.scale == 1.0f, "C11 TC values");
        exact<1>(&source, Json::array({fixture.at("sources")[0].at("exp2_source")[1]}));
    }
    void spectrum_exact(const FjSpectrumFixture& actual, const Json& expected) {
        exact<3>(actual.pre_xyz, expected.at("pre_xyz"));
        exact<3>(actual.consumer_white, expected.at("consumer_white"));
        exact<3>(actual.post_adapt_xyz, expected.at("post_adapt_xyz"));
        exact<81>(actual.spectrum, expected.at("spectrum"));
    }
    void spectra(const Json& fixture, const std::filesystem::path& resources) {
        JuicerAssets::AssetBridge assets(resources);
        const auto source = assets.copy_hanatos_lut();
        Diagnostic error;
        const auto ax = decode<81>(fixture.at("helper_tables").at("ax")), ay = decode<81>(fixture.at("helper_tables").at("ay")), az = decode<81>(fixture.at("helper_tables").at("az"));
        const auto sInverse = decode<9>(fixture.at("helper_s_inv"));
        std::array<std::array<float, 3>, 81> basis{}, sensitivity{};
        std::array<float, 81> illuminant{};
        for (auto& row : basis)
            row.fill(1);
        for (auto& row : sensitivity)
            row.fill(1);
        illuminant.fill(1);
        const std::array<std::uint32_t, 8> authored{0u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x7f800000u, 0xff800000u, 0x7fc46000u, 0xffc6a000u};
        for (std::size_t i = 0; i < authored.size(); ++i)
            basis[i / 3][i % 3] = std::bit_cast<float>(authored[i]);
        const FjMallettRawInput mallett{{1, 1, 1}, {basis.front().data(), 243}, {illuminant.data(), 81}, {sensitivity.front().data(), 243}};
        bool checkFaults = true;
        for (const auto& row : fixture.at("helpers")) {
            const auto rgb = decode<3>(row.at("rgb")), white = decode<3>(row.at("reference_white"));
            const FjHanatosSpectrumInput hanatos{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, {source.data.data(), source.data.size()}};
            FjTablesSpectrumInput tables{{rgb[0], rgb[1], rgb[2]}, {white[0], white[1], white[2]}, {}, {ax.data(), 81}, {ay.data(), 81}, {az.data(), 81}};
            std::copy(sInverse.begin(), sInverse.end(), tables.s_inverse);
            FjSpectrumFixture hanatosOut{}, tablesOut{};
            float bgr[3]{};
            require(fj_test_spectrum_abi_c(&hanatos, &hanatosOut, &tables, &tablesOut, &mallett, bgr, &error.error) == 0, "C11 spectrum prototypes/values");
            spectrum_exact(hanatosOut, row.at("hanatos_expected"));
            spectrum_exact(tablesOut, row.at("tables_expected"));
            require(bgr[0] == 235 && bgr[1] == 235 && bgr[2] == 235, "independent retained BGR leaf");
            if (!checkFaults)
                continue;
            checkFaults = false;
            for (std::uint32_t operation : {1u, 2u, 3u})
                for (std::uint32_t fault : {1u, 2u}) {
                    status(fj_test_exposure_fixture_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                    std::memset(&hanatosOut, 0x7f, sizeof(hanatosOut));
                    std::memset(&tablesOut, 0x7f, sizeof(tablesOut));
                    std::fill_n(bgr, 3, 99.0f);
                    const auto actual = operation == 1 ? fj_test_exposure_hanatos_spectrum(&hanatos, &hanatosOut, &error.error) : operation == 2 ? fj_test_exposure_tables_spectrum(&tables, &tablesOut, &error.error)
                                                                                                                                                 : fj_test_exposure_mallett_raw(&mallett, bgr, &error.error);
                    status(actual, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
                    require(fj_test_exposure_fixture_fault_consumed(operation, fault) == 1 && fj_test_exposure_fixture_fault_consumed(operation, fault) == 0, "closed fixture consumption before cleanup");
                    const auto* out = operation == 1 ? &hanatosOut : &tablesOut;
                    if (operation == 3)
                        require(bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0, "leaf failure clear");
                    else {
                        require(std::all_of(std::begin(out->spectrum), std::end(out->spectrum), [](float v) {
                                    return v == 0;
                                }),
                                "spectrum failure clear");
                        require(out->pre_xyz[0] == 0 && out->consumer_white[0] == 0 && out->post_adapt_xyz[0] == 0, "spectrum observations clear");
                    }
                    fj_test_exposure_fixture_clear_fault();
                }
        }
    }
    bool cleared(const FjMallettMidgray& result) {
        return std::all_of(std::begin(result.midgray_dwg_rgb), std::end(result.midgray_dwg_rgb), [](float v) {
                   return std::bit_cast<std::uint32_t>(v) == 0;
               }) &&
               std::all_of(std::begin(result.raw_midgray_bgr), std::end(result.raw_midgray_bgr), [](float v) {
                   return std::bit_cast<std::uint32_t>(v) == 0;
               }) &&
               result.raw_green == 0 && result.scale == 0;
    }
    void boundary(const Json& fixture) {
        FocusedInput input(fixture.at("focused")[0]);
        ReferenceInput reference(fixture.at("references")[0]);
        Diagnostic error;
        for (int bad = 0; bad < 12; ++bad) {
            auto view = input.input;
            FjMallettMidgray result{};
            std::memset(&result, 0x7f, sizeof(result));
            if (bad == 0)
                view.color.input_space = 4;
            if (bad == 1)
                view.color.decode_cctf = 2;
            if (bad == 2)
                view.color.adapt_xyz = 2;
            if (bad == 3)
                view.basis_rgb.count = 242;
            if (bad == 4)
                view.illuminant.count = 80;
            if (bad == 5)
                view.sensitivity_rgb.count = 244;
            if (bad == 6)
                view.basis_rgb.data = nullptr;
            if (bad == 7)
                view.illuminant.data = nullptr;
            if (bad == 8)
                view.sensitivity_rgb.data = nullptr;
            if (bad == 9)
                view.basis_rgb.count = std::numeric_limits<std::size_t>::max();
            if (bad == 10)
                view.basis_rgb.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(input.basis.data()) + 1);
            if (bad == 11)
                view.basis_rgb.data = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() - 3u);
            status(fj_legacy_exposure_mallett_midgray(&view, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(cleared(result), "structural failure clears completed fields");
            error.guards();
        }
        FjMallettMidgray result{};
        status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_mallett_midgray(&input.input, nullptr, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        FjErrorBuffer invalid{nullptr, 8, 77};
        std::memset(&result, 0x7f, sizeof(result));
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &invalid), FJ_STATUS_UNSUPPORTED_INPUT);
        require(cleared(result), "invalid diagnostic clears output");
        for (std::size_t capacity : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{512}}) {
            error.error.capacity = capacity;
            error.error.length = 99;
            status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(capacity == 0 ? error.error.length == 0 : error.error.length < capacity && error.error.data[error.error.length] == '\0', "bounded diagnostic/terminator");
            error.guards();
        }
        for (int bad = 0; bad < 4; ++bad) {
            auto view = reference.input;
            FjReferenceRaw raw;
            std::memset(&raw, 0x7f, sizeof(raw));
            if (bad == 0)
                view.basis_rgb.count = 242;
            if (bad == 1)
                view.illuminant.data = nullptr;
            if (bad == 2)
                view.sensitivity_rgb.count = std::numeric_limits<std::size_t>::max();
            status(fj_legacy_exposure_mallett_reference_raw(bad == 3 ? nullptr : &view, &raw, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
            require(raw.rgb[0] == 0 && raw.rgb[1] == 0 && raw.rgb[2] == 0, "reference structural clearing");
        }
        status(fj_legacy_exposure_mallett_reference_raw(&reference.input, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_tc_midgray(1, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        status(fj_legacy_exposure_reference_source(0, nullptr, nullptr), FJ_STATUS_UNSUPPORTED_INPUT);
        error.error.capacity = 512;
        for (std::uint32_t operation = 3; operation <= 6; ++operation) {
            status(fj_test_exposure_arm_fault(operation, 1, 3), FJ_STATUS_UNSUPPORTED_INPUT);
            for (std::uint32_t fault : {1u, 2u, 4u}) {
                status(fj_test_exposure_arm_fault(operation, 1, fault), FJ_STATUS_SUCCESS);
                FjMidgrayNormalization normalized{};
                float source = -1;
                FjReferenceRaw raw{};
                std::memset(&result, 0x7f, sizeof(result));
                std::memset(&normalized, 0x7f, sizeof(normalized));
                std::memset(&raw, 0x7f, sizeof(raw));
                const auto observed = operation == 3 ? fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error) : operation == 4 ? fj_legacy_exposure_tc_midgray(1, &normalized, &error.error)
                                                                                                                             : operation == 5   ? fj_legacy_exposure_reference_source(0, &source, &error.error)
                                                                                                                                                : fj_legacy_exposure_mallett_reference_raw(&reference.input, &raw, &error.error);
                status(observed, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : fault == 2 ? FJ_STATUS_INTERNAL_FAILURE
                                                                                       : FJ_STATUS_PREPARATION_FAILURE);
                require(fj_test_exposure_fault_consumed(operation, fault) == 1 && fj_test_exposure_fault_consumed(operation, fault) == 0, "same-thread consume-once before cleanup");
                require(operation != 3 || cleared(result), "panic/fault focused clear");
                require(operation != 4 || (normalized.raw_green == 0 && normalized.scale == 0), "TC fault clear");
                require(operation != 5 || source == 0, "source fault clear");
                require(operation != 6 || (raw.rgb[0] == 0 && raw.rgb[1] == 0 && raw.rgb[2] == 0), "reference fault clear");
                fj_test_exposure_clear_fault();
            }
        }
        fj_test_exposure_arm_facade_fault();
        status(fj_test_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_INTERNAL_FAILURE);
        require(fj_test_exposure_facade_fault_consumed() == 1 && cleared(result), "direct-core facade witness");
        status(fj_test_exposure_arm_fault(3, 1, 1), FJ_STATUS_SUCCESS);
        status(fj_legacy_exposure_mallett_midgray(nullptr, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 0, "malformed input skips mathematics");
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 1, "later valid call consumes armed fault");
        fj_test_exposure_clear_fault();
        for (std::uint32_t fault : {1u, 2u, 4u}) {
            status(fj_test_exposure_arm_fault(3, 1, fault), FJ_STATUS_SUCCESS);
            auto config = input.config();
            auto basis = input.native_basis();
            std::string diagnostic;
            bool exceptional = false;
            bool built = false;
            try {
                built = JuicerExposure::mallett_midgray(basis, input.illuminant, input.sensitivity, config, diagnostic);
            } catch (const JuicerCuda::ExecutionFailure& e) {
                exceptional = true;
                status(e.failure.status, fault == 1 ? FJ_STATUS_UNSUPPORTED_INPUT : FJ_STATUS_INTERNAL_FAILURE);
            }
            require(!built && exceptional == (fault != 4), "native ordinary-false versus exceptional mapping");
            require(fj_test_exposure_fault_consumed(3, fault) == 1, "native binding fault consumption");
            fj_test_exposure_clear_fault();
        }
        status(fj_test_exposure_arm_fault(3, 1, 1), FJ_STATUS_SUCCESS);
        FjStatus other{};
        std::thread worker([&] {
            FjMallettMidgray out{};
            FjErrorBuffer silent{nullptr, 0, 0};
            other = fj_legacy_exposure_mallett_midgray(&input.input, &out, &silent);
        });
        worker.join();
        status(other, FJ_STATUS_SUCCESS);
        status(fj_legacy_exposure_mallett_midgray(&input.input, &result, &error.error), FJ_STATUS_UNSUPPORTED_INPUT);
        require(fj_test_exposure_fault_consumed(3, 1) == 1, "thread-local fault witness");
        fj_test_exposure_clear_fault();
        FjMallettMidgray retained{};
        {
            FocusedInput temporary(fixture.at("focused")[0]);
            status(fj_legacy_exposure_mallett_midgray(&temporary.input, &retained, &error.error), FJ_STATUS_SUCCESS);
        }
        check(retained, fixture.at("focused")[0].at("expected"));
    }
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 5 || argc == 7, "manifest preset numerical|boundary|products|admission|spectra [resources fixtures]");
        const std::filesystem::path manifestPath(argv[1]);
        std::ifstream stream(manifestPath);
        const auto manifest = Json::parse(stream);
        std::ifstream values(manifestPath.parent_path() / manifest.at("presets").at(argv[2]).at("numerical").get<std::string>());
        const auto fixture = Json::parse(values);
        if ((std::string(argv[3]) == "products" || std::string(argv[3]) == "admission") && argc == 5) {
            std::ifstream productFile(manifestPath.parent_path() / manifest.at("presets").at(argv[2]).at("products").get<std::string>());
            const auto products = Json::parse(productFile);
            fj_test_mallett_callers(argv[4], products, manifestPath.parent_path() / "mallett-failure-contract.json", std::string(argv[3]) == "admission");
        } else if (std::string(argv[3]) == "numerical")
            numerical(fixture);
        else if (std::string(argv[3]) == "spectra" && argc == 7) {
            for (int i : {5, 6}) {
                std::ifstream stream(argv[i]);
                const auto captured = Json::parse(stream);
                spectra(captured, argv[4]);
            }
        } else if (std::string(argv[3]) == "boundary")
            boundary(fixture);
        else
            throw std::runtime_error("unknown fixed exposure group");
        std::puts("Mallett exposure qualification PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    } catch (const JuicerCuda::ExecutionFailure& e) {
        std::fprintf(stderr, "%s\n", e.failure.diagnostic.c_str());
        return 1;
    }
}
