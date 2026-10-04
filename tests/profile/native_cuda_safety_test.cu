#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);
extern "C" cudaError_t juicer_cuda_print_focused_capture_density_from_camera_film_linear(
    const JuicerCuda::PrintPipelineRunParams*, JuicerCuda::CameraFilmLinearExposurePlanes, float*, float*, float*, void*);

namespace {

    void require_cuda(cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

    class DeviceFloats {
    public:
        explicit DeviceFloats(std::size_t count) : _count(count) {
            require_cuda(cudaMalloc(reinterpret_cast<void**>(&_data), count * sizeof(float)));
        }
        ~DeviceFloats() noexcept {
            const cudaError_t status = cudaFree(_data);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "profile safety fixture cudaFree failed: %d\n", static_cast<int>(status));
                std::abort();
            }
        }
        DeviceFloats(const DeviceFloats&) = delete;
        DeviceFloats& operator=(const DeviceFloats&) = delete;

        float* data() const noexcept {
            return _data;
        }
        void upload(std::span<const float> values) {
            if (values.size() != _count) {
                throw std::runtime_error("fixture upload count mismatch");
            }
            require_cuda(cudaMemcpy(_data, values.data(), _count * sizeof(float), cudaMemcpyHostToDevice));
        }
        std::vector<float> download() const {
            std::vector<float> values(_count);
            require_cuda(cudaMemcpy(values.data(), _data, _count * sizeof(float), cudaMemcpyDeviceToHost));
            return values;
        }

    private:
        float* _data = nullptr;
        std::size_t _count;
    };

    __global__ void sample_curve_kernel(JuicerCuda::DeviceCurveView curve, const float* queries, float* out, int count) {
        const int index = static_cast<int>(threadIdx.x);
        if (index < count) {
            const float query = sanitize_inf_logE_for_curve_device(queries[index], curve);
            out[index] = sample_density_at_logE_device(curve, query, 1.0f);
        }
    }

    struct CurveCase {
        std::span<const float> axis;
        std::span<const float> density;
        std::span<const float> queries;
        std::span<const float> expected;
    };

    void check_curve(const CurveCase& testCase) {
        DeviceFloats x(testCase.axis.size()), y(testCase.density.size()), query(testCase.queries.size()), out(testCase.queries.size());
        x.upload(testCase.axis);
        y.upload(testCase.density);
        query.upload(testCase.queries);
        const int count = static_cast<int>(testCase.axis.size());
        JuicerCuda::DeviceCurveView curve{x.data(), y.data(), count, 0, count - 1};
        sample_curve_kernel<<<1, 32>>>(curve, query.data(), out.data(), static_cast<int>(testCase.queries.size()));
        require_cuda(cudaGetLastError());
        const auto actual = out.download();
        for (std::size_t sample = 0; sample < testCase.expected.size(); ++sample) {
            if (std::isnan(testCase.expected[sample])) {
                EXPECT_TRUE(std::isnan(actual[sample])) << sample;
            } else {
                EXPECT_EQ(actual[sample], testCase.expected[sample]) << sample;
            }
        }
    }

    TEST(ProfileCudaSafety, SingletonDuplicateAndInfiniteDomains) {
        require_cuda(cudaSetDevice(0));
        constexpr float kInf = std::numeric_limits<float>::infinity();
        constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
        // Defined endpoint/interpolation identities of the retained device operator.
        // Ordered infinite knots keep its nonfinite-denominator return of y0.
        check_curve({.axis = std::array{0.0f}, .density = std::array{5.0f}, .queries = std::array{-kInf, 0.0f, kInf, kNan}, .expected = std::array{5.0f, 5.0f, 5.0f, kNan}});
        check_curve({.axis = std::array{1.0f, 1.0f}, .density = std::array{5.0f, 7.0f}, .queries = std::array{0.0f, 1.0f, 2.0f}, .expected = std::array{5.0f, 5.0f, 7.0f}});
        check_curve({.axis = std::array{-kInf, -1.0f, 0.0f, 0.0f, 1.0f, kInf}, .density = std::array{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, .queries = std::array{-2.0f, -1.0f, 0.0f, 0.5f, 2.0f, -kInf, kInf, kNan}, .expected = std::array{0.0f, 1.0f, 3.0f, 3.5f, 4.0f, kNan, kNan, kNan}});
        std::vector<float> axis(257);
        for (std::size_t sample = 0; sample < axis.size(); ++sample) {
            axis[sample] = static_cast<float>(sample);
        }
        check_curve({.axis = axis, .density = axis, .queries = std::array{-kInf, 127.5f, kInf}, .expected = std::array{0.0f, 127.5f, 256.0f}});
    }

    __global__ void coordinate_kernel(const float* normalized, float* out) {
        const int index = static_cast<int>(threadIdx.x);
        if (index < 9) {
            int base = 0;
            float fraction = 0.0f;
            film_tc_cubic_coordinate_device(normalized[index], 192, base, fraction);
            const auto offset = static_cast<std::size_t>(index) * 4u;
            out[offset] = static_cast<float>(base);
            out[offset + 1] = fraction;
            pchip_coordinate_float_device(normalized[index], 17, base, fraction);
            out[offset + 2] = static_cast<float>(base);
            out[offset + 3] = fraction;
        }
    }

    TEST(ProfileCudaSafety, ClampsBeforeFloatToIndexConversion) {
        require_cuda(cudaSetDevice(0));
        const std::array normalized{std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), -1.0f, 0.0f, 0.5f, 1.0f};
        DeviceFloats input(normalized.size()), output(normalized.size() * 4u);
        input.upload(normalized);
        coordinate_kernel<<<1, 32>>>(input.data(), output.data());
        require_cuda(cudaGetLastError());
        const auto actual = output.download();
        constexpr std::array<float, 9> kFilmBase{0, 0, 190, 0, 190, 0, 0, 95, 190};
        constexpr std::array<float, 9> kScanBase{0, 0, 15, 0, 15, 0, 0, 8, 15};
        constexpr std::array<float, 9> kFilmFraction{0, 0, 1, 0, 1, 0, 0, 0.5f, 1};
        constexpr std::array<float, 9> kScanFraction{0, 0, 1, 0, 1, 0, 0, 0, 1};
        for (std::size_t sample = 0; sample < normalized.size(); ++sample) {
            EXPECT_EQ(actual[sample * 4], kFilmBase[sample]);
            EXPECT_EQ(actual[sample * 4 + 1], kFilmFraction[sample]);
            EXPECT_EQ(actual[sample * 4 + 2], kScanBase[sample]);
            EXPECT_EQ(actual[sample * 4 + 3], kScanFraction[sample]);
        }
    }

    struct ScanCoordinates {
        double extreme;
        double nan;
    };

    __global__ void lut_kernel(const float* filmLut, JuicerCuda::ScanStagePayload scan, ScanCoordinates coordinates, float* out) {
        const float xyz[3] = {1.0f, 1.0f, 1.0f};
        film_tc_layer_exposures_device(xyz, filmLut, 192, out);
        const double extreme[3] = {coordinates.extreme, 0.5, -coordinates.extreme};
        out[6] = sample_pchip_float_log2_scan_lut_device(scan, extreme, out + 3) ? 1.0f : 0.0f;
        const double invalid[3] = {coordinates.nan, 0.5, 0.5};
        out[7] = sample_pchip_float_log2_scan_lut_device(scan, invalid, out + 3) ? 1.0f : 0.0f;
    }

    TEST(ProfileCudaSafety, NonfiniteLutContentsAndNarrowingCoordinatesUseRealConsumers) {
        require_cuda(cudaSetDevice(0));
        std::vector<float> filmLut(std::size_t{192} * 192u * 4u, std::numeric_limits<float>::infinity());
        for (std::size_t sample = 0; sample < filmLut.size(); sample += 8) {
            filmLut[sample] = std::numeric_limits<float>::quiet_NaN();
        }
        DeviceFloats film(filmLut.size()), nodes(24), cells(3), output(8);
        film.upload(filmLut);
        nodes.upload(std::array<float, 24>{});
        cells.upload(std::array<float, 3>{});
        JuicerCuda::ScanStagePayload scan{};
        scan.scanLutRes = 2;
        scan.scanLutLog2PchipXYZ = scan.scanLutPchipSlopeC = scan.scanLutPchipSlopeM = scan.scanLutPchipSlopeY = nodes.data();
        scan.scanLutPchipCellMin = scan.scanLutPchipCellMax = cells.data();
        lut_kernel<<<1, 1>>>(film.data(), scan, ScanCoordinates{std::numeric_limits<double>::max(), std::numeric_limits<double>::quiet_NaN()}, output.data());
        require_cuda(cudaGetLastError());
        const auto actual = output.download();
        // Film's existing computed-sample sanitation returns zero. A constant
        // zero log2 scanner LUT interpolates to zero at any clamped coordinate.
        for (std::size_t channel = 0; channel < 6; ++channel) {
            EXPECT_EQ(actual[channel], 0.0f);
        }
        EXPECT_EQ(actual[6], 1.0f);
        EXPECT_EQ(actual[7], 0.0f);
    }

    TEST(ProfileCudaSafety, DirectAndPrintDevelopmentUseVariableAndDuplicateAxes) {
        require_cuda(cudaSetDevice(0));
        // Actual production launchers consume the same C/M/Y device curves.
        // Powers of ten fall outside these domains; endpoints give exact values.
        for (const int count : {1, 17, 257}) {
            std::vector<float> axis(static_cast<std::size_t>(count), 0.0f);
            std::vector<float> density(static_cast<std::size_t>(count), 2.0f);
            axis.back() = count == 1 ? 0.0f : 1.0f;
            density.back() = count == 1 ? 2.0f : 4.0f;
            DeviceFloats x(axis.size()), yC(density.size()), yM(density.size()), yY(density.size()), red(2), green(2), blue(2), c(2), m(2), yellow(2);
            x.upload(axis);
            yC.upload(density);
            for (float& value : density) {
                value *= 2.0f;
            }
            yM.upload(density);
            for (float& value : density) {
                value *= 2.0f;
            }
            yY.upload(density);
            for (DeviceFloats* plane : {&red, &green, &blue}) {
                plane->upload(std::array{0.0f, 1000.0f});
            }
            JuicerCuda::CameraFilmLinearExposurePlanes planes{red.data(), green.data(), blue.data(), 2};
            JuicerCuda::DirectPipelineRunParams direct{};
            direct.width = 2;
            direct.height = 1;
            direct.filmDevelop.densR = {x.data(), yC.data(), count, 0, count - 1};
            direct.filmDevelop.densG = {x.data(), yM.data(), count, 0, count - 1};
            direct.filmDevelop.densB = {x.data(), yY.data(), count, 0, count - 1};
            JuicerCuda::PrintPipelineRunParams print{};
            print.width = 2;
            print.height = 1;
            print.filmDevelop = direct.filmDevelop;
            for (const bool printRoute : {false, true}) {
                const auto status = printRoute ? juicer_cuda_print_focused_capture_density_from_camera_film_linear(&print, planes, c.data(), m.data(), yellow.data(), nullptr)
                                               : juicer_cuda_direct_focused_capture_density_from_camera_film_linear(&direct, planes, c.data(), m.data(), yellow.data(), nullptr);
                require_cuda(status);
                float channelScale = 1.0f;
                for (const DeviceFloats* output : {&c, &m, &yellow}) {
                    const auto actual = output->download();
                    EXPECT_EQ(actual[0], 2.0f * channelScale) << count << " print=" << printRoute;
                    EXPECT_EQ(actual[1], (count == 1 ? 2.0f : 4.0f) * channelScale) << count << " print=" << printRoute;
                    channelScale *= 2.0f;
                }
            }
        }
    }

} // namespace
