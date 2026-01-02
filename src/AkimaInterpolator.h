#pragma once

#include <cstddef>
#include <iterator>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace Interpolation {

    class AkimaInterpolator {
    public:
        enum class Method {
            Akima,
            Makima
        };

        struct StridedConstView {
            enum class ValueType {
                Float32,
                Float64
            };

            const void* data = nullptr;
            std::vector<size_t> shape;
            std::vector<ptrdiff_t> strides;
            ValueType valueType = ValueType::Float32;

            StridedConstView() = default;

            StridedConstView(const float* ptr,
                std::vector<size_t> shapeIn,
                std::vector<ptrdiff_t> strideIn)
                : data(ptr),
                shape(std::move(shapeIn)),
                strides(std::move(strideIn)),
                valueType(ValueType::Float32)
            {
            }

            StridedConstView(const double* ptr,
                std::vector<size_t> shapeIn,
                std::vector<ptrdiff_t> strideIn)
                : data(ptr),
                shape(std::move(shapeIn)),
                strides(std::move(strideIn)),
                valueType(ValueType::Float64)
            {
            }

            template <typename T, typename ShapeContainer, typename StrideContainer>
            StridedConstView(const T* ptr,
                const ShapeContainer& shapeIn,
                const StrideContainer& strideIn)
                : data(ptr),
                shape(std::begin(shapeIn), std::end(shapeIn)),
                strides(std::begin(strideIn), std::end(strideIn)),
                valueType(std::is_same<typename std::remove_cv<T>::type, double>::value ? ValueType::Float64 : ValueType::Float32)
            {
            }

            size_t rank() const { return shape.size(); }
            bool valid() const { return shape.size() == strides.size(); }

            static StridedConstView contiguous(const float* ptr, size_t count) {
                return StridedConstView(ptr,
                    std::vector<size_t>{ count },
                    std::vector<ptrdiff_t>{ 1 });
            }

            static StridedConstView contiguous(const std::vector<float>& values) {
                return contiguous(values.data(), values.size());
            }

            static StridedConstView contiguous(const double* ptr, size_t count) {
                return StridedConstView(ptr,
                    std::vector<size_t>{ count },
                    std::vector<ptrdiff_t>{ 1 });
            }

            static StridedConstView contiguous(const std::vector<double>& values) {
                return contiguous(values.data(), values.size());
            }
        };

        bool build(const std::vector<float>& x, const std::vector<float>& y, bool extrapolate = false, Method method = Method::Akima);
        bool build(const std::vector<float>& x, const std::vector<float>& y, bool extrapolate, const std::string& methodName);
        bool build(const std::vector<double>& x, const std::vector<double>& y, bool extrapolate = false, Method method = Method::Akima);
        bool build(const std::vector<double>& x, const std::vector<double>& y, bool extrapolate, const std::string& methodName);
        bool build(const std::vector<float>& x,
            const StridedConstView& y,
            int axis,
            bool extrapolate = false,
            Method method = Method::Akima);
        bool build(const std::vector<double>& x,
            const StridedConstView& y,
            int axis,
            bool extrapolate = false,
            Method method = Method::Akima);
        float evaluate(float x) const;
        double evaluate(double x) const;
        void evaluate_many(const std::vector<float>& xs, std::vector<float>& out) const;
        void evaluate_many(const std::vector<double>& xs, std::vector<double>& out) const;
        void evaluate_vector(float x, std::vector<float>& out) const;
        void evaluate_vector(double x, std::vector<double>& out) const;
        float derivative(float x, unsigned order = 1) const;
        double derivative(double x, unsigned order = 1) const;
        void derivative_vector(float x, std::vector<float>& out, unsigned order = 1) const;
        void derivative_vector(double x, std::vector<double>& out, unsigned order = 1) const;
        float antiderivative(float x, unsigned order = 1) const;
        double antiderivative(double x, unsigned order = 1) const;
        void antiderivative_vector(float x, std::vector<float>& out, unsigned order = 1) const;
        void antiderivative_vector(double x, std::vector<double>& out, unsigned order = 1) const;
        float integrate(float a, float b) const;
        double integrate(double a, double b) const;
        void integrate_vector(float a, float b, std::vector<float>& out) const;
        void integrate_vector(double a, double b, std::vector<double>& out) const;
        std::vector<float> roots(size_t channel = 0) const;
        std::vector<double> roots_double(size_t channel = 0) const;

        bool empty() const { return _xs.empty(); }
        Method method() const { return _method; }
        const std::vector<double>& slopes() const { return _slopes; }
        size_t channel_count() const { return _channelCount; }
        static bool try_parse_method(const std::string& methodName, Method& outMethod);
    private:
        bool select_segment(double x, size_t& i0, size_t& i1) const;
        void reset();
        double antiderivative_internal(size_t channel, double x, unsigned order) const;

        std::vector<double> _xs;
        std::vector<double> _ys;
        std::vector<double> _slopes;
        std::vector<double> _coeffs;
        size_t _channelCount = 0;
        bool _extrapolate = false;
        Method _method = Method::Akima;
    };

}
