#include "AkimaInterpolator.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace Interpolation {
    namespace {
        constexpr double kEpsilon = 1e-12;
        constexpr double kPi = 3.141592653589793238462643383279502884;

        double power(double base, unsigned exponent)
        {
            double result = 1.0;
            for (unsigned i = 0; i < exponent; ++i) {
                result *= base;
            }
            return result;
        }

        double evaluate_cubic(const double* coeffs, double dx)
        {
            double result = coeffs[3];
            result = result * dx + coeffs[2];
            result = result * dx + coeffs[1];
            result = result * dx + coeffs[0];
            return result;
        }

        double evaluate_cubic_derivative(const double* coeffs, double dx, unsigned order)
        {
            switch (order) {
            case 0:
                return evaluate_cubic(coeffs, dx);
            case 1:
                return coeffs[1] + (2.0 * coeffs[2] + 3.0 * coeffs[3] * dx) * dx;
            case 2:
                return 2.0 * coeffs[2] + 6.0 * coeffs[3] * dx;
            case 3:
                return 6.0 * coeffs[3];
            default:
                return 0.0;
            }
        }

        double antiderivative_polynomial(const double* coeffs, double dx, unsigned order)
        {
            if (order == 0) {
                return evaluate_cubic(coeffs, dx);
            }
            double result = 0.0;
            for (unsigned k = 0; k < 4; ++k) {
                const double coeff = coeffs[k];
                if (coeff == 0.0) {
                    continue;
                }
                double denom = 1.0;
                for (unsigned m = 1; m <= order; ++m) {
                    denom *= static_cast<double>(k + m);
                }
                const double powValue = power(dx, k + order);
                result += coeff * powValue / denom;
            }
            return result;
        }

        std::vector<double> solve_linear(double a1, double a0)
        {
            std::vector<double> roots;
            if (std::abs(a1) <= kEpsilon) {
                return roots;
            }
            roots.push_back(-a0 / a1);
            return roots;
        }

        std::vector<double> solve_quadratic(double a2, double a1, double a0)
        {
            std::vector<double> roots;
            if (std::abs(a2) <= kEpsilon) {
                return solve_linear(a1, a0);
            }
            const double discriminant = a1 * a1 - 4.0 * a2 * a0;
            if (discriminant < -kEpsilon) {
                return roots;
            }
            if (std::abs(discriminant) <= kEpsilon) {
                roots.push_back(-a1 / (2.0 * a2));
                return roots;
            }
            const double sqrtDisc = std::sqrt(discriminant);
            roots.push_back((-a1 - sqrtDisc) / (2.0 * a2));
            roots.push_back((-a1 + sqrtDisc) / (2.0 * a2));
            return roots;
        }

        std::vector<double> solve_cubic(double a3, double a2, double a1, double a0)
        {
            if (std::abs(a3) <= kEpsilon) {
                return solve_quadratic(a2, a1, a0);
            }

            const double invA3 = 1.0 / a3;
            const double b = a2 * invA3;
            const double c = a1 * invA3;
            const double d = a0 * invA3;

            const double Q = (3.0 * c - b * b) / 9.0;
            const double R = (9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / 54.0;
            const double D = Q * Q * Q + R * R;

            std::vector<double> roots;
            if (D > kEpsilon) {
                const double sqrtD = std::sqrt(D);
                const double S = std::cbrt(R + sqrtD);
                const double T = std::cbrt(R - sqrtD);
                roots.push_back(-b / 3.0 + (S + T));
                return roots;
            }

            if (std::abs(D) <= kEpsilon) {
                const double S = std::cbrt(R);
                roots.push_back(-b / 3.0 + 2.0 * S);
                roots.push_back(-b / 3.0 - S);
                return roots;
            }

            const double sqrtNegQ = std::sqrt(-Q);
            const double cosArg = std::max(-1.0, std::min(1.0, R / (sqrtNegQ * sqrtNegQ * sqrtNegQ)));
            const double theta = std::acos(cosArg);
            const double factor = 2.0 * sqrtNegQ;
            roots.push_back(factor * std::cos(theta / 3.0) - b / 3.0);
            roots.push_back(factor * std::cos((theta + 2.0 * kPi) / 3.0) - b / 3.0);
            roots.push_back(factor * std::cos((theta + 4.0 * kPi) / 3.0) - b / 3.0);
            return roots;
        }
    }

    void AkimaInterpolator::reset()
    {
        _xs.clear();
        _ys.clear();
        _slopes.clear();
        _coeffs.clear();
        _channelCount = 0;
        _extrapolate = false;
        _method = Method::Akima;
    }

    bool AkimaInterpolator::select_segment(double x, size_t& i0, size_t& i1) const
    {
        const size_t count = _xs.size();
        if (count < 2) {
            return false;
        }
        const double xmin = _xs.front();
        const double xmax = _xs.back();
        if (x < xmin) {
            if (!_extrapolate) {
                return false;
            }
            i0 = 0;
            i1 = 1;
            return true;
        }
        if (x > xmax) {
            if (!_extrapolate) {
                return false;
            }
            i1 = count - 1;
            i0 = i1 - 1;
            return true;
        }
        auto it = std::upper_bound(_xs.begin(), _xs.end(), x);
        size_t idx = static_cast<size_t>(std::distance(_xs.begin(), it));
        if (idx == 0) {
            idx = 1;
        }
        if (idx >= count) {
            idx = count - 1;
        }
        i0 = idx - 1;
        i1 = idx;
        return true;
    }

    bool AkimaInterpolator::try_parse_method(const std::string& methodName, Method& outMethod)
    {
        std::string normalized;
        normalized.reserve(methodName.size());
        for (char ch : methodName) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        if (normalized == "akima") {
            outMethod = Method::Akima;
            return true;
        }
        if (normalized == "makima") {
            outMethod = Method::Makima;
            return true;
        }
        return false;
    }

    bool AkimaInterpolator::build(const std::vector<float>& x,
        const std::vector<float>& y,
        bool extrapolate,
        Method method)
    {
        std::vector<double> xDouble(x.begin(), x.end());
        return build(xDouble, StridedConstView::contiguous(y), 0, extrapolate, method);
    }

    bool AkimaInterpolator::build(const std::vector<float>& x,
        const std::vector<float>& y,
        bool extrapolate,
        const std::string& methodName)
    {
        Method parsedMethod = Method::Akima;
        if (!try_parse_method(methodName, parsedMethod)) {
            return false;
        }
        return build(x, y, extrapolate, parsedMethod);
    }

    bool AkimaInterpolator::build(const std::vector<double>& x,
        const std::vector<double>& y,
        bool extrapolate,
        Method method)
    {
        return build(x, StridedConstView::contiguous(y), 0, extrapolate, method);
    }

    bool AkimaInterpolator::build(const std::vector<double>& x,
        const std::vector<double>& y,
        bool extrapolate,
        const std::string& methodName)
    {
        Method parsedMethod = Method::Akima;
        if (!try_parse_method(methodName, parsedMethod)) {
            return false;
        }
        return build(x, y, extrapolate, parsedMethod);
    }

    bool AkimaInterpolator::build(const std::vector<float>& x,
        const StridedConstView& y,
        int axis,
        bool extrapolate,
        Method method)
    {
        std::vector<double> xDouble(x.begin(), x.end());
        return build(xDouble, y, axis, extrapolate, method);
    }

    bool AkimaInterpolator::build(const std::vector<double>& x,
        const StridedConstView& y,
        int axis,
        bool extrapolate,
        Method method)
    {
        const size_t inputCount = x.size();
        if (inputCount < 2) {
            reset();
            return false;
        }
        if (!y.valid()) {
            reset();
            return false;
        }
        const size_t rank = y.rank();
        if (rank == 0) {
            reset();
            return false;
        }
        int normalizedAxis = axis;
        if (normalizedAxis < 0) {
            normalizedAxis += static_cast<int>(rank);
        }
        if (normalizedAxis < 0 || normalizedAxis >= static_cast<int>(rank)) {
            reset();
            return false;
        }
        const size_t axisIndex = static_cast<size_t>(normalizedAxis);
        if (y.shape[axisIndex] != inputCount) {
            reset();
            return false;
        }
        if (y.data == nullptr) {
            reset();
            return false;
        }

        for (size_t i = 0; i < inputCount; ++i) {
            if (!std::isfinite(x[i])) {
                reset();
                return false;
            }
        }

        if (!std::is_sorted(x.begin(), x.end())) {
            reset();
            return false;
        }


        size_t channelCount = 1;
        for (size_t dim = 0; dim < rank; ++dim) {
            const size_t dimSize = y.shape[dim];
            if (dimSize == 0) {
                reset();
                return false;
            }
            if (dim == axisIndex) {
                continue;
            }
            if (channelCount > std::numeric_limits<size_t>::max() / dimSize) {
                reset();
                return false;
            }
            channelCount *= dimSize;
        }

        std::vector<size_t> permutation(inputCount);
        std::iota(permutation.begin(), permutation.end(), size_t{ 0 });
        std::stable_sort(permutation.begin(), permutation.end(), [&](size_t a, size_t b) {
            return x[a] < x[b];
            });

        std::vector<size_t> sanitizedIndices;
        sanitizedIndices.reserve(inputCount);
        double lastUniqueX = 0.0;
        bool hasLastUniqueX = false;
        auto appendIfUnique = [&](size_t idx) {
            const double value = x[idx];
            if (!hasLastUniqueX || value != lastUniqueX) {
                sanitizedIndices.push_back(idx);
                lastUniqueX = value;
                hasLastUniqueX = true;
            }
            };
        for (size_t idx : permutation) {
            appendIfUnique(idx);
        }

        if (sanitizedIndices.size() < 2) {
            reset();
            return false;
        }

        _xs.resize(sanitizedIndices.size());
        for (size_t i = 0; i < sanitizedIndices.size(); ++i) {
            _xs[i] = x[sanitizedIndices[i]];
        }
        _channelCount = channelCount;
        const size_t sampleCount = _xs.size();
        _ys.resize(sampleCount * _channelCount);
        _slopes.assign(sampleCount * _channelCount, 0.0);

        const ptrdiff_t axisStride = y.strides[axisIndex];
        std::vector<size_t> nonAxisDims;
        nonAxisDims.reserve(rank > 0 ? rank - 1 : 0);
        for (size_t dim = 0; dim < rank; ++dim) {
            if (dim != axisIndex) {
                nonAxisDims.push_back(dim);
            }
        }

        const float* yFloat = y.valueType == StridedConstView::ValueType::Float32
            ? static_cast<const float*>(y.data)
            : nullptr;
        const double* yDouble = y.valueType == StridedConstView::ValueType::Float64
            ? static_cast<const double*>(y.data)
            : nullptr;

        std::vector<size_t> coordinate(nonAxisDims.size(), 0);
        for (size_t channel = 0; channel < _channelCount; ++channel) {
            size_t remainder = channel;
            for (size_t idx = 0; idx < nonAxisDims.size(); ++idx) {
                const size_t dim = nonAxisDims[idx];
                const size_t dimSize = y.shape[dim];
                const size_t coord = dimSize > 0 ? (remainder % dimSize) : 0;
                remainder = dimSize > 0 ? (remainder / dimSize) : 0;
                coordinate[idx] = coord;
            }

            ptrdiff_t baseOffset = 0;
            for (size_t idx = 0; idx < nonAxisDims.size(); ++idx) {
                const size_t dim = nonAxisDims[idx];
                baseOffset += static_cast<ptrdiff_t>(coordinate[idx]) * y.strides[dim];
            }

            for (size_t i = 0; i < sampleCount; ++i) {
                const size_t originalIndex = sanitizedIndices[i];
                const ptrdiff_t offset = baseOffset + static_cast<ptrdiff_t>(originalIndex) * axisStride;
                double value = 0.0;
                if (yFloat != nullptr) {
                    value = static_cast<double>(yFloat[offset]);
                }
                else if (yDouble != nullptr) {
                    value = yDouble[offset];
                }
                else {
                    reset();
                    return false;
                }
                if (!std::isfinite(value)) {
                    reset();
                    return false;
                }
                _ys[i * _channelCount + channel] = value;
            }
        }

        if (_channelCount == 0) {
            reset();
            return false;
        }

        const size_t n = sampleCount;
        if (n == 2) {
            const double dx = _xs[1] - _xs[0];
            if (dx == 0.0) {
                reset();
                return false;
            }
            for (size_t channel = 0; channel < _channelCount; ++channel) {
                const double y0 = _ys[channel];
                const double y1 = _ys[_channelCount + channel];
                const double slope = (y1 - y0) / dx;
                _slopes[channel] = slope;
                _slopes[_channelCount + channel] = slope;
            }
            _coeffs.resize((_xs.size() > 1 ? (_xs.size() - 1) : 0) * _channelCount * 4, 0.0);
            const double h = _xs[1] - _xs[0];
            const double invH = 1.0 / h;
            const double invH2 = invH * invH;
            for (size_t channel = 0; channel < _channelCount; ++channel) {
                const double y0 = _ys[channel];
                const double y1 = _ys[_channelCount + channel];
                const double m0 = _slopes[channel];
                const double m1 = _slopes[_channelCount + channel];
                const double delta = (y1 - y0) * invH;
                const size_t baseIndex = channel * 4;
                _coeffs[baseIndex] = y0;
                _coeffs[baseIndex + 1] = m0;
                _coeffs[baseIndex + 2] = (3.0 * delta - 2.0 * m0 - m1) * invH;
                _coeffs[baseIndex + 3] = (m0 + m1 - 2.0 * delta) * invH2;
            }
            _extrapolate = extrapolate;
            _method = method;
            return true;
        }

        const size_t mCount = n + 3;
        std::vector<double> dxs(n - 1, 0.0);
        for (size_t i = 0; i + 1 < n; ++i) {
            const double delta = _xs[i + 1] - _xs[i];
            if (delta <= 0.0) {
                reset();
                return false;
            }
            dxs[i] = delta;
        }

        std::vector<double> m(mCount, 0.0);
        std::vector<double> t(n, 0.0);
        std::vector<double> dm(mCount - 1, 0.0);

        const bool useMakima = method == Method::Makima;
        std::vector<double> pm;
        if (useMakima) {
            pm.assign(mCount - 1, 0.0);
        }

        std::vector<double> f1(n, 0.0);
        std::vector<double> f2(n, 0.0);
        std::vector<double> f12(n, 0.0);
        constexpr double break_mult = 1e-9;

        auto processChannel = [&](size_t channel, double cutoff, bool assignSlopes) {
            std::fill(m.begin(), m.end(), 0.0);
            for (size_t i = 0; i + 1 < n; ++i) {
                const double y0 = _ys[i * _channelCount + channel];
                const double y1 = _ys[(i + 1) * _channelCount + channel];
                m[i + 2] = (y1 - y0) / dxs[i];
            }
            m[1] = 2.0 * m[2] - m[3];
            m[0] = 2.0 * m[1] - m[2];
            m[mCount - 2] = 2.0 * m[mCount - 3] - m[mCount - 4];
            m[mCount - 1] = 2.0 * m[mCount - 2] - m[mCount - 3];

            for (size_t i = 0; i < n; ++i) {
                t[i] = 0.5 * (m[i] + m[i + 3]);
            }
            for (size_t i = 0; i + 1 < mCount; ++i) {
                dm[i] = std::abs(m[i + 1] - m[i]);
            }

            if (useMakima) {
                for (size_t i = 0; i + 1 < mCount; ++i) {
                    pm[i] = std::abs(m[i + 1] + m[i]);
                }
            }

            double channelMaxF12 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (useMakima) {
                    f1[i] = dm[i + 2] + 0.5 * pm[i + 2];
                    f2[i] = dm[i] + 0.5 * pm[i];
                }
                else {
                    f1[i] = dm[i + 2];
                    f2[i] = dm[i];
                }
                f12[i] = f1[i] + f2[i];
                channelMaxF12 = std::max(channelMaxF12, f12[i]);
                if (assignSlopes) {
                    double slopeValue = t[i];
                    if (f12[i] > cutoff) {
                        const double numer = f2[i] * (m[i + 2] - m[i + 1]);
                        slopeValue = m[i + 1] + numer / f12[i];
                    }
                    _slopes[i * _channelCount + channel] = slopeValue;
                }
            }
            return channelMaxF12;
            };

        double globalMaxF12 = 0.0;
        for (size_t channel = 0; channel < _channelCount; ++channel) {
            globalMaxF12 = std::max(globalMaxF12, processChannel(channel, 0.0, false));
        }
        const double cutoff = break_mult * globalMaxF12;
        for (size_t channel = 0; channel < _channelCount; ++channel) {
            processChannel(channel, cutoff, true);
        }
        const size_t segmentCount = n > 0 ? (n - 1) : 0;
        _coeffs.assign(segmentCount * _channelCount * 4, 0.0);
        for (size_t seg = 0; seg < segmentCount; ++seg) {
            const double h = _xs[seg + 1] - _xs[seg];
            const double invH = 1.0 / h;
            const double invH2 = invH * invH;
            for (size_t channel = 0; channel < _channelCount; ++channel) {
                const size_t baseIndex = (seg * _channelCount + channel) * 4;
                const double y0 = _ys[seg * _channelCount + channel];
                const double y1 = _ys[(seg + 1) * _channelCount + channel];
                const double m0 = _slopes[seg * _channelCount + channel];
                const double m1 = _slopes[(seg + 1) * _channelCount + channel];
                const double delta = (y1 - y0) * invH;
                _coeffs[baseIndex] = y0;
                _coeffs[baseIndex + 1] = m0;
                _coeffs[baseIndex + 2] = (3.0 * delta - 2.0 * m0 - m1) * invH;
                _coeffs[baseIndex + 3] = (m0 + m1 - 2.0 * delta) * invH2;
            }
        }
        _extrapolate = extrapolate;
        _method = method;
        return true;
    }

    double AkimaInterpolator::antiderivative_internal(size_t channel, double x, unsigned order) const
    {
        if (channel >= _channelCount || _coeffs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (order == 0) {
            size_t segStart = 0;
            size_t segEnd = 0;
            if (!select_segment(x, segStart, segEnd)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double dx = x - static_cast<double>(_xs[segStart]);
            const size_t coeffIndex = (segStart * _channelCount + channel) * 4;
            return evaluate_cubic_derivative(&_coeffs[coeffIndex], dx, 0);
        }

        size_t segStart = 0;
        size_t segEnd = 0;
        if (!select_segment(x, segStart, segEnd)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        std::vector<double> factorials(order + 1, 1.0);
        for (unsigned i = 2; i <= order; ++i) {
            factorials[i] = factorials[i - 1] * static_cast<double>(i);
        }

        std::vector<double> cumulative(order + 1, 0.0);
        std::vector<double> next(order + 1, 0.0);
        for (size_t seg = 0; seg < segStart; ++seg) {
            const double h = static_cast<double>(_xs[seg + 1]) - static_cast<double>(_xs[seg]);
            const double* coeff = &_coeffs[(seg * _channelCount + channel) * 4];
            for (unsigned r = 1; r <= order; ++r) {
                double value = cumulative[r];
                double dxPower = h;
                for (unsigned m = 1; m < r; ++m) {
                    value += cumulative[r - m] * dxPower / factorials[m];
                    dxPower *= h;
                }
                value += antiderivative_polynomial(coeff, h, r);
                next[r] = value;
            }
            for (unsigned r = 1; r <= order; ++r) {
                cumulative[r] = next[r];
            }
        }

        const double dx = x - static_cast<double>(_xs[segStart]);
        const double* coeff = &_coeffs[(segStart * _channelCount + channel) * 4];
        double result = cumulative[order];
        double dxPower = dx;
        for (unsigned m = 1; m < order; ++m) {
            result += cumulative[order - m] * dxPower / factorials[m];
            dxPower *= dx;
        }
        result += antiderivative_polynomial(coeff, dx, order);
        return result;
    }

    double AkimaInterpolator::evaluate(double x) const
    {
        size_t i0 = 0;
        size_t i1 = 0;
        if (!select_segment(x, i0, i1) || _channelCount == 0 || _coeffs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t coeffIndex = (i0 * _channelCount) * 4;
        const double dx = x - _xs[i0];
        return evaluate_cubic_derivative(&_coeffs[coeffIndex], dx, 0);
    }

    float AkimaInterpolator::evaluate(float x) const
    {
        const double value = evaluate(static_cast<double>(x));
        return static_cast<float>(value);
    }

    void AkimaInterpolator::evaluate_vector(double x, std::vector<double>& out) const
    {
        const size_t channelCount = _channelCount;
        out.resize(channelCount);
        if (channelCount == 0) {
            return;
        }

        size_t i0 = 0;
        size_t i1 = 0;
        if (!select_segment(x, i0, i1) || _coeffs.empty()) {
            std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
            return;
        }

        const double dx = x - _xs[i0];
        for (size_t channel = 0; channel < channelCount; ++channel) {
            const size_t coeffIndex = (i0 * channelCount + channel) * 4;
            out[channel] = evaluate_cubic_derivative(&_coeffs[coeffIndex], dx, 0);
        }
    }

    void AkimaInterpolator::evaluate_vector(float x, std::vector<float>& out) const
    {
        std::vector<double> temp;
        evaluate_vector(static_cast<double>(x), temp);
        out.resize(temp.size());
        for (size_t i = 0; i < temp.size(); ++i) {
            out[i] = static_cast<float>(temp[i]);
        }
    }

    void AkimaInterpolator::evaluate_many(const std::vector<double>& xs, std::vector<double>& out) const
    {
        if (_channelCount <= 1) {
            out.resize(xs.size());
            for (size_t i = 0; i < xs.size(); ++i) {
                out[i] = evaluate(xs[i]);
            }
            return;
        }

        out.resize(xs.size() * _channelCount);
        std::vector<double> buffer;
        buffer.reserve(_channelCount);
        for (size_t i = 0; i < xs.size(); ++i) {
            evaluate_vector(xs[i], buffer);
            for (size_t channel = 0; channel < _channelCount; ++channel) {
                out[i * _channelCount + channel] = buffer[channel];
            }
        }
    }

    void AkimaInterpolator::evaluate_many(const std::vector<float>& xs, std::vector<float>& out) const
    {
        if (_channelCount <= 1) {
            out.resize(xs.size());
            for (size_t i = 0; i < xs.size(); ++i) {
                out[i] = evaluate(xs[i]);
            }
            return;
        }

        out.resize(xs.size() * _channelCount);
        std::vector<float> buffer;
        buffer.reserve(_channelCount);
        for (size_t i = 0; i < xs.size(); ++i) {
            evaluate_vector(xs[i], buffer);
            for (size_t channel = 0; channel < _channelCount; ++channel) {
                out[i * _channelCount + channel] = buffer[channel];
            }
        }
    }

    double AkimaInterpolator::derivative(double x, unsigned order) const
    {
        size_t i0 = 0;
        size_t i1 = 0;
        if (!select_segment(x, i0, i1) || _channelCount == 0 || _coeffs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double dx = x - _xs[i0];
        const size_t coeffIndex = (i0 * _channelCount) * 4;
        return evaluate_cubic_derivative(&_coeffs[coeffIndex], dx, order);
    }

    float AkimaInterpolator::derivative(float x, unsigned order) const
    {
        const double value = derivative(static_cast<double>(x), order);
        return static_cast<float>(value);
    }

    void AkimaInterpolator::derivative_vector(double x, std::vector<double>& out, unsigned order) const
    {
        const size_t channelCount = _channelCount;
        out.resize(channelCount);
        if (channelCount == 0) {
            return;
        }
        size_t i0 = 0;
        size_t i1 = 0;
        if (!select_segment(x, i0, i1) || _coeffs.empty()) {
            std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const double dx = x - _xs[i0];
        for (size_t channel = 0; channel < channelCount; ++channel) {
            const size_t coeffIndex = (i0 * channelCount + channel) * 4;
            out[channel] = evaluate_cubic_derivative(&_coeffs[coeffIndex], dx, order);
        }
    }

    void AkimaInterpolator::derivative_vector(float x, std::vector<float>& out, unsigned order) const
    {
        std::vector<double> temp;
        derivative_vector(static_cast<double>(x), temp, order);
        out.resize(temp.size());
        for (size_t i = 0; i < temp.size(); ++i) {
            out[i] = static_cast<float>(temp[i]);
        }
    }

    double AkimaInterpolator::antiderivative(double x, unsigned order) const
    {
        if (_channelCount == 0 || _coeffs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return antiderivative_internal(0, x, order);
    }

    float AkimaInterpolator::antiderivative(float x, unsigned order) const
    {
        const double value = antiderivative(static_cast<double>(x), order);
        return static_cast<float>(value);
    }

    void AkimaInterpolator::antiderivative_vector(double x, std::vector<double>& out, unsigned order) const
    {
        const size_t channelCount = _channelCount;
        out.resize(channelCount);
        if (channelCount == 0 || _coeffs.empty()) {
            std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (size_t channel = 0; channel < channelCount; ++channel) {
            out[channel] = antiderivative_internal(channel, x, order);
        }
    }

    void AkimaInterpolator::antiderivative_vector(float x, std::vector<float>& out, unsigned order) const
    {
        std::vector<double> temp;
        antiderivative_vector(static_cast<double>(x), temp, order);
        out.resize(temp.size());
        for (size_t i = 0; i < temp.size(); ++i) {
            out[i] = static_cast<float>(temp[i]);
        }
    }

    double AkimaInterpolator::integrate(double a, double b) const
    {
        if (_channelCount == 0 || _coeffs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double upper = antiderivative_internal(0, b, 1);
        const double lower = antiderivative_internal(0, a, 1);
        if (!std::isfinite(upper) || !std::isfinite(lower)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return upper - lower;
    }

    float AkimaInterpolator::integrate(float a, float b) const
    {
        const double value = integrate(static_cast<double>(a), static_cast<double>(b));
        return static_cast<float>(value);
    }

    void AkimaInterpolator::integrate_vector(double a, double b, std::vector<double>& out) const
    {
        const size_t channelCount = _channelCount;
        out.resize(channelCount);
        if (channelCount == 0 || _coeffs.empty()) {
            std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (size_t channel = 0; channel < channelCount; ++channel) {
            const double upper = antiderivative_internal(channel, b, 1);
            const double lower = antiderivative_internal(channel, a, 1);
            if (!std::isfinite(upper) || !std::isfinite(lower)) {
                out[channel] = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                out[channel] = upper - lower;
            }
        }
    }

    void AkimaInterpolator::integrate_vector(float a, float b, std::vector<float>& out) const
    {
        std::vector<double> temp;
        integrate_vector(static_cast<double>(a), static_cast<double>(b), temp);
        out.resize(temp.size());
        for (size_t i = 0; i < temp.size(); ++i) {
            out[i] = static_cast<float>(temp[i]);
        }
    }

    std::vector<double> AkimaInterpolator::roots_double(size_t channel) const
    {
        std::vector<double> result;
        if (_channelCount == 0 || channel >= _channelCount || _coeffs.empty()) {
            return result;
        }

        const size_t segmentCount = _xs.size() > 1 ? (_xs.size() - 1) : 0;
        result.reserve(segmentCount);
        constexpr double tol = 1e-8;
        for (size_t seg = 0; seg < segmentCount; ++seg) {
            const double x0 = _xs[seg];
            const double x1 = _xs[seg + 1];
            const double h = x1 - x0;
            const double* coeff = &_coeffs[(seg * _channelCount + channel) * 4];
            const double c0 = coeff[0];
            const double c1 = coeff[1];
            const double c2 = coeff[2];
            const double c3 = coeff[3];
            const std::vector<double> polyRoots = solve_cubic(c3, c2, c1, c0);
            for (double dx : polyRoots) {
                const double xRoot = x0 + dx;
                const bool inInterval = dx >= -tol && dx <= h + tol;
                bool accept = false;
                if (inInterval) {
                    accept = true;
                }
                else if (_extrapolate) {
                    if (seg == 0 && dx < -tol) {
                        accept = true;
                    }
                    else if (seg + 1 == segmentCount && dx > h + tol) {
                        accept = true;
                    }
                }
                if (accept) {
                    result.push_back(xRoot);
                }
            }
        }
        std::sort(result.begin(), result.end());
        const double mergeTol = 1e-6;
        auto it = std::unique(result.begin(), result.end(), [mergeTol](double a, double b) {
            return std::abs(a - b) <= mergeTol;
            });
        result.erase(it, result.end());
        return result;
    }

    std::vector<float> AkimaInterpolator::roots(size_t channel) const
    {
        const std::vector<double> doubleRoots = roots_double(channel);
        std::vector<float> result;
        result.reserve(doubleRoots.size());
        for (double value : doubleRoots) {
            result.push_back(static_cast<float>(value));
        }
        return result;
    }

} // namespace Interpolation
