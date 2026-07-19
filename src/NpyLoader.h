#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <limits>

// Minimal .npy loader for little-endian float64 arrays shaped (N, N, K)
// Stores as float in row-major C-order: ((i*size + j) * K + k)
struct NpySpectraLUT {
    int size = 0;       // LUT is square
    int numSamples = 0; // wavelengths per spectrum
    std::vector<float> data;
};

struct NpyFloat2D {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;
};

inline bool npy_byte_count(std::size_t count, std::size_t elementSize, std::size_t& outBytes) noexcept {
    if (elementSize != 0 && count > (std::numeric_limits<std::size_t>::max)() / elementSize) {
        return false;
    }
    outBytes = count * elementSize;
    return true;
}

inline bool npy_read_exact(std::ifstream& f, void* data, std::size_t byteCount) {
    if (byteCount > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)())) {
        return false;
    }
    f.read(static_cast<char*>(data), static_cast<std::streamsize>(byteCount));
    return static_cast<bool>(f);
}

template <typename T>
inline bool npy_read_vector(std::ifstream& f, std::vector<T>& data) {
    std::size_t byteCount = 0;
    if (!npy_byte_count(data.size(), sizeof(T), byteCount)) {
        return false;
    }
    return npy_read_exact(f, data.data(), byteCount);
}

inline float npy_half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t f_exp, f_sig;
    if (h_exp == 0x7C00u) { // Inf/NaN
        f_exp = 0xFFu << 23;
        f_sig = h_sig ? (h_sig << 13) : 0;
    } else if (!h_exp) { // subnormal/zero
        if (!h_sig) {
            f_exp = 0;
            f_sig = 0;
        } else {
            int shift = 0;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                ++shift;
            }
            h_sig &= 0x03FFu;
            f_exp = (127 - 15 - shift) << 23;
            f_sig = h_sig << 13;
        }
    } else { // normal
        f_exp = ((h_exp >> 10) + (127 - 15)) << 23;
        f_sig = h_sig << 13;
    }
    uint32_t f_bits = sign | f_exp | f_sig;
    float f_val;
    std::memcpy(&f_val, &f_bits, sizeof(f_val));
    return f_val;
}

inline bool load_npy_spectra_lut(const std::string& path, NpySpectraLUT& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    // Magic + version
    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return false;
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    if (!f)
        return false;

    // Header length
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t hl16 = 0;
        f.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    if (!f)
        return false;

    // Header string
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f)
        return false;

    // Detect dtype
    bool littleEndian = (header.find('<') != std::string::npos || header.find('|') != std::string::npos);
    bool isF16 = (header.find("f2") != std::string::npos);
    bool isF32 = (header.find("f4") != std::string::npos);
    bool isF64 = (header.find("f8") != std::string::npos);
    if (!littleEndian || (!isF16 && !isF32 && !isF64)) {
        return false; // unsupported type
    }
    if (header.find("True") != std::string::npos) {
        return false; // Fortran order not supported
    }

    // Parse shape "(N, N, K)"
    int N0 = 0, N1 = 0, K = 0;
    {
        size_t lp = header.find('(');
        size_t rp = header.find(')', lp);
        if (lp == std::string::npos || rp == std::string::npos)
            return false;
        std::string inside = header.substr(lp + 1, rp - lp - 1);
        std::vector<int> dims;
        size_t start = 0;
        while (start < inside.size()) {
            size_t comma = inside.find(',', start);
            std::string token = inside.substr(start, (comma == std::string::npos ? inside.size() : comma) - start);
            size_t a = 0;
            while (a < token.size() && std::isspace(static_cast<unsigned char>(token[a])))
                ++a;
            size_t b = token.size();
            while (b > a && std::isspace(static_cast<unsigned char>(token[b - 1])))
                --b;
            if (b > a)
                dims.push_back(std::atoi(token.substr(a, b - a).c_str()));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        if (dims.size() != 3)
            return false;
        N0 = dims[0];
        N1 = dims[1];
        K = dims[2];
    }
    if (N0 <= 0 || N1 <= 0 || N0 != N1 || K <= 0)
        return false;

    // Read payload into temp buffer
    const size_t count = static_cast<size_t>(N0) * static_cast<size_t>(N1) * static_cast<size_t>(K);
    out.size = N0;
    out.numSamples = K;
    out.data.resize(count);

    if (isF64) {
        std::vector<double> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        for (size_t i = 0; i < count; ++i)
            out.data[i] = static_cast<float>(buf[i]);
    } else if (isF32) {
        std::vector<float> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        out.data = buf; // already float
    } else if (isF16) {
        struct F16 {
            uint16_t v;
        };
        std::vector<F16> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        for (size_t i = 0; i < count; ++i)
            out.data[i] = npy_half_to_float(buf[i].v);
    }
    return true;
}

inline bool load_npy_float2d(const std::string& path, NpyFloat2D& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return false;
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    if (!f)
        return false;

    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t hl16 = 0;
        f.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    if (!f)
        return false;

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f)
        return false;

    bool littleEndian = (header.find('<') != std::string::npos || header.find('|') != std::string::npos);
    bool isF16 = (header.find("f2") != std::string::npos);
    bool isF32 = (header.find("f4") != std::string::npos);
    bool isF64 = (header.find("f8") != std::string::npos);
    if (!littleEndian || (!isF16 && !isF32 && !isF64)) {
        return false;
    }
    if (header.find("True") != std::string::npos) {
        return false;
    }

    int rows = 0, cols = 0;
    {
        size_t lp = header.find('(');
        size_t rp = header.find(')', lp);
        if (lp == std::string::npos || rp == std::string::npos)
            return false;
        std::string inside = header.substr(lp + 1, rp - lp - 1);
        std::vector<int> dims;
        size_t start = 0;
        while (start < inside.size()) {
            size_t comma = inside.find(',', start);
            std::string token = inside.substr(start, (comma == std::string::npos ? inside.size() : comma) - start);
            size_t a = 0;
            while (a < token.size() && std::isspace(static_cast<unsigned char>(token[a])))
                ++a;
            size_t b = token.size();
            while (b > a && std::isspace(static_cast<unsigned char>(token[b - 1])))
                --b;
            if (b > a)
                dims.push_back(std::atoi(token.substr(a, b - a).c_str()));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        if (dims.size() != 2)
            return false;
        rows = dims[0];
        cols = dims[1];
    }
    if (rows <= 0 || cols <= 0)
        return false;

    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    out.rows = rows;
    out.cols = cols;
    out.data.resize(count);

    if (isF64) {
        std::vector<double> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        for (size_t i = 0; i < count; ++i)
            out.data[i] = static_cast<float>(buf[i]);
    } else if (isF32) {
        std::vector<float> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        out.data = buf;
    } else if (isF16) {
        struct F16 {
            uint16_t v;
        };
        std::vector<F16> buf(count);
        if (!npy_read_vector(f, buf))
            return false;
        for (size_t i = 0; i < count; ++i)
            out.data[i] = npy_half_to_float(buf[i].v);
    }
    return true;
}
