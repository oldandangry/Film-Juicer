// SpectralUpsampling.h

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace SpectralUpsampling {

    // A struct to hold the LUT data
    struct LutCoeffs {
        int header[4];
        std::vector<std::vector<std::vector<float>>> pixels;
    };

    // Function to load the LUT file
    inline LutCoeffs load_coeffs_lut(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open LUT file: " + filename);
        }

        LutCoeffs lut;

        // Read the header
        file.read(reinterpret_cast<char*>(lut.header), sizeof(int) * 4);

        int width = lut.header[2];
        int height = lut.header[3];

        lut.pixels.resize(width, std::vector<std::vector<float>>(height, std::vector<float>(4)));

        // Read the pixel data
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                file.read(reinterpret_cast<char*>(lut.pixels[i][j].data()), sizeof(float) * 4);
            }
        }

        return lut;
    }

} // namespace SpectralUpsampling