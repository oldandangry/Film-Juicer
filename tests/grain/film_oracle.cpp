#include "film_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace GrainPrototype {
    namespace {
        struct Interval {
            double low;
            double high;
        };

        struct IntegratedLight {
            double delta;
            double error;
        };

        enum class Integration : std::uint8_t {
            FloatPrefix,
            Subtexel,
            Periodic
        };

        constexpr double kUnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;

        constexpr double gamma(int operations) {
            const double product = operations * kUnitRoundoff;
            return product / (1.0 - product);
        }

        double entry(const float* table, int x, int y) {
            return table[y * kTileStride + x];
        }

        double pixel_overlap(Interval range, int cell) {
            return std::min(range.high, static_cast<double>(cell + 1)) -
                   std::max(range.low, static_cast<double>(cell));
        }

        double texel_delta(const float* table, int x, int y) {
            return (entry(table, x + 1, y + 1) - entry(table, x, y + 1)) -
                   (entry(table, x + 1, y) - entry(table, x, y));
        }

        // Independent rectangle integration: recover the exact piecewise light
        // from discrete mixed differences, then sum the covered texel areas.
        double integrate(const float* table, Interval x, Interval y) {
            double integral = 0;
            for (int iy = static_cast<int>(std::floor(y.low)); iy < y.high; ++iy) {
                for (int ix = static_cast<int>(std::floor(x.low)); ix < x.high; ++ix) {
                    const int tx = ix % kTileSide;
                    const int ty = iy % kTileSide;
                    integral += texel_delta(table, tx, ty) * pixel_overlap(x, ix) * pixel_overlap(y, iy);
                }
            }
            return integral;
        }

        std::array<double, kTileSide> texel_coverage(Interval interval) {
            const double periods = std::floor((interval.high - interval.low) / kTileSide);
            const Interval residual{interval.low, interval.high - periods * kTileSide};
            std::array<double, kTileSide> coverage{};
            for (int texel = 0; texel < kTileSide; ++texel) {
                coverage[texel] = periods + std::max(0.0, pixel_overlap(residual, texel)) +
                                  std::max(0.0, pixel_overlap(residual, texel + kTileSide));
            }
            return coverage;
        }

        // Match the horizontal then vertical operands of the texel integral.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        double integrate_periodic(const float* table, Interval x, Interval y) {
            const auto columns = texel_coverage(x);
            const auto rows = texel_coverage(y);
            double sum = 0.0;
            for (int row = 0; row < kTileSide; ++row)
                for (int column = 0; column < kTileSide; ++column)
                    sum += texel_delta(table, column, row) * columns[column] * rows[row];
            return sum;
        }

        double corner_scale(const float* table, double x, double y) {
            const int ix = std::min(static_cast<int>(x), kTileSide - 1);
            const int iy = std::min(static_cast<int>(y), kTileSide - 1);
            const double ax = x - ix;
            const double ay = y - iy;
            const double a = std::abs(entry(table, ix, iy));
            const double b = std::abs(entry(table, ix + 1, iy));
            const double c = std::abs(entry(table, ix, iy + 1));
            const double d = std::abs(entry(table, ix + 1, iy + 1));
            return a + (b + a) * ax + (c + a) * ay + (d + b + c + a) * ax * ay;
        }

        double wrapped_scale(const float* table, double x, double y) {
            double scale = 0;
            // A corner is the residual period plus each of the crossed strips
            // and their shared corner. Costs here bound, rather than integrate.
            for (int wrapY = 0; wrapY <= static_cast<int>(y >= kTileSide); ++wrapY) {
                for (int wrapX = 0; wrapX <= static_cast<int>(x >= kTileSide); ++wrapX) {
                    const double tx = wrapX == 0 ? std::fmod(x, kTileSide) : kTileSide;
                    const double ty = wrapY == 0 ? std::fmod(y, kTileSide) : kTileSide;
                    scale += corner_scale(table, tx, ty);
                }
            }
            return scale;
        }

        IntegratedLight integrate_level(const FilmBank& bank, int record, int level, Interval x, Interval y, double area, Integration mode) {
            const auto offset = (static_cast<std::size_t>(record) * kLevels + static_cast<std::size_t>(level)) * kTileEntries;
            const auto* table = bank.sums.data() + offset;
            const auto& means = bank.tables.records[record].levels[level];
            const double sum = mode == Integration::Periodic ? integrate_periodic(table, x, y) : integrate(table, x, y);
            const double light = means.meanLight + sum / area;
            double lightError;
            if (mode != Integration::FloatPrefix) {
                // The texel differences and weights are evaluated in Float64.
                // Twice the Float32 light rounding budget plus an absolute
                // unit-scale term encloses both Float64 sums and the cast.
                lightError = gamma(2) * (1.0 + std::abs(light));
            } else {
                const double scale = wrapped_scale(table, x.low, y.low) + wrapped_scale(table, x.low, y.high) +
                                     wrapped_scale(table, x.high, y.low) + wrapped_scale(table, x.high, y.high);
                // At most 20 rounded operations from prefix interpolation
                // through periodic strips, box, division and mean addition.
                lightError = gamma(20) * (scale / area + std::abs(means.meanLight));
            }
            constexpr double floor = 1e-9;
            const double density = -std::log10(std::max(light, floor));
            const double low = -std::log10(std::max(light + lightError, floor));
            const double high = -std::log10(std::max(light - lightError, floor));
            // Allow four ulps (using the next exponent) for the log plus
            // subtraction, including Swift's Float light/log round trip.
            const double remainder = gamma(8) * (1 + std::abs(density) + std::abs(means.meanDensity));
            return {density - means.meanDensity, std::max(high - density, density - low) + remainder};
        }

        std::uint32_t permutation(std::uint32_t input) {
            const std::uint32_t state = 2891336453U + 747796405U * input;
            const int shift = static_cast<int>(state >> 28U) + 4;
            const std::uint32_t word = 277803737U * ((state >> shift) ^ state);
            return word ^ (word >> 22U);
        }

        std::uint32_t placement(int bx, int by, const FilmProbe& probe, int field) {
            const auto stream = static_cast<std::uint32_t>(200 + probe.record + kRecords * field);
            const auto rowSeed = static_cast<std::uint32_t>(by) ^ permutation(probe.seed ^ (stream * 0x9e3779b9U));
            return permutation(static_cast<std::uint32_t>(bx) ^ permutation(rowSeed));
        }

        // The interval is centered at the placed pixel and spans the footprint.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        Interval tile_interval(float low, float high, bool flip, unsigned int offset) {
            const float first = flip ? kPlacementBlock - high : low;
            const float last = flip ? kPlacementBlock - low : high;
            const float displacement = static_cast<float>(offset) + kPlacementMargin;
            return {first + displacement, last + displacement};
        }

        Interval tile_phase(Interval placed) {
            const double translation = std::floor(placed.low / kTileSide) * kTileSide;
            // Match the represented interval passed to the GPU's prefix sums.
            // The oracle still integrates the resulting texels individually.
            return {static_cast<float>(placed.low - translation), static_cast<float>(placed.high - translation)};
        }

        // The hash offset and the continuous footprint have different units.
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        Interval periodic_interval(float center, bool flip, unsigned int offset, double width) {
            const double placed = tile_interval(center, center, flip, offset).low;
            const double lower = placed - width / 2.0;
            const double phase = lower - std::floor(lower / kTileSide) * kTileSide;
            return {phase, phase + width};
        }

        double share(double centerWithinBlock) {
            // A trapezoid, with a plateau between the two seam bands.
            const double distance = std::min(centerWithinBlock, kPlacementBlock - centerWithinBlock);
            return std::clamp((distance + kSeam) / (2 * kSeam), 0.0, 1.0);
        }

        struct LevelBlend {
            double fraction;
            int level;
        };

        OracleSample evaluate_laying(const FilmBank& bank, const FilmProbe& probe, LevelBlend blend, int field) {
            const double shift = static_cast<double>(kPlacementBlock * field) / kLayings;
            const float centerX = (static_cast<float>(probe.x) + 0.5F) * probe.pitch + static_cast<float>(shift);
            const float centerY = (static_cast<float>(probe.y) + 0.5F) * probe.pitch + static_cast<float>(shift);
            const int centralX = static_cast<int>(std::floor(centerX / kPlacementBlock));
            const int centralY = static_cast<int>(std::floor(centerY / kPlacementBlock));
            OracleSample result{};
            double norm = 0;
            for (int by = centralY - 1; by <= centralY + 1; ++by) {
                const double y = centerY - static_cast<float>(by * kPlacementBlock);
                for (int bx = centralX - 1; bx <= centralX + 1; ++bx) {
                    const double x = centerX - static_cast<float>(bx * kPlacementBlock);
                    const double weight = share(x) * share(y);
                    if (weight == 0)
                        continue;
                    const auto hash = placement(bx, by, probe, field);
                    const auto orientation = hash / 65536U % 8U;
                    const bool swap = orientation >= 4U;
                    const auto offsetX = hash % 256U;
                    const auto offsetY = hash / 256U % 256U;
                    const float width = bank.tables.footprint;
                    const bool wide = width > 37.5F;
                    const bool subtexel = probe.productionSubtexel && width < 1.0F;
                    const bool periodic = width > 255.0F;
                    const bool local = wide || subtexel;
                    const float x0 = (local ? static_cast<float>(x) : centerX) - width * 0.5F;
                    const float y0 = (local ? static_cast<float>(y) : centerY) - width * 0.5F;
                    const float left = local ? 0.F : static_cast<float>(bx * kPlacementBlock);
                    const float top = local ? 0.F : static_cast<float>(by * kPlacementBlock);
                    auto horizontal = tile_interval(x0 - left, (x0 + width) - left, orientation % 2U != 0, swap ? offsetY : offsetX);
                    auto vertical = tile_interval(y0 - top, (y0 + width) - top, orientation % 4U >= 2U, swap ? offsetX : offsetY);
                    double area = ((x0 + width) - left - (x0 - left)) * ((y0 + width) - top - (y0 - top));
                    if (periodic) {
                        horizontal = periodic_interval(static_cast<float>(x), orientation % 2U != 0, swap ? offsetY : offsetX, width);
                        vertical = periodic_interval(static_cast<float>(y), orientation % 4U >= 2U, swap ? offsetX : offsetY, width);
                        area = static_cast<double>(width) * width;
                    } else if (wide) {
                        horizontal = tile_phase(horizontal);
                        vertical = tile_phase(vertical);
                        const float spanX = static_cast<float>(horizontal.high - horizontal.low);
                        const float spanY = static_cast<float>(vertical.high - vertical.low);
                        area = spanX * spanY;
                    }
                    if (subtexel) {
                        horizontal.high = horizontal.low + width;
                        vertical.high = vertical.low + width;
                        area = static_cast<double>(width) * width;
                        result.crossesTexel |= std::floor(horizontal.low) + 1 < horizontal.high ||
                                               std::floor(vertical.low) + 1 < vertical.high;
                    }
                    const auto mode = periodic ? Integration::Periodic : subtexel ? Integration::Subtexel
                                                                                  : Integration::FloatPrefix;
                    const auto a = integrate_level(bank, probe.record, blend.level, swap ? vertical : horizontal, swap ? horizontal : vertical, area, mode);
                    const auto b = integrate_level(bank, probe.record, blend.level + 1, swap ? vertical : horizontal, swap ? horizontal : vertical, area, mode);
                    result.delta += weight * std::lerp(a.delta, b.delta, blend.fraction);
                    result.roundingBound += weight * std::lerp(a.error, b.error, blend.fraction);
                    norm += weight * weight;
                }
            }
            const double divisor = std::sqrt(norm);
            return {result.delta / divisor, result.roundingBound / divisor, result.crossesTexel};
        }
    } // namespace

    void calibrate_subtexel(FilmBank& bank, float pitch) {
        bank.tables.footprint = pitch;
        constexpr int side = 128;
        constexpr int samples = side * side;
        const double width = pitch;
        std::array<std::vector<double>, kLevels> densities;
        for (auto& level : densities)
            level.resize(samples);
        for (int record = 0; record < kRecords; ++record) {
            auto& levels = bank.tables.records[record].levels;
            const FilmProbe probe{.seed = 0x5a3f1e1dU, .record = record};
            for (int i = 0; i < samples; ++i) {
                const double x = (i % side * 7919 % 4096) * width;
                const double y = (i / side * 104729 % 4096) * width;
                const int bx = static_cast<int>((x + width / 2) / kPlacementBlock);
                const int by = static_cast<int>((y + width / 2) / kPlacementBlock);
                const auto hash = placement(bx, by, probe, 0);
                const auto orientation = (hash >> 16U) & 7U;
                const double u = x - static_cast<double>(bx) * kPlacementBlock;
                const double v = y - static_cast<double>(by) * kPlacementBlock;
                const double left = (orientation & 1U) ? kPlacementBlock - u - width : u;
                const double top = (orientation & 2U) ? kPlacementBlock - v - width : v;
                const bool swap = (orientation & 4U) != 0;
                const double tx = (swap ? top : left) + kPlacementMargin + (hash & 255U);
                const double ty = (swap ? left : top) + kPlacementMargin + ((hash >> 8U) & 255U);
                for (int level = 0; level < kLevels; ++level) {
                    const auto offset = (static_cast<std::size_t>(record) * kLevels + level) * kTileEntries;
                    const double deviation = integrate(bank.sums.data() + offset, {tx, tx + width}, {ty, ty + width});
                    const float light = levels[level].meanLight + static_cast<float>(deviation / (width * width));
                    densities[level][i] = -std::log10(std::max(static_cast<double>(light), 1e-9));
                }
            }
            std::array<double, kLevels> means{};
            for (int level = 0; level < kLevels; ++level) {
                for (double density : densities[level])
                    means[level] += density / samples;
                levels[level].meanDensity = static_cast<float>(means[level]);
            }
            for (int level = 0; level + 1 < kLevels; ++level) {
                double aa = 0, bb = 0, ab = 0;
                for (int i = 0; i < samples; ++i) {
                    const double a = densities[level][i] - means[level];
                    const double b = densities[level + 1][i] - means[level + 1];
                    aa += a * a;
                    bb += b * b;
                    ab += a * b;
                }
                levels[level].correlation = aa > 0 && bb > 0
                                                ? static_cast<float>(std::clamp(ab / std::sqrt(aa * bb), -1.0, 1.0))
                                                : 1.F;
            }
        }
    }

    OracleSample evaluate_sample(const FilmBank& bank, FilmProbe probe) {
        const auto& table = bank.tables.records[probe.record];
        // Use the represented density and the pinned Float32 level coordinate.
        // This fixes the discrete input to the independent integration.
        const float gross = probe.net + table.offset;
        const double coordinate = std::clamp((gross - table.minimumGross) /
                                                 (table.maximumGross - table.minimumGross),
                                             0.F,
                                             1.F) *
                                  (kLevels - 1);
        const int level = std::min(static_cast<int>(coordinate), kLevels - 2);
        const double fraction = coordinate - level;
        OracleSample result{};
        for (int field = 0; field < kLayings; ++field) {
            const auto laying = evaluate_laying(bank, probe, {fraction, level}, field);
            result.delta += laying.delta;
            result.roundingBound += laying.roundingBound;
            result.crossesTexel |= laying.crossesTexel;
        }
        const double rho = table.levels[level].correlation;
        const double complement = 1 - fraction;
        const double kept = 1 - 2 * fraction * complement * (1 - rho);
        const double normalization = std::sqrt(kLayings * kept);
        const double delta = result.delta / normalization;
        // The absolute error in kept is bounded by gamma(12) because every
        // intermediate term is at most one; bound amplification at its low edge.
        const double worstNormalization = std::sqrt(kLayings * (kept - gamma(12)));
        const double normalizationError = std::abs(result.delta) * (1 / worstNormalization - 1 / normalization);
        const double remainder = gamma(32) * (1 + std::abs(probe.net) + std::abs(delta));
        return {delta, result.roundingBound / worstNormalization + normalizationError + remainder, result.crossesTexel};
    }
} // namespace GrainPrototype
