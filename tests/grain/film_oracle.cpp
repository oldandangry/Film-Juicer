#include "film_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
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

        // Independent rectangle integration: recover the exact piecewise light
        // from discrete mixed differences, then sum the covered texel areas.
        double integrate(const float* table, Interval x, Interval y) {
            double integral = 0;
            for (int iy = static_cast<int>(std::floor(y.low)); iy < y.high; ++iy) {
                for (int ix = static_cast<int>(std::floor(x.low)); ix < x.high; ++ix) {
                    const int tx = ix % kTileSide;
                    const int ty = iy % kTileSide;
                    const double light = (entry(table, tx + 1, ty + 1) - entry(table, tx, ty + 1)) -
                                         (entry(table, tx + 1, ty) - entry(table, tx, ty));
                    integral += light * pixel_overlap(x, ix) * pixel_overlap(y, iy);
                }
            }
            return integral;
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

        IntegratedLight integrate_level(const FilmBank& bank, int record, int level, Interval x, Interval y) {
            const auto offset = (static_cast<std::size_t>(record) * kLevels + static_cast<std::size_t>(level)) * kTileEntries;
            const auto* table = bank.sums.data() + offset;
            const auto& means = bank.tables.records[record].levels[level];
            const double area = (x.high - x.low) * (y.high - y.low);
            const double light = means.meanLight + integrate(table, x, y) / area;
            const double scale = wrapped_scale(table, x.low, y.low) + wrapped_scale(table, x.low, y.high) +
                                 wrapped_scale(table, x.high, y.low) + wrapped_scale(table, x.high, y.high);
            // At most 20 rounded operations on any term from a prefix
            // through bilinear interpolation, periodic strips, box, area
            // division and mean-light addition. FMA can only shorten the path.
            const double lightError = gamma(20) * (scale / area + std::abs(means.meanLight));
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
        Interval tile_interval(double center, double width, bool flip, unsigned int offset) {
            const double oriented = flip ? kPlacementBlock - center : center;
            const double tileCenter = oriented + kPlacementMargin + offset;
            return {tileCenter - width / 2, tileCenter + width / 2};
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
            const double centerX = (probe.x + 0.5) * probe.pitch + shift;
            const double centerY = (probe.y + 0.5) * probe.pitch + shift;
            const int centralX = static_cast<int>(std::floor(centerX / kPlacementBlock));
            const int centralY = static_cast<int>(std::floor(centerY / kPlacementBlock));
            OracleSample result{};
            double norm = 0;
            for (int by = centralY - 1; by <= centralY + 1; ++by) {
                const double y = centerY - by * kPlacementBlock;
                for (int bx = centralX - 1; bx <= centralX + 1; ++bx) {
                    const double x = centerX - bx * kPlacementBlock;
                    const double weight = share(x) * share(y);
                    if (weight == 0)
                        continue;
                    const auto hash = placement(bx, by, probe, field);
                    const auto orientation = hash / 65536U % 8U;
                    const bool swap = orientation >= 4U;
                    const auto offsetX = hash % 256U;
                    const auto offsetY = hash / 256U % 256U;
                    const auto horizontal = tile_interval(x, bank.tables.footprint, orientation % 2U != 0, swap ? offsetY : offsetX);
                    const auto vertical = tile_interval(y, bank.tables.footprint, orientation % 4U >= 2U, swap ? offsetX : offsetY);
                    const auto a = integrate_level(bank, probe.record, blend.level, swap ? vertical : horizontal, swap ? horizontal : vertical);
                    const auto b = integrate_level(bank, probe.record, blend.level + 1, swap ? vertical : horizontal, swap ? horizontal : vertical);
                    result.delta += weight * std::lerp(a.delta, b.delta, blend.fraction);
                    result.roundingBound += weight * std::lerp(a.error, b.error, blend.fraction);
                    norm += weight * weight;
                }
            }
            const double divisor = std::sqrt(norm);
            return {result.delta / divisor, result.roundingBound / divisor};
        }
    } // namespace

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
        return {delta, result.roundingBound / worstNormalization + normalizationError + remainder};
    }
} // namespace GrainPrototype
