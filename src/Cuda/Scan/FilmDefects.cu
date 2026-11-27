#include "Cuda/Scan/PostEffects.h"
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Cuda/JuicerCudaDeviceHelpers.cuh"

#include "openrand/philox.h"

namespace {

    __device__ __forceinline__ std::uint64_t splitmix64_device(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    __device__ __forceinline__ float hash_to_unit_device(std::uint64_t h) {
        constexpr float kInv = 1.0f / 16777216.0f; // 2^24
        return static_cast<float>((h >> 40) & 0xFFFFFFu) * kInv;
    }

    __device__ __forceinline__ float defect_attribute_unit_device(
        std::uint64_t primitiveHash,
        std::uint64_t attributeSalt) {
        // Remix before truncation so event selection does not constrain attributes.
        return hash_to_unit_device(splitmix64_device(primitiveHash ^ attributeSalt));
    }

    struct DefectPrimitive {
        float cx;
        float cy;
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        float strength;
        float softness;
        // Ellipses: center, quadratic coefficients for horizontal intersections.
        float ex[3];
        float ey[3];
        float ea[3];
        float eb[3];
        float ec[3];
        // Four connected vertices; widths are full physical widths.
        float px[4];
        float py[4];
        float widths[4];
        float depths[4];
        int stroke;
        int active;
    };

    __device__ float defect_lane(std::uint64_t identity, unsigned int lane) {
        return defect_attribute_unit_device(identity,
                                            0xD6E8FEB86659FD93ULL * (static_cast<std::uint64_t>(lane) + 1ULL));
    }

    struct DefectCandidateAddress {
        std::int64_t cellX;
        std::int64_t cellY;
        int family;
        int slot;
        int offsetX;
        int offsetY;
    };

    struct DefectLengthMixture {
        float minimum;
        float bulkMaximum;
        float maximum;
        float tailFraction;
    };

    struct DefectStrokeGeometry {
        float lengthMm;
        float driftFraction;
        float angleRadians;
        float fractions[4];
        float widths[4];
        float depths[4];
    };

    struct DefectTileFootprint {
        int pixelX;
        int pixelY;
        int tileX;
        int tileY;
        int width;
        int height;
        float scale;
    };

    __device__ std::uint64_t defect_identity(const JuicerCuda::FilmDefectsPayload& p,
                                             const DefectCandidateAddress& address) {
        std::uint64_t h = splitmix64_device(p.sessionSeed);
        if (address.family < 2) {
            h = splitmix64_device(h ^ splitmix64_device(p.clipToken));
        }
        h = splitmix64_device(h ^ static_cast<std::uint64_t>(address.cellX));
        h = splitmix64_device(h ^ static_cast<std::uint64_t>(address.cellY));
        h = splitmix64_device(h ^ (0x8EBC6AF09C88C6E3ULL * static_cast<std::uint64_t>(address.family + 1)));
        return splitmix64_device(h ^ static_cast<std::uint64_t>(address.slot));
    }

    __device__ float defect_mixture(float2 random, const DefectLengthMixture& distribution) {
        const float t = random.y * random.y * random.y;
        return random.x < distribution.tailFraction ? distribution.bulkMaximum + (distribution.maximum - distribution.bulkMaximum) * t : distribution.minimum + (distribution.bulkMaximum - distribution.minimum) * t;
    }

    __device__ void build_defect_stroke(DefectPrimitive& q, std::uint64_t h, const DefectStrokeGeometry& geometry) {
        float dx[4] = {0.0f, (defect_lane(h, 12) - 0.5f) * geometry.driftFraction * geometry.lengthMm, (defect_lane(h, 13) - 0.5f) * geometry.driftFraction * geometry.lengthMm, (defect_lane(h, 14) - 0.5f) * geometry.driftFraction * geometry.lengthMm};
        float total = 0.0f;
        float maximumWidth = 0.0f;
        for (int i = 1; i < 4; ++i) {
            total += hypotf(dx[i] - dx[i - 1],
                            geometry.lengthMm * (geometry.fractions[i] - geometry.fractions[i - 1]));
        }
        for (float width : geometry.widths) {
            maximumWidth = fmaxf(maximumWidth, width);
        }
        const float norm = geometry.lengthMm / total;
        const float c = cosf(geometry.angleRadians);
        const float s = sinf(geometry.angleRadians);
        q.stroke = 1;
        q.xmin = q.ymin = 1.0e30f;
        q.xmax = q.ymax = -1.0e30f;
        for (int i = 0; i < 4; ++i) {
            const float y = (geometry.fractions[i] - 0.5f) * geometry.lengthMm * norm;
            const float x = dx[i] * norm;
            q.px[i] = x * c - y * s;
            q.py[i] = x * s + y * c;
            q.widths[i] = geometry.widths[i];
            q.depths[i] = geometry.depths[i];
            q.xmin = fminf(q.xmin, q.px[i] - maximumWidth * 0.5f);
            q.xmax = fmaxf(q.xmax, q.px[i] + maximumWidth * 0.5f);
            q.ymin = fminf(q.ymin, q.py[i] - maximumWidth * 0.5f);
            q.ymax = fmaxf(q.ymax, q.py[i] + maximumWidth * 0.5f);
        }
    }

    __device__ float defect_dust_opacity(
        const JuicerCuda::DefectDustPayload& dust,
        bool fiber,
        float u) {
        const float minimum = fiber ? dust.fiberOpacityMin : dust.compactOpacityMin;
        const float faintEnd = fiber ? dust.fiberOpacityFaintEnd : dust.compactOpacityFaintEnd;
        const float intermediateEnd =
            fiber ? dust.fiberOpacityIntermediateEnd : dust.compactOpacityIntermediateEnd;
        const float maximum = fiber ? dust.fiberOpacityMax : dust.compactOpacityMax;
        const float faintCumulative = dust.opacityFaintCumulative;
        const float intermediateCumulative = dust.opacityIntermediateCumulative;
        if (u < faintCumulative) {
            return minimum + (faintEnd - minimum) * (u / faintCumulative);
        }
        if (u < intermediateCumulative) {
            return faintEnd + (intermediateEnd - faintEnd) *
                                  ((u - faintCumulative) /
                                   (intermediateCumulative - faintCumulative));
        }
        return intermediateEnd + (maximum - intermediateEnd) *
                                     ((u - intermediateCumulative) /
                                      (1.0f - intermediateCumulative));
    }

    __device__ float defect_candidate_softness(
        float u,
        float minimumMm,
        float maximumMm,
        float sizeCapFraction,
        float nominalSizeMm) {
        return fminf(minimumMm + (maximumMm - minimumMm) * u * u,
                     nominalSizeMm * sizeCapFraction);
    }

    __device__ DefectPrimitive build_defect_candidate(const JuicerCuda::FilmDefectsPayload& p,
                                                      const DefectCandidateAddress& address) {
        DefectPrimitive q{};
        const auto& dust = address.family == 0 ? p.filmDust : p.gateDust;
        const auto& scratch = address.family == 1 ? p.filmScratch : p.gateScratch;
        const bool isDust = address.family == 0 || address.family == 2;
        const float probability = isDust ? dust.slotProbability : scratch.slotProbability;
        const std::uint64_t h = defect_identity(p, address);
        if (defect_lane(h, 0) >= probability) {
            return q;
        }
        const float cellWidth = isDust ? dust.cellWidthMm : scratch.cellWidthMm;
        const float cellHeight = isDust ? dust.cellHeightMm : scratch.cellHeightMm;
        q.cx = (static_cast<float>(address.offsetX) + defect_lane(h, 1)) * cellWidth;
        q.cy = (static_cast<float>(address.offsetY) + defect_lane(h, 2)) * cellHeight;
        q.active = 1;
        if (isDust) {
            const float angle = defect_lane(h, 4) * 6.28318530718f;
            const bool fiber = defect_lane(h, 5) < dust.fiberFraction;
            const float opacityDraw = defect_lane(h, 3);
            q.strength = defect_dust_opacity(dust, fiber, opacityDraw);
            if (fiber) {
                const float length = dust.fiberLengthMinMm +
                                     (dust.fiberLengthMaxMm - dust.fiberLengthMinMm) * defect_lane(h, 6);
                const float width = dust.fiberWidthMinMm +
                                    (dust.fiberWidthMaxMm - dust.fiberWidthMinMm) * defect_lane(h, 7);
                q.softness = defect_candidate_softness(defect_lane(h, 47),
                                                       dust.softnessMinMm,
                                                       dust.softnessMaxMm,
                                                       dust.softnessSizeCapFraction,
                                                       width);
                const float firstKnot = dust.fiberFirstKnotMin +
                                        (dust.fiberFirstKnotMax - dust.fiberFirstKnotMin) * defect_lane(h, 30);
                const float secondKnot = dust.fiberSecondKnotMin +
                                         (dust.fiberSecondKnotMax - dust.fiberSecondKnotMin) * defect_lane(h, 31);
                DefectStrokeGeometry geometry{};
                geometry.lengthMm = length;
                geometry.driftFraction = dust.fiberDriftFraction;
                geometry.angleRadians = angle;
                geometry.fractions[0] = 0.0f;
                geometry.fractions[1] = firstKnot;
                geometry.fractions[2] = secondKnot;
                geometry.fractions[3] = 1.0f;
                geometry.widths[1] = width *
                                     (dust.fiberInteriorWidthMinFraction +
                                      (dust.fiberInteriorWidthMaxFraction -
                                       dust.fiberInteriorWidthMinFraction) *
                                          defect_lane(h, 21));
                geometry.widths[2] = width *
                                     (dust.fiberInteriorWidthMinFraction +
                                      (dust.fiberInteriorWidthMaxFraction -
                                       dust.fiberInteriorWidthMinFraction) *
                                          defect_lane(h, 22));
                for (float& depth : geometry.depths) {
                    depth = 1.0f;
                }
                build_defect_stroke(q, h, geometry);
            } else {
                const float diameter = defect_mixture(make_float2(defect_lane(h, 6), defect_lane(h, 7)), DefectLengthMixture{dust.diameterMinMm, dust.diameterBulkMaxMm, dust.diameterMaxMm, dust.diameterTailFraction});
                q.softness = defect_candidate_softness(defect_lane(h, 47),
                                                       dust.softnessMinMm,
                                                       dust.softnessMaxMm,
                                                       dust.softnessSizeCapFraction,
                                                       diameter);
                float centerX[3]{};
                float centerY[3]{};
                float radiusX[3]{};
                float radiusY[3]{};
                float ellipseAngle[3]{};
                radiusX[0] = 1.0f;
                radiusY[0] = dust.compactDominantAspectMin +
                             (dust.compactDominantAspectMax - dust.compactDominantAspectMin) *
                                 defect_lane(h, 32);
                ellipseAngle[0] = angle;
                const float dominantCos = cosf(angle);
                const float dominantSin = sinf(angle);
                float enclosingRadius = 1.0f;
                for (int i = 1; i < 3; ++i) {
                    radiusX[i] = dust.compactSubsidiaryScaleMin +
                                 (dust.compactSubsidiaryScaleMax - dust.compactSubsidiaryScaleMin) *
                                     defect_lane(h, 32 + i * 5);
                    const float aspect = dust.compactSubsidiaryAspectMin +
                                         (dust.compactSubsidiaryAspectMax - dust.compactSubsidiaryAspectMin) *
                                             defect_lane(h, 33 + i * 5);
                    radiusY[i] = radiusX[i] * aspect;
                    const float polar = defect_lane(h, 34 + i * 5) * 6.28318530718f;
                    const float radial = dust.compactSubsidiaryOffsetMax * sqrtf(defect_lane(h, 35 + i * 5));
                    const float localX = radial * cosf(polar);
                    const float localY = radial * radiusY[0] * sinf(polar);
                    centerX[i] = localX * dominantCos - localY * dominantSin;
                    centerY[i] = localX * dominantSin + localY * dominantCos;
                    ellipseAngle[i] = angle +
                                      (2.0f * defect_lane(h, 36 + i * 5) - 1.0f) *
                                          dust.compactSubsidiaryAngleMaxRadians;
                    enclosingRadius = fmaxf(enclosingRadius,
                                            hypotf(centerX[i], centerY[i]) +
                                                fmaxf(radiusX[i], radiusY[i]));
                }
                const float scale = diameter * 0.5f / enclosingRadius;
                q.xmin = q.ymin = 1.0e30f;
                q.xmax = q.ymax = -1.0e30f;
                for (int i = 0; i < 3; ++i) {
                    const float rx = radiusX[i] * scale;
                    const float ry = radiusY[i] * scale;
                    const float c = cosf(ellipseAngle[i]);
                    const float s = sinf(ellipseAngle[i]);
                    q.ex[i] = centerX[i] * scale;
                    q.ey[i] = centerY[i] * scale;
                    q.ea[i] = c * c / (rx * rx) + s * s / (ry * ry);
                    q.eb[i] = c * s * (1.0f / (rx * rx) - 1.0f / (ry * ry));
                    q.ec[i] = s * s / (rx * rx) + c * c / (ry * ry);
                    const float extentX = sqrtf(rx * rx * c * c + ry * ry * s * s);
                    const float extentY = sqrtf(rx * rx * s * s + ry * ry * c * c);
                    q.xmin = fminf(q.xmin, q.ex[i] - extentX);
                    q.xmax = fmaxf(q.xmax, q.ex[i] + extentX);
                    q.ymin = fminf(q.ymin, q.ey[i] - extentY);
                    q.ymax = fmaxf(q.ymax, q.ey[i] + extentY);
                }
            }
        } else {
            q.strength = scratch.strengthMin +
                         (scratch.strengthMax - scratch.strengthMin) * defect_lane(h, 3);
            const float length = defect_mixture(make_float2(defect_lane(h, 6), defect_lane(h, 7)), DefectLengthMixture{scratch.lengthMinMm, scratch.lengthBulkMaxMm, scratch.lengthMaxMm, scratch.lengthTailFraction});
            const float width = defect_mixture(make_float2(defect_lane(h, 8), defect_lane(h, 9)), DefectLengthMixture{scratch.widthMinMm, scratch.widthBulkMaxMm, scratch.widthMaxMm, scratch.widthTailFraction});
            q.softness = defect_candidate_softness(defect_lane(h, 47),
                                                   scratch.softnessMinMm,
                                                   scratch.softnessMaxMm,
                                                   scratch.softnessSizeCapFraction,
                                                   width);
            DefectStrokeGeometry geometry{};
            geometry.lengthMm = length;
            geometry.driftFraction = scratch.driftFraction;
            const bool interrupted = defect_lane(h, 30) < scratch.interruptionProbability;
            if (interrupted) {
                const float gapCenter = scratch.gapCenterMin +
                                        (scratch.gapCenterMax - scratch.gapCenterMin) * defect_lane(h, 31);
                const float gapSpan = scratch.gapSpanMin +
                                      (scratch.gapSpanMax - scratch.gapSpanMin) * defect_lane(h, 32);
                geometry.fractions[1] = gapCenter - gapSpan * 0.5f;
                geometry.fractions[2] = gapCenter + gapSpan * 0.5f;
            } else {
                geometry.fractions[1] = scratch.firstKnotMin +
                                        (scratch.firstKnotMax - scratch.firstKnotMin) * defect_lane(h, 31);
                geometry.fractions[2] = scratch.secondKnotMin +
                                        (scratch.secondKnotMax - scratch.secondKnotMin) * defect_lane(h, 32);
            }
            geometry.fractions[3] = 1.0f;
            for (int i = 0; i < 4; ++i) {
                const float widthFraction = scratch.interiorWidthMinFraction +
                                            (scratch.interiorWidthMaxFraction -
                                             scratch.interiorWidthMinFraction) *
                                                defect_lane(h, 33 + i);
                geometry.widths[i] =
                    (i == 1 || i == 2 || defect_lane(h, 37 + i) < scratch.endpointAbruptProbability)
                        ? width * widthFraction
                        : 0.0f;
                geometry.depths[i] = scratch.interiorDepthMinFraction +
                                     (scratch.interiorDepthMaxFraction -
                                      scratch.interiorDepthMinFraction) *
                                         defect_lane(h, 41 + i);
            }
            if (interrupted) {
                geometry.depths[1] = 0.0f;
                geometry.depths[2] = 0.0f;
            }
            if (length <= scratch.scuffLengthMaxMm && defect_lane(h, 45) < scratch.scuffProbability) {
                geometry.angleRadians =
                    (2.0f * defect_lane(h, 46) - 1.0f) * scratch.scuffAngleMaxRadians;
            }
            build_defect_stroke(q, h, geometry);
        }
        return q;
    }

    // Integrate the union's horizontal intervals before applying one particle strength.
    __device__ float defect_cross_section(const DefectPrimitive& q, float y, float2 horizontalBounds) {
        float lo[5]{};
        float hi[5]{};
        float depth[5]{};
        int count = 0;
        for (int i = 0; i < 3; ++i) {
            float a = 0.0f;
            float b = 0.0f;
            if (!q.stroke) {
                const float dy = y - q.ey[i];
                const float disc = q.ea[i] - (q.ea[i] * q.ec[i] - q.eb[i] * q.eb[i]) * dy * dy;
                if (disc <= 0.0f) {
                    continue;
                }
                const float center = q.ex[i] - q.eb[i] * dy / q.ea[i];
                const float radius = sqrtf(disc) / q.ea[i];
                a = center - radius;
                b = center + radius;
            } else {
                // A tapered quadrilateral for each connected centerline segment.
                const float dx = q.px[i + 1] - q.px[i];
                const float dy = q.py[i + 1] - q.py[i];
                const float invLength = rsqrtf(dx * dx + dy * dy);
                const float nx = -dy * invLength * 0.5f;
                const float ny = dx * invLength * 0.5f;
                const float vx[4] = {q.px[i] + nx * q.widths[i], q.px[i + 1] + nx * q.widths[i + 1], q.px[i + 1] - nx * q.widths[i + 1], q.px[i] - nx * q.widths[i]};
                const float vy[4] = {q.py[i] + ny * q.widths[i], q.py[i + 1] + ny * q.widths[i + 1], q.py[i + 1] - ny * q.widths[i + 1], q.py[i] - ny * q.widths[i]};
                a = 1.0e30f;
                b = -1.0e30f;
                for (int edge = 0; edge < 4; ++edge) {
                    const int next = (edge + 1) % 4;
                    if (y >= fminf(vy[edge], vy[next]) && y < fmaxf(vy[edge], vy[next])) {
                        const float x = vx[edge] + (y - vy[edge]) / (vy[next] - vy[edge]) * (vx[next] - vx[edge]);
                        a = fminf(a, x);
                        b = fmaxf(b, x);
                    }
                }
            }
            a = fmaxf(horizontalBounds.x, a);
            b = fminf(horizontalBounds.y, b);
            if (b > a) {
                depth[count] = 1.0f;
                if (q.stroke) {
                    const float dx = q.px[i + 1] - q.px[i];
                    const float dy = q.py[i + 1] - q.py[i];
                    const float midpointX = 0.5f * (a + b);
                    const float denominator = dx * dx + dy * dy;
                    const float t = fminf(
                        1.0f,
                        fmaxf(0.0f,
                              ((midpointX - q.px[i]) * dx + (y - q.py[i]) * dy) /
                                  denominator));
                    depth[count] = q.depths[i] + (q.depths[i + 1] - q.depths[i]) * t;
                }
                lo[count] = a;
                hi[count] = b;
                ++count;
            }
        }
        if (q.stroke) {
            for (int i = 1; i < 3; ++i) {
                const float radius = q.widths[i] * 0.5f;
                const float dy = y - q.py[i];
                if (fabsf(dy) >= radius) {
                    continue;
                }
                const float halfChord = sqrtf(fmaxf(0.0f, radius * radius - dy * dy));
                const float a = fmaxf(horizontalBounds.x, q.px[i] - halfChord);
                const float b = fminf(horizontalBounds.y, q.px[i] + halfChord);
                if (b > a) {
                    lo[count] = a;
                    hi[count] = b;
                    depth[count] = q.depths[i];
                    ++count;
                }
            }
        }
        float ends[10]{};
        for (int i = 0; i < count; ++i) {
            ends[2 * static_cast<std::size_t>(i)] = lo[i];
            ends[2 * static_cast<std::size_t>(i) + 1] = hi[i];
        }
        for (int i = 1; i < 2 * count; ++i) {
            for (int j = i; j > 0 && ends[j] < ends[j - 1]; --j) {
                const float value = ends[j];
                ends[j] = ends[j - 1];
                ends[j - 1] = value;
            }
        }
        float length = 0.0f;
        for (int i = 1; i < 2 * count; ++i) {
            const float middle = (ends[i - 1] + ends[i]) * 0.5f;
            float strength = 0.0f;
            for (int segment = 0; segment < count; ++segment) {
                if (middle >= lo[segment] && middle <= hi[segment]) {
                    strength = fmaxf(strength, depth[segment]);
                }
            }
            length += (ends[i] - ends[i - 1]) * strength;
        }
        return length;
    }

    __device__ float defect_coverage(const DefectPrimitive& q, float x, float y, float sx, float sy) {
        x -= q.cx;
        y -= q.cy;
        if (x + sx * 0.5f + q.softness < q.xmin || x - sx * 0.5f - q.softness > q.xmax ||
            y + sy * 0.5f + q.softness < q.ymin || y - sy * 0.5f - q.softness > q.ymax) {
            return 0.0f;
        }
        // Compact, normalized physical box softness followed by the pixel footprint.
        // Integrating over the clipped shape interval retains unresolved particle area.
        float area = 0.0f;
        for (int soft = 0; soft < 3; ++soft) {
            const float shift = (static_cast<float>(soft) - 1.0f) * q.softness;
            const float bottom = fmaxf(q.ymin, y - sy * 0.5f + shift);
            const float top = fminf(q.ymax, y + sy * 0.5f + shift);
            if (!(top > bottom)) {
                continue;
            }
            float sum = 0.0f;
            for (int row = 0; row < 8; ++row) {
                const float sampleY = bottom + (static_cast<float>(row) + 0.5f) * (top - bottom) / 8.0f;
                for (int col = 0; col < 3; ++col) {
                    const float offset = (static_cast<float>(col) - 1.0f) * q.softness;
                    sum += defect_cross_section(q, sampleY, make_float2(x - sx * 0.5f + offset, x + sx * 0.5f + offset));
                }
            }
            area += sum * (top - bottom) / 72.0f;
        }
        return fminf(1.0f, fmaxf(0.0f, area / (sx * sy)));
    }

    template <int Capacity>
    __device__ float evaluate_defect_family(const JuicerCuda::FilmDefectsPayload& p, int family, const DefectTileFootprint& footprint, DefectPrimitive (&batch)[Capacity]) {
        const bool dust = family == 0 || family == 2;
        const auto& dp = family == 0 ? p.filmDust : p.gateDust;
        const auto& sp = family == 1 ? p.filmScratch : p.gateScratch;
        const float probability = dust ? dp.slotProbability : sp.slotProbability;
        if (!(probability > 0.0f)) {
            return 1.0f;
        }
        const float cw = dust ? dp.cellWidthMm : sp.cellWidthMm;
        const float ch = dust ? dp.cellHeightMm : sp.cellHeightMm;
        const float supportX = dust ? dp.supportXMm + dp.softnessMaxMm : sp.supportXMm + sp.softnessMaxMm;
        const float supportY = dust ? dp.supportYMm + dp.softnessMaxMm : sp.supportYMm + sp.softnessMaxMm;
        const auto& origin = p.origins[family];
        const float stepX = p.sampleStepXMm * footprint.scale;
        const float stepY = p.sampleStepYMm * footprint.scale;
        const float baseX = origin.localXMm;
        const float baseY = origin.localYMm;
        const int x0 = static_cast<int>(floorf((baseX + static_cast<float>(footprint.tileX) * stepX - supportX) / cw));
        const int y0 = static_cast<int>(floorf((baseY + static_cast<float>(footprint.tileY) * stepY - supportY) / ch));
        const int x1 = static_cast<int>(floorf((baseX + static_cast<float>(footprint.tileX + footprint.width) * stepX + supportX) / cw));
        const int y1 = static_cast<int>(floorf((baseY + static_cast<float>(footprint.tileY + footprint.height) * stepY + supportY) / ch));
        const int columns = x1 - x0 + 1;
        const int total = columns * (y1 - y0 + 1) * 2;
        const int lane = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
        const int lanes = static_cast<int>(blockDim.x * blockDim.y);
        float retained = 1.0f;
        const float x = baseX + (static_cast<float>(footprint.pixelX) + 0.5f) * stepX;
        const float y = baseY + (static_cast<float>(footprint.pixelY) + 0.5f) * stepY;
        for (int first = 0; first < total; first += Capacity) {
            const int count = min(Capacity, total - first);
            for (int i = lane; i < count; i += lanes) {
                const int ordinal = first + i;
                const int dx = x0 + (ordinal / 2) % columns;
                const int dy = y0 + (ordinal / 2) / columns;
                batch[i] = build_defect_candidate(p, DefectCandidateAddress{origin.cellX + dx, origin.cellY + dy, family, ordinal % 2, dx, dy});
            }
            __syncthreads();
            for (int i = 0; i < count; ++i) {
                if (batch[i].active) {
                    retained *= 1.0f - batch[i].strength * defect_coverage(batch[i], x, y, stepX, stepY);
                }
            }
            __syncthreads();
        }
        return retained;
    }

} // namespace

namespace {
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    __global__ void apply_film_defects_kernel(
        JuicerCuda::FilmDefectsPayload defects, int width, int height, float* ioC, float* ioM, float* ioY, float* dustTransmittance) {
        __shared__ DefectPrimitive batch[32];
        const int tx = static_cast<int>(blockIdx.x * blockDim.x);
        const int ty = static_cast<int>(blockIdx.y * blockDim.y);
        const int x = tx + static_cast<int>(threadIdx.x);
        const int y = ty + static_cast<int>(threadIdx.y);
        const int fullX = x + defects.roiOffsetX;
        const int fullY = y + defects.roiOffsetY;
        const float dust = evaluate_defect_family(defects, 0, DefectTileFootprint{fullX, fullY, tx + defects.roiOffsetX, ty + defects.roiOffsetY, static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), 1.0f}, batch);
        const float retained = evaluate_defect_family(defects, 1, DefectTileFootprint{fullX, fullY, tx + defects.roiOffsetX, ty + defects.roiOffsetY, static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), 1.0f}, batch);
        if (x < width && y < height) {
            const std::size_t i = static_cast<std::size_t>(y) * width + x;
            if (dustTransmittance) {
                dustTransmittance[i] = dust;
            }
            if (defects.filmScratch.slotProbability > 0.0f) {
                ioC[i] *= retained;
                ioM[i] *= retained;
                ioY[i] *= retained;
            }
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    __global__ void gate_defect_transmittance_kernel(
        JuicerCuda::FilmDefectsPayload defects, float* output, int width, int height) {
        __shared__ DefectPrimitive batch[32];
        const int tx = static_cast<int>(blockIdx.x * blockDim.x);
        const int ty = static_cast<int>(blockIdx.y * blockDim.y);
        const int x = tx + static_cast<int>(threadIdx.x);
        const int y = ty + static_cast<int>(threadIdx.y);
        const float dust = evaluate_defect_family(defects, 2, DefectTileFootprint{x, y, tx, ty, static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), 2.0f}, batch);
        const float scratch = evaluate_defect_family(defects, 3, DefectTileFootprint{x, y, tx, ty, static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), 2.0f}, batch);
        if (x < width && y < height) {
            output[static_cast<std::size_t>(y) * width + x] = dust * scratch;
        }
    }

    // Pitches, dimensions and offsets match the single post-output launch.

} // namespace

extern "C" cudaError_t juicer_cuda_apply_film_defects(
    const JuicerCuda::FilmDefectsPayload* defects,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* dustTransmittance,
    void* cudaStreamOpaque) {
    if (!defects || !densityC || !densityM || !densityY ||
        width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    if (defects->filmDust.slotProbability > 0.0f && !dustTransmittance) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (static_cast<unsigned int>(width) + threads.x - 1) / threads.x,
        (static_cast<unsigned int>(height) + threads.y - 1) / threads.y);
    apply_film_defects_kernel<<<blocks, threads, 0, stream>>>(
        *defects,
        width,
        height,
        densityC,
        densityM,
        densityY,
        dustTransmittance);
    return cudaGetLastError();
}

extern "C" cudaError_t juicer_cuda_build_gate_defect_transmittance_focused(
    const JuicerCuda::FilmDefectsPayload* defects,
    float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
    void* cudaStreamOpaque) {
    if (!defects || !gateTransmittance || gateTransmittanceWidth <= 0 || gateTransmittanceHeight <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaStream_t stream = cudaStreamOpaque
                              ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                              : nullptr;
    dim3 threads(32, 8);
    dim3 blocks(
        (static_cast<unsigned int>(gateTransmittanceWidth) + threads.x - 1) / threads.x,
        (static_cast<unsigned int>(gateTransmittanceHeight) + threads.y - 1) / threads.y);
    gate_defect_transmittance_kernel<<<blocks, threads, 0, stream>>>(
        *defects,
        gateTransmittance,
        gateTransmittanceWidth,
        gateTransmittanceHeight);
    return cudaGetLastError();
}
