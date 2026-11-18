// Adapted from Fotufilm FilmGrainTiles.swift at 87a3f46af093a9e6bf76acf21aca83d3b9e2c600.
// Copyright 2026 MUAStudio Inc.; SPDX-License-Identifier: Apache-2.0.
// Upstream license: https://github.com/muastudio/fotufilm/blob/87a3f46af093a9e6bf76acf21aca83d3b9e2c600/LICENSE
// Compile as part of FotufilmCore, so this invokes the pinned placement, rectangle and blend code.
import Foundation

extension FilmGrain {
    public func captureProbe(tiles: Tiles, pitch: Float, footprint: Float, x: Int, y: Int,
                             seed: UInt32, record r: Int, gross: Float) -> Float {
        let levels = tiles.levels[r]
        let table = pitchTables(tiles, footprint: footprint)
        let lo = tiles.dMin[r], hi = tiles.dMax[r]
        let t = min(max((gross - lo) / max(hi - lo, 1e-6), 0), 1) * Float(Self.tileLevels - 1)
        let k = min(Int(t), Self.tileLevels - 2), w = t - Float(k)
        let block = Self.blockTexels, seam = Self.seamTexels
        var total = 0.0
        for layer in 0..<Self.tileLayers {
            let shift = Self.layingShift(layer)
            let cx = (Double(x) + 0.5) * Double(pitch) + shift
            let cy = (Double(y) + 0.5) * Double(pitch) + shift
            let x0 = cx - 0.5 * Double(footprint), x1 = x0 + Double(footprint)
            let y0 = cy - 0.5 * Double(footprint), y1 = y0 + Double(footprint)
            let bx0 = Int(((cx - seam) / block).rounded(.down))
            let bx1 = Int(((cx + seam) / block).rounded(.down))
            let by0 = Int(((cy - seam) / block).rounded(.down))
            let by1 = Int(((cy + seam) / block).rounded(.down))
            var value = 0.0, norm = 0.0
            for by in by0...by1 {
                let wy = Self.blockWeight(cy, by)
                for bx in bx0...bx1 {
                    let weight = Self.blockWeight(cx, bx) * wy
                    guard weight > 0 else { continue }
                    let (a, b) = Self.blockLight(levels[k], levels[k + 1], x0, x1, y0, y1,
                                                  bx: bx, by: by, seed: seed, layer: layer, record: r)
                    let da = -log10(max(a, 1e-9)) - table.meanDensity[r][k]
                    let db = -log10(max(b, 1e-9)) - table.meanDensity[r][k + 1]
                    value += weight * Double(Self.blend(da, db, w, table.blendCorrelation[r][k]))
                    norm += weight * weight
                }
            }
            total += value / max(norm, 1e-12).squareRoot()
        }
        return Float(total / Double(Self.tileLayers).squareRoot())
    }
}


extension FilmGrain.Tiles {
    // Rehydrate the frozen bank for independent footprint-table measurements.
    public static func captureFrozen(directory: URL) throws -> FilmGrain.Tiles {
        func floats(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: directory.appendingPathComponent(name))
            precondition(data.count % 4 == 0)
            return stride(from: 0, to: data.count, by: 4).map { i in
                let word = UInt32(data[i]) | UInt32(data[i + 1]) << 8 | UInt32(data[i + 2]) << 16 | UInt32(data[i + 3]) << 24
                return Float(bitPattern: word)
            }
        }
        let records = try floats("records.f32"), sums = try floats("sums.f32")
        let entries = (FilmGrain.tileSide + 1) * (FilmGrain.tileSide + 1)
        precondition(records.count == 3 * 20 && sums.count == 3 * FilmGrain.tileLevels * entries)
        let levels = (0..<3).map { r in
            (0..<FilmGrain.tileLevels).map { k in
                let start = (r * FilmGrain.tileLevels + k) * entries
                return FilmGrain.TileLevel(sums: Array(sums[start..<(start + entries)]), meanLight: records[r * 20 + 3 + k])
            }
        }
        return FilmGrain.Tiles(levels: levels, dMin: (0..<3).map { records[$0 * 20] },
                              dMax: (0..<3).map { records[$0 * 20 + 1] })
    }
}
