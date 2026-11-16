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
