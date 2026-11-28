// Explicit maintenance adapter for Fotufilm 87a3f46af093a9e6bf76acf21aca83d3b9e2c600.
// Input is the completed Film-Juicer CDF3 response sampled into three CMY records.
// Run through scripts/generate-film-bank.py with an explicit calibration and destination.
import Foundation
import FotufilmCore

struct Request: Decodable {
    struct DataBlock: Decodable {
        let log_exposure: [Float]
        let density_curves: [[Float]]
    }
    struct Capture: Decodable {
        let stock: String
        let polarity: String
    }
    struct Calibration: Decodable {
        let strength: Float
        let size_mm: Float
        let layer_weights: [Float]
        let base_density_offset: Float
        let fog_density: Float
        let read_density: GranularityReadDensity
        let radius_span: Float
        let sublayer_shares: [Float]
        let coating_density_scale: Float
    }
    let data: DataBlock
    let capture: Capture
    let calibration: Calibration
}

let arguments = CommandLine.arguments
precondition(arguments.count >= 3, "usage: graincapture REQUEST_JSON OUTPUT_DIRECTORY [FOOTPRINT ...]")
let requestedLabels = Array(arguments.dropFirst(3))
let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
let calibration = request.calibration
let reversal = request.capture.polarity == "positive"
precondition(reversal || request.capture.polarity == "negative", "unexpected capture polarity")
let response = request.data
precondition(response.density_curves.count >= 2 &&
             response.density_curves.count == response.log_exposure.count &&
             response.density_curves.allSatisfy { $0.count == 3 }, "expected one CMY row per exposure")
let offset = calibration.base_density_offset
let curves = try (0..<3).map { channel -> CharacteristicCurve in
    let raw = response.density_curves.map { $0[channel] }
    precondition(zip(raw, raw.dropFirst()).allSatisfy { reversal ? $0.0 >= $0.1 : $0.0 <= $0.1 },
                 "CDF3 channel \(channel) does not follow the capture polarity")
    // Fotufilm's reversal stock expects increasing internal curves. Reflect
    // the density about the endpoint sum, then keep its reversal flag set.
    let densities = raw.map { (reversal ? raw[0] + raw[raw.count - 1] - $0 : $0) + offset }
    return CharacteristicCurve(dMin: densities[0], gamma: 1, toe: -3, toeWidth: 1,
                               shoulder: 4, shoulderWidth: 1,
                               sampled: try SampledCharacteristicCurve(logExposure: response.log_exposure,
                                                                        density: densities))
}
let stock = FilmStock(name: request.capture.stock,
                      sensitivity: [[1,0,0],[0,1,0],[0,0,1]], curves: curves,
                      couplerInhibition: [[0,0,0],[0,0,0],[0,0,0]], couplerDiffusionMM: 0,
                      grainStrength: calibration.strength, grainSizeMM: calibration.size_mm,
                      grainLayerWeights: calibration.layer_weights,
                      crystalGrainPopulation: CrystalGrainPopulation(radiusSpan: calibration.radius_span,
                          sublayerShares: calibration.sublayer_shares,
                          coatingDensityScale: calibration.coating_density_scale),
                      grainFogDensity: calibration.fog_density, granularityReadDensity: calibration.read_density,
                      halationStrength: [0,0,0], paperCurve: curves[0], isReversal: reversal)
let grain = FilmGrain(stock: stock)
let frozenDirectory = ProcessInfo.processInfo.environment["FROZEN_FILM_BANK"]
let tiles = try frozenDirectory.map { try FilmGrain.Tiles.captureFrozen(directory: URL(fileURLWithPath: $0)) } ?? grain.tiles()
// Cross-check the extraction against the unmodified public upstream sampler at its origin.
if frozenDirectory == nil {
let checkGross: [Float] = (0..<3).map { (tiles.dMin[$0] + tiles.dMax[$0]) / 2 }
let input = ImageBuffer(width: 1, height: 1, planes: checkGross.map { [$0] })
let output = grain.apply(to: input, pxPerMM: 1000, seed: 1)
let checkPitch: Float = (1 / Float(1000)) / FilmGrain.tileTexelMM
for r in 0..<3 {
    let delta = grain.captureProbe(tiles: tiles, pitch: checkPitch, footprint: checkPitch,
                                   x: 0, y: 0, seed: 1, record: r, gross: checkGross[r])
    precondition(output.planes[r][0] == checkGross[r] + delta, "upstream origin mismatch")
}
}
let destination = URL(fileURLWithPath: arguments[2], isDirectory: true)
func writeFloats(_ values: [Float], _ name: String) throws {
    var bytes = Data(capacity: values.count * 4)
    for value in values {
        var word = value.bitPattern.littleEndian
        withUnsafeBytes(of: &word) { bytes.append(contentsOf: $0) }
    }
    try bytes.write(to: destination.appendingPathComponent(name))
}
if frozenDirectory == nil {
try writeFloats(tiles.packed(), "sums.f32")
try writeFloats((0..<3).flatMap { r in
    [tiles.dMin[r], tiles.dMax[r], offset] + tiles.levels[r].map(\.meanLight)
}, "records.f32")
}
let tableLabels = frozenDirectory == nil ? Array(Set(["1", "9.375", "32"] + requestedLabels)).sorted() : requestedLabels
for label in tableLabels {
    guard let footprint = Float(label), footprint.isFinite, footprint > 0 else { fatalError("invalid footprint") }
    let table = grain.pitchTables(tiles, footprint: footprint)
    try writeFloats((0..<3).flatMap { table.meanDensity[$0] + table.blendCorrelation[$0] + [1] },
                    "footprint-\(label).f32")
}
if frozenDirectory == nil {
var lines: [String] = []
let coordinates = [(-4096,-4096), (-129,-128), (-65,-64), (-1,-1), (0,0),
                   (15,16), (63,64), (127,128), (255,256), (4095,4096)]
for (pitch, footprint): (Float, Float) in [(1,1), (9.375,9.375), (32,32), (128,32)] {
    for seed: UInt32 in [0, 1, 1514081821, 4294967295] {
        for (x,y) in coordinates {
            for r in 0..<3 {
                for fraction: Float in [-0.1, 0, 0.03125, 0.5, 0.9375, 1, 1.1] {
                    let gross = tiles.dMin[r] + fraction * (tiles.dMax[r] - tiles.dMin[r])
                    let net = gross - offset
                    // Reconstruct gross exactly as the consuming implementation does.
                    let delta = grain.captureProbe(tiles: tiles, pitch: pitch, footprint: footprint,
                                                   x: x, y: y, seed: seed, record: r,
                                                   gross: net + offset)
                    lines.append("\(pitch) \(footprint) \(x) \(y) \(seed) \(r) \(net) \(delta)")
                }
            }
        }
    }
}
try (lines.joined(separator: "\n") + "\n").write(to: destination.appendingPathComponent("reference.tsv"),
                                                    atomically: true, encoding: .utf8)

}

// Independent G2 probes on the exact frozen coating and newly measured tables.
if frozenDirectory != nil {
    var lines: [String] = []
    let coordinates = [(-4096,-4096), (-4096,4096), (-129,-128), (-17,197), (-1,-1),
                       (0,0), (15,16), (127,128), (4095,4096), (4096,-4096)]
    for label in requestedLabels {
        guard let footprint = Float(label), footprint.isFinite, footprint > 0 else { fatalError("invalid footprint") }
        for seed: UInt32 in [0, 1, 1514081821, 4294967295] {
            for (x,y) in coordinates {
                for r in 0..<3 {
                    for fraction: Float in [-0.1, 0, 0.03125, 0.5, 0.9375, 1, 1.1] {
                        let gross = tiles.dMin[r] + fraction * (tiles.dMax[r] - tiles.dMin[r])
                        let net = gross - offset
                        let delta = grain.captureProbe(tiles: tiles, pitch: footprint, footprint: footprint,
                                                       x: x, y: y, seed: seed, record: r, gross: net + offset)
                        lines.append("\(footprint) \(footprint) \(x) \(y) \(seed) \(r) \(net) \(delta)")
                    }
                }
            }
        }
    }
    try (lines.joined(separator: "\n") + "\n").write(to: destination.appendingPathComponent("reference-g2.tsv"),
                                                        atomically: true, encoding: .utf8)
}
