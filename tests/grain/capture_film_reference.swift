// Explicit maintenance adapter for Fotufilm 87a3f46af093a9e6bf76acf21aca83d3b9e2c600.
// Input is the completed Film-Juicer CDF3 response sampled into three CMY records.
// The generated stock-derived bank stays in ignored evidence.
import Foundation
import FotufilmCore

struct Profile: Decodable {
    struct DataBlock: Decodable {
        let log_exposure: [Float]
        let density_curves: [[Float]]
    }
    let data: DataBlock
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else { fatalError("usage: capture_film_reference RESPONSE_JSON OUTPUT_DIRECTORY") }
let source = try JSONDecoder().decode(Profile.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
precondition(source.data.density_curves.count == source.data.log_exposure.count &&
             source.data.density_curves.allSatisfy { $0.count == 3 }, "expected one CMY row per exposure")
for channel in 0..<3 {
    let values = source.data.density_curves.map { $0[channel] }
    precondition(zip(values, values.dropFirst()).allSatisfy { $0.0 <= $0.1 }, "CDF3 response is not monotone")
}
let offset: Float = 0.1
let curves = try (0..<3).map { channel -> CharacteristicCurve in
    let densities = source.data.density_curves.map { $0[channel] + offset }
    return CharacteristicCurve(dMin: densities[0], gamma: 1, toe: -3, toeWidth: 1,
                               shoulder: 4, shoulderWidth: 1,
                               sampled: try SampledCharacteristicCurve(logExposure: source.data.log_exposure,
                                                                        density: densities))
}
let stock = FilmStock(name: "Portra 400 response, synthetic G1 grain",
                      sensitivity: [[1,0,0],[0,1,0],[0,0,1]], curves: curves,
                      couplerInhibition: [[0,0,0],[0,0,0],[0,0,0]], couplerDiffusionMM: 0,
                      grainStrength: 0.010, grainSizeMM: 0.005, grainLayerWeights: [0.7,1,1.35],
                      crystalGrainPopulation: CrystalGrainPopulation(radiusSpan: 6,
                          sublayerShares: [0.25,0.25,0.25,0.25], coatingDensityScale: 1),
                      grainFogDensity: 0.03, halationStrength: [0,0,0], paperCurve: curves[0])
let grain = FilmGrain(stock: stock)
let tiles = grain.tiles()
// Cross-check the extraction against the unmodified public upstream sampler at its origin.
let checkGross: [Float] = (0..<3).map { (tiles.dMin[$0] + tiles.dMax[$0]) / 2 }
let input = ImageBuffer(width: 1, height: 1, planes: checkGross.map { [$0] })
let output = grain.apply(to: input, pxPerMM: 1000, seed: 1)
let checkPitch: Float = (1 / Float(1000)) / FilmGrain.tileTexelMM
for r in 0..<3 {
    let delta = grain.captureProbe(tiles: tiles, pitch: checkPitch, footprint: checkPitch,
                                   x: 0, y: 0, seed: 1, record: r, gross: checkGross[r])
    precondition(output.planes[r][0] == checkGross[r] + delta, "upstream origin mismatch")
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
try writeFloats(tiles.packed(), "sums.f32")
try writeFloats((0..<3).flatMap { r in
    [tiles.dMin[r], tiles.dMax[r], offset] + tiles.levels[r].map(\.meanLight)
}, "records.f32")
for (label, footprint): (String, Float) in [("1",1), ("9.375",9.375), ("32",32)] {
    let table = grain.pitchTables(tiles, footprint: footprint)
    try writeFloats((0..<3).flatMap { table.meanDensity[$0] + table.blendCorrelation[$0] + [1] },
                    "footprint-\(label).f32")
}
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
