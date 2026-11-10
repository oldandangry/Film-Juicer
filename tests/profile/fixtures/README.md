# Density and hash fixtures

These are immutable product-contract fixtures for the safe core. Ordinary Cargo
tests (also scheduled by `Rust.Core` in CTest) read these files and the shipped
profiles only. No private tooling, native probe, external checkout or reference
environment is needed. Tests never regenerate the fixtures.

On 30 September 2026 the owner approved direct Rust density evaluation using
exactly `libm = "=0.2.15"`, `libm::erfcf`, and its default `arch` feature after
visual inspection found no perceptible image or scopes/histogram difference.
This is a permanent, bounded exception to the migration's native bit-equality
rule: the characterized density differences and resulting curve/profile/recipe
hashes, glare seeds, grain/discrete decisions and rendered differences are
accepted. Hash algorithms, actual owned bits, computed-result validity, FP32 order,
f64 model precision and determinism remain required. No universal tolerance is
defined. The separately qualified standard f32 sensitivity math is unaffected.

`manifest.json` pins the source revision, resource digest, upstream provenance,
dependency checksum, capture hashes, extraction scope and payload SHA-256 values.
The 29 September captures precede this implementation. Native probes called the
actual `ProfileAssets.cpp`, `RenderRecipe.cpp` and `Hash.h` owners. Native replay
hashed supplied Rust sample bits through those owners; it did not supply density
expectations. Linux used GCC 13.3/glibc 2.39; Windows used ClangCL 22.1.3,
MSVC 14.51 and its Debug/Release CRTs. Both separate Rust installations used
Rust/Cargo 1.98.1 (LLVM 22.1.8), respectively GNU and MSVC x86_64 targets.

The source paths in the manifest are archival provenance identifiers, not test
inputs. Adoption copied their bits without running density, gamma or identity
algorithms. All four approved Rust captures agreed exactly; all four original
native captures agreed on historical model/admission/validity, and all four native replays
agreed on candidate-bit identities. Historical captures remain unchanged.

## Contents and encoding

- `density.json`: 138 cases, with deduplicated authored f64 axes. Integer model
  and axis entries are IEEE-754 **f64 bits**, never decimal sample values. Model
  matrices are `[channel][layer]`. Gamma-adjusted models are already adjusted
  frozen inputs; these tests do not implement or authorize a gamma API.
- `density-samples.bin`: 19,247 records of exactly 49 bytes each: one validity
  byte (`0` or `1`), followed by twelve little-endian IEEE-754 f32 bit patterns.
  The first three values are channel totals; the next nine are layers in
  `[layer][channel]` order. Invalid records contain zero padding, not valid zero
  samples. Case `sample_start` indexes records, not bytes. All 19,235 valid
  samples are exact approved-Rust expectations; 12 historical failures have no
  sample payload. Their current interpretation is specified below.
- Sixteen historically rejected coefficient cases have no sample records.
  Construction now retains all their authored f64 bits without coefficient
  admission. The private raw evaluator also sampled these models for diagnosis;
  those diagnostic values supply no approved sample expectations here.
- Each complete case's `identities` copies total, layer and raw print-curve
  identities from **independent native replay over the approved Rust bits**.
  Tests compose only the new primitives, preserving total sample/channel order,
  layer/channel/sample order and the raw u64-count/f32-axis/f32-total encoding.
  Final profile tokens, normalization and recipe/seed APIs remain later work.
- Eighty-four native witnesses retain the first differing row for each of the
  28 shipped baseline models on Linux, Windows Debug and Windows Release.
  Linux Debug/Release are identical. These are independent characterization,
  explicitly distinct from the approved Rust expectations, with no error bound.
- `hash.json`: sixteen independent native finite/NaN-mask vectors, including
  lengths 0, 1, 3, 63, 64, 65, 128 and 129, infinities, signed zero and NaN
  payloads. Expected entries are finite hash (native zero failure sentinel),
  value hash, NaN-mask hash and combined hash. Rust returns a typed error for a
  nonfinite finite-only input. Existing `tests/hash/hash_contract_test.cpp`
  additionally supplies byte/word-order and mask-boundary constants.

Coverage includes every authored sample of all 28 shipped models, all eight
print models at gamma endpoints, 1.1 and adjacent-to-one values, frozen sigma
floor inputs, both polarities, signed zero, narrowing/nonfinite/overflow cases,
rounding/saturation tails, and a variable axis with duplicate entries. Five
gamma-admission-only cases are listed as excluded in the manifest; they belong
to the later gamma API. No expectations were produced by the implementation
under test, and no old native identity is assigned to changed Rust samples.

## Current trusted-resource acceptance

The historical payloads and manifest remain unchanged. `supported` and validity
bytes describe the original capture, not current scientific admission rules.
All 19,235 frozen valid samples and all 119 replay identity compositions retain
their exact expectations. All sixteen formerly rejected models are constructed
and checked for exact retained coefficient bits, without inventing sample oracles.

Scalar exposures now reach the unchanged f64-to-f32 cast and CDF arithmetic.
For `exposure/narrowing-special`, zero-based rows 0 and 1 succeed with positive
zero in every layer and total. Rows 12, 13 and 14 succeed with `1.0f32` in every
layer and `3.0f32` in every channel total. These exact endpoint expectations
follow from the frozen zero centers, unit amplitudes/sigmas and negative polarity;
invalid-record zero padding is not an expected sample. Row 15 (NaN) fails with
`NonfiniteLayer { channel: 0, layer: 0 }` during arithmetic. Rows 0–2 of both
`overflow/total-positive` and `overflow/total-negative` retain
`NonfiniteTotal { channel: 0 }`. The previously admitted 19,247-row cohort now
has 19,240 successes and seven computed failures. Every historical failure row
has an explicit current assertion.

Focused tests separately cover unusual coefficients and scalar exposures using
exact CDF endpoints, the zero-argument half response, and actual computed-error
boundaries. They introduce no fixture regeneration, tolerance or gamma API.
