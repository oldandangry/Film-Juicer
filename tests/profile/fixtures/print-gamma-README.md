# Print-gamma fixtures

These immutable product-contract fixtures extend the existing density contract.
Ordinary tests use public resources and fixtures only. `print-gamma-manifest.json`
records source hashes, the accepted density exception, extraction and scope.

`print-gamma.json` adopts 48 bundled cases (eight profiles at 0.5, 1, 1.1, 2 and
both f64 neighbors of 1) and 30 synthetic floor cases. `source_model` contains
unadjusted f64 coefficient bits from the independent 29 September input manifest.
Case IDs refer to the existing `density.json` adjusted coefficient bits and
`density-samples.bin` approved libm 0.2.15 samples; neither earlier file changes.
All four native Debug/Release replay captures agree on these adjusted models and
identities. No gamma, sampler or identity algorithm ran during their extraction.

`downstream` adopts fixed-field native recipe/glare identity replay over the
approved Rust sample bits. The outer asset/normalized identities are the replay's
fixed inputs for that case, rather than the selected baseline profile token.
Tests insert the actual new curve hash and authored gamma into this composition.
This qualifies byte order and the curve/gamma dependency through recipe and glare
seed identities; it does not port a recipe or qualify production rendering.

Seventeen analytical cases cover negative/zero/underflow sigmas, oversized centers
and sigmas, and negative amplitudes through complete-profile gamma sampling.
`model`, `axis` and `gamma_bits` use IEEE f64 bits. `totals` use exposure-major CMY
IEEE f32 bits; `print_hash` uses u64 count then raw little-endian f32 axis and total
bytes, preserving zero signs. Expected samples follow exact Gaussian endpoints or
the half response and exactly representable sums. An independent Python byte
packer/FNV calculation fixed these expectations before C5 implementation, without
calling the Rust core or libm. Negative sigmas reverse the baseline response and
are floored only on the non-unit branch. Large authored exposures narrow to signed
infinity and still produce valid endpoints for these selected models.

Additional tests cover actual computed layer/total failures, finite positive gamma
outside the recipe's control range, invalid gamma, raw signed-zero/count/axis
identity, changed samples/polarity, source immutability, and preserved reuse for
unrelated spectral metadata. Private unit tests inspect exact adjusted bits,
including NaN comparison behavior, and inject one bounded capacity failure without
exhausting RAM. There is no promise that every unusual model computes successfully.
