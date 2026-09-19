# Scatter and halation tests

The host GoogleTest target checks authored-control validation, recipe and hash
identity, profile classification, executable descriptor dispatch, all-off
behavior, and public-resource bootstrap without loading the CUDA driver. The
integration executable links the production CUDA boundary and keeps preparation,
operator, route, carrier, zero-work, lifecycle, and benchmark processes isolated.

`fixtures/manifest.json` and `scatter_halation_reference.bin` are an external
reference cohort generated from the unmodified spektrafilm halation operator at
revision `48645a2b4bf58c20b6a3b75c8022d0f462db754a`. The manifest records the
operator and generator hashes, environment, shapes, semantic RGB order, byte
encoding, and per-plane checksums. Its historical `hostGate1Identity` name is
retained as fixture provenance, not as an active test-stage name.

The committed manifest and binary remain byte-identical to the pre-migration
cohort:

- manifest SHA-256:
  `f59f3edaffe8560f48c40dc6709947bcc244ee9995bf95e48be7fd958edd60f6`
- binary SHA-256:
  `b68f4704335263c3d95fd4fcb9152d54aca227264a8ee3081a87edae740f6cbe`

The retained comparison bounds are operator-specific. FIR rows require maximum
absolute error and impulse L1 error at most `2e-5`. YVV rows require mean error
at most `2e-4`, maximum error at most `2e-3`, and impulse L1 error at most
`2e-3`. Combined and downstream rows use source-normalized bounds of `3e-4`
mean and `3e-3` maximum; combined impulse rows also bound L1 at `3e-3`.
Non-finite comparisons fail.

The upstream spektrafilm software used for reference generation is GPLv3; its
source identity is recorded in the manifest. The public Film-Juicer test code
and generated cohort are distributed with this repository under its
GPL-3.0-only license. No spektrafilm profile or LUT asset is embedded in this
cohort.

Ordinary CTest execution reads the fixture but never modifies or regenerates
it. Reference regeneration remains separate private evidence work until its
generator and pinned environment form a complete public maintenance unit.
