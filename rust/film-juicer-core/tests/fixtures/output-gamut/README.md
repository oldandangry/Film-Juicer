# Native output-gamut captures

These are immutable Film-Juicer native characterizations at `d033f27`, captured
before A9 translation in all four supported presets. They are not spektrafilm
reference tables. All four presets select `extra-0.json` because the original
extra captures decode to the same ordered rows and exact words. The manifest
retains each preset’s original native digests and capture, source, tool and
compiler provenance. Both byte-pinned extra captures are available in the
[tagged output gamut fixtures](https://github.com/oldandangry/Film-Juicer/tree/archive/spectral-cutover-2026-11/tests/ffi/fixtures/output_gamut).

`records` contain all nine original source matrices/whites, completed transforms,
305-byte transform and 40-byte contract streams, bounded derivative/Horner/libm
cases and sampled cell cubics. `tables` retain every 46,080 raw f32 word per row,
lightness-major `[64][720]`, little endian. `extra` retain normalization, strict
root/deduplication, slack-neighbor and unpublished native failure cases. Synthetic
rows were declared before candidate implementation and remain separate from
unchanged registry literals. Comparisons are exact for the selected preset.

Ordinary tests never regenerate these bytes. Candidate generation is an explicit
maintenance operation under `out/validation/`; expectations require separate
review. Family-scoped Git attributes preserve the active fixture bytes on both
hosts.
