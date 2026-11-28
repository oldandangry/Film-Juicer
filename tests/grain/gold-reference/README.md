# Gold 200 Grain v2 reference

These are independent Fotufilm Swift sampler statistics and probes for the
frozen [Kodak Gold 200](../../../Resources/grain/gold-200/NOTICE.md) coating.
The Film-Juicer CDF3 response, historical capture parameter values and pinned
Swift 6.1.2 release produced the original G2 hashes below. The maintained
generator reproduced all six G2 files and captured the extra 37.5-texel
footprint used by the aperture tests. [Bank authoring](../../../docs/grain-v2-bank-authoring.md)

| File | SHA-256 |
| --- | --- |
| Bank `records.f32` | `1fc0a39ab8d0e46f8350c2a2b66934cc922f42e348ac3a6e0d4814b1329fa54d` |
| Bank `sums.f32` | `0dabacd1b8759604cfc4ad824c932c1650c1ec9dc75e8909781fcd43a13acf8f` |
| `footprint-1.f32` | `d8bb3bb0e7a22379990fe3bb7286612a7cd5f676210f303e4a0375b8d4048be3` |
| `footprint-9.375.f32` | `529eac3483f0bf412d4efe5b1e7bec753f975d9bb738d9408565fa7ac5cce85f` |
| `footprint-32.f32` | `f399c4f1500ed2df2e5466107b5e06e74489a4e38284fc90bca91ea6d6852b7c` |
| `reference.tsv` | `2e6e86ca9bfc18786dbfded30a6f0fce7fb23cf18c31c91c3067ce78581bfa37` |
| `footprint-37.5.f32` | `08468dbfe4e69042f1ecc8cee1e8d6adcc5f10d85aee2cb031091870861ef182` |

Each table has 17 mean densities, 16 adjacent-level correlations and one padding
correlation for each C/M/Y record. Probe columns are pitch, footprint, X, Y,
seed, record index, net density and signed density delta. The independent
scalar sampler checks all 3,360 rows. Production CUDA aperture cases exercise
both routes and both sampling choices with fixed seeds and fractional time.
[Aperture and probe tests](../film_apertures_test.cpp)

Gold's placement bands, paired light deficits, scaled density deltas and
wide periodic allowances were rounded outward from the frozen Gold SATs.
The maintained certificate tests independently recompute the enclosures from
the bank. Gold uses its measured table statistics for the selected footprint
and its own CDF3 maxima, spectral dyes and baseline. The existing Portra
certificate explains the shared enclosure mathematics.
[Density code](../../../rust/film-juicer-core/src/film/density/grain_v2.rs) ·
[Portra certificate](../portra-grain-v2-certificate.md)
