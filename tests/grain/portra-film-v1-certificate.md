# Portra Film v1 density certificate

The certificate covers the immutable Portra 400 bank and the two negative
routes. [`Raster::sampling`](../../rust/film-juicer-core/src/film/grain/film_v1.rs)
selects the table pitch. For calibrated sampling it uses `1000 × format / (rendered
long edge × Size)`, with the realized calibrated raster where applicable. For
integrated sampling the denominator uses the canonical Full extent. The sampling
pitch is rounded once to Float32. The proof domain includes every positive
Float32 word from `0.244140625` through `240000` (166,748,161 words), the full
range checked by the [table owner](../../rust/film-juicer-core/src/assets/grain.rs).
The rendered pitch has its own 0.25-texel lower limit. The proof uses gain 16
because `exp2(Amount) / Size` is at most 16. This covers the coupling between the
pitch and gain through Size. Widening the gain widens the bright and dark bounds
from their clean values. The signed spectral floor uses each wavelength's
brightest box corner, selecting the upper density wherever its dye is negative.

## Placement and projection

[`continuous_footprint_certificate`](../../rust/film-juicer-core/src/film/density/portra.rs)
recomputes the signed extrema and paired bright deficits from the frozen SAT
words. For subtexel placement, the Float64 weighted texels form a convex light
combination. For rectangles, the proof checks fractional translations between
integer cells and includes the Float32 aperture and light arithmetic allowances.
Large apertures in the original enclosure are divided into equal squares. Their
positive, normalized areas keep the light within the subrectangle bounds. The
periodic enclosure uses the exact zero whole-period totals. Each square is
`n × 256 ± remainder`, with signed residual strips and a shared residual corner.
The [periodic bank test](../../rust/film-juicer-core/src/film/density/portra/periodic_extrema.rs)
recomputes all 16 × 51 stored strip and corner entries. At widths at least 1536,
the relative residual is bounded by `1476 / (256 × width) + 4720 / width²`.

The [CUDA sampler](../../src/Cuda/JuicerCudaFilmGrain.cuh) scales a weighted
placement sum by its Euclidean normalization. Within an axis at most one of
the staggered layings straddles a seam. The maximum sum of the two layings'
normalized weight sums is 3, from 2 + 1 in the same laying, or √2 + √2 across
different layings. The bound uses 3.012 to cover arithmetic rounding. The
adjacent-level normalization uses the represented correlation `rho` and the
quadratic `(1 − t)² + t² + 2 × rho × t × (1 − t)`, where `t` is the level
fraction. Fractional time adds at most √2, cancelling the √2 in the laying
normalization. These bounds are independent of the PCG placement and time seeds.
Both extrema are carried through the exposure-level cuts, signed grain and
Colour projection. The tight enclosure treats each light endpoint as affine in
Colour and bounds the union of the signed density boxes at the ends of each
Colour interval. This covers near-endpoint words and changes in the limiting
dye, channel or wavelength. The retained and periodic enclosures bound the
projection using a nonnegative combination of transmission ratios. Integrated
proxies average relative transmission using nonnegative overlaps of Full cells.
The 0.002 D pad covers density addition, projection and the proxy conversion
back to density, followed by outward Float32 rounding.

## Calibration and admission

The [continuous calibration verifier](../../rust/film-juicer-core/src/film/density/portra/calibration_interval.rs)
uses the selected table's calibration lattice and PCG seeds. It splits an
interval at changes in the monotone block indices, period translations, and
corner cells. In a fixed cell the SAT integral is quadratic in width, so its
area average is quadratic in inverse width. The Bernstein coefficients enclose
the entire interval. The light casts and density transform preserve monotone
bounds. Centering is an orthogonal projection and can only reduce the error
radius. An interval field with midpoint norm `n` and error radius `e < n`
deviates by at most `asin(e / n)` from its midpoint direction. The angle between
each pair of adjacent levels, plus both angular errors, gives the conservative
correlation bound. At `e ≥ n` the interval carries no useful correlation bound
and can be split or rechecked from the production constructor.

[`Bounds::portra_film_v1`](../../rust/film-juicer-core/src/film/density.rs)
tries the original signed enclosure first. A calibrated table from 1 through
37.5 must also meet the production scaled mean and adjacent-correlation limits.
The tighter placement certificate covers the bands through 10. For widths
above 255 the periodic allowance supplies a second enclosure. After a retained
box passes, its original represented words remain authoritative. The proof checks
for a passing signed box throughout each interval, applying the 64 D span and
−30 D spectral floor. Inconclusive intervals split at a Float32 word boundary.
Their singleton rechecks use the represented table from the production owner.
Periodic widths can also be rechecked exactly, including quantized constant
fields whose adjacent correlation follows the table's zero-variance branch.

## Rechecking the domain

The runner and receipt instructions in this section reproduce the tagged
Task 8 certificate. Cleanup uses the [contributor checks](../../CONTRIBUTING.md#required-checks).
The mathematical domain, parent ranges, exact inputs and accepted numerical
bounds continue to define the certificate.

Workspace nextest runs the placement, periodic bank and calibration checks.
For a maintenance run, put `portra-certificate-request.json` in its output
directory and point `FJ_PORTRA_CERTIFICATE` at that directory.

```sh
FJ_PORTRA_CERTIFICATE=/absolute/path/to/validation-directory \
  cargo nextest run --locked -p film-juicer-core \
  --test-threads 1 -E 'test(calibration_intervals_enclose_executed_words_and_signed_guards)'
```

The request has schema `portra-certificate-request-v1`, a `mode`, and ordered,
disjoint `parents`. Each parent supplies inclusive `first` and `last` Float32
bit patterns as JSON integers. The modes have these obligations.

- `nonconstant` covers parent intervals through 10 texels. For each calibration
  record and level, it finds two sample light intervals with a positive gap
  after the production residual cast, Float32 light addition and positive
  floor. Per-sample Bernstein bounds cover the whole width interval. Samples
  come from the production lattice and seed. A sample budget can make a
  wider interval inconclusive, so the routine bisects as needed and reports
  the smallest certified gap. At a single word it can use the entire lattice.
  A positive gap at every word rules out the historical constant-light
  shortcut on any interval containing that word. The historical mean,
  correlation and all-Colour safeguard bounds then retain their stated scope.
- `interval` checks the signed guard coverage over each complete parent
  interval. This mode records limiting spans, wavelengths and singleton
  rechecks. The portable full sweep is expensive, especially in the periodic
  region. For full-domain coverage the adjacent ranges must account for all
  166,748,161 words.
- `exact` consumes `--portra-flags /path/to/flags.u32le`. Concatenate the
  original little-endian `u32` flag bytes in parent order. Each parent also
  gives the `words` count and `flags_sha256` of its segment. The constructor
  checks each selected word and the report retains the sorted words for every
  parent. Duplicate or out-of-parent words, extra bytes and wrong digests
  fail the request. Keep the corresponding parent proof with the receipt.

The original verifier copied the request and exact flag bytes into its run
bundle before execution. Its Rust record identifies those files and the terminal
report. Use the tagged guide to inspect retained migration evidence and verify
its digests, input membership, coverage and bounds.
[Tagged certificate guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/grain/portra-film-v1-certificate.md)

The signed density and enlarger witnesses in the core workspace tests, the held high-gain
GPU scanner XYZ reference, and the staged and linked CUDA cohorts exercise the
result. The all-word argument supplies the calibration and interval coverage
between those checks. Delivered RGB and historical cubic comparisons retain the
method and geometry recorded by their own witnesses.
