# Author a Grain v2 bank

The maintenance command lists capture stocks through the same catalog as the
effect. It evaluates the selected profile's CDF3 curves at its Float32 exposure
knots, at gamma 1 before development normalization or DIR. The response is in
C/M/Y order. For a positive stock the adapter reflects each record about the
sum of its endpoints for Fotufilm's increasing internal curve and sets its
reversal flag.

Use Ubuntu 24.04 x86-64, Python 3.13, the repository's pinned Rust, and Swift 6.1.2.
Verify the Ubuntu Swift archive with its detached signature and the Swift 6.x
release key before extraction. Check out Fotufilm at the exact revision below.
The command builds only the explicit maintenance tool, in a scratch directory
that it creates. Halide is disabled. The product build, fixture consumers and
the plug-in read frozen assets. [Swift tarball verification](https://www.swift.org/install/linux/tarball/) ·
[Pinned Fotufilm revision](https://github.com/DhaliwalX/fotufilm-engine/tree/87a3f46af093a9e6bf76acf21aca83d3b9e2c600)

```sh
scratch=$(mktemp -d)
python3.13 scripts/generate-film-bank.py prepare \
  --fotufilm /path/to/fotufilm-engine \
  --swift /path/to/swift-6.1.2-RELEASE-ubuntu24.04/usr/bin/swift \
  --output "$scratch/tool"
python3.13 scripts/generate-film-bank.py list
python3.13 scripts/generate-film-bank.py generate \
  --tool "$scratch/tool" --stock kodak_gold_200 \
  --calibration tests/grain/calibration/gold-200.json \
  --output "$scratch/kodak_gold_200"
```

For another stock, choose its exact capture key from `list` and supply a new
calibration JSON. A separate key identifies each push variant. Every field in
the Gold example is required. The destination must be a new directory.
[Gold calibration](../tests/grain/calibration/gold-200.json)

| Calibration field | Meaning |
| --- | --- |
| `strength` | RMS density target at the pinned 48 µm granularity aperture. Cite the measurement, fitting method, or authored value and its density convention. |
| `size_mm` | Physical Fotufilm grain size in millimetres. Supply the value used by the calibration method. |
| `layer_weights` | Three positive weights in C/M/Y order. |
| `base_density_offset` | Offset added to net CDF3 for the gross density used to index the coating. The loader removes this offset when interpreting net density. |
| `fog_density` | Separate grain-forming fog for the coating model. |
| `read_density` | `net` or `gross` for the RMS target, matching its source convention. |
| `radius_span` | Fotufilm crystal radius span, from 1 to 20 inclusive. |
| `sublayer_shares` | Four Fotufilm sublayer shares that sum to 1. Each emitted JSON number must decode to a Float32 strictly between 0 and 1. |
| `coating_density_scale` | Fotufilm coating density scale, from 0.25 to 4 inclusive. |
| `provenance` | A `basis` of `measured`, `fitted`, `authored` or `trial` and a `source` that identifies what was measured or assumed. |

Run `python3.13 tests/grain/test_generate_film_bank.py` from the repository
root to check calibration admission without starting Cargo or Swift.

Reuse a source calibration when stock, process and density conventions match.
Record which individual parameters came from a measurement and which remain
estimates. Gold's 0.0105 anchor is authored. A catalogue sweep may use explicitly
labelled trial parameters, including for push variants. Those trials exercise
coating generation.

The manifest hashes both payloads, the source profile and the exported response.
It records the calibration, polarity, adapter hashes, Swift version, Fotufilm
revision, coating seed and footprint measurement seed. The directory also
contains the evaluated response, independent footprint statistics at 1, 9.375
and 32 texels, and 3,360 reference samples from the pinned Swift sampler. Add
more tables with repeatable `--footprint` arguments. The qualified range is
1 to 37.5 texels, inclusive. At large widths, Fotufilm's centre-placement
measurement can clip the square at the placement margin. For ordinary
verification use the frozen independent tables and probes.

Shipping any additional stock also requires its own density and spectral
qualification, resource attribution, package entry and explicit product
admission. See the [Gold notice](../Resources/grain/gold-200/NOTICE.md) and
[Portra certificate](../tests/grain/portra-grain-v2-certificate.md).
