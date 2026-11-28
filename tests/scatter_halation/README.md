# Scatter and halation tests

The core optics suite checks authored inputs, frame descriptors and Gaussian
coefficients. Its [frame words](../../rust/film-juicer-core/tests/fixtures/optics/halation-frame/README.md) cover the original
schedule and completed Direct and Print route products. The same unit test
checks typed frame failures against the frozen expectations.

`Profile.Gpu.NativeSafety` exercises production device density helpers,
coordinate conversions, LUT consumers and the Direct and Print capture
development launches. The staged OFX probe checks guarded product renders.
See the [current test commands](../README.md).

The [tagged scatter and halation guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/scatter_halation/README.md)
records the historical native parity captures, independent spektrafilm inputs,
operator tolerances and benchmark procedures.
