# Benchmarks

Configure with `JUICER_BUILD_BENCHMARKS=ON` and build
`JuicerRenderBenchmark` to time the guarded cold and warm render cohort. The
elapsed time includes loading, rendering and readback. The harness loads the
staged OFX module and requires a CUDA device.

The [tagged benchmark guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/benchmarks/README.md)
records the historical comparison tools and capture inputs.
