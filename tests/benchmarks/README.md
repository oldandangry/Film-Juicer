# Benchmarks

Build the standalone tools by setting `JUICER_BUILD_BENCHMARKS=ON` on the
native Release preset. The example uses WSL. Substitute
`windows-clang-release`, the `.exe` paths and `Contents/Win64` on Windows.

```sh
cmake --preset linux-release -DBUILD_TESTING=OFF -DJUICER_BUILD_BENCHMARKS=ON
cmake --build --preset linux-release --target bundle-archive JuicerOfxMeasurement
```

Run the measurement binary with explicit module, resource and output paths.
Use `--mode check` to exercise the workload before selecting `--mode measure`.

```sh
out/build/linux-release/tests/bin/JuicerOfxMeasurement \
  --mode check --lane shipping --workload halation-1080p-direct \
  --module out/stage/linux-release/juicer.ofx.bundle/Contents/Linux-x86-64/juicer.ofx \
  --resources out/stage/linux-release/juicer.ofx.bundle/Contents/Resources \
  --output out/validation/linux-release/halation-1080p-direct.json
```

The [native measurement driver](ofx_measurement.cpp) owns the workload names,
accepted options and output fields. Build `JuicerOfxMeasurementModule` to use
`--lane observed`. Pass its module and staged test resources for observation.
Use the same input, GPU and native Release profile when comparing timings.

`JuicerOfxSpectralReplayModule` and `JuicerHalationComparisonProbe` support
explicit replay and supplied-module comparisons. The module stages beside its
resources under `out/build/<preset>/tests/spectral-replay/ordinary/Contents`.
Use the [OFX guide](../ofx/README.md#supplied-production-halation-comparison)
for the comparison invocation. Run `JuicerScannerCostBenchmark` with a
retained planar-density file, scanner packet and output path.

`JuicerGrainPrototype` is built with `BUILD_TESTING=ON` because its host
and GPU oracle cases are registered with CTest. Use `host` or `gpu`
with the executable to reproduce those checks. Its native source documents
the separate `reference`, `capture` and `benchmark` arguments.
Store measurements under the native preset's validation directory and keep
shared GPU runs serialized.

The [tagged benchmark guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/benchmarks/README.md)
preserves the migration-era campaign commands, inputs and comparison scope.
