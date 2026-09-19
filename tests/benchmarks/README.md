# Benchmarks

Benchmarks are explicit measurements, not CTest pass/fail checks. Build the
Release test target first, then record the GPU model, driver, preset/commit and
whether other GPU processes were active.

For the retained scatter/halation measurement on Linux:

```sh
out/build/linux-release/tests/bin/JuicerScatterHalationGpuTests \
  --case-group performance \
  --fixture-root tests/scatter_halation/fixtures \
  --resource-root out/build/linux-release/tests/Resources \
  --scratch-root out/validation/linux-release/scatter-performance-active \
  --performance-output out/validation/linux-release/scatter-performance-active/results.json \
  --performance-mode active \
  --device-index 0 --width 1920 --height 1080 \
  --warmup-count 10 --sample-count 30 --hold-milliseconds 0
```

Repeat with `--performance-mode inactive-baseline` and separate scratch/output
paths. Compare only runs with the same executable, dimensions, device, driver,
warmup/sample counts and external GPU load. The JSON result is diagnostic
evidence; no shared timing threshold is inferred from one machine.

On Windows, use the matching
`out\build\windows-clang-release\tests\bin\JuicerScatterHalationGpuTests.exe`
and Windows paths with the same arguments.
