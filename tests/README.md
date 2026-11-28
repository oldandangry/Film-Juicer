# Film-Juicer public tests

Native tests use the CMake product toolchains. Install Python 3.13 for the
host fixture and tooling checks.
Install Rust 1.98.1 and prebuilt cargo-nextest 0.9.143 for the workspace tests.
See the [build prerequisites](../docs/building.md).

### Validation cadence

Run the nearest affected test while editing. Before review, run each applicable
profile and platform from a fresh testing-enabled build. Use the corresponding
Windows preset in an x64 developer shell. Run the GPU selections on Windows
and WSL one at a time. CTest applies a 300-second limit per test and a requested
selection with zero tests fails.

The following example uses WSL Debug. The default build includes the
registered binaries, generated headers, runtime resources and the staged
module used by the production render probe.

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
ctest --preset linux-debug -L '^host$' --no-tests=error --output-on-failure
cargo nextest run --workspace --locked --test-threads 1
cargo test --workspace --doc --locked
ctest --preset linux-debug -L '^gpu$' -LE '^multi-gpu$' --no-tests=error --output-on-failure
```

Repeat the CMake and CTest steps with the Release preset. Run these Cargo
commands for Release.

```sh
cargo nextest run --workspace --locked --release --test-threads 1
cargo test --workspace --doc --locked --release
```

CMake builds the production archive for the module and linked native tests.
Cargo runs the core borrow-expiry doctest
and its compiling counterpart in both profiles. The plug-in is a static library,
so Cargo reports that it has no doctests.

Use `ctest --preset linux-debug -N` to discover names. For a focused run, use
`-R` with an anchored test-name expression alongside `--no-tests=error`.
The `host` label requires no GPU or driver at runtime. The `gpu` label needs
a device and driver. `reference` identifies fixture provenance. `multi-gpu`
marks coverage for hosts with two devices. Retain the CTest and Cargo output
and exit status for review.

### Release package checks

Configure the production closure separately, with testing and benchmarks off.
Run the package build in a toolchain environment without Python. Run both
strict C11 probes with CUDA hidden. Use Python 3.13 afterwards to inspect
the archive, dependencies, exports, resources and notices.

```sh
cmake --preset linux-release -DBUILD_TESTING=OFF -DJUICER_BUILD_BENCHMARKS=OFF
cmake --build --preset linux-release --target bundle-archive JuicerProductionOfxProbe JuicerProductionCudaAbiProbe
CUDA_VISIBLE_DEVICES=-1 out/build/linux-release/tests/bin/JuicerProductionOfxProbe out/stage/linux-release/juicer.ofx.bundle/Contents/Linux-x86-64/juicer.ofx
CUDA_VISIBLE_DEVICES=-1 out/build/linux-release/tests/bin/JuicerProductionCudaAbiProbe
python3.13 scripts/check.py bundle --stage out/stage/linux-release
```

On Windows use `windows-clang-release`, both `.exe` probes and the staged
`Contents/Win64/juicer.ofx`. Set `$env:CUDA_VISIBLE_DEVICES = '-1'` for the
probe processes. Keep the CUDA toolkit's `bin/x64` and `bin` directories on
PATH for linked executables. Run the bundle check from the VS x64 developer
shell with `dumpbin` available.

### Suite map

| Area | Test and fixture owners |
| --- | --- |
| Rust model, publication and assets | [Core tests](../rust/film-juicer-core/tests), [core fixtures](../rust/film-juicer-core/tests/fixtures) and unit tests beside their source owners. |
| ABI, value preparation and owner lifetime | [ABI and bindings](../docs/ffi.md) and [native FFI cases](ffi). |
| OFX host callbacks, production renders and recovery | [OFX host fixtures](ofx/README.md), [production staged probe](ffi/production-ofx-render.cpp), and the staged frame and Print edit cases. |
| Grain v2 Full and proxy sampling | [Optical aperture cases](grain/film_apertures_test.cpp), [Portra tables](grain/portra-table-fixtures), and [Gold tables and sampler probes](grain/gold-reference). |
| Scanner spectra, binding and schedules | [Scanner cases](scanner). |
| Device resources, exact-context lifetimes and failure ordering | [Resource cases](resource), [native CUDA ABI cases](ffi) and the epoch admission and retirement [route cases](scanner/routes_test.cpp). |
| Explicit performance work | [Benchmark guide](benchmarks/README.md). |

`Scanner.Gpu.Routes` links the [scanner route cases](scanner/routes_test.cpp)
with the [OFX cancellation and stale-selection cases](ofx/cancellation_test.cpp)
because they share its observed native entry. Select it for epoch admission,
retirement or OFX cancellation changes; an `Ofx` name filter does not run them.

Core-only fixtures live with the Rust crate. Native and cross-language fixtures
live here, including the [DIR boundary capture](dir/boundary-fixtures),
[Print preflash cases and products](print/preflash-fixtures), the
[Portra tables](grain/portra-table-fixtures) and the
[scanner binding vectors](scanner/color-fixtures/scanner-binding-v1.json).
The staged OFX probe opens the preset-named fixture and checks its reference
preset before reading the shared rows for that operating system.

For a matched Gold 200 comparison, the testing-enabled render probe accepts
`--gold-look MODULE SOURCE_RGBA_F32 OUTPUT_DIRECTORY WIDTH HEIGHT`.
Use a packed linear BT.2020 Float32 RGBA raster and an unused output directory.
The probe renders sRGB Float32 RGBA through the staged standard effect at
36 mm and frame 37 on both negative routes. It runs Legacy with the saved
descriptor defaults and Grain v2 at 0 EV, 1 µm and Colour 1. The four outputs
have identical framing within each source. Use the Release bundle under
review as the module argument.

Fixture manifests record logical preset membership, stored parts and original
capture provenance. A part can provide complete fields or replace named row
positions with preset-specific words. Historical captures and command
inventories remain available at the fixed archive tag.

Keep the checked-in fixture bytes and expected values fixed during ordinary
tests. Compare output with the independent expectation that the case names.
Write generated test outputs under the preset's validation directory. Reference
generation and benchmarking are explicit maintenance operations. Follow the
[contributor checks](../CONTRIBUTING.md#required-checks) for quality and ABI
changes. Resolve and live-driver-loss acceptance have their own evidence.

The [tagged test guide](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/README.md)
records the original migration suites and commands for reproducing that baseline.
The [tagged scanner LUT captures](https://github.com/oldandangry/Film-Juicer/tree/archive/spectral-cutover-2026-11/tests/scanner/lut-fixtures)
remain available with their original manifest.
