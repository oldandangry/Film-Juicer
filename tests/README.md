# Film-Juicer public tests

The public suite is organized by product domain and has one normal scheduler:
CTest. C++ tests use ordinary executables and selective GoogleTest discovery;
Python tests use `unittest`. Reference generation and benchmarks are explicit
maintenance operations, never hidden parts of a correctness run.

## Prerequisites

- CMake 4.3 or newer and Ninja.
- CUDA Toolkit 13.2, including its host runtime libraries. The project still
  configures CUDA even for host-only test selection.
- Linux: GCC/G++ 13 at `/usr/bin/g++-13`.
- Windows: Visual Studio 18, ClangCL from its LLVM tools, and NVCC with the VS
  18 MSVC host compiler.
- Python 3.13. Ordinary Python execution is intentionally tested on 3.13.
- A compatible NVIDIA driver and SM 7.5-or-newer GPU only for the `gpu` label.

Select the CUDA 13.2 installation through CMake's standard environment inputs
when it is not already the system default. This keeps machine-specific toolkit
paths out of the shared presets:

```sh
export CUDAToolkit_ROOT=/absolute/path/to/cuda-13.2
export CUDACXX="$CUDAToolkit_ROOT/bin/nvcc"
```

The configure step rejects a compiler or toolkit outside the CUDA 13.2 series.
These variables only select the installation; NVCC continues to use the GCC 13
host compiler fixed by the Linux preset.

Create an isolated ordinary-test environment on Linux:

```sh
python3.13 -m venv out/test-venv/linux
. out/test-venv/linux/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tests/requirements.txt
```

If `python3.13` is unavailable but `uv` is installed, let `uv` provision the
interpreter and environment:

```sh
uv python install 3.13
uv venv --python 3.13 out/test-venv/linux
uv pip install --python out/test-venv/linux/bin/python \
  -r tests/requirements.txt
. out/test-venv/linux/bin/activate
```

The equivalent `uv` setup on Windows is:

```powershell
uv venv --python 3.13 out/test-venv/windows
uv pip install --python out/test-venv/windows/Scripts/python.exe `
  -r tests/requirements.txt
```

On Windows, start from the Visual Studio 18 developer environment, make the
VS LLVM directory and CUDA 13.2 available as required by the project presets,
then run:

```powershell
py -3.13 -m venv out/test-venv/windows
out/test-venv/windows/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r tests/requirements.txt
```

GoogleTest 1.18.0 is fetched only when `BUILD_TESTING=ON`; its release archive
is checksum-pinned in `tests/CMakeLists.txt`. For an offline checkout, use
CMake's standard override with an already extracted source tree:

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/absolute/path/to/googletest
```

## Build and run

Linux Debug:

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
ctest --preset linux-debug
```

Windows Debug uses the same sequence with `windows-clang-debug`. Release uses
`linux-release` or `windows-clang-release`.

The unfiltered CTest command runs every automated correctness group applicable
to that platform and therefore requires a supported GPU and driver. Selection
uses standard CTest options:

```sh
ctest --preset linux-debug -N
ctest --preset linux-debug -L host
ctest --preset linux-debug -L gpu
ctest --preset linux-debug -L reference
ctest --preset linux-debug -R ScatterHalation
ctest --preset linux-debug -R Gamma
ctest --preset linux-debug --rerun-failed --output-on-failure
```

On a normal GPU development machine, prefer the single unfiltered command.
Labels are for CI, machines without a GPU, and focused diagnosis; they are not
additional passes that must be run after the full suite.

`host` means execution does not require an NVIDIA driver or device. It still
requires the CUDA 13.2 toolkit to configure/build and may load toolkit runtime
libraries. `gpu` means the executable loader or the case itself needs the
driver/device. `reference` is an additional meaning label for cases checked
against a defined external authority; each such test still has exactly one
execution requirement (`host` or `gpu`). CUDA test processes share one CTest
resource lock.

The Python comparator can also be debugged directly:

```sh
python3.13 -m unittest discover -s tests/gamma -p test_compare.py -v
```

Turning testing off preserves the normal plug-in build and performs no
GoogleTest fetch or Python discovery:

```sh
cmake --preset linux-debug -DBUILD_TESTING=OFF
cmake --build --preset linux-debug
```

## Suite map

| Domain | Meaning | Execution |
| --- | --- | --- |
| `scatter_halation/recipe_descriptor_test.cpp` | Product contracts for control validation, recipe identity, descriptor dispatch, public-resource bootstrap and profile loading | `host` |
| `scatter_halation/integration_test*` | Production-linked preparation, CUDA operator fixtures, route/carrier behavior, zero-work and lifecycle contracts | `gpu`; selected fixtures are also `reference` |
| `diffusion/host_reference_test.cpp` | Production host diffusion behavior against the pinned spektrafilm cohort | `host`, `reference` |
| `grain/scratch_reuse_test.cpp` | Prepared-frame DIR/grain scratch ownership, bitwise equivalence to dedicated grain intermediates, and lifecycle contracts | `gpu` |
| `grain/delta_fusion_test.cu` | Grain output against a separate FP32 blur/accumulation/delta oracle, including finite sanitation and final-layer dispatch | `gpu` |
| `dir/exposure_cache_test.cpp` | Source-pass versus separate-pass log-exposure caches and final capture density through production CUDA operators | `gpu` |
| `gamma/test_compare.py` | Comparator bounds, applicability, non-finite rejection and CLI failure propagation | `host` |
| `ofx/probe.py` | Linux synthetic OFX load/describe/unload and declared CUDA support | `host`, Linux only |

The diffusion fixture is an external reference. The scatter binary fixture is
also an external reference with revision, shape, channel order and numerical
limits recorded in its manifest. Gamma comparator rows are product/tooling
contracts; they do not make the stale historical gamma captures current.

Generated binaries live under `out/build/<preset>/tests/bin/`. CMake stages one
read-only runtime tree at `out/build/<preset>/tests/Resources/`. Test scratch
and failure artifacts go below `out/validation/<preset>/`. Checked-in fixtures
are read-only during an ordinary run.

## Diagnosing a failure

CTest prints the failing case, actual/expected values and relevant tolerance.
Run the named domain with `-R` and `--output-on-failure`; add `-V` when the
underlying command and paths are needed. GPU diagnostics retain case, route,
fixture and CUDA failure detail. A missing driver/device is a failure for a
selected `gpu` test, not a skip.

CTest can write JUnit output without a project-specific report layer:

```sh
ctest --preset linux-debug -L host \
  --output-junit out/validation/linux-debug/host-junit.xml
```

## Adding a regression case

For a small host C++ regression, add a normal `TEST(Suite, Name)` block to the
domain's existing GoogleTest source and rebuild. CMake discovers the case; no
registry or wrapper edit is required. Use production headers/targets and keep
the assertion at a product boundary.

For gamma comparator behavior, add a `unittest.TestCase` method to
`test_compare.py`. For a new integration executable or destructive lifecycle
group, declare one explicit CTest process in `tests/CMakeLists.txt`, give it a
finite timeout and its actual `host`/`gpu` requirement, and keep destructive
transitions isolated.

Fixtures must state whether they express a product contract, a Film-Juicer
characterization, or an external reference. Never regenerate or widen an
expectation from CTest. Reference updates are separate reviewed changes written
first to `out/validation/`.

## Boundaries and limitations

- The build still requires CUDA 13.2 even for `-L host`; a toolkit-free C++
  mode is outside this consolidation.
- Standard public CI runs Linux host/reference-host and Python checks. CUDA,
  Windows, installed-Resolve-library and Resolve-render evidence are reported
  separately.
- The historical gamma capture baseline remains deferred because its resource
  inventory and `Release`/`Release-Clang` identities do not establish current
  four-preset applicability. Two G09 captures are configuration-invalid and
  remain excluded, as detailed in `MIGRATION.md`.
- Reference generators require their own pinned environments and are not part
  of ordinary unittest discovery.

See [MIGRATION.md](MIGRATION.md) for the bounded family inventory and exact
retained/deferred/archive decisions, [benchmarks/README.md](benchmarks/README.md)
for measurement commands, and [manual/resolve.md](manual/resolve.md) for
owner-run host acceptance.

### Grain scratch maintenance benchmark

The ordinary `Grain.Gpu.ScratchReuse` CTest runs the bounded correctness and
lifecycle cases. Its procedural density inputs characterize Film-Juicer grain;
the oracle runs the same production grain kernel with independent intermediates,
requiring finite, bit-for-bit matching output. It covers all four scan routes,
sublayer/shared-chroma shapes, integer/fractional time, and debug views 0–6.
It does not establish whole-Resolve render or visual acceptance.

The explicit 6048×4032 maintenance case reports deduplicated CUDA allocation
capacities and 30 CUDA-event samples after 10 warmups. Events synchronize for
measurement; it is excluded from ordinary CTest runs. From a CUDA-enabled
Windows developer shell:

```powershell
out/build/windows-clang-release/tests/bin/JuicerGrainScratchTests.exe `
  --gtest_also_run_disabled_tests `
  --gtest_filter=GrainScratch.DISABLED_SixKMemoryAndTiming
```

### DIR exposure-cache maintenance benchmark

`Dir.Gpu.ExposureCache` compares source-pass cache output against a separate
cache-building pass, then requires finite, bit-for-bit matching developed film
density. Its 96 procedural cases cover four routes, all three reconstruction
methods, RGB and camera-film-linear inputs, FIR/YVV filtering, and two/three
cached channels. These are equivalence checks of production CUDA launchers;
they do not invoke the OFX host dispatcher or establish Resolve acceptance.

The optional 6048×4032 case times DIR source/filter, cache construction, and
final capture-film development. It alternates the two paths, records 30 samples
after 10 warmups per path, and synchronizes CUDA events for measurement. It
excludes grain and scanner execution, uploads, frame preparation, and Resolve:

```powershell
out/build/windows-clang-release/tests/bin/JuicerDirExposureCacheTests.exe `
  --gtest_also_run_disabled_tests `
  --gtest_filter=ExposureCache.DISABLED_SixKTiming
```

### Grain delta equivalence

`Grain.Gpu.DeltaFusion` checks 154 procedural cases through the production grain
launcher. It generates particles with the unchanged CUDA particle kernels, then
uses an independent scalar FP32 FIR and separate accumulation, bias, subtraction,
and reconstruction steps as the oracle. Outputs must be finite and bit-for-bit
equal. These are Film-Juicer numerical product contracts, not spektrafilm or
whole-Resolve reference fixtures.

The cases cover batched and separate layers, an unblurred final layer, all-zero
dye radii, simple grain, single-pixel and partial-block extents, integer/fractional
time, zero and signed-zero bias, non-finite density, and extreme blur weights.
The oracle's particle generation intentionally shares production code; this test
isolates the subsequent blur/delta schedule and does not validate the sampler.

### Scanner output encoding

`Scanner.Gpu.OutputEncoding` calls the production CUDA scanner launchers and
checks all nine output spaces, CCTF on/off, optional output matrix, transfer
breakpoint neighbors, negative/HDR/subnormal/nonfinite values, RGB/RGBA, odd
image extents, and padded rows. The numerical oracle is Film-Juicer's independent
host double-precision `OutputEncoding::applyEncoding`; this is a product
numerical contract, not an external-reference fixture. Clipped RGB must stay
within `1e-6` maximum and `1e-7` mean absolute error per case, with unchanged
nonfinite classification and bit-exact alpha/padding. Identity-matrix CCTF-off
output is bit-exact against the oracle.

The suite also observes a bounded linear pattern after production scanner blur,
unsharp, weave, and gate attenuation, then checks that encoding follows those
stages. Real prepared film/print resources exercise fused and separate output on
negative/positive direct/print routes, in sRGB and DaVinci Intermediate, with
output gamut compression on/off. Spatial DIR, visual grain, halation, and
camera/enlarger diffusion are disabled in these focused route checks; their
owning suites cover those contracts. No frozen fixtures or private workbench
files are required.
