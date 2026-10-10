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
- Rust and Cargo 1.98.1 for the current platform, selected by
  `rust-toolchain.toml`. The tracked quality gate also requires the matching
  rustfmt and Clippy components.
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

Install the pinned Rust toolchain in the same operating-system environment that
runs CMake. With that environment's own rustup installation:

```sh
rustup toolchain install 1.98.1 --profile minimal --component rustfmt --component clippy
rustc --version --verbose
cargo --version
```

Run this separately in a Linux or WSL shell for
`x86_64-unknown-linux-gnu` and in Windows for
`x86_64-pc-windows-msvc`. Do not invoke Windows Cargo from WSL or treat a
Windows installation as Linux qualification. CMake verifies the exact version
and native target during configuration.

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

The presets use four CTest workers. GPU groups share `juicer_gpu`, the two
product Cargo tests share `juicer_cargo`, and groups with `RUN_SERIAL` still run
alone. Keep GPU suites from separate Windows/WSL invocations serialized on a
shared device; CTest resource locks apply within one invocation. Override the
worker count with `--parallel <count>` when the machine needs a smaller limit.

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

During implementation, build the affected test targets and run only the owning
cases needed to check the current edit. Run new compiled cases on Linux and
Windows early so platform-specific compilation, lifetime or diagnostic issues
are found before the final matrix. For example, quality-tool changes can start
with these short development checks in each native Python environment:

```sh
python -m unittest discover -s tests/quality -p test_check_quality.py -v
python -m unittest discover -s tests/ctest -p test_report.py -v
```

Once the candidate is stable, run one unfiltered suite per applicable preset.
Do not precede it with another broad focused pass solely to produce a separate
receipt. Record focused-domain results from that same full-suite JUnit report:

```sh
ctest --preset linux-debug --parallel 4 --output-junit out/validation/linux-debug/ctest.xml
ctest --preset linux-debug -R 'Quality.Dispatcher|Ctest.Host.Report' --show-only=json-v1 > out/validation/linux-debug/focused-tests.json
python tests/ctest/report.py --junit out/validation/linux-debug/ctest.xml --inventory out/validation/linux-debug/focused-tests.json > out/validation/linux-debug/focused-summary.json
```

Use the required owning-domain selection and corresponding preset/campaign
paths. `--show-only` lists tests without executing them. The report command fails
for an empty selection, missing required case, failure, skip or malformed input;
its output explicitly identifies evidence extracted from the full run. It does
not replace checking the full CTest exit status. Keep inventory and report tied
to the same candidate/configuration; neither an older report nor a result from
another preset qualifies the current build. Preserve separately required
reference captures, transition checkpoints and production Release isolation.
When redirecting the inventory in Windows PowerShell 5, use `Set-Content
-Encoding utf8` so the JSON is UTF-8 rather than PowerShell's default UTF-16.

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
| `hash/hash_contract_test.cpp` | Exact byte/word, signed-zero and NaN-mask identity vectors used by the migration reference | `host` |
| `diffusion/host_reference_test.cpp` | Production host diffusion behavior against the pinned spektrafilm cohort | `host`, `reference` |
| `grain/scratch_reuse_test.cpp` | Prepared-frame DIR/grain scratch ownership, bitwise equivalence to dedicated grain intermediates, and lifecycle contracts | `gpu` |
| `grain/delta_fusion_test.cu` | Grain output against a separate FP32 blur/accumulation/delta oracle, including finite sanitation and final-layer dispatch | `gpu` |
| `dir/exposure_cache_test.cpp` | Source-pass versus separate-pass log-exposure caches and final capture density through production CUDA operators | `gpu` |
| `gamma/test_compare.py` | Comparator bounds, applicability, non-finite rejection and CLI failure propagation | `host` |
| `quality/test_check_quality.py` | Quality dispatcher selection, CUDA command translation and header owners, source hygiene, and failure propagation | `host` |
| `ctest/test_report.py` | Focused evidence extraction from full-suite JUnit, including missing, failed, skipped and malformed-input rejection | `host` |
| `quality/test_rust_naming.py` | Product naming-policy contract: current Rust code, accepted names, individually rejected names, test targets, and reasoned foreign-name exceptions under the actual workspace lints | `host`; pinned Cargo and Clippy required |
| `quality/test_rust_boundaries.py` | Dependency/build contract and compiler-enforced profile/CSV privacy, borrowed-view lifetimes and safe-module prohibitions, with valid consumer controls | `host`; pinned Cargo required |
| `ffi/spectral_owner_test.cpp`, `ffi/spectral_bootstrap_test.cpp` | Production spectral ABI, independent native copies, fail-closed bootstrap, selected computation and measured lifetime/capacity checks | `host` |
| `ffi/illuminant_calibration_test.cpp` | Closed CSV ABI, independent seven-curve captures, retained snapshots, selected CMY calibration and actual read/probe allocation transport through native recipes | `host` |
| `ffi/test_native_boundary.py` | Native operation/host/context boundaries and retired profile authority, including new nested source/header and forbidden-symbol controls | `host` |
| `ofx/probe.py` | Linux synthetic OFX load/describe/unload and captured describe properties; no parameter varargs or render | `host`, Linux only |
| `ofx/processor_reference_test.cpp` | Native OFX image/property seam driving the current CUDA processor for four routes, a combined optics/grain/print case, and signed-zero print transitions | `gpu` |
| `ofx/adapter_trace_test.cpp` | Genuine native OFX parameter/image fixture driving the actual `JuicerEffect` event and render callbacks through time, seek, preset/reset, undo/redo-like, and invalid/recovery sequences | `gpu` |

`Quality.RustNaming` uses the current workspace manifests and Clippy configuration
in temporary copies under `out/validation/<preset>/quality/rust-naming/`.
The checked-in `quality/fixtures/rust_naming/*.rs` files are product naming-policy
contracts, not external references. Both crates are checked in the preset's
profile, including test targets; the complete quality dispatcher checks both
development and release profiles. Expected failures must produce the specific
error code and primary source span for every rejected identifier; a compiler
setup failure is not a passing negative test. Valid names and a reasoned raw
binding exception must compile successfully. Cargo runs offline and locked;
the normal configured build supplies cached dependencies. Logs remain in the
artifact directory after temporary workspace cleanup. Run it with
`ctest --preset <preset> -R '^Quality.RustNaming$' --output-on-failure`, or use
the quality dispatcher, which supplies the same tool and artifact environment.

`Quality.RustBoundaries` uses one temporary workspace per run. Both compiler-probe
groups reuse Cargo artifacts under `out/build/<preset>/cargo/quality-probes/`,
separately from product artifacts. Cargo's fingerprints and build lock govern
reuse; every current probe and its required diagnostics still run. No test
outcome is cached. The groups remove temporary source workspaces on completion and keep
logs under `out/validation/<preset>/quality/rust-boundaries/`. Run
`ctest --preset <preset> -R '^Quality.RustBoundaries$' --output-on-failure` or the
quality dispatcher. Cargo is offline and locked; the configured build supplies
the dependencies. Each negative fixture in `quality/fixtures/rust_boundaries/`
must fail for its expected compiler code at its own source location, not because
of missing tooling or an unrelated error. These are product API contracts, not
external numerical references. Valid use runs with and without `test-support`;
CTest selects the preset's profile and complete quality gates check both
development and release. The probes compile without
linking or executing native CUDA calls.

Add construction/lifetime probes as the corresponding real API lands. Put
identity, reuse, conversion and failure-transition cases in the owning domain's
existing tests, using independent expected values. After a completed cutover,
extend `Ffi.Host.NativeBoundary` for concrete retired symbols/dependencies and
add accepted/rejected controls. Do not replace contextual ownership review with
function-size limits, generic source parsers or an expanding exception baseline.

The diffusion fixture is an external reference. The scatter binary fixture is
also an external reference with revision, shape, channel order and numerical
limits recorded in its manifest. Gamma comparator rows are product/tooling
contracts; they do not make the stale historical gamma captures current.
`ofx/descriptor_manifest.json` is a C++ characterization of only the describe
properties observable in the synthetic property suite at the pinned source
commit and binary digest recorded in that fixture. The admission rows in the
scatter integration group include exact fresh recipe/seed expectations and a
test-only post-snapshot handshake; the installed plug-in links the ordinary
uninstrumented admission object. The fresh identities are pinned separately
for Linux and Windows from the unchanged C++ reference because their captured
values differ across those supported platforms.
`ofx/processor_reference_pixels_{linux,windows}.dat` are Film-Juicer C++
characterizations, captured from unchanged production source at
`86bf1af12eea620baefb56001919848b9eb06997`. The isolated Linux Debug
capture binary SHA-256 is
`794260d6a1313fa89f05dcb33583ff6ce9db08d7a7ec4ee54021397e6008b47d`;
the Windows Debug binary SHA-256 is
`3bc6c45c6b0d25e9f5b2436ce2fd0cce3e4137ef6b98a35469ae31cc6115c8ad`.
Both capture worktrees added only the test target; production source and the
44 bundled resources remained at that commit. Each row stores interleaved
float RGB or RGBA pixels, row major, for a 7×5 image. The input has five
padding floats per row and fixed session, instance, clip and frame facts.
The test requires exact alpha and untouched destination padding; color
comparison uses `2e-4 + 3e-4 * abs(reference)` per channel. The two print
glare signed-zero rows are independent fresh builds; transition output must
agree with the appropriate fresh row. Every processor case audits the source
and destination image handles independently and rejects duplicate, unknown,
or missing releases. Fixture emission is an explicit
`--emit-reference` maintenance operation and is never part of CTest.

`Ffi.Gpu.CudaRender.negative-direct` and `.negative-print` cover the real C render
entry, bounded cancellation, native status/exception containment, callback scope,
context/stream ownership, and successful reuse after abort. Prepared-boundary cases
compare the C entry with direct execution and immutable captures; see
[the FFI guide](ffi/README.md#native-render-entry) for the exact evidence boundary.

`Ofx.Gpu.ExecutorFailureOrder` uses isolated test objects for the native executor,
processor adapter and prepared-frame abort observer. It drives both route failure
sites through the same native OFX fixture, checking failure classification,
prepared-frame abort, gate release, deferred DIR message attempt, adapter context
recovery and fatal OFX result order. It also covers non-DIR filtering,
percent escaping, failed/throwing host delivery and a failure that requires no
context retirement. The injected context-loss status exercises the existing
retirement/latch policy on the fixture's CUDA context; it does not simulate real
driver loss or establish Resolve recovery acceptance. The hook definitions are
absent from the product and ordinary processor-reference objects.

`Ofx.Gpu.ResourceFailure.DefectFence` and `Ofx.Gpu.ResourceFailure.GrainUpload`
extend that isolated build to the CUDA resource owner. They check Runtime event
creation/recording failures and ordered STBN/Wang upload outcomes through both
render routes, including exact status/API/code, diagnostic retention, abort and
recovery order, and latch invalidation. Grain cases cover ordinary failure
followed by context loss, the reverse order, and failure followed by success,
with diagnostic wording varied independently of status. These are bounded
operation failures on a live GPU, not actual driver-loss tests; production
objects contain none of their injection hooks.

`Ofx.Gpu.AdapterTrace` implements the C variadic parameter calls used by the
plug-in rather than returning fabricated success. It records current-value and
time-specific getter calls, authored and nested edit events, admitted exposure,
recipe/build/latch identities, output signatures, messages, and image releases.
Preset and advanced-reset operations must apply their expected parameter
values and produce respectively 22 and 16 synchronously suppressed nested
callbacks without changing state inside any nested callback. Every admitted
latch identity is valid and nonzero; the sequence checks replacement and reuse
at each time, edit, rejection, and recovery boundary. Every render, including
the expected invalid-control rejection, must release each of its two distinct
fetched image handles exactly once; duplicate, unknown, and missing releases
fail the case. The trace is written to
`out/validation/<preset>/ofx/adapter_trace.txt`; it is a generated diagnostic,
not a checked-in expectation. A test-only observer reads existing private
adapter/state facts; the installed plug-in gains no callback instrumentation.

`Resource.Gpu.PinnedUpload` exercises the production bounded staging pool in an
isolated resource-owner build. It proves that a native upload remains outstanding
while caller storage is overwritten and destroyed, then checks the completed
device bytes. Only a warmed, explicitly supplied nonblocking stream enters its
finite gate; no render or absent-stream path is blocked. Other cases cover staging
capacity, allocation/event/copy failures, typed status, device-allocation rollback,
reuse, pressure trim, exceptional completion, and normal/proven-loss purge. A
quarantined upload cannot use a previous completed event as proof of its latest
copy's completion. The test hooks are absent from production objects. Injected
errors and a purge-disposition test do not simulate actual driver/context loss.
Diagnostic-allocation failures also check terminal reservation transitions,
native-status preservation, and device-ledger rollback before owner teardown.

The `ScatterHalation.Gpu.PreparedFrame` lifetime rows call the actual
`Root::prepare_cuda_frame` path, complete the production scan-error staging
sequence, and then queue a pinned host-to-device upload into Root-owned
prepared carrier storage behind a finite test gate. A queried CUDA event must
report pending work before the frame's last legal CPU borrow. After `finish`,
the final borrowed view, request, and complete caller-owned preparation object
are destroyed while that event remains pending. The exact-context owner,
frame-use fence, production-owned scan-error host/event and copied identity,
native allocation records, and device ledger charges must remain; the uploaded
carrier's exact base address must be
present in the nonempty retire queue with nonzero retire bytes. The gate is then
released and completion must be observed within the timeout. The gate has a
hard self-release only to prevent a test deadlock, and a passing row requires
that fallback not to fire. A separately compiled prepared-frame test object has
one narrow, one-shot seam that returns a defined drain error before calling the
physical CUDA drain. It requires ownership and ledger retention, then retries
the ordinary drain successfully. The seam is absent from the plug-in build and
is removed when the native Root ownership boundary moves in migration stage
S2.B.
This is controlled failure-branch evidence; it is not a real CUDA context-loss
or driver-reset test and never resets a live host-owned context.

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

Tests verify the approved behavioral contract against its applicable authority:
spektrafilm for ported photographic behavior and approved Film-Juicer designs for
its extensions. Before restructuring, identify evidence independent of the changed
mechanism and reuse existing coverage when adequate. Check observable contracts,
numerical results, identities, lifecycle behavior, and failure semantics; do not
derive expected results solely from the implementation under test. Structural
checks may enforce explicit architectural requirements, but cannot substitute for
numerical or lifecycle evidence when those contracts are affected.

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
- Standard public CI runs Linux and Windows Debug host selections plus the
  tracked native/CUDA/Rust quality and Python checks. Both lanes install and
  verify the cuRAND and CCCL headers required by Clang's CUDA frontend. The
  `linux-debug-quality` and `windows-debug-quality` artifacts retain quality
  logs, translated CUDA commands, and the compilation database even on failure.
  Both lanes compile against CUDA 13.2 but expose no NVIDIA device and run no
  `gpu` tests. The host selection automatically includes
  `Ofx.Host.RenderAssertions`, which checks rejection of corrupted pixels,
  non-finite samples, changed identities/seeds, alpha, padding, and execution
  bits. CUDA pixel fixtures and accepted-capture comparisons run in the `gpu`
  selection. GPU, installed-Resolve-library, and Resolve-render evidence are
  reported separately.
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
output is bit-exact against the oracle. The same executable covers LUT-only
scanning on all four route/polarity combinations, explicit required-LUT failure
reporting in both scanner callers, and retained-resource transitions from LUT
resolution 17 to 33 and back to 17. The transition check compares descriptor
identity, allocation extent, LUT content, and restored fixed-input output; it
does not require the two resolutions to produce a visually observable
difference.

The suite also observes a bounded linear pattern after production scanner blur,
unsharp, weave, and gate attenuation, then checks that encoding follows those
stages. Real prepared film/print resources exercise fused and separate output on
negative/positive direct/print routes, in sRGB and DaVinci Intermediate, with
output gamut compression on/off. Spatial DIR, visual grain, halation, and
camera/enlarger diffusion are disabled in these focused route checks; their
owning suites cover those contracts. No frozen fixtures or private workbench
files are required.

### CUDA ABI declarations

`Ffi.Host.CudaAbi` compares C11, C++20 and generated Rust layouts and type-checks
the CUDA C boundary signatures. `Rust.Bridge` includes internal ABI checks. See
[the FFI guide](ffi/README.md) for bounded commands and pinned binding maintenance.

`Ffi.Host.CudaOwner` covers native owner creation, non-publication on failure,
duplicate rejection, noncreating teardown, scoped close/reload and consume-once
retention after a controlled metadata-only shutdown failure without a GPU.
`Ofx.Host.CudaOwnerUnload` verifies that the real factory's failed close returns
a fatal status through the OFX entry point and blocks replacement ownership.
Direct native fixtures explicitly own their runtime; `Root` access only borrows
it. `Ffi.Host.CudaTerminal.*` covers typed shutdown/destroy, no-retry retention,
exception containment and admission ordering. `ScatterHalation.Gpu.TerminalRetention`
checks retained GPU allocations/ledger charges; `Grain.Gpu.ScratchReuse` checks
instance retirement across exact contexts. See the FFI guide for the ownership
contract and the remaining host/hardware qualification limits.

`Ffi.Host.PreparedProjection` checks fixed C/native descriptor mappings and
malformed records. `Resource.Host.ScratchRequest` checks closed DIR construction,
extent and attachment constraints, and accepted direct/print scratch generations.
These are product contracts and need no device at runtime.

`Ffi.Gpu.PreparedBoundary.*` links a fixture-only C caller into the native executor.
It checks bit-exact equivalence between cold/warm C-boundary execution, the direct
executor and the processor adapter for all seven scenarios. Each case starts a
fresh process and native owner, calls the C boundary before any other render to
exercise cold uploads, then repeats with warm resources and calls the other paths.
Independent numerical expectations are checked separately by
`Ofx.Gpu.ProcessorReference` and `Ofx.Gpu.AcceptedCudaCaptures`, so a baseline-image
failure cannot prevent the path-equivalence checks from running. No case resets
the CUDA context. Production callbacks project their immutable state into the same
admitted native C-boundary render body. The separate render-contract, owner/terminal,
and processor-cutover cases cover admission, borrowed-storage expiry, and callback
lifecycle behavior. See the FFI guide for their scope and remaining host evidence.

`Resource.Gpu.DeferredDestroy.{lock,allocation,storage,ordinary,control}` exercises
exact-context deferred ownership with three fixture-owned CUDA contexts, three
real 4 KiB allocations and their ledger charges. The isolated resource object
injects at the drain/serving-mutex boundary after extraction. A failed batch keeps
the failed and unprocessed entries charged and discoverable; explicit subsequent
drains reach zero without disturbing the unrelated context. The storage case
rejects C++ allocation during extraction and restoration, and ordinary failure
uses a real context mismatch. Only fixture-created contexts are destroyed.

The `Ffi.Host.ProductionProfileOwner` group qualifies the production film/print
and owned gamma ABI independently of the feature-only C7 fixture interface. It
also exercises cold-conversion races, same-source print leases, memory capacities,
typed profile illuminants, canonical computational axes and consuming Hanatos
window behavior. `Ffi.Host.CudaTerminal.host-cache-failure` and
`.host-cache-native-precedence` cover fallible host cleanup publication and native
close precedence. These host-runtime groups require the normal CUDA build tools
but perform no driver/device operation.

The OFX render baselines now follow the accepted Rust density policy. See
[CUDA render baselines](ofx/README.md#cuda-render-baselines) for exact accepted
capture samples, independently derived procedural CUDA fixtures, strict pixel
limits and finite checks, exact identity/seed expectations, negative controls,
and separate lifecycle/boundary CTest processes. Ordinary runs do not generate
or update expectations.


`Ffi.Host.NoiseOwner` qualifies the complete production noise owner and the
single fixture facade: borrowed spans, cache/Assets expiry, concurrent readers,
consume-once release, cleared failures, native paths and layout/signatures.
`Ofx.Gpu.NoiseSourceLifetime` checks the actual direct/print caller's acquisition
before NativeCall, same-thread reentry rejection, deferred error precedence,
cancellation and release after gate/recovery scopes. Its scope observations use
the calling thread's actual NativeCall lifetime. Grain upload failure cases also
retain the source through abort/recovery. `Resource.Gpu.PinnedUpload` includes a
Rust noise owner released while the native staging transfer is still pending.
The prepared-boundary fixture acquires through the same safe Rust source via the
test facade and asserts that supplied views cause no fallback acquisition.
These seams exist only in test objects/test-support; normal Release has none.
Installed Resolve and actual driver/context-loss acceptance remain separate.

`Assets.Host.IlluminantConstruction` qualifies Rust illuminant math and its
synchronous C/native bindings against four frozen native-parent captures, with
exact products/identities, scoped source/lens lifetimes, cache and failure
contracts. The separate negative-print C render contract checks the later native
preflash unwind/completion terminal. See [the illuminant guide](ffi/README.md#illuminant-construction).

`Spectral.Host.{Tables,White,SInverse,Products,Admission}` and
`Ffi.Host.SpectralPreparation` qualify fixed spectral math, synchronous C11/native
bindings and production route preparation against four frozen native-parent
captures. They preserve exact family/enclosing identities, the distinct white
precision contracts and the film gate before inverse. See
[the spectral guide](ffi/README.md#spectral-tables-whites-and-film-s-inverse)
for provenance, failure, lifetime, allocation and isolation boundaries.

A5 spectral exposure migration is covered by the `Exposure.Host.*` and
`Ffi.Host.ExposurePreparation` groups. See [the FFI domain guide](ffi/README.md)
for exact fixture provenance, the scoped pinned-erff decision, immutable earlier
fixtures and independently qualified completed-product supplements.

### Film TC preparation and allocation ownership

`Reconstruction.Host.TcLut.*` covers complete Hanatos/Arctic tables, Mitchell
sampling, input remapping, raw C11/C++/Rust ABI, direct-core facade, failures,
actual route products/admission and final publication release. The TC fixture
manifest selects independently frozen native expectations by preset. Records
are shared only when all expanded fields and complete table bytes agree.
Existing fixture bytes and bounds remain unchanged.

Native-undefined coordinate cases have required ordinary preparation-failure
categories rather than manufactured native pixels. Defined brightness overflow
still reaches final sanitation. A completed TC result owns one Rust Vec; the
native move-only holder exposes readonly spans and releases the original
pointer/length/capacity once in Rust. Source and hull borrows expire on return.
The Lifetime group observes actual publication paths without an external old
state hold and checks release outside both publication and rebuild locks;
retained-reader and cross-thread cases are separate.

### Mallett exposure preparation

`Exposure.Host.Mallett.*` checks focused BGR mid-gray, separate synthetic RGB
reference reduction, shared reference-source arithmetic and TC normalization
against independently frozen native-parent expectations. Its manifest selects
exact preset records; older color/spectral/exposure fixtures and their assignments
remain unchanged. `mallett-failure-contract.json` separately records the approved
earlier focused missing-basis failure and original native outcomes.

The C11/C++/Rust boundary groups cover layouts, values, structural errors,
cleared outputs, bounded diagnostics and contained panic. Fixed math has no heap
allocation; fault category injection is not allocator-pressure evidence.
`products` and `admission` exercise actual publication, skip/consumption,
recovery, supersession, immutable holds, cache release and same-input reuse.
Feature-only spectrum operations preserve the current independent CAT02/input
color witnesses; they use fixed borrows and never provide a production renderer.
The four value exports remain temporary native bindings. Normal production
builds omit the fixture facade and fault storage.
