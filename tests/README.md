# Film-Juicer public tests

The public suite is organized by product domain and has one normal scheduler:
CTest. C++ tests use ordinary executables and selective GoogleTest discovery;
Python tests use `unittest`. Reference generation and benchmarks are explicit
maintenance operations, never hidden parts of a correctness run.

## Testing workflow milestone: 16 October 2026

The focused Linux Debug development path, direct CTest compiler probes, explicit
checkpoint reuse, fixture sharing, reduced repeated test construction and audited
parallel scheduling are delivered. Unrelated dirty fixture payloads are no longer
hashed by local route verification. Both native Release production/export
isolation gates passed with testing disabled. These are bounded results, not a
new complete product qualification or a general automatic ownership map.

Fixture storage is 38,478,395 bytes smaller, including 18,331,985 bytes of exact
copies. Numerical expectations and distinct failure/lifetime checks are retained.
The 26 audited global serialization flags are removed; actual GPU, Cargo and
build-tree ownership remains enforced.

Performance acceptance remains open. The completed unchanged-header checks took
20.028 and 22.177 seconds; the 20-second total and 10-second bookkeeping targets
are not established. The three maintenance tasks plus shared closeout took
76 minutes 25 seconds against a 60-minute target. There is no matched historical
complete-task baseline, so no overall percentage improvement is claimed.
Future real implementation work can supply complete task costs, including setup,
repairs, required checks, review and closeout; it is not itself a before/after
comparison. Full qualification and the remaining latency targets stay separate.

Native CI runs product and boundary groups in Debug and Release, tooling once per
native platform in Debug, and fixture-corpus checks once on Linux Debug. Discovery
prints the same selection that executes, empty selections fail, and direct probe
diagnostics are retained. These hosted host checks do not establish GPU or Resolve
acceptance. See [the contributor workflow](../CONTRIBUTING.md#required-checks)
for the supported local inputs and explicit checkpoint commands.

## Owner-authorized Step 6 final qualification

Step 6 runs one product/boundary **host** checkpoint in each preset.
It runs tooling once in Debug on each native OS and corpus once in Linux Debug.
The available GPU host is one RTX 4080, so GPU work is a set of explicit focused
registrations, not a full all-runtime checkpoint. By owner decision on
5 November 2026, physical multi-GPU qualification is outside acceptance for this
single-GPU target. `Ffi.Gpu.CudaInspect.MultiDevice` remains available as optional
hardware coverage; its unexecuted status is not an I8/S4 acceptance blocker or a
deferred hardware-acquisition requirement. Do not count it as passed or claim
full all-runtime coverage from these selections.

Run both Release OFF production closures before restoring either Release cache to
testing ON. Each command configures and clean-builds under the preset owner,
executes the maintained production contract, validates its compact outcome, and
copies the mandatory flat payloads `production-contract.json` and
`production-commands.json` into the result. Successful report scratch beneath
`results/<run-id>/production/` is removed after validation; failed commands retain
it for reproduction. Production executables and tools are not pinned in historical
results:

```sh
python -B scripts/verify.py prepare --preset linux-release --clean-first \
  --production-contract --targets juicer JuicerProductionPreparedRenderProbe --testing OFF
python -B scripts/verify.py prepare --preset windows-clang-release --clean-first \
  --production-contract --targets juicer JuicerProductionPreparedRenderProbe --testing OFF
```

The report records the actual prepared-render probe, four direct/print cold/warm
rows, C11 probe outcomes, production symbols/exports, and hashes for consumed
fixtures and required build inputs. Offline reading validates the closure's
assertions, probe identities, command outcomes and all four prepared-render rows
from the two copied payloads. It does not access original files, executables or
tools, and does not establish installed Resolve behavior or absent-driver behavior.

After both OFF closures, run these product/boundary host checkpoints on the
same candidate. Clean Debug builds establish the complete test graph; Release
reuses the clean production closure and builds its separate test-support closure:

```sh
python -B scripts/verify.py qualification --preset windows-clang-debug --clean-first \
  --reason 'Final candidate' --selection product boundary tooling --runtime host
python -B scripts/verify.py qualification --preset windows-clang-release \
  --reason 'Final candidate' --selection product boundary --runtime host
python -B scripts/verify.py qualification --preset linux-debug --clean-first \
  --reason 'Final candidate' --selection product boundary tooling corpus --runtime host
python -B scripts/verify.py qualification --preset linux-release \
  --reason 'Final candidate' --selection product boundary --runtime host
```

The Debug commands include the once-per-OS tooling membership and Linux corpus;
do not replay those groups separately.

On the available RTX 4080, run the supported affected GPU registrations as a
focused selection. MultiDevice is optional and requires two physical CUDA GPUs:

```sh
python -B scripts/verify.py development --preset linux-debug --runtime gpu \
  --tests Ffi.Gpu.PreparedBoundary.negative-direct Ffi.Gpu.PreparedBoundary.negative-print \
    Ffi.Gpu.PreparedBoundary.positive-direct Ffi.Gpu.PreparedBoundary.positive-print \
    Ffi.Gpu.PreparedStateLifetime ScatterHalation.Gpu.PreparedFrame \
    ScatterHalation.Gpu.RouteBoundaries ScatterHalation.Gpu.CapturedCarrier \
    ScatterHalation.Gpu.ZeroWork ScatterHalation.Gpu.Lifecycle Ffi.Gpu.CudaInspect \
    Ffi.Gpu.ProductionPreparedRender
```

Keep the standard `out/build/<preset>` trees. `--clean-first` runs under the
native preset owner and rebuilds the selected closure in that cache; do not create
campaign caches or run an unowned clean concurrently. Resolve, installed-host,
visual and other applicable unavailable hardware evidence stays outstanding.
The owner-removed multi-GPU requirement is not among the open acceptance gates.

## I8 integration checkpoint: 5 November 2026

The reviewed I8 candidate passes one fresh product/boundary host checkpoint in
each native preset: 181 Linux Debug, 174 Windows Debug, 162 Linux Release and
161 Windows Release registrations. Debug includes all 13 tooling groups per OS;
Linux Debug also includes the six corpus registrations. Separate focused GPU
runs pass all 55 supported registrations per preset on one RTX 4080. These are
678 host and 220 GPU executions. `Ffi.Gpu.CudaInspect.MultiDevice` was unexecuted
in all four contexts; the owner's subsequent decision above removes it from
acceptance while preserving that coverage limitation.

Cumulative analysis covers 125 translation units per native profile, including
seven CUDA units, and all 12 Rust package/feature/profile combinations. Reviewed
composition retains successful unchanged commands and completes failed or
unexecuted commands; ten original failed analysis reports remain failed. The
four fresh host bundles form a host checkpoint through the offline reader.
Focused GPU results and composed analysis do not form an all-runtime checkpoint.

The bounded repairs select the active Windows compiler and the actual preset
build for native object inspections, contain native test-output exceptions and
correct test-report construction diagnostics. Scientific fixtures, expectations
and production mathematics are unchanged. Earlier Release OFF evidence retains
its original scope; no production input changed to require another OFF cycle.
Independent review approves the implementation and available-context evidence.
Installed-host/Resolve, performance and applicable parent/release gates remain
open. MultiDevice no longer blocks acceptance. Publication does not itself
close I8 or the S4 migration stage.

The subsequent hosted Windows BuildEvidence failure came from nested fixture
paths exceeding compiler-tool path limits. Shorter disposable prefixes preserve
Unicode/space coverage, both fixture-family directories and all lifetime/cleanup
assertions. All seven BuildEvidence functions pass in focused native Debug checks
on each OS. These rechecks retain the earlier checkpoint's original source scope.

The 38 native attempts, including ten failed analysis attempts, record
323,849,321 bytes of source/Git/report/worker bookkeeping reads. Five offline
reports add 138,921,084 bytes. The advisory read/time targets remain unmet;
recorded overlapping intervals are not complete task wall time or physical I/O.

## S5C2 process/runtime ownership checkpoint: 6 November 2026

Safe Rust `Runtime` owns process Assets and the sole native CUDA handle. The
factory, effects and synchronous processors borrow it explicitly; native
`FjCuda`/`Root` retains CUDA resources, exact contexts/epochs, admission, drainage
and uncertain native retention. Native host composition and ambient owner/Root
discovery are removed. See the [FFI ownership guide](ffi/README.md).

Recorded qualification includes 678 host registrations across the four presets
(181 Linux Debug, 174 Windows Debug, 162 Linux Release and 161 Windows Release),
44 selected GPU registrations per preset, scoped/full quality and the ordered
native Release testing-OFF/ON closures. The host checkpoint includes Debug tooling
on both OSes and Linux Debug corpus. The Windows receipt-publication repair has
native transient/persistent-denial controls; historical failed attempts remain
failed and their original receipts are preserved.

Independent review identified omitted consuming-close checks in two GPU failure
branches and stale fixture-owner guidance. The correction passed the four
affected GPU registrations in every preset (16 total), plus scoped native Debug
quality for all three compiler variants on both OSes. These focused results and
the preceding broad checkpoints retain their separate source/assertion scopes.
Implementation and focused re-review are approved, and the owner accepts bounded
publication. NativeSubmissionState/temporal/latch work remains S5.D; capture/UI and
adapter replacement remain S6. Full S5 and installed-host/Resolve acceptance are
not established. MultiDevice remains optional and unexecuted.

`Ffi.Host.OfxAbi` supplies the independent C11/C++20/Rust OFX declaration
boundary in six cases. It is a product-linked host registration; see the
[FFI inventory and maintenance guide](ffi/README.md#ofx-c-abi-evidence).

## Validation cadence

The Rust host candidate has twelve focused host/boundary registrations:
`Ofx.Host.Actions`, `Ofx.Host.Render`, `Ofx.Host.RenderControls`,
`Ofx.Host.Module`, `Ofx.Host.ModuleControls.Construction`, `.Panic`, `.Terminal`,
`Ofx.Host.Context`, `Ofx.Host.ContextControls`, `Ofx.Host.Instance`,
`Ofx.Host.Capture` and `Ofx.Host.InstanceControls`. The three instance/capture
registrations have 8/11/5 named cases and reuse the existing 234 grain witnesses.
The module target declares 303 Describe, 24 Load and 7 lifecycle rows; the control target uses the existing real
Runtime/native owner and a separate test-support archive. See the
[OFX fixture guide](ofx/README.md) for scope and the independent C++ consumers.
The focused compiler method is
`test_rust_boundaries.RustBoundaryTests.test_ofx_callback_borrows`: seven fixtures,
both feature forms, fourteen logical cases in four batched Cargo checks per native
profile. The CUDA owner-borrow method has twelve fixtures, twenty-four logical
cases in six batched checks. `Ofx.Gpu.Render` and `.RenderTrace` exercise actual
candidate callbacks with device images and the existing independent expectation
readers. Shipping exports and discovery remain on the existing adapter.

Choose checks from the changed contract and the checkpoint being completed.
Before execution, name the supported target, test and quality commands, their
expected membership, and the affected native/CUDA/Rust compilation contexts and
platform/profile/feature combinations in the existing change description or
handoff. A focused handoff does not require complete aggregate qualification and
must not be described as full product acceptance.

Ordinary host/GPU CTest registrations use explicit `TIMEOUT 0`; the native
verifier dispatches CTest with `timeout=None` and rejects finite, missing or
malformed ordinary timeout properties before execution. Historical result readers
accept explicit numeric zero and valid finite positive GPU timeouts while retaining
physical-device locks and all other evidence checks. Explicit bounded child fault
controls, specified post-exit handling and hosted job limits remain unchanged.

| Change or checkpoint | Required checks |
| --- | --- |
| Documentation or status only | Whitespace, link/reference consistency and contextual review. No native build, product test or compiler gate solely for prose. |
| Editing a mathematical leaf or other bounded implementation | Touched formatting, compile/type checks, owning mathematical or behavioral cases and the nearest real consumer. Exercise touched C11/native fixture/helper compile cases and scoped quality checks on both native platforms early, before freezing for broad qualification. |
| Stable family cutover or source deletion | The affected family's prescribed numerical cases in Linux/Windows Debug/Release; affected ABI, complete-construction, failure, identity, cache, caller and downstream GPU route cases; scoped native/CUDA/Rust quality and independent contextual naming/ownership review. For a staged cutover, replacement fixture consumers and real caller evidence must pass before deleting their producer. As an alternative for a prospectively planned, reconstructible mathematical-producer cutover, capture the old producer independently and pass replacement core/ABI/fixture preparation before cutover; switch the caller and delete the old body in one candidate, then pass actual replacement-caller and downstream evidence before acceptance. This applies only to new handoffs; an existing handoff's stricter stage order remains unless explicitly amended. Lifecycle, ownership and other deletion gates remain. |
| New or changed ABI, exports, feature isolation or ownership boundary | Both native Release testing-OFF probes once on the stable boundary, plus relevant testing-ON boundary/lifetime checks. Changes to consumed probe inputs require fresh evidence. |
| Broad shared build/lifetime change, designated prepared-state integration, instance-state stage exit, OFX cutover, or final release qualification | Complete product and boundary selections in all four presets, tooling once per native platform in Debug, corpus once in Linux Debug, and the applicable OFF, installed-host and package gates. |
| Publication or merge | Existing required hosted checks on the actual candidate, plus local gates selected by the preceding rows. Hosted host-only results do not establish GPU or Resolve acceptance. |

For a leaf, freeze an assertion-to-change map before dispatch: each selected
case names the changed contract or actual consumer and the assertion it proves.
Compile and run one representative new native/C11 consumer first, then exercise
the required native platforms before broad fan-out. Run formatting and quality
on changed producer/helper translation units early. Once the candidate is stable,
run its required broad header-consumer quality closure once; do not repeat the same
closure before and after cutover. Keep newly affected consumers and required compile
contexts in scope even when their test family name differs.

The Rust compiler-fixture suite can run an existing unittest method directly with
the same environment used by `Quality.RustBoundaries`; this is a focused leaf check,
not a replacement for the full suite at a named broad checkpoint. Name the exact
method, Rust target, Debug/Release profile, enabled feature forms, artifact paths,
and expected fixture/check count in the handoff. For example, the grain-frame method
has six fixtures—complete, private, construction, default, mutation and borrow—and
checks both default and `test-support`, for 12 logical fixture/feature cases in
six Cargo checks per required profile/context. Accepted consumers, isolated E0451
construction and the remaining rejected fixtures form three batches; every required
primary code/marker remains checked in its own fixture file. Confirm the selected
method, printed fixture batches and six nonempty invocation logs; an empty selection
is invalid. The maintained prospective command
must be recorded with concrete preset paths and exact expected membership in the
existing handoff.

The Debug tooling selection also has a standalone CMake entry that does not
configure the product, CUDA or GoogleTest. Supply Python 3.13, native C/C++
tools, Git, Ninja, Rust/Cargo 1.98.1 and cargo-nextest 0.9.143. Configure it
with the platform's native compiler, then run it through the shared verifier.
For Linux:

```sh
cmake -S tests/tooling -B out/build/tooling-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-13
python -B scripts/verify.py qualification --preset linux-debug \
  --build-dir out/build/tooling-linux --reason 'Local tooling checks' \
  --selection tooling --runtime host --run-id tooling-local
```

On Windows use `out/build/tooling-windows` and `-DCMAKE_C_COMPILER=clang-cl.exe`
inside the qualified VS 18 environment. `Python3_EXECUTABLE` and
`JUICER_TOOLING_CARGO` can select qualified executables explicitly. Both entries
use `tests/ToolingTests.cmake` and register the same 13 Debug tooling members as
the product test suite. The verifier writes results under the native Debug preset's
`out/validation/<preset>/results/<run-id>/` directory. Linux also needs LLD for
the compiler-selected-linker freshness control.

Hosted Windows tooling uses three independent runners labelled `tooling-core`,
`tooling-BuildEvidence` and `tooling-Checkpoint`. The shard labels are selected
with an explicit CTest label regex; each shard result is focused and cannot be
combined into complete qualification. Every shard must succeed, and a failed
shard does not cancel the others. Local unfiltered tooling runs all 13 groups.

For WSL filesystem diagnosis, `JUICER_TOOLING_ARTIFACT_ROOT` can place synthetic
fixtures under an absolute native Linux `out/validation/linux-debug` root. Both
the product and standalone CMake entries accept this cache PATH option; it
defaults to the native preset's ordinary validation root. Product builds keep
other test artifacts at their usual locations. For the standalone entry, configure
a separate build directory on that filesystem and record both paths; do not
relocate an existing cache. This preserves the same tests and execution policy
while separating Windows-mounted filesystem costs from test execution. Retain
failed fixtures at their configured paths and keep the resulting qualification
explicit about the filesystem used.

The tooling job reports cumulative verifier scope using the existing source policy.
This rollout is report-only: all tooling still executes, and every aggregate child
must succeed. A naturally occurring candidate report and a separately reviewed
activation change are required before any skips. Missing/failed decisions never
mean a test passed.

Hosted Debug and Release product jobs prepare their selected host test closures
on each OS; Linux Debug also runs corpus checks. Each Debug job first prepares
`juicer_analysis_inputs` through the verifier, including ordinary objects,
production Rust and the generated diffusion header. Cumulative Debug native,
CUDA and Rust analysis then uses that job's compilation database; matching Rust
Release quality runs in the Release product job. Required artifact inspection can
still build GPU-linked test binaries without executing GPU tests. No separate
analysis job rebuilds either configuration, and no whole build cache is uploaded.
Tooling stays independent, including all Windows shards. The established required
host statuses aggregate the three surviving child jobs and accept only success,
including each product job's moved quality gate. Checkout still tests the event
ref; a PR base selects the cumulative quality diff rather than the checked-out
test candidate.

Linux product and production-probe links retain static cuFFT. Test executables
that exercise the native execution owner share the selected cuFFT runtime instead
of embedding its kernel database in every binary. Narrow host and CUDA-operator
tests retain only their actual library dependencies; ordinary and instrumented
native objects still use their separate existing owners. CMake supplies the test
runtime search path. Generic compile settings do not select CUDA libraries.

`Ffi.Host.BuildArtifacts` checks the actual configured outputs. On Linux it
verifies static product/probe symbols, shared test dependencies and resolved
cuFFT provider identity, loading and version, plus the absence of cuFFT from
narrow consumers. Both native platforms check that staged resources exactly
match the product's runtime resource list and original bytes. The default Linux
manifest covers two static links, 38 shared-runtime tests and 12 narrow binaries;
optional sanitizer executables add their own shared-runtime consumers. Test
resource staging excludes unconsumed artwork, generators and legacy payloads;
original source resources and historical fixture provenance remain unchanged.

Run each necessary platform/profile/package/feature combination once per stable
relevant source, tool and configuration context. During editing, use the smallest
owning checks; a stable leaf alone does not trigger the whole-product matrix.
Unknown dependencies justify broader affected checks with the uncertainty stated.
Keep reference provenance, exact identities/seeds, approved numerical exceptions,
operation order, ABI, immutable ownership and asynchronous lifetime obligations.
No tolerance widening, fixture regeneration during ordinary tests or omission of
required cases follows from this cadence.

Before a broad checkpoint, identify the exact candidate, affected scientific
expectations, actual consumers and required platform/profile/feature coverage.
Keep every approved numerical exception, fixture provenance, operation order,
ABI, ownership and asynchronous lifetime obligation. Do not widen tolerances,
regenerate checked-in fixtures during ordinary tests or omit required downstream
cases. A focused pass supports only the scope it names; complete qualification
uses the shared verifier and its complete fresh-membership collector, described
in [Shared native verification](#shared-native-verification).

For implementation work, use `prepare` only to build a declared target/file
closure. Use `development` for focused files, tests or targets, and
`qualification` for named CTest groups. These operations use fresh native owners
and results; a successful build, prior run or focused selection cannot be reused
to satisfy another qualification. The CLI examples and runtime selection rules
are collected below. Do not add per-change launchers, retry variants, matrix
drivers or collectors.

Use the current Python/CMake/Rust/CUDA versions in the Prerequisites section. Rust
Core/Bridge test binaries are prepared by CMake and executed through nextest once;
quality checks do not build or run a second Cargo test copy. The ordinary quality
entry keeps formatting, Clippy, source hygiene, native/CUDA analysis and compiler
boundary tests in their owning workflows. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
for its supported scope and local command choices.

At failure, retain diagnostic output, identify the cause and choose the smallest
check that can distinguish the proposed repair. Preserve the first failure while
working; a repeated result is not a root-cause fix. Do not raise deadlines, clear
Cargo state, delete outputs or restart presets without a concrete cause. Keep
unrelated failures unresolved and preserve required reference captures and
transition evidence.

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
- cargo-nextest 0.9.143 for CMake-owned Rust test execution.
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
cargo install cargo-nextest --version 0.9.143 --locked
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
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = Join-Path `
  $env:VSINSTALLDIR "VC\Tools\Llvm\x64\bin\lld-link.exe"
py -3.13 -m venv out/test-venv/windows
out/test-venv/windows/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r tests/requirements.txt
```

Cargo and direct Rust probes use that same VS LLVM linker. Hosted Windows CI
exports the setting during toolchain setup; set it in an ordinary developer
shell before running the tooling controls or standalone probes.

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
python -B scripts/verify.py qualification --preset linux-debug \
  --selection product boundary --runtime host \
  --reason "Product and boundary host checkpoint"
```

Windows Debug uses the same sequence with `windows-clang-debug`. Release uses
`linux-release` or `windows-clang-release`.

The Release testing-OFF contract builds its two existing C11 value probes as
`JuicerProductionFilmProbe` and `JuicerProductionOutputColorProbe`. CMake supplies
the real native engine and production Rust final link; their tracked C sources,
C11 assertions and value checks remain the same. These validation targets are
excluded from default product builds.

Test targets are excluded from the default product build. The verifier builds
the selected target closure before CTest; `ctest` alone expects its selected
binaries and Rust metadata to have been prepared. Unrelated test targets and
Rust test metadata are not unconditional `ALL` work.

For genuinely CUDA-independent work, use the same verifier and native preset:

```sh
python -B scripts/verify.py development --preset linux-debug --cuda-free \
  --tests Rust.Core Rust.Bridge
```

This configures `tests/host` in `out/build/<preset>/host-tests`, reusing the
parent preset's compiler settings and `cargo/{production,test-support}`. It
registers Rust.Core/Bridge, shared tooling/Rust quality groups, and the
HashContract and RenderAssertion native targets. No CUDA compiler or toolkit discovery is enabled. These are
shared definitions from `tests/CMakeLists.txt`; this configuration supplies no
product core, GPU tests, plug-in or alternate renderer. Product-linked `host`
tests continue to require CUDA for compilation. CudaAbi, ProfileOwner,
SpectralOwner and RouteBridge link the complete Rust plug-in archive and its real
native runtime dependency through the product entry. They remain host tests;
their native link requires CUDA 13.2 for compilation.

Debug tooling's `Quality.BuildRegistration` checks both configured graphs. Before
that tooling checkpoint, prepare the shared CUDA-free graph with
`verify.py prepare --preset <native-debug> --cuda-free --targets
JuicerHashContractTests JuicerRenderAssertionTests`. Product Debug registration
binds that graph and the product graph into the same complete tooling module.

Both configurations use the same preset build owner. The CUDA-independent
entry checks that ownership at configure/build time; invoke it through
`verify.py prepare --cuda-free --targets ...` or `development --cuda-free`.
Select explicit tests or targets: file-driven product quality and full checkpoint
selection stay on the product entry. Results identify `build_kind: cuda-free`
and remain focused. The existing standalone `tests/tooling` entry remains a
tooling-only configuration. Explicit immutable resource roots use canonical
`Resources/` directly. Module-relative lookup and package checks share the parent
preset's staged tree; selected immutable-only targets do not prepare that tree.

CMake always links `juicer` and the production prepared-render probe to
`film_juicer_rust`, built without default features in
`out/build/<preset>/cargo/production`. Facade and fault consumers select
`film_juicer_rust_test_support`, built separately with `test-support` in
`cargo/test-support`. Native object libraries and their neutral archive are shared;
each final consumer selects its Rust archive explicitly. Each variant has its own
`features.txt`, build stamp and archive byproduct. Enabling tests does not replace
the production archive. `Ffi.Host.BuildArtifacts` inspects the final link commands,
production symbols and strings, and native archive neutrality;
`Ffi.Gpu.ProductionPreparedRender` exercises cold and warm direct/print rendering
through production APIs. Its ordinary `--summary` invocation retains shapes,
compared sample counts, fixture-match and lifetime outcomes without duplicate
pixel arrays. The probe's default full-word output remains available to the
explicit production OFF validator; its capture contract is unchanged.

With `BUILD_TESTING=ON`, CMake's `juicer_rust_tests` target prepares both
`Rust.Core` and `Rust.Bridge` in the preset's single `cargo/test-support`
directory, saving nextest binaries-only and Cargo metadata. Cargo owns source
and included-fixture freshness. CTest discovers/runs these built binaries without
invoking another build. For a focused Rust rebuild, select
`cmake --build --preset <preset> --target juicer_rust_tests` before CTest.

`Rust.Core` exercises the production `state::Publication` authority with real,
resource-backed `recipe::prepare` results. Its cases cover ordered capture
tickets, registered host writes, revision-stamped snapshots, deferred reads and
conditional refresh, supplied input identity (including zero and equivalent
updates), all four routes, typed failures, obsolete outcomes and committed build
numbering. Concurrent admission, panic/poison/overflow blocking, fallible complete
backing and admitted holds through replacement/close/final destruction are also
covered. Test-only observations and bounded rendezvous establish overlap; scoped
workers join before completion.

Production `RenderInstance` delegates through the private C edge to safe Rust
`Instance` and `Publication`. The render callback retains one immutable admitted
`PreparedState` while the processor borrows it through synchronous processing.
Native capture identity and temporal/submission ownership remain separate from
publication. `Ffi.Host.PreparedState` checks the actual C11 protocol, while
`Ofx.Gpu.AdapterTrace` exercises getters, setters, callback reentry, late host-write
completion and exception/image-release behavior through the real adapter. These
fixture results do not establish installed Resolve retry behavior.

The presets use four CTest workers. GPU groups share `juicer_gpu`, the
product Cargo tests share `juicer_cargo`, and groups with `RUN_SERIAL` still run
alone. Keep GPU suites from separate Windows/WSL invocations serialized on a
shared device; CTest resource locks apply within one invocation. Override the
worker count with `--parallel <count>` when the machine needs a smaller limit.
Checkpoint host/GPU partitions use these same test presets, preserving their
worker accounting, named locks, exact membership and complete-result checks.

`Rust.Core` and `Rust.Bridge` each use one nextest worker and reserve one CTest slot. Tests that
spawn threads still exercise their internal concurrent publication and lifetime
witnesses. Ordinary tooling registrations have no elapsed kill deadline.
A larger explicit worker count can allow
overlap again. Verification control logs include the ten slowest unittest cases.
These scheduling choices preserve complete test membership and worker reservations.

Debug's `Optics.Host.DiffusionFrameProducts` also reserves four slots to
keep its full numerical fixture separate from other host tests at the standard
worker limit. It has no elapsed kill deadline and retains its complete case set.

`Ffi.Host.InputColorPreparation` uses two worker slots. Its admission hooks and
asset caches belong to its process, and its resources and parity fixtures are
read-only. It can overlap other eligible groups after the full build completes.
Use `--parallel 1` to roll back to serial execution when diagnosing contention.
Configuration and build writers must finish before tests read their build tree;
Cargo operations using the same target directory must also remain serialized.

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

Use the [validation cadence](#validation-cadence) to choose focused work or a
designated complete checkpoint. An unfiltered command is one way to execute a
complete preset; labels also select the exact checkpoint groups. Do not run label
selections again after a complete successful run merely to obtain another report.

For the bounded output-gamut/owner determinism investigation, use CTest's
`until-fail` repetition after the complete build. This is an explicit maintenance
run, supplementary to the unfiltered suite; ordinary CTest still runs every
group once. The exact cohort and preservation rules are in
[the FFI guide](ffi/README.md#bounded-cache-and-owner-repetition).
`Ctest.Host.Repetition` checks real CTest fresh-process repetition, a deliberate
second-attempt failure with no later retry, and rejection of empty selections.
Its synthetic CTest commands use the existing process-tree supervisor and output
lease. Successful outer controls, including expected failing children, dispose
their workspace after assertions and proven drainage. Failure, interruption,
unproved drainage or `JUICER_TEST_RETAIN_FIXTURES=1` retains and prints the path
under the preset's `ctest-repetition-controls/`. `JUICER_TEST_ARTIFACT_DIR` routes
output; it does not request retention.

The bounded Mallett-basis NPY pilot runs inside `Rust.Core` through the test-only
`data_io::npy_fuzz` module. Four fixed seeds (`4e505901`, `8103`, `deadbeef`,
`5eedcafe`) produce 512 cases each: unchanged unusual samples, truncation,
trailing bytes, bit flips, header deletion/insertion, header-length mutation and
payload mutation. Independent oracles check complete f32 sample bits, short reads
and extra payload rejection. Additional cases check supported f16/f32/f64 header
spellings and shape overflow. Every case also checks fragmented reads, typed
errors and owned results after source mutation. NaN, infinity and signed zero
remain valid authored content; this pilot adds no scientific admission policy.

For a focused replay, run the Rust core CTest owner through the shared verifier
and the standard preset CMake/Cargo target:

```sh
python -B scripts/verify.py development --preset linux-debug \
  --tests Rust.Core --runtime host --run-id npy-fuzz-replay
```

The explicit failing control sets `JUICER_NPY_FUZZ_CONTROL=nonfinite-clamp` for
that command. It simulates forbidden clamping inside the test and must fail the
sample oracle at seed `4e505901`, case 0. Keep the original nonzero exit and log;
clear the variable before ordinary tests. On a panic, the original input is saved
with seed/case in its filename. CTest sets `JUICER_TEST_ARTIFACT_DIR` to
`out/validation/<preset>/rust-core/`; direct Cargo uses an explicit override or
the `out/validation/rust-core/` fallback. An existing sample is reused only after
byte equality; different bytes get a new attempt filename. Save or diagnostic
write errors preserve the original panic, and an unsuccessful save is reported.
A partial write removes only the newly created incomplete file, with any removal
error included in the save diagnostic; existing samples remain intact.
This finite corpus provides structural regression evidence; it does not establish
general fuzz coverage, allocation-failure recovery or Rust sanitizer coverage.

Native CI uploads failed control evidence selected by
`tests/ctest/control_artifacts.py`. Existing checkpoint, build-evidence,
registration and repetition fixtures record their outer outcome in
`out/control-fixture.json` inside each synthetic workspace. Failed, interrupted
or unfinished controls retain their reproduction workspace, command receipts and
process diagnostics. Missing or malformed outcome markers remain unproved and
are selected within those four family directories. Successful expected-negative
controls are excluded; a blocked owner remains selected for diagnostics even if
the test marker says it passed. NPY
failure samples in each preset's `rust-core/` directory are included separately.
Each checkpoint fixture encloses its `project/` repository and any sibling
external inputs in one directory. Its outer outcome governs retention of the
whole directory, including external modules needed to reproduce a failed check.
Selected workspaces include hidden reproduction inputs such as Cargo configuration.
The shared result bundle and `LastTest.log` uploads remain unconditional. Local selection tests do
not establish hosted delivery; verify the uploaded artifact after download from
an authorized CI run.

## Shared native verification

Use the shared verifier for build preparation, focused development checks and
named qualification. Development and qualification create fresh native results
under `out/validation/<preset>/results/<run-id>/`; ordinary preparation is build-only.

`prepare` configures and builds the selected target/file closure. It does not run
tests or produce a qualification result. The explicit Release OFF
`--production-contract` mode above additionally executes the production closure:

```sh
python -B scripts/verify.py prepare --preset linux-debug --targets JuicerFilmRecipeTests --testing ON
python -B scripts/verify.py prepare --preset linux-release --targets juicer --testing OFF
```

`development` selects files, tests or targets and may specify the runtime
prerequisite. A focused result covers only its stated selection:

```sh
python -B scripts/verify.py development --preset linux-debug \
  --files rust/film-juicer-core/src/route.rs --runtime host --run-id route-edit
```

`qualification` builds the selected closure through the native owner, then runs
the exact selected CTest groups and required logical rows. `product`, `boundary`,
`tooling` and `corpus` are independent selections; `host`, `gpu` and `all` name
runtime prerequisites. A `--label` regex narrows the selected group and therefore
produces focused evidence only. `--build-dir` supplies an already configured
standalone tooling tree.

```sh
python -B scripts/verify.py qualification --preset linux-debug \
  --reason 'Boundary change' --selection boundary --runtime host --run-id boundary-edit
python -B scripts/verify.py qualification --preset linux-debug \
  --build-dir out/build/tooling-linux --reason 'Tooling checks' \
  --selection tooling --runtime host --run-id tooling-check
```

Every native operation supervises its selected build and test children, compares
source inputs at entry and exit, and publishes results only after descendants
have drained. `FJ_RUST_ROWS` and `FJ_HOST_CASES` carry the canonical current row
and case outcomes; these are the only live producers. The historical
`native-spatial-optics` decoder remains available only to offline reading of old
bundles. Results preserve actual selected functions, rows and operations.
Per-command `resource_usage` is optional OS-reported evidence: Windows Job
transfer bytes are logical transfers, not physical device I/O; Linux child block
counters remain block counts and are not converted to bytes. Measurement failure
is reported as unknown and does not affect command success or drainage.
Only GPU selections hold the shared physical-device lease. Bundles are flat and
can be inspected offline; a result for one preset or focused selection cannot be
assembled into a complete checkpoint by collection.

`collect-qualification` validates one complete fresh checkpoint for each
requested preset and rejects focused or mixed partial results:

```sh
python -B scripts/verify.py collect-qualification \
  --bundles <bundle-directory> --output <new-report.json>
```

Collection also accepts optional `--selection`, `--runtime` and `--presets`
arguments when the complete fresh checkpoint spans multiple requested presets.

Ordinary quality commands use the same native process owner. They do not admit
installed-tool hashes or reuse older successful runs. Rust naming and boundary
compiler fixtures remain CTest-owned; selected Rust test binaries are built by
CMake and run once through cargo-nextest, not rebuilt from inside the quality
command. Quality logs and diagnostics are kept under
`out/validation/<preset>/results/quality-<id>/`.

Native object inspections use `JUICER_PRODUCT_BUILD` supplied by the quality
command or CTest registration, so Release checks inspect the Release objects
regardless of the invocation directory. Windows compiler fixtures resolve
`clang-cl` from the active qualified VS 18 environment, including Community and
Enterprise installations.

The shared standalone and product tooling registration has 13 Debug tests:
`Quality.NativeResults`, `Quality.HostResults`, `Quality.GpuResults`,
`Quality.Dispatcher`, `Quality.ToolingLint`, `Quality.Execution`,
`Quality.BuildEvidence`, `Quality.BuildRegistration`, `Quality.Verify`,
`Quality.Checkpoint`, `Ctest.Host.Report`, `Ctest.Host.Repetition` and
`Ffi.Host.RustBuild`. All ordinary tooling registrations have explicit zero
timeouts. Windows CI runs three independent
shards: `tooling-core`, `tooling-BuildEvidence` and `tooling-Checkpoint`. A shard
uses its exact label regex and does not claim complete tooling qualification.

Native CI keeps product and boundary work in separate Debug/Release jobs, tooling
on each OS, and the corpus selection in Linux Debug. These host results do not
establish the Release testing-OFF artifact contract, GPU acceptance or installed
Resolve behavior. Keep existing scientific fixtures, domain assertions, C11/C++20/
Rust ABI checks and ownership/lifetime tests unchanged. See the [contributor
workflow](../CONTRIBUTING.md#required-checks) for supported local commands.

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
| `ffi/input_hull_*`, `ffi/test_input_hull_fixtures.py`, core `input_hull` | Exact ordered hull math, primitive context, shared ray, C11/core/native boundary, cache race/retry/retained ownership, complete caller TC bytes and admission | `host`; `Gamut.Host.InputHull.*` |
| `quality/test_check_quality.py` | Quality dispatcher selection, CUDA command translation and header owners, source hygiene, and failure propagation | `host` |
| `ctest/test_report.py` | Focused evidence extraction from full-suite JUnit, including missing, failed, skipped and malformed-input rejection | `host` |
| `quality/test_rust_naming.py` | Product naming-policy contract: current Rust code, accepted names, individually rejected names, test targets, and reasoned foreign-name exceptions under the actual workspace lints | `host`; pinned Cargo and Clippy required |
| `quality/test_tooling_lint.py` | Supporting-code runner scope, tool failures, argument boundaries and unchanged-input guarantees | `host`; lint executables are provisioned only for explicit lint runs |
| `quality/test_rust_boundaries.py` | Dependency/build contract and compiler-enforced profile/CSV privacy, borrowed-view lifetimes and safe-module prohibitions, with valid consumer controls | `host`; pinned Cargo required |
| `ffi/spectral_owner_test.cpp`, `ffi/spectral_bootstrap_test.cpp` | Raw spectral ABI, independently captured source values, lazy source acquisition, complete preparation and retained-owner lifetime/capacity checks | `host` |
| `ffi/illuminant_calibration_test.cpp` | Closed CSV ABI, independent seven-curve captures, raw source/cache holds, selected CMY calibration and actual complete-state publication/failure checks | `host` |
| `ffi/test_native_boundary.py` | Native operation/host/context boundaries and retired profile authority, including new nested source/header and forbidden-symbol controls | `host` |
| `ofx/probe.py` | Linux synthetic OFX load/describe/unload and captured describe properties; no parameter varargs or render | `host`, Linux only |
| `ofx/processor_reference_test.cpp` | Native OFX image/property seam driving the current CUDA processor for four routes, a combined optics/grain/print case, and signed-zero print transitions | `gpu` |
| `ofx/adapter_trace_test.cpp` | Genuine native OFX parameter/image fixture driving the actual `JuicerEffect` event and render callbacks through time, seek, preset/reset, undo/redo-like, and invalid/recovery sequences | `gpu` |

The quality dispatcher's three Clippy checks cover core, plugin production and
plugin `test-support` with all targets in the preset's matching profile. Across
all four presets these retain the required 12 native platform/profile/package/feature
combinations. Each native CI lane adds an explicit Release Rust selection
(`--files Cargo.toml`) after its Release build, alongside the existing Debug
quality selection.

`Quality.RustNaming` uses the current workspace manifests and Clippy configuration
in temporary copies. Required function/parameter outcomes remain in the native result.
The checked-in `quality/fixtures/rust_naming/*.rs` files are product naming-policy
contracts, not external references. Both crates are checked in the preset's
profile, including test targets. Expected failures must produce the specific
error code and primary source span for every rejected identifier; a compiler
setup failure is not a passing negative test. Valid names and a reasoned raw
binding exception must compile successfully. Cargo runs offline and locked;
the normal configured build supplies cached dependencies. Successful outer runs
remove their disposable source and log directories after all consumers finish.
Failures, interruptions, incomplete outcomes or `JUICER_TEST_RETAIN_FIXTURES=1`
retain the necessary workspace and diagnostics. Run it with
`ctest --preset <preset> -R '^Quality.RustNaming$' --output-on-failure`, or run its standalone harness with the configured tool and artifact environment.

`Quality.RustBoundaries` uses one temporary workspace per run. Both compiler-probe
groups reuse Cargo artifacts under `out/build/<preset>/cargo/quality-probes/`,
separately from product artifacts. Cargo's fingerprints and build lock govern
reuse; standalone runs compile every current probe and check its diagnostics.
Both groups apply the same outer-outcome scratch policy; expected compiler
rejections in a passing test do not pin disposable logs or source copies. Run
`ctest --preset <preset> -R '^Quality.RustBoundaries$' --output-on-failure` or its standalone harness. Cargo is offline and locked; the configured build supplies
the dependencies. Each negative fixture in `quality/fixtures/rust_boundaries/`
must fail for its expected compiler code at its own source location, not because
of missing tooling or an unrelated error. These are product API contracts, not
external numerical references. Valid use runs with and without `test-support`;
CTest selects the preset's profile; native CI/checkpoints retain both
development and release contracts. The probes compile without
linking or executing native CUDA calls.

Compiler probes default to `CARGO_INCREMENTAL=0` to avoid retaining incremental
sessions for disposable source variants. Product builds retain their configured
incremental policy. Set `CARGO_INCREMENTAL=1` explicitly when comparing probe
timings or choosing that cache tradeoff; the harnesses honor an existing setting.

At compiler-probe closeout, retain current results needed for review, active
diagnosis or rollback. Successful disposable compiler workspaces and per-command
logs are removed after drainage; required outcomes and candidate records remain.
The outer runner's optional compressed diagnostic stream follows the seven-day
window after explicit closeout described below. Review and reproduction holds
protect retained diagnostics. There is no background expiry service.

CTest owns compiler execution for both probe groups through the shared native
runner, including per-command descendant drainage before scratch cleanup.
Ordinary quality checks retain formatting, Clippy and native/CUDA analysis;
they do not dispatch these probes. Each current attempt keeps its required outcomes
and outer diagnostic stream; failed or held probe workspaces remain at their
reported paths. Probe-workspace Cargo configuration must match repository ancestry.

Relevance labels partition `product`, `boundary`, `tooling` and `corpus`;
`host`/`gpu` are independent prerequisites. CI executes product and boundary
groups in Debug and Release, tooling once per native platform in Debug, and
corpus once in Linux Debug. A focused run cannot stand in for an unselected group
or be combined with earlier results to claim a complete checkpoint. The verifier
collector validates fresh complete results per preset.

`JUICER_TEST_ARTIFACT_DIR` selects synthetic test output without requesting whole
fixture retention. Successful, fully drained workspaces clean up after assertions;
failures, interruptions, blocked owners and explicit retention preserve the
diagnostics and configured paths. Ordinary fixture cleanup does not sweep prior
runs or scientific capture inputs.

Completed result, candidate and required scientific/case files remain immutable.
Record explicit review and acceptance closeout separately on the matching native
platform; starting a new build does not imply acceptance:

```sh
python -B scripts/verify.py closeout --preset linux-debug --run <run-id> --accept --review closed
python -B scripts/verify.py closeout --preset linux-debug --run <run-id> --hold reproduction
python -B scripts/verify.py closeout --preset linux-debug --run <run-id> --release-hold reproduction
python -B scripts/verify.py closeout --preset linux-debug
```

`--review open` reopens review and clears its closeout clock. Holds are `review`,
`failure` and `reproduction`; acquire/release them under the same preset owner as
building and deletion. Only passed, unchanged and completely drained results can
be accepted. Seven days after explicit acceptance **and** review closure, an
unheld run's declared optional `diagnostics.gz` becomes eligible for expiry.
Missing/invalid closeout, active review, any hold, failed/undrained result,
missing required files or links/reparse traversal prevent deletion. File timestamps
do not authorize expiry. The immutable result remains readable offline after its
optional stream disappears; closeout does not reopen source, build or tool paths.

The next outer native invocation expires eligible diagnostics after acquiring its
preset owner and before build/test work; `closeout` performs the same pass without
running tools. Each pass visits at most 32 direct result directories and persists
one name cursor beside the preset's owner metadata. Directory names stream up to
that cursor; payloads and legacy trees are never traversed. The summary reports
visited directories, enumerated names, metadata reads/writes, actual deleted bytes,
deletion errors and whether another batch remains. Repeat `closeout` for a backlog;
no complete history inventory is needed. Failed deletion retains a retryable
disposition, including interruption between unlink and final metadata publication.
Expiry reads are included in native bookkeeping costs. Existing results without
explicit closeout keep their diagnostics; this is no historical cleanup command.

Asset-owner, spectral/exposure-product, bootstrap and illuminant tests create
isolated input workspaces under their configured scratch roots. Host preparation
tests copy only their consumed reconstruction, profile, illuminant and filter
assets; noise-owner tests keep the noise inputs they exercise. Successful outer
tests remove only the workspaces they created, after owner-drain checks pass.
Failures, interruptions and `JUICER_TEST_RETAIN_FIXTURES=1` preserve those inputs;
previous runs, logs and capture fixtures are not swept. A workspace-removal error
fails the test. Production OFF evidence retains its prepared-render executable
once under `production/build/tests/bin/`, serving both execution and build-input
identity checks.

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

`Ofx.Gpu.WorkspaceTransitions` uses the same isolated executable with
`--workspace-transitions` (GPU prerequisite, `juicer_gpu` lock, 300 seconds).
It checks the actual direct/print schedules and completed native workspace layouts,
logical grain work versus physical allocation, scanner/grain plane associations,
source/staged cache channels and the strict-YVV candidate order. The bounded seam
rejects the first zero through six outer candidates after normal backend validation;
its decision stays fixed through any reclaim retry of that candidate. Observations
count outer candidates and backend calls separately. Ordinary and context-loss
statuses stop the ladder. These synthetic rejections establish failure propagation,
not allocator failure, device loss or hardware recovery. Genuine pixel renders,
construction failures and injected failures are counted separately; existing grain,
DIR, scatter/halation and retirement groups retain the actual lifetime witnesses.

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

`juicer_test_resources` tracks each runtime file separately. Membership changes
remove obsolete staged files; a changed input copies only its own member, and
unchanged staging copies no payloads. `Quality.BuildRegistration` checks those
operations, missing inputs/outputs and the shared preset owner. Mutation fixtures
copy their explicit consumed inputs into private scratch, preserving canonical
files and avoiding writable links to them. Resource membership and scientific
fixture values are unchanged by staging.

Scatter profile mutations use a unique operation workspace under their requested
scratch root. Successful complete runs remove that workspace after CUDA shutdown
and owner destruction. Failures, incomplete owner closure and
`JUICER_TEST_RETAIN_FIXTURES=1` preserve it; existing siblings and explicit visual
or performance outputs are untouched.

## Diagnosing a failure

CTest prints the failing case, actual/expected values and relevant tolerance.
Run the named domain with `-R` and `--output-on-failure`; add `-V` when the
underlying command and paths are needed. GPU diagnostics retain case, route,
fixture and CUDA failure detail. A missing driver/device is a failure for a
selected `gpu` test, not a skip.

The real parallel analyzer control retains its complete failure directory in its
fresh native quality result bundle under
`out/validation/<preset>/results/quality-<id>/` and prints its location.
Standalone controls use the native Debug preset. Successful controls clean their
scratch directory; CI uploads retained command logs and raw process diagnostics.

Current native result bundles are flat outcomes for one fresh operation; a named
completed bundle does not admit or supply a pass to a later operation or
qualification. At closeout retain maintained scientific fixtures/provenance,
active work and required reproductions, and real unresolved failures. Frozen
historical acceptance obligations still require exact source identity or a
reconstructible patch, final case membership/outcomes, review/decision and limits,
with their required original receipt/log/process/payload closure. An old link or
receipt alone does not retain every intermediate attempt. Retire obsolete
historical scopes from reuse in a new disposition; never rewrite original hashes
or replace collector-required raw proof with a summary.

Ordinary raw success and redundant resolved-failure diagnostics without a continuing
consumer have a seven-day post-acceptance/resolution rollback window. Superseded
synthetic contents and verified redundant copies may go at closeout after checking
actual processes, descendants, leases, registered worktrees, path boundaries and
consumers. Expected-negative child receipts follow their outer test's outcome.
Preserve scientific exceptions and mandatory reproductions according to their
obligations. Review ordinary closed-slice narrative above 10 MiB, with scientific
data and required frozen historical evidence closure separately justified; this is not a deletion
quota. No automatic cleanup service is implied.

Native qualification runs CTest through the shared process owner and keeps its
JUnit output with that operation's flat result bundle. Use the qualification
command above when you need a CTest report tied to the selected build and test
execution.

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
finite `TIMEOUT` and its actual `host`/`gpu` requirement, and keep destructive
transitions isolated.

Fixtures must state whether they express a product contract, a Film-Juicer
characterization, or an external reference. Never regenerate or widen an
expectation from CTest. Reference updates are separate reviewed changes written
first to `out/validation/`.

## Boundaries and limitations

- The build still requires CUDA 13.2 even for `-L host`; a toolkit-free C++
  mode is outside this consolidation.
- Standard public CI runs Linux and Windows Debug and Release host selections,
  with one tracked native/CUDA/Rust quality pass per platform. The required job
  names retain `Debug host and Python checks` for branch-protection continuity.
  Each profile configures `BUILD_TESTING=ON` and uploads a separate host JUnit
  report; Release execution does not qualify the migration's distinct
  testing-OFF production Release gate. Both profiles execute their direct compiler
  probes; the quality pass retains ordinary Rust/native/CUDA analysis. Both lanes install and
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

The scratch-equivalence matrix retains 224 fresh frames in the original
route/layer/shared/time/debug order. Its seven host products per route and
layer/shared shape are reused only for the second time pass: time changes the
grain frame descriptor, which is rebuilt and checked exactly before preparation.
Each of the 112 distinct host configurations still constructs once. Per-frame
submission identities, dirty scratch, independent oracle allocations and every
finite/bitwise pixel assertion remain. Capacity failure, dedicated scratch,
queued abort, stream/context retirement/reset and instance ownership tests keep
their original independent setup; no host cold-construction or zero-call witness
is removed from those cases or the FFI admission matrix.

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

### Scanner spectral LUT construction

`Ffi.Host.ScannerLut` checks full six-array native-characterization words,
C11/C++/Rust ABI, raw/facade failure clearing, ten reservation stages, panic
containment and input expiry. Valid callback construction and span comparisons
stay inside the existing host-preparation scope. `Scanner.Host.LutProducts`
checks actual complete profile/route products and their held inputs.
`Scanner.Gpu.OutputEncoding` includes whole-array prepared upload comparisons,
cache reuse and resolution transitions; `Ofx.Gpu.ResourceFailure.ScannerLut`
exercises all six upload failure positions through actual render preparation.
The frozen `scanner/lut-fixtures/manifest.json` distinguishes native numerical
characterization from the safety/transport contract in `checked.json`.
Successful stored words are exact in each matching native platform/profile.
`Resource.Gpu.ScannerLut` exercises the actual native serving owner with the
same scoped Rust callbacks as held rendering. It checks all four route products,
warm hits with armed faults, individual construction/upload/retirement failures,
both owner rechecks, later gamut failure and upload backing expiry. Retirement
checks account exactly for prefix fences that are still queued or already reaped.
Cold null preparation rejects with the exact internal-failure status and no LUT
candidate, preserves any old six-array winner and clears route metadata. A
separate warmed-input case checks unchanged ledger/allocation accounting;
valid callbacks recover, while warm hits require no callback.
The fixture drains resources and pinned staging before destroying its CUDA context.
The four logical native captures share `lut-fixtures/native.json`; the manifest's
`contexts` map selects each preset's label and primitive results. Shared payload
and context fields must be disjoint. `storage_files` hashes the physical files;
the original per-preset hashes and capture/compiler/resource provenance remain
unchanged. `Ffi.Host.SharedFixtureFormat` reconstructs every original capture byte
for byte, including its line endings. Rust, raw ABI/C11 and prepared-upload
readers expand the same logical capture before their existing checks.

`Ffi.Host.ScannerLutDescriptor` and `Scanner.Host.LutDescriptorProducts` check
complete scanner spectral-LUT metadata through the safe Rust family, synchronous
legacy transport and gated facade. The independent
[descriptor fixtures](scanner/descriptor-fixtures/README.md) preserve 520 literal
cases and four products in four native contexts. Exact byte labels, NaN masks,
signed-zero equivalence, raw admission/zero clearing, closed failure envelopes,
C11/C++20/Rust ABI, expiry and actual complete-state early failure are covered.
Named missing-model-pointer outcomes remain historical dispositions in the
unchanged captures; the native model adapters are removed. Existing scanner
routes own the actual frame/cache/upload witnesses.
Compiler probes enforce private complete construction, readonly getters and owned
results. Both Release production gates execute 41 descriptor C11 cases alongside
the retained probes, with fixture/fault code excluded from normal artifacts.
These are bounded characterization and construction checks; installed Resolve,
visual acceptance, S4.D/E and parent closure remain separate.

### Scanner color construction

`Ffi.Host.ScannerColor` checks complete scanner output construction through the
Rust boundary and retained native adapter, with fixed-value C11/C++/Rust ABI,
full encoding even when CCTF is off, original-byte fallback identity, independent
failure clearing, staged faults, immutable copies and transport integrity. Its
[fixture guide](scanner/color-fixtures/README.md) identifies the unchanged native
supplement and reused output-colour/CAT02 expectations. `Ffi.Host.Cat02Preparation`
retains real route/publication/reuse and complete-product checks; the scanner
GPU output/reuse/correction and prepared-boundary cases remain downstream evidence.

### Scanner output encoding

`Ffi.Host.ScannerPostEffects` and `Scanner.Host.PostEffectsProducts` use the
bounded `scanner/post-effects-fixtures` supplement: 400 independently captured
whole native constructor cases with distinct Linux/Windows Debug/Release contexts.
Floats remain raw f32 words; all fields, radii, schema, exact hash, masking,
diagnostics and ordinary rejected attempts are checked through safe Rust, raw C
and actual complete-state products. The original producer executables and
source/configuration/tool receipts are retained in each context's provenance. Ordinary tests never
generate these expectations. The capture-only dispatch was removed at cutover.

The post-effects transport uses separate 32/56/64-byte input/result/failure
records, with independent output clearing and optional diagnostics. The permanent
48-byte CUDA execution record is unchanged. RadiusRange defines only consumed
finite int32 conversion overflow; multiplication overflow retains radius-zero
behavior. ON-only faults cover hash-zero, panic, descriptor rejection and recovery.
Both actual route schedules, prepared projection and Gaussian slot reuse/rebuild/
failed descriptor admission and retired-frame view rejection are exercised by
the owning host and scanner GPU groups.
Both native Release OFF probes additionally assert 410 exact post-effects results,
alongside every existing probe including Slice 8's 41 descriptor rows.

`Ffi.Host.ScannerCorrection` checks complete direct/print correction construction,
coordinated C11/C++20/Rust ABI facts, raw/facade failure and clearing rules,
all byte/flag demand combinations, independent fault consumption, integrity and
expired input ownership. `Scanner.Host.CorrectionProducts` checks 48 actual
route/state/resource inputs and their enclosing identities against the separately
adopted [native characterizations](scanner/correction-fixtures/manifest.json).
Both groups run through `JuicerFilmRecipeTests <resources> <manifest>
scanner-correction[-products] <preset>` and require no CUDA device at runtime.
The early `scanner-correction-primitives` maintenance mode compares native/Rust
math primitives; native whole-builder capture mode is retired after cutover.

`Scanner.Gpu.CorrectionRoutes` uses held prepared-state execution and raw C
bindings on all four routes, including active positive direct and both print
polarities. It checks correction-only LUT/print-resource reuse, exact repeated
and raw-boundary output, fault consumption/recovery and held immutable states.
`Ofx.Gpu.AdapterTrace` additionally checks actual
current getters, all correction flag combinations, encoded-level sanitation and
f64-to-f32 narrowing. Complete-state correction construction precedes publication;
native frame/resource execution retains the ordinary finish/abort owners.

The production C11 probe executes all 28 revision-1 correction cases and the
shared Rust demand query against each native Release testing-OFF archive. The
native exception captures and normative checked outcomes remain separate.

`Scanner.Gpu.OutputEncoding` calls the production CUDA scanner launchers and
checks all nine output spaces, CCTF on/off, optional output matrix, transfer
breakpoint neighbors, negative/HDR/subnormal/nonfinite values, RGB/RGBA, odd
image extents, and padded rows. The numerical oracle is Film-Juicer's independent
host double-precision `ScannerOracle::apply_encoding`; this is a product
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
owning suites cover those contracts. Frozen native words and matrices are
compiled into the owning test library; no private workbench files are required.

### CUDA ABI declarations

`Ffi.Host.CudaAbi` compares C11, C++20 and generated Rust layouts and type-checks
the CUDA C boundary signatures. `Rust.Bridge` includes internal ABI checks. See
[the FFI guide](ffi/README.md) for bounded commands and pinned binding maintenance.

`Ffi.Host.CudaOwner` covers Rust Runtime creation/borrowing through C11 and C++,
non-publication on failure, duplicate rejection, noncreating teardown, scoped
close/reload and consume-once retention after a controlled metadata-only shutdown
failure without a GPU. Isolated native lifecycle cases also exercise direct
`FjCuda` creation and consuming destruction.
`Ofx.Host.CudaOwnerUnload` verifies that the real factory's failed close returns
a fatal status through the OFX entry point and blocks replacement ownership.
Direct native fixtures explicitly own their runtime; `Root` access only borrows
it. `Ffi.Host.CudaTerminal.*` covers typed shutdown/destroy, no-retry retention,
exception containment and admission ordering. `ScatterHalation.Gpu.TerminalRetention`
checks retained GPU allocations/ledger charges; `Grain.Gpu.ScratchReuse` checks
instance retirement across exact contexts. See the FFI guide for the ownership
contract and the remaining host/hardware qualification limits.

`Ffi.Host.PreparedProjection` checks fixed C/native descriptor mappings and
malformed records. `Resource.Host.ScratchRequest` checks closed DIR/workspace
construction, private coupled storage, explicit inactive state, logical grain shape
and physical projection, extent/attachment constraints, canonical raw sentinels,
generation consistency and all eight accepted direct/print scratch generations.
These are product contracts and need no device at runtime.

`Ffi.Host.PreparedState` checks complete-state C11/C++/Rust layouts, clearing,
consume-once ownership and retained cache backing. `Prepare.Host.StateProducts`
compares complete Rust preparation with 31 ordered frozen expectations across
all twelve route/reconstruction combinations, perturbations and
competing failures. `Ffi.Gpu.PreparedStateLifetime` consumes held direct/print
states after enqueue with active spatial DIR and verifies native staging survives
borrow expiry. The fixture keeps optics, grain and physical effects disabled.
These groups add preparation/ownership evidence; production callback publication
and installed-host acceptance remain separate. See [the FFI guide](ffi/README.md#complete-prepared-state).

Prepared-state discovery returns before fixture or process-root setup. Actual
selected projection consumers load their frozen oracle through GoogleTest's
selected-test environment; ABI-only runs keep it unused. The two owning CTest
registrations independently assert unused/loaded state. Full projection selection
still requires all 31 rows in their frozen order, including wildcard/colon-list
filters; fixture provenance and numerical assertions are unchanged.

`Ffi.Gpu.PreparedBoundary.*` links a fixture-only C caller into the native executor.
It checks bit-exact equivalence between cold/warm C-boundary execution, the direct
executor and the processor adapter for all nine scenarios. Each case starts a
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
test facade. Active grain requires an explicit call-local noise view; missing
views reject before submission/resources, with no native fallback acquisition.
These seams exist only in test objects/test-support; normal Release has none.
Installed Resolve and actual driver/context-loss acceptance remain separate.

`Assets.Host.IlluminantConstruction` qualifies Rust illuminant math and its
synchronous C/native bindings against four frozen native-parent captures, with
exact products/identities, scoped source/lens lifetimes, cache and failure
contracts. The separate negative-print C render contract checks actual prepared
execution, abort and completion. See [the illuminant guide](ffi/README.md#illuminant-construction).

`Ffi.Host.PrintPreflash` checks the fixed preflash integral, compiled C11/C++20/Rust
ABI, actual C11/direct-facade agreement, independent failure clearing, diagnostics and
stage-11 recovery. `Print.Host.PreflashProducts` checks frozen native caller
products for both capture polarities and all three reconstruction methods, direct/
disabled omission, actual complete preparation and later executable-payload admission.
`Print.Gpu.PreflashCache` checks actual request/input preparation, full/partial hits,
descriptor dependencies, fault nonconsumption, failed commits and resource release.
The immutable `print/preflash-fixtures/manifest.json` binds native provenance and
exact words; ordinary tests never regenerate them. Finite double totals may narrow
successfully to float infinity here; the later active-preflash payload guard rejects
it. The superseded native wrapper/export and twelve standalone OFF rows are
removed; final I8 retains positive complete-state print execution and feature isolation.

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
raw owner exposes readonly spans and its consuming release returns the original
pointer/length/capacity to Rust once. Source and hull borrows expire on return.
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

### Output gamut construction and cache

`Gamut.Host.Output.{Math,boundary,binding,Cache,FixtureFormat}` checks the complete
nine-space native A9 family, exact identities/words, checked optional C diagnostics,
actual C11 layouts and calls, native failure binding and deterministic cache races.
The producer owns mathematical policy; the Library owns one nine-slot cache.
Warm hits retain a hold without foreign construction, copying or rehashing.
See [the output gamut guide](ffi/README.md#output-gamut-construction) and
[fixture provenance](ffi/fixtures/output_gamut/README.md). Existing product/GPU
suites retain distinct route, order, reuse, disabled-work and staging witnesses.

The optional Debug prepared-host ASan/UBSan pilot is documented in
[the FFI guide](ffi/README.md#prepared-host-sanitizer-pilot). It supplements the
complete ordinary suite and makes limited native instrumentation claims; Rust,
remaining core objects and CUDA devices are not covered by its instrumentation.

The output-color characterization in `tests/ffi/fixtures/output-color/` was
captured independently from the accepted native producer under all four presets
before translation. Ordinary tests compare its bytes/words and never regenerate
it. `Ffi.Host.OutputColor` checks fixed values, identities, C11 layout/value calls,
concurrent initialization, lifetime, clearing and faults. Prepared projection and
span tests require schema 1 plus a closed CCTF kind for both encoding states.
`Scanner.Gpu.OutputEncoding` uses the independent frozen double scanner oracle,
with unchanged maximum/mean error bounds, nonfinite and alpha/padding checks.


## Film development and route binding

`Ffi.Host.FilmReference` checks the complete borrowed film-reference boundary,
compiled C11/C++ layouts, exact frozen values/hash, independent output clearing,
raw admission, dependency faults and closed envelope integrity.
`Film.Host.ReferenceProducts` compares all 36 method/route/EV-auto products and
nine numerical observations per native profile, then exercises actual admission,
failure/recovery, supersession and actual held-owner release. Frozen native
provenance is under `film/reference-fixtures/`; the native capture entry has been
removed. Ordinary tests never regenerate expectations. `reference-primitives`
compares the actual native/Rust
log10 words before qualification. These are host construction checks.

`Ffi.Host.FilmDevelopment` compares the safe completed development family, raw
legacy transport and gated fixture facade against
[the native characterization](film/fixtures/manifest.json). It checks coordinated
C11/C++20/Rust ABI facts, invalid controls/tags, bounded reservation and raw view
failure, contained panics, consuming release and source-cache replacement while
a selected source remains held. It needs no device at runtime.

`Film.Host.DevelopmentProducts` covers every shipped capture profile on both
route selections, four film-gamma values and three grain policies. It compares
snapshot, development, synthetic-reference, scanner and final identities with
independent per-preset native captures. Admission failures must retain existing
immutable states and prevent publication of an intermediate identity. Rust core
tests also cover variable-length completed profiles and separately scoped native
primitive captures for NaN, infinity, signed-zero and 63/64/65-element hash masks.
These are Film-Juicer characterizations, not external-reference pixel fixtures.

`Ofx.Gpu.AdapterTrace` observes film-f32, print-f64 and resolved grain values from
actual current-value OFX capture, including invalid controls and recovery.
`Grain.Gpu.ScratchReuse` verifies that a required film-gamma host rebuild changes
development/final identities while reusing unchanged device curve/layer storage
and preserving every uploaded word. Identical snapshots instead reuse the whole
published state; equal curve hashes alone do not prove an absence of host work.

The immutable fixture directory is an ordinary test input. Capture generators
and maintenance environments are excluded from CTest. Generate and review
candidates under `out/validation/`; do not update expectations to obtain a pass.


Color/illuminant capture storage uses logical preset aliases and per-family
shared expected arrays. See the shared capture storage section in `ffi/README.md`.
Capture provenance and native numerical distinctions remain unchanged; logical
color identity still selects its qualified exposure supplement. The corpus
selection validates unique physical payloads, while each native numerical group
uses its own logical expectations. `Ffi.Host.SharedFixtureLoading` proves loader
admission and patch independence using owned scratch; ordinary fixtures stay
immutable.


Checkpoint CTest selections execute their exact host and GPU memberships in
separate supervised invocations under one build-tree owner. Host work runs
outside the shared GPU lease; GPU work holds it. Both invocations form one
relevance-group result: failure, missing cases or an incomplete process prevents
reuse of the whole group. Unchanged reuse validates every raw result and JUnit
report without acquiring the GPU lease. Ordinary CMake/CTest use is unchanged.

The audited host fixture/admission groups use process-local hooks and distinct
per-preset mutation roots; they have no global `RUN_SERIAL` requirement. Named
Cargo/GPU locks and build-tree ownership remain required. Read-only exposure and
spectral admission tests consume the staged immutable resources directly.
Input-color and TC fault matrices reuse only a prior immutable publication; each
fault, cold admission and recovery retains independent mutable state/counters.

Checkpoint registration admission rejects global serialization and duplicate
explicit `--scratch-root` destinations in the selected groups. Each such option
must occur once and name an absolute destination. Prefer isolated
per-case roots or the actual named resource lock. Implicit writer paths still
require review of their owning test; labels alone do not prove isolation.

`Ffi.Host.FilmInput` checks completed film-input construction through the direct
safe-core facade, synchronous legacy wrapper and a compiled C11 caller. Its
C11/C++20/Rust facts cover every field type, offset, extent, alignment and the
signature. `Film.Host.InputProducts` checks actual direct/print construction,
source/hull holds across cache detachment, selector priority, publication-slot
return/throw distinctions, supersession and recovery. Both compare the 96 base
and 44 control cases against independent native compound captures in
[the film-input fixtures](film/input-fixtures/README.md). The family owns complete
input color and sensitivity; later focused normalization and whole-state
publication retain their existing owners. Ordinary tests never regenerate these
expectations. Both native testing-OFF Release gates also exercise the new C
export's values and failures and exclude fixture/fault code from final artifacts.

`Ffi.Host.FilmRecipes` checks the completed DIR/minimum/film-bounds boundary,
C11/C++20/Rust field/layout/signature facts, independent development holds,
borrow expiry, raw owner release, typed failures and exact F1 diagnostics.
`Film.Host.DirDensityProducts` compares 64 complete products per configuration
with [independent native captures](film/dir-fixtures/README.md), including all RGB
axes, CMY bounds and snapshot/family/final identities, then exercises four-route
admission, recovery and unchanged reuse. Core tests also consume the frozen F1 and
focused captures. `Ofx.Gpu.AdapterTrace` observes actual conditional DIR capture,
single f64-to-f32 narrowing, polarity-selected K and invalid/valid recovery.
These focused groups do not establish installed Resolve or aggregate acceptance.


`Ffi.Host.PhysicalEffects` checks the stateless completed physical-effects recipe:
220 exact native cases through production/facade entries, all-field C11/C++20/Rust
ABI facts, independent output clearing, closed failures, diagnostic bounds, the
real computed-probability guard and panic/thread isolation, plus named native
projection. `Film.Host.PhysicalEffectsProducts` checks 92 genuine four-route
products, snapshot/grain/development/final identities and publication failure,
ordering, recovery and retained holds. Their immutable
[effects fixtures](film/effects-fixtures/README.md) retain all four native
configuration identities. Windows Release scratch expected length preserves its
native FMA contraction; other configurations preserve separate f32 rounding.
The existing adapter and cutover GPU groups also check actual physical-control
capture and each CUDA family, all-off behavior, completed-grain debug suppression
and alpha preservation. Both Release testing-OFF production gates compile, link
and run all nine effects cases against their final native archive. These scoped
checks do not establish installed Resolve, visual or aggregate acceptance.

`Ffi.Host.EffectsFrame` checks physical-effects frame construction through the
production ABI, facade and native bridge against 347 literal cases, 72 genuine
products and 57 independently captured primitives. `Film.Host.EffectsFrameProducts`
compares complete policy, geometry and identity fields for the genuine products.
The [frame fixtures](film/effects-frame-fixtures/README.md) retain four native
contexts and exact raw words, including the existing Windows Release static
scratch rounding. Each host group has a 180-second limit. Rust owns immutable
frame mathematics and identity; native code copies static policy and binds CUDA.

`Ofx.Gpu.EffectsFrame` checks both actual caller sites across all four routes,
with a GPU lock and 180-second limit. Pixel executions and successful construction
checks canceled before preparation are counted separately. Literal policy
substitutions stop before resource admission. Existing physical-family pixel,
prepared-resource and lifecycle groups retain insertion-point, alpha, all-off,
grain-debug and deterministic-coordinate assertions. Both Release testing-OFF
probes call the real `fj_legacy_effects_frame_prepare` with the shared strict C11
assertions and all 453 common-ABI full-result/failure/diagnostic cases. Native-only
missing-recipe and ON-only fault cohorts remain separate.

`Ffi.Host.SpatialOptics` checks all 553 frozen native literal recipes through
an actual C11 forwarder and the direct test facade, C11/C++20/Rust layouts and
signatures,
independent raw output admission/clearing, closed failure metadata, diagnostics,
call-bounded stage-9 panic and actual resolved-hash guard. Successful nonzero
lens values retain the render-preflight rejection check.
`Film.Host.SpatialOpticsProducts` checks 32 genuine products and
four-route publication, first-failure ordering, held states, recovery and authored
versus resolved identity. The [optics fixtures](film/optics-fixtures/README.md)
retain separate logical native configurations and immutable raw IEEE words.
Existing diffusion and scatter/halation fixtures prepare their recipes through
`fj_test_spatial_optics_prepare`; PSF/frame/operator expectations are unchanged.
The OFX adapter group compares all 96 captured getter/publication rows. The
superseded native wrapper/export and its 15 standalone OFF rows are removed;
actual complete-state optics preparation remains in Rust. Compiler privacy retains
the ordinary unbounded individual and boundary execution and two
isolated workers. These checks do not establish installed Resolve or visual
acceptance.


`Ffi.Host.PrintIlluminant` checks the fixed print illumination constructor through
an actual C11 forwarder and the direct facade, compiled C11/C++20/Rust facts,
exact approved sample words, actual Sample/Energy failures, independent clearing, optional diagnostics,
stage-10 panic recovery/thread isolation and owned lifetime.
It has explicit boundary relevance. `Print.Host.IlluminantProducts` checks 72
actual route/reconstruction/control products against immutable
[print illuminant fixtures](print/illuminant-fixtures/README.md), including direct
omission, main/preflash samples and consumed descriptor/balance/final identities.
Both are host groups without an elapsed deadline and with an exact preset argument.
The six `print_illuminant_*` compiler fixtures enforce construction, privacy,
read-only access and source-independent ownership under the ordinary execution policy.
The superseded native wrapper/export and twelve standalone OFF rows are removed.
Positive complete-state direct/print execution in both native Release testing-OFF
configurations remains a separate I8 gate. Test registration does not establish a
pass or numerical/caller/installed-host acceptance.

The owner-approved 20 October print numerical exception selects the existing
`libm::erff` 0.2.15 evaluator and accepts its characterized small native differences
and honest downstream content identities. Native captures and original failed
exact comparisons remain preserved; separate approved-Rust fixtures supply exact
expectations. No general floating-point tolerance or wider numerical exception is
used. This policy does not claim a new visual/scopes comparison for this slice.
Existing product fixtures retain their bytes; exact ordered supplements in the
print fixture directory align their dependent recipe/upload identities with the
selected evaluator. Unrelated product fields remain exact. Existing glare seed
composition follows the actual recipe hash, so identical pixels are not claimed.

The same supplement layer aligns older OFX, parameter-identity and processor
pixel expectations affected by the existing glare seed dependency. Original
fixtures and pixel bounds remain unchanged. See the print fixture README for
logical pixel-row aliases and measured changes; spectral ULP maxima must not be
used as image-pixel bounds.

`Ffi.Host.PrintBalance` compares all nine completed values and both identities
against independently adopted native words in all four presets. It checks raw
structure/diagnostics, first-failure order, u8 mode behavior, one-shot stage-12
panic consumption, thread isolation, native transport integrity and ownership.
`Print.Host.BalanceProducts` retains actual route/mode/dependency and immutable
publication/recovery cases. `Print.Gpu.BalanceCache` checks resolved binding,
source/normalization dependencies, unchanged-family residency/accounting and
absence of balance construction during resource preparation. These are bounded
Film-Juicer characterization/product contracts; ordinary tests never capture or
modify their frozen fixtures. Both Release OFF C11 probes retain sixteen frozen
balance cases and the retained boundary/feature-isolation checks.

### Shared Gaussian coefficients

`Ffi.Host.GaussianKernel` checks safe Rust optics, compiled C11/C++20/Rust ABI,
raw owner/view/release transport and call-local host-preparation callbacks against the
four independently captured native cohorts in `optics/gaussian-fixtures/`.
`Optics.Host.GaussianKernelProducts` consumes actual four-route scanner,
visual-grain and small FIR DIR products and compares every coefficient and its
metadata. Ordinary tests never regenerate expectations. The constructor preserves
ascending f64 exp accumulation and both f32 rounding points; family radius and
FIR/YVV selection remain with their existing owners.

`Resource.Gpu.GaussianKernel` is one 180-second process with `juicer_gpu` exclusion
and a real supported driver/device prerequisite. Its isolated resource object
exercises real candidate allocation, a one-shot synchronous-copy result, the real
map-node allocation rejection, deterministic two-upload insertion loser, explicit
cancellation, once-only physical candidate frees, old-state/ledger/Rust-owner
balance, recovery, warm hits and drained retirement. All waits are bounded to ten
seconds and workers join before fixture/context destruction. The unchanged
allocation probe is linked once per fixture executable. These are bounded injected
operation/allocation results, not physical driver failure or universal OOM proof.
Ordinary native objects and final testing-OFF artifacts exclude these controls.
Cold null preparation returns the exact preparation-failure envelope with no
candidate/cache/ledger/acquisition change; valid callbacks recover. Warm null
preparation returns the existing kernel without another acquisition.

The owning scanner, grain and DIR GPU groups also download their actual prepared
kernels and compare every word with the independent fixtures. Warm preparations
must not acquire another Rust coefficient owner. Existing route numerics,
identities, FIR/YVV separation, scratch and context-lifetime cases remain active.
The old Gaussian wrapper/export and standalone OFF Gaussian rows are removed.
Both Release OFF gates still require actual prepared rendering and the retained
post-effects/descriptor protocols; their execution remains a separate I8 gate.
The local Box allocation still has the established allocator-abort limit.

Gaussian Debug/Release captures share exact payload bytes within each platform;
four context sidecars preserve distinct provenance and original LF/CRLF capture
hashes. Logical cases, primitive exceptions and mutable reader isolation remain.
The print-illuminant test supplement is immutable per process and parsed once;
ordered patch validation and independent caller-owned patched JSON still apply.

### Frame domains and physical pixel size

`Ffi.Host.FrameGeometry` checks fixed C11/C++20/Rust layouts and signatures,
complete frozen values, independent clearing, raw admission, bounded diagnostics
and contained panic. `Frame.Host.GeometryProducts` checks the native projection's
complete signed fields, final f32 pixel word and its f64 diffusion widening.
Both are host groups without elapsed deadlines. The immutable
[frame fixtures](frame/geometry-fixtures/README.md) retain four native contexts.

`Ofx.Gpu.FrameGeometry` calls the real render callback in the existing isolated
adapter target, with valid pitched allocations for successful renders. It checks
same-state geometry changes, source/destination/window distinctions, cached source
pixel RoD/scale/PAR, live canonical RoD at the callback time, four routes, existing
full-domain restrictions, early failure order and image leases. Acquisition reads
metadata even with effects off; conditional effects consumption is proved by
three bounded completed-phase observations. Failed effects-domain conversion
retains the preceding canonical query and has no completed effects phase.

Both final native Release testing-OFF gates link the existing C11 film probe
against the exact Rust archive and check all 69 frame rows, plus all older probes.
The temporary callback bridge retires with its actual last native OFX consumer,
no later than S6.D; prepared-state integration remains separate.

### Scatter/halation frame construction

`Ffi.Host.HalationFrame` and `Optics.Host.HalationFrameProducts` exercise the fixed
Rust-backed twenty-one-record schedule boundary and completed route products.
Both need no driver/device and have no elapsed deadline. See the owning
[scatter/halation guide](scatter_halation/README.md) and
[fixture authority](scatter_halation/frame-fixtures/README.md). The existing
strict C11 Release testing-OFF film probe also executes the complete frozen
eighty-six-result membership alongside retained production probe results.

### Spatial DIR mathematical frame construction

`Ffi.Host.DirFrame` and `Film.Host.DirFrameProducts` check the fixed Rust boundary,
private complete results, ordinary failed progress and common native composition
against [four native captures](dir/frame-fixtures/README.md). Both are host groups
with 180-second limits. Every mathematical/default word and final native hash is
compared exactly; fixtures are never regenerated during tests.

`Ofx.Gpu.DirFrame` uses the existing executor-failure probe and GPU lock with a
180-second limit. Its 100 bounded invocations cover the two actual prepared route
sites, all four routes/polarities, current pixel/domain/origin facts, full-frame
fallback, inactive/Identity/FIR/YVV/mixed plans, zero-weight construction and
pre-builder/mathematical failure stops. Four genuine route products execute
pixels; the other successful constructions cancel at the existing first schedule
abort check before resource preparation. Existing DIR/cache, grain, prepared,
operator, route and lifecycle groups retain their complete downstream assertions.

Prepared projection retains the native decoder's distinct unconsumed zero-weight
target tiers and carries the original final hash. Both native Release OFF C11
probes execute 121 complete value/failure rows alongside the retained probes.
Reflected-YVV boundary construction has its own contract below. Wider workspace
construction, defect/weave mappings and S4.E integration remain later work.

### Spatial DIR reflected boundaries

`Ffi.Host.DirBoundary` and `Film.Host.DirBoundaryProducts` compare complete Rust
owners, key bytes, typed failures and native adapters against
[independent native captures](dir/boundary-fixtures/README.md). Each host group has
a 180-second limit. Comparisons cover 114 whole products and 56 bounded primitives
in each native profile, including the original explicit fused product residual.

The existing `Dir.Gpu.ExposureCache` group has no elapsed deadline and retains its GPU
lock. It also downloads entire actual Root boundary arrays, checks terminal
packing, warm reuse without acquisition, native staging after host-owner release,
failure rollback, same-key candidate disposal and context retirement. The native
owner still controls allocation, upload, cache publication and retirement.

Both Release testing-OFF C11 probes execute 171 full boundary value/failure/raw/
lifetime rows alongside retained probes. Diagnostic length counts
written bytes without NUL at capacities zero, one, short, M and M+1. Ordinary tests
never regenerate expectations.

### Visual-grain mathematical frames

`Ffi.Host.GrainFrame` and `Film.Host.GrainFrameProducts` compare complete raw
values, typed failures, native defaults and final native hashes against
[four independent native contexts](grain/frame-fixtures/README.md). Each host
group has a 180-second limit. The frozen cohort has 153 literals, 44 genuine
recipe/development products and 50 independently captured bounded primitives.
The core keeps original f64 Gaussian energy samples; uploaded f32 coefficient
arrays remain a separate owner. Published fields and hashes compare exactly.

The S4D6-R1 singleton curves `[[0,0,1],[1,0,1],[1,0,1]]` require all nine
row-major reductions, then a CMY loop that immediately computes each channel's
three layers after checking its total. C layer0 particle-count failure precedes
M total0: category3/API0/code0, reason2/field12/layer0/channel0. Legacy and facade
readers require cleared raw mathematics and native-default failed output.

`Ofx.Gpu.GrainFrame` uses the existing executor probe, GPU lock and 180-second
limit. Its 122 invocations distinguish 16 genuine pixel executions, 92 successful
constructions canceled before preparation and 14 failures. It checks both real
caller sites, four routes/polarities, current time/seeds/geometry, full-frame
fallback, inactive delegation and failure short-circuiting. Literal recipe
substitutions are restricted to construction paths. Existing scratch, grain
kernel, Gaussian-resource, prepared, route and lifecycle assertions stay intact.
The three native setup consumers use the maintained facade and shared composer.

Prepared transport preserves static recipe/chroma/amplitude/preset and layer-axis
metadata. Both native Release testing-OFF probes call the real
`fj_legacy_grain_frame_prepare` through strict C11 signatures and all 188 frozen
full-value/metadata/failure/diagnostic rows alongside retained probes.
Written-byte capacities 0/1/short/M/M+1 include R1. Ordinary tests never regenerate
fixtures; conversion safety expectations do not execute undefined native casts.

## Native result boundaries

The shared verifier uses the existing native preset trees and publishes a fresh
flat result directory for each invocation. Its offline collector checks complete
checkpoint membership; it does not combine focused results into a pass. GPU runs
hold the shared device lease only while GPU work executes. Ordinary host workflow
results do not establish GPU or installed-Resolve acceptance.

This workflow guidance leaves scientific fixtures, external-reference provenance,
ABI and lifetime assertions, and the established Release testing-OFF contract
unchanged. The host CI cutover does not perform OFF probes, a full platform matrix,
or establish shipping-feature isolation. It does not change the meaning of frozen
R4/I8 evidence or authorize later phases.
