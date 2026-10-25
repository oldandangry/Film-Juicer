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

## Validation cadence

Choose checks from the changed contract and the checkpoint being completed.
Before execution, name the supported target, test and quality commands, their
expected membership, and the affected native/CUDA/Rust compilation contexts and
platform/profile/feature combinations in the existing change description or
handoff. A focused handoff does not require complete aggregate qualification and
must not be described as full product acceptance.

| Change or checkpoint | Required checks |
| --- | --- |
| Documentation or status only | Whitespace, link/reference consistency and contextual review. No native build, product test or compiler gate solely for prose. |
| Editing a mathematical leaf or other bounded implementation | Touched formatting, compile/type checks, owning mathematical or behavioral cases and the nearest real consumer. Exercise touched C11/native fixture/helper compile cases and scoped quality checks on both native platforms early, before freezing for broad qualification. |
| Stable family cutover or source deletion | The affected family's prescribed numerical cases in Linux/Windows Debug/Release; affected ABI, complete-construction, failure, identity, cache, caller and downstream GPU route cases; scoped native/CUDA/Rust quality and independent contextual naming/ownership review. For a staged cutover, replacement fixture consumers and real caller evidence must pass before deleting their producer. As an alternative for a prospectively planned, reconstructible mathematical-producer cutover, capture the old producer independently and pass replacement core/ABI/fixture preparation before cutover; switch the caller and delete the old body in one candidate, then pass actual replacement-caller and downstream evidence before acceptance. This applies only to new handoffs; an existing handoff's stricter stage order remains unless explicitly amended. Lifecycle, ownership and other deletion gates remain. |
| New or changed ABI, exports, feature isolation or ownership boundary | Both native Release testing-OFF probes once on the stable boundary, plus relevant testing-ON boundary/lifetime checks. Changes to consumed probe inputs require fresh evidence. |
| Broad shared build/lifetime change, designated prepared-state integration, instance-state stage exit, OFX cutover, or final release qualification | Complete product and boundary selections in all four presets, tooling once per native platform in Debug, corpus once in Linux Debug, and the applicable OFF, installed-host and package gates. |
| Publication or merge | Existing required hosted checks on the actual candidate, plus local gates selected by the preceding rows. Hosted host-only results do not establish GPU or Resolve acceptance. |

Run each necessary platform/profile/package/feature combination once per stable
relevant source, tool and configuration context. During editing, use the smallest
owning checks; a stable leaf alone does not trigger the whole-product matrix.
Unknown dependencies justify broader affected checks with the uncertainty stated.
Keep reference provenance, exact identities/seeds, approved numerical exceptions,
operation order, ABI, immutable ownership and asynchronous lifetime obligations.
No tolerance widening, fixture regeneration during ordinary tests or omission of
required cases follows from this cadence.

Before a designated broad checkpoint, finish expectation adoption and inspect all
readers and downstream identity/seed consumers, including their later assertions.
Unsupported selectors do not justify an automatic `--targets all`; an
explicit `all` selection needs a stated dependency or checkpoint reason. Freeze
one candidate identity, then run its required selections once. Each partition
command must complete under supervision, drain its owned work and admit complete
exact JUnit membership. The enclosing outer owner must reach terminal completion,
drain its work and release the shared normal lease in a healthy state once the
operation ends before another owner is dispatched or a stable receipt is read.
Passing or reuse requires the applicable results to succeed. Apply existing
pending/failure admission rules for another owner; preserve failed status, while a
drained failure may permit a lawful repair or fresh preparation. Retries and
admission within the existing lease remain with that owner. Partial console output
or inner build success is not completion. Do not signal a healthy supervisor
except through its documented cancel path. Link original receipts from one final
packet and retain intermediate records only for a concrete consumer. Preserve
owner-required before/after source and tool hashes and completed receipts; do not
add a per-intermediate proof chain.
Keep separately required reference captures and transition checks. Do not edit
public inputs while their checks are running. Quality checks still bind the prose
they inspect; build leases do not exclude source writers.

The development dispatcher's supported inputs remain limited to those documented
in [CONTRIBUTING.md](../CONTRIBUTING.md#required-checks). For other work, use existing
explicit CMake targets, named CTest groups, native quality selections and Cargo
package/profile/feature commands. A nonempty focused pass can support its stated
handoff scope without manufacturing an aggregate pass. Use the existing wrappers
and domain capture modes; do not add per-change launchers, retry variants, matrix
drivers or collectors, including equivalent long inline scripts.

For preparation without a runtime selection, `scripts/verify.py prepare --preset
<preset> --targets <maintained-targets...> --testing ON|OFF` supports the four
standard native trees. See [target preparation](../CONTRIBUTING.md#required-checks)
for the bounded target list and Release OFF restrictions. This uses the existing
preparation/freshness and process owners; it does not supply product or boundary
test evidence. Retained interruption proof is still required for normal recovery.

This cadence does not change the current CLI or its admission rules. The separately
implemented Clippy correction checks only the preset's matching profile, with both
packages and plugin production/test-support forms retained across all four presets.
OFF execution now retains its inspected build files and configuration for collection
after ON restoration. Candidate snapshots and broad checkpoint discovery now omit
repository prose: root `README.md`/`CONTRIBUTING.md`, Markdown under `docs/` and
`tests/`, and explanatory `README.md`/`*-README.md` inside fixture directories.
Other fixture files, including Markdown payloads, stay relevant. Packaged documents
under `third_party/`, Resources, source/configuration metadata and unknown Markdown
locations stay relevant too. Documentation edits, additions, deletions and commits
within that prose scope do not invalidate reusable build/test evidence. Explicit
inputs and compiler/build-graph dependencies remain bound regardless of filename.
Quality runs still check and bind their selected prose; build/test reuse does not
qualify changed documentation. Older receipts retain their original recorded inputs
and may require fresh evidence; they are never rewritten or projected into this policy.

An explicit build or preparation command covers only its named targets and
dependencies. It cannot bypass pending owners, lease health, drainage or evidence
admission, and a successful build is not a qualification receipt.

Checkpoint preparation now separates a bounded set of proven runtime-only test
files from build inputs. Exact, absolute file operands under `tests/` in generated
CTest registrations are candidates; directories, manifest children and Python
imports are not inferred. The existing Ninja/compiler and ordered-include closure,
configuration inputs and build-command operands take precedence. Configure-time
copies, unresolved file reads and Cargo build scripts/embeddings retain conservative
scope. Compiled/embedded fixtures, generated inputs, copied/staged resources, tools
and artifact freshness remain preparation obligations. No directory or extension
is broadly excluded. Direct `prepare_build` explicit inputs remain authoritative.

Opaque configure file operations (`file`), subprocesses (`execute_process`) and
dynamic evaluation (`cmake_language`) keep preparation discovery broad. When that
scope needs fresh preparation, the checkpoint requests reconfiguration through the existing
producer and lease, so variable-built configure inputs refresh copied/staged output.
Literal filenames or subcommands are not proof of these operations' input closure.
Detection scans every recorded regeneration input regardless of filename suffix,
including build-tree, external and bundled modules. Non-UTF-8 input stays
conservative. Four exact `CMAKE_ROOT`-relative paths have reviewed CMake 4.3.1
LF/CRLF byte identities: `Modules/CMakeGenericSystem.cmake`,
`Modules/CTestTargets.cmake`, `Modules/Platform/WindowsPaths.cmake`, and the data
template `Templates/CTestScript.cmake.in`. These exceptions cover only known
bootstrap path conversions/directory creation and template data. Unknown bytes,
locations or CMake roots keep the conservative scan; source-discovered CMake
files and original raw-byte receipt requirements retain their existing treatment.
Projects without these unresolved consumers retain runtime-only preparation reuse.

Every required configure refresh retains fresh command, trace and process evidence.
When its generated-output declarations and exact redacted-trace hash match the
existing registry, the producer validates the original registry evidence and keeps
its raw bytes. Changed declarations, changed trace facts or unproved evidence
receive a fresh registry. Receipt-path changes alone therefore do not invalidate
unchanged artifacts; other configuration and dependency changes remain bound.
The miniature renewal replay requires exactly one configure refresh and one no-op
build per warm repair because its compiler/linker modules are opaque. It still
requires exact artifact bytes, zero invalidation and no repeated downstream work.

The same discovery serves preparation admission, refreshed membership and
collection. Fresh preparation observes initially excluded candidates before
commands and checks any newly admitted source input against that observation
before sealing refreshed identity. A fixture edit during reconfiguration cannot
be first admitted using its post-command contents. Candidates that remain
runtime-only stay excluded, and ordinary reuse takes no new admission observation.
Refresh and collection honor the producer's original dependencies;
older broad receipts and cold preparation without test registration may need one
fresh preparation to establish the narrower scope. Original receipts are never
rewritten or projected. Product, quality and runtime selections still bind runtime
files and require fresh execution after changes; complete aggregate membership,
pending-owner rejection and process drainage are unchanged. Avoid claiming native
recompilation merely because preparation redispatched or invalidated evidence.

General compilation/analysis/runtime separation beyond this boundary remains future
work, and collection still requires its current owners and artifacts. Do not claim those
remaining fixes are delivered or bypass rejection,
pending state or artifact checks. A prose edit requires only the prose checks above,
but does not make an older rejected receipt current. Record that limitation without
launching a broad run solely to repair its status. Keep actual ON and OFF execution
evidence and necessary artifacts distinct from cache restoration; rebuilt ON bytes
are not evidence that those bytes executed the earlier tests.

On failure, retain the diagnostic, identify its cause category and state the changed
hypothesis and smallest discriminating check before retrying. One identical retry
is permitted for an explicitly suspected transient; a second unchanged failure
requires diagnosis before another broad run. Do not raise deadlines, clean Cargo,
delete outputs or restart every preset without a concrete cause. Passing a repeat
establishes non-reproduction, not a root-cause fix. Preserve unrelated successful
work only through its existing admission rules, and leave unresolved work explicit.

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
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
ctest --preset linux-debug
```

Windows Debug uses the same sequence with `windows-clang-debug`. Release uses
`linux-release` or `windows-clang-release`.

With `BUILD_TESTING=ON`, the default build prepares the `Rust.Core` test
executables through `juicer_rust_core_tests` using the same Cargo command and
profile as CTest, with `--no-run`. Cargo checks source and included-fixture
freshness on each build. This keeps cold compilation outside the 120-second
CTest runtime budget. For a focused core-only rebuild, select
`cmake --build --preset <preset> --target juicer_rust_core_tests` before CTest.

The presets use four CTest workers. GPU groups share `juicer_gpu`, the
product Cargo tests share `juicer_cargo`, and groups with `RUN_SERIAL` still run
alone. Keep GPU suites from separate Windows/WSL invocations serialized on a
shared device; CTest resource locks apply within one invocation. Override the
worker count with `--parallel <count>` when the machine needs a smaller limit.

`Rust.Core` uses two libtest workers and reserves two CTest slots. Tests that
spawn threads still exercise their internal concurrent publication and lifetime
witnesses. On Windows, `Quality.Checkpoint` and `Quality.PreparationRenewal` each
reserve four slots so their repeated tool-byte validation runs alone at the
preset's four-worker limit. Their 600-second and 300-second deadlines remain
unchanged. A larger explicit worker count can allow overlap again. Verification
control logs include
the ten slowest unittest cases. These scheduling choices preserve complete test
membership, tool/input checks and the existing deadlines.

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

For a focused replay, use the standard preset Cargo directory and native target:

```sh
cargo test --locked -p film-juicer-core --lib --target <native-target> --target-dir out/build/<preset>/cargo data_io::npy_fuzz -- --nocapture
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
are selected within those four family directories. Successful expected-negative controls, blocked simulations
and explicit successful retention are excluded from automatic uploads. NPY
failure samples in each preset's `rust-core/` directory are included separately.
Each checkpoint fixture encloses its `project/` repository and any sibling
external inputs in one directory. Its outer outcome governs retention of the
whole directory, including external modules needed to reproduce a failed check.
Selected workspaces include hidden reproduction inputs such as Cargo configuration.
JUnit and `LastTest.log` uploads remain unconditional. Local selection tests do
not establish hosted delivery; verify the uploaded artifact after download from
an authorized CI run.

A1/A2 are accepted only for the four inputs in
[the contributor workflow](../CONTRIBUTING.md). Timing targets, general ownership
coverage and full qualification remain unproved. The bounded path selects `src/ScanRoute.cpp`, `src/ScanRoute.h`, Rust core `route.rs` and core
`color/output.rs` on the configured `linux-debug` tree. For example:
`python -B scripts/verify.py development --preset linux-debug --files rust/film-juicer-core/src/route.rs`.
Explicit entries beside CTest registration select the route bridge or output-color
FFI group and exact fixture; Rust scopes also select direct boundary probes and
named active-profile Cargo tests/Clippy. Uncertain public-interface changes add
plugin checks. Header compilation retains all actual configured variants without
selecting tests merely because they link the same library.

Initialization needs no old preparation/development receipts. Unsupported changes,
index-only edits and pending failures remain unresolved. A later, overlapping or
pending preparation invalidates affected prior reuse. A successful selected scope
can coexist with an incomplete request (exit 2). Unchanged reuse reports that no
checks executed, and unrelated saved scopes remain separate. Ordinary Rust edits
use Cargo incrementally; stale-time content edits invalidate only affected active
packages/reverse consumers. Missing or unresolved header consumers cannot pass.
Dirty unrelated fixture/resource payloads stay unresolved without global content
hashing. Selected fixture contents remain preparation inputs; tool, log, JUnit and
process-proof contents remain reuse inputs. Header discovery and preparation are
checked again before publishing success, including newly introduced consumers.
Unchanged reuse performs one fresh admission against the retained pass; it does
not repeat the same admission when no checks execute. Parsed compilation entries
and ordered include discovery are shared within one validation boundary, with
fresh discovery after configure, build, analysis and CTest operations.
For unchanged latency, count the complete supervised command as bookkeeping;
receipt phase timings end before publication and do not replace that full cost.

Named checkpoints and aggregate collection are available after A3's focused
verification; exact baseline recovery and cumulative size proof are recorded in
the current handoff. Use `scripts/verify.py qualification --preset <preset>
--reason <reason> --selection <groups...> --runtime <host|all>`; selections are
`product`, `boundary`, `tooling`, `corpus` or the separate Release/testing-OFF
`production` gate. Product selection requires `--base`. Exact membership prints
before execution. Successful stages reuse original records across campaign names;
failed or incomplete gates remain pending. Legacy sealed-build adapters are
unsupported. `collect-qualification` accepts only the complete explicit native,
profile, runtime, corpus and production gate set described in CONTRIBUTING.md.
Mixed checkpoint selections run their host and GPU parts separately under one
build-tree owner and one aggregate group result. Only the GPU part holds the
shared device lease; reuse requires exact membership and completed results from
every part. Enabling the dispatcher does not establish full product qualification.
Development verification selects individual existing state, process, native/transitive,
generated-input, shadow-header and partial-build controls. It does not dispatch the
entire build-evidence group or change product fixtures. Control artifacts remain
under the selected preset's validation directory. No full product/platform
qualification is claimed by this scope.
`Quality.BuildRegistration` uses the same outer-outcome retention rules, including
subtest/setup/cleanup errors and blocked owners. Its fixtures live under
`JUICER_TEST_ARTIFACT_DIR/registration-controls/`, or the native Debug preset's
`registration-controls/` when run directly without an artifact root. Completed
commands do not authorize releasing an interrupted fixture owner. Both fixture
families include focused lifecycle controls in their existing discovery groups.
Windows `Quality.Execution` covers the exact-path, Job-owned VCTIP cleanup,
natural-drain grace, other-child wait and cap, and failed-parent behavior;
`Quality.BuildEvidence` covers generated-header reconfiguration and unowned
shadow headers. Both run in the normal Debug tooling selection in Windows CI.

Build-evidence fault variants that start from the same prepared state share a
fixture within one test method and report each fault as a named subtest. Between
variants, the test restores its owned inputs, artifacts and provenance at the
original configured path, then checks preparation again before the next fault.
Each variant still runs the real build/publication path and checks rejection,
pending state and owner preservation. Cold preparation, renewal, missing-scope
admission and recovery remain separate scenarios. File-identity and lease tests
use lightweight fixtures without Cargo setup. Production freshness checks and
CTest deadlines are unchanged.

Preparation and checkpoint entry points validate stage names, declared membership,
nonempty commands/inputs and finite positive timeouts before creating pending
owners or reusing results. Invalid requests leave prior receipts, artifacts and
absent/pending owners unchanged. Focused build-evidence and checkpoint regressions
exercise rejection and subsequent valid reuse; actual execution or evidence
validation failures still block reuse until repaired through the owning path.

Interrupted preparation recovery requires a healthy normal build-tree lease and
the original attempt's sealed drainage proof. Every started configure, Cargo clean
and build command must have matching terminal process, result and output receipts.
Only the original live owner may attest that a failed attempt started no command.
A retry preserves the old pending marker and failed receipts before fresh
preparation; the scope's successful baseline remains until replacement succeeds.
Unproved, damaged or legacy owners remain blocked. This does not recover a hard
owner death or take over a blocked lease. Focused compiler-free fixtures exercise
real descendant interruption, successful fresh recovery and proof rejection before
outputs or ownership mutate.

Native build fixtures start with the C stub. Rust cases explicitly construct
their two-crate workspace once when enabling Rust, before any source mutation;
later preparation does not recreate edited or missing inputs. Initial sanity
checks in artifact/fallback/fresh-scope cases execute the native value witness
directly. Their selected/final CTest runs and failure/recovery executions still
exercise the real verification path.

During implementation, build the affected test targets and run only the owning
cases needed to check the current edit. Run new compiled cases on Linux and
Windows early so platform-specific compilation, lifetime or diagnostic issues
are found before the final matrix. For example, quality-tool changes can start
with these short development checks in each native Python environment:

```sh
python -m unittest discover -s tests/quality -p test_check_quality.py -v
python -m unittest discover -s tests/ctest -p test_report.py -v
```

At a designated complete checkpoint, run the required preset selections once.
When using an unfiltered suite, do not precede it with another broad focused pass
solely to produce a separate receipt. Record focused-domain results from that same
full-suite JUnit report:

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
| `ffi/input_hull_*`, `ffi/test_input_hull_fixtures.py`, core `input_hull` | Exact ordered hull math, primitive context, shared ray, C11/core/native boundary, cache race/retry/retained ownership, complete caller TC bytes and admission | `host`; `Gamut.Host.InputHull.*` |
| `quality/test_check_quality.py` | Quality dispatcher selection, CUDA command translation and header owners, source hygiene, and failure propagation | `host` |
| `ctest/test_report.py` | Focused evidence extraction from full-suite JUnit, including missing, failed, skipped and malformed-input rejection | `host` |
| `quality/test_rust_naming.py` | Product naming-policy contract: current Rust code, accepted names, individually rejected names, test targets, and reasoned foreign-name exceptions under the actual workspace lints | `host`; pinned Cargo and Clippy required |
| `quality/test_tooling_lint.py` | Supporting-code runner scope, tool failures, argument boundaries and unchanged-input guarantees | `host`; lint executables are provisioned only for explicit lint runs |
| `quality/test_rust_boundaries.py` | Dependency/build contract and compiler-enforced profile/CSV privacy, borrowed-view lifetimes and safe-module prohibitions, with valid consumer controls | `host`; pinned Cargo required |
| `ffi/spectral_owner_test.cpp`, `ffi/spectral_bootstrap_test.cpp` | Production spectral ABI, independent native copies, fail-closed bootstrap, selected computation and measured lifetime/capacity checks | `host` |
| `ffi/illuminant_calibration_test.cpp` | Closed CSV ABI, independent seven-curve captures, retained snapshots, selected CMY calibration and actual read/probe allocation transport through native recipes | `host` |
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
in temporary copies under `out/validation/<preset>/quality/rust-naming/`.
The checked-in `quality/fixtures/rust_naming/*.rs` files are product naming-policy
contracts, not external references. Both crates are checked in the preset's
profile, including test targets. Expected failures must produce the specific
error code and primary source span for every rejected identifier; a compiler
setup failure is not a passing negative test. Valid names and a reasoned raw
binding exception must compile successfully. Cargo runs offline and locked;
the normal configured build supplies cached dependencies. Logs remain in the
artifact directory after temporary workspace cleanup. Run it with
`ctest --preset <preset> -R '^Quality.RustNaming$' --output-on-failure`, or run its standalone harness with the configured tool and artifact environment.

`Quality.RustBoundaries` uses one temporary workspace per run. Both compiler-probe
groups reuse Cargo artifacts under `out/build/<preset>/cargo/quality-probes/`,
separately from product artifacts. Cargo's fingerprints and build lock govern
reuse; standalone runs compile every current probe and check its diagnostics.
The groups remove temporary source workspaces on completion and keep
logs under `out/validation/<preset>/quality/rust-boundaries/`. Run
`ctest --preset <preset> -R '^Quality.RustBoundaries$' --output-on-failure` or its standalone harness. Cargo is offline and locked; the configured build supplies
the dependencies. Each negative fixture in `quality/fixtures/rust_boundaries/`
must fail for its expected compiler code at its own source location, not because
of missing tooling or an unrelated error. These are product API contracts, not
external numerical references. Valid use runs with and without `test-support`;
CTest selects the preset's profile; native CI/checkpoints retain both
development and release contracts. The probes compile without
linking or executing native CUDA calls.

At compiler-probe closeout, the owning change selects the runs still needed for
reusable evidence, active diagnosis or rollback. Ordinary superseded success logs
follow the seven-day retention window in [failure diagnosis](#diagnosing-a-failure);
their mere presence does not require indefinite retention. Preserve original
receipts and required raw closure; prove outer outcome, consumers and drainage
before disposal. There is no background expiry or separate retention collector.

CTest directly owns compiler execution for both probe groups. Ordinary quality
checks retain formatting, Clippy and native/CUDA analysis; they do not dispatch
these probes. No publisher, manifest, alias receipt or recursive runtime seal is
involved. Compiler commands have finite timeouts and keep separate diagnostics
for every run. Probe-workspace Cargo configuration must match repository ancestry.

Relevance labels form one disjoint partition: `product`, `boundary`, `tooling`,
`corpus`; `host`/`gpu` are independent prerequisites. Both native CI jobs execute
product and boundary groups for Debug and Release. Tooling executes once per native
platform in Debug; corpus executes once in Linux Debug. The focused checkpoint
controls cover exact execution membership, input/configuration invalidation,
cross-campaign reuse, retained failures and inspected final-artifact binding.
Successful production OFF scopes retain their inspected archives/module/PDB and
consumed build configuration under the original production evidence directory.
Collection verifies those execution-bound copies, source/fixture membership,
tools, logs and process drainage without requiring the restored ON tree to match
OFF bytes. A receipt with `scope_status=passed` and `status=incomplete` remains
eligible after unrelated pending work is repaired; unresolved work and newer
failed production owners still prevent aggregate completion. Fresh production
execution remains restricted to a testing-OFF Release tree. Older receipts without
retained build inputs require fresh OFF execution; never backfill or rewrite them.
`Quality.Checkpoint` replaces retired publication/sealing protocol controls.
Its control module keeps its 600-second limit. `Quality.PreparationRenewal` owns
the complete native miniature repair/checkpoint/collection replay separately,
with a 600-second limit on Linux/WSL and a 300-second limit on Windows.
Both are required host/tooling tests in Debug. The
miniature replay does not establish real product, GPU or OFX qualification.
`JUICER_TEST_ARTIFACT_DIR` supplied by CTest selects artifact output without
requesting whole-fixture retention. Checkpoint fixtures use the native preset's
`checkpoint-controls`; BuildEvidence routes its fixtures below the artifact root.
Successful, fully drained synthetic workspaces clean up after the outer test
consumes their receipts. Set `JUICER_TEST_RETAIN_FIXTURES=1` for deliberate diagnostic
capture. Outer failures (including subtests), interrupted/undrained work and
unproved owners retain their diagnostics and configured paths.

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

The real parallel analyzer control retains its complete failure directory under
`out/validation/<preset>/quality/parallel-controls/` and prints its location.
Standalone controls use the native Debug preset. Successful controls clean their
scratch directory; CI uploads retained command logs and raw process diagnostics.

At closeout retain maintained scientific fixtures/provenance, active work and
required reproductions, real unresolved failures, and a small named set of reusable
executions with complete original receipt/log/process/payload closure. Historical
acceptance needs exact source identity or a reconstructible patch, final case
membership/outcomes, review/decision and limits. An old link or receipt alone does
not retain every intermediate attempt. Retire obsolete scopes from reuse in a new
disposition; never rewrite original hashes or replace collector-required raw proof
with a summary.

Ordinary raw success and redundant resolved-failure diagnostics without a continuing
consumer have a seven-day post-acceptance/resolution rollback window. Superseded
synthetic contents and verified redundant copies may go at closeout after checking
actual processes, descendants, leases, registered worktrees, path boundaries and
consumers. Expected-negative child receipts follow their outer test's outcome.
Preserve scientific exceptions and mandatory reproductions according to their
obligations. Review ordinary closed-slice narrative above 10 MiB, with scientific
data and required reusable closure separately justified; this is not a deletion
quota. No automatic cleanup service is implied.

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

`Ffi.Host.ScannerLut` checks the complete immutable host owner, full six-array
native words, C11/C++/Rust ABI, raw/facade failure clearing, ten reservation
stages, panic containment and input expiry. `Scanner.Host.LutProducts` checks
matching native profile/route products and the native move-only adapter.
`Scanner.Gpu.OutputEncoding` includes whole-array prepared upload comparisons,
cache reuse and resolution transitions; `Ofx.Gpu.ResourceFailure.ScannerLut`
exercises all six upload failure positions through actual render preparation.
The frozen `scanner/lut-fixtures/manifest.json` distinguishes native numerical
characterization from the safety/transport contract in `checked.json`.
Successful stored words are exact in each matching native platform/profile.
The four logical native captures share `lut-fixtures/native.json`; the manifest's
`contexts` map selects each preset's label and primitive results. Shared payload
and context fields must be disjoint. `storage_files` hashes the physical files;
the original per-preset hashes and capture/compiler/resource provenance remain
unchanged. `Ffi.Host.SharedFixtureFormat` reconstructs every original capture byte
for byte, including its line endings. Rust, ABI/native, prepared-upload and Release
testing-OFF readers expand the same logical capture before their existing checks.

`Ffi.Host.ScannerLutDescriptor` and `Scanner.Host.LutDescriptorProducts` check
complete scanner spectral-LUT metadata through the safe Rust family, synchronous
legacy transport, gated facade and same-signature native adapters. The independent
[descriptor fixtures](scanner/descriptor-fixtures/README.md) preserve 520 literal
cases and four products in four native contexts. Exact byte labels, NaN masks,
signed-zero equivalence, unrestricted native polarity/resolution lanes, admission
asymmetries, native defaults versus raw zero clearing, closed failure envelopes,
C11/C++20/Rust ABI, projection integrity, expiry and direct-state early failure
are covered. Existing scanner routes own the actual frame/cache/upload witnesses.
Compiler probes enforce private complete construction, readonly getters and owned
results. Both Release production gates execute 41 descriptor C11 cases alongside
every older probe, with fixture/fault code excluded from normal artifacts.
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
and native readers. The original producer executables and source/configuration/
tool receipts are retained in each context's provenance. Ordinary tests never
generate these expectations. The capture-only dispatch was removed at cutover.

The post-effects transport uses separate 32/56/64-byte input/result/failure
records, with independent output clearing and optional diagnostics. The permanent
48-byte CUDA execution record is unchanged. RadiusRange defines only consumed
finite int32 conversion overflow; multiplication overflow retains radius-zero
behavior. ON-only faults cover hash-zero, panic, projection rejection and recovery.
Both actual route schedules, prepared projection and Gaussian slot reuse/rebuild/
failed descriptor admission and retired-frame view rejection are exercised by
the owning host and scanner GPU groups.
Both native Release OFF probes additionally assert 410 exact post-effects results,
alongside every existing probe including Slice 8's 41 descriptor rows.

`Ffi.Host.ScannerCorrection` checks complete direct/print correction construction,
coordinated C11/C++20/Rust ABI facts, raw/facade/native failure and clearing rules,
all byte/flag demand combinations, independent fault consumption, integrity and
expired input ownership. `Scanner.Host.CorrectionProducts` checks 48 actual
route/state/resource inputs and their enclosing identities against the separately
adopted [native characterizations](scanner/correction-fixtures/manifest.json).
Both groups run through `JuicerFilmRecipeTests <resources> <manifest>
scanner-correction[-products] <preset>` and require no CUDA device at runtime.
The early `scanner-correction-primitives` maintenance mode compares native/Rust
math primitives; native whole-builder capture mode is retired after cutover.

`Scanner.Gpu.CorrectionRoutes` uses actual legacy execution and prepared C
projection on all four routes, including active positive direct and both print
polarities. It checks correction-only LUT/print-resource reuse, exact repeated
and projected output, frame-stage computed failure, fault consumption/recovery
and held immutable recipes. `Ofx.Gpu.AdapterTrace` additionally checks actual
current getters, all correction flag combinations, encoded-level sanitation and
f64-to-f32 narrowing. These witnesses preserve the correction construction point
after state publication and retain the ordinary finish/abort owners.

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

`Ffi.Host.PrintPreflash` checks the fixed preflash integral, compiled C11/C++20/Rust
ABI, raw/facade/native agreement, independent failure clearing, diagnostics and
stage-11 recovery. `Print.Host.PreflashProducts` checks frozen native caller
products for both capture polarities and all three reconstruction methods, direct/
disabled omission, the unconditional builder and later executable-payload admission.
`Print.Gpu.PreflashCache` checks actual request/input preparation, full/partial hits,
descriptor dependencies, fault nonconsumption, failed commits and resource release.
The immutable `print/preflash-fixtures/manifest.json` binds native provenance and
exact words; ordinary tests never regenerate them. Finite double totals may narrow
successfully to float infinity here; the later active-preflash payload guard rejects
it. Both Release testing-OFF witnesses execute twelve explicit preflash C11 cases
alongside every older production probe.

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
raw admission, dependency faults and native envelope integrity.
`Film.Host.ReferenceProducts` compares all 36 method/route/EV-auto products and
nine numerical observations per native profile, then exercises actual admission,
failure/recovery, supersession and temporary Development hold release. Frozen
native provenance is under `film/reference-fixtures/`; capture maintenance uses
the existing `JuicerFilmRecipeTests <resources> unused reference-capture <preset> <new-output>`
entry and cannot overwrite an existing candidate. Replacement captures never
establish expectations. `reference-primitives` compares the actual native/Rust
log10 words before qualification. These are host construction checks.

`Ffi.Host.FilmDevelopment` compares the safe completed development family, raw
legacy transport, gated fixture facade and native source projection against
[the native characterization](film/fixtures/manifest.json). It checks coordinated
C11/C++20/Rust ABI facts, invalid controls/tags, bounded reservation and copy
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
borrow expiry, native move/copy/unwind, typed failures and exact F1 diagnostics.
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

`Ffi.Host.SpatialOptics` checks all 553 frozen native literal recipes through the
production and test facade entries, C11/C++20/Rust layouts and signatures,
independent raw output admission/clearing, closed failure metadata, diagnostics,
call-bounded stage-9 panic and actual resolved-hash guard, and complete native
projection. `Film.Host.SpatialOpticsProducts` checks 32 genuine products and
four-route publication, first-failure ordering, held states, recovery and authored
versus resolved identity. The [optics fixtures](film/optics-fixtures/README.md)
retain separate logical native configurations and immutable raw IEEE words.
Existing diffusion and scatter/halation fixtures prepare their recipes through
`fj_test_spatial_optics_prepare`; PSF/frame/operator expectations are unchanged.
The OFX adapter group compares all 96 captured getter/publication rows. Both
Release testing-OFF production gates execute 15 optics cases against the actual
archive, including complete seeded semantic clearing. Compiler privacy retains
the existing 60-second individual and 240-second boundary deadlines and two
isolated workers. These checks do not establish installed Resolve or visual
acceptance.


`Ffi.Host.PrintIlluminant` checks the fixed print illumination constructor through
production and facade entries, compiled C11/C++20/Rust facts, exact approved sample
words, actual Sample/Energy failures, independent clearing, optional diagnostics,
stage-10 panic recovery/thread isolation, owned lifetime and native projection.
It has explicit boundary relevance. `Print.Host.IlluminantProducts` checks 72
actual route/reconstruction/control products against immutable
[print illuminant fixtures](print/illuminant-fixtures/README.md), including direct
omission, main/preflash samples and consumed descriptor/balance/final identities.
Both are host groups with a 180-second deadline and an exact preset argument.
The six `print_illuminant_*` compiler fixtures enforce construction, privacy,
read-only access and source-independent ownership under existing deadlines.
The Release production probe includes twelve print illuminant calls, with frozen
approved pinned-erff words and exact typed failure envelopes; successful OFF execution on each
platform is required separately. Test registration does not establish a pass or
numerical/caller/installed-host acceptance.

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
balance cases and all earlier boundary/feature-isolation checks.

### Shared Gaussian coefficients

`Ffi.Host.GaussianKernel` checks safe Rust optics, compiled C11/C++20/Rust ABI,
checked owner/view/release transport and the native move-only hold against the
four independently captured native cohorts in `optics/gaussian-fixtures/`.
`Optics.Host.GaussianKernelProducts` reconstructs actual four-route scanner,
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

The owning scanner, grain and DIR GPU groups also download their actual prepared
kernels and compare every word with the independent fixtures. Warm preparations
must not acquire another Rust coefficient owner. Existing route numerics,
identities, FIR/YVV separation, scratch and context-lifetime cases remain active.
Both Release OFF C11 probes execute 201 Gaussian value/failure cases, preserving
all earlier probe memberships, including 410 post-effects and 41 LUT descriptor
rows. The local Box allocation still has the established allocator-abort limit.

### Frame domains and physical pixel size

`Ffi.Host.FrameGeometry` checks fixed C11/C++20/Rust layouts and signatures,
complete frozen values, independent clearing, raw admission, bounded diagnostics
and contained panic. `Frame.Host.GeometryProducts` checks the native projection's
complete signed fields, final f32 pixel word and its f64 diffusion widening.
Both are host groups with 180-second deadlines. The immutable
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
Both need no driver/device and have 180-second deadlines. See the owning
[scatter/halation guide](scatter_halation/README.md) and
[fixture authority](scatter_halation/frame-fixtures/README.md). The existing
strict C11 Release testing-OFF film probe also executes the complete frozen
eighty-six-result membership, retaining all earlier production probe results.
