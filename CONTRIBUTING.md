# Contributing to Film-Juicer

Film-Juicer keeps photographic policy in immutable recipes and descriptors,
with CUDA code binding or executing those decisions. Preserve arithmetic
order, units, channel order, hashes, seeds, stage order, external identifiers,
and asynchronous lifetime while changing implementation language or structure.

## Design and ownership

- Give each process, instance, context, frame, and local resource one clear
  owner and teardown path. Prefer plain values, short-lived borrows, and RAII.
- Add a type, field, cache, or helper only for a current consumer. Keep helpers
  in the owning module and avoid generic managers, service layers, or fallback
  paths without a present contract.
- Resource files are trusted inputs. Decode supported representations and enforce
  the immediate consumer's necessary structure and memory/indexing safety;
  do not validate authored scientific content. Preserve established conversions,
  computations, and approved numerical exceptions. Complete immutable values
  are trusted downstream until a real external, numerical, or lifetime boundary
  intervenes. Unrelated user-control validation remains required.
- Rust core code is safe and CUDA/OFX independent. Raw foreign operations stay
  at the named plug-in or native boundary; native CUDA resources remain owned
  by the exact context and epoch.

## Naming

Use the shortest name that preserves the distinction visible at the call site.
Rust types and variants use `PascalCase`; functions, modules, fields, and values
use `snake_case`; constants use `SCREAMING_SNAKE_CASE`. Familiar type acronyms
are `Rgb`, `Lut`, `Ofx`, and `Cuda`. New native free functions use
`snake_case`, while retained owners may keep their established readable member
casing. Preserve unit and representation names such as `camera_ev`,
`sigma_px`, `radius_um`, `row_bytes`, `density_cmy`, and `context_epoch`.
External OFX strings, profile keys, ABI tags, and fixture identifiers do not
change as part of an internal rename.

Both handwritten Rust crates, including existing code and test targets, inherit
the workspace naming lints. Nonstandard Rust casing and uppercase type/variant
acronyms are errors; compound names such as `RgbToXyz` receive acronym checks
too. Clippy checks public cross-crate APIs as well as private items. These crates
ship together, so public Rust visibility is not a naming exemption.

Keep exact foreign spellings in raw bindings and explicit export/schema mappings.
Use a narrowly scoped, reasoned `#[expect(...)]` for a required naming exception;
stale expectations fail. Generated bindings may need a reasoned module-scoped
exception limited to their generated declarations. Handwritten wrappers retain
Rust names. Do not suppress naming lints across a crate to preserve C++ spelling.

Improve migrated names coherently with their consumers; valid snake case alone
does not establish a good Rust API. File/module placement, vocabulary, conversion
semantics, and searchability remain review responsibilities. No name-length limits
or word blacklists apply.

## Failure behavior

Expected input, allocation, preparation, CUDA, and host failures return typed
outcomes. Do not silently reuse older state, skip an enabled effect, parse
diagnostic text as policy, or introduce an output-changing fallback. Cleanup
and panic/exception containment belong at the owning boundary. Never reset a
CUDA context owned by a live Resolve process.

## Code review

Every code review includes correctness and regression risks plus contextual
naming and design review, unless the requester explicitly narrows the scope.
This applies to ordinary changes as well as Rust migration packages; no separate
request for G-Names or code hygiene is needed.

Review changed definitions and representative call sites, producers, and consumers
against the naming, ownership, and failure rules above. In particular:

- Check clear domain vocabulary, redundant context, units, stage/channel
  distinctions, and discoverable foreign-name mappings at actual use sites.
- Verify that validation, ownership, readiness, and completion implied by names
  match the implementation. During migration, assess names in their Rust module
  context rather than accepting mechanical C++ transliterations.
- Trace authoritative policy, construction/validation, permitted mutation,
  resource lifetime, completion, and failure propagation across touched boundaries.
- Question added state, caches, wrappers, and abstractions without a current
  consumer or invariant; identify duplicated policy and unnecessary complexity.

Keep review bounded to the requested change and the context needed to judge it.
Report concrete violations with locations, the affected rule, and their practical
consequence. Do not request equivalent renames or unrelated cleanup. Automated
checks provide mechanical evidence; they do not establish contextual review.
Within the required response format, briefly state whether contextual review was
completed for the requested scope and identify any limits, even when no findings
were found. Do not invent findings to demonstrate coverage. Migration handoffs
retain this assessment as their G-Names and contextual G-Quality result.

## Required checks

The test guide's [validation cadence](tests/README.md#validation-cadence) is the
common rule for selecting checks during editing, family cutover and broader
integration/release checkpoints. Documentation-only work needs whitespace,
link/reference checks and contextual review. A focused handoff does not require
complete aggregate qualification. The commands below describe available checks;
they are not a list to run for every change.

Supporting-code checks need Python 3.13 and the pinned tools installed explicitly:

```sh
python -m pip install -r scripts/quality-requirements.txt
python -m unittest discover -s tests/quality -p test_tooling_lint.py -v
python scripts/check-tooling.py
```

The command checks maintained Python and shell under `scripts/` and `tests/`,
plus GitHub workflows. It includes eligible untracked files and excludes fixture,
generated, vendor, private and output trees. Ruff uses `E4,E7,E9,F`; ShellCheck
resolves sourced scripts; actionlint uses the selected ShellCheck with Pyflakes
disabled. It neither modifies source nor installs tools, and missing/wrong tools,
lint failures or input changes fail the command. The independent Linux lint job
requires no CUDA setup. Its unit tests also run as `Quality.ToolingLint` in CTest;
ordinary CTest does not require the lint executables. Private workbench checks
are separate and are never inputs to public CI. Lints complement contextual review.

Rust is pinned by `rust-toolchain.toml` to 1.98.1 with its rustfmt and Clippy
components. Native formatting and tidy use LLVM 22; on Windows the qualified
tools are under
`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin`.
The commands below expose individual checks. A handoff names the supported target,
test and quality commands and their expected membership. For a leaf, map each
selected assertion to the changed contract or actual consumer, compile and run one
representative new native/C11 consumer before platform fan-out, and run changed
producer/helper quality early. After the candidate stabilizes, run its required
broad header-consumer quality closure once. Rust compiler fixtures may use their
existing named unittest method with the configured target/profile/features and
explicit expected nonempty membership; retain the full suite at its named broad
checkpoint. Record the concrete method command and all environment values in the
handoff. Use `scripts/verify.py`
within its supported development or checkpoint scope. Other focused work uses
explicit owning CMake targets, named CTest groups, native quality selections and
Cargo package/profile/feature commands. An unsupported selector does not trigger an
automatic `--targets all`; explicitly selecting `all` requires a stated affected
dependency or checkpoint reason. This complete-preset example is for a designated
broad checkpoint; setup explicitly enables tests before CTest:

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
python scripts/check-quality.py --base <accepted-parent> --preset linux-debug --jobs 2
ctest --preset linux-debug --output-on-failure
```

Use the corresponding Windows or Release preset when that is the affected
target. Keep the [CTest worker accounting](tests/README.md#build-and-run) when
reproducing CI timing failures: the default is four slots, `Rust.Core` reserves
two and uses two libtest workers, and Windows `Quality.Checkpoint` reserves four.
Internal race witnesses and all freshness checks remain active. Preserve the
first failed log and use the reported slow-case timings before changing setup or
scheduling; do not raise a deadline to hide contention.
`--files <paths...>` is the bounded package form; `--all-owned` is an
explicit audit. The dispatcher writes logs only under
`out/validation/<preset>/quality/runs/<run-id>/`, leaves source unchanged, and treats missing
tools or compiler/lint failures as incomplete work. Each fresh directory has a
`run.json` with source identities, command, terminal status, elapsed time and hashes
of that invocation's evidence. Keep the printed run path in a handoff; do not copy
the shared quality tree or use an older log as evidence of the current pass.
Historical logs remain untouched. CI uploads the run receipts and command sidecars.
With `--base`, a valid diff with no owned files still runs the diff and boundary
checks and completes receipt validation; the receipt records its empty selection
and exclusion reasons. This includes clean and deletion-only diffs. Empty
`--files` or `--all-owned` selections remain errors, as do invalid Git references
and discovered owned files that disappear before selection completes.
Receipts bind the Git revision, current owned source (including untracked and
generated inputs), policies, compilation database/build configuration and consumed
tools, dependency sources and project headers. These identities are checked again
before success; a changed or missing input fails qualification. Environment values
are hashed rather than printed. This inventory excludes historical output trees.
Formatting and lints do
not replace review of ownership, names, numerical parity, GPU lifetime, or
installed Resolve behavior.

Reuse the configured checkout path, including its casing, from
`CMAKE_HOME_DIRECTORY` in the preset's `CMakeCache.txt`. A differently spelled
filesystem alias can change compiler arguments and invalidate incremental builds
even when it reaches the same files. Keep configure, build and quality invocations
on that path; the quality dispatcher rejects a different spelling. Do not edit
cache paths to hide a relocation.

The bounded local development path uses the existing configured `linux-debug`
tree. A1/A2 are accepted only for the four inputs below. Use this path for their
owning checks; unsupported work remains unresolved. These results do not establish
timing targets, general ownership coverage or full qualification. Its selections are:

| Selected input | Owning development coverage |
| --- | --- |
| `src/ScanRoute.cpp` | `JuicerRouteBridgeTests`, selected native analysis, `Route.Host.Bridge` |
| `rust/film-juicer-core/src/route.rs` | Same native route witness, filtered core Cargo tests/Clippy, direct `Quality.RustBoundaries` |
| `src/ScanRoute.h` | Every discovered compile consumer variant, native/CUDA analysis, `Route.Host.Bridge` |
| `rust/film-juicer-core/src/color/output.rs` | `JuicerOutputColorTests`, filtered core Cargo tests/Clippy, `Ffi.Host.OutputColor`, direct boundaries, exact output-color fixture |

Run with the qualified Python 3.13 environment:

```sh
python -B scripts/verify.py development --preset linux-debug --files src/ScanRoute.cpp
```

The explicit owners in `tests/route/development.json` and
`tests/ffi/output-color-development.json` are consumed beside CTest registration.
Only the selected owners contribute runtime tests.
CMake/Ninja build its required dependencies. Selected native sources receive the existing hygiene, formatter and analyzer
checks; shared headers retain all actual configured compilation variants. Rust
uses the active Debug target and named packages/test filters. Proven edits to
existing route match results or private output-color literals need core checks;
public interface changes or uncertain edits add plugin tests/Clippy, including
production and required test-support features. Direct boundary probes remain
conservative for both Rust scopes. This is a route-adapter result, not full
route integration or platform qualification. No complete-build receipt, artifact
seal, unrelated fixtures or recursive runtime inventory is required.

No previous preparation or development receipt is needed to initialize. Current
dirty work starts unverified; unsupported paths and pending failures remain
unresolved. An unreadable pending record blocks admission. The latest per-preset
record is authoritative, so an older receipt cannot discard pending work. Use
`--previous <printed-development.json>` for the next edit or unchanged reuse, or
repeat the explicit command. A second edit to a dirty file invalidates its pass. Index and working-tree
changes are observed separately, including cancelling deltas.
A matching successful scope may be reused after relevant-input, configuration,
command and evidence checks; the output explicitly says that tests were not rerun.
An unchanged invocation admits that pass once using fresh complete consumer,
dependency, include-namespace and preparation-owner checks. Include discovery is
shared only within that validation boundary. Configure, build, analysis and CTest
operations require new discovery; no content or timestamp cache crosses invocations.
Preparation publication consumes the writer's just-checked post-build context;
it still checks the written command/process proof and required output existence.
Independent preparation readers always rediscover current inputs.
Missing outputs or incompatible inputs require preparation again.
Global change observation records dirty fixture/resource payloads as unresolved
without reading their contents. Selected fixtures and consumed tools, logs and
process proofs still receive content checks through their owning preparation and
reuse admission. Changing scopes does not change the global observation.

Development records include observation and the selection, preparation or reuse
admission timings for their branch. Their `seconds` starts before the build-tree lease and ends at the start
of publication; stdout reports publication time separately. Use the full
supervised command time for latency claims, including startup, lease release and
process drainage. All unchanged-invocation time is bookkeeping because no build,
analysis or tests execute.

`scope_status: passed` reports only the printed selected scope. Exit 0 means no
unresolved paths remain; exit 2 means that scope passed but other work remains
pending; exit 1 means failure or unsupported admission. Unsupported requested files
cannot hide observed changes. Other files and domains remain unsupported; no automatic qualification fallback exists.

For preparation alone, use `python -B scripts/verify.py prepare --preset linux-debug
--files src/ScanRoute.cpp`. It builds the same explicit target and preserves the
pending development record. A later preparation replaces the target’s producer;
earlier passes cannot be reused across that replacement or a pending attempt.
Overlapping target sets invalidate owners of outputs they may replace. Successful
scopes remain in the existing development record and can be reused independently
when their inputs and current preparation still match; prior runtime selections
are not unioned into a new run. Preparation is not test success.

Explicit target preparation is also available on all four standard native
Debug/Release trees, independently of the bounded development file selector:

```sh
python -B scripts/verify.py prepare --preset linux-debug --targets JuicerFilmRecipeTests
python -B scripts/verify.py prepare --preset linux-release --targets juicer --testing OFF
python -B scripts/verify.py prepare --preset linux-release --targets JuicerFilmRecipeTests --testing ON
```

Run through the qualified native environment at the configured source path.
Maintained targets are `juicer`, `JuicerFilmRecipeTests`, `JuicerRouteBridgeTests`,
`JuicerOutputColorTests`, and explicitly requested `all` by itself. Use `all` alone
only for a stated affected dependency or checkpoint reason; unsupported selectors
never widen to it automatically. Testing defaults to ON;
OFF requires a Release preset and `juicer` or `all`. Each invocation reconfigures
the requested testing state and builds only those targets and their dependencies
under the existing build-tree lease, supervisor, deadlines and freshness owner.
It dispatches no quality or runtime-test selection. It preserves development
pending state and never supplies product, boundary or production qualification.
Direct builds and preparation cover only their named targets and dependencies;
they cannot bypass owners, pending markers, normal lease health or drainage, and
cannot substitute for an admitted evidence receipt. Other pending target scopes
must first be resolved by their preparation owner; normal same-scope interruption
recovery still requires sealed drainage proof.

The one-time `retire-s4c1-pending --preset linux-debug|linux-release` command
implements the owner's 21 October 2026 amendment for exactly two retained legacy
markers. It requires the exact source/build paths, marker bytes, pinned retained
command/log/process evidence, complete preparation-attempt membership and a
healthy normal lease. Debug additionally requires the verified later Linux boot.
It retains original markers and any superseded completed owner, records
`retired-unqualified`, and removes only those active records. Fresh preparation
is a separate command. It does not change historical results or authorize other
legacy recovery; changed, unknown, undrained or active records and blocked or
hard-death leases remain rejected. Remove this exceptional entry after the two
dispositions have been reviewed and retained; it is not a force switch.

Normal Rust edits use Cargo incrementally. A content edit with restored/stale
timestamps uses supported Cargo cleaning only for affected packages, reverse
consumers and the active target/profile; necessary host build-script/proc-macro
products are included. Unchanged preparation preserves the archive and successful
Cargo stamp. Missing depfiles use compiler entries and ordered includes; unresolved
macro/generated consumers fail visibly instead of reporting an empty pass.

Named native checkpoints and aggregate collection are available after A3's
focused verification. Exact retained baseline objects establish the cumulative
verifier/control reduction; the current handoff records counts and evidence.
Enabling the dispatcher does not establish full product or phase qualification.
Ordinary CMake/CTest, standalone probes and bounded development remain available.

`scripts/verify.py qualification` implements an explicit native checkpoint. Name the
preset, reason, runtime prerequisite and required relevance selections:

```text
python scripts/verify.py qualification --preset linux-debug --reason "Rust boundary change" --selection boundary --runtime host
```

CTest registration owns disjoint `product`, `boundary`, `tooling` and `corpus`
labels, alongside `host`/`gpu` prerequisites. Product checkpoints also require
`--base <accepted-revision>` and include preparation and ordinary quality.
Preparation admission is read-only by default. Build explicitly with
`python scripts/verify.py prepare --preset <preset> --targets all --testing ON`,
or add `--prepare` to explicitly authorize preparation in the qualification command.
An admission mismatch reports the cause without invalidating existing outputs.
Both commands use the same all-target input contract.
Production isolation is a separate `--selection production` checkpoint on an
already prepared testing-OFF Release tree. It binds the inspected final artifacts.
The legacy sealed-build adapters remain unsupported.

Checkpoint stages reuse original successful command records across campaign
names when their current declared inputs, configuration, tools and commands match.
Renaming a campaign cannot clear a failed gate. Broad discovery excludes the
repository prose defined in the [validation cadence](tests/README.md#validation-cadence),
so those documentation-only edits preserve reusable build and test evidence.
Explicitly consumed files, packaged documents and quality checks retain their
freshness requirements. Other declarations can still be broader than a repair's
actual consumers. Checkpoint preparation now omits only explicit registered runtime
test files outside its proven build/configuration closure, as defined in the
[validation cadence](tests/README.md#validation-cadence). Direct preparation inputs,
compiled/embedded/generated/copied/staged dependencies and artifact/owner checks
remain authoritative. Runtime changes still require fresh affected test execution;
quality and product declarations are unchanged. Admission, refresh and collection
share this scope while honoring every original receipt requirement. An older broad
or cold-discovery receipt may need one fresh preparation; never rewrite it.

A repeat reason authorizes retrying failed work. Matching successful checks remain
reusable. CTest retries retain the original process/JUnit evidence and execute only
cases that have not passed for the same inputs, tools, commands and artifacts.
Original failed invocations remain failed; composition requires complete passed-case
coverage and rejects altered receipts, outputs, skipped cases and incomplete owners.

Ordinary preparation, analysis, compiler probes and qualification have no elapsed
kill deadline. CTest registrations use `TIMEOUT 0`. Quiet logs and low CPU usage
alone do not establish a hang. Inspect suspected stalls and cancel through the owner;
owner-loss, failure handling and descendant drainage remain enforced. Explicit finite
timeouts remain available for intentional fault-injection controls and post-exit drainage.
General input-closure separation beyond this boundary is not yet implemented.
Opaque configure file operations, subprocesses and dynamic evaluation retain broad
input scope and require a real reconfigure when preparation dispatches;
variable-built input paths or file-operation subcommands cannot be declared
runtime-only from literal matching.
Do not bypass that admission limit. A diagnosed rerun uses `--repeat-reason`.
Checkpoint output prints exact membership and preserves outstanding failures.
Each relevance group runs its exact host and GPU memberships in separate
supervised CTest invocations under one build-tree lease and one common deadline.
Host work runs outside the shared device lease; GPU work holds it, including
cross-native exclusion. Each partition command must complete under supervision,
drain its owned work and admit its complete exact JUnit membership. Passing or
reuse requires the applicable results to succeed. Before another owner is
dispatched or a stable receipt is read, the enclosing owner must reach terminal
completion, drain all operation work and release the shared normal lease in a
healthy state once the outer operation ends. Apply existing pending/failure
admission rules; preserve failed status, while a drained failure may permit a
lawful repair or fresh preparation.
Retries and admission within the existing lease remain with that owner.
Partial console output and inner build success do not establish completion. Do not
signal a healthy supervisor except through its documented cancel path. Unchanged
group reuse checks all original command records and reports without requesting the
device lease.
`collect-qualification --receipts <paths...> --output <path>` requires product and
boundary evidence for both native Debug/Release profiles, tooling once per native
platform, corpus once, and both production Release gates. Host-only product checks
and A1/A2 development receipts cannot supply that aggregate. Full product, GPU,
Resolve and release acceptance remain separately required and are not established
by this dispatcher implementation.

The existing `execution.py` supervisor and build-tree lease cover preparation,
analysis and test consumption. A fresh preparation may recover a pending owner
only under a healthy build-tree lease and with intact proof that every command
from that attempt has drained. Recovery preserves the old pending marker and
command receipts, then runs normal fresh preparation; it does not turn the
interrupted attempt into a pass. Missing, damaged or legacy proof remains blocked,
as do hard owner death and blocked leases. Do not remove a lock because its
immediate parent exited.

On Windows, supervised CMake configuration and builds identify `VCTIP.EXE` from
the active `VCToolsInstallDir`. After a successful parent exit, the Job waits 30
seconds for natural drainage, then may terminate only that exact Job-owned
executable. Other children in ordinary unbounded commands continue until they
finish and drain; finite drain caps are limited to explicit fault controls.
Parent failure or interruption
keeps the ordinary failure and whole-Job drainage behavior. The process receipt
records the selected executable, policy and any termination.

Follow the [validation cadence](tests/README.md#validation-cadence) for the current
change: owning checks while editing, prescribed affected family checks at cutover,
and the complete matrix only at its designated integration/release checkpoint.
Exercise touched C11/native fixture/helper compilation and scoped quality on both
native platforms early, before broad qualification. Complete approved
numerical-policy adoption and exercise every affected expectation reader, caller
and GPU identity group before freezing for that checkpoint. Use one candidate
identity and one final packet linking original receipts; preserve owner-required
before/after source and tool hashes and finished receipts, retaining intermediate
records only for a concrete consumer. Each partition must have completed its
supervised command, drainage and exact JUnit admission. The enclosing outer owner
must reach terminal completion, drain its work and release the shared normal lease
in a healthy state before another owner is dispatched or a stable receipt is read.
Passing or reuse requires the applicable results to succeed. Preserve failure
status and apply existing pending/failure admission rules: a drained failure may
permit a lawful repair or fresh preparation. Retries/admission within the existing
lease remain with that owner. Do not
repeat passing groups solely for another receipt or create a per-intermediate proof
chain.
Reference captures and new ABI/owner OFF gates retain their transition
requirements. Collection remains subject to the current CLI's artifact and
pending-owner checks; a focused result is not an aggregate pass.

`--jobs` limits concurrent native and CUDA analysis commands; the default is 2
and `--jobs 1` provides serial execution. Formatting, toolchain/CUDA preparation
and Cargo checks remain sequential. Each analysis command retains its compilation
arguments, working directory and separate log. After a failure is observed, no
additional commands are scheduled; already running commands finish and their
results are observed before the failing batch returns. Keep GPU campaigns serialized on a shared device
and avoid concurrent Cargo operations using the same target directory.

The dispatcher first verifies that every selected translation unit exists in the
configured compilation database. After adding test targets, refresh that preset
before quality checks. Header selection reads each source's literal includes once
per invocation, then resolves them in each compilation entry's ordered include
search paths, with separate quote/angle lookup. All conditional branches are
considered conservatively; distinct CUDA variants are retained. Compiler response
files must be expanded before selection. The graph
is discarded after selection, so a subsequent edit cannot reuse stale dependencies.
The dispatcher checks formatting/configuration and edited native translation
units before Rust probes and the remaining header consumers. A failed producer
batch stops the later batches; a successful run retains every consumer and CUDA
variant. Each command writes a `.command.json` beside its log with arguments,
working directory, duration and exit status, including launch failures.

CUDA files use Clang's CUDA frontend with the preset's real NVCC compilation
database entries. Both `.cu` sources and consuming translation units for `.cuh`
or shared headers are checked; distinct CUDA build variants are retained. The
supported presets require CUDA 13.2, C++20, `sm_75`/`compute_75`, GCC 13 on Linux,
and the matching VS developer environment on Windows. Install the cuRAND
development headers too (`libcurand-dev-13-2` on Linux, `curand_dev_13.2` in the
Windows installer): Clang's runtime wrapper includes them even when a source
does not use cuRAND. No GPU is needed for this analysis.

The dispatcher preserves ordered definitions and include paths, host runtime
and exception settings, and the compilation working directory. Unknown NVCC
or host options fail with a diagnostic instead of being silently discarded.
Each `cuda-command-*.json` log records the original and translated arguments.
Clang 22 needs three narrow parsing adaptations: an empty removed CUDA texture
header, an early `_NV_RSQRT_SPECIFIER` definition, and OpenRAND's existing
host/device attributes enabled during both Clang passes. The OpenRAND analysis
copy is generated from the vendored header with only that guard changed; its
implementation and license are preserved. These files live under the quality
log directory and are used only by analysis. The repository's normal tidy
checks and warnings-as-errors remain enabled.

Changes to the runner or its tests also run the dispatcher regression tests
and representative host, CUDA, and Rust checks.

Outside the bounded development mode, selected Rust or qualification work runs
workspace rustfmt and three Clippy checks in the preset's matching profile: core,
plugin production and plugin `test-support`, each with all targets. Debug presets
use the development profile; Release presets use `--release`. The four-preset
matrix retains all 12 native platform/profile/package/feature combinations. Each
native CI lane keeps its Debug quality selection and explicitly checks Release Rust
with `--files Cargo.toml` after its Release build, including on documentation-only
changes. The runner selects the repository's Clippy configuration.
`Quality.RustNaming` also runs through CTest without a GPU; it compiles temporary
workspace copies using the actual manifests and policy, verifies individual
rejected identifiers by compiler diagnostic code and location, and exercises
valid domain names and a narrow foreign-boundary exception. It adds no product
dependency and leaves the production source and checked-in fixtures untouched.

`Quality.RustBoundaries` checks the current two-crate dependency contract using
Cargo metadata is parsed from stdout alone; raw stdout/stderr logs are retained
and bound to the quality receipt, including package-cache lock diagnostics.
Cargo metadata, including inactive target-specific declarations and resolved
source overrides. Direct dependency changes, new workspace members and crate
build/link hooks require an explicit change to the contract in the quality
runner and contextual review. This does not audit dependency internals.
Compiler probes also require valid profile consumers to compile and reject
escaping borrowed views, early owner release, private profile-storage access,
and local unsafe overrides in the core and safe asset modules. Probes run in
disposable workspace copies in both profiles and check diagnostic codes and
locations. Extend them when a new construction or safe orchestration boundary
lands; do not create speculative production APIs for tests.

CTest is the sole execution owner of `Quality.RustNaming` and
`Quality.RustBoundaries`; their standalone harnesses also remain available.
Each selected profile directly compiles every case and checks its original
positive-consumer or compile-fail diagnostics. Ordinary `check-quality.py` retains
formatting, Clippy and native/CUDA analysis without dispatching these groups.
Cargo artifacts reuse `out/build/<preset>/cargo/quality-probes/`; source workspaces
are disposable, while each run retains separate bounded command logs under
`out/validation/<preset>/quality/`. Probe workspaces must inherit the repository's
Cargo configuration. No probe manifests, publisher or runtime sealing are used.

Independent native CI product jobs run product and boundary selections in Debug
and Release. Analysis owns cumulative Debug quality and matching Release Rust
quality, with its own configured compile databases and generated inputs.
Tooling uses the standalone Debug CTest entry on each native platform; Linux Debug owns the single
corpus selection. The existing tooling-lint job remains responsible for maintained
Python, shell and workflow lint. Direct probe diagnostics are uploaded for both
profiles. The existing Linux/Windows required check names aggregate all four jobs;
every missing or non-success child keeps that aggregate non-passing.
Required migration-stage numerical, ABI, production and host gates keep
their named checkpoint requirements; ordinary local handoff does not rerun an
unchanged full matrix.

The dispatcher always runs the inexpensive native boundary guards across their
whole source scope, including new nested headers. They reject host messaging in
native/CUDA execution, host-context reset calls, and the retired native profile
parser/store and build entries. These are narrow source-text checks, including
comments, not a general architectural proof. Add a guard when a cutover completes,
with accepted and rejected examples; do not forbid a still-required bridge early.
Family tests must separately establish relevant/irrelevant identity changes,
conversion ownership, reuse and failure publication at the actual consumer.

CI runs these tests in the existing Linux and Windows host lanes. Repository
administrators must make those statuses required and configure independent review
to enforce them at merge; checked-in workflows cannot establish branch protection.
Changes to a guard, its scope, or its exceptions require review of the boundary
and a negative control, not just a green run of the modified check.

Before invoking compiler tools, the shared dispatcher checks whole selected
owned C/C++/CUDA files for trailing whitespace, merge conflict markers, retired
`JUICER_TESTS` and `JUICER_BUILD_VALIDATION` flags, `std::endl`, and line-leading
`using namespace` directives in headers. These are source-text checks, including
comments and literals, not a C++ parser. Generated files, fixture directories,
and paths outside the existing native quality scope are excluded. Failures report
file, line, and column and are saved in `source-hygiene.log` in the quality log
directory. Both CI lanes already invoke this dispatcher.
