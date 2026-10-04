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

Rust is pinned by `rust-toolchain.toml` to 1.98.1 with its rustfmt and Clippy
components. Native formatting and tidy use LLVM 22; on Windows the qualified
tools are under
`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin`.
Configure the requested CMake preset, explicitly enabling tests before CTest:

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug
python scripts/check-quality.py --base <accepted-parent> --preset linux-debug --jobs 2
ctest --preset linux-debug --output-on-failure
```

Use the corresponding Windows or Release preset when that is the affected
target. `--files <paths...>` is the bounded package form; `--all-owned` is an
explicit audit. The dispatcher writes logs only under
`out/validation/<preset>/quality/`, leaves source unchanged, and treats missing
tools or compiler/lint failures as incomplete work. Formatting and lints do
not replace review of ownership, names, numerical parity, GPU lifetime, or
installed Resolve behavior.

During development, build the affected test targets and run the smallest owning
CTest selection that proves the changed behavior. Exercise new compiled tests
on both native platforms early, and check edited native sources before starting
broader qualification. At final qualification, run each applicable unfiltered
suite once for the stable candidate. Its JUnit report also supplies focused-domain
evidence; see [the test guide](tests/README.md). Separate reference
captures, transition checkpoints and production Release checks retain their
own requirements.

`--jobs` limits concurrent native and CUDA analysis commands; the default is 2
and `--jobs 1` provides serial execution. Formatting, toolchain/CUDA preparation
and Cargo checks remain sequential. Each analysis command retains its compilation
arguments, working directory and separate log; all command results are observed
before a failing batch returns. Keep GPU campaigns serialized on a shared device
and avoid concurrent Cargo operations using the same target directory.

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

Any selected Rust change runs rustfmt and Clippy over both complete crates, with
all targets in development and release profiles, followed by the Rust naming
enforcement tests. The runner selects the repository's Clippy configuration.
`Quality.RustNaming` also runs through CTest without a GPU; it compiles temporary
workspace copies using the actual manifests and policy, verifies individual
rejected identifiers by compiler diagnostic code and location, and exercises
valid domain names and a narrow foreign-boundary exception. It adds no product
dependency and leaves the production source and checked-in fixtures untouched.

`Quality.RustBoundaries` checks the current two-crate dependency contract using
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
