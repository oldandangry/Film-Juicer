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
Use the shared verifier for native preparation, focused development checks and
qualification. Keep the existing preset checkout path and cache; configure and
build through the native owner. A `prepare` command configures and builds only its
selected target or file closure and does not run tests or create a qualification
result:

```sh
python -B scripts/verify.py prepare --preset linux-debug --targets JuicerFilmRecipeTests --testing ON
python -B scripts/verify.py prepare --preset linux-release --targets juicer --testing OFF
```

`development` selects files, tests or targets for a focused run. It can specify
`--runtime host`, `gpu` or `all`, and an optional `--run-id`:

```sh
python -B scripts/verify.py development --preset linux-debug \
  --files src/ScanRoute.cpp --runtime host --run-id route-change
```

`qualification` runs named CTest relevance groups on the selected preset. The
native owner builds the selected closure before tests, compares source inputs at
entry and exit, supervises the exact test functions/rows/operations, and publishes
a flat result only after descendant drainage. `--selection` accepts `product`,
`boundary`, `tooling` and `corpus`; `--runtime` accepts `host`, `gpu` or `all`.
`--build-dir` selects an already configured standalone tooling tree. `--label`
accepts a CTest regex and makes the result focused; it cannot establish a complete
selection.

```sh
python -B scripts/verify.py qualification --preset linux-debug \
  --reason 'Boundary change' --selection boundary --runtime host --run-id boundary-change
python -B scripts/verify.py qualification --preset linux-debug \
  --build-dir out/build/tooling-linux --reason 'Tooling checks' \
  --selection tooling --runtime host --run-id tooling-check
```

Only actual GPU work holds the shared physical-device lease. Results are written
to `out/validation/<preset>/results/<run-id>/` and can be inspected offline.
`collect-qualification --bundles <paths...> --output <path>` accepts optional
`--selection`, `--runtime` and `--presets`; it validates one complete fresh
checkpoint per requested preset and does not assemble focused results into a pass.

The supported tooling environment uses Python 3.13, Rust/Cargo 1.98.1,
rustfmt/Clippy from that toolchain, cargo-nextest 0.9.143, and the native compiler
and CUDA 13.2 toolchain for the selected platform. CMake/CTest remains the product
build/test orchestrator. With `BUILD_TESTING=ON`, CMake builds `Rust.Core` and
`Rust.Bridge` test binaries and nextest metadata through the configured target;
quality commands do not invoke nested `cargo test` builds. Compiler boundary
probes run from their existing CTest registrations.

Tests and Rust test metadata are excluded from the default product build;
the verifier builds their selected dependencies. Use `verify.py development
--preset <preset> --cuda-free --tests ...` for the CUDA-independent Rust,
tooling and native ABI/owner/value targets listed in tests/README.md. Its small
CMake configuration shares the preset's Cargo artifacts and build owner, while
product-linked host tests retain their CUDA compilation requirements. CUDA-free
results stay focused and cannot supply a full product checkpoint.

The shipping Rust archive always uses production features under the preset's
`cargo/production`; facade and fault consumers link the separate `cargo/test-support`
archive. Native objects remain shared. Ordinary test runs do not switch the
shipping feature set. Inspect both final link assignments and production symbols
with `Ffi.Host.BuildArtifacts`, and execute the production prepared-render probe
with its GPU registration. The complete native Release OFF closures and installed
production checks remain required at their designated qualification checkpoints.

The native quality entry is `scripts/check-quality.py`. For example:

```sh
python -B scripts/check-quality.py --base <accepted-parent> --preset linux-debug
python -B scripts/check-quality.py --files Cargo.toml --preset linux-release
```

Quality commands use the same native process owner and write fresh logs and
diagnostics below `out/validation/<preset>/results/quality-<id>/`. They run
formatting, Clippy and configured native/CUDA analysis without installed-tool
admission or reuse of older results. Preserve the full compiler database and
configured include variants. CUDA analysis uses Clang's CUDA frontend with CUDA
13.2, C++20, `sm_75`/`compute_75`, GCC 13 on Linux and the active Visual Studio
18 environment on Windows. The existing narrow CUDA parser adaptations and
warnings-as-errors remain enabled.

After review and acceptance, use `scripts/verify.py closeout --preset <preset>
--run <run-id> --accept --review closed` on that native platform. Explicit review,
failure and reproduction holds serialize with the preset owner. Unheld successful
optional diagnostics become eligible seven days after closeout; the next native
invocation or a plain `closeout --preset <preset>` visits at most 32 direct runs.
Required results, candidate identity and scientific artifacts remain. See
[result retention](tests/README.md#validation-cadence) for commands and failure policy.

`Quality.RustNaming` checks current crate naming policy using temporary copies of
the real manifests and reports rejected identifiers by diagnostic code/location.
`Quality.RustBoundaries` checks workspace dependencies, valid consumer compilation,
borrow expiry, private profile storage and unsafe-policy boundaries. Positive and
negative cases remain in their owning tests; do not add product APIs just for a
compiler fixture. Cargo metadata is read from the command output and source
workspace.

The shared tooling CMake entry registers 13 Debug tests: the three native result
groups; Dispatcher, ToolingLint, Execution, BuildEvidence, BuildRegistration,
Verify and Checkpoint; CTest Report and Repetition; and the Rust build contract.
The native result groups have 120-second deadlines; the other tooling tests use
600 seconds. Windows CI runs core, BuildEvidence and Checkpoint as separate
label-selected shards. Each shard must succeed; a focused label result does not
establish full tooling membership.

CI runs cumulative Debug native/CUDA/Rust analysis and matching Release Rust quality
inside their native product jobs, using their prepared dependencies and compiler
databases. Product/boundary host selections run in Debug and Release on each OS;
Linux Debug also runs corpus checks. Tooling and supporting-code lint stay
independent. Required host statuses demand success from both product/quality jobs
and tooling; no duplicate analysis preparation job or whole-cache upload is used.
The shared host result does not establish GPU acceptance, the Release testing-OFF artifact
contract, shipping-feature isolation, or installed Resolve behavior. No OFF probes
or full platform matrix are part of this workflow cutover. The contributor guide
does not change scientific fixtures, numerical expectations, external-reference
provenance, the Release OFF contract, or frozen R4/I8 meanings.

For each change, record the target, selected groups, expected membership and
relevant native/CUDA/Rust compilation contexts in the existing handoff. Run the
smallest owning check while editing, retain actual failures and diagnostics, and
reserve complete affected-family evidence for its named checkpoint. A focused
result is not full qualification. Documentation-only work needs whitespace,
link/reference review and contextual review; it does not require native builds or
product tests solely because prose changed.
