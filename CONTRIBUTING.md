# Contributing to Film-Juicer

Film-Juicer keeps photographic policy in the complete immutable Rust `RenderState`
and its named recipes. Native values, descriptors and CUDA code bind or execute
those decisions. Preserve arithmetic
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
- Prepare state outside publication locks and recheck the captured input before
  admission. Host calls must not hold publication/UI locks or an exclusive Rust
  instance borrow across possible reentry. Asynchronous work retains its inputs
  through their last consumer; finish, abort and retirement belong to the owner.
- Keep ordinary native implementation in its owning `.cpp` or `.cu`; headers
  contain declarations and implementation that needs consumer visibility, such
  as templates, constexpr and host/device code.

## Ownership and maintenance map

Film-Juicer has one production path: a Rust host/model and a retained native CUDA
engine. Production pixel rendering requires CUDA; there is no CPU fallback or
implemented Metal backend.

| Responsibility | Entry points and ownership |
| --- | --- |
| OFX discovery and callbacks | [`native/ofx_exports.cpp`](native/ofx_exports.cpp) defines the two standard discovery exports and delegates to [Rust OFX](rust/film-juicer-plugin/src/ofx/mod.rs). Rust owns descriptors, parameters, actions and callback-local image release. Retained OFXS headers do not implement a second production host adapter. |
| Process and instance | [Runtime](rust/film-juicer-plugin/src/runtime.rs) composes Assets and the native CUDA owner; Assets remains alive through native close. [Instance](rust/film-juicer-plugin/src/instance.rs), [submission](rust/film-juicer-plugin/src/submission.rs), [OFX instance handling](rust/film-juicer-plugin/src/ofx/instance.rs) and core [Publication](rust/film-juicer-core/src/state.rs) own capture, publication/admission and session policy. |
| Photographic model and assets | Core [Assets](rust/film-juicer-core/src/assets.rs), [profiles](rust/film-juicer-core/src/profile.rs) and [recipes](rust/film-juicer-core/src/recipe.rs), with the film, print, optics and scanner modules, own source/derived data and resolved policy. Complete immutable `RenderState` retains the backing its consumers need. The core [forbids unsafe](rust/film-juicer-core/src/lib.rs) and has no OFX/CUDA dependency. |
| Foreign submission and GPU execution | Plug-in [CUDA](rust/film-juicer-plugin/src/cuda/mod.rs) borrows admitted state through the [native C API](native/juicer_cuda_api.h). [RenderValues](src/RenderValues.h) binds resolved policy; [Root](src/ProcessRoot.h), [CUDA resources](src/Cuda/JuicerCudaResources.h) and kernels own execution and exact-context-plus-epoch residency. `PreparedCudaFrame` retains resources through finish/abort and asynchronous retirement. Native render code does not discover asset files. |
| Build and test boundaries | [CMake/Cargo integration](cmake/JuicerRust.cmake) builds the two-crate workspace. Shipping and production witnesses link `cargo/production`; facade/fault consumers select `cargo/test-support`. [ABI and binding maintenance](tests/ffi/README.md) documents committed bindings, regeneration and the [test-preparation C facade](tests/ffi/juicer_test_api.h). Independent native oracles and nonshipping measurement helpers retain their test consumers. |

For a maintenance change, follow the authoritative value to its actual consumer:

| Change | Trace |
| --- | --- |
| Grain mapping/defaults | [OFX controls](rust/film-juicer-plugin/src/ofx/controls.rs) → [grain policy/conversion](rust/film-juicer-core/src/film/grain.rs) → [recipe identity](rust/film-juicer-core/src/recipe.rs) → [frame binding](rust/film-juicer-plugin/src/cuda/frame.rs) → [native film payloads](src/Cuda/JuicerCudaFilmPayloads.cpp). |
| Scanner encoding | [Scanner policy](rust/film-juicer-core/src/scanner.rs) and recipe `scanner_params` → completed scanner/color identity → [native scanner binding](src/Scanner.h). Encoding changes must preserve unrelated spectral-medium reuse. |
| Profile-derived preparation | Assets/profile → recipe foundation and gamma-derived preparation → retained `RenderState` backing. Releasing cache holds must not invalidate admitted frames; original f64 exposure and interpolation f32 axes have different consumers. |
| Admission | Core `Publication::admit` takes pending inputs, prepares outside the publication lock, then rechecks before publishing the complete state. Follow the [plug-in consumer](rust/film-juicer-plugin/src/instance.rs) for capture failure and reentry. |
| Native workspace attachment | [`WorkspaceLayout::create`](src/Cuda/ResourceManager/JuicerCudaResourceCore.cpp) → Root frame preparation → exact-context leases and execution consumers → completion-aware release/alias reuse in [`ProcessRoot.cpp`](src/ProcessRoot.cpp). |

Preserve arithmetic order, encodings, identities/seeds and fixture-specific bounds.
The approved [density policy](tests/profile/fixtures/README.md) uses pinned
`libm::erfcf`; [filtered print](tests/print/illuminant-fixtures/README.md) uses
`libm::erff` with exact selected Rust expectations and resulting identities.
The [input-color contract](tests/ffi/README.md#input-color-preparation) retains
separate-f32 inverse authority. These are bounded numerical policies, not a
general tolerance or permission to replace portable math for speed.

The foreign trust boundary includes handwritten C++/CUDA, generated bindings,
standard/dependency libraries, toolchains, CUDA/driver behavior and host lifetime
contracts. Pointer/null/extent checks cannot prove pointer provenance or
consume-once correctness. Panic containment does not make every allocation
fallible. Native failure retention does not retain foreign images, streams,
contexts or unloaded module code; driver loss and forced unload have no universal
safety guarantee. Review the [ABI/lifetime contract](native/juicer_cuda_api.h)
and [fixture limitations](tests/ffi/README.md) when changing those edges. The
plug-in as a whole is not entirely memory-safe.

## Naming

Use the shortest name that preserves the distinction visible at the call site.
Use established photographic, OFX and CUDA vocabulary consistently; distinguish
film/print density, RGB/CMY order, linear/log exposure, resolved policy/borrowed
views, and enqueue/completion where the surrounding type or module does not.
Review declarations and representative consumers together, and keep shared
concepts searchable through their owning module or explicit boundary mapping.
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
generated, vendor, private and output trees. Ruff lints with `E4,E7,E9,F` and
checks formatting with `ruff format --check` (`ruff.toml`: 120 columns, LF);
ShellCheck resolves sourced scripts; actionlint uses the selected ShellCheck with
Pyflakes disabled. It neither modifies source nor installs tools, and missing/wrong
tools, lint or format failures or input changes fail the command. To apply Python
formatting, run the pinned `ruff format` from the repository root; `ruff.toml`
limits its discovery to the same maintained scope, so it never rewrites the
hash-checked Python resources under `Resources/`. The independent Linux lint job
requires no CUDA setup. Its unit tests also run as `Quality.ToolingLint` in CTest;
ordinary CTest does not require the lint executables. Private workbench checks
are separate and are never inputs to public CI. Lints complement contextual review.

Rust is pinned by `rust-toolchain.toml` to 1.98.1 with its rustfmt and Clippy
components. Native formatting and tidy use LLVM 22; on Windows the qualified
tools are under
`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin`.
`scripts/format_touched_files.sh [file ...]` applies that clang-format to changed
C++/CUDA files from Git Bash or WSL; `--check` verifies without modifying them.
Preserve `.clang-format`, pinned rustfmt, configured Clippy/tidy and compiler
warnings-as-errors; do not weaken checks or language standards to land a change.
Use the shared verifier for native preparation, focused development checks and
qualification. Keep the existing preset checkout path and cache; configure and
build through the native owner. A `prepare` command configures and builds only its
selected target or file closure and does not run tests or create a qualification
result:

```sh
python -B scripts/verify.py prepare --preset linux-debug --targets JuicerFilmRecipeTests --testing ON
python -B scripts/verify.py prepare --preset linux-release --clean-first \
  --production-contract --targets juicer JuicerProductionPreparedRenderProbe --testing OFF
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

The owner-authorized Step 6 workflow uses `--clean-first` within the standard
preset owner for Debug and the Release OFF closure; Release ON reuses the clean
production objects. Run product/boundary host selections in all four presets, tooling
once per native Debug OS, and corpus once in Linux Debug. Run both native Release
OFF production-contract commands before restoring either Release cache to ON.
The available RTX 4080 is single-GPU only: run the affected GPU registrations as
explicit focused checks. By owner decision on 5 November 2026,
`Ffi.Gpu.CudaInspect.MultiDevice` is optional hardware coverage outside the
single-GPU target's acceptance requirements. Preserve its unexecuted status;
it is not an acceptance blocker or deferred hardware-acquisition requirement.
The complete command set, membership and output limits are in
[the test guide](tests/README.md#owner-authorized-step-6-final-qualification).
These host and focused GPU results do not claim a full all-runtime checkpoint or
Resolve acceptance.

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

The shared tooling CMake entry registers 14 Debug tests: the three native result
groups; Dispatcher, ToolingLint, Execution, BuildEvidence, BuildRegistration,
Verify and Checkpoint; CTest Report and Repetition; the Rust build contract; and the offline OFX measurement report tests.
Ordinary host/GPU CTest registrations use explicit `TIMEOUT 0`, and the native
verifier dispatches CTest with `timeout=None`. Live execution rejects finite,
missing or malformed ordinary timeouts. Historical result readers still accept
valid finite positive GPU timeouts as well as zero, with the physical-device lock
and other evidence checks intact. Explicit bounded child fault controls and hosted
job limits remain. Windows CI runs core, BuildEvidence and Checkpoint as separate
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
