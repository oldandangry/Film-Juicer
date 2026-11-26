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
| OFX discovery and callbacks | [`native/ofx_exports.cpp`](native/ofx_exports.cpp) defines the two standard discovery exports and delegates to [Rust OFX](rust/film-juicer-plugin/src/ofx/mod.rs). Rust owns descriptors, parameters, actions and callback-local image release. The vendored OpenFX C SDK headers supply the ABI declarations. Historical OFXS Support files are documented in the [vendor provenance](third_party/ofxs/PROVENANCE.json). |
| Process and instance | [Runtime](rust/film-juicer-plugin/src/runtime.rs) composes Assets and the native CUDA owner; Assets remains alive through native close. [Instance](rust/film-juicer-plugin/src/instance.rs), [submission](rust/film-juicer-plugin/src/submission.rs), [OFX instance handling](rust/film-juicer-plugin/src/ofx/instance.rs) and core [Publication](rust/film-juicer-core/src/state.rs) own capture, publication/admission and session policy. |
| Photographic model and assets | Core [Assets](rust/film-juicer-core/src/assets.rs), [profiles](rust/film-juicer-core/src/profile.rs) and [recipes](rust/film-juicer-core/src/recipe.rs), with the film, print, optics and scanner modules, own source/derived data and resolved policy. Complete immutable `RenderState` retains the backing its consumers need. The core [forbids unsafe](rust/film-juicer-core/src/lib.rs) and has no OFX/CUDA dependency. |
| Foreign submission and GPU execution | Plug-in [CUDA](rust/film-juicer-plugin/src/cuda/mod.rs) borrows admitted state through the [native C API](native/juicer_cuda_api.h). [RenderValues](src/RenderValues.h) binds resolved policy; [Root](src/ProcessRoot.h), [CUDA resources](src/Cuda/JuicerCudaResources.h) and kernels own execution and exact-context-plus-epoch residency. `PreparedCudaFrame` retains resources through finish/abort and asynchronous retirement. Native render code does not discover asset files. |
| Build and test boundaries | [CMake/Cargo integration](cmake/JuicerRust.cmake) builds the two-crate workspace. Shipping and linked native tests use `cargo/production`. [ABI maintenance](docs/ffi.md) covers the committed bindings, C11 inventories and regeneration. |

For a maintenance change, follow the authoritative value to its actual consumer:

| Change | Trace |
| --- | --- |
| Grain mapping/defaults | [OFX controls](rust/film-juicer-plugin/src/ofx/controls.rs) â†’ [grain policy/conversion](rust/film-juicer-core/src/film/grain.rs) â†’ [recipe identity](rust/film-juicer-core/src/recipe.rs) â†’ [frame binding](rust/film-juicer-plugin/src/cuda/frame.rs) â†’ [native film payloads](src/Cuda/JuicerCudaFilmPayloads.cpp). |
| Scanner encoding | [Scanner policy](rust/film-juicer-core/src/scanner.rs) and recipe `scanner_params` â†’ completed scanner/color identity â†’ [native scanner binding](src/Scanner.h). Encoding changes must preserve unrelated spectral-medium reuse. |
| Profile-derived preparation | Assets/profile â†’ recipe foundation and gamma-derived preparation â†’ retained `RenderState` backing. Releasing cache holds must not invalidate admitted frames; original f64 exposure and interpolation f32 axes have different consumers. |
| Admission | Core `Publication::admit` takes pending inputs, prepares outside the publication lock, then rechecks before publishing the complete state. Follow the [plug-in consumer](rust/film-juicer-plugin/src/instance.rs) for capture failure and reentry. |
| Native workspace attachment | [`WorkspaceLayout::create`](src/Cuda/ResourceManager/JuicerCudaResourceCore.cpp) â†’ Root frame preparation â†’ exact-context leases and execution consumers â†’ completion-aware release/alias reuse in [`ProcessRoot.cpp`](src/ProcessRoot.cpp). |

Preserve arithmetic order, encodings, identities/seeds and fixture-specific bounds.
The approved [density policy](tests/profile/fixtures/README.md) uses pinned
`libm::erfcf`; [filtered print](tests/print/illuminant-fixtures/README.md) uses
`libm::erff` with exact selected Rust expectations and resulting identities.
The [tagged input-color contract](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/tests/ffi/README.md#input-color-preparation) retains
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
were found. Do not invent findings to demonstrate coverage.

## Required checks

### Cleanup workflow on testing

The product baseline for cleanup is `archive/spectral-cutover-2026-11`.
Each cleanup PR is one commit. Run its checks on that commit and land it on
`testing` before starting the next PR.

Use CMake presets for the native build, shipping Rust archive and staged
bundle. Run native cases with CTest and Rust cases directly with locked
workspace nextest and Cargo doctests in Debug and Release. Run Rust tests
serially until their shared resources support parallel runs. The default
testing-enabled build supplies the registered binaries, modules and resources.
CTest enforces a 300-second per-test timeout. See the executable commands in
the [build](docs/building.md) and [test](tests/README.md) guides.

Use standard runner results for cleanup. A failed case or an empty requested
selection fails its job. Migration qualification groups, command/row receipts,
offline collection and receipt closeout apply to the tagged baseline. The
old `verify.py` and `check-quality.py` entry points and ordinary `TIMEOUT 0`
rule are superseded for cleanup. The tag preserves the [migration commands
and their original scope](https://github.com/oldandangry/Film-Juicer/blob/archive/spectral-cutover-2026-11/CONTRIBUTING.md#required-checks).

### Selecting checks

- While editing, exercise the changed behavior and its nearest consumer. At
  cutover, run the affected host, Rust and GPU checks on Windows and WSL in
  the applicable profiles. Run GPU checks on the shared device one OS at a
  time. `host` needs no GPU at runtime. `gpu` requires one. Multi-device
  coverage is optional on the single-GPU system.
- For ABI, ownership or package changes, check both native Release packages
  with testing off and the applicable testing-on consumers. Compare bindings,
  linked shipping inputs, exports, resources, lifetimes and typed failures at
  the boundary that changed. Test features must stay isolated from shipping.
- Preserve pinned rustfmt, Clippy, clang-format, clang-tidy, Ruff and compiler
  warnings-as-errors. Analyze native headers through their real translation
  units and configurations. Review changes to binding inputs against generated
  output on both platforms. See [binding maintenance](docs/ffi.md#binding-maintenance).
- Keep scientific expectations independent and checked-in fixtures immutable
  during tests. Report actual selections, outcomes and material limits.
  Automated results cover their exercised behavior. Resolve installation,
  visual acceptance and live hardware evidence retain their own scopes.
- For documentation-only changes, check whitespace, links, referenced
  commands and meaning in context. Executable changes also need their owning
  checks.
