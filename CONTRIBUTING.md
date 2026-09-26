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
- Validate external input at its construction or admission boundary. Complete
  immutable profiles, recipes, and descriptors are trusted by downstream
  consumers until a real external, numerical, or lifetime boundary intervenes.
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
python scripts/check-quality.py --base <accepted-parent> --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

Use the corresponding Windows or Release preset when that is the affected
target. `--files <paths...>` is the bounded package form; `--all-owned` is an
explicit audit. The dispatcher writes logs only under
`out/validation/<preset>/quality/`, leaves source unchanged, and treats missing
tools or compiler/lint failures as incomplete work. Formatting and lints do
not replace review of ownership, names, numerical parity, GPU lifetime, or
installed Resolve behavior.

Any selected Rust change runs rustfmt and Clippy over both complete crates, with
all targets in development and release profiles, followed by the Rust naming
enforcement tests. The runner selects the repository's Clippy configuration.
`Quality.RustNaming` also runs through CTest without a GPU; it compiles temporary
workspace copies using the actual manifests and policy, verifies individual
rejected identifiers by compiler diagnostic code and location, and exercises
valid domain names and a narrow foreign-boundary exception. It adds no product
dependency and leaves the production source and checked-in fixtures untouched.

Before invoking compiler tools, the shared dispatcher checks whole selected
owned C/C++/CUDA files for trailing whitespace, merge conflict markers, retired
`JUICER_TESTS` and `JUICER_BUILD_VALIDATION` flags, `std::endl`, and line-leading
`using namespace` directives in headers. These are source-text checks, including
comments and literals, not a C++ parser. Generated files, fixture directories,
and paths outside the existing native quality scope are excluded. Failures report
file, line, and column and are saved in `source-hygiene.log` in the quality log
directory. Both CI lanes already invoke this dispatcher.
