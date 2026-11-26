# Native ABI and binding maintenance

CMake links the module and native witnesses to the same `film_juicer_rust`
archive under `cargo/production`. The Rust-only workflow runs locked workspace
nextest and doctests without the CUDA toolchain.

`Ffi.Host.ProductionCudaAbi` checks 624 size, alignment, offset and value facts
from [the CUDA inventory](../tests/ffi/cuda_abi_facts.inc). The strict C11
consumer also checks the retained call signatures, DIR boundary ownership and
diffusion component admission. Locked nextest compares the generated Rust
types and values with the same frozen layout facts. The C11 OFX ABI case
checks vendored suite field types and signatures. The loaded module and OFX
host cases exercise discovery, variadic suite calls, reentry, and image leases.

The standard discovery exports in `native/ofx_exports.cpp` delegate to Rust.
Rust owns OFX callbacks and their image leases. Native Root owns CUDA
admission, frame resources and retirement. The current render continuation
and the DIR and diffusion C entries remain part of this ABI.

## Binding maintenance

The only regeneration entry point is `scripts/regenerate-rust-bindings.py`.
Install **bindgen-cli 0.72.1** separately on each native host:

```sh
cargo +1.98.1 install bindgen-cli --version 0.72.1 --locked
```

Use native x64 tools. Linux generation is qualified with Clang/libclang 22.1.8;
Windows uses VS 18's Clang/libclang 22.1.3. Both use the repository's Rust 1.98.1
rustfmt component, `rustfmt 1.9.0-stable (48a229ceae 2026-09-01)`. The entry point
checks each version and records the exact commands and output SHA-256.

Linux example (paths may be supplied explicitly for a local tool installation):

```sh
python3 scripts/regenerate-rust-bindings.py \
  --bindgen "$HOME/.cargo/bin/bindgen" \
  --clang /usr/bin/clang-22 \
  --libclang /usr/lib/llvm-22/lib/libclang-22.so.1 \
  --rustfmt "$(rustup which rustfmt)" --check --api cuda
```

Windows PowerShell example:

```powershell
$llvm = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin'
python scripts/regenerate-rust-bindings.py `
  --bindgen "$env:USERPROFILE\.cargo\bin\bindgen.exe" `
  --clang "$llvm\clang.exe" --libclang "$llvm\libclang.dll" `
  --rustfmt (rustup which rustfmt) --check --api cuda
```

When invoking PowerShell from WSL, first set
`$env:PATHEXT = '.COM;.EXE;.BAT;.CMD'`. The pinned Rust installation must be
selected in that environment; Windows tools do not qualify Linux generation.

The default `--api cuda` preserves the existing CUDA generation command.
Remove `--check` to update `rust/film-juicer-plugin/src/cuda/sys.rs`. Review the
header, generated diff and explicit C/Rust fact lists together. Never regenerate
layout expectations during ordinary tests. CUDA generation uses each host's native
target, C11 freestanding mode and only that Clang's resource headers; there is
no dependency on CUDA, OFX or a system SDK's definitions. The bindgen Rust syntax
target is 1.85 (its supported stable syntax baseline); actual compilation and
formatting use the pinned Rust 1.98.1 toolchain. Output uses native `usize` and
`isize` for address/count and signed-stride types and is identical on both hosts.

Bindgen 0.72.1 with Clang 22 emits warnings for unselected builtin type cursors.
They remain visible in generation logs. The emitted allowlist contains complete
named C fields and no opaque storage replacement except the intentional
incomplete `FjCuda` handle. C/C++/Rust field evidence and byte-for-byte native
regeneration qualify this output; these checks do not prove render lifetime.
The CUDA inventory contains 624 facts. It pins the 752-byte `FjFilmGrain`
layout, including the signed double `sample_origin` at byte 736.

Repeat the same native command with `--api ofx`. The wrapper includes
`ofxCore.h`, `ofxProperty.h`, `ofxParam.h`, `ofxImageEffect.h`, `ofxMessage.h`,
`ofxImageEffectExt.h` and `ofxParamExt.h` from
`third_party/ofxs/OpenFX-1.4/include/`; image-effect transitive headers are
read without importing their unrelated suites. The only vendored compatibility
patch selects `<cstdarg>` in C++ and `<stdarg.h>` in C in `ofxParamExt.h`.
All foreign declarations, macros and identifiers remain unchanged.

The exact, explicit allowlists are `OFX_TYPES` and `OFX_CONSTANTS` in
[`regenerate-rust-bindings.py`](../scripts/regenerate-rust-bindings.py).
The 14 named types and their referenced opaque handles implement the record
inventory above. The 160 constants cover current descriptor, dispatch,
property/parameter, image/render and message readers, including string-choice
and the four CUDA properties. `tests/ofx/descriptor_manifest.json` is an
expectation reader; it does not replace the vendored declarations. Generic
OFXS suite fetches alone do not justify additional capabilities.

OFX uses the same pinned bindgen/Clang/rustfmt versions and native targets
`x86_64-unknown-linux-gnu` / `x86_64-pc-windows-msvc`. Its flags add exact anchored
allowlists, `--clang-macro-fallback --default-macro-constant-type signed`
(for cast status macros), and the explicit vendored `-I` root to the common
`-x c -std=c11 -ffreestanding -nostdinc` and Clang resource include arguments.
Generation disables incidental include discovery and clears inherited bindgen
and C include overrides. Output logs contain wrapper/vendor hashes, full command,
allowlists, target and versions. It never regenerates CUDA as a side effect.

Native Linux Clang/libclang 22.1.8 and Windows 22.1.3 reproduce one common
`rust/film-juicer-plugin/src/ofx/sys.rs`, SHA-256
`a8c8446a08804cf2a992731970038ffaeb87fb625a9b36fea53dbd7ba0880bae`.
Reproduce on both native hosts before updating that identity or selecting target
files; a Linux cross-target run does not qualify Windows output. Product builds
consume committed declarations without a build script, bindgen or libclang.
Ordinary tests never regenerate bindings or independently frozen expectations.

References: [bindgen release](https://github.com/rust-lang/rust-bindgen/releases/tag/v0.72.1),
[libclang requirements](https://rust-lang.github.io/rust-bindgen/requirements.html).
