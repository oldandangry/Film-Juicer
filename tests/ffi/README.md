# CUDA C ABI evidence

`Ffi.Host.CudaAbi` links the CMake-built Rust static library to real C11 and
C++20 consumers. It compares 570 size, alignment, offset, tag and flag facts,
including the opaque owner's pointer and the nullable abort callback. Both
native languages check the reviewed values in `cuda_abi_facts.inc`. Bindgen's
constant layout assertions also compile in Rust. All six operation signatures
and the callback signature are type-checked without referencing or defining the
new CUDA operations in a linked binary. The fixture contains no CUDA build or
render implementation.

`Rust.Bridge` additionally runs the plugin's internal ABI tests. The fixture's
one Rust export is gated by the nondefault `test-support` feature selected by
CMake only for `BUILD_TESTING=ON`. Product builds consume the committed bindings;
they do not require bindgen or libclang. Toggling `BUILD_TESTING` rebuilds the
archive with the corresponding feature selection.

Configure and build first, then run the bounded evidence (substitute any of the
four supported presets):

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug --target juicer JuicerCudaAbiTests
ctest --preset linux-debug -R '^(Ffi\.Host\.CudaAbi|Rust\.Bridge)$'
```

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
  --rustfmt "$HOME/.cargo/bin/rustfmt" --check
```

Windows PowerShell example:

```powershell
$llvm = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin'
python scripts/regenerate-rust-bindings.py `
  --bindgen "$env:USERPROFILE\.cargo\bin\bindgen.exe" `
  --clang "$llvm\clang.exe" --libclang "$llvm\libclang.dll" `
  --rustfmt "$env:USERPROFILE\.cargo\bin\rustfmt.exe" --check
```

When invoking PowerShell from WSL, first set
`$env:PATHEXT = '.COM;.EXE;.BAT;.CMD'`. The pinned Rust installation must be
selected in that environment; Windows tools do not qualify Linux generation.

Remove `--check` to update `rust/film-juicer-plugin/src/cuda/sys.rs`. Review the
header, generated diff and explicit C/Rust fact lists together. Never regenerate
layout expectations during ordinary tests. Generation uses each host's native
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

References: [bindgen release](https://github.com/rust-lang/rust-bindgen/releases/tag/v0.72.1),
[libclang requirements](https://rust-lang.github.io/rust-bindgen/requirements.html).
