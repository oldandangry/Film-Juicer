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

## Native owner lifecycle

`Ffi.Host.CudaOwner` exercises the actual native owner with no GPU discovery:
invalid creation does not publish, duplicate creation cannot replace the
registered root, uninitialized teardown does not create a root, and scoped
close/reload constructs a fresh metadata owner. It also checks diagnostic
truncation and that callers cannot directly construct or destroy `Root`.

Its final case keeps a metadata-only registry submission active to make close
fail before any CUDA access. After ending the submission, repeated close and
destruction must leave the same blocked graph registered, without retrying
retirement or permitting a replacement owner. That failure-only graph is
intentionally retained until the test process exits.

`Ofx.Host.CudaOwnerUnload` links the real factory and OFX support entry point.
It checks successful factory close/reload and maps the same controlled native
shutdown failure to `kOfxStatErrFatal`, preserving consume-once retention. It
loads the factory directly without host suites; staged-module lifecycle probes
separately cover the complete host callback sequence. Neither test establishes
GPU completion or safe forced module unloading after a terminal failure.

Direct native fixtures use `JuicerCuda::Owner` for their process scope, just as
the OFX factory owns one handle across load/unload. Resource lookup remains a
borrow of that registered owner. The temporary C++ release bridge consumes its
handle once and deletes only after completed Root shutdown; failed shutdown
retains the graph and blocks a replacement owner. Final typed C shutdown/destroy
and failed-terminal-retention qualification remain a later migration boundary.
This test does not establish those terminal or GPU lifetime contracts.

The native execution object target has no OFX include or link dependency. Its
objects, current C++ host preparation, and OFX adapter objects enter one
`juicer_core` archive used by existing product and test links. Isolated test-hook
objects replace selected archive members without rebuilding CUDA in Cargo.

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

## Raw CUDA frame inspection

`fj_cuda_inspect` borrows the one registered owner and callback-local image
records. It never creates or switches a CUDA context. Success returns only the
physical device and exact current context; failure leaves that output unchanged.
The C header and Rust layouts are unchanged. Exceptions are contained at this
entry, and failed CUDA queries retain their API and numeric code.

Each float RGB/RGBA image has its own bounds origin and positive, float-aligned
row stride. Inspection checks the complete declared image extent against its
actual device allocation, using checked signed-size and address arithmetic.
Both images must cover the render window and belong to the current context.
Host and managed memory are unsupported. The source may be read across its full
bounds for metering. Destination writes cover the render window. Overlap between
those pixel ranges is rejected; padding is excluded, so disjoint images within
one allocation remain supported. No overlapping input/output algorithm has been
qualified: the output stages can read source pixels while writing destination
pixels, and distinct OFX image handles do not imply disjoint storage.

`Ffi.Host.CudaInspect` rejects malformed records without GPU discovery.
`Ffi.Gpu.CudaInspect` checks actual allocations, independent origins, exact
context identity, host-memory rejection, foreign contexts and row overlap.
`Ofx.Gpu.SequentialOwners` closes and recreates owners in the same live context,
checks increasing registry epochs, and compares all seven processor cases to
the unchanged platform captures, including the combined print/diffusion case. It also checks direct camera diffusion
on a constant exposure against the capture's immutable first-pixel sample (with a
no-diffusion control), checks distinct destination origins/pitches, and rejects
an uncovered destination before latch publication.
The instance latch has no epoch field. Native admission assigns it to the
transaction; execution reads the admitted snapshot and epoch from its prepared
frame. The existing diffusion mismatch rejection remains in force.

`Ofx.Gpu.StreamContract` dispatches actual render actions through the built
module and OFXS argument extraction. CUDA stream property presence is carried
separately from its value: supplied non-null and supplied null streams preserve
asynchronous submission; absence waits on the relevant default stream on normal
return and exception cleanup before image leases expire. Supplied streams receive
no new final wait or device-wide synchronization. The small local OFXS presence
patch is removed with the support library's compiled consumer at S6.

## Native render entry

`Ffi.Host.NativeBoundary` checks the native message/OFX exclusion, direct C++
production routing, the implemented operation set, and absence of live-context
reset calls. These structural checks supplement the runtime evidence below.

`Ffi.Gpu.PreparedBoundary.*` now calls the real `fj_cuda_render` from C11,
compares it with the direct C++ executor and processor adapter, and retains the
independent immutable platform captures as behavioral authority. Production
continues through the temporary direct C++ adapter during S2.D.

`Ffi.Gpu.CudaRender.negative-direct` and `.negative-print` additionally exercise
raw argument and prepared/submission binding failures, current-context and stream
ownership, zero/one/truncated error buffers, typed API/code preservation, and
exception containment. Isolated API/executor objects inject native exceptions;
no hook enters product objects. Abort queries run synchronously at three bounded
checkpoints and expire at return. Tests cover null queries, undefined return tags,
same-thread invocation, rejected reentry, cancellation followed by success, and
native epoch advancement after idle resource retirement without resetting the
host context. A finite piece of queued GPU work proves cancellation after enqueue:
the accepted abort path must either complete that work or retain its use fence.
The abort callback itself enqueues no work and changes no ownership. Existing
prepared-frame and pinned-upload tests independently cover uncertain retention
and host-span expiry. No injected error establishes actual driver-loss acceptance.

The private `NativeCall` borrow owns the one registered runtime's call gate from
inspection through execution. `fj_cuda_render` re-inspects under this admission;
a previously returned context record is not a retained capability. Supplied
non-null and explicitly null streams keep the existing asynchronous contract;
absence completes the relevant stream before native return, including unwind.
`Ofx.Gpu.ExecutorFailureOrder` checks native classification and abort before
adapter message delivery, actual gate reacquisition by the message fixture,
percent escaping, the existing message identifier, and recovery/latch/fatal order
independent of message-delivery failure. Native execution has no message callback.
