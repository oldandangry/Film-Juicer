# CUDA C ABI evidence

`Ffi.Host.CudaAbi` links the CMake-built Rust static library to real C11 and
C++20 consumers. It compares 575 size, alignment, offset, tag and flag facts,
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

Its final case keeps a metadata-only registry submission active to make borrowed
shutdown fail before CUDA access. After ending that submission, consuming close
and the wrapper destructor must leave the same blocked graph registered, without
retrying retirement or permitting a replacement owner.

`Ffi.Host.CudaTerminal.*` tests the three terminal C entries through isolated API
objects. It covers closed/repeated shutdown, accepting destroy, exception
containment, error capacities zero/one/truncated, saved typed status independent
of diagnostic wording, and allocation-free blocked destroy. A scoped diagnostic allocation failure preserves
the selected cuFFT status and uses the owner's fixed diagnostic fallback. Executable-local
allocation counting and destruction/attempt counters prove consume-once without
calling a stale address after successful deletion. The serialization case holds
an outer preparation guard while shutdown closes admission; the callback can
still enter the gate to reject and release its guard before teardown proceeds.
No terminal injector enters production objects.

The C boundary has these ownership transitions:

| Operation | Ownership after return |
| --- | --- |
| `fj_cuda_shutdown` succeeds | Caller owns a closed handle; repeated shutdown succeeds without another drain. |
| `fj_cuda_shutdown` fails | Caller owns a blocked handle and the retained native graph; repeated shutdown returns the saved failure. |
| `fj_cuda_destroy` receives accepting ownership | Caller has taken/cleared its pointer; exactly one shutdown attempt leads to deletion or failed native retention. |
| `fj_cuda_destroy` receives closed ownership | The closed graph is deleted once and a later create can establish a new owner. |
| `fj_cuda_destroy` receives blocked ownership | Caller ownership is consumed, the graph remains registered and retained, and no retry/destructor traversal occurs. |

`JuicerCuda::Owner::close()` takes its pointer before invoking consuming destroy.
Its destructor uses that same path only if explicit close has not consumed it.
Closed or blocked owners reject inspect, render and instance retirement.
Shutdown stops Root admission and drains outer preparation guards before taking
the native-call gate; it never waits on those guards while holding that gate.
`Root` no longer has a separate public shutdown or bootstrap-reopen path.

`Ofx.Host.CudaOwnerUnload` links the real factory and OFX support entry point.
It checks empty unload, successful close/reload, and fatal OFX mapping for failed
native shutdown while preserving consume-once retention. Failed native graphs
remain intentionally allocated until process exit. Retention does not keep host
images, streams, contexts, suites or module code alive and does not make forced
unload safe.

`fj_cuda_retire_instance` rejects zero and otherwise removes only that token's
static-grain membership across exact-context owners. `Grain.Gpu.ScratchReuse`
checks two instances and two contexts, preserving the other instance, frame
owner, epoch and ledger. Final foreign-context grain release enters the existing
deferred queue; explicit Root retirement drains that queue when the exact owner
context is current. Low-level resource destruction never calls Root retirement.
The reset cases also destroy a context owned exclusively by the fixture, then
retire its Root and deferred ownership while preserving the primary context's
resources and charges. They never reset a live host CUDA context.

`ScatterHalation.Gpu.TerminalRetention` creates actual prepared GPU resources,
injects the existing context-drain failure, and checks that borrowed shutdown and
consuming destroy retain the same allocations and ledger charges. This is a
controlled failure fixture, not real driver-loss or forced-unload qualification.
Ordinary prepared-frame/lifecycle and native failure/cancellation fixtures use
the same typed terminal operations for final cleanup.

Direct native fixtures use `JuicerCuda::Owner` for their process scope, just as
the OFX factory owns one handle across load/unload. Root lookup only borrows that
registered owner. Both production routes now use callback-local C-record
projection and the admitted body shared with `fj_cuda_render`.

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

Terminal regressions also cover unregistered handles during ordinary/reentrant
calls and owner-mutex admission exceptions, selected driver failures surviving
later shutdown/destroy admission exceptions, and a concurrent borrowed shutdown
whose result is still being selected. The isolated API object injects immediately
before acquiring the owner mutex; it exercises the real validation/catch order.
The host must still exclude consuming destruction while a handle is borrowed;
these tests never reuse a successfully deleted handle.

A malformed borrowed shutdown error buffer (nonzero capacity, null data) is
rejected before admission changes. Consuming destroy still consumes a valid
owner with malformed output: a saved/native terminal failure has precedence;
a successful close/delete reports unsupported output afterward. Neither malformed
output nor a later empty wrapper destructor retries consumption. Valid zero,
one-byte and truncated diagnostics preserve status and the saved diagnostic.

## Production projection and render outcome

The temporary OFX adapter holds one `NativeCall` across inspection, the short
instance submission latch, and `project_and_render`. The latter borrows the
immutable recipe/payload and current descriptors, builds `FjPreparedHostData`
locally, and immediately invokes `NativeCall::render`. No complete prepared
record escapes. The exported C entry acquires its own admission and invokes the
same render body, validation, decoder and executor. Reentry remains rejected;
there is no runtime selector or adapter-to-executor alternative.

`FjRenderOutcome` contains the existing 12-byte `FjStatus` followed by a 32-bit
flags field (size 16, alignment 4). Only `FJ_RENDER_DEFERRED_DIR_ERROR` is
assigned. The native scan readback's DIR failure bit supplies that provenance;
ordinary scanner RGB, polling and initialization failures do not set it. The
flag survives a zero-capacity C error buffer; all other bits remain zero. The
adapter selects its percent-safe DIR message solely from that flag, only after
releasing admission. Diagnostic wording never selects message delivery.
It retains the full native diagnostic string. The C export alone applies the
caller buffer's truncation contract. Pending context-loss detail retains the
first native failure and stage; Root recovery happens after admission release.
Native code alone owns absent-stream completion and prepared finish/abort.

`Ofx.Gpu.CutoverContract` exercises actual processor callbacks with changed
geometry/time/scale, sequential streams, another host thread, a fixture-owned
second context on the same device, same-allocation disjoint image regions, and
abort before/after preparation, including a throwing abort suite. Its constant
input geometry check disables glare and supplements the unchanged independent
processor pixel fixtures. `Ofx.Gpu.ExecutorFailureOrder` includes diagnostics
longer than 8 KiB, empty messages, and DIR messages without `component=dir`.
Ordinary scan-stage and grain failures containing `component=dir` must not
trigger a host message. `Scanner.Gpu.OutputEncoding` covers real readback bits
for scanner-only, DIR-only and combined failures on all four routes.

The render-contract fixture checks the new flag independently of status and
text capacity and checks shared admission with real reentry rejection. Its
expiry case borrows copied film-density storage through C, then mutates and
releases it after return while a native staging event is still pending. The
isolated post-executor seam submits that same span through the real pinned
uploader behind finite device copies. The fixture then verifies its exact bytes.
It inserts no blocking GPU gate into rendering; staging warmup completes before
the measured call. This isolates the C return/staging lifetime from legitimate
synchronous renderer paths. Existing pinned-upload tests independently cover
reservation, exceptional completion and quarantine. The uploader hook object
remains confined to the test executable.

## Native asset lookup safety

`Assets.Host.LookupSafety` runs the real private synthetic density, Scanner
interpolation/density and Hanatos window helpers through dedicated test objects.
`JUICER_ASSET_LOOKUP_TEST_HOOK` is defined only on those objects and their test
executable. Product objects and normal Release modules contain no wrappers.
The existing admission and context-drain seams are independent and unchanged.

The group checks finite endpoints/interpolation, distinct duplicate selection,
ascending/descending Scanner traversal, singleton axes, ordered infinities,
NaN/shape/channel failures, and public direct/print/synthetic builder failure
propagation before reference or descriptor publication. It performs no CUDA
runtime or driver operation. CUDA linking/build prerequisites remain required.
[Independent expectations and provenance](asset_lookup_expectations.md) explain
why synthetic invalid brackets are unreachable for admitted axes and define the
three Hanatos formula cases without changing production parameter admission.
Relaxed profile-to-recipe acceptance belongs to the later admission removal.

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug --target JuicerAssetLookupTests
ctest --preset linux-debug -R '^Assets\.Host\.LookupSafety$'
```
