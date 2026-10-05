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
Test exports are gated by the nondefault `test-support` feature selected by
CMake only for `BUILD_TESTING=ON`. Product builds consume the committed bindings;
they do not require bindgen or libclang. Toggling `BUILD_TESTING` rebuilds the
archive with the corresponding feature selection.

## Profile and resource terminology

The safe Rust API calls metadata-selected DIR, halation and reconstruction
parameters `FilmProcessingDefaults`, exposed by `FilmProfile::processing_defaults()`.
The fixed C ABI retains `FjFilmDigest` and `digest`; the explicit mapping lives
in `asset_bridge.rs`. Fixture keys and retained native spellings also stay fixed.
These defaults feed recipe controls; they are not identity values.

`ProfileTables::authored_log_exposure()` retains the original f64 sequence for
density-model sampling (including print-gamma resampling) and profile identity.
`interpolation_log_exposure()` exposes its admitted f32 density-lookup axis.
Admission requires a nonempty, NaN-free, nondecreasing narrowed axis; equal values
and infinities are allowed. The authored sequence need not be ordered when
narrowing hides the distinction. The production C ABI calls the interpolation
span `log_exposure`; the fixture ABI additionally calls the authored span
`source_log_exposure`. The source-only `ProfileSamples::log_exposure()` and JSON
`data.log_exposure` keep their single authored meaning.

`CmfRows` and `load_cmf_csv` describe decoded source rows
`[wavelength_nm,x_bar,y_bar,z_bar]`, before native axis/curve preparation.
`profile::Role` selects film/print use in the catalog and loader;
`profile::Support` is the authored photographic substrate, film or paper;
`profile::Stage` is the authored filming/printing step. Printing can use either
substrate, so these axes and their schema/ABI mappings remain distinct.

Identity terms have separate scopes and encodings:

| Rust term | Contributing inputs and zero rule | Boundary/consumer |
| --- | --- | --- |
| `hash::{bytes,u64s,finite_f32s,resource_f32s,f32s_with_nan_mask}` / `FloatSpanHash` | FNV-1a primitives with each function's byte, signed-zero and NaN rules; no final zero remapping. `FloatSpanHash` has separate value and NaN-mask streams. | Raw encoding building blocks, not asset identities by themselves. |
| `SpectraLut::asset_hash()` | Decoded reconstruction sample bits; signed zeros and NaNs canonicalized in one stream, final zero mapped to one. No path, metadata or evaluator version. | `FjSpectraLutView.asset_hash`, native reconstruction reuse. |
| `FilmProfile::asset_token()` / `PrintProfile::asset_token()` | Stock and consumed metadata, tagged sampled tables/adaptation presence, authored exposure/model bits and density evaluator version; final zero mapped to one. Display name, file path and JSON formatting do not contribute. | Profile views' `asset_token`, native `assetVersionToken` and downstream recipe identities. |
| `profile_token` / `finish_asset_token` | Private producer and zero finalizer for that same completed-profile token. | No second identity contract or process-handle allocation. |
| `PrintDensityCurves::hash()` | Count, raw narrowed-axis bits and gamma-adjusted CMY total bits, in order; zero is rejected, never remapped. | `FjPrintDensityView.hash`, native print development. |

These content identities are deterministic across process runs given the same
encoded inputs/evaluator outputs. They are not cryptographic digests or
interchangeable across domains, and do not promise equal numerical outputs on
different platforms. Equal profile tokens do not imply equal gamma-adjusted
curve hashes, and a raw sample hash cannot substitute for a profile token.

Generated CUDA bindings additionally transport native-owned recipe, descriptor,
table and submission hashes unchanged. Each enclosing ABI record establishes
the component and its native producer/zero contract. Their `clip_token`,
`instance_token` and `frame_token` instead identify clip/session/frame lifetime
or temporal facts; they are not resource content fingerprints. Fixed ABI field
names remain governed by `native/juicer_cuda_api.h`.

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

`Ffi.Gpu.PreparedBoundary.*` calls the real `fj_cuda_render` from C11 and checks
bit-exact equivalence between cold/warm C-boundary execution, the direct C++
executor and the processor adapter. Independent numerical expectations are checked
separately by `Ofx.Gpu.ProcessorReference` and `Ofx.Gpu.AcceptedCudaCaptures`;
a baseline-image failure cannot prevent these path-equivalence checks from running.
Production callbacks use the same admitted native C-boundary render body.

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
three Hanatos formula cases The production-profile consumer group now also reaches those three formula cases
with adaptation enabled through selected-profile recipe construction.

```sh
cmake --preset linux-debug -DBUILD_TESTING=ON
cmake --build --preset linux-debug --target JuicerAssetLookupTests
ctest --preset linux-debug -R '^Assets\.Host\.LookupSafety$'
```

`Ffi.Host.ProfileOwner` links the feature-only profile facade declared in
`juicer_test_api.h` to C11/C++20 consumers. It checks 24 layout/tag facts and four
signatures against Rust, exact exposure/CMY/layer transport using analytical
center/endpoint expectations, metadata/default tags, nullable spectra, failure
outputs, diagnostic bounds/uninitialized output, concurrent immutable reads,
release pairing and independent-copy lifetime.
`Rust.Bridge` compares projection pointers/bits and tokens directly with the
complete core owner, tests retention after explicit cache release, and uses Weak
for reclamation on release (including malformed diagnostics), unpublished
input/capacity failure and contained panic. No test dereferences an expired pointer.

Each acquisition uses the real Assets catalog and complete film producer; the
opaque owner retains its Arc after the temporary Assets is destroyed. The view
exposes only fields consumed by the current profile-row qualification and the
actual asset token. Other optional/model/digest data stays in the full immutable
owner. There is no profile-array conversion or duplicate decoder. The fixed view is 264 bytes on x64, including 24 copied halation-digest bytes.
It duplicates zero spectral/exposure/layer table bytes. The local
native lifetime test copies 80 payload bytes into its own vectors; those are test scratch,
not a production cache. Future production bridges share `asset_profile` projection
and obey the same lifetime; native invocation expiry still applies.

The host fixture creates isolated resources beneath the preset's
`out/validation/<preset>/ffi/profile-owner-resources`, including both required
catalog roles/default keys. It never writes Resources or checked-in fixtures,
links no CUDA runtime and performs no driver operation. Detailed pointer,
concurrency, status, diagnostic and consume-on-every-outcome rules are beside
its declarations in `juicer_test_api.h`. Incidental small Box/Arc/path/parser
allocation aborts remain outside panic recovery. Normal `BUILD_TESTING=OFF`
does not compile this fixture facade. Safe production owner/projection modules
remain available; the non-default `test-support` feature enables the fixture interface.

After configuring/building with tests explicitly ON, run the new boundary and
its Rust owner checks with `ctest --preset <preset> -R
'^(Ffi.Host.ProfileOwner|Rust.Bridge)$' --output-on-failure`. The fixture record is named `FjFilmFixtureView`; its 264-byte layout and
numerical oracle remain independent of the production profile records.

## Production catalog bridge

`Ffi.Host.CatalogOwner` checks the private production catalog ABI in C11,
C++20 and Rust, the retained-owner lifetime, failure-output clearing,
concurrent acquisition, source-error stickiness and native conversion retry.
It calls the actual native option accessors and selected film/print loaders.
`FjCatalogEntryView` contains key, label and polarity only
(40 bytes, alignment 8 on both supported x64 targets). Counts are role-local
entry counts; indices are zero-based. Text is length-delimited UTF-8, including
empty labels and embedded NUL bytes. Existing OFX C-string behavior remains.

`FjPathView` counts Linux native bytes or Windows uint16_t units, excluding a
terminator. Only the current platform's encoding is accepted; nonempty/NUL-free
input and a checked byte extent bounded by PTRDIFF_MAX are required. No Unicode
admission or diagnostic-text round trip selects files. Windows forwarding copies
wchar_t units by value. Selected-profile paths stay private to Rust Assets; the actual Rust loader opens
them under the retained native root. Unrepresentable path units are
escaped for diagnostics only. The Linux unusual-name opening fixture uses the
native temporary filesystem because WSL's Windows-mounted volume may replace
invalid UTF-8 bytes; it removes those fixture files on exit. Windows uses its
configured artifact directory. Other resource families still use their existing
root-string conversion until their own cutovers.

Assets create is lazy and performs no discovery or CUDA initialization. One
Assets belongs to the process Library. Catalog acquisition retains an immutable
Arc; independent handles survive Assets destruction and each other's release.
Reads allocate nothing. Catalog paths do not cross this boundary.
Caller excludes Assets destruction from acquisitions, and catalog release from
all reads and outstanding view uses. Outputs and diagnostics are aligned,
exclusive and disjoint. Every valid output is cleared before validation.
Malformed diagnostics skip ordinary work, but destroy/release consume their
owner once on every result. Truncation changes text only. Native `call_once`
publishes a complete conversion or complete ordinary source failure. Conversion
allocation/internal failure leaves it incomplete; subsequent explicit requests
retry against the unchanged Rust snapshot without a second parser or reload.

Root owns the whole Library through a detachable unique_ptr. Legal terminal
close requires the host to exclude **all** option/catalog/profile/rebuild readers,
render/preparation calls and outstanding views. A drained preparation count or
failed GPU drain does not establish that condition. Registered reentry retains
active host borrows and its blocked graph; it is not legal reader-excluded
terminal completion. Registration is checked before dereference. Under legal
terminal exclusion, Library detachment is allocation-free and precedes throwing
owner admission. It stays alive through the single native close attempt, then
Rust owners and native host copies are consumed outside every native lock.
Native failure keeps precedence over host/diagnostic cleanup failure; cleanup
never retries uncertain CUDA teardown. The retained native graph has no Library,
Rust owner or borrowed Rust/OFX storage, and asset access fails after detachment.
Successful or failed borrowed shutdown keeps Library attached; successful cache
release runs after native locks unwind and retains catalog ownership.
Construction and rejected-candidate destruction likewise run outside registration
locks. Terminal construction/reentry/lock-failure tests use isolated API objects;
catalog conversion seams and Rust owner/fault probes are test-only.

The catalog path transport/Windows encoding/native selected-profile opening
bridge has been removed. Native root-path transport remains an actual consumer. Asset conversion is removed in S4.E; native host Library ownership
is removed in S5.C. These are distinct boundaries. The fallible cache-clear export invokes the existing Rust cache-release operation.
Later-family production acquisition remains outside this boundary. Installed Resolve acceptance remains
separate from these automated ABI, host and GPU checks.

## Production profile and gamma bridge

`Ffi.Host.ProductionProfileOwner` calls the actual film/print acquisition, borrowed
view, gamma sampling and consume-once release APIs in `juicer_legacy_api.h`.
C11, C++20 and Rust check every production field, tag and signature. All 28
profiles and the accepted gamma cohort are compared with immutable public
captures. Owned gamma totals survive print release, and retained profiles survive
Assets release/destruction. Safe projections share the core Arc and borrow tables.

Native conversions publish one complete copy per finite catalog slot. Concurrent
cold candidates copy outside the slot mutex; losers release after unlock. A failed
attempt rechecks a concurrent winner. Warm requests reuse the published copy.
The print build carries one `PrintProfileSource` lease and samples that exact Rust
owner, including across cache release and source-file changes. Recipes and CUDA
state retain independently allocated native values. Cold-copy, Arc retention and
gamma-overlap capacity measurements exclude allocator/RSS claims.

The consumer cases cover typed original-decimal BB values, preserved named
illuminants, canonical positional wavelengths, authored-label hash encoding,
broader interpolation axes and active DIR prerequisites, direct builds with an
unselected malformed print, and the three consuming Hanatos window cases.
They create scratch resources and do not modify fixture expectations or Resources.

CUDA terminal cases distinguish native close from fallible host cache release.
Concurrent successful borrowed shutdown calls observe one recorded host outcome
outside native locks; repeated calls do not purge or retire CUDA again. Native
failure stays primary and prevents borrowed cache release. Consuming destroy
always detaches/consumes the host graph under the existing reader-exclusion
precondition; native close alone decides graph deletion or uncertain retention.
The retained uncertain graph contains no Rust profile source or build lease.


## Production spectral sources

`Ffi.Host.SpectralOwner` checks all ten production reconstruction/CMF operations
with C11/C++20/Rust signatures and x64 layouts. The 24-byte `FjSpectraLutView`
borrows C-order 192×192×81 samples with their actual C6 identity. Mallett is
wavelength-major 81×RGB; CMF rows are `[wavelength_nm,x_bar,y_bar,z_bar]`.
The fixture checks output clearing, diagnostic bounds, consuming release,
independent source failures, source sharing, concurrent views and owner expiry.

`Assets.Host.SpectralBootstrap` and `Assets.Host.SpectralCopy` use the actual
Library/Root bootstrap and native copy implementations. Isolated test objects
observe copy extents and inject structural/copy faults; an executable-local
allocator verifies partial-copy cleanup and actual CMF construction failure.
Product objects contain neither hook. Fresh resource roots cover supported NPY
widths, equivalent headers, signed special values, narrowing overflow, malformed
sources and native platform paths. Source bits and identities have independent
expectations before selected computation. Sequential Root success/failure must
clear stale CMFs and reconstruction records. Warm bootstrap makes no copies;
native bytes survive consuming Library destruction.

The independent processor pixels cover Hanatos negative-direct and positive-print,
Mallett negative-print and Arctic positive-direct, plus Hanatos combined/glare
cases. `Ffi.Gpu.PreparedBoundary.mallett-direct` and `.arctic-print` add the missing
direct/print execution paths through cold/warm C, native direct and OFX processor
entry. They check path consistency, finite output, alpha and canaries, with no new
numerical oracle or fixture tolerance. Full-resolution captures use Hanatos and
remain separate evidence.

After configuring/building with tests ON, run the affected host groups with
`ctest --preset <preset> -R 'Spectral|Rust.Bridge|Quality.RustBoundaries|NativeBoundary'`.
Capacity receipts report requested payload bytes and actual capacities, excluding
allocator metadata/RSS. Each LUT has one 11,943,936-byte Rust source and one equally
sized native copy. Projections duplicate no samples; each opaque Box holds an
8-byte Arc. CMF triplets expire after complete curve publication. Source math moves
in S4.A; conversion/global storage is removed in S4.E and remaining Library/Root
forwarding in S5.C. Installed Resolve and real recovery qualification stay separate.


## Illuminant and calibration source boundary

`Assets.Host.IlluminantAbi`, `IlluminantCopy`, `IlluminantLibrary`, `IlluminantMath`,
`CalibrationAbi`, `CalibrationNative` and `IlluminantNative` exercise the private
production CSV and selected calibration calls. All require no driver/device at
runtime. The C11/C++20/Rust boundary asserts fixed layouts, explicit tags and
signatures. Curve captures under `fixtures/illuminants/` characterize the accepted
native producer at the manifest's revision; exact target-local float bits are
compared, with no new tolerance. Existing raw CSV and 160-entry CMY captures retain
their independent frozen expectations.

The immutable native curve snapshot holds independently allocated derived data.
Seven successful Rust CSV cache slots retain source rows; call-local opaque owners
expire after native elementwise copies. No raw rows or calibration JSON cache is
retained in native code. Complete sets alone are published; ordinary unavailable
sources return uncached partial snapshots. Allocation/internal errors abort cold
construction. Retained readers survive cache release; candidates build and losing
or old snapshots destruct outside cache locks. Executable-local allocation probes
measure requested native bytes and exercise reentrant release during actual
snapshot/loser destruction. They do not measure Rust heap or RSS.

Only explicit `test-support` enables the core classifier read/probe fault seam;
normal Release builds exclude it. F1 tests inject errors at the actual core read
boundary, observe probe bypass and sticky first-outcome caching, then verify ABI
allocation failure with cleared results and native print `bad_alloc` before any
recipe fallback. The CSV category probe injects a typed reader-capacity/poison
error at the raw edge; accepted core reader/cache tests cover their owning
behavior. Native view/copy/publication hooks are confined to test object targets.
Their qualification/removal boundary is S4.E when native conversions move.

## CAT16 preparation

`Color.Host.Cat16` compares the safe core facade and actual production wrappers
against independently captured accepted Stage 3 native scalar/matrix bits. It
also runs a real C11 caller for status layout, pointer/null handling, exact write
extents and contained production panics. `Ffi.Host.ColorPreparation` checks all
nine transforms, four complete Cmax tables, 15 complete route/recipe cases,
state/table ownership and actual admission failure/retry/recovery. Isolated
admission/profile objects supply the existing hooks; the shared product targets
receive none. Cases modifying the process-global pending hook run serially.

The shared test-support color fault is thread-local and one-shot. Its four
operation tags select CAT16/CAT02 matrix/scalar production exports by positive
one-based call index. It consumes the
fault before action, and preserves production pointer validation/output clearing.
Invalid arm disarms; clear is idempotent. Supersession tests perform N2's valid
raw-export calls for the original armed count on the admission thread before
cleanup, proving consumption for newer valid, invalid and uninitialized input.
Other-category/deferred controls throw inside real profile conversion; cold
asset panic/capacity cases exercise the shared admission boundary separately.

`Color.Gpu.ScannerReuse` separately proves actual six-allocation scanner LUT
address/content/identity reuse across Hanatos/Mallett direct/print CCTF-only and
warm transitions. It records bounded preparation times and retires only its idle
test-owned context between completed triplets.
The host groups execute no device operation. All groups have finite CTest
process timeouts. No ordinary run generates or changes expectations.

`fixtures/color/manifest.json` records native-parent provenance, representation,
case membership and exact identity order. Finite values, signed zero and infinity
signs are exact; NaNs compare classification. Each complete-consumer run selects
its preset capture. Existing Rust-density/render fixtures and their limits stay
unchanged. The temporary native value bridge has removal owner S4.E.


## CAT02 preparation

`Color.Host.Cat02` independently compares production wrappers and direct-core
facades with 1,874 accepted-parent leaf cases and 162 strict-threshold cases.
Its C11 caller checks status layout, null handling, positive-zero clearing,
exact write extents, read-only aliases, nonfinite values and panic containment.
The shared `fj_test_color_arm_fault`/`fj_test_color_clear_fault` slot covers all
four production operations. Unrelated operations, invalid pointers and facades
do not count; replacement, invalid-arm disarming and thread locality are checked.

`Ffi.Host.Cat02Preparation` checks 56 complete recipe cases, 144 real scanner
color cases, two early-invalid scanner returns, and 32 cases per retained scalar
helper. Recipes, table payloads, whites, matrices, encoding and identities compare
exactly. The signed Hanatos and nonnegative tables sanitation paths stay separate.
Current failures, recovery and N2 supersession exercise actual film/scanner
matrix calls on all four routes. Both retained scalar helpers propagate typed
failures. Same-thread raw calls before cleanup witness consumed faults. Retained
route owners preserve values through state replacement and cache release.

`fixtures/color/cat02-manifest.json` identifies the accepted native parent,
source/configuration/resource capture, explicit defaults, ordered controls and
new per-preset expectations. These are native-parent characterization fixtures;
ordinary tests never regenerate them. Known exceptional leaf floats compare
NaN classification; completed consumers, identities and control fields compare
exactly. Comparator controls reject transpose, one-bit, reversed whites, changed
identity and the wrong platform fixture. Platform differences in downstream
recipes remain in their respective captures. Existing CAT16 fixtures and limits
are unchanged. Both temporary color bridges have removal owner S4.E; device
CAT02 and input/RGB conversion families retain their existing owners.
