# Foreign ABI evidence

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
CMake for facade/fault consumers with `BUILD_TESTING=ON`. Shipping and production
witnesses always select the separate `cargo/production` archive; test-support
consumers select `cargo/test-support`. Enabling tests does not switch shipping
features. Product builds consume committed bindings without bindgen or libclang.
See the [ownership and maintenance map](../../CONTRIBUTING.md#ownership-and-maintenance-map).

## OFX C ABI evidence

`Ffi.Host.OfxAbi` links the real test-support Rust archive and native core with
independent strict C11 and C++20 consumers of `native/ofx_bindings.h`. Its six
custom cases are `layout`, `constants`, `signatures`, `callbacks`, `parameters`
and `messages`. The frozen inventory in `ofx_abi_facts.inc` and exact field types
in `ofx_abi_test.h` come from the vendored headers, independently of bindgen:

- 111 layout observations: size/alignment of 11 complete records, status/time
  and six distinct opaque handles, plus all 73 field offsets.
- Exact declared types of all 73 fields, including signed rectangle/point
  coordinates, unsigned plugin versions, pointer constness and all 50 suite slots.
- 17 numeric constants and 143 exact string values including terminating NUL.
- 53 callback/slot signatures, preserving nullable C function pointers and
  variadic tails. Both native languages and handwritten Rust type-check them.

The callbacks case exercises generated Host/Plugin fields in both directions
with exact tokens, arguments, writes and status. The parameters case calls actual
C variadic stubs through generated slots for int, bool, choice, string-choice,
double, double2D and double3D: current/at-time get/set, 28 calls. The message
case calls the generated V1 slot with literal `%s` and text containing `%`.
Fact counts are checked before indexed comparisons and mismatch reports identify
the field/constant. Rust exports and native spans are test-support-only and
immutable/library-lived; fixture tokens are never dereferenced. Rust unit tests
compile declarations and type evidence without linking the C fixture stubs.

The selected records are Host, Plugin, PropertySuiteV1 (18 slots),
ParameterSuiteV1 (18), ImageEffectSuiteV1 (13), MessageSuiteV1 (1), RangeD,
PointI/D and RectI/D. Complete suite layouts retain unused slots. The standard discovery exports are defined by `native/ofx_exports.cpp` and
delegate count and descriptor lookup to Rust. Rust owns OFX callbacks, lifecycle,
reentry, image ownership and borrowed host strings; native CUDA Root retains CUDA
resources and frame admission. Installed Resolve behavior remains outside these
fixture claims.

Run through the maintained owner in any native preset:

```sh
python -B scripts/verify.py development --preset linux-debug \
  --tests Ffi.Host.OfxAbi --runtime host --run-id ofx-abi-change
```

This product-linked host test requires CUDA compilation and executes no GPU
work. `Ffi.Host.BuildArtifacts` includes it in its actual native link/feature
closure; selecting that registration builds the broader artifact consumers
without running their unrelated runtime tests. New ABI acceptance also requires
the two native stable Release OFF production-contract closures before either
Release ON restoration, followed by the affected ON checks.

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
`[wavelength_nm,x_bar,y_bar,z_bar]`, before complete Rust curve preparation.
`profile::Role` selects film/print use in the catalog and loader;
`profile::Support` is the authored photographic substrate, film or paper;
`profile::Stage` is the authored filming/printing step. Printing can use either
substrate, so these axes and their schema/ABI mappings remain distinct.

Identity terms have separate scopes and encodings:

| Rust term | Contributing inputs and zero rule | Boundary/consumer |
| --- | --- | --- |
| `hash::{bytes,u64s,finite_f32s,resource_f32s,f32s_with_nan_mask}` / `FloatSpanHash` | FNV-1a primitives with each function's byte, signed-zero and NaN rules; no final zero remapping. `FloatSpanHash` has separate value and NaN-mask streams. | Raw encoding building blocks, not asset identities by themselves. |
| `SpectraLut::asset_hash()` | Decoded reconstruction sample bits; signed zeros and NaNs canonicalized in one stream, final zero mapped to one. No path, metadata or evaluator version. | `FjSpectraLutView.asset_hash`, native reconstruction reuse. |
| `FilmProfile::asset_token()` / `PrintProfile::asset_token()` | Stock and consumed metadata, tagged sampled tables/adaptation presence, authored exposure/model bits and density evaluator version; final zero mapped to one. Display name, file path and JSON formatting do not contribute. | Profile views' `asset_token` and downstream complete-state identities. |
| `profile_token` / `finish_asset_token` | Private producer and zero finalizer for that same completed-profile token. | No second identity contract or process-handle allocation. |
| `PrintDensityCurves::hash()` | Count, raw narrowed-axis bits and gamma-adjusted CMY total bits, in order; zero is rejected, never remapped. | `FjPrintDensityView.hash`, native print development. |

These content identities are deterministic across process runs given the same
encoded inputs/evaluator outputs. They are not cryptographic digests or
interchangeable across domains, and do not promise equal numerical outputs on
different platforms. Equal profile tokens do not imply equal gamma-adjusted
curve hashes, and a raw sample hash cannot substitute for a profile token.

Generated CUDA bindings additionally transport completed recipe, descriptor,
table and submission hashes unchanged. Each enclosing ABI record establishes
the component and its native producer/zero contract. Their `clip_token`,
`instance_token` and `frame_token` instead identify clip/session/frame lifetime
or temporal facts; they are not resource content fingerprints. Fixed ABI field
names remain governed by `native/juicer_cuda_api.h`.

Prepare and run the focused evidence through the maintained owner (substitute
any of the four supported presets):

```sh
python -B scripts/verify.py development --preset linux-debug \
  --tests Ffi.Host.CudaAbi Rust.Bridge --runtime host
```

## Native owner lifecycle

The OFX module and instance callbacks borrow the Rust-owned Runtime and Instance
through the Rust OFX implementation. `native/ofx_exports.cpp` supplies only the
standard discovery entry points and delegates descriptor lookup to Rust. There
is no C++ ProcessRuntime projection or legacy Runtime C export. Native CUDA
`Root` remains the owner of CUDA resources, contexts, epochs and frame
admission/retirement. Rust Runtime owns process Assets and the native CUDA handle;
consuming close preserves native-before-host failure precedence and releases host
state outside native locks.

`Ffi.Host.CudaOwner` and `Ffi.Host.CudaTerminal.*` retain their existing native
CUDA-owner lifecycle coverage: no-GPU creation, terminal transition, error
buffer, admission/drain serialization and native graph retention cases. These
tests exercise the native CUDA lifecycle boundary, not an ambient production
owner registry. A failed native close retains the native graph without
permitting replacement; the wrapper does not retry retirement. The factory
unload case checks that an actual close error maps to fatal OFX status.

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
| `fj_cuda_shutdown` succeeds | Caller retains a closed native handle; repeated shutdown succeeds without another drain. |
| `fj_cuda_shutdown` fails | Caller retains the blocked native handle and graph; repeated shutdown returns the saved failure. |
| `fj_cuda_destroy` receives accepting ownership | The caller consumes its pointer; one shutdown attempt leads to deletion or failed native retention. |
| `fj_cuda_destroy` receives closed ownership | The closed graph is deleted once and a later create can establish a new owner. |
| `fj_cuda_destroy` receives blocked ownership | Native ownership is consumed, the graph remains retained, and no retry/destructor traversal occurs. |

Rust `cuda::Owner` takes its optional native handle before invoking consuming
destroy. Drop uses that path only for an unconsumed handle. Runtime keeps Assets
alive through the native attempt and then releases host state on every consuming
result, including native failure. Successful borrowed shutdown releases caches
once through Rust's saved result; repeated calls reuse that outcome. Closed or blocked native owners reject inspect, render and
instance retirement.
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

Fixtures that need process Assets use the Rust test-support facade; standalone
native lifecycle cases create and consume `FjCuda` directly. Product effects
borrow Runtime and Instance through Rust-owned OFX callbacks, while native CUDA
Root remains the execution and resource owner.

`native/ofx_exports.cpp` is the native OFX discovery shim. CUDA execution values,
composers and resource objects remain native; retired C++ host preparation,
ProcessRuntime, effect and OFXS implementation sources are not in the product
host source set. Test-only scalar binders under `tests/ffi` call checked facade
operations directly.

## Release production probes

With testing ON, `Ffi.Host.PreparedValues` and
`Ffi.Host.OutputColorValues` register complete prepared-value and output-color
checks as two strict-C11 cases. Their test-only C sources call direct checked
facade operations; the prepared-values case consumes the existing generated
`production-probes/production-film-expected.h` and synthetic duplicate-film
fixture arguments.

The Release testing-OFF contract uses strict-C11 standard OFX discovery and
native CUDA ABI probes plus the prepared-render/native-CUDA contract. It records
ten logical commands in `production-contract-v2`; the offline reader continues
to accept historical v1 reports. The ON value fixtures are not OFF probes.

`python -B tests/ffi/production_contract.py --preset <release-preset>
--artifacts <directory> --generate-expected-header` runs the same fixture conversion without native
commands or driver access. It generates a C header from existing expectations;
it does not regenerate scientific fixtures. The OFF contract writes a v2 command report for standard discovery, native ABI
and actual prepared rendering. It does not generate the ON value fixture header;
the offline collector continues to read historical v1 command reports.

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

For OFX, add `--api ofx` to the same native command. The wrapper includes
`ofxCore.h`, `ofxProperty.h`, `ofxParam.h`, `ofxImageEffect.h`, `ofxMessage.h`,
`ofxImageEffectExt.h` and `ofxParamExt.h` from
`third_party/ofxs/OpenFX-1.4/include/`; image-effect transitive headers are
read without importing their unrelated suites. The only vendored compatibility
patch selects `<cstdarg>` in C++ and `<stdarg.h>` in C in `ofxParamExt.h`.
All foreign declarations, macros and identifiers remain unchanged.

The exact, explicit allowlists are `OFX_TYPES` and `OFX_CONSTANTS` in
[`regenerate-rust-bindings.py`](../../scripts/regenerate-rust-bindings.py).
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

## Complete prepared state

`JuicerPreparedStateTests` registers `Ffi.Host.PreparedState` and
`Prepare.Host.StateProducts`. The C11 helper and C++ reader compare every declared
layout fact with Rust. The product reader compares the complete Rust state with
31 ordered frozen expectations across all four routes with Hanatos, Mallett and Arctic,
plus contributing/ignored controls, Boolean snapshot casts, signed zero, failure
order and final identities. Its six competing-failure cases also check visitation
and nonconsumption of a later fault. Checked-in expectations are never regenerated
by these tests.

The checked test facade returns one completed immutable owner or a typed failure. A view borrows that owner; its
arrays are not detached application storage. Create nulls the owner output before
work, view clears every defined field on failure, and release consumes one hold
even when diagnostic admission fails. The test facade calls the same private
construction and projection as production. Rust compiler fixtures reject public
construction, Default, mutation and detached borrowing in both feature forms.

The three asset-bound illuminant, input-hull and output-boundary families share
the entries used by complete preparation. Their holds retain live backing across
cache release and Assets destruction. Entry identity describes that backing while
held, rather than a fresh foreign wrapper. The existing cache-owner processes
observe allocation, publication/selection, generation, failures and reclamation at
the Rust owner; test-only observation holds retain no Assets or payload.

`Ffi.Gpu.PreparedStateLifetime` uses the real C render entry for direct and print
with active spatial DIR and disabled optics, grain and physical effects. DIR frame
completion uses the literal `FjDirFrameInput` helper and the existing native
composer/encoder. The state and invocation expire after enqueue while a real native
staging event remains pending; completion checks every staged source word. The
12 scenarios per route include supplied, present-null and absent streams, both
post-preparation cancellation checkpoints, actual pinned allocation/copy/event
failures, quarantine, idle retirement and actual deferred destruction/drain after
Rust expiry. Rust
holds do not replace native staging, frame fences, finish/abort or retirement.
The production `RenderInstance` owns one Rust Publication. Capture tickets order
host actions; registered write guards prevent snapshot reads and admission while
a setter can still commit. Successful write drainage can request a conditional
read-only refresh. Current capture failures block admission without revoking
earlier admitted holds. Rust owns reuse, final preparation recheck and committed
build numbering. Each admitted `PreparedState` retains the same complete backing
and its input/build metadata, including after instance destruction. The core's
substantial backing uses a fallible reservation after scientific preparation.

The C11 protocol rows cover capture/write/snapshot/admission tags, revisions,
consuming completion, bounded diagnostics and close. AdapterTrace exercises real
getter/setter overlap, partial failures, exception status and image-release order.
These adapter results do not establish that Resolve retries an in-progress render
rejection. Production borrows the admitted Rust state through its private render
projection; native temporal submission, staging, frame work and exact-context
resources retain their existing ownership.

```sh
cmake --build --preset <preset> --target JuicerPreparedStateTests JuicerPreparedBoundaryProbe
ctest --preset <preset> -R '^(Ffi\.Host\.PreparedState|Prepare\.Host\.StateProducts|Ffi\.Gpu\.PreparedStateLifetime)$' --no-tests=error --output-on-failure
```

These focused checks supplement the affected family, owner and downstream GPU
groups. They do not establish installed-host or visual acceptance.

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

`Assets.Host.LookupSafety` runs literal synthetic-density, Scanner lookup and
Hanatos-window cases in the actual test consumer. The window helper lives in
`asset_lookup_test.cpp`; its former product hook/object is removed. The existing
admission and context-drain seams are independent and unchanged.

The group checks finite endpoints/interpolation, distinct duplicate selection,
ascending/descending Scanner traversal, singleton axes, ordered infinities,
NaN/shape/channel failures, and actual complete-state constructor/publication
failure with prior-state retention and recovery. It performs no CUDA
runtime or driver operation. CUDA linking/build prerequisites remain required.
[Independent expectations and provenance](asset_lookup_expectations.md) explain
why synthetic invalid brackets are unreachable for admitted axes and define the
three Hanatos formula cases. The production-profile consumer group also reaches
those cases with adaptation enabled through complete Rust preparation.

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
runs host-only value checks through the complete plug-in boundary and links the real native engine. Detailed pointer,
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
It calls the actual Runtime catalog boundary and Rust-owned catalog/source boundary.
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

Runtime creation is lazy with respect to catalog discovery and performs no CUDA
initialization. Safe Rust Runtime owns the process Assets and the sole native
`FjCuda` owner. Catalog acquisition retains an immutable Rust `Arc`; independent
handles survive Runtime destruction and each other's release. Reads allocate
nothing. Catalog paths do not cross this boundary. Caller excludes Runtime
destruction from acquisitions and catalog release from reads and outstanding
view uses. Outputs and diagnostics are aligned, exclusive and disjoint. Every
valid output is cleared before validation. Malformed diagnostics skip ordinary
work, while destroy/release consume their owner once on every result.
Truncation changes text only. Rust publishes one cached catalog result,
including ordinary source failure. Rust owns the catalog result, keys, labels, roles, defaults and ordering used by
the OFX descriptor. Test-only catalog helpers observe the same Rust-owned result;
there is no copied C++ UI catalog. Conversion and publication hooks are test-only
and remain at the checked acquisition/publication points.

Rust Runtime owns Assets and applies native-before-host failure precedence during
consuming close. Test-only Bridge and Library wrappers live under
`tests/ffi/test_asset_*`; they exercise explicit Assets ownership and are not a
product service locator.

The catalog path transport/Windows encoding/native selected-profile opening
bridge has been removed. NativePath encoding is a test fixture helper under `tests/ffi/native_path.*`; it
is not a product ABI or production path transport. Later-family production
acquisition remains outside this boundary. Installed Resolve acceptance remains
separate from these automated ABI, host and GPU checks.

## Production profile and gamma bridge

`Ffi.Host.ProductionProfileOwner` calls the Rust-backed test facade for film/print
acquisition, borrowed view, gamma sampling and consume-once release. C11, C++20
and Rust check every record field, tag and signature. All 28
profiles and the accepted gamma cohort are compared with immutable public
captures. Owned gamma totals survive print release, and retained profiles survive
Assets release/destruction. Safe projections share the core Arc and borrow tables.

Selected profiles and development remain Rust-owned. Raw holders and actual
complete-state publications retain their source backing across cache release and
source-file changes. Native ConversionSlots, selected-profile wrappers and copied
profile/development models are removed. Raw Arc retention and gamma-overlap
capacity observations remain distinct from native allocation or RSS claims.

The consumer cases cover typed original-decimal BB values, preserved named
illuminants, canonical positional wavelengths, authored-label hash encoding,
broader interpolation axes and active DIR prerequisites, direct builds with an
unselected malformed print, and the three consuming Hanatos window cases.
They create scratch resources and do not modify fixture expectations or Resources.

CUDA terminal cases distinguish borrowed native shutdown from prechecked,
consuming process close. Native failure stays primary and prevents Rust host
cache release. Consuming close first checks the native owner; only accepted
close consumes the Runtime. Native-only uncertain retention contains no Rust
profile source or build lease.


## Production spectral sources

`Ffi.Host.SpectralOwner` checks all ten production reconstruction/CMF operations
with C11/C++20/Rust signatures and x64 layouts. The 24-byte `FjSpectraLutView`
borrows C-order 192×192×81 samples with their actual C6 identity. Mallett is
wavelength-major 81×RGB; CMF rows are `[wavelength_nm,x_bar,y_bar,z_bar]`.
The fixture checks output clearing, diagnostic bounds, consuming release,
independent source failures, source sharing, concurrent views and owner expiry.

`Assets.Host.SpectralBootstrap` uses explicit raw CMF, Hanatos, Arctic and Mallett
source holds and actual complete-state consumers. The copy-only process, native
spectral globals, Root bootstrap and copy adapters are removed. Fresh resource roots cover supported NPY
widths, equivalent headers, signed special values, narrowing overflow, malformed
sources and native platform paths. Source bits and identities have independent
expectations before selected computation. Lazy acquisition distinguishes an
unselected malformed source from a selected construction failure. Raw retained
owners survive cache release and runtime destruction without native source copies.

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
allocator metadata/RSS. Each LUT has one 11,943,936-byte Rust source; opaque raw
holds retain its backing without an equally sized native copy. Source and
complete-state views borrow their owners. Native conversion/global storage is
removed. Installed Resolve and real recovery qualification stay separate.


## Illuminant and calibration source boundary

`Assets.Host.IlluminantAbi`, `IlluminantLibrary`, `IlluminantMath`,
`CalibrationAbi`, `CalibrationNative` and `IlluminantNative` exercise the private
production CSV and selected calibration calls. All require no driver/device at
runtime. The C11/C++20/Rust boundary asserts fixed layouts, explicit tags and
signatures. Curve captures under `fixtures/illuminants/` characterize the accepted
native producer at the manifest's revision; exact target-local float bits are
compared, with no new tolerance. Existing raw CSV and 160-entry CMY captures retain
their independent frozen expectations.

The Rust seven-curve cache owns derived data and CSV source rows. Scoped raw
illuminant holders expose borrowed views; retained readers survive cache release.
Cold construction, partial retry, concurrent winners, late publication, reentry
and reclamation are observed at the actual Rust owner. Native curve snapshots,
CSV copies, copy-only tests and wrapper-allocation probes are removed. The derived
illuminant visited set remains part of cold construction: an unrelated required
source failure is not reinterpreted as selected-source laziness. Capacity metrics
describe their actual scoped storage, not allocator metadata or RSS.

Only explicit `test-support` enables the core classifier read/probe fault seam;
normal Release builds exclude it. F1 tests inject errors at the actual core read
boundary, observe probe bypass and sticky first-outcome caching, then verify ABI
allocation failure with cleared results and actual complete-state failure before
publication. The CSV category probe injects a typed reader-capacity/poison
error at the raw edge; accepted core reader/cache tests cover their owning
behavior. Native copy/view hooks are removed; retained calibration and catalog
publication observations remain confined to their actual test providers.

## CAT16 preparation

`Color.Host.Cat16` compares the safe core facade and actual production wrappers
against independently captured accepted Stage 3 native scalar/matrix bits. It
also runs a real C11 caller for status layout, pointer/null handling, exact write
extents and contained production panics. `Ffi.Host.ColorPreparation` checks all
nine transforms, four complete Cmax tables, 15 complete route/recipe cases,
state/table ownership and actual admission failure/retry/recovery. Isolated
admission/profile objects supply the existing hooks; the shared product targets
receive none. Cases modifying the process-global pending hook run serially.

The shared test-support color fault is thread-local and one-shot. Its ten
operation tags select the four CAT16/CAT02 and six input-color production exports by positive
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
The host groups execute no device operation. Ordinary CTest processes have no
elapsed deadline (`TIMEOUT 0`); explicit fault-control bounds remain. No ordinary
run generates or changes expectations.

`fixtures/color/manifest.json` records native-parent provenance, representation,
case membership and exact identity order. Finite values, signed zero and infinity
signs are exact; NaNs compare classification. Each complete-consumer run selects
its preset capture. Existing Rust-density/render fixtures and their numerical bounds stay
unchanged. The strict-C11 value fixture uses the direct checked test facade; no production
C++ value bridge is retained.


## CAT02 preparation

`Color.Host.Cat02` independently compares production wrappers and direct-core
facades with 1,874 accepted-parent leaf cases and 162 strict-threshold cases.
Its C11 caller checks status layout, null handling, positive-zero clearing,
exact write extents, read-only aliases, nonfinite values and panic containment.
The shared `fj_test_color_arm_fault`/`fj_test_color_clear_fault` slot covers all
ten production operations. Unrelated operations, invalid pointers and facades
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
are unchanged. The scalar color fixtures call direct checked boundary operations through the
test facade; device CAT02 execution retains its existing owner. Input-color host mathematics use the boundary described below; CUDA decoders remain unchanged.


## Input-color preparation

`Color.Host.InputColor` compares production wrappers and direct-core facades with
four matrix/tag/white selections, 432 decoder triplets, 1,729 explicit-matrix
conversion cases, 54 unsanitized linear leaves and 216 ordered linear projections.
The C11 caller proves both record layouts, signatures/tags, complete/partial output
clearing, null/bad-flag handling, read-only input sharing and contained panics.
The shared ten-operation TLS fault slot also proves cross-operation counting and
replacement, invalid-arm disarming, thread locality, null/flag non-consumption,
facade independence and default-construction non-consumption.

`Ffi.Host.InputColorPreparation` compares 96 base completed products and 44
bounded auto-exposure, compression and output-only controls with all frozen
settings/defaults. Matrices, whites, both distinct mid-gray scales, TC/reference/
print balance, sensitivity/scanner arrays/descriptors and final/upload identities
remain exact. Real early and late failures cover foundation matrices, both Mallett
conversions, TC mid-gray and both synthetic-reference evaluations on negative/
positive direct/print routes. Cold/current failures, recovery and deterministic
pending supersession retain owners/hashes/counters; same-thread matching raw calls
before cleanup witness consumption. Retained products stay exact across asset
cache release and state replacement. Both retained DWG helpers propagate failure.

Whole-product retention comparisons keep independently owned expected bit
snapshots and read current published arrays directly. Exact lengths, every sample
bit (including signed zero and NaN payloads), absent-array handling, boundary
validity/hashes/diagnostics and complete recipe/table JSON remain checked there.
Input-color admission compares the affected input fields and contributing
identities directly, leaving unrelated full-state JSON and TC arrays to those
whole-product and lifetime owners. All 140 product cases, 144 operation counts
(including zero calls), 208 cold/current failure/recovery cases and 624
supersession witnesses retain their variants and operation order. Only the prior
immutable publication is shared once per 24 route/method/decoded-space baselines;
each fault's mutable state, counters, cold construction and recovery remain fresh.
Comparator controls reject length and individual-bit changes independently of
those fixtures.

`fixtures/color/input-color-manifest.json` contains self-contained native-parent,
compiler, control, representation and reference-role provenance. The owner-adopted
6 October 2026 inverse mapping uses A from every agreeing completed recipe/config
as the new matrix producer's exact reference. Original standalone/default inverse
observations remain native characterization; Windows Release B is retained there.
All four expanded logical capture expectations and supplied inverse operands are
unchanged; physical payloads use the shared storage described below. Conversion
tests consume their supplied inverse, including B, without replacing it through
the matrix producer.
Every other numerical/identity obligation and existing fixture/limit is unchanged.
Ordinary tests read only public fixtures and never generate expectations.

Complete Rust preparation supplies all three consumed matrices and the adaptation
flag. Prepared decoding copies them directly into focused resource inputs and
frame-owned values; the former default-identity `FilmRawConfig` carrier is removed.
Host input mathematics remain value-only Rust operations with synchronous,
exclusive/disjoint foreign outputs. Native enum/labels, closed execution values
and CUDA decoding/execution retain their assigned responsibilities.

`Color.Gpu.ScannerReuse` additionally checks decoded BT.2020 and sRGB Mallett states
on negative direct/print routes. Cold, output-CCTF-only and warm preparation reuse
all six scanner allocations with identical contents/identities. It verifies exact
prepared film matrices and unconsumed host-math faults during native preparation.
These are GPU checks; host groups introduce no eager CUDA-driver dependency.


## Illuminant construction

`Assets.Host.IlluminantConstruction` compares raw production C calls, the direct
safe-core fixture facade and actual held direct/print callers against
four frozen accepted-parent captures. `fixtures/illuminants/construction-manifest.json`
records exact provenance; existing seven-curve fixtures remain immutable. It covers
41 BB cases, direct/Akima boundaries, all 270 caller outcomes and 144 complete
products, including reachable EQUAL print/preflash and the native 3200 K lens
contract. Matrices, spectral samples, whites, filtered lights and all compared
identities are exact. Synthetic interpolation NaNs use their established
classification contract; no tolerance is introduced.

The seven production math operations use required bounded diagnostics and cleared
failure outputs. Lens finish/release take and clear the owner before validating
other descriptors. Scoped raw CSV acquire/view/release holds retain Rust sources
through synchronous calls and cache/Assets release, without native source-row copies.
The Rust seven-curve cache remains the derived owner. Existing order,
partial retry, decoded-empty, concurrent winner, late-publication, retained
view and reentry cases remain active. Printed Rust source/cache and Akima Vec
capacities are actual scoped observations, excluding allocator
metadata/RSS; Rust capacities are cumulative allocations, not a live high-water mark.

Closed TLS fault selectors prove intended raw-export consumption and independence
from CSV/color slots and the facade's own slot. Checked scratch-capacity errors map
to AllocationFailure in production. Panic containment does not recover ordinary
Box/Arc allocator abort. Compiler probes reject default/field construction and
mutation of private `LensInput`; the safe core exposes only slice/value APIs.

Construction/admission tests retain old publications while rejecting failed
replacement and allow subsequent recovery. EQUAL construction fails before
publication and preserves the prior print state. The separate
`Ffi.Gpu.CudaRender.negative-print` rows exercise actual native prepared execution,
prepared-frame abort, absent-stream completion and its existing error precedence,
then reuse resources after clearing the fault. The
completion override is isolated test code following the physical synchronization;
it does not establish real driver failure or Resolve recovery acceptance.
All host groups remain free of driver/device runtime requirements. Normal Release
excludes the fault, memory and observation seams.

### Spectral tables, whites and film S inverse

`Spectral.Host.Tables`, `.White`, `.SInverse`, `.Products`, `.Admission` and
`Ffi.Host.SpectralPreparation` use the four native-parent captures in
`fixtures/spectral/`. The manifest identifies accepted source, preset, frozen
capture and independent hash derivation. These are Film-Juicer migration
contracts. Ordinary tests neither generate fixtures nor consume private or
upstream files. Finite bits, signed zero and identities are exact; arithmetic
NaN payloads compare by classification. Captured intermediates retain the
rounded float products, double sums and determinant for diagnosis.

The safe core owns fixed complete Tables/White results with private readonly
storage. Three synchronous C11 exports borrow initialized fixed source arrays,
clear defined outputs before fallible work and retain no pointers. The existing
native payload owns projected vectors. Table products/reciprocals and
scanner-white promoted products/reciprocals deliberately remain distinct.
The film table white/hash gate precedes the weighted-CMF S inverse; singular
S retains its strict determinant threshold and identity fallback.

Tests compare raw exports and the direct-core facade independently to captures,
then exercise real route products, all serialized controls and enclosing hashes.
They check closed white failure metadata, malformed spans, TLS raw/facade fault
independence, typed propagation, publication distinctions, admitted holds, cache
release, exact operation counts and recovery. A controlled CMF scaling reaches
actual float-reciprocal overflow and proves inverse suppression. Fixture-only
replacement of C++ allocation rejects the actual 324-byte native vector request;
Rust fixed-array math allocates no heap. Reported vector capacities are not RSS
or live-peak measurements. Compiler probes accept normal result consumers and
reject Default, field construction, direct mutation and readonly-view mutation
with and without test support. Definition-specific source controls guard the
removed native producers.

Products run in disposable resource copies under the selected validation tree.
These groups require no driver/device at runtime. Existing scanner GPU reuse,
full route/scanner suites and installed-host acceptance keep their separate scope.
Normal Release excludes fault slots, facade exports and fixture allocators.

### Hanatos reference and final sensitivity

`Exposure.Host.Reference`, `.Sensitivity`, `.Products`, `.Admission` and
`Ffi.Host.ExposurePreparation` check A5 reconstruction, final film sensitivity,
recipe Mallett scale, both C11 exports, direct-core facade, actual complete-state callers,
and publication/retained-owner behavior. The 68 reference cases and 544 sensitivity
cases retain exact finite bits and classify arithmetic NaNs. Complete products
cover 420 attempts across methods, routes, profiles, illuminants, windows, surfaces
and UV/IR activation. Reconstruction and TC construction are Rust-owned; native
execution borrows completed values.

The four numerical cohorts share `fixtures/exposure/numerical.json`; only public
consumer inputs and expectations are stored there. Intermediate native and
pinned-candidate traces remain in the original capture evidence. Complete-product
cases keep their preset-specific controls, failures and membership; successful
rows reference readonly records in `product-values.json`. Every completed field
is still compared. Admission tests require no numerical fixture and do not load
one. `Exposure.Host.FixtureFormat` checks manifest hashes, case membership,
record indices and exact result-sharing controls including signed zero, NaN bits
and 64-bit identities.

The manifest records the accepted source revision, original fixture hashes and
verified format migration. For explicit maintenance, prepare a candidate without
running any mathematical producer:

```sh
python tests/ffi/compact_exposure_fixtures.py --source-ref <accepted-revision> \
  --output-dir out/validation/<preset>/exposure-format-candidate
```

This operation requires the original per-preset format and an unused output
directory. It proves equality of all four numerical consumer projections and
every expanded complete-product document before writing candidates. Ordinary
tests never run this maintenance operation or read Git/private/reference data.

A5 uses pinned `libm::erff` 0.2.15 under the owner-approved numerical decision.
Native and pinned-Rust preimplementation studies retain provenance in the
exposure manifest. Reference reconstruction and ordinary f32 exp retain exact
native-defined expectations. Final hashes describe the actual Rust coefficients.
The small rounding differences and resulting enclosing identities are accepted
for this family; no numerical bounds in another family are widened.

Earlier color and spectral fixture bytes remain unchanged. Exact supplements in
`fixtures/exposure/updates/` use test/replace JSON operations over completed
products replayed independently through the frozen native parent with frozen
approved A5 values. The tests still compare every field and reject wrong preset
values. Leaf expectations, source controls and failure outcomes retain their
original authority. Ordinary tests do not regenerate any fixture.

The tensor is borrowed synchronously and no foreign source survives the call.
Completed core results have private readonly fields and no Default. Checked
kernel sizes and real fallible reservation distinguish impossible layouts from
allocation failure. Raw APIs return the typed status; actual complete-state
publication retains its existing allocation-failure exception mapping.
Closed raw/facade fault slots prove consumption before cleanup or recovery and
are absent from normal Release. Tests cover malformed consumed spans, ignored
inactive spans, panic containment, cleared outputs/metadata, concurrent calls,
source expiry, signed-zero/tiny blur, failure ordering, cache release and recovery.
Visual A/B comparison was declined; these automated checks do not establish
Resolve render acceptance or real driver/context-loss recovery.

The existing processor and parameter-identity tests select new exact per-preset
A5 fixtures under `tests/ofx/fixtures/exposure/`. Seven processor rows and ten
identities come from frozen-parent GPU replay using the original pinned study
values. Candidate GPU samples must agree bit for bit before adoption. The old
processor/identity files and every existing pixel bound remain unchanged. This
is automated numerical capture; no visual comparison images are generated.

## Output gamut construction

`Rust.Core` (the `output_gamut` integration binary) checks all nine transforms and complete 64×720 boundary
arrays against the four-preset native characterizations, including ordered
primitives, intermediate cubics, platform libm bits and raw identities.
`Gamut.Host.Output.boundary` checks both production exports and direct facades,
actual C11 layouts/calls, D1–D5 optional diagnostics, cleared failures, canaries
and consume-once transport. `binding` exercises native typed failure mapping.

`Gamut.Host.Output.Cache` executes finite, acknowledged W1–W7 schedules through
isolated test-only Library/binding holders. It checks selection across clear,
successful stale private returns, matching/conflicting same-generation controls,
own failure/retry, exhaustion and cleanup precedence, immutable retained arrays,
A8 hull retention and destruction outside the cache lock. These concurrent owner
observations qualify the changed cache contract; they are not native parity.
The fixture holders still exclude active readers during destruction. Existing
production CUDA groups retain their route, disabled-work and asynchronous
staging witnesses.

Native bindings expose completed fixed values. Rust privately owns complete
immutable mathematical values and one fallibly reserved temporary table, copied
once per successful cold candidate. Warm test-holder hits introduce no foreign
call, table copy or rehash. Test-support copy timings are absent from ordinary
production builds.
Injection demonstrates status/panic/zero-identity transport, not real OOM,
a zero-hash preimage or process-abort recovery. The capacity-overflow test
exercises the same actual fallible reserve boundary separately.

## Prepared-host sanitizer pilot

The optional Debug-only `JUICER_PREPARED_FFI_SANITIZERS` CMake option defaults to
OFF. With `BUILD_TESTING=ON`, configure either native Debug preset with
`-DJUICER_PREPARED_FFI_SANITIZERS=ON`, build the complete preset, then run the
unfiltered CTest preset. `ctest --preset <preset> -L sanitizer` is a focused
diagnostic invocation and does not replace that complete qualification.

The pilot compiles all three existing prepared-host test sources and production
`native/juicer_cuda_prepared.cpp` / `juicer_cuda_descriptors.cpp` with
`-fsanitize=address,undefined -fno-sanitize-recover=undefined`; Linux also retains
frame pointers. The original `Ffi.Host.PreparedProjection` target and assertions
remain unchanged. Four added host processes cover the same prepared-host cases,
a clean runtime control, production-decoder use-after-free and signed integer
overflow. Controls require nonzero exit and the specific sanitizer report; an
arbitrary crash, missing runtime or silent return cannot pass. Raw failure output
and executable/log hashes are retained under the preset's `prepared-sanitizers/`
artifact directory in separate fresh run directories.

Windows uses the selected Clang compiler's ASan DLL and symbolizer directories in
these tests' environment only; configure fails if the required DLL is missing.
The instrumented decoder/projection archive members are replaced only in pilot
executables. The ordinary core archive, Rust 1.98.1 static library, GoogleTest,
CUDA host objects elsewhere and CUDA device kernels remain uninstrumented. In
particular, `ThreeChannelSamplesView::sample` remains in an uninstrumented
resource translation unit; its existing behavioral assertions still run, but
this pilot does not claim complete instrumentation of that path. Leak detection
is disabled on both platforms (`detect_leaks=0`) and is not qualified. Allocation
interception is not equivalent to instrumentation of individual Rust loads.
Windows string/vector/optional annotations are also disabled in the pilot
objects to match the uninstrumented core's STL definitions. Consequently, reserved
capacity versus logical-size container overflows and disengaged-optional accesses
are not qualified, although
ordinary ASan access checks remain enabled. The linker ODR check is preserved;
see [Microsoft's annotation requirements](https://learn.microsoft.com/en-us/cpp/sanitizers/error-container-overflow).
No production rendering, Rust ownership checkpoint or scientific admission rule
changes. This pilot adds no CI lane.

## Bounded cache and owner repetition

For this explicit maintenance campaign, prepare the owning native Debug targets
through `verify.py prepare` before running this exact
seven-group host cohort (replace `linux-debug` with `windows-clang-debug` on
native Windows). Create a fresh campaign directory under the preset's validation
root and retain the verbose output, log and JUnit from that invocation:

The example uses shell line continuation; join it onto one line in PowerShell
and change the preset and output paths to `windows-clang-debug`.

```sh
ctest --preset linux-debug --parallel 1 --no-tests=error --repeat until-fail:20 --stop-on-failure \
  -R '^(Gamut[.]Host[.]Output[.]Cache|Ffi[.]Host[.]CudaOwner|Ffi[.]Host[.]CudaTerminal[.](serialization|shutdown-progress|host-cache-failure|host-cache-native-precedence)|Ofx[.]Host[.]CudaOwnerUnload)$' \
  --verbose --output-on-failure \
  --output-log out/validation/linux-debug/<campaign>/repetition.log \
  --output-junit out/validation/linux-debug/<campaign>/repetition.xml
```

Confirm discovery contains exactly these seven groups before execution. Each
successful invocation runs 140 independent test processes. Repetition launches
fresh processes rather than looping inside a retained process owner. Existing
arguments and `RUN_SERIAL`/resource-lock properties remain. Ordinary processes
use `TIMEOUT 0` with no elapsed deadline. This cohort executes no GPU work or
Cargo operation. Keep configuration/build
writers and other CTest invocations out of the same build tree while it runs.
Keep the finite repetition count and injected-control bounds. Historical campaign
deadlines and timings retain their observation-time meaning; ordinary execution
now has no outer elapsed kill deadline.

The cache test retains its W1–W7 acknowledged gates, matching/conflicting fixed
operands, all table-word comparisons, fault codes, generation exhaustion,
cleanup precedence, retained A8 hull and destruction-scope assertions. Owner
groups retain failed-close, consume-once, shutdown/admission serialization,
host-cache failure precedence and OFX fatal-unload contracts. These tests use no
PRNG; fixture bytes and forced schedules are the reproducibility inputs. Do not
add shuffling, change seeds in another test, widen numeric bounds or regenerate
fixtures for this investigation.

`until-fail` stops a test at its first failure. Preserve that original log and
receipt, stop the investigation on any failure/skip/timeout or missing group,
and diagnose before a separately named retry. A later pass cannot explain away
an earlier failure; do not substitute `until-pass` or `after-timeout`. JUnit
records the final result per group rather than every repeat: retain verbose
output and verify all 20 executions per named group. A clean bounded run proves
these repetitions, not exhaustive interleaving coverage or future absence of
flakiness. No required group moves to optional/nightly execution.
# CMake/Cargo build regression

`Ffi.Host.RustBuild` runs `run_rust_build_contract.py` with the configured native
Cargo, CMake, Ninja and C compiler. It exercises Debug and Release in miniature
workspaces isolated from the product Cargo directory: fresh/no-op completion,
test-only inputs, missing outputs, source/feature/glob propagation and failed
Cargo retries. Command logs and failing inputs remain under the preset's
`ffi/rust-build/` validation directory; successful fixture build caches are
removed. No GPU or reference-generation environment is required.

`Ffi.Host.OutputColor` checks the completed fixed output-color export and its
independent direct facade. Its frozen native characterization covers all nine
outputs and the distinct linear DWG source, exact f32 matrix/coefficient words,
canonical-zero component hashes, scanner fields and original fallback tags.
C11/C++/Rust assert the coordinated encoding/preparation/enclosing layouts.
Fresh concurrent first access, independently valid output clearing, canaries,
strict raw tags, status-only faults and operation-specific consume-once panics
remain separate witnesses. The catalogue has no fallible allocation contract.

`Ffi.Host.PreparedProjection` and the complete C render fixtures carry schema 1
encoding records through normal and cold preparation. Missing/unknown schemas
and kinds outside 0–5 fail even with CCTF disabled. Admission binds caller-owned
values without Rust calls or coefficient-based readiness. Both routes, both
encoding flags, complete value round trips and source expiry are covered.
Cold callers remain responsible for matching tags, prepared values, identities
and live initialized aligned extents. The scanner double encoder is a continuing
test-only oracle with frozen native f32 inputs/matrices and its original f64
arithmetic; candidate setup uses migrated preparation.


### Shared capture storage

Color and illuminant construction manifests retain the original native capture
identities and provenance. Their `payloads` entries map each logical capture name
to its physical JSON file; `storage_files` hashes the maintained representation.
The capture hashes still describe the original captured representation. Identical
Linux Debug/Release and applicable Windows color payloads share storage, while
the two distinct Windows input-color captures remain separate. TC preset
membership similarly shares the one byte-identical complete table file.

`exposure_fixture.h` expands per-family `expected_blocks` before applying the
qualified color supplement. The logical filename selects that supplement and
must still match `original_fixture`; its ordered `test`/`replace` operations are
unchanged. Raw Rust leaf/explicit-conversion and Mallett helper readers retain
their raw capture semantics. Ordinary tests never rewrite or recapture values.

`Ffi.Host.SharedFixtureLoading` exercises actual C++ alias/patch independence and
rejects unknown logical captures, invalid block indices and wrong supplement
identity. `Ffi.Host.SharedFixtureFormat` belongs to the corpus selection: it
checks each physical payload once, all native memberships and ordered recipe
inputs. Numerical groups still compare actual outputs with the applicable
native expectations. Small frozen foreign-value witnesses retain wrong-preset
rejection without loading three unrelated complete documents on each run.

### Film input and complete input color

`film::input::Recipe::new` constructs resolved film-input controls, a private
`color::InputColor` containing all three matrices and consumed whites, final
sensitivity, recipe Mallett scale and existing schema-2 TC/film-input identities.
Its resources are synchronous fixed-array borrows and existing identities; the
result retains no profile or resource pointer. Explicit-operand color leaves
remain separate, including the accepted inverse-B conversion fixtures.

The strict-C11 film-input fixture calls the direct checked test facade with the
fixed-value record.
Actual complete Rust preparation constructs the input recipe after its existing
control/source checks; the native foundation and `FilmSource` adapter are removed.
The direct fixture facade shares the
safe constructor and fixed projector. Both outputs are required; diagnostics
are optional. The legacy header documents numeric storage admission, caller
lifetime/disjointness obligations, inactive defaults, clearing and diagnostic
bounds. Failure dispatch uses the closed status/origin/reason envelope and
sensitivity metadata. Raw status/clearing and actual complete-state publication
failure remain separate witnesses; the vanished C++ translator is not recreated.
The raw result is complete input preparation, not a completed render state.

`Ffi.Host.FilmInput` and `Film.Host.InputProducts` own ABI/value/failure and actual
consumer coverage. `Quality.RustBoundaries` rejects default, field, mutation and
partial construction; source review/guards restrict internal complete-color
assembly to the recipe constructor. The existing input-color, exposure,
development and scanner-reuse groups retain their independent expectations.
The Release production gate compares C11 fixed values with the frozen native
capture, checks failure/clearing behavior and scans archive/module/PDB artifacts
for fixture/fault isolation. No per-frame preparation or CPU pixel path is added.

## Capture-film DIR and density bounds

The raw development retain/release protocol returns the same address with a
separate consuming obligation; its fixture holders exercise actual ownership and
borrow expiry. Complete Rust preparation owns the development and DIR products.
The native foundation/DevelopmentHold model is removed. Native execution binds
completed DIR arrays and fixed values, with native staging owning later
asynchronous CUDA inputs.

Minimum, DIR and film bounds retain their distinct construction positions and
failure priority. The safe bounds constructor borrows a completed minimum; its
raw adapter reconstructs it from the unchanged original request. This minimum
remains independent of Film-Juicer visual particle density. Print-medium bounds
retain their owner. Spatial-DIR frame mathematics is described below; workspace
policy remains native.

Raw prepare/acquire/view results and failure records are independently cleared.
Success commits `{NONE,3,UINT64_MAX}`. `LOG_EXPOSURE=16` is legal only for DIR
acquire with PreparationFailure, API_NONE, native code zero, channel 3 and the
first failing sample in `[1,N)`. Raw and complete-state cases retain the exact
strictly-increasing exposure diagnostic and typed envelope. Pointer checks do not
establish actual lifetime, extent or disjointness:
callers authorize initialized live storage, and views expire on consuming release.

See [immutable native characterization](../film/dir-fixtures/README.md).
`Ffi.Host.FilmRecipes` covers raw/C11/facade ownership and
`Film.Host.DirDensityProducts` covers complete callers/admission. Construction
compiler controls reject Default, partial field assembly, private storage access
and mutation. Both normal Release production probes cover the new symbols,
C11 values/failures and absence of test fault/facade code in final artifacts.

The spatial-optics owner is `film_juicer_core::optics::Recipe`: a complete fixed
value with private storage and named read-only getters. Actual complete-state
preparation constructs it in Rust. The native adapter and legacy export are
removed. `fj_test_spatial_optics_prepare` and its existing C11 consumer invoke the
checked raw operation for literal fixtures, copying authored f64 diffusion,
captured f32 halation/lens values, defaults and print selection. It commits all
fields or clears independently admitted outputs; `FjSpatialOpticsFailure`
preserves typed component/field/channel failures.
Stage 9 adds a one-shot panic and a zero substitution at the real resolved-hash
guard; all controls and strings remain testing-only. Raw cleared family/domain
zero differs from a successful inactive recipe's BlackProMist/film defaults.
See `Ffi.Host.SpatialOptics`, `Film.Host.SpatialOpticsProducts` and the immutable
[optics characterization](../film/optics-fixtures/README.md). The three superseded
optics/print-illuminant/preflash legacy exports and their 39 standalone OFF rows
are removed. ON raw C11/facade fixtures and actual complete-state products remain.
The current collector rejects their six retired report fields, including null
values; historical reports and captures retain their original bytes. Final I8
requires positive complete-state direct/print execution in both native Release
testing-OFF configurations, followed by ordered ON restoration and broad
qualification. Current test registration or miniature protocol checks do not
establish those gates.

## Diffusion sample, frame and component preparation

The strict-C11 diffusion fixture calls direct checked test-facade operations
for sample, frame and component values. These operations share checked Rust
bodies with product preparation. No pointer is retained and no
CUDA owner is introduced. The permanent CUDA records remain unchanged.

Sample support construction, component readiness and optional frame activity are
separate guarantees. Frame activity uses component hashes and preserves ignored
inputs on inactive work. Failures carry closed reason/stage/field metadata and
clear each independently admitted output. Native standalone failure keeps its
BlackProMist default; raw failure clears family to zero. Optional diagnostics
support absent/zero-capacity backing and bounded NUL-terminated truncation.

See the [exact native characterization](../diffusion/frame-fixtures/README.md),
`Ffi.Host.DiffusionFrame` and `Optics.Host.DiffusionFrameProducts`. The ON
strict-C11 fixture executes 507 complete frozen value/failure results through
the direct test facade. OFF Release commands verify standard discovery, native
ABI and prepared rendering. Native execution binds completed values and does
not reconstruct the removed owning recipe graph.

## Spatial DIR mathematical frames

`film_juicer_core::film::dir::frame::Plan` owns physical-to-pixel scaling,
geometry admission, fixed Gaussian components/operators and f64 YVV recurrence
coefficients. Its private immutable storage has read-only accessors. An ordinary
failure carries a separate `FailureTrace`, including completed components and a
partially written next slot; it cannot be consumed as an executable Plan.

The strict-C11 DIR-frame fixture calls the direct checked test facade with `FjDirFrameInput`, `FjDirFrameMath` and `FjDirFrameFailure`. Each
independently admitted output clears before argument admission. Raw executable
Math remains zero on failure; typed ordinary progress belongs only to the failure
record. Optional diagnostics use bounded NUL-terminated truncation. The complete
operation, including projection, contains panic. The testing facade shares that
same checked body; its one-shot fault controls are absent from production.

The closed native `JuicerDir::MathOutcome` admits completed raw mathematics for
native workspace composition. The fixture and prepared decoder retain the same
scratch tiers, plane roles/aliases and the existing sole
final descriptor hash. Native allocation, resources and lifetime are unchanged.
The zero-weight component target-tier difference in prepared decoding remains
intentional; its unconsumed representation is not repaired and the carried hash
is not recomputed. Synthetic active hash-zero characterization does not bypass
the permanent prepared boundary's existing identity admission.

See the [four-context native characterization](../dir/frame-fixtures/README.md),
`Ffi.Host.DirFrame`, `Film.Host.DirFrameProducts` and `Ofx.Gpu.DirFrame`.
The last group observes both actual prepared direct/print schedules before their
existing failure branches, and uses the existing GPU lock with no elapsed deadline.
The ON strict-C11 DIR-frame fixture executes all 121 mathematical/default/failure
rows through the direct checked test facade. OFF Release contract commands cover
standard discovery, native ABI and prepared rendering. Native execution inputs
are completed values. Reflected-YVV
boundary construction is described below.

## Spatial DIR reflected boundaries

`film::dir::boundary::Boundary` owns both axis products, the immutable combined
binary64 weights, packed terminal matrices and the existing 100-byte FNV key.
Private complete construction exposes read-only borrows. It preserves the native
double-double operation order and uses pinned `libm::fma` only for the original
explicit fused product residual. Signed size admission covers retained Root/CUDA
index expressions as well as vector extents and doubled axis periods.

The retained native `JuicerDirBoundary` operations
`fj_dir_boundary_{describe,acquire,view,release}` transport literal
`FjDirGaussian` coefficients. Describe derives only the key
after outer eligibility. The native `JuicerDirBoundary::Boundary` hold is acquired
after the warm-cache opportunity, outside the resource mutex. Its borrowed view
survives synchronous staging; native pinned memory owns later asynchronous reads.
Native context/epoch checks, collisions, device allocation and retirement remain
in their existing owner. The separate final frame-descriptor hash is unchanged.

Each independently admitted writable peer clears before other admission. Typed
failures preserve axis order and metadata. A NULL diagnostic is valid; admitted
zero capacity ignores backing and reports length zero. Positive capacity writes
at most capacity-1 bytes plus NUL, with length counting only written bytes.
Truncation changes neither status nor typed metadata. Release consumes a legitimate
owner exactly once even with malformed diagnostics or contained release panic.
Fallible vector reservations do not remove the existing infallible Box allocation
abort limitation.

See [the independent four-profile characterizations](../dir/boundary-fixtures/README.md),
`Ffi.Host.DirBoundary`, `Film.Host.DirBoundaryProducts` and the existing
`Dir.Gpu.ExposureCache`. The ON strict-C11 fixture checks all 171 boundary rows
through the test facade; the four `fj_dir_boundary_*` operations remain the native
boundary API. OFF Release commands verify standard discovery, native ABI and
prepared rendering.

## Visual-grain mathematical frame boundary

The safe immutable `film::grain::frame::Plan` owns callback-local temporal
conversion/seeds, particle/layer mathematics, Gaussian descriptors and energy
normalization. Construction completes before any result is transported; it
retains no curve borrow. The existing static visual-grain Recipe stays policy
authority. Native code retains scratch-shape selection, the sole final descriptor
hash and all CUDA allocation/context/epoch/upload/publication/retirement policy.
Permanent prepared ABI and recipe/chroma/amplitude/preset/layer-axis fields stay
unchanged.

`FjGrainFrameInput`, `FjGrainFrameMath` and `FjGrainFrameFailure` form the checked
synchronous temporary edge. Layer spans are admitted only when their ordered
reduction is reached. Raw admission and inactive-work behavior are checked
independently; vanished model-pointer-only outcomes remain historical capture
dispositions. Every independently admitted raw output clears
before other admission. Failure carries no partial executable mathematics; native
failed output returns its documented default. Optional diagnostics report written
bytes, with zero-capacity backing ignored and positive capacity reserving NUL.
The complete operation contains unwind. Test-only one-shot faults and observer
hooks stay outside ordinary objects and testing-OFF artifacts.

The ON strict-C11 fixture calls the direct checked test facade and native
composer; the retired legacy export is not part of the product ABI.
See [the independent fixtures](../grain/frame-fixtures/README.md),
`Ffi.Host.GrainFrame`, `Film.Host.GrainFrameProducts` and `Ofx.Gpu.GrainFrame`.
The C11 reader checks all fixed field/layout/signature facts and complete results.
R1 requires C layer0/count failure before M total0 after all nine reductions;
its cleared output, reason2/field12/layer0/channel0 and diagnostic are exact.
The ON strict-C11 fixture retains all 188 grain rows plus predecessor results.
OFF Release contract commands verify standard discovery, native ABI and prepared
rendering, with test-only fault/facade code excluded from product artifacts.

`FjEffectsFrameInput`, `FjEffectsFrameMath` and `FjEffectsFrameFailure` carry the
fixed physical-effects mathematical frame. The real ON reader and generated OFF
reader share `effects_frame_abi_assertions.inc`; `effects_frame_probe_contract.py`
owns the 180/40/1 codec dimensions and independent ordered common-ABI case IDs.
C projection remains explicit. Static 42-member dust and 36-member scratch policy
is bound from the completed prepared C record when a complete frame is present. Raw
failed/inactive results clear to zero; native defaults retain cosine one. A
present active zero-work result may have hash zero. The native bridge does not
repeat Root's later resource admission policy.

See [the independent frame fixtures](../film/effects-frame-fixtures/README.md),
`Ffi.Host.EffectsFrame`, `Film.Host.EffectsFrameProducts` and `Ofx.Gpu.EffectsFrame`.
The six test-support faults are closed and one-shot. Ordinary objects and both
production OFF artifacts must exclude their seams and the retired capture hook.
