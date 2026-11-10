# OFX host fixtures

`Ofx.Host.Module` calls the real default-feature Rust descriptor through the
small native discovery shim in `native/ofx_exports.cpp` and
`native/juicer_ofx_api.h`. Rust owns OFX callbacks and lifecycle; native CUDA Root
retains CUDA resources and frame admission. It checks
discovery/binding, transactional Load, Describe, dispatch, reentry, reload and
Unload. Describe compares thirteen authored properties plus host-seeded
`OfxPropType` with the unchanged Describe-only `descriptor_manifest.json` and
declares 303 status/ordering/target rows. Load and lifecycle add the current required-suite slot census and 7 rows,
including the required context and instance suite slots.

The three `Ofx.Host.ModuleControls.{Construction,Panic,Terminal}` registrations
link only the separate Rust test-support archive and run in distinct processes.
Their ten named cases cover C11 platform ABI/native paths, actual construction
rollback, panic containment, rejected-close backing preservation and consuming
native retention. The terminal process never resets retained native ownership.
Construction and panic cases include four deterministic worker interleavings for
late panic cleanup, admitted panic exclusion and invalidated-load rollback. The
feature-only pause controls report bounded-wait expiry as failed evidence. Native
path cases compare two literal parent extractions before normalization against
the C++ filesystem expression, including an immediate dot component.
`Ofx.Host.Context` compares the candidate DescribeInContext action with an independent 128-name/type census. `Ofx.Host.ContextControls`
uses the real Runtime with test-support fault controls for catalog readiness,
recoverable allocation and panic containment. These checks exercise descriptor
declarations, not instance parameter values or rendering.

`Ofx.Host.Instance` compares actual default-feature create/change/destroy
callbacks with an independent ordered census of
113 typed parameters and Source/Output. Its eight named cases cover acquisition,
actions, typed current getters/setters, UI edits, reentry and terminal cleanup.
`Ofx.Host.Capture` has eleven named cases and observes the existing safe
publication/preparation owner after actual callbacks. It reuses all 198 grain
capture and 36 edit witnesses with their original four-preset manifest and
approved overlay reader. `Ofx.Host.InstanceControls` has five named cases for
recoverable allocation, panic, failed-create retention, terminal cleanup and
capture interleavings. The latter two executables use only test-support Rust;
the default-feature instance executable retains its independent test oracle.

The additional `Ofx.Host.Actions`, `.Render` and `.RenderControls` groups exercise
real default-feature callbacks, remaining default-action argument/status traces,
ordered non-CUDA profile reads, callback-local image release and early-return
precedence. Controls link only test-support Rust. Images acquired before metadata
failure are released once, including the corrected image-construction failure path. Finite frame positions outside `[-2^63,2^63)`
are rejected with Fatal after the frame-rate getter, before native entry; earlier
no-op and failure results retain precedence. Nonfinite frame positions normalize
to zero only at frame assembly.

`Ofx.Gpu.Render` uses actual device images, the default-feature descriptor and the
independent test comparator. `Ofx.Gpu.RenderTrace` uses test-support controls for fixed
identities, temporal/publication/submission and completion/recovery observations.
It shares the existing instrumented processor failure hooks; original numerical
expectations and readers stay independent. Both own the real callback image/native
scope. No host-only fixture presents CPU storage as valid CUDA memory.

The twelve OFX host-fixture groups compile with CUDA but run without a device. They
and the two focused GPU groups exercise the Rust-owned descriptor and callbacks;
they do not install the candidate or establish installed-host or saved-project acceptance.
See the [ownership and maintenance map](../../CONTRIBUTING.md#ownership-and-maintenance-map)
for the production discovery shim and Rust/native responsibilities.

`RustBoundaryTests.test_ofx_callback_borrows` compiles seven real scoped/private/
image/thread/escape consumers, including actual instance access, typed parameter borrows
and terminal privacy, in default and test-support using the existing native
target/profile compiler-probe owner.

## Linux OFX host compatibility probe

`probe.py` exercises Juicer's OFX Load, Describe, and Unload actions with a
narrow synthetic host. Pass one or more `--preload` paths to reproduce symbol
resolution in a host that has already loaded those libraries globally.

The Linux host CTest group runs the synthetic lifecycle automatically. The
installed-Resolve-library form below is a host-qualification compatibility
check rather than an ordinary automated test. It passed for this migration on
2026-09-19; repeat it when qualifying a different Linux Resolve environment.

For DaVinci Resolve on Linux, run both forms against the trusted build output:

```sh
PYTHONDONTWRITEBYTECODE=1 python3.13 tests/ofx/probe.py \
  out/build/linux-release/juicer.ofx

PYTHONDONTWRITEBYTECODE=1 python3.13 tests/ofx/probe.py \
  out/build/linux-release/juicer.ofx \
  --preload /opt/resolve/libs/libProResRAW.so
```

The preloaded check protects the plug-in's C++ runtime boundary from host
libraries that expose incompatible libstdc++ implementation symbols. It is a
focused lifecycle and symbol-compatibility check, not Resolve render acceptance.

The probe intentionally stops before `DescribeInContext`, where Juicer first
needs its process root. A Debug build can therefore verify that unload remains
non-constructing by running the probe under GDB with a pending constructor
breakpoint:

```sh
gdb -q -batch \
  -ex 'set debuginfod enabled off' \
  -ex 'set breakpoint pending on' \
  -ex 'break JuicerProcess::Root::Root()' \
  -ex run --args python3.13 tests/ofx/probe.py \
  out/build/linux-debug/juicer.ofx
```

The expected result is normal inferior exit without the breakpoint being hit.

## CUDA render baselines

`Ofx.Gpu.ProcessorReference` checks the seven small procedural scenarios against
`processor_rust_pixels_{linux,windows}.dat`. These are Film-Juicer characterization
fixtures under the owner-accepted Rust `libm` 0.2.15 density policy, rather than
spektrafilm reference images. `rust_render_manifest.json` records the pinned
reference, exact source sample digest, scenario layout, native control, hashes,
and seed tuples. Independent preserved CUDA renderers generated these candidates
by replaying frozen accepted Rust density samples. Their native-value controls
first reproduced every original pixel and identity bit. The original
`processor_reference_pixels_*.dat` files remain historical evidence and are not
read by ordinary tests.

`Ofx.Gpu.AcceptedCudaCaptures` uses input/output pairs extracted directly from the
four previously accepted full-resolution clean CUDA renders, without recalculating
expected pixels. `accepted_cuda_clean_samples.json` retains original float bits,
source coordinates, full-frame geometry, settings, frame, color encoding, and
capture hashes. With all spatial and stochastic effects disabled, these pointwise
pairs can be packed into a 7×5 image while retaining the original full-frame region
of definition. Grain/glare captures cannot be compacted this way; their original
full-frame files remain acceptance evidence. The separate procedural fixtures
cover deterministic grain, glare, diffusion, halation, reconstruction modes,
positive/negative routes, RGBA, shifted origins, and signed zero.

Pixel comparisons retain `abs(actual-expected) <= 2e-4 + 3e-4*abs(expected)` and
explicitly reject non-finite actual or expected samples. No image-wide averaging,
percentile allowance, stochastic relaxation, or widened tolerance is used.
`Ofx.Host.RenderAssertions` proves the same assertions reject corrupted pixels,
channel swaps, NaN/infinity, a changed identity/seed lane, alpha, padding, and
execution bits. Density, gamma, profile, and channel leaf checks retain their
existing exact expectations.

`SequentialOwners`, `CutoverContract`, `PreparedBoundary.*`, and `CudaRender.*`
exercise lifecycle, contexts, ROI, stream, cancellation, alpha, padding, and
execution equivalence independently of the pixel fixture. Processor/direct/cold
C/warm C paths must agree bit-for-bit in boundary tests. A baseline failure cannot
prevent those separate CTest processes from running. The scatter/halation identity
rows separately require every recorded recipe, film-raw, print, and fixed glare
seed word in `parameter_rust_identities_{linux,windows}.json` to match exactly.

Ordinary tests only read checked-in fixtures. `--emit-reference` is an explicit
maintenance output aid, not an acceptance oracle: never adopt output from the
implementation under test alone. Generate candidates under `out/validation/`,
retain independent provenance, review them separately, and adopt an update only
when its behavior is approved. Changes affecting rendering use the owning
automated fixtures; documentation-only work follows the public validation cadence.

`Ofx.Gpu.FrameGeometry` uses `JuicerAdapterTraceProbe --frame-geometry
<frame-manifest> <preset> <trace-output> <halation-manifest>` with the existing GPU
lock and no elapsed deadline. Its three isolated observations are completed full-domain,
scalar-pixel and conditional effects-domain phases. The host independently records
image acquisition property reads and the live clip canonical query/time. Effects
off preserves acquisition reads and skips the conditional phase/query. Distinct
pixel RoD is observed before the existing prepared effects-domain rejection;
successful effects and weave cases honor their current full-domain rules.
Ordinary product objects contain none of the hook declarations or storage.

The separate bounded halation observation records invocation count, current f32
pixel size, boolean success and a copied optional descriptor. The accepted
three-phase geometry array stays unchanged. Four direct/print polarity routes
check inactive absence and every active descriptor field against the independently
captured native frame cohort. Diffusion failure precedes halation; a halation
pixel failure precedes the conditional effects query. Existing cached-image,
live-query and image-lease assertions still apply.
