# OFX host fixtures

For cleanup, use the [contributor checks](../../CONTRIBUTING.md#required-checks).
Verifier launchers, receipt collection and replay commands below describe
the tagged OFX evidence. Fixture inputs, callback assertions and host
lifetime requirements retain their documented scope.

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

The regular product witnesses check the established module identity, descriptor,
constructor and render callbacks. `Ofx.Host.Context` checks the Film v1
declarations and the Current default. `Ofx.Host.FilmV1Instance` checks the four
grain parameter handles and their types, the initial visibility, and cleanup.
`Ofx.Host.Render` checks the Current model read alongside the existing status,
ordering and release expectations. `Ffi.Gpu.FilmV1Staged` exercises the staged
normal bundle on CUDA. The platform-owned `Grain.Gpu.FilmOfxApertures` adds
the linked aperture and motion checks.

Retained preview and paired-discovery receipts are readable offline with
`scripts/qualification.py --bundle <receipt-directory>`.

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

### Supplied production halation comparison

`compare.py` compares two explicitly identified production modules through their
ordinary OFX exports. Prepare `JuicerHalationComparisonProbe` with
`scripts/verify.py prepare --preset <native-release> --testing OFF --targets
JuicerHalationComparisonProbe --run-id <name>`. This target links no Rust archive
and has no product build or staging dependency. It shares the production probe's
host and image/lifetime checks. Its explicit `--halation-capture` mode does not
read or update pixel fixtures; ordinary production-contract fixture mode remains
unchanged.

Run with the platform's maintained test environment (NumPy from
`tests/requirements.txt`):

```sh
python -B tests/ofx/compare.py run --run-id halation-shipping-comparison \
  --reference /absolute/rollback.bundle/Contents/ARCH/juicer.ofx \
  --reference-sha256 SHA256 \
  --candidate /absolute/candidate.bundle/Contents/ARCH/juicer.ofx \
  --candidate-sha256 SHA256
python -B tests/ofx/compare.py collect \
  out/validation/NATIVE-RELEASE/results/halation-shipping-comparison
```

Use the qualified native build environment for `run`. The runner holds the
existing native build and shared GPU leases, incrementally refreshes only the
comparison probe, supervises one process per module without an elapsed deadline,
and retains original process
receipts, harness source, whole bundle inventories, authored OFX values and
logical little-endian FP32 RGBA input/output files. It checks the supplied hashes
and equal bundled resources before use and unchanged bundles after use. Module
production provenance belongs to the supplied artifacts' existing build records;
this tool neither rebuilds nor installs them. A fresh harness snapshot does not
replace their production source identity.

The fixed matrix uses 1920×1080 full frames, render scale/PAR 1, 35 mm, time 37,
Hanatos, Rec. 2020, and active legacy halation with all four sliders at one.
Portra 400 and Provia 100F each cover direct/print, AE off/on, cold/warm: eight
scenarios, sixteen renders per module. Candidate advanced controls are untouched
descriptor defaults (Profile, three bounces, decay 0.5, renormalize, zero boost);
the comparator requires them and compares all common authored parameter values.
Other controls use `production_images.cpp`'s explicit settings, including grain
and diffusion off, scanner unsharp off and print glare on. Input uses its existing
deterministic modulo-19 RGB pattern and varying alpha, with padded rows.

Any FP32 bit change fails; no photographic tolerance is applied. Each module
also requires cold/warm identity, finite output, alpha/source/padding preservation,
one release per acquired image, context preservation and complete teardown.
The offline command rechecks membership, retained hashes and all pixel comparisons.
No saved project, proxy selection, source decoder or Resolve export is exercised.
These synthetic results do not replace a controlled repeat-export host comparison.

Comparator negative tests run without a GPU via the maintained verifier:
`scripts/verify.py python-tests --preset <native-release> --key ofx-comparison --
<same-python> -m unittest discover -s tests/ofx -p test_compare.py -v`.

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

## Actual-host boundary capture and replay

`capture.py` and `capture-agent.js` observe one Windows OFX render in an unchanged
supplied module. This optional diagnostic uses Frida (17.26.3 was exercised) in a
separate capture environment; it is not an ordinary CI prerequisite. Start the
observer before plug-in discovery and instance creation. It records the exact
module/bundle identities, typed current-value snapshots, actual getter history,
render and image geometry, and tightly packed little-endian FP32 source/output
rows in ascending logical y. Binary64 parameter words are retained separately
from JSON numbers. CUDA copies occur within the image lease and synchronize the
actual host stream; this changes timing and requires an unchanged paired export
as an observation witness.

Run `capture.py run --help` for the supplied executable/module/bundle and result
arguments. The tool holds the shared GPU lease until the observed process exits.
Write `control.json` in its result directory with `{"action":"arm","width":6048,
"height":4032}` after the disposable project is ready; optional `time` and
`effect` keys narrow the render. `{"action":"stop"}` unloads the observer without
closing the application. Partial chunks and failed attempts remain retained.
Never edit observer inputs while a session is live, or claim success from the
process exit alone: inspect the final manifest, complete capture and image hashes.

`resolve_capture.lua` is a companion for the installed Resolve scripting API.
It exports a named original without saving it and imports a separately named
copy. Its `copy`, `load`, `inspect`, `export` and `close` actions use
`JUICER_RESOLVE_CAPTURE_ACTION`, `JUICER_RESOLVE_CAPTURE_DIR`,
`JUICER_RESOLVE_ORIGINAL_PROJECT` and `JUICER_RESOLVE_CAPTURE_PROJECT`; all edits
and exports assert the copy's name. First-frame selection currently requires a
24 fps timeline and the second video-track item. Close uses the API's explicit
close-without-saving operation before quitting Resolve.

`compare.py run --frame <capture-directory>` extends the supplied-binary Release
OFF comparison with an actual frame. It uses the standalone
`JuicerHalationComparisonProbe --frame-replay` mode, restores recorded controls
and geometry, and retains cold/warm FP32 output from both exact modules. The
diagnostic result distinguishes successful execution from pixel equality and
compares the rollback replay with the actual host output first. It reports
snapshot/latest-read disagreements; arbitrary reentrant parameter history and
production session/clip-pointer seeds are not recreated. A host/replay mismatch
therefore prevents a claim that the captured state was fully reproduced. Whole
output-frame comparison currently requires a full output render window. Neither
this tool nor the observer changes installed bundles or checked-in fixtures.

For explicit diagnostic counterfactuals, add `--parameter-overrides <json>` to
the comparator. The dictionary maps captured names to full typed parameter
objects, including binary64 words. The replay records original/overridden values
and names absent from the legacy module; its receipt hashes the override file.
This does not mutate or relabel the original host capture. A counterfactual
measurement is distinct from host observation or fixture acceptance.

Use `--scanner-frame` with a frozen RGB or RGBA frame manifest. RGBA
comparisons verify source alpha. For a manifest with `grain_replay` and
`variant: "ordinary"`, supply `--current-scanner-observation` with the
matching symbol and layout declarations for the isolated companions. Both
arms must consume the frozen Current token and payload words.

Use `--print-scanner-observation` for an `observe_print` manifest. The
receipt records live developed paper C/M/Y, correction controls and the
pre-white glare field. It distinguishes glare-off, matched-inputs and
natural-pattern observations. Constant mode uses 1% with zero roughness and
blur. A separate same-instance reporter consumes the last compared draw's
deferred flag. The cold and warm photographs retain their original time.

The complete `matched-field` claim additionally requires the live pre-glare
XYZ capture for both compared draws in each arm, the field consumed by those
draws and matching scanner correction and color inputs. Offline `collect`
recomputes the complete verdict from its retained draw records and rasters.
It reads earlier constant-glare receipts without XYZ as `matched-inputs`.
