# Benchmarks

Benchmarks are explicit measurements, not CTest pass/fail checks. The direct
scatter/halation command below needs its Release target prepared first;
`highlight-check` prepares its production check through the shared verifier. Record
the GPU model, driver, preset/commit and whether other GPU processes were active.

### Scanner replay and paired timing

Use the native Release preset with testing enabled. Build
`JuicerOfxSpectralReplayModule` and `JuicerHalationComparisonProbe` through
`scripts/verify.py prepare`. Keep the ordinary and Grain Preview configurations
and staged products in their prescribed order. The replay companion copies
the configured staged Resources and links the matching test-support archive.

```sh
python -B tests/benchmarks/ofx.py spectral-seed-pilot --preset linux-release \
  --inputs <symbol-and-layout-inputs.json> --run-id replay-negative-direct \
  --route negative-direct --grain current
```

The inputs bind both module hashes, their symbol identities, the compiled
breakpoint entries and pending-readback layouts. Replay uses natural tokens
from the staged product and reproduces them on the isolated companion. The
ordinary pair uses Current. Preview can exercise both Current and Film v1.

`ofx.py scanner-paired` takes a native Release preset, an input manifest, a
fresh run ID and one or more `--workloads`. Its check mode uses the same tail
rendezvous as measure mode. Measure mode requires the corresponding check
receipt and retains the raw A/A and reordered cubic/spectral timing samples,
with median and linearly interpolated p90 for each arm. A/A ratios are the
second cubic arm divided by the first. A/B ratios are spectral divided by cubic.
For a full-length diagnostic trace, use `--mode profile --nsys <executable>`
with the same unprofiled check receipt. Profile mode retains the same render
sequence and a CUDA trace per arm. The 5% timing gate applies in measure mode.
All modes require the pinned products, symbol layouts and qualified workload
bindings. The product driver leaves the final measured render's owner live
until the observer admits the single untimed reporter.

Use `--mode replay` for an untimed Ordinary same-method check. Each method
declares its matching isolated test-support `companion` in the input manifest.
Prepare the selected Ordinary Release modules with
`scripts/verify.py prepare --scanner-replay-links` and the explicit module
targets. In each `build_binding`, supply the producing `result.json` as
`producer` and its `candidate.json` and `candidate.patch` as `source` and
`patch`. Before rendering any workload, the runner checks the snapshot and
patch against the passing build bundle, then verifies the target's link output
digest and the selected Rust and native link inputs from that build.

The build bundle retains its generated Cargo role record and final Ninja file
inputs, including the archives, directly linked objects and explicit linker
controls. The replay receipt binds the sealed build result and those retained
members by digest. If the manifest also supplies `cargo`, its bytes must match
the retained record. Existing receipts keep their recorded evidence scope.
The replay runner verifies the linked binary, rejects shipping/companion file
and content aliases, and records the source, link inputs, module, symbols,
Resources, compiled layout, observer, reader and driver identities.
The product runs with a natural draw. The companion receives the three token
words read from that product and replays the same workload. The driver and
observer verify the exact completed output words, actual Current grain
parameters or grain-off dispatch, pitched source preservation, and terminal
readback in both runs. The materialized rows also compare the live C/M/Y words
within each method. Replay receipts carry their own scope and cannot admit
natural timing.

In paired runs, each arm's `schedule_points` maps its compiled scanner entry
names to the exact offset and symbol. The reporter must reach the entry for
the admitted route and schedule once. The breakpoint is armed after timing.

For a Full 36 mm Direct Preview comparison with the archived fixture, bind the
tightly packed RGB raster with `preview_material.path` and
`preview_material.sha256` in the paired manifest. Declare
`benchmark_scanner: true` only for an archived benchmark arm that exports the
backend selector. The driver leaves both arms' OFX tokens natural.

`--read-cubic-grid` reads the six buffer bindings and effective grid at the
reporter launch. For an archived module without native payload type information,
the matching-symbol manifest can provide each route's compiled `cubic_scan_stage`
offset under the arm's `layout`. The ordinary reader also requires the effective
grid to equal the authored grid. For the Full 36 mm Direct Preview row,
`--capture-density` saves the reporter's exact planar Float32 C/M/Y, and
`--held-density <raster>` stages and verifies those words on each arm's supplied
stream before its reporter reaches the real scanner kernel. The held comparison
requires `--nsys <executable>` and check mode. Read the CUDA kernel interval
from the retained SQLite trace for the reporter. The three preceding check
renders keep their natural inputs. Profiled receipts cannot admit measure mode.

For this row with correction, glare and gamut compression off, the spectral
reporter also saves `scanner-packet.f32` with the live spectra and the two
color matrices. To separate the arithmetic and check costs, prepare
`JuicerScannerCostBenchmark` with the native Release verifier and run
`ofx.py scanner-cost --preset <preset> --run-id <id> --density <raster> --packet <packet>`.
The benchmark interleaves both precisions, with and without intermediate
result checks, and with wavelength validity determined per pixel or once
before timing. The supported input is the held Full 36 mm raster. It records
per-launch CUDA event times and differences in the pre-encoding output.

### Scatter and halation

For the retained scatter/halation measurement on Linux:

```sh
out/build/linux-release/tests/bin/JuicerScatterHalationGpuTests \
  --case-group performance \
  --fixture-root tests/scatter_halation/fixtures \
  --resource-root out/build/linux-release/tests/Resources \
  --scratch-root out/validation/linux-release/scatter-performance-active \
  --performance-output out/validation/linux-release/scatter-performance-active/results.json \
  --performance-mode active \
  --device-index 0 --width 1920 --height 1080 \
  --warmup-count 10 --sample-count 30 --hold-milliseconds 0
```

Repeat with `--performance-mode inactive-baseline` and separate scratch/output
paths. Compare only runs with the same executable, dimensions, device, driver,
warmup/sample counts and external GPU load. The JSON result is diagnostic
evidence; no shared timing threshold is inferred from one machine.

For advanced halation, pair `reflection-3` and `reflection-5` at the same
dimensions. Both use the same absolute RGB widths, strengths and stable FP32 YVV
stage with scatter off. Use `boost-sparse` and `boost-dense` for the prepared
boost stage, including its status event and separate CPU wait and preparation
samples. The useful device and pinned bytes in these reports describe the
admitted optical stage. The stage timing excludes test source upload and
destinations and is separate from the OFX callback timing. Retain the cold
stage sample and three alternating pairs of 10 warmups and 30 measured runs
at 1920×1080 and 3840×2160.

On Windows, use the matching
`out\build\windows-clang-release\tests\bin\JuicerScatterHalationGpuTests.exe`
and Windows paths with the same arguments.

### Film grain reference prototype

The nonshipping prototype compares the Film reference sampler with Current grain.
Prepare it on `linux-release` or `windows-clang-release`. The ordinary CTest
registrations exercise synthetic fixtures using the independent double-precision
oracle.

```sh
python -B scripts/verify.py prepare --preset linux-release --testing ON --targets JuicerGrainPrototype
python -B scripts/verify.py development --preset linux-release --runtime all \
  --tests Grain.Host.FilmReferenceOracle Grain.Gpu.FilmReference --run-id grain-reference-cases
```

Explicit `python -B tests/benchmarks/grain.py` operations take a native
`--preset` and a fresh `--run-id`. They write original results under
`out/validation/<preset>/results/<run-id>/`.

1. Run `response` to export completed current CDF3 knots in `response.log`.
   It defaults to Portra 400. Use `--stock kodak_gold_200` or
   `--stock fujifilm_provia_100f` for G2.
   In a temporary copy of the [pinned Fotufilm package](../grain/fotufilm-notice.md),
   place [capture_film_probe.swift](../grain/capture_film_probe.swift) in
   `Sources/FotufilmCore/` and [capture_film_reference.swift](../grain/capture_film_reference.swift)
   in `Sources/graincapture/main.swift`. Add the `graincapture` executable
   target with the `FotufilmCore` dependency to the temporary package manifest.
   Build with Swift 6.1.2 using
   `FOTUFILM_DISABLE_HALIDE=1 swift build -c release --product graincapture`.
   Run `graincapture gold <response.log> <empty-output-directory>` or
   `graincapture provia <response.log> <empty-output-directory>`. Keep the G1
   Portra bank and its original capture inputs. Its original two-argument
   command remains available. Preserve all bank and probe
   hashes. Stock-derived banks belong in local research output.
2. Run `reference --bank <directory>` to compare CUDA, the independent oracle
   and the pinned Swift outputs on the frozen bank.
3. Run `capture --input <new-file>` once to write the immutable
   production-developed C/M/Y planes. Pass the same file and bank to
   `benchmark --bank <directory> --input <file>` in both native presets.

The single workload is 3840×2160 Portra 400 at integer frame zero, with a 36 mm
long edge and DIR inactive. Current uses the OFX defaults with Add grain enabled.
Film uses its reference controls. Each arm records 10 warmups and 30 samples
in each of three alternating pairs. Timing includes Current's filters and Film's
density write. Cold applications, setup, input reset and upload are reported
separately. The script retains the raw samples and reports per-platform medians
and nearest-rank p90. These describe the unmatched G1 stage workload.

#### G2 calibration and stage quality

For independent G2 agreement, use `reference-g2 --bank <canonical-Portra-bank>
--tables <table-directory> --probes <reference-g2.tsv>`. In the Swift adapter,
`FROZEN_FILM_BANK` selects an existing SAT and captures signed-coordinate
probes for each extra footprint argument. Use the explicit `portra` stock,
its original CDF3 log and `0.25 0.78125 3.17 37.5` for the maintained G2
reference comparison. The
footprint tables contain density means and adjacent-level correlations for the
same bank and can be captured without preparing another emulsion. Copy the
canonical `records.f32` alongside any separate table directory and retain its
identity.

Run the explicit research operations with NumPy and SciPy installed. `freeze`
records the complete stock paths and vectors, frame split, geometry, looks and
size grid. It expects the stock banks and footprint tables in the experiment
directory and the completed CDF3 response logs in their declared result paths.

```sh
experiment=out/validation/grain-g2-2026-11-16/experiment.json
python -B tests/benchmarks/grain.py freeze --experiment "$experiment"
python -B tests/benchmarks/grain.py calibrate --experiment "$experiment" \
  --preset linux-release --run-id grain-g2-calibration
```

`calibrate` renders full frames and retains the central 1024-square crop plus
the aperture halo. A match requires all three per-record transmission-aperture
RMS residuals within 2% and the pooled raw-density correlation length within
5%, in both the calibration and held-out sets. The capture manifests record
resolved per-frame controls and seeds. The separate half-frame residuals have
their own disposition. `research --input <request.json>` captures an explicit
frame sequence via the same native owner. `grain_quality.py` reads completed
captures offline and adds density, spatial, temporal and preview observations.

```sh
quality=out/validation/grain-g2-2026-11-16/quality-requests.json
python -B tests/benchmarks/grain.py freeze --experiment "$experiment" --input "$quality"
python -B tests/benchmarks/grain.py quality --experiment "$experiment" --input "$quality" \
  --preset linux-release --run-id grain-g2-quality
python -B -m unittest discover -s tests/benchmarks -p test_grain_analysis.py
```

The offline reader also requires Pillow. Use `grain_quality.py patches OUTPUT
CAPTURE...` for density atlases, `grain_quality.py scale OUTPUT CAPTURE...` for
one engine's physical-pitch series, and `grain_quality.py OUTPUT CAPTURE...`
for flat fields and animation. Add `--preview --ffmpeg <encoder>` to the
last form for a randomized two-engine developed-density still and clip. The
clip uses lossless H.264 RGB and verifies the decoded pixel hash. Use
`--preview-only` to repeat the presentation from existing captures.

The reduced-chroma G2 variant takes `chroma_fraction` in a Film research
request. It preserves the scalar preview luminance while scaling the
same-luminance chromatic residual in transmission. Omit the field for the
original full-chroma reference. For reduced-chroma `calibrate`, set
`chroma_fraction` in the experiment to `0.2406350076198578` for the G2 comparison.
Set `reference_calibration` to identify the original full-chroma native result
path and its SHA-256. The search uses
those original calibration fields and recomputes the matched statistics from
fresh reduced-chroma CUDA frames. It freezes the same controls for the
disjoint integer and half-frame checks.

#### G3 qualified grain comparison

Prepare the nonshipping `JuicerGrainPrototype` and `JuicerOfxMeasurement`
targets with the maintained verifier on each native Release preset. The
ordinary `Grain.Host.FilmReferenceOracle` and `Grain.Gpu.FilmReference`
registrations remain correctness checks. Explicit G3 operations take a frozen
reduced-chroma experiment and completed Current and Film G2 capture receipts.
Pass the repository-local paths to the retained inputs. Historical receipt
paths are resolved in the checkout and their original bytes are preserved.

```sh
python -B tests/benchmarks/grain.py g3-inputs --preset linux-release --run-id g3-inputs \
  --experiment "$experiment" --current-receipt "$current_receipt" --film-receipt "$film_receipt" \
  --input "$manifest"
python -B tests/benchmarks/grain.py g3-check --preset linux-release --run-id g3-linux-check --input "$manifest"
python -B tests/benchmarks/grain.py g3 --preset linux-release --run-id g3-linux-stage \
  --input "$manifest" --check-result "$linux_check_result"
```

Use the same manifest for Windows `g3-check` and `g3` with
`--preset windows-clang-release` and fresh run IDs. The input operation
captures planar CMY from production development for each size, stock and route.
The static camera-linear pattern gives identical image membership between the
two engines and in both time modes. The varied mode changes frame and seed
identities. The check operation compares the measured dispatch at selected G2
integer and half frames and replays a signed frame sequence in both orders.

Each native decision row uses three separate A/B process pairs in alternating
order with 10 warmups and 30 measured requests per arm. Both repeatability
factors must fall below 1.05 for the single paired Film block to run. The reader
applies linearly interpolated median and p90 envelopes per row. Grain-off
controls use unchanged integer and half frames in all four Portra size/route
combinations. Timing, cold setup and output checks keep separate boundaries.
An edit observation reads a frozen alternate table while the SAT remains
resident. It measures existing table read and binding costs.

Measure the Current shipping routes with the qualified Release OFF module and
its module-relative resources. The G3 route runner infers the native driver from
the preset build and verifies the production closure. Use `--workload` with
`g3-hd-portra-direct-default-on-varied-half` for a focused check, or use
`--g3-manifest` for the complete on/off route inventory. The measure mode
records one cold render, ten warmups and thirty measurements per process.

```sh
python -B tests/benchmarks/ofx.py g3-route --preset linux-release --run-id g3-linux-routes \
  --candidate-production-result "$production_result" --module "$module" --resources "$resources" \
  --g3-manifest "$manifest" --mode measure
python -B tests/benchmarks/grain.py g3-report --input "$manifest" --output "$report" \
  --results "$linux_stage_result" "$windows_stage_result" \
  --route-results "$linux_route_result" "$windows_route_result"
```

Run GPU operations sequentially through the shared device owner. The offline
report preserves missing and invalid rows alongside measurements. It gives
Current callback-stream event time and the isolated stage interval separately.
Where both are available, the reported estimate adds median route-off event
time to each Film arm's median stage event time on the same workload. Both
intervals can include host submission gaps. The sum estimates serial
stream elapsed time under those observed conditions. The inputs, residency
and runtime identities differ between the route and stage fixtures, and
integration staging and synchronization are unmeasured.

#### Output-preserving grain experiments

The explicit Film selectors are `--film-variant reference|scalar|reuse|paired|quad` and
`--film-block 32x4|16x8|32x8`. Existing G1/G2 and G3 commands default to the
original reference at 32x4. The scalar variant uses fixed record indices and
scalar carrier inputs. Reuse additionally shares identical laying geometry
and per-record level preparation across temporal samples. Both retain the
accepted sampling, normalization and chroma operations.

`g3-verify-opt` checks all eleven workload labels across eighty frame identities
and their reverse replay, comparing full CMY words outside timing. It retains
all mismatch counts and the first differing coordinate. A completed agreement
for the exact executable, manifest, variant and launch admits `g3-screen`.
The screen uses three alternating reference/optimized pairs for unchanged Gold
half frames and the fixed seven-row control set defined in
`grain_optimization.py`. It reports descriptive per-pair ratios and control
times. Each invocation retains ten warmups and thirty measured requests.
New screens declare per-invocation device observations and retain query and
activity logs before and after each arm. Readback validates their membership
and hashes. Earlier receipts retain their original telemetry scope.

```sh
python -B tests/benchmarks/grain.py g3-verify-opt --preset linux-release \
  --run-id grain-opt-agreement --input "$manifest" --film-variant scalar
python -B tests/benchmarks/grain.py g3-screen --preset linux-release \
  --run-id grain-opt-screen --input "$manifest" --film-variant scalar \
  --check-result "$agreement_result"
```

`reference-scalar` and `reference-reuse` add bitwise optimized comparisons to
the original G1 probe reader. `reference-g2-scalar` and `reference-g2-reuse`
do the same for the G2 probe reader. They take the original commands' bank,
table and probe operands. The ordinary host/GPU registrations additionally
compare both variants on the synthetic carriers and all three launch shapes.

`g3-profile` takes an explicit native Nsight Compute executable through
`--profiler`. It captures three grain kernels after ten warmups for five fixed
workloads, with uncontrolled caches and clocks declared in the result. Profile
replays and their stage samples retain their attribution scope. The runner
records raw Nsight reports, kernel/launch membership and selected metrics.
CSV readback uses the native locale encoding. If postprocessing fails after
successful captures, `--profile-result <failed-result.json>` reimports those
closed captures and collects only missing fixed cases. It requires matching
candidate/tool/settings and successful drained commands, preserves the failed
receipt and records hashes at readback for reused artifacts.

```sh
python -B tests/benchmarks/grain.py g3-profile --preset windows-clang-release \
  --run-id grain-opt-profile --input "$manifest" --film-variant scalar \
  --profiler "$ncu"
python -B tests/benchmarks/grain.py g3-opt-report --input "$manifest" \
  --results "$reference_profile_result" "$agreement_result" "$screen_result" \
  --profile-reader "$nsight_extras_python" --output "$optimization_report"
```

The offline optimization report revalidates profile and screen membership and
agreement hashes, linking the original evidence. It describes the bounded
experiment before decision timing and carries the planned decision inventory
as unrun. A subsequent complete G3 comparison uses `g3` and `g3-report`.
The focused offline admission tests run with
`python -B -m unittest discover -s tests/benchmarks -p test_grain_optimization.py`.
For SAT attribution, `--profile-set sat --profile-reader <Nsight-extras-python>`
adds memory tables, imported source, source-correlated requests/stalls and an
explicit device-verified counter set. The result distinguishes per-operation
tag-stage wavefronts from combined local/global data-pipe wavefronts. It retains
invalid ancillary counters with unusable derived ratios, while invalid required
read counters fail attribution. Imported sampler text is hashed with UTF-8/LF
normalization and checked against frozen source. Closed-capture continuation
follows hashed receipt chains and never repeats a captured case.

Incremental screens can select `--baseline-variant scalar --baseline-block 32x8`.
Pass these selectors to `g3-verify-opt` too: the resident driver compares the
baseline and candidate directly to the original on every frame in the same
agreement campaign. The exact-binary agreement must cover that baseline before
the screen is admitted. The `paired` variant overlaps adjacent-level SAT reads
while preserving each level's original arithmetic. Its explicit pinned modes
are `reference-paired` and `reference-g2-paired`. Historical defaults remain
reference/32x4, with the original profiling section set.

The separate `quad` variant copies each 2x2 SAT cell's four original Float32
words into a 16-byte aligned entry. It retains scalar arithmetic and requires
53,477,376 device bytes instead of the canonical bank's 13,473,996. Timing binds
only the selected representation; direct verification holds both and reports
66,951,372 combined bytes. Cold preparation reports the host packing temporary
and packing time. `reference-quad` and `reference-g2-quad` run the original pinned
probe readers. `verify-quads --bank <directory>` exhaustively compares all
13,369,344 uploaded component words against that bank and retains a JSON report
under the ordinary native supervisor and GPU lease. This explicit maintenance
operation modifies no fixture.

For optimized G3 timing, `--check-result` identifies the matching `g3-check`
run, and `--agreement-result` identifies full-frame `g3-verify-opt` agreement.
Pass the selected variant and launch to both verification and timing. The
fixed 1.05 median/p90 decision rule applies to the complete qualified matrix.

### Production highlight boost check

`scatter.py highlight-check` builds and runs the actual production highlight
boost path against the frozen boost fixtures and 495 named supplemental cases
(509 cases total). It checks the production implementation only. The per-pixel
curve is stable FP32; FP64 remains in once-per-frame threshold and peak setup.
The standalone direct/stable variant probe is retired, and the old
`highlight-run` bundles remain historical comparison evidence.

The 509-case corpus is sampled evidence, not an exhaustive FP32-domain proof.
It contains no nonzero subnormal carrier inputs or disabled-with-invalid-carrier
witness. The retained maximum observed error of five ULPs is on a nonpeak sample
admitted by the separate relative-error alternative; the peak gate remains one
ULP. Inspect the per-row gates rather than interpreting a collector's successful
execution as a stronger accuracy claim.

Run a native preset with a fresh result ID:

```sh
python -B tests/benchmarks/scatter.py highlight-check \
  --preset linux-release --run-id highlight-production-linux-release
```

The command accepts any supported native preset: `linux-debug`, `linux-release`,
`windows-clang-debug`, or `windows-clang-release`. The offline
`highlight-collect` operation reads both the historical three-variant bundles
and current production-check results without rerunning CUDA:

```sh
python -B tests/benchmarks/scatter.py highlight-collect \
  --bundle out/validation/linux-release/results/highlight-production-linux-release
```

Use the maintained OFX campaign below for whole-render performance. The old
isolated boost-stage measurements do not predict a full callback speedup.

### Matched blur precision characterization

`scatter.py collect` validates retained three-way precision experiments from
before the stable FP32 production conversion. Those bundles contain their original
executable, source, fixture identities, errors and raw timing samples. The live
experimental precision switch and its runner are retired; current spatial and
timing tests use the ordinary production launch.

```sh
python -B tests/benchmarks/scatter.py collect \
  --bundle out/validation/linux-release/results/blur-precision-comparison
```

The collector verifies complete command and fixture membership, retained hashes,
ordinary-launch equality, compact/padded identity, and reported percentiles against
the raw timing arrays. The large-gain exception retains both independent ULP
checks and relative error reporting. A completed characterization is distinct from
a passing FP32 variant; individual gate failures remain explicit.

Retained timing uses the same three-bounce schedule and source at 1080p and UHD, with three
alternating variant orders, ten warmups and thirty samples each. CUDA event and
host submit/wait timings exclude source reset and include per-Gaussian host factor
construction and any resulting launch gaps. Reports retain cold samples, kernel
register counts and common frame storage. They are stage measurements, not OFX
callback or full application timings. That experiment derived factors per launch
on the CPU. Production prepares FP32 factors in Rust before native execution.

The OFX performance campaign uses `ofx.py` on each native Release platform.
Prepare the driver and companion with `scripts/verify.py prepare --testing ON`
and the explicit targets `JuicerOfxMeasurement JuicerOfxMeasurementModule`.
`prepare-reference` requires the pinned source revision, prescribed isolated
source/build directories, and the recorded observation patch. It does not fetch
refs. The `check` operation must pass before `measure`; both take `--preset`,
`--inputs` and `--run-id`. The input JSON has this shape (replace every path and
hash with its actual value):

```json
{
  "driver": {"path": "...", "sha256": "..."},
  "overlay": {"path": "...", "sha256": "..."},
  "reference": {
    "shipping": {"path": "...", "sha256": "..."},
    "observed": {"path": "...", "sha256": "..."},
    "resources": {"path": "...", "files": {"relative/path": "sha256"}},
    "observed_resources": {"path": "...", "files": {"relative/path": "sha256"}}
  },
  "candidate": {
    "shipping": {"path": "...", "sha256": "..."},
    "observed": {"path": "...", "sha256": "..."},
    "resources": {"path": "...", "files": {"relative/path": "sha256"}},
    "observed_resources": {"path": "...", "files": {"relative/path": "sha256"}}
  },
  "compatibility_check": "path/to/completed/check"
}
```

The compatibility path is required by `measure` and identifies a passed check
with identical inputs. Enumerate every regular resource file using relative POSIX
paths and SHA-256; linked files are rejected. Each Resources path must match its
module-relative `Resources` directory, and shipping and observed inventories
for each side must agree. After a drained failed reference configure or build,
use a new run ID and `--resume-from <failed-run-directory>` to reuse only the
unchanged exported source and overlay. Campaigns serialize against the
shared GPU and retain child logs and raw records. The `report --runs <linux>
<windows> --output <file>` operation reads completed evidence offline. Process
memory includes fixture and driver overhead. Observed upload-pool staging excludes
scan-error pinned memory; its drainage says nothing about that separate family.

Each measured process performs one cold render, ten warmups, thirty unchanged
measurements, three gamma edit/render/reuse episodes and two further owner cycles
(one cold, ten warmups and an endpoint each), then normal Unload. Three matched
pairs alternate reference/candidate, candidate/reference, reference/candidate for
each workload and lane. Cold is process/owner cold; storage caches are not flushed.
The four workloads are negative-direct and negative-print, each with grain off/on.
Construction uses paired spans plus successful Render, output and lifecycle checks;
a failed callback leaves its construction outcome unknown. OFX parameter edits may
return OK or the supported default reply; raw statuses and gamma readback remain
recorded. Upload-pool state arrays are available, reserved, in-flight, quarantined.

By default the candidate is the frozen S7.B shipping module. To measure a newly
qualified candidate, pass `--candidate-production-result <result.json>` to both
`check` and `measure`, naming the completed native Release OFF
`verify.py prepare --production-contract` result for that preset. The runner
requires its production module hash to match the supplied candidate, validates
its OFF configuration and exports, and binds its consumed inputs and source
snapshot at campaign entry and exit. It copies the closure result, contract and
source evidence into the run. The compatibility check and measurement must use
the same qualification. The pinned S2.E reference remains unchanged.

Halation callback measurements use `--reference-production-result` for the
clean `31a4d5d` Release OFF closure and `--candidate-production-result` for the
candidate closure. Choose `halation-1080p-direct`, `halation-1080p-print`,
`halation-1080p-direct-spatial`, `halation-1080p-print-spatial`, and the
corresponding `halation-uhd-*` rows in that order. Add the `-boost` and
`-composed-boost` variants for matched above-onset exposures at each size.
Both sides receive the same manual exposure. The composed rows also enable
camera diffusion and spatial halation on both sides. In this standard mode, the
candidate receives four stops of highlight boost with zero protection EV in
boosted rows; the pinned legacy reference has no boost control.
The paired control census
checks the common photographic settings. The candidate also verifies that the
new controls start in the legacy-compatible mode with zero boost. The
result retains both production proofs. Use the clean reference's empty
`candidate.patch` for the input `overlay` identity when its checked-in
observation target supplies the reference companion.

Focused follow-ups may pass --workloads with a nonempty subset in the documented
workload order. Check and measure require identical selections. The offline
reader validates complete pair membership for the explicit selection and requires
matching workloads across native platforms. Omitting the option selects all four
workloads and preserves the original campaign protocol.

#### Matched FP64/FP32 highlight callbacks

Use `--comparison matched-highlight` for the whole-render comparison between the
retained FP64 boost module and the stable FP32 candidate. Supply the prior
`halation-stable-fp32-production` result as `--reference-production-result`, the
new Release OFF production result as `--candidate-production-result`, and add
this field to the input JSON using the retained baseline manifest:

```json
"matched_highlight_baseline": {
  "path": "out/validation/halation-feature-parity/stable-fp32-highlight/baseline-artifacts.json",
  "sha256": "<sha256 of baseline-artifacts.json>"
}
```

The manifest binds the baseline shipping module, its observed measurement
module, and their Resources inventory for each native preset. The run retains a
copy of the manifest and the observed-module build receipt and source proof. The
matched mode requires advanced boost controls on both modules and verifies
4 EV boost with 0 EV protection on all eight above-onset rows. Composed rows
enable camera diffusion and spatial halation on both modules. Two direct rows at
1080p and UHD use the same 6 EV exposure with boost set to zero as controls.

The exact ordered workload selection is:

```text
halation-1080p-direct-boost halation-1080p-print-boost
halation-1080p-direct-composed-boost halation-1080p-print-composed-boost
halation-uhd-direct-boost halation-uhd-print-boost
halation-uhd-direct-composed-boost halation-uhd-print-composed-boost
halation-1080p-direct-boost-zero halation-uhd-direct-boost-zero
```

Pass the same `--comparison`, production result paths, inputs and workload list
to `check` and `measure`. The compatibility check must pass before measurement.
The offline report requires this mode and all ten rows on both platforms. Shipping
lane timings include the complete OFX callback and Rust frame preparation;
CUDA stream intervals include host gaps and are reported separately. These are
measurements for the recorded devices and runs, not a general product speedup.

Both sides of this comparison already use stable FP32 blur. Its timing delta
isolates the highlight conversion; historical blur-stage timing does not establish
the final Rust-prepared blur's whole-render cost. Assess first-process cold renders,
later owner cycles and parameter edits separately from warm samples. The retained
Linux runs contain first-process latency spikes, including an unchanged follow-up.
Subsequent tracing localizes the reproduced large delay to resource-file operations
before the first traced in-callback CUDA API, while highlight kernel execution is
faster. The underlying OS/filesystem cause remains unconfirmed; a warm-render
benefit does not establish neutral startup latency.

The current driver records output_guard_version 2. It rejects RGB channels left
at the finite destination sentinel and runs positive and rejection witnesses for
RGB and RGBA before each process. Required alpha and row-padding checks retain
their existing exact comparisons. Historical version-1 output_valid records have
the earlier finite-only RGB scope.
