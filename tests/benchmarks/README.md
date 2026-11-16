# Benchmarks

Benchmarks are explicit measurements, not CTest pass/fail checks. The direct
scatter/halation command below needs its Release target prepared first;
`highlight-check` prepares its production check through the shared verifier. Record
the GPU model, driver, preset/commit and whether other GPU processes were active.

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

The nonshipping G1 target compares the Film reference sampler with Current grain.
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

1. Run `response` to export the Portra 400 CDF3 response in `response.log`.
   In a temporary copy of the [pinned Fotufilm package](../grain/fotufilm-notice.md),
   place [capture_film_probe.swift](../grain/capture_film_probe.swift) in
   `Sources/FotufilmCore/` and [capture_film_reference.swift](../grain/capture_film_reference.swift)
   in `Sources/graincapture/main.swift`. Add the `graincapture` executable
   target with the `FotufilmCore` dependency to the temporary package manifest.
   Build with Swift 6.1.2 using
   `FOTUFILM_DISABLE_HALIDE=1 swift build -c release --product graincapture`.
   Run the executable with the response log and an empty output directory.
   Preserve the bank and probe file hashes. The bank uses synthetic calibration
   and belongs in local research output.
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
