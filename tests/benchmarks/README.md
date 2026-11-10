# Benchmarks

Benchmarks are explicit measurements, not CTest pass/fail checks. Build the
Release test target first, then record the GPU model, driver, preset/commit and
whether other GPU processes were active.

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

On Windows, use the matching
`out\build\windows-clang-release\tests\bin\JuicerScatterHalationGpuTests.exe`
and Windows paths with the same arguments.

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

Focused follow-ups may pass --workloads with a nonempty subset in the documented
workload order. Check and measure require identical selections. The offline
reader validates complete pair membership for the explicit selection and requires
matching workloads across native platforms. Omitting the option selects all four
workloads and preserves the original campaign protocol.

The current driver records output_guard_version 2. It rejects RGB channels left
at the finite destination sentinel and runs positive and rejection witnesses for
RGB and RGBA before each process. Required alpha and row-padding checks retain
their existing exact comparisons. Historical version-1 output_valid records have
the earlier finite-only RGB scope.
