# Gamma validation

This opt-in CUDA executable and runner preserve the pinned film/print gamma
reference, unchanged Film-Juicer baselines, and candidate comparisons. It uses
production route builders, descriptors, preparation, payload packers, and CUDA
stage launches. It is not a CPU renderer.

The reference environment must be a normal installation of the revision
recorded in `fixtures/manifest.json`. Reference generation refuses an existing
fixture root:

```text
python run_validation.py reference --reference-root PATH --fixture-root PATH
```

An unchanged-production capture stages a fresh executable-relative runtime and
refuses an existing output directory:

```text
python run_validation.py baseline --exe PATH --resource-root PATH --fixture-root PATH --output-root PATH --case-group default-paper
```

`--case-group` is one of `default-paper`, `default-routes`, `sampling-dir`,
`print-backend`, `state`, `routes`, `lifetime`, `performance`, or `all`. The
`all` baseline runs `default-paper` twice and requires byte-identical numerical
captures before running every other group sequentially.

After both optimized `all` captures have been reviewed, the separate promotion
operation freezes their captures, per-case limits, executable identities, and
environment record. It refuses any existing baseline or partial-promotion
destination; there is no overwrite switch:

```text
python run_validation.py promote-baseline --fixture-root PATH --clang-output-root PATH --release-output-root PATH --environment-record PATH
```

Candidate comparison reads and verifies `fixtures/baseline.json`, stages a new
runtime, and writes only to a fresh output root. Before either an ordinary
candidate comparison or candidate route probe can launch, the runner requires
the source resource tree, staged resource tree, file sizes, and file hashes to
match the inventory frozen by the applicable baseline reports. It separately
records the new source and staged executable identities and rechecks them after
the process exits:

```text
python run_validation.py compare --exe PATH --resource-root PATH --fixture-root PATH --output-root PATH --case-group sampling-dir
```

Image-bearing comparisons also write hashed binary PPM files for the actual
output, pinned expected output, and a fixed 16x absolute RGB difference. Their
paths, shape, encoding, and scale are recorded under `comparison_images` in the
report; these review images supplement the unquantized per-channel metrics and
raw density/output captures.

An `all` candidate comparison refuses to run until the separate Phase 3A route
probe has been frozen. After the real film-gamma recipe consumer lands, capture
the independent route probe in each optimized configuration. The runner builds
its transient input only from checksum-verified Phase 1 manifest data and
pinned print-backend tables. The executable replaces only a local launch
payload's print curve views after ordinary packing; it leaves production
preparation, stock bounds, balance/preflash, scanner correction, and resource
identities untouched:

```text
python run_validation.py route-probe --exe PATH --resource-root PATH --fixture-root PATH --output-root PATH
```

Review both fresh records, then freeze them with the separate no-overwrite
promotion operation. This adds `fixtures/route-probe/` and
`fixtures/route-probe.json`; it verifies but never rewrites the Phase 1
manifest or baseline:

```text
python run_validation.py promote-route-probe --fixture-root PATH --clang-output-root PATH --release-output-root PATH
```

After the production fitted-model derivation lands, the separate candidate
route operation verifies the frozen route-probe record and runs the same six
nonunit cases without injecting density curves. The executable receives only
the case stock and gamma controls; the production print recipe owns derivation,
identity, preparation, and CUDA consumption:

```text
python run_validation.py candidate-route-probe --exe PATH --resource-root PATH --fixture-root PATH --output-root PATH
```

The `state` group exercises the production gamma snapshot-assignment boundary,
exact route hashing, route-state builders, and admission/recovery behavior. The
`lifetime` group covers same-size in-place table updates, different-size
replacement, queued old/new work across streams, 1 → 1.25 → 1 reuse, and
abort/recovery through the production prepared-frame owner. Its isolated first
subcheck also exercises the production print-resource owner directly and
requires the stock-anchored preflash host/device spectrum, hashes, raw values,
and storage to survive `1 → 1.25 → unchanged 1.25 → 1 → unchanged 1` before
Root performs any durable GPU preparation.

Every current direct or eligible print capture records an explicit split/fused
applicability value and a finite maximum RGBA difference. Candidate comparison
binds that value to the matching checksum-verified historical capture for its
configuration and case; it does not use a pooled epsilon. Enlarger diffusion is
explicitly non-applicable because the fused production launch is not used for
that split route. Run the focused runner checks with bytecode and temporary
files directed to the current fresh evidence root:

```text
python -B -m unittest discover -s validation/gamma -p test_run_validation.py
```

Current captures also serialize effective route, geometry, DIR, diffusion,
scatter/halation, print exposure, normalization, correction, glare, blur, and
unsharp facts from the immutable recipe and resolved frame descriptors. The
validation-only injected print-curve binding records requested gamma separately
from the recipe-owned print gamma. Historical capture JSON predating this
serializer contains inaccurate hardcoded settings and must not be used to
establish component activation.

The pinned-runtime configuration tests are separate because they import and
digest the actual spektrafilm parameter schema:

```text
PINNED_REFERENCE_PYTHON -B validation/gamma/test_generate_reference.py
```

The corrected G09 generator explicitly disables unrelated camera diffusion,
lens blur, grain, halation, glare, and scanner unsharp processing, while keeping
`debug.deactivate_spatial_effects` false. It asserts the effective settings
after `digest_params()` for encoded and linear-output parameter instances. The
frozen `enlarger-diffusion-black-pro-mist-half` reference nevertheless has
diffusion disabled, while frozen `spatial-dir-ramp-edge-direct` retains
halation and scanner unsharp. `reference_issues.json` binds those demonstrated
configuration errors to the exact frozen manifest and capture hashes. The
runner continues to report their historical numerical metrics but always marks
those two comparisons configuration-invalid for acceptance. Corrected fresh
diagnostics do not replace, promote, or alter frozen evidence.

`print_log_exposure_rgb_planar.f32` now always records the actual
`print_development_input`. For enlargement diffusion it observes the
post-diffusion strided planes immediately before development and verifies exact
density reproduction through the production CUDA sampler. Existing frozen
diffusion captures predate this correction: their identically named file is a
historical pre-diffusion diagnostic and is not relabelled or rewritten.

The `performance` group preserves the frozen 3 warmups and 11 measured
unchanged-render samples. Candidate captures also record 3 warmups and 11
measured samples for each print and film gamma edit in both directions,
separating recipe-build time from exact-context preparation time. Those edit
samples have no pre-change acceptance ceiling; compare their distributions with
the unchanged-render baseline and investigate material regressions rather than
inventing a post-candidate tolerance.

The executable is launched only by the runner for recorded evidence. Its local
capture interface is:

```text
GammaValidation.exe --case-group default-paper --resource-root PATH --input PATH --output-dir PATH --width N --height N
```

`--resource-root` must be the executable-relative staged `Resources` directory;
it is verified before the process composition root is accessed. Candidate
comparison re-verifies every frozen fixture before and after execution. An
`all` comparison also verifies the separate frozen route-probe record before
and after its candidate runs.
