# Linux OFX host compatibility probe

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
when its behavior is approved. Routine changes require automated renders, not a
new user-supplied render or automatic baseline regeneration.
