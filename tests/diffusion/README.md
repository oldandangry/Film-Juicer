# Diffusion tests

`Diffusion.HostReference` is an external-reference contract for the production
host-side diffusion resolver and PSF descriptor. During the test build,
`generate_host_cases.py` translates the committed JSON/NPZ cohort into a C++
header under the build tree. The test does not import spektrafilm, generate new
expectations, or write into `tests/diffusion/fixtures/`.

The cohort was generated from spektrafilm revision
`48645a2b4bf58c20b6a3b75c8022d0f462db754a`. Its manifest records the source
files, Python 3.13 environment, shapes, dtypes, RGB channel order, units, and
reference identities. The host comparison retains its original relative
tolerance of `5e-12 * max(1, abs(reference))`.

The upstream spektrafilm software used to generate this synthetic cohort is
GPLv3; the exact imported modules and hashes are recorded in the manifest. The
public generator, test code, and generated cohort are distributed with
Film-Juicer under its GPL-3.0-only license. No spektrafilm profile or LUT asset
is embedded in this cohort.

## Relocation provenance

The original fixture manifest SHA-256 was
`df23d799ad01506255d959b959b99dfcb6eca603f11cd0d195bb24cf4958965b`.
The NPZ payload remains byte-identical at
`963c7de75e513cfd503e48d52fe072380e4272468fc69cfede5bf4ccea914b4b`.
The public relocation changes only the dependency-lock path from
`tools/diffusion_reference/requirements.lock` to
`tests/diffusion/reference/requirements.lock`, plus the lock's own generated
command comment and corresponding checksum. Numerical arrays, cases, bounds,
upstream revision, and source hashes were not regenerated.

## Reference maintenance

Reference generation is separate from CTest. It requires the pinned
spektrafilm checkout and environment described by
`reference/requirements.lock`:

```sh
bash tests/diffusion/reference/bootstrap_reference_env.sh
.tmp/diffusion/reference-venv/bin/python \
  tests/diffusion/reference/generate_reference.py \
  --output-dir out/validation/diffusion-reference-candidate
```

The command writes a fresh candidate. It must not overwrite the committed
cohort. Generator contract checks run only in that reference environment:

```sh
.tmp/diffusion/reference-venv/bin/python -m unittest \
  tests.diffusion.reference.tests.test_reference_contract -v
```

The CUDA convolution generator remains a maintenance tool. Its 1.4 GB output
is intentionally not an ordinary repository fixture and is not registered in
CTest.
