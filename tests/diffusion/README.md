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

Committed-fixture integrity checks need only the ordinary Python 3.13 test
environment and NumPy. They verify the accepted JSON/NPZ byte hashes, array
hashes, schemas and coverage, and the structure of historical provenance. They
do not import spektrafilm or require the old capture environment or current Git
revisions:

```sh
python -B -m unittest tests.diffusion.reference.tests.test_reference_contract -v
python -B -m unittest tests.diffusion.reference.tests.test_maintenance -v
```

Fresh-capture checks are explicit maintenance operations outside CTest. Use a
clean spektrafilm checkout and the hashed `reference/requirements.lock` in a
separate native Python 3.13 environment. A candidate records the actual current
reference revision; it does not redefine the accepted fixture's revision.
Generation requires an output directory and rejects the committed fixture tree.

The existing Linux bootstrap remains available for its historical `.tmp/`
layout. For a new Linux campaign, keep the environment and evidence separate:

```sh
DIFFUSION_RUN=out/validation/linux-debug/diffusion-reference-candidate
mkdir -p "$DIFFUSION_RUN/evidence"
uv venv --managed-python --python 3.13 "$DIFFUSION_RUN/environment"
uv pip sync --python "$DIFFUSION_RUN/environment/bin/python" tests/diffusion/reference/requirements.lock
uv pip install --python "$DIFFUSION_RUN/environment/bin/python" --no-deps -e external/spektrafilm
uv pip freeze --python "$DIFFUSION_RUN/environment/bin/python" > "$DIFFUSION_RUN/evidence/python-freeze.txt"
uv --version > "$DIFFUSION_RUN/evidence/uv-version.txt"
"$DIFFUSION_RUN/environment/bin/python" -B tests/diffusion/reference/generate_reference.py \
  --environment-dir "$DIFFUSION_RUN/environment" --evidence-dir "$DIFFUSION_RUN/evidence" \
  --output-dir "$DIFFUSION_RUN/candidate"
"$DIFFUSION_RUN/environment/bin/python" -B -m tests.diffusion.reference.tests.test_reference_contract \
  --capture-dir "$DIFFUSION_RUN/candidate" -v
```

For native Windows, use PowerShell from the repository root. These commands use
the same lock with native wheels and keep UTF-8 evidence free of a BOM:

```powershell
$diffusionRun = "out/validation/windows-clang-debug/diffusion-reference-candidate"
New-Item -ItemType Directory -Force "$diffusionRun/evidence" | Out-Null
uv venv --managed-python --python 3.13 "$diffusionRun/environment"
$diffusionPython = "$diffusionRun/environment/Scripts/python.exe"
uv pip sync --python $diffusionPython tests/diffusion/reference/requirements.lock
uv pip install --python $diffusionPython --no-deps -e external/spektrafilm
$diffusionUtf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$PWD/$diffusionRun/evidence/python-freeze.txt", ((uv pip freeze --python $diffusionPython) -join "`n") + "`n", $diffusionUtf8)
[IO.File]::WriteAllText("$PWD/$diffusionRun/evidence/uv-version.txt", ((uv --version) -join "`n") + "`n", $diffusionUtf8)
& $diffusionPython -B tests/diffusion/reference/generate_reference.py --environment-dir "$diffusionRun/environment" --evidence-dir "$diffusionRun/evidence" --output-dir "$diffusionRun/candidate"
& $diffusionPython -B -m tests.diffusion.reference.tests.test_reference_contract --capture-dir "$diffusionRun/candidate" -v
```

The explicit capture suite checks all fixture contracts plus the current native
interpreter/platform, environment evidence hashes, repository and clean
reference revisions, and actual files imported by the public generator. It
rejects stale evidence. Candidates require separate review before promotion;
ordinary tests and these commands leave the accepted fixture bytes unchanged.

The CUDA convolution generator remains a maintenance tool. Its 1.4 GB output
is intentionally not an ordinary repository fixture and is not registered in
CTest.
