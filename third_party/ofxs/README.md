# OpenFX 1.4 and Resolve OFXS snapshot

The current build uses the 17 vendored OpenFX 1.4 C SDK headers, including
the Resolve CUDA extensions. The 2026-09-12 snapshot also captured 18 OFXS C++
Support sources and headers that have since been removed from the source tree.

The original local SDK has no recorded source revision.
[`PROVENANCE.json`](PROVENANCE.json) records the original snapshot hash as
`sha256` for each of the 35 captured source files. Paths are relative to this
directory. In `files`, `vendored_sha256` identifies the current C SDK header
bytes. In `historical_files.files`, it identifies the removed Support files
at the Film-Juicer commit in `historical_files.vendored_revision`. From the
repository root, use `git cat-file blob <vendored_revision>:third_party/ofxs/<destination>`
to recover a historical file's vendored bytes.

For eight Support files, the preserved local `external/OFXS` snapshot is the
only verified source of the original pre-edit bytes. Those eight records
have different `sha256` and `vendored_sha256` values. Verify the snapshot bytes
against `sha256` when reconstructing the original capture.

The manifest inventories captured source and header files. The licenses and
this README are outside that inventory. CMake ships the manifest at
`Contents/Legal/ofxs/PROVENANCE.json` in the plug-in bundle.

The snapshot's initial C++20 portability edits were:

- Remove typed dynamic exception specifications from declarations and
  definitions. Preserve the bodies and the existing empty `throw()` contracts.
- Replace the nonstandard `linux` macro with `__linux__` in platform guards.
- Remove whitespace left by removing those exception specifications.

The manifest records the affected files and edit counts. It also records
the Film-Juicer commits that added the C `stdarg.h` include branch to
`ofxParamExt.h` and tracked CUDA stream-property presence separately from
the stream pointer in the historical `ofxsProcessing.h`.

The original license files remain with each component. `Legal/` contains
copies with distinct names for binary bundle installation. Source-level
copyright and license notices are retained in the vendored files.
