# OpenFX 1.4 and Resolve OFXS snapshot

This directory preserves the OpenFX 1.4 and Resolve OFXS files consumed by
Film-Juicer's Windows build on 2026-09-12, including its Resolve CUDA
extensions. The current build uses the vendored C SDK headers. The
snapshot also includes the inactive OFXS C++ Support sources and headers.

The original local SDK has no recorded source revision.
[`PROVENANCE.json`](PROVENANCE.json) records the original snapshot bytes in
`sha256` and the current vendored bytes in `vendored_sha256` for each of
the 35 captured files. The local `external/OFXS` copy matched all 35
files at capture. CMake ships the manifest under `Contents/Legal/ofxs/`
in the plug-in bundle.

The snapshot's initial C++20 portability edits were:

- Remove typed dynamic exception specifications from declarations and
  definitions. Preserve the bodies and the existing empty `throw()` contracts.
- Replace the nonstandard `linux` macro with `__linux__` in platform guards.
- Remove whitespace left by removing those exception specifications.

The manifest records the affected files and edit counts. It also records
the Film-Juicer commits that added the C `stdarg.h` include branch to
`ofxParamExt.h` and tracked CUDA stream-property presence separately from
the stream pointer in the retained `ofxsProcessing.h`.

The original license files remain with each component. `Legal/` contains
copies with distinct names for binary bundle installation. Source-level
copyright and license notices are retained in the vendored files.
