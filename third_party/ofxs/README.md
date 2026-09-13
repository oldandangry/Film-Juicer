# OpenFX 1.4 and Resolve OFXS snapshot

This directory vendors the headers and eight support implementation files
consumed by Film-Juicer's existing Windows build on 2026-09-12. It preserves
that OpenFX 1.4 snapshot and its Resolve CUDA extensions. No upstream SDK
upgrade is part of the CMake migration.

The original local SDK has no recorded source revision. `PROVENANCE.json`
identifies every consumed file by its original and vendored SHA-256 hashes.
Those content hashes identify this snapshot; they do not imply a Git revision.
The local `external/OFXS` copy was compared with the actually consumed files
and all 35 files were identical before the changes below.

The only source edits are C++20 portability fixes:

- Remove typed dynamic exception specifications from declarations and
  definitions. Preserve the bodies and the existing empty `throw()` contracts.
- Replace the nonstandard `linux` macro with `__linux__` in platform guards.
- Remove whitespace left by removing those exception specifications.

The manifest records the affected files and edit counts. The support sources
do not require Eigen or headers from a separate OpenFX checkout. Both build
systems use only the vendored include directories.

The original license files remain with each component. `Legal/` contains
copies with distinct names for binary bundle installation. Source-level
copyright and license notices are retained in the vendored files.
