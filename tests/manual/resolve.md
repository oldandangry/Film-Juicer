# DaVinci Resolve acceptance

CTest proves the public numerical, preparation, CUDA, and synthetic OFX
contracts. A short Resolve check covers the proprietary-host boundary that
CTest cannot exercise.

Migration status: the owner reported this checklist passed on Windows Release
on 2026-09-19. No further Windows Resolve evidence is outstanding for the
public test-suite consolidation.

The owner also reported native Linux migration and validation complete and
passing on 2026-09-19, including the installed Resolve-library preload. No
further Linux owner-run evidence is outstanding for this consolidation.

## Candidate preparation

From a Visual Studio 18 x64 developer PowerShell in `C:\Dev\Codex`:

```powershell
cmake --build --preset windows-clang-release
ctest --preset windows-clang-release
cmake --build out/build/windows-clang-release --target bundle-archive
```

The unfiltered CTest command is the complete automated run on a supported GPU
machine. The archive target cleanly restages the complete candidate at:

```text
C:\Dev\Codex\out\stage\windows-clang-release\juicer.ofx.bundle
```

Close Resolve, preserve any existing Juicer bundle outside the OFX search
directories, and copy the complete candidate to the normal plug-in directory:

```text
C:\Program Files\Common Files\OFX\Plugins
```

Do not install only the `.ofx` binary.

## Focused host check

Use a disposable project and a short chart or clip with neutral gray, saturated
color, shadows, highlights, and a hard bright edge. Keep input/output settings
fixed and check:

- Juicer is discovered under **OpenFX → Negative-juice**, its profile lists
  populate, and an instance can be added and removed.
- Kodak Portra 400 renders through negative direct and negative print routes.
- Kodak Ektachrome 100 renders through positive direct and positive print
  routes. Use Kodak Professional Portra Endura for the two print checks.
- An RGBA input preserves opaque, fractional, and transparent alpha.
- Scatter/halation Off, scatter only, back-reflection only, and Both render at
  full and half scale without an obvious physical-scale discontinuity.
- A longer render can be cancelled and followed by a successful render.
- Two instances can render and be removed, after which the project and Resolve
  close without a crash, hang, CUDA error, or stuck resource.

A dated owner statement that this focused checklist passed is sufficient for
the migration record. Save extra captures or diagnostics only when they help
explain a failure; a mandatory evidence form is intentionally not used.

This consolidation changed no production rendering or resource ownership, so
Debug repetition, forced context reset, profiling, benchmarks, and reference
regeneration are not acceptance gates.

## LUT-only scanner cleanup

After installing a bundle containing the LUT-only scanner cleanup, repeat the
four route checks above on one deterministic frame and additionally confirm:

- **Scanner use LUT** is absent, while **Scanner LUT resolution** remains and
  defaults to 17.
- Resolution 17 renders finite output on negative/positive direct/print routes.
- Changing one instance from 17 to 33 and back to 17 completes successfully and
  restores the original resolution-17 output.
- A project previously saved with `ScannerUseLUT=false` opens and renders with
  the LUT-only scanner without a missing-parameter or resource error.
- No selectable scanner mode produces the retired exact-path performance cost.

Record the installed bundle hash and any unavailable saved-project or platform
check. Repository tests cover CUDA output and resource transitions but cannot
establish Resolve's handling of a removed OFX parameter.
