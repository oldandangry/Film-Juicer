# Film-Juicer

Film-Juicer is an OpenFX 1.4 plug-in for DaVinci Resolve that performs physically-motivated film emulation using spectral modeling (81 wavelength samples from 380–780 nm at 5 nm steps). This project is a C++17 port of the reference implementation in [agx-emulsion] by Andrea Volpato, and aims to preserve modeling parity and behavior wherever practical.

## Contents

- [What It Does](#what-it-does)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start (DaVinci Resolve)](#quick-start-davinci-resolve)
- [Parameters Overview](#parameters-overview)
- [Data Assets and Profiles](#data-assets-and-profiles)
- [Building From Source](#building-from-source)
- [Packaging an OFX Bundle](#packaging-an-ofx-bundle)
- [Upstream Parity (agx-emulsion)](#upstream-parity-agx-emulsion)
- [Diagnostics and Troubleshooting](#diagnostics-and-troubleshooting)
- [Contributing](#contributing)
- [License and Third-Party Notices](#license-and-third-party-notices)

## What It Does

Film-Juicer implements a negative + print + scan pipeline driven by measured (or derived) spectral data:

- RGB input to spectral reconstruction (SPD), with optional UV/IR filtering
- Negative exposure and development (H-D curves, dye densities, masking / DIR couplers)
- Optional print simulation (paper profiles, dichroic enlarger filtration, neutral filter lookup)
- Scanner/viewing model (spectral to XYZ, glare injection, output encoding)
- Photochemical artifacts and optics (grain, halation, glare, basic scanner optics)

The intent is a data-driven workflow: film stocks and papers are defined by JSON profiles and spectral tables under `Resources/`, and the render pipeline consumes those assets at runtime.

## System Requirements

- Host: DaVinci Resolve (OpenFX host) or another OpenFX 1.4-compatible host
- Pixel format: float RGB/RGBA (the plug-in advertises float support)
- OS/build target: Windows x64 is the primary supported target in this repository

Notes:

- Tile/ROI rendering is not supported (the host must request full-frame renders).
- CUDA support is optional and compile-time gated (see source for `JUICER_ENABLE_CUDA`).

## Installation

### Windows (OFX)

1. Copy the bundle folder `Juicer.ofx.bundle` into the system OFX plug-in directory:
   - `C:\Program Files\Common Files\OFX\Plugins\`
2. Ensure the bundle layout includes both the binary and the resources directory:

   - `Juicer.ofx.bundle/Contents/Win64/juicer.ofx`
   - `Juicer.ofx.bundle/Contents/Resources/...`

3. Restart DaVinci Resolve.

Film-Juicer discovers its data directory relative to the plug-in module location. If `Contents/Resources` is missing or moved, film stocks/papers may fail to load.

## Quick Start (DaVinci Resolve)

1. Add the OFX effect named `Juicer` (group: `Negative-juice`) to a clip.
2. Set `Input color space` to match the clip signal feeding the node.
3. Pick a `Film stock`.
4. Choose one workflow:
   - Print path: leave `Bypass print` disabled and choose a `Print paper`.
   - Negative-only viewing: enable `Bypass print` (scanner/normalization behavior differs).
5. Set `Output color space` / output encoding options to match your grading pipeline.

## Parameters Overview

The UI is organized into groups that roughly map to the pipeline stages:

- Camera/exposure: exposure compensation, metering/auto exposure, film format scaling
- Spectral: upsampling mode, reference/enlarger illuminant selection
- Film development: stock selection, density curve behavior, masking / DIR couplers
- Print (optional): paper selection, print exposure + preflash, enlarger Y/M/C dichroics
- Scanner: spectral to tristimulus conversion, glare/compensation removal, output encoding
- Artifacts: halation, grain, gate weave/dust/scratches (where enabled)

For exact parameter names and defaults, see `src/main.cpp` (descriptor definitions) and `src/ParamNames.h`.

## Data Assets and Profiles

Runtime assets live under `Resources/` and are loaded via a computed `gDataDir` (see `src/main.cpp`).

Key directories:

- `Resources/profiles/`: JSON film stock and print paper profiles, plus neutral enlarger filter databases
- `Resources/film/` and `Resources/paper/`: spectral tables used by the negative and print models
- `Resources/filters/`: dichroics, heat-absorbing filters, lens transmission data
- `Resources/illuminants/`: illuminant SPDs (CSV) normalized onto the canonical wavelength axis
- `Resources/luts/`: spectral upsampling LUTs
- `Resources/Noise/`: noise/blue-noise tables (used by some paths)

Profile authoring notes:

- All spectral samples are expected on the canonical 380–780 nm @ 5 nm axis (81 samples).
- JSON ingestion paths validate wavelength grids; assets that do not match the reference axis are rejected.

## Building From Source

This repository is set up as a Visual Studio / MSBuild workflow.

Prerequisites (typical):

- Visual Studio 2022 with C++ build tools (C++17)
- Windows SDK (x64)
- An OpenFX 1.4 SDK / host support library (vendored headers are present under `external/` in this repo)

Build (from an "x64 Native Tools for VS 2022" prompt):

```bat
msbuild juicer.sln /p:Configuration=Debug /p:Platform=x64
msbuild juicer.sln /p:Configuration=Release /p:Platform=x64
```

Artifacts are expected under `juicer/x64/<Config>` (and MSVC intermediates under `x64/`). Do not commit either directory.

## Packaging an OFX Bundle

Film-Juicer is distributed as an OFX bundle that must include runtime resources.

Minimum bundle layout (Windows):

```
Juicer.ofx.bundle/
  Contents/
    Win64/
      juicer.ofx
    Resources/
      profiles/
      film/
      paper/
      filters/
      illuminants/
      luts/
      ...
```

An Inno Setup recipe is provided under `installer/` (see `installer/README.md`).

## Upstream Parity (agx-emulsion)

This project follows [agx-emulsion] as the modeling reference. The intent is to match:

- spectral sampling conventions (canonical wavelength axis)
- curve ingestion and resampling behaviors
- negative/print/scanner math and default parameter semantics

References in this repo:

- `UPSTREAM.md` (how to fetch upstream sources when needed)
- `agx-documentation.md` (parity notes and rules)

## Diagnostics and Troubleshooting

### Enable Logging

Runtime traces are controlled via environment variables:

- `JUICER_DIAGNOSTICS`:
  - `0` = off (default)
  - `1` = errors only
  - `2` = high-level state changes
  - `3` = verbose
- `JUICER_DIAGNOSTICS_PATH`: optional file path override for the log output

Default log file location (when enabled): the OS temp directory as `juicer_trace.txt`.

### Common Issues

- Missing film/paper options: verify the plug-in bundle includes `Contents/Resources` and the expected subfolders.
- Unexpected color management: confirm `Input color space` and output encoding match your Resolve color pipeline.
- Performance surprises: the spectral pipeline is heavier than 3x3 matrix LUT workflows; start with simpler settings (disable grain/halation/glare) and iterate.

## Contributing

Contributions are welcome, especially those that improve parity, stability, and performance.

- Keep changes localized to the appropriate module (state, spectral core, print, scanner, profiles).
- Preserve the canonical spectral axis and resampling rules; validate any asset/schema changes.
- Do not commit build outputs (`juicer/x64/*`, `x64/*`) or other host-specific artifacts.
- When behavior changes, document it in `docs/` and include reproduction notes (host/version, stock/paper, settings).

## License and Third-Party Notices

License terms for this repository depend on the presence of a top-level license file. If no `LICENSE` is present, treat the code as "all rights reserved" until licensing is clarified.

Third-party components and references:

- This project is based on the modeling work from [agx-emulsion] (see `UPSTREAM.md` and `external/agx-emulsion/` when present).
- Additional third-party code lives under `external/` and `third_party/`; consult the relevant headers/files for their license notices.

[agx-emulsion]: https://github.com/andreavolpato/agx-emulsion
