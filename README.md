# Film-Juicer

Film-Juicer is an OpenFX 1.4 plug-in for DaVinci Resolve that performs physically-motivated film emulation using spectral modeling (81 wavelength samples from 380–780 nm at 5 nm steps). This project is a C++17 port of the reference implementation in [agx-emulsion] by Andrea Volpato, and aims to preserve modeling parity and behavior wherever practical.

## Contents

- [What It Does](#what-it-does)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start (DaVinci Resolve)](#quick-start-davinci-resolve)
- [Color Management](#color-management)
- [Parameters Overview](#parameters-overview)
- [Known Limitations](#known-limitations)
- [Diagnostics and Troubleshooting](#diagnostics-and-troubleshooting)
- [Support](#support)
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
- OS: Windows x64

Notes:

- Tile/ROI rendering is not supported (the host must request full-frame renders).
- Some builds may include CUDA acceleration (depending on how the plug-in was compiled and packaged).

## Installation

### Windows (OFX)

1. Copy `Juicer.ofx.bundle` into the system OFX plug-in directory:
   - `C:\Program Files\Common Files\OFX\Plugins\`
2. Ensure the bundle layout includes both the binary and the resources directory:

   - `Juicer.ofx.bundle/Contents/Win64/juicer.ofx`
   - `Juicer.ofx.bundle/Contents/Resources/...`

3. Restart DaVinci Resolve.

Film-Juicer discovers its data directory relative to the plug-in module location. If `Contents/Resources` is missing or moved, film stocks/papers may fail to load.

### Uninstall

Remove `Juicer.ofx.bundle` from the OFX plug-in directory and restart DaVinci Resolve.

## Quick Start (DaVinci Resolve)

1. Add the OFX effect named `Juicer` (group: `Negative-juice`) to a clip.
2. Set `Input color space` to match the clip signal feeding the node.
3. Pick a `Film stock`.
4. Choose one workflow:
   - Print path: leave `Bypass print` disabled and choose a `Print paper`.
   - Negative-only viewing: enable `Bypass print` (scanner/normalization behavior differs).
5. Set `Output color space` / output encoding options to match your grading pipeline.

## Color Management

Film-Juicer operates in the selected RGB color space and expects linear-light RGB unless you explicitly enable decoding.

- `Input color space`: choose the space that best describes the RGB values being fed into the OFX (e.g., DaVinci Wide Gamut, BT.2020, ACES2065-1, sRGB/Rec.709).
- `Decode input CCTF`:
  - Use this when feeding display-referred, gamma-encoded `ITU-R BT.2020` or `sRGB / Rec.709` signals and you want Film-Juicer to decode them to linear-light internally.
  - For `DaVinci Wide Gamut` and `ACES2065-1`, this toggle does not apply decoding.

On output, use the effect's output encoding controls to match the space/transfer your downstream pipeline expects.

## Parameters Overview

The UI is organized into groups that roughly map to the pipeline stages:

- Camera/exposure: exposure compensation, metering/auto exposure, film format scaling
- Spectral: upsampling mode, reference/enlarger illuminant selection
- Film development: stock selection, density curve behavior, masking / DIR couplers
- Print (optional): paper selection, print exposure + preflash, enlarger Y/M/C dichroics
- Scanner: spectral to tristimulus conversion, glare/compensation removal, output encoding
- Artifacts: halation, grain, gate weave/dust/scratches (where enabled)

Most users should treat Film-Juicer as a self-contained OFX effect: pick an input color space, choose a film stock, then decide whether you want a full print simulation or a negative-only view.

## Known Limitations

- Tile/ROI rendering is not supported; the host must request full-frame renders.
- Performance is workload-dependent; spectral processing and film artifacts (grain/halation/glare) can be significantly heavier than LUT-based looks.
- Film stock and paper choices depend on the bundled `Contents/Resources` assets; missing resources will reduce available options.

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

On Windows, set environment variables via System Properties -> Environment Variables, then restart DaVinci Resolve.

### Common Issues

- Missing film/paper options: verify the plug-in bundle includes `Contents/Resources` (next to the plug-in binary inside the bundle).
- Unexpected color management: confirm `Input color space` and output encoding match your Resolve color pipeline.
- Performance surprises: the spectral pipeline is heavier than LUT-based workflows; start simple (disable grain/halation/glare), then add features back.
- Render errors in the host: Film-Juicer requires full-frame renders (tile/ROI rendering is not supported).

## Support

If you hit a bug or mismatch:

- Include your DaVinci Resolve version, OS, GPU (if relevant), and the exact Film-Juicer settings (stock/paper + key toggles).
- Enable diagnostics (`JUICER_DIAGNOSTICS=2` or `3`) and attach the resulting `juicer_trace.txt` when reporting issues.

## License and Third-Party Notices

License terms for this repository depend on the presence of a top-level license file. If no `LICENSE` is present, treat the code as "all rights reserved" until licensing is clarified.

Third-party components and references:

- This project is based on the modeling work from [agx-emulsion] (see `UPSTREAM.md` and `external/agx-emulsion/` when present).
- Additional third-party code lives under `external/` and `third_party/`; consult the relevant headers/files for their license notices.

[agx-emulsion]: https://github.com/andreavolpato/agx-emulsion
