# Film-Juicer

Film-Juicer is a CUDA-based OpenFX 1.4 plug-in for DaVinci Resolve that performs spectral film emulation (81 wavelength samples, 380-780 nm at 5 nm steps), including negative, print, and scan stages.

This project is a C++17 port of the [agx-emulsion] reference implementation by Andrea Volpato. The goal is technical parity with the upstream model and a parameterization that stays close to the underlying photographic process.

## Contents

- [Overview](#overview)
- [System Requirements (CUDA Required)](#system-requirements-cuda-required)
- [Install / Uninstall (Windows)](#install--uninstall-windows)
- [Using It in Resolve](#using-it-in-resolve)
- [Color Management (Input/Output)](#color-management-inputoutput)
- [Controls (Map)](#controls-map)
- [Performance Notes](#performance-notes)
- [Support](#support)
- [License and Third-Party Notices](#license-and-third-party-notices)

## Overview

At a high level, the pipeline is:

- RGB input -> spectral reconstruction (SPD)
- film negative exposure/development (curves, dyes, masking / DIR couplers)
- optional print simulation (paper + enlarger filtration)
- scanner model (spectral -> tristimulus -> output encoding)
- optional artifacts (grain, halation, glare, scanner optics)

Film stocks and papers are shipped as data profiles and loaded at runtime from the plug-in bundle.

## System Requirements (CUDA Required)

- Host: DaVinci Resolve (this plug-in is built for Resolve; other OFX hosts are not supported)
- OS: Windows 10/11 x64
- GPU: NVIDIA CUDA GPU with compute capability 7.5+ (Turing / SM75 or newer)
- Driver: NVIDIA driver compatible with the CUDA runtime shipped with the plug-in

Practical guidance:

- Minimum architecture: Turing (e.g. RTX 20-series, GTX 16-series, Quadro RTX).
- Newer (Ampere/Ada/Hopper) is strongly recommended for interactive work.

No CPU fallback is provided. AMD/Intel GPUs are not supported.

## Install / Uninstall (Windows)

### Install

1. Copy `Juicer.ofx.bundle` to:
   - `C:\\Program Files\\Common Files\\OFX\\Plugins\\`
2. Confirm the bundle contains both the plug-in binary and runtime resources:
   - `Juicer.ofx.bundle/Contents/Win64/juicer.ofx`
   - `Juicer.ofx.bundle/Contents/Resources/...`
3. Restart DaVinci Resolve.

If `Contents/Resources` is missing or moved, Film-Juicer will not be able to load film/paper profiles.

### Uninstall

Remove `Juicer.ofx.bundle` from the OFX plug-in directory and restart DaVinci Resolve.

## Using It in Resolve

1. Add the OFX effect `Juicer` (group: `Negative-juice`) to your node graph.
2. Set `Input color space` to match the RGB values being fed into the OFX.
3. Choose a `Film stock`.
4. Choose a print workflow:
   - Print simulation: select a `Print paper` and keep `Bypass print` disabled.
   - Negative view: enable `Bypass print`.
5. Configure output encoding so the result lands in the space/transfer your downstream pipeline expects.

## Color Management (Input/Output)

Film-Juicer expects linear-light RGB in the selected space unless you explicitly enable decoding.

Input:

- `Input color space`: selects the RGB primaries/matrix used by the plug-in for input handling.
- `Decode input CCTF`:
  - Enable when feeding gamma-encoded `ITU-R BT.2020` or `sRGB / Rec.709` and you want Film-Juicer to linearize internally.
  - For `DaVinci Wide Gamut` and `ACES2065-1`, this toggle does not perform a decode.

Output:

- `Output color space` + `Apply output CCTF`: encodes the final RGB for delivery in the chosen output space/transfer.
- `Output linear pass-through`: bypasses output encoding when you want Film-Juicer to stay in linear-light and handle transforms elsewhere (e.g. in Resolve color management).

Resolve examples (common):

- RCM with timeline working space = DaVinci Wide Gamut / Intermediate:
  - `Input color space = DaVinci Wide Gamut`, `Decode input CCTF = off`
  - typically prefer `Output linear pass-through = on` and keep transforms in your managed pipeline
- Non-managed Rec.709:
  - `Input color space = sRGB / Rec.709`, `Decode input CCTF = on`
  - set output encoding to your desired display/output target

## Controls (Map)

The UI is grouped by stage:

- Camera/exposure: exposure compensation, metering/auto exposure, film format scaling
- Spectral: upsampling mode, reference/enlarger illuminant selection
- Film: stock selection, development controls, masking / DIR couplers
- Print: paper selection, print exposure + preflash, enlarger Y/M/C dichroics
- Scanner/output: scanner controls, output encoding / pass-through
- Artifacts: grain, halation, glare, plus optional gate effects where present

## Performance Notes

Performance is dominated by spectral processing and artifact simulation.

For a stable baseline:

- disable grain/halation/glare first, then enable them one at a time
- use the scanner LUT controls when available for interactive tuning
- expect a meaningful GPU cost relative to LUT-based look transforms

## Support

When reporting issues, include:

- DaVinci Resolve version
- Windows version
- NVIDIA GPU model + NVIDIA driver version
- whether you are using Resolve color management / ACES, and your working space
- Film-Juicer settings (film stock, paper, print bypass, input/output settings)

If the film/paper list is empty or looks incomplete, verify `Juicer.ofx.bundle/Contents/Resources` is present next to the plug-in binary.

## License and Third-Party Notices

License terms for this repository depend on the presence of a top-level license file. If no `LICENSE` is present, treat the code as "all rights reserved" until licensing is clarified.

Third-party components and references:

- Modeling reference: [agx-emulsion] (Andrea Volpato)
- Additional third-party code lives under `external/` and `third_party/`; consult those directories for their license notices.

[agx-emulsion]: https://github.com/andreavolpato/agx-emulsion

