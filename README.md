# Film-Juicer

**Film-Juicer** is an OpenFX 1.4 plug-in for DaVinci Resolve implementing **spectral film emulation** with a physically-motivated negative → print → scan pipeline.

The core model is a C++ port of Andrea Volpato’s [agx-emulsion] reference implementation, with an explicit goal of **technical parity** (constants, transforms, stochastic processes, and serialization semantics).

---

## Contents

- [What This Is](#what-this-is)
- [Pipeline Overview](#pipeline-overview)
- [Model Invariants](#model-invariants)
- [Runtime Assets & Profiles](#runtime-assets--profiles)
- [System Requirements](#system-requirements)
- [Install / Uninstall (Windows)](#install--uninstall-windows)
- [Using It in Resolve](#using-it-in-resolve)
- [Color Management (Input/Output)](#color-management-inputoutput)
- [Controls (Map)](#controls-map)
- [Build From Source (Windows)](#build-from-source-windows)
- [Diagnostics & Debugging](#diagnostics--debugging)
- [Performance Notes](#performance-notes)
- [Upstream Parity](#upstream-parity)
- [Developer Tooling](#developer-tooling)
- [Support](#support)
- [License and Third-Party Notices](#license-and-third-party-notices)

---

## What This Is

Film-Juicer models the photographic pipeline explicitly:

- **Input RGB → spectral reconstruction (SPD)** (81 wavelength samples, 380–780 nm @ 5 nm)
- **Film exposure** using stock spectral sensitivities + camera spectral filtering
- **Film development** via H–D density curves, masking, and optional DIR coupler effects
- **Print exposure/development** (optional) via enlarger illuminant + dichroic filtration + paper model
- **Scanner/viewing model**: spectral → tristimulus → output color space + optional output encoding
- Optional artifacts (grain, halation, glare, optics)

The plug-in is **data-driven**: film stocks, papers, and spectral tables are shipped in `Resources/` and loaded at runtime from the plug-in bundle.

---

## Pipeline Overview

The codebase is organised around a stage-aligned pipeline mirroring `agx-emulsion`:

- `src/JuicerEffect.*`: OFX wiring, parameter definitions, instance lifecycle
- `src/JuicerState.*`, `src/WorkingState.h`: immutable base profile data + derived render-ready state (double-buffered)
- Spectral core: `src/SpectralData.h`, `src/SpectralContext.*`, `src/SpectralProcessing.h`, `src/SpectralMathAVX.cpp`
- Film stages: `src/ExposeFilmStage.*`, `src/DevelopFilmStage.*`, `src/FilmProcessing.h`, `src/Couplers.h`, `src/SpatialDIR.*`
- Print stages: `src/ExposePrintStage.*`, `src/DevelopPrintStage.*`, `src/Print.*`, `src/NeutralFilters.*`
- Scanner/output: `src/ScanStage.*`, `src/Scanner*.{h,cpp}`, `src/OutputEncoding.*`, `src/ColorTransforms.h`
- CUDA stages live under `src/Cuda/` (built when `JUICER_ENABLE_CUDA=1`)

Execution paths:

- **CPU path**: multithreaded `OFX::ImageProcessor` pipeline (also used for non-CUDA hosts / development).
- **CUDA path**: Resolve CUDA render entry points when built with CUDA; device pointers are used end-to-end to avoid round-trips.

---

## Model Invariants

These invariants are assumed across code, profiles, and LUTs:

- **Spectral axis**: 81 samples at **380–780 nm** in **5 nm** steps.
- **Log exposure axis**: 256 samples spanning **−3 EV → +4 EV** (used by density curve LUTs).
- **Enlarger neutral steps**: 170 steps (neutral Y/M/C values in the filter database are normalized to this).

If you change any of these, you must regenerate dependent assets and re-validate parity.

---

## Runtime Assets & Profiles

### Bundle layout (runtime)

At runtime the plug-in resolves `gDataDir` to:

- `Juicer.ofx.bundle/Contents/Resources/`

If the Resources directory is missing or moved, the plug-in will load fallback catalog entries and may refuse to build working state.

### `Resources/` structure (high level)

- `Resources/profiles/`: JSON profiles (film stocks + print papers) and neutral filter database
- `Resources/paper/<paper>/`: print-paper CSV spectra + baselines (paired with JSON profile metadata)
- `Resources/illuminants/`: standard illuminant SPDs (CSV)
- `Resources/filters/`: dichroics, KG3 heat filter, lens transmission spectra (CSV)
- `Resources/luts/spectral_upsampling/`: spectral upsampling tables (e.g. Hanatos 2025 LUTs)
- `Resources/Noise/`: stochastic textures/tiles (used by grain and related effects)

### Profile schema (JSON)

Profiles are dense arrays aligned to the reference axes. In particular:

- File structure:
  - `info`: lightweight metadata used for cataloging and defaults (`stock`, `name`, `type`, illuminants, etc.)
  - `data`: dense numeric arrays (spectral curves, density curves, axis arrays)
- `wavelengths`: 81 samples matching 380–780 nm @ 5 nm
- `log_exposure`: 256 samples matching −3 → +4 EV
- `log_sensitivity`: 81×3 (log10 sensitivities)
- `dye_density`: 81×5 (CMY + additional components as authored)
- `density_curves`: 256×3 (CMY optical densities vs log exposure)

The loaders validate axis compatibility; assets that do not match the reference grids are rejected.

Notes:

- Profiles in `Resources/profiles/` commonly use `NaN` to encode “missing” samples (toe/shoulder gaps); the implementation is written to preserve missingness semantics rather than silently “healing” NaNs.
- Catalog keys come from `info.stock` (and typically match the JSON filename stem). Keep these stable because they are also referenced by the neutral filter database.

Neutral filter database:

- `Resources/profiles/enlarger_neutral_ymc_filters.json` maps `[paper][illuminant][film] → (Y,M,C)` normalized to 170 enlarger steps.

### Catalog behavior (films/papers menus)

The UI film/paper option lists are derived from shipped data:

- Primary path: keys present in `Resources/profiles/enlarger_neutral_ymc_filters.json` (so neutral defaults exist for common paper/illuminant/film combinations).
- Fallback path: scan `Resources/profiles/*.json` and include any `type=negative` (film) and `type=paper` (print) profiles.
- Final fallback: a small hard-coded set of stocks/papers if the Resources directory is unavailable.

For print papers, the runtime also expects a corresponding `Resources/paper/<paper>/` directory containing the per-paper CSV curves (dye epsilons, log sensitivities, baselines). The mapping between a paper profile and its folder uses name/key heuristics; keep folder names closely related to `info.stock`/`info.name` to avoid mismatches.

---

## System Requirements

Runtime target:

- Host: DaVinci Resolve (other OFX hosts are untested / unsupported)
- OS: Windows 10/11 x64
- Images: float RGB/RGBA (OpenFX `eBitDepthFloat`)

CUDA target (default release configuration):

- NVIDIA GPU with **compute capability 7.5+** (Turing / SM75 or newer)
- A compatible NVIDIA driver for the CUDA runtime shipped with the plug-in

Notes:

- The shipped Visual Studio CUDA configuration targets `sm_75` by default (`juicer.vcxproj` `CudaCompile/CodeGeneration`).
- CPU rendering is available for development and non-CUDA builds, but the primary performance target is the CUDA path.

---

## Install / Uninstall (Windows)

### Install

1. Copy `Juicer.ofx.bundle` to:
   - `C:\\Program Files\\Common Files\\OFX\\Plugins\\`
2. Confirm the bundle contains both the plug-in binary and runtime resources:
   - `Juicer.ofx.bundle/Contents/Win64/juicer.ofx`
   - `Juicer.ofx.bundle/Contents/Resources/...`
3. Restart DaVinci Resolve.

Troubleshooting:

- If Resolve fails to load the plug-in due to missing CUDA runtime DLLs, ensure the required `cudart64_*.dll` is available in Resolve’s DLL search path (commonly shipped alongside the plug-in binary in `Contents/Win64/`, or installed system-wide).

### Uninstall

Remove `Juicer.ofx.bundle` from the OFX plug-in directory and restart DaVinci Resolve.

For installer packaging, see `installer/README.md`.

---

## Using It in Resolve

1. Add the OFX effect `Juicer` (group: `Negative-juice`) to your node graph.
2. Set `Input color space` to match the RGB values being fed into the OFX.
3. Choose a `Film stock`.
4. Choose a print workflow:
   - Print simulation: select a `Print paper` and keep `Bypass print` disabled.
   - Negative view: enable `Bypass print` (scanner is configured for negative viewing and may apply auto-gain).
5. Configure output encoding so the result lands in the space/transfer your downstream pipeline expects.

---

## Color Management (Input/Output)

Film-Juicer expects **linear-light RGB** in the selected space unless you explicitly enable decoding.

Input:

- `Input color space`: selects primaries/matrix used for input handling.
- `Decode input CCTF`:
  - Enable when feeding gamma-encoded `ITU-R BT.2020` or `sRGB / Rec.709` and you want Film-Juicer to linearize internally.
  - For scene-linear spaces (e.g. DaVinci Wide Gamut / Intermediate, ACES2065-1), keep this disabled.

Output:

- `Output color space` + `Apply output CCTF`: encodes the final RGB for delivery in the chosen output space/transfer.
- `Output linear pass-through`: bypasses output encoding when you want Film-Juicer to stay in linear and let Resolve/ACES handle transforms.

Resolve examples (common):

- RCM timeline working space = DaVinci Wide Gamut / Intermediate:
  - `Input color space = DaVinci Wide Gamut`, `Decode input CCTF = off`
  - Prefer `Output linear pass-through = on` and keep transforms in RCM
- Non-managed Rec.709:
  - `Input color space = sRGB / Rec.709`, `Decode input CCTF = on`
  - Set output encoding to your desired delivery target

For output encoding implementation notes, see `output-encoding.md`.

---

## Controls (Map)

The UI is grouped by pipeline stage:

- Camera/exposure: exposure compensation, metering/auto exposure, film format scaling
- Spectral: SPD reconstruction method, reference/enlarger illuminant selection
- Film: stock selection, development controls, masking / DIR couplers
- Print: paper selection, print exposure + preflash, enlarger Y/M/C filtration
- Scanner/output: scanner controls, output encoding / pass-through
- Artifacts: grain, halation, glare, plus optional gate effects

---

## Build From Source (Windows)

### Toolchain & dependencies

- Visual Studio 2022 (v143 toolset)
- CUDA Toolkit (the project is wired to VS CUDA build customizations; see `juicer.vcxproj`)
- OpenFX 1.4 headers + OFX Support library (OFXS)
- Eigen 3.4 (configured as an external include directory)

This repository vendors several dependencies under `external/` and `third_party/`, but the `.vcxproj` currently uses **machine-local include paths** by default (e.g. `C:\\Dev\\...`). Adjust them to match your environment.

Practical notes:

- The project conditionally enables CUDA when Visual Studio CUDA build customizations are present (see the `CUDA 13.1.props` import in `juicer.vcxproj`); this defines `JUICER_ENABLE_CUDA=1` in the CUDA-enabled build.
- The Visual Studio project references OFXS support sources via `..\\OFXS\\Support\\Library\\...`. If you want to use the vendored copy in `external/OFXS/`, either update the `.vcxproj` entries or provide an equivalent `..\\OFXS` path (e.g. junction/symlink).

### Build

From a “x64 Native Tools for VS 2022” prompt:

```bat
msbuild juicer.sln /p:Configuration=Debug /p:Platform=x64
msbuild juicer.sln /p:Configuration=Release /p:Platform=x64
msbuild juicer.sln /t:Clean /p:Platform=x64
```

Outputs:

- Plug-in binary: `juicer/x64/<Config>/juicer.ofx`
- MSVC intermediates: `x64/<Config>/` (do not commit)

### Bundle for Resolve

Create the following structure and copy the outputs:

- `Juicer.ofx.bundle/Contents/Win64/juicer.ofx` (from the build output)
- `Juicer.ofx.bundle/Contents/Resources/` (copy `Resources/` from the repo)

Install by copying the bundle to the Resolve OFX directory (see above).

---

## Diagnostics & Debugging

### Runtime tracing

Tracing is controlled via environment variables:

- `JUICER_DIAGNOSTICS`:
  - `0`: off (default)
  - `1`: errors
  - `2`: high-level state changes
  - `3`: verbose (per-frame / heavy traces)
- `JUICER_DIAGNOSTICS_PATH`: optional path override for the trace file

Default log path (when enabled): the OS temp directory, `juicer_trace.txt` (see `src/Logging.h`).

Set these environment variables **before launching Resolve** (the plug-in initializes logging from the process environment).

### Build-time / parity flags

Common build flags used during development:

- `JUICER_ENABLE_CUDA`: compile CUDA support and advertise CUDA render capability to the host
- `JUICER_CUDA_ONLY`: reject non-CUDA render entry points (useful when validating that Resolve is dispatching CUDA)
- `JUICER_CUDA_SELF_CHECK` / `JUICER_CUDA_VALIDATE_PRIMITIVES`: CUDA parity checks (see `UserOverrides.props`)
- `JUICER_SPD_DEBUG`: SPD sampling probes and related logging

Notes:

- `UserOverrides.props` is imported by the `Release|x64` configuration in `juicer.vcxproj`; set `<JuicerCudaValidatePrimitives>1</JuicerCudaValidatePrimitives>` to enable additional CUDA-vs-CPU validation during development.

---

## Performance Notes

Cost drivers are dominated by spectral processing and stochastic/spatial effects.

Practical tuning workflow:

- Disable grain/halation/glare first; enable them one at a time.
- Prefer running in a managed linear pipeline and avoid extra gamma transforms inside the plug-in.
- Validate performance in Resolve using the CUDA render path (CPU path exists but is not the primary target).

For deeper performance notes and experiments, see `docs/`.

---

## Upstream Parity

- Parity rules and upstream constant references: `agx-documentation.md`
- How to fetch upstream sources locally for inspection: `UPSTREAM.md`

When changing math, constants, or serialization details, update parity notes and re-validate against the upstream reference.

---

## Developer Tooling

The `tools/` directory contains scripts used to generate or validate shipped assets, including:

- Neutral filter database generation (`tools/generate_enlarger_neutral_ymc_filters.py`)
- Color space table generation (`tools/generate_colourspace_tables.py`)
- Negative curve derivation tooling (`tools/derive_negative_curves.py`, `tools/README-derive-negative-curves.md`)

These tools are not part of the runtime plug-in; they exist to keep assets reproducible and parity-auditable.

---

## Support

When reporting issues, include:

- DaVinci Resolve version
- Windows version
- GPU model + NVIDIA driver version (if using CUDA)
- Whether you are using Resolve color management / ACES, and your working space
- Film-Juicer settings (film stock, paper, print bypass, input/output settings)
- `juicer_trace.txt` output (with `JUICER_DIAGNOSTICS=2` or `3`) if possible

If the film/paper list is empty or looks incomplete, verify `Juicer.ofx.bundle/Contents/Resources/` is present next to the plug-in binary.

---

## License and Third-Party Notices

License terms for this repository depend on the presence of a top-level license file. If no `LICENSE` is present, treat the code as “all rights reserved” until licensing is clarified.

Third-party components and references:

- Modeling reference: [agx-emulsion] (Andrea Volpato)
- OpenFX support code under `external/OFXS/` (consult upstream licensing)
- Additional third-party code under `third_party/` (consult embedded notices)

[agx-emulsion]: https://github.com/andreavolpato/agx-emulsion
