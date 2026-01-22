# Film-Juicer (DaVinci Resolve OFX)

Film-Juicer is an OpenFX 1.4 plug-in for DaVinci Resolve that implements a **spectral, stage-based film pipeline**:

1. **RGB → spectral reconstruction** (estimated scene SPD)
2. **Negative exposure + development** (layer sensitivities, H–D curves, couplers)
3. Optional **print exposure + paper development** (enlarger illuminant + Y/M/C filtration + paper model)
4. **Scanner / viewing transform** (spectral → XYZ → output RGB, with optics/artifacts)

This is not a LUT. The look emerges from explicit modeling of **exposure, dye densities, illuminants, filtration, and scanning**.

Film-Juicer is a Resolve-targeted port of Andrea Volpato’s [agx-emulsion](https://github.com/andreavolpato/agx-emulsion) (simulation of color film photography from scratch). The control surface and stage order are designed to remain close to the upstream model.

## Contents

- [System Requirements](#system-requirements)
- [Install / Uninstall (Windows)](#install--uninstall-windows)
- [Using It in Resolve](#using-it-in-resolve)
- [Terminology (Quick)](#terminology-quick)
- [What It Simulates](#what-it-simulates)
- [Color Pipeline (Input → Output)](#color-pipeline-input--output)
- [How the Simulation Works](#how-the-simulation-works)
  - [1) RGB → SPD (Spectral Reconstruction)](#1-rgb--spd-spectral-reconstruction)
  - [2) Film Exposure (Per-Layer Raw)](#2-film-exposure-per-layer-raw)
  - [3) Film Development (Raw → Density CMY)](#3-film-development-raw--density-cmy)
  - [4) Print Simulation (Optional)](#4-print-simulation-optional)
  - [5) Scanner / Viewing Model (Density → RGB)](#5-scanner--viewing-model-density--rgb)
- [Controls (Physical Semantics)](#controls-physical-semantics)
- [Recommended Resolve Workflows](#recommended-resolve-workflows)
- [Performance / Quality Trade-offs](#performance--quality-trade-offs)
- [Validation & Common Pitfalls](#validation--common-pitfalls)
- [References](#references)

## System Requirements

- Host: DaVinci Resolve (OFX)
- OS: Windows 10/11 x64
- GPU: NVIDIA CUDA (the release configuration targets **SM75+ / Turing or newer**)

## Install / Uninstall (Windows)

### Install

1. Copy `Juicer.ofx.bundle` to:
   - `C:\\Program Files\\Common Files\\OFX\\Plugins\\`
2. Confirm the bundle contains:
   - `Juicer.ofx.bundle/Contents/Win64/juicer.ofx`
   - `Juicer.ofx.bundle/Contents/Resources/` (required runtime assets)
3. Restart Resolve.

### Uninstall

Remove `Juicer.ofx.bundle` from the OFX plug-in directory and restart Resolve.

## Using It in Resolve

1. Add the OFX effect `Juicer` (group: `Negative-juice`) to a node.
2. Set `Input color space` to match the RGB values being fed into the OFX.
3. Pick a `Film stock`.
4. Decide whether you want a print stage:
   - **Negative scan**: leave `Bypass print = on`
   - **Print simulation**: set `Bypass print = off`, then pick a `Print paper` and tune `Print` / `Enlarger` controls
5. Set output behavior:
   - Managed pipeline: enable `Output linear pass-through`
   - Display-referred: set `Output color space` and keep `Apply output CCTF` enabled

## Terminology (Quick)

- **Scene-linear**: values are proportional to light. Not log, not gamma-encoded.
- **SPD**: spectral power distribution — think “the spectrum” of the light.
- **Illuminant**: the light source spectrum used for a stage (reference, enlarger, viewing/scanner).
- **Density (OD)**: optical density — higher density means more absorption (darker / less light transmitted).
- **CMY density**: densities of the cyan, magenta, and yellow dyes formed by development.
- **XYZ**: CIE XYZ tristimulus values used as an intermediate for color conversion.
- **CCTF**: “color component transfer function” — basically the gamma/transfer function for encoding/decoding.
- **LUT**: lookup table — a precomputed approximation used for speed.

## What It Simulates

Film-Juicer explicitly models the “photographic chain” rather than applying a 3D LUT:

- **It works in spectra internally**: instead of only pushing RGB numbers around, the plug-in estimates a full visible spectrum per pixel. Internally this is sampled at **81 wavelengths** (380–780 nm in 5 nm steps).
- **Film negative**: the estimated spectrum is “seen” by three film layers (blue/green/red sensitive). Those three layer exposures are developed through film response curves into **CMY dye densities** (how much dye is formed).
- **Print (optional)**: the negative is projected onto paper using an enlarger light and Y/M/C filtration. Paper has its own sensitivities and response curves, producing print dye densities.
- **Scan / view**: dye densities are converted back into color by simulating how much light makes it through (or off) the medium, then converting to XYZ and finally to your chosen output RGB space.
- **Artifacts (optional)**: halation, grain, glare, blur/unsharp, and gate effects are applied in the stage where they physically belong (not as one “look” at the end).

## Color Pipeline (Input → Output)

### Input expectations

Film-Juicer expects **scene-linear RGB** in the selected input primaries unless you enable decoding.

Plain-English check:

- If your values “look like a photo” on a waveform/parade (already contrasty), they’re probably **not** linear.
- If they look “flat” and highlights feel huge, they’re more likely linear (or log).

Supported input primaries:

- DaVinci Wide Gamut
- ITU-R BT.2020
- ACES2065-1
- sRGB / Rec.709

`Decode input CCTF`:

- Think of this as “decode gamma / transfer function”.
- When enabled, Film-Juicer decodes **BT.2020** and **sRGB/Rec.709** into linear.
- For already-linear encodings (DWG / ACES2065-1), leave it off.
- Film-Juicer does **not** decode log working encodings like DaVinci Intermediate or ACEScct. If your node graph is in log, convert to a scene-linear signal (e.g. via Resolve CST) before the OFX.

Practical guidance:

- If your pipeline is managed (RCM/ACES), explicitly ensure the signal into Juicer is scene-linear (for example: CST into linear DWG or ACES2065-1), then keep decoding off.
- Avoid feeding display-referred, heavily clipped values; the model assumes physically-plausible radiometric inputs.

### Output

At the end of the scan/view stage, Film-Juicer produces linear RGB and can optionally:

- Transform into a selected `Output color space`
- Apply `Apply output CCTF` (display encoding)
- Or bypass encoding/clipping with `Output linear pass-through` for managed pipelines

Note: when `Output linear pass-through` is disabled, Film-Juicer applies output encoding (if enabled) and then clips to display range.

## How the Simulation Works

This section is written for technical users who want to reason about the behavior of the model.

### 1) RGB → SPD (Spectral Reconstruction)

RGB does not uniquely determine a spectrum (metamerism). Film-Juicer therefore **guesses a plausible spectrum** (SPD) that would produce the input RGB.

- `Spectral upsampling = Hanatos`: LUT-based reconstruction (generally higher fidelity).
- `Spectral upsampling = Mallett`: basis-based reconstruction (simpler approximation).

The reconstructed SPD is then used for all “spectral” steps: film exposure, illuminant interactions, and spectral integration for scanning.

Implication: highly non-standard emitters (narrow-band LEDs, lasers, display primaries) may not be reproduced with true spectral accuracy because the input is still 3-channel RGB.

### 2) Film Exposure (Per-Layer Raw)

The film negative is modeled as three spectrally sensitive layers (blue/green/red sensitive). For each pixel, Film-Juicer computes three “raw” exposure values — one per layer.

In plain terms: the spectrum is multiplied by each layer’s sensitivity curve and summed up across wavelengths.

`Exposure Compensation Ev` is applied like a camera exposure change:

- `+1 EV` = double the exposure into the negative layers
- `-1 EV` = half the exposure

### 3) Film Development (Raw → Density CMY)

Film development is modeled with film response curves in **log exposure** (the classic H–D curves).

In practice: Film-Juicer takes the three layer exposures, converts to log exposure, then samples each layer’s density curve.

Each layer is mapped through its density curve (including per-layer gamma factors from the profile), then remapped to **CMY dye densities**:

- Blue layer → Yellow dye density
- Green layer → Magenta dye density
- Red layer → Cyan dye density

#### Couplers (why they matter)

Data-sheet curves alone are typically not enough to reproduce convincing film behavior. Couplers are a major part of “why film looks like film”, and the upstream agx-emulsion model treats them as first-class components.

Film-Juicer models two important coupler families:

- **Masking couplers**: reduce spectral cross-talk between formed dyes and tend to increase apparent saturation. In practice this manifests as an “orange mask” character in the developed negative and is modeled as additional spectral absorption/offset terms.
- **DIR (Direct Inhibitor Release) couplers**: introduce inter-layer development interactions (inhibition) that can increase saturation/contrast. When enabled, they perturb effective log exposure before density sampling. Optional spatial diffusion (in micrometers, scaled by `Camera film format (mm)`) behaves like an adjacency effect and can change local contrast/sharpness.

### 4) Print Simulation (Optional)

If `Bypass print` is disabled, the model simulates enlarger + paper.

1) **Negative transmittance**

“Density” is basically “how much dye is there”. More dye means less light passes through the negative.

Film-Juicer converts CMY dye densities into “light transmitted vs wavelength” and uses that to compute what spectrum hits the paper.

2) **Enlarger light + filtration**

- Choose the enlarger light source (`Enlarger illuminant`).
- Apply Y/M/C filtration (filter set + neutral baseline + your Y/M/C shifts).
- In spectral terms, this is “enlarger spectrum multiplied by the three filter transmittance curves”.

3) **Paper exposure + development**

Paper “sees” the filtered enlarger light after it passes through the negative, then integrates that spectrum against the paper’s own sensitivity curves to get three paper-layer exposures.

`Print exposure` scales exposure energy; `Print preflash` adds a base exposure term. The result is developed through paper response curves into print CMY density.

### 5) Scanner / Viewing Model (Density → RGB)

Whether you are scanning a negative (`Bypass print = true`) or a print (`Bypass print = false`), Film-Juicer converts dye density back into color by spectral integration:

1) Convert dye density into “how much light gets through” per wavelength.
2) Convert that spectrum into XYZ under the viewing/scanner illuminant.
3) Convert XYZ into your output RGB space, then (optionally) apply output encoding.

Viewing/scanner illuminant:

- Currently this is **profile-driven** (chosen stock/paper metadata + internal defaults) rather than a user-facing control.

Scanner optics operate in image space:

- `Scanner lens blur (px)`: Gaussian blur
- `Scanner unsharp mask`: post-blur sharpening
- `Glare`: veiling glare model with optional compensation removal controls

## Controls (Physical Semantics)

This is a map of the most important controls in physical terms.

### Camera / Exposure

- `Camera auto exposure`: enables scene metering to set an exposure offset before film exposure.
- `Camera metering`: metering strategy (center-weighted vs median).
- `Exposure Compensation Ev`: exposure of the virtual negative in stops (`+1 EV` doubles exposure, `-1 EV` halves it).
- `Camera film format (mm)`: sets the physical scale for μm→pixel conversions (DIR spatial diffusion, halation radii, etc.).

### Spectral / Stock

- `Film stock`: selects the negative profile (sensitivities, dye densities, H–D curves, coupler metadata).
- `Spectral upsampling`: chooses the RGB→SPD reconstruction method.
- `Reference illuminant`: the “white light” the model uses when building spectral tables and interpreting the reconstructed spectrum.
  - Options: D65 / D55 / D50 / TH-KG3-L / T / K75P / Equal energy
  - Notes: `TH-KG3-L` is a tungsten-halogen source filtered by a KG3 heat filter (used to approximate enlarger-style spectra).

### Negative development interactions

- `DIR couplers`: models inter-layer development interactions; use this for characteristic “film crosstalk” behavior rather than post RGB channel mixing.

### Print

- `Bypass print`: when enabled, skips enlarger + paper and scans the negative directly.
- `Print paper`: selects the paper profile.
- `Enlarger illuminant`: the enlarger light source spectrum used for print exposure.
  - Options: D65 / D55 / D50 / TH-KG3-L / T / K75P / Equal energy
- `Enlarger dichroics`: selects the dichroic filter set; this also controls the neutral baseline behavior of the Y/M/C wheels.
- `Enlarger Y/M/C`: filtration shifts in enlarger “steps” around the neutral baseline (0 = neutral; positive increases filtration, negative decreases).
  - Neutral is defined as a starting point intended to render an 18% gray target neutral for the current paper/illuminant/film combination (the enlarger step model uses 170 steps).
- `Print exposure`: scalar on print exposure energy.
- `Print preflash`: adds a base exposure to paper (toe lift / shadow behavior).
- `Print exposure compensation`: keeps mid-gray behavior consistent when paper/illuminant/filtration changes.
- `Print Dmin`: minimum-density factor of the print paper (makes “paper white” less white).

### Scanner / Output

- `Scanner lens blur (px)`: Gaussian blur sigma in pixels.
- `Scanner unsharp mask`: (sigma px, amount) applied after scanner blur.
- `Scanner use LUT` + `Scanner LUT resolution`: use a precomputed lookup table for speed; higher resolutions are more accurate but heavier to build/use.
- `Output color space`: where you want the result to land (e.g. sRGB/Rec.709/BT.2020/DWG/ACES2065-1).
- `Apply output CCTF`: apply output gamma/transfer function for display delivery.
- `Output linear pass-through`: keep output linear for managed pipelines.

### Artifacts

- `Halation`: scattering in the film stage (physically earlier than “glow” post effects).
  - `Scattering strength (%)` / `Scattering size (μm)`: controls how much and how far light scatters before the main halo.
  - `Halation strength (%)` / `Halation size (μm)`: controls the strength and spread of the halo itself.
- `Grain`: stochastic density modulation; key controls include `Grain Amount (EV)`, `Grain Size (px)`, `Grain Sharpness`, `Grain Chroma`, and `Grain Texture`.
- `Gate weave / dust / scratches`: gate/transport artifacts.
- `Glare`: veiling glare / flare behavior in the scanner/view stage.
  - `Glare percent`, `Glare roughness`, `Glare blur sigma (px)` tune the glare model.
  - Compensation removal controls are intended to remove a modeled glare-compensation term over a density range (useful for matching certain paper profiles).

## Recommended Resolve Workflows

### Managed pipeline (RCM / ACES)

Goal: keep Juicer operating on scene-linear values, and keep color space transforms in the managed pipeline.

- Set `Input color space` to match the RGB values arriving at the OFX.
- Ensure the signal arriving at the OFX is **scene-linear**. If your working space is log-encoded (e.g. DaVinci Intermediate / ACEScct), insert a CST to convert into a linear encoding before Juicer.
- Keep `Decode input CCTF = off` unless you are feeding display-encoded `sRGB/Rec.709` or `BT.2020`.
- Prefer `Output linear pass-through = on`.
- Apply your timeline/output transforms outside Juicer (RCM/ACES handles it).

### Display-referred pipeline (not managed)

Goal: explicitly linearize on input and re-encode on output.

- Set `Input color space = sRGB / Rec.709` and enable `Decode input CCTF`.
- Choose `Output color space` and keep `Apply output CCTF = on` for display delivery.

## Performance / Quality Trade-offs

Primary cost drivers:

- Spectral reconstruction + scan mapping (especially with spectral reconstruction enabled, scan LUT disabled, or very high LUT resolution).
- Grain / halation / glare (stochastic and/or multi-pass blurs).

Tuning guidance:

- Start with artifacts off; dial the base negative/print/scanner behavior first.
- Use `Scanner use LUT = on` for interactive work; increase LUT resolution for higher fidelity if needed.

## Validation & Common Pitfalls

### 1) Color management mismatch

Most “unexpected contrast / saturation / density” reports reduce to **double-decoding**, **double-encoding**, or running the simulation on a **log-encoded** signal.

Baseline for a managed pipeline (RCM/ACES):

- Ensure the signal into Juicer is scene-linear (use CST if your working encoding is log).
- `Decode input CCTF = off` (unless feeding display-encoded sRGB/Rec.709 or BT.2020)
- `Output linear pass-through = on`

Baseline for a display-referred pipeline:

- `Decode input CCTF = on` when feeding `sRGB / Rec.709` or `ITU-R BT.2020` display-encoded values
- `Apply output CCTF = on` when delivering display-encoded output

### 2) Print stage is bypassed

Make sure you are actually running the print stage:

- `Bypass print = off`

Also note that enlarger filtration is defined as a **shift around a neutral baseline**. If you change paper / film / enlarger illuminant, the neutral baseline can change; re-evaluate filtration under the new combination (or leave `Print exposure compensation` enabled to keep mid-gray behavior stable).

### 3) Input gamut / metamerism limits

Even though the internal pipeline is spectral, your input is still RGB. Spectral reconstruction from RGB is inherently underdetermined:

- Narrow-band emitters / display primaries cannot be reconstructed uniquely from 3-channel RGB (metamerism: different spectra can match the same RGB).
- If you are chasing extreme saturation, feed a **wide-gamut, scene-linear** signal (DWG/ACES/BT.2020), not a clipped Rec.709 delivery image.

### 4) Performance baseline

Start from a “clean” baseline and add complexity:

- Disable grain/halation/glare first.
- Keep `Scanner use LUT = on` for interactive work; raise LUT resolution only when you need it.

### 5) Load failures

- Ensure your GPU meets the CUDA target (SM75+/Turing or newer) and that you installed a CUDA-enabled build of the plug-in.
- If Resolve fails due to missing CUDA runtime DLLs, ensure the required `cudart64_*.dll` is available in Resolve’s DLL search path (commonly shipped alongside the plug-in binary in `Contents/Win64/`, or installed system-wide).
- If film/paper menus are empty, confirm `Juicer.ofx.bundle/Contents/Resources/` is present next to the plug-in binary.

## References

- Upstream modeling reference: [agx-emulsion]
- Parity/porting notes (developer-facing): `agx-documentation.md`
- Contributing/build notes (developer-facing): `DEVELOPING.md`
- Background reading (as referenced by agx-emulsion):
  - Giorgianni, Madden — *Digital Color Management* (2nd ed., 2008)
  - Hunt — *The Reproduction of Colour* (6th ed., 2004)
  - Mallett, Yuksel — “Spectral Primary Decomposition for Rendering with sRGB Reflectance” (2019)

[agx-emulsion]: https://github.com/andreavolpato/agx-emulsion
