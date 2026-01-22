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

## What It Simulates

Film-Juicer explicitly models the “photographic chain” rather than applying a 3D LUT:

- **Spectral domain**: internal spectral tables are sampled at **81 wavelengths** (380–780 nm in 5 nm steps).
- **Units**: dye densities are optical densities (OD). “μm” controls are physical micrometers; they are converted to pixels using `Camera film format (mm)`.
- **Film negative**: exposure is formed by integrating the reconstructed SPD against **film layer sensitivities**, then developed through **H–D density curves** into **CMY dye densities**.
- **Print (optional)**: negative density modulates an **enlarger illuminant** filtered by **dichroic Y/M/C filtration**; the paper receives per-layer exposures and is developed through **paper density curves**.
- **Scanner/viewing**: dye density is converted back to tristimulus by Beer–Lambert transmittance and spectral integration against an illuminant + CMFs, then mapped into your target output color space (with optional output encoding).
- **Artifacts (optional)**: halation, grain, glare, blur/unsharp, and gate effects are applied at physically-relevant stages (not as a single post look).

## Color Pipeline (Input → Output)

### Input expectations

Film-Juicer expects **linear-light, scene-referred RGB** in the selected input primaries unless you enable decoding.

Supported input primaries:

- DaVinci Wide Gamut
- ITU-R BT.2020
- ACES2065-1
- sRGB / Rec.709

`Decode input CCTF`:

- When enabled, Film-Juicer decodes **BT.2020** and **sRGB/Rec.709** transfer functions to linear.
- For scene-linear encodings (DWG / ACES2065-1), leave it off.
- Film-Juicer does not decode log/scene-encoding curves like DaVinci Intermediate or ACEScct; if your node graph is in a non-linear working encoding, convert to a scene-linear signal (e.g. via Resolve CST) before the OFX.

Practical guidance:

- If your pipeline is managed (RCM/ACES), explicitly ensure the signal into Juicer is scene-linear (e.g. CST into linear DWG or ACES2065-1), then keep decoding off.
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

RGB does not uniquely determine a spectrum (metamerism). Film-Juicer reconstructs a plausible **spectral power distribution** (SPD) from RGB using one of two methods:

- `Spectral upsampling = Hanatos`: uses a LUT-based reconstruction (high fidelity when the LUT is applicable).
- `Spectral upsampling = Mallett`: uses a basis reconstruction (lower-dimensional approximation).

The reconstructed SPD is then used for all “spectral” steps: film exposure, illuminant interactions, and spectral integration for scanning.

Implication: highly non-standard emitters (narrow-band LEDs, lasers, display primaries) may not be reproduced with true spectral accuracy because the input is still 3-channel RGB.

### 2) Film Exposure (Per-Layer Raw)

The film negative is modeled as three spectrally sensitive layers. Conceptually:

- Reconstructed scene SPD: `S(λ)`
- Film sensitivities: `s_B(λ)`, `s_G(λ)`, `s_R(λ)`

Layer exposures (film “raw”) are formed by spectral contraction:

`E_k = ∫ S(λ) · s_k(λ) dλ` for k ∈ {B,G,R}

`Exposure Compensation Ev` applies as a physical exposure scalar:

`E_k ← E_k · 2^EV`

### 3) Film Development (Raw → Density CMY)

Film development is modeled with H–D curves in **log exposure**.

Log exposure is computed with a small epsilon to avoid `log(0)`:

`logE_k = log10(max(E_k, 0) + 1e-10)`

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

1) **Negative transmittance** via Beer–Lambert optical density:

- Per-wavelength density: `D(λ) = D_C·ε_C(λ) + D_M·ε_M(λ) + D_Y·ε_Y(λ) + base(λ)`
- Transmittance: `T(λ) = 10^{-D(λ)}`

2) **Enlarger illumination** filtered by dichroic filtration:

- Enlarger illuminant SPD: `E_e(λ)` (set by `Enlarger illuminant`)
- Dichroic filters (set family + neutral baseline + user shifts): `fY(λ), fM(λ), fC(λ)`
- Filtered enlarger SPD: `E_f(λ) = E_e(λ) · fY(λ) · fM(λ) · fC(λ)`

3) **Paper exposure + development**

Paper receives exposures by contracting `E_f(λ) · T(λ)` against paper sensitivities. `Print exposure` scales exposure energy; `Print preflash` adds a base exposure term. The result is developed through paper density curves into print CMY density.

### 5) Scanner / Viewing Model (Density → RGB)

Whether you are scanning a negative (`Bypass print = true`) or a print (`Bypass print = false`), Film-Juicer converts dye density back into color by spectral integration:

1) Form per-wavelength transmittance `T(λ)` from CMY density (same Beer–Lambert form as above).
2) Integrate against spectral tables to obtain XYZ under the viewing/scanner illuminant.
3) Apply chromatic adaptation + matrix to obtain output RGB, then apply output encoding if configured.

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
- `Exposure Compensation Ev`: multiplies film exposure by `2^EV`.
- `Camera film format (mm)`: sets the physical scale for μm→pixel conversions (DIR spatial diffusion, halation radii, etc.).

### Spectral / Stock

- `Film stock`: selects the negative profile (sensitivities, dye densities, H–D curves, coupler metadata).
- `Spectral upsampling`: chooses the RGB→SPD reconstruction method.
- `Reference illuminant`: selects the illuminant used when building spectral tables and interpreting the reconstructed spectrum.
  - Options: D65 / D55 / D50 / TH-KG3-L / T / K75P / Equal energy
  - Notes: `TH-KG3-L` is a tungsten-halogen source filtered by a KG3 heat filter (used to approximate enlarger-style spectra).

### Negative development interactions

- `DIR couplers`: models inter-layer development interactions; use this for characteristic “film crosstalk” behavior rather than post RGB channel mixing.

### Print

- `Bypass print`: when enabled, skips enlarger + paper and scans the negative directly.
- `Print paper`: selects the paper profile.
- `Enlarger illuminant`: illuminant SPD used for print exposure.
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
- `Scanner use LUT` + `Scanner LUT resolution`: trades accuracy vs speed for density→color mapping.
- `Output color space`: target RGB primaries/matrix.
- `Apply output CCTF`: apply display encoding (gamma/OETF).
- `Output linear pass-through`: keep output linear for managed pipelines.

### Artifacts

- `Halation`: scattering in the film stage (physically earlier than “glow” post effects).
  - `Scattering strength (%)` / `Scattering size (μm)`: controls the pre-halation scatter (Gaussian sigma in μm).
  - `Halation strength (%)` / `Halation size (μm)`: controls the halo component (Gaussian sigma in μm).
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

- Spectral pipeline (especially when SPD reconstruction is enabled and when scanner LUTs are disabled or high-res).
- Grain / halation / glare (stochastic and/or multi-pass blurs).

Tuning guidance:

- Start with artifacts off; dial the base negative/print/scanner behavior first.
- Use `Scanner use LUT = on` for interactive work; increase LUT resolution for higher fidelity if needed.

## Validation & Common Pitfalls

### 1) “My color management is wrong”

Most “this looks wrong” reports reduce to **double-decoding** or **double-encoding**.

Baseline for a managed pipeline (RCM/ACES):

- Ensure the signal into Juicer is scene-linear (use CST if your working encoding is log).
- `Decode input CCTF = off` (unless feeding display-encoded sRGB/Rec.709 or BT.2020)
- `Output linear pass-through = on`

Baseline for a display-referred pipeline:

- `Decode input CCTF = on` when feeding `sRGB / Rec.709` or `ITU-R BT.2020` display-encoded values
- `Apply output CCTF = on` when delivering display-encoded output

### 2) “Y/M/C doesn’t do anything”

Make sure you are actually running the print stage:

- `Bypass print = off`

Also note that enlarger filtration is defined as a **shift around a neutral baseline**. If you change paper / film / enlarger illuminant, the neutral baseline can change; re-evaluate filtration under the new combination (or leave `Print exposure compensation` enabled to keep mid-gray behavior stable).

### 3) “The image is unexpectedly clipped / desaturated”

Even though the internal pipeline is spectral, your input is still RGB. Spectral reconstruction from RGB is inherently underdetermined:

- Narrow-band emitters / display primaries cannot be reconstructed uniquely from 3-channel RGB (metamerism).
- If you are chasing extreme saturation, feed a **wide-gamut, scene-linear** signal (DWG/ACES/BT.2020), not a clipped Rec.709 delivery image.

### 4) “Performance tanks / playback is not interactive”

Start from a “clean” baseline and add complexity:

- Disable grain/halation/glare first.
- Keep `Scanner use LUT = on` for interactive work; raise LUT resolution only when you need it.

### 5) “Resolve can’t load the plug-in”

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
