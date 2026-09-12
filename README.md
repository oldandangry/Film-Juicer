<h1 align="center">Film-Juicer</h1>

<p align="center">
  <img src="Resources/banner.jpg" alt="Film-Juicer" width="1280">
</p>

<p align="center">
  <strong>Spectral film emulation for DaVinci Resolve</strong>
</p>

> [!IMPORTANT]
> Film-Juicer is built on **[spektrafilm](https://github.com/andreavolpato/spektrafilm)** by [Andrea Volpato](https://github.com/andreavolpato). Its photographic model, profile data, and reference implementation are the foundation of this plug-in. Film-Juicer would not exist without that work. Please consider adding your support to this project.

Film-Juicer brings the exposure, development, printing, and scanning of photographic film into DaVinci Resolve. It is a Windows OpenFX plug-in, with rendering handled by NVIDIA CUDA.

You choose a capture stock, expose it, and decide whether to scan the film directly or print it onto another photosensitive material first. Exposure, filtration, grain, and diffusion act at their respective stages, so changing how the negative is exposed also changes what the enlarger and scanner have to work with.

Underneath, the model works with light sampled at 81 wavelengths, from 380 to 780 nm. It uses the stock's spectral sensitivity to calculate exposure, develops that exposure into dye density, and calculates how light passes through the resulting material. LUTs accelerate parts of this process, including spectral reconstruction and scanning. The overall response comes from the interaction of these stages.

This is quite a lot of machinery to put between two RGB images. The attraction is being able to work with the photographic process itself, and follow an adjustment through to its consequences.

> [!NOTE]
> **Film-Juicer 1.0.0** is available from [GitHub Releases](https://github.com/oldandangry/Film-Juicer/releases). This is the first major release version.
> ## Support Film-Juicer
> If Film-Juicer is useful to you and you'd like to support its development:
>[☕ Buy me a coffee](https://buymeacoffee.com/oldandangry)

## Features

- 20 capture-film profiles and 8 paper or print-film profiles, derived from spektrafilm.
- Direct and optical-print scanning for negative and positive capture film.
- Hanatos 2025 and Mallett 2019 spectral reconstruction.
- Stock-specific sensitivity, development curves, and spectral dye density.
- DIR couplers for interactions within and between film layers, including spatial effects.
- Grain formed in film-density space, with physical format scaling and creative controls.
- Emulsion scatter, halation, and camera or enlarger diffusion.
- Spectral enlarger filtration, print exposure, and preflash.
- Scanner black/white correction, print-scan glare, lens blur, and sharpening.
- Gate weave, with separate film and gate dust and scratches.

## Requirements and installation

You need Windows 10/11 x64, DaVinci Resolve configured for CUDA rendering, and an NVIDIA Turing GPU or newer (compute capability 7.5 or higher). Production rendering requires CUDA; there is no CPU, OpenCL, or Metal render path.

1. Download the Windows installer from [GitHub Releases](https://github.com/oldandangry/Film-Juicer/releases).
2. Run the installer, then restart Resolve.
3. Find **Juicer** under **OpenFX → Negative-juice**.

## Getting started

Start with a shot whose exposure and colour you know reasonably well. A familiar face and a few highlights will tell you more than an image already carrying six creative transforms.

1. Prepare a scene-linear signal and set **Input color space** to match its primaries. Leave **Decode input CCTF** off when the input is already linear. See the colour-management section below for the supported encodings.
2. Select a **Film stock** and **Scan route**. The defaults are Portra 400 and Negative print scan, printing onto Portra Endura.
3. Set capture exposure with **Exposure Compensation Ev**. Camera auto exposure is on by default; turn it off when you want to set exposure manually without the meter responding to changes in the image.
4. On a print route, choose **Print paper**, then adjust **Print exposure** and the enlarger Y/M/C offsets to balance the print.
5. Set **Output color space** and **Apply output CCTF** for the signal your next node expects. The default output is encoded sRGB, so change it if you are returning to a different working space.
6. Once the colour and tonal response are in place, set the film format and refine grain, diffusion, halation, and scanner finishing.

Get stock, exposure, route, and colour management right first. Adding dirt to a colour-management problem gives you a dirty colour-management problem. The dust has done its job. The rest is still up to you.

## Colour management

Film-Juicer needs to know what the incoming RGB values mean. Selecting the right gamut is only half of that: the transfer function must also be accounted for.

The exposure model expects scene-linear light. DaVinci Intermediate and ACEScct are log encodings, and selecting DWG or ACES primaries does not decode them. In the UI, **CCTF** means *colour component transfer function*: the encoding or decoding applied to each RGB channel.

| Input color space | With Decode input CCTF off | With Decode input CCTF on |
| --- | --- | --- |
| DaVinci Wide Gamut | Linear DWG | No additional decoding |
| ITU-R BT.2020 | Linear BT.2020 | BT.2020 transfer-function decoding |
| ACES2065-1 | Linear ACES AP0 | No additional decoding |
| sRGB / Rec.709 | Linear sRGB/Rec.709 primaries | sRGB transfer-function decoding |

The combined **sRGB / Rec.709** label refers to their shared primaries. Its decoder uses the sRGB curve, which is different from gamma 2.4 and the BT.709 camera encoding. Linearise other encodings in Resolve before they enter the plug-in.

### Working in DWG / DaVinci Intermediate

For a node receiving DWG/Intermediate, use a Color Space Transform before Juicer to convert **DaVinci Wide Gamut / DaVinci Intermediate** to **DaVinci Wide Gamut / Linear**. Set Juicer's input to **DaVinci Wide Gamut**, with input decoding off.

To return DWG/Intermediate, select **DaVinci Wide Gamut Intermediate** as the output and enable **Apply output CCTF**. Your downstream nodes then receive that encoding. If you disable the checkbox, the output uses DWG primaries with linear values instead.

The same principle applies to ACES: convert the incoming encoding and primaries to a supported linear input, then make the return transform explicit. For example, ACEScct uses AP1 primaries, while Film-Juicer's **ACES2065-1** entry uses AP0. Both the gamut and encoding need to match.

These conversions establish the signal format. They do not decide where the film rendering belongs in your grade or how a downstream display transform should treat it. Check the whole chain, particularly if a display or look transform is already present.

### Output and range

Output choices are sRGB, DCI-P3, Display P3, Adobe RGB (1998), BT.2020, ProPhoto RGB, ACES2065-1, DaVinci Wide Gamut Intermediate, and Rec.709. **Apply output CCTF** enables the selected space's encoding; with it off, Film-Juicer returns linear RGB in those primaries. ACES2065-1 is linear in either case.

Rec.709 output uses gamma 2.4, even though the input's combined sRGB / Rec.709 entry decodes sRGB. Keep that difference in mind if you are sending an encoded image through the plug-in.

> [!IMPORTANT]
> Final RGB values are clipped to **0–1 after output encoding**, or after colour conversion when encoding is disabled. Turning off **Apply output CCTF** does not provide an unrestricted linear output. Any values clipped here are unavailable to downstream nodes, so check the output range before relying on later highlight or gamut recovery.

Decoding a display-encoded image removes its transfer function; it cannot undo a tone map or recover scene information already lost upstream. For scene-based exposure work, feed Film-Juicer the image before those operations.

## Direct scan or optical print?

The route determines which developed material reaches the scanner.

| Scan route | Path |
| --- | --- |
| **Negative direct scan** | Negative film → scanner |
| **Negative print scan** | Negative film → enlarger → paper or print film → scanner |
| **Positive direct scan** | Positive film → scanner |
| **Positive print scan** | Positive film → enlarger → paper or print film → scanner |

The capture profile determines the negative or positive behaviour. Film-Juicer resolves the route to that profile while retaining your choice of direct scan or printing.

Direct scanning gives you the capture material's response through the scanner. A print route adds another material, with its own sensitivity, development response, and dyes. The **Print paper** menu includes both photographic papers and motion-picture print films.

Printing is a substantial part of the colour rendering. Light from the enlarger passes through the developed capture film and exposes the print layers. Change the capture stock, enlarger filtration, or print medium and you change that interaction. Expect to rebalance when moving between them.

## Inside the photographic process

![Film-Juicer pipeline: spectral exposure, film development, direct or optical-print scanning, and RGB output.](Resources/readme/photographic_pipeline.svg)

The order matters. Camera diffusion spreads exposure before film development; scanner blur acts on the developed image near the end. Grain becomes part of the film density and is carried through printing and scanning. These operations can produce different results even when they appear to soften or texture the same image.

### Spectral exposure

RGB gives us three values. Photographic layers respond to a range of wavelengths, so the model needs a spectral representation of the incoming light before it can expose them.

**Hanatos 2025**, the default, uses a precomputed spectral reconstruction LUT. **Mallett 2019** uses a compact set of spectral basis functions defined for linear sRGB. That sRGB basis makes Mallett a more constrained model for wide-gamut colours; Hanatos is the default starting point for those workflows.

Neither method can recover the original scene spectrum from RGB. Many different spectra can produce the same three values. Reconstruction supplies a plausible spectrum that the film model can work with, and its assumptions are part of the result.

The selected stock's spectral sensitivities determine the exposure of the red-, green-, and blue-sensitive layers. Camera UV/IR filtration modifies that sensitivity, while the exposure controls set how much light reaches the model.

### Profiles and development

A stock profile describes several stages of photographic behaviour. The Portra 800 data makes the distinction easier to see:

<p align="center">
  <img src="Resources/readme/kodak_portra_800_spektrafilm_with_spectral_density.svg" alt="Portra 800 profile showing spectral sensitivity, characteristic curves, and spectral density" width="100%">
</p>

From left to right:

- **Spectral sensitivity** describes how each layer responds across the wavelength range.
- **Characteristic curves** relate exposure to developed density, including the toe, contrast, and shoulder.
- **Spectral density** describes how the developed material absorbs light at different wavelengths.

The density panel shows the cyan, magenta, and yellow contributions (**C**, **M**, **Y**), the processed unexposed material (**Min**), and a neutral midscale reference (**Mid**). Min and Mid are reference spectra, not two extra dye layers. The figure retains negative dye contributions associated with the profile's masking-coupler model and leaves missing source samples blank.

These figures show the bundled profile data. In particular, the characteristic panel plots the stored curves; it is not a reconstruction of the separate fitted layer model.

During development, layer exposures become dye densities. **DIR**, short for developer-inhibitor-releasing couplers, models how development in one layer inhibits development within that layer and in the others. Its spatial component spreads the interaction across neighbouring areas, affecting colour separation, local contrast, and apparent sharpness.

For grading, this means an exposure change can alter more than brightness. It moves the image through the stock's response and changes the densities handed to the next stage.

### Development tuning

The collapsed **Tuning** group sits immediately above **Output encoding**. **Film gamma factor** adjusts capture-film development on every route from 0.05 to 4.0. **Print gamma factor** adjusts the selected print medium from 0.5 to 2.0 and is enabled only on print routes. Both controls reset to 1.0, use a 0.5–2.0 slider display range, and accept in-range typed values without quantizing them to the 0.05 drag increment.

Values below 1 reduce development contrast; values above 1 increase it. Neither control is animated. Switching to a direct route disables the print control but retains its authored value for the next print route. The host snapshot validates the bounded Double values; film gamma is then retained once as Float32 for the existing CUDA film tables, while print gamma remains Float64 during fitted-model derivation before the resulting density table is uploaded as Float32.

At exactly print gamma 1, print development uses the fitted stock model's inactive mapping. Other accepted values apply the reference global gamma morph to that same fitted model. This intentionally corrects the default print-development model and its interpolation knots, along with the stock DIR defaults used by film development. Print balance, preflash, scanner bounds, and other stock-anchored corrections continue to use the authored stock tables. These reference corrections can change an established render even when both new controls remain at 1.

### Scatter, halation, and diffusion

Emulsion scatter spreads light within the film. Halation models light returning through the emulsion after reflection, contributing to the coloured spread around bright features. Both act on film-layer exposure before development.

**Add halation** enables the scatter/halation stage and is off by default. Separate amount and spatial-scale controls adjust scatter and halation, with the film profile supplying the halation response. The scale controls change the spread; the amount controls change its contribution.

Camera and print diffusion are separate stages, with Glimmerglass, Black Pro-Mist, Pro-Mist, and CineBloom families. Camera diffusion acts before film development. Print diffusion acts on the enlarger exposure before print development. Core, halo, bloom, warmth, and scale controls let you adjust the selected family's response.

Start by choosing where you want the diffusion to happen. That decision usually matters more than another small adjustment to its radius.

### Grain, format, and physical artifacts

Grain is generated in film-density space. Its appearance depends on exposure and density, particle statistics, layer structure, and the subsequent print and scan. Fine, Medium, and Coarse presets provide starting points; **Grain Amount, Size, Sharpness, Chroma, and Texture** are the main creative controls. Advanced settings expose particle area, sublayers, uniformity, dye-cloud blur, size mixtures, and micro-structure.

**Camera film format (mm)** sets the longest dimension of the simulated capture area. It is a physical length, not a stock or gauge selector: a value of 35 means a 35 mm long image area. The current range is 8–120 mm. This scale feeds grain, DIR, diffusion, and halation, so establish it before refining their appearance.

Film dust and scratches are attached to the virtual strip. Gate artifacts are attached to the gate, with weave introducing movement between the two. Keeping them separate makes their behaviour under motion more convincing. Gate weave defaults to zero.

Dust and scratches remain under refinement in this release candidate. You can already add enough dirt to suggest that the archive was stored in a cement mixer, but restraint tends to survive client review better.

### The enlarger and print medium

The enlarger combines its illuminant with spektrafilm's Custom dichroic filter model. **Enlarger Y/M/C offsets** are expressed in Kodak CC units around the neutral calibration for the selected combination. At zero, the calibration's filtration is still in place.

**Print exposure** is a multiplier: 1.0 is the baseline and 2.0 doubles the exposure. It is different from the camera's exposure compensation, which is expressed in stops. **Print preflash** adds a uniform exposure before print development, changing the response most noticeably where the image contributes little exposure. **Print exposure compensation** uses a simulated mid-grey reference to compensate for the camera exposure adjustment when calculating print exposure.

The selected print medium then develops that exposure according to its own response. Here is Kodak Ektacolor Edge:

<p align="center">
  <img src="Resources/readme/kodak_ektacolor_edge_spektrafilm_with_spectral_density.svg" alt="Ektacolor Edge print profile showing spectral sensitivity, characteristic curves, and spectral density" width="100%">
</p>

The panels describe the same properties as the Portra figure, but for the receiving material. Together, the two profiles describe both sides of the printing process. There is an appealingly stupid amount of work involved in recreating the act of shining a lamp through a negative. At least you can turn the room lights on.

### Scanning and finishing

The scanner calculates the colour of the developed film or print under its viewing illuminant, converts it through CIE XYZ, and returns RGB in the selected output space. The current CUDA renderer uses a spectral scanner LUT for this conversion. **Scanner LUT resolution** controls its sampling density, with a cost in memory and preparation time.

**Scanner black correction** and **Scanner white correction** set the route's black/white normalisation. Print routes can also add **Glare**, which introduces veiling light at the scan stage. **Scanner lens blur** and **Scanner unsharp mask** provide final softness and sharpening before output encoding.

These controls are useful for finishing the scan. If you are trying to change how the film was exposed or how the print was balanced, return to those earlier stages first.

## Included stocks

### Capture film — 20 profiles

- Fujifilm C200, Pro 400H, Provia 100F, Velvia 100, and X-Tra 400.
- Kodak Ektachrome 100, Ektar 100, Gold 200, Kodachrome 64, Ultramax 400, and Verita 200D.
- Kodak Portra 160, 400, and 800, plus separate Portra 800 Push 1 and Push 2 profiles.
- Kodak Vision3 50D, 200T, 250D, and 500T.

### Print media — 8 profiles

- Fujifilm Crystal Archive Type II.
- Kodak Vision 2383 and Vision Premier 2393.
- Kodak Ektacolor Edge.
- Kodak Professional Endura Premier, Portra Endura, Supra Endura, and Ultra Endura.

The profiles inherit spektrafilm's measurements, fitted models, and reference conditions. They give each stock a defined response within the simulation. Real rolls, processing lines, paper batches, and scanners vary, so a stock name cannot promise an exact match to every physical example. Use the profiles as photographic materials to work with, and judge the resulting image in the context of your grade.

## Performance

Spectral calculations, spatial DIR, grain, diffusion, halation, and scanner finishing all contribute to render time. Resolution and the spatial extent of an effect matter too, so there is no single useful frame-rate claim for every combination.

For interactive work, establish the stock, route, exposure, and print balance first, then add the spatial effects you need. Keep scanner LUT resolution at its default while establishing the look. Judge grain and other small-scale detail at the intended output resolution; a reduced preview cannot show exactly how those features will resolve in the final image.

The GPU will suffer according to your ambitions.

## Building from source

The default Windows build is **Release-Clang**, using ClangCl from Visual Studio 2026, CUDA Toolkit 13.2, and C++20 for both host and CUDA code. It also requires OpenFX 1.4 headers, the OpenFX support library, and Eigen 3.4.

The project contains local dependency paths. Adjust those to your installation before building; cloning the repository alone does not supply a complete build environment.

From a configured developer environment, run:

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" juicer.sln /p:Configuration=Release-Clang /p:Platform=x64
```

The `Release` configuration is also available for MSVC v145. The linker writes `juicer.ofx` to the configured MSBuild output directory. A usable OFX bundle also needs its runtime resources; the binary alone is not the installed plug-in.

## Project status

Film-Juicer is under active development. Changes to the photographic model and its implementation can change existing renders, so treat version changes as something to check against your grade.

The aim is to bring the photographic model into Resolve with native CUDA rendering and controls suited to grading. Film-Juicer does not reproduce every option in the Python reference application, and its grain and artifact controls have their own implementation. The shared model is the foundation; the two applications should not be assumed to produce identical results under every setting.

More features from spektrafilm may be added as the plug-in develops. Metal support is planned, but there is no timeline yet.

## Credits and licence

**Andrea Volpato** created [spektrafilm](https://github.com/andreavolpato/spektrafilm), whose photographic model, profiles, and reference implementation make Film-Juicer possible. The reconstruction methods also draw on the work of **Johannes Hanika (Hanatos)** and **Ian Mallett and Cem Yuksel**.

The profile figures above are Film-Juicer visualisations of spektrafilm data. If the plug-in is useful to you, please visit and support the original project. A considerable amount of the work you are benefiting from happened there.

Film-Juicer software is licensed under **GPL-3.0-only**; see [LICENSE](LICENSE). Spektrafilm software is licensed under GPLv3. Spektrafilm profiles, LUTs, and their direct derivatives are separately licensed under **CC BY-SA 4.0**, with their upstream attribution and modification notices retained. See the [spektrafilm asset licence](https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt).

Other third-party components retain their respective licences and copyright notices. Product and company names identify the materials and models being referenced; Film-Juicer is not affiliated with or endorsed by their manufacturers or rights holders.

## Further reading

- [spektrafilm](https://github.com/andreavolpato/spektrafilm) — the reference implementation and photographic model.
- [Mallett and Yuksel — Spectral Primary Decomposition for Rendering with sRGB Reflectance](https://diglib.eg.org/items/bbffa865-e99c-4c1f-bd33-70102dc8af78).
- Giorgianni and Madden — *Digital Color Management*, 2nd edition.
- Hunt — *The Reproduction of Colour*, 6th edition.
