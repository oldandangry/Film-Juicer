# Experimental spectral update testing

This document records the visual acceptance boundary for the selective spektrafilm experimental integration on Film-Juicer's `testing` branch. It is not a claim of full experimental-pipeline parity and is not release acceptance for `main`.

## Integrated scope

The integration uses spektrafilm experimental revision `28bf883e1672e884307edc75852549376e13644e` as the authority for Arctic 2026 beta04 reconstruction, input and output gamut compression, the existing colour-profile cohort, neutral print calibration, and Gaussian-derived capture and print curves. Hanatos remains the default reconstruction method and Mallett remains available.

Film-Juicer intentionally retains its existing visual effects, spatial optics, and DIR implementation. In particular, this integration does not adopt experimental's new DIR/Langmuir behavior.

## Reference comparison configuration

The investigated Arctic comparisons used:

- 6048 x 4032 `dpreview-test.exr`, Rec.2020 linear input
- Kodak Portra 160 capture film
- Fujifilm Crystal Archive Type II print medium
- 35 mm camera-film long edge, resolving to approximately 5.787 micrometres per pixel
- sRGB-encoded output with input and output gamut compression enabled
- center-weighted auto exposure for the AE-on comparison
- no scanner black-point or white-point correction
- no glare, halation, grain, camera diffusion, or enlarger diffusion
- DIR and spatial DIR enabled

The Python Arctic references additionally disabled scanner unsharp. Film-Juicer's scanner-unsharp default is sigma `0.7` and amount `0.7`; it must be set explicitly to `0, 0` for the same straight-render boundary.

## Intentional DIR discrepancy

The pinned experimental Python pipeline resolves Portra 160 through its revised generic colour-negative DIR preset:

| Parameter | Spektrafilm experimental | Film-Juicer testing |
| --- | --- | --- |
| Same-layer RGB gamma | `0.5159, 0.5934, 0.2829` | `0.336, 0.319, 0.273` |
| R to G/B | `0.4032, 0.2488` | `0.353, 0.302` |
| G to R/B | `0.2227, 0.4340` | `0.154, 0.353` |
| B to R/G | `0.1829, 0.1799` | `0.168, 0.226` |
| Spatial tail | 200 micrometres at weight `0.03` | 200 micrometres at weight `0.06` |
| Negative-film donor response | Langmuir saturation, normalized `K=1` | Existing linear donor response |

Both use a 20 micrometre spatial-DIR core. The differing inhibition matrix, tail weighting, and donor response affect tone, colour separation, and edge-local contrast. Consequently, residual structure around chart edges is expected and must not be attributed to Arctic reconstruction without first matching the DIR contracts.

## Measured results

All values below compare stored sRGB pixel values at identical 6048 x 4032 coordinates. PSNR uses a unit code-value range.

| Comparison | PSNR | RGB RMS error |
| --- | ---: | --- |
| Hanatos Resolve AE off vs pinned main reference | 50.69 dB | `0.004403, 0.001450, 0.002029` |
| Arctic Resolve AE off vs experimental reference | 28.79 dB | `0.031865, 0.035747, 0.040834` |
| Arctic Resolve AE on vs experimental reference | 29.09 dB | `0.031060, 0.035707, 0.038154` |
| Arctic Resolve AE off, scanner unsharp disabled, vs experimental reference | 37.40 dB | `0.011554, 0.012182, 0.016260` |
| Controlled experimental AE-off render with Film-Juicer DIR and scanner unsharp `0.7, 0.7` vs Resolve | 89.19 dB | `0.0000545, 0.0000205, 0.0000151` |
| Controlled experimental AE-on render with Film-Juicer DIR and scanner unsharp `0.7, 0.7` vs Resolve | 77.89 dB | `0.0001469, 0.0001184, 0.0001148` |

The no-unsharp Resolve render isolates the remaining experimental-DIR mismatch. Conversely, matching Film-Juicer's retained DIR parameters and scanner-unsharp default closes essentially the entire gap. The small remainder is consistent with the expected CUDA FP32 versus NumPy reference precision, Gaussian implementation, auto-exposure sampling, TIFF quantization, and ICC/export differences.

No pixel shift, crop, channel-order, orientation, or bit-depth discrepancy was found. The Arctic NPY, Kodak Portra 160 profile, and Fujifilm Crystal Archive Type II profile are byte-identical to the pinned experimental sources. The selected neutral-filter leaf is also identical at `C/M/Y = 0.0/77.8508/75.9537`.

## Acceptance boundary and follow-up

These measurements support the Arctic reconstruction and imported-asset implementation when the surrounding Film-Juicer contract is matched. They do not establish parity with spektrafilm experimental's revised DIR chemistry.

A future DIR update should be planned as a separate behavior-replacement phase. That plan needs to resolve the new stock presets, Langmuir donor/receiver behavior, spatial-tail change, controls and saved-project implications, numerical fixtures, CUDA ownership, performance, and host visual acceptance before implementation or promotion to `main`.
