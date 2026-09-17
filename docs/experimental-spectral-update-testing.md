# Experimental spectral update testing

This document records the visual and numerical acceptance boundary for the selective spektrafilm experimental integration on Film-Juicer's `testing` branch. It is not a claim of full experimental-pipeline parity and is not release acceptance for `main`.

## Integrated scope

The integration uses spektrafilm experimental revision `28bf883e1672e884307edc75852549376e13644e` as the authority for Arctic 2026 beta04 reconstruction, input and output gamut compression, the existing colour-profile cohort, neutral print calibration, Gaussian-derived capture and print curves, and supported-colour DIR behavior. Hanatos remains the default reconstruction method and Mallett remains available.

Film-Juicer retains its product visual effects and unrelated spatial optics. Its CUDA-only capture-film DIR path now adopts the target's stock presets, negative donor Langmuir response, positive receiver Langmuir response, spatial mixture, and supported controls. It does not add B&W DIR, print-medium DIR, experimental film-convert behavior, or a CPU production renderer.

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

## Historical pre-DIR discrepancy

The table below describes the earlier `testing` implementation before the DIR replacement. It is retained to explain the already-published Resolve comparisons; it no longer describes the current implementation. The pinned experimental Python pipeline resolved Portra 160 through its revised generic colour-negative DIR preset:

| Parameter | Spektrafilm experimental | Film-Juicer testing |
| --- | --- | --- |
| Same-layer RGB gamma | `0.5159, 0.5934, 0.2829` | `0.336, 0.319, 0.273` |
| R to G/B | `0.4032, 0.2488` | `0.353, 0.302` |
| G to R/B | `0.2227, 0.4340` | `0.154, 0.353` |
| B to R/G | `0.1829, 0.1799` | `0.168, 0.226` |
| Spatial tail | 200 micrometres at weight `0.03` | 200 micrometres at weight `0.06` |
| Negative-film donor response | Langmuir saturation, normalized `K=1` | Existing linear donor response |

Both used a 20 micrometre spatial-DIR core. The differing inhibition matrix, tail weighting, and donor response affected tone, colour separation, and edge-local contrast. That mismatch motivated the current replacement and remains relevant only to the historical measurements below.

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

At the time of capture, the no-unsharp Resolve render isolated the then-remaining experimental-DIR mismatch. Conversely, matching Film-Juicer's former DIR parameters and scanner-unsharp default closed essentially the entire gap. The small remainder was consistent with CUDA FP32 versus NumPy reference precision, Gaussian implementation, auto-exposure sampling, TIFF quantization, and ICC/export differences. These rows have not been relabelled as post-replacement acceptance results.

No pixel shift, crop, channel-order, orientation, or bit-depth discrepancy was found. The Arctic NPY, Kodak Portra 160 profile, and Fujifilm Crystal Archive Type II profile are byte-identical to the pinned experimental sources. The selected neutral-filter leaf is also identical at `C/M/Y = 0.0/77.8508/75.9537`.

## Current DIR contract and numerical evidence

The current implementation resolves the target's still-negative, daylight-cine, tungsten-cine, and positive/stock-override presets from the bundled profile metadata. Negative film applies Langmuir saturation to each donor before the RGB inhibition matrix. Positive film filters the signed donor arrival first and then applies the receiver Langmuir response. The default spatial mixture is a 20 micrometre core plus a 200 micrometre tail at weight `0.03`; a zero core disables all spatial filtering. Small components use the pinned reflect-boundary FIR radius `int(3*sigma + 0.5)` evaluated from the canonical FP32 sigma in double precision, including the identity endpoint. Large components retain the strict replicate-boundary YVV operator.

The local implementation evidence is generated directly from revision `28bf883e1672e884307edc75852549376e13644e`. Its [fixture manifest](../.juicer-local/validation/experimental-dir/fixtures/manifest.json) records the source and asset closure; the focused Release results are retained under [the experimental DIR validation workbench](../.juicer-local/validation/experimental-dir/). The demonstrated default pointwise maximum absolute error is `1.78814e-07` for Kodak Portra 160 and zero for Fujifilm Provia 100F. Host curve construction matched 14 valid target cases within `1e-6`, rejected the two intentionally non-monotonic cases, and produced two intentionally inactive cases. Exact operator/radius fixtures, including the `float32(5/6)` endpoint and the neighbors of sigma 3, pass independently of image tolerances. These results establish parity only for the generated supported-colour cases and their stated FP32 budgets.

Donor arrivals may be signed after the spatial operator; negative lobes are therefore valid inputs to the positive receiver stage. An exact receiver pole or a nonfinite receiver result is a GPU execution failure, while adjacent finite values remain subject to the documented conditioning budget. This is not a claim of near-pole FP32 parity for every unrestricted input.

Invalid controls, nonrepresentable values, nonfinite derived constants, and non-monotonic pre-corrected curves fail synchronously during snapshot or recipe admission. The failure blocks the new frame and reports route, profile, field/channel/sample, and the violated requirement; an immutable frame admitted earlier remains valid. This path is distinct from a receiver failure discovered asynchronously on the GPU. A completed GPU failure is retained with the originating route, profile, recipe/descriptor hashes, context epoch, and channel mask, then reported when a later render polls that completed readback before following the existing fatal path. A failed final submission, context retirement, teardown, or the absence of a later poll can prevent an OFX notification even though the device failure was detected. A completion wait alone does not promise host presentation.

## Acceptance boundary and follow-up

The historical measurements continue to support the Arctic reconstruction and imported-asset implementation under their stated inputs. The new focused fixtures establish the supported-colour DIR numerical contract, route execution, and selected resource-lifecycle behavior; they do not retroactively turn the earlier Resolve images into post-replacement visual evidence.

Release Resolve visual comparison, GUI persistence/enabled-state checks, normal-host presentation of both synchronous admission and later-observed GPU failures, and the hardware-specific overlap/retirement matrix remain separate owner evidence. The bounded DIR performance matrix also remains required before making a cost claim. Until those checks are recorded, this work must not be described as complete product, visual, notification, or performance acceptance and must not be promoted to `main` on numerical fixtures or compilation alone.
