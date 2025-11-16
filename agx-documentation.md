# agx-emulsion Technical Reference (Porting Edition)

This document is a parity guide for porting [`agx-emulsion`](https://github.com/andreavolpato/agx-emulsion) to C++/OpenFX. All facts are taken from upstream commit `0e0baf2e3dd51032e89df92c8bb281f05e3ce977` (2025-04-02 14:57:37 +0200). Code excerpts include their original line numbers to support one-to-one verification.

> **Parity rule**: Every constant, mathematical transformation, stochastic process, and serialization detail mentioned below must be duplicated exactly in the port unless explicitly noted as optional tooling.

---

## 1. Repository structure & ownership

| Path | Responsibility |
| --- | --- |
| `agx_emulsion/config.py` | Physical constants (spectral grids, log-exposure axis) and global colour matching data. |
| `agx_emulsion/model/` | Core simulation (AgXPhoto façade, Film emulsion, density curves, couplers, grain, diffusion, illuminants, colour filters, scanner). |
| `agx_emulsion/utils/` | Shared helpers (auto exposure, conversions, interpolation, LUTs, random sampling, warm-up, IO utilities). |
| `agx_emulsion/profiles/` | Profile loading/saving and regeneration pipeline (factory, balance, correct, reconstruct, unmix, plotting). |
| `agx_emulsion/data/` | Packaged CSV spectra, JSON profiles, LUT binaries for spectral upsampling. |
| `agx_emulsion/gui/` | PySide/Qt GUI shell that exercises the same parameters structure as the headless API. |
| `scripts/` | Asset builders (`make_profiles.py`, `make_spectral_upsampling_luts.py`). |
| `agx_emulsion/tests/` | Interactive demo scripts (`test_main_simulation.py`, `test_create_profile.py`, `test_midgray_balance.py`) used for manual smoke checks; no formal automated suite. |

When porting, reproduce the same separation of concerns: configuration constants must be globally visible, while runtime parameter state is isolated inside the `AgXPhoto` instance.

---

## 2. Global constants & configuration

Upstream constants are centralised in `config.py` and are assumed everywhere:

```python
# agx_emulsion/config.py:L4-L10
ENLARGER_STEPS = 170
LOG_EXPOSURE = np.linspace(-3,4,256)
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
STANDARD_OBSERVER_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)
```

* `LOG_EXPOSURE` is exactly 256 samples spanning −3 EV to +4 EV. Several LUTs (density curves, profiles) assume this discretisation; altering the spacing will break interpolations.
* `SPECTRAL_SHAPE` describes an 81-sample wavelength axis (380 nm → 780 nm at 5 nm steps). All spectral arrays (`sensitivity`, `dye_density`, illuminants) are stored on this grid.
* `ENLARGER_STEPS` scales neutral dichroic filter positions. Neutral positions from the database are expressed as fractions of 170 steps.
* `STANDARD_OBSERVER_CMFS` is aligned once and reused for every XYZ integration. Do not re-align per call because the upstream DotMap objects hold pre-aligned spectra.

Warm-up helpers (`utils.numba_warmup`) pre-JIT the fast statistics, LUT, and interpolation kernels with synthetic inputs so their first real invocation skips the compilation pause. They do not pull values from `config.py`, but you can still run them during plugin initialization when you want to amortize the JIT cost ahead of user-facing work.

---

## 3. Parameter bundle (`photo_params`) & neutral filters

`AgXPhoto` receives a `DotMap` produced by `photo_params` with fully-populated defaults. Reproduce the tree and defaults exactly:

```python
# agx_emulsion/model/process.py:L24-L92
params = DotMap()
params.negative = load_profile(negative)
params.print_paper = load_profile(print_paper)
...
params.camera.exposure_compensation_ev = 0.0
params.camera.auto_exposure = True
params.camera.auto_exposure_method = 'center_weighted'
params.camera.lens_blur_um = 0.0
params.camera.film_format_mm = 35.0
params.camera.filter_uv = (1, 410, 8)
params.camera.filter_ir = (1, 675, 15)
...
params.enlarger.illuminant = 'TH-KG3-L'
params.enlarger.print_exposure = 1.0
params.enlarger.print_exposure_compensation = True
params.enlarger.y_filter_shift = 0.0
params.enlarger.m_filter_shift = 0.0
if ymc_filters_from_database:
    params.enlarger.y_filter_neutral = ymc_filters[print_paper][params.enlarger.illuminant][negative][0]
    params.enlarger.m_filter_neutral = ...
    params.enlarger.c_filter_neutral = ...
else:
    params.enlarger.y_filter_neutral = 0.9
...
params.scanner.lens_blur = 0.55
params.scanner.unsharp_mask = (0.7,1.0)
params.io.input_color_space = 'ProPhoto RGB'
params.io.output_color_space = 'sRGB'
params.io.compute_negative = False
params.io.compute_film_raw = False
...
params.settings.rgb_to_raw_method = 'hanatos2025'
params.settings.use_camera_lut = False
params.settings.lut_resolution = 17
params.settings.use_fast_stats = False
```

Key expectations:

* Neutral filter lookups (`ymc_filters`) must load the same JSON database via `utils.io.read_neutral_ymc_filter_values`. The DotMap tree nests as `[print_paper][illuminant][negative][channel]`.
* `params.debug` contains toggles for intermediate outputs and instrumentation. Even when your port uses structured configs, expose equivalent switches so pipelines behave identically.
* The `preview_resize_factor × upscale_factor` product defaults to 1 but can differ; maintain double precision when applying it.

The profile loader returns arrays as `numpy.float64`, so maintain double precision when populating C++ structures to avoid spectral drift.

---

## 4. Profile schema and I/O

Profiles are JSON files deserialised into DotMap instances; they always contain dense arrays:

```python
# agx_emulsion/profiles/io.py:L24-L37
profile = DotMap(json.load(file))
profile.data.log_sensitivity = np.array(profile.data.log_sensitivity)
profile.data.dye_density = np.array(profile.data.dye_density)
profile.data.density_curves = np.array(profile.data.density_curves)
profile.data.log_exposure = np.array(profile.data.log_exposure)
profile.data.wavelengths = np.array(profile.data.wavelengths)
profile.data.density_curves_layers = np.array(profile.data.density_curves_layers)
```

Arrays use shape conventions:

* `log_sensitivity`: 81×3 log10 sensor sensitivity.
* `density_curves`: 256×3 CMY optical densities vs log exposure (post DIR-couplers).
* `density_curves_layers`: 256×3×3 per-sublayer densities (optional).
* `dye_density`: 81×5 (CMY + fog + mid-scale neutral contributions).
* `log_exposure`: aligned with `LOG_EXPOSURE` (256).
* `wavelengths`: copy of `SPECTRAL_SHAPE.wavelengths` (81).

Optional `profile.data.tune` DotMap currently uses `gamma_factor`, `gamma_correction`, `log_exposure_correction`, and `dye_density_min_factor` across the bundled profiles. If upstream ever introduces additional tuning entries, mirror them exactly and provide the same default values to preserve parity and avoid `AttributeError`s.

---

## 5. AgXPhoto façade and stage order

The façade orchestrates every stage. Port the control flow exactly, including debug exits:

```python
# agx_emulsion/model/process.py:L94-L163
class AgXPhoto():
    def __init__(self, params):
        self._params = copy.deepcopy(params)
        ...
        self.timings = {}
        self._apply_debug_switches()

    def process(self, image):
        image = np.double(np.array(image)[:,:,0:3])
        exposure_ev = self._auto_exposure(image)
        image, preview_resize_factor, pixel_size_um = self._crop_and_rescale(image)
        self._apply_profiles_changes()
        if not self.io.full_image:
            self.negative.grain.active = False
            self.negative.halation.active = False
        raw = self._expose_film(image, exposure_ev, pixel_size_um)
        if self.io.compute_film_raw: return raw
        log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
        density_cmy = self._develop_film(log_raw, pixel_size_um)
        if self.debug.return_negative_density_cmy: return density_cmy
        if not self.io.compute_negative:
            log_raw = self._expose_print(density_cmy)
            density_cmy = self._develop_print(log_raw)
            if self.debug.return_print_density_cmy: return density_cmy
        scan = self._scan(density_cmy)
        scan = self._rescale_to_original(scan, preview_resize_factor)
        return scan
```

Observations:

* Input images are forced to `float64` and truncated to three channels.
* Debug toggles can short-circuit the pipeline; implement the same early return semantics to aid debugging.
* Stage timings rely on the `@timeit` decorator. Provide equivalent instrumentation or emulate the decorator when porting.

---

## 6. Stage deep-dive

### 6.1 Debug switches

```python
# agx_emulsion/model/process.py:L110-L127
if self.debug.deactivate_spatial_effects:
    self.negative.halation.size_um = [0,0,0]
    ...
    self.scanner.unsharp_mask = (0.0, 0.0)
if self.debug.deactivate_stochastic_effects:
    self.negative.grain.active = False
    self.negative.glare.active = False
    self.print_paper.glare.active = False
```

Spatial toggles zero out halation radii, blur radii, and DIR coupler diffusion. Stochastic toggles disable grain and glare entirely. Apply these mutations before referencing profile data elsewhere.

### 6.2 Auto exposure

```python
# agx_emulsion/model/process.py:L167-L177
if self.camera.auto_exposure:
    autoexposure_ev = measure_autoexposure_ev(image, input_color_space, input_cctf, method=method)
    exposure_ev = autoexposure_ev + self.camera.exposure_compensation_ev
else:
    exposure_ev = self.camera.exposure_compensation_ev
```

`measure_autoexposure_ev` converts to XYZ, extracts Y, and uses either the image median or a centre-weighted Gaussian mask (`σ=0.2`):

```python
# agx_emulsion/utils/autoexposure.py:L4-L28
image_XYZ = colour.RGB_to_XYZ(image, color_space, apply_cctf_decoding=apply_cctf_decoding)
...
mask = np.exp(-(x**2 + y[:,None]**2)/(2*sigma**2))
mask /= np.sum(mask)
Y_exposure = np.sum(image_Y*mask)
exposure_compensation_ev = - np.log2(Y_exposure / 0.184)
if np.isinf(exposure_compensation_ev):
    exposure_compensation_ev = 0.0
    print('Warning: Autoexposure is Inf...')
```

Clamp infinite EV shifts to 0.0 and replicate the warning for traceability.

### 6.3 Geometry management

Resizing uses `skimage.transform.rescale` with `channel_axis=2` and updates micrometre-per-pixel scale:

```python
# agx_emulsion/model/process.py:L179-L192
pixel_size_um = film_format_mm*1000 / np.max(image.shape)
if self.io.crop:
    image = crop_image(image, center=self.io.crop_center, size=self.io.crop_size)
if preview_resize_factor*upscale_factor != 1.0:
    image = skimage.transform.rescale(image, preview_resize_factor*upscale_factor, channel_axis=2)
    pixel_size_um /= preview_resize_factor*upscale_factor
```

Maintain identical floating-point order: compute pixel pitch before resizing and divide afterward.

### 6.4 Profile adjustments

Viewing glare compensation removal mutates the print profile in place:

```python
# agx_emulsion/model/process.py:L194-L203
if self.print_paper.glare.compensation_removal_factor>0:
    le = self.print_paper.data.log_exposure
    dc = self.print_paper.data.density_curves
    dc_out = remove_viewing_glare_comp(le, dc, factor=..., density=..., transition=...)
    self.print_paper.data.density_curves = dc_out
```

The removal filter re-interpolates the log-exposure axis using Gaussian-blurred remapping (`remove_viewing_glare_comp` in `model/emulsion.py:L21-L53`). Preserve the `scipy.ndimage.gaussian_filter` smoothing window defined there.

### 6.5 Camera exposure

```python
# agx_emulsion/model/process.py:L205-L213
raw = self._rgb_to_film_raw(image, exposure_ev, color_space=..., apply_cctf_decoding=..., use_lut=...)
raw = apply_gaussian_blur_um(raw, self.camera.lens_blur_um, pixel_size_um)
raw = apply_halation_um(raw, self.negative.halation, pixel_size_um)
```

Key helpers:

* `_rgb_to_film_raw` multiplies sensitivity curves by UV/IR band-pass filters before calling either `rgb_to_raw_hanatos2025` or `rgb_to_raw_mallett2019`. It normalises exposures by `2**exposure_ev`.

```python
# agx_emulsion/model/process.py:L270-L299
sensitivity = 10**self.negative.data.log_sensitivity
if self.camera.filter_uv[0]>0 or self.camera.filter_ir[0]>0:
    band_pass_filter = compute_band_pass_filter(...)
    sensitivity *= band_pass_filter[:,None]
...
raw *= 2**exposure_ev
```

* `apply_gaussian_blur_um` (`model/diffusion.py`) converts micrometres to pixels and calls `scipy.ndimage.gaussian_filter` with the library defaults (boundary mode `'reflect'`, truncate `4.0`). Halation and scattering reuse the same boundary mode but explicitly widen the kernel with `truncate=7`. Preserve this difference.
* `apply_halation_um` adds per-channel halation/scattering contributions and renormalises by `(1 + strength)`.

### 6.6 Film development

`Film.develop` sequences interpolation, DIR couplers, and grain:

```python
# agx_emulsion/model/emulsion.py:L184-L198
density_cmy = self._interpolate_density_with_curves(log_raw)
density_cmy = self._apply_density_correction_dir_couplers(density_cmy, log_raw, pixel_size_um)
density_cmy = self._apply_grain(density_cmy, pixel_size_um, bypass_grain, use_fast_stats)
```

#### Density interpolation

```python
# agx_emulsion/model/emulsion.py:L113-L117
def _interpolate_density_with_curves(...):
    return interpolate_exposure_to_density(log_raw, density_curves, self.log_exposure, self.gamma_factor)
```

`interpolate_exposure_to_density` (in `model/density_curves.py`) relies on `utils.fast_interp.fast_interp`, a Numba kernel performing per-channel linear interpolation with endpoint clamping.

#### DIR couplers

```python
# agx_emulsion/model/emulsion.py:L235-L250
if self.dir_couplers.active:
    dir_couplers_amount_rgb = self.dir_couplers.amount * np.array(self.dir_couplers.ratio_rgb)
    M = compute_dir_couplers_matrix(dir_couplers_amount_rgb, self.dir_couplers.diffusion_interlayer)
    density_curves_0 = compute_density_curves_before_dir_couplers(self.density_curves, self.log_exposure, M, ...)
    density_max = np.nanmax(self.density_curves, axis=0)
    diffusion_size_pixel = self.dir_couplers.diffusion_size_um/pixel_size_um
    log_raw_0 = compute_exposure_correction_dir_couplers(log_raw, density_cmy, density_max, M, diffusion_size_pixel, ...)
    density_cmy = interpolate_exposure_to_density(log_raw_0, density_curves_0, self.log_exposure, self.gamma_factor)
```

* Diffusion radius is expressed in μm; convert to pixels using current `pixel_size_um`.
* `compute_exposure_correction_dir_couplers` blurs inhibitor densities with Gaussian kernels; replicate the same boundary handling.

#### Grain

```python
# agx_emulsion/model/emulsion.py:L252-L279
if self.grain.active and not bypass_grain:
    if not self.grain.sublayers_active:
        density_cmy = apply_grain_to_density(...)
    else:
        density_cmy_layers = interp_density_cmy_layers(...)
        density_cmy = apply_grain_to_density_layers(..., use_fast_stats=use_fast_stats)
```

`apply_grain_to_density` (grain.py:L63-L104) converts particle areas to per-pixel counts, samples binomial/Poisson noise, adds minimum fog, averages sublayers, and optionally Gaussian blurs the result. Upstream hard-wires deterministic seeds `[0,1,2]` per channel and increments them per sublayer; no `grain.fixed_seed` toggle is exposed in shipped profiles.

The sublayer path adds micro-structure multiplicative noise via `fast_lognormal_from_mean_std` and optionally blurs dye clouds.

### 6.7 Print exposure & development

```python
# agx_emulsion/model/process.py:L222-L335
film_density_cmy_normalized = self._normalize_film_density(film_density_cmy)
log_raw = self._spectral_lut_compute(..., spectral_calculation, use_lut=self.settings.use_enlarger_lut, save_enlarger_lut=True)
...
raw = contract('ijk, kl->ijl', light, sensitivity)
raw *= self.enlarger.print_exposure
raw_midgray_factor = self._compute_exposure_factor_midgray(...)
raw *= raw_midgray_factor
raw_preflash = self._compute_raw_preflash(...)
raw += raw_preflash
log_raw = np.log10(raw + 1e-10)
```

Highlights:

* `_normalize_film_density` adds `grain.density_min` fog before scaling to `[0,1]`.
* `_film_density_cmy_to_print_log_raw` calls `color_enlarger` with neutral filter positions scaled by `ENLARGER_STEPS` and user offsets. Preflash passes separate filter offsets.
* `_compute_exposure_factor_midgray` re-runs the camera + film development pipeline on a `0.184` neutral patch (green-normalised) to maintain mid-grey parity with exposure compensation EV.
* LUT acceleration uses `utils.lut.compute_with_lut`, storing the generated LUT in `self.debug.luts.enlarger_lut` when enabled.

Printing is bypassed entirely when `io.compute_negative=True`; ensure your port still respects the neutral normalisation path for scanning negatives.

### 6.8 Scanner path & output

```python
# agx_emulsion/model/process.py:L391-L440
if self.io.compute_negative:
    density_cmy_n = self._normalize_film_density(density_cmy)
    profile = self.negative
else:
    density_cmy_n = self._normalize_print_density(density_cmy)
    profile = self.print_paper
scan_illuminant = standard_illuminant(profile.info.viewing_illuminant)
normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
...
log_xyz = self._spectral_lut_compute(..., use_lut=use_lut, save_scanner_lut=True)
xyz = 10**log_xyz
illuminant_xyz = contract('k,kl->l', scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
xyz = add_glare(xyz, illuminant_xyz, profile)
illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
rgb = colour.XYZ_to_RGB(xyz, colourspace=self.io.output_color_space, apply_cctf_encoding=False, illuminant=illuminant_xy)
rgb = apply_gaussian_blur(rgb, self.scanner.lens_blur)
if unsharp_mask[0] > 0 and unsharp_mask[1] > 0:
    rgb = apply_unsharp_mask(rgb, sigma=unsharp_mask[0], amount=unsharp_mask[1])
if self.io.output_cctf_encoding:
    rgb = colour.RGB_to_RGB(rgb, color_space, color_space, apply_cctf_decoding=False, apply_cctf_encoding=True)
rgb = np.clip(rgb, a_min=0, a_max=1)
```

`add_glare` introduces lognormal glare noise and multiplies by illuminant XYZ. When LUT mode is active, store the resulting LUT in `self.debug.luts.scanner_lut` as Python does.

Rescaling at this stage only divides by `preview_resize_factor`, leaving any `upscale_factor` applied during `_crop_and_rescale` untouched. For example, if the preview downscale is `0.5` and an upscale of `2.0` is requested, `_rescale_to_original` outputs an image whose dimensions equal `original_size × 2.0`.

---

## 7. Spectral upsampling details

### 7.1 Hanatos 2025 LUT path

```python
# agx_emulsion/utils/spectral_upsampling.py:L203-L224
HANATOS2025_SPECTRA_LUT = load_spectra_lut()
def rgb_to_raw_hanatos2025(rgb, sensitivity, color_space, apply_cctf_decoding, reference_illuminant):
    if rgb.shape[1] == 1:
        spectrum = rgb_to_spectrum(...)
        raw = np.einsum('l,lm->m', spectrum, sensitivity)
        raw = np.array([[raw]])
    else:
        tc_raw, b = rgb_to_tc_b(rgb, ...)
        tc_lut  = contract('ijl,lm->ijm', HANATOS2025_SPECTRA_LUT, sensitivity)
        raw = apply_lut_cubic_2d(tc_lut, tc_raw)
        raw *= b[...,None]
    midgray_rgb = np.array([[[0.184]*3]])
    illuminant_midgray = rgb_to_spectrum(midgray_rgb, color_space=color_space, apply_cctf_decoding=False, reference_illuminant=reference_illuminant)
    raw_midgray  = np.einsum('k,km->m', illuminant_midgray, sensitivity)
    return raw / raw_midgray[1]
```

Key points:

* Single-pixel inputs fall back to direct spectrum reconstruction instead of LUT sampling.
* Chromaticities are transformed via `rgb_to_tc_b`, which handles colour space conversion, optional CCTF decoding, and chromatic adaptation using CAT02.
* The LUT stores raw spectral power distributions; every invocation contracts it with the current sensor sensitivity via `contract('ijl,lm->ijm', …)` immediately before `apply_lut_cubic_2d` runs. Skipping this multiplication would change the raw response, so do not remove it when porting.
* Normalisation uses the green channel response of a `0.184` neutral patch.

### 7.2 Mallett 2019 basis

```python
# agx_emulsion/utils/spectral_upsampling.py:L162-L199
MALLETT2019_BASIS = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(SPECTRAL_SHAPE)
raw  = contract('ijk,lk,lm->ijm', lrgb, basis_set_with_illuminant, sensitivity)
raw_midgray  = np.einsum('k,km->m', illuminant*0.184, sensitivity)
return raw / raw_midgray[1]
```

The basis functions are pre-multiplied by the reference illuminant to avoid repeated multiplication. Colour space conversion always routes via sRGB linear before contracting.

### 7.3 Band-pass filters

Both camera and utility modules share identical band-pass code:

```python
# agx_emulsion/utils/spectral_upsampling.py:L139-L157
wl = SPECTRAL_SHAPE.wavelengths
filter_uv  = 1-amp_uv + amp_uv*sigmoid_erf(wl, wl_uv, width=width_uv)
filter_ir  = 1-amp_ir + amp_ir*sigmoid_erf(wl, wl_ir, width=-width_ir)
band_pass_filter = filter_uv * filter_ir
```

Ensure your port clamps amplitudes to `[0,1]` and applies the same sign convention for IR width (negative for falling edge).

---

## 8. Diffusion, halation, and optical filtering

### 8.1 Diffusion utilities

`apply_gaussian_blur_um` and `apply_halation_um` live in `model/diffusion.py` (not shown here). Both convert μm to pixels using `pixel_size_um`. The generic blur keeps the SciPy defaults (`mode='reflect'`, `truncate=4.0`), whereas halation/scattering widen the kernel by setting `truncate=7` while leaving the default `'reflect'` mode in place. Keep these exact parameters so the C++ port mirrors the upstream kernel widths.

### 8.2 Colour enlarger and filters

```python
# agx_emulsion/model/color_filters.py:L126-L131
def color_enlarger(light_source, y_filter_value, m_filter_value, c_filter_value=0,
                   enlarger_steps=ENLARGER_STEPS,
                   filters=durst_digital_light_dicrhoic_filters):
    ymc_filter_values = np.array([y_filter_value, m_filter_value, c_filter_value]) / enlarger_steps
    filtered_illuminant = filters.apply(light_source, values=ymc_filter_values)
    return filtered_illuminant
```

`filters.apply` interpolates database filters (`load_dichroic_filters`) and blends them using Durst digital light transmittance. The neutral positions stored in `ymc_filters` correspond to slider positions in multiples of `1/170`.

Generic filters (`GenericFilter`) load additional transmissive spectra (heat filters, lens transmission). Respect the default UV/IR filter tuple `(1,410,8)` / `(1,675,15)` from `photo_params` to mimic protective filters.

---

## 9. Grain modelling specifics

Critical routines from `model/grain.py`:

```python
# agx_emulsion/model/grain.py:L11-L49
def layer_particle_model(..., method='poisson_binomial', use_fast_stats=False):
    if seed is not None:
        np.random.seed(seed)
    probability_of_development = np.clip(density/density_max, 1e-6, 1-1e-6)
    od_particle = density_max/n_particles_per_pixel
    if method=='poisson_binomial':
        if use_fast_stats:
            binom_rvs = fast_binomial
            poisson_rvs = fast_poisson
        else:
            binom_rvs = scipy.stats.binom.rvs
            poisson_rvs = scipy.stats.poisson.rvs
        saturation = 1 - probability_of_development*grain_uniformity*(1-1e-6)
        seeds = poisson_rvs(n_particles_per_pixel/saturation)
        grain = binom_rvs(seeds, probability_of_development)
        grain = np.double(grain)*od_particle*saturation
    if blur_particle>0:
        grain = scipy.ndimage.gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
```

* RNG seeds: the upstream code calls `np.random.seed(seed)` because SciPy RNGs share NumPy’s global generator. Match this behaviour; use a thread-local generator if required but ensure determinism when `fixed_seed` is set.
* `apply_grain_to_density` adds `density_min` before sampling and subtracts it afterward, ensuring fog is reintroduced only via `density_min` and not the random sample.
* Sublayer mode (`apply_grain_to_density_layers`) scales particle areas per layer and adds lognormal clumping when `grain_micro_structure` has significant sigma.

---

## 10. Lookup tables & interpolation

```python
# agx_emulsion/utils/lut.py:L4-L23
def _create_lut_3d(function, xmin=0, xmax=1, steps=32):
    x = np.linspace(xmin, xmax, steps, endpoint=True)
    X = np.meshgrid(x,x,x, indexing='ij')
    X = np.stack(X, axis=3)
    X = np.reshape(X, (steps**2, steps, 3))
    lut = np.reshape(function(X), (steps, steps, steps, 3))
    return lut

def compute_with_lut(data, function, xmin=0, xmax=1, steps=32):
    lut = _create_lut_3d(function, xmin, xmax, steps)
    return apply_lut_cubic_3d(lut, data), lut
```

Numba kernels in `utils.fast_interp_lut` provide cubic interpolation. They expect data arranged as `[steps, steps, steps, 3]` and `[H, W, 3]` inputs. When `use_enlarger_lut`/`use_scanner_lut` are true, the Python code returns both the sampled output and the LUT; store both in your port for diagnostics.

`lut_resolution` defaults to 17. Respect the same resolution range and cubic interpolation order to avoid tonal deviations.

---

## 11. Interpolation & fast math helpers

`fast_interp` (Numba JIT) provides per-channel linear interpolation with endpoint clamping and optional channel-specific axes:

```python
# agx_emulsion/utils/fast_interp.py:L5-L73
@numba.njit(parallel=True, fastmath=True, cache=True)
def fast_interp(image, x_axis, y_vals):
    ...
    if x <= xa[0]:
        flat_result[i, c] = y_vals[0, c]
    elif x >= xa[K - 1]:
        flat_result[i, c] = y_vals[K - 1, c]
    else:
        idx = np.searchsorted(xa, x)
        low = idx - 1
        t = (x - x0) * inv_dx_val[low]
        flat_result[i, c] = y_vals[low, c] + t * (y_vals[low + 1, c] - y_vals[low, c])
```

Match its interpolation semantics exactly (linear, clamped), and keep the monotonic assumption. Upstream provides no automated tests for `fast_interp`, so validate its density-curve behaviour manually when establishing parity baselines.

`utils.fast_stats` (not quoted here) implements approximate RNGs for Poisson/binomial/lognormal distributions. `use_fast_stats=True` switches Film grain sampling to these approximations; ensure your port exposes the same toggle and matches distribution parameters.

---

## 12. I/O utilities and datasets

### 12.1 Image IO (OpenImageIO)

```python
# agx_emulsion/utils/io.py:L13-L103
def load_image_oiio(filename):
    in_img = oiio.ImageInput.open(filename)
    ...
    np_pixels = np_pixels.reshape(spec.height, spec.width, spec.nchannels)
    if spec.format == oiio.TypeDesc("uint16"):
        np_pixels = np.double(np_pixels)/(2**16-1)
    ...
    return np_pixels

def save_image_oiio(filename, image_data, bit_depth=32):
    if ext == "png":
        img_uint16 = np.clip(image_data, 0, 1) * 65535.0
        img_uint16 = img_uint16.astype(np.uint16)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint16"))
        data_to_write = img_uint16
    elif ext=='exr' and bit_depth==16:
        img_half = image_data.astype(np.float16)
        spec = oiio.ImageSpec(..., oiio.TypeDesc("half"))
    elif ext=='exr' and bit_depth==32:
        img_float = image_data.astype(np.float32)
```

Normalise PNGs to [0,1] on load and clamp+scale when writing. The port can use OIIO’s C++ API; ensure identical scaling.

### 12.2 Spectral data loading

`load_agx_emulsion_data` reads CSVs, interpolates them to `SPECTRAL_SHAPE`, and aligns density curves using Akima splines by default. When regenerating profiles in C++, replicate the same interpolation method, donor logic, and log-exposure shift calculation (`(max+min)/2`).

### 12.3 Neutral filter database

`read_neutral_ymc_filter_values` (defined in `utils/io.py`, not quoted) loads JSON-coded filter wheel positions. Keep the same nested dictionary layout and ensure lookups handle missing keys with informative errors.

---

## 13. Profiles toolkit & scripts

* `profiles.factory.create_profile` reconstructs profiles from CSV data, calling into `load_agx_emulsion_data`, `model.density_curves.compute_density_curves`, and `profiles.balance` routines.
* `profiles.balance` provides white-balance functions (`balance_metameric_neutral`) crucial for matching neutral prints.
* `profiles.correct` introduces utilities like `align_midscale_neutral_exposures` and `remove_glare_compensation`. Use them when validating regenerated assets.
* `profiles.reconstruct` performs optimisation via SciPy’s `least_squares` to recover density curves or dye spectra from sparse measurements. If your port offers asset tooling, duplicate the optimisation strategies.
* `scripts/make_profiles.py` and `scripts/make_spectral_upsampling_luts.py` orchestrate these helpers. The LUT script outputs `.npy` arrays consumed by `rgb_to_raw_hanatos2025`; maintain identical file formats so the C++ port can reuse upstream assets.

---

## 14. GUI and automation context

`agx_emulsion/gui/main.py` builds a Qt interface around the same `photo_params` structure. It serialises parameters via `DotMap.toDict()` to JSON. Preserve this layout so presets remain compatible between Python and OpenFX versions.

Pytest coverage includes:

* `tests/test_main_simulation.py` – renders sample images and checks RMS error thresholds.
* `tests/test_midgray_balance.py` – asserts the auto exposure & mid-grey balancing remain stable.
* `tests/test_create_profile.py` – validates profile regeneration (CSV → JSON parity).

Porting strategy: export equivalent golden renders/density arrays and compare them byte-for-byte against upstream outputs using the same inputs.

---

## 15. External dependencies & expected precision

* **NumPy float64** – all arrays stay in double precision. Upstream adds `1e-10` before `log10` to avoid `-inf`; replicate the guard.
* **SciPy** – Gaussian blurs, Akima interpolation, statistical distributions. If replacing with custom kernels, match kernel radii and boundary modes exactly.
* **Colour-Science** – Colour space transforms, spectral shapes, chromatic adaptation, ACES utilities.
* **scikit-image** – Rescaling with anti-alias filtering; match `channel_axis=2` semantics.
* **DotMap** – Parameter containers; map to structured C++ types but keep attribute names for JSON interoperability.
* **Opt-Einsum** – Contracts (Einstein summation). Equivalent C++ implementation should mirror contraction order to maintain numeric stability.
* **Numba** – Accelerates interpolation/LUT/RNG. Recreate equivalent vectorised kernels in C++ for throughput parity.
* **OpenImageIO** – Image IO.

Always run double precision end-to-end; downcast only when writing files.

---

## 16. Porting checklist

1. **Constants** – Mirror `ENLARGER_STEPS`, `LOG_EXPOSURE`, `SPECTRAL_SHAPE`, and align all spectral data accordingly.
2. **Profiles** – Load JSON with float64 precision; honour runtime tweaks (glare removal, debug toggles).
3. **Parameter defaults** – Recreate the `photo_params` tree verbatim to guarantee baseline parity.
4. **Spectral upsampling** – Support both `hanatos2025` LUT and `mallett2019` basis. Ensure LUT assets load identically and mid-grey normalisation uses the green channel.
5. **Band-pass & filters** – Implement `compute_band_pass_filter`, `color_enlarger`, and neutral filter lookups exactly.
6. **Film development** – Keep the order: density interpolation → DIR couplers → grain. Respect micrometre-to-pixel conversions and `use_fast_stats` toggles.
7. **Printing** – Honour preflash, print exposure compensation, and mid-grey re-simulation. Provide cubic LUT acceleration with debug exports.
8. **Scanning** – Support negative vs paper pipelines, glare injection, Gaussian blur + unsharp mask, optional output CCTF encoding, and LUT caching.
9. **Randomness** – Reproduce RNG seeding rules (`fixed_seed` bypass, per-channel offsets, lognormal glare). Offer deterministic paths for tests.
10. **Interpolation & LUTs** – Implement clamped linear interpolation and cubic LUT sampling that match Numba kernels numerically.
11. **Precision guards** – Apply `np.log10(np.fmax(raw,0)+1e-10)` equivalents, clamp LUT inputs to `[0,1]`, and propagate NaNs/zeros the same way.
12. **Diagnostics** – Provide timing hooks and intermediate buffer exports mirroring debug flags.
13. **Testing** – Recreate pytest scenarios using identical assets to confirm parity frame-by-frame.

Following these instructions ensures the C++/OpenFX port produces renders indistinguishable from the upstream Python implementation and remains compatible with existing tooling, presets, and datasets.
