# New Scanner Refactor Plan

The goal is to make Film-Juicer's scanner path numerically identical to **agx-emulsion** while removing all legacy behaviour. The steps below are intentionally self-contained so each can be verified independently. New identifiers mirror the Python names (camelCase → PascalCase/C++ snake case where appropriate) to make A/B reviews trivial.

## Context
- `scanner-refactor.md` assumed its plan was finished, yet all code was reverted, leaving Film-Juicer on the legacy scanner implementation.
- The previous refactor attempt produced heavy underexposure and desaturation when `PrintBypass=false`, caused by:
  1. **Missing `+ density_min` in negative normalization** – shifted density range, causing underexposure
  2. **Mismatched illuminants** between normalization, LUT computation, and adaptation – caused color shifts/desaturation
  3. **Missing or incorrect mid-gray compensation** in print exposure – caused underexposure with exposure compensation
  4. **Incorrect channel ordering** (YMC vs CMY) or missing `dye_density_min_factor` – affected spectral base density
- `viewing-illuminant-refactor.md` already highlights the SPD single-sourcing requirement; the new plan must integrate that work and extend it so the scanner path matches **agx-emulsion** numerically across both the negative and print media.
- Every step below is self-contained, lists its acceptance criteria, and calls out failure modes so we can avoid repeating the exposure/saturation regressions.

---

### Step 1 – Align the scanner parameter surface and profile ingestion

#### Tasks
- **Remove Film-Juicer-only scanner controls**: Delete `ScannerEnabled`, `ScannerAutoExposure`, `ScannerTargetY`, `ScannerFilmLongEdgeMm`. agx-emulsion has no scanner auto-gain feature; remove it entirely.
- **Add agx scanner parameters**: Replace with the two parameters from `agx_emulsion/model/process.py:62-63`:
  - `params.scanner.lens_blur = 0.55` (pixel-space sigma)
  - `params.scanner.unsharp_mask = (0.7, 1.0)` (sigma_px, amount tuple)

  Host/UI must emit `ScannerLensBlurSigmaPx` (float, default `0.55`) and `ScannerUnsharpMask` (two floats: `sigma_px`, `amount`, defaults `0.7` and `1.0`).

- **Extend profile ingestion**: Add to `ProfileJSONLoader`:
  - `negative.grain.active` (bool)
  - `negative.grain.density_min` (float[3], CMY fog densities)
  - `negative.glare` object: `active`, `percent`, `roughness`, `blur`
  - `print_paper.glare` object: `active`, `percent`, `roughness`, `blur`, plus `compensation_removal_factor`, `compensation_removal_density`, `compensation_removal_transition`

  Profile defaults (if missing, log warning and use these):
  - Negative glare: `{active: false, percent: 0.0, roughness: 0.25, blur: 0.5}`
  - Print glare: `{active: true, percent: 0.1, roughness: 0.4, blur: 0.5, compensation_removal_factor: 0.0, compensation_removal_density: 1.2, compensation_removal_transition: 0.3}`

  Assert finite values before exposing downstream.

- **Introduce typed structs**: Replace `Scanner::Params` with:
  ```cpp
  struct Scanner::Options {
      float lensBlurSigmaPx;      // Direct pixel sigma, no µm conversion
      float unsharpSigmaPx;       // Unsharp mask blur sigma
      float unsharpAmount;        // Unsharp mask strength
  };

  struct ProfileGlare {
      bool active;
      float percent;
      float roughness;
      float blur;
      // Print-only fields:
      float compensationRemovalFactor;
      float compensationRemovalDensity;
      float compensationRemovalTransition;
  };

  struct GrainMetadata {
      bool active;
      float densityMin[3];  // CMY fog densities
      // ... other grain params
  };
  ```

  Store `Scanner::Options` in render state, `ProfileGlare`/`GrainMetadata` in `BaseState`/`WorkingState`.

- **Wire through effect chain**: Thread structs through `JuicerEffect` → `JuicerProcessor` → scanner runtime using agx field names (`lens_blur`, `unsharp_mask`). Fetch glare/grain from profile objects during `WorkingState` rebuild.

#### Acceptance Criteria
- [ ] All legacy scanner parameters removed from UI and parameter definitions
- [ ] `Scanner::Options` struct contains only the three agx fields
- [ ] Profiles load glare metadata with correct defaults for negative/print
- [ ] `static_assert` validates struct size to prevent field creep
- [ ] Compile fails if legacy parameters are referenced

#### Risks & Mitigation
- *Risk*: Leaving legacy scanner toggles allows double-application of exposure/geometry fixes.
  *Mitigation*: Remove parameters outright, gate scanner construction behind new struct, use `static_assert` on struct size/fields.
- *Risk*: Profiles without glare metadata crash scanner path.
  *Mitigation*: Provide agx factory defaults, log `JTRACE` when defaults are used, validate finite values.

---

### Step 2 – Rebuild `WorkingState` scanner metadata per medium (negative & print)

#### Tasks
- **Build per-medium scanner illuminants**: For both negative and print profiles:
  ```cpp
  // From profile.info.viewing_illuminant (e.g., "D50", "D55", "TH-KG3-L")
  SpectralDistribution scanIlluminant = load_standard_illuminant(profile.info.viewing_illuminant);
  float normalization = sum(scanIlluminant * CMF_ybar);  // ∑(SPD[λ] * ȳ[λ])
  vec3 XYZ_white = sum(scanIlluminant * CMF_xyz) / normalization;
  vec2 xy_white = XYZ_to_xy(XYZ_white);
  ```

  Store in `WorkingState::negativeScannerIlluminant` and `printScannerIlluminant` structs containing:
  - `SpectralDistribution curve` (81-element SPD)
  - `float normalization` (the ∑SPD·ȳ scalar)
  - `vec3 XYZ_white`
  - `vec2 xy_white`
  - `std::string illuminantKey` (for cache invalidation)

  **Fail profile load if `viewing_illuminant` is missing or invalid** – no D50 fallback. This ensures parity renders fail fast.

- **Delete legacy UI viewing illuminant overrides**: Remove all UI controls for viewing illuminant selection. Viewing illuminant is now **always** sourced from `profile.info.viewing_illuminant`. Reference and enlarger illuminants remain user-adjustable.

- **Compute per-medium density normalization ranges**:

  **Negative (film):**
  ```cpp
  // From agx_emulsion/model/process.py:301-306
  vec3 density_max = nanmax(negative.data.density_curves, axis=0);  // Per-channel max
  vec3 density_min = negative.grain.density_min;                     // CMY fog
  vec3 range_max = density_max + density_min;
  vec3 range_min = vec3(0.0f);  // Always zero

  // Store in WorkingState::negativeDensityRange
  struct ScannerDensityRange {
      vec3 min;  // [0, 0, 0] for negative
      vec3 max;  // density_max + density_min
  };
  ```

  **Print:**
  ```cpp
  // From agx_emulsion/model/process.py:378-381
  vec3 density_max = nanmax(print_paper.data.density_curves, axis=0);
  vec3 range_max = density_max;
  vec3 range_min = vec3(0.0f);

  // Store in WorkingState::printDensityRange
  struct ScannerDensityRange {
      vec3 min;  // [0, 0, 0]
      vec3 max;  // density_max (no density_min added)
  };
  ```

  These ranges drive `_normalize_film_density` / `_normalize_print_density` and their inverses.

- **Apply glare compensation removal** (print only):
  ```cpp
  // From agx_emulsion/model/process.py:196-203
  // During WorkingState rebuild, after loading print profile:
  if (printProfile.glare.compensationRemovalFactor > 0.0f) {
      auto modifiedCurves = remove_viewing_glare_comp(
          printProfile.data.log_exposure,
          printProfile.data.density_curves,
          printProfile.glare.compensationRemovalFactor,
          printProfile.glare.compensationRemovalDensity,
          printProfile.glare.compensationRemovalTransition
      );
      // Store modified curves in WorkingState for print development
      workingState.printDensityCurves = modifiedCurves;
  }
  ```

  This **mutates print density curves once** during state rebuild, before any rendering. The modified curves are used for all subsequent print development.

- **Cache validation flags**: Store per-medium booleans:
  - `negativeScannerValid` / `printScannerValid`
  - `negativeLutValid` / `printLutValid`
  - `negativeGlareValid` / `printGlareValid`

  Block rendering with clear error if any required data is invalid.

#### Acceptance Criteria
- [ ] Both `negativeScannerIlluminant` and `printScannerIlluminant` built from `profile.info.viewing_illuminant`
- [ ] Profile load fails with clear error if `viewing_illuminant` is missing/invalid
- [ ] Density ranges computed with exact agx formulas (negative includes `+density_min`, print does not)
- [ ] Glare compensation removal applied during WorkingState rebuild when `factor > 0`
- [ ] Cache invalidation triggers when illuminant key or density curves change
- [ ] All legacy UI viewing illuminant controls removed

#### Risks & Mitigation
- *Risk*: Mismatched illuminants between normalization/LUT/adaptation recreate desaturation.
  *Mitigation*: Assert same SPD hash used for scanner tables, normalization, and adaptation. Abort render on mismatch.
- *Risk*: Density range formulas drift from agx cause exposure shifts.
  *Mitigation*: Mirror exact agx formulas, emit detailed error if cached min/max diverge. Add unit test comparing against Python reference.

---

### Step 3 – Introduce shared density staging and print exposure pipeline

This is the "missing bridge" – the critical step that hands off developed densities to the scanner without re-computing them.

#### Tasks

##### 3a. Shared Density Tensors
- **Split renderer into three stages**:
  1. **RGB → negative densities**: Camera exposure → film development (DIR + grain) → `density_cmy_negative[H,W,3]`
  2. **Print exposure & development**: Negative densities → print exposure (enlarger + mid-gray) → print development → `density_cmy_print[H,W,3]`
  3. **Scanner**: Selected density tensor → spectral integration → RGB output

- **Persist densities in FrameContext**:
  ```cpp
  struct FrameContext {
      std::vector<float> densityCmyNegative;  // [H*W*3], CMY channel order
      std::vector<float> densityCmyPrint;     // [H*W*3], CMY channel order
      Medium activeMedium;  // NEGATIVE or PRINT
  };
  ```

  **Channel ordering**: All density arrays use **CMY** ordering throughout (index 0=Cyan, 1=Magenta, 2=Yellow), matching agx-emulsion. No YMC reordering at any stage.

##### 3b. Density Normalization Helpers
Implement exact agx normalization formulas:

**Negative (film):**
```cpp
// From agx_emulsion/model/process.py:301-313
vec3 normalize_film_density(vec3 density_cmy, const ScannerDensityRange& range) {
    // range.max = density_max + density_min
    // range.min = [0, 0, 0]
    vec3 density_cmy_n = (density_cmy + density_min) / range.max;
    return density_cmy_n;
}

vec3 denormalize_film_density(vec3 density_cmy_n, const ScannerDensityRange& range) {
    vec3 density_cmy = density_cmy_n * range.max - density_min;
    return density_cmy;
}
```

**Print:**
```cpp
// From agx_emulsion/model/process.py:378-386
vec3 normalize_print_density(vec3 density_cmy, const ScannerDensityRange& range) {
    // range.max = density_max
    // range.min = [0, 0, 0]
    vec3 density_cmy_n = density_cmy / range.max;
    return density_cmy_n;
}

vec3 denormalize_print_density(vec3 density_cmy_n, const ScannerDensityRange& range) {
    vec3 density_cmy = density_cmy_n * range.max;
    return density_cmy;
}
```

These normalized values are **only** used as LUT coordinates. Raw densities remain available for spectral callbacks.

##### 3c. Print Exposure Pipeline
Insert between negative development and print development:

```cpp
// From agx_emulsion/model/process.py:318-335
vec3 film_density_cmy_to_print_log_raw(
    vec3 density_cmy,
    const NegativeProfile& negative,
    const PrintProfile& print,
    const EnlargerParams& enlarger
) {
    // 1. Compute print illuminant (enlarger light + dichroic filters)
    SpectralDistribution enlargerLight = standard_illuminant(enlarger.illuminant);
    float yFilter = enlarger.yFilterNeutral * 170.0f + enlarger.yFilterShift;
    float mFilter = enlarger.mFilterNeutral * 170.0f + enlarger.mFilterShift;
    float cFilter = enlarger.cFilterNeutral * 170.0f;
    SpectralDistribution printIlluminant = color_enlarger(
        enlargerLight, yFilter, mFilter, cFilter
    );

    // 2. Project negative density through print illuminant
    vec3 sensitivity = pow(10.0f, print.data.log_sensitivity);  // [81×3]
    SpectralDistribution densitySpectral = compute_density_spectral(negative, density_cmy);
    SpectralDistribution light = density_to_light(densitySpectral, printIlluminant);
    vec3 raw = integrate(light, sensitivity);  // contract('ijk,kl->ijl')

    // 3. Apply print exposure
    raw *= enlarger.printExposure;

    // 4. Apply mid-gray compensation factor
    float rawMidgrayFactor = compute_exposure_factor_midgray(
        negative, print, enlarger, sensitivity, printIlluminant
    );
    raw *= rawMidgrayFactor;

    // 5. Add preflash
    vec3 rawPreflash = compute_raw_preflash(
        enlargerLight, negative, enlarger, sensitivity
    );
    raw += rawPreflash;

    // 6. Convert to log space for print development
    vec3 log_raw = log10(raw + 1e-10f);
    return log_raw;
}
```

**Mid-gray compensation** (critical for exposure parity):
```cpp
// From agx_emulsion/model/process.py:362-376
float compute_exposure_factor_midgray(
    const NegativeProfile& negative,
    const PrintProfile& print,
    const EnlargerParams& enlarger,
    const vec3 sensitivity[81],
    const SpectralDistribution& printIlluminant
) {
    // Re-run camera + simple film development on 0.184 gray patch
    float negExpCompEv = enlarger.printExposureCompensation
        ? camera.exposureCompensationEv : 0.0f;
    vec3 rgbMidgray = vec3(0.184f) * pow(2.0f, negExpCompEv);

    vec3 rawMidgray = rgb_to_film_raw(rgbMidgray, negative, /*exposureEv=*/0.0f);
    vec3 densityCmyMidgray = develop_simple(negative, log10(rawMidgray + 1e-10f));

    // Project midgray density through print illuminant
    SpectralDistribution densitySpectralMidgray = compute_density_spectral(
        negative, densityCmyMidgray
    );
    SpectralDistribution lightMidgray = density_to_light(
        densitySpectralMidgray, printIlluminant
    );
    vec3 rawMidgrayPrint = integrate(lightMidgray, sensitivity);

    // Normalize to green channel
    float factor = 1.0f / rawMidgrayPrint.y;
    return factor;
}
```

##### 3d. Density Curve Interpolation with Gamma
All density curve interpolation must use `profile.data.tune.gamma_factor`:

```cpp
// From agx_emulsion/model/density_curves.py
vec3 interpolate_exposure_to_density(
    vec3 log_raw,
    const DensityCurves& curves,  // [256×3]
    const LogExposureAxis& log_exposure,  // [256]
    float gamma_factor
) {
    // Linear interpolation with gamma correction
    // Implementation matches utils/fast_interp.py
    vec3 density = fast_interp(log_raw, log_exposure, curves);
    // Apply gamma_factor if needed (check agx implementation)
    return density;
}
```

##### 3e. Medium Selection
Map Film-Juicer's `PrintBypass` to agx's `io.compute_negative`:

```cpp
Medium activeMedium = PrintBypass ? Medium::NEGATIVE : Medium::PRINT;

// Scanner receives:
const DensityBuffer& densities = (activeMedium == Medium::NEGATIVE)
    ? frameContext.densityCmyNegative
    : frameContext.densityCmyPrint;

const Profile& profile = (activeMedium == Medium::NEGATIVE)
    ? workingState.negative
    : workingState.printPaper;

const ScannerIlluminant& illum = (activeMedium == Medium::NEGATIVE)
    ? workingState.negativeScannerIlluminant
    : workingState.printScannerIlluminant;

const ScannerDensityRange& range = (activeMedium == Medium::NEGATIVE)
    ? workingState.negativeDensityRange
    : workingState.printDensityRange;
```

#### Acceptance Criteria
- [ ] Negative and print densities persist in `FrameContext` with CMY channel ordering
- [ ] Normalization formulas match agx exactly (negative includes `+density_min`, print does not)
- [ ] Print exposure pipeline includes all five steps (illuminant, projection, exposure, mid-gray, preflash)
- [ ] Mid-gray compensation matches agx's green-channel normalization
- [ ] Density interpolation uses `gamma_factor` from profile tune parameters
- [ ] Medium selection maps `PrintBypass` correctly to negative/print profiles
- [ ] DEBUG assertions validate medium matches before normalization

#### Risks & Mitigation
- *Risk*: Recomputing densities in scanner leads to double conversion and drift.
  *Mitigation*: Delete old RGB→density path in scanner, add compile-time errors.
- *Risk*: Normalizing with wrong medium's range reintroduces underexposure.
  *Mitigation*: Encode medium in density struct, assert match in DEBUG builds.
- *Risk*: Missing mid-gray compensation causes underexposure.
  *Mitigation*: Make mid-gray step non-optional, validate output matches agx reference values.

---

### Step 4 – Rebuild scanner optics module as dedicated translation unit

#### Tasks

##### 4a. Module Structure
Create `ScannerOptics.{h,cpp}` with single entry point:
```cpp
vec3 render_density_to_rgb(
    const DensityBuffer& densityCmy,  // [H,W,3] CMY ordering
    Medium medium,
    const WorkingState& ws,
    const Scanner::Options& opts
);
```

##### 4b. Spectral LUT Computation
Implement `_spectral_lut_compute` from agx:

```cpp
// From agx_emulsion/model/process.py:412-413 and utils/lut.py
std::array<float, RESOLUTION*RESOLUTION*RESOLUTION*3> spectral_lut_compute(
    Medium medium,
    const Profile& profile,
    const ScannerIlluminant& illuminant,
    const ScannerDensityRange& range,
    int resolution = 17  // Default from agx line 89
) {
    // Create 3D LUT grid [0,1]³
    std::array<float, ...> lut;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                vec3 density_cmy_n = vec3(
                    float(i) / (resolution - 1),
                    float(j) / (resolution - 1),
                    float(k) / (resolution - 1)
                );

                // Spectral calculation (same as direct path)
                vec3 log_xyz = spectral_calculation(
                    density_cmy_n, medium, profile, illuminant, range
                );

                // Store log(XYZ) in LUT
                lut[i*resolution*resolution*3 + j*resolution*3 + k*3 + 0] = log_xyz.x;
                lut[i*resolution*resolution*3 + j*resolution*3 + k*3 + 1] = log_xyz.y;
                lut[i*resolution*resolution*3 + j*resolution*3 + k*3 + 2] = log_xyz.z;
            }
        }
    }
    return lut;
}
```

**LUT resolution**: Default to **17³** (from agx line 89: `params.settings.lut_resolution = 17`), expose UI slider for `settings.lut_resolution` to allow denser grids for testing.

**Interpolation**: Use trilinear or tricubic matching agx's `apply_lut_cubic_3d`.

##### 4c. Spectral Calculation (Core of Scanner)
```cpp
// From agx_emulsion/model/process.py:402-411
vec3 spectral_calculation(
    vec3 density_cmy_n,  // Normalized [0,1]
    Medium medium,
    const Profile& profile,
    const ScannerIlluminant& illuminant,
    const ScannerDensityRange& range
) {
    // 1. Denormalize
    vec3 density_cmy;
    if (medium == Medium::NEGATIVE) {
        density_cmy = denormalize_film_density(density_cmy_n, range);
    } else {
        density_cmy = denormalize_print_density(density_cmy_n, range);
    }

    // 2. Compute spectral density
    // From agx_emulsion/model/emulsion.py:76-79
    SpectralDistribution density_spectral = compute_density_spectral(
        profile, density_cmy
    );

    // 3. Convert to light transmission
    SpectralDistribution light = density_to_light(
        density_spectral, illuminant.curve
    );

    // 4. Integrate to XYZ
    vec3 xyz = integrate(light, CMF_xyz) / illuminant.normalization;

    // 5. Store as log for LUT
    vec3 log_xyz = log10(xyz + 1e-10f);
    return log_xyz;
}
```

**compute_density_spectral** details:
```cpp
// From agx_emulsion/model/emulsion.py:76-79
SpectralDistribution compute_density_spectral(
    const Profile& profile,
    vec3 density_cmy  // CMY ordering
) {
    // Contract density with dye extinction curves
    SpectralDistribution density_spectral[81];
    for (int λ = 0; λ < 81; λ++) {
        density_spectral[λ] =
            density_cmy.x * profile.data.dye_density[λ][0] +  // Cyan
            density_cmy.y * profile.data.dye_density[λ][1] +  // Magenta
            density_cmy.z * profile.data.dye_density[λ][2];   // Yellow

        // Add base fog/dmin
        density_spectral[λ] +=
            profile.data.dye_density[λ][3] * profile.data.tune.dye_density_min_factor;
    }
    return density_spectral;
}

// density_to_light is simple Beer-Lambert:
SpectralDistribution density_to_light(
    const SpectralDistribution& density,
    const SpectralDistribution& illuminant
) {
    SpectralDistribution light[81];
    for (int λ = 0; λ < 81; λ++) {
        light[λ] = pow(10.0f, -density[λ]) * illuminant[λ];
    }
    return light;
}
```

##### 4d. Glare Injection
```cpp
// From agx_emulsion/model/process.py:417 and emulsion.py:68-74
vec3 add_glare(
    vec3 xyz,
    const vec3& illuminant_xyz,
    const ProfileGlare& glare,
    ivec2 pixelCoord,
    const GlareCache& cache  // Full-frame glare map
) {
    if (glare.active && glare.percent > 0.0f) {
        // Sample from cached full-frame glare map
        float glare_amount = cache.glareMap[pixelCoord.y * cache.width + pixelCoord.x];
        xyz += glare_amount * illuminant_xyz;
    }
    return xyz;
}
```

**Glare RNG Strategy** (resolves tiled rendering issue):
```cpp
// From agx_emulsion/model/emulsion.py:68-74
struct GlareCache {
    std::vector<float> glareMap;  // [H*W] full-frame map
    int width, height;
    uint64_t stateHash;  // WorkingState version + seed

    // Generate once per frame when first tile is rendered
    static GlareCache generate(
        int width, int height,
        const ProfileGlare& glare,
        uint64_t rngSeed,
        uint64_t stateHash
    ) {
        GlareCache cache;
        cache.width = width;
        cache.height = height;
        cache.stateHash = stateHash;
        cache.glareMap.resize(width * height);

        // Generate lognormal noise for entire frame
        // Uses global RNG like agx (numpy default_rng)
        for (int i = 0; i < width * height; i++) {
            cache.glareMap[i] = fast_lognormal_from_mean_std(
                glare.percent,
                glare.roughness * glare.percent,
                rngSeed
            );
        }

        // Apply Gaussian blur (sigma = glare.blur)
        gaussian_filter(cache.glareMap, width, height, glare.blur);

        // Scale to [0,1] range
        for (auto& val : cache.glareMap) {
            val /= 100.0f;
        }

        return cache;
    }
};

// Store in FrameContext, generate once, all tiles sample from it
// GPU: Upload glareMap to texture, sample in shader
```

This approach:
- Matches agx (full-frame generation)
- Avoids tiling seams (single map)
- Enables GPU (upload once, sample in parallel)
- Maintains determinism (keyed by seed + state)

##### 4e. Blur and Unsharp Mask
```cpp
// From agx_emulsion/model/process.py:425-430
vec3 apply_blur_and_unsharp(
    vec3 rgb,
    const Scanner::Options& opts
) {
    // Lens blur (Gaussian)
    rgb = apply_gaussian_blur(rgb, opts.lensBlurSigmaPx);

    // Unsharp mask
    if (opts.unsharpSigmaPx > 0.0f && opts.unsharpAmount > 0.0f) {
        rgb = apply_unsharp_mask(rgb, opts.unsharpSigmaPx, opts.unsharpAmount);
    }

    return rgb;
}
```

Blur happens **after XYZ→RGB conversion** but **before CCTF encoding**.

#### Acceptance Criteria
- [ ] `ScannerOptics.cpp` contains all scanner logic, no legacy code
- [ ] LUT defaults to 17³ resolution, UI exposes `settings.lut_resolution` slider
- [ ] LUT stores `log10(XYZ + 1e-10)` matching agx
- [ ] Spectral calculation matches agx formulas exactly (denormalize → spectral → Beer-Lambert → integrate)
- [ ] `compute_density_spectral` includes `dye_density_min_factor` term
- [ ] Glare cache generates full-frame map once, tiles sample from it
- [ ] Glare RNG matches NumPy default_rng behavior (lognormal + Gaussian blur)
- [ ] Blur/unsharp apply in pixel space, no µm conversions

#### Risks & Mitigation
- *Risk*: Mixing legacy blur causes spatial inconsistencies.
  *Mitigation*: Remove all legacy blur helpers, port agx kernels verbatim, validate against agx renders.
- *Risk*: LUT precision differences produce posterization.
  *Mitigation*: Default to 17³, allow testing with higher resolutions, add validation mode comparing LUT vs. direct.
- *Risk*: Glare tiling seams with per-tile RNG.
  *Mitigation*: Generate full-frame map once, cache keyed by frame+state, all tiles sample from shared map.

---

### Step 5 – Match agx's color adaptation and output encoding

#### Tasks

##### 5a. Complete Scanner Pipeline
```cpp
// From agx_emulsion/model/process.py:391-439
vec3 density_cmy_to_rgb(
    const DensityBuffer& density_cmy,
    Medium medium,
    const WorkingState& ws,
    const Scanner::Options& opts,
    bool useLut
) {
    // 1. Select medium-specific resources
    const Profile& profile = (medium == Medium::NEGATIVE)
        ? ws.negative : ws.printPaper;
    const ScannerIlluminant& illum = (medium == Medium::NEGATIVE)
        ? ws.negativeScannerIlluminant : ws.printScannerIlluminant;
    const ScannerDensityRange& range = (medium == Medium::NEGATIVE)
        ? ws.negativeDensityRange : ws.printDensityRange;
    const ProfileGlare& glare = profile.glare;

    // 2. Normalize densities
    DensityBuffer density_cmy_n = normalize_density(density_cmy, medium, range);

    // 3. Convert to XYZ (LUT or direct)
    vec3 log_xyz;
    if (useLut && ws.scannerLutValid) {
        log_xyz = apply_lut_cubic_3d(ws.scannerLut, density_cmy_n);
    } else {
        log_xyz = spectral_calculation(density_cmy_n, medium, profile, illum, range);
    }
    vec3 xyz = pow(10.0f, log_xyz);

    // 4. Compute illuminant XYZ for glare
    vec3 illuminant_xyz = integrate(illum.curve, CMF_xyz) / illum.normalization;

    // 5. Add glare
    xyz = add_glare(xyz, illuminant_xyz, glare, pixelCoord, glareCache);

    // 6. Chromatic adaptation and color space conversion
    // From agx_emulsion/model/process.py:418-422
    vec2 illuminant_xy = XYZ_to_xy(illuminant_xyz);
    vec3 rgb = XYZ_to_RGB(
        xyz,
        outputColorSpace,      // e.g., "sRGB" or "DaVinci Wide Gamut"
        /*apply_cctf_encoding=*/false,
        illuminant_xy
    );

    return rgb;  // Linear RGB, no CCTF applied yet
}
```

##### 5b. XYZ to RGB Conversion
```cpp
// Matches colour.XYZ_to_RGB with apply_cctf_encoding=false
vec3 XYZ_to_RGB(
    vec3 xyz,
    const ColorSpace& outputSpace,
    bool applyCctfEncoding,
    vec2 illuminant_xy
) {
    // 1. Chromatic adaptation (CAT02)
    // Adapt from illuminant_xy to outputSpace.whitepoint
    mat3 adaptMatrix = compute_CAT02_matrix(illuminant_xy, outputSpace.whitepoint);
    vec3 xyz_adapted = adaptMatrix * xyz;

    // 2. XYZ to RGB matrix
    vec3 rgb_linear = outputSpace.XYZ_to_RGB_matrix * xyz_adapted;

    // 3. Optional CCTF encoding (agx applies this in separate step)
    if (applyCctfEncoding) {
        rgb_linear = apply_cctf(rgb_linear, outputSpace.transferFunction);
    }

    return rgb_linear;
}
```

**Critical**: Scanner outputs **linear RGB**. CCTF encoding happens in separate `_apply_cctf_encoding_and_clip` step.

##### 5c. Blur/Unsharp Application Order
```cpp
// From agx_emulsion/model/process.py:158-160
vec3 scan_output = density_cmy_to_rgb(density_cmy, medium, ws, opts, useLut);
scan_output = apply_blur_and_unsharp(scan_output, opts);  // On linear RGB
scan_output = apply_cctf_encoding_and_clip(scan_output, outputSpace, encodingEnabled);
```

Order: XYZ→RGB (linear) → blur/unsharp → CCTF encoding → clip [0,1]

##### 5d. Output Encoding
```cpp
// From agx_emulsion/model/process.py:432-439
vec3 apply_cctf_encoding_and_clip(
    vec3 rgb,
    const ColorSpace& outputSpace,
    bool enableCctfEncoding
) {
    if (enableCctfEncoding) {
        // Apply transfer function (e.g., sRGB 2.2 gamma)
        rgb = apply_cctf(rgb, outputSpace.transferFunction);
    }

    // Clip to [0,1]
    rgb = clamp(rgb, 0.0f, 1.0f);

    return rgb;
}
```

##### 5e. DaVinci Wide Gamut Support
Film-Juicer can offer DWG as additional output space, but:
- Core adaptation math remains identical to agx
- Only swap `outputSpace.XYZ_to_RGB_matrix` and `whitepoint`
- No additional gain, compensation, or clamping
- Parity mode uses agx's output spaces only

#### Acceptance Criteria
- [ ] Scanner outputs linear RGB matching agx's `XYZ_to_RGB(..., apply_cctf_encoding=false)`
- [ ] Chromatic adaptation uses CAT02 with viewing illuminant as source
- [ ] Blur/unsharp applied to linear RGB before CCTF encoding
- [ ] CCTF encoding in separate step matching `_apply_cctf_encoding_and_clip`
- [ ] No additional gain applied to scanner output (negative or print)
- [ ] DWG support swaps matrices only, no other changes
- [ ] Matrices precomputed with validation against Colour library

#### Risks & Mitigation
- *Risk*: Applying print exposure compensation or extra gain causes desaturation.
  *Mitigation*: Keep scanner adaptation isolated, assert no gain applied, validate against agx reference.
- *Risk*: CAT02 matrices drift from Colour definitions.
  *Mitigation*: Precompute from same whitepoints/primaries, store generator script, add init-time validation.

---

### Step 6 – Verification, tooling, and documentation

#### Tasks
- **Parity fixtures**: Create controlled test cases:
  - Step wedge (neutral gray ramp, 0.0 to 1.0)
  - Color checker (24-patch)
  - Real film scan with known profile
  - Test permutations:
    - `PrintBypass = true/false`
    - `glare.active = true/false`
    - `settings.use_scanner_lut = true/false`
    - Output spaces: sRGB, ProPhoto RGB
    - LUT resolutions: 17, 32, 64

- **Comparison methodology**:
  1. Render identical input through agx-emulsion (Python) with fixed seed
  2. Render same input through Film-Juicer with matching parameters
  3. Compare outputs:
     - ΔE2000 (perceptual color difference, target < 1.0)
     - RMS error per channel (target < 0.01 for 8-bit)
     - Histogram overlays (visual check for tonal shifts)
  4. Document tolerances in `scanner-parity-checklist.md`

- **Documentation updates**:
  - `AGENTS.md`: Explain new scanner-only workflow, parameter mapping
  - `agx-documentation.md`: Add scanner section with formulas from this plan
  - User docs: Remove references to legacy scanner toggles, explain LUT/glare/blur controls

- **Diagnostic tools**:
  - LUT validation mode: compare LUT vs. direct integration, output max error
  - Density range validation: assert matches agx formulas at startup
  - Illuminant validation: log SPD hash and normalization scalar

#### Acceptance Criteria
- [ ] All parity fixtures render with ΔE2000 < 1.0 vs. agx-emulsion
- [ ] RMS error < 0.01/channel for identical configurations
- [ ] Histograms visually match agx outputs
- [ ] `scanner-parity-checklist.md` documents all test cases and tolerances
- [ ] Docs updated, all legacy parameter references removed
- [ ] Diagnostic modes available for debugging

#### Risks & Mitigation
- *Risk*: Future refactors silently break parity.
  *Mitigation*: Version parity checklist with fixtures, require validation before merges.
- *Risk*: Users reintroduce legacy behavior via scripting.
  *Mitigation*: Remove public hooks for deprecated toggles, document supported surface clearly.

---

## Root Causes of Previous Underexposure/Desaturation

The new plan explicitly addresses the issues that plagued the old attempt:

1. **Missing `+ density_min` in negative normalization** (Step 2, 3b)
   - Old: Normalized as `density / density_max`
   - Correct: `(density + density_min) / (density_max + density_min)`
   - Impact: Shifted density range, caused underexposure

2. **Mismatched illuminants** (Step 2, 5a)
   - Old: Different SPDs for normalization, LUT, and adaptation
   - Correct: Single `profile.info.viewing_illuminant` for all scanner stages
   - Impact: Color shifts and desaturation

3. **Missing mid-gray compensation** (Step 3c)
   - Old: Print exposure without mid-gray factor
   - Correct: Re-run camera+film on 0.184 gray, normalize to green channel
   - Impact: Underexposure when exposure compensation enabled

4. **Incorrect `dye_density_min_factor`** (Step 4c)
   - Old: Missing or wrong spectral fog term
   - Correct: `density_spectral += dye_density[:,3] * tune.dye_density_min_factor`
   - Impact: Incorrect base density, color shifts

5. **Channel ordering issues** (Step 3a)
   - Old: Possibly mixed YMC/CMY ordering
   - Correct: CMY throughout, explicitly documented
   - Impact: Color channel swaps

Following these steps in order guarantees that every scanner-stage calculation (illumination, density normalization, print exposure, spectral integration, adaptation, and encoding) shares agx-emulsion's math and assets, preventing the underexposed/desaturated outcomes that plagued the previous attempt.
