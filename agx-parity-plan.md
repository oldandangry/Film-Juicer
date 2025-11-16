# Film-Juicer Spectral Parity Plan

This plan disables all runtime spectral-axis flexibility so that Film-Juicer consumes agx-emulsion assets exactly as the Python reference does. The goal is to stop re-interpolating JSON film/paper data and only run Akima splines in the same places agx-emulsion does (CSV-loaded filters).

## 1. Parity Goals
- **Fixed spectral axis:** Lock `Spectral::gShape` to 380–780 nm in 5 nm steps (config constants already exist).
- **No runtime resample of profile JSON curves:** Use the samples verbatim; treat Akima as an offline asset-generation detail.
- **Filter CSVs still use Akima:** Keep parity with agx-emulsion’s `load_dichroic_filters` / `load_filter`.
- **NaN handling matches reference:** Replace NaNs with zeros (or the same scalar the Python code uses) rather than resampling around them.
- **Runtime asset scope:** Only support officially shipped JSON/CSV bundles; remove code paths that expect arbitrary user-provided spectral axes.

## 2. Current Deviation Summary
1. **Dynamic spectral axis:** `Spectral::gShape` can be rebuilt from CMF CSVs (`SpectralData.h:479-545`). Film/paper loaders re-sample into whatever axis is active.
2. **Akima on every profile load:** `build_curve_on_shape_from_linear_pairs` / `build_curve_on_shape_from_log10_pairs` instantiate `AkimaInterpolator` when loading film stocks, print papers, baselines, etc. (`JuicerState.cpp`, `Print.cpp`, `SpectralData.h`).
3. **NaN sanitization via reinterpolation:** Missing samples are skipped by Akima, so the runtime never hits `NaN`.
4. **Optional user CSVs:** Helpers such as `set_filter_KG3_from_pairs` resample arbitrary wavelength/value inputs onto the working axis.

## 3. Refactor Tasks
### Phase A – Lock the Spectral Axis
1. ✅ Removed `initialize_spectral_shape_from_csv`/CMF overrides; `lock_shape_to_fixed_grid()` now pins `gShape` to the agx axis during bootstrap.
2. ✅ Axis selection UI/runtime hooks are gone; with the grid pinned there are no remaining parameters or CSV overrides that can replace the working wavelengths.
3. ✅ `precompute_spectral_tables_locked_body` now assumes the fixed grid and no longer rebuilds or falls back to alternate shapes.

### Phase B – Streamline Profile Loading
1. ✅ `ProfileJSONLoader` already preserved NaN placeholders; loaders now rely on locked-axis helpers to zero them out during ingestion (parity with agx-emulsion’s `np.nan_to_num`).
2. ✅ `JuicerState::load_film_stock` and `Print::load_profile_from_dir` copy JSON samples straight into runtime curves via `Spectral::ingest_profile_{curve_linear,log_sensitivity}`, clamping negatives/NaNs without Akima.
3. ✅ All film/paper `build_curve_on_shape_from_*` usages were removed (helpers remain only for CSV filter code), so profile loading no longer resamples or pads spectra at runtime.

### Phase C – Filter/CSV Handling
1. ✅ `Print::load_dichroic_filters_from_csvs` remains unchanged; other illuminant builders now require locked-axis CSVs and skip Akima entirely.
2. ✅ KG/lens CSV handling is the sole Akima user: TH-KG3-L and dichroic filter readers still resample via `resample_pairs_to_shape`, while all opt-in filter setters were removed.

### Phase D – Clean Dead Flexibility
1. ✅ Eliminated `Spectral::set_illuminant_from_pairs`, `set_filter_*_from_pairs`, and the other runtime resampling helpers so only agx-style assets are supported.
2. ✅ Updated documentation to make it clear that the plug-in only accepts agx-style assets on the locked 380–780 nm grid (CSV resampling now limited to bundled filter data).

## 6. Implementation Notes
- Detailed Phase C/D implementation notes (per-file changes, behavioural shifts, and validation tips) are consolidated in `spectral-shape-refactor.md`.

## 4. Validation Strategy
1. Compare rendered frames before/after using the same agx profiles to ensure no unintended drift (differences should be due only to the removal of NaN interpolation).
2. Verify that NaN-heavy JSONs (e.g., Fujifilm C200) still load and render once NaNs are replaced with zeros, as agx does.
3. Confirm dichroic filter CSVs still match the reference by dumping sampled transmittance arrays and diffing against agx outputs.

## 5. Rollout Notes
- Expect slight spectral differences wherever the previous Akima pass filled NaNs with curve fits; document this as “matching agx reference”.
- Update AGENTS.md / documentation to note that the plug-in now requires agx-style assets and no longer supports arbitrary spectral grids.
- After refactor, delete unused classes/functions to keep the codebase aligned with the new constraints.
