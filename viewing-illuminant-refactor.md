# Refactor Plan: Viewing Illuminant Parity with agx-emulsion

## Background
- Prior to this refactor, Film-Juicer mixed two different SPDs when scanning prints: the UI-driven viewing selector populated `S.printRT.illumView`/`tablesPrint`, while glare metadata and adaptation pulled the profile’s `viewing_illuminant`. agx-emulsion always drives the scanner from the profile metadata, so Film-Juicer rendered mismatched exposure/colour whenever the UI choice differed from the profile.

## Changes Implemented
1. **Single-source print viewing SPD**
   - `WorkingState::printScannerIlluminant` is now built first from the profile’s `viewing_illuminant` (with D50 fallback) and its curve feeds both `tablesPrint` and all print-scanner consumers. `tablesView` now reuses the negative scanner illuminant instead of the removed UI path.
2. **UI cleanup**
   - The “Viewing Illuminant” parameter, its overrides, and all `ParamSnapshot` plumbing were removed. Only reference and enlarger illuminants remain user-adjustable.
3. **Scanner/print helpers updated**
   - `Print::simulate_print_pixel`, `JUICER_TESTS` probes, LUT builders, and glare/XYZ conversion now read the same profile-derived SPD via `ws.printScannerIlluminant`. `Print::build_illuminant_from_choice` was simplified to handle enlarger selections only, and a reusable `load_illuminant_curve_from_choice` helper provides spectral curves when needed elsewhere (e.g., reference SPD fallback).
4. **State pruning**
   - `Print::Runtime::illumView`, `InstanceState::filmViewingIlluminant`, and reuse-context bookkeeping for the removed UI path were deleted. Scanner fallback tables no longer reference UI state.

## Validation Checklist
- Build or load a profile where `profile.viewing_illuminant` ≠ D50, render the print pipeline (with glare on/off), and compare against agx-emulsion to confirm matching exposure/colour.
- Repeat with `PrintBypass=true` to ensure negative scans (which still use `tablesScan`) remain unchanged.
- Exercise reference/enlarger illuminant overrides to confirm they continue to update `tablesRef` and `illumEnlarger` while the scanner stays pinned to the profile viewing SPD.
