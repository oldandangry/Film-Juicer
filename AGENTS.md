# Repository Guidelines

## Project Overview

**Film-Juicer** is an OpenFX 1.4 plug-in for DaVinci Resolve that implements spectral film emulation. It is a C++20 port of the [agx-emulsion](https://github.com/andreavolpato/agx-emulsion) Python reference implementation, providing physically-based negative film simulation, print paper rendering, and scanner emulation through spectral modeling. CUDA translation units compile with CUDA 13.2 as CUDA C++20 (`--std=c++20`).

The plug-in operates on spectral power distributions (SPDs) sampled at 81 wavelengths (380–780 nm, 5 nm steps), converting input RGB → spectral → film exposure → density curves → print exposure → print density → scanned RGB output.

## Project Structure & Module Organization
- The Visual Studio solution (`juicer.sln`) drives the OpenFX entry point in `main.cpp` and the processing glue in `mainProcessing.cpp`.
- Key modules: process/root (`ProcessRoot.*`), asset library (`ResourceAssetLibrary.*`), state (`JuicerState.*`), effect wiring (`JuicerEffect.*`), processing (`mainProcessing.*`), CUDA resource management (`Cuda/ResourceManager/*`, `Cuda/JuicerCudaResources*`), spectral core (`SpectralData.h`, `SpectralProcessing.h`, `FilmProcessing.h`, `ColorTransforms.h`, `SpectralContext.*`, `SpectralMathAVX.cpp`), print models (`Print.*`), and profile ingestion (`ProfileJSONLoader.*`).
- Shared headers stay header-only; keep helper logic near its domain module.
- Assets live in `Resources/` (filters, illuminants, lens data, spectral tables, JSON profiles). Runtime code resolves them from the process data directory through the asset library/`JuicerProcess::Root`; keep relative paths stable and avoid render-time file discovery.
- Build artefacts land in `juicer/x64/<Config>` while MSVC intermediates mirror in `x64/`; do not commit either directory.

## Architecture

### Core Pipeline Flow

The rendering pipeline follows the agx-emulsion reference exactly (see `agx-documentation.md` for parity rules):

1. **Input RGB** → Camera exposure (`JuicerProcessor`)
   - Spectral upsampling (Hanatos 2025 LUT or Mallett 2019 basis)
   - UV/IR band-pass filtering
   - Film sensitivity curves convolution
   - Lens blur, halation, glare

2. **Film exposure** → Film development (`mainProcessing.cpp`)
   - Log exposure → density curve interpolation
   - DIR (Development Inhibitor Release) couplers with spatial diffusion
   - Grain simulation (particle model with Poisson/binomial noise)

3. **Film density** → Print exposure (optional, unless `PrintBypass` is enabled)
   - Color enlarger with dichroic Y/M/C filters
   - Neutral filter database lookup for paper/illuminant/stock combinations
   - Mid-gray re-simulation for exposure compensation parity
   - Preflash support

4. **Print exposure** → Print development (`Print.cpp`)
   - Print paper density curves
   - Optional glare compensation removal

5. **Print/Negative density** → Scanner (`Scanner.h`)
   - Spectral → XYZ conversion with viewing illuminant
   - Glare injection (lognormal noise)
   - Output color space transform with optional CCTF encoding
   - Gaussian blur + unsharp mask

### Key Modules

**State Management** (`JuicerState.h/cpp`):
- `InstanceState`: Per-effect-instance clip/session state, pending parameter snapshots, atomic published `std::shared_ptr<const WorkingState>`, and clip-local temporal state
- `BaseState`: Immutable film stock spectral data (sensitivities, dye extinction, density curves) loaded from JSON profiles
- `WorkingState`: Immutable host-owned render recipe built from BaseState + current parameters and published by atomic shared-pointer swap; includes precomputed spectral tables for all illuminants, but not CUDA handles, frame-owned workspaces, or live mutable session state
- `ParamSnapshot`: Lightweight parameter bundle for detecting changes and triggering rebuilds

**Resource Management** (`ProcessRoot.*`, `ResourceAssetLibrary.*`, `Cuda/ResourceManager/*`):
- `JuicerProcess::Root` is the process composition root for bootstrap, asset-library access, CUDA frame preparation, context retire/reset, and ordered shutdown.
- `ResourceAssetLibrary` owns process-lifetime logical assets and validated asset payloads. Render paths consume resolved assets or published recipes, not ad hoc file discovery.
- `JuicerProcessor::FrameRequest` captures immutable per-render inputs before processing.
- CUDA durable residency is Root/context owned and keyed by exact `DeviceContextKey {deviceId, contextOpaque}` plus context epoch. Do not use instance identity, device ID alone, thread affinity, or raw file paths as substitutes.
- `PreparedCudaFrame` is the post-preparation CUDA boundary. Mutable frame scratch, staging, scan-error state, in-flight-use events, and overlap workspace are accessed through prepared-frame views/leases and released on `finish()` or `abort()`.
- Graph replay is optional and context-local; direct launch remains the correctness path. Device-wide coordination is not currently part of the architecture.

**Profile System** (`ProfileJSONLoader.h/cpp`):
- Loads JSON film stock and print paper profiles from `Resources/profiles/`
- Schema: `log_sensitivity[81×3]`, `dye_density[81×5]`, `density_curves[256×3]`, plus metadata (illuminants, DIR couplers, camera filters)
- Film stocks live in `Resources/Stock/{name}/` subdirectories with multiple tuning variants (AU/AUC/OC suffixes)
- Neutral filter database: `Resources/Print/enlarger_neutral_ymc_filters.json` maps [paper][illuminant][stock] → Y/M/C slider positions (normalized to 170 enlarger steps)
- JSON loaders must call `json_wavelengths_match_reference_axis()` and reject assets that do not match the 380–780 nm @ 5 nm grid before ingesting samples.

**Spectral Data & Context** (`SpectralData.h`, `SpectralContext.h`):
- Global spectral tables cached in the `SpectralContext` singleton with shared globals (gShape, gSens*, gIlluminantCurve, etc.)
- `lock_shape_to_reference_axis()` installs the immutable agx spectral shape; no runtime code may mutate `gShape`.
- `resample_pairs_linear_to_reference_axis()` is the default for standard illuminants/linear CSV data, while `resample_pairs_akima_to_reference_axis()` is reserved for KG3/lens/dichroic filters to mirror agx’s Akima-only paths.
- `Spectral::ingest_profile_{curve_linear,log_sensitivity}` mirror agx-emulsion’s NumPy ingestion for JSON profiles, while `normalize_illuminant_on_axis` / `load_standard_illuminant_curve` / `install_standard_illuminant` keep CSV illuminants pinned to the agx grid.
- `Spectral::SpectralTables` struct now defined here for reuse across modules
- `SpectralShape` is a constexpr `std::array<float,81>` holding the immutable 380–780 nm grid; use `assign_reference_axis()` when a mutable vector copy of the wavelengths is required, and never reallocate or overwrite the axis.
- Resampling helpers (`resample_pairs_linear_to_reference_axis`, `resample_pairs_akima_to_reference_axis`, and `build_curve_on_reference_axis_from_{linear,log10}_pairs`) implicitly target the canonical grid; they no longer accept shape parameters and must not be passed alternative axes.

**Spectral Processing** (`SpectralProcessing.h`, `SpectralMathAVX.cpp`):
- SPD reconstruction from DWG RGB using Hanatos 2025 LUT or Mallett 2019 basis with S‑matrix inversion
- Spectral table precomputation, Beer-Lambert integration, and AVX2 (`integrate_dyes_to_XYZ_avx2`) hooks
- Global processing utilities (`sigmoid_erf`, `safe_log10`, gPrecomputeStatus, mutex guards) centralised here
- `layerExposures_from_sceneSPD_with_curves()` and dyes→XYZ functions remain inline with dependency on `SpectralData.h`
- Include after `SpectralData.h` to satisfy dependency hierarchy

**Film Processing** (`FilmProcessing.h`):
- `NegativeCouplerParams` and masking helpers for negative film simulation
- Layer exposure aggregation, H‑D curve sampling, and exposure→density conversions
- DIR pre-mask application and density curve interpolation kept header-only for parity
- Includes only `SpectralData.h`; depends on spectral tables built upstream

**Color Transforms** (`ColorTransforms.h`):
- Input color space enumeration, film raw configuration, and canonical RGB↔XYZ matrices
- CAT02 chromatic adaptation and DWG conversion utilities
- Film raw preprocessing (`rgb_input_to_film_raw`, `compute_film_raw_midgray`) built on `SpectralData` tables
- Forward declares `Spectral::SpectralTables`; include `SpectralData.h` when table access is needed

**DIR Couplers** (`Couplers.h`):
- Development Inhibitor Release simulation with inter-layer diffusion
- `Couplers::Runtime` stores per-channel diffusion matrices and Gaussian blur kernels
- Spatial mode (`spatialSigmaMicrometers > 0`) applies separable Gaussian convolution to inhibitor densities before re-interpolating density curves
- Pre-correction mode (`couplersPrecorrect=true`) bakes DIR effects into density curves during WorkingState rebuild for performance

**Print System** (`Print.h/cpp`):
- `Print::Profile`: Print paper spectral dyes, density curves, glare compensation metadata
- `Print::Runtime`: Precomputed dichroic filter transmittance curves, illuminant SPDs
- `rebuild_print_runtime()`: Builds per-paper runtime by loading Durst digital light dichroic filter spectra and computing neutral positions
- `load_profile_from_dir()`: Loads CMY dye eps, baselines, sensitivities, and density curves exclusively from `Resources/profiles/*.json`; missing JSON now aborts with a `JTRACE` rather than resampling CSVs.
- Enlarger illuminant selection: D50/D55/D65/TH-KG3-L (tungsten halogen + KG3 heat filter + longpass)

**Processing** (`mainProcessing.h/cpp`, `JuicerEffect.cpp`):
- `JuicerProcessor`: Multi-threaded pixel processor inheriting from `OFX::ImageProcessor`
- `multiThreadProcessImages()`: Tile-based rendering with auto-exposure caching
- Auto-exposure: Center-weighted Gaussian mask (σ=0.2) on luminance, clamped to prevent inf EV
- Render scaling: Tracks `pixel_size_um` for μm-to-pixel conversions (lens blur, halation radii, DIR diffusion)
- CUDA rendering prepares through `Root::prepare_cuda_frame()` and consumes prepared-frame views after preparation rather than adding new public `ensure_*` chains in the render path.

**Scanner** (`Scanner.h`):
- Configurable auto-gain (when `PrintBypass=true`) normalizes film D-max to target Y
- Optional bypass mode for direct negative viewing
- Film format parameter (`scanFilmLongEdge`) determines pixel → micrometer scaling for grain/diffusion

## Build, Test, and Development Commands
- Host C++ builds are C++20. CUDA kernels/manual CUDA compile steps use CUDA 13.2 with CUDA C++20 (`--std=c++20`); do not downgrade host-side checks, fallback compiler args, or clang-tidy setup to C++17.
- MSVC 2026 / Visual Studio 18 is installed outside the Windows system PATH. Call tools by direct path instead of relying on `msbuild`, `clang-cl`, `clang-tidy`, or `clang-format` being discoverable:
  - `C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe`
  - `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe`
  - `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-tidy.exe`
  - `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-format.exe`
- CUDA 13.2 is the required CUDA toolkit. Release-Clang uses the VS 18 direct MSBuild/ClangCl paths for host C++, and manual `nvcc` uses the active VS 18 `VCToolsInstallDir` MSVC host compiler unless `JuicerNvccHostCompilerBin` is explicitly overridden with a tested direct compiler path.
- `"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" juicer.sln /p:Configuration=Debug /p:Platform=x64` — builds the plug-in and leaves symbols in `juicer/x64/Debug`.
- `"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" juicer.sln /p:Configuration=Release /p:Platform=x64` — produces the optimised plug-in for host validation, emitting OFX recipes under `juicer/x64/Release`.
- `"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" juicer.sln /p:Configuration=Release-Clang /p:Platform=x64` — required Clang build path for spektrafilm evidence; use the VS 18 direct path so ClangCl is found.
- `"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" juicer.sln /t:Clean /p:Platform=x64` — clears intermediates before switching branches or configurations.
- If invoking PowerShell tooling from WSL, set `PATHEXT` to include `.EXE` and prepend the VS 18 LLVM bin directory to `PATH`; WSL-launched Windows PowerShell may not inherit normal Windows command-extension lookup.

## Coding Style & Naming Conventions
- Follow the existing MSVC layout: four-space indentation, braces on the same line as the control statement, and no trailing commas in initializer lists.
- Use `PascalCase` for types, `snake_case` verbs for free functions, and reserve `k` prefixes for constants mirrored from upstream specs.
- Keep includes grouped (standard, third-party, project) and use `"..."` for project headers.
- Add brief comments only for non-obvious data flows or invariants in shared headers.
- For implementation hygiene, follow `docs/code-hygiene.md`; for resource-management architecture, use `docs/rm/resource-manager-simplification-plan.md` and `docs/rm/resource-manager-architecture-decisions-pass.md`.

## Testing Guidelines
- There is no standalone test harness yet; edge cases are covered through host integration and the spectral resampling helpers.
- `JUICER_TESTS` is deprecated and must not be added or revived.
- Validate changes through the real-project workflow: build in `C:\Dev\Juicer\`, run targeted manual checks in the host, and capture trace or evidence when relevant.
- Validate spectral data changes by running the Release build inside the target host and comparing print or spectral previews against reference assets in `Resources/`.
- Resource-management or CUDA lifetime changes need owner-run Debug/Release builds and targeted host evidence for overlap, abort, context reset/retire, or shutdown when relevant.

## Commit & Pull Request Guidelines
- Match the Git history: summaries are single-sentence, present-tense descriptions of the behavioural change (e.g., “Ensure log10 curve resampling exponentiates finite samples…”).
- Reference key modules touched, call out data migrations, and mention the user-facing fix in the body. Link issue IDs when available.
- Pull requests should include a problem summary, test notes (host, configuration, sample asset), and visuals or spectral plots when behaviour changes.
