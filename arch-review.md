# Architecture Review – New Scanner Refactor

This document reviews `new-scanner-refactor.md` with a focus on:
- Architectural soundness and fit with the existing Film‑Juicer pipeline.
- Parity with **agx-emulsion** (math, data flow, and asset usage).
- Suitability of the design for a future GPU implementation.
- Opportunities to simplify or de‑risk the design without losing parity.
- Alternative choices worth considering before implementing the refactor.

The intent is to validate the direction and flag areas where a lighter or more incremental design might deliver the same parity with less complexity or risk.

---

## 1. High-level Assessment

- The proposed architecture is broadly sound and consistent with the rest of Film‑Juicer:
  - It preserves the existing **BaseState → WorkingState → per‑render runtime** layering.
  - It keeps spectral data single‑sourced from profiles and the fixed agx spectral axis.
  - It separates long‑lived media state (per‑medium runtime) from per‑frame geometry/RNG.
- The plan is very parity‑driven:
  - Density ranges, glare metadata, illuminants, and LUT math are all anchored in explicit references to agx‑emulsion functions and formulas.
  - The proposed JSON schema expectations match the current agx profiles rather than inventing new behaviour.
- The GPU‑facing abstractions (SoA density slabs, `ScannerBackend`, LUT upload contracts) are forward‑looking and should map cleanly to CUDA/Metal/Vulkan later, provided they stay thin.
- The main concerns are:
  - **Complexity and surface area**: many small structs and hash keys (`ScannerMediumRuntime`, `ScannerIlluminant`, `ScannerDensityRange`, `ColorRuntime`, `SpectralLutBuffer`, `DensityBuffer`, `DensitySlabPool`, `DeviceDensitySlab`, multiple hashes) risk making the implementation harder to reason about and test.
  - **Performance of per‑frame hashing** if implemented literally (e.g., hashing full CMY planes every frame).
  - **Coupling to an unimplemented GPU backend**: some design choices are driven by hypothetical GPU behaviour that could be refined once there is a concrete backend.

Overall: the direction is good and compatible with the existing architecture and agx parity goals. The refactor is ambitious; careful scoping and a few targeted simplifications would reduce risk without compromising parity.

---

## 2. Architecture Soundness

### 2.1 State and Runtime Layering

**What works well**
- Keeping scanner‑relevant media data in `WorkingState` via `ScannerMediumRuntime` is aligned with how film stock and print runtime are handled elsewhere:
  - Profile → `BaseState` / `Print::Profile` remains immutable.
  - `WorkingState` holds derived, per‑parameter media runtimes (spectral tables, density curves, glare/print compensation).
  - `ScannerMediumRuntime` is a focused per‑medium slice referencing those tables plus scanner‑specific metadata.
- The immutable → clone → optional‑mutation flow is clearly defined:
  - JSON → immutable profile → `WorkingState` clone → optional glare‑compensation removal → density‑range capture → read‑only during rendering.
  - This mirrors agx’s `_apply_profiles_changes()` and avoids double‑removal.
- Separating:
  - `Scanner::Options` (per‑render optics sliders: blur, unsharp).
  - `Scanner::Settings` (LUT toggle, LUT resolution).
  - `ScannerMediumRuntime` (media‑specific spectral and density metadata).
  - `ScannerOptics::Runtime` (LUTs and optics caches keyed by medium/settings).
  …is clean and gives each layer a clear responsibility.

**Potential issues / clarifications**
- `ScannerMediumRuntime` currently bundles many concerns:
  - Media identity (`mediumTag`).
  - Density range.
  - Glare metadata.
  - Illuminant.
  - Grain metadata.
  - Pointers to spectral tables and color runtimes.
  - Multiple hashes plus a validity bit.
  - This is reasonable, but it will be important to keep usage patterns disciplined:
    - Any mutation to these fields must occur only in `rebuild_working_state()`.
    - Rendering and optics layers must treat the struct as strictly read‑only.
- The plan assumes that `WorkingState` always materialises **exactly two** `ScannerMediumRuntime` instances per build (Negative, Print):
  - This is simple and symmetric, and matches the agx view of the world.
  - It does mean that adding an extra medium type later (e.g., experimental modes) will require plumbing a third entry; that is likely acceptable given parity constraints.

### 2.2 Density Staging and Pipeline Split

**What works well**
- The Stage 1/2/3 split:
  1. RGB → negative densities (DIR + grain).
  2. Optional print simulation → print densities.
  3. Scanner optics from precomputed densities.
  - This ordering mirrors agx’s conceptual stages and removes the current “RGB→density inside scanner” coupling.
- `DensityBuffer` as a **SoA CMY buffer** with explicit:
  - `clipBounds` (Resolve window).
  - `paddedBounds` (halo‑inflated bounds).
  - Identical strides across planes.
  - This layout is good for both CPU vectorisation and GPU (one descriptor for both).
- The explicit halo calculations for DIR, grain blur, lens blur, unsharp, and glare are a nice improvement:
  - The design makes convolution footprints explicit rather than hidden inside each kernel.
  - Halos are clamped against `srcImg->getBounds()` with a clear failure mode when the host does not provide enough border pixels.

**Potential issues / simplifications**
- `DensitySlabPool` and the lease/fence machinery are quite heavy for the current CPU‑only implementation:
  - For now, Resolve runs one processor per render and full‑frame windows; two persistent `DensityBuffer` instances per `JuicerProcessor` (negative/print) would probably be sufficient.
  - A pool and per‑lease fences make more sense once multiple GPU queues and concurrent tiles/frames are in play.
  - Suggestion: start with simple per‑processor buffers, and introduce a pool abstraction only when you actually implement GPU slabs.
- `slabHash` and the per‑frame hashing of CMY planes:
  - `slabHash = hash_pack(…, hash_float_span(planes[0]), hash_float_span(planes[1]), hash_float_span(planes[2]))` is O(N pixels) extra work per frame.
  - For 4K CMY buffers, this is a non‑trivial cost and duplicates work the scanner already did to compute densities.
  - The hash is mainly valuable for **debug parity checks** (CPU vs. GPU) and perhaps some cache diagnostics.
  - Suggestion: gate slab hashing behind a debug or `JUICER_TESTS` flag rather than making it unconditional in release builds.

### 2.3 Hashing and Cache Invalidation

**What works well**
- Using `WorkingState::buildCounter` as the umbrella invalidation source is consistent with the rest of the plug‑in:
  - It avoids any reliance on pointer identity.
  - It keeps all caches logically tied to parameter changes.
- `mediumSpectralHash` vs. `mediumOpticsHash` separation:
  - `mediumSpectralHash` for “just the spectral/density inputs” (tables, density ranges, illuminant SPD).
  - `mediumOpticsHash` for glare parameters and color adaptation (CAT02/XYZ).
  - This split correctly distinguishes LUT content (spectral) from optics kernels/RNG (optics).
- `SpectralLutBuffer` containing both a **cache key** and a **content hash** is a good pattern:
  - The cache key drives rebuild/purge logic.
  - The content hash fingerprints actual samples for GPU upload parity.

**Concerns and potential improvements**
- The definition of `mediumSpectralHash` currently includes `buildCounter`:
  - `mediumSpectralHash = hash_pack(buildCounter, mediumTag, hash_float_span(illuminant.samples), …)`
  - This means even if spectral content and density ranges remain identical across rebuilds, the hash will churn whenever unrelated parameters change.
  - That partially defeats the idea of a “content‑only” spectral hash.
  - Suggestion:
    - Define `mediumSpectralHash` purely from content (`mediumTag`, illuminant samples, table CMY curves, density range).
    - Fold `buildCounter` in only where you actually need versioning, e.g. `cacheKey = hash_pack(buildCounter, mediumSpectralHash, scannerSettingsHash)`.
    - This keeps `mediumSpectralHash` a reusable, “what”‑only fingerprint and simplifies reasoning about cache reuse.
- Similarly, `slabHash` and any `GeometryRuntime.hash` should be used sparingly:
  - It is easy to over‑hash and inadvertently rebuild more than necessary.
  - Focus on a **minimal set of stable keys**:
    - Spectral inputs + density ranges.
    - Optics parameters (glare, blur, unsharp, color runtime).
    - Geometry (frame bounds, halo sizes).

### 2.4 Color and Illuminant Handling

**What works well**
- The refactor extends the already‑implemented viewing‑illuminant parity:
  - Scanner illuminants are built from `profile.info.viewing_illuminant` via CSV curves pinned to the agx spectral axis.
  - UI overrides for viewing illuminant are removed, matching agx’s use of profile metadata.
- `ScannerIlluminant` as a value type (curve + normalization + white XYZ/xy) is appropriate:
  - It avoids dangling pointers across working‑state swaps.
  - It decouples scanner illuminant consumption from the rest of the illuminant system.
- The color adaptation plan in Step 5 is strong:
  - Precompute CAT02 and `XYZ→RGB` matrices offline via Colour, commit them as generated code.
  - Produce a `ColorRuntime` per medium at `WorkingState` rebuild time.
  - Restructure `OutputEncoding` to assume its input is already in the target output space (encode‑only), matching agx.
  - Support DWG linear as an explicit output option via the `inputIsOutputSpace` flag.

**Potential issues / clarifications**
- The generator script introduces a dependency on a specific Colour version and its interpretation of primaries/whites:
  - This is acceptable, but it is important to document the exact Colour version and regeneration procedure (as the plan suggests) and to add sanity checks (white → white mapping).
  - Suggestion: add a debug‑mode self‑test that verifies each generated matrix maps the reference white to itself within a tight epsilon.
- The refactor assumes all scanner consumers will switch to the new adaptation path:
  - Be careful to audit any non‑scanner usages of `OutputEncoding` so that they do not rely on the old DWG→target multiply path.
  - Consider a temporary `JUICER_TESTS` hook to compare “old OutputEncoding” vs “new adaptation + encode” on a small set of known inputs.

---

## 3. Parity with agx-emulsion

### 3.1 Strengths

- The plan is very explicit about mirroring agx:
  - Density ranges:
    - Negatives: `np.nanmax(negative.density_curves, axis=0) + grain.density_min`.
    - Prints: `np.nanmax(print.density_curves, axis=0)`.
  - Glare compensation removal: reusing `remove_glare_compensation_from_curves()` in the same lifecycle slot as agx’s `_apply_profiles_changes()`.
  - Illuminant normalization and `∑SPD·ȳ` usage in `ScannerIlluminant`.
  - `_spectral_lut_compute`, `_density_cmy_to_rgb`, and glare math to be ported line‑for‑line.
  - LUT storage as `log10(XYZ + 1e-10)` matching agx’s `compute_with_lut`.
- JSON schema expectations match the agx profiles:
  - Required `negative.grain` and `glare` blocks; missing data is a fatal profile error.
  - CMY ordering and `density_min` semantics match the Python reference.
- The plan includes a dedicated parity verification step:
  - Captured fixtures and ΔE/RGB error thresholds.
  - Explicit LUT vs. direct spectral integration comparisons.

### 3.2 Risks / subtle parity traps

- **Numeric differences**:
  - agx runs in NumPy `float64`, Film‑Juicer in `float32`, so some differences are inevitable.
  - The plan mitigates this with tolerances, but worth explicitly expecting small ΔE/RGB drift even with identical formulas.
- **Ordering and normalization**:
  - It is crucial that the order “clone → optional glare removal → density range capture → normalization/denormalization” exactly matches agx.
  - The plan is careful about this, but the implementation will need thorough cross‑checks (especially for print glare compensation).
- **LUT vs. direct spectral path**:
  - There are two code paths in agx (direct integration and LUT).
  - The plan correctly proposes a validation path comparing LUT vs. direct integration when caches rebuild; this will be important to keep.
- **Profile coverage**:
  - Treating missing glare metadata as fatal is parity‑correct, but it will break any “non‑agx” community profiles.
  - This is acceptable given the stated parity goals, but should be called out clearly in user‑facing docs.

Overall: if the implementation follows the plan literally and the parity fixture suite is built, the architecture is capable of achieving numeric parity with agx‑emulsion.

---

## 4. GPU Port Suitability

### 4.1 Positive aspects

- The density staging layer is well‑suited to GPU:
  - SoA CMY planes, explicit strides and bounds, and halo‑aware `paddedBounds` are exactly what GPU kernels want.
  - Having Stage 1/2 produce CMY densities once, then reuse them for Stage 3 optics, avoids redundant CPU work and supports cheap device uploads or shared heaps.
- `ScannerBackend` as an abstraction:
  - `DeviceDensityView` with device pointer + fence is a natural fit for CUDA/Metal/Vulkan interop.
  - A single `ScannerBackend` on `InstanceState` allows you to switch between CPU‑only and GPU‑backed implementations without touching OFX glue.
- The LUT handling design is GPU‑friendly:
  - CPU always computes LUTs once.
  - GPU uploads are keyed by `{cacheKey, contentHash}` and decoupled from CPU caches.
  - `lut_ready()` / `has_device_lut()` give a clear contract for when GPU optics can run.
- RNG design:
  - Deriving seeds from `{clipToken, frameTimeBits, buildCounter, medium, channel, coords}` is deterministic and compatible with both CPU and GPU kernels.

### 4.2 Concerns / possible refinements

- The current plan partially optimizes for future GPU functionality that does not yet exist:
  - `DensitySlabPool`, `DeviceDensitySlab`, and fence management will only be exercised once you have actual GPU kernels.
  - Until then, they add implementation and testing surface without delivering runtime benefits.
  - Consider implementing GPU support in two phases:
    1. **CPU‑only** optics with the new staging layer and `ScannerOptics` entry point.
    2. Once stable, add a minimal `ScannerBackend` implementation and only then introduce pools/queues/fences as needed.
- `DeviceDensityView` as a single struct for all APIs is convenient but may need refinement:
  - Some APIs (Metal shared heaps vs. CUDA external memory) have different ownership semantics.
  - Leave room in the design for backend‑specific fields without over‑specifying them now—e.g., an opaque `BackendHandle` inside the view.

Overall, the architecture is a good foundation for GPU porting. The key is to keep the first implementation CPU‑centric and evolve the GPU contracts incrementally once the CPU path is proven and covered by parity tests.

---

## 5. Simplifications and Cleanups (Without Losing Parity)

The following changes would make the implementation simpler and more maintainable while preserving the parity guarantees in the plan:

1. **Separate versioning vs. content hashes**
   - Make `mediumSpectralHash` “content only” (no `buildCounter`) and derive cache keys as `hash_pack(buildCounter, mediumSpectralHash, scannerSettingsHash)`.
   - Do the same wherever possible: use content hashes for fingerprinting data, then combine with `buildCounter` only when you need to tie caches to a specific `WorkingState`.

2. **Gate large buffer hashing behind debug**
   - Keep `slabHash` and any per‑plane `hash_float_span(planes[i])` behind a `JUICER_DEBUG_SCANNER` or `JUICER_TESTS` flag.
   - In release builds, either skip slab hashes entirely or compute a cheaper checksum (e.g., a running XOR or a hash of a small sample).

3. **Start without a slab pool**
   - Initially, allocate two `DensityBuffer` instances per `JuicerProcessor` and reuse them across frames.
   - Introduce `DensitySlabPool` and per‑lease fences only when GPU optics are actually implemented and you need to manage shared device memory across processors.

4. **Keep the GPU backend interface minimal**
   - For the first pass, `ScannerBackend` only needs:
     - `upload_lut(cacheKey, contentHash, span, queue)`.
     - `acquire_density_view(const DensityBuffer&, queue)` (no pooling).
   - Additional functionality (pools, external semaphores, shared heaps) can be added later once profiles of real GPU kernels justify the extra complexity.

5. **Limit the number of distinct hash types**
   - Where possible, consolidate hashes:
     - One spectral hash per medium.
     - One optics hash per medium.
     - One geometry hash per frame.
   - Avoid proliferating separate hashes for closely related concerns unless they are clearly needed.

6. **Use simple structs for scanner options/settings**
   - Keep `Scanner::Options` and `Scanner::Settings` as small POD structs, and avoid adding scanner‑specific toggles to `ParamSnapshot::hash_params()` unless they truly require a `WorkingState` rebuild.
   - This aligns with agx where `params.scanner` is a per‑render DotMap, not a profile‑level knob.

---

## 6. Alternative Choices Worth Considering

Before committing to the full refactor, you may want to consider:

1. **More incremental rollout of the staging layer**
   - Option A (current plan): introduce `DensityBuffer` + Stage 1/2/3 + scanner refactor + LUTs all in one sequence.
   - Option B (simpler first step):
     - First, refactor the scanner to use `ScannerMediumRuntime` and per‑medium metadata while still consuming today’s AoS densities.
     - Then introduce `DensityBuffer` and Stage 1/2/3, but keep optics purely CPU for one version.
   - Benefit: you can validate parity with agx earlier, with fewer moving pieces, and reduce the risk that density staging bugs obscure scanner issues.

2. **Treat GPU support as a follow‑up refactor**
   - Finalize the CPU architecture, parity tests, and optics TU first.
   - Once parity is locked in and the CPU path is stable, introduce GPU kernels and the full `ScannerBackend`/pool abstraction in a separate branch of work.
   - This keeps the scanner refactor focused on correctness and agx parity, and isolates GPU‑specific risk.

3. **Simpler parity instrumentation**
   - Instead of depending heavily on plane hashes for CPU vs. GPU parity, you can:
     - Use a small set of deterministic fixtures with captured CPU outputs.
     - Compare GPU outputs to those fixtures using ΔE/RGB thresholds.
   - Hashes remain useful for debugging, but the primary parity signal comes from image‑level comparisons.

4. **Single “ScannerKey” struct for cache keys**
   - Consider defining a small `ScannerKey` type containing:
     - `mediumTag`.
     - `mediumSpectralHash`.
     - `mediumOpticsHash`.
     - `scannerSettingsHash`.
   - Cache entries (LUTs, optics runtimes) can then take a `ScannerKey` instead of recomputing/folding separate hashes, which simplifies reasoning and logging.

---

## 7. Summary

- The architecture in `new-scanner-refactor.md` is conceptually solid and matches both Film‑Juicer’s existing state model and agx‑emulsion’s scanning pipeline.
- The plan is capable of delivering close numeric parity with agx, provided the outlined formulas and ordering are implemented exactly and backed by fixture‑based tests.
- The GPU‑oriented parts of the design are generally well‑thought‑out but could be introduced more incrementally to avoid unnecessary complexity in the initial CPU implementation.
- The main opportunities for simplification are:
  - Reducing and clarifying hash usage.
  - Avoiding per‑frame full‑buffer hashing in release builds.
  - Starting with simple per‑processor density buffers and a minimal GPU backend interface.
- None of these simplifications compromise parity; they primarily reduce implementation and debugging overhead while keeping the core design aligned with agx‑emulsion and future GPU porting needs.

