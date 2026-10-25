# Diffusion construction characterization

This cohort characterizes the unchanged Film-Juicer native sample, frame and
nine-component producers at the source revision in `manifest.json`. It is
separate from the external spektrafilm reference fixtures in the parent folder.
Floating values use binary64 words; signed zeros and all 36 component values are
compared exactly. No expectations are generated during ordinary tests.

The four original native/profile captures contain 94 standalone sample cases,
35 frame cases, 92 completed-route products, 78 selected product samples and
eight primitive probes. A two-case supplement characterizes final component
overflow and bloom-power overflow through those same retained executables.
`edge-inputs.json` and the separate edge capture identities preserve the original
cohort bytes. Equal numerical payloads share storage; each context
retains its distinct executable, source and preparation identity in the manifest.
`checked.json` separately specifies closed typed failures and new safe/raw cases.
The standalone native default failure family is BlackProMist; raw failed sample
records clear every semantic field, including family, to zero.

`Ffi.Host.DiffusionFrame` exercises all three boundaries and bounded diagnostics,
clearing, fault isolation and adapter rejection. `Optics.Host.DiffusionFrameProducts`
uses completed route recipes through the Rust test-preparation facade. The strict
Release testing-OFF C11 probe executes the 507 frozen `production_cases`, binding
`production_inputs` to sample/frame metadata, identities and all component words.
The original external-reference files and their bounds are unchanged.

The capture-only maintenance dispatch was removed after independent capture. Its
retained executables accept six arguments including the executable itself:

```text
JuicerFilmRecipeTests <resource-root> tests/diffusion/frame-fixtures/inputs.json capture-diffusion-frames <preset> <output-directory>
```

They write `native.json` beneath that explicit output directory. They call the
whole original native producers; unsupported-family native undefined behavior is
excluded. Retained producers support numerical review/reproduction through slice
acceptance. Later retention follows their named consumer and existing policy.
