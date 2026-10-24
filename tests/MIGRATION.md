# Public test-suite migration record

Inventory basis: repository commit `40da0079a066d2426ff8cb5c41e77de6ccc2c8bc`,
with a clean working tree before migration. This is the bounded family
inventory required for consolidation; it is not a general coverage audit.

Migration status: complete on 2026-09-19. All applicable automated and
owner-run acceptance validation passed on Windows and native Linux. The
explicitly deferred families and reference regeneration below remain outside
this migration's acceptance scope; they are not failed or outstanding gates.

## Selected and retained families

| Origin | Protected behavior and prerequisites | Disposition |
| --- | --- | --- |
| `validation/scatter_halation/ScatterHalationValidation*` and `fixtures/` | Control canonicalization, recipe/hash identity, profile metadata, frame descriptors, admission/state, preparation, CUDA operators, routes, carrier reuse, zero work, lifecycle; CUDA groups require driver/device | Moved to `tests/scatter_halation/`. A driver-free GoogleTest host slice now exposes the core recipe/descriptor/bootstrap contracts. The production-linked groups remain explicit GPU CTest processes. Performance is an unregistered benchmark. |
| `validation/gamma/run_validation.py` fused-equivalence logic | Per-case split/fused bounds, explicit applicability, legacy-direct restriction, non-finite rejection and failure propagation; Python only | Replaced by the focused public `tests/gamma/compare.py` and `test_compare.py`. The old runner remains with the deferred capture family because its other consumers are not current ordinary tests. |
| `validation/ofx_host_compat/` | Linux synthetic OFX Load/Describe/Unload, symbol boundary and CUDA-support declaration; no rendering/device | Moved to `tests/ofx/` and registered as a Linux `host` test. Python `assert` verdicts were replaced by explicit failures so `python -O` cannot hide them. The installed Resolve-library preload passed during owner-run Linux qualification. |
| `tools/diffusion_reference/` host behavior and reference cohort | Resolver, PSF component, radius, authored-hash and invalid-input behavior against spektrafilm revision `48645a…`; host execution, Python/Numpy translation at build | Moved to `tests/diffusion/`. Bespoke VS compilation was replaced by CMake. The committed NPZ stayed byte-identical; path-sensitive provenance changes are recorded in the domain README. Reference-generation scripts/tests are public but not ordinary CTest work. |

Pre-migration evidence on the same source/resource state:

- Scatter `host-gate-1`: 57/57 cases passed in Linux Debug.
- Gamma runner tooling tests: 22/22 passed under the available Python 3.12
  environment. The migrated ordinary suite intentionally requires and tests
  Python 3.13.
- Scatter focused CUDA: provenance and five descriptor/comparator checks
  passed; 18 device/operator checks failed at allocation with
  `CUDA driver version is insufficient for CUDA runtime version`. This was the
  result observed in the then-managed Codex command sandbox, not evidence that
  the host NVIDIA driver was obsolete. Later unrestricted validation below
  passed the same reference and all other GPU groups.

## Explicitly deferred or private families

| Origin | Disposition and named reason |
| --- | --- |
| `validation/gamma/` frozen captures, executable, state/lifetime runner, staging tests and reference generator | Deferred as one historical evidence family. The frozen inventory differs from current Portra 400 and neutral-print resources, and only `Release`/`Release-Clang` configuration identities exist. No current four-preset mapping or independent new baseline is available. `enlarger-diffusion-black-pro-mist-half` is configuration-invalid because diffusion was disabled; `spatial-dir-ramp-edge-direct` is configuration-invalid because unrelated halation/unsharp remained active. Neither is acceptance coverage. |
| `validation/scatter_halation/ExperimentalReleaseValidation*` | Retained privately. These cross-domain spectral/output/route investigations depend on large private cohorts and are not an ordinary scatter/halation unit. |
| `validation/scatter_halation/ExperimentalDirValidation*` | Retained privately. This is the experimental DIR cohort, not part of the selected scatter family; it requires separate public fixture/provenance review. |
| `tools/diffusion_cuda_harness/` | Deferred. Current GPU/convolution/lifecycle checks depend on a path-bound harness and a separately generated 1.4 GB convolution bundle that is not a distributable ordinary fixture. Profiling also needs separation before migration. |
| `tools/dir_filter_harness/` | Deferred. The root mixes production helper checks, algorithm candidates, image qualification and timing; current production-contract cases need case-level separation from research candidates. |
| `.juicer-local/scripts/tests/` | Partly retired on 23 October 2026; see the later exact case map. Private-tool tests and explicitly limited investigations remain; obsolete layout checks and four standalone compiler wrappers are removed. Unmatched C++ oracles remain inert at their original paths. Structural assertions were not republished as behavior tests merely to increase counts. |
| `.juicer-local/scripts/spektrafilm_*` and `.juicer-local/validation/` | Retained privately as reference generation, historical evidence and investigations. Ordinary public tests have no dependency on them. |
| `.juicer-local/scripts/perf/` | Runtime-metrics capture and its interpretation remain private investigation tools. The unused P0–P8 phase framework, dormant Windows bridge and P2 cleanup script were retired on 24 October 2026; see the bounded disposition below. Only the current scatter/halation measurement mode received a public benchmark procedure. |

These are explicit deferrals, not passing coverage. The remaining CUDA,
resource-lifetime and DIR families need supported hardware and/or a complete,
reviewable public fixture unit before promotion.

## Cleanup and archive

- `validation/scatter_halation/CMakeLists.txt` was superseded by the public CTest
  integration and archived at its original relative path.
- `tools/diffusion_reference/build_host_behavior_contract.cmd`, its old runner
  test and its local-only README were superseded by CMake/public documentation
  and archived at their original relative paths.
- `.juicer-local/scripts/spektrafilm_validate_scatter_halation_structure.py`
  was archived after the public targets passed. Its source-layout assertions
  referred to the retired validation paths and are replaced by production-linked
  behavior tests plus CMake target ownership.
- Retained sources/fixtures were moved, not copied. The public build and tests
  do not read `ArchiveOldTestingFiles/`, `validation/`, `tools/`,
  `.juicer-local/` or `external/` during ordinary execution.
- `ArchiveOldTestingFiles/README.md` records local archive origins and
  replacements. The single root ignore rule keeps the archive unpublished.

## Migration validation

Linux validation used CUDA Toolkit 13.2.86, GCC/G++ 13.3, CMake 4.3.1,
Ninja, and Python 3.13.13 with NumPy 2.4.6. Native Windows validation used
CUDA Toolkit 13.2.51, Clang/LLVM 22.1.3, MSVC 19.51, CMake 4.3.3, Ninja, and
Python 3.13.14 with NumPy 2.4.6.

Post-handoff native Arch Linux validation used CUDA Toolkit 13.2.51 selected
through `CUDACXX`/`CUDAToolkit_ROOT`, GCC/G++ 13.4.1, CMake 4.4.3, Ninja
1.13.2, Python 3.13.15 with NumPy 2.4.6, NVIDIA driver 615.71.09, and an RTX
4080. The repository presets no longer encode a Linux toolkit installation
path; the existing configure checks still reject any compiler or toolkit
outside the CUDA 13.2 series.

- A public-files-only export in a separate `/tmp` directory excluded
  `.juicer-local/`, `ArchiveOldTestingFiles/`, `external/`, `out/`, `tools/`,
  and `validation/`. Linux Debug configured and built there, discovered all 16
  tests, and passed the nine `host` tests twice.
- Linux Debug and Release public test builds completed with warnings as errors.
  Each passed all nine `host` tests and all seven `gpu` tests on an NVIDIA
  GeForce RTX 4080. Debug host execution was also repeated without fixture
  mutation. A prior Linux Release build with `BUILD_TESTING=OFF` completed and
  contained neither GoogleTest sources nor test binaries.
- After transfer to native Arch Linux, the preserved native Debug and Release
  trees were reconfigured and rebuilt against the migrated source. Debug
  passed all 16 tests in 28.17 seconds and Release passed all 16 in 9.62
  seconds. A separate sandboxed Debug host run passed all nine `host` tests;
  the complete runs used direct device access and passed all seven `gpu`
  tests. The Linux Release `bundle-archive` target then cleanly restaged and
  packaged the complete bundle. Foreign Windows/WSL output was retained only
  as ignored evidence and was not consumed by these verdicts.
- Native Windows Debug and Release public test builds completed with the
  required VS 18 ClangCL/NVCC toolchain. Each passed all eight Windows `host`
  tests and all seven `gpu` tests. GoogleTest's test-only C language is pinned
  to `clang-cl.exe` in the shared Windows preset so CMake cannot combine
  GNU-mode `clang.exe` with MSVC-style flags.
- The Python comparator tests proved that a missing input and a deliberate
  one-double-above-bound mismatch both return nonzero with named diagnostics.
  The integration runner now fails any selected case group that produces no
  rows.
- The initial GPU and PowerShell failures were caused by a managed Codex
  command sandbox that restricted WSL `/dev/dxg` and VM-socket host bridges.
  In the unrestricted validation context, in-task `nvidia-smi` reported the
  RTX 4080 with Windows KMD 610.88 and CUDA UMD 13.3, PowerShell interop passed,
  and every Linux and Windows GPU CTest process passed. No CUDA driver-version
  incompatibility remains.
- The migrated scatter manifest, scatter binary, and diffusion NPZ retained
  SHA-256 values `f59f3eda…`, `b68f4704…`, and `963c7de7…` after repeated runs.
- The pinned Windows LLVM 22.1.3 formatter passed for every public C++ test
  source. Targeted Windows clang-tidy completed successfully and reported 13
  user-code findings in the migrated integration/diffusion harnesses (primarily
  swappable dimension parameters, explicit widening, qualified pointer `auto`,
  and conservative exception-escape diagnostics); it emitted no fatal error.
- On 2026-09-19, the owner reported the focused Windows Release Resolve
  checklist passed: discovery, four scan routes, alpha, scatter/halation at two
  scales, cancel/recovery, instance removal, and clean project/application
  shutdown. The accompanying native Windows command run passed all 15 CTest
  cases (8 `host`, 7 `gpu`, including 3 `reference`) in 5.04 seconds and cleanly
  restaged the bundle with `bundle-archive`. A concise owner pass is the retained
  host acceptance record; no separate capture form is required. No Windows
  owner-run validation remains for this migration.
- On 2026-09-19, the owner reported the native Linux migration and validation
  complete and passing, including the installed Resolve-library preload used
  to qualify the Linux host boundary. No Linux owner-run validation remains
  for this migration.
- This consolidation changed no production rendering or resource ownership, so
  Debug repetition, a forced context reset, profiling, and benchmarks are not
  acceptance gates.
- Reference regeneration was not rerun. The relocation retained numerical
  payloads/bounds and records its path-sensitive manifest and lock changes;
  regenerating either external cohort remains a separate reviewed action, not
  outstanding migration validation.

## Package 5 bounded retirement — 14 October 2026

The current consumer audit retires six duplicate private policy/helper files, with
byte-preserved original paths in the existing local archive. Public `scripts/`
owns native source classification and format/hygiene commands. The private
whole-`src` selector remains because its scope is distinct. The public quality
dispatcher is the acceptance default; the private tidy runner, inventory, coverage
tool and their tests remain for profile comparison, exploratory checks and
diagnostic/evidence inventories. No public command depends on private tooling.

The 14 October change was a retirement decision, not promotion or acceptance of
the deferred families. It removed no CTest process, fixture, numerical bound,
FFI contract or CUDA lifetime case. Private-tool tests are selected explicitly;
broad historical DB/RM discovery is not an acceptance command.

The current case mapping below incorporates the 23 October Package 1 retirement,
which explicitly supersedes the earlier decision to retain DB04C/D/E and DB06b
Python compiler drivers, redundant RM06 retry parsing and the event-pool spelling
guard. Unmatched `db*.cpp` sources remain at their original paths as inert
interpretation companions; their removed drivers are no longer supported
reproduction commands. No numerical oracle is regenerated. Required historical
captures, summaries, binaries and original evidence remain unchanged.
The 24 October Package 6 section refines the remaining family decisions and
records additional obsolete machinery actually removed. Its explicit obligations
replace blanket family retention; they do not upgrade historical acceptance.

| Private origin and exact cases | Disposition |
| --- | --- |
| `test_db02e_spatial_scale_and_dir_parity.py`: `test_duplicate_scale_identity_is_absent_from_product_source`, `test_resolved_and_sample_contracts_each_store_one_raw_spatial_scale`, `test_raw_scale_flows_to_radius_and_local_floor_flows_to_components`, `test_single_scale_is_validated_and_hashed_once_per_contract`, `test_existing_film_format_to_dir_scale_path_remains_direct`, `test_scale_matrix_derives_expected_values_and_threshold_sides`, `test_closed_case_matrix_covers_formats_stages_and_artifact_patterns`, `test_validator_emits_six_scale_v2_report_at_existing_tolerance` | Retain privately unchanged: scale/radius source checks (first five cases), fixed format/operator reference matrix (next two), and report validation (last case). Revisit each at the next spatial-scale/DIR owner change and migration closeout. The 400 mm probe is historical research, beyond current 8–120 mm authored controls; it must not become admission coverage. |
| `test_db04c_recipe_derivation_shrink.py`: `test_resolved_recipe_identity_and_authoritative_scanner_validation` | Python driver deleted, 23 October. Preserve `db04c_recipe_derivation_contract.cpp` at its original private path for unmatched DIR-equivalence, print filter/fallback/normalizer and scanner-identity expectations. No complete public replacement or new execution is claimed; revisit at the relevant recipe/scanner owner change and migration closeout. |
| `test_db04d_common_film_foundation.py`: `test_shared_input_builds_direct_and_print_foundations_once` | Python driver deleted, 23 October. Preserve `db04d_common_film_foundation_contract.cpp` and its direct DB04C source include at their original paths for unmatched shared-field equality and frozen hashes. Reconcile at the next foundation/identity owner change and migration closeout. |
| `test_db04e_focused_render_state.py`: `test_real_publishers_use_one_authoritative_filming_payload`, `test_pending_state_coalesces_only_on_the_full_hash` | Python driver and host-specific compiler/header-rewrite setup deleted, 23 October. Preserve both original `db04e_*_contract.cpp` sources and the existing frozen payload/summary. Public admission covers only the named subset below; duplicate publisher, stale-state and print missing-state rows remain historical investigations, not replaced coverage. Revisit at pending admission/rebuild or focused-payload owner change and migration closeout. |
| `test_db06b_print_profile_cache.py`: `test_deleted_modules_and_project_entries_are_absent`, `test_validated_profile_owner_is_direct_and_finite`, `test_resource_library_keeps_only_the_small_neutral_snapshot`, `test_ui_snapshot_uses_recipe_owned_types_directly`, `test_authoritative_recipe_consumers_remain`, `test_selected_profile_cache_and_strict_loader`, `test_neutral_calibration_snapshot_and_release` | Python driver and five obsolete layout cases deleted, 23 October. Current cache/hold/release, catalog snapshot, profile and calibration counterparts are mapped below. Preserve both original `db06b_*_contract.cpp` sources, including the unmatched concurrent cold-load winner requirement; no equivalence to concurrent reads is claimed. Numeric wavelength equality is obsolete Rust source admission; invalid JSON remains invalid. Revisit the unmatched cache row at assets/profile owner change and migration closeout. |
| `test_rm06_allocation_retry_cleanup.py`: former `test_01_helper_has_exact_action_and_reap_bounds`, `test_02_initial_and_non_capacity_outcomes_return_directly`, `test_03_zero_progress_returns_without_retry`, `test_04_reap_failure_preserves_both_exact_diagnostics`, `test_05_actual_retry_has_one_outcome_and_preserves_first_error`, `test_06_retired_and_misleading_diagnostics_are_absent`, `test_07_only_the_three_owned_consumers_remain` | Reduced to `test_initial_success_returns_without_reclaim`, the unmatched initial-success half of former `test_02`. Other behavioral assertions map to accepted runtime cases below; trace spellings/counts and caller topology are retired separately, with no identified exact-shape consumer. Revisit the remaining source investigation at the next allocation/reclaim owner change and migration closeout. |
| `test_rm06_frame_use_event_recycling.py`: `test_completed_frame_use_events_share_the_existing_completion_pool` | Module deleted, 23 October: exact field/helper/pool wiring is an implementation guard with no identified event-object pool-size/allocation-budget requirement. Event-based asynchronous lifetime remains required; existing fence/finish/abort coverage is not claimed to prove event-handle reuse. Any future concrete pool-budget requirement belongs at the CUDA resource owner. |
| `test_rm06_pinned_upload_cleanup.py`: `test_01_deleted_policy_surface_is_absent`, `test_02_block_has_one_real_lifecycle_state`, `test_03_upload_uses_the_resources_owner_identity`, `test_04_completion_errors_have_safe_state_transitions`, `test_05_context_cleanup_has_explicit_disposition` | Retain five source/lifetime investigations unchanged: retired surface, block lifecycle, exact owner identity, completion-error transitions and context cleanup. No complete accepted runtime replacement. Revisit when pinned-upload staging/completion/context retirement ownership next changes, and at migration closeout; no blanket S4.E expiry or grep-to-behavior equivalence claim. |

| Retained investigation family | Case/consumer mapping and retirement decision |
| --- | --- |
| `validation/gamma/` | Comparator-only split/fused cases map to public `tests/gamma/`. Retain the rest: `default-paper`/`default-routes` frozen captures, `sampling-dir`, `print-backend`, `state`, `routes`, separate injected/candidate route probes, `lifetime` table update/replacement, queued old/new streams, gamma reuse and abort/preflash retention, and `performance` unchanged/edit distributions. Preserve staging, before/after inventory checks, no-overwrite promotions and reference generation. Only old Release identities exist; two configuration-invalid cases remain explicitly invalid. No current four-preset baseline or complete lifetime replacement is established. |
| `ExperimentalReleaseValidation*` | Retain half/asset, table, matrix, hull, Hanatos-window, balance, output-host, raw-CUDA, output-CUDA, profile and route investigations. Public host/FFI/reference cohorts overlap several boundaries; the large private spectral/output/route cohorts and their provenance are not proved redundant. |
| `ExperimentalDirValidation*` | Retain pointwise, receiver, final-development, folded-neutral, spatial-source-failure, reflected-operator and infinity-sentinel probes with their experimental fixtures. Separate public-fixture and numerical-authority review is still required before promotion or retirement. |
| `tools/diffusion_cuda_harness/` | Retain plan-layout/execution-descriptor checks, spectrum/convolution, camera route/carrier and ledger/resource lifecycle investigations, separately from five-process cuFFT profiling/selection and its accepted executable/provenance. The generated 1.4 GB convolution bundle stays maintenance-only. Public diffusion host/reference and scatter/FFI owner tests are not complete replacements for these inputs. |
| `tools/dir_filter_harness/` | Retain strict FIR/YVV/tail reference probes and synthetic DIR source/filter/development separately from scanner/print/grain diagnostics and timings. Preserve reference generation/image qualification and rejected/approximate variants (size-aware, segmented/transpose/fused/streamed/parallel YVV, tiled FIR, downsampled tails and rank-2 DIR). Raw-log remains hard-blocked; its two obsolete success tests/calls and runnable examples were removed on 24 October 2026. Research alternatives are not product parity authority. |

The archive and accepted A7–A9 source, captures, rejection records and receipts
remain at their recorded locations. Standard build/Cargo trees, staged bundles,
path-bound historical harness binaries and current qualification runtimes retain
configuration, review or reference consumers and are not deletion candidates.
Archiving the six helpers retains 15,113 bytes and reclaims zero bytes.

A10 output-color translation retains the accepted A9 raw source/table fixtures,
D1–D5 diagnostic and W1–W7 cache witnesses. Output-space tags now have one core
color owner. The native literal/hash registry, lazy hash mutation, getters, DWG
multiplication and unused float encoder are deleted after caller migration.
Definition guards include explicit rejected controls. The continuing scanner
oracle owns frozen native inputs and f64 evaluation in the scanner test domain;
production uses prepared CCTF/DWG values in both C++ and cold C paths. The new
encoding schema is a representation requirement, not an identity contributor or
scientific admission gate. Native value bridges retire at S4.E; existing Library
forwarding retains S5.C. Family/stage acceptance remains a separate review gate.

## 23 October 2026 — obsolete private machinery

Package 1 uses accepted source `93c39e68313d5f31836e55a1967758a88a97d4e8`.
Slice 7 is now accepted; none of its code, fixtures, receipts or obligations are
changed by this cleanup. This disposition updates the earlier private-family
retention decisions above, not the scope of any historical acceptance result.
The counterparts below are assertions inspected in accepted source, not a claim
that these groups were freshly executed for private runner removal.

### Assertion and interpretation map

| Removed or reduced private case | Accepted counterpart or exact retirement reason | Retained meaning / remaining obligation |
| --- | --- | --- |
| DB06b `test_deleted_modules_and_project_entries_are_absent` | Reads absent `juicer.vcxproj`/`.filters` and requires a fixed count of 55 native source files. Those layout contracts are obsolete; no new equivalent is appropriate. | None; no product behavior or numerical oracle in this case. |
| DB06b `test_validated_profile_owner_is_direct_and_finite`, `test_resource_library_keeps_only_the_small_neutral_snapshot` | Assertions name retired native profile/cache/calibration structs, maps, loader entry points and snapshot representation; Rust owns their current behavior. | Retire spelling/layout enforcement, not current ownership or failure contracts. Compiled behavioral rows are mapped separately below. |
| DB06b `test_ui_snapshot_uses_recipe_owned_types_directly`, `test_authoritative_recipe_consumers_remain` | Exact native field/type/signature/assignment spellings and old `Validated*Profile` pointers do not define the migrated controls, recipe or selected-profile interfaces. | No independent numerical oracle. Current product/ABI/control tests remain unchanged; no text guard is renamed around Rust. |
| DB06b `test_selected_profile_cache_and_strict_loader`: bundled traversal, serial cache hit, held handle, release/reload and token stability | [profile.rs](../rust/film-juicer-core/tests/profile.rs) `loads_bundled_profiles`; [assets.rs](../rust/film-juicer-core/tests/assets.rs) `successful_hits_share_payloads_without_reparsing`, `handles_survive_release_and_owner_drop_with_fresh_reacquisition`, `bundled_handles_preserve_c4_tokens_and_c5_samples_after_release_and_drop`. | Preserve original `db06b_profile_cache_contract.cpp` at `.juicer-local/scripts/tests/`. Its all-profile nonzero-token traversal is not declared equivalent to the default-profile held-handle fixture. Resolve unmatched membership at assets/profile cache owner change or migration closeout. |
| Same DB06b case: concurrent cold canonical winner | No exact accepted counterpart established. `ProfileOwner.ConcurrentReadsUseDisjointOutputsAndExcludeRelease` in [profile_owner_test.cpp](ffi/profile_owner_test.cpp) views one already-acquired owner; it does not test racing cold loads. | Original C++ lines 79–97 remain inert, unchanged at the original path. Resolve the requirement at assets/profile cache owner change or migration closeout. |
| Same DB06b case: empty object, wrong role, wrong numeric wavelength label and literal `NaN` document | `profile.rs` `requires_consumed_sample_fields`, `enforces_profile_role`, `reports_source_errors`, `requires_complete_json_documents`; `preserves_authored_wavelength_positions` explicitly preserves numeric authored labels. | Empty/role/JSON structure remains enforced. The old `381.0` equality rejection is obsolete Rust source admission; `NaN` is still invalid JSON syntax. No scientific gate is restored and no new malformed corpus is generated. Original input construction is described with the retained private source. |
| DB06b `test_neutral_calibration_snapshot_and_release`: found/cache/release/missing-entry/malformed-root/selected-triplet | `assets.rs` `calibration_owner_preserves_selected_lookup_and_cmy_cc_order`, `calibration_load_errors_remain_snapshots_until_release`; [calibration.rs](../rust/film-juicer-core/tests/calibration.rs) `missing_file_and_entry`, `selected_branch_errors`, `malformed_root`; [illuminant_calibration_test.cpp](ffi/illuminant_calibration_test.cpp) `CalibrationAbi.SelectedFieldsExactUtf8KeysAndClearedInvalidInputs`, `CalibrationNative.LookupCompletesAcrossCacheReleaseAndStockKeysOwnRecipeSelection`. | Keep `db06b_neutral_calibration_contract.cpp` as the small original input/status interpretation companion while DB06b's unmatched row remains open. Current typed semantics replace old native statuses; no loader/compiler restoration. |
| Catalog snapshot requirements reviewed alongside DB06b cache retirement | `assets.rs` `catalog_success_is_fixed_across_release_and_directory_changes`, `catalog_error_is_shared_and_survives_repair_and_release`. | Success and error snapshots are distinct from held-profile payloads; both current contracts remain covered. |
| RM06 retry `test_01`, `test_02` non-capacity half, `test_03`, `test_04`, `test_05`: behavior | [cuda_failure_test.cpp](ffi/cuda_failure_test.cpp) `AllocationRetry.CapacityRetriesOnceRegardlessOfDiagnostic`, `IdenticalTextCannotEnableRetryForOtherCategories`, `NoReclaimedBytesStopsAfterFirstAttempt`, `ReclaimFailurePreservesOriginalDiagnosticAndTerminalCode`, `FailedRetryPreservesOriginalDiagnosticAndNeverRetriesAgain`. | Accepted cases check attempt/reclaim counts, category gating, no-progress exit, terminal status, first/reclaim/retry diagnostics and success cleanup. Source variable types/order and retired string-only error representation are not independent contracts. |
| RM06 retry `test_02` initial-success half | **Not replaced:** the accepted runtime test allocator always fails attempt one. | Keep only `test_initial_success_returns_without_reclaim` in the original Python module. This is a source investigation, not runtime proof; resolve at the next allocation/reclaim helper owner change or migration closeout. |
| RM06 retry `test_03`–`test_06` trace/count/spelling subassertions and `test_07` caller topology | No current consumer was found requiring the exact `trace_reap_pass` / `trace_allocation_retry` call counts, retired strings or fixed three-caller topology. | Retire those source-shape guards. Existing diagnostic production is unchanged; typed failure and diagnostic preservation are covered by the runtime cases above. |
| RM06 frame-use-event pool case | Exact pool fields/helper names/source wiring are implementation details; no event-object allocation budget or shared-pool shape requirement was identified. | Retire the guard. Required event-based completion remains; fence/finish/abort tests do not establish event-object reuse. No new behavior test is added for a hypothetical budget. |
| DB04C compiled case | Standalone `g++` source-list execution bypasses current CMake/Rust linking. Complete equivalence to current tests is unproved. | Preserve original `db04c_recipe_derivation_contract.cpp`: DIR equivalence, print filter/fallback/normalizer and scanner identity-mismatch rows. Reconcile at the relevant recipe/DIR/print/scanner owner change and migration closeout. |

## 24 October 2026 — remaining historical/research families

Package 6 uses accepted `93c39e68313d5f31836e55a1967758a88a97d4e8`;
uncommitted Packages 1–5 are preserved and do not establish replacement coverage.
No production owner, public test, fixture value, tolerance or migration gate
changes here. Decisions below distinguish required interpretation/reproduction
from an executable maintained acceptance command. Temporarily retained rows are
specific unresolved obligations, not passing coverage or permission to revive
the removed compilers. Reconcile them at their named owner change and again at
migration closeout; S4.E is not a blanket expiry.

### Actual retirements

| Removed maintenance surface | Evidence and smallest disposition |
| --- | --- |
| `tools/dir_filter_harness/tests/harness_cli_tests.py`: `test_raw_log_pipeline_selected_channels_verify_against_all_channel_helper`, `test_raw_log_cat02_precompute_variants_verify_against_production_helper`, and both main calls | The first expected successful B/G selected-channel output within `1e-5`; the second expected bit-exact integrated-Hanatos and precomputed-CAT02 variants. `dir_filter_harness.cu::run_cuda_raw_log` already unconditionally rejects this mode as unqualified because those probes no longer represent production reconstruction. Remove these contradictory success cases and runnable README examples. Preserve that rejection function and every other CLI case, strict/reference input and historical result. No current raw-log equivalence is claimed and no replacement test/build is needed to retire obsolete expectations. |
| `.juicer-local/scripts/perf/codebase_phase_checks.sh` | Remove the obsolete 3,271-line P0–P8 report/gate generator. It requires deleted `docs/perf` ledgers/manifests and reads removed `JuicerCudaResourcesServing.cpp`. The DB-07A tooling audit independently calls it dead infrastructure. Current Performance V2 and accepted P0-A/P0-B1 consumer records do not call it; the only live-tree reference was an uncalled Windows bridge. This is obsolete policy/report machinery, not a scientific oracle. No reports or captures are deleted. |
| `.juicer-local/scripts/perf/PerfWindowsCommon.ps1`: `Invoke-PhaseChecksFromWindows`, `ConvertTo-BashLiteral` | Remove the dormant bridge and its sole-use quoting helper with the framework. `run_regression_investigation_windows.ps1` imports the common file for `Invoke-Python` and `Resolve-RepoRoot`; both and the other common utilities remain byte-identical. No remaining caller of either removed function was found. |
| `.juicer-local/scripts/perf/rerun_p2_benchmarks_clean.ps1` | Remove the unreferenced old-P2 maintenance command. It only manages the retired `docs/perf/evidence/codebase-cleanup/p2/automation` location and prints capture commands. Its copy/delete sequence suppresses errors, so it cannot establish successful preservation before deletion. It is neither a benchmark nor a required reproduction. Do not execute it; original evidence and the separate runtime-metrics capture tools remain in place. |

### Required history, active tools and bounded obligations

| Family / exact retained unit | Decision, consumer and remaining owner obligation |
| --- | --- |
| DB04C/D: original recipe/foundation C++ sources | **Historical only** for pre-cutover literal hashes and original input interpretation; **temporarily retained** for unmatched equal resolved DIR identity, print calibrated/fallback filter/normalizer identity, scanner mismatch rejection and direct/print shared film-field equality. DB04D still includes DB04C, so keep the pair together. The actual current requirement is equality for equivalent resolved inputs, not preservation of obsolete hash literals. Reconcile at the respective DIR/print/scanner recipe or common-film foundation/identity change. Public constituent fixtures do not prove the full cross-route equality. The Package 1 compiler retirements stand. |
| DB04E: focused-state source, pending source, frozen payload and summary | **Historical only** for the 254-line pre-change payload and approved filming-field substitution; retain its original hash/order and interpretation. **Temporarily retained** for duplicate publishers, stale success/failure without hash/counter advance, latest pending publication, and missing direct/print state despite an accepted hash. These are pending admission/rebuild/focused-state owner questions; existing admission/recovery tests establish only the previously mapped subset. No old copied-header/compiler setup is restored. |
| DB06b: profile-cache and neutral-calibration C++ companions | **Historical only** for old status/input interpretation, including invalid JSON versus obsolete numeric wavelength equality. **Temporarily retained** for all-profile nonzero-token membership and the concurrent cold canonical winner. Concurrent reads of an acquired FFI owner are not racing cold loads. Resolve at assets/profile-cache ownership or release/publication changes; keep the small calibration companion with the profile source while this obligation remains. Accepted Rust profile/calibration cases remain the current authority. |
| DB02e: all eight original scale/DIR investigations | **Temporarily retained:** five source ownership/flow/validation/hash/film-format checks, fixed scale and closed operator/pattern matrix, and six-scale validator. The 35/40.30/40.34/65/120/400 mm matrix distinguishes threshold sides; public ordinary fixtures do not establish identical membership. The 400 mm row is **historical research only**, outside authored 8–120 mm admission. Keep it with `spektrafilm_parity.py` and `spektrafilm_validate_phase3d3_dir_numeric.py`; resolve each row at the spatial-scale/DIR descriptor, geometry or FIR/YVV dispatch owner change. Do not promote its historical range or source spelling into product policy. |
| RM06 allocation retry: `test_initial_success_returns_without_reclaim` | **Temporarily retained:** the accepted `AllocationScenario::allocate` always fails attempt one, so its runtime retry tests cannot prove initial success without reclaim. Keep this one limited source investigation. Resolve at the allocation/reclaim helper owner change; the previously retired trace/count/topology guards stay retired. No new runtime test or product change is introduced for cleanup. |
| RM06 pinned uploads: five source/lifetime cases | **Temporarily retained:** exact owner-context binding, reservation/in-flight/available/quarantine completion, event-record/stream-sync failures, ordinary retirement and proven-context-loss disposal remain distinct. Public CUDA render expiry/preflash/epoch cases overlap, but do not replace the full completion-error/context-loss set or establish an event-pool budget. Resolve with pinned-upload staging/completion and context-retirement owners when those paths next change. Preserve these limited investigations and required reproductions; generic frame finish/abort is insufficient replacement evidence. |
| Gamma comparator, captures, runner and generator | Ordinary split/fused comparison is **consolidated** into accepted `tests/gamma/compare.py` / `test_compare.py` (equal/next-double bound, zero bound, malformed/applicability/legacy cases and CLI failure). Keep the historical runner's copy and inventory/no-overwrite/reader/generator checks only with its original protocol; they are not a second ordinary correctness gate. Frozen default-paper/routes and requested/injected gamma records are **historical only**, bound to their original resources and Release/Release-Clang identities. `reference_issues.json` continues to exclude the two invalid G09 cases; pre-diffusion files retain their historical meaning. No new baseline is generated or inferred from a current similarly named case. |
| Gamma state/lifetime and performance | **Temporarily retained** reproduction source/inputs for same-size table updates versus replacements, queued old/new streams, `1 → 1.25 → 1` reuse, abort/recovery and stock-preflash retention. Resolve individually at gamma/table publication, prepared-frame retirement or print-preflash ownership changes; current resource expiry/abort checks do not establish the complete sequence. A new current baseline would require valid inputs and explicit per-preset provenance. The 3-warmup/11-sample performance records are **historical only**, not an open rerun gate; current performance work follows Performance V2. |
| ExperimentalRelease: half/asset and Hanatos-window samples | **Historical only** for the original Arctic f16 file-byte/source-width and finite/range characterization (192×192×81 values). Accepted Rust `data_io.rs::bundled_npy_bits` already checks full decoded count/fingerprint and maximum; representation width or scientific plausibility must not become new resource-admission policy. **Temporarily retained** are the original six half-subnormal/boundary words and Portra 400/D55/sigma-1.25 on/off final-sensitivity oracle. `supported_float_widths_preserve_authored_samples` covers the positive minimum subnormal, not all six original words; `exposure_preparation_test.cpp` checks reference-white blur, not the same full recipe effect. Preserve the original `.cpp/.cu/.h`, cohorts, manifests and upstream-generating source. Resolve the unmatched words/window effect at source-decoder or film-reference/window producer changes, without reviving the old compiler or numerical admission gates. |
| ExperimentalRelease: other table/matrix/hull/balance/output/profile/route probes | **Temporarily retained** their original distinct inputs/oracles and provenance, outside ordinary acceptance. Public color/FFI/recipe cohorts cover adjacent boundaries; a name or shared stage is insufficient to retire the cross-domain rows. Revisit only the affected probe when its table, gamut, balance, output or profile/route owner next changes. Accepted capture/provenance remains **historical only** after that requirement is resolved; no wholesale migration of the large cohort is authorized. |
| ExperimentalDIR: pointwise/receiver/development/neutral/source-failure/reflection/infinity probes | **Temporarily retained:** sampled pointwise data use the 65×49 Portra160 donor/Provia100F receiver arrays with channel reversal; infinity probes enter CUDA final development at gamma 1 and 0.5 and require endpoint values plus zero failure mask. Accepted film-recipe transport and ordered-axis/infinity lookup tests do not execute those same mathematical/stage oracles. Keep original arrays, sources and generator. Resolve at DIR donor/receiver, spatial operator, final-development or source-failure owner changes; rejected experimental behavior does not become parity authority. |
| Diffusion CUDA descriptor/convolution/carrier/lifecycle | **Temporarily retained** plan/execution shape, spectrum dedup/indexing, convolution, camera carrier and physical-device-ledger/slot/fault cases. Public descriptor-driven convolution, exact-context/epoch and prepared-frame tests overlap but do not reproduce the complete private arithmetic/layout and fault membership. Resolve at diffusion descriptor/spectrum/carrier changes or resource/context-retirement ownership changes. Keep deterministic correctness source distinct from profile selection; the generated 1.4 GB convolution bundle remains maintenance-only and unchanged, not an ordinary fixture to import. |
| Diffusion cuFFT profile selection and closed provenance | **Historical only** for the accepted five-process raw evidence, captured executable and manifest-bound tools; **temporarily retained** evaluators/promotion guards for the named layout/memory-selection decision and 20-workload inputs. `profile_contract.py`, its tests, collector, build and provenance writer remain together at their recorded paths because the local-tool manifest binds them. Revisit at a layout-selection owner decision/profile rebaseline, not every recipe edit. Do not rebuild or relocate the accepted executable to retire an unrelated test. |
| DIR strict filters versus approximation/image/timing research | **Temporarily retained** strict FIR/YVV/tail and synthetic source/filter/final-development reference units with their generators. **Historical only** for rejected/approximate size-aware, segmented/transpose/fused/streamed/parallel YVV, tiled FIR, downsampled tails and rank-2 diagnostic results; source remains interpretation/reproduction, not product authority. Keep image qualification and timing separate from correctness. Resolve only the affected exact operator/scratch or approximation decision at that owner change. The raw-log success cohort above is retired; its explicit rejection remains unchanged. |
| Private fixture/reference generators and experimental probes | **Maintained for explicit reproduction only:** `generate_experimental_dir_fixtures.py`, `generate_experimental_release_tc_fixtures.py`, `probe_experimental_release_models.py`, `audit_experimental_colour_profiles.py` serve the retained cohorts and pinned source/model interpretation. `spektrafilm_generate_grain_reference.py` / `spektrafilm_generate_scatter_halation_reference.py` and the DB02e validator retain their named domain/reference consumers. Their commands require the recorded pinned environments and review; no ordinary test executes them or rewrites expectations. Retire each producer with its cohort/owner decision, preserving the minimal original source/provenance needed to interpret retained numbers. |
| Private performance tools and one-off gamut probe | Runtime-metrics capture, `build_runtime_metrics_summary.py`, its normalization self-test and regression capture/common utilities are **maintained private investigation tools**, distinct from the deleted phase framework. Preserve accepted RM/Performance V2 evidence; no new performance acceptance follows. `probe_output_gamut_boundary.py` is **historical only** for the S4.A Slice 10 first-exit/re-entry design question: its handoff explicitly disclaims native-bit/candidate-oracle authority. Keep that small semantic source with its recorded decision; no ongoing test or approval gate is assigned to it. |
| Private-tool tests and archive | **Maintained:** the four explicit tidy inventory/coverage/runner/source-policy tests still have the private tidy runner/inventory consumers; `tidy_touched_files.ps1` selects them explicitly. **Historical only:** `ArchiveOldTestingFiles/` retains existing original paths/notes for previously superseded machinery. No new archive copy, rediscovery, public dependency or second maintained suite is created. Existing migration source/capture/diagnostic obligations remain at their recorded paths. |

This completes the bounded family decisions, with actual removals above and
specific obligations remaining explicitly open. None of the unmatched numerical
or lifetime rows is claimed replaced or freshly executed. Source parsing,
unchanged-case/consumer checks and preservation checks validate this retirement;
no historical harness, generator, build, GPU/OFF matrix or qualification was run.
The cleanup execution record retains exact removal identities and validation.
| DB04D compiled case | Same obsolete compiler path; whole-contract replacement is unproved. | Preserve `db04d_common_film_foundation_contract.cpp` together with its direct DB04C include: shared direct/print fields and frozen hashes. Reconcile at the next foundation/identity owner change and migration closeout. |
| DB04E focused-payload compiled case | Host-specific raw compilation and header rewriting are obsolete setup, not independent authority. | Preserve `db04e_focused_render_state_contract.cpp` and the existing `pre-change-payloads.txt` / `summary.md` under `.juicer-local/evidence/DB-04E-focused-render-state-ownership/`. The original print values and direct-only approved exposure/S-inverse/film-raw substitution remain interpretable. Reconcile at focused-payload owner change and migration closeout. |
| DB04E pending-coalescing compiled case | [integration_test.cpp](scatter_halation/integration_test.cpp) `pending/same-hash-ready-reuse`, `pending/same-hash-not-ready-rebuild`, `pending/edit-during-rebuild-retries-current` are accepted partial counterparts. | Preserve original `db04e_pending_coalescing_contract.cpp`: distinct duplicate publisher, stale success/failure with exact hash/counter preservation, and missing-print-state rows are not declared replaced. Reconcile at pending admission/rebuild/coalescing owner change and migration closeout. |

The six C++ companions remain byte-identical at their original private paths.
Their directory README records integrity and interpretation; it is not a public
test dependency. DB04D still includes the original DB04C source. DB04E's existing
frozen capture and summary retain the oracle order and approved substitution;
the removed wrapper adds no unique numerical expectation. No mandatory
path-bound reproducer for the removed Python paths was identified in the scoped
selector/import/consumer review. Historical review citations are evidence of the
old paths, not supported execution commands, and are not rewritten.

DB02e and pinned-upload investigations remain in place with the limited purposes
and owner-change triggers above. The four private analysis-tool test modules
retain their actual consumers. No maintained public test, fixture, CTest
registration, ABI/error/lifetime case, platform/profile coverage or original
evidence is deleted or weakened. No build or historical compiler run is needed to retire the
obsolete wrappers. Only the narrowed remaining Python source investigation needs
a focused execution check; documentation/source/reference and retained-byte
checks cover the rest of this package.
