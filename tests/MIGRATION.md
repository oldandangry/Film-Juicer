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
| `.juicer-local/scripts/tests/` | Retained privately pending domain review. It mixes private-tool tests, historical source-layout assertions and bespoke-compiler product contracts. Structural assertions were not republished as behavior tests merely to increase counts. |
| `.juicer-local/scripts/spektrafilm_*` and `.juicer-local/validation/` | Retained privately as reference generation, historical evidence and investigations. Ordinary public tests have no dependency on them. |
| `.juicer-local/scripts/perf/` | Retained privately as investigation tooling. Only the current scatter/halation measurement mode received a public benchmark procedure. |

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

This is a retirement decision, not promotion or acceptance of the deferred
families. No CTest process, fixture, numerical bound, FFI contract or CUDA lifetime
case is removed. The current private-tool tests are selected explicitly; a broad
discovery of historical DB/RM scripts is not an acceptance command. Historical
compilers require their original source/tool snapshot to reproduce old evidence.

The case mapping below deliberately retains material for which complete
replacement has not been proved. Companion `db*.cpp` sources stay with their
Python drivers. No numerical oracle is regenerated.

| Private origin and exact cases | Disposition |
| --- | --- |
| `test_db02e_spatial_scale_and_dir_parity.py`: `test_duplicate_scale_identity_is_absent_from_product_source`, `test_resolved_and_sample_contracts_each_store_one_raw_spatial_scale`, `test_raw_scale_flows_to_radius_and_local_floor_flows_to_components`, `test_single_scale_is_validated_and_hashed_once_per_contract`, `test_existing_film_format_to_dir_scale_path_remains_direct`, `test_scale_matrix_derives_expected_values_and_threshold_sides`, `test_closed_case_matrix_covers_formats_stages_and_artifact_patterns`, `test_validator_emits_six_scale_v2_report_at_existing_tolerance` | Retain privately: scale/radius source checks (first five cases), fixed format/operator reference matrix (next two), and report validation (last case). The 400 mm probe is historical research, beyond current 8–120 mm authored controls; it must not become admission coverage. |
| `test_db04c_recipe_derivation_shrink.py`: `test_resolved_recipe_identity_and_authoritative_scanner_validation` | Retain the numerical/scanner contract and its native compiler as historical reproduction source. Current recipe/math ownership has moved; no assertion-level replacement or new native acceptance is claimed. |
| `test_db04d_common_film_foundation.py`: `test_shared_input_builds_direct_and_print_foundations_once` | Retain the shared-foundation/hash oracle as historical numerical evidence; its bespoke native compile is not a current Rust-linked product build. |
| `test_db04e_focused_render_state.py`: `test_real_publishers_use_one_authoritative_filming_payload`, `test_pending_state_coalesces_only_on_the_full_hash` | Retain focused-publication numerical oracles and pending-state/coalescing assertions. Public scatter admission tests cover current coalescing/recovery, but that alone does not establish equivalence of every frozen hash oracle. |
| `test_db06b_print_profile_cache.py`: `test_deleted_modules_and_project_entries_are_absent`, `test_validated_profile_owner_is_direct_and_finite`, `test_resource_library_keeps_only_the_small_neutral_snapshot`, `test_ui_snapshot_uses_recipe_owned_types_directly`, `test_authoritative_recipe_consumers_remain`, `test_selected_profile_cache_and_strict_loader`, `test_neutral_calibration_snapshot_and_release` | Retain original source: the five layout cases describe the removed native profile/calibration owners, and the two compiled cases preserve cache/hold/release and old admission oracles. Current Rust asset/profile and FFI tests are authoritative; old authored-content gates must not be restored. |
| `test_rm06_allocation_retry_cleanup.py`: `test_01_helper_has_exact_action_and_reap_bounds`, `test_02_initial_and_non_capacity_outcomes_return_directly`, `test_03_zero_progress_returns_without_retry`, `test_04_reap_failure_preserves_both_exact_diagnostics`, `test_05_actual_retry_has_one_outcome_and_preserves_first_error`, `test_06_retired_and_misleading_diagnostics_are_absent`, `test_07_only_the_three_owned_consumers_remain` | Retain seven private source/flow assertions: retry/reap bounds, early outcomes, zero progress, original/reap diagnostics, first-error preservation, trace absence/count and consumer count. Public CUDA owner tests do not establish every source/diagnostic assertion as redundant. |
| `test_rm06_frame_use_event_recycling.py`: `test_completed_frame_use_events_share_the_existing_completion_pool` | Retain the source-level pool/reap/teardown/submission assertion. Existing public asynchronous/frame-owner tests do not by themselves prove identical pool implementation. |
| `test_rm06_pinned_upload_cleanup.py`: `test_01_deleted_policy_surface_is_absent`, `test_02_block_has_one_real_lifecycle_state`, `test_03_upload_uses_the_resources_owner_identity`, `test_04_completion_errors_have_safe_state_transitions`, `test_05_context_cleanup_has_explicit_disposition` | Retain five source/lifetime investigations: retired surface, block lifecycle, exact owner identity, completion-error transitions and context cleanup. Do not replace these with a grep-to-behavior equivalence claim. |

| Retained investigation family | Case/consumer mapping and retirement decision |
| --- | --- |
| `validation/gamma/` | Comparator-only split/fused cases map to public `tests/gamma/`. Retain the rest: `default-paper`/`default-routes` frozen captures, `sampling-dir`, `print-backend`, `state`, `routes`, separate injected/candidate route probes, `lifetime` table update/replacement, queued old/new streams, gamma reuse and abort/preflash retention, and `performance` unchanged/edit distributions. Preserve staging, before/after inventory checks, no-overwrite promotions and reference generation. Only old Release identities exist; two configuration-invalid cases remain explicitly invalid. No current four-preset baseline or complete lifetime replacement is established. |
| `ExperimentalReleaseValidation*` | Retain half/asset, table, matrix, hull, Hanatos-window, balance, output-host, raw-CUDA, output-CUDA, profile and route investigations. Public host/FFI/reference cohorts overlap several boundaries; the large private spectral/output/route cohorts and their provenance are not proved redundant. |
| `ExperimentalDirValidation*` | Retain pointwise, receiver, final-development, folded-neutral, spatial-source-failure, reflected-operator and infinity-sentinel probes with their experimental fixtures. Separate public-fixture and numerical-authority review is still required before promotion or retirement. |
| `tools/diffusion_cuda_harness/` | Retain plan-layout/execution-descriptor checks, spectrum/convolution, camera route/carrier and ledger/resource lifecycle investigations, separately from five-process cuFFT profiling/selection and its accepted executable/provenance. The generated 1.4 GB convolution bundle stays maintenance-only. Public diffusion host/reference and scatter/FFI owner tests are not complete replacements for these inputs. |
| `tools/dir_filter_harness/` | Retain strict FIR/YVV/tail reference probes and synthetic DIR source/filter/development separately from scanner/print/grain diagnostics and timings. Preserve reference generation/image qualification and rejected/approximate variants (size-aware, segmented/transpose/fused/streamed/parallel YVV, tiled FIR, downsampled tails and rank-2 DIR). Raw-log mode is already hard-blocked as obsolete. Research alternatives are not product parity authority. |

The archive and accepted A7–A9 source, captures, rejection records and receipts
remain at their recorded locations. Standard build/Cargo trees, staged bundles,
path-bound historical harness binaries and current qualification runtimes retain
configuration, review or reference consumers and are not deletion candidates.
Archiving the six helpers retains 15,113 bytes and reclaims zero bytes.
