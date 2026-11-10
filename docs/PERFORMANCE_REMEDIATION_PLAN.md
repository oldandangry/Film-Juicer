# Film-Juicer local build and verification remediation plan

Investigation date: **17 October 2026, Europe/Stockholm**. Planning only; no product, build, test, fixture, workflow or source-policy implementation was changed. Numbers below are seconds unless stated otherwise. Proposed after-times are acceptance targets or estimates, **not measured improvements**.

## 1. Executive diagnosis

The dominant observed local problem is repeatedly paying for broad verification while the candidate is still changing, amplified by slow WSL configuration and filesystem work. Slice 3 ran ten native quality passes: eight failed, four only because inputs changed during checking; those four consumed **39.1 minutes of overlapping command time**, covering approximately **20.4 minutes of elapsed time**. WSL CMake configuration alone consumed **952.2 seconds across ten regenerations** in that task, before generation, compilation or linking; native Windows configuration was much cheaper. Debug numerical tests remain expensive locally, and the checkpoint dispatcher additionally runs CTest serially in the inspected Windows qualification records. Warm native builds themselves are healthy: fresh Windows Release named-target probes took **0.50/0.51 seconds**, and Cargo test preparation took **0.74 seconds** without compilation. Broad output invalidation and Cargo cleaning are real in restoration receipts, but environment drift alone, Defender, ext4 relocation benefits, and optimal concurrency remain unisolated causes or unmeasured remedies.

**Recommended order:** establish one bounded edit-feedback path and a stable-candidate boundary; separate artifact freshness from verification/provenance identity; narrow and reuse native analysis; then address WSL configuration/probe costs and checkpoint scheduling. Preserve complete final qualification.

## 2. Environment, authority and measurement scope

### Exact source identities

| Evidence | Source identity and condition |
| --- | --- |
| Current checkout / fresh probes | Branch `rust-migration`, HEAD `e8b1dd710c1900bcc891e1660af9dd587d7ace96`; same HEAD as the investigation handoff |
| Existing uncommitted candidate | `scripts/build_dependencies.py`, `scripts/build_evidence.py`, `scripts/execution.py`, `scripts/process_tree.py`, `tests/quality/test_build_evidence.py`, `tests/quality/test_execution.py`; Windows restoration and generated-header fixes, present before this review |
| Slice 2 historical task | Started at `00d75a4d8c4d727e1c4e7961f557994c581cc3b0`; evolving 40-file candidate, subsequently published as `aef712df8abf620eba0247741043a00e39d3bb52` |
| Slice 3 historical task | Started at `aef712df8abf620eba0247741043a00e39d3bb52`; evolving 52-file candidate, subsequently published as `fe167391568f25725d815a6bc4a7443330701e2f` |

The evolving historical tasks are not benchmarks of an unchanged commit. Their command receipts bind actual input hashes. Current inspected script line numbers refer to HEAD plus the pre-existing restoration candidate. That candidate is not presumed reviewed or accepted by this report. The evidence manifest records individual dirty-file hashes and Git-diff identity in [environment.json](../out/validation/performance-review/2026-10-17/environment.json).

| Component | Native Windows | WSL |
| --- | --- | --- |
| System | Windows 11 IoT Enterprise LTSC, 10.0.26300 | Kernel `6.18.33.1-microsoft-standard-WSL2` |
| CPU / memory observed | Shared i5-14600K machine; about 31.7 GiB visible physical memory | 20 logical CPUs exposed; 15 GiB memory, 4 GiB swap; no swap used at inspection |
| Checkout | `C:\Dev\Codex` | `/mnt/c/Dev/Codex`, same files through `v9fs`/DrvFS |
| Storage | C: NTFS | Checkout, all four preset caches, and Linux Cargo target directories are on the Windows-mounted filesystem, not ext4 |
| Presets | `windows-clang-debug`, `windows-clang-release` | `linux-debug`, `linux-release` |
| Build directories | `out/build/<preset>`; all observed caches testing ON | Separate preset trees; not a Windows/Linux shared binary cache |
| CMake / Ninja | 4.3.3 / 1.13.2 | 4.3.1 / 1.13.2 |
| Host / CUDA | ClangCL 22.1.3; CUDA 13.2, NVCC host MSVC 14.51.36231 | GCC 13.3.0; CUDA/NVCC 13.2.86 |
| Rust | rustc/Cargo 1.98.1, `x86_64-pc-windows-msvc` | rustc/Cargo 1.98.1, `x86_64-unknown-linux-gnu`; configured rustup/Cargo installation under `.tmp/rust-linux/` |
| Python used | Shared venv, 3.13.7 after restoration | Shared venv, 3.13.13 |

Slice 2/3 Windows results used the then-approved A10 copied MSVC/embedded-Python environment. Current Windows observations use installed VS18/CUDA and the restored shared venv. These results must not be described as a matched before/after toolchain experiment. The current WSL shell did not expose Cargo on PATH; explicit existing `CARGO_HOME`, `RUSTUP_HOME` and configured launcher paths reported the pinned versions. No toolchain was installed or replaced here.

### Evidence and probe boundaries

The handoff was read in full before checks. This review read contributor/build/test guidance, naming/hygiene standards, the active local records, relevant implementations and representative callers, and two accessible Codex session transcripts. Raw reconstruction and small probe scripts are under [out/validation/performance-review/2026-10-17](../out/validation/performance-review/2026-10-17/README.md). This is ignored telemetry, not a new verification framework.

Fresh execution was restricted to dry-run inspection, two already-built Windows Release named-target builds, one warm Rust core preparation, version queries, and three repetitions of read-only metadata/content operations. The Windows builds used the existing toolchain wrapper, Campaign supervisor, a two-second build-lease admission budget and 90-second command deadlines. No compiler or linker ran. The named executable, Rust archive, Cargo stamp and Ninja file retained identical hashes and timestamps.

An independent Windows Debug qualification was active during inspection; its lease was left alone, and no test/GPU workload was launched here. Current WSL `tests/CMakeLists.txt` is newer than both generated Ninja files. A new WSL build could therefore regenerate the cache; historical warm no-op receipts were used instead. Ninja `-n` also pessimistically reports CMake regeneration behind the forced glob check, even on the Windows tree that subsequently proved warm; **dry-run output alone is not proof of an expensive rebuild**.

## 3. Reconstructed local timelines

All timeline timestamps below are **UTC**; add two hours for Stockholm on these dates. Receipts measure overlapping subprocess intervals. Their sums are work-demand indicators, not task wall time or CPU time. Uncovered intervals include authoring, inspection, repair and reporting; they cannot be labeled agent idle time.

### Task A: Slice 2, complete film-input migration

Measured interval: **16 October 16:30:12.737–18:28:36.280**, **118.39 minutes**; initial reading/preparation is outside that interval. The 187 primary-campaign attempts reproduce the existing task-cost receipt: **6,411.42 seconds summed**, **4,094.41 seconds interval union**, ten failed attempts totaling **782.08 seconds**. Ten additional nested checkpoint receipts describe work already covered by outer invocations; do not add them again.

| UTC interval / event | Commands and rerun cause | Recorded cost / implication |
| --- | --- | --- |
| 16:30 onward | Native capture preparation, compile/link/runtime on four presets | Necessary pre-translation authority; not removable duplicate qualification |
| 16:42–17:01 | Named CMake builds and CTest `Ffi.Host.FilmInput`, Rust.Core, Rust naming/privacy controls | One Linux build rejected concurrent edits; two first boundary runs rejected a safety-comment placement. Linux boundary retries took 214.8/194.2 s; Windows Release 62.8 s |
| 17:03–17:19 | Caller-adoption builds, nine owning groups | Linux builds 141.9, 173.0 and 136.5 s; owning Debug CTest about 136–143 s. These checked a different transition from the earlier raw ABI gate |
| 17:48 onward | Final owning targets, then 13-group CTest selections | Both native build failures retained; native test/Clippy repairs and caller deletion checks required reruns |
| 17:58–18:24 | `check-quality.py --base 00d75a4... --jobs 4` | Five runs: Windows 8.79/48.83 s failures, Linux 177.05 s failure, final Windows 454.62 s and Linux 504.27 s passes |
| 18:00–18:15 | Final host/boundary retries | Linux Release NativeBoundary hit its unchanged 30 s limit under concurrent scanning; reduced campaign test concurrency, then reran. Linux explicit privacy checkpoints ran again; Windows admitted some unchanged evidence |
| 17:58–18:27 | Release testing-OFF production gates, then ON restoration and owning pair | Initial C11 helper placement failed; later test-only tidy repair invalidated the gate's broad source identity, causing another production checkpoint. These reruns are broader than the changed product behavior |
| 18:28 | Final audit and review packet | Full aggregate/Resolve/hardware qualification was not established by this task |

Primary stage sums: build **1,507.41 s**; boundary-labeled attempts **2,027.88 s** (includes owning product CTest); quality-labeled attempts **2,164.47 s** (includes checkpoint wrappers, not just tidy); production **285.38 s**; capture/native preparation and other checks **426.28 s**. The five actual native quality runs total **1,193.57 s**, of which **234.68 s** failed. Exact commands, timestamps, statuses and immutable receipt paths for every attempt are in [slice-2-commands.md](../out/validation/performance-review/2026-10-17/slice-2-commands.md).

### Task B: Slice 3, DIR/density-bound migration

Inspection began approximately **19:40 UTC**; exact BEGIN snapshot **19:43:21**; final review packet **22:31:43.981**: approximately **171.73 minutes**. The 163 recorded attempts cover **5,263.33 seconds** of elapsed time and sum to **10,295.05 seconds** with overlap/nesting. Twenty-five attempts failed. Required native capture, construction-before-adoption and final stable qualification are distinct responsibilities.

| UTC interval / event | Commands and trigger | Cost |
| --- | --- | --- |
| 19:43–early capture | Installed Windows toolchain probe failed descendant drainage; bounded A10 continuation then authorized for that task | NVCC parent had succeeded in about 2 s; failed attempt consumed 30.14 s. Separate from normal compile performance |
| 20:19–20:46 | Named `JuicerFilmRecipeTests` / `JuicerFilmDevelopmentTests` builds, four configurations, generally `--parallel 4` | C11 path-layout and native test repairs caused failures; successful WSL builds 143.3/166.7/171.7 s |
| 20:47 | Windows Debug owning test run | Assertions passed, but the 44.8 s receipt rejected a concurrent input edit |
| 20:53–21:15 | Broader named-target builds | WSL 335.4/283.1/152.7 s runs rejected changed inputs; Windows corresponding completed builds 63.0/64.8 s |
| 20:56–21:08 | Early quality attempts | Windows 277.86/262.23 s and Linux 276.74 s failed on concrete enum/move/use-after-move diagnostics |
| 21:10–21:21 | First complete broad quality pair | Linux 611.03 s, Windows 628.65 s; both failed final freshness admission after source changed |
| 21:24–21:35 | Second complete broad quality pair | Linux 576.96 s, Windows 531.63 s; both again failed final freshness admission |
| 21:52–22:03 | Stable owning rebuilds | Linux Debug 293.7 s, including 91.5 s configure + 3.8 s generate; Linux Release 219.9 s, including 123.4 + 6.2 s. Compile/link is only the remaining portion |
| 21:59:50 | Linux Debug GPU request while Windows Debug ScratchReuse owned device | Actual lease timeout in transcript; default admission budget 30 s. No overlapping GPU test launched |
| 22:00–22:10 | Boundary and tidy repairs | Linux Debug RustBoundaries timed out at 240 s. Windows tidy found adjacent path parameters after 197.89 s. Linux Release boundary assertions passed (204.04 s), but the enclosing checkpoint rejected the concurrent C++ test edit |
| 22:09–22:22 | Final broad native quality pair | Windows **525.57 s**, Linux **586.04 s**, passed; both again analyzed 101 native/CUDA commands |
| 22:10–22:29 | OFF production checks, ON restoration, final owning groups, boundary retries | Linux Release RustBoundaries 235.26 s and Linux Debug 220.83 s finally qualified; unchanged limits and diagnostics retained |
| 22:31 | Packet completed | Later independent review/R1 work is excluded from this timeline |

Stage sums: quality **4,513.05 s**; builds **2,968.59 s**; boundaries **1,454.17 s**; selected GPU **479.67 s**; production **304.19 s**; other capture/core/environment work **575.39 s**. Quality intervals cover **2,705.30 s**, builds **2,129.56 s**, boundaries **1,266.65 s**; these unions still overlap each other.

The ten actual native quality runs total **4,474.59 s**; failed native quality consumes **3,362.98 s**. The four freshness-only failures total **2,348.26 s**, approximately **1,223 s** interval union. The last Windows retry changed only `tests/ffi/film_recipe_native_test.inc` and its consuming `.cpp`, yet the invocation still selected 52 accumulated paths. This is stronger evidence for a bounded repair path than a generic call to increase analyzer concurrency.

See [slice-3-commands.md](../out/validation/performance-review/2026-10-17/slice-3-commands.md), [quality.json](../out/validation/performance-review/2026-10-17/quality.json), and [transcript-events.json](../out/validation/performance-review/2026-10-17/transcript-events.json). The two transcripts corroborate private campaign launchers and the GPU admission failure; repeated quoted logs in later reviews are not counted as new executions.

### What was repeated, and what was not

Both tasks used named targets and selected CTest groups; neither routinely dispatched `Quality.BuildEvidence`. Their private launchers compensated for the four-file development dispatcher's missing migration-domain coverage. Some repeated tests were necessary at capture, raw boundary, adoption and deletion transitions. The avoidable portion is broad passes launched before repairs settled, unrelated-source invalidation, and whole-cohort reruns for narrow repairs. A generic `repeat_reason` bypasses Campaign's identical-pass guard; it does not provide stage-level incremental reuse.

## 4. Local measurements versus the CI brief

| Mechanism | Local evidence | Assessment against CI |
| --- | --- | --- |
| Broad quality selection | Slice 2: 39 paths, 98 actual tidy commands in each successful run; Slice 3: 52 paths, **101** commands, at least **146 parsed source variants** including a minimum one per CUDA command | Reproduces broad analysis, but local tasks used the previous slice base and **jobs 4**, not CI's PR base/jobs 2. Local final passes were 7.6–9.8 min, not the CI 17–23 min |
| Header/variant expansion | `RustAssetBridge.cpp` parsed eight variants, 155.27 s WSL / 181.79 s Windows summed command duration; ProcessRoot and CUDA resources each four variants | Many variants have different test-hook defines and are required. Do not deduplicate by filename. `spectral_products_test.cpp` has four potentially equivalent entries to audit separately |
| Tooling tests | Windows restoration `Quality.BuildEvidence` **353.896 s**; whole serial tooling partition **517.13 s** test sum | Expensive locally, but **zero occurrences in the two migration task selections**. Lower priority for their everyday feedback than the brief suggests; important for tooling edits and final qualification |
| Rust privacy probes | Slice 3 final WSL Debug/Release **220.83/235.26 s**; Windows **39.66/66.59 s**; WSL Debug also had a 240 s timeout | Local WSL is slower than the cited hosted Linux result. This is not just the same CI bottleneck scaled to a faster workstation |
| Ordinary warm named build | Fresh Windows Release `JuicerFilmInputTests`: **0.497/0.507 s** including Campaign bookkeeping; receipt durations 0.462/0.478 s, parent commands 0.281/0.294 s. Historical Linux Debug `JuicerSpectralPreparationTests`: **1.069/1.481 s**; historical Linux `JuicerFilmInputTests`: **1.511 s** | No repeated compilation/relink in these no-op observations; CMake/Ninja/Cargo incrementality is not universally broken |
| `juicer_rust_core_tests ALL` | Fresh Windows Release **0.736 s** including Campaign bookkeeping; Cargo reports **0.05 s**, no compilation | Real unconditional freshness command, negligible relative to measured minutes. Removing it first would optimize the wrong cost and risk restoring cold compilation inside CTest |
| WSL reconfiguration | Slice 2: seven configures **554.2 s**, versus Windows seven **20.9 s**. Slice 3: ten **952.2 s**, versus Windows eight **27.7 s** | Major local divergence. Stage attribution is proven; exact internal CMake call attribution is not |
| Checkpoint scheduling | `qualification.py` constructs `ctest --test-dir ...` without `--parallel`; observed 121-test Debug host partition **500.75 s** test sum and **502.38 s** supervised duration, sequential Start/Pass log | Preset jobs=4 does not apply to these commands. Current local checkpoint behavior differs from preset-based CTest |

The frozen CI figures remain those in the supplied handoff: [Windows run](https://github.com/oldandangry/Film-Juicer/actions/runs/38992382423) and [Linux run](https://github.com/oldandangry/Film-Juicer/actions/runs/38992382358). They were not re-downloaded and are not local benchmarks.

### Numerical test comparison from retained local JUnit

Windows restoration columns use the completed retry2 Debug and final Release partitions at the same product HEAD, with evolving verification tooling and some competing host activity. WSL columns use Slice 2's four-preset adoption cohort. These are directly recorded local timings, not controlled single-test A/B experiments.

| Test | Windows Debug | Windows Release | Ratio | WSL Debug | WSL Release |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ffi.Host.InputColorPreparation | 138.326 | 3.411 | 40.6x | 110.961 | 3.936 |
| Reconstruction.Host.TcLut.admission | 45.683 | 0.864 | 52.9x | Not selected | Not selected |
| Ffi.Host.Cat02Preparation | 19.453 | 1.100 | 17.7x | Not selected | Not selected |
| Film.Host.InputProducts | 35.495 | 3.251 | 10.9x | 24.645 | 3.601 |
| Exposure.Host.Products | 40.190 | 12.876 | 3.1x | 52.966 | 24.066 |

All five Windows cases together account for **279.15 s Debug versus 21.50 s Release**. Release retains C++ assertions (`CMakeLists.txt:36`) and Rust overflow checks (`Cargo.toml:33`); that does not make Release equivalent to Debug diagnostics or justify deleting Debug qualification.

## 5. Root causes and falsifiable hypothesis assessment

Evidence classes: **M** = fresh local measurement; **L** = retained local command/JUnit/receipt; **C** = code-confirmed; **H** = untested causal attribution or proposed effect.

| Finding / trigger | Evidence, impact and frequency | Smallest remedy / regression witness |
| --- | --- | --- |
| **H3: missing everyday domain coverage; broad repair loops** | **L/C.** `verify.py:28,110,235` supports four files and Linux Debug; `build_evidence.py:534,638` repeats the restriction. Private Slice 2/3 launchers use cumulative `--base`, generic repeats and whole dirty-input sets. Ten quality calls in Slice 3, four complete stale-input failures | Extend existing owner selections for the actual film-input/DIR/FFI domains and native presets; scope repair checks, then freeze a final candidate. Relevant edits must invalidate; unrelated edits remain pending without destroying a valid scope |
| **2.1: whole analysis repeats instead of affected units** | **L/C.** `check-quality.py:1054,1140,1179,1232`; native source consumes all configured variants, runner changes broaden representatives, quality snapshots include broad source. Final Slice 3 101 commands/146 minimum variants repeated after small changes | Keep PR-wide final analysis; use changed-owner/consumer analysis and per-command success reuse in development. Compare exact analysis membership and inject a header-consumer failure |
| **H1/H4: verification identity drives physical invalidation, including unchanged Rust** | **L/C.** `build_evidence.py:307,506,520,528,571,601`; `build_dependencies.py:155` unlinks outputs. Final Windows restoration removed **231 outputs per preset** and cleaned both Rust packages; build attempts **163.27 s Debug / 269.17 s Release**. Compared preceding/final Release proofs have identical Rust bytes/timestamps, changed verifier files, configuration inputs/include inventory and environment digest. Five Cargo clean command receipts exist across restoration; none in the two migration task families | Separate compiler/artifact inputs from verifier identity and environment provenance. Environment/configuration drift expands `rust_changes()` to all `.rs`, whose existing timestamps then trigger stale-time cleaning. Thus the clean is not proof of a real restored-time Rust edit. Preserve genuine restored-time detection; test on a disposable tree |
| **H2: environment-sensitive checkpoint reuse** | **C; local frequency unresolved.** `qualification.py:140` hashes nearly the whole environment. Reuse exists in historical Windows boundary checkpoints, so it is incorrect to say reuse never works. Restoration identities changed along with relevant tooling; no env-only checkpoint A/B was run | Record a compile/check-relevant identity separately from a redacted environment digest. Irrelevant sentinel drift must preserve reuse; compiler/linker/Cargo flags must invalidate the appropriate stages |
| **H6: WSL filesystem and configuration overhead** | **M/L.** Same 172 files, 3.90 MB: median stat **0.2092 s WSL / 0.00356 s Windows**; resolve **0.5271 / 0.01133 s**; hash **0.3875 / 0.01213 s**. WSL process CPU for those operations only about 0.013/0.029/0.031 s. Historical configuration totals above. Metadata/path overhead is measured; ext4's benefit, Defender's contribution and the exact 95 s CMake hotspot are **H** | Reduce repeated discovery/resolution inside existing validation boundaries; profile one disposable configure and one privacy cohort. Compare an isolated ext4 setup before proposing relocation. Never move a live cache |
| **H5: orchestration contention and process drainage** | **L/C.** One GPU admission timeout in Slice 3's implementation interval, nominal 30 s limit; retry followed the existing owner. Retained route-reuse lease waits were only 0.010–0.013 s. Installed NVCC/VCTIP probe parent 2.06 s versus 32.28 s drained. New restoration builds retain a 30 s post-parent VCTIP grace. No evidence that ordinary build-lock waits explain most task hours | Serialize device work explicitly, retain ownership; distinguish queue wait, parent execution and descendant drain. Review the already-present Windows repair instead of reimplementing it or increasing every timeout |
| **2.2/2.3: expensive work chosen for the feedback tier** | **L/C.** Tooling group and Debug numerical timings above; `tests/CMakeLists.txt:1227` already separates relevance labels. Heavy tooling is appropriate for the current tooling repair, but not every film edit | Use existing labels for relevance and optimized exhaustive numerical feedback, with focused Debug diagnostics. Retain all full-candidate tests and compiler privacy controls |
| **Additional local finding: serial checkpoint CTest** | **L/C.** `qualification.py:366` builds commands without explicit parallelism. Debug product 502.38 s supervised; tooling 517.95 s. This is separate from deliberate GPU/Cargo resource locks | Add explicit resource-tested worker selection to checkpoint commands; preserve partitions, locks, deadlines and exact JUnit membership |
| **2.4: unconditional Cargo freshness** | **M/C.** `tests/CMakeLists.txt:53`; warm measured cost below one second. `cmake/JuicerRust.cmake:93` stamp/archive unchanged in warm named builds | Retain current behavior initially. Any future incremental replacement must track included fixtures and feature/profile/toolchain inputs and prove no compile occurs inside Rust.Core |

See [invalidation-comparison.json](../out/validation/performance-review/2026-10-17/invalidation-comparison.json) for the unchanged-Rust comparison. **No env-only destructive probe was run.** The measured 163/269 s is the full rebuild-attempt cost, including genuine restoration obligations and approximately 30 s drainage; it is an upper bound on avoidable cost, not all attributable to environment drift. Historical raw logs lack complete CPU/RSS/I/O counters and build-lease timing; the rest of each build cannot be cleanly assigned to compiler CPU, storage or orchestration.

The latest retained unchanged development receipts already reuse results: **19.469/21.732 s before publication**, documented full supervised times **20.028/22.177 s**, with **14.972/16.147 s** reuse validation. Older 148 s unchanged receipts predate the landed simplification and are not the current baseline. No new verifier call was made against the valuable live cache merely to rediscover its invalidation behavior.

## 6. Ranked implementation plan

Effort estimates are engineering estimates, not commitments. Benefits overlap; do not sum the savings column into a promised task speedup. Each item must ship with the benchmark and correctness witnesses below. Implement no further mechanism unless the earlier item leaves a measured need.

### P0.1 — Make the normal edit path complete for the active domains

**Root / locations:** four-file/Linux-only dispatch in `scripts/verify.py:28,110,194,231`, duplicated admission limits in `scripts/build_evidence.py:534,638`; current owner documents `tests/route/development.json`, `tests/ffi/output-color-development.json`; future domain entries beside `tests/CMakeLists.txt:182–208,985–995`. The historical private `run.py`/`quality.py` launchers demonstrate the missing workflow; do not promote those whole-dirty-set wrappers as the solution.

**Change:** extend the existing explicit ownership scheme first for film-input, Development, DIR/density and their FFI/native tests on both native platforms. One command should print its affected targets, Cargo package/features, native consumers, exact tests and unresolved paths before execution. Keep boundary/compiler-probe membership conservative for public API changes. A leaf repair must not silently fall back to full qualification. Perform producer compilation/tidy before broad consumer work; finish repairs before dispatching the stable final candidate. Reuse full-suite JUnit for domain evidence instead of adding another domain rerun.

**Baseline → target / expected savings:** Slice 3 paid about 20.4 wall minutes for two complete stale quality pairs; avoiding one such pair saves roughly **9–11 elapsed minutes** on a similar task. This is a task-level opportunity, not a frequency claim for every edit. Existing two-file explicit analysis is 31.61 s Windows / 60.81 s WSL; target ordinary leaf feedback **under 60 s Windows, under 90 s WSL initially**, with shared-header/large bridge changes separately budgeted. No-op scope reuse target **≤10 s total**, from 20–22 s. Broad header edits may legitimately retain 101 consumers.

**Risk / verification:** missing consumers or accepting an old pass after a second edit. Benchmark B2/B3; add unseen-header-consumer, mixed supported/unsupported edit, staged/unstaged cancellation, changed fixture, missing artifact and second-edit controls using existing verification tests. Global unresolved work must remain visible. **Order 1; 2–4 engineering days.** This can initially use fresh focused checks without new persistent analysis caching.

### P0.2 — Separate artifact freshness from proof/provenance changes

**Root / locations:** `build_evidence.py:307,458,506,520,571–602`, `build_dependencies.py:155`, `qualification.py:140,205,399`, `execution.py:272`. Broad environment/proof changes currently reach output deletion and Rust package cleaning.

**Change:** retain a full hashed environment digest as provenance, but define the actual compile/Cargo/check inputs once and consume that identity at the owning stage. Include resolved compiler/linker identity, ordered include/library search, target/runtime/flags, toolchain/configuration, relevant Cargo configuration/features and profile. Verifier repairs invalidate proof and rerun verifier controls; they should not automatically delete otherwise fresh native products. Use dependency/content evidence for targeted invalidation. Permit Cargo to determine freshness for normal edits; reserve package/reverse-consumer cleaning for proven same-or-older-time content changes or a specifically demonstrated Cargo limitation. Emit the changed identity fields and invalidated output counts, without environment values.

**Baseline → target / expected savings:** restoration demonstrates five package-clean invocations and two final 231-output invalidations despite unchanged Rust. Target **zero deletions, recompilations or relinks** for an irrelevant variable or verifier-only edit; a fresh named build remains approximately 0.5 s Windows. Potential savings **2–4 minutes per falsely broad rebuild**, conditional on trigger; actual ordinary-edit incidence is unmeasured. This is a correctness-sensitive false-invalidation fix, not authorization to trust mtimes alone.

**Risk / verification:** stale ABI/headers/generated inputs, missed Rust included fixture/configuration changes, forged cross-owner reuse. B4 must compare irrelevant sentinel drift against relevant flags, compiler, manifest/lock, ABI and same-mtime edits in a disposable fixture project. Preserve overlapping/pending owner rejection and generated/unowned header controls. Rerun both native tooling groups and current restoration regressions before changing live identity rules. **Order 2; 2–4 days.** Gate P0.3's reusable keys on this contract.

### P0.3 — Reuse analysis at actual compilation-unit granularity

**Root / locations:** `check-quality.py:237,1054,1140,1179,1232`; `.github/workflows/tests-linux-host.yml:105`, `tests-windows-host.yml:169` retain final PR-base gates. Compilation database variants are the unit of semantics, not just source filenames.

**Change:** daily analysis selects changed producers and true transitive header consumers. A successful unit can be reused only with matching source/dependency contents, ordered compilation arguments, generated inputs, analyzer/tool/config identity and diagnostics completion. Keep failure history; a changed unit never inherits a pass. Keep the final PR-base complete analysis available and required. Do not take `--jobs 4` as an automatic recommendation for all machines or overlap it with another broad scan; it is the measured local baseline. First implement P0.1 selection; add persistent per-unit reuse only for remaining demonstrated repetition.

**Baseline → target / expected savings:** final broad passes **526/586 s**, 101 invocations; narrow existing native checks **27–61 s**. For a repair affecting one/few modest units, target **≤60 s**, plausibly saving **7–9 minutes per avoided broad invocation**. A genuine `RustAssetBridge.cpp` edit still needs eight distinct hook variants, whose current analysis takes 155–182 s; do not promise a sub-minute result there. Four failed full passes in Slice 3 quantify the repeated-work opportunity.

**Risk / verification:** stale included headers or dropping alternate macro/CUDA variants. B2/B3 must include source, transitive header, response file, include-order/shadow header, `.clang-tidy`, LLVM, CUDA/host-compiler and feature changes. Require failures from all actual variants and correct cancellation after first diagnostic. **Order 3; 2–4 days**, after P0.2 if persistent reuse is added.

### P1.1 — Remove measured WSL discovery overhead before moving caches

**Root / locations:** observed WSL mounted-filesystem costs; repeated inventories in `build_evidence.py:332`, `build_dependencies.py` (`NinjaGraph`, `CompilationInputs`), `verify.py:209–361`; Rust probe copy/command flow `tests/quality/test_rust_boundaries.py:29–95`. CMake configure entry points include `cmake/JuicerRust.cmake:29,43,78,125` and `tests/CMakeLists.txt:17`; their individual contribution is not yet attributed.

**Change:** instrument one disposable reconfiguration with CMake profiling and one boundary cohort with command/startup/copy timings. First remove repeated path normalization, enumeration and hashing within the existing fresh validation boundary; preserve new discovery after configure/build/check execution and content freshness across invocations. Then compare a bounded identical WSL probe on ext4 versus `/mnt/c`, including Rust toolchain, source, Cargo and temporary-workspace placement. Decide on a separately configured WSL workspace/cache only if measured savings justify changing the current real-project workflow. Never relocate live CMake/Cargo caches or edit stored absolute paths.

**Baseline → target / expected savings:** Slice 3 configure median **95.15 s**, total **952.2 s**; privacy cohort **221–235 s**; latest no-op receipt 20–22 s. Initial acceptance targets: unchanged configure **≤30 s**, boundary cohort **≤120 s**, unchanged scope **≤10 s** on WSL, all provisional. If achieved, save roughly **1 minute per reconfiguration** and **1.5–2 minutes per privacy cohort**. Same-file metadata comparison supports the investigation, not these projected ext4 gains.

**Risk / verification:** invalid discovery caching, hidden Cargo-config changes in copied probes, path/casing aliases, differing toolchain semantics. B5 must pin identical files, flags, toolchain and fixture hashes; verify positive and intended compile-fail diagnostics in both profiles. Profile before choosing a CMake edit. **Order 4; half-day profiling, 1–3 days conditional implementation.** No Defender exclusion or security-setting change is proposed.

### P1.2 — Give checkpoint CTest explicit, measured scheduling

**Root / locations:** `qualification.py:112,356–378` constructs `ctest --test-dir` without preset execution settings; `CMakePresets.json` supplies four workers only for preset invocations; `execution.py:507` owns cross-caller/device exclusion.

**Change:** expose/bind an explicit checkpoint worker count and pass it to every host partition; retain GPU/Cargo resource locks and separate device admission. Start with two workers for a bounded host cohort, then compare four, while sampling memory/CPU/I/O. Avoid overlapping full privacy suites and broad analysis on WSL. Record queued, executing and draining time separately; a device request should queue behind the known campaign owner instead of repeatedly timing out and recreating membership directories.

**Baseline → target / expected savings:** serial Windows Debug product **502.38 s**, serial tooling **517.95 s**. Target product **≤350 s** if the unchanged cohort supports it; projected saving **2–3 minutes per complete Debug checkpoint**, with no claim yet for tooling (BuildEvidence alone is 354 s). One observed GPU admission miss adds roughly a 30 s deadline and retry overhead; ordinary build-lock dominance is unproven.

**Risk / verification:** timeouts under contention, hidden shared outputs, broken cross-native GPU exclusion. B6 must preserve identical membership, deadlines, resource locks, failure/interrupt drainage and no skipped tests; include a busy-device negative control. Do not increase public timeouts to obtain a pass. **Order 5; 1–2 days.** Full 121-test benchmarking follows a successful small-cohort experiment, during planned qualification rather than a new diagnostic campaign.

### P1.3 — Use optimized exhaustive numerical feedback, retain full Debug qualification

**Root / locations:** `tests/ffi/color_preparation_test.cpp:1901` and owning CTest registrations `tests/CMakeLists.txt:193,985,992,1029`; C++ assertion retention at `CMakeLists.txt:36`; Cargo release overflow checks at `Cargo.toml:33`. Large Debug scenario matrices are real computational work.

**Change:** the daily owner map runs exhaustive numerical matrices using the existing Release configuration when already prepared, plus the affected Debug failure/lifetime/assertion witnesses. If necessary, add explicit named case selection in the owning test executables; default/full execution must still enumerate every original scenario. Keep all current Debug and Release groups at final qualification. No fixture regeneration, tolerance relaxation, dropped negative control, CPU pixel fallback or blanket optimized substitution.

**Baseline → target / expected savings:** the five measured Windows tests **279.15 → 21.50 s** for the optimized portion; target daily combined optimized matrix plus focused Debug witnesses **≤90 s**, roughly **3 minutes saved per such feedback cohort**. InputColor alone offers about **135 s** before the added Debug witnesses. These ratios already exist; the combined new workflow is not yet benchmarked.

**Risk / verification:** optimization hiding UB/lifetime failures or selector omitting scenarios. B7 must compare full scenario manifests and exact fixture hashes, execute error/ownership controls in Debug and both full suites on the final candidate. Cold Release preparation must be reported separately and must not silently turn a quick check into a full rebuild. **Order 6; 1–2 days.**

### P1.4 — Keep tooling self-tests relevant and qualify the existing Windows drain repair

**Root / locations:** tooling labels at `tests/CMakeLists.txt:1227`; real integration setup at `tests/quality/test_build_evidence.py:193,212`; Windows helper identification/drainage at `process_tree.py:224,322–390`, policy selection in `build_evidence.py`, Campaign identity in `execution.py:235–285`.

**Change:** preserve the existing tooling partition; the everyday domain selector should schedule it for build/verifier/process/registration changes and final qualification, not unrelated film edits. Start tooling repairs with the relevant method(s), then run the full native tooling group once the candidate is stable. Independently review and qualify the already-present VCTIP rule and generated-header fix before treating restoration as closed. Do not introduce a second cleanup mechanism. Consider shortening its 30 s grace only after observing helper type, required output completion and other-child drainage in a separate bounded study.

**Baseline → target / expected savings:** BuildEvidence **353.896 s**, tooling partition **517.95 s**; target zero unrelated tooling executions in routine product feedback, which the two measured migration tasks already meet. Thus **no new Slice 2/3 savings** are credited. A narrow tooling repair could avoid one approximately **8.6-minute partition** rerun. Current installed compiler grace costs about **30 s per triggering supervised invocation**; no grace reduction is promised.

**Risk / verification:** missing a verifier regression, terminating a PDB writer, or accepting undrained work. B8 retains complete tooling once per native platform, exact VCTIP image/Job checks, parent failure, unexpected child, exit-race, generated-header and stale-output tests. Default final qualification membership stays unchanged. **Order 7; 1 day review/selector work**, separate from acceptance of the existing restoration candidate.

### P2.1 — Audit equivalent analysis/build variants; defer blanket caching work

**Root / locations:** host tidy passes the full compilation database at `check-quality.py:1115`; repeated shared test source registrations in `tests/CMakeLists.txt`. Different test-hook variants are intentional; four `spectral_products_test.cpp` entries have matching inspected defines but require full argument/include comparison before claiming equivalence.

**Change:** deduplicate only proven equivalent AST command contexts, keeping the association to every consumer. If native compilation itself is material, share an existing test-helper object target only where all compilation semantics match. Do not collapse RustAssetBridge's eight distinct hook variants. Evaluate compiler caching/CI cache topology only after structural reruns are fixed; key by toolchain, ABI, flags, generated inputs and profile.

**Baseline → target / expected savings:** spectral-products analysis **64.25 s WSL / 84.47 s Windows** for four parses; equivalent single-context execution could save up to **48/63 s of command work**, much less wall time in a parallel batch. No local cold compiler-cache gain was measured; do not claim one. B9 requires identical diagnostics per consumer and a changed-define negative control. **Order 8; 0.5–1 day audit**, implementation conditional on equivalence and material savings.

### P2.2 — Leave Cargo test freshness/stamp semantics intact unless new evidence warrants change

**Root / locations:** `tests/CMakeLists.txt:53`, `cmake/JuicerRust.cmake:78–116`. The test-preparation target always invokes Cargo; the product archive uses a separate success stamp to avoid relinking an unchanged archive.

**Change:** no immediate replacement. Retain it as an accounted fast step; only if new measurement shows material overhead, make preparation dependency-complete for Rust sources, `include_*` fixtures, manifests, features, profile and toolchain. Never simply delete `ALL` and return cold Rust.Core compilation to its 120 s test budget.

**Baseline → target / expected savings:** warm Windows Release **0.736 s**, Cargo **0.05 s**; target **≤1 s**, no recompilation/relink. Available saving is below one second per invocation on this observation, not minutes. B1/B9 require test-only and included-fixture edits, profile/feature changes and cold preparation to rebuild before CTest. **Order 9; no implementation now; half-day bounded investigation only if the cost changes.**

## 7. Before/after benchmarks and retained qualification

Every future benchmark must retain commit + dirty-file hashes, preset, tool versions, exact environment *identity* without secret values, filesystem, named targets/tests, warm/cold state, selected membership and competing activity. Record process wall time separately from queue, supervision, configuration, compilation/linking, test execution and proof/reporting. Use three warm repetitions and median/range where cheap; a single slow cohort may use existing receipts plus one bounded repeat. Count compiler/linker/analyzer commands and invalidated files, not just elapsed time.

| ID | Reproducible experiment | Baseline and acceptance |
| --- | --- | --- |
| **B1** Warm build | In the existing native developer environment, run twice: `cmake --build --preset windows-clang-release --target JuicerFilmInputTests`; separately `--target juicer_rust_core_tests`. Repeat equivalent Linux commands only after that cache is already current. Hold its build lease | This review's [warm_probe.py](../out/validation/performance-review/2026-10-17/warm_probe.py) and [windows-warm.json](../out/validation/performance-review/2026-10-17/windows-warm.json): 0.50/0.51/0.74 s inclusive. After: no compilation/relink, unchanged executable/archive/stamp, ≤1 s Windows; preserve historical Linux no-op behavior, not a cold-build target |
| **B2** Analysis selection | Existing command: `python -B scripts/check-quality.py --preset <native-debug> --files src/RustColorBridge.cpp tests/ffi/color_preparation_test.cpp --jobs 2`. Compare printed owner/variant list with final `--base <accepted-parent>` membership; only execute when files/configuration are stable | Retained two-file runs 31.61/60.81 s Windows/WSL; broad Slice 3 525.57/586.04 s. After: irrelevant consumers absent in daily mode, every relevant variant present. Full-base gate still runs for final qualification |
| **B3** Reuse / actual edit | Existing supported starting witness: `python -B scripts/verify.py development --preset linux-debug --files src/ScanRoute.cpp`; repeat unchanged using printed receipt. For future domain mappings, apply a real bounded leaf repair in an isolated disposable checkout, then a second edit, transitive header and ABI edit | Existing header-scope reuse 20.03/22.18 s full supervised. After: no checks on unchanged reuse, ≤10 s; leaf checks match P0 targets. Preserve pending unrelated files. This verifier was intentionally **not** run against the current cache here |
| **B4** Invalidation correctness | Disposable existing miniature build-evidence fixture: baseline, unchanged repeat, add irrelevant sentinel env key, change compile flags/toolchain, restore original source timestamp after content edit, change generated/unowned header, interrupt owner, overlap target sets | Before: recorded broad invalidations/cleaning; env-only physical effect untested. After: zero invalidated outputs for irrelevant drift/verifier-only edits; every affected output changes for actual source/ABI/toolchain cases; no stale successful receipt |
| **B5** WSL configuration / probes | Re-run the retained 172-path `read_probe.py`; profile one identical disposable CMake configure (`--profiling-format=google-trace --profiling-output=<ignored-log>`) and one filtered Rust privacy cohort. Compare native filesystem placement only in a separately configured disposable environment | Median file operations and 95.15 s historical configure above. Record configure internals, subprocess count, CPU/RSS/I/O and copy time; require measured gain before changing placement. Do not include cold setup in the warm comparison |
| **B6** Scheduler | In already-built caches, compare `ctest --test-dir out/build/<preset> -R '^(Film\.Host\.InputProducts|Ffi\.Host\.Cat02Preparation)$' --parallel 1`, then 2 (and only if healthy, 4), each with a unique JUnit path. Follow with scheduled complete product qualification | After: identical names/results/deadlines, lower elapsed time and no memory/lock regression. A later complete Debug partition targets ≤350 s from 502.38 s; this gain is not yet measured |
| **B7** Numerical feedback | Compare `ctest --preset <debug-or-release> -R '^Film\.Host\.InputProducts$' --output-junit <unique-path>`, followed by the owning InputColor cohort only when justified by the edit. Use the proposed Debug witness selection after implementation | Retained product ratios above. Exact scenario membership/fixture hashes and numerical acceptance preserved; full Debug/Release still required. No tolerance changes |
| **B8** Tooling / Windows lifetime | Relevant `unittest` method first, then `ctest --preset <native-debug> -R '^Quality\.(BuildEvidence|Execution|BuildRegistration)$'` during planned tooling qualification. Keep supervised installed NVCC compile/link/runtime witness and unexpected-child controls | Full tooling remains mandatory for its own changes and final qualification. Before: 353.896 s BuildEvidence and approximately 30 s helper drain. Any improved grace requires a separate approved lifetime result |
| **B9** Variants / Cargo | Compare normalized compilation contexts and diagnostics; change a define to ensure variants separate. For Cargo, modify test-only/included-fixture inputs solely in a disposable copy; run preparation, then Rust.Core | Distinct macro/CUDA variants never disappear; equivalent consumers share only proven equivalent work. No cold compilation moved into CTest |

These commands are benchmark recipes, not instructions to rerun this entire table after every edit. Destructive same-mtime/invalidation cases belong only in disposable fixtures, never in the live warm caches.

**Final candidate qualification remains complete:** once stable, build and run the applicable unfiltered Windows and Linux Debug/Release suites, including scientific/reference matrices and exact manifests, Rust privacy/naming/default/test-support and ABI boundaries, CUDA/GPU routes/lifecycle, OFX contracts, and both testing-OFF Release production/export/isolation gates. Retain the existing required hosted statuses and complete PR-base native/CUDA/Rust quality plus supporting-code lint. Extract focused results from the final JUnit rather than execute them again. Tooling once per native platform and corpus once remain explicit aggregate obligations; if the chosen unfiltered suite already supplied them, reuse that exact evidence. Resolve-installed library/render/visual, exceptional hardware/context-loss/live-recovery and applicable toolchain acceptance remain separately required. Optimizing daily feedback changes none of those guarantees.

## 8. Remaining uncertainties and follow-up boundaries

- No complete campaign, new scientific matrix, new GPU/OFX/Resolve run, cache cleaning, deliberate invalidation, source edit, fixture regeneration or receipt deletion was performed for this review. Existing qualification failures were not relabeled.
- No matched ext4 build/configure/probe comparison, Defender/ETW attribution, ThinLTO profile, compiler-cache experiment, or concurrency sweep was performed. Current low WSL memory use/no swap does not establish historical pressure during the long runs.
- Native Windows was executing another Debug qualification while this review observed the machine. Historical and fresh numbers are local observations, not isolated-machine p95 measurements. Fresh file-operation probes are sequential, warm and interpreter-version-different; they establish a large local access-cost difference, not a precise filesystem-only causal multiplier.
- No environment variable names/values explaining the restoration digest drift were retained in its receipts; only hashes are available. Verifier/configuration and environment changes coincided. The pure irrelevant-variable physical invalidation experiment remains for B4. Genuine restored-time editing frequency outside these two tasks is unknown.
- Detailed historical compiler CPU, peak RSS, I/O, per-command queue time and configure-call profiles are unavailable. Configuration/generation/Cargo summaries come from logs; the remaining compile/link/staging time is not fully separable. Old pre-simplification proof-inventory costs were not used as today's baseline.
- Current Windows restoration code is pre-existing uncommitted work; this performance review does not replace its independent correctness review or declare its in-progress Debug qualification complete. Its partial generated-header/Job fixes must be reconciled before P0.2 edits those files.
- Fresh WSL no-op verification/build and another BuildEvidence run were deliberately skipped: current evidence already establishes their behavior, WSL needs regeneration, and the verifier can physically invalidate cached products. Long or destructive follow-ups require an explicitly scoped disposable setup and retirement record; live-cache relocation needs a separate owner decision under the output-layout rules.

Contextual review covered ownership and naming at artifact/proof identity construction, invalidation consumers, CTest selection, native/CUDA compilation variants, Cargo stamp ownership and process/lease completion. The design issue is conflation of artifact freshness with broad verification identity, not a naming-style preference. This review did not audit the entire restoration diff for product correctness and makes no scientific, ABI, GPU or host-acceptance claim.
