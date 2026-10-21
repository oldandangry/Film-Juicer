# Film-Juicer performance remediation plan V2

**20 October 2026 — execution policy corrected after the migration audit.**
The cadence below replaces this plan's unconditional four-preset qualification
at every slice handoff. The canonical rule is
[Validation cadence](../tests/README.md#validation-cadence). The initial correction
changed documentation and execution policy only. Separately authorized harness
fixes implemented request admission and OFF evidence retention on 20 October,
then matching-profile Clippy, documentation discovery and bounded preparation
recovery on 21 October. Their focused validation is recorded
below; no product code changes, product builds or evidence disposal accompany them.

Preserve [the original plan](PERFORMANCE_REMEDIATION_PLAN.md) and the
[earlier V2 investigation](../out/validation/performance-review/2026-10-20/V2-reprioritization/before/PERFORMANCE_REMEDIATION_PLAN_V2.md).
Historical timings retain their original scope and qualification limits; they do
not establish whole-task savings.

Evidence links under `out/` in this plan and the original investigation refer to
retained local checkout files, which are not published with the repository. The
original plan is historical context; the current execution policy is defined here
and in the linked public validation cadence.

## 1. Decision and delivered work

First use existing commands, complete consumer analysis and apply the corrected
cadence. Then repair demonstrated amplification in the existing preparation and
qualification owners. Do not begin another selector, proof store, owner map,
private execution framework or standalone batching project.

| Item | Current disposition |
| --- | --- |
| Local candidate | The audit examined HEAD `3569afe49bd3778437aa3c567ef4831bb29c4a23` plus uncommitted S4.C1 work. Inspect current source and execution state before further work; HEAD alone is insufficient. This plan does not review or accept that candidate. |
| P0-A | Landed as `e6ca3b8`, merged through PR #71. Preparation renewal separates artifact facts from renewed receipt provenance. Preserve current-proof, membership, payload and collection controls. |
| P0-B1 | Landed as `4506b4a`, merged through PR #72. An independently fresh first entry into a target scope need not destroy outputs merely because that scope lacks a receipt. General interrupted-owner recovery is not delivered by B1. Miniature measurements do not establish task savings; Windows full-overhead calls regressed. |
| BuildEvidence fixture follow-up | Landed at `3569afe`. One matched serial cohort fell from 816.319 to 681.690 seconds and 148 to 129 commands, a 16.49% local reduction. This is test-work evidence, not an accepted whole-task speedup. |
| P0-C optics | [Withdrawn and rolled back](../out/validation/performance-review/2026-10-20/P0-C-optics/IMPLEMENTATION_REPORT.txt) because staged runtime Resources were absent from its freshness proof. Apparent gains are withdrawn. The maintained selector still covers only four route/output-color inputs on Linux Debug. |
| E5 placement investigation | [Completed](../out/validation/performance-review/2026-10-20/E5/REPORT.txt) with no placement change. Full preparation attribution was inconclusive. Keep the checkout and caches in place; no follow-on experiment is scheduled. |

## 2. Measured problem and limits

The [S4.C1 elapsed audit](../out/validation/performance-review/2026-10-20/S4-C1-elapsed-audit/README.md)
and [accounting](../out/validation/performance-review/2026-10-20/S4-C1-elapsed-audit/accounting.json)
cover a fixed **299.14-minute prefix**, ending at 19:55:06 Europe/Stockholm on
20 October. Qualification was still running. Its interval accounting reports
199.47 minutes with a supervised receipt open, 31.52 minutes of other shell work,
0.62 minutes of other tools, 60.43 minutes of active agent time without observed
tool activity, and 7.10 minutes between turns. Generation, compaction and service
latency cannot be separately attributed within the agent-side remainder.

The same prefix contains **79.55 minutes in the union of hash intervals**,
already included above. One 1.320-second OFX capture recorded 57.720 seconds of
hashing. Four adjacent route captures accumulated roughly 204 seconds of hash
work per Linux profile. These establish repeated admission work; overlapping
command/hash sums are neither CPU time nor guaranteed recoverable elapsed time.

The earlier audit counted 62 private Python helpers. Later execution also exposed
missing host and GPU expectation consumers and private-driver errors. Complete
consumer analysis and maintained commands should precede another broad run.
Helper counts do not measure wasted minutes. The native/Rust numerical discrepancy
required an owner decision; approved sample changes also changed recipe identities
and glare seeds. Preserve numerical provenance, failures and downstream consequences.

The [Slice 2/3/6 audit](../out/validation/performance-review/2026-10-18-elapsed-audit/README.md)
corroborates repeated analysis/build work. Its timings do not add to the S4.C1
denominator. Retire the old 30/45-minute Slice 6 forecast as a current savings
claim. No whole-task improvement has yet been established.

## 3. Execution policy now

Use one current handoff, existing attempt records and one final packet linking
original reports. Before expensive execution, name the changed producer, complete
construction boundary, risky primitives, identities/seeds, expectation readers,
actual callers and staged runtime resources. Use registrations and source
inspection; no new dependency database is needed.

Diagnose risky numerical operations at the smallest valid boundary early. Complete
required native characterization, independent expectation adoption, reader checks
and later assertions before freezing a family cutover. Keep original fixtures;
ordinary tests must not regenerate their oracle or widen bounds. Source deletion
follows replacement consumers, caller/parity evidence and ownership checks.

**Add zero new or copied private orchestration programs.** This includes launcher
variants, final-matrix/OFF drivers, retention programs, collectors and equivalent
orchestration hidden in long inline scripts. Scientific cases belong in the
existing owning test/capture entry point. If a maintained operation is missing,
identify that one capability and change its owner when separately authorized.
A short read-only report over existing records is not a second qualification path.

During editing, use focused CMake/CTest and quality commands. Exercise new compiled
cases on both native platforms early. Finish formatting, generated inputs,
expectation adoption and affected checks before qualification. At a broader
checkpoint, use its built-in quality and JUnit for the same domain evidence;
do not add an identical full pass merely to produce another report.

Do not edit consumed inputs during their execution against that mutable checkout.
Documentation inspected by quality checks remains frozen for those checks. The
repository prose excluded by the public cadence no longer invalidates completed
build/test evidence. Build/device leases do not protect source from other writers. Queue unrelated
edits while valid work finishes; cancel only affected work through its supervisor.
Parallel execution must respect build/Cargo targets and shared GPU ownership.

## 4. Remediation order

The request-admission and OFF evidence-retention fixes below were separately
authorized and implemented in the working tree on 20 October 2026; matching-profile
Clippy, documentation discovery and bounded preparation recovery followed on 21 October. Other code
changes remain **planned, not delivered or authorized for execution by this update**.
Use existing owners and focused
regression fixtures when each change is authorized.

| Priority | Bounded change | Acceptance and limit |
| --- | --- | --- |
| 1 — policy and preflight | Apply section 8 and the workflow above using existing commands. | Zero private orchestration, complete expectation consumers before freeze, no duplicate broad pass for another packet. No code project is needed to begin this policy. |
| 2 — request admission | Implemented: reject invalid stage names, undeclared stages, empty commands/inputs and nonpositive/nonfinite timeouts before preparation ownership or checkpoint reuse. | Focused Linux/Windows regressions preserve admitted results and absent/pending owners; real execution failures still require repair. No manual owner clearing. |
| 2 — bounded preparation recovery | Implemented: require the original attempt's sealed proof that every started command drained before replacing its pending owner under a healthy normal build-tree lease. | Preserve original pending bytes and failed receipts, then run fresh preparation. Missing, damaged and legacy proof remains blocked. Hard owner death and blocked leases are outside this fix; no lease takeover or PID-only inference. |
| 2 — evidence scope and OFF collection | Implemented: retain inspected OFF build artifacts/configuration during execution and collect those original receipts after ON restoration or repair of unrelated pending work. Dependency-closure separation remains deferred. | Retained copies bind execution-time hashes; source, tool, payload and latest-owner checks remain mandatory. A passed scope with an incomplete aggregate remains eligible, while current pending work still blocks aggregate completion. Old manifests without retained build inputs require fresh OFF execution. |
| 2 — documentation discovery | Implemented: omit known repository prose from broad candidate/checkpoint discovery and initial development pending paths. | Prose edits, additions, deletions and commits preserve build/test reuse. Explicit inputs, build dependencies, packaged documents, runtime fixture payloads and quality checks remain fresh. Older receipts keep their original inputs; no retroactive projection or evidence rewrite. General dependency-closure separation remains deferred. |
| 3 — matching-profile Clippy | Implemented: run the matching profile once per preset while retaining both platforms, packages and plug-in feature forms across the aggregate. Both native CI lanes now explicitly select Release Rust after the Release build. | Four full preset invocations dispatch 12 Clippy analyses instead of 24: core, plugin production and plugin test-support in each matching context. Existing CI previously obtained both profiles through Debug-only calls; its complete Rust coverage remains 12 analyses across the two native lanes. Elapsed savings remain unmeasured. Boundary/privacy probes remain distinct. |
| 3 — necessary domain batching | Batch adjacent cases only within an already-needed domain extension, or after demonstrating payback from named remaining consumers. | Preserve complete inputs, individual results, process isolation where needed, membership and drainage. A standalone 0.5–1.5-day H1 patch for roughly two minutes per cohort is no longer the first code project. |
| 4 — ordinary closeout | Retire eligible scratch and obsolete helpers under section 7. | Storage/navigation benefit only; no compiler-speed claim. No disposal accompanies this update. |

General selector expansion, filesystem moves, analyzer caches, broad environment
normalization, new benchmarks and identity frameworks remain deferred. A P0-C
retry must first bind actual staged resource bytes and namespace membership.
Do not justify more machinery with invented future repetitions.

## 5. Current tooling constraints

The cadence is policy; public tools acquire no new capabilities through this edit.
`scripts/verify.py qualification --selection product` still schedules full
preparation and quality. `collect-qualification` still requires its complete
four-profile product/boundary evidence, tooling once per native platform, corpus
once and both Release production gates. Focused handoff evidence is not a passed
full aggregate. Use ordinary focused commands where appropriate rather than
claiming that the full collector supports leaf-only completion.

`Campaign.check_run_request` now rejects invalid request shape before preparation
ownership changes and checkpoint reuse. Nine focused regression tests passed on
each native platform, including real miniature preparation/CTest consumption and
failure recovery. This establishes the bounded harness contract, not product or
full aggregate qualification. Run logs remain under
`out/validation/{linux-debug,windows-clang-debug}/stage-request-admission-2026-10-20/`:
`focused-regressions.log` records seven passes and two preparation errors;
`build-regressions-repair.log` records both preparation controls passing after the
build/configure command binding was corrected. Linux `before-fix.log` records the
original pending-owner defect. Failed fixture paths are recorded in those logs
and remain intact.

OFF collection now uses execution-bound retained build files and configuration.
It accepts a completed scope whose original aggregate was incomplete, while
checking current pending and same-scope ownership before aggregate admission.
Fresh production dispatch remains testing-OFF only; collection reuses the original
receipt without rerunning its probes. Old manifests are not upgraded or backfilled.

Focused validation passed eight controls on Linux and nine on Windows, including
actual miniature OFF-to-ON archive changes, retained-payload tampering, source/tool
freshness and failed-owner rejection; Windows also covers PDB repair. Logs are in
`out/validation/{linux-debug,windows-clang-debug}/off-evidence-retention-2026-10-20/`.
The initial `focused-regressions.log` files each contain one test-class selector
error; only that missing control was rerun, successfully, in
`membership-control.log`. Retained fixture paths are recorded in the logs.
No product build or full product qualification was run.

Clippy now selects the preset's profile at command construction, keeping the native
target, preset Cargo directory, locked dependencies, all targets and warning denial.
Workspace formatting, tool-version checks and dependency admission are unchanged;
bounded development selections retain their packages, features and test filters.
Debug presets use `dev`; Release presets use `--release`, with matching log labels.
Both native CI lanes retain their Debug quality step and add a Release Rust check
using the existing `--files Cargo.toml` selection after the Release build. This
explicit Release selection also runs for documentation-only changes and avoids
repeating native/CUDA analysis. No new selector or receipt protocol was added.

Fourteen focused regressions passed on each native platform with Python 3.13,
covering the exact four-preset command matrix, package/feature/filter preservation,
dependency rejection, Clippy failure propagation and CI Release admission/order.
Logs remain in
`out/validation/{linux-debug,windows-clang-debug}/clippy-profile-dispatch-2026-10-21/focused-regressions.log`.
Ruff and actionlint/ShellCheck passed for the changed Python and workflows; their
logs are in the Linux directory. Contextual naming, ownership and consumer review
found no issues. These are dispatcher regressions with captured commands; no
product Clippy, build or full qualification ran, and elapsed savings remain unmeasured.

Broad source/checkpoint discovery now excludes the repository prose defined in
the public validation cadence. Initial development observation also keeps those
prose changes out of pending execution work; existing pending failures remain.
The rule defaults unknown Markdown locations and non-README fixture payloads to
relevant. Explicitly declared inputs and build-graph dependencies are still hashed,
including Markdown. Quality's own prose/whitespace checks and their freshness
contract are unchanged. Older evidence remains bound to its original recorded
inputs and membership, so this fix does not silently upgrade old receipts.

Ten focused regressions passed on each native platform with Python 3.13. A tiny
`LANGUAGES NONE` fixture configured/prepared once and ran one registered CTest;
documentation edits, additions, deletions and a commit then reused its original
preparation and test receipts with command execution forbidden. Original owner,
receipt and artifact bytes remained unchanged. Collection of its deliberately
partial scope still rejected incomplete aggregate membership after freshness checks.
Negative controls retained source/configuration/runtime membership, explicit
Markdown consumption, pending-failure rejection and quality's own prose freshness.
The older-observation transition drops only newly inferred prose differences,
preserving existing pending failures. Ruff passed, and contextual naming/ownership
review found no remaining issues. Logs are retained at
`out/validation/{linux-debug,windows-clang-debug}/documentation-evidence-reuse-2026-10-21/focused-regressions.log`;
the logs identify retained checkpoint fixtures. No product build, full qualification
or evidence disposal occurred; product-file content hashes are unchanged.

Preparation now records its invocation before starting work and seals failure
recovery proof while its original healthy build-tree lease is still held. The
proof binds every new configure, Cargo clean and build attempt to terminal process,
result and output receipts. An incomplete attempt directory or unproved drainage
cannot be treated as a command that never started. Only the live owner can seal
a failure with no started commands. A retry validates the exact scope and proof
before replacing a pending owner or invalidating outputs, retains the original
pending bytes, and links them from fresh preparation admission. Failed attempts
remain failed, and existing successful evidence is not rewritten into a new pass.
Legacy pending markers and damaged or missing proof remain blocked. This change
does not take over a lease retained after hard owner death or unproved drainage.
Later same-stage attempts in the original campaign conservatively block recovery;
unrelated campaigns and stages do not change the sealed command membership.

Eight focused controls passed on each native platform with Python 3.13. Three
compiler-free `LANGUAGES NONE` fixtures exercise a real descendant writer,
configure/build command binding, no-command failure recovery, missing/changed/
substituted proof and publication rejection. Existing controls retain invalid-request
admission, unknown-drainage locks, enclosing leases, hard-owner-death blocking and
the bounded public selector. The first run passed seven controls but exposed an
invalid test assumption that Ninja forwards interrupted child stdout. The repaired
test uses a separate file-write witness, proves writes stop after drainage and
remain stopped through fresh preparation/CTest, and passes on both platforms.
Original baseline and command bytes are checked through recovery; the retained
pending marker is linked from the new receipt. Contextual naming/ownership review,
Ruff and whitespace checks passed. Logs and build-evidence fixtures remain under
`out/validation/{linux-debug,windows-clang-debug}/preparation-recovery-2026-10-21/`:
`focused-regressions.log` preserves the initial test failure and
`interruption-control-repair.log` records the single repaired control. Product-file
content hashes are unchanged; no product build, full qualification or evidence
cleanup ran. These checks establish bounded harness behavior, not elapsed savings.

General compilation/analysis/runtime identity separation remains **planned and unimplemented**.
Current owners may conservatively reject reuse or require rebuilding. Preserve
that disposition, state the missing proof and use the required owning path.
For prose-only edits, report stale or rejected receipts without launching broad
execution solely to restore their status.
Do not skip checks, filter environment hashes, edit receipts, clear pending owners
or manually relabel a scoped pass as complete qualification. Existing results are
not retroactively weakened or upgraded by this policy change.

Use qualified C++20, CUDA 13.2 C++20, native Rust/Cargo 1.98.1 and Python 3.13,
standard preset/Cargo directories and exact configured paths. Preserve generated
header guards, deadlines, supervision, Windows VCTIP grace/drainage and cross-native
device exclusion. Missing proof differs from stale artifacts; existing files and
matching mtimes prove neither freshness nor execution.

Bind actual source/generated inputs, ordered includes and semantic variants,
effective commands, tools, relevant environment, staged resources, fixtures,
executed artifacts and test membership. Unknown dependencies stay conservative.
No timestamp-only admission, cross-command digest cache or post-run seal of
arbitrary remaining bytes. P0-A and P0-B1 retain their existing controls.

## 6. Failure and retry discipline

Preserve the first diagnostic and classify the failure: compilation, numerical,
expectation, environment, orchestration or collection. Before retrying, state the
changed hypothesis and smallest discriminating check. One identical retry may
test a suspected transient when recorded explicitly. After a second unchanged
failure, diagnose before another broad attempt. A passing repeat establishes
non-reproduction, not a root-cause fix.

Do not raise timeouts, clean Cargo, delete outputs or restart every preset without
a concrete cause. Determine the affected source/fixture/ABI/context closure;
preserve unaffected completed work only when existing validators admit it. New
campaign names and unchanged production source alone are not reuse proof. Keep
failed, partial, undrained and unavailable gates explicit.

Keep original ON and OFF execution identities. A restored ON binary is not the
previous OFF artifact or automatically a newly tested ON binary. Required feature
transitions are legitimate work. Complete each required scope through its owner;
the current aggregate stays incomplete until its complete contract passes.

## 7. Retention and closeout

Retain compact accepted numerical inputs/outputs and provenance, approved numerical
decisions, unique reference captures, independent review/acceptance records and
one exact source identity. Preserve original failures establishing an exception
or unresolved defect. Public tests continue to use maintained public fixtures.

After closure and reference resolution, ordinary successful raw logs default to
**30 days plus the latest two successful runs of each gate**. Keep unique failure
evidence until resolution plus 30 days. A **100 MiB compact-text record budget per
closed slice** triggers a retention review; required scientific/reference assets
and selected executed artifacts are excluded. It never truncates necessary
diagnostics or causes automatic deletion.

Review reproducible scratch, duplicate source/executable copies, repeated logs and
obsolete launchers during ordinary closeout. Keep every payload required by
retained reusable evidence. Removing a receipt-bound payload makes that receipt
historical rather than reusable; never rewrite hashes to conceal removal. Check
active processes and consumers first. Archival moves reclaim no bytes. Active
caches, unresolved evidence and unique reference artifacts are not disposable
merely because they are large.

This update deletes nothing, schedules no cleanup service and authorizes no
relocation. Future disposal needs its concrete scope and applicable authorization.

## 8. Qualification cadence and retained contracts

Follow [Validation cadence](../tests/README.md#validation-cadence), the canonical
public rule. This section replaces the former unfiltered four-preset requirement
at every leaf or family handoff.

| Change or checkpoint | Required coverage |
| --- | --- |
| Documentation/status only | Contextual, link and whitespace review. Prose alone creates no product build/test requirement; section 5 describes the discovery boundary and older receipts' retained input requirements. |
| Mathematical leaf during editing | Touched formatting and compilation/type checks, focused mathematical cases and nearest real consumer. Exercise new compiled cases on both natives early. |
| Stable family cutover/deletion | Required four-native numerical cases; affected ABI, construction, failure, identity, cache and actual-caller checks; downstream GPU routes; scoped native/CUDA/Rust analysis; independent naming/ownership review. Unrelated whole-product/tooling matrices do not belong at every family handoff. |
| Changed ABI, export, feature isolation or ownership boundary | Both native Release testing-OFF probes once on the stable boundary, relevant ON boundary/lifetime checks and required ON cache restoration. Repeat when consumed inputs change, not merely for unrelated prose/test-loader edits. |
| Broad shared build/lifetime change; S4.E integration; S5 exit; S6.D cutover; S7/release | Full applicable four-preset product/boundary matrix, tooling once per native platform, corpus once, and required OFF/host/package gates. |
| Publication/merge | Existing required hosted checks on the actual candidate, plus local gates required above. Hosted host-only passes do not establish GPU, OFF or Resolve acceptance. |

Each handoff names affected tests and compilation contexts. Unknown dependencies
justify broader affected checks with the uncertainty stated. All regression tests
remain available at designated broader checkpoints. This changes cadence, not
implementation-slice scope or scientific acceptance.

Preserve reference revision/fixture provenance, supported resource semantics,
numerical matrices and approved exceptions, honest hashes/seeds, polarity and
operation order. Preserve C11/C++/Rust ABI facts, input/clearing/diagnostic/panic
contracts, Rust privacy/default/ownership and feature boundaries, affected semantic
compilation contexts, actual direct/print callers, exact CUDA context and epoch
ownership, frame finish/abort/retirement and relevant GPU lifetime behavior.
Film-Juicer grain/artifact recipes keep their separate authority. No CPU production
renderer, new scientific admission or arbitrary tolerance follows from this plan.

Independent contextual review and owner acceptance remain separate from tests.
Installed Resolve compatibility, renders/visual acceptance, hardware/context-loss,
live recovery and required profiling/toolchain evidence stay outstanding when
unavailable. This update establishes none of those acceptance results.

## 9. Prove improvement in ordinary work

Use the next **three comparably bounded, warm-cache migration tasks**, not a new
synthetic campaign. Record in the existing packet:

- Authorization-to-review-ready elapsed, separating owner wait and active work.
  Target median **at most 60 minutes** and **none above 90 minutes without a named
  exceptional blocker**. These are acceptance targets, not predicted savings.
- **Zero new private orchestration programs** and **at most 10 minutes verification
  setup/authoring**. Count scientific-case implementation separately and include
  it in total elapsed.
- Zero broad matrices during editing; one at a designated broader checkpoint.
  Record each repeated pass and its changed inputs or failure hypothesis.
- Each required Clippy context once per stable relevant context through the
  preset's matching-profile dispatch; retain both native platforms, profiles,
  packages and plugin feature forms at their required checkpoint.
- Admission/collection as aggregate work and interval union, targeting below 15%
  of task elapsed. Record receipts, evidence growth and available cached-input,
  uncached-input and output-token totals without creating another success gate.

At 60 minutes, reassess remaining work and stop adding harness scope. Keep required
work and unresolved gates explicit; never waive them to meet the target. If these
tasks still miss it, address the largest observed residual before authorizing
another optimization. Component wins, parallel command sums, reduced file counts
and bytes merely moved do not establish recovered developer time.
