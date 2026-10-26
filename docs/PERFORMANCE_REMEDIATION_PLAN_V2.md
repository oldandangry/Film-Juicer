# Film-Juicer performance remediation plan V2

**20 October 2026 — execution policy corrected after the migration audit.**
The cadence below replaces this plan's unconditional four-preset qualification
at every slice handoff. The canonical rule is
[Validation cadence](../tests/README.md#validation-cadence). The initial correction
changed documentation and execution policy only. Separately authorized harness
fixes implemented request admission and OFF evidence retention on 20 October,
then matching-profile Clippy, documentation discovery, bounded preparation
recovery and runtime-only preparation discovery on 21 October. Their focused validation is recorded
below; no product code changes, product builds or evidence disposal accompany them.

Preserve [the original plan](PERFORMANCE_REMEDIATION_PLAN.md) and the
[earlier V2 investigation](../out/validation/performance-review/2026-10-20/V2-reprioritization/before/PERFORMANCE_REMEDIATION_PLAN_V2.md).
Historical timings retain their original scope and qualification limits; they do
not establish whole-task savings.

Evidence links under `out/` or `.juicer-local/` in this plan and the original
investigation refer to retained local checkout files, which are not published
with the repository. The
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

Use one current handoff, one candidate identity and one final packet linking
original reports and finished receipts. Before execution, name the supported
target, test and quality commands with expected membership, the changed producer,
complete construction boundary, risky primitives, identities/seeds, expectation
readers, actual callers and staged runtime resources. Retain intermediate records
only for a concrete consumer; preserve owner-required before/after source and tool
hashes. Use registrations and source inspection; no new dependency database or
per-intermediate proof chain is needed.

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

During editing, map each selected assertion to the changed contract or actual
consumer and freeze complete membership before capture. Compile and run one
representative new native/C11 consumer before platform fan-out; then exercise the
required native platforms early. Run formatting and quality on changed
producer/helper translation units before cutover. Finish expectation adoption and
affected checks before broad qualification. Once the candidate is stable, run the
required broad header-consumer quality closure once, rather than repeating that
same closure both before and after cutover. For leaf Rust compiler checks, use the
existing named unittest method with its configured target, profile, features,
timeout and exact nonempty fixture/check membership; do not replace the full suite
at its designated broad checkpoint. At a broader checkpoint, use its built-in
quality and JUnit for the same domain evidence; do not add an identical full pass
merely to produce another report. Each partition command must complete under
supervision, drain its owned work and admit complete exact JUnit membership.
Passing or reuse requires the applicable results to succeed. Before another owner
is dispatched or a stable receipt is read, the enclosing owner must reach terminal
completion, drain operation work and release the shared normal lease in a healthy
state once the outer operation ends. Apply existing pending/failure admission
rules; preserve failed status, while a drained failure may permit a lawful repair
or fresh preparation. Retries
and admission within the existing lease remain with that owner. Partial console
output and inner build success are not completion. Do not signal a healthy
supervisor except through its documented cancel path.

Do not edit consumed inputs during their execution against that mutable checkout.
Documentation inspected by quality checks remains frozen for those checks. The
repository prose excluded by the public cadence no longer invalidates completed
build/test evidence. Build/device leases do not protect source from other writers. Queue unrelated
edits while valid work finishes; cancel only affected work through its supervisor.
Parallel execution must respect build/Cargo targets and shared GPU ownership.

## 4. Remediation order

The request-admission and OFF evidence-retention fixes below were separately
authorized and implemented in the working tree on 20 October 2026; matching-profile
Clippy, documentation discovery and bounded preparation recovery followed on 21 October.
The included-configure-input repair below was independently reviewed and authorized
for publication on 22 October. Other code
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
| 2 — runtime-only preparation discovery | Implemented: omit explicit registered runtime test files only when existing build/configuration ownership proves they are outside preparation. | Direct preparation inputs remain authoritative. Build, include, generated, Cargo, tool, copied/staged and artifact requirements stay fresh; unknown dependencies remain conservative. Runtime edits require fresh affected execution. Admission, refresh and collection share preparation-specific membership; original receipts retain their requirements. |
| 2 — included configure inputs | Implemented, independently reviewed and owner-authorized for publication: scan effective regeneration inputs regardless of suffix, limit bootstrap exceptions to exact reviewed paths/bytes, and check newly admitted inputs against pre-preparation observations. | Compiler-free stale-reuse reproductions are fixed; 22 focused controls pass on each native and four final independent Linux controls pass. Original receipt/explicit-input requirements and genuine runtime-only reuse remain intact. No demonstrated Film-Juicer product regression, broader qualification or speedup claim. |
| 3 — matching-profile Clippy | Implemented: run the matching profile once per preset while retaining both platforms, packages and plug-in feature forms across the aggregate. Both native CI lanes now explicitly select Release Rust after the Release build. | Four full preset invocations dispatch 12 Clippy analyses instead of 24: core, plugin production and plugin test-support in each matching context. Existing CI previously obtained both profiles through Debug-only calls; its complete Rust coverage remains 12 analyses across the two native lanes. Elapsed savings remain unmeasured. Boundary/privacy probes remain distinct. |
| 3 — resolved tool hashes within one observation | Implemented, independently reviewed and owner-accepted for publication in draft PR #74: `tool_snapshot` reuses a resolved output key's first successful hash within that call. | Both native diagnostics demonstrate removable duplication. Every spelling still resolves and passes file checks; admission, pre/post-command and terminal collection reads remain separate. Focused freshness controls pass; single-pair timings below do not establish whole-task savings. |
| 3 — necessary domain batching | Batch adjacent cases only within an already-needed domain extension, or after demonstrating payback from named remaining consumers. | Preserve complete inputs, individual results, process isolation where needed, membership and drainage. A standalone 0.5–1.5-day H1 patch for roughly two minutes per cohort is no longer the first code project. |
| 4 — ordinary closeout | Retire eligible scratch and obsolete helpers under section 7. | Storage/navigation benefit only; no compiler-speed claim. No disposal accompanies this update. |

General selector expansion, filesystem moves, analyzer caches, broad environment
normalization, new benchmarks and identity frameworks remain deferred. A P0-C
retry must first bind actual staged resource bytes and namespace membership.
Do not justify more machinery with invented future repetitions.

## 5. Current tooling constraints

The public tools own admission, execution and collection. `qualification --selection
product` still schedules preparation and quality; `collect-qualification` requires
complete four-profile product/boundary evidence, tooling once per native platform,
corpus once and both Release production gates. Focused evidence is not a full
aggregate pass. Existing timeouts, exact membership, source/tool/payload freshness,
process supervision and build/Cargo/device leases remain required.

Request shape is checked before ownership changes or reuse. OFF evidence is bound
to retained execution-time build inputs/configuration and can survive ON restoration;
current pending owners and newer failed attempts still block aggregate completion.
Older manifests are never backfilled. Clippy retains every native/profile/feature
combination. Prose exclusion does not exclude runtime fixtures, packaged inputs or
explicitly declared dependencies; older receipts keep their original requirements.

Pending-owner recovery needs a healthy normal lease and sealed drainage proof;
blocked/hard-death owners remain blocked. Runtime-only preparation excludes only
proven CTest test-file operands outside build/configuration/generated closures.
Opaque CMake file/process/evaluation consumers remain broad. Included-module,
actual-byte tool identity and original process-proof checks retain their existing
contracts. A fresh configure/no-op build is not a compiler performance claim.
Windows checkpoint scheduling reserves four CTest slots; Rust.Core uses two
workers/slots. Finite test bounds and race checks remain unchanged.

Successful synthetic Checkpoint/BuildEvidence workspaces are disposable once the
outer test consumes them and all work drains. Artifact output routing alone does
not request retention. Failed/interrupted/undrained work and explicit diagnostic
captures remain protected; section 7 governs closeout. These lifecycle controls
establish harness behavior, not product/GPU/OFX acceptance. Historical execution
results, review decisions and original limitations are in the
[consolidated account](../.juicer-local/docs/execution-history-through-2026-10-23.md).

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

Select canonical scientific/fixture authority, active work/review and required
reproductions, a small named set of completed executions deliberately retained
for fresh reuse, and compact historical acceptance. Historical acceptance needs
exact source identity or a reconstructible patch, final membership/outcomes,
review/decision and limits. Reusable executions need their complete original
receipt/log/process/payload closure; a summary cannot supply the collector's proof.
Keep a final producer only for each distinct needed context or OFF binary closure.
Each retained producer/artifact needs a named current consumer and an end condition
for retirement. Acceptance summaries and fresh-reuse evidence are separate: a
summary cannot admit reuse, while a fresh-reuse receipt must preserve the original
completed command records and required hashes.

Ordinary raw success and redundant resolved-failure diagnostics without a continuing
consumer use a **seven-day post-acceptance/resolution rollback window**. Keep real
unresolved failures, unique scientific exceptions and mandatory reproductions
according to their actual obligations. Superseded synthetic workspace contents and
verified redundant copies may go at closeout after ownership/consumer checks.
Expected-negative child receipts in a passing outer test are not unresolved defects.
Use **10 MiB as a review trigger** for ordinary closed-slice narrative, with
scientific data and required reusable closure separately justified. It is never
a truncation/deletion quota.

Review reproducible scratch, duplicate source/executable copies, repeated logs and
obsolete launchers during ordinary closeout. Keep every payload required by
retained reusable evidence. Retire a retained binary only after its named consumer
and required OFF/ON or package closure have reached their recorded end condition;
acceptance-summary retention does not imply fresh reuse eligibility. Receipt
existence, old links and internal references inside a retired group do not keep
every attempt live. Explicitly retire removed payload scopes from reuse in a new
disposition; never rewrite original hashes. Check active processes and consumers
first. Archival moves reclaim no bytes. Active caches, unresolved evidence and
unique reference artifacts are not disposable merely because they are large.

Successful synthetic Checkpoint/BuildEvidence fixtures default to disposal once
consumed and drained; artifact routing alone does not request retention.
`JUICER_TEST_RETAIN_FIXTURES=1` requests diagnostic capture. Failures, interruption,
blocked/unproved owners and named downstream diagnostics remain protected. Keep
outer test outcomes without preserving every generated project. Current navigation
links to one history instead of prepending the same completed account. Disposal
remains scoped and manual; no scheduler, collector, receipt schema or retention
service is added.

## 8. Qualification cadence and retained contracts

Follow [Validation cadence](../tests/README.md#validation-cadence), the canonical
public rule. This section replaces the former unfiltered four-preset requirement
at every leaf or family handoff.

| Change or checkpoint | Required coverage |
| --- | --- |
| Documentation/status only | Contextual, link and whitespace review. Prose alone creates no product build/test requirement; section 5 describes the discovery boundary and older receipts' retained input requirements. |
| Mathematical leaf during editing | Touched formatting and compilation/type checks, focused mathematical cases and nearest real consumer. Exercise touched C11/native fixture/helper compiles and scoped quality on both natives early, before freezing for broad qualification. |
| Stable family cutover/deletion | Required four-native numerical cases; affected ABI, construction, failure, identity, cache and actual-caller checks; downstream GPU routes; scoped native/CUDA/Rust analysis; independent naming/ownership review. For a staged cutover, replacement fixture consumers and caller evidence pass before producer deletion. As an alternative for a prospectively planned, reconstructible mathematical producer, independently capture old behavior and pass replacement core/ABI/fixture preparation before cutover; switch the caller and delete the old body in one candidate, then pass actual replacement-caller and downstream evidence before acceptance. This applies only to new handoffs; an existing handoff's stricter stage order remains unless explicitly amended. Preserve lifecycle, ownership and other deletion gates. Unrelated whole-product/tooling matrices do not belong at every family handoff. |
| Changed ABI, export, feature isolation or ownership boundary | Both native Release testing-OFF probes once on the stable boundary, relevant ON boundary/lifetime checks and required ON cache restoration. Repeat when consumed inputs change, not merely for unrelated prose/test-loader edits. |
| Broad shared build/lifetime change; S4.E integration; S5 exit; S6.D cutover; S7/release | Full applicable four-preset product/boundary matrix, tooling once per native platform, corpus once, and required OFF/host/package gates. |
| Publication/merge | Existing required hosted checks on the actual candidate, plus local gates required above. Hosted host-only passes do not establish GPU, OFF or Resolve acceptance. |

Each handoff names the supported target, test and quality commands with expected
membership. Unsupported selectors never trigger an automatic `--targets all`; an
explicit `all` needs a stated affected dependency or checkpoint reason. Unknown
dependencies justify broader affected checks with the uncertainty stated. All
regression tests remain available at designated broader checkpoints. This changes
cadence, not implementation-slice scope or scientific acceptance.

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

Use existing attempt, command and receipt records for these measurements; add no
collection step. State unavailable values as unknown instead of reconstructing them.

At 60 minutes, reassess remaining work and stop adding harness scope. Keep required
work and unresolved gates explicit; never waive them to meet the target. If these
tasks still miss it, address the largest observed residual before authorizing
another optimization. Component wins, parallel command sums, reduced file counts
and bytes merely moved do not establish recovered developer time.
