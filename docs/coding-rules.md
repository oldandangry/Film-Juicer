# Coding Rules

This repo uses lightweight local guardrails to keep touched code readable before it is copied into `C:\Dev\Juicer\` for the real build and Git workflow.

## Scope

- Format touched C/C++/CUDA files with `scripts/format_touched_files.sh`.
- Run `scripts/check_touched_files.sh` before handing off larger changes.
- Prefer passing explicit file paths to these scripts. Do not assume this working copy's Git state matches the real repo.
- Do not mass-reformat unrelated files.

## Style

- Follow the repo `.clang-format`. It is intentionally conservative: no include sorting, no comment reflow, and no line-length wrapping.
- Match existing naming: `PascalCase` for types, `snake_case` for free functions, `k` prefixes for spec-style constants.
- Keep includes grouped as standard, third-party, then project headers.
- Prefer file-local helpers in anonymous namespaces over adding new cross-cutting utility headers.
- Keep headers lean. Move implementation detail into `.cpp` files unless header-only code is required for parity, templates, or inlining.
- Write comments only for invariants, parity constraints, and non-obvious intent.

## Validation

- `JUICER_TESTS` is deprecated and must not be added or revived.
- Validate changes through the real-project workflow: copy to `C:\Dev\Juicer\`, build there, and run targeted manual checks in the host.
- For risky changes, include a short validation note in the handoff describing what was checked.
