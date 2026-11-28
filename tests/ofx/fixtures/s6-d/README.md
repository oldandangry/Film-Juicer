# S6.D OFX references

These records preserve observations from the independently retained C++ OFX
producer at revision `d4bc2135d02ed88c91403040c4429a4fdb802a2b`. They are
characterization witnesses for the S6.D callback cutover. They do not replace
spektrafilm acceptance fixtures or authorize regenerating scientific assets.

`context.json`, `instance.json`, `actions.json` and `render.json` use shared
v2 envelopes. Their captured payloads were identical across `linux-debug`,
`linux-release`, `windows-clang-debug`, and `windows-clang-release`; their
envelopes name exactly those four native presets. The retained v1 readers
continue to accept each original per-preset record. The shared files preserve
the original ordered payloads and numeric words.

Action records retain 1,034 compared observations within 1,059 action rows.
Render records retain 85 compared observations within 90 rows; the five
out-of-domain time cases remain candidate-only. Context retains two clip and
128 parameter/group definitions. Instance retains the full ordered reference
and its 113 parameter/two clip census. The capture identity and C++ print-route
GPU records lost their readers during cleanup and remain at the
[archive tag](https://github.com/oldandangry/Film-Juicer/tree/archive/spectral-cutover-2026-11/tests/ofx/fixtures/s6-d).

`manifest.json` lists the SHA-256 and byte size of each adopted file and the
SHA-256 of each original capture under
`out/validation/<preset>/s6-d-reference/`. The original captures remain at those
paths. Regenerating or updating a record requires its own reference review; the
ordinary suite reads these records and compares real callbacks against them.
