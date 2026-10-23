# Scanner color construction characterization

`native.json` contains 16 complete unchanged-native builder results captured
independently in Linux/Windows Debug/Release before replacement. All field words
and identities match across those captures; `manifest.json` retains each original
capture/producer identity and maps the four contracts to one physical payload.
`case-spec.json` was pinned before capture. `checked.json` specifies the raw
production failure envelopes independently. Ordinary tests never regenerate them.

Matrices are row-major f32 words. Original scanner XYZ words are copied verbatim;
Params retain the original byte and force successful input-space true. Encoding
contains schema/kind, six coefficient words and nine DWG matrix words, even with
CCTF off. Hashes are ordered native FNV identities. Every integer/finite/signed-zero
word is compared exactly; no new tolerance or numerical exception applies.

The preserved output-color manifest/native payload supplies 80 scanner rows and
all nine complete catalogue entries. CAT02 logical preset names resolve through
the existing deduplicated loader. Six original leaf/threshold rows are additionally
consumed: threshold-row-1-offset--1/0/1 and source-special-0-3/4/5-xyz-0, with
catalogue entry 5 as destination. Their independent matrices and supplied white
words are reused without schema or payload regeneration. The original CAT02
preparation reader retains all 144 scanner rows, complete products and profile
witnesses and now separately checks every encoding field from the catalogue.

`Ffi.Host.ScannerColor` exercises production and test facade, full values,
compiled C11/C++/Rust layouts/signatures, clearing/canaries, exact failures,
constructor-stage faults, cold first access, held copies and projection integrity.
The independent scanner pixel oracle remains unchanged.
