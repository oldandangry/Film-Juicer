# Scanner correction characterization

These are Film-Juicer native characterizations captured independently in all four
native build profiles before translating the direct and print builders. The
manifest retains source, executable and compiler-context identities. Each preset
contains 150 bounded literals and 48 actual route/state/resource products, their
exact input words, returned descriptor words/identity and ordinary diagnostic.
Float words are unsigned IEEE-754 bit patterns; channels are C/M/Y and spectra
use the fixed 81-position axis. Tests require exact stored words and hashes.

The separate trace payloads retain actual native diagnostic text: Debug emits
enabled events and Release compiles them out. Native exceptions for active print
route bytes 4 and 255 retain `exception`, a null native bool, the actual reset
descriptor and empty diagnostic/trace. They are never ordinary-false captures.
`checked-expectations.json` separately defines the authorized checked rejection:
PREPARATION_FAILURE / PRINT_INPUTS, with a cleared raw result/trace. Inactive
literal bytes retain their byte and canonical no-op values.

The core scanner tests load the shared rows and the selected row replacements.
The retired native and C11 witnesses and their commands are documented in the
[tagged scanner correction guide](https://github.com/oldandangry/Film-Juicer/tree/archive/spectral-cutover-2026-11/tests/scanner/correction-fixtures).

The owner-approved diagnostic rule permits NaN-class equivalence only in float
tokens of rejected correction events. Tests report each differing native/actual
NaN spelling and retain raw traces; every other token, failure envelope and
successful word/hash stays exact.
