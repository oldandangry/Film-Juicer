# Scanner correction characterization

These are Film-Juicer native characterizations captured independently in all four
native build profiles before translating the direct and print builders. The
manifest retains source, executable and compiler-context identities. Each payload
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

`Ffi.Host.ScannerCorrection`, `Scanner.Host.CorrectionProducts`, core Rust tests
and the testing-OFF C11 probe consume these files. No ordinary test regenerates
them or reads a private workbench file. Native capture mode was retired at caller
cutover; historical native artifacts and command receipts are retained in the
slice's private review packet. This fixture is not an independent pixel oracle
or external spektrafilm render acceptance.

The owner-approved diagnostic rule permits NaN-class equivalence only in float
tokens of rejected correction events. Tests report each differing native/actual
NaN spelling and retain raw traces; every other token, failure envelope and
successful word/hash stays exact. Frozen payloads are unchanged.
