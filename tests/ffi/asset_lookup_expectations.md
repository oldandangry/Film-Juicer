# Asset lookup expectations

These expectations were frozen before modifying the native lookup helpers at
Film-Juicer `bae2bb24e131527e89db29ce29b1fb5f04d2e773`. Finite lookup rows
characterize that native behavior; failure rows are the explicit lookup-safety
contract. They are analytical constants, not captured candidate output.

- Synthetic axis `[0, 2, 4]`, CMY rows `[0, 4, 8]`, `[4, 8, 12]`,
  `[8, 12, 16]`: query 1 gives `[2, 6, 10]`; clamped endpoints retain
  the first/last row. All operations are exactly representable binary floats.
- Scanner ascending `[0, 2, 4]` with values `[0, 4, 8]`, or reversed pairs:
  query 1 gives 2; query 3 gives 6; endpoints/outside queries give 0/8.
- Singleton axis `[2]` retains its one value for every non-NaN query.
- Synthetic duplicate axis `[0, 2, 2, 4]`, values `[0, 4, 8, 12]`:
  query 2 gives 8 (`upper_bound` selects the last duplicate); query 3 gives 10.
  Scanner's forward traversal selects the first duplicate: query 2 gives 4,
  query 3 gives 10. Reversing both Scanner arrays retains those results.
- Ordered `[-infinity, 0, +infinity]` with values `[1, 2, 3]` retains its
  endpoint values. At query -1, accessible interpolation evaluates
  infinity/infinity and returns an engaged NaN; at query 1 it returns 2.
  Lookup success guarantees sample access, not a finite computed result.
- Empty/mismatched shapes, invalid CMY channel and NaN query fail before
  search/indexing. Scanner also fails if its traversal cannot supply a bracket.

For the synthetic sampler's accepted nonempty, NaN-free, nondecreasing axis,
an interior query satisfies `axis.front() < query < axis.back()`. The first
element therefore compares <= query and the last compares > query. The range
is partitioned for `upper_bound`, which can return neither begin nor end.
Endpoint guards also cover singleton/all-equal axes and ordered infinities.
Thus the defensive invalid-bracket branch is unreachable under the construction
invariant. Tests must not break partitioning to exercise it.

Hanatos reference: spektrafilm
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc`,
`src/spektrafilm/utils/spectral_upsampling.py`,
`eval_erf4_spectral_bandpass` and adaptation normalization. The native formula
narrows authored f64 parameters to f32 and uses its existing f32 sqrt(2).
These reference cases use erf sign/limit identities and IEEE classification,
not NumPy output or a new numerical tolerance:

- `[380, -100, 780, 100]`: all 81 canonical samples are finite/nonnegative,
  with positive total response; at 380 the UV edge is 1/2 and IR edge lies
  strictly between 1/2 and 1, so the sample lies in (1/4, 1/2). At 780 the
  sample lies in (0, 1/4).
- `[380, 1e40, 780, 100]`: the first width narrows to +infinity; every finite
  UV argument is zero, so its edge is exactly 1/2. At 780 the IR edge is
  exactly 1/2, giving exactly 1/4; all samples are finite/nonnegative with
  positive total response. No parameter replacement is performed.
- `[380, 0, 780, 100]`: the sample at 380 is NaN because the UV argument is
  0/0; later canonical samples have UV edge 1 and finite IR response. This
  prepares the computed-sample rejection case for C8.

Production Hanatos parameter admission remains intact. Tests of these formula
cases do not claim relaxed profile-to-recipe acceptance; that belongs with C8.
Ordinary tests read no external/private source or reference environment.
