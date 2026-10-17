# Capture-film DIR and density bounds

These immutable Film-Juicer characterizations were captured from the native
producers at `aef712df8abf620eba0247741043a00e39d3bb52`, before translation.
`manifest.json` retains the four native configurations, input/source identities,
compiler command receipts and original capture hashes. Debug and Release payloads
are shared only after complete JSON equality on the same native platform.

Each platform has 64 complete products: four stocks, direct/print selection,
gamma 1/0.85, and disabled/zero-amount/stock/asymmetric-custom DIR. Every fixed
f32 value and complete RGB compensated axis is stored as raw unsigned bit words.
CMY bounds and all family/snapshot/final identities compare exactly.

The 42 F1 records use real completed profiles/Development for duplicate,
signed-zero, narrowing-equal, ordered-infinite, wide-finite and singleton axes.
The 40 focused records retain diffusion/unused-K/zero-matrix/zero-receiver and
minimum edge cases. Synthetic changes alter only `data.log_exposure`; original
spektrafilm attribution and license remain in the resources. Duplicate/singleton
inputs reuse the existing `../fixtures/synthetic` resources without changes.

These are product characterizations, not an external-reference pixel oracle.
Nonlinear DIR retains photographic reference
`28bf883e1672e884307edc75852549376e13644e`; the newer upstream checkout does not
redefine these expectations. Ordinary tests never regenerate fixtures.
