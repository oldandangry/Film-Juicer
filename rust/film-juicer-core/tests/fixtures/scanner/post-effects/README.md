# Scanner post-effects native characterization

400 whole-constructor observations were captured independently in Linux/Windows
Debug/Release before replacement, from the unchanged Scanner.cpp at 67f28a5.
All four captures agree exactly. One shared payload uses four distinct contexts
through the existing fixture loaders; provenance retains each actual producer
executable, sources/configuration and original capture under out/validation.

Input/output floats are IEEE f32 words in glare percent, roughness, glare sigma,
lens sigma, unsharp sigma, amount order. Radii are glare/lens/unsharp. Every field,
schema, exact hash, diagnostic, ordinary attempted failure and native route
exception is asserted. These are Film-Juicer migration characterizations, with
no new external scientific authority or tolerance. Ordinary tests never generate
expectations. RadiusRange cases and artificial faults follow the separately
reviewed representation contract; no native capture executes undefined casts.
