# Fotufilm Film reference

[Fotufilm](https://fotufilm.com)

Copyright 2026 MUAStudio Inc.

The [Apache License, Version 2.0](LICENSE.Fotufilm) governs the upstream
engine source used for the sampler and reference capture. The source pin is
[87a3f46af093a9e6bf76acf21aca83d3b9e2c600](https://github.com/DhaliwalX/fotufilm-engine/tree/87a3f46af093a9e6bf76acf21aca83d3b9e2c600).

film_reference.cu adapts FilmTiles.h and the placement hash from Random.h to
CUDA and the G1 carrier. capture_film_probe.swift extracts the scope of the
loop in FilmGrainTiles.swift needed to sample signed full-frame coordinates.
It reuses the upstream table builder, placement, footprint and blend helpers.

The analytic rectangle and constant-light cases are Film-Juicer test stimuli.
The frozen Portra 400 bank is checked in and packaged with Film-Juicer Grain
Preview. Its [bank notice](../../Resources/grain/portra-400/NOTICE.md) records
the profile and generator attribution and where to find the accompanying
licenses.

Gold 200 and Provia 100F remain local research banks. Their base Film-Juicer
profiles retain the Andrea Volpato / spektrafilm CC BY-SA 4.0 lineage. They
use stock-specific grain parameters from Fotufilm profiles, which carry the
MUAStudio/Fotufilm
[CC BY-SA 4.0 film-profile notice](https://github.com/DhaliwalX/fotufilm-engine/blob/87a3f46af093a9e6bf76acf21aca83d3b9e2c600/licenses/FILM-PROFILES.txt).
