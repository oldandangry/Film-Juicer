# Portra normalization fixtures

These five 102-word little-endian Float32 tables are unchanged captures from
`out/validation/grain-g2-2026-11-16/tables-portra` for the frozen G1 Portra
CDF3 bank. Each contains three records, with 17 footprint mean-density words
followed by 16 adjacent correlations and a neutral padding word per record.
They were captured with the pinned Fotufilm implementation revision
`87a3f46af093a9e6bf76acf21aca83d3b9e2c600`; see
[attribution and provenance](../../../third_party/fotufilm/ENGINE-NOTICE.md). The widths cover both
supported limits, UHD default, HD default and the HD interior Size example.
The ordinary host witness reads these maintained copies, not the private capture.
