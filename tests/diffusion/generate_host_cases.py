from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_SCHEMA = "film-juicer.diffusion-reference.v1"
EXPECTED_FAMILIES = (
    "glimmerglass",
    "black_pro_mist",
    "pro_mist",
    "cinebloom",
)
EXPECTED_CHANNELS = ("R", "G", "B")
EXPECTED_COUNTS = {
    "strength_cases": 40,
    "family_cases": 4,
    "warmth_cases": 20,
    "advanced_cases": 16,
    "radius_cases": 16,
    "psf_cases": 80,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _cpp_float(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite fixture value: {value!r}")
    rendered = repr(number)
    if rendered == "-0.0":
        return "-0.0"
    if "." not in rendered and "e" not in rendered and "E" not in rendered:
        rendered += ".0"
    return rendered


def _cpp_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _cpp_array(values: Iterable[Any]) -> str:
    return "{{" + ", ".join(_cpp_float(value) for value in values) + "}}"


def _family_index(family: str) -> int:
    try:
        return EXPECTED_FAMILIES.index(family)
    except ValueError as error:
        raise ValueError(f"unknown fixture family: {family!r}") from error


def _authored_values(
    *,
    family: str,
    active: bool = True,
    strength: Any = 0.5,
    spatial_scale: Any = 1.0,
    halo_warmth: Any = 0.0,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    values = overrides or {}
    return [
        _cpp_bool(active),
        str(_family_index(family)),
        _cpp_float(strength),
        _cpp_float(spatial_scale),
        _cpp_float(halo_warmth),
        _cpp_float(values.get("core_intensity", 1.0)),
        _cpp_float(values.get("core_size", 1.0)),
        _cpp_float(values.get("halo_intensity", 1.0)),
        _cpp_float(values.get("halo_size", 1.0)),
        _cpp_float(values.get("bloom_intensity", 1.0)),
        _cpp_float(values.get("bloom_size", 1.0)),
    ]


def _authored_from_params(params: dict[str, Any]) -> list[str]:
    return _authored_values(
        family=str(params["filter_family"]),
        active=bool(params.get("active", True)),
        strength=params.get("strength", 0.5),
        spatial_scale=params.get("spatial_scale", 1.0),
        halo_warmth=params.get("halo_warmth", 0.0),
        overrides=params,
    )


def _validate_manifest(data: dict[str, Any]) -> None:
    if data.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(
            f"wrong fixture schema: {data.get('schema')!r}; expected {EXPECTED_SCHEMA!r}"
        )
    if tuple(data.get("family_order", ())) != EXPECTED_FAMILIES:
        raise ValueError(f"wrong family order: {data.get('family_order')!r}")
    if tuple(data.get("semantic_channel_order", ())) != EXPECTED_CHANNELS:
        raise ValueError(
            f"wrong semantic channel order: {data.get('semantic_channel_order')!r}"
        )
    for key, expected in EXPECTED_COUNTS.items():
        actual = len(data.get(key, ()))
        if actual != expected:
            raise ValueError(f"wrong {key} count: {actual}; expected {expected}")


def _emit_prelude(
    manifest_sha: str,
    arrays_sha: str,
) -> list[str]:
    return [
        "// Generated from the accepted spektrafilm fixture. Do not edit.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace DiffusionHostReference {",
        "",
        f'inline constexpr char kManifestSha256[] = "{manifest_sha}";',
        f'inline constexpr char kArraysSha256[] = "{arrays_sha}";',
        f'inline constexpr char kSchema[] = "{EXPECTED_SCHEMA}";',
        "inline constexpr std::array<std::string_view, 4> kFamilyNames{{",
        "    \"glimmerglass\", \"black_pro_mist\", \"pro_mist\", \"cinebloom\"}};",
        "inline constexpr std::array<std::string_view, 3> kSemanticChannels{{\"R\", \"G\", \"B\"}};",
        "",
        "struct AuthoredInput {",
        "    bool active;",
        "    std::uint8_t familyIndex;",
        "    double strength;",
        "    double spatialScale;",
        "    double haloWarmth;",
        "    double coreIntensity;",
        "    double coreSize;",
        "    double haloIntensity;",
        "    double haloSize;",
        "    double bloomIntensity;",
        "    double bloomSize;",
        "};",
        "",
        "struct StrengthCase {",
        "    std::string_view name;",
        "    AuthoredInput authored;",
        "    double scatterFraction;",
        "};",
        "",
        "struct FamilyCase {",
        "    std::string_view name;",
        "    std::uint8_t familyIndex;",
        "    std::array<double, 3> groupWeights;",
        "    std::array<double, 3> groupCentersUm;",
        "    double baseWarmth;",
        "    std::array<double, 2> coreLambdasUm;",
        "    std::array<double, 2> coreWeights;",
        "    std::array<double, 3> haloLambdasUm;",
        "    std::array<double, 3> haloWeights;",
        "    std::array<double, 4> bloomLambdasUm;",
        "    std::array<double, 4> bloomWeights;",
        "};",
        "",
        "struct WarmthCase {",
        "    std::string_view name;",
        "    AuthoredInput authored;",
        "    double effectiveWarmth;",
        "    std::array<std::array<double, 3>, 3> channelWeightsRgb;",
        "};",
        "",
        "struct AdvancedCase {",
        "    std::string_view name;",
        "    AuthoredInput authored;",
        "    double familyBaseWarmth;",
        "    std::array<double, 3> groupWeights;",
        "    std::array<double, 3> groupCentersUm;",
        "    std::array<double, 9> lambdasUm;",
        "    std::array<double, 2> coreWeights;",
        "    std::array<double, 3> haloWeights;",
        "    std::array<double, 4> bloomWeights;",
        "};",
        "",
        "struct RadiusCase {",
        "    std::string_view name;",
        "    AuthoredInput authored;",
        "    int height;",
        "    int width;",
        "    double pixelSizeUm;",
        "    int radiusPixels;",
        "};",
        "",
        "struct PsfCase {",
        "    std::string_view name;",
        "    AuthoredInput authored;",
        "    int height;",
        "    int width;",
        "    double pixelSizeUm;",
        "    std::size_t valueOffset;",
        "};",
        "",
    ]


def _emit_strength_cases(data: dict[str, Any]) -> list[str]:
    rows = data["strength_cases"]
    out = [
        f"inline constexpr std::array<StrengthCase, {len(rows)}> kStrengthCases{{{{"
    ]
    for row in rows:
        family = str(row["family"])
        name = f"{family}:strength={_cpp_float(row['strength'])}"
        authored = _authored_values(family=family, strength=row["strength"])
        out.append(
            "    StrengthCase{" + ", ".join(
                [
                    _cpp_string(name),
                    "AuthoredInput{" + ", ".join(authored) + "}",
                    _cpp_float(row["scatter_fraction"]),
                ]
            ) + "},"
        )
    out.extend(["}};", ""])
    return out


def _emit_family_cases(data: dict[str, Any]) -> list[str]:
    rows = data["family_cases"]
    out = [f"inline constexpr std::array<FamilyCase, {len(rows)}> kFamilyCases{{{{"]
    for row in rows:
        core = row["core"]
        halo = row["halo"]
        bloom = row["bloom"]
        out.append(
            "    FamilyCase{" + ", ".join(
                [
                    _cpp_string(str(row["family"])),
                    str(_family_index(str(row["family"]))),
                    _cpp_array((row["w_c"], row["w_h"], row["w_b"])),
                    _cpp_array((core["lambda_um"], halo["lambda_um"], bloom["lambda_um"])),
                    _cpp_float(row["halo_warmth_base"]),
                    _cpp_array(core["expanded_lambdas_um"]),
                    _cpp_array(core["expanded_weights"]),
                    _cpp_array(halo["expanded_lambdas_um"]),
                    _cpp_array(halo["expanded_weights"]),
                    _cpp_array(bloom["expanded_lambdas_um"]),
                    _cpp_array(bloom["expanded_weights"]),
                ]
            ) + "},"
        )
    out.extend(["}};", ""])
    return out


def _emit_warmth_cases(data: dict[str, Any]) -> list[str]:
    rows = data["warmth_cases"]
    out = [f"inline constexpr std::array<WarmthCase, {len(rows)}> kWarmthCases{{{{"]
    for index, row in enumerate(rows):
        family = str(row["family"])
        authored = _authored_values(
            family=family,
            halo_warmth=row["halo_warmth"],
        )
        channel_rows = ", ".join(
            _cpp_array(channel) for channel in row["channel_weights_rgb"]
        )
        out.append(
            "    WarmthCase{" + ", ".join(
                [
                    _cpp_string(f"{index}:{family}:warmth={_cpp_float(row['halo_warmth'])}"),
                    "AuthoredInput{" + ", ".join(authored) + "}",
                    _cpp_float(row["effective_clamped"]),
                    "{{" + channel_rows + "}}",
                ]
            ) + "},"
        )
    out.extend(["}};", ""])
    return out


def _emit_advanced_cases(data: dict[str, Any]) -> list[str]:
    rows = data["advanced_cases"]
    base_warmth = {
        str(row["family"]): float(row["halo_warmth_base"])
        for row in data["family_cases"]
    }
    out = [
        f"inline constexpr std::array<AdvancedCase, {len(rows)}> kAdvancedCases{{{{"
    ]
    for index, row in enumerate(rows):
        family = str(row["family"])
        overrides = dict(row["overrides"])
        authored = _authored_values(family=family, overrides=overrides)
        core = row["core"]
        halo = row["halo"]
        bloom = row["bloom"]
        lambdas = (
            list(core["expanded_lambdas_um"])
            + list(halo["expanded_lambdas_um"])
            + list(bloom["expanded_lambdas_um"])
        )
        out.append(
            "    AdvancedCase{" + ", ".join(
                [
                    _cpp_string(f"{index}:{family}:{row['name']}"),
                    "AuthoredInput{" + ", ".join(authored) + "}",
                    _cpp_float(base_warmth[family]),
                    _cpp_array((row["resolved_w_c"], row["resolved_w_h"], row["resolved_w_b"])),
                    _cpp_array((core["lambda_um"], halo["lambda_um"], bloom["lambda_um"])),
                    _cpp_array(lambdas),
                    _cpp_array(core["expanded_weights"]),
                    _cpp_array(halo["expanded_weights"]),
                    _cpp_array(bloom["expanded_weights"]),
                ]
            ) + "},"
        )
    out.extend(["}};", ""])
    return out


def _emit_radius_cases(data: dict[str, Any]) -> list[str]:
    rows = data["radius_cases"]
    out = [f"inline constexpr std::array<RadiusCase, {len(rows)}> kRadiusCases{{{{"]
    for index, row in enumerate(rows):
        height, width = (int(value) for value in row["extent"])
        authored = _authored_from_params(dict(row["params"]))
        out.append(
            "    RadiusCase{" + ", ".join(
                [
                    _cpp_string(f"{index}:{row['extent_class']}:{row['family']}"),
                    "AuthoredInput{" + ", ".join(authored) + "}",
                    str(height),
                    str(width),
                    _cpp_float(row["pixel_size_um"]),
                    str(int(row["psf_radius"])),
                ]
            ) + "},"
        )
    out.extend(["}};", ""])
    return out


def _emit_psf_cases(
    data: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
) -> list[str]:
    rows = data["psf_cases"]
    cases: list[str] = []
    flattened: list[float] = []
    for index, row in enumerate(rows):
        name = str(row["array"])
        if name not in arrays.files:
            raise ValueError(f"missing PSF array: {name}")
        values = arrays[name]
        height, width = (int(value) for value in row["extent"])
        expected_shape = (height, width, 3)
        if values.shape != expected_shape or values.dtype != np.dtype("float64"):
            raise ValueError(
                f"bad PSF array {name}: shape={values.shape} dtype={values.dtype}; "
                f"expected shape={expected_shape} dtype=float64"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite PSF array: {name}")
        offset = len(flattened)
        flattened.extend(float(value) for value in values.ravel(order="C"))
        authored = _authored_values(
            family=str(row["family"]),
            spatial_scale=row["spatial_scale"],
            halo_warmth=row["halo_warmth"],
            overrides=dict(row["overrides"]),
        )
        cases.append(
            "    PsfCase{" + ", ".join(
                [
                    _cpp_string(f"{index}:{name}"),
                    "AuthoredInput{" + ", ".join(authored) + "}",
                    str(height),
                    str(width),
                    _cpp_float(row["pixel_size_um"]),
                    str(offset),
                ]
            ) + "},"
        )

    expected_value_count = sum(
        int(row["extent"][0]) * int(row["extent"][1]) * 3 for row in rows
    )
    if len(flattened) != expected_value_count:
        raise ValueError(
            f"wrong flattened PSF value count: {len(flattened)}; "
            f"expected {expected_value_count}"
        )

    out = [f"inline constexpr std::array<PsfCase, {len(rows)}> kPsfCases{{{{"]
    out.extend(cases)
    out.extend(["}};", ""])
    out.append(
        f"inline constexpr std::array<double, {len(flattened)}> kPsfValues{{{{"
    )
    values_per_line = 6
    for start in range(0, len(flattened), values_per_line):
        rendered = ", ".join(
            _cpp_float(value) for value in flattened[start : start + values_per_line]
        )
        out.append(f"    {rendered},")
    out.extend(["}};", ""])
    return out


def generate(manifest_path: Path, arrays_path: Path, output_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(data)
    lines = _emit_prelude(_sha256(manifest_path), _sha256(arrays_path))
    lines.extend(_emit_strength_cases(data))
    lines.extend(_emit_family_cases(data))
    lines.extend(_emit_warmth_cases(data))
    lines.extend(_emit_advanced_cases(data))
    lines.extend(_emit_radius_cases(data))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        lines.extend(_emit_psf_cases(data, arrays))
    lines.extend(["} // namespace DiffusionHostReference", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate local C++ host-behavior cases from the accepted fixture."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.manifest, args.arrays, args.output)
    print(f"HOST_BEHAVIOR_CASES=PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
