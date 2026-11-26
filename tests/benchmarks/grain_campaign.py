"""Frozen G2 experiment inputs, calibration and held-out comparison."""

import json
import math

import grain_analysis as analysis
import numpy as np

PORTRA = "kodak_portra_400"
GOLD = "kodak_gold_200"
PROVIA = "fujifilm_provia_100f"
SIZES = (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
UNIT_AMOUNTS = (1.0, 1.0, 1.0)
SESSION_SEED = 17
CLIP_TOKEN = 23


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def zero_exposure(response):
    """Sample the completed CDF3 totals using the capture stage's LUT interpolation."""
    data = read_json(response)["data"]
    axis = np.asarray(data["log_exposure"], dtype=np.float32)
    records = np.asarray(data["density_curves"], dtype=np.float32)
    right = int(np.searchsorted(axis, 0.0))
    if axis.shape != (256,) or records.shape != (256, 3) or not 0 < right < len(axis):
        raise ValueError(f"invalid CDF3 response {response}")
    left = right - 1
    fraction = -axis[left] / (axis[right] - axis[left])
    return (records[left] + fraction * (records[right] - records[left])).tolist()


def freeze(root, shared, manifest):
    original = root / "out/validation/grain-g1-2026-11-16"
    linux = root / "out/validation/linux-release/results"
    source_paths = {
        PORTRA: (original / "bank-cdf3", shared / "tables-portra", linux / "grain-g1-cdf3-response/response.log"),
        GOLD: (shared / "bank-gold", shared / "bank-gold", linux / "grain-g2-gold-response/response.log"),
        PROVIA: (shared / "bank-provia", shared / "bank-provia", linux / "grain-g2-provia-response/response.log"),
    }
    stocks = {
        key: {
            "bank": bank.relative_to(root).as_posix(),
            "tables": tables.relative_to(root).as_posix(),
            "response": response.relative_to(root).as_posix(),
            "density": zero_exposure(response),
            "route": 2 if key == PROVIA else 0,
        }
        for key, (bank, tables, response) in source_paths.items()
    }
    rows = [
        {"id": f"portra-{geometry}-{look}", "stock": PORTRA, "geometry": geometry, "look": look}
        for geometry in ("hd-36", "uhd-36")
        for look in ("default", "expensive")
    ]
    rows.extend(
        {"id": name, "stock": stock, "geometry": geometry, "look": look}
        for name, stock, geometry, look in (
            ("gold-uhd-36-default", GOLD, "uhd-36", "default"),
            ("provia-uhd-36-default", PROVIA, "uhd-36", "default"),
            ("portra-uhd-12-default", PORTRA, "uhd-12", "default"),
            ("portra-uhd-36-simple", PORTRA, "uhd-36", "simple"),
        )
    )
    quality = {
        "density": {"stocks": list(stocks), "frames": list(range(16, 32)), "apertures_um": [12, 24, 48, 96]},
        "scale": {"stock": PORTRA, "side_um": 512, "pitches_um": [0.25, 0.5, 1, 2, 4, 8, 16], "time": 19},
        "animation": {"stock": PORTRA, "fps": 24, "frames": 240, "clip_token": 97, "session_seed": 167},
        "preview": {"conversion": "unit exposure CMY scalar transmission to RGB, sRGB encoding", "order_seed": 26391},
    }
    write_json(
        manifest,
        {
            "schema": "grain-g2-experiment-v1",
            "stocks": stocks,
            "rows": rows,
            "sizes": SIZES,
            "session_seed": SESSION_SEED,
            "clip_token": CLIP_TOKEN,
            "calibration": list(range(16)),
            "evaluation": list(range(16, 32)),
            "fractional": [time + 0.5 for time in range(16, 32)],
            "quality": quality,
        },
    )


def request_for(experiment, row, engine, times, *, size=1.0, amounts=UNIT_AMOUNTS):
    geometry, film_format = row["geometry"].split("-")
    width, height = (1920, 1080) if geometry == "hd" else (3840, 2160)
    pitch = float(np.float32(1000 * int(film_format) / width))
    halo = math.ceil(48 / (2 * pitch) - 0.5)
    stock = experiment["stocks"][row["stock"]]
    request = {
        "engine": engine,
        "stock": row["stock"],
        "look": row["look"],
        "route": stock["route"],
        "pitch_um": pitch,
        "long_edge_mm": float(film_format),
        "width": width,
        "height": height,
        "analysis_width": 1024,
        "analysis_height": 1024,
        "crop_width": 1024 + 2 * halo,
        "crop_height": 1024 + 2 * halo,
        "fps": 24,
        "session_seed": experiment["session_seed"],
        "clip_token": experiment["clip_token"],
        "requests": [{"time": time, "density": stock["density"]} for time in times],
    }
    if engine == "film":
        request.update(bank=stock["bank"], tables=stock["tables"], size=size, amounts=list(amounts))
    return request


def frames_in(path):
    manifest = read_json(path / "manifest.json")
    shape = (len(manifest["requests"]), 3, manifest["crop_height"], manifest["crop_width"])
    if (path / "frames.f32").stat().st_size != math.prod(shape) * 4 or len(manifest["resolved"]) != shape[0]:
        raise ValueError(f"incomplete frame output {path}")
    frames = np.memmap(path / "frames.f32", mode="r", dtype="<f4", shape=shape)
    h, w = manifest["analysis_height"], manifest["analysis_width"]
    window = ((shape[2] - h) // 2, (shape[3] - w) // 2, h, w)
    return manifest, frames, window


def statistics(path):
    metadata, frames, window = frames_in(path)
    pitch = metadata["pitch_um"]
    results = [analysis.analyze_match_frame(frame, pitch, window) for frame in frames]
    return analysis.summarize_frames(results, pitch)


def tune_amounts(path, unit, target):
    metadata, frames, window = frames_in(path)
    pitch = metadata["pitch_um"]
    density = metadata["requests"][0]["density"]
    target_strength = np.asarray(target["strength"])
    amounts = np.ones(3)
    strength = np.asarray(unit["strength"])
    history = []
    for _ in range(4):
        updated = np.clip(amounts * target_strength / strength, 0.125, 8.0)
        if np.array_equal(amounts, updated):
            break
        amounts = updated
        per_seed = np.asarray(
            [analysis.strength_for_amounts(frame, density, pitch, window, amounts) for frame in frames]
        )
        strength = np.sqrt(np.mean(per_seed**2, axis=0))
        history.append(
            {"amounts": amounts.tolist(), "strength": strength.tolist(), "per_seed_strength": per_seed.tolist()}
        )
        if np.max(np.abs(strength / target_strength - 1)) < 0.001:
            break
    return amounts.tolist(), history


def calibrate(experiment, render, output, selection):
    selected = [row for row in experiment["rows"] if not selection or row["id"] in selection]
    if not selected or (selection and set(selection) != {row["id"] for row in selected}):
        raise ValueError("unknown calibration row")
    if set(experiment["calibration"]) & set(experiment["evaluation"]):
        raise ValueError("overlapping calibration and evaluation frames")
    if "chroma_fraction" in experiment:
        from grain_chroma import recheck

        recheck(experiment, render, output, selected)
        return
    grids = {}
    for row in selected:
        print(f"Calibrating {row['id']}", flush=True)
        name = row["id"]
        report = {"row": row, "disposition": "unmatched", "sizes": [], "attempts": []}

        def capture(arm, phase, times, *, name=name, row=row, **controls):
            path = render(f"{name}-{phase}-{arm}", request_for(experiment, row, arm, times, **controls))
            summary = statistics(path)
            write_json(path / "statistics.json", summary)
            return path, summary

        try:
            times = experiment["calibration"]
            current_path, current = capture("current", "calibration", times)
            report["current_calibration"] = str(current_path)
            key = (row["stock"], row["geometry"])
            if key not in grids:
                grid = []
                for index, size in enumerate(experiment["sizes"]):
                    try:
                        path, summary = capture("film", f"size-{index}", times, size=size)
                    except ValueError as error:
                        grid.append((size, None, str(error)))
                    else:
                        grid.append((size, path, summary))
                grids[key] = grid
            ranked = []
            for size, path, summary in grids[key]:
                if path is None:
                    report["sizes"].append({"size": size, "error": summary})
                    continue
                residual = analysis.match_residuals(summary, current)
                report["sizes"].append({"size": size, "capture": str(path), "unit_residuals": residual})
                ranked.append((abs(residual["length"]), size, path, summary))
            if not ranked:
                raise ValueError("no valid size observations")
            ranked.sort()
            for index, (error, size, path, unit) in enumerate(ranked):
                if index and error > 0.05:
                    break
                amounts, history = tune_amounts(path, unit, current)
                fitted_path, fitted = capture("film", f"fit-{index}", times, size=size, amounts=amounts)
                residual = analysis.match_residuals(fitted, current)
                fit = {
                    "size": size,
                    "amounts": amounts,
                    "history": history,
                    "capture": str(fitted_path),
                    "calibration": residual,
                }
                report["attempts"].append(fit)
                if residual["matched"]:
                    report["frozen"] = fit
                    write_json(output / f"{name}-frozen.json", fit)
                    break
            evaluation = experiment["evaluation"]
            evaluation_path, evaluation_current = capture("current", "evaluation", evaluation)
            reference_path, reference_film = capture("film", "reference", evaluation)
            report["current_evaluation"] = str(evaluation_path)
            report["reference_comparison"] = {
                "capture": str(reference_path),
                "residuals": analysis.match_residuals(reference_film, evaluation_current),
            }
            if "frozen" in report:
                controls = {key: report["frozen"][key] for key in ("size", "amounts")}
                path, fitted = capture("film", "evaluation", evaluation, **controls)
                report["evaluation_capture"] = str(path)
                report["evaluation"] = analysis.match_residuals(fitted, evaluation_current)
                if report["evaluation"]["matched"]:
                    report["disposition"] = "qualified"
                half = experiment["fractional"]
                half_current_path, half_current = capture("current", "fractional", half)
                half_path, half_film = capture("film", "fractional", half, **controls)
                report["fractional"] = {
                    "current_capture": str(half_current_path),
                    "film_capture": str(half_path),
                    "residuals": analysis.match_residuals(half_film, half_current),
                }
            elif report["attempts"]:
                report["reason"] = "size or strength outside the declared calibration tolerances"

            if row["stock"] == PORTRA and row["look"] in {"default", "expensive"}:
                direct = request_for(experiment, row, "current", [0, 16, 16.5])
                print_route = {**direct, "route": 1}
                direct_path = render(f"{name}-route-direct", direct)
                print_path = render(f"{name}-route-print", print_route)
                direct_info, direct_frames, _ = frames_in(direct_path)
                print_info, print_frames, _ = frames_in(print_path)
                recipe_equal = all(
                    {k: v for k, v in a.items() if k != "route"} == {k: v for k, v in b.items() if k != "route"}
                    for a, b in zip(direct_info["resolved"], print_info["resolved"], strict=True)
                )
                report["route_witness"] = {
                    "direct": str(direct_path),
                    "print": str(print_path),
                    "resolved_equal": recipe_equal,
                    "frames_equal": bool(np.array_equal(direct_frames, print_frames)),
                }
        except ValueError as error:
            report["reason"] = str(error)
        write_json(output / f"{name}.json", report)


def footprint_input(root, request):
    value = np.float32(request["pitch_um"]) / np.float32(request["size"])
    directory = root / request["tables"]
    # Read the exact Float32 filename from the frozen capture regardless of its
    # shortest-decimal spelling on the Swift host.
    candidates = [file for file in directory.glob("footprint-*.f32") if np.float32(file.stem[10:]) == value]
    if len(candidates) != 1:
        raise ValueError(f"missing or ambiguous table for footprint {value} in {directory}")
    return candidates[0]
