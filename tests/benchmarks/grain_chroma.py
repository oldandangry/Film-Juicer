"""Refit the reduced-chroma G2 candidate using retained full-color fields."""

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import grain_analysis as analysis
from grain_campaign import frames_in, read_json, request_for, statistics, write_json


WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReferenceCaptures:
    """Read original completed captures with their native request and byte digests."""

    def __init__(self, source):
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("reference calibration receipt changed")
        receipt = read_json(path)
        if receipt["status"] != "passed" or not all(command["drained"] for command in receipt["commands"]):
            raise ValueError("reference calibration did not complete")
        self.captures = {Path(record["path"]).name: record for record in receipt["captures"]}
        if len(self.captures) != len(receipt["captures"]):
            raise ValueError("duplicate reference capture names")
        self.checked = set()

    def capture(self, name, request):
        record = self.captures[name]
        path = Path(record["path"])
        if name not in self.checked:
            manifest = read_json(path / "manifest.json")
            request = {**request}
            for key in ("bank", "tables"):
                if key in request:
                    request[key] = str(Path(request[key]).resolve())
            if (
                record["status"] != "passed"
                or any(manifest.get(key) != value for key, value in request.items())
                or digest(path / "manifest.json") != record["manifest"]
                or digest(path / "frames.f32") != record["frames"]
            ):
                raise ValueError(f"reference capture failed identity checks: {name}")
            self.checked.add(name)
        return path


def mix_density(density, amounts, frame, chroma):
    """Offline estimator from measured unit-amount fields; qualification uses CUDA output."""
    base = density[:, None, None]
    light = 10.0 ** (amounts[:, None, None] * (base - frame))
    weights = WEIGHTS * (10.0**-density)
    common = np.einsum("c,cyx->yx", weights / weights.sum(), light)
    return base - np.log10(chroma * light + (1 - chroma) * common)


def aperture_covariance(frames, pitch, window):
    covariance = np.zeros((3, 3))
    for frame in frames:
        field = analysis.aperture_density(frame, pitch, window).reshape(3, -1)
        centered = field - field.mean(axis=1, keepdims=True)
        covariance += (centered @ centered.T) / centered.shape[1]
    return covariance / len(frames)


def coupled_strength(covariance, density, chroma, amounts):
    weights = WEIGHTS * (10.0**-density)
    # Derivative of the density projection about the ungrained patch. Retain
    # measured covariance because the three estimates now share record noise.
    derivatives = (chroma * np.eye(3) + (1 - chroma) * (weights / weights.sum())) * amounts
    return np.sqrt(np.einsum("ij,jk,ik->i", derivatives, covariance, derivatives))


def estimate_fit(path, current, chroma, moments):
    metadata, frames, window = frames_in(path)
    pitch = metadata["pitch_um"]
    density = np.asarray(metadata["requests"][0]["density"])
    if path not in moments:
        moments[path] = aperture_covariance(frames, pitch, window)
    covariance = moments[path]
    target = np.asarray(current["strength"])

    def modeled(amounts):
        return coupled_strength(covariance, density, chroma, amounts)

    def solve(goal, initial):
        return least_squares(lambda a: (modeled(a) - goal) / target, initial, bounds=(0.125, 8.0)).x

    amounts = solve(target, np.ones(3))
    history = []
    for iteration in range(3):
        frames_stats = [
            analysis.analyze_match_frame(mix_density(density, amounts, frame, chroma), pitch, window)
            for frame in frames
        ]
        projected = analysis.summarize_frames(frames_stats, pitch)
        residual = analysis.match_residuals(projected, current)
        history.append({"amounts": amounts.tolist(), "residuals": residual})
        observed = np.asarray(projected["strength"])
        if iteration == 2 or np.max(np.abs(observed / target - 1)) <= 0.004:
            break
        # Correct the linearized strength model with measurements through the
        # full aperture and exact transmission projection at this estimate.
        updated = solve(target + modeled(amounts) - observed, amounts)
        if np.max(np.abs(updated - amounts)) < 0.001:
            break
        amounts = updated
    return {"amounts": amounts.tolist(), "history": history, "projected": residual}


def recheck(experiment, render, output, rows):
    fraction = float(experiment["chroma_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("refit requires a nonzero reduced chroma fraction")
    references = ReferenceCaptures(experiment["reference_calibration"])
    covariance = {}
    for row in rows:
        name = row["id"]
        print(f"Rechecking {name}", flush=True)
        original = read_json(Path(references.captures[f"{name}-calibration-current"]["path"]).parent / f"{name}.json")
        report = {"row": row, "chroma_fraction": fraction, "disposition": "unmatched", "search": [], "attempts": []}

        def capture(phase, arm, times, **controls):
            request = request_for(experiment, row, arm, times, **controls)
            key = f"{name}-{phase}-{arm}"
            if arm == "current" and key in references.captures:
                path = references.capture(key, request)
            else:
                if arm == "film":
                    request["chroma_fraction"] = fraction
                path = render(key, request)
            observed = statistics(path)
            write_json(output / f"{key}-statistics.json", observed)
            return str(path), observed

        try:
            times = experiment["calibration"]
            report["current_calibration"], current = capture("calibration", "current", times)
            start_size = (original.get("frozen") or original["attempts"][0])["size"]
            grid_owner = next(
                item
                for item in experiment["rows"]
                if (item["stock"], item["geometry"]) == (row["stock"], row["geometry"])
            )
            grid = sorted(enumerate(experiment["sizes"]), key=lambda item: abs(math.log2(item[1] / start_size)))

            def check_fit(phase, estimate):
                controls = {key: estimate[key] for key in ("size", "amounts")}
                captured, observed = capture(phase, "film", times, **controls)
                residual = analysis.match_residuals(observed, current)
                attempt = {**controls, "capture": captured, "calibration": residual}
                report["attempts"].append(attempt)
                if residual["matched"]:
                    report["frozen"] = attempt
                    write_json(output / f"{name}-frozen.json", attempt)

            for index, size in grid:
                path = references.capture(
                    f"{grid_owner['id']}-size-{index}-film",
                    request_for(experiment, grid_owner, "film", times, size=size),
                )
                estimate = {"size": size, "source": str(path), **estimate_fit(path, current, fraction, covariance)}
                report["search"].append(estimate)
                if estimate["projected"]["matched"]:
                    check_fit(f"fit-{index}", estimate)
                    if "frozen" in report:
                        break
            if "frozen" not in report:
                if not report["attempts"]:
                    closest = min(
                        report["search"],
                        key=lambda fit: max(
                            abs(fit["projected"]["length"]) / 0.05,
                            max(abs(value) for value in fit["projected"]["strength"]) / 0.02,
                        ),
                    )
                    check_fit("nearest", closest)
                if "frozen" not in report:
                    report["reason"] = "no bounded estimate passed CUDA calibration"
            if "frozen" in report:
                chosen = {key: report["frozen"][key] for key in ("size", "amounts")}
                for phase in ("evaluation", "fractional"):
                    times = experiment[phase]
                    current_path, reference = capture(phase, "current", times)
                    film_path, candidate = capture(phase, "film", times, **chosen)
                    report[phase] = {
                        "current_capture": current_path,
                        "film_capture": film_path,
                        "residuals": analysis.match_residuals(candidate, reference),
                    }
                if report["evaluation"]["residuals"]["matched"] and report["fractional"]["residuals"]["matched"]:
                    report["disposition"] = "qualified"
            if "route_witness" in original:
                report["original_current_route_witness"] = original["route_witness"]
        except ValueError as error:
            report["reason"] = str(error)
        write_json(output / f"{name}.json", report)
