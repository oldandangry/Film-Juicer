"""Request the frozen G2 quality witnesses using calibration capture semantics."""

import math

import numpy as np
from grain_campaign import PORTRA, PROVIA, request_for


def portra_row(experiment, geometry):
    return next(
        row
        for row in experiment["rows"]
        if row["stock"] == PORTRA and row["geometry"] == geometry and row["look"] == "default"
    )


def density_patches(root, experiment, key):
    stock = experiment["stocks"][key]
    words = np.fromfile(root / stock["bank"] / "records.f32", dtype="<f4").reshape(3, 20)
    lower, upper = words[:, 0] - words[:, 2], words[:, 1] - words[:, 2]
    anchor_span = np.clip(np.ones(3), 0.05 * (upper - lower), 0.9 * (upper - lower))
    anchor = upper - anchor_span if key == PROVIA else lower + anchor_span
    common = (float(max(lower)), float(min(upper)))
    labels, patches = [], []
    for record, record_name in enumerate(("neutral", "cyan", "magenta", "yellow"), start=-1):
        lo, hi = common if record < 0 else (float(lower[record]), float(upper[record]))
        at_anchor = float(anchor[1 if record < 0 else record])
        samples = [(f"level-{level:02d}", lo + (hi - lo) * level / 16) for level in range(17)]
        samples.extend(
            (name, lo + (hi - lo) * fraction)
            for name, fraction in (
                ("near-toe-01", 0.01),
                ("near-toe-03", 0.03),
                ("near-shoulder-97", 0.97),
                ("near-shoulder-99", 0.99),
            )
        )
        samples.append(("anchor", at_anchor))
        for name, density in samples:
            values = list(stock["density"]) if record >= 0 else [density] * 3
            if record >= 0:
                values[record] = density
            labels.append(f"{record_name}-{name}")
            patches.append(np.asarray(values, dtype=np.float32).tolist())
    return labels, patches


def quality_requests(root, experiment):
    quality = experiment["quality"]
    times = quality["density"]["frames"]
    for key in quality["density"]["stocks"]:
        row = next(
            row
            for row in experiment["rows"]
            if row["stock"] == key and row["geometry"] == "uhd-36" and row["look"] == "default"
        )
        labels, patches = density_patches(root, experiment, key)
        for arm in ("current", "film"):
            request = request_for(experiment, row, arm, times)
            request.update(
                crop_width=request["width"], crop_height=request["height"], analysis_width=96, analysis_height=96
            )
            for sample in request["requests"]:
                sample.update(stimulus="patches", columns=11, rows=8, patches=patches, patch_names=labels)
            yield f"density-{key.replace('_', '-')}-{arm}", request

    row = portra_row(experiment, "uhd-36")
    scale = quality["scale"]
    for pitch in scale["pitches_um"]:
        extent = round(scale["side_um"] / pitch)
        for arm in ("current", "film"):
            request = request_for(experiment, row, arm, [scale["time"]])
            request.update(
                pitch_um=pitch,
                long_edge_mm=scale["side_um"] / 1000,
                width=extent,
                height=extent,
                crop_width=extent,
                crop_height=extent,
                analysis_width=extent,
                analysis_height=extent,
            )
            yield f"scale-{str(pitch).replace('.', '-')}-{arm}", request

    animation = quality["animation"]
    row = portra_row(experiment, "hd-36")
    times = [index + (0.5 if index % 2 else 0) for index in range(animation["frames"])]
    rng = np.random.default_rng(quality["preview"]["order_seed"])
    repeat_times = [times[index] for index in rng.choice(len(times), 16, replace=False)]
    for stimulus in ("flat", "edge"):
        for arm in ("current", "film"):
            request = request_for(experiment, row, arm, times + repeat_times)
            request.update(
                clip_token=animation["clip_token"],
                session_seed=animation["session_seed"],
                crop_width=520,
                crop_height=296,
                analysis_width=512,
                analysis_height=288,
                sequence_length=len(times),
            )
            for sample in request["requests"]:
                sample["stimulus"] = stimulus
            yield f"animation-{stimulus}-{arm}", request

    for arm in ("current", "film"):
        # Three requests per held-out seed support direct I(k), I(k+.5), I(k+1)
        # cross-seed covariance and temporal mean-light measurements.
        times = [time + offset for time in experiment["evaluation"] for offset in (0, 0.5, 1)]
        request = request_for(experiment, row, arm, times)
        request.update(crop_width=520, crop_height=296, analysis_width=512, analysis_height=288)
        yield f"half-variance-{arm}", request

    # No bank is opened in the candidate bypass path.
    off = request_for(experiment, row, "off", [16])
    off.update(crop_width=520, crop_height=296, analysis_width=512, analysis_height=288)
    yield "identity-off", off


def freeze_quality(root, experiment, write_json, path):
    requests = dict(quality_requests(root, experiment))
    for name, request in requests.items():
        if not math.isfinite(request["pitch_um"]):
            raise ValueError(f"invalid quality request {name}")
    write_json(path, requests)
