"""Independent high-precision inputs for the production highlight operator.

This generates supplemental inputs, never checked-in fixture replacements.
The oracle evaluates the analytic curve on exact FP32 samples and exact resolved
FP64 descriptor words, independently of the CUDA implementation.
"""

from decimal import Decimal, localcontext
import math
import struct


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def rounded(value):
    return from_bits(float_bits(value))


def next_float(value, steps=1):
    return from_bits(float_bits(value) + steps)


def first_above(value):
    narrowed = rounded(value)
    return next_float(narrowed) if narrowed <= value else narrowed


def _normalized(t):
    # Independent analytic continuation: sum t^k/(k+2)!, to 75 digits.
    if t < Decimal("0.01"):
        term = Decimal("0.5")
        total = term
        for k in range(1, 100):
            term = term * t / (k + 2)
            total += term
            if abs(term) < Decimal("1e-75"):
                return total
        raise ValueError("highlight oracle series did not converge")
    return (t.exp() - t - 1) / (t * t)


def _samples(protection, maximum):
    values = {0.0, maximum}
    onset = first_above(protection)
    for value in (rounded(protection), onset, maximum):
        word = float_bits(value)
        for offset in range(-3, 4):
            if 0 <= word + offset <= 0x7F7FFFFF:
                values.add(from_bits(word + offset))
    for i in range(65):
        values.add(rounded(maximum * (i / 64)))
        values.add(rounded(protection + (maximum - protection) * (i / 64)))
    return sorted(v for v in values if 0 <= v <= maximum)


def _row(name, protection, sharpness, gain, maximum, *, values=None, active=True, inputs=None, width=None):
    if inputs is None:
        values = _samples(protection, maximum) if values is None else values
        # Channel permutations keep a shared maximum and expose channel-local mistakes.
        inputs = [v for i in range(len(values)) for v in (values[i], values[-1 - i], values[(i + 3) % len(values)])]
        width = len(values)
    if width is None or width <= 0 or len(inputs) % (3 * width):
        raise ValueError("invalid highlight oracle extent")
    invalid = any(not math.isfinite(v) or v < 0 for v in inputs)
    actual_maximum = max((v for v in inputs if math.isfinite(v) and v >= 0), default=0.0)
    limit = from_bits(0x7F7FFFFF)
    status = (
        "InvalidCarrier"
        if invalid
        else "Bypass"
        if actual_maximum <= protection or not active
        else "PeakOverflow"
        if actual_maximum * gain > limit
        else "Applied"
    )
    with localcontext() as context:
        context.prec = 80
        p, a, g, m = map(Decimal.from_float, (protection, sharpness, gain, actual_maximum))
        span = m - p
        peak_t = a * span / m if status == "Applied" else Decimal(0)
        denominator = _normalized(peak_t) if status == "Applied" else Decimal(1)
        expected = []
        for sample in inputs:
            x = Decimal.from_float(sample)
            if status == "Applied" and x > p:
                fraction = (x - p) / span
                result = x + m * (g - 1) * fraction * fraction * _normalized(peak_t * fraction) / denominator
            else:
                result = x
            expected.append(str(result))
    return {
        "id": name,
        "width": width,
        "height": len(inputs) // (3 * width),
        "descriptor": {"recipe_hash": int(active), "protection": protection, "sharpness": sharpness, "peak_gain": gain},
        "input_bits": [float_bits(v) for v in inputs],
        "expected": expected,
        "maximum_bits": float_bits(actual_maximum),
        "status": status,
        "monotonic": not invalid,
    }


def generate():
    rows = []
    gains = [math.exp2(2**-20), math.exp2(0.125), 16.0, 1048576.0]
    sharpnesses = [1.0, math.pow(28.0, 0.7), 28.0]
    for pi, protection in enumerate((0.184, 2.944, 0.184 * 1048576.0)):
        onset = first_above(protection)
        maxima = [
            onset,
            next_float(onset, 3),
            rounded(protection * 1.0001),
            rounded(protection * 2),
            rounded(protection * 32),
        ]
        for gi, gain in enumerate(gains):
            for ai, sharpness in enumerate(sharpnesses):
                for mi, maximum in enumerate(maxima):
                    rows.append(_row(f"supplement-grid-p{pi}-g{gi}-a{ai}-m{mi}", protection, sharpness, gain, maximum))

    # Frame constants span the valid Rust recipe range; resolve the exact accepted
    # carrier boundary independently of FP32 gain multiplication.
    limit = from_bits(0x7F7FFFFF)
    for gi, gain in enumerate(gains):
        threshold = rounded(limit / gain)
        while threshold * gain > limit:
            threshold = next_float(threshold, -1)
        for ai, sharpness in enumerate(sharpnesses):
            for offset in (-2, -1, 0, 1, 2):
                word = float_bits(threshold) + offset
                if word <= 0x7F7FFFFF:
                    rows.append(
                        _row(f"supplement-overflow-g{gi}-a{ai}-offset{offset}", 0.184, sharpness, gain, from_bits(word))
                    )

    # Values on either side of the FP32 rounding midpoint and exact float onset.
    # These are resolved descriptor tests, not alternative authored-control math.
    for base_index, base in enumerate((0.25, 1.0, 65536.0)):
        middle = (base + next_float(base)) * 0.5
        protections = [
            math.nextafter(base, 0),
            base,
            math.nextafter(base, math.inf),
            math.nextafter(middle, 0),
            middle,
            math.nextafter(middle, math.inf),
        ]
        for pi, protection in enumerate(protections):
            for ai, sharpness in enumerate(sharpnesses):
                rows.append(
                    _row(
                        f"supplement-midpoint-b{base_index}-p{pi}-a{ai}",
                        protection,
                        sharpness,
                        16.0,
                        next_float(base, 5),
                    )
                )

    for ai, sharpness in enumerate(sharpnesses):
        # Probe the proposed polynomial/exponential crossover from both sides.
        p, maximum = 0.184, 8.0
        values = _samples(p, maximum)
        for t in (0.49999, 0.5, 0.50001, 0.99999, 1.0, 1.00001):
            value = rounded(p + maximum * t / sharpness)
            if value <= maximum:
                values.extend([next_float(value, -1), value, next_float(value)])
        rows.append(_row(f"supplement-crossover-a{ai}", p, sharpness, 16.0, maximum, values=sorted(set(values))))

    # Probe neighboring carrier words at the refined experiment's ratio=1/2 and
    # peak-shape=1/2 joins. The oracle itself has neither branch.
    join_gains = (1.0, math.nextafter(1.0, math.inf), 1.0 + 2**-25, 1.0 + 2**-23, gains[0], gains[2], gains[3])
    for gi, gain in enumerate(join_gains):
        for ai, sharpness in enumerate(sharpnesses):
            for pi, protection in enumerate((0.184, 2.944)):
                huge = rounded(limit / gain)
                while huge * gain > limit:
                    huge = next_float(huge, -1)
                for mi, maximum in enumerate((rounded(protection * 32), huge)):
                    peak_t = sharpness * (maximum - protection) / maximum
                    denominator = math.expm1(peak_t) - peak_t
                    low, high = 0.0, 1.0
                    for _ in range(64):
                        middle = (low + high) * 0.5
                        t = peak_t * middle
                        if (math.expm1(t) - t) / denominator < 0.5:
                            low = middle
                        else:
                            high = middle
                    center = rounded(protection + (maximum - protection) * ((low + high) * 0.5))
                    values = _samples(protection, maximum)
                    values.extend(next_float(center, offset) for offset in range(-8, 9))
                    rows.append(
                        _row(
                            f"supplement-ratio-join-g{gi}-a{ai}-p{pi}-m{mi}",
                            protection,
                            sharpness,
                            gain,
                            maximum,
                            values=sorted(set(values)),
                        )
                    )
            protection = 2.944
            center = rounded(protection / (1 - 0.5 / sharpness))
            for offset in range(-2, 3):
                rows.append(
                    _row(
                        f"supplement-shape-join-g{gi}-a{ai}-offset{offset}",
                        protection,
                        sharpness,
                        gain,
                        next_float(center, offset),
                    )
                )

    # The peak and invalid carrier sit in the final reduction block, not block zero.
    width, height = 513, 2
    inputs = [rounded(0.1 + (i % 997) * (7.0 / 997)) for i in range(width * height * 3)]
    inputs[-1] = 8.0
    rows.append(_row("supplement-multiblock-peak", 0.184, 28.0, 16.0, 8.0, inputs=inputs, width=width))
    invalid_inputs = list(inputs)
    invalid_inputs[-2] = math.nan
    rows.append(_row("supplement-multiblock-invalid", 0.184, 28.0, 16.0, 8.0, inputs=invalid_inputs, width=width))

    rows.extend(
        [
            _row("supplement-zero-boost", 0.184, 28.0, 1.0, 8.0, active=False),
            _row("supplement-zero-maximum", 0.184, 28.0, 16.0, 0.0, values=[0.0, -0.0]),
            _row("supplement-below-protection", 2.944, 28.0, 1048576.0, 1.0, values=[0.0, 0.5, 1.0]),
            _row("supplement-exact-protection", 1.0, 28.0, 16.0, 1.0, values=[0.0, next_float(1.0, -1), 1.0]),
            _row("supplement-nan", 0.184, 28.0, 16.0, 8.0, values=[0.0, math.nan, 8.0]),
            _row("supplement-infinity", 0.184, 28.0, 16.0, 8.0, values=[0.0, math.inf, 8.0]),
            _row("supplement-negative", 0.184, 28.0, 16.0, 8.0, values=[-1.0, 0.0, 8.0]),
        ]
    )
    return {
        "schema": "boost-precision-supplement.v1",
        "origin": "Named supplemental production-operator inputs; checked-in fixtures remain immutable.",
        "oracle": "Decimal precision 80; exact binary input/descriptor values; analytic exponential continuation.",
        "rows": rows,
    }
