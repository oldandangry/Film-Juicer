"""Offline G3 grain timing decisions from retained invocation samples.

The caller owns the frozen workload manifest and native captures. This module
validates pair membership, retains raw input data, and applies the fixed G3
repeatability envelope without trimming or pooling observations.
"""

import math


BUDGET = 1.05
PAIR_ORDER = ("ab", "ba", "ab")
QUANTILES = (0.5, 0.9)
PHASE_COUNTS = {"warmup": 10, "measurement": 30}


def linear_quantile(values, q):
    """Return a linearly interpolated quantile at q*(n-1)."""
    if not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("quantile must be finite and in [0, 1]")
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("quantile values must be nonempty and finite")
    numbers.sort()
    position = q * (len(numbers) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return numbers[lower] + fraction * (numbers[upper] - numbers[lower])


def _samples(invocation):
    if not isinstance(invocation, dict) or not isinstance(invocation.get("samples"), list):
        raise ValueError("invocation must contain a samples array")
    samples = invocation["samples"]
    identities = set()
    values = {phase: [] for phase in PHASE_COUNTS}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample must be an object")
        phase = sample.get("phase")
        index = sample.get("index")
        if phase not in PHASE_COUNTS or not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("sample has invalid phase or index")
        identity = (phase, index)
        if identity in identities:
            raise ValueError("duplicate sample identity")
        identities.add(identity)
        if index < 0 or index >= PHASE_COUNTS[phase]:
            raise ValueError("sample index outside phase membership")
        if phase == "measurement":
            value = sample.get("gpu_ms")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("decision GPU times must be finite and positive")
            values[phase].append(float(value))
    expected = {(phase, index) for phase, count in PHASE_COUNTS.items() for index in range(count)}
    if identities != expected:
        raise ValueError("invocation sample membership is incomplete")
    return values["measurement"]


def _pair_ratio(pair, numerator, denominator, q):
    if (
        not isinstance(pair, dict)
        or not isinstance(pair.get(numerator), dict)
        or not isinstance(pair.get(denominator), dict)
    ):
        raise ValueError("pair must contain both distinct invocation objects")
    if pair[numerator] is pair[denominator]:
        raise ValueError("pair invocations must be distinct objects")
    return linear_quantile(_samples(pair[numerator]), q) / linear_quantile(_samples(pair[denominator]), q)


def _ordered_pairs(pairs, name):
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError(f"{name} must contain exactly three pairs")
    for index, (pair, expected) in enumerate(zip(pairs, PAIR_ORDER, strict=True)):
        if not isinstance(pair, dict) or pair.get("order") != expected:
            raise ValueError(f"{name} pair {index} has incorrect A/B order")
    return pairs


def analyze_row(row):
    """Add fixed G3 arithmetic to one manifest row; preserve all supplied raw data.

    Expected row keys: ``id``, ``disposition``, optional ``mandatory`` and
    ``baseline_pairs`` / ``candidate_pairs``. Pair objects use ``order`` equal
    to ``ab`` or ``ba`` and contain distinct ``a``/``b`` invocations.
    """
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        raise ValueError("row must have a string id")
    result = dict(row)
    disposition = row.get("disposition", "planned")
    if disposition in {"unmatched", "unrun", "deferred", "descriptive", "grain_off"}:
        result["decision"] = {"disposition": disposition}
        return result
    if disposition not in {"planned", "qualified", "invalid", "unresolved", "within_budget", "over_budget"}:
        raise ValueError(f"unknown row disposition: {disposition}")
    if disposition == "unrun":
        result["decision"] = {"disposition": "unrun"}
        return result
    try:
        baseline = _ordered_pairs(row.get("baseline_pairs"), "baseline_pairs")
        if row.get("candidate_pairs") is None:
            ratios = {q: [_pair_ratio(pair, "b", "a", q) for pair in baseline] for q in QUANTILES}
            allowances = {q: max(abs(math.log(value)) for value in values) for q, values in ratios.items()}
            factors = {q: math.exp(value) for q, value in allowances.items()}
            admitted = not any(value >= BUDGET for value in factors.values())
            result["decision"] = {
                "disposition": "unresolved" if not admitted else "unrun",
                "candidate_admitted": admitted,
                "budget": BUDGET,
                "baseline_ratios": {str(q): values for q, values in ratios.items()},
                "allowances": {str(q): allowances[q] for q in QUANTILES},
                "allowance_factors": {str(q): factors[q] for q in QUANTILES},
            }
            return result
        if any(
            math.exp(max(abs(math.log(_pair_ratio(pair, "b", "a", q))) for pair in baseline)) >= BUDGET
            for q in QUANTILES
        ):
            raise ValueError("candidate block present although repeatability omits it")
        candidate = _ordered_pairs(row["candidate_pairs"], "candidate_pairs")
        baseline_ratios = {}
        candidate_ratios = {}
        allowances = {}
        upper = {}
        lower = {}
        for q in QUANTILES:
            baseline_ratios[q] = [_pair_ratio(pair, "b", "a", q) for pair in baseline]
            allowances[q] = max(abs(math.log(value)) for value in baseline_ratios[q])
            candidate_ratios[q] = [_pair_ratio(pair, "b", "a", q) for pair in candidate]
            factor = math.exp(allowances[q])
            upper[q] = max(candidate_ratios[q]) * factor
            lower[q] = min(candidate_ratios[q]) / factor
        if all(upper[q] <= BUDGET for q in QUANTILES):
            verdict = "within_budget"
        elif any(lower[q] > BUDGET for q in QUANTILES):
            verdict = "over_budget"
        else:
            verdict = "unresolved"
        result["decision"] = {
            "disposition": verdict,
            "budget": BUDGET,
            "baseline_ratios": {str(q): baseline_ratios[q] for q in QUANTILES},
            "candidate_ratios": {str(q): candidate_ratios[q] for q in QUANTILES},
            "allowances": {str(q): allowances[q] for q in QUANTILES},
            "upper_envelopes": {str(q): upper[q] for q in QUANTILES},
            "lower_envelopes": {str(q): lower[q] for q in QUANTILES},
            "p95_candidate_ratios": [
                linear_quantile(_samples(pair["b"]), 0.95) / linear_quantile(_samples(pair["a"]), 0.95)
                for pair in candidate
            ],
        }
    except (TypeError, ValueError, OverflowError) as error:
        result["decision"] = {"disposition": "invalid", "reason": str(error)}
    return result


def summarize_campaign(rows):
    """Analyze the complete manifest inventory and summarize all dispositions."""
    if not isinstance(rows, list):
        raise ValueError("campaign rows must be an array")
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("campaign row IDs must be present and unique")
    analyzed = [analyze_row(row) for row in rows]
    counts = {}
    for row in analyzed:
        disposition = row["decision"]["disposition"]
        counts[disposition] = counts.get(disposition, 0) + 1
    mandatory = [row for row in analyzed if row.get("mandatory", False)]
    selection = (
        "within_budget"
        if mandatory and all(row["decision"]["disposition"] == "within_budget" for row in mandatory)
        else "not_established"
    )
    return {"rows": analyzed, "disposition_counts": counts, "mandatory_selection": selection}
