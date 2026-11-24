"""
Small numeric guards applied to every forecast before it is submitted.

The tournament scores with a log rule, so one confident miss costs far more than
many slightly-too-cautious hits gain. These guards cap how confident a submitted
forecast may be.
"""

from __future__ import annotations

BINARY_FLOOR = 0.02
BINARY_CEILING = 0.98
OPTION_FLOOR = 0.01


def clamp_probability(
    probability: float, floor: float = BINARY_FLOOR, ceiling: float = BINARY_CEILING
) -> float:
    if probability != probability:  # NaN
        raise ValueError("Probability is NaN")
    return max(floor, min(ceiling, float(probability)))


def floor_option_probabilities(
    probabilities: list[float], floor: float = OPTION_FLOOR
) -> list[float]:
    """
    Give every option at least `floor` and rescale the rest so the total is 1.
    Options already above the floor shrink proportionally and never fall below it.
    """
    if not probabilities:
        return []
    if any(p != p or p < 0 for p in probabilities):
        raise ValueError(f"Invalid option probabilities: {probabilities}")
    count = len(probabilities)
    floor = min(floor, 1.0 / count)
    total = sum(probabilities)
    if total <= 0:
        return [1.0 / count] * count
    scaled = [p / total for p in probabilities]

    low = [i for i, p in enumerate(scaled) if p < floor]
    # Raising some options to the floor shrinks the others, which can push more
    # of them under it, so repeat until the set of floored options is stable.
    while True:
        free_mass = 1.0 - floor * len(low)
        rest_total = sum(p for i, p in enumerate(scaled) if i not in low)
        result = [
            floor if i in low else p * free_mass / rest_total
            for i, p in enumerate(scaled)
        ]
        newly_low = [i for i, p in enumerate(result) if p < floor - 1e-12 and i not in low]
        if not newly_low:
            return result
        low.extend(newly_low)
