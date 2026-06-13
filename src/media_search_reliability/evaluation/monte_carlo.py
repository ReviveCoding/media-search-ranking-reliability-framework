from __future__ import annotations

import random

PROBABILITY_PARAMETERS = {
    "tag_observation_prob",
    "exploration_rate",
    "cold_start_fraction",
    "label_noise",
}


def sample_scenario_parameters(scenario: dict, seed: int, sampling: dict | None = None) -> dict:
    """Sample bounded uncertain parameters around a configured scenario center."""
    sampling = sampling or {}
    if not sampling.get("enabled", False):
        return dict(scenario)
    rng = random.Random(seed)
    relative_sd = float(sampling.get("relative_sd", 0.04))
    probability_sd = float(sampling.get("probability_sd", 0.015))
    sampled: dict[str, float] = {}
    for name, center_value in scenario.items():
        center = float(center_value)
        if name in PROBABILITY_PARAMETERS:
            value = rng.gauss(center, probability_sd)
            sampled[name] = float(min(1.0, max(0.0, value)))
        else:
            sd = max(abs(center) * relative_sd, 1e-6)
            sampled[name] = float(max(0.01, rng.gauss(center, sd)))
    return sampled
