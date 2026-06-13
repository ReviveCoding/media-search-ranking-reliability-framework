from media_search_reliability.evaluation.monte_carlo import sample_scenario_parameters


def test_parameter_sampling_is_reproducible_and_bounded():
    scenario = {
        "popularity_alpha": 1.05,
        "preference_strength": 1.35,
        "rating_noise": 0.7,
        "tag_observation_prob": 0.45,
        "exploration_rate": 0.12,
        "cold_start_fraction": 0.08,
        "label_noise": 0.03,
    }
    sampling = {"enabled": True, "relative_sd": 0.05, "probability_sd": 0.02}
    first = sample_scenario_parameters(scenario, seed=123, sampling=sampling)
    second = sample_scenario_parameters(scenario, seed=123, sampling=sampling)
    assert first == second
    for key in ["tag_observation_prob", "exploration_rate", "cold_start_fraction", "label_noise"]:
        assert 0 <= first[key] <= 1
    assert any(first[key] != scenario[key] for key in scenario)
