from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
TAGS = [
    "robots", "space", "detective", "rainy city", "restaurant", "family", "animals", "emotional",
    "dark", "psychological", "funny", "classic", "friendship", "revenge", "survival", "mystery",
    "future", "small town", "coming of age", "courtroom", "heist", "superhero", "sports", "music"
]


def _rng(seed: int) -> np.random.Generator:
    """Use an explicitly seeded modern bit generator for reproducible simulations."""
    bitgen = np.random.PCG64DXSM(seed) if hasattr(np.random, "PCG64DXSM") else np.random.PCG64(seed)
    return np.random.Generator(bitgen)


def generate_synthetic_movielens(
    n_movies: int = 1200,
    n_users: int = 800,
    n_ratings: int = 40000,
    seed: int = 42,
    popularity_alpha: float = 1.05,
    preference_strength: float = 1.35,
    rating_noise: float = 0.70,
    tag_observation_prob: float = 0.45,
    exploration_rate: float = 0.12,
    cold_start_fraction: float = 0.08,
):
    """Generate a Monte-Carlo-style MovieLens proxy dataset.

    The simulation intentionally creates realistic stressors rather than iid rows:
    a Zipf-like item popularity distribution, heterogeneous user genre preferences,
    latent item quality, sparse observed tags, noisy ratings, and a cold-start tail.
    All parameters are configurable so the Monte Carlo harness can vary data regimes.
    """
    if n_movies < 20 or n_users < 10 or n_ratings < n_users:
        raise ValueError("Synthetic dataset is too small for a meaningful ranking simulation.")
    if not 0 <= tag_observation_prob <= 1 or not 0 <= exploration_rate <= 1:
        raise ValueError("Probabilities must be in [0, 1].")

    rng = _rng(seed)
    genre_to_idx = {g: i for i, g in enumerate(GENRES)}

    # Randomize popularity ranks so movie_id itself does not leak popularity.
    ranks = np.arange(1, n_movies + 1, dtype=float)
    rng.shuffle(ranks)
    popularity = np.power(ranks, -max(0.05, float(popularity_alpha)))
    popularity /= popularity.sum()
    latent_quality = 1.0 + 4.0 * rng.beta(4.0, 2.5, size=n_movies)

    n_cold = max(1, int(round(n_movies * cold_start_fraction)))
    cold_idx = np.argsort(popularity)[:n_cold]
    cold_mask = np.zeros(n_movies, dtype=bool)
    cold_mask[cold_idx] = True
    popularity[cold_mask] *= 0.15
    popularity /= popularity.sum()

    title_left = np.array(["Hidden", "Last", "Bright", "Silent", "Future", "Midnight", "Lost", "Golden"])
    title_right = np.array(["Signal", "Road", "City", "Dream", "Planet", "Case", "Story", "Echo"])

    movies = []
    genre_matrix = np.zeros((n_movies, len(GENRES)), dtype=float)
    movie_tags: list[list[str]] = []
    for i, movie_id in enumerate(range(1, n_movies + 1)):
        genres = rng.choice(GENRES, size=int(rng.integers(1, 4)), replace=False).tolist()
        for genre in genres:
            genre_matrix[i, genre_to_idx[genre]] = 1.0
        year = int(rng.integers(1980, 2026))
        tags = rng.choice(TAGS, size=int(rng.integers(2, 5)), replace=False).tolist()
        movie_tags.append(tags)
        title = f"{rng.choice(title_left)} {rng.choice(title_right)} ({year})"
        movies.append({
            "movie_id": movie_id,
            "title": title,
            "genres": "|".join(genres),
            "year": year,
            "synthetic_tags": "|".join(tags),
            "latent_quality": float(latent_quality[i]),
            "population_probability": float(popularity[i]),
            "cold_start_design_flag": int(cold_mask[i]),
        })
    movies = pd.DataFrame(movies)

    # Sparse Dirichlet preferences yield heterogeneous users with 2-4 dominant genres.
    user_pref = rng.dirichlet(np.full(len(GENRES), 0.35), size=n_users)
    user_ids = np.arange(1, n_users + 1)
    latent_preferred_genres = [
        "|".join(GENRES[int(idx)] for idx in np.argsort(-user_pref[u_idx], kind="stable")[:3])
        for u_idx in range(n_users)
    ]

    # Precompute each user's sampling distribution over movies.
    affinity = user_pref @ genre_matrix.T
    utility = preference_strength * affinity + 0.25 * (latent_quality[None, :] - 3.0)
    utility = np.clip(utility, -8, 8)
    preference_weights = np.exp(utility)
    behavioral = preference_weights * popularity[None, :]
    behavioral /= behavioral.sum(axis=1, keepdims=True)
    sampling_probs = (1.0 - exploration_rate) * behavioral + exploration_rate / n_movies
    sampling_probs /= sampling_probs.sum(axis=1, keepdims=True)

    ratings = []
    # Ensure every synthetic user has at least one interaction so a stable latent
    # user profile is available independently of observation noise.
    remaining = max(0, n_ratings - n_users)
    sampled_users = np.concatenate([np.arange(n_users), rng.integers(0, n_users, size=remaining)])
    rng.shuffle(sampled_users)
    for u_idx in sampled_users:
        movie_idx = int(rng.choice(n_movies, p=sampling_probs[u_idx]))
        genre_affinity = float(affinity[u_idx, movie_idx] / max(1.0, genre_matrix[movie_idx].sum()))
        mean_rating = 1.6 + 2.2 * genre_affinity + 0.34 * latent_quality[movie_idx]
        rating = float(np.clip(np.round((mean_rating + rng.normal(0, rating_noise)) * 2) / 2, 1, 5))
        ratings.append({
            "user_id": int(user_ids[u_idx]),
            "movie_id": int(movie_idx + 1),
            "rating": rating,
            "timestamp": int(1_000_000_000 + rng.integers(0, 500_000_000)),
            "preference_affinity": genre_affinity,
            "latent_preferred_genres": latent_preferred_genres[u_idx],
        })
    ratings = pd.DataFrame(ratings)

    tag_rows = []
    for movie_idx, tags in enumerate(movie_tags):
        # More popular movies receive slightly more observed tags, while the base
        # probability controls overall metadata sparsity.
        popularity_boost = min(0.25, 2.0 * popularity[movie_idx] * n_movies)
        observe_p = float(np.clip(tag_observation_prob + popularity_boost, 0, 1))
        for tag in tags:
            if rng.random() < observe_p:
                tag_rows.append({
                    "user_id": int(rng.choice(user_ids)),
                    "movie_id": int(movie_idx + 1),
                    "tag": tag,
                    "timestamp": int(1_000_000_000 + rng.integers(0, 500_000_000)),
                })
    tags = pd.DataFrame(tag_rows, columns=["user_id", "movie_id", "tag", "timestamp"])
    return movies, ratings, tags
