from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from fastapi.testclient import TestClient
from media_search_reliability.api.app import app


def main() -> None:
    # Use a context manager so FastAPI/Starlette lifespan hooks are honored.
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok"
        assert health.json()["ready"] is True

        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text
        assert ready.json()["model_loaded"] is True
        assert "load_latency_ms" in ready.json()

        # Use signals that are stable across supported MovieLens variants.
        # Free-form catalog tags are dataset-dependent: a dataset may contain
        # "robot", "robots", or no robot tag at all. Genre and the framework's
        # deterministic mood mapping are the intended train-serving contract.
        resp = client.post(
            "/search",
            json={"query": "funny sci-fi movie with robots", "top_k": 5},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["query"]
        assert payload["model_variant"] == "hybrid_lambdarank_calibrated"
        assert "parsed_intent" in payload

        parsed = payload["parsed_intent"]
        assert parsed["target_genre"].lower() == "sci-fi", parsed
        assert parsed["target_mood_tag"] == "funny", parsed
        # target_tag is intentionally not asserted. It is derived from the
        # observed catalog tag vocabulary and therefore varies by dataset.
        assert parsed["query_type"] in {"genre_tag", "adhoc"}, parsed
        assert float(parsed["parser_confidence"]) > 0.0, parsed

        assert isinstance(payload["results"], list) and len(payload["results"]) > 0
        first = payload["results"][0]
        for key in [
            "movie_id",
            "title",
            "genres",
            "score",
            "calibrated_score",
            "matched_signals",
            "review_flags",
        ]:
            assert key in first, first
        for key in ["model_load_ms", "latency_ms", "total_latency_ms"]:
            assert key in payload
        assert payload["model_load_ms"] == 0.0
        assert payload["total_latency_ms"] >= payload["latency_ms"]

        anchor_id = int(first["movie_id"])
        similar = client.post(
            "/search",
            json={
                "query": f"movie like {first['title']}",
                "top_k": 5,
                "anchor_movie_id": anchor_id,
            },
        )
        assert similar.status_code == 200, similar.text
        assert all(int(item["movie_id"]) != anchor_id for item in similar.json()["results"])

        batch = client.post(
            "/batch_search",
            json={
                "items": [
                    {
                        "query": "funny sci-fi movie",
                        "preferred_genres": ["Comedy", "Sci-Fi"],
                    },
                    {"query": "dark thriller"},
                ],
                "top_k": 3,
            },
        )
        assert batch.status_code == 200, batch.text
        batch_payload = batch.json()
        assert len(batch_payload["responses"]) == 2
        assert all(len(response["results"]) > 0 for response in batch_payload["responses"])

        gate = client.get("/launch_gate")
        assert gate.status_code == 200, gate.text
        assert "decision" in gate.json(), gate.json()

        eval_resp = client.post("/evaluate")
        assert eval_resp.status_code == 200, eval_resp.text
        assert "metrics" in eval_resp.json(), eval_resp.json()

    print("API smoke test PASS")


if __name__ == "__main__":
    main()
