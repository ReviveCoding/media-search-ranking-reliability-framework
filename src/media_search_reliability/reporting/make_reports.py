from __future__ import annotations

from pathlib import Path
import pandas as pd


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_report(path: str | Path, title: str, sections: dict[str, str]) -> None:
    lines = [f"# {title}", ""]
    for name, body in sections.items():
        lines.extend([f"## {name}", "", str(body), ""])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


def write_model_card(path: str | Path, ranker_backend: str, dense_backend: str, launch_decision: str):
    write_report(path, "Model Card", {
        "Model purpose": "Public-data media search and discovery ranking for offline evaluation and local serving demos.",
        "Multimodal scope": "The default path uses visual-intent queries and scene/tag metadata as a lightweight cross-modal proxy. It does not claim CLIP, video-encoder, or VLM training unless an optional extension is explicitly enabled and separately reported.",
        "Training data": "MovieLens-style public ratings/tags plus synthetic query labels and synthetic user-context features.",
        "Retrieval model": f"BM25, dense retrieval, vector index, and hybrid fusion. Dense backend: {dense_backend}.",
        "Ranking model": f"{ranker_backend} with query-grouped graded relevance labels.",
        "Known limitations": "Synthetic query labels are proxies for human search relevance. MovieLens labels are also offline proxies. GPU backends, optional encoders, and production-scale latency require separate validation in the target environment. No real Apple, Siri, Spotlight, Apple TV, or private user data is used.",
        "Launch decision": launch_decision,
    })


def write_claim_boundary(path: str | Path):
    write_report(path, "Claim Boundary", {
        "Allowed claims": "Public-data, offline, local-runnable evaluation framework for media search, ranking, reliability, and lightweight visual-intent metadata proxies. Production-style may describe workflow structure, not production deployment or scale.",
        "Disallowed claims": "No Apple internal data, no Siri/Spotlight logs, no Apple TV usage data, no real user personalization data, no biometric data, no production iOS/Core ML deployment, no validated production-scale latency, and no CLIP/VLM or large-scale foundation-model training unless separately implemented and reported.",
    })
