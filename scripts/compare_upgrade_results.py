from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def metric(summary, key, default=None):
    return (summary or {}).get("metrics", {}).get(key, default)


def qmetric(summary, query_type, key="ndcg_at_10"):
    return (summary or {}).get("query_type_metrics", {}).get(query_type, {}).get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--enriched")
    ap.add_argument("--output", default="reports/27_quality_upgrade_comparison.md")
    args = ap.parse_args()
    baseline, core, enriched = load(args.baseline), load(args.core), load(args.enriched) if args.enriched else None
    rows = []
    for name, summary in [("baseline", baseline), ("core_upgrade", core), ("metadata_enriched", enriched)]:
        if not summary:
            continue
        rows.append({
            "variant": name,
            "ndcg@10": metric(summary, "ndcg_at_10"),
            "recall_efficiency@10": metric(summary, "recall_efficiency_at_10"),
            "mrr@10": metric(summary, "mrr_at_10"),
            "similar_to_ndcg@10": qmetric(summary, "similar_to"),
            "personalized_ndcg@10": qmetric(summary, "personalized"),
            "mood_decade_ndcg@10": qmetric(summary, "mood_decade"),
            "genre_tag_ndcg@10": qmetric(summary, "genre_tag"),
            "ranker_lift": summary.get("ranker_ndcg_lift_vs_hybrid"),
            "launch": summary.get("launch_decision"),
            "dense_backend": summary.get("dense_backend"),
            "ranker_backend": summary.get("ranker_backend"),
        })
    headers = list(rows[0]) if rows else []
    lines = ["# Quality Upgrade Comparison", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines += ["", "## Interpretation", "", "The core upgrade should improve similar-to and personalized slices without regressing genre/tag search materially. Metadata enrichment should only be claimed when mapping coverage is documented in reports 24–26."]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
