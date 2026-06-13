from pathlib import Path
import json
import pandas as pd

try:
    import streamlit as st
except Exception as exc:
    raise RuntimeError("Install the dashboard extra: pip install -e '.[dashboard]'") from exc

st.set_page_config(page_title="Media Search Reliability", layout="wide")
st.title("Media Search & Multimodal Discovery Reliability Framework")
st.caption("Public/synthetic offline evaluation only. No private product logs or production deployment claims.")

artifacts = Path("artifacts")
reports = Path("reports")
summary_path = artifacts / "eval_summary.json"
if not summary_path.exists():
    st.warning("Run `python scripts/run_pipeline.py --config configs/pipeline.yaml --mode demo` first.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = summary.get("metrics", {})
cols = st.columns(5)
cols[0].metric("Launch", summary.get("launch_decision", "UNKNOWN"))
cols[1].metric("NDCG@10", f"{metrics.get('ndcg_at_10', 0):.3f}")
cols[2].metric("Recall efficiency@10", f"{metrics.get('recall_efficiency_at_10', 0):.3f}")
cols[3].metric("ECE", f"{metrics.get('ece', 0):.3f}")
cols[4].metric("p95 latency", f"{metrics.get('p95_latency_ms', 0):.1f} ms")

with st.expander("Run metadata and claim boundary", expanded=False):
    st.json({
        "data_source": summary.get("data_source"),
        "evaluation_truth": summary.get("evaluation_truth"),
        "dense_backend": summary.get("dense_backend"),
        "vector_index_backend": summary.get("vector_index_backend"),
        "ranker_backend": summary.get("ranker_backend"),
        "calibration_method": summary.get("calibration_method"),
    })
    boundary = reports / "09_claim_boundary.md"
    if boundary.exists():
        st.markdown(boundary.read_text(encoding="utf-8"))

for title, filename in [
    ("Retrieval baselines", "retrieval_metrics.csv"),
    ("Ranking ablations", "ablation_metrics.csv"),
    ("Reliability slices", "slice_metrics.csv"),
    ("Quality-latency measurements", "latency_by_variant.csv"),
]:
    path = artifacts / filename
    if path.exists():
        st.subheader(title)
        st.dataframe(pd.read_csv(path), use_container_width=True)

st.subheader("Query-type performance")
query_rows = []
for query_type, values in summary.get("query_type_metrics", {}).items():
    query_rows.append({"query_type": query_type, **values})
if query_rows:
    st.dataframe(pd.DataFrame(query_rows), use_container_width=True)
else:
    st.info("No query-type metrics were generated.")

st.subheader("Launch Readiness Memo")
memo = reports / "07_launch_readiness_memo.md"
if memo.exists():
    st.markdown(memo.read_text(encoding="utf-8"))
