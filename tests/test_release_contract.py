from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "frozen_quality_ablation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_promotion_contract() -> None:
    summary = load_json(ARTIFACTS / "frozen_ablation_summary.json")
    diagnostics = load_json(ARTIFACTS / "frozen_replay_diagnostics.json")

    assert summary["champion"]["run_name"] == "combined_feature_only"
    assert summary["champion"]["eligible_for_promotion"] is True
    assert summary["canonical_core_baseline"]["run_name"] == "core_champion_replay"
    assert diagnostics["canonical_contract_validated"] is True
    assert diagnostics["strict_replay_claim"] is False


def test_documentation_is_synchronized() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- FINAL_RESULTS_START -->" in readme
    assert "<!-- FINAL_RESULTS_END -->" in readme
    assert "`combined_feature_only`" in readme
    assert "`0.322034`" in readme

    for relative in [
        "docs/FINAL_RESULTS.md",
        "docs/REPRODUCIBILITY.md",
        "docs/CLAIM_BOUNDARIES.md",
    ]:
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size > 0
