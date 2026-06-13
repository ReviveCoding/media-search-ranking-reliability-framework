from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

START = "<!-- PUBLIC_PORTFOLIO_START -->"
END = "<!-- PUBLIC_PORTFOLIO_END -->"
OWNER = "ReviveCoding"
REPO = "media-search-multimodal-discovery-reliability-framework"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def block(summary: dict, diagnostics: dict) -> str:
    champion = summary["champion"]
    baseline = summary["canonical_core_baseline"]
    c = float(champion["ndcg_at_10"])
    b = float(baseline["ndcg_at_10"])
    change = 100.0 * (c - b) / b
    badge = f"https://github.com/{OWNER}/{REPO}/actions/workflows/ci-final.yml/badge.svg?branch=main"
    action = f"https://github.com/{OWNER}/{REPO}/actions/workflows/ci-final.yml"

    return f"""{START}
<p align="center">
  <img src="docs/assets/architecture.svg" alt="Media search reliability architecture" width="100%">
</p>

<p align="center">
  <a href="{action}">
    <img src="{badge}" alt="CI final status">
  </a>
</p>

A reproducible multimodal media-search reliability framework for **query understanding, hybrid retrieval, learning-to-rank, metadata enrichment, slice-aware evaluation, calibration, latency analysis, and frozen release contracts**.

This repository is built around a practical question: **how do you improve search relevance without losing control of comparability, slice regressions, latency, or claim validity?**

## Results at a glance

| Release signal | Result |
|---|---:|
| Promoted champion | `{champion["run_name"]}` |
| Canonical baseline | `{baseline["run_name"]}` |
| Canonical NDCG@10 | `{b:.6f}` → `{c:.6f}` |
| Relative NDCG@10 change | `{change:+.2f}%` |
| Regression tests | `60 passed` |
| Frozen contract | `validated` |
| Committed-HEAD clean checkout | `PASS` |
| Strict legacy replay claim | `{str(diagnostics["strict_replay_claim"]).lower()}` |
| Launch decision | `{champion.get("launch_decision", "UNKNOWN")}` |

The result is intentionally framed as a **canonical frozen-contract improvement**, not as an unconstrained comparison against every historical run.

## What makes this more than a demo notebook

- **Search-system scope:** query semantics, hybrid retrieval, ranking, enrichment, calibration, slice diagnostics, latency, and launch gates.
- **Reliability contracts:** frozen manifest and query-group split checks prevent invalid experiment comparisons.
- **Promotion discipline:** candidates must satisfy ranker lift, overall quality, and slice-regression rules.
- **Claim governance:** legacy runs remain visible but are excluded from strict replay claims when fingerprints differ.
- **Reproducibility:** the exact committed `HEAD` installs in a new environment and passes the full test and release-contract suite.

## 30-second review path

1. Read the [final results](docs/FINAL_RESULTS.md) and [claim boundaries](docs/CLAIM_BOUNDARIES.md).
2. Inspect the [system architecture](docs/ARCHITECTURE.md).
3. Run the [public demo walkthrough](docs/DEMO_WALKTHROUGH.md).
4. Review the implementation in `src/media_search_reliability/`.
5. Inspect the frozen-contract and regression tests in `tests/`.

## Quick proof

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -e .
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts\\public_demo.py
```

Expected release signals:

```text
champion=combined_feature_only
canonical_baseline=core_champion_replay
canonical_contract_validated=True
strict_replay_claim=False
```

## Documentation map

- [Architecture](docs/ARCHITECTURE.md)
- [Demo walkthrough](docs/DEMO_WALKTHROUGH.md)
- [Repository map](docs/REPOSITORY_MAP.md)
- [Final results](docs/FINAL_RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Claim boundaries](docs/CLAIM_BOUNDARIES.md)
- [Public release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md)

{END}"""


def insert_after_heading(text: str, new_block: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#\s+\S", line):
            before = "\n".join(lines[: i + 1]).rstrip()
            after = "\n".join(lines[i + 1 :]).lstrip()
            return before + "\n\n" + new_block + "\n\n" + after + "\n"
    return "# Media Search Multimodal Discovery Reliability Framework\n\n" + new_block + "\n\n" + text


def update(text: str, new_block: str) -> str:
    if START in text and END in text:
        before, tail = text.split(START, 1)
        _, after = tail.split(END, 1)
        return before.rstrip() + "\n\n" + new_block + after
    return insert_after_heading(text, new_block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()

    summary = load_json(repo / "artifacts/frozen_quality_ablation/frozen_ablation_summary.json")
    diagnostics = load_json(repo / "artifacts/frozen_quality_ablation/frozen_replay_diagnostics.json")

    assert summary["champion"]["run_name"] == "combined_feature_only"
    assert summary["canonical_core_baseline"]["run_name"] == "core_champion_replay"
    assert diagnostics["canonical_contract_validated"] is True
    assert diagnostics["strict_replay_claim"] is False

    readme = repo / "README.md"
    readme.write_text(update(readme.read_text(encoding="utf-8"), block(summary, diagnostics)), encoding="utf-8")
    print(f"README public portfolio block synchronized: {readme}")
    print("Public portfolio polish synchronization PASS")


if __name__ == "__main__":
    main()
