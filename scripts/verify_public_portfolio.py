from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = [
    "docs/assets/architecture.svg",
    "docs/assets/social-preview.png",
    "docs/ARCHITECTURE.md",
    "docs/DEMO_WALKTHROUGH.md",
    "docs/REPOSITORY_MAP.md",
    "docs/PUBLIC_RELEASE_CHECKLIST.md",
    "scripts/public_demo.py",
    "SECURITY.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()

    readme = (repo / "README.md").read_text(encoding="utf-8")
    for value in [
        "<!-- PUBLIC_PORTFOLIO_START -->",
        "<!-- PUBLIC_PORTFOLIO_END -->",
        "combined_feature_only",
        "core_champion_replay",
        "0.312599",
        "0.322034",
        "What makes this more than a demo notebook",
        "docs/assets/architecture.svg",
    ]:
        assert value in readme, value

    for relative in REQUIRED:
        path = repo / relative
        assert path.is_file() and path.stat().st_size > 0, relative

    assert (repo / "docs/assets/social-preview.png").stat().st_size < 1_000_000
    print("Public portfolio contract PASS")


if __name__ == "__main__":
    main()
