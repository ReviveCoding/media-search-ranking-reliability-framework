from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_readme_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- PUBLIC_PORTFOLIO_START -->" in readme
    assert "<!-- PUBLIC_PORTFOLIO_END -->" in readme
    assert "combined_feature_only" in readme
    assert "core_champion_replay" in readme
    assert "0.312599" in readme
    assert "0.322034" in readme


def test_public_assets_exist() -> None:
    for relative in [
        "docs/assets/architecture.svg",
        "docs/assets/social-preview.png",
        "docs/ARCHITECTURE.md",
        "docs/DEMO_WALKTHROUGH.md",
        "docs/REPOSITORY_MAP.md",
        "docs/PUBLIC_RELEASE_CHECKLIST.md",
        "scripts/public_demo.py",
        "SECURITY.md",
    ]:
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size > 0


def test_social_preview_size() -> None:
    assert (ROOT / "docs/assets/social-preview.png").stat().st_size < 1_000_000
