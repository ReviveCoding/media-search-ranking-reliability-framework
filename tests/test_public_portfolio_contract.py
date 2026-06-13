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

    public_text_paths = [
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]

    forbidden = (
        "\u00c3\u00a2",
        "\u00e2\u2020\u2019",
        "\u00e2\u20ac",
        "\ufffd",
    )

    assert "\u2192" in readme

    finalization_generator = (
        ROOT / "scripts" / "finalize_repository.py"
    ).read_text(encoding="utf-8")

    assert "multimodal media-search" not in finalization_generator
    assert (
        "media-search retrieval, ranking, and reliability framework"
        in finalization_generator
    )
    assert (
        "media-search retrieval and ranking framework"
        in finalization_generator
    )
    assert (
        "## Frozen benchmark text-encoding limitation"
        in finalization_generator
    )

    for public_path in public_text_paths:
        content = public_path.read_text(encoding="utf-8")

        assert not any(
            token in content
            for token in forbidden
        ), f"Mojibake detected in {public_path}"


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
