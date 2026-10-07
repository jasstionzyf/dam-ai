"""Placeholder test suite (Phase 0 skeleton)."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

REQUIRED_DIRS = ["tagger", "embedder", "classic", "registry", "server", "deploy", "docs"]


def test_top_level_dirs_exist():
    for d in REQUIRED_DIRS:
        assert (ROOT / d).is_dir(), f"missing top-level dir: {d}"


def test_repo_meta_files_exist():
    for f in ["README.md", "LICENSE", "pyproject.toml", "PLAN.md"]:
        assert (ROOT / f).is_file(), f"missing {f}"


def test_license_is_agpl():
    head = (ROOT / "LICENSE").read_text()[:100]
    assert "AFFERO" in head
