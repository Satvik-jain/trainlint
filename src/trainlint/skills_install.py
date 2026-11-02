"""Copy the bundled Agent Skills into a skills directory."""

from __future__ import annotations

import shutil
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

# Project-relative skill folders that coding agents read.
AGENT_DIRS = {"claude": ".claude/skills", "agents": ".agents/skills"}


def bundled_skills() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


def default_destination(agent: str, user_wide: bool) -> Path:
    base = Path.home() if user_wide else Path.cwd()
    return base / AGENT_DIRS[agent]


def install(destination: Path, force: bool = False) -> tuple[list[Path], list[Path]]:
    """Copy each skill into `destination`. Returns (installed, skipped) target paths."""
    installed: list[Path] = []
    skipped: list[Path] = []
    for skill in bundled_skills():
        target = destination / skill.name
        if target.exists():
            if not force:
                skipped.append(target)
                continue
            shutil.rmtree(target)
        shutil.copytree(skill, target)
        installed.append(target)
    return installed, skipped
