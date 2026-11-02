import re
from pathlib import Path

import pytest

from trainlint.cli import main
from trainlint.rules import ALL_RULES
from trainlint.skills_install import bundled_skills, install

SKILLS = bundled_skills()
EXPECTED = {"ml-data-leakage", "ml-training-loop", "ml-metrics-eval", "ml-checkpoint-resume"}


def frontmatter(skill: Path) -> dict[str, str]:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    block = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in block.splitlines())


def test_all_skills_are_bundled():
    assert {s.name for s in SKILLS} == EXPECTED


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_skill_follows_the_agent_skills_format(skill):
    fields = frontmatter(skill)
    assert fields["name"] == skill.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name) and len(skill.name) <= 64
    assert 0 < len(fields["description"]) <= 1024
    assert len((skill / "SKILL.md").read_text(encoding="utf-8").splitlines()) < 500


def test_skills_only_cite_rules_that_exist_and_cover_every_rule():
    known = {rule.id for rule in ALL_RULES}
    cited = set()
    for skill in SKILLS:
        cited |= set(re.findall(r"TL\d{3}", (skill / "SKILL.md").read_text(encoding="utf-8")))
    assert cited == known


def test_install_copies_then_skips_then_overwrites(tmp_path: Path):
    installed, skipped = install(tmp_path)
    assert {p.name for p in installed} == EXPECTED and not skipped
    assert (tmp_path / "ml-training-loop" / "SKILL.md").is_file()

    (tmp_path / "ml-training-loop" / "SKILL.md").write_text("local edit")
    installed, skipped = install(tmp_path)
    assert not installed and len(skipped) == len(EXPECTED)
    assert (tmp_path / "ml-training-loop" / "SKILL.md").read_text() == "local edit"

    installed, _ = install(tmp_path, force=True)
    assert len(installed) == len(EXPECTED)
    assert (tmp_path / "ml-training-loop" / "SKILL.md").read_text() != "local edit"


def test_install_command(tmp_path: Path, capsys):
    assert main(["skills", "install", "--dest", str(tmp_path)]) == 0
    assert "ml-data-leakage" in capsys.readouterr().out
    assert (tmp_path / "ml-data-leakage" / "SKILL.md").is_file()


def test_install_command_defaults_to_project_claude_folder(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["skills", "install"]) == 0
    assert (tmp_path / ".claude" / "skills" / "ml-metrics-eval" / "SKILL.md").is_file()
