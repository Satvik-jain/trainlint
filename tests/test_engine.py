from pathlib import Path

import pytest

from trainlint.engine import lint_paths, lint_source, select_rules

LEAKY = "scaler.fit(X_test)\n"


def test_ignore_comment_suppresses_all_rules_on_the_line():
    assert not lint_source("scaler.fit(X_test)  # trainlint: ignore\n", "f.py")


def test_ignore_comment_with_matching_code():
    assert not lint_source("scaler.fit(X_test)  # trainlint: ignore[TL002]\n", "f.py")


def test_ignore_comment_for_other_code_does_not_suppress():
    assert lint_source("scaler.fit(X_test)  # trainlint: ignore[TL001]\n", "f.py")


def test_columns_are_in_characters_not_bytes():
    (finding,) = lint_source("name = 'é'; scaler.fit(X_test)\n", "f.py")
    assert finding.col == 13  # the byte offset would be 14
    assert finding.end_col == 13 + len("scaler.fit(X_test)")


def test_select_and_ignore():
    assert [r.id for r in select_rules(["tl002"])] == ["TL002"]
    assert "TL002" not in [r.id for r in select_rules(ignore=["TL002"])]


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError, match="TL999"):
        select_rules(["TL999"])


def test_directories_are_walked_and_noise_dirs_skipped(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "train.py").write_text(LEAKY)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text(LEAKY)
    (tmp_path / "notes.txt").write_text(LEAKY)

    report = lint_paths([tmp_path])

    assert report.files_checked == 1
    assert [f.path.rsplit("/", 1)[-1] for f in report.findings] == ["train.py"]


def test_unparseable_file_is_skipped_not_fatal(tmp_path: Path):
    (tmp_path / "old.py").write_text("print 'python 2'\n")
    (tmp_path / "new.py").write_text(LEAKY)

    report = lint_paths([tmp_path])

    assert report.files_checked == 1
    assert [path.rsplit("/", 1)[-1] for path, _ in report.skipped] == ["old.py"]
    assert len(report.findings) == 1


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        lint_paths([tmp_path / "nope.py"])
