import json
from pathlib import Path

from trainlint.cli import main


def write(tmp_path: Path, source: str) -> str:
    path = tmp_path / "train.py"
    path.write_text(source)
    return str(path)


def test_findings_exit_1_and_print_location(tmp_path: Path, capsys):
    path = write(tmp_path, "scaler.fit(X_test)\n")

    assert main(["check", path]) == 1

    out = capsys.readouterr().out
    assert "train.py:1:1: TL002" in out
    assert "^^^^" in out
    assert "help:" in out
    assert "Found 1 problem in 1 file" in out


def test_clean_file_exits_0(tmp_path: Path, capsys):
    path = write(tmp_path, "scaler.fit(X_train)\n")

    assert main(["check", path]) == 0
    assert "No problems found in 1 file." in capsys.readouterr().out


def test_exit_zero_flag(tmp_path: Path):
    assert main(["check", "--exit-zero", write(tmp_path, "scaler.fit(X_test)\n")]) == 0


def test_json_output(tmp_path: Path, capsys):
    main(["check", "--format", "json", write(tmp_path, "scaler.fit(X_test)\n")])

    payload = json.loads(capsys.readouterr().out)
    assert payload["files_checked"] == 1
    assert [(f["rule"], f["line"], f["column"]) for f in payload["findings"]] == [("TL002", 1, 1)]


def test_color_always_emits_ansi_and_never_does_not(tmp_path: Path, capsys):
    path = write(tmp_path, "scaler.fit(X_test)\n")

    main(["check", "--color", "always", path])
    assert "\x1b[" in capsys.readouterr().out

    main(["check", "--color", "never", path])
    assert "\x1b[" not in capsys.readouterr().out


def test_select_filters_rules(tmp_path: Path):
    path = write(tmp_path, "scaler.fit(X_test)\n")
    assert main(["check", "--select", "TL001", path]) == 0
    assert main(["check", "--ignore", "TL002", path]) == 0


def test_unknown_rule_exits_2(tmp_path: Path, capsys):
    assert main(["check", "--select", "TL999", write(tmp_path, "x = 1\n")]) == 2
    assert "unknown rule TL999" in capsys.readouterr().err


def test_missing_path_exits_2(tmp_path: Path, capsys):
    assert main(["check", str(tmp_path / "missing.py")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_rules_lists_ids(capsys):
    assert main(["rules"]) == 0
    assert "TL001" in capsys.readouterr().out
