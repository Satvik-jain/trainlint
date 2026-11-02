import re
import textwrap
from pathlib import Path

import pytest

from trainlint.engine import lint_source, select_rules

DOC = Path(__file__).parent.parent / "docs" / "rules.md"


def examples() -> dict[str, list[str]]:
    text = DOC.read_text(encoding="utf-8")
    result = {}
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        rule = section[:5]
        result[rule] = re.findall(r"```python\n(.*?)```", section, re.S)
    return result


EXAMPLES = examples()


def lint(rule: str, source: str):
    return lint_source(textwrap.dedent(source), "doc.py", select_rules([rule]))


def test_every_rule_is_documented():
    from trainlint.rules import ALL_RULES

    assert sorted(EXAMPLES) == [rule.id for rule in ALL_RULES]


@pytest.mark.parametrize("rule", sorted(EXAMPLES))
def test_documented_examples_behave_as_described(rule):
    flagged, clean = EXAMPLES[rule]
    assert lint(rule, flagged), f"{rule}: the flagged example is not flagged"
    assert not lint(rule, clean), f"{rule}: the clean example is flagged"
