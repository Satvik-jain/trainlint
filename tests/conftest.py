import textwrap
from collections.abc import Callable

import pytest

from trainlint.engine import Finding, lint_source, select_rules


@pytest.fixture
def lint() -> Callable[..., list[Finding]]:
    """lint(source, "TL001", ...) runs only the named rules (all rules if none given)."""

    def run(source: str, *rule_ids: str) -> list[Finding]:
        rules = select_rules(rule_ids)
        return lint_source(textwrap.dedent(source), "snippet.py", rules)

    return run
