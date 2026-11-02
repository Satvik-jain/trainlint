import pytest

from bench.corpus import load_pairs, rules_found, score

PAIRS = load_pairs()


def test_every_rule_has_a_buggy_example():
    from trainlint.rules import ALL_RULES

    covered = {rule for pair in PAIRS for rule in pair.expected}
    assert covered == {rule.id for rule in ALL_RULES}


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.name)
def test_buggy_file_is_flagged_for_exactly_the_expected_rules(pair):
    assert rules_found(pair.buggy) == pair.expected


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.name)
def test_fixed_twin_is_clean(pair):
    assert not rules_found(pair.fixed)


def test_score_reports_no_problems():
    _, problems = score(PAIRS)
    assert problems == []
