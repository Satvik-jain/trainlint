# Contributing

Reports of false positives and missed bugs are the most useful thing you can send. A rule that
cries wolf gets switched off, so precision matters more here than coverage.

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer.

```console
git clone https://github.com/Satvik-jain/trainlint
cd trainlint
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run python -m bench
```

## Reporting a problem

- **False positive:** use the "False positive" issue form. Paste the smallest snippet that is
  flagged and say why it is fine.
- **Missed bug:** use the "Rule idea" form. Show the bug and say what in the code gives it away.
  A rule has to work from one file's syntax tree, with no types and no imports resolved.

## Adding or changing a rule

1. Write the check as a function `(tree: ast.Module) -> Iterator[Hit]` in the module for its
   topic under `src/trainlint/rules/`, and register it in that module's `RULES`. Helpers for
   matching calls and names live in `src/trainlint/astutil.py`.
2. Add unit tests with both flagged and clean snippets. Clean snippets matter more: write down
   the ways a sensible script could look like the bug without being it.
3. Add a buggy script and its fixed twin to `bench/buggy/` and `bench/fixed/`. The buggy file
   starts with `# expect: TLxxx`.
4. Add the rule to `docs/rules.md` with one flagged and one clean example. A test runs both.
5. If a bundled skill should mention it, add it to the skill under `src/trainlint/skills/`.
   A test checks that skills only cite rules that exist and that every rule is cited.

A new rule should be quiet on ordinary code. If you can, run it over a few real repositories and
read what it reports before opening the pull request.

## Commits and pull requests

Keep commits small and use Conventional Commit prefixes (`feat`, `fix`, `docs`, `test`, `chore`,
`refactor`, `ci`), imperative mood, no trailing period. Say in the pull request what changed and
how you checked it.
