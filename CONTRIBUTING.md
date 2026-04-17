# Contributing to ALIENS

Thank you for your interest in contributing. ALIENS is an adaptive literacy system used in educational settings with real learners. Contributions that affect the engine's diagnostic logic, scaffold generation, or learner state updates carry instructional consequences and are reviewed with corresponding care.

---

## Before you start

- Read the [README](README.md) fully, particularly the [How it works](README.md#how-it-works) and [Deployment guidance](README.md#deployment-guidance) sections.
- Run all four test suites and confirm 791/791 pass before making any changes.
- Check open issues before opening a new one — many edge cases have documented histories.

---

## Types of contribution

### Bug reports

Open an issue using the bug report template. Include:

- The minimal input that reproduces the problem (`LearnerState`, `CanonicalPassage`, candidate text, and engine config).
- The actual output (diagnosis label, signal values, or error).
- The expected output, with a reference to the specific rule or contract that should have applied.
- Which module and function is implicated (`alien_system.py`, `alien_system_es.py`, or `alien_dyslexia.py`).

Bugs in the deterministic engine (scoring, selection, state update) are high priority. Bugs in the LLM prompt templates are medium priority — they affect quality but the deterministic fallback paths provide safety.

### Bug fixes

- One fix per pull request.
- Every fix must be accompanied by at least one new test in `test_fixes.py` that **fails before the fix and passes after**. Tests that only pass after the fix are the minimum acceptable evidence — they are the authoritative specification of what the bug was.
- Describe the root cause (not just the symptom) in the PR description. The five existing fixes in `test_fixes.py` are the model for this standard.
- Both the English module (`alien_system.py`) and the Spanish module (`alien_system_es.py`) must be patched if the bug exists in both. Check before assuming a fix is module-specific.

### New diagnostic labels or state update rules

The seven diagnosis labels and their state update rules are the system's core instructional logic. Changes here have direct consequences for learner progression.

Proposals to add or modify diagnostic labels must include:

1. A written justification referencing a specific instructional design principle and at least one peer-reviewed source.
2. Worked examples: at least two `LearnerState` + `ReadingSignals` combinations that would be classified differently under the proposed change, with an explanation of why the new classification is instructionally superior.
3. A full pass of all existing tests, plus new tests covering the proposed label.

Pull requests that modify `diagnose_fallback()` or `update_learner_state()` without this documentation are unlikely to be accepted.

### Scaffold dimensions and candidate plan logic

Changes to `build_candidate_plan()`, `ScaffoldProfile`, or the `Level` enum affect what texts are generated for every learner in every cycle. Proposals must explain the instructional rationale for the change and include simulation evidence showing the before/after effect on at least three learner profiles at different bands.

### New language ports

ALIENS is designed to support additional languages. A language port must:

- Implement a language-appropriate readability formula (FK equivalent) with documented validity evidence for the target language.
- Implement a language-appropriate stemmer and stopword list.
- Pass an equivalent test suite of at least 100 tests covering the core engine functions.
- Follow the `alien_system_es.py` architecture exactly, including importing `ALIENError` from `alien_system` rather than redefining it.
- Include a `LANGUAGE_PORT_NOTES.md` documenting any linguistic decisions (e.g. handling of agglutination, diacritics, orthographic conventions).

### Documentation

Documentation improvements are welcome without the constraints above. Corrections to factual errors in the README are especially welcome — please open a PR directly rather than an issue.

---

## Code standards

**Style.** The codebase follows the conventions already present in `alien_system.py`. There is no linter configuration; read the existing code and match its style. Key conventions:

- Frozen dataclasses for all data types.
- Type annotations throughout.
- `ratio()` for all division where the denominator may be zero.
- Deterministic paths never raise `NotImplementedError` or depend on external state.
- LLM calls always have a deterministic fallback.

**No new dependencies.** ALIENS uses the Python standard library only. Pull requests that add external dependencies (including `numpy`, `scipy`, or any NLP library) will not be merged. This constraint is intentional: the system must be deployable in environments with no internet access and minimal Python installations.

**Symmetry between EN and ES.** Any change to a shared algorithm in `alien_system.py` must be evaluated for `alien_system_es.py` and applied there if relevant. The DEGRADED log differentiation (Fix 4) and `split_sentences` regex extension (Fix 5) are examples of fixes that applied to both modules. Asymmetric fixes require explicit justification.

**Tests must be in-process.** All tests use the standard `assert`-equivalent pattern already in the test files. No `pytest`, `unittest`, or external test runner is introduced.

---

## Pull request checklist

Before submitting:

- [ ] All four existing test suites pass: 389 + 184 + 132 + 86 = 791 tests.
- [ ] New tests added for any new behaviour or bug fix.
- [ ] Both EN and ES modules patched if the change affects shared logic.
- [ ] `CHANGELOG.md` updated with a brief description of the change under the `[Unreleased]` heading.
- [ ] No new external dependencies introduced.
- [ ] PR description includes: what changed, why, and what the instructional consequence is.

---

## Reporting vulnerabilities

See [SECURITY.md](SECURITY.md).

---

## Code of conduct

This project is used in educational settings with children. Contributors are expected to engage with professionalism and to centre the interests of learners — particularly those with learning differences — in all design discussions. Contributions that would introduce systematic bias against any learner group, or that would allow misclassification of a dyslexic learner's comprehension as a deficit, will not be merged regardless of technical correctness.
