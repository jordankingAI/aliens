## What changed

Brief description of the change.

## Why

What problem does this solve, or what behaviour does it add? For diagnostic logic changes, include a reference to the instructional design principle and peer-reviewed source (required — see CONTRIBUTING.md).

## Type of change

- [ ] Bug fix (include root cause, not just symptom)
- [ ] New feature / diagnostic label / scaffold dimension
- [ ] Documentation update
- [ ] New language port

## Checklist

- [ ] All five test suites pass: `389 + 184 + 132 + 86 + 167 = 958 tests`
- [ ] New tests added for any new behaviour or bug fix (in `test_fixes.py` for bugs, `test_system.py` for integration scenarios)
- [ ] Both `alien_system.py` and `alien_system_es.py` patched if the change affects shared logic
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No new external dependencies introduced
- [ ] If diagnostic logic changed: worked examples included showing before/after classification

## Instructional consequence

What is the effect on learner progression? (Required for changes to `diagnose_fallback`, `update_learner_state`, or `build_candidate_plan`.)
