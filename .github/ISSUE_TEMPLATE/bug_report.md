---
name: Bug report
about: Report a reproducible defect in the engine
labels: bug
---

## Module and function

Which module and function is implicated?
- [ ] `alien_system.py`
- [ ] `alien_system_es.py`
- [ ] `alien_dyslexia.py`

**Function:** (e.g. `score_candidate`, `update_learner_state`, `diagnose_fallback`)

## Minimal reproduction

Provide the minimum inputs needed to reproduce the bug. Include all of:

```python
# LearnerState
learner = LearnerState(
    learner_id="...",
    current_band=...,
    # ... other fields
)

# CanonicalPassage or source text
source_text = "..."

# EngineConfig (if non-default)
config = EngineConfig(...)

# Candidate or assessment data (if relevant)
```

## Actual output

What did the system produce? (diagnosis label, signal values, error message, etc.)

## Expected output

What should it have produced? Reference the specific rule, contract, or invariant that was violated.

## Which test suite catches this?

- [ ] Existing test fails (which one?)
- [ ] No existing test covers this case

## Additional context

Is this bug present in both the English and Spanish modules? Have you checked `alien_system_es.py`?
