# ALIENS — Claude Code Guide

## Project overview

ALIENS (Adaptive Literacy Instruction Engagement Network) is a hybrid deterministic-LLM system for adaptive reading instruction. It generates learner-appropriate passage variants from a single canonical source, scores them against hard structural constraints, runs a diagnostic assessment, and updates learner state — all deterministically except for the 6 LLM calls per cycle.

**Standard library only. No external dependencies.**

---

## Repository structure

```
alien_system.py              # Core English engine          (2,349 lines)
alien_system_es.py           # Spanish edition              (2,867 lines)
alien_dyslexia.py            # Dyslexia extension             (897 lines)
test_alien.py                # English unit tests  — 389 tests
test_alien_es.py             # Spanish unit tests  — 184 tests
test_alien_dyslexia.py       # Dyslexia unit tests — 132 tests
test_fixes.py                # Fix regression suite — 86 tests
test_system.py               # System integration  — 167 tests
PRODUCTION_READINESS_REPORT.md
ALIEN_API_INTEGRATION_GUIDE.md
README.md / CHANGELOG.md / CONTRIBUTING.md / SECURITY.md / LICENSE
```

---

## Running tests

All test files are standalone — no test runner required. On Windows, set `PYTHONIOENCODING=utf-8` to avoid Unicode errors from the box-drawing characters used in the test harness.

```bash
PYTHONIOENCODING=utf-8 python test_alien.py           # 389 tests
PYTHONIOENCODING=utf-8 python test_alien_es.py        # 184 tests
PYTHONIOENCODING=utf-8 python test_alien_dyslexia.py  # 132 tests
PYTHONIOENCODING=utf-8 python test_fixes.py           #  86 tests
PYTHONIOENCODING=utf-8 python test_system.py          # 167 tests
```

**All 958 tests must pass before any commit.**

---

## Architecture

### Pipeline (6 LLM calls per cycle)

```
prepare_cycle():
  canonicalize_passage()   → CanonicalPassage          [LLM #1]
  build_candidate_plan()   → list[dict]                [deterministic]
  generate_candidates()    → list[CandidatePassage]    [LLM #2]
  score_candidate()        → DeterministicScores       [deterministic]
  estimate_fit()           → dict[str, FitEstimate]    [LLM #3]
  select_candidate()       → CandidatePassage          [deterministic]
  generate_assessment()    → AssessmentPackage         [LLM #4]

complete_cycle():
  score_mcq()              → float                     [deterministic]
  score_retell()           → dict                      [LLM #5, deterministic fallback]
  build_reading_signals()  → ReadingSignals            [deterministic]
  diagnose_outcome()       → DiagnosisLabel            [LLM #6, deterministic fallback]
  update_learner_state()   → LearnerState              [deterministic]
```

### Key types

- **`LearnerState`** — 11-field frozen dataclass: `learner_id`, `current_band`, `vocabulary_need`, `syntax_need`, `cohesion_need`, `support_dependence`, `readiness_to_increase`, `recent_outcomes`, `target_band`, `entry_band`, `cycles_on_passage`
- **`Level`** — `LOW / MEDIUM / HIGH` with `.up()` / `.down()` (clamped at boundaries)
- **`DiagnosisLabel`** — 7 values: `underchallenged`, `well_calibrated`, `successful_but_support_dependent`, `vocabulary_barrier`, `syntax_barrier`, `cohesion_inference_barrier`, `overloaded`
- **`SelectionMode`** — `VALIDATED` (all constraints pass) or `DEGRADED` (surface warnings only; blocking failures escalate to `ALIENError`)
- **`EngineConfig`** — frozen dataclass controlling all thresholds; `fk_tolerance` is the most commonly tuned field

### Deterministic vs LLM responsibilities

| Layer | Responsibility |
|---|---|
| LLM | Generation (passage variants, assessment), qualitative fit estimation, retell scoring, diagnosis labelling |
| Deterministic | All hard constraints, candidate selection, MCQ scoring, learner state updates, fallback diagnosis |

LLM failures in `score_retell` and `diagnose_outcome` are caught and resolved by deterministic fallback — the cycle always completes.

### DEGRADED selection

A candidate enters **DEGRADED** mode when it fails only surface-measure checks (FK grade out of tolerance, length deviation, vocabulary coverage). A candidate with any **blocking failure** (self-audit `meaning_preserved=False`, meaning coverage below threshold, sequence violation) is excluded entirely. If all candidates are blocked, `ALIENError` is raised.

---

## Symmetry rule: EN and ES

`alien_system_es.py` is a self-contained port. Any bug fix or algorithm change that applies to `alien_system.py` must be evaluated for `alien_system_es.py` and applied there if relevant. The modules share only `ALIENError`:

```python
# In alien_system_es.py:
from alien_system import ALIENError
```

This means `alien_system_es.ALIENError is alien_system.ALIENError` is `True`. Do not break this.

EN and ES define their own `Level`, `DiagnosisLabel`, `SelectionMode`, and all dataclasses independently. They are structurally identical but not the same objects.

---

## Test harness conventions

All test files use a lightweight custom harness — no `pytest` or `unittest`:

```python
def ck(name, cond, detail=""):     # assert with pass/fail tracking
def ck_raises(name, fn, exc):      # assert exception is raised
def group(name):                   # section header
```

- `test_fixes.py` contains a harmless `sys.path.insert(0, "/home/claude")` dev artifact (does nothing on Windows).
- New bug fixes go in `test_fixes.py` with a test that fails before the fix and passes after.
- New integration scenarios go in `test_system.py`.

---

## Documentation accuracy rule

When test counts, line counts, or file lists change, update all of:

- `README.md` — badge, repository structure block, Testing section, summary line
- `CHANGELOG.md` — test suite list and total
- `CONTRIBUTING.md` — pre-work checklist and PR checklist

Current counts: **8 Python files, 11,253 lines, 958 tests.**

---

## What not to do

- Do not add external dependencies. Standard library only.
- Do not modify `diagnose_fallback()` or `update_learner_state()` without a written instructional justification and peer-reviewed reference (see CONTRIBUTING.md).
- Do not set `decoding_disability = True` in tests or examples without explicitly noting it requires practitioner authorisation in production.
- Do not break the `ALIENError` shared identity between EN and ES modules.
- Do not use `eval()` anywhere — all JSON parsing uses `json.loads()`.
