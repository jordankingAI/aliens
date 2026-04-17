# ALIENS — Adaptive Literacy Instruction in English and Spanish

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-958%20passing-brightgreen)](#testing)
[![CI](https://github.com/jordankingAI/aliens/actions/workflows/tests.yml/badge.svg)](https://github.com/jordankingAI/aliens/actions/workflows/tests.yml)

ALIENS is a hybrid deterministic-LLM system for adaptive reading instruction. It takes a single canonical source passage, generates learner-appropriate variants that preserve the same protected meaning units and lesson-critical vocabulary, and diagnoses each learner's reading outcome to guide the next cycle. It operates in English and Spanish and includes a dedicated extension for learners with dyslexia.

---

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Learner model](#learner-model)
- [Diagnosis labels](#diagnosis-labels)
- [Dyslexia extension](#dyslexia-extension)
- [Spanish edition](#spanish-edition)
- [Engine configuration](#engine-configuration)
- [Testing](#testing)
- [LLM backend](#llm-backend)
- [Deployment guidance](#deployment-guidance)
- [Pedagogical basis](#pedagogical-basis)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

A classroom using ALIENS works like this:

1. A curriculum author writes one source passage at the level the subject demands.
2. ALIENS extracts the passage's protected meaning units, causal sequence, and required vocabulary — the **canonical representation** — in a single LLM call.
3. For each learner, ALIENS generates a family of candidate variants tuned to that learner's band, scaffold profile, and need dimensions.
4. The **deterministic engine** scores every candidate against the canonical constraints, selects the best one, and delivers it.
5. The learner reads their variant and completes a six-item diagnostic assessment.
6. ALIENS scores the assessment, constructs a signal profile, and classifies the outcome into one of seven diagnostic labels.
7. The learner state is updated deterministically. The next cycle adapts.

Every learner reads about the same events, encounters the same required vocabulary, and can participate in the same class discussion. What changes is the surface complexity of the text through which they access that shared content.

---

## Repository structure

```
aliens/
├── alien_system.py              # Core engine — English        (2,349 lines)
├── alien_system_es.py           # Spanish edition              (2,867 lines)
├── alien_dyslexia.py            # Dyslexia extension module      (897 lines)
├── test_alien.py                # English test suite  — 389 tests
├── test_alien_es.py             # Spanish test suite  — 184 tests
├── test_alien_dyslexia.py       # Dyslexia test suite — 132 tests
├── test_fixes.py                # Fix regression suite  — 86 tests
├── test_system.py               # System integration test — 167 tests
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── SECURITY.md
├── ALIEN_API_INTEGRATION_GUIDE.md
├── PRODUCTION_READINESS_REPORT.md
├── DESIGN_JUSTIFICATION.md
└── LICENSE
```

**Total: 11,253 lines across eight Python files. 958 tests, all passing. Standard library only — no external dependencies.**

---

## How it works

### Phase 1 — Prepare

```
canonicalize_passage()   → CanonicalPassage          [1 LLM call]
build_candidate_plan()   → list[dict]                [deterministic]
generate_candidates()    → list[CandidatePassage]    [1 LLM call]
score_candidate()        → DeterministicScores       [deterministic, per candidate]
estimate_fit()           → dict[str, FitEstimate]    [1 LLM call]
select_candidate()       → CandidatePassage          [deterministic]
generate_assessment()    → AssessmentPackage         [1 LLM call]
```

### Phase 2 — Complete

```
score_mcq()              → float                     [deterministic]
score_retell()           → dict                      [1 LLM call, deterministic fallback]
build_reading_signals()  → ReadingSignals            [deterministic]
diagnose_outcome()       → DiagnosisLabel            [1 LLM call, deterministic fallback]
update_learner_state()   → LearnerState              [deterministic]
```

**6 LLM calls per cycle.** Every hard constraint — meaning coverage, sequence validation, vocabulary coverage, candidate selection, MCQ scoring, learner state updates — is deterministic. LLM calls handle open-ended generation and qualitative judgment; the engine enforces correctness.

### Canonical invariants

The deterministic engine **blocks** any candidate that:

- Drops a required meaning unit (meaning coverage below the scaled threshold)
- Presents meaning units out of the protected sequence
- Omits a required vocabulary term
- Fails the LLM's own self-audit (`meaning_preserved = False`)

Candidates with only surface-measure warnings (FK outside tolerance, length deviation) are eligible for **DEGRADED selection** — the best available candidate is used and a differentiated log message is written, distinguishing FK-only degradation (expected for academic vocabulary at lower bands) from structural degradation (LLM generation failure requiring quality review).

---

## Quick start

### Requirements

- Python 3.11+
- Standard library only (`re`, `json`, `logging`, `dataclasses`, `uuid`, `time`)
- An LLM backend implementing the `LLMBackend` protocol

### Installation

```bash
git clone https://github.com/jordankingAI/aliens.git
cd aliens
python test_alien.py           # 389 tests
python test_alien_es.py        # 184 tests
python test_alien_dyslexia.py  # 132 tests
python test_fixes.py           #  86 tests
python test_system.py          # 167 tests — full system integration
```

### Minimal usage — English

```python
from alien_system import AdaptiveReadingSystem, LearnerState, Level, ReadingTelemetry

class MyLLM:
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        # call your LLM API here and return parsed JSON
        ...

system  = AdaptiveReadingSystem(llm=MyLLM())
learner = LearnerState(
    learner_id="student_001",
    current_band=5.0,
    vocabulary_need=Level.HIGH,
    syntax_need=Level.MEDIUM,
)

# Prepare cycle — 4 LLM calls
prep = system.prepare_cycle(
    source_text="Your source passage here...",
    passage_id="passage_001",
    instructional_objective="Understand the main theme.",
    learner=learner,
)
# prep.selected_candidate.text  → deliver this to the learner
# prep.assessment.items         → present these questions

# Complete cycle — 2 LLM calls
outcome = system.complete_cycle(
    learner=prep.prepared_learner,  # journey-initialised state from prepare_cycle
    prep=prep,
    learner_answers={
        "Q1": "B", "Q2": "C", "Q3": "A", "Q4": "B",
        "Q5": "Learner's retell text here...",
        "Q6": 3,
    },
    telemetry=ReadingTelemetry(
        fluency_score=0.75,
        hint_use_rate=0.15,
        reread_count=2,
        completion=True,
    ),
)

print(outcome.diagnosis)        # e.g. DiagnosisLabel.WELL_CALIBRATED
print(outcome.updated_learner)  # updated LearnerState — store for next cycle
```

### Dyslexia-aware usage

```python
from alien_dyslexia import (
    DyslexiaAwareSystem, DyslexicReadingTelemetry, seed_dyslexic_learner
)

# CRITICAL: seed from LISTENING COMPREHENSION, not a decoding or silent-reading score.
# Seeding from a decoding score is the most damaging deployment error.
learner = seed_dyslexic_learner(
    learner_id="student_002",
    comprehension_band=7.5,
    vocabulary_need=Level.MEDIUM,
)

system = DyslexiaAwareSystem(llm=MyLLM())

telemetry = DyslexicReadingTelemetry(
    fluency_score=0.29,     # slow decoding — not comprehension difficulty
    hint_use_rate=0.51,     # pronunciation lookups — not meaning scaffolding
    reread_count=10,
    completion=False,       # timing constraint — not cognitive overload
    oral_retell_text="The learner's spoken retell transcribed here...",
)
```

---

## Learner model

```python
@dataclass(frozen=True)
class LearnerState:
    learner_id:             str
    current_band:           float          # Flesch-Kincaid grade equivalent
    vocabulary_need:        Level          # LOW / MEDIUM / HIGH
    syntax_need:            Level
    cohesion_need:          Level
    support_dependence:     Level
    readiness_to_increase:  Level
    recent_outcomes:        tuple[DiagnosisLabel, ...]  # rolling window (default 3)
    target_band:            float | None   # source FK of current passage
    entry_band:             float | None   # band when this passage began
    cycles_on_passage:      int
```

`Level` is an ordered three-value enum: `LOW < MEDIUM < HIGH`. Each field exposes `.up()` and `.down()` methods capped at the enum boundaries. Learner state updates are fully deterministic and apply the same rules on every cycle with no judgment calls.

---

## Diagnosis labels

| Label | Primary trigger | Band | Key need update |
|---|---|---|---|
| `underchallenged` | comp ≥ 0.85 · fluency ≥ 0.75 · hints ≤ 0.10 · retell ≥ 0.75 | +1 step | readiness → HIGH |
| `well_calibrated` | adequate completion, no barrier | — | readiness UP |
| `successful_but_support_dependent` | comp ≥ 0.70, hints ≥ 0.30 | — | support_dependence UP |
| `cohesion_inference_barrier` | comp ≥ 0.70, inference < 0.55 | — | cohesion_need UP |
| `vocabulary_barrier` | comp < 0.70, vocab_need ≥ syntax_need | — | vocabulary_need UP |
| `syntax_barrier` | comp < 0.70, syntax_need > vocab_need | −1 step if severe | syntax_need UP |
| `overloaded` | comp < 0.50, OR incomplete, OR (hints ≥ 0.30 AND retell < 0.50) | −1 step | all needs UP |

Diagnosis is LLM-primary with a deterministic fallback. If the LLM call fails or returns an invalid label, the engine applies the rule table above without interrupting the cycle.

---

## Dyslexia extension

`alien_dyslexia.py` is a zero-modification wrapper around `alien_system.py` — no changes to the core engine. It adds three layers:

**Data types** — `DyslexicLearnerState`, `DyslexicReadingTelemetry` (with `oral_retell_text`), `DyslexicReadingSignals` (raw/adjusted signal pairs).

**Engine** — `DyslexiaAwareDeterministicEngine` adds a `decoding_support` slot to the candidate plan and checks for `decoding_barrier` before applying standard diagnostic labels.

**Orchestrator** — `DyslexiaAwareSystem` applies three signal adjustments when `decoding_disability = True` AND `comprehension ≥ 0.70`:

| Signal | Adjustment | Rationale |
|---|---|---|
| `fluency_score` | `max(raw, comprehension × 0.9)` | Slow decoding ≠ poor understanding |
| `hint_use_rate` | `raw × 0.5` | Pronunciation lookups ≠ meaning scaffolding |
| `completion` | `True` | Timing constraint ≠ cognitive overload |

Without these corrections, a dyslexic learner with grade-level comprehension and slow decoding receives the same raw signal profile as a learner who is genuinely overloaded. The system would diagnose `overloaded`, drop the band, and consign the learner to material well below their intellectual level. The adjustments prevent this misclassification.

The `oral_retell_text` field in `DyslexicReadingTelemetry` allows a transcribed spoken retell to be scored in place of the written Q5 response, which dyslexia affects directly.

---

## Spanish edition

`alien_system_es.py` is a self-contained Spanish port sharing no runtime dependencies with the English module except `ALIENError`.

| Feature | English | Spanish |
|---|---|---|
| Readability | Flesch-Kincaid Grade Level | Szigriszt-Pazos Perspicuity Index (1992) |
| Syllable counting | English rules | Spanish phonological rules (vowel nuclei) |
| Stemmer | English inflection rules | Spanish inflection rules (gender, number, verb morphology) |
| Stopword list | 76 words | 176 words including gendered variants |
| Negation set | `no, not, never, none…` | `no, nunca, jamás, nada, nadie, ningún…` |
| DEGRADED log language | English | Spanish |
| `ALIENError` | Defined in this module | `from alien_system import ALIENError` |

`alien_system_es.ALIENError is alien_system.ALIENError` evaluates to `True`. A single `except ALIENError` handler catches errors from both modules — essential for any language-routing pipeline.

---

## Engine configuration

```python
@dataclass(frozen=True)
class EngineConfig:
    band_step:                      float = 0.8
    # FK tolerance — see domain guidance below
    fk_tolerance:                   float = 1.2
    overall_meaning_threshold:      float = 0.75  # scales down with passage distance
    vocabulary_threshold:           float = 0.85
    length_deviation_threshold:     float = 0.40
    severe_comprehension_threshold: float = 0.50
    low_hint_use_threshold:         float = 0.10
    high_hint_use_threshold:        float = 0.30
    history_limit:                  int   = 3
```

### FK tolerance by domain

| Domain | Recommended `fk_tolerance` |
|---|---|
| General / narrative fiction | 1.2 – 2.0 (default covers most cases) |
| Academic / technical content | 2.5 – 4.0 |
| Highly specialised content | 4.0 – 6.0 (treat FK as advisory only) |

For academically dense passages, required domain vocabulary is inherently polysyllabic regardless of sentence simplicity. Rewrites for lower-band learners will consistently score FK above the target band. This is a property of the domain vocabulary, not a generation failure. When all candidates are DEGRADED solely due to FK/length warnings, `prepare_cycle` logs a specific `"DEGRADED (FK/length surface warnings only)"` message to distinguish it from structural DEGRADED (missing meaning units or broken sequence).

---

## Testing

```bash
python test_alien.py           # 389 tests — full English engine
python test_alien_es.py        # 184 tests — full Spanish engine
python test_alien_dyslexia.py  # 132 tests — dyslexia extension
python test_fixes.py           #  86 tests — fix regression suite
python test_system.py          # 167 tests — system integration (EN + ES pipelines end-to-end)

# Total: 958 tests. Standard library only. No test runner required.
```

The fix regression suite covers five specific repairs validated against the bugs discovered during end-to-end simulation:

| Fix | What it tests |
|---|---|
| 1 | Multi-word anchors split word-by-word (15 anchor phrases, EN + ES) |
| 2 | `sentence_unit_match_score` returns `0.0` not `1.0` for empty `unit_toks` |
| 3 | `ALIENError` shared identity across EN and ES modules |
| 4 | DEGRADED log differentiates FK-only from structural failures |
| 5 | `split_sentences` handles terminal punctuation inside quotation marks |

---

## LLM backend

```python
class LLMBackend(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return parsed JSON matching the requested output contract."""
        ...
```

The `TaskRoutingMockLLM` class is provided for testing. It routes by the `task` key in the JSON user prompt and supports error injection for testing fallback paths:

```python
from alien_system import TaskRoutingMockLLM

mock = TaskRoutingMockLLM(
    responses={
        "canonicalize_passage": {...},
        "generate_candidates":  {...},
        "estimate_fit":         {...},
        "generate_assessment":  {...},
        "score_retell":         {...},
        "diagnose_outcome":     {...},
    },
    error_on_tasks={"score_retell"},  # test deterministic fallback
)
```

All six task types have system prompts in `PromptLibrary`. Output contracts are validated before parsing; a `ValidationError` is raised on contract violations.

---

## Deployment guidance

### Minimum viable integration

1. Implement `LLMBackend.complete_json()` for your provider.
2. Seed each learner with `LearnerState` (standard) or `seed_dyslexic_learner()` (dyslexic). For dyslexic learners, `comprehension_band` must come from a listening comprehension instrument, not a silent reading or decoding score.
3. Store `outcome.updated_learner` between sessions. All persistence is the caller's responsibility — ALIENS holds no state.
4. Set `fk_tolerance` in `EngineConfig` based on your content domain.

### What ALIENS does not provide

- **A normed reading assessment.** `current_band` is an instructional working estimate updated by performance signals, not a validated Lexile or DRA score.
- **A dyslexia screening or diagnostic tool.** `decoding_disability = True` must be set by a qualified practitioner based on a formal assessment. ALIENS uses the flag to prevent misclassification; it does not detect dyslexia.
- **A teacher dashboard or reporting interface.** ALIENS is a backend engine.
- **Factual accuracy validation of generated content.** The system validates structural contracts — meaning units, sequence, vocabulary. Human review of generated passages is strongly recommended before classroom deployment.

### Data and privacy

ALIENS stores no learner data. The `LearnerState` is a frozen dataclass passed in and returned by value. No network calls are made outside the `LLMBackend` interface the caller provides.

---

## Pedagogical basis

ALIENS implements five evidence-based instructional design principles:

1. **Zone of Proximal Development** (Vygotsky, 1978) — the band system, candidate plan, and push/safety-net slot structure.
2. **Comprehensible Input / i+1** (Krashen, 1985) — the five scaffold dimensions and candidate generator prompt contracts.
3. **Scaffolding and Gradual Release** (Wood, Bruner & Ross, 1976; Pearson & Gallagher, 1983) — `support_dependence` tracking and the withdrawal pathway through `well_calibrated` cycles.
4. **Formative Assessment for Learning** (Black & Wiliam, 1998; Wiliam, 2011) — the six-item diagnostic, seven diagnosis labels, and immediate deterministic state update.
5. **Curriculum Coherence / Shared Content Principle** (Schmidt et al., 2005; Porter, 2002) — the canonical passage contract ensures every learner variant covers the same meaning units, in the same sequence, with the same required vocabulary.

For the full theoretical justification, empirical basis, and mapping of each principle to specific system components, see [DESIGN_JUSTIFICATION.md](DESIGN_JUSTIFICATION.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License — Copyright (c) 2026 Jordan King. See [LICENSE](LICENSE).
