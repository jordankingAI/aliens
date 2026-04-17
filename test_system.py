"""
test_system.py — Comprehensive system integration test for ALIENS v1.0.0

Exercises the full English and Spanish pipelines end-to-end with mocked LLM
calls (TaskRoutingMockLLM).  Complements the four existing unit-focused suites
by testing the orchestrator (AdaptiveReadingSystem) as a whole.

Group structure:
  S01  EN prepare_cycle — CyclePreparation structure and content
  S02  EN complete_cycle — full UNDERCHALLENGED path
  S03  EN complete_cycle — all 7 diagnosis labels
  S04  EN update_learner_state — all 7 label state mutations
  S05  EN journey tracking — begin_passage_journey, cycles_on_passage
  S06  EN DEGRADED — FK-only path (surface warnings, no blocking failures)
  S07  EN DEGRADED — structural path (blocking failures present)
  S08  EN error handling — ValidationError, ALIENError, LLM fallbacks
  S09  ES prepare_cycle — Spanish full cycle structure
  S10  ES complete_cycle — WELL_CALIBRATED path
  S11  ES diagnosis labels and state updates
  S12  ES configuration — SPANISH_CONFIG and readability_grade
  S13  Cross-module — ALIENError identity, shared type compatibility
"""

import sys
import json
import logging
from dataclasses import replace

sys.path.insert(0, "/home/claude")

from alien_system import (
    Level, DiagnosisLabel, SelectionMode,
    LearnerState, ReadingTelemetry, ReadingSignals, AssessmentResult,
    CanonicalPassage, MeaningUnit, SequenceConstraint, VocabularyTerm,
    CandidatePassage, ScaffoldProfile, SelfAudit, DeterministicScores,
    FitEstimate, CyclePreparation, CycleOutcome, EngineConfig,
    AdaptiveReadingSystem, TaskRoutingMockLLM, DeterministicEngine,
    ALIENError, ValidationError,
    flesch_kincaid_grade, meaning_profile, sequence_ok_from_positions,
    score_mcq, normalize_retell_score, weighted_average,
)
import alien_system_es as es_mod
from alien_system_es import (
    AdaptiveReadingSystem as ESSystem,
    TaskRoutingMockLLM as ESMockLLM,
    DeterministicEngine as ESDeterministicEngine,
    LearnerState as ESLearnerState,
    ReadingTelemetry as ESReadingTelemetry,
    CanonicalPassage as ESCanonicalPassage,
    MeaningUnit as ESMeaningUnit,
    SequenceConstraint as ESSequenceConstraint,
    VocabularyTerm as ESVocabularyTerm,
    CandidatePassage as ESCandidatePassage,
    ScaffoldProfile as ESScaffoldProfile,
    SelfAudit as ESSelfAudit,
    Level as ESLevel,
    DiagnosisLabel as ESDiagnosisLabel,
    SelectionMode as ESSelectionMode,
    ALIENError as ES_ALIENError,
    ValidationError as ES_ValidationError,
    SPANISH_CONFIG,
    readability_grade,
    flesch_kincaid_grade as es_fk,
)


# ══════════════════════════════════════════════════════════════════════════════
# Harness
# ══════════════════════════════════════════════════════════════════════════════

passed = failed = 0

def group(name):
    print(f"\n  {'─'*62}")
    print(f"  {name}")
    print(f"  {'─'*62}")

def ck(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"    ✓  {name}")
    else:
        failed += 1
        print(f"    ✗  {name}" + (f"\n         {detail}" if detail else ""))

def ck_raises(name, fn, exc=Exception):
    global passed, failed
    try:
        fn()
        failed += 1
        print(f"    ✗  {name}  (no exception raised)")
    except exc:
        passed += 1
        print(f"    ✓  {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

# ── English ───────────────────────────────────────────────────────────────────

EN_SOURCE = (
    "Johannes Gutenberg invented the printing press around 1440, transforming "
    "the spread of knowledge. Before this, manuscripts were hand-copied and "
    "scarce. Movable metal type allowed fast reproduction of texts. In decades, "
    "presses spread across Europe. This democratization catalyzed the Renaissance, "
    "Reformation, and Scientific Revolution."
)

EN_CAND_TEXT = (
    "Around 1440, Gutenberg invented the printing press. "
    "Before this, manuscripts were hand-copied and very scarce. "
    "He used movable metal type to reproduce texts quickly. "
    "In decades, presses proliferated across Europe. "
    "This democratization catalyzed the Renaissance, Reformation, "
    "and Scientific Revolution."
)

EN_CANON_DATA = {
    "passage_id": "sys_gut",
    "source_text": EN_SOURCE,
    "instructional_objective": "Explain how the printing press changed history.",
    "meaning_units": [
        {"id": "MU1", "text": "Gutenberg invented the printing press 1440",
         "required": True, "anchors": ["Gutenberg", "printing", "press", "1440", "invented"]},
        {"id": "MU2", "text": "manuscripts hand-copied scarce before press",
         "required": True, "anchors": ["manuscripts", "hand", "scarce"]},
        {"id": "MU3", "text": "movable metal type reproduction fast",
         "required": True, "anchors": ["movable", "metal", "type", "reproduction"]},
        {"id": "MU4", "text": "presses spread proliferated Europe decades",
         "required": True, "anchors": ["presses", "Europe", "decades", "proliferated"]},
        {"id": "MU5", "text": "democratization Renaissance Reformation Revolution",
         "required": True, "anchors": ["democratization", "Renaissance", "Reformation"]},
    ],
    "sequence_constraints": [
        {"before": "MU1", "after": "MU2"}, {"before": "MU2", "after": "MU3"},
        {"before": "MU3", "after": "MU4"}, {"before": "MU4", "after": "MU5"},
    ],
    "must_preserve_vocabulary": [
        {"term": "printing press",       "required": True,  "gloss_allowed": False},
        {"term": "manuscripts",          "required": True,  "gloss_allowed": True},
        {"term": "movable metal type",   "required": True,  "gloss_allowed": True},
        {"term": "Renaissance",          "required": True,  "gloss_allowed": True},
        {"term": "Reformation",          "required": True,  "gloss_allowed": True},
    ],
}

EN_CAND_DATA = {
    "candidates": [{
        "candidate_id": "A", "passage_id": "sys_gut", "relative_band": 0,
        "text": EN_CAND_TEXT,
        "scaffold": {
            "vocabulary_support": "medium", "syntax_support": "low",
            "cohesion_support": "low", "chunking_support": "low",
            "inference_support": "low",
        },
        "llm_self_audit": {
            "meaning_preserved": True, "sequence_preserved": True,
            "objective_preserved": True, "same_passage_identity": True,
            "notes": "All MUs covered.",
            "meaning_unit_coverage": {
                "MU1": True, "MU2": True, "MU3": True, "MU4": True, "MU5": True,
            },
        },
    }],
}

EN_FIT_DATA = {
    "fit_estimates": [{
        "candidate_id": "A", "access": "high", "growth": "medium",
        "support_burden": "low", "reason": "Well matched.",
    }],
}

EN_ASSESS_DATA = {
    "assessment_blueprint": {"passage_id": "sys_gut"},
    "items": [
        {"id": "Q1", "type": "literal_mcq", "target": "MU1", "question": "When did Gutenberg invent the press?",
         "choices": [{"id": "A", "text": "1440"}, {"id": "B", "text": "1520"},
                     {"id": "C", "text": "1380"}, {"id": "D", "text": "1600"}],
         "correct_answer": "A"},
        {"id": "Q2", "type": "sequence_mcq",
         "target": {"meaning_unit_ids": ["MU1", "MU2"], "relation": "before"},
         "question": "What came before manuscripts becoming scarce?",
         "choices": [{"id": "A", "text": "Gutenberg invented the press"},
                     {"id": "B", "text": "The Renaissance"}, {"id": "C", "text": "Movable type"},
                     {"id": "D", "text": "Presses spread"}],
         "correct_answer": "A"},
        {"id": "Q3", "type": "inference_mcq", "target": "MU5",
         "question": "What did the printing press enable?",
         "choices": [{"id": "A", "text": "Spread of ideas"}, {"id": "B", "text": "Slower writing"},
                     {"id": "C", "text": "Fewer books"}, {"id": "D", "text": "Isolation"}],
         "correct_answer": "A"},
        {"id": "Q4", "type": "vocabulary_mcq", "target": "movable metal type",
         "question": "What does 'movable metal type' mean?",
         "choices": [{"id": "A", "text": "Reusable metal letters for printing"},
                     {"id": "B", "text": "A type of metal alloy"},
                     {"id": "C", "text": "A portable press"}, {"id": "D", "text": "Bronze coins"}],
         "correct_answer": "A"},
        {"id": "Q5", "type": "retell_short_response", "target": None,
         "prompt": "Retell the passage in your own words.",
         "rubric": {
             "max_score": 4,
             "criteria": [
                 {"points": 1, "meaning_unit_ids": ["MU1"], "description": "Gutenberg printing press 1440"},
                 {"points": 1, "meaning_unit_ids": ["MU2"], "description": "manuscripts hand scarce"},
                 {"points": 1, "meaning_unit_ids": ["MU3"], "description": "movable metal type reproduction"},
                 {"points": 1, "meaning_unit_ids": ["MU4"], "description": "presses spread Europe"},
             ],
         }},
        {"id": "Q6", "type": "self_rating", "target": None,
         "prompt": "How difficult was the passage?", "scale": "1-5"},
    ],
    "scoring_blueprint": {
        "literal_item_ids": ["Q1"], "sequence_item_ids": ["Q2"],
        "inference_item_ids": ["Q3"], "vocabulary_item_ids": ["Q4"],
    },
    "signal_mapping": {
        "comprehension_score": {"formula": "wa", "weights": {"Q1": 0.25, "Q2": 0.25, "Q3": 0.25, "Q4": 0.25}},
        "inference_score":     {"formula": "wa", "weights": {"Q3": 0.6, "Q5": 0.4}},
        "vocabulary_signal":   {"formula": "Q4", "weights": {"Q4": 1.0}},
        "retell_quality":      {"formula": "Q5"},
        "difficulty_signal":   {"formula": "Q6"},
    },
}

EN_RETELL_DATA = {
    "raw_score": 4, "max_score": 4,
    "matched_meaning_units": ["MU1", "MU2", "MU3", "MU4"],
    "matched_relationships": [],
    "concise_reason": "All key ideas covered.",
}

# EngineConfig with wide tolerances: ensures VALIDATED for all system-path tests
LOOSE_CFG = EngineConfig(
    fk_tolerance=10.0,
    overall_meaning_threshold=0.40,
    vocabulary_threshold=0.40,
    length_deviation_threshold=2.0,
    length_ceiling=2.0,
)

# Config with very tight FK: forces DEGRADED FK-only
TIGHT_FK_CFG = EngineConfig(
    fk_tolerance=0.1,
    overall_meaning_threshold=0.40,
    vocabulary_threshold=0.40,
    length_deviation_threshold=2.0,
    length_ceiling=2.0,
)

def make_en_mock(extra=None, error_on=None):
    """Build a TaskRoutingMockLLM with standard EN responses, with optional overrides."""
    base = {
        "canonicalize_passage": EN_CANON_DATA,
        "generate_candidates":  EN_CAND_DATA,
        "estimate_fit":         EN_FIT_DATA,
        "generate_assessment":  EN_ASSESS_DATA,
        "score_retell":         EN_RETELL_DATA,
        "diagnose_outcome":     {"diagnosis": "well_calibrated", "reason": "Good."},
    }
    if extra:
        base.update(extra)
    return TaskRoutingMockLLM(responses=base, error_on_tasks=error_on or set())

EN_LEARNER = LearnerState("sys_learner", 5.0, vocabulary_need=Level.MEDIUM)

EN_GOOD_ANSWERS = {"Q1": "A", "Q2": "A", "Q3": "A", "Q4": "A", "Q5": "Gutenberg invented the printing press.", "Q6": 2}
EN_GOOD_TELEMETRY = ReadingTelemetry(fluency_score=0.90, hint_use_rate=0.05, reread_count=1, completion=True)
EN_HIGH_TELEMETRY = ReadingTelemetry(fluency_score=0.95, hint_use_rate=0.05, reread_count=0, completion=True)

# ── Spanish ───────────────────────────────────────────────────────────────────

ES_SOURCE = (
    "La Revolución Francesa comenzó con la toma de la Bastilla en 1789 y transformó Europa. "
    "El Antiguo Régimen colapsó ante la crisis económica y las ideas ilustradas. "
    "La Declaración de los Derechos del Hombre proclamó la igualdad y la soberanía popular. "
    "Pero el Terror segó miles de vidas bajo la guillotina. "
    "Napoleón llevó los ideales de libertad, igualdad y fraternidad por todo el continente."
)

ES_CAND_TEXT = (
    "En 1789, el pueblo tomó la Bastilla y comenzó la Revolución Francesa. "
    "El Antiguo Régimen cayó por la crisis económica y las ideas ilustradas. "
    "Los revolucionarios proclamaron la igualdad y la soberanía en la Declaración de los Derechos del Hombre. "
    "Sin embargo, el Terror causó miles de muertes bajo la guillotina. "
    "Más tarde, Napoleón extendió los ideales de libertad, igualdad y fraternidad por Europa."
)

ES_CANON_DATA = {
    "passage_id": "sys_rev",
    "source_text": ES_SOURCE,
    "instructional_objective": "Comprender las causas y consecuencias de la Revolución Francesa.",
    "meaning_units": [
        {"id": "UM1", "text": "Revolución Francesa Bastilla 1789 transformó",
         "required": True, "anchors": ["Revolución", "Bastilla", "1789", "transformó"]},
        {"id": "UM2", "text": "Antiguo Régimen colapsó crisis ilustradas",
         "required": True, "anchors": ["Antiguo", "Régimen", "colapsó", "crisis"]},
        {"id": "UM3", "text": "Declaración derechos igualdad soberanía",
         "required": True, "anchors": ["Declaración", "igualdad", "soberanía"]},
        {"id": "UM4", "text": "Terror guillotina miles vidas",
         "required": True, "anchors": ["Terror", "guillotina", "miles"]},
        {"id": "UM5", "text": "Napoleón libertad igualdad fraternidad Europa",
         "required": True, "anchors": ["Napoleón", "libertad", "fraternidad"]},
    ],
    "sequence_constraints": [
        {"before": "UM1", "after": "UM2"}, {"before": "UM2", "after": "UM3"},
        {"before": "UM3", "after": "UM4"}, {"before": "UM4", "after": "UM5"},
    ],
    "must_preserve_vocabulary": [
        {"term": "Revolución Francesa",  "required": True, "gloss_allowed": False},
        {"term": "Antiguo Régimen",       "required": True, "gloss_allowed": True},
        {"term": "Declaración",           "required": True, "gloss_allowed": True},
        {"term": "guillotina",            "required": True, "gloss_allowed": True},
        {"term": "Napoleón",              "required": True, "gloss_allowed": True},
    ],
}

ES_CAND_DATA = {
    "candidates": [{
        "candidate_id": "A", "passage_id": "sys_rev", "relative_band": 0,
        "text": ES_CAND_TEXT,
        "scaffold": {
            "vocabulary_support": "medium", "syntax_support": "low",
            "cohesion_support": "medium", "chunking_support": "low",
            "inference_support": "low",
        },
        "llm_self_audit": {
            "meaning_preserved": True, "sequence_preserved": True,
            "objective_preserved": True, "same_passage_identity": True,
            "notes": "Todas las UMs presentes.",
            "meaning_unit_coverage": {
                "UM1": True, "UM2": True, "UM3": True, "UM4": True, "UM5": True,
            },
        },
    }],
}

ES_FIT_DATA = {
    "fit_estimates": [{
        "candidate_id": "A", "access": "high", "growth": "medium",
        "support_burden": "low", "reason": "Buen ajuste para el alumno.",
    }],
}

ES_ASSESS_DATA = {
    "assessment_blueprint": {"passage_id": "sys_rev"},
    "items": [
        {"id": "P1", "type": "literal_mcq", "target": "UM1",
         "question": "¿En qué año comenzó la Revolución Francesa?",
         "choices": [{"id": "A", "text": "1789"}, {"id": "B", "text": "1815"},
                     {"id": "C", "text": "1750"}, {"id": "D", "text": "1804"}],
         "correct_answer": "A"},
        {"id": "P2", "type": "sequence_mcq",
         "target": {"meaning_unit_ids": ["UM1", "UM2"], "relation": "before"},
         "question": "¿Qué ocurrió antes del colapso del Antiguo Régimen?",
         "choices": [{"id": "A", "text": "La toma de la Bastilla"},
                     {"id": "B", "text": "El Terror"}, {"id": "C", "text": "Napoleón"},
                     {"id": "D", "text": "La guillotina"}],
         "correct_answer": "A"},
        {"id": "P3", "type": "inference_mcq", "target": "UM5",
         "question": "¿Qué consecuencia tuvo Napoleón para Europa?",
         "choices": [{"id": "A", "text": "Extendió los ideales revolucionarios"},
                     {"id": "B", "text": "Restauró la monarquía"},
                     {"id": "C", "text": "Aisló a Francia"}, {"id": "D", "text": "Abolió la guillotina"}],
         "correct_answer": "A"},
        {"id": "P4", "type": "vocabulary_mcq", "target": "guillotina",
         "question": "¿Qué era la guillotina?",
         "choices": [{"id": "A", "text": "Instrumento de ejecución"},
                     {"id": "B", "text": "Un tipo de cañón"}, {"id": "C", "text": "Una cárcel"},
                     {"id": "D", "text": "Un documento legal"}],
         "correct_answer": "A"},
        {"id": "P5", "type": "retell_short_response", "target": None,
         "prompt": "Resume el texto con tus propias palabras.",
         "rubric": {
             "max_score": 4,
             "criteria": [
                 {"points": 1, "meaning_unit_ids": ["UM1"], "description": "Revolución Francesa Bastilla 1789"},
                 {"points": 1, "meaning_unit_ids": ["UM2"], "description": "Antiguo Régimen crisis ilustradas"},
                 {"points": 1, "meaning_unit_ids": ["UM3"], "description": "Declaración igualdad soberanía"},
                 {"points": 1, "meaning_unit_ids": ["UM4"], "description": "Terror guillotina muertes"},
             ],
         }},
        {"id": "P6", "type": "self_rating", "target": None,
         "prompt": "¿Qué tan difícil fue el texto?", "scale": "1-5"},
    ],
    "scoring_blueprint": {
        "literal_item_ids": ["P1"], "sequence_item_ids": ["P2"],
        "inference_item_ids": ["P3"], "vocabulary_item_ids": ["P4"],
    },
    "signal_mapping": {
        "comprehension_score": {"formula": "wa", "weights": {"P1": 0.25, "P2": 0.25, "P3": 0.25, "P4": 0.25}},
        "inference_score":     {"formula": "wa", "weights": {"P3": 0.6, "P5": 0.4}},
        "vocabulary_signal":   {"formula": "P4", "weights": {"P4": 1.0}},
        "retell_quality":      {"formula": "P5"},
        "difficulty_signal":   {"formula": "P6"},
    },
}

ES_RETELL_DATA = {
    "raw_score": 3, "max_score": 4,
    "matched_meaning_units": ["UM1", "UM2", "UM3"],
    "matched_relationships": [],
    "concise_reason": "Tres ideas principales cubiertas.",
}

ES_LOOSE_CFG = EngineConfig(
    fk_tolerance=10.0,
    overall_meaning_threshold=0.40,
    vocabulary_threshold=0.40,
    length_deviation_threshold=2.0,
    length_ceiling=2.0,
)

def make_es_mock(extra=None, error_on=None):
    base = {
        "canonicalize_passage": ES_CANON_DATA,
        "generate_candidates":  ES_CAND_DATA,
        "estimate_fit":         ES_FIT_DATA,
        "generate_assessment":  ES_ASSESS_DATA,
        "score_retell":         ES_RETELL_DATA,
        "diagnose_outcome":     {"diagnosis": "well_calibrated", "reason": "Bien calibrado."},
    }
    if extra:
        base.update(extra)
    return ESMockLLM(responses=base, error_on_tasks=error_on or set())

ES_LEARNER = ESLearnerState("sys_alumno", 5.0, vocabulary_need=ESLevel.MEDIUM)
ES_GOOD_ANSWERS = {"P1": "A", "P2": "A", "P3": "A", "P4": "A", "P5": "La Revolución Francesa.", "P6": 2}
ES_GOOD_TELEMETRY = ESReadingTelemetry(fluency_score=0.80, hint_use_rate=0.10, reread_count=1, completion=True)


# ══════════════════════════════════════════════════════════════════════════════
# S01 — EN prepare_cycle: CyclePreparation structure
# ══════════════════════════════════════════════════════════════════════════════

group("S01 — EN prepare_cycle: CyclePreparation structure")

en_sys_s01 = AdaptiveReadingSystem(llm=make_en_mock(), engine=DeterministicEngine(LOOSE_CFG))
prep_s01 = en_sys_s01.prepare_cycle(EN_SOURCE, "sys_gut", "Explain how the printing press changed history.", EN_LEARNER)

ck("prepare_cycle returns CyclePreparation",           isinstance(prep_s01, CyclePreparation))
ck("canonical is CanonicalPassage",                    isinstance(prep_s01.canonical, CanonicalPassage))
ck("canonical.passage_id correct",                     prep_s01.canonical.passage_id == "sys_gut")
ck("canonical has 5 meaning units",                    len(prep_s01.canonical.meaning_units) == 5)
ck("canonical has 4 sequence constraints",             len(prep_s01.canonical.sequence_constraints) == 4)
ck("canonical has 5 vocabulary terms",                 len(prep_s01.canonical.must_preserve_vocabulary) == 5)
ck("selected_candidate is CandidatePassage",           isinstance(prep_s01.selected_candidate, CandidatePassage))
ck("selected_candidate.passage_id matches canonical",  prep_s01.selected_candidate.passage_id == "sys_gut")
ck("selected_scores is DeterministicScores",           isinstance(prep_s01.selected_scores, DeterministicScores))
ck("selected_scores.passed_constraints (VALIDATED)",   prep_s01.selected_scores.passed_constraints)
ck("selection_mode is VALIDATED",                      prep_s01.selection_mode == SelectionMode.VALIDATED)
ck("assessment has 6 items",                           len(prep_s01.assessment.items) == 6)
ck("assessment item types in correct order",
   [i.type for i in prep_s01.assessment.items] == [
       "literal_mcq", "sequence_mcq", "inference_mcq",
       "vocabulary_mcq", "retell_short_response", "self_rating"])
ck("fit_estimates dict has entry for candidate A",     "A" in prep_s01.fit_estimates)
ck("all_scores dict has entry for candidate A",        "A" in prep_s01.all_scores)
ck("cycle_id is non-empty string",                     isinstance(prep_s01.cycle_id, str) and len(prep_s01.cycle_id) > 0)
ck("prepared_learner is LearnerState",                 isinstance(prep_s01.prepared_learner, LearnerState))
ck("prepared_learner.target_band set to source_fk",
   prep_s01.prepared_learner.target_band == prep_s01.canonical.source_fk)
ck("prepared_learner.entry_band set to learner band",  prep_s01.prepared_learner.entry_band == EN_LEARNER.current_band)


# ══════════════════════════════════════════════════════════════════════════════
# S02 — EN complete_cycle: full UNDERCHALLENGED path
# ══════════════════════════════════════════════════════════════════════════════

group("S02 — EN complete_cycle: full UNDERCHALLENGED path")

en_sys_s02 = AdaptiveReadingSystem(
    llm=make_en_mock(extra={"diagnose_outcome": {"diagnosis": "underchallenged", "reason": "Very easy."}}),
    engine=DeterministicEngine(LOOSE_CFG),
)
prep_s02 = en_sys_s02.prepare_cycle(EN_SOURCE, "sys_gut", "Explain how the printing press changed history.", EN_LEARNER)
outcome_s02 = en_sys_s02.complete_cycle(
    learner=EN_LEARNER,
    prep=prep_s02,
    learner_answers=EN_GOOD_ANSWERS,
    telemetry=EN_HIGH_TELEMETRY,
)

ck("complete_cycle returns CycleOutcome",              isinstance(outcome_s02, CycleOutcome))
ck("diagnosis is UNDERCHALLENGED",                     outcome_s02.diagnosis == DiagnosisLabel.UNDERCHALLENGED)
ck("updated_learner is LearnerState",                  isinstance(outcome_s02.updated_learner, LearnerState))
ck("updated_learner differs from input",               outcome_s02.updated_learner != EN_LEARNER)
ck("assessment_result is AssessmentResult",            isinstance(outcome_s02.assessment_result, AssessmentResult))
ck("reading_signals is ReadingSignals",                isinstance(outcome_s02.reading_signals, ReadingSignals))
ck("cycle_id preserved from prep",                     outcome_s02.cycle_id == prep_s02.cycle_id)
ck("comprehension_score is 1.0 (all correct)",         outcome_s02.assessment_result.comprehension_score == 1.0)
ck("retell_quality = 4/4 = 1.0",                       outcome_s02.assessment_result.retell_quality == 1.0)
ck("difficulty_rating = 2",                            outcome_s02.assessment_result.difficulty_rating == 2)
ck("completion = True in reading_signals",             outcome_s02.reading_signals.completion is True)
ck("fluency_score in reading_signals",                 outcome_s02.reading_signals.fluency_score == 0.95)
ck("UNDERCHALLENGED: band increased",
   outcome_s02.updated_learner.current_band > EN_LEARNER.current_band)
ck("UNDERCHALLENGED: readiness_to_increase = HIGH",
   outcome_s02.updated_learner.readiness_to_increase == Level.HIGH)
ck("UNDERCHALLENGED in recent_outcomes",
   DiagnosisLabel.UNDERCHALLENGED in outcome_s02.updated_learner.recent_outcomes)
ck("cycles_on_passage incremented",
   outcome_s02.updated_learner.cycles_on_passage > 0)


# ══════════════════════════════════════════════════════════════════════════════
# S03 — EN complete_cycle: all 7 diagnosis labels
# ══════════════════════════════════════════════════════════════════════════════

group("S03 — EN complete_cycle: all 7 diagnosis labels")

_all_labels = [
    ("underchallenged",                  DiagnosisLabel.UNDERCHALLENGED),
    ("well_calibrated",                  DiagnosisLabel.WELL_CALIBRATED),
    ("successful_but_support_dependent", DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT),
    ("vocabulary_barrier",               DiagnosisLabel.VOCABULARY_BARRIER),
    ("syntax_barrier",                   DiagnosisLabel.SYNTAX_BARRIER),
    ("cohesion_inference_barrier",       DiagnosisLabel.COHESION_INFERENCE_BARRIER),
    ("overloaded",                       DiagnosisLabel.OVERLOADED),
]

for label_str, label_enum in _all_labels:
    _sys = AdaptiveReadingSystem(
        llm=make_en_mock(extra={"diagnose_outcome": {"diagnosis": label_str, "reason": "test"}}),
        engine=DeterministicEngine(LOOSE_CFG),
    )
    _prep = _sys.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER)
    _out = _sys.complete_cycle(EN_LEARNER, _prep, EN_GOOD_ANSWERS, EN_GOOD_TELEMETRY)
    ck(f"diagnosis {label_str!r} fires correctly",
       _out.diagnosis == label_enum,
       f"got {_out.diagnosis}")
    ck(f"{label_str!r} appears in updated_learner.recent_outcomes",
       label_enum in _out.updated_learner.recent_outcomes)


# ══════════════════════════════════════════════════════════════════════════════
# S04 — EN update_learner_state: all 7 label state mutations
# ══════════════════════════════════════════════════════════════════════════════

group("S04 — EN update_learner_state: all 7 label state mutations")

eng = DeterministicEngine(LOOSE_CFG)
_base = LearnerState("upd", 5.0,
    vocabulary_need=Level.MEDIUM, syntax_need=Level.MEDIUM,
    cohesion_need=Level.MEDIUM, support_dependence=Level.MEDIUM,
    readiness_to_increase=Level.LOW,
)
_sigs_good = ReadingSignals(0.90, 0.80, 0.90, 0.05, 0, 2, 0.80, True)
_sigs_ok   = ReadingSignals(0.75, 0.50, 0.70, 0.05, 1, 3, 0.60, True)
_sigs_hint = ReadingSignals(0.75, 0.40, 0.70, 0.35, 2, 3, 0.45, True)
_sigs_vcb  = ReadingSignals(0.60, 0.40, 0.60, 0.15, 1, 3, 0.40, True)
_sigs_fail = ReadingSignals(0.40, 0.30, 0.40, 0.40, 3, 5, 0.20, True)

# UNDERCHALLENGED
_u = eng.update_learner_state(_base, DiagnosisLabel.UNDERCHALLENGED, _sigs_good)
ck("UNDERCHALLENGED: band increases",             _u.current_band > _base.current_band)
ck("UNDERCHALLENGED: readiness = HIGH",           _u.readiness_to_increase == Level.HIGH)
ck("UNDERCHALLENGED: support_dependence down",    _u.support_dependence.score <= _base.support_dependence.score)

# WELL_CALIBRATED
_w = eng.update_learner_state(_base, DiagnosisLabel.WELL_CALIBRATED, _sigs_ok)
ck("WELL_CALIBRATED: readiness increases",        _w.readiness_to_increase.score >= _base.readiness_to_increase.score)
ck("WELL_CALIBRATED: band unchanged",             _w.current_band == _base.current_band)

# SUCCESSFUL_BUT_SUPPORT_DEPENDENT
_s = eng.update_learner_state(_base, DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT, _sigs_hint)
ck("SBSD: readiness = LOW",                       _s.readiness_to_increase == Level.LOW)
ck("SBSD: support_dependence increases",          _s.support_dependence.score > _base.support_dependence.score)

# VOCABULARY_BARRIER
_v = eng.update_learner_state(_base, DiagnosisLabel.VOCABULARY_BARRIER, _sigs_vcb)
ck("VOCABULARY_BARRIER: vocab_need increases",    _v.vocabulary_need.score > _base.vocabulary_need.score)
ck("VOCABULARY_BARRIER: readiness = LOW",         _v.readiness_to_increase == Level.LOW)

# SYNTAX_BARRIER
_base_syn = replace(_base, syntax_need=Level.LOW, vocabulary_need=Level.LOW)
_sy = eng.update_learner_state(_base_syn, DiagnosisLabel.SYNTAX_BARRIER, _sigs_vcb)
ck("SYNTAX_BARRIER: syntax_need increases",       _sy.syntax_need.score > _base_syn.syntax_need.score)
ck("SYNTAX_BARRIER: readiness = LOW",             _sy.readiness_to_increase == Level.LOW)

# COHESION_INFERENCE_BARRIER
_c = eng.update_learner_state(_base, DiagnosisLabel.COHESION_INFERENCE_BARRIER, _sigs_ok)
ck("CIB: cohesion_need increases",                _c.cohesion_need.score > _base.cohesion_need.score)
ck("CIB: readiness = LOW",                        _c.readiness_to_increase == Level.LOW)

# OVERLOADED
_o = eng.update_learner_state(_base, DiagnosisLabel.OVERLOADED, _sigs_fail)
ck("OVERLOADED: band decreases",                  _o.current_band < _base.current_band)
ck("OVERLOADED: vocab_need increases",            _o.vocabulary_need.score > _base.vocabulary_need.score)
ck("OVERLOADED: syntax_need increases",           _o.syntax_need.score > _base.syntax_need.score)
ck("OVERLOADED: cohesion_need increases",         _o.cohesion_need.score > _base.cohesion_need.score)
ck("OVERLOADED: support_dependence increases",    _o.support_dependence.score > _base.support_dependence.score)
ck("OVERLOADED: readiness = LOW",                 _o.readiness_to_increase == Level.LOW)

# History capped at history_limit
_h = _base
for lbl in [DiagnosisLabel.WELL_CALIBRATED, DiagnosisLabel.VOCABULARY_BARRIER,
            DiagnosisLabel.OVERLOADED, DiagnosisLabel.UNDERCHALLENGED]:
    _h = eng.update_learner_state(_h, lbl, _sigs_ok)
ck("recent_outcomes capped at history_limit (3)",  len(_h.recent_outcomes) <= eng.config.history_limit)


# ══════════════════════════════════════════════════════════════════════════════
# S05 — EN journey tracking
# ══════════════════════════════════════════════════════════════════════════════

group("S05 — EN journey tracking")

en_sys_s05 = AdaptiveReadingSystem(llm=make_en_mock(), engine=DeterministicEngine(LOOSE_CFG))

# New passage → target_band and entry_band set, cycles reset
learner_fresh = LearnerState("journey", 5.0)
prep_j1 = en_sys_s05.prepare_cycle(EN_SOURCE, "sys_gut", "test", learner_fresh)
ck("target_band set to source_fk",
   prep_j1.prepared_learner.target_band == prep_j1.canonical.source_fk)
ck("entry_band set to learner current_band",
   prep_j1.prepared_learner.entry_band == learner_fresh.current_band)
ck("cycles_on_passage = 0 on first assignment",
   prep_j1.prepared_learner.cycles_on_passage == 0)

# Complete one cycle — cycles_on_passage increments
out_j1 = en_sys_s05.complete_cycle(learner_fresh, prep_j1, EN_GOOD_ANSWERS, EN_GOOD_TELEMETRY)
ck("cycles_on_passage = 1 after first cycle",
   out_j1.updated_learner.cycles_on_passage == 1)

# Same passage again → cycles_on_passage continues, entry_band preserved
learner_after1 = out_j1.updated_learner
prep_j2 = en_sys_s05.prepare_cycle(EN_SOURCE, "sys_gut", "test", learner_after1)
ck("cycles_on_passage preserved after same-passage re-prepare",
   prep_j2.prepared_learner.cycles_on_passage == 1)
ck("entry_band preserved for same passage",
   prep_j2.prepared_learner.entry_band == learner_fresh.current_band)

# begin_passage_journey directly
_j = LearnerState("jtest", 4.0, target_band=None)
_canon_tmp = CanonicalPassage("p1", EN_SOURCE, "obj", (MeaningUnit("M1","test",True,()),), source_fk=8.0)
_j2 = en_sys_s05.begin_passage_journey(_j, _canon_tmp)
ck("begin_passage_journey sets target_band",    _j2.target_band == 8.0)
ck("begin_passage_journey sets entry_band",     _j2.entry_band == 4.0)
ck("begin_passage_journey resets cycles to 0",  _j2.cycles_on_passage == 0)

# Same passage: entry_band not overwritten
_j3_learner = replace(_j2, target_band=8.0, entry_band=4.0, cycles_on_passage=3)
_j4 = en_sys_s05.begin_passage_journey(_j3_learner, _canon_tmp)
ck("entry_band not overwritten on same passage", _j4.entry_band == 4.0)
ck("cycles_on_passage preserved on same passage", _j4.cycles_on_passage == 3)


# ══════════════════════════════════════════════════════════════════════════════
# S06 — EN DEGRADED: FK-only path
# ══════════════════════════════════════════════════════════════════════════════

group("S06 — EN DEGRADED: FK-only (surface warnings, no blocking failures)")

cap_s06 = type("CapH", (logging.Handler,), {"records": [], "emit": lambda self, r: self.records.append(r)})()
logger_s06 = logging.getLogger("s06")
logger_s06.handlers.clear()
logger_s06.addHandler(cap_s06)
logger_s06.setLevel(logging.DEBUG)

en_sys_s06 = AdaptiveReadingSystem(
    llm=make_en_mock(),
    engine=DeterministicEngine(TIGHT_FK_CFG),
    logger=logger_s06,
)
prep_s06 = en_sys_s06.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER)

ck("DEGRADED FK-only: selection_mode is DEGRADED",
   prep_s06.selection_mode == SelectionMode.DEGRADED)
ck("DEGRADED FK-only: no blocking_reasons",
   len(prep_s06.selected_scores.blocking_reasons) == 0)
ck("DEGRADED FK-only: warning_flags present",
   len(prep_s06.selected_scores.warning_flags) > 0)
ck("DEGRADED FK-only: fk_out_of_tolerance in warning_flags",
   any("fk_out_of_tolerance" in w for w in prep_s06.selected_scores.warning_flags))
warn_msgs = [r.getMessage() for r in cap_s06.records if r.levelno == logging.WARNING]
degraded_msgs = [m for m in warn_msgs if "DEGRADED" in m]
ck("DEGRADED FK-only: warning logged",
   len(degraded_msgs) > 0, f"warn_msgs={warn_msgs}")
ck("DEGRADED FK-only: log distinguishes FK-only from structural",
   any("surface" in m or "FK" in m or "fk" in m.lower() for m in degraded_msgs),
   f"msgs={degraded_msgs}")

# complete_cycle still works on a DEGRADED prep
out_s06 = en_sys_s06.complete_cycle(EN_LEARNER, prep_s06, EN_GOOD_ANSWERS, EN_GOOD_TELEMETRY)
ck("DEGRADED FK-only: complete_cycle still produces CycleOutcome",
   isinstance(out_s06, CycleOutcome))


# ══════════════════════════════════════════════════════════════════════════════
# S07 — EN DEGRADED: structural path
# ══════════════════════════════════════════════════════════════════════════════

group("S07 — EN blocking failures and ALIENError escalation")

# ── Scenario A: single candidate, self-audit meaning_preserved=False → ALIENError ──
# Blocking failures are never recoverable; when all candidates are blocked the
# engine raises ALIENError rather than returning DEGRADED.
_cand_one_blocked = {
    "candidates": [{
        "candidate_id": "B", "passage_id": "sys_gut", "relative_band": 0,
        "text": EN_CAND_TEXT,
        "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                     "cohesion_support": "low", "chunking_support": "low",
                     "inference_support": "low"},
        "llm_self_audit": {
            "meaning_preserved": False,   # ← blocking
            "sequence_preserved": True, "objective_preserved": True,
            "same_passage_identity": True, "notes": "MU missing.",
        },
    }],
}
_fit_b = {"fit_estimates": [{"candidate_id": "B", "access": "medium",
                              "growth": "low", "support_burden": "high", "reason": "Poor."}]}
en_sys_s07a = AdaptiveReadingSystem(
    llm=make_en_mock(extra={"generate_candidates": _cand_one_blocked, "estimate_fit": _fit_b}),
    engine=DeterministicEngine(LOOSE_CFG),
)
ck_raises(
    "single candidate: meaning_preserved=False → ALIENError",
    lambda: en_sys_s07a.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER),
    ALIENError,
)

# ── Scenario B: all candidates fully blocked → ALIENError ──
_all_blocked = {
    "candidates": [{
        "candidate_id": "C", "passage_id": "sys_gut", "relative_band": 0,
        "text": EN_CAND_TEXT,
        "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                     "cohesion_support": "low", "chunking_support": "low",
                     "inference_support": "low"},
        "llm_self_audit": {
            "meaning_preserved": False, "sequence_preserved": False,
            "objective_preserved": False, "same_passage_identity": False,
            "notes": "All failed.",
        },
    }],
}
_fit_c = {"fit_estimates": [{"candidate_id": "C", "access": "low",
                              "growth": "low", "support_burden": "high", "reason": "All bad."}]}
en_sys_s07b = AdaptiveReadingSystem(
    llm=make_en_mock(extra={"generate_candidates": _all_blocked, "estimate_fit": _fit_c}),
    engine=DeterministicEngine(LOOSE_CFG),
)
ck_raises(
    "all candidates blocked → ALIENError raised",
    lambda: en_sys_s07b.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER),
    ALIENError,
)

# ── Scenario C: mixed pool — one blocked, one surface-only → DEGRADED (surface wins) ──
# With TIGHT_FK_CFG both candidates fail the FK check (surface warning only).
# Candidate B additionally has a blocking self-audit failure.
# Candidate D has only the FK surface warning → goes into the degraded pool and wins.
_cand_mixed = {
    "candidates": [
        {
            "candidate_id": "B", "passage_id": "sys_gut", "relative_band": 0,
            "text": EN_CAND_TEXT,
            "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                         "cohesion_support": "low", "chunking_support": "low",
                         "inference_support": "low"},
            "llm_self_audit": {
                "meaning_preserved": False,  # ← blocking
                "sequence_preserved": True, "objective_preserved": True,
                "same_passage_identity": True, "notes": "MU missing.",
            },
        },
        {
            "candidate_id": "D", "passage_id": "sys_gut", "relative_band": 0,
            "text": EN_CAND_TEXT,
            "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                         "cohesion_support": "low", "chunking_support": "low",
                         "inference_support": "low"},
            "llm_self_audit": {
                "meaning_preserved": True,  # ← passes
                "sequence_preserved": True, "objective_preserved": True,
                "same_passage_identity": True, "notes": "OK",
            },
        },
    ]
}
_fit_mixed = {"fit_estimates": [
    {"candidate_id": "B", "access": "low",    "growth": "low", "support_burden": "high", "reason": "Bad."},
    {"candidate_id": "D", "access": "medium", "growth": "low", "support_burden": "low",  "reason": "OK."},
]}
en_sys_s07c = AdaptiveReadingSystem(
    llm=make_en_mock(extra={"generate_candidates": _cand_mixed, "estimate_fit": _fit_mixed}),
    engine=DeterministicEngine(TIGHT_FK_CFG),
)
prep_s07c = en_sys_s07c.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER)
ck("mixed pool: selection_mode is DEGRADED",
   prep_s07c.selection_mode == SelectionMode.DEGRADED)
ck("mixed pool: blocked candidate B was NOT selected",
   prep_s07c.selected_candidate.candidate_id != "B")
ck("mixed pool: surface-only candidate D was selected",
   prep_s07c.selected_candidate.candidate_id == "D")
ck("mixed pool: selected_scores has no blocking_reasons",
   len(prep_s07c.selected_scores.blocking_reasons) == 0)
ck("mixed pool: complete_cycle still succeeds on DEGRADED prep",
   isinstance(
       en_sys_s07c.complete_cycle(EN_LEARNER, prep_s07c, EN_GOOD_ANSWERS, EN_GOOD_TELEMETRY),
       CycleOutcome,
   ))


# ══════════════════════════════════════════════════════════════════════════════
# S08 — EN error handling: ValidationError, ALIENError, LLM fallbacks
# ══════════════════════════════════════════════════════════════════════════════

group("S08 — EN error handling: ValidationError, ALIENError, LLM fallbacks")

# Bad canonical (missing instructional_objective)
_bad_canon = {"passage_id": "p", "source_text": EN_SOURCE,
              "meaning_units": [{"id": "M1", "text": "test", "required": True, "anchors": []}],
              "sequence_constraints": [], "must_preserve_vocabulary": []}
en_sys_bad_canon = AdaptiveReadingSystem(
    llm=TaskRoutingMockLLM(responses={"canonicalize_passage": _bad_canon}),
    engine=DeterministicEngine(LOOSE_CFG),
)
ck_raises("bad canonical → ALIENError",
          lambda: en_sys_bad_canon.prepare_cycle(EN_SOURCE, "p", "obj", EN_LEARNER),
          ALIENError)

# Bad candidates (duplicate candidate_ids)
_bad_cands = {"candidates": [
    {"candidate_id": "X", "passage_id": "sys_gut", "relative_band": 0, "text": "a",
     "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                  "cohesion_support": "low", "chunking_support": "low", "inference_support": "low"},
     "llm_self_audit": {"meaning_preserved": True, "sequence_preserved": True,
                        "objective_preserved": True, "same_passage_identity": True, "notes": ""}},
    {"candidate_id": "X", "passage_id": "sys_gut", "relative_band": 0, "text": "b",
     "scaffold": {"vocabulary_support": "low", "syntax_support": "low",
                  "cohesion_support": "low", "chunking_support": "low", "inference_support": "low"},
     "llm_self_audit": {"meaning_preserved": True, "sequence_preserved": True,
                        "objective_preserved": True, "same_passage_identity": True, "notes": ""}},
]}
en_sys_bad_cands = AdaptiveReadingSystem(
    llm=make_en_mock(extra={"generate_candidates": _bad_cands}),
    engine=DeterministicEngine(LOOSE_CFG),
)
ck_raises("duplicate candidate_ids → ALIENError",
          lambda: en_sys_bad_cands.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER),
          ALIENError)

# Empty source text
en_sys_empty = AdaptiveReadingSystem(llm=make_en_mock(), engine=DeterministicEngine(LOOSE_CFG))
ck_raises("empty source_text → ValueError",
          lambda: en_sys_empty.prepare_cycle("", "p", "obj", EN_LEARNER),
          ValueError)
ck_raises("whitespace-only source_text → ValueError",
          lambda: en_sys_empty.prepare_cycle("   ", "p", "obj", EN_LEARNER),
          ValueError)

# ALIENError carries stage attribute
try:
    raise ALIENError("my_stage", "test message")
except ALIENError as exc:
    ck("ALIENError has .stage attribute",  exc.stage == "my_stage")
    ck("ALIENError str includes stage",    "my_stage" in str(exc))

# ValidationError subclasses ALIENError
ck("ValidationError is subclass of ALIENError",  issubclass(ValidationError, ALIENError))

# Diagnosis LLM failure → deterministic fallback fires
cap_s08 = type("Cap", (logging.Handler,), {"records": [], "emit": lambda self, r: self.records.append(r)})()
logger_s08 = logging.getLogger("s08")
logger_s08.handlers.clear()
logger_s08.addHandler(cap_s08)
logger_s08.setLevel(logging.DEBUG)

en_sys_s08 = AdaptiveReadingSystem(
    llm=make_en_mock(error_on={"diagnose_outcome"}),
    engine=DeterministicEngine(LOOSE_CFG),
    logger=logger_s08,
)
prep_s08 = en_sys_s08.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER)
out_s08 = en_sys_s08.complete_cycle(EN_LEARNER, prep_s08, EN_GOOD_ANSWERS, EN_HIGH_TELEMETRY)
ck("diagnosis fallback: CycleOutcome still returned",  isinstance(out_s08, CycleOutcome))
ck("diagnosis fallback: diagnosis is a valid label",   isinstance(out_s08.diagnosis, DiagnosisLabel))
warn_s08 = [r.getMessage() for r in cap_s08.records if r.levelno == logging.WARNING]
ck("diagnosis fallback: warning logged",
   any("fallback" in m.lower() or "deterministic" in m.lower() for m in warn_s08),
   f"warnings={warn_s08}")

# Retell LLM failure → keyword fallback fires
en_sys_s08b = AdaptiveReadingSystem(
    llm=make_en_mock(error_on={"score_retell"}),
    engine=DeterministicEngine(LOOSE_CFG),
    logger=logger_s08,
)
prep_s08b = en_sys_s08b.prepare_cycle(EN_SOURCE, "sys_gut", "test", EN_LEARNER)
out_s08b = en_sys_s08b.complete_cycle(EN_LEARNER, prep_s08b, EN_GOOD_ANSWERS, EN_GOOD_TELEMETRY)
ck("retell fallback: CycleOutcome still returned",      isinstance(out_s08b, CycleOutcome))
ck("retell fallback: retell_quality is float in [0,1]",
   0.0 <= out_s08b.assessment_result.retell_quality <= 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# S09 — ES prepare_cycle: Spanish full cycle structure
# ══════════════════════════════════════════════════════════════════════════════

group("S09 — ES prepare_cycle: Spanish full cycle structure")

es_sys_s09 = ESSystem(llm=make_es_mock(), engine=ESDeterministicEngine(ES_LOOSE_CFG))
prep_es = es_sys_s09.prepare_cycle(ES_SOURCE, "sys_rev",
                                   "Comprender la Revolución Francesa.", ES_LEARNER)

ck("ES prepare_cycle returns CyclePreparation",            type(prep_es).__name__ == "CyclePreparation")
ck("ES canonical.passage_id correct",                      prep_es.canonical.passage_id == "sys_rev")
ck("ES canonical has 5 meaning units",                     len(prep_es.canonical.meaning_units) == 5)
ck("ES canonical has 4 sequence constraints",              len(prep_es.canonical.sequence_constraints) == 4)
ck("ES canonical has 5 vocabulary terms",                  len(prep_es.canonical.must_preserve_vocabulary) == 5)
ck("ES selected_candidate.passage_id matches",             prep_es.selected_candidate.passage_id == "sys_rev")
ck("ES selected_scores.passed_constraints (VALIDATED)",    prep_es.selected_scores.passed_constraints)
ck("ES selection_mode is VALIDATED",                       prep_es.selection_mode == ESSelectionMode.VALIDATED)
ck("ES assessment has 6 items",                            len(prep_es.assessment.items) == 6)
ck("ES cycle_id is non-empty",                             isinstance(prep_es.cycle_id, str) and len(prep_es.cycle_id) > 0)
ck("ES prepared_learner.target_band set",                  prep_es.prepared_learner.target_band is not None)
ck("ES source_fk stored on canonical",                     prep_es.canonical.source_fk > 0)
ck("ES source_fk uses readability_grade (returns float)",
   isinstance(prep_es.canonical.source_fk, float))


# ══════════════════════════════════════════════════════════════════════════════
# S10 — ES complete_cycle: WELL_CALIBRATED path
# ══════════════════════════════════════════════════════════════════════════════

group("S10 — ES complete_cycle: WELL_CALIBRATED path")

es_sys_s10 = ESSystem(llm=make_es_mock(), engine=ESDeterministicEngine(ES_LOOSE_CFG))
prep_es10 = es_sys_s10.prepare_cycle(ES_SOURCE, "sys_rev", "Comprender la RF.", ES_LEARNER)
out_es10 = es_sys_s10.complete_cycle(
    learner=ES_LEARNER,
    prep=prep_es10,
    learner_answers=ES_GOOD_ANSWERS,
    telemetry=ES_GOOD_TELEMETRY,
)

ck("ES complete_cycle returns CycleOutcome",               type(out_es10).__name__ == "CycleOutcome")
ck("ES diagnosis is WELL_CALIBRATED",                      out_es10.diagnosis == ESDiagnosisLabel.WELL_CALIBRATED)
ck("ES updated_learner differs from input",                out_es10.updated_learner != ES_LEARNER)
ck("ES comprehension_score = 1.0 (all correct)",           out_es10.assessment_result.comprehension_score == 1.0)
ck("ES retell_quality = 3/4 = 0.75",                       out_es10.assessment_result.retell_quality == 0.75)
ck("ES cycle_id preserved",                                out_es10.cycle_id == prep_es10.cycle_id)
ck("ES WELL_CALIBRATED in recent_outcomes",
   ESDiagnosisLabel.WELL_CALIBRATED in out_es10.updated_learner.recent_outcomes)
ck("ES cycles_on_passage incremented",                     out_es10.updated_learner.cycles_on_passage > 0)
ck("ES reading_signals.completion = True",                 out_es10.reading_signals.completion is True)


# ══════════════════════════════════════════════════════════════════════════════
# S11 — ES diagnosis labels and state updates
# ══════════════════════════════════════════════════════════════════════════════

group("S11 — ES diagnosis labels and state updates")

for label_str, label_enum in [
    ("underchallenged",                  ESDiagnosisLabel.UNDERCHALLENGED),
    ("well_calibrated",                  ESDiagnosisLabel.WELL_CALIBRATED),
    ("successful_but_support_dependent", ESDiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT),
    ("vocabulary_barrier",               ESDiagnosisLabel.VOCABULARY_BARRIER),
    ("overloaded",                       ESDiagnosisLabel.OVERLOADED),
]:
    _es = ESSystem(
        llm=make_es_mock(extra={"diagnose_outcome": {"diagnosis": label_str, "reason": "prueba"}}),
        engine=ESDeterministicEngine(ES_LOOSE_CFG),
    )
    _ep = _es.prepare_cycle(ES_SOURCE, "sys_rev", "test", ES_LEARNER)
    _eo = _es.complete_cycle(ES_LEARNER, _ep, ES_GOOD_ANSWERS, ES_GOOD_TELEMETRY)
    ck(f"ES diagnosis {label_str!r} fires",
       _eo.diagnosis == label_enum, f"got {_eo.diagnosis}")

# ES diagnose_fallback mirrors EN behaviour
es_eng = ESDeterministicEngine(ES_LOOSE_CFG)
from alien_system_es import ReadingSignals as ESReadingSignals
_es_sigs_good = ESReadingSignals(0.90, 0.80, 0.90, 0.05, 0, 2, 0.80, True)
_es_sigs_fail = ESReadingSignals(0.40, 0.30, 0.40, 0.40, 3, 5, 0.20, True)
ck("ES fallback UNDERCHALLENGED fires",
   es_eng.diagnose_fallback(ES_LEARNER, _es_sigs_good) == ESDiagnosisLabel.UNDERCHALLENGED)
ck("ES fallback OVERLOADED fires",
   es_eng.diagnose_fallback(ES_LEARNER, _es_sigs_fail) == ESDiagnosisLabel.OVERLOADED)

# ES update_learner_state — UNDERCHALLENGED and OVERLOADED
_es_base = ESLearnerState("es_upd", 5.0)
_eu = es_eng.update_learner_state(_es_base, ESDiagnosisLabel.UNDERCHALLENGED, _es_sigs_good)
ck("ES UNDERCHALLENGED: band increases",     _eu.current_band > _es_base.current_band)
_eo2 = es_eng.update_learner_state(_es_base, ESDiagnosisLabel.OVERLOADED, _es_sigs_fail)
ck("ES OVERLOADED: band decreases",          _eo2.current_band < _es_base.current_band)


# ══════════════════════════════════════════════════════════════════════════════
# S12 — ES configuration: SPANISH_CONFIG and readability_grade
# ══════════════════════════════════════════════════════════════════════════════

group("S12 — ES configuration: SPANISH_CONFIG and readability_grade")

ck("SPANISH_CONFIG is EngineConfig",
   type(SPANISH_CONFIG).__name__ == "EngineConfig")
ck("SPANISH_CONFIG.fk_tolerance = 1.5",
   SPANISH_CONFIG.fk_tolerance == 1.5)
ck("SPANISH_CONFIG.overall_meaning_threshold = 0.70",
   SPANISH_CONFIG.overall_meaning_threshold == 0.70)
ck("SPANISH_CONFIG.vocabulary_threshold = 0.80",
   SPANISH_CONFIG.vocabulary_threshold == 0.80)
ck("SPANISH_CONFIG.length_ceiling = 0.72",
   SPANISH_CONFIG.length_ceiling == 0.72)
ck("SPANISH_CONFIG.history_limit = 3",
   SPANISH_CONFIG.history_limit == 3)

# readability_grade returns float in [0, 12]
_rg_simple = readability_grade("El niño fue a la escuela hoy.")
_rg_hard   = readability_grade(ES_SOURCE)
ck("readability_grade returns float",            isinstance(_rg_simple, float))
ck("readability_grade result in [0, 12]",        0.0 <= _rg_simple <= 12.0)
ck("harder text scores higher",                  _rg_hard > _rg_simple)
ck("flesch_kincaid_grade is alias for readability_grade",
   es_fk(ES_SOURCE) == readability_grade(ES_SOURCE))
ck("empty text returns 0.0",                     readability_grade("") == 0.0)
ck("readability_grade handles accented chars",   readability_grade("Los niños jugaron alegremente.") > 0.0)

# PromptLibrary language switching (ES mode)
from alien_system_es import PromptLibrary as ESPromptLibrary
pl_es = ESPromptLibrary("es")
pl_en = ESPromptLibrary("en")
ck("ES PromptLibrary language=es: CANONICALIZER_SYSTEM is Spanish",
   "español" in pl_es.CANONICALIZER_SYSTEM.lower() or "pasaje" in pl_es.CANONICALIZER_SYSTEM.lower()
   or "fuente" in pl_es.CANONICALIZER_SYSTEM.lower())
ck("ES PromptLibrary language=en: CANONICALIZER_SYSTEM is English",
   "passage" in pl_en.CANONICALIZER_SYSTEM.lower() or "extract" in pl_en.CANONICALIZER_SYSTEM.lower())
ck("ES and EN prompts are different",             pl_es.CANONICALIZER_SYSTEM != pl_en.CANONICALIZER_SYSTEM)


# ══════════════════════════════════════════════════════════════════════════════
# S13 — Cross-module: ALIENError identity, shared type compatibility
# ══════════════════════════════════════════════════════════════════════════════

group("S13 — Cross-module: ALIENError identity and shared type compatibility")

# ALIENError identity
ck("ES ALIENError is EN ALIENError (same class object)",
   ES_ALIENError is ALIENError)
ck("ES ValidationError subclasses EN ALIENError",
   issubclass(ES_ValidationError, ALIENError))

# EN except clause catches ES errors
_caught_es_via_en = False
try:
    raise ES_ValidationError("stage", "test")
except ALIENError:
    _caught_es_via_en = True
ck("ES ValidationError caught by except EN ALIENError",  _caught_es_via_en)

_caught_es_alien_via_en = False
try:
    raise ES_ALIENError("s", "m")
except ALIENError:
    _caught_es_alien_via_en = True
ck("ES ALIENError caught by except EN ALIENError",       _caught_es_alien_via_en)

# EN and ES define their own enum classes (independently) but with identical values.
# These tests confirm structural parity — not object identity.
ck("ES Level values match EN Level values",
   {e.value for e in ESLevel} == {e.value for e in Level})
ck("ES DiagnosisLabel values match EN DiagnosisLabel values",
   {e.value for e in ESDiagnosisLabel} == {e.value for e in DiagnosisLabel})
ck("ES SelectionMode values match EN SelectionMode values",
   {e.value for e in ESSelectionMode} == {e.value for e in SelectionMode})

# EN LearnerState is duck-type compatible with ES DeterministicEngine
# (same frozen dataclass fields); diagnose_fallback returns ES DiagnosisLabel.
_ls_en = LearnerState("shared", 5.0)
ck("EN LearnerState accepted by ES DeterministicEngine",
   isinstance(es_eng.diagnose_fallback(_ls_en, _es_sigs_good), ESDiagnosisLabel))

# EN and ES systems can operate in the same process without interference
_en_in_parallel = AdaptiveReadingSystem(llm=make_en_mock(), engine=DeterministicEngine(LOOSE_CFG))
_es_in_parallel = ESSystem(llm=make_es_mock(), engine=ESDeterministicEngine(ES_LOOSE_CFG))
_prep_en_p = _en_in_parallel.prepare_cycle(EN_SOURCE, "sys_gut", "obj", EN_LEARNER)
_prep_es_p = _es_in_parallel.prepare_cycle(ES_SOURCE, "sys_rev", "obj", ES_LEARNER)
ck("EN and ES systems run in parallel without interference",
   _prep_en_p.canonical.passage_id == "sys_gut"
   and _prep_es_p.canonical.passage_id == "sys_rev")
ck("EN canonical unaffected by ES run",   _prep_en_p.canonical.passage_id == "sys_gut")
ck("ES canonical unaffected by EN run",   _prep_es_p.canonical.passage_id == "sys_rev")

# LearnerState JSON round-trip (EN)
_ls_rt = LearnerState("rt", 6.5, vocabulary_need=Level.HIGH,
                      recent_outcomes=(DiagnosisLabel.WELL_CALIBRATED,))
_js = _ls_rt.to_json()
_ls_rt2 = LearnerState.from_json(_js)
ck("LearnerState to_json/from_json round-trip: id",              _ls_rt2.learner_id == "rt")
ck("LearnerState to_json/from_json round-trip: current_band",    _ls_rt2.current_band == 6.5)
ck("LearnerState to_json/from_json round-trip: vocabulary_need", _ls_rt2.vocabulary_need == Level.HIGH)
ck("LearnerState to_json/from_json round-trip: recent_outcomes",
   DiagnosisLabel.WELL_CALIBRATED in _ls_rt2.recent_outcomes)


# ══════════════════════════════════════════════════════════════════════════════
# Final report
# ══════════════════════════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{'═'*65}")
print(f"  {'PASS ✓' if failed == 0 else 'FAIL ✗'}  │  "
      f"{passed} passed  │  {failed} failed  │  {total} total")
print(f"{'═'*65}")

import sys as _sys
_sys.exit(0 if failed == 0 else 1)
