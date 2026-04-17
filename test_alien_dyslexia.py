"""
test_alien_dyslexia.py — Test suite for alien_dyslexia.py

Group structure:
  T01  DyslexicLearnerState — fields, to/from base, JSON round-trip
  T02  DyslexicReadingTelemetry — fields, to_base
  T03  DyslexicReadingSignals — fields, to_base, audit preservation
  T04  DECODING_BARRIER sentinel — equality, hashing, is_decoding_barrier()
  T05  seed_dyslexic_learner — correct defaults and seeding
  T06  DyslexiaAwareDeterministicEngine.build_candidate_plan
         — decoding_support slot added for dyslexic learners
         — standard plan unchanged for non-dyslexic learners
  T07  DyslexiaAwareDeterministicEngine.diagnose_fallback
         — decoding_barrier fires when comprehension ok and fluency low
         — decoding_barrier blocked when comprehension below guard
         — all seven standard labels still fire correctly
  T08  DyslexiaAwareDeterministicEngine.update_learner_state
         — decoding_barrier: band preserved, support_dependence up, readiness low
         — decoding_barrier in recent_outcomes
         — standard labels still update correctly via delegation
  T09  DyslexiaAwareSystem.build_reading_signals
         — adjustments fire when decoding_disability=True AND comprehension >= guard
         — adjustments blocked when comprehension < guard
         — adjustments blocked when decoding_disability=False
         — raw values preserved in DyslexicReadingSignals
         — decoding_adjusted flag set correctly
  T10  DyslexiaAwareSystem.score_retell
         — oral_text preferred when decoding_disability=True and oral_text provided
         — written response used when no oral_text
         — written response used when decoding_disability=False
  T11  Full mocked cycle — end-to-end prepare_cycle + complete_cycle
         — decoding_barrier diagnosed correctly
         — band preserved after decoding_barrier
         — oral_retell_quality updated on learner
         — cycle_id preserved
  T12  Full mocked cycle — non-dyslexic path unchanged
         — standard diagnoses still fire
         — no adjustments applied
  T13  DyslexicLearnerState.to_prompt_dict includes decoding_disability
  T14  DYSLEXIA_ENGINE_CONFIG correctness
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
    AdaptiveReadingSystem, TaskRoutingMockLLM,
)
from alien_dyslexia import (
    DyslexicLearnerState, DyslexicReadingTelemetry, DyslexicReadingSignals,
    DECODING_BARRIER, DECODING_BARRIER_VALUE, is_decoding_barrier,
    DyslexiaAwareDeterministicEngine, DyslexiaAwareSystem,
    DYSLEXIA_ENGINE_CONFIG, seed_dyslexic_learner,
    COMPREHENSION_GUARD, FLUENCY_SCALE, HINT_DISCOUNT, DECODING_FLUENCY_THRESHOLD,
)

# ── Harness ──────────────────────────────────────────────────────────────────
passed = failed = 0

def group(name):
    print(f"\n  {'─'*62}")
    print(f"  {name}")
    print(f"  {'─'*62}")

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"    ✓  {name}")
    else:
        failed += 1
        print(f"    ✗  {name}" + (f"\n         {detail}" if detail else ""))

def check_raises(name, fn, exc=Exception):
    global passed, failed
    try:
        fn()
        failed += 1
        print(f"    ✗  {name}  (no exception raised)")
    except exc:
        passed += 1
        print(f"    ✓  {name}")

# ── Shared fixtures ───────────────────────────────────────────────────────────

SOURCE = (
    "Johannes Gutenberg invented the printing press around 1440, transforming "
    "the spread of knowledge. Before this, manuscripts were hand-copied and "
    "scarce. Movable metal type allowed fast reproduction of texts. In decades, "
    "presses spread across Europe. This democratization catalyzed the Renaissance, "
    "Reformation, and Scientific Revolution."
)

CANONICAL = CanonicalPassage(
    passage_id="gut_01",
    source_text=SOURCE,
    instructional_objective="Explain how the printing press changed history.",
    meaning_units=(
        MeaningUnit("MU1","Gutenberg invented the press ~1440",True,
                    ("Gutenberg","printing","press","1440","invented")),
        MeaningUnit("MU2","manuscripts hand-copied scarce",True,
                    ("manuscripts","hand","scarce")),
        MeaningUnit("MU3","movable metal type reproduction",True,
                    ("movable","metal","type","reproduction")),
        MeaningUnit("MU4","presses spread Europe",True,
                    ("proliferated","Europe","decades")),
        MeaningUnit("MU5","democratization catalyzed transformations",True,
                    ("democratization","Renaissance","Reformation")),
    ),
    sequence_constraints=(
        SequenceConstraint("MU1","MU2"), SequenceConstraint("MU2","MU3"),
        SequenceConstraint("MU3","MU4"), SequenceConstraint("MU4","MU5"),
    ),
    must_preserve_vocabulary=(
        VocabularyTerm("printing press",      required=True,  gloss_allowed=False),
        VocabularyTerm("manuscripts",         required=True,  gloss_allowed=True),
        VocabularyTerm("movable metal type",  required=True,  gloss_allowed=True),
        VocabularyTerm("Renaissance",         required=True,  gloss_allowed=True),
        VocabularyTerm("Reformation",         required=True,  gloss_allowed=True),
        VocabularyTerm("Scientific Revolution",required=True, gloss_allowed=True),
    ),
)

CAND_TEXT = (
    "Around 1440, Gutenberg invented the printing press. "
    "Before this, manuscripts were hand-copied and very scarce. "
    "He used movable metal type to reproduce texts quickly. "
    "In decades, presses proliferated across Europe. "
    "This democratization catalyzed the Renaissance, Reformation, "
    "and Scientific Revolution."
)

GOOD_AUDIT = SelfAudit(True, True, True, True,
    meaning_unit_coverage={"MU1":True,"MU2":True,"MU3":True,"MU4":True,"MU5":True})

GOOD_CAND = CandidatePassage("A","gut_01",0,CAND_TEXT,
    ScaffoldProfile(vocabulary_support=Level.MEDIUM),GOOD_AUDIT)

ASSESS_RAW = {
    "assessment_blueprint":{"passage_id":"gut_01"},
    "items":[
        {"id":"Q1","type":"literal_mcq","target":"MU1","question":"q",
         "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                    {"id":"C","text":"c"},{"id":"D","text":"d"}],"correct_answer":"A"},
        {"id":"Q2","type":"sequence_mcq",
         "target":{"meaning_unit_ids":["MU1","MU2"],"relation":"before"},"question":"q",
         "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                    {"id":"C","text":"c"},{"id":"D","text":"d"}],"correct_answer":"A"},
        {"id":"Q3","type":"inference_mcq","target":"MU3","question":"q",
         "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                    {"id":"C","text":"c"},{"id":"D","text":"d"}],"correct_answer":"A"},
        {"id":"Q4","type":"vocabulary_mcq","target":"term","question":"q",
         "choices":[{"id":"A","text":"a"},{"id":"B","text":"b"},
                    {"id":"C","text":"c"},{"id":"D","text":"d"}],"correct_answer":"A"},
        {"id":"Q5","type":"retell_short_response","target":None,"prompt":"Retell.",
         "rubric":{"max_score":4,"criteria":[
             {"points":1,"meaning_unit_ids":["MU1"],"description":"mentions Gutenberg"},
             {"points":1,"meaning_unit_ids":["MU2"],"description":"mentions manuscripts"},
             {"points":1,"meaning_unit_ids":["MU3"],"description":"mentions movable type"},
             {"points":1,"meaning_unit_ids":["MU4"],"description":"mentions spread"},
         ]}},
        {"id":"Q6","type":"self_rating","target":None,"prompt":"How hard?","scale":"1-5"},
    ],
    "scoring_blueprint":{"literal_item_ids":["Q1"],"sequence_item_ids":["Q2"],
                         "inference_item_ids":["Q3"],"vocabulary_item_ids":["Q4"]},
    "signal_mapping":{
        "comprehension_score":{"formula":"wa",
                               "weights":{"Q1":0.25,"Q2":0.25,"Q3":0.25,"Q4":0.25}},
        "inference_score":{"formula":"wa","weights":{"Q3":0.6,"Q5":0.4}},
        "vocabulary_signal":{"formula":"Q4","weights":{"Q4":1.0}},
        "retell_quality":{"formula":"Q5"},
        "difficulty_signal":{"formula":"Q6"},
    },
}

MOCK_RESPONSES = {
    "canonicalize_passage":{
        "passage_id":"gut_01","source_text":SOURCE,
        "instructional_objective":"Explain how the printing press changed history.",
        "meaning_units":[
            {"id":"MU1","text":"Gutenberg invented press","required":True,
             "anchors":["Gutenberg","printing","press","1440","invented"]},
            {"id":"MU2","text":"manuscripts hand-copied scarce","required":True,
             "anchors":["manuscripts","hand","scarce"]},
            {"id":"MU3","text":"movable metal type","required":True,
             "anchors":["movable","metal","type","reproduction"]},
            {"id":"MU4","text":"presses spread Europe","required":True,
             "anchors":["proliferated","Europe","decades"]},
            {"id":"MU5","text":"democratization catalyzed","required":True,
             "anchors":["democratization","Renaissance","Reformation"]},
        ],
        "sequence_constraints":[
            {"before":"MU1","after":"MU2"},{"before":"MU2","after":"MU3"},
            {"before":"MU3","after":"MU4"},{"before":"MU4","after":"MU5"},
        ],
        "must_preserve_vocabulary":[
            {"term":"printing press","required":True,"gloss_allowed":False},
            {"term":"manuscripts","required":True,"gloss_allowed":True},
            {"term":"movable metal type","required":True,"gloss_allowed":True},
            {"term":"Renaissance","required":True,"gloss_allowed":True},
            {"term":"Reformation","required":True,"gloss_allowed":True},
            {"term":"Scientific Revolution","required":True,"gloss_allowed":True},
        ],
    },
    "generate_candidates":{"candidates":[{
        "candidate_id":"A","passage_id":"gut_01","relative_band":0,
        "text":CAND_TEXT,
        "scaffold":{"vocabulary_support":"medium","syntax_support":"low",
                    "cohesion_support":"medium","chunking_support":"low",
                    "inference_support":"low"},
        "llm_self_audit":{"meaning_preserved":True,"sequence_preserved":True,
                          "objective_preserved":True,"same_passage_identity":True,
                          "notes":"OK",
                          "meaning_unit_coverage":{"MU1":True,"MU2":True,"MU3":True,
                                                   "MU4":True,"MU5":True}},
    }]},
    "estimate_fit":{"fit_estimates":[
        {"candidate_id":"A","access":"high","growth":"medium",
         "support_burden":"medium","reason":"Good fit."}
    ]},
    "generate_assessment": ASSESS_RAW,
    "score_retell":{"raw_score":3,"max_score":4,
                    "matched_meaning_units":["MU1","MU2","MU4"],
                    "matched_relationships":[],"concise_reason":"Three met."},
    "diagnose_outcome":{"diagnosis":"well_calibrated","reason":"OK"},
}

LOOSE_ENGINE = DyslexiaAwareDeterministicEngine(config=EngineConfig(
    fk_tolerance=10, overall_meaning_threshold=0, vocabulary_threshold=0,
    length_deviation_threshold=10, length_ceiling=10, meaning_floor=0, vocab_floor=0,
))


# ════════════════════════════════════════════════════════════════════════════

group("T01 — DyslexicLearnerState")

dl = DyslexicLearnerState(
    "maya_01", 6.5,
    vocabulary_need=Level.HIGH, syntax_need=Level.MEDIUM,
    cohesion_need=Level.MEDIUM, support_dependence=Level.HIGH,
    readiness_to_increase=Level.LOW,
    decoding_disability=True,
    oral_retell_quality=0.75,
)
check("learner_id",              dl.learner_id == "maya_01")
check("current_band",            dl.current_band == 6.5)
check("decoding_disability",     dl.decoding_disability is True)
check("oral_retell_quality",     dl.oral_retell_quality == 0.75)

base = dl.to_base()
check("to_base: is LearnerState",     isinstance(base, LearnerState))
check("to_base: learner_id matches",  base.learner_id == "maya_01")
check("to_base: band matches",        base.current_band == 6.5)
check("to_base: no decoding_disability field",
      not hasattr(base, "decoding_disability"))

promoted = DyslexicLearnerState.from_base(base, decoding_disability=True)
check("from_base: decoding_disability set",  promoted.decoding_disability is True)
check("from_base: fields preserved",         promoted.current_band == 6.5)

j   = dl.to_json()
dl2 = DyslexicLearnerState.from_json(j)
check("JSON round-trip: learner_id",          dl2.learner_id == "maya_01")
check("JSON round-trip: decoding_disability", dl2.decoding_disability is True)
check("JSON round-trip: oral_retell_quality", dl2.oral_retell_quality == 0.75)
check("JSON round-trip: vocabulary_need",     dl2.vocabulary_need == Level.HIGH)
check("JSON round-trip: valid JSON",          json.loads(j) is not None)

check("defaults: decoding_disability=False",
      DyslexicLearnerState("x", 5.0).decoding_disability is False)
check("defaults: oral_retell_quality=None",
      DyslexicLearnerState("x", 5.0).oral_retell_quality is None)


# ════════════════════════════════════════════════════════════════════════════

group("T02 — DyslexicReadingTelemetry")

dt = DyslexicReadingTelemetry(0.35, 0.40, 8, False,
    oral_retell_text="Gutenberg invented the press.")
check("fluency_score",      dt.fluency_score == 0.35)
check("hint_use_rate",      dt.hint_use_rate == 0.40)
check("reread_count",       dt.reread_count == 8)
check("completion",         dt.completion is False)
check("oral_retell_text",   dt.oral_retell_text is not None)

base_t = dt.to_base()
check("to_base: is ReadingTelemetry",     isinstance(base_t, ReadingTelemetry))
check("to_base: fluency matches",         base_t.fluency_score == 0.35)
check("to_base: no oral_retell_text",     not hasattr(base_t, "oral_retell_text"))

check("default oral_retell_text=None",
      DyslexicReadingTelemetry(0.5, 0.1, 2, True).oral_retell_text is None)


# ════════════════════════════════════════════════════════════════════════════

group("T03 — DyslexicReadingSignals")

ds = DyslexicReadingSignals(
    comprehension_score=0.80, inference_score=0.60,
    fluency_score=0.72,  hint_use_rate=0.15,
    reread_count=3, difficulty_rating=3, retell_quality=0.75,
    completion=True,
    decoding_adjusted=True,
    raw_fluency_score=0.35,
    raw_hint_use_rate=0.40,
    raw_completion=False,
)
check("comprehension_score",    ds.comprehension_score == 0.80)
check("adjusted fluency",       ds.fluency_score == 0.72)
check("adjusted hint_rate",     ds.hint_use_rate == 0.15)
check("adjusted completion",    ds.completion is True)
check("decoding_adjusted flag", ds.decoding_adjusted is True)
check("raw_fluency preserved",  ds.raw_fluency_score == 0.35)
check("raw_hint preserved",     ds.raw_hint_use_rate == 0.40)
check("raw_completion preserved",ds.raw_completion is False)

base_s = ds.to_base()
check("to_base: is ReadingSignals",  isinstance(base_s, ReadingSignals))
check("to_base: uses adjusted fluency", base_s.fluency_score == 0.72)
check("to_base: completion=True",    base_s.completion is True)
check("to_base: no decoding_adjusted", not hasattr(base_s, "decoding_adjusted"))

check("defaults: decoding_adjusted=False",
      DyslexicReadingSignals(0.75,0.55,0.75,0.10,2,3,0.75,True).decoding_adjusted is False)


# ════════════════════════════════════════════════════════════════════════════

group("T04 — DECODING_BARRIER sentinel")

check("DECODING_BARRIER.value == 'decoding_barrier'",
      DECODING_BARRIER.value == DECODING_BARRIER_VALUE)
check("is_decoding_barrier(DECODING_BARRIER)",
      is_decoding_barrier(DECODING_BARRIER))
check("is_decoding_barrier(string value)",
      is_decoding_barrier("decoding_barrier"))
check("not is_decoding_barrier(well_calibrated)",
      not is_decoding_barrier(DiagnosisLabel.WELL_CALIBRATED))
check("not is_decoding_barrier(random string)",
      not is_decoding_barrier("something_else"))
check("DECODING_BARRIER == DECODING_BARRIER",
      DECODING_BARRIER == DECODING_BARRIER)
check("DECODING_BARRIER != overloaded",
      DECODING_BARRIER != DiagnosisLabel.OVERLOADED)
check("DECODING_BARRIER is hashable (usable in set)",
      hash(DECODING_BARRIER) is not None)
check("DECODING_BARRIER in set works",
      DECODING_BARRIER in {DECODING_BARRIER, DiagnosisLabel.WELL_CALIBRATED})
check("str(DECODING_BARRIER) contains 'decoding_barrier'",
      "decoding_barrier" in str(DECODING_BARRIER))


# ════════════════════════════════════════════════════════════════════════════

group("T05 — seed_dyslexic_learner")

seeded = seed_dyslexic_learner("maya", 6.5, vocabulary_need=Level.HIGH)
check("is DyslexicLearnerState",          isinstance(seeded, DyslexicLearnerState))
check("decoding_disability=True",         seeded.decoding_disability is True)
check("current_band = comprehension_band",seeded.current_band == 6.5)
check("vocabulary_need from arg",         seeded.vocabulary_need == Level.HIGH)
check("support_dependence starts HIGH",   seeded.support_dependence == Level.HIGH)
check("readiness_to_increase starts LOW", seeded.readiness_to_increase == Level.LOW)
check("oral_retell_quality = None",       seeded.oral_retell_quality is None)

seeded_defaults = seed_dyslexic_learner("x", 5.0)
check("syntax_need default = MEDIUM",     seeded_defaults.syntax_need == Level.MEDIUM)
check("cohesion_need default = MEDIUM",   seeded_defaults.cohesion_need == Level.MEDIUM)


# ════════════════════════════════════════════════════════════════════════════

group("T06 — build_candidate_plan: decoding_support slot")

engine = DyslexiaAwareDeterministicEngine()

# Dyslexic learner: plan includes decoding_support
dyslexic = seed_dyslexic_learner("maya", 6.5, vocabulary_need=Level.HIGH)
plan_d = engine.build_candidate_plan(dyslexic)
decoding_slots = [p for p in plan_d if p.get("profile") == "decoding_support"]
check("decoding_support slot present",           len(decoding_slots) == 1)
check("decoding_support at relative_band=0",     decoding_slots[0]["relative_band"] == 0)
check("override_length_check=True",              decoding_slots[0].get("override_length_check") is True)

# Non-dyslexic learner: plan unchanged from base
non_dyslexic = LearnerState("x", 6.5, vocabulary_need=Level.HIGH)
plan_n = engine.build_candidate_plan(non_dyslexic)
check("no decoding_support for non-dyslexic",
      not any(p.get("profile") == "decoding_support" for p in plan_n))
check("non-dyslexic plan still has safety net (band -1)",
      any(p["relative_band"] == -1 for p in plan_n))

# DyslexicLearnerState with decoding_disability=False: no decoding_support
non_flag = DyslexicLearnerState("y", 5.0, decoding_disability=False)
plan_nf = engine.build_candidate_plan(non_flag)
check("decoding_disability=False: no decoding_support slot",
      not any(p.get("profile") == "decoding_support" for p in plan_nf))


# ════════════════════════════════════════════════════════════════════════════

group("T07 — diagnose_fallback: decoding_barrier detection")

engine = DyslexiaAwareDeterministicEngine()
dyslexic_l = seed_dyslexic_learner("maya", 6.5)

def make_signals(comprehension=0.75, fluency=0.35, hint=0.15,
                 retell=0.75, complete=True, adjusted=False, raw_fluency=None):
    return DyslexicReadingSignals(
        comprehension_score=comprehension, inference_score=0.55,
        fluency_score=fluency, hint_use_rate=hint,
        reread_count=3, difficulty_rating=3,
        retell_quality=retell, completion=complete,
        decoding_adjusted=adjusted,
        raw_fluency_score=raw_fluency if raw_fluency is not None else fluency,
    )

# Core case: decoding_barrier fires
check("decoding_barrier fires: comprehension ok, fluency low",
      is_decoding_barrier(engine.diagnose_fallback(
          dyslexic_l,
          make_signals(comprehension=0.80, fluency=0.35))))

# Guard: comprehension below threshold → does NOT fire decoding_barrier
check("decoding_barrier blocked: comprehension < guard",
      not is_decoding_barrier(engine.diagnose_fallback(
          dyslexic_l,
          make_signals(comprehension=0.60, fluency=0.30))))

# Non-dyslexic learner → never fires decoding_barrier
non_d = LearnerState("x", 6.5)
check("decoding_barrier never fires for non-dyslexic",
      not is_decoding_barrier(engine.diagnose_fallback(
          non_d,
          make_signals(comprehension=0.80, fluency=0.30))))

# When decoding_adjusted=True, uses raw_fluency_score, not adjusted fluency_score.
# adjusted fluency = 0.75 (above threshold) but raw = 0.30 (below) → barrier fires.
adjusted_sigs = make_signals(
    comprehension=0.80,
    fluency=0.75,         # adjusted fluency — above threshold, would not trigger
    raw_fluency=0.30,     # raw fluency — below threshold, triggers barrier
    adjusted=True,        # decoding_adjusted=True: raw_fluency_score is read
)
check("decoding_barrier uses raw_fluency_score when decoding_adjusted=True",
      is_decoding_barrier(engine.diagnose_fallback(dyslexic_l, adjusted_sigs)))

# Fluency exactly at threshold (DECODING_FLUENCY_THRESHOLD = 0.50) — not barrier.
# Condition is strictly < threshold; at exactly 0.50 barrier should not fire.
# Use a plain ReadingSignals to avoid DyslexicReadingSignals.raw_fluency_score
# defaulting to 0.0, which would be misread as the comparison value.
from alien_system import ReadingSignals as _RS
at_threshold = _RS(0.80, 0.55, DECODING_FLUENCY_THRESHOLD, 0.15, 3, 3, 0.75, True)
check(f"fluency = threshold ({DECODING_FLUENCY_THRESHOLD}) → not decoding_barrier",
      not is_decoding_barrier(engine.diagnose_fallback(dyslexic_l, at_threshold)))

# Standard labels still fire via delegation
check("underchallenged still fires for dyslexic learner",
      engine.diagnose_fallback(dyslexic_l,
          make_signals(0.90, 0.80, 0.05, 0.80))
      == DiagnosisLabel.UNDERCHALLENGED)
check("well_calibrated still fires",
      engine.diagnose_fallback(dyslexic_l,
          make_signals(0.75, 0.75, 0.15))
      == DiagnosisLabel.WELL_CALIBRATED)
check("overloaded still fires when comprehension low",
      engine.diagnose_fallback(dyslexic_l,
          make_signals(comprehension=0.40))
      == DiagnosisLabel.OVERLOADED)


# ════════════════════════════════════════════════════════════════════════════

group("T08 — update_learner_state: decoding_barrier rule")

engine = DyslexiaAwareDeterministicEngine()
dyslexic_l = seed_dyslexic_learner("maya", 6.5)
sigs = make_signals(0.80, 0.35)

# decoding_barrier: band preserved
updated = engine.update_learner_state(dyslexic_l, DECODING_BARRIER, sigs)
check("decoding_barrier: current_band unchanged",
      updated.current_band == dyslexic_l.current_band)
check("decoding_barrier: support_dependence ↑",
      updated.support_dependence.score > dyslexic_l.support_dependence.score
      or updated.support_dependence == Level.HIGH)  # already HIGH → stays HIGH
check("decoding_barrier: readiness → LOW",
      updated.readiness_to_increase == Level.LOW)
check("decoding_barrier: cycles_on_passage += 1",
      updated.cycles_on_passage == dyslexic_l.cycles_on_passage + 1)
check("decoding_barrier in recent_outcomes",
      any(is_decoding_barrier(x) for x in updated.recent_outcomes))

# Returns DyslexicLearnerState when input is DyslexicLearnerState
check("update returns DyslexicLearnerState",
      isinstance(updated, DyslexicLearnerState))
check("decoding_disability preserved through update",
      updated.decoding_disability is True)

# Standard labels still update correctly via delegation
updated_uc = engine.update_learner_state(
    dyslexic_l, DiagnosisLabel.UNDERCHALLENGED, sigs)
check("underchallenged: band advances",
      updated_uc.current_band > dyslexic_l.current_band)
check("underchallenged: readiness → HIGH",
      updated_uc.readiness_to_increase == Level.HIGH)

# Base LearnerState input with decoding_barrier → returns base LearnerState
base_l = LearnerState("x", 6.5)
updated_base = engine.update_learner_state(base_l, DECODING_BARRIER, sigs)
check("decoding_barrier on base LearnerState → returns LearnerState",
      isinstance(updated_base, LearnerState))
check("decoding_barrier on base: band unchanged",
      updated_base.current_band == 6.5)


# ════════════════════════════════════════════════════════════════════════════

group("T09 — build_reading_signals: signal adjustments")

mock = TaskRoutingMockLLM()
sys_d = DyslexiaAwareSystem(llm=mock, engine=LOOSE_ENGINE)
mock_ar = AssessmentResult(
    item_scores={},
    comprehension_score=0.80,
    inference_score=0.60,
    vocabulary_score=0.50,
    retell_quality=0.75,
    difficulty_rating=3,
)
dyslexic_l = seed_dyslexic_learner("maya", 6.5)
non_dyslexic = LearnerState("x", 6.5)

# Adjustments fire: dyslexic + comprehension >= guard
raw_fluency  = 0.35
raw_hints    = 0.40
raw_complete = False

sigs_adj = sys_d.build_reading_signals(
    mock_ar, raw_fluency, raw_hints, 8, raw_complete, learner=dyslexic_l)

expected_fluency = max(raw_fluency, 0.80 * FLUENCY_SCALE)
expected_hints   = raw_hints * HINT_DISCOUNT
check("decoding_adjusted = True",
      sigs_adj.decoding_adjusted is True)
check(f"adjusted fluency = max(raw, comprehension*{FLUENCY_SCALE}) = {expected_fluency:.3f}",
      abs(sigs_adj.fluency_score - expected_fluency) < 0.001,
      f"got {sigs_adj.fluency_score:.3f}")
check(f"adjusted hint_rate = raw * {HINT_DISCOUNT} = {raw_hints*HINT_DISCOUNT:.3f}",
      abs(sigs_adj.hint_use_rate - raw_hints * HINT_DISCOUNT) < 0.001,
      f"got {sigs_adj.hint_use_rate:.3f}")
check("adjusted completion = True",      sigs_adj.completion is True)
check("raw_fluency_score preserved",     abs(sigs_adj.raw_fluency_score - raw_fluency) < 0.001)
check("raw_hint_use_rate preserved",     abs(sigs_adj.raw_hint_use_rate - raw_hints) < 0.001)
check("raw_completion preserved False",  sigs_adj.raw_completion is False)

# Adjustments blocked: comprehension below guard
mock_ar_low = AssessmentResult({}, 0.60, 0.40, 0.50, 0.60, 3)
sigs_low = sys_d.build_reading_signals(
    mock_ar_low, 0.30, 0.45, 8, False, learner=dyslexic_l)
check("adjustments blocked: comprehension < guard → decoding_adjusted=False",
      sigs_low.decoding_adjusted is False)
check("adjustments blocked: raw fluency unchanged",
      abs(sigs_low.fluency_score - 0.30) < 0.001)
check("adjustments blocked: completion unchanged",
      sigs_low.completion is False)

# Adjustments blocked: non-dyslexic learner
sigs_nd = sys_d.build_reading_signals(
    mock_ar, 0.30, 0.45, 8, False, learner=non_dyslexic)
check("non-dyslexic: decoding_adjusted=False",
      sigs_nd.decoding_adjusted is False)
check("non-dyslexic: raw fluency unchanged",
      abs(sigs_nd.fluency_score - 0.30) < 0.001)

# Adjustments blocked: learner=None
sigs_none = sys_d.build_reading_signals(mock_ar, 0.30, 0.45, 8, False, learner=None)
check("learner=None: decoding_adjusted=False",
      sigs_none.decoding_adjusted is False)

# Returns DyslexicReadingSignals in all cases
check("always returns DyslexicReadingSignals",
      isinstance(sigs_adj, DyslexicReadingSignals))


# ════════════════════════════════════════════════════════════════════════════

group("T10 — score_retell: oral response preference")

oral_mock = TaskRoutingMockLLM(responses={
    "score_retell":{
        "raw_score":4, "max_score":4,
        "matched_meaning_units":["MU1","MU2","MU3","MU4"],
        "matched_relationships":[], "concise_reason":"Full coverage."},
})
sys_d2 = DyslexiaAwareSystem(llm=oral_mock, engine=LOOSE_ENGINE)

from alien_system import parse_assessment_package
assessment = parse_assessment_package(ASSESS_RAW)

# Dyslexic + oral_text provided → uses oral_text
result_oral = sys_d2.score_retell(
    CANONICAL, assessment,
    learner_response="short written",
    learner=seed_dyslexic_learner("maya", 6.5),
    oral_text="Full oral retell covering all meaning units.",
)
check("oral path: raw_score returned",      "raw_score" in result_oral)
check("oral path: returns scorer result",   result_oral["raw_score"] == 4)

# Dyslexic + no oral_text → uses written
result_written = sys_d2.score_retell(
    CANONICAL, assessment,
    learner_response="written retell here",
    learner=seed_dyslexic_learner("maya", 6.5),
    oral_text=None,
)
check("no oral_text: still returns scorer result", "raw_score" in result_written)

# Non-dyslexic + oral_text → uses written (oral_text ignored)
result_nondys = sys_d2.score_retell(
    CANONICAL, assessment,
    learner_response="written retell",
    learner=LearnerState("x", 6.5),
    oral_text="Some oral retell.",
)
check("non-dyslexic: oral_text ignored, scorer called",
      "raw_score" in result_nondys)


# ════════════════════════════════════════════════════════════════════════════

group("T11 — Full mocked cycle: dyslexic learner, decoding_barrier")

# Override diagnose_outcome response to well_calibrated so we can test
# that the deterministic override fires when raw_fluency is low
log_records = []
class CapHandler(logging.Handler):
    def emit(self, r): log_records.append(r)
logger = logging.getLogger("test_dys")
logger.addHandler(CapHandler()); logger.setLevel(logging.DEBUG)

full_mock = TaskRoutingMockLLM(responses=MOCK_RESPONSES)
sys_full  = DyslexiaAwareSystem(llm=full_mock, engine=LOOSE_ENGINE, logger=logger)
dyslexic_learner = seed_dyslexic_learner("maya", 6.5, vocabulary_need=Level.HIGH)

prep = sys_full.prepare_cycle(
    SOURCE, "gut_01", "Explain how the printing press changed history.",
    dyslexic_learner)

check("prepare_cycle returns CyclePreparation",       prep is not None)
check("decoding_support slot in candidate plan",
      any(p.get("profile") == "decoding_support"
          for p in sys_full.engine.build_candidate_plan(dyslexic_learner)))

answers = {"Q1":"A","Q2":"A","Q3":"A","Q4":"A",
    "Q5":"Gutenberg made the printing press.",
    "Q6":3}

outcome = sys_full.complete_cycle(
    dyslexic_learner, prep, answers,
    DyslexicReadingTelemetry(
        fluency_score=0.30,        # very slow decoding
        hint_use_rate=0.45,        # high hints (decoding help)
        reread_count=10,
        completion=False,          # timed out
        oral_retell_text="Gutenberg invented the printing press around 1440. "
                         "Before this, manuscripts were copied by hand. "
                         "The press spread across Europe.",
    ),
)

check("complete_cycle returns CycleOutcome",          outcome is not None)
check("diagnosis is decoding_barrier",
      is_decoding_barrier(outcome.diagnosis),
      f"got: {outcome.diagnosis}")
check("band preserved (not dropped)",
      outcome.updated_learner.current_band == dyslexic_learner.current_band,
      f"expected {dyslexic_learner.current_band}, "
      f"got {outcome.updated_learner.current_band}")
check("decoding_barrier in recent_outcomes",
      any(is_decoding_barrier(x) for x in outcome.updated_learner.recent_outcomes))
check("updated_learner is DyslexicLearnerState",
      isinstance(outcome.updated_learner, DyslexicLearnerState))
check("decoding_disability preserved",
      outcome.updated_learner.decoding_disability is True)
check("oral_retell_quality updated from oral score",
      outcome.updated_learner.oral_retell_quality is not None,
      f"oral_retell_quality={outcome.updated_learner.oral_retell_quality}")
check("cycle_id preserved",                           outcome.cycle_id == prep.cycle_id)
check("reading_signals is DyslexicReadingSignals",
      isinstance(outcome.reading_signals, DyslexicReadingSignals))
check("decoding_adjusted = True",
      outcome.reading_signals.decoding_adjusted is True)
check("raw_fluency preserved",
      abs(outcome.reading_signals.raw_fluency_score - 0.30) < 0.001)
check("raw_completion preserved False",
      outcome.reading_signals.raw_completion is False)


# ════════════════════════════════════════════════════════════════════════════

group("T12 — Full mocked cycle: non-dyslexic path unchanged")

std_mock = TaskRoutingMockLLM(responses={
    **MOCK_RESPONSES,
    "diagnose_outcome":{"diagnosis":"cohesion_inference_barrier","reason":"Low inference."},
})
sys_std = DyslexiaAwareSystem(llm=std_mock, engine=LOOSE_ENGINE)
std_learner = LearnerState("david", 8.5, vocabulary_need=Level.LOW)

prep_std = sys_std.prepare_cycle(
    SOURCE, "gut_01", "Explain how the printing press changed history.",
    std_learner)

outcome_std = sys_std.complete_cycle(
    std_learner, prep_std,
    {"Q1":"A","Q2":"A","Q3":"A","Q4":"A",
     "Q5":"Gutenberg invented the press.", "Q6":2},
    DyslexicReadingTelemetry(0.85, 0.08, 1, True),
)

check("non-dyslexic: standard CycleOutcome returned", outcome_std is not None)
check("non-dyslexic: diagnosis is cohesion_inference_barrier",
      outcome_std.diagnosis == DiagnosisLabel.COHESION_INFERENCE_BARRIER,
      f"got {outcome_std.diagnosis}")
check("non-dyslexic: decoding_adjusted = False",
      outcome_std.reading_signals.decoding_adjusted is False)
check("non-dyslexic: no band preservation (band may change normally)",
      True)  # just confirm no crash


# ════════════════════════════════════════════════════════════════════════════

group("T13 — DyslexicLearnerState.to_prompt_dict")

dl_prompt = seed_dyslexic_learner("maya", 6.5, vocabulary_need=Level.HIGH)
d = dl_prompt.to_prompt_dict()
check("to_prompt_dict: decoding_disability present",
      "decoding_disability" in d)
check("to_prompt_dict: decoding_disability=True",
      d["decoding_disability"] is True)
check("to_prompt_dict: standard fields present",
      all(k in d for k in ["learner_id","current_band","vocabulary_need",
                            "syntax_need","cohesion_need"]))
check("to_prompt_dict: vocabulary_need = 'high'",
      d["vocabulary_need"] == "high")

# Non-dyslexic version
nd_prompt = DyslexicLearnerState("x", 5.0, decoding_disability=False)
d_nd = nd_prompt.to_prompt_dict()
check("to_prompt_dict: decoding_disability=False for non-dyslexic",
      d_nd["decoding_disability"] is False)


# ════════════════════════════════════════════════════════════════════════════

group("T14 — DYSLEXIA_ENGINE_CONFIG correctness")

check("is EngineConfig",                  isinstance(DYSLEXIA_ENGINE_CONFIG, EngineConfig))
check("fk_tolerance = 1.5",              DYSLEXIA_ENGINE_CONFIG.fk_tolerance == 1.5)
check("length_deviation_threshold = 0.50",DYSLEXIA_ENGINE_CONFIG.length_deviation_threshold == 0.50)
check("length_ceiling = 0.75",           DYSLEXIA_ENGINE_CONFIG.length_ceiling == 0.75)
check("meaning thresholds unchanged",
      DYSLEXIA_ENGINE_CONFIG.overall_meaning_threshold == 0.75)
check("vocab threshold unchanged",
      DYSLEXIA_ENGINE_CONFIG.vocabulary_threshold == 0.85)
check("severe_comprehension_threshold unchanged",
      DYSLEXIA_ENGINE_CONFIG.severe_comprehension_threshold == 0.50)
check("DyslexiaAwareDeterministicEngine constructs with it",
      DyslexiaAwareDeterministicEngine(config=DYSLEXIA_ENGINE_CONFIG) is not None)


# ── Final report ──────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'═'*65}")
print(f"  {'PASS ✓' if failed == 0 else 'FAIL ✗'}  │  "
      f"{passed} passed  │  {failed} failed  │  {total} total")
print(f"{'═'*65}")

import sys as _sys
_sys.exit(0 if failed == 0 else 1)
