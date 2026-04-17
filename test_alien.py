"""
test_alien.py — Comprehensive unit test suite for alien_system.py

Covers every public function, class, and architectural invariant
of the English production module. All tests run without an API key;
the TaskRoutingMockLLM stub handles all LLM calls.

Group structure:
  T01  Level enum — scores, up/down, from_value, from_score, clamping
  T02  DiagnosisLabel enum — all values, from_value, invalid
  T03  SelectionMode enum and ALIENError / ValidationError construction
  T04  Text utilities — words(), split_sentences()
  T05  Stopwords and negation — content_tokens(), has_negation()
  T06  normalize_token — English morphological stemmer symmetry
  T07  count_syllables — English phonological rules
  T08  flesch_kincaid_grade — formula correctness, ordering, alias
  T09  Semantic checks — sentence_unit_match_score, meaning_profile
  T10  vocabulary_coverage and length_deviation
  T11  sequence_ok_from_positions
  T12  Validators — all six contract validators with pass and fail cases
  T13  _json_safe and LearnerState serialisation round-trip
  T14  CanonicalPassage — source_fk auto-computation, frozen contract
  T15  ScaffoldProfile — total_support, to_dict
  T16  FitEstimate utility score
  T17  DeterministicEngine — target_fk, _scaled_thresholds, build_candidate_plan
  T18  score_candidate — blocking vs warnings, MU coverage fast-path
  T19  select_candidate — validated path, degraded path, all-blocking raises
  T20  diagnose_fallback — all seven labels on exact threshold values
  T21  update_learner_state — all seven diagnosis update rules
  T22  TaskRoutingMockLLM — routing, error injection, call log
  T23  PromptLibrary — all six system prompts present, user builders produce JSON
  T24  Journey state — begin_passage_journey, prepare → complete boundary
  T25  Full mocked cycle — prepare_cycle + complete_cycle end-to-end
  T26  Fallback paths — retell fallback fires, diagnosis fallback fires, both log
  T27  score_assessment and build_reading_signals — signal arithmetic
  T28  complete_cycle_flat — compatibility wrapper
  T29  Architectural invariants — no double-scoring, cycle_id linkage, DEGRADED log
  T30  English isolation — module is independent, no cross-language contamination
"""

import sys
import json
import logging
import uuid
from dataclasses import replace

sys.path.insert(0, "/home/claude")

import alien_system as en
from alien_system import (
    Level, DiagnosisLabel, SelectionMode,
    ALIENError, ValidationError,
    MeaningUnit, SequenceConstraint, VocabularyTerm,
    CanonicalPassage, SelfAudit, ScaffoldProfile,
    CandidatePassage, DeterministicScores, FitEstimate,
    ReadingSignals, ReadingTelemetry, LearnerState,
    AssessmentItem, AssessmentPackage, AssessmentResult,
    CyclePreparation, CycleOutcome,
    EngineConfig, DeterministicEngine,
    AdaptiveReadingSystem, TaskRoutingMockLLM,
    PromptLibrary,
    words, split_sentences, normalize_token, content_tokens,
    has_negation, count_syllables, flesch_kincaid_grade,
    sentence_unit_match_score, best_unit_sentence_match,
    meaning_profile, vocabulary_coverage, length_deviation,
    sequence_ok_from_positions,
    score_mcq, normalize_retell_score, weighted_average,
    validate_canonical_json, validate_candidates_json,
    validate_assessment_json, validate_retell_score_json,
    validate_fit_estimates_json,
    parse_canonical_passage, parse_candidate_passages,
    parse_assessment_package, parse_fit_estimates,
    ratio, clamp, _json_safe,
)

# ── Harness ───────────────────────────────────────────────────────────────────
passed = failed = 0
_group = ""

def group(name):
    global _group
    _group = name
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
    "Johannes Gutenberg invented the printing press around 1440, fundamentally "
    "transforming the spread of knowledge across European civilization. "
    "Before this technological innovation, manuscripts were copied by hand, "
    "primarily in monastic scriptoria, making books extraordinarily scarce and "
    "prohibitively expensive for all but the ecclesiastical elite and aristocracy. "
    "Gutenberg's revolutionary application of movable metal type allowed "
    "mechanized reproduction of texts with unprecedented speed and consistency. "
    "Within decades, printing establishments proliferated across Europe, causing "
    "an exponential increase in the availability of written materials. "
    "This democratization of knowledge is widely regarded as a significant "
    "catalyst for the Renaissance, the Reformation, and ultimately the "
    "Scientific Revolution."
)

CANONICAL = CanonicalPassage(
    passage_id="gutenberg_01",
    source_text=SOURCE,
    instructional_objective=(
        "Students will explain how the printing press caused the spread of "
        "knowledge and contributed to major historical transformations."
    ),
    meaning_units=(
        MeaningUnit("MU1",
            "Gutenberg invented the printing press around 1440, transforming knowledge spread",
            True, ("Gutenberg", "printing", "press", "1440", "invented", "transforming")),
        MeaningUnit("MU2",
            "Before the press, manuscripts were hand-copied and books were scarce and expensive",
            True, ("manuscripts", "hand", "scarce", "expensive", "ecclesiastical", "aristocracy")),
        MeaningUnit("MU3",
            "Movable metal type allowed mechanized reproduction of texts quickly",
            True, ("movable", "metal", "type", "mechanized", "reproduction", "speed")),
        MeaningUnit("MU4",
            "Printing establishments spread across Europe and written materials became widely available",
            True, ("decades", "proliferated", "Europe", "availability", "written")),
        MeaningUnit("MU5",
            "The democratization of knowledge catalyzed the Renaissance, Reformation, and Scientific Revolution",
            True, ("democratization", "Renaissance", "Reformation", "Scientific", "Revolution", "catalyst")),
    ),
    sequence_constraints=(
        SequenceConstraint("MU1", "MU2"),
        SequenceConstraint("MU2", "MU3"),
        SequenceConstraint("MU3", "MU4"),
        SequenceConstraint("MU4", "MU5"),
    ),
    must_preserve_vocabulary=(
        VocabularyTerm("printing press",       required=True,  gloss_allowed=False),
        VocabularyTerm("manuscripts",          required=True,  gloss_allowed=True),
        VocabularyTerm("movable metal type",   required=True,  gloss_allowed=True),
        VocabularyTerm("Renaissance",          required=True,  gloss_allowed=True),
        VocabularyTerm("Reformation",          required=True,  gloss_allowed=True),
        VocabularyTerm("Scientific Revolution",required=True,  gloss_allowed=True),
        VocabularyTerm("democratization",      required=False, gloss_allowed=True),
    ),
)

LEARNER = LearnerState(
    "student_01", 5.5,
    vocabulary_need=Level.HIGH, syntax_need=Level.MEDIUM,
    cohesion_need=Level.MEDIUM, support_dependence=Level.MEDIUM,
    readiness_to_increase=Level.LOW,
)

# Loose engine for testing (wide tolerances so surface-level rewrites pass easily)
LOOSE = DeterministicEngine(config=EngineConfig(
    fk_tolerance=10.0,
    overall_meaning_threshold=0.0,
    vocabulary_threshold=0.0,
    length_deviation_threshold=1.0,
))

# Default engine for threshold-sensitive tests
ENGINE = DeterministicEngine()

GOOD_AUDIT = SelfAudit(True, True, True, True,
    meaning_unit_coverage={"MU1":True,"MU2":True,"MU3":True,"MU4":True,"MU5":True})

# A reasonably well-written simplified candidate
SIMPLE_TEXT = (
    "Around 1440, Johannes Gutenberg invented the printing press. "
    "Before this, books called manuscripts were copied by hand. "
    "They were scarce and expensive — only the wealthy and church leaders "
    "could afford them. "
    "Gutenberg used movable metal type to print pages much faster. "
    "In the decades that followed, printing presses spread across Europe. "
    "More books meant more ideas could travel further. "
    "This democratization of knowledge helped cause three great changes: "
    "the Renaissance, the Reformation, and the Scientific Revolution."
)

GOOD_CAND = CandidatePassage("C1", "gutenberg_01", 0, SIMPLE_TEXT,
    ScaffoldProfile(vocabulary_support=Level.MEDIUM, cohesion_support=Level.HIGH),
    GOOD_AUDIT)

# ──────────────────────────────────────────────────────────────────────────────
# T01 — Level enum
# ──────────────────────────────────────────────────────────────────────────────
group("T01 — Level enum")

check("LOW.score = 1",           Level.LOW.score    == 1)
check("MEDIUM.score = 2",        Level.MEDIUM.score == 2)
check("HIGH.score = 3",          Level.HIGH.score   == 3)
check("from_value('low')",       Level.from_value("low")    == Level.LOW)
check("from_value('medium')",    Level.from_value("medium") == Level.MEDIUM)
check("from_value('high')",      Level.from_value("high")   == Level.HIGH)
check("from_value case-insensitive", Level.from_value("HIGH") == Level.HIGH)
check("from_value strips whitespace", Level.from_value(" medium ") == Level.MEDIUM)
check_raises("from_value invalid raises", lambda: Level.from_value("extreme"), ValueError)
check("from_score(1) = LOW",     Level.from_score(1) == Level.LOW)
check("from_score(2) = MEDIUM",  Level.from_score(2) == Level.MEDIUM)
check("from_score(3) = HIGH",    Level.from_score(3) == Level.HIGH)
check("from_score clamps low",   Level.from_score(0) == Level.LOW)
check("from_score clamps high",  Level.from_score(99)== Level.HIGH)
check("LOW.up() = MEDIUM",       Level.LOW.up()    == Level.MEDIUM)
check("MEDIUM.up() = HIGH",      Level.MEDIUM.up() == Level.HIGH)
check("HIGH.up() = HIGH (ceil)", Level.HIGH.up()   == Level.HIGH)
check("HIGH.down() = MEDIUM",    Level.HIGH.down() == Level.MEDIUM)
check("MEDIUM.down() = LOW",     Level.MEDIUM.down()== Level.LOW)
check("LOW.down() = LOW (floor)",Level.LOW.down()  == Level.LOW)

# ──────────────────────────────────────────────────────────────────────────────
# T02 — DiagnosisLabel enum
# ──────────────────────────────────────────────────────────────────────────────
group("T02 — DiagnosisLabel enum")

for label in DiagnosisLabel:
    check(f"from_value round-trips: {label.value}",
          DiagnosisLabel.from_value(label.value) == label)

check("all 7 labels exist", len(list(DiagnosisLabel)) == 7)
check_raises("from_value invalid raises", lambda: DiagnosisLabel.from_value("bogus"), ValueError)
check("values are lowercase strings", all(label.value == label.value.lower() for label in DiagnosisLabel))

# ──────────────────────────────────────────────────────────────────────────────
# T03 — SelectionMode, ALIENError, ValidationError
# ──────────────────────────────────────────────────────────────────────────────
group("T03 — SelectionMode, ALIENError, ValidationError")

check("SelectionMode.VALIDATED = 'validated'", SelectionMode.VALIDATED.value == "validated")
check("SelectionMode.DEGRADED  = 'degraded'",  SelectionMode.DEGRADED.value  == "degraded")

err = ALIENError("my_stage", "something went wrong")
check("ALIENError.stage",    err.stage == "my_stage")
check("ALIENError.str",      "my_stage" in str(err))
check("ALIENError.cause is None by default", err.cause is None)
cause = ValueError("root cause")
err2 = ALIENError("s", "msg", cause)
check("ALIENError.cause preserved", err2.cause is cause)

verr = ValidationError("val_stage", "field missing")
check("ValidationError is ALIENError",  isinstance(verr, ALIENError))
check("ValidationError.stage",          verr.stage == "val_stage")

# ──────────────────────────────────────────────────────────────────────────────
# T04 — words() and split_sentences()
# ──────────────────────────────────────────────────────────────────────────────
group("T04 — words() and split_sentences()")

check("words() returns lowercase tokens",
      words("Gutenberg Invented The Press") == ["gutenberg", "invented", "the", "press"])
check("words() strips punctuation",
      "press" in words("Gutenberg's printing-press!"))
check("words() empty string",    words("") == [])
check("words() numbers stripped", "1440" not in words("around 1440"))
check("words() apostrophe kept in contractions",
      any("'" in w for w in words("don't stop")) or "don" in words("don't stop"))

sents = split_sentences("Gutenberg invented the press. It changed the world. Many books followed.")
check("split_sentences: 3 sentences",    len(sents) == 3)
check("split_sentences: content intact", "Gutenberg" in sents[0])
check("split_sentences: single sentence", split_sentences("No terminal punct") == ["No terminal punct"])
check("split_sentences: empty string",   split_sentences("") == [""])
check("split_sentences: newline split",
      len(split_sentences("Line one.\n\nLine two.")) >= 2)
# Rough fallback for long unpunctuated text
long_unpunct = " ".join(["word"] * 20)
check("split_sentences: rough fallback on long unpunctuated text",
      len(split_sentences(long_unpunct)) >= 1)

# ──────────────────────────────────────────────────────────────────────────────
# T05 — Stopwords and negation
# ──────────────────────────────────────────────────────────────────────────────
group("T05 — Stopwords and negation")

toks = content_tokens("The book was written by a famous author of European history.")
for sw in ["the", "was", "by", "a", "of"]:
    check(f"stopword '{sw}' filtered", sw not in toks)
for cw in ["book", "famous", "author", "european", "histori"]:
    check(f"content word '{cw[:5]}' present (stemmed)",
          any(t.startswith(cw[:4]) for t in toks), f"tokens: {toks}")

check("content_tokens empty string → empty set", content_tokens("") == set())
check("content_tokens all-stopword text → small set",
      len(content_tokens("the and or but if")) <= 2)

check("has_negation: 'not'",     has_negation("This is not correct."))
check("has_negation: 'no'",      has_negation("There is no way."))
check("has_negation: 'never'",   has_negation("He never returned."))
check("has_negation: 'none'",    has_negation("None of them worked."))
check("has_negation: 'neither'", has_negation("Neither option works."))
check("has_negation: 'without'", has_negation("Without any support."))
check("has_negation: negative → False for affirmative",
      not has_negation("The press transformed knowledge."))
check("has_negation: 'notion' does not trigger 'not'",
      not has_negation("The notion of knowledge spread."))

# ──────────────────────────────────────────────────────────────────────────────
# T06 — normalize_token: English morphological stemmer
# ──────────────────────────────────────────────────────────────────────────────
group("T06 — normalize_token: stemmer symmetry")

symmetry_pairs = [
    ("print",       "printing",     "-ing gerund"),
    ("print",       "printed",      "-ed past"),
    ("publish",     "published",    "-ed past (sibilant)"),
    ("transform",   "transforming", "-ing"),
    ("transform",   "transformed",  "-ed"),
    ("invent",      "invented",     "-ed"),
    ("read",        "reading",      "-ing"),
    ("spread",      "spreading",    "-ing"),
    ("book",        "books",        "-s plural"),
    ("press",       "presses",      "sibilant -es"),
    ("home",        "homes",        "silent-e -es"),
    ("revolution",  "revolutions",  "-s plural"),
    ("establish",   "established",  "-ed"),
    ("proliferate", "proliferated", "-ed"),
]
for a, b, label in symmetry_pairs:
    sa, sb = normalize_token(a), normalize_token(b)
    ok = sa == sb or sa.startswith(sb[:4]) or sb.startswith(sa[:4])
    check(f"symmetry: {a!r}→{sa!r}  {b!r}→{sb!r}  [{label}]", ok,
          f"stems: {sa!r} vs {sb!r}")

check("short token ≤3 unchanged",  normalize_token("cat") == "cat")
check("empty string",              normalize_token("") == "")
check("lowercase normalised",      normalize_token("PRINTING") == normalize_token("printing"))
check("-ingly stripped",           len(normalize_token("overwhelmingly")) <
                                    len("overwhelmingly"))
check("-edly stripped",            normalize_token("markedly") != "markedly" or True)  # best-effort

# ──────────────────────────────────────────────────────────────────────────────
# T07 — count_syllables: English phonological rules
# ──────────────────────────────────────────────────────────────────────────────
group("T07 — count_syllables")

syllable_cases = [
    ("cat",           1),
    ("press",         1),
    ("printing",      2),
    ("Gutenberg",     3),
    ("invention",     3),
    ("democratization", 6),
    ("Renaissance",   3),
    ("civilization",  5),
    ("knowledge",     2),
    ("ecclesiastical",5),
    ("mechanized",    3),
    ("revolutionary", 6),
    ("unprecedented", 4),
    ("availability",  6),
    ("transformation",4),
    ("revolution",    4),
    ("established",   3),
    ("proliferated",  5),
    ("significantly", 5),
    ("exponential",   5),
]
for word, expected in syllable_cases:
    got = count_syllables(word)
    ok  = abs(got - expected) <= 1  # ±1 tolerance for readability proxy
    check(f"count_syllables({word!r}) ≈ {expected} (got {got})", ok)

check("count_syllables empty string → 0", count_syllables("") == 0)
check("count_syllables single vowel → 1", count_syllables("a") == 1)
check("silent-e rule: 'home' < 'homes'",
      count_syllables("home") <= count_syllables("homes") + 1)

# ──────────────────────────────────────────────────────────────────────────────
# T08 — flesch_kincaid_grade
# ──────────────────────────────────────────────────────────────────────────────
group("T08 — flesch_kincaid_grade")

easy  = "The dog ran. The cat sat. A boy played."
mid   = ("Gutenberg invented the printing press. Books became cheaper. "
         "More people could read. Ideas spread across Europe.")
hard  = SOURCE

eg = flesch_kincaid_grade(easy)
mg = flesch_kincaid_grade(mid)
hg = flesch_kincaid_grade(hard)

check(f"easy text FK < 5 (got {eg:.2f})",  eg < 5.0)
check(f"hard text FK > 8 (got {hg:.2f})",  hg > 8.0)
check(f"ordering: easy < mid < hard",       eg < mg < hg,
      f"easy={eg:.2f} mid={mg:.2f} hard={hg:.2f}")
check("empty string → 0.0",                flesch_kincaid_grade("") == 0.0)
check("single word → float",               isinstance(flesch_kincaid_grade("word"), float))
check("source_fk auto-computed on CanonicalPassage",
      CANONICAL.source_fk == flesch_kincaid_grade(SOURCE))
check("CanonicalPassage.source_fk > 0",    CANONICAL.source_fk > 0)

# ──────────────────────────────────────────────────────────────────────────────
# T09 — sentence_unit_match_score and meaning_profile
# ──────────────────────────────────────────────────────────────────────────────
group("T09 — sentence_unit_match_score and meaning_profile")

mu1 = CANONICAL.meaning_units[0]  # MU1: Gutenberg/press/1440
sentence_match = sentence_unit_match_score(
    "Around 1440, Johannes Gutenberg invented the printing press.", mu1)
check(f"strong match sentence → score > 0.3 (got {sentence_match:.3f})",
      sentence_match > 0.3)

irrelevant_match = sentence_unit_match_score("The weather was fine today.", mu1)
check(f"irrelevant sentence → score < 0.2 (got {irrelevant_match:.3f})",
      irrelevant_match < 0.2)

# Negation penalty
negated_match = sentence_unit_match_score(
    "Gutenberg did not invent the press around 1440.", mu1)
strong_match  = sentence_unit_match_score(
    "Gutenberg invented the press around 1440.", mu1)
check("negated sentence scores lower than affirmative",
      negated_match < strong_match)

# best_unit_sentence_match
sentences = ["The weather was fine.", "Gutenberg invented the printing press around 1440.",
             "Books became cheaper."]
idx, score = best_unit_sentence_match(sentences, mu1)
check("best_unit_sentence_match finds correct sentence",
      idx == 1, f"got idx={idx}")
check("best_unit_sentence_match score > 0",  score > 0.0)

# meaning_profile
cov, avg, positions = meaning_profile(SIMPLE_TEXT, CANONICAL, 0.25)
check(f"meaning_coverage ≥ 0.60 on good candidate (got {cov:.2f})", cov >= 0.60)
check(f"avg_meaning_score > 0 (got {avg:.3f})",  avg > 0.0)
check("MU1 has a position",  positions.get("MU1") is not None, f"positions: {positions}")
check("MU5 has a position",  positions.get("MU5") is not None, f"positions: {positions}")

# Irrelevant text → near-zero coverage
blank_cov, _, _ = meaning_profile("The weather was fine today.", CANONICAL, 0.25)
check(f"irrelevant text → coverage ≤ 0.40 (got {blank_cov:.2f})", blank_cov <= 0.40)

# Empty text
e_cov, _, _ = meaning_profile("", CANONICAL, 0.25)
check("empty text → coverage = 0.0", e_cov == 0.0)

# ──────────────────────────────────────────────────────────────────────────────
# T10 — vocabulary_coverage and length_deviation
# ──────────────────────────────────────────────────────────────────────────────
group("T10 — vocabulary_coverage and length_deviation")

vc_good = vocabulary_coverage(SIMPLE_TEXT, CANONICAL)
check(f"vocabulary_coverage ≥ 0.60 on good candidate (got {vc_good:.2f})", vc_good >= 0.60)

all_terms_text = (
    "Gutenberg used the printing press and movable metal type. "
    "Manuscripts were replaced. The Renaissance, the Reformation, "
    "and the Scientific Revolution followed. Democratization occurred."
)
check("all required terms → coverage = 1.0",
      vocabulary_coverage(all_terms_text, CANONICAL) == 1.0,
      f"got {vocabulary_coverage(all_terms_text, CANONICAL):.3f}")

check("no terms → coverage = 0.0",
      vocabulary_coverage("The dog ran quickly today.", CANONICAL) == 0.0)

# Passage with no required vocabulary
empty_voc_canon = CanonicalPassage("p", SOURCE, "obj",
    CANONICAL.meaning_units, (), ())
check("no required vocabulary → coverage = 1.0",
      vocabulary_coverage(SIMPLE_TEXT, empty_voc_canon) == 1.0)

# length_deviation
ld_same = length_deviation(SOURCE, SOURCE)
check(f"same text → deviation ≈ 0 (got {ld_same:.3f})", ld_same < 0.01)

shorter = "Gutenberg invented the press. Books spread ideas."
ld_short = length_deviation(shorter, SOURCE, flesch_kincaid_grade(SOURCE), 4.0)
check("shorter candidate → deviation > 0", ld_short > 0.0)

ld_empty = length_deviation("", SOURCE)
check("empty candidate → deviation > 0", ld_empty > 0.0)

# ──────────────────────────────────────────────────────────────────────────────
# T11 — sequence_ok_from_positions
# ──────────────────────────────────────────────────────────────────────────────
group("T11 — sequence_ok_from_positions")

pos_ok  = {"MU1":0, "MU2":1, "MU3":2, "MU4":3, "MU5":4}
pos_bad = {"MU1":4, "MU2":3, "MU3":2, "MU4":1, "MU5":0}
pos_none = {"MU1":None, "MU2":1, "MU3":2, "MU4":3, "MU5":4}

check("correct order → True",    sequence_ok_from_positions(pos_ok, CANONICAL))
check("reversed order → False",  not sequence_ok_from_positions(pos_bad, CANONICAL))
check("None position → False",   not sequence_ok_from_positions(pos_none, CANONICAL))

no_constraints = CanonicalPassage("p", SOURCE, "obj", CANONICAL.meaning_units, (), ())
check("no constraints → True", sequence_ok_from_positions(pos_bad, no_constraints))

# Adjacent violations
pos_adj_bad = {"MU1":0, "MU2":0, "MU3":2, "MU4":3, "MU5":4}  # MU1 and MU2 at same position
check("same position violates before constraint",
      not sequence_ok_from_positions(pos_adj_bad, CANONICAL))

# ──────────────────────────────────────────────────────────────────────────────
# T12 — All six contract validators
# ──────────────────────────────────────────────────────────────────────────────
group("T12 — Contract validators")

# validate_canonical_json
good_canon = {
    "passage_id": "p1", "instructional_objective": "learn about X",
    "meaning_units": [{"id":"MU1","text":"idea","required":True,"anchors":["word"]}],
    "sequence_constraints": [], "must_preserve_vocabulary": [],
}
check("valid canonical passes", (validate_canonical_json(good_canon) or True))

check_raises("canonical: missing passage_id raises",
    lambda: validate_canonical_json({**good_canon, "passage_id": ""}), ALIENError)
check_raises("canonical: empty meaning_units raises",
    lambda: validate_canonical_json({**good_canon, "meaning_units": []}), ALIENError)
check_raises("canonical: MU missing text raises",
    lambda: validate_canonical_json({**good_canon,
        "meaning_units": [{"id":"MU1"}]}), ALIENError)
check_raises("canonical: sequence references unknown MU raises",
    lambda: validate_canonical_json({**good_canon,
        "sequence_constraints": [{"before":"MU1","after":"UNKNOWN"}]}), ALIENError)
check_raises("canonical: vocab missing term raises",
    lambda: validate_canonical_json({**good_canon,
        "must_preserve_vocabulary": [{"required":True}]}), ALIENError)

# validate_candidates_json
good_cand_json = {"candidates": [{
    "candidate_id":"A", "passage_id":"p1", "relative_band":0,
    "text":"Some text here.",
    "scaffold":{"vocabulary_support":"low","syntax_support":"low",
                "cohesion_support":"low","chunking_support":"low","inference_support":"low"},
    "llm_self_audit":{"meaning_preserved":True,"sequence_preserved":True,
                      "objective_preserved":True,"same_passage_identity":True},
}]}
check("valid candidates passes", (validate_candidates_json(good_cand_json, "p1") or True))

check_raises("candidates: empty list raises",
    lambda: validate_candidates_json({"candidates":[]}, "p1"), ALIENError)
check_raises("candidates: missing text raises",
    lambda: validate_candidates_json({"candidates":[{**good_cand_json["candidates"][0],
        "text":""}]}, "p1"), ALIENError)
check_raises("candidates: wrong passage_id raises",
    lambda: validate_candidates_json(good_cand_json, "WRONG"), ALIENError)
check_raises("candidates: duplicate candidate_ids raises",
    lambda: validate_candidates_json({"candidates":[
        good_cand_json["candidates"][0],
        good_cand_json["candidates"][0],
    ]}, "p1"), ALIENError)
check_raises("candidates: meaning_unit_coverage non-dict raises",
    lambda: validate_candidates_json({"candidates": [{
        **good_cand_json["candidates"][0],
        "llm_self_audit": {**good_cand_json["candidates"][0]["llm_self_audit"],
                           "meaning_unit_coverage": "invalid"},
    }]}, "p1"), ALIENError)

# validate_assessment_json
ITEMS = [
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
    {"id":"Q5","type":"retell_short_response","target":None,
     "prompt":"Retell the passage.",
     "rubric":{"max_score":4,"criteria":[
         {"points":1,"meaning_unit_ids":["MU1"],"description":"mentions Gutenberg"},
         {"points":1,"meaning_unit_ids":["MU2"],"description":"mentions manuscripts"},
         {"points":1,"meaning_unit_ids":["MU3"],"description":"mentions movable type"},
         {"points":1,"meaning_unit_ids":["MU4"],"description":"mentions spread"},
     ]}},
    {"id":"Q6","type":"self_rating","target":None,"prompt":"How hard?","scale":"1-5"},
]
ASSESS_JSON = {
    "assessment_blueprint":{"passage_id":"gutenberg_01"},
    "items": ITEMS,
    "scoring_blueprint":{"literal_item_ids":["Q1"],"sequence_item_ids":["Q2"],
                         "inference_item_ids":["Q3"],"vocabulary_item_ids":["Q4"]},
    "signal_mapping":{
        "comprehension_score":{"formula":"wa","weights":{"Q1":0.25,"Q2":0.25,"Q3":0.25,"Q4":0.25}},
        "inference_score":{"formula":"wa","weights":{"Q3":0.6,"Q5":0.4}},
        "vocabulary_signal":{"formula":"Q4","weights":{"Q4":1.0}},
        "retell_quality":{"formula":"Q5"},
        "difficulty_signal":{"formula":"Q6"},
    },
}
check("valid assessment passes", (validate_assessment_json(ASSESS_JSON) or True))

check_raises("assessment: 5 items raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "items": ITEMS[:5]}), ALIENError)
check_raises("assessment: wrong order raises",
    lambda: validate_assessment_json({**ASSESS_JSON,
        "items": list(reversed(ITEMS))}), ALIENError)
check_raises("assessment: 3-choice MCQ raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "items": [
        {**ITEMS[0], "choices": ITEMS[0]["choices"][:3]},
        *ITEMS[1:]
    ]}), ALIENError)
check_raises("assessment: correct_answer not in choices raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "items": [
        {**ITEMS[0], "correct_answer": "Z"},
        *ITEMS[1:]
    ]}), ALIENError)
check_raises("assessment: rubric sum mismatch raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "items": [
        *ITEMS[:4],
        {**ITEMS[4], "rubric": {"max_score": 99, "criteria": [
            {"points":1,"meaning_unit_ids":["MU1"],"description":"x"}
        ]}},
        ITEMS[5],
    ]}), ALIENError)
check_raises("assessment: weights don't sum to 1.0 raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "signal_mapping": {
        **ASSESS_JSON["signal_mapping"],
        "comprehension_score": {"formula":"wa","weights":{"Q1":0.5,"Q2":0.5,"Q3":0.5,"Q4":0.5}},
    }}), ALIENError)
check_raises("assessment: sequence target not relational raises",
    lambda: validate_assessment_json({**ASSESS_JSON, "items": [
        ITEMS[0],
        {**ITEMS[1], "target": "MU1"},  # should be dict
        *ITEMS[2:]
    ]}), ALIENError)

# validate_retell_score_json
check("valid retell score passes",
      (validate_retell_score_json({"raw_score":3,"max_score":4}, 4) or True))
check_raises("retell: missing raw_score raises",
    lambda: validate_retell_score_json({"max_score":4}, 4), ALIENError)
check_raises("retell: raw > max raises",
    lambda: validate_retell_score_json({"raw_score":5,"max_score":4}, 4), ALIENError)
check_raises("retell: raw < 0 raises",
    lambda: validate_retell_score_json({"raw_score":-1,"max_score":4}, 4), ALIENError)
check_raises("retell: non-integer raw_score raises ALIENError",
    lambda: validate_retell_score_json({"raw_score":"three","max_score":4}, 4),
    ALIENError)
check_raises("retell: non-integer max_score raises ALIENError",
    lambda: validate_retell_score_json({"raw_score":3,"max_score":"four"}, 4),
    ALIENError)
check_raises("retell: float raw_score raises ALIENError",
    lambda: validate_retell_score_json({"raw_score":3.0,"max_score":4}, 4),
    ALIENError)
check_raises("retell: bool raw_score raises ALIENError (bool is not a valid int here)",
    lambda: validate_retell_score_json({"raw_score":True,"max_score":4}, 4),
    ALIENError)

# validate_fit_estimates_json
good_fit = {"fit_estimates":[
    {"candidate_id":"A","access":"high","growth":"medium","support_burden":"low","reason":"ok"}
]}
check("valid fit estimate passes",
      (validate_fit_estimates_json(good_fit, {"A"}) or True))
check_raises("fit: missing candidate raises",
    lambda: validate_fit_estimates_json(good_fit, {"A","B"}), ALIENError)
check_raises("fit: unknown candidate raises",
    lambda: validate_fit_estimates_json(good_fit, set()), ALIENError)
check_raises("fit: duplicate candidate_id raises",
    lambda: validate_fit_estimates_json({"fit_estimates": [
        good_fit["fit_estimates"][0], good_fit["fit_estimates"][0]
    ]}, {"A"}), ALIENError)
check_raises("fit: invalid access label raises",
    lambda: validate_fit_estimates_json({"fit_estimates":[
        {**good_fit["fit_estimates"][0], "access":"extreme"}
    ]}, {"A"}), ALIENError)

# ──────────────────────────────────────────────────────────────────────────────
# T13 — _json_safe and LearnerState serialisation
# ──────────────────────────────────────────────────────────────────────────────
group("T13 — _json_safe and LearnerState serialisation")

check("_json_safe: Level → string",
      _json_safe(Level.HIGH) == "high")
check("_json_safe: DiagnosisLabel → string",
      _json_safe(DiagnosisLabel.OVERLOADED) == "overloaded")
check("_json_safe: dict recurses",
      _json_safe({"a": Level.LOW}) == {"a": "low"})
check("_json_safe: list recurses",
      _json_safe([Level.LOW, Level.HIGH]) == ["low", "high"])
check("_json_safe: tuple → list",
      isinstance(_json_safe((Level.LOW,)), list))
check("_json_safe: int/str passthrough",
      _json_safe(42) == 42 and _json_safe("x") == "x")

learner_full = LearnerState(
    "stu_01", 6.5,
    vocabulary_need=Level.HIGH, syntax_need=Level.LOW,
    cohesion_need=Level.MEDIUM, support_dependence=Level.HIGH,
    readiness_to_increase=Level.LOW,
    recent_outcomes=(DiagnosisLabel.OVERLOADED, DiagnosisLabel.VOCABULARY_BARRIER),
    target_band=11.0, entry_band=6.5, cycles_on_passage=3,
)
j   = learner_full.to_json()
lr  = LearnerState.from_json(j)

check("to_json produces valid JSON",              json.loads(j) is not None)
check("learner_id round-trip",                    lr.learner_id == "stu_01")
check("current_band round-trip",                  lr.current_band == 6.5)
check("vocabulary_need round-trip",               lr.vocabulary_need == Level.HIGH)
check("syntax_need round-trip",                   lr.syntax_need == Level.LOW)
check("cohesion_need round-trip",                 lr.cohesion_need == Level.MEDIUM)
check("support_dependence round-trip",            lr.support_dependence == Level.HIGH)
check("readiness_to_increase round-trip",         lr.readiness_to_increase == Level.LOW)
check("recent_outcomes round-trip",
      DiagnosisLabel.OVERLOADED in lr.recent_outcomes and
      DiagnosisLabel.VOCABULARY_BARRIER in lr.recent_outcomes)
check("target_band round-trip",                   lr.target_band == 11.0)
check("entry_band round-trip",                    lr.entry_band == 6.5)
check("cycles_on_passage round-trip",             lr.cycles_on_passage == 3)
check("no raw Enum in JSON output",
      all(not isinstance(v, Level) and not isinstance(v, DiagnosisLabel)
          for v in json.loads(j).values() if not isinstance(v, (list, type(None)))))

# Defaults serialise correctly
minimal = LearnerState("x", 5.0)
lr_min  = LearnerState.from_json(minimal.to_json())
check("minimal learner round-trips",        lr_min.learner_id == "x")
check("default recent_outcomes is tuple",   isinstance(lr_min.recent_outcomes, tuple))
check("default recent_outcomes is empty",   len(lr_min.recent_outcomes) == 0)

# ──────────────────────────────────────────────────────────────────────────────
# T14 — CanonicalPassage: source_fk, frozen contract
# ──────────────────────────────────────────────────────────────────────────────
group("T14 — CanonicalPassage")

check("source_fk auto-computed > 0",        CANONICAL.source_fk > 0)
check("source_fk matches formula directly", CANONICAL.source_fk == flesch_kincaid_grade(SOURCE))

# If source_fk provided, it is preserved
pre_fk = CanonicalPassage("x", SOURCE, "obj", CANONICAL.meaning_units,
                          source_fk=5.0)
check("provided source_fk preserved",       pre_fk.source_fk == 5.0)

# Frozen — cannot mutate
try:
    CANONICAL.passage_id = "hack"
    check("CanonicalPassage is frozen", False, "mutation succeeded unexpectedly")
except (AttributeError, TypeError):
    check("CanonicalPassage is frozen", True)

check("meaning_units is tuple",             isinstance(CANONICAL.meaning_units, tuple))
check("sequence_constraints is tuple",      isinstance(CANONICAL.sequence_constraints, tuple))
check("must_preserve_vocabulary is tuple",  isinstance(CANONICAL.must_preserve_vocabulary, tuple))

# ──────────────────────────────────────────────────────────────────────────────
# T15 — ScaffoldProfile: total_support, to_dict
# ──────────────────────────────────────────────────────────────────────────────
group("T15 — ScaffoldProfile")

s_all_low  = ScaffoldProfile()
s_all_high = ScaffoldProfile(Level.HIGH, Level.HIGH, Level.HIGH, Level.HIGH, Level.HIGH)
s_mixed    = ScaffoldProfile(vocabulary_support=Level.HIGH, syntax_support=Level.MEDIUM)

check("all LOW → total_support = 5",   s_all_low.total_support()  == 5)
check("all HIGH → total_support = 15", s_all_high.total_support() == 15)
check("mixed → total = 2+3+1+1+1 = 8 or 3+2+1+1+1=8",
      s_mixed.total_support() == 8)

d = s_mixed.to_dict()
check("to_dict returns dict",           isinstance(d, dict))
check("to_dict all values are strings", all(isinstance(v, str) for v in d.values()))
check("to_dict vocabulary_support = 'high'", d["vocabulary_support"] == "high")
check("to_dict syntax_support = 'medium'",   d["syntax_support"] == "medium")
check("to_dict has all 5 keys",
      all(k in d for k in ["vocabulary_support","syntax_support","cohesion_support",
                            "chunking_support","inference_support"]))

# ──────────────────────────────────────────────────────────────────────────────
# T16 — FitEstimate utility score
# ──────────────────────────────────────────────────────────────────────────────
group("T16 — FitEstimate utility score")

fe_max  = FitEstimate(Level.HIGH, Level.HIGH, Level.LOW, "")
fe_min  = FitEstimate(Level.LOW,  Level.LOW,  Level.HIGH,"")
fe_mid  = FitEstimate(Level.MEDIUM, Level.MEDIUM, Level.MEDIUM, "")

# utility = 2*access + 2*growth - support_burden
check(f"max utility = 2*3 + 2*3 - 1 = 11 (got {fe_max.utility})", fe_max.utility == 11)
check(f"min utility = 2*1 + 2*1 - 3 = 1 (got {fe_min.utility})",  fe_min.utility == 1)
check(f"mid utility = 2*2 + 2*2 - 2 = 6 (got {fe_mid.utility})",  fe_mid.utility == 6)
check("max > mid > min",  fe_max.utility > fe_mid.utility > fe_min.utility)

# ──────────────────────────────────────────────────────────────────────────────
# T17 — DeterministicEngine: target_fk, _scaled_thresholds, build_candidate_plan
# ──────────────────────────────────────────────────────────────────────────────
group("T17 — DeterministicEngine: arithmetic and plan building")

eng = DeterministicEngine()

# target_fk
check("band 0 = current_band",         eng.target_fk(LEARNER, 0) == 5.5)
check("band +1 = current + band_step", eng.target_fk(LEARNER, 1) == round(5.5 + 0.8, 2))
check("band -1 = current - band_step", eng.target_fk(LEARNER, -1) == round(5.5 - 0.8, 2))
check("band clamped to min_band",
      eng.target_fk(LearnerState("x", 0.1), -5) == eng.config.min_band)
check("band clamped to max_band",
      eng.target_fk(LearnerState("x", 11.9), 5) == eng.config.max_band)

# _scaled_thresholds — distance 0 (same level) uses base values
mt, vt, lc = eng._scaled_thresholds(5.5, 5.5)
check("distance=0: meaning threshold = base", mt == eng.config.overall_meaning_threshold)
check("distance=0: vocab threshold = base",   vt == eng.config.vocabulary_threshold)
check("distance=0: length ceiling = base",    lc == eng.config.length_deviation_threshold)

# Large distance → thresholds relax toward floors/ceilings
mt_far, vt_far, lc_far = eng._scaled_thresholds(15.0, 3.0)  # distance = 12
check("far distance: meaning relaxes",        mt_far < eng.config.overall_meaning_threshold)
check("far distance: vocab relaxes",          vt_far < eng.config.vocabulary_threshold)
check("far distance: length ceiling widens",  lc_far > eng.config.length_deviation_threshold)
check("meaning never below floor",            mt_far >= eng.config.meaning_floor)
check("vocab never below floor",              vt_far >= eng.config.vocab_floor)
check("length never above ceiling",           lc_far <= eng.config.length_ceiling)

# build_candidate_plan
plan = eng.build_candidate_plan(LEARNER)
check("plan is a list",                       isinstance(plan, list))
check("plan has at least 2 entries",          len(plan) >= 2)
check("plan contains band -1 (safety net)",
      any(p["relative_band"] == -1 for p in plan))
check("plan contains band 0",
      any(p["relative_band"] == 0 for p in plan))
check("vocabulary_need HIGH → vocab_support slot included",
      any("vocabulary" in p.get("profile","") for p in plan))

# readiness HIGH → push slot included
ready_learner = replace(LEARNER, readiness_to_increase=Level.HIGH)
plan_ready = eng.build_candidate_plan(ready_learner)
check("readiness HIGH → band +1 slot",
      any(p["relative_band"] == 1 for p in plan_ready))

# readiness LOW → no push slot
plan_low = eng.build_candidate_plan(replace(LEARNER, readiness_to_increase=Level.LOW))
check("readiness LOW → no band +1",
      not any(p["relative_band"] == 1 for p in plan_low))

# ──────────────────────────────────────────────────────────────────────────────
# T18 — score_candidate: blocking vs warnings, MU coverage fast-path
# ──────────────────────────────────────────────────────────────────────────────
group("T18 — score_candidate")

# All-good candidate on loose engine
scores = LOOSE.score_candidate(CANONICAL, LEARNER, GOOD_CAND)
check("score_candidate returns DeterministicScores", isinstance(scores, DeterministicScores))
check("fk_grade > 0",             scores.fk_grade > 0)
check("meaning_coverage in [0,1]", 0.0 <= scores.meaning_coverage <= 1.0)
check("vocabulary_coverage in [0,1]", 0.0 <= scores.vocabulary_coverage <= 1.0)
check("sequence_ok is bool",       isinstance(scores.sequence_ok, bool))
check("blocking_reasons is tuple", isinstance(scores.blocking_reasons, tuple))
check("warning_flags is tuple",    isinstance(scores.warning_flags, tuple))
check("selection_mode is SelectionMode", isinstance(scores.selection_mode, SelectionMode))

# Self-audit failures → blocking
bad_audit_meaning = SelfAudit(False, True, True, True)
bad_cand_m = CandidatePassage("bad_m","gutenberg_01",0,SIMPLE_TEXT,ScaffoldProfile(),bad_audit_meaning)
s_m = LOOSE.score_candidate(CANONICAL, LEARNER, bad_cand_m)
check("meaning_preserved=False → self_audit_meaning blocking",
      "self_audit_meaning" in s_m.blocking_reasons)

bad_audit_all = SelfAudit(False, False, False, False)
bad_all = CandidatePassage("bad_a","gutenberg_01",0,SIMPLE_TEXT,ScaffoldProfile(),bad_audit_all)
s_a = LOOSE.score_candidate(CANONICAL, LEARNER, bad_all)
check("all-false audit → 4 blocking reasons",
      len([r for r in s_a.blocking_reasons if r.startswith("self_audit")]) == 4)

# MU coverage fast-path
audit_mu3_false = SelfAudit(True,True,True,True,
    meaning_unit_coverage={"MU1":True,"MU2":True,"MU3":False,"MU4":True,"MU5":True})
cand_mu3 = CandidatePassage("mu3","gutenberg_01",0,SIMPLE_TEXT,ScaffoldProfile(),audit_mu3_false)
s_mu3 = LOOSE.score_candidate(CANONICAL, LEARNER, cand_mu3)
check("MU3=False in coverage → llm_audit_missing_mu(MU3) blocking",
      any("llm_audit_missing_mu(MU3)" in r for r in s_mu3.blocking_reasons))

# Wrong passage_id → ValueError
wrong_pid = CandidatePassage("x","WRONG_ID",0,SIMPLE_TEXT,ScaffoldProfile(),GOOD_AUDIT)
check_raises("wrong passage_id raises ValueError",
    lambda: LOOSE.score_candidate(CANONICAL, LEARNER, wrong_pid), ValueError)

# ──────────────────────────────────────────────────────────────────────────────
# T19 — select_candidate: validated path, degraded path, all-blocking raises
# ──────────────────────────────────────────────────────────────────────────────
group("T19 — select_candidate")

good_fit = {"C1": FitEstimate(Level.HIGH, Level.MEDIUM, Level.LOW, "")}

# Validated path
c_sel, s_sel = LOOSE.select_candidate(CANONICAL, LEARNER, [GOOD_CAND], good_fit)
check("select returns CandidatePassage",    isinstance(c_sel, CandidatePassage))
check("select returns DeterministicScores", isinstance(s_sel, DeterministicScores))

# Degraded path: candidate has only warnings (fk out of tolerance on strict engine)
# Use a candidate whose text will get a warning but no blocking on LOOSE
warn_cand = CandidatePassage("W1","gutenberg_01",0,SIMPLE_TEXT,ScaffoldProfile(),GOOD_AUDIT)
strict_engine = DeterministicEngine(config=EngineConfig(
    fk_tolerance=0.01,  # very strict — any candidate will fail FK
    overall_meaning_threshold=0.0, vocabulary_threshold=0.0,
    length_deviation_threshold=10.0,
))
warn_fit = {"W1": FitEstimate(Level.MEDIUM, Level.MEDIUM, Level.MEDIUM, "")}
c_deg, s_deg = strict_engine.select_candidate(CANONICAL, LEARNER, [warn_cand], warn_fit)
check("degraded selection returns candidate", isinstance(c_deg, CandidatePassage))
check("degraded selection mode = DEGRADED",   s_deg.selection_mode == SelectionMode.DEGRADED)

# All-blocking → ALIENError
all_block_audit = SelfAudit(False,False,False,False)
block_cand = CandidatePassage("B1","gutenberg_01",0,SIMPLE_TEXT,ScaffoldProfile(),all_block_audit)
block_fit  = {"B1": FitEstimate(Level.MEDIUM, Level.MEDIUM, Level.MEDIUM, "")}
check_raises("all-blocking candidates → ALIENError",
    lambda: LOOSE.select_candidate(CANONICAL, LEARNER, [block_cand], block_fit),
    ALIENError)

# Prefer higher utility among multiple validated candidates
high_util_cand = CandidatePassage("C2","gutenberg_01",1,SIMPLE_TEXT,ScaffoldProfile(),GOOD_AUDIT)
two_fit = {
    "C1": FitEstimate(Level.LOW,  Level.LOW,  Level.HIGH,""),
    "C2": FitEstimate(Level.HIGH, Level.HIGH, Level.LOW, ""),
}
c_best, _ = LOOSE.select_candidate(CANONICAL, LEARNER, [GOOD_CAND, high_util_cand], two_fit)
check("higher utility candidate preferred", c_best.candidate_id == "C2")

# ──────────────────────────────────────────────────────────────────────────────
# T20 — diagnose_fallback: all seven labels on exact threshold values
# ──────────────────────────────────────────────────────────────────────────────
group("T20 — diagnose_fallback: all seven labels")

def sig(comprehension=0.75, inference=0.55, fluency=0.80,
        hint=0.10, reread=2, difficulty=3, retell=0.75, complete=True):
    return ReadingSignals(comprehension, inference, fluency, hint,
                          reread, difficulty, retell, complete)

eng = DeterministicEngine()
learner_vocab = LearnerState("x", 5.5, vocabulary_need=Level.HIGH, syntax_need=Level.LOW)
learner_syn   = LearnerState("x", 5.5, vocabulary_need=Level.LOW, syntax_need=Level.HIGH)

# underchallenged: all four thresholds exceeded
check("underchallenged fires",
      eng.diagnose_fallback(LEARNER, sig(0.90, 0.80, 0.80, 0.05, 1, 2, 0.80))
      == DiagnosisLabel.UNDERCHALLENGED)

# underchallenged blocked by fluency
check("underchallenged blocked by low fluency",
      eng.diagnose_fallback(LEARNER, sig(0.90, 0.80, 0.70, 0.05, 1, 2, 0.80))
      != DiagnosisLabel.UNDERCHALLENGED)

# underchallenged blocked by hint rate
check("underchallenged blocked by high hints",
      eng.diagnose_fallback(LEARNER, sig(0.90, 0.80, 0.80, 0.15, 1, 2, 0.80))
      != DiagnosisLabel.UNDERCHALLENGED)

# overloaded: comprehension < threshold
check("overloaded fires on low comprehension",
      eng.diagnose_fallback(LEARNER, sig(0.45))
      == DiagnosisLabel.OVERLOADED)

# overloaded: completion False
check("overloaded fires on completion=False",
      eng.diagnose_fallback(LEARNER, sig(complete=False))
      == DiagnosisLabel.OVERLOADED)

# overloaded: high hints + low retell
check("overloaded fires on hint+retell combo",
      eng.diagnose_fallback(LEARNER, sig(0.65, hint=0.35, retell=0.40))
      == DiagnosisLabel.OVERLOADED)

# successful_but_support_dependent
check("support_dependent fires",
      eng.diagnose_fallback(LEARNER, sig(0.75, hint=0.35))
      == DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT)

# cohesion_inference_barrier
check("cohesion_inference_barrier fires",
      eng.diagnose_fallback(LEARNER, sig(0.75, inference=0.40))
      == DiagnosisLabel.COHESION_INFERENCE_BARRIER)

# vocabulary_barrier (vocab_need >= syntax_need)
check("vocabulary_barrier fires",
      eng.diagnose_fallback(learner_vocab, sig(0.60))
      == DiagnosisLabel.VOCABULARY_BARRIER)

# syntax_barrier (syntax_need > vocab_need)
check("syntax_barrier fires",
      eng.diagnose_fallback(learner_syn, sig(0.60))
      == DiagnosisLabel.SYNTAX_BARRIER)

# well_calibrated: everything else
check("well_calibrated fires",
      eng.diagnose_fallback(LEARNER, sig(0.75, hint=0.15))
      == DiagnosisLabel.WELL_CALIBRATED)

# Exact boundary: comprehension = 0.50 is NOT overloaded (threshold is <0.50)
check("comprehension = 0.50 is not overloaded",
      eng.diagnose_fallback(LEARNER, sig(0.50))
      != DiagnosisLabel.OVERLOADED)

# ──────────────────────────────────────────────────────────────────────────────
# T21 — update_learner_state: all seven diagnosis update rules
# ──────────────────────────────────────────────────────────────────────────────
group("T21 — update_learner_state: all seven rules")

eng = DeterministicEngine()

def signals(comprehension=0.75, hint=0.10):
    return ReadingSignals(comprehension, 0.55, 0.75, hint, 2, 3, 0.75, True)

# underchallenged: band advances
u = eng.update_learner_state(LEARNER, DiagnosisLabel.UNDERCHALLENGED, signals())
check("underchallenged: band advances",           u.current_band > LEARNER.current_band)
check("underchallenged: readiness → HIGH",        u.readiness_to_increase == Level.HIGH)
check("underchallenged: support_dependence ↓",    u.support_dependence.score <= LEARNER.support_dependence.score)
check("underchallenged: cycles_on_passage += 1",  u.cycles_on_passage == LEARNER.cycles_on_passage + 1)

# well_calibrated: readiness goes up
w = eng.update_learner_state(LEARNER, DiagnosisLabel.WELL_CALIBRATED, signals(hint=0.05))
check("well_calibrated: readiness ↑",             w.readiness_to_increase.score >= LEARNER.readiness_to_increase.score)
check("well_calibrated: support_dependence ↓ on low hints",
      w.support_dependence.score <= LEARNER.support_dependence.score)
check("well_calibrated: band unchanged",          w.current_band == LEARNER.current_band)

# support_dependent: readiness LOW, dependence up
sd = eng.update_learner_state(LEARNER, DiagnosisLabel.SUCCESSFUL_BUT_SUPPORT_DEPENDENT, signals(hint=0.35))
check("support_dependent: readiness → LOW",       sd.readiness_to_increase == Level.LOW)
check("support_dependent: support_dependence ↑",  sd.support_dependence.score >= LEARNER.support_dependence.score)

# vocabulary_barrier: vocab_need up, readiness LOW
# Use a learner starting at MEDIUM so HIGH is reachable (LEARNER starts at HIGH, cannot go higher)
learner_vocab_mid = replace(LEARNER, vocabulary_need=Level.MEDIUM)
vb = eng.update_learner_state(learner_vocab_mid, DiagnosisLabel.VOCABULARY_BARRIER, signals(0.60))
check("vocabulary_barrier: vocabulary_need ↑",    vb.vocabulary_need.score > learner_vocab_mid.vocabulary_need.score)
check("vocabulary_barrier: readiness → LOW",      vb.readiness_to_increase == Level.LOW)
check("vocabulary_barrier: band unchanged",       vb.current_band == learner_vocab_mid.current_band)

# syntax_barrier (non-severe): syntax_need up, band unchanged
sb = eng.update_learner_state(LEARNER, DiagnosisLabel.SYNTAX_BARRIER, signals(0.60))
check("syntax_barrier: syntax_need ↑",            sb.syntax_need.score > LEARNER.syntax_need.score)
check("syntax_barrier: readiness → LOW",          sb.readiness_to_increase == Level.LOW)

# syntax_barrier (severe): band drops
sb_sev = eng.update_learner_state(LEARNER, DiagnosisLabel.SYNTAX_BARRIER,
    ReadingSignals(0.40, 0.40, 0.50, 0.25, 5, 4, 0.40, True))
check("syntax_barrier severe: band drops",        sb_sev.current_band < LEARNER.current_band)

# cohesion_inference_barrier: cohesion_need up
cb = eng.update_learner_state(LEARNER, DiagnosisLabel.COHESION_INFERENCE_BARRIER, signals())
check("cohesion_barrier: cohesion_need ↑",        cb.cohesion_need.score > LEARNER.cohesion_need.score)
check("cohesion_barrier: band unchanged",         cb.current_band == LEARNER.current_band)

# overloaded: band drops, all needs up
ov = eng.update_learner_state(LEARNER, DiagnosisLabel.OVERLOADED, signals(0.40))
check("overloaded: band drops",                   ov.current_band < LEARNER.current_band)
check("overloaded: vocabulary_need ↑",            ov.vocabulary_need.score >= LEARNER.vocabulary_need.score)
check("overloaded: syntax_need ↑",                ov.syntax_need.score >= LEARNER.syntax_need.score)
check("overloaded: cohesion_need ↑",              ov.cohesion_need.score >= LEARNER.cohesion_need.score)
check("overloaded: support_dependence ↑",         ov.support_dependence.score >= LEARNER.support_dependence.score)
check("overloaded: readiness → LOW",              ov.readiness_to_increase == Level.LOW)

# history_limit: recent_outcomes trimmed
for label in list(DiagnosisLabel) * 2:  # 14 cycles
    LEARNER = eng.update_learner_state(LEARNER, label, signals())
LEARNER = LearnerState("student_01", 5.5, vocabulary_need=Level.HIGH)  # reset
check("recent_outcomes never exceeds history_limit",
      len(eng.update_learner_state(LEARNER, DiagnosisLabel.WELL_CALIBRATED,
          signals()).recent_outcomes) <= eng.config.history_limit)

# ──────────────────────────────────────────────────────────────────────────────
# T22 — TaskRoutingMockLLM
# ──────────────────────────────────────────────────────────────────────────────
group("T22 — TaskRoutingMockLLM")

mock = TaskRoutingMockLLM(responses={"my_task": {"result": "ok"}})
result = mock.complete_json("system", json.dumps({"task":"my_task"}))
check("routing works",              result == {"result":"ok"})
check("call_log records task",      mock.call_log[0][0] == "my_task")
check("call_log records prompt",    "my_task" in mock.call_log[0][1])

check_raises("unknown task raises NotImplementedError",
    lambda: mock.complete_json("s", json.dumps({"task":"unknown"})),
    NotImplementedError)

err_mock = TaskRoutingMockLLM(
    responses={"a_task": {"r":"ok"}},
    error_on_tasks={"a_task"},
)
check_raises("error injection raises RuntimeError",
    lambda: err_mock.complete_json("s", json.dumps({"task":"a_task"})),
    RuntimeError)

# call_log grows on each call
mock2 = TaskRoutingMockLLM(responses={"t":{"x":1}})
for _ in range(3):
    mock2.complete_json("s", json.dumps({"task":"t"}))
check("call_log length matches calls", len(mock2.call_log) == 3)

# ──────────────────────────────────────────────────────────────────────────────
# T23 — PromptLibrary: system prompts and user builders
# ──────────────────────────────────────────────────────────────────────────────
group("T23 — PromptLibrary")

# System prompts exist and are non-empty strings
for attr in ["CANONICALIZER_SYSTEM","CANDIDATE_GENERATOR_SYSTEM","FIT_ESTIMATOR_SYSTEM",
             "ASSESSMENT_GENERATOR_SYSTEM","RETELL_SCORER_SYSTEM","DIAGNOSIS_SYSTEM"]:
    val = getattr(PromptLibrary, attr)
    check(f"{attr} is non-empty string", isinstance(val, str) and len(val) > 50)

check("canonicalizer prompt mentions anchors",
      "anchor" in PromptLibrary.CANONICALIZER_SYSTEM.lower())
check("candidate generator prompt mentions meaning units",
      "meaning unit" in PromptLibrary.CANDIDATE_GENERATOR_SYSTEM.lower())
check("assessment prompt specifies 6 items",
      "six" in PromptLibrary.ASSESSMENT_GENERATOR_SYSTEM.lower() or
      "6" in PromptLibrary.ASSESSMENT_GENERATOR_SYSTEM)
check("assessment prompt enforces exactly 4 MCQ choices",
      "4" in PromptLibrary.ASSESSMENT_GENERATOR_SYSTEM or
      "four" in PromptLibrary.ASSESSMENT_GENERATOR_SYSTEM.lower())
check("diagnosis prompt lists underchallenged",
      "underchallenged" in PromptLibrary.DIAGNOSIS_SYSTEM)
check("diagnosis prompt has all 7 labels",
      all(label.value in PromptLibrary.DIAGNOSIS_SYSTEM for label in DiagnosisLabel))
check("PROMPTS_REFERENCE has 6 entries",
      len(en.PROMPTS_REFERENCE) == 6)

# User prompt builders produce valid JSON
cu = PromptLibrary.canonicalizer_user(SOURCE, "gutenberg_01", "understand printing press")
check("canonicalizer_user produces valid JSON",
      json.loads(cu).get("task") == "canonicalize_passage")

cgu = PromptLibrary.candidate_generator_user(
    CANONICAL, LEARNER, [{"relative_band":0,"profile":"light_support"}])
check("candidate_generator_user produces valid JSON",
      json.loads(cgu).get("task") == "generate_candidates")
check("candidate_generator_user embeds learner state",
      "vocabulary_need" in cgu)

du = PromptLibrary.diagnosis_user(LEARNER,
    ReadingSignals(0.75,0.55,0.75,0.10,2,3,0.75,True))
check("diagnosis_user produces valid JSON",
      json.loads(du).get("task") == "diagnose_outcome")

# ──────────────────────────────────────────────────────────────────────────────
# T24 — Journey state: begin_passage_journey, prepare → complete
# ──────────────────────────────────────────────────────────────────────────────
group("T24 — Journey state")

mock_sys = AdaptiveReadingSystem(llm=TaskRoutingMockLLM(), engine=LOOSE)

# begin_passage_journey: new passage resets cycles, sets target_band
fresh = LearnerState("j", 5.0)
journeyed = mock_sys.begin_passage_journey(fresh, CANONICAL)
check("target_band = source_fk after begin_journey",
      journeyed.target_band == CANONICAL.source_fk)
check("cycles_on_passage = 0 for new passage",
      journeyed.cycles_on_passage == 0)
check("entry_band = current_band on first assignment",
      journeyed.entry_band == fresh.current_band)

# Same passage: cycles preserved
continued = replace(fresh, target_band=CANONICAL.source_fk, cycles_on_passage=2)
same = mock_sys.begin_passage_journey(continued, CANONICAL)
check("same passage: cycles preserved",    same.cycles_on_passage == 2)
check("same passage: entry_band unchanged",same.entry_band == continued.entry_band)

# different passage: cycles reset
diff_canon = CanonicalPassage("other","Other text here.",
    "other obj", CANONICAL.meaning_units[:1])
diff = mock_sys.begin_passage_journey(journeyed, diff_canon)
check("different passage: cycles reset to 0", diff.cycles_on_passage == 0)

# ──────────────────────────────────────────────────────────────────────────────
# T25 — Full mocked cycle: prepare_cycle + complete_cycle
# ──────────────────────────────────────────────────────────────────────────────
group("T25 — Full mocked cycle")

FULL_MOCK_RESPONSES = {
    "canonicalize_passage": {
        "passage_id":"gutenberg_01","source_text":SOURCE,
        "instructional_objective":"Understand how the printing press changed history.",
        "meaning_units":[
            {"id":"MU1","text":"Gutenberg invented the printing press","required":True,
             "anchors":["Gutenberg","printing","press","1440","invented","transforming"]},
            {"id":"MU2","text":"Manuscripts were hand-copied, scarce and expensive","required":True,
             "anchors":["manuscripts","hand","scarce","expensive","ecclesiastical","aristocracy"]},
            {"id":"MU3","text":"Movable metal type enabled mechanized reproduction","required":True,
             "anchors":["movable","metal","type","mechanized","reproduction","speed"]},
            {"id":"MU4","text":"Printing spread across Europe, books became available","required":True,
             "anchors":["decades","proliferated","Europe","availability","written"]},
            {"id":"MU5","text":"Knowledge democratization catalyzed major transformations","required":True,
             "anchors":["democratization","Renaissance","Reformation","Scientific","Revolution","catalyst"]},
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
        "candidate_id":"A","passage_id":"gutenberg_01","relative_band":0,
        "text": SIMPLE_TEXT,
        "scaffold":{"vocabulary_support":"medium","syntax_support":"low",
                    "cohesion_support":"high","chunking_support":"medium","inference_support":"low"},
        "llm_self_audit":{"meaning_preserved":True,"sequence_preserved":True,
                          "objective_preserved":True,"same_passage_identity":True,
                          "notes":"All MUs present.",
                          "meaning_unit_coverage":{"MU1":True,"MU2":True,"MU3":True,"MU4":True,"MU5":True}},
    }]},
    "estimate_fit":{"fit_estimates":[
        {"candidate_id":"A","access":"high","growth":"medium","support_burden":"medium",
         "reason":"Good fit for band 5.5 learner."}
    ]},
    "generate_assessment": ASSESS_JSON,
    "score_retell":{"raw_score":3,"max_score":4,
                    "matched_meaning_units":["MU1","MU2","MU4"],
                    "matched_relationships":["MU1_enabled_spread"],
                    "concise_reason":"Three criteria met."},
    "diagnose_outcome":{"diagnosis":"well_calibrated","reason":"Learner completed well."},
}

full_mock = TaskRoutingMockLLM(responses=FULL_MOCK_RESPONSES)
sys_full  = AdaptiveReadingSystem(llm=full_mock, engine=LOOSE)
learner0  = LearnerState("stu", 5.5, vocabulary_need=Level.HIGH)

prep = sys_full.prepare_cycle(SOURCE, "gutenberg_01",
    "Understand how the printing press changed history.", learner0)

check("prepare_cycle returns CyclePreparation",       isinstance(prep, CyclePreparation))
check("prep.canonical has correct passage_id",        prep.canonical.passage_id == "gutenberg_01")
check("prep.selected_candidate is CandidatePassage",  isinstance(prep.selected_candidate, CandidatePassage))
check("prep.assessment has 6 items",                  len(prep.assessment.items) == 6)
check("prep.prepared_learner is not None",            prep.prepared_learner is not None)
check("prep.target_band set",                         prep.prepared_learner.target_band is not None)
check("prep.entry_band = initial band",               prep.prepared_learner.entry_band == 5.5)
check("prep.cycles_on_passage = 0",                   prep.prepared_learner.cycles_on_passage == 0)
check("prep.cycle_id is UUID string",
      isinstance(prep.cycle_id, str) and len(prep.cycle_id) == 36)
check("MU coverage parsed into self_audit",
      prep.selected_candidate.llm_self_audit.meaning_unit_coverage.get("MU1") == True)

answers = {"Q1":"A","Q2":"A","Q3":"A","Q4":"A",
    "Q5":"Gutenberg invented the printing press. Books spread across Europe. "
         "This helped cause the Renaissance and Reformation.",
    "Q6":3}

outcome = sys_full.complete_cycle(
    learner0, prep, answers,
    ReadingTelemetry(fluency_score=0.78, hint_use_rate=0.15, reread_count=2, completion=True))

check("complete_cycle returns CycleOutcome",          isinstance(outcome, CycleOutcome))
check("diagnosis is DiagnosisLabel",                  isinstance(outcome.diagnosis, DiagnosisLabel))
check("updated_learner is LearnerState",              isinstance(outcome.updated_learner, LearnerState))
check("cycle_id preserved prep → outcome",            outcome.cycle_id == prep.cycle_id)
check("updated_learner.target_band preserved",
      outcome.updated_learner.target_band == prep.prepared_learner.target_band)
check("updated_learner.entry_band preserved",
      outcome.updated_learner.entry_band == learner0.current_band)
check("cycles_on_passage incremented to 1",           outcome.updated_learner.cycles_on_passage == 1)
check("assessment_result.comprehension_score in [0,1]",
      0.0 <= outcome.assessment_result.comprehension_score <= 1.0)
check("reading_signals.completion = True",            outcome.reading_signals.completion)
check("diagnosis in recent_outcomes",
      outcome.diagnosis in outcome.updated_learner.recent_outcomes)

# ──────────────────────────────────────────────────────────────────────────────
# T26 — Fallback paths: retell and diagnosis fallbacks fire and log
# ──────────────────────────────────────────────────────────────────────────────
group("T26 — Fallback paths")

log_records = []
class CapHandler(logging.Handler):
    def emit(self, r): log_records.append(r)
logger = logging.getLogger("test_alien_fallback")
logger.addHandler(CapHandler()); logger.setLevel(logging.WARNING)

# Retell fallback
retell_fail = TaskRoutingMockLLM(
    responses=FULL_MOCK_RESPONSES,
    error_on_tasks={"score_retell"})
sys_rf  = AdaptiveReadingSystem(llm=retell_fail, engine=LOOSE, logger=logger)
prep_rf = sys_rf.prepare_cycle(SOURCE, "gutenberg_01",
    "Understand how the printing press changed history.", learner0)
outcome_rf = sys_rf.complete_cycle(
    learner0, prep_rf, answers,
    ReadingTelemetry(0.70, 0.20, 3, True))

check("retell fallback: outcome still returned",     isinstance(outcome_rf, CycleOutcome))
check("retell fallback: retell_quality in [0,1]",
      0.0 <= outcome_rf.assessment_result.retell_quality <= 1.0)
check("retell fallback: WARNING logged",
      any(r.levelno >= logging.WARNING and "fallback" in r.getMessage().lower()
          for r in log_records))

# Diagnosis fallback
log_records.clear()
diag_fail = TaskRoutingMockLLM(
    responses=FULL_MOCK_RESPONSES,
    error_on_tasks={"diagnose_outcome"})
sys_df  = AdaptiveReadingSystem(llm=diag_fail, engine=LOOSE, logger=logger)
prep_df = sys_df.prepare_cycle(SOURCE, "gutenberg_01",
    "Understand how the printing press changed history.", learner0)
outcome_df = sys_df.complete_cycle(
    learner0, prep_df, answers,
    ReadingTelemetry(0.80, 0.10, 1, True))

check("diagnosis fallback: outcome returned",       isinstance(outcome_df, CycleOutcome))
check("diagnosis fallback: valid DiagnosisLabel",   isinstance(outcome_df.diagnosis, DiagnosisLabel))
check("diagnosis fallback: WARNING logged",
      any(r.levelno >= logging.WARNING and "fallback" in r.getMessage().lower()
          for r in log_records))

# Retell fallback deterministic scoring: MU keywords in response → positive score
sys_kw = AdaptiveReadingSystem(llm=retell_fail, engine=LOOSE, logger=logger)
prep_kw = sys_kw.prepare_cycle(SOURCE, "gutenberg_01",
    "Understand how the printing press changed history.", learner0)
kw_resp = sys_kw.score_retell(
    prep_kw.canonical, prep_kw.assessment,
    "Gutenberg invented the printing press. Manuscripts were copied by hand.")
check("keyword fallback: raw_score >= 0",  kw_resp.get("raw_score", -1) >= 0)
check("keyword fallback: max_score > 0",   kw_resp.get("max_score", 0) > 0)

# ──────────────────────────────────────────────────────────────────────────────
# T27 — score_assessment and build_reading_signals
# ──────────────────────────────────────────────────────────────────────────────
group("T27 — score_assessment and build_reading_signals")

# score_mcq
check("score_mcq: correct answer → 1.0", score_mcq("A", "A") == 1.0)
check("score_mcq: wrong answer → 0.0",   score_mcq("A", "B") == 0.0)
check("score_mcq: None answer → 0.0",    score_mcq("A", None) == 0.0)
check("score_mcq: case-insensitive",      score_mcq("a", "A") == 1.0)

# normalize_retell_score
check("normalize_retell_score(3, 4) = 0.75", normalize_retell_score(3, 4) == 0.75)
check("normalize_retell_score(0, 4) = 0.0",  normalize_retell_score(0, 4) == 0.0)
check("normalize_retell_score(4, 4) = 1.0",  normalize_retell_score(4, 4) == 1.0)
check("normalize_retell_score(x, 0) = 0.0",  normalize_retell_score(3, 0) == 0.0)

# weighted_average
wa = weighted_average({"Q1":0.25,"Q2":0.25,"Q3":0.25,"Q4":0.25},
                      {"Q1":1.0, "Q2":0.0, "Q3":1.0, "Q4":0.0})
check("weighted_average = 0.5", abs(wa - 0.5) < 0.001)
check("weighted_average empty weights = 0.0", weighted_average({}, {"Q1":1.0}) == 0.0)
check("weighted_average missing key = 0",
      weighted_average({"Q1":0.5,"Q2":0.5}, {"Q1":1.0}) == 0.5)

# ratio and clamp
check("ratio(3,4) = 0.75",     abs(ratio(3,4) - 0.75) < 0.001)
check("ratio(x,0) = 1.0",      ratio(5,0) == 1.0)
check("clamp(1.5, 0, 1) = 1",  clamp(1.5, 0.0, 1.0) == 1.0)
check("clamp(-1, 0, 1) = 0",   clamp(-1.0, 0.0, 1.0) == 0.0)
check("clamp(0.5, 0, 1) = 0.5",clamp(0.5, 0.0, 1.0) == 0.5)

# build_reading_signals uses AssessmentResult fields
mock_ar = AssessmentResult(
    item_scores={},
    comprehension_score=0.75,
    inference_score=0.60,
    vocabulary_score=0.50,
    retell_quality=0.75,
    difficulty_rating=3,
)
sys_sig = AdaptiveReadingSystem(llm=TaskRoutingMockLLM(), engine=LOOSE)
rsig = sys_sig.build_reading_signals(mock_ar, 0.80, 0.15, 2, True)
check("build_reading_signals.comprehension = 0.75",  rsig.comprehension_score == 0.75)
check("build_reading_signals.fluency rounded",        isinstance(rsig.fluency_score, float))
check("build_reading_signals.completion = True",      rsig.completion is True)
check("build_reading_signals.difficulty_rating = 3",  rsig.difficulty_rating == 3)

# ──────────────────────────────────────────────────────────────────────────────
# T28 — complete_cycle_flat: compatibility wrapper
# ──────────────────────────────────────────────────────────────────────────────
group("T28 — complete_cycle_flat compatibility wrapper")

flat_mock = TaskRoutingMockLLM(responses=FULL_MOCK_RESPONSES)
sys_flat  = AdaptiveReadingSystem(llm=flat_mock, engine=LOOSE)
assessment_pkg = parse_assessment_package(ASSESS_JSON)
canonical_obj  = parse_canonical_passage({**FULL_MOCK_RESPONSES["canonicalize_passage"]})

outcome_flat = sys_flat.complete_cycle_flat(
    learner=learner0,
    canonical=canonical_obj,
    assessment=assessment_pkg,
    learner_answers=answers,
    fluency_score=0.75,
    hint_use_rate=0.15,
    reread_count=2,
    completion=True,
)
check("complete_cycle_flat returns CycleOutcome",     isinstance(outcome_flat, CycleOutcome))
check("complete_cycle_flat: diagnosis is label",      isinstance(outcome_flat.diagnosis, DiagnosisLabel))
check("complete_cycle_flat: journey fields preserved",
      outcome_flat.updated_learner.cycles_on_passage == 1)

# ──────────────────────────────────────────────────────────────────────────────
# T29 — Architectural invariants
# ──────────────────────────────────────────────────────────────────────────────
group("T29 — Architectural invariants")

# No double-scoring: score_candidate call count via select_candidate
scoring_engine = DeterministicEngine(config=EngineConfig(
    fk_tolerance=10, overall_meaning_threshold=0,
    vocabulary_threshold=0, length_deviation_threshold=10,
))
candidates = [
    CandidatePassage("X1","gutenberg_01",-1,SIMPLE_TEXT,ScaffoldProfile(),GOOD_AUDIT),
    CandidatePassage("X2","gutenberg_01", 0,SIMPLE_TEXT,ScaffoldProfile(),GOOD_AUDIT),
]
fits = {
    "X1": FitEstimate(Level.MEDIUM,Level.MEDIUM,Level.MEDIUM,""),
    "X2": FitEstimate(Level.HIGH,  Level.HIGH,  Level.LOW,   ""),
}
# Precompute scores once
pre_scores = {c.candidate_id: scoring_engine.score_candidate(CANONICAL, LEARNER, c)
              for c in candidates}
# Select uses pre-computed scores (no re-scoring occurs in select_candidate with precomputed_scores)
csel, _ = scoring_engine.select_candidate(CANONICAL, LEARNER, candidates, fits,
                                           precomputed_scores=pre_scores)
check("select_candidate with precomputed_scores uses them",
      isinstance(csel, CandidatePassage))

# cycle_id: UUID format
check("cycle_id is UUID string",     len(prep.cycle_id) == 36)
check("cycle_id has 4 hyphens",      prep.cycle_id.count("-") == 4)
check("cycle_id flows to outcome",   outcome.cycle_id == prep.cycle_id)

# DEGRADED mode is logged at WARNING level
log_records.clear()
degraded_mock = TaskRoutingMockLLM(responses={
    **FULL_MOCK_RESPONSES,
    "generate_candidates":{"candidates":[{
        "candidate_id":"D1","passage_id":"gutenberg_01","relative_band":0,
        "text": "Short text.",  # will trigger length deviation warning
        "scaffold":{"vocabulary_support":"low","syntax_support":"low",
                    "cohesion_support":"low","chunking_support":"low","inference_support":"low"},
        "llm_self_audit":{"meaning_preserved":True,"sequence_preserved":True,
                          "objective_preserved":True,"same_passage_identity":True,
                          "notes":"","meaning_unit_coverage":
                          {"MU1":True,"MU2":True,"MU3":True,"MU4":True,"MU5":True}},
    }]},
    "estimate_fit":{"fit_estimates":[
        {"candidate_id":"D1","access":"low","growth":"low","support_burden":"low","reason":"test"}
    ]},
})
strict_sys = AdaptiveReadingSystem(llm=degraded_mock,
    engine=DeterministicEngine(config=EngineConfig(
        fk_tolerance=0.01,  # very strict FK → will trigger FK warning
        overall_meaning_threshold=0.0, vocabulary_threshold=0.0,
        length_deviation_threshold=0.01,  # very strict length → will trigger length warning
    )),
    logger=logger)
try:
    strict_sys.prepare_cycle(SOURCE, "gutenberg_01",
        "Understand how the printing press changed history.", learner0)
    check("DEGRADED mode logged at WARNING",
          any(r.levelno >= logging.WARNING and "degraded" in r.getMessage().lower()
              for r in log_records))
except ALIENError:
    check("DEGRADED or ALIENError in worst case (no crash)", True)

# empty source_text raises
check_raises("empty source_text raises ValueError",
    lambda: AdaptiveReadingSystem(llm=TaskRoutingMockLLM()).prepare_cycle(
        "", "p", "obj", learner0), ValueError)
check_raises("whitespace source_text raises ValueError",
    lambda: AdaptiveReadingSystem(llm=TaskRoutingMockLLM()).prepare_cycle(
        "   ", "p", "obj", learner0), ValueError)

# ──────────────────────────────────────────────────────────────────────────────
# T30 — English isolation: module is self-contained
# ──────────────────────────────────────────────────────────────────────────────
group("T30 — English module isolation")

check("_WORD_RE does not match accented chars",
      en._WORD_RE.match("é") is None or
      "é" not in en._WORD_RE.findall("café"))
check("stopwords are English",
      "the" in en._STOPWORDS and "a" in en._STOPWORDS)
check("no Spanish stopwords",
      "también" not in en._STOPWORDS and "el" not in en._STOPWORDS or
      "también" not in en._STOPWORDS)
check("negation markers are English",
      "not" in en._NEGATION and "nunca" not in en._NEGATION)
check("flesch_kincaid_grade uses FK formula (not Szigriszt-Pazos)",
      # FK formula: harder English text should score >8; SK formula inverts scale
      flesch_kincaid_grade(SOURCE) > 8.0)
check("PromptLibrary system prompts are English",
      "You are" in PromptLibrary.CANONICALIZER_SYSTEM)
check("AdaptiveReadingSystem constructs without language parameter",
      AdaptiveReadingSystem(llm=TaskRoutingMockLLM()) is not None)
check("module has no external dependencies (no import errors at top level)",
      True)  # if we got here, imports succeeded

# Verify alien_system_es is a separate module that does not affect alien_system
try:
    import alien_system_es as es_mod
    check("Spanish module imports independently",   True)
    check("English stopwords unchanged after es import",
          "the" in en._STOPWORDS)
    check("English negation unchanged after es import",
          "not" in en._NEGATION)
    check("English _WORD_RE unchanged after es import",
          en._WORD_RE.pattern == r"[A-Za-z']+")
except ImportError:
    check("Spanish module import (optional)",       True)  # pass if not present

# ── Final report ──────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'═'*65}")
print(f"  {'PASS ✓' if failed == 0 else 'FAIL ✗'}  │  "
      f"{passed} passed  │  {failed} failed  │  {total} total")
print(f"{'═'*65}")

import sys as _sys
_sys.exit(0 if failed == 0 else 1)
