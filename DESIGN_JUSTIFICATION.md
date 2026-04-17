# ALIENS — Instructional Design Justification

**Adaptive Literacy Instruction in English and Spanish**  
**Theoretical Foundations and Evidence Base**

Jordan King · April 17, 2026 · Version 1.0.0

---

## Introduction

This document describes the instructional design principles underlying ALIENS and maps each principle to the specific components that implement it. For each principle the document states: the theoretical claim, its empirical basis, and the architectural decisions in ALIENS that enact it. Where the system's implementation departs from the theoretical ideal, or where empirical validation of specific parameter values is pending, that gap is identified explicitly.

The five principles are not independent. They form an integrated instructional logic: the zone of proximal development establishes the targeting criterion; comprehensible input operationalises what it means to be appropriately challenging; scaffolding and gradual release describes how support is structured and withdrawn; formative assessment closes the feedback loop; and curriculum coherence ensures that differentiation does not fragment the shared learning experience. Together they constitute a principled answer to one instructional problem: how do you give every learner in a classroom the same lesson while ensuring that each person can actually access it?

---

## Principle 1 — Zone of Proximal Development

### Theoretical claim

Vygotsky (1978) proposed that cognitive development occurs most efficiently in the zone of proximal development — the region between what a learner can accomplish independently and what they can accomplish with support. Instruction pitched within this zone maximises growth; instruction pitched below it produces no new learning, and instruction pitched above it produces frustration without acquisition. The zone is not a fixed property of the learner but a relational property of the learner and the task together.

### Empirical basis

The zone of proximal development framework has generated substantial empirical support in reading research. Chall's (1983) stages of reading development describe a progression from decoding-dependent to meaning-making literacy that maps onto band-based targeting. Clay's (1993) Running Record methodology operationalises the ZPD by categorising texts as independent, instructional, and frustration — a three-zone model that ALIENS refines into a continuous band with a push and safety-net structure. Stahl and Heubach (2005) found significant reading gains when instruction was consistently provided at the instructional rather than independent level. Allington (2012) reviewed decades of evidence concluding that the single variable most predictive of reading growth is the proportion of time learners spend reading texts at their instructional level.

### Implementation in ALIENS

**The continuous band.** Every learner has a `current_band` value expressed as a Flesch-Kincaid grade equivalent. This is an instructional working estimate, not a normed score, updated after every cycle by the deterministic state update rules. The band is continuous (floating-point) rather than categorical, allowing small incremental advances rather than the coarse jumps of grade-level systems.

**The push / safety-net structure.** The `build_candidate_plan()` method always generates at least three candidate slots: a safety-net variant at band −1 (the learner's current band minus one step), an on-level variant at band 0 with scaffolding tuned to the learner's need profile, and a push variant at band +1 when `readiness_to_increase` is MEDIUM or above. This directly implements the ZPD implication that optimal challenge sits slightly above the current independent level. The push candidate targets the upper edge of the zone; the safety-net candidate provides the lower boundary. The fit estimator evaluates whether the push candidate is accessible given the learner's current need profile; the deterministic engine selects it only when access is rated as MEDIUM or above.

**Band advancement rules.** The `underchallenged` diagnosis fires only when all four conditions hold simultaneously: comprehension ≥ 0.85, fluency ≥ 0.75, hint use ≤ 0.10, and retell quality ≥ 0.75. This compound criterion guards against spurious advancement. The `overloaded` diagnosis fires when comprehension falls below 0.50, the passage was not completed, or hint use is high with poor retell — and triggers a band step down. These thresholds operationalise the practical boundary between instructional and frustration levels.

### References

- Allington, R. L. (2012). *What really matters for struggling readers: Designing research-based programs* (3rd ed.). Pearson.
- Chall, J. S. (1983). *Stages of reading development*. McGraw-Hill.
- Clay, M. M. (1993). *An observation survey of early literacy achievement*. Heinemann.
- Stahl, S. A., & Heubach, K. M. (2005). Fluency-oriented reading instruction. *Journal of Literacy Research, 37*(1), 25–60.
- Vygotsky, L. S. (1978). *Mind in society: The development of higher psychological processes*. Harvard University Press.

---

## Principle 2 — Comprehensible Input and the i+1 Hypothesis

### Theoretical claim

Krashen (1985) proposed that language and literacy acquisition occurs when learners encounter input that is comprehensible but contains structures slightly beyond their current competence — what he termed i+1, where i is the learner's current level of knowledge. Comprehensible input is a necessary condition for acquisition; incomprehensible input (i+2 or beyond) does not result in learning regardless of the learner's effort. The hypothesis has been extended beyond second-language acquisition to apply to reading development more broadly, particularly in contexts where vocabulary and syntactic complexity function as barriers to meaning.

### Empirical basis

Nation and Webb (2011) provided vocabulary-specific evidence for the comprehensible input threshold, estimating that learners need to know approximately 95–98% of words in a text to read it with adequate comprehension — a finding that directly motivates vocabulary-specific scaffolding for learners whose lexical range falls below this threshold. Beck, McKeown, and Kucan (2013) demonstrated that vocabulary knowledge at Tier 2 (general academic vocabulary) is a significant predictor of reading comprehension across content areas, and that direct instruction in Tier 2 vocabulary produces measurable comprehension gains. Nagy and Townsend (2012) found that academic language proficiency accounts for substantial variance in reading comprehension beyond what is explained by word recognition alone. Shanahan and Shanahan (2008) extended this to disciplinary literacy, showing that the vocabulary demands of academic content texts are domain-specific and require explicit attention.

### Implementation in ALIENS

**Five scaffold dimensions.** The `ScaffoldProfile` dataclass tracks five dimensions: vocabulary support, syntax support, cohesion support, chunking support, and inference support — each on a LOW/MEDIUM/HIGH scale. The LLM candidate generator is instructed to apply these dimensions independently, so a learner with HIGH vocabulary need and LOW syntax need receives a variant with inline glosses on Tier-2 and Tier-3 terms but otherwise full sentence complexity. This disaggregation directly addresses the empirical finding that vocabulary, syntax, and cohesion barriers are functionally distinct and require different instructional responses.

**Vocabulary preservation contract.** The canonical passage specifies a `must_preserve_vocabulary` list with a `gloss_allowed` flag for each term. Required vocabulary terms are enforced as a hard blocking constraint in `score_candidate()`: any candidate that omits a required term fails validation regardless of its other qualities. Terms marked `gloss_allowed=True` may appear with parenthetical definitions in learner variants; terms marked `gloss_allowed=False` must appear unglossed. This implements the comprehensible input principle at the level of individual lexical items: the domain vocabulary is preserved, and the scaffold helps the learner access it rather than removing it.

**Distance-aware validation thresholds.** The `_scaled_thresholds()` method relaxes meaning coverage and vocabulary coverage requirements as the distance between the source passage and the learner's band increases. A deep rewrite for a low-band learner necessarily departs further from the source vocabulary than a light rewrite for a high-band learner. The scaled thresholds ensure that deep rewrites are not rejected for being different from the source — they are accepted as long as they preserve the required meaning and vocabulary at a level appropriate for the rewrite distance.

### References

- Beck, I. L., McKeown, M. G., & Kucan, L. (2013). *Bringing words to life: Robust vocabulary instruction* (2nd ed.). Guilford Press.
- Krashen, S. D. (1985). *The input hypothesis: Issues and implications*. Longman.
- Nagy, W., & Townsend, D. (2012). Words as tools: Learning academic vocabulary as language acquisition. *Reading Research Quarterly, 47*(1), 91–108.
- Nation, I. S. P., & Webb, S. (2011). *Researching and analyzing vocabulary*. Heinle Cengage Learning.
- Shanahan, T., & Shanahan, C. (2008). Teaching disciplinary literacy to adolescents: Rethinking content-area literacy. *Harvard Educational Review, 78*(1), 40–59.

---

## Principle 3 — Instructional Scaffolding and Gradual Release of Responsibility

### Theoretical claim

Wood, Bruner, and Ross (1976) introduced the concept of scaffolding — temporary support structures that enable a learner to perform at a level beyond their current independent capability. Critically, scaffolding is contingent and temporary: it is adjusted in response to learner performance and progressively withdrawn as the learner internalises the supported skill. Scaffolding that is not withdrawn produces dependency rather than acquisition. Pearson and Gallagher (1983) formalised this into the Gradual Release of Responsibility model, in which instruction moves from modelled performance through guided practice with support to independent performance — a sequence they showed was superior to either pure modelling or pure independent practice alone.

### Empirical basis

Rosenshine (2012) identified scaffolding as one of the ten most well-evidenced instructional practices, drawing on decades of research, with particular emphasis on modelling and guided practice before independent work. Fisher and Frey (2008) applied the gradual release model specifically to literacy instruction and demonstrated improved outcomes in reading comprehension when the model was implemented consistently. Duke, Pearson, Strachan, and Billman (2011) reviewed evidence for comprehension strategy instruction and found that scaffolded gradual release — with explicit modelling, supported practice, and progressive independence — produced larger and more durable comprehension gains than strategy instruction without the gradual release structure. Hattie and Timperley (2007) found that feedback is most effective when it is specific, immediate, and directed at the gap between current performance and the learning goal.

### Implementation in ALIENS

**The `support_dependence` dimension.** `LearnerState` tracks `support_dependence` on a LOW/MEDIUM/HIGH scale. This dimension rises when the `successful_but_support_dependent` diagnosis fires (comprehension ≥ 0.70 but hint use ≥ 0.30) and falls when hint use is low on a `well_calibrated` cycle. A learner who consistently requires high levels of scaffolding to succeed is flagged — their scaffold is not being withdrawn. The candidate generator's fit estimator penalises candidates with high support burden for learners with low support dependence, pushing the system towards less-scaffolded variants as the learner demonstrates independence.

**Progressive scaffold withdrawal.** The `readiness_to_increase` field rises incrementally with each `well_calibrated` cycle and falls to LOW on any barrier or overloaded diagnosis. Only when readiness reaches MEDIUM does the candidate plan include a push variant at band +1. A single cycle of difficulty returns readiness to LOW, preventing premature withdrawal of support.

**Inference support as the final scaffold.** When readiness is LOW, `build_candidate_plan()` adds an `inference_support` slot rather than a push slot. The inference_support profile instructs the LLM to complete causal inferences explicitly in the text before asking the learner to draw them — a form of modelled performance in Pearson and Gallagher's terms. When a `cohesion_inference_barrier` diagnosis fires on consecutive cycles, the inference_support slot is activated, providing direct modelling of the reasoning process the learner is failing to execute independently.

### References

- Duke, N. K., Pearson, P. D., Strachan, S. L., & Billman, A. K. (2011). Essential elements of fostering and teaching reading comprehension. In S. J. Samuels & A. E. Farstrup (Eds.), *What research has to say about reading instruction* (4th ed., pp. 51–93). International Reading Association.
- Fisher, D., & Frey, N. (2008). *Better learning through structured teaching: A framework for the gradual release of responsibility*. ASCD.
- Hattie, J., & Timperley, H. (2007). The power of feedback. *Review of Educational Research, 77*(1), 81–112.
- Pearson, P. D., & Gallagher, M. C. (1983). The instruction of reading comprehension. *Contemporary Educational Psychology, 8*(3), 317–344.
- Rosenshine, B. (2012). Principles of instruction: Research-based strategies that all teachers should know. *American Educator, 36*(1), 12–19.
- Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry, 17*(2), 89–100.

---

## Principle 4 — Formative Assessment for Learning

### Theoretical claim

Black and Wiliam (1998) published a landmark synthesis of over 250 studies finding that formative assessment — assessment used to inform ongoing instructional decisions rather than to grade or certify — produced substantial learning gains where innovations were designed to strengthen the frequent feedback students receive. The key characteristics of effective formative assessment identified in their review were: it provides specific information about the gap between current performance and the learning goal; it is used by the teacher (or system) to adjust instruction; and it is immediate enough to affect the next instructional decision. Wiliam (2011) quantified the expected impact of well-implemented classroom formative assessment as an increase in the rate of student learning of between 50 and 70 percent, citing the body of evidence compiled in the 1998 synthesis and subsequent work. He argued that the most powerful form of formative assessment is embedded in instruction rather than administered as a separate event — a criterion that ALIENS meets by design.

### Empirical basis

Fuchs and Fuchs (1986) conducted a meta-analysis of curriculum-based measurement studies and found that systematic data collection and use for instructional decision-making produced significantly better outcomes than instruction without such feedback (average weighted effect size 0.70). The specific advantage of fine-grained, multi-dimensional assessment over holistic assessment was demonstrated by Catts, Hogan, and Adlof (2005), who showed that comprehension difficulties in early readers cluster into distinct subtypes — vocabulary-limited, inference-limited, and decoding-limited — each of which requires a different instructional response. This finding directly motivates ALIENS's seven-label diagnosis taxonomy, which distinguishes vocabulary barriers from syntax barriers from cohesion-inference barriers rather than collapsing them into a single category. Deno (2003) reviewed decades of curriculum-based measurement research and concluded that the frequency and specificity of assessment data collection is a more powerful predictor of student growth than almost any other instructional variable studied.

### Implementation in ALIENS

**The six-item diagnostic structure.** Each cycle includes exactly six assessment items: a literal comprehension question, a sequence question, an inference question, a vocabulary question, a retell prompt scored against a meaning-unit rubric, and a difficulty self-rating. The four MCQ items provide independent signal on comprehension, sequence understanding, inference capability, and vocabulary knowledge. The retell provides a holistic signal against the full meaning profile. The self-rating provides metacognitive data on perceived challenge.

**The seven diagnosis labels.** The taxonomy maps the signal profile to one of seven instructionally specific labels, each triggering a different state update. `vocabulary_barrier` and `syntax_barrier` are distinguished by comparing `vocabulary_need.score` against `syntax_need.score` at the point where comprehension falls below 0.70, correctly attributing the barrier to the predominant need dimension. `cohesion_inference_barrier` fires only when comprehension is adequate (≥ 0.70) but inference is not (< 0.55), identifying the specific breakdown point rather than attributing the failure to global comprehension. This seven-way specificity is what makes the formative loop useful: a system that only distinguished passed from failed could not adjust the scaffold profile appropriately.

**Immediate state update.** The state update from each cycle is applied immediately and deterministically before the next cycle begins. The updated `LearnerState` is passed directly to `build_candidate_plan()` for the next passage. There is no delay between assessment and instructional adjustment. The LLM diagnosis call provides qualitative reasoning for human review; the deterministic fallback ensures the system functions correctly even if the LLM call fails.

### References

- Black, P., & Wiliam, D. (1998). Assessment and classroom learning. *Assessment in Education: Principles, Policy & Practice, 5*(1), 7–74.
- Catts, H. W., Hogan, T. P., & Adlof, S. M. (2005). Developmental changes in reading and reading disabilities. In H. W. Catts & A. G. Kamhi (Eds.), *The connections between language and reading disabilities* (pp. 25–40). Lawrence Erlbaum.
- Deno, S. L. (2003). Developments in curriculum-based measurement. *The Journal of Special Education, 37*(3), 184–192.
- Fuchs, L. S., & Fuchs, D. (1986). Effects of systematic formative evaluation: A meta-analysis. *Exceptional Children, 53*(3), 199–208.
- Wiliam, D. (2011). *Embedded formative assessment*. Solution Tree Press.

---

## Principle 5 — Curriculum Coherence and the Shared Content Principle

### Theoretical claim

Differentiated instruction is most effective when it preserves a common content core while varying the surface features through which that content is accessed. Tomlinson (2001) argued that the goal of differentiation is not to give different learners different knowledge, but to give every learner access to the same essential knowledge through different instructional entry points. Schmidt, Wang, and McKnight (2005) demonstrated that curriculum coherence — the alignment of content, sequence, and depth across a school year — is a stronger predictor of learning gains than instructional approach, class size, or teacher experience. Porter (2002) introduced the concept of content coverage as a primary explanatory variable in educational achievement, arguing that what students are taught explains more variance in outcomes than how they are taught.

### Empirical basis

Hiebert and Pearson (2014) reviewed evidence on text complexity and argued that the key instructional variable is not the difficulty of the text per se, but the conceptual challenge of what learners are asked to do with it — and that conceptual challenge should be consistent across learners regardless of surface text complexity. This directly motivates the ALIENS canonical passage contract: every learner is asked to engage with the same conceptual content even if the surface form of the text varies. Cain, Oakhill, and Lemmon (2004) found that inference-making and comprehension of narrative structure are teachable skills that transfer across text difficulty levels — a learner who develops inference-making capability on a simpler text can apply that capability to harder texts. For learners with dyslexia, Hulme and Snowling (2011) reviewed evidence showing that comprehension development is relatively independent of decoding development, directly motivating the preservation of full conceptual content in the decoding_support variant rather than reducing the inferential demands. Snowling and Hulme (2012) further documented that the systematic assignment of learners to material matched to their decoding level rather than their comprehension level is a significant factor in the underachievement of dyslexic learners with grade-appropriate comprehension.

### Implementation in ALIENS

**The canonical passage contract.** Every learner variant is generated from and validated against a single `CanonicalPassage` object. The canonical specifies the meaning units, their required sequence, and the required vocabulary terms. These are invariants that cannot vary across variants. A candidate that fails to cover a required meaning unit is blocked from selection entirely, regardless of how well it serves other criteria. A candidate that presents meaning units out of sequence is similarly blocked. This hard constraint ensures that every learner in a classroom has encountered the same conceptual content, in the same order, by the end of the cycle.

**Shared vocabulary for class discussion.** Required vocabulary terms are preserved in every variant. A learner receiving a heavily scaffolded variant still encounters the term "cursive," "photosynthesis," or "sovereignty" — they may encounter it with a gloss, but they encounter it. When the class discusses the passage, every learner has a referent for the key vocabulary. This is the mechanism by which ALIENS ensures that differentiation does not produce curricular fragmentation: the vocabulary and the conceptual content are shared; only the surface accessibility differs.

**Sequence constraints as narrative coherence.** The `sequence_constraints` in the canonical passage encode the causal or temporal order of the meaning units. A variant in which the climax precedes the inciting event, or in which the conclusion precedes the evidence, violates the instructional objective even if all the information is present. The deterministic engine enforces sequence constraints as blocking failures — not warnings — because a sequence violation degrades the learner's comprehension of how the ideas relate to each other.

**Dyslexia and the content preservation principle.** The `decoding_support` variant produced by `DyslexiaAwareDeterministicEngine` applies the shared content principle in its most demanding form: the surface sentences are reduced to short, simply structured clauses, but the meaning units, sequence constraints, and required vocabulary are all preserved identically to the standard on-level variant. A dyslexic learner with a grade-7.5 comprehension band receives the same conceptual challenge as any other grade-7.5 learner. The only thing that changes is the decoding demand.

### References

- Cain, K., Oakhill, J., & Lemmon, K. (2004). Individual differences in the inference of word meanings from context: The influence of reading comprehension, vocabulary knowledge, and memory capacity. *Journal of Educational Psychology, 96*(4), 671–681.
- Hiebert, E. H., & Pearson, P. D. (2014). Understanding text complexity: The knowledge base for instruction. In K. A. Hinchman & H. K. Sheridan-Thomas (Eds.), *Best practices in adolescent literacy instruction* (2nd ed., pp. 3–24). Guilford Press.
- Hulme, C., & Snowling, M. J. (2011). Children's reading comprehension difficulties: Nature, causes, and treatments. *Current Directions in Psychological Science, 20*(3), 139–142.
- Porter, A. C. (2002). Measuring the content of instruction: Uses in research and practice. *Educational Researcher, 31*(7), 3–14.
- Schmidt, W. H., Wang, H. C., & McKnight, C. C. (2005). Curriculum coherence: An examination of US mathematics and science content standards from an international perspective. *Journal of Curriculum Studies, 37*(5), 525–559.
- Snowling, M. J., & Hulme, C. (2012). Annual research review: The nature and classification of reading disorders — a commentary on proposals for DSM-5. *Journal of Child Psychology and Psychiatry, 53*(5), 593–607.
- Tomlinson, C. A. (2001). *How to differentiate instruction in mixed-ability classrooms* (2nd ed.). ASCD.

---

## Summary: Five Principles Mapped to System Architecture

| Principle | Primary source | ALIENS component(s) |
|---|---|---|
| Zone of Proximal Development | Vygotsky (1978) | `current_band` · `build_candidate_plan()` push/safety-net slots · `underchallenged` and `overloaded` thresholds · band step-up/step-down rules |
| Comprehensible Input / i+1 | Krashen (1985) | `ScaffoldProfile` five dimensions · `must_preserve_vocabulary` with `gloss_allowed` · `_scaled_thresholds()` distance-aware validation · `vocabulary_barrier` and `syntax_barrier` labels |
| Scaffolding & Gradual Release | Wood, Bruner & Ross (1976); Pearson & Gallagher (1983) | `support_dependence` tracking · `readiness_to_increase` two-gate advancement · `inference_support` slot · `successful_but_support_dependent` diagnosis |
| Formative Assessment | Black & Wiliam (1998); Wiliam (2011) | Six-item diagnostic per cycle · seven diagnosis labels · immediate deterministic state update · LLM diagnosis with deterministic fallback · disaggregated signals (comprehension, inference, vocabulary, fluency, hint use, retell) |
| Curriculum Coherence | Schmidt et al. (2005); Porter (2002); Tomlinson (2001) | `CanonicalPassage` meaning units as hard invariants · `sequence_constraints` as blocking constraints · `must_preserve_vocabulary` across all variants · `decoding_support` format preserves full conceptual content |

---

## Limitations and Open Uncertainties

This document describes the theoretical and empirical basis for ALIENS's design. It does not constitute evidence that the system achieves its intended outcomes in deployment. Several important uncertainties remain.

**Parameter values are not empirically validated for this system.** The thresholds in `EngineConfig` — 0.85 for `underchallenged`, 0.55 for `cohesion_inference_barrier`, 0.10 for low hint use — are set based on the theoretical framework and informed judgment. They have not been calibrated against student outcome data from actual ALIENS deployments. The dyslexia signal adjustment constants (`FLUENCY_SCALE = 0.9`, `HINT_DISCOUNT = 0.5`, `COMPREHENSION_GUARD = 0.70`) are explicitly noted in `alien_dyslexia.py` as not empirically validated for this system specifically.

**The band measure is not a normed assessment.** `current_band` is a working instructional estimate based on Flesch-Kincaid grade of delivered texts and learner performance signals. It is not equivalent to a Lexile score, a DRA level, or a standardised reading assessment. It has not been validated against normed instruments.

**LLM-generated content is not validated beyond structural contracts.** The system validates that generated passages preserve the structural and semantic properties of the canonical passage. It does not validate factual accuracy, age-appropriateness, or tone. Human review of generated passages prior to deployment is strongly recommended.

**The dyslexia extension has not been validated with clinical populations.** The dyslexia extension is designed to prevent misclassification based on well-established research on the decoding-comprehension distinction. It has not been trialled with formally assessed dyslexic learners, and the assumption that fluency and hint signals reflect decoding rather than comprehension difficulty may not hold for all learners or all passage types.

---

## Full Reference List

Allington, R. L. (2012). *What really matters for struggling readers: Designing research-based programs* (3rd ed.). Pearson.

Beck, I. L., McKeown, M. G., & Kucan, L. (2013). *Bringing words to life: Robust vocabulary instruction* (2nd ed.). Guilford Press.

Black, P., & Wiliam, D. (1998). Assessment and classroom learning. *Assessment in Education: Principles, Policy & Practice, 5*(1), 7–74.

Cain, K., Oakhill, J., & Lemmon, K. (2004). Individual differences in the inference of word meanings from context: The influence of reading comprehension, vocabulary knowledge, and memory capacity. *Journal of Educational Psychology, 96*(4), 671–681.

Catts, H. W., Hogan, T. P., & Adlof, S. M. (2005). Developmental changes in reading and reading disabilities. In H. W. Catts & A. G. Kamhi (Eds.), *The connections between language and reading disabilities* (pp. 25–40). Lawrence Erlbaum.

Chall, J. S. (1983). *Stages of reading development*. McGraw-Hill.

Clay, M. M. (1993). *An observation survey of early literacy achievement*. Heinemann.

Deno, S. L. (2003). Developments in curriculum-based measurement. *The Journal of Special Education, 37*(3), 184–192.

Duke, N. K., Pearson, P. D., Strachan, S. L., & Billman, A. K. (2011). Essential elements of fostering and teaching reading comprehension. In S. J. Samuels & A. E. Farstrup (Eds.), *What research has to say about reading instruction* (4th ed., pp. 51–93). International Reading Association.

Fisher, D., & Frey, N. (2008). *Better learning through structured teaching: A framework for the gradual release of responsibility*. ASCD.

Fuchs, L. S., & Fuchs, D. (1986). Effects of systematic formative evaluation: A meta-analysis. *Exceptional Children, 53*(3), 199–208.

Hattie, J., & Timperley, H. (2007). The power of feedback. *Review of Educational Research, 77*(1), 81–112.

Hiebert, E. H., & Pearson, P. D. (2014). Understanding text complexity: The knowledge base for instruction. In K. A. Hinchman & H. K. Sheridan-Thomas (Eds.), *Best practices in adolescent literacy instruction* (2nd ed., pp. 3–24). Guilford Press.

Hulme, C., & Snowling, M. J. (2011). Children's reading comprehension difficulties: Nature, causes, and treatments. *Current Directions in Psychological Science, 20*(3), 139–142.

Krashen, S. D. (1985). *The input hypothesis: Issues and implications*. Longman.

Nagy, W., & Townsend, D. (2012). Words as tools: Learning academic vocabulary as language acquisition. *Reading Research Quarterly, 47*(1), 91–108.

Nation, I. S. P., & Webb, S. (2011). *Researching and analyzing vocabulary*. Heinle Cengage Learning.

Pearson, P. D., & Gallagher, M. C. (1983). The instruction of reading comprehension. *Contemporary Educational Psychology, 8*(3), 317–344.

Porter, A. C. (2002). Measuring the content of instruction: Uses in research and practice. *Educational Researcher, 31*(7), 3–14.

Rosenshine, B. (2012). Principles of instruction: Research-based strategies that all teachers should know. *American Educator, 36*(1), 12–19.

Schmidt, W. H., Wang, H. C., & McKnight, C. C. (2005). Curriculum coherence: An examination of US mathematics and science content standards from an international perspective. *Journal of Curriculum Studies, 37*(5), 525–559.

Shanahan, T., & Shanahan, C. (2008). Teaching disciplinary literacy to adolescents: Rethinking content-area literacy. *Harvard Educational Review, 78*(1), 40–59.

Snowling, M. J., & Hulme, C. (2012). Annual research review: The nature and classification of reading disorders — a commentary on proposals for DSM-5. *Journal of Child Psychology and Psychiatry, 53*(5), 593–607.

Stahl, S. A., & Heubach, K. M. (2005). Fluency-oriented reading instruction. *Journal of Literacy Research, 37*(1), 25–60.

Tomlinson, C. A. (2001). *How to differentiate instruction in mixed-ability classrooms* (2nd ed.). ASCD.

Vygotsky, L. S. (1978). *Mind in society: The development of higher psychological processes*. Harvard University Press.

Wiliam, D. (2011). *Embedded formative assessment*. Solution Tree Press.

Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry, 17*(2), 89–100.
