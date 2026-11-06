# Reassessment against Assignment 3

Standalone G/H note (6 November 2026): Appendix M now isolates G/H-only camera-y180 augmentation on the original-data SVM, with no mirroring, calibration or routing rule. It corrects 4 errors and loses 5 (net -1/60,251), with macro-F1 0.97524391 to 0.97523305. This is an exploratory negative result; the historical grade estimate below has not been reassessed. It does not establish real back-view recognition.

Mirror follow-up note (6 November 2026): the paper now additionally reports an RGB-mirroring experiment in Appendix L, with fresh detection of both presentations, a fixed balanced test mix, two learners and explicit original-view regressions. The historical assessment and grade estimate below have not been reassessed. Synthetic reflection results do not establish real left-handed signer performance.

Revision note (6 November 2026): the assessment below predates the same-coordinate control and is retained as a historical, indicative review, not a new score. The current paper centers interpretable hand descriptions, reports the full package's lack of SVM benefit, and distinguishes the world-joint net +67 from the same-coordinate net +4 result (Appendix K). The control does not isolate category-only value. A new G/H image-rotation experiment is future work; the existing G/H candidate rotates feature vectors. The numerical grade estimate below has not been reassessed for this revision.

Assessed: the revised English `report.pdf` and separate `appendix.pdf`. The main question is: **To what extent do explicit hand-shape features improve ASL letter recognition relative to a Shin-inspired reference?** Webcam, augmentation and later G/H experiments support the diagnosis; they are not additional requirements for answering this main question.

This is an AI-generated indicative assessment using the rubric supplied by the user, not an official grade or guarantee. Course level, the assessor's interpretation and whether the appendix counts can change the outcome. No new empirical data were collected for this revision. Improvements concern scope, evidence and reporting, rather than stronger independent validation.

## Indicative score

| Rubric component | Maximum | Estimate | Reason and main limitation |
|---|---:|---:|---|
| Abstract | 5 | 4 | Includes motivation, question, method, sample counts, accuracy/F1 and a bounded conclusion. Still information dense. |
| Introduction | 10 | 8 | Explicit main question, three testable subquestions, concrete motivation and bounded empirical contribution. No new algorithm. |
| Related work | 10 | 7 | Connects six primary works to geometry, landmarks, depth sensors, boosting, feature reliance and selection bias. Not a broad or recent ASL literature review. |
| Methods | 20 | 15 | Explains features, compatibility scores, algorithms, objectives, preprocessing and variants; technical formalization is in the appendix. Limited methodological novelty and incomplete historical detector provenance. |
| Experiments and Analysis | 40 | 28 | Shared images/folds, two algorithms, additions, ablations, permutation importance, nested weighting, all-letter and paired error analysis. No untouched final test or verified person-separated splits; repeated exploratory selection on the same evaluation. |
| Discussion | 10 | 8 | Answers the question and covers small effects, regressions, selection bias, detection coverage, webcam limitations, ethics and AI use. Some explanations remain hypotheses. |
| Conclusions | 5 | 4 | Concise answer with effect size, scope and focused future work. Generalization beyond the dataset remains unresolved. |
| **Total** | **100** | **74** | **Approximate grade: 7.4; a broad 7–8 range is more realistic than an exact grade.** |

## 1. Were ML techniques applied correctly?

The supplied rubric asks:

> Is the evaluation correctly performed from a data analytics/machine learning perspective (depending on the field)

It also asks:

> Does the evaluation contain proper baselines?
>
> Are there ablation experiments?
>
> Were there statistical analysis?

**Assessment: largely satisfied, with explicit limits.**

Strengths:

- The main table compares reference and feature variants on the same 60,251 accepted images and five frozen folds, reporting accuracy and macro-F1 for SVM and LightGBM.
- Angle imputation and scaling are learned on training partitions. Nested weighting separates inner selection and outer evaluation within that search.
- Addition, removal/retraining and permutation importance are distinguished. Importance is not presented as a causal or additive feature contribution.
- The 67 net additional correct predictions are decomposed into 143 corrected errors and 76 lost correct predictions. All letters and complete confusion matrices remain available.
- Statistical stability analysis is explicitly not an independent confidence interval or confirmatory significance test.
- The Shin comparison describes protocol differences and claims neither exact reproduction nor superiority to the published 87.60% result.

Limitations that still cost points:

1. Filename blocks are not verified people/sessions; related images may span groups.
2. The same evaluation repeatedly informed representation and follow-up choices. Nesting one weighting experiment does not remove selection bias across the whole development process.
3. Scores are conditional on successful detection (77.25% coverage), not end-to-end accuracy over all source images.
4. There is no untouched final test. Selected webcam errors are diagnostic material, not a representative test set.
5. Workbook references/relaxations were developed exploratorily, and an isolated category-only, no-workbook comparison is missing. Conclusions are limited to the experiments performed.

For the bounded research question these constrain the evidence; they do not require repeating every existing experiment. Strong claims about new users need prospective evidence.

## 2. Are the research questions substantive enough?

The supplied rubric has no literal requirement that questions must not be “too simple”. Its closest criteria are:

> Is the method complex/innovative?
>
> Is the contribution well argued?
>
> Is the method well formalized/explained?

**Assessment: sufficient for a bounded empirical ML study, with limited algorithmic novelty.**

The main question concerns more than building an app or fixing every webcam error. RQ1 tests the complete feature package and classifier dependence, RQ2 isolates continuous measurements, and RQ3 examines distribution and stability of the effect. Representation choices, observed gains and limitations are therefore testable.

The inconsistent effect of the full package, small SVM improvement from continuous angles, different LightGBM response and per-letter regressions are meaningful results. More models or features do not automatically improve the research. Assess the contribution as a controlled empirical comparison with error analysis, rather than a new algorithm.

Current data do not establish a causal palm/back-view effect, clinically accurate joint measurements, or resolution of webcam problems. These claims are excluded from the central conclusion.

## Improvements made in the report revision

- Abstract, questions, results and conclusion address the same main question.
- The main numerical comparison and all-letter effects appear in the main text, not only the appendix.
- RQ1 distinguishes the full package, category removal and missing isolation experiments.
- The main text briefly explains model settings, training-only preprocessing, selection bias and statistical limits.
- Later experiments remain transparent in Appendix I–J without making the main study appear unnecessarily unfinished.
- AI generation and the limited evidential scope of code inspection/unit tests are explicit.

## What would most improve the assessment?

1. One predefined comparison on a new, unseen participant/session. This strengthens the highest-weighted component; 120 photos are not an explicit rubric requirement.
2. A broader focused comparison with established hand-shape/orientation-robust representations, supported by reliable primary sources, to position novelty more clearly.
3. Independent review of workbook descriptions and targeted review of core code. Unit tests alone do not establish linguistic or anatomical validity.

Before submission, fill in author/student details and personally check sources and interpretation. These administrative requirements are not additional experimental demands.
