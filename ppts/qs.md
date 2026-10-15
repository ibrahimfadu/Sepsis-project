# Capstone Project Preparation Document

## A Comparative Framework for Early Sepsis Prediction Using Ensemble and Explainable Deep Learning on ICU Data

---

## How to Use This Document

This document contains everything you need to prepare for the capstone review. It is organized as follows:

1. **Project Summary** — one-page overview
2. **Slide-by-Slide Explanation** — what each slide says and why
3. **Likely Questions & Answers** — what reviewers may ask
4. **Key Concepts Explained Simply** — for quick understanding
5. **Key Papers & Links** — all references
6. **Team Roles** — who is responsible for what

---

## 1. Project Summary

**Title**: A Comparative Framework for Early Sepsis Prediction Using Ensemble and Explainable Deep Learning on ICU Data

**Project ID**: 186

**One-Sentence Summary**: We compare ensemble models (XGBoost, Random Forest) against deep learning models (LSTM) for early sepsis prediction under identical conditions, and build a reliability layer that detects, diagnoses, and mitigates unreliable SHAP explanations, abstaining when reliability is too low.

**Two Objectives**:
1. **Unified Comparison**: Compare ensemble vs. deep learning models under identical preprocessing, patient-level splits, and evaluation protocols.
2. **Reliability Layer**: Build a framework that detects unstable, unfaithful, or non-robust SHAP explanations, diagnoses the failure mode, applies targeted mitigation, and abstains when reliability is too low.

**Dataset**: PhysioNet/CinC Challenge 2019 (~40,336 ICU patients, 40 clinical variables, hourly time-series)

**Novelty**: First systematic reliability evaluation framework for SHAP explanations in early sepsis prediction, including explanation-level abstention.

---

## 2. Slide-by-Slide Explanation

### Slide 1: Title

**Content**: Project title, ID, guide, team members.

**What to say**: Brief introduction of the project and team.

---

### Slide 2: Agenda

**Content**: List of sections: Problem Statement, Scope and Feasibility, Background, Background Work, Proposed Approach, Expected Deliverables, Timeline, Any Other Information.

**What to say**: Walk through the structure of the presentation.

---

### Slide 3: Problem Statement

**Content**:
- Sepsis is a life-threatening organ dysfunction where mortality rises every hour treatment is delayed
- Early prediction 6–12h before onset is critical
- ML models achieve strong performance (ensemble AUROC 0.85–0.91, LSTM ~0.85)
- SHAP is the default explanation method
- **Two gaps**: 
  1. No fair comparison — models evaluated on different datasets, splits, protocols
  2. No reliability check — SHAP explanations never tested for stability, faithfulness, robustness
- **Objective**: (i) unified comparison, (ii) reliability layer with abstention

**What to say**: Sepsis is deadly and time-critical. AI helps predict it, but two problems exist: we cannot fairly compare models, and we never check if SHAP explanations are trustworthy.

**Likely Question**: "Why is SHAP reliability a problem?"

**Answer**: Clinical variables are correlated, and 85–95% of lab values are missing. 2026 theoretical work proves that when features are correlated, no feature ranking can simultaneously be faithful, stable, and complete. So SHAP explanations in sepsis are mathematically guaranteed to be unstable under retraining — but clinicians receive them with no warning.

---

### Slide 4: Scope and Feasibility Study

**Content**:
- **In scope**: PhysioNet 2019, XGBoost/RF/LSTM, TreeSHAP/KernelSHAP/IG, stability/faithfulness/robustness, detect/diagnose/mitigate/abstain
- **Out of scope**: New architectures, AUROC records, prospective validation, multimodal data, real-time deployment
- **Feasibility**: Implementation (mature libraries), Evaluation (independent failure labels), Systematic (detection not elimination), Practical (post-hoc layer)
- **Challenges–Mitigation**: No benchmark → define labels; 85–95% missing → compare imputed vs dropped; correlated features → Latent SHAP; circular evaluation → labels from retraining/perturbation/disagreement

**What to say**: We are scoped to a 12-week prototype. We do not build new prediction models or deploy in hospitals. All components use mature libraries. The key challenge is defining independent failure labels before building the detector, which avoids circular evaluation.

**Likely Question**: "Why not use MIMIC-IV as primary dataset?"

**Answer**: PhysioNet 2019 is publicly available without access restrictions, widely used as a benchmark, and has standardized labels. MIMIC-IV requires data access approval and is listed for external validation if feasible.

---

### Slide 5: Background (Domain Context)

**Content**:
- Stat cards: AUROC 0.85–0.91 (ensemble), 85–95% missing labs, +67%/yr explainability adoption
- Domain context: Early sepsis prediction uses ensemble + deep learning, SHAP is default. Attribution Impossibility (2026) proves no ranking can be faithful, stable, complete under correlation. Clinician sees "respiratory rate" then "heart rate" after retraining — prediction stable, explanation not.

**What to say**: Three numbers establish the context: ensemble models reach AUROC 0.85–0.91, labs are 85–95% missing, and explainability adoption is growing 67% per year. The theoretical result proves explanation instability is mathematically guaranteed.

**Likely Question**: "What is Attribution Impossibility?"

**Answer**: A 2026 theoretical result proving that when features are correlated, no feature ranking method can simultaneously be faithful (reflect model behavior), stable (consistent across retraining), and complete (cover all relevant features). Since clinical variables are inherently correlated, this applies directly to sepsis prediction.

---

### Slide 6: Background Work

**Content**: Table with four columns — Work, What it shows, What is missing:
- Ensemble models: AUROC 0.85–0.91, compared across different pipelines
- LSTM-GAT; MIMIC-Sepsis: AUROC 0.853/0.919, no explanation reliability
- XAI review (37 studies): +67%/yr, key biomarkers rarely rank high, diagnoses gap but no solution
- SHAP failure theory: SHAP can mislead, many variants differ, theoretical only
- Latent SHAP, ReCalX, UbiQTree: each fixes one failure mode, not a detector, not applied to sepsis
- **Our framework**: Unified comparison + detect, diagnose, mitigate, abstain

**What to say**: This table summarizes what exists and what is missing. Every existing work either focuses on prediction accuracy, diagnoses the gap without a solution, or fixes one failure mode in isolation. No one has built a complete reliability framework for sepsis.

**Likely Question**: "What is the research gap?"

**Answer**: No study evaluates whether SHAP explanations for sepsis are stable, faithful, and robust — or abstains when they are not. That is our gap.

---

### Slide 7: Proposed Approach

**Content**: Flow diagram:
ICU time-series → Prediction models (XGBoost, RF, LSTM) → Explanations (TreeSHAP, KernelSHAP, IG) → **Detect → Diagnose → Mitigate → Abstain** (blue boxes = our contribution)
- Reliability score: R(x) = w_s·Stability + w_f·Faithfulness + w_r·Robustness
- Weights start at 1/3 each, tuned on validation set
- Example output: "Prediction: 82% sepsis risk. Explanation reliability: LOW. Top feature ranking is unstable across model retraining. Use clinical judgment."

**What to say**: The blue boxes are our contribution. The first three steps (data, models, explanations) are existing. We add four new stages: detect, diagnose, mitigate, abstain. The reliability score combines three metrics. If R(x) < 0.5, we abstain — show the prediction but not the explanation.

**Likely Question**: "What is explanation-level abstention?"

**Answer**: Unlike prediction abstention (model refuses to predict), explanation abstention means the model predicts but refuses to explain. We show the risk score with a LOW-reliability warning instead of a feature ranking. This prevents clinicians from trusting unstable explanations.

---

### Slide 8: Reliability Layer — Detect

**Content**: Table with metrics:
- **Stability**: 10 retrains, Spearman + Jaccard, flag if Jaccard < 0.6
- **Faithfulness**: SHAP vs IG top-K overlap, flag on strong disagreement
- **Robustness**: ±5% input perturbation, cosine similarity, flag if cosine < 0.8
- **Temporal stability** (LSTM): important time steps at 6h, 8h, 10h, flag on window shifts
- **Independent failure labels**: retraining instability, perturbation sensitivity, missingness sensitivity, explainer disagreement

**What to say**: Four detection metrics. Stability: train 10 times with different seeds, check if SHAP rankings stay the same. Faithfulness: compare SHAP with another explainer. Robustness: perturb inputs slightly, check if explanations change. Temporal: for LSTM, check if important time steps shift. Failure labels are defined before the detector is built.

**Likely Question**: "What is Jaccard overlap?"

**Answer**: A measure of set similarity. If SHAP says top-5 features are {A, B, C, D, E} in one training and {A, B, X, Y, Z} in another, Jaccard = |intersection| / |union| = 2/8 = 0.25, which is below 0.6, so we flag it as unstable.

---

### Slide 9: Diagnosis, Mitigate and Abstain

**Content**: Table with failure modes:
- Feature correlation → pairwise correlation → Latent SHAP
- Missingness sensitivity → compare imputed vs dropped → informative missingness modeling
- Model instability → retraining variance → ensemble attribution
- Perturbation sensitivity → explanation change under small input changes → ReCalX-style calibration
- Temporal instability → important time window shifts → DeltaSHAP
- **Stage 4 — Abstain**: If R(x) < 0.5, show risk score with LOW-reliability warning

**What to say**: Once we detect a failure, we diagnose the cause and apply a targeted fix. For example, if the failure is due to correlated features, we use Latent SHAP. If it is due to missingness, we use informative missingness modeling. If reliability remains below threshold after mitigation, we abstain.

**Likely Question**: "Why not just fix all explanations?"

**Answer**: Attribution Impossibility proves perfect reliability is impossible under collinearity. So the goal is detection and mitigation, not elimination. When mitigation fails, abstention is the safe option.

---

### Slide 10: Applications / Use Cases

**Content**:
1. ICU decision support — reliability label beside every risk score
2. Model auditing — catch explanation drift over time
3. XAI method comparison — evaluate explanation reliability on clinical time-series
4. Regulatory support — quantitative reliability evidence for AI medical devices
5. Reusable pattern — detect, diagnose, mitigate, abstain for any clinical prediction task with correlated features and missing data

**What to say**: Five applications. The first is direct clinical use. The second is governance. The third is research. The fourth is regulatory. The fifth is generalization beyond sepsis.

**Likely Question**: "How is this different from existing clinical decision support?"

**Answer**: Existing systems show predictions and explanations. Ours adds a reliability label. If the explanation is unreliable, the system says so, which prevents misplaced trust.

---

### Slide 11: Expected Deliverables

**Content**:
- **Capstone-I**: Literature review, dataset preparation, baselines, explanation pipeline, feasibility study
- **Capstone-II**: Reliability metrics, failure labels, detector, diagnosis/mitigation, LLD
- **Capstone-III**: Complete implementation, evaluation, statistical analysis, paper, reproducibility
- Milestones: Baselines running, working detector, end-to-end framework

**What to say**: Three phases. Phase I establishes foundations. Phase II builds the core novelty (reliability layer). Phase III evaluates and writes.

**Likely Question**: "What is the Low-Level Design (LLD)?"

**Answer**: Detailed technical specification: data schemas, module interfaces, algorithm pseudocode, metric formulas, and experimental protocol. It ensures the implementation is reproducible.

---

### Slide 12: Timeline (Gantt)

**Content**: Gantt chart with tasks: Problem statement, literature review, dataset preparation, prediction baselines, explanation pipeline, reliability metrics, failure labels, detector construction, mitigation strategies, full experiments, report/publication. Color-coded: blue (foundations), green (core novelty), orange (evaluation and paper).

**What to say**: 12-week timeline across three phases. Blue bars are foundations. Green bars are the core novelty. Orange bars are evaluation and writing. Be honest: it is a prototype schedule, not a guarantee.

**Likely Question**: "Is 12 weeks realistic?"

**Answer**: For a prototype, yes. Phase I is 4 weeks, Phase II is 4 weeks, Phase III is 4 weeks. If needed, Phase II can extend. The goal is a working framework and paper draft, not a production system.

---

### Slide 13: Team Roles

**Content**: Table with four members:
- Kiran BG — Prediction models and comparison (25%)
- Mohammad Anan — Explanations and reliability metrics (25%)
- Ibrahim Khaleel — Failure labels and detector (25%)
- Nikhil KR — Diagnosis, mitigation, integration (25%)
- Shared: literature review, own paper section, joint evaluation, review presentations, final demo

**What to say**: Each member owns a vertical slice. Everyone shares literature review and paper writing. 25% each.

**Likely Question**: "How do you coordinate?"

**Answer**: Weekly sync, shared repository, joint evaluation sessions. Each member has clear phase-wise responsibilities.

---

### Slide 14: Any Other Information — Data and Evaluation

**Content**:
- **Datasets**: PhysioNet/CinC 2019 (primary), MIMIC-IV (external validation if feasible), Reliability annotations (ours)
- **Prediction metrics**: AUROC, AUPRC, sensitivity, specificity, calibration
- **Reliability detection metrics**: Precision, recall, F1, AUROC for failure detection
- **Protocol**: Patient-level splits, training-only preprocessing, no future-data leakage

**What to say**: Three datasets. Primary is PhysioNet. External is MIMIC-IV if feasible. Custom annotations for reliability. Two sets of metrics: prediction and reliability detection.

**Likely Question**: "What are independent failure labels?"

**Answer**: Labels defined before the detector is built, using criteria separate from the detection method. Examples: retraining instability (top-3 features change across seeds), perturbation sensitivity, missingness sensitivity, explainer disagreement. This avoids circular evaluation.

---

### Slide 15: Thank You

**Content**: "Predict sepsis early, check whether the explanation can be trusted, and say so when it cannot."

**What to say**: Closing statement.

---

## 3. Likely Questions & Answers

### Q1: What is sepsis?

**Answer**: Sepsis is a life-threatening organ dysfunction caused by the body's dysregulated response to infection. It affects millions annually, and mortality increases every hour treatment is delayed. Early prediction 6–12 hours before onset enables timely intervention.

---

### Q2: What is SHAP?

**Answer**: SHAP (SHapley Additive exPlanations) is a game-theoretic method for explaining model predictions. It assigns each feature a contribution value based on Shapley values from cooperative game theory. It is the default explanation method in clinical AI.

---

### Q3: What is the Attribution Impossibility?

**Answer**: A 2026 theoretical result proving that when features are correlated, no feature ranking method can simultaneously be faithful, stable, and complete. Since clinical variables are inherently correlated, this means SHAP explanations in sepsis are mathematically guaranteed to be unstable under retraining.

---

### Q4: Why is SHAP reliability a problem?

**Answer**: Clinical variables are correlated (heart rate and respiratory rate, BUN and creatinine), and labs have 85–95% missingness. Under these conditions, SHAP rankings change across retraining, are sensitive to which features are measured, and may not reflect the model's actual behavior. Clinicians receive these explanations with no reliability indicator.

---

### Q5: What is stability?

**Answer**: Stability measures whether SHAP rankings stay consistent across model retraining. We train the same model 10 times with different random seeds, generate SHAP for the same patient, and compute Spearman rank correlation and top-k Jaccard overlap. If Jaccard < 0.6, the explanation is flagged as unstable.

---

### Q6: What is faithfulness?

**Answer**: Faithfulness measures whether SHAP rankings reflect the model's actual decision logic. We compare SHAP with another explainer (e.g., Integrated Gradients). If they disagree strongly, one or both may be unreliable.

---

### Q7: What is robustness?

**Answer**: Robustness measures whether SHAP rankings stay stable under small, clinically plausible input changes. We perturb one feature at a time (±5% within clinical range), regenerate SHAP, and compute cosine similarity. If cosine < 0.8, the explanation is flagged as sensitive.

---

### Q8: What is explanation-level abstention?

**Answer**: Unlike prediction abstention (model refuses to predict), explanation abstention means the model predicts but refuses to explain. We show the risk score with a LOW-reliability warning instead of a feature ranking. This prevents clinicians from trusting unstable explanations.

---

### Q9: Why not just use a better model?

**Answer**: Better prediction does not solve explanation reliability. Even a perfect model can have unstable explanations under feature correlation. The problem is in the explanation method, not the model. Our reliability layer is model-agnostic.

---

### Q10: How is this different from Nair et al. (2026)?

**Answer**: Nair et al. proposed an interpretable sepsis prediction framework using SHAP-enhanced attention. They use SHAP as a tool and assume it is reliable. We evaluate SHAP itself — detecting, diagnosing, and mitigating unreliable explanations. They are part of the gap we address.

---

### Q11: What is the primary dataset?

**Answer**: PhysioNet/CinC Challenge 2019 — 40,336 adult ICU patients from two US hospitals, 40 clinical variables (8 vital signs + 26 labs + 6 demographics), hourly time-series, Sepsis-3 labels, 85–95% lab missingness.

---

### Q12: What models are you comparing?

**Answer**: XGBoost and Random Forest (ensemble) and LSTM (deep learning), all under identical preprocessing, patient-level splits, and evaluation protocols.

---

### Q13: What explainers are you using?

**Answer**: TreeSHAP for tree models, Kernel SHAP and Integrated Gradients for LSTM. TimeSHAP for temporal comparison.

---

### Q14: What is the reliability score formula?

**Answer**: R(x) = w_s·Stability(x) + w_f·Faithfulness(x) + w_r·Robustness(x). Weights start at 1/3 each and are tuned on the validation set. If R(x) < 0.5, we abstain.

---

### Q15: How do you avoid circular evaluation?

**Answer**: We define independent failure labels before building the detector. Examples: retraining instability (top-3 features change across seeds), perturbation sensitivity (explanation changes under ±5% input noise), missingness sensitivity, explainer disagreement. The detector is evaluated against these pre-defined labels.

---

### Q16: What is the timeline?

**Answer**: 12 weeks, three phases. Phase I (weeks 1–4): foundations. Phase II (weeks 5–8): core novelty. Phase III (weeks 9–12): evaluation and paper.

---

### Q17: What are the deliverables?

**Answer**: Literature review, dataset preparation, baselines, explanation pipeline, reliability metrics, failure labels, detector, diagnosis/mitigation, evaluation, paper, reproducibility materials.

---

### Q18: What is the novelty?

**Answer**: First systematic reliability evaluation framework for SHAP explanations in early sepsis prediction, including detection, diagnosis, mitigation, and explanation-level abstention.

---

### Q19: What are the limitations?

**Answer**: Single dataset (PhysioNet 2019), no external validation yet, reliability thresholds are provisional, no clinical expert review planned. MIMIC-IV external validation if feasible.

---

### Q20: What is the next step?

**Answer**: Complete literature review and novelty matrix, prepare dataset, implement prediction baselines (XGBoost, RF, LSTM), and build the explanation pipeline.

---

## 4. Key Concepts Explained Simply

| Concept | Simple Explanation |
|---------|-------------------|
| **Sepsis** | Life-threatening organ dysfunction from infection; time-critical |
| **SHAP** | Method to explain which features drove a model's prediction |
| **Stability** | Does the explanation stay the same when you retrain? |
| **Faithfulness** | Does the explanation reflect what the model actually does? |
| **Robustness** | Does the explanation stay stable when inputs change slightly? |
| **Abstention** | Refusing to give an explanation when it cannot be trusted |
| **Attribution Impossibility** | Under feature correlation, no ranking can be faithful, stable, and complete at once |
| **Independent failure labels** | Defining what counts as a "bad explanation" before building the detector |
| **Jaccard overlap** | Measure of set similarity; used for top-k feature comparison |
| **Spearman correlation** | Measure of rank similarity; used for feature importance ordering |
| **Cosine similarity** | Measure of vector similarity; used for attribution vector comparison |
| **MoRF** | Most Relevant First — perturbation strategy for faithfulness |
| **Latent SHAP** | SHAP variant that handles correlated features |
| **ReCalX** | Calibration method for perturbation-sensitive explanations |
| **DeltaSHAP** | Explains changes between time steps, not absolute values |

---

## 5. Key Papers & Links

### SHAP Failure Theory (Free/Open Access)

| Paper | Link |
|-------|------|
| Huang & Marques-Silva (2024). On the failings of Shapley values for explainability | https://arxiv.org/abs/2307.14784 |
| Sundararajan & Najmi (ICML 2020). The Many Shapley Values for Model Explanation | https://arxiv.org/abs/1908.08474 |
| Kumar et al. (ICML 2020). Problems with Shapley-value-based explanations | https://arxiv.org/abs/2002.11097 |
| Aas et al. (2021). Explaining individual predictions when features are dependent | https://arxiv.org/abs/1903.10464 |
| Zhao et al. (2024). Error Analysis of Shapley Value-Based Model Explanations | https://arxiv.org/abs/2404.13522 |
| Hu et al. (2024). Manifold-based Shapley explanations | https://arxiv.org/abs/2410.04799 |
| UbiQTree (2025). Uncertainty quantification in XAI with tree ensembles | https://arxiv.org/abs/2502.07719 |

### Sepsis Prediction Baselines

| Paper | Link |
|-------|------|
| Khorram & Kouchaki (2026). LSTM-GAT for PhysioNet 2019 | https://informatics.bmj.com/content/bmjhci/33/1/e101762 |
| MIMIC-Sepsis Benchmark (2025) | https://arxiv.org/abs/2505.17435 |
| Nair et al. (2026). Interpretable and Uncertainty-Aware Deep Learning Framework | https://doi.org/10.3390/make8050129 |

### Tools & Datasets

| Resource | Link |
|----------|------|
| SHAP library | https://github.com/shap/shap |
| TimeSHAP | https://github.com/feedzai/timeshap |
| Latent SHAP | https://github.com/Teriri1999/Latent-SHAP |
| UbiQTree | https://github.com/dubeyakshat07/UBiQTree |
| Captum | https://captum.ai/ |
| PhysioNet 2019 | https://physionet.org/content/challenge-2019/1.0.0/ |
| MIMIC-IV | https://physionet.org/content/mimiciv/ |

---

## 6. Team Roles

| Member | Role | Phase I | Phase II | Phase III |
|--------|------|---------|----------|-----------|
| **Kiran BG** | Prediction models and comparison | Study baselines; define evaluation protocol | Implement XGBoost, RF, LSTM; run unified comparison | Analyse performance; ablations; documentation |
| **Mohammad Anan** | Explanations and reliability metrics | Study SHAP failure modes; define metrics | Implement SHAP/IG pipelines; compute metrics | Tune metrics; analyse failure patterns |
| **Ibrahim Khaleel** | Failure labels and detector | Define independent failure labels; design detection strategy | Generate failure annotations; build and evaluate detector | Evaluate detection across scenarios |
| **Nikhil KR** | Diagnosis, mitigation, integration | Study diagnosis and mitigation | Implement diagnosis, mitigation, abstention | Integrate modules; end-to-end testing |

**Shared by all four**: Literature review, own section of the paper, joint evaluation, review presentations, final demo.

---

## 7. One-Page Cheat Sheet

| Question | Answer |
|----------|--------|
| What is sepsis? | Life-threatening organ dysfunction from infection; time-critical |
| What is the problem? | SHAP explanations in sepsis prediction are never checked for reliability |
| What is the solution? | A reliability layer: detect, diagnose, mitigate, abstain |
| What is the novelty? | First explanation-level abstention framework for sepsis XAI |
| What dataset? | PhysioNet/CinC 2019 (40,336 patients) |
| What models? | XGBoost, Random Forest, LSTM |
| What explainers? | TreeSHAP, Kernel SHAP, Integrated Gradients |
| What metrics? | Stability (Spearman, Jaccard), Faithfulness (explainer agreement), Robustness (cosine) |
| What is abstention? | Refusing to show an explanation when reliability is too low |
| What is the score? | R(x) = w_s·Stability + w_f·Faithfulness + w_r·Robustness |
| What is the threshold? | R(x) < 0.5 triggers abstention |
| What is the timeline? | 12 weeks, three phases |
| What are the deliverables? | Comparison report, reliability framework, code, paper |
| What is the gap? | No study evaluates SHAP reliability for sepsis |
| Why is it novel? | First to detect, diagnose, mitigate, and abstain for sepsis XAI |

---

**End of Document**
