# Final Report Results Draft

## Research Questions and Contributions

This project extends Cortez and Silva's 2008 student performance prediction benchmark by testing whether modern machine learning can move from accurate prediction toward actionable early intervention. The main contribution is an integrated pipeline that compares algorithms across temporal feature sets, quantifies the accuracy-timing tradeoff, produces educator-facing explanations, and profiles student archetypes for differentiated support.

## Methodology Summary

The analysis combines the UCI mathematics and Portuguese student performance datasets into a 1,044-row dataset. Classification predicts whether a student passes with `G3 >= 10`; regression predicts final grade `G3`.

Three feature sets represent different intervention moments:

- **T0:** information available before current-year grades.
- **T1:** T0 plus absences and first-period grade G1.
- **T2:** T1 plus second-period grade G2.

This design separates true early-warning prediction from later, higher-accuracy prediction that leaves less time for intervention.

## Main Findings

### RQ1: Algorithm Benchmarking

The best classification result was Hist Gradient Boosting on T2, with 0.916 accuracy, 0.947 F1, and 0.969 ROC-AUC. This does not exceed the original Cortez & Silva 93% classification benchmark, although it remains a strong result and uses a combined-subject setup.

The best regression result was Hist Gradient Boosting on T2, with RMSE 1.438, MAE 0.880, and R2 0.855. This improves meaningfully on the Cortez & Silva RMSE ~2.0 benchmark.

Interpretation: modern methods provide stronger regression performance and strong classification discrimination, but the classification accuracy improvement claim should be framed cautiously.

### RQ2: Early Warning Timing

The best T0 no-current-grade model was Random Forest, with 0.788 accuracy and 0.794 ROC-AUC. This is close to, but below, the proposed 80% action threshold.

The first feature set to exceed 80% accuracy was T1, after the first-period grade became available. Hist Gradient Boosting achieved 0.875 accuracy at this point.

Interpretation: on this dataset, reliable early warning is more realistic after the first formal grade than at enrollment. This result aligns with the literature showing that demographics and social features alone often provide limited predictive signal, while early academic performance substantially improves prediction.

### RQ3: Explainability

Permutation importance showed G2 as the dominant predictor in the best T2 model, followed by G1, absences, subject, and social/behavioral factors. This confirms that academic trajectory drives the strongest predictions once grades are available.

Educator-facing interpretation:

- Low G1 or G2 should trigger immediate academic support.
- Absences remain useful as a modifiable behavior signal.
- Prior failures and low T0 performance identify students who may need proactive monitoring even before current grades arrive.

### RQ4: Student Archetypes

K-means clustering on T0 features identified four archetype clusters. The clearest high-risk group contained 108 students, with a 41.7% pass rate, mean final grade 7.82, and mean prior failures 1.81.

Practical intervention mapping:

- High-risk prior-failure cluster: intensive academic advising, tutoring, and frequent monitoring.
- Lower-risk higher-pass clusters: lighter-touch progress checks and enrichment.
- Social/behavioral risk cluster with high alcohol/social activity indicators: engagement and attendance-oriented support.

The cluster validation scores were low, so archetypes should be treated as exploratory rather than definitive segmentation.

## Effectiveness, Efficiency, and Stability

Effectiveness was measured with accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, and R2. The strongest predictive performance occurs at T2, while T1 gives the best practical balance between accuracy and remaining intervention time.

Efficiency was measured through fit time. All models completed quickly on the 1,044-row dataset, with logistic/linear models fastest and tree ensembles still practical.

Stability was assessed through three-fold cross-validation. Additional repeated cross-validation could be added for the final submission if time permits.

## Limitations

- The UCI dataset comes from two Portuguese secondary schools, limiting generalizability to Ontario or broader Canadian education contexts.
- The dataset is small by modern machine learning standards.
- Some social and behavioral variables are self-reported.
- Absences represent accumulated current-year data, so they must be handled carefully in early-warning claims.
- The current explainability layer uses permutation importance; SHAP would provide richer local explanations if installed and validated.
- Cluster separation is weak, so archetype findings should be presented as exploratory.

## Ethical Considerations

Student risk prediction can create stigma if labels are used punitively or without context. Predictions should support human decision-making, not replace educator judgment. Demographic and family-background features may encode structural inequities, so any deployment would require bias review, transparency, privacy controls, and intervention-focused use.

## Recommendations

- Use first-period grade G1 as the practical trigger point for confident intervention.
- Use T0 models for monitoring, not high-stakes intervention decisions.
- Prioritize academic support for students with prior failures and weak first-period performance.
- Use model explanations to guide support conversations, not as deterministic labels.
- Extend future work with richer LMS engagement data, SHAP explanations, and external validation on a larger local dataset.

