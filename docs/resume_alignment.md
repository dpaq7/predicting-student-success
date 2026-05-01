# Resume Alignment: Predicting Student Success

## Target Roles

- Junior data scientist
- Educational data analyst
- Applied machine learning analyst
- Responsible AI / model evaluation analyst

## Skills Demonstrated

- Public dataset analysis and reproducible pipelines
- Temporal feature engineering to avoid leakage in early-warning claims
- Cross-validated classification and regression
- Model comparison and metric interpretation
- Permutation-importance explanation
- Exploratory clustering and limitation-aware interpretation
- Academic reporting and stakeholder-facing documentation

## Evidence-Based Resume Bullets

- Built a reproducible educational analytics pipeline on the 1,044-row UCI Student Performance Dataset, generating EDA figures, cross-validated classification/regression metrics, and report-ready artifacts.
- Compared temporal early-warning feature sets and found that no-current-grade prediction reached 0.788 accuracy, while the first-grade feature set reached 0.875 accuracy with histogram gradient boosting.
- Improved final-grade regression performance to RMSE 1.438 on the T2 feature set and documented limitations around generalizability, self-reported variables, and exploratory clustering.

## Interview Talking Points

- Why separating T0, T1, and T2 feature sets matters for defensible early-warning claims.
- Why T0 prediction is close to but below the 80% threshold and how that affects intervention timing.
- Why the strongest practical recommendation is to intervene after first-period grades rather than claiming reliable prediction at enrollment.
- How permutation importance supports educator-facing explanations without overclaiming SHAP-level local interpretability.
- Why the clustering output is useful for exploration but not a validated student segmentation model.

