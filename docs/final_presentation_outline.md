# Final Presentation Outline

Target length: 10 minutes plus Q&A.

## Slide 1: Title

Predicting Student Performance for Early Intervention  
Dan Paquin, CIND820

## Slide 2: Problem

Educational institutions need to identify at-risk students early enough for intervention. The key challenge is balancing prediction accuracy against remaining time to help.

## Slide 3: Research Questions

1. Do modern models improve on Cortez & Silva's benchmark?
2. How early can risk be predicted?
3. Which features explain risk?
4. Do student archetypes support differentiated interventions?

## Slide 4: Dataset

UCI Student Performance Dataset:

- 1,044 students
- 395 mathematics, 649 Portuguese
- 33 features
- Target: pass/fail and final grade G3

Use `grade_distribution_by_subject.png` or `pass_rate_by_subject.png`.

## Slide 5: Methodology

Show temporal feature sets:

- T0: no current grades
- T1: after G1
- T2: after G2

Explain that this prevents overstating early-warning performance.

## Slide 6: Classification Results

Use `classification_accuracy_temporal.png`.

Key point: T0 reaches 0.788 accuracy; T1 reaches 0.875; T2 reaches 0.916.

## Slide 7: Regression Results

Use `regression_rmse_temporal.png`.

Key point: best RMSE is 1.438, improving on the Cortez & Silva RMSE ~2.0 benchmark.

## Slide 8: Explainability

Use `top_permutation_importance.png`.

Key point: G2 and G1 dominate once available; absences and behavioral factors remain useful for intervention framing.

## Slide 9: Student Archetypes

Use `cluster_pass_rates.png`.

Key point: one cluster is clearly high risk, with 41.7% pass rate and high prior failures.

## Slide 10: Interpretation

Enrollment-time prediction is close but not strong enough for high-confidence intervention. First-period grade is the best practical point for action.

## Slide 11: Limitations and Ethics

Mention dataset size, Portuguese context, self-reported variables, weak cluster separation, privacy, bias, and avoiding punitive labels.

## Slide 12: Conclusion

Modern ML improves regression and supports an intervention framework, but the main value is timing and interpretation rather than chasing marginal accuracy gains.

