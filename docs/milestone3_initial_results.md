# Milestone 3: Initial Results and Code

## Cover Page

**Project:** Predicting Student Performance for Early Intervention  
**Student:** Dan Paquin  
**Student Number:** 501284328  
**Supervisor:** Dr. Babaoglu  
**Course:** CIND820 Big Data Analytics Project  

## 1. Data Analysis

The project uses the UCI Student Performance Dataset, combining the mathematics and Portuguese-language files into one analysis dataset with 1,044 student records. The mathematics file contains 395 students and the Portuguese file contains 649 students. The combined data includes 33 original attributes covering demographics, family background, school context, social behavior, study habits, prior failures, absences, and three grade periods: G1, G2, and G3.

The target for classification is pass/fail, defined as `G3 >= 10`. The regression target is final grade `G3` on the 0-20 scale.

Key descriptive findings:

- Mathematics students had a lower pass rate, 67.1%, than Portuguese students, 84.6%.
- The combined pass rate was 78.0%, indicating moderate class imbalance.
- Mean final grade was 10.42 in mathematics and 11.91 in Portuguese.
- Prior failures are strongly associated with lower final grades, supporting their use as an early risk signal.
- G1 and G2 have strong relationships with G3, so temporal feature sets are required to avoid overstating early-warning performance.

Generated figures:

- `data/08_reporting/figures/grade_distribution_by_subject.png`
- `data/08_reporting/figures/pass_rate_by_subject.png`
- `data/08_reporting/figures/numeric_correlation_heatmap.png`
- `data/08_reporting/figures/grade_by_prior_failures.png`

## 2. Data Preparation

The raw files were copied into `data/01_raw/` and left unchanged. A reproducible pipeline combines them, adds a `subject` column, and creates the binary target `pass_fail`.

Three temporal feature sets were created:

- **T0 no current grades:** demographic, family, social, school, support, study, behavior, and prior-failure features. It excludes G1, G2, G3, and current-year absences.
- **T1 after first grade:** T0 plus current absences and G1.
- **T2 after second grade:** T1 plus G2.

Categorical features are one-hot encoded. Numeric features are passed through unchanged for supervised models and standardized for k-means clustering.

## 3. Model Evaluation

Classification models:

- Logistic Regression: interpretable baseline
- Random Forest: closest benchmark to Cortez & Silva's 2008 Random Forest result
- Histogram Gradient Boosting: modern scikit-learn gradient boosting model

Regression models:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor

Models were evaluated with three-fold cross-validation. Classification metrics include accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC. Regression metrics include RMSE, MAE, and R2.

Initial results:

- Best no-current-grade model: Random Forest, 0.788 accuracy and 0.794 ROC-AUC.
- First actionable timing point: T1 after first-period grade, Hist Gradient Boosting, 0.875 accuracy.
- Best full-feature classifier: Hist Gradient Boosting on T2, 0.916 accuracy and 0.969 ROC-AUC.
- Best full-feature regressor: Hist Gradient Boosting on T2, RMSE 1.438, improving on Cortez & Silva's RMSE ~2.0 benchmark.

Generated result files:

- `data/08_reporting/tables/classification_metrics.csv`
- `data/08_reporting/tables/regression_metrics.csv`
- `data/08_reporting/figures/classification_accuracy_temporal.png`
- `data/08_reporting/figures/regression_rmse_temporal.png`

## 4. Explainability and Archetypes

Permutation importance was used as a package-light, model-agnostic explainability method. For the best T2 classification model, G2 was the dominant predictor, followed by G1, absences, subject, and selected behavioral/contextual features. This confirms that prediction becomes much stronger after interim academic performance is available.

The clustering analysis used T0 features to avoid defining archetypes from final grades. Four clusters were selected based on silhouette and Davies-Bouldin scores. The clearest high-risk archetype contained 108 students, with a 41.7% pass rate, mean G3 of 7.82, and mean prior failures of 1.81. This group is the strongest candidate for intensive intervention.

Generated files:

- `data/08_reporting/tables/permutation_importance.csv`
- `data/08_reporting/tables/cluster_metrics.csv`
- `data/08_reporting/tables/cluster_profiles.csv`
- `data/08_reporting/figures/top_permutation_importance.png`
- `data/08_reporting/figures/cluster_pass_rates.png`
- `data/08_reporting/figures/cluster_silhouette_scores.png`

## 5. Code Documentation

The source code is organized as follows:

- `src/data/student_performance.py`: raw data loading, target creation, and temporal feature definitions.
- `src/model/evaluation.py`: preprocessing, model suites, cross-validation, and permutation importance.
- `src/model/clustering.py`: k-means model selection and cluster profiling.
- `src/pipelines/training_pipeline/run_analysis.py`: end-to-end artifact generation.
- `tests/test_student_performance.py`: reproducibility and feature-leakage tests.

Run the full analysis:

```bash
python -m src.pipelines.training_pipeline.run_analysis
```

Run quality checks:

```bash
python -m ruff check src tests
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## 6. Initial Interpretation

The initial results partially support the project proposal. Models without current-year grades approach but do not cross the 80% action threshold, reaching 78.8% accuracy. This suggests that enrollment-time prediction alone may be insufficient for confident intervention decisions on this dataset. Once the first grade is available, accuracy rises to 87.5%, creating a practical intervention window. With both G1 and G2, prediction improves further, but the remaining intervention time is shorter.

The regression results are stronger: the best T2 model achieved RMSE 1.438, clearly improving on the historical RMSE ~2.0 benchmark. For classification, the best T2 result, 91.6% accuracy, is slightly below the 93% Cortez benchmark but has strong ROC-AUC. This should be interpreted carefully: the combined-subject modeling setup and cross-validation strategy differ from the original paper, and the stronger contribution may be temporal interpretation rather than raw classification accuracy improvement.

## 7. Next Steps

- Add SHAP if package compatibility allows, especially for local student-level explanations.
- Expand the final report's literature integration around the finding that early prediction without grades remains difficult.
- Convert the Markdown report material into the TMU Word/PDF template.
- Create the final PowerPoint with the generated figures and concise interpretation.

