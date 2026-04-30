# Data Directory

The raw UCI Student Performance CSV files used by this project are:

- `data/01_raw/student-mat.csv`
- `data/01_raw/student-por.csv`

The analysis pipeline writes reproducible derived files to:

- `data/03_primary/student_performance_combined.csv`
- `data/08_reporting/tables/`
- `data/08_reporting/figures/`
- `data/08_reporting/analysis_summary.md`

Run:

```bash
python -m src.pipelines.training_pipeline.run_analysis
```

Raw data should remain unchanged. Regenerate derived outputs from source code when methods change.
