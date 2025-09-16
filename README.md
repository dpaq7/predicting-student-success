# Predicting Student Success

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## 📌 Overview

A machine learning project designed to predict student academic success using various data science techniques. This project follows modern best practices for reproducible and maintainable data science workflows.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- UV package manager (will be installed if not present)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/predicting-student-success.git
cd predicting-student-success
```

2. **Set up the environment**:
```bash
make init_env
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install data science libraries**:
```bash
make install_data_libs
```

4. **Run tests to verify setup**:
```bash
make test
```

## 📁 Project Structure

```
├── .code_quality/          # Code quality configuration
│   ├── mypy.ini           # MyPy configuration
│   └── ruff.toml          # Ruff linter configuration
├── conf/                   # Configuration files
│   └── config.yaml        # Main project configuration
├── data/                   # Data directory (gitignored)
│   ├── 01_raw/            # Raw immutable data
│   ├── 02_intermediate/   # Cleaned data
│   ├── 03_primary/        # Domain model data
│   ├── 04_feature/        # Feature engineered data
│   ├── 05_model_input/    # Model training data
│   ├── 06_models/         # Trained models
│   ├── 07_model_output/   # Model predictions
│   └── 08_reporting/      # Reports and visualizations
├── docs/                   # Documentation
├── models/                 # Saved model artifacts
├── notebooks/              # Jupyter notebooks
│   ├── 1-data/            # Data extraction and cleaning
│   ├── 2-exploration/     # EDA notebooks
│   ├── 3-analysis/        # Statistical analysis
│   ├── 4-feat_eng/        # Feature engineering
│   ├── 5-models/          # Model training
│   ├── 6-interpretation/  # Model interpretation
│   ├── 7-deploy/          # Deployment preparation
│   └── 8-reports/         # Final reports
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── model/             # Model training modules
│   ├── inference/         # Prediction modules
│   └── pipelines/         # End-to-end pipelines
├── tests/                  # Unit tests
├── .gitignore             # Git ignore file
├── .pre-commit-config.yaml # Pre-commit hooks
├── Makefile               # Automation commands
├── mkdocs.yml             # Documentation config
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## 🛠️ Available Commands

Run `make` or `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make init_env` | Set up Python environment with UV |
| `make init_git` | Initialize git repository |
| `make install_data_libs` | Install data science libraries |
| `make test` | Run tests with coverage |
| `make check` | Run code quality checks |
| `make docs_view` | Build and serve documentation |
| `make notebook` | Start Jupyter notebook server |
| `make clean` | Clean generated files |

## 📊 Workflow

1. **Data Processing**: Place raw data in `data/01_raw/`
2. **Exploration**: Use notebooks in `notebooks/2-exploration/` for EDA
3. **Feature Engineering**: Create features in `notebooks/4-feat_eng/`
4. **Model Training**: Train models in `notebooks/5-models/`
5. **Production Code**: Move stable code to `src/` modules
6. **Testing**: Write tests in `tests/`
7. **Documentation**: Update docs in `docs/`

## 🧪 Testing

Run the test suite:
```bash
make test
```

Run tests for specific modules:
```bash
pytest tests/data/
pytest tests/model/
```

## 📝 Code Quality

This project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking
- **Pre-commit**: Git hooks for code quality
- **pytest**: Testing framework

Run all quality checks:
```bash
make check
```

## 📚 Documentation

View the project documentation:
```bash
make docs_view
```

Then open your browser to `http://localhost:8000`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This project structure is based on the [Data Science Project Template](https://github.com/JoseRZapata/data-science-project-template) which incorporates best practices from various sources including Cookiecutter Data Science, Kedro, and modern Python tooling.

## 📮 Contact

For questions or feedback, please open an issue on GitHub.