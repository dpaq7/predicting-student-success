.PHONY: help init_env init_git install_data_libs test check docs_test docs_view clean pre-commit_update

# Self-documenting Makefile
help: ## Display this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment Setup

init_env: ## Install dependencies with uv and activate env
	@echo "📦 Setting up Python environment with UV..."
	uv venv
	@echo "✅ Virtual environment created"
	@echo "📦 Installing base dependencies..."
	uv pip install -e ".[dev,docs]"
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "🎉 Environment setup complete!"
	@echo "To activate the environment, run:"
	@echo "  source .venv/bin/activate"

init_git: ## Initialize git repository
	@echo "📦 Initializing Git repository..."
	git init
	git add .
	git commit -m "Initial commit: Project structure setup"
	@echo "✅ Git repository initialized"

install_data_libs: ## Install pandas, scikit-learn, Jupyter, seaborn
	@echo "📦 Installing data science libraries..."
	uv pip install -e ".[data]"
	@echo "✅ Data science libraries installed"

##@ Code Quality

check: ## Run code quality tools with pre-commit hooks
	@echo "🔍 Running code quality checks..."
	pre-commit run --all-files

lint: ## Run ruff linter
	@echo "🔍 Running Ruff linter..."
	uv run ruff check src tests

format: ## Format code with ruff
	@echo "✨ Formatting code with Ruff..."
	uv run ruff format src tests

type-check: ## Run mypy type checking
	@echo "🔍 Running type checks..."
	uv run mypy src

##@ Testing

test: ## Test the code with pytest and coverage
	@echo "🧪 Running tests..."
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-watch: ## Run tests in watch mode
	@echo "👀 Running tests in watch mode..."
	uv run pytest-watch

##@ Documentation

docs_test: ## Test if documentation can be built without warnings or errors
	@echo "📚 Testing documentation build..."
	uv run mkdocs build --strict

docs_view: ## Build and serve the documentation
	@echo "📚 Building and serving documentation..."
	uv run mkdocs serve

docs_build: ## Build documentation
	@echo "📚 Building documentation..."
	uv run mkdocs build

##@ Maintenance

pre-commit_update: ## Update pre-commit hooks
	@echo "🔄 Updating pre-commit hooks..."
	pre-commit autoupdate

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -rf *.egg-info 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true
	rm -rf site/ 2>/dev/null || true
	@echo "✨ Clean complete"

##@ Jupyter

notebook: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	uv run jupyter notebook

lab: ## Start JupyterLab server
	@echo "🧪 Starting JupyterLab..."
	uv run jupyter lab