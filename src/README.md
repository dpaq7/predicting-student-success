# Source Code Structure

This directory contains all the source code for the Predicting Student Success project.

## Directory Structure

```
src/
├── data/              # Data processing and validation modules
├── model/             # Model training, evaluation, and validation
├── inference/         # Model prediction and serving
└── pipelines/         # End-to-end pipeline orchestration
    ├── feature_pipeline/    # Transforms raw data into features
    ├── training_pipeline/   # Transforms features into trained models
    └── inference_pipeline/  # Makes predictions with trained models
```

## Module Descriptions

### `data/`
Contains modules for:
- Data extraction from various sources
- Data validation and quality checks
- Data preprocessing and cleaning
- Feature engineering transformations

### `model/`
Contains modules for:
- Model architecture definitions
- Training procedures
- Hyperparameter tuning
- Model evaluation metrics
- Model selection and comparison

### `inference/`
Contains modules for:
- Model loading and caching
- Batch and real-time prediction
- Result post-processing
- Performance monitoring

### `pipelines/`
Contains orchestration code for:
- **Feature Pipeline**: Processes raw data into model-ready features
- **Training Pipeline**: Trains and evaluates models
- **Inference Pipeline**: Generates predictions for new data

## Usage Examples

### Data Processing
```python
from src.data import load_raw_data, preprocess, validate

# Load and process data
raw_data = load_raw_data("path/to/data")
validated_data = validate(raw_data)
processed_data = preprocess(validated_data)
```

### Model Training
```python
from src.model import train_model, evaluate

# Train and evaluate model
model = train_model(X_train, y_train)
metrics = evaluate(model, X_test, y_test)
```

### Making Predictions
```python
from src.inference import load_model, predict

# Load model and make predictions
model = load_model("path/to/model")
predictions = predict(model, new_data)
```

## Development Guidelines

1. **Modularity**: Keep functions small and focused on a single task
2. **Documentation**: Add docstrings to all functions and classes
3. **Type Hints**: Use type annotations for all function parameters and returns
4. **Testing**: Write unit tests for all new functionality
5. **Error Handling**: Implement proper error handling and logging

## Testing

Run tests for specific modules:
```bash
pytest tests/data/
pytest tests/model/
pytest tests/inference/
pytest tests/pipelines/
```