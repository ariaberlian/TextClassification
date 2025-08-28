# Indonesian Text Classification with Modular Pipeline

A flexible, modular text classification system for Indonesian sentiment analysis with switchable vectorizers and classifiers.

## Features

🔧 **Modular Architecture**
- Switchable vectorizers (TF-IDF, IndoBERT)
- Multiple classifiers (Logistic Regression, SVM, Random Forest, Naive Bayes)
- Consistent API across all components

🇮🇩 **Indonesian Language Support**
- Indonesian text preprocessing with Sastrawi
- Support for IndoNLU datasets
- IndoBERT integration for contextual embeddings

⚡ **Easy Experimentation**
- Quick pipeline comparison
- Automated hyperparameter testing
- Built-in evaluation metrics

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from pipeline import TextClassificationPipeline

# Create a pipeline
pipeline = TextClassificationPipeline(
    vectorizer_type="tfidf",
    classifier_type="logistic_regression"
)

# Train and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Interactive Notebook

Open `text_classification_demo.ipynb` for a complete interactive tutorial with:
- Dataset loading and exploration
- Component comparison
- Feature analysis
- Custom pipeline building

## Architecture

### Vectorizers (`vectorizers.py`)

- **TFIDFVectorizer**: TF-IDF with Indonesian preprocessing
- **IndoBERTVectorizer**: Contextual embeddings using IndoBERT
- **BaseVectorizer**: Abstract base class for custom vectorizers

### Classifiers (`classifiers.py`)

- **LogisticRegressionClassifier**: Linear classification
- **SVMClassifier**: Support Vector Machine
- **RandomForestClassifier**: Ensemble method
- **NaiveBayesClassifier**: Probabilistic classifier

### Pipeline (`pipeline.py`)

- **TextClassificationPipeline**: Main pipeline class
- **PipelineComparison**: Compare multiple configurations
- **create_preset_pipelines()**: Pre-configured pipeline sets

### Dataset Loader (`dataset_loader.py`)

- **IndonesianSentimentLoader**: Load Indonesian sentiment datasets
- **DatasetFactory**: Create dataset loaders
- Support for IndoNLU and custom datasets

## Usage Examples

### Single Pipeline

```python
from pipeline import TextClassificationPipeline

# TF-IDF + Logistic Regression
pipeline = TextClassificationPipeline(
    vectorizer_type="tfidf",
    classifier_type="logistic_regression",
    vectorizer_params={'max_features': 10000, 'ngram_range': (1, 2)},
    classifier_params={'C': 1.0}
)

pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test)
```

### IndoBERT Pipeline

```python
# IndoBERT + Logistic Regression
bert_pipeline = TextClassificationPipeline(
    vectorizer_type="indobert",
    classifier_type="logistic_regression",
    vectorizer_params={
        'model_name': 'indolem/indobert-base-uncased',
        'pooling_strategy': 'mean'
    }
)
```

### Pipeline Comparison

```python
from pipeline import PipelineComparison, create_preset_pipelines

# Compare multiple configurations
comparison = PipelineComparison()

# Add preset pipelines
presets = create_preset_pipelines()
for i, pipeline in enumerate(presets):
    comparison.add_pipeline(pipeline, f"Preset_{i+1}")

# Run comparison
results = comparison.compare_pipelines(X_train, y_train, X_test, y_test)
best = comparison.get_best_pipeline(metric='f1_weighted')
```

### Dataset Loading

```python
from dataset_loader import DatasetFactory

# Load IndoNLU dataset
loader = DatasetFactory.create_loader("indonlu")
X_train, X_val, X_test, y_train, y_val, y_test = loader.get_train_test_split()

# Or use sample data for testing
sample_loader = DatasetFactory.create_loader("sample")
```

## Available Components

### Vectorizers
- `tfidf`: TF-IDF with Indonesian preprocessing
- `indobert`: IndoBERT contextual embeddings

### Classifiers
- `logistic_regression` (or `lr`): Logistic Regression
- `svm`: Support Vector Machine
- `random_forest` (or `rf`): Random Forest
- `naive_bayes` (or `nb`): Naive Bayes

## Configuration Options

### TF-IDF Parameters
```python
vectorizer_params = {
    'max_features': 10000,          # Maximum number of features
    'ngram_range': (1, 2),          # N-gram range
    'use_preprocessing': True,       # Use Indonesian preprocessing
}
```

### IndoBERT Parameters
```python
vectorizer_params = {
    'model_name': 'indolem/indobert-base-uncased',
    'max_length': 512,              # Maximum sequence length
    'pooling_strategy': 'mean',     # 'mean', 'cls', or 'max'
}
```

### Classifier Parameters
```python
# Logistic Regression
classifier_params = {'C': 1.0, 'max_iter': 1000}

# SVM
classifier_params = {'C': 1.0, 'kernel': 'linear'}

# Random Forest
classifier_params = {'n_estimators': 100, 'max_depth': 10}

# Naive Bayes
classifier_params = {'alpha': 1.0}
```

## Performance Tips

1. **For quick experiments**: Use TF-IDF with smaller feature counts
2. **For best accuracy**: Try IndoBERT (requires GPU for reasonable speed)
3. **For interpretability**: Use Logistic Regression or Naive Bayes
4. **For robust performance**: Try Random Forest or SVM

## File Structure

```
TextClassification/
├── requirements.txt                    # Dependencies
├── vectorizers.py                     # Vectorizer classes
├── classifiers.py                     # Classifier classes
├── pipeline.py                        # Pipeline and comparison tools
├── dataset_loader.py                  # Dataset loading utilities
├── text_classification_demo.ipynb     # Interactive tutorial
└── README.md                          # This file
```

## Extending the System

### Adding a New Vectorizer

```python
from vectorizers import BaseVectorizer

class MyCustomVectorizer(BaseVectorizer):
    def fit(self, texts):
        # Implement fitting logic
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        # Implement transformation logic
        return vectors
```

### Adding a New Classifier

```python
from classifiers import BaseClassifier

class MyCustomClassifier(BaseClassifier):
    def fit(self, X, y):
        # Implement training logic
        self.is_fitted = True
        return self
    
    def predict(self, X):
        # Implement prediction logic
        return predictions
```

## Requirements

- Python 3.7+
- scikit-learn
- pandas, numpy
- transformers (for IndoBERT)
- torch (for IndoBERT)
- sastrawi (Indonesian preprocessing)
- datasets (for IndoNLU)

## License

This project is open source. Feel free to use and modify as needed.