import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

from vectorizers import BaseVectorizer, VectorizerFactory
from classifiers import BaseClassifier, ClassifierFactory


class TextClassificationPipeline:
    """
    Modular text classification pipeline with switchable vectorizers and classifiers
    """
    
    def __init__(self, vectorizer_type: str = "tfidf", classifier_type: str = "logistic_regression",
                 vectorizer_params: Optional[Dict[str, Any]] = None,
                 classifier_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline with specified vectorizer and classifier
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf' or 'indobert')
            classifier_type: Type of classifier ('logistic_regression', 'svm', 'random_forest', 'naive_bayes')
            vectorizer_params: Parameters for the vectorizer
            classifier_params: Parameters for the classifier
        """
        self.vectorizer_type = vectorizer_type
        self.classifier_type = classifier_type
        self.vectorizer_params = vectorizer_params or {}
        self.classifier_params = classifier_params or {}
        
        # Initialize components
        self.vectorizer = VectorizerFactory.create_vectorizer(
            vectorizer_type, **self.vectorizer_params
        )
        self.classifier = ClassifierFactory.create_classifier(
            classifier_type, **self.classifier_params
        )
        
        # Training state
        self.is_fitted = False
        self.training_time = None
        self.vectorizer_time = None
        self.classifier_time = None
        self.classes_ = None
    
    def fit(self, X_train: List[str], y_train: List[Union[str, int]], 
           force_retrain: bool = False, auto_save: bool = True, 
           model_dir: str = "saved_models", verbose: bool = True) -> 'TextClassificationPipeline':
        """
        Fit the pipeline to training data with auto-save/load functionality
        
        Args:
            X_train: List of training texts
            y_train: List of training labels
            force_retrain: If True, retrain even if saved model exists
            auto_save: If True, automatically save the trained model
            model_dir: Directory to save/load models
            verbose: If True, show detailed training progress
        """
        import os
        import hashlib
        import joblib
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate unique model filename based on configuration and data
        config_str = f"{self.vectorizer_type}_{self.classifier_type}_{str(self.vectorizer_params)}_{str(self.classifier_params)}"
        data_hash = hashlib.md5(str(X_train[:100] + y_train[:100]).encode()).hexdigest()[:8]  # Hash first 100 samples
        
        # Clean filename by removing/replacing invalid characters
        model_filename = f"pipeline_{config_str}_{data_hash}.pkl"
        # Replace problematic characters for Windows filenames
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\', "'"]
        for char in invalid_chars:
            model_filename = model_filename.replace(char, '-')
        model_filename = model_filename.replace(' ', '').replace('{', '').replace('}', '')
        model_path = os.path.join(model_dir, model_filename)
        
        # Try to load existing model if not forcing retrain
        if not force_retrain and os.path.exists(model_path):
            try:
                print(f"Loading existing model from {model_path}...")
                loaded_pipeline = joblib.load(model_path)
                
                # Copy loaded attributes to current instance
                self.vectorizer = loaded_pipeline.vectorizer
                self.classifier = loaded_pipeline.classifier
                self.training_time = loaded_pipeline.training_time
                self.vectorizer_time = getattr(loaded_pipeline, 'vectorizer_time', None)
                self.classifier_time = getattr(loaded_pipeline, 'classifier_time', None)
                self.is_fitted = loaded_pipeline.is_fitted
                self.classes_ = loaded_pipeline.classes_
                
                print(f"Model loaded successfully! (Original training time: {self.training_time:.2f}s)")
                if self.vectorizer_time is not None and self.classifier_time is not None:
                    print(f"   Original vectorizer time: {self.vectorizer_time:.2f}s")
                    print(f"   Original classifier time: {self.classifier_time:.2f}s")
                return self
                
            except Exception as e:
                print(f"Failed to load model: {e}. Training new model...")
        
        # Train new model
        start_time = time.time()
        
        if verbose:
            print(f"Training new model: {self.vectorizer_type} + {self.classifier_type}")
            print(f"Training data: {len(X_train)} samples")
            print("=" * 50)
        
        # Handle fine-tuning vectorizer differently
        if self.vectorizer_type == 'indobert_finetune':
            if verbose:
                print(f"Fine-tuning IndoBERT end-to-end (no separate classifier needed)...")
            
            vectorizer_start = time.time()
            # For fine-tuning, we need to pass labels to the vectorizer
            self.vectorizer.fit(X_train, y_train)
            vectorizer_time = time.time() - vectorizer_start
            classifier_time = 0  # No separate classifier training
            
            # Store timing information
            self.vectorizer_time = vectorizer_time
            self.classifier_time = classifier_time
            
            # The fine-tuned model handles both vectorization and classification
            X_train_vectors = None  # Not needed for fine-tuning approach
            
            if verbose:
                print(f"      [OK] IndoBERT fine-tuned in {vectorizer_time:.2f}s")
        else:
            # Traditional approach: vectorizer + classifier
            if verbose:
                print(f"[1/2] Fitting {self.vectorizer_type} vectorizer...")
            vectorizer_start = time.time()
            X_train_vectors = self.vectorizer.fit_transform(X_train)
            vectorizer_time = time.time() - vectorizer_start
            if verbose:
                print(f"      [OK] Vectorizer fitted in {vectorizer_time:.2f}s")
                print(f"      [OK] Feature dimensions: {X_train_vectors.shape}")
            
            # Classifier training with progress
            if verbose:
                print(f"[2/2] Fitting {self.classifier_type} classifier...")
            classifier_start = time.time()
            
            # Add verbose training for supported classifiers
            if verbose and self.classifier_type == 'logistic_regression':
                # Enable verbose for logistic regression
                if hasattr(self.classifier.model, 'verbose'):
                    self.classifier.model.set_params(verbose=1)
            
            elif verbose and self.classifier_type == 'svm':
                # Enable verbose for SVM
                if hasattr(self.classifier.model, 'verbose'):
                    self.classifier.model.set_params(verbose=True)
            
            # Fit the classifier
            self.classifier.fit(X_train_vectors, y_train)
            classifier_time = time.time() - classifier_start
            if verbose:
                print(f"      [OK] Classifier fitted in {classifier_time:.2f}s")
            
            # Store timing information
            self.vectorizer_time = vectorizer_time
            self.classifier_time = classifier_time
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        # Handle classes detection differently for fine-tuning
        if self.vectorizer_type == 'indobert_finetune':
            # For fine-tuning, get classes from the data since vectorizer handles classification
            self.classes_ = np.unique(y_train)
        else:
            # Traditional approach: get classes from classifier
            self.classes_ = self.classifier.classes_
        
        if verbose:
            print("=" * 50)
            print(f"[DONE] Pipeline training completed!")
            print(f"   Total time: {self.training_time:.2f}s")
            print(f"   Vectorizer: {vectorizer_time:.2f}s ({vectorizer_time/self.training_time*100:.1f}%)")
            print(f"   Classifier: {classifier_time:.2f}s ({classifier_time/self.training_time*100:.1f}%)")
            print(f"   Classes detected: {len(self.classes_)} -> {list(self.classes_)}")
        
        # Auto-save the trained model
        if auto_save:
            try:
                joblib.dump(self, model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")
        
        return self
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        """
        Predict labels for test data
        
        Args:
            X_test: List of test texts
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Handle fine-tuning vectorizer differently
        if self.vectorizer_type == 'indobert_finetune':
            # Fine-tuned model handles prediction directly
            return self.vectorizer.predict(X_test)
        else:
            # Traditional approach: vectorizer + classifier
            X_test_vectors = self.vectorizer.transform(X_test)
            return self.classifier.predict(X_test_vectors)
    
    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Predict class probabilities for test data
        
        Args:
            X_test: List of test texts
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Handle fine-tuning vectorizer differently
        if self.vectorizer_type == 'indobert_finetune':
            # Fine-tuned model returns probabilities directly
            return self.vectorizer.transform(X_test)
        else:
            # Traditional approach: vectorizer + classifier
            X_test_vectors = self.vectorizer.transform(X_test)
            return self.classifier.predict_proba(X_test_vectors)
    
    def evaluate(self, X_test: List[str], y_test: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the pipeline on test data
        
        Args:
            X_test: List of test texts
            y_test: List of true test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        start_time = time.time()
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, predictions, average='macro'
        )
        
        results = {
            'accuracy': accuracy * 100,  # Convert to percentage
            'precision_weighted': precision * 100,
            'recall_weighted': recall * 100,
            'f1_weighted': f1 * 100,  # Convert to percentage
            'precision_macro': precision_macro * 100,
            'recall_macro': recall_macro * 100,
            'f1_macro': f1_macro * 100,  # Convert to percentage
            'prediction_time': prediction_time,
            'training_time': self.training_time,
            'vectorizer_type': self.vectorizer_type,
            'classifier_type': self.classifier_type,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return results
    
    def get_classification_report(self, X_test: List[str], y_test: List[Union[str, int]]) -> str:
        """Get detailed classification report"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)
    
    def get_confusion_matrix(self, X_test: List[str], y_test: List[Union[str, int]]) -> np.ndarray:
        """Get confusion matrix"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        predictions = self.predict(X_test)
        return confusion_matrix(y_test, predictions)
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the entire pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        import joblib
        joblib.dump(self, filepath)
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'TextClassificationPipeline':
        """Load a saved pipeline"""
        import joblib
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        return joblib.load(filepath)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration"""
        return {
            'vectorizer_type': self.vectorizer_type,
            'classifier_type': self.classifier_type,
            'vectorizer_params': self.vectorizer_params,
            'classifier_params': self.classifier_params,
            'is_fitted': self.is_fitted,
            'classes': list(self.classes_) if self.classes_ is not None else None,
            'training_time': self.training_time
        }


class PipelineComparison:
    """
    Class for comparing different pipeline configurations
    """
    
    def __init__(self):
        self.results = []
        self.pipelines = []
    
    def add_pipeline(self, pipeline: TextClassificationPipeline, name: str = None) -> None:
        """Add a pipeline to the comparison"""
        if name is None:
            name = f"{pipeline.vectorizer_type}_{pipeline.classifier_type}"
        
        self.pipelines.append({
            'name': name,
            'pipeline': pipeline
        })
    
    def compare_pipelines(self, X_train: List[str], y_train: List[Union[str, int]],
                         X_test: List[str], y_test: List[Union[str, int]]) -> pd.DataFrame:
        """
        Compare all added pipelines on the same dataset
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for pipeline_info in self.pipelines:
            name = pipeline_info['name']
            pipeline = pipeline_info['pipeline']
            
            print(f"\nTraining and evaluating: {name}")
            print("-" * 50)
            
            try:
                # Fit and evaluate
                pipeline.fit(X_train, y_train)
                eval_results = pipeline.evaluate(X_test, y_test)
                
                # Add pipeline name to results
                eval_results['pipeline_name'] = name
                results.append(eval_results)
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
        
        self.results = results
        
        # Convert to DataFrame for easy comparison
        df_results = pd.DataFrame(results)
        
        # Select key metrics for comparison
        comparison_cols = [
            'pipeline_name', 'accuracy', 'f1_weighted', 'f1_macro',
            'precision_weighted', 'recall_weighted',
            'training_time', 'prediction_time'
        ]
        
        return df_results[comparison_cols].round(4)
    
    def get_best_pipeline(self, metric: str = 'f1_weighted') -> Dict[str, Any]:
        """Get the best performing pipeline based on a metric"""
        if not self.results:
            raise ValueError("No comparison results available. Run compare_pipelines first.")
        
        best_idx = np.argmax([result[metric] for result in self.results])
        return self.results[best_idx]


def create_preset_pipelines() -> List[TextClassificationPipeline]:
    """
    Create a set of preset pipeline configurations for comparison
    
    Returns:
        List of configured pipelines
    """
    pipelines = []
    
    # TF-IDF + Logistic Regression (baseline)
    pipelines.append(TextClassificationPipeline(
        vectorizer_type="tfidf",
        classifier_type="logistic_regression",
        vectorizer_params={'max_features': 10000, 'ngram_range': (1, 2)},
        classifier_params={'C': 1.0, 'max_iter': 1000}
    ))
    
    # TF-IDF + SVM
    pipelines.append(TextClassificationPipeline(
        vectorizer_type="tfidf",
        classifier_type="svm",
        vectorizer_params={'max_features': 5000, 'ngram_range': (1, 2)},
        classifier_params={'C': 1.0, 'kernel': 'linear'}
    ))
    
    # TF-IDF + Random Forest
    pipelines.append(TextClassificationPipeline(
        vectorizer_type="tfidf",
        classifier_type="random_forest",
        vectorizer_params={'max_features': 8000, 'ngram_range': (1, 2)},
        classifier_params={'n_estimators': 100, 'max_depth': 10}
    ))
    
    # IndoBERT + Logistic Regression
    pipelines.append(TextClassificationPipeline(
        vectorizer_type="indobert",
        classifier_type="logistic_regression",
        vectorizer_params={'pooling_strategy': 'mean'},
        classifier_params={'C': 1.0, 'max_iter': 1000}
    ))
    
    # IndoBERT + SVM
    pipelines.append(TextClassificationPipeline(
        vectorizer_type="indobert",
        classifier_type="svm",
        vectorizer_params={'pooling_strategy': 'cls'},
        classifier_params={'C': 1.0, 'kernel': 'rbf'}
    ))
    
    return pipelines