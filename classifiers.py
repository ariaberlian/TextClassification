from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from typing import Union, Optional, Dict, Any
import joblib
import os


class BaseClassifier(ABC):
    """Abstract base class for text classifiers"""
    
    def __init__(self):
        self.is_fitted = False
    
        self.classes_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """Fit the classifier to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseClassifier':
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier wrapper"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, 
                 random_state: int = 42, **kwargs):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionClassifier':
        """Fit the logistic regression model"""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature coefficients (importance)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.coef_[0] if len(self.model.coef_.shape) == 2 and self.model.coef_.shape[0] == 1 else self.model.coef_


class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier wrapper"""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', 
                 probability: bool = True, random_state: int = 42, **kwargs):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=probability,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit the SVM model"""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if not self.probability:
            raise ValueError("Probability estimation was not enabled during fitting")
        return self.model.predict_proba(X)


class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier wrapper"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Fit the Random Forest model"""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.feature_importances_


class NaiveBayesClassifier(BaseClassifier):
    """Naive Bayes classifier wrapper"""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.kwargs = kwargs
        
        self.model = MultinomialNB(alpha=alpha, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """Fit the Naive Bayes model"""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class DummyClassifier(BaseClassifier):
    """Dummy classifier for fine-tuning approaches that don't need separate classifiers"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DummyClassifier':
        """Dummy fit - does nothing"""
        self.is_fitted = True
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dummy predict - should not be called"""
        raise NotImplementedError("DummyClassifier should not be used for prediction")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dummy predict_proba - should not be called"""
        raise NotImplementedError("DummyClassifier should not be used for prediction")


class ClassifierFactory:
    """Factory class to create different types of classifiers"""
    
    @staticmethod
    def create_classifier(classifier_type: str, **kwargs) -> BaseClassifier:
        """Create a classifier based on type"""
        classifier_type = classifier_type.lower()
        
        if classifier_type == "logistic_regression" or classifier_type == "lr":
            return LogisticRegressionClassifier(**kwargs)
        elif classifier_type == "svm":
            return SVMClassifier(**kwargs)
        elif classifier_type == "random_forest" or classifier_type == "rf":
            return RandomForestClassifier(**kwargs)
        elif classifier_type == "naive_bayes" or classifier_type == "nb":
            return NaiveBayesClassifier(**kwargs)
        elif classifier_type == "dummy":
            return DummyClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    @staticmethod
    def get_available_classifiers() -> Dict[str, str]:
        """Get dictionary of available classifier types and their descriptions"""
        return {
            "logistic_regression": "Logistic Regression (also 'lr')",
            "svm": "Support Vector Machine",
            "random_forest": "Random Forest (also 'rf')",
            "naive_bayes": "Naive Bayes (also 'nb')"
        }