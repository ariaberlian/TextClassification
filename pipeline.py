import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import time
import warnings
import joblib
warnings.filterwarnings('ignore')


from vectorizers import VectorizerFactory
from classifiers import ClassifierFactory


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
            classifier_type: Type of classifier ('logistic_regression', 'svm', 'naive_bayes')
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
    
    def _is_transformer_vectorizer(self) -> bool:
        """Check if the vectorizer is transformer-based"""
        return self.vectorizer_type.lower() in ['indobert', 'indobert_finetune']
    
    def _supports_huggingface_save(self) -> bool:
        """Check if the vectorizer supports HuggingFace save format"""
        return (hasattr(self.vectorizer, 'save_pretrained') and 
                hasattr(self.vectorizer, 'from_pretrained'))
    
    def _get_model_save_strategy(self) -> str:
        """Determine the save strategy based on model type"""
        if self._is_transformer_vectorizer() and self._supports_huggingface_save():
            return 'huggingface'
        else:
            return 'joblib'
    
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
        
        # Generate unique model filename based on configuration and data hash
        # Create a hash of the entire configuration to avoid long filenames
        config_str = f"{self.vectorizer_type}_{self.classifier_type}_{str(self.vectorizer_params)}_{str(self.classifier_params)}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]  # 12 chars for config
        data_hash = hashlib.md5(str(X_train[:100] + y_train[:100]).encode()).hexdigest()[:8]  # 8 chars for data
        
        # Create short, descriptive filename
        model_filename = f"model_{self.vectorizer_type}_{self.classifier_type}_{config_hash[:8]}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        
        # Try to load existing model if not forcing retrain
        hf_model_dir = model_path.replace('.pkl', '_hf')
        
        if not force_retrain and (os.path.exists(model_path) or os.path.exists(hf_model_dir)):
            try:
                # Check for HuggingFace format first
                if os.path.exists(hf_model_dir):
                    print(f"Loading existing HuggingFace model from {hf_model_dir}...")
                    
                    if self.vectorizer_type == 'indobert_finetune':
                        # Load fine-tuned model directly
                        from vectorizers import IndoBERTFineTuneVectorizer
                        self.vectorizer = IndoBERTFineTuneVectorizer.from_pretrained(hf_model_dir)
                        self.training_time = getattr(self.vectorizer, 'training_time', None)
                        
                    else:
                        # Load hybrid model (IndoBERT + classifier)
                        vectorizer_dir = hf_model_dir + '_vectorizer'
                        classifier_path = hf_model_dir + '_classifier.pkl'
                        metadata_path = hf_model_dir + '_metadata.json'
                        
                        # Load vectorizer
                        if self.vectorizer_type == 'indobert':
                            from vectorizers import IndoBERTVectorizer
                            self.vectorizer = IndoBERTVectorizer.from_pretrained(vectorizer_dir)
                        
                        # Load classifier
                        self.classifier = joblib.load(classifier_path)
                        
                        # Load metadata
                        if os.path.exists(metadata_path):
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            self.training_time = metadata.get('training_time')
                            self.vectorizer_time = metadata.get('vectorizer_time')
                            self.classifier_time = metadata.get('classifier_time')
                            self.classes_ = np.array(metadata['classes']) if metadata.get('classes') else None
                    
                    self.is_fitted = True
                    print(f"HuggingFace model loaded successfully!")
                    if self.training_time:
                        print(f"   Original training time: {self.training_time:.2f}s")
                    return self
                
                # Fallback to joblib format
                elif os.path.exists(model_path):
                    print(f"Loading existing joblib model from {model_path}...")
                    loaded_pipeline = joblib.load(model_path)
                    
                    # Copy loaded attributes to current instance
                    self.vectorizer = loaded_pipeline.vectorizer
                    self.classifier = loaded_pipeline.classifier
                    self.training_time = loaded_pipeline.training_time
                    self.vectorizer_time = getattr(loaded_pipeline, 'vectorizer_time', None)
                    self.classifier_time = getattr(loaded_pipeline, 'classifier_time', None)
                    self.is_fitted = loaded_pipeline.is_fitted
                    self.classes_ = loaded_pipeline.classes_
                    
                    print(f"Joblib model loaded successfully! (Original training time: {self.training_time:.2f}s)")
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
        
        # Auto-save the trained model using appropriate strategy
        if auto_save:
            try:
                save_strategy = self._get_model_save_strategy()
                if save_strategy == 'huggingface':
                    # Use HuggingFace format for transformer models
                    save_dir = model_path.replace('.pkl', '_hf')
                    
                    if self.vectorizer_type == 'indobert_finetune':
                        # Fine-tuned model handles everything
                        self.vectorizer.save_pretrained(save_dir)
                    else:
                        # IndoBERT + classifier: save both components in one folder
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Save vectorizer (HuggingFace model) in the main directory
                        self.vectorizer.save_pretrained(save_dir)
                        
                        # Save classifier in the same directory
                        classifier_path = os.path.join(save_dir, 'classifier.pkl')
                        joblib.dump(self.classifier, classifier_path)
                        
                        # Save pipeline metadata in the same directory
                        import json
                        metadata = {
                            'vectorizer_type': self.vectorizer_type,
                            'classifier_type': self.classifier_type,
                            'vectorizer_params': self.vectorizer_params,
                            'classifier_params': self.classifier_params,
                            'training_time': self.training_time,
                            'vectorizer_time': self.vectorizer_time,
                            'classifier_time': self.classifier_time,
                            'classes': [int(c) for c in self.classes_] if self.classes_ is not None else None,
                            'save_format': 'hybrid_huggingface_unified'
                        }
                        metadata_path = os.path.join(save_dir, 'pipeline_metadata.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    print(f"Model saved in HuggingFace format to {save_dir}")
                else:
                    # Use joblib for traditional ML models
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
        """Save the entire pipeline using appropriate format"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        save_strategy = self._get_model_save_strategy()
        
        if save_strategy == 'huggingface':
            # Use HuggingFace format for transformer models
            if filepath.endswith('.pkl'):
                save_dir = filepath.replace('.pkl', '_hf')
            else:
                save_dir = filepath + '_hf'
            
            if self.vectorizer_type == 'indobert_finetune':
                # Fine-tuned model handles everything
                self.vectorizer.save_pretrained(save_dir)
            else:
                # IndoBERT + classifier: save both components in one folder
                os.makedirs(save_dir, exist_ok=True)
                
                # Save vectorizer (HuggingFace model) in the main directory
                self.vectorizer.save_pretrained(save_dir)
                
                # Save classifier in the same directory
                classifier_path = os.path.join(save_dir, 'classifier.pkl')
                joblib.dump(self.classifier, classifier_path)
                
                # Save pipeline metadata in the same directory
                import json
                metadata = {
                    'vectorizer_type': self.vectorizer_type,
                    'classifier_type': self.classifier_type,
                    'vectorizer_params': self.vectorizer_params,
                    'classifier_params': self.classifier_params,
                    'training_time': self.training_time,
                    'vectorizer_time': self.vectorizer_time,
                    'classifier_time': self.classifier_time,
                    'classes': [int(c) for c in self.classes_] if self.classes_ is not None else None,
                    'save_format': 'hybrid_huggingface_unified'
                }
                metadata_path = os.path.join(save_dir, 'pipeline_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"Pipeline saved in HuggingFace format to {save_dir}")
        else:
            # Use joblib for traditional ML models
            joblib.dump(self, filepath)
            print(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'TextClassificationPipeline':
        """Load a saved pipeline from either HuggingFace or joblib format"""
        import os
        
        # Determine if this is already a HuggingFace directory or needs conversion
        if os.path.isdir(filepath):
            # It's already a directory, check if it's a HuggingFace model
            config_files = ['config.json', 'vectorizer_config.json', 'pytorch_model.bin', 'model.safetensors']
            if any(os.path.exists(os.path.join(filepath, f)) for f in config_files):
                hf_save_dir = filepath  # Use as-is
            else:
                # Not a HuggingFace directory, treat as regular path
                hf_save_dir = filepath + '_hf' if not filepath.endswith('_hf') else filepath
        else:
            # It's a file path, generate HuggingFace directory name
            if filepath.endswith('.pkl'):
                hf_save_dir = filepath.replace('.pkl', '_hf')
            else:
                hf_save_dir = filepath + '_hf'
        
        # Try to load HuggingFace format
        if os.path.exists(hf_save_dir) and os.path.isdir(hf_save_dir):
            try:
                print(f"Loading HuggingFace pipeline from {hf_save_dir}...")
                
                # Check if it's a pure fine-tuned model (has vectorizer_config.json)
                vectorizer_config_path = os.path.join(hf_save_dir, 'vectorizer_config.json')
                
                if os.path.exists(vectorizer_config_path):
                    # Check vectorizer type to determine loading method
                    import json
                    with open(vectorizer_config_path, 'r') as f:
                        config = json.load(f)
                    
                    vectorizer_type = config.get('vectorizer_type', 'indobert_finetune')
                    
                    if vectorizer_type == 'indobert_finetune':
                        # Pure fine-tuned model
                        instance = cls(vectorizer_type=vectorizer_type, classifier_type='dummy')
                        
                        from vectorizers import IndoBERTFineTuneVectorizer
                        instance.vectorizer = IndoBERTFineTuneVectorizer.from_pretrained(hf_save_dir)
                        instance.is_fitted = True
                        
                        print(f"Fine-tuned IndoBERT pipeline loaded successfully!")
                        return instance
                    
                    elif vectorizer_type == 'indobert':
                        # This is actually a hybrid model's vectorizer directory
                        # Fall through to hybrid model loading logic below
                        pass
                
                # Check for hybrid model metadata (IndoBERT + classifier)
                # Look for metadata in various expected locations (new and old formats)
                metadata_paths = [
                    # New unified format: metadata inside the directory
                    os.path.join(hf_save_dir, 'pipeline_metadata.json'),
                    # Old format: metadata adjacent to directory
                    hf_save_dir + '_metadata.json',
                    os.path.join(os.path.dirname(hf_save_dir), os.path.basename(hf_save_dir) + '_metadata.json'),
                    # For vectorizer directories, look for metadata without _vectorizer suffix
                    hf_save_dir.replace('_vectorizer', '_metadata.json'),
                    os.path.join(os.path.dirname(hf_save_dir), os.path.basename(hf_save_dir).replace('_vectorizer', '_metadata.json'))
                ]
                
                metadata_path = None
                for path in metadata_paths:
                    if os.path.exists(path):
                        metadata_path = path
                        break
                
                if metadata_path:
                    # Hybrid model (IndoBERT + classifier)
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Create instance with original parameters
                    instance = cls(
                        vectorizer_type=metadata['vectorizer_type'],
                        classifier_type=metadata['classifier_type'],
                        vectorizer_params=metadata.get('vectorizer_params', {}),
                        classifier_params=metadata.get('classifier_params', {})
                    )
                    
                    # Load vectorizer (hf_save_dir should be the vectorizer directory)
                    if metadata['vectorizer_type'] == 'indobert':
                        from vectorizers import IndoBERTVectorizer
                        instance.vectorizer = IndoBERTVectorizer.from_pretrained(hf_save_dir)
                    
                    # Load classifier (look for classifier file in new and old formats)
                    classifier_paths = [
                        # New unified format: classifier inside the directory
                        os.path.join(hf_save_dir, 'classifier.pkl'),
                        # Old format: classifier adjacent to directory
                        hf_save_dir.replace('_vectorizer', '_classifier.pkl'),
                        os.path.join(os.path.dirname(hf_save_dir), os.path.basename(hf_save_dir).replace('_vectorizer', '_classifier.pkl')),
                        # Also try with full path replacement for old format
                        hf_save_dir.replace('_hf_vectorizer', '_hf_classifier.pkl')
                    ]
                    
                    classifier_path = None
                    for path in classifier_paths:
                        if os.path.exists(path):
                            classifier_path = path
                            break
                    
                    if classifier_path:
                        instance.classifier = joblib.load(classifier_path)
                    else:
                        raise Exception(f"Classifier file not found for hybrid model")
                    
                    # Restore metadata
                    instance.training_time = metadata.get('training_time')
                    instance.vectorizer_time = metadata.get('vectorizer_time')
                    instance.classifier_time = metadata.get('classifier_time')
                    instance.classes_ = np.array(metadata['classes']) if metadata.get('classes') else None
                    instance.is_fitted = True
                    
                    print(f"Hybrid IndoBERT pipeline loaded successfully!")
                    return instance
                
                # If we reach here, it might be a directory but not our format
                raise Exception(f"Unrecognized HuggingFace model format in {hf_save_dir}")
                
            except Exception as e:
                print(f"Failed to load HuggingFace model: {e}")
                # Don't fall through to joblib for directories
                raise e
        
        # Try joblib format for files
        if os.path.exists(filepath) and os.path.isfile(filepath):
            try:
                print(f"Loading joblib pipeline from {filepath}...")
                loaded_pipeline = joblib.load(filepath)
                print(f"Joblib pipeline loaded successfully!")
                return loaded_pipeline
                
            except Exception as e:
                print(f"Failed to load joblib model: {e}")
                raise e
        
        # Nothing found
        raise FileNotFoundError(f"Pipeline not found: {filepath} (also checked {hf_save_dir})")


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
