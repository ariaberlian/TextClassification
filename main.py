"""
Indonesian Text Classification - Configurable Main Script

A comprehensive script to run text classification with full parameter control.
Supports TF-IDF, IndoBERT frozen features, and IndoBERT fine-tuning approaches.

Example usage:
    # Basic usage
    python main.py
    
    # TF-IDF with custom parameters
    python main.py --model tfidf_lr --max-samples 1000 --tfidf-max-features 10000 --tfidf-ngram-max 3 --lr-C 0.5
    
    # TF-IDF with Neural Network
    python main.py --model tfidf_nn --nn-hidden-layers 100 50 --nn-activation relu --nn-early-stopping
    
    # IndoBERT frozen features
    python main.py --model indobert_frozen --bert-max-length 256 --bert-pooling cls
    
    # IndoBERT with Neural Network  
    python main.py --model indobert_nn --nn-hidden-layers 128 64 32 --nn-solver adam --nn-max-iter 300
    
    # IndoBERT fine-tuning with custom parameters
    python main.py --model indobert_finetune --bert-epochs 3 --bert-batch-size 16 --bert-lr 3e-5
    
    # Dataset parameters
    python main.py --dataset-subset smsa --test-size 0.25 --val-size 0.15 --random-state 123
"""

import sys
import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import Dict, Any, Optional

# Try to import required libraries
try:
    from dataset_loader import DatasetFactory
    from pipeline import TextClassificationPipeline
    print("[OK] Successfully imported custom modules")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Make sure all project files are uploaded to Colab")
    sys.exit(1)


def create_argument_parser():
    """Create comprehensive argument parser for all tunable parameters"""
    parser = argparse.ArgumentParser(
        description='Indonesian Text Classification with Full Parameter Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python main.py
  
  # TF-IDF with custom parameters
  python main.py --model tfidf_lr --tfidf-max-features 10000 --tfidf-ngram-max 3 --lr-C 0.5
  
  # TF-IDF with Neural Network
  python main.py --model tfidf_nn --nn-hidden-layers 100 50 --nn-activation relu --nn-early-stopping
  
  # IndoBERT frozen features with custom settings
  python main.py --model indobert_frozen --bert-max-length 256 --bert-pooling cls --lr-max-iter 2000
  
  # IndoBERT with Neural Network
  python main.py --model indobert_nn --nn-hidden-layers 128 64 32 --nn-solver adam --nn-max-iter 300
  
  # IndoBERT fine-tuning with custom hyperparameters
  python main.py --model indobert_finetune --bert-epochs 5 --bert-batch-size 16 --bert-lr 3e-5
  
  # Custom dataset and split parameters
  python main.py --dataset-subset smsa --test-size 0.25 --val-size 0.15 --max-samples 2000
        """
    )
    
    # Model selection
    parser.add_argument('--model', choices=['tfidf_lr', 'tfidf_svm', 'tfidf_rf', 'tfidf_nb', 'tfidf_nn',
                                           'indobert_frozen', 'indobert_finetune', 'indobert_nn'], 
                       default='tfidf_lr', help='Model type to use')
    
    # Dataset parameters
    parser.add_argument('--dataset-subset', default='smsa', help='Dataset subset to use')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to use')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0.0-1.0)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size (0.0-1.0)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    # TF-IDF Vectorizer parameters
    tfidf_group = parser.add_argument_group('TF-IDF Vectorizer Parameters')
    tfidf_group.add_argument('--tfidf-max-features', type=int, default=5000, help='Maximum number of TF-IDF features')
    tfidf_group.add_argument('--tfidf-ngram-min', type=int, default=1, help='Minimum n-gram size')
    tfidf_group.add_argument('--tfidf-ngram-max', type=int, default=2, help='Maximum n-gram size')
    tfidf_group.add_argument('--tfidf-preprocessing', action='store_true', default=True, help='Use Indonesian text preprocessing')
    tfidf_group.add_argument('--no-tfidf-preprocessing', dest='tfidf_preprocessing', action='store_false', help='Disable preprocessing')
    
    # IndoBERT Vectorizer parameters
    bert_group = parser.add_argument_group('IndoBERT Vectorizer Parameters')
    bert_group.add_argument('--bert-model', default='indolem/indobert-base-uncased', help='BERT model name')
    bert_group.add_argument('--bert-max-length', type=int, default=128, help='Maximum sequence length for BERT')
    bert_group.add_argument('--bert-pooling', choices=['mean', 'cls', 'max'], default='mean', 
                           help='Pooling strategy for BERT embeddings')
    
    # IndoBERT Fine-tuning parameters
    finetune_group = parser.add_argument_group('IndoBERT Fine-tuning Parameters')
    finetune_group.add_argument('--bert-epochs', type=int, default=2, help='Number of fine-tuning epochs')
    finetune_group.add_argument('--bert-batch-size', type=int, default=8, help='Batch size for fine-tuning')
    finetune_group.add_argument('--bert-lr', type=float, default=2e-5, help='Learning rate for fine-tuning')
    finetune_group.add_argument('--bert-warmup-steps', type=int, default=50, help='Warmup steps for scheduler')
    finetune_group.add_argument('--bert-num-labels', type=int, default=2, help='Number of classification labels')
    
    # Logistic Regression parameters
    lr_group = parser.add_argument_group('Logistic Regression Parameters')
    lr_group.add_argument('--lr-C', type=float, default=1.0, help='Regularization parameter C')
    lr_group.add_argument('--lr-max-iter', type=int, default=1000, help='Maximum iterations')
    lr_group.add_argument('--lr-solver', choices=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], 
                         default='lbfgs', help='Solver algorithm')
    
    # SVM parameters
    svm_group = parser.add_argument_group('SVM Parameters')
    svm_group.add_argument('--svm-C', type=float, default=1.0, help='SVM regularization parameter')
    svm_group.add_argument('--svm-kernel', choices=['linear', 'poly', 'rbf', 'sigmoid'], 
                          default='rbf', help='SVM kernel type')
    svm_group.add_argument('--svm-gamma', choices=['scale', 'auto'], default='scale', help='Kernel coefficient')
    
    # Random Forest parameters
    rf_group = parser.add_argument_group('Random Forest Parameters')
    rf_group.add_argument('--rf-n-estimators', type=int, default=100, help='Number of trees')
    rf_group.add_argument('--rf-max-depth', type=int, help='Maximum depth of trees')
    rf_group.add_argument('--rf-min-samples-split', type=int, default=2, help='Min samples to split node')
    rf_group.add_argument('--rf-min-samples-leaf', type=int, default=1, help='Min samples in leaf node')
    
    # Naive Bayes parameters
    nb_group = parser.add_argument_group('Naive Bayes Parameters')
    nb_group.add_argument('--nb-alpha', type=float, default=1.0, help='Additive smoothing parameter')
    
    # Neural Network parameters
    nn_group = parser.add_argument_group('Neural Network Parameters')
    nn_group.add_argument('--nn-hidden-layers', type=int, nargs='+', default=[100], 
                         help='Hidden layer sizes (e.g., --nn-hidden-layers 100 50 25)')
    nn_group.add_argument('--nn-activation', choices=['identity', 'logistic', 'tanh', 'relu'], 
                         default='relu', help='Activation function')
    nn_group.add_argument('--nn-solver', choices=['lbfgs', 'sgd', 'adam'], default='adam', 
                         help='Solver for weight optimization')
    nn_group.add_argument('--nn-alpha', type=float, default=0.0001, help='L2 penalty parameter')
    nn_group.add_argument('--nn-batch-size', type=str, default='auto', 
                         help='Batch size (use "auto" for min(200, n_samples))')
    nn_group.add_argument('--nn-learning-rate', choices=['constant', 'invscaling', 'adaptive'], 
                         default='constant', help='Learning rate schedule')
    nn_group.add_argument('--nn-learning-rate-init', type=float, default=0.001, 
                         help='Initial learning rate')
    nn_group.add_argument('--nn-max-iter', type=int, default=200, help='Maximum number of iterations')
    nn_group.add_argument('--nn-early-stopping', action='store_true', 
                         help='Enable early stopping to prevent overfitting')
    nn_group.add_argument('--nn-validation-fraction', type=float, default=0.1, 
                         help='Fraction of training data for early stopping validation')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--force-retrain', action='store_true', help='Force retraining even if model exists')
    train_group.add_argument('--no-auto-save', action='store_true', help='Disable automatic model saving')
    train_group.add_argument('--model-dir', default='saved_models', help='Directory to save/load models')
    train_group.add_argument('--verbose', action='store_true', default=True, help='Verbose training output')
    train_group.add_argument('--quiet', dest='verbose', action='store_false', help='Disable verbose output')
    
    return parser


def build_pipeline_from_args(args) -> TextClassificationPipeline:
    """Build a pipeline based on command line arguments"""
    
    # Parse model type
    if args.model.startswith('tfidf_'):
        vectorizer_type = "tfidf"
        classifier_type = args.model.split('_')[1]  # lr, svm, rf, nb
        
        # Map abbreviated classifier names
        classifier_map = {
            'lr': 'logistic_regression',
            'svm': 'svm', 
            'rf': 'random_forest',
            'nb': 'naive_bayes',
            'nn': 'neural_network'
        }
        classifier_type = classifier_map.get(classifier_type, classifier_type)
        
    elif args.model == 'indobert_frozen':
        vectorizer_type = "indobert"
        classifier_type = "logistic_regression"
        
    elif args.model == 'indobert_nn':
        vectorizer_type = "indobert"
        classifier_type = "neural_network"
        
    elif args.model == 'indobert_finetune':
        vectorizer_type = "indobert_finetune"
        classifier_type = "dummy"
        
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Build vectorizer parameters
    vectorizer_params = {}
    
    if vectorizer_type == "tfidf":
        vectorizer_params = {
            'max_features': args.tfidf_max_features,
            'ngram_range': (args.tfidf_ngram_min, args.tfidf_ngram_max),
            'use_preprocessing': args.tfidf_preprocessing
        }
    
    elif vectorizer_type == "indobert":
        vectorizer_params = {
            'model_name': args.bert_model,
            'max_length': args.bert_max_length,
            'pooling_strategy': args.bert_pooling
        }
    
    elif vectorizer_type == "indobert_finetune":
        vectorizer_params = {
            'model_name': args.bert_model,
            'max_length': args.bert_max_length,
            'num_labels': args.bert_num_labels,
            'learning_rate': args.bert_lr,
            'num_epochs': args.bert_epochs,
            'batch_size': args.bert_batch_size,
            'warmup_steps': args.bert_warmup_steps
        }
    
    # Build classifier parameters
    classifier_params = {}
    
    if classifier_type == "logistic_regression":
        classifier_params = {
            'C': args.lr_C,
            'max_iter': args.lr_max_iter,
            'solver': args.lr_solver
        }
    
    elif classifier_type == "svm":
        classifier_params = {
            'C': args.svm_C,
            'kernel': args.svm_kernel,
            'gamma': args.svm_gamma,
            'probability': True  # Enable probability estimation
        }
    
    elif classifier_type == "random_forest":
        classifier_params = {
            'n_estimators': args.rf_n_estimators,
            'max_depth': args.rf_max_depth,
            'min_samples_split': args.rf_min_samples_split,
            'min_samples_leaf': args.rf_min_samples_leaf
        }
    
    elif classifier_type == "naive_bayes":
        classifier_params = {
            'alpha': args.nb_alpha
        }
    
    elif classifier_type == "neural_network":
        # Handle batch_size parameter conversion
        batch_size = args.nn_batch_size
        if batch_size == 'auto':
            batch_size = 'auto'
        else:
            try:
                batch_size = int(batch_size)
            except ValueError:
                batch_size = 'auto'
        
        classifier_params = {
            'hidden_layer_sizes': tuple(args.nn_hidden_layers),
            'activation': args.nn_activation,
            'solver': args.nn_solver,
            'alpha': args.nn_alpha,
            'batch_size': batch_size,
            'learning_rate': args.nn_learning_rate,
            'learning_rate_init': args.nn_learning_rate_init,
            'max_iter': args.nn_max_iter,
            'early_stopping': args.nn_early_stopping,
            'validation_fraction': args.nn_validation_fraction
        }
    
    # Create pipeline
    return TextClassificationPipeline(
        vectorizer_type=vectorizer_type,
        classifier_type=classifier_type,
        vectorizer_params=vectorizer_params,
        classifier_params=classifier_params
    )


def run_classification_pipeline(args):
    """Run the classification pipeline with given arguments"""
    
    print("=== Indonesian Text Classification ===")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Dataset subset: {args.dataset_subset}")
    print()
    
    # Load dataset
    print("[LOAD] Loading dataset...")
    loader = DatasetFactory.create_loader("indonlu", subset=args.dataset_subset)
    if args.verbose:
        loader.print_dataset_info()
    
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_train_test_split(
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state, 
        max_samples=args.max_samples
    )
    
    print(f"\n[DATA] Data splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # Build pipeline from arguments
    pipeline = build_pipeline_from_args(args)
    
    if args.verbose:
        print(f"\n[CONFIG] Pipeline Configuration:")
        config_info = pipeline.get_pipeline_info()
        print(f"   Vectorizer: {config_info['vectorizer_type']}")
        print(f"   Classifier: {config_info['classifier_type']}")
        print(f"   Vectorizer params: {config_info['vectorizer_params']}")
        print(f"   Classifier params: {config_info['classifier_params']}")
    
    # Train pipeline
    print(f"\n[TRAIN] Training {args.model} pipeline...")
    pipeline.fit(
        X_train, y_train, 
        verbose=args.verbose,
        force_retrain=args.force_retrain,
        auto_save=not args.no_auto_save,
        model_dir=args.model_dir
    )
    
    # Evaluate
    print("\n[EVAL] Evaluating on test set...")
    results = pipeline.evaluate(X_test, y_test)
    
    print("\n=== [RESULTS] ===")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.2f}%")
    print(f"F1-Score (Macro): {results['f1_macro']:.2f}%")
    print(f"Precision (Weighted): {results['precision_weighted']:.2f}%")
    print(f"Recall (Weighted): {results['recall_weighted']:.2f}%")
    print(f"Training Time: {results['training_time']:.1f} seconds")
    
    # Display vectorizer and classifier timing breakdown if available
    if hasattr(pipeline, 'vectorizer_time') and hasattr(pipeline, 'classifier_time'):
        print(f"  - Vectorizer Time: {pipeline.vectorizer_time:.1f} seconds")
        print(f"  - Classifier Time: {pipeline.classifier_time:.1f} seconds")
    
    print(f"Prediction Time: {results['prediction_time']:.3f} seconds")
    
    # Test with sample texts
    print("\n=== [SAMPLES] ===")
    test_texts = [
        "Makanan ini sangat enak dan pelayanannya memuaskan!",
        "Pelayanan buruk sekali, sangat mengecewakan",
        "Film yang menakjubkan, sangat saya rekomendasikan!",
        "Produk ini tidak sesuai dengan ekspektasi saya"
    ]
    
    predictions = pipeline.predict(test_texts)
    probabilities = pipeline.predict_proba(test_texts)
    
    for i, text in enumerate(test_texts):
        pred_label = loader.label_mapping[predictions[i]]
        confidence = max(probabilities[i])
        print(f"[TEXT] '{text}'")
        print(f"   -> {pred_label} (confidence: {confidence:.3f})")
    
    return pipeline, results


def main():
    """Main function with comprehensive argument parsing"""
    
    # Create argument parser
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Run the classification pipeline
    pipeline, results = run_classification_pipeline(args)
    
    print("\n=== [COMPLETED] ===")
    print(f"Final accuracy: {results['accuracy']:.2f}%")
    print(f"Final F1-score: {results['f1_weighted']:.2f}%")
    
    # Tips for users
    print("\n[TIPS] Optimization tips:")
    print("- Models are automatically saved and reused (unless --force-retrain is used)")
    print("- Use --force-retrain to retrain existing models")
    print("- Adjust --tfidf-max-features for TF-IDF models")
    print("- Try different --bert-pooling strategies for IndoBERT")
    print("- Increase --bert-epochs for better fine-tuning results")
    print("- Use --quiet to reduce output verbosity")
    print("- Run with --help to see all available parameters")
    
    return pipeline, results


if __name__ == "__main__":
    main()