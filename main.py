"""
Indonesian Text Classification - Google Colab Compatible Main Script

Simple script to run the text classification pipeline in Google Colab.
Run this after uploading the project files to Colab.

Example usage in Colab:
    # Upload files to Colab, then run:
    !python main.py
    
    # Or with custom parameters:
    !python main.py --model indobert_finetune --max-samples 500
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from dataset_loader import DatasetFactory
    from pipeline import TextClassificationPipeline
    print("✅ Successfully imported custom modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all project files are uploaded to Colab")
    sys.exit(1)


def run_simple_demo(model_type='tfidf_lr', max_samples=1000):
    """Run a simple demonstration of the text classification system"""
    
    print("=== Indonesian Text Classification Demo ===")
    print(f"Model: {model_type}")
    print(f"Max samples: {max_samples}")
    print()
    
    # Load dataset
    print("📥 Loading dataset...")
    loader = DatasetFactory.create_loader("indonlu")
    loader.print_dataset_info()
    
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_train_test_split(
        test_size=0.2, val_size=0.1, random_state=42, max_samples=max_samples
    )
    
    print(f"\n📊 Data splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # Create pipeline based on model type
    if model_type == 'tfidf_lr':
        pipeline = TextClassificationPipeline(
            vectorizer_type="tfidf",
            classifier_type="logistic_regression",
            vectorizer_params={
                'max_features': 5000,
                'ngram_range': (1, 2),
                'use_preprocessing': True
            },
            classifier_params={
                'C': 1.0,
                'max_iter': 1000
            }
        )
    elif model_type == 'indobert_frozen':
        pipeline = TextClassificationPipeline(
            vectorizer_type="indobert",
            classifier_type="logistic_regression",
            vectorizer_params={
                'model_name': 'indolem/indobert-base-uncased',
                'max_length': 128,
                'pooling_strategy': 'mean'
            }
        )
    elif model_type == 'indobert_finetune':
        pipeline = TextClassificationPipeline(
            vectorizer_type="indobert_finetune",
            classifier_type="dummy",
            vectorizer_params={
                'model_name': 'indolem/indobert-base-uncased',
                'max_length': 128,
                'num_labels': 2,
                'learning_rate': 2e-5,
                'num_epochs': 2,
                'batch_size': 8,
                'warmup_steps': 50
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\n🔧 Created {model_type} pipeline")
    
    # Train pipeline
    print("\n🚀 Training pipeline...")
    pipeline.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    print("\n📈 Evaluating on test set...")
    results = pipeline.evaluate(X_test, y_test)
    
    print("\n=== 🎯 Results ===")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.2f}%")
    print(f"F1-Score (Macro): {results['f1_macro']:.2f}%")
    print(f"Training Time: {results['training_time']:.1f} seconds")
    print(f"Prediction Time: {results['prediction_time']:.3f} seconds")
    
    # Test with sample texts
    print("\n=== 🧪 Sample Predictions ===")
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
        print(f"📝 '{text}'")
        print(f"   → {pred_label} (confidence: {confidence:.3f})")
    
    return pipeline, results


def main():
    """Main function - can be customized for different experiments"""
    
    # Parse simple command line arguments
    import sys
    
    model_type = 'tfidf_lr'  # Default
    max_samples = None      # Default
    
    # Simple argument parsing
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_type = sys.argv[idx + 1]
    
    if '--max-samples' in sys.argv:
        idx = sys.argv.index('--max-samples')
        if idx + 1 < len(sys.argv):
            max_samples = int(sys.argv[idx + 1])
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python main.py [--model MODEL] [--max-samples N]")
        print("Models: tfidf_lr, indobert_frozen, indobert_finetune")
        print("Example: python main.py --model indobert_finetune --max-samples 500")
        return
    
    # Run the demo
    pipeline, results = run_simple_demo(model_type, max_samples)
    
    print("\n=== ✅ Demo completed! ===")
    print(f"Final accuracy: {results['accuracy']:.2f}%")
    
    # Colab-specific tips
    print("\n💡 Tips for Google Colab:")
    print("• Use GPU runtime for faster IndoBERT training")
    print("• Start with smaller max_samples (100-500) for quick testing")
    print("• Try different models: tfidf_lr, indobert_frozen, indobert_finetune")
    print("• Monitor memory usage with large datasets")


if __name__ == "__main__":
    main()