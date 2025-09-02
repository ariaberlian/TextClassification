"""
Streamlit GUI for Indonesian Text Classification
Interactive web interface for main.py functionality
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time
import io
from contextlib import redirect_stdout, redirect_stderr

try:
    from main import run_classification_pipeline
    st.success("Successfully imported modules")
except ImportError as e:
    st.error(f"L Import error: {e}")
    st.error("Make sure all required files are in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Indonesian Text Classification",
    page_icon=">",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Indonesian Text Classification")
st.markdown("Interactive interface for text classification using TF-IDF and IndoBERT models")

# Sidebar for model configuration
st.sidebar.title("Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
model = st.sidebar.selectbox(
    "Choose Model",
    ['tfidf_lr', 'tfidf_svm', 'tfidf_nb', 'tfidf_nn',
     'indobert_lr', 'indobert_svm', 'indobert_nb', 'indobert_nn', 'indobert_finetune'],
    index=0,
    help="Select the model architecture to use"
)

# Dataset parameters
st.sidebar.subheader("Dataset Parameters")
dataset_subset = st.sidebar.selectbox("Dataset Subset", ["smsa"], index=0)
max_samples = st.sidebar.number_input("Max Samples", min_value=100, max_value=10000, value=None, 
                                     help="Maximum number of samples to use (None = use all)")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
val_size = st.sidebar.slider("Validation Size", 0.05, 0.3, 0.1, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, help="For reproducible results")

# Model-specific parameters
st.sidebar.subheader("Model Parameters")

# TF-IDF parameters (shown for TF-IDF models)
if model.startswith('tfidf_'):
    with st.sidebar.expander("TF-IDF Parameters"):
        tfidf_max_features = st.number_input("Max Features", 1000, 20000, 5000)
        tfidf_ngram_min = st.number_input("N-gram Min", 1, 3, 1)
        tfidf_ngram_max = st.number_input("N-gram Max", 1, 5, 2)
        tfidf_preprocessing = st.checkbox("Use Preprocessing", value=True)

# IndoBERT parameters (shown for IndoBERT models)
if model.startswith('indobert_'):
    with st.sidebar.expander("IndoBERT Parameters"):
        bert_model = st.selectbox("BERT Model", 
                                 ["indolem/indobert-base-uncased", "indobenchmark/indobert-base-p1"],
                                 index=0)
        bert_max_length = st.slider("Max Length", 64, 512, 128)
        bert_pooling = st.selectbox("Pooling Strategy", ["mean", "cls", "max"], index=0)

# Fine-tuning parameters (for indobert_finetune)
if model == 'indobert_finetune':
    with st.sidebar.expander("Fine-tuning Parameters"):
        bert_epochs = st.slider("Epochs", 1, 10, 2)
        bert_batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        bert_lr = st.selectbox("Learning Rate", [1e-5, 2e-5, 3e-5, 5e-5], index=1)
        bert_warmup_steps = st.number_input("Warmup Steps", 10, 200, 50)

# Classifier-specific parameters
classifier_type = model.split('_')[1] if '_' in model else 'lr'

if classifier_type == 'lr' or (model == 'indobert_lr'):
    with st.sidebar.expander("Logistic Regression Parameters"):
        lr_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        lr_max_iter = st.number_input("Max Iterations", 100, 5000, 1000)
        lr_solver = st.selectbox("Solver", ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], index=1)

elif classifier_type == 'svm':
    with st.sidebar.expander("SVM Parameters"):
        svm_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        svm_kernel = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
        svm_gamma = st.selectbox("Gamma", ['scale', 'auto'], index=0)


elif classifier_type == 'nb':
    with st.sidebar.expander("Naive Bayes Parameters"):
        nb_alpha = st.slider("Smoothing (Alpha)", 0.01, 5.0, 1.0, 0.01)

elif classifier_type == 'nn' or model == 'indobert_nn':
    with st.sidebar.expander("Neural Network Parameters"):
        nn_hidden_layers = st.text_input("Hidden Layers (space-separated)", "100", 
                                        help="e.g., '100 50 25'")
        nn_activation = st.selectbox("Activation", ['identity', 'logistic', 'tanh', 'relu'], index=3)
        nn_solver = st.selectbox("Solver", ['lbfgs', 'sgd', 'adam'], index=2)
        nn_alpha = st.number_input("L2 Regularization", 0.0001, 0.1, 0.0001, format="%.4f")
        nn_max_iter = st.slider("Max Iterations", 50, 1000, 200)
        nn_early_stopping = st.checkbox("Early Stopping", value=False)

# Training options
st.sidebar.subheader("Training Options")
force_retrain = st.sidebar.checkbox("Force Retrain", value=False)
verbose = st.sidebar.checkbox("Verbose Output", value=True)

# Model loading section
st.sidebar.subheader("Load Saved Model")
import os
import glob

# Get available saved models
def get_saved_models():
    model_files = []
    if os.path.exists('saved_models'):
        # Get pickle files (traditional ML models)
        pkl_files = glob.glob('saved_models/*.pkl')
        
        # Get HuggingFace model directories (should contain config.json or vectorizer_config.json)
        dirs = []
        for d in glob.glob('saved_models/*'):
            if os.path.isdir(d):
                # Check if it's a HuggingFace model directory
                config_files = ['config.json', 'vectorizer_config.json', 'pytorch_model.bin', 'model.safetensors']
                if any(os.path.exists(os.path.join(d, f)) for f in config_files):
                    dirs.append(d)
        
        model_files.extend(pkl_files)
        model_files.extend(dirs)
    
    return model_files

saved_models = get_saved_models()

if saved_models:
    # Extract readable model names
    model_names = []
    for model_path in saved_models:
        if model_path.endswith('.pkl'):
            # Extract model info from filename
            filename = os.path.basename(model_path)
            if 'tfidf' in filename and 'logistic' in filename:
                model_names.append(f"TF-IDF + Logistic Regression ({filename[:50]}...)")
            elif 'tfidf' in filename and 'svm' in filename:
                model_names.append(f"TF-IDF + SVM ({filename[:50]}...)")
            elif 'tfidf' in filename and 'naive_bayes' in filename:
                model_names.append(f"TF-IDF + Naive Bayes ({filename[:50]}...)")
            elif 'tfidf' in filename and 'neural_network' in filename:
                model_names.append(f"TF-IDF + Neural Network ({filename[:50]}...)")
            elif 'indobert' in filename and 'finetune' in filename:
                model_names.append(f"IndoBERT Fine-tuned ({filename[:50]}...)")
            elif 'indobert' in filename and 'logistic' in filename:
                model_names.append(f"IndoBERT + Logistic Regression ({filename[:50]}...)")
            elif 'indobert' in filename and 'svm' in filename:
                model_names.append(f"IndoBERT + SVM ({filename[:50]}...)")
            elif 'indobert' in filename and 'naive_bayes' in filename:
                model_names.append(f"IndoBERT + Naive Bayes ({filename[:50]}...)")
            elif 'indobert' in filename and 'neural_network' in filename:
                model_names.append(f"IndoBERT + Neural Network ({filename[:50]}...)")
            elif 'indobert' in filename:
                model_names.append(f"IndoBERT ({filename[:50]}...)")
            else:
                model_names.append(f"Model ({filename[:50]}...)")
        else:
            # Directory (HuggingFace model) - detect type from directory name and contents
            dirname = os.path.basename(model_path)
            
            # Check if it's a unified format model (has both HuggingFace files and classifier.pkl)
            has_classifier = os.path.exists(os.path.join(model_path, 'classifier.pkl'))
            has_metadata = os.path.exists(os.path.join(model_path, 'pipeline_metadata.json'))
            has_vectorizer_config = os.path.exists(os.path.join(model_path, 'vectorizer_config.json'))
            
            if has_classifier and has_metadata:
                # New unified format - read metadata to get exact model type
                try:
                    import json
                    metadata_path = os.path.join(model_path, 'pipeline_metadata.json')
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    vectorizer_type = metadata.get('vectorizer_type', 'unknown')
                    classifier_type = metadata.get('classifier_type', 'unknown')
                    
                    # Create descriptive name based on actual configuration
                    if vectorizer_type == 'indobert':
                        if classifier_type == 'logistic_regression':
                            model_names.append(f"IndoBERT + Logistic Regression [Unified] ({dirname[:30]}...)")
                        elif classifier_type == 'neural_network':
                            model_names.append(f"IndoBERT + Neural Network [Unified] ({dirname[:30]}...)")
                        elif classifier_type == 'svm':
                            model_names.append(f"IndoBERT + SVM [Unified] ({dirname[:30]}...)")
                        elif classifier_type == 'naive_bayes':
                            model_names.append(f"IndoBERT + Naive Bayes [Unified] ({dirname[:30]}...)")
                        else:
                            model_names.append(f"IndoBERT + {classifier_type.title()} [Unified] ({dirname[:30]}...)")
                    else:
                        model_names.append(f"{vectorizer_type.title()} + {classifier_type.title()} [Unified] ({dirname[:30]}...)")
                        
                except:
                    model_names.append(f"IndoBERT Hybrid [Unified] ({dirname[:40]}...)")
                    
            elif has_vectorizer_config and not has_classifier:
                # Check if it's a fine-tuned model or feature-only model
                try:
                    import json
                    config_path = os.path.join(model_path, 'vectorizer_config.json')
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    vectorizer_type = config.get('vectorizer_type', 'unknown')
                    if vectorizer_type == 'indobert_finetune':
                        model_names.append(f"IndoBERT Fine-tuned ({dirname[:40]}...)")
                    elif vectorizer_type == 'indobert':
                        model_names.append(f"IndoBERT Features Only ({dirname[:40]}...)")
                    else:
                        model_names.append(f"{vectorizer_type.title()} Model ({dirname[:40]}...)")
                except:
                    if 'finetune' in dirname:
                        model_names.append(f"IndoBERT Fine-tuned ({dirname[:40]}...)")
                    else:
                        model_names.append(f"IndoBERT Model ({dirname[:40]}...)")
            else:
                # Fallback for other HuggingFace models
                if 'finetune' in dirname or 'fine_tune' in dirname:
                    model_names.append(f"IndoBERT Fine-tuned ({dirname[:40]}...)")
                elif 'indobert' in dirname:
                    model_names.append(f"IndoBERT Model ({dirname[:40]}...)")
                else:
                    model_names.append(f"HuggingFace Model ({dirname[:40]}...)")
    
    selected_model_idx = st.sidebar.selectbox(
        "Select Model to Load",
        range(len(model_names)),
        format_func=lambda x: model_names[x],
        help="Choose a pre-trained model to load"
    )
    
    if st.sidebar.button("🔄 Load Selected Model"):
        try:
            selected_model_path = saved_models[selected_model_idx]
            
            st.sidebar.info("Loading model...")
            
            # Use the new hybrid loading approach
            from pipeline import TextClassificationPipeline
            
            try:
                # Load using the new hybrid approach
                pipeline = TextClassificationPipeline.load_pipeline(selected_model_path)
                
                # Validate the loaded pipeline
                if not hasattr(pipeline, 'predict') or not hasattr(pipeline, 'predict_proba'):
                    raise Exception("Loaded object is not a valid classifier pipeline")
                
                # Test the pipeline with a dummy prediction to ensure it's working
                try:
                    test_prediction = pipeline.predict(["Test text"])
                    test_proba = pipeline.predict_proba(["Test text"])
                    if len(test_prediction) == 0 or len(test_proba) == 0:
                        raise Exception("Pipeline prediction test failed")
                except Exception as e:
                    raise Exception(f"Pipeline validation failed: {str(e)}")
                
                # Determine model type based on path and pipeline
                if selected_model_path.endswith('.pkl'):
                    model_type = 'pickle'
                elif os.path.isdir(selected_model_path):
                    model_type = 'huggingface'
                else:
                    # Check if HuggingFace format was loaded
                    model_type = 'huggingface' if pipeline._get_model_save_strategy() == 'huggingface' else 'pickle'
                
                # Store in session state
                st.session_state['pipeline'] = pipeline
                st.session_state['trained'] = True
                st.session_state['loaded_model_path'] = selected_model_path
                st.session_state['loaded_model_type'] = model_type
                
                # Show success message with model info
                st.sidebar.success(f"✅ Model loaded successfully!")
                st.sidebar.info(f"📁 {os.path.basename(selected_model_path)}")
                
                # Show additional info about the model
                if hasattr(pipeline, '_is_transformer_vectorizer') and pipeline._is_transformer_vectorizer():
                    st.sidebar.info(f"🤗 Transformer model (GPU→CPU compatible)")
                    if hasattr(pipeline.vectorizer, 'device'):
                        st.sidebar.info(f"🔧 Device: {pipeline.vectorizer.device}")
                else:
                    st.sidebar.info(f"⚙️ Traditional ML model")
                
            except Exception as loading_error:
                # If new loading fails, try fallback methods for backward compatibility
                st.sidebar.warning("New loading method failed, trying fallback...")
                
                if selected_model_path.endswith('.pkl'):
                    # Fallback to direct joblib loading for old pickle files
                    import joblib
                    pipeline = joblib.load(selected_model_path)
                    model_type = 'pickle'
                    
                    # Store in session state
                    st.session_state['pipeline'] = pipeline
                    st.session_state['trained'] = True
                    st.session_state['loaded_model_path'] = selected_model_path
                    st.session_state['loaded_model_type'] = model_type
                    
                    st.sidebar.success(f"✅ Legacy model loaded!")
                    st.sidebar.info(f"📁 {os.path.basename(selected_model_path)}")
                    
                else:
                    # Re-raise the original error for HuggingFace models
                    raise loading_error
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            import traceback
            st.sidebar.error(f"Details: {traceback.format_exc()}")

else:
    st.sidebar.info("No saved models found in 'saved_models' directory")

# Main interface
tab1, tab2 = st.tabs(["Training", "Inference"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Train & Evaluate Model")
        
        if st.button("Start Training", type="primary", use_container_width=True):
            # Create arguments object similar to argparse
            class Args:
                def __init__(self):
                    self.model = model
                    self.dataset_subset = dataset_subset
                    self.max_samples = max_samples
                    self.test_size = test_size
                    self.val_size = val_size
                    self.random_state = random_state
                    self.force_retrain = force_retrain
                    self.verbose = verbose
                    self.no_auto_save = False
                    self.model_dir = 'saved_models'
                    
                    # TF-IDF parameters
                    if model.startswith('tfidf_'):
                        self.tfidf_max_features = tfidf_max_features
                        self.tfidf_ngram_min = tfidf_ngram_min
                        self.tfidf_ngram_max = tfidf_ngram_max
                        self.tfidf_preprocessing = tfidf_preprocessing
                    else:
                        self.tfidf_max_features = 5000
                        self.tfidf_ngram_min = 1
                        self.tfidf_ngram_max = 2
                        self.tfidf_preprocessing = True
                    
                    # IndoBERT parameters
                    if model.startswith('indobert_'):
                        self.bert_model = bert_model
                        self.bert_max_length = bert_max_length
                        self.bert_pooling = bert_pooling
                    else:
                        self.bert_model = 'indolem/indobert-base-uncased'
                        self.bert_max_length = 128
                        self.bert_pooling = 'mean'
                    
                    # Fine-tuning parameters
                    if model == 'indobert_finetune':
                        self.bert_epochs = bert_epochs
                        self.bert_batch_size = bert_batch_size
                        self.bert_lr = bert_lr
                        self.bert_warmup_steps = bert_warmup_steps
                        self.bert_num_labels = 2
                    else:
                        self.bert_epochs = 2
                        self.bert_batch_size = 8
                        self.bert_lr = 2e-5
                        self.bert_warmup_steps = 50
                        self.bert_num_labels = 2
                    
                    # Classifier parameters
                    self.lr_C = lr_C if classifier_type == 'lr' or model == 'indobert_lr' else 1.0
                    self.lr_max_iter = lr_max_iter if classifier_type == 'lr' or model == 'indobert_lr' else 1000
                    self.lr_solver = lr_solver if classifier_type == 'lr' or model == 'indobert_lr' else 'lbfgs'
                    
                    self.svm_C = svm_C if classifier_type == 'svm' else 1.0
                    self.svm_kernel = svm_kernel if classifier_type == 'svm' else 'rbf'
                    self.svm_gamma = svm_gamma if classifier_type == 'svm' else 'scale'
                    
                    
                    self.nb_alpha = nb_alpha if classifier_type == 'nb' else 1.0
                    
                    if classifier_type == 'nn' or model == 'indobert_nn':
                        self.nn_hidden_layers = [int(x) for x in nn_hidden_layers.split()]
                        self.nn_activation = nn_activation
                        self.nn_solver = nn_solver
                        self.nn_alpha = nn_alpha
                        self.nn_max_iter = nn_max_iter
                        self.nn_early_stopping = nn_early_stopping
                        self.nn_validation_fraction = 0.1
                        self.nn_batch_size = 'auto'
                        self.nn_learning_rate = 'constant'
                        self.nn_learning_rate_init = 0.001
                    else:
                        self.nn_hidden_layers = [100]
                        self.nn_activation = 'relu'
                        self.nn_solver = 'adam'
                        self.nn_alpha = 0.0001
                        self.nn_max_iter = 200
                        self.nn_early_stopping = False
                        self.nn_validation_fraction = 0.1
                        self.nn_batch_size = 'auto'
                        self.nn_learning_rate = 'constant'
                        self.nn_learning_rate_init = 0.001
        
            args = Args()
        
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Capture output
            output_container = st.container()
            with output_container:
                output_text = st.empty()
                
            # Run training
            try:
                # Capture stdout and stderr
                f = io.StringIO()
                with redirect_stdout(f), redirect_stderr(f):
                    status_text.text("Loading dataset...")
                    progress_bar.progress(20)
                    
                    pipeline, results = run_classification_pipeline(args)
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                
                # Display captured output
                output = f.getvalue()
                if output:
                    output_text.text_area("Training Log", output, height=400)
                
                # Store results in session state
                st.session_state['pipeline'] = pipeline
                st.session_state['results'] = results
                st.session_state['trained'] = True
                
                # Display results
                st.success("Training completed successfully!")
                
                col_acc, col_f1, col_time = st.columns(3)
                with col_acc:
                    st.metric("Accuracy", f"{results['accuracy']:.2f}%")
                with col_f1:
                    st.metric("F1-Score", f"{results['f1_weighted']:.2f}%")
                with col_time:
                    st.metric("Training Time", f"{results['training_time']:.1f}s")
                
            except Exception as e:
                st.error(f"L Error during training: {str(e)}")
                status_text.text("Training failed!")
                progress_bar.progress(0)

    with col2:
        st.header("Model Info")
        
        # Display current configuration
        config_data = {
            "Parameter": ["Model", "Dataset", "Max Samples", "Test Size", "Val Size"],
            "Value": [model, dataset_subset, max_samples or "All", f"{test_size:.1%}", f"{val_size:.1%}"]
        }
        
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

with tab2:
    st.header("Model Inference")
    
    # Check if model is trained
    if 'trained' in st.session_state and st.session_state['trained']:
        
        # Display loaded model info
        if 'loaded_model_path' in st.session_state:
            st.subheader("📋 Loaded Model Information")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                model_path = st.session_state['loaded_model_path']
                model_type = st.session_state.get('loaded_model_type', 'unknown')
                
                if model_type == 'huggingface':
                    st.info(f"**🤗 HuggingFace Model:** {os.path.basename(model_path)}")
                    
                    # Try to show model config if available
                    config_path = os.path.join(model_path, 'config.json')
                    if os.path.exists(config_path):
                        import json
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            st.write(f"**Model Architecture:** {config.get('_name_or_path', 'Unknown')}")
                            st.write(f"**Number of Labels:** {config.get('num_labels', 'Unknown')}")
                            st.write(f"**Max Position Embeddings:** {config.get('max_position_embeddings', 'Unknown')}")
                        except:
                            st.write("Config file found but couldn't be parsed")
                            
                elif model_type == 'pickle':
                    st.info(f"**🥒 Pickle Model:** {os.path.basename(model_path)}")
                    
                    # Extract model info from filename
                    filename = os.path.basename(model_path)
                    if 'tfidf' in filename:
                        st.write("**Feature Extraction:** TF-IDF")
                        if 'logistic' in filename:
                            st.write("**Classifier:** Logistic Regression")
                        elif 'svm' in filename:
                            st.write("**Classifier:** Support Vector Machine")
                        elif 'nb' in filename:
                            st.write("**Classifier:** Naive Bayes")
                    
                    # Show file size
                    file_size = os.path.getsize(model_path)
                    size_mb = file_size / (1024 * 1024)
                    st.write(f"**Model Size:** {size_mb:.1f} MB")
        
        
        st.divider()
        st.subheader("Text Prediction")
        st.success("✅ Model is ready for predictions")
        
        # Text input for prediction

        test_text = st.text_area(
            "Enter text to classify:",
            placeholder="Makanan ini sangat enak dan pelayanannya memuaskan!",
            height=100,
            key="inference_text"
        )
        
        if test_text and st.button("🔍 Predict", type="primary", use_container_width=True):
            try:
                import time
                
                # Start timing
                start_time = time.time()
                
                # Use the unified pipeline interface (works for all model types)
                pipeline = st.session_state['pipeline']
                
                # Make prediction (measure total pipeline time)
                inference_start = time.time()
                prediction = pipeline.predict([test_text])
                probabilities = pipeline.predict_proba([test_text])[0]
                inference_time = time.time() - inference_start
                
                # Calculate total prediction time
                total_time = time.time() - start_time
                
                # Get label mapping (assuming binary classification)
                label_mapping = {0: "Negative", 1: "Positive"}
                
                pred_label = label_mapping.get(prediction[0], f"Class {prediction[0]}")
                confidence = max(probabilities)
                
                # Prediction Results Section
                st.subheader("🎯 Prediction Results")
                
                # Main prediction display
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                with col1:
                    if prediction[0] == 1:
                        st.success(f"**Prediction: {pred_label}** 😊")
                    else:
                        st.error(f"**Prediction: {pred_label}** 😞")
                
                with col2:
                    st.info(f"**Confidence: {confidence:.1%}**")
                
                with col3:
                    certainty = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    certainty_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                    st.metric("Certainty", f"{certainty} {certainty_color}")
                
                with col4:
                    # Display prediction time
                    if total_time < 0.001:
                        time_display = f"{total_time*1000000:.0f}µs"
                    elif total_time < 1:
                        time_display = f"{total_time*1000:.0f}ms"
                    else:
                        time_display = f"{total_time:.2f}s"
                    st.metric("⏱️ Time", time_display)
                
                # Detailed metrics
                st.subheader("📊 Detailed Metrics")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Probability distribution
                    prob_df = pd.DataFrame({
                        'Class': ['Negative', 'Positive'],
                        'Probability': probabilities
                    })
                    st.bar_chart(prob_df.set_index('Class'), height=300)
                
                with col2:
                    # Additional metrics
                    st.markdown("**📈 Prediction Metrics:**")
                    
                    # Confidence level indicator
                    confidence_pct = confidence * 100
                    st.progress(confidence, text=f"Confidence: {confidence_pct:.1f}%")
                    
                    # Timing details
                    st.markdown("**⏱️ Timing Breakdown:**")
                    st.markdown(f"- **Pipeline Time**: {inference_time*1000:.1f}ms")
                    st.markdown(f"- **Total Time**: {total_time*1000:.1f}ms")
                    
                    # Show model type info
                    model_type = st.session_state.get('loaded_model_type', 'unknown')
                    if model_type == 'huggingface':
                        st.markdown("- **Format**: HuggingFace (GPU→CPU compatible)")
                    else:
                        st.markdown("- **Format**: Traditional ML")
                    
                    # Class probabilities
                    st.markdown("**Class Probabilities:**")
                    for i, (class_name, prob) in enumerate(zip(['Negative', 'Positive'], probabilities)):
                        color = "green" if i == prediction[0] else "gray"
                        st.markdown(f"- **{class_name}**: {prob:.3f} ({prob*100:.1f}%)")
                    
                    # Prediction uncertainty
                    uncertainty = 1 - confidence
                    st.markdown(f"**Prediction Uncertainty**: {uncertainty:.3f} ({uncertainty*100:.1f}%)")
                
                # Store prediction for batch analysis
                if 'predictions_history' not in st.session_state:
                    st.session_state['predictions_history'] = []
                
                st.session_state['predictions_history'].append({
                    'text': test_text[:50] + "..." if len(test_text) > 50 else test_text,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'prediction_time': total_time,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

        
        # Prediction history
        if 'predictions_history' in st.session_state and st.session_state['predictions_history']:
            st.divider()
            st.subheader("📈 Recent Predictions")
            
            history_df = pd.DataFrame(st.session_state['predictions_history'][-10:])  # Show last 10
            if not history_df.empty:
                # Format prediction time for display
                if 'prediction_time' in history_df.columns:
                    history_df['time_ms'] = (history_df['prediction_time'] * 1000).round(1)
                    display_columns = ['text', 'prediction', 'confidence', 'time_ms']
                    column_config = {
                        'time_ms': st.column_config.NumberColumn(
                            "Time (ms)",
                            help="Prediction time in milliseconds",
                            format="%.1f ms"
                        )
                    }
                else:
                    display_columns = ['text', 'prediction', 'confidence']
                    column_config = None
                
                st.dataframe(
                    history_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                if st.button("🗑️ Clear History"):
                    st.session_state['predictions_history'] = []
                    st.rerun()
    
    else:
        st.warning("⚠️ Please train a model first in the Training tab to use inference features")