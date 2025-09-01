"""
Streamlit GUI for Indonesian Text Classification
Interactive web interface for main.py functionality
"""

import streamlit as st
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from typing import Dict, Any, List
import time
import io
from contextlib import redirect_stdout, redirect_stderr

# Import main.py functions
try:
    from main import build_pipeline_from_args, run_classification_pipeline
    from dataset_loader import DatasetFactory
    from pipeline import TextClassificationPipeline
    st.success(" Successfully imported modules")
except ImportError as e:
    st.error(f"L Import error: {e}")
    st.error("Make sure all required files are in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Indonesian Text Classification",
    page_icon=">",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("> Indonesian Text Classification")
st.markdown("Interactive interface for text classification using TF-IDF and IndoBERT models")

# Sidebar for model configuration
st.sidebar.title("� Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
model = st.sidebar.selectbox(
    "Choose Model",
    ['tfidf_lr', 'tfidf_svm', 'tfidf_rf', 'tfidf_nb', 'tfidf_nn',
     'indobert_frozen', 'indobert_finetune', 'indobert_nn'],
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

if classifier_type == 'lr' or (model == 'indobert_frozen'):
    with st.sidebar.expander("Logistic Regression Parameters"):
        lr_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        lr_max_iter = st.number_input("Max Iterations", 100, 5000, 1000)
        lr_solver = st.selectbox("Solver", ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], index=1)

elif classifier_type == 'svm':
    with st.sidebar.expander("SVM Parameters"):
        svm_C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        svm_kernel = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
        svm_gamma = st.selectbox("Gamma", ['scale', 'auto'], index=0)

elif classifier_type == 'rf':
    with st.sidebar.expander("Random Forest Parameters"):
        rf_n_estimators = st.slider("Number of Trees", 10, 500, 100)
        rf_max_depth = st.number_input("Max Depth", value=None, help="None for unlimited")
        rf_min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        rf_min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

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

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("=� Train & Evaluate Model")
    
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
                self.lr_C = lr_C if classifier_type == 'lr' or model == 'indobert_frozen' else 1.0
                self.lr_max_iter = lr_max_iter if classifier_type == 'lr' or model == 'indobert_frozen' else 1000
                self.lr_solver = lr_solver if classifier_type == 'lr' or model == 'indobert_frozen' else 'lbfgs'
                
                self.svm_C = svm_C if classifier_type == 'svm' else 1.0
                self.svm_kernel = svm_kernel if classifier_type == 'svm' else 'rbf'
                self.svm_gamma = svm_gamma if classifier_type == 'svm' else 'scale'
                
                self.rf_n_estimators = rf_n_estimators if classifier_type == 'rf' else 100
                self.rf_max_depth = rf_max_depth if classifier_type == 'rf' else None
                self.rf_min_samples_split = rf_min_samples_split if classifier_type == 'rf' else 2
                self.rf_min_samples_leaf = rf_min_samples_leaf if classifier_type == 'rf' else 1
                
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
            st.success(" Training completed successfully!")
            
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
    st.header("=� Model Info")
    
    # Display current configuration
    config_data = {
        "Parameter": ["Model", "Dataset", "Max Samples", "Test Size", "Val Size"],
        "Value": [model, dataset_subset, max_samples or "All", f"{test_size:.1%}", f"{val_size:.1%}"]
    }
    
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

# Text prediction section
st.header("=. Test Predictions")

# Check if model is trained
if 'trained' in st.session_state and st.session_state['trained']:
    st.success(" Model is ready for predictions")
    
    # Text input for prediction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_text = st.text_area(
            "Enter text to classify:",
            placeholder="Makanan ini sangat enak dan pelayanannya memuaskan!",
            height=100
        )
    
    with col2:
        st.markdown("### Sample Texts")
        if st.button("Positive Review", use_container_width=True):
            st.session_state['test_text'] = "Makanan ini sangat enak dan pelayanannya memuaskan!"
        if st.button("Negative Review", use_container_width=True):
            st.session_state['test_text'] = "Pelayanan buruk sekali, sangat mengecewakan"
        if st.button("Positive Movie", use_container_width=True):
            st.session_state['test_text'] = "Film yang menakjubkan, sangat saya rekomendasikan!"
        if st.button("Negative Product", use_container_width=True):
            st.session_state['test_text'] = "Produk ini tidak sesuai dengan ekspektasi saya"
    
    # Use session state text if available
    if 'test_text' in st.session_state:
        test_text = st.session_state['test_text']
    
    if test_text and st.button("Predict", type="primary"):
        try:
            pipeline = st.session_state['pipeline']
            
            # Make prediction
            prediction = pipeline.predict([test_text])
            probabilities = pipeline.predict_proba([test_text])
            
            # Get label mapping (assuming binary classification)
            label_mapping = {0: "Negative", 1: "Positive"}  # Adjust based on your dataset
            
            pred_label = label_mapping.get(prediction[0], f"Class {prediction[0]}")
            confidence = max(probabilities[0])
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 1:
                    st.success(f"**Prediction: {pred_label}** =")
                else:
                    st.error(f"**Prediction: {pred_label}** =")
            
            with col2:
                st.info(f"**Confidence: {confidence:.3f}**")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Class': ['Negative', 'Positive'],
                'Probability': probabilities[0]
            })
            st.bar_chart(prob_df.set_index('Class'))
            
        except Exception as e:
            st.error(f"L Prediction error: {str(e)}")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'text' column",
        type=['csv'],
        help="CSV file should have a 'text' column with texts to classify"
    )
    
    if uploaded_file and st.button("Predict Batch"):
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("L CSV file must have a 'text' column")
            else:
                pipeline = st.session_state['pipeline']
                
                # Make predictions
                predictions = pipeline.predict(df['text'].tolist())
                probabilities = pipeline.predict_proba(df['text'].tolist())
                
                # Add results to dataframe
                label_mapping = {0: "Negative", 1: "Positive"}
                df['prediction'] = [label_mapping.get(p, f"Class {p}") for p in predictions]
                df['confidence'] = [max(prob) for prob in probabilities]
                
                st.success(f" Processed {len(df)} texts")
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"L Batch prediction error: {str(e)}")

else:
    st.warning("� Please train a model first to make predictions")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Indonesian Text Classification GUI | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)