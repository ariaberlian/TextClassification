from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union, Optional
import re

# Optional imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


class BaseVectorizer(ABC):
    """Abstract base class for text vectorizers"""
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, texts: List[str]) -> 'BaseVectorizer':
        """Fit the vectorizer to the training texts"""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors"""
        pass
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(texts).transform(texts)


class IndonesianTextPreprocessor:
    """Indonesian text preprocessing utilities"""
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming and SASTRAWI_AVAILABLE
        self.remove_stopwords = remove_stopwords and SASTRAWI_AVAILABLE
        
        if self.use_stemming and SASTRAWI_AVAILABLE:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = None
        
        if self.remove_stopwords and SASTRAWI_AVAILABLE:
            factory = StopWordRemoverFactory()
            self.stopword_remover = factory.create_stop_word_remover()
        else:
            self.stopword_remover = None
        
        # Basic Indonesian stopwords fallback
        if not SASTRAWI_AVAILABLE:
            self.basic_stopwords = {
                'dan', 'atau', 'yang', 'ini', 'itu', 'adalah', 'akan', 'ada', 'dengan',
                'untuk', 'pada', 'dari', 'ke', 'di', 'dalam', 'tidak', 'juga', 'oleh',
                'dapat', 'bisa', 'sudah', 'telah', 'masih', 'harus', 'lebih', 'sangat',
                'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami'
            }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove non-alphabetic characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text.lower()
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        
        if self.remove_stopwords:
            if self.stopword_remover is not None:
                text = self.stopword_remover.remove(text)
            else:
                # Fallback stopword removal
                words = text.split()
                words = [word for word in words if word.lower() not in self.basic_stopwords]
                text = ' '.join(words)
        
        if self.use_stemming and self.stemmer is not None:
            text = self.stemmer.stem(text)
        
        return text


class TFIDFVectorizer(BaseVectorizer):
    """TF-IDF based vectorizer with Indonesian preprocessing"""
    
    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2),
                 use_preprocessing: bool = True, **tfidf_kwargs):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_preprocessing = use_preprocessing
        
        if self.use_preprocessing:
            self.preprocessor = IndonesianTextPreprocessor()
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **tfidf_kwargs
        )
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts if enabled"""
        if self.use_preprocessing:
            return [self.preprocessor.preprocess(text) for text in texts]
        return texts
    
    def fit(self, texts: List[str]) -> 'TFIDFVectorizer':
        """Fit the TF-IDF vectorizer"""
        processed_texts = self._preprocess_texts(texts)
        self.vectorizer.fit(processed_texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        processed_texts = self._preprocess_texts(texts)
        return self.vectorizer.transform(processed_texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (words/ngrams)"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()


class IndoBERTVectorizer(BaseVectorizer):
    """IndoBERT based vectorizer using Hugging Face transformers (frozen features)"""
    
    def __init__(self, model_name: str = "indolem/indobert-base-uncased",
                 max_length: int = 512, pooling_strategy: str = "mean"):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for IndoBERT. Install with: pip install torch transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def fit(self, texts: List[str]) -> 'IndoBERTVectorizer':
        """IndoBERT doesn't need fitting, but we set the flag for consistency"""
        self.is_fitted = True
        return self
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Get BERT embeddings for texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if self.pooling_strategy == "cls":
                    # Use CLS token
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.pooling_strategy == "mean":
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                elif self.pooling_strategy == "max":
                    # Max pooling
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    token_embeddings[input_mask_expanded == 0] = -1e9
                    batch_embeddings = torch.max(token_embeddings, 1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to IndoBERT embeddings"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        return self._get_embeddings(texts)


class IndoBERTFineTuneVectorizer(BaseVectorizer):
    """IndoBERT with fine-tuning capability for sentiment classification"""
    
    def __init__(self, model_name: str = "indolem/indobert-base-uncased",
                 max_length: int = 256, num_labels: int = 2, 
                 learning_rate: float = 2e-5, num_epochs: int = 3,
                 batch_size: int = 16, warmup_steps: int = 100):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for IndoBERT fine-tuning. Install with: pip install torch transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # Import required components
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from torch.utils.data import DataLoader, Dataset
        import torch.nn.functional as F
        
        # Try importing AdamW from different locations (version compatibility)
        try:
            from transformers import AdamW
        except ImportError:
            from torch.optim import AdamW
        
        # Initialize tokenizer and model for classification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Store imports for later use
        self._AdamW = AdamW
        self._get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
        self._DataLoader = DataLoader
        self._F = F
        
    class SentimentDataset(torch.utils.data.Dataset):
        """Custom dataset for sentiment analysis"""
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    def fit(self, texts: List[str], labels: List[int] = None) -> 'IndoBERTFineTuneVectorizer':
        """Fine-tune IndoBERT on the provided texts and labels"""
        if labels is None:
            raise ValueError("Labels are required for fine-tuning IndoBERT")
        
        import torch
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            def tqdm(iterable, desc="Processing"):
                print(f"{desc}...")
                return iterable
        
        print(f"Fine-tuning IndoBERT on {len(texts)} samples...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}, LR: {self.learning_rate}")
        
        # Create dataset and dataloader
        dataset = self.SentimentDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = self._DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = self._AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.num_epochs
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
        
        self.model.eval()
        self.is_fitted = True
        print("✅ Fine-tuning completed!")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Get predictions from fine-tuned model (returns probabilities)"""
        if not self.is_fitted:
            raise ValueError("Model must be fine-tuned before transform")
        
        import torch
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc="Processing"):
                print(f"{desc}...")
                return iterable
        
        self.model.eval()
        all_probs = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating predictions"):
            batch_texts = texts[i:i + self.batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = self._F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
        return np.vstack(all_probs)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Get class predictions"""
        probs = self.transform(texts)
        return np.argmax(probs, axis=1)


class VectorizerFactory:
    """Factory class to create different types of vectorizers"""
    
    @staticmethod
    def create_vectorizer(vectorizer_type: str, **kwargs) -> BaseVectorizer:
        """Create a vectorizer based on type"""
        if vectorizer_type.lower() == "tfidf":
            return TFIDFVectorizer(**kwargs)
        elif vectorizer_type.lower() == "indobert":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers and torch are required for IndoBERT. Install with: pip install torch transformers")
            return IndoBERTVectorizer(**kwargs)
        elif vectorizer_type.lower() == "indobert_finetune":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers and torch are required for IndoBERT fine-tuning. Install with: pip install torch transformers")
            return IndoBERTFineTuneVectorizer(**kwargs)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
    
    @staticmethod
    def get_available_vectorizers() -> List[str]:
        """Get list of available vectorizer types"""
        available = ["tfidf"]
        if TRANSFORMERS_AVAILABLE:
            available.extend(["indobert", "indobert_finetune"])
        return available