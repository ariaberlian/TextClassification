import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class IndonesianSentimentLoader:
    """
    Loader for Indonesian sentiment analysis datasets
    """
    
    def __init__(self, dataset_name: str = "indonlu", subset: str = "smsa"):
        """
        Initialize the dataset loader
        
        Args:
            dataset_name: Name of the dataset to load
            subset: Subset of the dataset (for indonlu: 'smsa' for sentiment analysis)
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None
        self.label_mapping = None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the Indonesian sentiment dataset
        
        Returns:
            Dictionary containing train/validation/test DataFrames
        """
        print(f"Loading {self.dataset_name} dataset ({self.subset})...")
        
        try:
            if self.dataset_name == "indonlu" and self.subset == "smsa":
                # Load IndoNLU SMSA from direct TSV files on GitHub
                print("Loading SMSA dataset from GitHub TSV files...")
                
                base_url = "https://raw.githubusercontent.com/indobenchmark/indonlu/master/dataset/smsa_doc-sentiment-prosa"
                data_files = {
                    "train": f"{base_url}/train_preprocess.tsv",
                    "validation": f"{base_url}/valid_preprocess.tsv", 
                    "test": f"{base_url}/test_preprocess.tsv"
                }
                
                # Load each TSV file separately with proper column names
                dataset = load_dataset("csv", data_files=data_files, delimiter="\t", 
                                     column_names=["text", "label"], header=None)
                
                # Convert to pandas DataFrames
                splits = {}
                for split_name in dataset.keys():
                    df = pd.DataFrame(dataset[split_name])
                    splits[split_name] = df
                
                # Create binary label mapping (only negative and positive)
                label_to_num = {'negative': 0, 'positive': 1}
                self.label_mapping = {0: 'negative', 1: 'positive'}
                
                # Convert string labels to numbers and filter out neutral samples
                for split_name in splits:
                    # Filter out neutral samples
                    splits[split_name] = splits[split_name][splits[split_name]['label'] != 'neutral'].copy()
                    # Convert remaining labels to numbers
                    splits[split_name]['label'] = splits[split_name]['label'].map(label_to_num)
                    # Reset index after filtering
                    splits[split_name] = splits[split_name].reset_index(drop=True)
                
                self.dataset = splits
                return splits
                
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name} - {self.subset}")
                
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            # print("Falling back to sample dataset...")
            # return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Create a sample Indonesian sentiment dataset for testing
        """
        sample_data = {
            'text': [
                "Saya sangat suka makanan ini, rasanya enak sekali!",
                "Pelayanan di restoran ini sangat buruk dan mengecewakan.",
                "Film ini benar-benar membosankan, tidak ada yang menarik.",
                "Terima kasih atas bantuan yang sangat luar biasa!",
                "Produk ini kualitasnya sangat jelek, tidak sesuai harga.",
                "Pengalaman berbelanja yang menyenangkan di toko ini.",
                "Makanan di warung ini tidak enak dan mahal.",
                "Layanan customer service sangat lambat dan tidak responsif.",
                "Buku ini sangat bagus dan memberikan banyak inspirasi.",
                "Aplikasi ini sering error dan sangat mengganggu.",
                "Tempat wisata yang sangat indah dan menakjubkan.",
                "Harga produk ini terlalu mahal untuk kualitas yang biasa saja.",
                "Guru ini mengajar dengan sangat baik dan sabar.",
                "Website ini loading-nya lambat dan sering hang.",
                "Konser kemarin benar-benar spektakuler dan menghibur.",
                "Transportasi umum di kota ini sangat tidak nyaman.",
                "Terimakasih untuk pelayanan yang ramah dan cepat.",
                "Game ini sangat membosankan dan tidak challenging."
            ],
            'label': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Split into train/val/test (without stratification due to small sample size)
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Create binary label mapping
        self.label_mapping = {0: 'negative', 1: 'positive'}
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        self.dataset = splits
        print("Sample dataset created with 20 Indonesian sentiment examples")
        return splits
    
    def get_train_test_split(self, test_size: float = 0.2, val_size: float = 0.1,
                           random_state: int = 42, max_samples: int = None) -> Tuple[List[str], List[str], List[str], 
                                                          List[Union[str, int]], List[Union[str, int]], List[Union[str, int]]]:
        """
        Get train/validation/test split
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            max_samples: Maximum number of samples to use (None for all data)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.dataset is None:
            self.load_data()
        
        # If dataset already has splits, use them
        if 'train' in self.dataset and 'validation' in self.dataset and 'test' in self.dataset:
            train_df = self.dataset['train'].copy()
            val_df = self.dataset['validation'].copy()
            test_df = self.dataset['test'].copy()
            
            # Limit data if max_samples is specified
            if max_samples is not None:
                train_samples = int(max_samples * 0.8)  # 80% for training
                val_samples = int(max_samples * 0.1)    # 10% for validation
                test_samples = int(max_samples * 0.1)   # 10% for testing
                
                train_df = train_df.sample(min(train_samples, len(train_df)), random_state=random_state)
                val_df = val_df.sample(min(val_samples, len(val_df)), random_state=random_state)
                test_df = test_df.sample(min(test_samples, len(test_df)), random_state=random_state)
            
            X_train = train_df['text'].tolist()
            y_train = train_df['label'].tolist()
            
            X_val = val_df['text'].tolist()
            y_val = val_df['label'].tolist()
            
            X_test = test_df['text'].tolist()
            y_test = test_df['label'].tolist()
            
        else:
            # Create splits manually
            if 'train' in self.dataset:
                df = self.dataset['train'].copy()
            else:
                # Combine all available data
                df = pd.concat([self.dataset[split] for split in self.dataset.keys()], ignore_index=True)
            
            # Limit data if max_samples is specified
            if max_samples is not None and len(df) > max_samples:
                df = df.sample(max_samples, random_state=random_state)
            
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            
            # First split: train and temp (test + val) - no stratification for small samples
            X_train, X_temp, y_train, y_temp = train_test_split(
                texts, labels, test_size=(test_size + val_size), 
                random_state=random_state
            )
            
            # Second split: temp into test and val
            if val_size > 0:
                val_ratio = val_size / (test_size + val_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=(1 - val_ratio),
                    random_state=random_state
                )
            else:
                X_test, y_test = X_temp, y_temp
                X_val, y_val = [], []
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_label_info(self) -> Dict[str, Union[Dict, List]]:
        """Get information about the labels"""
        if self.dataset is None:
            self.load_data()
        
        info = {
            'label_mapping': self.label_mapping,
            'num_classes': len(self.label_mapping) if self.label_mapping else 0
        }
        
        if self.dataset and 'train' in self.dataset:
            train_df = self.dataset['train']
            label_counts = train_df['label'].value_counts().sort_index().to_dict()
            info['label_distribution'] = label_counts
        
        return info
    
    def print_dataset_info(self):
        """Print information about the loaded dataset"""
        if self.dataset is None:
            print("Dataset not loaded yet. Call load_data() first.")
            return
        
        print(f"\nDataset: {self.dataset_name} ({self.subset})")
        print("-" * 40)
        
        for split_name, df in self.dataset.items():
            print(f"{split_name.capitalize()} set: {len(df)} samples")
        
        print(f"\nLabel mapping: {self.label_mapping}")
        
        if 'train' in self.dataset:
            train_df = self.dataset['train']
            print(f"\nLabel distribution in training set:")
            label_counts = train_df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                label_name = self.label_mapping.get(label, label) if self.label_mapping else label
                print(f"  {label_name}: {count} samples")
        
        print(f"\nSample texts:")
        if 'train' in self.dataset:
            sample_df = self.dataset['train'].head(3)
            for idx, row in sample_df.iterrows():
                label_name = self.label_mapping.get(row['label'], row['label']) if self.label_mapping else row['label']
                print(f"  [{label_name}] {row['text'][:100]}...")


class DatasetFactory:
    """Factory for creating different dataset loaders"""
    
    @staticmethod
    def create_loader(dataset_name: str, **kwargs) -> IndonesianSentimentLoader:
        """Create a dataset loader"""
        if dataset_name.lower() == "indonlu":
            return IndonesianSentimentLoader(dataset_name="indonlu", subset="smsa")
        elif dataset_name.lower() == "sample":
            loader = IndonesianSentimentLoader()
            loader._create_sample_dataset()
            return loader
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available datasets"""
        return ["indonlu", "sample"]