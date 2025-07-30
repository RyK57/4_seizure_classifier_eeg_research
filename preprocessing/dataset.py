import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

class EEGDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        
        self.feature_cols = [col for col in data.columns if col.startswith('X')]
        self.target_col = 'y'
        
        if self.target_col not in data.columns:
            raise ValueError("Target column 'y' not found in the dataset")
        
        self.features = data[self.feature_cols].values.astype(np.float32)
        self.targets = data[self.target_col].values.astype(np.int64)
        
        print(f"Dataset created with {len(self.features)} samples and {len(self.feature_cols)} features")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Target distribution: {np.bincount(self.targets)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

class EEGDataModule:
    """
   all the arguments : 
        data_path: Path to the CSV file
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        batch_size: Batch size for data loaders
        random_state: Random state for reproducibility
        transform: Transform to apply to training data
        val_transform: Transform to apply to validation data
        test_transform: Transform to apply to test data
    """
    
    def __init__(self, data_path: str, test_size: float = 0.2, val_size: float = 0.2,
                 batch_size: int = 32, random_state: int = 42, transform=None,
                 val_transform=None, test_transform=None):
        
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.transform = transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        
        self._load_and_split_data()
        self._create_data_loaders()
    
    def _load_and_split_data(self):
        print(f"Loading data from {self.data_path}")
        
        self.full_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.full_data)} samples")
        
        train_data, test_data = train_test_split(
            self.full_data, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.full_data['y']
        )
        
        train_data, val_data = train_test_split(
            train_data,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_data['y']
        )
        
        print(f"Train set: {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples")
        print(f"Test set: {len(test_data)} samples")
        
        self.train_dataset = EEGDataset(train_data, transform=self.transform)
        self.val_dataset = EEGDataset(val_data, transform=self.val_transform)
        self.test_dataset = EEGDataset(test_data, transform=self.test_transform)
    
    def _create_data_loaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"sizes - {self.batch_size}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self) -> Tuple[EEGDataset, EEGDataset, EEGDataset]:
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_class_weights(self) -> torch.Tensor:
        targets = self.train_dataset.targets
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights}")
        
        return torch.tensor(class_weights, dtype=torch.float32)

def create_simple_split(data_path: str, test_size: float = 0.2, batch_size: int = 32,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    data = pd.read_csv(data_path)
    
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data['y']
    )
    
    train_dataset = EEGDataset(train_data)
    test_dataset = EEGDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_dataset_info(data_path: str) -> Dict:
    data = pd.read_csv(data_path)
    
    feature_cols = [col for col in data.columns if col.startswith('X')]
    target_col = 'y'
    
    info = {
        'total_samples': len(data),
        'n_features': len(feature_cols),
        'n_classes': len(data[target_col].unique()),
        'feature_columns': feature_cols,
        'target_column': target_col,
        'class_distribution': data[target_col].value_counts().sort_index().to_dict(),
        'missing_values': data.isnull().sum().sum(),
        'feature_ranges': {
            col: (data[col].min(), data[col].max()) for col in feature_cols
        }
    }
    
    return info

def create_kfold_splits(data_path: str, n_splits: int = 5, batch_size: int = 32,
                       random_state: int = 42) -> List[Tuple[DataLoader, DataLoader]]:
        
    data = pd.read_csv(data_path)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Creating fold {fold + 1}/{n_splits}")
        
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        train_dataset = EEGDataset(train_data)
        val_dataset = EEGDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        splits.append((train_loader, val_loader))
    
    return splits
