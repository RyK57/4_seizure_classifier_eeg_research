import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, Any
import warnings
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    def __init__(self, scaler_type: str = 'standard', apply_pca: bool = False,
                 n_components: Optional[int] = None, remove_outliers: bool = False,
                 outlier_threshold: float = 3.0):
        
        self.scaler_type = scaler_type
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        self._initialize_scalers()
        
        self.is_fitted = False
        self.feature_names = None
        self.n_features_original = None
    
    def _initialize_scalers(self):
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"type : {self.scaler_type}")
        
        if self.apply_pca:
            self.pca = PCA(n_components=self.n_components)
    
    def fit(self, data: pd.DataFrame) -> 'EEGPreprocessor':
        print(f"{self.scaler_type} scaler")
        
        feature_cols = [col for col in data.columns if col.startswith('X')]
        self.feature_names = feature_cols
        self.n_features_original = len(feature_cols)
        
        X = data[feature_cols].values
        
        if self.remove_outliers:
            X = self._remove_outliers(X)
            print(f"remaining samples: {len(X)}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.apply_pca:
            X_pca = self.pca.fit_transform(X_scaled)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Total explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        feature_cols = [col for col in data.columns if col.startswith('X')]
        
        X = data[feature_cols].values
        
        if self.remove_outliers:
            X = self._remove_outliers(X)
        
        X_scaled = self.scaler.transform(X)
        
        if self.apply_pca:
            X_transformed = self.pca.transform(X_scaled)
            feature_cols = [f'PC_{i+1}' for i in range(X_transformed.shape[1])]
        else:
            X_transformed = X_scaled
        
        transformed_data = pd.DataFrame(X_transformed, columns=feature_cols)
        
        if 'y' in data.columns:
            transformed_data['y'] = data['y'].values
        
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
    
    def _remove_outliers(self, X: np.ndarray) -> np.ndarray:
        z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
        outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
        return X[~outlier_mask]
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        info = {
            'scaler_type': self.scaler_type,
            'apply_pca': self.apply_pca,
            'n_features_original': self.n_features_original,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info['feature_names'] = self.feature_names
            
            if self.apply_pca:
                info['n_components'] = self.pca.n_components_
                info['explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
                info['total_explained_variance'] = self.pca.explained_variance_ratio_.sum()
        
        return info

class EEGTransform:
    
    def __init__(self, scaler=None, apply_pca=False, pca=None):
        self.scaler = scaler
        self.apply_pca = apply_pca
        self.pca = pca
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        if self.apply_pca and self.pca is not None:
            features = self.pca.transform(features.reshape(1, -1)).flatten()
        
        return features

def create_preprocessing_pipeline(data_path: str, scaler_type: str = 'standard',
                                apply_pca: bool = False, n_components: Optional[int] = None,
                                remove_outliers: bool = False, outlier_threshold: float = 3.0,
                                test_size: float = 0.2, val_size: float = 0.2,
                                batch_size: int = 32, random_state: int = 42) -> Tuple:
    """
    arguments L 
        data_path: Path to the CSV file
        scaler_type: Type of scaler to use
        apply_pca: Whether to apply PCA
        n_components: Number of PCA components
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        batch_size: Batch size for data loaders
        random_state: Random state for reproducibility
        
  output
        Tuple of (preprocessor, train_loader, val_loader, test_loader)
    """
    from .dataset import EEGDataModule
    
    data = pd.read_csv(data_path)
    
    preprocessor = EEGPreprocessor(
        scaler_type=scaler_type,
        apply_pca=apply_pca,
        n_components=n_components,
        remove_outliers=remove_outliers,
        outlier_threshold=outlier_threshold
    )
    
    transformed_data = preprocessor.fit_transform(data)
    
    transformed_path = data_path.replace('.csv', '_transformed.csv')
    transformed_data.to_csv(transformed_path, index=False)
    print(f"Transformed data saved to {transformed_path}")
    
    data_module = EEGDataModule(
        data_path=transformed_path,
        test_size=test_size,
        val_size=val_size,
        batch_size=batch_size,
        random_state=random_state
    )
    
    return preprocessor, *data_module.get_data_loaders()

def normalize_features(data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    feature_cols = [col for col in data.columns if col.startswith('X')]
    
    if method == 'zscore':
        data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()
    elif method == 'minmax':
        data[feature_cols] = (data[feature_cols] - data[feature_cols].min()) / (data[feature_cols].max() - data[feature_cols].min())
    elif method == 'robust':
        Q1 = data[feature_cols].quantile(0.25)
        Q3 = data[feature_cols].quantile(0.75)
        IQR = Q3 - Q1
        data[feature_cols] = (data[feature_cols] - data[feature_cols].median()) / IQR
    else:
        raise ValueError(f"Unknown : {method}")
    
    return data

def get_feature_importance(data: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    feature_cols = [col for col in data.columns if col.startswith('X')]
    
    correlations = data[feature_cols].corrwith(data[target_col]).abs()
    
    variances = data[feature_cols].var()
    
    mi_scores = mutual_info_classif(data[feature_cols], data[target_col])
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Correlation': correlations,
        'Variance': variances,
        'Mutual_Info': mi_scores,
        'Importance_Score': (correlations + variances/max(variances) + mi_scores/max(mi_scores)) / 3
    })
    
    return importance_df.sort_values('Importance_Score', ascending=False)