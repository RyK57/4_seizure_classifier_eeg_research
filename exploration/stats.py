import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    return df[channel_cols].describe()

def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    
    class_counts = df['y'].value_counts().sort_index()
    
    class_percentages = (class_counts / len(df) * 100).round(2)
    
    print("Class Distribution:")
    for class_label, count in class_counts.items():
        percentage = class_percentages[class_label]
        print(f"Class {class_label}: {count} samples ({percentage}%)")
    
    return class_counts

def get_channel_correlations(df: pd.DataFrame) -> pd.DataFrame:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    return df[channel_cols].corr()

def get_missing_values(df: pd.DataFrame) -> pd.Series:

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)
    
    missing_info = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    })
    
    missing_info = missing_info[missing_info['Missing_Count'] > 0]
    
    if missing_info.empty:
        print("no missing")
        return pd.Series(dtype=float)
    
    print("Missing Values Summary:")
    print(missing_info)
    
    return missing_counts

def get_skew_kurtosis(df: pd.DataFrame) -> pd.DataFrame:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    skewness = df[channel_cols].skew()
    kurtosis = df[channel_cols].kurtosis()
    
    stats_df = pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis
    })
    
    return stats_df

def get_outlier_counts(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
    channel_cols = [col for col in df.columns if col.startswith('X')]

    outlier_counts = {}
    
    for col in channel_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_counts[col] = (z_scores > z_thresh).sum()
    
    outlier_series = pd.Series(outlier_counts)
    
    print(f"Outlier Counts (Z-score > {z_thresh}):")
    print(outlier_series)
    
    return outlier_series

def get_classwise_stats(df: pd.DataFrame) -> Dict:

    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    class_stats = {}
    
    for class_label in sorted(df['y'].unique()):
        class_data = df[df['y'] == class_label][channel_cols]
        class_stats[f'Class_{class_label}'] = {
            'count': len(class_data),
            'mean': class_data.mean().to_dict(),
            'std': class_data.std().to_dict(),
            'min': class_data.min().to_dict(),
            'max': class_data.max().to_dict()
        }
    
    return class_stats

def get_channel_ranges(df: pd.DataFrame) -> pd.Series:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    
    ranges = df[channel_cols].max() - df[channel_cols].min()
    
    print("Channel Ranges (Max - Min):")
    print(ranges)
    
    return ranges

def get_channel_variances(df: pd.DataFrame) -> pd.Series:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    variances = df[channel_cols].var()
    
    print("Channel Variances:")
    print(variances)
    
    return variances

def get_top_n_variance_channels(df: pd.DataFrame, n: int = 5) -> List[str]:
    channel_cols = [col for col in df.columns if col.startswith('X')]

    variances = df[channel_cols].var()
    top_channels = variances.nlargest(n).index.tolist()
    
    print(f"Top {n} Channels by Variance:")
    for i, channel in enumerate(top_channels, 1):
        variance = variances[channel]
        print(f"{i}. {channel}: {variance:.2f}")
    
    return top_channels

def get_channel_means_by_class(df: pd.DataFrame) -> pd.DataFrame:
    
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    return df.groupby('y')[channel_cols].mean()

def get_channel_stds_by_class(df: pd.DataFrame) -> pd.DataFrame:
    
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    
    return df.groupby('y')[channel_cols].std()

def get_data_summary(df: pd.DataFrame) -> Dict:
    summary = {}
    
    summary['total_samples'] = len(df)
    summary['total_channels'] = len([col for col in df.columns if col.startswith('X')])
    summary['classes'] = sorted(df['y'].unique()) if 'y' in df.columns else []
    
    if 'y' in df.columns:
        summary['class_distribution'] = df['y'].value_counts().sort_index().to_dict()
    
    summary['missing_values'] = df.isnull().sum().sum()
    summary['missing_percentage'] = (summary['missing_values'] / (len(df) * len(df.columns))) * 100
    
    channel_cols = [col for col in df.columns if col.startswith('X')]
    if channel_cols:
        summary['channel_stats'] = {
            'mean_range': (df[channel_cols].mean().min(), df[channel_cols].mean().max()),
            'std_range': (df[channel_cols].std().min(), df[channel_cols].std().max()),
            'variance_range': (df[channel_cols].var().min(), df[channel_cols].var().max())
        }
    
    print("Dataset Summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total channels: {summary['total_channels']}")
    print(f"Classes: {summary['classes']}")
    print(f"Missing values: {summary['missing_values']} ({summary['missing_percentage']:.2f}%)")
    
    if 'class_distribution' in summary:
        print("\nClass Distribution:")
        for class_label, count in summary['class_distribution'].items():
            percentage = (count / summary['total_samples']) * 100
            print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    return summary
