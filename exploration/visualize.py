import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_class_distribution(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    class_counts = df['y'].value_counts().sort_index()
    sns.countplot(data=df, x='y', ax=ax1, palette='viridis')
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Label')
    ax1.set_ylabel('Count')
    
    for i, count in enumerate(class_counts):
        ax1.text(i, count + max(class_counts) * 0.01, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
    ax2.pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_channel_distributions(df: pd.DataFrame, channels: Optional[List[str]] = None, 
                             figsize: Tuple[int, int] = (15, 10)) -> None:
    if channels is None:
        channels = [col for col in df.columns if col.startswith('X')]
    
    n_channels = len(channels)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_channels > 1 else [axes]
    
    for i, channel in enumerate(channels):
        if i < len(axes):
            sns.histplot(data=df, x=channel, kde=True, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'{channel} Distribution', fontweight='bold')
            axes[i].set_xlabel('Amplitude')
            axes[i].set_ylabel('Frequency')
            
            mean_val = df[channel].mean()
            std_val = df[channel].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'+1σ: {mean_val + std_val:.2f}')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8, label=f'-1σ: {mean_val - std_val:.2f}')
            axes[i].legend()
    
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    corr_matrix = df[channel_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('EEG Channel Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_class_comparison_boxplots(df: pd.DataFrame, channels: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
    if channels is None:
        channels = [col for col in df.columns if col.startswith('X')][:8]
    
    n_channels = len(channels)
    n_cols = 2
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_channels > 1 else [axes]
    
    for i, channel in enumerate(channels):
        if i < len(axes):
            sns.boxplot(data=df, x='y', y=channel, ax=axes[i], palette='Set3')
            axes[i].set_title(f'{channel} by Class', fontweight='bold')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Amplitude')
    
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_eeg_waveforms(df: pd.DataFrame, sample_indices: Optional[List[int]] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    
    if sample_indices is None:
        sample_indices = list(range(min(4, len(df))))
    
    n_samples = len(sample_indices)
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, sample_idx in enumerate(sample_indices):
        if i < len(axes) and sample_idx < len(df):
            sample_data = df.iloc[sample_idx][channel_cols].values
            x_positions = np.arange(len(channel_cols))
            
            axes[i].plot(x_positions, sample_data, 'b-', linewidth=2, marker='o', markersize=4)
            axes[i].set_title(f'Sample {sample_idx} (Class: {df.iloc[sample_idx]["y"] if "y" in df.columns else "N/A"})', 
                            fontweight='bold')
            axes[i].set_xlabel('Channel')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(x_positions)
            axes[i].set_xticklabels(channel_cols, rotation=45)
            
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_channel_means_by_class(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)) -> None:

    
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    means_by_class = df.groupby('y')[channel_cols].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for class_label in means_by_class.index:
        ax1.plot(channel_cols, means_by_class.loc[class_label], 
                marker='o', linewidth=2, label=f'Class {class_label}')
    
    ax1.set_title('Mean Channel Values by Class', fontweight='bold')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Mean Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(channel_cols, rotation=45)
    
    # Heatmap
    sns.heatmap(means_by_class, annot=True, cmap='RdBu_r', center=0, ax=ax2, 
                cbar_kws={"shrink": .8})
    ax2.set_title('Mean Values Heatmap by Class', fontweight='bold')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Class')
    
    plt.tight_layout()
    plt.show()

def plot_variance_analysis(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 6)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    variances = df[channel_cols].var()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    bars = ax1.bar(channel_cols, variances, color='skyblue', alpha=0.7)
    ax1.set_title('Channel Variances', fontweight='bold')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Variance')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(channel_cols, rotation=45)
    
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(variances) * 0.01,
                f'{var:.1f}', ha='center', va='bottom', fontweight='bold')
    
    sorted_vars = variances.sort_values(ascending=False)
    bars2 = ax2.bar(range(len(sorted_vars)), sorted_vars, color='lightcoral', alpha=0.7)
    ax2.set_title('Channel Variances (Sorted)', fontweight='bold')
    ax2.set_xlabel('Channel Rank')
    ax2.set_ylabel('Variance')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(sorted_vars)))
    ax2.set_xticklabels(sorted_vars.index, rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_outlier_analysis(df: pd.DataFrame, z_thresh: float = 3.0,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    n_channels = len(channel_cols)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_channels > 1 else [axes]
    
    for i, channel in enumerate(channel_cols):
        if i < len(axes):
            z_scores = np.abs((df[channel] - df[channel].mean()) / df[channel].std())
            outliers = z_scores > z_thresh
            
            ax = axes[i]
            ax.scatter(range(len(df)), df[channel], c='blue', alpha=0.6, s=20, label='Normal')
            ax.scatter(np.where(outliers)[0], df[channel][outliers], 
                      c='red', s=30, label='Outliers', zorder=5)
            
            ax.set_title(f'{channel} - Outliers (Z>{z_thresh})', fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_skewness_kurtosis(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 6)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    skewness = df[channel_cols].skew()
    kurtosis = df[channel_cols].kurtosis()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    bars1 = ax1.bar(channel_cols, skewness, color='lightgreen', alpha=0.7)
    ax1.set_title('Channel Skewness', fontweight='bold')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Skewness')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Normal Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(channel_cols, rotation=45)
    ax1.legend()
    
    for bar, skew in zip(bars1, skewness):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                f'{skew:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    bars2 = ax2.bar(channel_cols, kurtosis, color='lightblue', alpha=0.7)
    ax2.set_title('Channel Kurtosis', fontweight='bold')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Kurtosis')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Normal Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels(channel_cols, rotation=45)
    ax2.legend()
    
    for bar, kurt in zip(bars2, kurtosis):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                f'{kurt:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_eeg_topography(df: pd.DataFrame, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 8)) -> None:
    channel_cols = [col for col in df.columns if col.startswith('X')]
    if sample_idx >= len(df):
        raise ValueError(f"Sample index {sample_idx} is out of range")
    
    sample_data = df.iloc[sample_idx][channel_cols].values
    
    grid_size = int(np.sqrt(len(channel_cols)))
    topography = sample_data.reshape(grid_size, grid_size)
    
    plt.figure(figsize=figsize)
    
    im = plt.imshow(topography, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, label='Amplitude')
    
    for i in range(grid_size):
        for j in range(grid_size):
            channel_idx = i * grid_size + j
            if channel_idx < len(channel_cols):
                plt.text(j, i, f'{channel_cols[channel_idx]}\n{sample_data[channel_idx]:.1f}',
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.title(f'EEG Topography - Sample {sample_idx} (Class: {df.iloc[sample_idx]["y"] if "y" in df.columns else "N/A"})', 
              fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_class_separation_analysis(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
    
    channel_cols = [col for col in df.columns if col.startswith('X')]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    means_by_class = df.groupby('y')[channel_cols].mean()
    mean_diff = means_by_class.max() - means_by_class.min()
    
    bars1 = ax1.bar(channel_cols, mean_diff, color='lightcoral', alpha=0.7)
    ax1.set_title('Max Mean Difference Between Classes', fontweight='bold')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Mean Difference')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(channel_cols, rotation=45)
    
    stds_by_class = df.groupby('y')[channel_cols].std()
    for class_label in stds_by_class.index:
        ax2.plot(channel_cols, stds_by_class.loc[class_label], 
                marker='o', linewidth=2, label=f'Class {class_label}')
    
    ax2.set_title('Standard Deviation by Class', fontweight='bold')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels(channel_cols, rotation=45)
    
    cv_by_class = (stds_by_class / np.abs(means_by_class)) * 100
    for class_label in cv_by_class.index:
        ax3.plot(channel_cols, cv_by_class.loc[class_label], 
                marker='s', linewidth=2, label=f'Class {class_label}')
    
    ax3.set_title('Coefficient of Variation by Class (%)', fontweight='bold')
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('CV (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels(channel_cols, rotation=45)
    
    ranges_by_class = df.groupby('y')[channel_cols].max() - df.groupby('y')[channel_cols].min()
    for class_label in ranges_by_class.index:
        ax4.plot(channel_cols, ranges_by_class.loc[class_label], 
                marker='^', linewidth=2, label=f'Class {class_label}')
    
    ax4.set_title('Value Range by Class', fontweight='bold')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Range')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels(channel_cols, rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_eda_report(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    print("\n1. Class Distribution Analysis")
    plot_class_distribution(df)
    
    print("\n2. Channel Distribution Analysis")
    plot_channel_distributions(df)
    
    print("\n3. Channel Correlation Analysis")
    plot_correlation_heatmap(df)
    
    print("\n4. Class Comparison Analysis")
    plot_class_comparison_boxplots(df)
    
    print("\n5. EEG Waveform Analysis")
    plot_eeg_waveforms(df)
    
    print("\n6. Channel Means by Class")
    plot_channel_means_by_class(df)
    
    print("\n7. Variance Analysis")
    plot_variance_analysis(df)
    
    print("\n8. Outlier Analysis")
    plot_outlier_analysis(df)
    
    print("\n9. Distribution Shape Analysis")
    plot_skewness_kurtosis(df)
    
    print("\n10. EEG Topography")
    plot_eeg_topography(df)
    
    print("\n11. Class Separation Analysis")
    plot_class_separation_analysis(df)
    
    if save_path:
        print(f"Report saved to: {save_path}")