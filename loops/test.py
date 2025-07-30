import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import pandas as pd

import sys
sys.path.append('..')
from utils.metrics import ClassificationMetrics, print_metrics

class ModelTester:

    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = 'cpu',
                 config: Optional[Dict] = None):
        
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
        self.config = {
            'save_predictions': True,
            'save_confusion_matrix': True,
            'save_roc_curves': True,
            'save_embeddings': False,
            'output_dir': 'results/',
            'class_names': ['Healthy', 'Generalized Seizures', 'Focal Seizures', 'Seizure Events'],
            'confidence_threshold': 0.5
        }
        
        if config:
            self.config.update(config)
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.test_metrics = ClassificationMetrics()
        
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.embeddings = []
    
    def test(self) -> Dict[str, Any]:
        """Run complete testing and evaluation."""
        print(f"Starting testing on {self.device}")
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # predictions and probabilities
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                self.predictions.extend(preds.cpu().numpy())
                self.targets.extend(target.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())
                
                self.test_metrics.update(output, target)
        
        final_metrics = self.test_metrics.compute()
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print_metrics(final_metrics)
        
        analysis_results = self._generate_detailed_analysis()
        
        # Save results
        if self.config['save_predictions']:
            self._save_predictions()
        
        return {
            'metrics': final_metrics,
            'analysis': analysis_results,
            'predictions': np.array(self.predictions),
            'targets': np.array(self.targets),
            'probabilities': np.array(self.probabilities)
        }
    
    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        analysis = {}
        
        if self.config['save_confusion_matrix']:
            analysis['confusion_matrix'] = self._create_confusion_matrix()
        
        analysis['classification_report'] = classification_report(
            self.targets, self.predictions, 
            target_names=self.config['class_names'],
            output_dict=True
        )
        
        if self.config['save_roc_curves']:
            analysis['roc_curves'] = self._create_roc_curves()
        
        analysis['per_class_analysis'] = self._per_class_analysis()
        
        analysis['error_analysis'] = self._error_analysis()
        
        return analysis
    
    def _create_confusion_matrix(self) -> np.ndarray:
        cm = confusion_matrix(self.targets, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config['class_names'],
                   yticklabels=self.config['class_names'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        save_path = os.path.join(self.config['output_dir'], 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def _create_roc_curves(self) -> Dict[str, float]:
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(12, 8))
        
        for i in range(len(self.config['class_names'])):
            fpr[i], tpr[i], _ = roc_curve(targets == i, probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], lw=2,
                    label=f'{self.config["class_names"][i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        save_path = os.path.join(self.config['output_dir'], 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def _per_class_analysis(self) -> Dict[str, Any]:
        analysis = {}
        
        for i, class_name in enumerate(self.config['class_names']):
            class_mask = np.array(self.targets) == i
            class_preds = np.array(self.predictions)[class_mask]
            class_probs = np.array(self.probabilities)[class_mask]
            
            analysis[class_name] = {
                'total_samples': np.sum(class_mask),
                'correct_predictions': np.sum(class_preds == i),
                'accuracy': np.sum(class_preds == i) / np.sum(class_mask),
                'mean_confidence': np.mean(class_probs[:, i]),
                'std_confidence': np.std(class_probs[:, i]),
                'max_confidence': np.max(class_probs[:, i]),
                'min_confidence': np.min(class_probs[:, i])
            }
        
        return analysis
    
    def _error_analysis(self) -> Dict[str, Any]:
        errors = np.array(self.predictions) != np.array(self.targets)
        error_indices = np.where(errors)[0]
        
        analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(self.predictions),
            'error_distribution': {}
        }
        
        for i, class_name in enumerate(self.config['class_names']):
            class_errors = np.sum((np.array(self.targets) == i) & errors)
            analysis['error_distribution'][class_name] = class_errors
        
        error_patterns = []
        for idx in error_indices:
            true_class = self.targets[idx]
            pred_class = self.predictions[idx]
            error_patterns.append((true_class, pred_class))
        
        from collections import Counter
        pattern_counts = Counter(error_patterns)
        analysis['common_errors'] = pattern_counts.most_common(5)
        
        return analysis
    
    def _save_predictions(self):
        results_df = pd.DataFrame({
            'true_label': self.targets,
            'predicted_label': self.predictions,
            'confidence': np.max(self.probabilities, axis=1),
            'correct': np.array(self.predictions) == np.array(self.targets)
        })
        
        for i, class_name in enumerate(self.config['class_names']):
            results_df[f'prob_{class_name}'] = np.array(self.probabilities)[:, i]
        
        save_path = os.path.join(self.config['output_dir'], 'predictions.csv')
        results_df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    
    def plot_confidence_distribution(self):
        correct_mask = np.array(self.predictions) == np.array(self.targets)
        confidences = np.max(self.probabilities, axis=1)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_mask], bins=20, alpha=0.7, label='Correct', density=True)
        plt.hist(confidences[~correct_mask], bins=20, alpha=0.7, label='Incorrect', density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for i, class_name in enumerate(self.config['class_names']):
            class_mask = np.array(self.targets) == i
            if np.sum(class_mask) > 0:
                class_confidences = confidences[class_mask]
                plt.hist(class_confidences, bins=15, alpha=0.7, label=class_name, density=True)
        
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence by Class')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config['output_dir'], 'confidence_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_test_report(self, save_path: Optional[str] = None):
        if save_path is None:
            save_path = os.path.join(self.config['output_dir'], 'test_report.txt')
        
        with open(save_path, 'w') as f:
            f.write("EEG CLASSIFICATION TEST\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("BASIC METRICS:\n")
            f.write("-" * 20 + "\n")
            metrics = self.test_metrics.compute()
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("PER-CLASS ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            per_class = self._per_class_analysis()
            for class_name, stats in per_class.items():
                f.write(f"\n{class_name}:\n")
                for stat, value in stats.items():
                    f.write(f"  {stat}: {value:.4f}\n")
            
            f.write("\nERROR ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            error_analysis = self._error_analysis()
            f.write(f"Total errors: {error_analysis['total_errors']}\n")
            f.write(f"Error rate: {error_analysis['error_rate']:.4f}\n")
            f.write("\nCommon errors:\n")
            for (true_class, pred_class), count in error_analysis['common_errors']:
                f.write(f"  {self.config['class_names'][true_class]} -> {self.config['class_names'][pred_class]}: {count}\n")
        
        print(f"Test report saved to {save_path}")

def test_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu',
               config: Optional[Dict] = None) -> ModelTester:
    tester = ModelTester(model, test_loader, device, config)
    results = tester.test()
    return tester

def load_and_test_model(model_path: str, model: nn.Module, test_loader: DataLoader,
                       device: str = 'cpu', config: Optional[Dict] = None) -> ModelTester:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tester = ModelTester(model, test_loader, device, config)
    results = tester.test()
    
    return tester
