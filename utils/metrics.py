import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ClassificationMetrics:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        if torch.is_tensor(outputs):
            outputs = outputs.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        preds = np.argmax(outputs, axis=1)
        probs = torch.softmax(torch.tensor(outputs), dim=1).numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targets)
        self.probabilities.extend(probs)
    
    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        metrics['kappa'] = cohen_kappa_score(targets, predictions)
        metrics['mcc'] = matthews_corrcoef(targets, predictions)
        
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = precision_score(targets, predictions, average=None, zero_division=0)[i]
            metrics[f'recall_class_{i}'] = recall_score(targets, predictions, average=None, zero_division=0)[i]
            metrics[f'f1_class_{i}'] = f1_score(targets, predictions, average=None, zero_division=0)[i]
        
        try:
            if self.num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(targets, probabilities, multi_class='ovo', average='macro')
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        if not self.predictions:
            return np.array([])
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        if not self.predictions:
            return ""
        return classification_report(self.targets, self.predictions, zero_division=0)

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                     num_classes: int = 4) -> Dict[str, float]:
    metrics_calculator = ClassificationMetrics(num_classes)
    metrics_calculator.update(outputs, targets)
    return metrics_calculator.compute()

def calculate_epoch_metrics(model, data_loader, device: str = 'cpu') -> Dict[str, float]:
    model.eval()
    metrics_calculator = ClassificationMetrics()
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            metrics_calculator.update(outputs, targets)
    
    return metrics_calculator.compute()

def print_metrics(metrics: Dict[str, float], epoch: Optional[int] = None):
    if epoch is not None:
        print(f"\nEpoch {epoch} Metrics:")
    else:
        print("\nMetrics:")
    print("=" * 50)
    
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
    print(f"Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
    print(f"F1-Score (Macro): {metrics.get('f1_macro', 0):.4f}")
    print(f"Cohen's Kappa: {metrics.get('kappa', 0):.4f}")
    print(f"Matthews Correlation: {metrics.get('mcc', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    elif 'roc_auc_ovr' in metrics:
        print(f"ROC AUC (OVR): {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC AUC (OVO): {metrics['roc_auc_ovo']:.4f}")

def get_best_metric(metrics_history: List[Dict[str, float]], 
                   metric_name: str = 'f1_macro') -> Tuple[int, float]:
    best_value = -1
    best_epoch = -1
    
    for epoch, metrics in enumerate(metrics_history):
        if metric_name in metrics and metrics[metric_name] > best_value:
            best_value = metrics[metric_name]
            best_epoch = epoch
    
    return best_epoch, best_value

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model=None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)