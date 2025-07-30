import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 4):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        return torch.mean(torch.sum(-targets_one_hot * log_probs, dim=1))

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, num_classes: int = 4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.alpha * ce + (1 - self.alpha) * focal

def get_loss_function(loss_type: str = 'cross_entropy', **kwargs) -> nn.Module:
    if loss_type == 'cross_entropy':
        weight = kwargs.pop('weight', None)
        return nn.CrossEntropyLoss(weight=weight, **kwargs)
    
    elif loss_type == 'focal':
        kwargs.pop('weight', None)
        alpha = kwargs.pop('alpha', None)
        gamma = kwargs.pop('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
    
    elif loss_type == 'label_smoothing':
        kwargs.pop('weight', None)
        smoothing = kwargs.pop('smoothing', 0.1)
        num_classes = kwargs.pop('num_classes', 4)
        return LabelSmoothingLoss(smoothing=smoothing, num_classes=num_classes, **kwargs)
    
    elif loss_type == 'dice':
        smooth = kwargs.pop('smooth', 1e-6)
        num_classes = kwargs.pop('num_classes', 4)
        return DiceLoss(smooth=smooth, num_classes=num_classes, **kwargs)
    
    elif loss_type == 'combined':
        alpha = kwargs.pop('alpha', 0.5)
        gamma = kwargs.pop('gamma', 2.0)
        class_weights = kwargs.pop('class_weights', None)
        return CombinedLoss(alpha=alpha, gamma=gamma, class_weights=class_weights, **kwargs)
    
    elif loss_type == 'mse':
        return nn.MSELoss(**kwargs)
    
    elif loss_type == 'l1':
        return nn.L1Loss(**kwargs)
    
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss(**kwargs)
    
    else:
        raise ValueError(f"Error")

def calculate_class_weights(data_loader) -> torch.Tensor:
    class_counts = torch.zeros(4)  
    
    for _, targets in data_loader:
        for target in targets:
            class_counts[target] += 1
    
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return class_weights.float()

class WeightedCrossEntropyLoss(nn.Module):
    
    def __init__(self, data_loader):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = calculate_class_weights(data_loader)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(inputs, targets)