import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, MultiStepLR
from typing import Dict, Any, Optional, Union
import math

def get_optimizer(optimizer_type: str = 'adam', **kwargs) -> torch.optim.Optimizer:
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'rprop': optim.Rprop
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers[optimizer_type](**kwargs)

def get_scheduler(scheduler_type: str = 'step', optimizer=None, **kwargs):
    schedulers = {
        'step': StepLR,
        'cosine': CosineAnnealingLR,
        'reduce_on_plateau': ReduceLROnPlateau,
        'onecycle': OneCycleLR,
        'exponential': ExponentialLR,
        'multistep': MultiStepLR
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    if scheduler_type == 'step':
        if 'step_size' not in kwargs:
            kwargs['step_size'] = 10
    if scheduler_type == 'cosine':
        if 'T_max' not in kwargs:
            kwargs['T_max'] = 50
    if scheduler_type == 'onecycle':
        if 'max_lr' not in kwargs:
            raise ValueError("OneCycleLR requires 'max_lr' argument.")
        if 'steps_per_epoch' not in kwargs or 'epochs' not in kwargs:
            raise ValueError("OneCycleLR requires 'steps_per_epoch' and 'epochs' arguments.")
    if scheduler_type == 'exponential':
        if 'gamma' not in kwargs:
            kwargs['gamma'] = 0.9
    if scheduler_type == 'multistep':
        if 'milestones' not in kwargs:
            kwargs['milestones'] = [30, 80]
        if 'gamma' not in kwargs:
            kwargs['gamma'] = 0.1

    return schedulers[scheduler_type](optimizer, **kwargs)

class OptimizerConfig:
    def __init__(
        self,
        optimizer_type: str = 'adam',
        lr: float = 0.001,
        weight_decay: float = 0.0,
        scheduler_type: str = 'step',
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
    ):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

    def create_optimizer(self, model_params) -> torch.optim.Optimizer:
        return get_optimizer(
            self.optimizer_type,
            params=model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs
        )

    def create_scheduler(self, optimizer) -> Any:
        return get_scheduler(self.scheduler_type, optimizer, **self.scheduler_kwargs)

def create_optimizer_scheduler(model, config: OptimizerConfig):
    optimizer = config.create_optimizer(model.parameters())
    scheduler = config.create_scheduler(optimizer)
    return optimizer, scheduler

def get_adam_config(lr: float = 0.001, weight_decay: float = 1e-5) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer_type='adam',
        lr=lr,
        weight_decay=weight_decay,
        scheduler_type='step',
        optimizer_kwargs={},
        scheduler_kwargs={'step_size': 30, 'gamma': 0.1}
    )

def get_adamw_config(lr: float = 0.001, weight_decay: float = 0.01) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer_type='adamw',
        lr=lr,
        weight_decay=weight_decay,
        scheduler_type='cosine',
        T_max=100
    )

def get_sgd_config(lr: float = 0.01, momentum: float = 0.9) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer_type='sgd',
        lr=lr,
        momentum=momentum,
        scheduler_type='multistep',
        milestones=[30, 60, 90],
        gamma=0.1
    )

def get_onecycle_config(lr: float = 0.001, epochs: int = 100) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer_type='adam',
        lr=lr,
        scheduler_type='onecycle',
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=1
    )

class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

class CustomScheduler:
    def __init__(self, optimizer, warmup_epochs: int = 5, total_epochs: int = 100,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int):
        """Update learning rate."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def get_optimizer_summary(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    summary = {
        'type': type(optimizer).__name__,
        'param_groups': len(optimizer.param_groups),
        'parameters': {}
    }
    
    for i, param_group in enumerate(optimizer.param_groups):
        summary['parameters'][f'group_{i}'] = {
            'lr': param_group['lr'],
            'weight_decay': param_group.get('weight_decay', 0),
            'momentum': param_group.get('momentum', 0),
            'betas': param_group.get('betas', (0.9, 0.999))
        }
    
    return summary