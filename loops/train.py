import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import time
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sys
sys.path.append('..')
from utils.losses import get_loss_function
from utils.metrics import ClassificationMetrics, EarlyStopping, print_metrics
from utils.optimizers import OptimizerConfig, create_optimizer_scheduler

class ModelTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: Optional[DataLoader] = None, device: str = 'cpu',
                 config: Optional[Dict] = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.config = {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'step',
            'loss_type': 'cross_entropy',
            'weight_decay': 1e-5,
            'batch_size': 32,
            'early_stopping_patience': 10,
            'save_best_model': True,
            'model_save_path': 'models/',
            'log_interval': 10,
            'use_class_weights': False,
            'gradient_clipping': None,
            'mixed_precision': False,
            'scheduler_params': {},
            'loss_params': {},
            'optimizer_params': {}
        }
        
        if config:
            self.config.update(config)
        
        self._initialize_training_components()
        
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'metrics': []
        }
        
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'metrics': []
        }
        
        self.best_val_metric = 0.0
        self.best_epoch = 0
        
        os.makedirs(self.config['model_save_path'], exist_ok=True)
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, loss function, and metrics."""
        
        class_weights = None
        if self.config['use_class_weights']:
            class_weights = self._calculate_class_weights()
        
        loss_params = self.config['loss_params'].copy()
        if class_weights is not None:
            loss_params['weight'] = class_weights.to(self.device)
        
        self.criterion = get_loss_function(
            self.config['loss_type'], 
            **loss_params
        )
        
        optimizer_config = OptimizerConfig(
            optimizer_type=self.config['optimizer_type'],
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            scheduler_type=self.config['scheduler_type'],
            **self.config['optimizer_params']
        )
        
        self.optimizer, self.scheduler = create_optimizer_scheduler(
            self.model, optimizer_config
        )
        
        self.train_metrics = ClassificationMetrics()
        self.val_metrics = ClassificationMetrics()
        
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )
        
        if self.config['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _calculate_class_weights(self) -> torch.Tensor:
        class_counts = torch.zeros(4)  
        
        for _, targets in self.train_loader:
            for target in targets:
                class_counts[target] += 1
        
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        return class_weights.float()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.config['gradient_clipping']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config['gradient_clipping'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                
                if self.config['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config['gradient_clipping'])
                
                self.optimizer.step()
            
            epoch_loss += loss.item()
            self.train_metrics.update(output, target)
            
            if batch_idx % self.config['log_interval'] == 0:
                current_metrics = self.train_metrics.compute()
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_metrics.get("accuracy", 0):.4f}'
                })
        
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        epoch_loss = 0.0
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                epoch_loss += loss.item()
                self.val_metrics.update(output, target)
        
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return epoch_metrics
    
    def train(self) -> Dict[str, List]:
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Configuration: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate_epoch()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', float('inf')))
                else:
                    self.scheduler.step()
            
            self.train_history['loss'].append(train_metrics.get('loss', 0))
            self.train_history['accuracy'].append(train_metrics.get('accuracy', 0))
            self.train_history['metrics'].append(train_metrics)
            
            if val_metrics:
                self.val_history['loss'].append(val_metrics.get('loss', 0))
                self.val_history['accuracy'].append(val_metrics.get('accuracy', 0))
                self.val_history['metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)
            
            if val_metrics:
                current_metric = val_metrics.get('f1_macro', val_metrics.get('accuracy', 0))
                if self.early_stopping(current_metric, self.model):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.best_epoch = epoch
                    if self.config['save_best_model']:
                        self._save_model('best_model.pth')
            
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation metric: {self.best_val_metric:.4f} at epoch {self.best_epoch+1}")
        
        self._save_model('final_model.pth')
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_val_metric
        }
    
    def _print_epoch_summary(self, epoch: int, train_metrics: Dict, 
                           val_metrics: Dict, epoch_time: float):
        print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} ({epoch_time:.2f}s)")
        print("-" * 50)
        
        print("Training Metrics:")
        print_metrics(train_metrics)
        
        if val_metrics:
            print("\nValidation Metrics:")
            print_metrics(val_metrics)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\nLearning Rate: {current_lr:.6f}")
    
    def _save_model(self, filename: str):
        save_path = os.path.join(self.config['model_save_path'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_val_metric,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.config['model_save_path'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }, checkpoint_path)
    
    def load_model(self, model_path: str):
        """Load model from saved state."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        if 'val_history' in checkpoint:
            self.val_history = checkpoint['val_history']
        
        print(f"Model loaded from {model_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        axes[0, 0].plot(epochs, self.train_history['loss'], label='Train Loss')
        if self.val_history['loss']:
            axes[0, 0].plot(epochs, self.val_history['loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_history['accuracy'], label='Train Accuracy')
        if self.val_history['accuracy']:
            axes[0, 1].plot(epochs, self.val_history['accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        if self.train_history['metrics']:
            train_f1 = [m.get('f1_macro', 0) for m in self.train_history['metrics']]
            axes[1, 0].plot(epochs, train_f1, label='Train F1')
            if self.val_history['metrics']:
                val_f1 = [m.get('f1_macro', 0) for m in self.val_history['metrics']]
                axes[1, 0].plot(epochs, val_f1, label='Val F1')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                config: Optional[Dict] = None, device: str = 'cpu') -> ModelTrainer:
    trainer = ModelTrainer(model, train_loader, val_loader, device, config)
    trainer.train()
    return trainer