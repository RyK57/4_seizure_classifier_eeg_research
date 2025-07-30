import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union
import math

class ResNet_EEG(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 16, 
                 depth: str = 'resnet18', dropout_rate: float = 0.5):
        super(ResNet_EEG, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        layer_sizes = [64, 128, 256, 512]
        
        # Initial layer
        self.input_layer = nn.Linear(input_channels, layer_sizes[0])
        
        # Residual blocks
        self.layer1 = self._make_residual_block(layer_sizes[0], layer_sizes[1])
        self.layer2 = self._make_residual_block(layer_sizes[1], layer_sizes[2])
        self.layer3 = self._make_residual_block(layer_sizes[2], layer_sizes[3])
        
        # Final layers
        self.final_layer = nn.Linear(layer_sizes[3], layer_sizes[3])
        self.classifier = nn.Linear(layer_sizes[3], num_classes)
        
        self._initialize_weights()
    
    def _make_residual_block(self, in_features: int, out_features: int) -> nn.Module:
        return ResidualBlock(in_features, out_features, self.dropout_rate)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.relu(self.final_layer(x))
        x = self.classifier(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.5):
        super(ResidualBlock, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class ResNet_EEG_Simplified(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 16, dropout_rate: float = 0.5):
        super(ResNet_EEG_Simplified, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.features = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(64, 128, dropout_rate),
            ResidualBlock(128, 256, dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def resnet18_eeg(num_classes: int = 4, input_channels: int = 16, **kwargs) -> ResNet_EEG:
    return ResNet_EEG(num_classes, input_channels, depth='resnet18', **kwargs)

def resnet34_eeg(num_classes: int = 4, input_channels: int = 16, **kwargs) -> ResNet_EEG:
    return ResNet_EEG(num_classes, input_channels, depth='resnet34', **kwargs)

def resnet50_eeg(num_classes: int = 4, input_channels: int = 16, **kwargs) -> ResNet_EEG:
    return ResNet_EEG(num_classes, input_channels, depth='resnet50', **kwargs)

def create_resnet1d(model_type: str = 'resnet18', num_classes: int = 4, 
                   input_channels: int = 16, simplified: bool = False, **kwargs) -> nn.Module:
    if simplified:
        return ResNet_EEG_Simplified(num_classes, input_channels, **kwargs)
    
    model_functions = {
        'resnet18': resnet18_eeg,
        'resnet34': resnet34_eeg,
        'resnet50': resnet50_eeg,
    }

    return model_functions[model_type](num_classes, input_channels, **kwargs)