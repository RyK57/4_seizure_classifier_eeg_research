import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet_EEG(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 16, 
                 dropout_rate: float = 0.5):
        super(AlexNet_EEG, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  
        
        x = self.classifier(x)  
        
        return x

def create_alexnet1d(num_classes: int = 4, input_channels: int = 16, 
                     simplified: bool = False, **kwargs) -> nn.Module:
    return AlexNet_EEG(num_classes, input_channels, **kwargs)