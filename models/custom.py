import torch 
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 16, dropout_rate: float = 0.25):
        super(CustomModel, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(input_channels, 64)
        self.res1 = ResidualBlock(64, 64, dropout_rate)
        self.fc2 = nn.Linear(64, 128)
        self.res2 = ResidualBlock(128, 128, dropout_rate)
        self.fc3 = nn.Linear(128, 256)
        self.res3 = ResidualBlock(256, 256, dropout_rate)
        self.fc4 = nn.Linear(256, 512)
        self.res4 = ResidualBlock(512, 512, dropout_rate)
        self.fc5 = nn.Linear(512, 1048)
        self.res5 = ResidualBlock(1048, 1048, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Linear(1048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res3(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res4(x)

        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res5(x)

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