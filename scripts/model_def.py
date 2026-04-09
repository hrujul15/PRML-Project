from torch.optim import lr_scheduler
import torch.nn as nn
import torch
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # FEATURE EXTRACTOR: Learns spatial patterns (edges, textures, shapes)
        self.features = nn.Sequential(
            # Layer 1: Conv -> ReLU -> Pool
            # Input: 3 channels (RGB), 32x32 image. Output: 16 channels, 32x32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            # MaxPool reduces dimensions by half. Output: 16 channels, 16x16 image
            nn.MaxPool2d(kernel_size=5, stride=5),

            # Layer 2: Conv -> ReLU -> Pool
            # Input: 16 channels, 16x16. Output: 32 channels, 16x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # Output: 32 channels, 8x8 image
            nn.MaxPool2d(kernel_size=5, stride=5)
        )

        # CLASSIFIER: Takes the extracted features and makes a decision
        self.classifier = nn.Sequential(
            
            nn.Linear(4160, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.67), # Drops 67% of neurons randomly to prevent memorization (overfitting)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)         # 1. Pass through convolutional layers
        x = torch.flatten(x, 1)      # 2. Flatten from 2D grids into a 1D vector
        x = self.classifier(x)       # 3. Pass through dense/linear layers
        return x