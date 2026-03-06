import torch.nn as nn

class SimpleCNNv2(nn.Module):
    """
    Simple CNN network for image classification.

    Architecture:
        3 convolutional blocks (Conv → BatchNorm → ReLU → Pool)
        AdaptiveAvgPool at the end → produces a fixed 4×4 feature map regardless of the input size
        Classifier: Linear(1024, 256) → ReLU → Dropout → Linear(256, num_classes)
        
    Input:  (B, 3, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int):
        super(SimpleCNNv2, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # /2
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # /2
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                 # → 64×4×4
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x