import torch
import torch.nn as nn
import torchvision.models as models

class VideoResNet(nn.Module):
    def __init__(self, num_classes):
        super(VideoResNet, self).__init__()
        # Load pre-trained ResNet-3D models
        self.r3d_18 = models.video.r3d_18(pretrained=True)
        
        # Freeze all parameters except the last fully connected layer
        for name, param in self.r3d_18.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # Replace the last fully connected layer to accommodate the new classification task
        self.r3d_18.fc = nn.Linear(self.r3d_18.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.r3d_18(x)
