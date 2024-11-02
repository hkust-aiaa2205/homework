import torch
import torch.nn as nn
import torchvision.models as models

class VideoResNet(nn.Module):
    def __init__(self, num_classes):
        super(VideoResNet, self).__init__()
        # 加载预训练的 ResNet-3D 模型
        self.r3d_18 = models.video.r3d_18(pretrained=True)
        
        # 冻结除了最后一个全连接层以外的所有参数
        for name, param in self.r3d_18.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # 替换最后的全连接层以适应新的分类任务
        self.r3d_18.fc = nn.Linear(self.r3d_18.fc.in_features, num_classes)
    
    def forward(self, x):
        # 前向传播
        return self.r3d_18(x)
