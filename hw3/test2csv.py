import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from dataset import MyDataset
from models import VideoResNet
from tqdm import tqdm

# 检查cuda是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化操作
])

# 加载测试数据集
test_dataset = MyDataset("/root/hkustgz-aiaa-5032-hw3/hw3_16fpv", 
"/root/hw-private-main/hw3/test.csv", 
stage="test", ratio=0.2, transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Length of test loader: {len(test_loader)}")

# 加载模型
model = VideoResNet(num_classes=10).cuda()
model.load_state_dict(torch.load('/root/hw-private-main/hw3/ResNet18_last.pth'))

# 加载视频ID
fread = open("/root/hw-private-main/hw3/hkustgz-aiaa-5032-hw3/test_for_student.label", "r")
video_ids = [os.path.splitext(line.strip())[0] for line in fread.readlines()]

# 测试模型
model.eval()
result = []
with torch.no_grad():
    for data in tqdm(test_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        result.extend(predicted.cpu().numpy())

# 保存结果
with open('result_ResNet18_3D.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines(f"{video_ids[i]},{pred_class}\n")
