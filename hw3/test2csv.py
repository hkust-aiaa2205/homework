import os
import pandas as pd
from PIL import Image
from dataset import MyDataset
from models import VideoResNet
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define aug
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalization
])

# Load test dataset
test_dataset = MyDataset("/root/hkustgz-aiaa-5032-hw3/hw3_16fpv", 
"/root/hw-private-main/hw3/test.csv", 
stage="test", ratio=0.2, transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Length of test loader: {len(test_loader)}")

# Load model
model = VideoResNet(num_classes=10).cuda()
model.load_state_dict(torch.load('/root/hw-private-main/hw3/ResNet18_last.pth'))

# Load video ID
fread = open("/root/hw-private-main/hw3/hkustgz-aiaa-5032-hw3/test_for_student.label", "r")
video_ids = [os.path.splitext(line.strip())[0] for line in fread.readlines()]

# Val stage
model.eval()
result = []
with torch.no_grad():
    for data in tqdm(test_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        result.extend(predicted.cpu().numpy())

# Save result
with open('result_ResNet18_3D.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines(f"{video_ids[i]},{pred_class}\n")
