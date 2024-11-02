import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from dataset import MyDataset
from models import VideoResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

val_dataset = MyDataset('/root/hkustgz-aiaa-5032-hw3/hw3_16fpv', '/root/hkustgz-aiaa-5032-hw3/trainval.csv', stage="val", ratio=0.2, transform=transforms)
train_dataset = MyDataset('/root/hkustgz-aiaa-5032-hw3/hw3_16fpv', '/root/hkustgz-aiaa-5032-hw3/trainval.csv', stage="train", ratio=0.2, transform=transforms)

print('dataset loaded')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print('train ', len(train_loader))
print('val ', len(val_loader))

model = VideoResNet(num_classes=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("model loaded")

best_acc_train = 0
best_acc_val = 0
counter = 0  # Initialize counter to track epochs since last improvement

print('start training')
for epoch in range(50):
    start_time = time.time() 
    running_loss_train = 0.0
    running_loss_val = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0

    model.train()  # Set the model to train mode
    print('train ...')
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_train = criterion(outputs, labels)
        loss_train.backward()
        optimizer.step()
        running_loss_train += loss_train.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        print('val ...')
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
            val_outputs = model(val_inputs)
            loss_val = criterion(val_outputs, val_labels)
            running_loss_val += loss_val.item()
            _, predicted_val = torch.max(val_outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted_val == val_labels).sum().item()
    
    acc_train = correct_train / total_train
    acc_val = correct_val / total_val
    
    print(f'acc train   {acc_train}       {correct_train}/{total_train}')
    print(f'acc val     {acc_val}         {correct_val}/{total_val}')

    if acc_train > best_acc_train:
        best_acc_train = acc_train
        torch.save(model.state_dict(), './ResNet18_best_train.pth')

    if acc_val > best_acc_val:
        best_acc_val = acc_val
        torch.save(model.state_dict(), './ResNet18_best_val.pth')
  
    torch.save(model.state_dict(), './ResNet18_last.pth')
