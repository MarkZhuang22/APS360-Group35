import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import transforms
import PIL
import urllib
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l
import os



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1_2 = nn.BatchNorm2d(16)
        self.relu1_2 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2_2 = nn.BatchNorm2d(32)
        self.relu2_2 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3_1 = nn.BatchNorm2d(64)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3_2 = nn.BatchNorm2d(64)
        self.relu3_2 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.maxpool(x)

        x = self.dropout(x)


        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).view(x.size(0), -1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).view(x.size(0), -1))))
        out = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        out = x * channel_out * spatial_out
        return out

class AnchorGenerator:
    def __init__(self):
        self.sizes = [[0.205, 0.2844],
                      [0.3855, 0.4667],
                      [0.5646, 0.6451],
                      [0.7386, 0.8291],
                      [0.9126, 0.9987]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) -1

def get_anchors(self):
        anchors = []
        for size in self.sizes:
            for ratio in self.ratios:
                w = size[0] * ratio
                h = size[1] / ratio
                anchors.append((w, h))
        return anchors

AnchorInfo = AnchorGenerator()
num_anchors = AnchorInfo.num_anchors
sizes = AnchorInfo.sizes
ratios = AnchorInfo.ratios

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)
        self.cbam = CBAM(out_channels, reduction_ratio = 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool(x)
        #x = self.cbam(x)

        return x

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels, reduction_ratio=4)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size= 1 ,stride=stride)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)
            residual = self.bn3(residual)

        out += residual
        out = self.cbam(out)

        return out

class small_SSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
            super(small_SSD, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.CNNmodel = CNN()
            cls_ouput_chanels = (num_anchors) * (num_classes + 1)
            bbox_ouput_chanels = (num_anchors) * 4
            #cls_ouput_chanels_2 = (num_anchors+0) * (num_classes + 1)
            #bbox_ouput_chanels_2 = (num_anchors+0) * 4
            self.cls_0 = nn.Conv2d(64, cls_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bbox_0 = nn.Conv2d(64, bbox_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.cls_1 = nn.Conv2d(128, cls_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bbox_1 = nn.Conv2d(128, bbox_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.cls_2 = nn.Conv2d(128, cls_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bbox_2 = nn.Conv2d(128, bbox_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.cls_3 = nn.Conv2d(128, cls_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bbox_3 = nn.Conv2d(128, bbox_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.cls_4 = nn.Conv2d(128, cls_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bbox_4 = nn.Conv2d(128, bbox_ouput_chanels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.block0 = ResNetBlock(64,128)
            self.block1 = ResNetBlock(128,128)
            self.block2 = ResNetBlock(128,128)
            self.adaptiveMaxPool = nn.AdaptiveMaxPool2d((1, 1))
    def forward(self, x):
            anchors = [None for _ in range(5)]
            cls_preds = [None for _ in range(5)]
            bbox_preds = [None for _ in range(5)]
            x = self.CNNmodel(x)
            anchors[0] = d2l.multibox_prior(x, sizes=sizes[0], ratios=ratios[0])
            cls_preds[0] = self.cls_0(x)
            bbox_preds[0] = self.bbox_0(x)
            x = self.block0(x)
            anchors[1] = d2l.multibox_prior(x, sizes=sizes[1], ratios=ratios[1])
            cls_preds[1] = self.cls_1(x)
            bbox_preds[1] = self.bbox_1(x)
            x = self.block1(x)
            anchors[2] = d2l.multibox_prior(x, sizes=sizes[2], ratios=ratios[2])
            cls_preds[2] = self.cls_2(x)
            bbox_preds[2] = self.bbox_2(x)
            x = self.block2(x)
            anchors[3] = d2l.multibox_prior(x, sizes=sizes[3], ratios=ratios[3])
            cls_preds[3] = self.cls_3(x)
            bbox_preds[3] = self.bbox_3(x)
            x = self.adaptiveMaxPool(x)
            anchors[4] = d2l.multibox_prior(x, sizes=sizes[4], ratios=ratios[4])
            cls_preds[4] = self.cls_4(x)
            bbox_preds[4] = self.bbox_4(x)

            anchors = torch.cat(anchors, dim=1)
            #cls_preds = concat_preds(cls_preds)
            concatenated_cls_preds = torch.cat([pred.permute(0, 2, 3, 1).flatten(start_dim=1) for pred in cls_preds], dim=1)
            concatenated_cls_preds = concatenated_cls_preds.reshape(concatenated_cls_preds.shape[0], -1, self.num_classes + 1)
           # bbox_preds = concat_preds(bbox_preds)
            concatenated_bbox_preds = torch.cat([pred.permute(0, 2, 3, 1).flatten(start_dim=1) for pred in bbox_preds], dim=1)
            return anchors, concatenated_cls_preds, concatenated_bbox_preds

net_SSSD = small_SSD(num_classes=1)

net_SSSD = small_SSD(num_classes=1)
X = torch.zeros((32, 3, 128, 128))
anchors, cls_preds, bbox_preds = net_SSSD(X)

print( anchors.shape)
print( cls_preds.shape)
print( bbox_preds.shape)

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size = cls_preds.shape[0]
    num_classes = cls_preds.shape[2]

    cls = cls_loss(cls_preds.reshape(-1, num_classes),cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)

    bbox = bbox_loss(bbox_preds * bbox_masks,bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def _getitem_(self, idx):
        return self.data[idx]


def trainnet(net,train_iter,lr=0.2,weight_decay=5e-4,num_epochs=10):
    #trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features, target
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            print(cls_eval(cls_preds, cls_labels)/cls_labels.numel(),'-',bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.numel())
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
data_dir = d2l.download_extract('banana-detection')

def read_data_bananas(is_train=True):

    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []

    for img_name, target in csv_data.iterrows():
        image_path = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')
        image_tensor= Image.open(image_path)
        images.append(image_tensor)

        # Here `target` contains (class, upper-left x, upper-left y, lower-right x, lower-right y),
        # where all the images have the same banana class (index 0)
        targets.append(list(target))

    return images, torch.from_numpy(np.expand_dims(np.array(targets), 1)) / 256.0

class BananasDataset(Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'Read {len(self.features)} {"training" if is_train else "validation"} examples')

    def __getitem__(self, idx):
        transform = transforms.Compose([
              transforms.ToTensor(),  # Convert the image to a tensor
              transforms.Resize(128), # change to 128
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        image = transform(self.features[idx])
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.features)
def load_data_bananas(batch_size):
    train_dataset = BananasDataset(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = BananasDataset(is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
net2 = small_SSD(num_classes=1)
anchors, cls_preds, bbox_preds = net2(batch[0])
bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, batch[1])
      # Calculate the loss function using the predicted and labeled values
      # of the classes and offsets
print(cls_eval(cls_preds, cls_labels)/cls_labels.numel(),'-',bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.numel())

net = small_SSD(num_classes=1)
 trainnet(net,train_iter)
