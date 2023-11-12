


import xml.etree.ElementTree as ET
import torch
import cv2
import os
import numpy as np

#from matplotlib import pyplot as plt
#import plotly.express as px
#import plotly.graph_objects as go
import sys 

from six.moves import urllib
import requests
from pdb import set_trace
import glob
# from download_dataset import download_extract


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

    

def load_data(data_dir = "./", batch_size = 1):
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # }

    # image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
    files = glob.glob(data_dir)
    try:
        image_dataset = datasets.ImageFolder(os.path.join(data_dir)) 
        data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    except:
        print("Please download data with download_dataset.py first")
        sys.exit()
    return data_loader

def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    size = []
    bnd_label = []
    bndbox = []
    difficult = []
    for i in range(len(root)):
        if root[i].tag == 'size':
            for ele in root[i]:
                size.append(int(ele.text))
        elif root[i].tag == 'object':
            for ele in root[i]:
                if ele.tag == 'name':
                    bnd_label.append(ele.text)
                elif ele.tag == 'difficult':
                    difficult.append((int)(ele.text))    
                elif ele.tag == 'bndbox':
                    bnd = []
                    for ele_ in ele:
                        bnd.append(int(ele_.text))
                    bndbox.append(bnd)
    return [size, bnd_label, bndbox, difficult]

def resize_img(img, imgsize, min_size = 600, max_size = 1000):
    H, W = imgsize
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = cv2.resize(img.permute(1,2,0).numpy(), (int(W*scale), int(H*scale)))
    return img, scale

def resize_box(bbox, in_size, out_size):
    bbox = np.array(bbox).copy()
    y_scale = float(out_size[0]) / int(in_size[0])
    x_scale = float(out_size[1]) / int(in_size[1])
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def data_loader(dataloader, i):
    # set_trace()
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    
    path = path[:-3]+'xml'
    imgsize, boxlabel, bndbox, difficult = loadxml(path)
    # set_trace()
    if (bndbox == []) | (imgsize == []) | (boxlabel == []) | (0 in imgsize):
        return [None, None, None, None]             

    return img, bndbox, boxlabel, difficult

def retrieve_gt(path, split, limit=0):

  assert split in ["train", "val", "test"]
  # download_extract(path)
  
  path = path + "/FaceMaskDataset"
  # path = os.path.join(path, "/FaceMaskDataset")
  dataloader = load_data(data_dir = path)
  images = []
  bndboxes = []
  boxlabels = []
  difficults = []
  # set_trace()
  print(limit)
  if split == "train":
    i = 0
    N = 6000
    if limit:
        N= limit
  elif split == "val":
    i = 700
    #i = 6000
    #N = 6240
    if limit:
        N = i + 200
  elif split == "test":
    i = 6240
    N = 7959

    if limit:
        N = i + limit

  while i < N:
    img, bndbox, boxlabel, difficult = data_loader(dataloader, i)
    print(i)
    i += 1
    if img is None:
        continue
    boxlabel = [2 if label == "with_mask" else 1 for label in boxlabel]
    images.append(img)
    bndboxes.append(bndbox)
    boxlabels.append(boxlabel)
    difficults.append(difficult)

        
  print("finish retrieving data")
  
  # boxlabel = ["face", "face_masks"]
  return images, bndboxes, boxlabels, difficults

if __name__=="__main__":
    images, bndboxes, boxlabels, difficults = retrieve_gt("../FaceMaskDataset/", "val")
