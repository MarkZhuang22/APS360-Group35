import os
import sys
import glob
from pdb import set_trace

import cv2
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir = "./", batch_size = 1):
#    files = glob.glob(data_dir)
    try:
        image_dataset = datasets.ImageFolder(os.path.join(data_dir)) 
        data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    except:
        print("Please download data")
        print(data_dir)
        sys.exit()
    return data_loader

def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    #size = [int(ele.text) for ele in root.find('size')]
   # print(size)
    bnd_label = [obj.find('name').text for obj in root.findall('object')]
    bndbox = [[int(ele.text) for ele in obj.find('bndbox')] for obj in root.findall('object')]
    return  bnd_label, bndbox

def data_loader(dataloader, i):
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    path = path[:-3]+'xml'
    boxlabel, bndbox = loadxml(path)
    if not all([bndbox, boxlabel]):
        return [None, None, None]
    return img, bndbox, boxlabel

def retrieve_gt(path, split, limit=0):
    assert split in ["train", "val", "test"]
    path = os.path.join(path, "FaceMaskDataset")
    dataloader = load_data(data_dir=path)
    train_path = os.path.join(path,"train")
    val_path = os.path.join(path,"val")
    t_len = len(os.listdir(train_path)) // 2
    v_len = len(os.listdir(val_path)) // 2
    images, bndboxes, boxlabels = [], [], []


    start_index = {"train": 0, "val": t_len, "test": t_len}[split]
    end_index = start_index + (limit if limit else { "train": t_len, "val": v_len, "test": v_len }[split])
    for i in range(start_index, end_index):
        img, bndbox, boxlabel = data_loader(dataloader, i)
        if img is None:
            continue
        boxlabel = [2 if label == "with_mask" or label == "face_mask" else 1 for label in boxlabel]
        images.append(img)
        bndboxes.append(bndbox)
        boxlabels.append(boxlabel)

    print("Finish retrieving data")
    return images, bndboxes, boxlabels

if __name__=="__main__":
    images, bndboxes, boxlabels = retrieve_gt("../FaceMaskDataset/", "val")
