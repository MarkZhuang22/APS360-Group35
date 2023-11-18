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
        print("Please download data first")
        sys.exit()
    return data_loader

def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    size = [int(ele.text) for ele in root.find('size')]
    bnd_label = [obj.find('name').text for obj in root.findall('object')]
    bndbox = [[int(ele.text) for ele in obj.find('bndbox')] for obj in root.findall('object')]
    return size, bnd_label, bndbox
'''
def loadxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    size = []
    bnd_label = []
    bndbox = []

    for i in range(len(root)):
        if root[i].tag == 'size':
            for ele in root[i]:
                size.append(int(ele.text))
        elif root[i].tag == 'object':
            for ele in root[i]:
                if ele.tag == 'name':
                    bnd_label.append(ele.text)  
                elif ele.tag == 'bndbox':
                    bnd = []
                    for ele_ in ele:
                        bnd.append(int(ele_.text))
                    bndbox.append(bnd)
    return [size, bnd_label, bndbox]


def data_loader(dataloader, i):
    # set_trace()
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    
    path = path[:-3]+'xml'
    imgsize, boxlabel, bndbox = loadxml(path)
    # set_trace()
    if (bndbox == []) | (imgsize == []) | (boxlabel == []) | (0 in imgsize):
        return [None, None, None, None]             

    return img, bndbox, boxlabel
    '''
def data_loader(dataloader, i):
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    path = path[:-3]+'xml'
    imgsize, boxlabel, bndbox = loadxml(path)
    if not all([bndbox, imgsize, boxlabel]) or 0 in imgsize:
        return [None, None, None, None]
    return img, bndbox, boxlabel

def retrieve_gt(path, split, limit=0):
    assert split in ["train", "val", "test"]
    path = os.path.join(path, "FaceMaskDataset")
    dataloader = load_data(data_dir=path)
    images, bndboxes, boxlabels = [], [], []

    start_index = {"train": 0, "val": 800, "test": 8000}[split]
    end_index = start_index + (limit if limit else { "train": 6000, "val": 1000, "test": 1000 }[split])
    if(split == 'val'):
        end_index = start_index+ 100
    for i in range(start_index, end_index):
        img, bndbox, boxlabel = data_loader(dataloader, i)
        if img is None:
            continue
        boxlabel = [2 if label == "with_mask" else 1 for label in boxlabel]
        images.append(img)
        bndboxes.append(bndbox)
        boxlabels.append(boxlabel)

    print("Finish retrieving data")
    return images, bndboxes, boxlabels
'''
def retrieve_gt(path, split, limit=0):
    assert split in ["train", "val", "test"]
    path = os.path.join(path, "FaceMaskDataset")
    dataloader = load_data(data_dir=path)
    images, bndboxes, boxlabels = [], [], []
    train_size = 6000
    val_index = train_size+1000
    test_index = val_index+1000
    if split == "train":
        i = 0
        N = train_size if not limit else limit
    elif split == "val":
        i = 700 #use 100 for initial val
        N = i + 100 if limit else val_index
    elif split == "test":
        i = 8000
        N = i + limit if limit else test_index

    while i < N:
        img, bndbox, boxlabel = data_loader(dataloader, i)
        i += 1
        if img is None:
            continue
        boxlabel = [2 if label == "with_mask" else 1 for label in boxlabel]
        images.append(img)
        bndboxes.append(bndbox)
        boxlabels.append(boxlabel)

    print("finish retrieving data")
    return images, bndboxes, boxlabels

'''
if __name__=="__main__":
    images, bndboxes, boxlabels = retrieve_gt("../FaceMaskDataset/", "val")
