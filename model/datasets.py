import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

# Heavily modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
class FaceMaskDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    # def __init__(self, data_folder, split, output_info, keep_difficult=False):
    def __init__(self, images, bnd_boxes, labels, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.lower()

        assert self.split in {'train', 'test'}
        
        self.images = images
        self.bnd_boxes = bnd_boxes
        self.labels = labels
    
    
    def __getitem__(self, i):
        # Read image
        image = self.images[i]
        boxes = torch.FloatTensor(self.bnd_boxes[i])
        labels = torch.FloatTensor(self.labels[i])

        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels
    

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        # difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            # difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        # return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
        return images, boxes, labels