import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
from sklearn.cluster import KMeans

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

        assert self.split in {'train', 'val', 'test'}
        
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

        images, boxes, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, boxes, labels
    
##########################################################################
def calculate_anchor_sizes(ground_truth_boxes, num_anchors_per_layer):
    anchor_sizes = []
    
    for num_anchors in num_anchors_per_layer:

        widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
        heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
        
        features = torch.stack((widths, heights), dim=1)
        
        features_np = features.numpy()
        
        kmeans = KMeans(n_clusters=num_anchors)
        kmeans.fit(features_np)
        
        anchor_sizes.append(kmeans.cluster_centers_)
    
    return anchor_sizes