import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt
import numpy as np
import argparse
# from model import SSD300, MultiBoxLoss
from model import SSD300, MultiBoxLoss
from config import Config
from datasets import FaceMaskDataset
from utils import *
from dataload import retrieve_gt
#from eval import evaluate
from pdb import set_trace
from dataload import retrieve_gt
import os

cudnn.benchmark = True

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Slightly modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
def train(config, train_dataset, val_dataset, model, optimizer, start_epoch):
    """
    Training.
    """
    # global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint

    # Move to default device
    model = model.to(config.device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(config.device)
    
    # set_trace()
    # Custom dataloaders
                                     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=config.workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    epochs = config.epochs

    batch_train_losses = []
    batch_val_losses = []
    # Epochs
    print("start training....")
    for epoch in range(start_epoch, epochs):


        # One epoch's training
        batch_train_losses.append(train_one_epoch(config, train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch))
        batch_val_losses.append(val_one_epoch(config, val_loader=val_loader, 
              model=model, 
              criterion=criterion, 
              epoch=epoch))
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, config.checkpoint)

    x = np.arange(1, len(batch_train_losses) + 1)
    fig = plt.figure()
    plt.subplot(121)
    plt.plot(x, batch_train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")

    x = np.arange(1, len(batch_val_losses) + 1)
    plt.subplot(122)
    plt.plot(x, batch_val_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    fig.savefig("train_val_losses.jpg")
    

# Slightly modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    loss_sum = 0
    
    # Batches
    # for i, (images, boxes, labels, _) in enumerate(train_loader):
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(config.device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(config.device) for b in boxes]
        labels = [l.to(config.device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        loss_sum += loss

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return loss_sum.item() / len(train_loader)

# Heavily modified from train_one_epoch of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
def val_one_epoch(config, val_loader, model, criterion, epoch):
    """
    One epoch's training.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    
    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    loss_sum = 0

    with torch.no_grad():
        # Batches
        # for i, (images, boxes, labels, _) in enumerate(train_loader):
        for i, (images, boxes, labels) in enumerate(val_loader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(config.device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(config.device) for b in boxes]
            labels = [l.to(config.device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            loss_sum += loss


            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))
        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return loss_sum.item() / len(val_loader)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FaceMaskDetection")
    
    parser.add_argument('--dest', type=str, default="./FaceMaskDataset/", help='path to dataset.')
    parser.add_argument('--limit', type=int, default=0, help='limit number of images.')
    parser.add_argument('--epochs', type=int, default=30, help='limit number of images.')
    parser.add_argument('--device', type=int, default=0, help='limit number of images.')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint_ssd300.pth.tar', help='limit number of images.')
    parser.add_argument('--batch_size', type=int, default=16, help='limit number of images.')


    args = parser.parse_args()
    config = Config()
    config.epochs = args.epochs
    # config.device = args.device
    config.checkpoint = args.checkpoint
    config.batch_size = args.batch_size
 
    # Training Phase
    if config.checkpoint is None or not os.path.exists(config.checkpoint):
        start_epoch = 0
        model = SSD300(n_classes=config.n_classes)
        # set_trace()
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * config.lr}, {'params': not_biases}],
                                    lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    print("loading training images")
    
    images, bnd_boxes, labels, difficults = retrieve_gt(args.dest, "train", limit=args.limit)
    print("%d images has been retrieved" %len(images))
    # set_trace()
    
    
    print("finish loading images")
    
    train_dataset = FaceMaskDataset(images, bnd_boxes, labels, "train")

    print("loading val images")
    
    images, bnd_boxes, labels, difficults = retrieve_gt(args.dest, "val", limit=args.limit)
    print("%d images has been retrieved" %len(images))
    # set_trace()
    
    
    print("finish loading images")

    val_dataset = FaceMaskDataset(images, bnd_boxes, labels, "train")
    
    train(config, train_dataset, val_dataset, model, optimizer, start_epoch)
