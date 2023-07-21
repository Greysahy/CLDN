#!/usr/bin/env python
import argparse
import builtins
import os
import random
import time
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import models.resnet
import models.vgg
import NCLD.loader
import NCLD.builder


parser = argparse.ArgumentParser(description='NCLD')
parser.add_argument('--data', default='data/rafdb',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=135, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.25, type=float,
                    help='softmax temperature (default: 0.07)')


def main():
    args = parser.parse_args()
    args.num_class = 7
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    best_acc = 0.0
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    model = NCLD.builder.SSFL(
        models.vgg.VggFeatures, 
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.num_class)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    epochs = 75

    cudnn.benchmark = True

    # Data path
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')

    vgg_augmentation = [
        transforms.RandomResizedCrop(96, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.TenCrop(80),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
    ]
    vgg_k_augmentation = [
        transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
        transforms.RandomApply([NCLD.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.TenCrop(80),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
    ]
    valtest_augmentation = [
        transforms.TenCrop(80),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
    ]
    
    train_dataset = datasets.ImageFolder(
        traindir,NCLD.loader.TwoCropsTransform(transforms.Compose(vgg_augmentation),transforms.Compose(vgg_k_augmentation))
    )
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose(valtest_augmentation)
    )
    test_dataset = datasets.ImageFolder(
        testdir, transforms.Compose(valtest_augmentation)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    
    for epoch in range(epochs):
        # train for one epoch
        acc_tr, loss_tr = train(train_loader, model, criterion, optimizer, args)
        acc_v = evaluate(val_loader, model, args)
        scheduler.step(acc_v)
        
        if acc_v > best_acc:
            best_acc = acc_v
            torch.save(model.state_dict(), f'checkpoints/rafdb/stage_i/best_model.pth')
            
        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')

    acc_t = evaluate(testloader, model, args)
    print('Test Accuracy: %2.4f %%' % acc_t)


def train(train_loader, model, criterion, optimizer, args, tencrops = True):
    # switch to train mode
    model.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    
    for images, label in tqdm(train_loader):
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
        
        if tencrops is True:
            bs, ncrops, c, h, w = images[0].shape
            images[0] =  images[0].view(-1, c, h, w)
            images[1] =  images[1].view(-1, c, h, w)
            label = torch.repeat_interleave(label, repeats=ncrops, dim=0)
        
        # compute output
        pred, cont_loss = model(im_q=images[0], im_k=images[1], labels=label)
        ce_loss = criterion(pred, label)
        loss = ce_loss + cont_loss

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_tr += loss.item()

        _, preds = torch.max(pred.data, 1)
        correct_count += (preds == label).sum().item()
        n_samples += label.size(0)

    acc = 100 * correct_count / n_samples  # 训练准确率
    loss = loss_tr / n_samples

    return acc, loss


def evaluate(dataloader, model, args=None, tencrops=False):
    model.eval()
    class_cnt = [0.0] * args.num_class
    class_cnt_right = [0.0] * args.num_class
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            
            if tencrops is True:
                bs, ncrops, c, h, w = images.shape
                images =  images.view(-1, c, h, w)
        
            logits = model(images)
            
            if tencrops is True:
                logits = logits.view(bs, ncrops, -1)
                logits = torch.sum(logits, dim=1) / ncrops
            
            _, preds = torch.max(logits.data, dim=1)
            
            correct_count += (preds == labels).sum().item()
            n_samples += labels.size(0)

            for i in range(args.num_class):
                class_cnt[i] += (labels == i).sum().item()
                class_cnt_right[i] += ((preds == labels) & (labels == i)).sum().item()

        for i in range(args.num_class):
            acc_ = None
            if class_cnt[i] == 0:
                acc_ = 0
            else:
                acc_ = class_cnt_right[i] / class_cnt[i]
            print(f'class{i} has {class_cnt[i]} samples, {100 * acc_}% accuracy')
    
    acc = 100 * correct_count / n_samples
    return acc


if __name__ == '__main__':
    main()