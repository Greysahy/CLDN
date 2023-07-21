#!/usr/bin/env python
import argparse
import builtins
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import models.vgg
import models.tau_norm_classifier
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
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
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
parser.add_argument('--teacher_ckpt', type=str, default="checkpoints/affectnet/stage_i/best_model.pth")


def main():
    args = parser.parse_args()
    args.num_class = 7
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    best_acc_h = 0.0
    best_acc_s = 0.0
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    model = NCLD.builder.SDL(
        models.vgg.VggFeatures,
        models.vgg.VggFeatures,
        models.tau_norm_classifier.tau_norm_classifier,
        args.num_class, args.teacher_ckpt)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    
    # define loss function (criterion) and optimizer
    criterion_h = nn.CrossEntropyLoss().cuda(args.gpu)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    epochs = 75

    cudnn.benchmark = True

    # Data loading code
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

    valtest_augmentation = [
        transforms.TenCrop(80),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
    ]

    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose(vgg_augmentation)
    )
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose(valtest_augmentation)
    )
    test_dataset = datasets.ImageFolder(
        testdir, transforms.Compose(valtest_augmentation)
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    for epoch in range(epochs):
        # train for one epoch
        acc_tr_h, acc_tr_s, loss_tr = train(train_loader, model, criterion_h, optimizer, args)
        acc_v_h, acc_v_s = evaluate(val_loader, model, args)
        scheduler.step(acc_t_h)
        
        if acc_v_h > best_acc_h:
            best_acc_h = acc_t_h
            torch.save(model.state_dict(), 'checkpoints/rafdb/stage_ii/best_model.pth')
        if acc_v_s > best_acc_s:
            best_acc_s = acc_t_s
            
        print('Epoch %2d' % (epoch + 1))
        print('Train Accuracy hard: %2.4f %%' % acc_tr_h,
              'Train Accuracy soft: %2.4f %%' % acc_tr_s,
              sep='\t\t')
        print('Val Accuracy hard: %2.4f %%' % acc_v_h,
              'Val Accuracy soft: %2.4f %%' % acc_v_s,
              sep='\t\t')
        
    acc_t = evaluate(testloader, model, args)
    print('Test Accuracy: %2.4f %%' % acc_t)

def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()
    loss_tr, correct_count_h, correct_count_s , n_samples = 0.0, 0.0, 0.0, 0.0

    for images, label in tqdm(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
        
        bs, ncrops, c, h, w = images.shape
        images =  images.view(-1, c, h, w)
        label = torch.repeat_interleave(label, repeats=ncrops, dim=0)
        
        # compute output
        dkd_alpha = 1
        dkd_beta = 1
        dkd_temperature = 1
        pred_h, pred_s, kd_loss = model(images, label, dkd_temperature)
        
        ce_loss = criterion(pred_h, label)
        loss = ce_loss + kd_loss

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_tr += loss.item()

        _, preds_h = torch.max(pred_h.data, 1)
        _, preds_s = torch.max(pred_s.data, 1)
        correct_count_h += (preds_h == label).sum().item()
        correct_count_s += (preds_s == label).sum().item()
        n_samples += label.size(0)

    acc_h = 100 * correct_count_h / n_samples
    acc_s = 100 * correct_count_s / n_samples
    loss = loss_tr / n_samples

    return acc_h, acc_s, loss

def evaluate(dataloader, model, args=None):
    model.eval()
    
    class_cnt = [0.0] * args.num_class
    class_cnt_right_h = [0.0] * args.num_class
    class_cnt_right_s = [0.0] * args.num_class
    loss_tr, correct_count_h, correct_count_s , n_samples = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            
            bs, ncrops, c, h, w = images.shape
            images =  images.view(-1, c, h, w)
            v_label = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
            
            nkl_alpha = 1
            nkl_beta = 1
            nkl_temperature = 1
            pred_h, pred_s = model(images, labels, nkl_alpha, nkl_beta, nkl_temperature)
            
            pred_h = pred_h.view(bs, ncrops, -1)
            pred_h = torch.sum(pred_h, dim=1) / ncrops
            pred_s = pred_s.view(bs, ncrops, -1)
            pred_s = torch.sum(pred_s, dim=1) / ncrops
            
            _, preds_h = torch.max(pred_h.data, 1)
            _, preds_s = torch.max(pred_s.data, 1)
            
            
            correct_count_h += (preds_h == labels).sum().item()
            correct_count_s += (preds_s == labels).sum().item()
            n_samples += labels.size(0)

            for i in range(7):
                class_cnt[i] += (labels == i).sum().item()
                class_cnt_right_h[i] += ((preds_h == labels) & (labels == i)).sum().item()
            for i in range(7):
                class_cnt_right_s[i] += ((preds_s == labels) & (labels == i)).sum().item()
                
        for i in range(7):
            print(f'class{i} has {class_cnt[i]} Samples, {100 * class_cnt_right_h[i] / class_cnt[i]}% hard Accuracy, {100 * class_cnt_right_s[i] / class_cnt[i]}% soft Accuracy')
   
    acc_h = 100 * correct_count_h / n_samples
    acc_s = 100 * correct_count_s / n_samples
    loss = loss_tr / n_samples
    sum_v /= vcnt
    
    return acc_h, acc_s


if __name__ == '__main__':
    main()
