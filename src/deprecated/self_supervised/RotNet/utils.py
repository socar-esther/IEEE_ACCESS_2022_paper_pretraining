import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

import os
import shutil
import numpy as np
from statistics import mean
import copy

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import *


def create_dataloader(path, batch_size, istrain):
    
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )
    train_transformer = transforms.Compose([
            transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3
            transforms.ToTensor(),
            normalize
        ])

    test_transformer = transforms.Compose([
            transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3
            transforms.ToTensor(),
            normalize
        ])
    
    if istrain:
        data = datasets.ImageFolder(path, transform=train_transformer)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        
    else:
        data = datasets.ImageFolder(path, transform=test_transformer)
        dataloader = DataLoader(data, shuffle=False)

    return dataloader, data


def calculate_metrics(trues, preds):
    accuracy = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    auc = 0
    precision = precision_score(trues, preds, average='macro')
    recall = recall_score(trues, preds, average='macro')

    return accuracy, f1, auc, precision, recall

def train(dataloader, net, learning_rate, weight_decay_level, device):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay_level,
    )

    net.train()

    train_losses = list()
    train_preds = list()
    train_trues = list()

    for idx, (img, label) in enumerate(dataloader):

        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        out = net(img)

        _, pred = torch.max(out, 1)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_trues.extend(label.view(-1).cpu().numpy().tolist())
        train_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())
        
    acc, f1, auc, prec, rec = calculate_metrics(train_trues, train_preds)
    print(confusion_matrix(train_trues, train_preds))

    return net, acc, prec, rec, f1


def test(dataloader, net, device):

    net.eval()
    test_losses = list()
    test_trues = list()
    test_preds = list()

    for idx, (img, label) in enumerate(dataloader):

        img = img.to(device)
        label = label.to(device)

        out = net(img)
        
        softmax_out = torch.softmax(out, 1)
        softmax_max = softmax_out.detach().cpu().numpy().max()

        _, pred = torch.max(out, 1) # make prediction

        criterion = torch.nn.CrossEntropyLoss()
        
        loss = criterion(out, label)

        test_losses.append(loss.item())
        test_trues.extend(label.view(-1).cpu().numpy().tolist())
        test_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())


    acc, f1, auc, prec, rec = calculate_metrics(test_trues, test_preds)
    print(confusion_matrix(test_trues, test_preds))

    return net, acc, prec, rec, f1