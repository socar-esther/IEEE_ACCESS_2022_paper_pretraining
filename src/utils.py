import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
# For our model
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
    
    # Normalization 정의
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )

    ## Training 단계에서 사용하는 image augmentation 기법들
    train_transformer = transforms.Compose([

            # resizing
    #         transforms.Resize(256), # option 1 
    #         transforms.CenterCrop(256), # option 2 
            transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3
    #         transforms.RandomResizedCrop(256),

            # augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),

            # normalize
            normalize
        ])

    ## Test 단계에서 사용하는 image augmentation 기법들
    # Note; Test 단게에서는 Random Flip 등은 사용하면 안됨
    # 왜? input을 랜덤하게 잘라주면 randomness에 의해서 inference 결과가 달라질 수 있기 때문
    test_transformer = transforms.Compose([

    #         transforms.Resize(300),
    #         transforms.CenterCrop(256),
            transforms.Resize((256,256), interpolation=Image.NEAREST), # option 3

            transforms.ToTensor(),

            # normalize
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
#     try:
    accuracy = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
#     auc = roc_auc_score(trues, preds)
    auc = 0
    precision = precision_score(trues, preds, average='macro')
    recall = recall_score(trues, preds, average='macro')

#     except:
#         accuracy, f1, auc, precision, recall = -1, -1, -1, -1, -1

    return accuracy, f1, auc, precision, recall

def train(dataloader, net, learning_rate, weight_decay_level, device):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr = learning_rate, 
        weight_decay = weight_decay_level
    )

    net.train()

    train_losses = list()
    train_preds = list()
    train_trues = list()

    for idx, (img, label) in enumerate(dataloader):

        img = img.to(device)
        label = label.to(device)

        # zero the parameter gradient
        optimizer.zero_grad()

        out = net(img)
#         print('out: ', out.shape)

        _, pred = torch.max(out, 1)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_trues.extend(label.view(-1).cpu().numpy().tolist())
        train_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())

#         if idx % 50 == 0:
#             print('idx: ', idx)

    acc, f1, auc, prec, rec = calculate_metrics(train_trues, train_preds)

#     print('> Number of Training Set; ', len(train_trues))
#     print(len(train_preds))

#     print(train_trues[:10])
#     print(train_preds[:10])

    print('\n''====== Training Metrics ======')
    print('Loss: ', mean(train_losses))
    print('Acc: ', acc)
    print('F1: ', f1)
    print('Prec: ', prec)
    print('Rec: ', rec)
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

        _, pred = torch.max(out, 1) # make prediction

        criterion = torch.nn.CrossEntropyLoss()
        
        loss = criterion(out, label)

        test_losses.append(loss.item())
        test_trues.extend(label.view(-1).cpu().numpy().tolist())
        test_preds.extend(pred.view(-1).cpu().detach().numpy().tolist())

#         if idx % 50 == 0:
#             print('idx: ', idx)


    acc, f1, auc, prec, rec = calculate_metrics(test_trues, test_preds)

#     print(len(test_trues))
#     print(len(test_preds))

#     print(test_trues[:10])
#     print(test_preds[:10])

    print('====== Test Metrics ======')
    print('Test Loss: ', mean(test_losses))
    print('Test Acc: ', acc)
    print('Test F1: ', f1)
    print('Test Prec: ', prec)
    print('Test Rec: ', rec)

    print(confusion_matrix(test_trues, test_preds))

    return net, acc, prec, rec, f1