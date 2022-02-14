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

import random
import pytorch_lightning as pl
from torch.nn.modules.loss import _WeightedLoss
import torch.optim as optim
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Precision, Recall, F1

import logging
from datetime import datetime
from downstream_modules.accida_classification_utils import BasicConv, resnet50, count_parameters, jigsaw_generator

from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger 

from downstream_modules.downstream_accida_classification import *
from downstream_modules.accida_classification_utils import *


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


def load_model(model_name, loss, learning_rate, batch_size, num_workers, regularizer, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
            
        if upstream_weight_type == 'naive':
            print('>> upstream : naive')
            net = resnet50(pretrained=False)
        
        elif upstream_weight_type == 'car_class' :
            print('>> upstream : car_class classification')
            net = resnet50(pretrained = False)
            
            in_features = net.fc.in_features
            
            upstream_weight_path = '10class_imagenet_acc_0.91796.pth'
            
            net.fc = nn.Sequential(
                nn.Linear(
                    in_features,
                    10 # due to the number of classes at 10_class_classifciation
                )
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path), 
                                          map_location = device))
            net.to(device)
            print('Successfully Loaded the Weight!') 

        elif upstream_weight_type == 'imagenet':
            print('>> upstream : Imagenet')
            net = resnet50(pretrained=True)

        elif upstream_weight_type == 'stanford-car':
            print('>> upstream : stanford_car')
            upstream_weight_path = 'resnet50_acc91_stanfordcar_pretrain_true.pth'

            if os.path.exists(os.path.join('upstream_artifacts/', upstream_weight_path)):
                pass

            else:
                print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')

                if not os.path.exists('upstream_artifacts'):
                    os.makedirs('upstream_artifacts')

                os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/' + upstream_weight_path + ' ./upstream_artifacts/.')
                #time.sleep(5) # wait until the download is completed

            net = resnet50(pretrained=False)
            in_features = net.fc.in_features


            net.fc = nn.Linear(
                in_features,
                196 # due to the number of classes at stanford-car
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))

            print('Successfully Loaded the Weight!')

        elif upstream_weight_type == 'byol':
            print('>> upstream : byol')
            upstream_weight_path = '2class_resnet50_acc_70_byol_pretrain_true.pth'

            net = resnet50(pretrained=False)

            in_features = net.fc.in_features

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))
            print('Successfully Loaded the Weight!')

        elif upstream_weight_type == 'rotation':
            print('>> upstream : rotnet')
                
            if dataset == '2_class_classification':    
                upstream_weight_path = '2class_resnet50_acc90_rotnet_pretrain_true.pth'


            if os.path.exists(os.path.join('upstream_artifacts/', upstream_weight_path)):
                pass

            else:
                print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')

                if not os.path.exists('upstream_artifacts'):
                    os.makedirs('upstream_artifacts')

                os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/' + upstream_weight_path + ' ../upstream_artifacts/.')
                #time.sleep(5) # wait until the download is completed

            net = resnet50(pretrained=False)
            in_features = net.fc.in_features

            net.fc = nn.Sequential(
                nn.Linear(
                    in_features,
                    4 # due to the number of classes at stanford-car
                )
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))
                    
            print('Successfully Loaded the Weight!')   

        net = PMG(net, loss=loss, feature_size = 512, classes_num = CLASSES, batch_size=batch_size,
                  num_workers=num_workers, lr = learning_rate, reg=regularizer, root='custom') ## this should work right?

    return net
