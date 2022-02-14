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

import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

import os
import shutil
import pandas as pd
import numpy as np
from statistics import mean
import copy

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import *

from utils import *

import neptune


def create_base_dir(dataset):
    return os.path.join('../dataset/', dataset)

def create_dir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_downstream_classfier(dataset, upstream_weight_type, batch_size, n_epochs, learning_rate, weight_decay, device):
    
    neptune.init(
        'accida/sandbox',
        api_token =
    'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOWNlZmJkMDYtODI2Ny00NWM5LTkwZmQtYjUxMDFmM2FlYWU0In0='
    )

    # create the experiment
    experiment_object = neptune.create_experiment(
        name='socarvision',
    )
    neptune.append_tag(dataset)
    neptune.append_tag(upstream_weight_type)
    neptune.append_tag(learning_rate)
    
    dataset_path = create_base_dir(dataset)
    train_path = os.path.join(dataset_path, 'Training')
    test_path = os.path.join(dataset_path, 'Test')

    train_loader, _train_data = create_dataloader(train_path, batch_size, True)
    test_loader, _test_data = create_dataloader(test_path, batch_size, False)
    
    target_class_num = len(os.listdir(test_path))
    print('target_class_num: ', target_class_num)
    
    print(_train_data.class_to_idx)
    print(_test_data.class_to_idx)
    
    if upstream_weight_type == 'naive':
        net = models.resnet50(pretrained=False)
        net.fc = nn.Sequential(
        nn.Linear(
            net.fc.in_features,
            target_class_num
        ))

    elif upstream_weight_type == 'imagenet':
        net = models.resnet50(pretrained=True)
        net.fc = nn.Sequential(
        nn.Linear(
            net.fc.in_features,
            target_class_num
        ))
        
    elif upstream_weight_type == 'stanford-car':
        
        upstream_weight_path = 'resnet50_acc91_stanfordcar_pretrain_true.pth'
        
        if os.path.exists(os.path.join('./upstream_artifacts/', upstream_weight_path)):
            pass
        
        else:
            print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')
            
            if not os.path.exists('./upstream_artifacts'):
                os.makedirs('./upstream_artifacts')
      
            os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/upstream/' + upstream_weight_path + ' ./upstream_artifacts/.')
            time.sleep(5) # wait until the download is completed
        
        net = models.resnet50(pretrained=False)
        in_features = net.fc.in_features
        
        print(net)
        
        net.fc = nn.Linear(
            in_features,
            196 # due to the number of classes at stanford-car
        )
                
        net.load_state_dict(torch.load(os.path.join('./upstream_artifacts/', upstream_weight_path)))
        net.fc = nn.Linear(
                in_features,
                target_class_num
        )
        
        print('Successfully Loaded the Weight!')
                        
    elif upstream_weight_type == 'byol':
        
        if '2_class_classification' in dataset:
            upstream_weight_path = '2class_resnet50_acc_63_byol_pretrain_false.pth'
            
        elif '10_class_classification' in dataset:
            upstream_weight_path = '10class_resnet50_acc27_byol_pretrain_false.pth'
            
        if os.path.exists(os.path.join('./upstream_artifacts/', upstream_weight_path)):
            pass
        
        else:
            print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')
            
            if not os.path.exists('./upstream_artifacts'):
                os.makedirs('./upstream_artifacts')
                                       
            os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/upstream/' + upstream_weight_path + ' ./upstream_artifacts/.')
            time.sleep(5) # wait until the download is completed
        
        net = models.resnet50(pretrained=False)
        
        in_features = net.fc.in_features
        
        net.load_state_dict(torch.load(os.path.join('./upstream_artifacts/', upstream_weight_path)))
        print('Successfully Loaded the Weight!')
        
        net.fc = nn.Linear(
                in_features,
                target_class_num
            )
            
    elif upstream_weight_type == 'rotation':
        if '2_class_classification' in dataset:    
            upstream_weight_path = '2class_resnet50_acc90_rotnet_pretrain_true.pth'
                
        elif '10_class_classification' in dataset:
            upstream_weight_path = '10class_resnet50_acc92_rotnet_pretrain_true.pth'
        
        if os.path.exists(os.path.join('./upstream_artifacts/', upstream_weight_path)):
            pass
        
        else:
            print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')
            
            if not os.path.exists('./upstream_artifacts'):
                os.makedirs('./upstream_artifacts')
                                       
            os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/upstream/' + upstream_weight_path + ' ./upstream_artifacts/.')
            time.sleep(5) # wait until the download is completed
        
        net = models.resnet50(pretrained=False)
        in_features = net.fc.in_features
                
        net.fc = nn.Sequential(
            nn.Linear(
                in_features,
                4 # due to the number of classes at stanford-car
            )
        )
                
        net.load_state_dict(torch.load(os.path.join('./upstream_artifacts/', upstream_weight_path)))
        net.fc = nn.Sequential(nn.Linear(
                in_features,
                target_class_num
            )
        )
#         net.load_state_dict(torch.load(os.path.join('./downstream_artifacts/2_class_classification/rotation_acc_0.94405.pth')))

        print('Successfully Loaded the Weight!')   

#     nn.DataParallel(net, output_device=1) # for multi-GPU Acceleration
#     net.cuda()
    net.to(device)
        
    best_model_weight = copy.deepcopy(net.state_dict())
    best_test_acc = 0
    best_init_dict = {}

    model_save_base = os.path.join('downstream_artifacts', dataset)
    create_dir_if_absent(model_save_base)

    for epoch in range(n_epochs):

        net, train_acc, train_prec, train_rec, train_f1 = train(train_loader, net, learning_rate, weight_decay, device)
        net, test_acc, test_prec, test_rec, test_f1 = test(test_loader, net, device)
        
        neptune.log_metric('Train Accuraacy', train_acc)
        neptune.log_metric('Train F1 Score', train_f1)
        neptune.log_metric('Test Accuracy', test_acc)
        neptune.log_metric('Test F1 Score', test_f1)

        if test_acc > best_test_acc:

            best_init_dict['acc'] = test_acc
            best_init_dict['prec'] = test_prec
            best_init_dict['rec'] = test_rec
            best_init_dict['f1'] = test_f1

            best_test_acc = test_acc
            test_acc_str = '%.5f' % test_acc

            print('[Notification] Best Model Updated!')
            best_model_weight = copy.deepcopy(net.state_dict())

            model_save_path = os.path.join(model_save_base, upstream_weight_type + '_acc_' + str(test_acc_str) + '.pth') 
            torch.save(net.state_dict(), model_save_path)

            with open(os.path.join(model_save_base, upstream_weight_type + '_best_dict.pkl'), 'wb') as f:
                pickle.dump(best_init_dict, f)
