import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import time
import os
import copy
import statistics
import pickle
import PIL.Image as Image
import neptune

from utils import *
from rotate_datasets import *


def train_self_supervised_weight(train_dir, test_dir, target_data_name, self_train_num_per_class, self_model, device, self_n_epochs, self_batch_size, self_learning_rate, self_weight_decay_level):
    
    print('=============================================')
    print('> Start Training Self Supervision Weight ...')
    print('=============================================')
    
    train_path = train_dir
    test_path = test_dir
    
    train_loader, _train_data = create_dataloader(train_path, self_batch_size, True)
    test_loader, _test_data = create_dataloader(test_path, self_batch_size, False)
    
    print('Train Information...')
    print(_train_data.class_to_idx)
    
    print('Test Information...')
    print(_test_data.class_to_idx)
    
    print('> Selected Model as ', self_model)
    
#     net = model_dict[self_model]
    if self_model == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    
    net.fc = nn.Sequential(
        nn.Linear(
            net.fc.in_features,
            4 # degree 0, 90, 180, 270
    ))
    net.to(device)
    
    self_best_model_weight = copy.deepcopy(net.state_dict())
    self_best_test_acc = 0
    self_best_eval_dict = {}
    
    for epoch in range(self_n_epochs):
        print('> Epoch: ', epoch)
        
        net, train_acc, train_prec, train_rec, train_f1 = train(train_loader, net, self_learning_rate, self_weight_decay_level, device)
        net, test_acc, test_prec, test_rec, test_f1 = test(test_loader, net, device)
        
        if epoch % 5 == 0:
            print('----------------------------')
            print('> Training Metrics')
            print('Train Acc: ', train_acc)
            print('Train F1: ', train_f1)

            print('> Test Metrics')
            print('Test Acc: ', test_acc)
            print('Test F1: ', test_f1)
            print('----------------------------')

        if test_acc > self_best_test_acc:

            self_best_eval_dict['acc'] = test_acc
            self_best_eval_dict['prec'] = test_prec
            self_best_eval_dict['rec'] = test_rec
            self_best_eval_dict['f1'] = test_f1

            self_best_test_acc = test_acc

            print('[Notification] Best Model Updated!')
            self_best_model_weight = copy.deepcopy(net.state_dict())
        

        print('>> Save Best Self Model and Performance...')
        
        model_save_base = os.path.join('self_models', target_data_name)
        if not os.path.exists(model_save_base):
            os.makedirs(model_save_base)

            
        self_best_test_acc_str = '%.5f' % self_best_test_acc
        model_save_path = os.path.join(model_save_base, 'self_weight_' + self_model + '_train_size_' + str(self_train_num_per_class) + '_acc_' + self_best_test_acc_str + '.pth')

        torch.save(net.state_dict(), model_save_path)
        with open(os.path.join(model_save_base, 'self_metrics_' + self_model + '_train_size_' + str(self_train_num_per_class) + '_acc_' + self_best_test_acc_str + '.pkl'), 'wb') as f:
            pickle.dump(self_best_eval_dict, f)