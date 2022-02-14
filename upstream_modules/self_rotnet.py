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

from utils import *


def path_to_train_test_filenames(base_dir, self_train_size, self_test_size, exp_task_type):
    candidate_train_filenames = list()
    candidate_test_filenames = list()
    
    target_src_path = os.path.join(base_dir, exp_task_type, 'Training')
    print('>>>>>>', target_src_path)
    print(os.listdir(target_src_path))
    
    for class_name in sorted(os.listdir(target_src_path)):
        
        class_path = os.path.join(target_src_path, class_name)
        class_test_filenames = sorted(os.listdir(class_path))[:self_test_size]
        class_train_filenames = sorted(os.listdir(class_path))[self_test_size : self_test_size+self_train_size]
        
        for class_filename in class_test_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            candidate_test_filenames.append(class_filepath)
        
        for class_filename in class_train_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            candidate_train_filenames.append(class_filepath)
            
    print('Train for Self: ', len(candidate_train_filenames))            
    print('Test for Self: ', len(candidate_test_filenames))
    
    return candidate_train_filenames, candidate_test_filenames


def create_rotnet_set(base_dir, self_train_size, self_test_size, exp_task_type):
    
    # extract Candidate Train, Test Filename 
    candidate_train_filenames, candidate_test_filenames = \
    path_to_train_test_filenames(base_dir, self_train_size, self_test_size, exp_task_type)
    
    # start Rotation task 
    print('>> Start Making a Rotated Set...')
    save_base = os.path.join(base_dir, 'self_dataset', 'rotnet_' + exp_task_type + '_train_size_' + str(self_train_size))
    
    print('> Start Rotating Training Set...')
    train_save_dir = os.path.join(save_base, 'Training')
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    rotate_and_save(candidate_train_filenames, train_save_dir)
        
    print('> Start Rotating Test Set...')
    test_save_dir = os.path.join(save_base, 'Test')
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    rotate_and_save(candidate_test_filenames, test_save_dir)
    

def rotate_and_save(src_filenames, save_dir):
    rotation_list = [0, 90, 180, 270]
        
    for rotation_degree in rotation_list:
        print('> working on degree '+ str(rotation_degree))
        
        rot_save_dir = os.path.join(save_dir, 'degree_' + str(rotation_degree))
        if not os.path.exists(rot_save_dir):
            os.makedirs(rot_save_dir)

        for src_filename in src_filenames:
            #print('>> check filenames :', src_filename)

            img = cv2.imread(src_filename)
            rotated_img = image_rotator(img, rotation_degree)
    
            cv2.imwrite(os.path.join(rot_save_dir, src_filename.split('/')[-1]), rotated_img)
    
    
def image_rotator(img, rotate_type):
    
    if rotate_type == 0:
        rotated_img = img
    
    elif rotate_type == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
    elif rotate_type == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        
    elif rotate_type == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    else:
        raise ValueError

    return rotated_img
    
# def train_rotnet_weight(base_dir, self_train_num_per_class, self_model, device, self_n_epochs, self_batch_size, self_learning_rate, self_weight_decay_level):
    
def train_rotnet_weight(base_dir, self_train_size, self_test_size, exp_task_type, \
                        self_batch_size, self_n_epochs, self_learning_rate, self_weight_decay_level, \
                        device):
    
    train_path =  os.path.join(base_dir, 'self_dataset', 'rotnet_'+ exp_task_type + '_train_size_' + str(self_train_size), 'Training')
    test_path =  os.path.join(base_dir, 'self_dataset', 'rotnet_'+ exp_task_type + '_train_size_' + str(self_train_size), 'Test')
    
    train_loader, _train_data = create_dataloader(train_path, self_batch_size, True)
    test_loader, _test_data = create_dataloader(test_path, self_batch_size, False)
    
    print('Train Information...')
    print(_train_data.class_to_idx)
    
    print('Test Information...')
    print(_test_data.class_to_idx)
        
    net = torchvision.models.resnet50(pretrained=False)
    
    net.fc = nn.Sequential(
        nn.Linear(
            net.fc.in_features,
            4 # degree 0, 90, 180, 270
    ))
    net.to(device)
    
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
        
        print('>> Save Best Self Model and Performance...')
        
        model_save_base = os.path.join('upstream_artifacts', 'rotnet')
        if not os.path.exists(model_save_base):
            os.makedirs(model_save_base)

        self_best_test_acc_str = '%.5f' % self_best_test_acc
        model_save_path = os.path.join(model_save_base, 'rotnet_' + exp_task_type + '_train_size_' + str(self_train_size) \
                                      + '_acc_' + self_best_test_acc_str + '.pth')

        torch.save(net.state_dict(), model_save_path)
        with open(os.path.join(model_save_base, 'rotnet_' + exp_task_type + '_train_size_' + str(self_train_size) + '.pkl'), 'wb') as f:
            pickle.dump(self_best_eval_dict, f)
