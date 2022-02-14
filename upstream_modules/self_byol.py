import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn import preprocessing
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

from byol_pytorch import BYOL
from torch.utils.data.dataloader import DataLoader
from utils import *

device = "cuda:0"

'''
Dataset dir (../dataset/)
ㄴ unlabeled Training 
ㄴ labeled Training 
ㄴ labeled Test
'''

def path_to_train_test_filenames(base_dir, self_train_size, self_test_size, exp_task_type):
    candidate_train_filenames_unlabeled = list() # unlabeled
    candidate_train_filenames = list() # labeled 
    candidate_test_filenames = list() # labeled
    
    target_src_path = os.path.join(base_dir, exp_task_type, 'Training')
    print(os.listdir(target_src_path))
    
    for class_name in sorted(os.listdir(target_src_path)):
        # test data for each class
        test_filenames_temp = list()
        train_filenames_temp = list()
        
        class_path = os.path.join(target_src_path, class_name)
        class_test_filenames = sorted(os.listdir(class_path))[:self_test_size]
        class_train_filenames = sorted(os.listdir(class_path))[self_test_size : self_test_size+self_train_size]
        
        for class_filename in class_test_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            test_filenames_temp.append(class_filepath)
        
        for class_filename in class_train_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            train_filenames_temp.append(class_filepath)
            candidate_train_filenames_unlabeled.append(class_filepath)
        
        candidate_test_filenames.append(test_filenames_temp)
        candidate_train_filenames.append(train_filenames_temp)
            
    print('Train unlabeled : ', len(candidate_train_filenames_unlabeled))
    print('Train class for Selt :', len(candidate_train_filenames))
    print('Test class for Self: ', len(candidate_test_filenames))
    
    return candidate_train_filenames_unlabeled, candidate_train_filenames, candidate_test_filenames



def move_files(src_filenames, save_dir):
    for src_filename in src_filenames:
        dst_filename = os.path.join(save_dir, src_filename.split('/')[-1])
        shutil.copy(src_filename, dst_filename)


        
def create_byol_set(base_dir, self_train_size, self_test_size, exp_task_type):
    # extract candidate Train, Test Filename 
    candidate_train_filenames_unlabeled, candidate_train_filenames, candidate_test_filenames = \
    path_to_train_test_filenames(base_dir, self_train_size, self_test_size, exp_task_type)
    
    # BYOL dataset 
    print('>> Start Making a BYOL Set...')
    save_base = os.path.join(base_dir, 'self_dataset', 'byol_train_size_' + str(self_train_size))
    
    ## Training : unlabeled datasets
    print('> Start BYOL unlabeled Training Set...')
    train_save_dir = os.path.join(save_base, 'Training_unlabeled', 'images')
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    move_files(candidate_train_filenames_unlabeled, train_save_dir)
    
    ## Training : labeled datasets 
    print('> Start BYOL labeled Training Set...')
    train_save_dir_out = os.path.join(save_base, 'Training_labeled')
    for i in range(len(candidate_train_filenames)) : 
        train_save_dir = os.path.join(save_base, 'Training_labeled', str(i))
        if not os.path.exists(train_save_dir) :
            os.makedirs(train_save_dir)
        move_files(candidate_train_filenames[i], train_save_dir)
    
    ## Test : labeled datasets
    print('> Start BYOL Test Set...')
    test_save_dir_out = os.path.join(save_base, 'Test_labeled')
    for i in range(len(candidate_test_filenames)) : 
        test_save_dir = os.path.join(save_base, 'Test_labeled', str(i))
        if not os.path.exists(test_save_dir) :
            os.makedirs(test_save_dir)
        move_files(candidate_test_filenames[i], test_save_dir)
    
    print('check unlabeled train dir : ', train_save_dir)
    print('check labeled train dir : ', train_save_dir_out)
    print('check labeled test dir : ', test_save_dir_out)

    
def train_byol_weight(unlabeled_train_root, train_root, test_root, num_epochs): 
    
    # 1. define dataset, loader 
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    ## byol train dataset
    unlabeled_train_dataset = datasets.ImageFolder(unlabeled_train_root, transform)
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset,batch_size = 32, shuffle = True)
    
    ## byol val dataset
    train_dataset = datasets.ImageFolder(train_root, transform)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    
    test_dataset = datasets.ImageFolder(test_root, transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
    
    
    ## byol model
    resnet = models.resnet50(pretrained=False) # encoder

    learner = BYOL(
        resnet,
        image_size = 256,
        hidden_layer = 'avgpool'
    )
    output_feature_dim = learner.online_encoder.projector.net[0].in_features
    logreg = LogisticRegression(1000, 10)
    logreg = logreg.to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    
    resnet.eval()
    x_train, y_train = get_features_from_encoder(resnet, train_dataloader)
    x_test, y_test = get_features_from_encoder(resnet, test_dataloader)
    if len(x_train.shape) > 2:
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test)
    
    optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 2. Training BYOL model
    for i in range(num_epochs) :
        print('>> check Training epoch : ', i)
        # 2_1. start training
        for idx, (img, label) in enumerate(unlabeled_train_dataloader) :
            if idx % 10 == 0 : 
                print('>> check train idx :', idx)
            loss = learner(img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        
        # 2_2. check the results on training phase
        print('>> Test phase')
        total = 0 
        correct = 0 
        for x, y in train_loader : 
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = logreg(x)
            predictions = torch.argmax(logits, dim =1)
            
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total += y.size(0)
            correct += (predictions == y).sum().item()
        train_acc = 100 * correct/total
        print(f'=> Training accuracy:{train_acc}')
            
        # 2_2. check the results on test phase
        ## feature vector
        print('> start inference')
        total = 0
        correct = 0
        for x, y in test_loader :
            x = x.to(device)
            y = y.to(device)
            
            logits = logreg(x)
            predictions = torch.argmax(logits, dim = 1)
            
            total += y.size(0)
            correct += (predictions == y).sum().item()
        acc = 100 * correct/total
        
        print(f"Testing accuracy: {acc}")
            
        # 3. save the model
        save_dir = 'self_byol_weight/'
        save_filename = f'self_byol_weight/byol_acc_{acc}.pth'
        print('>> check saved model path : ', save_filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(resnet.state_dict(), save_filename)



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

    
def get_features_from_encoder(encoder, loader):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    return train_loader, test_loader
